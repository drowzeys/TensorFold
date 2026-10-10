#!/bin/bash
# GLM-5.3 Zig port, Phase 1 gate on ONE GB10 node: build the Zig engine in the TP4 image, capture the Triton AOT set
# and the fixtures from the Python engine (first 4 layers, rank 0 of world 4 with the other ranks absent), run tf-glm53-layers against them, and print one
# summary line:  PHASE1 PASS|FAIL ...
#
# Run on a node (never on .4), in a scheduled window, from a copy of ~/zig-port:
#   rsync -a --delete ~/zig-port/ <node>:zig-port/          # from .4 (tf/ is the port's git working copy)
#   ssh <node> 'bash ~/zig-port/run_phase1_node.sh'
#
# Environment (defaults in brackets):
#   PORT     the zig-port folder on this node                        [$HOME/zig-port]
#   PYSRC    the champion's Python source (holds src/tensorfold)     [required]
#   MODEL    the GLM-5.3 EXL3 checkpoint                             [/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit]
#   IMAGE    the TP4 image (Triton 3.7.1, torch, CUDA 13)            [ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05]
#   ZIG_DIR  folder holding the zig 0.17.0 binary (mounted /opt/z)   [$HOME/opt/zig]
#   WORK     this run's output folder                                [$PORT/runs/<utc stamp>]
#   LAYERS   [4]   WORLD [4]  RANK [0] (the cut loaded; the other ranks are absent)   LONG  the long chain's prefix [20000]   GPU_LOCK  a flock file wrapped around GPU steps [none]
#   ALLOW_GPU_BUSY=1   start even while other compute apps hold the GPU (default: refuse)
#   SKIP_BUILD=1 / SKIP_ORACLE=1   reuse WORK's zig-out / aot + fixtures from an earlier run (set WORK to it)
set -u
PORT="${PORT:-$HOME/zig-port}"
PYSRC="${PYSRC:?PYSRC: the Python engine source tree (holds src/tensorfold)}"
MODEL="${MODEL:-/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit}"
IMAGE="${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05}"
ZIG_DIR="${ZIG_DIR:-$HOME/opt/zig}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
WORK="${WORK:-$PORT/runs/$STAMP}"
LAYERS="${LAYERS:-4}"
WORLD="${WORLD:-4}"
RANK="${RANK:-0}"
LONG="${LONG:-20000}"
TF="$PORT/tf"
CACHE="$PORT/cache"
LABEL="tensorfold.glm53-phase1=$STAMP"
mkdir -p "$WORK" "$CACHE/zig-local" "$CACHE/zig-global" "$CACHE/torch_ext" "$CACHE/cuda_cache"
LOG="$WORK/run.log"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
summary() { echo "PHASE1 $1 node=$(hostname) $2 work=$WORK" | tee -a "$LOG"; exit "${3:-1}"; }
cleanup() { for c in $(docker ps -q --filter "label=$LABEL" 2>/dev/null); do docker stop --time 15 "$c" >/dev/null 2>&1; done; }
trap 'cleanup; summary FAIL "interrupted" 143' TERM INT

# ---- preflight: this must never become the thing that OOMs a node --------------------------------------------
avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
say "node $(hostname): MemAvailable ${avail_gb} GB; work $WORK"
[ -f "$HOME/NODE_CLAIM.txt" ] && { say "NODE_CLAIM.txt:"; sed 's/^/  /' "$HOME/NODE_CLAIM.txt" | tee -a "$LOG"; }
[ "$avail_gb" -ge 30 ] || summary FAIL "refused: MemAvailable ${avail_gb} GB < 30 GB" 3
if docker ps --format '{{.Names}} {{.Image}}' | grep -q 'tf-glm53'; then
  docker ps --format '{{.Names}} {{.Image}}' | grep 'tf-glm53' | tee -a "$LOG"
  summary FAIL "refused: a tf-glm53 container is running" 3
fi
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
if [ -n "$apps" ] && [ "${ALLOW_GPU_BUSY:-0}" != 1 ]; then
  echo "$apps" | tee -a "$LOG"
  summary FAIL "refused: compute apps hold the GPU (ALLOW_GPU_BUSY=1 to start anyway)" 3
fi
for p in "$TF/build.zig" "$PYSRC/src/tensorfold/families/glm_moe_dsa/cuda/fused.py" "$MODEL/config.json" "$ZIG_DIR/zig"; do
  [ -e "$p" ] || summary FAIL "missing $p" 3
done
gpu() { if [ -n "${GPU_LOCK:-}" ]; then flock "$GPU_LOCK" "$@"; else "$@"; fi; }
COMMON=(--rm --label "$LABEL" --network none --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1)

# ---- 0. the image's versions and the kernel copies against the champion source (host, no GPU) ----------------
say "image versions"
docker run "${COMMON[@]}" --memory 8g --entrypoint python3 "$IMAGE" -c \
  "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)" 2>&1 | tee "$WORK/versions.txt" | tee -a "$LOG"
grep -q 'triton 3\.7' "$WORK/versions.txt" || say "WARNING: the AOT launcher copies Triton 3.7's (zig/src/cuda/triton.zig); this image has $(cat "$WORK/versions.txt")"
say "EXL3 kernel copies vs $PYSRC (tools/glm53/copy_exl3_kernels.py --check)"
python3 -B "$TF/tools/glm53/copy_exl3_kernels.py" --check --src "$PYSRC/src/tensorfold/cuda/exl3" >> "$LOG" 2>&1 \
  || summary FAIL "the device-only EXL3 copies drifted from the champion source (re-run copy_exl3_kernels.py on .4 and commit)" 1
python3 -B "$TF/zig/tests/cuda/nemotron/test_aot_manifest.py" >> "$LOG" 2>&1 || say "WARNING: test_aot_manifest.py failed (see run.log)"

# ---- 1. build (CPU only: no --gpus), memory-capped -------------------------------------------------------------
BIN="$WORK/zig-out/bin/tf-glm53-layers"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  say "build: zig build test-glm53, then fatbins + tf-glm53-layers (docker --memory 24g)"
  timeout 3600 docker run "${COMMON[@]}" --memory 24g --memory-swap 24g \
    -v "$TF:/work/tf" -v "$WORK:/out" -v "$CACHE:/cache" -v "$ZIG_DIR:/opt/z:ro" -w /work/tf \
    -e ZIG_LOCAL_CACHE_DIR=/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=/cache/zig-global -e PATH=/opt/z:/usr/local/cuda/bin:/usr/bin:/bin \
    --entrypoint bash "$IMAGE" -c '
      set -u; rc=0
      zig version
      zig build test-glm53 --summary all || rc=10
      zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix /out/zig-out -j8 fatbins tf-glm53-layers || rc=$((rc + 20))
      for f in glm53_exl3_linear glm53_exl3_experts; do
        [ -f /out/zig-out/fatbin/$f.fatbin ] && cuobjdump -symbols /out/zig-out/fatbin/$f.fatbin > /out/symbols-$f.txt
      done
      exit $rc' > "$WORK/build.log" 2>&1
  brc=$?
  tail -40 "$WORK/build.log" >> "$LOG"
  [ "$brc" = 0 ] || { tail -40 "$WORK/build.log"; summary FAIL "build rc=$brc (10: host tests, 20: engine build; build.log)" 1; }
  missing=0
  while read -r sym; do
    case "$sym" in "#"*|"") continue ;; esac
    grep -q -- "$sym" "$WORK"/symbols-glm53_exl3_*.txt || { say "kernel symbol missing from the fatbins: $sym"; missing=$((missing + 1)); }
  done < "$TF/tools/glm53/expected_symbols.txt"
  [ "$missing" = 0 ] || summary FAIL "build: $missing kernel symbols missing (symbols-*.txt)" 1
fi
[ -x "$BIN" ] || summary FAIL "no $BIN" 1

# ---- 2. the Python oracle: Triton AOT capture + fixtures (GPU) ------------------------------------------------
if [ "${SKIP_ORACLE:-0}" != 1 ]; then
  rm -rf "$WORK/triton" "$WORK/aot" "$WORK/fixtures"
  mkdir -p "$WORK/triton"
  say "oracle: $LAYERS layers, rank $RANK of world $WORLD (others absent), long prefix $LONG (record + fixtures; docker --memory 64g)"
  gpu timeout 3600 docker run "${COMMON[@]}" --gpus all --memory 64g --memory-swap 64g \
    -v "$MODEL:/model:ro" -v "$PYSRC/src:/opt/tensorfold/src:ro" -v "$TF:/tf:ro" -v "$WORK:/out" -v "$CACHE:/cache" \
    -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/out/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
    -e CUDA_CACHE_PATH=/cache/cuda_cache -e TORCH_CUDA_ARCH_LIST=12.1 -w /tmp \
    --entrypoint python3 "$IMAGE" -B /tf/tools/glm53/oracle.py --model /model --out /out --record --fixtures \
    --layers "$LAYERS" --world "$WORLD" --rank "$RANK" --long "$LONG" --tools /tf > "$WORK/oracle.log" 2>&1
  orc=$?
  grep '\[oracle\]' "$WORK/oracle.log" | tee -a "$LOG"
  [ "$orc" = 0 ] || { tail -30 "$WORK/oracle.log"; summary FAIL "oracle rc=$orc (oracle.log)" 1; }
fi
[ -f "$WORK/aot/aot.json" ] && [ -f "$WORK/fixtures/meta.json" ] || summary FAIL "no aot.json / fixtures in $WORK" 1
nvar=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['kernels']))" "$WORK/aot/aot.json")
radix=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('radix_equal'))" "$WORK/fixtures/meta.json")

# ---- 2b. SASS of our fatbins against the Python extensions' builds (diagnostic, CPU) -------------------------
sass="skipped"
if [ -d "$CACHE/torch_ext/tensorfold_exl3_linear_v5" ]; then
  docker run "${COMMON[@]}" --memory 8g -v "$WORK:/out" -v "$CACHE:/cache:ro" -v "$TF:/tf:ro" --entrypoint bash "$IMAGE" -c '
    set -u; bad=0
    cuobjdump -sass /cache/torch_ext/tensorfold_exl3_linear_v5/linear.cuda.o > /out/sass-python-linear.txt
    { cuobjdump -sass /cache/torch_ext/tensorfold_exl3_experts_v4/experts.cuda.o
      cuobjdump -sass /cache/torch_ext/tensorfold_exl3_experts_v4/experts_cb2.cuda.o; } > /out/sass-python-experts.txt
    cuobjdump -sass /out/zig-out/fatbin/glm53_exl3_linear.fatbin > /out/sass-zig-linear.txt
    cuobjdump -sass /out/zig-out/fatbin/glm53_exl3_experts.fatbin > /out/sass-zig-experts.txt
    for x in linear experts; do
      echo "--- $x"; python3 -B /tf/zig/tests/cuda/nemotron/sass_compare.py /out/sass-python-$x.txt /out/sass-zig-$x.txt || bad=1
    done
    exit $bad' > "$WORK/sass.log" 2>&1 && sass="equal" || sass="DIFFERS(sass.log)"
  grep -c "SASS-EQUAL" "$WORK/sass.log" >/dev/null 2>&1 && say "sass: $(grep -c SASS-EQUAL "$WORK/sass.log") equal, $(grep -c SASS-DIFFER "$WORK/sass.log") differ"
fi

# ---- 3. the Zig layers against the fixtures (GPU) --------------------------------------------------------------
say "compare: tf-glm53-layers (docker --memory 48g)"
gpu timeout 1800 docker run "${COMMON[@]}" --gpus all --memory 48g --memory-swap 48g \
  -v "$MODEL:/model:ro" -v "$WORK:/out" --entrypoint /out/zig-out/bin/tf-glm53-layers "$IMAGE" \
  /model /out/aot /out/fixtures --layers "$LAYERS" > "$WORK/compare.log" 2>&1
crc=$?
grep -E '^(PASS|FAIL|RESULT|WARN|diag|  differs)' "$WORK/compare.log" | head -80 | tee -a "$LOG"
result=$(grep '^RESULT glm53-layers' "$WORK/compare.log" | tail -1)
[ -n "$result" ] || result="no RESULT line (rc $crc; compare.log tail: $(tail -3 "$WORK/compare.log" | tr '\n' ' ' | cut -c1-200))"
detail="aot_variants=$nvar radix_equal=$radix sass=$sass | $result"
if [ "$crc" = 0 ] && echo "$result" | grep -q 'RESULT glm53-layers PASS'; then summary PASS "$detail" 0; fi
summary FAIL "$detail" 1
