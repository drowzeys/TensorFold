#!/bin/bash
# GLM-5.3 Zig port, Phase 3a gate: the Zig engine against the Python engine in its SERVED configuration, on the four
# GB10 nodes at TP4. Three parts, one run:
#   1. the RoCE one-shot all-reduce / small all-gather in Zig (b12x RoCEnante's proxy compiled by Zig + its kernels in
#      plain CUDA): `tf-glm53-generate --mode roce-bench` - its bits against NCCL's rank-order sum, then microseconds a
#      reduction for the decode windows' shapes (target <= 65 us; the Python engine's b12x ~57 us);
#   2. the served prompt path (8192 / 4096-row chunks, sequence-parallel halves, the prompt GEMM, torch.bmm absorb /
#      expand through cuBLAS - checked on a probe first -, prompt experts, radix top-k, NCCL's bf16 ring exchanges) and
#      RoCE decode windows: `tf-glm53-generate --mode 3a` against tools/glm53/reference3a.py (Glm53Engine with the
#      2026-10-05 image's defaults: RoCE on, PROMPT_SP on, copy drafts off; TF_EXL3_PROMPT_DET=slots16 in both engines,
#      the default red.add being irreproducible run to run) - every token identical;
#   3. the report: Zig vs served Python - 32K prose decode tok/s (MTP k=2, sampled, thinking on), prefill tok/s at 32K
#      and 128K, load time. PASS: token identity, drafted == serial, and Zig >= RATIO (0.97) x Python on the 32K prose
#      decode and on both prefills.
# Prints one line at the end:  PHASE3A PASS|FAIL ...
#
# Run from .4 (the orchestrator: it only syncs, starts containers over ssh and copies results; every build and GPU
# job runs on the nodes). Free the cluster first - this script stops nothing it did not start:
#   bash ~/zig-port/run_phase3a_cluster.sh
#
# Steps: preflight (as 2b; also refuses a leftover zp3a-* container) -> rsync ~/zig-port + the champion source ->
# prompts (reference3a.py --make-prompts: 2b's prompts + the 128K prompt; rank 0 node, CPU) -> build on every node
# (host tests, fatbins, tf-glm53-generate; the new kernels' symbols checked with cuobjdump) -> RoCE bench (4 containers,
# no model) -> the served Python reference at TP4 (records the Triton AOT set, the tile table, the bmm probe, the BLAS
# setup) -> [SPEED_DEFAULT=1: a speed-only Python pass at the image's default prompt-experts mode] -> tf-glm53-generate
# --mode 3a at TP4 -> compare_phase3a.py.
#
# Environment (defaults in brackets):
#   NODES MODEL IMAGE PORT_DIR PYSRC ZIG_DIR ZIG_FROM MASTER IF RAILS MEM_CAP DROP_CACHES ALLOW_GPU_BUSY: as phase 2b
#   TOKENS [256] picks a run of the short prompts   TOKENS_LONG [512] the 32K prose prompt's   TOKENS_128K [32]
#   RUNS [g:2,s20:2,s20:0]  RUNS_LONG [s20:2,s20:0]  RUNS_128K [g:2,g:0]   K [2]   LONG [4500]
#   CONTEXT32K [$HOME/glm53-speed-20260920/context-32k.txt] (the 128K prompt is that text four times over)
#   CONTEXT [auto: the longest prompt + its picks, rounded up to 1024; must stay under the DCP cut, 200,000]
#   RATIO [0.97]   BENCH_ROWS [1,2,3,4,8,16,32]   BENCH_ITERS [200]   SPEED_DEFAULT [0]
#   Zig decode knobs (the image's served defaults): ZSIDE [af] (TF_GLM53_SIDE)  ZL2PF [1] (TF_GLM53_L2PF: 1|0|lines|touch)
#   ZL2PF_MB [8]  ZMULTI_SELECT [1] (decode-sized top-k over many blocks; 0: the one-program radix select)
#   ZDEVICE_CANDS [1] (sampled windows: each rank's candidates picked on the device)
#   ZMTP_DENSE [1] (the MTP layer's prompt rows through cuBLAS: its 8-bit experts as dense bf16 built at load, ~4.8 GB
#     a rank, and eh_proj; drafts only. 0: the decode expert kernel in blocks and the router kernel)
#   TPR_TOL [0.02]: the gate's tokens/round tolerance |zig - py| / py on the 32K prose drafted runs
#   PROFILE [0]: 1 = the prefill of PROFILE_PROMPT [prose32k]'s first run timed by phase in both engines (Zig:
#     --prompt-prof, CUDA events a chunk; Python: Runner.prefill's torch.profiler table rolled up by
#     reference3a.py --profile-prefill), printed after each engine's runs as PROMPT-PROF / PREFILL-PROF lines and kept
#     in prefill-prof.txt. That run's prefill is slowed in both engines; the gate (unchanged) takes each prompt's best
#     run, so give the profiled prompt two runs (prose32k has two by default)
#   REF_PORT [29571] (TCPStore; RoCE's gloo setup at +11)  ZIG_PORT [29581]  BENCH_PORT [29591]
#   SKIP_SYNC=1 SKIP_BUILD=1 SKIP_BENCH=1 SKIP_REF=1 (with REUSE=<stamp>: its prompts, zig-out and ref/)
set -u
NODES=(${NODES:-10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5})
[ ${#NODES[@]} -eq 4 ] || { echo "NODES must list four fabric addresses, rank 0 first"; exit 2; }
MASTER=${MASTER:-${NODES[0]}}
MODEL=${MODEL:-/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit}
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05}
PORT_DIR=${PORT_DIR:-$HOME/zig-port}
PYSRC=${PYSRC:?PYSRC: the Python engine source tree on .4}
ZIG_DIR=${ZIG_DIR:-$HOME/opt/zig}
ZIG_FROM=${ZIG_FROM:-10.100.10.5}
TOKENS=${TOKENS:-256}
TOKENS_LONG=${TOKENS_LONG:-512}
TOKENS_128K=${TOKENS_128K:-32}
RUNS=${RUNS:-g:2,s20:2,s20:0}
RUNS_LONG=${RUNS_LONG:-s20:2,s20:0}
RUNS_128K=${RUNS_128K:-g:2,g:0}
K=${K:-2}
LONG=${LONG:-4500}
CONTEXT32K=${CONTEXT32K-$HOME/glm53-speed-20260920/context-32k.txt}
CONTEXT=${CONTEXT:-auto}
RATIO=${RATIO:-0.97}
BENCH_ROWS=${BENCH_ROWS:-1,2,3,4,8,16,32}
BENCH_ITERS=${BENCH_ITERS:-200}
SPEED_DEFAULT=${SPEED_DEFAULT:-0}
ZSIDE=${ZSIDE:-af}
ZL2PF=${ZL2PF:-1}
ZL2PF_MB=${ZL2PF_MB:-8}
ZMULTI_SELECT=${ZMULTI_SELECT:-1}
ZDEVICE_CANDS=${ZDEVICE_CANDS:-1}
ZMTP_DENSE=${ZMTP_DENSE:-1}
TPR_TOL=${TPR_TOL:-0.02}
PROFILE=${PROFILE:-0}
PROFILE_PROMPT=${PROFILE_PROMPT:-prose32k}
REF_PORT=${REF_PORT:-29571}
ZIG_PORT=${ZIG_PORT:-29581}
BENCH_PORT=${BENCH_PORT:-29591}
IF=${IF:-enp1s0f1np1}
RAILS=${RAILS:-2}
MEM_CAP=${MEM_CAP:-110g}
DROP_CACHES=${DROP_CACHES:-1}
STAMP=${REUSE:-$(date -u +%Y%m%d-%H%M%S)}
RHOME=${RHOME:-$HOME}
RPORT=$RHOME/zig-port
WORK=$RPORT/runs/p3a-$STAMP                   # on every node
LOCAL=$PORT_DIR/runs/p3a-$STAMP               # on .4: logs and the result files
LABEL="tensorfold.glm53-phase3a=$STAMP"
mkdir -p "$LOCAL"
LOG=$LOCAL/run.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
summary() { echo "PHASE3A $1 $2 work=$LOCAL" | tee -a "$LOG"; exit "${3:-1}"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; }
cleanup() {
  for n in "${NODES[@]}"; do
    rsh "$n" "docker ps -aq --filter label=$LABEL | xargs -r docker rm -f" >/dev/null 2>&1 &
  done; wait
}
trap 'say "interrupted: removing this run'"'"'s containers"; cleanup; summary FAIL "interrupted" 143' TERM INT

# ---- rails: one-shot.sh's detection (RoCE devices matched by subnet; GID indices looked up now) ------------------
rail_scan='for d in /sys/class/net/*; do n=${d##*/}; [ -d $d/device/infiniband ] || continue
  [ "$(cat $d/operstate 2>/dev/null)" = up ] || continue
  a=$(ip -4 -o addr show dev $n | awk "{print \$4}" | head -1); [ -n "$a" ] || continue
  net=$(ip -4 -o route show dev $n scope link proto kernel | awk "{print \$1}" | head -1)
  hca=$(ls $d/device/infiniband | head -1); h=$(printf "%02x%02x:%02x%02x" $(echo ${a%/*} | tr . " "))
  p=/sys/class/infiniband/$hca/ports/1; g=
  for i in $(ls $p/gids | sort -n); do [ "$(cat $p/gid_attrs/types/$i 2>/dev/null)" = "RoCE v2" ] &&
    grep -q "ffff:$h\$" $p/gids/$i && { g=$i; break; }; done
  [ -n "$g" ] || continue
  o=$(ibv_devices 2>/dev/null | awk "NR>2{print \$1}" | grep -nx "$hca" | cut -d: -f1)
  echo "${net:-?} $hca $g ${o:-0} $n"; done'
declare -a HCAS=() RGIDS=() NGID=() ZHCAS=() ZGID=()
XNIC=0
detect_rails() {
  local ref="" r lines own nets
  for r in 0 1 2 3; do
    lines=$(rsh "${NODES[$r]}" "$rail_scan" | sort) || { say "ssh to ${NODES[$r]} failed"; return 1; }
    own=$(awk -v ip="${NODES[$r]}" '{split($1, c, "[./]"); split(ip, d, "."); if (c[1]==d[1] && c[2]==d[2] && c[3]==d[3]) print}' <<<"$lines")
    [ -n "$own" ] || { say "${NODES[$r]}: no RoCE v2 device on its own subnet"; return 1; }
    lines=$( { echo "$own"; grep -vxF "$own" <<<"$lines"; } | grep . | head -n "$RAILS")
    nets=$(awk '{print $1}' <<<"$lines" | paste -sd' ')
    [ -z "$ref" ] && ref=$nets
    [ "$nets" = "$ref" ] || { say "rank $r rails on [$nets], rank 0 on [$ref]: RAILS=1 or fix addressing"; return 1; }
    HCAS[$r]=$(awk '{print $2}' <<<"$lines" | paste -sd,)
    RGIDS[$r]=$(awk '{print $3}' <<<"$lines" | paste -sd,)
    NGID[$r]=$(awk '{print $3}' <<<"$lines" | sort -u | awk 'END{if (NR==1) print}')
    [ "$(awk '{print $4}' <<<"$lines" | paste -sd' ')" = "$(awk '{print $4}' <<<"$lines" | sort -n | paste -sd' ')" ] || XNIC=1
    # roce.rails(): b12x takes one GID index for all its devices - devices whose GIDs differ fall back to the first
    if [ -n "${NGID[$r]}" ]; then ZHCAS[$r]=${HCAS[$r]}; ZGID[$r]=${NGID[$r]}
    else ZHCAS[$r]=${HCAS[$r]%%,*}; ZGID[$r]=${RGIDS[$r]%%,*}; fi
    say "rank $r (${NODES[$r]}): rails ${HCAS[$r]} on [$nets], GIDs ${RGIDS[$r]}; RoCE one-shot on ${ZHCAS[$r]} (GID ${ZGID[$r]})"
  done
}
# one-shot.sh's environment: NCCL's rails and the RoCE runtime's devices (both engines get the same, so NCCL picks the
# same algorithms - its ring sums of prompt rows are part of the bits)
nccl_env() {
  local r=$1 e="-e NCCL_SOCKET_IFNAME=$IF -e NCCL_IB_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_GIDS=${RGIDS[$1]}"
  if [ -n "${NGID[$r]}" ]; then e="$e -e NCCL_IB_GID_INDEX=${NGID[$r]}"
  else e="$e -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET"; fi
  if (( XNIC )); then e="$e -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1"; else e="$e -e NCCL_CROSS_NIC=0"; fi
  echo "$e"
}

# ---- 0. preflight ------------------------------------------------------------------------------------------------
say "phase 3a: nodes ${NODES[*]} (rank 0 first), model $MODEL, image $IMAGE, context $CONTEXT, runs [$RUNS] x $TOKENS, 32K [$RUNS_LONG] x $TOKENS_LONG, 128K [$RUNS_128K] x $TOKENS_128K, k $K, ratio $RATIO, work $WORK"
bad=0
for r in 0 1 2 3; do n=${NODES[$r]}
  out=$(rsh "$n" "echo avail=\$(awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo)
    echo tfglm=\$(docker ps --format '{{.Names}} {{.Image}}' | grep -c tf-glm53)
    echo zp=\$(docker ps -a --format '{{.Names}}' | grep -cE '^zp(2[ab]|3a)-')
    echo apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ -f '$MODEL/config.json' ] && echo model=ok || echo model=missing
    docker image inspect '$IMAGE' >/dev/null 2>&1 && echo image=ok || echo image=missing
    [ -f ~/NODE_CLAIM.txt ] && sed 's/^/claim: /' ~/NODE_CLAIM.txt; true" 2>&1) || { say "rank $r ($n): ssh failed"; bad=1; continue; }
  say "rank $r ($n): $(grep -v '^claim:' <<<"$out" | tr '\n' ' ')"
  grep '^claim:' <<<"$out" | sed "s/^/  $n /" | tee -a "$LOG"
  a=$(sed -n 's/^avail=//p' <<<"$out"); [ "${a:-0}" -ge 100 ] || { say "  REFUSE $n: MemAvailable ${a} GB < 100 GB"; bad=1; }
  [ "$(sed -n 's/^tfglm=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a tf-glm53 container is running"; bad=1; }
  [ "$(sed -n 's/^zp=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a zp2a-* / zp2b-* / zp3a-* container exists (an earlier run's: docker rm -f them)"; bad=1; }
  if [ "$(sed -n 's/^apps=//p' <<<"$out")" != 0 ] && [ "${ALLOW_GPU_BUSY:-0}" != 1 ]; then say "  REFUSE $n: compute apps hold the GPU (ALLOW_GPU_BUSY=1 to start anyway)"; bad=1; fi
  grep -q model=missing <<<"$out" && { say "  REFUSE $n: no $MODEL/config.json"; bad=1; }
  grep -q image=missing <<<"$out" && { say "  REFUSE $n: image not pulled"; bad=1; }
done
[ $bad = 0 ] || summary FAIL "refused by the preflight (run.log)" 3
detect_rails || summary FAIL "RoCE rail detection failed (RAILS=1?)" 3

# ---- 1. sync -----------------------------------------------------------------------------------------------------
if [ "${SKIP_SYNC:-0}" != 1 ]; then
  [ -f "$PYSRC/src/tensorfold/families/glm_moe_dsa/cuda/fused.py" ] || summary FAIL "no champion source at $PYSRC" 3
  for n in "${NODES[@]}"; do
    ( rsync -a --delete --exclude /runs/ --exclude /cache/ --exclude /pysrc/ --exclude .zig-cache/ --exclude zig-out/ \
        "$PORT_DIR/" "$n:zig-port/" && rsync -a --delete "$PYSRC/" "$n:zig-port/pysrc/" ) >> "$LOG" 2>&1 \
      || echo "sync to $n failed" >> "$LOCAL/sync.err" &
  done; wait
  [ -s "$LOCAL/sync.err" ] && summary FAIL "$(tr '\n' ' ' < "$LOCAL/sync.err")" 3
  for n in "${NODES[@]}"; do
    if ! rsh "$n" "test -x '$ZIG_DIR/zig'"; then
      say "$n: no $ZIG_DIR/zig - copying it from $ZIG_FROM"
      rsh "$ZIG_FROM" "tar -C '$(dirname "$ZIG_DIR")' -c '$(basename "$ZIG_DIR")'" | rsh "$n" "mkdir -p '$(dirname "$ZIG_DIR")' && tar -C '$(dirname "$ZIG_DIR")' -x" \
        || summary FAIL "could not copy zig to $n" 3
    fi
  done
  say "synced $PORT_DIR and $PYSRC to every node"
fi
for n in "${NODES[@]}"; do rsh "$n" "mkdir -p $WORK/ref $RPORT/cache/zig-local $RPORT/cache/zig-global $RPORT/cache/torch_ext $RPORT/cache/cuda_cache $RPORT/cache/vllm/b12x-compile $RPORT/cache/b12x-roce $RPORT/cache/xdg"; done
USER_FLAG='--user $(id -u):$(id -g)'
COMMON="--label $LABEL $USER_FLAG -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1"
RANK_FLAGS="--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory $MEM_CAP --memory-swap $MEM_CAP"

# ---- 2. prompts: 2b's + the 128K prompt (chat template, thinking on), made once on rank 0's node -----------------
if [ "${SKIP_REF:-0}" != 1 ]; then
  [ -n "$CONTEXT32K" ] && [ -f "$CONTEXT32K" ] || summary FAIL "no 32K background text at $CONTEXT32K (the 32K and 128K prompts need it)" 3
  rsh "${NODES[0]}" "cat > $WORK/context-32k.txt" < "$CONTEXT32K" || summary FAIL "could not copy $CONTEXT32K" 3
  say "prompts: rendering the chat template on ${NODES[0]} (CPU container)"
  rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 8g -v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
    -v $RPORT/tf:/tf:ro -v $WORK:/work -e PYTHONPATH=/opt/tensorfold/src --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/reference3a.py --model /model --make-prompts /work/prompts.json --long $LONG \
    --context-file /work/context-32k.txt --tokens $TOKENS --tokens-long $TOKENS_LONG --tokens-128k $TOKENS_128K \
    --runs $RUNS --runs-long $RUNS_LONG --runs-128k $RUNS_128K" >> "$LOG" 2>&1 || summary FAIL "prompt rendering failed (run.log)" 1
  rsh "${NODES[0]}" "cat $WORK/prompts.json" > "$LOCAL/prompts.json" || summary FAIL "no prompts.json on ${NODES[0]}" 1
  for n in "${NODES[@]}"; do rsh "$n" "cat > $WORK/prompts.json" < "$LOCAL/prompts.json"; done
fi
[ -s "$LOCAL/prompts.json" ] || summary FAIL "no prompts.json" 1
say "prompts: $(grep -o '"name": "[^"]*"' "$LOCAL/prompts.json" | cut -d'"' -f4 | paste -sd' ') ($(wc -c < "$LOCAL/prompts.json") bytes)"
if [ "$CONTEXT" = auto ]; then
  need=$(grep -o '"context_needed": [0-9]*' "$LOCAL/prompts.json" | grep -o '[0-9]*$')
  [ -n "$need" ] || summary FAIL "prompts.json has no context_needed (CONTEXT=N)" 1
  CONTEXT=$(( (need + 1023) / 1024 * 1024 ))
fi
(( CONTEXT < 200000 )) || summary FAIL "context $CONTEXT would turn on DCP (Phase 3b): fewer 128K tokens" 1
say "context: $CONTEXT tokens (both engines: Runner capacity $((CONTEXT + K + 1)))"

# ---- 3. build on every node (CPU only, memory-capped); the new kernels' symbols checked ---------------------------
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  say "build: zig build test-glm53, then fatbins + tf-glm53-generate on all four nodes (docker --memory 24g)"
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "timeout 3600 docker run --rm $COMMON --network none --memory 24g --memory-swap 24g \
      -v $RPORT/tf:/work/tf -v $WORK:/out -v $RPORT/cache:/cache -v $ZIG_DIR:/opt/z:ro -w /work/tf \
      -e ZIG_LOCAL_CACHE_DIR=/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=/cache/zig-global -e PATH=/opt/z:/usr/local/cuda/bin:/usr/bin:/bin \
      --entrypoint bash $IMAGE -c 'set -u; rc=0; zig version
        zig build test-glm53 --summary all || rc=10
        zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix /out/zig-out -j8 fatbins tf-glm53-generate || rc=\$((rc + 20))
        for f in /out/zig-out/fatbin/glm53_*.fatbin; do [ -f \"\$f\" ] && cuobjdump -symbols \"\$f\"; done > /out/symbols-glm53.txt 2>&1
        exit \$rc'" > "$LOCAL/build-r$r.log" 2>&1 &
    bpid[$r]=$!
  done
  brc=0; for r in 0 1 2 3; do wait "${bpid[$r]}" || brc=1; done
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "test -x $WORK/zig-out/bin/tf-glm53-generate" || { say "build failed on ${NODES[$r]}: $(tail -5 "$LOCAL/build-r$r.log" | tr '\n' ' ' | cut -c1-300)"; brc=1; }
  done
  [ $brc = 0 ] || summary FAIL "build failed (build-r*.log: rc 10 host tests, 20 engine build)" 1
  rsh "${NODES[0]}" "cat $WORK/symbols-glm53.txt" > "$LOCAL/symbols-glm53.txt"
  missing=0
  while read -r sym; do
    case "$sym" in "#"*|"") continue ;; esac
    grep -q -- "$sym" "$LOCAL/symbols-glm53.txt" || { say "kernel symbol missing from the fatbins: $sym"; missing=$((missing + 1)); }
  done < "$PORT_DIR/tf/tools/glm53/expected_symbols.txt"
  [ "$missing" = 0 ] || summary FAIL "build: $missing kernel symbols missing (symbols-glm53.txt lists what the fatbins have)" 1
  say "build: OK on all four nodes, every expected kernel symbol present"
fi

# ---- helpers (as 2b) ---------------------------------------------------------------------------------------------
start_ranks() {   # $1 = name prefix, $2 = function printing rank r's docker arguments after the flags
  local r
  for r in 3 2 1 0; do
    rsh "${NODES[$r]}" "docker run -d --name $1-r$r $COMMON $RANK_FLAGS $(nccl_env $r) $($2 $r)" >> "$LOG" 2>&1 \
      || { say "could not start $1-r$r on ${NODES[$r]}"; return 1; }
  done
}
wait_ranks() {    # $1 = name prefix, $2 = timeout (s); 0 when all four exited 0
  local t=0 r st running failed
  while :; do
    running=0; failed=""
    for r in 0 1 2 3; do
      st=$(rsh "${NODES[$r]}" "docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' $1-r$r" 2>/dev/null)
      case "$st" in "true "*) running=$((running + 1)) ;; "false 0") ;; *) failed="$failed r$r($st)" ;; esac
    done
    [ -n "$failed" ] && { say "$1: failed:$failed"; return 1; }
    [ $running = 0 ] && return 0
    (( t >= $2 )) && { say "$1: still running after $2 s"; return 1; }
    if [ "$DROP_CACHES" = 1 ]; then
      for n in "${NODES[@]}"; do rsh "$n" 'sync; sudo -n sysctl -q -w vm.drop_caches=1' >/dev/null 2>&1 & done; wait
    fi
    sleep 15; t=$((t + 15))
    (( t % 300 == 0 )) && say "$1: $running ranks running ($t s); rank 0: $(rsh "${NODES[0]}" "docker logs --tail 1 $1-r0 2>&1" | cut -c1-160)"
  done
}
collect_ranks() { # $1 = name prefix: logs to .4, containers removed
  local r
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "docker logs $1-r$r" > "$LOCAL/$1-r$r.log" 2>&1
    rsh "${NODES[$r]}" "docker rm -f $1-r$r" >/dev/null 2>&1
  done
}
wait_memory() {
  local i n a ok
  for i in $(seq 1 24); do ok=1
    for n in "${NODES[@]}"; do
      a=$(rsh "$n" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null); [ "${a:-0}" -ge 100 ] || ok=0
    done; [ $ok = 1 ] && return 0; sleep 5; done
  return 1
}
jstr() { grep -o "\"$1\": \"[^\"]*\"" "$2" | head -1 | cut -d'"' -f4; }   # a string field of a flat JSON file

# ---- 4. the RoCE one-shot alone (no model): bit check against NCCL, microseconds a reduction --------------------
if [ "${SKIP_BENCH:-0}" != 1 ]; then
  bench_args() {
    echo "-v $WORK:/work --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE --mode roce-bench --rank $1 --world 4 \
      --master $MASTER:$BENCH_PORT --hcas ${ZHCAS[$1]} --gid ${ZGID[$1]} --bench-rows $BENCH_ROWS --bench-iters $BENCH_ITERS \
      --nccl-lib ${BENCH_NCCL:-/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2} \
      --dims 6144 --out /work/roce-bench-r$1.json --timeout 900"
  }
  say "roce-bench: the Zig RoCE one-shot at TP4 (rows $BENCH_ROWS)"
  start_ranks zp3a-bench bench_args || { cleanup; summary FAIL "roce-bench start failed" 1; }
  wait_ranks zp3a-bench 1200; brc=$?
  collect_ranks zp3a-bench
  grep -h '^ROCE-BENCH' "$LOCAL"/zp3a-bench-r0.log | tee -a "$LOG"
  rsh "${NODES[0]}" "cat $WORK/roce-bench-r0.json" > "$LOCAL/roce-bench-r0.json" 2>/dev/null
  [ $brc = 0 ] || say "roce-bench: FAILED or over 65 us (zp3a-bench-r*.log) - continuing to the token check"
fi

# ---- 5. the Python engine served (Glm53Engine, the image's defaults; records the Triton AOT set) ------------------
ref_args() {   # $1 = rank, $2 = out dir under /work, $3 = extra reference3a.py flags
  echo "-v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/tf:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
    -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/$2/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
    -e CUDA_CACHE_PATH=/cache/cuda_cache -e TORCH_CUDA_ARCH_LIST=12.1 \
    -e VLLM_CACHE_ROOT=/cache/vllm -e B12X_COMPILE_CACHE_DIR=/cache/vllm/b12x-compile -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce \
    -e XDG_CACHE_HOME=/cache/xdg -w /tmp --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/reference3a.py --model /model --rank $1 --master $MASTER --port $REF_PORT \
    --prompts /work/${4:-prompts.json} --context $CONTEXT --k $K --out /work/$2 $3"
}
# PROFILE=1: Runner.prefill's torch.profiler over PROFILE_PROMPT's first run (reference3a.py --profile-prefill)
ref_main() { ref_args "$1" ref "--record$([ "$PROFILE" = 1 ] && echo " --profile-prefill $PROFILE_PROMPT")"; }
if [ "${SKIP_REF:-0}" != 1 ]; then
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/ref && mkdir -p $WORK/ref/triton"; done
  wait_memory || summary FAIL "a node is under 100 GB MemAvailable before the reference" 1
  say "reference: Glm53Engine served (RoCE, PROMPT_SP, image defaults; TF_EXL3_PROMPT_DET=slots16) at TP4"
  t0=$(date +%s)
  start_ranks zp3a-ref ref_main || { cleanup; summary FAIL "reference start failed" 1; }
  wait_ranks zp3a-ref "${REF_TIMEOUT:-10800}"; rrc=$?
  collect_ranks zp3a-ref
  grep -h '\[ref3a\] rank 0' "$LOCAL"/zp3a-ref-r0.log | tail -40 | tee -a "$LOG"
  grep -ah 'RoCE reduce unavailable' "$LOCAL"/zp3a-ref-r*.log | head -4 | tee -a "$LOG"
  [ $rrc = 0 ] || { cleanup; summary FAIL "the Python reference failed (zp3a-ref-r*.log: $(tail -3 "$LOCAL/zp3a-ref-r0.log" | tr '\n' ' ' | cut -c1-240))" 1; }
  say "reference: done in $(( $(date +%s) - t0 )) s"
fi
for r in 0 1 2 3; do
  rsh "${NODES[$r]}" "cat $WORK/ref/ref-r$r.json" > "$LOCAL/ref-r$r.json" || summary FAIL "no ref-r$r.json on ${NODES[$r]}" 1
  [ $r = 0 ] || rsh "${NODES[0]}" "cat > $WORK/ref/ref-r$r.json.copy" < "$LOCAL/ref-r$r.json"
  rsh "${NODES[$r]}" "test -f $WORK/ref/aot/aot.json && test -f $WORK/ref/inv_freq.bin && test -s $WORK/ref/nccl.txt && test -s $WORK/ref/tiles-r$r.json && test -s $WORK/ref/bmm_probe.json && test -s $WORK/ref/blas.json" \
    || summary FAIL "${NODES[$r]}: the reference left no aot/, inv_freq.bin, nccl.txt, tiles-r$r.json, bmm_probe.json or blas.json in $WORK/ref" 1
done
rsh "${NODES[0]}" "cat $WORK/ref/blas.json" > "$LOCAL/blas.json"
grep -q '"roce": true' "$LOCAL/ref-r0.json" || say "WARNING: the reference ran WITHOUT the RoCE one-shot (its log says why): the gate will fail"

# ---- 5b. optional: the Python engine at the image's default prompt-experts mode, prefill timing only --------------
if [ "$SPEED_DEFAULT" = 1 ]; then
  rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 4g -v $WORK:/work --entrypoint python3 $IMAGE -c '
import json; d = json.load(open(\"/work/prompts.json\"))
d[\"prompts\"] = [dict(p, runs=[\"g:0\"], tokens=8) for p in d[\"prompts\"] if p[\"name\"].startswith((\"prose32k\", \"long128k\"))]
json.dump(d, open(\"/work/prompts-speed.json\", \"w\"))'" >> "$LOG" 2>&1
  for n in "${NODES[@]}"; do rsh "${NODES[0]}" "cat $WORK/prompts-speed.json" | rsh "$n" "cat > $WORK/prompts-speed.json"; rsh "$n" "rm -rf $WORK/refd && mkdir -p $WORK/refd"; done
  ref_speed() { ref_args "$1" refd --speed-only prompts-speed.json; }
  wait_memory || summary FAIL "a node is still under 100 GB MemAvailable after the reference" 1
  say "reference (speed only): the image's default TF_EXL3_PROMPT_DET (red.add)"
  start_ranks zp3a-refd ref_speed || { cleanup; summary FAIL "speed-only reference start failed" 1; }
  wait_ranks zp3a-refd "${REF_TIMEOUT:-10800}" || say "speed-only reference failed (zp3a-refd-r*.log; not gated)"
  collect_ranks zp3a-refd
  rsh "${NODES[0]}" "cat $WORK/refd/ref-r0.json" > "$LOCAL/refd-r0.json" 2>/dev/null
fi

# ---- 6. tf-glm53-generate --mode 3a at TP4 -----------------------------------------------------------------------
wait_memory || summary FAIL "a node is still under 100 GB MemAvailable after the reference (another job?)" 1
declare -a NLIB=()
for r in 0 1 2 3; do NLIB[$r]=$(rsh "${NODES[$r]}" "head -1 $WORK/ref/nccl.txt"); done
CUBLAS=$(jstr libcublas "$LOCAL/blas.json"); EXPERTS=$(jstr experts_impl "$LOCAL/blas.json")
[ -n "$CUBLAS" ] || summary FAIL "the reference's blas.json names no libcublas (torch loaded none?)" 1
say "zig: NCCL ${NLIB[0]}, cuBLAS $CUBLAS ($(jstr preferred_blas "$LOCAL/blas.json")), experts ${EXPERTS:-?}"
gen_args() {
  echo "-v $MODEL:/model:ro -v $WORK:/work --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE \
    --mode 3a --rank $1 --world 4 --master $MASTER:$ZIG_PORT --model /model --aot /work/ref/aot \
    --prompts /work/prompts.json --tokens $TOKENS --context $CONTEXT --k $K --prewarm 1 --graphs 1 --fast-load 1 \
    --draft-vocab 32768 --mtp-reuse 2 --inv-ref /work/ref/inv_freq.bin --nccl-lib ${NLIB[$1]} \
    --roce 1 --hcas ${ZHCAS[$1]} --gid ${ZGID[$1]} --roce-health 1 --tiles /work/ref/tiles-r$1.json \
    --prompt-rows 8192 --prompt-rows-short 4096 --prompt-sp 1 --experts ${EXPERTS:-shared} --pe-det slots16 \
    --cublas-lib $CUBLAS --unpack-mb 384 --bmm-probe /work/ref/bmm_probe.json \
    --side $ZSIDE --l2pf $ZL2PF --l2pf-mb $ZL2PF_MB --multi-select $ZMULTI_SELECT --device-cands $ZDEVICE_CANDS \
    --mtp-dense $ZMTP_DENSE $([ "$PROFILE" = 1 ] && echo "--prompt-prof $PROFILE_PROMPT") \
    --timeout ${ZIG_RDV_TIMEOUT:-5400} --out /work/gen-r$1.json"
}
say "zig: tf-glm53-generate --mode 3a at TP4 (docker --memory $MEM_CAP a rank)"
t0=$(date +%s)
start_ranks zp3a-gen gen_args || { cleanup; summary FAIL "zig start failed" 1; }
wait_ranks zp3a-gen "${GEN_TIMEOUT:-10800}"; grc=$?
collect_ranks zp3a-gen
grep -h -E '^(RESULT|WARN|bmm probe|rank 0 |error)' "$LOCAL"/zp3a-gen-r*.log | tail -60 | tee -a "$LOG"
if [ "$PROFILE" = 1 ]; then   # both engines' prefill phases (rank 0), side by side in prefill-prof.txt
  { echo "== python (torch.profiler, GPU time a kernel class; streams overlap) =="
    grep -ah 'PREFILL-PROF' "$LOCAL"/zp3a-ref-r0.log
    echo "== zig (CUDA events a phase and chunk; main / comm streams overlap) =="
    grep -ah '^PROMPT-PROF rank 0' "$LOCAL"/zp3a-gen-r0.log; } > "$LOCAL/prefill-prof.txt"
  tee -a "$LOG" < "$LOCAL/prefill-prof.txt"
fi
[ $grc = 0 ] || { cleanup; summary FAIL "tf-glm53-generate failed (zp3a-gen-r*.log: $(grep -h RESULT "$LOCAL"/zp3a-gen-r*.log | head -2 | tr '\n' ' ' | cut -c1-240))" 1; }
say "zig: done in $(( $(date +%s) - t0 )) s"

# ---- 7. compare on rank 0's node ------------------------------------------------------------------------------------
for r in 0 1 2 3; do
  rsh "${NODES[$r]}" "cat $WORK/gen-r$r.json" > "$LOCAL/gen-r$r.json" || summary FAIL "no gen-r$r.json on ${NODES[$r]}" 1
  [ $r = 0 ] || rsh "${NODES[0]}" "cat > $WORK/gen-r$r.json" < "$LOCAL/gen-r$r.json"
done
extra=""
[ -s "$LOCAL/roce-bench-r0.json" ] && extra="$extra --roce-bench $WORK/roce-bench-r0.json"
[ -s "$LOCAL/refd-r0.json" ] && extra="$extra --speed-default $WORK/refd/ref-r0.json"
rsh "${NODES[0]}" "python3 -B $RPORT/tf/tools/glm53/compare_phase3a.py $WORK/ref/ref-r0.json $WORK/gen-r0.json $WORK/gen-r1.json $WORK/gen-r2.json $WORK/gen-r3.json \
  --ref-ranks $WORK/ref/ref-r1.json.copy $WORK/ref/ref-r2.json.copy $WORK/ref/ref-r3.json.copy --ratio $RATIO --tpr-tol $TPR_TOL $extra" > "$LOCAL/compare.log" 2>&1
crc=$?
grep -v '^PHASE3A' "$LOCAL/compare.log" | tee -a "$LOG"
line=$(grep '^PHASE3A' "$LOCAL/compare.log" | tail -1)
[ -n "$line" ] || line="PHASE3A FAIL compare printed no result (rc $crc)"
echo "$line work=$LOCAL" | tee -a "$LOG"
[ $crc = 0 ] && exit 0 || exit 1
