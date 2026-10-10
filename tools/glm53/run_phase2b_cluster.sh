#!/bin/bash
# GLM-5.3 Zig port, Phase 2b gate: FULL GLM-5.3 + the MTP layer at TP4 over NCCL on the four GB10 nodes - MTP
# drafts (k=2, verify windows of 3 rows, the q4 draft head over the 32,768-id draft vocabulary), keyed sampled decoding
# (temperature / top_p from generation_config.json; top_k 20 = the server's default through the sharded "local" head,
# top_k 0 = top_p alone through Glm53Engine._sample's 264-candidate rule) and greedy, CUDA graphs for the decode
# windows, the O_DIRECT + io_uring load - the Zig engine (tf-glm53-generate --mode 2b) against the Python engine's own
# Runner (tools/glm53/reference2b.py, same knobs, NCCL all-gathers). Token streams must be IDENTICAL to Python's, and
# drafted == serial in both engines; decode tok/s of both are printed side by side (32K prose prompt included).
# Prints one line at the end:  PHASE2B PASS|FAIL ...
#
# Run from .4 (the orchestrator: it only syncs, starts containers over ssh and copies results; every build and GPU
# job runs on the nodes). Free the cluster first - this script stops nothing it did not start:
#   bash ~/zig-port/run_phase2b_cluster.sh
#
# Steps: preflight (refuses: any node < 100 GB MemAvailable, a tf-glm53* container running, GPU compute apps unless
# ALLOW_GPU_BUSY=1, a leftover zp2a-* / zp2b-* container) -> rsync ~/zig-port and the champion source to every node ->
# prompts (chat template, thinking on; rank 0 node, CPU; the 32K prose prompt from CONTEXT32K) -> zig build on every
# node in the image (CPU, docker --memory 24g) -> the sampler check (Python's exact_sampling vs sampling.zig on 1,500
# generated draws, rank 0 node, CPU; a FAIL stops here) -> the Python reference at TP4 (4 containers; records each
# node's Triton AOT set) -> tf-glm53-generate --mode 2b at TP4 (4 containers) -> compare (compare_phase2b.py).
#
# Environment (defaults in brackets):
#   NODES    fabric addresses, rank 0 first            [10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5]
#   MODEL    checkpoint, same path on every node       [/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit]
#   IMAGE    the TP4 image                             [ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05]
#   PORT_DIR this folder on .4 (synced to ~/zig-port)  [$HOME/zig-port]
#   PYSRC    the champion's Python source on .4        [required] (-> nodes' zig-port/pysrc)
#   ZIG_DIR  zig 0.17.0 folder on the nodes            [$HOME/opt/zig]; copied from ZIG_FROM [10.100.10.5] if missing
#   TOKENS [256] picks a run of the short prompts   TOKENS_LONG [512] picks a run of the 32K prose prompt
#   RUNS [g:2,g:0,s20:2,s20:0,s0:2,s0:0] the short prompts' runs ("g" greedy, "s<top_k>" sampled; ":K" drafts, 0 serial)
#   RUNS_LONG [s20:2,s20:0] the 32K prose prompt's runs   K [2] MTP drafts the engines are built for
#   CONTEXT32K  tfbench's 32K background text on .4   [$HOME/glm53-speed-20260920/context-32k.txt] ("": no 32K prompt)
#   CONTEXT [auto: the longest prompt + its picks, rounded up to 1024]  WINDOW [128] (prompt chunk rows, both engines)
#   LONG [4500] (the 2a long prompt's tokens, 0: none)  LAYERS [0 = all]  SERVED_PROSE [32.2] (the served figure)
#   FAST_LOAD [1] (Zig: O_DIRECT + io_uring)  PREWARM [1]  GRAPHS [1]  SAMPLER_CASES [1500]  SAMPLER_FATAL [1]
#   REF_PORT [29571] (TCPStore)  ZIG_PORT [29581] (tf-glm53-generate rendezvous)  IF [enp1s0f1np1]  RAILS [2]
#   MEM_CAP  docker --memory of each rank [110g]     DROP_CACHES [1]: sudo -n drop clean page cache every 15 s
#            while ranks load (as one-shot.sh wait does; skipped where sudo needs a password)
#   SKIP_SYNC=1  SKIP_BUILD=1  SKIP_REF=1 (with REUSE=<stamp of an earlier run>: its prompts, zig-out, ref + aot)
#   ALLOW_GPU_BUSY=1   start even while other compute apps hold a GPU
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
RUNS=${RUNS:-g:2,g:0,s20:2,s20:0,s0:2,s0:0}
RUNS_LONG=${RUNS_LONG:-s20:2,s20:0}
K=${K:-2}
CONTEXT32K=${CONTEXT32K-$HOME/glm53-speed-20260920/context-32k.txt}
CONTEXT=${CONTEXT:-auto}
WINDOW=${WINDOW:-128}
LONG=${LONG:-4500}
LAYERS=${LAYERS:-0}
SERVED_PROSE=${SERVED_PROSE:-32.2}
FAST_LOAD=${FAST_LOAD:-1}
PREWARM=${PREWARM:-1}
GRAPHS=${GRAPHS:-1}
SAMPLER_CASES=${SAMPLER_CASES:-1500}
SAMPLER_FATAL=${SAMPLER_FATAL:-1}
REF_PORT=${REF_PORT:-29571}
ZIG_PORT=${ZIG_PORT:-29581}
IF=${IF:-enp1s0f1np1}
RAILS=${RAILS:-2}
MEM_CAP=${MEM_CAP:-110g}
DROP_CACHES=${DROP_CACHES:-1}
STAMP=${REUSE:-$(date -u +%Y%m%d-%H%M%S)}
RHOME=${RHOME:-$HOME}                         # the nodes' home (same user, same path)
RPORT=$RHOME/zig-port
WORK=$RPORT/runs/p2b-$STAMP                   # on every node
LOCAL=$PORT_DIR/runs/p2b-$STAMP               # on .4: logs and the result files
LABEL="tensorfold.glm53-phase2b=$STAMP"
mkdir -p "$LOCAL"
LOG=$LOCAL/run.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
summary() { echo "PHASE2B $1 $2 work=$LOCAL" | tee -a "$LOG"; exit "${3:-1}"; }
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
declare -a HCAS=() RGIDS=() NGID=()
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
    say "rank $r (${NODES[$r]}): rails ${HCAS[$r]} on [$nets], GIDs ${RGIDS[$r]}"
  done
}
nccl_env() {
  local r=$1 e="-e NCCL_SOCKET_IFNAME=$IF -e NCCL_IB_HCA=${HCAS[$1]}"
  if [ -n "${NGID[$r]}" ]; then e="$e -e NCCL_IB_GID_INDEX=${NGID[$r]}"
  else e="$e -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET"; fi
  if (( XNIC )); then e="$e -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1"; else e="$e -e NCCL_CROSS_NIC=0"; fi
  echo "$e"
}

# ---- 0. preflight: never the thing that OOMs a node or collides with a live engine --------------------------------
say "phase 2b: nodes ${NODES[*]} (rank 0 first), model $MODEL, image $IMAGE, context $CONTEXT, runs [$RUNS] x $TOKENS picks, 32K [$RUNS_LONG] x $TOKENS_LONG, k $K, work $WORK"
bad=0
for r in 0 1 2 3; do n=${NODES[$r]}
  out=$(rsh "$n" "echo avail=\$(awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo)
    echo tfglm=\$(docker ps --format '{{.Names}} {{.Image}}' | grep -c tf-glm53)
    echo zp2a=\$(docker ps -a --format '{{.Names}}' | grep -cE '^zp2[ab]-')
    echo apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ -f '$MODEL/config.json' ] && echo model=ok || echo model=missing
    docker image inspect '$IMAGE' >/dev/null 2>&1 && echo image=ok || echo image=missing
    [ -f ~/NODE_CLAIM.txt ] && sed 's/^/claim: /' ~/NODE_CLAIM.txt; true" 2>&1) || { say "rank $r ($n): ssh failed"; bad=1; continue; }
  say "rank $r ($n): $(grep -v '^claim:' <<<"$out" | tr '\n' ' ')"
  grep '^claim:' <<<"$out" | sed "s/^/  $n /" | tee -a "$LOG"
  a=$(sed -n 's/^avail=//p' <<<"$out"); [ "${a:-0}" -ge 100 ] || { say "  REFUSE $n: MemAvailable ${a} GB < 100 GB"; bad=1; }
  [ "$(sed -n 's/^tfglm=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a tf-glm53 container is running"; bad=1; }
  [ "$(sed -n 's/^zp2a=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a zp2a-* / zp2b-* container exists (an earlier run's: docker rm -f them)"; bad=1; }
  if [ "$(sed -n 's/^apps=//p' <<<"$out")" != 0 ] && [ "${ALLOW_GPU_BUSY:-0}" != 1 ]; then say "  REFUSE $n: compute apps hold the GPU (ALLOW_GPU_BUSY=1 to start anyway)"; bad=1; fi
  grep -q model=missing <<<"$out" && { say "  REFUSE $n: no $MODEL/config.json"; bad=1; }
  grep -q image=missing <<<"$out" && { say "  REFUSE $n: image not pulled"; bad=1; }
done
[ $bad = 0 ] || summary FAIL "refused by the preflight (run.log)" 3
detect_rails || summary FAIL "RoCE rail detection failed (RAILS=1?)" 3

# ---- 1. sync: the port (keeping each node's runs/ and cache/) and the champion source; zig 0.17.0 ---------------
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
for n in "${NODES[@]}"; do rsh "$n" "mkdir -p $WORK/ref $RPORT/cache/zig-local $RPORT/cache/zig-global $RPORT/cache/torch_ext $RPORT/cache/cuda_cache"; done
USER_FLAG='--user $(id -u):$(id -g)'
COMMON="--label $LABEL $USER_FLAG -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1"
RANK_FLAGS="--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory $MEM_CAP --memory-swap $MEM_CAP"

# ---- 2. prompts: the chat template's ids (thinking on), made once on rank 0's node, copied to the others ---------
if [ "${SKIP_REF:-0}" != 1 ]; then
  if [ -n "${PROMPTS:-}" ]; then
    cp "$PROMPTS" "$LOCAL/prompts.json"
  else
    say "prompts: rendering the chat template on ${NODES[0]} (CPU container)"
    cfile=""
    if [ -n "$CONTEXT32K" ]; then
      [ -f "$CONTEXT32K" ] || summary FAIL "no 32K background text at $CONTEXT32K (CONTEXT32K= for none)" 3
      rsh "${NODES[0]}" "cat > $WORK/context-32k.txt" < "$CONTEXT32K" || summary FAIL "could not copy $CONTEXT32K" 3
      cfile="--context-file /work/context-32k.txt"
    fi
    rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 8g -v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
      -v $RPORT/tf:/tf:ro -v $WORK:/work -e PYTHONPATH=/opt/tensorfold/src --entrypoint python3 $IMAGE \
      -B /tf/tools/glm53/reference2b.py --model /model --make-prompts /work/prompts.json --long $LONG $cfile \
      --tokens $TOKENS --tokens-long $TOKENS_LONG --runs $RUNS --runs-long $RUNS_LONG" >> "$LOG" 2>&1 \
      || summary FAIL "prompt rendering failed (run.log)" 1
    rsh "${NODES[0]}" "cat $WORK/prompts.json" > "$LOCAL/prompts.json" || summary FAIL "no prompts.json on ${NODES[0]}" 1
  fi
  for n in "${NODES[@]}"; do rsh "$n" "cat > $WORK/prompts.json" < "$LOCAL/prompts.json"; done
fi
[ -s "$LOCAL/prompts.json" ] || summary FAIL "no prompts.json" 1
say "prompts: $(grep -o '"name": "[^"]*"' "$LOCAL/prompts.json" | cut -d'"' -f4 | paste -sd' ') ($(wc -c < "$LOCAL/prompts.json") bytes)"
if [ "$CONTEXT" = auto ]; then
  need=$(grep -o '"context_needed": [0-9]*' "$LOCAL/prompts.json" | grep -o '[0-9]*$')
  [ -n "$need" ] || summary FAIL "prompts.json has no context_needed (CONTEXT=N)" 1
  CONTEXT=$(( (need + 1023) / 1024 * 1024 ))
fi
say "context: $CONTEXT tokens (both engines: Runner capacity $((CONTEXT + K + 1)))"

# ---- 3. build on every node (CPU only, memory-capped): host tests, the fatbins, tf-glm53-generate ---------------
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  say "build: zig build test-glm53, then fatbins + tf-glm53-generate on all four nodes (docker --memory 24g)"
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "timeout 3600 docker run --rm $COMMON --network none --memory 24g --memory-swap 24g \
      -v $RPORT/tf:/work/tf -v $WORK:/out -v $RPORT/cache:/cache -v $ZIG_DIR:/opt/z:ro -w /work/tf \
      -e ZIG_LOCAL_CACHE_DIR=/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=/cache/zig-global -e PATH=/opt/z:/usr/local/cuda/bin:/usr/bin:/bin \
      --entrypoint bash $IMAGE -c 'set -u; rc=0; zig version
        zig build test-glm53 --summary all || rc=10
        zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix /out/zig-out -j8 fatbins tf-glm53-generate || rc=\$((rc + 20))
        exit \$rc'" > "$LOCAL/build-r$r.log" 2>&1 &
    bpid[$r]=$!
  done
  brc=0; for r in 0 1 2 3; do wait "${bpid[$r]}" || brc=1; done
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "test -x $WORK/zig-out/bin/tf-glm53-generate" || { say "build failed on ${NODES[$r]}: $(tail -5 "$LOCAL/build-r$r.log" | tr '\n' ' ' | cut -c1-300)"; brc=1; }
  done
  [ $brc = 0 ] || summary FAIL "build failed (build-r*.log: rc 10 host tests, 20 engine build)" 1
  say "build: OK on all four nodes"
fi

# ---- 3b. the sampler, host only: Python's exact_sampling against sampling.zig on generated draws and seeds ---------
if [ "${SKIP_SAMPLER:-0}" != 1 ]; then
  rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 16g --memory-swap 16g -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
    -v $RPORT/tf:/tf:ro -v $WORK:/out -e PYTHONPATH=/opt/tensorfold/src --entrypoint bash $IMAGE -c \
    'python3 -B /tf/tools/glm53/sampler_cases.py --out /out/sampler-cases.json --n $SAMPLER_CASES && /out/zig-out/bin/tf-glm53-generate --sampler-check /out/sampler-cases.json'" \
    > "$LOCAL/sampler.log" 2>&1
  sline=$(grep '^SAMPLER' "$LOCAL/sampler.log" | tail -1)
  say "sampler: ${sline:-no result (sampler.log)}"
  case "$sline" in "SAMPLER PASS"*) ;; *) [ "$SAMPLER_FATAL" = 1 ] && summary FAIL "the Zig sampler disagrees with Python's ($sline; sampler.log)" 1 ;; esac
fi

# ---- helpers: run one container a rank, wait for all four, drop clean page cache while they load -----------------
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
wait_memory() {   # every node back near full before the next four ranks start (up to 2 min)
  local i n a ok
  for i in $(seq 1 24); do ok=1
    for n in "${NODES[@]}"; do
      a=$(rsh "$n" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null); [ "${a:-0}" -ge 100 ] || ok=0
    done; [ $ok = 1 ] && return 0; sleep 5; done
  return 1
}
opt_layers() { [ "$LAYERS" != 0 ] && echo "--layers $LAYERS"; }

# ---- 4. the Python reference at TP4 (records each node's Triton AOT set) -----------------------------------------
ref_args() {
  echo "-v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/tf:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
    -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/ref/triton -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
    -e CUDA_CACHE_PATH=/cache/cuda_cache -e TORCH_CUDA_ARCH_LIST=12.1 -w /tmp --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/reference2b.py --model /model --rank $1 --world 4 --master $MASTER --port $REF_PORT \
    --prompts /work/prompts.json --context $CONTEXT --window $WINDOW --k $K $(opt_layers) \
    $( [ "$PREWARM" = 1 ] || echo --no-prewarm ) --out /work/ref --record"
}
if [ "${SKIP_REF:-0}" != 1 ]; then
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/ref && mkdir -p $WORK/ref/triton"; done
  say "reference: the Python engine's Runner at TP4 (docker --memory $MEM_CAP a rank)"
  t0=$(date +%s)
  start_ranks zp2b-ref ref_args || { cleanup; summary FAIL "reference start failed" 1; }
  wait_ranks zp2b-ref "${REF_TIMEOUT:-10800}"; rrc=$?
  collect_ranks zp2b-ref
  grep -h '\[ref2b\] rank 0' "$LOCAL"/zp2b-ref-r0.log | tail -40 | tee -a "$LOG"
  [ $rrc = 0 ] || { cleanup; summary FAIL "the Python reference failed (zp2b-ref-r*.log: $(tail -3 "$LOCAL/zp2b-ref-r0.log" | tr '\n' ' ' | cut -c1-240))" 1; }
  say "reference: done in $(( $(date +%s) - t0 )) s"
fi
for r in 0 1 2 3; do
  rsh "${NODES[$r]}" "cat $WORK/ref/ref-r$r.json" > "$LOCAL/ref-r$r.json" || summary FAIL "no ref-r$r.json on ${NODES[$r]}" 1
  [ $r = 0 ] || rsh "${NODES[0]}" "cat > $WORK/ref/ref-r$r.json.copy" < "$LOCAL/ref-r$r.json"
done
for r in 0 1 2 3; do
  rsh "${NODES[$r]}" "test -f $WORK/ref/aot/aot.json && test -f $WORK/ref/inv_freq.bin && test -s $WORK/ref/nccl.txt" \
    || summary FAIL "${NODES[$r]}: the reference left no aot/aot.json, inv_freq.bin or nccl.txt in $WORK/ref" 1
done

# ---- 5. tf-glm53-generate at TP4 -------------------------------------------------------------------------------
wait_memory || summary FAIL "a node is still under 100 GB MemAvailable after the reference (another job?)" 1
declare -a NLIB=()
for r in 0 1 2 3; do NLIB[$r]=$(rsh "${NODES[$r]}" "head -1 $WORK/ref/nccl.txt"); done
say "zig: NCCL library (the one the Python engine opened): ${NLIB[0]}"
gen_args() {
  echo "-v $MODEL:/model:ro -v $WORK:/work --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE \
    --mode 2b --rank $1 --world 4 --master $MASTER:$ZIG_PORT --model /model --aot /work/ref/aot \
    --prompts /work/prompts.json --tokens $TOKENS --context $CONTEXT --window $WINDOW --k $K $(opt_layers) \
    --prewarm $PREWARM --graphs $GRAPHS --fast-load $FAST_LOAD --draft-vocab 32768 --mtp-reuse 2 \
    --inv-ref /work/ref/inv_freq.bin --nccl-lib ${NLIB[$1]} --timeout ${ZIG_RDV_TIMEOUT:-5400} --out /work/gen-r$1.json"
}
say "zig: tf-glm53-generate --mode 2b at TP4 (docker --memory $MEM_CAP a rank)"
t0=$(date +%s)
start_ranks zp2b-gen gen_args || { cleanup; summary FAIL "zig start failed" 1; }
wait_ranks zp2b-gen "${GEN_TIMEOUT:-10800}"; grc=$?
collect_ranks zp2b-gen
grep -h -E '^(RESULT|WARN|inv_freq|rank 0 |error)' "$LOCAL"/zp2b-gen-r*.log | tail -48 | tee -a "$LOG"
[ $grc = 0 ] || { cleanup; summary FAIL "tf-glm53-generate failed (zp2b-gen-r*.log: $(grep -h RESULT "$LOCAL"/zp2b-gen-r*.log | head -2 | tr '\n' ' ' | cut -c1-240))" 1; }
say "zig: done in $(( $(date +%s) - t0 )) s"

# ---- 6. compare on rank 0's node (the other ranks' outputs copied there) ------------------------------------------
for r in 0 1 2 3; do
  rsh "${NODES[$r]}" "cat $WORK/gen-r$r.json" > "$LOCAL/gen-r$r.json" || summary FAIL "no gen-r$r.json on ${NODES[$r]}" 1
  [ $r = 0 ] || rsh "${NODES[0]}" "cat > $WORK/gen-r$r.json" < "$LOCAL/gen-r$r.json"
done
rsh "${NODES[0]}" "python3 -B $RPORT/tf/tools/glm53/compare_phase2b.py $WORK/ref/ref-r0.json $WORK/gen-r0.json $WORK/gen-r1.json $WORK/gen-r2.json $WORK/gen-r3.json \
  --ref-ranks $WORK/ref/ref-r1.json.copy $WORK/ref/ref-r2.json.copy $WORK/ref/ref-r3.json.copy --served-prose $SERVED_PROSE" > "$LOCAL/compare.log" 2>&1
crc=$?
grep -v '^PHASE2B' "$LOCAL/compare.log" | tee -a "$LOG"
line=$(grep '^PHASE2B' "$LOCAL/compare.log" | tail -1)
[ -n "$line" ] || line="PHASE2B FAIL compare printed no result (rc $crc)"
echo "$line work=$LOCAL" | tee -a "$LOG"
[ $crc = 0 ] && exit 0 || exit 1
