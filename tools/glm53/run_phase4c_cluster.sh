#!/bin/bash
# GLM-5.3 Zig port, Phase 4c gate: the speculative drafters for concurrent streams (--parallel N with DRAFTER=dflash
# or dspark; DRAFTER=mtp is Phase 4b's run) in the Zig engine, on the four GB10 nodes at TP4, every reference in this
# same run. A copy of run_phase4b_cluster.sh (same preflight, watchdog, fail-fast waits, reference, AOT coverage,
# clients) with the drafter knobs. Steps and gates (tools/glm53/gate4b.py gate --drafter DRAFTER):
#   (cli) tf-glm53-generate --mode 4b --parallel N --draft-mode DRAFTER (--dflash / --dspark DRAFTER_PATH): every
#         stream drafts with the drafter - each group concurrently, each prompt alone through the concurrent decoder,
#         and (SINGLE=1) through the one-stream path (Runner.run with the one-stream drafter); greedy and keyed sampled.
#         GATE cli: tokens equal everywhere, every rank agreeing, the run's draft mode the one asked;
#   (py)  DRAFTER=dflash: the Python server with DFlash2 for every concurrent request (TF_GLM53_DFLASH, TF_GLM53_CONC_MODE
#         =dflash: multi.MultiDrafter) -> py-<kind>.json; DRAFTER=mtp: Phase 4b's Python MTP server -> py-<kind>.json;
#         DRAFTER=dspark: no Python DSpark server (engine.py refuses DSpark with --parallel). PY_BAR=1 (default, drafters
#         only): the Python MTP server too, the speed bar -> py-mtp-<kind>.json;
#   (zig) tensorfold-native serve --parallel N with the drafter (TF_GLM53_MTP=DRAFTER) and the same client;
#         GATE conc-prose / conc-code: each greedy reply alone == the same prompt and seed among N streams (the server's
#         single-stream vs concurrent identity); GATE text-*: every greedy reply == Python's (dflash: Python DFlash2;
#         dspark: Python MTP - greedy replies never depend on the drafts); GATE speed-*: aggregate tok/s at N streams
#         >= RATIO x Python's with the same drafter, or (dspark) >= BAR_RATIO x Python MTP's (0: reported only);
#         GATE served-cli as 4b. The DFlash2 run also reports Zig against the Python MTP bar.
# Prints one line at the end:  PHASE4C PASS|FAIL ... work=...
#
# Run from .4 (the orchestrator: it only syncs, starts containers over ssh and copies results). Free the cluster
# first - this script stops nothing it did not start:
#   DRAFTER=dflash bash ~/zig-port/run_phase4c_cluster.sh
#   DRAFTER=dspark bash ~/zig-port/run_phase4c_cluster.sh
#   DRAFTER=dflash DRAFTER_PATH=/mnt/spark2-models-local/<private checkpoint> PRIVATE=1 bash ~/zig-port/run_phase4c_cluster.sh
# A memory watchdog runs from launch to exit (label tensorfold.glm53-phase4c=<stamp>): < 4 GB MemAvailable or > 2 GB
# swap on any node removes this run's containers ("PHASE4C FAIL watchdog").
#
# Environment (defaults in brackets; the rest as run_phase4b_cluster.sh):
#   TREE [tf-cdraft]  DRAFTER [dflash] (mtp | dflash | dspark)  DRAFTER_PATH [dflash: /mnt/spark2-models-local/
#   GLM-5.3-DFlash2; dspark: /mnt/spark2-models-local/GLM-5.3-speculator.dspark-keys-ft2] (a checkpoint directory on
#   every node; private checkpoints by path only, PRIVATE=1 keeps the path out of the logs)  PY_BAR [1]  BAR_RATIO [1.0]
#   DFLASH_DEPTH [7] DFLASH_CONFIDENCE [0.3] DSPARK_POLICY [confidence] DSPARK_CONFIDENCE [0.3] DSPARK_DEPTH []
#   DRAFT_CUT_DFLASH [] (TF_GLM53_DRAFT_CUT_DFLASH on both servers and the CLI; unset: TF_GLM53_DRAFT_CUT's 0.6)
#   TEMPLATE2 [] (a node directory with another reference's manifest.json + aot/, e.g. a Phase 4 reference4.py run's
#   ref-df: the drafter kernels' captured variants as sweep templates)  K [2]  PARALLEL [4]  CONTEXT [32768]
#   ports: ZHTTP [18992] PYHTTP [18993] REF_PORT [29872] ZIG_PORT [29882] PY_MPORT [29892] CLI_PORT [29902]
#   ZIG_ENV [] (extra docker arguments for the Zig server's ranks, e.g. -e TF_GLM53_ROUND_PROFILE=1)
#   SKIP_SYNC SKIP_BUILD SKIP_REF SKIP_COVER SKIP_CLI SKIP_PY SKIP_BAR SKIP_ZIG =1 (with REUSE=<stamp>: its work dir)
set -u
NODES=(${NODES:-10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5})
[ ${#NODES[@]} -eq 4 ] || { echo "NODES must list four fabric addresses, rank 0 first"; exit 2; }
MASTER=${MASTER:-${NODES[0]}}
MODEL=${MODEL:-/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit}
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05}
PORT_DIR=${PORT_DIR:-$HOME/zig-port}
TREE=${TREE:-tf-cdraft}                       # the source tree under zig-port/ that is built and whose tools run
PYSRC=${PYSRC:-$HOME/tf-wt/dspark}
ZIG_DIR=${ZIG_DIR:-$HOME/opt/zig}
ZIG_FROM=${ZIG_FROM:-10.100.10.5}
CONTEXT32K=${CONTEXT32K-$HOME/glm53-speed-20260920/context-32k.txt}
PARALLEL=${PARALLEL:-4}
CONTEXT=${CONTEXT:-32768}
K=${K:-2}
TOKENS=${TOKENS:-512}
SAMPLED=${SAMPLED:-1}
SINGLE=${SINGLE:-1}
RATIO=${RATIO:-0.97}
DRAFTER=${DRAFTER:-dflash}
case "$DRAFTER" in
  dflash) DRAFTER_PATH=${DRAFTER_PATH:-/mnt/spark2-models-local/GLM-5.3-DFlash2} ;;
  dspark) DRAFTER_PATH=${DRAFTER_PATH:-/mnt/spark2-models-local/GLM-5.3-speculator.dspark-keys-ft2} ;;
  mtp) DRAFTER_PATH="" ;;
  *) echo "DRAFTER=$DRAFTER: mtp, dflash or dspark"; exit 2 ;;
esac
PRIVATE=${PRIVATE:-0}
DSHOW=$([ "$PRIVATE" = 1 ] && echo "(private checkpoint)" || echo "$DRAFTER_PATH")
PY_BAR=${PY_BAR:-1}
BAR_RATIO=${BAR_RATIO:-1.0}
DFLASH_DEPTH=${DFLASH_DEPTH:-7}
DFLASH_CONFIDENCE=${DFLASH_CONFIDENCE:-0.3}
DSPARK_POLICY=${DSPARK_POLICY:-confidence}
DSPARK_CONFIDENCE=${DSPARK_CONFIDENCE:-0.3}
DSPARK_DEPTH=${DSPARK_DEPTH:-}
DRAFT_CUT_DFLASH=${DRAFT_CUT_DFLASH:-}
TEMPLATE2=${TEMPLATE2:-}
REUSE_ON=${REUSE_ON:-1}
REF_FROM=${REF_FROM:-}
ZSIDE=${ZSIDE:-af}
ZL2PF=${ZL2PF:-1}
ZL2PF_MB=${ZL2PF_MB:-8}
ZMULTI_SELECT=${ZMULTI_SELECT:-1}
ZDEVICE_CANDS=${ZDEVICE_CANDS:-1}
ZMTP_DENSE=${ZMTP_DENSE:-1}
ZHTTP=${ZHTTP:-18992}
PYHTTP=${PYHTTP:-18993}
REF_PORT=${REF_PORT:-29872}
ZIG_PORT=${ZIG_PORT:-29882}
PY_MPORT=${PY_MPORT:-29892}
CLI_PORT=${CLI_PORT:-29902}
IF=${IF:-enp1s0f1np1}
RAILS=${RAILS:-2}
MEM_CAP=${MEM_CAP:-110g}
DROP_CACHES=${DROP_CACHES:-1}
STALL_MIN=${STALL_MIN:-60}
PROGRESS_MIN=${PROGRESS_MIN:-15}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-3600}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-3600}
SWEEP_TIMEOUT=${SWEEP_TIMEOUT:-10800}
SWEEP_MEM=${SWEEP_MEM:-32g}
PROMPT_ROWS=${PROMPT_ROWS:-8192}
ZERR_PAT=${ZERR_PAT:-'MissingTritonVariant|AmbiguousTritonVariant|NoTritonSet|request [0-9]+ failed|not admitted|concurrent round failed|out of step|stream error|panic: |Segmentation fault|illegal memory access|CUDA_ERROR_|unhandled cuda error|NCCL WARN .*(failed|error)|Traceback \(most recent'}
STAMP=${REUSE:-$(date -u +%Y%m%d-%H%M%S)}
RHOME=${RHOME:-$HOME}
RPORT=$RHOME/zig-port
WORK=$RPORT/runs/p4c-$STAMP                   # on every node
LOCAL=$PORT_DIR/runs/p4c-$STAMP               # on .4: logs and the result files
LABEL="tensorfold.glm53-phase4c=$STAMP"
mkdir -p "$LOCAL"
LOG=$LOCAL/run.log
# wait for this shell's jobs except the memory watchdog (a bare `wait` would wait for it forever)
waitjobs() { local j; for j in $(jobs -p); do [ "$j" = "${WD_PID:-}" ] || wait "$j" 2>/dev/null; done; }
rm -f "$LOCAL/WATCHDOG" "$LOCAL/watchdog.stop"
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; }
phase() { echo "$1" > "$LOCAL/phase"; say "== phase $1"; }
cleanup() {
  for n in "${NODES[@]}"; do
    rsh "$n" "docker ps -aq --filter label=$LABEL | xargs -r docker rm -f" >/dev/null 2>&1 &
  done; waitjobs
}
WD_PID=""
stop_watchdog() { touch "$LOCAL/watchdog.stop"; [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null; WD_PID=""; }
summary() {
  stop_watchdog
  echo "PHASE4C $1 $2 work=$LOCAL" | tee -a "$LOG"; exit "${3:-1}"
}
watchdog_fired() { [ -e "$LOCAL/WATCHDOG" ]; }
check_wd() { watchdog_fired && { cleanup; summary FAIL "watchdog: $(cat "$LOCAL/WATCHDOG")" 4; }; return 0; }
fatal() { say "FATAL: $1"; cleanup; summary FAIL "$1" "${2:-1}"; }
trap 'say "interrupted: removing this run'"'"'s containers"; cleanup; summary FAIL "interrupted" 143' TERM INT
trap 'stop_watchdog' EXIT

# ---- the memory watchdog: every ~5 s, every node; < 4 GB available or > 2 GB swap -> this run's containers go ----
watchdog() {
  local n out a s
  while [ ! -e "$LOCAL/watchdog.stop" ]; do
    for n in "${NODES[@]}"; do
      ( out=$(rsh -o ConnectTimeout=4 "$n" "awk '/MemAvailable/{a=\$2} /SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print int(a/1024), int((t-f)/1024)}' /proc/meminfo" 2>/dev/null) || exit 0
        a=${out% *}; s=${out#* }
        echo "$(date +%s) $(cat "$LOCAL/phase" 2>/dev/null) $n $a $s" >> "$LOCAL/watchdog.log"
        if [ "${a:-99999}" -lt 4096 ] || [ "${s:-0}" -gt 2048 ]; then
          [ -e "$LOCAL/WATCHDOG" ] || echo "$n MemAvailable ${a} MB swap ${s} MB in phase $(cat "$LOCAL/phase" 2>/dev/null)" > "$LOCAL/WATCHDOG"
        fi ) &
    done; wait
    if [ -e "$LOCAL/WATCHDOG" ] && [ ! -e "$LOCAL/WATCHDOG.done" ]; then
      touch "$LOCAL/WATCHDOG.done"
      echo "[$(date -u +%H:%M:%S)] WATCHDOG: $(cat "$LOCAL/WATCHDOG") - removing this run's containers" >> "$LOG"
      for n in "${NODES[@]}"; do rsh "$n" "docker ps -aq --filter label=$LABEL | xargs -r docker rm -f" >/dev/null 2>&1 & done; wait
    fi
    sleep 5
  done
}
watchdog & WD_PID=$!

# minimum MemAvailable (GB) a node during a phase, from the watchdog log: "10.100.10.1:12.3,..."
mem_min() { awk -v p="$1" '$2 == p { k = $3; v = $4 / 1024; if (!(k in m) || v < m[k]) m[k] = v } END { s = ""; for (k in m) s = s (s ? "," : "") k ":" sprintf("%.1f", m[k]); print s }' "$LOCAL/watchdog.log" 2>/dev/null; }

# ---- rails (as 3a) -------------------------------------------------------------------------------------------------
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
    if [ -n "${NGID[$r]}" ]; then ZHCAS[$r]=${HCAS[$r]}; ZGID[$r]=${NGID[$r]}
    else ZHCAS[$r]=${HCAS[$r]%%,*}; ZGID[$r]=${RGIDS[$r]%%,*}; fi
    say "rank $r (${NODES[$r]}): rails ${HCAS[$r]} on [$nets], GIDs ${RGIDS[$r]}; RoCE one-shot on ${ZHCAS[$r]} (GID ${ZGID[$r]})"
  done
}
nccl_env() {
  local r=$1 e="-e NCCL_SOCKET_IFNAME=$IF -e NCCL_IB_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_GIDS=${RGIDS[$1]}"
  if [ -n "${NGID[$r]}" ]; then e="$e -e NCCL_IB_GID_INDEX=${NGID[$r]}"
  else e="$e -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET"; fi
  if (( XNIC )); then e="$e -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1"; else e="$e -e NCCL_CROSS_NIC=0"; fi
  echo "$e"
}

# ---- 0. preflight ------------------------------------------------------------------------------------------------
phase preflight
say "phase 4c: nodes ${NODES[*]} (rank 0 first), tree $PORT_DIR/$TREE, model $MODEL, image $IMAGE, --parallel $PARALLEL, context $CONTEXT, k $K, drafter $DRAFTER ${DSHOW}, $TOKENS tokens, work $WORK"
[ -f "$PORT_DIR/$TREE/zig/src/families/glm_moe_dsa/cuda_mdraft.zig" ] || summary FAIL "no $PORT_DIR/$TREE/zig/src/families/glm_moe_dsa/cuda_mdraft.zig (TREE=?)" 2
[ "$DRAFTER" = mtp ] || (( PARALLEL * 8 <= 64 )) || summary FAIL "PARALLEL=$PARALLEL: a drafter's N blocks of 8 rows must fit one 64-row pass" 2
(( PARALLEL >= 2 )) || summary FAIL "PARALLEL=$PARALLEL: concurrent streams need 2 or more" 2
(( PARALLEL * (K + 1) <= 32 )) || summary FAIL "PARALLEL x (K + 1) = $((PARALLEL * (K + 1))) > 32: not a decode window" 2
(( CONTEXT < 200000 )) || summary FAIL "CONTEXT=$CONTEXT would turn DCP on: concurrent streams need it off" 2
bad=0
for r in 0 1 2 3; do n=${NODES[$r]}
  out=$(rsh "$n" "echo avail=\$(awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo)
    echo swap=\$(awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print int((t-f)/1024)}' /proc/meminfo)
    echo tfglm=\$(docker ps --format '{{.Names}} {{.Image}}' | grep -c tf-glm53)
    echo zp=\$(docker ps -a --format '{{.Names}}' | grep -cE '^zp')
    echo apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ -f '$MODEL/config.json' ] && echo model=ok || echo model=missing
    docker image inspect '$IMAGE' >/dev/null 2>&1 && echo image=ok || echo image=missing
    [ -z '$DRAFTER_PATH' ] || { [ -f '$DRAFTER_PATH/config.json' ] && echo drafter=ok || echo drafter=missing; }
    [ -f ~/NODE_CLAIM.txt ] && sed 's/^/claim: /' ~/NODE_CLAIM.txt; true" 2>&1) || { say "rank $r ($n): ssh failed"; bad=1; continue; }
  say "rank $r ($n): $(grep -v '^claim:' <<<"$out" | tr '\n' ' ')"
  grep '^claim:' <<<"$out" | sed "s/^/  $n /" | tee -a "$LOG"
  a=$(sed -n 's/^avail=//p' <<<"$out"); [ "${a:-0}" -ge 100 ] || { say "  REFUSE $n: MemAvailable ${a} GB < 100 GB"; bad=1; }
  s=$(sed -n 's/^swap=//p' <<<"$out"); [ "${s:-0}" -le 2048 ] || { say "  REFUSE $n: ${s} MB of swap in use (> 2 GB)"; bad=1; }
  [ "$(sed -n 's/^tfglm=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a tf-glm53 container is running"; bad=1; }
  [ "$(sed -n 's/^zp=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a zp* container exists (an earlier run's: docker rm -f them)"; bad=1; }
  if [ "$(sed -n 's/^apps=//p' <<<"$out")" != 0 ] && [ "${ALLOW_GPU_BUSY:-0}" != 1 ]; then say "  REFUSE $n: compute apps hold the GPU (ALLOW_GPU_BUSY=1 to start anyway)"; bad=1; fi
  grep -q model=missing <<<"$out" && { say "  REFUSE $n: no $MODEL/config.json"; bad=1; }
  grep -q image=missing <<<"$out" && { say "  REFUSE $n: image not pulled"; bad=1; }
  grep -q drafter=missing <<<"$out" && { say "  REFUSE $n: no $DRAFTER checkpoint (config.json) at ${DSHOW}"; bad=1; }
done
[ $bad = 0 ] || summary FAIL "refused by the preflight (run.log)" 3
detect_rails || summary FAIL "RoCE rail detection failed (RAILS=1?)" 3
check_wd

# ---- 1. sync -----------------------------------------------------------------------------------------------------
phase sync
if [ "${SKIP_SYNC:-0}" != 1 ]; then
  [ -f "$PYSRC/src/tensorfold/families/glm_moe_dsa/cuda/multi.py" ] || summary FAIL "no Python source with multi.py at $PYSRC" 3
  rm -f "$LOCAL/sync.err"
  for n in "${NODES[@]}"; do
    ( rsync -a --delete --exclude /runs/ --exclude /cache/ --exclude /pysrc/ --exclude .zig-cache/ --exclude zig-out/ \
        "$PORT_DIR/" "$n:zig-port/" && rsync -a --delete --exclude .git/ "$PYSRC/" "$n:zig-port/pysrc/" ) >> "$LOG" 2>&1 \
      || echo "sync to $n failed" >> "$LOCAL/sync.err" &
  done; waitjobs
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
for n in "${NODES[@]}"; do rsh "$n" "mkdir -p $WORK/ref $WORK/cli $WORK/gate $RPORT/cache/zig-local $RPORT/cache/zig-global $RPORT/cache/torch_ext $RPORT/cache/cuda_cache $RPORT/cache/vllm/b12x-compile $RPORT/cache/b12x-roce $RPORT/cache/xdg"; done
USER_FLAG='--user $(id -u):$(id -g)'
DMOUNT=$([ -n "$DRAFTER_PATH" ] && echo "-v $DRAFTER_PATH:$DRAFTER_PATH:ro")
# the drafter's flags (tf-glm53-generate) and environment (both servers)
zig_drafter_flags() {
  case "$DRAFTER" in
    dflash) echo "--dflash $DRAFTER_PATH --dflash-depth $DFLASH_DEPTH --dflash-confidence $DFLASH_CONFIDENCE --draft-mode dflash" ;;
    dspark) echo "--dspark $DRAFTER_PATH --dspark-policy $DSPARK_POLICY --dspark-confidence $DSPARK_CONFIDENCE${DSPARK_DEPTH:+ --dspark-depth $DSPARK_DEPTH} --draft-mode dspark" ;;
    *) echo "--draft-mode mtp" ;;
  esac
}
drafter_env() {
  local e="${DRAFT_CUT_DFLASH:+-e TF_GLM53_DRAFT_CUT_DFLASH=$DRAFT_CUT_DFLASH}"
  case "$DRAFTER" in
    dflash) e="$e -e TF_GLM53_DFLASH=$DRAFTER_PATH -e TF_GLM53_DFLASH_DEPTH=$DFLASH_DEPTH -e TF_GLM53_DFLASH_CONFIDENCE=$DFLASH_CONFIDENCE" ;;
    dspark) e="$e -e TF_GLM53_DSPARK=$DRAFTER_PATH -e TF_GLM53_DSPARK_POLICY=$DSPARK_POLICY -e TF_GLM53_DSPARK_CONFIDENCE=$DSPARK_CONFIDENCE${DSPARK_DEPTH:+ -e TF_GLM53_DSPARK_DEPTH=$DSPARK_DEPTH}" ;;
  esac
  echo "$e"
}
COMMON="--label $LABEL $USER_FLAG -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1"
rank_flags() { echo "--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory $1 --memory-swap $1"; }
CACHE_ENV="-e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e CUDA_CACHE_PATH=/cache/cuda_cache -e TORCH_CUDA_ARCH_LIST=12.1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e B12X_COMPILE_CACHE_DIR=/cache/vllm/b12x-compile -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce -e XDG_CACHE_HOME=/cache/xdg"
# a CPU container on rank 0's node: the prompt makers and the gate (host network: the servers on 127.0.0.1)
cpu_job() {
  rsh "${NODES[0]}" "docker run --rm $COMMON --network host --memory 8g -v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
    -v $RPORT/$TREE:/tf:ro -v $WORK:/work -e PYTHONPATH=/opt/tensorfold/src --entrypoint python3 $IMAGE -B $*"
}
to_all() { local f=$1 n; rsh "${NODES[0]}" "cat $WORK/$f" > "$LOCAL/$(basename "$f")" || return 1; for n in "${NODES[@]:1}"; do rsh "$n" "cat > $WORK/$f" < "$LOCAL/$(basename "$f")" || return 1; done; }

# ---- 2. prompts: conc_chat.py's as 4b groups; reference3a's for the reference -------------------------------------
phase prompts
if [ "${SKIP_REF:-0}" != 1 ] || [ ! -s "$LOCAL/prompts-4b.json" ]; then
  cpu_job /tf/tools/glm53/gate4b.py make-prompts --model /model --out /work/gate/prompts-4b.json --parallel $PARALLEL \
    --tokens $TOKENS --sampled $SAMPLED >> "$LOG" 2>&1 || summary FAIL "gate4b.py make-prompts failed (run.log)" 1
  to_all gate/prompts-4b.json || summary FAIL "could not spread prompts-4b.json" 1
  if [ -z "$REF_FROM" ]; then
    [ -n "$CONTEXT32K" ] && [ -f "$CONTEXT32K" ] || summary FAIL "no 32K background text at $CONTEXT32K" 3
    for n in "${NODES[@]}"; do rsh "$n" "cat > $WORK/context-32k.txt" < "$CONTEXT32K" || summary FAIL "could not copy $CONTEXT32K to $n" 3; done
    cpu_job /tf/tools/glm53/reference3a.py --model /model --make-prompts /work/prompts.json --context-file /work/context-32k.txt \
      >> "$LOG" 2>&1 || summary FAIL "reference3a.py --make-prompts failed (run.log)" 1
    to_all prompts.json || summary FAIL "could not spread prompts.json" 1
  fi
fi
need=$(grep -o '"context_needed": [0-9]*' "$LOCAL/prompts-4b.json" | grep -o '[0-9]*$')
(( ${need:-0} <= CONTEXT )) || summary FAIL "the 4b prompts need $need tokens of context > CONTEXT=$CONTEXT" 1
check_wd

# ---- 3. build on every node (CPU only, memory-capped) ------------------------------------------------------------
phase build
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  say "build: zig build test-glm53, then fatbins + tf-glm53-generate + native (tensorfold-native) on all four nodes (docker --memory 24g)"
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "timeout 3600 docker run --rm $COMMON --network none --memory 24g --memory-swap 24g \
      -v $RPORT/$TREE:/work/tf -v $WORK:/out -v $RPORT/cache:/cache -v $ZIG_DIR:/opt/z:ro -w /work/tf \
      -e ZIG_LOCAL_CACHE_DIR=/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=/cache/zig-global -e PATH=/opt/z:/usr/local/cuda/bin:/usr/bin:/bin \
      --entrypoint bash $IMAGE -c 'set -u; rc=0; zig version
        zig build test-glm53 --summary all || rc=10
        zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix /out/zig-out -j8 fatbins tf-glm53-generate native || rc=\$((rc + 20))
        for f in /out/zig-out/fatbin/glm53_*.fatbin /out/zig-out/fatbin/qmm_group.fatbin; do [ -f \"\$f\" ] && cuobjdump -symbols \"\$f\"; done > /out/symbols-glm53.txt 2>&1
        exit \$rc'" > "$LOCAL/build-r$r.log" 2>&1 &
    bpid[$r]=$!
  done
  brc=0; for r in 0 1 2 3; do wait "${bpid[$r]}" || brc=1; done
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "test -x $WORK/zig-out/bin/tf-glm53-generate && test -x $WORK/zig-out/native/bin/tensorfold-native" \
      || { say "build failed on ${NODES[$r]}: $(tail -5 "$LOCAL/build-r$r.log" | tr '\n' ' ' | cut -c1-300)"; brc=1; }
  done
  [ $brc = 0 ] || summary FAIL "build failed (build-r*.log: rc 10 host tests, 20 engine / server build)" 1
  rsh "${NODES[0]}" "cat $WORK/symbols-glm53.txt" > "$LOCAL/symbols-glm53.txt"
  missing=0
  while read -r sym; do
    case "$sym" in "#"*|"") continue ;; esac
    grep -q -- "$sym" "$LOCAL/symbols-glm53.txt" || { say "kernel symbol missing from the fatbins: $sym"; missing=$((missing + 1)); }
  done < "$PORT_DIR/$TREE/tools/glm53/expected_symbols.txt"
  [ "$missing" = 0 ] || summary FAIL "build: $missing kernel symbols missing (symbols-glm53.txt)" 1
  say "build: OK on all four nodes, every expected kernel symbol present"
fi
check_wd

# ---- helpers (as 3b, watchdog-aware) -----------------------------------------------------------------------------
start_ranks() {   # $1 = name prefix, $2 = function printing rank r's docker arguments after the flags, $3 = memory cap
  local r
  for r in 3 2 1 0; do
    rsh "${NODES[$r]}" "docker run -d --name $1-r$r $COMMON $(rank_flags "${3:-$MEM_CAP}") $(nccl_env $r) $($2 $r)" >> "$LOG" 2>&1 \
      || { say "could not start $1-r$r on ${NODES[$r]}"; return 1; }
  done
}
drop_caches() { [ "$DROP_CACHES" = 1 ] || return 0; local n; for n in "${NODES[@]}"; do rsh "$n" 'sync; sudo -n sysctl -q -w vm.drop_caches=1' >/dev/null 2>&1 & done; waitjobs; }
WAIT_REASON=""
wait_ranks() {    # $1 = name prefix, $2 = timeout (s), $3 = stall minutes; 0 when all four exited 0
  local t=0 r st running failed errs h last="" still=0 stall=${3:-$STALL_MIN}
  WAIT_REASON=""
  while :; do
    watchdog_fired && { WAIT_REASON="$1: watchdog fired"; say "$WAIT_REASON"; return 1; }
    running=0; failed=""
    for r in 0 1 2 3; do
      st=$(rsh "${NODES[$r]}" "docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' $1-r$r" 2>/dev/null)
      case "$st" in "true "*) running=$((running + 1)) ;; "false 0") ;; *) failed="$failed r$r($st)" ;; esac
    done
    [ -n "$failed" ] && { WAIT_REASON="$1: failed:$failed: $(rsh "${NODES[0]}" "docker logs --tail 3 $1-r0 2>&1" | tr '\n' ' ' | cut -c1-300)"; say "$WAIT_REASON"; return 1; }
    [ $running = 0 ] && return 0
    errs=$(rsh "${NODES[0]}" "docker logs $1-r0 2>&1 | grep -E '$ZERR_PAT' | head -3" 2>/dev/null)
    [ -n "$errs" ] && { WAIT_REASON="$1: error in rank 0's log: $(tr '\n' ' ' <<<"$errs" | cut -c1-400)"; say "$WAIT_REASON"; return 1; }
    h=$(rsh "${NODES[0]}" "docker logs --tail 50 $1-r0 2>&1 | md5sum" 2>/dev/null)
    if [ "$h" = "$last" ]; then still=$((still + 15)); else still=0; last=$h; fi
    (( still >= stall * 60 )) && { WAIT_REASON="$1: no output from rank 0 for $stall min"; say "$WAIT_REASON"; return 1; }
    (( t >= $2 )) && { WAIT_REASON="$1: still running after $2 s"; say "$WAIT_REASON"; return 1; }
    sleep 15; t=$((t + 15))
    (( t % 300 == 0 )) && say "$1: $running ranks running ($t s); rank 0: $(rsh "${NODES[0]}" "docker logs --tail 1 $1-r0 2>&1" | cut -c1-160)"
  done
}
wait_http() {     # $1 = name prefix, $2 = HTTP port on rank 0, $3 = timeout (s): 0 once /v1/models answers 200
  local t=0 r code
  while :; do
    watchdog_fired && { say "$1: watchdog fired while loading"; return 1; }
    code=$(rsh "${NODES[0]}" "curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$2/v1/models" 2>/dev/null)
    [ "$code" = 200 ] && { say "$1: up after $t s"; return 0; }
    for r in 0 1 2 3; do
      rsh "${NODES[$r]}" "docker ps -q -f name=^$1-r$r\$" | grep -q . || { say "$1: rank $r exited: $(rsh "${NODES[$r]}" "docker logs --tail 5 $1-r$r 2>&1" | tr '\n' ' ' | cut -c1-300)"; return 1; }
    done
    (( t >= $3 )) && { say "$1: not up after $3 s"; return 1; }
    drop_caches
    sleep 15; t=$((t + 15))
    (( t % 300 == 0 )) && say "$1: loading ($t s); rank 0: $(rsh "${NODES[0]}" "docker logs --tail 1 $1-r0 2>&1" | cut -c1-160)"
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
  for i in $(seq 1 36); do ok=1
    for n in "${NODES[@]}"; do
      a=$(rsh "$n" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null); [ "${a:-0}" -ge 100 ] || ok=0
    done; [ $ok = 1 ] && return 0; sleep 5; done
  return 1
}
jstr() { grep -o "\"$1\": \"[^\"]*\"" "$2" | head -1 | cut -d'"' -f4; }
GATE_FAIL=""
note_fail() { GATE_FAIL="$GATE_FAIL $1"; say "FAILED: $1"; }

# A watched client run (gate4b.py client in a named container on rank 0's node): ends at once on the client failing, a
# rank of the server exiting, a request error in the server's rank 0 log (ZERR_PAT), PROGRESS_MIN minutes without a
# change in the client's or the server's log, or CLIENT_TIMEOUT. Fatal (the run ends) on any of them.
#   $1 = label  $2 = server container prefix  the rest: gate4b.py client's arguments
client_watch() {
  local label=$1 srv=$2; shift 2
  local cname="zp4c-c-$label" t=0 last="" still=0 h="" now r st errs rc=0 cpid reason=""
  rsh "${NODES[0]}" "docker rm -f $cname" >/dev/null 2>&1
  rsh "${NODES[0]}" "docker run --rm --name $cname $COMMON --network host --memory 4g -v $RPORT/$TREE:/tf:ro -v $WORK:/work \
    --entrypoint python3 $IMAGE -B /tf/tools/glm53/gate4b.py client $* --idle $((PROGRESS_MIN * 60))" > "$LOCAL/client-$label.log" 2>&1 &
  cpid=$!
  while kill -0 "$cpid" 2>/dev/null; do
    sleep 15; t=$((t + 15))
    if watchdog_fired; then reason="watchdog: $(cat "$LOCAL/WATCHDOG")"; break; fi
    for r in 0 1 2 3; do
      st=$(rsh "${NODES[$r]}" "docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' $srv-r$r" 2>/dev/null)
      case "$st" in "true "*) ;; *) reason="$srv-r$r not running (${st:-gone}) during $label: $(rsh "${NODES[$r]}" "docker logs --tail 3 $srv-r$r 2>&1" | tr '\n' ' ' | cut -c1-300)"; break 2 ;; esac
    done
    errs=$(rsh "${NODES[0]}" "docker logs $srv-r0 2>&1 | grep -E '$ZERR_PAT' | head -3" 2>/dev/null)
    [ -n "$errs" ] && { reason="server error during $label: $(tr '\n' ' ' <<<"$errs" | cut -c1-400)"; break; }
    h=$(rsh "${NODES[0]}" "docker logs --tail 50 $srv-r0 2>&1 | md5sum" 2>/dev/null)
    now="$h $(md5sum < "$LOCAL/client-$label.log")"
    if [ "$now" = "$last" ]; then still=$((still + 15)); else still=0; last=$now; fi
    (( still >= PROGRESS_MIN * 60 )) && { reason="no progress for $PROGRESS_MIN min during $label"; break; }
    (( t >= CLIENT_TIMEOUT )) && { reason="$label still running after $CLIENT_TIMEOUT s"; break; }
  done
  if [ -n "$reason" ]; then
    rsh "${NODES[0]}" "docker rm -f $cname" >/dev/null 2>&1
    kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
  else
    wait "$cpid"; rc=$?
    errs=$(rsh "${NODES[0]}" "docker logs $srv-r0 2>&1 | grep -E '$ZERR_PAT' | head -3" 2>/dev/null)
    [ -n "$errs" ] && reason="server error during $label: $(tr '\n' ' ' <<<"$errs" | cut -c1-400)"
    [ -z "$reason" ] && [ $rc != 0 ] && reason="client $label failed (exit $rc): $(grep -E 'error|Error' "$LOCAL/client-$label.log" | tail -3 | tr '\n' ' ' | cut -c1-400)"
  fi
  cat "$LOCAL/client-$label.log" >> "$LOG"
  [ -z "$reason" ] && return 0
  collect_ranks "$srv"
  fatal "$reason"
}
# the client runs a server takes: prose and code, streams 1 and N, greedy; SAMPLED=1 also sampled ($1 = py | zig)
clients() {
  local who=$1 srv=$2 url=$3 kind
  for kind in prose code; do
    client_watch "$who-$kind" "$srv" --url $url --kind $kind --streams 1,$PARALLEL --tokens $TOKENS --greedy --out /work/gate/$who-$kind.json
    if [ "$SAMPLED" = 1 ]; then
      client_watch "$who-$kind-s" "$srv" --url $url --kind $kind --streams 1,$PARALLEL --tokens $TOKENS --out /work/gate/$who-$kind-s.json
    fi
  done
}

# ---- 4. the Python reference (reference3a.py --record: AOT set, tiles, inv_freq, NCCL, BLAS, bmm probe) ----------
phase ref
if [ -n "$REF_FROM" ]; then
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/ref && cp -a '$REF_FROM' $WORK/ref" || summary FAIL "$n: no $REF_FROM" 3; done
  say "reference: reused $REF_FROM"
elif [ "${SKIP_REF:-0}" != 1 ]; then
  rneed=$(grep -o '"context_needed": [0-9]*' "$LOCAL/prompts.json" | grep -o '[0-9]*$')
  [ -n "$rneed" ] || summary FAIL "prompts.json has no context_needed" 1
  REF_CONTEXT=$(( (rneed + 1023) / 1024 * 1024 ))
  ref_main() {
    echo "-v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/$TREE:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
      -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/ref/triton $CACHE_ENV -w /tmp --entrypoint python3 $IMAGE \
      -B /tf/tools/glm53/reference3a.py --model /model --rank $1 --master $MASTER --port $REF_PORT \
      --prompts /work/prompts.json --context $REF_CONTEXT --k $K --out /work/ref --record"
  }
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/ref && mkdir -p $WORK/ref/triton"; done
  wait_memory || fatal "a node is under 100 GB MemAvailable before the reference"
  say "reference: Glm53Engine served configuration at TP4, context $REF_CONTEXT (reference3a.py --record)"
  start_ranks zp4c-ref ref_main || { cleanup; summary FAIL "reference start failed" 1; }
  wait_ranks zp4c-ref "${REF_TIMEOUT:-10800}" "$STALL_MIN"; rrc=$?
  collect_ranks zp4c-ref
  check_wd
  [ $rrc = 0 ] || fatal "the Python reference failed: $WAIT_REASON (zp4c-ref-r*.log)"
fi
for r in 0 1 2 3; do
  rsh "${NODES[$r]}" "test -f $WORK/ref/aot/aot.json && test -f $WORK/ref/inv_freq.bin && test -s $WORK/ref/nccl.txt && test -s $WORK/ref/tiles-r$r.json && test -s $WORK/ref/bmm_probe.json && test -s $WORK/ref/blas.json" \
    || summary FAIL "${NODES[$r]}: the reference left no aot/, inv_freq.bin, nccl.txt, tiles-r$r.json, bmm_probe.json or blas.json in $WORK/ref" 1
done
rsh "${NODES[0]}" "cat $WORK/ref/blas.json" > "$LOCAL/blas.json"
declare -a NLIB=()
for r in 0 1 2 3; do NLIB[$r]=$(rsh "${NODES[$r]}" "head -1 $WORK/ref/nccl.txt"); done
CUBLAS=$(jstr libcublas "$LOCAL/blas.json"); EXPERTS=$(jstr experts_impl "$LOCAL/blas.json")
[ -n "$CUBLAS" ] || summary FAIL "the reference's blas.json names no libcublas" 1
say "engines: context $CONTEXT, --parallel $PARALLEL, NCCL ${NLIB[0]}, cuBLAS $CUBLAS, experts ${EXPERTS:-shared}"

# ---- 5. AOT coverage with the concurrent windows: every launch the Zig engine can make, or nothing Zig starts -----
listv() {   # $1 = aot dir, $2 = needs file (under /work)
  rsh "${NODES[0]}" "docker run --rm $COMMON --gpus all --network none --memory 16g -v $MODEL:/model:ro -v $WORK:/work \
    $DMOUNT --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE --mode list-variants --model /model --world 4 \
    --context $CONTEXT --prompt-rows $PROMPT_ROWS --window 128 --k $K --dcp 1 --parallel $PARALLEL $(lv_drafter) \
    --tiles /work/ref/tiles-r0.json,/work/ref/tiles-r1.json,/work/ref/tiles-r2.json,/work/ref/tiles-r3.json \
    --aot /work/$1 --out /work/$2" > "$LOCAL/cover-listv-$(basename "$2" .json).log" 2>&1
}
lv_drafter() { case "$DRAFTER" in dflash) echo "--dflash $DRAFTER_PATH" ;; dspark) echo "--dspark $DRAFTER_PATH" ;; esac; }
TEMPLATES="--template /work/ref"
[ -n "$TEMPLATE2" ] && TEMPLATES="$TEMPLATES --template /work/template2"
ZAOT=ref/aot
if [ "${SKIP_COVER:-0}" != 1 ]; then
  phase aot-cover
  for r in 1 2 3; do   # every rank's tile table on rank 0's node (the prompt GEMM's shapes)
    rsh "${NODES[$r]}" "cat $WORK/ref/tiles-r$r.json" | rsh "${NODES[0]}" "cat > $WORK/ref/tiles-r$r.json" \
      || fatal "aot: could not copy tiles-r$r.json from ${NODES[$r]}"
  done
  rsh "${NODES[0]}" "rm -rf $WORK/cover $WORK/aot-full && mkdir -p $WORK/cover/triton" || fatal "aot: no work dir"
  if [ -n "$TEMPLATE2" ]; then
    rsh "${NODES[0]}" "rm -rf $WORK/template2 && cp -a '$TEMPLATE2' $WORK/template2 && test -f $WORK/template2/manifest.json" \
      || fatal "aot: TEMPLATE2=$TEMPLATE2 has no manifest.json on ${NODES[0]}"
  fi
  listv ref/aot cover/needs.json; rc=$?
  grep -E '^(variants|VARIANTS|  missing)' "$LOCAL/cover-listv-needs.log" | head -12 | tee -a "$LOG"
  [ $rc = 0 ] || [ $rc = 3 ] || fatal "aot: list-variants failed (exit $rc, cover-listv-needs.log)"
  say "aot: compiling the missing variants (sweep_aot.py on ${NODES[0]}, templates: the reference${TEMPLATE2:+ and TEMPLATE2}; the concurrent drafter kernels without a template take their defaults)"
  rsh "${NODES[0]}" "timeout $SWEEP_TIMEOUT docker run --rm $COMMON --gpus all --network none --memory $SWEEP_MEM --memory-swap $SWEEP_MEM \
    -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/$TREE:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
    -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/cover/triton $CACHE_ENV -w /tmp --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/sweep_aot.py --needs /work/cover/needs.json $TEMPLATES --out /work/cover/sweep --merged /work/aot-full" \
    > "$LOCAL/cover-sweep.log" 2>&1 || fatal "aot: sweep_aot.py failed (cover-sweep.log: $(grep -E 'FAILED|Error' "$LOCAL/cover-sweep.log" | tail -2 | tr '\n' ' ' | cut -c1-300))"
  grep '^\[sweep\]' "$LOCAL/cover-sweep.log" | grep -v '/[0-9]* compiled (' | tail -6 | tee -a "$LOG"
  listv aot-full cover/check.json; rc=$?
  grep -E '^(VARIANTS|  missing)' "$LOCAL/cover-listv-check.log" | head -12 | tee -a "$LOG"
  [ $rc = 0 ] || fatal "aot: the merged set aot-full still misses launches the Zig engine can make (cover-listv-check.log)"
  for r in 1 2 3; do
    rsh "${NODES[0]}" "tar -C $WORK -c aot-full" | rsh "${NODES[$r]}" "rm -rf $WORK/aot-full && tar -C $WORK -x" \
      || fatal "aot: could not copy aot-full to ${NODES[$r]}"
  done
  ZAOT=aot-full
  say "aot: aot-full complete for context $CONTEXT, --parallel $PARALLEL, drafter $DRAFTER, on every node"
  check_wd
elif rsh "${NODES[0]}" "test -f $WORK/aot-full/aot.json"; then
  ZAOT=aot-full; say "SKIP_COVER=1: using the earlier $WORK/aot-full unchecked"
else
  say "WARNING: SKIP_COVER=1 and no aot-full: the Zig engine runs on the reference's capture alone"
fi

# ---- 6. the Zig CLI: tf-glm53-generate --mode 4b (concurrent == alone == one stream) ------------------------------
if [ "${SKIP_CLI:-0}" != 1 ]; then
  phase zig-cli
  cli_args() {
    echo "-v $MODEL:/model:ro -v $WORK:/work $DMOUNT ${DRAFT_CUT_DFLASH:+-e TF_GLM53_DRAFT_CUT_DFLASH=$DRAFT_CUT_DFLASH} --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE \
      --mode 4b --rank $1 --world 4 --master $MASTER:$CLI_PORT --model /model --aot /work/$ZAOT \
      --prompts /work/gate/prompts-4b.json --tokens $TOKENS --context $CONTEXT --k $K --parallel $PARALLEL --single $SINGLE \
      --prewarm 1 --graphs 1 --fast-load 1 --draft-vocab 32768 --mtp-reuse 2 --inv-ref /work/ref/inv_freq.bin \
      --nccl-lib ${NLIB[$1]} --roce 1 --hcas ${ZHCAS[$1]} --gid ${ZGID[$1]} --roce-health 1 --tiles /work/ref/tiles-r$1.json \
      --prompt-rows 8192 --prompt-rows-short 4096 --prompt-sp 1 --experts ${EXPERTS:-shared} --pe-det slots16 \
      --cublas-lib $CUBLAS --unpack-mb 384 --bmm-probe /work/ref/bmm_probe.json \
      --side $ZSIDE --l2pf $ZL2PF --l2pf-mb $ZL2PF_MB --multi-select $ZMULTI_SELECT --device-cands $ZDEVICE_CANDS \
      --mtp-dense $ZMTP_DENSE --stop-eos 1 --reuse 0 --dcp 1 $(zig_drafter_flags) \
      --timeout ${ZIG_RDV_TIMEOUT:-5400} --out /work/cli/cli-r$1.json"
  }
  wait_memory || fatal "zig cli: a node under 100 GB MemAvailable"
  say "zig cli: tf-glm53-generate --mode 4b --parallel $PARALLEL --draft-mode $DRAFTER at TP4, context $CONTEXT, k $K, AOT /work/$ZAOT"
  start_ranks zp4c-cli cli_args || { collect_ranks zp4c-cli; fatal "zig cli start failed"; }
  wait_ranks zp4c-cli "${CLI_TIMEOUT:-10800}" "$STALL_MIN"; crc=$?
  collect_ranks zp4c-cli
  grep -h -E '^(RESULT|rank 0 RESULT4B)' "$LOCAL"/zp4c-cli-r*.log | tail -24 | tee -a "$LOG"
  check_wd
  rsh "${NODES[0]}" "cp $WORK/cli/cli-r0.json $WORK/gate/cli-4b.json" 2>/dev/null
  # a stream that differs is a gate failure, reported below; a crash ends the run
  [ $crc = 0 ] || grep -q 'RESULT glm53-generate FAIL rank 0: phase 4b' "$LOCAL/zp4c-cli-r0.log" \
    || fatal "the Zig CLI failed: $WAIT_REASON (zp4c-cli-r*.log)"
fi

# ---- 7. the Python servers: the same drafter (dflash; mtp as 4b) and the MTP speed bar, each with the client ------
py_args() {   # $1 = rank; PY_MODE: "drafter" (DFlash2 for every concurrent request) or "mtp"
  local de=""
  [ "$PY_MODE" = drafter ] && de="$DMOUNT $(drafter_env) -e TF_GLM53_CONC_MODE=$DRAFTER"
  echo "-v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $WORK:/work -v $RPORT/cache:/cache $de \
    -e PYTHONPATH=/opt/tensorfold/src $CACHE_ENV -e TF_GLM53_PROMPT_REUSE=$REUSE_ON -e TF_EXL3_PROMPT_DET=slots16 \
    -e TF_GLM53_TILES=load:/work/ref/tiles-r$1.json -w /tmp --entrypoint python3 $IMAGE -m tensorfold.cli serve /model \
    --tp 4 --rank $1 --master $MASTER --master-port $PY_MPORT --host 0.0.0.0 --port $PYHTTP --name glm-5.3-tf \
    --context $CONTEXT --parallel $PARALLEL --mtp-drafts $K"
}
py_session() {   # $1 = PY_MODE, $2 = client file prefix (py | py-mtp), $3 = phase name
  PY_MODE=$1
  phase "$3"
  wait_memory || fatal "python server ($1): a node under 100 GB MemAvailable"
  say "python server ($([ "$1" = drafter ] && echo "$DRAFTER ${DSHOW}" || echo mtp)): tensorfold.cli serve --tp 4 --parallel $PARALLEL --context $CONTEXT (TF_GLM53_PROMPT_REUSE=$REUSE_ON, slots16, the reference's tiles)"
  start_ranks zp4c-py py_args || { collect_ranks zp4c-py; fatal "python server start failed"; }
  wait_http zp4c-py $PYHTTP "$LOAD_TIMEOUT" || { collect_ranks zp4c-py; fatal "the python server ($1) did not come up (zp4c-py-r*.log)"; }
  clients "$2" zp4c-py http://127.0.0.1:$PYHTTP
  collect_ranks zp4c-py
  for r in 0 1 2 3; do mv -f "$LOCAL/zp4c-py-r$r.log" "$LOCAL/zp4c-$2-r$r.log" 2>/dev/null; done
  say "python server ($1): memory min (GB) $(mem_min "$3")"
  check_wd
}
if [ "${SKIP_PY:-0}" != 1 ]; then
  case "$DRAFTER" in
    dflash) py_session drafter py python ;;
    mtp) py_session mtp py python ;;
    dspark) say "python server: no DSpark server (engine.py refuses TF_GLM53_DSPARK with --parallel); the MTP bar stands in" ;;
  esac
fi
if [ "$DRAFTER" != mtp ] && [ "$PY_BAR" = 1 ] && [ "${SKIP_BAR:-0}" != 1 ]; then
  py_session mtp py-mtp python-bar
fi

# ---- 8. the Zig server (rank 0 serves, 1..3 follow) and the same client -------------------------------------------
if [ "${SKIP_ZIG:-0}" != 1 ]; then
  phase zig-server
  # ZIG_WRAP0=1: rank 0 under `nsys launch --session-new=zs` (ZIG_HOOK then runs `nsys start/stop --session=zs`)
  zig_entry() {
    if [ "${ZIG_WRAP0:-0}" = 1 ] && [ "$1" = 0 ]; then
      echo "-v /opt/nvidia/nsight-systems:/opt/nsys:ro --entrypoint /opt/nsys/2025.3.2/target-linux-sbsa-armv8/nsys $IMAGE launch --session-new=zs --trace=cuda,nvtx,osrt --cuda-graph-trace=node /work/zig-out/native/bin/tensorfold-native"
    else
      echo "--entrypoint /work/zig-out/native/bin/tensorfold-native $IMAGE"
    fi
  }
  zig_args() {
    echo "-v $MODEL:/model:ro -v $WORK:/work \
      -e TF_GLM53_RANK=$1 -e TF_GLM53_WORLD=4 -e TF_GLM53_MASTER=$MASTER:$ZIG_PORT -e TF_GLM53_AOT=/work/$ZAOT \
      -e TENSORFOLD_CUDA_KERNELS=/work/$ZAOT -e TF_GLM53_NCCL_LIB=${NLIB[$1]} -e TF_GLM53_CUBLAS_LIB=$CUBLAS \
      -e TF_GLM53_TILES=/work/ref/tiles-r$1.json -e TF_GLM53_INV_REF=/work/ref/inv_freq.bin -e TF_GLM53_BMM_PROBE=/work/ref/bmm_probe.json \
      -e TF_GLM53_HCAS=${ZHCAS[$1]} -e TF_GLM53_GID=${ZGID[$1]} -e TF_GLM53_EXPERTS=${EXPERTS:-shared} -e TF_GLM53_PE_DET=slots16 \
      -e TF_GLM53_PROMPT_REUSE=$REUSE_ON -e TF_GLM53_DCP=0 -e TF_GLM53_K=$K -e TF_GLM53_DRAFT_VOCAB=32768 -e TF_GLM53_MTP_REUSE=2 \
      -e TF_GLM53_SIDE=$ZSIDE -e TF_GLM53_L2PF=$ZL2PF -e TF_GLM53_L2PF_MB=$ZL2PF_MB -e TF_GLM53_MULTI_SELECT=$ZMULTI_SELECT \
      -e TF_GLM53_DEVICE_CANDS=$ZDEVICE_CANDS -e TF_GLM53_MTP_DENSE=$ZMTP_DENSE -e TF_GLM53_UNPACK_MB=384 -e TF_GLM53_PROMPT_ROWS=8192 \
      -e TF_GLM53_PROMPT_ROWS_SHORT=4096 -e TF_GLM53_PROMPT_SP=1 -e TF_GLM53_ROCE=1 -e TF_GLM53_TIMEOUT=${ZIG_RDV_TIMEOUT:-5400} \
      -e TF_GLM53_DUMP=/work/gate/dump-zig.jsonl ${ZIG_ENV:-} $DMOUNT $(drafter_env) $([ "$DRAFTER" != mtp ] && echo "-e TF_GLM53_MTP=$DRAFTER") \
      $(zig_entry $1) serve /model --host 0.0.0.0 --port $ZHTTP --name glm-5.3-tf \
      --context $CONTEXT --parallel $PARALLEL --no-update-check"
  }
  rsh "${NODES[0]}" "rm -f $WORK/gate/dump-zig.jsonl"
  wait_memory || fatal "zig server: a node under 100 GB MemAvailable"
  say "zig server: tensorfold-native serve --parallel $PARALLEL at TP4, drafter $DRAFTER, context $CONTEXT, AOT /work/$ZAOT, docker --memory $MEM_CAP"
  start_ranks zp4c-zig zig_args || { collect_ranks zp4c-zig; fatal "zig server start failed"; }
  wait_http zp4c-zig $ZHTTP "$LOAD_TIMEOUT" || { collect_ranks zp4c-zig; fatal "the zig server did not come up (zp4c-zig-r*.log)"; }
  grep -q . <<<"$(rsh "${NODES[1]}" "docker logs zp4c-zig-r1 2>&1 | grep -i 'following rank 0' | head -1")" || say "WARNING: rank 1 never logged 'following rank 0'"
  if [ -n "${ZIG_HOOK:-}" ]; then
    NODE0=${NODES[0]} SRV=zp4c-zig-r0 ZHTTP=$ZHTTP WORK=$WORK LOCAL=$LOCAL RPORT=$RPORT TREE=$TREE IMAGE=$IMAGE bash "$ZIG_HOOK" 2>&1 | tee -a "$LOG"
  else
    clients zig zp4c-zig http://127.0.0.1:$ZHTTP
  fi
  collect_ranks zp4c-zig
  say "zig server: memory min (GB) $(mem_min zig-server)"
  check_wd
fi

# ---- 9. the gates (gate4b.py on rank 0's node, in a CPU container) -----------------------------------------------
phase gate
rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 4g -v $RPORT/$TREE:/tf:ro -v $WORK:/work --entrypoint python3 $IMAGE \
  -B /tf/tools/glm53/gate4b.py gate --work /work/gate --parallel $PARALLEL --ratio $RATIO --drafter $DRAFTER --bar-ratio $BAR_RATIO" > "$LOCAL/gate.log" 2>&1
grc=$?
for f in cli-4b.json py-prose.json py-code.json zig-prose.json zig-code.json py-prose-s.json py-code-s.json zig-prose-s.json zig-code-s.json \
         py-mtp-prose.json py-mtp-code.json py-mtp-prose-s.json py-mtp-code-s.json dump-zig.jsonl; do
  rsh "${NODES[0]}" "cat $WORK/gate/$f" > "$LOCAL/$f" 2>/dev/null || rm -f "$LOCAL/$f"
done
grep -E '^(GATE|REPORT)' "$LOCAL/gate.log" > "$LOCAL/gate.txt"
tee -a "$LOG" < "$LOCAL/gate.log" >/dev/null
cat "$LOCAL/gate.txt"
say "memory min (GB) by phase: cli $(mem_min zig-cli); python $(mem_min python); python bar $(mem_min python-bar); zig $(mem_min zig-server)"
[ -s "$LOCAL/gate.txt" ] || summary FAIL "the gate printed nothing (gate.log)${GATE_FAIL:+;$GATE_FAIL}" 1
gates=$(grep '^GATE' "$LOCAL/gate.txt" | awk '{printf "%s%s=%s", (NR>1?" ":""), $2, $3}')
if [ $grc = 0 ] && [ -z "$GATE_FAIL" ]; then
  summary PASS "$gates" 0
fi
summary FAIL "$gates${GATE_FAIL:+; steps:$GATE_FAIL}" 1
