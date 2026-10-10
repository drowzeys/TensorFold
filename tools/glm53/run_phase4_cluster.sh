#!/bin/bash
# GLM-5.3 Zig port, Phase 4 gate: the speculative drafters - DSpark, DFlash2 and copy (prompt-lookup) drafts - in the
# Zig engine against the Python engine, on the four GB10 nodes at TP4, every reference in this same run. Sessions
# (each: the Python engine's reference4.py, then the Zig engine's tf-glm53-generate --mode 4, same prompts, same
# tiles / inv_freq / libraries; sampled T 1.0 / top_p 0.95 / top_k 20, seed 1729, thinking on, a fixed token count):
#   ds   DSpark alone (--mtp-drafts 0, TF_GLM53_DSPARK=$DSPARK, policy confidence 0.3 - the published recipe);
#   df   DFlash2 alone (--mtp-drafts 0, TF_GLM53_DFLASH=$DFLASH; depth 7, confidence 0.3);
#   mc   (MTPC=1) the MTP head (k 2) with copy drafts - the served default's path;
#   priv  (PRIV_DFLASH=<path>, private, off by default) the priv DFlash2 at short / 32K / ~120K against the champion's numbers.
# Runs a prompt (tfbench's prose_beekeeper and code_parser, short and with the 32K background): "s20:<d>+c" (drafts
# with copies), "s20:<d>" (drafts alone), "s20:0" (serial), and on the short ones "g:<d>+c" / "g:0" (greedy).
# Gates (tools/glm53/gate4.py): Zig == Python token for token on every run; drafted == serial in each engine; tokens a
# round within TPR_TOL of Python's on the 32K drafted runs; decode tok/s >= RATIO x Python's on the 32K sampled
# drafted runs with copies, prose and code. Prints one line at the end:  PHASE4 PASS|FAIL ... work=...
#
# Run from .4 (the orchestrator: it only syncs, starts containers over ssh and copies results; builds, references and
# GPU jobs all run on the nodes). Free the cluster first - this script stops nothing it did not start:
#   bash ~/zig-port/run_phase4_cluster.sh
# A memory watchdog runs from launch to exit: any node under 4 GB MemAvailable or over 2 GB of swap in use and it
# removes this run's containers (label tensorfold.glm53-phase4=<stamp>); the run then ends "PHASE4 FAIL watchdog".
#
# Steps: preflight (refuses tf-glm53* / zp* containers, swap > 2 GB, a missing drafter) -> rsync -> prompts
# (reference4.py --make-prompts on rank 0's node, one file a session) -> build (test-glm53, fatbins,
# tf-glm53-generate; the kernel symbols) -> the Python references (reference4.py --record a session: its tiles, AOT
# capture of the drafters' kernels) -> AOT coverage (list-variants with the drafters, sweep_aot.py over every
# reference's capture, nothing missing or no Zig engine starts) -> the Zig sessions -> gate4.py.
# Fail-fast: rank jobs end the run on an error in rank 0's log (ZERR_PAT), any rank exiting non-zero, no output for
# STALL_MIN minutes or the job's timeout ("PHASE4 FAIL <reason>", this run's containers removed).
#
# Environment (defaults in brackets):
#   NODES [10.100.10.1 .2 .3 .5] MODEL IMAGE PORT_DIR PYSRC ZIG_DIR ZIG_FROM MASTER IF RAILS DROP_CACHES ALLOW_GPU_BUSY
#   DSPARK [/mnt/spark2-models-local/GLM-5.3-speculator.dspark-keys-ft2]  DFLASH [/mnt/spark2-models-local/GLM-5.3-DFlash2]
#   PRIV_120K [1] (0: the priv session at short + 32K only, context 33792)  PRIV_DFLASH [] (a private DFlash2 checkpoint: the priv session; never printed into a recipe)  MTPC [1]  WITH_120K [0]
#   SESSIONS [ds,df + mc when MTPC=1 + priv when PRIV_DFLASH is set]  TOKENS [512]  CONTEXT32K [~/glm53-speed-20260920/context-32k.txt]
#   CONTEXT [auto: the prompts' need rounded up to 1024; < 200,000]  RATIO [0.97]  TPR_TOL [0.02]  MEM_CAP [110g]
#   DSPARK_POLICY [confidence] DSPARK_CONFIDENCE [0.3] DFLASH_DEPTH [7] DFLASH_CONFIDENCE [0.3] COPY_MIN [8] COPY_MAX [15]
#   Zig decode knobs as 3b: ZSIDE [af] ZL2PF [1] ZL2PF_MB [8] ZMULTI_SELECT [1] ZDEVICE_CANDS [1] ZMTP_DENSE [1]
#   ports: REF_PORT [29771] GEN_PORT [29781]   SKIP_SYNC SKIP_BUILD SKIP_REF SKIP_COVER SKIP_ZIG =1 (REUSE=<stamp>)
#   fail-fast: STALL_MIN [60] REF_TIMEOUT [10800] GEN_TIMEOUT [10800] SWEEP_TIMEOUT [10800] SWEEP_MEM [32g] ZERR_PAT
set -u
NODES=(${NODES:-10.100.10.1 10.100.10.2 10.100.10.3 10.100.10.5})
[ ${#NODES[@]} -eq 4 ] || { echo "NODES must list four fabric addresses, rank 0 first"; exit 2; }
MASTER=${MASTER:-${NODES[0]}}
MODEL=${MODEL:-/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit}
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05}
PORT_DIR=${PORT_DIR:-$HOME/zig-port}
PYSRC=${PYSRC:-$HOME/tf-wt/dspark-deep}
ZIG_DIR=${ZIG_DIR:-$HOME/opt/zig}
ZIG_FROM=${ZIG_FROM:-10.100.10.5}
CONTEXT32K=${CONTEXT32K-$HOME/glm53-speed-20260920/context-32k.txt}
CONTEXT=${CONTEXT:-auto}
DSPARK=${DSPARK:-/mnt/spark2-models-local/GLM-5.3-speculator.dspark-keys-ft2}
DFLASH=${DFLASH:-/mnt/spark2-models-local/GLM-5.3-DFlash2}
PRIV_DFLASH=${PRIV_DFLASH:-}
MTPC=${MTPC:-1}
WITH_120K=${WITH_120K:-0}
TOKENS=${TOKENS:-512}
RATIO=${RATIO:-0.97}
TPR_TOL=${TPR_TOL:-0.02}
DSPARK_POLICY=${DSPARK_POLICY:-confidence}
DSPARK_CONFIDENCE=${DSPARK_CONFIDENCE:-0.3}
DFLASH_DEPTH=${DFLASH_DEPTH:-7}
DFLASH_CONFIDENCE=${DFLASH_CONFIDENCE:-0.3}
COPY_MIN=${COPY_MIN:-8}
COPY_MAX=${COPY_MAX:-15}
ZSIDE=${ZSIDE:-af}
ZL2PF=${ZL2PF:-1}
ZL2PF_MB=${ZL2PF_MB:-8}
ZMULTI_SELECT=${ZMULTI_SELECT:-1}
ZDEVICE_CANDS=${ZDEVICE_CANDS:-1}
ZMTP_DENSE=${ZMTP_DENSE:-1}
REF_PORT=${REF_PORT:-29771}
GEN_PORT=${GEN_PORT:-29781}
IF=${IF:-enp1s0f1np1}
RAILS=${RAILS:-2}
MEM_CAP=${MEM_CAP:-110g}
DROP_CACHES=${DROP_CACHES:-1}
STALL_MIN=${STALL_MIN:-60}
PROGRESS_MIN=$STALL_MIN
SWEEP_TIMEOUT=${SWEEP_TIMEOUT:-10800}
SWEEP_MEM=${SWEEP_MEM:-32g}
PROMPT_ROWS=${PROMPT_ROWS:-8192}
ZERR_PAT=${ZERR_PAT:-'MissingTritonVariant|AmbiguousTritonVariant|NoTritonSet|NoDrafter|NoDraftHead|panic: |Segmentation fault|illegal memory access|CUDA_ERROR_|unhandled cuda error|NCCL WARN .*(failed|error)|Traceback \(most recent'}
if [ -z "${SESSIONS:-}" ]; then
  SESSIONS=ds,df
  [ "$MTPC" = 1 ] && SESSIONS=$SESSIONS,mc
  [ -n "$PRIV_DFLASH" ] && SESSIONS=$SESSIONS,priv
fi
IFS=, read -r -a SESS <<<"$SESSIONS"
STAMP=${REUSE:-$(date -u +%Y%m%d-%H%M%S)}
RHOME=${RHOME:-$HOME}
RPORT=$RHOME/zig-port
WORK=$RPORT/runs/p4-$STAMP                    # on every node
LOCAL=$PORT_DIR/runs/p4-$STAMP                # on .4: logs and the result files
LABEL="tensorfold.glm53-phase4=$STAMP"
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
  echo "PHASE4 $1 $2 work=$LOCAL" | tee -a "$LOG"; exit "${3:-1}"
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
say "phase 4: nodes ${NODES[*]} (rank 0 first), model $MODEL, image $IMAGE, sessions $SESSIONS, work $WORK"
say "drafters: DSpark $DSPARK, DFlash2 $DFLASH$([ -n "$PRIV_DFLASH" ] && echo ', priv (private) set')"
bad=0
for r in 0 1 2 3; do n=${NODES[$r]}
  out=$(rsh "$n" "echo avail=\$(awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo)
    echo swap=\$(awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print int((t-f)/1024)}' /proc/meminfo)
    echo tfglm=\$(docker ps --format '{{.Names}} {{.Image}}' | grep -c tf-glm53)
    echo zp=\$(docker ps -a --format '{{.Names}}' | grep -cE '^zp')
    echo apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ -f '$MODEL/config.json' ] && echo model=ok || echo model=missing
    [ -f '$DSPARK/config.json' ] && echo dspark=ok || echo dspark=missing
    [ -f '$DFLASH/config.json' ] && echo dflash=ok || echo dflash=missing
    [ -z '$PRIV_DFLASH' ] || { [ -f '$PRIV_DFLASH/config.json' ] && echo priv=ok || echo priv=missing; }
    docker image inspect '$IMAGE' >/dev/null 2>&1 && echo image=ok || echo image=missing
    [ -f ~/NODE_CLAIM.txt ] && sed 's/^/claim: /' ~/NODE_CLAIM.txt; true" 2>&1) || { say "rank $r ($n): ssh failed"; bad=1; continue; }
  say "rank $r ($n): $(grep -v '^claim:' <<<"$out" | tr '\n' ' ')"
  grep '^claim:' <<<"$out" | sed "s/^/  $n /" | tee -a "$LOG"
  a=$(sed -n 's/^avail=//p' <<<"$out"); [ "${a:-0}" -ge 100 ] || { say "  REFUSE $n: MemAvailable ${a} GB < 100 GB"; bad=1; }
  s=$(sed -n 's/^swap=//p' <<<"$out"); [ "${s:-0}" -le 2048 ] || { say "  REFUSE $n: ${s} MB of swap in use (> 2 GB)"; bad=1; }
  [ "$(sed -n 's/^tfglm=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a tf-glm53 container is running"; bad=1; }
  [ "$(sed -n 's/^zp=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a zp* container exists (an earlier run's: docker rm -f them)"; bad=1; }
  if [ "$(sed -n 's/^apps=//p' <<<"$out")" != 0 ] && [ "${ALLOW_GPU_BUSY:-0}" != 1 ]; then say "  REFUSE $n: compute apps hold the GPU (ALLOW_GPU_BUSY=1 to start anyway)"; bad=1; fi
  grep -q model=missing <<<"$out" && { say "  REFUSE $n: no $MODEL/config.json"; bad=1; }
  grep -q dspark=missing <<<"$out" && [[ ",$SESSIONS," == *",ds,"* ]] && { say "  REFUSE $n: no DSpark drafter at $DSPARK"; bad=1; }
  grep -q dflash=missing <<<"$out" && [[ ",$SESSIONS," == *",df,"* ]] && { say "  REFUSE $n: no DFlash2 drafter at $DFLASH"; bad=1; }
  grep -q priv=missing <<<"$out" && { say "  REFUSE $n: PRIV_DFLASH is set but has no config.json on this node"; bad=1; }
  grep -q image=missing <<<"$out" && { say "  REFUSE $n: image not pulled"; bad=1; }
done
[ $bad = 0 ] || summary FAIL "refused by the preflight (run.log)" 3
detect_rails || summary FAIL "RoCE rail detection failed (RAILS=1?)" 3
check_wd

# ---- 1. sync -----------------------------------------------------------------------------------------------------
phase sync
if [ "${SKIP_SYNC:-0}" != 1 ]; then
  [ -f "$PYSRC/src/tensorfold/families/glm_moe_dsa/cuda/dspark.py" ] || summary FAIL "no Python source with DSpark at $PYSRC" 3
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
for n in "${NODES[@]}"; do rsh "$n" "mkdir -p $WORK $RPORT/cache/zig-local $RPORT/cache/zig-global $RPORT/cache/torch_ext $RPORT/cache/cuda_cache $RPORT/cache/vllm/b12x-compile $RPORT/cache/b12x-roce $RPORT/cache/xdg"; done
USER_FLAG='--user $(id -u):$(id -g)'
COMMON="--label $LABEL $USER_FLAG -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1"
rank_flags() { echo "--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory $1 --memory-swap $1"; }
CACHE_ENV="-e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e CUDA_CACHE_PATH=/cache/cuda_cache -e TORCH_CUDA_ARCH_LIST=12.1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e B12X_COMPILE_CACHE_DIR=/cache/vllm/b12x-compile -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce -e XDG_CACHE_HOME=/cache/xdg"
DMOUNTS=""   # only the drafters the sessions load (docker would create a missing host path)
[[ ",$SESSIONS," == *",ds,"* ]] && DMOUNTS="$DMOUNTS -v $DSPARK:$DSPARK:ro"
[[ ",$SESSIONS," == *",df,"* ]] && DMOUNTS="$DMOUNTS -v $DFLASH:$DFLASH:ro"
[[ ",$SESSIONS," == *",priv,"* ]] && DMOUNTS="$DMOUNTS -v $PRIV_DFLASH:$PRIV_DFLASH:ro"
to_all() { local f=$1 n; rsh "${NODES[0]}" "cat $WORK/$f" > "$LOCAL/$f" || return 1; for n in "${NODES[@]:1}"; do rsh "$n" "cat > $WORK/$f" < "$LOCAL/$f" || return 1; done; }

# ---- 2. prompts: one file a session (reference4.py --make-prompts, chat template with thinking on) ---------------
phase prompts
session_runs() {   # $1 = session -> "short|32k|120k" run lists and the drafts label
  case "$1" in
    ds)  echo "g:dspark+c,g:0,s20:dspark+c,s20:dspark,s20:0|s20:dspark+c,s20:dspark,s20:0|s20:dspark+c" ;;
    df)  echo "g:dflash+c,g:0,s20:dflash+c,s20:dflash,s20:0|s20:dflash+c,s20:dflash,s20:0|s20:dflash+c" ;;
    mc)  echo "g:2+c,g:0,s20:2+c,s20:2,s20:0|s20:2+c,s20:2,s20:0|s20:2+c" ;;
    priv) echo "s20:dflash+c,s20:dflash,s20:0|s20:dflash+c,s20:dflash,s20:0|s20:dflash+c" ;;
  esac
}
if [ "${SKIP_REF:-0}" != 1 ] || [ ! -s "$LOCAL/prompts-${SESS[0]}.json" ]; then
  [ -n "$CONTEXT32K" ] && [ -f "$CONTEXT32K" ] || summary FAIL "no 32K background text at $CONTEXT32K" 3
  for n in "${NODES[@]}"; do rsh "$n" "cat > $WORK/context-32k.txt" < "$CONTEXT32K" || summary FAIL "could not copy $CONTEXT32K to $n" 3; done
  for s in "${SESS[@]}"; do
    IFS='|' read -r rs r32 r120 <<<"$(session_runs "$s")"
    w120=""; { [ "$WITH_120K" = 1 ] || { [ "$s" = priv ] && [ "${PRIV_120K:-1}" = 1 ]; }; } && w120="--with-120k --runs-120k $r120"
    rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 8g -v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
      -v $RPORT/tf:/tf:ro -v $WORK:/work -e PYTHONPATH=/opt/tensorfold/src --entrypoint python3 $IMAGE \
      -B /tf/tools/glm53/reference4.py --model /model --make-prompts /work/prompts-$s.json --context-file /work/context-32k.txt \
      --tokens $TOKENS --runs-short $rs --runs-32k $r32 $w120" >> "$LOG" 2>&1 || summary FAIL "prompt rendering for $s failed (run.log)" 1
    to_all "prompts-$s.json" || summary FAIL "could not spread prompts-$s.json" 1
  done
fi
if [ "$CONTEXT" = auto ]; then
  need=0
  for s in "${SESS[@]}"; do
    v=$(grep -o '"context_needed": [0-9]*' "$LOCAL/prompts-$s.json" | grep -o '[0-9]*$'); (( v > need )) && need=$v
  done
  CONTEXT=$(( (need + 1023) / 1024 * 1024 ))
fi
(( CONTEXT < 200000 )) || summary FAIL "context $CONTEXT would turn DCP on: keep it under 200,000 (WITH_120K prompts need ~131K)" 1
say "context $CONTEXT for every session"

# ---- 3. build on every node (CPU only, memory-capped) ------------------------------------------------------------
phase build
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  say "build: zig build test-glm53, then fatbins + tf-glm53-generate on all four nodes (docker --memory 24g)"
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "timeout 3600 docker run --rm $COMMON --network none --memory 24g --memory-swap 24g \
      -v $RPORT/tf:/work/tf -v $WORK:/out -v $RPORT/cache:/cache -v $ZIG_DIR:/opt/z:ro -w /work/tf \
      -e ZIG_LOCAL_CACHE_DIR=/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=/cache/zig-global -e PATH=/opt/z:/usr/local/cuda/bin:/usr/bin:/bin \
      --entrypoint bash $IMAGE -c 'set -u; rc=0; zig version
        zig build test-glm53 --summary all || rc=10
        zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix /out/zig-out -j8 fatbins tf-glm53-generate || rc=\$((rc + 20))
        for f in /out/zig-out/fatbin/glm53_*.fatbin /out/zig-out/fatbin/qmm_group.fatbin; do [ -f \"\$f\" ] && cuobjdump -symbols \"\$f\"; done > /out/symbols-glm53.txt 2>&1
        exit \$rc'" > "$LOCAL/build-r$r.log" 2>&1 &
    bpid[$r]=$!
  done
  brc=0; for r in 0 1 2 3; do wait "${bpid[$r]}" || brc=1; done
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "test -x $WORK/zig-out/bin/tf-glm53-generate" \
      || { say "build failed on ${NODES[$r]}: $(tail -5 "$LOCAL/build-r$r.log" | tr '\n' ' ' | cut -c1-300)"; brc=1; }
  done
  [ $brc = 0 ] || summary FAIL "build failed (build-r*.log: rc 10 host tests, 20 engine build)" 1
  rsh "${NODES[0]}" "cat $WORK/symbols-glm53.txt" > "$LOCAL/symbols-glm53.txt"
  missing=0
  while read -r sym; do
    case "$sym" in "#"*|"") continue ;; esac
    grep -q -- "$sym" "$LOCAL/symbols-glm53.txt" || { say "kernel symbol missing from the fatbins: $sym"; missing=$((missing + 1)); }
  done < "$PORT_DIR/tf/tools/glm53/expected_symbols.txt"
  [ "$missing" = 0 ] || summary FAIL "build: $missing kernel symbols missing (symbols-glm53.txt)" 1
  say "build: OK on all four nodes, every expected kernel symbol present"
fi
check_wd

# ---- helpers (as 3a, watchdog-aware) -----------------------------------------------------------------------------
start_ranks() {   # $1 = name prefix, $2 = function printing rank r's docker arguments after the flags, $3 = memory cap
  local r
  for r in 3 2 1 0; do
    rsh "${NODES[$r]}" "docker run -d --name $1-r$r $COMMON $(rank_flags "${3:-$MEM_CAP}") $(nccl_env $r) $($2 $r)" >> "$LOG" 2>&1 \
      || { say "could not start $1-r$r on ${NODES[$r]}"; return 1; }
  done
}
drop_caches() { [ "$DROP_CACHES" = 1 ] || return 0; local n; for n in "${NODES[@]}"; do rsh "$n" 'sync; sudo -n sysctl -q -w vm.drop_caches=1' >/dev/null 2>&1 & done; waitjobs; }
WAIT_REASON=""
wait_ranks() {    # $1 = name prefix, $2 = timeout (s), $3 = stall minutes [PROGRESS_MIN]; 0 when all four exited 0
  local t=0 r st running failed errs h last="" still=0 stall=${3:-$PROGRESS_MIN}
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
    drop_caches
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

# what a session loads: the Python reference's flags, the Zig engine's flags, MTP drafts
sess_k() { [ "$1" = mc ] && echo 2 || echo 0; }
py_drafter() {
  case "$1" in
    ds) echo "--dspark $DSPARK --dspark-policy $DSPARK_POLICY --dspark-confidence $DSPARK_CONFIDENCE" ;;
    df) echo "--dflash $DFLASH --dflash-depth $DFLASH_DEPTH --dflash-confidence $DFLASH_CONFIDENCE" ;;
    priv) echo "--dflash $PRIV_DFLASH --dflash-depth $DFLASH_DEPTH --dflash-confidence $DFLASH_CONFIDENCE" ;;
    *) echo "" ;;
  esac
}
zig_drafter() {
  case "$1" in
    ds) echo "--dspark $DSPARK --dspark-policy $DSPARK_POLICY --dspark-confidence $DSPARK_CONFIDENCE" ;;
    df) echo "--dflash $DFLASH --dflash-depth $DFLASH_DEPTH --dflash-confidence $DFLASH_CONFIDENCE" ;;
    priv) echo "--dflash $PRIV_DFLASH --dflash-depth $DFLASH_DEPTH --dflash-confidence $DFLASH_CONFIDENCE" ;;
    *) echo "" ;;
  esac
}
# ---- 4. the Python references, one a session (reference4.py --record: tiles, inv_freq, libraries, AOT capture) ---
phase ref
if [ "${SKIP_REF:-0}" != 1 ]; then
  for s in "${SESS[@]}"; do
    REF_S=$s
    ref_args() {
      echo "-v $MODEL:/model:ro $DMOUNTS -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/tf:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
        -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/ref-$REF_S/triton $CACHE_ENV -w /tmp --entrypoint python3 $IMAGE \
        -B /tf/tools/glm53/reference4.py --model /model --rank $1 --master $MASTER --port $REF_PORT \
        --prompts /work/prompts-$REF_S.json --context $CONTEXT --k $(sess_k $REF_S) $(py_drafter $REF_S) \
        --copy-min $COPY_MIN --copy-max $COPY_MAX --out /work/ref-$REF_S --record"
    }
    for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/ref-$s && mkdir -p $WORK/ref-$s/triton"; done
    wait_memory || fatal "a node is under 100 GB MemAvailable before the $s reference"
    phase "ref-$s"
    say "python $s: reference4.py at TP4, context $CONTEXT, k $(sess_k $s) $([ "$s" = priv ] && echo '(priv, private)' || py_drafter $s)"
    start_ranks zp4-ref-$s ref_args || { cleanup; summary FAIL "the $s reference did not start" 1; }
    wait_ranks zp4-ref-$s "${REF_TIMEOUT:-10800}" "$STALL_MIN"; rrc=$?
    collect_ranks zp4-ref-$s
    grep -h '^\[ref4\] rank 0' "$LOCAL/zp4-ref-$s-r0.log" | tail -20 | tee -a "$LOG"
    check_wd
    [ $rrc = 0 ] || fatal "the Python $s reference failed: $WAIT_REASON (zp4-ref-$s-r*.log)"
    for r in 0 1 2 3; do
      rsh "${NODES[$r]}" "test -f $WORK/ref-$s/ref-r$r.json && test -f $WORK/ref-$s/tiles-r$r.json && test -s $WORK/ref-$s/nccl.txt && test -f $WORK/ref-$s/inv_freq.bin" \
        || fatal "${NODES[$r]}: the $s reference left no ref-r$r.json / tiles / nccl.txt / inv_freq.bin"
    done
    rsh "${NODES[0]}" "test -f $WORK/ref-$s/aot/aot.json" || fatal "the $s reference packed no AOT set"
  done
fi

# ---- 5. AOT coverage: every launch the Zig engine can make with these drafters, or nothing Zig starts -----------
listv() {   # $1 = label dir, $2 = aot dir, $3 = needs file (all under /work)
  rsh "${NODES[0]}" "docker run --rm $COMMON --gpus all --network none --memory 16g -v $MODEL:/model:ro $DMOUNTS -v $WORK:/work \
    --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE --mode list-variants --model /model --world 4 \
    --context $CONTEXT --prompt-rows $PROMPT_ROWS --window 128 --k $KMAX --dcp 1 \
    --tiles /work/$1/tiles-r0.json,/work/$1/tiles-r1.json,/work/$1/tiles-r2.json,/work/$1/tiles-r3.json \
    $LV_DRAFTERS --aot /work/$2 --out /work/$3" > "$LOCAL/$1-listv-$(basename "$3" .json).log" 2>&1
}
ZAOT=""
KMAX=0; for s in "${SESS[@]}"; do (( $(sess_k $s) > KMAX )) && KMAX=$(sess_k $s); done
LV_DRAFTERS=""
[[ ",$SESSIONS," == *",ds,"* ]] && LV_DRAFTERS="$LV_DRAFTERS --dspark $DSPARK"
if [[ ",$SESSIONS," == *",df,"* ]]; then LV_DRAFTERS="$LV_DRAFTERS --dflash $DFLASH"
elif [ -n "$PRIV_DFLASH" ]; then LV_DRAFTERS="$LV_DRAFTERS --dflash $PRIV_DFLASH"; fi
if [ "${SKIP_COVER:-0}" != 1 ]; then
  phase aot-cover
  sets=""; tmpl=""
  for s in "${SESS[@]}"; do sets="$sets --set /work/ref-$s/aot"; tmpl="$tmpl --template /work/ref-$s"; done
  rsh "${NODES[0]}" "rm -rf $WORK/cover $WORK/aot-pre $WORK/aot-full && mkdir -p $WORK/cover/triton" || fatal "aot: no work dir"
  for r in 0 1 2 3; do   # every rank's tile table (the prompt GEMM's shapes) on rank 0's node
    rsh "${NODES[$r]}" "cat $WORK/ref-${SESS[0]}/tiles-r$r.json" | rsh "${NODES[0]}" "cat > $WORK/cover/tiles-r$r.json" \
      || fatal "aot: could not copy tiles-r$r.json from ${NODES[$r]}"
  done
  rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 8g -v $RPORT/tf:/tf:ro -v $WORK:/work --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/sweep_aot.py --merge-only $sets --merged /work/aot-pre" >> "$LOG" 2>&1 || fatal "aot: could not merge the references' sets"
  listv cover aot-pre cover/needs.json; rc=$?
  grep -E '^(variants|VARIANTS|  missing)' "$LOCAL/cover-listv-needs.log" | head -12 | tee -a "$LOG"
  [ $rc = 0 ] || [ $rc = 3 ] || fatal "aot: list-variants failed (exit $rc, cover-listv-needs.log)"
  say "aot: compiling the missing variants (sweep_aot.py on ${NODES[0]}, templates:$tmpl)"
  rsh "${NODES[0]}" "timeout $SWEEP_TIMEOUT docker run --rm $COMMON --gpus all --network none --memory $SWEEP_MEM --memory-swap $SWEEP_MEM \
    -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/tf:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
    -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/cover/triton $CACHE_ENV -w /tmp --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/sweep_aot.py --needs /work/cover/needs.json $tmpl --out /work/cover/sweep --merged /work/aot-full" \
    > "$LOCAL/cover-sweep.log" 2>&1 || fatal "aot: sweep_aot.py failed (cover-sweep.log: $(grep -E 'FAILED|Error' "$LOCAL/cover-sweep.log" | tail -2 | tr '\n' ' ' | cut -c1-300))"
  grep '^\[sweep\]' "$LOCAL/cover-sweep.log" | grep -v '/[0-9]* compiled (' | tail -6 | tee -a "$LOG"
  listv cover aot-full cover/check.json; rc=$?
  grep -E '^(VARIANTS|  missing)' "$LOCAL/cover-listv-check.log" | head -12 | tee -a "$LOG"
  [ $rc = 0 ] || fatal "aot: the merged set aot-full still misses launches the Zig engine can make (cover-listv-check.log)"
  for r in 1 2 3; do
    rsh "${NODES[0]}" "tar -C $WORK -c aot-full" | rsh "${NODES[$r]}" "rm -rf $WORK/aot-full && tar -C $WORK -x" \
      || fatal "aot: could not copy aot-full to ${NODES[$r]}"
  done
  ZAOT=aot-full
  say "aot: aot-full complete for context $CONTEXT and the drafters, on every node"
  check_wd
elif rsh "${NODES[0]}" "test -f $WORK/aot-full/aot.json"; then
  ZAOT=aot-full; say "SKIP_COVER=1: using the earlier $WORK/aot-full unchecked"
else
  ZAOT=ref-${SESS[0]}/aot; say "WARNING: SKIP_COVER=1 and no aot-full: the Zig engine runs on the ${SESS[0]} capture alone"
fi

# ---- 6. the Zig sessions: tf-glm53-generate --mode 4 over each session's prompts ---------------------------------
if [ "${SKIP_ZIG:-0}" != 1 ]; then
  for s in "${SESS[@]}"; do
    phase "zig-$s"
    rsh "${NODES[0]}" "cat $WORK/ref-$s/blas.json" > "$LOCAL/blas-$s.json" 2>/dev/null
    CUBLAS=$(jstr libcublas "$LOCAL/blas-$s.json"); EXPERTS=$(jstr experts_impl "$LOCAL/blas-$s.json")
    [ -n "$CUBLAS" ] || fatal "the $s reference's blas.json names no libcublas"
    declare -a NLIB=()
    for r in 0 1 2 3; do NLIB[$r]=$(rsh "${NODES[$r]}" "head -1 $WORK/ref-$s/nccl.txt"); done
    GEN_S=$s
    gen_args() {
      echo "-v $MODEL:/model:ro $DMOUNTS -v $WORK:/work --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE \
        --mode 4 --rank $1 --world 4 --master $MASTER:$GEN_PORT --model /model --aot /work/$ZAOT \
        --prompts /work/prompts-$GEN_S.json --context $CONTEXT --k $(sess_k $GEN_S) $(zig_drafter $GEN_S) \
        --copy 1 --copy-min $COPY_MIN --copy-max $COPY_MAX --prewarm 1 --graphs 1 --fast-load 1 \
        --draft-vocab 32768 --mtp-reuse 2 --inv-ref /work/ref-$GEN_S/inv_freq.bin --nccl-lib ${NLIB[$1]} \
        --roce 1 --hcas ${ZHCAS[$1]} --gid ${ZGID[$1]} --roce-health 1 --tiles /work/ref-$GEN_S/tiles-r$1.json \
        --prompt-rows 8192 --prompt-rows-short 4096 --prompt-sp 1 --experts ${EXPERTS:-shared} --pe-det slots16 \
        --cublas-lib $CUBLAS --unpack-mb 384 --bmm-probe /work/ref-$GEN_S/bmm_probe.json \
        --side $ZSIDE --l2pf $ZL2PF --l2pf-mb $ZL2PF_MB --multi-select $ZMULTI_SELECT --device-cands $ZDEVICE_CANDS \
        --mtp-dense $ZMTP_DENSE --stop-eos 0 --reuse 0 --dcp 1 \
        --timeout ${ZIG_RDV_TIMEOUT:-5400} --out /work/gen-$GEN_S/gen-r$1.json"
    }
    for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/gen-$s && mkdir -p $WORK/gen-$s"; done
    wait_memory || fatal "zig $s: a node under 100 GB MemAvailable"
    say "zig $s: tf-glm53-generate --mode 4 at TP4, context $CONTEXT, k $(sess_k $s) $([ "$s" = priv ] && echo '(priv, private)' || zig_drafter $s), AOT /work/$ZAOT"
    start_ranks zp4-zig-$s gen_args || { collect_ranks zp4-zig-$s; fatal "zig $s start failed"; }
    wait_ranks zp4-zig-$s "${GEN_TIMEOUT:-10800}" "$STALL_MIN" || { collect_ranks zp4-zig-$s; fatal "zig $s failed: $WAIT_REASON (zp4-zig-$s-r*.log)"; }
    collect_ranks zp4-zig-$s
    grep -h -E '^(RESULT|rank 0 )' "$LOCAL"/zp4-zig-$s-r*.log | tail -24 | tee -a "$LOG"
    check_wd
  done
fi

# ---- 7. the gates (gate4.py on rank 0's node, in a CPU container, over every rank's result files) ----------------
phase gate
rsh "${NODES[0]}" "rm -rf $WORK/gate && mkdir -p $WORK/gate"
for s in "${SESS[@]}"; do
  mkdir -p "$LOCAL/ref-$s" "$LOCAL/gen-$s"
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "cat $WORK/ref-$s/ref-r$r.json" > "$LOCAL/ref-$s/ref-r$r.json" 2>/dev/null || rm -f "$LOCAL/ref-$s/ref-r$r.json"
    rsh "${NODES[$r]}" "cat $WORK/gen-$s/gen-r$r.json" > "$LOCAL/gen-$s/gen-r$r.json" 2>/dev/null || rm -f "$LOCAL/gen-$s/gen-r$r.json"
  done
  tar -C "$LOCAL" -c "ref-$s" "gen-$s" | rsh "${NODES[0]}" "tar -C $WORK/gate -x" || note_fail "could not copy the $s results to ${NODES[0]}"
done
rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 4g -v $RPORT/tf:/tf:ro -v $WORK:/work --entrypoint python3 $IMAGE \
  -B /tf/tools/glm53/gate4.py gate --work /work/gate --sessions $SESSIONS --ratio $RATIO --tpr-tol $TPR_TOL" > "$LOCAL/gate.log" 2>&1
grc=$?
grep -E '^(GATE|REPORT)' "$LOCAL/gate.log" > "$LOCAL/gate.txt"
tee -a "$LOG" < "$LOCAL/gate.log" >/dev/null
cat "$LOCAL/gate.txt"
[ -s "$LOCAL/gate.txt" ] || summary FAIL "the gate printed nothing (gate.log)${GATE_FAIL:+;$GATE_FAIL}" 1
gates=$(grep '^GATE' "$LOCAL/gate.txt" | awk '{printf "%s%s=%s", (NR>1?" ":""), $2, $3}')
if [ $grc = 0 ] && [ -z "$GATE_FAIL" ]; then
  summary PASS "$gates" 0
fi
summary FAIL "$gates${GATE_FAIL:+; steps:$GATE_FAIL}" 1
