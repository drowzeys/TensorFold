#!/bin/bash
# GLM-5.3 Zig port, Phase 3b gate: the Zig engine SERVED (tensorfold-native serve on rank 0, ranks 1..3 following)
# with prompt reuse, --learn, 1M context (DCP 4) and the OpenAI routes, on the four GB10 nodes at TP4. Four gates:
#   (a) reuse: a 3-turn conversation (system prompt + tools, turn 1 carries the 32K text: ~33K-token prompts, sampled
#       T 1.0 / top_p 0.95 / top_k 20, fixed seeds) sent warm, then turns 2 and 3 again cold (the flush file makes the
#       server forget every kept state) and turn 3 with "draft": false - warm == cold == serial token for token, warm
#       turns resumed (begin > 0). TTFT (client: first streamed delta; engine: the dump's ttft_s) warm vs cold, and the
#       same conversation against the PYTHON server (TF_GLM53_PROMPT_REUSE=1) side by side (soft target: Zig warm
#       TTFT <= 1.10 x Python's - reported, not gated);
#   (b) --learn: a server with --learn takes turn 1 (the shared system + tools state goes to disk), all four ranks are
#       stopped and started again with --learn, turns 2 and 3 are sent: the first resumes from disk (dump "learned" >
#       0) and both equal (a)'s cold references; TTFT from disk vs cold reported;
#   (c) 1M: a needle at ~830K tokens (the 32K text repeated; chat template, thinking on; greedy) through
#       tf-glm53-generate --mode 3b --context CTX_1M --dcp 0 (auto: DCP 4): the reply names the needle's secret, the
#       nodes keep >= 4 GB available; decode tok/s at depth reported. PY1M=1: the Python engine on the same prompt
#       (reference3b.py, DCP auto) too, and Zig >= RATIO x Python on decode at depth is gated; PY1M=0 (default): gated
#       on the needle and memory only (a short Python run with TF_GLM53_DCP=4 records the DCP Triton kernels);
#   (d) server: OpenAI chat completions non-streaming, streaming (SSE to [DONE]), a tool call (finish_reason
#       tool_calls, parsed arguments), a Hermes-style 3-request multi-turn smoke; then served == CLI: every request
#       session (a) served (the engine's dump: prompt ids, sampling, reply ids) through tf-glm53-generate --mode 3b
#       --reuse 1 --stop-eos 1 - every reply's ids equal.
# Prints one line at the end:  PHASE3B PASS|FAIL ... work=...
#
# Run from .4 (the orchestrator: it only syncs, starts containers over ssh and copies results; builds, servers,
# clients and GPU jobs all run on the nodes). Free the cluster first - this script stops nothing it did not start:
#   bash ~/zig-port/run_phase3b_cluster.sh
# A memory watchdog runs from launch to exit: any node under 4 GB MemAvailable or over 2 GB of swap in use and it
# removes this run's containers (label tensorfold.glm53-phase3b=<stamp>); the run then ends "PHASE3B FAIL watchdog".
#
# Steps: preflight (also refuses tf-glm53* / zp* containers, swap > 2 GB) -> rsync -> prompts (3a's, made on rank 0's
# node) and the conversation -> build (test-glm53, fatbins, tf-glm53-generate, native) -> the Python reference at TP4
# (3a's reference3a.py --record: AOT set, tiles, inv_freq, bmm probe, BLAS; REF_FROM=<node dir> reuses a 3a run's
# ref/) -> AOT coverage (below) -> Zig server A (reuse; conversation + API) -> Zig servers B / C (--learn, a restart
# between) -> CLI served == CLI -> Python server P (TTFT) -> DCP AOT capture (or the Python 1M run) -> AOT coverage at
# 1M (DCP 4) -> Zig 1M -> gate3b.py gate.
#
# AOT coverage (run 2, 2026-10-08: a resumed 41-row window wanted _router_part BM 64, which the reference's schedule
# never launched): tf-glm53-generate --mode list-variants enumerates every Triton launch shape the Zig engine can make
# (cuda_coverage.zig: cuda_triton.Tri in probe mode over windows of 1..8192 rows, every index-range class, DCP 1 and 4
# with every rank) and marks which the captured set lacks; tools/glm53/sweep_aot.py compiles exactly those (warmup, no
# launch) on rank 0's GPU, packs them and merges them with the capture into $WORK/aot-full (aot-1m for the 1M run); a
# second list-variants must report nothing missing (VARIANTS PASS) or no Zig engine starts. Every node gets the set.
#
# Fail-fast: each client run (conversation, API) is watched every 15 s - the client failing, any rank of the server
# exiting, a request error in the server's log (ZERR_PAT: MissingTritonVariant, "request N failed", stream errors,
# panics, CUDA / NCCL errors), no change in the client's and server's logs for PROGRESS_MIN minutes or the gate's
# timeout end the whole run at once ("PHASE3B FAIL <reason>", this run's containers removed; the Python server's
# session only notes it). Rank jobs (references, CLI, 1M) fail the same way on errors / stalls (STALL_LONG_MIN for
# the Python references and the 1M runs).
#
# Environment (defaults in brackets):
#   NODES MODEL IMAGE PORT_DIR PYSRC ZIG_DIR ZIG_FROM MASTER IF RAILS DROP_CACHES ALLOW_GPU_BUSY: as phase 3a
#   MEM_CAP [110g] a rank   MEM_CAP_1M [118g] the 1M runs   CONTEXT32K [$HOME/glm53-speed-20260920/context-32k.txt]
#   CONTEXT [auto: 3a's prompts' need, rounded up to 1024] the reference's and the servers' context
#   K [2]  CONV_TOKENS [300]  CACHE_GIB [4]  REUSE_GAP [1024]  CACHE_ENTRIES [32]  LEARN_GIB [32]
#   CTX_1M [1000000]  NEEDLE_DEPTH [830000]  NEEDLE_TOKENS [256]  PY1M [0]  RATIO [0.97]  ZMTP_DENSE_1M [0]
#   PYCONV [1] (Python server TTFT session)   REF_FROM [] (a node path holding a 3a reference's ref/ directory)
#   Zig decode knobs as 3a: ZSIDE [af] ZL2PF [1] ZL2PF_MB [8] ZMULTI_SELECT [1] ZDEVICE_CANDS [1] ZMTP_DENSE [1]
#   ports: ZHTTP [18890] PYHTTP [18891] REF_PORT [29671] ZIG_PORT [29681] PY_MPORT [29691] CLI_PORT [29701]
#   SKIP_SYNC SKIP_BUILD SKIP_REF SKIP_A SKIP_LEARN SKIP_CLI SKIP_PY SKIP_1M =1 (with REUSE=<stamp>: its work dir)
#   fail-fast: PROGRESS_MIN [15] CONV_TIMEOUT [5400] API_TIMEOUT [2400] STALL_LONG_MIN [120] ZERR_PAT [see below]
#   AOT coverage: SWEEP_TIMEOUT [10800] SWEEP_MEM [32g] PROMPT_ROWS [8192] SKIP_COVER=1 (use $WORK/aot-full as it is,
#   else the reference's ref/aot unchecked)
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
CONTEXT32K=${CONTEXT32K-$HOME/glm53-speed-20260920/context-32k.txt}
CONTEXT=${CONTEXT:-auto}
K=${K:-2}
CONV_TOKENS=${CONV_TOKENS:-300}
CACHE_GIB=${CACHE_GIB:-4}
REUSE_GAP=${REUSE_GAP:-1024}
CACHE_ENTRIES=${CACHE_ENTRIES:-32}
LEARN_GIB=${LEARN_GIB:-32}
CTX_1M=${CTX_1M:-1000000}
NEEDLE_DEPTH=${NEEDLE_DEPTH:-830000}
NEEDLE_TOKENS=${NEEDLE_TOKENS:-256}
PY1M=${PY1M:-0}
PYCONV=${PYCONV:-1}
RATIO=${RATIO:-0.97}
REF_FROM=${REF_FROM:-}
ZSIDE=${ZSIDE:-af}
ZL2PF=${ZL2PF:-1}
ZL2PF_MB=${ZL2PF_MB:-8}
ZMULTI_SELECT=${ZMULTI_SELECT:-1}
ZDEVICE_CANDS=${ZDEVICE_CANDS:-1}
ZMTP_DENSE=${ZMTP_DENSE:-1}
ZMTP_DENSE_1M=${ZMTP_DENSE_1M:-0}
ZHTTP=${ZHTTP:-18890}
PYHTTP=${PYHTTP:-18891}
REF_PORT=${REF_PORT:-29671}
ZIG_PORT=${ZIG_PORT:-29681}
PY_MPORT=${PY_MPORT:-29691}
CLI_PORT=${CLI_PORT:-29701}
IF=${IF:-enp1s0f1np1}
RAILS=${RAILS:-2}
MEM_CAP=${MEM_CAP:-110g}
MEM_CAP_1M=${MEM_CAP_1M:-118g}
DROP_CACHES=${DROP_CACHES:-1}
PROGRESS_MIN=${PROGRESS_MIN:-15}
CONV_TIMEOUT=${CONV_TIMEOUT:-5400}
API_TIMEOUT=${API_TIMEOUT:-2400}
STALL_LONG_MIN=${STALL_LONG_MIN:-120}
SWEEP_TIMEOUT=${SWEEP_TIMEOUT:-10800}
SWEEP_MEM=${SWEEP_MEM:-32g}
PROMPT_ROWS=${PROMPT_ROWS:-8192}
ZERR_PAT=${ZERR_PAT:-'MissingTritonVariant|AmbiguousTritonVariant|NoTritonSet|request [0-9]+ failed|stream error|panic: |Segmentation fault|illegal memory access|CUDA_ERROR_|unhandled cuda error|NCCL WARN .*(failed|error)'}
STAMP=${REUSE:-$(date -u +%Y%m%d-%H%M%S)}
RHOME=${RHOME:-$HOME}
RPORT=$RHOME/zig-port
WORK=$RPORT/runs/p3b-$STAMP                   # on every node
LOCAL=$PORT_DIR/runs/p3b-$STAMP               # on .4: logs and the result files
LABEL="tensorfold.glm53-phase3b=$STAMP"
mkdir -p "$LOCAL"
LOG=$LOCAL/run.log
# wait for this shell's jobs except the memory watchdog (a bare `wait` would wait for it forever: run 1, 2026-10-08)
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
  echo "PHASE3B $1 $2 work=$LOCAL" | tee -a "$LOG"; exit "${3:-1}"
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
say "phase 3b: nodes ${NODES[*]} (rank 0 first), model $MODEL, image $IMAGE, context $CONTEXT, 1M $CTX_1M (PY1M=$PY1M), work $WORK"
bad=0
for r in 0 1 2 3; do n=${NODES[$r]}
  out=$(rsh "$n" "echo avail=\$(awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo)
    echo swap=\$(awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print int((t-f)/1024)}' /proc/meminfo)
    echo tfglm=\$(docker ps --format '{{.Names}} {{.Image}}' | grep -c tf-glm53)
    echo zp=\$(docker ps -a --format '{{.Names}}' | grep -cE '^zp')
    echo apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [ -f '$MODEL/config.json' ] && echo model=ok || echo model=missing
    docker image inspect '$IMAGE' >/dev/null 2>&1 && echo image=ok || echo image=missing
    [ -f ~/NODE_CLAIM.txt ] && sed 's/^/claim: /' ~/NODE_CLAIM.txt; true" 2>&1) || { say "rank $r ($n): ssh failed"; bad=1; continue; }
  say "rank $r ($n): $(grep -v '^claim:' <<<"$out" | tr '\n' ' ')"
  grep '^claim:' <<<"$out" | sed "s/^/  $n /" | tee -a "$LOG"
  a=$(sed -n 's/^avail=//p' <<<"$out"); [ "${a:-0}" -ge 100 ] || { say "  REFUSE $n: MemAvailable ${a} GB < 100 GB"; bad=1; }
  s=$(sed -n 's/^swap=//p' <<<"$out"); [ "${s:-0}" -le 2048 ] || { say "  REFUSE $n: ${s} MB of swap in use (> 2 GB: the watchdog would fire at once; swapoff -a && swapon -a)"; bad=1; }
  [ "$(sed -n 's/^tfglm=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a tf-glm53 container is running"; bad=1; }
  [ "$(sed -n 's/^zp=//p' <<<"$out")" = 0 ] || { say "  REFUSE $n: a zp* container exists (an earlier run's: docker rm -f them)"; bad=1; }
  if [ "$(sed -n 's/^apps=//p' <<<"$out")" != 0 ] && [ "${ALLOW_GPU_BUSY:-0}" != 1 ]; then say "  REFUSE $n: compute apps hold the GPU (ALLOW_GPU_BUSY=1 to start anyway)"; bad=1; fi
  grep -q model=missing <<<"$out" && { say "  REFUSE $n: no $MODEL/config.json"; bad=1; }
  grep -q image=missing <<<"$out" && { say "  REFUSE $n: image not pulled"; bad=1; }
done
[ $bad = 0 ] || summary FAIL "refused by the preflight (run.log)" 3
detect_rails || summary FAIL "RoCE rail detection failed (RAILS=1?)" 3
check_wd

# ---- 1. sync -----------------------------------------------------------------------------------------------------
phase sync
if [ "${SKIP_SYNC:-0}" != 1 ]; then
  [ -f "$PYSRC/src/tensorfold/families/glm_moe_dsa/cuda/fused.py" ] || summary FAIL "no champion source at $PYSRC" 3
  rm -f "$LOCAL/sync.err"
  for n in "${NODES[@]}"; do
    ( rsync -a --delete --exclude /runs/ --exclude /cache/ --exclude /pysrc/ --exclude .zig-cache/ --exclude zig-out/ \
        "$PORT_DIR/" "$n:zig-port/" && rsync -a --delete "$PYSRC/" "$n:zig-port/pysrc/" ) >> "$LOG" 2>&1 \
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
for n in "${NODES[@]}"; do rsh "$n" "mkdir -p $WORK/ref $WORK/learn $WORK/pyflag $WORK/cli $RPORT/cache/zig-local $RPORT/cache/zig-global $RPORT/cache/torch_ext $RPORT/cache/cuda_cache $RPORT/cache/vllm/b12x-compile $RPORT/cache/b12x-roce $RPORT/cache/xdg"; done
USER_FLAG='--user $(id -u):$(id -g)'
COMMON="--label $LABEL $USER_FLAG -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1"
rank_flags() { echo "--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory $1 --memory-swap $1"; }
CACHE_ENV="-e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e CUDA_CACHE_PATH=/cache/cuda_cache -e TORCH_CUDA_ARCH_LIST=12.1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e B12X_COMPILE_CACHE_DIR=/cache/vllm/b12x-compile -e B12X_ROCE_CACHE_DIR=/cache/b12x-roce -e XDG_CACHE_HOME=/cache/xdg"
# a CPU container on rank 0's node: the HTTP clients, prompt makers and gates (host network: the servers on 127.0.0.1)
client() {
  rsh "${NODES[0]}" "docker run --rm $COMMON --network host --memory 8g -v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
    -v $RPORT/tf:/tf:ro -v $WORK:/work -e PYTHONPATH=/opt/tensorfold/src --entrypoint python3 $IMAGE -B /tf/tools/glm53/gate3b.py $*"
}
to_all() { local f=$1 n; rsh "${NODES[0]}" "cat $WORK/$f" > "$LOCAL/$f" || return 1; for n in "${NODES[@]:1}"; do rsh "$n" "cat > $WORK/$f" < "$LOCAL/$f" || return 1; done; }

# ---- 2. prompts: 3a's set (the reference's AOT set and tiles), the conversation ---------------------------------
phase prompts
if [ "${SKIP_REF:-0}" != 1 ] || [ ! -s "$LOCAL/conv.json" ]; then
  [ -n "$CONTEXT32K" ] && [ -f "$CONTEXT32K" ] || summary FAIL "no 32K background text at $CONTEXT32K" 3
  for n in "${NODES[@]}"; do rsh "$n" "cat > $WORK/context-32k.txt" < "$CONTEXT32K" || summary FAIL "could not copy $CONTEXT32K to $n" 3; done
  if [ -z "$REF_FROM" ] && [ "${SKIP_REF:-0}" != 1 ]; then
    say "prompts: 3a's prompt set (reference3a.py --make-prompts) on ${NODES[0]}"
    rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 8g -v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
      -v $RPORT/tf:/tf:ro -v $WORK:/work -e PYTHONPATH=/opt/tensorfold/src --entrypoint python3 $IMAGE \
      -B /tf/tools/glm53/reference3a.py --model /model --make-prompts /work/prompts.json --context-file /work/context-32k.txt" >> "$LOG" 2>&1 \
      || summary FAIL "prompt rendering failed (run.log)" 1
    to_all prompts.json || summary FAIL "could not spread prompts.json" 1
  fi
  client make-conv --context-file /work/context-32k.txt --out /work/conv.json --max-tokens $CONV_TOKENS >> "$LOG" 2>&1 || summary FAIL "make-conv failed" 1
  to_all conv.json
fi

# ---- 3. build on every node (CPU only, memory-capped) ------------------------------------------------------------
phase build
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  say "build: zig build test-glm53, then fatbins + tf-glm53-generate + native (tensorfold-native) on all four nodes (docker --memory 24g)"
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "timeout 3600 docker run --rm $COMMON --network none --memory 24g --memory-swap 24g \
      -v $RPORT/tf:/work/tf -v $WORK:/out -v $RPORT/cache:/cache -v $ZIG_DIR:/opt/z:ro -w /work/tf \
      -e ZIG_LOCAL_CACHE_DIR=/cache/zig-local -e ZIG_GLOBAL_CACHE_DIR=/cache/zig-global -e PATH=/opt/z:/usr/local/cuda/bin:/usr/bin:/bin \
      --entrypoint bash $IMAGE -c 'set -u; rc=0; zig version
        zig build test-glm53 --summary all || rc=10
        zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix /out/zig-out -j8 fatbins tf-glm53-generate native || rc=\$((rc + 20))
        exit \$rc'" > "$LOCAL/build-r$r.log" 2>&1 &
    bpid[$r]=$!
  done
  brc=0; for r in 0 1 2 3; do wait "${bpid[$r]}" || brc=1; done
  for r in 0 1 2 3; do
    rsh "${NODES[$r]}" "test -x $WORK/zig-out/bin/tf-glm53-generate && test -x $WORK/zig-out/native/bin/tensorfold-native" \
      || { say "build failed on ${NODES[$r]}: $(tail -5 "$LOCAL/build-r$r.log" | tr '\n' ' ' | cut -c1-300)"; brc=1; }
  done
  [ $brc = 0 ] || summary FAIL "build failed (build-r*.log: rc 10 host tests, 20 engine/server build)" 1
  say "build: OK on all four nodes"
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

# A watched client run (gate3b.py in a named container on rank 0's node): ends at once on the client failing, a rank of
# the server exiting, a request error in the server's rank 0 log (ZERR_PAT), PROGRESS_MIN minutes without a change in
# the client's or the server's log, or the gate's timeout.
#   $1 = label  $2 = timeout (s)  $3 = server container prefix ("" none)  $4 = 1: fatal (the run ends: PHASE3B FAIL
#   <reason>, containers removed) | 0: return 1 with WATCH_REASON set;  the rest: gate3b.py's arguments
WATCH_REASON=""
client_watch() {
  local label=$1 limit=$2 srv=$3 fatal_on=$4; shift 4
  local cname="zp3b-c-$label" t=0 last="" still=0 h="" now r st errs rc=0 cpid
  rsh "${NODES[0]}" "docker rm -f $cname" >/dev/null 2>&1
  rsh "${NODES[0]}" "docker run --rm --name $cname $COMMON --network host --memory 8g -v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro \
    -v $RPORT/tf:/tf:ro -v $WORK:/work -e PYTHONPATH=/opt/tensorfold/src --entrypoint python3 $IMAGE -B /tf/tools/glm53/gate3b.py $*" \
    > "$LOCAL/client-$label.log" 2>&1 &
  cpid=$!
  WATCH_REASON=""
  while kill -0 "$cpid" 2>/dev/null; do
    sleep 15; t=$((t + 15))
    if watchdog_fired; then WATCH_REASON="watchdog: $(cat "$LOCAL/WATCHDOG")"; break; fi
    if [ -n "$srv" ]; then
      for r in 0 1 2 3; do
        st=$(rsh "${NODES[$r]}" "docker inspect -f '{{.State.Running}} {{.State.ExitCode}}' $srv-r$r" 2>/dev/null)
        case "$st" in "true "*) ;; *) WATCH_REASON="$srv-r$r not running (${st:-gone}) during $label: $(rsh "${NODES[$r]}" "docker logs --tail 3 $srv-r$r 2>&1" | tr '\n' ' ' | cut -c1-300)"; break 2 ;; esac
      done
      errs=$(rsh "${NODES[0]}" "docker logs $srv-r0 2>&1 | grep -E '$ZERR_PAT' | head -3" 2>/dev/null)
      [ -n "$errs" ] && { WATCH_REASON="server error during $label: $(tr '\n' ' ' <<<"$errs" | cut -c1-400)"; break; }
      h=$(rsh "${NODES[0]}" "docker logs --tail 50 $srv-r0 2>&1 | md5sum" 2>/dev/null)
    fi
    now="$h $(md5sum < "$LOCAL/client-$label.log")"
    if [ "$now" = "$last" ]; then still=$((still + 15)); else still=0; last=$now; fi
    (( still >= PROGRESS_MIN * 60 )) && { WATCH_REASON="no progress for $PROGRESS_MIN min during $label (client and server logs unchanged)"; break; }
    (( t >= limit )) && { WATCH_REASON="$label still running after $limit s"; break; }
  done
  if [ -n "$WATCH_REASON" ]; then
    rsh "${NODES[0]}" "docker rm -f $cname" >/dev/null 2>&1
    kill "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
  else
    wait "$cpid"; rc=$?
    if [ -n "$srv" ]; then
      errs=$(rsh "${NODES[0]}" "docker logs $srv-r0 2>&1 | grep -E '$ZERR_PAT' | head -3" 2>/dev/null)
      [ -n "$errs" ] && WATCH_REASON="server error during $label: $(tr '\n' ' ' <<<"$errs" | cut -c1-400)"
    fi
    [ -z "$WATCH_REASON" ] && [ $rc != 0 ] && WATCH_REASON="client $label failed (exit $rc): $(grep -E 'FAILED|Error|error' "$LOCAL/client-$label.log" | tail -3 | tr '\n' ' ' | cut -c1-400)"
  fi
  cat "$LOCAL/client-$label.log" >> "$LOG"
  [ -z "$WATCH_REASON" ] && return 0
  if [ "$fatal_on" = 1 ]; then
    if [ -n "$srv" ]; then
      collect_ranks "$srv"
      rsh "${NODES[0]}" "cat $WORK/dump-$Z_SESSION.jsonl" > "$LOCAL/dump-$Z_SESSION.jsonl" 2>/dev/null
    fi
    fatal "$WATCH_REASON"
  fi
  say "FAILED ($label): $WATCH_REASON"
  return 1
}

# ---- 4. the Python reference (3a's reference3a.py --record: AOT set, tiles, inv_freq, bmm probe, BLAS) ----------
phase ref
if [ -n "$REF_FROM" ]; then
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/ref && cp -a '$REF_FROM' $WORK/ref" || summary FAIL "$n: no $REF_FROM" 3; done
  say "reference: reused $REF_FROM"
elif [ "${SKIP_REF:-0}" != 1 ]; then
  if [ "$CONTEXT" = auto ]; then
    need=$(grep -o '"context_needed": [0-9]*' "$LOCAL/prompts.json" | grep -o '[0-9]*$')
    [ -n "$need" ] || summary FAIL "prompts.json has no context_needed (CONTEXT=N)" 1
    CONTEXT=$(( (need + 1023) / 1024 * 1024 ))
  fi
  ref_main() {
    echo "-v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/tf:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
      -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/ref/triton $CACHE_ENV -w /tmp --entrypoint python3 $IMAGE \
      -B /tf/tools/glm53/reference3a.py --model /model --rank $1 --master $MASTER --port $REF_PORT \
      --prompts /work/prompts.json --context $CONTEXT --k $K --out /work/ref --record"
  }
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/ref && mkdir -p $WORK/ref/triton"; done
  wait_memory || summary FAIL "a node is under 100 GB MemAvailable before the reference" 1
  say "reference: Glm53Engine served configuration at TP4, context $CONTEXT (--record)"
  start_ranks zp3b-ref ref_main || { cleanup; summary FAIL "reference start failed" 1; }
  wait_ranks zp3b-ref "${REF_TIMEOUT:-10800}" "$STALL_LONG_MIN"; rrc=$?
  collect_ranks zp3b-ref
  check_wd
  [ $rrc = 0 ] || fatal "the Python reference failed: $WAIT_REASON (zp3b-ref-r*.log)"
fi
for r in 0 1 2 3; do
  rsh "${NODES[$r]}" "test -f $WORK/ref/aot/aot.json && test -f $WORK/ref/inv_freq.bin && test -s $WORK/ref/nccl.txt && test -s $WORK/ref/tiles-r$r.json && test -s $WORK/ref/bmm_probe.json && test -s $WORK/ref/blas.json" \
    || summary FAIL "${NODES[$r]}: the reference left no aot/, inv_freq.bin, nccl.txt, tiles-r$r.json, bmm_probe.json or blas.json in $WORK/ref" 1
done
rsh "${NODES[0]}" "cat $WORK/ref/blas.json" > "$LOCAL/blas.json"
rsh "${NODES[0]}" "cat $WORK/ref/ref-r0.json" > "$LOCAL/ref-r0.json" 2>/dev/null
if [ "$CONTEXT" = auto ]; then
  CONTEXT=$(grep -o '"context": [0-9]*' "$LOCAL/ref-r0.json" | head -1 | grep -o '[0-9]*$')
  [ -n "$CONTEXT" ] || summary FAIL "the reference's context is unknown (CONTEXT=N)" 1
fi
(( CONTEXT < 200000 )) || summary FAIL "context $CONTEXT would turn DCP on in the servers: keep it under 200,000" 1
declare -a NLIB=()
for r in 0 1 2 3; do NLIB[$r]=$(rsh "${NODES[$r]}" "head -1 $WORK/ref/nccl.txt"); done
CUBLAS=$(jstr libcublas "$LOCAL/blas.json"); EXPERTS=$(jstr experts_impl "$LOCAL/blas.json")
[ -n "$CUBLAS" ] || summary FAIL "the reference's blas.json names no libcublas" 1
say "servers: context $CONTEXT, NCCL ${NLIB[0]}, cuBLAS $CUBLAS, experts ${EXPERTS:-shared}"

# ---- AOT coverage: every launch shape the Zig engine can make has a variant, or nothing Zig starts ------------------
# $1 = label (dir under $WORK), $2 = context, $3 = list-variants --dcp (0: 1 and 4, 1: one rank), $4 = the set the
# needs are checked against (dir under $WORK), $5 = output set (dir under $WORK), rest: template run dirs under $WORK
# (manifest.json + aot/: the captures whose constexprs / floats the new variants take). Fatal on any failure.
listv() {   # $1 = label, $2 = context, $3 = dcp, $4 = aot dir, $5 = needs file (under /work)
  rsh "${NODES[0]}" "docker run --rm $COMMON --gpus all --network none --memory 16g -v $MODEL:/model:ro -v $WORK:/work \
    --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE --mode list-variants --model /model --world 4 \
    --context $2 --prompt-rows $PROMPT_ROWS --window 128 --k $K --dcp $3 \
    --tiles /work/$1/tiles-r0.json,/work/$1/tiles-r1.json,/work/$1/tiles-r2.json,/work/$1/tiles-r3.json \
    --aot /work/$4 --out /work/$5" > "$LOCAL/$1-listv-$(basename "$5" .json).log" 2>&1
}
aot_cover() {
  local lab=$1 ctx=$2 dcp=$3 have=$4 dest=$5; shift 5
  local tmpl="" d r rc
  for d in "$@"; do tmpl="$tmpl --template /work/$d"; done
  phase aot-$lab
  rsh "${NODES[0]}" "rm -rf $WORK/$lab $WORK/$dest && mkdir -p $WORK/$lab/triton" || fatal "aot $lab: no work dir"
  for r in 0 1 2 3; do   # every rank's tile table (the prompt GEMM's shapes) on rank 0's node
    rsh "${NODES[$r]}" "cat $WORK/ref/tiles-r$r.json" | rsh "${NODES[0]}" "cat > $WORK/$lab/tiles-r$r.json" \
      || fatal "aot $lab: could not copy tiles-r$r.json from ${NODES[$r]}"
  done
  listv "$lab" "$ctx" "$dcp" "$have" "$lab/needs.json"; rc=$?
  grep -E '^(variants|VARIANTS|  missing)' "$LOCAL/$lab-listv-needs.log" | head -12 | tee -a "$LOG"
  [ $rc = 0 ] || [ $rc = 3 ] || fatal "aot $lab: list-variants failed (exit $rc, $lab-listv-needs.log)"
  say "aot $lab: compiling the missing variants (sweep_aot.py on ${NODES[0]}, templates:$tmpl)"
  rsh "${NODES[0]}" "timeout $SWEEP_TIMEOUT docker run --rm $COMMON --gpus all --network none --memory $SWEEP_MEM --memory-swap $SWEEP_MEM \
    -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/tf:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
    -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/$lab/triton $CACHE_ENV -w /tmp --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/sweep_aot.py --needs /work/$lab/needs.json $tmpl --out /work/$lab/sweep --merged /work/$dest" \
    > "$LOCAL/$lab-sweep.log" 2>&1 || fatal "aot $lab: sweep_aot.py failed ($lab-sweep.log: $(grep -E 'FAILED|Error' "$LOCAL/$lab-sweep.log" | tail -2 | tr '\n' ' ' | cut -c1-300))"
  grep '^\[sweep\]' "$LOCAL/$lab-sweep.log" | grep -v '/[0-9]* compiled (' | tail -6 | tee -a "$LOG"
  listv "$lab" "$ctx" "$dcp" "$dest" "$lab/check.json"; rc=$?
  grep -E '^(VARIANTS|  missing)' "$LOCAL/$lab-listv-check.log" | head -12 | tee -a "$LOG"
  [ $rc = 0 ] || fatal "aot $lab: the merged set $dest still misses launches the Zig engine can make ($lab-listv-check.log)"
  for r in 1 2 3; do
    rsh "${NODES[0]}" "tar -C $WORK -c $dest" | rsh "${NODES[$r]}" "rm -rf $WORK/$dest && tar -C $WORK -x" \
      || fatal "aot $lab: could not copy $dest to ${NODES[$r]}"
  done
  say "aot $lab: $dest complete for context $ctx (dcp $dcp) and on every node"
}
ZAOT=ref/aot
if [ "${SKIP_COVER:-0}" != 1 ]; then
  aot_cover cover "$CONTEXT" 1 ref/aot aot-full ref
  ZAOT=aot-full
  check_wd
elif rsh "${NODES[0]}" "test -f $WORK/aot-full/aot.json"; then
  ZAOT=aot-full
  say "SKIP_COVER=1: using the earlier $WORK/aot-full unchecked"
else
  say "WARNING: SKIP_COVER=1 and no aot-full: the Zig engine runs on the reference's capture alone (run 2's failure)"
fi

# ---- the Zig server (rank 0 serves, 1..3 follow) -----------------------------------------------------------------
Z_SESSION=A; Z_LEARN=0
zig_env() {   # $1 = rank, $2 = aot dir under /work, $3 = MTP dense
  echo "-e TF_GLM53_RANK=$1 -e TF_GLM53_WORLD=4 -e TF_GLM53_MASTER=$MASTER:$ZIG_PORT -e TF_GLM53_AOT=/work/$2 \
    -e TENSORFOLD_CUDA_KERNELS=/work/$2 -e TF_GLM53_NCCL_LIB=${NLIB[$1]} -e TF_GLM53_CUBLAS_LIB=$CUBLAS \
    -e TF_GLM53_TILES=/work/ref/tiles-r$1.json -e TF_GLM53_INV_REF=/work/ref/inv_freq.bin -e TF_GLM53_BMM_PROBE=/work/ref/bmm_probe.json \
    -e TF_GLM53_HCAS=${ZHCAS[$1]} -e TF_GLM53_GID=${ZGID[$1]} -e TF_GLM53_EXPERTS=${EXPERTS:-shared} -e TF_GLM53_PE_DET=slots16 \
    -e TF_GLM53_PROMPT_REUSE=1 -e TF_GLM53_CACHE_GIB=$CACHE_GIB -e TF_GLM53_REUSE_GAP=$REUSE_GAP -e TF_GLM53_CACHE_ENTRIES=$CACHE_ENTRIES \
    -e TF_GLM53_DCP=0 -e TF_GLM53_FLUSH_FILE=/work/REUSE_FLUSH -e TF_GLM53_K=$K -e TF_GLM53_DRAFT_VOCAB=32768 -e TF_GLM53_MTP_REUSE=2 \
    -e TF_GLM53_SIDE=$ZSIDE -e TF_GLM53_L2PF=$ZL2PF -e TF_GLM53_L2PF_MB=$ZL2PF_MB -e TF_GLM53_MULTI_SELECT=$ZMULTI_SELECT \
    -e TF_GLM53_DEVICE_CANDS=$ZDEVICE_CANDS -e TF_GLM53_MTP_DENSE=$3 -e TF_GLM53_UNPACK_MB=384 -e TF_GLM53_PROMPT_ROWS=8192 \
    -e TF_GLM53_PROMPT_ROWS_SHORT=4096 -e TF_GLM53_PROMPT_SP=1 -e TF_GLM53_ROCE=1 -e TF_GLM53_TIMEOUT=${ZIG_RDV_TIMEOUT:-5400}"
}
zserve_args() {
  local learn=""
  [ "$Z_LEARN" = 1 ] && learn="--learn --learn-dir /learn --learn-gib $LEARN_GIB"
  echo "-v $MODEL:/model:ro -v $WORK:/work -v $WORK/learn:/learn $(zig_env $1 $ZAOT $ZMTP_DENSE) -e TF_GLM53_DUMP=/work/dump-$Z_SESSION.jsonl \
    --entrypoint /work/zig-out/native/bin/tensorfold-native $IMAGE serve /model --host 0.0.0.0 --port $ZHTTP --name glm-5.3-tf \
    --context $CONTEXT --parallel 1 --no-update-check $learn"
}
start_zig() {     # $1 = session letter, $2 = learn 0|1
  Z_SESSION=$1; Z_LEARN=$2
  wait_memory || fatal "zig-$1: a node under 100 GB MemAvailable before start"
  rsh "${NODES[0]}" "rm -f $WORK/dump-$1.jsonl $WORK/REUSE_FLUSH"
  say "zig server $1 (learn $2): tensorfold-native serve at TP4, context $CONTEXT, AOT /work/$ZAOT, docker --memory $MEM_CAP"
  start_ranks zp3b-z$1 zserve_args || { collect_ranks zp3b-z$1; fatal "zig-$1 start failed"; }
  wait_http zp3b-z$1 $ZHTTP "${LOAD_TIMEOUT:-3600}" || { collect_ranks zp3b-z$1; fatal "zig-$1 did not come up (zp3b-z$1-r*.log)"; }
  grep -q . <<<"$(rsh "${NODES[1]}" "docker logs zp3b-z$1-r1 2>&1 | grep -i 'following rank 0' | head -1")" || say "WARNING: rank 1 never logged 'following rank 0'"
}
stop_zig() { collect_ranks "zp3b-z$1"; rsh "${NODES[0]}" "cat $WORK/dump-$1.jsonl" > "$LOCAL/dump-$1.jsonl" 2>/dev/null; check_wd; }
ZURL=http://127.0.0.1:$ZHTTP

# ---- 5. session A: reuse (warm / cold / serial) and the API -------------------------------------------------------
phase zigA
if [ "${SKIP_A:-0}" != 1 ]; then
  if start_zig A 0; then
    client_watch convA "$CONV_TIMEOUT" zp3b-zA 1 conv --url $ZURL --spec /work/conv.json --out /work/convA.json \
      --warm 1,2,3 --cold 2,3 --serial 3 --flush-file /work/REUSE_FLUSH --idle $((PROGRESS_MIN * 60)); check_wd
    client_watch apiA "$API_TIMEOUT" zp3b-zA 1 api --url $ZURL --out /work/apiA.json --idle $((PROGRESS_MIN * 60)); check_wd
    rsh "${NODES[0]}" "cat $WORK/convA.json" > "$LOCAL/convA.json" 2>/dev/null || note_fail "conversation A left no convA.json"
    rsh "${NODES[0]}" "cat $WORK/apiA.json" > "$LOCAL/apiA.json" 2>/dev/null || note_fail "API checks left no apiA.json"
    stop_zig A
  fi
fi

# ---- 6. sessions B and C: --learn, a full restart between ----------------------------------------------------------
phase zigB
if [ "${SKIP_LEARN:-0}" != 1 ] && [ -s "$LOCAL/convA.json" ]; then
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/learn && mkdir -p $WORK/learn"; done
  if start_zig B 1; then
    client_watch convB "$CONV_TIMEOUT" zp3b-zB 1 conv --url $ZURL --spec /work/conv.json --out /work/convB.json --warm 1 \
      --idle $((PROGRESS_MIN * 60))
    stop_zig B
    for r in 0 1 2 3; do say "learned on rank $r: $(rsh "${NODES[$r]}" "find $WORK/learn -type f | wc -l; du -sh $WORK/learn | cut -f1" | paste -sd' ') (files, size)"; done
    phase zigC
    if start_zig C 1; then
      client_watch convC "$CONV_TIMEOUT" zp3b-zC 1 conv --url $ZURL --spec /work/conv.json --out /work/convC.json --warm 2,3 \
        --history /work/convA.json --idle $((PROGRESS_MIN * 60))
      stop_zig C
    fi
  fi
fi

# ---- 7. served == CLI: session A's requests through tf-glm53-generate --mode 3b ------------------------------------
phase cli
if [ "${SKIP_CLI:-0}" != 1 ] && [ -s "$LOCAL/dump-A.jsonl" ]; then
  if client dump2cli --dump /work/dump-A.jsonl --out /work/cli-prompts.json >> "$LOG" 2>&1 && to_all cli-prompts.json; then
    cli_args() {
      echo "-v $MODEL:/model:ro -v $WORK:/work -e TF_GLM53_REUSE_GAP=$REUSE_GAP --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE \
        --mode 3b --rank $1 --world 4 --master $MASTER:$CLI_PORT --model /model --aot /work/$ZAOT \
        --prompts /work/cli-prompts.json --context $CONTEXT --k $K --prewarm 1 --graphs 1 --fast-load 1 \
        --draft-vocab 32768 --mtp-reuse 2 --inv-ref /work/ref/inv_freq.bin --nccl-lib ${NLIB[$1]} \
        --roce 1 --hcas ${ZHCAS[$1]} --gid ${ZGID[$1]} --roce-health 1 --tiles /work/ref/tiles-r$1.json \
        --prompt-rows 8192 --prompt-rows-short 4096 --prompt-sp 1 --experts ${EXPERTS:-shared} --pe-det slots16 \
        --cublas-lib $CUBLAS --unpack-mb 384 --bmm-probe /work/ref/bmm_probe.json \
        --side $ZSIDE --l2pf $ZL2PF --l2pf-mb $ZL2PF_MB --multi-select $ZMULTI_SELECT --device-cands $ZDEVICE_CANDS \
        --mtp-dense $ZMTP_DENSE --stop-eos 1 --reuse 1 --dcp 0 \
        --timeout ${ZIG_RDV_TIMEOUT:-5400} --out /work/cli/gen-r$1.json"
    }
    wait_memory || fatal "cli: a node under 100 GB MemAvailable"
    say "cli: tf-glm53-generate --mode 3b --reuse 1 --stop-eos 1 over session A's $(grep -o '"name": "req' "$LOCAL/cli-prompts.json" | wc -l) requests"
    start_ranks zp3b-cli cli_args || { collect_ranks zp3b-cli; fatal "cli start failed"; }
    wait_ranks zp3b-cli "${GEN_TIMEOUT:-10800}" || { collect_ranks zp3b-cli; fatal "cli run failed: $WAIT_REASON (zp3b-cli-r*.log)"; }
    collect_ranks zp3b-cli
    grep -h -E '^(RESULT|WARN|error)' "$LOCAL"/zp3b-cli-r*.log | tail -20 | tee -a "$LOG"
    check_wd
  else
    note_fail "dump2cli: no finished requests in session A's dump"
  fi
fi

# ---- 8. session P: the Python server, the same conversation (TTFT side by side) ----------------------------------
phase python
if [ "${SKIP_PY:-0}" != 1 ] && [ "$PYCONV" = 1 ] && [ -s "$LOCAL/convA.json" ]; then
  py_args() {
    echo "-v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $WORK:/work -v $RPORT/cache:/cache \
      -e PYTHONPATH=/opt/tensorfold/src $CACHE_ENV -e TF_GLM53_PROMPT_REUSE=1 -e TF_GLM53_CACHE_GIB=$CACHE_GIB \
      -e TF_GLM53_REUSE_GAP=$REUSE_GAP -e TF_GLM53_CACHE_ENTRIES=$CACHE_ENTRIES -e TF_EXL3_PROMPT_DET=slots16 \
      -e TF_GLM53_COPY_DRAFTS=0 -e TF_GLM53_PROFILE_FLAG=/work/pyflag/PROFILE -e TF_GLM53_TILES=load:/work/ref/tiles-r$1.json \
      -w /tmp --entrypoint python3 $IMAGE -m tensorfold.cli serve /model --tp 4 --rank $1 --master $MASTER \
      --master-port $PY_MPORT --host 0.0.0.0 --port $PYHTTP --name glm-5.3-tf --context $CONTEXT --parallel 1"
  }
  if wait_memory; then
    say "python server: tensorfold.cli serve --tp 4 (TF_GLM53_PROMPT_REUSE=1, slots16, copy drafts off), context $CONTEXT"
    if start_ranks zp3b-py py_args && wait_http zp3b-py $PYHTTP "${LOAD_TIMEOUT:-3600}"; then
      client_watch convP "$CONV_TIMEOUT" zp3b-py 0 conv --url http://127.0.0.1:$PYHTTP --spec /work/conv.json --out /work/convP.json \
        --warm 1,2,3 --cold 2,3 --flush-file /work/pyflag/REUSE_FLUSH --history /work/convA.json --idle $((PROGRESS_MIN * 60)) \
        || say "python session: $WATCH_REASON (TTFT side by side incomplete)"
    else
      say "python server did not come up (zp3b-py-r*.log): no Python TTFT"
    fi
    collect_ranks zp3b-py
    check_wd
  else
    say "python server skipped: a node under 100 GB MemAvailable"
  fi
fi

# ---- 9. the DCP kernels: the Python 1M run (PY1M=1) or a short TF_GLM53_DCP=4 capture ------------------------------
phase dcp-aot
AOT1M=""
if [ "${SKIP_1M:-0}" != 1 ]; then
  client make-needle --model /model --context-file /work/context-32k.txt --out /work/prompts-1m.json \
    --depth $NEEDLE_DEPTH --tokens $NEEDLE_TOKENS >> "$LOG" 2>&1 && to_all prompts-1m.json || note_fail "make-needle failed"
  if [ "$PY1M" = 1 ]; then
    REFD=ref1m; RPROMPTS=prompts-1m.json; RCTX=$CTX_1M; RDCP=-1; RCAP=$MEM_CAP_1M
  else
    client make-dcp --model /model --context-file /work/context-32k.txt --out /work/prompts-dcp.json >> "$LOG" 2>&1 && to_all prompts-dcp.json || note_fail "make-dcp failed"
    REFD=refdcp; RPROMPTS=prompts-dcp.json; RCTX=${DCP_CAPTURE_CTX:-32768}; RDCP=4; RCAP=$MEM_CAP
  fi
  ref3b_args() {
    echo "-v $MODEL:/model:ro -v $RPORT/pysrc/src:/opt/tensorfold/src:ro -v $RPORT/tf:/tf:ro -v $WORK:/work -v $RPORT/cache:/cache \
      -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/work/$REFD/triton $CACHE_ENV -w /tmp --entrypoint python3 $IMAGE \
      -B /tf/tools/glm53/reference3b.py --model /model --rank $1 --master $MASTER --port $REF_PORT \
      --prompts /work/$RPROMPTS --context $RCTX --k $K --dcp $RDCP --out /work/$REFD --record"
  }
  for n in "${NODES[@]}"; do rsh "$n" "rm -rf $WORK/$REFD && mkdir -p $WORK/$REFD/triton"; done
  if wait_memory; then
    phase py-$REFD
    say "python $REFD: reference3b.py at context $RCTX, DCP $RDCP (-1: auto), --record (docker --memory $RCAP)"
    if start_ranks zp3b-$REFD ref3b_args "$RCAP" && wait_ranks zp3b-$REFD "${REF1M_TIMEOUT:-14400}" "$STALL_LONG_MIN"; then
      if rsh "${NODES[0]}" "grep -q _attn_dcp $WORK/$REFD/aot/aot.json && grep -q _dcp_combine $WORK/$REFD/aot/aot.json"; then
        AOT1M=$REFD/aot
      else
        note_fail "$REFD: the AOT set lists no _attn_dcp / _dcp_combine (the run did not use DCP)"
      fi
    else
      note_fail "python $REFD failed: $WAIT_REASON (zp3b-$REFD-r*.log)"
    fi
    collect_ranks zp3b-$REFD
    grep -h '\[ref3b\] rank 0' "$LOCAL"/zp3b-$REFD-r0.log | tail -6 | tee -a "$LOG"
    rsh "${NODES[0]}" "cat $WORK/$REFD/ref-r0.json" > "$LOCAL/$REFD-r0.json" 2>/dev/null
    check_wd
  else
    note_fail "$REFD: a node under 100 GB MemAvailable"
  fi
fi

# ---- 9b. AOT coverage at 1M: DCP 1 and 4, every rank, the DCP capture as a template ---------------------------------
if [ "${SKIP_1M:-0}" != 1 ] && [ -n "$AOT1M" ] && [ "${SKIP_COVER:-0}" != 1 ]; then
  # the needs are checked against everything captured or compiled so far (merged first, no GPU)
  rsh "${NODES[0]}" "docker run --rm $COMMON --network none --memory 8g -v $RPORT/tf:/tf:ro -v $WORK:/work --entrypoint python3 $IMAGE \
    -B /tf/tools/glm53/sweep_aot.py --merge-only --set /work/$ZAOT --set /work/$AOT1M --merged /work/aot-pre1m" >> "$LOG" 2>&1 \
    || fatal "aot cover1m: could not merge $ZAOT and $AOT1M"
  tdirs="ref $REFD"
  rsh "${NODES[0]}" "test -f $WORK/cover/sweep/manifest.json" && tdirs="$tdirs cover/sweep"
  aot_cover cover1m "$CTX_1M" 0 aot-pre1m aot-1m $tdirs
  AOT1M=aot-1m
  check_wd
fi

# ---- 10. Zig at 1M: the needle through tf-glm53-generate --mode 3b, DCP auto ------------------------------------------
phase zig1m
if [ "${SKIP_1M:-0}" != 1 ] && [ -n "$AOT1M" ]; then
  z1m_args() {
    echo "-v $MODEL:/model:ro -v $WORK:/work --entrypoint /work/zig-out/bin/tf-glm53-generate $IMAGE \
      --mode 3b --rank $1 --world 4 --master $MASTER:$CLI_PORT --model /model --aot /work/$AOT1M \
      --prompts /work/prompts-1m.json --context $CTX_1M --k $K --prewarm 1 --graphs 1 --fast-load 1 \
      --draft-vocab 32768 --mtp-reuse 2 --inv-ref /work/ref/inv_freq.bin --nccl-lib ${NLIB[$1]} \
      --roce 1 --hcas ${ZHCAS[$1]} --gid ${ZGID[$1]} --roce-health 1 --tiles /work/ref/tiles-r$1.json \
      --prompt-rows 8192 --prompt-rows-short 4096 --prompt-sp 1 --experts ${EXPERTS:-shared} --pe-det slots16 \
      --cublas-lib $CUBLAS --unpack-mb 384 --bmm-probe /work/ref/bmm_probe.json \
      --side $ZSIDE --l2pf $ZL2PF --l2pf-mb $ZL2PF_MB --multi-select $ZMULTI_SELECT --device-cands $ZDEVICE_CANDS \
      --mtp-dense $ZMTP_DENSE_1M --stop-eos 1 --reuse 0 --dcp 0 \
      --timeout ${ZIG_RDV_TIMEOUT:-5400} --out /work/gen1m-r$1.json"
  }
  if wait_memory; then
    say "zig 1M: needle prompt at context $CTX_1M (DCP auto), docker --memory $MEM_CAP_1M"
    start_ranks zp3b-z1m z1m_args "$MEM_CAP_1M" || { collect_ranks zp3b-z1m; fatal "zig 1M start failed"; }
    wait_ranks zp3b-z1m "${GEN1M_TIMEOUT:-14400}" "$STALL_LONG_MIN" || { collect_ranks zp3b-z1m; fatal "zig 1M run failed: $WAIT_REASON (zp3b-z1m-r*.log)"; }
    collect_ranks zp3b-z1m
    grep -h -E '^(RESULT|WARN|error|rank 0 )' "$LOCAL"/zp3b-z1m-r*.log | tail -20 | tee -a "$LOG"
    rsh "${NODES[0]}" "cat $WORK/gen1m-r0.json" > "$LOCAL/gen1m-r0.json" 2>/dev/null
    check_wd
  else
    note_fail "zig 1M: a node under 100 GB MemAvailable"
  fi
elif [ "${SKIP_1M:-0}" != 1 ]; then
  note_fail "zig 1M not run: no AOT set with the DCP kernels"
fi
MEM1M=$(mem_min zig1m)
say "min MemAvailable during the Zig 1M run (GB): ${MEM1M:-none}"

# ---- 11. the gates -------------------------------------------------------------------------------------------------
phase gate
rsh "${NODES[0]}" "cat $WORK/cli/gen-r0.json" > "$LOCAL/cli-gen-r0.json" 2>/dev/null
client gate --work /work --model /model --py1m $PY1M --ratio $RATIO ${MEM1M:+--mem1m $MEM1M} > "$LOCAL/gate.log" 2>&1
grc=$?
grep -E '^(GATE|REPORT)' "$LOCAL/gate.log" > "$LOCAL/gate.txt"
tee -a "$LOG" < "$LOCAL/gate.log" >/dev/null
cat "$LOCAL/gate.txt"
[ -s "$LOCAL/gate.txt" ] || summary FAIL "the gate printed nothing (gate.log)${GATE_FAIL:+;$GATE_FAIL}" 1
gates=$(grep '^GATE' "$LOCAL/gate.txt" | awk '{printf "%s%s=%s", (NR>1?" ":""), $2, $3}')
if [ $grc = 0 ] && [ -z "$GATE_FAIL" ]; then
  summary PASS "$gates$([ "$PY1M" = 1 ] || echo ' (1M: needle + memory only, PY1M=0)')" 0
fi
summary FAIL "$gates${GATE_FAIL:+; steps:$GATE_FAIL}" 1
