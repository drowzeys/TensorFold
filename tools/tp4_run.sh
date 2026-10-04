#!/bin/bash
# Full GLM-5.3 on four DGX Sparks with TensorFold (run from the orchestrating machine).
#   tools/tp4_run.sh comm            NCCL fabric check (tools/tp4_comm_check.py) on all four ranks
#   tools/tp4_run.sh serve [ARGS]    tensorfold serve --tp 4 (rank 0 serves HTTP on :$PORT); extra ARGS go to serve
#   tools/tp4_run.sh rails           print each rank's RoCE rails and NCCL env (read-only; RAILS=1|2)
#   tools/tp4_run.sh stop            remove the containers
#   tools/tp4_run.sh logs [RANK]     follow a rank's log
set -u
# Configure (env): NODES = the four Sparks' fabric addresses, rank 0 first (it serves HTTP); IMAGE = a container with
# CUDA 13 + torch (+ optionally cuda-exl3 and vLLM for the shared expert layout's prompt GEMM); CKPT = the EXL3
# checkpoint directory, at the same path on every node (e.g. an NFS export). IF: the netdev of NCCL's bootstrap socket.
# Rails: a DGX Spark's QSFP port is two PCIe x4 RoCE devices ("twins": rocep1s0f1 on enp1s0f1np1 and roceP2p1s0f1 on
# enP2p1s0f1np1, ~112 Gb/s each). RAILS=2 (default) finds, on each node, every up RoCE netdev with an IPv4 address and a
# RoCE v2 GID for it, orders them by subnet (the subnet of NODES[r] first) so rail i is one subnet on every node (node
# .2 has swapped its PCIe enumeration across reboots: devices are matched by subnet, never by name), and passes them to
# NCCL_IB_HCA and the RoCE reductions (TF_GLM53_ROCE_HCA / TF_GLM53_ROCE_GIDS) with each device's GID. RAILS=1: the
# rail of NODES[r] only. HCA (and GIDS, one index a rank) set: that one device, no detection (the old behaviour).
: "${NODES:?set NODES to the fabric addresses of the four nodes, rank 0 first}"
: "${IMAGE:?set IMAGE to a CUDA 13 + torch container image}"
: "${CKPT:?set CKPT to the GLM-5.3 EXL3 checkpoint directory (same path on every node)}"
NODES=($NODES)
MASTER=${MASTER:-${NODES[0]}}
PORT=${PORT:-8890}
SRC=${SRC:-$HOME/tf-glm53}
IF=${IF:-enp1s0f1np1}
RAILS=${RAILS:-2}
NAME=tf-glm53

# rail_scan (on a node, read-only): "<subnet> <rdma device> <RoCE v2 GID index> <ibverbs order> <netdev>" a line for
# every up RoCE netdev with an IPv4 address whose GID table holds that address as RoCE v2 (::ffff:a.b.c.d)
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

# Per rank: HCAS[r] (devices, rail order), RGIDS[r] (their GID indices), NGID[r] (one index for NCCL, or empty)
declare -a HCAS=() RGIDS=() NGID=()
XNIC=0
detect_rails() {
  if [ -n "${HCA:-}" ]; then                     # one named device (old behaviour)
    local g=(${GIDS:-3 3 3 3})
    for r in 0 1 2 3; do HCAS[$r]=$HCA; RGIDS[$r]=${g[$r]}; NGID[$r]=${g[$r]}; done
    return 0
  fi
  local ref="" r lines own nets
  for r in 0 1 2 3; do
    lines=$(ssh "${NODES[$r]}" "$rail_scan" | sort) || return 1
    # the subnet of NODES[r] first (the rail NCCL's bootstrap address sits on), then the others by subnet
    own=$(awk -v ip="${NODES[$r]}" '{split($1, c, "[./]"); split(ip, d, "."); if (c[1]==d[1] && c[2]==d[2] && c[3]==d[3]) print}' <<<"$lines")
    lines=$( { echo "$own"; grep -vxF "$own" <<<"$lines"; } | grep . | head -n "$RAILS")
    [ -n "$lines" ] || { echo "no RoCE v2 rail found on ${NODES[$r]}"; return 1; }
    nets=$(awk '{print $1}' <<<"$lines" | paste -sd' ')
    [ -z "$ref" ] && ref=$nets
    [ "$nets" = "$ref" ] || { echo "rank $r rails on [$nets], rank 0 on [$ref]: subnets differ, fix addressing or RAILS=1"; return 1; }
    HCAS[$r]=$(awk '{print $2}' <<<"$lines" | paste -sd,)
    RGIDS[$r]=$(awk '{print $3}' <<<"$lines" | paste -sd,)
    NGID[$r]=$(awk '{print $3}' <<<"$lines" | sort -u | awk 'END{if (NR==1) print}')
    # NCCL numbers devices in ibverbs order and, with NCCL_CROSS_NIC=0, joins device i to device i: if that order is
    # not the subnet order on some node (a swapped enumeration), let NCCL cross NICs by subnet instead
    [ "$(awk '{print $4}' <<<"$lines" | paste -sd' ')" = "$(awk '{print $4}' <<<"$lines" | sort -n | paste -sd' ')" ] || XNIC=1
    echo "rank $r (${NODES[$r]}): rails ${HCAS[$r]} on [$nets], GIDs ${RGIDS[$r]}"
  done
  (( XNIC )) && echo "warning: ibverbs order differs from subnet order on some node: NCCL_CROSS_NIC=1 + NCCL_IB_SUBNET_AWARE_ROUTING=1"
  return 0
}

nccl_env() {   # rank -> docker -e arguments for NCCL and the RoCE reductions
  local r=$1
  local e="-e NCCL_SOCKET_IFNAME=$IF -e NCCL_IB_HCA=${HCAS[$r]} -e TF_GLM53_ROCE_HCA=${HCAS[$r]} -e TF_GLM53_ROCE_GIDS=${RGIDS[$r]}"
  if [ -n "${NGID[$r]}" ]; then e="$e -e NCCL_IB_GID_INDEX=${NGID[$r]}"
  else e="$e -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET"; fi   # GIDs differ: NCCL finds each
  if (( XNIC )); then e="$e -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1"; else e="$e -e NCCL_CROSS_NIC=0"; fi
  echo "$e"
}

sync_src() {
  for n in "${NODES[@]}"; do
    rsync -a --delete --exclude .git --exclude '.torch_ext*' --exclude __pycache__ "$SRC/" "$n:tf-glm53/" &
  done
  wait
}

run_rank() {   # rank node cmd
  local r=$1 n=$2 cmd=$3
  ssh "$n" "docker rm -f $NAME-r$r >/dev/null 2>&1; docker run -d --name $NAME-r$r --gpus all --network host --ipc=host \
    --device=/dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
    -v \$HOME/tf-glm53:/tf -v $(dirname "$CKPT"):$(dirname "$CKPT"):ro -w /tf \
    -e PYTHONPATH=/tf/src -e PYTHONUNBUFFERED=1 -e TORCH_EXTENSIONS_DIR=/tf/.torch_ext \
    $(nccl_env $r) -e NCCL_DEBUG=WARN -e NCCL_PROTO=${NCCL_PROTO:-Simple} -e NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-2} ${DOCKER_ENV:-} \
    -e TENSORFOLD_NO_UPDATE_CHECK=1 -e HF_HUB_OFFLINE=1 \
    --entrypoint bash $IMAGE -c '
      P=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include; X=/tmp/xinc; mkdir -p \$X
      for f in \$P/*.h; do b=\$(basename \$f); [ -e /usr/local/cuda/include/\$b ] || ln -s \$f \$X/\$b; done
      export CPATH=\$X NVCC_APPEND_FLAGS=-I\$X
      $cmd'" >/dev/null && echo "rank $r started on $n"
}

case "${1:-}" in
  comm)
    detect_rails || exit 1
    sync_src
    for r in 3 2 1 0; do
      run_rank $r "${NODES[$r]}" "python3 tools/${CHECK:-tp4_comm_check.py} --rank $r --master $MASTER"
    done
    ssh "${NODES[0]}" "docker wait $NAME-r0 >/dev/null; docker logs $NAME-r0 2>&1 | grep -E 'tp4-comm|roce-big|overlap\]|Error|error' | tail -12"
    ;;
  serve)
    shift
    detect_rails || exit 1
    sync_src
    for r in 3 2 1 0; do
      run_rank $r "${NODES[$r]}" "python3 -m tensorfold.cli serve $CKPT --tp 4 --rank $r --master $MASTER \
        --port $PORT --host 0.0.0.0 --name glm-5.3-tf $*"
    done
    ;;
  rails)
    detect_rails || exit 1
    for r in 0 1 2 3; do echo "rank $r: $(nccl_env $r)"; done ;;
  stop)
    for r in 0 1 2 3; do ssh "${NODES[$r]}" "docker rm -f $NAME-r$r >/dev/null 2>&1"; done; echo stopped ;;
  logs)
    r=${2:-0}; ssh "${NODES[$r]}" "docker logs -f $NAME-r$r" ;;
  *) sed -n 2,7p "$0"; exit 1 ;;
esac
