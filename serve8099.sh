#!/bin/zsh
# serve8099.sh [ENV=VAL...]: Cosmos3-Edge reasoner on :8099 (name cosmos3), pid in ~/tf-src/srv8099.pid
cd ~/tf-src
: ${COSMOS_EXTRA=--keep-warm 0}  # beside Bolt and Drama: no idle GPU ticks (voice turns stay at their no-Cosmos latency)
[ -f srv8099.pid ] && kill $(cat srv8099.pid) 2>/dev/null && sleep 1
env "$@" zig-out-cosmos/native/bin/tensorfold-native serve ~/models/cosmos3-tf --name cosmos3 --host 127.0.0.1 --port 8099 --no-update-check --vision --context 16384 --parallel 2 ${=COSMOS_EXTRA} > srv8099.log 2>&1 &
P=$!; echo $P > srv8099.pid
for i in {1..240}; do curl -sf localhost:8099/v1/models >/dev/null && break; kill -0 $P 2>/dev/null || { echo EXITED; tail -20 srv8099.log; exit 1; }; sleep 1; done
echo "ready pid $P after ${i}s"; grep -iE "cosmos3|warn|error" srv8099.log | head
