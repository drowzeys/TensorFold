#!/usr/bin/env bash
# Run one GPU command on the head node for the Drama 3 port: serialized with the oracle by ~/drama3/gpu.lock, under a
# MemAvailable kill-guard (the command's PID is killed when MemAvailable < GUARD_GB, default 8). Prints the lowest
# MemAvailable seen.   usage: run_guarded.sh <cmd> [args...]
set -u
GUARD_GB=${GUARD_GB:-8}
exec 9>"$HOME/drama3/gpu.lock"
echo "[guard] waiting for gpu.lock ..." >&2
flock 9
echo "[guard] got gpu.lock; MemAvailable $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GB" >&2
"$@" 9>&- &
PID=$!
MIN=999999999
while kill -0 $PID 2>/dev/null; do
  A=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  [ "$A" -lt "$MIN" ] && MIN=$A
  if [ "$A" -lt $((GUARD_GB * 1024 * 1024)) ]; then
    echo "[guard] MemAvailable ${A} kB < ${GUARD_GB} GB: killing $PID" >&2
    kill -9 $PID
    break
  fi
  sleep 0.25
done
wait $PID
RC=$?
echo "[guard] exit $RC; lowest MemAvailable $((MIN / 1024)) MB" >&2
exit $RC
