#!/bin/bash
# Serve Nemotron 3.5 Bolt 74B-A9B on Metal through tensorfold-native: OpenAI-compatible API, LiLiCorr drafts,
# thinking off by default. Settings: MODEL, PORT (8093), NAME, CONTEXT (65536), PARALLEL (1), DRAFTS=0 for one row a step.
cd "$(dirname "$0")"
MODEL="${MODEL:-$HOME/models/bolt-tf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB}"
SERVER="${SERVER:-../../zig-out/native/bin/tensorfold-native}"
export TENSORFOLD_BOLT_LIB="${TENSORFOLD_BOLT_LIB:-$PWD/build/libbolt.dylib}"
EXTRA=()
[ "${DRAFTS:-1}" = 0 ] && EXTRA+=(--no-drafts)
exec "$SERVER" serve "$MODEL" --name "${NAME:-nvidia-nemotron-3.5-bolt}" --host "${HOST:-0.0.0.0}" --port "${PORT:-8093}" \
    --no-thinking --context "${CONTEXT:-65536}" --parallel "${PARALLEL:-1}" "${EXTRA[@]}" "$@"
