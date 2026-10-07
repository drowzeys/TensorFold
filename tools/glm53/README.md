# GLM-5.3 Zig port, Phase 1: the Python oracle and the AOT kernel set

Phase 1 gate: GLM-5.3's first 4 layers (dense 0-2, MoE 3) on one GB10 in Zig, bit for bit against the Python
glm_moe_dsa engine (champion source `vcruz-port @ d7a54bb`) on fixed tokens. Everything here runs on a node inside
`ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05` (Triton 3.7.1, torch, CUDA 13), never on .4.

## One command (does every step below)

```sh
rsync -a --delete ~/zig-port/ <node>:zig-port/            # from .4
ssh <node> 'bash ~/zig-port/run_phase1_node.sh'           # prints: PHASE1 PASS|FAIL ... work=<dir>
```

The script refuses to start below 30 GB MemAvailable, while a `tf-glm53` container runs, or while other compute
apps hold the GPU (`ALLOW_GPU_BUSY=1` overrides the last). Defaults: `PYSRC=~/champion-ft8-src`,
`MODEL=/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit`, `ZIG_DIR=~/opt/zig` (node .5), output in
`~/zig-port/runs/<utc>/` (build.log, oracle.log, sass.log, compare.log, aot/, fixtures/).

## The pieces

| file | what |
|---|---|
| `oracle.py` | the Python engine as rank 0 of world 4 (fused path), the first N layers loaded through `weights.RankReader` / `load_layer` at that cut; two chains of windows (below); `--record` and/or `--fixtures` |
| `capture_aot.py` | `oracle.py --record`: Triton AOT capture only |
| `dump_fixtures.py` | `oracle.py --fixtures`: fixtures only |
| `copy_exl3_kernels.py` | regenerates `zig/kernels/cuda/glm53_exl3_{linear,experts}.cu` + `glm53/*.cuh` (device-only copies of `cuda/exl3`); `--check` compares |
| `expected_symbols.txt` | the kernel symbols `cuda_kernels.zig` resolves (the script checks the built fatbins list them) |

**Rank 0 of 4, other ranks absent.** World 1 is not a served shape (all 64 heads on one GPU: `_attn_chunks` needs
152 KiB of shared memory, GB10 has 99 KiB), so Phase 1 runs one rank's serving shapes (16 heads, quarter MLP /
expert widths, replicated q_a / kv_a / indexer) with an identity communicator on both sides: `LocalComm` has only
`all_gather`, so `fused.gather` always takes the all-gather branch; it writes this rank's partial in its slot and
zeros in the other three, and `residual_add` sums the four slots in rank order. Every per-layer output is rank 0's
alone (not the full model's) and must still match bit for bit; the Zig forward (`Forward.gather`) does the same.

Windows (the shapes the Zig side runs, so every specialization they need is captured):

* `short`: 24-row prompt window from an empty state (prompt tiling: one pass, `ATTN_PROMPT` 2048/64/8/2), then a
  1-row and a 3-row decode window (decode tiling 256/32/4/2, 8 chunks + `_merge`). No selection (< index_topk).
* `long`: a 20,000-token prefix in 128-row windows; its caches become the Zig side's starting state; then 1- and
  3-row decode windows at 20,000.. with index bucket 32,768 (the indexer scores and selects). The decode windows
  also run through the radix select (`TF_GLM53_RADIX_MIN_ROWS=1`), which is what the Zig side uses for every
  window; `meta.json: radix_equal` must be true (the champion's torch.topk + sort picks the same keys).

Frozen knobs (`oracle.FROZEN`, recorded in meta.json): `TF_GLM53_TUNE=0` (x3linear.plan tiles), `SIDE=0`,
`EXPERTS=tf`, `FOLD_SHARED=1`, `TF_EXL3_LINEAR_LOADS=35`, `TF_EXL3_EXPERTS_LOADS=1`, `_FUSE=1`, `_PDL=0`, bf16 KV.

## Manual steps (what run_phase1_node.sh does), inside the image

```sh
# oracle: capture + fixtures (GPU). TRITON_CACHE_DIR must be fresh: the cubins are packed from it.
docker run --rm --gpus all --network none --memory 64g --entrypoint python3 \
  -v $MODEL:/model:ro -v ~/champion-ft8-src/src:/opt/tensorfold/src:ro -v ~/zig-port/tf:/tf:ro -v $OUT:/out \
  -e PYTHONPATH=/opt/tensorfold/src -e TRITON_CACHE_DIR=/out/triton -e TORCH_EXTENSIONS_DIR=/out/torch_ext \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 $IMAGE \
  -B /tf/tools/glm53/oracle.py --model /model --out /out --record --fixtures --layers 4 --world 4 --rank 0 --long 20000 --tools /tf
# -> /out/launches.json, jit.json, manifest.json, aot/{aot.json,cubins/,variants.txt}, fixtures/{meta.json,*.safetensors}

# Zig build (CPU; zig 0.17.0 at /opt/z), host tests first
zig build test-glm53
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix $OUT/zig-out fatbins tf-glm53-layers

# compare (GPU)
$OUT/zig-out/bin/tf-glm53-layers $MODEL $OUT/aot $OUT/fixtures --layers 4      # RESULT glm53-layers PASS|FAIL
```

`aot_pack.py` (zig/tests/cuda/nemotron) now records `num_stages` per variant and writes a `None` constexpr (e.g.
`BASE=None`) as `{"none": true}`; `zig/src/cuda/aot.zig` keys variants on `num_warps` / `num_stages` when the call
site pins them (`Set.runWith(..., .{ .num_warps = 8, .num_stages = 2 })`) and refuses an ambiguous pick.
