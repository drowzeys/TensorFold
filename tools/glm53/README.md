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
apps hold the GPU (`ALLOW_GPU_BUSY=1` overrides the last). `PYSRC` (the Python engine source tree) is required,
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
  -v $MODEL:/model:ro -v $PYSRC/src:/opt/tensorfold/src:ro -v ~/zig-port/tf:/tf:ro -v $OUT:/out \
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

# Phase 2a: full GLM-5.3 at TP4 over NCCL, greedy, token parity with the Python engine

Gate: all 78 layers + final norm + the vocabulary-sharded argmax head on the four GB10 nodes (.1 .2 .3 .5 = rank
0..3, RoCE), greedy without drafts, the SAME tokens as the Python engine for fixed prompts (3 chat prompts + one
~4,500-token prompt past index_topk, chat template with thinking on, 128 picks each) and, per pick, the same gathered
head records (each rank's max logit and id, fp32 bits).

## One command, from .4 (free the cluster first; it stops nothing it did not start)

```sh
bash ~/zig-port/run_phase2a_cluster.sh        # last line: PHASE2A PASS|FAIL ... work=~/zig-port/runs/p2a-<utc>
```

It refuses to start while any node has < 100 GB MemAvailable, runs a tf-glm53* container, has GPU compute apps
(`ALLOW_GPU_BUSY=1` overrides) or holds a leftover `zp2a-*` container. Then: rsync ~/zig-port (keeping the nodes'
runs/ and cache/) and the champion source (-> zig-port/pysrc) to the four nodes; prompts rendered once on rank 0's
node; `zig build test-glm53` + `fatbins tf-glm53-generate` on every node in the image (CPU, `--memory 24g`); the Python
reference at TP4 (`zp2a-ref-r*`, which also records each node's Triton AOT set); `tf-glm53-generate` at TP4
(`zp2a-gen-r*`); compare on rank 0's node. Knobs (TOKENS, CONTEXT, LAYERS, SKIP_BUILD / SKIP_REF with REUSE=<stamp>,
MEM_CAP, ...) are listed at the top of the script.

| file | what |
|---|---|
| `reference.py` | the Python engine (fused path) at TP: one process a rank, NCCL (`tensorfold.cuda.comm.NCCL`) shown as all_gather-only so fused.gather takes the all-gather + rank-order-sum branch for every window; prompt in windows of 128 rows, the last row's pick through `fused.head` "argmax", then one-row decode windows; `--record` packs this schedule's Triton set into `ref/aot`; writes `ref-r<R>.json`, `inv_freq.bin` (torch's bytes), `nccl.txt` (the libnccl it opened). `--make-prompts` renders the chat template |
| `zig/tests/cuda/glm53_generate.zig` | `tf-glm53-generate`: the same schedule in Zig (TCP rendezvous for the NCCL id, ncclCommInitRank, the weights layer by layer, `Forward.head`), `gen-r<R>.json` in the same format |
| `compare_phase2a.py` | tokens and records against the reference, the four Zig ranks against each other; `PHASE2A PASS|FAIL` |
| `run_phase2a_cluster.sh` | the cluster run above (a copy lives at ~/zig-port/run_phase2a_cluster.sh) |

Frozen beyond Phase 1's set: `TF_GLM53_RADIX_MIN_ROWS=1` (radix select for every window, the Zig side's), the
all-gather exchange for prompt windows too (the champion uses NCCL's bf16 ring all-reduce for prompt chunks, which the
Zig port does not reproduce yet), no RoCE one-shot, no MTP layer. RoPE: `glm53_rope.cu` computes fused.inv_freq with
the same libdevice powf torch's kernel calls; the tool checks it against the reference's `inv_freq.bin` (and uses the
reference bytes, saying so, if they ever differ).

# Phase 2b: MTP drafts, keyed sampling, CUDA graphs, the fast load - token parity with the Python Runner

Gate: full GLM-5.3 + the MTP layer at TP4 (.1 .2 .3 .5), the Zig engine against the Python engine's own `Runner`
(runner.py) on the same prompts: (a) MTP drafts k=2 (normed/normed, TF_GLM53_MTP_REUSE=2, the q4 draft head over
TF_GLM53_DRAFT_VOCAB=32768) verified in windows of 3 rows; (b) keyed sampled decoding at generation_config.json's
temperature 1.0 / top_p 0.95 - top_k 20 (the server's default: each rank's top 28 of its vocabulary share gathered,
Runner._sample_local) and top_k 0 (Glm53Engine._sample's torch.topk(264) rule) - plus greedy; every token IDENTICAL
to Python's, and drafted == serial in both engines; (c) the decode windows (1..3 rows, every key bucket, both heads,
the MTP steps) captured as CUDA graphs and prewarmed as Runner.prewarm does; (d) the weights read with O_DIRECT through
io_uring a layer ahead (cuda_source.zig); (e) decode tok/s, tokens a round and acceptance of both engines side by
side, the 32K prose prompt (tfbench's beekeeper task over context-32k.txt, seed 1729) included.

## One command, from .4 (free the cluster first; it stops nothing it did not start)

```sh
bash ~/zig-port/run_phase2b_cluster.sh        # last line: PHASE2B PASS|FAIL ... work=~/zig-port/runs/p2b-<utc>
```

Same preflight refusals, memory caps and page-cache drops as 2a. Steps: sync -> prompts (reference2b.py
--make-prompts: the 2a prompts with runs `g:2,g:0,s20:2,s20:0,s0:2,s0:0` x 256 picks, prose32k with `s20:2,s20:0` x 512)
-> build -> **sampler check** (sampler_cases.py: 1,500 draws + 200 seeds from exact_sampling; `tf-glm53-generate
--sampler-check`; a FAIL stops the run before any GPU work) -> reference2b.py at TP4 (records the Triton AOT set) ->
`tf-glm53-generate --mode 2b` at TP4 -> compare_phase2b.py. CONTEXT is the longest prompt + its picks rounded up to
1024 (both engines: Runner capacity CONTEXT + k + 1). Knobs at the top of the script (RUNS, RUNS_LONG, TOKENS,
TOKENS_LONG, CONTEXT32K, FAST_LOAD, PREWARM, GRAPHS, SKIP_* / REUSE as in 2a).

| file | what |
|---|---|
| `reference2b.py` | the Python Runner, frozen: TF_GLM53_PROMPT_ROWS(_SHORT)=128 (row-invariant prompt chunks), all-gather-only comm (no ring all-reduce, no RoCE), copy drafts off, MTP/draft-head defaults; caches zeroed before every run; `ref-r<R>.json` |
| `zig/tests/cuda/glm53_generate.zig --mode 2b` | the Zig side (`glm.Runner`), `gen-r<R>.json` in the same format |
| `sampler_cases.py` | generated draws for `--sampler-check` (sampling.zig against exact_sampling, bit for bit: picks, nucleus sizes, seed_for) |
| `compare_phase2b.py` | tokens, drafted == serial, rank agreement, seeds; the speed table; `PHASE2B PASS|FAIL` |
| `run_phase2b_cluster.sh` | the cluster run (a copy lives at ~/zig-port/run_phase2b_cluster.sh) |

Zig pieces: `cuda_runner.zig` (rounds, graphs, prewarm, the sampled head's candidate exchange), `sampling.zig`
(seed_for, splitmix uniform, choose_rows with numpy's pairwise sum and glibc exp/log), `cuda_forward.zig` (device ids,
`headDevice` argmax / local / draft, `mtpCompute`), `cuda_weights.zig` (MTP layer, q4 draft head: quantize4 + shared
qmm.pack on the host), `cuda_source.zig` (the O_DIRECT prefetcher), `glm53_head.cu` (argmax records, rank resolve,
embedding rows, fp32 -> bf16), `qmm_group.cu` (+ the fp32-out tile-2 instantiation the draft head launches).

Not in 2b (both engines run without them, so speeds compare like with like): the RoCE one-shot (P3), copy drafts,
DFlash2, the depth policy, prompt reuse, prompt chunks over 128 rows (x3prefill GEMM / bmm absorb), side streams.
Candidates of sampled rows are selected on the host from each rank's downloaded logits (P3: a device top-k).

# Phase 3a: the RoCE one-shot and the served prompt path - against the Python engine SERVED

Gate (one run): (1) the RoCE one-shot all-reduce / small all-gather in Zig - b12x RoCEnante's proxy
(`zig/src/families/glm_moe_dsa/roce_proxy.c`, verbatim, compiled by Zig against libibverbs) and its two CuTe kernels
re-written in plain CUDA (`zig/kernels/cuda/glm53_roce.cu`: the same protocol and PTX, the fixed rank-order fp32 sum) -
bit-equal to NCCL's rank-order sum and timed for the decode windows' shapes (target <= 65 us a reduction); (2) the
served prompt path and RoCE decode windows, every token identical to `Glm53Engine` running the 2026-10-05 image's
defaults; (3) Zig vs served Python: 32K prose decode tok/s (MTP k=2, sampled, thinking on), prefill tok/s at 32K and
128K, load time. PASS: identity and Zig >= 0.97 x Python on the 32K prose decode and both prefills.

## One command, from .4 (free the cluster first; it stops nothing it did not start)

```sh
bash ~/zig-port/run_phase3a_cluster.sh        # last line: PHASE3A PASS|FAIL ... work=~/zig-port/runs/p3a-<utc>
```

Steps: preflight as 2b -> sync -> prompts (2b's + `long128k`, the 32K background text four times over) -> build (the
new kernels' symbols checked with cuobjdump) -> `tf-glm53-generate --mode roce-bench` (no model: the bit check, then
us a reduction of [1..32, 6144] fp32 from a CUDA graph, eager, and NCCL's all-gather of the same) -> the Python engine
served (`reference3a.py`) -> [`SPEED_DEFAULT=1`: a speed-only Python pass at the default prompt-experts mode] ->
`tf-glm53-generate --mode 3a` -> `compare_phase3a.py`. Knobs at the top of the script.

| file | what |
|---|---|
| `reference3a.py` | `Glm53Engine` itself with the image's defaults (RoCE, PROMPT_SP, 8192 / 4096-row chunks, ring reduce, timed tiles, side stream, L2 prefetch, prompt experts); frozen only: `TF_EXL3_PROMPT_DET=slots16`, `TF_GLM53_TILES=save:` (the table the Zig ranks load), copy drafts / prompt reuse off. Writes `ref-r<R>.json`, the Triton AOT set, `tiles-r<R>.json`, `bmm_probe.json`, `blas.json`, `inv_freq.bin`, `nccl.txt` |
| `zig/tests/cuda/glm53_generate.zig --mode 3a` | the Zig engine with the same: `Forward.fast` (RoCE), `.ring`, `.wide` (prompt GEMM + cuBLAS bmm), `cuda_prompt.Sp` (sequence-parallel halves), `tiles.load`, the bmm probe first |
| `zig/tests/cuda/glm53_generate.zig --mode roce-bench` | the RoCE runtime alone; `ROCE-BENCH PASS|FAIL` |
| `compare_phase3a.py` | tokens, drafted == serial, ranks; the speed gates; `PHASE3A PASS|FAIL` |
| `run_phase3a_cluster.sh` | the cluster run (a copy lives at ~/zig-port/run_phase3a_cluster.sh) |

Why TF_EXL3_PROMPT_DET=slots16 in both engines: the image's default prompt-experts mode adds a row's expert outputs
with fp32 `red.add` in arrival order, so a long prompt's caches - and the tokens after it - differ from run to run of
the same engine; slots16 (fp16 pair rows summed in slot order) is the image's reproducible mode (~4-6 % slower
prefill). `SPEED_DEFAULT=1` prints the Python prefill at the default too (not gated).

Speed work after run 2 (2026-10-08; every change keeps the one-stream bits): the decode side stream
(`cuda_decode.Side`, TF_GLM53_SIDE "af": the key path and the shared expert beside the query path / routed experts),
the L2 prefetch of the target's decode windows (`cuda_decode.L2pf`, TF_GLM53_L2PF bulk, 8 MiB a site, sites "afo",
l2pf.py's table), decode-sized index selections through a multi-block top-k (`glm53_decode.cu`; the one-program radix
select cost 21 indexer layers ~0.25-0.5 ms each a window at 32K-128K keys - Python takes torch.topk there), sampled
candidates picked on the device (one read a window), the MTP layer's prompt rows in blocks of ~1360 rows (scratch inside
the prompt buffers' idle `gath`: ~3x fewer expert weight reads than blocks of 128). Zig knobs: `ZSIDE`, `ZL2PF`,
`ZL2PF_MB`, `ZMULTI_SELECT`, `ZDEVICE_CANDS`. `PROFILE=1` times PROFILE_PROMPT's (prose32k) first prefill by phase
in both engines (`prefill-prof.txt`). The reference container now gets a writable b12x cache (`VLLM_CACHE_ROOT`):
run 2's Python ran WITHOUT RoCE ("Permission denied: /root/.cache/vllm/b12x-compile"), so its decode figures were
NCCL's.

After run 4 (prefill 32K 0.960, 128K 0.946; the MTP layer's prompt rows 2.1 s of a 29.4 s 32K prefill, serial on the
main stream): `--mtp-dense 1` (`ZMTP_DENSE`, default 1) runs the MTP layer's prompt rows through cuBLAS -
`cuda_prompt.MtpDense`: its 8-bit experts kept a second time as dense bf16 (EXL3 rotations folded in, built at load
from the same trellis: `unpack_kernel`, `glm53_mtp.cu`'s Hadamard passes), the pairs grouped by expert on the host (one
sync a chunk), a GEMM an expert for gate | up and for down, combine in slot order; eh_proj as one GEMM (which drops its
1.6 GB of router partials from the prompt buffers). ~4.97 GB a rank, ~3.2 GB net. Drafts only: the target verifies
every token, so tokens cannot move; the gate adds a tokens/round check on the 32K prose drafted runs (`TPR_TOL`, 2 %).

What must match bit for bit beyond Phase 2b's kernels: the prompt GEMM (`x3prefill._gemm`, Triton, captured), its
weight decode (`unpack_kernel`) and input rotation (`rot_in`), torch.bmm (cuBLAS `GemmStridedBatchedEx` with torch's
arguments, workspace and math mode, the library file torch loaded - `--bmm-probe` checks three shapes first), the prompt
experts (`glm53_prompt_experts.cu`, generated by `copy_exl3_kernels.py` from `prompt_experts.cu`), NCCL's bf16 ring
all-reduce / reduce-scatter (same libnccl, same NCCL environment, same sizes) and the served tile table.
