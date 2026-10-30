# Qwen3.5 CUDA decode on GB10: voice-turn first sentences (DGX Spark)

Where a resumed voice turn's decode goes on Qwen3.8-27B with the DFlash2 drafter, and two changes on top of the
turn-start resume and mid-row prompt tiles (both rebased onto 1.1.9 here). Replies are byte-identical throughout.

## Setup

One NVIDIA DGX Spark (GB10, sm_121, 48 SMs, 128 GB unified memory), driver 580.173.02, Linux 6.17, nvcc 13.0
(V13.0.88), Zig 0.17.0. Target `TensorFold/Qwen3.8-27B-MLX-4bit`, drafter `z-lab/Qwen3.8-27B-DFlash2`,
`--context 32768 --parallel 2 --prompt-cache-gib 2 --no-thinking`, BF16 KV. Baseline: the 1.1.9 release archive
for linux-aarch64 (sha256 `e1d36b67f0298ee1cacf4f6c40482bbc3a4f539c771d01006bf7f6e0927c669a`).

The text bench is a voice assistant's chat: its system prompt and 16 spoken requests in one conversation, each sent
first as a one-token warm request and then streamed at temperature 0.7 with a fixed seed a turn (as the voice
agent sends them). "First sentence" is the end of the first sentence after the mood tag. Builds alternate in rounds
on an otherwise idle GPU; every build's 16 replies are the same text in every round.

## Where a decode round goes

A round is one verify window through a captured CUDA graph, then a DFlash2 hold for the next drafts.

| part | ms a round (1.1.9) |
|---|---|
| verify window (16 rows typical; 48 and 64 for copied-continuation windows) | 78 to 80 mean over widths |
| drafter hold (5 layers and the head's spans, 16 rows) | 9.3 |
| absorb, keep, host | about 1 |

Inside a 16-row verify graph (nsys, node tracing) the 4-bit GEMV and GEMM tiles are about 95% of the kernel time:
gate and up 34 ms (128 launches at 270 us, about 200 GB/s), down, out and o 22 ms, the gated-delta qkv and z 14 ms,
the head 3.4 ms, attention q 2.8 ms. Two groups ran far below the rest: the gated-delta a and b projections, one
64-column tile each, ran on one CTA walking all of K (96 launches, about 50 us each, 4.8 ms), and attention's k and
v (16 tiles) on 16 CTAs (32 launches at 68 us). At 14.4 GB of weights a pass and about 273 GB/s, the bandwidth floor
is about 53 ms; a 16-row verify ran 79 ms median (66.9 ms with the GPU otherwise idle, see below).

Two things set the round count. A resumed turn's drafter had only the rows since the resume point as context, so
its drafts landed less often than after a full prompt pass (a 16-row verify kept 3.36 tokens against 3.79). And
the first round of each turn is a copied-continuation probe (48 or 64 rows, about 100 to 118 ms): it often fails on
fresh text but lands a whole "I am on it." reply in one round, so it stays.

## The changes

1. `affine4_kernels`: GB10 takes the segmented split-K launch (`tf_lane_gemv_seg`) for the projections of one input
   (gated-delta qkv, z, a, b; attention q, k, v), with dependent launch. Each block is one projection's tile and K
   slice and each output sums its slices in order, so the bytes are `decode()`'s. A lone projection stays on
   `lane_gemv`, which was as quick as the split form on out, o and down. `TF_GB10_SEG=0` restores the old launches.
2. `cuda_draft_resume`: a released drafter context keeps its stream's tokens; a new stream takes the pooled context
   whose tokens its prompt repeats longest and keeps those rows (within the ring's window). Only drafts change; the
   target draws every token. `TF_DRAFT_RESUME=0` starts each context empty.

A lane-planner change (carrying a lone stream's lane odds and price corrections to the next stream) was tried and
dropped: over three rounds it changed no window choice.

## Kernel times (nsys, 16-row verify graphs)

| group | 1.1.9 | this branch |
|---|---|---|
| gated-delta qkv, z, a, b, a layer | 394 us (4 launches) | 245 us (1 launch) |
| attention q, k, v, a layer | 312 us (3 launches) | 203 us (1 launch) |
| all 4-bit projections, a graph | 83.3 ms | 75.4 ms |
| graph span, median | 79.3 ms | 76.7 ms |

With the GPU otherwise idle the branch's 16-row graph ran 66.9 ms median (1 row 65.2 ms, 64 rows 88.6 ms).

## Voice turns (text bench, three alternating rounds, 48 turns a build)

| build | TTFT | first sentence | after first token | ms a round | tokens a round | ms a token |
|---|---|---|---|---|---|---|
| 1.1.9 | 542 | 817 | 274 | 88.8 | 3.47 | 25.6 |
| turn-start resume + mid-row + change 1 | 113 | 406 | 285 | 86.2 | 3.21 | 26.9 |
| + change 2 (this branch) | 114 | 372 | 267 | 86.5 | 3.45 | 25.1 |

Medians in ms. 1.1.9 has no turn-start resume, so it passes each turn's whole prompt (TTFT 542 ms).

## Resumed prompts: mid-row tiles against 1.1.9's fused prompt grid

1.1.9 runs the gated-delta and attention input projections of short prompts (17 to 256 rows) as one
`tf_qmm_prefill_many` grid on the 128-row tile. With the mid-row tiles attached each projection takes its own sized
tile instead (`TF_Q27_FUSED_FIRST=1` keeps the grid first). Ten resumed turns, two rounds each:

| path | resumed TTFT median | 4-bit prompt GEMM (nsys) |
|---|---|---|
| fused grid first | 166 to 172 ms | 2,005 ms |
| mid-row tiles first | 153 to 156 ms | 1,846 ms |

## Voice agent, voice to voice

The voice agent (ASR, this brain, Kokoro voice) over 16 recorded requests, two passes a run, three rounds a build
alternating, Bolt (`nvidia-nemotron-3.5-bolt` with its drafter) in the same session:

| brain | voice to voice median (p25 to p75) | first sentence | first token | first sentence after first token |
|---|---|---|---|---|
| Qwen3.8-27B on 1.1.9 | 780 (694 to 960), n = 60 | 782 | 638 | 197 |
| Qwen3.8-27B on this branch | 482 (295 to 610), n = 76 | 404 | 208 | 144 |
| Bolt | 358 (253 to 418), n = 89 | 282 | 113 | 144 |

The first sentence after the first token now matches Bolt's; the remaining gap is the first token (prompt pass and
queueing behind the warm request).

## Checks

- A fixed eight-turn chat in three modes (greedy and two seeded samplings), 24 replies: byte-identical to 1.1.9 with
  the prompt cache on and with `--prompt-cache-gib 0`.
- `tf-cuda-test affine4-decode` also checks GB10's own resolve (decode() on lane_gemv, decodeMany on the segmented
  launch with dependent launch) byte for byte at 1 to 16 rows on the 27B's and the 35B's shapes.
- `tf-cuda-test affine4-mid` passes on the rebased tiles; `zig build test test-cuda-host` pass; `lean_check.py`
  adds no problems. No new kernels: the segmented and split launches were already registered.
- `tf-cuda-test kernel-contract` on the rebased tree: every lane_gemv entry (the decode tile, split-K and the
  segmented launch) and all ten qmm_midrow tiles pass gates 1 to 4, and the mid-row control (the 80x256 tile summing
  K's groups in reverse) is red. The run then stops at gate 5 on 1.1.9's own `fp8_experts` registration, whose
  image `contract_limits.image` does not name; the other failing entries are kernels these branches do not touch.
