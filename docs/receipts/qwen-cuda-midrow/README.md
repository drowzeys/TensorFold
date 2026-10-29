# Qwen3.5 CUDA prompt rows on mid-row tiles (DGX Spark)

The affine 4-bit prompt GEMM now runs prompts of tens to hundreds of rows on tiles that fit them. Every output keeps
the 128-row tile's bits, so replies are unchanged; on Qwen3.8-27B a resumed chat turn's prompt pass takes about half
the GPU time it did.

## Where the time went

One NVIDIA DGX Spark (GB10, sm_121, 48 SMs, 128 GB unified memory), driver 580.173.02, Linux 6.17, host nvcc 13.0
(V13.0.88), Zig 0.17.0. Target `TensorFold/Qwen3.8-27B-MLX-4bit` (revision
`22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`), drafter `z-lab/Qwen3.8-27B-DFlash2` (revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`), `--context 32768 --parallel 2`, BF16 KV.

In an nsys trace of resumed turns (each a follow-up question resumed from the turn-start prompt state), the 4-bit
prompt GEMM was 82% of kernel time. Two losses, neither in the tile's arithmetic:

- The 128 by 128 tile pads rows to whole tiles: 129 to 256 prompt rows cost 256 rows of MMAs.
- Projections of at most 1,024 columns run as one to sixteen CTAs, each walking all of K alone. The 48-column
  a and b projections take about 105 us at any row count, 96 launches a pass.

The MMA throughput itself is close to the chip's: the 128-row tile reaches 77 TFLOPS at 2,048 rows and cuBLAS's BF16
GEMM 81 TFLOPS on the same shapes, so a faster mainloop was not the lever.

## The change

`qmm_midrow.cu` adds ten tiles, 16 to 128 rows by 32 to 256 columns. Each weight is rounded once to bf16 by
`fma.rn.bf16x2` and each output is one fp32 MMA chain over K in the 128-row tile's k16 order, then one bf16 rounding;
only the rows and columns a CTA owns, its warps and its stages differ. `affine4_mid.pick` chooses by rows and columns:
narrow tiles for at most 2,048 columns, else the wide tile whose waves (CTAs per SM, rounded up) times outputs times a
fitted per-tile cost is least, and the 128-row tile wherever none is quicker. The Qwen3.5 CUDA engine attaches the
tiles on GB10 only.

## Kernel times

Median of 25 launches, idle GPU, weights rotated past L2, every output byte-equal to the 128-row tile's.

| projection | N x K | rows | 128-row tile us | picked tile | us | speedup |
|---|---|---|---|---|---|---|
| delta qkv | 10240 x 5120 | 17 | 207 | 32x256 | 132 | 1.57x |
| delta qkv | 10240 x 5120 | 129 | 412 | 80x256 | 224 | 1.84x |
| delta qkv | 10240 x 5120 | 161 | 414 | 96x128 | 281 | 1.47x |
| delta qkv | 10240 x 5120 | 257 | 523 | 96x128 | 355 | 1.47x |
| delta z | 6144 x 5120 | 129 | 215 | 80x256 | 129 | 1.66x |
| delta a/b | 48 x 5120 | 17 to 289 | 105 | 16x32 | 19 | 5.5x |
| o / out | 5120 x 6144 | 129 | 253 | 80x256 | 152 | 1.66x |
| attn q | 12288 x 5120 | 129 | 419 | 80x256 | 281 | 1.49x |
| attn k/v | 1024 x 5120 | 129 | 105 | 32x64 | 41 | 2.55x |
| gate / up | 17408 x 5120 | 129 | 629 | 80x256 | 345 | 1.82x |
| gate / up | 17408 x 5120 | 193 | 634 | 80x256 | 546 | 1.16x |
| down | 5120 x 17408 | 129 | 707 | 80x256 | 416 | 1.70x |
| down | 5120 x 17408 | 513 | 1770 | 80x256 | 1140 | 1.55x |

Summed over prompts of 17 to 497 rows (8-row steps to 320, then 32): 1.35x over all eight projections (best tile per
row count: 1.38x), and no row count more than 3% slower than the 128-row tile. At 97 to 128 rows on 5,120 to 17,408
columns, and on some shapes at 193 to 225 rows, the 128-row tile is kept. The costs were fitted on these
measurements.

## Turns

Resumed turns (a six-turn chat, then ten different follow-ups with `max_tokens` 1), three alternating rounds a build,
each started on an idle GPU:

| build | resumed TTFT median (n = 60) | p90 | cold first request |
|---|---|---|---|
| before | 237 ms | 273 ms | 1,055 to 1,079 ms |
| after | 156 ms | 179 ms | 773 to 803 ms |

In an nsys trace of the same requests, the ten resumed passes' 4-bit prompt GEMM went from 3,060 ms to 1,407 ms and
their kernel time from 3,305 ms to 1,601 ms; the cold pass's GEMM from 852 ms to 451 ms.

## Replies

A fixed eight-turn chat in three modes (greedy and two seeded samplings), 24 replies: byte-identical before and after
with the prompt cache on and with `--prompt-cache-gib 0`.

## Checks

- `tf-cuda-test affine4-mid`: 600 tile launches on Qwen3.8-27B's projections and two ragged shapes at 17 to 513 rows,
  and each pick, byte-equal to the 128-row tile, NaN and both infinities included.
- `tf-cuda-test kernel-contract`: the ten tiles pass gates 1 to 4 as replacements of the 128-row tile, their planted
  control (the 80x256 tile summing K's groups in reverse) is red at gate 3, and their gate 5 records match the live
  GB10 (430 kernels pass gates 1 to 4, 313 controls red; none of the run's 199 limit problems is a qmm_midrow
  launch). The run predates the fitted pick, which no gate calls. The sm_80 to sm_120 records come from ptxas, not
  from `kernel-contract-limits`.
- `zig build test-cuda-host` passes; `lean_check.py` adds no problems.
