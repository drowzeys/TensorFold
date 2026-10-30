# Float64 reference receipt: LFM3-1.7B NVFP4 on CUDA

- Checkpoint: [LiquidAI/LFM3-1.7B-NVFP4](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4) at revision
  `b30025bbbbcbf62f6c110b40d51d36af6579f6a2`.
- Engine: `9265a0f5f`, which is the PR head `08b338c7c` before its rebase (range-diff: the same patches except one
  test-only line in `contract_limits.zig`). Served on a GB10 (sm_121), greedy, drafts off,
  `--parallel 8 --context 32768`. The `collect` step was repeated on 2026-10-28 and returned the same 957 tokens.
- Reference: `ref64.py` on the CPU in float64 (Python 3.12, torch 2.14, safetensors 0.8, tokenizers 0.23), 2026-10-28.

Reproduce (the reference pass takes about a minute and 20 GB of host memory):

```
zig-out/native/bin/tensorfold-native serve MODEL_DIR --port 8080 --parallel 8 --context 32768
python -I tools/lfm3/ref64.py collect http://127.0.0.1:8080 MODEL_DIR engine.json
python -I tools/lfm3/ref64.py MODEL_DIR engine.json --bf16
```

Rows. A prompt row is the engine's one-token greedy pick after the prefix [0, t] of a prompt, one request per prefix.
A reply row is a token of the engine's 32-token greedy reply, teacher-forced on that reply. `t` is the logits row after
token t, so prompt row P - 1 and reply token 0 are the same position. The prompts (`prompts()` in the script) are
capitals (20 tokens), code (40), a chat turn (19) and items (750, a list of 60 items and a question; it crosses the
256-key sliding window).

The float64 pass dequantizes the NVFP4 weights exactly. The `--bf16` pass adds the engine's bf16 storage points (see
the script's docstring). It shows which side of a miss rounding alone lands on; it is not bit-exact.

## Result

- f64: prompt rows 825/829, reply rows 127/128
- bf16: prompt rows 827/829, reply rows 126/128

Float64 top-1 misses (margin = f64 logit of its top minus f64 logit of the engine's token; bf16 margin = the same difference in the bf16-points pass, < 0 means it prefers the engine's token):

| case | row | t | engine | float64 | f64 margin | bf16 margin | bf16 picks |
|---|---|---|---|---|---|---|---|
| capitals | reply | 24 | 11097 " Spain" | 3888 " Japan" | 0.1176 | +0.1250 (pre-round +0.1802) | f64's |
| chat | prompt | 12 | 593 " per" | 22 "." | 0.1530 | -0.0625 (pre-round -0.0671) | engine's |
| items | prompt | 5 | 1008 " first" | 1579 " number" | 0.0996 | +0.0000 (pre-round +0.1009) | engine's |
| items | prompt | 9 | 29 "5" | 654 "10" | 0.0033 | +0.0000 (pre-round +0.0093) | engine's |
| items | prompt | 180 | 28 "4" | 30 "6" | 0.0674 | +0.0000 (pre-round +0.0076) | engine's |

bf16-points misses where float64 agrees with the engine (margin = bf16 logit of its pick minus the engine's token; f64 margin = how far float64 prefers the engine's token):

| case | row | t | engine | bf16 pick | bf16 margin | f64 margin |
|---|---|---|---|---|---|---|
| items | prompt | 4 | 440 " The" | 229 " " | 0.0000 (pre-round -0.0985) | 0.0881 |
| items | prompt | 631 | 32 "8" | 25 "1" | 0.0000 (pre-round -0.0467) | 0.0636 |
| items | reply | 772 | 41501 " Item" | 1672 " So" | 0.0000 (pre-round -0.0814) | 0.1797 |

## LFM3-VL vision tower and projector

`vision_ref64.py MODEL_DIR IMAGE OUT.pt [--hf]` runs the tower and projector in float64 on the CPU, from the HF processor's `pixel_values`, on the stored bf16 weights. MLX keeps the depthwise conv as [C, 3, 3, 1] and torch as [C, 1, 3, 3]; the script reads both. `--hf` also runs transformers' `Lfm3VlVisionModel` and projector on the same weights and prints the gap, in float32 by default and in bf16 with `LFM3VL_HF_DTYPE=bf16`. That gives the "reference's own bf16" yardstick. `vision_check.py REF.pt compare OUT.bin` compares an engine's bf16 feature dump against the float64 features, and `dump` writes the pixels as bf16 for an engine.

LFM3-VL-2B-MLX-4bit @ `c823fade`, the synthetic 217×300 picture from #1584's `patches_test.zig` (grid 1×14×20, 70 tokens), on a GB10:

| features [70, 2048] against float64 | rel L2 | cos | worst row cos |
|---|---|---|---|
| transformers 5.19, fp32 | 1.46e-5 | 1.000000 | 1.000000 |
| transformers 5.19, bf16 | 4.74e-2 | 0.998877 | 0.971098 |
| a bf16 CUDA tower (jschmied/TensorFold:lfm3-vl-cuda-tower) | 3.38e-2 | 0.999430 | 0.980361 |

A planted control (the grid passed as 20×14) gives rel 0.82 and cos 0.67.
