# LFM3-1.7B NVFP4, native CUDA

`lfm3` serves LiquidAI's ModelOpt NVFP4 text export,
[`LiquidAI/LFM3-1.7B-NVFP4`](https://huggingface.co/LiquidAI/LFM3-1.7B-NVFP4) at revision
`b30025bbbbcbf62f6c110b40d51d36af6579f6a2`, on one NVIDIA GPU through the lane host. Checked on a GB10 (sm_121).
The MLX affine and FP8 exports are not loaded on CUDA yet, and Metal refuses this export by name
(`Lfm3ModeloptNvfp4NotOnMetal`). LFM3-VL-2B (bf16, with images) is [below](#lfm3-vl-2b-chat-with-images).

```bash
hf download LiquidAI/LFM3-1.7B-NVFP4 --revision b30025bbbbcbf62f6c110b40d51d36af6579f6a2 \
  --local-dir "$HOME/models/lfm3-1.7b-nvfp4"
zig build native -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm3-1.7b-nvfp4" --port 8080 --parallel 8
```

## Weight-only

The mode is weight-only. The FFN's FP4 weights (w1, w3, w2, E2M1 in groups of 16 with e4m3 block scales and one fp32
tensor scale) are used as stored, and every activation stays bf16. The export also carries static FP4 input scales;
they are read, because the admission requires them, but not applied. The mixers (conv, q/k/v/gate, o), the embedding
and the tied head are bf16 in the export and run as bf16.

## What runs

- Conv layers (13): Liquid's 7-tap short conv with its rolling state (`liquid_conv.cuh`, shared with LFM2.5).
- Full-attention layers (6): keys and values in pages of the core KV pool, partial RoPE through `qk_rows`' frequency
  table, `pattn_kernel<128, 8, 4, 8>` for prompt rows and the shared paged chunk and merge kernels for decode rows.
- Sliding-attention layers (5, window 256): a ring a stream through the shared window-attention kernels.
- The lane backend, the kept states of the prompt cache and the native glue are shared with LFM2.5-24B-A2B
  (`zig/src/families/liquid/`). Kept states live in pages allocated once inside the prompt-cache budget.
- The default window is 32,768 tokens (`--context`); the model allows 262,144.

## Receipt

GB10, greedy, `--parallel 8`:

- drafted == plain and concurrent == solo on 8 of 8 prompts, prompts of 300 to 2,960 tokens across the window,
  prompt rows == one-row decode at 32 cut points, resumed == fresh, and the token hashes after a restart;
- a float64 forward of the same weights on the CPU (`tools/lfm3/ref64.py`) picks the engine's greedy token on 825 of
  829 teacher-forced prompt rows and 127 of 128 reply rows; `tools/lfm3/REF64.md` lists every miss with both tokens
  and margins;
- decode, total tok/s: 112.8 at 1 stream, 420.0 at 4, 767.4 at 8; cold TTFT 0.150 s at 2k tokens, 0.615 s at 8k,
  2.84 s at 32k.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint), as the MLX exports are. Use, changes and redistribution are allowed; commercial
use is licensed only to an organisation under US$10 million in annual revenue, and a non-profit may use it for
research or other non-commercial purposes regardless. TensorFold's code stays Apache-2.0.

## LFM3-VL-2B chat with images

`lfm3_vl` checkpoints without d1's `d1_*` settings chat on the same lane engine from the bf16 release,
[`LiquidAI/LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B) at revision
`ba44bd51db20a2753f28c273a1571ccaa210d954`; d1-2B keeps its decision engine. Checked on a GB10 (sm_121).

```bash
hf download LiquidAI/LFM3-VL-2B --revision ba44bd51db20a2753f28c273a1571ccaa210d954 \
  --local-dir "$HOME/models/lfm3-vl-2b"
zig-out/native/bin/tensorfold-native serve "$HOME/models/lfm3-vl-2b" --vision --no-thinking --context 8192 \
  --prompt-cache-gib 1 --port 8080
```

- Images go in as `image_url` parts (PNG; JPEG and WebP with their libraries), up to `--vision`'s image count (16 at
  most) and 1,600 tokens (1280 x 1280) each: LFM3-VL's processor (smart_resize, no cap) on the CPU, then the vision
  tower and projector on CUDA (`lfm3v.cu`, shared with d1-2B). Their rows replace the template's `<|image|>` rows
  over the prompt chunk's embedding (`zig/src/families/liquid/cuda_pictures.zig`); a kept state already holding an
  image skips it.
- The text decoder's projections are loaded twice: lane weights for decode rows, and plain twins that prompt rows read
  through cuBLASLt (fp32 sums rounded once, as torch's Linear). About 7.3 GiB of device memory with an 8k window.
- `--no-thinking` (or `chat_template_kwargs: {"enable_thinking": false}`) answers at once; the template thinks
  otherwise.
- `logprobs` and `top_logprobs` are served for every Liquid family on CUDA, read on the host from each row's logits.

Receipt, GB10, greedy, 44 synthetic stills (generated pictures of people at a desk, 640 x 480) with a camera question
each, 48 reply tokens, against transformers 5.21 running the checkpoint's own code on the same GPU:

- replies identical to transformers bf16 on 41 of 44 and to transformers fp32 on 42 of 44 (bf16 against fp32: 43);
- teacher-forced along the engine's tokens, its top-20 probabilities sit 0.000277 from fp32 on average (transformers
  bf16 0.000266; its cached decode rows, the engine's own kind, 0.00625 per row against the engine's 0.00596), and it
  picks fp32's top token on 545 of 547 rows (transformers bf16: 546); the two misses are 0.18 and 0.008 nats apart;
- eight text-only prompts, short and after a 450-token passage: 0.000633 and 0.000613 from fp32 (transformers bf16
  0.000674 and 0.000591), each row as often closer as further than transformers' cached decode rows;
- the tower's 300 rows of one picture: 0.023 relative from fp32 (transformers bf16: 0.027);
- camera answers (36 questions, one picture each, a quiet GPU): first word 0.090 s median (transformers 0.102 s), the
  whole answer 0.315 s (0.400 s); the prompt pass is 60 ms of GPU time, a decode row 16.7 ms.

The weights are under the LFM Open License v1.0, as above.
