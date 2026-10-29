# LFM3-Omni-2B speech to speech, native Metal

`lfm3_omni` also runs LiquidAI's [`LiquidAI/LFM3-Omni-2B`](https://huggingface.co/LiquidAI/LFM3-Omni-2B) (revision
`5797d07c125503bd4f4c0e7f915ae2d3d703e8de`) on Apple silicon: a spoken turn in, interleaved text and 24 kHz speech out,
streamed. Same weights (the bf16 release as stored, the bundled Liquid codec included), same tool and server as the
CUDA port ([lfm3-omni-cuda.md](lfm3-omni-cuda.md)); the engine is `zig/src/families/lfm3_omni/metal/` over one
metallib (`zig/kernels/metal/lfm3_omni/`). Checked on an M5 Ultra (macOS 27). The vision tower is not loaded.

```bash
hf download LiquidAI/LFM3-Omni-2B --revision 5797d07c125503bd4f4c0e7f915ae2d3d703e8de --local-dir "$HOME/models/lfm3-omni-2b"
# the Metal compiler: xcodebuild -downloadComponent MetalToolchain (macOS 26 and later), with Xcode selected
zig build lfm3-omni-generate -Doptimize=ReleaseFast
zig-out/bin/lfm3-omni-generate --model "$HOME/models/lfm3-omni-2b" --wav question.wav --out reply.wav \
  --system "Respond with interleaved text and audio. You are a helpful voice assistant."
zig-out/bin/lfm3-omni-generate --model "$HOME/models/lfm3-omni-2b" --serve 127.0.0.1:8441 --system-file system.txt
```

`POST /v1/audio/chat` and `GET /health` are the CUDA recipe's (server-sent text deltas, base64 16-bit 24 kHz audio
chunks, a `done` event with the timings). About 4.4 GB of unified memory.

## What runs

- The CUDA engine's operations, one for one, rounding to bf16 where the reference's op rounds: log-mel as fp32
  products, the speech encoder (strided convolutions, 17 LFM2 layers) and projector, the 24-layer backbone, the depth
  decoder's three mask passes with each codebook's pick on the GPU (repetition penalty, greedy or a Gumbel-max draw),
  the Liquid codec's decoder and its spectrum head; overlap-add on the host. Weights stay in shared buffers.
- Products of many rows (prompt chunks of 512, the encoder, codec chunks) on the M5 tensor unit, through the prefill
  GEMM's 32x32 blocks (`zig/kernels/metal/prefill/gemm_nax.metal`), fp32 accumulation; a GPU without tensor units
  takes a simdgroup-matrix product (correct, but slow on long prompts; `LFM3_OMNI_NAX=0` forces it).
- Up to 16 rows (a backbone row, the depth decoder's 12 slots, a codec chunk's first rows): a weight-streaming
  product (a simdgroup a 4 weight rows, about 1 TB/s on the backbone), the heads' norm, rotary and cache write in one
  dispatch, one-row attention with LFM3's output gate inside, SwiGLU as the w1 | w3 product's epilogue (those rows
  are stored as pairs), a sandwich layer's post norm inside the next RMSNorm; the greedy text pick in two stages.
- One serial command buffer from one copy back to the next: a backbone row is committed and runs while the host
  does the overlap-add; the first frame is decoded alone, then every 8.

`LFM3_OMNI_PROFILE=1` logs each phase's GPU time per reply (backbone row, speech frame, text pick, codec chunk);
`LFM3_OMNI_KBENCH=1` times the decode path's dispatches alone at load.

## Receipt

M5 Ultra, the CUDA receipt's 16 synthetic spoken prompts (macOS voices) and voice-assistant persona, against the
reference (`transformers` with the checkpoint's own code, on the same Mac's GPU through MPS, bf16; and fp32 for the
reference's own rounding noise). Identity codebook order, greedy text and codes, as the CUDA receipt.

| stage | check | Metal vs MPS bf16 | (CUDA vs CUDA bf16) |
| --- | --- | --- | --- |
| log-mel | cosine, max abs | 1.000000, 3.1e-4 | 1.000000, 2.8e-4 |
| speech encoder | cosine (min of 16) | 0.99999 | 0.99999 |
| projector | cosine (min) | 0.99990 | 0.99990 |
| prompt | token ids | 16/16 identical | 16/16 |
| backbone prefill | row cosine (median, min of medians) | 0.99982, 0.99977 | 0.99976, 0.99915 |
| text | teacher-forced greedy top-1 | 520/524 (99.2%) | 98.5% |
| depth decoder | on the reference's own states: greedy agreement | 95.5%, logits cosine 0.999994 / 0.99989 | 94.4% |
| codes | teacher-forced, the whole loop | 89.2% | 89.7% |
| codec | the reference's codes to audio: SNR (median, min) | 36.6 dB, 25.1 dB; head rows cosine >= 0.99994 | 36.3, 24.1 dB |

Against the fp32 reference (MPS fp32) the port sits where the reference's own bf16 run sits: encoder 0.999995 (bf16
reference 0.999995), projector 0.99995 (0.99995), prefill row cosine median 0.99988 (0.99988), the first frame's depth
logits 0.9995 (0.9996); teacher-forced text 460/468 (98.3%), codes 86.8%, the depth decoder on fp32 states 92.4%
greedy agreement (logits cosine 0.999993 / 0.99969), codec SNR 36.8 dB median, 22.7 dB min. A reply's streamed chunks (the first frame alone, then 8) against one decode of all its frames: 34 to
43 dB SNR, not bit-identical (a 6-row chunk takes the few-row product, a long one the tensor-unit product).

Replies with the card's audio settings (temperature 0.4, penalty 1.2), 16 turns, through the server and a streaming
client (two passes, identical); the reference runtime timed as the CUDA receipt times it:

| | Metal (M5 Ultra) | reference runtime, same Mac (MPS bf16) | CUDA port (GB10) | reference runtime (GB10) |
| --- | --- | --- | --- | --- |
| first text, median | 0.033 s | | 0.036 s | |
| first audio, median | 0.062 s | 0.280 s | 0.144 to 0.156 s | 0.236 s |
| whole reply, median | 0.64 s (6.5 s of speech) | 2.71 s (6.4 s) | 2.0 to 2.2 s | 4.14 s |
| UTMOS22 strong, median (min) | 4.43 (4.35) | 4.44 (4.15) | 4.42 (3.91) | 4.42 (4.24) |
| speech vs its own text, WER | 1.5% | 1.2% | 3.0% | 1.7% |

WER is the reference model's own ASR on each reply's audio against the reply's streamed text. GPU time per reply
(`LFM3_OMNI_PROFILE=1`): a backbone row 3.5 ms (2.97 GB of bf16 weights; the products alone run in 2.9 ms), a speech
frame (three depth passes) 1.9 ms, a codec chunk 2.5 ms. Timings were taken with other GPU work on the same machine
at times; the medians above are from runs where the two passes agreed.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint). Use, changes and redistribution are allowed; commercial use is licensed only to an
organisation under US$10 million in annual revenue, and a qualified non-profit may use it for research or other
non-commercial purposes regardless. TensorFold's code stays Apache-2.0.
