# LFM3-Omni-2B speech to speech, native CUDA

`lfm3_omni` runs LiquidAI's [`LiquidAI/LFM3-Omni-2B`](https://huggingface.co/LiquidAI/LFM3-Omni-2B) (revision
`5797d07c125503bd4f4c0e7f915ae2d3d703e8de`) on one NVIDIA GPU: a spoken turn in, interleaved text and 24 kHz speech
out, streamed. The weights are the bf16 release as stored, the bundled Liquid codec (`audio_tokenizer/`) included.
Checked on a GB10 (sm_121). The vision tower is not loaded.

```bash
hf download LiquidAI/LFM3-Omni-2B --revision 5797d07c125503bd4f4c0e7f915ae2d3d703e8de --local-dir "$HOME/models/lfm3-omni-2b"
zig build lfm3-omni-generate -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc
# one turn: a WAV in (any rate, resampled to 16 kHz), the reply as a WAV and its text
zig-out/bin/lfm3-omni-generate --model "$HOME/models/lfm3-omni-2b" --wav question.wav --out reply.wav \
  --system "Respond with interleaved text and audio. You are a helpful voice assistant."
# a server: POST /v1/audio/chat, GET /health
zig-out/bin/lfm3-omni-generate --model "$HOME/models/lfm3-omni-2b" --serve 127.0.0.1:8441 --system-file system.txt
```

## The endpoint

`POST /v1/audio/chat` takes JSON: `audio` (a base64 WAV, required), `system`, `stream` (default true),
`audio_temperature` (0.4), `audio_repetition_penalty` (1.2), `seed`, `max_new` (generated slots, a text token or a
speech frame each). With `stream` the answer is server-sent events as the reply is made:

```text
data: {"type":"text","delta":"I'm on it,"}
data: {"type":"audio","format":"pcm16","sample_rate":24000,"data":"<base64 s16le>"}
...
data: {"type":"done","text":"...","frames":87,"speech_s":6.96,"first_text_ms":36.0,"first_audio_ms":150.2,"total_ms":2160.4}
data: [DONE]
```

Without it, one JSON object with the same fields and `audio`, the whole reply as a base64 WAV. The server takes one
request at a time; it is the tool's own (not yet a route of `tensorfold-native serve`, whose speech route is text in).
Text is greedy, as the reference's `do_sample=False`; speech frames are sampled with the card's settings.

## What runs

- Log-mel (16 kHz, 400-point periodic Hann, 128 Slaney bins) as fp32 GEMMs on the device; the speech encoder's three
  stride-2 convolutions, its 17 LFM2 layers (3-tap conv, causal attention over 32 tokens) and the projector.
- The LFM3 backbone (24 layers: 7-tap gated conv, full attention with half rotary, sliding attention over 256), the
  prompt in chunks of 512 rows, then one row a slot replayed as a CUDA graph.
- The depth decoder: three bidirectional LFM3 layers over a frame's 12 codebook slots, three mask passes (4, 4, 4
  codebooks), each codebook's pick on the device (repetition penalty over the last 64 frames, then greedy or a
  Gumbel-max draw, exactly a softmax sample); the frame is one CUDA graph and one copy back.
- The Liquid codec's decoder: FSQ codes to latents, 6x upsampling, 8 LFM3 layers (sliding attention over 30 rows,
  RoPE in fp32), the spectrum head and the inverse STFT as an fp32 GEMM (window folded in); overlap-add on the host.
  The first frame is decoded as soon as it exists, then every 8; streamed chunks equal the whole decode.
- Matrix products: a few-row GEMV (a warp a weight row, up to 16 input rows in shared memory) for decode and the
  depth decoder, cuBLAS for prompt rows. Every op rounds to bf16 where the reference's op does.

## Receipt

GB10, 16 synthetic spoken prompts (macOS voices), a voice-assistant persona as the system text, against the reference
(`transformers` with the checkpoint's own code, bf16, same GPU):

| stage | check | result |
| --- | --- | --- |
| log-mel | cosine, max abs | 1.000000, 2.8e-4 |
| speech encoder | cosine (min of 16) | 0.99999 |
| projector | cosine (min) | 0.99990 |
| prompt | token ids | 16/16 identical |
| backbone prefill | cosine of every row (median, min of medians) | 0.99976, 0.99915 |
| text | teacher-forced greedy top-1 | 466/473 (98.5%); misses are near ties (ref margin <= 2 bf16 ulp) |
| depth decoder | on the reference's own states: logits cosine median / min | 0.999992 / 0.99995 |
| depth decoder | greedy code agreement on the reference's states | 94.4%; every miss within 2 bf16 ulp (15% of the reference's own top-1s are exact bf16 ties) |
| codes | teacher-forced, the whole loop | 89.7% |
| codec | the reference's codes to audio: SNR (median, min) | 36.3 dB, 24.1 dB; head rows cosine >= 0.9997 |

Replies with the card's audio settings (temperature 0.4, penalty 1.2), 16 turns:

| | reference runtime | TensorFold |
| --- | --- | --- |
| UTMOS22 strong, median (min) | 4.42 (4.24) | 4.42 (3.91) |
| speech vs its own text, WER (reference model's ASR) | 1.7% | 3.0% |
| first text, median | | 0.036 s |
| first audio, median | 0.236 s | 0.15 s (streaming client, 0.144 to 0.156 over two passes) |
| whole reply, median (6.6 to 7.0 s of speech) | 4.14 s | 2.0 to 2.2 s |

The reference's first audio is its first sampled frame plus the codec's decode of 12 frames (about 4 ms); ours is the
first PCM a streaming client receives (the first frame decoded alone). Each backbone row reads 2.9 GB of bf16 weights;
the few-row GEMV runs at 240 GB/s, and the row (about 15 ms) bounds the 6 text slots before the first frame.

`--dump DIR`, `--force DIR` and `--depth-check DIR` write and score the stages against a reference's dumps.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint). Use, changes and redistribution are allowed; commercial use is licensed only to an
organisation under US$10 million in annual revenue, and a qualified non-profit may use it for research or other
non-commercial purposes regardless. TensorFold's code stays Apache-2.0.
