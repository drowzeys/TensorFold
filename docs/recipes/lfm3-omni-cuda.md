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
  The first frame is decoded as soon as it exists, then every 8.
- Matrix products: a few-row GEMV (a warp a weight row, up to 16 input rows in shared memory) for decode and the
  depth decoder, cuBLAS for prompt rows. Every op rounds to bf16 where the reference's op does.

## Receipt

One GB10 (sm_121), Ubuntu 24.04, driver 580.173.02, CUDA 13.0 (nvcc 13.0.88, cuBLAS 13.1.1); the reference is
`transformers` 5.21.0 with the checkpoint's own code on PyTorch 2.14.1+cu130, bf16, the same GPU. The 16 turns of
`tools/lfm3_omni/utterances.txt` (macOS voices), a voice-assistant system text, identity codebook order, greedy text
and codes, through `tools/lfm3_omni/parity.sh`:

| stage | check | CUDA vs CUDA bf16 |
| --- | --- | --- |
| log-mel | cosine (min), max abs | 1.000000, 3.7e-4 |
| speech encoder | cosine (min of 16) | 0.99999 |
| projector | cosine (min) | 0.99990 |
| prompt | token ids | 16/16 identical |
| backbone prefill | row cosine (median of medians, min of medians) | 0.99982, 0.99978 |
| text | teacher-forced greedy top-1 | 463/473 (97.9%) |
| depth decoder | on the reference's own states: greedy agreement, logits cosine median / min | 94.9%, 0.999992 / 0.99988 |
| codes | teacher-forced through the whole loop | 89.3% |
| codec | the reference's codes to audio: SNR median (min); spectrum rows cosine (min) | 36.0 dB (23.6 dB); 0.99978 |

Text and code misses are near ties (within 2 bf16 ulps of the reference's pick), as on Metal.

Replies with the card's audio settings (temperature 0.4, penalty 1.2, seeds 1000 + i), the server and
`speech_chat_client.py`, the reference through `reference_speech.py` (its first audio is the first sampled frame plus
that frame's codec decode). Two rounds, alternated, with no other GPU work but a resident idle server:

| | TensorFold CUDA | reference runtime (CUDA bf16) |
| --- | --- | --- |
| first text, median | 0.032 s | |
| first audio, median | 0.139 to 0.144 s | 0.225 to 0.237 s |
| whole reply, median (7.0 s of speech) | 1.95 to 1.99 s | 3.28 to 3.35 s |
| UTMOS22 strong, median (min) | 4.42 (4.07) | 4.45 (3.83) |
| speech vs its own text, WER | 1.9% | 4.6% |

UTMOS and WER were scored on the Mac (the checkpoint's own ASR on MPS) from the GB10's replies. Each backbone row
reads 2.9 GB of bf16 weights through the few-row GEMV and bounds the six text slots before the first frame.

`--dump DIR`, `--force DIR` and `--depth-check DIR` write and score the stages against a reference's dumps.

## Licence

The weights are under Liquid AI's LFM Open License v1.0 (`license: other`, `lfm1.0` on the model card; the text is the
`LICENSE` file in the checkpoint). Use, changes and redistribution are allowed; commercial use is licensed only to an
organisation under US$10 million in annual revenue, and a qualified non-profit may use it for research or other
non-commercial purposes regardless. TensorFold's code stays Apache-2.0.
