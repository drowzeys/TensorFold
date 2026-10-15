# Fish Audio Drama 3 on Apple Silicon (MLX)

**Built with Fish Audio.** TensorFold runs Fish Audio Drama 3 text to speech locally, natively in MLX:
the Qwen3.5 MoE backbone (mlx-lm's model), the 13-layer codebook tower, the sampler (top-k / top-p with RAS)
and the Fish codec (decoder and encoder, for voice cloning). Nothing calls Fish's API.

## Licence

Drama 3's weights and codec are Fish Audio Materials under the **Fish Audio Community License v1.0**
("These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI,
INC. All Rights Reserved."). TensorFold's code is Apache-2.0; it never ships or redistributes Fish's weights,
and the converted folder it writes stays on your machine (the converter puts Fish's LICENSE and a NOTICE that
lists our changes into it). In particular:

- Show "Built with Fish Audio" wherever a product uses it (the CLI prints it).
- Research, non-commercial and small-company commercial use only; a **Hosted Service** (letting third parties
  generate speech or clone voices from text or samples they supply, by API or an app whose main purpose is TTS)
  needs a separate licence from Fish Audio.
- **Never clone the voice of a real person without their consent.** No impersonation, no use of outputs to train
  other speech models.

## Setup

```sh
python -m tensorfold.families.drama3.convert ~/models/drama-3 ~/models/drama-3-mlx   # once, ~1 min, needs torch for codec.pth
```

The routed experts (FP8 E4M3, 128x128 block scales) are dequantized and requantized to MLX 8-bit affine
(groups of 64; `--experts mxfp8` or `bf16` are the alternatives); everything else stays bf16, the codec float32.
The converted folder is 39 GB; a run peaks at about 50 GB of unified memory.

## Usage

```sh
python -m tensorfold.drama3 --text "Hello there." --out hello.wav
python -m tensorfold.drama3 --text "<|speaker:0|> [whispering] Over here. <|speaker:1|> Where?" --out talk.wav
python -m tensorfold.drama3 --text "..." --ref-audio voice.wav --ref-text "what the clip says" --out clone.wav
python -m tensorfold.drama3 --text-file chapter.txt --seed 3 --out chapter.wav --stats stats.json
```

- `<|speaker:0|>` .. `<|speaker:15|>` switch voices; `[bracketed directions]` (`[sighs]`, `[laughs]`,
  `[whispering]`...) steer delivery. Text without a tag is speaker 0.
- `--ref-audio/--ref-text` (repeatable) clone a voice from a clip and its transcript; `--ref-codes x.npy` reuses
  saved codes. Any sample rate or channel count; it is resampled to 44.1 kHz mono.
- Long text is split at sentence ends into chunks of about `--max-chars` (300). Without a reference, later chunks
  take the first chunk's own audio and text as their voice reference (when it looks like a clean reading), so the
  voice stays the same. A chunk that runs into its frame cap (about 4 frames a spoken character) is drawn again.
- Sampling defaults are Fish's: temperature 1.0, top-p 0.9, top-k 30, plus RAS (a main token repeating one of
  the last 16 is redrawn at 1.0 / 0.95 / 30). `--greedy` is argmax everywhere, for parity checks only: pure
  greedy loops and does not reach EOS.
- `--model` or `TF_DRAMA3_MODEL` point at the converted folder (default `~/models/drama-3-mlx`).
- From Python: `tensorfold.families.drama3.generate.Drama3Generator(model_dir).synthesize(text, ...)`.

## Speed (Mac Studio M5 Ultra, single stream)

About 62-64 frames/s while decoding (real time is 21.5 frames/s, so ~2.9x), first frame after ~0.25 s, a
whole request (prefill, frames, codec) at a real-time factor of 0.37-0.45. A frame costs ~10.5 ms of backbone
and ~5 ms of tower and sampling.

## Parity

`tools/drama3_parity.py` checks a converted model against the PyTorch reference's dumps (prompt rows, every
backbone and tower layer, logits, teacher-forced decoding, greedy frames, codec waveform). The reference run
on two devices (its own GPU and CPU runs) disagrees with itself as much as MLX disagrees with it: bf16 logit ties
and code-head margins of 0.06-0.25 flip greedy codes in the first frame either way. Notes:

- MLX's float32 matmuls and convolutions on M5 GPUs round like TF32 (~8e-4 relative), so the codec on the GPU
  reproduces the PyTorch codec at ~41 dB SNR (bit-near on the MLX CPU path) and reference encoding matches
  93-99% of codes.
- `TF_DRAMA3_MOE_ACT=fp8` emulates SGLang's FP8 expert activations, for parity runs.
