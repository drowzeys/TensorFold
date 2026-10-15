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
pip install mlx-whisper            # for the end-of-text check (or faster-whisper, CPU); without either it is skipped
```

The routed experts (FP8 E4M3, 128x128 block scales) are dequantized and requantized to MLX 8-bit affine
(groups of 64; `--experts mxfp8` or `bf16` are the alternatives, see below); everything else stays bf16, the codec
float32. The converted folder is 39 GB; a run peaks at about 50 GB of unified memory (plus ~2.6 GB for Whisper).

## Usage

```sh
python -m tensorfold.drama3 --text "Hello there." --out hello.wav
python -m tensorfold.drama3 --text "<|speaker:0|> [whispering] Over here. <|speaker:1|> Where?" --out talk.wav
python -m tensorfold.drama3 --text "..." --ref-audio voice.wav --ref-text "what the clip says" --out clone.wav
python -m tensorfold.drama3 --text-file chapter.txt --seed 3 --out chapter.wav --stats stats.json
python -m tensorfold.drama3 --text "Hello there." --stream --out live.wav           # a WAV that grows as it is spoken
python -m tensorfold.drama3 --text "Hello there." --stream --out - | ffplay -nodisp -autoexit -   # WAV on stdout
```

- `<|speaker:0|>` .. `<|speaker:15|>` switch voices; `[bracketed directions]` (`[sighs]`, `[laughs]`,
  `[whispering]`...) steer delivery. Text without a tag is speaker 0.
- Dialogue: each speaker turn is put on a line of its own before the prompt is built (a tag that starts mid-line
  gets a line break before it), and long dialogue is chunked so every line opens on its speaker's tag. Drama 3
  reads turns written on one line badly: on three dialogues x 16 seeds, streamed with the check and the stop rule,
  12 of 48 takes dropped later turns and ran on for 10-24 s with turns on one line, 1 of 48 with each turn on its
  line (9/16 -> 1/16 on "<|speaker:0|> [whispering] Did you hear that noise downstairs? <|speaker:1|> [sighs] It's
  probably just the cat. Go back to sleep. <|speaker:0|> Cats don't knock on doors."; not streamed, with redraws,
  0 of 48). The prompt rows match the reference builder (`drama3_prompt.py`) for the same text, and teacher-forced
  main-token and code-head logits of a failing take match the PyTorch reference (KL ~0.002): this is the model's
  behaviour, not the port's. Those failures have no p(EOS) signal, so a streamed take cannot be stopped early;
  only the frame cap ends them.
- `--ref-audio/--ref-text` (repeatable) clone a voice from a clip and its transcript; `--ref-codes x.npy` reuses
  saved codes. Any sample rate or channel count; it is resampled to 44.1 kHz mono.
- Long text is split at sentence ends into chunks of about `--max-chars` (300). Without a reference, later chunks
  take the first chunk's own audio and text as their voice reference (when it looks like a clean reading), so the
  voice stays the same. A frame cap (about 4 frames a spoken character plus 5 s) bounds every chunk.
- Sampling defaults are Fish's: temperature 1.0, top-p 0.9, top-k 30, plus RAS (a main token repeating one of
  the last 16 is redrawn at 1.0 / 0.95 / 30). `--greedy` is argmax everywhere, for parity checks only: pure
  greedy loops and does not reach EOS.
- `--model` or `TF_DRAMA3_MODEL` point at the converted folder (default `~/models/drama-3-mlx`).

| flag | default | |
|---|---|---|
| `--verify / --no-verify` | on | end-of-text check of every chunk with speech recognition (below) |
| `--verify-retries N` | 2 | redraws of a chunk with missing words; the best take is kept |
| `--asr auto\|mlx\|cpu` | auto | Whisper large-v3-turbo on MLX (mlx-whisper) or the CPU (faster-whisper); `TF_DRAMA3_ASR` |
| `--stop-rule / --no-stop-rule` | on | stop a runaway on the model's own p(EOS) signal (below) |
| `--stream` | off | write audio while it is generated: a growing WAV at `--out`, or a stream on stdout with `--out -` (logs go to stderr) |
| `--stream-format wav\|pcm` | wav | `--stream` output: WAV (header with "unknown length" sizes, then 16-bit PCM) or raw s16le PCM; a growing `.wav` file gets its real sizes when done |
| `--first-window N` | 10 | `--stream`: frames in the first codec window (10 frames = 0.46 s of audio) |
| `--holdback S` | 0 | `--stream`: seconds of the newest audio kept unsent all along |

`--stats` JSON keeps its earlier keys and adds `first_audio_s` (time to the first audio sent; for a non-streamed
run, the whole synthesis), `stream`, `verify` (per chunk: words heard / matched, missing words, words after the
text, `action` ok / trim / redraw, `attempts`, `cut_s`, `asr_s`) and `eos_prob_max` (per chunk).

### From Python

```python
from tensorfold.families.drama3.generate import Drama3Generator, StopRule
from tensorfold.families.drama3.sampling import Sampling
from tensorfold.families.drama3.stream import Drama3Stream, StreamOptions, stream_wav, wav_stream, GrowingWav
from tensorfold.families.drama3.verify import Transcriber

gen = Drama3Generator("~/models/drama-3-mlx")
asr = Transcriber("auto"); asr.warm_up()                      # load Whisper once, share it

audio, results = gen.synthesize(text, references=None, sampling=Sampling(), seed=0, max_chars=300,
                                max_frames=None, verify=asr, verify_retries=2, stop=True)
# results[i].stats["verify"]: the chunk's check; results[i].eos_probs: p(EOS) at every step

stream = Drama3Stream(gen, text, references=None, sampling=Sampling(), seed=0, max_chars=300, max_frames=None,
                      options=StreamOptions(first=10, holdback_s=0.0), stop=True, verify=asr)
for chunk in stream:          # float32 numpy, 44.1 kHz mono, as soon as each codec window is decoded
    ...
stream.stats                  # first_audio_s, total_s, audio_s, frames, chunks[i] (finish, held_from_s, verify...)

for piece in stream_wav(gen, text, seed=0):    # bytes: WAV header, then 16-bit PCM -> a chunked HTTP response
    response.write(piece)
```

`Drama3Generator.iter_frames(rows, sampling, seed=, max_frames=, stop=)` yields each frame as it is drawn and
returns the `Result`; `generate_rows` (unchanged signature, plus `stop=`) runs it to the end.

### Persistent server (`--serve`)

`python -m tensorfold.drama3 --serve` (or `--serve 1`, as TensorFold's CUDA engine `drama3-generate --serve 1`)
loads the model, the codec and (unless `--no-verify`) Whisper once, warms the compiled decode step and the codec
windows, and serves until its stdin closes. The other command-line options (`--model`, sampling, `--asr`,
`--max-chars`, `--first-window`, `--holdback`, `--no-stop-rule`, `--no-verify`) set the defaults of the requests.

- stdin: one JSON request a line. Stdout is used only for audio when a request asks for `"pcm": "-"`.
- stderr: the protocol and the logs. Every protocol line is one JSON object starting with `{`; log lines never
  do. First `{"ready":true,"engine":"tensorfold-drama3-mlx","load_s":..,"verify":true,"stream":true}` (or
  `{"ok":false,"error":"load failed: ..."}`), then exactly one answer line a request, in order.

Request fields (only `text` is required):

| field | | default |
|---|---|---|
| `text` | what is said (speaker tags, [directions]; long text is chunked as on the command line) | |
| `wav` | write the take there (16-bit WAV; while streaming it grows and gets its sizes at the end) | none |
| `pcm` | raw s16le 44.1 kHz mono, written as it is made (`"-"`: stdout) | none |
| `stream` | stream while generating (the stop rule holds back a suspect tail; no redraws) | true when `pcm` is given |
| `ref_audio` + `ref_text` | a voice to clone (encoded once and cached by path and mtime) | |
| `ref_codes` + `ref_text` | a reference as saved codes (.npy, [10, frames] or [frames, 10]) | |
| `refs` | several references: `[{"audio" or "codes", "text"}]` | |
| `temperature`, `top_k`, `top_p`, `seed`, `greedy` | sampling | 1.0, 30, 0.9, 0, false |
| `max_frames` | frame cap a chunk | from the text length |
| `max_chars` | chunk size | 300 |
| `verify`, `verify_retries`, `stop_rule` | the end-of-text guards | true, 2, true |
| `first_window`, `holdback` | streaming | 10, 0 |
| `save_codes`, `stats` | also save the frames (.npy) / the full stats (JSON) there | |
| `id` | echoed back | |

Answer: `{"ok":true,"id","frames","eos","audio_s","prompt_rows","ref_frames","encode_ms","prefill_ms",
"first_audio_ms","total_ms","fps","rtf","wav","pcm","stream","chunks","finish","stop_rule","verify",
"peak_memory_gb"}` (`eos`: every chunk ended by itself, not at its frame cap; `finish` per chunk: `stop`,
`stop_rule` or `length`; `first_audio_ms` includes the reference encode; `verify` per chunk as in `--stats`),
or `{"ok":false,"error":"...","id"}` (a bad line: `{"ok":false,"error":"bad json"}`). The server keeps ~50 GB
(model, codec, Whisper); the MLX cache is capped at 8 GB and cleared after every request.

## End-of-text check

Drama 3 stops by sampling EOS. A draw that misses the stop keeps going past the end of the text (a *runaway*:
more words, or seconds of breath and noise), and a take now and then drops or garbles words. Two guards:

**Speech recognition (`--verify`, `synthesize(verify=...)`).** Every chunk's audio is transcribed with word
timestamps (Whisper large-v3-turbo; on MLX ~0.1 s for a 10 s chunk, ~0.5 s for 50 s; the model loads once,
~1 s) and aligned to the chunk's text (`verify.check_words`: speaker tags and [directions] are not words, numbers
are spelled out on both sides, case and punctuation ignored, near-misses and split/merged words such as
"Tensor Fold" match). Then:

- `ok` when the text is there and nothing follows it;
- `trim` when the text is there and words follow it, or more than 1.2 s of sound follows the last word (4 s when
  the text ends with a [direction] such as [laughs]): the audio is cut 0.3 s after the last aligned word (before
  the next word starts), with a 40 ms fade, and the frames with it (the codec is causal);
- `redraw` when under 85% of the words are heard, three or more in a row are missing, the last two are, or the
  last word is and it is a content word (not a short function word such as "to" or "you", not part of a number,
  which Whisper writes in more than one way; [directions] are never words). A split or merged word only matches
  when near-exact, so a dropped short word cannot hide in its neighbour. Then the
  chunk is drawn again with a new seed, up to `--verify-retries` times, keeping the best take.

**The model's own signal (`--stop-rule`, `StopRule`).** Mid-text p(EOS) stays far below 0.1 (in 152 measured
chunks, the only higher steps were within a word or a closing laugh of the end); near the end it rises, and the
step that draws EOS has p(EOS) near 1. A runaway is a take that put real weight on EOS, drew a sound instead and
went on. The first step past 0.55 frames a character with p(EOS) >= 0.1 is a *candidate end*; a take still going
64 frames (~3 s) later is stopped there and then (finish `stop_rule`) instead of decoding up to its frame cap, and,
without speech recognition, cut 14 frames after the candidate. In the measured takes no take that passed a
candidate went on more than 39 frames before drawing EOS. Length alone is no signal: short sentences with pauses or
digits read out run to 3-7 frames a character. Fish's own server has no stop logic beyond EOS and
`max_new_tokens` (2048).

Measured with `tools/drama3_eot_bench.py` (each chunk judged by an independent recogniser, faster-whisper beam 5
on the CPU: a *runaway* has words after the text or >2 s of sound after its last word; *missing* means under 90% of
the words heard, three in a row missing, or the ending missing):

| set | runs (chunks) | guard | runaway | missing words | wall time |
|---|---|---|---|---|---|
| standard: short, long (3 chunks), two speakers, clone, tags; seeds 1-8 | 40 (56) | frame cap only (before) | 0 | 0 | 261 s for 774 s of audio |
| | | `--verify --stop-rule` | 0 | 0 | 257 s (outputs identical: nothing to fix) |
| hard: "Yes.", "Okay. Fine. Let's go.", a closing [laughs], digits, a question, a short clone; seeds 1-16 | 96 | frame cap only | 1 ("Yes." drawn out into 2.6 s) | 1 | 149 s |
| | | `--verify --stop-rule` | 0 | 0 | 156 s |
| provoked: EOS logit -6 when drawing (`Sampling(eos_bias=-6)`), 5 texts x seeds 1-4 | 20 | frame cap only | 3 (5-8.6 s of sound after the text) | 1 | 102 s |
| | | `--stop-rule` only | 0 | 0 | 64 s |
| | | `--verify --stop-rule` | 0 | 0 | 69 s |

At Fish's defaults runaways are rare on ordinary text (0 in 56 chunks, also 0 in 20 runs at temperature 1.3 /
top-p 0.95 / top-k 60); the failure that does recur is a take that never reaches the end of a short text: half the
first takes of "Okay. Fine. Let's go." wander off before "go" with p(EOS) under 0.004 throughout (no model-side
signal) and run to the frame cap. The cap's retry redrew those before, unchecked; the check now redraws them
because words are missing (10 of 152 chunks needed a second or third take, all fixed). A check costs ~0.13-0.2 s a
chunk (max 0.97 s for a 25 s take).

## Streaming

`Drama3Stream` / `--stream` yields audio while the frames are drawn. Frames go to the codec in windows of 10, 10,
12, 32 and then 64 frames (Fish's schedule). The codec is causal, but its quantizer stacks 8 transformer layers
that each look 128 frames back, so decoding a window from a short overlap alone drifts (Fish's 32-frame overlap
gives ~11 dB SNR against the one-pass decode, 64 frames ~17-26 dB). `codec.StreamDecoder` instead recomputes the
quantizer latent over up to the last 1024 frames (~6 ms) and runs the convolutional decoder (the costly part) on
the new frames plus 16 frames of context: the streamed audio equals the one-pass decode of the same frames to
float noise (78-90 dB SNR), and the frames equal a non-streamed run with the same seed.

| CLI, 9.2 s of audio (one chunk) | first audio | total |
|---|---|---|
| not streamed | 3.06 s (the whole synthesis) | 3.1 s |
| `--stream` (first window 10 frames, 0.46 s of audio) | 0.21 s | 3.1 s |
| `--stream --first-window 4` | 0.14 s | 3.1 s |

The CLI warms the decode step and the codec windows up before it starts the clock (`stream.warm_up`, ~0.2 s, part
of start-up). The codec windows cost ~0.6 s per 10 s of audio against ~0.35 s for one pass, so frames are drawn
at ~70 a second while streaming (3.2x real time). A 47.5 s, three-chunk text streams with the first audio after
0.21-0.46 s and finishes in 15.8 s.

On provoked runaways (EOS logit -6), streaming ran 12-16 s past the text without the guard; with the stop rule the
tail was 0.9-1.2 s, with the stop rule and `verify` 0.3-0.6 s (one of three cuts came 0.25 s after audio had been
sent, reported as `late_s`).

Sent audio cannot be taken back, so streaming guards differently: the stop rule (on by default) holds back the
audio from a candidate end on (the last frame or two of a normal take, so the holdback costs nothing until the
end), stops a runaway 64 frames past it, and cuts it back there; with `verify`, the whole chunk is transcribed when
it ends and a tail after the last word is trimmed from what is still held back (`stats["chunks"][i]["late_s"]`
reports a cut that came after audio was sent). A take with missing words cannot be redrawn once it is playing; it
is reported. `--holdback S` keeps S seconds unsent all along, for a runaway with no EOS signal, at the cost of
that much later first audio.

## Speed (Mac Studio M5 Ultra, single stream)

| decoding | sampled (Fish defaults) | greedy |
|---|---|---|
| f623fb1a (mlx-lm's eager forwards) | 64-66 frames/s | 70-73 frames/s |
| now | 82-84 frames/s (3.9x real time) | 87-88 frames/s |

(600-frame runs of a 290-character prompt; real time is 21.5 frames/s.) A whole request (prefill, frames, codec,
check) runs at a real-time factor of ~0.33. What changed (`families/drama3/fast.py`):

- projections that read the same input are fused: linear attention q|k|v|z|b|a, full attention q|k|v, the router
  with the shared expert's gate|up and its gate, the routed experts' gate|up (one gathered matmul), the tower's
  w1|w3. Single rows give the separate products bit for bit; prefill multiplies by each part's rows on its own,
  because MLX picks other matrix kernels for narrow outputs and their rounding differs;
- every layer's single-row work is one `mx.compile`d function (the linear-attention layers whole, with their
  convolution and recurrent states passed in and out; full attention and the tower around the KV-cache update);
  the main-token draw (candidate logits, top-k/top-p, RAS, p(EOS)) and each code draw compile with their head;
- the next backbone step embeds only the main token and code 1, so it no longer waits for the last eight tower
  layers: they run on the GPU alongside it (+7%);
- one host sync a frame (the frame itself), with the next step already queued; sampling and the RAS window stay
  on the device.

Bandwidth floor: a frame reads 6.01 GB (backbone attention, shared experts and routers 2.86 GB bf16; the 8 active
of 256 routed experts 1.07 GB at 8 bits; tower layers 1.91 GB and code heads 0.15 GB bf16; candidate head rows
17 MB). The M5 Ultra reads 1.0-1.1 TB/s (6 GiB matrix-vector product), so the floor is 5.4-6.0 ms a frame
(166-186 frames/s); a frame now takes ~12 ms (backbone ~8.3, tower ~4.5 of which ~2.8 overlap). What is left is
latency, not bytes: a dependent GPU kernel costs ~5.7 us, and a frame is still several hundred of them in series
(the MoE block's routing chain makes it run at ~270 GB/s, the dense matvecs at 640-880 GB/s). Closing the gap
needs fused Metal kernels (router + top-k + experts) or fewer bytes (8-bit attention and tower weights, a quality
trade not taken here). `TF_DRAMA3_FAST=0` turns all of it off.

## Expert quantization

`tools/drama3_quant_kl.py` teacher-forces 200 frames of the oracle dumps through each conversion and compares
the logits with the bf16-expert conversion (`--experts bf16`):

| experts | sampled frames/s | weights in memory | main token KL (plain / clone) | main top-1 | code heads KL | heads top-1 |
|---|---|---|---|---|---|---|
| 8-bit affine, groups of 64 (default) | 83 | 41.6 GB | 0.0020 / 0.0020 | 99.5% / 98.5% | 0.0025 / 0.0030 | 97.4% / 98.1% |
| mxfp8 | 83 | 40.6 GB | 0.0019 / 0.0022 | 100% / 99.0% | 0.0029 / 0.0032 | 97.7% / 97.8% |
| bf16 | 75-78 | 71.8 GB | (reference) | | | |

mxfp8 matches 8-bit affine in speed and quality (its gathered matmul is ~8% faster in isolation, which the MoE's
routing latency hides), so the default stays 8-bit affine.

## Parity

`tools/drama3_parity.py` checks a converted model against the PyTorch reference's dumps (prompt rows, every
backbone and tower layer, logits, teacher-forced decoding, greedy frames, codec waveform). The reference run
on two devices (its own GPU and CPU runs) disagrees with itself as much as MLX disagrees with it: bf16 logit ties
and code-head margins of 0.06-0.25 flip greedy codes in the first frame either way. Notes:

- MLX's float32 matmuls and convolutions on M5 GPUs round like TF32 (~8e-4 relative), so the codec on the GPU
  reproduces the PyTorch codec at ~41 dB SNR (bit-near on the MLX CPU path) and reference encoding matches
  93-99% of codes.
- `TF_DRAMA3_MOE_ACT=fp8` emulates SGLang's FP8 expert activations, for parity runs (it turns the fused step off).
- `TF_DRAMA3_FAST=0` runs the plain mlx-lm forwards instead of the fused, compiled step; the parity numbers are
  identical either way.
- `tools/drama3_quant_kl.py` compares expert quantizations (teacher-forced logits, KL and top-1 against bf16).

Current numbers (unchanged from the first port, f623fb1a, to the last digit): plain / clone dump, backbone layers
min cos 0.999846 / 0.999865, candidate logits 0.999736 / 0.999784, tower prefill min 0.9918 / 0.9949, first-step
code-head logits min cos 0.99962 / 0.999594, teacher-forced main token 19/19 / 17/19 (15/16 where the oracle's
margin is above 0), codec 41.4 / 41.3 dB.
