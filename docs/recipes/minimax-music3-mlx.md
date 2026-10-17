# MiniMax Music 3 on MLX (`music3`)

Text and lyrics to a full song, 44.1 kHz stereo, on Apple silicon. Three models in a chain: an AR language model
(a Qwen3-8B body that draws 8 RVQ codes a frame, 25 frames a second, with a 4-layer depth decoder for codes 1..7),
a flow-matching DiT (36 blocks, width 2048) that turns the AR's frame states into latents, and the DAV decoder.
Packages: `src/tensorfold/families/music3/` (`prompt.py`, `model.py`, `generate.py`) and the CLI / server in
`src/tensorfold/music3.py`. Tools: `tools/music3_parity.py`, `tools/music3_ref_comfy.py`, `tools/music3_listen.py`,
`tools/music3_seams.py`.

Weights: Comfy-Org/MiniMax-Music-3's single files, read as they are (`text_encoders/..._pruned_bf16`, `diffusion_models/..._dit_fp16`,
`vae/minimax_music3_dav`). MiniMax-Music3 is under the MiniMax-Music3 Community License: a commercial product must show
"MiniMax-Music3" in its interface; above USD 20M yearly revenue it needs MiniMax's written permission. TensorFold
ships no weights.

Measured on a Mac Studio M5 Ultra (256 GB), MLX 0.32.3, with the GPU shared with other jobs (times are a few
percent noisy; each run held the machine's GPU lock).

## Commands

```bash
# a song (bf16 AR, fp16 DiT, averaged DiT windows: the defaults)
python -m tensorfold.music3 --caption-file song.caption --lyrics-file song.lyrics --seconds 120 --seed 7 \
    --out song.wav --stats song.json

# int8 AR weights (body, code-0 head and depth decoder at 8 bits, group 64)
python -m tensorfold.music3 ... --quant int8

# streaming: the WAV at --out grows while the song is made (unknown-length header until the end), --pcm gets raw
# interleaved 16-bit stereo PCM as it comes; --out - streams the WAV to stdout (logs then go to stderr)
python -m tensorfold.music3 ... --quant int8 --stream --pcm song.pcm --out song.wav

# the persistent server (Drama 3's JSON-lines protocol: requests on stdin, answers on stderr)
python -m tensorfold.music3 --serve [--quant int8]

# checks
python tools/music3_parity.py REF_DIR --caption-file C --lyrics-file L [--stages ar,probe,dit,dav] [--quant int8]
python tools/music3_listen.py song.wav=song.lyrics [--backend mlx|cpu] [--words-dir DIR] [--report all.jsonl]
python tools/music3_seams.py song.wav --stats song.json
```

### Server requests

Unchanged fields: `caption`, `lyrics` / `lyrics_file`, `wav`, `seconds`, `seed`, `ar_cfg`, `top_k`, `steps`,
`dit_cfg`, `id`, `progress`, `stats`. The answer: `ok id wav audio_s frames eos ar_ms dit_ms dav_ms total_ms rtf
first_audio_ms stream seed peak_memory_gb`.

Opt-in (an older client never sees a difference):

| Field | Effect |
| --- | --- |
| `"stream": true`, or a `"pcm"` path (as Drama 3's server: a `pcm` path asks for a stream unless `"stream": false`) | sequential DiT while the AR draws; `wav` is written as a growing WAV, `pcm` as raw interleaved s16le stereo, both as each piece is decoded; `first_audio_ms` is the time to the first piece |
| `"progress": true` with a stream | also `{"id":..,"progress":{"stage":"stream","done":seconds sent,"total":cap}}` |
| `"dit_mode": "sequential"` | diffusers' sequential windows without streaming |
| `"overlap": false` with a stream | run each DiT window in one go between AR frames (default true: its steps go on a second GPU stream; the audio is the same byte for byte) |

The answer to a request that used any of these also carries `pcm dit_mode windows quant`. The ready line adds
`"stream": true`, `"quant"` (what the server loaded; `--quant` is a load-time flag) and `"dit_modes"`.

## Quantization (`--quant`)

`none` (default), `int8`, `int8-d6`, `int8-d4`, `int6`, `int4`, or per-part overrides: `body=`, `head=`, `depth=`,
`dit=` (bits 2-8, 0 = keep), `group=` (32 / 64 / 128), e.g. `int8,dit=8`. MLX affine quantization (`nn.quantize`,
`QuantizedLinear`) applied at load; embeddings stay bf16.

### Quality: teacher-forced against the ComfyUI reference (`tools/music3_parity.py --stages ar --quant ...`)

The reference's own drawn codes are fed back, so every config sees the same 750 frames (30 s hip-hop song); the
bf16 row is the floor (MLX bf16 against PyTorch bf16).

| `--quant` | frame states cos (DiT condition) | code-0 cos mean / min | code-0 greedy agree (cond / guided) | top-50 overlap | depth cos min | depth greedy agree (mean of 7) | forced AR s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (bf16) | 0.999951 | 0.999973 / 0.99972 | 97.7% / 96.4% | 98.7% | 0.99993 | 97.1% | 16.5 |
| int8 (group 64) | 0.999889 | 0.999938 / 0.99862 | 98.0% / 95.1% | 98.4% | 0.99986 | 95.7% | 11.3 |
| int8, group 128 | 0.999878 | 0.999930 / 0.99831 | 97.3% / 95.2% | 98.4% | 0.99985 | 96.2% | 11.5 |
| int8-d6 (depth 6-bit) | 0.999826 | 0.999938 / 0.99862 | 98.0% / 95.1% | 98.4% | 0.99980 | 95.7% | 11.0 |
| int8-d4 (depth 4-bit) | 0.998841 | 0.999938 / 0.99862 | 98.0% / 95.1% | 98.4% | 0.99897 | 90.7% | 9.8 |
| int6 | 0.999418 | 0.999683 / 0.99430 | 95.6% / 94.4% | 97.4% | 0.99935 | 92.9% | 10.4 |
| int4 (head 8) | 0.993219 | 0.996598 / 0.96541 | 87.4% / 83.4% | 92.2% | 0.99262 | 77.9% | 8.6 |

int8 sits at the bf16 floor on code 0 and a little under it on the depth codebooks (95.7% against 97.1%); 4-bit
depth and anything 6-bit or below fall clearly off. The reference's drawn code ranks first (median) in every row.

### Speed and memory, 60 s hip-hop song, seed 7, 30 DiT steps

| Run | AR frames/s | AR s | DiT s | DAV s | total s | RTF | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (bf16), averaged | 40.1 / 40.0 | 37.6 / 37.7 | 34.6 / 38.0 | 2.0 / 2.0 | 75.5 / 79.2 | 1.26 / 1.32 | 27.6 |
| int8, averaged | 64.9 / 64.8 | 23.3 / 23.3 | 41.4 / 35.4 | 3.6 / 2.1 | 69.5 / 62.1 | 1.16 / 1.04 | 20.5 |
| int8-d6 | 58.2 | 25.9 | 42.9 | 2.1 | 72.3 | 1.21 | 20.3 |
| int6 | 71.3 | 21.2 | 41.4 | 2.1 | 66.0 | 1.10 | 18.6 |
| int8, dit=8 | 64.2 | 23.6 | 48.1 | 2.1 | 75.0 | 1.25 | 18.2 |

Two rows give two runs (the DiT time moves by about 4 s from run to run on this shared GPU). The AR reads
~22 GB a frame in bf16 and is bandwidth-bound; int8 takes it from 40 to 65 frames a second. MLX's 6-bit kernels
are slower than its 8-bit ones here (int8-d6 reads fewer bytes and runs slower), and an int8 DiT (compute-bound,
batched windows) is slower than fp16. Lyric accuracy (below) shows no loss either.

Recommendation: `--quant int8` (measured within noise on every check here). The default is left at `none` for
now because other jobs on this machine run this engine and a changed default would change their songs for the
same seed; flipping it is one line (`--quant` in `src/tensorfold/music3.py`).

## Streaming

`--stream` (CLI) or `"stream": true` (server) runs diffusers' sequential windows (the `minimax_music3` modular
pipeline in diffusers: 200-frame windows every 100 frames, each conditioned on its own frames, its first 172 latent
frames held to the previous window's latents re-noised to each step's level and restored at the end; decoded per
window, dropping 86 latent frames on the left and 258 on the right of every inner window). A window needs only its own
frames, so it runs while the AR is still drawing: the first audio arrives after 200 frames plus one window. The
unconditional branch is the condition of zero states (ComfyUI's), not diffusers' zero condition. Each window's 30
DiT steps are submitted on a second MLX GPU stream, one per AR frame (`overlap`, default on): the AR is bandwidth-
bound and the DiT compute-bound, and the GPU overlaps them; the audio is the same byte for byte as without overlap
(`--no-overlap`), and `--dit-mode sequential` without streaming gives the same audio again.

The level rule (divide by 5 x std when that is above 1) cannot see the whole song in a stream: it is applied as
pieces arrive with a divisor that never falls (`LevelRule`). Sequential windows tile 345 latent frames per 344.5 of
AR time, so a streamed / sequential song is about 6 ms longer per window than the AR timeline (0.4 s at 300 s), as
in diffusers.

| 60 s hip-hop, seed 7 | AR frames/s | total s | RTF | first audio s | no-stall start s | peak GB |
| --- | --- | --- | --- | --- | --- | --- |
| none, averaged (default) | 40.0 | 75.5-79.2 | 1.26-1.32 | = total | - | 27.6 |
| none, sequential (not streamed) | 39.6 | 84.1 | 1.40 | = total | - | 29.3 |
| none, stream, no overlap | 38.6 | 84.7 | 1.41 | 9.2 | 31.7 | 29.4 |
| none, stream, overlap | 36.0 | 80.8 | 1.35 | 10.6 | 27.7 | 29.5 |
| int8, averaged | 64.8 | 62.1-69.5 | 1.04-1.16 | = total | - | 20.5 |
| int8, stream, no overlap | 62.5 | 70.5 | 1.17 | 7.3 | 17.4 | 22.2 |
| int8, stream, overlap | 55.4 | 63.4 | 1.06 | 7.6 | 10.3 | 22.3 |

"No-stall start" is the earliest a player could start and never run dry (each piece there before the audio before
it has played). A stream is not real time on this machine (RTF > 1), so a live player has to start about 10 s
(int8, 60 s song) to 53 s (int8, 300 s) after the request, or play with gaps.

Seams (`tools/music3_seams.py`: spectral flux at each seam against the song's own flux distribution; the control is
the same positions in averaged-mode songs, which have no seam there):

| Songs | seams | flux percentile, mean | seams above the song's 99th percentile | flux / median | level step across, mean |dB| |
| --- | --- | --- | --- | --- | --- |
| sequential / streamed (11 x 60 s, 2 x 300 s) | 288 | 68.4 | 9 (3.1%) | 1.26 | 3.6 |
| averaged controls (19 x 60 s, 120, 180, 300 s) | 388 | 66.9 | 14 (3.6%) | 1.29 | 3.7 |

The seams are not distinguishable from any other moment of the songs by this measure. Lyric accuracy of streamed
songs is the same as averaged ones (below). Averaged windows stay the default: they are what the ComfyUI parity was
measured on, and streaming is no faster in total.

## Lyric accuracy

`tools/music3_listen.py`: Whisper large-v3-turbo (mlx-whisper) with word timestamps, once over the whole song and
once in independent 20 s chunks (5 s overlap); a lyric word counts when either pass heard it (one pass loses a
30 s window when Whisper hallucinates on an instrumental intro; chunks lose words elsewhere), aligned to the lyrics
with Drama 3's normalisation, a repeated chorus matched where it first comes. "Sung" accuracy is over the lyrics up
to the end of the last run of 4 lyric words heard in a row (a 60 s cap does not reach all verses). No source
separation (none is installed on the Mac); Whisper hears the full mix. Word and line timestamps go to
`<song>.words.json` (`words`, `words_chunked`, `lyric_words_timed`, `lines` with start / end per lyric line).

Same-seed A/B, 60 s cap, hip-hop and pop prompts, seeds 1-3 (6 songs a config). A streamed song draws the same AR
codes as the averaged one of its seed; only the DiT differs.

| Config | word accuracy | sung accuracy | precision | per song (sung) |
| --- | --- | --- | --- | --- |
| none, averaged | 0.834 | 0.867 | 0.955 | 0.99 0.99 0.73 0.75 0.74 1.00 |
| int8, averaged | 0.836 | 0.963 | 0.968 | 0.97 0.99 0.99 0.86 0.97 1.00 |
| none, streamed | 0.872 | 0.904 | 0.956 | 0.99 0.99 0.99 0.71 0.74 0.99 |

Per song the spread is large (one hip-hop seed scores 0.73 averaged and 0.99 streamed from the same AR codes), so the
three means are within noise of each other; nothing points at a loss from int8 or from streaming. The two pop seeds
near 0.74 in both bf16 runs are the AR's own (same codes, same score): it sang past some lines.

Other songs: demos (hip-hop anthem 1.00, pop chorus 1.00, cinematic instrumental: no lyrics), the earlier benches
(30 / 60 / 120 s caps: 0.99 / 0.99 / 1.00 sung), the 60 s speed runs above (0.97-0.99 sung for every config).

## Long songs

Lyrics: `long.lyrics` (459 words: 5 verses, choruses, a bridge) for 120 and 180 s, `long300.lyrics` (776 words) for
300 s. Seed 21. None of these ended on its own (the model drew no stop code); the earlier 120 s cap with shorter
lyrics stopped at 60.9 s, the pop demo at 69.6 s.

| Cap | Run | audio s | stopped | AR frames/s | AR s | DiT s | total s | RTF | first audio s | peak GB (AR peak) | sung accuracy (lyric words reached) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 120 | none, averaged | 120.0 | no | 39.3 | 76.6 | 70.6 | 152.4 | 1.27 | 152.4 | 27.7 (23.4) | 0.99 (266 / 459) |
| 180 | none, averaged | 180.0 | no | 39.0 | 115.8 | 106.0 | 229.0 | 1.27 | 229.0 | 27.8 (24.0) | 1.00 (409 / 459) |
| 300 | none, averaged | 300.0 | no | 33.0 | 227.8 | 211.2 | 450.8 | 1.50 | 450.8 | 28.0 (25.4) | 0.88 (702 / 776) |
| 300 | none, streamed | 300.4 | no | 31.5 | 238.7 | 205.0 | 460.1 | 1.53 | 10.2 (no-stall 166.7) | 31.4 | 0.86 (702 / 776) |
| 300 | int8, streamed | 300.4 | no | 50.5 | 148.9 | 182.1 | 346.1 | 1.15 | 8.0 (no-stall 52.7) | 24.2 | 1.00 (583 / 776) |

Memory stays flat with length: the averaged DiT at 300 s peaks at 28.0 GB (the AR's KV cache grows from 23.4 to
25.4 GB peak; the DiT runs after the AR, 8 windows a batch). A stream holds the AR and the DiT at once: 31.4 GB in
bf16, 24.2 GB with int8. The AR slows with length (40 frames/s at 60 s, 33 at 300 s: attention over a 7,800-row
cache), which is why 300 s runs at RTF 1.5.

## Parity with ComfyUI (bf16 / fp16, 30 s song)

| Stage | Result |
| --- | --- |
| Prompt ids | equal |
| AR frame states (teacher-forced with the reference's codes) | cos 0.99995 mean (0.99982 min) |
| Code-0 logits | cos 0.99997; greedy agree 97.7% (guided 96.4%); top-50 overlap 98.7% |
| Depth logits, codebooks 1..7 | cos >= 0.99993; greedy agree 95-98% |
| DiT one forward at sigma 1.0 / 0.5 / 0.1 | cos >= 0.99997 |
| DiT 30-step latent from the reference's noise and states | cos 0.99984 (0.99995 with `--comfy-rope`) |
| DAV waveform | cos 0.9999994, 50.5 dB |
| DAV time, 30 s of audio | MLX 1.26 s; ComfyUI (PyTorch MPS, fp32) 1.44 s |

## What is untested

- Listening by ear: every quality claim here is from logits, Whisper, and spectral seam statistics. int8 and
  streaming were not judged by a listener.
- int8 against bf16 MLX directly (the table compares each to the ComfyUI reference); the lyric A/B has 6 songs a
  config, too few to see a small difference.
- Source separation before Whisper (no separator weights on the Mac).
- Streaming over a live consumer: the Studio's Voice-tab player reads `pcm` / growing WAVs in mono; a stereo reader
  for music (`channels` 2 in the header, interleaved L R) is not written yet. The server test here checked the files
  (the PCM equals the finished WAV's body, the WAV header is valid, `--out -` gives the same bytes).
- Quantized DiT quality (int8 DiT was slower, so it was not pursued) and group sizes other than 64 / 128.
- Songs that end on their own past 70 s, songs above 300 s (the cap is 360 s), other languages, the `--dit-dtype
  bfloat16` path, CPU-only machines (tests are CPU-runnable; generation was run only on the M5 Ultra).
- A diffusers reference run of the sequential mode (it is implemented from diffusers' source, not checked against
  its output).
