# Drama 3 voices

Drama 3 can speak in a reference voice. The prompt carries a clip's codec codes and the clip's transcript, the way the pinned reference (sglang-omni `prompt.py`, `build_prompt_rows`) builds it.
The server does not encode audio during a request. A clip is prepared and encoded once, offline, into a voice file.

## Whose voice

Encode only a voice the model invented, your own, or one whose owner has consented on record. Never clone a real person without that consent.
Each voice file says which, in `source`, and the server refuses a file that does not say. `source` is the operator's attestation: nothing checks the consent behind it.

Voice files written before `source` existed are refused too. Add an accurate `source` to each before restarting the server; do not guess one.

## Prepare a clip

```sh
python3 tools/zig/drama_voice_clip.py take.wav --out clips/NAME.wav --text take.txt
python3 tools/zig/drama_voice_clip.py take.wav --out clips/NAME.wav --asr WHISPER_DIR
```

The tool reads WAV, FLAC, Ogg and MP3, and writes 16-bit mono 44.1 kHz.

- **Level.** Aimed at -20 dBFS speech level, the mean power of the loud 10 ms frames. When that gain would take a peak past -1 dBFS, the clip gets less gain and comes out quieter than the target; the report gives the level reached.
- **Cut.** The tool sees loud and quiet 10 ms frames, not words. A frame more than 25 dB under the speech level is quiet, and a quiet run of at least 0.15 s is a pause.
  - The clip starts 0.25 s before the first loud frame.
  - When the take runs past 30 s, the clip ends inside a pause 10 to 30 s on. It prefers the last pause of at least 0.4 s (a sentence's end, usually) to a later, shorter one.
  - Speech quieter than the threshold, such as a trailing breathy word, can read as a pause. Listen to the clip against its transcript before encoding it.
- **Refusals.** A take is refused when:
  - it is shorter than 10 s, silent, or clipped (3 or more runs of 3 samples held at its top or bottom value), or it starts inside a loud frame;
  - the kept clip, counted from its first loud frame to its last plus the 0.25 s edges, would be shorter than 10 s, or there is no pause to cut at. Pauses count toward that duration;
  - its background is less than 30 dB under the speech, on the whole take and again on the kept clip;
  - its pauses hold a steady tone, music or hum (an octave band from 100 Hz to 6.4 kHz with spectral flatness under 0.1);
  - the median pitch of its stretches of speech spreads more than 6 semitones between the 20th and 80th percentile.
- **Limits.** The gates are cheap.
  - The speaker check hears pitch only. It misses two speakers of a similar pitch, and a speaker who moves between very different registers can trip it.
  - Music or noise more than 60 dB under the speech is not checked.
  - The noise floor is read from the quietest tenth of the kept clip, so a clip with almost no pauses reads as noisy.
- **Transcript.** It must be the exact words of the clip as cut.
  - `--text` copies the take's transcript to `NAME.txt`, and is refused when loud frames are cut away.
  - `--asr` runs a local Whisper model (`transformers`, on the CPU) over the cut clip and writes `NAME.draft.txt`. Correct the draft against the clip, then save it as `NAME.txt`. The encoder refuses a `.draft.txt`.
  - `--asr` refuses to replace a clip whose `NAME.txt` exists, because that transcript would no longer match. Remove it, or use another name.

## Encode a clip

```sh
python3 tools/zig/drama_voice.py --checkpoint DRAMA3_DIR --audio clips/NAME.wav --text clips/NAME.txt \
  --out voices/NAME.json --source-kind owner --source-note "my own voice, recorded 2026-10-16"
```

- `--source-kind` is `model-invented`, `owner` or `consented`. `--source-note` says where the voice comes from; for `consented`, where the consent is recorded.
- The encoder:
  - `--reference SGLANG_OMNI_CHECKOUT` or `--native BINARY` chooses it; the two exclude each other.
  - Without either, the tool uses the `tensorfold-native` on `PATH`, if its `--help` lists `voice encode`. Older builds serve speech but cannot encode, and the tool refuses them rather than run them.
- `--reference` runs the reference server's own encode for a `references[].audio_path`: `load_audio` at the codec's 44.1 kHz in mono, then `FishReferenceEncodeHook` under bfloat16 autocast on CUDA (the reference's default `reference_encode_dtype`). With `--device cpu` it encodes in float32. A 15.7 s clip takes about 2.6 GB of device memory and 10 to 20 s, most of it loading `codec.pth`.
- Clips longer than 60 s are refused, as the reference refuses them. Leading and trailing white space in the transcript is dropped.

Without Python, the native binary encodes a WAV clip (PCM or IEEE float, channels averaged) on CUDA:

```sh
tensorfold-native voice encode clip.wav --model DRAMA3_DIR --text "the clip's exact words" --name NAME --out voices/NAME.json
```

It runs the same encode in BF16 with the reference's roundings, in under 1 GB of device memory and about 4.5 s for a 50 s clip.
Its codes are not yet identical to the reference's: on three clips 2.5 to 3 % of semantic codes and 3 to 14 % of residual codes differ.
It resamples any common rate up to 384 kHz to 44.1 kHz.
It refuses a rate whose resampling filter would pass 2048 taps or 2^20 coefficients (44056 Hz, for one), a truncated file or chunk, a partial sample frame and a malformed WAVE_FORMAT_EXTENSIBLE header.
Run alone it writes no `source`, so `--voices` refuses the file; `drama_voice.py --native BINARY` runs it and adds one.

A voice file is JSON with these fields:

- `format`: `drama3-voice/1`.
- `text`: the transcript.
- `source`: `{"kind": "model-invented" | "owner" | "consented", "note": "..."}`. The note must hold more than ASCII white space; the tool and the loader both check this.
- `vq_codes`: the `[10, frames]` code grid.
- `codes_sha256`: the grid as little-endian int32, codebook-major.
- Provenance: `audio_sha256`, `frames` and `encoder` (reference commit, `codec.pth` digest, dtype, device, peak memory).

Voice files and clips hold a person's voice: keep them out of source control.

### Several references in one voice

A voice can hold several reference passages, spoken into the prompt in order. `drama_voice.py combine` joins voice files into one:

```sh
python3 tools/zig/drama_voice.py combine voices/A.json voices/D.json voices/C.json --out voices/NAME.json \
  --source-note "three takes of my own voice"
```

- The result is `format` `drama3-voice/2`: `source` once, and `references`, a list of entries that each hold a passage's `text`, `vq_codes`, `codes_sha256`, `frames`, `audio_sha256` and `encoder`, copied unchanged from the files named.
- A `drama3-voice/2` input adds all its references, in order. Every input must carry a `source` the loader accepts, with a known `kind` and a non-blank `note`, and all of one `kind`; the combined file gets that kind and the note given. Codes that do not match their digest, an `audio_sha256` that is not 64 lowercase hex digits, or fewer than two references in all are refused.
- The prompt is the one a request builds when it sends the first passages as `references` and names a voice holding the last: `drama3-voice/2` with A, D, C equals a request with references A and D for the voice C, row for row.
- A single-reference file stays `drama3-voice/1` and builds the prompt it built before. `/2` is a new format so that a server that reads only `/1` refuses the file instead of using one of its passages.
- The loader checks each entry as it checks a `/1` file, plus a 64-hex `audio_sha256`; its refusals name the entry (`references.1: codes_sha256 does not match vq_codes`).
- At startup, a voice whose passages leave no prompt row for an input stops the server: `voice references too long: NAME.json's 3 references take R prompt rows before any input, and a prompt may take M`. A request whose input then does not fit is refused as any overlong prompt is.

## Serve voices

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --default-voice NAME
```

- `--voices DIR` reads every regular `NAME.json` file in `DIR` once, at startup. A relative `DIR` is resolved from the server's working directory, not from the model's directory, so a launcher that starts the server elsewhere should pass an absolute path.
  - Names are 1 to 64 characters of `A-Z a-z 0-9 . _ -` and start with a letter or digit. `default` is reserved.
  - Hidden files, other extensions, directories and symbolic links are skipped.
  - A file with a bad name, an unknown format, no `source`, a transcript the reference would refuse, codes that do not match `codes_sha256`, or references that fill the prompt stops the server from starting.
  - Requests only name voices; no request path reaches the file system.
- `--default-voice NAME` is used when a request names no voice, or names `default`. Without it, those requests use the model's built-in voice and are unchanged.
- `GET /v1/voices` lists the loaded voices, OpenAI-list style. There is no upload route, because encoding stays offline.

```json
{"object": "list", "data": [{"object": "voice", "name": "NAME", "references": 1, "frames": 339, "seconds": 15.74, "source": {"kind": "model-invented"}}]}
```

## Requests

```json
{"model": "drama-3", "input": "The next train departs at nine.", "voice": "NAME"}
```

`references: [{"text": ..., "vq_codes": [[...], ...]}]` sends references inline, as the reference accepts them. A named voice's references follow the request's own references.
Refusals carry the reference's messages, in its order.
Some refusals are the native server's own:

- Reference audio (`audio_path`, `ref_audio`, `audio`, `data`) is refused: encode it with the tool and send `vq_codes`.
- Unknown reference fields are refused, not dropped.

## Expressiveness

The model card's documented controls are written in the input text:

- Point tags in square brackets, such as `[sighs]` and `[laughs]`, placed where the event should land. The card shows those two and lists no full vocabulary.
- Speaker tags. Only `<|speaker:0|>` is served, as in the reference.

The prompt builder treats point tags as ordinary text, so they reach the model unchanged.
Pacing also follows the reference clip and the punctuation.

### Allowed controls

```sh
tensorfold serve DRAMA3_DIR --allowed-controls 'concerned,curious,warmly,sighs,small laugh,laughs'
```

With `--allowed-controls LIST` (a comma list), a speech input keeps only the delivery controls on the list. Every other one is taken out before anything reads the input: every draw, the first one included, the short-reply count and the `¶` marks all see the filtered text.

- **Normalized names.** Names in the list and in the input compare after the same normalization: Unicode NFC, then lower case (any script), white space runs (tabs included) as one space, trimmed, and trailing `!` `?` `.` `…` dropped. `[Warmly]` and `[ small   LAUGH! ]` match `warmly` and `small laugh`. A listed control stays, rewritten in its normalized form (`[warmly]`). A list name with `[` or `]` in it stops the server at startup.
- **What a control is.** A `[…]` span is a candidate when it closes within 120 bytes and is not:
  - right after a letter, digit, `_`, `]` or `` ` ``: an index such as `x[i]` or `m[1][2]`, or quoted syntax such as `` `[pleased]` ``;
  - right before `(`: a Markdown link label, `[the docs](http://x)`.

  A span right after a control that was a candidate is still one, so stacked controls (`[sad][whispering]`) go one by one. A candidate that is not listed is a control only when every word of it (split at spaces, commas and hyphens) is in the shipped vocabulary, `zig/src/server/drama_cues.txt`, and at least one of those words is a cue rather than a modifier (`a`, `with`, `little`, …). The vocabulary holds emotions, sounds and pauses, and delivery words, taken from Fish Audio's Drama 3 and S2 emotion-control pages and the Drama 3 technical report. `[Pleased!]`, `[PLEASED]`, `[ softly, whispering ]`, `[relieved, with a little smile]` and `[takes a breath]` are controls.
- **Not controls.** Anything else stays as sent: names (`[John]`), notes (`[see note 3]`, `[TODO: fix]`), links, indexes, digits, a span longer than 120 bytes, an unclosed `[`, and free-form briefs with words outside the vocabulary (`[barely holding it together]`). Such a brief then reaches the model even when it is not listed. A bracket kept as text opens no control inside it, up to its matching `]` with nested brackets counted, so `[see [note] [sighs]]` stays whole. When the bracket has no matching `]`, it protects only up to its first `]`, so a stray `[` cannot shield the rest of the input.
- **Limits.** This policy is a heuristic, not a parser:
  - Bracketed text made only of vocabulary words is taken for a control even when it is meant as content. A person called `[Joy]` and a note reading `[calm]` are both removed.
  - Code is protected only by the character right before the `[`. `x[i]` stays, but `array [pause]` with a space before the bracket loses `[pause]`, and so does a control inside a Markdown code block or fence that does not touch a backtick.
  - Names compare in lower case, not with full Unicode case folding.
- **Scoped tags.** Opening and closing `<whisper>`, `<emphasis>`, `<soft>`, `<fast>`, `<slow>`, `<stress>` and `<forceful>` tags always go, their names matched in any case (`<WHISPER>`), and the text between them stays. That includes tags inside brackets: `[note <soft>quietly</soft>]` becomes `[note quietly]`. A control's name is read without its tags, so `[<soft>Warmly</soft>]` is the listed `[warmly]`.
- **Removal.** A control goes together with the white space on one side of it. White space here is ASCII: spaces, tabs and line breaks. Where both sides have white space, the run with more line breaks stays (the one before it on a tie), and white space before a control at the start of the input goes with it. `Hi [sighs] there` becomes `Hi there`, a blank line after a control stays, and `word [sighs].` becomes `word .`, as in a plain redraw. A plain redraw with `--silence-redraw-plain-scope controls` then starts from the filtered text and drops the listed controls too.
- **Count.** The server logs how many controls and tags each request lost, never what they were. Every audio reply carries `X-Controls-Stripped: N`, 0 included, in its head (a stream's too).
- **Off.** Without the flag, every request is served exactly as before, with no header.

## Frame limit

One synthesis generates at most 1024 frames, about 47.6 s at 21.5 frames a second. On CUDA, `--max-frames N` (1 to 8191) raises the limit, so one synthesis can speak a whole long reply with no joins:

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --default-voice NAME --max-frames 4096
```

- **Context.** The context holds the prompt and every frame, so `--context` (default and at most 8192, the tower attention's key limit) must exceed N. Prompts may take `min(4096, context - N)` rows. At N = 4096 that is 4096 rows; a voice reference takes one row per frame of its clip (339 for a 16 s clip) plus its transcript's tokens.
- **Memory.** The codec's activation planes hold a bounded decode chunk plus causal halo: 22,413,312 device bytes in total once the frame limit reaches 19. QKV history, RoPE, codes and PCM still grow with the utterance limit, by 57,512 device bytes per frame; causal carry remains fixed. At 4096 frames, bounded activations save 4,809,424,896 bytes versus the full-plane layout. The server's PCM buffer grows by 8 KiB a frame. When buffers do not fit the memory budget, startup fails with `OutOfDeviceMemory`.
- **Default.** Without the flag, the generation limit stays unchanged; bounded activation scratch also applies to the default limit.

With `--long-form`, each segment may use up to N frames, and a reply may use 6000 frames in all. Three English replies of 686 to 1483 characters, spoken in a reference voice, took 1.44 to 1.56 frames a character, pauses included. So `--long-form-chars` at about N / 2 leaves a segment room to spare. For example, `--max-frames 4096 --long-form --long-form-chars 2048` speaks inputs of up to 2048 characters in one synthesis. `--long-form-chars` keeps its default whatever N is.

## Guidance

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --default-voice NAME --cfg-gamma 2
```

On CUDA, `--cfg-gamma G` (0 for off, or 1 to 5; default 0) applies classifier-free guidance to a referenced prompt's semantic head. The main token of each frame is drawn from l_c + (G − 1)(l_c − l_u) instead of l_c: l_c are the logits of the prompt as sent (the voice's references, then the input), and l_u are those of the same turn as a plain prompt, the one the built-in voice gets. So the draw moves away from what the model says without the references. The mix covers all 4097 semantic logits, the end-of-speech logit included, so guidance also moves where a draw ends. It comes before the sampler, which is unchanged and not greedy: a draw seeded per request, at temperature 1, top-p 0.9 and top-k 30, with the repetition-aware redraw (RAS: a main token that repeats one of the last 16 frames' is replaced by a second draw at top-p 0.95). The tower, and with it codebooks 1 to 9, still reads the referenced prompt's hidden row.

- **The second sequence.** The engine keeps l_u in a second backbone sequence that shares the model's weights: 0.33 GB at context 8192 (the KV caches of the 10 attention layers and the states of the 30 DeltaNet layers). It prefills the plain prompt before the referenced one, and then reads every frame the referenced one reads. That is one more backbone run per frame, and the server's startup line gives the sequence's size.
- **`--cfg-every K`** (default 1) would run the second sequence only every K frames, reading the K frames in one pass, its last logits guiding the draws in between. Only K = 1 is accepted for now: tests check that a catch-up run reads every frame once and in order, but nothing has yet compared its logits with K runs one frame each, so the server refuses K > 1 at startup.
- **`--cfg-batched`** (needs guidance on) decodes each frame's two runs as one batch of two rows. The projections (DeltaNet and attention in and out, the router and the shared expert) read their weights once for both rows; each row keeps its own conv, DeltaNet and KV state and its attention, and the routed experts are read once for an expert both rows pick and once each otherwise, so a frame reads 8 to 16 experts a layer. Each row goes through the same arithmetic as its own decode, so its numbers, and so the draws, should equal the unbatched ones bit for bit: `tf-cuda-test drama-gemv-rows` and `tf-drama-cfg-pair` check this on a GPU. It needs a kernel image with `drama_gemv_bf16_r8_o0`; the server refuses to start without one.
- **What is guided.** A prompt with audio rows: a voice's references, or with `--long-form`, earlier segments. A built-in-voice prompt with none is the plain prompt already, so it is drawn unguided. Each draw is guided by the plain prompt of what it speaks, so a plain redraw's guide has no cues either, and each long-form segment's guide is that segment's own text.
- **G = 1** draws from l_c unchanged. The server then sends the engine nothing to guide with, and the draw is the unguided one.
- **`--cfg-force-guided`** (a diagnostic; needs guidance on) guides at G = 1 too: the second sequence runs and the mix adds 0 × (l_c − l_u) to every logit. Each draw must then come out as the unguided one, codes and audio, which checks that running the second sequence beside the referenced one leaves the referenced one's draws alone. It costs what guidance costs, so a redraw that depends on the time left can still differ.
- **Per request.** With `--cfg-allow-request-gamma`, a request's `X-Cfg-Gamma` header (1 to 5) sets its G; without the header, `--cfg-gamma` applies (1 when it is 0). Any other value is a 400. This is for sweeps: one server can render every G.
- **Response header.** With guidance on, every reply carries `X-Cfg-Gamma`, the G it was served with.
- **Engine log.** With guidance on, the engine logs one line a synthesis, guided or not: its frames, audio seconds, wall seconds and their ratio (`rtf`), whether it was batched, the plain prompt's rows, the second sequence's runs and rows, and the seconds they and the mixes took (`guidance_s`). Batched, `guidance_s` leaves out the decodes the sequences share.
- **Off (0) is the default.** The engine then makes no second sequence, the server neither reads `X-Cfg-Gamma` nor sends it, and every reply is as before. Text models refuse these flags.

## Expert mask

Each of the backbone's 40 layers routes a row to 8 of its 256 experts. On CUDA, `--expert-mask FILE` makes only some of them eligible, to try a pruned model's routing without pruning the checkpoint:

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --default-voice NAME --expert-mask kept.json
```

- **File.** A JSON object from layer index to the expert ids that layer keeps: `{"3": [3, 4, 5, 7, ...], "4": [...]}`. A layer the file leaves out keeps all 256. The server refuses the file at startup when a key is not a layer from 0 to 39, an id is not from 0 to 255 or repeats, or a layer keeps fewer than the 8 a row picks.
- **Routing.** A masked expert's router logit counts as -inf, so the softmax runs over the kept experts and the 8 most probable of them are picked and renormalized, as a checkpoint pruned to them routes (drama3-small's runtime drops the pruned experts' router rows and renormalizes the picks). A masked expert is never picked, even where every kept expert but one has a probability that rounds to 0. The weights, the shared expert and the per-row compute are unchanged: this tries the routing, not the smaller memory.
- **Scope.** Only the backbone's routed experts are masked. The codebook tower (its four shared and nine residual layers) has no routed experts: each layer has one dense feed-forward.
- **Generating a mask.** `tools/zig/drama_expert_mask.py MASS.json --ratio R [--whole 0,1,2,36] --out kept.json` keeps, in each layer of a routing-mass file `{"layer": [256 masses]}`, the ceil(R × 256) experts of most mass (at least 8, the lower id on a tie); `--whole` layers keep all.
- **Default.** Without the flag, every expert is eligible and the router's arithmetic is the one it was. The startup line names the mask's total, such as `routed experts masked to 4640 of 10240`.

### Compact experts

A mask saves no memory. `tf-drama-quant --compact-experts kept.json` writes a checkpoint holding only the kept experts ([docs](quantization.md#drama-3-compact-routed-experts)), FP8 or NVFP4, and the server loads it like any other Drama 3 directory:

- **Map.** The index's `metadata.drama_experts` lists, for every one of the 40 layers, the original ids of the experts it holds, ascending; layers may hold different counts. Each layer's experts are uploaded in that order, so device memory holds only them. A layer's format is its first held expert's.
- **Routing.** The router keeps its full 256 rows. A row is routed exactly as `--expert-mask` with the same ids routes the full checkpoint: -inf for the experts the layer does not hold, softmax, the 8 most probable, renormalized. The picks and weights are those original ids and values; the router then also writes each pick's slot in the layer's held experts, which the expert kernels read.
- **Refused.** The server refuses a map that is not an object of the 40 layers, an id outside 0 to 255, an id listed twice or out of ascending order, a layer holding fewer than 8, and a stored expert tensor the map does not list; a held expert without its tensors is a missing tensor. `--expert-mask` on a compact checkpoint may narrow a layer's held experts further, and is refused if it keeps one the layer does not hold.
- **Default.** A checkpoint without the map loads and routes as before. The startup line names the total, such as `routed experts held 4640 of 10240`.

## Silence guard

A synthesis can collapse: the model stops speaking and generates silence until its end of speech or the frame limit. In the default voice that silence is near-digital: one measured reply did this for 64 s; another stayed silent for 37 s until the frame limit. It can also be a steady hum mostly between -65 and -50 dBFS, like a recording's noise floor: one reply in a cloned voice spoke for 1.3 s and hummed for the next 30 s, and the built-in voice, asked to speak `[quietly, almost whispering]`, said one word and hummed for 97 s. The model can also stall and then recover: one reply was silent for 7.4 s and then spoke the rest of its text correctly.

So the engine decodes audio while it generates, every 4 frames for a stream and every 16 frames (0.74 s) otherwise. By default it ends a synthesis only once 10 s of frames in a row are below -40 dBFS (an RMS of 0.01), so a stall that has not recovered by then is treated as a collapse. Shorter stalls are kept as they are: the server does not trim them. `--silence-onset-s` (below) adds an earlier cut for a synthesis that has voiced less than 1 s.

- The audio up to the frame that completes the 10 s is kept, the silent run included. That is the same frame for a stream and a whole reply of the same generation: each chunk is checked before a stream's client receives it, and frames decoded past the cut are dropped. The reply carries `X-Finish-Reason: silence`: a header on a whole reply, a trailer on a stream.
- Silence that completes 10 s only in the last frames still counts. It outranks both the end of speech and the frame limit, because the reply collapsed however generation stopped.
- With `--long-form`, a segment ended this way ends the reply, as a clipped one does.
- **Whole replies decode in chunks of 16 frames,** with the incremental decode that streams use. This was measured on the 48-request default-voice capture against the previous build:
  - All 46 normal replies were byte-identical (two passes of 23), including the one with the recovering 7.4 s stall. 9 of 9 streams matched.
  - The reply with the 37 s silence was cut in both passes at 20.8 s, 10 s into its stall. What remained was a byte-identical prefix of the previous build's audio. That build cut at the end of a 16-frame decode chunk; the guard now cuts at the frame that completes the 10 s, at most 15 frames (0.7 s) earlier.
- **The -40 dBFS floor** was chosen on 308 renders of the default voice, model-invented voices and voices cloned from one recording, replayed offline:
  - Each of the 17 collapses has 10 s with no frame above -51.7 dBFS. Each of the other 291 renders has a frame at -19.2 dBFS or louder in every 10 s, and none has a run below -40 dBFS longer than the recovering stall (7.5 s at this floor).
  - A -60 dBFS floor missed 7 collapses into a noise floor, whose frames kept rising above -60 dBFS and restarting the run, and ended an 8th only 15.4 s late, once its hum had sunk below -60 dBFS for good. At -40 dBFS each ends 10 s after it began.
  - Whispered and soft speech stays clear of it. Eleven renders in fr2-cut and the built-in voice, of `<whisper>`, `<soft>`, `[quietly, almost whispering]`, `[sighs]`, a bedtime story and speech turning into a whisper, each 17-35 s, all ended `stop`. Some whispered words peak below -40 dBFS (the quietest at -50.8), but louder words follow within 2.5 s at most, and every 10 s holds a frame at -34.8 dBFS or louder.
  - Speech quieter than -40 dBFS for 10 s is taken for a collapse. Nothing else measured per frame tells the hum from a pause: it is nearly as periodic as voiced speech, its spectrum is less flat than the room tone between words, and its semantic codes (re-encoded from the audio) vary from frame to frame like a pause's, where digital silence repeats one or two codes.

### Onset guard

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --silence-redraws 2 --silence-onset-s 3
```

Some collapses come right after a word or two: a reply says "Listen" and then hums, and the 10 s rule ends it 10.6 s in. Of the 189 collapses measured below, 42 had voiced less than 1 s before the run that ended them. 101 had spoken 3 s or more first, and this rule does not reach those. `--silence-onset-s S` (CUDA; 0 for off, or 1 to below 10; default 0) also ends a synthesis at the frame where S seconds of frames in a row are below -40 dBFS, if it has voiced less than 1 s in all at that point (counting every frame at or above -40 dBFS). After 1 s of voice only the 10 s rule applies. The finish reason is `silence`, as after a 10 s trip. Each long-form segment is its own synthesis, so the rule applies again from the start of each segment.

- **S = 3 is the setting measured.** Offline replay, in the engine's frame accounting (the 10 s rule at 216 frames of 2048 samples, S = 3 at 65, whole frames only), over 1480 draws: 1450 unique renders and 30 later long-form segments. The segment figures are approximate, because they were cut from the joined output:
  - It ends 47 of the 189 collapses: the 42 above, and 5 that stalled early before collapsing later. Their audio stops at a median of 3.62 s (at most 4.64 s), where the 10 s rule stops it at a median of 10.63 s. Generation stops at the decode that hears the cut: a median of 3.76 s for a stream and 3.72 s for a whole reply.
  - It also ends 8 of the 1291 other draws. Two of those were collapses too: one went silent for 8.4 s until its end of speech, and the other stayed silent for its first 6.5 s before speaking. The other six stalled for 3.1 to 4.3 s after their first word, then went on speaking.
- **A cut draw is redrawn only if a redraw can start.** With `--silence-redraws`, a whole reply cut this way is drawn again under the usual rules: a redraw left, t + d within the budget, a prompt that can be prepared, and an engine free to take it. When any of these fails, the cut draw is the reply, `X-Finish-Reason: silence`, and the speech it would have recovered is lost.
- **Streams lose the rest of a recovering stall.** A stream is never redrawn, a single synthesis or long-form. If the stall would have recovered, the stream ends at the cut with `X-Finish-Reason: silence`, and the words after the stall are never spoken. On the draws above that is 6 of 1291 normal draws, plus the reply that was silent for its first 6.5 s. Turn the flag on for streaming clients only if ending such a stream early is worth the faster end of a collapse.
- **Why 1 s of voice, not none.** 1 of the 189 collapses never voiced a frame before the run that ended it. 13 more voiced less than 0.5 s, and 28 between 0.5 and 1 s, so a rule that waited for no voice at all would have caught 1. No normal draw took longer than 0.93 s to voice its first frame, so S starts at 1.
- **Off (0) is the default.** The guard then behaves exactly as without the flag. Text models refuse the flag.

### Holes that end a reply

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --silence-redraws 2 --silence-mid-s 3
```

A reply can fall silent mid-way for less than 10 s and then end, its last sentences never spoken: one 4-bit take spoke four sentences, went silent for 9.0 s and stopped. `--silence-mid-s S` (CUDA; 0 for off, or 3 to below 10; default 0) makes such an end a silence: when end of speech comes inside a run of S seconds or more below -40 dBFS, after 1 s or more has been voiced, the synthesis ends `silence` with its audio as it is.

- **Only the end counts.** A hole the reply recovers from is left alone, however long under 10 s: a caller's wrapper shortens every quiet run to 0.55 s, so a recovered hole is not heard, and redrawing it would only cost time. A reply the frame cap stops (`length`) is not judged, and the onset rule keeps the holes of draws voiced under 1 s.
- **Measured.** Offline replay, in the engine's frame accounting, over 2272 draws (every render on record; 225 collapses). Each candidate was transcribed with Whisper small.en: words were lost when WER was over 0.2 or the text's last 3 words were missing from the last 15 heard.
  - At S = 3 the replay ends 15 of the 2047 other draws. All 8 with an English text to judge by had lost words, from the last sentence to most of the reply. Of the other 7, Whisper heard the 4-bit take above, two corpus items cut short after a 3.2-3.5 s hole and one draw that stopped in a 9.1 s hole end the same way; two ran into the frame cap, which the engine leaves alone, and one later long-form segment was not transcribed.
  - Draws that end normally hold 1.0 to 2.65 s of silence before end of speech, so S starts at 3.
  - A rule that ended every hole of S seconds as it happened would also catch these, and end collapses about 6 s sooner, but at S = 4 it cut 24 judged draws that went on to speak in full, costing about 14 s of rendering each.
- **Redraws.** With `--silence-redraws`, such a draw is redrawn as any silence is: a redraw left, t + d within the budget, a prompt, a free engine. A redraw costs a whole draw's rendering, and only replies that lost words pay it. Without one, the reply is sent as it was, marked `X-Finish-Reason: silence`. A long-form segment is judged at its own end and redrawn in place; a stream is never redrawn and ends with the `silence` trailer.
- **Response header.** With S above 0, every whole reply carries `X-Silence-Mid-Trips`, the draws (segments included) the rule ended, and the log gives each one's hole in seconds. Streams and error replies do not carry it.
- **Off (0) is the default:** every reply is exactly as without the flag. Text models refuse the flag.

### Silence redraws

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --silence-redraws 2 --silence-redraw-plain
```

`--silence-redraws N` (0 to 8, default 0) draws a whole reply again when it trips, up to N times: when the guard ended it, or, with `--short-reply-ratio`, when it stopped far short of its text. A redraw keeps the request's voice and seeds with the server's salt plus k for the k-th redraw, so it is a different draw that is still deterministic. Salts overlap: with the same prompt, redraw k at salt s is the same draw as the first draw at salt s + k (`TENSORFOLD_SEED_SALT`), so a sweep over salts should leave N between the salts it compares.

- **Deadline.** Each redraw starts only if it should end within the budget: the lower of the request's `X-Deadline-S` (seconds the client waits, counted from when the speech route takes the request, before it reads the body) and `--silence-redraw-budget-s` (default 120). After a draw that tripped (ended `silence`, or a short `stop`, below) at t seconds from receipt and took d seconds, the next one is started only if t + d ≤ budget, since a stalled draw's duration predicts the next one's. Otherwise the last draw is sent as it is, with its own `X-Finish-Reason`. For example, a client that allows 96 s for a 509-character reply gets a redraw after a trip at 17 s (17 + 17 = 34 ≤ 96). After a 50 s stall it gets none (50 + 50 = 100 > 96), so it receives the tripped reply within its deadline instead of a timeout. A redraw that finds the engine busy with another request waits for it only until the latest start that still fits (budget − d after receipt), then sends the tripped draw. The deadline therefore binds every start; only a draw already generating can run past it. `--silence-redraw-budget-s` and `--silence-redraw-plain` need `--silence-redraws` of 1 or more.
- **Plain redraws.** With `--silence-redraw-plain`, every redraw speaks the input without its cues: each `[sighs]`, `[laughs]`, `[small laugh]`, `[pause]` and `[hesitates]` becomes a space, then white space is collapsed to single spaces and trimmed (`done. [pause] Next` is spoken `done. Next`; `word [sighs].` is spoken `word .`). An input without one of those cues is redrawn as it is. A plain redraw never speaks the cues again: if nothing is left without them, or the cue-free text cannot be prompted, there is no redraw, the tripped draw is sent, and the log says why. The cues drove the trips measured: on texts that tripped the guard, a cue-free draw finished 14 of 14, a draw with the cues and a new seed 5 of 14.
- **Plain scope.** `--silence-redraw-plain-scope controls` (default `cues`, which behaves as above) widens what a plain redraw drops to every Drama 3 delivery control: each `[…]` span (point tags such as `[pause]` and style prefixes such as `[pleased]`; the first `]` closes it, and a `[` with no `]` after it is text), each opening and closing tag of the documented scoped set `whisper`, `emphasis`, `soft`, `fast`, `slow`, `stress` and `forceful` (the text between them is kept), and a director's note head: a first line `Direction: …` together with the `Line:` that opens the second line. Each becomes a space before the same collapse, so `Direction: warm` / `Line: [pleased] It <emphasis>works</emphasis> now.` is redrawn as `It works now.` Other tags, such as `<b>` or `<french>`, and capitalised ones are kept as text: the list is deliberately narrower than a transcript scorer that drops every lowercase tag, because a tag the redraw keeps is only spoken as sent. The controls are found in the whole input, so in a long-form reply a span that a segment cut crosses is dropped from both segments, and only the input's own first line can be a note head, never a later segment's. A redrawn segment keeps its place and pauses in the reply; only its text loses its part of the controls. A client that collapses white space before sending (the assistant's wrapper does) removes the line break a note head needs, so the head is only dropped from requests that keep it.
- **Short replies.** A collapse can also end in an end of speech rather than in 10 s of silence: one 509-character reply spoke one sentence, stayed silent for 8.45 s at a `[pause]` and stopped after 9.7 s, where about 33 s were due. `--short-reply-ratio R` (0 to below 1, default 0: off) treats a whole reply that ends `stop` with less than R × chars / 15.6 seconds of audio as a trip too, where chars is the character count of the text the draw spoke (the cue-free text for a plain redraw) and 15.6 characters a second is the rate measured on earlier replies. Only texts of 200 characters or more are judged: in a shorter reply the fixed lead-in and tail make the rate noisy. The same budget and plain-redraw rules apply, the log names the trip `short`, and a long-form reply checks each segment against its own text. On 120 normal replies of 435 to 720 characters, audio ran 0.72 to 1.14 of the expected length (median 0.90). The stalled reply ran 0.30, so R = 0.5 redraws it and none of the normal ones. Texts from 200 to 435 characters were not measured.
- **Response headers.** With N above 0, every reply carries `X-Silence-Redraws`, the redraws started (0 to N), refusals included: a 400 for a malformed `X-Deadline-S` or body, a 503 for a busy engine, a 500 for a redraw that failed. The server logs each decision with t, d and the budget.
- **Streams are never redrawn.** A stream sends its first audio after one frame, and by default the guard needs 10 s of frames, so the audio is already out when it trips (`--silence-onset-s` can trip a stream sooner; see above). A stream ends with `X-Finish-Reason: silence` as before, with `X-Silence-Redraws: 0`.
- **Long-form.** A whole long-form reply redraws a segment that trips, silenced or short of its own text, in place, and later segments hear the redrawn one. N counts the redraws of the whole reply. A long-form stream is never redrawn. `--short-reply-ratio`, like the budget and plain flags, needs `--silence-redraws` of 1 or more.
- **N = 0** is the default: the server does not read `X-Deadline-S` or send `X-Silence-Redraws`, and every reply is as before.

### Register redraws

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --register-redraw-st 4 --register-redraw-high 20 --register-high-hz 210
```

A draw can open one phrase far above the rest of the reply, in a thin, high register that sounds stitched in. `--register-redraw-st R` (0 to 24, default 0: off) measures each whole reply after it is drawn and draws it once more when it trips: a phrase after the first opens R semitones or more above the median of the reply's other phrase openings, and `--register-redraw-high F` percent (above 0 to 100, default 20) or more of its voiced frames lie above `--register-high-hz H` (70 to 400, default 210). On 51 replies of a day's use, R = 4, F = 20 and H = 210 tripped 6.

- **Measure.** The audio, resampled to 16 kHz, is cut into 10 ms frames. Pitch comes from the autocorrelation of each 40 ms Hann window, 70 to 400 Hz, at half the peak's lag when that scores 0.9 of the peak or more, so a pitch is not read an octave low. A frame counts when it is voiced and within 25 dB of the speech level, the mean power of the frames within 20 dB of the 95th percentile. Rests of 0.10 s or more part phrases, a phrase under 0.5 s joins the one before, and a phrase's opening is the median pitch of its first quarter (at least 0.4 s), in semitones from the reply's median. It costs about 1.2 ms per second of audio (3.2 s for 2711 s of replies and takes). On those 77 takes, at R = 4, F = 20 and H = 210, it decided every trip as the numpy measure it was calibrated with did; that is evidence on a corpus, not proof: its arithmetic differs from numpy's in the last bits and its rounding at exact ties, so a value on a threshold may fall the other way. A draw with a NaN or infinite sample is never measured: the reply fails, as it does with the guard off.
- **One redraw, the better draw sent.** A trip draws the reply once more at the next salt, under the silence redraws' rule: it starts only if t + d fits the budget, the lower of `X-Deadline-S` and `--silence-redraw-budget-s` (default 120). The first draw is kept while the redraw is made, and the redraw is sent only if it ended neither in silence nor short and jumps less (or not at all); a tie keeps the first. A reply gets at most one register redraw, and it is not counted in `X-Silence-Redraws`.
- **With silence redraws.** A draw that tripped a silence or short redraw is redrawn as before and not measured; the draw that follows it is, so a jump after a recovered silence still gets its redraw. A register redraw that ends in silence or short is redrawn under `--silence-redraws` if one is left; the first draw is sent only if no later draw is better.
- **Response headers.** With R above 0, a whole reply carries `X-Register-Trip` (1 when a draw tripped) and `X-Register-Redraws` (0 or 1), so a trip the budget or a busy engine left without a redraw shows as `1` and `0`. The log gives each measure's largest excess after the first phrase, its high share, start, phrase count, median pitch and cost, then `register trip skipped: budget` with t, d and the budget, or the redraw and which draw was sent.
- **Not guarded.** Streams, whose audio is out before a measure could run, and long-form replies of more than one segment are sent as drawn, without the headers, as are error replies.
- **R = 0** is the default: no reply is measured, and every reply is as before.

## Long-form replies

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --default-voice NAME --long-form
```

With `--long-form`, a speech input longer than `--long-form-chars` (default 500 code points) is spoken in segments instead of in one synthesis. Each segment hears the segments before it: the earlier segments' text and generated codes are added as references after the request's references and the voice's, with the latest last. This follows the way Fish's long-form inference keeps earlier segments in context.

- **Segments.** The input is cut into the fewest segments that fit the limit, with cuts only at sentence ends (`.` `!` `?` `…` followed by white space, keeping closing quotes and brackets, and line breaks).
  - Among the cuts that give that many segments, the server picks the most balanced lengths, and prefers a cut where the text turns.
  - The text turns at a blank line, after a question, or where the next sentence opens a new move: it starts with So, Okay, OK, Now, Right, Anyway, Also, Next, Finally, But or Here's, or with a number and "thing"/"things" (One thing, Two things, 3 things).
  - A turn wins over a cut that is up to about 15% of a segment better balanced.
  - Only a sentence longer than the limit is cut inside, at `;` `:` `,`, then at spaces, then anywhere.
- **History.** `--long-form-history K` (0 to 3, default 1) sets how many earlier segments a segment hears. The oldest is dropped when the prompt would exceed the engine's prompt limit.
- **Joins.**
  - **Pause.** At each join, the silence already there is measured: the end of one segment plus the start of the next, in 10 ms frames below -42 dBFS. Only the shortfall to the boundary's target is added, never cutting existing silence. Targets:
    - `.` or `!`: 0.42 s
    - `…` or `...`: 0.55 s
    - a turn (a question, or a sentence opening a new move): 0.60 s
    - a blank line: 0.70 s
    - a cut inside a sentence (at `,` `;` `:` or a space): 0.20 s
  - **Room tone.** The added pause is copied from the earlier segment's quietest 100 ms in its last second, played back and forth. It is digital silence when that segment has no quiet tail.
  - **Level.** A segment whose speech level (the median dBFS of its frames at or above -42 dBFS) is more than 1.5 dB from the median of the earlier segments gets one constant gain. The gain brings it to 1.5 dB, is at most 3 dB either way, and never raises a peak past 0.99.
- **Seeds.** Each segment's seed comes from its own prompt rows, as a single synthesis's does.
- **Limits.**
  - More than 12 segments are refused up front.
  - A reply may use 6000 frames in all; each segment gets at most what is left.
  - A segment that is clipped (by the engine's frame limit or the budget) or ended by the silence guard ends the reply, and its audio is kept.
- **Response headers.** `X-Finish-Reason` is `stop` only when every segment stopped by itself; otherwise it is the last segment's `length` or `silence`. `X-Long-Form-Segments: N;S2,S3,…` gives the number of segments spoken and the sample offset at which each segment after the first starts, after the added pauses.
- **Streaming.** With `stream: true`, segment 1 streams as it is made. Each later segment is sent whole, with its pause, once it is complete, because its level and pause need all of it. Both headers come as trailers, declared by `Trailer: X-Finish-Reason, X-Long-Form-Segments`. They are formatted without allocating, so a stream that finishes always ends with them.

An input that fits in one segment, and every request when `--long-form` is off, is served exactly as before.

### Pauses at topic shifts

```sh
tensorfold serve DRAMA3_DIR --voices voices/ --default-voice NAME --long-form --shift-pause-s 0.85
```

A client marks a topic shift with `¶` (U+00B6), sent as ` ¶ `. A caller's wrapper writes one for each blank line it collapses. With `--shift-pause-s T` (0 to 3 seconds, default 0: off; needs `--long-form`), the reply pauses for at least T seconds at each mark, and says where those pauses are.

- **Marks.** A mark counts only after `.` `!` `?` or `…` (closing quotes and brackets may sit between), and only with words on both sides of it. Here a word means a Unicode letter or number (general category L or N, so `次へ`, `Été` and `２` count, while `！！！`, `×` and `€` do not), outside `[…]` and `<…>` controls, between the mark and the last cut and between it and the next mark. A cue may open the text after a mark (`done. ¶ [curious] Should I…`), but a mark followed by punctuation, symbols or cues alone does not count. Several marks in a row count as one. Every other mark is dropped, and the server logs how many. Every mark, with the white space after it, is removed before anything else reads the input. The tokenizer, the long-form history, a plain redraw's text and the short-reply rule's character count never see one.
- **Cues that trail a sentence.** Put the mark after the sentence's end and before a cue that trails it: `nights in a row now. ¶ [small laugh] The calendar`. The cue then opens the next segment. Written after the cue (`now. [small laugh] ¶`), the mark follows `]`, not a sentence end, so it is dropped.
- **Cuts.** Each mark that counts cuts a long-form segment there, even in an input short enough for one segment. The text between marks is split as long-form splits any input, and the segment cap still applies.
- **Pauses.** At a mark's join, the silent 10 ms frames around the join (under -42 dBFS, counted on the reply's own grid from its first sample, as a client's pacing counts them) are made to last at least T seconds. Quiet already there counts, and is never cut. The fill is added, then measured in the audio as sent. If the crossfades or the grid leave it short, more room tone goes in its middle until it is long enough. The fill is room tone: the quietest silent 100 ms (but not digital silence, under -90 dBFS) of the earlier segment's last second. Failing that it comes from all audio earlier in the reply, then from the next segment. If none of these has one, it is noise at -65 dBFS, never zeros. Its samples are kept 6 dB under the floor, so every 10 ms of it is silent. The fill fades in from the earlier segment's last samples and out into the next segment's first samples, each over 15 ms at equal power. Both segments are mirrored at their seams, so neither segment's own audio changes.
- **`X-Pause-Spans: s1-e1,s2-e2,…`** lists one span per mark the reply reached, as sample offsets into the audio exactly as sent (start inclusive, end exclusive, ascending). A span runs from the start of the earlier segment's trailing quiet to the end of the next segment's leading quiet, fill included, and always contains the silent frames measured around the join, so it holds at least T seconds of them. With `stream: true` the spans come as a trailer, declared next to `X-Long-Form-Segments`. The header is absent when there are no spans.
- **Redraws.** Redraws work per segment as usual. The spans are measured on each segment's final take, so they always describe the audio delivered. Segments under 200 characters fall outside the short-reply rule, and only the silence guard covers them.
- **Off.** Without the flag, marks are only removed: a request is served byte for byte as the same input without them.

## NVFP4 experts

A checkpoint whose routed experts are stored in NVFP4, converted from the FP8 checkpoint by `tf-drama-quant` ([docs](quantization.md#drama-3-routed-experts-in-nvfp4)), is served the same way. It is an experimental candidate that an operator selects explicitly by its directory, never a replacement for the faithful FP8 checkpoint:

```sh
tensorfold serve DRAMA3_NVFP4_DIR --voices voices/ --default-voice NAME
```

- **Format.** Each layer's routed experts are block-FP8 as published or NVFP4. An NVFP4 projection is `weight` (U8 [n, k/2], two e2m1 codes a byte, low nibble first), `weight_scale` (F8_E4M3 [n, k/16]) and `weight_scale_2` (an F32 scalar), under ModelOpt's names. The dtype of expert 0's `gate_proj.weight` decides a layer, and every other expert of the layer must match it. `config.json` is the FP8 checkpoint's, unchanged.
- **Refusals.** Startup stops on a missing or mis-shaped tensor, FP8 and NVFP4 tensors in one layer, a tensor left over (such as a `weight_scale_inv`), a scale code 0x7F, 0xFF or negative, or a global scale that is zero, negative or not finite.
- **Arithmetic.** An NVFP4 layer runs W4A16: the bf16 MoE input against e2m1 × e4m3 × global weights with fp32 sums, where the FP8 path quantizes the input to e4m3 first. SwiGLU rounds as the FP8 path's does, and each pair's output is weighted and rounded to bf16 as the FP8 path stores it, before the same sum over the top 8 and the gated shared expert. Routed experts use the decode kernel form at every row count, preserving each pair's accumulation order. Whole-model prompt chunk/reset equality is qualified separately.
- **Memory.** The routed experts of 40 layers take 16.88 GiB instead of 30.01 GiB. Loading stages one layer's stored experts (432 MiB) on the GPU at a time.
- **Startup.** The startup line includes `; routed experts NVFP4 in N of 40 layers`.
- **FP8 checkpoints** load and run as before: the same kernels, launches and scratch, and no NVFP4 kernels loaded.

## FP8 dense projections

A checkpoint may also store the backbone's large dense projections in FP8, converted from a BF16-dense one by `tf-drama-quant --dense-fp8` ([docs](quantization.md)). Like NVFP4 experts, it is an experimental candidate an operator selects by its directory:

- **Format.** Each group of each layer is BF16 as published or FP8, independently of the other groups and of the routed experts' format: `X.weight` (F8_E4M3 [n, k], never the codes 0x7F or 0xFF) beside `X.weight_scale_inv` (F32 [n, k/64], one positive finite scale a row and 64 inputs; positive f32 subnormals are admitted), under the BF16 tensor's name. The groups are `linear_attn.in_proj_qkv` with `in_proj_z`, `linear_attn.out_proj`, `self_attn.q_proj` with `k_proj` and `v_proj`, `self_attn.o_proj`, `mlp.shared_expert.gate_proj` with `up_proj`, and `mlp.shared_expert.down_proj`; the dtype of a group's first tensor decides. `in_proj_b`/`in_proj_a`, `conv1d`, `A_log`, `dt_bias`, the norms, the router `mlp.gate`, `shared_expert_gate`, the embeddings and `lm_head` stay BF16. `config.json` is unchanged.
- **Refusals.** Startup stops on a group whose tensors mix BF16 and FP8, a missing, mis-typed or mis-shaped `weight_scale_inv`, one left over beside a BF16 weight, a code 0x7F or 0xFF, or a scale that is zero, negative or not finite.
- **Arithmetic.** An FP8 group runs W8A16 on the lane matmul's fp8g form: the bf16 rows against the codes' exact bf16 values, fp32 sums a 64-input group, each group's sum times its scale added in fp32, one bf16 rounding. Activations are never quantized. It uses the same kernel form at every row count, with K slices fixed by the shape, so a row's bits do not depend on how many rows a pass holds; whole-model prompt chunk/reset equality is qualified separately. The lane writes packed rows, so beside an FP8 `in_proj_qkv`/`in_proj_z` the `in_proj_b`/`in_proj_a` rows, and beside an FP8 shared gate/up the router, run as their own BF16 GEMM, whose outputs are bit-equal to the fused GEMM's columns.
- **Memory.** The six groups of all 40 layers take 1.39 GiB in FP8 instead of 2.62 GiB in BF16.
- **Startup.** The startup line ends with `; dense FP8 in N of 40 backbone layers (gdn_in a, ...), M of 13 tower layers (tower_qkv b, ...)`: the layers storing any group FP8, then per class the layers storing it so, for the backbone and the [tower](#fp8-tower-projections) together.
- **BF16 dense checkpoints** load and run as before: the same kernels, launches and scratch, and no lane module loaded.

## Measuring the model

`tf-drama-probe` (a development tool, `zig build drama-probe`, in `zig-out/oracle/bin`) runs requests through the production prompt pass and frame loop and keeps every step's semantic logits, to compare checkpoints, such as a quantized one against the FP8 original.

```sh
tf-drama-probe DRAMA3_DIR voices/ requests.jsonl runs/fp8 --stats --rows runs/fp8-rows
tf-drama-probe QUANTIZED_DIR voices/ requests.jsonl runs/q --force runs/fp8/codes
python3 tools/zig/drama_kl.py runs/fp8 runs/q --out runs/q-kl.json
```

- **Requests.** One JSON object a line: `id`, `voice` (a file in the voices directory; `null` or `default` for the built-in voice), `text` and `max_frames` (1 to 8191), with `salt` (default 0), `temperature`, `top_p` and `top_k` (default the model's 1.0, 0.9 and 30). The prompt and seed are the server's for that input, voice and seed salt. A request is one synthesis: long-form splitting does not apply.
- **What runs.** The backbone and tower as the server loads them, context 8192, prompts in passes of 128 rows (`--prompt-chunk N`, 1 to 128, sets another pass size for chunk-equivalence checks). No codec, silence guard or redraw: the probe measures the model, the server what it delivers.
- **Output.** `OUT_DIR` must not exist. `codes/ID.u32` holds the frames' codes, `u32` `[frames][10]`, codebook 0 first. `logits/ID.f32` holds `f32` `[steps][4097]`: EOS, then the 4096 semantic ids, one row a step, as the server steps: `frames + 1` steps when the last chose EOS, `frames` when `max_frames` cut the reply. `manifest.json` names the tool's commit, the checkpoint (its `config.json` digest and each layer's expert format), the requests' digest, and per request its prompt rows and digest, frames, steps, seed, finish (`eos` or `length`) and the file digests.
- **Teacher forcing.** `--force CODES_DIR`, a probe output's `codes`, feeds each request's frames whole instead of sampling, for as many steps as that run took (its `manifest.json` beside `CODES_DIR` gives the finish). A request whose prompt, prompt length or codes differ from what that manifest records, or a manifest whose ids repeat or whose steps disagree with its frames and finish, is refused before the model loads. A run forced with its own codes reproduces its logits bit for bit. It also writes `tower/ID.f32`, `f32` `[frames][9]`: each tower head's negative log-likelihood of the forced codebook 1 to 9 code at temperature 1, from the head the sampler reads, given the frame's earlier forced codes. Non-finite semantic or head logits stop the run. The tower teacher-forces a prompt chunk with one cuBLASLt plan per row count, so its NLL depends on the prompt chunk size: compare tower NLL only between runs with the same `--prompt-chunk` (default 128, the server's). The semantic logits of a forced run do not depend on it.
- **Statistics.** `--stats` also writes `stats.safetensors`: per layer, expert and row class (prompt text, prompt audio, generated), F64 routed-pair counts, routing mass, the summed norm of the stored routing-weighted down rows, and per input column the sum of squares of the MoE input and of the SwiGLU activation. In an NVFP4 layer the activation is the gate-up kernel's own bf16 SwiGLU, and the down rows are the bf16(w × down) pair rows its combine sums. `mass.json` is each layer's mass, every class summed. The sums are added on the GPU in a fixed order: the same requests give the same file.
- **Rows.** `--rows DIR` (must not exist) keeps every backbone row's MoE input, picks and routing weights: `class.u8` `[R]`, and per layer `layer-LL.x.bf16` `[R][2048]`, `layer-LL.picks.i32` and `layer-LL.w.f32` `[R][8]`, in run order; `rows.json` gives each chunk's request and positions.
- **Comparing.** `drama_kl.py A B` gives KL(A‖B) over the 4097 columns at temperature 1, per request and pooled (mean, median, 99th percentile, maximum), the first step's KL apart, top-1 agreement, the worst rows and, when both runs hold them, the tower NLL differences. It refuses runs whose prompts, codes or steps differ.

## FP8 tower projections

An experimental checkpoint may store the tower's four large projections per layer, `attention.wqkv`, `attention.wo`, `feed_forward.w1` with `w3`, and `feed_forward.w2`, in FP8 instead of BF16. Like NVFP4 experts, an operator selects it explicitly by its directory; it never replaces the faithful checkpoint.

- **Format.** A converted projection is `weight` (F8_E4M3 [n, k], torch Linear order) and `weight_scale_inv` (F32 [n, k/64]): weight = e4m3(code) × the scale of its row and 64 inputs. Each layer's projection, `w1` and `w3` together, is BF16 or FP8 by its stored dtype, whatever the other layers or the routed experts hold. `eh_proj`, `enorm`, `hnorm`, the embeddings and the nine heads stay BF16.
- **Refusals.** Startup stops on a missing or mis-shaped tensor or companion, a companion of another dtype, `w1` and `w3` in different formats, a companion left over beside a BF16 weight, a code 0x7F or 0xFF (the GPU reads them as ±480, not NaN), or a scale that is zero, negative or not finite. Positive subnormal scales are accepted.
- **Arithmetic.** An FP8 projection runs W8A16 on the lane matmul: bf16 rows times the exact e4m3 values in bf16 MMAs, an fp32 sum for every 64 inputs scaled by its f32 scale, input slices joined in a fixed order, bf16 output. It uses one kernel form at every row count, so its rows do not depend on how many rows a call holds, and it never takes the one-row `drama_tower_gemv` or a cuBLASLt plan. BF16 projections run exactly as before. The rest of the tower still selects by row count (attention, `eh_proj`, heads), so whole-tower chunk equality is measured, not implied.
- **Memory.** The four projections of 13 layers take 897,974,272 bytes (codes and scales) instead of 1,690,304,512 in BF16. Loading stages one layer's stored codes (62 MiB) on the GPU at a time. The lane kernels and, on GPUs without thread-block clusters, up to 9 MiB of slice scratch are loaded only when a layer holds FP8.
- **Startup.** The startup line's [dense FP8 count](#fp8-dense-projections) gives `M of 13 tower layers (tower_qkv a, tower_o b, tower_gate_up c, tower_down d)`. A checkpoint without FP8 tower projections loads and runs as before: the same allocations, plans and launches, and no lane kernels.
- **Checks.** `tf-cuda-test drama-fp8-tower CHECKPOINT [LAYERS]` prints its fixed criteria, then holds the lane to float64 and to the BF16 path on the dequantized weights at 1, 7 and 128 rows, checks its rows bit-equal across those row counts, and replays every projection of prefills that mix FP8 and BF16 within and across layers; `LAYERS` (such as `0,4,12`) adds the checkpoint's stored FP8 projections. `tools/zig/drama_trace_same.py A B` compares two `tf-drama-tower-trace` outputs file by file, for a BF16 checkpoint traced by two builds.
