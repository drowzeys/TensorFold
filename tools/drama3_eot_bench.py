"""Runaway / missing-word rates of Drama 3 over texts and seeds, with and without the end-of-text check.

    python tools/drama3_eot_bench.py --seeds 6 --out runs.jsonl [--verify] [--ref-audio a.wav --ref-text "..."]

Each run synthesizes a text with one seed (Fish's sampling defaults); every chunk's audio is then judged by an
independent recogniser (faster-whisper on the CPU, beam 5; the check itself uses mlx-whisper): a chunk is a
*runaway* when words follow its text or more than 2 s of sound follows its last word (5 s when the text ends with a
[direction]); it *misses words* when under 90% of its words are heard, three or more in a row are missing, or its
ending is. Per-step p(EOS) traces go to ``<out>.npz`` for the model-side stop study.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TEXTS = {
    "short": "Hello! This is the Wasserman Slate AI Studio, powered by TensorFold, built with integrated Fish Audio "
             "Drama 3.",
    "long": None,      # LONG below
    "two": "<|speaker:0|> Did you hear that noise downstairs? <|speaker:1|> It's probably just the cat. Go back to "
           "sleep. <|speaker:0|> Cats don't knock on doors.",
    "clone": "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.",
    "tags": "[whispering] Keep your voice down. They can hear us. [laughs] Just kidding, nobody is here. [sighs] "
            "Let's go home, it's been a long night.",
}
LONG = ("The lighthouse keeper had not spoken to another person in forty-one days. Each morning he climbed the "
        "spiral stairs, wiped the salt from the great lens, and wrote a single line in the logbook: weather, ships, "
        "nothing else. He liked it that way, or he told himself he did.\n\nOn the forty-second day a small boat "
        "appeared on the horizon, rowing hard against the swell. He watched it through the brass telescope for nearly "
        "an hour before he understood that the figure at the oars was waving at him. [whispering] Not now, he "
        "thought. Not after all this time.\n\n<|speaker:1|> Hello up there! Is anyone home? <|speaker:0|> [clears "
        "throat] That depends on who is asking. <|speaker:1|> A very tired woman with a broken compass and a basket "
        "of oranges. <|speaker:0|> [laughs] Then you had better come inside before the tide turns.")
TEXTS["long"] = LONG
# a harder set: very short texts, a closing direction, digits, a question (runaways are rarer on plain prose)
TEXTS.update({
    "tiny": "Yes.",
    "okay": "Okay. Fine. Let's go.",
    "endtag": "I can't believe you actually did that. [laughs]",
    "numbers": "Call me at 555-0142 before 9:30 on March 3rd, 2027.",
    "question": "Wait, what was that?",
    "clone_short": "Thanks, see you tomorrow.",
    # dialogue, each written on one line (as typed into a text box)
    "dlg_vo": "<|speaker:0|> [whispering] Did you hear that noise downstairs? <|speaker:1|> [sighs] It's probably just "
              "the cat. Go back to sleep. <|speaker:0|> Cats don't knock on doors.",
    "dlg_four": "<|speaker:0|> [laughs] You really thought that would work? <|speaker:1|> [annoyed] It worked last "
                "time. <|speaker:0|> Last time was different. We had a map. <|speaker:1|> [sighs] Fine. Next time "
                "you lead.",
})
CLONE_REF_TEXT = "We asked over twenty different people, and they all said it was his."


def judge(check, text: str) -> dict:
    from tensorfold.families.drama3.verify import ends_with_direction

    allowed = 5.0 if ends_with_direction(text) else 2.0
    runaway = bool(check.tail_words) or check.tail_s > allowed
    missing = bool(check.coverage < 0.9 or check.longest_missing_run >= 3 or check.ending_missing)
    return {"runaway": runaway, "missing": missing, "coverage": round(check.coverage, 3),
            "tail_words": check.tail_words, "tail_s": round(check.tail_s, 2), "missing_words": check.missing,
            "heard": check.heard}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="~/models/drama-3-mlx")
    p.add_argument("--texts", default="short,long,two,clone,tags")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--verify", action="store_true", help="synthesize with the end-of-text check")
    p.add_argument("--stop-rule", action="store_true", help="synthesize with the model-side StopRule")
    p.add_argument("--stream", action="store_true", help="stream (Drama3Stream) instead of synthesize")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--eos-bias", type=float, default=0.0, help="< 0 lowers the EOS logit when drawing (provokes "
                   "runaways: the model's p(EOS) still rises at the end, the draw misses it)")
    p.add_argument("--ref-audio", type=Path, default=Path("~/drama3-work/common_voice_en_10119832.wav"))
    p.add_argument("--ref-text", default=CLONE_REF_TEXT)
    p.add_argument("--wav-dir", type=Path, help="also save each run's audio here")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    import mlx.core as mx

    from tensorfold.families.drama3.codec import load_audio, resample, save_wav
    from tensorfold.families.drama3.generate import Drama3Generator, spoken_chars, split_text
    from tensorfold.families.drama3.prompt import Reference
    from tensorfold.families.drama3.sampling import Sampling
    from tensorfold.families.drama3.verify import Policy, Transcriber, Word, check_words

    gen = Drama3Generator(args.model)
    guard = Transcriber("mlx")
    judge_asr = Transcriber("cpu")
    clone_ref = Reference(transcript=args.ref_text,
                          codes=np.array(gen.codec.encode(load_audio(args.ref_audio.expanduser()))))
    traces: dict[str, np.ndarray] = {}
    rows = []
    if args.wav_dir:
        args.wav_dir.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for name in args.texts.split(","):
            text = TEXTS[name]
            refs = [clone_ref] if name.startswith("clone") else []
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                started = time.perf_counter()
                sampling = Sampling(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                    eos_bias=args.eos_bias)
                if args.stream:
                    from tensorfold.families.drama3.stream import Drama3Stream

                    st = Drama3Stream(gen, text, references=refs, sampling=sampling, seed=seed,
                                      verify=guard if args.verify else False, stop=args.stop_rule, log=lambda s: None)
                    pieces = list(st)
                    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
                    results = st.results
                else:
                    audio, results = gen.synthesize(text, references=refs, sampling=sampling, seed=seed,
                                                    verify=guard if args.verify else False, stop=args.stop_rule,
                                                    log=lambda s: None)
                total_s = time.perf_counter() - started
                if args.wav_dir:
                    save_wav(args.wav_dir / f"{name}_s{seed}.wav", audio)
                chunks = split_text(text, 300)
                chunk_rows = []
                for k, (chunk, r) in enumerate(zip(chunks, results)):
                    wav = np.array(gen.codec.decode(r.codes)) if len(r.frames) else np.zeros(0, np.float32)
                    mx.clear_cache()
                    judge_asr._load()
                    x = resample(wav, 44100, 16000) if len(wav) else wav
                    segs, _ = judge_asr._impl.transcribe(x, language="en", beam_size=5, word_timestamps=True,
                                                         condition_on_previous_text=False)
                    words = [Word(w.word, float(w.start), float(w.end)) for s in segs for w in (s.words or [])]
                    verdict = judge(check_words(chunk, words, len(wav) / 44100, Policy()), chunk)
                    key = f"{name}_s{seed}_c{k}"
                    traces[key] = r.eos_probs
                    chunk_rows.append({"chunk": k, "frames": len(r.frames), "finish": r.finish,
                                       "frames_per_char": round(len(r.frames) / max(1, spoken_chars(chunk)), 2),
                                       "eos_max": round(float(r.eos_probs.max()), 4) if len(r.eos_probs) else None,
                                       "fps": round(len(r.frames) / max(r.decode_s, 1e-9), 1),
                                       "guard": r.stats.get("verify"),
                                       "stop_rule_cut_frames": r.stats.get("stop_rule_cut_frames"), **verdict})
                row = {"text": name, "seed": seed, "verify": args.verify, "stop_rule": args.stop_rule,
                       "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                       "eos_bias": args.eos_bias, "audio_s": round(len(audio) / 44100, 2),
                       "total_s": round(total_s, 2), "frames": sum(len(r.frames) for r in results),
                       "runaway": any(c["runaway"] for c in chunk_rows),
                       "missing": any(c["missing"] for c in chunk_rows),
                       "chunks": chunk_rows}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{name:6s} seed {seed}: {row['audio_s']:5.1f}s audio in {row['total_s']:5.1f}s "
                      f"runaway={row['runaway']} missing={row['missing']} "
                      + " | ".join(f"c{c['chunk']} {c['frames']}f {c['finish']} cov {c['coverage']} "
                                   f"tail {len(c['tail_words'])}w/{c['tail_s']}s"
                                   + (f" guard {c['guard']['action']}x{c['guard'].get('attempts', 1)}" if c["guard"] else "")
                                   for c in chunk_rows), flush=True)
    np.savez(args.out.with_suffix(".npz"), **traces)
    n = len(rows)
    summary = {"runs": n, "runaway": sum(r["runaway"] for r in rows), "missing": sum(r["missing"] for r in rows),
               "chunks": sum(len(r["chunks"]) for r in rows),
               "chunk_runaway": sum(c["runaway"] for r in rows for c in r["chunks"]),
               "chunk_missing": sum(c["missing"] for r in rows for c in r["chunks"]),
               "audio_s": round(sum(r["audio_s"] for r in rows), 1),
               "total_s": round(sum(r["total_s"] for r in rows), 1)}
    print(json.dumps(summary), flush=True)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
