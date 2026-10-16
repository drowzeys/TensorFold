"""MiniMax Music 3: how much of the lyrics a song sings, and when (Whisper large-v3-turbo with word timestamps).

    python tools/music3_listen.py song.wav --lyrics song.lyrics [--words-dir DIR] [--report all.jsonl]
    python tools/music3_listen.py a.wav=a.lyrics b.wav=b.lyrics --backend cpu

Each song is transcribed with word timestamps (mlx-whisper on the GPU, or faster-whisper on the CPU with --backend
cpu) and aligned to its lyrics the way Drama 3's end-of-text check aligns speech to text (section tags such as
[Verse] are not words, numbers are spelled out, case and punctuation are ignored, near spellings count). Printed per
song, one JSON line:

  word_accuracy   lyric words heard, in order, over all lyric words
  sung_accuracy   the same over the lyrics up to the last lyric word heard (a song capped short of its lyrics is
                  not penalised for the verses it never reached)
  precision       heard words that are lyric words (the rest: ad-libs, crowd hooks, mishearings)

and ``<wav stem>.words.json`` (next to the wav, or in --words-dir) with every heard word's start and end, every lyric
word's matched time, and each lyric line's span: what lyric-timed captions need. No source separation is applied
(none is installed on the Mac this was written on); Whisper hears the full mix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100


def lyric_lines(lyrics: str) -> list[tuple[str, list[str]]]:
    """(line text, its words) for each line of the lyrics that has words (section tags are not words)."""

    from tensorfold.families.drama3.verify import words_of

    return [(line.strip(), words_of(line)) for line in lyrics.splitlines() if words_of(line)]


def score(lyrics: str, heard: list, audio_s: float | None = None) -> dict:
    """Align heard words (objects with .text .start .end) to the lyrics. Returns scores and per-word / per-line
    timings."""

    from tensorfold.families.drama3.verify import MATCH_SIMILARITY, align, similarity, words_of

    lines = lyric_lines(lyrics)
    ref, line_of = [], []
    for n, (_, words) in enumerate(lines):
        ref += words
        line_of += [n] * len(words)
    hyp, hyp_word = [], []
    for k, w in enumerate(heard):
        for piece in words_of(w.text):
            hyp.append(piece)
            hyp_word.append(k)
    times: list = [None] * len(ref)
    used = set()
    for refs, hyps in (align(ref, hyp) if ref and hyp else []):
        if not refs or not hyps:
            continue
        if len(refs) == 1 and len(hyps) == 1 and similarity(ref[refs[0]], hyp[hyps[0]]) < MATCH_SIMILARITY:
            continue
        words = [heard[hyp_word[j]] for j in hyps]
        for i in refs:
            times[i] = (min(w.start for w in words), max(w.end for w in words))
        used.update(hyps)
    matched = sum(t is not None for t in times)
    reached = max((i + 1 for i, t in enumerate(times) if t is not None), default=0)
    longest, run = 0, 0
    for t in times[:reached]:
        run = run + 1 if t is None else 0
        longest = max(longest, run)
    out_lines = []
    for n, (text, words) in enumerate(lines):
        spans = [times[i] for i in range(len(ref)) if line_of[i] == n and times[i] is not None]
        out_lines.append({"text": text, "words": len(words), "matched": len(spans),
                          "start": round(min(s for s, _ in spans), 2) if spans else None,
                          "end": round(max(e for _, e in spans), 2) if spans else None})
    return {
        "lyric_words": len(ref), "heard_words": len(hyp), "matched": matched, "reached_words": reached,
        "word_accuracy": round(matched / max(len(ref), 1), 3),
        "sung_accuracy": round(matched / max(reached, 1), 3) if reached else 0.0,
        "precision": round(len(used) / max(len(hyp), 1), 3),
        "longest_missing_run": longest,
        "lyrics": [{"word": ref[i], "line": line_of[i], "start": None if t is None else round(t[0], 2),
                    "end": None if t is None else round(t[1], 2)} for i, t in enumerate(times)],
        "lines": out_lines,
    }


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.T, rate                               # [C, n]


def spectral(mono: np.ndarray, rate: int) -> dict:
    head = mono[: rate * 30]
    spec = np.abs(np.fft.rfft(head))
    freqs = np.fft.rfftfreq(len(head), 1 / rate)
    return {"rms_db": round(float(20 * np.log10(np.sqrt((mono ** 2).mean()) + 1e-9)), 1),
            "centroid_hz": int(round(float((spec * freqs).sum() / max(spec.sum(), 1e-9))))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("songs", nargs="+", help="song.wav, or song.wav=song.lyrics")
    p.add_argument("--lyrics", type=Path, help="lyrics for every song given without its own")
    p.add_argument("--backend", choices=("auto", "mlx", "cpu"), default="auto")
    p.add_argument("--language", default="en")
    p.add_argument("--words-dir", type=Path, help="where the .words.json files go (default: next to each wav)")
    p.add_argument("--no-words", action="store_true", help="do not write .words.json files")
    p.add_argument("--report", type=Path, help="also append each song's JSON line here")
    p.add_argument("--transcript", action="store_true", help="include the full transcript in the printed line")
    a = p.parse_args(argv)

    from tensorfold.families.drama3.verify import Transcriber

    asr = Transcriber(backend=a.backend)
    for item in a.songs:
        wav, _, lyr = item.partition("=")
        wav = Path(wav).expanduser()
        lyrics_path = Path(lyr).expanduser() if lyr else a.lyrics
        lyrics = lyrics_path.read_text() if lyrics_path else ""
        audio, rate = read_audio(wav)
        mono = audio.mean(axis=0)
        started = time.perf_counter()
        heard = asr.words(mono, rate, a.language)
        asr_s = time.perf_counter() - started
        result = score(lyrics, heard) if lyrics else {}
        transcript = "".join(w.text for w in heard).strip()
        line = {"wav": str(wav), "lyrics": str(lyrics_path) if lyrics_path else None,
                "audio_s": round(len(mono) / rate, 2), **spectral(mono, rate), "backend": asr.backend,
                "asr_s": round(asr_s, 2), "separated": False,
                **{k: v for k, v in result.items() if k not in ("lyrics", "lines")}}
        if a.transcript:
            line["transcript"] = transcript
        if not a.no_words:
            folder = a.words_dir or wav.parent
            folder.mkdir(parents=True, exist_ok=True)
            words_path = folder / (wav.stem + ".words.json")
            words_path.write_text(json.dumps({
                **line, "transcript": transcript,
                "words": [{"text": w.text.strip(), "start": round(w.start, 2), "end": round(w.end, 2)} for w in heard],
                "lyric_words_timed": result.get("lyrics", []), "lines": result.get("lines", [])}, indent=1) + "\n")
            line["words_json"] = str(words_path)
        text = json.dumps(line)
        print(text, flush=True)
        if a.report:
            with open(a.report, "a") as f:
                f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
