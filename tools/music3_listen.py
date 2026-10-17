"""MiniMax Music 3: how much of the lyrics a song sings, and when (Whisper large-v3-turbo with word timestamps).

    python tools/music3_listen.py song.wav --lyrics song.lyrics [--words-dir DIR] [--report all.jsonl]
    python tools/music3_listen.py a.wav=a.lyrics b.wav=b.lyrics --backend cpu

Each song is transcribed with word timestamps (mlx-whisper on the GPU, or faster-whisper on the CPU with --backend
cpu) once over the whole song and once in independent 20 s chunks that overlap by 5 s (one pass loses whole 30 s windows
when Whisper hallucinates on an instrumental stretch and skips ahead; chunks lose words at other places), and aligned to its lyrics the way Drama 3's end-of-text check
aligns speech to text (section tags such as [Verse] are not words, numbers are spelled out, case and punctuation are
ignored, near spellings count; a repeated chorus is matched where it first comes). Printed per song, one JSON line:

  word_accuracy   lyric words heard, in order, over all lyric words
  sung_accuracy   the same over the lyrics up to the end of the last run of 4 lyric words heard in a row (a song
                  capped short of its lyrics is not penalised for the verses it never reached)

A lyric word counts as heard when the one-pass or the chunked transcription heard it (``passes`` gives each alone).
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


def align_earliest(ref: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """(lyric index, heard index) matches of a minimum-cost alignment, as Drama 3's (a near word costs up to 0.4, a
    different one 1.5, a missing or extra word 1), with one change for songs: a missing lyric word costs a little
    more the earlier it is, so a repeated chorus is matched where it first comes, not to a later repeat with the
    verses between it counted as missed."""

    from tensorfold.families.drama3.verify import MATCH_SIMILARITY, similarity

    n, m = len(ref), len(hyp)
    eps = 1e-3 / max(n, 1)
    cost = np.full((n + 1, m + 1), np.inf)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)                # 1 diagonal, 2 missing lyric word, 3 extra word
    cost[0, :] = np.arange(m + 1)
    move[0, 1:] = 3
    sims: dict = {}
    for i in range(1, n + 1):
        cost[i, 0] = cost[i - 1, 0] + 1 + eps * (n - i)
        move[i, 0] = 2
        a = ref[i - 1]
        for j in range(1, m + 1):
            b = hyp[j - 1]
            if (a, b) not in sims:
                sim = similarity(a, b)
                sims[(a, b)] = 1.0 - sim if sim >= MATCH_SIMILARITY else 1.5
            best, how = cost[i - 1, j - 1] + sims[(a, b)], 1
            c = cost[i - 1, j] + 1 + eps * (n - i)
            if c < best:
                best, how = c, 2
            c = cost[i, j - 1] + 1
            if c < best:
                best, how = c, 3
            cost[i, j], move[i, j] = best, how
    pairs, i, j = [], n, m
    while i or j:
        how = move[i, j]
        if how == 1:
            if sims[(ref[i - 1], hyp[j - 1])] < 1.0:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif how == 2:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def heard_words(asr, mono: np.ndarray, rate: int, language: str, chunk_s: float, overlap_s: float) -> list:
    """Whisper word timestamps over independent overlapping chunks (``chunk_s`` 0: one pass over the whole song).

    One pass loses a whole 30 s window when Whisper hallucinates on an instrumental stretch and skips ahead; chunks
    transcribed on their own cannot carry that over. Each chunk keeps the words whose middle falls in its own share
    of the overlaps."""

    if not chunk_s or len(mono) <= chunk_s * rate:
        return asr.words(mono, rate, language)
    from types import SimpleNamespace

    hop = chunk_s - overlap_s
    out, start, total = [], 0.0, len(mono) / rate
    while start < total:
        end = min(start + chunk_s, total)
        lo = 0.0 if start == 0 else start + overlap_s / 2
        hi = total if end >= total else end - overlap_s / 2
        for w in asr.words(mono[int(start * rate):int(end * rate)], rate, language):
            mid = start + (w.start + w.end) / 2
            if lo <= mid < hi:
                out.append(SimpleNamespace(text=w.text, start=start + w.start, end=start + w.end))
        if end >= total:
            break
        start += hop
    return out


def score(lyrics: str, heard: list, audio_s: float | None = None) -> dict:
    """Align heard words (objects with .text .start .end) to the lyrics. Returns scores and per-word / per-line
    timings."""

    from tensorfold.families.drama3.verify import words_of

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
    for i, j in (align_earliest(ref, hyp) if ref and hyp else []):
        w = heard[hyp_word[j]]
        times[i] = (w.start, w.end)
        used.add(j)
    return finish_score(ref, line_of, lines, times, len(hyp), len(used))


RUN = 4                              # a stretch of this many lyric words heard in a row marks how far a song got


def finish_score(ref, line_of, lines, times, heard_words: int, used: int) -> dict:
    matched = sum(t is not None for t in times)
    reached, run = 0, 0
    for i, t in enumerate(times):
        run = run + 1 if t is not None else 0
        if run >= RUN or (t is not None and i < RUN and run == i + 1):
            reached = i + 1
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
        "lyric_words": len(ref), "heard_words": heard_words, "matched": matched, "reached_words": reached,
        "word_accuracy": round(matched / max(len(ref), 1), 3),
        "sung_accuracy": round(sum(t is not None for t in times[:reached]) / reached, 3) if reached else 0.0,
        "precision": round(used / max(heard_words, 1), 3),
        "longest_missing_run": longest,
        "lyrics": [{"word": ref[i], "line": line_of[i], "start": None if t is None else round(t[0], 2),
                    "end": None if t is None else round(t[1], 2)} for i, t in enumerate(times)],
        "lines": out_lines,
    }


def union(lyrics: str, passes: list[dict]) -> dict:
    """Several transcriptions' scores merged: a lyric word counts as heard when any pass heard it (its time from the
    first pass that did). Whisper drops different stretches of a song in different passes; inventing the very lyric
    word, in order, is far rarer."""

    lines = lyric_lines(lyrics)
    ref, line_of = [], []
    for n, (_, words) in enumerate(lines):
        ref += words
        line_of += [n] * len(words)
    times = [None] * len(ref)
    for p in passes:
        for i, w in enumerate(p["lyrics"]):
            if times[i] is None and w["start"] is not None:
                times[i] = (w["start"], w["end"])
    heard = max(p["heard_words"] for p in passes)
    used = max(round(p["precision"] * p["heard_words"]) for p in passes)
    return finish_score(ref, line_of, lines, times, heard, used)


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
    p.add_argument("--chunk", type=float, default=20.0,
                   help="also transcribe in independent chunks of this many seconds and count a lyric word heard "
                        "in either pass (0: one pass only)")
    p.add_argument("--chunk-overlap", type=float, default=5.0)
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
        chunked = heard_words(asr, mono, rate, a.language, a.chunk, a.chunk_overlap) if a.chunk else None
        asr_s = time.perf_counter() - started
        result = score(lyrics, heard) if lyrics else {}
        passes = {}
        if lyrics and chunked is not None:
            second = score(lyrics, chunked)
            passes = {"one_pass": {k: result[k] for k in ("word_accuracy", "sung_accuracy", "matched")},
                      "chunked": {k: second[k] for k in ("word_accuracy", "sung_accuracy", "matched")}}
            result = union(lyrics, [result, second])
        transcript = "".join(w.text for w in heard).strip()
        line = {"wav": str(wav), "lyrics": str(lyrics_path) if lyrics_path else None,
                "audio_s": round(len(mono) / rate, 2), **spectral(mono, rate), "backend": asr.backend,
                "asr_s": round(asr_s, 2), "separated": False,
                **{k: v for k, v in result.items() if k not in ("lyrics", "lines")}, **({"passes": passes} if passes else {})}
        if a.transcript:
            line["transcript"] = transcript
        if not a.no_words:
            folder = a.words_dir or wav.parent
            folder.mkdir(parents=True, exist_ok=True)
            words_path = folder / (wav.stem + ".words.json")
            words_path.write_text(json.dumps({
                **line, "transcript": transcript,
                "words": [{"text": w.text.strip(), "start": round(w.start, 2), "end": round(w.end, 2)} for w in heard],
                **({"words_chunked": [{"text": w.text.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
                                      for w in chunked]} if chunked is not None else {}),
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
