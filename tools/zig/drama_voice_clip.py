"""Prepare a reference clip for drama_voice.py: mono 44.1 kHz, set to one speech level, cut at a pause to 10-30 s."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 44100
HOP = RATE // 100
MIN_SECONDS, MAX_SECONDS = 10.0, 30.0
LEVEL_DBFS = -20.0
PEAK_DBFS = -1.0
# A frame this far under the speech level is quiet; a quiet run this long is a pause, not a gap inside a word.
QUIET_DB = 25.0
PAUSE_FRAMES = 15
SENTENCE_FRAMES = 40
EDGE_FRAMES = 25
# A file that opens or closes on fewer quiet frames was cut inside a word.
BOUNDARY_FRAMES = 2
CLIP_RUN, CLIP_RUNS = 3, 3
MIN_SNR_DB = 30.0
# Band flatness under this in the pauses is a tone, music or hum, not room noise; under AUDIBLE_DB nobody hears it.
MIN_FLATNESS = 0.1
AUDIBLE_DB = 60.0
MAX_SPREAD_ST = 6.0
WINDOW = 4096
BANDS = (100, 200, 400, 800, 1600, 3200)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """The [start, end) runs where `mask` holds."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def clipped(x: np.ndarray) -> int:
    """Runs of CLIP_RUN or more samples held at the clip's top or bottom value: the flat tops clipping leaves."""
    return sum(end - start >= CLIP_RUN for edge in (x.max(), x.min()) for start, end in runs(x == edge))


def resample(x: np.ndarray, rate: int, to: int) -> np.ndarray:
    """Band-limited resampling by the FFT; a clip starts and ends quiet, so the wrap-around is harmless."""
    if rate == to:
        return x
    n = round(len(x) * to / rate)
    spectrum = np.zeros(n // 2 + 1, complex)
    kept = min(len(spectrum), len(x) // 2 + 1)
    spectrum[:kept] = np.fft.rfft(x)[:kept]
    return np.fft.irfft(spectrum, n) * (n / len(x))


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    return 10 * np.log10(np.mean(x[: n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-10)


def speech_level(db: np.ndarray) -> float:
    """The mean power of the frames within 20 dB of the loud end, so pauses and soft consonants do not lower it."""
    loud = db[db >= np.percentile(db, 95) - 20]
    return float(10 * np.log10(np.mean(10 ** (loud / 10))))


def pauses(quiet: np.ndarray) -> list[tuple[int, int]]:
    return [(start, end) for start, end in runs(quiet) if end - start >= PAUSE_FRAMES]


def cut(db: np.ndarray, level: float) -> tuple[int, int, int]:
    """Frames [start, end) from just before the first sound to a pause 10-30 s on, and the loud frames left out."""
    quiet = db < level - QUIET_DB
    n = len(db)
    if not quiet[:BOUNDARY_FRAMES].all():
        raise SystemExit("the clip starts inside speech: record a moment of silence before the first word")
    first = int(np.argmin(quiet))
    start, whole = max(0, first - EDGE_FRAMES), min(n, n - int(np.argmin(quiet[::-1])) + EDGE_FRAMES)
    if (whole - start) / 100 < MIN_SECONDS:
        raise SystemExit(f"the clip is {(whole - start) / 100:.1f} s from its first sound to its last; a voice needs {MIN_SECONDS:.0f} s")
    if quiet[-BOUNDARY_FRAMES:].all() and (whole - start) / 100 <= MAX_SECONDS:
        return start, whole, 0
    ends = [(e - s >= SENTENCE_FRAMES, s + min(EDGE_FRAMES, (e - s) // 2)) for s, e in pauses(quiet) if first < s and e < n]
    ends = [end for end in ends if MIN_SECONDS <= (end[1] - start) / 100 <= MAX_SECONDS]
    if not ends:
        raise SystemExit(f"no pause between {MIN_SECONDS:.0f} and {MAX_SECONDS:.0f} s after the first sound to cut at")
    end = max(ends)[1]
    return start, end, int(np.count_nonzero(~quiet[end:]))


def flatness(x: np.ndarray, db: np.ndarray, level: float) -> float | None:
    """The median, over windows wholly inside pauses, of the least flat octave band; None when no pause is that long."""
    half = WINDOW // 2 // HOP + 1
    quiet = np.convolve((db < level - QUIET_DB).astype(float), np.ones(2 * half + 1), "same") == 2 * half + 1
    centres = np.flatnonzero(quiet)[::5]
    if len(centres) == 0 or np.median(db[centres]) < level - AUDIBLE_DB:
        return None
    freqs = np.fft.rfftfreq(WINDOW, 1 / RATE)
    bands = [(freqs >= low) & (freqs < 2 * low) for low in BANDS]
    starts = centres * HOP + HOP // 2 - WINDOW // 2
    power = np.abs(np.fft.rfft(x[starts[:, None] + np.arange(WINDOW)] * np.hanning(WINDOW), axis=1)) ** 2 + 1e-20
    least = np.min([np.exp(np.mean(np.log(power[:, b]), axis=1)) / np.mean(power[:, b], axis=1) for b in bands], axis=0)
    return float(np.median(least))


def pitch(x: np.ndarray) -> np.ndarray:
    """F0 in Hz per 20 ms, NaN unvoiced: the first autocorrelation peak within 10% of the best, never an octave low."""
    size, hop, low, high = 1764, 882, RATE // 400, RATE // 60
    n = max(0, 1 + (len(x) - size) // hop)
    window = np.hanning(size)
    lagged = np.fft.irfft(np.abs(np.fft.rfft(window, 4096)) ** 2)[: high + 1]
    f0 = np.full(n, np.nan)
    for at in range(0, n, 256):
        frames = x[(np.arange(at, min(n, at + 256)) * hop)[:, None] + np.arange(size)] * window
        ac = np.fft.irfft(np.abs(np.fft.rfft(frames, 4096, axis=1)) ** 2, axis=1)[:, : high + 1]
        r = ac[:, low:] / np.maximum(ac[:, :1], 1e-12) / (lagged[low:] / lagged[0])
        best = r.max(axis=1)
        first = np.argmax(r >= 0.9 * best[:, None], axis=1)
        # the crossing is on the peak's rising side; the peak is within half the shortest period after it
        rising = np.minimum(first[:, None] + np.arange(low // 2), r.shape[1] - 1)
        lag = low + first + np.argmax(np.take_along_axis(r, rising, axis=1), axis=1)
        f0[at : at + len(frames)] = np.where(best >= 0.5, RATE / lag, np.nan)
    return f0


def pitch_spread(x: np.ndarray, db: np.ndarray, level: float) -> tuple[float, int]:
    """Semitones between the 20th and 80th percentile, by voiced time, of each stretch of speech's median pitch."""
    f0 = pitch(x)
    centre = 2 * np.arange(len(f0)) + 2
    heard = ~np.isnan(f0) & (db[centre] >= level - 15)
    talk = np.ones(len(db), bool)
    for start, end in pauses(db < level - QUIET_DB):
        talk[start:end] = False
    medians, weights = [], []
    for start, end in runs(talk):
        voiced = f0[heard & (centre >= start) & (centre < end)]
        if len(voiced) >= 25:
            medians.append(12 * np.log2(np.median(voiced) / 55))
            weights.append(len(voiced))
    if len(medians) < 2:
        return 0.0, len(medians)
    order = np.argsort(medians)
    share = np.cumsum(np.array(weights)[order]) / sum(weights)
    sorted_medians = np.array(medians)[order]
    return float(sorted_medians[np.argmax(share >= 0.8)] - sorted_medians[np.argmax(share >= 0.2)]), len(medians)


def noise(db: np.ndarray, level: float) -> float:
    """Decibels from the speech level down to the quietest tenth of the frames; with too few pauses it reads low."""
    snr = level - float(np.percentile(db, 10))
    if snr < MIN_SNR_DB:
        raise SystemExit(f"the background is {snr:.1f} dB under the speech; a voice needs {MIN_SNR_DB:.0f} dB: record somewhere quieter")
    return snr


def prepare(x: np.ndarray, rate: int) -> tuple[np.ndarray, dict]:
    """The clip ready to encode and what was measured, or SystemExit saying why it cannot make a voice."""
    if len(x) < MIN_SECONDS * rate:
        raise SystemExit(f"the clip is {len(x) / rate:.2f} s long; a voice needs {MIN_SECONDS:.0f} s")
    if np.max(np.abs(x)) < 1e-4:
        raise SystemExit("the clip is silent")
    channels = x.reshape(len(x), -1).T
    if (flat := sum(clipped(channel) for channel in channels)) >= CLIP_RUNS:
        raise SystemExit(f"the clip is clipped ({flat} flat tops): record it again with less gain")
    x = resample(channels.mean(axis=0), rate, RATE)
    db = frame_db(x)
    level = speech_level(db)
    # first on the whole take, as noise hides the pauses the cut looks for; then on what is kept
    noise(db, level)
    start, end, dropped = cut(db, level)
    x, db = x[start * HOP : end * HOP], db[start:end]
    level = speech_level(db)
    snr = noise(db, level)
    tonal = flatness(x, db, level)
    if tonal is not None and tonal < MIN_FLATNESS:
        raise SystemExit(f"the pauses hold a steady tone (band flatness {tonal:.3f}): music or hum under the voice")
    spread, stretches = pitch_spread(x, db, level)
    if spread > MAX_SPREAD_ST:
        raise SystemExit(f"the pitch moves {spread:.1f} semitones between stretches of speech: more than one speaker")
    gain = min(LEVEL_DBFS - level, PEAK_DBFS - 20 * float(np.log10(np.max(np.abs(x)))))
    report = {
        "seconds": round(len(x) / RATE, 2),
        "source_seconds": [round(start / 100, 2), round(end / 100, 2)],
        "speech_left_out_seconds": round(dropped / 100, 2),
        "level_dbfs": round(level + gain, 2),
        "gain_db": round(gain, 2),
        "snr_db": round(snr, 1),
        "pause_flatness": None if tonal is None else round(tonal, 3),
        "pitch_spread_semitones": round(spread, 2),
        "speech_stretches": stretches,
    }
    return x * 10 ** (gain / 20), report


def transcribe(audio: np.ndarray, model: Path) -> str:
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=str(model), device="cpu")
    return asr({"raw": resample(audio, RATE, 16000).astype(np.float32), "sampling_rate": 16000})["text"].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip", type=Path, help="any file libsndfile reads: WAV, FLAC, Ogg, MP3")
    parser.add_argument("--out", type=Path, required=True, help="the prepared clip, NAME.wav; its transcript goes next to it")
    words = parser.add_mutually_exclusive_group(required=True)
    words.add_argument("--text", type=Path, help="the clip's exact transcript (UTF-8); refused when the cut leaves speech out")
    words.add_argument("--asr", type=Path, help="a local Whisper model directory: writes NAME.draft.txt to correct and save as NAME.txt")
    args = parser.parse_args()
    confirmed = args.out.with_suffix(".txt")
    if args.asr and confirmed.exists():
        raise SystemExit(f"{confirmed} holds a confirmed transcript of an earlier clip: remove it, or prepare this take under another name")
    x, rate = sf.read(args.clip, always_2d=False)
    audio, report = prepare(x, rate)
    if args.text:
        transcript = args.text.read_text(encoding="utf-8").strip()
        if not transcript:
            raise SystemExit(f"{args.text} holds no transcript")
        if report["speech_left_out_seconds"]:
            raise SystemExit(f"the cut at {report['source_seconds'][1]} s leaves speech out, so {args.text} no longer matches: use --asr")
        transcript_path = confirmed
        # a draft of an earlier take would sit next to this transcript
        args.out.with_suffix(".draft.txt").unlink(missing_ok=True)
    else:
        transcript = transcribe(audio, args.asr)
        transcript_path = args.out.with_suffix(".draft.txt")
    sf.write(args.out, audio, RATE, subtype="PCM_16")
    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    print(json.dumps({**report, "clip": str(args.out), "transcript": str(transcript_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
