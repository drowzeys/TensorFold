"""Capture drama_register_fixture.json on the CPU: the calibration's numpy register measure on synthetic tones."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parents[2] / "zig/src/server/drama_register_fixture.json"
# The rule as calibrated on real replies, at register.py's 210 Hz line: an opening 4 st up with 20% of frames over it.
ST, HIGH = 4.0, 0.2
AMP = 0.3

# Each part is [seconds, start Hz, end Hz]: a falling glide of 4 harmonics, or a rest of zeros when both are 0.
CALM = [[1.6, 180, 150], [0.3, 0, 0], [1.4, 178, 148], [0.25, 0, 0], [1.8, 182, 152], [0.3, 0, 0], [1.2, 176, 150],
        [0.3, 0, 0], [1.5, 180, 150]]
PLAIN = [1, 1 / 2, 1 / 3, 1 / 4]
CASES = [
    ("calm", 44100, CALM, PLAIN),
    ("jump", 44100, CALM[:6] + [[1.4, 240, 205]] + CALM[7:], PLAIN),
    ("small-jump", 44100, CALM[:6] + [[1.4, 195, 170]] + CALM[7:], PLAIN),
    ("jump-under-the-line", 16000, [[1.6, 140, 120], [0.3, 0, 0], [1.4, 138, 118], [0.3, 0, 0], [1.5, 142, 120],
                                    [0.3, 0, 0], [1.4, 180, 160], [0.3, 0, 0], [1.5, 140, 120]], PLAIN),
    ("first-phrase-high", 48000, [[1.4, 240, 205]] + CALM[1:], PLAIN),
    # a 0.08 s gap does not part phrases, 0.12 s does, and a 0.3 s piece joins the phrase before it
    ("short-rests", 44100, [[1.6, 180, 150], [0.08, 0, 0], [0.9, 170, 150], [0.3, 0, 0], [1.4, 178, 148], [0.3, 0, 0],
                            [0.3, 160, 150], [0.12, 0, 0], [1.4, 240, 205], [0.3, 0, 0], [1.5, 180, 150]], PLAIN),
    # a weak fundamental under a strong second harmonic: the half lag scores about 0.85 of the peak, short of the 0.9
    ("weak-fundamental", 16000, [[1.6, 120, 100], [0.3, 0, 0], [1.4, 118, 98], [0.3, 0, 0], [1.5, 122, 100]], [0.2, 1, 0.2, 0]),
]


def tones(rate: int, parts: list[list[float]], harmonics: list[float]) -> np.ndarray:
    """The parts as float32 samples: each glide's phase is its frequency's integral from the part's own start."""
    out = []
    for seconds, start, end in parts:
        n = round(seconds * rate)
        t = np.arange(n) / rate
        if start == 0:
            out.append(np.zeros(n))
            continue
        phase = start * t + (end - start) * t * t / (2 * seconds)
        out.append(AMP * sum(w * np.sin(2 * math.pi * h * phase) for h, w in enumerate(harmonics, 1)))
    return np.concatenate(out).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="the calibration's register.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("register", args.reference)
    register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(register)
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, rate, parts, harmonics in CASES:
            path = Path(tmp) / f"{name}.wav"
            sf.write(path, tones(rate, parts, harmonics), rate, subtype="FLOAT")
            median, rows, _, segs = register.analyse(register.load16(path))
            phrases = [{"start": int(a), "end": int(b), "opening": r["opening_st"], "excess": r["excess_st"], "high": r["high"]} for r, (a, b) in zip(rows, segs)]
            cases.append({"name": name, "rate": rate, "parts": parts, "harmonics": harmonics, "median_hz": round(median, 3), "phrases": phrases,
                          "trips": register.flags(rows, "excess_st", ST, HIGH)})
    doc = {"source": "register.py analyse() and flags('excess_st', 4.0, 0.2) at 210 Hz, sha256 " + hashlib.sha256(args.reference.read_bytes()).hexdigest(),
           "amp": AMP, "cases": cases}
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps([{"name": c["name"], "phrases": len(c["phrases"]), "trips": c["trips"]} for c in cases]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
