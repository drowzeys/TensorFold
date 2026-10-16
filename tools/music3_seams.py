"""MiniMax Music 3: are the window seams audible? Spectral continuity at the seams against the rest of the song.

    python tools/music3_seams.py song.wav --stats song.json      # seams from the stats ("seams_s", sequential mode)
    python tools/music3_seams.py song.wav --frames 1500          # an averaged-mode song: the nominal seams of
                                                                 # sequential windows over the same frames (control)

Log-magnitude spectra (46 ms frames, 11.6 ms hop, mono) give a spectral flux per hop (mean absolute change of the
log spectrum). For each seam the largest flux within +-1 hop of it is compared with the song's flux distribution:
its percentile, and its ratio to the median. Seams that sit at the song's own level (percentiles spread like any
other moment, ratios near 1-2) are not a discontinuity; a click or a jump in timbre lands at the top percentiles.
Also: the level step across each seam (RMS 100 ms before vs after, dB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RATE = 44100
N_FFT, HOP = 2048, 512


def nominal_seams(frames: int) -> list[float]:
    """Where sequential windows would meet over ``frames`` AR frames (seconds), for an averaged-mode control."""

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tensorfold.families.music3.generate import sequential_crop, sequential_starts
    from tensorfold.families.music3.model import frames_to_latents

    starts = sequential_starts(frames)
    out, sent = [], 0
    for i, start in enumerate(starts):
        length = frames_to_latents(min(start + 200, frames) - start)
        a, b = sequential_crop(length, i == 0, i == len(starts) - 1)
        if i:
            out.append(sent * 512 / RATE)
        sent += b - a
    return out


def flux(mono: np.ndarray) -> np.ndarray:
    window = np.hanning(N_FFT).astype(np.float32)
    count = 1 + (len(mono) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None] + HOP * np.arange(count)[:, None]
    spec = np.log(np.abs(np.fft.rfft(mono[idx] * window, axis=1)) + 1e-5)
    return np.abs(np.diff(spec, axis=0)).mean(axis=1)                   # flux[i]: between frames i and i + 1


def analyse(wav: Path, seams: list[float]) -> dict:
    import soundfile as sf

    audio, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    f = flux(mono)
    centre = (np.arange(len(f)) + 1) * HOP + N_FFT // 2                  # sample at the middle of each flux step
    rows = []
    for seam in seams:
        at = int(round(seam * rate))
        if at <= N_FFT or at >= len(mono) - N_FFT:
            continue
        i = int(np.argmin(np.abs(centre - at)))
        peak = float(f[max(0, i - 1):i + 2].max())
        w = int(0.1 * rate)
        before = np.sqrt((mono[at - w:at] ** 2).mean() + 1e-12)
        after = np.sqrt((mono[at:at + w] ** 2).mean() + 1e-12)
        rows.append({"seam_s": round(seam, 3), "flux_pct": round(float((f < peak).mean() * 100), 1),
                     "flux_ratio": round(peak / float(np.median(f)), 2),
                     "level_step_db": round(float(20 * np.log10(after / before)), 2)})
    pct = [r["flux_pct"] for r in rows]
    return {"wav": str(wav), "seams": len(rows),
            "flux_pct_mean": round(float(np.mean(pct)), 1) if rows else None,
            "flux_pct_max": max(pct) if rows else None,
            "seams_over_p99": sum(p > 99 for p in pct),
            "flux_ratio_mean": round(float(np.mean([r["flux_ratio"] for r in rows])), 2) if rows else None,
            "level_step_db_abs_mean": round(float(np.mean([abs(r["level_step_db"]) for r in rows])), 2) if rows else None,
            "rows": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", type=Path)
    p.add_argument("--stats", type=Path, help="the song's --stats JSON (seams_s, or frames for a control)")
    p.add_argument("--frames", type=int, help="AR frames: nominal sequential seams (control for an averaged song)")
    p.add_argument("--rows", action="store_true", help="print every seam")
    a = p.parse_args(argv)
    stats = json.loads(a.stats.read_text()) if a.stats else {}
    if stats.get("seams_s"):
        seams, kind = stats["seams_s"], "seams"
    else:
        seams, kind = nominal_seams(a.frames or stats["frames"]), "control"
    out = {"kind": kind, **analyse(a.wav, seams)}
    if not a.rows:
        out.pop("rows")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
