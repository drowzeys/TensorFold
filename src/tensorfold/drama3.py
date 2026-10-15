"""Fish Audio Drama 3 text to speech on Apple Silicon (MLX). Built with Fish Audio.

    python -m tensorfold.drama3 --model ~/models/drama-3-mlx --text "Hello there." --out hello.wav
    python -m tensorfold.drama3 --text "<|speaker:0|> [whispering] Over here. <|speaker:1|> Where?" --out talk.wav
    python -m tensorfold.drama3 --text "..." --ref-audio voice.wav --ref-text "what the clip says" --out clone.wav

Speaker tags <|speaker:0|> .. <|speaker:15|> switch voices; [bracketed directions] steer delivery. Only clone a
real person's voice with their consent. Convert Fish's checkpoint once with
``python -m tensorfold.families.drama3.convert <fish dir> <mlx dir>``. See DRAMA3.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_DRAMA3_MODEL", "~/models/drama-3-mlx")


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tensorfold.drama3", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"converted MLX model folder (default {DEFAULT_MODEL})")
    text = p.add_mutually_exclusive_group(required=True)
    text.add_argument("--text", help="text to speak")
    text.add_argument("--text-file", type=Path, help="read the text from a file")
    p.add_argument("--ref-audio", action="append", default=[], type=Path,
                   help="voice reference clip (repeatable, paired with --ref-text)")
    p.add_argument("--ref-text", action="append", default=[], help="transcript of each --ref-audio clip")
    p.add_argument("--ref-codes", action="append", default=[], type=Path,
                   help="a reference given as saved codes (.npy [10, frames]) instead of audio, with --ref-text")
    p.add_argument("--out", type=Path, required=True, help="output .wav (44.1 kHz mono)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="argmax every draw (deterministic; RAS has no effect)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=None, help="frame cap a chunk (default: from text length)")
    p.add_argument("--max-chars", type=int, default=300, help="long text is split into chunks of about this size")
    p.add_argument("--save-codes", type=Path, help="also save the generated frames (.npy [T, 11])")
    p.add_argument("--stats", type=Path, help="write timing and memory as JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    import mlx.core as mx
    import numpy as np

    from tensorfold.families.drama3.codec import load_audio, save_wav
    from tensorfold.families.drama3.generate import FRAME_RATE, Drama3Generator
    from tensorfold.families.drama3.prompt import Reference
    from tensorfold.families.drama3.sampling import Sampling

    text = args.text if args.text is not None else args.text_file.read_text()
    sources = [("audio", a) for a in args.ref_audio] + [("codes", c) for c in args.ref_codes]
    if len(sources) != len(args.ref_text):
        raise SystemExit("[drama3] give one --ref-text for each --ref-audio / --ref-codes")
    print("[drama3] Built with Fish Audio. Fish Audio Drama 3 runs locally under the Fish Audio Community "
          "License; clone only voices you have consent to use.", flush=True)
    gen = Drama3Generator(args.model)
    print(f"[drama3] loaded {gen.model_dir} in {gen.load_s:.1f}s", flush=True)
    references = []
    for (kind, path), transcript in zip(sources, args.ref_text):
        if kind == "audio":
            started = time.perf_counter()
            codes = np.array(gen.codec.encode(load_audio(path)))
            print(f"[drama3] encoded {path.name}: {codes.shape[1]} frames in {time.perf_counter() - started:.2f}s",
                  flush=True)
        else:
            codes = np.load(path)
        references.append(Reference(transcript=transcript, codes=codes))
    sampling = Sampling(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, greedy=args.greedy)
    mx.reset_peak_memory()
    started = time.perf_counter()
    audio, results = gen.synthesize(text, references=references, sampling=sampling, seed=args.seed,
                                    max_chars=args.max_chars, max_frames=args.max_frames)
    total_s = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_wav(args.out, audio)
    frames = sum(len(r.frames) for r in results)
    decode_s = sum(r.decode_s for r in results)
    seconds = len(audio) / 44100
    stats = {"frames": frames, "audio_s": round(seconds, 3), "decode_s": round(decode_s, 3),
             "frames_per_s": round(frames / max(decode_s, 1e-9), 2),
             "prefill_s": round(sum(r.prefill_s for r in results), 3),
             "first_frame_s": round(results[0].first_frame_s, 3) if results else None,
             "total_s": round(total_s, 3), "rtf": round(total_s / max(seconds, 1e-9), 3),
             "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "chunks": len(results),
             "finish": [r.finish for r in results]}
    print(f"[drama3] wrote {args.out}: {seconds:.1f}s audio, {frames} frames at {stats['frames_per_s']} frames/s "
          f"(real time is {FRAME_RATE:.1f}); total {total_s:.1f}s, RTF {stats['rtf']}; peak "
          f"{stats['peak_memory_gb']} GB", flush=True)
    if args.save_codes:
        np.save(args.save_codes, np.concatenate([r.frames for r in results]) if results else np.zeros((0, 11)))
    if args.stats:
        args.stats.write_text(json.dumps(stats, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
