#!/usr/bin/env python3
"""Render FastH3 clips end to end on the native transformer: export, denoise, decode, one prompt or several.

  h3_render.py --prompt-file p.txt --seed 4 --out clip.mp4
  h3_render.py --job a.txt:4:a.mp4 --job b.txt:5:b.mp4 --width 1280 --height 768 --crop 1280x720 --frames 243

Each clip is three stages: tools/zig/h3_case.py exports the prompt's rows, tables and noise (Python: the text
encoder and the checkpoint's small projections), tf-h3-dit runs the transformer's passes, and h3_case.py --decode
turns the rows into an MP4 with sound (Python: the video and audio decoders). With several jobs the stages
overlap: the next clip's export runs while this one denoises, and this clip's decode runs while the next one
denoises (--no-overlap runs them one after another). The denoise and the video decode both use the GPU, so an
overlapped decode slows the passes beside it; the per-stage times printed at the end say what it cost.

Environment: H3_PYTHON (the Python with MLX, TensorFold 0.6 and minimax-h3-mlx importable; PYTHONPATH is passed
through), H3_MODEL_DIR, FASTH3_DIR, H3_TOOLS (the folder holding h3_generate_dev.py), TF_H3_DIT (the native tool).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--job", action="append", default=[], metavar="PROMPT:SEED:OUT", help="one clip; repeat for a batch")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--passes", type=int, default=0, help="transformer passes, when not the checkpoint's own count")
    parser.add_argument("--crop", help="WxH: centre-crop the decoded frames")
    parser.add_argument("--options", default="", help="tf-h3-dit's last argument (kernel options), if any")
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep each clip's case folder")
    args = parser.parse_args()

    jobs = [tuple(j.rsplit(":", 2)) for j in args.job]
    if args.prompt_file:
        jobs.insert(0, (args.prompt_file, str(args.seed), args.out or "h3.mp4"))
    if not jobs:
        parser.error("give --prompt-file or --job")
    home = Path.home()
    python = os.environ.get("H3_PYTHON", sys.executable)
    model = os.environ.get("H3_MODEL_DIR", str(home / "h3-models/MiniMax-H3"))
    fasth3 = os.environ.get("FASTH3_DIR", str(home / "h3-models/FastH3-8-Step-V2"))
    tools = os.environ.get("H3_TOOLS", str(home / "TensorFold/tools"))
    native = os.environ.get("TF_H3_DIT", str(REPO / "zig-out/bin/tf-h3-dit"))
    shards = len(list((Path(fasth3) / "transformer").glob("diffusion_pytorch_model-*.safetensors")))
    case_py = [python, str(HERE / "h3_case.py"), model, fasth3]
    size = ["--width", str(args.width), "--height", str(args.height), "--frames", str(args.frames)]
    spent = [dict() for _ in jobs]
    failed: list[str] = []

    def run(index: int, stage: str, command: list[str], log) -> bool:
        mark = time.perf_counter()
        done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        spent[index][stage] = round(time.perf_counter() - mark, 1)
        if done.returncode:
            failed.append(f"clip {index + 1} {stage}")
        return done.returncode == 0

    folders = [Path(tempfile.mkdtemp(prefix="h3case_", dir=str(Path(out).resolve().parent))) for _, _, out in jobs]
    logs = [open(str(Path(out).with_suffix(".log")), "w") for _, _, out in jobs]

    def export(index: int) -> bool:
        prompt, seed, _ = jobs[index]
        extra = ["--steps", str(args.passes)] if args.passes else []
        return run(index, "export", case_py + [str(folders[index]), "--tools", tools, "--prompt-file", prompt, "--seed", seed,
                                               "--no-reference", *size, *extra], logs[index])

    def decode(index: int) -> None:
        _, _, out = jobs[index]
        extra = ["--crop", args.crop] if args.crop else []
        if run(index, "decode", case_py + [str(folders[index]), "--tools", tools, *size, "--decode", str(folders[index] / "zig"),
                                           "--mp4", out, *extra], logs[index]) and not args.keep:
            for leftover in folders[index].iterdir():
                leftover.unlink()
            folders[index].rmdir()

    started = time.perf_counter()
    ready: dict[int, bool] = {0: export(0)}
    decoding: list[threading.Thread] = []
    for index in range(len(jobs)):
        ahead = None
        if index + 1 < len(jobs):
            if args.no_overlap:
                pass
            else:
                ahead = threading.Thread(target=lambda i=index + 1: ready.__setitem__(i, export(i)))
                ahead.start()
        if ready.get(index):
            command = [native, str(Path(fasth3) / "transformer"), str(shards), str(folders[index] / "case.safetensors"),
                       str(folders[index] / "zig")] + ([args.options] if args.options else [])
            if run(index, "denoise", command, logs[index]):
                (folders[index] / "case.safetensors").unlink()
                if args.no_overlap or index + 1 == len(jobs):
                    decode(index)
                else:
                    worker = threading.Thread(target=decode, args=(index,))
                    worker.start()
                    decoding.append(worker)
        if ahead is not None:
            ahead.join()
        elif index + 1 < len(jobs):
            ready[index + 1] = export(index + 1)
    for worker in decoding:
        worker.join()
    total = round(time.perf_counter() - started, 1)
    for log in logs:
        log.close()
    for (_, seed, out), parts in zip(jobs, spent, strict=True):
        passes = [float(line.split(": ")[1].split(" s wall")[0]) for line in Path(out).with_suffix(".log").read_text().splitlines()
                  if line.startswith("step ") and " s wall" in line]
        parts["per_pass"] = round(sum(passes) / len(passes), 2) if passes else None
        print(json.dumps({"out": out, "seed": int(seed), **parts}))
    print(json.dumps({"clips": len(jobs), "total_s": total, "overlap": not args.no_overlap, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
