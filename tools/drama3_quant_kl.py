"""Expert-quantization quality: teacher-forced logits of a converted model, and KL / top-1 against a reference.

    python tools/drama3_quant_kl.py dump --model ~/models/drama-3-mlx --dump dumps/plain --out affine8_plain.npz
    python tools/drama3_quant_kl.py compare --ref bf16_plain.npz affine8_plain.npz mxfp8_plain.npz

``dump`` runs the oracle dump's prompt and then its frames, teacher-forced, through the model and saves, for every
step, the candidate (main token) logits and the nine code-head logits (each head given the oracle's codes).
``compare`` reports, per model against the reference (the bf16-expert conversion): mean / max KL(ref || model) of the
temperature-1 distributions, and how often the argmax agrees, for the main token and the code heads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def dump(model_dir: str, dump_dir: Path, out: Path, steps: int) -> None:
    import mlx.core as mx

    from tensorfold.families.drama3.generate import Drama3Generator

    gen = Drama3Generator(model_dir, load_codec=False)
    model, c = gen.model, gen.model.config
    d = mx.load(str(dump_dir / "vectors.safetensors"))
    rows = np.asarray(d["prompt_rows"]).astype(np.int64)
    frames = np.asarray(d["frames_all"] if "frames_all" in d else d["frames"]).astype(np.int64)
    frames = frames[:steps]
    bcache, tcaches = model.make_cache()
    x = mx.array(rows.astype(np.int32))[None]
    h = model.backbone_hidden(x, bcache)
    model.tower_prefill(h[:, :-1], x[:, :-1], x[:, 1:], tcaches)
    hidden, frame_in = h[:, -1:], x[:, -1:]
    main_logits, head_logits = [], []
    for n, fr in enumerate(frames):
        main_logits.append(np.array(model.candidate_logits(hidden[:, -1]))[0])
        heads = []

        def draw_code(lg, r, fr=fr, heads=heads):
            heads.append(np.array(lg)[0])
            return mx.array([int(fr[2 + r])], dtype=mx.int32)

        mx.eval(model.tower_step(hidden, frame_in, mx.array([int(fr[0])]), mx.array([int(fr[1])]), tcaches,
                                 draw_code))
        head_logits.append(np.stack(heads))
        frame_in = mx.array(fr.astype(np.int32))[None, None]
        hidden = model.backbone_hidden(frame_in, bcache)
    np.savez(out, main=np.stack(main_logits), heads=np.stack(head_logits),
             experts=json.dumps(model.config.tensorfold.get("experts")))
    print(f"[quant_kl] {out}: {len(frames)} steps, experts {model.config.tensorfold.get('experts')}", flush=True)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def kl_top1(ref: np.ndarray, other: np.ndarray) -> dict:
    lp, lq = _log_softmax(ref), _log_softmax(other)
    kl = (np.exp(lp) * (lp - lq)).sum(axis=-1)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top1": float((ref.argmax(-1) == other.argmax(-1)).mean())}


def compare(ref_path: Path, others: list[Path]) -> dict:
    ref = np.load(ref_path)
    report = {}
    for path in others:
        o = np.load(path)
        n = min(len(ref["main"]), len(o["main"]))
        report[path.name] = {"steps": n, "main": kl_top1(ref["main"][:n], o["main"][:n]),
                             "heads": kl_top1(ref["heads"][:n].reshape(-1, ref["heads"].shape[-1]),
                                              o["heads"][:n].reshape(-1, o["heads"].shape[-1]))}
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--model", required=True)
    d.add_argument("--dump", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--steps", type=int, default=200)
    c = sub.add_parser("compare")
    c.add_argument("--ref", type=Path, required=True)
    c.add_argument("others", type=Path, nargs="+")
    args = p.parse_args(argv)
    if args.cmd == "dump":
        dump(args.model, args.dump, args.out, args.steps)
    else:
        print(json.dumps(compare(args.ref, args.others), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
