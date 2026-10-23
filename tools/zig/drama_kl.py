"""KL(A||B) of two aligned tf-drama-probe outputs' semantic logits at T=1, per request and pooled, with tower NLL."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COLUMNS = 4097
HEADS = 9
CODEBOOKS = 10
WORST = 10


def steps_of(finish: str, frames: int) -> int:
    """The steps a run of `frames` frames took: the EOS decision is a step of its own, a capped run has none."""
    return frames + 1 if finish == "eos" else frames


def requests(out: Path) -> dict[str, dict]:
    """A probe output's manifest entries by id, refused when an id repeats or steps disagree with finish."""
    entries = json.loads((out / "manifest.json").read_text())["requests"]
    found = {}
    for e in entries:
        if e["id"] in found:
            raise SystemExit(f"{out}: request {e['id']} appears twice in manifest.json")
        if e["finish"] not in ("eos", "length") or e["steps"] != steps_of(e["finish"], e["frames"]):
            raise SystemExit(f"{out}: {e['id']}: {e['steps']} steps disagree with {e['frames']} frames ending {e['finish']}")
        found[e["id"]] = e
    return found


def rows(path: Path, width: int, count: int) -> np.ndarray:
    """A probe file of exactly `count` finite float32 rows of `width`, as float64; anything else is refused."""
    size = path.stat().st_size
    if size % (width * 4):
        raise SystemExit(f"{path}: {size} bytes, not whole rows of {width} float32")
    data = np.fromfile(path, dtype="<f4")
    if data.size != count * width:
        raise SystemExit(f"{path}: {data.size} floats, the manifest says {count} rows of {width}")
    data = data.reshape(count, width)
    bad = np.argwhere(~np.isfinite(data))
    if bad.size:
        row, col = bad[0]
        raise SystemExit(f"{path}: row {row} column {col} is {data[row, col]}, not finite")
    return data.astype(np.float64)


def check_codes(out: Path, rid: str, entry: dict) -> None:
    """The codes file holds the manifest's frames and hashes to its codes_sha256."""
    data = (out / "codes" / f"{rid}.u32").read_bytes()
    if len(data) != entry["frames"] * CODEBOOKS * 4:
        raise SystemExit(f"{out}/codes/{rid}.u32: {len(data)} bytes, the manifest says {entry['frames']} frames")
    if hashlib.sha256(data).hexdigest() != entry["codes_sha256"]:
        raise SystemExit(f"{out}/codes/{rid}.u32: does not match the manifest's codes_sha256")


def log_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def kl_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """KL(softmax(a) || softmax(b)) of each finite row, in nats; a column A gives no probability adds nothing."""
    la, lb = log_softmax(a), log_softmax(b)
    p = np.exp(la)
    return np.where(p > 0, p * (la - lb), 0.0).sum(axis=1)


def summary(kl: np.ndarray, agree: np.ndarray) -> dict:
    return {"rows": int(kl.size), "mean": float(kl.mean()), "p50": float(np.percentile(kl, 50)),
            "p99": float(np.percentile(kl, 99)), "max": float(kl.max()), "top1_agreement": float(agree.mean())}


def tower(a_out: Path, b_out: Path, rid: str, frames: int) -> dict | None:
    """Per codebook mean NLL of each run that holds tower NLL, and B - A when both do."""
    found = {side: rows(out / "tower" / f"{rid}.f32", HEADS, frames) for side, out in (("a", a_out), ("b", b_out))
             if (out / "tower" / f"{rid}.f32").exists()}
    if not found or frames == 0:
        return None
    means = {side: nll.mean(axis=0) for side, nll in found.items()}
    report = {f"{side}_mean_nll": [float(v) for v in m] for side, m in means.items()}
    if len(means) == 2:
        report["delta_mean_nll"] = [float(v) for v in means["b"] - means["a"]]
    return report


def compare(a_out: Path, b_out: Path) -> dict:
    """Two runs of the same requests, each file checked against its manifest: KL, agreement and worst rows."""
    a_requests, b_requests = requests(a_out), requests(b_out)
    if a_requests.keys() != b_requests.keys():
        only = sorted(a_requests.keys() ^ b_requests.keys())
        raise SystemExit(f"the runs hold different requests: {', '.join(only)} in only one")
    report = {"a": str(a_out), "b": str(b_out), "columns": COLUMNS, "requests": {}}
    pooled_kl, pooled_agree, first = [], [], []
    for rid, a_entry in a_requests.items():
        b_entry = b_requests[rid]
        for key in ("prompt_sha256", "prompt_rows", "codes_sha256", "frames", "steps", "finish"):
            if a_entry[key] != b_entry[key]:
                raise SystemExit(f"{rid}: misaligned runs, {key} {a_entry[key]} against {b_entry[key]}")
        check_codes(a_out, rid, a_entry)
        check_codes(b_out, rid, b_entry)
        a = rows(a_out / "logits" / f"{rid}.f32", COLUMNS, a_entry["steps"])
        b = rows(b_out / "logits" / f"{rid}.f32", COLUMNS, b_entry["steps"])
        kl = kl_rows(a, b)
        top_a, top_b = a.argmax(axis=1), b.argmax(axis=1)
        agree = top_a == top_b
        worst = np.argsort(-kl, kind="stable")[:WORST]
        entry = summary(kl, agree) | {
            "frame0_kl": float(kl[0]),
            "worst": [{"row": int(i), "kl": float(kl[i]), "top1": [int(top_a[i]), int(top_b[i])]} for i in worst]}
        nll = tower(a_out, b_out, rid, a_entry["frames"])
        if nll is not None:
            entry["tower"] = nll
        report["requests"][rid] = entry
        pooled_kl.append(kl)
        pooled_agree.append(agree)
        first.append(kl[0])
    if not pooled_kl:
        raise SystemExit("the runs hold no request")
    report["pooled"] = summary(np.concatenate(pooled_kl), np.concatenate(pooled_agree))
    report["frame0"] = {"rows": len(first), "mean": float(np.mean(first)), "max": float(np.max(first))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a_out", type=Path, help="the reference run (P), e.g. the FP8 checkpoint's")
    parser.add_argument("b_out", type=Path, help="the compared run (Q), teacher-forced with A's codes")
    parser.add_argument("--out", type=Path, help="write the whole report as JSON here")
    args = parser.parse_args()
    report = compare(args.a_out, args.b_out)
    for rid, r in report["requests"].items():
        print(f"{rid}: {r['rows']} rows, KL mean {r['mean']:.6f} p50 {r['p50']:.6f} p99 {r['p99']:.6f} "
              f"max {r['max']:.6f}, frame 0 {r['frame0_kl']:.6f}, top-1 {r['top1_agreement']:.4f}")
        if "delta_mean_nll" in r.get("tower", {}):
            print(f"  tower NLL B - A by codebook: {' '.join(f'{v:+.4f}' for v in r['tower']['delta_mean_nll'])}")
    p, f = report["pooled"], report["frame0"]
    print(f"pooled: {p['rows']} rows, KL mean {p['mean']:.6f} p50 {p['p50']:.6f} p99 {p['p99']:.6f} "
          f"max {p['max']:.6f}, top-1 {p['top1_agreement']:.4f}; frame 0 mean {f['mean']:.6f} max {f['max']:.6f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
