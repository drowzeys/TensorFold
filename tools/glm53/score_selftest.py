#!/usr/bin/env python3
"""No GPU: score_format.py's files against tf-glm53-generate's reader, and the tool's KL against numpy's.

  score_selftest.py TF_GLM53_GENERATE [WORKDIR]

Writes a toy rows file, a reference from random "reference" logits (records_from_logits), the four ranks' partials
from noisy "served" logits computed here in numpy (not with score.zig's code), then runs --mode score-info and
--mode score-combine and checks the KL, the bounds and the top-1 agreement. Prints `SELFTEST OK|FAIL`.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_format as sf

PART_HEAD = struct.Struct("<8sIIIIIIQQQI12x")
PART = struct.Struct("<dddffIf")
assert PART_HEAD.size == 72 and PART.size == 40


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def run(tool, work, topk, world=4, vocab=4096, seed=1):
    rng = np.random.default_rng(seed)
    lens = [40, 17, 64, 33, 9]
    rows = [rng.integers(0, vocab, n).tolist() for n in lens]
    sets = [("alpha", 2), ("beta", 3)]
    rows_path, ref_path = os.path.join(work, "rows.bin"), os.path.join(work, f"ref{topk}.bin")
    sf.write_rows(rows_path, vocab, sets, rows)
    w = sf.ReferenceWriter(rows_path, ref_path, topk)
    share = vocab // world
    parts = [[] for _ in range(world)]
    kl_sum, agree, n_pos = 0.0, 0, 0
    for r in rows:
        n = len(r) - 1
        ref = rng.normal(0, 3, (n, vocab)).astype(np.float32)
        srv = (ref + rng.normal(0, 0.8, (n, vocab))).astype(np.float32)
        labels = np.asarray(r[1:])
        recs = sf.records_from_logits(ref, labels, topk)
        w.add_row(recs)
        lp, lq = log_softmax(ref), log_softmax(srv)
        kl_sum += float((np.exp(lp) * (lp - lq)).sum())
        agree += int((lp.argmax(1) == srv.argmax(1)).sum())
        n_pos += n
        rec = np.frombuffer(recs, dtype=np.uint8).reshape(n, sf.pos_bytes(topk))
        ids = rec[:, 16:16 + 4 * topk].copy().view("<u4")
        p_top = np.exp(rec[:, 16 + 4 * topk:].copy().view("<f4").astype(np.float64))
        for k in range(world):
            lg = srv[:, k * share:(k + 1) * share].astype(np.float64)
            m = lg.max(1)
            mine = (ids >= k * share) & (ids < (k + 1) * share)
            at = np.clip(ids.astype(np.int64) - k * share, 0, share - 1)
            got = np.take_along_axis(lg, at, 1)
            for i in range(n):
                lab = labels[i] - k * share
                parts[k].append(PART.pack(float(np.exp(lg[i] - m[i]).sum()), float((p_top[i] * got[i])[mine[i]].sum()),
                                          float(np.exp(got[i] - m[i])[mine[i]].sum()), float(m[i]), float(lg[i].min()),
                                          int(lg[i].argmax()) + k * share, float(lg[i, lab]) if 0 <= lab < share else float("nan")))
    w.close()
    head = sf.read(ref_path)
    files = []
    for k in range(world):
        path = os.path.join(work, f"part{topk}-r{k}.bin")
        with open(path, "wb") as f:
            f.write(PART_HEAD.pack(b"GLM53PT1", 1, k, world, k * share, share, topk, n_pos, head["tokens_digest"], head["body_digest"], 0))
            f.write(b"".join(parts[k]))
        files.append(path)
    info = subprocess.run([tool, "--mode", "score-info", "--ref", ref_path], capture_output=True, text=True)
    out_json = os.path.join(work, f"out{topk}.json")
    comb = subprocess.run([tool, "--mode", "score-combine", "--ref", ref_path, "--parts", ",".join(reversed(files)), "--out", out_json],
                          capture_output=True, text=True)
    if info.returncode or comb.returncode:
        print(info.stderr[-400:], comb.stderr[-800:])
        return False
    tot = json.load(open(out_json))["sets"][-1]
    exact, top1 = kl_sum / n_pos, agree / n_pos
    ok = tot["tokens"] == n_pos and abs(tot["top1"] - top1) < 1e-6 and tot["kl"] <= exact + 1e-6 <= tot["kl_upper"] + 2e-6
    if topk == vocab:
        ok = ok and abs(tot["kl"] - exact) < 1e-5 and abs(tot["kl_upper"] - exact) < 1e-5
    print(f"topk {topk}: exact KL {exact:.6f}, tool {tot['kl']:.6f} .. {tot['kl_upper']:.6f}; top-1 {top1:.4f} / {tot['top1']:.4f}: {'ok' if ok else 'MISMATCH'}")
    return ok


def main():
    tool = sys.argv[1]
    work = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="glm53-score-")
    os.makedirs(work, exist_ok=True)
    good = all([run(tool, work, k) for k in (4096, 512, 32)])
    print("SELFTEST OK" if good else "SELFTEST FAIL")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
