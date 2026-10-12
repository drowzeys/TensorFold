#!/usr/bin/env python3
"""Debugging: the served engine's likeliest tokens beside the reference's, for the rows a score run dumped.

  score_dump.py --ref reference.bin --parts p0,p1,p2,p3 --dumps d0,d1,d2,d3 [--model DIR] [--row N] [--worst 20]

Merges the ranks' dump files (tf-glm53-generate --mode score --score-dump FILE --score-dump-rows ..) and partials,
and prints the --worst positions by KL (lower bound): the real next token, the reference's top 5 and the served top 5
with probabilities. --model decodes the ids with its tokenizer.
"""
import argparse
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_format as sf

PART_HEAD = struct.Struct("<8sIIIIIIQQQI12x")
PART = np.dtype([("s", "<f8"), ("a", "<f8"), ("t", "<f8"), ("m", "<f4"), ("lo", "<f4"), ("arg", "<u4"), ("label", "<f4")])
DUMP = np.dtype([("row", "<u4"), ("pos", "<u4"), ("ids", "<u4", 16), ("lg", "<f4", 16)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--model")
    ap.add_argument("--row", type=int)
    ap.add_argument("--worst", type=int, default=20)
    a = ap.parse_args()
    ref = sf.read(a.ref)
    topk = ref["topk"]
    recs = np.memmap(a.ref, dtype=np.uint8, mode="r", offset=ref["pos_offset"]).reshape(ref["n_pos"], sf.pos_bytes(topk))
    parts = [np.fromfile(p, dtype=PART, offset=PART_HEAD.size) for p in a.parts.split(",")]
    dumps = [np.fromfile(p, dtype=DUMP) for p in a.dumps.split(",")]
    first = np.concatenate([[0], np.cumsum([len(r) - 1 for r in ref["rows"]])])
    tok = None
    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)

    def name(i):
        return f"{i}" if tok is None else f"{i} {tok.decode([int(i)])!r}"

    out = []
    for i in range(len(dumps[0])):
        row, pos = int(dumps[0]["row"][i]), int(dumps[0]["pos"][i])
        if a.row is not None and row != a.row:
            continue
        at = int(first[row]) + pos
        m = max(float(p["m"][at]) for p in parts)
        z = sum(float(p["s"][at]) * math.exp(float(p["m"][at]) - m) for p in parts)
        log_z = m + math.log(z)
        ids = np.concatenate([d["ids"][i] for d in dumps])
        lg = np.concatenate([d["lg"][i] for d in dumps])
        order = np.lexsort((ids, -lg))[:5]
        rec = recs[at]
        r_ids = rec[16:16 + 4 * topk].copy().view("<u4")
        r_lp = rec[16 + 4 * topk:].copy().view("<f4").astype(np.float64)
        tail = float(rec[:4].copy().view("<f4")[0])
        p_top = np.exp(r_lp)
        a_sum = sum(float(p["a"][at]) for p in parts)
        kept = float((p_top * r_lp).sum()) - (a_sum - float(p_top.sum()) * log_z)
        q_top = sum(float(p["t"][at]) * math.exp(float(p["m"][at]) - m) for p in parts) / z
        kl = kept + (tail * math.log(tail / max(1 - q_top, 1e-300)) if tail > 0 else 0.0)
        out.append((kl, row, pos, r_ids[:5], p_top[:5], ids[order], np.exp(lg[order].astype(np.float64) - log_z)))
    out.sort(key=lambda x: -x[0])
    for kl, row, pos, r_ids, r_p, s_ids, s_p in out[:a.worst]:
        label = int(ref["rows"][row][pos + 1])
        print(f"row {row} position {pos}: KL {kl:.4f}; next token {name(label)}")
        print("  reference: " + "; ".join(f"{name(i)} {p:.3f}" for i, p in zip(r_ids, r_p)))
        print("  served:    " + "; ".join(f"{name(i)} {p:.3f}" for i, p in zip(s_ids, s_p)))


if __name__ == "__main__":
    main()
