#!/usr/bin/env python3
"""Cases for ``tf-glm53-generate --sampler-check``: the Python engine's keyed draws (engine/exact_sampling.py) on
generated candidate rows, so the Zig sampler (zig/src/families/glm_moe_dsa/sampling.zig) is checked bit for bit on the
host before any GPU run: the pick, the nucleus size (choose_rows' ``keep``) and seed_for.

Rows look like the runner's: every rank's top candidates of its vocabulary share (4 x (top_k + MARGIN), or 4 x 264
for top_k 0, where Glm53Engine._sample's torch.topk(264) set is the first 264 of the merged order), float32 logits
with a peaked head and ties, temperatures 1.0 / 0.7 / 0.3, top_p 0.95 / 0.8 / 1.0, min_p 0 / 0.05.

  sampler_cases.py --out cases.json [--n 1500] [--seed 7]   (CPU; PYTHONPATH at the champion source's src)
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows, seed_for


def keep_of(values: np.ndarray, ids: np.ndarray, s: Sampling) -> int:
    """choose_rows' nucleus size for one row (its own float64 steps)."""
    order = np.lexsort((ids, -values))
    width = len(ids)
    k = max(1, min(int(s.top_k) if s.top_k else width, width))
    scaled = values[order][:k].astype(np.float64)[None, :] / max(float(s.temperature), 1e-6)
    if not 0.0 < s.top_p < 1.0:
        return k
    probs = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return int(min((np.cumsum(probs, axis=-1) < s.top_p).sum(axis=-1)[0] + 1, k))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pr = random.Random(a.seed)
    cases = []
    for i in range(a.n):
        top_k = pr.choice([20, 20, 20, 0, 0, 40, 1])
        world = 4
        per = (top_k if top_k else 256) + MARGIN
        n = world * per
        spread = pr.choice([0.5, 2.0, 6.0, 15.0])
        vals = rng.normal(0.0, spread, n).astype(np.float32)
        if i % 3 == 0:                                   # a peaked head
            vals[: pr.randint(1, 6)] += np.float32(pr.uniform(2.0, 12.0))
        if i % 5 == 0:                                   # ties, by value and across ranks
            j = rng.integers(0, n, size=8)
            vals[j] = vals[j[0]]
        if i % 7 == 0:                                   # bf16-like logits: few distinct values
            vals = (vals.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
        ids = rng.choice(154880, size=n, replace=False).astype(np.int64)
        s = Sampling(int(rng.integers(0, 2 ** 63 - 1)), pr.choice([1.0, 1.0, 0.7, 0.3]), top_k,
                     pr.choice([0.95, 0.95, 0.8, 1.0]), pr.choice([0.0, 0.0, 0.0, 0.05]))
        position = int(rng.integers(1, 1 << 20))
        if top_k:                                        # Runner._sample_local: the gathered candidates as they are
            v, d = vals, ids
        else:                                            # Glm53Engine._sample: torch.topk(264)'s set, merged order
            order = np.lexsort((ids, -vals))[: 256 + MARGIN]
            v, d = vals[order], ids[order]
        pick = int(choose_rows(v[None, :], d[None, :], [position], s)[0])
        cases.append({"seed": str(s.seed), "temperature": s.temperature, "top_k": top_k, "top_p": s.top_p,
                      "min_p": s.min_p, "position": position, "values": [float(x) for x in vals],
                      "ids": [int(x) for x in ids], "pick": pick, "keep": keep_of(v, d, s)})
    seeds = []
    for i in range(200):
        toks = [int(x) for x in rng.integers(0, 154880, size=pr.randint(1, 400))]
        salt = 0 if i % 4 else pr.randint(-5, 99)
        seeds.append({"ids": toks, "salt": salt, "seed": str(seed_for(toks, salt))})
    with open(a.out, "w") as f:
        json.dump({"generator": "tools/glm53/sampler_cases.py", "cases": cases, "seeds": seeds}, f)
    print(f"[sampler] {len(cases)} draws, {len(seeds)} seeds -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
