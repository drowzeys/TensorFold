#!/usr/bin/env python3
"""Routed-expert pruning's rule in numpy fp32, the cases prune.zig's host test replays, and what a rule keeps (no GPU).

The rule is the one Jay Leaton's DeepSeek-V4.1-Flash family of TensorFold uses (prune.py's `reference`, MIT): a row's
routed weights w_0..w_{K-1} give fractions f_i = w_i / sum w (the sum in slot order). By f descending (ties: the lower
slot first) a slot is kept while the fraction mass before it is below TOPP, and the first MIN_K slots always. A row
that keeps every slot is untouched. Otherwise a dropped slot gets weight 0 and the pick DROP (no expert), and a kept
slot keeps its weight (renorm "orig") or gets scale * (w_i / sum of the kept w, in slot order) (renorm "kept").

  prune_ref.py cases OUT.bin [--route ROUTE.bin]   the fixture: synthetic rows (ties, flat and peaked rows) and, with
                                                   --route, rows of a --score-route file (real router weights)
  prune_ref.py stats ROUTE.bin [--min-k 3]         mean experts kept at 0.80 / 0.85 / 0.90 over a --score-route file
  prune_ref.py check CASES.bin                     re-derive a fixture's expected rows (exit 1 on a difference)

Fixture: "GLM53PR1", u32 cases, u32 K, then a case: f32 topp, u32 min_k, u32 kept (0 orig, 1 kept), f32 scale,
i32 drop, f32 w[K], i32 pick[K], f32 w_out[K], i32 pick_out[K]. Little-endian.
"""
import struct
import sys

import numpy as np

F32 = np.float32
MAGIC = b"GLM53PR1"


def prune_row(w, p, topp, min_k, kept, scale, drop):
    """One row: (w fp32 [K], p int32 [K]) -> new (w, p). Every sum is sequential in fp32."""
    w = np.asarray(w, dtype=F32).copy()
    p = np.asarray(p, dtype=np.int32).copy()
    K = len(w)
    total = F32(0)
    for i in range(K):
        total = F32(total + w[i])
    f = np.array([F32(w[i] / total) for i in range(K)], dtype=F32)
    rank = [sum(1 for j in range(K) if f[j] > f[i] or (f[j] == f[i] and j < i)) for i in range(K)]
    excl, acc = [F32(0)] * K, F32(0)
    for r in range(K):
        fr = F32(0)
        for i in range(K):
            if rank[i] == r:
                excl[i], fr = acc, f[i]
        acc = F32(acc + fr)
    keep = [bool(excl[i] < F32(topp)) or rank[i] < min_k for i in range(K)]
    if all(keep):
        return w, p
    ks = F32(0)
    for i in range(K):
        ks = F32(ks + (w[i] if keep[i] else F32(0)))
    for i in range(K):
        if keep[i]:
            if kept:
                w[i] = F32(F32(scale) * F32(w[i] / ks))
        else:
            w[i] = F32(0)
            p[i] = drop
    return w, p


def read_route(path, limit_bytes=None):
    """A --score-route file's weights [positions x layers flattened, K] and picks, K = 8."""
    K = 8
    raw = np.fromfile(path, dtype=np.uint8, count=limit_bytes if limit_bytes else -1)
    at, ws, ps = 0, [], []
    while at + 8 <= len(raw):
        _layer, R = struct.unpack_from("<II", raw, at)
        at += 8
        n = R * K * 4
        if at + 2 * n > len(raw):
            break
        ws.append(raw[at:at + n].view("<f4").reshape(R, K))
        ps.append(raw[at + n:at + 2 * n].view("<i4").reshape(R, K))
        at += 2 * n
    return np.concatenate(ws), np.concatenate(ps)


def router_rows(rng, n, K, E, scale, kind):
    """Rows as the router writes them: scale * s_i / sum s, picks distinct."""
    out = []
    for _ in range(n):
        if kind == "flat":
            s = rng.uniform(0.4, 0.6, K)
        elif kind == "peak":
            s = rng.uniform(0.0, 1.0, K) ** 6 + 1e-4
        elif kind == "tie":
            s = np.repeat(rng.uniform(0.1, 1.0, K // 2), 2)
            s = s[rng.permutation(K)]
        elif kind == "equal":
            s = np.full(K, rng.uniform(0.1, 1.0))
        else:
            s = rng.uniform(0.0, 1.0, K)
        s = s.astype(F32)
        tot = F32(0)
        for v in s:
            tot = F32(tot + v)
        w = np.array([F32(F32(scale) * F32(v / tot)) for v in s], dtype=F32)
        out.append((w, rng.choice(E, K, replace=False).astype(np.int32)))
    return out


def write_cases(path, route):
    rng = np.random.default_rng(53)
    K, E, scale = 8, 256, 2.5
    rules = [(0.85, 3, 0), (0.85, 3, 1), (0.80, 3, 1), (0.90, 3, 0), (0.90, 2, 1), (0.5, 1, 1), (0.85, 7, 0),
             (0.999, 3, 1), (0.85, 8, 1)]
    rows = []
    for kind in ("uniform", "flat", "peak", "tie", "equal"):
        rows += router_rows(rng, 24, K, E, scale, kind)
    if route:
        w, p = read_route(route, 64 << 20)
        for i in rng.choice(len(w), 160, replace=False):
            rows.append((w[i].copy(), p[i].copy()))
    cases = []
    for n, (w, p) in enumerate(rows):
        topp, min_k, kept = rules[n % len(rules)]
        nw, np_ = prune_row(w, p, topp, min_k, kept, scale, E)
        cases.append((topp, min_k, kept, scale, E, w, p, nw, np_))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<II", len(cases), K))
        for topp, min_k, kept, scale, drop, w, p, nw, np_ in cases:
            f.write(struct.pack("<fIIfi", topp, min_k, kept, scale, drop))
            f.write(w.astype("<f4").tobytes() + p.astype("<i4").tobytes() + nw.astype("<f4").tobytes() + np_.astype("<i4").tobytes())
    changed = sum(1 for c in cases if not np.array_equal(c[6], c[8]))
    print(f"wrote {path}: {len(cases)} cases, {changed} with a dropped slot")


def check(path):
    raw = open(path, "rb").read()
    assert raw[:8] == MAGIC
    n, K = struct.unpack_from("<II", raw, 8)
    at, bad = 16, 0
    for _ in range(n):
        topp, min_k, kept, scale, drop = struct.unpack_from("<fIIfi", raw, at)
        at += 20
        a = np.frombuffer(raw, dtype="<f4", count=K, offset=at)
        b = np.frombuffer(raw, dtype="<i4", count=K, offset=at + 4 * K)
        c = np.frombuffer(raw, dtype="<f4", count=K, offset=at + 8 * K)
        d = np.frombuffer(raw, dtype="<i4", count=K, offset=at + 12 * K)
        at += 16 * K
        nw, np_ = prune_row(a, b, topp, min_k, kept, scale, drop)
        bad += not (nw.tobytes() == c.tobytes() and np.array_equal(np_, d))
    print(f"{n} cases, {bad} differ")
    return bad == 0


def stats(path, min_k):
    w, _ = read_route(path)
    S = -np.sort(-w.astype(np.float64), axis=-1)
    S = S / S.sum(-1, keepdims=True)
    before = np.cumsum(S, -1) - S
    for th in (0.80, 0.85, 0.90):
        kept = np.maximum((before < th).sum(-1), min_k)
        print(f"topp {th:.2f} min_k {min_k}: {kept.mean():.3f} of {w.shape[1]} experts kept a row, "
              f"{(kept < w.shape[1]).mean() * 100:.1f}% of rows pruned, {(1 - (S * (np.arange(w.shape[1]) < kept[:, None])).sum(-1)).mean() * 100:.2f}% of the routed weight dropped")


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "cases":
        write_cases(a[1], a[a.index("--route") + 1] if "--route" in a else None)
    elif len(a) >= 2 and a[0] == "check":
        sys.exit(0 if check(a[1]) else 1)
    elif len(a) >= 2 and a[0] == "stats":
        stats(a[1], int(a[a.index("--min-k") + 1]) if "--min-k" in a else 3)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
