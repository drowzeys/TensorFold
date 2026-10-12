#!/usr/bin/env python3
"""Where the ranks wait for each other in decode windows, from the RoCE exchange trace (TF_GLM53_ROCE_TRACE, no GPU).

Each rank's one-shot kernel stamps every exchange on its GPU's globaltimer: start, doorbell (its payload staged),
each peer's flag seen, end. The engine appends the entries to TF_GLM53_ROCE_TRACE_FILE + ".r<rank>" (cuda_roce.zig).
The ranks' timers differ by an offset and drift, so only one rank's own intervals are compared:

  stage  doorbell - start            copying the partial into the pinned slot
  wait   last flag seen - doorbell   how long this rank stood ready before the last peer's payload arrived
  copy   end - last flag seen        summing the peers' slots
  gap    start - previous end        this rank's compute between two exchanges

For one exchange (the same sequence number on every rank) the rank with the shortest wait arrived last: the others
waited for it. The same rank's gap before that exchange is the longest, and `skew` = longest gap - shortest gap is
what the other ranks lose there. A window's exchanges repeat with a fixed period (two a layer), found from the long
gaps between windows or given with --period, so the loss is reported by position in the window as well.

  roce_trace.py FILE.r0 FILE.r1 [FILE.r2 ...] [--period N] [--top 12] [--json OUT]
  roce_trace.py --selftest

After the idea of Jay Leaton's exchange trace for his DeepSeek-V4.1-Flash family of TensorFold (two ranks there).
"""
import json
import struct
import sys

import numpy as np

MAGIC = b"GLM53RT1"
ENTRY = 16  # u64 words


def read(path):
    raw = open(path, "rb").read()
    if raw[:8] != MAGIC:
        sys.exit(f"{path}: not a RoCE trace")
    rank, world, hcas, eb = struct.unpack_from("<IIII", raw, 8)
    if eb != ENTRY * 8:
        sys.exit(f"{path}: entries of {eb} bytes")
    e = np.frombuffer(raw, dtype="<u8", offset=24).reshape(-1, ENTRY)
    e = e[e[:, 1] != 0]  # an entry never completed (start 0)
    seq = (e[:, 0] & 0xFFFFFFFF).astype(np.int64)
    order = np.argsort(seq, kind="stable")
    e, seq = e[order], seq[order]
    slots = min(8, world * hcas)
    seen = e[:, 4:4 + slots].astype(np.int64)
    mine = [rank * hcas + h for h in range(hcas) if rank * hcas + h < slots]
    seen[:, mine] = 0
    last = seen.max(1)
    return {"rank": rank, "world": world, "hcas": hcas, "seq": seq, "bytes": (e[:, 0] >> 32).astype(np.int64),
            "start": e[:, 1].astype(np.int64), "bell": e[:, 2].astype(np.int64), "end": e[:, 3].astype(np.int64),
            "last": last, "seen": seen}


def analyse(traces, period=None, top=12):
    traces = sorted(traces, key=lambda t: t["rank"])
    world = len(traces)
    common = traces[0]["seq"]
    for t in traces[1:]:
        common = np.intersect1d(common, t["seq"])
    # consecutive runs only: a gap needs the previous exchange of the same rank
    cols = {}
    for t in traces:
        idx = np.searchsorted(t["seq"], common)
        cols[t["rank"]] = {k: t[k][idx] for k in ("start", "bell", "end", "last", "bytes")}
    n = len(common)
    if n < 4:
        sys.exit("fewer than 4 exchanges are in every rank's trace")
    stage = np.stack([cols[r]["bell"] - cols[r]["start"] for r in range(world)], 1) / 1e3   # us
    wait = np.stack([cols[r]["last"] - cols[r]["bell"] for r in range(world)], 1) / 1e3
    copy = np.stack([cols[r]["end"] - cols[r]["last"] for r in range(world)], 1) / 1e3
    whole = np.stack([cols[r]["end"] - cols[r]["start"] for r in range(world)], 1) / 1e3
    gap = np.full((n, world), np.nan)
    cont = np.zeros(n, bool)
    cont[1:] = common[1:] == common[:-1] + 1
    for r in range(world):
        gap[1:, r] = (cols[r]["start"][1:] - cols[r]["end"][:-1]) / 1e3
    gap[~cont] = np.nan
    late = wait.argmin(1)                                # the rank the others waited for
    lost = wait.max(1) - wait.min(1)                     # the longest any rank stood ready beyond the late one's poll
    out = {"world": world, "exchanges": int(n), "bytes_median": int(np.median(cols[0]["bytes"]))}
    out["per_rank"] = [{"rank": r, "late_share": float((late == r).mean()), "wait_us_mean": float(wait[:, r].mean()),
                        "stage_us_median": float(np.median(stage[:, r])), "copy_us_median": float(np.median(copy[:, r])),
                        "kernel_us_mean": float(whole[:, r].mean()),
                        "gap_us_median": float(np.nanmedian(gap[:, r]))} for r in range(world)]
    ok = cont & ~np.isnan(gap).any(1)
    g = gap[ok]
    # windows: a gap far above the median on every rank is the host between windows
    med = np.median(g)
    inner = (g.max(1) < 20 * med) if len(g) else np.zeros(0, bool)
    skew = g.max(1) - g.min(1)
    out["skew_us_mean_inner"] = float(skew[inner].mean()) if inner.any() else 0.0
    out["slowest_share"] = [float((g[inner].argmax(1) == r).mean()) if inner.any() else 0.0 for r in range(world)]
    out["poll_floor_us"] = float(np.median(wait.min(1)))
    out["lost_us_mean"] = float(lost.mean())
    if period is None:
        starts = np.flatnonzero(ok)[~inner] if len(g) else np.zeros(0, int)
        d = np.diff(starts)
        period = int(np.bincount(d).argmax()) if len(d) > 2 else 0
    out["period"] = int(period)
    if period > 1:
        first = np.flatnonzero(ok)[~inner]
        anchor = int(first[0]) if len(first) else 0
        pos = (np.arange(n) - anchor) % period
        out["lost_us_a_window"] = float(lost.mean() * period)
        out["kernel_us_a_window"] = [float(whole[:, r].mean() * period) for r in range(world)]
        rows = []
        for p in range(period):
            m = (pos == p) & ok
            m[np.flatnonzero(ok)[~inner]] = False
            if m.sum() < 2:
                continue
            gg = gap[m]
            rows.append({"position": p, "count": int(m.sum()), "skew_us": float((gg.max(1) - gg.min(1)).mean()),
                         "slowest_rank": int(np.bincount(gg.argmax(1), minlength=world).argmax()),
                         "gap_us": [float(x) for x in gg.mean(0)], "lost_us": float(lost[m].mean())})
        rows.sort(key=lambda x: -x["lost_us"])
        out["worst_positions"] = rows[:top]
        out["lost_us_by_parity"] = [float(lost[(pos % 2 == q)].mean()) for q in (0, 1)]
    return out


def show(o):
    print(f"{o['exchanges']} exchanges on {o['world']} ranks, median {o['bytes_median']} bytes; "
          f"poll floor {o['poll_floor_us']:.2f} us; a rank stands ready {o['lost_us_mean']:.2f} us an exchange beyond it")
    print("rank  late%  wait us  stage us  copy us  kernel us  gap us (median)  slowest%")
    for r, s in zip(o["per_rank"], o["slowest_share"]):
        print(f"{r['rank']:>4}  {r['late_share'] * 100:5.1f}  {r['wait_us_mean']:7.2f}  {r['stage_us_median']:8.2f}  "
              f"{r['copy_us_median']:7.2f}  {r['kernel_us_mean']:9.2f}  {r['gap_us_median']:15.2f}  {s * 100:7.1f}")
    print(f"compute skew inside windows: {o['skew_us_mean_inner']:.2f} us an exchange (longest gap - shortest)")
    if o.get("period", 0) > 1:
        print(f"period {o['period']} exchanges a window: {o['lost_us_a_window'] / 1e3:.3f} ms a window standing ready; "
              f"exchange kernels {', '.join(f'{x / 1e3:.3f}' for x in o['kernel_us_a_window'])} ms a rank; "
              f"by parity {o['lost_us_by_parity'][0]:.2f} / {o['lost_us_by_parity'][1]:.2f} us")
        print("worst positions in a window (position, lost us, skew us, slowest rank, each rank's gap us):")
        for w in o["worst_positions"]:
            print(f"  {w['position']:>4}  {w['lost_us']:7.2f}  {w['skew_us']:7.2f}  r{w['slowest_rank']}  "
                  + " ".join(f"{x:8.1f}" for x in w["gap_us"]))


def write(path, rank, world, hcas, seq, start, bell, end, seen):
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<IIII", rank, world, hcas, ENTRY * 8))
        for i in range(len(seq)):
            e = np.zeros(ENTRY, dtype="<u8")
            e[0] = int(seq[i]) | (24576 << 32)
            e[1], e[2], e[3] = int(start[i]), int(bell[i]), int(end[i])
            e[4:4 + world * hcas] = seen[i]
            f.write(e.tobytes())


def selftest():
    import os
    import tempfile
    rng = np.random.default_rng(7)
    world, hcas, period, windows = 4, 2, 12, 40
    slow = 2                                    # rank 2 computes 30 us longer before every odd exchange
    t = np.zeros(world)                         # true time, ns
    off = np.array([0, 5e12, 9e12, 2e12])       # each rank's timer offset
    rec = {r: [] for r in range(world)}
    seq = 0
    for _ in range(windows):
        t[:] = t.max() + 2e6                    # the host between windows
        for p in range(period):
            seq += 1
            comp = 100e3 + rng.normal(0, 500, world)
            if p % 2 == 1:
                comp[slow] += 30e3
            start = t + comp
            bell = start + 2e3
            arrive = bell.max() + 5e3           # the last payload lands one transfer after the last doorbell
            end = np.full(world, arrive + 4e3)
            for r in range(world):
                seen = np.zeros(world * hcas)
                for peer in range(world):
                    if peer != r:
                        seen[peer * hcas:(peer + 1) * hcas] = max(bell[peer] + 5e3, bell[r] + 900) + off[r]
                rec[r].append((seq, start[r] + off[r], bell[r] + off[r], end[r] + off[r], seen))
            t = end
    d = tempfile.mkdtemp()
    files = []
    for r in range(world):
        p = os.path.join(d, f"t.r{r}")
        a = rec[r]
        write(p, r, world, hcas, [x[0] for x in a], [x[1] for x in a], [x[2] for x in a], [x[3] for x in a], [x[4] for x in a])
        files.append(p)
    o = analyse([read(p) for p in files])
    show(o)
    ok = (o["period"] == period and o["per_rank"][slow]["late_share"] > 0.45 and o["slowest_share"][slow] > 0.45
          and abs(o["lost_us_by_parity"][0] - o["lost_us_by_parity"][1]) > 20
          and all(w["slowest_rank"] == slow for w in o["worst_positions"][:period // 2 - 1]))
    print("SELFTEST OK" if ok else "SELFTEST FAIL")
    return ok


def main():
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    period = int(a[a.index("--period") + 1]) if "--period" in a else None
    top = int(a[a.index("--top") + 1]) if "--top" in a else 12
    out = a[a.index("--json") + 1] if "--json" in a else None
    files, skip = [], False
    for i, x in enumerate(a):
        if skip:
            skip = False
        elif x in ("--period", "--top", "--json"):
            skip = True
        else:
            files.append(x)
    if len(files) < 2:
        sys.exit(__doc__)
    o = analyse([read(p) for p in files], period, top)
    show(o)
    if out:
        json.dump(o, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
