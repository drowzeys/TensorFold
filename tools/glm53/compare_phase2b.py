#!/usr/bin/env python3
"""Phase 2b gate: tf-glm53-generate --mode 2b against tools/glm53/reference2b.py (the Python engine's Runner).

  compare_phase2b.py REF-r0.json GEN-r0.json [GEN-r1.json ...] [--ref-ranks REF-r1.json ...] [--served-prose 32.2]

Checks, per (prompt, run):
  * the Zig tokens equal the Python tokens (greedy and keyed sampled draws, drafted and serial);
  * drafted == serial: within each engine, every run of a prompt with the same draw rule ("g", "s20", "s0") and any
    number of drafts gives the same tokens;
  * every Zig rank wrote the same tokens; the Zig seeds (seed_for of the prompt ids) equal Python's.
Then the speed table: decode tok/s, tokens a round and prefill seconds of both engines on the same prompts and
runs, the drafted runs' Zig / Python ratio, the 32K prose line beside the served Python figure (RoCE one-shot, copy
drafts on - not this comparison's NCCL-only, copies-off setting), and both engines' load time. Prints one line
``PHASE2B PASS|FAIL ...`` last; exit 0 only on PASS (speed never fails the gate).
"""

from __future__ import annotations

import argparse
import json
import math


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def first_diff(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def rule(run: str) -> str:
    return run.partition(":")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("gen", nargs="+")
    ap.add_argument("--ref-ranks", nargs="*", default=[])
    ap.add_argument("--served-prose", type=float, default=32.2, help="the served Python engine's 32K prose MTP tok/s")
    a = ap.parse_args()
    ref, gens = load(a.ref), [load(p) for p in a.gen]
    gen = gens[0]
    problems = []
    for key in ("world", "layers", "context", "capacity", "window", "k"):
        if ref.get(key) != gen.get(key):
            problems.append(f"setting {key}: reference {ref.get(key)} vs zig {gen.get(key)}")
    rr = {(r["name"], r["run"]): r for r in ref["runs"]}
    gr = {(r["name"], r["run"]): r for r in gen["runs"]}
    if sorted(rr) != sorted(gr):
        problems.append(f"runs differ: reference {sorted(rr)} vs zig {sorted(gr)}")
    tok_eq = tok_all = runs_eq = 0
    first = []
    rows = []
    for key, r in rr.items():
        g = gr.get(key)
        if g is None:
            continue
        n = min(len(r["tokens"]), len(g["tokens"]))
        tok_all += len(r["tokens"])
        tok_eq += sum(1 for i in range(n) if r["tokens"][i] == g["tokens"][i])
        d = first_diff(r["tokens"], g["tokens"])
        if d is None:
            runs_eq += 1
        else:
            first.append(f"{key[0]}/{key[1]}@{d}")
        if g.get("seed_ok") is False or str(g.get("seed")) != str(r.get("seed")):
            problems.append(f"{key[0]}: seed zig {g.get('seed')} vs python {r.get('seed')}")
        rows.append((key, r, g, d))

    # drafted == serial, per engine
    ds_ok = ds_all = 0
    for label, runs in (("python", rr), ("zig", gr)):
        groups: dict[tuple[str, str], list[dict]] = {}
        for (name, run), x in runs.items():
            groups.setdefault((name, rule(run)), []).append(x)
        for (name, rl), xs in sorted(groups.items()):
            if len(xs) < 2:
                continue
            ds_all += 1
            base = xs[0]["tokens"]
            bad = [x["run"] for x in xs[1:] if x["tokens"] != base]
            if bad:
                problems.append(f"{label} {name} {rl}: drafted != serial ({xs[0]['run']} vs {', '.join(bad)}: first "
                                f"difference at {first_diff(base, next(x['tokens'] for x in xs if x['run'] == bad[0]))})")
            else:
                ds_ok += 1
    for o in gens[1:]:
        for x in o["runs"]:
            y = gr.get((x["name"], x["run"]))
            if y is None or x["tokens"] != y["tokens"]:
                problems.append(f"zig rank {o.get('rank')} disagrees with rank {gen.get('rank')} on {x['name']} "
                                f"{x['run']}")
    for p in a.ref_ranks:
        o = load(p)
        for x in o["runs"]:
            y = rr.get((x["name"], x["run"]))
            if y is None or x["tokens"] != y["tokens"]:
                problems.append(f"python rank {o.get('rank')} disagrees with rank 0 on {x['name']} {x['run']}")

    print(f"{'prompt':12s} {'run':7s} {'P':>6s} {'n':>4s}  {'tokens':8s}  {'py tok/s':>8s} {'zig tok/s':>9s} "
          f"{'zig/py':>6s}  {'py t/rd':>7s} {'zig t/rd':>8s}  {'py pf s':>7s} {'zig pf s':>8s}")
    ratios = []
    prose = None
    for (name, run), r, g, d in rows:
        rt, zt = float(r.get("tok_s") or 0), float(g.get("tok_s") or 0)
        ratio = zt / rt if rt > 0 else 0.0
        if r.get("k", 0) and rt > 0 and zt > 0:
            ratios.append(ratio)
        if name.startswith("prose32k") and r.get("k", 0) and not r.get("greedy"):
            prose = (rt, zt, float(r.get("tokens_per_round") or 0), float(g.get("tokens_per_round") or 0))
        status = "equal" if d is None else f"@{d}"
        print(f"{name:12s} {run:7s} {r['prompt_len']:6d} {len(r['tokens']):4d}  {status:8s}  {rt:8.2f} {zt:9.2f} "
              f"{ratio:6.3f}  {float(r.get('tokens_per_round') or 0):7.3f} {float(g.get('tokens_per_round') or 0):8.3f}"
              f"  {float(r.get('prefill_s') or 0):7.1f} {float(g.get('prefill_s') or 0):8.1f}")
        if d is not None and d < min(len(r["tokens"]), len(g["tokens"])):
            print(f"    first difference at pick {d}: python {r['tokens'][d]}, zig {g['tokens'][d]}")
    geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else 0.0
    print(f"load: python rank 0 {ref.get('load_s')} s, zig rank 0 {gen.get('load_s')} s "
          f"({gen.get('read_gib')} GiB read direct, {gen.get('direct_files')} files O_DIRECT, "
          f"{gen.get('read_fallbacks')} fallback reads); zig ranks: "
          + ", ".join(f"r{o.get('rank')} {o.get('load_s')} s" for o in gens))
    print(f"prewarm: python {ref.get('prewarm_graphs')} graphs in {ref.get('prewarm_s')} s, zig "
          f"{gen.get('prewarm_graphs')} graphs in {gen.get('prewarm_s')} s")
    if prose:
        print(f"32K prose (drafted, sampled): python {prose[0]:.2f} tok/s ({prose[2]:.3f} tokens/round), zig "
              f"{prose[1]:.2f} tok/s ({prose[3]:.3f}); served python {a.served_prose} (RoCE one-shot + copy drafts)")
    if ref.get("nccl_version") != gen.get("nccl_version"):
        print(f"note: NCCL {ref.get('nccl_version')} (python) vs {gen.get('nccl_version')} (zig)")
    for p in problems:
        print("problem:", p)
    ok = not problems and tok_all > 0 and tok_eq == tok_all and runs_eq == len(rows)
    print(f"PHASE2B {'PASS' if ok else 'FAIL'} runs={runs_eq}/{len(rows)} tokens={tok_eq}/{tok_all} "
          f"drafted==serial={ds_ok}/{ds_all} zig/py_decode={geo:.3f}"
          + (f" prose32k_py={prose[0]:.2f} prose32k_zig={prose[1]:.2f}" if prose else "")
          + f" load_py={ref.get('load_s')} load_zig={max(float(o.get('load_s') or 0) for o in gens):.0f}"
          + (f" first_divergence={','.join(first[:6])}" if first else "")
          + (f" problems={len(problems)}" if problems else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
