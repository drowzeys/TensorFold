#!/usr/bin/env python3
"""Phase 3a gate: tf-glm53-generate --mode 3a against tools/glm53/reference3a.py (the Python engine served).

  compare_phase3a.py REF-r0.json GEN-r0.json [GEN-r1.json ...] [--ref-ranks REF-r1.json ...]
                     [--roce-bench BENCH-r0.json] [--speed-default REF-SPEED-r0.json] [--ratio 0.97]
                     [--tpr-tol 0.02]

PASS needs every one of:
  * tokens: every (prompt, run) of the Zig engine equal to Python's (greedy and keyed sampled, drafted and serial);
    drafted == serial within each engine; every rank of each engine wrote the same tokens; the seeds agree;
  * speed: Zig >= RATIO x Python on the 32K prose decode (the drafted sampled run, MTP k = 2) and on prefill at 32K
    and 128K (prompt tokens / prefill seconds, the best of each prompt's runs: both engines prefill a prompt once a
    run, and the first run of the first long prompt pays first-use costs in neither engine - both prewarm);
  * drafts: on every 32K prose drafted run (MTP k > 0), Zig's tokens a round within TPR_TOL (relative, default 2 %)
    of Python's - the MTP layer's prompt rows may take another implementation (--mtp-dense: cuBLAS over dense bf16
    experts; Python's cuda-exl3 GEMM is not reproducible either), which can move drafts but never a token;
  * both engines ran the served exchanges: RoCE one-shot on (the reference's and Zig's), and the reference's prompt
    path (sequence parallel, 8192 / 4096-row chunks).
Also printed: the speed table of every run, both engines' load time (not gated), the RoCE micro-benchmark (us a
reduction for the decode windows' shapes; target <= 65 us) and, with --speed-default, the Python engine's prefill at
the image's default TF_EXL3_PROMPT_DET (red.add, not reproducible) beside the gated slots16 figures.
Prints one line ``PHASE3A PASS|FAIL ...`` last; exit 0 only on PASS.
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


def prefill_rate(runs: dict, prefix: str) -> tuple[float, int]:
    """Best prompt tokens / prefill seconds over the runs of the prompt named ``prefix``*: (tok/s, prompt tokens)."""
    best, n = 0.0, 0
    for (name, _), r in runs.items():
        if name.startswith(prefix) and float(r.get("prefill_s") or 0) > 0:
            rate = r["prompt_len"] / float(r["prefill_s"])
            if rate > best:
                best, n = rate, r["prompt_len"]
    return best, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("gen", nargs="+")
    ap.add_argument("--ref-ranks", nargs="*", default=[])
    ap.add_argument("--roce-bench", default="")
    ap.add_argument("--speed-default", default="")
    ap.add_argument("--ratio", type=float, default=0.97)
    ap.add_argument("--tpr-tol", type=float, default=0.02)
    a = ap.parse_args()
    ref, gens = load(a.ref), [load(p) for p in a.gen]
    gen = gens[0]
    problems = []
    for key in ("world", "layers", "context", "capacity", "k"):
        if ref.get(key) != gen.get(key):
            problems.append(f"setting {key}: reference {ref.get(key)} vs zig {gen.get(key)}")
    if (ref.get("prompt_rows"), ref.get("prompt_rows_short")) != (gen.get("prompt_rows"), gen.get("prompt_rows_short")):
        problems.append(f"prompt rows: reference {ref.get('prompt_rows')}/{ref.get('prompt_rows_short')} vs zig "
                        f"{gen.get('prompt_rows')}/{gen.get('prompt_rows_short')}")
    if not ref.get("roce"):
        problems.append("the reference ran without the RoCE one-shot (not the served configuration)")
    if not gen.get("roce"):
        problems.append("the Zig engine ran without the RoCE one-shot")
    if not ref.get("prompt_sp"):
        problems.append("the reference ran without sequence-parallel prompt chunks")
    if ref.get("experts_impl") == "shared" and gen.get("experts") != "shared":
        problems.append("the reference's experts are SharedExperts (prompt experts), the Zig run's are not")
    if gen.get("bmm_probe") != "equal":
        problems.append(f"the cuBLAS bmm probe was {gen.get('bmm_probe')!r}")
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
                problems.append(f"{label} {name} {rl}: drafted != serial ({xs[0]['run']} vs {', '.join(bad)})")
            else:
                ds_ok += 1
    for o in gens[1:]:
        for x in o["runs"]:
            y = gr.get((x["name"], x["run"]))
            if y is None or x["tokens"] != y["tokens"]:
                problems.append(f"zig rank {o.get('rank')} disagrees with rank {gen.get('rank')} on {x['name']} {x['run']}")
    for p in a.ref_ranks:
        o = load(p)
        for x in o["runs"]:
            y = rr.get((x["name"], x["run"]))
            if y is None or x["tokens"] != y["tokens"]:
                problems.append(f"python rank {o.get('rank')} disagrees with rank 0 on {x['name']} {x['run']}")

    print(f"{'prompt':12s} {'run':7s} {'P':>6s} {'n':>4s}  {'tokens':8s}  {'py tok/s':>8s} {'zig tok/s':>9s} "
          f"{'zig/py':>6s}  {'py t/rd':>7s} {'zig t/rd':>8s}  {'py pf/s':>8s} {'zig pf/s':>8s}")
    prose = None
    tpr = []   # (run, python t/rd, zig t/rd) of the 32K prose drafted runs
    for (name, run), r, g, d in rows:
        rt, zt = float(r.get("tok_s") or 0), float(g.get("tok_s") or 0)
        ratio = zt / rt if rt > 0 else 0.0
        if name.startswith("prose32k") and r.get("k", 0) and not r.get("greedy"):
            prose = (rt, zt, float(r.get("tokens_per_round") or 0), float(g.get("tokens_per_round") or 0))
        if name.startswith("prose32k") and r.get("k", 0):
            tpr.append((run, float(r.get("tokens_per_round") or 0), float(g.get("tokens_per_round") or 0)))
        rp = r["prompt_len"] / float(r["prefill_s"]) if float(r.get("prefill_s") or 0) > 0 else 0.0
        zp = g["prompt_len"] / float(g["prefill_s"]) if float(g.get("prefill_s") or 0) > 0 else 0.0
        status = "equal" if d is None else f"@{d}"
        print(f"{name:12s} {run:7s} {r['prompt_len']:6d} {len(r['tokens']):4d}  {status:8s}  {rt:8.2f} {zt:9.2f} "
              f"{ratio:6.3f}  {float(r.get('tokens_per_round') or 0):7.3f} {float(g.get('tokens_per_round') or 0):8.3f}"
              f"  {rp:8.0f} {zp:8.0f}")
        if d is not None and d < min(len(r["tokens"]), len(g["tokens"])):
            print(f"    first difference at pick {d}: python {r['tokens'][d]}, zig {g['tokens'][d]}")
    py32, n32 = prefill_rate(rr, "prose32k")
    zg32, _ = prefill_rate(gr, "prose32k")
    py128, n128 = prefill_rate(rr, "long128k")
    zg128, _ = prefill_rate(gr, "long128k")
    gates = []
    if prose:
        gates.append(("32K prose decode", prose[0], prose[1]))
        print(f"32K prose decode (MTP k=2, sampled, thinking on): python {prose[0]:.2f} tok/s ({prose[2]:.3f} "
              f"tokens/round), zig {prose[1]:.2f} tok/s ({prose[3]:.3f}) -> {prose[1] / max(prose[0], 1e-9):.3f}")
    else:
        problems.append("no 32K prose drafted sampled run")
    for label, p, z, n in (("prefill 32K", py32, zg32, n32), ("prefill 128K", py128, zg128, n128)):
        if p > 0 and z > 0:
            gates.append((label, p, z))
            print(f"{label} ({n} tokens): python {p:.0f} tok/s, zig {z:.0f} tok/s -> {z / p:.3f}")
        else:
            problems.append(f"no {label} figure (python {p:.0f}, zig {z:.0f})")
    if a.speed_default:
        sd = load(a.speed_default)
        sr = {(r["name"], r["run"]): r for r in sd["runs"]}
        d32, _ = prefill_rate(sr, "prose32k")
        d128, _ = prefill_rate(sr, "long128k")
        print(f"python at the image default TF_EXL3_PROMPT_DET (red.add, not reproducible; not gated): prefill 32K "
              f"{d32:.0f} tok/s, 128K {d128:.0f} tok/s")
    print(f"load: python rank 0 {ref.get('load_s')} s (engine init {ref.get('init_s')} s), zig rank 0 {gen.get('load_s')} s; "
          f"zig ranks: " + ", ".join(f"r{o.get('rank')} {o.get('load_s')} s" for o in gens))
    print(f"prewarm: python {ref.get('prewarm_graphs')} graphs, zig {gen.get('prewarm_graphs')} graphs in "
          f"{gen.get('prewarm_s')} s; zig tiles from the reference boot ({gen.get('tiles_changed')} differ from the "
          f"plan), unpack cache {gen.get('unpack_hits')} hits / {gen.get('unpack_misses')} misses, RoCE ops "
          f"{gen.get('roce_ops')}, experts {ref.get('experts_impl')} / {gen.get('experts')}")
    bench_worst = None
    if a.roce_bench:
        b = load(a.roce_bench)
        bench_worst = b.get("decode_worst_us")
        print("RoCE one-shot (us a reduction, CUDA graph replay): " + ", ".join(
            f"{x['rows']} rows {x['roce_us']:.1f} (NCCL all-gather {x['nccl_gather_us']:.1f})" for x in b["rows"])
            + f"; bit check {b.get('bit_check')}; decode windows worst {bench_worst} us (target <= 65)")
    if ref.get("nccl_version") != gen.get("nccl_version"):
        print(f"note: NCCL {ref.get('nccl_version')} (python) vs {gen.get('nccl_version')} (zig)")
    slow = [f"{lab} {z / p:.3f}" for lab, p, z in gates if z < a.ratio * p]
    tpr_bad = [f"{run} {py:.3f}/{zg:.3f}" for run, py, zg in tpr if py <= 0 or abs(zg - py) > a.tpr_tol * py]
    if not tpr:
        problems.append("no 32K prose drafted run for the tokens/round check")
    print("32K prose tokens/round (drafted runs, python / zig, tolerance "
          f"{a.tpr_tol:.1%}): " + ", ".join(f"{run} {py:.3f} / {zg:.3f}" for run, py, zg in tpr)
          + f"; zig MTP prompt rows: {'dense cuBLAS' if gen.get('mtp_dense') else 'decode kernel'}"
          + (f" ({gen.get('mtp_dense_gb')} GB a rank, {gen.get('mtp_chunks')} chunks, {gen.get('mtp_gemms')} GEMMs)"
             if gen.get('mtp_dense') else ""))
    for p in problems:
        print("problem:", p)
    tokens_ok = not problems and tok_all > 0 and tok_eq == tok_all and runs_eq == len(rows)
    ok = tokens_ok and not slow and len(gates) == 3 and not tpr_bad
    geo = math.exp(sum(math.log(z / p) for _, p, z in gates) / len(gates)) if gates else 0.0
    print(f"PHASE3A {'PASS' if ok else 'FAIL'} runs={runs_eq}/{len(rows)} tokens={tok_eq}/{tok_all} "
          f"drafted==serial={ds_ok}/{ds_all} "
          + (f"prose32k_py={prose[0]:.2f} prose32k_zig={prose[1]:.2f} " if prose else "")
          + f"prefill32k_py={py32:.0f} prefill32k_zig={zg32:.0f} prefill128k_py={py128:.0f} prefill128k_zig={zg128:.0f} "
          + f"speed_geo={geo:.3f} load_py={ref.get('load_s')} "
          + f"load_zig={max(float(o.get('load_s') or 0) for o in gens):.0f}"
          + (f" roce_us={bench_worst}" if bench_worst is not None else "")
          + (f" below_{a.ratio}=[{'; '.join(slow)}]" if slow else "")
          + (f" tpr_off=[{'; '.join(tpr_bad)}]" if tpr_bad else "")
          + (f" first_divergence={','.join(first[:6])}" if first else "")
          + (f" problems={len(problems)}" if problems else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
