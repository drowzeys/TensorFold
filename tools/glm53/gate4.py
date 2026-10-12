#!/usr/bin/env python3
"""Phase 4 gate: the Zig engine's speculative drafters (tf-glm53-generate --mode 4) against the Python engine's
(reference4.py), one session per drafter configuration, every session run in the same cluster script.

  gate4.py gate --work DIR --sessions ds,df[,mc][,priv] [--ratio 0.97] [--tpr-tol 0.02]
                [--py-dspark-32k prose=33.8,code=39.0] [--champion-priv ...]

A session S reads <work>/ref-S/ref-r*.json (Python, every rank) and <work>/gen-S/gen-r*.json (Zig, every rank). Runs
are matched by (prompt name, run label); a label names the pick ("g" greedy / "s<top_k>" keyed sample), the drafts
("K" MTP steps, 0 serial; "dspark"; "dflash") and "+c" copy drafts ahead of them.

GATE lines (all must PASS):
  * S-tokens: every run of the Zig engine equal to Python's token for token (drafts never move a token: the target
    verifies every row and every emitted token is its own keyed pick); every rank of each engine wrote the same tokens;
  * S-drafted==serial: within each engine, every drafted run (drafter, MTP or copies) equal to the same prompt's serial
    run with the same pick rule;
  * S-tpr: on the 32K prompts' drafted runs, Zig's tokens a round within --tpr-tol (relative) of Python's (the
    drafters' arithmetic is the Python kernels' except a few small dots, so acceptance should match closely);
  * S-speed: on the 32K prompts' sampled drafted runs with copies ("s20:<drafter>+c", the served default), Zig's decode
    tok/s >= --ratio x Python's, prose and code separately (the mean over the class's prompts);
  * priv (private, PRIV_DFLASH=path in the run script): as above, plus REPORT lines against the champion's published numbers
    (prose 39.1 / 34.2 / 31.2, code 44.8 / 42.9 / 39.4 at short / 32K / ~120K; served, tfbench repeats - indicative).
REPORT lines: every run side by side (tok/s, tokens a round, acceptance, copy rounds), DSpark's Python 32K figures
against the 2026-10-07 reference (33.8 prose / 39.0 code; a gap there means the cluster, not the port).
Prints ``PHASE4 PASS|FAIL ...`` last; exit 0 only on PASS.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def runs_of(doc: dict) -> dict:
    return {(r["name"], r["run"]): r for r in doc.get("runs", [])}


def label_parts(run: str) -> tuple[str, str, bool]:
    """(pick, drafts, copies): "s20:dspark+c" -> ("s20", "dspark", True)."""
    pick, _, rest = run.partition(":")
    copies = rest.endswith("+c")
    return pick, rest[:-2] if copies else rest, copies


def drafted(run: str) -> bool:
    _, d, c = label_parts(run)
    return c or d != "0"


def klass(name: str) -> tuple[str, str]:
    """(kind, length) of a prompt name: prose_short_x -> (prose, short), code32k_x -> (code, 32k)."""
    head = name.split("_")[0]
    if name.startswith(head + "_short"):
        return head, "short"
    for length in ("32k", "120k"):
        if head.endswith(length):
            return head[: -len(length)], length
    return head, "?"


def first_diff(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


CHAMPION_PRIV = {("prose", "short"): 39.1, ("prose", "32k"): 34.2, ("prose", "120k"): 31.2,
                ("code", "short"): 44.8, ("code", "32k"): 42.9, ("code", "120k"): 39.4}


def session(work: str, s: str, a, gates: list, reports: list) -> None:
    refs = sorted(glob.glob(os.path.join(work, f"ref-{s}", "ref-r*.json")))
    gens = sorted(glob.glob(os.path.join(work, f"gen-{s}", "gen-r*.json")))
    if not refs or not gens:
        gates.append((f"{s}-files", False, f"ref-{s}: {len(refs)} rank files, gen-{s}: {len(gens)}"))
        return
    ref_docs, gen_docs = [load(p) for p in refs], [load(p) for p in gens]
    ref, gen = runs_of(ref_docs[0]), runs_of(gen_docs[0])
    reports.append(f"{s}: python {ref_docs[0].get('dspark') or ref_docs[0].get('dflash') or 'MTP'} k {ref_docs[0].get('k')}"
                   f" context {ref_docs[0].get('context')} widths {ref_docs[0].get('widths')}; zig dspark "
                   f"{gen_docs[0].get('dspark') or '-'} dflash {gen_docs[0].get('dflash') or '-'} k {gen_docs[0].get('k')}"
                   f" load {gen_docs[0].get('load_s')} s, prewarm graphs {gen_docs[0].get('prewarm_graphs')}")
    # every rank of each engine alike
    ranks_ok, why = True, ""
    for docs, who in ((ref_docs, "python"), (gen_docs, "zig")):
        base = runs_of(docs[0])
        for d in docs[1:]:
            other = runs_of(d)
            for key, r in base.items():
                if key in other and other[key]["tokens"] != r["tokens"]:
                    ranks_ok, why = False, f"{who} rank {d.get('rank')} differs on {key}"
    # Zig == Python
    tok_ok, details = ranks_ok, [why] if why else []
    missing = [k for k in ref if k not in gen]
    if missing:
        tok_ok = False
        details.append(f"zig lacks {missing[:3]}")
    for key, r in ref.items():
        g = gen.get(key)
        if g is None:
            continue
        d = first_diff(r["tokens"], g["tokens"])
        if d is not None:
            tok_ok = False
            details.append(f"{key[0]} {key[1]} first differs at {d}")
    gates.append((f"{s}-tokens", tok_ok, f"{len(ref)} runs" + ("; " + "; ".join(details[:4]) if details else "")))
    # drafted == serial within each engine
    ds_ok, dd = True, []
    for runs, who in ((ref, "python"), (gen, "zig")):
        for (name, run), r in runs.items():
            if not drafted(run):
                continue
            pick = label_parts(run)[0]
            serial = runs.get((name, f"{pick}:0"))
            if serial is None:
                continue
            d = first_diff(r["tokens"], serial["tokens"])
            if d is not None:
                ds_ok = False
                dd.append(f"{who} {name} {run} vs {pick}:0 at {d}")
    gates.append((f"{s}-drafted==serial", ds_ok, "; ".join(dd[:4]) or "every drafted run equals its serial run"))
    # tokens a round and speed at 32K
    tpr_ok, tpr_bad = True, []
    speed: dict[str, list[tuple[float, float]]] = {}
    for key, r in sorted(ref.items()):
        g = gen.get(key)
        if g is None:
            continue
        kind, length = klass(key[0])
        line = (f"{s} {key[0]:28s} {key[1]:16s} python {r['tok_s']:6.2f} tok/s {r['tokens_per_round']:6.3f} t/r "
                f"accept {float(r.get('accept') or 0):.3f} copies {r.get('copy_rounds', 0):3d} | zig {g['tok_s']:6.2f} "
                f"tok/s {g['tokens_per_round']:6.3f} t/r accept {float(g.get('accept') or 0):.3f} copies "
                f"{g.get('copy_rounds', 0):3d} | ratio {g['tok_s'] / max(r['tok_s'], 1e-9):.3f}")
        reports.append(line)
        if not drafted(key[1]) or length != "32k":
            continue
        tp, tz = float(r["tokens_per_round"]), float(g["tokens_per_round"])
        if abs(tz - tp) > a.tpr_tol * tp:
            tpr_ok = False
            tpr_bad.append(f"{key[0]} {key[1]} {tz:.3f} vs {tp:.3f}")
        pick, drafts, copies = label_parts(key[1])
        if pick.startswith("s") and copies:
            speed.setdefault(kind, []).append((float(r["tok_s"]), float(g["tok_s"])))
    gates.append((f"{s}-tpr", tpr_ok, "; ".join(tpr_bad[:4]) or f"within {a.tpr_tol:.0%} on every 32K drafted run"))
    if not speed:
        gates.append((f"{s}-speed", False, "no sampled drafted run with copies on a 32K prompt"))
    for kind, pairs in sorted(speed.items()):
        py, zg = mean([p for p, _ in pairs]), mean([z for _, z in pairs])
        gates.append((f"{s}-speed-{kind}32k", zg >= a.ratio * py, f"zig {zg:.2f} vs python {py:.2f} tok/s (ratio "
                                                                 f"{zg / max(py, 1e-9):.3f}, need >= {a.ratio})"))
        if s == "ds":
            want = a.py_dspark.get(kind)
            if want:
                reports.append(f"ds python {kind} 32K {py:.2f} tok/s vs the 2026-10-07 reference {want} "
                               f"({'ok' if py >= 0.95 * want else 'LOW: the cluster, not the port'})")
    if s == "priv":
        by: dict[tuple[str, str], list[float]] = {}
        for key, g in gen.items():
            if drafted(key[1]) and label_parts(key[1])[2]:
                by.setdefault(klass(key[0]), []).append(float(g["tok_s"]))
        for k, want in CHAMPION_PRIV.items():
            got = by.get(k)
            if got:
                reports.append(f"priv zig {k[0]} {k[1]}: {mean(got):.2f} tok/s vs the champion {want} "
                               f"({mean(got) / want:.3f})")


def parse_kv(text: str) -> dict:
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--work", required=True)
    g.add_argument("--sessions", required=True)
    g.add_argument("--ratio", type=float, default=0.97)
    g.add_argument("--tpr-tol", type=float, default=0.02)
    g.add_argument("--py-dspark-32k", default="prose=33.8,code=39.0")
    a = ap.parse_args()
    a.py_dspark = parse_kv(a.py_dspark_32k)
    gates: list[tuple[str, bool, str]] = []
    reports: list[str] = []
    for s in [x for x in a.sessions.split(",") if x]:
        session(a.work, s, a, gates, reports)
    for line in reports:
        print(f"REPORT {line}")
    for name, ok, detail in gates:
        print(f"GATE {name} {'PASS' if ok else 'FAIL'} {detail}")
    ok = bool(gates) and all(x[1] for x in gates)
    print(f"PHASE4 {'PASS' if ok else 'FAIL'} " + " ".join(f"{n}={'PASS' if o else 'FAIL'}" for n, o, _ in gates))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
