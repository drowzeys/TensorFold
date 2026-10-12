#!/usr/bin/env python3
"""Phase 2a gate: tf-glm53-generate's picks against tools/glm53/reference.py's (the Python engine, same schedule).

  compare_phase2a.py REF.json GEN.json [GEN-r1.json ...] [--ref-rank REF-r1.json ...]

Per prompt: the token streams must be equal, and so must every pick's gathered head records (each rank's max logit
and its id, fp32 bits: per-step logit-argmax equality). Extra files: the other ranks' outputs, which must carry the
same tokens and records as the first (every rank picks from the same gathered records). Prints one line
``PHASE2A PASS|FAIL ...`` last; exit 0 only on PASS.
"""

from __future__ import annotations

import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def first_diff(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    ref, gen = load(argv[0]), load(argv[1])
    others = [load(p) for p in argv[2:]]
    problems = []
    for key in ("world", "layers", "context", "window", "tokens"):
        if ref.get(key) != gen.get(key):
            problems.append(f"setting {key}: reference {ref.get(key)} vs zig {gen.get(key)}")
    rp = {p["name"]: p for p in ref["prompts"]}
    gp = {p["name"]: p for p in gen["prompts"]}
    if sorted(rp) != sorted(gp):
        problems.append(f"prompts differ: reference {sorted(rp)} vs zig {sorted(gp)}")
    tok_eq = tok_all = rec_eq = rec_all = 0
    first = []
    for name in rp:
        if name not in gp:
            continue
        r, g = rp[name], gp[name]
        if r["prompt_len"] != g["prompt_len"]:
            problems.append(f"{name}: prompt length {r['prompt_len']} vs {g['prompt_len']}")
        n = min(len(r["tokens"]), len(g["tokens"]))
        tok_all += len(r["tokens"])
        tok_eq += sum(1 for i in range(n) if r["tokens"][i] == g["tokens"][i])
        rec_all += len(r["records"])
        rec_eq += sum(1 for i in range(min(len(r["records"]), len(g["records"])))
                      if r["records"][i] == g["records"][i])
        dt = first_diff(r["tokens"], g["tokens"])
        dr = first_diff(r["records"], g["records"])
        status = "equal" if dt is None and dr is None else "DIFFERS"
        line = f"{name:12s} prompt {r['prompt_len']:6d}  picks {len(r['tokens']):4d}  {status}"
        if dr is not None:
            line += f"; first record difference at pick {dr}"
            if dr < len(r["records"]) and dr < len(g["records"]):
                line += f"\n    ref {r['records'][dr]}\n    zig {g['records'][dr]}"
        if dt is not None:
            line += f"; first token difference at pick {dt}"
            if dt < n:
                line += f" (ref {r['tokens'][dt]}, zig {g['tokens'][dt]})"
            first.append(f"{name}@{dt}")
        print(line)
        print(f"    ref prefill {r.get('prefill_s', 0):.1f}s decode {r.get('decode_s', 0):.1f}s | zig prefill "
              f"{g.get('prefill_s', 0):.1f}s decode {g.get('decode_s', 0):.1f}s")
    for o in others:
        for p in o["prompts"]:
            q = gp.get(p["name"])
            if q is None or p["tokens"] != q["tokens"] or p["records"] != q["records"]:
                problems.append(f"zig rank {o.get('rank')} disagrees with rank {gen.get('rank')} on {p['name']}")
    if ref.get("nccl_version") != gen.get("nccl_version"):
        print(f"note: NCCL {ref.get('nccl_version')} (reference) vs {gen.get('nccl_version')} (zig): gathers move bytes, "
              "so the bits do not depend on it")
    for p in problems:
        print("problem:", p)
    ok = not problems and tok_eq == tok_all and rec_eq == rec_all and tok_all > 0
    inv = gen.get("inv_freq", "?")
    print(f"PHASE2A {'PASS' if ok else 'FAIL'} prompts={len(rp)} tokens={tok_eq}/{tok_all} "
          f"records={rec_eq}/{rec_all} layers={gen.get('layers')} world={gen.get('world')} inv_freq=\"{inv}\""
          + (f" first_divergence={','.join(first)}" if first else "")
          + (f" problems={len(problems)}" if problems else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
