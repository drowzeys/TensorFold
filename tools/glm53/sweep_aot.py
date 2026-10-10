#!/usr/bin/env python3
"""Complete a GLM-5.3 Triton AOT set for the Zig engine: compile every launch shape the Zig side can request that the
Python reference's own schedule never launched.

Input: the needs JSON of ``tf-glm53-generate --mode list-variants --aot <captured set>`` (every launch key the engine
can produce - function, num_warps / num_stages, the constexprs it names, each runtime argument's type and
specialization class - each marked "have"). For every need without a variant this compiles the Triton function with
exactly that specialization (``JITFunction.warmup``: no launch, no memory touched) under the launch recorder, packs the
new cubins (triton_aot_manifest.py + aot_pack.py, as reference.pack does) and writes the union of the captured set(s)
and the new variants to --merged (aot.json + cubins/). Constexprs the Zig launch does not name (Python call-site
defaults such as BASE=None) and the exact Python floats behind fp32 constexprs come from a captured variant of the same
function (the --template sets' manifest.json); a need with no captured variant of its function falls back to the
function's defaults (reported). The run_phase3b script then checks the merged set with list-variants --aot again:
nothing missing, or no server starts.

usage (inside the image, one GPU, PYTHONPATH at the champion source's src; TRITON_CACHE_DIR a fresh directory):
  sweep_aot.py --needs NEEDS.json --template REFDIR [--template REFDIR2] --out DIR --merged DIR [--tools TREE]
  sweep_aot.py --merge-only --set AOTDIR [--set AOTDIR2 ...] --merged DIR       (no GPU: the union of packed sets)
REFDIR: a reference run's directory (manifest.json, aot/aot.json, aot/cubins/).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# where each function the Zig engine launches lives (cuda_triton.zig's Python call sites); a template's manifest
# ("function": module.qualname) takes precedence
MODULES = {
    "_rmsnorm": "tensorfold.families.glm5_next.cuda.glue",
    "_router_part": "tensorfold.families.glm5_next.cuda.glue",
    "_router_sum": "tensorfold.families.glm5_next.cuda.glue",
    "_topk": "tensorfold.families.glm5_next.cuda.glue",
    "_residual_add": "tensorfold.families.glm5_next.cuda.glue",
    "_merge": "tensorfold.families.glm5_next.cuda.latent",
    "_group_sums": "tensorfold.families.glm5_next.cuda.qmm",
    "_gemm": "tensorfold.cuda.exl3.prefill",
    "_select_keys": "tensorfold.families.glm_moe_dsa.cuda.topk",
    # Phase 4: the drafters (cuda_draft.zig's launches)
    "_embed_b16": "tensorfold.families.glm5_next.cuda.glue",
    "_swiglu": "tensorfold.families.glm5_next.cuda.glue",
    "_prep_kernel": "tensorfold.families.glm5_next.cuda.dflash2",
    "_dconv_kernel": "tensorfold.families.glm5_next.cuda.dflash2",
    "_dattn_ring": "tensorfold.families.glm_moe_dsa.cuda.dflash",
    # Phase 4c: the concurrent DFlash2 drafter (dflash.MultiDrafter's kernels)
    "_dconv_seg": "tensorfold.families.glm_moe_dsa.cuda.dflash",
    "_dattn_seg": "tensorfold.families.glm_moe_dsa.cuda.dflash",
    "_block_attn_kernel": "tensorfold.families.glm_moe_dsa.cuda.dspark",
}
FUSED = "tensorfold.families.glm_moe_dsa.cuda.fused"
DTYPES = {"*bf16": "bfloat16", "*fp16": "float16", "*fp32": "float32", "*i32": "int32", "*i64": "int64",
          "*u8": "uint8", "*i8": "int8", "*u32": "uint32", "*i16": "int16"}


def f32_of_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def load_templates(dirs: list[Path]) -> tuple[dict, dict]:
    """Per kernel name: the captured variants' constexprs (manifest JSON values) and metadata; name -> module path."""
    by_name: dict[str, list[dict]] = {}
    where: dict[str, str] = {}
    for d in dirs:
        m = d / "manifest.json"
        if not m.is_file():
            print(f"[sweep] no {m}: not a template", flush=True)
            continue
        for k in json.loads(m.read_text())["kernels"]:
            by_name.setdefault(k["name"], []).append(k)
            mod, _, _ = k["function"].rpartition(".")
            where.setdefault(k["name"], mod)
    return by_name, where


def const_matches(tv, need: dict) -> bool:
    """A template constexpr value against the need's {"int": x} / {"f32": bits}."""
    if "int" in need:
        if isinstance(tv, bool):
            return int(tv) == need["int"]
        return isinstance(tv, int) and tv == need["int"]
    if "f32" in need and isinstance(tv, dict) and "fp32_bits" in tv:
        return int(tv["fp32_bits"], 16) == need["f32"]
    return False


def best_template(cands: list[dict], need: dict) -> dict | None:
    """The captured variant agreeing with the most of the need's named constexprs (ties: the first)."""
    best, score = None, -1
    for k in cands:
        s = sum(1 for n, v in need["consts"].items() if n in k["constexprs"] and const_matches(k["constexprs"][n], v))
        if s > score:
            best, score = k, s
    return best


def py_const(name: str, v: dict, cands: list[dict]):
    """The Python value for one named constexpr: a bool/int as a template has it, a float as Python's own fp64."""
    if "int" in v:
        for k in cands:
            tv = k["constexprs"].get(name)
            if isinstance(tv, bool) and int(tv) == v["int"]:
                return tv
        return v["int"]
    bits = v["f32"]
    for k in cands:
        tv = k["constexprs"].get(name)
        if isinstance(tv, dict) and "fp32_bits" in tv and int(tv["fp32_bits"], 16) == bits:
            return struct.unpack("<d", struct.pack("<Q", int(tv["fp64_bits"], 16)))[0]
    print(f"[sweep] WARNING {name}: no captured fp64 value for fp32 bits {bits:#010x}; using the fp32 value", flush=True)
    return f32_of_bits(bits)


def template_value(tv):
    if isinstance(tv, dict) and "fp64_bits" in tv:
        return struct.unpack("<d", struct.pack("<Q", int(tv["fp64_bits"], 16)))[0]
    return tv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--needs", default="")
    ap.add_argument("--template", action="append", default=[], help="a reference run's dir (manifest.json, aot/)")
    ap.add_argument("--out", default="", help="this sweep's launches / manifest / aot")
    ap.add_argument("--merge-only", action="store_true", help="only merge the --set directories into --merged")
    ap.add_argument("--set", action="append", default=[], help="a packed set (aot.json + cubins/) for --merge-only")
    ap.add_argument("--merged", required=True, help="the union of the templates' aot sets and the sweep's")
    ap.add_argument("--tools", default="")
    a = ap.parse_args()
    if a.merge_only:
        merge([Path(x) for x in a.set], Path(a.merged))
        return 0
    if not (a.needs and a.out):
        ap.error("--needs and --out are needed for a sweep")
    if not os.environ.get("TRITON_CACHE_DIR"):
        raise SystemExit("TRITON_CACHE_DIR must name a fresh directory (the cubins are packed from it)")
    tree = Path(a.tools) if a.tools else HERE.parents[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    templates = [Path(t) for t in a.template]
    doc = json.loads(Path(a.needs).read_text())
    needs = [n for n in doc["needs"] if not n.get("have", False)]
    print(f"[sweep] {len(doc['needs'])} launch shapes, {len(needs)} without a captured variant", flush=True)

    sys.path.insert(0, str(tree / "tools" / "zig"))
    import triton_aot_manifest as aotm

    rec = aotm.Recorder()
    rec.record_warmup = True
    rec.install()

    import torch

    by_name, where = load_templates(templates)
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    bufs: dict[tuple[str, bool], torch.Tensor] = {}

    def ptr_arg(ty: str, aligned: bool) -> torch.Tensor:
        key = (ty, aligned)
        if key not in bufs:
            dt = getattr(torch, DTYPES[ty])
            base = torch.zeros(64, dtype=dt, device=dev)
            assert base.data_ptr() % 16 == 0
            bufs[key] = base if aligned else base[1:]
            assert (bufs[key].data_ptr() % 16 == 0) == aligned
        return bufs[key]

    fails, done, fallback = [], 0, set()
    t0 = time.time()
    for i, n in enumerate(needs):
        name = n["fn"]
        mod_name = where.get(name) or MODULES.get(name) or FUSED
        fn = getattr(importlib.import_module(mod_name), name)
        cands = by_name.get(name, [])
        tmpl = best_template(cands, n) if cands else None
        if tmpl is None:
            fallback.add(name)
        kwargs = {}
        for arg in n["args"]:
            if arg["kind"] == "ptr":
                kwargs[arg["name"]] = ptr_arg(arg["type"], arg["align16"])
            elif arg["kind"] == "i32":
                kwargs[arg["name"]] = int(arg["value"])
            elif arg["kind"] == "f32":
                kwargs[arg["name"]] = f32_of_bits(arg["bits"])
            else:
                kwargs[arg["name"]] = int(arg["value"])
        for cname, v in n["consts"].items():
            kwargs[cname] = py_const(cname, v, cands)
        # constexprs the Zig launch leaves to the call site (BASE=None, U32=False, ...): the template's values
        if tmpl is not None:
            constexpr_params = {p.name for p in fn.params if p.is_constexpr}
            for cname, tv in tmpl["constexprs"].items():
                if cname in constexpr_params and cname not in kwargs:
                    kwargs[cname] = template_value(tv)
        opts = {}
        if n.get("num_warps") is not None:
            opts["num_warps"] = int(n["num_warps"])
        if n.get("num_stages") is not None:
            opts["num_stages"] = int(n["num_stages"])
        elif tmpl is not None and tmpl["metadata"].get("num_stages") is not None:
            opts["num_stages"] = int(tmpl["metadata"]["num_stages"])
        try:
            fn.warmup(grid=(1,), **kwargs, **opts)
            done += 1
        except Exception as e:  # noqa: BLE001 - report every failure, then fail the sweep
            fails.append(f"{name} {n['consts']} {[(x['name'], x.get('value')) for x in n['args'] if x['kind'] == 'i32']}: {e}")
            print(f"[sweep] FAILED {fails[-1]}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"[sweep] {i + 1}/{len(needs)} compiled ({time.time() - t0:.0f} s)", flush=True)
    print(f"[sweep] compiled {done}/{len(needs)} in {time.time() - t0:.0f} s; {len(fails)} failed; no template "
          f"(defaults used) for: {sorted(fallback) or 'none'}", flush=True)
    if fails:
        return 1

    from oracle import specialization
    from reference import pack

    if done:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
        pack(tree, out, Path(os.environ["TRITON_CACHE_DIR"]))
    merge([*(t / "aot" for t in templates), *([out / "aot"] if done else [])], Path(a.merged))
    return 0


def merge(sets: list[Path], dest: Path) -> None:
    """aot.json + cubins/ of every set, one entry per hash (the first set's wins), sorted as aot_pack.py sorts."""
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "cubins").mkdir(parents=True)
    kernels, seen = [], set()
    for s in sets:
        if not (s / "aot.json").is_file():
            print(f"[sweep] merge: no {s}/aot.json, skipped", flush=True)
            continue
        for k in json.loads((s / "aot.json").read_text())["kernels"]:
            if k["hash"] in seen:
                continue
            seen.add(k["hash"])
            shutil.copyfile(s / "cubins" / f"{k['hash']}.cubin", dest / "cubins" / f"{k['hash']}.cubin")
            kernels.append(k)
    kernels.sort(key=lambda x: (x["fn"], x["hash"]))
    (dest / "aot.json").write_text(json.dumps({"generator": "tools/glm53/sweep_aot.py (merged)", "kernels": kernels},
                                              indent=1) + "\n")
    print(f"[sweep] merged {len(kernels)} variants of {len({k['fn'] for k in kernels})} functions from "
          f"{len(sets)} sets -> {dest}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
