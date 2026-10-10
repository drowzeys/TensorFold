#!/usr/bin/env python3
"""GLM-5.3's Python CUDA engine as the Zig port's Phase 3b reference for long contexts: reference3a.py's served
configuration (FROZEN_3A: TF_EXL3_PROMPT_DET=slots16, copy drafts off, prompt reuse off, profiling off) WITH decode
context parallelism allowed - TF_GLM53_DCP passes through (unset: the engine's own rule, DCP 4 above 200,000 tokens of
context; --dcp N forces it). Two uses in run_phase3b_cluster.sh:

* PY1M=1: the 1M needle prompt at --context 1000000 (DCP auto -> 4): decode speed at depth for the gate, and with
  --record the Triton AOT set the Zig engine's DCP path needs (_attn_dcp, _dcp_combine, DCP=4 variants of _kv_write /
  _ik_write / _index_scores).
* PY1M=0: a SHORT capture (--dcp 4 at a small context, a ~20K-token prompt, a few tokens) whose only purpose is that
  AOT set.

Per prompt and run ("g:K" / "s<top_k>:K"; a prompt's own "temperature" / "top_p" / "min_p" override the file's
"sampling"): caches zeroed, ``Runner.generate`` with EOS as a stop (as tf-glm53-generate --mode 3b --stop-eos 1).
Writes <out>/ref-r<R>.json (3a's format plus "dcp" and per-run "peak_gib"); --record also the AOT pack in <out>/aot.

usage (one rank, inside the image, PYTHONPATH at the champion source's src):
  reference3b.py --model DIR --rank R --master IP --port P --prompts FILE --out DIR --context N [--k 2] [--dcp 0|1|4]
                 [--record] [--tiles save|none]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference import nccl_path, nccl_version, pack  # noqa: E402
from reference3a import FROZEN_3A  # noqa: E402

UNSET_3B = ("TF_GLM53_DFLASH", "TF_GLM53_CAPTURE_DIR", "TF_GLM53_DEFAULT_DRAFT", "TF_GLM53_TILES")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29671)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--dcp", type=int, default=-1, help="-1: leave TF_GLM53_DCP as the environment has it")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tiles", default="save", choices=("save", "none"))
    ap.add_argument("--tools", default="")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(FROZEN_3A)
    for k, v in env.items():
        os.environ[k] = v
    for k in UNSET_3B:
        os.environ.pop(k, None)
    if a.dcp >= 0:
        os.environ["TF_GLM53_DCP"] = str(a.dcp)
    if a.tiles == "save":
        os.environ["TF_GLM53_TILES"] = f"save:{out}/tiles-r{a.rank}.json"
    env["TF_GLM53_DCP"] = os.environ.get("TF_GLM53_DCP", "")
    tree = Path(a.tools) if a.tools else HERE.parents[1]
    rec = None
    if a.record:
        if not os.environ.get("TRITON_CACHE_DIR"):
            raise SystemExit("--record needs TRITON_CACHE_DIR (a fresh directory)")
        sys.path.insert(0, str(tree / "tools" / "zig"))
        import triton_aot_manifest as aotm

        rec = aotm.Recorder().install()

    import torch

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm_moe_dsa.cuda import engine as engine_mod
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    from reference2b import parse_run

    doc_in = json.loads(Path(a.prompts).read_text())
    prompts = doc_in["prompts"]
    sset = doc_in.get("sampling", {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0})
    torch.cuda.set_device(0)
    t0 = time.time()
    eng = Glm53Engine(Path(a.model), rank=a.rank, master=a.master, port=a.port, context=a.context, mtp_drafts=a.k)
    init_s = time.time() - t0
    rn = eng.runner
    w = rn.w
    print(f"[ref3b] rank {a.rank}: engine up in {init_s:.0f}s (load {eng.load_s:.0f}s), context {a.context}, DCP "
          f"{w.dcp}, RoCE {'on' if w.fast is not None else 'off'}", flush=True)
    lib_path, lib_ver = nccl_path(eng.comm.lib), nccl_version(eng.comm.lib)
    (out / "nccl.txt").write_text(f"{lib_path}\n{lib_ver}\n")

    def zero_state() -> None:
        st = rn.st
        for t in list(st.kc) + list(st.ic.values()) + [x for x in (st.mkc, st.mic) if x is not None]:
            for p in (t.planes() if hasattr(t, "planes") else [t]):
                p.zero_()
        st.pos.zero_()
        st.mpos.zero_()
        rn.carry.zero_()
        torch.cuda.synchronize()

    eos = set(int(x) for x in eng.eos)
    results = []
    with torch.no_grad():
        for p in prompts:
            ids = [int(x) for x in p["ids"]]
            temp = float(p.get("temperature", sset.get("temperature", 1.0)))
            top_p = float(p.get("top_p", sset.get("top_p", 0.95)))
            min_p = float(p.get("min_p", sset.get("min_p", 0.0)))
            for run in p["runs"]:
                greedy, top_k, k = parse_run(run)
                s = None if greedy else Sampling(int(p["seed"]), temp, top_k, top_p, min_p)
                sample = None if s is None else (lambda lg, pos, s=s: eng._sample(lg, pos, s))
                zero_state()
                torch.cuda.reset_peak_memory_stats()
                st = rn.generate(ids, int(p["tokens"]), sample, lambda t: t in eos, lambda toks: None, min(k, a.k),
                                 mode=None, sampling=s, copies=False)
                toks = [int(t) for t in st["out"]]
                free, total = torch.cuda.mem_get_info()
                pf = len(ids) / st["prefill_s"] if st["prefill_s"] > 0 else 0.0
                print(f"[ref3b] rank {a.rank} {p['name']} {run}: prompt {len(ids)} in {st['prefill_s']:.1f}s "
                      f"({pf:.0f} tok/s); {len(toks)} tokens, decode {st['tok_s']:.2f} tok/s, "
                      f"{st['tokens_per_round']} tokens/round; device free {free / 2**30:.1f} GiB", flush=True)
                results.append({"name": p["name"], "run": run, "prompt_len": len(ids), "seed": str(p["seed"]), "k": k,
                                "greedy": greedy, "top_k": top_k, "prefill_s": round(st["prefill_s"], 3),
                                "decode_s": round(st["decode_s"], 3), "tok_s": st["tok_s"], "rounds": st["rounds"],
                                "tokens_per_round": st["tokens_per_round"], "accept": st["accept"] or 0.0,
                                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                "free_gib": round(free / 2**30, 2), "tokens": toks})
        torch.cuda.synchronize()
    eng.comm.barrier()
    doc = {"tool": "tools/glm53/reference3b.py", "mode": "3b", "rank": a.rank, "world": engine_mod.WORLD,
           "context": a.context, "capacity": rn.capacity, "dcp": int(w.dcp), "k": a.k, "env": env,
           "nccl_lib": lib_path, "nccl_version": lib_ver, "load_s": round(eng.load_s, 1), "init_s": round(init_s, 1),
           "roce": w.fast is not None, "sampling": sset, "runs": results}
    (out / f"ref-r{a.rank}.json").write_text(json.dumps(doc, indent=1) + "\n")
    if rec is not None:
        from oracle import specialization

        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
        pack(tree, out, Path(os.environ["TRITON_CACHE_DIR"]))
    print("[ref3b] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
