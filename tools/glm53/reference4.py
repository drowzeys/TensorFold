#!/usr/bin/env python3
"""GLM-5.3's Python CUDA engine as the Zig port's Phase 4 reference: the speculative drafters. reference3a.py's served
configuration (FROZEN_3A: TF_EXL3_PROMPT_DET=slots16, prompt reuse off, profiling off) with the drafters loaded and
copy drafts allowed:

* TF_GLM53_DSPARK=<--dspark> (+ TF_GLM53_DSPARK_POLICY / _CONFIDENCE / _DEPTH from --dspark-policy / ...),
  TF_GLM53_DFLASH=<--dflash> (+ TF_GLM53_DFLASH_DEPTH / _CONFIDENCE), --k MTP drafts (0: --mtp-drafts 0, no MTP layer);
* TF_GLM53_COPY_DRAFTS=1 (the image default; MIN 8, MAX 15): the runner captures the copy widths, and each run asks
  for copies or not (Runner.generate's ``copies``).

Per prompt and run (labels as tf-glm53-generate --mode 4: "g:K" / "s<top_k>:K" the MTP head, K = 0 serial;
"g:dspark" / "s20:dflash" a drafter; a "+c" suffix copy drafts ahead of it): caches zeroed, ``Runner.generate`` with
EOS not a stop (every run makes the prompt's token count; tf-glm53-generate --stop-eos 0 does the same), the drafter
mode (``mode="dspark"`` / ``"dflash"``) and ``copies``. Writes <out>/ref-r<R>.json (3a's format plus each run's
"draft_mode", "copies", "copy_rounds" / "copy_drafted" / "copy_accepted"); --record also the Triton AOT set of
everything the run launched (the drafters' kernels: _prep_kernel, _block_attn_kernel, _dattn_ring, _dconv_kernel,
_embed_b16, _swiglu, _rmsnorm SUMS, ...) for the Zig engine's coverage sweep.

Modes:
  --make-prompts OUT  (CPU) the Phase 4 prompts: tfbench's tasks (prose_beekeeper, code_parser, code_cache) short
                      and with the 32K background (prose32k, code32k), and with --with-120k the 4x background ones
                      (prose120k, code120k); chat template with thinking on; seed 1729 (tfbench's first repeat);
                      runs per class from --runs-short / --runs-32k / --runs-120k
  (default)           one rank: --rank R --master IP --port P --model DIR --prompts FILE --out DIR --context N
                      [--k 0] [--dspark DIR ...] [--dflash DIR ...] [--record]

usage (inside the image, PYTHONPATH at the champion source's src).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference import nccl_path, nccl_version, pack  # noqa: E402
from reference3a import FROZEN_3A, bmm_probe, loaded_lib  # noqa: E402

# bench/bench.py's tasks (keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark): the published decode figures are measured on them
TASKS = {
    "prose_beekeeper": "Write a vivid literary short story (at least 700 words) about a beekeeper during a drought. "
                       "No headings, flowing prose only.",
    "code_parser": "Write a complete Python recursive-descent parser for arithmetic expressions with parentheses, unary "
                   "minus, exponentiation, multiplication and addition. Include helpful syntax errors, examples, and "
                   "unittest tests. Code only.",
    "code_cache": "Write a complete, well-documented Python module implementing an LRU cache with TTL expiry, thread "
                  "safety, statistics, and a small CLI demo. Name the class Cache7. Code only, no prose.",
}
BENCH_SEED = 1729
UNSET_4 = ("TF_GLM53_CAPTURE_DIR", "TF_GLM53_DEFAULT_DRAFT", "TF_GLM53_TILES", "TF_GLM53_DSPARK", "TF_GLM53_DFLASH",
           "TF_GLM53_MTP")


def parse_run4(text: str) -> tuple[bool, int, int, str | None, bool]:
    """(greedy, top_k, MTP drafts, drafter mode or None, copies) of a run label."""
    pick, _, rest = text.partition(":")
    copies = rest.endswith("+c")
    if copies:
        rest = rest[:-2]
    mode, k = None, 0
    if rest in ("dspark", "dflash"):
        mode = rest
    else:
        k = int(rest)
    if copies and mode is None and k == 0:
        raise SystemExit(f"run {text!r}: the serial reference never copies")
    if pick == "g":
        return True, 0, k, mode, copies
    if pick.startswith("s"):
        return False, int(pick[1:]), k, mode, copies
    raise SystemExit(f"run {text!r}: g:K / s<top_k>:K / g:dspark / s20:dflash+c ...")


def tile_rows(lins) -> list[list]:
    """tiles.rows: [k, n, bits, codebook, layout, K splits, warps] of every tunable linear (the boot's picks; a linear
    never retiled runs x3linear.plan's), in fused.Weights.tunable order - what tf-glm53-generate --tiles loads."""
    from tensorfold.cuda.exl3 import linear as x3linear

    out = []
    for x in lins:
        sk, wk = x.split if x.split is not None else x3linear.plan(x.k, x.n)
        out.append([int(x.k), int(x.n), float(x.bits), str(x.codebook), str(x.layout), int(sk), int(wk)])
    return out


def make_prompts(a) -> int:
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    model = Path(a.model)
    tmpl = ChatTemplate(model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    gen = json.loads((model / "generation_config.json").read_text()) if (model / "generation_config.json").exists() \
        else {}
    sampling = {"temperature": float(gen.get("temperature", 1.0)), "top_p": float(gen.get("top_p", 0.95)),
                "min_p": 0.0}
    background = Path(a.context_file).read_text() if a.context_file else ""
    rows = []

    def add(name: str, content: str, runs: str, tokens: int) -> None:
        labels = [r for r in runs.split(",") if r]
        if not labels:
            return
        text = tmpl.render([{"role": "user", "content": content}], tools=None, enable_thinking=True)
        ids = [int(t) for t in tok.encode(text, add_special_tokens=False).ids]
        rows.append({"name": name, "ids": ids, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "tokens": tokens, "runs": labels, "seed": BENCH_SEED, "seed_from_prompt": False})

    tasks = [t for t in a.tasks.split(",") if t]
    for t in tasks:
        kind = t.split("_")[0]
        add(f"{kind}_short_{t}", TASKS[t], a.runs_short, a.tokens)
    if background:
        for t in tasks:
            kind = t.split("_")[0]
            add(f"{kind}32k_{t}", "Background reading:\n" + background + "\n\nTask:\n" + TASKS[t], a.runs_32k,
                a.tokens)
        if a.with_120k:
            big = "\n\n".join([background] * 4)
            for t in tasks:
                kind = t.split("_")[0]
                add(f"{kind}120k_{t}", "Background reading:\n" + big + "\n\nTask:\n" + TASKS[t], a.runs_120k,
                    a.tokens)
    need = max(len(r["ids"]) + r["tokens"] for r in rows)
    doc = {"generator": "tools/glm53/reference4.py --make-prompts", "thinking": True, "model": str(model),
           "sampling": sampling, "context_needed": need, "prompts": rows}
    Path(a.make_prompts).write_text(json.dumps(doc) + "\n")
    print(f"[ref4] prompts -> {a.make_prompts}: " + ", ".join(f"{r['name']} {len(r['ids'])} ids x {r['runs']}"
                                                             for r in rows) + f"; context needed {need}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--make-prompts", default="")
    ap.add_argument("--context-file", default="")
    ap.add_argument("--tasks", default="prose_beekeeper,code_parser")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--runs-short", default="")
    ap.add_argument("--runs-32k", default="")
    ap.add_argument("--runs-120k", default="")
    ap.add_argument("--with-120k", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29771)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--dspark", default="")
    ap.add_argument("--dspark-policy", default="confidence")
    ap.add_argument("--dspark-confidence", default="0.3")
    ap.add_argument("--dspark-depth", default="")
    ap.add_argument("--dflash", default="")
    ap.add_argument("--dflash-depth", default="7")
    ap.add_argument("--dflash-confidence", default="0.3")
    ap.add_argument("--copy-min", default="8")
    ap.add_argument("--copy-max", default="15")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tools", default="")
    a = ap.parse_args()
    if a.make_prompts:
        return make_prompts(a)
    if not (a.prompts and a.out):
        ap.error("--prompts and --out are needed for a run")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(FROZEN_3A)
    env["TF_GLM53_COPY_DRAFTS"] = "1"
    env["TF_GLM53_COPY_MIN"] = a.copy_min
    env["TF_GLM53_COPY_MAX"] = a.copy_max
    for k in UNSET_4:
        os.environ.pop(k, None)
    if a.dspark:
        env["TF_GLM53_DSPARK"] = a.dspark
        env["TF_GLM53_DSPARK_POLICY"] = a.dspark_policy
        env["TF_GLM53_DSPARK_CONFIDENCE"] = a.dspark_confidence
        if a.dspark_depth:
            env["TF_GLM53_DSPARK_DEPTH"] = a.dspark_depth
    if a.dflash:
        env["TF_GLM53_DFLASH"] = a.dflash
        env["TF_GLM53_DFLASH_DEPTH"] = a.dflash_depth
        env["TF_GLM53_DFLASH_CONFIDENCE"] = a.dflash_confidence
    env["TF_GLM53_TILES"] = f"save:{out}/tiles-r{a.rank}.json"
    for k, v in env.items():
        os.environ[k] = v
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

    doc_in = json.loads(Path(a.prompts).read_text())
    prompts = doc_in["prompts"]
    sset = doc_in.get("sampling", {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0})
    context = a.context or int(doc_in.get("context_needed", 32768))
    torch.cuda.set_device(0)
    t0 = time.time()
    eng = Glm53Engine(Path(a.model), rank=a.rank, master=a.master, port=a.port, context=context, mtp_drafts=a.k)
    init_s = time.time() - t0
    rn = eng.runner
    w = rn.w
    if w.dcp != 1:
        raise SystemExit(f"reference: DCP {w.dcp} at context {context}; keep the context under the DCP cut")
    print(f"[ref4] rank {a.rank}: engine up in {init_s:.0f}s (load {eng.load_s:.0f}s), context {context}, k {rn.k}, "
          f"RoCE {'on' if w.fast is not None else 'off'}, DSpark {rn.dspark is not None}, DFlash2 "
          f"{rn.drafter is not None}, copy drafts {rn.copy_on} (widths {rn.widths})", flush=True)
    lib_path, lib_ver = nccl_path(eng.comm.lib), nccl_version(eng.comm.lib)
    (out / "nccl.txt").write_text(f"{lib_path}\n{lib_ver}\n")
    tiles_path = out / f"tiles-r{a.rank}.json"
    if not tiles_path.exists():                  # a source without TF_GLM53_TILES (tiles.py): the same table
        tiles_path.write_text(json.dumps({"count": len(w.tunable), "linears": tile_rows(w.tunable)}, sort_keys=True))
        print(f"[ref4] rank {a.rank}: wrote the boot's tile table ({len(w.tunable)} linears) to {tiles_path}", flush=True)
    (out / "inv_freq.bin").write_bytes(w.inv.contiguous().cpu().numpy().tobytes())
    probe = bmm_probe(w.heads, w.cfg.qk_nope_head_dim, w.cfg.qk_rope_head_dim, w.cfg.kv_lora_rank, w.cfg.v_head_dim)
    (out / "bmm_probe.json").write_text(json.dumps(probe, indent=1) + "\n")
    from tensorfold.families.glm_moe_dsa.cuda import weights as weights_mod

    blas = {"libcublas": loaded_lib("libcublas.so"), "libcublasLt": loaded_lib("libcublasLt.so"),
            "workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""), "experts_impl": weights_mod.EXPERTS_IMPL,
            "torch": torch.__version__}
    (out / "blas.json").write_text(json.dumps(blas, indent=1) + "\n")

    def zero_state() -> None:
        st = rn.st
        for t in list(st.kc) + list(st.ic.values()) + [x for x in (st.mkc, st.mic) if x is not None]:
            for p in (t.planes() if hasattr(t, "planes") else [t]):
                p.zero_()
        st.pos.zero_()
        st.mpos.zero_()
        rn.carry.zero_()
        torch.cuda.synchronize()

    results = []
    with torch.no_grad():
        for p in prompts:
            ids = [int(x) for x in p["ids"]]
            temp = float(p.get("temperature", sset.get("temperature", 1.0)))
            top_p = float(p.get("top_p", sset.get("top_p", 0.95)))
            min_p = float(p.get("min_p", sset.get("min_p", 0.0)))
            for run in p["runs"]:
                greedy, top_k, k, mode, copies = parse_run4(run)
                if k > a.k:
                    raise SystemExit(f"run {run}: more MTP drafts than --k {a.k}")
                s = None if greedy else Sampling(int(p["seed"]), temp, top_k, top_p, min_p)
                sample = None if s is None else (lambda lg, pos, s=s: eng._sample(lg, pos, s))
                zero_state()
                st = rn.generate(ids, int(p["tokens"]), sample, lambda t: False, lambda toks: None, k, mode=mode,
                                 sampling=s, copies=copies)
                toks = [int(t) for t in st["out"]]
                pf = len(ids) / st["prefill_s"] if st["prefill_s"] > 0 else 0.0
                print(f"[ref4] rank {a.rank} {p['name']} {run}: prompt {len(ids)} in {st['prefill_s']:.2f}s "
                      f"({pf:.0f} tok/s); {len(toks)} tokens, decode {st['tok_s']:.2f} tok/s, "
                      f"{st['tokens_per_round']} tokens/round ({st['rounds']} rounds), accept {st['accept']}, copies "
                      f"{st.get('copy_rounds', '-')} rounds; first {toks[:8]}", flush=True)
                results.append({"name": p["name"], "run": run, "prompt_len": len(ids), "seed": str(p["seed"]), "k": k,
                                "greedy": greedy, "top_k": top_k, "draft_mode": mode or "mtp", "copies": copies,
                                "prefill_s": round(st["prefill_s"], 3), "decode_s": round(st["decode_s"], 3),
                                "tok_s": st["tok_s"], "rounds": st["rounds"],
                                "tokens_per_round": st["tokens_per_round"], "accept": st["accept"] or 0.0,
                                "copy_rounds": st.get("copy_rounds", 0), "copy_drafted": st.get("copy_drafted", 0),
                                "copy_accepted": st.get("copy_accepted", 0), "capture_s": st["capture_s"],
                                "graphs": st["graphs"], "tokens": toks})
        torch.cuda.synchronize()
    eng.comm.barrier()
    doc = {"tool": "tools/glm53/reference4.py", "mode": "4", "rank": a.rank, "world": engine_mod.WORLD,
           "layers": len(w.layers), "context": context, "capacity": rn.capacity, "k": rn.k, "env": env,
           "nccl_lib": lib_path, "nccl_version": lib_ver, "load_s": round(eng.load_s, 1), "init_s": round(init_s, 1),
           "roce": w.fast is not None, "sampling": sset, "widths": list(rn.widths), "copy_on": bool(rn.copy_on),
           "dspark": os.path.basename(a.dspark.rstrip("/")) if a.dspark else "",
           "dflash": os.path.basename(a.dflash.rstrip("/")) if a.dflash else "", "runs": results}
    (out / f"ref-r{a.rank}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[ref4] rank {a.rank}: wrote {out / f'ref-r{a.rank}.json'}", flush=True)
    if rec is not None:
        from oracle import specialization

        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
        pack(tree, out, Path(os.environ["TRITON_CACHE_DIR"]))
    print("[ref4] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
