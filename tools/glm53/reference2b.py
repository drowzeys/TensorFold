#!/usr/bin/env python3
"""GLM-5.3's Python CUDA engine as the Zig port's Phase 2b reference: the champion's own ``Runner`` (runner.py) with
the MTP layer, the 4-bit draft head and CUDA graphs, decoding greedy and keyed sampled requests exactly as a served
request runs (minus what the Zig port does not have yet, frozen off below).

Per prompt and run ("g:K" greedy, "s<top_k>:K" sampled at the prompts file's temperature / top_p / min_p with the
prompt's seed; K MTP drafts a round, 0 = serial): the caches are zeroed (both engines start every request from
zeros), then ``Runner.generate`` with stop() never true (EOS is not a stop: every run makes the prompt's token
count), copies off. Its tokens and stats (prefill_s, decode_s, tok_s, rounds, tokens_per_round, accept, capture_s)
go to ref-r<R>.json in the format tf-glm53-generate --mode 2b writes; compare_phase2b.py checks tokens, drafted ==
serial and prints both engines' decode speed.

Frozen (beyond oracle.FROZEN; every knob is one the Zig port implements, or one it must not see):

* ``TF_GLM53_PROMPT_ROWS`` = ``TF_GLM53_PROMPT_ROWS_SHORT`` = --window (128): prompt chunks the row-invariant
  path takes (the Zig port has no x3prefill GEMM / bmm absorb yet), chunked by prefixes.even_chunks on both sides.
* ``TF_GLM53_RADIX_MIN_ROWS=1`` (radix select for every window), the communicator shows only ``all_gather``: every
  exchange is the all-gather + rank-order sum (no bf16 ring all-reduce, no RoCE one-shot: TF_GLM53_ROCE is not
  started here) - decode speed is compared NCCL against NCCL.
* ``TF_GLM53_COPY_DRAFTS=0`` (copy drafts are Phase 4), no depth policy, no DFlash2, no L2 prefetch, one stream.
* ``TF_GLM53_MTP=normed/normed``, ``TF_GLM53_MTP_REUSE=2``, ``TF_GLM53_DRAFT_HEAD=q4``, ``TF_GLM53_DRAFT_VOCAB=32768``,
  ``TF_GLM53_VERIFY_HEAD=bf16``, ``TF_GLM53_SHARDED_SAMPLE=1``: the champion's defaults, which the Zig port follows.

Modes:
  --make-prompts OUT   (CPU) the 2a prompts (3 chat prompts + the ~4,500-token one) and the 32K prose prompt
                       (tfbench.py's: "Background reading:" + --context-file + "Task:" + prose_beekeeper, seed 1729),
                       chat template with thinking on; each with its runs and token count
  (default)            one rank: --rank R --world W --master IP --port P --model DIR --prompts FILE --out DIR
                       [--context N] [--window 128] [--k 2] [--layers N] [--record]

Must run inside the GLM-5.3 TP4 image with PYTHONPATH at the champion source's ``src``.
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

from reference import (PROMPTS, LONG_TOKENS, GatherComm, long_prompt, nccl_path, nccl_version,  # noqa: E402
                       pack)

# tfbench.py / bench.py's prose task (the 32K MTP prose figure is measured on it, seed 1729)
BEEKEEPER = ("Write a vivid literary short story (at least 700 words) about a beekeeper during a drought. No headings, "
             "flowing prose only.")
BENCH_SEED = 1729
DEFAULT_RUNS = "g:2,g:0,s20:2,s20:0,s0:2,s0:0"
DEFAULT_RUNS_LONG = "s20:2,s20:0"


def make_prompts(a) -> int:
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.engine.exact_sampling import seed_for

    model = Path(a.model)
    tmpl = ChatTemplate(model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    gen = json.loads((model / "generation_config.json").read_text()) if (model / "generation_config.json").exists() \
        else {}
    sampling = {"temperature": float(gen.get("temperature", 1.0)), "top_p": float(gen.get("top_p", 0.95)),
                "min_p": 0.0}

    def ids_of(content: str) -> tuple[list[int], str]:
        text = tmpl.render([{"role": "user", "content": content}], tools=None, enable_thinking=True)
        return [int(t) for t in tok.encode(text, add_special_tokens=False).ids], text

    runs = [r for r in a.runs.split(",") if r]
    runs_long = [r for r in a.runs_long.split(",") if r]
    rows = []

    def add(name, ids, text, tokens, run_list, seed=None):
        rows.append({"name": name, "ids": ids, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "tokens": tokens, "runs": run_list,
                     "seed": int(seed_for(ids)) if seed is None else int(seed), "seed_from_prompt": seed is None})

    for name, content in PROMPTS:
        ids, text = ids_of(content)
        add(name, ids, text, a.tokens, runs)
    if a.long > 0:
        n = max(16, a.long // 20)
        while True:
            ids, text = ids_of(long_prompt(n))
            if len(ids) >= a.long or n > 100000:
                break
            n = int(n * a.long / max(len(ids), 1)) + 8
        add(f"long{len(ids)}", ids, text, a.tokens, runs)
    if a.context_file:
        body = "Background reading:\n" + Path(a.context_file).read_text() + "\n\nTask:\n" + BEEKEEPER
        ids, text = ids_of(body)
        add("prose32k", ids, text, a.tokens_long, runs_long, seed=BENCH_SEED)
    need = max(len(r["ids"]) + r["tokens"] for r in rows)
    doc = {"generator": "tools/glm53/reference2b.py --make-prompts", "thinking": True, "model": str(model),
           "sampling": sampling, "context_needed": need, "prompts": rows}
    Path(a.make_prompts).write_text(json.dumps(doc) + "\n")
    print(f"[ref2b] prompts -> {a.make_prompts}: " + ", ".join(f"{r['name']} {len(r['ids'])} ids x {r['runs']}"
                                                              for r in rows) + f"; context needed {need}", flush=True)
    return 0


FROZEN_2B = {
    "TF_GLM53_RADIX_MIN_ROWS": "1",
    "TF_GLM53_COPY_DRAFTS": "0",
    "TF_GLM53_DEPTH_POLICY": "",
    "TF_GLM53_MTP": "normed/normed",
    "TF_GLM53_MTP_REUSE": "2",
    "TF_GLM53_DRAFT_HEAD": "q4",
    "TF_GLM53_DRAFT_VOCAB": "32768",
    "TF_GLM53_VERIFY_HEAD": "bf16",
    "TF_GLM53_SHARDED_SAMPLE": "1",
    "TF_GLM53_CAPTURE_LABEL": "0",
    "TF_GLM53_PROMPT_SP": "0",
    "TENSORFOLD_NUCLEUS_UNION": "0",
    "TENSORFOLD_SEED_SALT": "",
    "TF_GLM53_PROFILE_FLAG": "/tmp/tf-glm53-no-profile/PROFILE",
}
UNSET_2B = ("TF_GLM53_DFLASH", "TF_GLM53_CAPTURE_DIR", "TF_GLM53_DCP")


def parse_run(text: str) -> tuple[bool, int, int]:
    mode, _, k = text.partition(":")
    if mode == "g":
        return True, 0, int(k)
    if mode.startswith("s"):
        return False, int(mode[1:]), int(k)
    raise SystemExit(f"run {text!r}: g:K or s<top_k>:K")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--make-prompts", default="")
    ap.add_argument("--long", type=int, default=LONG_TOKENS)
    ap.add_argument("--context-file", default="", help="the 32K prose prompt's background text (tfbench's)")
    ap.add_argument("--tokens", type=int, default=256, help="picks a run of the short prompts")
    ap.add_argument("--tokens-long", type=int, default=512, help="picks a run of the 32K prose prompt")
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--runs-long", default=DEFAULT_RUNS_LONG)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--master", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29571)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--context", type=int, default=0, help="the engine's context (0: the prompts file's need)")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--no-prewarm", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--tools", default="")
    a = ap.parse_args()
    if a.make_prompts:
        return make_prompts(a)
    if not (a.prompts and a.out):
        ap.error("--prompts and --out are needed for a run")

    from oracle import FROZEN, freeze_env, specialization

    env = freeze_env()
    for k, v in FROZEN_2B.items():
        os.environ[k] = v
        env[k] = v
    os.environ["TF_GLM53_PROMPT_ROWS"] = env["TF_GLM53_PROMPT_ROWS"] = str(a.window)
    os.environ["TF_GLM53_PROMPT_ROWS_SHORT"] = env["TF_GLM53_PROMPT_ROWS_SHORT"] = str(a.window)
    for k in UNSET_2B:
        os.environ.pop(k, None)
    tree = Path(a.tools) if a.tools else HERE.parents[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.record:
        if not os.environ.get("TRITON_CACHE_DIR"):
            raise SystemExit("--record needs TRITON_CACHE_DIR (a fresh directory: the cubins are packed from it)")
        sys.path.insert(0, str(tree / "tools" / "zig"))
        import triton_aot_manifest as aotm

        rec = aotm.Recorder().install()

    import numpy as np
    import torch

    from tensorfold.cuda.comm import NCCL
    from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows
    from tensorfold.families.glm_moe_dsa.config import Config
    from tensorfold.families.glm_moe_dsa.cuda import fused, headq
    from tensorfold.families.glm_moe_dsa.cuda import runner as runner_mod
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_layers, load_mtp, trim_host

    assert fused.RADIX_MIN_ROWS == 1 and not fused.TUNE and not fused.SIDE, "the frozen knobs did not take"
    assert fused.PROMPT_ROWS == a.window and fused.PROMPT_ROWS_SHORT == a.window, "prompt rows did not take"
    assert fused.MTP_REUSE == 2 and fused.DRAFT_VOCAB == 32768 and headq.DRAFT_HEAD == "q4", "MTP knobs did not take"
    assert runner_mod.SHARDED, "TF_GLM53_SHARDED_SAMPLE did not take"
    torch.backends.cuda.matmul.allow_tf32 = False
    doc_in = json.loads(Path(a.prompts).read_text())
    prompts = doc_in["prompts"]
    sset = doc_in.get("sampling", {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0})
    context = a.context or int(doc_in.get("context_needed", 32768))
    torch.cuda.set_device(0)
    t0 = time.time()
    nccl = NCCL(a.rank, a.world, a.master, a.port)
    comm = GatherComm(nccl)
    lib_path, lib_ver = nccl_path(nccl.lib), nccl_version(nccl.lib)
    print(f"[ref2b] rank {a.rank}: NCCL {lib_ver} from {lib_path}", flush=True)
    (out / "nccl.txt").write_text(f"{lib_path}\n{lib_ver}\n")

    cfg = Config.from_dict(json.loads((Path(a.model) / "config.json").read_text()))
    n_layers = a.layers or cfg.num_hidden_layers
    with torch.no_grad():
        r = RankReader(Path(a.model), a.rank, a.world)
        embed, norm, head = (r.get(t, "cuda") for t in ("model.embed_tokens.weight", "model.norm.weight",
                                                         "lm_head.weight"))
        r.release()
        layers = load_layers(r, cfg, n_layers, verbose=a.rank == 0)
        mtp = load_mtp(r, cfg) if a.k else None
        r.release()
        w = fused.Weights(cfg, a.rank, a.world, comm, embed, norm, head, layers, mtp)
        if a.k and fused.DRAFT_VOCAB:                    # Glm53Engine's reduced draft vocabulary, WORLD = a.world
            q = fused.DRAFT_VOCAB // a.world
            sl = r._file("lm_head.weight").get_slice("lm_head.weight")
            V = sl.get_shape()[0]
            spans = [(a.rank * q, (a.rank + 1) * q)] + ([(V - fused.SPECIALS, V)] if a.rank == a.world - 1 else [])
            w.set_draft_head(torch.cat([sl[x:y] for x, y in spans]).cuda(),
                             torch.cat([torch.arange(x, y) for x, y in spans]).cuda())
        r.drop()
        headq.prepare(w, drafts=bool(a.k))
        r.drop_page_cache()
        del r
        trim_host()
        if w.dcp != 1 or w.fast is not None or w.l2pf is not None:
            raise SystemExit("reference: DCP 1, no RoCE one-shot and no L2 prefetch expected")
        load_s = time.time() - t0
        print(f"[ref2b] rank {a.rank}: {n_layers} layers{' + MTP layer' if mtp else ''} + head loaded in {load_s:.0f}s; "
              f"vocabulary rows {w.vocab_off}..{w.vocab_off + w.vocab_part}", flush=True)
        (out / "inv_freq.bin").write_bytes(w.inv.contiguous().cpu().numpy().tobytes())
        capacity = context + a.k + 1                     # Glm53Engine: Runner(limit + k + 1)
        rn = Runner(w, capacity, a.k, graphs=True)
        prewarm_s, prewarm_graphs = 0.0, 0
        if not a.no_prewarm:
            tp = time.time()
            rn.prewarm()
            prewarm_s, prewarm_graphs = time.time() - tp, len(rn.G.graphs)
        nccl.barrier()

        def zero_state() -> None:
            st = rn.st
            for t in list(st.kc) + list(st.ic.values()) + [x for x in (st.mkc, st.mic) if x is not None]:
                t.zero_()
            st.pos.zero_()
            st.mpos.zero_()
            rn.carry.zero_()
            torch.cuda.synchronize()

        def engine_sample(s):                            # Glm53Engine._sample
            def sample(logits, position):
                kk = min(logits.shape[-1], (int(s.top_k) if s.top_k else 256) + MARGIN)
                vals, ids = torch.topk(logits[-1:].float(), kk, dim=-1)
                return int(choose_rows(vals.cpu().numpy(), ids.cpu().numpy().astype(np.int64), [position], s)[0])
            return sample

        results = []
        for p in prompts:
            ids = [int(x) for x in p["ids"]]
            for run in p["runs"]:
                greedy, top_k, k = parse_run(run)
                if k > a.k:
                    raise SystemExit(f"run {run}: more drafts than --k {a.k}")
                s = None if greedy else Sampling(int(p["seed"]), float(sset["temperature"]), top_k,
                                                 float(sset["top_p"]), float(sset.get("min_p", 0.0)))
                zero_state()
                st = rn.generate(ids, int(p["tokens"]), engine_sample(s) if s else None, lambda t: False,
                                 lambda toks: None, k, mode=None, sampling=s, copies=False)
                toks = [int(t) for t in st["out"]]
                print(f"[ref2b] rank {a.rank} {p['name']} {run}: prompt {len(ids)} in {st['prefill_s']:.1f}s; "
                      f"{len(toks)} tokens, decode {st['tok_s']:.2f} tok/s, {st['tokens_per_round']} tokens/round "
                      f"({st['rounds']} rounds), accept {st['accept']}, capture {st['capture_s']}s; first {toks[:8]}",
                      flush=True)
                results.append({"name": p["name"], "run": run, "prompt_len": len(ids), "seed": str(p["seed"]),
                                "k": k, "greedy": greedy, "top_k": top_k, "prefill_s": round(st["prefill_s"], 3),
                                "decode_s": round(st["decode_s"], 3), "tok_s": st["tok_s"], "rounds": st["rounds"],
                                "tokens_per_round": st["tokens_per_round"], "accept": st["accept"] or 0.0,
                                "capture_s": st["capture_s"], "graphs": st["graphs"], "tokens": toks})
        torch.cuda.synchronize()
        nccl.barrier()

    doc = {"tool": "tools/glm53/reference2b.py", "mode": "2b", "rank": a.rank, "world": a.world, "layers": n_layers,
           "context": context, "capacity": capacity, "window": a.window, "k": a.k, "vocab_off": w.vocab_off,
           "vocab_part": w.vocab_part, "nccl_version": lib_ver, "nccl_lib": lib_path, "env": env, "frozen_base": FROZEN,
           "torch": torch.__version__, "load_s": round(load_s, 1), "prewarm_graphs": prewarm_graphs,
           "prewarm_s": round(prewarm_s, 1), "sampling": sset,
           "inv_freq_sha256": hashlib.sha256((out / "inv_freq.bin").read_bytes()).hexdigest(), "runs": results}
    try:
        import triton

        doc["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    (out / f"ref-r{a.rank}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[ref2b] rank {a.rank}: wrote {out / f'ref-r{a.rank}.json'}", flush=True)
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
        pack(tree, out, Path(os.environ["TRITON_CACHE_DIR"]))
    print("[ref2b] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
