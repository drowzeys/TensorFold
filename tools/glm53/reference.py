#!/usr/bin/env python3
"""GLM-5.3's Python CUDA engine (families/glm_moe_dsa, the fused path) as the Zig port's Phase 2a reference.

Full GLM-5.3 at tensor parallelism (one process a rank, NCCL), GREEDY, no drafts, on exactly the schedule
``tf-glm53-generate`` runs: a fresh ``fused.State`` a prompt, the prompt in windows of up to ``--window`` rows
(``fused.compute``; the last window's last row picks through ``fused.head`` mode "argmax"), then one-row decode
windows, ``--tokens`` picks a prompt (EOS is not a stop). Every pick's gathered head records (``b.amax_all``: each
rank's [max logit, id, 0, 0] fp32) are written as hex words beside the tokens, so tools/glm53/compare_phase2a.py can
check per-step logit-argmax equality bit for bit, not only the tokens.

What it freezes (the champion's knobs the Zig port implements; oracle.FROZEN plus):

* ``TF_GLM53_RADIX_MIN_ROWS=1``: the indexer's radix select for every window (the Zig side's selection; Phase 1's
  oracle checked it picks the bits torch.topk + sort does on decode windows).
* the communicator is NCCL (tensorfold.cuda.comm.NCCL, the engine's own, same TCPStore bootstrap) wrapped so it shows
  only ``all_gather`` / ``barrier``: fused.gather then takes its all-gather + rank-order-sum branch for every window
  (prompt windows too, instead of the bf16 ring all-reduce / exact reduce-scatter), which is what the Zig forward
  implements in 2a. No RoCE one-shot (``w.fast`` stays None), no L2 prefetch, DCP 1, bf16 latent cache, no MTP layer.

Modes:
  --make-prompts OUT   (CPU) render the chat template (thinking on) for the fixed prompts and tokenize them:
                       {"prompts": [{"name", "ids", "text_sha256"}...]} - the ids both engines read
  (default)            one rank of the run: --rank R --world W --master IP --port P --model DIR --prompts FILE
                       --tokens N --out DIR [--context 32768] [--window 128] [--layers N] [--record]
                       writes DIR/ref-r<R>.json, DIR/inv_freq.bin (torch's fused.inv_freq bytes), DIR/nccl.txt and
                       with --record DIR/aot/{aot.json,cubins/} (TRITON_CACHE_DIR must be fresh): the Triton kernel
                       set the Zig rank on this node loads (every specialization this schedule launches)

Must run inside the GLM-5.3 TP4 image with PYTHONPATH at the champion source's ``src``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROMPTS = (
    ("prose", "Write a detailed explanation of how a hash table works, including collisions and resizing."),
    ("code", "Write a Python function that parses an ISO 8601 date string without using datetime, with tests."),
    ("reason", "Explain step by step why the sum of the first n odd numbers is n squared, then give two "
               "different proofs."),
)
LONG_TOKENS = 4500           # the long prompt: past index_topk (2048), so prompt and decode windows select keys


def long_prompt(n_lines: int) -> str:
    filler = [f"Entry {i}: shelf {i % 97} holds volume {i * 31 % 1009} of the river survey." for i in range(n_lines)]
    filler.insert(len(filler) // 2, "The passphrase is CRIMSON-OTTER-7731.")
    return "\n".join(filler) + "\n\nWhat is the passphrase? Reply with it only."


def make_prompts(model: Path, out: Path, long_tokens: int) -> int:
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    tmpl = ChatTemplate(model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))

    def ids_of(content: str) -> tuple[list[int], str]:
        text = tmpl.render([{"role": "user", "content": content}], tools=None, enable_thinking=True)
        return [int(t) for t in tok.encode(text, add_special_tokens=False).ids], text

    rows = []
    for name, content in PROMPTS:
        ids, text = ids_of(content)
        rows.append({"name": name, "ids": ids, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    if long_tokens > 0:
        n = max(16, long_tokens // 20)
        while True:
            ids, text = ids_of(long_prompt(n))
            if len(ids) >= long_tokens or n > 100000:
                break
            n = int(n * long_tokens / max(len(ids), 1)) + 8
        rows.append({"name": f"long{len(ids)}", "ids": ids, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    out.write_text(json.dumps({"generator": "tools/glm53/reference.py --make-prompts", "thinking": True,
                               "model": str(model), "prompts": rows}) + "\n")
    print(f"[ref] prompts -> {out}: " + ", ".join(f"{r['name']} {len(r['ids'])} ids" for r in rows), flush=True)
    return 0


class GatherComm:
    """The engine's NCCL communicator showing only all_gather and barrier: fused.gather's all-gather + rank-order-sum
    branch for every window (no all_reduce / all_to_all to pick another), as the Zig forward does in Phase 2a."""

    def __init__(self, base) -> None:
        self.base, self.rank, self.world = base, base.rank, base.world

    def all_gather(self, send, recv) -> None:
        self.base.all_gather(send, recv)

    def barrier(self) -> None:
        self.base.barrier()


def nccl_path(lib) -> str:
    """The file the engine's libnccl was loaded from (dladdr of one of its symbols): the Zig rank dlopens the same."""
    import ctypes

    class DlInfo(ctypes.Structure):
        _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p), ("dli_sname", ctypes.c_char_p),
                    ("dli_saddr", ctypes.c_void_p)]

    try:
        libc = ctypes.CDLL(None)
        info = DlInfo()
        addr = ctypes.cast(lib.ncclGetVersion, ctypes.c_void_p)
        if libc.dladdr(addr, ctypes.byref(info)) and info.dli_fname:
            return os.path.realpath(info.dli_fname.decode())
    except Exception:  # noqa: BLE001
        pass
    return getattr(lib, "_name", "libnccl.so.2")


def nccl_version(lib) -> int:
    import ctypes

    v = ctypes.c_int(0)
    lib.ncclGetVersion(ctypes.byref(v))
    return int(v.value)


def pack(tree: Path, out: Path, cache: Path) -> None:
    """oracle.pack without its Phase 1 kernel list: manifest.json, then aot.json + cubins/ in OUT/aot."""
    py = sys.executable
    subprocess.run([py, "-B", str(tree / "tools" / "zig" / "triton_aot_manifest.py"), "--cache", str(cache),
                    "--launches", str(out / "launches.json"), "--out", str(out / "manifest.json")], check=True)
    subprocess.run([py, "-B", str(tree / "zig" / "tests" / "cuda" / "nemotron" / "aot_pack.py"), "--manifest",
                    str(out / "manifest.json"), "--cache", str(cache), "--jit", str(out / "jit.json"), "--out",
                    str(out / "aot")], check=True)
    aot = json.loads((out / "aot" / "aot.json").read_text())
    names = sorted({k["fn"] for k in aot["kernels"]})
    print(f"[ref] aot: {len(aot['kernels'])} variants of {len(names)} functions: {', '.join(names)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--make-prompts", default="", help="write the prompts file here and exit (CPU)")
    ap.add_argument("--long", type=int, default=LONG_TOKENS, help="the long prompt's length (0: none)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--master", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29571)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--layers", type=int, default=0, help="first N layers only (debug; 0: all)")
    ap.add_argument("--out", default="")
    ap.add_argument("--record", action="store_true", help="record the Triton launches; pack OUT/aot")
    ap.add_argument("--tools", default="", help="TensorFold tree (default: this file's)")
    a = ap.parse_args()
    model = Path(a.model)
    if a.make_prompts:
        return make_prompts(model, Path(a.make_prompts), a.long)
    if not (a.prompts and a.out):
        ap.error("--prompts and --out are needed for a run")

    from oracle import FROZEN, freeze_env, specialization

    env = freeze_env()
    os.environ["TF_GLM53_RADIX_MIN_ROWS"] = env["TF_GLM53_RADIX_MIN_ROWS"] = "1"
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

    import torch

    from tensorfold.cuda.comm import NCCL
    from tensorfold.families.glm_moe_dsa.config import Config
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_layers, trim_host

    assert fused.RADIX_MIN_ROWS == 1 and not fused.TUNE and not fused.SIDE, "the frozen knobs did not take"
    torch.backends.cuda.matmul.allow_tf32 = False
    prompts = json.loads(Path(a.prompts).read_text())["prompts"]
    torch.cuda.set_device(0)
    t0 = time.time()
    nccl = NCCL(a.rank, a.world, a.master, a.port)
    comm = GatherComm(nccl)
    lib_path, lib_ver = nccl_path(nccl.lib), nccl_version(nccl.lib)
    print(f"[ref] rank {a.rank}: NCCL {lib_ver} from {lib_path}", flush=True)
    (out / "nccl.txt").write_text(f"{lib_path}\n{lib_ver}\n")

    cfg = Config.from_dict(json.loads((model / "config.json").read_text()))
    n_layers = a.layers or cfg.num_hidden_layers
    with torch.no_grad():
        r = RankReader(model, a.rank, a.world)
        embed, norm, head = (r.get(t, "cuda") for t in ("model.embed_tokens.weight", "model.norm.weight",
                                                         "lm_head.weight"))
        r.release()
        layers = load_layers(r, cfg, n_layers, verbose=a.rank == 0)
        r.drop_page_cache()
        del r
        trim_host()
        w = fused.Weights(cfg, a.rank, a.world, comm, embed, norm, head, layers, None)
        if w.dcp != 1 or w.fast is not None or w.l2pf is not None:
            raise SystemExit("reference: DCP 1, no RoCE one-shot and no L2 prefetch expected")
        load_s = time.time() - t0
        print(f"[ref] rank {a.rank}: {n_layers} layers + head loaded in {load_s:.0f}s; vocabulary rows "
              f"{w.vocab_off}..{w.vocab_off + w.vocab_part}", flush=True)
        (out / "inv_freq.bin").write_bytes(w.inv.contiguous().cpu().numpy().tobytes())
        nccl.barrier()

        topk = cfg.index_topk
        score_cols = fused.bucket(a.context, topk) or max(a.context, 64)
        b = fused.Buffers(w, a.window, score_cols)
        results = []
        for p in prompts:
            ids = [int(x) for x in p["ids"]]
            P = len(ids)
            if P + a.tokens > a.context:
                raise SystemExit(f"{p['name']}: {P} prompt tokens + {a.tokens} picks exceed the context {a.context}")
            st = fused.State(w, a.context)                # torch.zeros: a fresh request
            toks, recs = [], []

            def keep() -> None:
                torch.cuda.synchronize()
                toks.append(int(b.argmax[0].item()))
                g = b.amax_all[:a.world * 4].contiguous().view(torch.int32).cpu().tolist()
                recs.append(",".join("%08x" % (v & 0xFFFFFFFF) for v in g))

            ids_t = torch.tensor(ids, dtype=torch.long)
            t1 = time.time()
            s0 = 0
            while s0 < P:
                e = min(s0 + a.window, P)
                R = e - s0
                b.ids[:R].copy_(ids_t[s0:e].cuda())
                st.pos.fill_(s0)
                fused.compute(w, st, b, R, fused.bucket(e, topk), logits="last" if e == P else "none",
                              pick="argmax")
                s0 = e
            keep()
            prefill_s = time.time() - t1
            t2 = time.time()
            pos = P
            while len(toks) < a.tokens:
                b.ids[:1].fill_(toks[-1])
                st.pos.fill_(pos)
                fused.compute(w, st, b, 1, fused.bucket(pos + 1, topk), logits="all", pick="argmax")
                keep()
                pos += 1
            decode_s = time.time() - t2
            print(f"[ref] rank {a.rank} {p['name']}: prompt {P} in {prefill_s:.1f}s, {a.tokens} picks in "
                  f"{decode_s:.1f}s; first picks {toks[:8]}", flush=True)
            results.append({"name": p["name"], "prompt_len": P, "prefill_s": round(prefill_s, 3),
                            "decode_s": round(decode_s, 3), "tokens": toks, "records": recs})
            del st
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        nccl.barrier()

    doc = {"tool": "tools/glm53/reference.py", "rank": a.rank, "world": a.world, "layers": n_layers,
           "context": a.context, "window": a.window, "tokens": a.tokens, "vocab_off": w.vocab_off,
           "vocab_part": w.vocab_part, "nccl_version": lib_ver, "nccl_lib": lib_path, "env": env,
           "frozen_base": FROZEN, "torch": torch.__version__, "load_s": round(load_s, 1),
           "inv_freq_sha256": hashlib.sha256((out / "inv_freq.bin").read_bytes()).hexdigest(), "prompts": results}
    try:
        import triton

        doc["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    (out / f"ref-r{a.rank}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[ref] rank {a.rank}: wrote {out / f'ref-r{a.rank}.json'}", flush=True)
    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
        pack(tree, out, Path(os.environ["TRITON_CACHE_DIR"]))
    print("[ref] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
