#!/usr/bin/env python3
"""GLM-5.3's Python CUDA engine (families/glm_moe_dsa, the fused path) as the Zig port's oracle, Phase 1.

One GB10, rank 0 of world 4 (the serving shapes of one rank: 16 heads, a quarter of each MLP / expert width; the
replicated q_a / kv_a / indexer), with an IDENTITY communicator (``LocalComm``): no other rank exists, so every
exchange carries this rank's partial only. fused.gather takes its all-gather branch (the comm has no all_reduce /
all_to_all), and LocalComm.all_gather writes this rank's slot and ZEROS in the other ranks' slots; glue.residual_add
then sums [own, 0, 0, 0] in rank order (WORLD 4). The per-layer outputs are therefore rank 0's alone, not the full
model's: a Phase 1 oracle of one rank's arithmetic, which the Zig side reproduces with the same zero slots. The first N
layers (default 4: dense 0-2, MoE 3). Two chains of windows over fixed random tokens, the window shapes the Zig
gate runs:

* ``short``: a 24-row prompt window from an empty state (prompt tiling: ATTN_PROMPT, one pass), then a 1-row and a
  3-row decode window (decode tiling, 8 key chunks + merge). All rows below index_topk: no selection.
* ``long``: a 20,000-token prefix in 128-row windows (row-invariant kernels throughout), whose caches are saved as the
  Zig side's starting state; then a 1-row and a 3-row decode window at 20,000.. (index bucket 32,768: the indexer
  scores and selects, the attention reads the selected keys). The decode windows run twice: the champion's
  selection (torch.topk + sort for decode-sized windows) and the radix select the Zig side uses
  (``TF_GLM53_RADIX_MIN_ROWS=1``); the two must give the same bits (checked here, recorded in meta.json).

``--record``: tools/zig/triton_aot_manifest.py's Recorder logs every Triton specialization and EXL3 extension call;
at the end the manifest and aot_pack.py turn the Triton cache into aot.json + cubins/ (the kernel set the Zig engine
loads). ``--fixtures``: per window, each layer's output rows, cache rows written, selections and end-of-layer
scratch (diagnostics) as safetensors, plus meta.json (frozen env, tiles, versions, weight digests).

Must run inside the GLM-5.3 TP4 image (torch, Triton 3.7, the EXL3 extensions build on first use) with
PYTHONPATH at the champion source's ``src`` and TRITON_CACHE_DIR set to a fresh directory (for ``--record``).
Usage: see tools/glm53/README.md.
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

# The champion's knobs as this oracle freezes them (read at import by fused.py / exl3): every value the Zig port
# implements. TUNE=0: x3linear.plan's tiles (timed tiles differ per boot); SIDE=0: one stream (the side stream gives the
# same bits); EXPERTS=tf: TensorFold's own expert layout (decode windows run tfx.routed either way).
FROZEN = {
    "TF_GLM53_TUNE": "0",
    "TF_GLM53_TILES": "",
    "TF_GLM53_SIDE": "0",
    "TF_GLM53_EXPERTS": "tf",
    "TF_GLM53_RADIX": "1",
    "TF_GLM53_RADIX_MIN_ROWS": "32",
    "TF_GLM53_ATTN_PROMPT": "2048,64,8,2",
    "TF_GLM53_FOLD_SHARED": "1",
    "TF_GLM53_INDEX_SPLIT": "1",
    "TF_GLM53_UNPACK_CACHE_MB": "0",
    "TF_GLM53_L2PF": "0",
    "TF_EXL3_LINEAR_LOADS": "35",
    "TF_EXL3_EXPERTS_LOADS": "1",
    "TF_EXL3_EXPERTS_FUSE": "1",
    "TF_EXL3_EXPERTS_PDL": "0",
    "TF_EXL3_PROMPT_EXPERTS": "1",
}

SHORT_PROMPT = 24
LONG_PREFIX = 20000
LONG_CHUNK = 128
SHORT_SEED, LONG_SEED = 5, 11
# the end-of-layer scratch dumped per window and layer (diagnostics: the Zig side compares what it has)
DIAG = ("normed", "qa", "qn", "kva", "q", "qlat", "qrot", "ol", "o", "ik", "iw", "iq", "mlog", "pick", "wts", "sy",
        "part")
TRITON_KERNELS = ("_rmsnorm", "_router_part", "_router_sum", "_topk", "_residual_add", "_kv_write", "_ik_write",
                  "_iq_rope", "_index_scores", "_select_keys", "_absorb", "_attn_chunks", "_merge", "_expand",
                  "_swiglu2")


def freeze_env() -> dict:
    changed = {}
    for k, v in FROZEN.items():
        if os.environ.get(k, v) != v:
            changed[k] = os.environ[k]
        os.environ[k] = v
    if changed:
        print(f"[oracle] overriding caller env for the frozen set: {changed}", flush=True)
    return dict(FROZEN)


def tokens(n: int, seed: int):
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 150000, (n,), generator=g)


def sha(t) -> str:
    import torch

    raw = t.detach().contiguous()
    if raw.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(raw.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class LocalComm:
    """One rank of ``world`` with the others absent (an identity communicator). Deliberately no all_reduce and no
    all_to_all, so fused.gather always takes the all-gather + rank-order-sum branch; all_gather puts this rank's
    words in its own slot and zeros in every other slot (the absent ranks contribute nothing). The Zig port's
    forward does exactly the same (zeroed [world, R, D] buffer, own slot copied, residual_add WORLD = world)."""

    def __init__(self, rank: int, world: int) -> None:
        self.rank, self.world = rank, world

    def all_gather(self, send, recv):
        n = send.numel()
        if recv.numel() != n * self.world:
            raise ValueError("all_gather: recv must hold world x send")
        flat = recv.view(-1)
        flat.zero_()
        flat[self.rank * n:(self.rank + 1) * n].copy_(send.reshape(-1))

    def barrier(self):
        pass


class Taps:
    """fused.layer wrapped: after each layer of a dumped window, its output rows, cache rows, selection, scratch."""

    def __init__(self) -> None:
        self.prefix: str | None = None
        self.out: dict = {}

    def install(self) -> None:
        from tensorfold.families.glm_moe_dsa.cuda import fused

        orig = fused.layer
        taps = self

        def traced(w, L, b, x, R, cache, icache, pos, T, base=None, reuse=0):
            orig(w, L, b, x, R, cache, icache, pos, T, base, reuse)
            if taps.prefix is not None:
                taps.layer(w, L, b, x, R, cache, icache, pos, T)

        fused.layer = traced

    def layer(self, w, L, b, x, R, cache, icache, pos, T) -> None:
        import torch

        torch.cuda.synchronize()
        i, p = L.index, self.prefix
        p0 = int(pos.item())
        o = self.out
        o[f"{p}.x.{i}"] = x[:R].clone()
        o[f"{p}.kc.{i}"] = cache[p0:p0 + R].clone()
        if icache is not None:
            o[f"{p}.ic.{i}"] = icache[p0:p0 + R].clone()
        if L.indexer is not None and T is not None:
            o[f"{p}.tok.{i}"] = b.tok[:R].clone()
        for name in DIAG:
            t = getattr(b, name, None)
            if t is None:
                continue
            if name in ("ik", "iw", "iq") and L.indexer is None:
                continue
            if name in ("mlog", "pick", "wts", "sy") and L.experts is None:
                continue
            if name in ("iw", "iq") and T is None:
                continue
            o[f"{p}.diag.{name}.{i}"] = t[:R].clone()
        width = L.shared["gate"].n
        for name in ("g", "u", "act"):
            o[f"{p}.diag.{name}.{i}"] = getattr(b, name).view(-1)[:R * width].clone()


def load_engine(model: Path, n_layers: int, rank: int, world: int):
    """fused.Weights for layers 0..n-1 of rank ``rank`` of ``world`` (RankReader's cut, as the engine loads)."""
    import torch

    from tensorfold.families.glm_moe_dsa.config import Config
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.weights import RankReader, load_layer

    cfg = Config.from_dict(json.loads((model / "config.json").read_text()))
    r = RankReader(model, rank, world)
    embed = r.get("model.embed_tokens.weight", "cuda")
    norm = r.get("model.norm.weight", "cuda")
    head = torch.zeros((128, cfg.hidden_size), dtype=torch.bfloat16, device="cuda")   # logits="none": never read
    layers = []
    for i in range(n_layers):
        layers.append(load_layer(r, cfg, i))
        r.release()
    w = fused.Weights(cfg, rank, world, LocalComm(rank, world), embed, norm, head, layers, None)
    if w.dcp != 1 or w.fast is not None or w.l2pf is not None:
        raise SystemExit("oracle: DCP 1, no RoCE one-shot and no L2 prefetch expected")
    return cfg, w


def tiles_table(w) -> list:
    from tensorfold.cuda.exl3 import linear as x3linear
    from tensorfold.families.glm_moe_dsa.cuda import fused

    out = []
    for L in w.layers:
        names = {"q_a": L.q_a, "kv_a": L.kv_a, "q_b": L.q_b, "o_proj": L.o_proj,
                 **{f"mlp.{k}": v for k, v in L.shared.items()}}
        if L.indexer is not None:
            names["wq_b"] = L.indexer["wq_b"]
        for n, lin in names.items():
            out.append({"layer": L.index, "name": n, "k": lin.k, "n": lin.n, "k2": lin.k2, "codebook": lin.codebook,
                        "layout": lin.layout, "split": list(lin.split), "plan": list(x3linear.plan(lin.k, lin.n))})
        ident = {id(v): k for k, v in names.items()}
        out.append({"layer": L.index, "groups": [[ident.get(id(g), "?") for g in grp] for grp in fused.groups(L)],
                    "groupable": [bool(x3linear.groupable(grp)) for grp in fused.groups(L)]})
    return out


def digests(w) -> dict:
    """sha256 of the device tensors the fused kernels read, by the names the Zig loader uses for them."""
    out = {"inv": sha(w.inv)}
    for L in w.layers:
        p = f"layers.{L.index}"
        out[f"{p}.input_norm"] = sha(L.input_norm)
        out[f"{p}.post_attn_norm"] = sha(L.post_attn_norm)
        out[f"{p}.q_a_norm"] = sha(L.q_a_norm)
        out[f"{p}.kv_a_norm"] = sha(L.kv_a_norm)
        out[f"{p}.wk"] = sha(L.extra["wk"])
        out[f"{p}.wv"] = sha(L.extra["wv"])
        lins = {"q_a": L.q_a, "kv_a": L.kv_a, "q_b": L.q_b, "o_proj": L.o_proj,
                **{f"mlp.{k}": v for k, v in L.shared.items()}}
        if L.indexer is not None:
            lins["wq_b"] = L.indexer["wq_b"]
            out[f"{p}.idx.wk"] = sha(L.indexer["wk"])
            out[f"{p}.idx.weights_proj"] = sha(L.indexer["weights_proj"])
            out[f"{p}.idx.k_norm_w"] = sha(L.extra["ik_w"])
            out[f"{p}.idx.k_norm_b"] = sha(L.extra["ik_b"])
        for n, lin in lins.items():
            out[f"{p}.{n}.words"] = sha(lin.words)
            out[f"{p}.{n}.suh"] = sha(lin.suh)
            out[f"{p}.{n}.svh"] = sha(lin.svh)
        if L.router is not None:
            out[f"{p}.router.weight"] = sha(L.router[0])
            out[f"{p}.router.bias"] = sha(L.extra["bias"])
        ex = L.experts
        if ex is not None:
            for n in ("gate_k2", "up_k2", "down_k2", "suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d"):
                out[f"{p}.experts.{n}"] = sha(getattr(ex, n))
            h = hashlib.sha256()
            for t in ex.keep:              # gate trellises in expert order, then up, then down (prepare's order)
                h.update(t.contiguous().view(__import__("torch").uint8).cpu().numpy().tobytes())
            out[f"{p}.experts.trellis"] = h.hexdigest()
            out[f"{p}.experts.k2_gu"] = list(ex.k2_gu)
            out[f"{p}.experts.k2_d"] = list(ex.k2_d)
    return out


def run_window(w, st, b, toks, a: int, e: int, T):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    R = e - a
    b.ids[:R].copy_(toks[a:e].cuda())
    st.pos.fill_(a)
    fused.compute(w, st, b, R, T, logits="none")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--world", type=int, default=4, help="the TP world whose cut is loaded (serving: 4)")
    ap.add_argument("--rank", type=int, default=0, help="the rank whose cut is loaded; the others are absent")
    ap.add_argument("--long", type=int, default=LONG_PREFIX, help="the long chain's prefix length (0: skip it)")
    ap.add_argument("--record", action="store_true", help="record launches; pack aot.json + cubins into OUT/aot")
    ap.add_argument("--fixtures", action="store_true", help="write OUT/fixtures/{meta.json,*.safetensors}")
    ap.add_argument("--tools", default="", help="TensorFold tree holding tools/zig/triton_aot_manifest.py and "
                                                "zig/tests/cuda/nemotron/aot_pack.py (default: this file's tree)")
    a = ap.parse_args()
    if not (a.record or a.fixtures):
        ap.error("nothing to do: --record and/or --fixtures")
    env = freeze_env()
    tree = Path(a.tools) if a.tools else Path(__file__).resolve().parents[2]
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
    from contextlib import nullcontext

    from tensorfold.families.glm_moe_dsa.cuda import fused

    def phase(name: str, detail: bool = False):
        return rec.scope(name, detail) if rec is not None else nullcontext()

    torch.backends.cuda.matmul.allow_tf32 = False
    model = Path(a.model)
    t0 = time.time()
    with torch.no_grad():
        with phase("load"):
            cfg, w = load_engine(model, a.layers, a.rank, a.world)
        if rec is not None:
            from tensorfold.cuda.exl3 import experts as x3experts
            from tensorfold.cuda.exl3 import linear as x3linear

            rec.wrap(x3linear._ext(), ("rot_in", "linear"), "exl3_linear")
            rec.wrap(x3experts._ext(), ("group", "rot_in", "grouped", "gateup_epilogue", "down_epilogue",
                                        "down_combine", "combine"), "exl3_experts")
        print(f"[oracle] {a.layers} layers loaded in {time.time() - t0:.1f}s", flush=True)
        taps = Taps()
        taps.install()
        topk = cfg.index_topk
        meta: dict = {"generator": "tools/glm53/oracle.py", "layers": a.layers, "world": a.world, "rank": a.rank,
                      "comm": "local: own slot, zeros for the absent ranks (all-gather branch only)", "env": env,
                      "model": str(model), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                      "diag": list(DIAG) + ["g", "u", "act"], "chains": {}}
        try:
            import triton

            meta["triton"] = triton.__version__
        except Exception:  # noqa: BLE001
            pass
        meta["tiles"] = tiles_table(w)
        meta["tiles_are_plan"] = all(t["split"] == t["plan"] for t in meta["tiles"] if "split" in t)
        if not meta["tiles_are_plan"]:          # the Zig side runs x3linear.plan's tiles
            print("[oracle] WARNING: some linears do not run x3linear.plan's tiles (TF_GLM53_TUNE / TILES?)", flush=True)
        common = {"inv": w.inv.clone()}

        # -- short chain: prompt 24 rows, then decode 1 and 3 rows -----------------------------------------------
        n_short = SHORT_PROMPT + 4
        toks = tokens(n_short, SHORT_SEED)
        st = fused.State(w, 64)
        b = fused.Buffers(w, 32, 64)
        windows = [("prompt24", 0, SHORT_PROMPT), ("decode1", SHORT_PROMPT, SHORT_PROMPT + 1),
                   ("decode3", SHORT_PROMPT + 1, SHORT_PROMPT + 4)]
        short: dict = {"ids": toks.clone()}
        taps.out = short
        rows = []
        for name, s, e in windows:
            T = fused.bucket(e, topk)
            taps.prefix = name
            with phase(f"short-{name}", detail=True):
                run_window(w, st, b, toks, s, e, T)
            taps.prefix = None
            short[f"{name}.hidden"] = b.hidden[:e - s].clone()
            rows.append({"name": name, "a": s, "e": e, "T": T or 0})
        meta["chains"]["short"] = {"file": "short.safetensors", "capacity": 64, "buffer_rows": 32, "score_cols": 64,
                                   "seed": SHORT_SEED, "windows": rows}
        del st, b

        # -- long chain: a prefix in 128-row windows, then decode windows past index_topk ------------------------
        long: dict = {}
        if a.long > 0:
            P = a.long
            toks = tokens(P + 4, LONG_SEED)
            long["ids"] = toks.clone()
            cap = P + 64
            st = fused.State(w, cap)
            big = fused.Buffers(w, LONG_CHUNK, P)
            t1 = time.time()
            with phase("long-prefix"):
                for s in range(0, P, LONG_CHUNK):
                    e = min(s + LONG_CHUNK, P)
                    run_window(w, st, big, toks, s, e, fused.bucket(e, topk) and e)
            torch.cuda.synchronize()
            print(f"[oracle] long prefix {P} tokens in {time.time() - t1:.1f}s", flush=True)
            del big
            for i, L in enumerate(w.layers):
                long[f"in.kc.{i}"] = st.kc[i][:P].clone()
                if L.indexer is not None:
                    long[f"in.ic.{L.index}"] = st.ic[L.index][:P].clone()
            small = fused.Buffers(w, 4, 32768)
            windows = [("decode1", P, P + 1), ("decode3", P + 1, P + 4)]
            rows = []
            taps.out = long
            for name, s, e in windows:
                T = fused.bucket(e, topk)
                taps.prefix = name
                with phase(f"long-{name}", detail=True):
                    run_window(w, st, small, toks, s, e, T)
                taps.prefix = None
                long[f"{name}.hidden"] = small.hidden[:e - s].clone()
                rows.append({"name": name, "a": s, "e": e, "T": T or 0})
            # the same windows through the radix select (the Zig side's selection for every window size)
            radix: dict = {}
            taps.out = radix
            keep = fused.RADIX_MIN_ROWS
            fused.RADIX_MIN_ROWS = 1
            try:
                for name, s, e in windows:
                    taps.prefix = name
                    with phase(f"long-radix-{name}", detail=True):
                        run_window(w, st, small, toks, s, e, fused.bucket(e, topk))
                    taps.prefix = None
            finally:
                fused.RADIX_MIN_ROWS = keep
            diff = [k for k in radix if k in long and not torch.equal(radix[k], long[k])]
            meta["radix_equal"] = not diff
            meta["radix_differs"] = diff[:20]
            if diff:
                print(f"[oracle] WARNING: radix and torch.topk selections give other bits: {diff[:5]}", flush=True)
            meta["chains"]["long"] = {"file": "long.safetensors", "capacity": cap, "prefix": P, "buffer_rows": 4,
                                      "score_cols": 32768, "seed": LONG_SEED, "windows": rows}
            del radix
        torch.cuda.synchronize()

    if a.fixtures:
        from safetensors.torch import save_file

        fx = out / "fixtures"
        fx.mkdir(parents=True, exist_ok=True)
        meta["weights_sha256"] = digests(w)
        save_file({k: v.contiguous() for k, v in short.items()}, str(fx / "short.safetensors"))
        if long:
            save_file({k: v.contiguous() for k, v in long.items()}, str(fx / "long.safetensors"))
        save_file(common, str(fx / "common.safetensors"))
        (fx / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        print(f"[oracle] fixtures -> {fx} ({len(short)} + {len(long)} tensors)", flush=True)

    if rec is not None:
        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
        pack(tree, out, Path(os.environ["TRITON_CACHE_DIR"]))
    print("[oracle] done", flush=True)
    return 0


def specialization() -> dict:
    """Each JIT function's parameter names and which ones Triton never specializes (as Nemotron's capture.py)."""
    import gc

    from triton.runtime.jit import JITFunction

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            name = f"{fn.fn.__module__}.{fn.fn.__qualname__}"
            out[name] = {"params": [p.name for p in fn.params],
                         "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                         "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


def pack(tree: Path, out: Path, cache: Path) -> None:
    """manifest.json (each launched specialization's cubin and ABI), then aot.json + cubins/ in OUT/aot."""
    py = sys.executable
    subprocess.run([py, "-B", str(tree / "tools" / "zig" / "triton_aot_manifest.py"), "--cache", str(cache),
                    "--launches", str(out / "launches.json"), "--out", str(out / "manifest.json")], check=True)
    subprocess.run([py, "-B", str(tree / "zig" / "tests" / "cuda" / "nemotron" / "aot_pack.py"), "--manifest",
                    str(out / "manifest.json"), "--cache", str(cache), "--jit", str(out / "jit.json"), "--out",
                    str(out / "aot")], check=True)
    aot = json.loads((out / "aot" / "aot.json").read_text())
    have = {k["fn"] for k in aot["kernels"]}
    missing = [k for k in TRITON_KERNELS if k not in have]
    lines = [f"{k['fn']:16s} warps {k['num_warps']} stages {k.get('num_stages', 0)} hash {k['hash'][:12]} "
             f"params {[p['name'] for p in k['params']]} consts "
             f"{ {n: (c.get('int', c.get('f32', 'none'))) for n, c in k['consts'].items()} }" for k in aot["kernels"]]
    (out / "aot" / "variants.txt").write_text("\n".join(lines) + "\n")
    print(f"[oracle] aot: {len(aot['kernels'])} variants of {len(have)} functions; missing {missing or 'none'}",
          flush=True)
    if missing:
        raise SystemExit(f"kernels the Zig layers need were never launched: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
