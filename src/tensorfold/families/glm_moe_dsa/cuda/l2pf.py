"""L2 prefetch of the next kernels' weights in GLM-5.3 decode windows (TF_GLM53_L2PF, default on; the same bits).

A decode window's layer alternates DRAM-bound matmuls (EXL3 projections, routed and shared experts) with phases that
leave DRAM idle: the two all-reduces a layer (the RoCE wait: ~13 ms a round over 156 of them at TP=4), the router and
top-k, and the attention core. At three sites a layer a small kernel on a side stream (``l2pf.cu``) asks for the
weights the main stream reads next to be brought into L2; the kernel that follows then finds part of its bytes there.
The prefetch only reads, every model kernel keeps its inputs, outputs and stream order, so replies keep their bits.

Sites (``Prefetch.site``, keyed by (layer index, name)), fired from ``fused``:

- ``a``, after the attention block's partial (before its all-reduce): the post-attention norm, the router and its
  bias, then the shared expert's gate/up and down (a dense layer: its MLP). With TF_GLM53_FOLD_SHARED (default) the
  shared expert runs before the routed experts, so the routed experts' stream does not evict it first; with it off
  the shared expert is left out of the site.
- ``f``, after the FFN's partial (before its all-reduce): the next layer's input norm, q_a and kv_a (one launch: each
  gets its share of the budget), their norms, then the indexer's small projections. Nothing after the last layer
  (the head is 475 MB a rank).
- ``o``, after absorb (before the attention core): kv_b's value half (expand) and the output projection.

Each site takes up to TF_GLM53_L2PF_MB (default 8) MiB, in the order the main stream reads: within an EXL3 linear its
scales first, then its words - whole when they fit, else the head of every warp's k range (each program of a one-wave
launch starts on L2 hits; a tensor's leading bytes would only help its first programs). TF_GLM53_L2PF_SITES (default
``afo``) picks the sites; TF_GLM53_L2PF=1|bulk (one cp.async.bulk.prefetch.L2 a piece), ``lines``
(prefetch.global.L2::evict_last a line) or ``touch`` (ld.global.cg a line); 0 turns it off. Windows over
TF_GLM53_L2PF_ROWS (default 32, fused.DECODE_ROWS) rows, prompt chunks and the MTP / draft passes skip every site.

The side stream forks from the main stream at each site and joins it before ``fused.compute`` returns, so a captured
CUDA graph holds the prefetches as a side branch that rejoins before the graph's end.

Ported from MiaAI-Lab patch 0046 (l2pf.cu / l2pf.py), itself adapted from jayleaton/glm53-tensorfold-spark
patches/0460 (Apache-2.0, Copyright 2026 Jay Leaton). Changes: the sites and weight groups are this family's (MLA +
DSA layers, EXL3 linears and their warp-chunk heads, the routed/shared expert order of fused.experts_part); the
prefetcher hangs off the rank's Weights (ranks as threads stay apart) instead of a module global."""

from __future__ import annotations

import dataclasses
import os
from functools import lru_cache
from typing import Any

import torch

from tensorfold.cuda.exl3 import linear as x3linear

MODES = {"1": 0, "on": 0, "bulk": 0, "lines": 1, "touch": 2}
PIECE = 32 << 10           # bytes a table piece covers at most (a bulk prefetch's size, a warp's walk); TF_GLM53_L2PF_KB
ALIGN = 16                 # cp.async.bulk wants 16-byte addresses and sizes
MIN_HEAD = 512             # a warp chunk's head under this many bytes is not worth a piece


@dataclasses.dataclass
class Settings:
    mode: int = -1          # -1: off
    mb: float = 8.0
    sites: str = "afo"
    rows: int = 32
    blocks: int = 0         # 0: enough for the site's pieces (a thread a piece in bulk mode, a warp a piece else)
    threads: int = 128
    piece: int = PIECE

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        s = cls()
        v = (env.get("TF_GLM53_L2PF", "") or "1").strip().lower()
        if v not in ("0", "off", *MODES):
            raise ValueError(f"TF_GLM53_L2PF: 0, 1 (bulk), bulk, lines or touch, not {v!r}")
        s.mode = MODES.get(v, -1)
        try:
            s.mb = float(env.get("TF_GLM53_L2PF_MB", "") or 8.0)
        except ValueError:
            s.mb = -1.0
        if not 0 < s.mb <= 64:
            raise ValueError("TF_GLM53_L2PF_MB: above 0 and at most 64 (MiB a site)")
        s.sites = (env.get("TF_GLM53_L2PF_SITES", "") or "afo").strip().lower()
        if not s.sites or set(s.sites) - set("afo"):
            raise ValueError(f"TF_GLM53_L2PF_SITES: letters of 'afo', not {s.sites!r}")
        s.rows = int(env.get("TF_GLM53_L2PF_ROWS", "") or 32)
        s.blocks = int(env.get("TF_GLM53_L2PF_BLOCKS", "") or 0)
        s.threads = int(env.get("TF_GLM53_L2PF_THREADS", "") or 128)
        s.piece = int(env.get("TF_GLM53_L2PF_KB", "") or PIECE >> 10) << 10
        if s.blocks < 0 or s.threads < 32 or s.threads % 32 or s.threads > 1024:
            raise ValueError("TF_GLM53_L2PF_BLOCKS / _THREADS: 0 (auto) or more blocks of 32..1024 threads (whole warps)")
        if not 1 << 10 <= s.piece <= 1 << 20:
            raise ValueError("TF_GLM53_L2PF_KB: 1 .. 1024 KiB a piece")
        return s

    @property
    def on(self) -> bool:
        return self.mode >= 0


@lru_cache(maxsize=1)
def _ext():
    from pathlib import Path

    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm53_l2pf_v1", sources=[str(here / "l2pf.cpp"), str(here / "l2pf.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


# -- what a site covers: (address, bytes) spans in the order the main stream reads them --------------------------------
def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _span(t: torch.Tensor, left: int) -> tuple[int, int] | None:
    """The leading min(size, left) bytes of t, widened to 16-byte bounds."""
    a = t.data_ptr()
    lo = a - a % ALIGN
    hi = min(a + _nbytes(t), lo + left)
    hi += (-hi) % ALIGN
    return (lo, hi - lo) if hi > lo else None


def exl3_heads(lin, budget: int) -> list[tuple[int, int]]:
    """An EXL3 linear in the strips layout (words [N/128, K/16, 8, 4 * K2] int32): the first ``budget / chunks``
    bytes of every warp's contiguous k range (program (128 columns, split), warp w reads k steps
    [(split * WK + w) * per, + per) of its strip), so every program of the launch starts on L2 hits. [] when the
    heads would be under MIN_HEAD bytes or the layout is not strips."""
    if getattr(lin, "layout", None) != "strips" or lin.split is None:
        return []
    sk, wk = lin.split
    kt = lin.k // 16
    if kt % (sk * wk):
        return []
    step = 8 * 4 * lin.k2 * 4                    # bytes of a k step of one strip (8 tiles of 4 * K2 words)
    per = kt // (sk * wk) * step                 # a warp's chunk
    nb, chunks = lin.n // 128, sk * wk
    head = min(per, budget // (nb * chunks)) // ALIGN * ALIGN
    base = lin.words.data_ptr()
    if head < MIN_HEAD or base % ALIGN:
        return []
    strip = kt * step
    return [(base + b * strip + c * per, head) for b in range(nb) for c in range(chunks)]


def _is_exl3(x: Any) -> bool:
    return isinstance(x, x3linear.Exl3Linear)


def _exl3_small(lin) -> list[torch.Tensor]:
    return [t for t in (lin.suh, lin.svh, lin.bias) if t is not None and t.numel()]


def ranges(items: list, budget: int) -> list[tuple[int, int]]:
    """(address, bytes) spans of ``items`` - tensors, EXL3 linears, or lists of EXL3 linears that run as one launch -
    up to ``budget`` bytes, in order. An EXL3 linear: its scales, then its words whole when they fit, else the heads
    of every warp chunk (``exl3_heads``), else their leading bytes; a launch group splits what is left in proportion
    to its members' words."""
    out: list[tuple[int, int]] = []
    left = budget

    def take(t: torch.Tensor) -> None:
        nonlocal left
        sp = _span(t, left) if left >= ALIGN else None
        if sp is not None:
            out.append(sp)
            left -= sp[1]

    def words(lins: list) -> None:
        nonlocal left
        total = sum(_nbytes(lin.words) for lin in lins)
        if total <= left:
            for lin in lins:
                take(lin.words)
            return
        share = left
        for lin in lins:
            mine = share * _nbytes(lin.words) // max(total, 1)
            heads = exl3_heads(lin, mine)
            if heads:
                out.extend(heads)
                left -= sum(n for _, n in heads)
            else:
                sp = _span(lin.words, mine // ALIGN * ALIGN)
                if sp is not None and sp[1] <= left:
                    out.append(sp)
                    left -= sp[1]

    for it in items:
        if left < ALIGN:
            break
        if it is None:
            continue
        if isinstance(it, torch.Tensor):
            if it.numel():
                take(it)
        elif _is_exl3(it):
            for t in _exl3_small(it):
                take(t)
            words([it])
        elif isinstance(it, (list, tuple)) and it and all(_is_exl3(x) for x in it):
            for lin in it:
                for t in _exl3_small(lin):
                    take(t)
            words(list(it))
        else:
            raise TypeError(f"l2pf: cannot prefetch {type(it).__name__}")
    return out


def pieces(spans: list[tuple[int, int]], piece: int = PIECE) -> list[tuple[int, int]]:
    out = []
    for a, n in spans:
        for o in range(0, n, piece):
            out.append((a + o, min(piece, n - o)))
    return out


# -- the sites of this family ----------------------------------------------------------------------------------------
def site_a(L, fold_shared: bool) -> list:
    """At the attention all-reduce: what ffn_part reads first."""
    items: list = [L.post_attn_norm]
    if L.router is not None:
        items += [L.router[0], L.extra.get("bias")]
    if L.shared is not None and (L.experts is None or fold_shared):
        items += [[L.shared["gate"], L.shared["up"]], L.shared["down"]]
    return items


def site_f(nxt) -> list:
    """At the FFN all-reduce: what the next layer's attention_part reads first (nothing after the last layer)."""
    if nxt is None:
        return []
    items: list = [nxt.input_norm, [nxt.q_a, nxt.kv_a], nxt.q_a_norm, nxt.kv_a_norm]
    if nxt.indexer is not None:
        ix = nxt.indexer
        items += [ix["wk"], nxt.extra.get("ik_w"), nxt.extra.get("ik_b"), ix["weights_proj"]]
    return items


def site_o(L) -> list:
    """After absorb: expand's kv_b value half, then the output projection."""
    return [L.extra.get("wv"), L.o_proj]


class Prefetch:
    """The sites' pieces for one rank's weights (one device table; (first, count, grid) a site) and its side stream.
    Build it after the linears' tiles are final (tuning / share_tiles) and before any graph capture."""

    def __init__(self, w, settings: Settings | None = None, fold_shared: bool = True) -> None:
        self.s = settings or Settings.from_env()
        self.sites: dict[tuple[int, str], tuple[int, int, int]] = {}
        self.bytes: dict[tuple[int, str], int] = {}
        budget = int(self.s.mb * (1 << 20))
        rows: list[tuple[int, int]] = []
        layers = list(w.layers)
        for i, L in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            plan = {"a": site_a(L, fold_shared), "f": site_f(nxt), "o": site_o(L)}
            for name, items in plan.items():
                if name not in self.s.sites or not items:
                    continue
                spans = ranges(items, budget)
                p = pieces(spans, self.s.piece)
                if p:
                    self.sites[(L.index, name)] = (len(rows), len(p), self._grid(len(p)))
                    self.bytes[(L.index, name)] = sum(n for _, n in spans)
                    rows += p
        dev = w.device
        self.table = torch.tensor(rows or [(0, 0)], dtype=torch.int64, device=dev).view(-1, 2).contiguous()
        self.sink = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.side = torch.cuda.Stream(device=dev)
        _ext()                                           # built / loaded now, before any graph capture
        self.active = False                              # a decode window of fused.compute is running its sites
        self.forked = False

    def _grid(self, n: int) -> int:
        if self.s.blocks:
            return self.s.blocks
        per = self.s.threads if self.s.mode == 0 else self.s.threads // 32
        return max(1, min(48, -(-n // per)))

    def site(self, index: int, name: str) -> None:
        """Fork the side stream at this point of the main stream and prefetch the site's pieces there."""
        key = self.sites.get((index, name)) if self.active else None
        if key is None:
            return
        first, count, grid = key
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            _ext().prefetch(self.table, first, count, self.s.mode, grid, self.s.threads, self.sink)
        self.forked = True

    def join(self) -> None:
        """The main stream waits for the side stream (a captured graph's branch rejoins before its end)."""
        if self.forked:
            torch.cuda.current_stream().wait_stream(self.side)
            self.forked = False

    def summary(self) -> str:
        per: dict[str, list[int]] = {}
        for (_, name), n in self.bytes.items():
            per.setdefault(name, []).append(n)
        parts = [f"{k} {len(v)} sites {sum(v) / len(v) / (1 << 20):.1f} MiB" for k, v in sorted(per.items())]
        mode = {0: "bulk", 1: "lines", 2: "touch"}[self.s.mode]
        return f"L2 prefetch in decode windows ({mode}, up to {self.s.mb:g} MiB a site): " + ", ".join(parts)


def install(w, settings: Settings | None = None) -> Prefetch | None:
    """w.l2pf = a Prefetch when TF_GLM53_L2PF (or ``settings``) is on and CUDA is there, else None."""
    from . import fused

    s = settings or Settings.from_env()
    w.l2pf = Prefetch(w, s, fold_shared=fused.FOLD_SHARED) if s.on and torch.cuda.is_available() else None
    return w.l2pf
