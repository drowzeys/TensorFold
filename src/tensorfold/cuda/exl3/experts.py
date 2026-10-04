"""Routed EXL3 experts of any codebook, a width per expert, one grouped launch a projection; rows never depend on the window."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
ACT_BF16, ACT_F32 = 0, 1          # SwiGLU with the GLM family's bf16 roundings / in fp32
# Half-bits a value: 1..8 bits (2, 4, .. 16) and every half-integer rate 1.5..7.5 (3, 5, .. 15).
K2_SUPPORTED = tuple(range(2, 17))

# (n tiles a block, warps, K splits, tiles in flight): GLM's settings, whose arithmetic order this keeps bit for bit
GLM_GATEUP = (8, 4, 4, 1)
GLM_DOWN = (8, 4, 1, 1)


def _knob(name: str, default: int, lo: int, hi: int) -> int:
    v = (os.environ.get(name, "") or str(default)).strip()
    try:
        x = int(v)
    except ValueError:
        x = lo - 1
    if not lo <= x <= hi:
        raise ValueError(f"{name}: an integer {lo}..{hi}, not {v!r}")
    return x


# TF_EXL3_EXPERTS_LOADS: the trellis loads of the 8-tile / 4-warp settings (GLM_GATEUP, GLM_DOWN). 0: 32-bit loads,
# the setting's tiles in flight (the original walk); 1..4: 16-byte ld.global.nc loads 1..4 k steps ahead, staged in
# shared memory (MiaAI-Lab patch 0047: 1 measured best there). Data movement only: every value gives the same bits.
LOADS = _knob("TF_EXL3_EXPERTS_LOADS", 1, 0, 4)
# TF_EXL3_EXPERTS_FUSE (a bit mask; 8 n tiles a program, else that part stays separate): 1 (default): the down
# projection runs its epilogue and the weighted combine (``out``, + ``sy`` when given) itself, the last arriving block
# of a row summing its slots in slot order - no Z round trip, no down_combine launch; 2: gate/up's epilogue in the last
# of an expert's 2 * splits blocks (MiaAI-Lab measured it slower on its shapes: off by default, try 3). 0: the
# separate kernels. Every value gives the same bits (MiaAI-Lab patch 0016's decode kernel, adapted).
FUSE = _knob("TF_EXL3_EXPERTS_FUSE", 1, 0, 3)
# TF_EXL3_EXPERTS_PDL (default 0: the bench at 3-4 rows, the decode windows, ran 2-3 % slower with it): the chain's kernels launch with programmatic dependent launch (as linear.cu's);
# each waits for the previous kernel before touching memory. 0: plain launches. The same bits either way.
PDL = _knob("TF_EXL3_EXPERTS_PDL", 0, 0, 1)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    srcs = [str(here / f) for f in ("experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu")]
    return load(name="tensorfold_exl3_experts_v4", sources=srcs, extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False)


def codebook_id(name: str) -> int:
    """'3inst' / 'mcg' / 'mul1' (the checkpoint's quantization_config.codebook, or the marker tensor's name)."""

    ids = {"3inst": CB_3INST, "mcg": CB_MCG, "mul1": CB_MUL1}
    if name not in ids:
        raise ValueError(f"unknown EXL3 codebook {name!r}")
    return ids[name]


def k2_of(trellis: torch.Tensor) -> int:
    """Half-bits a value of a trellis int16 [K/16, N/16, 16 * K] (16 * K + 8 for the half-integer rates)."""

    w = trellis.shape[-1]
    if w % 8:
        raise ValueError(f"trellis last dim {w} is not a multiple of 8")
    k2 = w // 8
    if k2 not in K2_SUPPORTED:
        raise ValueError(f"unsupported EXL3 bit width {k2 / 2}")
    return k2


@dataclass
class Exl3RoutedExperts:
    """One layer's routed experts: trellis pointers and widths per projection, stacked suh/svh, the tensors keeping the trellises alive."""

    gate_ptr: torch.Tensor    # int64 [E]
    up_ptr: torch.Tensor
    down_ptr: torch.Tensor
    gate_k2: torch.Tensor     # int32 [E], half-bits a value
    up_k2: torch.Tensor
    down_k2: torch.Tensor
    suh_g: torch.Tensor       # fp16 [E, D]
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # fp16 [E, I]
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # fp16 [E, I]
    svh_d: torch.Tensor       # fp16 [E, D]
    count: int                # E
    dims: int                 # D (model width)
    width: int                # I (expert width on this rank)
    cb: int
    k2_gu: tuple[int, int]    # (min, max) K2 over gate and up
    k2_d: tuple[int, int]
    trellis_bytes: torch.Tensor   # int64 [E], gate + up + down trellis bytes of each expert (for GB/s)
    keep: list = field(default_factory=list, repr=False)

    def nbytes_read(self, ids: Sequence[int]) -> int:
        return int(self.trellis_bytes[list(ids)].sum())

    gu_stride: int = 0            # gate/up trellis row stride in tiles (0: contiguous [D/16, I/16, *]); a larger
                                  # stride reads each as a column block of one fused [D/16, 2I/16, *] trellis
    aligned16: bool = True        # every trellis starts on 16 bytes (the 16-byte load path; else 32-bit loads)

def prepare(gate: Sequence[tuple], up: Sequence[tuple], down: Sequence[tuple], codebook: int | str,
            device="cuda", gu_stride: int = 0) -> Exl3RoutedExperts:
    """A layer from per-expert (trellis, suh, svh) triples, trellises referenced in place, each at its own width."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    E = len(gate)
    if not (len(up) == len(down) == E) or E == 0:
        raise ValueError("gate, up and down need the same, non-zero number of experts")
    D, I = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
    keep = []

    def table(mats, k, n, stride=0):
        ptrs, k2s = [], []
        for t, _, _ in mats:
            if stride:                   # a column block of a wider [K/16, stride, 16 * bits] trellis
                ok = (t.dtype == torch.int16 and t.device.type == "cuda" and t.stride(2) == 1
                      and t.stride(1) == t.shape[2] and t.stride(0) == stride * t.shape[2])
            else:
                ok = t.dtype == torch.int16 and t.is_contiguous() and t.device.type == "cuda"
            if not ok:
                raise ValueError("trellis must be a contiguous CUDA int16 tensor (or a row-strided block)")
            if t.shape[0] * 16 != k or t.shape[1] * 16 != n:
                raise ValueError(f"expert shape {tuple(t.shape)} does not match [{k // 16}, {n // 16}, *]")
            k2s.append(k2_of(t))
            ptrs.append(t.data_ptr())
            keep.append(t)
        return (torch.tensor(ptrs, dtype=torch.int64, device=device),
                torch.tensor(k2s, dtype=torch.int32, device=device), k2s)

    gp, gk, gks = table(gate, D, I, gu_stride)
    upp, uk, uks = table(up, D, I, gu_stride)
    dp, dk, dks = table(down, I, D)

    def stack(mats, j, n):
        out = torch.empty((E, n), dtype=torch.float16, device=device)
        for e, m in enumerate(mats):
            out[e].copy_(m[j].reshape(-1))
        return out

    tb = torch.tensor([(D * I // 256) * (gks[e] + uks[e] + dks[e]) * 16 for e in range(E)], dtype=torch.int64)
    aligned = all(t.data_ptr() % 16 == 0 for t in keep)
    return Exl3RoutedExperts(gp, upp, dp, gk, uk, dk, stack(gate, 1, D), stack(up, 1, D), stack(gate, 2, I),
                             stack(up, 2, I), stack(down, 1, I), stack(down, 2, D), E, D, I, cb,
                             (min(gks + uks), max(gks + uks)), (min(dks), max(dks)), tb, keep, gu_stride, aligned)


def prepare_stacked(gt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d,
                    codebook: int | str) -> Exl3RoutedExperts:
    """A uniform-width layer stacked per projection: trellis [E, K/16, N/16, 16K] (int16, or GLM's int32 words), suh/svh [E, n]."""

    def as16(t):
        return t.view(torch.int16) if t.dtype == torch.int32 else t

    gt, ut, dt = as16(gt), as16(ut), as16(dt)
    E = gt.shape[0]
    return prepare([(gt[e], suh_g[e], svh_g[e]) for e in range(E)], [(ut[e], suh_u[e], svh_u[e]) for e in range(E)],
                   [(dt[e], suh_d[e], svh_d[e]) for e in range(E)], codebook, device=gt.device)


def default_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """The tile setting for a K -> N projection (GLM's where it divides): the shape's alone, so rows stay independent."""

    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


class Scratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots; slots whose pick is not a routed expert are left to the caller."""

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, cfg_gu=None, cfg_d=None, device="cuda") -> None:
        D, I = ex.dims, ex.width
        self.cfg_gu = cfg_gu or default_config(D, I, True)
        self.cfg_d = cfg_d or default_config(I, D, False)
        P = rows * slots
        self.xg = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        # gate and up write 2 * splits * P * I partials, down splits * P * D
        self.z = torch.zeros((max(2 * self.cfg_gu[2] * I, self.cfg_d[2] * D) * P,), dtype=torch.float32, device=device)
        self.y = torch.zeros((P, D), dtype=torch.float32, device=device)
        maxu = min(P, ex.count)
        self.ids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.members_buf = torch.full((maxu * rows,), -1, dtype=torch.int32, device=device)
        self.rows, self.slots, self.count_experts = rows, slots, ex.count
        # fused epilogues' arrival counts (zero between launches; each launch's last arrival resets its own):
        # gate/up a (expert, member tile, 128 columns), the combine a (row, 128 columns)
        self.done_gu = torch.zeros((maxu * (-(-rows // 16)) * max(I // 128, 1),), dtype=torch.int32, device=device)
        self.done_d = torch.zeros((rows * max(D // 128, 1),), dtype=torch.int32, device=device)
        self.none_h = torch.empty((0,), dtype=torch.float16, device=device)
        self.none_f = torch.empty((0,), dtype=torch.float32, device=device)
        self.none_i = torch.empty((0,), dtype=torch.int32, device=device)

    def window(self, R: int):
        """(ids, members) sized for R rows: the grids only span what R rows can use."""

        maxu = min(R * self.slots, self.count_experts)
        return self.ids[:maxu], self.members_buf[:maxu * R].view(maxu, R)


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor | None, ex: Exl3RoutedExperts, s: Scratch,
           out: torch.Tensor | None, R: int, limit: float = math.inf, act_mode: int = ACT_F32,
           group: bool = True, loads: int | None = None, fuse: int | None = None, pdl: int | None = None,
           sy: torch.Tensor | None = None, sy_ready=None) -> torch.Tensor:
    """Routed experts of R rows (picks >= E skipped): Y per slot, or ``out`` = the wts-weighted sum when ``wts``; no host sync.
    ``sy`` (fp32 [R, D], with ``wts``): out = (the weighted sum) + sy, the bits of ``out.add_(sy)`` after it.
    ``sy_ready``: called right before the first launch that reads ``sy`` (a caller whose sy comes from another
    stream joins it there).
    ``loads`` / ``fuse`` / ``pdl``: TF_EXL3_EXPERTS_LOADS / _FUSE / _PDL for this call (None: the module's); none
    changes a bit."""

    ext = _ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    ids, members = s.window(R)
    vec = (LOADS if loads is None else int(loads)) if ex.aligned16 else 0
    fuse = FUSE if fuse is None else int(fuse)
    pdl = PDL if pdl is None else int(pdl)
    nh, nf, ni = s.none_h, s.none_f, s.none_i
    if group:
        ext.group(pick, ids, s.count, members, R, slots, E, pdl)
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E, pdl)
    nt, w, sk, pf = s.cfg_gu
    fgu = 2 if fuse & 2 and nt == 8 else 0
    ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1], ex.gu_stride, vec, fgu, pick, E,
                ex.svh_g, ex.svh_u, ex.suh_d, s.xd, float(limit), act_mode, nh, nf, nf, nf, nf,
                s.done_gu if fgu else ni, pdl)
    if not fgu:
        ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, sk, slots, E, float(limit),
                            act_mode, pdl)
    nt, w, sk, pf = s.cfg_d
    if wts is not None and out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    if fuse & 1 and nt == 8 and sk == 1:
        # down with its epilogue (y) and, given wts, the combine (+ sy) in the same launch
        if sy_ready is not None:
            sy_ready()
        ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                    D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1], 0, vec, 1, pick, E, nh, nh, nh, nh,
                    0.0, act_mode, ex.svh_d, s.y, nf if wts is None else wts, nf if out is None else out,
                    nf if (sy is None or wts is None) else sy, s.done_d, pdl)
        return s.y[:P] if wts is None else out
    ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1], 0, vec, 0, pick, E, nh, nh, nh, nh, 0.0,
                act_mode, nh, nf, nf, nf, nf, ni, pdl)
    if wts is None:
        ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, sk, slots, E, pdl)
        return s.y[:P]
    # the down epilogue and the combine in one launch (the same arithmetic in the same order as the two)
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, sk, slots, E, pdl)
    if sy_ready is not None:
        sy_ready()
    if sy is not None:
        out.add_(sy)
    return out


def dequant(trellis: torch.Tensor, codebook: int | str) -> torch.Tensor:
    """W_q [K, N] fp16 of one matrix through the kernels' own lane decode (ExLlamaV3's ``reconstruct``); for tests."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    k2 = k2_of(trellis)
    K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((K, N), dtype=torch.float16, device=trellis.device)
    _ext().dequant(trellis.contiguous(), out, k2, cb)
    return out
