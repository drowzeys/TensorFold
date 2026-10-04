"""The output head's quantized copies: a 4-bit draft head (exact for replies) and an optional FP8 verify head (lossy).

Each rank holds a [vocab / world, hidden] BF16 share of lm_head (475.8 MB on full GLM-5.3 at TP4). Read through the
router GEMV it costs ~1.9 ms a pass at GB10's bandwidth, and a DFlash2 round reads it twice: the drafter's block and
the verify window.

TF_GLM53_DRAFT_HEAD (default ``q4``): drafts read a 4-bit affine copy of the share (groups of 64, ~134 MB a rank):
glm5_next's ``quantize4`` - upstream's own rule there, "Draft steps use the quantized head; verification keeps the
original head". That covers the DFlash2 drafter's block (dflash.py) and the MTP steps' heads (the reduced draft
vocabulary of TF_GLM53_DRAFT_VOCAB, or the full share under an MTP mode's ":full"). Exact for replies: drafts only
propose and the target verifies every one with its own head, so only acceptance can move. ``bf16``: the old reads.

TF_GLM53_VERIFY_HEAD (default ``bf16``): ``fp8`` holds the head that verify windows and prompts read as FP8 e4m3 with
an fp32 scale a row and 128-column block (absmax / 448), multiplied by MiaAI-Lab's ``_fmm`` (GLM-5.3-Flash patch
0002-glm-dense-fp8, Apache-2.0). LOSSY: every logit changes, so greedy picks and sampled draws can differ from BF16 and
replies change. It needs a quality A/B (needles, logprob KLD against bf16, prose/code evals) before anyone adopts it.
The kernel computes each row alone like the BF16 one, so drafted == serial still holds under it. With both
``fp8`` and ``q4`` nothing reads the BF16 share any more and it is freed (-476 MB a rank).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda import glue, qmm

DRAFT_HEAD = os.environ.get("TF_GLM53_DRAFT_HEAD", "q4")       # q4 | bf16
VERIFY_HEAD = os.environ.get("TF_GLM53_VERIFY_HEAD", "bf16")   # bf16 | fp8 (lossy)
if DRAFT_HEAD not in ("q4", "bf16"):
    raise ValueError(f"TF_GLM53_DRAFT_HEAD={DRAFT_HEAD!r}: q4 or bf16")
if VERIFY_HEAD not in ("bf16", "fp8"):
    raise ValueError(f"TF_GLM53_VERIFY_HEAD={VERIFY_HEAD!r}: bf16 or fp8")

F8_BLOCK = 128
F8_MAX = 448.0
F8_BN = 64
F8_BN_DECODE = 32          # windows up to 16 rows: narrower column tiles, deeper pipeline (MiaAI: +8% measured)
# (warps, stages) by row bucket; neither they nor the column tile change a row's bits
F8_CONFIG = {16: (4, 6), 32: (4, 3), 64: (4, 2), 128: (8, 2)}


@dataclass
class F8:
    """A BF16 matrix [n, k] held as FP8 e4m3 [n, k] and fp32 scales [n, k / 128]: W ~ q * scale."""

    weight: torch.Tensor      # [n, k] torch.float8_e4m3fn
    scale: torch.Tensor       # [n, k // 128] fp32
    n: int
    k: int

    def nbytes(self) -> int:
        return self.weight.numel() + self.scale.numel() * 4


def make_f8(weight: torch.Tensor, rows: int = 8192) -> F8:
    """Quantize a BF16 matrix a block of 128 columns at a time (absmax / 448 scale, round to nearest)."""
    n, k = weight.shape
    if k % F8_BLOCK:
        raise ValueError(f"FP8 head: hidden {k} is not a multiple of {F8_BLOCK}")
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((n, k // F8_BLOCK), dtype=torch.float32, device=weight.device)
    for r in range(0, n, rows):
        g = weight[r:r + rows].float().view(-1, k // F8_BLOCK, F8_BLOCK)
        s = (g.abs().amax(-1) / F8_MAX).clamp_min(1e-30)
        q[r:r + rows] = (g / s[..., None]).clamp(-F8_MAX, F8_MAX).view(-1, k).to(torch.float8_e4m3fn)
        scale[r:r + rows] = s
    return F8(q, scale, int(n), int(k))


@triton.jit
def _fmm(X, W, S, OUT, PART, M, x_stride, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
    """glm5_next's _bmm over FP8 weights: each BK step's product (bf16 inputs, fp32 sums) times its 128-column block's
    scales (MiaAI-Lab 0002-glm-dense-fp8)."""
    PER: tl.constexpr = K // SK
    KB: tl.constexpr = K // 128
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_s * PER, pid_s * PER + PER, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
        s = tl.load(S + rn.to(tl.int64) * KB + k0 // 128, mask=n_ok, other=0.0)
        acc = acc + tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def matmul_f8(x: torch.Tensor, q: F8, out: torch.Tensor, f32: bool = True) -> torch.Tensor:
    m, k = x.shape
    if k != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16:
        raise ValueError(f"matmul_f8: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    if out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul_f8: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    bm = qmm.bucket(min(m, 128))
    warps, stages = F8_CONFIG[bm]
    sk = qmm.b16_split_k(q.n, q.k)                       # a function of the shape (1 for a 38720 x 6144 share)
    part = torch.empty((sk * m * q.n,), dtype=torch.float32, device=x.device) if sk > 1 else out
    bn = F8_BN_DECODE if bm == 16 else F8_BN
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, bn), sk)
    _fmm[grid](x, q.weight, q.scale, out, part, m, x.stride(0), N=q.n, K=k, SK=sk, BM=bm, BLOCK_N=bn,
               BK=qmm.B16_BK, F32=f32, num_warps=warps, num_stages=stages)
    if sk > 1:
        total = m * q.n
        qmm._reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


def rows_of(table) -> int:
    """Vocabulary rows of a head table (bf16 tensor, Q4 or F8)."""
    return table.n if isinstance(table, (qmm.Q4, F8)) else table.shape[0]


def logits(x: torch.Tensor, table, out: torch.Tensor, xs: torch.Tensor | None = None) -> torch.Tensor:
    """Rows x [m, hidden] bf16 times a head table -> out [m, rows] fp32: the router GEMV for a BF16 share, the 4-bit
    lane matmul for a Q4 copy (``xs``: x's 64-input group sums, computed when None), the FP8 kernel for an F8 one."""
    if isinstance(table, qmm.Q4):
        return qmm.matmul(x, table, xs, out=out, f32=True)
    if isinstance(table, F8):
        return matmul_f8(x, table, out)
    return glue.router(x, table, out)


def prepare(w, drafts: bool = True) -> None:
    """Build the rank's head copies per TF_GLM53_DRAFT_HEAD / TF_GLM53_VERIFY_HEAD (engine load, after the reduced
    draft vocabulary is set): w.draft_lm_head (full-share drafts), w.draft_head (reduced-vocabulary drafts) and
    w.verify_head (verify windows and prompts). ``drafts`` False (no MTP, no DFlash2): no draft copies."""
    k = w.lm_head.shape[1]
    made = []
    if drafts and DRAFT_HEAD == "q4" and k % qmm.GS == 0:
        w.draft_lm_head = qmm.quantize4(w.lm_head)
        made.append(f"draft head q4 ({w.draft_lm_head.nbytes() / 2**20:.0f} MiB)")
        if w.draft_head is not None and isinstance(w.draft_head, torch.Tensor):
            w.draft_head = qmm.quantize4(w.draft_head)
            made.append(f"reduced draft vocabulary q4 ({w.draft_head.nbytes() / 2**20:.0f} MiB)")
    if VERIFY_HEAD == "fp8":
        w.verify_head = make_f8(w.lm_head)
        made.append(f"verify head FP8 ({w.verify_head.nbytes() / 2**20:.0f} MiB, LOSSY)")
        if drafts and w.draft_lm_head is w.lm_head:
            made.append("bf16 share kept for drafts (TF_GLM53_DRAFT_HEAD=bf16)")
        else:                                           # nothing reads the BF16 share any more
            if w.draft_lm_head is w.lm_head:
                w.draft_lm_head = None
            w.lm_head = None
            made.append("bf16 share freed")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if w.rank == 0:
        print(f"[tensorfold] output head: {', '.join(made) or 'bf16 (drafts and verify)'}", flush=True)
