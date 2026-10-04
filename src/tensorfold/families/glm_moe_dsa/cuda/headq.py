"""The output head's quantized copies: a 4-bit draft head (exact for replies).

Each rank holds a [vocab / world, hidden] BF16 share of lm_head (475.8 MB on full GLM-5.3 at TP4). Read through the
router GEMV it costs ~1.9 ms a pass at GB10's bandwidth, and a DFlash2 round reads it twice: the drafter's block and
the verify window.

TF_GLM53_DRAFT_HEAD (default ``q4``): drafts read a 4-bit affine copy of the share (groups of 64, ~134 MB a rank):
glm5_next's ``quantize4`` - upstream's own rule there, "Draft steps use the quantized head; verification keeps the
original head". That covers the DFlash2 drafter's block (dflash.py) and the MTP steps' heads (the reduced draft
vocabulary of TF_GLM53_DRAFT_VOCAB, or the full share under an MTP mode's ":full"). Exact for replies: drafts only
propose and the target verifies every one with its own head, so only acceptance can move. ``bf16``: the old reads.
"""

from __future__ import annotations

import os

import torch

from tensorfold.families.glm5_next.cuda import glue, qmm

DRAFT_HEAD = os.environ.get("TF_GLM53_DRAFT_HEAD", "q4")       # q4 | bf16
if DRAFT_HEAD not in ("q4", "bf16"):
    raise ValueError(f"TF_GLM53_DRAFT_HEAD={DRAFT_HEAD!r}: q4 or bf16")


def rows_of(table) -> int:
    """Vocabulary rows of a head table (bf16 tensor or Q4)."""
    return table.n if isinstance(table, qmm.Q4) else table.shape[0]


def logits(x: torch.Tensor, table, out: torch.Tensor, xs: torch.Tensor | None = None) -> torch.Tensor:
    """Rows x [m, hidden] bf16 times a head table -> out [m, rows] fp32: the router GEMV for a BF16 share, the 4-bit
    lane matmul for a Q4 copy (``xs``: x's 64-input group sums, computed when None)."""
    if isinstance(table, qmm.Q4):
        return qmm.matmul(x, table, xs, out=out, f32=True)
    return glue.router(x, table, out)


def prepare(w, drafts: bool = True) -> None:
    """Build the rank's head copies per TF_GLM53_DRAFT_HEAD (engine load, after the reduced draft vocabulary is set):
    w.draft_lm_head (full-share drafts) and w.draft_head (reduced-vocabulary drafts); w.verify_head stays the share.
    ``drafts`` False (no MTP, no DFlash2): no draft copies."""
    k = w.lm_head.shape[1]
    made = []
    if drafts and DRAFT_HEAD == "q4" and k % qmm.GS == 0:
        w.draft_lm_head = qmm.quantize4(w.lm_head)
        made.append(f"draft head q4 ({w.draft_lm_head.nbytes() / 2**20:.0f} MiB)")
        if w.draft_head is not None and isinstance(w.draft_head, torch.Tensor):
            w.draft_head = qmm.quantize4(w.draft_head)
            made.append(f"reduced draft vocabulary q4 ({w.draft_head.nbytes() / 2**20:.0f} MiB)")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if w.rank == 0:
        print(f"[tensorfold] output head: {', '.join(made) or 'bf16 (drafts and verify)'}", flush=True)
