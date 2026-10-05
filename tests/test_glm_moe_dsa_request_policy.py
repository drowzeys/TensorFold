"""Full GLM-5.3's request policy on a CPU, no weights: a request that names no drafter runs DFlash2 when the checkpoint
has no MTP layer and a drafter is loaded."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm_moe_dsa.cuda import engine as E  # noqa: E402
from tensorfold.families.glm_moe_dsa.cuda import fused, runner  # noqa: E402


def _engine(rank=0, k=0, drafter=True, cfg=(7, 0.6), shared=None):
    """A Glm53Engine without weights: _share records (rank 0) or replays (followers); _run records its arguments."""
    e = object.__new__(E.Glm53Engine)
    e.rank, e.k, e.limit, e.scheduler, e.multi = rank, k, 4096, None, None
    e.runner = SimpleNamespace(drafter=object() if drafter else None, _dflash_cfg=lambda: cfg)
    e.sent, e.runs = [], []
    feed = list(shared or [])

    def share(values):
        if rank == 0:
            e.sent.append(list(values))
            return list(values)
        return feed.pop(0)

    e._share = share
    e._run = lambda *a: e.runs.append(a) or {}
    return e


def _mode(e):
    """The mode a request ran: the header's mode index on the wire, as followers decode it."""
    mi = e.sent[0][2] >> 8 if e.rank == 0 else None
    return fused.MTP_MODES[mi - 1] if mi else None


def test_no_mtp_layer_drafts_with_dflash_by_default():
    e = _engine()
    e.generate([1, 2, 3], 8, None, lambda t: None)
    assert _mode(e) == "dflash" and e.runs[0][6] == "dflash"


def test_draft_false_stays_serial():
    e = _engine()
    e.generate([1, 2, 3], 8, None, lambda t: None, draft=False)
    assert _mode(e) is None and e.runs[0][5] == 0 and e.runs[0][6] is None


@pytest.mark.parametrize("k,drafter", [(2, True), (0, False)])
def test_default_unchanged_with_an_mtp_layer_or_no_drafter(k, drafter):
    e = _engine(k=k, drafter=drafter)
    e.generate([1, 2, 3], 8, None, lambda t: None)
    assert _mode(e) is None


def test_a_named_mode_wins():
    e = _engine()
    e.generate([1, 2, 3], 8, None, lambda t: None, mtp_mode="auto")
    assert _mode(e) == "auto"
