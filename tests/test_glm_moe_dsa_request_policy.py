"""Full GLM-5.3's request policy on a CPU, no weights: a request that names no drafter runs DFlash2 when the checkpoint
has no MTP layer and a drafter is loaded, and every rank decodes with rank 0's DFlash2 depth and confidence (shared in
the request header), whatever its own environment or DFLASH_CFG file says."""

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


def test_followers_take_rank_0s_dflash_policy():
    """Rank 0 at depth 5 / confidence 0.6; a follower whose own files say 7 / 0.3 (a torn or stale DFLASH_CFG, another
    environment) still decodes the request at 5 / 0.6, the window rank 0 verifies."""
    lead = _engine(cfg=(5, 0.6))
    lead.generate([1, 2, 3], 8, None, lambda t: None)
    header, prompt = lead.sent
    follower = _engine(rank=1, cfg=(7, 0.3), shared=[header, prompt])
    follower.follow(requests=1)
    assert lead.runs[0][7] == (5, 0.6)
    assert follower.runs[0][6] == "dflash" and follower.runs[0][7] == (5, 0.6)


def _runner(cfg_file_policy=(7, 0.3)):
    """A Runner that only gets as far as choosing its DFlash2 rounds (prefill and the rounds are recorded)."""
    r = object.__new__(runner.Runner)
    r.vb = r.mb = r.cap = None
    r.w = SimpleNamespace(fast=None)                     # NCCL reductions (no RoCE runtime to ask)
    r.k, r.drafter = 0, object()
    r.prefill = lambda prompt: torch.zeros((1, 16))
    r._dflash_cfg = lambda: cfg_file_policy
    r.seen = []
    r._generate_dflash = lambda *a: r.seen.append(a[-1]) or {}
    return r


def test_runner_uses_the_shared_policy_not_its_own():
    r = _runner()
    r.generate([1, 2], 4, None, lambda t: False, lambda t: None, 0, "dflash", dflash=(5, 0.6))
    assert r.seen == [(5, 0.6)]


def test_concurrent_requests_default_to_dflash_too():
    """--parallel N: the scheduler gets DFlash2 for a request that names no drafter on a checkpoint without MTP."""
    e = _engine()
    got = []
    e.multi = SimpleNamespace(dr=object())
    e.scheduler = SimpleNamespace(submit=lambda prompt, n, s, want, emit, stop_eos: got.append(want) or {})
    e.generate([1, 2, 3], 8, None, lambda t: None)
    e.generate([1, 2, 3], 8, None, lambda t: None, draft=False)
    assert got == ["dflash", False]


@pytest.mark.parametrize("rounds", ["_generate_dflash", "_generate_auto"])
def test_rounds_never_read_their_own_policy_when_given_one(monkeypatch, rounds):
    """With rank 0's (depth, confidence) passed in, a rank's own env / DFLASH_CFG is not read at all."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    r = object.__new__(runner.Runner)
    r.w = r.vb = r.mb = r.cap = None
    r.drafter, r.G = SimpleNamespace(block=8), SimpleNamespace(graphs={}, capture_s=0.0)

    def own():
        raise AssertionError("read this rank's own DFlash2 policy")

    r._dflash_cfg = own
    stop = lambda t: True                                 # noqa: E731  the first token ends the reply: no round
    if rounds == "_generate_dflash":
        st = r._generate_dflash([1], [5], 5, 1, 8, None, stop, lambda t: None, 0.0, None, (5, 0.6))
        assert (st["depth"], st["confidence"]) == (5, 0.6)
    else:
        r._generate_auto([5], 5, 1, 8, None, stop, lambda t: None, 0.0, None, 2, (5, 0.6))
