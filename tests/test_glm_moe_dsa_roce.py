"""Full GLM-5.3's RoCE decode reductions (b12x RoCEnante) on a CPU, no NICs: NCCL-style NCCL_IB_HCA values reach the
proxy as device names, a sum other than the NCCL rank-order sum is refused at startup, and a runtime that timed out
stops every rank before a window's tokens are read."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm_moe_dsa.cuda import fused, multi, roce, runner  # noqa: E402


@pytest.mark.parametrize("value,names", [(None, None), ("", None), ("rocep1s0f0", ["rocep1s0f0"]),
                                         ("=rocep1s0f0", ["rocep1s0f0"]), ("=mlx5_0:1,mlx5_1:1", ["mlx5_0", "mlx5_1"]),
                                         ("rocep1s0f0, roceP2p1s0f0", ["rocep1s0f0", "roceP2p1s0f0"])])
def test_nccl_ib_hca_as_device_names(value, names):
    assert roce.hca_names(value) == names


def test_an_exclusion_list_is_refused():
    with pytest.raises(ValueError, match="excludes devices"):
        roce.hca_names("^mlx5_2")


class _Nccl:
    """all_gather for one process standing in for every rank: this rank's value, then the others' (``others``)."""

    def __init__(self, others=()):
        self.others = list(others)

    def all_gather(self, send, recv):
        flat = send.reshape(-1)
        n = flat.numel()
        if recv.numel() == n:
            recv.view(-1).copy_(flat)
            return
        recv.view(-1)[:n].copy_(flat)
        for i, v in enumerate(self.others):
            recv.view(-1)[(i + 1) * n:(i + 2) * n].fill_(v)


def _reduce(world=1, others=(), poisoned=False, summed=lambda x: x):
    r = object.__new__(roce.RoceReduce)
    r.rank, r.world, r.device, r.nccl = 0, world, torch.device("cpu"), _Nccl(others)
    r._flag = torch.zeros((1,), dtype=torch.int32)
    r._every = torch.zeros((world,), dtype=torch.int32)

    def check_health():
        raise RuntimeError("b12x: RoCE wait timed out")

    r.rt = SimpleNamespace(poisoned=poisoned, check_health=check_health)
    r.all_reduce = lambda x, out: out.copy_(summed(x))
    return r


def test_startup_accepts_the_rank_order_sum():
    _reduce()._check(_Nccl())


def test_startup_refuses_another_sum():
    r = _reduce(summed=lambda x: x * (1 + 2 ** -20))
    with pytest.raises(RuntimeError, match="not the NCCL rank-order sum"):
        r._check(_Nccl())


def test_healthy_ranks_go_on():
    _reduce(world=4, others=[0, 0, 0]).healthy_everywhere()


def test_a_timeout_on_another_rank_stops_this_one():
    with pytest.raises(RuntimeError, match="failed on rank 2"):
        _reduce(world=4, others=[0, 1, 0]).healthy_everywhere()


def test_the_rank_that_timed_out_reports_b12x_reason():
    with pytest.raises(RuntimeError, match="b12x: RoCE wait timed out"):
        _reduce(world=4, others=[0, 0, 0], poisoned=True).healthy_everywhere()


class _Spy:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def healthy_everywhere(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("RoCE reductions failed on rank 1")


def _runner(fast, monkeypatch):
    monkeypatch.setattr(fused, "compute", lambda *a, **k: None)
    r = object.__new__(runner.Runner)
    r.w, r.st, r.vb = SimpleNamespace(fast=fast), SimpleNamespace(pos=torch.zeros(1)), None
    r.G = runner.GraphSet(enabled=False)
    return r


def test_every_verify_window_is_checked(monkeypatch):
    spy = _Spy()
    r = _runner(spy, monkeypatch)
    r._verify(3, 10, None, "argmax")
    r._verify(1, 13, None, "argmax")
    assert spy.calls == 2


def test_nccl_windows_are_not_checked(monkeypatch):
    _runner(None, monkeypatch)._verify(3, 10, None, "argmax")          # no RoCE runtime: nothing to ask


def test_concurrent_windows_are_checked(monkeypatch):
    monkeypatch.setattr(fused, "compute", lambda *a, **k: None)
    spy = _Spy()
    m = object.__new__(multi.GlmMultiDecoder)
    m.w, m.vb, m.vrows = SimpleNamespace(fast=spy), None, None
    m.runner = SimpleNamespace(G=runner.GraphSet(enabled=False))
    m._verify(4, None, "argmax")
    assert spy.calls == 1


def test_no_token_leaves_a_prefill_whose_reductions_failed(monkeypatch):
    """A short prompt's chunk reduces over RoCE too: a failure there stops the request before its first token."""
    r = _runner(_Spy(fail=True), monkeypatch)
    r.mb, r.cap, r.k, r.drafter = None, None, 0, None
    r.prefill = lambda prompt: torch.zeros((1, 16))
    sent = []
    with pytest.raises(RuntimeError, match="failed on rank 1"):
        r.generate([1, 2, 3], 4, None, lambda t: False, sent.extend, 0)
    assert sent == []
