"""Full GLM-5.3's RoCE decode reductions (b12x RoCEnante) on a CPU, no NICs: NCCL-style NCCL_IB_HCA values reach the
proxy as device names, a sum other than the NCCL rank-order sum is refused at startup, a runtime that timed out
stops every rank before a window's tokens are read, and the RoCE setup rendezvous starts only once every rank has
loaded.

The setup tests run four rank threads: ``ready`` is the NCCL store barrier, ``all_gather`` a rank-order
concatenation, and a stand-in RoceReduce joins a rendezvous with a short timeout, as b12x's gloo rendezvous does."""

from __future__ import annotations

import threading
import time
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


class _Ranks:
    """What four rank threads share: the store barrier behind ``ready``, the all-gather slots, and the RoCE setup
    rendezvous, whose ``join_s`` stands in for the 120 s gloo join timeout."""

    def __init__(self, join_s, world=4):
        self.world = world
        self.loaded = threading.Barrier(world, timeout=30)
        self.gathered = threading.Barrier(world, timeout=30)
        self.rendezvous = threading.Barrier(world, timeout=join_s)
        self.slots = [None] * world


class _RankComm:
    def __init__(self, ranks, rank):
        self.ranks, self.rank, self.world, self.labels = ranks, rank, ranks.world, []

    def ready(self, label):
        self.labels.append(label)
        self.ranks.loaded.wait()

    def all_gather(self, send, recv):
        self.ranks.slots[self.rank] = send.reshape(-1).clone()
        self.ranks.gathered.wait()
        recv.view(-1).copy_(torch.cat(self.ranks.slots))
        self.ranks.gathered.wait()


def _start_four(monkeypatch, *, late=(), late_s=0.0, join_s=0.5, fails=()):
    """roce.start_everywhere on four rank threads; ranks in ``late`` reach it ``late_s`` after the others."""
    ranks = _Ranks(join_s)

    class Rendezvous:                            # RoceReduce's setup: every rank joins within join_s, or none does
        def __init__(self, rank, world, master, port, nccl=None):
            ranks.rendezvous.wait()              # threading.BrokenBarrierError: "3/4 clients joined"
            if rank in fails:
                raise RuntimeError("no RoCE v2 GID for this address")
            self.rank = rank

    monkeypatch.setattr(roce, "RoceReduce", Rendezvous)
    comms = [_RankComm(ranks, r) for r in range(4)]
    got, errors = [None] * 4, [None] * 4

    def rank_thread(r):
        if r in late:
            time.sleep(late_s)                   # this rank's weights take longer to load
        try:
            got[r] = roce.start_everywhere(r, 4, "127.0.0.1", 29861, comms[r], device="cpu")
        except Exception as exc:                 # noqa: BLE001
            errors[r] = exc

    threads = [threading.Thread(target=rank_thread, args=(r,)) for r in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return got, errors, comms


def test_the_setup_rendezvous_waits_for_a_rank_that_loads_late(monkeypatch):
    """rank 0 hosts the rendezvous and loads last: the others wait for it at the store barrier, not in the join."""
    got, errors, comms = _start_four(monkeypatch, late=(0,), late_s=1.5, join_s=0.5)
    assert errors == [None] * 4
    assert [g.rank if g is not None else None for g in got] == [0, 1, 2, 3]
    assert [c.labels for c in comms] == [["loading"]] * 4


def test_one_rank_without_roce_leaves_every_rank_on_nccl(monkeypatch):
    got, errors, _ = _start_four(monkeypatch, fails=(2,))
    assert errors == [None] * 4
    assert got == [None] * 4


def test_a_rank_missing_at_the_barrier_stops_startup(monkeypatch):
    """ready() raises after its hour naming the missing rank: that is an error, not a silent NCCL fallback, and the
    rendezvous never opens."""
    built = []
    monkeypatch.setattr(roce, "RoceReduce", lambda *a, **k: built.append(a))

    class Missing(_Nccl):
        def ready(self, label):
            raise RuntimeError(f"rank 1 finished {label} but rank 0 has not after 60 min")

    with pytest.raises(RuntimeError, match="rank 0 has not"):
        roce.start_everywhere(1, 1, "127.0.0.1", 29861, Missing(), device="cpu")
    assert built == []
