"""Full GLM-5.3's RoCE decode reductions (b12x RoCEnante) on a CPU, no NICs: NCCL-style NCCL_IB_HCA values reach the
proxy as device names, and a sum other than the NCCL rank-order sum is refused at startup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm_moe_dsa.cuda import roce  # noqa: E402


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
    r.rank, r.world, r.device = 0, world, torch.device("cpu")

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
