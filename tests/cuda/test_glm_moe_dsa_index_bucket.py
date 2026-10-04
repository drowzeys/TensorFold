"""The decode index-key buckets (fused.bucket, TF_GLM53_INDEX_SPLIT): a narrower key range selects the same keys.

``select`` over the power-of-two range and over the finer bucket must leave identical b.tok for every row - rows with
index_topk or more visible keys and rows with fewer - on both selection paths (torch.topk for few rows, the radix
select for 32-row windows). Synthetic index queries and keys, one GPU, no checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NH, D, K = 32, 128, 2048


def test_bucket_bounds():
    from tensorfold.families.glm_moe_dsa.cuda import fused

    prev = 0
    for t in range(K + 1, 300_000, 97):
        b = fused.bucket(t, K)
        p = max(2 * K, 1 << (t - 1).bit_length())
        assert t <= b <= p and b % 128 == 0 and b >= prev, (t, b, p)
        prev = b
    assert fused.bucket(K, K) is None


def _select(T, rows, p0, seed=3, ties=False):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    g = torch.Generator(device="cpu").manual_seed(seed)
    cfg = SimpleNamespace(index_n_heads=NH, index_head_dim=D, index_topk=K)
    w = SimpleNamespace(cfg=cfg, dcp=1, rank=0, world=1, fast=None, comm=None)
    T_max = 1 << 17
    ic = torch.randn((T_max, D), generator=g).to(torch.bfloat16).cuda()
    if ties:                                       # many exact score ties (relu zeros): ties go to lower positions
        ic[::3] = 0
    b = SimpleNamespace(iq=torch.randn((rows, NH * D), generator=g).to(torch.bfloat16).cuda(),
                        iw=torch.randn((rows, NH), generator=g).cuda(),
                        sc=torch.empty((min(rows, fused.SEL_ROWS) * T,), dtype=torch.int64, device="cuda"),
                        tok=torch.zeros((rows, K), dtype=torch.int32, device="cuda"), small=rows)
    pos = torch.tensor([p0], dtype=torch.int32, device="cuda")
    fused.select(w, b, ic, pos, rows, T)
    return b.tok.clone()


@pytest.mark.parametrize("rows", [1, 4, 8, 32])
@pytest.mark.parametrize("ties", [False, True])
def test_fine_bucket_selects_the_same_keys(rows, ties):
    from tensorfold.families.glm_moe_dsa.cuda import fused

    for p0 in (K - 3, 20_000, 32_768 - rows, 32_769, 40_000, 70_000):
        t = p0 + rows
        fine = fused.bucket(t, K)
        if fine is None:                           # the whole window below index_topk: no selection
            continue
        wide = max(2 * K, 1 << (t - 1).bit_length())
        a = _select(fine, rows, p0, ties=ties)
        b = _select(wide, rows, p0, ties=ties)
        assert torch.equal(a, b), (rows, p0, fine, wide)
