"""The routed EXL3 experts' decode-path knobs (``tensorfold/cuda/exl3/experts``) give the original path's bits.

- TF_EXL3_EXPERTS_LOADS / ``routed(loads=)``: 16-byte non-coherent trellis loads 1..4 k steps ahead against the
  32-bit walk (0), per slot (``wts`` None) and combined, on mixed-width layers of every codebook, on GLM-5.3's TP=4
  shapes (6144 -> 512 -> 6144, gate|up as column blocks of one fused trellis) and in windows of 1..32 rows.
"""

from __future__ import annotations

import math

import pytest
import torch

from test_exl3_experts import MIXED, _layer, _picks, _scale, _trellis

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _glm53_layer(E: int, D: int, I: int, cb: int, seed: int):
    """GLM-5.3's rank share: gate|up of each expert as column blocks of one [D/16, 2I/16, *] trellis (experts_cx's
    layout, ``gu_stride``), widths 2..4 bits a expert."""

    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(seed)
    gate, up, down = [], [], []
    for e in range(E):
        k2 = (4, 5, 6, 7, 8)[e % 5] if cb == 2 else (4, 6, 8)[e % 3]
        it = I // 16
        fused = torch.empty((D // 16, 2 * it, 8 * k2), dtype=torch.int16, device="cuda")
        fused[:, :it].copy_(_trellis(D, I, k2, g))
        fused[:, it:].copy_(_trellis(D, I, k2, g))
        gate.append((fused[:, :it], _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        up.append((fused[:, it:], _scale(D, 1 / math.sqrt(D), g), _scale(I, 1.0, g)))
        down.append((_trellis(I, D, k2, g), _scale(I, 1 / math.sqrt(I), g), _scale(D, 0.25, g)))
    return experts.prepare(gate, up, down, cb, gu_stride=2 * (I // 16))


def _compare_loads(ex, D: int, E: int, topk: int, rows: tuple, shared: bool, seed: int, act_mode=None) -> None:
    from tensorfold.cuda.exl3 import experts

    slots = topk + (1 if shared else 0)
    maxr = max(rows)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((maxr, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, maxr, topk, g, shared=shared)
    s = experts.Scratch(ex, maxr, slots)
    kw = {} if act_mode is None else {"act_mode": act_mode}
    for R in rows:
        pick, wts = sel[:R].contiguous(), w[:R].contiguous()
        ref_y, ref_out = None, None
        for loads in (0, 1, 2, 3, 4):
            s.y.fill_(3.0)                                   # the non-routed slot's output (the caller's)
            y = experts.routed(x[:R], pick, None, ex, s, None, R, loads=loads, **kw).clone()
            out = experts.routed(x[:R], pick, wts, ex, s, None, R, loads=loads, **kw).clone()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
            if loads == 0:
                ref_y, ref_out = y, out
                continue
            assert torch.equal(_bits(y), _bits(ref_y)), ("y", loads, R)
            assert torch.equal(_bits(out), _bits(ref_out)), ("out", loads, R)


@pytest.mark.parametrize("name,cb,kfun", MIXED, ids=[m[0] + str(i) for i, m in enumerate(MIXED)])
def test_vector_loads_equal_the_32bit_walk_mixed_widths(name, cb, kfun):
    E, D, I = 24, 512, 256
    ex, _ = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=41 + cb)
    assert ex.aligned16
    _compare_loads(ex, D, E, 6, (1, 2, 3, 8, 17, 32), shared=True, seed=7)


@pytest.mark.parametrize("cb", (1, 2), ids=("mcg", "mul1"))
def test_vector_loads_equal_the_32bit_walk_glm53_shapes(cb):
    from tensorfold.cuda.exl3 import experts

    E, D, I = 40, 6144, 512
    ex = _glm53_layer(E, D, I, cb, seed=5 + cb)
    assert ex.aligned16 and ex.gu_stride == 2 * (I // 16)
    _compare_loads(ex, D, E, 8, (1, 3, 8, 16, 32), shared=False, seed=9, act_mode=experts.ACT_BF16)


def test_unaligned_trellis_falls_back_to_32bit_loads():
    """A trellis that starts 4 bytes past a 16-byte boundary: prepare() says so and routed() keeps the 32-bit walk
    (which needs 4-byte words) whatever ``loads`` asks for."""

    from tensorfold.cuda.exl3 import experts

    E, D, I = 4, 256, 128
    g = torch.Generator().manual_seed(1)
    mats = []
    for _ in range(3 * E):
        k, n = (D, I) if len(mats) % 3 < 2 else (I, D)
        t = _trellis(k, n, 8, g)
        flat = torch.empty((t.numel() + 2,), dtype=torch.int16, device="cuda")
        view = flat[2:].view(t.shape)                        # 4 bytes past the allocation's start
        view.copy_(t)
        mats.append((view, _scale(k, 0.05, g), _scale(n, 0.5, g)))
    ex = experts.prepare(mats[0::3], mats[1::3], mats[2::3], 1)
    assert not ex.aligned16
    _compare_loads(ex, D, E, 2, (1, 4), shared=False, seed=3)
