"""The routed EXL3 experts' decode-path knobs (``tensorfold/cuda/exl3/experts``) give the original path's bits.

- TF_EXL3_EXPERTS_LOADS / ``routed(loads=)``: 16-byte non-coherent trellis loads 1..4 k steps ahead against the
  32-bit walk (0), per slot (``wts`` None) and combined, on mixed-width layers of every codebook, on GLM-5.3's TP=4
  shapes (6144 -> 512 -> 6144, gate|up as column blocks of one fused trellis) and in windows of 1..32 rows.
- TF_EXL3_EXPERTS_FUSE / ``routed(fuse=)`` and TF_EXL3_EXPERTS_PDL / ``routed(pdl=)``: the down epilogue and the
  combine inside the down launch (1), the gate/up epilogue in gate/up's last block (2), both (3), with and without
  programmatic dependent launch, against the separate kernels (fuse 0, pdl 0): per-slot y, the combined rows, the
  folded shared-expert add (``sy``) against ``out + sy``, rows whose slots are all non-routed, and CUDA graphs
  replayed several times (the arrival counts reset themselves).
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


# ------------------------------------------------------------------------------------- fused epilogues and PDL

def _compare_fuse(ex, D: int, E: int, topk: int, rows: tuple, shared: bool, seed: int, act_mode=None,
                  empty_rows: bool = False) -> None:
    from tensorfold.cuda.exl3 import experts

    slots = topk + (1 if shared else 0)
    maxr = max(rows)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((maxr, D), generator=g).to(torch.bfloat16).cuda()
    sel, w = _picks(E, maxr, topk, g, shared=shared)
    if empty_rows:                                      # rows 1 and 3: no routed slot (ids >= E or -1)
        for r in (1, 3):
            if r < maxr:
                sel[r] = torch.tensor([E + (q % 2) if q % 3 else -1 for q in range(slots)], dtype=torch.int32)
    sy = torch.randn((maxr, D), generator=g).float().cuda()
    caller_y = torch.randn((maxr * slots, D), generator=g).float().cuda()
    s = experts.Scratch(ex, maxr, slots)
    kw = {} if act_mode is None else {"act_mode": act_mode}
    for R in rows:
        pick, wts, P = sel[:R].contiguous(), w[:R].contiguous(), R * slots

        def run(fuse, pdl):
            s.y[:P].copy_(caller_y[:P])
            y = experts.routed(x[:R], pick, None, ex, s, None, R, fuse=fuse, pdl=pdl, **kw).clone()
            s.y[:P].copy_(caller_y[:P])
            out = experts.routed(x[:R], pick, wts, ex, s, None, R, fuse=fuse, pdl=pdl, **kw).clone()
            y2 = s.y[:P].clone()                         # the combine's per-slot outputs (routed ones written)
            s.y[:P].copy_(caller_y[:P])
            folded = experts.routed(x[:R], pick, wts, ex, s, None, R, fuse=fuse, pdl=pdl, sy=sy[:R], **kw).clone()
            torch.cuda.synchronize()
            return y, out, y2, folded

        ref_y, ref_out, ref_y2, ref_folded = run(0, 0)
        assert torch.isfinite(ref_out).all()
        assert torch.equal(_bits(ref_folded), _bits(ref_out + sy[:R])), ("sy", R)
        for fuse in (0, 1, 2, 3):
            for pdl in (0, 1):
                if fuse == 0 and pdl == 0:
                    continue
                y, out, y2, folded = run(fuse, pdl)
                assert torch.equal(_bits(y), _bits(ref_y)), ("y", fuse, pdl, R)
                assert torch.equal(_bits(y2), _bits(ref_y2)), ("y with wts", fuse, pdl, R)
                assert torch.equal(_bits(out), _bits(ref_out)), ("out", fuse, pdl, R)
                assert torch.equal(_bits(folded), _bits(ref_folded)), ("folded", fuse, pdl, R)


@pytest.mark.parametrize("name,cb,kfun", MIXED, ids=[m[0] + str(i) for i, m in enumerate(MIXED)])
def test_fused_epilogues_and_pdl_equal_the_separate_kernels(name, cb, kfun):
    E, D, I = 24, 512, 256
    ex, _ = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=51 + cb)
    _compare_fuse(ex, D, E, 6, (1, 2, 3, 8, 17, 32), shared=True, seed=11)


def test_fused_combine_rows_without_routed_slots():
    """Rows whose every slot is non-routed get no arrival: the designated block combines the caller's y for them."""

    E, D, I = 16, 512, 256
    _, cb, kfun = MIXED[2]
    ex, _ = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=61)
    _compare_fuse(ex, D, E, 4, (2, 4, 9), shared=True, seed=13, empty_rows=True)


@pytest.mark.parametrize("cb", (1, 2), ids=("mcg", "mul1"))
def test_fused_epilogues_glm53_shapes(cb):
    from tensorfold.cuda.exl3 import experts

    E, D, I = 40, 6144, 512
    ex = _glm53_layer(E, D, I, cb, seed=15 + cb)
    _compare_fuse(ex, D, E, 8, (1, 4, 8, 32), shared=False, seed=17, act_mode=experts.ACT_BF16)


@pytest.mark.parametrize("fuse", (1, 3))
def test_fused_pdl_graphs_replay_equal_eager(fuse):
    """The fused chain with PDL captured at 1, 2, 4 and 8 rows (one scratch, as an engine captures them) replays
    three times, with new inputs between, to the unfused eager call's bits: the arrival counts reset themselves."""

    from tensorfold.cuda.exl3 import experts

    E, D, I, TOPK = 32, 512, 256, 6
    _, cb, kfun = MIXED[1]
    ex, _ = _layer(E, D, I, [kfun(e) for e in range(E)], cb, seed=71)
    ROWS = (1, 2, 4, 8)
    scratch = experts.Scratch(ex, max(ROWS), TOPK + 1)
    g = torch.Generator().manual_seed(19)

    def inputs(R):
        x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
        sel, w = _picks(E, R, TOPK, g, shared=True)
        return x, sel, w, torch.randn((R, D), generator=g).float().cuda()

    bufs, graphs = {}, {}
    for R in ROWS:
        x, sel, w, sy = inputs(R)
        out = torch.empty((R, D), dtype=torch.float32, device="cuda")
        bufs[R] = (x, sel, w, sy, out)
        experts.routed(x, sel, w, ex, scratch, out, R, fuse=fuse, pdl=1, sy=sy)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            experts.routed(x, sel, w, ex, scratch, out, R, fuse=fuse, pdl=1, sy=sy)
        graphs[R] = graph
    for trial in range(3):
        for R in ROWS:
            x, sel, w, sy, out = bufs[R]
            if trial:
                nx, nsel, nw, nsy = inputs(R)
                x.copy_(nx)
                sel.copy_(nsel)
                w.copy_(nw)
                sy.copy_(nsy)
            eager = experts.routed(x, sel, w, ex, scratch, None, R, fuse=0, pdl=0, sy=sy).clone()
            out.fill_(float("nan"))
            graphs[R].replay()
            torch.cuda.synchronize()
            assert torch.equal(_bits(out), _bits(eager)), (fuse, R, trial)
    assert int(scratch.done_d.abs().sum()) == 0 and int(scratch.done_gu.abs().sum()) == 0
