"""TF_GLM53_L2PF (``families/glm_moe_dsa/cuda/l2pf``): the decode windows' L2 prefetch.

- The plan: an EXL3 linear's warp-chunk heads start exactly where the kernel's warps start their k ranges (strips
  layout), stay inside the words, and a site never exceeds its budget.
- The kernel in each mode runs on a side stream, inside a CUDA graph too, and changes no byte of what it reads.
- On real weights (TF_GLM53_CKPT, four ranks as threads, layers 0-3): a decode window's hidden rows and logits are
  bit-identical with the prefetcher on and off, and with the routed-expert knobs of tests/cuda/
  test_exl3_experts_decode_paths.py (loads, fused epilogues, PDL, the folded shared-expert add) at every setting.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _linear(k: int, n: int, bits: int, split, device="cuda"):
    from tensorfold.cuda.exl3 import linear as x3linear

    g = torch.Generator().manual_seed(k + n + bits)
    t = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int32, generator=g).to(torch.int16)
    suh = torch.ones((k,), dtype=torch.float16)
    svh = torch.ones((n,), dtype=torch.float16)
    lin = x3linear.Exl3Linear.from_tensors(t, suh, svh, "mcg", device=device)
    lin.split = split
    return lin


@pytest.mark.parametrize("k,n,bits,split", [(4096, 6144, 4, (4, 4)), (6144, 2048, 3, (8, 4)), (2048, 1024, 2, (1, 8))])
def test_heads_start_where_the_warps_start(k, n, bits, split):
    from tensorfold.families.glm_moe_dsa.cuda import l2pf

    lin = _linear(k, n, bits, split)
    base, size = lin.words.data_ptr(), lin.words.numel() * 4
    sk, wk = split
    kt = k // 16
    step = lin.strides[0] * 4                    # bytes of one k step of a strip
    per_warp = kt // (sk * wk)
    budget = size // 3
    heads = l2pf.exl3_heads(lin, budget)
    assert heads and sum(b for _, b in heads) <= budget
    starts = {base + nb * lin.strides[1] * 4 + (s * wk + w) * per_warp * lin.strides[0] * 4
              for nb in range(n // 128) for s in range(sk) for w in range(wk)}
    assert {a for a, _ in heads} == starts
    for a, b in heads:
        assert a % 16 == 0 and b % 16 == 0 and base <= a and a + b <= base + size
        assert b <= per_warp * step


def test_site_budget_and_groups():
    from tensorfold.families.glm_moe_dsa.cuda import l2pf

    q_a, kv_a = _linear(6144, 2048, 4, (8, 4)), _linear(6144, 640, 4, (16, 4))
    norm = torch.ones((6144,), dtype=torch.bfloat16, device="cuda")
    for mb in (0.5, 2, 8, 64):
        budget = int(mb * (1 << 20))
        spans = l2pf.ranges([norm, [q_a, kv_a], norm], budget)
        total = sum(b for _, b in spans)
        assert total <= budget + 16 * len(spans)
        if mb == 64:                                     # everything fits: whole tensors
            assert total >= q_a.words.numel() * 4 + kv_a.words.numel() * 4
        for a, b in l2pf.pieces(spans, 32 << 10):
            assert a % 16 == 0 and 0 < b <= 32 << 10


@pytest.mark.parametrize("mode", (0, 1, 2), ids=("bulk", "lines", "touch"))
def test_prefetch_kernel_reads_only_and_captures(mode):
    from tensorfold.families.glm_moe_dsa.cuda import l2pf

    w = torch.randn((8 << 20,), device="cuda")
    before = w.clone()
    spans = l2pf.pieces([(w.data_ptr(), w.numel() * 4)], 32 << 10)
    table = torch.tensor(spans, dtype=torch.int64, device="cuda").view(-1, 2).contiguous()
    sink = torch.zeros((1,), dtype=torch.int32, device="cuda")
    side = torch.cuda.Stream()
    ext = l2pf._ext()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ext.prefetch(table, 0, table.shape[0], mode, 48, 128, sink)
    torch.cuda.current_stream().wait_stream(side)
    y = torch.empty_like(w)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            ext.prefetch(table, 1, table.shape[0] - 1, mode, 48, 128, sink)
        torch.mul(w, 2.0, out=y)
        main.wait_stream(side)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(w, before) and torch.equal(y, before * 2.0) and int(sink.item()) == 0


# ---------------------------------------------------------------------------------------- real weights, layers 0-3

CKPT = os.environ.get("TF_GLM53_CKPT", "")


@pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")
def test_decode_window_bits_with_prefetch_and_expert_knobs(monkeypatch):
    from test_glm_moe_dsa_fused import _fused, _prefill, _tokens, _window
    from threadcomm import run_ranks

    from tensorfold.cuda.exl3 import experts as x3experts
    from tensorfold.families.glm_moe_dsa.cuda import fused, l2pf, runner

    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))
    toks = _tokens(40, seed=23)
    # (loads, fuse, pdl, fold, l2pf mode): the first is the old path
    configs = [(0, 0, 0, False, None), (1, 1, 1, True, None), (1, 1, 1, True, 0), (4, 3, 1, True, 0),
               (2, 0, 1, False, 1), (1, 1, 0, True, 2)]

    def run(rank, comm):
        w, st = _fused(rank, 4, comm, 64)
        big = fused.Buffers(w, 64, 64)
        small = fused.Buffers(w, 4, 64)
        _prefill(w, st, big, toks, 0, 30)
        out = []
        for loads, fuse, pdl, fold, mode in configs:
            comm.barrier()                               # module knobs are shared by the rank threads
            if rank == 0:
                x3experts.LOADS, x3experts.FUSE, x3experts.PDL = loads, fuse, pdl
                fused.FOLD_SHARED = fold
            comm.barrier()
            s = l2pf.Settings.from_env({"TF_GLM53_L2PF": "0" if mode is None else ("1", "lines", "touch")[mode]})
            w.l2pf = l2pf.Prefetch(w, s, fold_shared=fold) if s.on else None
            assert w.l2pf is None or w.l2pf.sites
            out.append(_window(w, st, small, toks, 30, 33))
        comm.barrier()
        return out

    saved = (x3experts.LOADS, x3experts.FUSE, x3experts.PDL, fused.FOLD_SHARED)
    try:
        results = run_ranks(run, 4)
    finally:
        x3experts.LOADS, x3experts.FUSE, x3experts.PDL, fused.FOLD_SHARED = saved
    for out in results:
        h0, l0 = out[0]
        for i, (h, lg) in enumerate(out[1:], 1):
            assert torch.equal(h.view(torch.int16), h0.view(torch.int16)), ("hidden", configs[i])
            assert torch.equal(lg.view(torch.int32), l0.view(torch.int32)), ("logits", configs[i])
