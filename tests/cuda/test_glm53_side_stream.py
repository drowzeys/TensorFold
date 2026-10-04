"""TF_GLM53_SIDE (``families/glm_moe_dsa/cuda/fused``): decode windows queue the key path (kv_a, the latent / rope key
and index key writes) and the shared expert on a second stream.

On real weights (TF_GLM53_CKPT, layers 0-3):

- four ranks as threads, eager: a decode window's hidden rows and logits - below index_topk and past it (rows that
  select) - are bit-identical with the side stream off and in every mode ("af", "wf", "a", "f"), with the shared
  expert's add folded or not, and beside the L2 prefetcher;
- one thread (rank 0's quarter, device-copy collectives), CUDA graphs: a captured window with the side stream
  replays the one-stream eager window's bits.
"""

from __future__ import annotations

import os

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402

# (side letters, fold the shared add, L2 prefetch): the first is the one-stream reference
CONFIGS = [("", True, False), ("af", True, False), ("wf", True, False), ("a", False, False), ("f", False, False),
           ("af", True, True), ("wf", False, True)]


def _prefill_to(w, st, big, toks, n):
    from test_glm_moe_dsa_fused import _prefill

    for a in range(0, n, 128):
        _prefill(w, st, big, toks, a, min(a + 128, n))


def test_side_stream_window_bits(monkeypatch):
    from test_glm_moe_dsa_fused import _fused, _tokens, _window

    from tensorfold.families.glm_moe_dsa.cuda import fused, l2pf, runner

    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))
    toks = _tokens(2100, seed=31)

    def run(rank, comm):
        w, st = _fused(rank, 4, comm, 4096)
        big = fused.Buffers(w, 128, 4096)
        small = fused.Buffers(w, 4, 4096, decode=True)
        _prefill_to(w, st, big, toks, 2096)
        out = []
        for side, fold, pf in CONFIGS:
            comm.barrier()                               # module knobs are shared by the rank threads
            if rank == 0:
                fused.SIDE, fused.FOLD_SHARED = side, fold
            comm.barrier()
            w.l2pf = l2pf.Prefetch(w, l2pf.Settings.from_env({"TF_GLM53_L2PF": "1"}), fold_shared=fold) if pf else None
            dense = _window(w, st, small, toks, 30, 33)          # rewrites rows 30-32 with their own bits
            sparse = _window(w, st, small, toks, 2096, 2099)     # rows past index_topk: select after the join
            out.append((dense, sparse))
        comm.barrier()
        return out

    saved = (fused.SIDE, fused.FOLD_SHARED)
    try:
        results = run_ranks(run, 4)
    finally:
        fused.SIDE, fused.FOLD_SHARED = saved
    for out in results:
        for i, got in enumerate(out[1:], 1):
            for name, (h, lg), (h0, l0) in zip(("dense", "sparse"), got, out[0]):
                assert torch.equal(h.view(torch.int16), h0.view(torch.int16)), (name, "hidden", CONFIGS[i])
                assert torch.equal(lg.view(torch.int32), l0.view(torch.int32)), (name, "logits", CONFIGS[i])


def test_side_stream_graph_replay_bits(monkeypatch):
    """One thread, CUDA graphs: the side stream's branch is captured and rejoins inside the graph."""
    from test_glm_moe_dsa_fused import _tokens, _window
    from test_glm_moe_dsa_multi import _Alone, _weights

    from tensorfold.families.glm_moe_dsa.cuda import fused, runner
    from tensorfold.families.glm_moe_dsa.cuda.runner import GraphSet

    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))
    toks = _tokens(2100, seed=37)
    saved = fused.SIDE
    try:
        with torch.no_grad():
            w = _weights(0, 4, _Alone())
            st = fused.State(w, 4096)
            big = fused.Buffers(w, 128, 4096)
            small = fused.Buffers(w, 4, 4096, decode=True)
            _prefill_to(w, st, big, toks, 2096)
            for a, e in ((30, 33), (2096, 2099)):
                fused.SIDE = ""
                h0, l0 = _window(w, st, small, toks, a, e)
                T = fused.bucket(e, w.cfg.index_topk)
                for side in ("af", "wf"):
                    fused.SIDE = side
                    G = GraphSet(True)
                    small.ids[:e - a].copy_(toks[a:e].cuda())
                    st.pos.fill_(a)
                    fn = lambda: fused.compute(w, st, small, e - a, T, logits="all")   # noqa: E731
                    G.run(("w", side), fn)               # eager, then captured
                    small.hidden.zero_()
                    small.logits.zero_()
                    G.run(("w", side), fn)               # replayed
                    torch.cuda.synchronize()
                    assert torch.equal(small.hidden[:e - a].view(torch.int16), h0.view(torch.int16)), (a, side)
                    assert torch.equal(small.logits[:e - a].view(torch.int32), l0.view(torch.int32)), (a, side)
                    del G
    finally:
        fused.SIDE = saved
