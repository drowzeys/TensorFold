"""TF_GLM53_MTP_REUSE (``families/glm_moe_dsa/cuda/fused``, ``runner``): a draft chain's later MTP steps attend the
first step's selection instead of scoring and selecting their own.

On real weights (TF_GLM53_CKPT, layers 0-3 + the MTP layer), a prompt past index_topk (so the chain's rows select)
and MTP depth 3: drafted greedy replies equal the serial reply in every reuse mode (0, 1, 2) - eager on four rank
threads, and with CUDA graphs on one thread (rank 0's quarter, device-copy collectives), where the reuse steps run
as their own captured graphs. The side stream (TF_GLM53_SIDE) is on, as served.
"""

from __future__ import annotations

import os

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402

CAPACITY = 2400
K = 3


@pytest.fixture(autouse=True)
def _prompt_rows_4096(monkeypatch):
    from tensorfold.families.glm_moe_dsa.cuda import runner

    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))


def _prompt(n=2100, seed=41):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 150000, (n,), generator=g).tolist()


def _replies(w, graphs, modes, comm=None):
    """{(mode, k): (reply, acceptance)} for each reuse mode, drafted at depth K and serial (k = 0)."""
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    prompt = _prompt()
    out = {}
    for mode in modes:
        if comm is not None:
            comm.barrier()                               # module knobs are shared by the rank threads
            if w.rank == 0:
                fused.MTP_REUSE = mode
            comm.barrier()
        else:
            fused.MTP_REUSE = mode
        run = Runner(w, CAPACITY + K + 1, K, graphs=graphs)
        assert run.reuse == mode
        for k in (K, 0):
            st = run.generate(prompt, 24, None, lambda t: False, lambda new: None, k)
            out[(mode, k)] = (st["out"], st["accept"])
        del run
        torch.cuda.empty_cache()
    return out


def _check(out, modes):
    serial = out[(modes[0], 0)][0]
    for mode in modes:
        assert out[(mode, 0)][0] == serial, f"reuse {mode}: serial replies differ"
        assert out[(mode, K)][0] == serial, f"reuse {mode}: drafted != serial"
        assert out[(mode, K)][1] is not None


def test_mtp_reuse_drafted_equals_serial_ranks():
    from test_glm_moe_dsa_multi import _weights

    from tensorfold.families.glm_moe_dsa.cuda import fused

    modes = (0, 1, 2)
    saved = fused.MTP_REUSE
    try:
        results = run_ranks(lambda rank, comm: _replies(_weights(rank, 4, comm), False, modes, comm), 4)
    finally:
        fused.MTP_REUSE = saved
    for out in results:
        _check(out, modes)
    assert all(r == results[0] for r in results[1:]), "ranks disagree"


def test_mtp_reuse_drafted_equals_serial_graphs():
    from test_glm_moe_dsa_multi import _Alone, _weights

    from tensorfold.families.glm_moe_dsa.cuda import fused

    modes = (0, 1, 2)
    saved = fused.MTP_REUSE
    try:
        with torch.no_grad():
            out = _replies(_weights(0, 4, _Alone()), True, modes)
    finally:
        fused.MTP_REUSE = saved
    _check(out, modes)
