"""Full GLM-5.3 copy drafts (copies.py, TF_GLM53_COPY_*; after MiaAI-Lab 0007 / 0032) on real weights (layers 0-3 +
the MTP layer), four thread ranks on one GPU: replies drafted with copies equal the serial reply bit for bit.

A truncated 4-layer target rarely quotes its prompt, so the mechanics are driven by an oracle index that proposes the
serial reply's own continuation - every draft kept, so 16-row windows, padding, the MTP backlog's flush and DFlash2's
tap passes in pieces all run - and, in a second pass, wrong at every third proposal (a partial keep). A last test runs
the real index (match 2) on a prompt that repeats itself. Needs TF_GLM53_CKPT (TF_GLM53_DFLASH_TEST: DFlash2 / auto)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from test_glm_moe_dsa_fused import DFLASH, _parts, _tokens  # noqa: E402
from threadcomm import run_ranks  # noqa: E402

TOKENS = 60
SAMPLINGS = [None, (99, 0.9, 20, 0.95, 0.0)]


@pytest.fixture(autouse=True)
def _prompt_rows_4096(monkeypatch):
    """Four rank threads share one GPU: 4096-row prompt buffers (as the other runner tests)."""
    from tensorfold.families.glm_moe_dsa.cuda import runner

    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))


class Oracle:
    """A stand-in for copies.CopyIndex: proposes the serial reply's next tokens (this rank's serial run), the middle
    one changed at every ``bad``-th proposal (0: never)."""

    def __init__(self, prompt, serial, most, bad):
        self.n0, self.reply, self.most, self.bad = len(prompt), list(serial), most, bad
        self.length, self.calls = len(prompt), 0

    def extend(self, tokens):
        self.length += len(tokens)

    def propose(self, room=None):
        k = self.most if room is None else min(self.most, int(room))
        self.calls += 1
        i = self.length - self.n0                        # reply tokens so far, the pending one included
        out = self.reply[i:i + k]
        if out and self.bad and self.calls % self.bad == 0:
            j = len(out) // 2
            out[j] = (out[j] + 1) % 150000
        return out


def _runner(rank, comm, dflash):
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner
    from tensorfold.families.glm_moe_dsa.cuda.weights import load_mtp

    cfg, r, (embed, norm, head), layers = _parts(rank, 4)
    w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, load_mtp(r, cfg))
    if dflash:
        cfg_d = json.loads((Path(DFLASH) / "config.json").read_text())
        w.tap_slot = {int(i): s for s, i in enumerate(cfg_d["dflash_config"]["target_layer_ids"])}
    return w, Runner


def _sample_fn(s):
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    return None if s is None else (lambda lg, pos: Glm53Engine._sample(None, lg, pos, s))


def _oracle_runs(monkeypatch, sampling, modes, graphs, dflash):
    """Every rank: the serial reply, then each (mode, k) drafted with oracle copies kept whole and with misses."""
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm_moe_dsa.cuda import copies

    s = Sampling(*sampling) if sampling else None
    prompt = _tokens(150, seed=43).tolist()
    mine: dict[int, tuple[list[int], int]] = {}
    monkeypatch.setattr(copies, "SETTINGS", (1, 8, 15))

    def oracle(p, match, most):                         # this rank thread's serial reply and miss rate
        serial, bad = mine[threading.get_ident()]
        return Oracle(p, serial, most, bad)
    monkeypatch.setattr(copies, "CopyIndex", oracle)

    def run(rank, comm):
        w, Runner = _runner(rank, comm, dflash)
        run_ = Runner(w, 512, 2, graphs=graphs)
        if dflash:
            from tensorfold.families.glm_moe_dsa.cuda.dflash import GlmDrafter

            run_.drafter = GlmDrafter(DFLASH, w, capacity=512)
        if graphs:
            run_.prewarm()
        fn = _sample_fn(s)
        serial = run_.generate(prompt, TOKENS, fn, lambda t: False, lambda new: None, 0, "normed/normed",
                               sampling=s)["out"]
        out = {}
        for bad in (0, 3):
            mine[threading.get_ident()] = (serial, bad)
            for mode, k in modes:
                st = run_.generate(prompt, TOKENS, fn, lambda t: False, lambda new: None, k, mode, sampling=s)
                out[(mode, bad)] = st
        return serial, out, run_.widths

    return run_ranks(run, 4)


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "top_k"])
def test_mtp_with_copies_equals_serial(monkeypatch, sampling, graphs):
    for rank, (serial, out, widths) in enumerate(_oracle_runs(monkeypatch, sampling, [("normed/normed", 2)],
                                                              graphs, dflash=False)):
        assert widths == [1, 2, 3, 8, 16], widths
        for (mode, bad), st in out.items():
            assert st["out"] == serial, (rank, mode, bad, st)
            assert st["copy_rounds"] > 0 and st["copy_accepted"] > 0, st
            if not bad:                                  # whole copies: ~16 tokens a round
                assert st["rounds"] <= TOKENS // 8, st


@pytest.mark.skipif(not DFLASH, reason="set TF_GLM53_DFLASH_TEST to a GLM-5.3 DFlash2 checkpoint")
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "top_k"])
def test_dflash_and_auto_with_copies_equal_serial(monkeypatch, sampling):
    for rank, (serial, out, widths) in enumerate(_oracle_runs(monkeypatch, sampling, [("dflash", 2), ("auto", 2)],
                                                              False, dflash=True)):
        assert 16 in widths, widths
        for (mode, bad), st in out.items():
            assert st["out"] == serial, (rank, mode, bad, st)
            assert st["copy_rounds"] > 0, st
            if mode == "auto":
                assert st["arms"]["c"] == st["copy_rounds"], st


def test_real_index_on_a_repeating_prompt_equals_serial(monkeypatch):
    """The real index (match 2, so a truncated target's reply finds matches) on a prompt that repeats a block; the
    reply must equal serial whatever it copies. MTP, and DFlash2 / auto when TF_GLM53_DFLASH_TEST is set."""
    from tensorfold.families.glm_moe_dsa.cuda import copies

    monkeypatch.setattr(copies, "SETTINGS", (1, 2, 15))
    block = _tokens(30, seed=47).tolist()
    prompt = block * 4 + block[:12]
    dflash = bool(DFLASH)
    modes = [("normed/normed", 2)] + ([("dflash", 2), ("auto", 2)] if dflash else [])

    def run(rank, comm):
        w, Runner = _runner(rank, comm, dflash)
        run_ = Runner(w, 512, 2, graphs=False)
        if dflash:
            from tensorfold.families.glm_moe_dsa.cuda.dflash import GlmDrafter

            run_.drafter = GlmDrafter(DFLASH, w, capacity=512)
        serial = run_.generate(prompt, TOKENS, None, lambda t: False, lambda new: None, 0, "normed/normed")["out"]
        off = run_.generate(prompt, TOKENS, None, lambda t: False, lambda new: None, 2, "normed/normed",
                            copies=False)
        return serial, off, {mode: run_.generate(prompt, TOKENS, None, lambda t: False, lambda new: None, k, mode)
                             for mode, k in modes}

    for serial, off, out in run_ranks(run, 4):
        assert off["out"] == serial and "copy_rounds" not in off
        for mode, st in out.items():
            assert st["out"] == serial, (mode, st)
            assert "copy_rounds" in st, st
