"""Full GLM-5.3's DSpark drafter (dspark.py, TF_GLM53_DSPARK) on real weights: the target's layers 0-3 + the MTP layer,
four thread ranks on one GPU, the RedHatAI/GLM-5.3-speculator.dspark checkpoint (TF_GLM53_DSPARK_TEST).

1. DSpark-drafted replies equal the serial reply bit for bit - greedy and sampled (keyed draws), eager (four rank
   threads) and CUDA graphs (one thread as rank 0 of four, ``_Alone``: thread collectives cannot be captured), every
   cut policy, the fixed one forcing full 9-row verify windows (8 drafts + the pending token);
2. the captured block and tap passes give the eager passes' candidates and confidence words;
3. prompt reuse's ring windows put back give the same candidates;
4. with DFlash2 loaded too (TF_GLM53_DFLASH_TEST): the union of both drafters' taps, and DFlash2, DSpark and auto
   requests on one runner all equal serial.

Only layer 1's taps exist on a 4-layer target (the others stay zero), so drafts are poor; exactness is the point.
Needs TF_GLM53_CKPT and TF_GLM53_DSPARK_TEST."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
DSPARK = os.environ.get("TF_GLM53_DSPARK_TEST", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint"),
              pytest.mark.skipif(not DSPARK, reason="set TF_GLM53_DSPARK_TEST to RedHatAI/GLM-5.3-speculator.dspark")]

import test_glm_moe_dsa_multi as base  # noqa: E402
from test_glm_moe_dsa_fused import DFLASH, _parts, _tokens  # noqa: E402
from threadcomm import run_ranks  # noqa: E402

TOKENS = 48
SAMPLINGS = [None, (99, 0.9, 20, 0.95, 0.0), (7, 0.8, 0, 0.9, 0.05)]


@pytest.fixture(autouse=True)
def _prompt_rows_4096(monkeypatch):
    """Four rank threads share one GPU: 4096-row prompt buffers (as the other runner tests)."""
    from tensorfold.families.glm_moe_dsa.cuda import runner

    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))


def _settings(**over):
    from tensorfold.families.glm_moe_dsa.cuda import dspark_host

    s = dspark_host.settings(8)
    s.update(over)
    return s


def _setup(rank, comm, graphs, settings, dflash=False):
    """A rank's runner with DSpark (and DFlash2 when ``dflash``) attached as the engine attaches them."""
    from tensorfold.families.glm_moe_dsa.cuda import dspark_host, fused
    from tensorfold.families.glm_moe_dsa.cuda.dspark import SparkDrafter
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner
    from tensorfold.families.glm_moe_dsa.cuda.weights import load_mtp

    cfg, r, (embed, norm, head), layers = _parts(rank, 4)
    w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, load_mtp(r, cfg))
    dlayers = None
    if dflash:
        cfg_d = json.loads((Path(DFLASH) / "config.json").read_text())
        dlayers = [int(i) for i in cfg_d["dflash_config"]["target_layer_ids"]]
    scfg = dspark_host.DSparkConfig.read(DSPARK)
    w.tap_slot, (dcols, scols) = dspark_host.tap_plan([dlayers, scfg.tap_layers])
    run_ = Runner(w, 512, 2, graphs=graphs, draft_rows=scfg.block + 1,
                  extra_bytes=dspark_host.memory(scfg, 4)["total"])
    if dflash:
        from tensorfold.families.glm_moe_dsa.cuda.dflash import GlmDrafter

        run_.drafter = GlmDrafter(DFLASH, w, capacity=512)
        run_.drafter.tap_cols = dcols
        if graphs:
            run_.drafter.capture()
    sd = SparkDrafter(DSPARK, w, capacity=512, settings=settings, tap_cols=scols)
    if graphs:
        sd.capture()
    run_.dspark = sd
    if graphs:
        run_.prewarm()                                   # every width 1..9 captured; the DSpark cost calibration
    return w, run_


def _on_ranks(run, graphs: bool) -> list:
    """Eager: four rank threads. Graphs: one thread as rank 0 of four (``_Alone``: device copies, capturable)."""
    if not graphs:
        return run_ranks(run, 4)
    with torch.no_grad():
        return [run(0, base._Alone())]


def _sample_fn(s):
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    return None if s is None else (lambda lg, pos: Glm53Engine._sample(None, lg, pos, s))


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("sampling", SAMPLINGS, ids=["greedy", "top_k", "nucleus"])
def test_dspark_drafted_equals_serial(sampling, graphs):
    from tensorfold.engine.exact_sampling import Sampling

    s = Sampling(*sampling) if sampling else None
    prompt = _tokens(150, seed=61).tolist()
    policies = [("fixed", None), ("confidence", 0.3), ("confidence", 0.9), ("cost", None)]

    def run(rank, comm):
        w, run_ = _setup(rank, comm, graphs, _settings(policy="fixed"))
        fn = _sample_fn(s)
        serial = run_.generate(prompt, TOKENS, fn, lambda t: False, lambda new: None, 0, "normed/normed",
                               sampling=s, copies=False)
        out = {}
        for policy, conf in policies:
            run_.dspark.settings = _settings(policy=policy, **({"confidence": conf} if conf is not None else {}))
            out[(policy, conf)] = run_.generate(prompt, TOKENS, fn, lambda t: False, lambda new: None, 2, "dspark",
                                                sampling=s, copies=False)
        return serial["out"], out, run_.widths, run_.round_ms

    for rank, (serial, out, widths, round_ms) in enumerate(_on_ranks(run, graphs)):
        assert 9 in widths, widths                       # 8 drafts + the pending token
        if graphs:
            assert round_ms is not None and len(round_ms) == 9, round_ms
        for key, st in out.items():
            assert st["out"] == serial, (rank, key, st)
            assert st["mtp_mode"] == "dspark" and st["policy"] == key[0], st
        fixed = out[("fixed", None)]
        assert fixed["depth"] == 8 and fixed["accept"] is not None, fixed


def test_graph_passes_equal_eager_passes():
    """Two drafters on the same taps - one eager, one captured - give the same candidates, base logits and
    confidence words (the tap passes in pieces of 1..8 rows and 64-row prompt pieces, then block passes). One thread
    as rank 0 of four (graphs)."""
    from tensorfold.families.glm_moe_dsa.cuda import dspark_host
    from tensorfold.families.glm_moe_dsa.cuda.dspark import SparkDrafter

    def run(rank, comm):
        from tensorfold.families.glm_moe_dsa.cuda import fused

        cfg, _, (embed, norm, head), layers = _parts(rank, 4)
        w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, None)
        scfg = dspark_host.DSparkConfig.read(DSPARK)
        w.tap_slot, _ = dspark_host.tap_plan([None, scfg.tap_layers])
        g = torch.Generator(device="cpu").manual_seed(100 + 0)          # the same taps on every rank
        taps = (torch.randn((211, 5 * cfg.hidden_size), generator=g) * 0.5).to(torch.bfloat16).cuda()
        eager = SparkDrafter(DSPARK, w, capacity=512, settings=_settings(policy="fixed"))
        graph = SparkDrafter(DSPARK, w, capacity=512, settings=_settings(policy="fixed"))
        graph.capture()
        got = []
        for dr in (eager, graph):
            dr.add_taps(taps[:150])                      # 64-row pieces (eager in both)
            for a in range(150, 211, 5):                 # kept rows of decode rounds: 1..8-row pieces (graphs)
                dr.add_taps(taps[a:min(211, a + 5)])
            got.append([dr.candidates(t, 8) for t in (11, 1234, 154000)])
        return got

    for eager, graph in _on_ranks(run, True):
        for (t1, v1, c1), (t2, v2, c2) in zip(eager, graph):
            assert (t1 == t2).all() and (v1 == v2).all() and (c1 == c2).all()


def test_ring_window_round_trip():
    """prefixes.drafter_windows at n, the drafter moved on past the ring's reach (its slots of the window
    overwritten), put back: the next block pass equals the one at n (prompt reuse's kept drafter state)."""
    from types import SimpleNamespace

    from tensorfold.families.glm_moe_dsa.cuda import dspark_host, prefixes
    from tensorfold.families.glm_moe_dsa.cuda.dspark import SparkDrafter

    def run(rank, comm):
        from tensorfold.families.glm_moe_dsa.cuda import fused

        cfg, _, (embed, norm, head), layers = _parts(rank, 4)
        w = fused.Weights(cfg, rank, 4, comm, embed, norm, head, layers, None)
        scfg = dspark_host.DSparkConfig.read(DSPARK)
        w.tap_slot, _ = dspark_host.tap_plan([None, scfg.tap_layers])
        dr = SparkDrafter(DSPARK, w, capacity=8192, settings=_settings(policy="fixed"))
        rn = SimpleNamespace(drafters=[dr])
        g = torch.Generator(device="cpu").manual_seed(7)
        taps = (torch.randn((4500, 5 * cfg.hidden_size), generator=g) * 0.5).to(torch.bfloat16).cuda()
        dr.add_taps(taps[:2400])
        want = dr.candidates(42, 8)
        kept = prefixes.drafter_windows(rn, 2400)
        dr.add_taps(taps[2400:])                         # slots of positions 352.. rewritten (ring of 4096)
        dr.reset()
        prefixes.put_drafter_windows(rn, 2400, kept)
        return want, dr.candidates(42, 8), len(kept)

    for (t1, v1, c1), (t2, v2, c2), n in run_ranks(run, 4):
        assert n == 6                                    # K and V of three draft layers
        assert (t1 == t2).all() and (v1 == v2).all() and (c1 == c2).all()


@pytest.mark.skipif(not DFLASH, reason="set TF_GLM53_DFLASH_TEST to a GLM-5.3 DFlash2 checkpoint")
@pytest.mark.parametrize("sampling", SAMPLINGS[:2], ids=["greedy", "top_k"])
def test_dspark_beside_dflash2_equals_serial(sampling):
    """Both drafters loaded (union taps: DFlash2's layout first, DSpark's columns picked out): DFlash2, DSpark and auto
    requests on one runner, in turn, all give the serial reply; each drafter keeps taking every round's kept rows."""
    from tensorfold.engine.exact_sampling import Sampling

    s = Sampling(*sampling) if sampling else None
    prompt = _tokens(150, seed=67).tolist()

    def run(rank, comm):
        w, run_ = _setup(rank, comm, False, _settings(), dflash=True)
        fn = _sample_fn(s)
        serial = run_.generate(prompt, TOKENS, fn, lambda t: False, lambda new: None, 0, "normed/normed",
                               sampling=s, copies=False)["out"]
        got = {mode: run_.generate(prompt, TOKENS, fn, lambda t: False, lambda new: None, 2, mode, sampling=s,
                                   copies=False)["out"] for mode in ("dspark", "dflash", "auto", "dspark")}
        return serial, got, len(w.tap_slot)

    for serial, got, slots in run_ranks(run, 4):
        assert slots > 5, slots
        for mode, out in got.items():
            assert out == serial, mode
