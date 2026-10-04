"""Full GLM-5.3 prompt reuse (TF_GLM53_PROMPT_REUSE=1, prefixes.py) on real weights (first layers only), four thread
ranks on one GPU: a next turn resumes at the previous prompt's end and replies exactly as the cold serial reference
(cut at the same keep points, nothing resumed), an identical resend replays its kept end with nothing prefilled, and
kept states survive another conversation (saved rows, put back). Needs TF_GLM53_CKPT."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

CKPT = os.environ.get("TF_GLM53_CKPT", "")
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
              pytest.mark.skipif(not CKPT, reason="set TF_GLM53_CKPT to a GLM-5.3 EXL3 checkpoint")]

from threadcomm import run_ranks  # noqa: E402

GMASK, SOP, SYSTEM, USER, ASSISTANT, THINK, END_THINK = 154822, 154824, 154826, 154827, 154828, 154841, 154842


@pytest.fixture(autouse=True)
def _reuse_small(monkeypatch):
    """Reuse on, keep points every 16 tokens (short test prompts), reproducible prompts (slots16 expert sums, the
    exact reduce, no overlap), 4096-row prompt buffers (four rank threads share one GPU), no concurrent graph
    capture."""
    from tensorfold.cuda.exl3 import prompt_experts
    from tensorfold.families.glm_moe_dsa.cuda import engine, fused, prefixes, runner

    monkeypatch.setenv("TF_GLM53_PROMPT_REUSE", "1")
    monkeypatch.setenv("TF_GLM53_REUSE_GAP", "16")
    monkeypatch.setattr(prefixes, "SYSTEM_MIN", 8)
    monkeypatch.setattr(runner, "PROMPT_ROWS", min(runner.PROMPT_ROWS, 4096))
    monkeypatch.setattr(engine, "GRAPHS", False)
    monkeypatch.setattr(fused, "PREFILL_REDUCE", "rs")
    monkeypatch.setattr(fused, "PROMPT_OVERLAP", False)
    monkeypatch.setattr(prompt_experts, "DETERMINISTIC", True)
    monkeypatch.setattr(prompt_experts, "DET_MODE", 4)


def _text(rng, n):
    return [int(t) for t in rng.integers(1000, 150000, size=n)]


def _turns():
    rng = np.random.default_rng(7)
    p1 = [GMASK, SOP, SYSTEM] + _text(rng, 40) + [USER] + _text(rng, 60) + [ASSISTANT, THINK]
    p2 = p1 + [END_THINK] + _text(rng, 30) + [USER] + _text(rng, 20) + [ASSISTANT, THINK]
    other = p1[:44] + _text(rng, 90) + [ASSISTANT, THINK]          # the same system block, another question
    return p1, p2, other


def _engine(rank, comm):
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    return Glm53Engine(CKPT, rank=rank, master="", port=0, context=1024, comm=comm, layers=4, mtp_drafts=2)


def _conversation(rank, comm, sampling):
    e = _engine(rank, comm)
    p1, p2, other = _turns()
    if rank:
        e.follow(requests=6)
        return None
    runs = []
    for prompt, draft in ((p1, True), (p2, True), (p2, False), (p2, True), (other, True), (p2, True)):
        got = []
        stats = e.generate(prompt, 12, sampling, got.extend, draft=draft, stop_eos=False)
        runs.append((got, stats))
    return runs, len(p1), len(p2)


@pytest.mark.parametrize("sampled", [False, True])
def test_next_turn_resend_and_other_conversation_equal_cold(sampled):
    from tensorfold.engine.exact_sampling import Sampling

    s = Sampling(4321, 0.8, 20, 0.95, 0.0) if sampled else None
    runs, n1, n2 = run_ranks(lambda r, c: _conversation(r, c, s), 4)[0]
    (_, s1), (warm, s2), (cold, s3), (again, s4), (_, s5), (back, s6) = runs
    assert s1["cached"] == 0
    assert s2["cached"] == n1, s2                        # turn 2 resumed at turn 1's end (a point of both)
    assert s3["cached"] == 0                             # "draft": false: the cold reference
    assert warm == cold, (warm, cold)
    assert s4["cached"] == n2 and s4.get("replay"), s4   # the identical resend: its kept end, nothing prefilled
    assert again == warm
    assert s5["cached"] == 43                            # another conversation on the same system block
    assert s6["cached"] == n2 and back == warm, (s6, back, warm)   # p2's rows were saved, then put back


def _cold_then_resend(rank, comm):
    e = _engine(rank, comm)
    p1, p2, _ = _turns()
    if rank:
        e.follow(requests=5)
        return None
    runs = []
    for prompt, draft in ((p1, True), (p2, True), (p2, False), (p2, True), (p1 + [END_THINK, USER, ASSISTANT, THINK],
                                                                             True)):
        got = []
        runs.append((got, e.generate(prompt, 12, None, got.extend, draft=draft, stop_eos=False)))
    return runs, len(p1), len(p2)


def test_cold_reference_keeps_the_states_it_rewrites(monkeypatch):
    """No saved copy fits (a 60K-token conversation's states past TF_GLM53_CACHE_GIB on the cluster): the cold
    reference of turn 2 rewrites turn 2's kept states at its own cut points and keeps them again from its own rows,
    so the identical resend still replays, and a next turn still resumes at turn 1's end (renewed in place)."""
    from tensorfold.families.glm_moe_dsa.cuda import prefixes

    monkeypatch.setattr(prefixes, "row_bytes", lambda st, n, dcp=1: 1 << 50)     # never a copy
    runs, n1, n2 = run_ranks(_cold_then_resend, 4)[0]
    (_, s1), (warm, s2), (cold, s3), (again, s4), (_, s5) = runs
    assert s2["cached"] == n1 and s3["cached"] == 0
    assert warm == cold, (warm, cold)
    assert s4["cached"] == n2 and s4.get("replay"), s4
    assert again == cold
    assert s5["cached"] == n1, s5                        # turn 1's end, renewed by the cold run of turn 2


def _flush(rank, comm):
    from tensorfold.families.glm_moe_dsa.cuda import runner

    e = _engine(rank, comm)
    p1, p2, _ = _turns()
    if rank:
        e.follow(requests=3)
        return None
    out = []
    for i, prompt in enumerate((p1, p2, p2)):
        if i == 2:
            flag = os.path.join(os.path.dirname(runner.PROFILE_FLAG), "REUSE_FLUSH")
            os.makedirs(os.path.dirname(flag), exist_ok=True)
            open(flag, "w").close()
        got = []
        out.append((got, e.generate(prompt, 8, None, got.extend, stop_eos=False)))
    return out


def test_flush_file_forgets_every_state(monkeypatch, tmp_path):
    from tensorfold.families.glm_moe_dsa.cuda import runner

    monkeypatch.setattr(runner, "PROFILE_FLAG", str(tmp_path / "PROFILE"))
    (_, a), (warm, b), (cold, c) = run_ranks(_flush, 4)[0]
    assert a["cached"] == 0 and b["cached"] > 0 and c["cached"] == 0
    assert warm == cold


# ---------------------------------------------------------------------------------------- concurrent streams ---
def _admit_all(dec, items):
    """Rank 0: items (name, prompt, tokens, sampling, draft, round it arrives at) through the decoder's rounds;
    returns each name's tokens and the prompt tokens it found cached."""
    from tensorfold.cuda.streams import Stream

    got, streams = {}, {}
    pending = sorted(items, key=lambda x: x[5])
    rnd = 0
    while pending or dec.live():
        while pending and pending[0][5] <= rnd and dec.free:
            name, prompt, n, s, draft, _ = pending.pop(0)
            got[name] = []
            st = Stream(list(prompt), n, s, draft=draft, stop_eos=False)
            st.emit = lambda new, name=name: got[name].extend(new)
            dec.admit(st)
            streams[name] = st
        dec.finish(dec.round())
        rnd += 1
    return {name: (got[name], streams[name].cached) for name in got}


def _slots(rank, comm, sampling):
    from test_glm_moe_dsa_multi import K, _sample_fn, _weights

    from tensorfold.families.glm_moe_dsa.cuda.multi import GlmMultiDecoder
    from tensorfold.families.glm_moe_dsa.cuda.prefixes import PromptPlan, SlotReuse
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    w = _weights(rank, 4, comm)
    run = Runner(w, 1024 + K + 1, K, graphs=False, slots=2)
    dec = GlmMultiDecoder(run, rank=rank, world=4, comm=comm, limit=1024, eos=tuple(w.cfg.eos_token_ids),
                          sample=_sample_fn())
    dec.reuse = SlotReuse(PromptPlan(USER, ASSISTANT, THINK, gap=16, system_min=8), 1 << 30, 32, False)
    p1, p2, _ = _turns()
    p3 = p2 + [END_THINK] + _text(np.random.default_rng(9), 40) + [USER] + _text(np.random.default_rng(10), 9) + \
        [ASSISTANT, THINK]
    out = None
    if rank:
        dec.follow()
    else:
        out = [_admit_all(dec, [("a", p1, 10, sampling, True, 0)]),
               _admit_all(dec, [("b", p2, 10, sampling, True, 0)]),
               _admit_all(dec, [("cold", p2, 10, sampling, False, 0)]),
               _admit_all(dec, [("replay", p2, 10, sampling, True, 0)]),
               # p3 resumes in place at p2's end; p2 sent again meanwhile replays from a copy of that busy slot
               _admit_all(dec, [("p3", p3, 10, sampling, True, 0), ("fork", p2, 10, sampling, True, 1)]),
               _admit_all(dec, [("p3cold", p3, 10, sampling, False, 0)])]
        dec.stop()
    torch.cuda.synchronize()
    comm.barrier()
    return out, len(p1), len(p2)


@pytest.mark.parametrize("sampled", [False, True])
def test_concurrent_streams_resume_in_place_and_from_busy_slots(monkeypatch, sampled):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm_moe_dsa.cuda import multi

    monkeypatch.setattr(multi, "FILL_MIN_ROWS", 0)       # resumed fills in layer steps too
    s = Sampling(77, 0.8, 20, 0.95, 0.0) if sampled else None
    out, n1, n2 = run_ranks(lambda r, c: _slots(r, c, s), 4)[0]
    merged = {k: v for part in out for k, v in part.items()}
    assert merged["a"][1] == 0
    assert merged["b"][1] == n1                          # turn 2 in place in turn 1's slot
    assert merged["cold"][1] == 0 and merged["b"][0] == merged["cold"][0]
    assert merged["replay"][1] == n2 and merged["replay"][0] == merged["b"][0]
    assert merged["p3"][1] == n2 and merged["p3"][0] == merged["p3cold"][0]
    assert merged["fork"][1] == n2 and merged["fork"][0] == merged["b"][0]
