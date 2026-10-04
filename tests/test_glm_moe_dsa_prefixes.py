"""Full GLM-5.3 prompt reuse, host side (CPU, no weights): keep points are token-prefix-determined, a prompt resumed at
any of its points runs the chunks a fresh one runs past it, and the kept-state store picks, keeps and evicts as the
engine relies on (prefixes.py)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.families.glm_moe_dsa.cuda import prefixes
from tensorfold.families.glm_moe_dsa.cuda.prefixes import (Kept, KeptPrompts, PromptPlan, cut_chunks, even_chunks)

GMASK, SOP, SYSTEM, USER, ASSISTANT, THINK, END_THINK = 154822, 154824, 154826, 154827, 154828, 154841, 154842


def _text(rng, n):
    return [int(t) for t in rng.integers(1000, 150000, size=n)]


def _turn1(rng, system=600, user=1200):
    return [GMASK, SOP, SYSTEM] + _text(rng, system) + [USER] + _text(rng, user) + [ASSISTANT, THINK]


def _plan(gap=1024, system_min=256):
    return PromptPlan(USER, ASSISTANT, THINK, gap=gap, system_min=system_min)


# -------------------------------------------------------------------------------------------------- chunks ---
def _legacy_chunks(L0, rows=8192, short=4096):
    if L0 < 3 * rows:
        rows = min(rows, short)
    n = -(-L0 // rows)
    step = -(-L0 // n)
    return [(a, min(a + step, L0)) for a in range(0, L0, step)]


@pytest.mark.parametrize("L0", [1, 2, 7, 4096, 4097, 8192, 12000, 24575, 24576, 32768, 131072, 131073])
def test_even_chunks_from_zero_are_the_old_chunks(L0):
    assert even_chunks(0, L0, 8192, 4096) == _legacy_chunks(L0)


def test_resumed_at_a_stop_runs_the_fresh_chunks_past_it():
    rng = np.random.default_rng(1)
    for _ in range(200):
        L = int(rng.integers(2, 70000))
        stops = sorted(set(int(x) for x in rng.integers(1, L, size=int(rng.integers(0, 8)))))
        fresh = cut_chunks(0, L, stops, 8192, 4096)
        assert fresh[0][0] == 0 and fresh[-1][1] == L
        assert all(a < e and e == nxt for (a, e), (nxt, _) in zip(fresh, fresh[1:] + [(L, L)]))
        ends = {e for _, e in fresh}
        assert set(stops) <= ends                        # every stop ends a chunk: a state exists there
        for p in stops:
            assert cut_chunks(p, L, stops, 8192, 4096) == [c for c in fresh if c[0] >= p]


# ---------------------------------------------------------------------------------------------- keep points ---
def test_points_are_prefix_determined():
    """Prompts agreeing up to a point agree on every point up to it: a prompt ending in an assistant opener has its
    end as a point, and every continuation (reasoning kept or dropped) keeps it."""
    rng = np.random.default_rng(2)
    plan = _plan()
    p1 = _turn1(rng)
    pts1 = plan.points(p1)
    assert pts1 == [603, len(p1)]                       # the system block's end, the opener (after <think>)
    kept = p1 + _text(rng, 700) + [END_THINK] + _text(rng, 400) + [USER] + _text(rng, 50) + [ASSISTANT, THINK]
    dropped = p1 + [END_THINK] + _text(rng, 400) + [USER] + _text(rng, 1100) + [ASSISTANT, THINK]
    for p2 in (kept, dropped):
        pts2 = plan.points(p2)
        assert [p for p in pts2 if p <= len(p1)] == pts1
        assert pts2[-1] == len(p2)


def test_points_follow_the_gap_and_the_system_minimum():
    rng = np.random.default_rng(3)
    short = [GMASK, SOP, SYSTEM] + _text(rng, 20) + [USER] + _text(rng, 30) + [ASSISTANT, THINK]
    assert _plan().points(short) == []                  # short prompts are cut as before: no points
    assert _plan(gap=16, system_min=8).points(short) == [23, len(short)]
    many = short[:]
    for _ in range(10):
        many += _text(rng, 300) + [USER] + _text(rng, 10) + [ASSISTANT, THINK]
    pts = _plan(gap=1024).points(many)
    assert all(b - a >= 1024 for a, b in zip(pts, pts[1:]))
    for p in pts:                                         # each is right after an <|assistant|><think> opener
        assert many[p - 2:p] == [ASSISTANT, THINK]
    # random prefixes of a long prompt: their points are the long prompt's points inside them
    for cut in rng.integers(1, len(many), size=50):
        head = many[:int(cut)]
        assert plan_points_inside(_plan(gap=1024), many, head)


def plan_points_inside(plan, full, head):
    inside = [p for p in plan.points(full) if p < len(head)]
    mine = [p for p in plan.points(head) if p < len(head)]
    return inside == mine


def test_system_point():
    rng = np.random.default_rng(4)
    plan = _plan()
    p1 = _turn1(rng)
    assert plan.system_point(plan.points(p1), p1) == 603
    nosys = [GMASK, SOP] + _text(rng, 10) + [USER] + _text(rng, 2000) + [ASSISTANT, THINK]
    assert plan.system_point(plan.points(nosys), nosys) is None


# ------------------------------------------------------------------------------------------------- the store ---
class _T:
    """A tensor's byte count, no device."""

    def __init__(self, nbytes):
        self.nbytes = nbytes

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1

    def clone(self):
        return _T(self.nbytes)


def _kept(ids, mode="normed", head=False, shared=False, size=10):
    return Kept(np.asarray(ids, dtype=np.int64), mode, carry=_T(size), ring=[_T(size)],
                head=_T(size) if head else None, shared=shared)


def test_best_resumes_only_at_the_prompts_points_or_replays_whole():
    s = KeptPrompts(1 << 30)
    a = list(range(100))
    s.live = np.asarray(a + [7] * 50, dtype=np.int64)
    for n, head in ((40, False), (60, False), (100, True)):
        assert s.remember(_kept(a[:n], head=head))
    prompt = np.asarray(a + [5, 6], dtype=np.int64)
    assert s.best(prompt, [40, 60, 100], "normed").n == 100     # the longest at one of the prompt's points
    assert s.best(prompt, [40, 60], "normed").n == 60           # 100 is not a point of this prompt
    assert s.best(prompt, [40, 60], "raw") is None              # another MTP input: not resumed
    assert s.best(np.asarray(a, dtype=np.int64), [40], "normed").n == 100   # the whole prompt with its head row
    s.loose = True
    assert s.best(prompt, [], "normed").n == 100
    s.loose = False
    s.live = np.asarray(a[:50], dtype=np.int64)                 # rows past 50 unknown (a failed request)
    assert s.best(prompt, [40, 60, 100], "normed").n == 40
    assert s.named(prompt, 60) is None and s.named(prompt, 40) is not None


def test_superseded_states_go_first_then_lru():
    s = KeptPrompts(1 << 30, entries=4)
    a, b = list(range(100)), list(range(1000, 1100))
    s.live = np.asarray(a, dtype=np.int64)
    sys_a = _kept(a[:20], shared=True)
    for e in (sys_a, _kept(a[:50]), _kept(b[:80]), _kept(a[:100], head=True)):
        assert s.remember(e)
    # past the cap: a's superseded non-shared state (50) goes first, not b (the only state of its conversation)
    assert s.remember(_kept(b[:90], head=True))
    assert sorted(e.n for e in s.entries) == [20, 80, 90, 100]
    assert s.remember(_kept(list(range(5000, 5010))))         # then b's superseded 80, then a's shared 20
    assert sorted(e.n for e in s.entries) == [10, 20, 90, 100]
    assert s.remember(_kept(list(range(6000, 6010))))
    assert sorted(e.n for e in s.entries) == [10, 10, 90, 100]
    assert not any(e.shared for e in s.entries)


def test_budget_and_protection():
    s = KeptPrompts(100)
    a = list(range(100))
    s.live = np.asarray(a, dtype=np.int64)
    hit = _kept(a[:10], size=20)                                # 40 bytes
    assert s.remember(hit)
    assert s.remember(_kept(a[:30], size=20))
    assert not s.remember(_kept(a[:50], size=40), protect=(hit,))   # 80 + the hit's 40 > 100
    assert [e.n for e in s.entries] == [10, 30]                  # nothing dropped for a state that cannot fit
    assert s.remember(_kept(a[:50], size=30), protect=(hit,))   # drops 30 (least recently used), keeps the hit
    assert sorted(e.n for e in s.entries) == [10, 50]
    assert s.held() == 100
    assert not s.fit(1, protect=tuple(s.entries))


def test_overwritten_and_same_ids_replace():
    s = KeptPrompts(1 << 30)
    a, b = list(range(100)), list(range(500, 600))
    s.live = np.asarray(a, dtype=np.int64)
    for e in (_kept(a[:20]), _kept(a[:60]), _kept(a[:100], head=True)):
        s.remember(e)
    saved = _kept(b[:40])
    saved.rows = [_T(5)]
    s.remember(saved)
    prompt = np.asarray(a[:60] + [1, 2, 3], dtype=np.int64)
    assert sorted(e.n for e in s.overwritten(prompt, 60)) == [100]     # 20 and 60 stay; a saved state never
    assert sorted(e.n for e in s.overwritten(prompt, 0)) == [20, 60, 100]
    s.remember(_kept(a[:60]))
    assert sorted(e.n for e in s.entries) == [20, 40, 60, 100]
    assert len(s.entries) == 4


def test_choose_keeps_end_system_and_fork():
    rng = np.random.default_rng(5)

    class _Runner:
        class w:
            dcp = 1

    plan = _plan()
    r = prefixes.PromptReuse(_Runner(), plan, 1 << 30, 32, False)
    p1 = _turn1(rng)
    begin, stops, keeps = r.choose(p1, True, "normed")
    assert begin == 0 and stops == [603]
    assert keeps == [603 * 2 + 1]                        # the system block (shared); the end is kept anyway
    r.store.live = np.asarray(p1, dtype=np.int64)
    r.store.remember(_kept(p1[:603], shared=True))
    r.store.remember(_kept(p1, head=True))
    p2 = p1 + [END_THINK] + _text(rng, 1500) + [USER] + _text(rng, 30) + [ASSISTANT, THINK]
    begin, stops, keeps = r.choose(p2, True, "normed")
    assert begin == len(p1) and stops == []              # turn 2 resumes at turn 1's end, a point of both
    assert r.choose(p1, True, "normed")[0] == len(p1)    # an identical resend replays
    assert r.choose(p1, False, "normed") == (0, [603], [])   # the serial reference resumes and keeps nothing
    other = p1[:604] + _text(rng, 3000) + [ASSISTANT, THINK]      # the same system block and <|user|>
    begin, stops, keeps = r.choose(other, True, "normed")
    assert begin == 603 and stops == [] and keeps == []  # a new conversation on the same system block


# ---------------------------------------------------------------------------------------- concurrent streams ---
def _slot_kept(ids, slot, head=False, ring=True):
    e = _kept(ids, head=head)
    e.ring = [_T(10)] if ring else None
    e.extra["slot"] = slot
    return e


def test_slot_reuse_in_place_fork_and_overwrite():
    rng = np.random.default_rng(6)
    r = prefixes.SlotReuse(_plan(), 1 << 30, 32, False)
    st = r.store
    p1 = _turn1(rng)
    p2 = p1 + [END_THINK] + _text(rng, 1500) + [USER] + _text(rng, 30) + [ASSISTANT, THINK]
    st.lives[1] = np.asarray(p1, dtype=np.int64)
    st.remember(_slot_kept(p1[:603], 1))
    st.remember(_slot_kept(p1, 1, head=True, ring=False))       # an MTP stream's state: no drafter window
    # slot 1 free: in place; a DFlash2 stream cannot take the ring-less end state, only the system block's
    assert r.choose(p2, True, "normed", [0, 1, 2], ring=False)[:3] == (1, len(p1), -1)
    assert r.choose(p2, True, "normed", [0, 1, 2], ring=True)[:3] == (1, 603, -1)
    # slot 1 busy: the emptiest free slot gets a copy of slot 1's rows
    st.remember(_slot_kept(list(range(10)), 2))
    assert r.choose(p2, True, "normed", [0, 2], ring=False)[:3] == (0, len(p1), 1)
    # the serial reference resumes nothing and takes the free slot used least recently (slot 1: its states are older)
    assert r.choose(p2, False, "normed", [1, 2], ring=False)[:3] == (1, 0, -1)
    # admitted in place at p1's end: slot 1 keeps its prefixes of p2; another prompt there drops them
    assert r.admit(p2, 1, len(p1), -1, None).n == len(p1)
    assert sorted(e.n for e in st.in_slot(1)) == [603, len(p1)]
    r.filled(1, p2)
    other = _text(rng, 50)
    r.admit(other, 1, 0, -1, None)
    assert st.in_slot(1) == [] and [e.n for e in st.in_slot(2)] == [10]
