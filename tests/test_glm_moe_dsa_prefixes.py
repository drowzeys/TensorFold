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


def test_loaded_rows_keep_only_prefixes_cut_alike():
    """A saved state loaded over the live rows: a live prefix state survives only at one of the prompt's points (an
    end state that was not a point had its rows cut differently in the loaded prompt)."""
    s = KeptPrompts(1 << 30)
    a = list(range(100))
    s.live = np.asarray(a[:50], dtype=np.int64)
    for n in (20, 50):
        s.remember(_kept(a[:n]))
    prompt = np.asarray(a + [1], dtype=np.int64)
    assert s.overwritten(prompt, 100) == []
    gone = s.overwritten(prompt, 100, same=lambda e: e.n in {20})
    assert [e.n for e in gone] == [50]


# ----------------------------------------------------------------------- one stream: copies and renewed states ---
class _CpuRows:
    """A one-stream Runner's caches on the host: what prefixes.row_views reads (kc, ic, mkc, mic)."""

    def __init__(self, torch, cap=512):
        self.kc = [torch.zeros((cap, 8), dtype=torch.float32) for _ in range(2)]
        self.ic = {0: torch.zeros((cap, 4), dtype=torch.float32), 1: torch.zeros((cap, 4), dtype=torch.float32)}
        self.mkc = torch.zeros((cap, 8), dtype=torch.float32)
        self.mic = torch.zeros((cap, 4), dtype=torch.float32)


class _CpuRunner:
    def __init__(self, torch):
        from types import SimpleNamespace

        self.torch = torch
        self.st = _CpuRows(torch)
        self.carry = torch.zeros((8,), dtype=torch.float32)
        self.drafter = None
        self.w = SimpleNamespace(rank=0, dcp=1)
        self.prefilled = []                               # rows each request ran

    def gen(self, prompt, salt):
        """Runner.generate as prefixes.PromptReuse.run drives it: rows [begin, L) cut at the stops (each row's
        value: its token and the request's salt), keep(n) at each cut, keep(L, logits) at the end; a replay runs
        nothing."""
        torch = self.torch
        arr = np.asarray(prompt, dtype=np.int64)
        L = len(arr)

        def gen(begin, stops, keep, head):
            if head is not None:
                self.prefilled.append(0)
                return {"out": []}
            cuts = sorted(set(int(p) for p in stops if begin < p < L))
            a = begin
            for e in cuts + [L]:
                vals = torch.as_tensor((arr[a:e] % 997 + 1000 * salt).astype(np.float32))[:, None]
                for t in self.st.kc + [self.st.ic[i] for i in sorted(self.st.ic)] + [self.st.mkc, self.st.mic]:
                    t[a:e] = vals
                self.carry.fill_(float(arr[e - 1] % 997 + 1000 * salt))
                if e < L:
                    keep(e)
                a = e
            self.prefilled.append(L - begin)
            keep(L, torch.full((1, 4), float(salt)))
            return {"out": []}
        return gen


def _ask(r, rn, prompt, draft=True, salt=0):
    begin, stops, keeps = r.choose(prompt, draft, "normed")
    return r.run(prompt, begin, stops, keeps, 0, draft, "normed", rn.gen(prompt, salt))


def _reuse(budget):
    torch = pytest.importorskip("torch")
    rn = _CpuRunner(torch)
    return prefixes.PromptReuse(rn, _plan(gap=16, system_min=8), budget, 32, False), rn, torch


def _small_turns(rng):
    p1 = [GMASK, SOP, SYSTEM] + _text(rng, 20) + [USER] + _text(rng, 30) + [ASSISTANT, THINK]
    p2 = p1 + [END_THINK] + _text(rng, 30) + [USER] + _text(rng, 10) + [ASSISTANT, THINK]
    return p1, p2


def test_cold_reference_renews_the_states_it_rewrites():
    """The cold reference ("draft": false) of a kept prompt rewrites its rows cut alike: it keeps those states again
    from its own rows - nothing is copied, nothing dropped - and the identical resend after it still replays, with a
    budget too small for any copy (a long conversation's states on the cluster)."""
    r, rn, _ = _reuse(1024)
    p1, p2 = _small_turns(np.random.default_rng(11))
    assert _ask(r, rn, p1)["cached"] == 0
    assert _ask(r, rn, p2, salt=1)["cached"] == len(p1)
    cold = _ask(r, rn, p2, draft=False, salt=2)
    assert cold["cached"] == 0 and rn.prefilled[-1] == len(p2)
    assert all(e.rows is None for e in r.store.entries)          # renewed in place, not saved
    end = [e for e in r.store.entries if e.n == len(p2)]
    assert len(end) == 1 and end[0].head is not None and float(end[0].head[0, 0]) == 2.0   # the cold run's own row
    assert float(end[0].carry[0]) == float(p2[-1] % 997 + 2000)
    again = _ask(r, rn, p2, salt=3)
    assert again["cached"] == len(p2) and again.get("replay") and rn.prefilled[-1] == 0
    nxt = p2 + [END_THINK] + _text(np.random.default_rng(12), 20) + [USER, ASSISTANT, THINK]
    assert _ask(r, rn, nxt, salt=4)["cached"] == len(p2)  # turn 3 resumes at turn 2's renewed prompt end


def test_another_conversation_costs_one_copy_shared_by_the_shorter_states():
    r, rn, torch = _reuse(1 << 30)
    rng = np.random.default_rng(13)
    p1, p2 = _small_turns(rng)
    _ask(r, rn, p1)
    _ask(r, rn, p2, salt=1)
    chain = sorted(e.n for e in r.store.entries)
    assert len(chain) >= 3                               # the system block, turn 1's end, turn 2's end
    rows_p2 = [t[:len(p2)].clone() for t in rn.st.kc]
    other = [GMASK, SOP, SYSTEM] + _text(rng, 60) + [USER] + _text(rng, 40) + [ASSISTANT, THINK]
    _ask(r, rn, other, salt=5)
    p2_states = [e for e in r.store.entries if prefixes.is_prefix(e.ids, np.asarray(p2))]
    copies = {id(e.saved) for e in p2_states}
    assert len(copies) == 1 and None not in [e.saved for e in p2_states]   # one copy, every state reads it
    longest = max(p2_states, key=lambda e: e.n)
    assert r.store.held() == sum(e.held() for e in r.store.entries) + longest.saved.nbytes
    back = _ask(r, rn, p2, salt=6)                       # replays turn 2's end: its rows come back
    assert back.get("replay") and back["cached"] == len(p2)
    assert all(torch.equal(t[:len(p2)], old) for t, old in zip(rn.st.kc, rows_p2))
    assert all(e.rows is None for e in r.store.entries   # the shorter states that shared the copy: live again
               if prefixes.is_prefix(e.ids, np.asarray(p2)))
    assert len({id(e.saved) for e in r.store.entries if e.saved is not None}) == 1    # the other conversation's


def test_the_longest_state_that_fits_is_copied():
    """A budget that holds the system block's state but not the conversation's end: the end is dropped, the system
    block is copied (a new conversation on it still resumes)."""
    rng = np.random.default_rng(14)
    p1 = [GMASK, SOP, SYSTEM] + _text(rng, 40) + [USER] + _text(rng, 200) + [ASSISTANT, THINK]
    per_token = (2 * 8 + 2 * 4 + 8 + 4) * 4
    r, rn, _ = _reuse(60 * per_token)
    _ask(r, rn, p1)
    other = p1[:44] + _text(rng, 100) + [ASSISTANT, THINK]
    q = [GMASK, SOP] + _text(rng, 300)
    _ask(r, rn, q, salt=1)
    assert sorted(e.n for e in r.store.entries if e.saved is not None) == [43]
    assert _ask(r, rn, other, salt=2)["cached"] == 43


def test_thinking_off_prompt_is_cut_after_its_think_opener():
    """Thinking off, the prompt ends "<|assistant|><think></think>": its point (after <think>) is one token short of
    the end - a 1-row tail chunk - and the next turn (that history rendered alike) resumes there."""
    r, rn, _ = _reuse(1 << 30)
    rng = np.random.default_rng(15)
    p1 = [GMASK, SOP, SYSTEM] + _text(rng, 4) + [USER] + _text(rng, 60) + [ASSISTANT, THINK, END_THINK]
    begin, stops, keeps = r.choose(p1, True, "normed")
    assert stops == [len(p1) - 1] and keeps == [(len(p1) - 1) * 2]
    assert cut_chunks(0, len(p1), stops, 8192, 4096)[-1] == (len(p1) - 1, len(p1))
    _ask(r, rn, p1)
    p2 = p1 + _text(rng, 20) + [USER] + _text(rng, 30) + [ASSISTANT, THINK, END_THINK]
    assert _ask(r, rn, p2, salt=1)["cached"] == len(p1) - 1
