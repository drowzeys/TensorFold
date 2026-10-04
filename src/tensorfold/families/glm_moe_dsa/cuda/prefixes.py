"""Prompt-state reuse for full GLM-5.3 (one stream): kept prompt states by token ids, resumed instead of prefilled.

Our prompt path is not chunk-invariant (bf16 ring reductions, tensor-core prompt GEMMs, two micro-batches, the prompt
experts' arrival-order sums), so a prompt's rows depend on where its chunks start. A kept state is therefore only
resumed where every prompt that shares it is cut alike, as MiaAI-Lab's GLM-5.3-Flash patch 0008-glm-prompt-grid does
it: a prompt prefills to each of its keep points as a prompt of that length would, (maybe) keeps that state, and goes
on from it as a later request resuming there would. Our keep points are a function of the tokens before them
(``PromptPlan.points``: the end of the system block, then assistant openers at least ``gap`` tokens apart), so a fresh
prompt and one resumed at any of its points run the same chunks - fresh == resumed without chunk invariance, as far as
a prefill is reproducible at all (``TF_EXL3_PROMPT_DET=slots16``; with the default arrival-order expert sums a
resumed prompt is one valid fresh prefill, as any cold one is).

What a state at n keeps (``Kept``): the target's latent and index-key rows [0, n) and the MTP layer's [0, n - 1) - left
in the live caches, copied out only before another conversation overwrites them (``save_rows``: one copy a
conversation, its shorter states read the first rows of it) - the MTP carry (the target hidden n - 1), DFlash2's sliding
window (``ring_window``: decode overwrites the ring), and at a prompt's own end the head's logits row, so an identical
prompt samples its first token with nothing prefilled (0042-glm-prompt-replay). A request that rewrites a kept state's
rows at one of its own cut points (the cold reference of the same prompt, a resume in another MTP mode) keeps that
state again from its own rows instead of saving it (``PromptReuse.run``). Kept points besides the end
(0015-glm-shared-prefix): the system block's end, the last point short of the end, and where the prompt stops sharing a
kept one. Past ``TF_GLM53_CACHE_ENTRIES`` or the byte budget a conversation's superseded shorter states go first, then
the least recently used (0063-glm-kept-cap-superseded-first).

What a cold prompt (nothing resumed) pays: the chunks cut at its points (a 1-row tail chunk when it ends in
"<|assistant|><think></think>", thinking off: its point is after <think>; a system block's own chunk), one device copy
of the previous conversation's longest state that fits the budget (none when it fits nowhere or the new prompt rewrites
it at its own points), and the host bookkeeping.

Every rank keeps the same store: rank 0 picks the resume point and the keep points and shares them with the request;
every rank then saves, drops and keeps alike (byte counts do not depend on the rank, the budget is the ranks' least).

MiaAI-Lab patches 0008 / 0015 / 0025 / 0042 / 0063 (Apache-2.0), adapted to glm_moe_dsa.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GAP = 1024              # TF_GLM53_REUSE_GAP: the fewest tokens between two assistant keep points
SYSTEM_MIN = 256        # the system block's end is a keep point when it is at least this far in (0015's 256)
KEEP_MOST = 3           # keep points a request adds besides its own end
ENTRIES = 32            # TF_GLM53_CACHE_ENTRIES
CACHE_GIB = 4.0         # TF_GLM53_CACHE_GIB: most bytes a rank holds for kept states (bounded by free memory)


def enabled() -> bool:
    """TF_GLM53_PROMPT_REUSE=1: keep and resume prompt states (off by default)."""
    v = os.environ.get("TF_GLM53_PROMPT_REUSE", "0").strip().lower()
    if v not in ("0", "1", "", "off", "on"):
        raise ValueError(f"TF_GLM53_PROMPT_REUSE={v!r}: 0 or 1")
    return v in ("1", "on")


def settings() -> dict:
    """The reuse knobs every rank must share (the engine compares them over the ranks)."""
    gap = int(os.environ.get("TF_GLM53_REUSE_GAP", str(GAP)))
    entries = int(os.environ.get("TF_GLM53_CACHE_ENTRIES", str(ENTRIES)))
    gib = float(os.environ.get("TF_GLM53_CACHE_GIB", str(CACHE_GIB)))
    loose = os.environ.get("TF_GLM53_REUSE_LOOSE", "0") == "1"
    if gap < 1 or entries < 1 or gib < 0:
        raise ValueError("TF_GLM53_REUSE_GAP and TF_GLM53_CACHE_ENTRIES must be >= 1, TF_GLM53_CACHE_GIB >= 0")
    return {"gap": gap, "entries": entries, "gib": gib, "loose": loose}


def special_ids(model_dir: str | Path) -> dict[str, int | None]:
    """GLM's <|user|>, <|assistant|> and <think> ids from the checkpoint's tokenizer.json (None where absent)."""
    names = {"user": "<|user|>", "assistant": "<|assistant|>", "think": "<think>"}
    out: dict[str, int | None] = {k: None for k in names}
    path = Path(model_dir) / "tokenizer.json"
    try:
        added = json.loads(path.read_text()).get("added_tokens", [])
    except (OSError, ValueError):
        return out
    for t in added:
        for k, name in names.items():
            if t.get("content") == name:
                out[k] = int(t["id"])
    return out


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    """How many leading ids two int64 arrays share."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n


def is_prefix(short: np.ndarray, long: np.ndarray) -> bool:
    return len(short) <= len(long) and bool(np.array_equal(long[:len(short)], short))


def even_chunks(begin: int, end: int, rows: int, short_rows: int) -> list[tuple[int, int]]:
    """Runner.chunks: [begin, end) in equal chunks of at most ``rows`` (``short_rows`` under 3 x rows)."""
    L = end - begin
    if L <= 0:
        return []
    if L < 3 * rows:                                     # big chunks pay only over several of them
        rows = min(rows, short_rows)
    n = -(-L // rows)
    step = -(-L // n)
    return [(a, min(a + step, end)) for a in range(begin, end, step)]


def cut_chunks(begin: int, end: int, stops, rows: int, short_rows: int) -> list[tuple[int, int]]:
    """Runner.segments: [begin, end) cut at the ``stops`` inside it, each piece in ``even_chunks``."""
    out, a = [], begin
    for t in sorted(set(int(p) for p in stops if begin < p < end)) + [end]:
        out += even_chunks(a, t, rows, short_rows)
        a = t
    return out


class PromptPlan:
    """Where a prompt's keep points are, from its tokens alone: prompts that agree up to a point agree on every point
    up to it, so they cut their prompt chunks alike there (``Runner.segments``).

    Points: the first <|user|> (the end of the system block: the template's effort line, tools and system messages)
    when at least ``system_min`` tokens in; then, at least ``gap`` tokens after the last point, the position after an
    <|assistant|> opener - after its <think> when one follows, which is where a next turn's history (reasoning kept or
    dropped: "<think>..." or "<think></think>") departs from the prompt that generated it. A prompt's own end is a
    point when it ends in such an opener."""

    def __init__(self, user: int | None, assistant: int | None, think: int | None, gap: int = GAP,
                 system_min: int = SYSTEM_MIN) -> None:
        self.user, self.assistant, self.think = user, assistant, think
        self.gap, self.system_min = int(gap), int(system_min)

    def points(self, prompt) -> list[int]:
        arr = np.asarray(prompt, dtype=np.int64)
        L = len(arr)
        out: list[int] = []
        last = 0
        if self.user is not None:
            at = np.flatnonzero(arr == self.user)
            if len(at) and int(at[0]) >= self.system_min:
                last = int(at[0])
                out.append(last)
        if self.assistant is not None:
            for i in np.flatnonzero(arr == self.assistant):
                c = int(i) + 1
                if self.think is not None and c < L and arr[c] == self.think:
                    c += 1
                if c - last >= self.gap:
                    out.append(c)
                    last = c
        return out

    def system_point(self, points: list[int], prompt) -> int | None:
        """The system block's point among ``points`` (the first <|user|>), or None."""
        if not points or self.user is None:
            return None
        p = points[0]
        return p if p < len(prompt) and int(prompt[p]) == self.user else None


@dataclass(eq=False)
class Kept:
    """A prompt state at n = len(ids). ``rows`` None: the target / MTP rows sit in the live caches."""
    ids: np.ndarray
    mode: str                              # the MTP input the carry and MTP rows were made with ("normed", "raw")
    carry: object = None                   # the target hidden n - 1 as the MTP reads it [hidden]
    ring: list | None = None               # DFlash2's window before n (``ring_window``)
    head: object = None                    # at a prompt's end: the logits row its first token was sampled from
    shared: bool = False                   # a system block or fork point: outlives its conversation's turns
    rows: list | None = None               # saved copies of the cache rows (``save_rows``): views of ``saved``
    saved: object = None                   # the ``Saved`` copy ``rows`` view (a conversation's states share one)
    tick: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)

    def held(self) -> int:
        """Bytes of this state's own tensors (a shared ``Saved`` copy is counted once, by ``KeptPrompts.held``)."""
        ts = [t for t in (self.carry, self.head) if t is not None] + list(self.ring or [])
        if self.saved is None:
            ts += list(self.rows or [])
        return sum(t.numel() * t.element_size() for t in ts)


class KeptPrompts:
    """The kept states (host bookkeeping; the tensors are the caller's). Every rank runs the same calls in the same
    order, so every rank holds, saves and drops the same entries."""

    def __init__(self, budget: int, entries: int = ENTRIES, loose: bool = False) -> None:
        self.budget, self.cap, self.loose = int(budget), max(1, int(entries)), bool(loose)
        self.entries: list[Kept] = []
        self.live = np.zeros((0,), dtype=np.int64)     # the ids whose rows the live caches hold
        self.clock = 0

    def held(self, entries=None) -> int:
        """Bytes ``entries`` (all states) hold: their own tensors, and each saved copy once."""
        es = self.entries if entries is None else entries
        copies = {id(e.saved): e.saved.nbytes for e in es if e.saved is not None}
        return sum(e.held() for e in es) + sum(copies.values())

    def touch(self, e: Kept) -> None:
        self.clock += 1
        e.tick = self.clock

    def clear(self) -> None:
        for e in list(self.entries):
            self.drop(e)

    def drop(self, e: Kept) -> None:
        self.entries = [x for x in self.entries if x is not e]
        e.carry = e.head = e.ring = e.rows = e.saved = None     # a shared copy goes with its last state

    def best(self, prompt: np.ndarray, points, mode: str, fits=None) -> Kept | None:
        """Rank 0: the longest state ``prompt`` resumes from - a strict prefix at one of the prompt's own points
        (any kept prefix with TF_GLM53_REUSE_LOOSE=1), or the whole prompt when it kept its head row (a replay);
        ``fits(e)``: what else the request needs of it."""
        L, at = len(prompt), set(points)
        best = None
        for e in self.entries:
            n = e.n
            if e.mode != mode or n > L or (best is not None and n <= best.n) or not self.usable(e):
                continue
            if fits is not None and not fits(e):
                continue
            if n == L:
                ok = e.head is not None
            else:
                ok = n in at or self.loose
            if ok and np.array_equal(prompt[:n], e.ids):
                best = e
        return best

    def named(self, prompt: np.ndarray, n: int) -> Kept | None:
        """The state rank 0 resumes from (followers)."""
        return next((e for e in self.entries
                     if e.n == n and self.usable(e) and np.array_equal(prompt[:n], e.ids)), None)

    def usable(self, e: Kept) -> bool:
        """Saved, or still in the live caches (a request that failed mid-prefill left later rows unknown)."""
        return e.rows is not None or self.in_live(e)

    def longest_shared(self, prompt: np.ndarray) -> int:
        """The most leading tokens ``prompt`` shares with a kept state."""
        return max((common_prefix(prompt, e.ids) for e in self.entries), default=0)

    def superseded(self, e: Kept) -> bool:
        """A strict prefix of another kept state: an earlier point of a conversation that went on."""
        return any(f is not e and f.n > e.n and is_prefix(e.ids, f.ids) for f in self.entries)

    def victim(self, protect=()) -> Kept | None:
        """The state to drop first (0063): a superseded one that is not shared (oldest first), then a superseded
        shared one, then the least recently used."""
        order = sorted((e for e in self.entries if all(e is not p for p in protect)), key=lambda e: e.tick)
        if not order:
            return None
        sup = [e for e in order if self.superseded(e)]
        for e in sup:
            if not e.shared:
                return e
        return sup[0] if sup else order[0]

    def fit(self, need: int, protect=()) -> bool:
        """Drop states until ``need`` more bytes fit the budget; False (nothing dropped) when even dropping every
        unprotected state would not make room."""
        floor = self.held([e for e in self.entries if any(e is p for p in protect)])
        if floor + need > self.budget:
            return False
        while self.held() + need > self.budget:
            v = self.victim(protect)
            if v is None:
                return False
            self.drop(v)
        return True

    def remember(self, e: Kept, protect=()) -> bool:
        """Keep ``e`` (newest), replacing a state of the same ids; within the entry cap and the byte budget."""
        for old in [x for x in self.entries if self.same_place(x, e) and x.n == e.n and np.array_equal(x.ids, e.ids)]:
            self.drop(old)
        protect = tuple(protect) + (e,)
        while len(self.entries) >= self.cap:
            v = self.victim(protect)
            if v is None:
                return False
            self.drop(v)
        if not self.fit(e.held(), protect):
            return False
        self.touch(e)
        self.entries.append(e)
        return True

    def same_place(self, a: Kept, b: Kept) -> bool:
        """Whether two states' rows live in the same caches (one stream: always)."""
        return True

    def overwritten(self, prompt: np.ndarray, begin: int, same=None) -> list[Kept]:
        """The states in the live caches that a request resumed at ``begin`` overwrites (newest first): all but
        prefixes of its prompt no longer than ``begin`` - and when the resumed rows replace the live ones (a saved
        state loaded), only those ``same(e)`` accepts (cut alike, so the replacing rows have their bits)."""
        out = []
        for e in self.entries:
            if e.rows is not None:
                continue
            if e.n <= begin and np.array_equal(prompt[:e.n], e.ids) and (same is None or same(e)):
                continue
            out.append(e)
        return sorted(out, key=lambda e: -e.tick)

    def in_live(self, e: Kept) -> bool:
        return e.n <= len(self.live) and np.array_equal(self.live[:e.n], e.ids)


# ------------------------------------------------------------------------------------------------ device rows ---
def row_views(st, n: int, dcp: int = 1) -> list:
    """The cache rows a state of n tokens reads: target latent / index keys of positions < n, the MTP layer's < n - 1.
    DCP: a rank holds positions p % dcp == rank at p // dcp; every rank takes ceil(n / dcp) rows (the most any rank
    has), so every rank counts the same bytes."""
    m = -(-n // dcp)
    mm = -(-max(n - 1, 0) // dcp)
    views = [t[:m] for t in st.kc] + [st.ic[i][:m] for i in sorted(st.ic)]
    if st.mkc is not None:
        views += [st.mkc[:mm], st.mic[:mm]]
    return views


def row_bytes(st, n: int, dcp: int = 1) -> int:
    """The bytes ``save_rows`` takes for a state of n tokens (``Saved``: each view's rows aligned)."""
    return Saved.size([v.numel() * v.element_size() for v in row_views(st, n, dcp)])[0]


class Saved:
    """One copy of the cache rows of a state (``row_views``) in one allocation: a single block torch's pool hands back
    to the next copy (no cudaMalloc, no new pages on the request path), and the states of the same conversation with
    fewer tokens (its earlier points: prefixes of the same live rows) read its first rows (``share_rows``)."""

    ALIGN = 256

    def __init__(self, views: list) -> None:
        import torch

        sizes = [v.numel() * v.element_size() for v in views]
        o, offs = self.size(sizes)
        self.nbytes = o
        self.flat = torch.empty((max(o, 1),), dtype=torch.uint8, device=views[0].device)
        self.parts = []
        for v, a, b in zip(views, offs, sizes):
            part = self.flat[a:a + b].view(v.dtype).view(v.shape)
            part.copy_(v)
            self.parts.append(part)

    @classmethod
    def size(cls, sizes: list[int]) -> tuple[int, list[int]]:
        """(total bytes, each part's offset) of parts of ``sizes`` bytes, each starting ALIGN-aligned."""
        offs, o = [], 0
        for b in sizes:
            offs.append(o)
            o += -(-b // cls.ALIGN) * cls.ALIGN
        return o, offs

    def rows_for(self, views: list) -> list:
        """The first rows of the copy that ``views`` (a shorter state's ``row_views``) cover."""
        if len(views) != len(self.parts) or any(v.shape[0] > p.shape[0] or v.shape[1:] != p.shape[1:]
                                                for v, p in zip(views, self.parts)):
            raise RuntimeError("a prompt state does not fit the saved rows it would share")
        return [p[:v.shape[0]] for p, v in zip(self.parts, views)]


def save_rows(st, e: Kept, dcp: int = 1) -> Saved:
    e.saved = Saved(row_views(st, e.n, dcp))
    e.rows = e.saved.parts
    return e.saved


def share_rows(st, saved: Saved, e: Kept, dcp: int = 1) -> None:
    """``e`` (a live prefix of the state ``saved`` was copied from: the same live rows) reads its rows there."""
    e.rows = saved.rows_for(row_views(st, e.n, dcp))
    e.saved = saved


def load_rows(st, e: Kept, dcp: int = 1) -> None:
    views = row_views(st, e.n, dcp)
    if len(views) != len(e.rows):
        raise RuntimeError("a saved prompt state's rows do not match the caches they go back to")
    for dst, src in zip(views, e.rows):
        dst.copy_(src)
    e.rows = e.saved = None                             # live again (states sharing the copy keep it)


def ring_window(dr, n: int) -> list:
    """DFlash2's context rows a later block pass at n (or past it) reads: positions [n - window - 1, n) of every draft
    layer's ring (slot p % RING). Decode overwrites slot p % RING once it passes p + RING, so they are copied."""
    import torch

    from .dflash import RING

    lo = max(0, n - dr.window - 1)
    if n <= lo:
        return []
    slots = torch.arange(lo, n, device=dr.dev) % RING
    return [c.index_select(1, slots) for c in dr.kc] + [c.index_select(1, slots) for c in dr.vc]


def put_ring_window(dr, n: int, rows: list) -> None:
    """``ring_window``'s rows back, the drafter's context ending at n."""
    import torch

    from .dflash import RING

    lo = max(0, n - dr.window - 1)
    if rows:
        slots = torch.arange(lo, n, device=dr.dev) % RING
        for c, r in zip(list(dr.kc) + list(dr.vc), rows):
            c.index_copy_(1, slots, r)
    dr.context_end = n
    dr.pos_dev.fill_(n)


def mode_key(mode: str | None, default: str) -> str:
    """The MTP input a request's prompt rows are made with (``Runner.set_mode``: DFlash2 / auto use the default)."""
    m = mode if mode and mode not in ("dflash", "auto") else default
    return m.partition(":")[0].split("/")[0]


def cut_and_keep(plan: PromptPlan, store: KeptPrompts, arr: np.ndarray, points: list[int], begin: int,
                 resume: bool) -> tuple[list[int], list[int]]:
    """Rank 0: a prompt resumed at ``begin``: its points past it short of its end (where its chunks are cut), and
    the ones it keeps (n * 2 + shared; its end is kept besides): the last one (a next turn's resume point when the
    end is not one), the system block's end, and where it stops sharing a kept state (0015), at most KEEP_MOST.
    ``resume`` False (the serial reference): cut alike, nothing kept."""
    L = len(arr)
    stops = [p for p in points if begin < p < L]
    if not resume:
        return stops, []
    keeps: dict[int, bool] = {}
    if stops:
        keeps[stops[-1]] = False
    sp = plan.system_point(points, arr)
    if sp is not None and begin < sp < L:
        keeps[sp] = True
    shared = store.longest_shared(arr)
    fork = max((p for p in stops if p <= shared), default=None)
    if fork is not None and fork not in keeps:
        keeps[fork] = True
    chosen = list(keeps.items())[:KEEP_MOST]
    return stops, sorted(n * 2 + int(s) for n, s in chosen)


class PromptReuse:
    """Rank 0 picks (``choose``); every rank runs (``run``) the keep / resume bookkeeping around Runner.generate."""

    FLUSH = 1                                            # header flag: forget every kept state first

    def __init__(self, runner, plan: PromptPlan, budget: int, entries: int, loose: bool) -> None:
        self.runner, self.plan = runner, plan
        self.store = KeptPrompts(budget, entries, loose)
        self.dcp = runner.w.dcp

    # -- rank 0 ---------------------------------------------------------------------------------------------------
    def choose(self, prompt: list[int], resume: bool, mode: str) -> tuple[int, list[int], list[int]]:
        """(resume point, the prompt's points past it short of its end, the ones kept: n * 2 + shared)."""
        arr = np.asarray(prompt, dtype=np.int64)
        points = self.plan.points(arr)
        hit = self.store.best(arr, points, mode) if resume else None
        begin = hit.n if hit is not None else 0
        stops, keeps = cut_and_keep(self.plan, self.store, arr, points, begin, resume)
        return begin, stops, keeps

    def flush_requested(self) -> bool:
        """Rank 0: a REUSE_FLUSH file next to the profile flag forgets every kept state (cold-vs-warm tests)."""
        from .runner import PROFILE_FLAG

        path = os.path.join(os.path.dirname(PROFILE_FLAG), "REUSE_FLUSH")
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
        except OSError:
            pass
        return True

    # -- every rank -----------------------------------------------------------------------------------------------
    def run(self, prompt: list[int], begin: int, stops: list[int], keeps: list[int], flags: int, keep_end: bool,
            mode: str, gen) -> dict:
        """Take the live caches over for this prompt, restore its resume point, prefill (``gen(begin, stops, keep,
        head)`` -> Runner.generate's stats) keeping the states rank 0 named and the prompt's end (``keep_end``).
        Kept states of this prompt's own ids at its cut points past ``begin`` (and at its end) are rewritten by it,
        cut alike: it keeps them again from its own rows (``renew``) - neither saved nor dropped - so a cold
        reference ("draft": false, ``keep_end`` False) of a kept prompt leaves it resumable and replayable."""
        import torch

        rn, store = self.runner, self.store
        if flags & self.FLUSH:
            store.clear()
            torch.cuda.empty_cache()
        arr = np.asarray(prompt, dtype=np.int64)
        L = len(arr)
        hit = None
        if begin:
            hit = store.named(arr, begin)
            if hit is None:
                raise RuntimeError(f"rank {rn.w.rank} has no kept prompt state of the {begin} tokens rank 0 resumes "
                                   "from")
        same = None
        if hit is not None and hit.rows is not None:     # its saved rows replace the live ones before ``begin``:
            at = set(self.plan.points(arr))               # a live prefix keeps its bits only where it was cut alike
            same = lambda e: e.mode == hit.mode and (e.n in at or e.n == begin)    # noqa: E731
        cut = set(int(p) for p in stops if begin < p < L) | ({L} if begin < L else set())
        renew = [e for e in store.entries
                 if e is not hit and e.n in cut and np.array_equal(arr[:e.n], e.ids)]
        self._take_over(arr, begin, hit, same, renew)
        if hit is not None:
            if hit.rows is not None:
                copy = hit.saved
                load_rows(rn.st, hit, self.dcp)
                for e in store.entries:                  # its shorter states that shared the copy: live again
                    if copy is not None and e.saved is copy and e.n <= begin:
                        e.rows = e.saved = None
            rn.carry.copy_(hit.carry)
            if rn.drafter is not None:
                put_ring_window(rn.drafter, begin, hit.ring or [])
            store.touch(hit)
        store.live = arr[:begin]                         # rows past it are about to change (a failed request: the
        self._trim()                                     # states past ``begin`` stay unusable, ``usable``)
        named = {k >> 1: bool(k & 1) for k in keeps}
        renewed = {e.n: e.shared for e in renew}
        added: list[int] = []

        def keep(n: int, head=None) -> None:
            if n < L and n not in named and n not in renewed:
                return
            if n == L and not keep_end and n not in renewed:
                return
            e = Kept(arr[:n].copy(), mode, carry=rn.carry.clone(),
                     ring=ring_window(rn.drafter, n) if rn.drafter is not None else None,
                     head=head.clone() if head is not None and n == L else None,
                     shared=named.get(n, False) or renewed.get(n, False))
            if store.remember(e, protect=(hit,) if hit is not None else ()):    # replaces a state of the same ids
                added.append(n)

        replay = hit is not None and begin == L
        stats = gen(begin, stops, keep, hit.head if replay else None)
        store.live = arr
        stats.update(cached=begin, kept=added, kept_states=len(store.entries),
                     kept_mib=round(store.held() / 2**20, 1))
        if replay:
            stats["replay"] = True
        return stats

    def _take_over(self, arr: np.ndarray, begin: int, hit: Kept | None, same=None, renew=()) -> bool:
        """The live states this request overwrites (but ``renew``s): every one is a prefix of the live ids, so they
        are one conversation's - the longest one that fits the budget is copied out once (``save_rows``) and the
        shorter ones read their rows from that copy (``share_rows``); longer ones that do not fit are dropped, as are
        states whose rows are no longer live. At most one copy a request; none when the request resumes past every
        live state or rewrites them at its own cut points. True when any state was dropped."""
        store, st = self.store, self.runner.st
        dropped = False
        chain = []
        for e in store.overwritten(arr, begin, same):
            if e is hit or any(e is r for r in renew):
                continue
            if store.in_live(e):
                chain.append(e)
            else:
                store.drop(e)
                dropped = True
        copy = None
        for e in sorted(chain, key=lambda x: -x.n):
            if e not in store.entries:                   # dropped making room for the copy
                continue
            if copy is not None:
                share_rows(st, copy, e, self.dcp)
                continue
            need = row_bytes(st, e.n, self.dcp)
            if store.fit(need, protect=tuple(x for x in (hit, e) if x is not None) + tuple(renew)):
                copy = save_rows(st, e, self.dcp)
            else:
                store.drop(e)
                dropped = True
        return dropped

    def _trim(self) -> None:
        """Dropped copies stay in torch's pool, which hands them to the next copy (a cudaMalloc of GiBs maps fresh
        pages of GB10's unified memory, and the empty_cache it took before synchronized the device and also gave back
        the prompt path's scratch, every request); the pool gives them back to the system only when what it holds
        unused passes the kept-state budget (so kept + idle stays within about twice the budget, which
        ``Runner.kept_budget`` sized to half of what is free past the reserve)."""
        import torch

        if not torch.cuda.is_available():
            return
        if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > max(self.store.budget, 1 << 30):
            torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------ concurrent streams ---
# --parallel N (multi.GlmMultiDecoder): a state stays in the cache rows of the slot its stream filled, valid while
# that slot's rows still hold its ids (``SlotPrompts.lives``); a request resumes in place when that slot is free, else
# its rows are copied into the free slot it gets (``copy_slot_rows``, rows a decoding stream never writes). States are
# never saved out of the slots: a slot that takes another conversation forgets the states it overwrites.

class SlotPrompts(KeptPrompts):
    """Kept states in their slots (``e.extra["slot"]``); ``lives[slot]``: the ids whose rows that slot holds."""

    def __init__(self, budget: int, entries: int = ENTRIES, loose: bool = False) -> None:
        super().__init__(budget, entries, loose)
        self.lives: dict[int, np.ndarray] = {}

    def in_live(self, e: Kept) -> bool:
        live = self.lives.get(e.extra["slot"])
        return live is not None and e.n <= len(live) and np.array_equal(live[:e.n], e.ids)

    def usable(self, e: Kept) -> bool:
        return self.in_live(e)

    def same_place(self, a: Kept, b: Kept) -> bool:
        return a.extra["slot"] == b.extra["slot"]

    def in_slot(self, slot: int) -> list[Kept]:
        return [e for e in self.entries if e.extra["slot"] == slot]

    def named_in(self, prompt: np.ndarray, n: int, slot: int) -> Kept | None:
        return next((e for e in self.in_slot(slot)
                     if e.n == n and self.usable(e) and np.array_equal(prompt[:n], e.ids)), None)

    def overwritten_slot(self, slot: int, prompt: np.ndarray, begin: int, same=None) -> list[Kept]:
        """The states of ``slot`` a request resumed there at ``begin`` overwrites (``same``: as in ``overwritten``,
        when rows copied from another slot replace the slot's)."""
        return [e for e in self.in_slot(slot)
                if not (e.n <= begin and np.array_equal(prompt[:e.n], e.ids) and (same is None or same(e)))]


def copy_slot_rows(st, src: int, dst: int, n: int) -> None:
    """Slot ``src``'s cache rows of a state of n tokens into slot ``dst`` (target < n, MTP < n - 1; DCP is off with
    slots)."""
    a, b = src * st.local, dst * st.local
    for t in list(st.kc) + [st.ic[i] for i in sorted(st.ic)]:
        t[b:b + n].copy_(t[a:a + n])
    if st.mkc is not None and n > 1:
        for t in (st.mkc, st.mic):
            t[b:b + n - 1].copy_(t[a:a + n - 1])


def slot_ring_window(md, slot: int, n: int) -> list:
    """``ring_window`` of a MultiDrafter slot's ring (rows slot * RING + p % RING of its pools)."""
    import torch

    from .dflash import RING

    lo = max(0, n - md.d.window - 1)
    if n <= lo:
        return []
    idx = slot * RING + torch.arange(lo, n, device=md.dev) % RING
    return [c.index_select(1, idx) for c in md.kc] + [c.index_select(1, idx) for c in md.vc]


def put_slot_ring_window(md, slot: int, n: int, rows: list) -> None:
    import torch

    from .dflash import RING

    lo = max(0, n - md.d.window - 1)
    if rows:
        idx = slot * RING + torch.arange(lo, n, device=md.dev) % RING
        for c, r in zip(list(md.kc) + list(md.vc), rows):
            c.index_copy_(1, idx, r)
    md.end[slot] = n


class SlotReuse:
    """Prompt reuse for concurrent streams: rank 0 ``choose``s the slot, the state and its points; every rank
    ``admit``s and ``keep``s alike (multi.GlmMultiDecoder)."""

    def __init__(self, plan: PromptPlan, budget: int, entries: int, loose: bool) -> None:
        self.plan = plan
        self.store = SlotPrompts(budget, entries, loose)

    def choose(self, prompt: list[int], resume: bool, mode: str, free: list[int], ring: bool):
        """Rank 0: (slot, resume point, source slot or -1 (in place / none), cut points, kept points). ``ring``: the
        stream drafts with DFlash2 (a state without the drafter's window does not fit it)."""
        arr = np.asarray(prompt, dtype=np.int64)
        points = self.plan.points(arr)
        st = self.store
        hit = st.best(arr, points, mode, fits=(lambda e: e.ring is not None) if ring else None) if resume else None
        if hit is not None and hit.extra["slot"] in free:
            slot, src = hit.extra["slot"], -1
        else:                                            # the free slot whose states are worth least (none first)
            slot = min(free, key=lambda f: (max((e.tick for e in st.in_slot(f)), default=0), f))
            src = hit.extra["slot"] if hit is not None else -1
        begin = hit.n if hit is not None else 0
        stops, keeps = cut_and_keep(self.plan, st, arr, points, begin, resume)
        return slot, begin, src, stops, keeps

    def admit(self, prompt: list[int], slot: int, begin: int, src: int, st) -> Kept | None:
        """Every rank: forget the states of ``slot`` this request overwrites, bring the resumed state's rows into it
        (copied from ``src`` unless in place); returns that state."""
        arr = np.asarray(prompt, dtype=np.int64)
        store = self.store
        hit = None
        if begin:
            hit = store.named_in(arr, begin, src if src >= 0 else slot)
            if hit is None:
                raise RuntimeError(f"no kept prompt state of the {begin} tokens rank 0 resumes from in slot "
                                   f"{src if src >= 0 else slot}")
        same = None
        if src >= 0 and hit is not None:                 # the source slot's rows replace this slot's before ``begin``
            at = set(self.plan.points(arr))
            same = lambda e: e.mode == hit.mode and (e.n in at or e.n == begin)    # noqa: E731
        for e in store.overwritten_slot(slot, arr, begin, same):
            if e is not hit:
                store.drop(e)
        if src >= 0:
            copy_slot_rows(st, src, slot, begin)
        store.lives[slot] = arr[:begin]
        if hit is not None:
            store.touch(hit)
        return hit

    def flush_requested(self) -> bool:
        return PromptReuse.flush_requested(self)

    def keep(self, slot: int, ids: np.ndarray, mode: str, carry, ring, head, shared: bool) -> bool:
        e = Kept(np.array(ids, dtype=np.int64), mode, carry=carry.clone(), ring=ring,
                 head=head.clone() if head is not None else None, shared=shared)
        e.extra["slot"] = slot
        return self.store.remember(e)

    def filled(self, slot: int, prompt: list[int]) -> None:
        """Every rank: the slot's rows hold the whole prompt now."""
        self.store.lives[slot] = np.asarray(prompt, dtype=np.int64)
