"""Prompt-lookup ("copy") drafts for full GLM-5.3: when the reply's last ``match`` tokens (the pending one included)
occurred before - in the prompt or earlier in the reply - the tokens that followed that occurrence are proposed as the
round's drafts, ahead of (instead of) the MTP head's or DFlash2's. Quoting, editing and refactoring turns copy long runs
of their prompt; prose rarely repeats 8 tokens, so it rarely changes.

Adapted from MiaAI-Lab's GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold patches 0007-glm-copy-drafts and
0032-glm-code-copy-drafts (Apache-2.0, Copyright 2026 MiaAI-Lab), re-implemented for this family: the same choice of
occurrence (the latest one with ``k`` tokens after it, else the earliest, which has the most) and the same policy (a
round that finds a copy verifies it and runs no drafter), with a hashed index instead of a scan of the whole context
each round, so a round with no match costs a dict lookup and a binary search.

Exact: drafts only propose - the target verifies every row and keeps its own samples, so replies equal serial ones bit
for bit. Every rank holds the same prompt and samples the same tokens, so every rank computes the same proposals with
no exchange; TF_GLM53_COPY_* must agree across ranks (engine.py compares them at startup).

TF_GLM53_COPY_DRAFTS (default 1): 0 turns copy drafts off. TF_GLM53_COPY_MIN (default 8, 2..64): the match length.
TF_GLM53_COPY_MAX (default 15, 1..fused.DECODE_ROWS - 1): drafts a copied round proposes; runner.py captures verify
windows of 8 and TF_GLM53_COPY_MAX + 1 rows besides the drafters' and pads a copy up to the next of them.

Pure numpy (no torch): tests/test_glm_moe_dsa_copies.py runs it on the CPU."""

from __future__ import annotations

import os
from bisect import bisect_right
from typing import Sequence

import numpy as np

MATCH = 8                 # TF_GLM53_COPY_MIN: the context's last this many tokens must have occurred before
MOST = 15                 # TF_GLM53_COPY_MAX: drafts a copied round proposes (MiaAI-Lab's COPY_MAX 15: edit +42%)
WIDEST = 32               # fused.DECODE_ROWS: the widest decode window (a pending token and 31 drafts)
_B = 0x9E3779B97F4A7C15   # the rolling hash's base (odd), mod 2**64
_MASK = (1 << 64) - 1


def settings(env=None) -> tuple[int, int, int]:
    """(on, match, most) from TF_GLM53_COPY_DRAFTS / _MIN / _MAX; (0, 0, 0) when off. The ints engine.py compares."""
    env = os.environ if env is None else env
    on = env.get("TF_GLM53_COPY_DRAFTS", "1").strip()
    if on not in ("", "0", "1"):
        raise ValueError(f"TF_GLM53_COPY_DRAFTS: 0 or 1, not {on!r}")
    if on == "0":
        return 0, 0, 0

    def number(name: str, default: int, low: int, high: int) -> int:
        text = env.get(name, "").strip()
        if text == "":
            return default
        if not text.isdecimal() or not low <= int(text) <= high:
            raise ValueError(f"{name}: {low} to {high}, not {text!r}")
        return int(text)

    return 1, number("TF_GLM53_COPY_MIN", MATCH, 2, 64), number("TF_GLM53_COPY_MAX", MOST, 1, WIDEST - 1)


SETTINGS = settings()


def _hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """The rolling hash of every ``n`` tokens of ``tokens`` (uint64; start i -> tokens[i:i + n]), as ``extend`` rolls them."""
    m = tokens.shape[0] - n + 1
    if m <= 0:
        return np.empty((0,), dtype=np.uint64)
    t = tokens.astype(np.uint64)
    h = np.zeros((m,), dtype=np.uint64)
    b = np.uint64(_B)
    for j in range(n):                    # uint64 arithmetic wraps: mod 2**64, as extend's masked ints
        h = h * b + t[j:j + m]
    return h


class CopyIndex:
    """One request's context (prompt, then the reply's tokens with the pending one) and its copy proposals.

    The prompt's n-grams are hashed once (numpy), sorted, and looked up by binary search; the reply's are added to a
    dict as they arrive (``extend``). Lookups compare the tokens themselves, so a hash collision only costs a check."""

    def __init__(self, prompt: Sequence[int], match: int = MATCH, most: int = MOST) -> None:
        if match < 1 or most < 1:
            raise ValueError("match and most must be positive")
        self.match, self.most = int(match), int(most)
        n = len(prompt)
        self.buf = np.empty((max(1024, 2 * n),), dtype=np.int32)
        if n:
            self.buf[:n] = np.asarray(prompt, dtype=np.int32)
        self.length = n
        h = _hashes(self.buf[:n], self.match)
        self.order = np.argsort(h, kind="stable")          # equal hashes keep ascending starts
        self.keys = h[self.order]
        self.base = h.shape[0]                             # n-grams starting before this are in keys / order
        self.tail: dict[int, list[int]] = {}               # hash -> ascending starts of the reply's n-grams
        self.h = int(h[-1]) if h.shape[0] else None        # the hash of the context's last n tokens

    def __len__(self) -> int:
        return self.length

    def tokens(self) -> list[int]:
        return self.buf[:self.length].tolist()

    def extend(self, tokens: Sequence[int]) -> None:
        """Tokens that joined the context (the reply's, the last one the next round's pending token)."""
        k = len(tokens)
        if not k:
            return
        if self.length + k > self.buf.shape[0]:
            grown = np.empty((2 * (self.length + k),), dtype=np.int32)
            grown[:self.length] = self.buf[:self.length]
            self.buf = grown
        self.buf[self.length:self.length + k] = np.asarray(tokens, dtype=np.int32)
        n = self.match
        lead = pow(_B, n - 1, 1 << 64)
        for L in range(self.length + 1, self.length + k + 1):
            s = L - n                                      # the n-gram that ends with this token
            if s < 0:
                continue
            if self.h is None or s == 0:
                h = 0
                for t in self.buf[s:L].tolist():
                    h = (h * _B + t) & _MASK
            else:                                          # drop the token before s, take the new one
                h = ((self.h - int(self.buf[s - 1]) * lead) * _B + int(self.buf[L - 1])) & _MASK
            self.h = h
            self.tail.setdefault(h, []).append(s)
        self.length += k

    def _same(self, s: int) -> bool:
        L, n = self.length, self.match
        return bool(np.array_equal(self.buf[s:s + n], self.buf[L - n:L]))

    def _starts(self, h: int) -> tuple[np.ndarray, list[int]]:
        """Ascending starts with hash ``h``: the prompt's, then the reply's (every reply start follows the prompt's)."""
        key = np.uint64(h)
        lo = int(np.searchsorted(self.keys, key, side="left"))
        hi = int(np.searchsorted(self.keys, key, side="right"))
        return self.order[lo:hi], self.tail.get(h, [])

    def find(self, k: int) -> int | None:
        """The start of the earlier occurrence of the context's last ``match`` tokens to copy ``k`` drafts from: the
        latest with ``k`` tokens after it, else the earliest (the most after it); None when there is none."""
        L, n = self.length, self.match
        if L <= n or self.h is None:
            return None
        head, tail = self._starts(self.h)
        if not head.size and not tail:
            return None
        last = L - n - k                                   # latest start leaving k tokens after its match
        for i in range(bisect_right(tail, last) - 1, -1, -1):
            if self._same(tail[i]):
                return tail[i]
        for i in range(int(np.searchsorted(head, last, side="right")) - 1, -1, -1):
            if self._same(int(head[i])):
                return int(head[i])
        for s in self._ascending(head, tail):
            if s >= L - n:                                 # the context's own last n tokens (and later)
                break
            if self._same(s):
                return s
        return None

    @staticmethod
    def _ascending(head: np.ndarray, tail: list[int]):
        for v in head:
            yield int(v)
        yield from tail

    def propose(self, room: int | None = None) -> list[int]:
        """Up to ``min(most, room)`` drafts: what followed the chosen earlier occurrence; [] when there is none."""
        k = self.most if room is None else min(self.most, int(room))
        if k < 1:
            return []
        s = self.find(k)
        if s is None:
            return []
        n = self.match
        return self.buf[s + n:min(s + n + k, self.length)].tolist()


def reference(context: Sequence[int], match: int, k: int) -> list[int]:
    """MiaAI-Lab 0007's scan (every start compared, no index): what ``CopyIndex.propose(k)`` must return (tests)."""
    ctx = list(context)
    L, n = len(ctx), match
    if L <= n or k < 1:
        return []
    q = ctx[L - n:]
    hits = [s for s in range(L - n) if ctx[s:s + n] == q]
    if not hits:
        return []
    full = [s for s in hits if s <= L - n - k]
    s = full[-1] if full else hits[0]
    return ctx[s + n:min(s + n + k, L)]
