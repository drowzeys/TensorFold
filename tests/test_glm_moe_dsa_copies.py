"""GLM-5.3's copy drafts (families/glm_moe_dsa/cuda/copies, TF_GLM53_COPY_*; after MiaAI-Lab 0007 / 0032), on the
CPU: the hashed index proposes exactly what MiaAI-Lab's scan of the whole context would, round after round as the
reply grows, hash collisions included; nothing when the suffix never occurred; and the settings parse and refuse
like the engine's other window-changing knobs. The runner's padding to captured widths is checked on a stand-in."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.families.glm_moe_dsa.cuda import copies


def _rounds(prompt, reply, match, most, steps):
    """Feed ``reply`` in uneven pieces (a round keeps 1..most + 1 tokens); before each piece the index's proposal
    and the reference scan's, for every room 1..most."""
    cx = copies.CopyIndex(prompt, match, most)
    ctx = list(prompt)
    i = 0
    rng = np.random.default_rng(len(prompt) + match)
    while i < len(reply):
        for room in range(1, most + 1):
            assert cx.propose(room) == copies.reference(ctx, match, min(room, most)), (i, room)
        take = int(rng.integers(1, min(steps, len(reply) - i) + 1))
        cx.extend(reply[i:i + take])
        ctx.extend(reply[i:i + take])
        i += take
        assert cx.tokens() == ctx


@pytest.mark.parametrize("match", [2, 4, 8])
def test_index_equals_the_reference_scan(match):
    rng = np.random.default_rng(match)
    text = [int(t) for t in rng.integers(1000, 1060, size=400)]            # a small vocabulary: many repeats
    prompt = text[:300]
    reply = text[120:200] + [int(t) for t in rng.integers(1000, 1060, size=40)] + text[10:90]   # quotes + new text
    _rounds(prompt, reply, match, 15, 16)


def test_the_latest_occurrence_with_room_wins_else_the_earliest():
    a = [1, 2, 3, 4]
    prompt = a + [10, 11] + a + [20]                                       # a occurs twice: at 0 and at 6
    cx = copies.CopyIndex(prompt, 4, 15)
    cx.extend(a)                                                           # the reply ends with a (pending token: 4)
    assert cx.propose(1) == [20]                                           # the latest (6) has 1 token after it
    assert cx.propose(6) == [10, 11, 1, 2, 3, 4]                           # only 0 has 6 after it
    assert cx.propose(12) == [10, 11, 1, 2, 3, 4, 20, 1, 2, 3, 4]          # none has 12: the earliest, all it has
    for k in (1, 5, 6, 12):
        assert cx.propose(k) == copies.reference(cx.tokens(), 4, k)


def test_no_match_and_short_contexts_propose_nothing():
    cx = copies.CopyIndex([5, 6, 7], 8, 15)
    assert cx.propose() == []
    cx.extend(list(range(100, 120)))
    assert cx.propose() == []                                              # nothing repeats
    cx = copies.CopyIndex([], 2, 3)
    cx.extend([1, 2, 3, 1, 2])
    assert cx.propose(3) == [3, 1, 2] == copies.reference([1, 2, 3, 1, 2], 2, 3)
    assert cx.propose(0) == []


def test_the_reply_copies_itself():
    cx = copies.CopyIndex([9, 9, 9], 3, 4)
    loop = [1, 2, 3, 4, 5, 6]
    cx.extend(loop + loop[:3])                                             # the reply repeats its own start
    assert cx.propose(4) == [4, 5, 6, 1]


def test_hash_collisions_are_checked(monkeypatch):
    """A base of 0 hashes every n-gram to its last token: every lookup collides, the token check decides."""
    monkeypatch.setattr(copies, "_B", 0)
    rng = np.random.default_rng(3)
    text = [int(t) for t in rng.integers(0, 12, size=300)]
    _rounds(text[:200], text[50:150], 5, 7, 8)


def test_grows_past_its_buffer():
    cx = copies.CopyIndex([1, 2, 3], 2, 4)
    for i in range(3000):
        cx.extend([i % 50])
    assert len(cx) == 3003 and cx.tokens()[-50:] == list(range(50))
    assert cx.propose(4) == copies.reference(cx.tokens(), 2, 4)


def test_settings():
    assert copies.settings({}) == (1, 8, 15)                              # on by default
    assert copies.settings({"TF_GLM53_COPY_DRAFTS": "0", "TF_GLM53_COPY_MAX": "99"}) == (0, 0, 0)
    assert copies.settings({"TF_GLM53_COPY_MIN": "4", "TF_GLM53_COPY_MAX": "31"}) == (1, 4, 31)
    for bad in ({"TF_GLM53_COPY_DRAFTS": "yes"}, {"TF_GLM53_COPY_MIN": "1"}, {"TF_GLM53_COPY_MAX": "32"},
                {"TF_GLM53_COPY_MAX": "0"}, {"TF_GLM53_COPY_MIN": "x"}):
        with pytest.raises(ValueError):
            copies.settings(bad)


@pytest.mark.torch
def test_runner_pads_to_captured_widths():
    """Runner._copied (needs torch to import): a proposal whose window falls between captured widths is padded with
    its last draft when the room allows, else cut to the widest captured width below it."""
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.glm_moe_dsa.cuda.runner import Runner

    run = SimpleNamespace(copy_max=15, capacity=10_000, widths=list(range(1, 9)) + [16])
    a = list(range(500, 508))
    src = [600, 601, 602, 603] + a                                         # what follows a's first occurrence
    prompt = a + src                                                       # ends with a: its match is at 0

    def copied(left, P=100):
        return Runner._copied(run, copies.CopyIndex(prompt, 8, 15), left, P)

    assert copied(15) == src + [src[-1]] * 3                               # 12 drafts (13 rows) -> 16 rows
    assert copied(11) == src[:7]                                           # 11 (12 rows), no room for 16 -> 8 rows
    assert copied(5) == src[:5]                                            # 6 rows: a captured width
    assert copied(15, P=10_000 - 5) == src[:4]                             # the cache's room bounds it too
    assert Runner._copied(run, None, 15, 100) == []
