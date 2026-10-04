"""GLM-5.3's MTP draft depth by acceptance (families/glm_moe_dsa/cuda/depth, TF_GLM53_DEPTH_POLICY), on the CPU:
drafts kept every round keep the deepest chain, drafts rejected at once fall to the minimum (2) with a probe one
deeper every PROBE rounds, an in-between rate picks the depth with the most expected tokens per cost, and only the
positions a round verified move."""

from __future__ import annotations

import pytest

from tensorfold.families.glm_moe_dsa.cuda import depth


def test_parse():
    assert depth.parse(None) == depth.parse("") == depth.parse("0") == depth.parse("off") == 0.0
    assert depth.parse("1") == depth.parse("on") == depth.COST_ON
    assert depth.parse("0.2") == 0.2
    with pytest.raises(ValueError):
        depth.parse("-1")


def test_kept_drafts_keep_the_deepest_chain():
    p = depth.DepthPolicy(3, cost=0.27, low=2)
    got = []
    for _ in range(40):
        d = p.depth()
        got.append(d)
        p.update(d, d)
    assert set(got) == {3}


def test_rejected_drafts_fall_to_the_minimum_and_probe_deeper():
    p = depth.DepthPolicy(3, cost=0.27, low=2)
    got = []
    for _ in range(64):
        d = p.depth()
        got.append(d)
        p.update(d, 0)
    tail = got[32:]
    assert min(tail) == 2 and max(tail) == 3
    assert sum(d == 3 for d in tail) == len(tail) // depth.PROBE
    assert p.depths == {2: got.count(2), 3: got.count(3)}


def test_in_between_rates_weigh_tokens_against_cost():
    p = depth.DepthPolicy(3, cost=0.27, low=2, probe=0)
    p.a = [0.7, 0.7, 0.7]
    assert p.best() == 2              # 2.19 / 1.54 = 1.422 beats 2.533 / 1.81 = 1.400
    p.a = [0.8, 0.8, 0.8]
    assert p.best() == 3              # 2.952 / 1.81 = 1.631 beats 2.44 / 1.54 = 1.584
    p.cost = 0.05
    p.a = [0.6, 0.6, 0.6]
    assert p.best() == 3              # cheap draft rows: the whole chain
    lo = depth.DepthPolicy(3, cost=0.27, low=1, probe=0)
    lo.a = [0.3, 0.3, 0.3]
    assert lo.best() == 1             # with the minimum at 1 a poor chain stops at one draft


def test_only_verified_positions_move():
    p = depth.DepthPolicy(3, cost=0.27, low=2)
    p.update(3, 1)                    # first kept, second rejected, third never verified
    assert p.a[0] > 0.8 and p.a[1] < 0.8 and p.a[2] == 0.8
    p.update(2, 2)                    # a depth-2 round that kept both: the third position is untouched
    assert p.a[2] == 0.8


def test_off_or_no_choice_gives_no_policy(monkeypatch):
    monkeypatch.setattr(depth, "COST", 0.0)
    assert depth.for_runner(3) is None
    monkeypatch.setattr(depth, "COST", 0.27)
    monkeypatch.setattr(depth, "LOW", 2)
    assert depth.for_runner(2) is None    # nothing to choose between
    assert depth.for_runner(3) is not None
