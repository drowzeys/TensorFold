"""MTP draft depth by acceptance (TF_GLM53_DEPTH_POLICY), per request: the Runner drafts between TF_GLM53_DEPTH_MIN
(default 2) and its --mtp-drafts depth, whichever the request's running acceptance makes the most tokens per unit of
round time. Drafts only propose - the target verifies every token - so the depth changes speed, never a reply.

Ported from the idea of bertholomus' TensorFold fork (14b43b0, decode.AcceptPolicy / TF_GLM_DEPTH_COST). Each draft
position's acceptance given the ones before it is a running mean (rate TF_GLM53_DEPTH_RATE) over the rounds that
verified it; a depth d is worth (1 + a1 + a1 a2 + ... + a1..ad) / (1 + cost d) - tokens a round over the round's
time relative to a round with no drafts, ``cost`` being what one draft position adds (an MTP step plus a verify row).
Every PROBE rounds one position deeper, so the deeper estimates stay current. Only the rounds' drafted / accepted
counts decide (the same on every rank), never a clock: every rank runs the same windows.

Pure Python (no torch): tests/test_glm53_depth_policy.py runs it on the CPU."""

from __future__ import annotations

import os

PROBE = 8                 # every this many rounds one position deeper
COST_ON = 0.27            # TF_GLM53_DEPTH_POLICY=1 / on: a draft position ~ 0.27 of a no-draft round (an MTP step
                          # ~2 ms + a verify row ~8.5 ms over a ~39 ms one-row round: the fork's TP4 graph replays)


def parse(v: str | None) -> float:
    """TF_GLM53_DEPTH_POLICY -> the cost of a draft position (0: off, the fixed --mtp-drafts depth)."""
    v = (v or "").strip().lower()
    if v in ("", "0", "off"):
        return 0.0
    if v in ("1", "on"):
        return COST_ON
    cost = float(v)
    if not 0.0 < cost < 10.0:
        raise ValueError(f"TF_GLM53_DEPTH_POLICY={v}: 0 (off), 1 / on (cost {COST_ON}) or a draft position's cost "
                         "relative to a no-draft round, above 0")
    return cost


COST = parse(os.environ.get("TF_GLM53_DEPTH_POLICY"))
LOW = int(os.environ.get("TF_GLM53_DEPTH_MIN", "2") or 2)
RATE = float(os.environ.get("TF_GLM53_DEPTH_RATE", "0.125") or 0.125)


class DepthPolicy:
    """One request's draft depth, ``low`` .. ``most`` (``low`` clipped to 1 .. most)."""

    def __init__(self, most: int, cost: float = COST, low: int = LOW, prior: float = 0.8, rate: float = RATE,
                 probe: int = PROBE) -> None:
        self.most, self.cost, self.rate, self.probe = int(most), float(cost), float(rate), int(probe)
        self.low = max(1, min(int(low), self.most))
        self.a = [float(prior)] * self.most
        self.rounds = 0
        self.depths: dict[int, int] = {}                 # depth -> rounds that drafted it (stats)

    def update(self, drafted: int, accepted: int) -> None:
        """A round that drafted ``drafted`` tokens and kept the first ``accepted`` of them: positions 1 .. accepted
        were kept, position accepted + 1 (if drafted) rejected, later ones never verified (left alone)."""
        for j in range(min(drafted, accepted + 1, self.most)):
            self.a[j] += self.rate * ((1.0 if j < accepted else 0.0) - self.a[j])

    def best(self) -> int:
        """The depth with the most expected tokens per cost (ties: the shallower)."""
        depth, value, chain, gain = self.low, -1.0, 1.0, 1.0
        for d in range(1, self.most + 1):
            chain *= self.a[d - 1]
            gain += chain
            if d < self.low:
                continue
            v = gain / (1.0 + self.cost * d)
            if v > value:
                depth, value = d, v
        return depth

    def depth(self) -> int:
        """This round's depth (call once a round, before drafting)."""
        self.rounds += 1
        d = self.best()
        if self.probe and self.rounds % self.probe == 0:
            d = min(self.most, d + 1)
        self.depths[d] = self.depths.get(d, 0) + 1
        return d


def for_runner(k: int) -> DepthPolicy | None:
    """A request's policy when TF_GLM53_DEPTH_POLICY is on and there is a choice (k above the minimum), else None."""
    if COST <= 0 or k <= max(1, LOW):
        return None
    return DepthPolicy(k)
