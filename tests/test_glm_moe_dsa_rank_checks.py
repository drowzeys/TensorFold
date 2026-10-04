"""glm_moe_dsa rank checks (MiaAI-Lab 0065): rank 0's messages carry a sequence number and checksum."""

from __future__ import annotations

import pytest

multi = pytest.importorskip("tensorfold.families.glm_moe_dsa.cuda.multi")


def test_digest_names_the_message_and_its_place():
    msg = [2, 3, 0, 7, 1 << 40, -5]
    assert multi.digest(msg, 9) == multi.digest(list(msg), 9)
    assert multi.digest(msg, 9) != multi.digest(msg, 10)                  # another message number
    assert multi.digest(msg, 9) != multi.digest(msg[:-1] + [-4], 9)       # one value changed
    assert multi.digest(msg, 9) != multi.digest([msg[1], msg[0], *msg[2:]], 9)   # two values swapped
    assert 0 <= multi.digest([], 0) < multi.SEAL_MOD


def test_watchdog_off_by_default(monkeypatch):
    calls = []
    import faulthandler

    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls.append(k))
    multi.watch(0.0)
    assert calls == []
    multi.watch(5.0)
    assert calls and calls[0]["exit"] is True
