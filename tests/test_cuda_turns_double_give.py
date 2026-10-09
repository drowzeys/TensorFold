"""The one-at-a-time engine's turn is given back once, by the request holding it (field report on image 2026-10-05:
a background reply cancelled while waiting to replay gave the turn a second time, a second foreground request entered
the TP4 engine beside the first, and the four ranks hung on mismatched collectives)."""

import threading
import time

import pytest

from tensorfold.cuda.server import RequestCancelled
from tensorfold.cuda.turns import Turns
from tests.test_cuda_priority import Steady, _body, _until


def test_a_give_from_a_request_not_holding_the_turn_is_ignored():
    turns = Turns()
    holder_in, holder_out = threading.Event(), threading.Event()

    def holder():                                             # B: takes the engine and keeps it
        turns.take(False)
        holder_in.set()
        holder_out.wait(10)
        turns.give()

    b = threading.Thread(target=holder)
    b.start()
    assert holder_in.wait(10)
    gone = threading.Event()
    try:                                                      # A: a replay's take, cancelled while it waits
        threading.Thread(target=lambda: (time.sleep(0.05), gone.set())).start()
        with pytest.raises(RequestCancelled):
            turns.take(True, gone.is_set)
    finally:
        turns.give()                                          # the server's finally: A holds nothing
    assert turns.busy                                         # B still holds it
    entered = threading.Event()
    c = threading.Thread(target=lambda: (turns.take(False), entered.set(), turns.give()))
    c.start()
    assert not entered.wait(0.3)                              # C waits for B (before the fix it entered at once)
    holder_out.set()
    assert entered.wait(10)
    b.join()
    c.join()
    assert not turns.busy


class Inside(Steady):
    """Steady, counting the requests inside ``generate`` at once."""

    def __init__(self):
        super().__init__()
        self.inside, self.most = 0, 0
        self.lock = threading.Lock()

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        with self.lock:
            self.inside += 1
            self.most = max(self.most, self.inside)
        try:
            return super().generate(prompt, max_tokens, sampling, on_tokens, draft)
        finally:
            with self.lock:
                self.inside -= 1


def test_a_background_reply_cancelled_while_waiting_to_replay_lets_one_request_in_at_a_time(tmp_path):
    """The field repro without a GPU: background request, a foreground one cuts it, the background client leaves while
    waiting to replay, a second foreground request arrives - never two requests in the engine at once."""

    from tests.test_cuda_tool_choice import app_for

    engine = Inside()
    app = app_for(tmp_path, engine)
    turns = app._turns()
    deltas, left, errors = [], threading.Event(), []

    def run(body, cancelled=None, sink=None):
        try:
            app.run(body, True, (lambda d: sink.append(d) or True) if sink is not None else (lambda d: True),
                    cancelled=cancelled)
        except RequestCancelled:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    back = threading.Thread(target=run, args=(_body("tell me a story", 200, True), left.is_set, deltas))
    back.start()
    _until(lambda: len(deltas) >= 5)                          # the background reply decodes
    front1 = threading.Thread(target=run, args=(_body("first question", 1000, False),))
    front1.start()                                            # it cuts the background reply, which waits to replay
    _until(lambda: len(engine.prompts) >= 2 and turns.parked >= 1)
    left.set()                                                # the background client leaves while it waits
    back.join(10)
    assert not back.is_alive()
    front2 = threading.Thread(target=run, args=(_body("second question", 20, False),))
    front2.start()
    front1.join(20)
    front2.join(20)
    assert not errors
    assert engine.most == 1                                   # before the fix: 2 (the second entered beside the first)
    assert not turns.busy


def test_a_multi_rank_glm53_engine_never_yields(tmp_path):
    """Glm53Engine says it runs on several ranks, so the server never cuts its background replies (it had no ``tp``:
    read as one rank)."""

    import ast
    import pathlib

    src = pathlib.Path(__file__).parents[1] / "src/tensorfold/families/glm_moe_dsa/cuda/engine.py"
    tree = ast.parse(src.read_text())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert any(isinstance(t, ast.Attribute) and t.attr == "tp" for n in ast.walk(init) if isinstance(n, ast.Assign)
               for t in n.targets)
