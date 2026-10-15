"""Drama 3 pieces that need no weights: prompt rows, text chunking, sampling rules."""

import numpy as np
import pytest

from tensorfold.families.drama3.prompt import Reference, build_rows, check_text, with_speaker


class FakeTokenizer:
    """One id per character (offset 1000); the semantic-begin token at 248099."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def token_to_id(self, token):
        return 248099 if token == "<|semantic:0|>" else None

    def encode(self, text):
        return self._Enc([1000 + ord(ch) for ch in text])


def test_rows_text_and_reference():
    codes = np.arange(20).reshape(10, 2) % 7
    rows = build_rows("hi", [Reference("ref", codes)], FakeTokenizer())
    assert rows.shape[1] == 11
    audio = rows[rows[:, 0] >= 248099]
    assert audio.shape[0] == 2
    assert (audio[:, 0] == 248099 + codes[0]).all() and (audio[:, 1:] == codes.T).all()
    text = rows[rows[:, 0] < 248099]
    assert (text[:, 1:] == 0).all()


def test_speaker_tags_and_structural_tokens():
    assert with_speaker("hello") == "<|speaker:0|> hello"
    assert with_speaker("  <|speaker:3|> hi") == "  <|speaker:3|> hi"
    with pytest.raises(ValueError):
        check_text("<|im_end|>", "x")
    with pytest.raises(ValueError):
        check_text("<|speaker:16|> no", "x")


def test_split_text_carries_speaker():
    pytest.importorskip("mlx.core")
    from tensorfold.families.drama3.generate import split_text

    text = "<|speaker:1|> " + " ".join(f"Sentence number {i} is here." for i in range(30))
    chunks = split_text(text, 120)
    assert len(chunks) > 1 and all(c.lstrip().startswith("<|speaker:1|>") for c in chunks)
    assert all(len(c) <= 140 for c in chunks)


def test_top_p_keeps_rank0_and_drops_crossing_token():
    mx = pytest.importorskip("mlx.core")
    from tensorfold.families.drama3.sampling import draw

    # probabilities ~ [0.95, 0.05, ...]: the inclusive cumulative of rank 0 already exceeds 0.9, so only it stays
    logits = mx.array([[5.0, 2.0, -10.0, -10.0]])
    seen = {int(draw(logits, 1.0, 30, 0.9, mx.random.key(s))[0]) for s in range(50)}
    assert seen == {0}
    assert int(draw(logits, 1.0, 30, 0.9, None, greedy=True)[0]) == 0
