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


def test_fused_projections_match_separate():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from tensorfold.families.drama3.fast import _fused_linear, _project
    from tensorfold.families.drama3.model import FeedForward

    parts = [nn.Linear(64, n, bias=False) for n in (48, 16, 8)]
    fused = _fused_linear(*parts)
    for rows in (1, 7):
        x = mx.random.normal((1, rows, 64), key=mx.random.key(rows))
        for got, part in zip(_project(fused, x, [48, 16, 8]), parts):
            assert mx.allclose(got, part(x), atol=1e-5).item()
    ff = FeedForward(64, 96)
    x = mx.random.normal((1, 3, 64), key=mx.random.key(0))
    before = ff(x)
    ff.fuse()
    assert "w1" not in ff and mx.allclose(ff(x), before, atol=1e-5).item()


def test_end_of_text_check():
    from tensorfold.families.drama3.verify import Word, check_words, ends_with_direction, words_of

    assert words_of("<|speaker:1|> [sighs] It's 41 days, 2nd!") == ["it's", "forty", "one", "days", "second"]
    assert ends_with_direction("Fine. [laughs]") and not ends_with_direction("[laughs] Fine.")
    heard = [Word(t, s, s + 0.3) for t, s in ((" Hello", 0.0), (" there.", 0.4), (" This", 1.0), (" is", 1.3),
                                                (" Tensor", 1.6), (" Fold.", 1.9), (" And", 2.6), (" then.", 2.9))]
    text = "Hello there. This is TensorFold."
    clean = check_words(text, heard[:6], 2.5)
    assert clean.action == "ok" and clean.coverage == 1.0
    runaway = check_words(text, heard, 3.5)
    assert runaway.action == "trim" and runaway.tail_words == ["and", "then"] and 2.2 <= runaway.cut_s <= 2.5
    dropped = check_words(text + " Goodbye now, my friend.", heard[:6], 2.5)
    assert dropped.action == "redraw" and dropped.ending_missing
    long_tail = check_words(text, heard[:6], 6.0)
    assert long_tail.action == "trim" and abs(long_tail.cut_s - 2.5) < 1e-6
    assert check_words(text + " [laughs]", heard[:6], 4.0).action == "ok"


def test_stop_rule_missed_stop():
    pytest.importorskip("mlx.core")
    from tensorfold.families.drama3.generate import StopRule

    rule = StopRule(min_frames=10, max_overrun=5, pad_frames=2)
    assert not any(rule(n, [0.5 if n == 4 else 0.01]) for n in range(1, 10))   # before min_frames: ignored
    assert rule.hold_from is None
    assert not rule(12, [0.3]) and rule.hold_from == 11 and rule.cut_frames == 13
    assert not any(rule(n, [0.0]) for n in range(13, 17))
    assert rule(17, [0.0])                                                     # 6 frames past the candidate


def test_streamed_wav_bytes(tmp_path):
    pytest.importorskip("mlx.core")
    import soundfile

    from tensorfold.families.drama3.stream import GrowingWav, pcm16, wav_header, wav_stream

    chunks = [np.linspace(-1, 1, 300, dtype=np.float32), np.zeros(100, dtype=np.float32)]
    data = b"".join(wav_stream(chunks))
    assert data[:4] == b"RIFF" and len(data) == 44 + 2 * 400 and data[44:] == pcm16(np.concatenate(chunks))
    assert wav_header(800)[40:44] == (800).to_bytes(4, "little")
    path = tmp_path / "grow.wav"
    wav = GrowingWav(path)
    for c in chunks:
        wav.write(c)
    wav.close()
    audio, rate = soundfile.read(str(path), dtype="float32")
    assert rate == 44100 and len(audio) == 400 and abs(audio[0] + 1) < 1e-3


def test_dialogue_turns_on_lines_and_chunks():
    pytest.importorskip("mlx.core")
    from tensorfold.families.drama3.generate import split_text, turns_on_lines

    one = "<|speaker:0|> [whispering] Hi there. <|speaker:1|> [sighs] Go back to sleep. <|speaker:0|> No."
    assert turns_on_lines(one) == ("<|speaker:0|> [whispering] Hi there.\n<|speaker:1|> [sighs] Go back to sleep."
                                   "\n<|speaker:0|> No.")
    assert turns_on_lines("<|speaker:3|> Just one. Two sentences.") == "<|speaker:3|> Just one. Two sentences."
    assert split_text(one, 300) == [turns_on_lines(one)]
    long = " ".join(f"<|speaker:{i % 2}|> Turn number {i} is said here. And a second sentence {i}." for i in range(12))
    chunks = split_text(long, 150)
    assert len(chunks) > 2
    for chunk in chunks:
        assert all(line.startswith("<|speaker:") for line in chunk.split("\n")), chunk


def test_missing_last_content_word_redraws():
    from tensorfold.families.drama3.verify import Word, check_words

    def heard(text):
        return [Word(w, 0.3 * i, 0.3 * i + 0.25) for i, w in enumerate(text.split())]

    text = "Did you hear that? It's probably just the cat."
    got = check_words(text, heard("Did you hear that? It's probably just the"), 2.6)
    assert got.action == "redraw" and got.ending_missing
    question = "Did you hear that noise? Where are you going to?"
    assert check_words(question, heard("Did you hear that noise? Where are you going"), 2.9).action == "ok"


def test_serve_arguments():
    from tensorfold.drama3 import parse

    assert parse(["--serve", "1"]).serve and parse(["--serve"]).serve and not parse(["--text", "x", "--out", "o"]).serve
    with pytest.raises(SystemExit):
        parse(["--text", "x"])


def test_serve_hears_a_reference_left_without_its_transcript(tmp_path):
    """--serve: a reference without ref_text is heard by the server's Whisper, once per file (no weights here)."""
    import argparse

    import numpy as np
    import soundfile

    from tensorfold import drama3
    from tensorfold.families.drama3.verify import Word

    clip = tmp_path / "ref.wav"
    soundfile.write(str(clip), np.zeros(22050, dtype=np.float32), 22050)
    calls = []

    class Heard:
        def words(self, audio, sample_rate=44100, language=None):
            calls.append(len(audio))
            return [Word(" Hello", 0.0, 0.4), Word(" there.", 0.4, 0.8)]

    cache, logged = {}, []
    text, spent = drama3.reference_transcript(clip, Heard(), argparse.Namespace(asr="auto"), cache, logged.append)
    assert text == "Hello there." and spent >= 0 and calls == [44100]
    assert drama3.reference_transcript(clip, Heard(), argparse.Namespace(asr="auto"), cache, logged.append) == ("Hello there.", 0.0)
    assert calls == [44100]  # cached by path and mtime

    class Silent:
        def words(self, audio, sample_rate=44100, language=None):
            return []

    import pytest

    with pytest.raises(ValueError, match="no speech"):
        drama3.reference_transcript(clip, Silent(), argparse.Namespace(asr="auto"), {}, logged.append)
