"""MiniMax Music 3 checks that need no weights (CPU)."""

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.families.music3 import prompt as P  # noqa: E402
from tensorfold.families.music3.generate import derive_seed, sigma_schedule, windows  # noqa: E402
from tensorfold.families.music3.model import frames_to_latents, nearest, rotate  # noqa: E402


def test_prompt_layout():
    text = P.prompt_text("# Hip-hop\n**male** rap  <|bpm 92|>", "[Verse 1] one ^ two\n[CHORUS]\nhey")
    assert text == ("<|im_start|><|caption_start|>Hip-hop\nmale rap  bpm is 92<|caption_end|><|lyrics_start|>"
                    "[start]\n[verse 1]\none\ntwo\n[chorus]\nhey<|lyrics_end|><|im_end|><|audio_start|>")


def test_unconditional_keeps_ends():
    ids = [1, 2, 3, 4, 5, 6]
    assert P.unconditional(ids) == [1, 151654, 151654, 151654, 5, 6]


def test_timeline():
    assert frames_to_latents(200) == 689 and frames_to_latents(100) == 344
    assert windows(689) == [(0, 689)]
    assert windows(1000) == [(0, 689), (344, 1000)]
    assert nearest(3, 7) == [0, 0, 0, 1, 1, 2, 2]


def test_schedule_and_seed():
    s = sigma_schedule(30)
    assert len(s) == 31 and s[0] == 1.0 and s[-1] == 0.0 and abs(s[3] - 0.9) < 1e-6
    assert derive_seed(222, "ar") == derive_seed(222, "ar") != derive_seed(222, "dit")


def test_rope_split_halves():
    mx.set_default_device(mx.cpu)
    x = mx.random.normal((1, 1, 3, 8))
    angles = mx.arange(3, dtype=mx.float32)[:, None] * mx.array([1.0, 0.5])[None]
    out = np.array(rotate(x, mx.cos(angles), mx.sin(angles)))
    xn = np.array(x)
    a, b = xn[..., 0], xn[..., 2]
    t = 2 * 1.0
    assert np.allclose(out[0, 0, 2, 0], a[0, 0, 2] * math.cos(t) - b[0, 0, 2] * math.sin(t), atol=1e-5)
    assert np.allclose(out[..., 4:], xn[..., 4:])


# -- quantization, sequential windows, streaming, the lyric check ---------------------------------------------------

def test_parse_quant():
    from tensorfold.families.music3.model import parse_quant, quant_name

    assert quant_name(parse_quant(None)) == "none" == quant_name(parse_quant("none"))
    plan = parse_quant("int8")
    assert (plan["body"], plan["head"], plan["depth"], plan["dit"], plan["group"]) == (8, 8, 8, 0, 64)
    assert parse_quant("int8-d4")["depth"] == 4
    plan = parse_quant("int8,dit=8,group=128")
    assert plan["dit"] == 8 and plan["group"] == 128
    assert quant_name(parse_quant("body=8,depth=6")) == "body=8,depth=6,group=64"
    for bad in ("int7", "body=7", "group=48", "tail=8"):
        with pytest.raises(ValueError):
            parse_quant(bad)


def test_quantize_only_linears():
    import mlx.nn as nn

    from tensorfold.families.music3.model import _linears

    mx.set_default_device(mx.cpu)
    m = nn.Sequential(nn.Embedding(16, 64), nn.Linear(64, 64, bias=False))
    nn.quantize(m, group_size=64, bits=8, class_predicate=_linears)
    assert isinstance(m.layers[0], nn.Embedding) and isinstance(m.layers[1], nn.QuantizedLinear)


def test_sequential_starts_and_crops():
    from tensorfold.families.music3.generate import sequential_crop, sequential_starts

    assert sequential_starts(150) == [0] and sequential_starts(200) == [0]
    assert sequential_starts(201) == [0, 100] and sequential_starts(300) == [0, 100]
    assert sequential_starts(301) == [0, 100, 200]
    assert sequential_starts(1500) == list(range(0, 1400, 100))
    assert sequential_crop(689, True, False) == (0, 431)
    assert sequential_crop(689, False, False) == (86, 431)
    assert sequential_crop(400, False, True) == (86, 400)
    assert sequential_crop(689, True, True) == (0, 689)


class _FakeGen:
    """Stands in for Music3Generator: a window's latents carry their global latent position."""

    def __init__(self):
        self.calls = []

    def denoise_window(self, states, index, s, previous=None, noise=None, progress=None):
        from tensorfold.families.music3.model import frames_to_latents

        start = int(np.array(states[0, 0]))
        length = frames_to_latents(states.shape[0])
        self.calls.append((start, states.shape[0], previous is not None))
        x = mx.broadcast_to((mx.arange(length, dtype=mx.float32) + start * 1000)[:, None], (length, 128))
        return x, (x[-344:-172], x[-344:-172])

    def window_steps(self, states, index, s, previous=None, noise=None, stream=None):
        for _ in range(s.steps):
            yield
        return self.denoise_window(states, index, s, previous, noise)

    def decode_raw(self, latents, span=None):
        a, b = span
        values = np.array(latents[a:b, 0])
        return np.stack([np.repeat(values, 512), np.repeat(values, 512)]).astype(np.float32) * 1e-8


def _states(frames):
    # frame f's state starts with f, so a window knows its first frame
    return mx.broadcast_to(mx.arange(frames, dtype=mx.float32)[:, None], (frames, 4))


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("frames", [150, 200, 250, 300, 301, 1000, 1037])
def test_sequential_stream_equals_offline(frames, overlap):
    from tensorfold.families.music3.generate import (SequentialSong, Settings, frames_to_latents, sequential_crop,
                                                     sequential_starts)

    mx.set_default_device(mx.cpu)
    s = Settings(seconds=frames / 25, overlap=overlap)
    offline = SequentialSong(_FakeGen(), s)
    offline.finish(_states(frames))
    got = []
    gen = _FakeGen()
    streamed = SequentialSong(gen, s, sink=got.append)
    states = _states(frames)
    kept = []
    for f in range(frames):
        kept.append(states[f])
        streamed.ready(kept)
    streamed.finish(states)
    starts = sequential_starts(frames)
    assert [c[0] for c in gen.calls] == starts and offline.windows == streamed.windows == len(starts)
    expected = 0
    for i, start in enumerate(starts):
        length = frames_to_latents(min(start + 200, frames) - start)
        a, b = sequential_crop(length, i == 0, i == len(starts) - 1)
        expected += (b - a) * 512
    a1, a2 = offline.audio(), streamed.audio()
    assert a1.shape == a2.shape == (2, expected)
    assert np.allclose(np.concatenate(got, axis=1), a2)
    # raw pieces equal; only the level rule (whole song vs running) may differ, and here it is 1 for both
    assert np.allclose(np.concatenate(offline.pieces, axis=1), np.concatenate(streamed.pieces, axis=1))
    assert len(streamed.seams) == len(starts) - 1


def test_level_rule_never_falls():
    from tensorfold.families.music3.generate import LevelRule

    rule = LevelRule()
    loud = np.full((2, 1000), 0.5, dtype=np.float32) * np.sign(np.random.default_rng(0).standard_normal((2, 1000)))
    quiet = np.full((2, 1000), 0.01, dtype=np.float32)
    assert np.allclose(rule(quiet.copy()), quiet)                   # 5 x std < 1: untouched
    rule(loud.astype(np.float32))
    level = rule.level
    assert level > 1.0
    rule(quiet.copy())
    assert rule.level == level


def test_stream_files(tmp_path):
    import wave

    from tensorfold.families.music3.generate import GrowingWav, RawPcm, pcm16, wav_header

    assert len(wav_header()) == 44 and wav_header()[4:8] == b"\xff\xff\xff\xff"
    a = np.array([[0.5, -0.5], [0.25, 1.5]], dtype=np.float32)
    assert np.frombuffer(pcm16(a), "<i2").tolist() == [16384, 8192, -16384, 32767]   # interleaved L R L R
    w, r = GrowingWav(tmp_path / "s.wav"), RawPcm(tmp_path / "s.pcm")
    for _ in range(3):
        w.write(a)
        r.write(a)
    head = (tmp_path / "s.wav").read_bytes()[:44]
    assert head[40:44] == b"\xdb\xff\xff\xff"                       # still growing: unknown size
    w.close()
    r.close()
    with wave.open(str(tmp_path / "s.wav")) as f:
        assert (f.getnchannels(), f.getframerate(), f.getnframes()) == (2, 44100, 6)
    assert (tmp_path / "s.pcm").read_bytes() == pcm16(a) * 3


def test_listen_score():
    import importlib.util
    import pathlib
    from types import SimpleNamespace as W

    path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "music3_listen.py"
    spec = importlib.util.spec_from_file_location("music3_listen", path)
    listen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(listen)
    lyrics = "[Verse]\nHello bright world\nwe sing 2 songs\n[Chorus]\nnever heard line"
    heard = [W(text=" Hello,", start=1.0, end=1.4), W(text=" bright", start=1.5, end=1.8), W(text=" word", start=1.9, end=2.2),
             W(text=" yeah", start=2.5, end=2.6), W(text=" we", start=3.0, end=3.1), W(text=" sing", start=3.2, end=3.5),
             W(text=" two", start=3.6, end=3.8), W(text=" songs", start=3.9, end=4.4)]
    out = listen.score(lyrics, heard)
    assert out["lyric_words"] == 10 and out["matched"] == 7 and out["reached_words"] == 7
    assert out["word_accuracy"] == 0.7 and out["sung_accuracy"] == 1.0
    assert out["lines"][0]["start"] == 1.0 and out["lines"][1]["end"] == 4.4 and out["lines"][2]["matched"] == 0
    assert out["lyrics"][6] == {"word": "songs", "line": 1, "start": 3.9, "end": 4.4}


def test_listen_repeated_chorus_matches_first():
    import importlib.util
    import pathlib
    from types import SimpleNamespace as W

    path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "music3_listen.py"
    spec = importlib.util.spec_from_file_location("music3_listen", path)
    listen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(listen)
    lyrics = "[Verse]\none two three\n[Chorus]\nla la hey\n[Verse]\nfour five six\n[Chorus]\nla la hey"
    heard = [W(text=t, start=k, end=k + 0.5) for k, t in enumerate("one two three la la hey".split())]
    out = listen.score(lyrics, heard)
    assert out["matched"] == 6 and out["reached_words"] == 6 and out["sung_accuracy"] == 1.0


def test_listen_union_and_stray_match():
    import importlib.util
    import pathlib
    from types import SimpleNamespace as W

    path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "music3_listen.py"
    spec = importlib.util.spec_from_file_location("music3_listen", path)
    listen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(listen)
    lyrics = "a1 b1 c1 d1 e1 f1\ng1 h1 i1 j1\nyeah"
    first = [W(text=t, start=k, end=k + 0.5) for k, t in enumerate("a1 b1 c1 yeah".split())]
    second = [W(text=t, start=k, end=k + 0.5) for k, t in enumerate("d1 e1 f1".split(), start=3)]
    one = listen.score(lyrics, first)
    assert one["matched"] == 4 and one["reached_words"] == 3          # the stray "yeah" does not move "reached"
    both = listen.union(lyrics, [one, listen.score(lyrics, second)])
    assert both["matched"] == 7 and both["reached_words"] == 6 and both["sung_accuracy"] == 1.0
