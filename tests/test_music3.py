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
