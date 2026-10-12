"""H3 sampler: first-frame conditioning rows are held, generated rows are stepped."""

import mlx.core as mx
import pytest

from tensorfold.families.h3.config import DiTConfig
from tensorfold.families.h3.sampler import denoise


class FakeDiT:
    """Returns zero velocity and records the row counts it was called with."""

    def __init__(self):
        self.config = DiTConfig(num_layers=1)
        self.calls = []

    def cache_modulation(self, timestep, release=False):
        return 0

    def __call__(self, video, audio, text, timestep, timestep_rows, tags, position_ids, video_rows, audio_rows,
                 text_rows):
        self.calls.append((video.shape[1], audio.shape[1], int(timestep_rows.shape[0])))
        return mx.zeros(video.shape), mx.zeros(audio.shape)


def run(condition=None, keyframes=()):
    dit = FakeDiT()
    text = mx.zeros((1, 3, dit.config.text_dim))
    out = denoise(dit, text, [1, 1, 1], 64, 64, 22, points=3, seed=1, condition=condition, keyframes=keyframes)
    return dit, out


def test_text_to_video_steps_every_video_row():
    dit, out = run()
    assert dit.calls == [(7 * 4, 74, 3 + 74 + 28)] * 2
    assert out.video_rows.shape == (28, 96) and out.audio_rows.shape == (74, 32)


def test_first_frame_rows_are_passed_in_and_not_returned():
    condition = mx.random.normal((4, 96))
    dit, out = run(condition, ("first",))
    assert dit.calls == [(4 + 28, 74, 3 + 4 + 74 + 28)] * 2
    assert out.video_rows.shape == (28, 96)
    assert out.packed.condition_video_rows == 4


def test_condition_rows_must_match_the_keyframes():
    with pytest.raises(ValueError):
        run(mx.zeros((3, 96)), ("first",))
    with pytest.raises(ValueError):
        run(None, ("first",))


def test_references_hold_their_video_and_audio_rows():
    from tensorfold.families.h3.packing import Reference

    references = (Reference("image", 4, 2), Reference("video", 2, 2, latent_frames=2, audio_latents=5),
                  Reference("audio", audio_latents=3))
    dit = FakeDiT()
    text = mx.zeros((1, 3, dit.config.text_dim))
    condition, clean = mx.random.normal((2 + 2, 96)), mx.random.normal((16, 32)) + 7.0
    out = denoise(dit, text, [1, 1, 1], 64, 64, 22, points=3, seed=1, condition=condition, references=references,
                  audio_condition=clean)
    assert dit.calls == [(4 + 28, 16 + 74, 3 + 4 + 16 + 74 + 28)] * 2
    assert out.video_rows.shape == (28, 96) and out.audio_rows.shape == (74, 32)
    assert (out.packed.condition_video_rows, out.packed.condition_audio_rows) == (4, 16)
    assert abs(float(mx.mean(out.audio_rows).item())) < 1.0  # the clean reference rows are not returned


def test_reference_rows_must_match_the_references():
    from tensorfold.families.h3.packing import Reference

    dit = FakeDiT()
    text = mx.zeros((1, 3, dit.config.text_dim))
    with pytest.raises(ValueError):
        denoise(dit, text, [1, 1, 1], 64, 64, 22, points=3, references=(Reference("audio", audio_latents=3),))
    with pytest.raises(ValueError):
        denoise(dit, text, [1, 1, 1], 64, 64, 22, points=3, references=(Reference("image", 2, 2),),
                condition=mx.zeros((1, 96)), keyframes=("first",))
