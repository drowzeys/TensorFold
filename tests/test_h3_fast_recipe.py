"""H3 many-step savings: the velocity cache and attention reuse."""

import mlx.core as mx
import numpy as np

from tensorfold.families.h3.config import DiTConfig
from tensorfold.families.h3.dit import H3DiT
from tensorfold.families.h3.sampler import attention_refresh, denoise, scaled_gate

SMALL = DiTConfig(hidden_size=32, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                  attention_head_dim=16, ffn_hidden_size=48, latents_dim=24, audio_latents_dim=32, text_dim=20,
                  timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=24, adaln_out_features=576,
                  final_adaln_out_features=64, rope_inv_freq_len=2)


def small_model(seed=0):
    mx.random.seed(seed)
    model = H3DiT(SMALL)
    mx.eval(model.parameters())
    return model


def test_gates_scale_with_the_run_and_never_vanish():
    assert [scaled_gate(30, 4), scaled_gate(30, 2), scaled_gate(50, 4)] == [4, 2, 4]
    assert [scaled_gate(20, 4), scaled_gate(20, 2), scaled_gate(4, 4), scaled_gate(4, 2)] == [2, 1, 1, 1]


def test_attention_refresh_schedule_at_thirty_steps():
    reused = [i for i in range(30) if not attention_refresh(i, 30, 2)]
    assert reused == [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
    assert all(attention_refresh(i, 30, 0) for i in range(30))
    assert not all(attention_refresh(i, 8, 2) for i in range(8))


class Still:
    """A model whose velocity is tiny, so every step the cache may skip, it skips."""

    def __init__(self):
        self.config, self.blocks, self.calls = DiTConfig(num_layers=1), [None], 0

    def cache_modulation(self, timestep, release=False):
        return 0

    def __call__(self, video, audio, *rest, **extra):
        self.calls += 1
        return mx.zeros(video.shape), mx.zeros(audio.shape)


def test_velocity_cache_skips_at_most_two_running_and_never_the_ends():
    model = Still()
    text = mx.zeros((1, 3, model.config.text_dim))
    out = denoise(model, text, [1, 1, 1], 64, 64, 22, points=11, seed=1, step_cache=0.05)
    # 10 steps: step 0 runs, then skip, skip, run, skip, skip, run, skip, skip, and the last always runs
    assert model.calls == 4 and out.skipped_steps == 6 and out.reused_steps == 0
    plain = Still()
    denoise(plain, text, [1, 1, 1], 64, 64, 22, points=11, seed=1)
    assert plain.calls == 10


def test_attention_reuse_changes_only_the_reusing_steps():
    text = mx.random.normal((1, 3, SMALL.text_dim))
    plain = denoise(small_model(), text, [1, 1, 1], 64, 64, 22, points=9, seed=3)
    every = denoise(small_model(), text, [1, 1, 1], 64, 64, 22, points=9, seed=3, attention_every=1)
    reuse = denoise(small_model(), text, [1, 1, 1], 64, 64, 22, points=9, seed=3, attention_every=2)
    np.testing.assert_array_equal(np.asarray(plain.video_rows), np.asarray(every.video_rows))
    assert reuse.reused_steps == sum(not attention_refresh(i, 8, 2) for i in range(8)) > 0
    assert np.isfinite(np.asarray(reuse.video_rows)).all()
    assert not np.array_equal(np.asarray(plain.video_rows), np.asarray(reuse.video_rows))


def test_kept_attention_is_read_back_when_not_refreshing():
    model = small_model(4)
    text = mx.random.normal((1, 3, SMALL.text_dim))
    first = denoise(model, text, [1, 1, 1], 64, 64, 22, points=3, seed=2)
    packed = first.packed
    table = mx.array(np.array([0.3], dtype=np.float32))
    plan = mx.zeros((packed.rows,), dtype=mx.int32)
    model.cache_modulation(table)
    args = (text.astype(mx.bfloat16), table, plan, packed.tags, packed.position_ids, packed.video_rows,
            packed.audio_rows, packed.text_rows)
    kept = [None] * SMALL.num_layers
    video, audio = first.video_rows[None], first.audio_rows[None]
    fresh = model(video, audio, *args, kept=kept, refresh=True)
    assert all(slot is not None for slot in kept)
    again = model(video, audio, *args, kept=kept, refresh=False)
    np.testing.assert_allclose(np.asarray(again[0]), np.asarray(fresh[0]), atol=1e-5)
