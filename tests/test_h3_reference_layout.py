"""The Ref2VA layout: reference blocks in request order on one clock, ahead of the clip's audio and video."""

import numpy as np
import pytest

from tensorfold.families.h3.packing import (AUDIO_CONDITION_TIMESTEP, KEYFRAME_NOISE, TAG_AUDIO, TAG_TEXT, TAG_VIDEO,
                                            Reference, layout, reference_layout, timestep_plan)

PATCH = (1, 2, 2)


def build(references, text=3, frames=7, height=4, width=6, audio=10):
    return reference_layout([TAG_TEXT] * text, references, frames, height, width, audio, PATCH)


def test_without_references_it_is_the_plain_layout():
    ours, plain = build(()), layout([TAG_TEXT] * 3, 7, 4, 6, 10, PATCH)
    assert np.array_equal(np.array(ours.position_ids), np.array(plain.position_ids))
    assert np.array_equal(np.array(ours.tags), np.array(plain.tags))
    assert ours.video_rows.tolist() == plain.video_rows.tolist()
    assert ours.audio_rows.tolist() == plain.audio_rows.tolist()


def test_a_picture_takes_one_unit_of_the_clock():
    packed = build((Reference("image", 2, 8), Reference("image", 4, 4)))
    positions = np.array(packed.position_ids)
    assert packed.condition_video_rows == 4 + 4 and packed.condition_audio_rows == 0
    assert positions[3:7, 0].tolist() == [3.0] * 4 and positions[7:11, 0].tolist() == [4.0] * 4
    # the clip's audio and its first frame start after both pictures
    assert positions[11, 0] == 5.0 and positions[11 + 20, 0] == 5.0
    # each picture sits on a grid normalised by its own area: a 2x8 latent is one row of four columns
    assert len(set(positions[3:7, 1].tolist())) == 1 and len(set(positions[3:7, 2].tolist())) == 4
    assert packed.video_rows.tolist()[:8] == list(range(3, 11))


def test_an_audio_clip_takes_one_unit_per_latent_at_the_ends_of_the_clips_width():
    packed = build((Reference("image", 2, 2), Reference("audio", audio_latents=5)))
    positions, plain = np.array(packed.position_ids), np.array(layout([TAG_TEXT] * 3, 7, 4, 6, 10, PATCH).position_ids)
    block = positions[4:14]
    assert block[:, 0].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0] * 2 and not block[:, 1].any()
    assert set(block[:5, 2].tolist()) == {plain[3, 2]} and set(block[5:, 2].tolist()) == {plain[3 + 10, 2]}
    assert (packed.condition_video_rows, packed.condition_audio_rows) == (1, 10)
    assert packed.audio_rows.tolist()[:10] == list(range(4, 14))
    assert positions[14, 0] == 9.0  # 3 text + 1 picture + 5 latents
    assert np.array(packed.tags)[4:14].tolist() == [TAG_AUDIO] * 10


def test_a_video_packs_its_soundtrack_first_and_takes_the_longer_span():
    video = Reference("video", 2, 4, latent_frames=7, audio_latents=40)
    packed = build((video,))
    positions = np.array(packed.position_ids)
    sound, frames = positions[3:83], positions[83:83 + 14]
    assert sound[:40, 0].tolist() == [3.0 + i for i in range(40)]
    assert len(set(sound[:40, 2].tolist())) == 1 and sound[0, 2] != sound[40, 2]
    # the soundtrack sits at the ends of the video's own width grid, not the clip's
    assert sound[0, 2] == frames[0, 2] and sound[40, 2] == frames[1, 2]
    times = [3.0 + 5.0 / 3.0 * step for step in (0, 1, 5, 9, 13, 17, 18)]
    assert np.allclose(frames[::2, 0], times, rtol=0, atol=1e-5)
    assert np.array(packed.tags)[83:97].tolist() == [TAG_VIDEO] * 14
    # 40 latents are longer than seven frames (5/3 * 22 = 36.7), so the clip starts 40 units on
    assert positions[97, 0] == 43.0
    silent = np.array(build((Reference("video", 2, 4, latent_frames=7),)).position_ids)
    assert np.isclose(silent[3 + 14, 0], 3.0 + 5.0 / 3.0 * 22)


def test_reference_audio_rows_are_held_clean_and_video_rows_at_the_keyframe_level():
    packed = build((Reference("video", 2, 2, latent_frames=2, audio_latents=4), Reference("audio", audio_latents=2)))
    table, plan = timestep_plan(packed, [0.1, 0.5], [0.2, 0.6])
    for index, (video_t, audio_t) in enumerate(((0.1, 0.2), (0.5, 0.6))):
        per_row = np.array(table)[np.array(plan[index])]
        video_rows, audio_rows = np.array(packed.video_rows), np.array(packed.audio_rows)
        assert np.allclose(per_row[video_rows[:2]], KEYFRAME_NOISE) and np.allclose(per_row[video_rows[2:]], video_t)
        assert np.allclose(per_row[audio_rows[:12]], AUDIO_CONDITION_TIMESTEP)
        assert np.allclose(per_row[audio_rows[12:]], audio_t)


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        build((Reference("picture", 2, 2),))
