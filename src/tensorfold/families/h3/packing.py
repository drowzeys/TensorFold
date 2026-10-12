"""The packed H3 sequence: row order, rotary positions, modality tags and latent patching."""

# Adapted from minimax-h3-mlx (Apache-2.0, https://github.com/mrbizarro/minimax-h3-mlx, revision 7919020).

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2
AUDIO_CHANNELS = 2
KEYFRAME_NOISE = 0.999  # conditioning rows are held at this timestep for every step

# One latent frame spans 5/3 of the frames it covers, in the VAE's 17-frames-to-5-latents grouping; the
# spatial axes are normalised by the square root of the latent area and scaled by 32.
_FRAME_RESCALE = 5.0 / 3.0
_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_SPATIAL_SCALE = 32


@dataclass
class Layout:
    """Rows of one packed sequence, in the order [text | keyframe conditions | audio | video], or with
    references [text | reference blocks | audio | video]."""

    rows: int
    position_ids: mx.array  # (rows, 3) float32 over (t, h, w)
    tags: mx.array  # (rows,) modality per row
    video_rows: mx.array  # conditioning rows first, then generated rows
    audio_rows: mx.array  # reference soundtrack rows first, then generated rows
    text_rows: mx.array
    condition_video_rows: int
    condition_audio_rows: int = 0


AUDIO_CONDITION_TIMESTEP = 1.0  # reference audio rows are clean and held at this timestep for every step


@dataclass(frozen=True)
class Reference:
    """Latent geometry of one Ref2VA reference, in request order.

    ``kind`` is ``image`` (one latent frame), ``video`` (``latent_frames`` of them, with ``audio_latents`` of its
    own soundtrack or 0) or ``audio`` (``audio_latents`` alone). Height and width are the reference's own latent
    size, never the clip's.
    """

    kind: str
    latent_height: int = 0
    latent_width: int = 0
    latent_frames: int = 1
    audio_latents: int = 0

    def video_rows(self, patch: tuple[int, int, int]) -> int:
        if self.kind == "audio":
            return 0
        return self.latent_frames * (self.latent_height // patch[1]) * (self.latent_width // patch[2])

    @property
    def audio_rows(self) -> int:
        return self.audio_latents * AUDIO_CHANNELS


def patchify(latents: mx.array, patch: tuple[int, int, int]) -> mx.array:
    """(B, C, F, H, W) video latents to (rows, C * patch volume), frame-major then row-major."""

    pt, ph, pw = patch
    b, c, f, h, w = latents.shape
    x = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw).transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * pt * ph * pw)


def unpatchify(rows: mx.array, frames: int, height: int, width: int, channels: int,
               patch: tuple[int, int, int]) -> mx.array:
    """Inverse of :func:`patchify`: (B, C, F, H, W)."""

    pt, ph, pw = patch
    x = rows.reshape(-1, frames // pt, height // ph, width // pw, channels, pt, ph, pw)
    return x.transpose(0, 4, 1, 5, 2, 6, 3, 7).reshape(-1, channels, frames, height, width)


def unpack_audio(rows: mx.array, latents: int) -> mx.array:
    """Channel-major audio rows to (2, latent channels, latents): one batch item per stereo channel."""

    return rows.reshape(AUDIO_CHANNELS, latents, rows.shape[-1]).transpose(0, 2, 1)


def _spatial_grid(size: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = size / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, size // patch, endpoint=False) * _SPATIAL_SCALE


def _frame_times(frames: int, origin: float) -> np.ndarray:
    spans = np.array([_FRAME_RESCALE * _FRAMES_PER_LATENT[i % 5] for i in range(frames)], dtype=np.float64)
    return origin + np.concatenate([np.zeros(1), np.cumsum(spans[:-1])])


def _time_span(frames: int) -> float:
    # summed pairwise, as the reference computes the keyframe anchor; sequential summation differs in the last
    # ulp from 16 latent frames on
    spans = np.ones(frames, dtype=np.float64) * _FRAME_RESCALE
    for i in range(5):
        spans[i::5] *= _FRAMES_PER_LATENT[i]
    return float(spans.sum())


def layout(text_tags, latent_frames: int, latent_height: int, latent_width: int, audio_latents: int,
           patch: tuple[int, int, int], keyframes: tuple[str, ...] = ()) -> Layout:
    """Build the packed layout. ``text_tags`` is the modality of every text row (a keyframe's vision block is
    tagged video); ``keyframes`` names each conditioning block, ``first`` or ``last``."""

    _, ph, pw = patch
    text_tags = np.asarray(text_tags, dtype=np.int64)
    per_frame = (latent_height // ph) * (latent_width // pw)
    text = int(text_tags.shape[0])
    condition = len(keyframes) * per_frame
    audio = audio_latents * AUDIO_CHANNELS
    video = latent_frames * per_frame
    rows = text + condition + audio + video
    audio_start = text + condition
    video_start = audio_start + audio

    # text rows sit on the time axis at their own index and the media rows continue from there
    positions = np.zeros((rows, 3), dtype=np.float64)
    positions[:text, 0] = np.arange(text)
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_grid(latent_height, ph, sqrt_area)
    width_grid = _spatial_grid(latent_width, pw, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    frame = np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)

    for index, anchor in enumerate(keyframes):
        if anchor not in ("first", "last"):
            raise ValueError(f"a keyframe anchor is 'first' or 'last', got {anchor!r}")
        when = float(text) if anchor == "first" else float(text) + _time_span(latent_frames) - _FRAME_RESCALE
        lo = text + index * per_frame
        positions[lo:lo + per_frame, 0] = when
        positions[lo:lo + per_frame, 1:] = frame

    # audio shares the video clock (40 latents a second equals 24 fps * 5/3), has no height and sits at the
    # two ends of the width grid, one per stereo channel
    positions[audio_start:video_start, 0] = np.tile(float(text) + np.arange(audio_latents), AUDIO_CHANNELS)
    positions[audio_start:video_start, 2] = np.concatenate([np.full(audio_latents, width_grid[0]),
                                                            np.full(audio - audio_latents, width_grid[-1])])
    video_positions = np.empty((latent_frames, per_frame, 3), dtype=np.float64)
    video_positions[:, :, 0] = _frame_times(latent_frames, float(text))[:, None]
    video_positions[:, :, 1:] = frame[None]
    positions[video_start:] = video_positions.reshape(-1, 3)

    video_rows = np.concatenate([np.arange(text, audio_start), np.arange(video_start, rows)])
    audio_rows = np.arange(audio_start, video_start)
    tags = np.empty(rows, dtype=np.int64)
    tags[:text] = text_tags
    tags[audio_rows] = TAG_AUDIO
    tags[video_rows] = TAG_VIDEO
    return Layout(rows, mx.array(positions.astype(np.float32)), mx.array(tags.astype(np.int32)),
                  mx.array(video_rows.astype(np.int32)), mx.array(audio_rows.astype(np.int32)),
                  mx.array(np.arange(text, dtype=np.int32)), condition)


def _frame_grid(latent_height: int, latent_width: int, patch: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(height, width) coordinates of one latent frame's rows, and the width axis they were built from."""

    _, ph, pw = patch
    sqrt_area = np.sqrt(latent_height * latent_width)
    width_grid = _spatial_grid(latent_width, pw, sqrt_area)
    hh, ww = np.meshgrid(_spatial_grid(latent_height, ph, sqrt_area), width_grid, indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1), width_grid


def _place_audio(positions: np.ndarray, lo: int, latents: int, origin: float, width_grid: np.ndarray) -> None:
    """One channel-major audio block: no height, the two ends of a width grid, one clock unit per latent."""

    positions[lo:lo + latents * AUDIO_CHANNELS, 0] = np.tile(origin + np.arange(latents), AUDIO_CHANNELS)
    positions[lo:lo + latents * AUDIO_CHANNELS, 2] = np.concatenate([np.full(latents, width_grid[0]),
                                                                     np.full(latents, width_grid[-1])])


def reference_layout(text_tags, references, latent_frames: int, latent_height: int, latent_width: int,
                     audio_latents: int, patch: tuple[int, int, int]) -> Layout:
    """The Ref2VA layout: [text | one block per reference, in request order | audio | video].

    After MiniMax's ``build_ref2va_packed_sequence``. The references share one clock that starts where the text
    ends: a picture is one latent frame on its own height/width grid and takes one unit; an audio clip takes one
    unit per latent and sits at the two ends of the clip's width grid; a video packs its soundtrack rows (at the
    ends of its own width grid) immediately before its frames, both from the same origin, and takes the longer of
    the two. The clip's audio and video start where the references leave the clock. Reference rows are counted
    in ``condition_video_rows`` and ``condition_audio_rows`` and are never stepped.
    """

    references = tuple(references)
    text_tags = np.asarray(text_tags, dtype=np.int64)
    text = int(text_tags.shape[0])
    per_frame = (latent_height // patch[1]) * (latent_width // patch[2])
    audio = audio_latents * AUDIO_CHANNELS
    held_video = sum(reference.video_rows(patch) for reference in references)
    held_audio = sum(reference.audio_rows for reference in references)
    audio_start = text + held_video + held_audio
    video_start = audio_start + audio
    rows = video_start + latent_frames * per_frame

    positions = np.zeros((rows, 3), dtype=np.float64)
    positions[:text, 0] = np.arange(text)
    frame, width_grid = _frame_grid(latent_height, latent_width, patch)
    video_index, audio_index = [], []
    cursor, clock = text, float(text)
    for reference in references:
        if reference.kind not in ("image", "video", "audio"):
            raise ValueError(f"a reference is an image, a video or an audio clip, got {reference.kind!r}")
        if reference.kind == "audio":
            if reference.audio_latents < 1:
                raise ValueError("an audio reference needs at least one audio latent")
            _place_audio(positions, cursor, reference.audio_latents, clock, width_grid)
            audio_index.append(np.arange(cursor, cursor + reference.audio_rows))
            cursor += reference.audio_rows
            clock += float(reference.audio_latents)
            continue
        own, own_width = _frame_grid(reference.latent_height, reference.latent_width, patch)
        if reference.kind == "image":
            if reference.latent_frames != 1 or reference.audio_latents:
                raise ValueError("a picture reference is one latent frame without sound")
            positions[cursor:cursor + own.shape[0], 0] = clock
            positions[cursor:cursor + own.shape[0], 1:] = own
            video_index.append(np.arange(cursor, cursor + own.shape[0]))
            cursor += own.shape[0]
            clock += 1.0  # a picture takes one whole unit, not a latent frame's 5/3
            continue
        if reference.audio_latents:
            _place_audio(positions, cursor, reference.audio_latents, clock, own_width)
            audio_index.append(np.arange(cursor, cursor + reference.audio_rows))
            cursor += reference.audio_rows
        count = reference.latent_frames * own.shape[0]
        positions[cursor:cursor + count, 0] = np.repeat(_frame_times(reference.latent_frames, clock), own.shape[0])
        positions[cursor:cursor + count, 1:] = np.tile(own, (reference.latent_frames, 1))
        video_index.append(np.arange(cursor, cursor + count))
        cursor += count
        # summed in sequence here, unlike the keyframe anchor's pairwise sum: the reference keeps both
        span = sum(_FRAME_RESCALE * _FRAMES_PER_LATENT[i % 5] for i in range(reference.latent_frames))
        clock += max(float(reference.audio_latents), span)

    _place_audio(positions, audio_start, audio_latents, clock, width_grid)
    positions[video_start:, 0] = np.repeat(_frame_times(latent_frames, clock), per_frame)
    positions[video_start:, 1:] = np.tile(frame, (latent_frames, 1))

    video_rows = np.concatenate([*video_index, np.arange(video_start, rows)]).astype(np.int64)
    audio_rows = np.concatenate([*audio_index, np.arange(audio_start, video_start)]).astype(np.int64)
    tags = np.empty(rows, dtype=np.int64)
    tags[:text] = text_tags
    tags[audio_rows] = TAG_AUDIO
    tags[video_rows] = TAG_VIDEO
    return Layout(rows, mx.array(positions.astype(np.float32)), mx.array(tags.astype(np.int32)),
                  mx.array(video_rows.astype(np.int32)), mx.array(audio_rows.astype(np.int32)),
                  mx.array(np.arange(text, dtype=np.int32)), held_video, held_audio)


def timestep_plan(packed: Layout, video_timesteps, audio_timesteps) -> tuple[mx.array, list[mx.array]]:
    """One table of the distinct timesteps of a run and, for every step, each row's index into it.

    Generated video and audio rows follow their own schedules inside one forward; conditioning video rows stay
    at ``max(t, KEYFRAME_NOISE)`` and reference audio rows at ``AUDIO_CONDITION_TIMESTEP``; text rows never reach
    an output and take the video timestep.
    """

    video_index = np.asarray(packed.video_rows.tolist(), dtype=np.int64)
    audio_index = np.asarray(packed.audio_rows.tolist(), dtype=np.int64)
    steps = []
    for video_t, audio_t in zip(video_timesteps, audio_timesteps, strict=True):
        per_row = np.full(packed.rows, np.float32(video_t), dtype=np.float32)
        per_row[video_index[:packed.condition_video_rows]] = np.float32(max(float(video_t), KEYFRAME_NOISE))
        per_row[audio_index] = np.float32(audio_t)
        per_row[audio_index[:packed.condition_audio_rows]] = np.float32(AUDIO_CONDITION_TIMESTEP)
        steps.append(per_row)
    table = np.unique(np.concatenate(steps))
    return mx.array(table), [mx.array(np.searchsorted(table, per_row).astype(np.int32)) for per_row in steps]
