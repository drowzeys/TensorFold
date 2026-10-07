"""The H3 joint denoise loop: one forward per step moves video and audio rows down their own schedules."""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from . import config as h3
from .dit import MODALITIES
from .packing import AUDIO_CHANNELS, KEYFRAME_NOISE, TAG_AUDIO, Layout, layout, patchify, timestep_plan
from .schedule import AUDIO_SHIFT, VIDEO_SHIFT, Schedule


@dataclass
class Latents:
    """Denoised rows and the geometry needed to decode them."""

    video_rows: mx.array
    audio_rows: mx.array
    packed: Layout
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_latents: int
    step_seconds: list[float]
    reused_steps: int = 0   # attention outputs reused, the rest of the forward run
    skipped_steps: int = 0  # no forward: the previous velocity reused


def start_noise(config, latent_frames: int, latent_height: int, latent_width: int, audio_latents: int, seed: int,
                condition: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """Video then audio noise from one seed, in the reference's draw order.

    Keyframe ``condition`` rows are noised first, from the same generator, to ``KEYFRAME_NOISE`` and placed ahead
    of the generated video rows; they keep that level for the whole run.
    """

    mx.random.seed(seed)
    if condition is not None:
        noise = mx.random.normal(condition.shape).astype(mx.float32)
        level = float(np.float32(KEYFRAME_NOISE))
        condition = level * condition.astype(mx.float32) + float(np.float32(1.0) - np.float32(KEYFRAME_NOISE)) * noise
    video = mx.random.normal((1, config.latents_dim, latent_frames, latent_height, latent_width))
    audio = mx.random.normal((audio_latents * AUDIO_CHANNELS, config.audio_latents_dim))
    video = patchify(video.astype(mx.float32), config.patch_size)
    if condition is not None:
        video = mx.concatenate([condition, video])
    return video, audio.astype(mx.float32)


def check_noise(name: str, rows: mx.array) -> None:
    """Refuse a modality that does not start from noise.

    Video and audio denoise in one sequence, so a constant or zero audio start corrupts the picture as well as
    the sound; the engine this guards against shipped exactly that.
    """

    mean, std = float(mx.mean(rows).item()), float(mx.std(rows).item())
    if not (abs(mean) < 0.05 and 0.9 < std < 1.1):
        raise ValueError(f"{name} start is not unit noise (mean {mean:.3f}, std {std:.3f})")


def scaled_gate(steps: int, at_30: int) -> int:
    """A window ``at_30`` steps wide on a 30-step schedule, scaled to ``steps``: at least 1, never wider."""

    return max(1, min(at_30, steps * at_30 // 30))


def attention_refresh(index: int, steps: int, every: int) -> bool:
    """Whether step ``index`` computes attention afresh under reuse every ``every`` steps.

    The first steps (4 of 30) and the last (2 of 30) always do; in between, every ``every``-th one.
    """

    if every <= 1:
        return True
    warmup, tail = scaled_gate(steps, 4), scaled_gate(steps, 2)
    return index < warmup or index + tail >= steps or (index - warmup) % every == 0


def denoise(dit, text, text_tags, width: int, height: int, frames: int, points: int, seed: int = 0,
            subset: tuple[int, ...] | None = None, on_step=None, forward=None, release: bool = False,
            condition: mx.array | None = None, keyframes: tuple[str, ...] = (),
            audio_shift: float | None = None, step_cache: float = 0.0, attention_every: int = 0,
            nodes: tuple[float, ...] | None = None, video_shift: float | None = None, prepare=None) -> Latents:
    """Denoise one clip. ``points`` is the number of sigma points, so ``points - 1`` forwards (fewer with
    ``subset``). ``forward`` replaces the plain DiT call. The AdaLN tables for the whole run are projected once
    up front; ``release`` then frees the projection weights. ``condition`` holds the encoded keyframe rows for
    ``keyframes`` (``first`` or ``last`` each); they condition every step and are not denoised. ``audio_shift``
    replaces the audio schedule's sigma shift (3 in the released model).

    Two savings for many-step runs, after mlx-serve's fast recipe (ddalcu), which combines a TeaCache-style
    velocity cache with PAB-style attention reuse. ``step_cache`` (0.05 there) skips a forward and reuses the last
    velocity while the summed relative move ``|d sigma| mean|v| / mean|x|`` since the last forward stays under it;
    never on the first steps or the last, and at most twice running. ``attention_every`` (2 there) computes
    attention only on every that-many-th step between the opening and closing steps and reuses each block's
    attention output otherwise. Both change the result; neither suits a few-step adapter, whose steps are large.

    ``nodes`` are a distilled model's own unshifted sigma rungs (``points`` is then ignored) and ``video_shift``
    its video shift. ``prepare(packed, latent_frames, latent_height, latent_width)`` is called once the sequence
    layout is known, before the first forward."""

    config = dit.config
    latent_frames = h3.latent_frames(frames)
    latent_height, latent_width = height // h3.VAE_SPATIAL_RATIO, width // h3.VAE_SPATIAL_RATIO
    audio_latents = h3.audio_latents(frames)
    packed = layout(text_tags, latent_frames, latent_height, latent_width, audio_latents, config.patch_size,
                    keyframes)
    held = packed.condition_video_rows
    if (condition is None) != (held == 0) or (condition is not None and condition.shape[0] != held):
        raise ValueError(f"{len(keyframes)} keyframes need {held} condition rows")
    video_rows, audio_rows = start_noise(config, latent_frames, latent_height, latent_width, audio_latents, seed,
                                         condition)
    check_noise("video", video_rows[held:])
    check_noise("audio", audio_rows)

    if prepare is not None:
        prepare(packed, latent_frames, latent_height, latent_width)
    video_schedule = Schedule(VIDEO_SHIFT if video_shift is None else video_shift, points, subset, nodes)
    audio_schedule = Schedule(AUDIO_SHIFT if audio_shift is None else audio_shift, points, subset, nodes)
    table, plan = timestep_plan(packed, video_schedule.timesteps, audio_schedule.timesteps)
    text = text.astype(mx.bfloat16)
    dit.cache_modulation(table, release=release)
    run = forward or dit
    seconds = []
    steps = len(video_schedule)
    kept = [None] * len(dit.blocks) if attention_every > 1 else None
    last_video = last_audio = None
    moved, running, reused, skipped = 0.0, 0, 0, 0
    for index in range(steps):
        started = time.perf_counter()
        skip = False
        if step_cache > 0 and last_video is not None and index >= scaled_gate(steps, 2) and index + 1 < steps \
                and running < 2:
            jump = abs(float(video_schedule.sigmas[index + 1] - video_schedule.sigmas[index]))
            size = max(float(mx.mean(mx.abs(video_rows[held:])).item()), 1e-8)
            move = jump * float(mx.mean(mx.abs(last_video)).item()) / size
            if moved + move < step_cache:
                moved += move
                skip = True
        if skip:
            video_velocity, audio_velocity = last_video, last_audio
            running += 1
            skipped += 1
        else:
            refresh = attention_refresh(index, steps, attention_every)
            extra = {} if kept is None else {"kept": kept, "refresh": refresh}
            out_video, out_audio = run(video_rows[None], audio_rows[None], text, table, plan[index], packed.tags,
                                       packed.position_ids, packed.video_rows, packed.audio_rows, packed.text_rows,
                                       **extra)
            video_velocity, audio_velocity = out_video[0, held:], out_audio[0]
            if step_cache > 0:
                last_video, last_audio = video_velocity, audio_velocity
            moved, running = 0.0, 0
            reused += int(kept is not None and not refresh)
        stepped = video_schedule.step(index, video_velocity, video_rows[held:])
        video_rows = mx.concatenate([video_rows[:held], stepped]) if held else stepped
        audio_rows = audio_schedule.step(index, audio_velocity, audio_rows)
        mx.eval(video_rows, audio_rows)
        seconds.append(time.perf_counter() - started)
        if on_step is not None:
            on_step(index + 1, steps, seconds[-1])
    return Latents(video_rows[held:], audio_rows, packed, latent_frames, latent_height, latent_width, audio_latents,
                   seconds, reused, skipped)


def _block_inputs(block, x, table, adaln):
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = table
    return block.norm1(x) * (1.0 + scale_a[adaln]) + shift_a[adaln], gate_a[adaln], scale_m[adaln], shift_m[adaln], \
        gate_m[adaln]


def _block_finish(block, x, mixed, gate_a, scale_m, shift_m, gate_m):
    x = x + (gate_a * block.attn.out_proj(mixed.astype(x.dtype)).astype(x.dtype)).astype(x.dtype)
    return x + (gate_m * block.mlp(block.norm2(x) * (1.0 + scale_m) + shift_m)).astype(x.dtype)


@dataclass
class HeldContext:
    """Every block's keys and values over the rows that are not audio, and what an audio-only pass needs."""

    kv: list[tuple[mx.array, mx.array]]
    rotary: tuple[mx.array, mx.array]
    tables: list
    final: mx.array
    first: int
    last: int


def held_context(dit, video_rows, audio_rows, text, table, plan_rows, packed: Layout) -> HeldContext:
    """One whole-sequence pass that keeps each block's keys and values over the non-audio rows."""

    audio_index = np.asarray(packed.audio_rows.tolist(), dtype=np.int64)
    first, last = int(audio_index[0]), int(audio_index[-1]) + 1
    if not np.array_equal(audio_index, np.arange(first, last)):
        raise ValueError("audio rows are expected to be one run of the sequence")
    x, adaln, rotary = dit.pack(video_rows[None], audio_rows[None], text, table, plan_rows, packed.tags,
                                packed.position_ids, packed.video_rows, packed.audio_rows, packed.text_rows)
    tables, final = dit.modulation(table)
    kv = []
    for block, block_table in zip(dit.blocks, tables, strict=True):
        h, *rest = _block_inputs(block, x, block_table, adaln)
        q, k, v = block.attn.qkv(h, rotary)
        kept = (mx.concatenate([k[:, :, :first], k[:, :, last:]], axis=2),
                mx.concatenate([v[:, :, :first], v[:, :, last:]], axis=2))
        x = _block_finish(block, x, block.attn.mix(q, k, v), *rest)
        mx.eval(x, *kept)
        kv.append(kept)
    return HeldContext(kv, (rotary[0][first:last], rotary[1][first:last]), tables, final, first, last)


def audio_velocity(dit, audio_rows, plan_rows, context: HeldContext) -> mx.array:
    """The audio velocity from the audio rows alone, attending to the kept keys and values and to themselves."""

    rows = plan_rows[context.first:context.last]
    adaln = rows * MODALITIES + TAG_AUDIO
    x = dit.audio_patch_proj(audio_rows.astype(mx.float32)).astype(mx.bfloat16)[None]
    for block, block_table, (keys, values) in zip(dit.blocks, context.tables, context.kv, strict=True):
        h, *rest = _block_inputs(block, x, block_table, adaln)
        q, k, v = block.attn.qkv(h, context.rotary)
        mixed = block.attn.mix(q, mx.concatenate([keys, k], axis=2), mx.concatenate([values, v], axis=2))
        x = _block_finish(block, x, mixed, *rest)
    x = dit.final_layer.norm_out(x, context.final, rows).astype(mx.float32)
    return dit.final_layer.audio_out(x)[0]


def revoice(dit, text, latents: Latents, points: int, seed: int = 0, condition: mx.array | None = None,
            exact: bool = False, on_step=None, release: bool = False) -> mx.array:
    """Denoise the audio rows again, from noise, against the finished video of ``latents``; returns audio rows.

    The video, keyframe and text rows are held at the keyframe timestep for every step and only the audio rows
    move down the audio schedule, so a model without a few-step adapter can voice a clip whose picture a few-step
    run made. ``exact`` runs the whole sequence each step. Otherwise the held rows go through the stack once,
    beside the audio that came with the clip, and each block's keys and values over them are kept; a step then
    runs the audio rows alone against those. That is the same computation except that the held rows do not see
    the audio changing between steps.
    """

    packed = latents.packed
    held = packed.condition_video_rows
    if (condition is None) != (held == 0):
        raise ValueError("pass the keyframe rows the clip was made with")
    level = float(np.float32(KEYFRAME_NOISE))
    rest = float(np.float32(1.0) - np.float32(KEYFRAME_NOISE))
    mx.random.seed(seed)

    def settle(rows):
        return level * rows.astype(mx.float32) + rest * mx.random.normal(rows.shape).astype(mx.float32)

    video_rows = settle(latents.video_rows)
    if held:
        video_rows = mx.concatenate([settle(condition), video_rows])
    reference = settle(latents.audio_rows)
    audio_rows = mx.random.normal(latents.audio_rows.shape).astype(mx.float32)
    check_noise("audio", audio_rows)

    schedule = Schedule(AUDIO_SHIFT, points)
    audio_index = np.asarray(packed.audio_rows.tolist(), dtype=np.int64)
    steps = []
    for audio_t in schedule.timesteps:
        per_row = np.full(packed.rows, np.float32(KEYFRAME_NOISE), dtype=np.float32)
        per_row[audio_index] = np.float32(audio_t)
        steps.append(per_row)
    values = np.unique(np.concatenate([*steps, np.array([KEYFRAME_NOISE], dtype=np.float32)]))
    plan = [mx.array(np.searchsorted(values, per_row).astype(np.int32)) for per_row in steps]
    held_plan = mx.array(np.full(packed.rows, np.searchsorted(values, np.float32(KEYFRAME_NOISE)), dtype=np.int32))
    table = mx.array(values)
    text = text.astype(mx.bfloat16)
    dit.cache_modulation(table, release=release)
    context = None if exact else held_context(dit, video_rows, reference, text, table, held_plan, packed)

    for index in range(len(schedule)):
        started = time.perf_counter()
        if exact:
            _, velocity = dit(video_rows[None], audio_rows[None], text, table, plan[index], packed.tags,
                              packed.position_ids, packed.video_rows, packed.audio_rows, packed.text_rows)
            velocity = velocity[0]
        else:
            velocity = audio_velocity(dit, audio_rows, plan[index], context)
        audio_rows = schedule.step(index, velocity, audio_rows)
        mx.eval(audio_rows)
        if on_step is not None:
            on_step(index + 1, len(schedule), time.perf_counter() - started)
    return audio_rows
