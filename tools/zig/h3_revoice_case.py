#!/usr/bin/env python3
"""A re-voice case for the native H3 transformer: the released MiniMax-H3 voices a finished picture again.

Writes case.safetensors as h3_case.py does, for the released model and families/h3/sampler.py `revoice` (its
whole-sequence form): every row but the generated audio is held at the keyframe level, the video rows settled from the
finished clip as the sampler settles them, the audio rows fresh noise. `--at K` lets the reference take steps 0..K-1
first and writes the audio rows at step K with the reference velocity there (the native tool then runs from=K).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="the MiniMax H3 pipeline folder (its FL2VA partition is used)")
    parser.add_argument("finished", help="prefix of the finished clip's raw rows (PREFIX.video.f32)")
    parser.add_argument("out_dir")
    parser.add_argument("--tools", required=True, help="folder holding h3_generate_dev.py")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30, help="audio steps")
    parser.add_argument("--at", type=int, default=0)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, args.tools)
    import h3_generate_dev as tool

    from tensorfold.families.h3 import config as h3
    from tensorfold.families.h3.dit import MODALITIES, rotary_tables, timestep_embedding
    from tensorfold.families.h3.packing import KEYFRAME_NOISE, layout
    from tensorfold.families.h3.schedule import AUDIO_SHIFT, Schedule
    from tensorfold.families.h3.vendor import fastvideo_vsa as vsa
    from tensorfold.families.h3.weights import load_dit

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = h3.pipeline_root(args.model_dir)
    frames_l = h3.latent_frames(args.frames)
    lat_h, lat_w = args.height // h3.VAE_SPATIAL_RATIO, args.width // h3.VAE_SPATIAL_RATIO
    audio_l = h3.audio_latents(args.frames)
    dit = load_dit(args.model_dir)
    config = dit.config
    text, tags = tool.encode_text(root, Path(args.prompt_file).read_text())
    packed = layout(tags, frames_l, lat_h, lat_w, audio_l, config.patch_size)
    finished = np.fromfile(args.finished + ".video.f32", dtype=np.float32).reshape(-1, config.latents_dim * 4)

    # revoice()'s draws in its order: the settled video, the clip's audio settled (unused here), fresh audio
    level = float(np.float32(KEYFRAME_NOISE))
    rest = float(np.float32(1.0) - np.float32(KEYFRAME_NOISE))
    mx.random.seed(args.seed)
    video = level * mx.array(finished) + rest * mx.random.normal(finished.shape).astype(mx.float32)
    audio_shape = (int(packed.audio_rows.shape[0]), config.audio_latents_dim)
    mx.random.normal(audio_shape)
    audio = mx.random.normal(audio_shape).astype(mx.float32)

    schedule = Schedule(AUDIO_SHIFT, args.steps + 1)
    audio_index = np.asarray(packed.audio_rows.tolist(), dtype=np.int64)
    pinned = np.full(packed.rows, np.float32(KEYFRAME_NOISE), dtype=np.float32)
    steps = []
    for audio_t in schedule.timesteps:
        per_row = pinned.copy()
        per_row[audio_index] = np.float32(audio_t)
        steps.append(per_row)
    values = np.unique(np.concatenate([*steps, pinned]))
    plan = [mx.array(np.searchsorted(values, per_row).astype(np.int32)) for per_row in steps]
    table = mx.array(values)

    # the native engine attends in 64-row tiles even where every tile sees every tile
    shape = (frames_l // config.patch_size[0], lat_h // config.patch_size[1], lat_w // config.patch_size[2])
    segments = tuple(n for n in (int(packed.text_rows.shape[0]), int(packed.audio_rows.shape[0])) if n > 0)
    geometry = vsa.build_h3_tile_geometry(segments, shape, 64)
    text = text.astype(mx.bfloat16)
    rows = dit.token_refiner(dit.condition_proj(text.astype(dit.condition_proj.weight.dtype)).astype(mx.bfloat16))
    temb = dit.time_embedder(timestep_embedding(table, config.timestep_input_dim))
    cos, sin = rotary_tables(config, packed.position_ids)
    lines = np.maximum(np.asarray(packed.tags), 0)
    counts = [geometry.num_prefix_tiles, geometry.num_video_tiles, geometry.num_video_tiles,
              int(packed.text_rows.shape[0]), int(packed.audio_rows.shape[0]), int(packed.video_rows.shape[0])]
    case = {
        "text": rows[0].astype(mx.bfloat16), "video": video, "audio": audio,
        "cos": cos.astype(mx.float32), "sin": sin.astype(mx.float32),
        "adaln": mx.array(np.stack([np.asarray(p) * MODALITIES + lines for p in plan]).astype(np.int32)),
        "times": mx.array(np.stack([np.asarray(p) for p in plan]).astype(np.int32)),
        "final": dit.final_layer.adaln_proj(temb).astype(mx.bfloat16),
        "tile_slot": mx.array(geometry.untile_combined_index.astype(np.int32)),
        "tile_sizes": mx.array(geometry.variable_block_sizes.astype(np.int32)),
        "geometry": mx.array(np.asarray(counts, dtype=np.int32)),
        "video_in.weight": dit.video_patch_proj.weight.astype(mx.float32),
        "video_in.bias": dit.video_patch_proj.bias.astype(mx.float32),
        "audio_in.weight": dit.audio_patch_proj.weight.astype(mx.float32),
        "audio_in.bias": dit.audio_patch_proj.bias.astype(mx.float32),
        "video_out.weight": dit.final_layer.video_out.weight.astype(mx.float32),
        "video_out.bias": dit.final_layer.video_out.bias.astype(mx.float32),
        "audio_out.weight": dit.final_layer.audio_out.weight.astype(mx.float32),
        "audio_out.bias": dit.final_layer.audio_out.bias.astype(mx.float32),
        "final_norm.weight": dit.final_layer.norm.weight.astype(mx.bfloat16),
        "video_step": mx.array(np.asarray([(0.0, 1.0)] * len(schedule), dtype=np.float32)),
        "audio_step": mx.array(np.asarray([(float(np.float32(1.0) - schedule.timesteps[i]),
                                            float(schedule.sigmas[i + 1] / schedule.sigmas[i]))
                                           for i in range(len(schedule))], dtype=np.float32)),
    }
    for index, block in enumerate(dit.blocks):
        case[f"tables.{index}"] = mx.stack(block.adaln_proj.tables(temb)).astype(mx.bfloat16)
    mx.eval(*case.values())
    mark = time.perf_counter()
    for i in range(args.at + 1):
        started = time.perf_counter()
        first = dit(video[None], audio[None], text, table, plan[i], packed.tags, packed.position_ids,
                    packed.video_rows, packed.audio_rows, packed.text_rows)
        mx.eval(*first)
        print(f"forward {i}: {time.perf_counter() - started:.2f} s", flush=True)
        if i < args.at:
            audio = schedule.step(i, first[1][0], audio)
            mx.eval(audio)
    case["audio"] = audio.astype(mx.float32)
    case["video_velocity"] = first[0][0].astype(mx.float32)
    case["audio_velocity"] = first[1][0].astype(mx.float32)
    mx.save_safetensors(str(out / "case.safetensors"), case)
    print(json.dumps({"rows": packed.rows, "audio": audio_shape[0], "timesteps": int(table.shape[0]),
                      "tiles": geometry.num_tiles, "prefix_tiles": geometry.num_prefix_tiles, "at": args.at,
                      "reference_s": round(time.perf_counter() - mark, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
