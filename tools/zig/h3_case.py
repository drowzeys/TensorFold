#!/usr/bin/env python3
"""One reference case for the native FastH3 transformer (zig/src/families/h3), and the decode of what it returns.

Writes case.safetensors from the Python 0.6 family, float projections and FastVideo's reference routing:
  text                 the prompt's rows after the condition projection and the token refiner (they never change)
  video, audio         the starting noise rows
  condition            with --first-frame, the image's rows at the keyframe noise level; they are never stepped
  cos, sin             rotary tables for every packed row, in the order [text | keyframe | audio | video]
  adaln, times         per step, each row's line in a block's modulation tables and in the final layer's
  tables.N, final      every block's six modulation tables for the run's timesteps, and the final layer's
  video_step, audio_step  per step (sigma seen, Euler ratio)
  tile_slot, tile_sizes, geometry   FastVideo's tile map: each row's padded slot, each tile's rows, the counts
  the small float projections (patches in, heads out) and the first forward's two velocities
The block projections are read by the native tool from the checkpoint itself.
`--decode PREFIX` instead decodes PREFIX.video.f32 and PREFIX.audio.f32 (raw rows) into --mp4.
`--revoice N` with it first makes the audio rows again with the released model (no distillation) in N steps against
the finished picture, as families/h3/sampler.py `revoice` does; it needs the prompt (--prompt-file) and --seed.
`--first-frame IMAGE` starts the clip from an image: its vision tokens join the prompt's rows and its encoded rows sit
between the text and the audio, as tools/h3_generate_dev.py --first-frame does (the image is stretched onto the canvas).
`--steps N` exports N forwards instead of the checkpoint's own count, rungs spread evenly from its first rung.
`--lora PATH[:STRENGTH]` (repeatable) applies style adapters as families/h3/lora.py `merge` does: the block
projections' low-rank factors go to lora.safetensors for the native tool (`lora=PATH`), which adds them to the bf16
weights in float32 before its int8 rounding; updates on the small modules this export computes (the condition
projection, token refiner, time embedder, modulation and final layers) are merged here. An adapter that changes
anything else the native tool reads from the checkpoint (a norm, a dense projection correction) is refused with
exit status 3. Without --no-reference the reference forward runs with the whole adapter merged in float32.
Both outputs are written under another name and renamed when complete, so the native tool can start with `wait`
before they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="the MiniMax H3 pipeline folder (text encoder and decoders)")
    parser.add_argument("fasth3", help="the FastH3 checkpoint folder")
    parser.add_argument("out_dir")
    parser.add_argument("--tools", required=True, help="folder holding h3_generate_dev.py")
    parser.add_argument("--prompt-file")
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode", help="prefix of the raw float32 rows to decode into --mp4")
    parser.add_argument("--mp4")
    parser.add_argument("--crop", help="with --decode: WxH, centre-crop the decoded frames before the MP4 is written")
    parser.add_argument("--steps", type=int, default=0, help="forwards, when not the checkpoint's trained count")
    parser.add_argument("--revoice", type=int, default=0, metavar="STEPS",
                        help="with --decode: audio made again by the released model in this many steps")
    parser.add_argument("--first-frame", help="image the clip starts from (image to video)")
    parser.add_argument("--lora", action="append", default=[], metavar="PATH[:STRENGTH]",
                        help="a style adapter (families/h3/lora.py layouts); its block factors go to lora.safetensors")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip the reference forward (its velocities are written as zeros): for renders, not parity")
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, args.tools)
    import h3_generate_dev as tool

    from tensorfold.families.h3 import config as h3
    from tensorfold.families.h3 import fasth3
    from tensorfold.families.h3.dit import MODALITIES, rotary_tables, timestep_embedding
    from tensorfold.families.h3.packing import layout, timestep_plan
    from tensorfold.families.h3.sampler import start_noise
    from tensorfold.families.h3.schedule import Schedule

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = h3.pipeline_root(args.model_dir)
    frames_l = h3.latent_frames(args.frames)
    lat_h, lat_w = args.height // h3.VAE_SPATIAL_RATIO, args.width // h3.VAE_SPATIAL_RATIO
    audio_l = h3.audio_latents(args.frames)
    config = fasth3.config(args.fasth3)
    # first, so the native tool can merge the adapters while the prompt is encoded
    small = export_lora(args.lora, config, out) if args.lora and not args.decode else ([], [])

    if args.decode:
        video = np.fromfile(args.decode + ".video.f32", dtype=np.float32).reshape(-1, config.latents_dim * 4)
        audio = np.fromfile(args.decode + ".audio.f32", dtype=np.float32).reshape(-1, config.audio_latents_dim)
        latents = SimpleNamespace(video_rows=mx.array(video), audio_rows=mx.array(audio), latent_frames=frames_l,
                                  latent_height=lat_h, latent_width=lat_w, audio_latents=audio_l)
        if args.revoice:
            import gc

            from tensorfold.families.h3.sampler import revoice

            # the native run keeps no layout: rebuild it from the prompt, as the export did
            text, tags = tool.encode_text(root, Path(args.prompt_file).read_text())
            latents.packed = layout(tags, frames_l, lat_h, lat_w, audio_l, config.patch_size)
            mark = time.perf_counter()
            base = tool.load_dit(args.model_dir)
            tool.int8_mlp(base)
            tool.int8_attention(base, qkv=True, out=True, fused=True)
            latents.audio_rows = revoice(base, text, latents, args.revoice + 1, args.seed, release=True,
                                         on_step=lambda i, n, t: print(f"[tensorfold] voice {i}/{n} {t:.2f}s", flush=True))
            mx.eval(latents.audio_rows)
            print(f"[tensorfold] re-voiced in {time.perf_counter() - mark:.1f}s", flush=True)
            del base
            gc.collect()
            mx.clear_cache()
        frames, wave, rate = tool.decode(args.model_dir, root, latents, h3.DiTConfig.from_checkpoint(args.model_dir))
        from minimax_h3_mlx.media import save_mp4

        if args.crop:
            crop_w, crop_h = (int(v) for v in args.crop.lower().split("x"))
            top, left = (frames.shape[1] - crop_h) // 2, (frames.shape[2] - crop_w) // 2
            frames = np.ascontiguousarray(frames[:, top : top + crop_h, left : left + crop_w])
        save_mp4(args.mp4, frames, h3.FPS, audio=wave, sample_rate=rate)
        print(json.dumps({"mp4": args.mp4, "frames": int(frames.shape[0])}))
        return 0

    image, condition, keyframes = None, None, ()
    if args.first_frame:
        from minimax_h3_mlx.packing import prepare_keyframe_image
        from PIL import Image

        image = prepare_keyframe_image(Image.open(args.first_frame).convert("RGB"), args.height, args.width, stretch=True)
    text, tags = tool.encode_text(root, Path(args.prompt_file).read_text(), image)
    if image is not None:
        condition, keyframes = tool.encode_first_frame(root, image, args.width, args.height, config.patch_size), ("first",)
    dit, gates, fast = fasth3.load_fasth3(args.fasth3)
    if args.lora and not args.no_reference:
        from tensorfold.families.h3.lora import merge

        # the reference forward sees every update: the whole adapter in float32, as the MLX engine merges it
        for spec in args.lora:
            path, _, strength = spec.partition(":")
            merge(dit, path, float(strength or 1.0))
    else:
        merge_small(dit, *small)
    packed = layout(tags, frames_l, lat_h, lat_w, audio_l, config.patch_size, keyframes)
    held = packed.condition_video_rows
    # the keyframe's rows come first among the video rows, already at their noise level
    video, audio = start_noise(config, frames_l, lat_h, lat_w, audio_l, args.seed, condition)
    prepare = fasth3.route(dit, gates, fast.sparsity, fast.tile, "reference")
    geometry = prepare(packed, frames_l, lat_h, lat_w)
    nodes = fast.nodes
    if args.steps and args.steps != len(nodes):
        nodes = tuple(nodes[0] * (1 - i / args.steps) for i in range(args.steps))
    schedules = [Schedule(fast.video_shift, 0, None, nodes), Schedule(fast.audio_shift, 0, None, nodes)]
    table, plan = timestep_plan(packed, schedules[0].timesteps, schedules[1].timesteps)
    text = text.astype(mx.bfloat16)
    rows = dit.token_refiner(dit.condition_proj(text.astype(dit.condition_proj.weight.dtype)).astype(mx.bfloat16))
    temb = dit.time_embedder(timestep_embedding(table, config.timestep_input_dim))
    cos, sin = rotary_tables(config, packed.position_ids)
    lines = np.maximum(np.asarray(packed.tags), 0)
    case = {
        "text": rows[0].astype(mx.bfloat16), "video": video[held:].astype(mx.float32), "audio": audio.astype(mx.float32),
        "cos": cos.astype(mx.float32), "sin": sin.astype(mx.float32),
        "adaln": mx.array(np.stack([np.asarray(p) * MODALITIES + lines for p in plan]).astype(np.int32)),
        "times": mx.array(np.stack([np.asarray(p) for p in plan]).astype(np.int32)),
        "final": dit.final_layer.adaln_proj(temb).astype(mx.bfloat16),
        "tile_slot": mx.array(geometry.untile_combined_index.astype(np.int32)),
        "tile_sizes": mx.array(geometry.variable_block_sizes.astype(np.int32)),
        "geometry": mx.array(np.asarray([geometry.num_prefix_tiles, geometry.num_video_tiles,
                                         fasth3_keep(fast.sparsity, geometry.num_video_tiles),
                                         int(packed.text_rows.shape[0]), int(packed.audio_rows.shape[0]),
                                         int(packed.video_rows.shape[0]) - held, held][: 7 if held else 6],
                                        dtype=np.int32)),
        "video_in.weight": dit.video_patch_proj.weight.astype(mx.float32),
        "video_in.bias": dit.video_patch_proj.bias.astype(mx.float32),
        "audio_in.weight": dit.audio_patch_proj.weight.astype(mx.float32),
        "audio_in.bias": dit.audio_patch_proj.bias.astype(mx.float32),
        "video_out.weight": dit.final_layer.video_out.weight.astype(mx.float32),
        "video_out.bias": dit.final_layer.video_out.bias.astype(mx.float32),
        "audio_out.weight": dit.final_layer.audio_out.weight.astype(mx.float32),
        "audio_out.bias": dit.final_layer.audio_out.bias.astype(mx.float32),
        "final_norm.weight": dit.final_layer.norm.weight.astype(mx.bfloat16),
    }
    if held:
        case["condition"] = video[:held].astype(mx.float32)
    for name, schedule in zip(("video_step", "audio_step"), schedules, strict=True):
        steps = [(float(np.float32(1.0) - schedule.timesteps[i]), float(schedule.sigmas[i + 1] / schedule.sigmas[i]))
                 for i in range(len(schedule))]
        case[name] = mx.array(np.asarray(steps, dtype=np.float32))
    for index, block in enumerate(dit.blocks):
        case[f"tables.{index}"] = mx.stack(block.adaln_proj.tables(temb)).astype(mx.bfloat16)   # (6, lines, hidden)
    mx.eval(*case.values())
    mark = time.perf_counter()
    if args.no_reference:
        case["video_velocity"], case["audio_velocity"] = mx.zeros_like(video[held:]), mx.zeros_like(audio)
    else:
        first = dit(video[None], audio[None], text, table, plan[0], packed.tags, packed.position_ids,
                    packed.video_rows, packed.audio_rows, packed.text_rows)
        mx.eval(*first)
        case["video_velocity"] = first[0][0, held:].astype(mx.float32)
        case["audio_velocity"] = first[1][0].astype(mx.float32)
    mx.save_safetensors(str(out / "case.partial.safetensors"), case)
    os.replace(out / "case.partial.safetensors", out / "case.safetensors")
    print(json.dumps({"rows": packed.rows, "text": int(packed.text_rows.shape[0]), "audio": int(packed.audio_rows.shape[0]),
                      "video": int(packed.video_rows.shape[0]) - held, "keyframe": held, "timesteps": int(table.shape[0]),
                      "tiles": geometry.num_tiles, "prefix_tiles": geometry.num_prefix_tiles,
                      "hidden": config.hidden_size, "layers": config.num_layers,
                      "reference_forward_s": round(time.perf_counter() - mark, 1)}))
    return 0


# the block projections whose factors the native tool merges: the H3 module name, and its own names for the parts
NATIVE = re.compile(r"blocks\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)$")
# what this export computes itself, so an adapter's update there is merged into the MLX modules here
SMALL = ("condition_proj.", "token_refiner.", "time_embedder.", "video_patch_proj.", "audio_patch_proj.",
         "final_layer.")
REFUSED = 3


def export_lora(specs: list[str], config, out: Path) -> tuple[list, list]:
    """Write the adapters' block factors to out/lora.safetensors; return the small modules' pairs and corrections.

    Factors are float32 `{block}.{part}.a` (R, K) and `.b` (N, R), the strength folded into b and several adapters
    (or several pairs of one) concatenated along R, for parts to_q, to_k, to_v (the H3 qkv rows are per head
    [q, k, v]), to_out, fc_gate and fc_value (H3 fc1 is [gate; value]) and fc2: the checkpoint's own projections.
    """

    import mlx.core as mx

    from tensorfold.families.h3 import lora

    heads, dim, width = config.num_attention_heads, config.attention_head_dim, config.ffn_hidden_size
    inner = heads * dim
    parts: dict[str, list] = {}
    pairs, diffs = [], []

    def add(block: str, part: str, a, b) -> None:
        if not bool(mx.any(b != 0)):
            return  # FastVideo's format fills the slots of q, k and v it does not train with zeros
        parts.setdefault(f"{block}.{part}", []).append((a, b))

    for spec in specs:
        path, _, strength = spec.partition(":")
        strength = float(strength or 1.0)
        adapter = lora.load(path, config)
        for module, found in adapter.pairs.items():
            native = NATIVE.match(module)
            for a, b in found:
                if native is None:
                    if not module.startswith(SMALL) and not re.match(r"blocks\.\d+\.adaln_proj\.", module):
                        refuse(path, module)
                    pairs.append((module, a, b, strength))
                    continue
                block, kind = native.group(1), native.group(2)
                b = b * strength
                if kind == "attn.qkv_proj":
                    rows = b.reshape(heads, 3, dim, b.shape[-1])
                    for slot, part in enumerate(("to_q", "to_k", "to_v")):
                        add(block, part, a, rows[:, slot].reshape(inner, b.shape[-1]))
                elif kind == "mlp.fc1":
                    add(block, "fc_gate", a, b[:width])
                    add(block, "fc_value", a, b[width:])
                else:
                    add(block, "to_out" if kind == "attn.out_proj" else "fc2", a, b)
        for parameter, diff in adapter.diffs.items():
            if not parameter.startswith(SMALL) and not re.match(r"blocks\.\d+\.adaln_proj\.", parameter):
                refuse(path, parameter)
            diffs.append((parameter, diff, strength))
    factors = {}
    for name, found in parts.items():
        factors[name + ".a"] = mx.concatenate([a for a, _ in found], axis=0).astype(mx.float32)
        factors[name + ".b"] = mx.concatenate([b for _, b in found], axis=1).astype(mx.float32)
    mx.eval(*factors.values())
    save_aligned(out / "lora.partial.safetensors", factors)
    os.replace(out / "lora.partial.safetensors", out / "lora.safetensors")
    print(f"[h3_case] adapters: {len(parts)} block projections to the native tool, {len(pairs)} small updates and "
          f"{len(diffs)} corrections merged here", file=sys.stderr, flush=True)
    return pairs, diffs


def save_aligned(path: Path, tensors: dict) -> None:
    """float32 tensors to a safetensors file whose payload starts on 8 bytes, so the GPU reads every one in place."""

    import numpy as np

    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()}
    header, offset = {}, 0
    for name, array in arrays.items():
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as handle:
        handle.write(len(text).to_bytes(8, "little"))
        handle.write(text)
        for array in arrays.values():
            handle.write(np.ascontiguousarray(array).tobytes())


def refuse(path: str, name: str) -> None:
    print(f"{Path(path).name} changes {name}, which the native tool reads from the checkpoint", file=sys.stderr)
    raise SystemExit(REFUSED)


def merge_small(dit, pairs: list, diffs: list) -> None:
    """families/h3/lora.py `merge` for the small modules: float32 sums, vectors back to their own type."""

    import mlx.core as mx

    from tensorfold.families.h3.lora import _resolve

    for module, a, b, strength in pairs:
        owner, leaf = _resolve(dit, module)
        linear = getattr(owner, leaf)
        linear.weight = linear.weight.astype(mx.float32) + strength * (b @ a)
        mx.eval(linear.weight)
    for parameter, diff, strength in diffs:
        owner, leaf = _resolve(dit, parameter)
        current = getattr(owner, leaf)
        updated = current.astype(mx.float32) + strength * diff
        setattr(owner, leaf, updated.astype(current.dtype) if updated.ndim == 1 and leaf == "weight" else updated)
        mx.eval(getattr(owner, leaf))


def fasth3_keep(sparsity: float, video_tiles: int) -> int:
    from tensorfold.families.h3.vendor import fastvideo_vsa as vsa

    return vsa.compute_topk(sparsity, video_tiles)


if __name__ == "__main__":
    raise SystemExit(main())
