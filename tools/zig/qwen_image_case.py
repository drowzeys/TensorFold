#!/usr/bin/env python3
"""Weights and one reference case for the native Qwen-Image-2.1 transformer (zig/src/families/qwen_image).

Writes, from the Python 0.6 family in float (no int8 kernels):
  weights.safetensors  every projection transposed to (inputs, outputs) bfloat16, the layout the gemm kernel reads
  case.safetensors     per-block text keys (heads, 128, tokens) and values (heads, tokens, 128), the image rows'
                       rotary tables, the starting latents, the noise levels, the first velocity, the final latents
  reference.png        the image the Python family decodes from its final latents
`--decode FILE` instead decodes raw float32 latents (rows * columns, 64) written by tf-qwen-image-dit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

NODES = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)


def save_png(model_dir, latents, path):
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    from tensorfold.families.qwen_image import vae

    decoder = vae.load_decoder(model_dir, mx.float32)
    image = decoder.decode(latents)
    mx.eval(image)
    Image.fromarray((np.asarray(image[0]) * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--tools", required=True, help="folder holding qwen_image_generate_dev.py (its prompt encoder)")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--lora")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode", help="raw float32 latents to decode into --png")
    parser.add_argument("--png")
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    from tensorfold.families.qwen_image import lora, sampler, weights
    from tensorfold.families.qwen_image.config import latent_grid
    from tensorfold.families.qwen_image.schedule import sigmas

    rows, columns = latent_grid(args.width, args.height)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.decode:
        raw = np.fromfile(args.decode, dtype=np.float32).reshape(1, rows, columns, 64).transpose(0, 3, 1, 2)
        save_png(args.model_dir, mx.array(raw), args.png)
        return 0

    sys.path.insert(0, args.tools)
    from qwen_image_generate_dev import encode_prompt

    text = encode_prompt(args.model_dir, Path(args.prompt_file).read_text().strip())
    dit = weights.load_dit(args.model_dir)
    if args.lora:
        print(json.dumps(lora.merge(dit, args.lora)))
        lora.settle(dit)

    from mlx.utils import tree_flatten

    tensors = {}
    for name, value in tree_flatten(dit.parameters()):
        tensors[name] = (value.T if value.ndim == 2 else value).astype(mx.bfloat16)
    mx.save_safetensors(str(out / "weights.safetensors"), tensors)
    del tensors

    levels = sigmas(0, args.width, args.height, NODES)
    prefix = dit.prefix(text, rows, columns)
    start = sampler.start_noise(args.seed, args.width, args.height)
    first = dit(start.astype(mx.float32), float(levels[0]), prefix)
    mx.eval(first)
    case = {"sigmas": mx.array(levels), "latents": start[0].astype(mx.float32),
            "cos": prefix.rotary.cos.astype(mx.float32), "sin": prefix.rotary.sin.astype(mx.float32),
            "velocity": first[0].astype(mx.float32)}
    for index, (k, v) in enumerate(prefix.kv):
        case[f"text_k.{index}"] = k[0].transpose(0, 2, 1).astype(mx.bfloat16)   # (heads, 128, tokens)
        case[f"text_v.{index}"] = v[0].astype(mx.bfloat16)                      # (heads, tokens, 128)
    times = []
    x = start.astype(mx.float32)
    for index in range(len(levels) - 1):
        mark = time.perf_counter()
        velocity = dit(x, float(levels[index]), prefix)
        x = x + velocity.astype(mx.float32) * float(levels[index + 1] - levels[index])
        mx.eval(x)
        times.append(time.perf_counter() - mark)
    case["final"] = x[0]
    mx.save_safetensors(str(out / "case.safetensors"), case)
    final = x.reshape(1, rows, columns, -1).transpose(0, 3, 1, 2)
    del dit
    mx.clear_cache()
    save_png(args.model_dir, final, str(out / "reference.png"))
    print(json.dumps({"rows": rows * columns, "text_tokens": int(text.shape[1]), "sigmas": [float(v) for v in levels],
                      "python_float_forward_s": [round(t, 3) for t in times]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
