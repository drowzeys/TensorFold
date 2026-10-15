"""Convert Fish's Drama 3 checkpoint to the MLX weight set this family loads.

The routed experts arrive as FP8 E4M3 with one float32 ``weight_scale_inv`` per 128x128 block. Each is
dequantized (value times its block scale), stacked per layer into ``switch_mlp`` tensors and quantized to
MLX 8-bit affine (groups of 64) or mxfp8. Every other backbone and tower tensor stays bfloat16; norm weights
that the checkpoint stores zero-centred (applied as 1 + w) are stored as 1 + w. The codec stays float32.

    python -m tensorfold.families.drama3.convert ~/models/drama-3 ~/models/drama-3-mlx

The source folder is only read.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from tensorfold.families.drama3 import CONVERTED_FORMAT

BACKBONE = "model.language_model."
EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
# zero-centred norms of the Qwen3.5 backbone (mlx-lm's sanitize list); the DeltaNet gated norm is not one
BACKBONE_NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
                  ".q_norm.weight", ".k_norm.weight")
TOWER_RENAMES = (
    ("semantic_embedding.", "token_embedding."),
    ("codebook_embedding.", "token_embedding."),
    ("transformer_layer.", ""),
)
FISH_NOTICE = """These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0.
Copyright \u00a9 39 AI, INC. All Rights Reserved. (See LICENSE.)

Built with Fish Audio.

Modified by TensorFold's converter (tensorfold.families.drama3.convert): the routed MoE experts were
dequantized from FP8 E4M3 with 128x128 block scales and requantized to MLX 8-bit affine (or mxfp8) and stacked
per layer; zero-centred norm weights are stored as 1 + w; the codec weights were converted from codec.pth to
safetensors; tensors were renamed for the MLX model. For local use only: TensorFold does not distribute this folder.
"""
COPY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "LICENSE", "README.md")


def tower_name(name: str, shared_layers: int) -> str:
    """The converted name of an ``audio_decoder*`` tensor (shared layers first, then residual layers)."""

    if name.startswith("audio_decoder.codebook_embeddings."):
        return "tower." + name.removeprefix("audio_decoder.")
    if name.startswith("audio_decoder_mtp.heads."):
        return "tower." + name.removeprefix("audio_decoder_mtp.")
    match = re.match(r"audio_decoder_mtp(_shared)?\.layers\.(\d+)\.(.+)$", name)
    if match is None:
        raise ValueError(f"{name!r} is not a Drama 3 tower tensor")
    index = int(match.group(2)) + (0 if match.group(1) else shared_layers)
    rest = match.group(3)
    for old, new in TOWER_RENAMES:
        rest = rest.replace(old, new)
    return f"tower.layers.{index}.{rest}"


def dequantize_fp8(weight, scale_inv, block: tuple[int, int]):
    """FP8 E4M3 bytes [rows, cols] times their per-block scales -> float32."""

    import mlx.core as mx

    rows, cols = weight.shape
    br, bc = block
    values = mx.from_fp8(weight, dtype=mx.float32).reshape(rows // br, br, cols // bc, bc)
    return (values * scale_inv[:, None, :, None].astype(mx.float32)).reshape(rows, cols)


def quantize(stacked, mode: str):
    import mlx.core as mx

    if mode == "affine8":
        q, scales, biases = mx.quantize(stacked, group_size=64, bits=8, mode="affine")
        return {"weight": q, "scales": scales.astype(mx.bfloat16), "biases": biases.astype(mx.bfloat16)}
    if mode == "mxfp8":
        q, scales = mx.quantize(stacked, group_size=32, bits=8, mode="mxfp8")
        return {"weight": q, "scales": scales}
    if mode == "bf16":
        return {"weight": stacked.astype(mx.bfloat16)}
    raise ValueError(f"unknown expert format {mode!r}")


def expert_quantization(mode: str) -> dict | None:
    return {"affine8": {"group_size": 64, "bits": 8, "mode": "affine"},
            "mxfp8": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}.get(mode)


def convert_codec_file(source: Path, target: Path) -> None:
    import mlx.core as mx

    from tensorfold.families.drama3.codec import convert_codec

    target.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(target / "codec.safetensors"), convert_codec(source / "codec.pth"),
                        metadata={"format": "mlx"})
    print(f"[drama3] wrote {target / 'codec.safetensors'}", flush=True)


def convert(source: Path, target: Path, *, experts: str = "affine8", shard_layers: int = 4) -> None:
    import mlx.core as mx

    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    decoder = config["audio_decoder_config"]
    block = tuple(config.get("quantization_config", {}).get("weight_block_size", (128, 128)))
    layers = int(text["num_hidden_layers"])
    num_experts = int(text["num_experts"])
    shared_layers = int(decoder["shared_enrichment_layers"])
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {name: mx.load(str(source / name)) for name in sorted(set(weight_map.values()))}

    def tensor(name: str):
        return shards[weight_map[name]][name]

    target.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}

    def save(file_name: str, tensors: dict) -> None:
        mx.eval(*tensors.values())
        mx.save_safetensors(str(target / file_name), tensors, metadata={"format": "mlx"})
        index.update({name: file_name for name in tensors})

    started = time.time()
    rest_names = [n for n in weight_map if not EXPERT.match(n) and not n.endswith("weight_scale_inv")]
    # -- backbone layers, a few per file -----------------------------------------------------------------------
    for first in range(0, layers, shard_layers):
        tensors: dict = {}
        for layer in range(first, min(first + shard_layers, layers)):
            prefix = f"{BACKBONE}layers.{layer}."
            for name in [n for n in rest_names if n.startswith(prefix)]:
                value = tensor(name)
                if value.dtype == mx.uint8:
                    raise ValueError(f"{name} is FP8; only routed experts are expected to be")
                out = "backbone.model." + name.removeprefix(BACKBONE)
                if "conv1d.weight" in name and value.shape[-1] != 1:
                    value = value.moveaxis(2, 1)
                if any(out.endswith(s) for s in BACKBONE_NORMS):
                    value = (value.astype(mx.float32) + 1.0).astype(mx.bfloat16)
                elif name.endswith("A_log"):
                    value = value.astype(mx.float32)
                else:
                    value = value.astype(mx.bfloat16)
                tensors[out] = value
            for proj in ("gate", "up", "down"):
                stacked = mx.stack([
                    dequantize_fp8(tensor(f"{prefix}mlp.experts.{e}.{proj}_proj.weight"),
                                   tensor(f"{prefix}mlp.experts.{e}.{proj}_proj.weight_scale_inv"), block)
                    for e in range(num_experts)])
                for part, value in quantize(stacked, experts).items():
                    tensors[f"backbone.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"] = value
                mx.eval(*[v for k, v in tensors.items() if f"layers.{layer}.mlp.switch_mlp.{proj}" in k])
        save(f"backbone-{first:02d}.safetensors", tensors)
        print(f"[drama3] layers {first}-{min(first + shard_layers, layers) - 1} done "
              f"({time.time() - started:.0f}s)", flush=True)
    # -- embeddings, final norm, lm_head -------------------------------------------------------------------------
    tensors = {}
    for name in [n for n in rest_names if n.startswith(BACKBONE) and not n.startswith(f"{BACKBONE}layers.")]:
        out = "backbone.model." + name.removeprefix(BACKBONE)
        value = tensor(name)
        value = (value.astype(mx.float32) + 1.0) if out.endswith("model.norm.weight") else value
        tensors[out] = value.astype(mx.bfloat16)
    tensors["backbone.lm_head.weight"] = tensor("lm_head.weight").astype(mx.bfloat16)
    save("backbone-embed.safetensors", tensors)
    # -- codebook tower -----------------------------------------------------------------------------------------
    tensors = {}
    for name in [n for n in rest_names if n.startswith("audio_decoder")]:
        out = tower_name(name, shared_layers)
        value = tensor(name)
        if out.endswith("norm.weight"):     # every tower norm is a zero-centred (Gemma) RMSNorm
            value = value.astype(mx.float32) + 1.0
        else:
            value = value.astype(mx.bfloat16)
        tensors[out] = value
    save("tower.safetensors", tensors)
    leftover = set(weight_map) - {n for n in weight_map if EXPERT.match(n) or n.endswith("weight_scale_inv")} \
        - set(rest_names)
    if leftover:
        raise ValueError(f"unconverted tensors: {sorted(leftover)[:8]}")
    for name in COPY_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    out_config = json.loads((source / "config.json").read_text())
    out_config.pop("quantization_config", None)
    out_config["tensorfold"] = {"format": CONVERTED_FORMAT, "experts": experts,
                                "expert_quantization": expert_quantization(experts),
                                "source": str(source), "codec": "codec.safetensors"}
    (target / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
    (target / "NOTICE").write_text(FISH_NOTICE)
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": CONVERTED_FORMAT}, "weight_map": index}, indent=1) + "\n")
    print(f"[drama3] wrote {target} in {time.time() - started:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Fish's Drama 3 folder (safetensors, codec.pth)")
    parser.add_argument("target", type=Path, help="output folder")
    parser.add_argument("--experts", choices=("affine8", "mxfp8", "bf16"), default="affine8",
                        help="routed expert format (default: MLX 8-bit affine, groups of 64)")
    parser.add_argument("--part", choices=("all", "model", "codec"), default="all",
                        help="convert the model, the codec (needs torch to read codec.pth), or both")
    args = parser.parse_args(argv)
    source, target = args.source.expanduser(), args.target.expanduser()
    if args.part in ("all", "model"):
        convert(source, target, experts=args.experts)
    if args.part in ("all", "codec"):
        convert_codec_file(source, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
