"""FastH3: FastVideo's step-distilled MiniMax H3 checkpoints, with their routed (sparse) attention."""

# FastH3 checkpoints are whole transformers in the diffusers layout, trained to attend sparsely: the packed
# sequence is cut into tiles of 64 rows, pooled queries and keys score tile against tile, each video tile attends
# to the prefix (text, keyframe and audio rows) and to its best fifth of the video tiles, and a learned
# projection (`to_gate_compress`) gates a pooled copy of the attention back in. The routing itself is FastVideo's
# MLX code, vendored in `vendor/`. This module reads the checkpoint into the H3 module tree and wires the routing
# into its attention. Name and layout mapping as in `lora.py`.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import DiTConfig
from .lora import _rename

CONTRACT = "fastvideo_inference.json"
CONFIG_NAMES = {"num_attention_heads": "num_attention_heads", "attention_head_dim": "attention_head_dim",
                "hidden_size": "hidden_size", "num_layers": "num_layers", "num_refiner_layers":
                "token_refiner_num_layers", "ffn_dim": "ffn_hidden_size", "in_channels": "latents_dim",
                "audio_in_channels": "audio_latents_dim", "text_dim": "text_dim", "freq_dim": "timestep_input_dim",
                "time_embed_hidden_dim": "time_embed_hidden_size", "time_embed_dim": "time_embed_dim",
                "rope_freq_dim": "rope_inv_freq_len", "rope_theta": "rope_theta", "norm_eps": "norm_eps",
                "qk_norm_eps": "qk_norm_eps", "final_norm_eps": "final_norm_eps"}
FLOAT32 = ("video_patch_proj.", "audio_patch_proj.", "time_embedder.", "final_layer.video_out.",
           "final_layer.audio_out.")
GATE = ".attn.to_gate_compress.weight"


@dataclass(frozen=True)
class Contract:
    """How a FastH3 checkpoint was trained to be sampled."""

    nodes: tuple[float, ...]
    video_shift: float
    audio_shift: float
    sparsity: float
    tile: int
    task: str

    @property
    def forwards(self) -> int:
        return len(self.nodes)


def contract(checkpoint_dir) -> Contract:
    with open(Path(checkpoint_dir) / CONTRACT) as handle:
        raw = json.load(handle)
    sparse = raw.get("attention_backend") == "VIDEO_SPARSE_ATTN_H3"
    return Contract(tuple(step / 1000.0 for step in raw["dmd_denoising_steps"]), float(raw["video_scheduler_shift"]),
                    float(raw["audio_scheduler_shift"]), float(raw.get("vsa_sparsity", 0.0)) if sparse else 0.0,
                    int(raw.get("vsa_tile_size", 64)), str(raw.get("task", "")))


def config(checkpoint_dir) -> DiTConfig:
    with open(Path(checkpoint_dir) / "transformer" / "config.json") as handle:
        raw = json.load(handle)
    fields = {ours: raw[theirs] for theirs, ours in CONFIG_NAMES.items() if theirs in raw}
    if "patch_size" in raw:
        fields["patch_size"] = tuple(raw["patch_size"])
    hidden = fields.get("hidden_size", DiTConfig.hidden_size)
    # the diffusers config leaves these implied: six AdaLN tables for each of three modalities, two for the head
    fields.setdefault("adaln_out_features", 18 * hidden)
    fields.setdefault("final_adaln_out_features", 2 * hidden)
    return DiTConfig(**fields)


def assemble(tensors: dict, cfg: DiTConfig) -> tuple[dict, dict]:
    """(parameters under the H3 module names, routing gates by block) from diffusers-named tensors.

    Separate ``to_q``/``to_k``/``to_v`` become each head's ``[q, k, v]`` rows of the fused projection, and the
    SwiGLU projection stored ``[value; gate]`` becomes ``[gate; value]``.
    """

    import mlx.core as mx

    heads, dim = cfg.num_attention_heads, cfg.attention_head_dim
    out, gates, split = {}, {}, {}
    for name, tensor in tensors.items():
        if name.endswith(GATE):
            gates[int(name.split(".")[1])] = tensor
        elif name.rsplit(".", 2)[-2] in ("to_q", "to_k", "to_v") and name.endswith(".weight"):
            prefix, kind, _ = name.rsplit(".", 2)
            split.setdefault(prefix, {})[kind] = tensor
        elif name.endswith(".ff.net.0.proj.weight"):
            half = tensor.shape[0] // 2
            out[_rename(name)] = mx.concatenate([tensor[half:], tensor[:half]], axis=0)
        elif name.endswith(".ff.net.0.proj.bias"):
            half = tensor.shape[0] // 2
            out[_rename(name)] = mx.concatenate([tensor[half:], tensor[:half]], axis=0)
        else:
            # the per-head q/k norms are named the other way round in the checkpoint this family reads
            out[_rename(name).replace(".attn.norm_q.", ".attn.q_norm.").replace(".attn.norm_k.", ".attn.k_norm.")] = tensor
    for prefix, parts in split.items():
        if set(parts) != {"to_q", "to_k", "to_v"}:
            raise KeyError(f"{prefix} has only {sorted(parts)}")
        width = parts["to_q"].shape[1]
        rows = mx.stack([parts[kind].reshape(heads, dim, width) for kind in ("to_q", "to_k", "to_v")], axis=1)
        out[_rename(prefix) + ".qkv_proj.weight"] = rows.reshape(3 * heads * dim, width)
    return out, gates


def load_fasth3(checkpoint_dir, blocks: int | None = None):
    """(transformer, routing gates by block, contract) from a FastH3 checkpoint folder."""

    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from .dit import H3DiT

    root = Path(checkpoint_dir)
    cfg = config(root)
    if blocks is not None:
        cfg = DiTConfig(**{**cfg.__dict__, "num_layers": blocks})
    files = sorted((root / "transformer").glob("diffusion_pytorch_model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no transformer safetensors under {root / 'transformer'}")
    raw = {}
    for file in files:
        raw.update(mx.load(str(file)))
    if blocks is not None:
        raw = {k: v for k, v in raw.items() if not k.startswith("transformer_blocks.")
               or int(k.split(".")[1]) < blocks}
    found, gates = assemble(raw, cfg)
    del raw
    model = H3DiT(cfg)
    expected = {name: value.shape for name, value in tree_flatten(model.parameters())}
    extra = sorted(found.keys() - expected.keys())
    missing = sorted(expected.keys() - found.keys())
    if extra or missing:
        raise KeyError(f"FastH3 checkpoint does not fit the model: {len(extra)} extra (e.g. {extra[:3]}), "
                       f"{len(missing)} missing (e.g. {missing[:3]})")
    for name, tensor in found.items():
        if tuple(tensor.shape) != tuple(expected[name]):
            raise ValueError(f"{name} is {tuple(tensor.shape)}, the model expects {tuple(expected[name])}")
        if name.startswith(FLOAT32) and tensor.dtype != mx.float32:
            found[name] = tensor.astype(mx.float32)
    model.update(tree_unflatten(list(found.items())))
    mx.eval(model.parameters())
    return model, gates, contract(root)


def route(model, gates: dict, sparsity: float, tile: int = 64, impl: str = "reference"):
    """Wire routed attention into ``model``; returns the ``prepare`` hook the sampler calls with the layout.

    ``impl`` is FastVideo's ``reference`` (gather and batched attention) or ``simd`` (its Metal kernel, for tiles
    of 64 and heads of 128; falls back to reference on failure).
    """

    import mlx.core as mx

    from .vendor import fastvideo_vsa as vsa

    cfg = model.config
    heads, dim = cfg.num_attention_heads, cfg.attention_head_dim
    state = {"geometry": None}
    for index, block in enumerate(model.blocks):
        gate = gates.get(index)

        def sparse(attention, x, q, k, v, gate=gate):
            geometry = state["geometry"]
            rows = q.shape[2]
            if geometry is None or geometry.total_seq_length != rows:
                return attention.mix(q, k, v)
            compress = None if gate is None else (x[0] @ gate.T.astype(x.dtype)).reshape(rows, heads, dim)
            out = vsa.h3_vsa_attention(q[0].transpose(1, 0, 2), k[0].transpose(1, 0, 2), v[0].transpose(1, 0, 2),
                                       geometry, sparsity=sparsity, exempt=True, gate_compress=compress, impl=impl)
            return out.reshape(1, rows, heads * dim)

        block.attn.sparse = sparse

    def prepare(packed, latent_frames, latent_height, latent_width):
        pt, ph, pw = cfg.patch_size
        text = int(packed.text_rows.shape[0])
        audio = int(packed.audio_rows.shape[0])
        segments = tuple(n for n in (text, int(packed.condition_video_rows), audio) if n > 0)
        shape = (latent_frames // pt, latent_height // ph, latent_width // pw)
        state["geometry"] = vsa.build_h3_tile_geometry(segments, shape, tile)
        mx.eval(state["geometry"].tile_gather_index, state["geometry"].untile_index)
        return state["geometry"]

    return prepare
