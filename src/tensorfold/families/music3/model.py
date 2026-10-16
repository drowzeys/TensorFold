"""MiniMax Music 3 in MLX: the AR language model with its depth decoder, the flow-matching DiT and the DAV decoder.

Read straight from Comfy-Org/MiniMax-Music-3's single-file checkpoints (bf16 text encoder, fp16 DiT, fp32 DAV):

* AR: a Qwen3-8B body (36 layers, GQA 32/8, q/k RMS norms, merged qkv and gate|up) over a pruned vocabulary.
  Each audio frame (25 a second) is 8 RVQ codes. The body predicts code 0 (16384 codes + stop) from its last state;
  a 4-layer depth decoder (width 4096, 16 heads of 256, learned positions, no rope) predicts codes 1..7 in turn.
  A frame is fed back as (code-0 embedding + the 7 residual embeddings) / sqrt(8).
  The DiT's condition for a frame is the body state that drew code 0 and the 7 depth-decoder states, 8 x 4096.
* DiT: 36 blocks of width 2048 (32 heads of 64, rope on the first 32 dims of each head, split halves), one leading
  timestep token. Input channels are [latent 128, zeros 128, condition 2048].
* DAV: a DAC-style decoder (snake activations, weight-normed convolutions, strides 8 8 4 2) from 64 latent
  channels to one channel at 44.1 kHz; the 128 latent channels are the left and right halves.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# -- the AR side ---------------------------------------------------------------------------------------------------

HIDDEN = 4096
CODEBOOKS = 8
CODEBOOK_SIZE = 1024
C0_CODES = 16384


class Body(nn.Module):
    """The Qwen3-8B body with merged projections, as the checkpoint stores it."""

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = nn.RMSNorm(HIDDEN, eps=1e-6)
            self.post_attention_layernorm = nn.RMSNorm(HIDDEN, eps=1e-6)
            self.self_attn = Body.Attention()
            self.mlp = Body.MLP()

        def __call__(self, x, mask, cache):
            x = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return x + self.mlp(self.post_attention_layernorm(x))

    class Attention(nn.Module):
        heads, kv_heads, head_dim = 32, 8, 128

        def __init__(self) -> None:
            super().__init__()
            self.qkv_proj = nn.Linear(HIDDEN, (self.heads + 2 * self.kv_heads) * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.heads * self.head_dim, HIDDEN, bias=False)
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.rope = nn.RoPE(self.head_dim, traditional=False, base=1_000_000.0)

        def __call__(self, x, mask, cache):
            B, L, _ = x.shape
            q_w, kv_w = self.heads * self.head_dim, self.kv_heads * self.head_dim
            qkv = self.qkv_proj(x)
            q = self.q_norm(qkv[..., :q_w].reshape(B, L, self.heads, -1)).transpose(0, 2, 1, 3)
            k = self.k_norm(qkv[..., q_w:q_w + kv_w].reshape(B, L, self.kv_heads, -1)).transpose(0, 2, 1, 3)
            v = qkv[..., q_w + kv_w:].reshape(B, L, self.kv_heads, -1).transpose(0, 2, 1, 3)
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask=mask)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    class MLP(nn.Module):
        def __init__(self, width: int = 12288) -> None:
            super().__init__()
            self.gate_up_proj = nn.Linear(HIDDEN, 2 * width, bias=False)
            self.down_proj = nn.Linear(width, HIDDEN, bias=False)

        def __call__(self, x):
            gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
            return self.down_proj(nn.silu(gate) * up)

    def __init__(self) -> None:
        super().__init__()
        self.layers = [Body.Layer() for _ in range(36)]
        self.norm = nn.RMSNorm(HIDDEN, eps=1e-6)

    def __call__(self, x: mx.array, caches: list[Any]) -> mx.array:
        mask = "causal" if x.shape[1] > 1 else None
        for layer, cache in zip(self.layers, caches):
            x = layer(x, mask, cache)
        return self.norm(x)


class DepthLayer(nn.Module):
    heads = 16

    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(HIDDEN, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN, eps=1e-6)
        self.qkv_proj = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = Body.MLP(6144)

    def __call__(self, x, cache):
        B, L, _ = x.shape
        q, k, v = (t.reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
                   for t in mx.split(self.qkv_proj(self.input_layernorm(x)), 3, axis=-1))
        k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=(HIDDEN // self.heads) ** -0.5,
                                                   mask="causal" if L > 1 else None)
        x = x + self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
        return x + self.mlp(self.post_attention_layernorm(x))


class DepthDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.pos_embedding = nn.Embedding(16, HIDDEN)
        self.audio_heads = [nn.Linear(HIDDEN, CODEBOOK_SIZE, bias=False) for _ in range(CODEBOOKS - 1)]
        self.layers = [DepthLayer() for _ in range(4)]
        self.norm = nn.RMSNorm(HIDDEN, eps=1e-6)

    def __call__(self, x: mx.array, position: int, caches: list[Any]) -> mx.array:
        """Projected inputs x [B, L, D] at positions position.. ; the normed state of the last row [B, D]."""

        x = x + self.pos_embedding.weight[position:position + x.shape[1]]
        for layer, cache in zip(self.layers, caches):
            x = layer(x, cache)
        return self.norm(x[:, -1])


class AR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = Body()
        self.embed_tokens_prefill = nn.Embedding(AUDIO_TEXT_VOCAB, HIDDEN)
        self.embed_tokens_audio = nn.Embedding(C0_CODES, HIDDEN)
        self.lm_head_pruned = nn.Linear(HIDDEN, C0_CODES + 1, bias=False)
        self.audio_extra_embedding = nn.Embedding(CODEBOOK_SIZE * (CODEBOOKS - 1), HIDDEN)
        self.audio_decoder = DepthDecoder()
        self._offsets = mx.arange(CODEBOOKS - 1, dtype=mx.int32) * CODEBOOK_SIZE

    @staticmethod
    def remap(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Checkpoint names (``model.*``) to this module's; the depth layers' attention is flattened."""

        out = {}
        for name, value in weights.items():
            if not name.startswith("model."):
                continue
            name = name[len("model."):]
            if name.startswith("layers.") or name == "norm.weight":
                name = "layers." + name
            elif name.startswith("audio_decoder.layers."):
                name = name.replace(".self_attn.", ".")
            out[name] = value
        return out

    def frame_embedding(self, codes: mx.array) -> mx.array:
        """Feedback input of frames codes [B, 8] -> [B, 1, D]."""

        c0 = self.embed_tokens_audio(codes[:, 0])
        extra = self.audio_extra_embedding(codes[:, 1:] + self._offsets).sum(axis=1)
        return ((c0 + extra) * (CODEBOOKS ** -0.5))[:, None]


AUDIO_TEXT_VOCAB = 151675


# -- weight quantization (opt-in) -----------------------------------------------------------------------------------

QUANT_PRESETS = {
    "none": {},
    "int8": {"body": 8, "head": 8, "depth": 8},
    "int8-d6": {"body": 8, "head": 8, "depth": 6},
    "int8-d4": {"body": 8, "head": 8, "depth": 4},
    "int6": {"body": 6, "head": 8, "depth": 6},
    "int4": {"body": 4, "head": 8, "depth": 4},
}
QUANT_PARTS = ("body", "head", "depth", "dit")


def parse_quant(spec: str | None) -> dict:
    """``--quant`` to a plan {"body", "head", "depth", "dit": bits or 0, "group": size}.

    A preset name (none, int8, int8-d6, int8-d4, int6, int4), optionally followed by per-part overrides, or overrides
    alone: "int8", "int8,dit=8", "body=8,head=8,depth=6,group=64". body = the AR's Qwen3-8B layers, head = its code-0
    head, depth = the depth decoder (projection, 4 layers, 7 heads), dit = the DiT blocks' linears. Bits are MLX
    affine quantization (2, 3, 4, 5, 6 or 8; 0 keeps bf16 / fp16); group is the quantization group (32, 64 or 128)."""

    plan = {part: 0 for part in QUANT_PARTS}
    plan["group"] = 64
    for item in (spec or "none").replace(" ", "").split(","):
        if not item:
            continue
        if "=" not in item:
            if item not in QUANT_PRESETS:
                raise ValueError(f"unknown --quant preset {item!r} (one of {', '.join(QUANT_PRESETS)})")
            plan.update(QUANT_PRESETS[item])
            continue
        key, value = item.split("=", 1)
        if key not in plan:
            raise ValueError(f"unknown --quant part {key!r} (one of {', '.join(QUANT_PARTS)}, group)")
        bits = 0 if value in ("0", "none", "bf16", "fp16") else int(value)
        if key == "group" and bits not in (32, 64, 128):
            raise ValueError("--quant group is 32, 64 or 128")
        if key != "group" and bits not in (0, 2, 3, 4, 5, 6, 8):
            raise ValueError(f"--quant {key} bits are 2, 3, 4, 5, 6 or 8")
        plan[key] = bits
    return plan


def quant_name(plan: dict) -> str:
    parts = [f"{p}={plan[p]}" for p in QUANT_PARTS if plan.get(p)]
    return "none" if not parts else ",".join(parts + [f"group={plan['group']}"])


def _linears(path: str, module) -> bool:
    return isinstance(module, nn.Linear)


def quantize_ar(ar: "AR", plan: dict) -> None:
    """Swap the AR's linears for MLX quantized ones in place (embeddings stay bf16: only their rows are read)."""

    group = plan["group"]
    if plan.get("body"):
        nn.quantize(ar.layers, group_size=group, bits=plan["body"], class_predicate=_linears)
    if plan.get("head"):
        ar.lm_head_pruned = nn.QuantizedLinear.from_linear(ar.lm_head_pruned, group, plan["head"])
    if plan.get("depth"):
        nn.quantize(ar.audio_decoder, group_size=group, bits=plan["depth"], class_predicate=_linears)
    mx.eval(ar.parameters())


def quantize_dit(dit: "DiT", plan: dict) -> None:
    """The DiT blocks' linears (attention and feed-forward; 98% of the DiT's weights) quantized in place."""

    if plan.get("dit"):
        for block in dit.blocks:
            nn.quantize(block, group_size=plan["group"], bits=plan["dit"], class_predicate=_linears)
        mx.eval(dit.parameters())


# -- the DiT -------------------------------------------------------------------------------------------------------

def frames_to_latents(frames: int) -> int:
    """Latent frames (44100 / 512 a second) for AR frames (25 a second), the way the model was trained to align."""

    return max(1, int(frames * 44100 / 24000 * 960 / 512))


class LayerNorm(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.gamma = mx.ones((dims,))
        self.beta = mx.zeros((dims,))

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.gamma, self.beta, 1e-5)


class DiTBlock(nn.Module):
    heads, head_dim, rotary = 32, 64, 32

    def __init__(self) -> None:
        super().__init__()
        self.pre_norm = LayerNorm(2048)
        self.ff_norm = LayerNorm(2048)
        self.to_qkv = nn.Linear(2048, 3 * 2048, bias=False)
        self.to_out = nn.Linear(2048, 2048, bias=False)
        self.ff_in = nn.Linear(2048, 2 * 8192)
        self.ff_out = nn.Linear(8192, 2048)

    def __call__(self, x, cos, sin):
        B, L, _ = x.shape
        q, k, v = (t.reshape(B, L, self.heads, self.head_dim).transpose(0, 2, 1, 3)
                   for t in mx.split(self.to_qkv(self.pre_norm(x)), 3, axis=-1))
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5)
        x = x + self.to_out(out.transpose(0, 2, 1, 3).reshape(B, L, -1))
        value, gate = mx.split(self.ff_in(self.ff_norm(x)), 2, axis=-1)
        return x + self.ff_out(value * nn.silu(gate))


def rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rope on the first 2 * cos.shape[-1] dims of x [B, H, L, d], split halves (pair i with i + half)."""

    half = cos.shape[-1]
    a, b, rest = x[..., :half], x[..., half:2 * half], x[..., 2 * half:]
    return mx.concatenate([a * cos - b * sin, b * cos + a * sin, rest], axis=-1)


class DiT(nn.Module):
    """The flow-matching transformer. ``velocity`` returns the reference's model output (already negated)."""

    def __init__(self) -> None:
        super().__init__()
        self.cond_layer_logits = mx.zeros((CODEBOOKS,))
        self.cond_layer_scale = mx.ones((1,))
        self.cond_conv = nn.Conv1d(HIDDEN, 2048, kernel_size=3, padding=1)
        self.preprocess = nn.Linear(2304, 2304, bias=False)          # 1x1 convolutions as linears
        self.postprocess = nn.Linear(128, 128, bias=False)
        self.time_features = mx.zeros((128, 1))
        self.time_in = nn.Linear(256, 2048)
        self.time_out = nn.Linear(2048, 2048)
        self.project_in = nn.Linear(2304, 2048, bias=False)
        self.project_out = nn.Linear(2048, 128, bias=False)
        self.inv_freq = mx.zeros((16,))
        self.blocks = [DiTBlock() for _ in range(36)]
        self.angle_dtype = None        # None: rope angles in float32; a dtype: rounded to it first, as ComfyUI does

    @staticmethod
    def remap(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        names = {
            "latent_conditioners.0.weight": "cond_conv.weight", "latent_conditioners.0.bias": "cond_conv.bias",
            "cond_layer_logits": "cond_layer_logits", "cond_layer_scale": "cond_layer_scale",
            "diffusion_transformer.preprocess_conv.weight": "preprocess.weight",
            "diffusion_transformer.postprocess_conv.weight": "postprocess.weight",
            "diffusion_transformer.timestep_features.weight": "time_features",
            "diffusion_transformer.to_timestep_embed.0.weight": "time_in.weight",
            "diffusion_transformer.to_timestep_embed.0.bias": "time_in.bias",
            "diffusion_transformer.to_timestep_embed.2.weight": "time_out.weight",
            "diffusion_transformer.to_timestep_embed.2.bias": "time_out.bias",
            "diffusion_transformer.transformer.project_in.weight": "project_in.weight",
            "diffusion_transformer.transformer.project_out.weight": "project_out.weight",
            "diffusion_transformer.transformer.rotary_pos_emb.inv_freq": "inv_freq",
        }
        block_names = {"pre_norm.gamma": "pre_norm.gamma", "pre_norm.beta": "pre_norm.beta",
                       "ff_norm.gamma": "ff_norm.gamma", "ff_norm.beta": "ff_norm.beta",
                       "self_attn.to_qkv.weight": "to_qkv.weight", "self_attn.to_out.weight": "to_out.weight",
                       "ff.ff.0.proj.weight": "ff_in.weight", "ff.ff.0.proj.bias": "ff_in.bias",
                       "ff.ff.2.weight": "ff_out.weight", "ff.ff.2.bias": "ff_out.bias"}
        out = {}
        prefix = "diffusion_transformer.transformer.layers."
        for name, value in weights.items():
            for strip in ("model.diffusion_model.", "diffusion_model."):
                if name.startswith(strip):
                    name = name[len(strip):]
            if name.startswith(prefix):
                index, rest = name[len(prefix):].split(".", 1)
                out[f"blocks.{index}.{block_names[rest]}"] = value
                continue
            target = names[name]
            if target == "cond_conv.weight":
                value = value.transpose(0, 2, 1)                       # [O, I, K] -> [O, K, I]
            elif target in ("preprocess.weight", "postprocess.weight"):
                value = value[..., 0]
            out[target] = value
        return out

    # -- the condition ---------------------------------------------------------------------------------------------
    def condition(self, hidden: mx.array) -> mx.array:
        """AR frame states hidden [T, 8 * 4096] -> the latent-aligned condition [L, 2048] (L = frames_to_latents(T))."""

        frames = hidden.shape[0]
        weights = mx.softmax(self.cond_layer_logits.astype(mx.float32), axis=0).astype(hidden.dtype)
        mixed = (hidden.reshape(frames, CODEBOOKS, HIDDEN) * weights[:, None]).sum(axis=1)
        mixed = (self.cond_layer_scale.astype(hidden.dtype) * mixed)[None]
        projected = self.cond_conv(mixed)[0]                         # [T, 2048]
        return projected[mx.array(nearest(frames, frames_to_latents(frames)))]

    def null_condition(self, length: int, dtype) -> mx.array:
        """The unconditional branch: the condition of all-zero states, the projection's bias everywhere."""

        return mx.broadcast_to(self.cond_conv.bias.astype(dtype), (length, 2048))

    # -- one forward over windows ------------------------------------------------------------------------------------
    def rope_tables(self, length: int) -> tuple[mx.array, mx.array]:
        angles = mx.arange(length, dtype=mx.float32)[:, None] * self.inv_freq.astype(mx.float32)[None]
        if self.angle_dtype is not None:
            angles = angles.astype(self.angle_dtype).astype(mx.float32)
        return mx.cos(angles), mx.sin(angles)

    def __call__(self, x: mx.array, t: mx.array, condition: mx.array, tables) -> mx.array:
        """x [B, L, 128] noisy latents, t [B] the reference's timestep (1 - sigma), condition [B, L, 2048].
        Returns the reference's model output (the negated transformer output) [B, L, 128]."""

        dtype = x.dtype
        full = mx.concatenate([x, mx.zeros_like(x), condition], axis=-1)
        full = full + self.preprocess(full)
        angles = 2 * math.pi * t.astype(mx.float32)[:, None] * self.time_features[:, 0].astype(mx.float32)[None]
        features = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1).astype(dtype)
        time = self.time_out(nn.silu(self.time_in(features)))
        h = mx.concatenate([time[:, None], self.project_in(full)], axis=1)
        cos, sin = (a.astype(dtype)[None, None] for a in tables)
        for block in self.blocks:
            h = block(h, cos, sin)
        out = self.project_out(h[:, 1:])
        return -(out + self.postprocess(out))


def nearest(source: int, target: int) -> list[int]:
    """Nearest-neighbour indices of PyTorch's ``interpolate(mode="nearest")`` (float32 scale, floor)."""

    scale = np.float32(source) / np.float32(target)
    index = np.floor(np.arange(target, dtype=np.float32) * scale).astype(np.int64)
    return np.minimum(index, source - 1).tolist()


# -- the DAV decoder -----------------------------------------------------------------------------------------------

def snake(x: mx.array, alpha: mx.array) -> mx.array:
    return x + mx.reciprocal(alpha + 1e-9) * mx.square(mx.sin(alpha * x))


class DAV(nn.Module):
    """Latents [B, L, 128] -> stereo audio [B, 2, 512 L] at 44.1 kHz (float32 throughout, like the reference)."""

    STRIDES = (8, 8, 4, 2)
    DILATIONS = (1, 3, 9)

    def __init__(self, weights: dict[str, mx.array]) -> None:
        super().__init__()
        self.w = fold_weight_norm(weights)

    def conv(self, name: str, x: mx.array, dilation: int = 1, padding: int | None = None) -> mx.array:
        weight = self.w[name + ".weight"]
        k = weight.shape[1]
        pad = (k - 1) // 2 * dilation if padding is None else padding
        return mx.conv1d(x, weight, padding=pad, dilation=dilation) + self.w[name + ".bias"]

    def __call__(self, latents: mx.array) -> mx.array:
        B, L, _ = latents.shape
        x = latents.astype(mx.float32).reshape(B, L, 2, 64).transpose(0, 2, 1, 3).reshape(2 * B, L, 64)
        x = self.conv("dec_in_proj", x)
        x = self.conv("decoder.model.0", x)
        for i, stride in enumerate(self.STRIDES):
            p = f"decoder.model.{i + 1}.block"
            x = snake(x, self.w[p + ".0.alpha"])
            x = mx.conv_transpose1d(x, self.w[p + ".1.weight"], stride=stride,
                                    padding=math.ceil(stride / 2)) + self.w[p + ".1.bias"]
            for j, dilation in enumerate(self.DILATIONS):
                r = f"{p}.{j + 2}.block"
                y = self.conv(r + ".1", snake(x, self.w[r + ".0.alpha"]), dilation=dilation)
                x = x + self.conv(r + ".3", snake(y, self.w[r + ".2.alpha"]))
        x = self.conv("decoder.model.6", snake(x, self.w["decoder.model.5.alpha"]))
        return mx.tanh(x).reshape(B, 2, -1)


def fold_weight_norm(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """weight_g / weight_v pairs to plain weights, all in MLX layouts: conv [O, K, I], transposed conv [O, K, I],
    snake alpha [C] (broadcast over [N, L, C])."""

    out: dict[str, mx.array] = {}
    for name, value in weights.items():
        value = value.astype(mx.float32)
        if name.endswith(".weight_v"):
            base = name[:-len(".weight_v")]
            g = weights[base + ".weight_g"].astype(mx.float32)
            norm = mx.sqrt(mx.square(value).sum(axis=(1, 2), keepdims=True))
            w = g * value / norm                                       # weight norm over dims 1, 2 (dim=0)
            transposed = w.shape[0] != weights[base + ".bias"].shape[0]   # ConvTranspose1d stores [I, O, K]
            out[base + ".weight"] = w.transpose(1, 2, 0) if transposed else w.transpose(0, 2, 1)
        elif name.endswith(".weight_g"):
            continue
        elif name.endswith(".alpha"):
            out[name] = value.reshape(-1)
        elif name == "dec_in_proj.weight":
            out[name] = value.transpose(0, 2, 1)
        else:
            out[name] = value
    mx.eval(out)
    return out
