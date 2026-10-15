"""Drama 3 in MLX: the Qwen3.5 MoE backbone (mlx-lm's model), frame embeddings and the codebook tower.

A frame row is [main token, code 0, ..., code 9]; code 0 is the main token's offset in the semantic band.
One step runs the backbone on the newest frame, draws the next main token from the semantic band and EOS,
then runs the 13 tower layers (4 shared, 9 residual) at the same position, each residual layer feeding the
code it was given into the head that draws the next code. Each tower layer keeps its own key/value cache
over the sequence, at the backbone's positions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.families.drama3 import fast


@dataclass(frozen=True)
class Drama3Config:
    eos_token_id: int
    semantic_begin: int
    semantic_end: int
    codebooks: int
    codebook_size: int
    shared_layers: int
    dim: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    norm_eps: float
    rope_base: float
    rotary_dim: int
    text: dict
    tensorfold: dict

    @classmethod
    def read(cls, model_dir: Path) -> "Drama3Config":
        config = json.loads((Path(model_dir) / "config.json").read_text())
        if config.get("model_type") != "fish_omni":
            raise ValueError(f"{model_dir} is not a Drama 3 (fish_omni) checkpoint")
        d = config["audio_decoder_config"]
        if not d.get("rotary_interleaved", True) or not d.get("attention_output_gate", True):
            raise NotImplementedError("the tower is written for interleaved rotary and gated attention")
        if int(d["vq_backbone_codebooks"]) != 1:
            raise NotImplementedError("the backbone embedding is written for one codebook (code 1)")
        return cls(eos_token_id=int(config["eos_token_id"]),
                   semantic_begin=int(config["semantic_start_token_id"]),
                   semantic_end=int(config["semantic_end_token_id"]),
                   codebooks=int(d["num_codebooks"]), codebook_size=int(d["vocab_size"]),
                   shared_layers=int(d["shared_enrichment_layers"]), dim=int(d["dim"]),
                   heads=int(d["n_head"]), kv_heads=int(d["n_local_heads"]), head_dim=int(d["head_dim"]),
                   intermediate=int(d["intermediate_size"]), norm_eps=float(d["norm_eps"]),
                   rope_base=float(d["rope_base"]),
                   rotary_dim=int(int(d["head_dim"]) * float(d["rotary_percent"])),
                   text=config["text_config"], tensorfold=config.get("tensorfold", {}))


def fp8_activations(x: mx.array, group: int = 128) -> mx.array:
    """SGLang's dynamic FP8 MoE inputs: per row and 128-wide group, E4M3 with scale amax / 448, dequantized (fp32)."""

    shape = x.shape
    xg = x.astype(mx.float32).reshape(*shape[:-1], shape[-1] // group, group)
    scale = mx.maximum(mx.abs(xg).max(axis=-1, keepdims=True), 1e-10) / 448.0
    q = mx.from_fp8(mx.to_fp8(mx.clip(xg / scale, -448.0, 448.0)), dtype=mx.float32)
    return (q * scale).reshape(shape)


def emulate_fp8_moe(backbone: Any) -> None:
    """Run the routed experts the way SGLang's FP8 MoE serves them (parity checks against the reference).

    Expert inputs and activations are quantized per 128-wide group to E4M3 and back, expert products stay float32,
    and the routed sum is rounded to bfloat16 before the shared expert is added.
    """

    from mlx_lm.models.switch_layers import SwitchGLU

    class Fp8SwitchGLU(SwitchGLU):
        def __call__(self, x, indices):
            x = mx.expand_dims(fp8_activations(x), (-2, -3))
            up = self.up_proj(x, indices).astype(mx.bfloat16)
            gate = self.gate_proj(x, indices).astype(mx.bfloat16)
            act = (nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)).astype(mx.bfloat16)
            return self.down_proj(fp8_activations(act), indices).squeeze(-2)

    def moe(self, x):
        probs = mx.softmax(self.gate(x).astype(mx.float32), axis=-1)
        k = self.top_k
        inds = mx.argpartition(probs, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(probs, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)
        routed = (self.switch_mlp(x, inds).astype(mx.float32) * scores[..., None]).sum(axis=-2).astype(x.dtype)
        shared = self.shared_expert(x) * mx.sigmoid(self.shared_expert_gate(x))
        return routed + shared

    for layer in backbone.model.layers:
        block = layer.mlp
        block.switch_mlp.__class__ = Fp8SwitchGLU
        block.__class__ = type("Fp8MoeBlock", (type(block),), {"__call__": moe})


def gemma_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm in float32 with a stored 1 + w weight (float32), back to the input's dtype."""

    return mx.fast.rms_norm(x.astype(mx.float32), weight, eps).astype(x.dtype)


class Norm(nn.Module):
    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return gemma_norm(x, self.weight, self.eps)


class TowerAttention(nn.Module):
    def __init__(self, c: Drama3Config) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = c.heads, c.kv_heads, c.head_dim
        q = c.heads * c.head_dim
        kv = c.kv_heads * c.head_dim
        # rows: query (all heads), gate (all heads), key, value -- the checkpoint's layout
        self.wqkv = nn.Linear(c.dim, 2 * q + 2 * kv, bias=False)
        self.wo = nn.Linear(q, c.dim, bias=False)
        self.q_norm = Norm(c.head_dim, c.norm_eps)
        self.k_norm = Norm(c.head_dim, c.norm_eps)
        self.rotary_dim, self.rope_base = c.rotary_dim, c.rope_base
        self.scale = c.head_dim ** -0.5

    def project(self, x: mx.array, offset) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Rotated queries and keys, values [B, heads, L, head_dim] and the output gate [B, L, heads * head_dim]."""

        B, L, _ = x.shape
        q_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        qkv = self.wqkv(x)
        queries = qkv[..., :q_width].reshape(B, L, self.heads, self.head_dim)
        gate = qkv[..., q_width:2 * q_width]
        keys = qkv[..., 2 * q_width:2 * q_width + kv_width].reshape(B, L, self.kv_heads, self.head_dim)
        values = qkv[..., 2 * q_width + kv_width:].reshape(B, L, self.kv_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = mx.fast.rope(queries, self.rotary_dim, traditional=True, base=self.rope_base, scale=1.0,
                               offset=offset)
        keys = mx.fast.rope(keys, self.rotary_dim, traditional=True, base=self.rope_base, scale=1.0, offset=offset)
        return queries, keys, values, gate

    def finish(self, out: mx.array, gate: mx.array) -> mx.array:
        B, _, L, _ = out.shape
        return self.wo(out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate))

    def __call__(self, x: mx.array, cache: Any) -> mx.array:
        queries, keys, values, gate = self.project(x, cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
        mask = "causal" if x.shape[1] > 1 else None
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        return self.finish(out, gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def fuse(self) -> None:
        """w1 and w3 as one [2 * hidden, dim] matrix (w1 rows first); the separate matrices are dropped."""

        if "w13" in self:
            return
        w13 = nn.Linear(int(self.w1.weight.shape[1]), 2 * int(self.w1.weight.shape[0]), bias=False)
        w13.weight = mx.concatenate([self.w1.weight, self.w3.weight], axis=0)
        mx.eval(w13.weight)
        self.pop("w1")
        self.pop("w3")
        self.w13 = w13

    def __call__(self, x: mx.array) -> mx.array:
        if "w13" in self:
            gate, up = mx.split(self.w13(x), 2, axis=-1)
            return self.w2(nn.silu(gate) * up)
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TowerLayer(nn.Module):
    """Mix a code embedding into the hidden state, then one pre-norm attention + MLP block."""

    def __init__(self, c: Drama3Config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(c.codebook_size, c.dim)
        self.enorm = Norm(c.dim, c.norm_eps)
        self.hnorm = Norm(c.dim, c.norm_eps)
        self.eh_proj = nn.Linear(2 * c.dim, c.dim, bias=False)
        self.attention = TowerAttention(c)
        self.feed_forward = FeedForward(c.dim, c.intermediate)
        self.attention_norm = Norm(c.dim, c.norm_eps)
        self.ffn_norm = Norm(c.dim, c.norm_eps)
        self.final_layernorm = Norm(c.dim, c.norm_eps)
        self.codebook_size = c.codebook_size

    def __call__(self, hidden: mx.array, codes: mx.array, semantic: mx.array, extra: mx.array | None,
                 cache: Any) -> mx.array:
        """hidden [B, L, D]; codes, semantic [B, L]; extra [B, L, D] or None."""

        embedding = self.token_embedding(mx.clip(codes, 0, self.codebook_size - 1))
        embedding = mx.where(semantic[..., None], embedding, mx.zeros_like(embedding))
        if extra is not None:
            embedding = embedding + extra.astype(embedding.dtype)
        mixed = self.eh_proj(mx.concatenate([self.enorm(embedding), self.hnorm(hidden)], axis=-1))
        h = self.attention(self.attention_norm(mixed), cache) + mixed
        return self.final_layernorm(self.feed_forward(self.ffn_norm(h)) + h)


class Tower(nn.Module):
    def __init__(self, c: Drama3Config) -> None:
        super().__init__()
        self.codebook_embeddings = nn.Embedding(c.codebook_size * c.codebooks, c.dim)
        self.layers = [TowerLayer(c) for _ in range(c.shared_layers + c.codebooks - 1)]
        self.heads = [nn.Linear(c.dim, c.codebook_size, bias=False) for _ in range(c.codebooks - 1)]


class Drama3(nn.Module):
    def __init__(self, c: Drama3Config) -> None:
        super().__init__()
        from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

        self.config = c
        text = dict(c.text)
        text["full_attention_interval"] = text["layer_types"].index("full_attention") + 1
        self.backbone = TextModel(TextModelArgs.from_dict(text))
        self.tower = Tower(c)
        candidates = sorted({c.eos_token_id, *range(c.semantic_begin, c.semantic_end + 1)})
        self._candidate_ids = mx.array(candidates, dtype=mx.int32)
        self.eos_index = candidates.index(c.eos_token_id)
        self._offsets = mx.arange(c.codebooks, dtype=mx.int32) * c.codebook_size
        self._head_rows: mx.array | None = None       # lm_head rows of the candidates, set after loading
        self.fast_step: Any = None                         # the fused, compiled single-row step (fast.FastStep)

    # -- loading ----------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, model_dir: str | Path) -> "Drama3":
        from mlx.utils import tree_flatten

        model_dir = Path(model_dir)
        c = Drama3Config.read(model_dir)
        if not c.tensorfold.get("format"):
            raise SystemExit(f"[drama3] {model_dir} holds Fish's own checkpoint; convert it first:\n"
                             f"  python -m tensorfold.families.drama3.convert {model_dir} <out dir>")
        model = cls(c)
        quant = c.tensorfold.get("expert_quantization")
        if quant:
            nn.quantize(model.backbone, group_size=int(quant["group_size"]), bits=int(quant["bits"]),
                        mode=quant.get("mode", "affine"),
                        class_predicate=lambda path, module: path.endswith("switch_mlp.gate_proj")
                        or path.endswith("switch_mlp.up_proj") or path.endswith("switch_mlp.down_proj"))
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        weights: dict = {}
        for file_name in sorted(set(index.values())):
            weights.update(mx.load(str(model_dir / file_name)))
        expected = {name for name, _ in tree_flatten(model.parameters())}
        missing, unknown = expected - set(weights), set(weights) - expected
        if missing or unknown:
            raise ValueError(f"weights do not match the model: missing {sorted(missing)[:6]}, "
                             f"unknown {sorted(unknown)[:6]}")
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
        fp8 = os.environ.get("TF_DRAMA3_MOE_ACT", "bf16") == "fp8"
        if fp8:
            emulate_fp8_moe(model.backbone)
            print("[drama3] MoE inputs emulate SGLang's FP8 activations (TF_DRAMA3_MOE_ACT=fp8)", flush=True)
        model._head_rows = mx.take(model.backbone.lm_head.weight, model._candidate_ids, axis=0)
        mx.eval(model._head_rows)
        if fast.enabled() and not fp8:
            fast.fuse_model(model, experts=not fp8)
            model.fast_step = fast.FastStep(model)
        return model

    # -- caches -----------------------------------------------------------------------------------------------------
    def make_cache(self) -> tuple[list[Any], list[Any]]:
        from mlx_lm.models.cache import KVCache

        return self.backbone.make_cache(), [KVCache() for _ in self.tower.layers]

    # -- embeddings -------------------------------------------------------------------------------------------------
    def is_semantic(self, tokens: mx.array) -> mx.array:
        return (tokens >= self.config.semantic_begin) & (tokens <= self.config.semantic_end)

    def code_sum(self, frames: mx.array, first: int, end: int) -> mx.array:
        """Summed codebook embeddings of codes first..end-1 of frames [..., 11] (float32)."""

        rows = mx.clip(frames[..., 1 + first:1 + end], 0, self.config.codebook_size - 1) + self._offsets[first:end]
        return self.tower.codebook_embeddings(rows).astype(mx.float32).sum(axis=-2)

    def embed(self, frames: mx.array) -> mx.array:
        """Backbone input of frames [B, L, 11]: semantic frames add the code 1 embedding, scaled 1/sqrt(2)."""

        text = self.backbone.model.embed_tokens(frames[..., 0])
        mixed = (text + self.code_sum(frames, 1, 2).astype(text.dtype)) * (1.0 / math.sqrt(2.0))
        return mx.where(self.is_semantic(frames[..., 0])[..., None], mixed, text)

    def shared_extra(self, frames: mx.array) -> mx.array:
        """The first shared layer's extra input: codes 2..9 of semantic frames."""

        extra = self.code_sum(frames, 2, self.config.codebooks)
        return mx.where(self.is_semantic(frames[..., 0])[..., None], extra, mx.zeros_like(extra))

    # -- forwards ---------------------------------------------------------------------------------------------------
    def backbone_hidden(self, frames: mx.array, cache: list[Any]) -> mx.array:
        """Final-norm backbone states [B, L, 2048] of frames [B, L, 11]."""

        if self.fast_step is not None and frames.shape[:2] == (1, 1):
            return self.fast_step.backbone(frames, cache)
        return self.backbone.model(None, cache, input_embeddings=self.embed(frames))

    def candidate_logits(self, hidden: mx.array) -> mx.array:
        """Logits over [semantic band + EOS] (sorted ids, ``candidate_ids``), float32."""

        return (hidden @ self._head_rows.T).astype(mx.float32)

    def tower_prefill(self, hidden: mx.array, frames: mx.array, next_frames: mx.array, caches: list[Any]) -> None:
        """Fill every tower layer's cache with prompt rows: row p is labelled with frame p + 1."""

        c = self.config
        semantic = self.is_semantic(next_frames[..., 0])
        extra = self.shared_extra(frames)
        for i, layer in enumerate(self.tower.layers):
            codebook = 0 if i < c.shared_layers else i - c.shared_layers
            hidden = layer(hidden, next_frames[..., 1 + codebook], semantic, extra if i == 0 else None, caches[i])

    def tower_step(self, hidden: mx.array, frame: mx.array, main: mx.array, code0: mx.array, caches: list[Any],
                   draw_code) -> list[mx.array]:
        """Run the tower for one row: hidden [B, 1, D], the row's input frame [B, 1, 11], its sampled main token
        and code 0 [B]; ``draw_code(logits [B, V], head index)`` draws codes 1..9. Returns codes 0..9."""

        c = self.config
        semantic = self.is_semantic(main)[:, None]
        extra = self.shared_extra(frame)
        h = hidden
        for i in range(c.shared_layers):
            h = self.tower.layers[i](h, code0[:, None], semantic, extra if i == 0 else None, caches[i])
        codes = [code0]
        for r, head in enumerate(self.tower.heads):
            h = self.tower.layers[c.shared_layers + r](h, codes[-1][:, None], semantic, None,
                                                        caches[c.shared_layers + r])
            codes.append(draw_code(head(h[:, -1]).astype(mx.float32), r))
        return codes
