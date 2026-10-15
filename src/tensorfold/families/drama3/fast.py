"""Drama 3's decode step, fused and compiled.

A decode step runs one row through the backbone (40 layers), draws the main token and runs the 13 tower layers with
nine sequential code draws. At one row every matrix product is a matrix-vector product, so the step is bound by
memory traffic and by the number of GPU kernels it launches. This module cuts both:

- projections that read the same input are fused into one matrix (linear attention q|k|v|z|b|a, full attention
  q|k|v, the router with the shared expert's gate|up and its gate, the routed experts' gate|up, the tower's w1|w3),
  which the eager forwards (prefill, parity probes) use as well, so no weight is held twice;
- each layer's single-row work is one ``mx.compile``d function (the linear-attention layers whole, with their
  convolution and recurrent states passed in and out; the full-attention layers and tower layers around the
  key/value cache update and attention, which stay eager because the cache grows);
- the main-token draw (candidate logits, top-k/top-p, RAS) and each code draw are compiled with the head before it.

Set ``TF_DRAMA3_FAST=0`` to run the plain mlx-lm forwards instead (the parity reference for this module).
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.families.drama3.sampling import Sampling, draw, draw_main


def enabled() -> bool:
    return os.environ.get("TF_DRAMA3_FAST", "1") != "0"


# -- weight fusion ------------------------------------------------------------------------------------------------

def _fused_linear(*parts: nn.Linear) -> nn.Linear:
    weight = mx.concatenate([p.weight for p in parts], axis=0)
    fused = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=False)
    fused.weight = weight
    return fused


def _project(linear: nn.Linear, x: mx.array, widths: list[int]) -> list[mx.array]:
    """``x`` through a fused projection, split into parts of ``widths``.

    A single row takes one matrix-vector product, which gives the separate products bit for bit. Longer inputs
    (prefill) multiply by each part's rows on their own: the matrix kernels MLX picks for a narrow output are not
    the ones it picks for the fused width, and their rounding differs.
    """

    bounds = [0]
    for w in widths:
        bounds.append(bounds[-1] + w)
    if x.ndim < 2 or x.shape[-2] == 1:
        p = linear(x)
        return [p[..., a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    weight = linear.weight
    return [x @ weight[a:b].T for a, b in zip(bounds[:-1], bounds[1:])]


def _swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


class FusedSwitchGLU(nn.Module):
    """mlx-lm's SwitchGLU with the gate and up projections in one gathered matmul (gate rows first)."""

    def __call__(self, x, indices):
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate, up = mx.split(self.gate_up_proj(x, idx, sorted_indices=do_sort), 2, axis=-1)
        x = self.down_proj(_swiglu(gate, up), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


def _fuse_switch(switch: Any) -> None:
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    gate, up = switch.gate_proj, switch.up_proj
    if isinstance(gate, QuantizedSwitchLinear):
        fused = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
        nn.Module.__init__(fused)
        fused.weight = mx.concatenate([gate.weight, up.weight], axis=1)
        fused.scales = mx.concatenate([gate.scales, up.scales], axis=1)
        fused.biases = None if gate.biases is None else mx.concatenate([gate.biases, up.biases], axis=1)
        fused.group_size, fused.bits, fused.mode = gate.group_size, gate.bits, gate.mode
        fused.freeze()
    else:
        from mlx_lm.models.switch_layers import SwitchLinear

        fused = SwitchLinear.__new__(SwitchLinear)
        nn.Module.__init__(fused)
        fused.weight = mx.concatenate([gate.weight, up.weight], axis=1)
    mx.eval(fused.parameters())
    switch.pop("gate_proj")
    switch.pop("up_proj")
    switch.gate_up_proj = fused
    switch.__class__ = FusedSwitchGLU


class FusedMoeBlock(nn.Module):
    """Qwen3-Next's sparse MoE block with the router, the shared expert's gate|up and its gate in one matmul."""

    def router_logits(self, x: mx.array) -> mx.array:
        return _project(self.in_proj, x, [self.num_experts, 2 * self.shared_width + 1])[0]

    def __call__(self, x: mx.array) -> mx.array:
        E, S = self.num_experts, self.shared_width
        router, shared_gate, shared_up, expert_gate = _project(self.in_proj, x, [E, S, S, 1])
        gates = mx.softmax(router, axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
        shared = self.shared_down(_swiglu(shared_gate, shared_up))
        return y + mx.sigmoid(expert_gate) * shared


def router_logits(block: Any, x: mx.array) -> mx.array:
    """The MoE router's logits, whether or not the block is fused."""

    return block.router_logits(x) if isinstance(block, FusedMoeBlock) else block.gate(x)


def _fuse_moe(block: Any, *, experts: bool) -> None:
    if experts:
        _fuse_switch(block.switch_mlp)
    block.shared_width = int(block.shared_expert.gate_proj.weight.shape[0])
    block.in_proj = _fused_linear(block.gate, block.shared_expert.gate_proj, block.shared_expert.up_proj,
                                  block.shared_expert_gate)
    block.shared_down = block.shared_expert.down_proj
    mx.eval(block.in_proj.parameters())
    for name in ("gate", "shared_expert", "shared_expert_gate"):
        block.pop(name)
    block.__class__ = FusedMoeBlock


class FusedGatedDeltaNet(nn.Module):
    """mlx-lm's GatedDeltaNet with in_proj q|k|v, z, b, a fused (eager, any length)."""

    def split_proj(self, x: mx.array):
        B, S, _ = x.shape
        qkv, z, b, a = _project(self.in_proj_all, x, [self.conv_dim, self.value_dim, self.num_v_heads,
                                                       self.num_v_heads])
        return qkv, z.reshape(B, S, self.num_v_heads, self.head_v_dim), b, a

    def mix(self, qkv, z, b, a, conv_state, state, mask=None):
        """Convolution, recurrence, gated norm and output projection; returns (out, conv_input, state)."""

        from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

        B, S = qkv.shape[:2]
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [t.reshape(B, S, h, d) for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim])]
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim ** -0.5, eps=1e-6)
        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                                        use_kernel=not self.training)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1)), conv_input, state

    def __call__(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
        B = inputs.shape[0]
        qkv, z, b, a = self.split_proj(inputs)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        state = cache[1] if cache else None
        out, conv_input, state = self.mix(qkv, z, b, a, conv_state, state, mask)
        if cache is not None:
            if cache.lengths is not None:
                raise NotImplementedError("Drama 3 runs one unpadded sequence")
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):, :])
            cache[1] = state
            cache.advance(inputs.shape[1])
        return out


def _fuse_linear_attention(attn: Any) -> None:
    attn.in_proj_all = _fused_linear(attn.in_proj_qkv, attn.in_proj_z, attn.in_proj_b, attn.in_proj_a)
    mx.eval(attn.in_proj_all.parameters())
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        attn.pop(name)
    attn.__class__ = FusedGatedDeltaNet


class FusedAttention(nn.Module):
    """Qwen3-Next full attention with q(+gate)|k|v in one matmul (eager, any length)."""

    def project(self, x: mx.array, offset):
        B, L, _ = x.shape
        kw = self.num_key_value_heads * self.head_dim
        q, keys, values = _project(self.qkv_proj, x, [self.num_attention_heads * self.head_dim * 2, kw, kw])
        queries, gate = mx.split(q.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = self.rope(self.q_norm(queries).transpose(0, 2, 1, 3), offset=offset)
        keys = self.rope(self.k_norm(keys).transpose(0, 2, 1, 3), offset=offset)
        return queries, keys, values, gate

    def finish(self, out: mx.array, gate: mx.array) -> mx.array:
        B, _, L, _ = out.shape
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1) * mx.sigmoid(gate))

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        from mlx_lm.models.base import scaled_dot_product_attention

        queries, keys, values, gate = self.project(x, 0 if cache is None else cache.offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        return self.finish(out, gate)


def _fuse_attention(attn: Any) -> None:
    attn.qkv_proj = _fused_linear(attn.q_proj, attn.k_proj, attn.v_proj)
    mx.eval(attn.qkv_proj.parameters())
    for name in ("q_proj", "k_proj", "v_proj"):
        attn.pop(name)
    attn.__class__ = FusedAttention


def fuse_model(model: Any, *, experts: bool = True) -> None:
    """Fuse the backbone's and tower's projections in place (the old matrices are dropped)."""

    for layer in model.backbone.model.layers:
        if layer.is_linear:
            _fuse_linear_attention(layer.linear_attn)
        else:
            _fuse_attention(layer.self_attn)
        _fuse_moe(layer.mlp, experts=experts)
        mx.clear_cache()
    for layer in model.tower.layers:
        layer.feed_forward.fuse()
        mx.clear_cache()


# -- the compiled decode step -------------------------------------------------------------------------------------

class FastStep:
    """Single-row backbone and tower forwards of a fused model, as compiled pieces around the attention caches."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.c = model.config
        core = model.backbone.model
        self.layers = []
        for layer in core.layers:
            if not isinstance(layer.mlp, FusedMoeBlock):
                raise ValueError("FastStep needs a fused model (fuse_model)")
            if layer.is_linear:
                self.layers.append(("linear", mx.compile(self._linear_layer(layer))))
            else:
                pre, post = self._attention_layer(layer)
                self.layers.append(("full", mx.compile(pre), mx.compile(post), layer.self_attn.scale))
        self.embed = mx.compile(model.embed)
        self.final_norm = core.norm
        self.tower = [self._tower_layer(i, layer) for i, layer in enumerate(model.tower.layers)]
        self._heads: dict = {}

    # backbone ------------------------------------------------------------------------------------------------------
    @staticmethod
    def _linear_layer(layer):
        attn, mlp = layer.linear_attn, layer.mlp

        def run(x, conv_state, state):
            qkv, z, b, a = attn.split_proj(layer.input_layernorm(x))
            out, conv_input, state = attn.mix(qkv, z, b, a, conv_state, state)
            h = x + out
            h = h + mlp(layer.post_attention_layernorm(h))
            return h, conv_input[:, 1:, :], state

        return run

    @staticmethod
    def _attention_layer(layer):
        attn, mlp = layer.self_attn, layer.mlp

        def pre(x, offset):
            return attn.project(layer.input_layernorm(x), offset)

        def post(x, out, gate):
            h = x + attn.finish(out, gate)
            return h + mlp(layer.post_attention_layernorm(h))

        return pre, post

    def backbone(self, frames: mx.array, cache: list) -> mx.array:
        """Final-norm backbone state [1, 1, D] of one frame row [1, 1, 11]."""

        x = self.embed(frames)
        offset = None
        for spec, layer_cache in zip(self.layers, cache):
            if spec[0] == "linear":
                if layer_cache[1] is None:
                    raise ValueError("the fast step decodes after a prefill")
                x, layer_cache[0], layer_cache[1] = spec[1](x, layer_cache[0], layer_cache[1])
                layer_cache.advance(1)
            else:
                _, pre, post, scale = spec
                if offset is None:
                    offset = mx.array(layer_cache.offset, dtype=mx.int32)
                q, k, v, gate = pre(x, offset)
                k, v = layer_cache.update_and_fetch(k, v)
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
                x = post(x, out, gate)
        return self.final_norm(x)

    # tower ---------------------------------------------------------------------------------------------------------
    def _tower_layer(self, i: int, layer: Any):
        attn = layer.attention

        def pre(hidden, codes, semantic, extra, offset):
            embedding = layer.token_embedding(mx.clip(codes, 0, layer.codebook_size - 1))
            embedding = mx.where(semantic[..., None], embedding, mx.zeros_like(embedding))
            if extra is not None:
                embedding = embedding + extra.astype(embedding.dtype)
            mixed = layer.eh_proj(mx.concatenate([layer.enorm(embedding), layer.hnorm(hidden)], axis=-1))
            q, k, v, gate = attn.project(layer.attention_norm(mixed), offset)
            return q, k, v, gate, mixed

        def post(out, gate, mixed):
            h = attn.finish(out, gate) + mixed
            return layer.final_layernorm(layer.feed_forward(layer.ffn_norm(h)) + h)

        first = i == 0
        if first:
            pre_fn = mx.compile(pre)
        else:
            pre_fn = mx.compile(lambda hidden, codes, semantic, offset: pre(hidden, codes, semantic, None, offset))
        return {"pre": pre_fn, "post": post, "attn": attn, "first": first, "draws": {}}

    def _tower_post(self, i: int, sampling: Sampling | None):
        """Layer i's compiled post-attention part; with ``sampling``, followed by its head and code draw."""

        spec = self.tower[i]
        key = sampling            # a frozen dataclass: every setting is part of the key
        if key not in spec["draws"]:
            post = spec["post"]
            if sampling is None:
                spec["draws"][key] = mx.compile(post)
            else:
                head = self.model.tower.heads[i - self.c.shared_layers]

                def post_draw(out, gate, mixed, rng):
                    h = post(out, gate, mixed)
                    logits = head(h[:, -1]).astype(mx.float32)
                    return h, draw(logits, sampling.temperature, sampling.top_k, sampling.top_p,
                                   None if sampling.greedy else rng, sampling.greedy)

                spec["draws"][key] = mx.compile(post_draw)
        return spec["draws"][key]

    def _attend(self, spec, q, k, v, cache):
        k, v = cache.update_and_fetch(k, v)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=spec["attn"].scale, mask=None)

    def tower_step(self, hidden, frame, main, code0, caches, sampling: Sampling, keys) -> list[mx.array]:
        """Codes 0..9 [B] of one row: as ``Drama3.tower_step`` with draws by ``sampling`` and ``keys`` [9]."""

        h, codes = self.tower_head(hidden, frame, main, code0, caches, sampling, keys)
        return self.tower_tail(h, main, codes, caches, sampling, keys)

    def tower_head(self, hidden, frame, main, code0, caches, sampling: Sampling, keys):
        """The shared layers and the first residual layer: (hidden after it, [code 0, code 1]). Code 1 is all the
        next backbone step needs of this row besides the main token, so the rest of the tower can run alongside it."""

        c, model = self.c, self.model
        semantic = model.is_semantic(main)[:, None]
        extra = model.shared_extra(frame)
        offset = mx.array(caches[0].offset, dtype=mx.int32)
        h = hidden
        codes = [code0]
        for i in range(c.shared_layers + 1):
            h = self._tower_layer_step(i, h, codes, semantic, extra, offset, caches, sampling, keys)
        return h, codes

    def tower_tail(self, h, main, codes, caches, sampling: Sampling, keys) -> list[mx.array]:
        """The residual layers after the first: codes 2..9 appended to ``codes``."""

        semantic = self.model.is_semantic(main)[:, None]
        offset = mx.array(caches[-1].offset, dtype=mx.int32)
        for i in range(self.c.shared_layers + 1, len(self.tower)):
            h = self._tower_layer_step(i, h, codes, semantic, None, offset, caches, sampling, keys)
        return codes

    def _tower_layer_step(self, i, h, codes, semantic, extra, offset, caches, sampling, keys):
        c = self.c
        spec = self.tower[i]
        given = (codes[0] if i < c.shared_layers else codes[-1])[:, None]
        if spec["first"]:
            q, k, v, gate, mixed = spec["pre"](h, given, semantic, extra, offset)
        else:
            q, k, v, gate, mixed = spec["pre"](h, given, semantic, offset)
        out = self._attend(spec, q, k, v, caches[i])
        if i < c.shared_layers:
            return self._tower_post(i, None)(out, gate, mixed)
        h, code = self._tower_post(i, sampling)(out, gate, mixed, keys[i - c.shared_layers])
        codes.append(code)
        return h

    # main token ----------------------------------------------------------------------------------------------------
    def head(self, sampling: Sampling):
        """compiled (hidden [1, 1, D], RAS window [1, 16], key) -> (candidate logits, main, code 0, new window,
        code keys [9], next key, p(EOS)); greedy draws ignore the keys."""

        key = sampling
        if key not in self._heads:
            model, c = self.model, self.c

            def run(h, window, rng):
                keys = mx.random.split(rng, 12)
                logits = model.candidate_logits(h[:, -1])
                main, code0 = draw_main(logits, model._candidate_ids, window, sampling,
                                        None if sampling.greedy else keys[:2], c.semantic_begin, c.codebook_size,
                                        c.eos_token_id)
                new_window = mx.concatenate([window[:, 1:], main[:, None]], axis=1)
                temp = max(float(sampling.temperature), 1e-5)
                p_eos = mx.softmax(logits / temp, axis=-1)[:, model.eos_index]
                return logits, main, code0, new_window, keys[2:11], keys[11], p_eos

            self._heads[key] = mx.compile(run)
        return self._heads[key]
