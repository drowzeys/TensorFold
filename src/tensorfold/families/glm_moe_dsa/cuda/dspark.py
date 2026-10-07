"""DSpark drafts for full GLM-5.3 (RedHatAI/GLM-5.3-speculator.dspark): a DFlash-style block pass over committed target
taps, then the host's sequential Markov chain, learned confidences and cut (dspark_host.py). A draft mode next to the
MTP head and DFlash2 (dflash.py): TF_GLM53_DSPARK=<dir> loads it, a request's "tf_mtp": "dspark" drafts with it.

The model (``config.json`` with a nested ``transformer_layer_config``; tensors in ``dspark_host.expected_tensors``):

* taps: the target residual at the INPUT of layers ``aux_hidden_state_layer_ids`` (2, 20, 39, 58, 75) - the engine's
  tap points after layers 1, 19, 38, 57, 74 (``DSparkConfig.tap_layers``);
* context row j: c_j = hidden_norm(fc(concat taps_j)); every draft layer's K_j = RoPE(k_norm(k_proj c_j), j),
  V_j = v_proj c_j (no input_layernorm on the context; all layers read the same c) - kept in a ring of RING slots;
* block at the committed length p: [anchor = the pending token, mask x (block - 1)] at positions p .. p + block - 1,
  embedded with the VERIFIER's embed_tokens; plain Qwen3 pre-norm layers (q_norm / k_norm, then neox RoPE, theta 8e6),
  causal inside the block, every slot sees the context rows [p - sliding_window, p) (the training mask's window, per
  block) and the block's slots up to its own;
* h = norm(x); base logits = the verifier's lm_head(h) for EVERY slot (``sample_from_anchor``: slot d predicts the
  token at p + 1 + d, so one pass gives ``block`` = 8 drafts and a verify window of up to 9 rows);
* the Markov bias, the confidences and the cut on the host (``dspark_host.MarkovChain``).

Tensor parallelism over the four ranks as our DFlash2 (glm5_next dflash2.Drafter): query / KV heads (64 / 64 -> 16 /
16 a rank) and MLP width (12288 -> 3072 a rank) split evenly; o_proj / down_proj row-parallel partials gathered and
summed in rank order (``_row``, the RoCE one-shot gather when it is up); fc, the norms, the confidence head's hidden
part replicated on the device; the Markov tables replicated on the host (bf16, 2 x 79 MB); each rank's top-K of its
vocabulary share of the verifier head - its 4-bit draft copy (headq, TF_GLM53_DRAFT_HEAD) - gathered and merged by
value then id (``dflash.merge_candidates``). Weights 4-bit (TF_GLM53_DSPARK_QUANT=q4, the default; bf16 optional).
Per rank (q4): ~265 MB of weights, a 50 MB ring, ~2 MB of buffers, 159 MB of host tables.

Exact: drafts only propose. The target verifies [pending, drafts] in one window whose rows have serial steps' bits
(fused.py) and every emitted token is the target's own pick (greedy, or the keyed exact sample), so drafted replies
equal serial ones whatever the drafts are; every rank holds the same gathered candidates and confidence words and runs
the same host chain, so every rank verifies the same window.

Adapted (Apache-2.0) from vllm-project/speculators @ 36a19ca (dflash/core.py and dflash/attention.py: the anchored
block forward and its mask; dflash/model_definitions.py: the Qwen3 DFlash layer) and from MiaAI-Lab's TensorFold port
for full GLM-5.3 (patch 0114-glm-full-dspark: ``_block_attn_kernel``, the loader's per-rank slices, the block pass's
candidates and hidden-part confidences in one pinned read).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl
from safetensors import safe_open

from tensorfold.families.glm5_next.cuda import glue, qmm
from tensorfold.families.glm5_next.cuda.dflash2 import NO_LIMIT, Drafter

from . import dspark_host as host
from . import fused, headq
from .dflash import RING, TapColumns, merge_candidates


@triton.jit
def _block_attn_kernel(Q, K, V, OUT, POS, W, scale, N: tl.constexpr, MP: tl.constexpr, NH: tl.constexpr,
                       HD: tl.constexpr, RING_: tl.constexpr, BK: tl.constexpr, CAUSAL: tl.constexpr):
    """One head (MHA): the block's N queries (padded to MP rows for tl.dot) attend the context rows [s - W, s) and the
    block's keys (causal: up to their own slot), keys at ring slot position % RING_, visiting [s - W, s + N) only
    (MiaAI-Lab 0114's kernel over our ring)."""
    h = tl.program_id(0)
    m = tl.arange(0, MP)
    valid = m < N
    d = tl.arange(0, HD)
    q = tl.load(Q + (h * N + m[:, None]) * HD + d[None, :], mask=valid[:, None], other=0.0)
    s = tl.load(POS).to(tl.int32)
    klen = s + N
    qpos = s + m
    lo = tl.maximum(s - W, 0) // BK * BK
    m_i = tl.full([MP], -1e30, tl.float32)
    l_i = tl.zeros([MP], tl.float32)
    acc = tl.zeros([MP, HD], tl.float32)
    for start in range(lo, klen, BK):
        kk = start + tl.arange(0, BK)
        kin = kk < klen
        slot = kk % RING_
        k = tl.load(K + (h * RING_ + slot[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        v = tl.load(V + (h * RING_ + slot[:, None]) * HD + d[None, :], mask=kin[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        ctx = (kk < s) & (kk >= s - W)
        blk = kk[None, :] >= s
        if CAUSAL:
            blk = blk & (kk[None, :] <= qpos[:, None])
        ok = kin[None, :] & (ctx[None, :] | blk)
        sc = tl.where(ok, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(OUT + m[:, None] * (NH * HD) + h * HD + d[None, :], out.to(tl.bfloat16), mask=valid[:, None])


@dataclass
class SparkLayer:
    in_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: qmm.Q4 | qmm.B16      # this rank's [q heads | k heads | v heads]
    kv: qmm.Q4 | qmm.B16       # this rank's [k heads | v heads] (context rows; bf16: a view of qkv's rows)
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    o: qmm.Q4 | qmm.B16        # [D, this rank's heads * head_dim]: a row-parallel partial
    gu: qmm.Q4 | qmm.B16       # this rank's [gate | up]
    down: qmm.Q4 | qmm.B16     # [D, this rank's MLP width]: a row-parallel partial


class SparkDrafter(TapColumns, Drafter):
    """DSpark on one sequence, the runner-facing API of ``dflash.GlmDrafter``: ``tap_layers``, ``add_taps``,
    ``propose``, ``context_end``, ``pos_dev``, ``reset``, ``capture``, ``window`` / ``kc`` / ``vc`` (prompt reuse's ring
    windows, ``prefixes.ring_window``), ``nbytes``, ``block``. Reuses the base Drafter's ``_rotary``, ``_prep``,
    ``_norm``, ``reset`` and ``capture`` (tap passes of 1..block rows and the block pass as CUDA graphs over static
    buffers)."""

    def __init__(self, draft_dir: str | Path, w: fused.Weights, capacity: int, *, settings: dict | None = None,
                 quant: str | None = None, tap_cols: tuple[int, ...] | None = None) -> None:
        path = Path(draft_dir)
        c = self.config = host.DSparkConfig.read(path)
        host.check(c, w.world, hidden=w.cfg.hidden_size, vocab=w.cfg.vocab_size,
                   layers=w.cfg.num_hidden_layers)
        self.settings = settings or host.settings(c.block)
        self.quant = quant or host.quant()
        self.w = self.fw = w
        self.dev = w.device
        self.rank, self.world = w.rank, w.world
        self.D, self.hd = c.hidden, c.head_dim
        self.heads = self.kvh = c.heads // self.world
        self.inter = c.inter // self.world
        q0, m0 = self.rank * self.heads, self.rank * self.inter
        self.eps = c.eps
        self.mask_id = c.mask_id
        self.block = c.block
        self.top_k = int(self.settings["top_k"])
        self.tap_layers = c.tap_layers
        self.tap_cols = tap_cols
        self.span = c.window if c.window > 0 else RING - c.block - 64      # context rows a block sees
        if self.span + c.block + 64 > RING:
            raise ValueError(f"DSpark window {self.span} + block + tap batch exceeds the {RING}-slot ring")
        self.window = self.span - 1                      # DFlash2's meaning (prefixes.ring_window): kept rows - 1
        self.causal = c.causal
        H, hd, D, dev = self.heads, self.hd, self.D, self.dev

        def dense(t: torch.Tensor):
            t = t.to(dev, torch.bfloat16).contiguous()       # q4: 1024-row chunks keep the fp32 staging small
            return qmm.quantize4(t, chunk=1024) if self.quant == "q4" else qmm.make_b16(t)

        def vec(t: torch.Tensor) -> torch.Tensor:
            return t.to(dev, torch.bfloat16).contiguous()

        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as f:
            want = host.expected_tensors(c)
            have = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
            bad = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
            if bad:
                raise ValueError(f"{path}: DSpark tensors missing or misshapen (have, want): {bad}")

            def rows(name: str, a: int, b: int) -> torch.Tensor:
                return f.get_slice(name)[a:b]

            def cols(name: str, a: int, b: int) -> torch.Tensor:
                return f.get_slice(name)[:, a:b]

            self.fc = dense(f.get_tensor("fc.weight"))
            self.hidden_norm = vec(f.get_tensor("hidden_norm.weight"))
            self.norm = vec(f.get_tensor("norm.weight"))
            self.layers: list[SparkLayer] = []
            for i in range(c.layers):
                p = f"layers.{i}."
                q = rows(p + "self_attn.q_proj.weight", q0 * hd, (q0 + H) * hd)
                k = rows(p + "self_attn.k_proj.weight", q0 * hd, (q0 + H) * hd)
                v = rows(p + "self_attn.v_proj.weight", q0 * hd, (q0 + H) * hd)
                qkv = dense(torch.cat((q, k, v)))
                kvw = qmm.B16(qkv.weight[H * hd:], 2 * H * hd, D) if isinstance(qkv, qmm.B16) \
                    else dense(torch.cat((k, v)))
                g = rows(p + "mlp.gate_proj.weight", m0, m0 + self.inter)
                u = rows(p + "mlp.up_proj.weight", m0, m0 + self.inter)
                self.layers.append(SparkLayer(
                    in_norm=vec(f.get_tensor(p + "input_layernorm.weight")),
                    post_norm=vec(f.get_tensor(p + "post_attention_layernorm.weight")),
                    qkv=qkv, kv=kvw,
                    q_norm=vec(f.get_tensor(p + "self_attn.q_norm.weight")),
                    k_norm=vec(f.get_tensor(p + "self_attn.k_norm.weight")),
                    o=dense(cols(p + "self_attn.o_proj.weight", q0 * hd, (q0 + H) * hd)),
                    gu=dense(torch.cat((g, u))),
                    down=dense(cols(p + "mlp.down_proj.weight", m0, m0 + self.inter))))
                del q, k, v, g, u
            # the sequential stage's tables on the host, every rank alike (bf16 words; rows read per candidate)
            self._w1 = self._w2 = None
            if c.markov_rank > 0:
                self._w1 = f.get_tensor("markov_head.markov_w1.weight").to(torch.bfloat16).contiguous()
                self._w2 = f.get_tensor("markov_head.markov_w2.weight").to(torch.bfloat16).contiguous()
            self.conf_h = None                           # device fp32 [D]: the confidence head's hidden part
            conf_m, conf_b = None, 0.0
            if c.confidence:
                cw = f.get_tensor("confidence_head.proj.weight").float()[0]
                self.conf_h = cw[:D].to(dev).contiguous()
                if c.confidence_markov:
                    conf_m = cw[D:].double().numpy().copy()
                conf_b = float(f.get_tensor("confidence_head.proj.bias").float()[0])
        self.chain = host.MarkovChain(None if self._w1 is None else self._w1.view(torch.int16).numpy(),
                                      None if self._w2 is None else self._w2.view(torch.int16).numpy(),
                                      conf_m, conf_b, c.confidence)
        torch.cuda.empty_cache()
        self.inv_freq = 1.0 / c.theta ** (torch.arange(hd // 2, device=dev, dtype=torch.float32) * 2 / hd)
        self.capacity = capacity                         # positions add_taps accepts (the ring holds the window)
        self.cap = capacity + self.block
        self.ring = RING
        self.kc = [torch.zeros((self.kvh, RING, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.vc = [torch.zeros((self.kvh, RING, hd), dtype=torch.bfloat16, device=dev) for _ in self.layers]
        self.pos_dev = torch.zeros((1,), dtype=torch.int64, device=dev)
        self.context_end = 0
        n = self.block
        self.ids = torch.full((n,), self.mask_id, dtype=torch.int32, device=dev)
        self.ids_host = torch.zeros((1,), dtype=torch.int32, pin_memory=True)
        self.ar = torch.arange(max(64, n), device=dev)
        self.tap_in = torch.zeros((64, len(self.tap_layers) * D), dtype=torch.bfloat16, device=dev)
        self.logits = torch.empty((n, w.vocab_part), dtype=torch.float32, device=dev)
        # a pass's candidates (every rank's [block, 2 top_k]: values, then ids as fp32 bits) and each slot's
        # confidence logit's hidden part [block], one device buffer read back by one pinned copy
        self.cand_n = self.world * n * 2 * self.top_k
        self.cand = torch.zeros((self.cand_n + n,), dtype=torch.float32, device=dev)
        self.cand_host = torch.zeros(self.cand.shape, dtype=torch.float32, pin_memory=True)
        self.cand_ready = torch.cuda.Event()
        self.pool = None
        self.block_graph = None
        self.tap_graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def nbytes(self) -> int:
        """Device bytes: weights, the ring (host tables: ``host_bytes``)."""
        total = self.fc.nbytes() + sum(q.nbytes() for L in self.layers for q in (L.qkv, L.o, L.gu, L.down))
        if self.quant == "q4":
            total += sum(L.kv.nbytes() for L in self.layers)
        return total + 2 * sum(t.numel() * 2 for t in self.kc)

    def host_bytes(self) -> int:
        return sum(t.numel() * 2 for t in (self._w1, self._w2) if t is not None)

    # -- pieces -------------------------------------------------------------------------------------------------
    def _row(self, x: torch.Tensor, q, xs: torch.Tensor | None = None) -> torch.Tensor:
        """A projection whose input is split over the ranks: fp32 partials gathered (RoCE when up), added in rank
        order, then bf16 - every rank gets the same bits."""
        part = qmm.matmul(x.contiguous(), q, xs, f32=True)
        if self.world == 1:
            return part.to(torch.bfloat16)
        g = fused.small_gather(self.fw, part.reshape(-1)).view(self.world, *part.shape)
        acc = g[0].clone()
        for i in range(1, self.world):
            acc += g[i]
        return acc.to(torch.bfloat16)

    def _attend(self, i: int, q: torch.Tensor) -> torch.Tensor:
        rows = q.shape[1]
        out = torch.empty((rows, self.heads * self.hd), dtype=torch.bfloat16, device=self.dev)
        _block_attn_kernel[(self.heads,)](q, self.kc[i], self.vc[i], out, self.pos_dev, self.span, self.hd ** -0.5,
                                          N=rows, MP=max(16, triton.next_power_of_2(rows)), NH=self.heads, HD=self.hd,
                                          RING_=RING, BK=64, CAUSAL=self.causal, num_warps=4)
        return out

    def _layer(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               slots: torch.Tensor) -> torch.Tensor:
        L = self.layers[i]
        rows = x.shape[0]
        normed, xs = self._norm(x, L.in_norm)
        q, k, v = self._prep(qmm.matmul(normed, L.qkv, xs), L, cos, sin, self.heads)
        self.kc[i].index_copy_(1, slots, k)
        self.vc[i].index_copy_(1, slots, v)
        x = x + self._row(self._attend(i, q), L.o)
        normed, xs = self._norm(x, L.post_norm)
        gu = qmm.matmul(normed, L.gu, xs)
        act = torch.empty((rows, self.inter), dtype=torch.bfloat16, device=self.dev)
        axs = torch.empty((rows, self.inter // 64), dtype=torch.float32, device=self.dev)
        glue.swiglu(gu, act, axs, NO_LIMIT)
        return x + self._row(act, L.down, axs)

    # -- the context --------------------------------------------------------------------------------------------
    def _taps_compute(self, n: int) -> None:
        ctx, _ = self._norm(qmm.matmul(self.tap_in[:n], self.fc), self.hidden_norm)
        cos, sin = self._rotary(n)
        slots = (self.pos_dev + self.ar[:n]) % RING
        for i, L in enumerate(self.layers):
            _, k, v = self._prep(qmm.matmul(ctx, L.kv), L, cos, sin, 0)
            self.kc[i].index_copy_(1, slots, k)
            self.vc[i].index_copy_(1, slots, v)
        self.pos_dev += n

    # -- the block ----------------------------------------------------------------------------------------------
    def _block_compute(self) -> None:
        """[pending, mask x (block - 1)] at the committed length -> each rank's top-K of its head share for every slot
        (gathered into ``cand``) and each slot's confidence logit's hidden part (behind them). Static buffers only:
        the block graph replays it."""
        n = self.block
        x = torch.empty((n, self.D), dtype=torch.bfloat16, device=self.dev)
        glue.embed(self.ids, self.fw.embed, self.D, 1, x)
        cos, sin = self._rotary(n)
        slots = (self.pos_dev + self.ar[:n]) % RING          # past the context: the next update overwrites them
        for i in range(len(self.layers)):
            x = self._layer(i, x, cos, sin, slots)
        h, hs = self._norm(x, self.norm)
        headq.logits(h, self.fw.draft_lm_head, self.logits, hs)
        vals, local = torch.topk(self.logits, self.top_k, dim=-1)
        gids = (local + self.fw.vocab_off).to(torch.int32)
        packed = torch.cat([vals, gids.view(torch.float32)], dim=1).contiguous()
        fused.small_gather(self.fw, packed.view(-1), self.cand[:self.cand_n])
        if self.conf_h is not None:
            self.cand[self.cand_n:].copy_(h.float() @ self.conf_h)
        else:
            self.cand[self.cand_n:].zero_()

    @torch.no_grad()
    def candidates(self, pending: int, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merged candidate ids [depth, top_k] of the slots after ``pending``, their base logits, and each slot's
        confidence logit's hidden part [depth]: one block pass (graph), one pinned read."""
        self.ids_host[0] = pending
        self.ids[:1].copy_(self.ids_host, non_blocking=True)
        if self.block_graph is not None:
            self.block_graph.replay()
        else:
            self._block_compute()
        self.cand_host.copy_(self.cand, non_blocking=True)
        self.cand_ready.record()
        self.cand_ready.synchronize()
        g = self.cand_host[:self.cand_n].view(self.world, self.block, 2 * self.top_k)[:, :depth]
        return merge_candidates(g, self.cand_host[self.cand_n:self.cand_n + depth], self.top_k)

    def propose(self, pending: int, depth: int, sampling=None, confidence: float | None = None, *,
                policy: str | None = None, round_ms=None) -> list[int]:
        """Up to ``depth`` (<= block) drafts for the positions after the pending token (at context_end), cut by
        ``policy`` (default: the settings'): ``cost`` at ``best_depth`` of the learned confidences over ``round_ms``
        (ms of a round with 0, 1, .. drafts; None: the linear TF_GLM53_DSPARK_ROW_COST model), ``confidence`` below the
        survival-product ``confidence``, ``fixed`` never."""
        s = self.settings
        depth = min(int(depth), self.block)
        if depth < 1 or self.context_end == 0:
            return []
        tokens, values, hconf = self.candidates(pending, depth)
        pol = policy or s["policy"]
        if pol == "cost" and round_ms is None:
            round_ms = host.linear_costs(depth, s["row_cost"])
        return self.chain.walk(tokens, values, hconf, pending, self.context_end + 1, sampling, policy=pol,
                               confidence=s["confidence"] if confidence is None else confidence, round_ms=round_ms,
                               noise=s["noise"], filtered=s["filter"])

    @torch.no_grad()
    def block_ms(self, reps: int = 7) -> float:
        """The block pass's ms (fastest of ``reps`` graph replays; every rank together: it gathers). Startup only."""
        import time

        best = float("inf")
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if self.block_graph is not None:
                self.block_graph.replay()
            else:
                self._block_compute()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        return 1e3 * best
