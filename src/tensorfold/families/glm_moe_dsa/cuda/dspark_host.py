"""DSpark's host side for full GLM-5.3 (dspark.py runs the block pass): the checkpoint's config, the tap plan, the
per-rank memory estimate, the settings every rank must share, and the sequential stage - the Markov chain over each
slot's merged candidates, the learned confidences and where a block is cut. NumPy only (no torch, no CUDA), so the
CPU tests run it as the engine does.

The chain, for drafts after the pending token (the anchor) at position p (``sample_from_anchor``: slot d of the block
pass predicts the token at p + 1 + d):

* prev_0 = anchor, prev_d = draft d - 1;
* slot d's score of candidate t = (base_d[t] + W2[t] . W1[prev_d]) / T - base the block pass's draft-head logit (each
  rank's top-K of its vocabulary share, merged), W2 W1 the vanilla Markov head (``MarkovHead.block_bias``); the pick is
  its argmax (sampled: plus the target's keyed Gumbel noise times ``noise``, among the candidates the target's own
  top_k / top_p / min_p would keep);
* confidence c_d = sigmoid(w_h . h_d + w_m . W1[prev_d] + b) (``ConfidenceHead`` over [hidden; Markov embedding], in
  float64 on the host after the hidden part's fp32 dot on the device) - trained on the per-slot acceptance rate, so
  prod_{j <= d} c_j estimates P(drafts 0..d all kept);
* the cut: ``cost`` - the k maximizing E[tokens | k drafts] / round_ms[k] (the startup's measured round times),
  ``confidence`` - stop once the survival product drops below the threshold (the first draft always kept), ``fixed`` -
  every slot up to the depth.

Drafts only propose: the target verifies every one and every emitted token is its own sample, so nothing here can
change a reply - only how many rows a round keeps. Every rank runs the same chain on the same gathered candidates and
the same confidence words, so every rank drafts alike (the verify windows must agree).

Adapted (Apache-2.0) from
* vllm-project/speculators @ 36a19ca: src/speculators/models/dspark/model_definitions.py (MarkovHead "vanilla",
  ConfidenceHead), dspark/core.py (prev-token alignment under sample_from_anchor, the confidence features),
  dspark/config.py and dflash/core.py (config fields, the anchored block);
* MiaAI-Lab's TensorFold port for full GLM-5.3 (patches 0114-glm-full-dspark, 0117-glm-full-dspark-sampling,
  0125-glm-full-dspark-host-markov, 0140's ``best_depth``): the host Markov chain over merged top-K candidates
  (vLLM's dspark top-k mode), the cost cut, the sampled-draft filter, the bf16 row reads through NumPy.
"""

from __future__ import annotations

import json
import math
import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

POLICIES = ("cost", "confidence", "fixed")
QUANTS = ("q4", "bf16")
TOP_K = 64                 # merged base candidates a slot (the Markov bias is applied over these only)
NOISE = 0.7                # the target's keyed noise weight in a sampled draft (DFlash2's)
ROW_COST = 0.06            # policy "cost" without measured round times: a draft row's cost / a one-row round


# -- the checkpoint ------------------------------------------------------------------------------------------------
@dataclass
class DSparkConfig:
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    layers: int
    eps: float
    theta: float
    window: int                    # context rows a block sees before its anchor (sliding_window); 0: unbounded
    block: int
    mask_id: int
    aux_ids: tuple[int, ...]       # aux_hidden_state_layer_ids: the INPUTS of those target layers
    vocab: int                     # draft vocabulary (== the verifier's: no d2t / t2d)
    markov_rank: int
    markov_type: str
    confidence: bool
    confidence_markov: bool
    causal: bool                   # within the block (sliding layers, sliding_window_non_causal false)
    crc: int = 0                   # config.json's CRC-32 (the ranks compare it)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def tap_layers(self) -> tuple[int, ...]:
        """The engine's tap points (``fused.Weights.tap_slot``: a layer's OUTPUT rows): aux id i is the input of target
        layer i (vLLM's aux hidden states, ``hidden_states + residual`` before layer i runs), i.e. layer i - 1's
        output."""
        return tuple(i - 1 for i in self.aux_ids)

    @classmethod
    def read(cls, draft_dir: str | Path) -> "DSparkConfig":
        text = (Path(draft_dir) / "config.json").read_bytes()
        return cls.parse(json.loads(text), zlib.crc32(text) & 0x7FFFFFFF)

    @classmethod
    def parse(cls, cfg: dict, crc: int = 0) -> "DSparkConfig":
        kind = cfg.get("speculators_model_type", (cfg.get("speculators_config") or {}).get("algorithm"))
        if kind != "dspark":
            raise ValueError(f"not a DSpark speculator (speculators_model_type {kind!r})")
        t = cfg["transformer_layer_config"]
        heads = int(t["num_attention_heads"])
        hd = int(t.get("head_dim") or int(t["hidden_size"]) // heads)
        types = list(t.get("layer_types") or [])
        sliding = t.get("sliding_window") is not None and bool(t.get("use_sliding_window", True))
        if types and any(x != "sliding_attention" for x in types):
            if any(x == "sliding_attention" for x in types):
                raise ValueError("mixed sliding / full DSpark draft layers are not ported")
            sliding = False
        mtype = str(cfg.get("markov_head_type", "vanilla"))
        rank = int(cfg.get("markov_rank", 0) or 0)
        if rank > 0 and mtype != "vanilla":
            raise ValueError(f"only the vanilla Markov head is ported, not {mtype!r}")
        if not cfg.get("sample_from_anchor", True):
            raise ValueError("only sample_from_anchor=true (slot 0 predicts the token after the anchor) is ported")
        if cfg.get("t2d") is not None or cfg.get("d2t") is not None:
            raise ValueError("a reduced draft vocabulary (t2d / d2t) is not ported")
        theta = (t.get("rope_parameters") or {}).get("rope_theta", t.get("rope_theta", 10000.0))
        if (t.get("rope_parameters") or {}).get("rope_type", "default") != "default":
            raise ValueError("only default RoPE is ported")
        conf = bool(cfg.get("enable_confidence_head", False))
        return cls(hidden=int(t["hidden_size"]), heads=heads, kv_heads=int(t["num_key_value_heads"]), head_dim=hd,
                   inter=int(t["intermediate_size"]), layers=int(t["num_hidden_layers"]), eps=float(t["rms_norm_eps"]),
                   theta=float(theta), window=int(t["sliding_window"]) if sliding else 0,
                   block=int(cfg["block_size"]), mask_id=int(cfg["mask_token_id"]),
                   aux_ids=tuple(int(i) for i in cfg["aux_hidden_state_layer_ids"]),
                   vocab=int(cfg.get("draft_vocab_size") or t["vocab_size"]), markov_rank=rank, markov_type=mtype,
                   confidence=conf, confidence_markov=conf and bool(cfg.get("confidence_head_with_markov", False)),
                   causal=sliding and not bool(cfg.get("sliding_window_non_causal", False)), crc=crc, raw=cfg)


def expected_tensors(c: DSparkConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor a DSpark checkpoint holds and its shape (no embed_tokens / lm_head: those are the verifier's)."""
    D, hd = c.hidden, c.head_dim
    out = {"fc.weight": (D, len(c.aux_ids) * D), "hidden_norm.weight": (D,), "norm.weight": (D,)}
    for i in range(c.layers):
        p = f"layers.{i}."
        out.update({p + "input_layernorm.weight": (D,), p + "post_attention_layernorm.weight": (D,),
                    p + "self_attn.q_proj.weight": (c.heads * hd, D),
                    p + "self_attn.k_proj.weight": (c.kv_heads * hd, D),
                    p + "self_attn.v_proj.weight": (c.kv_heads * hd, D),
                    p + "self_attn.o_proj.weight": (D, c.heads * hd),
                    p + "self_attn.q_norm.weight": (hd,), p + "self_attn.k_norm.weight": (hd,),
                    p + "mlp.gate_proj.weight": (c.inter, D), p + "mlp.up_proj.weight": (c.inter, D),
                    p + "mlp.down_proj.weight": (D, c.inter)})
    if c.markov_rank > 0:
        out["markov_head.markov_w1.weight"] = (c.vocab, c.markov_rank)
        out["markov_head.markov_w2.weight"] = (c.vocab, c.markov_rank)
    if c.confidence:
        out["confidence_head.proj.weight"] = (1, D + (c.markov_rank if c.confidence_markov else 0))
        out["confidence_head.proj.bias"] = (1,)
    return out


def check(c: DSparkConfig, world: int, *, hidden: int, vocab: int, layers: int) -> None:
    """Refuse a speculator this engine cannot run against this target at ``world`` ranks."""
    if c.hidden != hidden:
        raise ValueError(f"DSpark hidden size {c.hidden} != the target's {hidden}")
    if c.vocab != vocab:
        raise ValueError(f"DSpark draft vocabulary {c.vocab} != the target's {vocab} (a reduced one is not ported)")
    if c.heads != c.kv_heads:
        raise ValueError("the DSpark block attention kernel is multi-head (heads == kv_heads)")
    if c.heads % world or c.inter % world or (c.inter // world) % 64:
        raise ValueError(f"DSpark heads {c.heads} / MLP {c.inter} must split evenly over {world} ranks (64-wide)")
    if c.head_dim != 64:
        raise ValueError("the DSpark block attention kernel takes head_dim 64")
    if not 0 < c.block <= 16:
        raise ValueError(f"DSpark block {c.block}: 1 to 16 (one tile of the block attention)")
    if any(i < 1 or i > layers for i in c.aux_ids):
        raise ValueError(f"DSpark aux layers {c.aux_ids}: 1 to {layers} (inputs of target layers)")


# -- taps --------------------------------------------------------------------------------------------------------
def tap_plan(wanted: Sequence[Sequence[int] | None]) -> tuple[dict[int, int], list[tuple[int, ...] | None]]:
    """Each loaded drafter's tap layers (None: not loaded) -> (``fused.Weights.tap_slot``: layer -> column block of
    ``Buffers.taps``, each drafter's column blocks in its own order or None when they are the taps' own order). Slots
    are handed out in first-seen order, so the first drafter (DFlash2) keeps the layout it has alone; a layer two
    drafters share is tapped once."""
    slot: dict[int, int] = {}
    for layers in wanted:
        for i in layers or ():
            if int(i) < 0:
                raise ValueError(f"tap layer {i}: a drafter that reads the embeddings is not ported")
            slot.setdefault(int(i), len(slot))
    cols: list[tuple[int, ...] | None] = []
    for layers in wanted:
        if layers is None:
            cols.append(None)
            continue
        c = tuple(slot[int(i)] for i in layers)
        cols.append(None if c == tuple(range(len(slot))) else c)
    return slot, cols


def tap_runs(cols: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Column blocks ``cols`` as runs (first own block, first source block, width): (0, 1, 2, 7) -> [(0, 0, 3),
    (3, 7, 1)]."""
    runs: list[list[int]] = []
    for j, s in enumerate(cols):
        if runs and runs[-1][1] + runs[-1][2] == s:
            runs[-1][2] += 1
        else:
            runs.append([j, s, 1])
    return [tuple(r) for r in runs]


# -- memory ------------------------------------------------------------------------------------------------------
def q4_bytes(n: int, k: int) -> int:
    """``qmm.quantize4``'s bytes for an [n, k] matrix: 4-bit codes plus a bf16 scale and bias per 64 inputs."""
    return n * k // 2 + 2 * (n * (k // 64) * 2)


def memory(c: DSparkConfig, world: int, *, quant: str = "q4", ring: int = 4096) -> dict[str, int]:
    """Bytes the drafter holds on each rank: device weights (fc replicated; attention heads and MLP width split; the
    q4 kind keeps a separate [k | v] copy for context rows), its K/V ring and block buffers, and host (the replicated
    Markov tables, bf16). The verifier's embedding and head are shared, not counted."""
    D, hd = c.hidden, c.head_dim
    q, kv, m = c.heads // world * hd, c.kv_heads // world * hd, c.inter // world

    def mat(n: int, k: int) -> int:
        return q4_bytes(n, k) if quant == "q4" else n * k * 2

    per_layer = mat(q + 2 * kv, D) + mat(D, q) + mat(2 * m, D) + mat(D, m) + (mat(2 * kv, D) if quant == "q4" else 0)
    weights = mat(D, len(c.aux_ids) * D) + c.layers * per_layer + (2 + 2 * c.layers) * D * 2 + c.layers * 4 * hd \
        + (D * 4 if c.confidence else 0)
    kvring = 2 * c.layers * (c.kv_heads // world) * ring * hd * 2
    vocab_part = -(-c.vocab // world)
    fixed = 64 * len(c.aux_ids) * D * 2 + c.block * vocab_part * 4 + 4 * c.block * (D + m) * 4
    host = 2 * c.vocab * c.markov_rank * 2
    return {"weights": weights, "kv": kvring, "fixed": fixed, "device": weights + kvring + fixed, "host": host,
            "total": weights + kvring + fixed + host}


# -- settings ----------------------------------------------------------------------------------------------------
def settings(block: int, override: str | None = None) -> dict:
    """This rank's DSpark settings (env, every rank alike - the engine compares them at startup): TF_GLM53_DSPARK_DEPTH
    (most drafts a round, default the block: 8), _POLICY (cost: the round-time cut, the default; confidence: the
    survival-product threshold; fixed: always the depth), _CONFIDENCE (the threshold, 0.3), _TOPK (merged candidates a
    slot, 64), _NOISE (the keyed-noise weight of sampled drafts, 0.7), _FILTER (1: sampled drafts only among the
    target's kept candidates), _ROW_COST (policy cost without measured rounds: a draft row / a one-row round, 0.06).
    ``override``: a JSON object of the same keys (lower case, no prefix) read at run time (``DSPARK_CFG``)."""
    env = os.environ.get
    out = {"depth": int(env("TF_GLM53_DSPARK_DEPTH", "") or block),
           "policy": (env("TF_GLM53_DSPARK_POLICY", "") or "cost").strip().lower(),
           "confidence": float(env("TF_GLM53_DSPARK_CONFIDENCE", "") or 0.3),
           "top_k": int(env("TF_GLM53_DSPARK_TOPK", "") or TOP_K),
           "noise": float(env("TF_GLM53_DSPARK_NOISE", "") or NOISE),
           "filter": (env("TF_GLM53_DSPARK_FILTER", "") or "1") != "0",
           "row_cost": float(env("TF_GLM53_DSPARK_ROW_COST", "") or ROW_COST)}
    if override:
        try:
            got = json.loads(override)
        except ValueError:
            got = {}
        out.update({k: type(out[k])(v) for k, v in got.items() if k in out and k != "top_k"})
    out["depth"] = max(0, min(int(out["depth"]), block))
    if out["policy"] not in POLICIES:
        raise ValueError(f"TF_GLM53_DSPARK_POLICY={out['policy']!r}: one of {POLICIES}")
    if not 1 <= out["top_k"] <= 256:
        raise ValueError(f"TF_GLM53_DSPARK_TOPK={out['top_k']}: 1 to 256")
    return out


def quant() -> str:
    """TF_GLM53_DSPARK_QUANT: q4 (the default: 4-bit draft-only copies, as DFlash2's) or bf16 (the checkpoint's)."""
    kind = (os.environ.get("TF_GLM53_DSPARK_QUANT", "") or "q4").strip().lower()
    if kind not in QUANTS:
        raise ValueError(f"TF_GLM53_DSPARK_QUANT={kind!r}: one of {QUANTS}")
    return kind


def costs_env() -> tuple[float, ...] | None:
    """TF_GLM53_DSPARK_COSTS="ms0,ms1,...": a round's ms with 0, 1, ... drafts (instead of the startup measurement)."""
    v = os.environ.get("TF_GLM53_DSPARK_COSTS", "").strip()
    if not v:
        return None
    ms = tuple(float(x) for x in v.split(","))
    if len(ms) < 2 or any(m <= 0 for m in ms):
        raise ValueError(f"TF_GLM53_DSPARK_COSTS={v!r}: two or more positive ms values (0, 1, ... drafts)")
    return ms


def words(c: DSparkConfig | None, s: dict | None, q: str = "q4") -> list[int]:
    """The settings that change which windows and collectives a round runs, as int32 words for the startup check."""
    if c is None or s is None:
        return [0] * 10
    costs = zlib.crc32(os.environ.get("TF_GLM53_DSPARK_COSTS", "").strip().encode()) & 0x7FFFFFFF
    return [1, c.crc, s["depth"], POLICIES.index(s["policy"]), int(round(s["confidence"] * 1e6)), s["top_k"],
            int(round(s["noise"] * 1e6)), int(s["filter"]) | QUANTS.index(q) << 1, int(round(s["row_cost"] * 1e6)),
            costs]


# -- the sequential stage ----------------------------------------------------------------------------------------
def bf16_rows(bits: np.ndarray, idx) -> np.ndarray:
    """Rows of a bf16 matrix held as its raw int16 words -> float64: exactly bf16 -> double (bf16 is float32's top
    half), without torch's per-call dispatch (MiaAI-Lab 0125)."""
    return (bits[idx].astype(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def target_kept(score: np.ndarray, ids: np.ndarray, s) -> np.ndarray:
    """Which of a slot's candidates the target's sampling rule keeps, judged on the drafter's scores (already over the
    temperature), in ``choose_rows``' order (score descending, id ascending): top_k, the top_p prefix, min_p
    (MiaAI-Lab 0117). A draft the target's rule cannot pick is never kept, so it is not proposed."""
    n = score.shape[0]
    order = np.lexsort((ids, -score))
    k = max(1, min(int(s.top_k) if s.top_k else n, n))
    order = order[:k]
    sc = score[order].astype(np.float64)
    if 0.0 < s.top_p < 1.0:
        probs = np.exp(sc - sc.max())
        probs /= probs.sum()
        keep = int((np.cumsum(probs) < s.top_p).sum()) + 1
        order, sc = order[:keep], sc[:keep]
    if s.min_p > 0.0:
        order = order[sc >= sc[0] + s.min_log]
    kept = np.zeros(n, dtype=bool)
    kept[order] = True
    return kept


def best_depth(conf: Sequence[float], round_ms: Sequence[float]) -> int:
    """The k (0 .. len(conf)) maximizing E[tokens a round | k drafts] / round_ms[k]: E = 1 + sum_{j <= k} prod_{i <= j}
    conf_i (each conf the per-slot acceptance given the slots before; MiaAI-Lab's ``best_depth``). round_ms shorter
    than conf: the last value's slope continues."""
    q = np.asarray(conf, dtype=np.float64)
    alive = np.concatenate([[1.0], np.cumprod(q)])
    expect = np.cumsum(alive)
    ms = list(float(m) for m in round_ms[:len(q) + 1])
    while len(ms) < len(q) + 1:
        ms.append(ms[-1] + (ms[-1] - ms[-2] if len(ms) > 1 else ms[-1] * ROW_COST))
    return int(np.argmax(expect / np.asarray(ms)))


def linear_costs(depth: int, row_cost: float) -> tuple[float, ...]:
    """Relative round times without a measurement: 1 + row_cost x k for k = 0 .. depth."""
    return tuple(1.0 + row_cost * k for k in range(depth + 1))


class MarkovChain:
    """The sequential stage over host copies of the Markov tables (bf16 words) and the confidence head's Markov part.
    ``w1`` / ``w2``: [vocab, rank] int16 views of the bf16 tables (or None: no Markov head); ``conf_m`` [rank] float64
    or None; ``conf_b`` the head's bias; ``confident``: the checkpoint has a confidence head (else a slot's confidence
    is its pick's probability among the candidates)."""

    def __init__(self, w1: np.ndarray | None, w2: np.ndarray | None, conf_m: np.ndarray | None, conf_b: float,
                 confident: bool) -> None:
        self.w1, self.w2, self.conf_m, self.conf_b, self.confident = w1, w2, conf_m, float(conf_b), bool(confident)
        self.last: list[float] = []                   # the last walk's per-slot confidences

    def embed(self, token: int) -> np.ndarray | None:
        return bf16_rows(self.w1, int(token)) if self.w1 is not None else None

    def bias(self, cand: np.ndarray, emb: np.ndarray | None) -> np.ndarray:
        """W2[cand] . W1[prev] in float64: the Markov bias of the candidates only."""
        if emb is None:
            return np.zeros(cand.shape, dtype=np.float64)
        return bf16_rows(self.w2, np.asarray(cand, dtype=np.int64)) @ emb

    def walk(self, tokens: np.ndarray, values: np.ndarray, hconf: np.ndarray, anchor: int, first: int,
             sampling=None, *, policy: str = "cost", confidence: float = 0.3, round_ms: Sequence[float] | None = None,
             noise: float = NOISE, filtered: bool = True) -> list[int]:
        """Drafts for slots 0 .. len(tokens) - 1 (candidates ``tokens`` [depth, K] with base logits ``values``, merged
        by value then id; ``hconf`` [depth] the confidence head's hidden part w_h . h_d), left to right from
        ``anchor``; ``first``: the position of slot 0's token (keyed noise when sampling). Cut by ``policy``."""
        depth = tokens.shape[0]
        sampled = sampling is not None and sampling.temperature > 0
        temp = float(sampling.temperature) if sampled else 1.0
        gumbel = None
        if sampled and depth:
            from tensorfold.engine.exact_sampling import uniform_rows

            gumbel = -np.log(-np.log(uniform_rows(sampling.seed, first + np.arange(depth), tokens)))
        out: list[int] = []
        conf: list[float] = []
        prev, alive = int(anchor), 1.0
        for d in range(depth):
            emb = self.embed(prev)
            score = (values[d] + self.bias(tokens[d], emb)) / temp
            pick = score + noise * gumbel[d] if gumbel is not None else score
            if gumbel is not None and filtered:
                pick = np.where(target_kept(score, tokens[d], sampling), pick, -np.inf)
            j = int(np.argmax(pick))
            if self.confident:
                z = float(hconf[d]) + self.conf_b
                if self.conf_m is not None and emb is not None:
                    z += float(self.conf_m @ emb)
                c = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
            else:
                p = np.exp(score - score.max())
                c = float(p[j] / p.sum())
            if policy == "confidence" and confidence > 0:
                alive *= c
                if d > 0 and alive < confidence:
                    break
            conf.append(c)
            prev = int(tokens[d, j])
            out.append(prev)
        self.last = conf
        if policy == "cost" and out:
            out = out[:best_depth(conf, round_ms if round_ms is not None else linear_costs(depth, ROW_COST))]
        return out
