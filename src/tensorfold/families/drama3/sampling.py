"""Drama 3 sampling: top-k / top-p draws and the semantic RAS re-draw, as the reference samples them.

A draw scales logits by 1 / temperature, drops ranks >= top_k and tokens whose inclusive cumulative
probability (over the whole distribution) exceeds top_p, always keeping rank 0, then samples the rest.
When a frame's semantic first draw repeats a main token of the last 16 rows, the frame takes a second
draw at temperature 1.0, top-k 30, top-p 0.95 (both draws always run). Greedy decoding takes argmax
everywhere, which leaves RAS without effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

RAS_WINDOW = 16
RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P = 1.0, 30, 0.95
MINIMUM_TEMPERATURE = 1e-5
EMPTY_SLOT = -1


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_k: int = 30
    top_p: float = 0.9
    greedy: bool = False


def draw(logits: mx.array, temperature: float, top_k: int, top_p: float, key: mx.array | None,
         greedy: bool = False) -> mx.array:
    """One token per row of float32 logits [B, V] (int32 [B])."""

    if greedy or key is None:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    scaled = logits / max(float(temperature), MINIMUM_TEMPERATURE)
    k = max(1, min(int(top_k), int(scaled.shape[-1])))
    top = mx.argpartition(-scaled, kth=k - 1, axis=-1)[..., :k]
    values = mx.take_along_axis(scaled, top, axis=-1)
    order = mx.argsort(-values, axis=-1)
    values = mx.take_along_axis(values, order, axis=-1)
    top = mx.take_along_axis(top, order, axis=-1)
    cumulative = mx.cumsum(mx.exp(values - mx.logsumexp(scaled, axis=-1, keepdims=True)), axis=-1)
    dropped = cumulative > top_p
    dropped = mx.concatenate([mx.zeros_like(dropped[..., :1]), dropped[..., 1:]], axis=-1)
    choice = mx.random.categorical(mx.where(dropped, -mx.inf, values), axis=-1, key=key)
    return mx.take_along_axis(top, choice[..., None], axis=-1)[..., 0].astype(mx.int32)


def draw_main(logits: mx.array, candidate_ids: mx.array, window: mx.array, s: Sampling, keys: mx.array | None,
              semantic_begin: int, codebook_size: int) -> tuple[mx.array, mx.array]:
    """The main token [B] (vocabulary ids) and code 0 [B] from candidate logits [B, C] and the RAS window [B, 16]."""

    first = candidate_ids[draw(logits, s.temperature, s.top_k, s.top_p, None if keys is None else keys[0], s.greedy)]
    if s.greedy or keys is None:
        main = first
    else:
        again = candidate_ids[draw(logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, keys[1])]
        repeat = (window == first[:, None]).any(axis=-1)
        semantic = (first >= semantic_begin) & (first < semantic_begin + codebook_size)
        main = mx.where(repeat & semantic, again, first)
    return main, mx.clip(main - semantic_begin, 0, codebook_size - 1)
