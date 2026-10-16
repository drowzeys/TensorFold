"""MiniMax Music 3 generation on MLX: AR frames, then the DiT over overlapping windows, then the DAV decoder.

Stages (timed separately):
  ar     prompt prefill (conditional + unconditional rows) and the frame loop: code 0 from the body with
         classifier-free guidance (candidates = the conditional row's top-k), codes 1..7 from the depth decoder,
         each guided the same way; a frame's 8 x 4096 states become the DiT's condition. The model ends a song by
         drawing the stop code; ``seconds`` caps it.
  dit    Euler flow matching (30 steps, guidance 1.7 against the all-zero condition) on the latent timeline
         (44100 / 512 a second), in windows of 200 AR frames that start every 100 and are averaged where they meet.
  dav    the decoder, in chunks with enough context on either side to equal one pass, then the reference's level
         rule (divide by 5 x std when that is above 1).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from tensorfold.families.music3 import prompt as P
from tensorfold.families.music3.model import (AR, C0_CODES, CODEBOOK_SIZE, CODEBOOKS, DAV, DiT, frames_to_latents)

FRAME_RATE = 25                      # AR frames a second
SAMPLE_RATE = 44100
MAX_FRAMES = 9000                    # 6 minutes
WINDOW_FRAMES, HOP_FRAMES = 200, 100
DAV_CHUNK, DAV_HALO = 512, 32        # latent frames a decoder chunk, and context on each side (receptive field ~10)
CACHE_LIMIT = 8 << 30

TEXT_ENCODER = "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors"
DIFFUSION = "diffusion_models/minimax_music3_dit_fp16.safetensors"
DECODER = "vae/minimax_music3_dav.safetensors"


@dataclass(frozen=True)
class Settings:
    seconds: float = 60.0
    seed: int = 0
    ar_cfg: float = 1.5
    top_k: int = 50
    steps: int = 30
    dit_cfg: float = 1.7
    window_batch: int = 8            # DiT windows run together in one forward (x2 for guidance)


@dataclass
class Song:
    audio: np.ndarray                                # [2, samples] float32, 44.1 kHz
    frames: int
    stopped: bool                                    # the model drew its stop code (else it hit the cap)
    stats: dict = field(default_factory=dict)


def derive_seed(seed: int, *parts) -> int:
    """A stage's seed from the song seed (the reference's derivation: blake2b-64, person "minimax-ttm")."""

    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def sigma_schedule(steps: int) -> list[float]:
    """ComfyUI's "simple" schedule over 1000 discrete flow sigmas (shift 1): 1.0 down to 0."""

    table = (np.arange(1, 1001, dtype=np.float32) / np.float32(1000)).tolist()
    stride = len(table) / steps
    return [table[-(1 + int(i * stride))] for i in range(steps)] + [0.0]


def windows(length: int) -> list[tuple[int, int]]:
    size, hop = frames_to_latents(WINDOW_FRAMES), frames_to_latents(HOP_FRAMES)
    if length <= size:
        return [(0, length)]
    spans, start = [], 0
    while True:
        end = min(start + size, length)
        spans.append((start, end))
        if end == length:
            return spans
        start += hop


class _Cache:
    """A key/value cache that grows by concatenation (the depth decoder holds at most 8 rows)."""

    def __init__(self) -> None:
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, k, v):
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = mx.concatenate([self.keys, k], axis=2)
            self.values = mx.concatenate([self.values, v], axis=2)
        self.offset += k.shape[2]
        return self.keys, self.values


def _topk_draw(values: mx.array, ranks: mx.array, key: mx.array) -> mx.array:
    """Draw from softmax(values) over the given candidates; returns the candidate ids ``ranks`` picked."""

    choice = mx.random.categorical(values, axis=-1, key=key)
    return mx.take_along_axis(ranks, choice[..., None], axis=-1)[..., 0]


def guided_draw(logits: mx.array, cfg: float, top_k: int, key: mx.array, restrict: bool) -> mx.array:
    """One code from [2, V] float32 logits (row 0 conditional, row 1 unconditional): guided = u + (c - u) * cfg,
    drawn among the top_k of the conditional row (``restrict``, code 0) or of the guided logits (codes 1..7)."""

    cond, uncond = logits[0:1], logits[1:2]
    guided = uncond + (cond - uncond) * cfg
    rank_source = cond if restrict else guided
    ranks = mx.argpartition(-rank_source, kth=top_k - 1, axis=-1)[..., :top_k]
    return _topk_draw(mx.take_along_axis(guided, ranks, axis=-1), ranks, key)[0]


class Music3Generator:
    def __init__(self, model_dir: str | Path, *, dit_dtype=mx.float16, load_ar: bool = True,
                 load_dit: bool = True, load_dav: bool = True) -> None:
        from tokenizers import Tokenizer

        started = time.perf_counter()
        self.model_dir = Path(model_dir).expanduser()
        mx.set_cache_limit(CACHE_LIMIT)
        self.ar = self.dit = self.dav = self.tokenizer = None
        if load_ar:
            weights = mx.load(str(self.model_dir / TEXT_ENCODER))
            raw = np.array(weights.pop("tokenizer_json"), dtype=np.uint8).tobytes().decode("utf-8")
            self.tokenizer = Tokenizer.from_str(raw)
            P.check_tokenizer(self.tokenizer)
            self.ar = AR()
            self.ar.load_weights(list(AR.remap(weights).items()), strict=True)
            self.ar.eval()
            mx.eval(self.ar.parameters())
            del weights
        if load_dit:
            self.dit = DiT()
            weights = DiT.remap(mx.load(str(self.model_dir / DIFFUSION)))
            self.dit.load_weights([(k, v.astype(dit_dtype) if k != "inv_freq" else v.astype(mx.float32))
                                   for k, v in weights.items()], strict=True)
            self.dit.eval()
            mx.eval(self.dit.parameters())
            del weights
        if load_dav:
            self.dav = DAV(mx.load(str(self.model_dir / DECODER)))
        self.dit_dtype = dit_dtype
        self.load_s = time.perf_counter() - started

    # -- AR --------------------------------------------------------------------------------------------------------
    def depth(self, hidden: mx.array, c0: mx.array, cfg: float, top_k: int, keys: mx.array,
              forced: np.ndarray | None = None, trace: list | None = None):
        """Codes 1..7 for the frame whose body state is hidden [2, D] and code 0 is c0 (scalar).
        Returns (codes [8] int32, the depth decoder's states of the conditional row [7, D]).
        ``forced`` [8] feeds given codes instead of drawn ones; ``trace`` collects each head's [2, 1024] logits."""

        ar = self.ar
        dec = ar.audio_decoder
        caches = [_Cache() for _ in dec.layers]
        c0_embed = ar.embed_tokens_audio(mx.broadcast_to(c0, (2,)))
        out = dec(dec.projection(mx.stack([hidden, c0_embed], axis=1)), 0, caches)
        codes, states = [c0], []
        for index in range(1, CODEBOOKS):
            states.append(out[0])
            logits = dec.audio_heads[index - 1](out).astype(mx.float32)
            if trace is not None:
                trace.append(logits)
            code = (guided_draw(logits, cfg, top_k, keys[index], restrict=False) if forced is None
                    else mx.array(int(forced[index]), dtype=mx.uint32))
            codes.append(code)
            if index < CODEBOOKS - 1:
                embed = ar.audio_extra_embedding(mx.broadcast_to(code + (index - 1) * CODEBOOK_SIZE, (2,)))
                out = dec(dec.projection(embed)[:, None], index + 1, caches)
        return mx.stack(codes).astype(mx.int32), mx.stack(states)

    def ar_frames(self, caption: str, lyrics: str, s: Settings, progress: Callable | None = None,
                  forced_codes: np.ndarray | None = None, trace: dict | None = None):
        """The AR stage: (frame states [T, 8 * 4096] bf16, stopped, stats).

        ``forced_codes`` [N, 8] (teacher forcing, for parity checks) replaces every drawn code with the given one;
        ``trace`` (a dict) collects the per-frame code-0 logits [2, 16385] and depth logits [7, 2, 1024]."""

        from mlx_lm.models.cache import KVCache

        ar = self.ar
        ids = P.encode(self.tokenizer, caption, lyrics)
        limit = min(int(round(s.seconds * FRAME_RATE)), MAX_FRAMES)
        if forced_codes is not None:
            limit = min(limit, len(forced_codes) - 1)
        started = time.perf_counter()
        caches = [KVCache() for _ in ar.layers.layers]
        rows = mx.array([ids, P.unconditional(ids)], dtype=mx.int32)
        hidden = ar.layers(ar.embed_tokens_prefill(rows), caches)[:, -1]
        mx.eval(hidden)
        prefill_s = time.perf_counter() - started
        key = mx.random.key(derive_seed(s.seed, "ar"))
        kept: list[mx.array] = []
        pending = None                   # (frame index, its drawn code-0 token, its states)
        stopped = False
        first_frame_s = None
        for frame in range(limit + 1):
            key, sub = mx.random.split(key)
            keys = mx.random.split(sub, CODEBOOKS)
            logits = ar.lm_head_pruned(hidden).astype(mx.float32)          # [2, 16385]; 0 = stop
            token = guided_draw(logits, s.ar_cfg, s.top_k, keys[0], restrict=True)
            forced = None
            depth_trace = None
            if forced_codes is not None:
                forced = forced_codes[frame]
                token = mx.array(int(forced[0]) + 1, dtype=mx.uint32)
                if trace is not None:
                    trace.setdefault("c0_logits", []).append(logits)
                    depth_trace = trace.setdefault("depth_logits", [])
            c0 = mx.maximum(token.astype(mx.int32) - 1, 0)
            codes, states = self.depth(hidden, c0, s.ar_cfg, s.top_k, keys, forced, depth_trace)
            frame_state = mx.concatenate([hidden[0:1], states]).reshape(-1)
            hidden = ar.layers(ar.frame_embedding(mx.broadcast_to(codes[None], (2, CODEBOOKS))), caches)[:, -1]
            mx.async_eval(hidden, token, frame_state)
            if pending is not None:                                           # the frame before: kept or stop?
                p_frame, p_token, p_state = pending
                if int(p_token.item()) == 0:
                    stopped = True
                    break
                if p_frame > 0:
                    kept.append(p_state)
                    if progress is not None and len(kept) % 25 == 0:
                        progress("ar", len(kept), limit)
                if first_frame_s is None:
                    first_frame_s = time.perf_counter() - started
                if len(kept) >= limit:
                    break
            pending = (frame, token, frame_state)
        else:
            p_frame, p_token, p_state = pending
            if int(p_token.item()) == 0:
                stopped = True
            elif p_frame > 0 and len(kept) < limit:
                kept.append(p_state)
        if not kept:
            raise ValueError("MiniMax Music 3 drew its stop code before any audio frame")
        states = mx.stack(kept)
        mx.eval(states)
        ar_s = time.perf_counter() - started
        stats = {"prompt_tokens": len(ids), "frames": len(kept), "stopped": stopped, "prefill_s": round(prefill_s, 3),
                 "ar_s": round(ar_s, 3), "first_frame_s": round(first_frame_s or ar_s, 3),
                 "frames_per_s": round(len(kept) / max(ar_s - prefill_s, 1e-9), 2)}
        return states, stopped, stats

    # -- DiT -------------------------------------------------------------------------------------------------------
    def denoise(self, states: mx.array, s: Settings, noise: mx.array | None = None,
                progress: Callable | None = None) -> mx.array:
        """Frame states [T, 32768] -> clean latents [L, 128] (float32)."""

        dit = self.dit
        dtype = self.dit_dtype
        cond = dit.condition(states.astype(dtype))
        length = cond.shape[0]
        null = dit.null_condition(length, dtype)
        if noise is None:
            noise = mx.random.normal((length, 128), key=mx.random.key(derive_seed(s.seed, "dit")))
        x = noise.astype(mx.float32)
        spans = windows(length)
        sigmas = sigma_schedule(s.steps)
        tables = {}
        for step in range(s.steps):
            sigma, following = sigmas[step], sigmas[step + 1]
            t = 1.0 - sigma
            total = mx.zeros_like(x)
            count = np.zeros((length, 1), dtype=np.float32)
            xd = x.astype(dtype)
            groups: dict[int, list[tuple[int, int]]] = {}
            for span in spans:
                groups.setdefault(span[1] - span[0], []).append(span)
            for size, members in groups.items():
                if size not in tables:
                    tables[size] = dit.rope_tables(size + 1)
                for g in range(0, len(members), s.window_batch):
                    batch = members[g:g + s.window_batch]
                    xs = mx.stack([xd[a:b] for a, b in batch])
                    cs = mx.stack([cond[a:b] for a, b in batch])
                    ns = mx.stack([null[a:b] for a, b in batch])
                    n = len(batch)
                    out = dit(mx.concatenate([xs, xs]), mx.full((2 * n,), t, dtype=mx.float32),
                              mx.concatenate([cs, ns]), tables[size]).astype(mx.float32)
                    guided = out[n:] + (out[:n] - out[n:]) * s.dit_cfg
                    for (a, b), v in zip(batch, guided):
                        total = total.at[a:b].add(v)
                        count[a:b] += 1
            velocity = total / mx.array(count)
            x = x + velocity * (following - sigma)
            mx.eval(x)
            if progress is not None:
                progress("dit", step + 1, s.steps)
        return x

    # -- DAV -------------------------------------------------------------------------------------------------------
    def decode(self, latents: mx.array, chunk: int = DAV_CHUNK, halo: int = DAV_HALO) -> np.ndarray:
        """Latents [L, 128] -> stereo [2, 512 L] float32, decoded in chunks, then the reference's level rule."""

        length = latents.shape[0]
        pieces = []
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            a, b = max(0, start - halo), min(length, end + halo)
            audio = self.dav(latents[a:b][None])[0]
            pieces.append(np.array(audio[:, (start - a) * 512:(end - a) * 512]))
        audio = np.concatenate(pieces, axis=1)
        level = max(1.0, float(np.std(audio.astype(np.float64), ddof=1)) * 5.0)
        return (audio / level).astype(np.float32)

    # -- all of it ---------------------------------------------------------------------------------------------------
    def generate(self, caption: str, lyrics: str, s: Settings, progress: Callable | None = None) -> Song:
        mx.reset_peak_memory()
        started = time.perf_counter()
        states, stopped, ar_stats = self.ar_frames(caption, lyrics, s, progress)
        ar_peak = mx.get_peak_memory()
        t = time.perf_counter()
        latents = self.denoise(states, s, progress=progress)
        dit_s = time.perf_counter() - t
        t = time.perf_counter()
        audio = self.decode(latents)
        dav_s = time.perf_counter() - t
        total = time.perf_counter() - started
        seconds = audio.shape[1] / SAMPLE_RATE
        stats = {**ar_stats, "dit_s": round(dit_s, 3), "dav_s": round(dav_s, 3), "total_s": round(total, 3),
                 "audio_s": round(seconds, 3), "rtf": round(total / max(seconds, 1e-9), 3),
                 "ar_rtf": round(ar_stats["ar_s"] / max(seconds, 1e-9), 3),
                 "dit_rtf": round(dit_s / max(seconds, 1e-9), 3), "dav_rtf": round(dav_s / max(seconds, 1e-9), 3),
                 "latent_frames": int(latents.shape[0]), "steps": s.steps, "seed": s.seed,
                 "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "ar_peak_memory_gb": round(ar_peak / 1e9, 2)}
        mx.clear_cache()
        return Song(audio=audio, frames=ar_stats["frames"], stopped=stopped, stats=stats)


def save_wav(path: str | Path, audio: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Stereo (or mono) float audio [C, N] to a 16-bit PCM WAV."""

    import wave

    audio = np.atleast_2d(audio)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2").T.copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[0])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def write_stats(path: str | Path, stats: dict) -> None:
    Path(path).write_text(json.dumps(stats, indent=1) + "\n")
