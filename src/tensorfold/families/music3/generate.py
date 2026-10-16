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

``dit_mode="sequential"`` is diffusers' MiniMax Music 3 pipeline instead: the same 200 / 100-frame windows, but one
after another, each conditioned on its own frames and holding its first 172 latent frames to the previous window's
(re-noised to each step's level, then restored), each decoded on its own and cut where the next one takes over.
A window needs only its own frames, so ``stream`` runs it while the AR is still drawing: the first audio comes after
200 frames and one window, and each further 100 frames bring about 4 s more.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from tensorfold.families.music3 import prompt as P
from tensorfold.families.music3.model import (AR, C0_CODES, CODEBOOK_SIZE, CODEBOOKS, DAV, DiT, frames_to_latents,
                                              parse_quant, quant_name, quantize_ar, quantize_dit)

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
    dit_mode: str = "averaged"       # "averaged" (ComfyUI, the default) or "sequential" (diffusers; what streams)
    overlap: bool = True             # streaming: a window's DiT steps go on their own GPU stream between AR frames
                                     # (the same audio, byte for byte; faster)


# diffusers' sequential windows: the first 172 latent frames of a window are held to the previous window's latents
# [L - 344, L - 172); a decoded window drops its first 86 latent frames (not the first window) and its last 258
# (not the last window)
SEQ_OVERLAP, SEQ_CROP_LEFT, SEQ_CROP_RIGHT = 172, 86, 344 - 86


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


class _no_stream:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def sequential_starts(frames: int) -> list[int]:
    """diffusers' window starts (in AR frames) for a song of ``frames`` frames: every 100, while 100 remain after."""

    return [0] if frames <= WINDOW_FRAMES else list(range(0, frames - HOP_FRAMES, HOP_FRAMES))


def sequential_crop(length: int, first: bool, last: bool) -> tuple[int, int]:
    """The latent span [a, b) of a decoded sequential window that goes into the song."""

    return (0 if first else SEQ_CROP_LEFT), (length if last else max(0, length - SEQ_CROP_RIGHT))


class LevelRule:
    """The reference's level rule (divide by 5 x std when that is above 1) applied as audio arrives: the divisor is
    the largest the rule has asked for so far (it never falls, so a streamed song never gets louder mid-way)."""

    def __init__(self) -> None:
        self.sum = self.sumsq = 0.0
        self.count = 0
        self.level = 1.0

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float64)
        self.sum += float(x.sum())
        self.sumsq += float((x * x).sum())
        self.count += x.size
        if self.count > 1:
            var = (self.sumsq - self.sum * self.sum / self.count) / (self.count - 1)
            self.level = max(self.level, math.sqrt(max(var, 0.0)) * 5.0)
        return (audio / self.level).astype(np.float32)


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
                 load_dit: bool = True, load_dav: bool = True, quant: str | dict | None = None) -> None:
        from tokenizers import Tokenizer

        started = time.perf_counter()
        self.quant = parse_quant(quant) if not isinstance(quant, dict) else quant
        self.quant_name = quant_name(self.quant)
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
            quantize_ar(self.ar, self.quant)
            mx.clear_cache()
        if load_dit:
            self.dit = DiT()
            weights = DiT.remap(mx.load(str(self.model_dir / DIFFUSION)))
            self.dit.load_weights([(k, v.astype(dit_dtype) if k != "inv_freq" else v.astype(mx.float32))
                                   for k, v in weights.items()], strict=True)
            self.dit.eval()
            mx.eval(self.dit.parameters())
            del weights
            quantize_dit(self.dit, self.quant)
            mx.clear_cache()
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
                  forced_codes: np.ndarray | None = None, trace: dict | None = None,
                  on_frame: Callable | None = None):
        """The AR stage: (frame states [T, 8 * 4096] bf16, stopped, stats).

        ``forced_codes`` [N, 8] (teacher forcing, for parity checks) replaces every drawn code with the given one;
        ``trace`` (a dict) collects the per-frame code-0 logits [2, 16385] and depth logits [7, 2, 1024].
        ``on_frame(kept)`` is called with the list of kept frame states after each new one (streaming); the time it
        takes is reported as ``hook_s`` and left out of ``ar_s`` / ``frames_per_s``."""

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
        hook_s = 0.0

        def hook():
            nonlocal hook_s
            if on_frame is not None:
                t = time.perf_counter()
                on_frame(kept)
                hook_s += time.perf_counter() - t

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
                    hook()
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
        ar_s = time.perf_counter() - started - hook_s
        stats = {"prompt_tokens": len(ids), "frames": len(kept), "stopped": stopped, "prefill_s": round(prefill_s, 3),
                 "ar_s": round(ar_s, 3), "first_frame_s": round(first_frame_s or ar_s, 3),
                 "frames_per_s": round(len(kept) / max(ar_s - prefill_s, 1e-9), 2)}
        if on_frame is not None:
            stats["hook_s"] = round(hook_s, 3)
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

    def denoise_window(self, states: mx.array, index: int, s: Settings, previous: tuple | None = None,
                       noise: mx.array | None = None, progress: Callable | None = None):
        """One sequential (diffusers) window: its frame states [F, 32768] -> (latents [L, 128] float32, the carry
        for the next window). ``previous`` is the carry of the window before: (latents, condition) over its
        [L - 344, L - 172) latent frames, which this window's first 172 frames are held to."""

        steps = self.window_steps(states, index, s, previous, noise)
        step = 0
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value
            step += 1
            if progress is not None:
                progress(step)

    def window_steps(self, states: mx.array, index: int, s: Settings, previous: tuple | None = None,
                     noise: mx.array | None = None, stream=None):
        """``denoise_window`` one Euler step at a time: a generator that yields after each step and returns
        (latents, carry). With ``stream`` (an MLX GPU stream) the steps are built on it and only submitted
        (``async_eval``), so they can run on the GPU while the caller goes on drawing AR frames."""

        dit = self.dit
        dtype = self.dit_dtype
        ctx = (lambda: mx.stream(stream)) if stream is not None else _no_stream
        finish = mx.async_eval if stream is not None else mx.eval
        with ctx():
            cond = dit.condition(states.astype(dtype))
            length = cond.shape[0]
            null = dit.null_condition(length, dtype)
            if noise is None:
                noise = mx.random.normal((length, 128), key=mx.random.key(derive_seed(s.seed, "dit", index)))
            noise = noise.astype(mx.float32)
            overlap = 0
            if previous is not None:
                prev_latent, prev_cond = previous
                overlap = min(prev_latent.shape[0], length)
                cond = mx.concatenate([prev_cond[:overlap].astype(cond.dtype), cond[overlap:]])
            x = noise
            tables = dit.rope_tables(length + 1)
            conds = mx.stack([cond, null])
        sigmas = sigma_schedule(s.steps)
        for step in range(s.steps):
            sigma, following = sigmas[step], sigmas[step + 1]
            t = 1.0 - sigma
            with ctx():
                if overlap:
                    held = (1.0 - (1.0 - 1e-6) * t) * noise[:overlap] + t * prev_latent[:overlap]
                    x = mx.concatenate([held, x[overlap:]])
                xd = x.astype(dtype)
                out = dit(mx.stack([xd, xd]), mx.full((2,), t, dtype=mx.float32), conds, tables).astype(mx.float32)
                velocity = out[1] + (out[0] - out[1]) * s.dit_cfg
                x = x + velocity * (following - sigma)
                finish(x)
            yield
        with ctx():
            if overlap:
                x = mx.concatenate([prev_latent[:overlap], x[overlap:]])
            a = max(0, length - 2 * SEQ_OVERLAP)
            b = max(a, length - SEQ_OVERLAP)
            carry = (x[a:b], cond[a:b])
        mx.eval(x, *carry)
        return x, carry

    # -- DAV -------------------------------------------------------------------------------------------------------
    def decode_raw(self, latents: mx.array, chunk: int = DAV_CHUNK, halo: int = DAV_HALO,
                   span: tuple[int, int] | None = None) -> np.ndarray:
        """Latents [L, 128] -> stereo [2, 512 (b - a)] float32 for the latent span [a, b) (all of it by default),
        decoded in chunks with ``halo`` latent frames of context on either side; no level rule."""

        length = latents.shape[0]
        lo, hi = span if span is not None else (0, length)
        pieces = []
        for start in range(lo, hi, chunk):
            end = min(start + chunk, hi)
            a, b = max(0, start - halo), min(length, end + halo)
            audio = self.dav(latents[a:b][None])[0]
            pieces.append(np.array(audio[:, (start - a) * 512:(end - a) * 512]))
        return np.concatenate(pieces, axis=1) if pieces else np.zeros((2, 0), dtype=np.float32)

    def decode(self, latents: mx.array, chunk: int = DAV_CHUNK, halo: int = DAV_HALO) -> np.ndarray:
        """Latents [L, 128] -> stereo [2, 512 L] float32, decoded in chunks, then the reference's level rule."""

        audio = self.decode_raw(latents, chunk, halo)
        level = max(1.0, float(np.std(audio.astype(np.float64), ddof=1)) * 5.0)
        return (audio / level).astype(np.float32)

    # -- all of it ---------------------------------------------------------------------------------------------------
    def generate(self, caption: str, lyrics: str, s: Settings, progress: Callable | None = None,
                 sink: Callable | None = None) -> Song:
        """A song. With ``sink`` it streams: sequential DiT windows run while the AR draws, and ``sink(audio)`` gets
        each piece of the song (float32 [2, n], level rule applied as it goes) as soon as it is decoded."""

        if sink is not None or s.dit_mode == "sequential":
            return self._generate_sequential(caption, lyrics, s, progress, sink)
        if s.dit_mode != "averaged":
            raise ValueError(f"dit_mode is \"averaged\" or \"sequential\", not {s.dit_mode!r}")
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
        stats = self._stats(ar_stats, dit_s, dav_s, total, audio, s, ar_peak)
        stats.update(latent_frames=int(latents.shape[0]), dit_mode="averaged", stream=False,
                     first_audio_s=round(total, 3))
        mx.clear_cache()
        return Song(audio=audio, frames=ar_stats["frames"], stopped=stopped, stats=stats)

    def _stats(self, ar_stats, dit_s, dav_s, total, audio, s, ar_peak) -> dict:
        seconds = audio.shape[1] / SAMPLE_RATE
        return {**ar_stats, "dit_s": round(dit_s, 3), "dav_s": round(dav_s, 3), "total_s": round(total, 3),
                "audio_s": round(seconds, 3), "rtf": round(total / max(seconds, 1e-9), 3),
                "ar_rtf": round(ar_stats["ar_s"] / max(seconds, 1e-9), 3),
                "dit_rtf": round(dit_s / max(seconds, 1e-9), 3), "dav_rtf": round(dav_s / max(seconds, 1e-9), 3),
                "steps": s.steps, "seed": s.seed, "quant": self.quant_name,
                "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "ar_peak_memory_gb": round(ar_peak / 1e9, 2)}

    def _generate_sequential(self, caption, lyrics, s: Settings, progress, sink) -> Song:
        mx.reset_peak_memory()
        started = time.perf_counter()
        song = SequentialSong(self, s, sink, progress, started)
        hook = song.ready if sink is not None else None
        states, stopped, ar_stats = self.ar_frames(caption, lyrics, s, progress, on_frame=hook)
        ar_peak = mx.get_peak_memory()
        song.finish(states)
        audio = song.audio()
        total = time.perf_counter() - started
        stats = self._stats(ar_stats, song.dit_s, song.dav_s, total, audio, s, ar_peak)
        stats.update(dit_mode="sequential", stream=sink is not None, windows=song.windows, overlap=song.overlap,
                     first_audio_s=round(song.first_audio_s if song.first_audio_s is not None else total, 3),
                     seams_s=[round(x / SAMPLE_RATE, 3) for x in song.seams])
        if song.emitted:
            # the earliest a player could start and never run dry: each piece must be there before the audio
            # before it has played out
            stats["no_stall_start_s"] = round(max(t - before / SAMPLE_RATE for t, before in song.emitted), 3)
        mx.clear_cache()
        return Song(audio=audio, frames=ar_stats["frames"], stopped=stopped, stats=stats)


class SequentialSong:
    """diffusers' sequential windows over a growing list of frame states. ``ready(kept)`` runs the next window once
    its 200 frames are there (streaming; with ``overlap``, one DiT step per AR frame, submitted on a second GPU
    stream so the two can share the GPU); ``finish(states)`` runs the rest when the AR is done. Each window's piece of
    the song is decoded at once; the last 258 latent frames of a window wait until it is known whether it is the
    last (they are the song's end) or the next window takes over from them."""

    def __init__(self, gen: Music3Generator, s: Settings, sink: Callable | None = None,
                 progress: Callable | None = None, started: float | None = None) -> None:
        self.gen, self.s, self.sink, self.progress = gen, s, sink, progress
        self.started = time.perf_counter() if started is None else started
        self.carry = None
        self.windows = 0
        self.held = None                         # (latents, from) of the newest window: its unsent tail
        self.pieces: list[np.ndarray] = []
        self.seams: list[int] = []               # sample positions where one window's audio meets the next's
        self.sent = 0
        self.level = LevelRule()
        self.dit_s = self.dav_s = 0.0
        self.first_audio_s = None
        self.emitted: list[tuple[float, int]] = []   # (seconds since start, samples sent before this piece)
        self.overlap = bool(s.overlap) and sink is not None
        self.dit_stream = None
        if self.overlap and mx.default_device() == mx.gpu:
            self.dit_stream = mx.new_stream(mx.gpu)
        self.job = None                          # [steps, window index, last, steps done] of the window in hand
        self.expected = max(1, len(sequential_starts(min(int(round(s.seconds * FRAME_RATE)), MAX_FRAMES))))

    def ready(self, kept: list) -> None:
        if self.job is not None:
            self._advance()
        start = (self.windows + (self.job is not None)) * HOP_FRAMES
        if len(kept) >= start + WINDOW_FRAMES:
            if self.overlap:
                if self.job is None:
                    self._start(kept, start, last=False)
                    self._advance()
            else:
                self._window(kept, start, last=False)

    def finish(self, states) -> None:
        while self.job is not None:
            self._advance()
        starts = sequential_starts(len(states))
        if self.windows > len(starts):
            raise RuntimeError("sequential windows ran past the song's end")
        rest = starts[self.windows:]
        for i, start in enumerate(rest):
            self._window(states, start, last=i == len(rest) - 1)
        if not rest and self.held is not None:             # the newest window was the last after all: its tail
            latents, a = self.held
            self._emit(latents, (a, latents.shape[0]), seam=False)
        self.held = None

    def _start(self, kept, start: int, last: bool) -> None:
        window = kept[start:min(start + WINDOW_FRAMES, len(kept))]
        states = mx.stack(window) if isinstance(window, list) else window
        index = self.windows
        steps = self.gen.window_steps(states, index, self.s, self.carry, stream=self.dit_stream)
        self.job = [steps, index, last, 0]

    def _advance(self) -> None:
        """One Euler step of the window in hand; the window is finished (decoded and sent) after its last."""

        steps, index, last, done = self.job
        t = time.perf_counter()
        try:
            next(steps)
            self.job[3] = done = done + 1
            self.dit_s += time.perf_counter() - t
            if self.progress is not None:
                self.progress("dit", index * self.s.steps + done, max(self.expected, index + 1) * self.s.steps)
            return
        except StopIteration as end:
            latents, self.carry = end.value
        self.dit_s += time.perf_counter() - t
        self.job = None
        a, b = sequential_crop(latents.shape[0], index == 0, last)
        self.held = None
        self.windows += 1
        self._emit(latents, (a, b), seam=index > 0)
        if not last:
            self.held = (latents, b)

    def _window(self, kept, start: int, last: bool) -> None:
        self._start(kept, start, last)
        while self.job is not None:
            self._advance()

    def _emit(self, latents, span, seam: bool) -> None:
        t = time.perf_counter()
        audio = self.gen.decode_raw(latents, span=span)
        self.dav_s += time.perf_counter() - t
        if seam:
            self.seams.append(self.sent)
        self.sent += audio.shape[1]
        if self.sink is not None:
            audio = self.level(audio)
            if self.first_audio_s is None and audio.shape[1]:
                self.first_audio_s = time.perf_counter() - self.started
            self.emitted.append((time.perf_counter() - self.started, self.sent - audio.shape[1]))
            self.sink(audio)
            if self.progress is not None:
                self.progress("stream", int(self.sent / SAMPLE_RATE), int(self.s.seconds))
        self.pieces.append(audio)

    def audio(self) -> np.ndarray:
        audio = np.concatenate(self.pieces, axis=1) if self.pieces else np.zeros((2, 0), dtype=np.float32)
        if self.sink is None:                    # all of it at once: the reference's rule over the whole song
            level = max(1.0, float(np.std(audio.astype(np.float64), ddof=1)) * 5.0) if audio.size > 1 else 1.0
            audio = (audio / level).astype(np.float32)
        return audio


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


STREAMING_SIZE = 0xFFFFFFFF


def wav_header(data_bytes: int | None = None, rate: int = SAMPLE_RATE, channels: int = 2) -> bytes:
    """A 44-byte 16-bit PCM WAV header; ``data_bytes`` None writes the "unknown length" sizes streams use (as
    Drama 3's stream does)."""

    import struct

    data = STREAMING_SIZE - 36 if data_bytes is None else data_bytes
    riff = STREAMING_SIZE if data_bytes is None else 36 + data_bytes
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate,
                                                                           rate * channels * 2, channels * 2, 16)
            + b"data" + struct.pack("<I", data))


def pcm16(audio: np.ndarray) -> bytes:
    """float32 [C, n] in [-1, 1] -> interleaved little-endian 16-bit PCM bytes."""

    audio = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2").T.tobytes()


class GrowingWav:
    """A stereo WAV written as audio arrives: readable while it grows (unknown-length header), its sizes set on
    ``close``. ``path`` "-" writes the stream to stdout (sizes left unknown)."""

    def __init__(self, path, rate: int = SAMPLE_RATE, channels: int = 2) -> None:
        import sys

        self.stdout = str(path) == "-"
        if self.stdout:
            self.f = sys.stdout.buffer
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "wb")
        self.rate, self.channels, self.bytes = rate, channels, 0
        self.f.write(wav_header(None, rate, channels))
        self.f.flush()

    def write(self, audio: np.ndarray) -> None:
        data = pcm16(audio)
        self.f.write(data)
        self.f.flush()
        self.bytes += len(data)

    def close(self) -> None:
        if self.stdout:
            self.f.flush()
            return
        self.f.seek(0)
        self.f.write(wav_header(self.bytes, self.rate, self.channels))
        self.f.close()


class RawPcm:
    """Raw interleaved 16-bit PCM appended as it comes (the ``pcm`` file a Drama 3 stream writes, stereo here)."""

    def __init__(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "wb")

    def write(self, audio: np.ndarray) -> None:
        self.f.write(pcm16(audio))
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def write_stats(path: str | Path, stats: dict) -> None:
    Path(path).write_text(json.dumps(stats, indent=1) + "\n")
