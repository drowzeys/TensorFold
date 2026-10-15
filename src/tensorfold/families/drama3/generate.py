"""Drama 3 generation on MLX: prompt prefill, the frame loop, long-text chunking and voice references."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from tensorfold.families.drama3.model import Drama3
from tensorfold.families.drama3.prompt import WHITE_SPACE, Reference, build_rows
from tensorfold.families.drama3.sampling import EMPTY_SLOT, RAS_WINDOW, Sampling, draw, draw_main

PREFILL_CHUNK = 1024
FRAME_RATE = 44100 / 2048           # ~21.53 frames a second


@dataclass
class Result:
    frames: np.ndarray                     # [T, 11] generated frames, EOS frame dropped
    prompt_rows: int
    finish: str                            # "stop" (EOS) or "length"
    prefill_s: float
    decode_s: float
    first_frame_s: float
    stats: dict = field(default_factory=dict)

    @property
    def codes(self) -> np.ndarray:
        return self.frames[:, 1:].T


CACHE_LIMIT = 8 << 30               # MLX keeps at most this much freed memory for reuse
RETRIES = 2                          # a chunk that runs into its frame cap is drawn again with a new seed


def spoken_chars(text: str) -> int:
    """Characters that are read out: speaker tags and [directions] excluded."""

    return len(re.sub(r"<\|speaker:\d+\|>|\[[^\]]*\]", "", text).strip(WHITE_SPACE))


def default_frame_cap(text: str) -> int:
    """A runaway guard: about 4 frames a spoken character (natural speech takes 1-2.5) plus 5 s."""

    return int(4 * spoken_chars(text) + 5 * FRAME_RATE)


def usable_as_voice(result: "Result", text: str) -> bool:
    """Whether a chunk's own output may serve as the voice reference of the chunks after it."""

    frames = len(result.frames)
    per_char = frames / max(1, spoken_chars(text))
    return result.finish == "stop" and 0 < frames <= 25 * FRAME_RATE and 0.6 <= per_char <= 2.6


class Drama3Generator:
    def __init__(self, model_dir: str | Path, *, load_codec: bool = True) -> None:
        from tokenizers import Tokenizer

        started = time.time()
        self.model_dir = Path(model_dir).expanduser()
        mx.set_cache_limit(CACHE_LIMIT)
        self.model = Drama3.load(self.model_dir)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.codec = None
        if load_codec:
            from tensorfold.families.drama3.codec import Codec

            self.codec = Codec(self.model_dir / self.model.config.tensorfold.get("codec", "codec.safetensors"))
        self.load_s = time.time() - started

    # -- one prompt ---------------------------------------------------------------------------------------------
    def rows(self, text: str, references: list[Reference]) -> np.ndarray:
        c = self.model.config
        return build_rows(text, references, self.tokenizer, codebooks=c.codebooks, codebook_size=c.codebook_size)

    def generate_rows(self, rows: np.ndarray, sampling: Sampling, *, seed: int = 0, max_frames: int = 2000,
                      on_frame: Callable[[np.ndarray], None] | None = None, trace: dict | None = None) -> Result:
        """Frames after prompt ``rows`` [N, 11] until EOS or ``max_frames``.

        ``trace`` (a dict) collects the prompt's final hidden states and per-step candidate logits for parity checks.
        """

        model, c = self.model, self.model.config
        bcache, tcaches = model.make_cache()
        frames = mx.array(rows.astype(np.int32))[None]                 # [1, N, 11]
        n = int(rows.shape[0])
        started = time.perf_counter()
        hidden = []
        for s in range(0, n, PREFILL_CHUNK):
            chunk = frames[:, s:s + PREFILL_CHUNK]
            h = model.backbone_hidden(chunk, bcache)
            hidden.append(h)
            # tower rows of this chunk, except the prompt's last row (the first step runs it)
            e = min(s + PREFILL_CHUNK, n - 1)
            if e > s:
                model.tower_prefill(h[:, :e - s], chunk[:, :e - s], frames[:, s + 1:e + 1], tcaches)
            mx.eval(h, *[t.keys for t in tcaches if t.keys is not None])
        prefill_s = time.perf_counter() - started
        last = hidden[-1][:, -1:]
        if trace is not None:
            trace["prompt_hidden"] = mx.concatenate(hidden, axis=1)[0]
            trace["logits"] = []
        window = np.full((1, RAS_WINDOW), EMPTY_SLOT, dtype=np.int32)
        tail = rows[-RAS_WINDOW:, 0]
        window[0, RAS_WINDOW - len(tail):] = tail
        window = mx.array(window)
        key = mx.random.key(int(seed))

        def step(h: mx.array, frame_in: mx.array, window: mx.array, key: mx.array):
            keys = None if sampling.greedy else mx.random.split(key, 12)
            logits = model.candidate_logits(h[:, -1])
            main, code0 = draw_main(logits, model._candidate_ids, window, sampling,
                                    None if keys is None else keys[:2], c.semantic_begin, c.codebook_size)

            def draw_code(lg, r):
                return draw(lg, sampling.temperature, sampling.top_k, sampling.top_p,
                            None if keys is None else keys[2 + r], sampling.greedy)

            codes = model.tower_step(h, frame_in, main, code0, tcaches, draw_code)
            frame = mx.stack([main, *codes], axis=-1)[:, None]           # [1, 1, 11]
            new_window = mx.concatenate([window[:, 1:], main[:, None]], axis=1)
            next_key = keys[11] if keys is not None else key
            return frame, new_window, next_key, logits

        frame, window, key, logits = step(last, frames[:, -1:], window, key)
        mx.async_eval(frame, window)
        out: list[np.ndarray] = []
        finish = "length"
        first_frame_s = 0.0
        decode_started = time.perf_counter()
        while True:
            pending = None
            if len(out) + 1 < max_frames:
                h = model.backbone_hidden(frame, bcache)
                pending = step(h, frame, window, key)
                mx.async_eval(pending[0], pending[1])
            if trace is not None:
                trace["logits"].append(np.array(logits[0]))
            value = np.array(frame[0, 0])
            if not out:
                first_frame_s = time.perf_counter() - started
            if int(value[0]) == c.eos_token_id:
                finish = "stop"
                break
            out.append(value)
            if on_frame is not None:
                on_frame(value)
            if pending is None:
                break
            frame, window, key, logits = pending
        decode_s = time.perf_counter() - decode_started
        result = np.stack(out).astype(np.int64) if out else np.zeros((0, c.codebooks + 1), dtype=np.int64)
        return Result(frames=result, prompt_rows=n, finish=finish, prefill_s=prefill_s, decode_s=decode_s,
                      first_frame_s=first_frame_s)

    # -- text in, audio out -------------------------------------------------------------------------------------
    def synthesize(self, text: str, *, references: list[Reference] | None = None, sampling: Sampling = Sampling(),
                   seed: int = 0, max_chars: int = 300, max_frames: int | None = None,
                   log: Callable[[str], None] = print) -> tuple[np.ndarray, list[Result]]:
        """Audio (float32, 44.1 kHz) for ``text``; long text goes in sentence-aligned chunks.

        With no reference given, later chunks keep the first chunk's voice by taking its text and frames as their
        reference, when that output looks like a clean reading (it stopped, is short, has a plausible length).
        A chunk that runs into its frame cap is drawn again with another seed (greedy runs are not).
        """

        references = list(references or [])
        chunks = split_text(text, max_chars)
        results: list[Result] = []
        audio: list[np.ndarray] = []
        voice = list(references)
        for i, chunk in enumerate(chunks):
            rows = self.rows(chunk, voice)
            cap = max_frames or default_frame_cap(chunk)
            for attempt in range(1 + (0 if sampling.greedy else RETRIES)):
                result = self.generate_rows(rows, sampling, seed=seed + i + 1000 * attempt, max_frames=cap)
                if result.finish == "stop":
                    break
                log(f"[drama3] chunk {i + 1} ran to its {cap}-frame cap; drawing again")
            results.append(result)
            frames = len(result.frames)
            log(f"[drama3] chunk {i + 1}/{len(chunks)}: {result.prompt_rows} prompt rows, prefill "
                f"{result.prefill_s:.2f}s; {frames} frames ({frames / FRAME_RATE:.1f}s audio) in "
                f"{result.decode_s:.2f}s = {frames / max(result.decode_s, 1e-9):.1f} frames/s; {result.finish}")
            if frames and self.codec is not None:
                audio.append(np.array(self.codec.decode(result.codes)))
                mx.clear_cache()
            if not references and len(voice) == 0 and i + 1 < len(chunks) and usable_as_voice(result, chunk):
                voice = [Reference(transcript=chunk, codes=result.codes)]
        silence = np.zeros(int(0.15 * 44100), dtype=np.float32)
        joined = []
        for i, part in enumerate(audio):
            joined += [part] if i == 0 else [silence, part]
        return (np.concatenate(joined) if joined else np.zeros(0, dtype=np.float32)), results


SENTENCE_END = re.compile(r"(?<=[.!?。！？…])[" + re.escape(WHITE_SPACE) + r"]+|\n{2,}")
SPEAKER_START = re.compile(r"(?=<\|speaker:\d+\|>)")


def split_text(text: str, max_chars: int) -> list[str]:
    """Sentence-aligned chunks of at most ``max_chars`` (a longer sentence stays whole). Speaker tags carry over:
    a chunk that does not begin with one gets the tag in force where it starts."""

    text = text.strip(WHITE_SPACE)
    if len(text) <= max_chars:
        return [text]
    pieces = [p for p in SENTENCE_END.split(text) if p and p.strip(WHITE_SPACE)]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    tagged, speaker = [], None
    for chunk in chunks:
        lead = re.match(r"\s*<\|speaker:(\d+)\|>", chunk)
        if lead is None and speaker is not None:
            chunk = f"<|speaker:{speaker}|> {chunk}"
        tags = re.findall(r"<\|speaker:(\d+)\|>", chunk)
        if tags:
            speaker = tags[-1]
        tagged.append(chunk)
    return tagged
