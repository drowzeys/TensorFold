"""Drama 3 generation on MLX: prompt prefill, the frame loop, long-text chunking and voice references."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mlx.core as mx
import numpy as np

from tensorfold.families.drama3.model import Drama3
from tensorfold.families.drama3.prompt import WHITE_SPACE, Reference, build_rows
from tensorfold.families.drama3.sampling import EMPTY_SLOT, RAS_WINDOW, Sampling, draw, draw_main
from tensorfold.families.drama3.verify import trim_take

if TYPE_CHECKING:
    from tensorfold.families.drama3.verify import Policy, Transcriber

PREFILL_CHUNK = 1024
FRAME_RATE = 44100 / 2048           # ~21.53 frames a second


@dataclass
class Result:
    frames: np.ndarray                     # [T, 11] generated frames, EOS frame dropped
    prompt_rows: int
    finish: str                            # "stop" (EOS), "stop_rule" (StopRule) or "length" (frame cap)
    prefill_s: float
    decode_s: float
    first_frame_s: float
    stats: dict = field(default_factory=dict)
    eos_probs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))   # p(EOS) each step

    @property
    def codes(self) -> np.ndarray:
        return self.frames[:, 1:].T


CACHE_LIMIT = 8 << 30               # MLX keeps at most this much freed memory for reuse
RETRIES = 2                          # a chunk that runs into its frame cap is drawn again with a new seed
VERIFY_RETRIES = 2                   # with the end-of-text check, a take with missing words is drawn again


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
    clean = result.finish in ("stop", "stop_rule") or bool(result.stats.get("verified"))
    return clean and 0 < frames <= 25 * FRAME_RATE and 0.6 <= per_char <= 2.6


MIN_FRAMES_PER_CHAR = 0.55          # no reading is faster than this (natural speech takes ~1-2.5)


@dataclass
class StopRule:
    """The model-side end-of-text signal: a missed stop.

    The model ends a chunk by sampling EOS. Mid-text its p(EOS) stays far below 0.1 (in 152 measured chunks the
    only higher steps came within a word or a closing laugh of the end); near the end it rises, and the step that
    draws EOS usually has p(EOS) near 1. A runaway is a take that reached the end, put real weight on EOS, drew a
    sound instead and went on. So the first step at or past ``min_frames`` with p(EOS) >= ``candidate_eos`` marks a
    *candidate end* (``hold_from``, a frame index); if the take is still going ``max_overrun`` frames later, the
    chunk ends (finish "stop_rule"). In the measured chunks a take that passed a candidate drew EOS at most 39
    frames later (median 1). The frames after a candidate are the suspect ones: streaming holds them back, and
    without speech recognition a stopped take is cut ``pad_frames`` after the candidate (room for a last word the
    model finished after it). Length alone is no signal: short sentences with pauses or digits read out run to
    3-7 frames a character.
    """

    candidate_eos: float = 0.1
    max_overrun: int = 64               # ~3 s
    min_frames: int = 0
    pad_frames: int = 14                # ~0.65 s
    hold_from: int | None = None        # the candidate end, once seen

    @classmethod
    def for_text(cls, text: str, **kwargs) -> "StopRule":
        return cls(min_frames=int(MIN_FRAMES_PER_CHAR * spoken_chars(text)), **kwargs)

    def __call__(self, frames: int, p_eos) -> bool:
        """After ``frames`` frames, the last of which was drawn at a step with ``p_eos``: stop now?"""

        if self.hold_from is None and frames >= self.min_frames:
            if float(np.asarray(p_eos).reshape(-1)[0]) >= self.candidate_eos:
                self.hold_from = frames - 1
        return self.hold_from is not None and frames - self.hold_from > self.max_overrun

    @property
    def cut_frames(self) -> int | None:
        return None if self.hold_from is None else self.hold_from + self.pad_frames


class Drama3Generator:
    def __init__(self, model_dir: str | Path, *, load_codec: bool = True) -> None:
        from tokenizers import Tokenizer

        started = time.time()
        self.model_dir = Path(model_dir).expanduser()
        mx.set_cache_limit(CACHE_LIMIT)
        self.model = Drama3.load(self.model_dir)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.codec = None
        self._transcriber = None
        if load_codec:
            from tensorfold.families.drama3.codec import Codec

            self.codec = Codec(self.model_dir / self.model.config.tensorfold.get("codec", "codec.safetensors"))
        self.load_s = time.time() - started

    # -- one prompt ---------------------------------------------------------------------------------------------
    def rows(self, text: str, references: list[Reference]) -> np.ndarray:
        c = self.model.config
        return build_rows(text, references, self.tokenizer, codebooks=c.codebooks, codebook_size=c.codebook_size)

    def generate_rows(self, rows: np.ndarray, sampling: Sampling, *, seed: int = 0, max_frames: int = 2000,
                      on_frame: Callable[[np.ndarray], None] | None = None, trace: dict | None = None,
                      stop: "StopRule | None" = None) -> Result:
        """Frames after prompt ``rows`` [N, 11] until EOS or ``max_frames``.

        ``trace`` (a dict) collects the prompt's final hidden states and per-step candidate logits for parity checks.
        ``stop`` (a ``StopRule``) may end decoding early on the model's own stop signal.
        """

        frames = self.iter_frames(rows, sampling, seed=seed, max_frames=max_frames, trace=trace, stop=stop)
        while True:
            try:
                value = next(frames)
            except StopIteration as done:
                return done.value
            if on_frame is not None:
                on_frame(value)

    def iter_frames(self, rows: np.ndarray, sampling: Sampling, *, seed: int = 0, max_frames: int = 2000,
                    trace: dict | None = None, stop: "StopRule | None" = None):
        """``generate_rows`` as a generator: yields each frame [11] (int64) as it is drawn, returns the Result."""

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

        fast_step = model.fast_step
        head = fast_step.head(sampling) if fast_step is not None else None
        filler = mx.zeros((1, c.codebooks - 2), dtype=mx.int32)

        def begin(h: mx.array, frame_in: mx.array, window: mx.array, key: mx.array) -> dict:
            """The part of a row the next backbone step depends on: main token, codes 0 and 1."""

            if fast_step is not None:
                logits, main, code0, new_window, code_keys, next_key, p_eos = head(h, window, key)
                hidden, codes = fast_step.tower_head(h, frame_in, main, code0, tcaches, sampling, code_keys)
                return {"logits": logits, "main": main, "codes": codes, "window": new_window, "key": next_key,
                        "p_eos": p_eos, "hidden": hidden, "code_keys": code_keys}
            keys = None if sampling.greedy else mx.random.split(key, 12)
            logits = model.candidate_logits(h[:, -1])
            main, code0 = draw_main(logits, model._candidate_ids, window, sampling,
                                    None if keys is None else keys[:2], c.semantic_begin, c.codebook_size,
                                    c.eos_token_id)

            def draw_code(lg, r):
                return draw(lg, sampling.temperature, sampling.top_k, sampling.top_p,
                            None if keys is None else keys[2 + r], sampling.greedy)

            codes = model.tower_step(h, frame_in, main, code0, tcaches, draw_code)
            return {"logits": logits, "main": main, "codes": codes,
                    "window": mx.concatenate([window[:, 1:], main[:, None]], axis=1),
                    "key": keys[11] if keys is not None else key,
                    "p_eos": mx.softmax(logits / max(float(sampling.temperature), 1e-5), axis=-1)[:, model.eos_index]}

        def finish(row: dict) -> mx.array:
            """The rest of the tower (fast path): the whole frame [1, 1, 11]."""

            codes = row["codes"]
            if fast_step is not None:
                codes = fast_step.tower_tail(row["hidden"], row["main"], list(codes), tcaches, sampling,
                                             row["code_keys"])
            return mx.stack([row["main"], *codes], axis=-1)[:, None]

        def backbone_input(row: dict, frame: mx.array) -> mx.array:
            # the backbone embeds the main token and code 1 only: without the later codes in the graph, the next
            # backbone step does not wait for the rest of this row's tower and the GPU runs both together
            if fast_step is None:
                return frame
            return mx.concatenate([row["main"][:, None], row["codes"][0][:, None], row["codes"][1][:, None],
                                   filler], axis=-1)[:, None]

        row = begin(last, frames[:, -1:], window, key)
        out: list[np.ndarray] = []
        eos_probs: list[mx.array] = []
        finish_reason = "length"
        first_frame_s = 0.0
        decode_started = time.perf_counter()
        while True:
            frame = finish(row)
            pending = None
            if len(out) + 1 < max_frames:
                h = model.backbone_hidden(backbone_input(row, frame), bcache)
                pending = begin(h, frame, row["window"], row["key"])
                mx.async_eval(frame, row["p_eos"], pending["main"], pending["codes"][-1], pending["window"],
                              pending["p_eos"])
            else:
                mx.async_eval(frame, row["p_eos"])
            if trace is not None:
                trace["logits"].append(np.array(row["logits"][0]))
            value = np.array(frame[0, 0])
            eos_probs.append(row["p_eos"])
            if not out:
                first_frame_s = time.perf_counter() - started
            if int(value[0]) == c.eos_token_id:
                finish_reason = "stop"
                break
            out.append(value)
            yield value.astype(np.int64)
            if stop is not None and stop(len(out), row["p_eos"]):
                finish_reason = "stop_rule"
                break
            if pending is None:
                break
            row = pending
        decode_s = time.perf_counter() - decode_started
        result = np.stack(out).astype(np.int64) if out else np.zeros((0, c.codebooks + 1), dtype=np.int64)
        probs = np.array(mx.concatenate(eos_probs)) if eos_probs else np.zeros(0, dtype=np.float32)
        return Result(frames=result, prompt_rows=n, finish=finish_reason, prefill_s=prefill_s, decode_s=decode_s,
                      first_frame_s=first_frame_s, eos_probs=probs)

    # -- text in, audio out -------------------------------------------------------------------------------------
    def synthesize(self, text: str, *, references: list[Reference] | None = None, sampling: Sampling = Sampling(),
                   seed: int = 0, max_chars: int = 300, max_frames: int | None = None,
                   log: Callable[[str], None] = print, verify: "bool | Transcriber" = False,
                   verify_retries: int = VERIFY_RETRIES, policy: "Policy | None" = None,
                   stop: "StopRule | bool" = False) -> tuple[np.ndarray, list[Result]]:
        """Audio (float32, 44.1 kHz) for ``text``; long text goes in sentence-aligned chunks.

        With no reference given, later chunks keep the first chunk's voice by taking its text and frames as their
        reference, when that output looks like a clean reading (it stopped, is short, has a plausible length).

        ``verify`` (True, or a ``verify.Transcriber`` to share one) checks each chunk's audio against its text with
        speech recognition: speech after the end of the text is trimmed, and a take with missing words is drawn
        again with another seed up to ``verify_retries`` times, keeping the best take (``Result.stats["verify"]``
        holds the check). Without it, a chunk that runs into its frame cap is drawn again (greedy runs are not).
        ``stop`` (True, or a ``StopRule`` factory taking the chunk text) also ends a runaway chunk on the model's own
        p(EOS) signal (``StopRule``), without waiting for the frame cap.
        """

        checker = None
        if verify:
            from tensorfold.families.drama3.verify import Policy as _Policy, Transcriber as _Transcriber

            checker = verify if isinstance(verify, _Transcriber) else self.transcriber()
            policy = policy or _Policy()
        references = list(references or [])
        chunks = split_text(text, max_chars)
        results: list[Result] = []
        audio: list[np.ndarray] = []
        voice = list(references)
        for i, chunk in enumerate(chunks):
            rows = self.rows(chunk, voice)
            cap = max_frames or default_frame_cap(chunk)
            retries = 0 if sampling.greedy else (verify_retries if checker is not None else RETRIES)
            best = None
            for attempt in range(1 + retries):
                rule = (StopRule.for_text(chunk) if stop is True else stop(chunk)) if stop else None
                result = self.generate_rows(rows, sampling, seed=seed + i + 1000 * attempt, max_frames=cap,
                                            stop=rule)
                wav = None
                if len(result.frames) and self.codec is not None:
                    wav = np.array(self.codec.decode(result.codes))
                    mx.clear_cache()
                if checker is None:
                    if result.finish == "stop_rule" and rule.cut_frames is not None and wav is not None:
                        result.frames, wav = trim_take(result.frames, wav, rule.cut_frames / FRAME_RATE)
                        result.stats["stop_rule_cut_frames"] = rule.cut_frames
                    best = (result, wav, None)
                    if result.finish != "length":
                        break
                    log(f"[drama3] chunk {i + 1} ran to its {cap}-frame cap; drawing again")
                    continue
                check = checker.check(chunk, wav if wav is not None else np.zeros(0, np.float32), policy)
                result.stats["attempt"] = attempt
                if best is None or check.score > best[2].score:
                    best = (result, wav, check)
                if check.ok:
                    break
                log(f"[drama3] chunk {i + 1} take {attempt + 1}: {check.summary()}"
                    + ("; drawing again" if attempt < retries else ""))
            result, wav, check = best
            if check is not None:
                info = check.as_dict()
                info["attempts"] = attempt + 1
                if check.action == "trim" and wav is not None:
                    before = len(result.frames)
                    result.frames, wav = trim_take(result.frames, wav, check.cut_s, fade_s=policy.fade_s)
                    info["frames_trimmed"] = before - len(result.frames)
                result.stats["verify"] = info
                result.stats["verified"] = check.ok
            results.append(result)
            frames = len(result.frames)
            log(f"[drama3] chunk {i + 1}/{len(chunks)}: {result.prompt_rows} prompt rows, prefill "
                f"{result.prefill_s:.2f}s; {frames} frames ({frames / FRAME_RATE:.1f}s audio) in "
                f"{result.decode_s:.2f}s = {frames / max(result.decode_s, 1e-9):.1f} frames/s; {result.finish}"
                + (f"; check {check.summary()} in {check.asr_s:.2f}s" if check is not None else ""))
            if wav is not None and frames:
                audio.append(wav)
            if not references and len(voice) == 0 and i + 1 < len(chunks) and usable_as_voice(result, chunk):
                voice = [Reference(transcript=chunk, codes=result.codes)]
        silence = np.zeros(int(0.15 * 44100), dtype=np.float32)
        joined = []
        for i, part in enumerate(audio):
            joined += [part] if i == 0 else [silence, part]
        return (np.concatenate(joined) if joined else np.zeros(0, dtype=np.float32)), results

    def transcriber(self) -> "Transcriber":
        """The speech recogniser the end-of-text check uses (loaded once, on first use)."""

        if self._transcriber is None:
            from tensorfold.families.drama3.verify import Transcriber

            self._transcriber = Transcriber(backend=os.environ.get("TF_DRAMA3_ASR", "auto"))
        return self._transcriber


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
