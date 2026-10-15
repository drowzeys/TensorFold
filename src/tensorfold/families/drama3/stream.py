"""Streaming Drama 3: 44.1 kHz audio chunks while the frames are being drawn.

    from tensorfold.families.drama3.generate import Drama3Generator
    from tensorfold.families.drama3.stream import Drama3Stream, wav_stream

    gen = Drama3Generator("~/models/drama-3-mlx")
    stream = Drama3Stream(gen, "Hello there.", seed=1)
    for chunk in stream:                    # float32 numpy arrays, 44.1 kHz mono
        play(chunk)
    print(stream.stats)                     # first_audio_s, frames, trims, ...
    # or bytes for an HTTP response / a pipe: a WAV header, then 16-bit PCM as it comes
    for piece in wav_stream(Drama3Stream(gen, "Hello there.")):
        response.write(piece)

Frames are decoded in windows (``codec.StreamDecoder``): a small first window (10 frames, ~0.46 s of audio) for a
short time to first audio, then growing ones; each recomputes the codec quantizer's latent over the frames before it
and decodes only the new frames (plus a little context), so the stream matches the one-pass decode to float noise.
Long text streams chunk after chunk, with the same chunking and voice carry-over as ``Drama3Generator.synthesize``.

Audio that has been sent cannot be taken back, so the end-of-text guard works differently than in ``synthesize``
(which can listen to a whole take, trim it, or draw it again):

- the model-side ``StopRule`` (on by default) watches p(EOS). From the first step that puts real weight on EOS (a
  *candidate end*, usually the last frame or two of a take) the audio is held back instead of sent; if the take
  goes on for ``max_overrun`` frames past it, the chunk is stopped as a runaway and cut shortly after the
  candidate. Mid-text p(EOS) stays low, so the holdback costs nothing until the very end of a take;
- with ``verify``, a chunk's whole audio is checked by speech recognition when it ends, and a tail after the last
  word is trimmed from what is still held back (``stats["chunks"][i]["verify"]``; ``late_s`` reports a cut that
  came after audio was sent). A take with missing words cannot be redrawn once it is playing: it is reported;
- ``holdback_s`` additionally keeps the newest audio unsent all along, for a runaway with no EOS signal at all (at
  the cost of that much later first audio).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from tensorfold.families.drama3.codec import SAMPLE_RATE, StreamDecoder
from tensorfold.families.drama3.generate import (Drama3Generator, Result, StopRule, default_frame_cap, split_text,
                                                 usable_as_voice)
from tensorfold.families.drama3.prompt import Reference
from tensorfold.families.drama3.sampling import Sampling

CHUNK_GAP_S = 0.15


@dataclass
class StreamOptions:
    first: int = 10                          # frames in the first codec window (~0.46 s of audio)
    ramp: tuple[int, ...] = (10, 12, 32)     # the next windows
    stride: int = 64                         # then every 64 frames (~3 s)
    context: int = 16                        # frames of decoder context decoded with each window
    crossfade: int = 0                       # samples blended at each seam (0: the seams are already exact)
    holdback_s: float = 0.0                  # newest audio kept unsent, for the end-of-chunk trim


class Drama3Stream:
    """Iterate for float32 audio chunks; ``results`` and ``stats`` are filled in as it runs."""

    def __init__(self, gen: Drama3Generator, text: str, *, references: list[Reference] | None = None,
                 sampling: Sampling = Sampling(), seed: int = 0, max_chars: int = 300, max_frames: int | None = None,
                 options: StreamOptions = StreamOptions(), stop=True,
                 verify=False, log: Callable[[str], None] = print) -> None:
        if gen.codec is None:
            raise ValueError("streaming needs the codec (Drama3Generator(load_codec=True))")
        self.gen, self.text, self.references = gen, text, list(references or [])
        self.sampling, self.seed, self.max_chars, self.max_frames = sampling, seed, max_chars, max_frames
        self.options, self.stop, self.verify, self.log = options, stop, verify, log
        self.results: list[Result] = []
        self.stats: dict = {"first_audio_s": None, "chunks": []}

    def __iter__(self) -> Iterator[np.ndarray]:
        started = time.perf_counter()
        sent_total = 0
        checker = None
        if self.verify:
            from tensorfold.families.drama3.verify import Transcriber

            checker = self.verify if isinstance(self.verify, Transcriber) else self.gen.transcriber()

        def emit(audio: np.ndarray):
            nonlocal sent_total
            if self.stats["first_audio_s"] is None and len(audio):
                self.stats["first_audio_s"] = time.perf_counter() - started
            sent_total += len(audio)
            return audio

        chunks = split_text(self.text, self.max_chars)
        voice = list(self.references)
        hold = int(self.options.holdback_s * SAMPLE_RATE)
        for i, chunk in enumerate(chunks):
            if i:
                yield emit(np.zeros(int(CHUNK_GAP_S * SAMPLE_RATE), dtype=np.float32))
            rows = self.gen.rows(chunk, voice)
            cap = self.max_frames or default_frame_cap(chunk)
            rule = None
            if self.stop:
                rule = StopRule.for_text(chunk) if self.stop is True else self.stop(chunk)
            o = self.options
            decoder = StreamDecoder(self.gen.codec, first=o.first, ramp=o.ramp, stride=o.stride, context=o.context,
                                    crossfade=o.crossfade)
            frames = self.gen.iter_frames(rows, self.sampling, seed=self.seed + i, max_frames=cap, stop=rule)
            parts: list[np.ndarray] = []        # this chunk's audio, sent and held
            hop = self.gen.codec.frame_length
            sent = 0                            # samples of this chunk already sent
            while True:
                try:
                    frame = next(frames)
                except StopIteration as done:
                    result = done.value
                    break
                for audio in decoder.push(frame[1:]):
                    parts.append(audio)
                    total = sum(len(p) for p in parts)
                    release = total - hold
                    if rule is not None and rule.hold_from is not None:     # past a candidate end: hold it
                        release = min(release, rule.hold_from * hop)
                    if release > sent:
                        out = np.concatenate(parts)[sent:release]
                        sent += len(out)
                        yield emit(out)
            parts.append(decoder.flush())
            audio = np.concatenate(parts)
            info = {"frames": len(result.frames), "finish": result.finish, "codec_s": round(decoder.decode_s, 3),
                    "decode_s": round(result.decode_s, 3), "prefill_s": round(result.prefill_s, 3),
                    "held_from_s": None if rule is None or rule.hold_from is None
                    else round(rule.hold_from * hop / SAMPLE_RATE, 3)}
            if result.finish == "stop_rule" and checker is None and rule.cut_frames is not None:
                cut = max(sent, rule.cut_frames * hop)
                info["stop_rule_cut_s"] = round(cut / SAMPLE_RATE, 3)
                audio = audio[:cut].copy()
                n = min(len(audio) - sent, int(0.04 * SAMPLE_RATE))
                if n > 0:
                    audio[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
                result.frames = result.frames[:-(-cut // hop)]
            if checker is not None and len(audio):
                check = checker.check(chunk, audio)
                info["verify"] = check.as_dict()
                if check.action == "trim":
                    cut = int(check.cut_s * SAMPLE_RATE)
                    if cut >= sent:
                        audio = audio[:cut].copy()
                        n = min(len(audio) - sent, int(0.04 * SAMPLE_RATE))
                        if n > 0:
                            audio[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
                        info["trimmed_s"] = round((sum(len(p) for p in parts) - cut) / SAMPLE_RATE, 3)
                        result.frames = result.frames[:-(-cut // self.gen.codec.frame_length)]
                    else:
                        info["late_s"] = round((sent - cut) / SAMPLE_RATE, 3)   # already played past the cut
                        audio = audio[:sent]
                result.stats["verify"] = info["verify"]
                result.stats["verified"] = check.ok
            if len(audio) > sent:
                yield emit(audio[sent:])
            self.results.append(result)
            self.stats["chunks"].append(info)
            self.log(f"[drama3] chunk {i + 1}/{len(chunks)}: {info['frames']} frames, {result.finish}; "
                     f"first audio {self.stats['first_audio_s'] or 0:.2f}s")
            if not self.references and not voice and i + 1 < len(chunks) and usable_as_voice(result, chunk):
                voice = [Reference(transcript=chunk, codes=result.codes)]
        self.stats.update(total_s=round(time.perf_counter() - started, 3), audio_s=round(sent_total / SAMPLE_RATE, 3),
                          frames=sum(len(r.frames) for r in self.results))
        if self.stats["first_audio_s"] is not None:
            self.stats["first_audio_s"] = round(self.stats["first_audio_s"], 3)


def warm_up(gen: Drama3Generator, options: StreamOptions = StreamOptions(), sampling: Sampling = Sampling()) -> None:
    """Draw a few frames and run the codec over the first windows' shapes once: the compiled decode step and the
    codec's GPU kernels are built on first use (~0.2 s), which would otherwise delay a fresh process's first audio."""

    gen.generate_rows(gen.rows("Hello.", []), sampling, seed=0, max_frames=3)
    decoder = StreamDecoder(gen.codec, first=options.first, ramp=options.ramp, stride=options.stride,
                            context=options.context)
    decoder.push(np.zeros((10, options.first + sum(options.ramp[:2])), dtype=np.int64))


# -- bytes ----------------------------------------------------------------------------------------------------------

STREAMING_SIZE = 0xFFFFFFFF


def wav_header(data_bytes: int | None = None, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    """A 44-byte 16-bit PCM WAV header; ``data_bytes`` None writes the "unknown length" sizes streams use."""

    data = STREAMING_SIZE - 36 if data_bytes is None else data_bytes
    riff = STREAMING_SIZE if data_bytes is None else 36 + data_bytes
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                                                           sample_rate * channels * 2, channels * 2, 16)
            + b"data" + struct.pack("<I", data))


def pcm16(audio: np.ndarray) -> bytes:
    """float32 [-1, 1] -> little-endian 16-bit PCM bytes."""

    return (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def wav_stream(chunks, sample_rate: int = SAMPLE_RATE) -> Iterator[bytes]:
    """Bytes of a streamed WAV (for a chunked HTTP response or a pipe): the header, then PCM as chunks arrive."""

    yield wav_header(None, sample_rate)
    for chunk in chunks:
        if len(chunk):
            yield pcm16(chunk)


def stream_wav(gen: Drama3Generator, text: str, **kwargs) -> Iterator[bytes]:
    """``wav_stream(Drama3Stream(gen, text, **kwargs))``: what a local HTTP handler returns for chunked audio/wav."""

    return wav_stream(Drama3Stream(gen, text, **kwargs))


class GrowingWav:
    """A WAV file written as audio arrives: readable while it grows (unknown-length header), its sizes fixed on
    ``close``."""

    def __init__(self, path, sample_rate: int = SAMPLE_RATE) -> None:
        self.f = open(path, "wb")
        self.sample_rate = sample_rate
        self.bytes = 0
        self.f.write(wav_header(None, sample_rate))
        self.f.flush()

    def write(self, audio: np.ndarray) -> None:
        data = pcm16(audio)
        self.f.write(data)
        self.f.flush()
        self.bytes += len(data)

    def close(self) -> None:
        self.f.seek(0)
        self.f.write(wav_header(self.bytes, self.sample_rate))
        self.f.close()
