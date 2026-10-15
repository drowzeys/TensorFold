"""Fish Audio Drama 3 text to speech on Apple Silicon (MLX). Built with Fish Audio.

    python -m tensorfold.drama3 --model ~/models/drama-3-mlx --text "Hello there." --out hello.wav
    python -m tensorfold.drama3 --text "<|speaker:0|> [whispering] Over here. <|speaker:1|> Where?" --out talk.wav
    python -m tensorfold.drama3 --text "..." --ref-audio voice.wav --ref-text "what the clip says" --out clone.wav
    python -m tensorfold.drama3 --serve        # load once; one JSON request a line on stdin (see DRAMA3.md)

Speaker tags <|speaker:0|> .. <|speaker:15|> switch voices; [bracketed directions] steer delivery. Only clone a
real person's voice with their consent. Convert Fish's checkpoint once with
``python -m tensorfold.families.drama3.convert <fish dir> <mlx dir>``. See DRAMA3.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_DRAMA3_MODEL", "~/models/drama-3-mlx")
BUILT_WITH = ("[drama3] Built with Fish Audio. Fish Audio Drama 3 runs locally under the Fish Audio Community "
              "License; clone only voices you have consent to use.")


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tensorfold.drama3", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"converted MLX model folder (default {DEFAULT_MODEL})")
    text = p.add_mutually_exclusive_group()
    text.add_argument("--text", help="text to speak")
    text.add_argument("--text-file", type=Path, help="read the text from a file")
    p.add_argument("--ref-audio", action="append", default=[], type=Path,
                   help="voice reference clip (repeatable, paired with --ref-text)")
    p.add_argument("--ref-text", action="append", default=[], help="transcript of each --ref-audio clip")
    p.add_argument("--ref-codes", action="append", default=[], type=Path,
                   help="a reference given as saved codes (.npy [10, frames]) instead of audio, with --ref-text")
    p.add_argument("--out", type=Path,
                   help="output .wav (44.1 kHz mono); with --stream, '-' writes the stream to stdout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="argmax every draw (deterministic; RAS has no effect)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=None, help="frame cap a chunk (default: from text length)")
    p.add_argument("--max-chars", type=int, default=300, help="long text is split into chunks of about this size")
    p.add_argument("--save-codes", type=Path, help="also save the generated frames (.npy [T, 11])")
    p.add_argument("--stats", type=Path, help="write timing and memory as JSON")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                   help="check each chunk with speech recognition: trim speech after the end of the text, draw a "
                        "take with missing words again (default on; needs mlx-whisper or faster-whisper)")
    p.add_argument("--verify-retries", type=int, default=2, help="redraws of a chunk with missing words (default 2)")
    p.add_argument("--asr", choices=("auto", "mlx", "cpu"), default=os.environ.get("TF_DRAMA3_ASR", "auto"),
                   help="speech recogniser of --verify: mlx-whisper on the GPU or faster-whisper on the CPU")
    p.add_argument("--stream", action="store_true",
                   help="write audio while it is generated: a growing WAV at --out (its header sizes are fixed when "
                        "done), or a WAV / raw PCM stream on stdout with --out -")
    p.add_argument("--stream-format", choices=("wav", "pcm"), default="wav",
                   help="--stream output: WAV (header, then 16-bit PCM) or raw 16-bit little-endian PCM")
    p.add_argument("--first-window", type=int, default=10,
                   help="--stream: frames in the first codec window (10 = 0.46 s of audio)")
    p.add_argument("--holdback", type=float, default=0.0,
                   help="--stream: seconds of the newest audio kept unsent so the end-of-text check can still trim")
    p.add_argument("--stop-rule", action=argparse.BooleanOptionalAction, default=None,
                   help="stop a runaway chunk on the model's own p(EOS) signal: the take passed a likely end and went "
                        "on for 3 s (default on)")
    p.add_argument("--serve", nargs="?", const="1", default="0", metavar="1",
                   help="load once and serve: one JSON request a line on stdin, one JSON answer a line on stderr "
                        "('--serve' or '--serve 1'; the protocol is in DRAMA3.md)")
    args = p.parse_args(argv)
    args.serve = args.serve not in ("0", "", "false", "no")
    if not args.serve:
        if args.text is None and args.text_file is None:
            p.error("one of the arguments --text --text-file is required (or --serve)")
        if args.out is None:
            p.error("the following arguments are required: --out (or --serve)")
    return args


def speak(gen, text: str, *, references, sampling, seed: int, max_chars: int = 300, max_frames: int | None = None,
          verify=False, verify_retries: int = 2, stop_rule: bool = True, stream: bool = False,
          out: Path | str | None = None, stream_format: str = "wav", pcm: Path | str | None = None,
          first_window: int = 10, holdback: float = 0.0, log=print) -> tuple:
    """One request: text to audio, written to ``out`` (a WAV, or '-' for stdout when streaming) and/or ``pcm``
    (raw 16-bit PCM, written as it comes when streaming). Returns (audio, results, stats)."""

    import mlx.core as mx
    import numpy as np

    from tensorfold.families.drama3.codec import save_wav
    from tensorfold.families.drama3.stream import Drama3Stream, GrowingWav, StreamOptions, pcm16, wav_header

    mx.reset_peak_memory()
    started = time.perf_counter()
    first_audio_s = None
    out = None if out is None else str(out)
    pcm = None if pcm is None else str(pcm)
    if stream:
        options = StreamOptions(first=first_window, holdback_s=holdback)
        s = Drama3Stream(gen, text, references=references, sampling=sampling, seed=seed, max_chars=max_chars,
                         max_frames=max_frames, verify=verify, log=log, stop=stop_rule, options=options)
        sinks, closers, pieces = [], [], []
        if out == "-" or pcm == "-":
            sink = sys.stdout.buffer
            if out == "-" and stream_format == "wav":
                sink.write(wav_header(None))
            sinks.append(lambda c, sink=sink: (sink.write(pcm16(c)), sink.flush()))
        if out not in (None, "-"):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            if stream_format == "wav":
                wav = GrowingWav(out)
                sinks.append(wav.write)
                closers.append(wav.close)
            else:
                f = open(out, "wb")
                sinks.append(lambda c, f=f: (f.write(pcm16(c)), f.flush()))
                closers.append(f.close)
        if pcm not in (None, "-"):
            Path(pcm).parent.mkdir(parents=True, exist_ok=True)
            f = open(pcm, "wb")
            sinks.append(lambda c, f=f: (f.write(pcm16(c)), f.flush()))
            closers.append(f.close)
        try:
            for chunk in s:
                for sink in sinks:
                    sink(chunk)
                pieces.append(chunk)
        finally:
            for close in closers:
                close()
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        results = s.results
        first_audio_s = s.stats["first_audio_s"]
        total_s = time.perf_counter() - started
    else:
        audio, results = gen.synthesize(text, references=references, sampling=sampling, seed=seed,
                                        max_chars=max_chars, max_frames=max_frames, verify=verify,
                                        verify_retries=verify_retries, log=log, stop=stop_rule)
        total_s = time.perf_counter() - started
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            save_wav(out, audio)
        if pcm not in (None, "-"):
            Path(pcm).parent.mkdir(parents=True, exist_ok=True)
            Path(pcm).write_bytes(pcm16(audio))
    frames = sum(len(r.frames) for r in results)
    decode_s = sum(r.decode_s for r in results)
    seconds = len(audio) / 44100
    stats = {"frames": frames, "audio_s": round(seconds, 3), "decode_s": round(decode_s, 3),
             "frames_per_s": round(frames / max(decode_s, 1e-9), 2),
             "prefill_s": round(sum(r.prefill_s for r in results), 3),
             "first_frame_s": round(results[0].first_frame_s, 3) if results else None,
             "first_audio_s": round(first_audio_s if first_audio_s is not None else total_s, 3),
             "stream": bool(stream), "stop_rule": bool(stop_rule), "verify_on": bool(verify),
             "total_s": round(total_s, 3), "rtf": round(total_s / max(seconds, 1e-9), 3),
             "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "chunks": len(results),
             "finish": [r.finish for r in results],
             "verify": [r.stats.get("verify") for r in results] if verify else None,
             "eos_prob_max": [round(float(r.eos_probs.max()), 4) if len(r.eos_probs) else None for r in results]}
    mx.clear_cache()
    return audio, results, stats


def load_references(gen, pairs, log=print) -> list:
    """[(kind 'audio' | 'codes', path, transcript)] -> References (audio is encoded with the codec)."""

    import numpy as np

    from tensorfold.families.drama3.codec import load_audio
    from tensorfold.families.drama3.prompt import Reference

    references = []
    for kind, path, transcript in pairs:
        path = Path(path).expanduser()
        if kind == "audio":
            started = time.perf_counter()
            codes = np.array(gen.codec.encode(load_audio(path)))
            log(f"[drama3] encoded {path.name}: {codes.shape[1]} frames in {time.perf_counter() - started:.2f}s")
        else:
            codes = np.load(path)
            if codes.ndim == 2 and codes.shape[0] != 10 and codes.shape[1] == 10:
                codes = codes.T
        references.append(Reference(transcript=transcript, codes=codes))
    return references


def make_transcriber(backend: str, log):
    from tensorfold.families.drama3.verify import Transcriber

    asr = Transcriber(backend=backend)
    try:
        asr.warm_up()
    except ImportError:
        log("[drama3] --verify needs mlx-whisper or faster-whisper; running without the check")
        return False
    return asr


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.serve:
        return serve(args)
    import numpy as np

    from tensorfold.families.drama3.generate import FRAME_RATE, Drama3Generator
    from tensorfold.families.drama3.sampling import Sampling

    to_stdout = args.stream and str(args.out) == "-"
    if str(args.out) == "-" and not args.stream:
        raise SystemExit("[drama3] --out - needs --stream")

    def say(message: str) -> None:
        print(message, file=sys.stderr if to_stdout else sys.stdout, flush=True)

    text = args.text if args.text is not None else args.text_file.read_text()
    sources = [("audio", a) for a in args.ref_audio] + [("codes", c) for c in args.ref_codes]
    if len(sources) != len(args.ref_text):
        raise SystemExit("[drama3] give one --ref-text for each --ref-audio / --ref-codes")
    say(BUILT_WITH)
    gen = Drama3Generator(args.model)
    say(f"[drama3] loaded {gen.model_dir} in {gen.load_s:.1f}s")
    references = load_references(gen, [(k, p, t) for (k, p), t in zip(sources, args.ref_text)], say)
    sampling = Sampling(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, greedy=args.greedy)
    verify = make_transcriber(args.asr, say) if args.verify else False
    stop_rule = True if args.stop_rule is None else args.stop_rule
    if args.stream:
        from tensorfold.families.drama3.stream import StreamOptions, warm_up

        warm_up(gen, StreamOptions(first=args.first_window, holdback_s=args.holdback), sampling)
    audio, results, stats = speak(gen, text, references=references, sampling=sampling, seed=args.seed,
                                  max_chars=args.max_chars, max_frames=args.max_frames, verify=verify,
                                  verify_retries=args.verify_retries, stop_rule=stop_rule, stream=args.stream,
                                  out=args.out, stream_format=args.stream_format, first_window=args.first_window,
                                  holdback=args.holdback, log=say)
    say(f"[drama3] wrote {args.out}: {stats['audio_s']:.1f}s audio, {stats['frames']} frames at "
        f"{stats['frames_per_s']} frames/s (real time is {FRAME_RATE:.1f}); first audio {stats['first_audio_s']}s, "
        f"total {stats['total_s']:.1f}s, RTF {stats['rtf']}; peak {stats['peak_memory_gb']} GB")
    if args.save_codes:
        np.save(args.save_codes, np.concatenate([r.frames for r in results]) if results else np.zeros((0, 11)))
    if args.stats:
        args.stats.write_text(json.dumps(stats, indent=1) + "\n")
    return 0


# -- the persistent server ---------------------------------------------------------------------------------------

def answer(message: dict) -> None:
    """One JSON line on stderr (the protocol channel; log lines never start with '{')."""

    sys.stderr.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stderr.flush()


def serve(args: argparse.Namespace) -> int:
    """``--serve``: TensorFold's CUDA engine's protocol (drama3-generate --serve 1), on MLX.

    Loads the model, the codec and (with --verify) Whisper once, warms the compiled decode step and the codec
    windows, writes ``{"ready":true}`` on stderr, then reads one JSON request a line on stdin until it closes and
    answers each with one JSON line on stderr. Logs go to stderr too (lines that do not start with '{')."""

    import mlx.core as mx

    from tensorfold.families.drama3.generate import CACHE_LIMIT, Drama3Generator
    from tensorfold.families.drama3.sampling import Sampling
    from tensorfold.families.drama3.stream import StreamOptions, warm_up

    def log(message: str) -> None:
        print(message.lstrip("{"), file=sys.stderr, flush=True)

    log(BUILT_WITH)
    started = time.perf_counter()
    try:
        gen = Drama3Generator(args.model)
        asr = make_transcriber(args.asr, log) if args.verify else False
        default_sampling = Sampling(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        warm_up(gen, StreamOptions(first=args.first_window), default_sampling)
        mx.set_cache_limit(CACHE_LIMIT)
        mx.clear_cache()
    except Exception as error:                              # noqa: BLE001
        answer({"ok": False, "error": f"load failed: {type(error).__name__}: {error}"})
        return 1
    log(f"[drama3] serving {gen.model_dir}: loaded and warm in {time.perf_counter() - started:.1f}s; "
        f"memory {mx.get_active_memory() / 1e9:.1f} GB")
    answer({"ready": True, "engine": "tensorfold-drama3-mlx", "load_s": round(time.perf_counter() - started, 2),
            "verify": bool(asr), "stream": True})
    encoded: dict = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            q = json.loads(line)
            if not isinstance(q, dict):
                raise ValueError
        except ValueError:
            answer({"ok": False, "error": "bad json"})
            continue
        try:
            answer(serve_one(gen, q, args, asr, encoded, log))
        except Exception as error:                          # noqa: BLE001
            answer({"ok": False, "error": f"{type(error).__name__}: {error}", "id": q.get("id")})
        finally:
            mx.clear_cache()
    return 0


def serve_one(gen, q: dict, args, asr, encoded: dict, log) -> dict:
    import numpy as np

    from tensorfold.families.drama3.codec import load_audio
    from tensorfold.families.drama3.prompt import Reference
    from tensorfold.families.drama3.sampling import Sampling

    began = time.perf_counter()
    text = q.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("a request needs \"text\"")
    references, encode_ms, ref_frames = [], 0.0, 0
    refs = q.get("refs") or ([{"audio": q.get("ref_audio"), "codes": q.get("ref_codes"), "text": q.get("ref_text")}]
                             if (q.get("ref_audio") or q.get("ref_codes")) else [])
    for ref in refs:
        if not ref.get("text"):
            raise ValueError("a reference needs its transcript (\"ref_text\")")
        if ref.get("codes"):
            codes = np.load(Path(ref["codes"]).expanduser())
            if codes.ndim == 2 and codes.shape[0] != 10 and codes.shape[1] == 10:
                codes = codes.T
        else:
            path = Path(ref["audio"]).expanduser()
            key = (str(path), path.stat().st_mtime_ns)
            if key not in encoded:
                t = time.perf_counter()
                encoded[key] = np.array(gen.codec.encode(load_audio(path)))
                encode_ms += (time.perf_counter() - t) * 1000
                if len(encoded) > 16:
                    encoded.pop(next(iter(encoded)))
            codes = encoded[key]
        ref_frames += int(codes.shape[1])
        references.append(Reference(transcript=ref["text"], codes=codes))
    sampling = Sampling(temperature=float(q.get("temperature", args.temperature)),
                        top_k=int(q.get("top_k", args.top_k)), top_p=float(q.get("top_p", args.top_p)),
                        greedy=bool(q.get("greedy", False)))
    pcm = q.get("pcm")
    stream = bool(q.get("stream", pcm is not None))
    verify = asr if (asr and q.get("verify", True)) else False
    stop_rule = q.get("stop_rule", True if args.stop_rule is None else args.stop_rule)
    max_frames = q.get("max_frames")
    audio, results, stats = speak(
        gen, text, references=references, sampling=sampling, seed=int(q.get("seed", args.seed)),
        max_chars=int(q.get("max_chars", args.max_chars)), max_frames=None if max_frames is None else int(max_frames),
        verify=verify, verify_retries=int(q.get("verify_retries", args.verify_retries)), stop_rule=bool(stop_rule),
        stream=stream, out=q.get("wav"), stream_format="wav", pcm=pcm,
        first_window=int(q.get("first_window", args.first_window)), holdback=float(q.get("holdback", args.holdback)),
        log=log)
    if q.get("save_codes"):
        np.save(Path(q["save_codes"]).expanduser(),
                np.concatenate([r.frames for r in results]) if results else np.zeros((0, 11)))
    total_ms = (time.perf_counter() - began) * 1000
    decode_s = stats["decode_s"]
    reply = {"ok": True, "id": q.get("id"), "frames": stats["frames"],
             "eos": all(f != "length" for f in stats["finish"]), "audio_s": stats["audio_s"],
             "prompt_rows": sum(r.prompt_rows for r in results), "ref_frames": ref_frames,
             "encode_ms": round(encode_ms, 1), "prefill_ms": round(stats["prefill_s"] * 1000, 1),
             "first_audio_ms": round(stats["first_audio_s"] * 1000 + encode_ms, 1), "total_ms": round(total_ms, 1),
             "fps": round(stats["frames"] / decode_s, 2) if decode_s else None,
             "rtf": round(total_ms / 1000 / stats["audio_s"], 3) if stats["audio_s"] else None,
             "wav": q.get("wav") or "", "pcm": pcm or "", "stream": stats["stream"], "chunks": stats["chunks"],
             "finish": stats["finish"], "stop_rule": stats["stop_rule"], "verify": stats["verify"],
             "peak_memory_gb": stats["peak_memory_gb"]}
    if q.get("stats"):
        Path(q["stats"]).expanduser().write_text(json.dumps({**stats, **reply}, indent=1) + "\n")
    return reply


if __name__ == "__main__":
    sys.exit(main())
