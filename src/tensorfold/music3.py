"""MiniMax Music 3 text and lyrics to a full song on Apple Silicon (MLX): 44.1 kHz stereo.

    python -m tensorfold.music3 --caption "anthemic hip-hop, male rap vocal, 92 bpm" --lyrics-file song.txt \\
        --seconds 60 --seed 7 --out song.wav
    python -m tensorfold.music3 --serve        # load once; one JSON request a line on stdin (docs/recipes/minimax-music3-mlx.md)

The weights are Comfy-Org/MiniMax-Music-3's single files (text encoder pruned bf16, DiT fp16, DAV), read as they are
from --model. MiniMax-Music3 is released under the MiniMax-Music3 Community License: a commercial product must show
"MiniMax-Music3" in its interface, and one above USD 20M yearly revenue needs MiniMax's written permission.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("TF_MUSIC3_MODEL", "~/studio/models/MiniMax-Music3-Comfy")
BUILT_WITH = ("[music3] MiniMax-Music3 runs locally under the MiniMax-Music3 Community License "
              "(show \"MiniMax-Music3\" in a commercial UI).")


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tensorfold.music3", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Comfy-Org/MiniMax-Music-3 folder (default {DEFAULT_MODEL})")
    p.add_argument("--caption", help="style / instruments / mood / vocal description")
    p.add_argument("--caption-file", type=Path)
    lyrics = p.add_mutually_exclusive_group()
    lyrics.add_argument("--lyrics", help="lyrics with [Verse] / [Chorus] / [Instrumental] section tags")
    lyrics.add_argument("--lyrics-file", type=Path)
    p.add_argument("--seconds", type=float, default=60.0, help="length cap; the model may end the song earlier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ar-cfg", type=float, default=1.5, help="AR classifier-free guidance (reference default 1.5)")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--steps", type=int, default=30, help="DiT Euler steps")
    p.add_argument("--dit-cfg", type=float, default=1.7, help="DiT guidance against the empty condition")
    p.add_argument("--dit-dtype", choices=("float16", "bfloat16"), default="float16",
                   help="DiT weights/compute (the checkpoint is fp16)")
    p.add_argument("--quant", default="none",
                   help="weights: none (bf16 / fp16, the default), int8 (AR body, code-0 head and depth decoder at "
                        "8 bits), int8-d6, int8-d4, int6, int4, or per-part overrides such as int8,dit=8 or "
                        "body=8,head=8,depth=6,group=64 (see docs/recipes/minimax-music3-mlx.md)")
    p.add_argument("--dit-mode", choices=("averaged", "sequential"), default="averaged",
                   help="DiT windows: averaged (ComfyUI's, the default) or sequential (diffusers'; what --stream uses)")
    p.add_argument("--stream", action="store_true",
                   help="run the DiT and decoder window by window while the AR draws, writing --out as it grows "
                        "(a WAV with unknown-length sizes until done; '-' streams the WAV to stdout)")
    p.add_argument("--no-overlap", dest="overlap", action="store_false",
                   help="with --stream: run each DiT window in one go between AR frames instead of submitting its "
                        "steps on a second GPU stream (same audio, slower)")
    p.add_argument("--pcm", type=Path, help="with --stream: also write raw interleaved 16-bit stereo PCM as it comes")
    p.add_argument("--out", type=Path, help="output .wav (44.1 kHz stereo, 16-bit)")
    p.add_argument("--stats", type=Path, help="write timing and memory as JSON")
    p.add_argument("--serve", nargs="?", const="1", default="0", metavar="1",
                   help="load once and serve: one JSON request a line on stdin, JSON lines on stderr")
    args = p.parse_args(argv)
    args.serve = args.serve not in ("0", "", "false", "no")
    if args.caption_file is not None:
        args.caption = args.caption_file.read_text()
    if not args.serve:
        if args.caption is None:
            p.error("--caption or --caption-file is required (or --serve)")
        if args.out is None:
            p.error("--out is required (or --serve)")
        if str(args.out) == "-" and not args.stream:
            p.error("--out - needs --stream")
        if args.pcm is not None and not args.stream:
            p.error("--pcm needs --stream")
    try:
        from tensorfold.families.music3.model import parse_quant

        parse_quant(args.quant)
    except ValueError as error:
        p.error(str(error))
    return args


def settings_of(q: dict, args: argparse.Namespace):
    from tensorfold.families.music3.generate import Settings

    mode = str(q.get("dit_mode", args.dit_mode))
    if mode not in ("averaged", "sequential"):
        raise ValueError("\"dit_mode\" is \"averaged\" or \"sequential\"")
    return Settings(seconds=float(q.get("seconds", args.seconds)), seed=int(q.get("seed", args.seed)),
                    ar_cfg=float(q.get("ar_cfg", args.ar_cfg)), top_k=int(q.get("top_k", args.top_k)),
                    steps=int(q.get("steps", args.steps)), dit_cfg=float(q.get("dit_cfg", args.dit_cfg)),
                    dit_mode=mode, overlap=bool(q.get("overlap", args.overlap)))


def load(args):
    import mlx.core as mx

    from tensorfold.families.music3.generate import Music3Generator

    return Music3Generator(args.model, dit_dtype=getattr(mx, args.dit_dtype), quant=args.quant)


def stream_song(gen, caption: str, lyrics: str, settings, progress, wav=None, pcm=None):
    """A streamed song: ``wav`` grows as the song is made (or "-" for stdout), ``pcm`` gets raw stereo PCM."""

    from tensorfold.families.music3.generate import GrowingWav, RawPcm

    sinks = []
    if wav is not None:
        sinks.append(GrowingWav(wav))
    if pcm is not None:
        sinks.append(RawPcm(pcm))
    try:
        return gen.generate(caption, lyrics, settings, progress, sink=lambda a: [k.write(a) for k in sinks])
    finally:
        for k in sinks:
            k.close()


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.serve:
        return serve(args)
    from tensorfold.families.music3.generate import save_wav, write_stats

    lyrics = args.lyrics if args.lyrics is not None else (args.lyrics_file.read_text() if args.lyrics_file else "")
    to_stdout = args.stream and str(args.out) == "-"
    say = (lambda m: print(m, file=sys.stderr, flush=True)) if to_stdout else (lambda m: print(m, flush=True))
    say(BUILT_WITH)
    gen = load(args)
    say(f"[music3] loaded {gen.model_dir} in {gen.load_s:.1f}s (quant {gen.quant_name})")

    def progress(stage, done, total):
        say(f"[music3] {stage} {done}/{total}")

    if args.stream:
        song = stream_song(gen, args.caption, lyrics, settings_of({}, args), progress, wav=args.out, pcm=args.pcm)
    else:
        song = gen.generate(args.caption, lyrics, settings_of({}, args), progress)
        save_wav(args.out, song.audio)
    st = song.stats
    say(f"[music3] wrote {args.out}: {st['audio_s']:.1f}s ({st['frames']} frames, "
        f"{'model ended it' if song.stopped else 'hit the cap'}); AR {st['ar_s']:.1f}s, DiT {st['dit_s']:.1f}s, "
        f"DAV {st['dav_s']:.1f}s, total {st['total_s']:.1f}s, RTF {st['rtf']}; first audio {st['first_audio_s']:.1f}s; "
        f"peak {st['peak_memory_gb']} GB")
    if args.stats:
        write_stats(args.stats, st)
    return 0


# -- the persistent server ---------------------------------------------------------------------------------------

def answer(message: dict) -> None:
    """One JSON line on stderr (the protocol channel; log lines never start with '{')."""

    sys.stderr.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stderr.flush()


def serve(args: argparse.Namespace) -> int:
    """``--serve``: Drama 3's protocol. Loads once, writes ``{"ready":true,...}`` on stderr, then answers one JSON
    request a line from stdin with one JSON line on stderr. A request with ``"progress": true`` also gets
    ``{"id":..,"progress":{"stage":"ar"|"dit"|"dav","done":n,"total":m}}`` lines before its answer.

    Opt-in fields (older requests are answered as before): ``"stream": true`` (or a ``"pcm"`` path, as Drama 3's
    server) runs the sequential DiT while the AR draws and writes ``wav`` as a growing WAV (unknown-length header
    until done, sizes set at the end) and ``pcm`` as raw interleaved 16-bit stereo PCM, both as each piece is decoded;
    with ``"progress": true`` a stream also says ``{"stage":"stream","done":seconds sent,"total":cap}``.
    ``"dit_mode": "sequential"`` uses the sequential windows without streaming. The ready line also says
    ``"stream": true``, ``"quant"`` (the --quant the server loaded) and ``"dit_modes"``."""

    import mlx.core as mx

    from tensorfold.families.music3.generate import CACHE_LIMIT

    def log(message: str) -> None:
        print(message.lstrip("{"), file=sys.stderr, flush=True)

    log(BUILT_WITH)
    started = time.perf_counter()
    try:
        gen = load(args)
        mx.set_cache_limit(CACHE_LIMIT)
        mx.clear_cache()
    except Exception as error:                              # noqa: BLE001
        answer({"ok": False, "error": f"load failed: {type(error).__name__}: {error}"})
        return 1
    log(f"[music3] serving {gen.model_dir}: loaded in {time.perf_counter() - started:.1f}s; "
        f"memory {mx.get_active_memory() / 1e9:.1f} GB")
    answer({"ready": True, "engine": "tensorfold-music3-mlx", "load_s": round(time.perf_counter() - started, 2),
            "sample_rate": 44100, "channels": 2, "license_display": "MiniMax-Music3", "stream": True,
            "quant": gen.quant_name, "dit_modes": ["averaged", "sequential"]})
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
            answer(serve_one(gen, q, args, log))
        except Exception as error:                          # noqa: BLE001
            answer({"ok": False, "error": f"{type(error).__name__}: {error}", "id": q.get("id")})
        finally:
            mx.clear_cache()
    return 0


def serve_one(gen, q: dict, args, log) -> dict:
    from tensorfold.families.music3.generate import save_wav

    caption = q.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("a request needs \"caption\"")
    lyrics = q.get("lyrics")
    if lyrics is None and q.get("lyrics_file"):
        lyrics = Path(q["lyrics_file"]).expanduser().read_text()
    wav = q.get("wav")
    if not wav:
        raise ValueError("a request needs \"wav\" (the output path)")
    s = settings_of(q, args)
    rid = q.get("id")
    last = {}

    def progress(stage, done, total):
        if q.get("progress") and (done == total or time.perf_counter() - last.get(stage, 0) > 1.0):
            last[stage] = time.perf_counter()
            answer({"id": rid, "progress": {"stage": stage, "done": done, "total": total}})

    pcm = q.get("pcm")
    stream = bool(q.get("stream", pcm is not None))      # as Drama 3's server: a "pcm" path asks for a stream
    log(f"[music3] request {rid}: {s.seconds:.0f}s cap, seed {s.seed}"
        + (", streamed" if stream else "") + (f", {s.dit_mode} DiT" if s.dit_mode != "averaged" else ""))
    if stream:
        song = stream_song(gen, caption, lyrics or "", s, progress, wav=Path(wav).expanduser(),
                           pcm=None if not pcm else Path(pcm).expanduser())
    else:
        song = gen.generate(caption, lyrics or "", s, progress)
        save_wav(Path(wav).expanduser(), song.audio)
        if pcm:
            from tensorfold.families.music3.generate import pcm16

            Path(pcm).expanduser().parent.mkdir(parents=True, exist_ok=True)
            Path(pcm).expanduser().write_bytes(pcm16(song.audio))
    st = song.stats
    reply = {"ok": True, "id": rid, "wav": str(wav), "audio_s": st["audio_s"], "frames": st["frames"],
             "eos": song.stopped, "ar_ms": round(st["ar_s"] * 1000, 1), "dit_ms": round(st["dit_s"] * 1000, 1),
             "dav_ms": round(st["dav_s"] * 1000, 1), "total_ms": round(st["total_s"] * 1000, 1), "rtf": st["rtf"],
             "first_audio_ms": round(st["first_audio_s"] * 1000, 1), "stream": stream, "seed": s.seed,
             "peak_memory_gb": st["peak_memory_gb"]}
    if stream or pcm or s.dit_mode != "averaged":            # new fields only for requests that used the new options
        reply.update(pcm=pcm or "", dit_mode=st["dit_mode"], windows=st.get("windows"), quant=st["quant"])
    if q.get("stats"):
        Path(q["stats"]).expanduser().write_text(json.dumps({**st, **reply}, indent=1) + "\n")
    return reply


if __name__ == "__main__":
    sys.exit(main())
