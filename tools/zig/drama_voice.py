"""Encode a clip into a Drama 3 voice file, or combine voice files into one holding several references."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import string
import sys
from pathlib import Path

FORMAT = "drama3-voice/1"
# Several references, spoken in their order; a loader that knows only /1 refuses it instead of using one of them.
SEVERAL = "drama3-voice/2"
# The fields a reference keeps when files are combined: the codes and the provenance of its own clip.
REFERENCE_FIELDS = ("text", "codes_sha256", "frames", "audio_sha256", "encoder", "vq_codes")
# The reference's max_reference_audio_seconds default.
MAX_SECONDS = 60.0
NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 4096
# Whose voice a file holds; docs/drama-voices.md says which voices may be encoded.
KINDS = ("model-invented", "owner", "consented")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def codes_sha256(rows: list[list[int]]) -> str:
    """The digest the native loader checks: the [codebooks, frames] grid as little-endian int32, codebook-major."""
    return sha256(b"".join(code.to_bytes(4, "little", signed=True) for row in rows for code in row))


def encode(reference: Path, checkpoint: Path, audio: Path, device: str) -> tuple[list[list[int]], dict]:
    sys.path.insert(0, str(reference))
    import torch
    from sglang_omni.models.fishaudio_s2_pro.stages import FishReferenceEncodeHook, FishReferenceInput, load_codec

    codec = load_codec(str(checkpoint), device)
    on_cuda = torch.device(device).type == "cuda"
    hook = FishReferenceEncodeHook(
        codec=codec,
        checkpoint_id=str(checkpoint / "codec.pth"),
        device=device,
        autocast_dtype=torch.bfloat16 if on_cuda else None,
        max_reference_audio_seconds=MAX_SECONDS,
    )
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    codes = hook.encode_one(FishReferenceInput("path", str(audio)))
    if codes.dtype not in (torch.int32, torch.int64) or codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS:
        raise SystemExit(f"the codec returned {codes.dtype} {tuple(codes.shape)}, not [{NUM_CODEBOOKS}, frames]")
    if codes.numel() == 0 or codes.min() < 0 or codes.max() >= CODEBOOK_SIZE:
        raise SystemExit(f"the codec returned codes outside [0, {CODEBOOK_SIZE})")
    encoder = {
        "reference_commit": subprocess.run(["git", "-C", str(reference), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "codec_sha256": sha256((checkpoint / "codec.pth").read_bytes()),
        "sample_rate": int(codec.sample_rate),
        "frame_length": int(codec.frame_length),
        "dtype": "bfloat16" if on_cuda else "float32",
        "device": torch.cuda.get_device_name() if on_cuda else "cpu",
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()) if on_cuda else 0,
    }
    return codes.tolist(), encoder


def encodes_voices(binary: str) -> bool:
    """Whether ``binary --help`` lists ``voice encode``: older tensorfold-native builds serve but cannot encode."""
    shown = subprocess.run([binary, "--help"], capture_output=True, text=True)
    return shown.returncode == 0 and "voice encode" in shown.stdout


def encode_native(binary: str, checkpoint: Path, audio: Path, transcript: str, out: Path) -> dict:
    """``tensorfold-native voice encode`` writes the same voice file, minus its source."""
    command = [binary, "voice", "encode", str(audio), "--model", str(checkpoint), "--text", transcript, "--name", out.stem, "--out", str(out)]
    # its summary lacks the source; the one printed after it is the file's
    if subprocess.run(command, stdout=subprocess.DEVNULL).returncode != 0:
        raise SystemExit(f"{binary} voice encode failed")
    return json.loads(out.read_text(encoding="utf-8"))


def encode_reference(reference: Path, checkpoint: Path, audio: Path, transcript: str, device: str) -> dict:
    rows, encoder = encode(reference, checkpoint, audio, device)
    return {
        "format": FORMAT,
        "text": transcript,
        "codes_sha256": codes_sha256(rows),
        "frames": len(rows[0]),
        "audio_sha256": sha256(audio.read_bytes()),
        "encoder": encoder,
        "vq_codes": rows,
    }


def references(path: Path) -> tuple[list[dict], str]:
    """A voice file's references and source kind, refused where --voices would refuse its source or digests."""
    voice = json.loads(path.read_text(encoding="utf-8"))
    if voice.get("format") not in (FORMAT, SEVERAL):
        raise SystemExit(f"{path}: format is not {FORMAT} or {SEVERAL}")
    source = voice.get("source") if isinstance(voice.get("source"), dict) else {}
    kind, note = source.get("kind"), source.get("note")
    if kind not in KINDS:
        raise SystemExit(f"{path} does not say whose voice it holds (source.kind); a combined file can only carry one source")
    if not isinstance(note, str) or not note.strip(string.whitespace):
        raise SystemExit(f"{path} does not say where its voice comes from (source.note), so --voices would refuse it")
    found = (voice.get("references") or []) if voice["format"] == SEVERAL else [voice]
    for i, reference in enumerate(found):
        if codes_sha256(reference.get("vq_codes", [])) != reference.get("codes_sha256"):
            raise SystemExit(f"{path}: reference {i}: codes_sha256 does not match vq_codes")
        if not isinstance(reference.get("audio_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", reference["audio_sha256"]):
            raise SystemExit(f"{path}: reference {i}: audio_sha256 is not a clip's sha256 (64 lowercase hex digits)")
    return [{key: reference[key] for key in REFERENCE_FIELDS if key in reference} for reference in found], kind


def combine(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="drama_voice.py combine", description="Combine voice files into one whose references are spoken in the order given.")
    parser.add_argument("voices", type=Path, nargs="+", help="voice files, the voice's own last, as a request's references precede it")
    parser.add_argument("--out", type=Path, required=True, help="the combined voice file; its stem is the voice's name")
    parser.add_argument("--source-note", required=True, help="where the combined voice comes from; for consented, where the consent is recorded")
    args = parser.parse_args(argv)
    note = args.source_note.strip(string.whitespace)
    if not note:
        raise SystemExit("--source-note must say where the voice comes from")
    found = [references(path) for path in args.voices]
    kinds = {kind for _, kind in found}
    if len(kinds) != 1:
        raise SystemExit(f"a voice file has one source, and these say {', '.join(sorted(kinds))}: combine voices of one kind")
    combined = [reference for entries, _ in found for reference in entries]
    if len(combined) < 2:
        raise SystemExit("one reference is a single-reference voice file already: name at least two")
    voice = {"format": SEVERAL, "source": {"kind": kinds.pop(), "note": note}, "references": combined}
    args.out.write_text(json.dumps(voice, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [{key: value for key, value in reference.items() if key != "vq_codes"} for reference in combined]
    print(json.dumps({"format": SEVERAL, "source": voice["source"], "references": summary}, ensure_ascii=False))
    return 0


def main() -> int:
    if sys.argv[1:2] == ["combine"]:
        return combine(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog="drama_voice.py combine A.json B.json ... --out NAME.json combines voice files")
    encoder = parser.add_mutually_exclusive_group()
    encoder.add_argument("--native", help="a tensorfold-native with voice encode; default: the one on PATH")
    encoder.add_argument("--reference", type=Path, help="the pinned sglang-omni checkout: the reference's encoder")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the Drama 3 checkpoint holding codec.pth")
    parser.add_argument("--audio", type=Path, required=True, help="the reference clip, at most 60 s (drama_voice_clip.py prepares one)")
    parser.add_argument("--text", type=Path, required=True, help="the clip's exact transcript (UTF-8; outer white space dropped)")
    parser.add_argument("--out", type=Path, required=True, help="the voice file; its stem is the voice's name")
    parser.add_argument("--source-kind", choices=KINDS, required=True, help="whose voice the clip is")
    parser.add_argument("--source-note", required=True, help="where it comes from; for consented, where the consent is recorded")
    parser.add_argument("--device", default="cuda", help="the reference encoder's device")
    args = parser.parse_args()
    if args.text.name.endswith(".draft.txt"):
        raise SystemExit(f"{args.text} is a Whisper draft: correct it against the clip and save it as {args.text.name[: -len('.draft.txt')]}.txt")
    transcript = args.text.read_text(encoding="utf-8").strip()
    if not transcript:
        raise SystemExit(f"{args.text} holds no transcript")
    # the loader trims the same ASCII white space, so both refuse the same blank notes
    note = args.source_note.strip(string.whitespace)
    if not note:
        raise SystemExit("--source-note must say where the voice comes from")
    if args.reference:
        voice = encode_reference(args.reference, args.checkpoint, args.audio, transcript, args.device)
    else:
        binary = args.native or shutil.which("tensorfold-native")
        if binary is None:
            raise SystemExit("no tensorfold-native on PATH: pass --native BINARY, or --reference SGLANG_OMNI_CHECKOUT")
        if not encodes_voices(binary):
            raise SystemExit(f"{binary} has no `voice encode` (an older build): pass --native BINARY, or --reference SGLANG_OMNI_CHECKOUT")
        voice = encode_native(binary, args.checkpoint, args.audio, transcript, args.out)
    voice["source"] = {"kind": args.source_kind, "note": note}
    args.out.write_text(json.dumps(voice, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in voice.items() if key != "vq_codes"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
