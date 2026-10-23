"""Capture drama_voice_fixture.json on the CPU: the reference's prompt rows and refusals for synthetic codes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "zig/src/server/drama_voice_fixture.json"
LIGHTHOUSE = "The old lighthouse on the northern point has guided ships through fog and storm for more than a century."


def grid(frames: int, salt: int) -> list[list[int]]:
    """Synthetic [10, frames] codes reaching both ends of [0, 4096)."""
    return [[(salt * 977 + book * 409 + frame * 1531) % 4096 if (book + frame) % 7 else 4095 * (frame % 2) for frame in range(frames)] for book in range(10)]


ROWS = [
    ("plain", "Hello there.", []),
    ("one", "Hello there.", [{"text": "Hi, this is a test.", "vq_codes": grid(3, 1)}]),
    ("two", "<|speaker:0|> Good morning, everyone.", [{"text": "First clip.", "vq_codes": grid(2, 2)}, {"text": "  <|speaker:0|> second clip", "vq_codes": grid(4, 3)}]),
    ("empty-transcript", "Short.", [{"text": "", "vq_codes": grid(1, 4)}]),
    ("missing-transcript", "Short.", [{"vq_codes": grid(2, 5)}]),
    ("white-space", "\u3000mid <|speaker:0|> tag", [{"text": "\u3000\n<|speaker:0|>\tspaced <|speaker:0|> twice\n", "vq_codes": grid(2, 6)}]),
    ("content-tokens", "hi <|audio_separator|> there", [{"text": "a <|fim_prefix|> b", "vq_codes": grid(1, 7)}]),
    ("lighthouse", "Attention please: the next train departs at nine.", [{"text": LIGHTHOUSE, "vq_codes": grid(20, 8)}]),
    ("point-tags", "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.", [{"text": LIGHTHOUSE, "vq_codes": grid(5, 10)}]),
]

# Long-form segments: the voice's reference, then earlier segments as references, the latest last.
CARRY = [
    ("voice-and-carry", "And the second sentence follows it.", {"text": LIGHTHOUSE, "vq_codes": grid(20, 11)},
     [{"text": "The first segment of a long reply ends here.", "vq_codes": grid(12, 12)}]),
    ("voice-and-two-carries", "Third, and last.", {"text": LIGHTHOUSE, "vq_codes": grid(6, 13)},
     [{"text": "One.", "vq_codes": grid(3, 14)}, {"text": "<|speaker:0|> Two, then.", "vq_codes": grid(4, 15)}]),
    ("carry-only", "Plain voice, second part.", None, [{"text": "Plain voice, first part!", "vq_codes": grid(5, 16)}]),
]

TEN = grid(1, 9)
REFUSALS = [
    ("references-object", "hi", {"text": "x"}),
    ("references-scalar", "hi", [1]),
    ("text-number", "hi", [{"text": 3, "vq_codes": TEN}]),
    ("other-speaker", "hi", [{"text": "<|speaker:3|> x", "vq_codes": TEN}]),
    ("other-speaker-input", "<|speaker:2|> hi", [{"text": "<|speaker:5|> x", "vq_codes": TEN}]),
    ("no-codes", "hi", [{"text": "x"}]),
    ("null-codes", "hi", [{"text": "x", "vq_codes": None}]),
    ("ragged", "hi", [{"text": "x", "vq_codes": [[1, 2], [3]]}]),
    ("empty-grid", "hi", [{"text": "x", "vq_codes": []}]),
    ("empty-row", "hi", [{"text": "x", "vq_codes": [[]]}]),
    ("float", "hi", [{"text": "x", "vq_codes": [[1.5]]}]),
    ("bool", "hi", [{"text": "x", "vq_codes": [[True]]}]),
    ("flat", "hi", [{"text": "x", "vq_codes": [1, 2]}]),
    ("past-int64", "hi", [{"text": "x", "vq_codes": [[2**63]]}]),
    ("codebooks", "hi", [{"text": "x", "vq_codes": [[1, 2]]}]),
    ("too-high", "hi", [{"text": "x", "vq_codes": [[4096]] * 10}]),
    ("negative", "hi", [{"text": "x", "vq_codes": [[-1]] * 10}]),
    ("huge-in-range-shape", "hi", [{"text": "x", "vq_codes": [[-(2**63)]] * 10}]),
    ("structural", "hi", [{"text": "a <|im_end|> b", "vq_codes": TEN}]),
    ("bad-tag", "hi", [{"text": "<|speaker:01|> x", "vq_codes": TEN}]),
    ("bad-tag-inside", "hi", [{"text": "a <|speaker:99|>", "vq_codes": TEN}]),
    ("grid-before-text", "hi", [{"text": "<|voice|>", "vq_codes": TEN}, {"text": "y", "vq_codes": [[1], [2, 3]]}]),
    ("text-before-shape", "hi", [{"text": "<|voice|>", "vq_codes": [[1]]}]),
    ("second-reference", "hi", [{"text": "fine", "vq_codes": TEN}, {"text": "fine", "vq_codes": [[7]] * 9}]),
    ("input-empty", " \u3000", [{"text": "fine", "vq_codes": TEN}]),
    ("input-structural", "say <|semantic:3|>", [{"text": "fine", "vq_codes": TEN}]),
]


def rows_sha256(rows: list[list[int]]) -> str:
    """The prompt rows as little-endian int64, row-major, as the native test hashes them."""
    return hashlib.sha256(b"".join(v.to_bytes(8, "little", signed=True) for row in rows for v in row)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True, help="the pinned sglang-omni checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="the Drama 3 checkpoint (tokenizer.json, config.json)")
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference))
    from tokenizers import Tokenizer

    from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
    from sglang_omni.models.fishaudio_drama3.speech_request import Drama3SpeechPreprocessor
    from sglang_omni.proto.request import OmniRequest, StagePayload

    class NoAudio:
        def get_or_encode(self, *_args, **_kwargs):
            raise AssertionError("the fixture sends codes only")

    preprocess = Drama3SpeechPreprocessor(
        tokenizer=Tokenizer.from_file(str(args.checkpoint / "tokenizer.json")),
        config=FishOmniConfig.from_pretrained(str(args.checkpoint)),
        max_prompt_rows=4096,
        reference_encoder=NoAudio(),
    )

    def run(text: str, references: object):
        inputs = {"text": text, "references": references} if references else text
        payload = StagePayload(request_id="fixture", request=OmniRequest(inputs=inputs, metadata={"tts_params": {}}), data=None)
        return preprocess(payload).data.prompt_rows.tolist()

    def unconditioned(name: str, text: str) -> dict:
        """A guided draw's plain prompt: the same input without references, as the built-in voice is prompted."""
        out = run(text, None)
        return {"name": f"{name} unconditioned", "rows": out, "rows_sha256": rows_sha256(out)}

    rows = []
    for name, text, references in ROWS:
        out = run(text, references)
        rows.append({"name": name, "input": text, "references": references, "rows": out, "rows_sha256": rows_sha256(out),
                     "unconditioned": unconditioned(name, text)})
    carries = []
    for name, text, voice, carry in CARRY:
        out = run(text, ([voice] if voice else []) + carry)
        carries.append({"name": name, "input": text, "voice": voice, "carry": carry, "rows": out, "rows_sha256": rows_sha256(out),
                        "unconditioned": unconditioned(name, text)})
    refusals = []
    for name, text, references in REFUSALS:
        try:
            run(text, references)
        except ValueError as error:
            refusals.append({"name": name, "input": text, "references": references, "message": str(error)})
        else:
            raise SystemExit(f"{name}: the reference accepted it")
    tokenizer = (args.checkpoint / "tokenizer.json").read_bytes()
    doc = {
        "source": "sglang-omni Drama3SpeechPreprocessor; rows_sha256 hashes the rows as little-endian int64, row-major",
        "reference_commit": subprocess.run(["git", "-C", str(args.reference), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(),
        "rows": rows,
        "carry": carries,
        "refusals": refusals,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(rows)} prompts, {len(refusals)} refusals, sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
