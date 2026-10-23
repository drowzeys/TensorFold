"""The encoder seam (an explicit encoder wins, else a native that can voice encode) and combining voice files."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import drama_voice

ROWS = [[book * 7 + frame for frame in range(3)] for book in range(drama_voice.NUM_CODEBOOKS)]
TOOL = Path(__file__).with_name("drama_voice.py")
# A stand-in tensorfold-native: its --help, and for voice encode the file and summary the real one writes.
FAKE = """#!{python}
import json, sys
from pathlib import Path
args = sys.argv[1:]
if args == ["--help"]:
    print("usage: tensorfold-native --version | serve MODEL [flags]{voice}")
    sys.exit(0)
Path(sys.argv[0] + ".ran").write_text(json.dumps(args))
if "{voice}" == "" or args[:2] != ["voice", "encode"]:
    sys.exit(2)
voice = {{"format": "drama3-voice/1", "text": args[args.index("--text") + 1], "codes_sha256": "c0de", "vq_codes": {rows}}}
Path(args[args.index("--out") + 1]).write_text(json.dumps(voice))
print(json.dumps({{key: value for key, value in voice.items() if key != "vq_codes"}}))
"""


def no_reference_encoder(*_):
    raise AssertionError("the reference encoder ran")


class EncodeTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.audio, self.text, self.out = self.folder / "me.wav", self.folder / "me.txt", self.folder / "me.json"
        self.audio.write_bytes(b"RIFF")
        self.text.write_text(" Hello there. \n", encoding="utf-8")

    def native(self, encodes: bool) -> Path:
        """A tensorfold-native on its own PATH directory, with or without voice encode."""
        bin_dir = self.folder / ("new" if encodes else "old")
        bin_dir.mkdir()
        binary = bin_dir / "tensorfold-native"
        binary.write_text(FAKE.format(python=sys.executable, voice=" | voice encode CLIP.wav [flags]" if encodes else "", rows=ROWS))
        binary.chmod(0o755)
        return binary

    def argv(self, *words: str, text: Path | None = None) -> list[str]:
        return ["--checkpoint", "ckpt", "--audio", str(self.audio), "--text", str(text or self.text), "--out", str(self.out), *words]

    def run_main(self, *words: str, text: Path | None = None, path: str = "") -> dict:
        with patch("sys.argv", ["drama_voice.py", *self.argv(*words, text=text)]), patch.dict(os.environ, {"PATH": path}):
            with contextlib.redirect_stdout(io.StringIO()):
                drama_voice.main()
        return json.loads(self.out.read_text(encoding="utf-8"))

    def test_a_native_on_path_that_can_voice_encode_is_used_and_prints_one_summary(self):
        binary = self.native(encodes=True)
        command = [sys.executable, str(TOOL), *self.argv("--source-kind", "owner", "--source-note", " my own voice ")]
        done = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "PATH": str(binary.parent)})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["source"], {"kind": "owner", "note": "my own voice"})
        ran = json.loads(Path(str(binary) + ".ran").read_text())
        self.assertEqual(ran, ["voice", "encode", str(self.audio), "--model", "ckpt", "--text", "Hello there.", "--name", "me", "--out", str(self.out)])
        voice = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual((voice["codes_sha256"], voice["source"]["kind"]), ("c0de", "owner"))

    def test_an_explicit_reference_wins_over_a_native_on_path(self):
        binary = self.native(encodes=True)
        with patch.object(drama_voice, "encode", lambda *_: (ROWS, {"device": "cpu"})):
            voice = self.run_main("--reference", "sglang-omni", "--source-kind", "model-invented", "--source-note", "the model's own", path=str(binary.parent))
        self.assertFalse(Path(str(binary) + ".ran").exists())
        self.assertEqual(voice["frames"], 3)
        self.assertEqual(voice["codes_sha256"], drama_voice.codes_sha256(ROWS))
        self.assertEqual(voice["audio_sha256"], drama_voice.sha256(b"RIFF"))
        self.assertEqual(voice["source"], {"kind": "model-invented", "note": "the model's own"})

    def test_a_native_without_voice_encode_is_refused_and_never_run(self):
        binary = self.native(encodes=False)
        with patch.object(drama_voice, "encode", no_reference_encoder):
            for words, path in ((("--source-kind", "owner", "--source-note", "mine"), str(binary.parent)),
                                (("--native", str(binary), "--source-kind", "owner", "--source-note", "mine"), "")):
                with self.assertRaisesRegex(SystemExit, "has no `voice encode`"):
                    self.run_main(*words, path=path)
            with self.assertRaisesRegex(SystemExit, "no tensorfold-native on PATH"):
                self.run_main("--source-kind", "owner", "--source-note", "mine")
        self.assertFalse(Path(str(binary) + ".ran").exists())
        self.assertFalse(self.out.exists())

    def test_drafts_missing_consent_and_two_encoders_are_refused_before_encoding(self):
        draft = self.folder / "me.draft.txt"
        draft.write_text("Hello there.\n", encoding="utf-8")
        with patch.object(drama_voice, "encode", no_reference_encoder):
            with self.assertRaisesRegex(SystemExit, "Whisper draft: correct it against the clip and save it as me.txt"):
                self.run_main("--reference", "r", "--source-kind", "owner", "--source-note", "mine", text=draft)
            for blank in ("  ", "\t\n"):
                with self.assertRaisesRegex(SystemExit, "--source-note must say"):
                    self.run_main("--reference", "r", "--source-kind", "owner", "--source-note", blank)
            for words in (["--source-kind", "celebrity", "--source-note", "a film"], ["--source-note", "mine"],
                          ["--native", "tf", "--source-kind", "owner", "--source-note", "mine"]):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    self.run_main("--reference", "r", *words)
        self.assertFalse(self.out.exists())


def single(text: str, salt: int, kind: str = "model-invented") -> dict:
    """A single-reference voice file as drama_voice.py writes it, with codes and a clip digest of its own."""
    rows = [[(salt * 131 + book * 17 + frame) % drama_voice.CODEBOOK_SIZE for frame in range(2 + salt)] for book in range(drama_voice.NUM_CODEBOOKS)]
    return {"format": drama_voice.FORMAT, "text": text, "codes_sha256": drama_voice.codes_sha256(rows), "frames": len(rows[0]),
            "audio_sha256": drama_voice.sha256(bytes([salt])), "encoder": {"device": "cpu"}, "vq_codes": rows,
            "source": {"kind": kind, "note": f"take {salt}"}}


class CombineTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.folder / "duet.json"

    def write(self, name: str, voice: dict) -> str:
        (self.folder / name).write_text(json.dumps(voice, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(self.folder / name)

    def combine(self, *words: str) -> dict:
        with patch("sys.argv", ["drama_voice.py", "combine", *words, "--out", str(self.out)]), contextlib.redirect_stdout(io.StringIO()):
            drama_voice.main()
        return json.loads(self.out.read_text(encoding="utf-8"))

    def test_references_keep_their_order_codes_and_clips_under_one_source(self):
        takes = [single("First.", 1), single("Second.", 2), single("Own.", 3)]
        paths = [self.write(f"{i}.json", take) for i, take in enumerate(takes)]
        voice = self.combine(*paths, "--source-note", " three takes of one invented voice ")
        self.assertEqual(voice["format"], "drama3-voice/2")
        self.assertEqual(voice["source"], {"kind": "model-invented", "note": "three takes of one invented voice"})
        self.assertEqual([r["text"] for r in voice["references"]], ["First.", "Second.", "Own."])
        written = self.out.read_text(encoding="utf-8")
        for take, reference in zip(takes, voice["references"]):
            self.assertIn(json.dumps(take["vq_codes"]), written)
            self.assertEqual(reference["codes_sha256"], take["codes_sha256"])
            self.assertEqual(reference["audio_sha256"], take["audio_sha256"])
            self.assertNotIn("source", reference)
        # a combined file combines again, flattened in order
        again = self.write("duet-in.json", voice)
        self.out = self.folder / "trio.json"
        self.assertEqual([r["text"] for r in self.combine(again, paths[0], "--source-note", "n")["references"]], ["First.", "Second.", "Own.", "First."])

    def test_mixed_sources_bad_codes_unsourced_single_and_blank_notes_write_nothing(self):
        tampered = single("Two.", 2)
        tampered["vq_codes"][0][0] += 1
        unsourced = single("Two.", 2)
        del unsourced["source"]
        one = self.write("one.json", single("One.", 1))
        cases = [
            ([one, self.write("owner.json", single("Two.", 2, "owner"))], "one source, and these say model-invented, owner"),
            ([one, self.write("tampered.json", tampered)], "reference 0: codes_sha256 does not match vq_codes"),
            ([one, self.write("unsourced.json", unsourced)], "does not say whose voice it holds"),
            ([one], "name at least two"),
        ]
        for paths, message in cases:
            with self.assertRaisesRegex(SystemExit, message):
                self.combine(*paths, "--source-note", "n")
        with self.assertRaisesRegex(SystemExit, "--source-note must say"):
            self.combine(one, one, "--source-note", " \t")
        self.assertFalse(self.out.exists())

    def test_every_input_carries_the_attestation_the_loader_asks(self):
        one = self.write("one.json", single("One.", 1))
        for i, note in enumerate((" \t\n", "", 3, None)):
            unnoted = single("Two.", 2)
            unnoted["source"]["note"] = note
            with self.assertRaisesRegex(SystemExit, "unnoted.*source.note"):
                self.combine(one, self.write(f"unnoted{i}.json", unnoted), "--source-note", "n")
        noteless = single("Two.", 2)
        del noteless["source"]["note"]
        with self.assertRaisesRegex(SystemExit, "noteless.*source.note"):
            self.combine(one, self.write("noteless.json", noteless), "--source-note", "n")
        self.assertFalse(self.out.exists())

    def test_a_clip_digest_must_be_64_lowercase_hex_digits_as_the_loader_reads_it(self):
        one = self.write("one.json", single("One.", 1))
        good = single("Two.", 2)["audio_sha256"]
        for i, digest in enumerate((good.upper(), "g" * 64, good[:63] + " ", good[:63], 7 * 10**63)):
            bad = single("Two.", 2)
            bad["audio_sha256"] = digest
            with self.assertRaisesRegex(SystemExit, "reference 0: audio_sha256 is not a clip's sha256"):
                self.combine(one, self.write(f"bad{i}.json", bad), "--source-note", "n")
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
