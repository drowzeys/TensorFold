"""drama_trace_same.py on hand-built trace outputs: same despite identity and timing, any computed difference found."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import drama_trace_same


def output(root: Path, name: str, data: dict[str, bytes], **fields) -> Path:
    """A trace output holding `data` (file -> bytes), listed as one capture each and one case's finals."""
    out = root / name
    out.mkdir()
    for file, raw in data.items():
        (out / file).write_bytes(raw)
    listed = [{"file": f, "sha256": hashlib.sha256(b).hexdigest(), "shape": [1, len(b) // 2]} for f, b in data.items()]
    m = {"producer": "tf-drama-tower-trace", "evidence_valid": True, "checkpoint": str(out), "binary_sha256": name,
         "tower_source_sha256": name, "captures": listed[:-1],
         "cases": [{"rows": 1, "timing_s": {"traced": len(name)}, "finals": {"traced": {"hidden": listed[-1]}}}],
         "allocations": {"cuda_device_bytes": 100}}
    m.update(fields)
    (out / "manifest.json").write_text(json.dumps(m))
    return out


DATA = {"native-1-layer-00-qkv.bf16": b"\x01\x02\x03\x04", "native-1-traced-hidden.bf16": b"\x05\x06"}


class Same(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_paths_and_timing_do_not_count(self):
        a = output(self.root, "base", DATA)
        b = output(self.root, "candidate-binary", DATA)
        self.assertEqual(drama_trace_same.differences(a, b), [])

    def test_a_capture_or_final_that_differs_is_named(self):
        a = output(self.root, "a", DATA)
        b = output(self.root, "b", {**DATA, "native-1-traced-hidden.bf16": b"\x05\x07"})
        was, now = (hashlib.sha256(raw).hexdigest() for raw in (b"\x05\x06", b"\x05\x07"))
        self.assertEqual(drama_trace_same.differences(a, b), [f"native-1-traced-hidden.bf16: sha256 {was} != {now}"])

    def test_bytes_must_match_their_manifest(self):
        a = output(self.root, "a", DATA)
        b = output(self.root, "b", DATA)
        (b / "native-1-layer-00-qkv.bf16").write_bytes(b"\x00\x02\x03\x04")
        self.assertEqual(drama_trace_same.differences(a, b), [f"{b / 'native-1-layer-00-qkv.bf16'}: bytes do not match its manifest"])

    def test_allocations_and_missing_files_count(self):
        a = output(self.root, "a", DATA)
        b = output(self.root, "b", {"native-1-traced-hidden.bf16": DATA["native-1-traced-hidden.bf16"]}, allocations={"cuda_device_bytes": 101})
        self.assertEqual(drama_trace_same.differences(a, b), [
            "native-1-layer-00-qkv.bf16: only in A", "manifests differ outside files, identity and timing"])

    def test_an_invalid_trace_is_refused(self):
        a = output(self.root, "a", DATA)
        b = output(self.root, "b", DATA, evidence_valid=False)
        with self.assertRaises(SystemExit):
            drama_trace_same.differences(a, b)


if __name__ == "__main__":
    unittest.main()
