"""drama_kl.py on hand-built probe outputs: zero for identical runs, the closed form for a known shift, refusals."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import drama_kl

COLUMNS = drama_kl.COLUMNS
BAD = [math.nan, math.inf, -math.inf]


def codes(frames: int, salt: int = 0) -> bytes:
    return np.arange(salt, salt + frames * 10, dtype="<u4").tobytes()


def output(root: Path, name: str, runs: dict, tower: dict | None = None, edit=None) -> Path:
    """A probe output of EOS-ended `runs` (id -> logits rows, frames), `tower` NLL rows, manifest changed by `edit`."""
    out = root / name
    for sub in ("logits", "codes", "tower"):
        (out / sub).mkdir(parents=True)
    entries = []
    for rid, (logits, frames) in runs.items():
        np.asarray(logits, dtype="<f4").tofile(out / "logits" / f"{rid}.f32")
        data = codes(frames)
        (out / "codes" / f"{rid}.u32").write_bytes(data)
        entries.append({"id": rid, "prompt_rows": 5, "prompt_sha256": "p", "codes_sha256": hashlib.sha256(data).hexdigest(),
                        "frames": frames, "steps": frames + 1, "finish": "eos"})
    for rid, nll in (tower or {}).items():
        np.asarray(nll, dtype="<f4").tofile(out / "tower" / f"{rid}.f32")
    if edit:
        edit(entries)
    (out / "manifest.json").write_text(json.dumps({"requests": entries}))
    return out


class KlTests(unittest.TestCase):
    def refuses(self, why: str, a: dict, b: dict, **kw):
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(SystemExit, why):
            root = Path(folder)
            drama_kl.compare(output(root, "a", a, kw.get("tower_a"), kw.get("edit_a")),
                             output(root, "b", b, kw.get("tower_b"), kw.get("edit_b")))

    def test_identical_outputs_have_zero_kl_and_full_agreement(self):
        logits = np.random.default_rng(7).normal(size=(5, COLUMNS)) * 4
        with tempfile.TemporaryDirectory() as folder:
            report = drama_kl.compare(output(Path(folder), "a", {"r": (logits, 4)}),
                                      output(Path(folder), "b", {"r": (logits, 4)}))
        r = report["requests"]["r"]
        self.assertEqual(r["rows"], 5)
        self.assertAlmostEqual(r["max"], 0.0, places=12)
        self.assertEqual(r["frame0_kl"], 0.0)
        self.assertEqual(r["top1_agreement"], 1.0)
        self.assertNotIn("tower", r)
        self.assertEqual(report["pooled"]["rows"], 5)

    def test_known_shift_matches_the_closed_form_frame_zero_and_tower_deltas(self):
        a = np.zeros((3, COLUMNS))
        b = np.zeros((3, COLUMNS))
        b[0, 0] = math.log(2)  # Q doubles column 0's odds against uniform P, at frame 0
        a[2, 5] = 1.0
        b[2, 7] = 1.0  # top-1 differs
        p, q0, q = 1 / COLUMNS, 2 / (COLUMNS + 1), 1 / (COLUMNS + 1)
        doubled = p * math.log(p / q0) + (COLUMNS - 1) * p * math.log(p / q)
        with tempfile.TemporaryDirectory() as folder:
            report = drama_kl.compare(output(Path(folder), "a", {"x": (a, 2)}, {"x": [[1.0] * 9, [2.0] * 9]}),
                                      output(Path(folder), "b", {"x": (b, 2)}, {"x": [[1.5] * 9, [2.0] * 9]}))
        r = report["requests"]["x"]
        shift = drama_kl.kl_rows(a[2:], b[2:])[0]
        self.assertAlmostEqual(r["frame0_kl"], doubled, places=9)
        self.assertAlmostEqual(r["mean"], (doubled + shift) / 3, places=9)
        self.assertEqual(r["worst"][-1], {"row": 1, "kl": 0.0, "top1": [0, 0]})
        self.assertAlmostEqual(r["top1_agreement"], 2 / 3)
        self.assertEqual([w["top1"] for w in r["worst"] if w["row"] == 2], [[5, 7]])
        self.assertEqual(r["tower"]["delta_mean_nll"], [0.25] * 9)
        self.assertAlmostEqual(report["frame0"]["mean"], doubled, places=9)

    def test_non_finite_logits_or_tower_nll_in_either_run_are_refused_by_file_and_row(self):
        good = np.zeros((3, COLUMNS))
        nll = [[1.0] * 9, [2.0] * 9]
        for value in BAD:
            bad = good.copy()
            bad[1, 40] = value
            self.refuses(r"a/logits/r\.f32: row 1 column 40", {"r": (bad, 2)}, {"r": (good, 2)})
            self.refuses(r"b/logits/r\.f32: row 1 column 40", {"r": (good, 2)}, {"r": (bad, 2)})
            bad_nll = [[1.0] * 9, [2.0] * 8 + [value]]
            self.refuses(r"a/tower/r\.f32: row 1 column 8", {"r": (good, 2)}, {"r": (good, 2)}, tower_a={"r": bad_nll},
                         tower_b={"r": nll})
            self.refuses(r"b/tower/r\.f32: row 1 column 8", {"r": (good, 2)}, {"r": (good, 2)}, tower_a={"r": nll},
                         tower_b={"r": bad_nll})

    def test_files_that_disagree_with_their_manifest_are_refused(self):
        logits = np.zeros((4, COLUMNS))
        # truncated logits: both runs short of the manifest's steps
        self.refuses(r"logits/r\.f32: 12291 floats, the manifest says 4 rows", {"r": (logits[:3], 3)}, {"r": (logits[:3], 3)},
                     edit_a=lambda e: e[0].update(steps=4, frames=3), edit_b=lambda e: e[0].update(steps=4, frames=3))

        def stale(entries):  # same frames, a digest of other codes
            entries[0]["codes_sha256"] = hashlib.sha256(codes(3, salt=1)).hexdigest()
        self.refuses(r"a/codes/r\.u32: does not match the manifest's codes_sha256", {"r": (logits, 3)}, {"r": (logits, 3)},
                     edit_a=stale, edit_b=stale)
        self.refuses(r"tower/r\.f32: 18 floats, the manifest says 3 rows of 9", {"r": (logits, 3)}, {"r": (logits, 3)},
                     tower_a={"r": [[1.0] * 9] * 2})
        self.refuses("4 steps disagree with 3 frames ending length", {"r": (logits, 3)}, {"r": (logits, 3)},
                     edit_a=lambda e: e[0].update(finish="length"))
        self.refuses("5 steps disagree with 3 frames ending eos", {"r": (logits, 3)}, {"r": (logits, 3)},
                     edit_b=lambda e: e[0].update(steps=5))
        self.refuses("misaligned runs, prompt_rows 5 against 6", {"r": (logits, 3)}, {"r": (logits, 3)},
                     edit_b=lambda e: e[0].update(prompt_rows=6))

    def test_files_with_a_partial_trailing_row_are_refused(self):
        logits = np.zeros((4, COLUMNS))
        nll = [[1.0] * 9] * 3
        for sub, name, why in (("logits", "r.f32", r"a/logits/r\.f32: 65554 bytes, not whole rows of 4097"),
                               ("tower", "r.f32", r"a/tower/r\.f32: 110 bytes, not whole rows of 9"),
                               ("codes", "r.u32", r"a/codes/r\.u32: 122 bytes, the manifest says 3 frames")):
            def tail(entries, sub=sub, name=name):  # two stray bytes after the last whole row, as a torn write leaves
                path = Path(self.root) / "a" / sub / name
                path.write_bytes(path.read_bytes() + b"\0\0")
            with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(SystemExit, why):
                self.root = folder
                drama_kl.compare(output(Path(folder), "a", {"r": (logits, 3)}, {"r": nll}, tail),
                                 output(Path(folder), "b", {"r": (logits, 3)}, {"r": nll}))

    def test_misaligned_or_duplicated_request_sets_are_refused(self):
        logits = np.zeros((4, COLUMNS))
        self.refuses("misaligned runs, codes_sha256", {"r": (logits, 3)}, {"r": (logits, 3)},
                     edit_b=lambda e: e[0].update(codes_sha256="0" * 64))
        self.refuses("the runs hold different requests: s in only one", {"r": (logits, 3)}, {"r": (logits, 3), "s": (logits, 3)})
        self.refuses("the runs hold different requests: r, s", {"r": (logits, 3)}, {"s": (logits, 3)})
        self.refuses("request r appears twice", {"r": (logits, 3)}, {"r": (logits, 3)}, edit_a=lambda e: e.append(dict(e[0])))


if __name__ == "__main__":
    unittest.main()
