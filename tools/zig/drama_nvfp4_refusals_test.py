"""drama_nvfp4_refusals.py on a tiny fake checkpoint: variants, hard links, untouched originals, the verdict."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import drama_nvfp4_refusals as r


def shard(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, data = {}, b""
    for name, (dtype, shape, value) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(value)]}
        data += value
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + data)


def checkpoint(root: Path) -> Path:
    ck = root / "ck"
    ck.mkdir()
    experts = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        experts[r.tensor(1, 7, proj, "weight")] = ("U8", [4, 8], bytes(range(32)))
        experts[r.tensor(1, 7, proj, "weight_scale")] = ("F8_E4M3", [4, 1], b"\x38" * 4)
        experts[r.tensor(1, 7, proj, "weight_scale_2")] = ("F32", [], struct.pack("<f", 1.5))
    shard(ck / "model-00002.safetensors", experts)
    shard(ck / "model-00001.safetensors", {"model.language_model.norm.weight": ("BF16", [2], b"\0" * 4)})
    names = {n: "model-00002.safetensors" for n in experts} | {"model.language_model.norm.weight": "model-00001.safetensors"}
    (ck / r.INDEX).write_text(json.dumps({"weight_map": names}))
    (ck / "config.json").write_text("{}")
    return ck


def split(root: Path) -> Path:
    """Expert 7's gate, up and down in shards a, b and c, down's global alone in shard d."""
    ck = root / "split"
    ck.mkdir()
    shards: dict[str, dict] = {s: {} for s in "abcd"}
    for proj, s in (("gate_proj", "a"), ("up_proj", "b"), ("down_proj", "c")):
        shards[s][r.tensor(1, 7, proj, "weight")] = ("U8", [4, 8], bytes(range(32)))
        shards[s][r.tensor(1, 7, proj, "weight_scale")] = ("F8_E4M3", [4, 1], b"\x38" * 4)
        shards["d" if proj == "down_proj" else s][r.tensor(1, 7, proj, "weight_scale_2")] = ("F32", [], struct.pack("<f", 1.5))
    names = {}
    for s, tensors in shards.items():
        shard(ck / f"model-{s}.safetensors", tensors)
        names |= {n: f"model-{s}.safetensors" for n in tensors}
    (ck / r.INDEX).write_text(json.dumps({"weight_map": names}))
    return ck


class Variants(unittest.TestCase):
    def test_each_variant_rewrites_one_shard_and_links_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = checkpoint(Path(tmp))
            before = {f.name: f.read_bytes() for f in ck.iterdir()}
            for v in r.VARIANTS:
                out = Path(tmp) / v.name
                r.make(ck, out, v, 1, 7)
                self.assertEqual(os.stat(out / "model-00001.safetensors").st_ino, os.stat(ck / "model-00001.safetensors").st_ino)
                self.assertEqual(os.stat(out / "config.json").st_ino, os.stat(ck / "config.json").st_ino)
                self.assertNotEqual(os.stat(out / "model-00002.safetensors").st_ino, os.stat(ck / "model-00002.safetensors").st_ino)
                header, data = r.read_header(out / "model-00002.safetensors")
                index = json.loads((out / r.INDEX).read_text())["weight_map"]
                self.assertEqual(set(header), set(index) - {"model.language_model.norm.weight"}, v.name)
                raw = (out / "model-00002.safetensors").read_bytes()
                if v.name == "scale-0x7f":
                    begin = header[r.tensor(1, 7, "up_proj", "weight_scale")]["data_offsets"][0]
                    self.assertEqual(raw[data + begin], 0x7F)
                if v.name == "global-negative":
                    begin = header[r.tensor(1, 7, "down_proj", "weight_scale_2")]["data_offsets"][0]
                    self.assertEqual(struct.unpack("<f", raw[data + begin:data + begin + 4])[0], -1.0)
                if v.name == "dropped-companion":
                    self.assertNotIn(r.tensor(1, 7, "down_proj", "weight_scale_2"), header)
                if v.name == "fp8-projection":
                    self.assertEqual(header[r.tensor(1, 7, "up_proj", "weight")]["shape"], [4, 16])
                if v.name == "scale-0xff":
                    self.assertEqual(raw[data + header[r.tensor(1, 7, "down_proj", "weight_scale")]["data_offsets"][0]], 0xFF)
                if v.name == "companion-dtype":
                    self.assertEqual(header[r.tensor(1, 7, "gate_proj", "weight_scale")]["dtype"], "U8")
                if v.name == "companion-shape":
                    self.assertEqual(header[r.tensor(1, 7, "up_proj", "weight_scale")]["shape"], [1, 4])
            self.assertEqual(before, {f.name: f.read_bytes() for f in ck.iterdir()})

    def test_each_variant_rewrites_the_shard_holding_its_target(self):
        target = {"dropped-companion": "d", "leftover-scale-inv": "a", "input-scale": "a", "scale-0x7f": "b",
                  "global-zero": "d", "global-negative": "d", "global-nan": "d", "fp8-projection": "b",
                  "scale-0xff": "c", "companion-dtype": "a", "companion-shape": "b"}
        with tempfile.TemporaryDirectory() as tmp:
            ck = split(Path(tmp))
            before = {f.name: f.read_bytes() for f in ck.iterdir()}
            for v in r.VARIANTS:
                out = Path(tmp) / v.name
                r.make(ck, out, v, 1, 7)
                rewritten = {f"model-{target[v.name]}.safetensors", r.INDEX}
                for f in ck.iterdir():
                    linked = os.stat(out / f.name).st_ino == os.stat(f).st_ino
                    self.assertEqual(linked, f.name not in rewritten, f"{v.name}: {f.name}")
                header, data = r.read_header(out / f"model-{target[v.name]}.safetensors")
                raw = (out / f"model-{target[v.name]}.safetensors").read_bytes()
                if v.name == "scale-0x7f":
                    self.assertEqual(raw[data + header[r.tensor(1, 7, "up_proj", "weight_scale")]["data_offsets"][0]], 0x7F)
                if v.name == "global-nan":
                    begin = header[r.tensor(1, 7, "down_proj", "weight_scale_2")]["data_offsets"][0]
                    self.assertNotEqual(*struct.unpack("<ff", raw[data + begin:data + begin + 4] * 2))
                if v.name == "dropped-companion":
                    self.assertNotIn(r.tensor(1, 7, "down_proj", "weight_scale_2"), header)
                if v.name == "input-scale":
                    self.assertIn(r.tensor(1, 7, "gate_proj", "input_scale"), header)
                if v.name == "fp8-projection":
                    self.assertEqual(header[r.tensor(1, 7, "up_proj", "weight")]["dtype"], "F8_E4M3")
                if v.name == "scale-0xff":
                    self.assertEqual(raw[data + header[r.tensor(1, 7, "down_proj", "weight_scale")]["data_offsets"][0]], 0xFF)
                if v.name == "companion-dtype":
                    self.assertEqual(header[r.tensor(1, 7, "gate_proj", "weight_scale")]["dtype"], "U8")
                if v.name == "companion-shape":
                    self.assertEqual(header[r.tensor(1, 7, "up_proj", "weight_scale")]["shape"], [1, 4])
            self.assertEqual(before, {f.name: f.read_bytes() for f in ck.iterdir()})

    def test_run_passes_only_named_refusals_that_keep_no_device_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = checkpoint(Path(tmp))
            fake = Path(tmp) / "fake"
            errors = {v.name: v.error for v in r.VARIANTS}
            fake.write_text(f"#!{sys.executable}\nimport sys, os\nerr = {errors!r}[os.path.basename(sys.argv[2])]\n"
                            "print(f'RESULT refused: {err}; device bytes held after the refusal 7 (before 7)')\nsys.exit(1)\n")
            fake.chmod(0o755)
            self.assertEqual(r.run(ck, fake, Path(tmp) / "work", 1, 7, set()), 0)
            fake.write_text(f"#!{sys.executable}\nprint('RESULT refused: MissingDramaTensor; device bytes held after the refusal 9 (before 7)')\n"
                            "raise SystemExit(1)\n")
            self.assertEqual(r.run(ck, fake, Path(tmp) / "work", 1, 7, {"dropped-companion"}), 1)


if __name__ == "__main__":
    unittest.main()
