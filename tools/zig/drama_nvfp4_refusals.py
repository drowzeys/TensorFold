"""Malformed copies of a converted Drama 3 NVFP4 checkpoint, each refused by the production loader by its error."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

INDEX = "model.safetensors.index.json"
HELD = "device bytes held after the refusal "


def read_header(path: Path) -> tuple[dict, int]:
    """A shard's header and where its data starts."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def write_shard(src: Path, dst: Path, header: dict, patches: dict[str, bytes]) -> None:
    """`src`'s data under `header`, each patched tensor's first bytes overwritten; `dst` must not exist."""
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    _, data = read_header(src)
    with open(src, "rb") as f, open(dst, "xb") as out:
        out.write(struct.pack("<Q", len(text)) + text)
        f.seek(data)
        shutil.copyfileobj(f, out, 64 << 20)
        for name, value in patches.items():
            out.seek(8 + len(text) + header[name]["data_offsets"][0])
            out.write(value)


class Variant:
    """One malformation of expert `expert` in layer `layer`: the tensor whose shard it rewrites, edits, its error."""

    def __init__(self, name: str, error: str, target: tuple[str, str], edit):
        self.name, self.error, self.target, self.edit = name, error, target, edit


def tensor(layer: int, expert: int, proj: str, part: str) -> str:
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{part}"


def drop(header: dict, layer: int, expert: int) -> tuple[dict[str, bytes], list[str], list[str]]:
    name = tensor(layer, expert, "down_proj", "weight_scale_2")
    del header[name]
    return {}, [], [name]


def extra(part: str, dtype: str, shape: list[int], size: int):
    def add(header: dict, layer: int, expert: int) -> tuple[dict[str, bytes], list[str], list[str]]:
        begin = header[tensor(layer, expert, "gate_proj", "weight")]["data_offsets"][0]
        name = tensor(layer, expert, "gate_proj", part)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, begin + size]}
        return {}, [name], []
    return add


def patch(proj: str, part: str, value: bytes):
    def set_bytes(header: dict, layer: int, expert: int) -> tuple[dict[str, bytes], list[str], list[str]]:
        return {tensor(layer, expert, proj, part): value}, [], []
    return set_bytes


def fp8(header: dict, layer: int, expert: int) -> tuple[dict[str, bytes], list[str], list[str]]:
    entry = header[tensor(layer, expert, "up_proj", "weight")]
    n, half = entry["shape"]
    begin = entry["data_offsets"][0]
    entry.update(dtype="F8_E4M3", shape=[n, 2 * half], data_offsets=[begin, begin + n * 2 * half])
    return {}, [], []


def retype(proj: str, part: str, dtype: str):
    """`proj`'s `part` under another dtype of the same element size: its bytes stay where they are."""
    def edit(header: dict, layer: int, expert: int) -> tuple[dict[str, bytes], list[str], list[str]]:
        header[tensor(layer, expert, proj, part)]["dtype"] = dtype
        return {}, [], []
    return edit


def transpose(proj: str, part: str):
    """`proj`'s `part` with its two dimensions swapped: the same byte count, the wrong shape."""
    def edit(header: dict, layer: int, expert: int) -> tuple[dict[str, bytes], list[str], list[str]]:
        entry = header[tensor(layer, expert, proj, part)]
        entry["shape"] = entry["shape"][::-1]
        return {}, [], []
    return edit


# extra() adds a gate companion, so it rewrites gate_proj.weight's shard, where the new entry's bytes lie
VARIANTS = [
    Variant("dropped-companion", "MissingDramaTensor", ("down_proj", "weight_scale_2"), drop),
    Variant("leftover-scale-inv", "UnusedDramaTensors", ("gate_proj", "weight"), extra("weight_scale_inv", "F32", [4, 16], 256)),
    Variant("input-scale", "UnusedDramaTensors", ("gate_proj", "weight"), extra("input_scale", "F32", [], 4)),
    Variant("scale-0x7f", "BadDramaNvfp4Scale", ("up_proj", "weight_scale"), patch("up_proj", "weight_scale", b"\x7f")),
    Variant("global-zero", "BadDramaNvfp4Global", ("down_proj", "weight_scale_2"), patch("down_proj", "weight_scale_2", struct.pack("<f", 0.0))),
    Variant("global-negative", "BadDramaNvfp4Global", ("down_proj", "weight_scale_2"), patch("down_proj", "weight_scale_2", struct.pack("<f", -1.0))),
    Variant("global-nan", "BadDramaNvfp4Global", ("down_proj", "weight_scale_2"), patch("down_proj", "weight_scale_2", struct.pack("<f", float("nan")))),
    Variant("fp8-projection", "UnexpectedDramaTensor", ("up_proj", "weight"), fp8),
    Variant("scale-0xff", "BadDramaNvfp4Scale", ("down_proj", "weight_scale"), patch("down_proj", "weight_scale", b"\xff")),
    Variant("companion-dtype", "UnexpectedDramaTensor", ("gate_proj", "weight_scale"), retype("gate_proj", "weight_scale", "U8")),
    Variant("companion-shape", "UnexpectedDramaTensor", ("up_proj", "weight_scale"), transpose("up_proj", "weight_scale")),
]


def stamp(path: Path) -> tuple[int, int, int]:
    s = path.stat()
    return s.st_ino, s.st_size, s.st_mtime_ns


def make(checkpoint: Path, out: Path, v: Variant, layer: int, expert: int) -> None:
    """`out`: hard links to every file of `checkpoint` but the one shard (and the index) the variant rewrites."""
    index = json.loads((checkpoint / INDEX).read_text())
    shard = index["weight_map"][tensor(layer, expert, *v.target)]
    header, _ = read_header(checkpoint / shard)
    patches, added, removed = v.edit(header, layer, expert)
    out.mkdir()
    for f in sorted(checkpoint.iterdir()):
        if f.is_file() and f.name not in (shard, INDEX):
            os.link(f, out / f.name)  # same filesystem required: a copy would not be cheap
    write_shard(checkpoint / shard, out / shard, header, patches)
    for name in added:
        index["weight_map"][name] = shard
    for name in removed:
        del index["weight_map"][name]
    (out / INDEX).write_text(json.dumps(index, indent=2))


def held(output: str) -> tuple[int, int] | None:
    """The refusal line's (after, before) device bytes."""
    for line in output.splitlines():
        if HELD in line:
            after, before = line.split(HELD, 1)[1].split(" (before ")
            return int(after), int(before.rstrip(")"))
    return None


def run(checkpoint: Path, binary: Path, work: Path, layer: int, expert: int, only: set[str]) -> int:
    """Each variant refused with its error and no device bytes kept; the originals' inode, size and mtime unchanged."""
    before = {f: stamp(f) for f in checkpoint.iterdir() if f.is_file()}
    work.mkdir()
    failed = 0
    for v in VARIANTS:
        if only and v.name not in only:
            continue
        out = work / v.name
        make(checkpoint, out, v, layer, expert)
        done = subprocess.run([str(binary), "drama-nvfp4-load", str(out), str(layer)], capture_output=True, text=True)
        text = done.stdout + done.stderr
        kept = held(text)
        ok = done.returncode != 0 and v.error in text and kept is not None and kept[0] == kept[1]
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {v.name}: exit {done.returncode}, expected {v.error}, held {kept}")
        if not ok:
            print(text[-4000:])
        shutil.rmtree(out)
    work.rmdir()
    changed = [f.name for f, s in before.items() if stamp(f) != s]
    if changed:
        print(f"FAIL originals changed: {changed}")
        failed += 1
    print(f"RESULT {failed} failed")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="a converted checkpoint whose layer LAYER is NVFP4")
    p.add_argument("binary", type=Path, help="tf-cuda-test")
    p.add_argument("work", type=Path, help="new directory on the checkpoint's filesystem, removed when done")
    p.add_argument("--layer", type=int, default=1, help="layer to malform (default 1: after layer 0's pack)")
    p.add_argument("--expert", type=int, default=7)
    p.add_argument("--only", action="append", default=[], choices=[v.name for v in VARIANTS])
    a = p.parse_args()
    return run(a.checkpoint, a.binary, a.work, a.layer, a.expert, set(a.only))


if __name__ == "__main__":
    sys.exit(main())
