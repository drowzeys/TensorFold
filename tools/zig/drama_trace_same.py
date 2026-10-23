"""Two tf-drama-tower-trace outputs computed the same: every file bit-identical, manifests equal but for identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# What names the build, the paths or the clock rather than what the tower computed.
IDENTITY = {"checkpoint", "input_directory", "binary_sha256", "tower_source_sha256", "trace_source_sha256", "timing_s"}


def manifest(out: Path) -> dict:
    """A valid trace output's manifest."""
    m = json.loads((out / "manifest.json").read_text())
    if m.get("producer") != "tf-drama-tower-trace" or m.get("evidence_valid") is not True:
        raise SystemExit(f"{out}: not a valid tf-drama-tower-trace output")
    return m


def files(node, found: dict[str, str]) -> dict[str, str]:
    """Every {file, sha256} entry under `node`."""
    if isinstance(node, dict):
        if "file" in node and "sha256" in node:
            found[node["file"]] = node["sha256"]
        for v in node.values():
            files(v, found)
    elif isinstance(node, list):
        for v in node:
            files(v, found)
    return found


def computed(node):
    """`node` without its identity fields or file hashes, which `files` compares."""
    if isinstance(node, dict):
        return {k: computed(v) for k, v in node.items() if k not in IDENTITY and k != "sha256"}
    if isinstance(node, list):
        return [computed(v) for v in node]
    return node


def differences(a: Path, b: Path) -> list[str]:
    """What differs between two outputs: files listed, their bytes against each manifest, the rest of the manifests."""
    ma, mb = manifest(a), manifest(b)
    fa, fb = files(ma, {}), files(mb, {})
    found = [f"{name}: only in {'A' if name in fa else 'B'}" for name in sorted(fa.keys() ^ fb.keys())]
    found += [f"{name}: sha256 {fa[name]} != {fb[name]}" for name in sorted(fa.keys() & fb.keys()) if fa[name] != fb[name]]
    for out, listed in ((a, fa), (b, fb)):
        for name, sha in sorted(listed.items()):
            if hashlib.sha256((out / name).read_bytes()).hexdigest() != sha:
                found.append(f"{out / name}: bytes do not match its manifest")
    if computed(ma) != computed(mb):
        found.append("manifests differ outside files, identity and timing")
    return found


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path, help="trace output of the reference binary")
    p.add_argument("b", type=Path, help="trace output of the candidate binary")
    args = p.parse_args()
    found = differences(args.a, args.b)
    for line in found:
        print(line)
    count = len(files(manifest(args.a), {}))
    print(f"{'DIFFERENT' if found else 'SAME'}: {count} files, {len(found)} differences")
    raise SystemExit(1 if found else 0)


if __name__ == "__main__":
    main()
