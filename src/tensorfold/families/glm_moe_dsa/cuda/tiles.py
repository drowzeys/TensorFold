"""Saved EXL3 tile tables: the (K splits, warps) every dense EXL3 linear of a rank runs with, kept across restarts.

``fused.tune_linears`` / ``tune_groups`` time tiles at each load, and the picks differ from boot to boot. Every pick
is exact within a boot, but a different pick sums in another order, so the same greedy request can decode other
tokens after a restart. ``TF_GLM53_TILES=save:PATH`` writes the table a boot runs (after rank 0's picks are shared);
``load:PATH`` runs a saved table instead of timing, refused unless every linear's shape matches.
"""

from __future__ import annotations

import hashlib
import json
import os

ENV = "TF_GLM53_TILES"


def spec(value: str | None = None) -> tuple[str, str]:
    """("", "") when unset, else ("save" or "load", path)."""
    value = os.environ.get(ENV, "") if value is None else value
    if not value:
        return "", ""
    mode, _, path = value.partition(":")
    if mode not in ("save", "load") or not path:
        raise ValueError(f"{ENV}={value!r}: save:PATH or load:PATH")
    return mode, path


def rows(lins) -> list[list]:
    """[k, n, bits, codebook, layout, K splits, warps] for each linear, in the engine's order (``Weights.tunable``)."""
    return [[int(x.k), int(x.n), float(x.bits), str(x.codebook), str(x.layout), int(x.split[0]), int(x.split[1])]
            for x in lins]


def _blob(table: list[list]) -> str:
    return json.dumps({"count": len(table), "linears": table}, sort_keys=True)


def digest(lins) -> str:
    """sha256 of the table as ``save`` writes it: equal digests, equal tiles."""
    return hashlib.sha256(_blob(rows(lins)).encode()).hexdigest()


def save(lins, path: str) -> str:
    """Write the table (atomically: ranks sharing a file system may write the same one); returns its sha256."""
    blob = _blob(rows(lins))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(blob)
    os.replace(tmp, path)
    return hashlib.sha256(blob.encode()).hexdigest()


def load(lins, path: str) -> tuple[str, int]:
    """Give every linear its saved tile; returns (sha256 of the file, how many tiles changed). Nothing changes unless
    the table has exactly these linears, in order, with these shapes, widths, codebooks and layouts."""
    with open(path, "rb") as f:
        raw = f.read()
    table = json.loads(raw)["linears"]
    if len(table) != len(lins):
        raise ValueError(f"{path}: a table of {len(table)} linears, this engine has {len(lins)}: refusing it")
    for i, (row, mine) in enumerate(zip(table, rows(lins))):
        if len(row) != 7 or [int(row[0]), int(row[1]), float(row[2]), str(row[3]), str(row[4])] != mine[:5]:
            raise ValueError(f"{path}: linear {i} is {row[:5]} in the table and {mine[:5]} here: refusing it")
    changed = 0
    for x, row in zip(lins, table):
        tile = (int(row[5]), int(row[6]))
        changed += tuple(x.split) != tile
        x.split = tile
    return hashlib.sha256(raw).hexdigest(), changed


def after_load(fw, rank: int) -> None:
    """After the ranks share tiles: save the table when asked, and print the sha of the tiles this rank runs."""
    mode, path = spec()
    sha = save(fw.tunable, path) if mode == "save" else digest(fw.tunable)
    print(f"[tensorfold] rank {rank}: tiles in use sha {sha[:16]}" + (f" (saved to {path})" if mode == "save" else ""),
          flush=True)
