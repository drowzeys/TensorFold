"""The rows / reference file of zig/src/families/glm_moe_dsa/score.zig, read and written from Python (numpy only).

  Head (64 bytes), sets [n_sets] (name[24], first_row u32, n_rows u32), rows [n_rows] (len u32, set u32, first_pos u64),
  tokens [sum of len] u32, then with topk > 0, for every position: tail f32, label_logp f32, entropy f32, reserved u32,
  ids [topk] u32, logp [topk] f32 (most likely first). A row of len tokens has len - 1 positions.
Digests are zlib.crc32 of the tokens' bytes and of the position records' bytes.
"""
import struct
import zlib

import numpy as np

MAGIC = b"GLM53RF1"
VERSION = 1
HEAD = struct.Struct("<8sIIIIIIQQQ8x")
SET = struct.Struct("<24sII")
ROW = struct.Struct("<IIQ")
assert HEAD.size == 64 and SET.size == 32 and ROW.size == 16


def pos_bytes(topk):
    return 16 + 8 * topk


def write_prefix(f, vocab, topk, sets, rows, body_digest=0):
    """sets: [(name, n_rows)] in order; rows: lists of token ids, grouped by set in that order."""
    n_pos = sum(len(r) - 1 for r in rows)
    tokens = np.concatenate([np.asarray(r, dtype="<u4") for r in rows]).tobytes()
    f.write(HEAD.pack(MAGIC, VERSION, vocab, topk, len(rows), len(sets), 0, n_pos, zlib.crc32(tokens), body_digest))
    first = 0
    for name, n in sets:
        raw = name.encode()
        if not 0 < len(raw) <= 23 or n <= 0:
            raise ValueError(f"set {name!r}: a name of 1..23 bytes and at least one row")
        f.write(SET.pack(raw, first, n))
        first += n
    if first != len(rows):
        raise ValueError("the sets do not cover the rows")
    at = 0
    si, left = 0, sets[0][1]
    for r in rows:
        if len(r) < 2:
            raise ValueError("a row needs at least two tokens")
        if left == 0:
            si += 1
            left = sets[si][1]
        left -= 1
        f.write(ROW.pack(len(r), si, at))
        at += len(r) - 1
    f.write(tokens)
    return n_pos


def write_rows(path, vocab, sets, rows):
    with open(path, "wb") as f:
        return write_prefix(f, vocab, 0, sets, rows)


def read(path):
    """-> dict(vocab, topk, sets [(name, first_row, n_rows)], rows [np.uint32 arrays], row_set, pos_offset, n_pos)."""
    with open(path, "rb") as f:
        head = f.read(HEAD.size)
        magic, version, vocab, topk, n_rows, n_sets, _, n_pos, tok_digest, body_digest = HEAD.unpack(head)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{path}: not a GLM53RF1 file")
        sets = []
        for _ in range(n_sets):
            name, first, n = SET.unpack(f.read(SET.size))
            sets.append((name.rstrip(b"\0").decode(), first, n))
        recs = [ROW.unpack(f.read(ROW.size)) for _ in range(n_rows)]
        raw = f.read(4 * sum(r[0] for r in recs))
        if zlib.crc32(raw) != tok_digest:
            raise ValueError(f"{path}: the tokens do not match their digest")
        toks = np.frombuffer(raw, dtype="<u4")
        rows, at = [], 0
        for n, _, _ in recs:
            rows.append(toks[at:at + n])
            at += n
        return {"vocab": vocab, "topk": topk, "sets": sets, "rows": rows, "row_set": [r[1] for r in recs],
                "pos_offset": f.tell(), "n_pos": n_pos, "tokens_digest": tok_digest, "body_digest": body_digest}


def position_records(logp_top, ids_top, tail, label_logp, entropy):
    """One row's records as bytes. logp_top / ids_top: [n, topk] (most likely first), the rest [n]."""
    n, topk = logp_top.shape
    out = np.zeros((n, pos_bytes(topk)), dtype=np.uint8)
    head = np.zeros((n, 4), dtype="<f4")
    head[:, 0], head[:, 1], head[:, 2] = tail, label_logp, entropy
    out[:, :12] = head[:, :3].view(np.uint8).reshape(n, 12)
    out[:, 16:16 + 4 * topk] = np.ascontiguousarray(ids_top, dtype="<u4").view(np.uint8).reshape(n, 4 * topk)
    out[:, 16 + 4 * topk:] = np.ascontiguousarray(logp_top, dtype="<f4").view(np.uint8).reshape(n, 4 * topk)
    return out.tobytes()


def records_from_logits(logits, labels, topk):
    """numpy reference of what score_reference.py computes on the device: logits [n, vocab] (float), labels [n]."""
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    order = np.lexsort((np.arange(logp.shape[1])[None, :].repeat(len(logp), 0), -logp), axis=1)[:, :topk]
    top = np.take_along_axis(logp, order, axis=1)
    p = np.exp(logp)
    tail = np.clip(1.0 - np.exp(top).sum(axis=1), 0.0, None) if topk < logp.shape[1] else np.zeros(len(logp))
    entropy = -(p * logp).sum(axis=1)
    return position_records(top, order, tail, logp[np.arange(len(logp)), labels], entropy)


class ReferenceWriter:
    """The rows of `rows_path` copied, then position records appended row by row; close() seals the digests."""

    def __init__(self, rows_path, out_path, topk):
        self.src = read(rows_path)
        if self.src["topk"] != 0:
            raise ValueError(f"{rows_path} already holds positions")
        self.topk = topk
        self.f = open(out_path, "wb")
        sets = [(name, n) for name, _, n in self.src["sets"]]
        self.n_pos = write_prefix(self.f, self.src["vocab"], topk, sets, [r.tolist() for r in self.src["rows"]])
        self.crc = 0
        self.done = 0

    def add_row(self, records):
        want = (len(self.src["rows"][self.done]) - 1) * pos_bytes(self.topk)
        if len(records) != want:
            raise ValueError(f"row {self.done}: {len(records)} bytes of records, {want} expected")
        self.f.write(records)
        self.crc = zlib.crc32(records, self.crc)
        self.done += 1

    def close(self):
        if self.done != len(self.src["rows"]):
            raise ValueError(f"{self.done} of {len(self.src['rows'])} rows written")
        self.f.seek(HEAD.size - 16)   # body_digest, then the 8 pad bytes
        self.f.write(struct.pack("<Q", self.crc))
        self.f.close()
