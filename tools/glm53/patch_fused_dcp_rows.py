#!/usr/bin/env python3
"""Several streams under decode context parallelism: the two kernel changes the Zig engine's *RowsDcp launches need
(docs/glm53-dcp-streams.md), applied to a copy of the Python engine's fused.py before sweep_aot.py compiles them.

  patch_fused_dcp_rows.py FUSED.py [--check]      edits the file in place (again: no change); --check only reports

1. _kv_write / _ik_write with ROWS: the ROWS branch ignored DCP (the Python engine never has both). With DCP > 1 row r
   is stored only by the owner of its position (POS[r] % DCP == RANK), at BASE[r] + POS[r] // DCP. With DCP 1 the
   address is the one it was (pg // 1), so packed ROWS variants stay valid.
2. _attn_dcp_rows: _attn_dcp with a position and a cache base a row (POS[r], LC + BASE[r] rows), derived from the
   file's own _attn_dcp so the two cannot drift. A function of its own: no variant of _attn_dcp can answer its launch.

No call site in fused.py changes: only the Zig engine launches these (cuda_triton.zig kvWriteRowsDcp, ikWriteRowsDcp,
indexScoresRowsDcp - _index_scores composes ROWS and DCP as written - and attnDcpRows).
"""
import re
import sys

ROWS_OLD = """    if ROWS:
        pg = tl.load(POS + r).to(tl.int64)
        p = tl.load(BASE + r).to(tl.int64) + pg
    else:
"""
ROWS_NEW = """    if ROWS:
        pg = tl.load(POS + r).to(tl.int64)
        if DCP > 1:                              # several streams, sharded caches: the owner's row of its stream
            if pg % DCP != RANK:
                return
        p = tl.load(BASE + r).to(tl.int64) + pg // DCP
    else:
"""


def patch(src: str) -> tuple[str, list[str]]:
    did = []
    n_old, n_new = src.count(ROWS_OLD), src.count(ROWS_NEW)
    # twice in the bf16-cache source (_kv_write, _ik_write); a third time where _kv_write_q exists (the quantized
    # latent cache, which this engine does not launch): the same branch, the same change
    if n_old >= 2 and n_new == 0:
        src = src.replace(ROWS_OLD, ROWS_NEW)
        did.append(f"ROWS honours DCP in {n_old} key writers (_kv_write, _ik_write{', _kv_write_q' if n_old > 2 else ''})")
    elif not (n_old == 0 and n_new >= 2):
        raise SystemExit(f"fused.py: expected the ROWS branch of _kv_write and _ik_write, found {n_old} old / {n_new} new")
    for fn in ("_kv_write", "_ik_write"):
        body = re.search(rf"\ndef {fn}\(.*?(?=\n\n\n@triton\.jit\n)", src, re.S)
        if not body or ROWS_NEW not in body.group(0):
            raise SystemExit(f"fused.py: {fn} did not take the change")
    if "def _attn_dcp_rows(" not in src:
        m = re.search(r"@triton\.jit\ndef _attn_dcp\(.*?(?=\n\n\n@triton\.jit\n)", src, re.S)
        if not m:
            raise SystemExit("fused.py: no _attn_dcp")
        f = m.group(0)
        for old, new in (
            ("def _attn_dcp(QALL, LC, TOK, CNT, POS, PO, PM, PL, R,", "def _attn_dcp_rows(QALL, LC, TOK, CNT, POS, BASE, PO, PM, PL, R,"),
            ("    p = tl.load(POS) + r\n", "    p = tl.load(POS + r)                         # the row's own position, its stream's cache rows\n"
                                           "    LC = LC + tl.load(BASE + r).to(tl.int64) * (LW + RD)\n"),
        ):
            if f.count(old) != 1:
                raise SystemExit(f"fused.py: _attn_dcp is not the source this patch knows ({old.strip()!r} x {f.count(old)})")
            f = f.replace(old, new)
        f = f.replace('    """Program (row, chunk, head group g)', '    """_attn_dcp for several streams in one window (row r at POS[r], cache rows from BASE[r]). '
                      'Program (row, chunk, head group g)', 1)
        src = src[:m.end()] + "\n\n\n" + f + src[m.end():]
        did.append("_attn_dcp_rows added")
    return src, did


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path, check = sys.argv[1], "--check" in sys.argv[2:]
    old = open(path).read()
    new, did = patch(old)
    compile(new, path, "exec")
    again, more = patch(new)
    assert again == new and not more
    if check:
        print("needs:", "; ".join(did) if did else "nothing (already patched)")
        return 0
    if did:
        open(path, "w").write(new)
    print("patched:" if did else "already patched:", "; ".join(did) if did else path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
