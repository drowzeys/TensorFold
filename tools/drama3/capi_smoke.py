#!/usr/bin/env python3
"""libtensorfold_drama3.so smoke test: open, build prompt rows (Zig tokenizer), generate, write frames.
usage: capi_smoke.py LIB MODEL_DIR TEXT OUT.frames"""
import ctypes as C
import sys

lib = C.CDLL(sys.argv[1])
lib.tf_drama3_open.restype = C.c_void_p
lib.tf_drama3_open.argtypes = [C.c_char_p, C.c_int, C.c_int]
lib.tf_drama3_prompt.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_void_p, C.c_int, C.POINTER(C.c_int32), C.c_int]
lib.tf_drama3_generate.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.c_float, C.c_int, C.c_float, C.c_uint64, C.c_int, C.POINTER(C.c_int32), C.POINTER(C.c_double)]
lib.tf_drama3_error.restype = C.c_char_p
h = lib.tf_drama3_open(sys.argv[2].encode(), 4096, 2048)
if not h:
    sys.exit(f"open: {lib.tf_drama3_error().decode()}")
rows = (C.c_int32 * (4096 * 11))()
n = lib.tf_drama3_prompt(h, sys.argv[3].encode(), None, None, 0, rows, 4096)
frames = (C.c_int32 * (2048 * 11))()
stats = (C.c_double * 4)()
for run in range(2):
    m = lib.tf_drama3_generate(h, rows, n, 1.0, 30, 0.9, 7, 1500, frames, stats)
    if m < 0:
        sys.exit(f"generate: {lib.tf_drama3_error().decode()}")
    print(f"run {run}: {n} prompt rows -> {m} frames; prefill {stats[0]:.1f} ms, first frame {stats[1]:.1f} ms, {stats[3]:.2f} frames/s")
open(sys.argv[4], "wb").write(bytes(frames)[: m * 11 * 4])
lib.tf_drama3_close(C.c_void_p(h))
print("closed")
