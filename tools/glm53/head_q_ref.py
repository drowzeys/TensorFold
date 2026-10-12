#!/usr/bin/env python3
"""numpy reference for the quantized output head (zig/src/families/glm_moe_dsa/headq.zig, TF_GLM53_HEAD_Q).

An EXL3 linear lm_head [vocab, hidden] is trellis int16 [hidden/16, vocab/16, 16*bits], suh fp16 [hidden], svh fp16
[vocab] and the mul1 codebook. This decodes it from the format's description alone (the tile's bit stream, the
codebook's arithmetic, the tile placement, the 128-wide Hadamard blocks) and computes the logits of bf16 rows the way
the engine's kernels round the input: xh = fp16(((x suh) H) / sqrt(128)) in fp32 with the kernel's butterfly order,
then everything else in float64. headq.zig's CPU replay of a rank's launch (its own blocks of the head, the share cut
out of them) must agree with these logits to rounding.

  head_q_ref.py --out zig/src/families/glm_moe_dsa/headq_cases.bin      the cases the host test embeds
  head_q_ref.py --selftest                                              the decode against a brute-force bit walk

File: magic "GLM53HQ1", u32 count, then a case: u32 k2 (half-bits), hidden, vocab, world, rows; trellis int16; suh
fp16; svh fp16; x bf16 [rows, hidden]; logits fp32 [rows, vocab].
"""
import argparse
import struct
import sys

import numpy as np

MUL1 = 0x83DCD12D
SCALE = np.array([0x1EEE], np.uint16).view(np.float16)[0]
BIAS = np.array([0xC931], np.uint16).view(np.float16)[0]
HAD = 128


def stream_ends(k2):
    p = np.arange(1, 257)
    return p * k2 // 2 if k2 % 2 == 0 else (p * k2 - (p % 2)) // 2


def placement():
    p = np.arange(256)
    lane, j = p // 8, p % 8
    return 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1), lane // 4 + 8 * (j >> 2)


def tile_states(tiles, k2):
    """tiles int16 [T, 8*k2] -> the 256 16-bit states of each, in stream order."""
    t = tiles.astype(np.int16).view(np.uint16).reshape(len(tiles), -1, 2)
    words = t[:, :, 0].astype(np.uint32) | (t[:, :, 1].astype(np.uint32) << 16)
    bits = np.unpackbits(words.astype(">u4").view(np.uint8), axis=1)      # MSB first inside each 32-bit word
    ring = 128 * k2
    assert bits.shape[1] == ring
    start = (stream_ends(k2) - 16) % ring
    idx = (start[:, None] + np.arange(16)[None, :]) % ring                 # [256, 16]
    w = bits[:, idx].astype(np.uint32)                                     # [T, 256, 16]
    return (w << np.arange(15, -1, -1, dtype=np.uint32)).sum(-1).astype(np.uint32)


def mul1(states):
    x = (states.astype(np.uint64) * MUL1) & 0xFFFFFFFF
    h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    return (h.astype(np.float64) * np.float64(SCALE) + np.float64(BIAS)).astype(np.float16)


def unpack(trellis, k2):
    """W_q fp16 [K, N] (the rotated domain) from trellis int16 [K/16, N/16, 8*k2]."""
    kt, nt, _ = trellis.shape
    vals = mul1(tile_states(trellis.reshape(kt * nt, -1), k2)).reshape(kt, nt, 256)
    row, col = placement()
    out = np.zeros((kt, nt, 16, 16), np.float16)
    out[:, :, row, col] = vals
    return out.transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def fwht_f32(v):
    """Unscaled Walsh-Hadamard of each 128 block along the last axis, fp32, strides 1, 2, ... 64 (decode.cuh fwht128)."""
    v = v.astype(np.float32).reshape(v.shape[:-1] + (-1, HAD)).copy()
    i = np.arange(HAD)
    s = 1
    while s < HAD:
        lo, hi = i[(i & s) == 0], i[(i & s) != 0]
        a, b = v[..., lo].copy(), v[..., hi].copy()
        v[..., lo], v[..., hi] = a + b, a - b
        s *= 2
    return v.reshape(v.shape[:-2] + (-1,))


def hadamard():
    i = np.arange(HAD)
    par = np.zeros((HAD, HAD), np.int64)
    x = i[:, None] & i[None, :]
    while x.any():
        par ^= x & 1
        x >>= 1
    return 1.0 - 2.0 * par


def bf16(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def logits(trellis, k2, suh, svh, x_bits):
    """float64 logits [rows, vocab] of bf16 rows x (bits), the input rounded as rot_in_kernel rounds it."""
    scale = np.float32(0.08838834764831845)
    xh = (fwht_f32(bf16(x_bits) * suh.astype(np.float32)) * scale).astype(np.float16)
    acc = xh.astype(np.float64) @ unpack(trellis, k2).astype(np.float64)
    H = hadamard()
    acc = (acc.reshape(len(acc), -1, HAD) @ H).reshape(len(acc), -1)       # H is symmetric
    return acc * np.float64(scale) * svh.astype(np.float64)


def make_case(rng, k2, hidden, vocab, world, rows):
    trellis = rng.integers(-32768, 32768, (hidden // 16, vocab // 16, 8 * k2), dtype=np.int64).astype(np.int16)
    suh = (rng.standard_normal(hidden) * 0.2 + np.sign(rng.standard_normal(hidden))).astype(np.float16)
    svh = (rng.standard_normal(vocab) * 0.1 + np.sign(rng.standard_normal(vocab)) * 0.5).astype(np.float16)
    x = rng.standard_normal((rows, hidden)).astype(np.float32)
    xb = ((x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)   # round to bf16
    lg = logits(trellis, k2, suh, svh, xb).astype(np.float32)
    return struct.pack("<5I", k2, hidden, vocab, world, rows) + trellis.tobytes() + suh.tobytes() + svh.tobytes() + xb.tobytes() + lg.tobytes()


def selftest():
    rng = np.random.default_rng(7)
    for k2 in (6, 8, 10, 12, 16):
        tile = rng.integers(-32768, 32768, (3, 8 * k2), dtype=np.int64).astype(np.int16)
        got = tile_states(tile, k2)
        ends = stream_ends(k2)
        for t in range(3):
            w = tile[t].view(np.uint16)
            for p in (0, 1, 7, 100, 255):
                s = 0
                for b in range(16):
                    i = (int(ends[p]) - 16 + b) % (128 * k2)
                    word = int(w[2 * (i // 32)]) | (int(w[2 * (i // 32) + 1]) << 16)
                    s = (s << 1) | ((word >> (31 - i % 32)) & 1)
                assert s == int(got[t, p]), (k2, t, p)
    v = rng.standard_normal((2, 256)).astype(np.float32)
    assert np.allclose(fwht_f32(v).reshape(2, 2, HAD), v.reshape(2, 2, HAD) @ hadamard(), rtol=1e-4, atol=1e-4)
    # the dense matrix diag(suh) H W_q H diag(svh) gives the same logits when the input is not rounded to fp16
    k2, K, N = 12, 256, 384
    tr = rng.integers(-32768, 32768, (K // 16, N // 16, 8 * k2), dtype=np.int64).astype(np.int16)
    suh = rng.standard_normal(K).astype(np.float16)
    svh = rng.standard_normal(N).astype(np.float16)
    Hk = np.kron(np.eye(K // HAD), hadamard()) / np.sqrt(HAD)
    Hn = np.kron(np.eye(N // HAD), hadamard()) / np.sqrt(HAD)
    W = np.diag(suh.astype(np.float64)) @ Hk @ unpack(tr, k2).astype(np.float64) @ Hn @ np.diag(svh.astype(np.float64))
    xb = np.array([[0x3F80, 0xBF00] * (K // 2)], np.uint16)                 # 1.0, -0.5: exact in every float
    exact = bf16(xb).astype(np.float64) @ W
    got = logits(tr, k2, suh, svh, xb)
    err = np.abs(got - exact).max() / np.abs(exact).max()
    assert err < 2e-3, err                                                 # the fp16 rounding of xh
    print("head_q_ref selftest OK (fp16 input rounding moves the logits by %.1e of their largest)" % err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.out:
        rng = np.random.default_rng(53)
        # vocab / world is 2.5 Hadamard blocks, as GLM-5.3's 154880 / 4 = 302.5: odd ranks start inside a block
        cases = [make_case(rng, 12, 256, 1280, 4, 3), make_case(rng, 10, 256, 1280, 4, 2), make_case(rng, 16, 128, 640, 2, 9),
                 make_case(rng, 12, 384, 1280, 1, 1)]
        open(a.out, "wb").write(b"GLM53HQ1" + struct.pack("<I", len(cases)) + b"".join(cases))
        print("wrote", a.out, sum(len(c) for c in cases), "bytes,", len(cases), "cases")
    if not a.selftest and not a.out:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
