"""H3 tile-routed attention on the M5 tensor units: each query tile reads only its chosen key tiles, in place."""

# The routing (which key tiles a query tile keeps) is FastVideo's VSA-H3 (Apache-2.0,
# https://github.com/hao-ai-lab/FastVideo); this kernel only evaluates it. The single pass over key tiles with an
# online softmax is FlashAttention's layout, as in `attention_int8.py` on the development branch. No source is copied.

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

from .mlp_int8 import _HEADER

# Measured on an M5 Ultra at 69,281 rows, 56 heads, 260 of 1,210 tiles kept: 516 ms a block against 1,933 ms for
# MLX's dense attention and 4,961 ms for FastVideo's SIMD-group kernel. A tile is 64 rows, which is also the
# fastest threadgroup of those tried for the dense form, so one threadgroup owns one query tile of one head.
TILE = 64          # rows per query tile and per key tile
DIM = 128          # channels per head
LIFT = 1024.0      # probabilities carry the value scale and this factor so small ones stay normal in half

# One thread per (head, row): the row's largest magnitude sets its scale, values round to int8.
# X: (HEADS, MP, 128) bfloat.  Q: the same in int8.  XS: (HEADS, MP) float.
_QUANTIZE = r"""
  constexpr int D = 128;
  const int at = thread_position_in_grid.x;
  const int64_t from = (int64_t)at * D;
  float v[D];
  float top = 1e-12f;
  for (int c = 0; c < D; c++) { v[c] = float(X[from + c]); top = max(top, abs(v[c])); }
  const float scale = top / 127.0f, inverse = 127.0f / top;
  XS[at] = scale;
  for (int c = 0; c < D; c++) Q[from + c] = (int8_t)clamp(int(rint(v[c] * inverse)), -127, 127);
"""

# One threadgroup owns the 64 rows of one video query tile of one head and walks its chosen key tiles once.
# Q, K, V: (HEADS, MP, 128) int8 in tile order.  QS, KS, VS: (HEADS, MP) float.
# IDX: (HEADS, NQ, KSEL) key tiles per query tile.  SIZES: (tiles,) real rows in each tile; the rest is padding.
# mdims: MP, NQ, KSEL and the number of prefix tiles before the first video tile.  Y: (NQ * 64, HEADS * 128) bfloat.
_ATTENTION = r"""
  constexpr int D = 128;
  constexpr int TPR = 256 / TQ;
  constexpr int C1 = TQ * KEYS / 256, C2 = TQ * D / 256;
  const int MP = mdims[0], NQ = mdims[1], KSEL = mdims[2], P0 = mdims[3];
  const int head = threadgroup_position_in_grid.y;
  const int qt = threadgroup_position_in_grid.x;
  const int r0 = (P0 + qt) * TQ;
  const int tid = thread_position_in_threadgroup.x;
  const int64_t hb = (int64_t)head * MP;
  const int64_t ib = ((int64_t)head * NQ + qt) * KSEL;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> q((device int8_t*)Q + hb * D, dextents<int32_t, 2>(D, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> k((device int8_t*)K + hb * D, dextents<int32_t, 2>(D, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> v((device int8_t*)V + hb * D, dextents<int32_t, 2>(D, MP));
  threadgroup PTYPE pt[TQ * KEYS];
  threadgroup float tops[256];
  threadgroup float fac[TQ];
  threadgroup float qsc[TQ];
  tensor<threadgroup PTYPE, dextents<int32_t, 2>, tensor_inline> p((threadgroup PTYPE*)pt, dextents<int32_t, 2>(KEYS, TQ));
  constexpr auto d1 = matmul2d_descriptor(TQ, KEYS, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto d2 = matmul2d_descriptor(TQ, D, KEYS, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d1, execution_simdgroups<8>> op1;
  matmul2d<d2, execution_simdgroups<8>> op2;
  auto a = q.slice<D, TQ>(0, r0);
  auto b0 = k.slice<D, KEYS>(0, 0);
  auto v0 = v.slice<D, KEYS>(0, 0);
  auto acc = op1.template get_destination_cooperative_tensor<decltype(a), decltype(b0), int32_t>();
  auto out = op2.template get_destination_cooperative_tensor<decltype(p), decltype(v0), float>();
  short srow[C1], scol[C1], orow[C2], ocol[C2];
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < C1; i++) { auto ids = acc.get_multidimensional_index(i); scol[i] = ids[0]; srow[i] = ids[1]; }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < C2; i++) {
    auto ids = out.get_multidimensional_index(i);
    ocol[i] = ids[0]; orow[i] = ids[1]; out[i] = 0.0f;
  }
  for (int j = tid; j < TQ; j += 256) qsc[j] = QS[hb + r0 + j] * SCALE;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int row = tid / TPR, first = (tid % TPR) * C1;
  const int mine = row * KEYS + first;
  float top = -1e30f, total = 0.0f;
  float s[C1];
  for (int sel = 0; sel < KSEL; sel++) {
    const int tile = IDX[ib + sel];
    const int k0 = tile * KEYS;
    const int size = SIZES[tile];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < C1; i++) acc[i] = 0;
    auto b = k.slice<D, KEYS>(0, k0);
    op1.run(a, b, acc);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < C1; i++) {
      const float score = scol[i] < size ? float(acc[i]) * qsc[srow[i]] * KS[hb + k0 + scol[i]] : -60000.0f;
      pt[srow[i] * KEYS + scol[i]] = PTYPE(score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local = -1e30f;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < C1; j++) { s[j] = float(pt[mine + j]); local = max(local, s[j]); }
    tops[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float next = top;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < TPR; j++) next = max(next, tops[row * TPR + j]);
    const float shrink = exp(top - next);
    if (tid % TPR == 0) fac[row] = shrink;
    total *= shrink;
    top = next;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < C1; j++) {
      const float weight = exp(s[j] - next);
      total += weight;
      pt[mine + j] = PTYPE(weight * VS[hb + k0 + first + j] * LIFT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < C2; i++) out[i] *= fac[orow[i]];
    auto vb = v.slice<D, KEYS>(0, k0);
    op2.run(p, vb, out);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  tops[tid] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % TPR == 0) {
    float sum = 0.0f;
    #pragma clang loop unroll(full)
    for (ushort j = 0; j < TPR; j++) sum += tops[tid + j];
    fac[row] = 1.0f / (sum * LIFT);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < C2; i++) {
    const int at = qt * TQ + orow[i];
    Y[((int64_t)at * HEADS + head) * D + ocol[i]] = static_cast<bfloat>(out[i] * fac[orow[i]]);
  }
"""

_compiled: dict[tuple, Any] = {}


def _kernel(kind: str, heads: int = 0, scale: float = 0.0) -> Any:
    key = (kind, heads, scale)
    run = _compiled.get(key)
    if run is None:
        if kind == "quantize":
            source, inputs, outputs = _QUANTIZE, ["X"], ["Q", "XS"]
        else:
            source = (f"  constexpr int HEADS = {heads};\n  constexpr int KEYS = {TILE};\n  constexpr int TQ = {TILE};\n"
                      f"  typedef half PTYPE;\n  constexpr float SCALE = {scale!r}f;\n"
                      f"  constexpr float LIFT = {LIFT!r}f;\n") + _ATTENTION
            inputs, outputs = ["Q", "QS", "K", "KS", "V", "VS", "IDX", "SIZES", "mdims"], ["Y"]
        name = f"h3_tile_attention_{kind}_" + hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        run = _compiled[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                                    source=source, header=_HEADER)
    return run


def quantize_heads(x: mx.array) -> tuple[mx.array, mx.array]:
    """(heads, rows, 128) -> the same in int8 and one float32 scale per head and row."""

    heads, rows, dim = x.shape
    if dim != DIM:
        raise ValueError(f"tile attention needs {DIM}-channel heads, got {dim}")
    q, scale = _kernel("quantize")(inputs=[mx.contiguous(x.astype(mx.bfloat16))], grid=(heads * rows, 1, 1),
                                   threadgroup=(DIM, 1, 1), output_shapes=[(heads, rows, dim), (heads, rows)],
                                   output_dtypes=[mx.int8, mx.float32])
    return q, scale


def attention(q: mx.array, k: mx.array, v: mx.array, chosen: mx.array, sizes: mx.array, prefix_tiles: int,
              scale: float) -> mx.array:
    """Routed ``softmax(q k^T scale) v`` for the video query tiles.

    q, k, v are (heads, tiles * 64, 128) in tile order, padding rows anywhere inside a tile's 64. ``chosen`` is
    (heads, video tiles, kept) key-tile numbers and ``sizes`` the real rows of every tile. Returns
    (video tiles * 64, heads * 128); rows that are padding hold no meaning.
    """

    heads, rows, dim = q.shape
    if rows % TILE or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"tile attention takes q, k, v of one shape (heads, tiles * {TILE}, {DIM}), got {q.shape}, "
                         f"{k.shape}, {v.shape}")
    if chosen.ndim != 3 or chosen.shape[0] != heads or chosen.shape[1] != rows // TILE - prefix_tiles:
        raise ValueError(f"chosen tiles {chosen.shape} do not match {heads} heads and "
                         f"{rows // TILE - prefix_tiles} video tiles")
    (qq, qs), (kq, ks), (vq, vs) = quantize_heads(q), quantize_heads(k), quantize_heads(v)
    queries, kept = chosen.shape[1], chosen.shape[2]
    dims = mx.array([rows, queries, kept, prefix_tiles], dtype=mx.int32)
    return _kernel("attention", heads, float(scale))(
        inputs=[qq, qs, kq, ks, vq, vs, chosen.astype(mx.int32), sizes.astype(mx.int32), dims],
        grid=(queries * 256, heads, 1), threadgroup=(256, 1, 1), output_shapes=[(queries * TILE, heads * dim)],
        output_dtypes=[mx.bfloat16])[0]


def available() -> bool:
    """Whether this GPU compiles and runs the tile attention kernel."""

    try:
        x = mx.ones((1, 2 * TILE, DIM), dtype=mx.bfloat16)
        mx.eval(attention(x, x, x, mx.zeros((1, 1, 1), dtype=mx.int32), mx.array([TILE, TILE]), 1, 0.1))
    except (RuntimeError, ValueError):
        return False
    return True
