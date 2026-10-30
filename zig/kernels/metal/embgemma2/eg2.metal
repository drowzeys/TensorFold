// EmbeddingGemma-2's text encoder glue at torch's bf16 rounding points, after gemm_nax.metal (nax.h inlined).
using namespace metal;

#ifndef EG2_ACC
#define EG2_ACC float // the planted control's half accumulators (kernel contract) swap this
#endif

inline float eg2_rbf(float x) { return float(bfloat(x)); }

// A threadgroup's sum of every thread's v (nsg simdgroups), the same total in every thread.
inline float eg2_sum(float v, threadgroup float* part, uint sg, uint lane, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? part[lane] : 0.0f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// the model's RMSNorm factor: pow(mean(x^2) + eps, -0.5) in fp32
inline float eg2_inv_rms(float sum, int n, float eps) { return precise::rsqrt(sum / float(n) + eps); }

// torch's gelu(approximate="tanh") in fp32
inline float eg2_gelu(float x) {
  const float k = 0.7978845608028654f; // sqrt(2 / pi)
  return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x * x * x)));
}

// ---- the projections: x [M, K] by the stored w [N, K] in bf16 with fp32 sums, rounded to bf16 ----

// PLAIN stores y; GLU (w's gate and up rows interleaved 16 at a time) rbf(gelu(gate)) * up; PLE rbf(gelu(y)) * pli
constant constexpr int EG2_PLAIN = 0;
constant constexpr int EG2_GLU = 1;
constant constexpr int EG2_PLE = 2;

// Many rows: gemm_nax's 64 x 128 tiles (a simdgroup a 32 x 32 block); a row's sums never depend on its batch.
template <int EPI, bool AM>
inline void eg2_gemm_t(const device bfloat* A, const device bfloat* B, device bfloat* D, const device int* P,
                       const device bfloat* X, uint3 tg, uint sg, uint lane) {
  using namespace tfp;
  constexpr int BM = 64, BN = 128, BK = 256, WN = 4;
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) return;
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(row) * lda;
  B += long(col) * ldb;
  const int rows = AM ? 32 : min(32, P[0] - row);
  const short2 home = frag_home(ushort(lane));
  frag<float> acc[2][2];
  if (AM || rows == 32) {
    gemm_block<bfloat, false, true, true, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, 32, home);
  } else {
    gemm_block<bfloat, false, true, false, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, 32, home);
    if (rows <= 0) return;
  }
  if (EPI == EG2_PLAIN) {
    D += long(row) * ldd + col;
    TF_UNROLL
    for (short i = 0; i < 2; i++) {
      TF_UNROLL
      for (short j = 0; j < 2; j++) frag_put_in(acc[i][j], D, ldd, 16 * i, 16 * j, home, rows, 32);
    }
  } else if (EPI == EG2_GLU) {
    D += long(row) * ldd + col / 2;
    TF_UNROLL
    for (short i = 0; i < 2; i++) {
      frag<float> o;
      TF_UNROLL
      for (short e = 0; e < 8; e++) o[e] = eg2_rbf(eg2_rbf(eg2_gelu(eg2_rbf(acc[i][0][e]))) * eg2_rbf(acc[i][1][e]));
      frag_put_in(o, D, ldd, 16 * i, 0, home, rows, 16);
    }
  } else {
    D += long(row) * ldd + col;
    X += long(row) * P[10] + P[11] + col;
    TF_UNROLL
    for (short i = 0; i < 2; i++) {
      TF_UNROLL
      for (short j = 0; j < 2; j++) {
        frag<float> p;
        frag_get_in(p, X, P[10], 16 * i, 16 * j, home, rows, 32);
        frag<float> o;
        TF_UNROLL
        for (short e = 0; e < 8; e++) o[e] = eg2_rbf(eg2_gelu(eg2_rbf(acc[i][j][e]))) * p[e];
        frag_put_in(o, D, ldd, 16 * i, 16 * j, home, rows, 32);
      }
    }
  }
}

#define EG2_GEMM(NAME, EPI, AM)                                                                                     \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                  \
      const device bfloat* A [[buffer(0)]], const device bfloat* B [[buffer(1)]], const device int* P [[buffer(2)]],  \
      device bfloat* D [[buffer(3)]], const device bfloat* X [[buffer(4)]],                                        \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                        \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                 \
    eg2_gemm_t<EPI, AM>(A, B, D, P, X, tg, sg, lane);                                                              \
  }
EG2_GEMM(eg2_gemm_0, EG2_PLAIN, false)
EG2_GEMM(eg2_gemm_1, EG2_PLAIN, true)
EG2_GEMM(eg2_gemm_glu_0, EG2_GLU, false)
EG2_GEMM(eg2_gemm_glu_1, EG2_GLU, true)
EG2_GEMM(eg2_gemm_ple_0, EG2_PLE, false)
EG2_GEMM(eg2_gemm_ple_1, EG2_PLE, true)

// Few rows: a simdgroup an output column (Spark2.5's sp_matvec), k % 256 == 0; x is read up to ROWS rows.
struct eg2_mv_p { uint rows; uint n; uint k; uint ldp; uint poff; };

template <uint ROWS, int EPI>
inline void eg2_matvec_t(const device bfloat* X, const device bfloat* W, constant eg2_mv_p& a, device bfloat* Y,
                         const device bfloat* pli, uint3 tg, uint sg, uint lane) {
  const uint j = tg.x * 8 + sg; // the output column (EG2_GLU: of gate * up)
  if (j >= a.n) return;
  const device bfloat* w = W + (size_t)(EPI == EG2_GLU ? 32 * (j / 16) + j % 16 : j) * a.k;
  const device bfloat* u = w + 16 * (size_t)a.k; // EG2_GLU: the matching up row
  float acc[ROWS], acu[EPI == EG2_GLU ? ROWS : 1];
  TF_UNROLL
  for (uint r = 0; r < ROWS; ++r) acc[r] = 0.0f;
  TF_UNROLL
  for (uint r = 0; r < (EPI == EG2_GLU ? ROWS : 1); ++r) acu[r] = 0.0f;
  for (uint i = 8 * lane; i < a.k; i += 256) {
    const float4 w0 = float4(*(const device bfloat4*)(w + i)), w1 = float4(*(const device bfloat4*)(w + i + 4));
    float4 u0 = 0.0f, u1 = 0.0f;
    if (EPI == EG2_GLU) {
      u0 = float4(*(const device bfloat4*)(u + i));
      u1 = float4(*(const device bfloat4*)(u + i + 4));
    }
    TF_UNROLL
    for (uint r = 0; r < ROWS; ++r) {
      const device bfloat* x = X + (size_t)r * a.k + i;
      const float4 x0 = float4(*(const device bfloat4*)x), x1 = float4(*(const device bfloat4*)(x + 4));
      acc[r] += dot(w0, x0) + dot(w1, x1);
      if (EPI == EG2_GLU) acu[EPI == EG2_GLU ? r : 0] += dot(u0, x0) + dot(u1, x1);
    }
  }
  TF_UNROLL
  for (uint r = 0; r < ROWS; ++r) {
    const float s = simd_sum(acc[r]);
    float v = s;
    if (EPI == EG2_GLU) v = eg2_rbf(eg2_gelu(eg2_rbf(s))) * eg2_rbf(simd_sum(acu[EPI == EG2_GLU ? r : 0]));
    if (EPI == EG2_PLE) v = eg2_rbf(eg2_gelu(eg2_rbf(s))) * float(pli[(size_t)r * a.ldp + a.poff + j]);
    if (lane == 0 && r < a.rows) Y[(size_t)r * a.n + j] = bfloat(v);
  }
}

#define EG2_MATVEC(NAME, ROWS, EPI)                                                                                \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                 \
      const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]], constant eg2_mv_p& a [[buffer(2)]], \
      device bfloat* Y [[buffer(3)]], const device bfloat* pli [[buffer(4)]],                                     \
      uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]) {                                                                  \
    eg2_matvec_t<ROWS, EPI>(X, W, a, Y, pli, tg, sg, lane);                                                        \
  }
EG2_MATVEC(eg2_matvec_4, 4, EG2_PLAIN)
EG2_MATVEC(eg2_matvec_8, 8, EG2_PLAIN)
EG2_MATVEC(eg2_matvec_16, 16, EG2_PLAIN)
EG2_MATVEC(eg2_matvec_32, 32, EG2_PLAIN)
EG2_MATVEC(eg2_matvec_glu_4, 4, EG2_GLU)
EG2_MATVEC(eg2_matvec_glu_8, 8, EG2_GLU)
EG2_MATVEC(eg2_matvec_glu_16, 16, EG2_GLU)
EG2_MATVEC(eg2_matvec_glu_32, 32, EG2_GLU)
EG2_MATVEC(eg2_matvec_ple_4, 4, EG2_PLE)
EG2_MATVEC(eg2_matvec_ple_8, 8, EG2_PLE)
EG2_MATVEC(eg2_matvec_ple_16, 16, EG2_PLE)
EG2_MATVEC(eg2_matvec_ple_32, 32, EG2_PLE)

struct eg2_embed_p { uint n; float scale; };

// y [rows, n] = table[ids] * scale (a bf16 product). Threads (n, rows).
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void eg2_embed(const device int* ids [[buffer(0)]], const device bfloat* table [[buffer(1)]],
                          device bfloat* y [[buffer(2)]], constant eg2_embed_p& p [[buffer(3)]],
                          uint2 at [[thread_position_in_grid]]) {
  if (at.x >= p.n) return;
  const long src = long(ids[at.y]) * p.n + at.x;
  y[long(at.y) * p.n + at.x] = bfloat(float(table[src]) * p.scale);
}

struct eg2_rms_p { uint n; float eps; float pre; uint scaled; };

// y [rows, n] = RMSNorm(rbf(x * pre)) (* w when scaled); a threadgroup of 128 a row, n <= 512.
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_rms(
    const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant eg2_rms_p& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4];
  const long at = long(row) * p.n;
  float v[4];
  EG2_ACC s = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const uint c = t + 128 * i;
    float u = c < p.n ? float(x[at + c]) : 0.0f;
    if (p.pre != 1.0f) u = eg2_rbf(u * p.pre);
    v[i] = u;
    s += EG2_ACC(u * u);
  }
  const float r = eg2_inv_rms(eg2_sum(float(s), part, sg, lane, 4), p.n, p.eps);
  for (int i = 0; i < 4; ++i) {
    const uint c = t + 128 * i;
    if (c >= p.n) continue;
    float o = v[i] * r;
    if (p.scaled) o = o * float(w[c]);
    y[at + c] = bfloat(o);
  }
}

// RoPE's cos then sin at every position and frequency, bf16 as torch rounds them, made once at load.
struct eg2_rope_p { uint hd; float theta; uint positions; };

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_rope_table(device bfloat* cs [[buffer(0)]], constant eg2_rope_p& p [[buffer(1)]],
                               uint2 at [[thread_position_in_grid]]) {
  const uint half_ = p.hd / 2;
  if (at.x >= half_ || at.y >= p.positions) return;
  const float inv = 1.0f / precise::pow(p.theta, float(2 * at.x) / float(p.hd));
  const float f = float(at.y) * inv;
  cs[long(at.y) * half_ + at.x] = bfloat(precise::cos(f));
  cs[long(p.positions + at.y) * half_ + at.x] = bfloat(precise::sin(f));
}

// The rotation of a normed pair (a, b) = (x[c], x[c + hd / 2]) by the table's cs, sn, at torch's roundings.
inline float2 eg2_rotate(float a, float b, float cs, float sn) {
  return float2(eg2_rbf(eg2_rbf(a * cs) + eg2_rbf(-b * sn)), eg2_rbf(eg2_rbf(b * cs) + eg2_rbf(a * sn)));
}

struct eg2_qkv_p { uint hq; uint hkv; uint hd; uint positions; float eps; };

// q, k: RMSNorm with weights then RoPE at the row's position; v: RMSNorm. 128 threads a (row, head).
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_qkv(
    device bfloat* qkv [[buffer(0)]], const device bfloat* qn [[buffer(1)]], const device bfloat* kn [[buffer(2)]],
    const device int* pos [[buffer(3)]], constant eg2_qkv_p& p [[buffer(4)]], const device bfloat* rope [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4];
  const uint row = g.x, head = g.y; // q heads, then k heads, then v heads
  const uint heads = p.hq + 2 * p.hkv;
  device bfloat* h = qkv + (long(row) * heads + head) * p.hd;
  const bool is_v = head >= p.hq + p.hkv;
  const device bfloat* w = head < p.hq ? qn : kn;
  const uint half_ = p.hd / 2;
  float lo[2], hi[2];
  EG2_ACC s = 0.0f;
  for (int i = 0; i < 2; ++i) {
    const uint c = t + 128 * i;
    lo[i] = c < half_ ? float(h[c]) : 0.0f;
    hi[i] = c < half_ ? float(h[c + half_]) : 0.0f;
    s += EG2_ACC(lo[i] * lo[i] + hi[i] * hi[i]);
  }
  const float r = eg2_inv_rms(eg2_sum(float(s), part, sg, lane, 4), p.hd, p.eps);
  const long at = long(pos[row]) * half_;
  for (int i = 0; i < 2; ++i) {
    const uint c = t + 128 * i;
    if (c >= half_) continue;
    float a = lo[i] * r, b = hi[i] * r;
    if (!is_v) {
      a = a * float(w[c]);
      b = b * float(w[c + half_]);
    }
    a = eg2_rbf(a);
    b = eg2_rbf(b);
    if (!is_v) {
      const float2 ab = eg2_rotate(a, b, float(rope[at + c]), float(rope[long(p.positions) * half_ + at + c]));
      a = ab.x;
      b = ab.y;
    }
    h[c] = bfloat(a);
    h[c + half_] = bfloat(b);
  }
}

struct eg2_add_norm_p { uint n; float eps; float scalar; uint next; };

// post norm, residual, layer scalar and the next norm in one pass: h += rbf(norm(x) * wp); y = norm(h) * wn
[[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_add_norm(
    const device bfloat* x [[buffer(0)]], const device bfloat* wp [[buffer(1)]], device bfloat* h [[buffer(2)]],
    const device bfloat* wn [[buffer(3)]], device bfloat* y [[buffer(4)]], constant eg2_add_norm_p& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4];
  const long at = long(row) * p.n;
  float v[4];
  EG2_ACC s = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const uint c = t + 128 * i;
    v[i] = c < p.n ? float(x[at + c]) : 0.0f;
    s += EG2_ACC(v[i] * v[i]);
  }
  const float r = eg2_inv_rms(eg2_sum(float(s), part, sg, lane, 4), p.n, p.eps);
  EG2_ACC u = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const uint c = t + 128 * i;
    if (c >= p.n) {
      v[i] = 0.0f;
      continue;
    }
    float o = eg2_rbf(v[i] * r * float(wp[c]));
    o = eg2_rbf(float(h[at + c]) + o);
    if (p.scalar != 1.0f) o = eg2_rbf(o * p.scalar);
    h[at + c] = bfloat(o);
    v[i] = o;
    u += EG2_ACC(o * o);
  }
  if (!p.next) return;
  const float r2 = eg2_inv_rms(eg2_sum(float(u), part, sg, lane, 4), p.n, p.eps);
  for (int i = 0; i < 4; ++i) {
    const uint c = t + 128 * i;
    if (c < p.n) y[at + c] = bfloat(v[i] * r2 * float(wn[c]));
  }
}

struct eg2_pool_p { uint n; uint dim; };

// Mean pooling, the projection (linear, so taken on the mean) and L2 normalisation: 256 threads an input.
[[max_total_threads_per_threadgroup(256)]] [[kernel]] void eg2_pool(
    const device bfloat* x [[buffer(0)]], const device int* ends [[buffer(1)]], const device bfloat* w [[buffer(2)]],
    device float* out [[buffer(3)]], constant eg2_pool_p& p [[buffer(4)]], uint b [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float mean[512];
  threadgroup float part[8];
  const int lo = b == 0 ? 0 : ends[b - 1];
  const int hi = ends[b];
  for (uint c = t; c < p.n; c += 256) {
    float s = 0.0f;
    for (int r = lo; r < hi; ++r) s += float(x[long(r) * p.n + c]);
    mean[c] = s / float(hi - lo);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float vals[4];
  float q = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const uint o = t + 256 * i;
    float s = 0.0f;
    if (o < p.dim)
      for (uint c = 0; c < p.n; ++c) s = fma(mean[c], float(w[long(o) * p.n + c]), s);
    vals[i] = s;
    q += s * s;
  }
  const float norm = precise::sqrt(eg2_sum(q, part, sg, lane, 8));
  for (int i = 0; i < 4; ++i) {
    const uint o = t + 256 * i;
    if (o < p.dim) out[long(b) * p.dim + o] = vals[i] / norm;
  }
}
