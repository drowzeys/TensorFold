// Qwen-Image-2.1 transformer kernels for the M5 tensor units: bf16 rows, fp32 arithmetic inside.
// The fragment layout and the 16x32x16 multiply-accumulate are ../nax.h's (its M5 branch), inlined so this file
// compiles alone with newLibraryWithSource; qi_gemm is ops/nax_gemm.metal's kernel under another name.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define QI_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
using qi_frag = vec<T, 8>;

inline short2 qi_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

template <typename T, typename S>
inline void qi_get(thread qi_frag<T>& f, const device S* p, long ld, int r, int c, short2 home, int nr, int nc) {
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? T(p[long(rr) * ld + cc]) : T(0);
  }
}

template <typename O>
inline void qi_put(thread const qi_frag<float>& f, device O* p, long ld, int r, int c, short2 home, int nr, int nc) {
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    if (rr < nr && cc < nc) p[long(rr) * ld + cc] = O(f[e]);
  }
}

template <typename C, typename A, typename B>
inline void qi_mma(thread qi_frag<C>& lo, thread qi_frag<C>& hi, thread const qi_frag<A>& a,
                   thread const qi_frag<B>& b0, thread const qi_frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// D[z] = A[z] B[z]: A (M, K), B (K, N), D (M, N), row-major with leading dimensions lda, ldb, ldd.
// P: M, N, K. LD: lda, ldb, ldd, then the element strides of A, B and D from one z to the next.
// Threadgroups [ceil(N / 128), ceil(M / 64), batch] of [32, 4, 1].
[[kernel]] void qi_gemm(
  const device bfloat* A [[buffer(0)]],
  const device bfloat* B [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* LD [[buffer(3)]],
  device bfloat* D [[buffer(4)]],
  uint sg [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = P[0], N = P[1], K = P[2];
  const long lda = LD[0], ldb = LD[1], ldd = LD[2];
  const device bfloat* a = A + long(tg.z) * LD[3];
  const device bfloat* b = B + long(tg.z) * LD[4];
  device bfloat* d = D + long(tg.z) * LD[5];
  const int col0 = int(tg.x) * 128 + 32 * int(sg);
  if (col0 >= N) return;
  const short2 o = qi_home(ushort(lane));
  const int chunks = (K + 15) / 16;
  for (int row0 = int(tg.y) * 64; row0 < min(M, int(tg.y) * 64 + 64); row0 += 16) {
    qi_frag<float> c0 = 0.0f, c1 = 0.0f;
    for (int ch = 0; ch < chunks; ++ch) {
      const int k0 = 16 * ch;
      qi_frag<bfloat> fa, f0, f1;
      qi_get(fa, a, lda, row0, k0, o, M, K);
      qi_get(f0, b, ldb, k0, col0, o, K, N);
      qi_get(f1, b, ldb, k0, col0 + 16, o, K, N);
      qi_mma(c0, c1, fa, f0, f1);
    }
    qi_put(c0, d, ldd, row0, col0, o, M, N);
    qi_put(c1, d, ldd, row0, col0 + 16, o, M, N);
  }
}

// qi_gemm for whole 128 x 128 tiles: N and K multiples of 128, a's rows readable up to the next multiple of 128
// (what lies there is multiplied and dropped). One threadgroup of 256 threads owns one output tile and lets the
// tensor operation read its operands in place. Threadgroups [N / 128, ceil(M / 128), batch] of [256, 1, 1].
[[kernel]] void qi_tile_gemm(
  const device bfloat* A [[buffer(0)]],
  const device bfloat* B [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* LD [[buffer(3)]],
  device bfloat* D [[buffer(4)]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int T = 128;
  constexpr int CAP = T * T / 256;
  const int M = P[0], N = P[1], K = P[2];
  const int MP = (M + T - 1) / T * T;
  device bfloat* a = (device bfloat*)A + long(tg.z) * LD[3];
  device bfloat* b = (device bfloat*)B + long(tg.z) * LD[4];
  device bfloat* d = D + long(tg.z) * LD[5];
  const int n0 = int(tg.x) * T, r0 = int(tg.y) * T;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x(a, dextents<int32_t, 2>(int(LD[0]), MP));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> w(b, dextents<int32_t, 2>(int(LD[1]), K));
  constexpr auto desc = matmul2d_descriptor(T, T, T, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<8>> op;
  auto a0 = x.slice<T, T>(0, r0);
  auto b0 = w.slice<T, T>(n0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) acc[i] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += T) {
    auto at = x.slice<T, T>(k0, r0);
    auto bt = w.slice<T, T>(n0, k0);
    op.run(at, bt, acc);
  }
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    auto ids = acc.get_multidimensional_index(i);
    const int row = r0 + ids[1];
    if (row < M) d[long(row) * LD[2] + n0 + ids[0]] = bfloat(acc[i]);
  }
}

// Y = layer_norm(X) * S, one simdgroup a row. X, Y: (R, C). S: (C), one plus the modulation's scale. P: C. E: eps.
// Threadgroups [R, 1, 1] of [32, 1, 1].
[[kernel]] void qi_norm_scale(
  const device bfloat* X [[buffer(0)]],
  const device bfloat* S [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant float* E [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int C = P[0];
  const device bfloat* x = X + long(tg.x) * C;
  device bfloat* y = Y + long(tg.x) * C;
  float sum = 0.0f;
  for (int c = int(lane); c < C; c += 32) sum += float(x[c]);
  const float mean = simd_sum(sum) / float(C);
  float sq = 0.0f;
  for (int c = int(lane); c < C; c += 32) { const float v = float(x[c]) - mean; sq += v * v; }
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  for (int c = int(lane); c < C; c += 32) y[c] = bfloat(float(bfloat((float(x[c]) - mean) * inv)) * float(S[c]));
}

// The projections' rows laid out for attention, one thread per (row, head): q and k take their head's RMSNorm
// and the rotary rotation of channel pairs (2c, 2c + 1); the row's keys and values land after the T text rows.
// QR, KR, VR: (R, H * 128). NQ, NK: (128). COS, SIN: (R, 64) float. P: R, H, T, N. E: eps.
// QH: (H, R, 128). KT: (H, 128, N). VH: (H, N, 128), N the keys' padded count. Threads [R, H, 1].
[[kernel]] void qi_heads(
  const device bfloat* QR [[buffer(0)]],
  const device bfloat* KR [[buffer(1)]],
  const device bfloat* VR [[buffer(2)]],
  const device bfloat* NQ [[buffer(3)]],
  const device bfloat* NK [[buffer(4)]],
  const device float* COS [[buffer(5)]],
  const device float* SIN [[buffer(6)]],
  const constant int32_t* P [[buffer(7)]],
  const constant float* E [[buffer(8)]],
  device bfloat* QH [[buffer(9)]],
  device bfloat* KT [[buffer(10)]],
  device bfloat* VH [[buffer(11)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], H = P[1], T = P[2], N = P[3];
  const int row = int(at.x), head = int(at.y);
  if (row >= R || head >= H) return;
  const long from = (long(row) * H + head) * 128;
  const device float* cs = COS + long(row) * 64;
  const device float* sn = SIN + long(row) * 64;
  float q[128], k[128];
  float qq = 0.0f, kk = 0.0f;
  for (int c = 0; c < 128; c++) {
    q[c] = float(QR[from + c]);
    k[c] = float(KR[from + c]);
    qq += q[c] * q[c];
    kk += k[c] * k[c];
  }
  const float qi = rsqrt(qq / 128.0f + E[0]), ki = rsqrt(kk / 128.0f + E[0]);
  for (int c = 0; c < 128; c++) {
    q[c] = float(bfloat(q[c] * qi * float(NQ[c])));
    k[c] = float(bfloat(k[c] * ki * float(NK[c])));
  }
  device bfloat* qo = QH + (long(head) * R + row) * 128;
  device bfloat* ko = KT + long(head) * 128 * N + T + row;
  device bfloat* vo = VH + (long(head) * N + T + row) * 128;
  for (int c = 0; c < 64; c++) {
    const float co = cs[c], si = sn[c];
    qo[2 * c] = bfloat(q[2 * c] * co - q[2 * c + 1] * si);
    qo[2 * c + 1] = bfloat(q[2 * c] * si + q[2 * c + 1] * co);
    ko[long(2 * c) * N] = bfloat(k[2 * c] * co - k[2 * c + 1] * si);
    ko[long(2 * c + 1) * N] = bfloat(k[2 * c] * si + k[2 * c + 1] * co);
  }
  for (int c = 0; c < 128; c++) vo[c] = VR[from + c];
}

// softmax(S * scale) along each row in place, one simdgroup a row; columns from N to the leading dimension become 0.
// S: (rows, ld). P: N, ld. SC: scale. Threadgroups [rows, 1, 1] of [32, 1, 1].
[[kernel]] void qi_softmax(
  device bfloat* S [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  const constant float* SC [[buffer(2)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int N = P[0], ld = P[1];
  device bfloat* s = S + long(tg.x) * ld;
  float top = -3.0e38f;
  for (int c = int(lane); c < N; c += 32) top = max(top, float(s[c]));
  top = simd_max(top);
  float mass = 0.0f;
  for (int c = int(lane); c < N; c += 32) mass += exp((float(s[c]) - top) * SC[0]);
  const float inv = 1.0f / simd_sum(mass);
  for (int c = int(lane); c < ld; c += 32) s[c] = c < N ? bfloat(exp((float(s[c]) - top) * SC[0]) * inv) : bfloat(0.0f);
}

// (H, R, 128) to (R, H * 128), one thread per (row, head). P: R, H. Threads [R, H, 1].
[[kernel]] void qi_rows(
  const device bfloat* OH [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device bfloat* OR [[buffer(2)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], H = P[1];
  if (int(at.x) >= R || int(at.y) >= H) return;
  const device bfloat* from = OH + (long(at.y) * R + at.x) * 128;
  device bfloat* to = OR + (long(at.x) * H + at.y) * 128;
  for (int c = 0; c < 128; c++) to[c] = from[c];
}

// X += G * Y by column, the product rounded to bf16 first as the reference does. X, Y: (R, C). G: (C). P: R, C.
// Threads [C, R, 1].
[[kernel]] void qi_gate_add(
  device bfloat* X [[buffer(0)]],
  const device bfloat* Y [[buffer(1)]],
  const device bfloat* G [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], C = P[1];
  if (int(at.x) >= C || int(at.y) >= R) return;
  const long i = long(at.y) * C + at.x;
  X[i] = bfloat(float(X[i]) + float(bfloat(float(G[at.x]) * float(Y[i]))));
}

// H = silu(A) * B, each stage rounded to bf16. P: count. Threads [1024, ceil(count / 1024), 1].
[[kernel]] void qi_swiglu(
  const device bfloat* A [[buffer(0)]],
  const device bfloat* B [[buffer(1)]],
  const constant int64_t* P [[buffer(2)]],
  device bfloat* H [[buffer(3)]],
  uint3 at [[thread_position_in_grid]]) {
  const long i = long(at.y) * 1024 + at.x;
  if (i >= P[0]) return;
  const float a = float(A[i]);
  const float gate = float(bfloat(a * float(bfloat(1.0f / (1.0f + exp(-a))))));
  H[i] = bfloat(gate * float(B[i]));
}
