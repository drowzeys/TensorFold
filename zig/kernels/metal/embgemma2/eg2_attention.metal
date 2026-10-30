// EmbeddingGemma-2's attention kernels, after eg2.metal (native/embgemma2_model.zig).

// A simdgroup's head row (x[i] = lane + 32 i): RMSNorm, then with w its scale and RoPE (i pairs with i + D / 64).
template <int D>
inline void eg2_head_norm(thread float* x, const device bfloat* w, const device bfloat* rope, long at, long sin_at,
                          float eps, uint lane) {
  constexpr int per = D / 32;
  float s = 0.0f;
  TF_UNROLL
  for (int i = 0; i < per; ++i) s += x[i] * x[i];
  const float r = eg2_inv_rms(simd_sum(s), D, eps);
  TF_UNROLL
  for (int i = 0; i < per; ++i) x[i] = eg2_rbf(w ? x[i] * r * float(w[lane + 32 * i]) : x[i] * r);
  if (!w) return;
  TF_UNROLL
  for (int i = 0; i < per / 2; ++i) {
    const uint c = lane + 32 * i;
    const float2 ab = eg2_rotate(x[i], x[i + per / 2], float(rope[at + c]), float(rope[sin_at + c]));
    x[i] = ab.x;
    x[i + per / 2] = ab.y;
  }
}

struct eg2_attn_p { uint rows; uint hq; uint hkv; };

// softmax(q k^T) v over keys lo[row] <= key < hi[row], scale 1: a simdgroup per (row, q head).
template <int D>
inline void eg2_attend(const device bfloat* qkv, device bfloat* out, const device int* lo, const device int* hi,
                       constant eg2_attn_p& p, uint gid, uint lane) {
  constexpr int per = D / 32;
  const uint warp = gid;
  if (warp >= p.rows * p.hq) return;
  const uint row = warp / p.hq, head = warp % p.hq;
  const uint kvh = head / (p.hq / p.hkv);
  const long stride = long(p.hq + 2 * p.hkv) * D;
  const device bfloat* q = qkv + row * stride + head * D;
  float qv[per], acc[per];
  for (int i = 0; i < per; ++i) {
    qv[i] = float(q[lane + 32 * i]);
    acc[i] = 0.0f;
  }
  const int a = lo[row], b = hi[row];
  float m = -INFINITY, l = 0.0f;
  for (int j = a; j < b; ++j) {
    const device bfloat* k = qkv + j * stride + (p.hq + kvh) * D;
    const device bfloat* v = qkv + j * stride + (p.hq + p.hkv + kvh) * D;
    EG2_ACC s = 0.0f;
    for (int i = 0; i < per; ++i) s += EG2_ACC(qv[i] * float(k[lane + 32 * i]));
    const float sc = simd_sum(float(s));
    const float mn = max(m, sc);
    const float c = precise::exp(m - mn);
    const float e = precise::exp(sc - mn);
    l = l * c + e;
    for (int i = 0; i < per; ++i) acc[i] = fma(e, float(v[lane + 32 * i]), acc[i] * c);
    m = mn;
  }
  device bfloat* o = out + (long(row) * p.hq + head) * D;
  for (int i = 0; i < per; ++i) o[lane + 32 * i] = bfloat(acc[i] / l);
}

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_attn256(
    const device bfloat* qkv [[buffer(0)]], device bfloat* out [[buffer(1)]], const device int* lo [[buffer(2)]],
    const device int* hi [[buffer(3)]], constant eg2_attn_p& p [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  eg2_attend<256>(qkv, out, lo, hi, p, g * 4 + sg, lane);
}

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_attn512(
    const device bfloat* qkv [[buffer(0)]], device bfloat* out [[buffer(1)]], const device int* lo [[buffer(2)]],
    const device int* hi [[buffer(3)]], constant eg2_attn_p& p [[buffer(4)]], uint g [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  eg2_attend<512>(qkv, out, lo, hi, p, g * 4 + sg, lane);
}

// Short batches (32 rows at most): q/k/v norms and RoPE inside; lane j scores key j, the softmax crosses lanes.
struct eg2_short_p { uint rows; uint hq; uint hkv; uint positions; float eps; };

template <int D>
inline void eg2_attend_short(const device bfloat* qkv, device bfloat* out, const device int* lo, const device int* hi,
                             const device int* pos, const device bfloat* qn, const device bfloat* kn,
                             const device bfloat* rope, constant eg2_short_p& p, threadgroup float* qs, uint gid,
                             uint lane) {
  constexpr int per = D / 32, half_ = D / 2;
  if (gid >= p.rows * p.hq) return;
  const uint row = gid / p.hq, head = gid % p.hq;
  const uint kvh = head / (p.hq / p.hkv);
  const long stride = long(p.hq + 2 * p.hkv) * D;
  const long sin_at = long(p.positions) * half_;
  float qv[per];
  TF_UNROLL
  for (int i = 0; i < per; ++i) qv[i] = float(qkv[row * stride + head * D + lane + 32 * i]);
  eg2_head_norm<D>(qv, qn, rope, long(pos[row]) * half_, sin_at + long(pos[row]) * half_, p.eps, lane);
  TF_UNROLL
  for (int i = 0; i < per; ++i) qs[lane + 32 * i] = qv[i];
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int a = lo[row], nk = hi[row] - a;
  // lane j: key a + j's score and value a + j's norm
  float score = -INFINITY, rv = 0.0f;
  if (int(lane) < nk) {
    const int j = a + int(lane);
    const device bfloat* k = qkv + j * stride + (p.hq + kvh) * D;
    const device bfloat* v = qkv + j * stride + (p.hq + p.hkv + kvh) * D;
    float sk = 0.0f, sv = 0.0f;
    for (int c = 0; c < D; ++c) {
      const float x = float(k[c]), y = float(v[c]);
      sk += x * x;
      sv += y * y;
    }
    const float rk = eg2_inv_rms(sk, D, p.eps);
    rv = eg2_inv_rms(sv, D, p.eps);
    const long at = long(pos[j]) * half_;
    float s = 0.0f;
    for (int c = 0; c < half_; ++c) {
      const float x = eg2_rbf(float(k[c]) * rk * float(kn[c])), y = eg2_rbf(float(k[c + half_]) * rk * float(kn[c + half_]));
      const float2 r = eg2_rotate(x, y, float(rope[at + c]), float(rope[sin_at + at + c]));
      s = fma(qs[c], r.x, s);
      s = fma(qs[c + half_], r.y, s);
    }
    score = s;
  }
  const float m = simd_max(score);
  const float e = int(lane) < nk ? precise::exp(score - m) : 0.0f;
  const float l = simd_sum(e);
  float acc[per];
  TF_UNROLL
  for (int i = 0; i < per; ++i) acc[i] = 0.0f;
  for (int t = 0; t < nk; ++t) {
    const float et = simd_shuffle(e, ushort(t)), rt = simd_shuffle(rv, ushort(t));
    const device bfloat* v = qkv + (a + t) * stride + (p.hq + p.hkv + kvh) * D;
    TF_UNROLL
    for (int i = 0; i < per; ++i) acc[i] = fma(et, eg2_rbf(float(v[lane + 32 * i]) * rt), acc[i]);
  }
  device bfloat* o = out + (long(row) * p.hq + head) * D;
  TF_UNROLL
  for (int i = 0; i < per; ++i) o[lane + 32 * i] = bfloat(acc[i] / l);
}

#define EG2_SHORT(D)                                                                                               \
  [[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_attn##D##_short(                                  \
      const device bfloat* qkv [[buffer(0)]], device bfloat* out [[buffer(1)]], const device int* lo [[buffer(2)]],  \
      const device int* hi [[buffer(3)]], constant eg2_short_p& p [[buffer(4)]], const device int* pos [[buffer(5)]], \
      const device bfloat* qn [[buffer(6)]], const device bfloat* kn [[buffer(7)]],                               \
      const device bfloat* rope [[buffer(8)]], uint g [[threadgroup_position_in_grid]],                           \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                      \
    threadgroup float qs[4 * D];                                                                                   \
    eg2_attend_short<D>(qkv, out, lo, hi, pos, qn, kn, rope, p, qs + sg * D, g * 4 + sg, lane);                    \
  }
EG2_SHORT(256)
EG2_SHORT(512)

// ---- tensor-unit attention (Gemma 4's prompt attention, bidirectional, over packed inputs) ----

// Tiles of 32 rows of one input, key blocks from the input's first row: a row's bits never depend on its batch.
constant constexpr float eg2_masked = -3.402823466e+38f;
constant constexpr float eg2_log2e = 1.44269504089f;
constant constexpr int EG2_RT = 32;
constant constexpr int EG2_KS = 32;
constant constexpr int EG2_PH = 128;
constant constexpr int EG2_PP = EG2_PH + 8;

template <bool MAX>
inline void eg2_fold(thread const tfp::frag<float>& f, thread float* r) {
  TF_UNROLL
  for (short h = 0; h < 2; h++) {
    const short b = 4 * h;
    float t = MAX ? max(max(f[b], f[b + 1]), max(f[b + 2], f[b + 3])) : (f[b] + f[b + 1]) + (f[b + 2] + f[b + 3]);
    const float u = simd_shuffle_xor(t, ushort(1));
    t = MAX ? max(t, u) : t + u;
    const float w = simd_shuffle_xor(t, ushort(8));
    t = MAX ? max(t, w) : t + w;
    r[h] = MAX ? max(r[h], t) : r[h] + t;
  }
}

template <bool TR>
inline tfp::frag<bfloat> eg2_blk(const device bfloat* p, int ld, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const device bfloat* q = p + (r + home.y) * ld + c + home.x;
  return tfp::frag<bfloat>(*(const device bfloat4*)q, *(const device bfloat4*)(q + 8 * ld));
#else
  tfp::frag<bfloat> f;
  if (TR) {
    tfp::frag_get_t(f, p, ld, r, c, home);
  } else {
    tfp::frag_get(f, p, ld, r, c, home);
  }
  return f;
#endif
}

inline tfp::frag<float> eg2_prob(const threadgroup float* pt, int r, int c, short2 home) {
#ifndef TF_SIMD_FRAGS
  const threadgroup float* q = pt + (r + home.y) * EG2_PP + c + home.x;
  return tfp::frag<float>(*(const threadgroup float4*)q, *(const threadgroup float4*)(q + 8 * EG2_PP));
#else
  tfp::frag<float> f;
  tfp::frag_get(f, pt, EG2_PP, r, c, home);
  return f;
#endif
}

inline void eg2_prob_put(threadgroup float* pt, int r, int c, short2 home, thread const tfp::frag<float>& s) {
#ifndef TF_SIMD_FRAGS
  threadgroup float* q = pt + (r + home.y) * EG2_PP + c + home.x;
  *(threadgroup float4*)q = float4(s[0], s[1], s[2], s[3]);
  *(threadgroup float4*)(q + 8 * EG2_PP) = float4(s[4], s[5], s[6], s[7]);
#else
  TF_UNROLL
  for (short e = 0; e < 8; e++) pt[(r + home.y + (e >> 2) * 8) * EG2_PP + c + home.x + TF_COL(e)] = s[e];
#endif
}

// q, k, v in place in qkv; tiles (first row, rows, input start); probabilities then row maxes in tg memory.
template <int DH>
inline void eg2_attend_tiles(const device bfloat* QKV, device bfloat* O, const device int* LO, const device int* HI,
                             const device int4* T, constant eg2_attn_p& P, threadgroup float* pt, threadgroup float* rowmax,
                             uint sg, uint lane, uint3 tg) {
  using namespace tfp;
  constexpr int NS = DH / 64;
  constexpr int BKT = EG2_KS * NS;
  const int hq = int(P.hq), hkv = int(P.hkv);
  const int S = (hq + 2 * hkv) * DH, OS = hq * DH;
  const int qh = int(tg.y), kv = qh / (hq / hkv);
  const int4 tile = T[tg.x];
  const int r0 = tile.x, q_rows = tile.y, base = tile.z;
  const device bfloat* q = QKV + long(r0) * S + qh * DH;
  const device bfloat* kc = QKV + (hq + kv) * DH;
  const device bfloat* vc = QKV + (hq + hkv + kv) * DH + int(sg) * 64;
  const short2 home = frag_home(ushort(lane));
  // the keys of this thread's four rows (home.y + 8 j); a row past the batch reads none
  int ra[4], rb[4];
  TF_UNROLL
  for (short j = 0; j < 4; j++) {
    const int local = home.y + 8 * j;
    ra[j] = local < q_rows ? LO[r0 + local] : 0;
    rb[j] = local < q_rows ? HI[r0 + local] : 0;
  }
  const int A = LO[r0], B = HI[r0 + q_rows - 1];
  frag<float> acc[8];
  TF_UNROLL
  for (short i = 0; i < 8; i++) acc[i] = frag<float>(0);
  float top[4] = {eg2_masked, eg2_masked, eg2_masked, eg2_masked}, total[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int kb = (A - base) / BKT; base + kb * BKT < B; kb++) {
    const int kpos = base + kb * BKT + int(sg) * EG2_KS;
    const device bfloat* kblk = kc + long(kpos) * S;
    frag<float> s[4];
    TF_UNROLL
    for (short i = 0; i < 4; i++) s[i] = frag<float>(0);
    for (int d = 0; d < DH / 16; d++) {
      const frag<bfloat> k0 = eg2_blk<true>(kblk, S, 0, 16 * d, home);
      const frag<bfloat> k1 = eg2_blk<true>(kblk, S, 16, 16 * d, home);
      TF_UNROLL
      for (short h = 0; h < 2; h++) { // rows past q_rows read the padded scratch; their outputs stay unwritten
        const frag<bfloat> qf = eg2_blk<false>(q, S, 16 * h, 16 * d, home);
        mma_16x32<false, true>(s[2 * h], s[2 * h + 1], qf, k0, k1);
      }
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int j = 2 * (i >> 1) + (e >> 2);
        const int col = kpos + 16 * (i & 1) + home.x + TF_COL(e);
        s[i][e] = (col < ra[j] || col >= rb[j]) ? eg2_masked : s[i][e] * eg2_log2e;
      }
    }
    float top_new[4] = {top[0], top[1], top[2], top[3]};
    TF_UNROLL
    for (short i = 0; i < 4; i++) eg2_fold<true>(s[i], top_new + 2 * (i >> 1));
    if (home.x == 0) {
      TF_UNROLL
      for (short j = 0; j < 4; j++) rowmax[int(sg) * EG2_RT + home.y + 8 * j] = top_new[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      for (int t = 0; t < NS; t++) top_new[j] = max(top_new[j], rowmax[t * EG2_RT + home.y + 8 * j]);
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        s[i][e] = s[i][e] == eg2_masked ? 0.0f : fast::exp2(s[i][e] - top_new[2 * (i >> 1) + (e >> 2)]);
      }
    }
    float keep[4];
    TF_UNROLL
    for (short j = 0; j < 4; j++) {
      keep[j] = top[j] == eg2_masked ? 0.0f : fast::exp2(top[j] - top_new[j]);
      top[j] = top_new[j];
      total[j] = total[j] * keep[j];
    }
    TF_UNROLL
    for (short i = 0; i < 4; i++) eg2_fold<false>(s[i], total + 2 * (i >> 1));
    TF_UNROLL
    for (short i = 0; i < 8; i++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * keep[2 * (i >> 2) + (e >> 2)];
    }
    for (int ph = 0; ph < BKT / EG2_PH; ph++) {
      if (ph > 0) threadgroup_barrier(mem_flags::mem_threadgroup); // the phase before has read the tile
      if (int(sg) / (EG2_PH / EG2_KS) == ph) {
        TF_UNROLL
        for (short i = 0; i < 4; i++) eg2_prob_put(pt, 16 * (i >> 1), int(sg) * EG2_KS - ph * EG2_PH + 16 * (i & 1), home, s[i]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int kk = 0; kk < EG2_PH / 16; kk++) {
        const int kp = base + kb * BKT + ph * EG2_PH + 16 * kk;
        if (kp >= B) break; // keys past the tile's last: masked, and their rows may be another batch's
        const int valid = min(16, B - kp);
        const device bfloat* vblk = vc + long(kp) * S;
        frag<bfloat> v0, v1;
        TF_UNROLL
        for (short d = 0; d < 4; d += 2) {
          if (valid == 16) {
            v0 = eg2_blk<false>(vblk, S, 0, 16 * d, home);
            v1 = eg2_blk<false>(vblk, S, 0, 16 * d + 16, home);
          } else {
            frag_get_in(v0, vblk, S, 0, 16 * d, home, valid, DH);
            frag_get_in(v1, vblk, S, 0, 16 * d + 16, home, valid, DH);
          }
          TF_UNROLL
          for (short h = 0; h < 2; h++) {
            const frag<float> pf = eg2_prob(pt, 16 * h, 16 * kk, home);
            mma_16x32<false, false>(acc[4 * h + d], acc[4 * h + d + 1], pf, v0, v1);
          }
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (home.x == 0) {
    TF_UNROLL
    for (short j = 0; j < 4; j++) rowmax[int(sg) * EG2_RT + home.y + 8 * j] = total[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[4];
  TF_UNROLL
  for (short j = 0; j < 4; j++) {
    float t = 0.0f;
    for (int u = 0; u < NS; u++) t += rowmax[u * EG2_RT + home.y + 8 * j];
    inv[j] = t > 0.0f ? 1.0f / t : 0.0f;
  }
  device bfloat* o = O + long(r0) * OS + qh * DH + int(sg) * 64;
  TF_UNROLL
  for (short i = 0; i < 8; i++) {
    TF_UNROLL
    for (short e = 0; e < 8; e++) acc[i][e] = acc[i][e] * inv[2 * (i >> 2) + (e >> 2)];
    frag_put_in(acc[i], o, OS, 16 * (i >> 2), 16 * (i & 3), home, q_rows, 64);
  }
}

[[max_total_threads_per_threadgroup(128)]] [[kernel]] void eg2_attn256_tiled(
    const device bfloat* qkv [[buffer(0)]], device bfloat* out [[buffer(1)]], const device int* lo [[buffer(2)]],
    const device int* hi [[buffer(3)]], constant eg2_attn_p& p [[buffer(4)]], const device int4* tiles [[buffer(5)]],
    threadgroup float* tgm [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  eg2_attend_tiles<256>(qkv, out, lo, hi, tiles, p, tgm, tgm + EG2_RT * EG2_PP, sg, lane, tg);
}

[[max_total_threads_per_threadgroup(256)]] [[kernel]] void eg2_attn512_tiled(
    const device bfloat* qkv [[buffer(0)]], device bfloat* out [[buffer(1)]], const device int* lo [[buffer(2)]],
    const device int* hi [[buffer(3)]], constant eg2_attn_p& p [[buffer(4)]], const device int4* tiles [[buffer(5)]],
    threadgroup float* tgm [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  eg2_attend_tiles<512>(qkv, out, lo, hi, tiles, p, tgm, tgm + EG2_RT * EG2_PP, sg, lane, tg);
}

