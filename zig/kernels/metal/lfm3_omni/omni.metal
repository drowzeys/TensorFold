// LFM3-Omni on Metal: the four Liquid stacks' ops (speech encoder, backbone, depth decoder, Liquid codec decoder), the
// speech encoder's strided convolutions, the depth decoder's inputs and picks, the codec's spectra and the log-mel
// front end. Ported from zig/kernels/cuda/lfm3_omni.cu: every op that rounds to bf16 in the reference rounds here.
#include "om_common.h"

// ---- norms

struct NormArgs {
  int D, ld, ldy;
  float eps;
  int mode, ldw, has_r, has_pw;
};

// RMSNorm over rows of D (a group of NT a row: 1024 for the few-row path, where a row is all there is). With R: X = bf16(X + R) first (written back to X), R itself first
// normed by PW when given (a sandwich layer's post norm, mode 0). mode 0: bf16(w * bf16(x * rsqrt)); mode 1:
// bf16(x * rsqrt * w). W rows `ldw` apart (0: one weight for all rows).
template <uint NT>
inline void rmsnorm_body(device bf* X, device const bf* R, device const bf* Wb, device bf* Y, device const bf* PW, constant NormArgs& a,
                         uint row, uint tid, threadgroup float* red) {
  device const bf* W = Wb + (long)row * a.ldw;
  device bf* x = X + (long)row * a.ld;
  device bf* y = Y + (long)row * a.ldy;
  device const bf* r = R + (long)row * a.ld;
  float inv_r = 0.0f;
  if (a.has_r && a.has_pw) {
    float sr = 0.0f;
    for (int i = tid; i < a.D; i += NT) sr += f(r[i]) * f(r[i]);
    inv_r = precise::rsqrt(group_sum<NT>(sr, red, tid) / float(a.D) + a.eps);
  }
  float ss = 0.0f;
  for (int i = tid; i < a.D; i += NT) {
    float v = f(x[i]);
    if (a.has_r) {
      const float add = a.has_pw ? rb(f(PW[i]) * rb(f(r[i]) * inv_r)) : f(r[i]);
      v = rb(v + add);
      x[i] = b(v);
    }
    ss += v * v;
  }
  const float inv = precise::rsqrt(group_sum<NT>(ss, red, tid) / float(a.D) + a.eps);
  for (int i = tid; i < a.D; i += NT) {
    const float v = f(x[i]) * inv;
    y[i] = a.mode == 0 ? b(f(W[i]) * rb(v)) : b(v * f(W[i]));
  }
}

#define OM_RMSNORM(NAME, NT)                                                                                                       \
  [[max_total_threads_per_threadgroup(NT)]] kernel void NAME(                                                                       \
      device bf* X [[buffer(0)]], device const bf* R [[buffer(1)]], device const bf* Wb [[buffer(2)]], device bf* Y [[buffer(3)]],   \
      device const bf* PW [[buffer(4)]], constant NormArgs& a [[buffer(5)]], uint row [[threadgroup_position_in_grid]],             \
      uint tid [[thread_index_in_threadgroup]]) {                                                                                  \
    threadgroup float red[NT / 32];                                                                                                \
    rmsnorm_body<NT>(X, R, Wb, Y, PW, a, row, tid, red);                                                                           \
  }
OM_RMSNORM(om_rmsnorm, 256)
OM_RMSNORM(om_rmsnorm_1k, 1024)

struct HeadNormArgs {
  int T, H, hd, ld;
  float eps;
};

// Per-head RMSNorm (mode 0 rounding) in place: one simdgroup a (row, head).
kernel void om_head_norm(device bf* X [[buffer(0)]], device const bf* W [[buffer(1)]], constant HeadNormArgs& a [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
  const int gw = gid >> 5, lane = gid & 31;
  if (gw >= a.T * a.H) return;
  device bf* x = X + (long)(gw / a.H) * a.ld + (gw % a.H) * a.hd;
  float ss = 0.0f;
  for (int i = lane; i < a.hd; i += 32) ss += f(x[i]) * f(x[i]);
  ss = simd_sum(ss);
  const float inv = precise::rsqrt(ss / float(a.hd) + a.eps);
  for (int i = lane; i < a.hd; i += 32) x[i] = b(f(W[i]) * rb(f(x[i]) * inv));
}

struct LayerNormArgs {
  int D;
  float eps;
};

// LayerNorm with bias over rows of D in fp32, one rounding (a group a row).
[[max_total_threads_per_threadgroup(256)]] kernel void om_layernorm(device const bf* X [[buffer(0)]], device const bf* W [[buffer(1)]],
                                                                 device const bf* Bb [[buffer(2)]], device bf* Y [[buffer(3)]],
                                                                 constant LayerNormArgs& a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                                                                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float red[8];
  device const bf* x = X + (long)row * a.D;
  float s = 0.0f;
  for (int i = tid; i < a.D; i += 256) s += f(x[i]);
  const float mean = group_sum<256>(s, red, tid) / float(a.D);
  float v = 0.0f;
  for (int i = tid; i < a.D; i += 256) {
    const float d = f(x[i]) - mean;
    v += d * d;
  }
  const float rstd = precise::rsqrt(group_sum<256>(v, red, tid) / float(a.D) + a.eps);
  for (int i = tid; i < a.D; i += 256) Y[(long)row * a.D + i] = b((f(x[i]) - mean) * rstd * f(W[i]) + f(Bb[i]));
}

// ---- rotary, caches, positions

struct RopeArgs {
  int T, H, hd, ld, pos0, mode;
};

// RoPE in place (rotate-half pairs p, p + hd/2), row t at position pos0 + t; INV: hd/2 inverse frequencies (zero past
// the rotated share). mode 0: fp32 cos/sin; 1: cos/sin rounded to bf16; 2: every product and the sum rounded.
kernel void om_rope(device bf* X [[buffer(0)]], device const float* INV [[buffer(1)]], constant RopeArgs& a [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  const int half_ = a.hd >> 1;
  const long i = gid;
  if (i >= (long)a.T * a.H * half_) return;
  const int p = int(i % half_), h = int((i / half_) % a.H), t = int(i / ((long)half_ * a.H));
  device bf* x = X + (long)t * a.ld + h * a.hd;
  const float fr = float(a.pos0 + t) * INV[p];
  float c = precise::cos(fr), s = precise::sin(fr);
  if (a.mode != 0) {
    c = rb(c);
    s = rb(s);
  }
  const float xa = f(x[p]), z = f(x[p + half_]);
  if (a.mode == 2) {
    x[p] = b(rb(xa * c) + rb(-z * s));
    x[p + half_] = b(rb(z * c) + rb(xa * s));
  } else {
    x[p] = b(xa * c - z * s);
    x[p + half_] = b(z * c + xa * s);
  }
}

struct CopyRowsArgs {
  int T, n, lds, ldd;
};

kernel void om_copy_rows(device const bf* S [[buffer(0)]], device bf* D [[buffer(1)]], constant CopyRowsArgs& a [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.n) return;
  const int t = int(i / a.n), c = int(i % a.n);
  D[(long)t * a.ldd + c] = S[(long)t * a.lds + c];
}

// n 32-bit words from the argument bytes (small host writes in order with the dispatches).
kernel void om_set_words(device uint* D [[buffer(0)]], constant uint* V [[buffer(1)]], constant uint& n [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid < n) D[gid] = V[gid];
}

kernel void om_zero_words(device uint* D [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) D[gid] = 0;
}

kernel void om_copy_words(device const uint* S [[buffer(0)]], device uint* D [[buffer(1)]], constant uint& n [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid < n) D[gid] = S[gid];
}

// ---- Liquid's gated short conv

struct ConvArgs {
  int ld, ldy, T, D, K;
};

// y = C * conv(B * x) over T rows (B | C | x, rows `ld` apart), K-1 rows of history in STATE [D, K-1]; a thread a
// channel, the state updated.
kernel void om_short_conv(device const bf* BCx [[buffer(0)]], device const bf* Wc [[buffer(1)]], device bf* STATE [[buffer(2)]],
                          device bf* Y [[buffer(3)]], constant ConvArgs& a [[buffer(4)]], uint d [[thread_position_in_grid]]) {
  if (int(d) >= a.D) return;
  const int K = a.K, D = a.D;
  float hist[8];
  float w[8];
  for (int j = 0; j < K; ++j) w[j] = f(Wc[d * K + j]);
  for (int j = 0; j < K - 1; ++j) hist[j] = f(STATE[d * (K - 1) + j]);
  for (int t = 0; t < a.T; ++t) {
    device const bf* row = BCx + (long)t * a.ld;
    const float bx = rb(f(row[d]) * f(row[2 * D + d]));
    float acc = 0.0f;
    for (int j = 0; j < K - 1; ++j) acc += w[j] * hist[j];
    acc += w[K - 1] * bx;
    for (int j = 0; j + 1 < K - 1; ++j) hist[j] = hist[j + 1];
    if (K > 1) hist[K - 2] = bx;
    Y[(long)t * a.ldy + d] = b(f(row[D + d]) * rb(acc));
  }
  for (int j = 0; j < K - 1; ++j) STATE[d * (K - 1) + j] = b(hist[j]);
}

// As om_short_conv for many rows, a thread a (row, channel); the state is left alone (om_conv_state follows).
kernel void om_short_conv_rows(device const bf* BCx [[buffer(0)]], device const bf* Wc [[buffer(1)]], device const bf* STATE [[buffer(2)]],
                               device bf* Y [[buffer(3)]], constant ConvArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.D) return;
  const int K = a.K, D = a.D;
  const int t = int(i / D), d = int(i % D);
  float acc = 0.0f;
  for (int j = 0; j < K; ++j) {
    const int src = t - (K - 1) + j;
    float v;
    if (src >= 0) {
      device const bf* row = BCx + (long)src * a.ld;
      v = rb(f(row[d]) * f(row[2 * D + d]));
    } else {
      v = f(STATE[d * (K - 1) + (K - 1 + src)]);
    }
    acc += f(Wc[d * K + j]) * v;
  }
  Y[(long)t * a.ldy + d] = b(f(BCx[(long)t * a.ld + D + d]) * rb(acc));
}

kernel void om_conv_state(device const bf* BCx [[buffer(0)]], device bf* STATE [[buffer(1)]], constant ConvArgs& a [[buffer(2)]],
                          uint d [[thread_position_in_grid]]) {
  if (int(d) >= a.D) return;
  const int K = a.K, D = a.D, T = a.T;
  for (int j = 0; j < K - 1; ++j) {
    const int src = T - (K - 1) + j;
    if (src >= 0) {
      device const bf* row = BCx + (long)src * a.ld;
      STATE[d * (K - 1) + j] = b(f(row[d]) * f(row[2 * D + d]));
    } else {
      STATE[d * (K - 1) + j] = STATE[d * (K - 1) + j + T];
    }
  }
}

// ---- attention

struct AttnArgs {
  int ldq, ldkv, ldo, T, H, Hkv, hd, pos0, kv_len, window, causal;
  float scale;
  int has_gate, ldg;
};

// Many query rows: a group takes 16 rows of one head and walks the cache in tiles of 32 keys held in threadgroup
// memory, with an online softmax per row (4 simdgroups, 4 rows each). Causal rows see keys up to their own position,
// `window` > 0 the last `window` keys, causal 0 every key; LFM3's per-head sigmoid output gate (G, rows ldg apart)
// applied when given. hd <= 128, hd % 32 == 0.
#define AT_Q 16
#define AT_K 32
[[max_total_threads_per_threadgroup(128)]] kernel void om_attention_tiles(device const bf* Q [[buffer(0)]], device const bf* Kc [[buffer(1)]],
                                                                       device const bf* Vc [[buffer(2)]], device bf* O [[buffer(3)]],
                                                                       device const bf* G [[buffer(4)]], constant AttnArgs& a [[buffer(5)]],
                                                                       uint2 tg [[threadgroup_position_in_grid]],
                                                                       uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float qs[AT_Q * 128];
  threadgroup float ss[AT_Q * AT_K];
  threadgroup bf ks[AT_K * 130];
  threadgroup bf vs[AT_K * 130];
  const int hd = a.hd, hdp = hd + 2;
  const int h = tg.y, kh = h / (a.H / a.Hkv);
  const int r0 = tg.x * AT_Q;
  const int warp = tid >> 5, lane = tid & 31;
  const int dpl = hd >> 5;
  for (int i = tid; i < AT_Q * hd; i += 128) {
    const int r = i / hd, c = i % hd;
    qs[i] = r0 + r < a.T ? f(Q[(long)(r0 + r) * a.ldq + h * hd + c]) : 0.0f;
  }
  const int last = min(a.T, r0 + AT_Q) - 1;
  int hi = a.causal ? a.pos0 + last + 1 : a.kv_len;
  hi = min(hi, a.kv_len);
  const int lo = (a.causal && a.window > 0) ? max(0, a.pos0 + r0 + 1 - a.window) : 0;
  float m[4], l[4], acc[4][4];
  for (int x = 0; x < 4; ++x) {
    m[x] = -INFINITY;
    l[x] = 0.0f;
    for (int c = 0; c < 4; ++c) acc[x][c] = 0.0f;
  }
  for (int k0 = lo; k0 < hi; k0 += AT_K) {
    const int nk = min(AT_K, hi - k0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < nk * hd; i += 128) {
      const int k = i / hd, c = i % hd;
      const long g = (long)(k0 + k) * a.ldkv + kh * hd + c;
      ks[k * hdp + c] = Kc[g];
      vs[k * hdp + c] = Vc[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < AT_Q * AT_K; i += 128) {
      const int r = i / AT_K, k = i % AT_K;
      const int pos = a.pos0 + r0 + r, kp = k0 + k;
      float s = -INFINITY;
      if (k < nk && r0 + r < a.T && (!a.causal || kp <= pos) && (!(a.causal && a.window > 0) || kp > pos - a.window)) {
        s = 0.0f;
        threadgroup const float* qq = qs + r * hd;
        threadgroup const bf* kr = ks + k * hdp;
        for (int c = 0; c < hd; c += 2) s += qq[c] * f(kr[c]) + qq[c + 1] * f(kr[c + 1]);
        s *= a.scale;
      }
      ss[i] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int x = 0; x < 4; ++x) {
      const int r = warp * 4 + x;
      threadgroup float* sr = ss + r * AT_K;
      const float mx = simd_max(sr[lane]);
      const float mn = max(m[x], mx);
      if (mn == -INFINITY) continue;
      const float scl = precise::exp(m[x] - mn);
      const float p0 = precise::exp(sr[lane] - mn);
      sr[lane] = p0;
      const float ps = simd_sum(p0);
      l[x] = l[x] * scl + ps;
      m[x] = mn;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int c = 0; c < 4; ++c) acc[x][c] *= scl;
      for (int k = 0; k < nk; ++k) {
        const float p = sr[k];
        threadgroup const bf* vr = vs + k * hdp;
        for (int c = 0; c < 4; ++c)
          if (c < dpl) acc[x][c] += p * f(vr[lane + 32 * c]);
      }
    }
  }
  for (int x = 0; x < 4; ++x) {
    const int r = r0 + warp * 4 + x;
    if (r >= a.T) continue;
    const float inv = 1.0f / l[x];
    const float g = a.has_gate ? rb(sigm(f(G[(long)r * a.ldg + h]))) : 1.0f;
    for (int c = 0; c < 4; ++c)
      if (c < dpl) {
        const bf o = b(acc[x][c] * inv);
        O[(long)r * a.ldo + h * hd + lane + 32 * c] = a.has_gate ? b(f(o) * g) : o;
      }
  }
}

// ---- elementwise

struct SwigluArgs {
  int T, F, inter;
};

// y = bf16(bf16(silu(a)) * c) from rows of 2F: a | c halves, or (inter) pairs a_j, c_j side by side.
kernel void om_swiglu(device const bf* AC [[buffer(0)]], device bf* Y [[buffer(1)]], constant SwigluArgs& a [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.F) return;
  const long t = i / a.F;
  const int c = int(i % a.F);
  const long ia = a.inter ? t * 2 * a.F + 2 * c : t * 2 * a.F + c;
  const long ic = a.inter ? ia + 1 : ia + a.F;
  const float x = f(AC[ia]);
  Y[i] = b(rb(x / (1.0f + precise::exp(-x))) * f(AC[ic]));
}

struct BiasActArgs {
  int rows, cols, per_row, act, has_bias;
};

// Y = act(bf16(A + bias)) from an fp32 product; bias per column (per_row 0) or per row (1); act 1 SiLU, 2 exact GELU.
kernel void om_bias_act(device const float* A [[buffer(0)]], device const bf* Bias [[buffer(1)]], device bf* Y [[buffer(2)]],
                        constant BiasActArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.rows * a.cols) return;
  const int r = int(i / a.cols), c = int(i % a.cols);
  float v = A[i] + (a.has_bias ? f(Bias[a.per_row ? r : c]) : 0.0f);
  v = rb(v);
  if (a.act == 1) v = v / (1.0f + precise::exp(-v));
  else if (a.act == 2) v = 0.5f * v * (1.0f + erf_f(v * 0.70710678118654752f));
  Y[i] = b(v);
}

struct EmbedArgs {
  int T, D, NQ, size;
};

// Y[t] = TABLE[IDS[t]]; ids < 0 leave the row alone.
kernel void om_embed(device const bf* TABLE [[buffer(0)]], device const int* IDS [[buffer(1)]], device bf* Y [[buffer(2)]],
                     constant EmbedArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.D) return;
  const int t = int(i / a.D), c = int(i % a.D);
  const int id = IDS[t];
  if (id >= 0) Y[i] = TABLE[(long)id * a.D + c];
}

// A speech frame's input row: the fp32 sum over the codebooks of TABLE[code + q * size], one rounding.
kernel void om_embed_codes(device const bf* TABLE [[buffer(0)]], device const int* CODES [[buffer(1)]], device bf* Y [[buffer(2)]],
                           constant EmbedArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.D) return;
  const int t = int(i / a.D), c = int(i % a.D);
  float s = 0.0f;
  for (int q = 0; q < a.NQ; ++q) s += f(TABLE[((long)CODES[t * a.NQ + q] + (long)q * a.size) * a.D + c]);
  Y[i] = b(s);
}

// The depth decoder's slots: X[q] = bf16(DIN[q] + (MASKED[q] ? MASK : TABLES[q][CODES[q]])).
kernel void om_depth_in(device const bf* DIN [[buffer(0)]], device const bf* MASK [[buffer(1)]], device const bf* TABLES [[buffer(2)]],
                        device const int* CODES [[buffer(3)]], device const int* MASKED [[buffer(4)]], device bf* X [[buffer(5)]],
                        constant EmbedArgs& a [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  const int i = gid;
  if (i >= a.NQ * a.D) return;
  const int q = i / a.D, c = i % a.D;
  const float e = MASKED[q] ? f(MASK[c]) : f(TABLES[((long)q * a.size + CODES[q]) * a.D + c]);
  X[i] = b(f(DIN[i]) + e);
}

struct Conv2dArgs {
  int CIN, H, W, C, H2, W2, silu;
};

// A stride-2 3x3 convolution (padding 1) over [C, H, W] planes, bias added and rounded; input channel 0 for every
// output channel when CIN is 1, else its own (depthwise); then SiLU (rounded) when asked.
kernel void om_conv2d_s2(device const bf* X [[buffer(0)]], device const bf* Wt [[buffer(1)]], device const bf* Bias [[buffer(2)]],
                         device bf* Y [[buffer(3)]], constant Conv2dArgs& a [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.C * a.H2 * a.W2) return;
  const int c = int(i / ((long)a.H2 * a.W2)), h = int((i / a.W2) % a.H2), w = int(i % a.W2);
  device const bf* x = X + (long)(a.CIN == 1 ? 0 : c) * a.H * a.W;
  float acc = 0.0f;
  for (int kh = 0; kh < 3; ++kh) {
    const int ih = 2 * h - 1 + kh;
    if (ih < 0 || ih >= a.H) continue;
    for (int kw = 0; kw < 3; ++kw) {
      const int iw = 2 * w - 1 + kw;
      if (iw < 0 || iw >= a.W) continue;
      acc += f(Wt[c * 9 + kh * 3 + kw]) * f(x[(long)ih * a.W + iw]);
    }
  }
  float v = rb(acc + f(Bias[c]));
  if (a.silu) v = rb(v / (1.0f + precise::exp(-v)));
  Y[i] = b(v);
}

kernel void om_f32_bf16(device const float* X [[buffer(0)]], device bf* Y [[buffer(1)]], constant uint& n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < n) Y[gid] = b(X[gid]);
}

kernel void om_bf16_f32(device const bf* X [[buffer(0)]], device float* Y [[buffer(1)]], constant uint& n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < n) Y[gid] = f(X[gid]);
}

struct FsqArgs {
  int n, NQ;
};

// The codec's finite scalar quantizer read back: CODES [n, NQ] to latents [n, NQ * 4], levels 8 7 6 6.
kernel void om_fsq_decode(device const int* CODES [[buffer(0)]], device bf* L [[buffer(1)]], constant FsqArgs& a [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  const int i = gid;
  if (i >= a.n * a.NQ) return;
  const int t = i / a.NQ, q = i % a.NQ;
  const int levels[4] = {8, 7, 6, 6};
  const int basis[4] = {1, 8, 56, 336};
  const int idx = CODES[t * a.NQ + q];
  for (int j = 0; j < 4; ++j) {
    const int hf = levels[j] / 2;
    const int digit = (idx / basis[j]) % levels[j] - hf;
    L[(long)t * a.NQ * 4 + q * 4 + j] = b(float(digit) / float(hf));
  }
}

struct RepeatArgs {
  int n, D, rep;
};

kernel void om_repeat_rows(device const bf* X [[buffer(0)]], device bf* Y [[buffer(1)]], constant RepeatArgs& a [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.n * a.rep * a.D) return;
  const long r = i / a.D;
  Y[i] = X[(r / a.rep) * a.D + i % a.D];
}

struct PolarArgs {
  int T, NB;
};

// The codec head's rows (log-magnitude | phase) to a real | imaginary spectrum in fp32.
kernel void om_polar(device const bf* X [[buffer(0)]], device float* S [[buffer(1)]], constant PolarArgs& a [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.T * a.NB) return;
  const long t = i / a.NB;
  const int k = int(i % a.NB);
  const float m = precise::exp(f(X[t * 2 * a.NB + k])), ph = f(X[t * 2 * a.NB + a.NB + k]);
  S[t * 2 * a.NB + k] = m * precise::cos(ph);
  S[t * 2 * a.NB + a.NB + k] = m * precise::sin(ph);
}

inline void better(thread float& best, thread int& at, float v, int i) {
  if (v > best || (v == best && i < at)) {
    best = v;
    at = i;
  }
}

// ---- products of many rows: C [M, N] = A [M, K] B [K, N] (either stored transposed), fp32 accumulate on the
// simdgroup matrix units, the operands staged as fp32 tiles; a batch on the grid's z.

struct GemmArgs {
  int M, N, K, lda, ldb, ldc, ta, tb;
  long sa, sb, sc;
};

template <typename TI>
inline float ld_in(device const TI* p, long i);
template <>
inline float ld_in<bf>(device const bf* p, long i) {
  return f(p[i]);
}
template <>
inline float ld_in<float>(device const float* p, long i) {
  return p[i];
}

template <typename TO>
inline void st_out(device TO* p, long i, float v);
template <>
inline void st_out<bf>(device bf* p, long i, float v) {
  p[i] = b(v);
}
template <>
inline void st_out<float>(device float* p, long i, float v) {
  p[i] = v;
}

template <typename TI, typename TO>
inline void gemm_body(device const TI* A, device const TI* B, device TO* C, constant GemmArgs& a, uint3 tg, uint tid, uint sg,
                      uint lane, threadgroup float* As, threadgroup float* Bs, threadgroup float* Cs) {
  constexpr int BM = 64, BN = 64, BK = 16;
  A += (long)tg.z * a.sa;
  B += (long)tg.z * a.sb;
  C += (long)tg.z * a.sc;
  const int m0 = tg.y * BM, n0 = tg.x * BN;
  const int wm = (sg >> 1) * 32, wn = (sg & 1) * 32;
  simdgroup_float8x8 acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
  for (int k0 = 0; k0 < a.K; k0 += BK) {
    // A tile [BM][BK], B tile [BK][BN]: 1024 values each, 8 a thread
    for (int i = tid; i < BM * BK; i += 128) {
      int r, k;
      if (a.ta) {
        r = i % BM;
        k = i / BM;
      } else {
        r = i / BK;
        k = i % BK;
      }
      const int gm = m0 + r, gk = k0 + k;
      float v = 0.0f;
      if (gm < a.M && gk < a.K) v = ld_in<TI>(A, a.ta ? (long)gk * a.lda + gm : (long)gm * a.lda + gk);
      As[r * BK + k] = v;
    }
    for (int i = tid; i < BK * BN; i += 128) {
      int k, c;
      if (a.tb) {
        k = i % BK;
        c = i / BK;
      } else {
        k = i / BN;
        c = i % BN;
      }
      const int gk = k0 + k, gn = n0 + c;
      float v = 0.0f;
      if (gn < a.N && gk < a.K) v = ld_in<TI>(B, a.tb ? (long)gn * a.ldb + gk : (long)gk * a.ldb + gn);
      Bs[k * BN + c] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 am[4], bm[4];
      for (int i = 0; i < 4; ++i) simdgroup_load(am[i], As + (wm + i * 8) * BK + kk, BK);
      for (int j = 0; j < 4; ++j) simdgroup_load(bm[j], Bs + kk * BN + wn + j * 8, BN);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* cs = Cs + sg * 64;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], cs, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (int e = lane; e < 64; e += 32) {
        const int gm = m0 + wm + i * 8 + e / 8, gn = n0 + wn + j * 8 + e % 8;
        if (gm < a.M && gn < a.N) st_out<TO>(C, (long)gm * a.ldc + gn, cs[e]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define OM_GEMM(NAME, TI, TO)                                                                                                      \
  [[max_total_threads_per_threadgroup(128)]] kernel void NAME(device const TI* A [[buffer(0)]], device const TI* B [[buffer(1)]],  \
                                                             device TO* C [[buffer(2)]], constant GemmArgs& a [[buffer(3)]],       \
                                                             uint3 tg [[threadgroup_position_in_grid]],                             \
                                                             uint tid [[thread_index_in_threadgroup]],                              \
                                                             uint sg [[simdgroup_index_in_threadgroup]],                            \
                                                             uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float As[64 * 16];                                                                                                 \
    threadgroup float Bs[16 * 64];                                                                                                 \
    threadgroup float Cs[4 * 64];                                                                                                  \
    gemm_body<TI, TO>(A, B, C, a, tg, tid, sg, lane, As, Bs, Cs);                                                                  \
  }
OM_GEMM(om_gemm_bb, bf, bf)
OM_GEMM(om_gemm_bf, bf, float)
OM_GEMM(om_gemm_ff, float, float)

// ---- the depth decoder's picks on the device

inline ulong mix64(ulong z) {
  z += 0x9E3779B97F4A7C15ul;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
  return z ^ (z >> 31);
}

struct PickArgs {
  int p0, hn;
  float penalty, temp;
  ulong seed;
  int frame, NQ, size, pad;
};

// One pass's picks: codebook q = ORDER[p0 + group] gets its code from LOG (bf16 [NQ, size]) after the repetition
// penalty over its codes in the hn frames of HIST, greedy at temp 0 (ties to the lower index), else a Gumbel-max draw
// from a hash of seed, frame, q and the code. Writes CODES[q] and clears MASKED[q].
[[max_total_threads_per_threadgroup(256)]] kernel void om_depth_pick(device const bf* LOG [[buffer(0)]], device const int* ORDER [[buffer(1)]],
                                                                  device const int* HIST [[buffer(2)]], device int* CODES [[buffer(3)]],
                                                                  device int* MASKED [[buffer(4)]], constant PickArgs& a [[buffer(5)]],
                                                                  uint blk [[threadgroup_position_in_grid]],
                                                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uchar seen[2048];
  threadgroup float bv[8];
  threadgroup int bi[8];
  const int q = ORDER[a.p0 + blk];
  for (int i = tid; i < a.size; i += 256) seen[i] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = tid; j < a.hn; j += 256) seen[HIST[j * a.NQ + q]] = 1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float best = -INFINITY;
  int at = 0x7fffffff;
  for (int i = tid; i < a.size; i += 256) {
    float v = f(LOG[(long)q * a.size + i]);
    if (seen[i]) v = v < 0.0f ? v * a.penalty : v / a.penalty;
    if (a.temp > 0.0f) {
      const ulong h = mix64(a.seed ^ mix64(((ulong)a.frame << 32) ^ ((ulong)q << 16) ^ (ulong)i));
      const float u = (float(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
      v = v / a.temp - precise::log(-precise::log(u));
    }
    better(best, at, v, i);
  }
  for (ushort o = 16; o > 0; o >>= 1) better(best, at, simd_shuffle_xor(best, o), simd_shuffle_xor(at, o));
  if ((tid & 31) == 0) {
    bv[tid >> 5] = best;
    bi[tid >> 5] = at;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    best = bv[0];
    at = bi[0];
    for (int w = 1; w < 8; ++w) better(best, at, bv[w], bi[w]);
    CODES[q] = at;
    MASKED[q] = 0;
  }
}

struct FrameEndArgs {
  int NQ, eos, slot, push;
};

// A finished frame: any end-of-speech code makes it the all-EOS frame; with push (and no EOS) the frame joins HIST
// at row `slot` (the repetition window, a ring).
kernel void om_frame_end(device int* CODES [[buffer(0)]], device int* HIST [[buffer(1)]], constant FrameEndArgs& a [[buffer(2)]],
                         uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  bool any = false;
  for (int q = 0; q < a.NQ; ++q) any |= CODES[q] == a.eos;
  if (any)
    for (int q = 0; q < a.NQ; ++q) CODES[q] = a.eos;
  else if (a.push)
    for (int q = 0; q < a.NQ; ++q) HIST[a.slot * a.NQ + q] = CODES[q];
}

// ---- the log-mel front end: frames of the padded clip, |DFT|^2 by product, mel by product, log

struct MelArgs {
  int len, nf, nfft, hop;
};

kernel void om_mel_frames(device const float* X [[buffer(0)]], device const float* WIN [[buffer(1)]], device float* FR [[buffer(2)]],
                          constant MelArgs& a [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.nf * a.nfft) return;
  const int t = int(i / a.nfft), n = int(i % a.nfft);
  const int s = t * a.hop + n;
  FR[i] = s < a.len ? X[s] * WIN[n] : 0.0f;
}

struct PowerArgs {
  int nf, bins;
};

kernel void om_power(device const float* S [[buffer(0)]], device float* P [[buffer(1)]], constant PowerArgs& a [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.nf * a.bins) return;
  const long t = i / a.bins;
  const int k = int(i % a.bins);
  const float re = S[t * 2 * a.bins + k], im = S[t * 2 * a.bins + a.bins + k];
  const float m = precise::sqrt(re * re + im * im);
  P[i] = m * m;
}

struct LogmelArgs {
  int nf, mels;
  float floor_;
};

kernel void om_logmel(device const float* M [[buffer(0)]], device float* OUT [[buffer(1)]], constant LogmelArgs& a [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  const long i = gid;
  if (i >= (long)a.nf * a.mels) return;
  const int t = int(i / a.mels), m = int(i % a.mels);
  OUT[(long)m * a.nf + t] = (max(precise::log10(M[i]), a.floor_) + 4.0f) / 4.0f;
}
