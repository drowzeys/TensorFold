// ours: Fish Audio Drama 3 (fish_drama3 family) on CUDA. The Qwen3.5-MoE backbone's steps (zero-centred RMSNorm,
// Gated DeltaNet conv + recurrence, gated full attention, the router and the expert combine; the routed FP8 experts
// themselves run on fp8_experts.cu) and the codebook tower's (frame embeddings, enorm/hnorm mixing, interleaved
// rotary, its per-layer KV cache), the bf16 GEMV every decode row takes, and the samplers (top-k / top-p with an
// exponential race on Triton's Philox stream, so a seed draws what SGLang-Omni's sampler draws, and RAS).
//
// Every kernel that a decode frame runs reads its position and step from device memory, so a frame replays as one
// CUDA graph. Plain extern "C" symbols; built with -O3 and torch's default flags.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

typedef __nv_bfloat16 bf16;

namespace {

constexpr unsigned FULL = 0xFFFFFFFFu;

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16(x)); }  // round to bf16

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
  return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ float block_sum(float v) {
  __shared__ float part[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < nw ? part[lane] : 0.f;
    s = warp_sum(s);
    if (lane == 0) total = s;
  }
  __syncthreads();
  return total;
}

__device__ float block_max(float v) {
  __shared__ float part[32];
  __shared__ float total;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) part[w] = v;
  __syncthreads();
  if (w == 0) {
    float s = lane < nw ? part[lane] : -INFINITY;
    s = warp_max(s);
    if (lane == 0) total = s;
  }
  __syncthreads();
  return total;
}

__device__ __forceinline__ uint4 ld_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ float lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

__device__ __forceinline__ float dot8(uint4 a, uint4 b) {
  float s = lo(a.x) * lo(b.x);
  s = fmaf(hi(a.x), hi(b.x), s);
  s = fmaf(lo(a.y), lo(b.y), s);
  s = fmaf(hi(a.y), hi(b.y), s);
  s = fmaf(lo(a.z), lo(b.z), s);
  s = fmaf(hi(a.z), hi(b.z), s);
  s = fmaf(lo(a.w), lo(b.w), s);
  s = fmaf(hi(a.w), hi(b.w), s);
  return s;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.f ? x : log1pf(expf(x)); }

__device__ __forceinline__ bool is_semantic(int tok, int sem_begin, int sem_end) {
  return tok >= sem_begin && tok <= sem_end;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// GEMV: y[n] = sum_k W[n, k] x[k], W bf16 [N, K] row-major (K a multiple of 256), x bf16 [K]; y bf16 and / or fp32.
// A warp a row, 8 warps a block; each lane streams 16-byte chunks of the row, 8 chunks in flight. x staged in shared
// memory as bf16 (dynamic: K * 2 bytes).

extern "C" __global__ void __launch_bounds__(256) d3_gemv(const bf16* __restrict__ W, const bf16* __restrict__ x,
                                                          int K, int N, bf16* __restrict__ y, float* __restrict__ yf) {
  extern __shared__ uint4 xs[];
  const int chunks = K >> 3;
  for (int i = threadIdx.x; i < chunks; i += blockDim.x) xs[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint4* row = reinterpret_cast<const uint4*>(W + (size_t)n * K);
  float acc = 0.f;
  int c = lane;
  for (; c + 7 * 32 < chunks; c += 8 * 32) {
    uint4 w[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) w[u] = ld_stream(row + c + u * 32);
#pragma unroll
    for (int u = 0; u < 8; ++u) acc += dot8(w[u], xs[c + u * 32]);
  }
  for (; c < chunks; c += 32) acc += dot8(ld_stream(row + c), xs[c]);
  acc = warp_sum(acc);
  if (lane == 0) {
    if (y) y[n] = f2b(acc);
    if (yf) yf[n] = rb(acc);  // logits: rounded to bf16 as the bf16 heads return them
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Norms. Zero-centred RMSNorm (Qwen3.5 / Gemma): y = x * rsqrt(mean(x^2) + eps) * (1 + w), in fp32, rounded to bf16
// once. d3_add_rmsnorm first makes the residual stream r = bf16(r + a) (HF's bf16 residual add), then norms it.

extern "C" __global__ void d3_rmsnorm(const bf16* __restrict__ x, int x_stride, const bf16* __restrict__ w,
                                      bf16* __restrict__ y, int y_stride, int D, float eps, int zero_centred) {
  const bf16* xr = x + (size_t)blockIdx.x * x_stride;
  bf16* yr = y + (size_t)blockIdx.x * y_stride;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = b2f(xr[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float g = zero_centred ? 1.f + b2f(w[i]) : b2f(w[i]);
    yr[i] = f2b(b2f(xr[i]) * inv * g);
  }
}

// r = bf16(r + a) (a's rows a_stride apart), y = rmsnorm(r). r is [rows, D] contiguous.
extern "C" __global__ void d3_add_rmsnorm(bf16* __restrict__ r, const bf16* __restrict__ a, int a_stride,
                                          const bf16* __restrict__ w, bf16* __restrict__ y, int D, float eps) {
  bf16* rr = r + (size_t)blockIdx.x * D;
  const bf16* ar = a + (size_t)blockIdx.x * a_stride;
  bf16* yr = y + (size_t)blockIdx.x * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = rb(b2f(rr[i]) + b2f(ar[i]));
    rr[i] = f2b(v);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss) / D + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = f2b(b2f(rr[i]) * inv * (1.f + b2f(w[i])));
}

// ---------------------------------------------------------------------------------------------------------------------
// Backbone frame embeddings: text rows take embed_tokens[main]; semantic rows (main in [sem_begin, sem_end]) take
// bf16(bf16(embed_tokens[main] + codebook_embeddings[1 * cb_size + code1]) * (1 / sqrt(2))). frames int32 [rows, 11].

extern "C" __global__ void d3_embed_backbone(const int* __restrict__ frames, const bf16* __restrict__ embed,
                                             const bf16* __restrict__ cb_emb, bf16* __restrict__ out, int D,
                                             int sem_begin, int sem_end, int cb_size) {
  const int* f = frames + (size_t)blockIdx.x * 11;
  const int tok = f[0];
  const bool sem = is_semantic(tok, sem_begin, sem_end);
  int code1 = min(max(f[2], 0), cb_size - 1);
  const bf16* e = embed + (size_t)tok * D;
  const bf16* c = cb_emb + (size_t)(cb_size + code1) * D;
  bf16* o = out + (size_t)blockIdx.x * D;
  const float scale = 0.70710678118654752f;  // 1 / sqrt(1 + 1)
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = b2f(e[i]);
    o[i] = sem ? f2b(rb(v + b2f(c[i])) * scale) : e[i];
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Gated DeltaNet (linear attention) layer pieces. The fused input projection row is
// [qkv 8192 | z 4096 | b 32 | a 32] (stride `ps`). Conv: depthwise causal kernel 4 over the 8192 qkv channels with
// SiLU; the state keeps the last 3 inputs (bf16) per channel. One thread a channel walks the T rows in order.

extern "C" __global__ void d3_conv_silu(const bf16* __restrict__ proj, int ps, int T, const bf16* __restrict__ w,
                                        bf16* __restrict__ state, bf16* __restrict__ out, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const float w0 = b2f(w[c * 4 + 0]), w1 = b2f(w[c * 4 + 1]), w2 = b2f(w[c * 4 + 2]), w3 = b2f(w[c * 4 + 3]);
  float s0 = b2f(state[c * 3 + 0]), s1 = b2f(state[c * 3 + 1]), s2 = b2f(state[c * 3 + 2]);
  for (int t = 0; t < T; ++t) {
    const float x = b2f(proj[(size_t)t * ps + c]);
    float acc = s0 * w0;
    acc = fmaf(s1, w1, acc);
    acc = fmaf(s2, w2, acc);
    acc = fmaf(x, w3, acc);
    out[(size_t)t * C + c] = f2b(siluf_(acc));
    s0 = s1;
    s1 = s2;
    s2 = x;
  }
  state[c * 3 + 0] = f2b(s0);
  state[c * 3 + 1] = f2b(s1);
  state[c * 3 + 2] = f2b(s2);
}

// The recurrence, one block (a warp) per (value head h, 32-column slice): S[h] is [128 keys][128 values] fp32, each
// lane holding its value column in registers across the T rows. q / k (key head h / 2) are l2-normed in fp32, q scaled
// by 1/sqrt(128); g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b) (all fp32).
// conv: [T, 8192] (q 0..2047, k 2048..4095, v 4096..8191); o: [T, 4096] fp32.
extern "C" __global__ void __launch_bounds__(32) d3_gdn(const bf16* __restrict__ conv, const bf16* __restrict__ proj,
                                                        int ps, int T, const bf16* __restrict__ A_log,
                                                        const bf16* __restrict__ dt_bias, float* __restrict__ S,
                                                        float* __restrict__ o) {
  constexpr int DK = 128, DV = 128, NV = 32;
  const int h = blockIdx.x, slice = blockIdx.y, lane = threadIdx.x;
  const int kh = h >> 1;
  const int v = slice * 32 + lane;
  __shared__ float qs[DK], ks[DK];
  float s[DK];
  float* Sh = S + (size_t)h * DK * DV;
#pragma unroll
  for (int k = 0; k < DK; ++k) s[k] = Sh[k * DV + v];
  const float A = -expf(b2f(A_log[h]));
  const float dtb = b2f(dt_bias[h]);
  const float qscale = rsqrtf((float)DK);
  for (int t = 0; t < T; ++t) {
    const bf16* row = conv + (size_t)t * 8192;
    float qv[4], kv[4];
    float qq = 0.f, kk = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv[j] = b2f(row[kh * DK + lane + 32 * j]);
      kv[j] = b2f(row[2048 + kh * DK + lane + 32 * j]);
      qq += qv[j] * qv[j];
      kk += kv[j] * kv[j];
    }
    qq = warp_sum(qq);
    kk = warp_sum(kk);
    const float qi = rsqrtf(qq + 1e-6f) * qscale, ki = rsqrtf(kk + 1e-6f);
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qs[lane + 32 * j] = qv[j] * qi;
      ks[lane + 32 * j] = kv[j] * ki;
    }
    __syncwarp();
    const bf16* pr = proj + (size_t)t * ps;
    const float bb = sigmoidf_(b2f(pr[12288 + h]));
    const float g = A * softplusf_(b2f(pr[12288 + NV + h]) + dtb);
    const float decay = expf(g);
    const float vv = b2f(row[4096 + h * DV + v]);
    float mem = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
      s[k] *= decay;
      mem = fmaf(s[k], ks[k], mem);
    }
    const float delta = (vv - mem) * bb;
    float out = 0.f;
#pragma unroll
    for (int k = 0; k < DK; ++k) {
      s[k] = fmaf(ks[k], delta, s[k]);
      out = fmaf(s[k], qs[k], out);
    }
    o[(size_t)t * (NV * DV) + h * DV + v] = out;
  }
#pragma unroll
  for (int k = 0; k < DK; ++k) Sh[k * DV + v] = s[k];
}

// Gated RMSNorm per value head (plain gamma, fp32): y = bf16(o * rsqrt(mean(o^2) + eps) * w * silu(z)). A warp a head, 8
// heads a block: grid (T, 4).
extern "C" __global__ void d3_gated_norm(const float* __restrict__ o, const bf16* __restrict__ proj, int ps,
                                         const bf16* __restrict__ w, bf16* __restrict__ y, float eps) {
  const int t = blockIdx.x, h = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const float* orow = o + (size_t)t * 4096 + h * 128;
  const bf16* z = proj + (size_t)t * ps + 8192 + h * 128;
  float v[4], ss = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v[j] = orow[lane + 32 * j];
    ss += v[j] * v[j];
  }
  const float inv = rsqrtf(warp_sum(ss) / 128.f + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    const float n = v[j] * inv * b2f(w[i]);
    y[(size_t)t * 4096 + h * 128 + i] = f2b(n * siluf_(b2f(z[i])));
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Gated full attention (backbone and tower alike: 16 query heads, 2 KV heads, head_dim 256, rotary on the first 64).
// d3_qk_rope: from a projection row (per layout: the backbone's q_proj interleaves [q 256 | gate 256] per head; the
// tower's wqkv is [q 4096 | gate 4096 | k 512 | v 512]), q and k zero-centred-RMSNormed over the head (bf16), then
// rotated: NeoX halves (backbone) or interleaved pairs (tower) over 64 dims, theta 1e7, fp32 angles; q to qout
// [T, 16, 256], k and v into the cache rows [pos, 2, 256]. Block per (row, head of 18: 16 q then 2 k), 256 threads.
// pos[t] = pos0[0] + t.

extern "C" __global__ void d3_qk_rope(const bf16* __restrict__ proj, int ps, int q_off, int q_head_stride, int k_off,
                                      int v_off, const bf16* __restrict__ qn_w, const bf16* __restrict__ kn_w,
                                      const int* __restrict__ pos0, bf16* __restrict__ qout, bf16* __restrict__ kc,
                                      bf16* __restrict__ vc, int interleaved, float theta, float eps) {
  const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
  const int pos = pos0[0] + t;
  const bf16* row = proj + (size_t)t * ps;
  const bool isq = hh < 16;
  const bf16* src = isq ? row + q_off + hh * q_head_stride : row + k_off + (hh - 16) * 256;
  const float x = b2f(src[i]);
  const float inv = rsqrtf(block_sum(x * x) / 256.f + eps);
  const float wv = b2f((isq ? qn_w : kn_w)[i]);
  __shared__ float xn[256];
  xn[i] = rb(x * inv * (1.f + wv));
  __syncthreads();
  float outv = xn[i];
  if (i < 64) {
    int j, partner;
    float sign;
    if (interleaved) {
      j = i >> 1;
      partner = i ^ 1;
      sign = (i & 1) ? 1.f : -1.f;
    } else {
      j = i & 31;
      partner = i < 32 ? i + 32 : i - 32;
      sign = i < 32 ? -1.f : 1.f;
    }
    const float inv_freq = 1.f / powf(theta, (float)(2 * j) / 64.f);
    const float ang = (float)pos * inv_freq;
    float c = cosf(ang), s = sinf(ang);
    outv = xn[i] * c + sign * xn[partner] * s;
  }
  if (isq) {
    qout[((size_t)t * 16 + hh) * 256 + i] = f2b(outv);
  } else {
    const int g = hh - 16;
    kc[((size_t)pos * 2 + g) * 256 + i] = f2b(outv);
    vc[((size_t)pos * 2 + g) * 256 + i] = row[v_off + g * 256 + i];
  }
}

// Decode attention for rows whose keys are cache rows 0..pos (pos = pos0[0] + t): one block a (row, query head),
// 8 warps over the keys (online softmax each), merged in shared memory; out = bf16(attn) * sigmoid(gate) -> [T, 16, 256].
extern "C" __global__ void __launch_bounds__(256) d3_attn_decode(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                                 const bf16* __restrict__ vc, const int* __restrict__ pos0,
                                                                 const bf16* __restrict__ proj, int ps, int gate_off,
                                                                 int gate_head_stride, bf16* __restrict__ out) {
  const int t = blockIdx.x, h = blockIdx.y, g = h >> 3;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int L = pos0[0] + t + 1;
  const uint4 qv = reinterpret_cast<const uint4*>(q + ((size_t)t * 16 + h) * 256)[lane];
  float m = -INFINITY, l = 0.f, acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  const float scale = 0.0625f;  // 1/sqrt(256)
  for (int p = w; p < L; p += 8) {
    const uint4 kv = reinterpret_cast<const uint4*>(kc + ((size_t)p * 2 + g) * 256)[lane];
    const float sc = warp_sum(dot8(qv, kv)) * scale;
    const uint4 vv = reinterpret_cast<const uint4*>(vc + ((size_t)p * 2 + g) * 256)[lane];
    const float nm = fmaxf(m, sc);
    const float corr = expf(m - nm), e = expf(sc - nm);
    l = l * corr + e;
    acc[0] = acc[0] * corr + e * lo(vv.x);
    acc[1] = acc[1] * corr + e * hi(vv.x);
    acc[2] = acc[2] * corr + e * lo(vv.y);
    acc[3] = acc[3] * corr + e * hi(vv.y);
    acc[4] = acc[4] * corr + e * lo(vv.z);
    acc[5] = acc[5] * corr + e * hi(vv.z);
    acc[6] = acc[6] * corr + e * lo(vv.w);
    acc[7] = acc[7] * corr + e * hi(vv.w);
    m = nm;
  }
  __shared__ float sm[8], sl[8], sacc[8][256];
  if (lane == 0) {
    sm[w] = m;
    sl[w] = l;
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) sacc[w][lane * 8 + j] = acc[j];
  __syncthreads();
  const int i = threadIdx.x;
  float M = -INFINITY;
#pragma unroll
  for (int k = 0; k < 8; ++k) M = fmaxf(M, sm[k]);
  float Lsum = 0.f, A = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (sm[k] == -INFINITY) continue;
    const float c = expf(sm[k] - M);
    Lsum += sl[k] * c;
    A += sacc[k][i] * c;
  }
  const float a = rb(A / Lsum);
  const float gt = b2f(proj[(size_t)t * ps + gate_off + h * gate_head_stride + i]);
  out[((size_t)t * 16 + h) * 256 + i] = f2b(a * rb(sigmoidf_(gt)));
}

// bf16 -> fp32 (the V cache rows for the prompt PV GEMM).
extern "C" __global__ void d3_bf16_f32(const bf16* __restrict__ x, float* __restrict__ y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = b2f(x[i]);
}

// Prompt attention through cuBLAS: q regrouped per KV head for the score GEMM: qg [2][T][8][256] from q [T, 16, 256].
extern "C" __global__ void d3_q_group(const bf16* __restrict__ q, bf16* __restrict__ qg, int T) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  qg[(((size_t)(h >> 3) * T + t) * 8 + (h & 7)) * 256 + i] = q[((size_t)t * 16 + h) * 256 + i];
}

// Scores [2][T * 8][ld] fp32 (already scaled) -> causal softmax probabilities, fp32 in place (as the reference):
// row (g, t, hh) sees keys 0..pos0 + t. One block a row.
extern "C" __global__ void d3_causal_softmax(float* __restrict__ s, int T, int ld, const int* __restrict__ pos0) {
  const size_t r = blockIdx.x;
  const int t = (int)((r / 8) % T);
  const int L = pos0[0] + t + 1;
  float* sr = s + r * ld;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < L; j += blockDim.x) m = fmaxf(m, sr[j]);
  m = block_max(m);
  float sum = 0.f;
  for (int j = threadIdx.x; j < L; j += blockDim.x) sum += expf(sr[j] - m);
  const float inv = 1.f / block_sum(sum);
  __syncthreads();
  for (int j = threadIdx.x; j < ld; j += blockDim.x) sr[j] = j < L ? expf(sr[j] - m) * inv : 0.f;
}

// O [2][T][8][256] fp32 (the PV GEMM's output) -> out [T, 16, 256] = bf16(O) * bf16(sigmoid(gate)).
extern "C" __global__ void d3_attn_gate(const float* __restrict__ og, const bf16* __restrict__ proj, int ps,
                                        int gate_off, int gate_head_stride, bf16* __restrict__ out, int T) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  const float a = rb(og[(((size_t)(h >> 3) * T + t) * 8 + (h & 7)) * 256 + i]);
  const float gt = b2f(proj[(size_t)t * ps + gate_off + h * gate_head_stride + i]);
  out[((size_t)t * 16 + h) * 256 + i] = f2b(a * rb(sigmoidf_(gt)));
}

// ---------------------------------------------------------------------------------------------------------------------
// MoE. The shared projection row is [shared gate 512 | shared up 512 | router 256 | shared-expert gate 1 | pad].
// d3_router: softmax (fp32) over the 256 router logits, top 8 (ties to the lower expert), renormed (fp32).

extern "C" __global__ void d3_router(const bf16* __restrict__ sp, int ss, int r_off, int* __restrict__ picks,
                                     float* __restrict__ weights) {
  const int t = blockIdx.x, e = threadIdx.x;  // 256 threads
  const float lg = b2f(sp[(size_t)t * ss + r_off + e]);
  const float m = block_max(lg);
  const float p = expf(lg - m);
  const float sum = block_sum(p);
  float prob = p / sum;
  __shared__ float best_v[8];
  __shared__ int best_i[8];
  __shared__ float wv[32];
  __shared__ int wi[32];
  float mine = prob;
  for (int k = 0; k < 8; ++k) {
    float v = mine;
    int idx = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(FULL, v, o);
      const int oi = __shfl_xor_sync(FULL, idx, o);
      if (ov > v || (ov == v && oi < idx)) {
        v = ov;
        idx = oi;
      }
    }
    if ((e & 31) == 0) {
      wv[e >> 5] = v;
      wi[e >> 5] = idx;
    }
    __syncthreads();
    if (e == 0) {
      float bv = wv[0];
      int bi = wi[0];
      for (int j = 1; j < 8; ++j)
        if (wv[j] > bv || (wv[j] == bv && wi[j] < bi)) {
          bv = wv[j];
          bi = wi[j];
        }
      best_v[k] = bv;
      best_i[k] = bi;
    }
    __syncthreads();
    if (e == best_i[k]) mine = -1.f;
    __syncthreads();
  }
  if (e < 8) {
    float s = 0.f;
    for (int j = 0; j < 8; ++j) s += best_v[j];
    picks[t * 8 + e] = best_i[e];
    weights[t * 8 + e] = best_v[e] / s;
  }
}

// y [T, N] = bf16(bf16(silu(g)) * u) with g, u at offsets of rows `stride` apart.
extern "C" __global__ void d3_swiglu(const bf16* __restrict__ x, int stride, int g_off, int u_off, int N,
                                     bf16* __restrict__ y) {
  const int t = blockIdx.y;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
    const float g = b2f(x[(size_t)t * stride + g_off + i]), u = b2f(x[(size_t)t * stride + u_off + i]);
    y[(size_t)t * N + i] = f2b(rb(siluf_(g)) * u);
  }
}

// r = bf16(r + bf16(bf16(sum_k w[t,k] down[t*8+k]) + bf16(bf16(sigmoid(sg)) * shared))): the routed experts' down rows (fp32,
// unweighted) and the gated shared expert added to the residual stream. D = 2048, grid (T), 256 threads.
extern "C" __global__ void d3_moe_combine(bf16* __restrict__ r, const float* __restrict__ down,
                                          const float* __restrict__ w, const bf16* __restrict__ shared,
                                          const bf16* __restrict__ sp, int ss, int sg_off, int D) {
  const int t = blockIdx.x;
  const float sg = rb(sigmoidf_(b2f(sp[(size_t)t * ss + sg_off])));
  float wk[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) wk[k] = w[t * 8 + k];
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) acc = fmaf(wk[k], down[((size_t)t * 8 + k) * D + i], acc);
    const float sh = rb(sg * b2f(shared[(size_t)t * D + i]));
    const float moe = rb(rb(acc) + sh);
    r[(size_t)t * D + i] = f2b(b2f(r[(size_t)t * D + i]) + moe);
  }
}

// SGLang's per_token_group_quant_fp8 (the routed experts' dynamic activations): per row and 128 inputs,
// scale = max(amax, 1e-10) / 448, q = e4m3(clamp(x / scale)) written back as bf16 (exact); s [rows, K / 128].
// grid (rows, K / 128), 128 threads.
extern "C" __global__ void d3_act_quant(const bf16* __restrict__ x, int xs, bf16* __restrict__ q, int qs,
                                        float* __restrict__ s) {
  const int r = blockIdx.x, g = blockIdx.y, i = threadIdx.x;
  const float v = b2f(x[(size_t)r * xs + g * 128 + i]);
  const float amax = block_max(fabsf(v));
  const float sc = fmaxf(amax, 1e-10f) / 448.f;
  const float y = fminf(fmaxf(__fdiv_rn(v, sc), -448.f), 448.f);
  const __nv_fp8_e4m3 e(y);
  q[(size_t)r * qs + g * 128 + i] = f2b(float(e));
  if (i == 0) s[(size_t)r * gridDim.y + g] = sc;
}

// ---------------------------------------------------------------------------------------------------------------------
// The codebook tower. d3_tower_mix builds eh_proj's input row [enorm(e) | hnorm(h)] (bf16, 4096 wide) for tower layer
// `layer`: e = (label main token semantic ? table[label code] : 0) + (layer == 0 and the input frame semantic ?
// sum_{c=2..9} codebook_embeddings[c * cb + code_c] : 0). label frames lf [T, 11] (stride lfs ints), the code at
// column code_col; input frames inf [T, 11].

extern "C" __global__ void d3_tower_mix(const int* __restrict__ lf, int lfs, int code_col, const int* __restrict__ inf,
                                        int with_shared, const bf16* __restrict__ table,
                                        const bf16* __restrict__ cb_emb, const bf16* __restrict__ h,
                                        const bf16* __restrict__ enorm_w, const bf16* __restrict__ hnorm_w,
                                        bf16* __restrict__ out, int D, int sem_begin, int sem_end, int cb, float eps) {
  const int t = blockIdx.x;
  const int* l = lf + (size_t)t * lfs;
  const int* in = inf + (size_t)t * 11;
  const bool lsem = is_semantic(l[0], sem_begin, sem_end);
  const int code = min(max(l[code_col], 0), cb - 1);
  const bool shared = with_shared && is_semantic(in[0], sem_begin, sem_end);
  __shared__ float ev[2048];
  float se = 0.f, sh = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float e = lsem ? b2f(table[(size_t)code * D + i]) : 0.f;
    if (shared) {
      float s = 0.f;
      for (int c = 2; c < 10; ++c) s += b2f(cb_emb[(size_t)(c * cb + min(max(in[1 + c], 0), cb - 1)) * D + i]);
      e = rb(e + rb(s));
    }
    ev[i] = e;
    se += e * e;
    const float hv = b2f(h[(size_t)t * D + i]);
    sh += hv * hv;
  }
  const float ie = rsqrtf(block_sum(se) / D + eps);
  const float ih = rsqrtf(block_sum(sh) / D + eps);
  bf16* o = out + (size_t)t * 2 * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    o[i] = f2b(ev[i] * ie * (1.f + b2f(enorm_w[i])));
    o[D + i] = f2b(b2f(h[(size_t)t * D + i]) * ih * (1.f + b2f(hnorm_w[i])));
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Sampling (sampling_kernels.py's semantics): scores = logits / max(temperature, 1e-5); keep rank 0 and the ranks
// below top_k whose inclusive cumulative probability is <= top_p; the kept candidate with the largest
// weight / Exp(1) noise wins, the noise from Triton's tl.philox(seed ^ (step >> 32 << 32), index, row, stream,
// step & 0xFFFFFFFF) first word: -log(((bits >> 9) + 0.5) / 2^23).

namespace {

__device__ __forceinline__ uint32_t umulhi32(uint32_t a, uint32_t b) { return __umulhi(a, b); }

__device__ uint32_t philox_first(uint64_t seed, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
  uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFFull), k1 = (uint32_t)(seed >> 32);
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    const uint32_t _c0 = c0, _c2 = c2;
    c0 = umulhi32(0xCD9E8D57u, _c2) ^ c1 ^ k0;
    c2 = umulhi32(0xD2511F53u, _c0) ^ c3 ^ k1;
    c1 = 0xCD9E8D57u * _c2;
    c3 = 0xD2511F53u * _c0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c0;
}

__device__ __forceinline__ float exp_noise(uint64_t seed, uint64_t step, int row, int stream, int index) {
  const uint64_t key = seed ^ ((step >> 32) << 32);
  const uint32_t bits = philox_first(key, (uint32_t)index, (uint32_t)row, (uint32_t)stream, (uint32_t)(step & 0xFFFFFFFFull));
  const float u = ((float)(bits >> 9) + 0.5f) * (1.0f / 8388608.0f);
  return -logf(u);
}

// Order-preserving key of a float score with the index (lower index wins ties), 0 for no candidate.
__device__ __forceinline__ unsigned long long score_key(float s, int idx) {
  uint32_t b = __float_as_uint(s == 0.f ? 0.f : s);
  b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((unsigned long long)b << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t)idx);
}
__device__ __forceinline__ float key_score(unsigned long long k) {
  uint32_t b = (uint32_t)(k >> 32);
  b = (b & 0x80000000u) ? (b & 0x7FFFFFFFu) : ~b;
  return __uint_as_float(b);
}
__device__ __forceinline__ int key_index(unsigned long long k) { return (int)(0xFFFFFFFFu - (uint32_t)k); }

// Bitonic sort of keys[0..N) descending, N a power of two, blockDim.x threads.
template <int N>
__device__ void sort_desc(unsigned long long* keys) {
  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const unsigned long long a = keys[i], b = keys[ixj];
          const bool desc = (i & k) == 0;
          if (desc ? (a < b) : (a > b)) {
            keys[i] = b;
            keys[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

// One draw from keys sorted descending by raw logit (count valid). Returns the candidate index (block-uniform).
template <int N>
__device__ int draw_sorted(const unsigned long long* keys, int count, float temperature, int top_k, float top_p,
                           uint64_t seed, uint64_t step, int row, int stream, float* scan) {
  const float t = fmaxf(temperature, 1e-5f);
  const float mx = key_score(keys[0]) / t;
  // weights over the valid candidates, their sum, and the inclusive prefix sum in sorted order (per-thread chunks)
  constexpr int PER = N / 1024 > 0 ? N / 1024 : 1;
  const int base = threadIdx.x * PER;
  float local[PER];
  float run = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = base + j;
    const float wv = (i < count && i < N) ? expf(key_score(keys[i]) / t - mx) : 0.f;
    run += wv;
    local[j] = run;
  }
  // exclusive scan of the per-thread totals
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  float incl = run;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float n = __shfl_up_sync(FULL, incl, o);
    if (lane >= o) incl += n;
  }
  __shared__ float wsum[32];
  __shared__ float total;
  __syncthreads();
  if (lane == 31) wsum[w] = incl;
  __syncthreads();
  if (w == 0) {
    float x = lane < (int)(blockDim.x >> 5) ? wsum[lane] : 0.f;
    float xi = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float n = __shfl_up_sync(FULL, xi, o);
      if (lane >= o) xi += n;
    }
    wsum[lane] = xi - x;  // exclusive
    if (lane == 31) total = xi;
  }
  __syncthreads();
  const float before = wsum[w] + (incl - run);
  const float sum = total;
  // race over the kept candidates
  float best = -1.f;
  int best_rank = 0x7FFFFFFF;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = base + j;
    if (i >= count || i >= N) break;
    const float cum = (before + local[j]) / sum;
    const bool kept = i == 0 || (i < top_k && cum <= top_p);
    if (!kept) continue;
    const float wv = expf(key_score(keys[i]) / t - mx);
    const float race = wv / exp_noise(seed, step, row, stream, key_index(keys[i]));
    if (race > best) {
      best = race;
      best_rank = i;
    }
  }
  // block argmax (ties to the lower rank)
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(FULL, best, o);
    const int orank = __shfl_xor_sync(FULL, best_rank, o);
    if (ob > best || (ob == best && orank < best_rank)) {
      best = ob;
      best_rank = orank;
    }
  }
  __shared__ float bb[32];
  __shared__ int br[32];
  __shared__ int chosen;
  if (lane == 0) {
    bb[w] = best;
    br[w] = best_rank;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float b = bb[0];
    int r = br[0];
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
      if (bb[k] > b || (bb[k] == b && br[k] < r)) {
        b = bb[k];
        r = br[k];
      }
    chosen = key_index(keys[r]);
  }
  __syncthreads();
  (void)scan;
  return chosen;
}

}  // namespace

struct D3Sampling {
  float temperature;
  int top_k;
  float top_p;
  int pad;
  unsigned long long seed;
};

// The main token and code 0 of a frame: logits [count] fp32 over the candidates cand_ids (sorted token ids: eos, then
// the semantic band); first draw at the request's settings (stream 0), RAS draw at (1.0, 30, 0.95) (stream 1); a
// semantic first draw found in the window of the last 16 main tokens takes the RAS draw. frame[0] = main token,
// frame[1] = clamp(main - sem_begin, 0, cb - 1). step = *step_ptr. 1024 threads, 64 KiB dynamic shared memory.
extern "C" __global__ void __launch_bounds__(1024) d3_sample_main(const float* __restrict__ logits, int count,
                                                                  const int* __restrict__ cand_ids,
                                                                  const int* __restrict__ window, D3Sampling sp,
                                                                  const unsigned long long* __restrict__ step_ptr,
                                                                  int* __restrict__ frame, int sem_begin, int cb) {
  constexpr int N = 8192;
  extern __shared__ unsigned long long keys[];
  for (int i = threadIdx.x; i < N; i += blockDim.x) keys[i] = i < count ? score_key(logits[i], i) : 0ull;
  __syncthreads();
  sort_desc<N>(keys);
  const uint64_t step = *step_ptr;
  const int first = draw_sorted<N>(keys, count, sp.temperature, sp.top_k, sp.top_p, sp.seed, step, 0, 0, nullptr);
  const int ras = draw_sorted<N>(keys, count, 1.0f, 30, 0.95f, sp.seed, step, 0, 1, nullptr);
  if (threadIdx.x == 0) {
    const int ft = cand_ids[first], rt = cand_ids[ras];
    bool repeat = false;
    for (int i = 0; i < 16; ++i) repeat |= window[i] == ft;
    const bool sem = ft >= sem_begin && ft < sem_begin + cb;
    const int main = (repeat && sem) ? rt : ft;
    frame[0] = main;
    frame[1] = min(max(main - sem_begin, 0), cb - 1);
  }
}

// A codebook head's code: logits [4096] fp32, the request's settings, stream 2 + head; *out = code.
extern "C" __global__ void __launch_bounds__(1024) d3_sample_code(const float* __restrict__ logits, int count,
                                                                  D3Sampling sp,
                                                                  const unsigned long long* __restrict__ step_ptr,
                                                                  int stream, int* __restrict__ out) {
  constexpr int N = 4096;
  extern __shared__ unsigned long long keys[];
  for (int i = threadIdx.x; i < N; i += blockDim.x) keys[i] = i < count ? score_key(logits[i], i) : 0ull;
  __syncthreads();
  sort_desc<N>(keys);
  const int c = draw_sorted<N>(keys, count, sp.temperature, sp.top_k, sp.top_p, sp.seed, *step_ptr, 0, stream, nullptr);
  if (threadIdx.x == 0) *out = c;
}

// Greedy picks (argmax, ties to the lower index): the main token over the candidates, or a code.
extern "C" __global__ void d3_argmax_main(const float* __restrict__ logits, int count, const int* __restrict__ cand_ids,
                                          int* __restrict__ frame, int sem_begin, int cb) {
  float bv = -INFINITY;
  int bi = 0x7FFFFFFF;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float v = logits[i];
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  __shared__ float sv[32];
  __shared__ int si[32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(FULL, bv, o);
    const int oi = __shfl_xor_sync(FULL, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    sv[threadIdx.x >> 5] = bv;
    si[threadIdx.x >> 5] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < (int)(blockDim.x >> 5); ++k)
      if (sv[k] > bv || (sv[k] == bv && si[k] < bi)) {
        bv = sv[k];
        bi = si[k];
      }
    if (cand_ids) {
      const int main = cand_ids[bi];
      frame[0] = main;
      frame[1] = min(max(main - sem_begin, 0), cb - 1);
    } else {
      frame[0] = bi;
    }
  }
}

// After a frame: frames_out[step0 + i] = frame (11 ints, i = *count), the RAS window shifted by the main token,
// the input frame for the next backbone row = frame, *pos += 1, *step += 1, *count += 1.
extern "C" __global__ void d3_finish_frame(const int* __restrict__ frame, int* __restrict__ frames_out,
                                           int* __restrict__ next_input, int* __restrict__ window,
                                           int* __restrict__ pos, unsigned long long* __restrict__ step,
                                           int* __restrict__ count) {
  if (threadIdx.x != 0) return;
  const int n = *count;
  for (int i = 0; i < 11; ++i) {
    frames_out[n * 11 + i] = frame[i];
    next_input[i] = frame[i];
  }
  for (int i = 0; i < 15; ++i) window[i] = window[i + 1];
  window[15] = frame[0];
  *pos += 1;
  *step += 1;
  *count = n + 1;
}

// Copy n ints (small device-side moves inside a graph).
extern "C" __global__ void d3_copy_i32(const int* __restrict__ src, int* __restrict__ dst, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
