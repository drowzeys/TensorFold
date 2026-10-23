// Codec kernel bodies for the FP16 decode and BF16 encode, each rounded to T where the reference holds a T tensor.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

namespace drama_codec {

__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(float v) { return v; }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

// The FP32 value a T tensor would hold.
template <typename T> __device__ __forceinline__ float round_to(float v) { return to_f(from_f<T>(v)); }

constexpr int kHeadDim = 64;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* part) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, n = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  return warp_sum(l < n ? part[l] : 0.f);
}

__device__ __forceinline__ float block_max(float v, float* part) {
  v = warp_max(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, n = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  return warp_max(l < n ? part[l] : -__int_as_float(0x7f800000));
}

// RMSNorm: Out(round_to<In>(x * rsqrt(mean(x*x) + eps)) * w), FP32 statistics, type_as(x) before w. Grid rows; 256.
template <typename In, typename W, typename Out>
__device__ __forceinline__ void rms_norm(const In* __restrict__ x, const W* __restrict__ w, Out* __restrict__ y, int width,
                                         float eps) {
  __shared__ float part[32];
  const In* xr = x + (size_t)blockIdx.x * width;
  Out* yr = y + (size_t)blockIdx.x * width;
  float s = 0.f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    const float v = to_f(xr[c]);
    s += __fmul_rn(v, v);
  }
  s = block_sum(s, part);
  const float r = rsqrtf(__fadd_rn(s / (float)width, eps));
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    yr[c] = from_f<Out>(__fmul_rn(round_to<In>(__fmul_rn(to_f(xr[c]), r)), to_f(w[c])));
}

// apply_rotary_emb on q, k of [q | k | v] rows: x0*c - x1*s, x1*c + x0*s in FP32, table [rows][32][2]. Grid rows; 256.
template <typename T>
__device__ __forceinline__ void rope(T* __restrict__ qkv, const T* __restrict__ table, int width, int ld) {
  T* row = qkv + (size_t)blockIdx.x * ld;
  const T* tab = table + (size_t)blockIdx.x * kHeadDim;
  const int pairs = width / 2;
  for (int p = threadIdx.x; p < 2 * pairs; p += blockDim.x) {
    const int which = p / pairs, pp = p % pairs, i = pp % (kHeadDim / 2);
    T* e = row + which * width + pp * 2;
    const float x0 = to_f(e[0]), x1 = to_f(e[1]);
    const float c = to_f(tab[2 * i]), s = to_f(tab[2 * i + 1]);
    e[0] = from_f<T>(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
    e[1] = from_f<T>(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
  }
}

// Causal window attention as Torch's efficient path: 64-key tiles, exp2, T probabilities, FP32 sums. Grid (q, h); 128.
template <typename T>
__device__ __forceinline__ void window_attention(const T* __restrict__ qkv, T* __restrict__ out, int t0, int heads,
                                                 int window, float scale) {
  constexpr int tile = 64;
  __shared__ float q[kHeadDim], p[tile], part[32];
  const int t = t0 + blockIdx.x, h = blockIdx.y, j = threadIdx.x;
  const size_t ld = (size_t)3 * heads * kHeadDim;
  if (j < kHeadDim) q[j] = to_f(qkv[(size_t)t * ld + h * kHeadDim + j]);
  __syncthreads();
  const int first = max(0, t - window + 1);
  float maximum = -__int_as_float(0x7f800000), denominator = 0.f, numerator = 0.f;
  for (int start = first / tile * tile; start <= t; start += tile) {
    const int key = start + j;
    const bool valid = j < tile && key >= first && key <= t;
    float score = -__int_as_float(0x7f800000);
    if (valid) {
      const T* k = qkv + (size_t)key * ld + (size_t)heads * kHeadDim + h * kHeadDim;
      float dot = 0.f;
#pragma unroll 16
      for (int d = 0; d < kHeadDim; ++d) dot = fmaf(q[d], to_f(k[d]), dot);
      score = (dot * scale) * 1.4426950408889634074f;
    }
    const float next_maximum = fmaxf(maximum, block_max(score, part));
    const float rescale = exp2f(maximum - next_maximum);
    const float probability = valid ? exp2f(score - next_maximum) : 0.f;
    if (j < tile) p[j] = round_to<T>(probability);
    const float mass = block_sum(probability, part);  // also publishes p[]
    denominator = denominator * rescale + mass;
    if (j < kHeadDim) {
      numerator *= rescale;
      for (int i = 0; i < tile && start + i <= t; ++i) {
        if (start + i < first) continue;
        const T v = qkv[(size_t)(start + i) * ld + (size_t)2 * heads * kHeadDim + h * kHeadDim + j];
        numerator = fmaf(p[i], to_f(v), numerator);
      }
    }
    maximum = next_maximum;
    __syncthreads();
  }
  if (j < kHeadDim) out[(size_t)t * heads * kHeadDim + h * kHeadDim + j] = from_f<T>(numerator / denominator);
}

// F.silu(w1(x)) * w3(x) over [rows][2 * width] (w1 | w3): silu rounded, then the product. Grid ceil(rows*width/256).
template <typename T>
__device__ __forceinline__ void silu_mul(const T* __restrict__ gu, T* __restrict__ out, long long rows, int width) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  const long long r = i / width, c = i % width;
  const float a = to_f(gu[r * 2 * width + c]);
  const float b = to_f(gu[r * 2 * width + width + c]);
  out[i] = from_f<T>(round_to<T>(a / (1.0f + expf(-a))) * b);
}

// nn.LayerNorm (eps 1e-6): FP32 mean and biased variance, Out(gamma * (rstd * (x - mean)) + beta). Grid rows; 256.
template <typename In, typename W, typename Out>
__device__ __forceinline__ void layer_norm(const In* __restrict__ x, const W* __restrict__ w, const W* __restrict__ b,
                                           Out* __restrict__ y, int width, float eps) {
  __shared__ float part[32];
  const In* xr = x + (size_t)blockIdx.x * width;
  Out* yr = y + (size_t)blockIdx.x * width;
  float s = 0.f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) s += to_f(xr[c]);
  const float mean = block_sum(s, part) / (float)width;
  float v = 0.f;
  for (int c = threadIdx.x; c < width; c += blockDim.x) {
    const float d = to_f(xr[c]) - mean;
    v = fmaf(d, d, v);
  }
  const float rstd = rsqrtf(block_sum(v, part) / (float)width + eps);
  for (int c = threadIdx.x; c < width; c += blockDim.x)
    yr[c] = from_f<Out>(fmaf(to_f(w[c]), rstd * (to_f(xr[c]) - mean), to_f(b[c])));
}

// ConvNeXt's causal depthwise conv 7, then the bias add_: rows t0.. of [..][width], w [width][7]. Grid over elements.
template <typename T>
__device__ __forceinline__ void dwconv7(const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ b,
                                        T* __restrict__ y, long long t0, long long rows, int width) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * width) return;
  const long long t = t0 + i / width;
  const int c = (int)(i % width);
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 7; ++k) {
    const long long r = t - 6 + k;
    if (r >= 0) acc = fmaf(to_f(x[r * width + c]), to_f(w[c * 7 + k]), acc);
  }
  y[t * width + c] = from_f<T>(round_to<T>(acc) + to_f(b[c]));
}

// Implicit GEMM, 128x128 tiles: acc[m][n] = sum over tap, ci of x[m - (taps-1-tap) * d][ci] * w[n][tap*cin + ci].
constexpr int BM = 128, BN = 128, BK = 32, LDS = BK + 8;

// stride: the encode's; its Blocked convs read row m * stride + (tap - taps + 1) * dilation + stride - 1.
struct GemmShape {
  long long m, m_begin;
  int n, cin, taps, dilation, ldx, stride;
};

// Blocked: K over 32-channel blocks, then taps, then channels, as cuDNN sums a BF16 conv (weights laid out alike).
template <typename T, bool Blocked = false, typename Store>
__device__ __forceinline__ void conv_gemm(const T* __restrict__ x, const T* __restrict__ w, const GemmShape& g,
                                          Store store) {
  using namespace nvcuda;
  __shared__ __align__(128) T as[BM * LDS];
  __shared__ __align__(128) T bs[BN * LDS];
  __shared__ __align__(128) float cs[8][16 * 16];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 2, wn = warp & 3;
  const long long m0 = (g.m_begin / BM + blockIdx.x) * BM;
  const int n0 = blockIdx.y * BN;
  const int ktot = g.taps * g.cin;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  uint4 ra[2], rb[2];
  auto fetch = [&](int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 2, c = (v & 3) * 8;
      const int kk = k0 + c;
      uint4 a = make_uint4(0u, 0u, 0u, 0u), b = make_uint4(0u, 0u, 0u, 0u);
      const long long m = m0 + r;
      if (m >= g.m_begin && m < g.m && kk < ktot) {
        int tap, ci;
        long long rin;
        if constexpr (Blocked) {
          const int block = kk / (32 * g.taps), within = kk % (32 * g.taps);
          tap = within / 32;
          ci = block * 32 + within % 32;
          rin = m * g.stride + (long long)(tap - g.taps + 1) * g.dilation + g.stride - 1;
        } else {
          tap = kk / g.cin;
          ci = kk - tap * g.cin;
          rin = m - (long long)(g.taps - 1 - tap) * g.dilation;
        }
        if (rin >= 0) a = *reinterpret_cast<const uint4*>(x + rin * g.ldx + ci);
      }
      const int n = n0 + r;
      if (n < g.n && kk < ktot) b = *reinterpret_cast<const uint4*>(w + (size_t)n * ktot + kk);
      ra[i] = a;
      rb[i] = b;
    }
  };

  fetch(0);
  for (int k0 = 0; k0 < ktot; k0 += BK) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int v = tid + i * 256;
      const int r = v >> 2, c = (v & 3) * 8;
      *reinterpret_cast<uint4*>(&as[r * LDS + c]) = ra[i];
      *reinterpret_cast<uint4*>(&bs[r * LDS + c]) = rb[i];
    }
    __syncthreads();
    if (k0 + BK < ktot) fetch(k0 + BK);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> af[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> bf[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(af[i], as + (wm * 64 + i * 16) * LDS + kk, LDS);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(bf[j], bs + (wn * 32 + j * 16) * LDS + kk, LDS);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
  }

  float* c = cs[warp];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(c, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      for (int e = lane; e < 256; e += 32) {
        const long long m = m0 + wm * 64 + i * 16 + (e >> 4);
        const int n = n0 + wn * 32 + j * 16 + (e & 15);
        if (m >= g.m_begin && m < g.m && n < g.n) store(m, n, c[e]);
      }
      __syncwarp();
    }
  }
}

}  // namespace drama_codec
