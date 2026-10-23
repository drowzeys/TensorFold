// Drama 3 codec encode (DAC encoder, downsampler, pre_module, RVQ) as FishReferenceEncodeHook runs it in BF16 autocast.

// Convs and linears round to BF16 (b()); Snake, norms, gamma and residual sums stay FP32; activations are [rows][ch].

#include "drama_codec.cuh"

using namespace drama_codec;
using bf16 = __nv_bfloat16;

namespace {
constexpr int kLatent = 1024;
constexpr int kCodebookDim = 8;
constexpr int kBooks = 10;
constexpr int kSemanticSize = 4096;
constexpr int kResidualSize = 1024;
constexpr int kInChannels = 64;
// The VQ distance tile: 16 frames by 16 codes, the codebook dimension zero-padded from 8 to the MMA's 16.
constexpr int kTile = 16;
constexpr int kVqWarps = 4;

__device__ __forceinline__ float rb(float v) { return round_to<bf16>(v); }
}  // namespace

// encoder.block[0], conv 1 -> 64 kernel 7 over the clip cast to BF16: [rows] -> [rows][64] = b(b(acc) + bias).
extern "C" __global__ void __launch_bounds__(256) drama_encode_conv_in(const float* __restrict__ audio,
                                                                       const bf16* __restrict__ w,
                                                                       const bf16* __restrict__ b,
                                                                       bf16* __restrict__ y, long long rows) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * kInChannels) return;
  const long long t = i / kInChannels;
  const int c = (int)(i % kInChannels);
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 7; ++k) {
    const long long r = t - 6 + k;
    if (r >= 0) acc = fmaf(rb(audio[r]), to_f(w[c * 7 + k]), acc);
  }
  y[i] = from_f<bf16>(rb(acc) + to_f(b[c]));
}

// Snake1d in FP32 to BF16; `fused`: TorchScript's fused kernel (last multiply-add one FMA), else op by op. In place ok.
template <typename In>
__device__ __forceinline__ void snake(const In* x, const float* __restrict__ alpha, const float* __restrict__ inv_alpha,
                                      bf16* y, long long n, int width, int fused) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = (int)(i % width);
  const float v = to_f(x[i]);
  const float s = sinf(__fmul_rn(alpha[c], v));
  const float s2 = __fmul_rn(s, s);
  y[i] = from_f<bf16>(fused ? fmaf(inv_alpha[c], s2, v) : __fadd_rn(v, __fmul_rn(inv_alpha[c], s2)));
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_snake(const bf16* x, const float* alpha,
                                                                     const float* inv_alpha, bf16* y, long long n,
                                                                     int width, int fused) {
  snake(x, alpha, inv_alpha, y, n, width, fused);
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_snake_f32(const float* x, const float* alpha,
                                                                         const float* inv_alpha, bf16* y, long long n,
                                                                         int width, int fused) {
  snake(x, alpha, inv_alpha, y, n, width, fused);
}

// RMSNorm over BF16 or FP32 rows with its FP32 weight, into the BF16 a linear reads or the FP32 the next op keeps.
extern "C" __global__ void __launch_bounds__(256) drama_encode_rms_norm(const bf16* x, const float* w, bf16* y,
                                                                        int width, float eps) {
  rms_norm(x, w, y, width, eps);
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_rms_norm_to_f32(const bf16* x, const float* w,
                                                                               float* y, int width, float eps) {
  rms_norm(x, w, y, width, eps);
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_rms_norm_f32(const float* x, const float* w, bf16* y,
                                                                            int width, float eps) {
  rms_norm(x, w, y, width, eps);
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_rms_norm_f32_to_f32(const float* x, const float* w,
                                                                                   float* y, int width, float eps) {
  rms_norm(x, w, y, width, eps);
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_rope(bf16* qkv, const bf16* table, int width, int ld) {
  rope(qkv, table, width, ld);
}

extern "C" __global__ void __launch_bounds__(128) drama_encode_window_attention(const bf16* qkv, bf16* out, int t0,
                                                                                int heads, int window, float scale) {
  window_attention(qkv, out, t0, heads, window, scale);
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_silu_mul(const bf16* gu, bf16* out, long long rows,
                                                                        int width) {
  silu_mul(gu, out, rows, width);
}

// ConvNeXtBlock.norm: autocast runs layer_norm in FP32 with its FP32 weights; pwconv1 reads the result as BF16.
extern "C" __global__ void __launch_bounds__(256) drama_encode_layer_norm(const bf16* x, const float* w, const float* b,
                                                                          bf16* y, int width, float eps) {
  layer_norm(x, w, b, y, width, eps);
}

extern "C" __global__ void __launch_bounds__(256) drama_encode_dwconv7(const bf16* x, const bf16* w, const bf16* b,
                                                                       bf16* y, long long rows, int width) {
  dwconv7(x, w, b, y, 0, rows, width);
}

// Every encode conv and linear: K in cuDNN's order (32-channel blocks, taps, channels); conv row m * s - (k - s) + tap.
struct EncodeGemmParams {
  const bf16* x;
  const bf16* w;
  const bf16* bias;        // [n] or null
  const float* scale;      // [n] or null
  const void* residual;    // [m][ldr] or null (may alias y)
  void* y;
  long long m;
  int n, cin, taps, dilation, stride, ldx, ldy, ldr, bias_mode, act, flags;
};

enum : int { kRoundScaled = 1, kResidualF32 = 2, kOutF32 = 4 };

// Epilogue in the reference's order: bias, GELU, FP32 scale (rounded for LayerScale's mul_), FP32 residual sum.
extern "C" __global__ void __launch_bounds__(256) drama_encode_gemm(const EncodeGemmParams p) {
  const GemmShape g = {p.m, 0, p.n, p.cin, p.taps, p.dilation, p.ldx, p.stride};
  conv_gemm<bf16, true>(p.x, p.w, g, [&](long long m, int n, float acc) {
    float v;
    if (p.bias_mode == 1) {
      v = rb(acc + to_f(p.bias[n]));
    } else {
      v = rb(acc);
      if (p.bias_mode == 2) v = rb(v + to_f(p.bias[n]));
    }
    if (p.act == 1) v = rb(v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f)));
    if (p.scale != nullptr) {
      v = v * p.scale[n];
      if (p.flags & kRoundScaled) v = rb(v);
    }
    if (p.residual != nullptr) {
      const long long at = m * p.ldr + n;
      v = (p.flags & kResidualF32 ? static_cast<const float*>(p.residual)[at]
                                  : to_f(static_cast<const bf16*>(p.residual)[at])) + v;
    }
    const long long at = m * p.ldy + n;
    if (p.flags & kOutF32) static_cast<float*>(p.y)[at] = v;
    else static_cast<bf16*>(p.y)[at] = from_f<bf16>(v);
  });
}

// dac RVQ, 16 frames a block: per book the first argmin of (sum x^2 - 2 x . c) + sum c^2, then r -= out_proj(z_q).
extern "C" __global__ void __launch_bounds__(kVqWarps * 32) drama_encode_rvq(
    float* __restrict__ z, uint32_t* __restrict__ codes, const bf16* __restrict__ in_w, const bf16* __restrict__ in_b,
    const bf16* __restrict__ books, const float* __restrict__ books_sq, const float* __restrict__ books_raw,
    const bf16* __restrict__ out_w, const bf16* __restrict__ out_b, int frames) {
  using namespace nvcuda;
  __shared__ float ze[kTile][kCodebookDim], x_sq[kTile];
  __shared__ __align__(32) bf16 x[kTile * kTile];
  __shared__ __align__(32) float tile[kVqWarps][kTile * kTile];
  __shared__ float best_dist[kVqWarps * 32];
  __shared__ int best_code[kVqWarps * 32];
  __shared__ bf16 zq[kTile][kCodebookDim];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int f0 = blockIdx.x * kTile;
  const int rows = min(kTile, frames - f0);
  int book_start = 0;
  for (int book = 0; book < kBooks; ++book) {
    const int size = book == 0 ? kSemanticSize : kResidualSize;
    {  // in_proj: one thread per (frame, output)
      const int f = tid / kCodebookDim, j = tid % kCodebookDim;
      float v = 0.f;
      if (f < rows) {
        const float* r = z + (size_t)(f0 + f) * kLatent;
        const bf16* wr = in_w + ((size_t)book * kCodebookDim + j) * kLatent;
        float acc = 0.f;
        for (int c = 0; c < kLatent; ++c) acc = fmaf(rb(r[c]), to_f(wr[c]), acc);
        v = rb(rb(acc) + to_f(in_b[book * kCodebookDim + j]));
      }
      ze[f][j] = v;
    }
    __syncthreads();
    if (tid < kTile) {  // F.normalize and sum of squares over the strided [frames, 8] view
      float acc[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] = __fadd_rn(__fmul_rn(ze[tid][j], ze[tid][j]), __fmul_rn(ze[tid][j + 4], ze[tid][j + 4]));
      const float norm = fmaxf(sqrtf(__fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3])), 1e-12f);
      float xn[kCodebookDim];
#pragma unroll
      for (int j = 0; j < kCodebookDim; ++j) xn[j] = __fdiv_rn(ze[tid][j], norm);
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] = __fadd_rn(__fmul_rn(xn[j], xn[j]), __fmul_rn(xn[j + 4], xn[j + 4]));
      x_sq[tid] = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
      // 2 * x in BF16: the reference scales x before the matmul; doubling commutes with every rounding
#pragma unroll
      for (int j = 0; j < kTile; ++j) x[tid * kTile + j] = from_f<bf16>(j < kCodebookDim ? 2.f * xn[j] : 0.f);
    }
    __syncthreads();
    {  // distances to every code of the book, 16 x 16 per MMA; each lane keeps one frame's best over its codes
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, bf16, wmma::row_major> a;
      wmma::load_matrix_sync(a, x, kTile);
      const int f = lane % kTile, half = lane / kTile;
      float best = __int_as_float(0x7f800000);
      int arg = 0;
      for (int k0 = warp * kTile; k0 < size; k0 += kVqWarps * kTile) {
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, bf16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> d;
        wmma::fill_fragment(d, 0.f);
        wmma::load_matrix_sync(b, books + (size_t)(book_start + k0) * kTile, kTile);
        wmma::mma_sync(d, a, b, d);
        wmma::store_matrix_sync(tile[warp], d, kTile, wmma::mem_row_major);
        __syncwarp();
        for (int i = half * 8; i < half * 8 + 8; ++i) {
          const int code = k0 + i;
          const float dist = __fadd_rn(__fsub_rn(x_sq[f], rb(tile[warp][f * kTile + i])), books_sq[book_start + code]);
          if (dist < best) {
            best = dist;
            arg = code;
          }
        }
        __syncwarp();
      }
      best_dist[tid] = best;
      best_code[tid] = arg;
    }
    __syncthreads();
    {  // first argmin over the lanes holding each frame, then the straight-through z_q
      const int f = tid / kCodebookDim, j = tid % kCodebookDim;
      float best = __int_as_float(0x7f800000);
      int arg = 0;
      for (int w = 0; w < kVqWarps; ++w) {
        for (int half = 0; half < 2; ++half) {
          const int at = w * 32 + half * kTile + f;
          if (best_dist[at] < best || (best_dist[at] == best && best_code[at] < arg)) {
            best = best_dist[at];
            arg = best_code[at];
          }
        }
      }
      if (f < rows && j == 0) codes[(size_t)(f0 + f) * kBooks + book] = arg;
      const float e = ze[f][j];
      zq[f][j] = from_f<bf16>(__fadd_rn(e, __fsub_rn(books_raw[(size_t)(book_start + arg) * kCodebookDim + j], e)));
    }
    __syncthreads();
    for (int c = tid; c < kLatent; c += blockDim.x) {  // out_proj, subtracted from the FP32 residual
      const bf16* wr = out_w + ((size_t)book * kLatent + c) * kCodebookDim;
      const float bias = to_f(out_b[book * kLatent + c]);
      for (int f = 0; f < rows; ++f) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < kCodebookDim; ++j) acc = fmaf(to_f(zq[f][j]), to_f(wr[j]), acc);
        float* r = z + (size_t)(f0 + f) * kLatent + c;
        *r = __fsub_rn(*r, rb(rb(acc) + bias));
      }
    }
    __syncthreads();
    book_start += size;
  }
}
