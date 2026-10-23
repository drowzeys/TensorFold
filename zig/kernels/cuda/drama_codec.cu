// Drama 3 codec decode: the Fish modded DAC's quantizer decode (DownsampleResidualVectorQuantize.decode:
// RVQ codes, the causal window-limited transformer, the ConvNeXt upsampler) and its causal decoder, as
// sglang-omni's FP16 path runs them (ChannelsLastDecoder over a weight-folded, half-cast codec).
//
// Storage is FP16 everywhere the reference holds an FP16 tensor; every reduction runs in FP32 and every
// value is rounded to FP16 exactly where the reference materialises an FP16 tensor:
//   cuDNN convolutions (bias added by a separate add_):  y = h(h(acc) + b)
//   cuBLASLt linears (bias in the epilogue):             y = h(acc + b)
//   elementwise ops: computed in FP32 from FP16 inputs, one rounding each, in the reference's op order.
// h() is round-to-nearest-even FP16. Activations are channels-last: [rows][channels].

#include "drama_codec.cuh"

using namespace drama_codec;

namespace {

__device__ __forceinline__ float h2f(__half v) { return __half2float(v); }
__device__ __forceinline__ __half f2h(float v) { return __float2half_rn(v); }
__device__ __forceinline__ float rh(float v) { return round_to<__half>(v); }

constexpr int kLatent = 1024;     // DownsampleResidualVectorQuantize input_dim
constexpr int kCodebookDim = 8;   // codebook_dim
constexpr int kBooks = 10;        // 1 semantic + 9 residual
constexpr int kSemanticSize = 4096;
constexpr int kResidualSize = 1024;

}  // namespace

// rvq.py DownsampleResidualVectorQuantize.decode, before post_module:
//   codes clamped (semantic to 4095, residual to 1023); each book: embedding row (8) -> out_proj
//   (weight-folded 1x1 conv 8 -> 1024, cuDNN + bias add_); residual books summed in order from
//   0.0 (dac ResidualVectorQuantize.from_codes); z = semantic + residual.
// codes [frames][10] u32; semantic_book [4096][8]; residual_books [9][1024][8]; out_w [10][1024][8];
// out_b [10][1024]; z [frames][1024]. Grid: frames; block: 256.
extern "C" __global__ void __launch_bounds__(256) drama_codec_rvq_latent(
    const uint32_t* __restrict__ codes, const __half* __restrict__ semantic_book,
    const __half* __restrict__ residual_books, const __half* __restrict__ out_w,
    const __half* __restrict__ out_b, __half* __restrict__ z) {
  const size_t t = blockIdx.x;
  __shared__ float e[kBooks][kCodebookDim];
  if (threadIdx.x < kBooks * kCodebookDim) {
    const int book = threadIdx.x / kCodebookDim, i = threadIdx.x % kCodebookDim;
    uint32_t code = codes[t * kBooks + book];
    const __half* row;
    if (book == 0) {
      code = min(code, (uint32_t)(kSemanticSize - 1));
      row = semantic_book + (size_t)code * kCodebookDim;
    } else {
      code = min(code, (uint32_t)(kResidualSize - 1));
      row = residual_books + ((size_t)(book - 1) * kResidualSize + code) * kCodebookDim;
    }
    e[book][i] = h2f(row[i]);
  }
  __syncthreads();
  for (int c = threadIdx.x; c < kLatent; c += blockDim.x) {
    float semantic = 0.f, residual = 0.f;
    for (int book = 0; book < kBooks; ++book) {
      const __half* w = out_w + ((size_t)book * kLatent + c) * kCodebookDim;
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < kCodebookDim; ++i) acc = fmaf(e[book][i], h2f(w[i]), acc);
      const float q = rh(rh(acc) + h2f(out_b[book * kLatent + c]));
      if (book == 0) semantic = q;
      else if (book == 1) residual = 0.f + q;
      else residual = rh(residual + q);
    }
    z[t * kLatent + c] = f2h(semantic + residual);
  }
}

extern "C" __global__ void __launch_bounds__(256) drama_codec_rms_norm(
    const __half* __restrict__ x, const __half* __restrict__ w, __half* __restrict__ y, int width,
    float eps) {
  rms_norm(x, w, y, width, eps);
}

extern "C" __global__ void __launch_bounds__(256) drama_codec_rope(
    __half* __restrict__ qkv, const __half* __restrict__ table, int width, int ld) {
  rope(qkv, table, width, ld);
}

extern "C" __global__ void __launch_bounds__(128) drama_codec_window_attention(
    const __half* __restrict__ qkv, __half* __restrict__ out, int t0, int heads, int window, float scale) {
  window_attention(qkv, out, t0, heads, window, scale);
}

extern "C" __global__ void __launch_bounds__(256) drama_codec_silu_mul(
    const __half* __restrict__ gu, __half* __restrict__ out, long long rows, int width) {
  silu_mul(gu, out, rows, width);
}

extern "C" __global__ void __launch_bounds__(256) drama_codec_layer_norm(
    const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
    __half* __restrict__ y, int width, float eps) {
  layer_norm(x, w, b, y, width, eps);
}

extern "C" __global__ void __launch_bounds__(256) drama_codec_dwconv7(
    const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
    __half* __restrict__ y, long long t0, long long rows, int width) {
  dwconv7(x, w, b, y, t0, rows, width);
}

// dac Snake1d as channels_last_decoder.py's Triton kernel computes it: alpha and 1 / (alpha + 1e-9)
// in FP32, s = sin(alpha * x), y = x + inv * s * s with the final multiply-add fused. In place allowed.
// Grid: ceil(n / 256).
extern "C" __global__ void __launch_bounds__(256) drama_codec_snake(
    const __half* x, const float* __restrict__ alpha, const float* __restrict__ inv_alpha, __half* y,
    long long n, int width) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = (int)(i % width);
  const float v = h2f(x[i]);
  const float s = sinf(alpha[c] * v);
  y[i] = f2h(fmaf(inv_alpha[c] * s, s, v));
}

// The last decoder conv (96 -> 1, kernel 7, causal) and nn.Tanh, written as FP32 PCM:
// y = float(h(tanh(h(h(acc) + b)))). Rows t0..t0+rows-1; x [..][width]; w [7][width]. Grid: ceil(rows / 256).
extern "C" __global__ void __launch_bounds__(256) drama_codec_conv_out_tanh(
    const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
    float* __restrict__ y, long long t0, long long rows, int width) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows) return;
  const long long t = t0 + i;
  float acc = 0.f;
  for (int k = 0; k < 7; ++k) {
    const long long r = t - 6 + k;
    if (r < 0) continue;
    const __half* xr = x + r * width;
    const __half* wk = w + k * width;
    for (int c = 0; c < width; ++c) acc = fmaf(h2f(xr[c]), h2f(wk[c]), acc);
  }
  const float v = rh(rh(acc) + h2f(b[0]));
  y[t] = rh(tanhf(v));
}

// The decode's GEMM (drama_codec.cuh conv_gemm): ConvTranspose1d(stride s, kernel 2s) is two taps (rows q-1, q)
// over n = j * cout + co, its output rows [q][j] being the output samples q * s + j.
// Epilogue, in the reference's order: bias (fused: h(acc + b); separate: h(h(acc) + b)), GELU
// (erf form), per-column scale (LayerScale / ConvNeXt gamma), residual add, each rounded to FP16.
struct GemmParams {
  const __half* x;
  const __half* w;
  const __half* bias;      // [period] or null
  const __half* scale;     // [period] or null
  const __half* residual;  // [m][ldr] or null (may alias y)
  __half* y;
  long long m, m_begin;
  int n, cin, taps, dilation, ldx, ldy, ldr, period, bias_mode, act;
};

namespace {
__device__ __forceinline__ float gemm_epilogue(const GemmParams& p, long long m, int n, float acc) {
  const int bn = n % p.period;
  float v;
  if (p.bias_mode == 1) {
    v = rh(acc + h2f(p.bias[bn]));
  } else {
    v = rh(acc);
    if (p.bias_mode == 2) v = rh(v + h2f(p.bias[bn]));
  }
  if (p.act == 1) v = rh(v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f)));
  if (p.scale != nullptr) v = rh(v * h2f(p.scale[bn]));
  if (p.residual != nullptr) v = rh(h2f(p.residual[m * p.ldr + n]) + v);
  return v;
}
}  // namespace

// Requires cin, ldx and ldy multiples of 8 and x, w 16-byte aligned.
extern "C" __global__ void __launch_bounds__(256) drama_codec_conv_gemm(const GemmParams p) {
  const GemmShape g = {p.m, p.m_begin, p.n, p.cin, p.taps, p.dilation, p.ldx};
  conv_gemm(p.x, p.w, g, [&](long long m, int n, float acc) { p.y[m * p.ldy + n] = f2h(gemm_epilogue(p, m, n, acc)); });
}
