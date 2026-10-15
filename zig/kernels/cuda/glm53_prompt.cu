// The torch steps of GLM-5.3's prompt chunks (fused.attention_core's wide path, gather's ring branch,
// experts_part's prompt branch) as plain kernels: each is data movement or one IEEE operation an element, so any
// implementation has torch's bits.
//   glm53_bhx_to_hbx  out[r, h, :] = in[h, r, :] (bf16; Tensor.copy_ of the bmm result into the [R, H, X] view)
//   glm53_bf16_f32    out[i] = float(in[i]) (Tensor.copy_ bf16 -> fp32: exact)
//   glm53_add_f32     a[i] += b[i] (Tensor.add_, fp32: one rounding)
//   glm53_hadamard    the 128 x 128 +-1 Hadamard matrix x3prefill.Workspace.hadamard builds (bf16)
#include <cstdint>
#include <cuda_bf16.h>

extern "C" __global__ void glm53_bhx_to_hbx(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int H, int R,
                                            int X) {
    const int r = blockIdx.x, h = blockIdx.y;
    const uint16_t* src = in + ((size_t)h * R + r) * X;
    uint16_t* dst = out + ((size_t)r * H + h) * X;
    for (int i = threadIdx.x; i < X; i += blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void glm53_bf16_f32(const __nv_bfloat16* __restrict__ in, float* __restrict__ out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

extern "C" __global__ void glm53_add_f32(float* __restrict__ a, const float* __restrict__ b, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = __fadd_rn(a[i], b[i]);
}

extern "C" __global__ void glm53_hadamard(__nv_bfloat16* __restrict__ out) {
    const int i = blockIdx.x, j = threadIdx.x;
    out[i * 128 + j] = __float2bfloat16_rn((__popc(i & j) & 1) ? -1.0f : 1.0f);
}

// ---- the prompt GEMM through cuBLASLt (TF_GLM53_PROMPT_GEMM=lt16 / lt8, glm53-prefill) ---------------------------
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// x3prefill's _gemm computes OUT[m, blk] = ((xh[m] @ W_q[:, blk]) @ H) * SCALE * svh[blk] for every 128-column block.
// Folding the right-hand side into the weight once gives one plain product: OUT = xh @ W_eff with
//   W_eff[k, n] = SCALE * svh[n] * sum_j W_q[k, blk + j] H[j, n - blk]      (H the Sylvester +-1 matrix, natural order)
// written transposed, out[n * K + k] (fp16), the layout cuBLASLt's "TN" product reads. A block: 128 columns x 32 k
// rows; thread t keeps column t of the 32 rows, the in-place FWHT over the 128 columns runs through shared memory.
extern "C" __global__ void __launch_bounds__(128) glm53_fold_w(const __half* __restrict__ wq, const __half* __restrict__ svh,
                                                               float scale, __half* __restrict__ out, int K, int N) {
    __shared__ float s[32][129];
    const int t = threadIdx.x, n0 = blockIdx.x * 128, k0 = blockIdx.y * 32;
    float v[32];
#pragma unroll
    for (int r = 0; r < 32; ++r) v[r] = __half2float(wq[(size_t)(k0 + r) * N + n0 + t]);
#pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
#pragma unroll
        for (int r = 0; r < 32; ++r) s[r][t] = v[r];
        __syncthreads();
        const bool lo = !(t & h);
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const float o = s[r][t ^ h];
            v[r] = lo ? v[r] + o : o - v[r];
        }
        __syncthreads();
    }
    const float m = scale * __half2float(svh[n0 + t]);
    __half* dst = out + (size_t)(n0 + t) * K + k0;
#pragma unroll
    for (int r = 0; r < 32; r += 2) *reinterpret_cast<__half2*>(dst + r) = __floats2half2_rn(v[r] * m, v[r + 1] * m);
}

// Rows of fp16 [rows, C] to e4m3 with one fp32 scale a row: scale = amax / 448 (1 for an all-zero row), out = v / scale
// rounded to nearest, saturating. One block a row. Serves both W_eff's rows (an output channel each) and the rotated
// input's rows (a token each), so cuBLASLt's outer-vector scales give OUT[m, n] = sa[m] sb[n] sum_k qa[m, k] qb[n, k].
extern "C" __global__ void __launch_bounds__(256) glm53_rows_e4m3(const __half* __restrict__ in, unsigned char* __restrict__ out,
                                                                  float* __restrict__ scale, int C) {
    __shared__ float red[8];
    const size_t row = blockIdx.x;
    const __half* x = in + row * C;
    float a = 0.f;
    for (int i = threadIdx.x; i < C; i += 256) a = fmaxf(a, fabsf(__half2float(x[i])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    a = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) a = fmaxf(a, red[w]);
    const float sc = a > 0.f ? a / 448.f : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) scale[row] = sc;
    unsigned char* y = out + row * C;
    for (int i = threadIdx.x; i < C; i += 256)
        y[i] = (unsigned char)__nv_cvt_float_to_fp8(__half2float(x[i]) * inv, __NV_SATFINITE, __NV_E4M3);
}

// MXFP8 rows: fp16 [rows, C] -> e4m3 [rows, C] with one power-of-two scale (UE8M0) per 32 values along C, the scale
// factors in cuBLASLt's VEC32_UE8M0 layout (128 x 4 tiles of 512 bytes, row-major over tiles; inside a tile byte
// (r % 32) * 16 + (r / 32 % 4) * 4 + c % 4). The scale is the smallest power of two with amax / scale <= 448, so a
// block's values keep their relative precision; an all-zero block takes 2^0. One thread a 32-value block.
__device__ __forceinline__ size_t mx_sf_off(int r, int c, int cblocks) {
    const int ct = (cblocks + 3) / 4;
    return ((size_t)(r / 128) * ct + c / 4) * 512 + (r % 32) * 16 + (r / 32 % 4) * 4 + (c % 4);
}
extern "C" __global__ void glm53_rows_mx8(const __half* __restrict__ in, unsigned char* __restrict__ out,
                                          unsigned char* __restrict__ sf, int rows, int C) {
    const int cblocks = C / 32;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cblocks) return;
    const int r = (int)(i / cblocks), c = (int)(i % cblocks);
    const __half* x = in + (size_t)r * C + c * 32;
    float v[32], a = 0.f;
#pragma unroll
    for (int j = 0; j < 32; j += 2) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(x + j));
        v[j] = f.x; v[j + 1] = f.y; a = fmaxf(a, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    int e = 0;                                   // scale 2^e
    if (a > 0.f) { e = (int)ceilf(log2f(a / 448.f)); if (ldexpf(448.f, e) < a) ++e; }
    e = max(-127, min(127, e));
    const float inv = ldexpf(1.f, -e);
    uint32_t w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t p = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) p |= (uint32_t)__nv_cvt_float_to_fp8(v[4 * j + b] * inv, __NV_SATFINITE, __NV_E4M3) << (8 * b);
        w[j] = p;
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (size_t)r * C + c * 32);
    dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
    dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    sf[mx_sf_off(r, c, cblocks)] = (unsigned char)(e + 127);
}
