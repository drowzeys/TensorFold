// GLM-5.3's MTP layer over prompt rows (Phase 3a speed), ours. The MTP layer only drafts - the target verifies every
// token - so its prompt rows need no bit-exact path (the Python engine runs cuda-exl3's split-K grouped GEMM there,
// not reproducible run to run). Its 8-bit routed experts are kept a second time as dense bf16 matrices with the EXL3
// rotations folded in (built once at load), and prompt rows run them as per-expert cuBLAS GEMMs:
//
//   glm53_mtp_had_rows / glm53_mtp_had_cols: load time. W_q [K, N] fp16 (unpack_kernel's output, the rotated domain)
//     -> W [K, N] bf16 with x @ W = EXL3 linear(x): W[k, n] = suh[k] * (H W_q H)[k, n] * svh[n] / 128, H the
//     block-diagonal 128 x 128 Sylvester Hadamard (rot_in's fwht128 on the input, the epilogue's on the output, each
//     scaled by 1 / sqrt(128)). Rows pass: fp32 t = W_q H along n; cols pass: H t along k, scaled, bf16 into a row
//     stride `ldo` (gate and up side by side in one [D, 2I] matrix).
//   glm53_mtp_gather: xs[p] = x[perm[p] / slots] (bf16 rows, 16-byte copies): the routed pairs grouped by expert.
//   glm53_mtp_swiglu: h [n, 2I] (gate | up) -> h[:, :I] = bf16(silu(gate) * up), fp32 math, in place.
//   glm53_mtp_combine: out[r] = sy[r] + sum over slots (in slot order) of wts[r, s] * y[inv[r, s]] (fp32).
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
// y = H x for the 128 values of a warp (lane holds 4 consecutive: index 4 lane + j), H Sylvester-ordered: decode.cuh's
// fwht128 (rot_in_kernel / the linear epilogue use it), copied so this file stands alone.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}
constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
}  // namespace

// t[k, n] = (W_q H)[k, n] along n in 128-blocks: one warp a (row, 128-block). 256 threads, grid ceil(K N / 128 / 8).
extern "C" __global__ void __launch_bounds__(256) glm53_mtp_had_rows(const __half* __restrict__ wq,
                                                                      float* __restrict__ t, int K, int N) {
    const long long v = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int nb = N >> 7;
    if (v >= (long long)K * nb) return;                  // a whole warp leaves together
    const size_t base = (size_t)(v / nb) * N + (size_t)(v % nb) * 128 + 4 * lane;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __half2float(wq[base + j]);
    fwht128(x, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) t[base + j] = x[j];
}

// out[k * ldo + n] = bf16((H t)[k, n] * suh[k] * svh[n] / 128) along k in 128-blocks: one warp a (128-block, column).
// 256 threads, grid ceil(K / 128 * N / 8).
extern "C" __global__ void __launch_bounds__(256) glm53_mtp_had_cols(const float* __restrict__ t,
                                                                      const __half* __restrict__ suh,
                                                                      const __half* __restrict__ svh,
                                                                      __nv_bfloat16* __restrict__ out, int K, int N,
                                                                      int ldo) {
    const long long v = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (v >= (long long)(K >> 7) * N) return;
    const int n = (int)(v % N);
    const int k0 = (int)(v / N) * 128 + 4 * lane;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = t[(size_t)(k0 + j) * N + n];
    fwht128(x, lane);
    const float sv = __half2float(svh[n]) * (HAD_SCALE * HAD_SCALE);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[(size_t)(k0 + j) * ldo + n] = __float2bfloat16_rn(x[j] * __half2float(suh[k0 + j]) * sv);
}

// xs[p, :] = x[perm[p] / slots, :] for p < gridDim.x (rows of D16 16-byte pieces; x rows x_stride16 pieces apart).
extern "C" __global__ void __launch_bounds__(256) glm53_mtp_gather(const uint4* __restrict__ x, int x_stride16,
                                                                    const int* __restrict__ perm, int slots,
                                                                    uint4* __restrict__ xs, int D16) {
    const int p = blockIdx.x;
    const uint4* src = x + (size_t)(perm[p] / slots) * x_stride16;
    uint4* dst = xs + (size_t)p * D16;
    for (int i = threadIdx.x; i < D16; i += blockDim.x) dst[i] = src[i];
}

// Row p = blockIdx.x of h [n, 2I]: h[p, i] = bf16(silu(h[p, i]) * h[p, I + i]) for i < I (fp32 math).
extern "C" __global__ void __launch_bounds__(256) glm53_mtp_swiglu(__nv_bfloat16* __restrict__ h, int I) {
    __nv_bfloat16* row = h + (size_t)blockIdx.x * 2 * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x) {
        const float g = __bfloat162float(row[i]);
        const float u = __bfloat162float(row[I + i]);
        row[i] = __float2bfloat16_rn(g / (1.0f + __expf(-g)) * u);
    }
}

// Row r = blockIdx.x: out[r, :] = sy[r, :] (when given) + sum_s wts[r, s] * y[inv[r, s], :] in slot order (inv < 0:
// an unrouted slot, no term). y bf16 [*, D]; out, sy fp32 [*, D]; slots <= 32.
extern "C" __global__ void __launch_bounds__(256) glm53_mtp_combine(const __nv_bfloat16* __restrict__ y,
                                                                     const int* __restrict__ inv,
                                                                     const float* __restrict__ wts,
                                                                     const float* __restrict__ sy,
                                                                     float* __restrict__ out, int slots, int D) {
    const int r = blockIdx.x;
    __shared__ int pos[32];
    __shared__ float w[32];
    if (threadIdx.x < slots) {
        pos[threadIdx.x] = inv[(size_t)r * slots + threadIdx.x];
        w[threadIdx.x] = wts[(size_t)r * slots + threadIdx.x];
    }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < slots; ++s) {
            const int p = pos[s];
            if (p >= 0) acc = fmaf(w[s], __bfloat162float(y[(size_t)p * D + d]), acc);
        }
        out[(size_t)r * D + d] = sy ? sy[(size_t)r * D + d] + acc : acc;
    }
}
