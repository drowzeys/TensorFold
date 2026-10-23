// tf-drama-probe's routed-expert statistics (families/drama/moe_stats.zig): fp64 sums of exact bf16 squares.

#include <cuda_bf16.h>

namespace {

constexpr int experts = 256, top = 8, hidden = 2048, inner = 512, threads = 256, classes = 3;

__device__ __forceinline__ double sq(__nv_bfloat16 v) {
    const double x = __bfloat162float(v);
    return x * x;
}

}  // namespace

// One layer's run of `rows` rows added to its sums (moe_stats.zig's layout); a block owns one expert, in pair order.
extern "C" __global__ void __launch_bounds__(threads)
    drama_probe_moe_stats(const int* ids, int semantic_start, int semantic_end, int generated, const int* picks,
                          const float* weights, const __nv_bfloat16* input, const __nv_bfloat16* act,
                          const __nv_bfloat16* down, int rows, double* sums) {
    const int e = blockIdx.x, t = threadIdx.x;
    __shared__ double partial[threads];
    for (int c = 0; c < classes; ++c) {
        double gu[hidden / threads] = {}, dn[inner / threads] = {};
        double count = 0.0, mass = 0.0, reap = 0.0;
        for (int p = 0; p < rows * top; ++p) {
            const int r = p / top;
            const int row_class = generated ? 2 : (ids[r] >= semantic_start && ids[r] <= semantic_end ? 1 : 0);
            if (picks[p] != e || row_class != c) continue;
            const __nv_bfloat16* x = input + static_cast<size_t>(r) * hidden;
            const __nv_bfloat16* a = act + static_cast<size_t>(p) * inner;
            const __nv_bfloat16* d = down + static_cast<size_t>(p) * hidden;
            double norm = 0.0;
#pragma unroll
            for (int j = 0; j < hidden / threads; ++j) {
                gu[j] += sq(x[t + j * threads]);
                norm += sq(d[t + j * threads]);
            }
#pragma unroll
            for (int j = 0; j < inner / threads; ++j) dn[j] += sq(a[t + j * threads]);
            partial[t] = norm;
            __syncthreads();
            for (int s = threads / 2; s > 0; s >>= 1) {
                if (t < s) partial[t] += partial[t + s];
                __syncthreads();
            }
            count += 1.0;
            mass += weights[p];
            reap += sqrt(partial[0]); // the stored row carries the routing weight: w times the unscaled norm
            __syncthreads();
        }
        if (count == 0.0) continue;
        double* gate_up = sums + 3 * classes * experts + (static_cast<size_t>(c) * experts + e) * hidden;
        double* act_sums = sums + 3 * classes * experts + classes * experts * hidden + (static_cast<size_t>(c) * experts + e) * inner;
#pragma unroll
        for (int j = 0; j < hidden / threads; ++j) gate_up[t + j * threads] += gu[j];
#pragma unroll
        for (int j = 0; j < inner / threads; ++j) act_sums[t + j * threads] += dn[j];
        if (t == 0) {
            sums[c * experts + e] += count;
            sums[(classes + c) * experts + e] += mass;
            sums[(2 * classes + c) * experts + e] += reap;
        }
    }
}

// An NVFP4 layer's pair rows as drama_moe_combine_f32 rounds them before its sum: out[p] = bf16(w[p] * down[p]).
extern "C" __global__ void __launch_bounds__(threads) drama_probe_weigh(const float* down, const float* w, __nv_bfloat16* out) {
    const size_t p = blockIdx.x;
    const float wp = w[p];
    for (int c = threadIdx.x; c < hidden; c += threads) out[p * hidden + c] = __float2bfloat16_rn(wp * down[p * hidden + c]);
}
