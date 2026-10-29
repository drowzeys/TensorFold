// LFM3-VL's vision tower glue (lfm3_vision.zig), at torch's rounding points:
//   RMSNorm: bf16(w * bf16(x * rsqrt(mean(x^2) + eps))), the mean in fp32;
//   q and k: that norm per head, then RoPE in fp32 on the bf16 rows (cos, sin fp32 [P, D]), rounded once;
//   the gated depthwise 3x3 conv: bf16(C * bf16(conv(bf16(B * x)))) over each frame's raster grid, rows in merge order.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float s) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    return s;
}

// torch's rsqrt of the mean square plus eps, as the other towers here take it
__device__ __forceinline__ float inv_rms(float sum, int n, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, static_cast<float>(n)), eps)));
}
}  // namespace

// y [rows, n] = RMSNorm(x) * w; a block of 256 a row, n <= 2048.
extern "C" __global__ void __launch_bounds__(256) tf_lfm3v_rms(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                               __nv_bfloat16* __restrict__ y, int n, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * n;
    float v[8];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        v[i] = c < n ? bf(x[at + c]) : 0.0f;
        s = __fadd_rn(s, __fmul_rn(v[i], v[i]));
    }
    __shared__ float part[8];
    s = warp_sum(s);
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < 8 ? part[threadIdx.x] : 0.0f;
        t = warp_sum(t);
        if (threadIdx.x == 0) part[0] = t;
    }
    __syncthreads();
    const float r = inv_rms(part[0], n, eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        if (c < n) y[at + c] = __float2bfloat16_rn(__fmul_rn(bf(w[c]), rbf(__fmul_rn(v[i], r))));
    }
}

// qkv [P, 3 H 64] in place: q (part 0) and k (part 1) heads RMSNormed then rotated; v untouched. A warp a (row, head, part).
extern "C" __global__ void __launch_bounds__(32) tf_lfm3v_qk(__nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ qw,
                                                             const __nv_bfloat16* __restrict__ kw, const float* __restrict__ cs,
                                                             const float* __restrict__ sn, int P, int H, float eps) {
    constexpr int D = 64;
    const int g = blockIdx.x, lane = threadIdx.x;
    const int part = g % 2, head = (g / 2) % H, row = g / (2 * H);
    if (row >= P) return;
    __nv_bfloat16* x = qkv + (static_cast<int64_t>(row) * 3 + part) * H * D + head * D;
    const __nv_bfloat16* w = part == 0 ? qw : kw;
    const float a = bf(x[lane]), b = bf(x[lane + 32]);
    const float r = inv_rms(warp_sum(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b))), D, eps);
    // the normed head, bf16 as torch hands it to the rotation
    const float na = rbf(__fmul_rn(bf(w[lane]), rbf(__fmul_rn(a, r))));
    const float nb = rbf(__fmul_rn(bf(w[lane + 32]), rbf(__fmul_rn(b, r))));
    const float* c = cs + static_cast<int64_t>(row) * D;
    const float* s = sn + static_cast<int64_t>(row) * D;
    // rotate_half: (-x2, x1); dim d < 32 pairs with d + 32
    x[lane] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(na, c[lane]), __fmul_rn(-nb, s[lane])));
    x[lane + 32] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(nb, c[lane + 32]), __fmul_rn(na, s[lane + 32])));
}

// The raster (y, x) of merge-order row i of a gh x gw frame (2 x 2 merge blocks).
__device__ __forceinline__ void raster(int i, int gw, int& y, int& x) {
    const int block = i / 4, inner = i % 4, bw = gw / 2;
    y = (block / bw) * 2 + inner / 2;
    x = (block % bw) * 2 + inner % 2;
}

__device__ __forceinline__ int merged(int y, int x, int gw) { return ((y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2; }

// out [P, C] = bf16(C * bf16(dwconv3x3(bf16(B * x)))) of bcx [P, 3C] (B | C | x), w [C, 9]; one frame of gh x gw rows.
// A thread a (row, channel); zero padding outside the frame.
extern "C" __global__ void __launch_bounds__(256) tf_lfm3v_conv(const __nv_bfloat16* __restrict__ bcx, const __nv_bfloat16* __restrict__ w,
                                                                __nv_bfloat16* __restrict__ out, int gh, int gw, int C) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int P = gh * gw;
    if (t >= static_cast<int64_t>(P) * C) return;
    const int i = static_cast<int>(t / C), ch = static_cast<int>(t % C);
    int y, x;
    raster(i, gw, y, x);
    float acc = 0.0f;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= gh || xx < 0 || xx >= gw) continue;
            const int64_t j = static_cast<int64_t>(merged(yy, xx, gw)) * 3 * C;
            const float gated = rbf(__fmul_rn(bf(bcx[j + ch]), bf(bcx[j + 2 * C + ch])));
            acc = __fmaf_rn(bf(w[ch * 9 + (dy + 1) * 3 + (dx + 1)]), gated, acc);
        }
    }
    const float cgate = bf(bcx[static_cast<int64_t>(i) * 3 * C + C + ch]);
    out[t] = __float2bfloat16_rn(__fmul_rn(cgate, rbf(acc)));
}

// y [rows, n] = bf16(a + b), torch's residual add.
extern "C" __global__ void __launch_bounds__(256) tf_lfm3v_add(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                                                               __nv_bfloat16* __restrict__ y, long long count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) y[i] = __float2bfloat16_rn(__fadd_rn(bf(a[i]), bf(b[i])));
}

// y [rows, n] = bf16(silu(bf16 g) ) * u as torch: F.silu on the bf16 w1 rows, then the bf16 product; gu [rows, 2n] (w1 | w3).
extern "C" __global__ void __launch_bounds__(256) tf_lfm3v_silu_mul(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ y,
                                                                    int n, long long count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t row = i / n, c = i % n;
    const float g = bf(gu[row * 2 * n + c]);
    const float s = rbf(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))));
    y[i] = __float2bfloat16_rn(__fmul_rn(s, bf(gu[row * 2 * n + n + c])));
}

// d1's `set` attention on a full-attention layer of one packed row (attention.py `Sees`): a query sees key j when the
// query or the key is in the state (row < state) or both are in the same question. q [R, H 128], kv [R, 2 Hk 128]
// (keys then values, normed and rotated), out [R, H 128]; GQA H / Hk. A warp a (row, head), fp32 online softmax.
extern "C" __global__ void __launch_bounds__(32) tf_lfm3v_set_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                                                                   const int* __restrict__ group, __nv_bfloat16* __restrict__ out, int R,
                                                                   int H, int Hk, int state, float scale) {
    constexpr int D = 128;
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (row >= R) return;
    const int kvh = head / (H / Hk), kw = Hk * D;
    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = bf(q[static_cast<int64_t>(row) * H * D + head * D + lane + 32 * i]);
    const int g = group[row];
    float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < R; ++j) {
        if (!(row < state || j < state || group[j] == g)) continue;
        const __nv_bfloat16* k = kv + static_cast<int64_t>(j) * 2 * kw + kvh * D;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) s = __fmaf_rn(qv[i], bf(k[lane + 32 * i]), s);
        s = warp_sum(s) * scale;
        const float mn = fmaxf(m, s);
        const float c = __expf(m - mn), p = __expf(s - mn);
        l = l * c + p;
        const __nv_bfloat16* v = k + kw;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = o[i] * c + rbf(p) * bf(v[lane + 32 * i]);  // flash's bf16 P for P.V
        m = mn;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) out[static_cast<int64_t>(row) * H * D + head * D + lane + 32 * i] = __float2bfloat16_rn(o[i] / l);
}
