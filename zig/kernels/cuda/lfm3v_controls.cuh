// lfm3v.cu's planted controls: each kernel's twin with one realistic defect, which its gate must report.
#pragma once
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_ctl_lfm3v {
__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float s) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    return s;
}

__device__ __forceinline__ float inv_rms(float sum, int n, float eps) {
    return __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum, static_cast<float>(n)), eps)));
}
}  // namespace tf_ctl_lfm3v

// tf_lfm3v_rms with x * r left in fp32: the bf16 point torch rounds the normed row at is dropped.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_lfm3v_rms(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                                   __nv_bfloat16* __restrict__ y, int n, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * n;
    float v[8];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        v[i] = c < n ? tf_ctl_lfm3v::bf(x[at + c]) : 0.0f;
        s = __fadd_rn(s, __fmul_rn(v[i], v[i]));
    }
    __shared__ float part[8];
    s = tf_ctl_lfm3v::warp_sum(s);
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < 8 ? part[threadIdx.x] : 0.0f;
        t = tf_ctl_lfm3v::warp_sum(t);
        if (threadIdx.x == 0) part[0] = t;
    }
    __syncthreads();
    const float r = tf_ctl_lfm3v::inv_rms(part[0], n, eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        if (c < n) y[at + c] = __float2bfloat16_rn(__fmul_rn(tf_ctl_lfm3v::bf(w[c]), __fmul_rn(v[i], r)));
    }
}

// tf_lfm3v_qk rotating the normed head in fp32: the bf16 head torch hands to the rotation is skipped.
extern "C" __global__ void __launch_bounds__(32) tf_ctl_lfm3v_qk(__nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ qw,
                                                                 const __nv_bfloat16* __restrict__ kw, const float* __restrict__ cs,
                                                                 const float* __restrict__ sn, int P, int H, float eps) {
    constexpr int D = 64;
    const int g = blockIdx.x, lane = threadIdx.x;
    const int part = g % 2, head = (g / 2) % H, row = g / (2 * H);
    if (row >= P) return;
    __nv_bfloat16* x = qkv + (static_cast<int64_t>(row) * 3 + part) * H * D + head * D;
    const __nv_bfloat16* w = part == 0 ? qw : kw;
    const float a = tf_ctl_lfm3v::bf(x[lane]), b = tf_ctl_lfm3v::bf(x[lane + 32]);
    const float r = tf_ctl_lfm3v::inv_rms(tf_ctl_lfm3v::warp_sum(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b))), D, eps);
    const float na = __fmul_rn(tf_ctl_lfm3v::bf(w[lane]), tf_ctl_lfm3v::rbf(__fmul_rn(a, r)));
    const float nb = __fmul_rn(tf_ctl_lfm3v::bf(w[lane + 32]), tf_ctl_lfm3v::rbf(__fmul_rn(b, r)));
    const float* c = cs + static_cast<int64_t>(row) * D;
    const float* s = sn + static_cast<int64_t>(row) * D;
    x[lane] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(na, c[lane]), __fmul_rn(-nb, s[lane])));
    x[lane + 32] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(nb, c[lane + 32]), __fmul_rn(na, s[lane + 32])));
}

// tf_lfm3v_conv taking the rows in raster order: a grid neighbour read as the row next to it in memory.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_lfm3v_conv(const __nv_bfloat16* __restrict__ bcx, const __nv_bfloat16* __restrict__ w,
                                                                    __nv_bfloat16* __restrict__ out, int gh, int gw, int C) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int P = gh * gw;
    if (t >= static_cast<int64_t>(P) * C) return;
    const int i = static_cast<int>(t / C), ch = static_cast<int>(t % C);
    const int y = i / gw, x = i % gw;
    float acc = 0.0f;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int yy = y + dy, xx = x + dx;
            if (yy < 0 || yy >= gh || xx < 0 || xx >= gw) continue;
            const int64_t j = static_cast<int64_t>(yy * gw + xx) * 3 * C;
            const float gated = tf_ctl_lfm3v::rbf(__fmul_rn(tf_ctl_lfm3v::bf(bcx[j + ch]), tf_ctl_lfm3v::bf(bcx[j + 2 * C + ch])));
            acc = __fmaf_rn(tf_ctl_lfm3v::bf(w[ch * 9 + (dy + 1) * 3 + (dx + 1)]), gated, acc);
        }
    }
    const float cgate = tf_ctl_lfm3v::bf(bcx[static_cast<int64_t>(i) * 3 * C + C + ch]);
    out[t] = __float2bfloat16_rn(__fmul_rn(cgate, tf_ctl_lfm3v::rbf(acc)));
}

// tf_lfm3v_add with a whole-block fast path and no tail: a count that is not a multiple of 256 loses its last block.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_lfm3v_add(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                                                                   __nv_bfloat16* __restrict__ y, long long count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (static_cast<int64_t>(blockIdx.x + 1) * blockDim.x > count) return;
    y[i] = __float2bfloat16_rn(__fadd_rn(tf_ctl_lfm3v::bf(a[i]), tf_ctl_lfm3v::bf(b[i])));
}

// tf_lfm3v_silu_mul with silu left in fp32: the bf16 point torch rounds silu(g) at is dropped.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_lfm3v_silu_mul(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ y, int n,
                                                                        long long count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t row = i / n, c = i % n;
    const float g = tf_ctl_lfm3v::bf(gu[row * 2 * n + c]);
    const float s = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    y[i] = __float2bfloat16_rn(__fmul_rn(s, tf_ctl_lfm3v::bf(gu[row * 2 * n + n + c])));
}

// tf_lfm3v_set_attn mapping query head h to kv head h % Hk: GQA's interleaved map where LFM3 groups heads h / (H / Hk).
extern "C" __global__ void __launch_bounds__(32) tf_ctl_lfm3v_set_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                                                                       const int* __restrict__ group, __nv_bfloat16* __restrict__ out, int R, int H,
                                                                       int Hk, int state, float scale) {
    constexpr int D = 128;
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (row >= R) return;
    const int kvh = head % Hk, kw = Hk * D;
    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = tf_ctl_lfm3v::bf(q[static_cast<int64_t>(row) * H * D + head * D + lane + 32 * i]);
    const int g = group[row];
    float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < R; ++j) {
        if (!(row < state || j < state || group[j] == g)) continue;
        const __nv_bfloat16* k = kv + static_cast<int64_t>(j) * 2 * kw + kvh * D;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) s = __fmaf_rn(qv[i], tf_ctl_lfm3v::bf(k[lane + 32 * i]), s);
        s = __fmul_rn(tf_ctl_lfm3v::warp_sum(s), scale);
        const float mn = fmaxf(m, s);
        const float c = __expf(m - mn), p = __expf(s - mn);
        l = __fadd_rn(__fmul_rn(l, c), p);
        const __nv_bfloat16* v = k + kw;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = __fadd_rn(__fmul_rn(o[i], c), __fmul_rn(tf_ctl_lfm3v::rbf(p), tf_ctl_lfm3v::bf(v[lane + 32 * i])));
        m = mn;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) out[static_cast<int64_t>(row) * H * D + head * D + lane + 32 * i] = __float2bfloat16_rn(__fdiv_rn(o[i], l));
}

// tf_lfm3v_set_attn_tiled skipping a tile by its first key's group alone: a question starting mid-tile loses keys.
extern "C" __global__ void __launch_bounds__(512) tf_ctl_lfm3v_set_attn_tiled(const __nv_bfloat16* __restrict__ q,
                                                                              const __nv_bfloat16* __restrict__ kv, const int* __restrict__ group,
                                                                              __nv_bfloat16* __restrict__ out, int R, int H, int Hk, int state,
                                                                              float scale) {
    constexpr int D = 128, ROWS = 4, TILE = 32;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int kvh = blockIdx.y, per = H / Hk;
    const int row = blockIdx.x * ROWS + warp / per, head = kvh * per + warp % per;
    const int kw = Hk * D;
    __shared__ __nv_bfloat16 ks[TILE][D];
    __shared__ __nv_bfloat16 vs[TILE][D];
    __shared__ int gs[TILE];
    const bool live = row < R;
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int g = 0;
    if (live) {
#pragma unroll
        for (int i = 0; i < 4; ++i) qv[i] = tf_ctl_lfm3v::bf(q[static_cast<int64_t>(row) * H * D + head * D + lane + 32 * i]);
        g = group[row];
    }
    const int r0 = blockIdx.x * ROWS, r1 = min(R, r0 + ROWS);
    const bool block_state = r0 < state;
    const int g_lo = group[r0], g_hi = group[r1 - 1];
    float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t0 = 0; t0 < R; t0 += TILE) {
        const int t1 = min(R, t0 + TILE);
        if (!block_state && t0 >= state) {
            const int a = group[t0];
            if (a < g_lo || a > g_hi) continue;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < TILE * D / 8; i += blockDim.x) {
            const int r = i / (D / 8), c = (i % (D / 8)) * 8, key = t0 + r;
            uint4 kk = make_uint4(0, 0, 0, 0), vv = kk;
            if (key < R) {
                const __nv_bfloat16* src = kv + static_cast<int64_t>(key) * 2 * kw + kvh * D + c;
                kk = *reinterpret_cast<const uint4*>(src);
                vv = *reinterpret_cast<const uint4*>(src + kw);
            }
            *reinterpret_cast<uint4*>(&ks[r][c]) = kk;
            *reinterpret_cast<uint4*>(&vs[r][c]) = vv;
        }
        if (threadIdx.x < TILE) gs[threadIdx.x] = t0 + threadIdx.x < R ? group[t0 + threadIdx.x] : -2;
        __syncthreads();
        if (!live) continue;
        for (int r = 0; r < t1 - t0; ++r) {
            const int j = t0 + r;
            if (!(row < state || j < state || gs[r] == g)) continue;
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) s = __fmaf_rn(qv[i], tf_ctl_lfm3v::bf(ks[r][lane + 32 * i]), s);
            s = __fmul_rn(tf_ctl_lfm3v::warp_sum(s), scale);
            const float mn = fmaxf(m, s);
            const float c = __expf(m - mn), p = __expf(s - mn);
            l = __fadd_rn(__fmul_rn(l, c), p);
#pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = __fadd_rn(__fmul_rn(o[i], c), __fmul_rn(tf_ctl_lfm3v::rbf(p), tf_ctl_lfm3v::bf(vs[r][lane + 32 * i])));
            m = mn;
        }
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < 4; ++i) out[static_cast<int64_t>(row) * H * D + head * D + lane + 32 * i] = __float2bfloat16_rn(__fdiv_rn(o[i], l));
}
