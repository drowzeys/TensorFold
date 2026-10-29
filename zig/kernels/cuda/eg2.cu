// EmbeddingGemma-2's text encoder glue (native/embgemma2.zig) at torch's bf16 rounding points.
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

// a block of 128 threads' sum, every thread gets it
__device__ __forceinline__ float block_sum(float s, float* part) {
    s = warp_sum(s);
    __syncthreads();
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = s;
    __syncthreads();
    float t = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) t = warp_sum(t);
    if (threadIdx.x == 0) part[0] = t;
    __syncthreads();
    return part[0];
}

// the model's RMSNorm: x * pow(mean(x^2) + eps, -0.5) in fp32
__device__ __forceinline__ float inv_rms(float sum, int n, float eps) {
    return powf(__fadd_rn(__fdiv_rn(sum, static_cast<float>(n)), eps), -0.5f);
}

// torch's gelu(approximate="tanh") in fp32
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
}  // namespace

// y [rows, n] = table[ids] * scale (bf16 product); a block of 128 a row, n <= 1024.
extern "C" __global__ void __launch_bounds__(128) tf_eg2_embed(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
                                                               __nv_bfloat16* __restrict__ y, int n, float scale) {
    const int64_t src = static_cast<int64_t>(ids[blockIdx.x]) * n;
    const int64_t dst = static_cast<int64_t>(blockIdx.x) * n;
    for (int c = threadIdx.x; c < n; c += 128) y[dst + c] = __float2bfloat16_rn(__fmul_rn(bf(table[src + c]), scale));
}

// y [rows, n] = RMSNorm(rbf(x * pre)) * w (w null: no scale); a block of 128 a row, n <= 512.
extern "C" __global__ void __launch_bounds__(128) tf_eg2_rms(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                             __nv_bfloat16* __restrict__ y, int n, float eps, float pre) {
    __shared__ float part[4];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * n;
    float v[4];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + 128 * i;
        float t = c < n ? bf(x[at + c]) : 0.0f;
        if (pre != 1.0f) t = rbf(__fmul_rn(t, pre));
        v[i] = t;
        s = __fadd_rn(s, __fmul_rn(t, t));
    }
    const float r = inv_rms(block_sum(s, part), n, eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + 128 * i;
        if (c >= n) continue;
        float o = __fmul_rn(v[i], r);
        if (w != nullptr) o = __fmul_rn(o, bf(w[c]));
        y[at + c] = __float2bfloat16_rn(o);
    }
}

// qkv [rows, (hq + 2 * hkv) * hd] in place: q and k RMSNorm with their weights then RoPE at the row's position in
// its input, v RMSNorm without a scale. A block of 128 per (row, head), hd <= 512.
extern "C" __global__ void __launch_bounds__(128) tf_eg2_qkv(__nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ qn,
                                                             const __nv_bfloat16* __restrict__ kn, const int* __restrict__ pos, int hq, int hkv,
                                                             int hd, float theta, float eps) {
    __shared__ float part[4];
    const int row = blockIdx.x;
    const int head = blockIdx.y;  // q heads, then k heads, then v heads
    const int heads = hq + 2 * hkv;
    __nv_bfloat16* h = qkv + (static_cast<int64_t>(row) * heads + head) * hd;
    const bool is_v = head >= hq + hkv;
    const __nv_bfloat16* w = head < hq ? qn : (is_v ? nullptr : kn);
    const int half = hd / 2;
    // thread owns pairs (c, c + half) for c = threadIdx.x + 128 * i < half
    float lo[2], hi[2];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int c = threadIdx.x + 128 * i;
        lo[i] = c < half ? bf(h[c]) : 0.0f;
        hi[i] = c < half ? bf(h[c + half]) : 0.0f;
        s = __fadd_rn(s, __fadd_rn(__fmul_rn(lo[i], lo[i]), __fmul_rn(hi[i], hi[i])));
    }
    const float r = inv_rms(block_sum(s, part), hd, eps);
    const float p = static_cast<float>(pos[row]);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int c = threadIdx.x + 128 * i;
        if (c >= half) continue;
        float a = __fmul_rn(lo[i], r), b = __fmul_rn(hi[i], r);
        if (w != nullptr) {
            a = __fmul_rn(a, bf(w[c]));
            b = __fmul_rn(b, bf(w[c + half]));
        }
        a = rbf(a);
        b = rbf(b);
        if (!is_v) {
            const float inv = __fdiv_rn(1.0f, powf(theta, __fdiv_rn(static_cast<float>(2 * c), static_cast<float>(hd))));
            const float f = __fmul_rn(p, inv);
            const float cs = rbf(cosf(f)), sn = rbf(sinf(f));
            const float na = rbf(__fadd_rn(rbf(__fmul_rn(a, cs)), rbf(__fmul_rn(-b, sn))));
            const float nb = rbf(__fadd_rn(rbf(__fmul_rn(b, cs)), rbf(__fmul_rn(a, sn))));
            a = na;
            b = nb;
        }
        h[c] = __float2bfloat16_rn(a);
        h[c + half] = __float2bfloat16_rn(b);
    }
}

// out [rows, hq * hd] = softmax(q k^T) v over the row's own input (lo[row] <= key < hi[row]), and within `window`
// positions either side when window > 0; scale 1 (the q/k norms carry it). A warp per (row, q head); hd 256 or 512.
template <int D>
__device__ __forceinline__ void attend(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out, const int* __restrict__ lo,
                                       const int* __restrict__ hi, int rows, int hq, int hkv, int window) {
    constexpr int per = D / 32;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= rows * hq) return;
    const int row = warp / hq;
    const int head = warp % hq;
    const int kvh = head / (hq / hkv);
    const int heads = hq + 2 * hkv;
    const int64_t stride = static_cast<int64_t>(heads) * D;
    const __nv_bfloat16* q = qkv + row * stride + head * D;
    float qv[per], acc[per];
#pragma unroll
    for (int i = 0; i < per; ++i) {
        qv[i] = bf(q[lane + 32 * i]);
        acc[i] = 0.0f;
    }
    int a = lo[row], b = hi[row];
    if (window > 0) {
        a = max(a, row - window);
        b = min(b, row + window + 1);
    }
    float m = -INFINITY, l = 0.0f;
    for (int j = a; j < b; ++j) {
        const __nv_bfloat16* k = qkv + j * stride + (hq + kvh) * D;
        const __nv_bfloat16* v = qkv + j * stride + (hq + hkv + kvh) * D;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < per; ++i) s = fmaf(qv[i], bf(k[lane + 32 * i]), s);
        s = warp_sum(s);
        const float mn = fmaxf(m, s);
        const float c = __expf(m - mn);
        const float e = __expf(s - mn);
        l = l * c + e;
#pragma unroll
        for (int i = 0; i < per; ++i) acc[i] = fmaf(e, bf(v[lane + 32 * i]), acc[i] * c);
        m = mn;
    }
    __nv_bfloat16* o = out + (static_cast<int64_t>(row) * hq + head) * D;
#pragma unroll
    for (int i = 0; i < per; ++i) o[lane + 32 * i] = __float2bfloat16_rn(acc[i] / l);
}

extern "C" __global__ void __launch_bounds__(128) tf_eg2_attn256(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
                                                                 const int* __restrict__ lo, const int* __restrict__ hi, int rows, int hq,
                                                                 int hkv, int window) {
    attend<256>(qkv, out, lo, hi, rows, hq, hkv, window);
}

extern "C" __global__ void __launch_bounds__(128) tf_eg2_attn512(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out,
                                                                 const int* __restrict__ lo, const int* __restrict__ hi, int rows, int hq,
                                                                 int hkv, int window) {
    attend<512>(qkv, out, lo, hi, rows, hq, hkv, window);
}

// h = rbf(h + rbf(RMSNorm(x) * wp)), then h = rbf(h * scalar) when scalar != 1, then y = RMSNorm(h) * wn when wn is set:
// a sub-block's post norm, its residual add, the layer scalar and the next norm in one pass. A block of 128 a row, n <= 512.
extern "C" __global__ void __launch_bounds__(128) tf_eg2_add_norm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ wp,
                                                                  __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ wn,
                                                                  __nv_bfloat16* __restrict__ y, int n, float eps, float scalar) {
    __shared__ float part[4];
    const int64_t at = static_cast<int64_t>(blockIdx.x) * n;
    float v[4];
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + 128 * i;
        v[i] = c < n ? bf(x[at + c]) : 0.0f;
        s = __fadd_rn(s, __fmul_rn(v[i], v[i]));
    }
    const float r = inv_rms(block_sum(s, part), n, eps);
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + 128 * i;
        if (c >= n) {
            v[i] = 0.0f;
            continue;
        }
        float o = rbf(__fmul_rn(__fmul_rn(v[i], r), bf(wp[c])));
        o = rbf(__fadd_rn(bf(h[at + c]), o));
        if (scalar != 1.0f) o = rbf(__fmul_rn(o, scalar));
        h[at + c] = __float2bfloat16_rn(o);
        v[i] = o;
        t = __fadd_rn(t, __fmul_rn(o, o));
    }
    if (wn == nullptr) return;
    const float r2 = inv_rms(block_sum(t, part), n, eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + 128 * i;
        if (c < n) y[at + c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[i], r2), bf(wn[c])));
    }
}

// y [rows, inter] = rbf(gelu(gate)) * up from gu [rows, gate | up]
extern "C" __global__ void __launch_bounds__(256) tf_eg2_gelu_mul(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ y, int inter,
                                                                  int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t r = i / inter;
    const int c = static_cast<int>(i % inter);
    const float g = rbf(gelu_tanh(bf(gu[r * 2 * inter + c])));
    y[i] = __float2bfloat16_rn(__fmul_rn(g, bf(gu[r * 2 * inter + inter + c])));
}

// the per-layer input gate: y [rows, n] = rbf(gelu(g)) * pli[row, layer] (pli [rows, layers, n])
extern "C" __global__ void __launch_bounds__(256) tf_eg2_ple_gate(const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ pli,
                                                                  __nv_bfloat16* __restrict__ y, int n, int layers, int layer, int64_t count) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t r = i / n;
    const int c = static_cast<int>(i % n);
    const float a = rbf(gelu_tanh(bf(g[i])));
    y[i] = __float2bfloat16_rn(__fmul_rn(a, bf(pli[(r * layers + layer) * n + c])));
}

// out [inputs, dim] f32 = L2-normalised (mean of x's rows for each input) projected by w [dim, n]: mean pooling,
// the embedding projection (linear, so taken once after the mean) and the normalisation. A block of 256 an input.
extern "C" __global__ void __launch_bounds__(256) tf_eg2_pool(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ends,
                                                              const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int n, int dim) {
    __shared__ float mean[512];
    __shared__ float part[8];
    const int b = blockIdx.x;
    const int lo = b == 0 ? 0 : ends[b - 1];
    const int hi = ends[b];
    for (int c = threadIdx.x; c < n; c += 256) {
        float s = 0.0f;
        for (int r = lo; r < hi; ++r) s = __fadd_rn(s, bf(x[static_cast<int64_t>(r) * n + c]));
        mean[c] = __fdiv_rn(s, static_cast<float>(hi - lo));
    }
    __syncthreads();
    float vals[4];
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int o = threadIdx.x + 256 * i;
        float s = 0.0f;
        if (o < dim)
            for (int c = 0; c < n; ++c) s = fmaf(mean[c], bf(w[static_cast<int64_t>(o) * n + c]), s);
        vals[i] = s;
        t = __fadd_rn(t, __fmul_rn(s, s));
    }
    t = warp_sum(t);
    if (threadIdx.x % 32 == 0) part[threadIdx.x / 32] = t;
    __syncthreads();
    if (threadIdx.x < 32) {
        float u = threadIdx.x < 8 ? part[threadIdx.x] : 0.0f;
        u = warp_sum(u);
        if (threadIdx.x == 0) part[0] = u;
    }
    __syncthreads();
    const float norm = sqrtf(part[0]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int o = threadIdx.x + 256 * i;
        if (o < dim) out[static_cast<int64_t>(b) * dim + o] = vals[i] / norm;
    }
}
