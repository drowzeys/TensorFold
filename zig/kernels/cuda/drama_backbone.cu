// Drama (Qwen3.5-MoE hybrid) backbone glue: embedding mix, zero-centered residual norms, the Gated DeltaNet
// convolution, decode recurrence and chunked prefill (drama_gdn_chunk.cuh), gated full attention with partial NeoX
// RoPE, the softmax top-8 router, dynamic 1 x 128 FP8 activation quantization, SwiGLU and the routed + gated shared
// expert combine.
//
// Every formula names the reference op it mirrors (SGLang v0.5.21 / flashinfer 0.6.18 / FLA as pinned by
// sglang-omni ab095ef); math runs in fp32 and rounds to bf16 where the reference stores or casts to a bf16 value.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace drama {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 tobf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float round_bf(float v) { return bf(tobf(v)); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, m));
    return x;
}

// Block-wide sum for blocks of `nwarps` full warps; every thread gets the total.
__device__ __forceinline__ float block_sum(float x, float* scratch, int nwarps) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    x = warp_sum(x);
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    float s = lane < nwarps ? scratch[lane] : 0.f;
    s = warp_sum(s);
    __syncthreads();
    return s;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = bf(b[i]);
}

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&v)[8]) {
    uint4 u;
    __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) b[i] = tobf(v[i]);
    *reinterpret_cast<uint4*>(p) = u;
}

// drama_act_quant's e4m3 byte of v under its group's multiplier inv = 448 / amax.
__device__ __forceinline__ uint32_t e4m3(float v, float inv) {
    const float c = fminf(fmaxf(v * inv, -448.0f), 448.0f);
    return static_cast<uint32_t>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3));
}

// Programmatic dependent launch: every kernel here starts with this, before any load or store of global memory. The
// wait returns once the previous kernel in the stream has completed and its writes are visible (at once if it has);
// the trigger lets the next PDL launch start. sm_90+ only, where drama_pdl marks the image.
__device__ __forceinline__ void grid_dependency() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

}  // namespace drama

using namespace drama;

#if __CUDA_ARCH__ >= 900
// Present only in images whose every kernel begins with grid_dependency(): backbone_kernels.zig launches with PDL only
// when both backbone images hold it, so an older or mixed prebuilt image set launches plain.
extern "C" __device__ const int drama_pdl = 1;
#endif

// Semantic rows (ids in [lo, hi]): out = bf16(bf16(out + extra) * 0.70710677), audio_decoder.embed_backbone with
// one backbone codebook: (text_embeddings + codebook_embedding_sum.to(bf16)) * (1 / sqrt(2)), each torch op rounding.
// Other rows keep the token embedding. Grid: rows; block 256.
extern "C" __global__ void __launch_bounds__(256) drama_semantic_mix(const int* ids, __nv_bfloat16* out,
                                                                    const __nv_bfloat16* extra, int width,
                                                                    int lo, int hi) {
    grid_dependency();
    const int r = blockIdx.x;
    const int id = ids[r];
    if (id < lo || id > hi) return;
    const float scale = 0.70710678118654752f;
    __nv_bfloat16* o = out + static_cast<size_t>(r) * width;
    const __nv_bfloat16* e = extra + static_cast<size_t>(r) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float s = round_bf(bf(o[i]) + bf(e[i]));
        o[i] = tobf(s * scale);
    }
}

// GemmaRMSNorm with the installed FlashInfer CuTe-DSL order for 2048-wide BF16 rows:
// eight contiguous values per lane, FP32 FMA squares, then XOR offsets 1, 2, 4, 8, 16.
// first: v = x, res = x; else v = float(x) + float(res), with one BF16 residual store.
// The reduction finishes before any in-place residual write. q / s (both or neither): y's FP8 quantization as
// drama_act_quant, for the routed experts. width % 256 == 0. Grid: rows; block: 256.
extern "C" __global__ void __launch_bounds__(256) drama_add_norm(const __nv_bfloat16* x, __nv_bfloat16* res,
                                                                const __nv_bfloat16* w, __nv_bfloat16* y,
                                                                uint8_t* q, float* s, int width, float eps,
                                                                int first) {
    grid_dependency();
    __shared__ float square_sum;
    const size_t base = static_cast<size_t>(blockIdx.x) * width;
    const int t = threadIdx.x;
    if (t < 32) {
        // Unrolled so every load is in flight before the FMA chain, which keeps the serial reference order.
        float sum = 0.f;
#pragma unroll 8
        for (int block = 0; block < width; block += 256) {
            float value[8], other[8];
            load8(x + base + block + 8 * t, value);
            if (!first) load8(res + base + block + 8 * t, other);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                if (!first) value[v] += other[v];
                sum = fmaf(value[v], value[v], sum);
            }
        }
        for (int offset = 1; offset < 32; offset <<= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (t == 0) square_sum = sum;
    }
    __syncthreads();
    const float rcp = rsqrtf(square_sum / static_cast<float>(width) + eps);
    for (int c = 8 * t; c < width; c += 8 * blockDim.x) {
        float value[8], other[8], weight[8], out[8];
        load8(x + base + c, value);
        if (!first) load8(res + base + c, other);
        load8(w + c, weight);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (!first) value[i] += other[i];
            out[i] = value[i] * rcp * (1.0f + weight[i]);
        }
        store8(res + base + c, value);
        store8(y + base + c, out);
        if (q == nullptr) continue;
        // A 128-wide group is 16 lanes; the max is exact in any order.
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = round_bf(out[i]);
            amax = fmaxf(amax, fabsf(out[i]));
        }
#pragma unroll
        for (int m = 8; m; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        amax = fmaxf(amax, 1e-10f);
        const float inv = __fdividef(448.0f, amax);
        uint2 packed = {0u, 0u};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            packed.x |= e4m3(out[i], inv) << (8 * i);
            packed.y |= e4m3(out[4 + i], inv) << (8 * i);
        }
        *reinterpret_cast<uint2*>(q + base + c) = packed;
        if ((t & 15) == 0) s[static_cast<size_t>(blockIdx.x) * (width / 128) + c / 128] = amax * (1.0f / 448.0f);
    }
}

// Drama's 32 value heads, width 128. Canonical SGLang FLA full-extend arithmetic:
// eight contiguous values per lane, two heads per warp. Keep this order for every
// token count; the reference's one-head-per-warp short layout has different rounding.
// Grid: (rows, 32 / (block / 16)); block 32 or 256. No auxiliary sums output.
extern "C" __global__ void __launch_bounds__(256) drama_gated_rms_128(
    const __nv_bfloat16* y, long long y_ld,
    const __nv_bfloat16* z, long long z_ld,
    const __nv_bfloat16* w, __nv_bfloat16* out, float eps) {
    grid_dependency();
    const int sub = threadIdx.x & 15;
    const int hd = (blockIdx.y * blockDim.x + threadIdx.x) / 16;
    const long long row = blockIdx.x;
    float v[8], gate[8], weight[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int col = sub * 8 + i;
        v[i] = __bfloat162float(y[row * y_ld + hd * 128 + col]);
        gate[i] = __bfloat162float(z[row * z_ld + hd * 128 + col]);
        weight[i] = __bfloat162float(w[col]);
    }
    float sum = __fmul_rn(v[1], v[1]);
    sum = __fmaf_rn(v[0], v[0], sum);
#pragma unroll
    for (int i = 2; i < 8; ++i) sum = __fmaf_rn(v[i], v[i], sum);
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, mask));
    float variance;
    asm("div.full.f32 %0, %1, %2;" : "=f"(variance) : "f"(sum), "f"(128.0f));
    variance = __fadd_rn(variance, eps);
    float inv;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(variance));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float exponent = __fmul_rn(__fsub_rn(0.0f, gate[i]), __int_as_float(0x3fb8aa3b));
        float e;
        asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
        const float denom = __fadd_rn(e, 1.0f);
        float sigmoid;
        asm("div.full.f32 %0, %1, %2;" : "=f"(sigmoid) : "f"(1.0f), "f"(denom));
        const float norm = __fmul_rn(__fmul_rn(v[i], inv), weight[i]);
        const float result = __fmul_rn(norm, __fmul_rn(gate[i], sigmoid));
        out[(row * 32 + hd) * 128 + sub * 8 + i] = __float2bfloat16_rn(result);
    }
}

// Causal depthwise conv over the packed q|k|v channels (Qwen3.5 GatedDeltaNet conv1d, kernel 4, no bias) with
// silu, as CUDA causal_conv1d_update: FP32 products and accumulation, y = bf16(silu(acc)).
// state: bf16 [3][C], the three previous raw inputs (oldest first), read then advanced. Grid: ceil(C / 256); block 256.
extern "C" __global__ void __launch_bounds__(256) drama_gdn_conv(const __nv_bfloat16* in, int ld_in,
                                                                const __nv_bfloat16* w, __nv_bfloat16* state,
                                                                __nv_bfloat16* out, int rows, int C) {
    grid_dependency();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = bf(w[4 * c]), w1 = bf(w[4 * c + 1]), w2 = bf(w[4 * c + 2]), w3 = bf(w[4 * c + 3]);
    __nv_bfloat16 h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    for (int t = 0; t < rows; ++t) {
        const __nv_bfloat16 xt = in[static_cast<size_t>(t) * ld_in + c];
        float acc = w0 * bf(h0);
        acc += w1 * bf(h1);
        acc += w2 * bf(h2);
        acc += w3 * bf(xt);
        out[static_cast<size_t>(t) * C + c] = tobf(silu(acc));
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }
    state[c] = h0;
    state[C + c] = h1;
    state[2 * C + c] = h2;
}

// Triton prefill has a different precision boundary: BF16 products, then FP32 accumulation.
// Match its ex2.approx / div.full activation too; keep CUDA decode above unchanged.
extern "C" __global__ void __launch_bounds__(256) drama_gdn_conv_prefill(
    const __nv_bfloat16* in, int ld_in, const __nv_bfloat16* w, __nv_bfloat16* state,
    __nv_bfloat16* out, int rows, int C) {
    grid_dependency();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = bf(w[4 * c]), w1 = bf(w[4 * c + 1]), w2 = bf(w[4 * c + 2]), w3 = bf(w[4 * c + 3]);
    __nv_bfloat16 h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
    for (int t = 0; t < rows; ++t) {
        const __nv_bfloat16 xt = in[static_cast<size_t>(t) * ld_in + c];
        float acc = 0.0f + bf(tobf(w0 * bf(h0)));
        acc += bf(tobf(w1 * bf(h1)));
        acc += bf(tobf(w2 * bf(h2)));
        acc += bf(tobf(w3 * bf(xt)));
        const float exponent = __fmul_rn(__fsub_rn(0.0f, acc), __int_as_float(0x3fb8aa3b));
        float exponential;
        asm("ex2.approx.f32 %0, %1;" : "=f"(exponential) : "f"(exponent));
        const float denominator = __fadd_rn(1.0f, exponential);
        float y;
        asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(acc), "f"(denominator));
        out[static_cast<size_t>(t) * C + c] = tobf(y);
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }
    state[c] = h0;
    state[C + c] = h1;
    state[2 * C + c] = h2;
}

// Gated delta rule, token by token, the per-step math of FLA's packed decode (fused_recurrent.py):
//   g = -exp(A_log) * softplus(a + dt_bias) (softplus = log(1 + exp(x)) for x <= 20, else x),
//   beta = bf16(sigmoid(b)) (packed decode casts it to b's dtype, fused_gdn_gating stores it as bf16);
//   q = q / sqrt(sum q^2 + 1e-6) * scale, k = k / sqrt(sum k^2 + 1e-6);
//   S *= exp(g); v -= sum_k S k; v *= beta; S += k (x) v; o = sum_k S q.
// qkv: bf16 rows [q (NK x 128) | k (NK x 128) | v (NV x 128)] ld_qkv apart (the conv output);
// ba: bf16 rows [b (NV) | a (NV)] ld_ba apart; key head of value head h is h / (NV / NK) (repeat_interleave).
// state: fp32 [NV][128 (v)][128 (k)], k contiguous; y: bf16 [rows][NV][128].
// Grid: (128 / (8 * 4), NV); block 128: a warp owns 8 value columns, a lane 4 k entries of each.
// Decode only: packed decode keeps the normalized q/k in fp32. Prefill is the chunked algorithm of
// drama_gdn_chunk.cuh. Same state layout.
extern "C" __global__ void __launch_bounds__(128) drama_gdn_recur(const __nv_bfloat16* qkv, int ld_qkv,
                                                                 const __nv_bfloat16* ba, int ld_ba,
                                                                 const float* a_log, const float* dt_bias,
                                                                 float* state, __nv_bfloat16* y, int rows, int nk,
                                                                 int nv, float scale) {
    grid_dependency();
    constexpr int DK = 128, DV = 128, R = 8;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int col0 = (blockIdx.x * 4 + warp) * R;
    const int kh = head / (nv / nk);
    float* s0 = state + static_cast<size_t>(head) * DV * DK;
    float s[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float4 v = *reinterpret_cast<const float4*>(s0 + static_cast<size_t>(col0 + r) * DK + lane * 4);
        s[r][0] = v.x;
        s[r][1] = v.y;
        s[r][2] = v.z;
        s[r][3] = v.w;
    }
    const float A = expf(a_log[head]);
    const float bias = dt_bias[head];
    for (int t = 0; t < rows; ++t) {
        const __nv_bfloat16* row = qkv + static_cast<size_t>(t) * ld_qkv;
        float q[4], k[4], v[R];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = bf(row[kh * DK + lane * 4 + i]);
            k[i] = bf(row[nk * DK + kh * DK + lane * 4 + i]);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = bf(row[2 * nk * DK + head * DV + col0 + r]);
        const float bb = bf(ba[static_cast<size_t>(t) * ld_ba + head]);
        const float aa = bf(ba[static_cast<size_t>(t) * ld_ba + nv + head]);
        const float x = aa + bias;
        const float sp = x <= 20.0f ? logf(1.0f + expf(x)) : x;
        const float decay = expf(-A * sp);
        const float beta = round_bf(1.0f / (1.0f + expf(-bb)));
        float qq = 0.f, kk = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            qq += q[i] * q[i];
            kk += k[i] * k[i];
        }
        qq = warp_sum(qq);
        kk = warp_sum(kk);
        const float qn = sqrtf(qq + 1e-6f), kn = sqrtf(kk + 1e-6f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = (q[i] / qn) * scale;
            k[i] = k[i] / kn;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float mem = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[r][i] *= decay;
                mem += s[r][i] * k[i];
            }
            mem = warp_sum(mem);
            const float delta = (v[r] - mem) * beta;
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[r][i] += k[i] * delta;
                o += s[r][i] * q[i];
            }
            o = warp_sum(o);
            if (lane == r) y[(static_cast<size_t>(t) * nv + head) * DV + col0 + r] = tobf(o);
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
        *reinterpret_cast<float4*>(s0 + static_cast<size_t>(col0 + r) * DK + lane * 4) =
            make_float4(s[r][0], s[r][1], s[r][2], s[r][3]);
}

#include "drama_gdn_chunk.cuh"

// Full-attention prologue (fused_qk_gemma_rmsnorm_rope_gate semantics): per head of 256, GemmaRMSNorm
// bf16(v * rsqrt(mean(v^2) + eps) * (1 + w)) (the reference rounds the norm to bf16 before RoPE), then NeoX
// RoPE in fp32 on the leading 64 dims (pairs d, d + 32) with fp32 cos/sin of pos * inv_freq[d % 32] (SGLang
// keeps the cos/sin cache in fp32 on CUDA) and a second bf16 rounding at the store; the other dims round once.
// One head in a warp, a lane owning dims lane * 8 .. lane * 8 + 7; o is the value before the store's rounding.
__device__ __forceinline__ void qk_norm_rope(const __nv_bfloat16* src, const __nv_bfloat16* w, const float* inv_freq,
                                             int pos, float eps, int lane, float (&o)[8]) {
    constexpr int D = 256, ROT = 64, HALF = 32;
    float v[8];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = bf(src[lane * 8 + i]);
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float rcp = rsqrtf(ss / static_cast<float>(D) + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = round_bf(v[i] * rcp * (1.0f + bf(w[lane * 8 + i])));
    // lanes 0..3 hold dims 0..31, lanes 4..7 dims 32..63: partner lane ^ 4
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float other = __shfl_xor_sync(0xffffffffu, v[i], 4);
        const int d = lane * 8 + i;
        if (d < ROT) {
            const float ang = static_cast<float>(pos) * inv_freq[d % HALF];
            const float c = cosf(ang), s = sinf(ang);
            o[i] = d < HALF ? v[i] * c - other * s : v[i] * c + other * s;
        } else {
            o[i] = v[i];
        }
    }
}

// qkv rows: [per q head: q (256) | gate (256)] x H, then k (HK x 256), then v (HK x 256), ld_qkv apart.
// q goes to q_out [rows][H][256]; k and v to the caches [context][HK][256] at pos0 + row.
// Grid: (rows, H + HK); block 32 (a lane owns 8 dims).
extern "C" __global__ void __launch_bounds__(32) drama_attn_prep(const __nv_bfloat16* qkv, int ld_qkv,
                                                                const __nv_bfloat16* qn_w, const __nv_bfloat16* kn_w,
                                                                const float* inv_freq, __nv_bfloat16* q_out,
                                                                __nv_bfloat16* kc, __nv_bfloat16* vc, int pos0, int H,
                                                                int HK, float eps) {
    grid_dependency();
    constexpr int D = 256;
    const int r = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const int pos = pos0 + r;
    const __nv_bfloat16* row = qkv + static_cast<size_t>(r) * ld_qkv;
    const bool is_q = h < H;
    const int kh = h - H;
    const __nv_bfloat16* src = is_q ? row + h * 2 * D : row + H * 2 * D + kh * D;
    float o[8];
    qk_norm_rope(src, is_q ? qn_w : kn_w, inv_freq, pos, eps, lane, o);
    __nv_bfloat16* dst = is_q ? q_out + (static_cast<size_t>(r) * H + h) * D
                              : kc + (static_cast<size_t>(pos) * HK + kh) * D;
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[lane * 8 + i] = tobf(o[i]);
    if (!is_q) {
        const __nv_bfloat16* vs = row + H * 2 * D + HK * D + kh * D;
        __nv_bfloat16* vd = vc + (static_cast<size_t>(pos) * HK + kh) * D;
#pragma unroll
        for (int i = 0; i < 8; ++i) vd[lane * 8 + i] = vs[lane * 8 + i];
    }
}

// Causal GQA attention and output gate as Triton compiles the reference's extend kernel (BLOCK_N 64 at head_dim
// 256 on sm_12x) and fused_sigmoid_mul: 64-key tiles from key 0, one running max m per tile, p = tl.exp(s - m),
// l = fma(l, c, sum p), o = fma(o, c, sum bf16(p) v) with c = tl.exp(m_old - m); out = bf16(bf16(o / l) * sigmoid);
// tl.exp = ex2.approx(x * log2 e), / = div.full. Tiles depend only on key positions, so a row's bits ignore its
// call's rows; decode rows share this arithmetic (the reference's split-KV decode kernel is not reproduced).
// Row r (position pos0 + r) sees keys 0..pos0 + r; q head h reads kv head h / (H / HK); gate at qkv + h * 512 + 256.
// With qn_w (one row only) the launch also does drama_attn_prep's work: each block runs the prologue for its q head
// and, in the warp that owns key pos0, for the row's k head, whose k and v it takes from registers rather than the
// cache; the first q head of each kv head appends them to the caches. q is then unused.
// Grid: (rows, H); block 256: warp w holds keys w + 8u (u < 8) of every tile, a lane 8 dims. out: bf16 [rows][H * 256].
extern "C" __global__ void __launch_bounds__(256) drama_attention(const __nv_bfloat16* q, __nv_bfloat16* kc,
                                                                 __nv_bfloat16* vc, const __nv_bfloat16* qkv,
                                                                 int ld_qkv, __nv_bfloat16* out, int pos0, int H,
                                                                 int HK, float scale, const __nv_bfloat16* qn_w,
                                                                 const __nv_bfloat16* kn_w, const float* inv_freq,
                                                                 float eps) {
    using drama_gdn_chunk::div_full;
    using drama_gdn_chunk::tl_exp;
    grid_dependency();
    constexpr int D = 256, NW = 8, U = 8, TILE = NW * U;
    __shared__ float sm_max[2][NW], sm_l[NW];
    __shared__ float sm_o[NW][D];
    const int r = blockIdx.x, h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kh = h / (H / HK);
    const int last = pos0 + r;
    const bool prologue = qn_w != nullptr;
    const __nv_bfloat16* row = qkv + static_cast<size_t>(r) * ld_qkv;
    // A warp's cached keys of the next tile load while this tile is scored and summed. Loads past `cached` reload key
    // `cached` instead of branching; with no cached key (the fused row at position 0) nothing is loaded, since row 0
    // is being appended.
    const int cached = prologue ? last - 1 : last;
    const __nv_bfloat16* kp = kc + kh * D + lane * 8;
    const __nv_bfloat16* vp = vc + kh * D + lane * 8;
    uint4 kw[U]{}, vw[U]{};
    if (cached >= 0) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const size_t at = static_cast<size_t>(min(warp + u * NW, cached)) * HK * D;
            kw[u] = *reinterpret_cast<const uint4*>(kp + at);
            vw[u] = *reinterpret_cast<const uint4*>(vp + at);
        }
    }
    float qv[8], o[8];
    if (prologue) {
        qk_norm_rope(row + h * 2 * D, qn_w, inv_freq, last, eps, lane, qv);
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = round_bf(qv[i]);
    } else {
        const __nv_bfloat16* qr = q + (static_cast<size_t>(r) * H + h) * D + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) qv[i] = bf(qr[i]);
    }
    // The row's own key is key `last`, held by warp last % NW; scored now, while the first loads are in flight.
    const bool own = prologue && warp == last % NW;
    float own_score = 0.f;
    uint4 own_value{};
    if (own) {
        float kv[8];
        qk_norm_rope(row + H * 2 * D + kh * D, kn_w, inv_freq, last, eps, lane, kv);
        own_value = *reinterpret_cast<const uint4*>(row + H * 2 * D + HK * D + kh * D + lane * 8);
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) dot += qv[i] * round_bf(kv[i]);
        own_score = warp_sum(dot) * scale;
        if (h % (H / HK) == 0) {
            const size_t at = (static_cast<size_t>(last) * HK + kh) * D + lane * 8;
            store8(kc + at, kv);
            *reinterpret_cast<uint4*>(vc + at) = own_value;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = 0.f;
    float m = -INFINITY, l = 0.f;
    // Every warp sees every tile, so the __syncthreads in the loop is reached uniformly.
    for (int t = 0; t * TILE <= last; ++t) {
        const int j0 = t * TILE + warp;
        float sc[U];
        uint4 vb[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const __nv_bfloat16* k = reinterpret_cast<const __nv_bfloat16*>(&kw[u]);
            float dot = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) dot += qv[i] * bf(k[i]);
            sc[u] = dot;
            vb[u] = vw[u];
            if (cached >= 0) {
                const size_t at = static_cast<size_t>(min(j0 + TILE + u * NW, cached)) * HK * D;
                kw[u] = *reinterpret_cast<const uint4*>(kp + at);
                vw[u] = *reinterpret_cast<const uint4*>(vp + at);
            }
        }
        float tile_max = -INFINITY;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = j0 + u * NW;
            sc[u] = warp_sum(sc[u]) * scale;
            if (own && j == last) {
                sc[u] = own_score;
                vb[u] = own_value;
            } else if (j > cached) {
                sc[u] = -INFINITY;
            }
            tile_max = fmaxf(tile_max, sc[u]);
        }
        // Double-buffered: a warp writing tile t + 2's maxima has passed tile t + 1's barrier, after all reads of t's.
        if (lane == 0) sm_max[t & 1][warp] = tile_max;
        __syncthreads();
        float mn = m;
#pragma unroll
        for (int w = 0; w < NW; ++w) mn = fmaxf(mn, sm_max[t & 1][w]);
        const float corr = tl_exp(m - mn);  // m = -inf on the first tile: 0
        float sum = 0.f, pv[8] = {};
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const float p = tl_exp(sc[u] - mn);  // a masked key: 0
            sum += p;
            const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&vb[u]);
#pragma unroll
            for (int i = 0; i < 8; ++i) pv[i] += round_bf(p) * bf(v[i]);
        }
        l = __fmaf_rn(l, corr, sum);
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __fmaf_rn(o[i], corr, pv[i]);
        m = mn;
    }
    // All warps rescaled by the same maxima: their shares add without a merge rescale.
    if (lane == 0) sm_l[warp] = l;
#pragma unroll
    for (int i = 0; i < 8; ++i) sm_o[warp][lane * 8 + i] = o[i];
    __syncthreads();
    const int d = threadIdx.x;  // 256 threads, one dim each
    float L = 0.f, acc = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        L += sm_l[w];
        acc += sm_o[w][d];
    }
    const float attn = round_bf(div_full(acc, L));
    const float gate = bf(qkv[static_cast<size_t>(r) * ld_qkv + h * 2 * D + D + d]);
    const float sigmoid_gate = div_full(1.0f, __fadd_rn(tl_exp(__fsub_rn(0.0f, gate)), 1.0f));
    out[(static_cast<size_t>(r) * H + h) * D + d] = tobf(__fmul_rn(attn, sigmoid_gate));
}

// Router: probabilities = softmax(float(logits)) over the experts keep allows (null: all 256; a masked logit is -inf,
// as in a checkpoint pruned to the kept), the 8 largest (lower id on ties), renormalized to sum 1 (norm_topk_prob).
// ids/w [rows][8] by descending probability, bf16 logits; a slot table (compact) adds picks' held slots. Grid: rows.
extern "C" __global__ void __launch_bounds__(256) drama_topk(const __nv_bfloat16* logits, int ld, int* ids, float* wts,
                                                            int top, const unsigned* keep, const int* slot, int* slots) {
    grid_dependency();
    __shared__ float scratch[32];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ float picked_p[8];
    __shared__ int picked_i[8];
    const int r = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const bool kept = keep == nullptr || (keep[warp] >> lane & 1u);
    const float x = kept ? bf(logits[static_cast<size_t>(r) * ld + e]) : -INFINITY;
    float mx = warp_max(x);
    if (lane == 0) scratch[warp] = mx;
    __syncthreads();
    mx = lane < 8 ? scratch[lane] : -INFINITY;
    mx = warp_max(mx);
    __syncthreads();
    const float ex = expf(x - mx);
    const float sum = block_sum(ex, scratch, 8);
    float p = ex / sum;
    // a masked expert is never picked, even where a kept one's probability underflows to the same 0
    bool taken = !kept;
    for (int k = 0; k < top; ++k) {
        float cv = taken ? -1.0f : p;
        int ci = e;
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, cv, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ci, m);
            if (ov > cv || (ov == cv && oi < ci)) {
                cv = ov;
                ci = oi;
            }
        }
        if (lane == 0) {
            bv[warp] = cv;
            bi[warp] = ci;
        }
        __syncthreads();
        if (e == 0) {
            float best = bv[0];
            int at = bi[0];
            for (int w = 1; w < 8; ++w)
                if (bv[w] > best || (bv[w] == best && bi[w] < at)) {
                    best = bv[w];
                    at = bi[w];
                }
            picked_p[k] = best;
            picked_i[k] = at;
        }
        __syncthreads();
        if (e == picked_i[k]) taken = true;
    }
    if (e == 0) {
        float s = 0.f;
        for (int k = 0; k < top; ++k) s += picked_p[k];
        for (int k = 0; k < top; ++k) {
            ids[r * top + k] = picked_i[k];
            if (slot != nullptr) slots[r * top + k] = slot[picked_i[k]];
            wts[r * top + k] = picked_p[k] / s;
        }
    }
}

// Dynamic group-128 activation quantization: s = max(amax, 1e-10) * (1 / 448).
// The pinned per_token_group_quant_fp8 JIT uses --use_fast_math: its multiplier is the approximate 448 / amax,
// not 1 / the rounded scale. Keep that arithmetic local with __fdividef; other kernels retain precise division.
// q = e4m3(clamp(x * inv, -448, 448)) round-to-nearest-even. x: bf16 rows ld_x apart; q: [rows][K] bytes;
// s: fp32 [rows][K / 128]. Grid: (rows, K / 512); block 128 (a warp a group, a lane 4 values).
extern "C" __global__ void __launch_bounds__(128) drama_act_quant(const __nv_bfloat16* x, int ld_x, uint8_t* q,
                                                                 float* s, int K) {
    grid_dependency();
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.y * 4 + warp;
    if (group * 128 >= K) return;
    const __nv_bfloat16* xr = x + static_cast<size_t>(r) * ld_x + group * 128 + lane * 4;
    float v[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = bf(xr[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(warp_max(amax), 1e-10f);
    const float sc = amax * (1.0f / 448.0f);
    const float inv = __fdividef(448.0f, amax);
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= e4m3(v[i], inv) << (8 * i);
    *reinterpret_cast<uint32_t*>(q + static_cast<size_t>(r) * K + group * 128 + lane * 4) = packed;
    if (lane == 0) s[static_cast<size_t>(r) * (K / 128) + group] = sc;
}

// SwiGLU (sgl_kernel silu_and_mul): h = bf16(silu(float(gate)) * float(up)), gate | up halves of I each in a row.
// Rows below `routed` are the routed experts' rows of gu (ld_gu apart), stored as h's FP8 quantization as
// drama_act_quant: hq [routed][I] bytes, hs fp32 [routed][I / 128]. The rest are the shared expert's rows of sgu
// (ld_sgu apart), stored as bf16 h_out [rows - routed][I], so both experts take one launch.
// Grid: (rows, I / 512); block 128.
extern "C" __global__ void __launch_bounds__(128) drama_silu_mul(const __nv_bfloat16* gu, int ld_gu, int routed,
                                                                uint8_t* hq, float* hs, const __nv_bfloat16* sgu,
                                                                int ld_sgu, __nv_bfloat16* h_out, int I) {
    grid_dependency();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int group = blockIdx.y * 4 + warp;
    if (group * 128 >= I) return;
    const bool quant = blockIdx.x < routed;
    const int r = quant ? blockIdx.x : blockIdx.x - routed;
    const int c0 = group * 128 + lane * 4;
    const __nv_bfloat16* row = quant ? gu + static_cast<size_t>(r) * ld_gu : sgu + static_cast<size_t>(r) * ld_sgu;
    float h[4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = round_bf(silu(bf(row[c0 + i])) * bf(row[I + c0 + i]));
        amax = fmaxf(amax, fabsf(h[i]));
    }
    if (!quant) {
#pragma unroll
        for (int i = 0; i < 4; ++i) h_out[static_cast<size_t>(r) * I + c0 + i] = tobf(h[i]);
        return;
    }
    amax = fmaxf(warp_max(amax), 1e-10f);
    const float sc = amax * (1.0f / 448.0f);
    const float inv = __fdividef(448.0f, amax);
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= e4m3(h[i], inv) << (8 * i);
    *reinterpret_cast<uint32_t*>(hq + static_cast<size_t>(r) * I + c0) = packed;
    if (lane == 0) hs[static_cast<size_t>(r) * (I / 128) + group] = sc;
}

// MoE output: routed = bf16(sum over the top slots of float(down[row * top + s])) (moe_sum_reduce, scaling 1),
// then fused_gate_sigmoid_mul_add: out = bf16(sigmoid(sum_i float(x[i]) float(wg[i])) * float(shared) + routed).
// x: the MoE input row (post-attention norm); wg: shared_expert_gate.weight [width]. width % 8 == 0.
// Grid: rows; block 256 (a thread 8 contiguous outputs).
extern "C" __global__ void __launch_bounds__(256) drama_moe_combine(const __nv_bfloat16* down, int top,
                                                                   const __nv_bfloat16* x, const __nv_bfloat16* wg,
                                                                   const __nv_bfloat16* shared, __nv_bfloat16* out,
                                                                   int width) {
    grid_dependency();
    __shared__ float scratch[32];
    const size_t r = blockIdx.x;
    float part = 0.f;
    for (int i = threadIdx.x; i < width; i += 256) part += bf(x[r * width + i]) * bf(wg[i]);
    const float gate = sigmoid(block_sum(part, scratch, 8));
    for (int c = 8 * threadIdx.x; c < width; c += 8 * 256) {
        float acc[8], value[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll 8
        for (int s = 0; s < top; ++s) {
            load8(down + (r * top + s) * width + c, value);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += value[i];
        }
        load8(shared + r * width + c, value);
#pragma unroll
        for (int i = 0; i < 8; ++i) value[i] = gate * value[i] + round_bf(acc[i]);
        store8(out + r * width + c, value);
    }
}

// drama_moe_combine on unweighted fp32 rows: each pair's bf16(w * down) as FP8 MODE 1 stores it, then the same sums.
extern "C" __global__ void __launch_bounds__(256) drama_moe_combine_f32(const float* down, const float* w, int top,
                                                                       const __nv_bfloat16* x, const __nv_bfloat16* wg,
                                                                       const __nv_bfloat16* shared, __nv_bfloat16* out,
                                                                       int width) {
    grid_dependency();
    __shared__ float scratch[32];
    const size_t r = blockIdx.x;
    float part = 0.f;
    for (int i = threadIdx.x; i < width; i += 256) part += bf(x[r * width + i]) * bf(wg[i]);
    const float gate = sigmoid(block_sum(part, scratch, 8));
    for (int c = 8 * threadIdx.x; c < width; c += 8 * 256) {
        float acc[8], value[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.f;
        for (int s = 0; s < top; ++s) {
            const float ws = w[r * top + s];
            const float4* d = reinterpret_cast<const float4*>(down + (r * top + s) * width + c);
            const float4 lo = d[0], hi = d[1];
            const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += round_bf(ws * v[i]);
        }
        load8(shared + r * width + c, value);
#pragma unroll
        for (int i = 0; i < 8; ++i) value[i] = gate * value[i] + round_bf(acc[i]);
        store8(out + r * width + c, value);
    }
}
