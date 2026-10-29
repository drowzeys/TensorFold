// LFM3-Omni on CUDA: the glue around cuBLAS for its four Liquid stacks (speech encoder, backbone, depth decoder,
// Liquid codec decoder), the speech encoder's strided convolutions, the depth decoder's inputs and the codec's spectra.
// Rounding follows the bf16 reference op for op: every op that rounds to bf16 in the reference rounds here.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf;

__device__ __forceinline__ float f(bf x) { return __bfloat162float(x); }
__device__ __forceinline__ bf b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rb(float x) { return f(b(x)); }

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = 0.f;
    if (threadIdx.x < 32) {
        t = threadIdx.x < NT / 32 ? red[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

template <int NT>
__device__ __forceinline__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < NT / 32 ? red[threadIdx.x] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    return red[0];
}

// RMSNorm over rows of D (one block a row; rows `ld` apart). With R: X = X + R first (rounded, written back to X).
// mode 0: w * bf16(x * rsqrt) (the backbone's norm); mode 1: bf16(x * rsqrt * w) (the heads' norm, one rounding).
// W rows `ldw` apart (0: one weight for every row).
extern "C" __global__ void __launch_bounds__(256) om_rmsnorm(bf* __restrict__ X, const bf* __restrict__ R, const bf* __restrict__ Wb,
                                                            bf* __restrict__ Y, int D, int ld, int ldy, float eps, int mode, int ldw) {
    __shared__ float red[32];
    const bf* W = Wb + (long long)blockIdx.x * ldw;
    bf* x = X + (long long)blockIdx.x * ld;
    bf* y = Y + (long long)blockIdx.x * ldy;
    const bf* r = R ? R + (long long)blockIdx.x * ld : nullptr;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) {
        float v = f(x[i]);
        if (r) {
            v = rb(v + f(r[i]));
            x[i] = b(v);
        }
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum<256>(ss, red) / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += 256) {
        const float v = f(x[i]) * inv;
        y[i] = mode == 0 ? b(f(W[i]) * rb(v)) : b(v * f(W[i]));
    }
}

// Per-head RMSNorm (mode 0 rounding) in place: rows of `hd` at `X + row * ld + head * hd`, one warp a (row, head).
extern "C" __global__ void om_head_norm(bf* __restrict__ X, const bf* __restrict__ W, int T, int H, int hd, int ld, float eps) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (gw >= T * H) return;
    bf* x = X + (long long)(gw / H) * ld + (gw % H) * hd;
    float ss = 0.f;
    for (int i = lane; i < hd; i += 32) ss += f(x[i]) * f(x[i]);
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = lane; i < hd; i += 32) x[i] = b(f(W[i]) * rb(f(x[i]) * inv));
}

// RoPE in place on heads of `hd` (rotate-half pairs i, i + hd/2), row t at position pos0 + t; INV holds hd/2 inverse
// frequencies (zero past the rotated part). mode 0: fp32 cos/sin; 1: cos/sin rounded to bf16, fp32 products;
// 2: every product and the sum rounded (the speech encoder's bf16 arithmetic).
// POSP, if given, holds pos0 (a replayed graph's position).
extern "C" __global__ void om_rope(bf* __restrict__ X, const float* __restrict__ INV, int T, int H, int hd, int ld, int pos0, int mode,
                                   const int* __restrict__ POSP) {
    if (POSP) pos0 = *POSP;
    const int half = hd >> 1;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * H * half) return;
    const int p = (int)(i % half), h = (int)((i / half) % H), t = (int)(i / ((long long)half * H));
    bf* x = X + (long long)t * ld + h * hd;
    const float fr = (float)(pos0 + t) * INV[p];
    float c = cosf(fr), s = sinf(fr);
    if (mode != 0) {
        c = rb(c);
        s = rb(s);
    }
    const float a = f(x[p]), z = f(x[p + half]);
    if (mode == 2) {
        x[p] = b(rb(a * c) + rb(-z * s));
        x[p + half] = b(rb(z * c) + rb(a * s));
    } else {
        x[p] = b(a * c - z * s);
        x[p + half] = b(z * c + a * s);
    }
}

// Copies T rows of `n` bf16 from S (rows `lds` apart) to rows `ldd` apart in D, from row *POSP of D if given.
extern "C" __global__ void om_copy_rows(const bf* __restrict__ S, bf* __restrict__ D, int T, int n, int lds, int ldd,
                                        const int* __restrict__ POSP) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * n) return;
    const int t = (int)(i / n), c = (int)(i % n);
    const long long at = POSP ? (long long)*POSP : 0;
    D[(at + t) * ldd + c] = S[(long long)t * lds + c];
}

// *P = v (set 1) or *P += v (set 0): a stack's cache position on the device.
extern "C" __global__ void om_pos(int* __restrict__ P, int v, int set) {
    if (threadIdx.x == 0) *P = set ? v : *P + v;
}

// Liquid's gated short conv over T rows: BCx rows (B | C | x, `ld` apart), y = C * conv(B * x) with the K-1 rows of
// history in STATE [D, K-1] (updated to the last K-1 rows of B * x); one thread a channel.
extern "C" __global__ void om_short_conv(const bf* __restrict__ BCx, int ld, const bf* __restrict__ Wc, bf* __restrict__ STATE,
                                         bf* __restrict__ Y, int ldy, int T, int D, int K) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float hist[8];
    float w[8];
    for (int j = 0; j < K; ++j) w[j] = f(Wc[d * K + j]);
    for (int j = 0; j < K - 1; ++j) hist[j] = f(STATE[d * (K - 1) + j]);
    for (int t = 0; t < T; ++t) {
        const bf* row = BCx + (long long)t * ld;
        const float bx = rb(f(row[d]) * f(row[2 * D + d]));
        float acc = 0.f;
        for (int j = 0; j < K - 1; ++j) acc += w[j] * hist[j];
        acc += w[K - 1] * bx;
        for (int j = 0; j + 1 < K - 1; ++j) hist[j] = hist[j + 1];
        if (K > 1) hist[K - 2] = bx;
        Y[(long long)t * ldy + d] = b(f(row[D + d]) * rb(acc));
    }
    for (int j = 0; j < K - 1; ++j) STATE[d * (K - 1) + j] = b(hist[j]);
}

// Attention for T query rows at positions pos0 + t over the cache rows [lo, hi) (K, V rows `ldkv` apart, Hkv heads of
// hd): causal rows see keys up to their own position, `window` > 0 keeps the last `window` keys, causal 0 sees all.
// One block a (row, head); scores in shared memory (`span` floats). Output rows `ldo` apart.
extern "C" __global__ void __launch_bounds__(128) om_attention(const bf* __restrict__ Q, int ldq, const bf* __restrict__ Kc,
                                                              const bf* __restrict__ Vc, int ldkv, bf* __restrict__ O, int ldo,
                                                              int H, int Hkv, int hd, int pos0, int kv_len, int window,
                                                              int causal, float scale, const int* __restrict__ POSP) {
    extern __shared__ float sm[];
    __shared__ float red[32];
    if (POSP) {
        // a replayed graph's row: one row at the device's position, the cache through it
        pos0 = *POSP;
        kv_len = pos0 + 1;
    }
    float* q = sm;          // hd
    float* sc = sm + 128;   // keys
    const int t = blockIdx.x / H, h = blockIdx.x % H, kh = h / (H / Hkv);
    const int pos = pos0 + t;
    int hi = causal ? pos + 1 : kv_len;
    int lo = (causal && window > 0) ? max(0, pos + 1 - window) : 0;
    if (hi > kv_len) hi = kv_len;
    const bf* qr = Q + (long long)t * ldq + h * hd;
    for (int i = threadIdx.x; i < hd; i += 128) q[i] = f(qr[i]);
    __syncthreads();
    float m = -INFINITY;
    for (int k = lo + threadIdx.x; k < hi; k += 128) {
        const bf* kr = Kc + (long long)k * ldkv + kh * hd;
        float s = 0.f;
        for (int i = 0; i < hd; i += 2) {
            const __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(kr + i);
            s += q[i] * __low2float(kv) + q[i + 1] * __high2float(kv);
        }
        s *= scale;
        sc[k - lo] = s;
        m = fmaxf(m, s);
    }
    m = block_max<128>(m, red);
    float l = 0.f;
    for (int k = lo + threadIdx.x; k < hi; k += 128) {
        const float e = __expf(sc[k - lo] - m);
        sc[k - lo] = e;
        l += e;
    }
    l = block_sum<128>(l, red);
    const float inv = 1.f / l;
    for (int i = threadIdx.x; i < hd; i += 128) {
        float acc = 0.f;
        for (int k = lo; k < hi; ++k) acc += sc[k - lo] * f(Vc[(long long)k * ldkv + kh * hd + i]);
        O[(long long)t * ldo + h * hd + i] = b(acc * inv);
    }
}

// O *= bf16(sigmoid(G)) per head (LFM3's output gate); G rows `ldg` apart, O rows `ldo` apart.
extern "C" __global__ void om_gate(bf* __restrict__ O, int ldo, const bf* __restrict__ G, int ldg, int T, int H, int hd) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * H * hd) return;
    const int t = (int)(i / ((long long)H * hd)), h = (int)((i / hd) % H), c = (int)(i % hd);
    const float g = rb(1.f / (1.f + expf(-f(G[(long long)t * ldg + h]))));
    bf* o = O + (long long)t * ldo + h * hd + c;
    *o = b(f(*o) * g);
}

// y = bf16(bf16(silu(a)) * c) with a | c the halves of each row of AC (rows of 2F).
extern "C" __global__ void om_swiglu(const bf* __restrict__ AC, bf* __restrict__ Y, int T, int F) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * F) return;
    const long long t = i / F;
    const int c = (int)(i % F);
    const float a = f(AC[t * 2 * F + c]);
    Y[i] = b(rb(a / (1.f + expf(-a))) * f(AC[t * 2 * F + F + c]));
}

// Y = bf16(X + R) elementwise (n elements).
extern "C" __global__ void om_add(bf* __restrict__ Y, const bf* __restrict__ X, const bf* __restrict__ R, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = b(f(X[i]) + f(R[i]));
}

// Y[r, c] = act(bf16(A[r, c] + bias)) from an fp32 GEMM result; bias per column (per_row 0) or per row (1).
// act 0: none, 1: SiLU, 2: exact GELU (each rounded once more).
extern "C" __global__ void om_bias_act(const float* __restrict__ A, const bf* __restrict__ Bias, bf* __restrict__ Y, int rows, int cols,
                                       int per_row, int act) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    const int r = (int)(i / cols), c = (int)(i % cols);
    float v = A[i] + (Bias ? f(Bias[per_row ? r : c]) : 0.f);
    v = rb(v);
    if (act == 1) v = v / (1.f + expf(-v));
    else if (act == 2) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    Y[i] = b(v);
}

// LayerNorm with bias over rows of D in fp32, one rounding (one block a row).
extern "C" __global__ void __launch_bounds__(256) om_layernorm(const bf* __restrict__ X, const bf* __restrict__ W, const bf* __restrict__ Bb,
                                                              bf* __restrict__ Y, int D, float eps) {
    __shared__ float red[32];
    const bf* x = X + (long long)blockIdx.x * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) s += f(x[i]);
    const float mean = block_sum<256>(s, red) / (float)D;
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) {
        const float d = f(x[i]) - mean;
        v += d * d;
    }
    const float rstd = rsqrtf(block_sum<256>(v, red) / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += 256) Y[(long long)blockIdx.x * D + i] = b((f(x[i]) - mean) * rstd * f(W[i]) + f(Bb[i]));
}

// Rows of a table by id: Y[t] = TABLE[IDS[t]] (D wide); ids < 0 leave the row alone (filled elsewhere).
extern "C" __global__ void om_embed(const bf* __restrict__ TABLE, const int* __restrict__ IDS, bf* __restrict__ Y, int T, int D) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * D) return;
    const int t = (int)(i / D), c = (int)(i % D);
    const int id = IDS[t];
    if (id >= 0) Y[i] = TABLE[(long long)id * D + c];
}

// A speech frame's input row: the sum over the NQ codebooks of TABLE[code + q * size] (fp32 sum, one rounding).
// CODES [T, NQ].
extern "C" __global__ void om_embed_codes(const bf* __restrict__ TABLE, const int* __restrict__ CODES, bf* __restrict__ Y, int T, int D,
                                          int NQ, int size) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * D) return;
    const int t = (int)(i / D), c = (int)(i % D);
    float s = 0.f;
    for (int q = 0; q < NQ; ++q) s += f(TABLE[((long long)CODES[t * NQ + q] + (long long)q * size) * D + c]);
    Y[i] = b(s);
}

// The depth decoder's slots: X[q] = bf16(DIN[q] + (MASKED[q] ? MASK : TABLES[q][CODES[q]])) for NQ slots of D.
extern "C" __global__ void om_depth_in(const bf* __restrict__ DIN, const bf* __restrict__ MASK, const bf* __restrict__ TABLES,
                                       const int* __restrict__ CODES, const int* __restrict__ MASKED, bf* __restrict__ X, int NQ, int D,
                                       int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NQ * D) return;
    const int q = i / D, c = i % D;
    const float e = MASKED[q] ? f(MASK[c]) : f(TABLES[((long long)q * size + CODES[q]) * D + c]);
    X[i] = b(f(DIN[i]) + e);
}

// A stride-2 3x3 convolution (padding 1) over [C, H, W] planes, bias added and rounded; input channel 0 for every
// output channel when CIN is 1, else its own (depthwise). silu 1 applies SiLU after (rounded again).
extern "C" __global__ void om_conv2d_s2(const bf* __restrict__ X, int CIN, int H, int W, const bf* __restrict__ Wt, const bf* __restrict__ Bias,
                                        bf* __restrict__ Y, int C, int H2, int W2, int silu) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)C * H2 * W2) return;
    const int c = (int)(i / ((long long)H2 * W2)), h = (int)((i / W2) % H2), w = (int)(i % W2);
    const bf* x = X + (long long)(CIN == 1 ? 0 : c) * H * W;
    float acc = 0.f;
    for (int kh = 0; kh < 3; ++kh) {
        const int ih = 2 * h - 1 + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int iw = 2 * w - 1 + kw;
            if (iw < 0 || iw >= W) continue;
            acc += f(Wt[c * 9 + kh * 3 + kw]) * f(x[(long long)ih * W + iw]);
        }
    }
    float v = rb(acc + f(Bias[c]));
    if (silu) v = rb(v / (1.f + expf(-v)));
    Y[i] = b(v);
}

// fp32 mel features [128, T] to bf16 (the encoder's input cast).
extern "C" __global__ void om_f32_bf16(const float* __restrict__ X, bf* __restrict__ Y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = b(X[i]);
}

// bf16 to fp32 (logits for the host).
extern "C" __global__ void om_bf16_f32(const bf* __restrict__ X, float* __restrict__ Y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = f(X[i]);
}

// The codec's finite scalar quantizer read back: CODES [n, NQ] (frames) to latents L [n, NQ * 4] in bf16, levels
// 8 7 6 6.
extern "C" __global__ void om_fsq_decode(const int* __restrict__ CODES, bf* __restrict__ L, int n, int NQ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * NQ) return;
    const int t = i / NQ, q = i % NQ;
    const int levels[4] = {8, 7, 6, 6};
    const int basis[4] = {1, 8, 56, 336};
    const int idx = CODES[t * NQ + q];
    for (int j = 0; j < 4; ++j) {
        const int half = levels[j] / 2;
        const int digit = (idx / basis[j]) % levels[j] - half;
        L[(long long)t * NQ * 4 + q * 4 + j] = b((float)digit / (float)half);
    }
}

// Each of n rows of D repeated `rep` times (nearest-exact upsampling by an integer factor).
extern "C" __global__ void om_repeat_rows(const bf* __restrict__ X, bf* __restrict__ Y, int n, int D, int rep) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)n * rep * D) return;
    const long long r = i / D;
    Y[i] = X[(r / rep) * D + i % D];
}

// The codec head's rows (log-magnitude | phase, NB bins each) to a real | imaginary spectrum in fp32.
extern "C" __global__ void om_polar(const bf* __restrict__ X, float* __restrict__ S, int T, int NB) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * NB) return;
    const long long t = i / NB;
    const int k = (int)(i % NB);
    const float m = expf(f(X[t * 2 * NB + k])), a = f(X[t * 2 * NB + NB + k]);
    S[t * 2 * NB + k] = m * cosf(a);
    S[t * 2 * NB + NB + k] = m * sinf(a);
}

// Greedy text: the first index of the largest of N bf16 logits, ids in SUP (ns of them) excluded; one block.
extern "C" __global__ void __launch_bounds__(1024) om_argmax(const bf* __restrict__ X, int N, const int* __restrict__ SUP, int ns,
                                                             int* __restrict__ OUT) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < N; i += 1024) {
        bool skip = false;
        for (int j = 0; j < ns; ++j) skip |= SUP[j] == i;
        if (skip) continue;
        const float v = f(X[i]);
        if (v > best || (v == best && i < at)) {
            best = v;
            at = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, at, o);
        if (ov > best || (ov == best && oi < at)) {
            best = ov;
            at = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = best;
        bi[threadIdx.x >> 5] = at;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        best = bv[threadIdx.x];
        at = bi[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, at, o);
            if (ov > best || (ov == best && oi < at)) {
                best = ov;
                at = oi;
            }
        }
        if (threadIdx.x == 0) OUT[0] = at;
    }
}

// ---- few-row products: a warp a weight row, the T input rows staged in shared memory, the weights streamed once

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float dot8(const uint4 a, const uint4 b) {
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* y = reinterpret_cast<const __nv_bfloat162*>(&b);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p = __bfloat1622float2(x[i]), q = __bfloat1622float2(y[i]);
        s += p.x * q.x + p.y * q.y;
    }
    return s;
}

// Y[t, n] = sum_k W[n, k] X[t, k] for t < T <= TM: W bf16 [N, K] (K % 8 == 0), X rows `ldx` apart; bf16 out in Y
// (rows `ldy` apart) and / or fp32 out in YF (rows N apart). 8 rows a block of 256; blockIdx.y picks a product of a
// batch (W, X and Y `sw`, `sx`, `sy` elements apart).
template <int TM>
__device__ __forceinline__ void gemv_rows(const bf* __restrict__ W, const bf* __restrict__ X, int ldx, int K, int N, int T,
                                          bf* __restrict__ Y, int ldy, float* __restrict__ YF, long long sw, long long sx, long long sy) {
    extern __shared__ uint4 xs[];
    W += blockIdx.y * sw;
    X += blockIdx.y * sx;
    if (Y) Y += blockIdx.y * sy;
    const int chunks = K >> 3;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
    const uint4* row = reinterpret_cast<const uint4*>(W + (long long)min(n, N - 1) * K);
    constexpr int U = 4;
    uint4 w[U];
    int c = lane;
#pragma unroll
    for (int u = 0; u < U; ++u) w[u] = c + u * 32 < chunks ? ld_stream(row + c + u * 32) : make_uint4(0, 0, 0, 0);
    for (int i = threadIdx.x; i < T * chunks; i += blockDim.x) {
        const int t = i / chunks, j = i - t * chunks;
        xs[i] = reinterpret_cast<const uint4*>(X + (long long)t * ldx)[j];
    }
    __syncthreads();
    float acc[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) acc[t] = 0.f;
    while (true) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (c + u * 32 < chunks) {
#pragma unroll
                for (int t = 0; t < TM; ++t)
                    if (t < T) acc[t] += dot8(w[u], xs[t * chunks + c + u * 32]);
            }
        }
        c += U * 32;
        if (c >= chunks) break;
#pragma unroll
        for (int u = 0; u < U; ++u) w[u] = c + u * 32 < chunks ? ld_stream(row + c + u * 32) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && n < N && t < T) {
            if (Y) Y[(long long)t * ldy + n] = b(v);
            if (YF) YF[(long long)t * N + n] = v;
        }
    }
}

#define OM_GEMV(TM)                                                                                                         \
    extern "C" __global__ void __launch_bounds__(256) om_gemv_##TM(const bf* __restrict__ W, const bf* __restrict__ X, int ldx, int K, \
                                                                  int N, int T, bf* __restrict__ Y, int ldy, float* __restrict__ YF, \
                                                                  long long sw, long long sx, long long sy) {                       \
        gemv_rows<TM>(W, X, ldx, K, N, T, Y, ldy, YF, sw, sx, sy);                                                         \
    }
OM_GEMV(1)
OM_GEMV(4)
OM_GEMV(8)
OM_GEMV(12)
OM_GEMV(16)

// ---- the depth decoder's picks on the device

__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// One pass's picks: codebook q = ORDER[p0 + blockIdx.x] gets its code from LOG (bf16 logits [NQ, size]) after the
// repetition penalty over the codes it took in the hn frames of HIST ([hcap, NQ]); greedy at temp 0 (ties to the lower
// index), else a Gumbel-max draw (exactly a softmax(logits / temp) sample) from a hash of seed, frame, q and the code.
// Writes CODES[q] and clears MASKED[q]. Shared: `size` bytes.
extern "C" __global__ void __launch_bounds__(256) om_depth_pick(const bf* __restrict__ LOG, const int* __restrict__ ORDER, int p0,
                                                               const int* __restrict__ HIST, int hn, float penalty, float temp,
                                                               unsigned long long seed, int frame, int* __restrict__ CODES,
                                                               int* __restrict__ MASKED, int NQ, int size, const int* __restrict__ META) {
    extern __shared__ unsigned char seen[];
    if (META) {
        // a replayed graph's frame: the window's frames and the frame number on the device
        hn = META[0];
        frame = META[1];
    }
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int q = ORDER[p0 + blockIdx.x];
    for (int i = threadIdx.x; i < size; i += 256) seen[i] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < hn; j += 256) seen[HIST[j * NQ + q]] = 1;
    __syncthreads();
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < size; i += 256) {
        float v = f(LOG[(long long)q * size + i]);
        if (seen[i]) v = v < 0.f ? v * penalty : v / penalty;
        if (temp > 0.f) {
            const unsigned long long h = mix64(seed ^ mix64(((unsigned long long)frame << 32) ^ ((unsigned long long)q << 16) ^ (unsigned long long)i));
            const float u = ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
            v = v / temp - logf(-logf(u));
        }
        if (v > best || (v == best && i < at)) {
            best = v;
            at = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, at, o);
        if (ov > best || (ov == best && oi < at)) {
            best = ov;
            at = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = best;
        bi[threadIdx.x >> 5] = at;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        best = bv[0];
        at = bi[0];
        for (int w = 1; w < 8; ++w)
            if (bv[w] > best || (bv[w] == best && bi[w] < at)) {
                best = bv[w];
                at = bi[w];
            }
        CODES[q] = at;
        MASKED[q] = 0;
    }
}

// A finished frame: any end-of-speech code makes it the all-EOS frame (the backbone reads that one back).
extern "C" __global__ void om_frame_end(int* __restrict__ CODES, int NQ, int eos) {
    if (threadIdx.x != 0) return;
    bool any = false;
    for (int q = 0; q < NQ; ++q) any |= CODES[q] == eos;
    if (any)
        for (int q = 0; q < NQ; ++q) CODES[q] = eos;
}

// HIST row `slot` = CODES (the repetition penalty's window, a ring of frames).
extern "C" __global__ void om_hist_push(const int* __restrict__ CODES, int* __restrict__ HIST, int slot, int NQ) {
    if (threadIdx.x < NQ) HIST[slot * NQ + threadIdx.x] = CODES[threadIdx.x];
}

// ---- the log-mel front end on the device: frames of the padded clip, |DFT|^2 by GEMM, mel by GEMM, log

// FR[t, n] = x[t * hop + n] * window[n] (zero past the clip) for t < nf, n < nfft.
extern "C" __global__ void om_mel_frames(const float* __restrict__ X, int len, const float* __restrict__ WIN, float* __restrict__ FR,
                                         int nf, int nfft, int hop) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nf * nfft) return;
    const int t = (int)(i / nfft), n = (int)(i % nfft);
    const int s = t * hop + n;
    FR[i] = s < len ? X[s] * WIN[n] : 0.f;
}

// P[t, k] = re^2 + im^2 from S[t] = [re (bins) | im (bins)], as |X|^2 in fp32.
extern "C" __global__ void om_power(const float* __restrict__ S, float* __restrict__ P, int nf, int bins) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nf * bins) return;
    const long long t = i / bins;
    const int k = (int)(i % bins);
    const float re = S[t * 2 * bins + k], im = S[t * 2 * bins + bins + k];
    const float a = sqrtf(re * re + im * im);
    P[i] = a * a;
}

// OUT[m, t] = (max(log10(M[t, m]), floor) + 4) / 4: the mel rows [nf, mels] to features [mels, nf].
extern "C" __global__ void om_logmel(const float* __restrict__ M, float* __restrict__ OUT, int nf, int mels, float floor_) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nf * mels) return;
    const int t = (int)(i / mels), m = (int)(i % mels);
    OUT[(long long)m * nf + t] = (fmaxf(log10f(M[i]), floor_) + 4.f) / 4.f;
}

// ---- prompt rows: the short conv across rows and channels at once, and tiled attention

// As om_short_conv for many rows: one thread a (row, channel), the K taps read from B * x rows (or STATE before
// the first row); the state is left alone (om_conv_state follows).
extern "C" __global__ void om_short_conv_rows(const bf* __restrict__ BCx, int ld, const bf* __restrict__ Wc, const bf* __restrict__ STATE,
                                              bf* __restrict__ Y, int ldy, int T, int D, int K) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * D) return;
    const int t = (int)(i / D), d = (int)(i % D);
    float acc = 0.f;
    for (int j = 0; j < K; ++j) {
        const int src = t - (K - 1) + j;
        float v;
        if (src >= 0) {
            const bf* row = BCx + (long long)src * ld;
            v = rb(f(row[d]) * f(row[2 * D + d]));
        } else {
            v = f(STATE[d * (K - 1) + (K - 1 + src)]);
        }
        acc += f(Wc[d * K + j]) * v;
    }
    Y[(long long)t * ldy + d] = b(f(BCx[(long long)t * ld + D + d]) * rb(acc));
}

// The conv state after T rows: the last K-1 values of B * x (older ones shifted from the state).
extern "C" __global__ void om_conv_state(const bf* __restrict__ BCx, int ld, bf* __restrict__ STATE, int T, int D, int K) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    for (int j = 0; j < K - 1; ++j) {
        const int src = T - (K - 1) + j;
        if (src >= 0) {
            const bf* row = BCx + (long long)src * ld;
            STATE[d * (K - 1) + j] = b(f(row[d]) * f(row[2 * D + d]));
        } else {
            STATE[d * (K - 1) + j] = STATE[d * (K - 1) + j + T];
        }
    }
}

// Attention for many query rows: a block takes 16 rows of one head and walks the cache in tiles of 64 keys held in
// shared memory, with an online softmax per row (4 warps, 4 rows each). Masks as om_attention. hd <= 128, hd % 32 == 0.
#define AT_Q 16
#define AT_K 64
extern "C" __global__ void __launch_bounds__(128) om_attention_tiles(const bf* __restrict__ Q, int ldq, const bf* __restrict__ Kc,
                                                                    const bf* __restrict__ Vc, int ldkv, bf* __restrict__ O, int ldo,
                                                                    int T, int H, int Hkv, int hd, int pos0, int kv_len, int window,
                                                                    int causal, float scale) {
    extern __shared__ float sm[];
    const int hdp = hd + 2;  // padded rows (bf16) against bank conflicts
    float* qs = sm;                                    // [AT_Q][hd]
    float* ss = qs + AT_Q * hd;                        // [AT_Q][AT_K]
    bf* ks = reinterpret_cast<bf*>(ss + AT_Q * AT_K);  // [AT_K][hdp]
    bf* vs = ks + AT_K * hdp;                          // [AT_K][hdp]
    const int h = blockIdx.y, kh = h / (H / Hkv);
    const int r0 = blockIdx.x * AT_Q;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int dpl = hd >> 5;
    for (int i = threadIdx.x; i < AT_Q * hd; i += 128) {
        const int r = i / hd, c = i % hd;
        qs[i] = r0 + r < T ? f(Q[(long long)(r0 + r) * ldq + h * hd + c]) : 0.f;
    }
    const int last = min(T, r0 + AT_Q) - 1;
    int hi = causal ? pos0 + last + 1 : kv_len;
    hi = min(hi, kv_len);
    int lo = (causal && window > 0) ? max(0, pos0 + r0 + 1 - window) : 0;
    float m[4], l[4], acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        m[a] = -INFINITY;
        l[a] = 0.f;
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[a][c] = 0.f;
    }
    for (int k0 = lo; k0 < hi; k0 += AT_K) {
        const int nk = min(AT_K, hi - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < nk * (hd / 2); i += 128) {
            const int k = i / (hd / 2), c = (i % (hd / 2)) * 2;
            const long long g = (long long)(k0 + k) * ldkv + kh * hd + c;
            *reinterpret_cast<__nv_bfloat162*>(ks + k * hdp + c) = *reinterpret_cast<const __nv_bfloat162*>(Kc + g);
            *reinterpret_cast<__nv_bfloat162*>(vs + k * hdp + c) = *reinterpret_cast<const __nv_bfloat162*>(Vc + g);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < AT_Q * AT_K; i += 128) {
            const int r = i / AT_K, k = i % AT_K;
            const int pos = pos0 + r0 + r, kp = k0 + k;
            float s = -INFINITY;
            if (k < nk && r0 + r < T && (!causal || kp <= pos) && (!(causal && window > 0) || kp > pos - window)) {
                s = 0.f;
                const float* q = qs + r * hd;
                const bf* kr = ks + k * hdp;
                for (int c = 0; c < hd; c += 2) {
                    const float2 kv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(kr + c));
                    s += q[c] * kv.x + q[c + 1] * kv.y;
                }
                s *= scale;
            }
            ss[i] = s;
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int r = warp * 4 + a;
            float* sr = ss + r * AT_K;
            float mx = fmaxf(sr[lane], sr[lane + 32]);
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mn = fmaxf(m[a], mx);
            if (mn == -INFINITY) continue;  // nothing visible yet
            const float sc = __expf(m[a] - mn);
            const float p0 = __expf(sr[lane] - mn), p1 = __expf(sr[lane + 32] - mn);
            sr[lane] = p0;
            sr[lane + 32] = p1;
            float ps = p0 + p1;
            for (int o = 16; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
            l[a] = l[a] * sc + ps;
            m[a] = mn;
            __syncwarp();
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][c] *= sc;
            for (int k = 0; k < nk; ++k) {
                const float p = sr[k];
                const bf* vr = vs + k * hdp;
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    if (c < dpl) acc[a][c] += p * f(vr[lane + 32 * c]);
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const int r = r0 + warp * 4 + a;
        if (r >= T) continue;
        const float inv = 1.f / l[a];
#pragma unroll
        for (int c = 0; c < 4; ++c)
            if (c < dpl) O[(long long)r * ldo + h * hd + lane + 32 * c] = b(acc[a][c] * inv);
    }
}
