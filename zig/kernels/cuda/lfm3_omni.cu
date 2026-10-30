// LFM3-Omni's glue around cuBLAS, rounding to bf16 wherever the bf16 reference rounds; the few-row half is .cuh.

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

// RMSNorm a block a row, X = bf16(X + R) first; mode 0 bf16(w * bf16(x * rsqrt)), 1 bf16(x * rsqrt * w).
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

// Rotate-half RoPE in place at pos0 + t (or *POSP); mode 0 fp32 cos/sin, 1 bf16 cos/sin, 2 every product rounded.
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

// Liquid's gated short conv y = C * conv(B * x) over T rows of B | C | x, a thread a channel, STATE [D, K-1] updated.
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

// Attention a block a (row, head), scores in shared memory; causal to its position, `window` > 0 the last keys only.
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

// Y = act(bf16(A + bias)) from an fp32 product, bias per column or (per_row) per row; act 1 SiLU, 2 exact GELU.
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

// A speech frame's input row: the fp32 sum over CODES [T, NQ] of TABLE[code + q * size], rounded once.
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

// Stride-2 3x3 conv (padding 1, bias, rounded) over [C, H, W]; CIN 1 reads plane 0, else depthwise; then SiLU.
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

// The codec's finite scalar quantizer read back: CODES [n, NQ] to latents [n, NQ * 4] in bf16, levels 8 7 6 6.
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

#include "lfm3_omni_rows.cuh"
