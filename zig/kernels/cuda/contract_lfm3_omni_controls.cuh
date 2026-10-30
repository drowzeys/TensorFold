// LFM3-Omni's planted controls: copies of lfm3_omni.cu's kernels, each with one fault its contract gate must catch.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

typedef __nv_bfloat16 om_bf;
__device__ __forceinline__ float om_f(om_bf x) { return __bfloat162float(x); }
__device__ __forceinline__ om_bf om_b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float om_rb(float x) { return om_f(om_b(x)); }
__device__ __forceinline__ float om_sum256(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < 8; ++i) t += red[i];
    __syncthreads();
    return t;
}
__device__ __forceinline__ long long om_gid() { return (long long)blockIdx.x * blockDim.x + threadIdx.x; }

// om_rmsnorm, mode 0 without x * rsqrt's own rounding.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_om_rmsnorm(om_bf* X, const om_bf* R, const om_bf* Wb, om_bf* Y, int D, int ld, int ldy,
                                                                   float eps, int mode, int ldw) {
    __shared__ float red[32];
    const om_bf* W = Wb + (long long)blockIdx.x * ldw;
    om_bf* x = X + (long long)blockIdx.x * ld;
    om_bf* y = Y + (long long)blockIdx.x * ldy;
    const om_bf* r = R ? R + (long long)blockIdx.x * ld : nullptr;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) {
        float v = om_f(x[i]);
        if (r) {
            v = om_rb(v + om_f(r[i]));
            x[i] = om_b(v);
        }
        ss += v * v;
    }
    const float inv = rsqrtf(om_sum256(ss, red) / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += 256) y[i] = om_b(om_f(W[i]) * (om_f(x[i]) * inv));
}

// om_head_norm without x * rsqrt's own rounding.
extern "C" __global__ void tf_ctl_om_head_norm(om_bf* X, const om_bf* W, int T, int H, int hd, int ld, float eps) {
    const int gw = (int)(om_gid() >> 5), lane = threadIdx.x & 31;
    if (gw >= T * H) return;
    om_bf* x = X + (long long)(gw / H) * ld + (gw % H) * hd;
    float ss = 0.f;
    for (int i = lane; i < hd; i += 32) ss += om_f(x[i]) * om_f(x[i]);
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss / (float)hd + eps);
    for (int i = lane; i < hd; i += 32) x[i] = om_b(om_f(W[i]) * (om_f(x[i]) * inv));
}

// om_rope's bf16 steps (mode 2) without the a * c product's rounding.
extern "C" __global__ void tf_ctl_om_rope(om_bf* X, const float* INV, int T, int H, int hd, int ld, int pos0, int mode, const int* POSP) {
    const int half = hd >> 1;
    const long long i = om_gid();
    if (i >= (long long)T * H * half) return;
    const int p = (int)(i % half), h = (int)((i / half) % H), t = (int)(i / ((long long)half * H));
    om_bf* x = X + (long long)t * ld + h * hd;
    const float fr = (float)(pos0 + t) * INV[p];
    const float c = om_rb(cosf(fr)), s = om_rb(sinf(fr));
    const float a = om_f(x[p]), z = om_f(x[p + half]);
    x[p] = om_b(a * c + om_rb(-z * s));
    x[p + half] = om_b(om_rb(z * c) + om_rb(a * s));
}

// om_copy_rows reading the first row for every row.
extern "C" __global__ void tf_ctl_om_copy_rows(const om_bf* S, om_bf* D, int T, int n, int lds, int ldd, const int* POSP) {
    const long long i = om_gid();
    if (i >= (long long)T * n) return;
    D[(i / n) * ldd + i % n] = S[i % n];
}

// om_pos advancing by twice the step.
extern "C" __global__ void tf_ctl_om_pos(int* P, int v, int set) {
    if (threadIdx.x == 0) *P = set ? v : *P + 2 * v;
}

// om_short_conv without B * x's rounding.
extern "C" __global__ void tf_ctl_om_short_conv(const om_bf* BCx, int ld, const om_bf* Wc, om_bf* STATE, om_bf* Y, int ldy, int T, int D, int K) {
    const int d = (int)om_gid();
    if (d >= D) return;
    float hist[8], w[8];
    for (int j = 0; j < K; ++j) w[j] = om_f(Wc[d * K + j]);
    for (int j = 0; j < K - 1; ++j) hist[j] = om_f(STATE[d * (K - 1) + j]);
    for (int t = 0; t < T; ++t) {
        const om_bf* row = BCx + (long long)t * ld;
        const float bx = om_f(row[d]) * om_f(row[2 * D + d]);
        float acc = 0.f;
        for (int j = 0; j < K - 1; ++j) acc += w[j] * hist[j];
        acc += w[K - 1] * bx;
        for (int j = 0; j + 1 < K - 1; ++j) hist[j] = hist[j + 1];
        if (K > 1) hist[K - 2] = bx;
        Y[(long long)t * ldy + d] = om_b(om_f(row[D + d]) * om_rb(acc));
    }
    for (int j = 0; j < K - 1; ++j) STATE[d * (K - 1) + j] = om_b(hist[j]);
}

// om_attention's sliding window one key short.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_om_attention(const om_bf* Q, int ldq, const om_bf* Kc, const om_bf* Vc, int ldkv, om_bf* O, int ldo,
                                                                     int H, int Hkv, int hd, int pos0, int kv_len, int window, int causal,
                                                                     float scale, const int* POSP) {
    extern __shared__ float sm[];
    float* q = sm;
    float* sc = sm + 128;
    const int t = blockIdx.x / H, h = blockIdx.x % H, kh = h / (H / Hkv);
    const int pos = pos0 + t;
    int hi = causal ? pos + 1 : kv_len;
    const int lo = (causal && window > 0) ? max(0, pos + 2 - window) : 0;
    if (hi > kv_len) hi = kv_len;
    for (int i = threadIdx.x; i < hd; i += 128) q[i] = om_f(Q[(long long)t * ldq + h * hd + i]);
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int k = lo; k < hi; ++k) {
            float s = 0.f;
            for (int i = 0; i < hd; ++i) s += q[i] * om_f(Kc[(long long)k * ldkv + kh * hd + i]);
            sc[k - lo] = s * scale;
            m = fmaxf(m, sc[k - lo]);
        }
        float l = 0.f;
        for (int k = lo; k < hi; ++k) {
            sc[k - lo] = __expf(sc[k - lo] - m);
            l += sc[k - lo];
        }
        sc[hi - lo] = 1.f / l;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += 128) {
        float acc = 0.f;
        for (int k = lo; k < hi; ++k) acc += sc[k - lo] * om_f(Vc[(long long)k * ldkv + kh * hd + i]);
        O[(long long)t * ldo + h * hd + i] = om_b(acc * sc[hi - lo]);
    }
}

// om_gate without the gate's own rounding.
extern "C" __global__ void tf_ctl_om_gate(om_bf* O, int ldo, const om_bf* G, int ldg, int T, int H, int hd) {
    const long long i = om_gid();
    if (i >= (long long)T * H * hd) return;
    const int t = (int)(i / ((long long)H * hd)), h = (int)((i / hd) % H), c = (int)(i % hd);
    const float g = 1.f / (1.f + expf(-om_f(G[(long long)t * ldg + h])));
    om_bf* o = O + (long long)t * ldo + h * hd + c;
    *o = om_b(om_f(*o) * g);
}

// om_swiglu without silu's own rounding.
extern "C" __global__ void tf_ctl_om_swiglu(const om_bf* AC, om_bf* Y, int T, int F) {
    const long long i = om_gid();
    if (i >= (long long)T * F) return;
    const long long t = i / F;
    const int c = (int)(i % F);
    const float a = om_f(AC[t * 2 * F + c]);
    Y[i] = om_b(a / (1.f + expf(-a)) * om_f(AC[t * 2 * F + F + c]));
}

// om_add with the residual dropped every eighth element.
extern "C" __global__ void tf_ctl_om_add(om_bf* Y, const om_bf* X, const om_bf* R, long long n) {
    const long long i = om_gid();
    if (i < n) Y[i] = om_b(om_f(X[i]) + (i % 8 == 7 ? 0.f : om_f(R[i])));
}

// om_bias_act with the bias sum left in fp32 before the activation.
extern "C" __global__ void tf_ctl_om_bias_act(const float* A, const om_bf* Bias, om_bf* Y, int rows, int cols, int per_row, int act) {
    const long long i = om_gid();
    if (i >= (long long)rows * cols) return;
    const int r = (int)(i / cols), c = (int)(i % cols);
    float v = A[i] + (Bias ? om_f(Bias[per_row ? r : c]) : 0.f);
    if (act == 1) v = v / (1.f + expf(-v));
    else if (act == 2) v = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    Y[i] = om_b(v);
}

// om_layernorm with its mean rounded to bf16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_om_layernorm(const om_bf* X, const om_bf* W, const om_bf* Bb, om_bf* Y, int D, float eps) {
    __shared__ float red[32];
    const om_bf* x = X + (long long)blockIdx.x * D;
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) s += om_f(x[i]);
    const float mean = om_rb(om_sum256(s, red) / (float)D);
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += 256) v += (om_f(x[i]) - mean) * (om_f(x[i]) - mean);
    const float rstd = rsqrtf(om_sum256(v, red) / (float)D + eps);
    for (int i = threadIdx.x; i < D; i += 256) Y[(long long)blockIdx.x * D + i] = om_b((om_f(x[i]) - mean) * rstd * om_f(W[i]) + om_f(Bb[i]));
}

// om_embed writing zeros where a placeholder row must be left alone.
extern "C" __global__ void tf_ctl_om_embed(const om_bf* TABLE, const int* IDS, om_bf* Y, int T, int D) {
    const long long i = om_gid();
    if (i >= (long long)T * D) return;
    const int id = IDS[i / D];
    Y[i] = id >= 0 ? TABLE[(long long)id * D + i % D] : om_b(0.f);
}

// om_embed_codes summed in bf16.
extern "C" __global__ void tf_ctl_om_embed_codes(const om_bf* TABLE, const int* CODES, om_bf* Y, int T, int D, int NQ, int size) {
    const long long i = om_gid();
    if (i >= (long long)T * D) return;
    const int t = (int)(i / D), c = (int)(i % D);
    float s = 0.f;
    for (int q = 0; q < NQ; ++q) s = om_rb(s + om_f(TABLE[((long long)CODES[t * NQ + q] + (long long)q * size) * D + c]));
    Y[i] = om_b(s);
}

// om_depth_in with every slot's code embedded, masked or not.
extern "C" __global__ void tf_ctl_om_depth_in(const om_bf* DIN, const om_bf* MASK, const om_bf* TABLES, const int* CODES, const int* MASKED, om_bf* X,
                                              int NQ, int D, int size) {
    const int i = (int)om_gid();
    if (i >= NQ * D) return;
    const int q = i / D, c = i % D;
    X[i] = om_b(om_f(DIN[i]) + om_f(TABLES[((long long)q * size + CODES[q]) * D + c]));
}

// om_conv2d_s2 with SiLU taken before the bias sum's rounding.
extern "C" __global__ void tf_ctl_om_conv2d_s2(const om_bf* X, int CIN, int H, int W, const om_bf* Wt, const om_bf* Bias, om_bf* Y, int C, int H2, int W2,
                                               int silu) {
    const long long i = om_gid();
    if (i >= (long long)C * H2 * W2) return;
    const int c = (int)(i / ((long long)H2 * W2)), h = (int)((i / W2) % H2), w = (int)(i % W2);
    const om_bf* x = X + (long long)(CIN == 1 ? 0 : c) * H * W;
    float acc = 0.f;
    for (int kh = 0; kh < 3; ++kh)
        for (int kw = 0; kw < 3; ++kw) {
            const int ih = 2 * h - 1 + kh, iw = 2 * w - 1 + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) acc += om_f(Wt[c * 9 + kh * 3 + kw]) * om_f(x[(long long)ih * W + iw]);
        }
    float v = acc + om_f(Bias[c]);
    if (silu) v = v / (1.f + expf(-v));
    Y[i] = om_b(v);
}

// om_f32_bf16 truncating instead of rounding to nearest even.
extern "C" __global__ void tf_ctl_om_f32_bf16(const float* X, om_bf* Y, long long n) {
    const long long i = om_gid();
    if (i < n) Y[i] = __float2bfloat16_rz(X[i]);
}

// om_bf16_f32 one element late.
extern "C" __global__ void tf_ctl_om_bf16_f32(const om_bf* X, float* Y, long long n) {
    const long long i = om_gid();
    if (i < n) Y[i] = om_f(X[i > 0 ? i - 1 : 0]);
}

// om_fsq_decode with 8 levels in the second digit.
extern "C" __global__ void tf_ctl_om_fsq_decode(const int* CODES, om_bf* L, int n, int NQ) {
    const int i = (int)om_gid();
    if (i >= n * NQ) return;
    const int levels[4] = {8, 8, 6, 6};
    const int basis[4] = {1, 8, 56, 336};
    for (int j = 0; j < 4; ++j) {
        const int half = levels[j] / 2;
        L[(long long)i * 4 + j] = om_b((float)((CODES[i] / basis[j]) % levels[j] - half) / (float)half);
    }
}

// om_repeat_rows repeating each row one time too many.
extern "C" __global__ void tf_ctl_om_repeat_rows(const om_bf* X, om_bf* Y, int n, int D, int rep) {
    const long long i = om_gid();
    if (i >= (long long)n * rep * D) return;
    Y[i] = X[((i / D) / (rep + 1)) * D + i % D];
}

// om_polar with the magnitude rounded to bf16.
extern "C" __global__ void tf_ctl_om_polar(const om_bf* X, float* S, int T, int NB) {
    const long long i = om_gid();
    if (i >= (long long)T * NB) return;
    const long long t = i / NB;
    const int k = (int)(i % NB);
    const float m = om_rb(expf(om_f(X[t * 2 * NB + k]))), a = om_f(X[t * 2 * NB + NB + k]);
    S[t * 2 * NB + k] = m * cosf(a);
    S[t * 2 * NB + NB + k] = m * sinf(a);
}

// om_argmax tying to the higher id.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_om_argmax(const om_bf* X, int N, const int* SUP, int ns, int* OUT) {
    if (threadIdx.x != 0) return;
    float best = -INFINITY;
    int at = -1;
    for (int i = 0; i < N; ++i) {
        bool skip = false;
        for (int j = 0; j < ns; ++j) skip |= SUP[j] == i;
        const float v = om_f(X[i]);
        if (!skip && v >= best) {
            best = v;
            at = i;
        }
    }
    OUT[0] = at;
}

// om_depth_pick, greedy, tying to the higher id.
extern "C" __global__ void tf_ctl_om_depth_pick(const om_bf* LOG, const int* ORDER, int p0, const int* HIST, int hn, float penalty, float temp,
                                                unsigned long long seed, int frame, int* CODES, int* MASKED, int NQ, int size, const int* META) {
    if (threadIdx.x != 0) return;
    const int q = ORDER[p0 + blockIdx.x];
    float best = -INFINITY;
    int at = 0;
    for (int i = 0; i < size; ++i) {
        bool seen = false;
        for (int j = 0; j < hn; ++j) seen |= HIST[j * NQ + q] == i;
        float v = om_f(LOG[(long long)q * size + i]);
        if (seen) v = v < 0.f ? v * penalty : v / penalty;
        if (v >= best) {
            best = v;
            at = i;
        }
    }
    CODES[q] = at;
    MASKED[q] = 0;
}

// om_frame_end ending only the first code of a frame that ended.
extern "C" __global__ void tf_ctl_om_frame_end(int* CODES, int NQ, int eos) {
    if (threadIdx.x != 0) return;
    bool any = false;
    for (int q = 0; q < NQ; ++q) any |= CODES[q] == eos;
    if (any) CODES[0] = eos;
}

// om_hist_push into the window's first row.
extern "C" __global__ void tf_ctl_om_hist_push(const int* CODES, int* HIST, int slot, int NQ) {
    if (threadIdx.x < NQ) HIST[threadIdx.x] = CODES[threadIdx.x];
}

// om_mel_frames without the zeros past the clip.
extern "C" __global__ void tf_ctl_om_mel_frames(const float* X, int len, const float* WIN, float* FR, int nf, int nfft, int hop) {
    const long long i = om_gid();
    if (i >= (long long)nf * nfft) return;
    const int s = (int)(i / nfft) * hop + (int)(i % nfft);
    FR[i] = X[s < len ? s : len - 1] * WIN[i % nfft];
}

// om_power without the imaginary part.
extern "C" __global__ void tf_ctl_om_power(const float* S, float* P, int nf, int bins) {
    const long long i = om_gid();
    if (i >= (long long)nf * bins) return;
    const float re = S[(i / bins) * 2 * bins + i % bins];
    P[i] = re * re;
}

// om_logmel without its floor.
extern "C" __global__ void tf_ctl_om_logmel(const float* M, float* OUT, int nf, int mels, float floor_) {
    const long long i = om_gid();
    if (i >= (long long)nf * mels) return;
    OUT[(i % mels) * nf + i / mels] = (log10f(M[i]) + 4.f) / 4.f;
}

// om_short_conv_rows reading the state one tap off.
extern "C" __global__ void tf_ctl_om_short_conv_rows(const om_bf* BCx, int ld, const om_bf* Wc, const om_bf* STATE, om_bf* Y, int ldy, int T, int D, int K) {
    const long long i = om_gid();
    if (i >= (long long)T * D) return;
    const int t = (int)(i / D), d = (int)(i % D);
    float acc = 0.f;
    for (int j = 0; j < K; ++j) {
        const int src = t - (K - 1) + j;
        const om_bf* row = BCx + (long long)src * ld;
        const float v = src >= 0 ? om_rb(om_f(row[d]) * om_f(row[2 * D + d])) : om_f(STATE[d * (K - 1) + (K - 1 + src + 1) % (K - 1)]);
        acc += om_f(Wc[d * K + j]) * v;
    }
    Y[(long long)t * ldy + d] = om_b(om_f(BCx[(long long)t * ld + D + d]) * om_rb(acc));
}

// om_conv_state keeping the rows one late.
extern "C" __global__ void tf_ctl_om_conv_state(const om_bf* BCx, int ld, om_bf* STATE, int T, int D, int K) {
    const int d = (int)om_gid();
    if (d >= D) return;
    for (int j = 0; j < K - 1; ++j) {
        const int src = T - K + j;
        if (src >= 0) STATE[d * (K - 1) + j] = om_b(om_f(BCx[(long long)src * ld + d]) * om_f(BCx[(long long)src * ld + 2 * D + d]));
    }
}

// om_attention_tiles: every row stops one key before its own (a causal mask off by one).
extern "C" __global__ void __launch_bounds__(128) tf_ctl_om_attention_tiles(const om_bf* Q, int ldq, const om_bf* Kc, const om_bf* Vc, int ldkv, om_bf* O,
                                                                           int ldo, int T, int H, int Hkv, int hd, int pos0, int kv_len,
                                                                           int window, int causal, float scale) {
    const int h = blockIdx.y, kh = h / (H / Hkv);
    for (int r = blockIdx.x * 16; r < min(T, blockIdx.x * 16 + 16); ++r) {
        const int pos = pos0 + r;
        const int hi = causal ? max(pos, 1) : kv_len;
        const int lo = (causal && window > 0) ? max(0, pos + 1 - window) : 0;
        for (int i = threadIdx.x; i < hd; i += 128) {
            float m = -INFINITY;
            for (int k = lo; k < hi; ++k) {
                float s = 0.f;
                for (int c = 0; c < hd; ++c) s += om_f(Q[(long long)r * ldq + h * hd + c]) * om_f(Kc[(long long)k * ldkv + kh * hd + c]);
                m = fmaxf(m, s * scale);
            }
            float l = 0.f, acc = 0.f;
            for (int k = lo; k < hi; ++k) {
                float s = 0.f;
                for (int c = 0; c < hd; ++c) s += om_f(Q[(long long)r * ldq + h * hd + c]) * om_f(Kc[(long long)k * ldkv + kh * hd + c]);
                const float e = __expf(s * scale - m);
                l += e;
                acc += e * om_f(Vc[(long long)k * ldkv + kh * hd + i]);
            }
            O[(long long)r * ldo + h * hd + i] = om_b(acc / l);
        }
    }
}

// om_gemv_1 summed in fp16.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_om_gemv(const om_bf* W, const om_bf* X, int ldx, int K, int N, int T, om_bf* Y, int ldy, float* YF,
                                                                long long sw, long long sx, long long sy) {
    const int n = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (n >= N) return;
    for (int t = 0; t < T; ++t) {
        __half acc = __float2half(0.f);
        for (int k = lane; k < K; k += 32) acc = __hadd(acc, __float2half(om_f(W[(long long)n * K + k]) * om_f(X[(long long)t * ldx + k])));
        float v = __half2float(acc);
        for (int o = 16; o > 0; o >>= 1) v = __half2float(__float2half(v + __shfl_xor_sync(0xffffffffu, v, o)));
        if (lane == 0) {
            if (Y) Y[(long long)t * ldy + n] = om_b(v);
            if (YF) YF[(long long)t * N + n] = v;
        }
    }
}
