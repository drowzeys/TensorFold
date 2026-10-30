// LFM3-Omni's few-row products, device picks, log-mel and prompt rows; lfm3_omni.cu includes it after its helpers.
#pragma once

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

// Y[t, n] = W[n] . X[t] for T <= TM rows (K % 8 == 0), bf16 Y and / or fp32 YF; blockIdx.y picks a batch member.
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

// Codebook ORDER[p0 + block]'s code after HIST's repetition penalty: greedy (lower id on ties) or a hashed Gumbel max.
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

// om_short_conv a thread a (row, channel), taps before row 0 from STATE, which is left alone (om_conv_state follows).
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

// Prompt attention: 16 rows of a head a block over 64-key tiles, online softmax; masks as om_attention, hd % 32 == 0.
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
            // only the row's visible keys: a masked key's zero weight times a non-finite value is NaN
            const int pos = pos0 + r0 + r;
            const int k_lo = (causal && window > 0) ? max(0, pos - window + 1 - k0) : 0;
            const int k_hi = causal ? min(nk, pos - k0 + 1) : nk;
            for (int k = k_lo; k < k_hi; ++k) {
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
