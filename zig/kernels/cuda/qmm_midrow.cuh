// Ours: qmm_midrow's tiles; weights rounded once to bf16, one fp32 MMA chain over K each: qmm_prefill's bits.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "qmm_frag.cuh"

namespace tf_qmm_midrow {

using namespace qmm_frag;

template <int BM, int BN, int WM, int WN, int STAGES>
struct Mid {
    static constexpr int GS = 64;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;  // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;   // n8 tiles a warp
    static constexpr int ROW = GS * 2;       // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;       // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;    // weights (256 bytes an n8 tile),
    static constexpr int S = BN * 2;         // scales and biases (bf16)
    static constexpr int STAGE = X + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
    static_assert(MT >= 1 && NT >= 1 && BM % (WM * 16) == 0 && BN % (WN * 8) == 0 && BN % 32 == 0, "tile");
};

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// REVERSE walks the K groups last to first: the planted control, which must miss prefill_kernel's bits.
template <int BM, int BN, int WM, int WN, int STAGES, bool REVERSE = false>
__device__ __forceinline__ void mid(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                                    const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                                    __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {
    using T = Mid<BM, BN, WM, WN, STAGES>;
    constexpr int GS = T::GS, MT = T::MT, NT = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int step) {
        const int g = REVERSE ? KG - 1 - step : step;
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int n8 = n0 / 8 + c / 16, off = c % 16;  // the chunk's n8 tile, then its 16-byte piece of 256
            const size_t tile = static_cast<size_t>(n8 / 8) * KG + g;
            if (n8 * 8 < npad)
                cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * (64 * GS / 2) + (n8 % 8) * 256 + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        uint32_t words[NT][GS / 32], sv[NT], bv[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);  // this lane's B-fragment column
            sv[j] = ps[col] * 0x10001u;                              // (s, s) and (b, b) as bf16 pairs
            bv[j] = ps[BN + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t b0 = fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                const uint32_t b1 = fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                } else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
        }
}

}  // namespace tf_qmm_midrow
