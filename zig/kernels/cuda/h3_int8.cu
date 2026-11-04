// MiniMax H3 / FastH3's transformer blocks on CUDA tensor cores in int8: the Metal family's arithmetic
// (zig/kernels/metal/h3/h3.metal) with the products on mma.sync m16n8k32. Rows are the packed
// [text | keyframe | audio | video] sequence; a block's projections are int8 with a scale per output channel and its
// activations int8 with a scale per row.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace tf_h3 {

__device__ __forceinline__ uint32_t smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem(p)));
}
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_u8s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Block order: bands of `group` row tiles; inside a band one column tile's row tiles run next to each other.
__device__ __forceinline__ int2 tile_of(int b, int rows_t, int cols_t, int group) {
    const int band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2(first + in_band % height, in_band / height);
}

enum { plain = 0, swiglu = 1, wide = 2 };
constexpr int wide_group = 1024;

// A CTA owns BM rows by BN outputs; K arrives in slices of KC bytes (128 or 64), two stages deep. WM x WN warps.
// An SM has about 99 KiB of shared memory: a 32 KiB CTA lets three share it and overlap each other's waits.
// plain:  Y[m, n] = D[m, n] XS[m] WS[n]                                   Y: (M, N) bf16
// swiglu: W's rows alternate value, gate; Y[m, j] = silu(g) v             Y: (M, N / 2) bf16
// wide:   X carries a scale per (row, 1024 inputs): XS: (MP, K / 1024)    Y: (M, N) bf16
// X: (MP, K) int8 with MP a multiple of BM, rows from M zero. W: (N, K) int8, N a multiple of BN, K of KC.
template <int MODE, int BM, int BN, int WM, int WN, int KC = 128>
__device__ __forceinline__ void gemm_i8(
        const int8_t* __restrict__ X, const float* __restrict__ XS, const int8_t* __restrict__ W,
        const float* __restrict__ WS, __nv_bfloat16* __restrict__ Y, int M, int N, int K, int group) {
    constexpr int CH = KC / 16, THREADS = WM * WN * 32;
    constexpr int MT = BM / WM / 16, NT = BN / WN / 8;
    constexpr int XB = BM * KC, STAGE = (BM + BN) * KC;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const int rows_t = (M + BM - 1) / BM, cols_t = N / BN;
    const int2 at = tile_of(blockIdx.x, rows_t, cols_t, group);
    const int m0 = at.x * BM, n0 = at.y * BN;

    // where a row's 16-byte chunk sits in a stage: eight rows read at one chunk fall in different banks
    auto spot = [](int r, int ch) {
        return KC == 64 ? (r >> 1) * 128 + (r & 1) * 64 + ((ch ^ ((r >> 1) & 3)) << 4) : r * KC + ((ch ^ (r & 7)) << 4);
    };
    auto load = [&](int s, int k0) {
        unsigned char* px = buf + s * STAGE;
        unsigned char* pw = px + XB;
        for (int c = tid; c < BM * CH; c += THREADS) {
            const int r = c / CH, ch = c % CH;
            cp16(px + spot(r, ch), X + static_cast<size_t>(m0 + r) * K + k0 + ch * 16);
        }
        for (int c = tid; c < BN * CH; c += THREADS) {
            const int r = c / CH, ch = c % CH;
            cp16(pw + spot(r, ch), W + static_cast<size_t>(n0 + r) * K + k0 + ch * 16);
        }
    };

    int acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;
    float facc[MODE == wide ? MT : 1][MODE == wide ? NT : 1][4];
    if (MODE == wide) {
#pragma unroll
        for (int i = 0; i < (MODE == wide ? MT : 1); ++i)
#pragma unroll
            for (int j = 0; j < (MODE == wide ? NT : 1); ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) facc[i][j][e] = 0.0f;
    }
    const int row_a = (lane & 7) + ((lane >> 3) & 1) * 8, chunk_a = lane >> 4;
    const int row_b = (lane & 7) + ((lane >> 4) & 1) * 8, chunk_b = (lane >> 3) & 1;
    const int rbase = m0 + wm * (BM / WM) + (lane >> 2), cbase = n0 + wn * (BN / WN) + (lane & 3) * 2;

    load(0, 0);
    commit();
    const int slices = K / KC;
    for (int s = 0; s < slices; ++s) {
        if (s + 1 < slices) load((s + 1) & 1, (s + 1) * KC);
        commit();
        wait<1>();
        __syncthreads();
        const unsigned char* px = buf + (s & 1) * STAGE + wm * (BM / WM) * KC;
        const unsigned char* pw = buf + (s & 1) * STAGE + XB + wn * (BN / WN) * KC;
#pragma unroll
        for (int ks = 0; ks < KC / 32; ++ks) {
            uint32_t a[MT][4], b[NT / 2][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i * 16 + row_a;
                ldmatrix4(a[i], px + spot(r, ks * 2 + chunk_a));
            }
#pragma unroll
            for (int j = 0; j < NT / 2; ++j) {
                const int r = j * 16 + row_b;
                ldmatrix4(b[j], pw + spot(r, ks * 2 + chunk_b));
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j) mma_s8(acc[i][j], a[i], b[j / 2][(j & 1) * 2], b[j / 2][(j & 1) * 2 + 1]);
        }
        if (MODE == wide && ((s + 1) * KC) % wide_group == 0) {
            const int g = ((s + 1) * KC) / wide_group - 1, groups = K / wide_group;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int row = rbase + i * 16;
                const float s0 = XS[static_cast<size_t>(row) * groups + g], s1 = XS[static_cast<size_t>(row + 8) * groups + g];
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][0] += float(acc[i][j][0]) * s0;
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][1] += float(acc[i][j][1]) * s0;
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][2] += float(acc[i][j][2]) * s1;
                    facc[MODE == wide ? i : 0][MODE == wide ? j : 0][3] += float(acc[i][j][3]) * s1;
                    acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = rbase + i * 16 + h * 8;
            if (row >= M) continue;
            const float xs = MODE == wide ? 1.0f : XS[row];
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int col = cbase + j * 8;
                float v0, v1;
                if (MODE == wide) {
                    v0 = facc[MODE == wide ? i : 0][MODE == wide ? j : 0][2 * h] * WS[col];
                    v1 = facc[MODE == wide ? i : 0][MODE == wide ? j : 0][2 * h + 1] * WS[col + 1];
                } else {
                    v0 = float(acc[i][j][2 * h]) * xs * WS[col];
                    v1 = float(acc[i][j][2 * h + 1]) * xs * WS[col + 1];
                }
                if (MODE == swiglu) {
                    Y[static_cast<size_t>(row) * (N / 2) + col / 2] = __float2bfloat16(v1 / (1.0f + __expf(-v1)) * v0);
                } else {
                    __nv_bfloat162 o;
                    o.x = __float2bfloat16(v0);
                    o.y = __float2bfloat16(v1);
                    *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(row) * N + col) = o;
                }
            }
        }
    }
}

constexpr int head_dim = 128, tile_rows = 64;

// softmax(q k^T scale) v over each query tile's key tiles with an online softmax: a key tile's weights are rounded
// to 0..255 against each row's largest in that tile, so both products are int8 on the tensor cores.
// Q, K: (H, slots, 128) int8 in tile order. QS: (H, slots) a scale per query. KS, VS: (H, tiles) a scale per key tile.
// VT: (H, tiles, 128, 64) int8, each tile's values stored channel-major. LIST: key tiles, one list per (head, query
// tile) when per_query, else one list for all. SIZES: (tiles) real rows of a tile. ROWOF: (slots) the row at a slot
// or -1. Y: (R, H * 128) bf16. SOUT: when not null, (H, slots, slots) float that receives every score the kernel
// forms, in units of log 2 (a key past its tile's rows gets -1e30). Grid [query tiles, H], 128 threads, 44 KiB shared.
__device__ __forceinline__ void attention_body(
        const int8_t* __restrict__ Q, const float* __restrict__ QS, const int8_t* __restrict__ K,
        const float* __restrict__ KS, const int8_t* __restrict__ VT, const float* __restrict__ VS,
        const int* __restrict__ LIST, const int* __restrict__ SIZES, const int* __restrict__ ROWOF,
        __nv_bfloat16* __restrict__ Y, int slots, int tiles, int heads, int queries, int keys, int first_query,
        int per_query, float scale, float* __restrict__ SOUT) {
    constexpr int D = head_dim, T = tile_rows, THREADS = 128;
    constexpr int QB = T * D, KB = T * D, VB = D * T, STAGE = KB + VB, PB = T * T;
    extern __shared__ __align__(128) unsigned char buf[];
    unsigned char* const qsm = buf;
    unsigned char* const psm = buf + QB;
    unsigned char* const stages = buf + QB + PB;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, qt = first_query + blockIdx.x;
    const size_t hs = static_cast<size_t>(head) * slots, ht = static_cast<size_t>(head) * tiles;
    const int* list = LIST + (per_query ? (static_cast<size_t>(head) * queries + blockIdx.x) * keys : 0);

    // a row of 128 bytes holds eight 16-byte chunks, a row of 64 holds four: both placed so eight rows read at
    // one chunk fall in different banks
    auto at128 = [](int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); };
    auto at64 = [](int r, int c) { return (r >> 1) * 128 + (r & 1) * 64 + ((c ^ ((r >> 1) & 3)) << 4); };
    auto load = [&](int s, int tile) {
        unsigned char* pk = stages + s * STAGE;
        unsigned char* pv = pk + KB;
        const int8_t* k = K + (hs + static_cast<size_t>(tile) * T) * D;
        const int8_t* v = VT + (ht + tile) * VB;
        for (int c = tid; c < T * 8; c += THREADS) cp16(pk + at128(c >> 3, c & 7), k + c * 16);
        for (int c = tid; c < D * 4; c += THREADS) cp16(pv + at64(c >> 2, c & 3), v + c * 16);
    };

    const int8_t* q = Q + (hs + static_cast<size_t>(qt) * T) * D;
    for (int c = tid; c < T * 8; c += THREADS) cp16(qsm + at128(c >> 3, c & 7), q + c * 16);
    load(0, list[0]);
    commit();

    const int row_a = (lane & 7) + ((lane >> 3) & 1) * 8, chunk_a = lane >> 4;
    const int row_b = (lane & 7) + ((lane >> 4) & 1) * 8, chunk_b = (lane >> 3) & 1;
    const int r0 = warp * 16 + (lane >> 2), c0 = (lane & 3) * 2;
    const float qs0 = QS[hs + qt * T + r0] * scale * 1.4426950408889634f;
    const float qs1 = QS[hs + qt * T + r0 + 8] * scale * 1.4426950408889634f;
    float out[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) out[j][0] = out[j][1] = out[j][2] = out[j][3] = 0.0f;
    float top0 = -1e30f, top1 = -1e30f, mass0 = 0.0f, mass1 = 0.0f;
    uint32_t qa[4][4];

    for (int sel = 0; sel < keys; ++sel) {
        const int tile = list[sel];
        if (sel + 1 < keys) load((sel + 1) & 1, list[sel + 1]);
        commit();
        wait<1>();
        __syncthreads();
        if (sel == 0) {
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int r = warp * 16 + row_a;
                ldmatrix4(qa[ks], qsm + at128(r, ks * 2 + chunk_a));
            }
        }
        const unsigned char* pk = stages + (sel & 1) * STAGE;
        const unsigned char* pv = pk + KB;
        const int size = SIZES[tile];
        const float ks_t = KS[ht + tile], vs_t = VS[ht + tile];

        int sc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b[4];
                ldmatrix4(b, pk + at128(j * 16 + row_b, ks * 2 + chunk_b));
                mma_s8(sc[2 * j], qa[ks], b[0], b[1]);
                mma_s8(sc[2 * j + 1], qa[ks], b[2], b[3]);
            }
        }
        // each row's 64 scores sit in the four lanes of one group: its maximum needs two shuffles
        float s[8][4], m0 = -1e30f, m1 = -1e30f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + c0;
            const float o0 = col < size ? 0.0f : -1e30f, o1 = col + 1 < size ? 0.0f : -1e30f;
            s[j][0] = float(sc[j][0]) * (qs0 * ks_t) + o0;
            s[j][1] = float(sc[j][1]) * (qs0 * ks_t) + o1;
            s[j][2] = float(sc[j][2]) * (qs1 * ks_t) + o0;
            s[j][3] = float(sc[j][3]) * (qs1 * ks_t) + o1;
            m0 = fmaxf(m0, fmaxf(s[j][0], s[j][1]));
            m1 = fmaxf(m1, fmaxf(s[j][2], s[j][3]));
            if (SOUT) {
                float* o = SOUT + (hs + qt * T + r0) * slots + tile * T + col;
                o[0] = s[j][0];
                o[1] = s[j][1];
                o[8 * static_cast<size_t>(slots)] = s[j][2];
                o[8 * static_cast<size_t>(slots) + 1] = s[j][3];
            }
        }
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
        int l0 = 0, l1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + c0;
            const int u0 = min(255, int(exp2f(s[j][0] - m0) * 255.0f + 0.5f)), u1 = min(255, int(exp2f(s[j][1] - m0) * 255.0f + 0.5f));
            const int u2 = min(255, int(exp2f(s[j][2] - m1) * 255.0f + 0.5f)), u3 = min(255, int(exp2f(s[j][3] - m1) * 255.0f + 0.5f));
            l0 += u0 + u1;
            l1 += u2 + u3;
            *reinterpret_cast<uint16_t*>(psm + at64(r0, col >> 4) + (col & 15)) = uint16_t(u0 | (u1 << 8));
            *reinterpret_cast<uint16_t*>(psm + at64(r0 + 8, col >> 4) + (col & 15)) = uint16_t(u2 | (u3 << 8));
        }
        l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
        l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
        l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
        __syncwarp();

        int pr[16][4];
#pragma unroll
        for (int j = 0; j < 16; ++j) pr[j][0] = pr[j][1] = pr[j][2] = pr[j][3] = 0;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t a[4];
            ldmatrix4(a, psm + at64(warp * 16 + row_a, ks * 2 + chunk_a));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t b[4];
                ldmatrix4(b, pv + at64(j * 16 + row_b, ks * 2 + chunk_b));
                mma_u8s8(pr[2 * j], a, b[0], b[1]);
                mma_u8s8(pr[2 * j + 1], a, b[2], b[3]);
            }
        }
        const float n0 = fmaxf(top0, m0), n1 = fmaxf(top1, m1);
        const float f0 = exp2f(top0 - n0), f1 = exp2f(top1 - n1), e0 = exp2f(m0 - n0), e1 = exp2f(m1 - n1);
        const float g0 = e0 * vs_t, g1 = e1 * vs_t;
        mass0 = mass0 * f0 + e0 * float(l0);
        mass1 = mass1 * f1 + e1 * float(l1);
        top0 = n0;
        top1 = n1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            out[j][0] = out[j][0] * f0 + float(pr[j][0]) * g0;
            out[j][1] = out[j][1] * f0 + float(pr[j][1]) * g0;
            out[j][2] = out[j][2] * f1 + float(pr[j][2]) * g1;
            out[j][3] = out[j][3] * f1 + float(pr[j][3]) * g1;
        }
        __syncthreads();
    }
    const int to0 = ROWOF[qt * T + r0], to1 = ROWOF[qt * T + r0 + 8];
    const float i0 = 1.0f / fmaxf(mass0, 1e-30f), i1 = 1.0f / fmaxf(mass1, 1e-30f);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = head * D + j * 8 + c0;
        __nv_bfloat162 o;
        if (to0 >= 0) {
            o.x = __float2bfloat16(out[j][0] * i0);
            o.y = __float2bfloat16(out[j][1] * i0);
            *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(to0) * heads * D + col) = o;
        }
        if (to1 >= 0) {
            o.x = __float2bfloat16(out[j][2] * i1);
            o.y = __float2bfloat16(out[j][3] * i1);
            *reinterpret_cast<__nv_bfloat162*>(Y + static_cast<size_t>(to1) * heads * D + col) = o;
        }
    }
}

} // namespace tf_h3

namespace tf_h3 {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ int8_t q8(float x) { return int8_t(max(-127, min(127, __float2int_rn(x)))); }

// A value every thread of the block agrees on: the sum or the largest of what each holds.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < (blockDim.x >> 5); ++w) t += scratch[w];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < (blockDim.x >> 5); ++w) t = fmaxf(t, scratch[w]);
    __syncthreads();
    return t;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

} // namespace tf_h3

using namespace tf_h3;

extern "C" {

__global__ void __launch_bounds__(128) h3_gemm(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                               __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<plain, 128, 128, 2, 2, 64>(X, XS, W, WS, Y, M, N, K, group);
}
__global__ void __launch_bounds__(256) h3_gemm_n256(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                                    __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<plain, 128, 256, 2, 4>(X, XS, W, WS, Y, M, N, K, group);
}
__global__ void __launch_bounds__(128) h3_gemm_swiglu(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                                      __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<swiglu, 128, 128, 2, 2, 64>(X, XS, W, WS, Y, M, N, K, group);
}
__global__ void __launch_bounds__(256) h3_gemm_wide(const int8_t* X, const float* XS, const int8_t* W, const float* WS,
                                                    __nv_bfloat16* Y, int M, int N, int K, int group) {
    gemm_i8<wide, 256, 128, 4, 2>(X, XS, W, WS, Y, M, N, K, group);
}

// A projection stored (N, K) in bf16 to int8 with one scale per output channel. Row n lands at row0 + n, or with
// `half` at 2 (n - half) for n >= half and 2 n + 1 below it (a SwiGLU's [gate; value] rows as value, gate pairs).
// Grid [N], 128 threads.
__global__ void h3_quant_weight(const __nv_bfloat16* W, int8_t* W8, float* WS, int K, int row0, int half) {
    __shared__ float scratch[8];
    const int n = blockIdx.x;
    const __nv_bfloat16* w = W + static_cast<size_t>(n) * K;
    float top = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) top = fmaxf(top, fabsf(f(w[k])));
    top = fmaxf(block_max(top, scratch), 1e-12f);
    const int to = half > 0 ? (n >= half ? 2 * (n - half) : 2 * n + 1) : row0 + n;
    if (threadIdx.x == 0) WS[to] = top / 127.0f;
    const float inverse = 127.0f / top;
    int8_t* o = W8 + static_cast<size_t>(to) * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) o[k] = q8(f(w[k]) * inverse);
}

// A block's modulated RMSNorm straight to int8, one CTA a row; with a gate part it first adds the branch before
// it: X += GT[line, gate part] * Y. X, Y: (M, C) bf16. NW: (C) norm weight. GT, TAB: (L, 6, C) float tables.
// LINE: (M). Q: (MP, C) int8, rows from M zero. XS: (MP). Grid [MP], 128 threads.
__global__ void h3_norm_q8(__nv_bfloat16* X, const __nv_bfloat16* Y, const __nv_bfloat16* NW, const float* GT,
                           const float* TAB, const int* LINE, int8_t* Q, float* XS, int M, int C, int gate,
                           int scale, int shift, float eps) {
    __shared__ float scratch[8];
    const int row = blockIdx.x;
    int8_t* q = Q + static_cast<size_t>(row) * C;
    if (row >= M) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) q[c] = 0;
        if (threadIdx.x == 0) XS[row] = 0.0f;
        return;
    }
    const int line = LINE[row];
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    float sq = 0.0f;
    if (gate >= 0) {
        const float* g = GT + (static_cast<size_t>(line) * 6 + gate) * C;
        const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = bf(f(x[c]) + bf(bf(g[c]) * f(y[c])));
            x[c] = __float2bfloat16(v);
            sq += v * v;
        }
    } else {
        for (int c = threadIdx.x; c < C; c += blockDim.x) sq += f(x[c]) * f(x[c]);
    }
    const float inv = rsqrtf(block_sum(sq, scratch) / float(C) + eps);
    const float* s = TAB + (static_cast<size_t>(line) * 6 + scale) * C;
    const float* h = TAB + (static_cast<size_t>(line) * 6 + shift) * C;
    float top = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        top = fmaxf(top, fabsf(bf(bf(bf(f(x[c]) * inv * f(NW[c])) * bf(1.0f + s[c])) + bf(h[c]))));
    top = fmaxf(block_max(top, scratch), 1e-12f);
    if (threadIdx.x == 0) XS[row] = top / 127.0f;
    const float inverse = 127.0f / top;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        q[c] = q8(bf(bf(bf(f(x[c]) * inv * f(NW[c])) * bf(1.0f + s[c])) + bf(h[c])) * inverse);
}

// X += GT[line, part] * Y: the last block's MLP branch. Grid [M], 128 threads.
__global__ void h3_gate_add(__nv_bfloat16* X, const __nv_bfloat16* Y, const float* GT, const int* LINE, int C, int part) {
    const int row = blockIdx.x;
    const float* g = GT + (static_cast<size_t>(LINE[row]) * 6 + part) * C;
    __nv_bfloat16* x = X + static_cast<size_t>(row) * C;
    const __nv_bfloat16* y = Y + static_cast<size_t>(row) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) x[c] = __float2bfloat16(f(x[c]) + bf(bf(g[c]) * f(y[c])));
}

// The projections' rows laid out for tile attention: q and k take their head's RMSNorm and the split-half rotary
// over the first 2 ROT channels, then q, k and v round to int8 and land at the row's slot in tile order (v
// channel-major inside its tile). q has a scale per row, k and v one per (tile, head): a CTA owns a tile's 64 slots
// and walks the heads, so the tile's largest k and v are found (a block reduction) while the rows' values are still
// in registers. A thread holds a quarter of one row: 32 channels of q, k and v.
// QKV: (R, stride) bf16 with q, k, v at columns 0, H 128, 2 H 128. NQ, NK: (128). COS, SIN: (R, ROT) float.
// ROWOF: (slots) the row at a slot or -1. TQ, TK: (H, slots, 128); TQS: (H, slots). TVT: (H, tiles, 128, 64).
// KS, VS: (H, tiles). Slots without a row are left as they are (zero). Grid [tiles], 256 threads.
__global__ void __launch_bounds__(256) h3_heads(
        const __nv_bfloat16* QKV, const __nv_bfloat16* NQ, const __nv_bfloat16* NK, const float* COS, const float* SIN,
        const int* ROWOF, int8_t* TQ, float* TQS, int8_t* TK, int8_t* TVT, float* KS, float* VS, int H, int stride,
        int ROT, int slots, int tiles, float eps) {
    __shared__ float tops[16];
    const int tile = blockIdx.x, s = threadIdx.x >> 2, c0 = (threadIdx.x & 3) * 32, slot = tile * 64 + s;
    const int row = ROWOF[slot];
    const bool live = row >= 0;
    const float* cs = COS + static_cast<size_t>(live ? row : 0) * ROT;
    const float* sn = SIN + static_cast<size_t>(live ? row : 0) * ROT;
    for (int head = 0; head < H; ++head) {
        const __nv_bfloat16* qr = QKV + static_cast<size_t>(live ? row : 0) * stride + head * 128;
        const __nv_bfloat16* kr = qr + H * 128;
        const __nv_bfloat16* vr = kr + H * 128;
        float q[32], k[32], v[32], sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            q[j] = live ? f(qr[c0 + j]) : 0.0f;
            k[j] = live ? f(kr[c0 + j]) : 0.0f;
            v[j] = live ? f(vr[c0 + j]) : 0.0f;
            sq += q[j] * q[j];
            sk += k[j] * k[j];
        }
        // a row's four quarters sit in neighbouring lanes
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        const float qi = rsqrtf(sq / 128.0f + eps), ki = rsqrtf(sk / 128.0f + eps);
        float qt = 0.0f, kt = 0.0f, vt = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int c = c0 + j;
            float a = bf(q[j] * qi * f(NQ[c])), b = bf(k[j] * ki * f(NK[c]));
            if (c < 2 * ROT && live) {
                const bool low = c < ROT;
                const int p = low ? c + ROT : c - ROT, t = low ? c : c - ROT;
                const float qp = bf(f(qr[p]) * qi * f(NQ[p])), kp = bf(f(kr[p]) * ki * f(NK[p]));
                a = low ? a * cs[t] - qp * sn[t] : qp * sn[t] + a * cs[t];
                b = low ? b * cs[t] - kp * sn[t] : kp * sn[t] + b * cs[t];
            }
            q[j] = a;
            k[j] = b;
            qt = fmaxf(qt, fabsf(a));
            kt = fmaxf(kt, fabsf(b));
            vt = fmaxf(vt, fabsf(v[j]));
        }
        qt = fmaxf(qt, __shfl_xor_sync(0xffffffffu, qt, 1));
        qt = fmaxf(fmaxf(qt, __shfl_xor_sync(0xffffffffu, qt, 2)), 1e-12f);
        // the tile's largest k and v: each warp's, then the eight warps'
        kt = warp_max(kt);
        vt = warp_max(vt);
        if ((threadIdx.x & 31) == 0) {
            tops[threadIdx.x >> 5] = kt;
            tops[8 + (threadIdx.x >> 5)] = vt;
        }
        __syncthreads();
        kt = vt = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            kt = fmaxf(kt, tops[w]);
            vt = fmaxf(vt, tops[8 + w]);
        }
        __syncthreads();
        kt = fmaxf(kt, 1e-12f);
        vt = fmaxf(vt, 1e-12f);
        const size_t ht = static_cast<size_t>(head) * tiles + tile, at = static_cast<size_t>(head) * slots + slot;
        if (threadIdx.x == 0) {
            KS[ht] = kt / 127.0f;
            VS[ht] = vt / 127.0f;
        }
        if (!live) continue;
        if ((threadIdx.x & 3) == 0) TQS[at] = qt / 127.0f;
        const float qv = 127.0f / qt, kv = 127.0f / kt, vv = 127.0f / vt;
        int8_t* vo = TVT + (ht * 128 + c0) * 64 + s;
#pragma unroll
        for (int j = 0; j < 32; j += 4) {
            uint32_t pq = 0, pk = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                pq |= uint32_t(uint8_t(q8(q[j + e] * qv))) << (8 * e);
                pk |= uint32_t(uint8_t(q8(k[j + e] * kv))) << (8 * e);
                vo[(j + e) * 64] = q8(v[j + e] * vv);
            }
            *reinterpret_cast<uint32_t*>(TQ + at * 128 + c0 + j) = pq;
            *reinterpret_cast<uint32_t*>(TK + at * 128 + c0 + j) = pk;
        }
    }
}

// Each tile's mean q, k and v over its real rows, one warp per (tile, head), four channels a lane.
// QP, KP, VP: (H, tiles, 128) float. Grid [tiles, H], 32 threads.
__global__ void h3_pool(const int8_t* TQ, const float* TQS, const int8_t* TK, const float* KS, const int8_t* TVT,
                        const float* VS, const int* SIZES, float* QP, float* KP, float* VP, int slots, int tiles) {
    const int tile = blockIdx.x, head = blockIdx.y, size = SIZES[tile], c0 = 4 * threadIdx.x;
    const size_t first = static_cast<size_t>(head) * slots + tile * 64, ht = static_cast<size_t>(head) * tiles + tile;
    float q[4] = {0, 0, 0, 0};
    int k[4] = {0, 0, 0, 0}, v[4] = {0, 0, 0, 0};
    for (int s = 0; s < size; ++s) {
        const int8_t* q8p = TQ + (first + s) * 128 + c0;
        const int8_t* k8p = TK + (first + s) * 128 + c0;
        const float qs = TQS[first + s];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] += float(q8p[j]) * qs;
            k[j] += k8p[j];
            v[j] += TVT[(ht * 128 + c0 + j) * 64 + s];
        }
    }
    const float inv = 1.0f / float(size), ks = KS[ht] * inv, vs = VS[ht] * inv;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        QP[ht * 128 + c0 + j] = q[j] * inv;
        KP[ht * 128 + c0 + j] = float(k[j]) * ks;
        VP[ht * 128 + c0 + j] = float(v[j]) * vs;
    }
}

// The tiles' means laid out as a sequence of their own for h3_attention, 64 tiles a group: q with a scale per tile,
// k and v with one scale per (group, head), v channel-major inside its group. One thread a tile.
// QP, KP, VP: (H, tiles, 128) float. PQ, PK: (H, groups * 64, 128) int8; PQS: (H, groups * 64).
// PVT: (H, groups, 128, 64) int8. PKS, PVS: (H, groups). Grid [groups, H], 64 threads.
__global__ void h3_pool_quant(const float* QP, const float* KP, const float* VP, int8_t* PQ, float* PQS, int8_t* PK,
                              float* PKS, int8_t* PVT, float* PVS, int tiles, int groups) {
    __shared__ float scratch[8];
    const int group = blockIdx.x, head = blockIdx.y, tile = group * 64 + threadIdx.x;
    const bool real = tile < tiles;
    const size_t from = (static_cast<size_t>(head) * tiles + tile) * 128;
    float qt = 0.0f, kt = 0.0f, vt = 0.0f;
    if (real) {
        for (int c = 0; c < 128; ++c) {
            qt = fmaxf(qt, fabsf(QP[from + c]));
            kt = fmaxf(kt, fabsf(KP[from + c]));
            vt = fmaxf(vt, fabsf(VP[from + c]));
        }
    }
    qt = fmaxf(qt, 1e-12f);
    kt = fmaxf(block_max(kt, scratch), 1e-12f);
    vt = fmaxf(block_max(vt, scratch), 1e-12f);
    const size_t hg = static_cast<size_t>(head) * groups + group, to = (static_cast<size_t>(head) * groups * 64 + tile) * 128;
    if (threadIdx.x == 0) {
        PKS[hg] = kt / 127.0f;
        PVS[hg] = vt / 127.0f;
    }
    PQS[static_cast<size_t>(head) * groups * 64 + tile] = real ? qt / 127.0f : 0.0f;
    const float qv = 127.0f / qt, kv = 127.0f / kt, vv = 127.0f / vt;
    for (int c = 0; c < 128; ++c) {
        PQ[to + c] = real ? q8(QP[from + c] * qv) : 0;
        PK[to + c] = real ? q8(KP[from + c] * kv) : 0;
        PVT[(hg * 128 + c) * 64 + threadIdx.x] = real ? q8(VP[from + c] * vv) : 0;
    }
}

// Each video query tile's key tiles: every prefix tile, then its KEEP best video tiles by score, in tile order.
// The KEEP-th largest score is found a bit at a time over the scores' ordered bit patterns; equal scores are taken
// lowest tile first. One warp per (video tile, head), so a round is one warp reduction and no barrier.
// S: (H, span, span) with the tiles first on both sides. IDX: (H, video tiles, P0 + KEEP).
// Grid [video tiles, H], 32 threads.
__global__ void h3_topk(const float* S, int* IDX, int tiles, int span, int P0, int KEEP) {
    const int NV = tiles - P0, tq = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float* s = S + (static_cast<size_t>(head) * span + P0 + tq) * span + P0;
    int* out = IDX + (static_cast<size_t>(head) * NV + tq) * (P0 + KEEP);
    for (int i = lane; i < P0; i += 32) out[i] = i;
    auto key = [&](int j) {
        const uint32_t b = __float_as_uint(s[j]);
        return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    };
    uint32_t thr = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = thr | (1u << bit);
        uint32_t mine = 0;
        for (int j = lane; j < NV; j += 32) mine += key(j) >= cand;
        if (__reduce_add_sync(0xffffffffu, mine) >= uint32_t(KEEP)) thr = cand;
    }
    // a kept tile's place is the number kept before it: lanes count in order, each after the lanes before it
    uint32_t above = 0;
    for (int j = lane; j < NV; j += 32) above += key(j) > thr;
    const int ties = KEEP - int(__reduce_add_sync(0xffffffffu, above));
    int kept = 0, tied = 0;
    for (int j0 = 0; j0 < NV; j0 += 32) {
        const int j = j0 + lane;
        const uint32_t k = j < NV ? key(j) : 0u;
        const uint32_t eq = __ballot_sync(0xffffffffu, j < NV && k == thr);
        const uint32_t gt = __ballot_sync(0xffffffffu, j < NV && k > thr);
        // of this round's equal scores, the first `ties - tied` are kept
        const int my_tie = tied + __popc(eq & ((1u << lane) - 1));
        const uint32_t take = __ballot_sync(0xffffffffu, (gt >> lane & 1) || ((eq >> lane & 1) && my_tie < ties));
        if (take >> lane & 1) out[P0 + kept + __popc(take & ((1u << lane) - 1))] = P0 + j;
        kept += __popc(take);
        tied += __popc(eq);
    }
}

// The pooled branch gated into the attention output and the row rounded to int8 in one pass, one CTA a row.
// Y: (R, W) bf16. G: the gate projection's rows, (R, stride) bf16. C: (tiles, W) bf16, each tile's pooled
// attention, or null for dense attention (nothing is mixed in). Q: (MP, W) int8, rows from R zero. XS: (MP).
// Grid [MP], 128 threads.
__global__ void h3_mix_quant(const __nv_bfloat16* Y, const __nv_bfloat16* G, const __nv_bfloat16* C, const int* SLOT,
                             int8_t* Q, float* XS, int R, int W, int stride) {
    __shared__ float scratch[8];
    const int row = blockIdx.x;
    int8_t* q = Q + static_cast<size_t>(row) * W;
    if (row >= R) {
        for (int c = threadIdx.x; c < W; c += blockDim.x) q[c] = 0;
        if (threadIdx.x == 0) XS[row] = 0.0f;
        return;
    }
    const __nv_bfloat16* y = Y + static_cast<size_t>(row) * W;
    const __nv_bfloat16* g = G + static_cast<size_t>(row) * stride;
    const __nv_bfloat16* coarse = C ? C + static_cast<size_t>(SLOT[row] >> 6) * W : nullptr;
    auto mixed = [&](int c) {
        if (!coarse) return f(y[c]);
        return bf(f(y[c]) + bf(f(coarse[c]) * f(g[c])));
    };
    float top = 0.0f;
    for (int c = threadIdx.x; c < W; c += blockDim.x) top = fmaxf(top, fabsf(mixed(c)));
    top = fmaxf(block_max(top, scratch), 1e-12f);
    if (threadIdx.x == 0) XS[row] = top / 127.0f;
    const float inverse = 127.0f / top;
    for (int c = threadIdx.x; c < W; c += blockDim.x) q[c] = q8(mixed(c) * inverse);
}

// The MLP's wide rows to int8 with a scale per (row, 1024 columns), one warp a group. X: (R, W) bf16.
// Q: (MP, W) int8, rows from R zero. XS: (MP, W / 1024). Grid [MP], 128 threads.
__global__ void h3_quant_wide(const __nv_bfloat16* X, int8_t* Q, float* XS, int R, int W) {
    const int row = blockIdx.x, lane = threadIdx.x & 31, groups = W / wide_group;
    for (int g = threadIdx.x >> 5; g < groups; g += blockDim.x >> 5) {
        int8_t* q = Q + static_cast<size_t>(row) * W + g * wide_group;
        if (row >= R) {
            for (int c = lane * 4; c < wide_group; c += 128) *reinterpret_cast<uint32_t*>(q + c) = 0;
            if (lane == 0) XS[static_cast<size_t>(row) * groups + g] = 0.0f;
            continue;
        }
        const __nv_bfloat16* x = X + static_cast<size_t>(row) * W + g * wide_group;
        float top = 0.0f;
        for (int c = lane; c < wide_group; c += 32) top = fmaxf(top, fabsf(f(x[c])));
        top = fmaxf(warp_max(top), 1e-12f);
        if (lane == 0) XS[static_cast<size_t>(row) * groups + g] = top / 127.0f;
        const float inverse = 127.0f / top;
        for (int c = lane * 4; c < wide_group; c += 128) {
            uint32_t p = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) p |= uint32_t(uint8_t(q8(f(x[c + j]) * inverse))) << (8 * j);
            *reinterpret_cast<uint32_t*>(q + c) = p;
        }
    }
}

__global__ void __launch_bounds__(128) h3_attention(
        const int8_t* Q, const float* QS, const int8_t* K, const float* KS, const int8_t* VT, const float* VS,
        const int* LIST, const int* SIZES, const int* ROWOF, __nv_bfloat16* Y, int slots, int tiles, int heads,
        int queries, int keys, int first_query, int per_query, float scale, float* SOUT) {
    attention_body(Q, QS, K, KS, VT, VS, LIST, SIZES, ROWOF, Y, slots, tiles, heads, queries, keys, first_query, per_query, scale, SOUT);
}

} // extern "C"
