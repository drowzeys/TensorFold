// GLM-5.3 decode-window helpers (Phase 3a speed work), ours unless noted:
//
//   glm53_topk_hist / glm53_topk_count / glm53_topk_write: the K best columns of a few rows, ascending (ties to the
//     lower column) - topk._select_keys' result (one program a row) with a row spread over blocks of TOPK_CHUNK
//     columns, for decode windows (fused.select keeps torch.topk + sort below RADIX_MIN_ROWS rows because the
//     one-program radix select loses there: 21 indexer layers x 0.4-0.5 ms a window at 32K-128K keys). Three
//     histogram passes over the 32-bit order word (bits 31-21, 20-10, 9-0; the last arriving block of a row picks
//     the digit and clears the histogram for the next pass), per-block counts of the keys above / at the K-th, then
//     an ordered write. Selection only compares words, so any algorithm that keeps the same set in column order
//     returns the same int32 columns: the bits of the Triton select and of torch.topk + sort.
//     fmode 0: uint32 order words (_index_scores PACK=False); 1: fp32 values ordered as sampling.topBetter orders
//     them (value descending, NaN above everything, -0 == +0, the lower column first among equals) - the sampled
//     verify head's per-rank candidates (sampling.selectTop's set).
//   glm53_cands: candidate words of a sampled window, [values (cnt) ; global ids as int32 bits (cnt)] a row, from the
//     selected columns (the layout cuda_runner.sampleRows sends; `choose` sorts the gathered candidates itself, so
//     their order within a row does not matter).
//   glm53_l2pf: MiaAI-Lab patch 0046's l2pf_kernel (families/glm_moe_dsa/cuda/l2pf.cu, itself adapted from
//     jayleaton/glm53-tensorfold-spark patches/0460, Apache-2.0, Copyright 2026 Jay Leaton), unchanged: it only
//     prefetches or loads, so no result can change.
#include <cstdint>

namespace {
constexpr int SEL_THREADS = 256;
constexpr int SEL_BINS = 2048;

__device__ __forceinline__ unsigned sel_key(unsigned bits, int fmode) {
    if (!fmode) return bits;
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0xFFFFFFFFu;     // NaN: first, ties by column
    if (bits == 0x80000000u) bits = 0u;                               // -0 compares equal to +0
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);       // monotone in the float's order
}

__device__ __forceinline__ int sel_shift(int pass) { return pass == 0 ? 21 : pass == 1 ? 10 : 0; }
__device__ __forceinline__ int sel_bins(int pass) { return pass == 2 ? 1024 : 2048; }
}  // namespace

// Grid (blocks of `chunk` columns, rows), SEL_THREADS threads. state [rows, 4] uint32: prefix, fixed mask, keys
// still needed at the prefix, arrival count (0 between launches). ghist [rows, SEL_BINS] uint32 (0 between passes).
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_hist(const unsigned* __restrict__ keys, int nc, long long ld, int chunk, int pass, int K, int fmode,
                unsigned* __restrict__ ghist, unsigned* __restrict__ state) {
    __shared__ unsigned h[SEL_BINS];
    __shared__ unsigned suf[SEL_THREADS];
    __shared__ int last;
    const int row = blockIdx.y, t = threadIdx.x;
    const unsigned* s = keys + (size_t)row * ld;
    unsigned* gh = ghist + (size_t)row * SEL_BINS;
    unsigned* st = state + (size_t)row * 4;
    const int shift = sel_shift(pass), nb = sel_bins(pass);
    // read before any block of this launch can rewrite them (the last block writes only after every block arrived)
    const unsigned prefix = pass ? st[0] : 0u;
    const unsigned fixed = pass ? st[1] : 0u;
    const unsigned need = pass ? st[2] : (unsigned)K;
    for (int i = t; i < nb; i += SEL_THREADS) h[i] = 0u;
    __syncthreads();
    const int c0 = blockIdx.x * chunk;
    const int c1 = min(nc, c0 + chunk);
    for (int i = c0 + t; i < c1; i += SEL_THREADS) {
        const unsigned u = sel_key(s[i], fmode);
        if ((u & fixed) == prefix) atomicAdd(&h[(u >> shift) & (unsigned)(nb - 1)], 1u);
    }
    __syncthreads();
    for (int i = t; i < nb; i += SEL_THREADS)
        if (h[i]) atomicAdd(&gh[i], h[i]);
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(&st[3], 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = t; i < nb; i += SEL_THREADS) {
        h[i] = __ldcg(&gh[i]);
        gh[i] = 0u;
    }
    __syncthreads();
    const int per = nb / SEL_THREADS;                 // 8 or 4 bins a thread, ascending
    unsigned mine = 0;
    for (int j = 0; j < per; ++j) mine += h[t * per + j];
    suf[t] = mine;
    __syncthreads();
    for (int off = 1; off < SEL_THREADS; off <<= 1) {   // suf[t] = keys in the bins of threads t..
        const unsigned v = t + off < SEL_THREADS ? suf[t + off] : 0u;
        __syncthreads();
        suf[t] += v;
        __syncthreads();
    }
    unsigned acc = suf[t] - mine;                     // keys in the bins above this thread's
    for (int j = per - 1; j >= 0; --j) {              // the digit: keys above it < need <= keys at or above it
        const unsigned c = h[t * per + j];
        if (acc < need && acc + c >= need) {
            st[0] = prefix | ((unsigned)(t * per + j) << shift);
            st[1] = fixed | ((unsigned)(nb - 1) << shift);
            st[2] = need - acc;
        }
        acc += c;
    }
    if (t == 0) st[3] = 0u;
}

// Grid (blocks, rows): cnt[(row * blocks + block) * 2 + {0, 1}] = keys of the block above / equal to the K-th key.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_count(const unsigned* __restrict__ keys, int nc, long long ld, int chunk, int fmode,
                 const unsigned* __restrict__ state, unsigned* __restrict__ cnt) {
    __shared__ unsigned sg[SEL_THREADS / 32], se[SEL_THREADS / 32];
    const int row = blockIdx.y, t = threadIdx.x;
    const unsigned* s = keys + (size_t)row * ld;
    const unsigned prefix = state[(size_t)row * 4];
    const int c0 = blockIdx.x * chunk;
    const int c1 = min(nc, c0 + chunk);
    unsigned g = 0, e = 0;
    for (int i = c0 + t; i < c1; i += SEL_THREADS) {
        const unsigned u = sel_key(s[i], fmode);
        g += u > prefix;
        e += u == prefix;
    }
    for (int o = 16; o > 0; o >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, o);
        e += __shfl_down_sync(0xffffffffu, e, o);
    }
    if ((t & 31) == 0) {
        sg[t >> 5] = g;
        se[t >> 5] = e;
    }
    __syncthreads();
    if (t == 0) {
        unsigned a = 0, b = 0;
        for (int w = 0; w < SEL_THREADS / 32; ++w) {
            a += sg[w];
            b += se[w];
        }
        cnt[((size_t)row * gridDim.x + blockIdx.x) * 2] = a;
        cnt[((size_t)row * gridDim.x + blockIdx.x) * 2 + 1] = b;
    }
}

// Grid (blocks, rows): out[row * out_ld + j], j < K = the row's K best columns ascending: every key above the K-th
// and the first `need` keys equal to it (column order). A thread walks chunk / SEL_THREADS consecutive columns.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_write(const unsigned* __restrict__ keys, int nc, long long ld, int chunk, int fmode,
                 const unsigned* __restrict__ state, const unsigned* __restrict__ cnt, int* __restrict__ out,
                 long long out_ld) {
    __shared__ unsigned sg[SEL_THREADS], se[SEL_THREADS];
    __shared__ unsigned bg, be;
    const int row = blockIdx.y, t = threadIdx.x;
    const unsigned* s = keys + (size_t)row * ld;
    int* o = out + (size_t)row * out_ld;
    const unsigned prefix = state[(size_t)row * 4];
    const unsigned need = state[(size_t)row * 4 + 2];
    if (t == 0) {
        unsigned a = 0, b = 0;
        for (unsigned i = 0; i < blockIdx.x; ++i) {
            a += cnt[((size_t)row * gridDim.x + i) * 2];
            b += cnt[((size_t)row * gridDim.x + i) * 2 + 1];
        }
        bg = a;
        be = b;
    }
    const int per = chunk / SEL_THREADS;
    const int i0 = blockIdx.x * chunk + t * per;
    const int i1 = min(nc, i0 + per);
    unsigned g = 0, e = 0;
    for (int i = i0; i < i1; ++i) {
        const unsigned u = sel_key(s[i], fmode);
        g += u > prefix;
        e += u == prefix;
    }
    sg[t] = g;
    se[t] = e;
    __syncthreads();
    for (int off = 1; off < SEL_THREADS; off <<= 1) {   // inclusive prefix sums over threads
        const unsigned a = t >= off ? sg[t - off] : 0u;
        const unsigned b = t >= off ? se[t - off] : 0u;
        __syncthreads();
        sg[t] += a;
        se[t] += b;
        __syncthreads();
    }
    unsigned gb = bg + sg[t] - g;                     // keys above the K-th before this thread's first column
    unsigned eb = be + se[t] - e;                     // keys equal to it before that column
    for (int i = i0; i < i1; ++i) {
        const unsigned u = sel_key(s[i], fmode);
        if (u > prefix) {
            o[gb + min(eb, need)] = i;
            ++gb;
        } else if (u == prefix) {
            if (eb < need) o[gb + eb] = i;
            ++eb;
        }
    }
}

// Grid (rows), block >= cnt threads: send[r, :cnt] = lg[r, cols[r, j]], send[r, cnt:] = int32 bits of cols + off.
extern "C" __global__ void glm53_cands(const float* __restrict__ lg, long long ld, const int* __restrict__ cols,
                                       int cnt, long long off, float* __restrict__ send) {
    const int r = blockIdx.x, j = threadIdx.x;
    if (j >= cnt) return;
    const int c = cols[(size_t)r * cnt + j];
    send[(size_t)r * 2 * cnt + j] = lg[(size_t)r * ld + c];
    reinterpret_cast<int*>(send)[(size_t)r * 2 * cnt + cnt + j] = (int)(c + off);
}

// l2pf.cu's kernel: TABLE [n, 2] int64 (address, bytes) pieces, 16-byte aligned. MODE 0 (bulk): one
// cp.async.bulk.prefetch.L2 a piece; 1 (lines): prefetch.global.L2::evict_last a 128-byte line; 2 (touch): ld.global.cg.
extern "C" __global__ void glm53_l2pf(const long long* __restrict__ table, int n, int mode, unsigned* __restrict__ sink) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    if (mode == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            const unsigned long long a = (unsigned long long)table[2 * i];
            const unsigned bytes = (unsigned)table[2 * i + 1];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"(bytes) : "memory");
#else
            for (unsigned o = 0; o < bytes; o += 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(a + o));
#endif
        }
        return;
    }
    unsigned acc = 0;
    for (int i = warp; i < n; i += warps) {
        const unsigned long long a = (unsigned long long)table[2 * i];
        const unsigned bytes = (unsigned)table[2 * i + 1];
        for (unsigned o = lane * 128u; o < bytes; o += 32u * 128u) {
            if (mode == 1) {
                asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a + o));
            } else {
                unsigned x, y, z, w;
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(a + o));
                acc ^= x ^ y ^ z ^ w;
            }
        }
    }
    if (mode == 2 && acc == 0x9e3779b9u && sink != nullptr) sink[0] = acc;
}
