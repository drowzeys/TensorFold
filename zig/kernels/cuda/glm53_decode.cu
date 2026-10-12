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
//     verify head's per-rank candidates (sampling.selectTop's set). 2 (Phase 3b, DCP): packed int64 keys of
//     _index_scores PACK=True (ld counts int64 keys), compared by their high word in unsigned order - the score's
//     order word, the padding (int64 min + 1) at 0; ties to the lower column = the lower position, which the low word
//     orders the same way, so the set is torch.topk's over the whole int64 keys (fused.select's DCP branch).
//   glm53_cands: candidate words of a sampled window, [values (cnt) ; global ids as int32 bits (cnt)] a row, from the
//     selected columns (the layout cuda_runner.sampleRows sends; `choose` sorts the gathered candidates itself, so
//     their order within a row does not matter).
//   glm53_dcp_qpack / glm53_dcp_cands / glm53_dcp_merge (Phase 3b, decode context parallelism; fused._attention_dcp's
//     qpack copy and fused.select's DCP branch after the per-rank top-k): the absorbed query and its rotated part as
//     one [rows, H, LW + RD] row; this rank's candidate keys padded to K (int64 min + 1); the K best of every rank's
//     gathered candidates, this rank's own slots ascending (torch.topk + where + sort, or topk.top_keys' high-word
//     select on the radix branch). Moves and integer comparisons only.
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

// key i of a row at `s` (a row's base: keys + row * ld words, two words a key in fmode 2)
__device__ __forceinline__ unsigned key_at(const unsigned* s, int i, int fmode) {
    if (fmode == 2) return s[2 * (size_t)i + 1] ^ 0x80000000u;    // the packed key's high word, unsigned order
    return sel_key(s[i], fmode);
}

__device__ __forceinline__ const unsigned* row_of(const unsigned* keys, int row, long long ld, int fmode) {
    return keys + (size_t)row * (size_t)ld * (fmode == 2 ? 2 : 1);
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
    const unsigned* s = row_of(keys, row, ld, fmode);
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
        const unsigned u = key_at(s, i, fmode);
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
    const unsigned* s = row_of(keys, row, ld, fmode);
    const unsigned prefix = state[(size_t)row * 4];
    const int c0 = blockIdx.x * chunk;
    const int c1 = min(nc, c0 + chunk);
    unsigned g = 0, e = 0;
    for (int i = c0 + t; i < c1; i += SEL_THREADS) {
        const unsigned u = key_at(s, i, fmode);
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
    const unsigned* s = row_of(keys, row, ld, fmode);
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
        const unsigned u = key_at(s, i, fmode);
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
        const unsigned u = key_at(s, i, fmode);
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

// ---- Phase 3b: decode context parallelism (fused._attention_dcp's qpack, fused.select's DCP branch) ---------------

// Grid (rows * heads), 128 threads: out[i] = [qlat[i] (lw) ; qrot[i] (rd)] (bf16 bits), i = row * H + head
// (qp[:, :, :lw].copy_(b.qlat[:R]); qp[:, :, lw:].copy_(b.qrot[:R])).
extern "C" __global__ void glm53_dcp_qpack(const unsigned short* __restrict__ qlat, const unsigned short* __restrict__ qrot,
                                           unsigned short* __restrict__ out, int lw, int rd) {
    const size_t i = blockIdx.x;
    unsigned short* o = out + i * (size_t)(lw + rd);
    for (int k = threadIdx.x; k < lw; k += blockDim.x) o[k] = qlat[i * lw + k];
    for (int k = threadIdx.x; k < rd; k += blockDim.x) o[lw + k] = qrot[i * rd + k];
}

// Grid (rows), 256 threads: this rank's candidates mine[r, :K] (fused.select's `mine`: int64 min + 1, then the top kk).
// mode 0: src = int32 columns [rows, sld] (ascending, glm53_topk_write) -> keys sc[r * ld + col]; mode 1: src = int64
// keys [rows, sld] (topk.top_keys) as they are. Entries kk..K-1 keep the padding value.
extern "C" __global__ void glm53_dcp_cands(const long long* __restrict__ sc, long long ld, const void* __restrict__ src,
                                           long long sld, int kk, int K, int mode, long long* __restrict__ out) {
    const size_t r = blockIdx.x;
    const long long pad = -9223372036854775807LL;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        long long v = pad;
        if (j < kk) {
            if (mode == 0) {
                const int c = reinterpret_cast<const int*>(src)[r * sld + j];
                v = sc[r * ld + c];
            } else {
                v = reinterpret_cast<const long long*>(src)[r * sld + j];
            }
        }
        out[r * K + j] = v;
    }
}

namespace {
constexpr int MERGE_THREADS = 1024;
constexpr int MERGE_MAX_K = 2048;   // index_topk (GLM-5.3: 2048); the own-slot sort's width
}  // namespace

// Grid (rows), MERGE_THREADS threads. cand [dcp, rows, K] int64 (dcp_gather of every rank's `mine`); the row's
// candidates in the order of allc.permute(1, 0, 2).reshape(rows, dcp * K): column c = s * K + j. Keeps the K best:
// hi = 0 by the whole int64 (torch.topk: keys are distinct but for the padding), hi = 1 by the high word in unsigned
// order with ties to the lower column (topk.top_keys, the radix branch); then gpos = 0x7FFFFFFF - low word, own =
// gpos % dcp == rank, tok[r, :] = own slots gpos / dcp ascending then 0 (sort of where(own, gpos // dcp, 1 << 40) cast
// to int32), cnt[r] = own count. Requires K <= MERGE_MAX_K.
extern "C" __global__ void __launch_bounds__(MERGE_THREADS)
glm53_dcp_merge(const long long* __restrict__ cand, int rows, int K, int dcp, int rank, int hi,
                int* __restrict__ tok, int* __restrict__ cnt) {
    __shared__ unsigned hist[256];
    __shared__ unsigned long long s_prefix, s_fixed;
    __shared__ unsigned s_need;
    __shared__ unsigned scan[MERGE_THREADS];
    __shared__ int buf[MERGE_MAX_K];
    __shared__ unsigned s_own;
    const int r = blockIdx.x, t = threadIdx.x;
    const int M = dcp * K;
    const unsigned long long mask = hi ? 0xFFFFFFFF00000000ull : 0xFFFFFFFFFFFFFFFFull;
    auto key_of = [&](int c) -> long long {
        const int s = c / K, j = c - s * K;
        return cand[((size_t)s * rows + r) * K + j];
    };
    auto ord_of = [&](long long k) -> unsigned long long {
        return ((unsigned long long)k ^ 0x8000000000000000ull) & mask;
    };
    if (t == 0) {
        s_prefix = 0ull;
        s_fixed = 0ull;
        s_need = (unsigned)K;
        s_own = 0u;
    }
    __syncthreads();
    const int passes = hi ? 4 : 8;
    for (int p = 0; p < passes; ++p) {
        const int shift = 56 - 8 * p;
        for (int i = t; i < 256; i += MERGE_THREADS) hist[i] = 0u;
        __syncthreads();
        const unsigned long long prefix = s_prefix, fixed = s_fixed;
        for (int c = t; c < M; c += MERGE_THREADS) {
            const unsigned long long u = ord_of(key_of(c));
            if ((u & fixed) == prefix) atomicAdd(&hist[(u >> shift) & 0xFFull], 1u);
        }
        __syncthreads();
        if (t == 0) {
            unsigned acc = 0, need = s_need;
            for (int d = 255; d >= 0; --d) {
                if (acc + hist[d] >= need) {
                    s_prefix = prefix | ((unsigned long long)d << shift);
                    s_fixed = fixed | (0xFFull << shift);
                    s_need = need - acc;
                    break;
                }
                acc += hist[d];
            }
        }
        __syncthreads();
    }
    const unsigned long long thr = s_prefix;
    const unsigned need = s_need;
    // keys above the K-th, and the first `need` equal to it in column order
    const int per = (M + MERGE_THREADS - 1) / MERGE_THREADS;
    const int c0 = t * per, c1 = min(M, c0 + per);
    unsigned e = 0;
    for (int c = c0; c < c1; ++c) e += ord_of(key_of(c)) == thr;
    scan[t] = e;
    __syncthreads();
    for (int off = 1; off < MERGE_THREADS; off <<= 1) {   // inclusive prefix sums over threads
        const unsigned a = t >= off ? scan[t - off] : 0u;
        __syncthreads();
        scan[t] += a;
        __syncthreads();
    }
    unsigned eb = scan[t] - e;
    for (int i = t; i < MERGE_MAX_K; i += MERGE_THREADS) buf[i] = 0x7FFFFFFF;
    __syncthreads();
    for (int c = c0; c < c1; ++c) {
        const long long k = key_of(c);
        const unsigned long long u = ord_of(k);
        bool take = u > thr;
        if (u == thr) {
            take = eb < need;
            ++eb;
        }
        if (!take) continue;
        const long long gpos = 0x7FFFFFFFLL - (long long)((unsigned long long)k & 0xFFFFFFFFull);
        if (gpos % dcp == rank) {
            const unsigned at = atomicAdd(&s_own, 1u);
            if (at < MERGE_MAX_K) buf[at] = (int)(gpos / dcp);
        }
    }
    __syncthreads();
    for (int k = 2; k <= MERGE_MAX_K; k <<= 1) {           // bitonic sort, ascending
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = t; i < MERGE_MAX_K; i += MERGE_THREADS) {
                const int x = i ^ j;
                if (x > i) {
                    const bool up = (i & k) == 0;
                    const int a = buf[i], b = buf[x];
                    if ((a > b) == up) {
                        buf[i] = b;
                        buf[x] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    const unsigned own = min(s_own, (unsigned)K);
    for (int j = t; j < K; j += MERGE_THREADS) tok[(size_t)r * K + j] = (unsigned)j < own ? buf[j] : 0;
    if (t == 0) cnt[r] = (int)own;
}
