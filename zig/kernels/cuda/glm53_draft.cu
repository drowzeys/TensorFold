// ours (Phase 4): the torch steps of GLM-5.3's speculative drafters on the device (cuda/dspark.py, cuda/dflash.py,
// glm5_next/cuda/dflash2.py), so a drafter's block pass and tap passes replay as CUDA graphs with no host step:
//   glm53_draft_rotary   Drafter._rotary: pos = (pos_dev + arange(n)).float(); phase = pos[:, None] * inv_freq;
//                        cos / sin of it (torch's cos / sin: cosf / sinf, no fast math)
//   glm53_draft_scatter  kc[i].index_copy_(1, (pos_dev + arange(n)) % RING, k): [heads, n, hd] rows into the ring
//   glm53_draft_pos_add  pos_dev += n (int64)
//   glm53_draft_rowsum   Drafter._row's tail: acc = g[0]; acc += g[1] ...; acc.to(bfloat16) (fp32 adds in rank order)
//   glm53_draft_dot      SparkDrafter: h.float() @ conf_h (one fp32 dot a row; cuBLAS' order differs: drafts only)
//   glm53_draft_linear   GlmDrafter: F.linear(h, hproj).float() (bf16 rows, fp32 sums, the result rounded to bf16)
//   glm53_copy_rows      strided row copies (a tap layer's rows into its column block, tap column runs)
// Phase 4b (--parallel N, cuda_multi.zig: multi.py's index_select / index_copy_ on its round tables, and the draft cut):
//   glm53_rows_gather    dst[r] = src[idx[r]]   (torch.index_select(src, 0, idx, out=dst): rows of 32-bit words)
//   glm53_rows_scatter   dst[idx[r]] = src[r]   (dst.index_copy_(0, idx, src))
//   glm53_cut_stats      the draft cut's numbers of an MTP step's S picks: this rank's log-sum-exp over its share of
//                        the draft head's logits (torch.logsumexp; our reduction order: decisions only, never tokens)
//                        and the picked draft's logit (the gathered maximum over the ranks' records)
// Phase 4c (--parallel N with DFlash2 / DSpark, cuda_mdraft.zig: dflash.MultiDrafter's tables):
//   glm53_draft_rotary_rows   MultiDrafter._rotary(pos): cos / sin of each row's own position (an int64 table)
//   glm53_draft_scatter_rows  kc[i].index_copy_(1, idx, k) into pooled rings: row r to its slot's ring at its position
//                             (a slot < 0 writes nothing: the padding of a captured row bucket)
// Drafts only propose - the target verifies every row - so a step here may round differently from torch without
// moving any reply; the steps that matter for tokens a round (rotary, the ring, the rank-order sums) are torch's.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// Grid (rows), block >= half: cos / sin [rows, half] fp32.
extern "C" __global__ void glm53_draft_rotary(const long long* __restrict__ pos, const float* __restrict__ inv, int half,
                                              float* __restrict__ cos_out, float* __restrict__ sin_out) {
    const int r = blockIdx.x, i = threadIdx.x;
    if (i >= half) return;
    const float p = static_cast<float>(pos[0] + r);
    const float ph = p * inv[i];
    cos_out[static_cast<size_t>(r) * half + i] = cosf(ph);
    sin_out[static_cast<size_t>(r) * half + i] = sinf(ph);
}

// Grid (n, heads), block hd / 8 threads (16 bytes each): src [heads, n, hd] bf16 -> ring [heads, ring_rows, hd] at
// slot (pos + r) % ring_rows (slot_base: the ring's first row inside a pooled buffer; 0 for one stream).
extern "C" __global__ void glm53_draft_scatter(const uint4* __restrict__ src, uint4* __restrict__ ring, const long long* pos,
                                               int n, int hd, int ring_rows) {
    const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    const int w = hd / 8;  // uint4 words a row
    if (j >= w) return;
    const long long slot = (pos[0] + r) % ring_rows;
    ring[(static_cast<size_t>(h) * ring_rows + static_cast<size_t>(slot)) * w + j] =
        src[(static_cast<size_t>(h) * n + r) * w + j];
}

extern "C" __global__ void glm53_draft_pos_add(long long* pos, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) pos[0] += n;
}

// Grid (ceil(count / 256)): out[i] = bf16(g[0][i] + g[1][i] + ...), rank `k`'s words at g + k * stride.
extern "C" __global__ void glm53_draft_rowsum(const float* __restrict__ g, long long stride, int world, int count,
                                              __nv_bfloat16* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float acc = g[i];
    for (int k = 1; k < world; ++k) acc = __fadd_rn(acc, g[static_cast<size_t>(k) * stride + i]);
    out[i] = __float2bfloat16(acc);
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float part[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    if (lane == 0) part[warp] = v;
    __syncthreads();
    float t = 0.0f;
    if (warp == 0) {
        t = lane < warps ? part[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xFFFFFFFFu, t, o);
    }
    return t;  // valid in thread 0
}

// Grid (rows), block 256: out[r] = sum_d float(h[r, d]) * w[d].
extern "C" __global__ void glm53_draft_dot(const __nv_bfloat16* __restrict__ h, const float* __restrict__ w, int D,
                                           float* __restrict__ out) {
    const int r = blockIdx.x;
    float acc = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) acc += __bfloat162float(h[static_cast<size_t>(r) * D + d]) * w[d];
    const float t = block_sum(acc);
    if (threadIdx.x == 0) out[r] = t;
}

// Grid (rows, R), block 256: out[r, j] = float(bf16(sum_d h[r, d] * W[j, d])).
extern "C" __global__ void glm53_draft_linear(const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ W,
                                              int D, int R, float* __restrict__ out) {
    const int r = blockIdx.x, j = blockIdx.y;
    float acc = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        acc += __bfloat162float(h[static_cast<size_t>(r) * D + d]) * __bfloat162float(W[static_cast<size_t>(j) * D + d]);
    const float t = block_sum(acc);
    if (threadIdx.x == 0) out[static_cast<size_t>(r) * R + j] = __bfloat162float(__float2bfloat16(t));
}

// Grid (rows), block 256: row r's `words` 16-byte words from src + r * src_pitch to dst + r * dst_pitch (pitches in
// 16-byte words).
extern "C" __global__ void glm53_copy_rows(uint4* __restrict__ dst, long long dst_pitch, const uint4* __restrict__ src,
                                           long long src_pitch, int words) {
    const int r = blockIdx.x;
    for (int j = threadIdx.x; j < words; j += blockDim.x)
        dst[static_cast<size_t>(r) * dst_pitch + j] = src[static_cast<size_t>(r) * src_pitch + j];
}

// Grid (n), block 256: dst[r, :words] = src[idx[r], :words] (32-bit words; idx int64).
extern "C" __global__ void glm53_rows_gather(unsigned* __restrict__ dst, const unsigned* __restrict__ src,
                                             const long long* __restrict__ idx, int words) {
    const int r = blockIdx.x;
    const size_t from = static_cast<size_t>(idx[r]) * words, to = static_cast<size_t>(r) * words;
    for (int j = threadIdx.x; j < words; j += blockDim.x) dst[to + j] = src[from + j];
}

// Grid (n), block 256: dst[idx[r], :words] = src[r, :words] (32-bit words; idx int64; distinct targets).
extern "C" __global__ void glm53_rows_scatter(unsigned* __restrict__ dst, const long long* __restrict__ idx,
                                              const unsigned* __restrict__ src, int words) {
    const int r = blockIdx.x;
    const size_t to = static_cast<size_t>(idx[r]) * words, from = static_cast<size_t>(r) * words;
    for (int j = threadIdx.x; j < words; j += blockDim.x) dst[to + j] = src[from + j];
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float part[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFu, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    if (lane == 0) part[warp] = v;
    __syncthreads();
    float t = -INFINITY;
    if (warp == 0) {
        t = lane < warps ? part[lane] : -INFINITY;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_down_sync(0xFFFFFFFFu, t, o));
    }
    return t;  // valid in thread 0
}

// Grid (S), block 256: lse[s] = max + log(sum_v exp(lg[s, v] - max)) over lg [S, V] (row stride ld floats);
// top[s] = max over the ranks of amax_all[w, s, 0] (the [world, S, 4] head records).
extern "C" __global__ void glm53_cut_stats(const float* __restrict__ lg, int V, int ld,
                                           const float* __restrict__ amax_all, int world, int S,
                                           float* __restrict__ lse, float* __restrict__ top) {
    __shared__ float shared_max;
    const int s = blockIdx.x;
    const float* row = lg + static_cast<size_t>(s) * ld;
    float m = -INFINITY;
    for (int v = threadIdx.x; v < V; v += blockDim.x) m = fmaxf(m, row[v]);
    m = block_max(m);
    if (threadIdx.x == 0) shared_max = m;
    __syncthreads();
    m = shared_max;
    float acc = 0.0f;
    if (m != -INFINITY)
        for (int v = threadIdx.x; v < V; v += blockDim.x) acc += expf(row[v] - m);
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        lse[s] = m == -INFINITY ? -INFINITY : m + logf(acc);
        float t = amax_all[static_cast<size_t>(s) * 4];
        for (int w = 1; w < world; ++w) t = fmaxf(t, amax_all[(static_cast<size_t>(w) * S + s) * 4]);
        top[s] = t;
    }
}

// Grid (rows), block >= half: cos / sin [rows, half] fp32 of positions pos[r] (int64 table; glm53_draft_rotary's
// arithmetic: the position as fp32, times inv_freq, cosf / sinf).
extern "C" __global__ void glm53_draft_rotary_rows(const long long* __restrict__ pos, const float* __restrict__ inv,
                                                   int half, float* __restrict__ cos_out, float* __restrict__ sin_out) {
    const int r = blockIdx.x, i = threadIdx.x;
    if (i >= half) return;
    const float p = static_cast<float>(pos[r]);
    const float ph = p * inv[i];
    cos_out[static_cast<size_t>(r) * half + i] = cosf(ph);
    sin_out[static_cast<size_t>(r) * half + i] = sinf(ph);
}

// Grid (n, heads), block hd / 8 threads (16 bytes each): src [heads, n, hd] bf16 -> pool row
// h * head_stride + slot[r] * slot_stride + pos[r] % ring_rows (strides in rows of hd); slot[r] < 0: row r skipped.
extern "C" __global__ void glm53_draft_scatter_rows(const uint4* __restrict__ src, uint4* __restrict__ pool,
                                                    const long long* __restrict__ slot, const long long* __restrict__ pos,
                                                    int n, int hd, long long head_stride, long long slot_stride,
                                                    int ring_rows) {
    const int r = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    const int w = hd / 8;  // uint4 words a row
    if (j >= w) return;
    const long long sl = slot[r];
    if (sl < 0) return;
    const long long row = static_cast<long long>(h) * head_stride + sl * slot_stride + pos[r] % ring_rows;
    pool[static_cast<size_t>(row) * w + j] = src[(static_cast<size_t>(h) * n + r) * w + j];
}
