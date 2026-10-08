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
