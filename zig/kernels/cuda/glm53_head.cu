// ours: the small torch steps of GLM-5.3's decode windows on the device, so a window (and an MTP draft step) replays
// as one CUDA graph with no host step inside (Phase 2b). Each kernel does what the Python engine's torch op does:
//   glm53_argmax_rec   fused.head "argmax" / "draft": torch.argmax(lg, -1) (first maximum, NaN above every number),
//                      a[:, 0] = lg.gather(1, i), a[:, 1] = (i + vocab_off).float() or draft_ids[i].float()
//   glm53_pick_rank    fused.head's resolve: torch.argmax(g[:, :, 0], dim=0) over the ranks (the lowest rank among equal
//                      maxima), b.argmax[:n] = g[best, r, 1].long()
//   glm53_embed_rows   torch.index_select(embed, 0, ids, out=x) (int64 ids), and mtp_compute's me[0].zero_()
//   glm53_f32_bf16     Tensor.copy_ of fp32 into bf16 (c10::BFloat16 on the device: __float2bfloat16, round to nearest)
// Moves and comparisons only (plus one cast), so no flag changes a bit; built with torch's default nvcc flags.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr unsigned FULL = 0xFFFFFFFFu;

// torch.argmax's order: NaN above every number (the first NaN wins), then the larger value, ties to the lower index
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) {
    const bool na = isnan(a), nb = isnan(b);
    if (na != nb) return na;
    if (na) return ia < ib;
    return a > b || (a == b && ia < ib);
}

}  // namespace

// One block a row (blockDim.x a multiple of 32, at most 1024): lg row r is lg + r * ld, V values; the record goes to
// amax[r * 4 + 0..1]; amax[r * 4 + 2..3] are left alone (the Python head never writes them: a stop vote's spare word).
extern "C" __global__ void glm53_argmax_rec(const float* lg, int V, int ld, const long long* idmap, long long off,
                                            float* amax) {
    const int r = blockIdx.x;
    const float* row = lg + static_cast<size_t>(r) * static_cast<size_t>(ld);
    float bv = -INFINITY;
    int bi = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float v = row[i];
        if (better(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(FULL, bv, o);
        const int oi = __shfl_down_sync(FULL, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    bv = lane < warps ? sv[lane] : -INFINITY;
    bi = lane < warps ? si[lane] : 0x7FFFFFFF;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(FULL, bv, o);
        const int oi = __shfl_down_sync(FULL, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane != 0) return;
    if (bi >= V) bi = 0;                              // V == 0 never happens; keep the read inside the row
    amax[r * 4 + 0] = row[bi];
    const long long id = idmap != nullptr ? idmap[bi] : static_cast<long long>(bi) + off;
    amax[r * 4 + 1] = static_cast<float>(id);
}

// g [world, rows, 4] (every rank's records, rank order) -> out[r] (int64) for r < rows; one thread a row.
extern "C" __global__ void glm53_pick_rank(const float* g, int world, int rows, long long* out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int best = 0;
    float bv = g[static_cast<size_t>(r) * 4];
    for (int k = 1; k < world; ++k) {
        const float v = g[(static_cast<size_t>(k) * rows + r) * 4];
        if (better(v, k, bv, best)) {
            bv = v;
            best = k;
        }
    }
    out[r] = static_cast<long long>(g[(static_cast<size_t>(best) * rows + r) * 4 + 1]);
}

// out[r, :] = table[ids[r], :] for rows r < gridDim.x (D bf16 a row, D % 8 == 0, 16-byte aligned rows); zero_first:
// row 0 zeroed instead (position 0's embedding is masked in the MTP layer).
extern "C" __global__ void glm53_embed_rows(const __nv_bfloat16* table, const long long* ids, int D,
                                            __nv_bfloat16* out, int zero_first) {
    const int r = blockIdx.x;
    uint4* dst = reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * D);
    const int n = D / 8;
    if (zero_first != 0 && r == 0) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const uint4* src = reinterpret_cast<const uint4*>(table + static_cast<size_t>(ids[r]) * D);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// y[i] = bfloat16(x[i]) for i < n
extern "C" __global__ void glm53_f32_bf16(const float* x, __nv_bfloat16* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}
