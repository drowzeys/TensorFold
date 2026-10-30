// Device code of src/tensorfold/cuda/kernels/prefill_attention.cu (comments, ATen includes dropped), paged keys added.
#include "prefill_attention.cuh"

namespace tf_prefill_attention {

template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
pattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
             const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
             int G, float scale, const int* __restrict__ PT, long long PS) {
    constexpr bool causal = true;
    constexpr int skip = 0;
    constexpr bool F8 = false, HM = false;  // bf16 keys, row-major
    constexpr int CAP = 0, PTOK = 1;
    const float *ks = nullptr, *vs = nullptr;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// One segment's rows from row `first` on, keys from p0 on `skip` rows further on: the body for tree_kernel.
template <int D, int WARPS, int HPC, int NS>
__device__ __forceinline__ void
pattn_rows(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
           const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
           int G, float scale, int first, int skip) {
    constexpr bool causal = true;
    constexpr const int* PT = nullptr;  // contiguous keys
    constexpr long long PS = 0;
    constexpr int PTOK = 1;
    constexpr bool F8 = false, HM = false;  // bf16 keys, row-major
    constexpr int CAP = 0;
    const float *ks = nullptr, *vs = nullptr;
    const int HO = H;
    auto first_row = [=] { return first; };
#include "prefill_attention_rows.cuh"
}

// Packed rows: a causal trunk of P rows, then branches at P + cu[b], each over the trunk and its own earlier rows.
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
tree_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
            const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, const int* __restrict__ cu, int P,
            int n, int H, int HK, int G, float scale) {
    constexpr int ROWS = 16 * (WARPS / HPC);
    int tile = blockIdx.x, b = -1, W = P;                // the trunk's row tiles, then each branch's
    while (tile >= (W + ROWS - 1) / ROWS) {
        tile -= (W + ROWS - 1) / ROWS;
        if (++b == n) return;
        W = cu[b + 1] - cu[b];
    }
    const int64_t at = (int64_t)(b < 0 ? 0 : P + cu[b]) * H * D;
    pattn_rows<D, WARPS, HPC, NS>(q + at, kc, vc, out + at, b < 0 ? 0 : P, W, H, HK, G, scale, tile * ROWS,
                                  b < 0 ? 0 : cu[b]);
}

// One sequence, every row over every key (no mask), a block a head: q, k, v heads of qkv [W, 3 HO D] -> out [W, HO D].
template <int D, int WARPS, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
full_kernel(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out, int W, int HO, float scale) {
    constexpr int HPC = 1, G = 1, p0 = 0, skip = 0;
    constexpr bool causal = false;
    constexpr const int* PT = nullptr;  // contiguous keys
    constexpr long long PS = 0;
    constexpr int PTOK = 1;
    constexpr bool F8 = false, HM = false;  // bf16 keys, row-major
    constexpr int CAP = 0;
    const float *ks = nullptr, *vs = nullptr;
    const int H = 3 * HO, HK = H;
    const __nv_bfloat16* __restrict__ q = qkv;
    const __nv_bfloat16* __restrict__ kc = qkv + HO * D;
    const __nv_bfloat16* __restrict__ vc = qkv + 2 * HO * D;
    auto first_row = [] { return blockIdx.x * 16 * WARPS; };
#include "prefill_attention_rows.cuh"
}

// A head-major cache (qwen35.cu's: each key head's CAP rows in a run, or PT's pages of PTOK rows a head, PS apart).
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
hpattn_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
              const __nv_bfloat16* __restrict__ vc, __nv_bfloat16* __restrict__ out, int p0, int W, int H, int HK,
              int G, float scale, int CAP, const int* __restrict__ PT, long long PS, int PTOK) {
    constexpr bool causal = true, F8 = false, HM = true;
    constexpr int skip = 0;
    const int HO = H;
    const float *ks = nullptr, *vs = nullptr;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// hpattn_kernel over e4m3 rows with an fp32 scale a (key head, row) (qwen35_kv8.cu's layout, PS in bytes).
template <int D, int WARPS, int HPC, int NS>
__global__ void __launch_bounds__(32 * WARPS, 1)
hpattn8_kernel(const __nv_bfloat16* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
               const float* __restrict__ ks, const float* __restrict__ vs, __nv_bfloat16* __restrict__ out, int p0,
               int W, int H, int HK, int G, float scale, int CAP, const int* __restrict__ PT, long long PS, int PTOK) {
    constexpr bool causal = true, F8 = true, HM = true;
    constexpr int skip = 0;
    const int HO = H;
    auto first_row = [] { return blockIdx.x * 16 * (WARPS / HPC); };
#include "prefill_attention_rows.cuh"
}

// full_kernel's row arithmetic, one barrier a 64-key tile: two buffers of odd-padded rows, the next tile loading.
template <int D, int WARPS>
__global__ void __launch_bounds__(32 * WARPS)
vision_kernel(const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ out, int W, int HO, float scale) {
#include "prefill_attention_vision.cuh"
}

} // namespace tf_prefill_attention

// Head dim 128, eight warps, eight query heads a block, eight staging slots (Nemotron's 16 heads a KV head).
template __global__ void tf_prefill_attention::pattn_kernel<128, 8, 8, 8>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int, int,
    int, int, float, const int* __restrict__, long long);
// Head dim 128, eight warps, four query heads a block, eight staging slots, over paged caches (LFM3: 4 a KV head).
template __global__ void tf_prefill_attention::pattn_kernel<128, 8, 4, 8>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int, int,
    int, int, float, const int* __restrict__, long long);
// Head dim 64, eight warps, four query heads a block, eight staging slots, over paged caches (LFM2.5-24B-A2B's chunks).
template __global__ void tf_prefill_attention::pattn_kernel<64, 8, 4, 8>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int, int,
    int, int, float, const int* __restrict__, long long);
// Head dim 64, eight warps, four query heads a block, eight staging slots (LFM2's 4 heads a KV head, causal or trees).
template __global__ void tf_prefill_attention::tree_kernel<64, 8, 4, 8>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, const int* __restrict__, int, int, int, int, int, float);
// Head dim 72, eight warps of one head, eight staging slots (SigLIP2's tower, 72 dims padded to 80 inside).
template __global__ void tf_prefill_attention::full_kernel<72, 8, 8>(const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int,
    float);
// Head dim 80, eight warps, two blocks an SM (RADIO's tower, 16 heads of 80): full_kernel<80>'s bits, 1.35x faster.
template __global__ void tf_prefill_attention::vision_kernel<80, 8>(const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int,
    float);
// Head dim 64, eight warps (LFM3-VL's tower, 16 heads of 64, its 8 kv heads repeated in the rows).
template __global__ void tf_prefill_attention::vision_kernel<64, 8>(const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int,
    float);
// Head dim 256, eight warps, two query heads a block, two staging slots (Qwen3.8-27B's prompt chunks, bf16 and fp8).
template __global__ void tf_prefill_attention::hpattn_kernel<256, 8, 2, 2>(const __nv_bfloat16* __restrict__,
    const __nv_bfloat16* __restrict__, const __nv_bfloat16* __restrict__, __nv_bfloat16* __restrict__, int, int, int, int,
    int, float, int, const int* __restrict__, long long, int);
template __global__ void tf_prefill_attention::hpattn8_kernel<256, 8, 2, 2>(const __nv_bfloat16* __restrict__,
    const uint8_t* __restrict__, const uint8_t* __restrict__, const float* __restrict__, const float* __restrict__,
    __nv_bfloat16* __restrict__, int, int, int, int, int, float, int, const int* __restrict__, long long, int);
