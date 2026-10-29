// Ours: qmm_prefill's prompt GEMM on smaller tiles, bit-equal to it, for prompts of tens to hundreds of rows.

#include "qmm_midrow.cuh"

// tf_qmm_mid_<rows>x<columns>: Mid<BM, BN, WM, WN, STAGES> on prefill_kernel's arguments (affine4_mid.zig's table).
#define TF_QMM_MID(BM, BN, WM, WN, ST)                                                                                   \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32)                                                           \
        tf_qmm_mid_##BM##x##BN(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,                      \
                               const __nv_bfloat16* __restrict__ s, const __nv_bfloat16* __restrict__ b,                 \
                               __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int ldx, int group) {     \
        tf_qmm_midrow::mid<BM, BN, WM, WN, ST>(x, w, s, b, out, M, N, K, npad, ldx, group);                              \
    }

TF_QMM_MID(16, 32, 1, 4, 10)
TF_QMM_MID(16, 64, 1, 4, 8)
TF_QMM_MID(32, 64, 1, 4, 6)
TF_QMM_MID(32, 128, 1, 4, 4)
TF_QMM_MID(32, 256, 1, 8, 3)
TF_QMM_MID(48, 256, 1, 8, 3)
TF_QMM_MID(64, 128, 1, 8, 4)
TF_QMM_MID(80, 256, 1, 8, 2)
TF_QMM_MID(96, 128, 1, 8, 2)
TF_QMM_MID(128, 256, 2, 4, 2)
#undef TF_QMM_MID

static_assert(tf_qmm_midrow::Mid<16, 32, 1, 4, 10>::SMEM == 32000, "mid smem");
static_assert(tf_qmm_midrow::Mid<16, 64, 1, 4, 8>::SMEM == 34816, "mid smem");
static_assert(tf_qmm_midrow::Mid<32, 64, 1, 4, 6>::SMEM == 38400, "mid smem");
static_assert(tf_qmm_midrow::Mid<32, 128, 1, 4, 4>::SMEM == 34816, "mid smem");
static_assert(tf_qmm_midrow::Mid<32, 256, 1, 8, 3>::SMEM == 39936, "mid smem");
static_assert(tf_qmm_midrow::Mid<48, 256, 1, 8, 3>::SMEM == 46080, "mid smem");
static_assert(tf_qmm_midrow::Mid<64, 128, 1, 8, 4>::SMEM == 51200, "mid smem");
static_assert(tf_qmm_midrow::Mid<80, 256, 1, 8, 2>::SMEM == 38912, "mid smem");
static_assert(tf_qmm_midrow::Mid<96, 128, 1, 8, 2>::SMEM == 33792, "mid smem");
static_assert(tf_qmm_midrow::Mid<128, 256, 2, 4, 2>::SMEM == 51200, "mid smem");
