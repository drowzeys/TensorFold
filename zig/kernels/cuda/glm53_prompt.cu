// The torch steps of GLM-5.3's prompt chunks (fused.attention_core's wide path, gather's ring branch,
// experts_part's prompt branch) as plain kernels: each is data movement or one IEEE operation an element, so any
// implementation has torch's bits.
//   glm53_bhx_to_hbx  out[r, h, :] = in[h, r, :] (bf16; Tensor.copy_ of the bmm result into the [R, H, X] view)
//   glm53_bf16_f32    out[i] = float(in[i]) (Tensor.copy_ bf16 -> fp32: exact)
//   glm53_add_f32     a[i] += b[i] (Tensor.add_, fp32: one rounding)
//   glm53_hadamard    the 128 x 128 +-1 Hadamard matrix x3prefill.Workspace.hadamard builds (bf16)
#include <cstdint>
#include <cuda_bf16.h>

extern "C" __global__ void glm53_bhx_to_hbx(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int H, int R,
                                            int X) {
    const int r = blockIdx.x, h = blockIdx.y;
    const uint16_t* src = in + ((size_t)h * R + r) * X;
    uint16_t* dst = out + ((size_t)r * H + h) * X;
    for (int i = threadIdx.x; i < X; i += blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void glm53_bf16_f32(const __nv_bfloat16* __restrict__ in, float* __restrict__ out, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

extern "C" __global__ void glm53_add_f32(float* __restrict__ a, const float* __restrict__ b, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = __fadd_rn(a[i], b[i]);
}

extern "C" __global__ void glm53_hadamard(__nv_bfloat16* __restrict__ out) {
    const int i = blockIdx.x, j = threadIdx.x;
    out[i * 128 + j] = __float2bfloat16_rn((__popc(i & j) & 1) ? -1.0f : 1.0f);
}
