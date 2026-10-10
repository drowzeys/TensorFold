// GLM-5.3's RoPE inverse frequencies on the GPU, as fused.inv_freq has torch compute them:
//   1.0 / (theta ** (torch.arange(0, dim, 2, dtype=float32, device="cuda") / dim))
// torch's CUDA kernels behind each step, in float32:
//   arange          start + step * i                       -> 2i (exact)
//   / dim           true_divide by a CPU scalar: a * (1 / dim), the reciprocal taken on the host in float
//   theta ** t      pow(Scalar, Tensor): pow(float(theta), t) -> powf (libdevice __nv_powf), not exp2 (theta != 2)
//   1.0 / p         Tensor.__rtruediv__: reciprocal() (1.0f / p, IEEE) times 1.0 (exact)
// Built with torch's own nvcc defaults (-O3, no fast math, fmad on) so powf and the division are the same code.
#include <cuda_runtime.h>

extern "C" __global__ void glm53_inv_freq(float* out, int half, int dim, float inv_dim, float theta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    const float t = static_cast<float>(2 * i) * inv_dim;
    const float p = powf(theta, t);
    out[i] = 1.0f / p;
}
