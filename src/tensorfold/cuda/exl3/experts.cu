// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomic sums (only arrival counts); 4-bit mcg matches GLM's kernel bit for bit.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "experts_grouped.cuh"

namespace {

using tf_exl3x::HAD_SCALE;
using tf_exl3x::bf16r;
using tf_exl3x::fwht128;
using tf_exl3x::griddep_launch;
using tf_exl3x::griddep_wait;
using tf_exl3x::launch_ex;

// Every kernel here starts with griddep_wait / griddep_launch: launched with PDL (TF_EXL3_EXPERTS_PDL) it waits for
// the previous kernel before touching memory; launched without, both are no-ops.

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    griddep_wait();
    griddep_launch();
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    griddep_wait();
    griddep_launch();
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    griddep_wait();
    griddep_launch();
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    tf_exl3x::gateup_epi<false>(Z, p, e, blk * 128 + 4 * lane, P, N, SK, svh_g, svh_u, suh_d, xd, limit, act_mode,
                                lane);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    griddep_wait();
    griddep_launch();
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4], o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    tf_exl3x::down_epi(v, e, n, D, svh_d, lane, o);
#pragma unroll
    for (int j = 0; j < 4; ++j) y[(size_t)p * D + n + j] = o[j];
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    griddep_wait();
    griddep_launch();
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits).
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    griddep_wait();
    griddep_launch();
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        tf_exl3x::down_epi(v, e, n, D, svh_d, lane, o);
#pragma unroll
        for (int j = 0; j < 4; ++j) y[(size_t)p * D + n + j] = o[j];
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = acc[j];
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

namespace tf_exl3x {
extern template void grouped_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void dequant_launch<0>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<1>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<2>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

void exl3x_grouped_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                        const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                        const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                        int64_t SK, int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo,
                        int64_t hi, int64_t ns, int64_t vec, int64_t fuse, const at::Tensor& pick, int64_t E,
                        const at::Tensor& svh_g, const at::Tensor& svh_u, const at::Tensor& suh_d, at::Tensor& xd,
                        double limit, int64_t act_mode, const at::Tensor& svh_d, at::Tensor& y,
                        const at::Tensor& wts, at::Tensor& out, const at::Tensor& sy, at::Tensor& done, int64_t pdl) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    TORCH_CHECK(vec >= 0 && vec <= 4, "vec: 0 (32-bit loads) or 1..4 (16-byte loads that many k steps ahead)");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    a.ns = (int)ns;
    a.vec = (int)vec;
    a.pdl = pdl != 0;
    auto hp = [](const at::Tensor& t) { return t.numel() ? reinterpret_cast<const half*>(t.data_ptr()) : nullptr; };
    auto fp = [](const at::Tensor& t) { return t.numel() ? t.data_ptr<float>() : nullptr; };
    a.ep.fuse = (int)fuse;
    if (fuse) {
        a.ep.pick = pick.data_ptr<int>();
        a.ep.E = (int)E;
        a.ep.svh_g = hp(svh_g);
        a.ep.svh_u = hp(svh_u);
        a.ep.suh_d = hp(suh_d);
        a.ep.xd = xd.numel() ? reinterpret_cast<half*>(xd.data_ptr()) : nullptr;
        a.ep.limit = (float)limit;
        a.ep.act_mode = (int)act_mode;
        a.ep.svh_d = hp(svh_d);
        a.ep.y = fp(y);
        a.ep.wts = fp(wts);
        a.ep.out = fp(out);
        a.ep.sy = fp(sy);
        a.ep.done = done.numel() ? done.data_ptr<int>() : nullptr;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_launch<0>(a, stream);
    else if (cb == 1) tf_exl3x::grouped_launch<1>(a, stream);
    else if (cb == 2) tf_exl3x::grouped_launch<2>(a, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_dequant_cuda(const at::Tensor& T, at::Tensor& out, int64_t K, int64_t N, int64_t k2, int64_t cb) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = reinterpret_cast<const uint32_t*>(T.data_ptr());
    auto o = reinterpret_cast<half*>(out.data_ptr());
    if (cb == 0) tf_exl3x::dequant_launch<0>(t, o, (int)K, (int)N, (int)k2, stream);
    else if (cb == 1) tf_exl3x::dequant_launch<1>(t, o, (int)K, (int)N, (int)k2, stream);
    else tf_exl3x::dequant_launch<2>(t, o, (int)K, (int)N, (int)k2, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_cuda(const at::Tensor& pick, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members, int64_t R,
                      int64_t slots, int64_t E, int64_t pdl) {
    TORCH_CHECK(E <= GROUP_THREADS * GROUP_PER_THREAD, "too many experts for the grouping kernel");
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    const size_t smem = (size_t)R * slots * sizeof(int);
    constexpr size_t static_smem = GROUP_THREADS / 32 * sizeof(int);
    if (smem + static_smem > 48 * 1024) {
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, group_kernel));
        const auto* device = at::cuda::getCurrentDeviceProperties();
        const size_t limit = device->sharedMemPerBlockOptin - attributes.sharedSizeBytes;
        TORCH_CHECK(smem <= limit, "EXL3 grouping needs ", smem, " dynamic shared-memory bytes; this GPU allows ",
                    limit, " after the kernel's static storage");
        if (smem > (size_t)attributes.maxDynamicSharedSizeBytes)
            C10_CUDA_CHECK(cudaFuncSetAttribute(group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)limit));
    }
    launch_ex(group_kernel, dim3(1), dim3(GROUP_THREADS), smem, at::cuda::getCurrentCUDAStream(), pdl != 0,
              pick.data_ptr<int>(), uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), (int)R,
              (int)slots, (int)E, (int)members.size(1));
}

void exl3x_rot_in_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                       const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1, int64_t rows, int64_t K,
                       int64_t slots, int64_t E, int64_t pdl) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    if (x.scalar_type() == at::kBFloat16)
        launch_ex(rot_in_kernel<__nv_bfloat16>, grid, dim3(32), 0, stream, pdl != 0,
                  reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x_stride, pick.data_ptr<int>(), s0, s1,
                  o0, o1, (int)K, (int)slots, (int)E);
    else
        launch_ex(rot_in_kernel<half>, grid, dim3(32), 0, stream, pdl != 0, reinterpret_cast<const half*>(x.data_ptr()),
                  (int)x_stride, pick.data_ptr<int>(), s0, s1, o0, o1, (int)K, (int)slots, (int)E);
}

void exl3x_gateup_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g,
                                const at::Tensor& svh_u, const at::Tensor& suh_d, at::Tensor& xd, int64_t rows,
                                int64_t P, int64_t N, int64_t SK, int64_t slots, int64_t E, double limit,
                                int64_t act_mode, int64_t pdl) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(N / 128));
    launch_ex(gateup_epilogue_kernel, grid, dim3(32), 0, at::cuda::getCurrentCUDAStream(), pdl != 0,
              Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_g.data_ptr()),
              reinterpret_cast<const half*>(svh_u.data_ptr()), reinterpret_cast<const half*>(suh_d.data_ptr()),
              reinterpret_cast<half*>(xd.data_ptr()), (int)P, (int)N, (int)SK, (int)E, (float)limit, (int)act_mode);
}

void exl3x_down_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                              int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E,
                              int64_t pdl) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(D / 128));
    launch_ex(down_epilogue_kernel, grid, dim3(32), 0, at::cuda::getCurrentCUDAStream(), pdl != 0,
              Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
              y.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E);
}

void exl3x_combine_cuda(const at::Tensor& y, const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t D,
                        int64_t slots, int64_t pdl) {
    dim3 grid((unsigned)rows, (unsigned)((D + 255) / 256));
    launch_ex(combine_kernel, grid, dim3(256), 0, at::cuda::getCurrentCUDAStream(), pdl != 0, y.data_ptr<float>(),
              wts.data_ptr<float>(), out.data_ptr<float>(), (int)D, (int)slots);
}

void exl3x_down_combine_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                             const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t P, int64_t D, int64_t SK,
                             int64_t slots, int64_t E, int64_t pdl) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    launch_ex(down_combine_kernel, grid, dim3((unsigned)(32 * slots)), 0, at::cuda::getCurrentCUDAStream(), pdl != 0,
              Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
              y.data_ptr<float>(), wts.data_ptr<float>(), out.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E,
              (int)slots);
}
