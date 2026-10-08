// GLM-5.3's RoCE one-shot all-reduce and all-gather for decode windows: b12x RoCEnante's two CuTe DSL kernels
// (b12x/comm/roce/_oneshot_cute.py and _allgather_cute.py, Apache-2.0, the b12x contributors) rewritten in plain CUDA
// with the same protocol, the same PTX for every access to the pinned region and the same rank-order fp32 sum, so a
// reduction stores exactly the bits of the NCCL all-gather + rank-order sum on every rank.
//
// One launch is one collective:
//   1. stage: the local input -> send[seq & 1] in pinned host memory (the NIC reads it there);
//   2. doorbell: the last block to finish staging writes nbytes (ctrl + 4, and the per-slot word ctrl + 16 + 4 slot,
//      which the proxy reads when it catches up on two doorbells) and then seq (ctrl + 0), system scope;
//   3. wait: one thread a (peer, HCA) spins on flag[peer][slot][hca] == seq (ld.acquire.sys); a wait past spin_limit
//      polls records peer (ctrl + 12), HCA (ctrl + 24), seq (ctrl + 8) and poisons the runtime (device word);
//   4. data: mode 0 sums the local input and every peer slot in rank order (fp32, rank 0's value first, one add a
//      source: torch's ref = parts[0]; ref += parts[r]); mode 1 copies shard s to column block s of every output row;
//   5. epoch: the last block to finish advances the device-resident epoch, so seq is a runtime value and a CUDA graph
//      replays correctly. A failed sequence keeps the epoch: every later launch does nothing until the host raises.
// Staging and tail arrivals have separate counters for each power-of-two grid size (counters[1 + c], counters[1 +
// classes + c]); the poison word follows them. Compiled without fast math: the adds are add.rn.f32, no FTZ.
//
// Why plain CUDA and not the CuTe cubins: the CuTe DSL needs the Python compiler at run time (or a cubin export per
// rank, dtype, world and HCA count - b12x specializes on all four), while these two kernels are ~150 lines whose
// every memory access is inline PTX anyway; written here they build with the rest of the fatbins, take world / rank /
// HCA count as launch arguments, and their SASS can be read next to the rest of the engine's.
#include <cstdint>

namespace {

constexpr int kSlots = 2;          // _roce_proxy.c ROCE_SLOTS
constexpr int kFlagStride = 128;   // ROCE_FLAG_STRIDE
constexpr int kPack = 16;          // bytes a pack

__device__ __forceinline__ unsigned ld_relaxed_gpu(unsigned long long a) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
    return v;
}
__device__ __forceinline__ unsigned ld_relaxed_sys(unsigned long long a) {
    unsigned v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
    return v;
}
__device__ __forceinline__ unsigned atom_add_relaxed_gpu(unsigned long long a, unsigned x) {
    unsigned v;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(v) : "l"(a), "r"(x) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned long long a, unsigned x) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(a), "r"(x) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys(unsigned long long a, unsigned x) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(a), "r"(x) : "memory");
}
__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

// 0 once the word equals `expected`, 1 after `limit` polls without it (b12x spin_until_eq_acquire_sys)
__device__ __forceinline__ unsigned spin_until_eq(unsigned long long a, unsigned expected, unsigned limit) {
    unsigned polls = 0;
    while (true) {
        unsigned seen;
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(seen) : "l"(a) : "memory");
        if (seen == expected) return 0;
        if (++polls >= limit) return 1;
    }
}

__device__ __forceinline__ uint4 ld_v4(unsigned long long a) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a)
                 : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(unsigned long long a) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(a)
                 : "memory");
    return v;
}
__device__ __forceinline__ void st_v4(unsigned long long a, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

}  // namespace

// mode 0: out [packs] fp32 = rank-order sum of every rank's in [packs]; mode 1: all-gather, shard s -> column block s
// of each of the shard's rows (row_packs packs a row; dim-0 concat: row_packs = size_packs). Grid: a power of two
// <= 8 blocks of `threads` threads (threads >= world * hcas: one thread a stripe flag).
extern "C" __global__ void __launch_bounds__(512) glm53_roce_oneshot(
    unsigned long long in, unsigned long long out, int size_packs, int nbytes, int row_packs, int mode,
    unsigned long long recv_base, unsigned long long flag_base, unsigned long long send_base,
    unsigned long long ctrl_base, unsigned long long slot_bytes, unsigned long long epoch_ptr,
    unsigned long long stage_ctr, unsigned long long tail_ctr, unsigned long long poison_ptr, unsigned spin_limit,
    int world, int rank, int hcas) {
    const int tid = threadIdx.x;
    const unsigned gdim = gridDim.x;
    // every block reads the epoch before any can advance it (the advance waits for all blocks at the tail)
    const unsigned epoch = ld_relaxed_gpu(epoch_ptr);
    const unsigned seq = epoch + 1u;
    const unsigned long long slot = seq & 1u;
    const unsigned long long send_slot = send_base + slot * slot_bytes;
    const int index = (int)blockIdx.x * (int)blockDim.x + tid;
    const int stride = (int)gdim * (int)blockDim.x;

    if (ld_relaxed_gpu(poison_ptr) != 0u) return;   // a recorded timeout: later launches do nothing

    // 1. stage the local input into the pinned send slot
    for (int i = index; i < size_packs; i += stride)
        st_v4(send_slot + (unsigned long long)i * kPack, ld_v4(in + (unsigned long long)i * kPack));
    __syncthreads();

    // 2. the last block to finish staging rings the proxy's doorbell
    if (tid == 0) {
        fence_sc_sys();
        const unsigned prior = atom_add_relaxed_gpu(stage_ctr, 1u);
        if ((prior + 1u) % gdim == 0u) {
            st_relaxed_sys(ctrl_base + 4, (unsigned)nbytes);
            st_relaxed_sys(ctrl_base + 16 + slot * 4, (unsigned)nbytes);
            fence_sc_sys();
            st_relaxed_sys(ctrl_base, seq);
        }
    }

    // 3. wait for every peer's payload-stripe flags (one thread a peer and HCA)
    if (tid < world * hcas) {
        const int peer = tid / hcas;
        const int hca = tid - peer * hcas;
        if (peer != rank) {
            const unsigned long long flag =
                flag_base + (((unsigned long long)peer * kSlots + slot) * (unsigned long long)hcas + (unsigned long long)hca) *
                                kFlagStride;
            if (spin_until_eq(flag, seq, spin_limit) != 0u) {
                st_relaxed_sys(ctrl_base + 12, (unsigned)peer);
                st_relaxed_sys(ctrl_base + 24, (unsigned)hca);
                st_relaxed_sys(ctrl_base + 8, seq);
                st_release_gpu(poison_ptr, seq);
            }
        }
    }
    __syncthreads();

    // 4. data, unless a wait of this block timed out (a peer slot is then unreliable)
    if (ld_relaxed_gpu(poison_ptr) == 0u) {
        if (mode == 0) {
            for (int i = index; i < size_packs; i += stride) {
                const unsigned long long off = (unsigned long long)i * kPack;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                for (int s = 0; s < world; ++s) {
                    const uint4 w = s == rank ? ld_v4(in + off)
                                              : ld_relaxed_sys_v4(recv_base + ((unsigned long long)s * kSlots + slot) *
                                                                                  slot_bytes + off);
                    if (s == 0) {
                        a0 = __uint_as_float(w.x);
                        a1 = __uint_as_float(w.y);
                        a2 = __uint_as_float(w.z);
                        a3 = __uint_as_float(w.w);
                    } else {
                        a0 = __fadd_rn(a0, __uint_as_float(w.x));
                        a1 = __fadd_rn(a1, __uint_as_float(w.y));
                        a2 = __fadd_rn(a2, __uint_as_float(w.z));
                        a3 = __fadd_rn(a3, __uint_as_float(w.w));
                    }
                }
                st_v4(out + off, make_uint4(__float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                                            __float_as_uint(a3)));
            }
        } else {
            const int out_row = world * row_packs;
            for (int s = 0; s < world; ++s) {
                for (int i = index; i < size_packs; i += stride) {
                    const int row = i / row_packs;
                    const int col = i - row * row_packs;
                    const unsigned long long dst =
                        out + ((unsigned long long)row * out_row + (unsigned long long)s * row_packs + col) * kPack;
                    const unsigned long long off = (unsigned long long)i * kPack;
                    const uint4 w = s == rank ? ld_v4(in + off)
                                              : ld_relaxed_sys_v4(recv_base + ((unsigned long long)s * kSlots + slot) *
                                                                                  slot_bytes + off);
                    st_v4(dst, w);
                }
            }
        }
    }

    // 5. the last block to finish publishes the next epoch (not after a failed sequence)
    fence_sc_gpu();
    __syncthreads();
    if (tid == 0) {
        const unsigned prior = atom_add_relaxed_gpu(tail_ctr, 1u);
        if ((prior + 1u) % gdim == 0u) {
            fence_sc_gpu();
            if (ld_relaxed_sys(ctrl_base + 8) == 0u) st_release_gpu(epoch_ptr, seq);
        }
    }
}
