// Drama Gated DeltaNet prefill: the canonical 64-token chunked algorithm. Included by drama_backbone.cu after its
// `drama` helpers (bf, tobf, round_bf, grid_dependency); backbone_gdn.zig is the only host driver.
//
// Source algorithm: SGLang 0.5.21 (Apache License 2.0) chunk_gated_delta_rule with use_qk_l2norm_in_kernel=True,
// whose fla/ kernels are adapted from flash-linear-attention (https://github.com/fla-org/flash-linear-attention,
// MIT License), Copyright (c) 2023-2025, Songlin Yang, Yu Zhang. Each stage mirrors one source kernel:
//   drama_gdn_chunk_l2norm  l2norm_fwd(q), l2norm_fwd(k)                         fla/l2norm.py
//   drama_gdn_chunk_gate    fused_gdn_gating, chunk_local_cumsum(g, 64)          fla/fused_gdn_gating.py, fla/cumsum.py
//   drama_gdn_chunk_kkt     chunk_gated_delta_rule_fwd_kkt_solve_kernel           fla/chunk_fwd.py
//   drama_gdn_chunk_wu      recompute_w_u_fwd_kernel                              fla/wy_fast.py
//   drama_gdn_chunk_h       chunk_gated_delta_rule_fwd_kernel_h_blockdim64        fla/chunk_delta_h.py
//   drama_gdn_chunk_o       chunk_fwd_kernel_o                                    fla/chunk_o.py
// with every source-defined BF16 materialization: normalized q/k, beta, A = (I + A)^-1, w/u, bf16(v * beta),
// bf16(k * beta * exp(g)), per-chunk h, v_new, bf16(decayed v_new), bf16(masked q.k) and y. g, gc and the persistent
// state stay fp32.
//
// Arithmetic follows the reference's compiled Triton kernels, not only the Python source:
//   * tl.exp      = mul.f32 x, log2(e); ex2.approx.f32; safe_exp(x) = exp(x <= 0 ? x : -inf) (fla/op.py).
//   * tl.log      = CUDA logf (the softplus of the gate).
//   * sigmoid     = div.full.f32(1, 1 + ex2.approx((0 - b) * log2(e))), rounded to bf16.
//   * l2norm      = the compiled l2norm_fwd_kernel's reduction tree, sqrt.approx.ftz and div.full.
//   * cumsum      = Hillis-Steele inclusive warp scan (shfl.up 1..16) plus warp 0's total for elements 32..63.
//   * every bf16 tl.dot = mma.sync m16n8k16 bf16 -> f32 chained over k in ascending 16-steps from zero.
//   * the block-inverse merge = mma.sync m16n8k8 tf32 on raw fp32 bits (Triton hands fp32 registers to the tf32
//                   MMA); `dot + dot (+ dot)` sums fused into one accumulator chain; negation = 0 - x.
//   * forward substitution (num_warps 1, BK 64): per column fma(Ai[g], a[g], Ai[g+8] * a[g+8]) per lane group g,
//                   then butterfly adds over g ^ 4, g ^ 2, g ^ 1, then b_a + sum.
//   * state update S = fma(S, exp(g_last), dot).
//   * output o = fma(b_o * exp(g), scale, scale * dot), the BT 64 contraction, whose final dot spans 64 rows.
//
// Canonical streaming (backbone_gdn.zig): chunk boundaries are every 64 prefill tokens since the last reset or
// decode, across calls. A call is one segment of `rows` tokens that starts at a chunk boundary; the h kernel reads
// the immutable state_in and writes the state after the segment's last row to a distinct state_out. o writes only
// rows emit_from .. rows - 1, to y row (t - emit_from). Every short segment uses the BT 64 output form, so a row's
// output does not depend on how many later rows of its chunk exist yet.
//
// Dimensions: D = head_k_dim = head_v_dim = 128; NK key heads, NV value heads, NV % NK == 0; value head h reads key
// head h / (NV / NK). NT = ceil(rows / 64).
// Buffers (caller-owned; every workspace element a kernel reads was written earlier in the same segment):
//   qkv   bf16 [rows][ld_qkv]  [q NK x 128 | k NK x 128 | v NV x 128]
//   ba    bf16 [rows][ld_ba]   [b NV | a NV]
//   a_log fp32 [NV]; dt_bias fp32 [NV]
//   state_in, state_out fp32 [NV][128 v][128 k], distinct;  y bf16 [rows - emit_from][NV][128]
//   workspace: qk bf16 [rows][2 NK][128] (normalized q heads then k heads); g, gc, beta fp32 [rows][NV];
//              A bf16 [rows][NV][64]; w, u, vn bf16 [rows][NV][128]; h bf16 [NT][NV][128 v][128 k]
// Launch sequence per segment (stream order; no dynamic shared memory):
//   1 drama_gdn_chunk_l2norm  grid ceil(rows * 2 NK / 16)  block 256  static smem 0
//   2 drama_gdn_chunk_gate    grid (NT, NV)                block 64   static smem 4 B
//   3 drama_gdn_chunk_kkt     grid (NT, NV)                block 32   static smem 34,304 B
//   4 drama_gdn_chunk_wu      grid (NT, NV)                block 128  static smem 40,960 B
//   5 drama_gdn_chunk_h       grid (128 / 32, NV)          block 128  static smem 28,928 B
//   6 drama_gdn_chunk_o       grid (128 / 64, NT, NV)      block 128  static smem 8,448 B
// drama_gdn_chunk_append (grid rows, block 256, no shared memory) fills the unfinished chunk's row cache.
#pragma once

namespace drama_gdn_chunk {

constexpr int D = 128;   // head_k_dim = head_v_dim
constexpr int BT = 64;   // chunk.py CHUNK_SIZE
constexpr int BC = 16;   // chunk_fwd.py BC
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float log2e() { return __uint_as_float(0x3FB8AA3Bu); }
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xFF800000u); }

__device__ __forceinline__ float ex2_approx(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// tl.exp(x): mul.f32 x, log2(e); ex2.approx.f32 (non-ftz).
__device__ __forceinline__ float tl_exp(float x) { return ex2_approx(__fmul_rn(x, log2e())); }

// fla/op.py safe_exp(x) = exp(where(x <= 0, x, -inf)): mul, select, ex2 (ex2(-inf) = +0).
__device__ __forceinline__ float safe_exp(float x) {
    const float m = __fmul_rn(x, log2e());
    return ex2_approx(x <= 0.f ? m : neg_inf());
}

// Sum of squares of a half-warp's 8 values a lane, in the compiled l2norm_fwd_kernel's order: x1 * x1, then fma of
// x0, x2 .. x7, then butterfly adds over lane ^ 8, ^ 4, ^ 2, ^ 1 (own + partner).
__device__ __forceinline__ float reference_sumsq(const float (&x)[8]) {
    float s = __fmul_rn(x[1], x[1]);
    s = __fmaf_rn(x[0], x[0], s);
#pragma unroll
    for (int j = 2; j < 8; ++j) s = __fmaf_rn(x[j], x[j], s);
#pragma unroll
    for (int m = 8; m; m >>= 1) s = __fadd_rn(s, __shfl_xor_sync(FULL, s, m));
    return s;
}

__device__ __forceinline__ float sqrt_approx_ftz(float a) {
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ unsigned bits(__nv_bfloat16 v) { return __bfloat16_as_ushort(v); }
__device__ __forceinline__ unsigned pack2(unsigned lo, unsigned hi) { return (lo & 0xFFFFu) | (hi << 16); }

// One warp: C (16 x 8, fp32) += A (16 x 16 ksteps) * B (16 ksteps x 8), A and B bf16 bit patterns returned by
// a(row, k) and b(k, col); mma.sync m16n8k16, k ascending. g = lane / 4, t = lane % 4.
// Fragments (PTX ISA, mma.m16n8k16 .bf16): A {(g, 2t..2t+1), (g+8, 2t..), (g, 2t+8..), (g+8, 2t+8..)};
// B {(2t..2t+1, g), (2t+8..2t+9, g)}; C c0, c1 = (g, 2t), (g, 2t+1); c2, c3 = (g+8, 2t), (g+8, 2t+1).
template <class FA, class FB>
__device__ __forceinline__ void mma_bf16_chain(float (&c)[4], FA a, FB b, int ksteps) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    for (int s = 0; s < ksteps; ++s) {
        const int k0 = s * 16 + 2 * t;
        const unsigned a0 = pack2(a(g, k0), a(g, k0 + 1));
        const unsigned a1 = pack2(a(g + 8, k0), a(g + 8, k0 + 1));
        const unsigned a2 = pack2(a(g, k0 + 8), a(g, k0 + 9));
        const unsigned a3 = pack2(a(g + 8, k0 + 8), a(g + 8, k0 + 9));
        const unsigned b0 = pack2(b(k0, g), b(k0 + 1, g));
        const unsigned b1 = pack2(b(k0 + 8, g), b(k0 + 9, g));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

__device__ __forceinline__ int c_row(int e) { return ((threadIdx.x & 31) >> 2) + ((e >> 1) << 3); }
__device__ __forceinline__ int c_col(int e) { return ((threadIdx.x & 3) << 1) + (e & 1); }

// One warp: acc (16 x 16 as two n8 tiles) += X (16 x 16) * Y (16 x 16), fp32 row-major in shared memory, as raw
// tf32 operands; mma.sync m16n8k8 .tf32, per tile k-step 0 then 1.
// Fragments (PTX ISA, mma.m16n8k8 .tf32): A {(g, t), (g+8, t), (g, t+4), (g+8, t+4)}; B {(t, g), (t+4, g)}.
__device__ __forceinline__ void mma_tf32_16(float (&acc)[2][4], const float* X, int ldx, const float* Y, int ldy) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
        const unsigned a0 = __float_as_uint(X[g * ldx + ks * 8 + t]);
        const unsigned a1 = __float_as_uint(X[(g + 8) * ldx + ks * 8 + t]);
        const unsigned a2 = __float_as_uint(X[g * ldx + ks * 8 + t + 4]);
        const unsigned a3 = __float_as_uint(X[(g + 8) * ldx + ks * 8 + t + 4]);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const unsigned b0 = __float_as_uint(Y[(ks * 8 + t) * ldy + nt * 8 + g]);
            const unsigned b1 = __float_as_uint(Y[(ks * 8 + t + 4) * ldy + nt * 8 + g]);
            asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(acc[nt][0]), "+f"(acc[nt][1]), "+f"(acc[nt][2]), "+f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
}

__device__ __forceinline__ void zero16(float (&acc)[2][4]) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
}

// dst (16 x 16, ld) = NEG ? 0 - acc : acc
template <bool NEG>
__device__ __forceinline__ void store16(const float (&acc)[2][4], float* dst, int ld) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) dst[c_row(e) * ld + nt * 8 + c_col(e)] = NEG ? __fsub_rn(0.f, acc[nt][e]) : acc[nt][e];
}

// One row of `cols` bf16 values by the whole block: 16 bytes a thread when both rows allow it, else 2 bytes.
__device__ __forceinline__ void copy_row(const __nv_bfloat16* src, __nv_bfloat16* dst, int cols) {
    const bool vec = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0 && (cols & 7) == 0;
    if (vec) {
        const uint4* s = reinterpret_cast<const uint4*>(src);
        uint4* d = reinterpret_cast<uint4*>(dst);
        for (int i = threadIdx.x; i < cols / 8; i += blockDim.x) d[i] = s[i];
    } else {
        const unsigned short* s = reinterpret_cast<const unsigned short*>(src);
        unsigned short* d = reinterpret_cast<unsigned short*>(dst);
        for (int i = threadIdx.x; i < cols; i += blockDim.x) d[i] = s[i];
    }
}

}  // namespace drama_gdn_chunk

// Appends gridDim.x input rows to the unfinished chunk's row cache: qkv row r (qkv_cols values) to
// qkv_cache[at + r] and ba row r (ba_cols values) to ba_cache[at + r]; the caches are dense, qkv_cols and ba_cols
// apart. Grid: rows; block 256. ABI 'p i p i p p i i i' = qkv ld_qkv ba ld_ba qkv_cache ba_cache at qkv_cols ba_cols.
extern "C" __global__ void __launch_bounds__(256) drama_gdn_chunk_append(const __nv_bfloat16* qkv, int ld_qkv,
                                                                       const __nv_bfloat16* ba, int ld_ba,
                                                                       __nv_bfloat16* qkv_cache,
                                                                       __nv_bfloat16* ba_cache, int at, int qkv_cols,
                                                                       int ba_cols) {
    using namespace drama_gdn_chunk;
    grid_dependency();
    const size_t r = blockIdx.x, c = at + r;
    copy_row(qkv + r * ld_qkv, qkv_cache + c * qkv_cols, qkv_cols);
    copy_row(ba + r * ld_ba, ba_cache + c * ba_cols, ba_cols);
}

// 1. l2norm_fwd of every q and k head row (fla/l2norm.py, eps passed as fp32): y = bf16(x / sqrt(sum x^2 + eps)).
// Row r = t * 2 NK + j reads qkv[t][j * 128 ..] (j < NK: q head j, else k head j - NK) and writes qk[r][0..127].
// Half-warp per row, 8 values a lane. ABI 'p i p i i f' = qkv ld_qkv qk rows nk eps.
extern "C" __global__ void __launch_bounds__(256) drama_gdn_chunk_l2norm(const __nv_bfloat16* qkv, int ld_qkv,
                                                                       __nv_bfloat16* qk, int rows, int nk,
                                                                       float eps) {
    using namespace drama_gdn_chunk;
    grid_dependency();
    const int nrow = rows * 2 * nk;
    const int r = static_cast<int>(blockIdx.x) * 16 + (threadIdx.x >> 4);
    const int col = (threadIdx.x & 15) * 8;
    const bool valid = r < nrow;
    const int t = valid ? r / (2 * nk) : 0, j = valid ? r % (2 * nk) : 0;
    const __nv_bfloat16* x = qkv + static_cast<size_t>(t) * ld_qkv + j * D + col;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = valid ? bf(x[i]) : 0.f;  // masked load, other = 0
    const float s = reference_sumsq(v);
    const float n = sqrt_approx_ftz(__fadd_rn(eps, s));
    if (!valid) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) qk[static_cast<size_t>(r) * D + col + i] = tobf(div_full(v[i], n));
}

// 2. fused_gdn_gating (fla/fused_gdn_gating.py) and chunk_local_cumsum (fla/cumsum.py, BT 64, scalar kernel,
// num_warps 8: element i of a chunk is lane i % 32 of warp i / 32). Per token t, value head h:
//   x = a + dt_bias; sp = x <= 20 ? logf(exp(x) + 1) : x; g = (0 - exp(A_log)) * sp;
//   beta = fp32(bf16(div.full(1, exp(0 - b) + 1)));  gc = chunk-local inclusive cumsum of g.
// Grid (NT, NV); block 64. ABI 'p i p p p p p i i' = ba ld_ba a_log dt_bias g gc beta rows nv.
extern "C" __global__ void __launch_bounds__(64) drama_gdn_chunk_gate(const __nv_bfloat16* ba, int ld_ba,
                                                                    const float* a_log, const float* dt_bias,
                                                                    float* g, float* gc, float* beta, int rows,
                                                                    int nv) {
    using namespace drama_gdn_chunk;
    grid_dependency();
    __shared__ float total0;
    const int h = blockIdx.y, i = threadIdx.x, lane = i & 31;
    const int t = static_cast<int>(blockIdx.x) * BT + i;
    const bool valid = t < rows;
    float gv = 0.f;  // masked load of g, other = 0
    if (valid) {
        const float bb = bf(ba[static_cast<size_t>(t) * ld_ba + h]);
        const float aa = bf(ba[static_cast<size_t>(t) * ld_ba + nv + h]);
        const float x = __fadd_rn(aa, dt_bias[h]);
        const float sp = x <= 20.f ? logf(__fadd_rn(tl_exp(x), 1.f)) : x;
        gv = __fmul_rn(__fsub_rn(0.f, tl_exp(a_log[h])), sp);
        const float s = tl_exp(__fsub_rn(0.f, bb));
        const float be = round_bf(div_full(1.f, __fadd_rn(s, 1.f)));
        g[static_cast<size_t>(t) * nv + h] = gv;
        beta[static_cast<size_t>(t) * nv + h] = be;
    }
    float sc = gv;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const float up = __shfl_up_sync(FULL, sc, d);
        if (lane >= d) sc = __fadd_rn(sc, up);
    }
    if (i == 31) total0 = sc;
    __syncthreads();
    if (i >= 32) sc = __fadd_rn(total0, sc);
    if (valid) gc[static_cast<size_t>(t) * nv + h] = sc;
}

// 3. chunk_gated_delta_rule_fwd_kkt_solve_kernel (fla/chunk_fwd.py): for the chunk's 10 lower 16 x 16 blocks
// A_ij = (k_i k_j^T) * safe_exp(g_r - g_c) [diag: strictly lower and valid only] * beta_r; diagonal blocks inverted by
// forward substitution; block merge in tf32; A (I + A)^-1 stored bf16 with zero upper blocks. One warp per
// (chunk, value head), the autotuned num_warps = 1 config. Grid (NT, NV); block 32.
// ABI 'p p p p i i i' = qk gc beta A rows nk nv.
extern "C" __global__ void __launch_bounds__(32) drama_gdn_chunk_kkt(const __nv_bfloat16* qk, const float* gc,
                                                                   const float* beta, __nv_bfloat16* A, int rows,
                                                                   int nk, int nv) {
    using namespace drama_gdn_chunk;
    grid_dependency();
    __shared__ float sA[BT][BT];   // gated, masked, beta-scaled lower blocks
    __shared__ float sI[BT][BT];   // (I + A)^-1 blocks
    __shared__ float sT[BC][BC];   // merge temporary
    __shared__ float sg[BT], sb[BT];
    const int it = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    const int t0 = it * BT, kh = h / (nv / nk);
    for (int i = lane; i < BT; i += 32) {
        const int t = t0 + i;
        sg[i] = t < rows ? gc[static_cast<size_t>(t) * nv + h] : 0.f;
        sb[i] = t < rows ? beta[static_cast<size_t>(t) * nv + h] : 0.f;
    }
    __syncwarp();
    auto kbits = [&](int r, int d) -> unsigned {
        const int t = t0 + r;
        return t < rows ? bits(qk[(static_cast<size_t>(t) * 2 * nk + nk + kh) * D + d]) : 0u;
    };
    // Steps 1-2 (chunk_fwd.py:126-228)
    for (int bi = 0; bi < 4; ++bi) {
        for (int bj = 0; bj <= bi; ++bj) {
            for (int nt = 0; nt < 2; ++nt) {
                float c[4] = {0.f, 0.f, 0.f, 0.f};
                mma_bf16_chain(
                    c, [&](int r, int k) { return kbits(bi * BC + r, k); },
                    [&](int k, int n) { return kbits(bj * BC + nt * 8 + n, k); }, D / 16);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r = c_row(e), cl = nt * 8 + c_col(e);
                    const int R = bi * BC + r, C = bj * BC + cl;
                    float v = __fmul_rn(c[e], safe_exp(__fsub_rn(sg[R], sg[C])));
                    if (bi == bj) v = (r > cl && t0 + R < rows && t0 + C < rows) ? v : 0.f;
                    sA[R][C] = __fmul_rn(v, sb[R]);
                }
            }
        }
    }
    __syncwarp();
    // Step 3 (chunk_fwd.py:230-266): Ai = -A on the diagonal blocks, forward substitution, Ai += I
    for (int d = 0; d < 4; ++d) {
        const int o = d * BC;
        for (int idx = lane; idx < BC * BC; idx += 32) {
            const int r = idx / BC, c = idx % BC;
            sI[o + r][o + c] = -sA[o + r][o + c];
        }
        __syncwarp();
        const int lim = min(BC, rows - (t0 + o));
        for (int i = 2; i < lim; ++i) {
            float a_new = 0.f;
            if (lane < BC) {
                const int c = lane;
                float P[8];
#pragma unroll
                for (int gq = 0; gq < 8; ++gq) {
                    const float ag = gq < i ? -sA[o + i][o + gq] : 0.f;
                    const float ag8 = gq + 8 < i ? -sA[o + i][o + gq + 8] : 0.f;
                    P[gq] = __fmaf_rn(sI[o + gq][o + c], ag, __fmul_rn(sI[o + gq + 8][o + c], ag8));
                }
                const float S = __fadd_rn(__fadd_rn(__fadd_rn(P[0], P[4]), __fadd_rn(P[2], P[6])),
                                          __fadd_rn(__fadd_rn(P[1], P[5]), __fadd_rn(P[3], P[7])));
                const float bc = c < i ? -sA[o + i][o + c] : 0.f;
                a_new = __fadd_rn(bc, S);
            }
            __syncwarp();
            if (lane < BC) sI[o + i][o + lane] = a_new;
            __syncwarp();
        }
        for (int idx = lane; idx < BC * BC; idx += 32) {
            const int r = idx / BC, c = idx % BC;
            sI[o + r][o + c] = __fadd_rn(sI[o + r][o + c], r == c ? 1.f : 0.f);
        }
        __syncwarp();
    }
    // Step 4 (chunk_fwd.py:269-307): block merge, tf32 MMA, source order
    float acc[2][4];
    // Ai10 = -(Ai11 @ A10) @ Ai00
    zero16(acc); mma_tf32_16(acc, &sI[16][16], BT, &sA[16][0], BT); store16<false>(acc, &sT[0][0], BC); __syncwarp();
    zero16(acc); mma_tf32_16(acc, &sT[0][0], BC, &sI[0][0], BT); store16<true>(acc, &sI[16][0], BT); __syncwarp();
    // Ai21 = -(Ai22 @ A21) @ Ai11
    zero16(acc); mma_tf32_16(acc, &sI[32][32], BT, &sA[32][16], BT); store16<false>(acc, &sT[0][0], BC); __syncwarp();
    zero16(acc); mma_tf32_16(acc, &sT[0][0], BC, &sI[16][16], BT); store16<true>(acc, &sI[32][16], BT); __syncwarp();
    // Ai32 = -(Ai33 @ A32) @ Ai22
    zero16(acc); mma_tf32_16(acc, &sI[48][48], BT, &sA[48][32], BT); store16<false>(acc, &sT[0][0], BC); __syncwarp();
    zero16(acc); mma_tf32_16(acc, &sT[0][0], BC, &sI[32][32], BT); store16<true>(acc, &sI[48][32], BT); __syncwarp();
    // Ai20 = -Ai22 @ (A20 @ Ai00 + A21 @ Ai10)
    zero16(acc);
    mma_tf32_16(acc, &sA[32][0], BT, &sI[0][0], BT);
    mma_tf32_16(acc, &sA[32][16], BT, &sI[16][0], BT);
    store16<false>(acc, &sT[0][0], BC); __syncwarp();
    zero16(acc); mma_tf32_16(acc, &sI[32][32], BT, &sT[0][0], BC); store16<true>(acc, &sI[32][0], BT); __syncwarp();
    // Ai31 = -Ai33 @ (A31 @ Ai11 + A32 @ Ai21)
    zero16(acc);
    mma_tf32_16(acc, &sA[48][16], BT, &sI[16][16], BT);
    mma_tf32_16(acc, &sA[48][32], BT, &sI[32][16], BT);
    store16<false>(acc, &sT[0][0], BC); __syncwarp();
    zero16(acc); mma_tf32_16(acc, &sI[48][48], BT, &sT[0][0], BC); store16<true>(acc, &sI[48][16], BT); __syncwarp();
    // Ai30 = -Ai33 @ (A30 @ Ai00 + A31 @ Ai10 + A32 @ Ai20)
    zero16(acc);
    mma_tf32_16(acc, &sA[48][0], BT, &sI[0][0], BT);
    mma_tf32_16(acc, &sA[48][16], BT, &sI[16][0], BT);
    mma_tf32_16(acc, &sA[48][32], BT, &sI[32][0], BT);
    store16<false>(acc, &sT[0][0], BC); __syncwarp();
    zero16(acc); mma_tf32_16(acc, &sI[48][48], BT, &sT[0][0], BC); store16<true>(acc, &sI[48][0], BT); __syncwarp();
    // Step 5 (chunk_fwd.py:309-338): rows < rows only; upper blocks are the source's torch.zeros
    for (int idx = lane; idx < BT * BT; idx += 32) {
        const int R = idx / BT, C = idx % BT, t = t0 + R;
        if (t >= rows) continue;
        const float v = (C / BC <= R / BC) ? sI[R][C] : 0.f;
        A[(static_cast<size_t>(t) * nv + h) * BT + C] = tobf(v);
    }
}

// 4. recompute_w_u_fwd_kernel (fla/wy_fast.py): u = A @ bf16(v * beta); w = A @ bf16(k * beta * exp(gc)).
// v is the raw value row (qkv), k the normalized key head. Grid (NT, NV); block 128 (4 warps, 128 output tiles).
// ABI 'p i p p p p p p i i i' = qkv ld_qkv qk gc beta A w u rows nk nv.
extern "C" __global__ void __launch_bounds__(128) drama_gdn_chunk_wu(const __nv_bfloat16* qkv, int ld_qkv,
                                                                   const __nv_bfloat16* qk, const float* gc,
                                                                   const float* beta, const __nv_bfloat16* A,
                                                                   __nv_bfloat16* w, __nv_bfloat16* u, int rows,
                                                                   int nk, int nv) {
    using namespace drama_gdn_chunk;
    grid_dependency();
    __shared__ unsigned short sAb[BT][BT];
    __shared__ unsigned short sV[BT][D];
    __shared__ unsigned short sK[BT][D];
    const int it = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5;
    const int t0 = it * BT, kh = h / (nv / nk);
    for (int idx = tid; idx < BT * BT; idx += blockDim.x) {
        const int r = idx / BT, c = idx % BT, t = t0 + r;
        sAb[r][c] = t < rows ? static_cast<unsigned short>(bits(A[(static_cast<size_t>(t) * nv + h) * BT + c])) : 0;
    }
    for (int idx = tid; idx < BT * D; idx += blockDim.x) {
        const int r = idx / D, d = idx % D, t = t0 + r;
        unsigned short vb = 0, kb = 0;
        if (t < rows) {
            const float be = beta[static_cast<size_t>(t) * nv + h];
            const float eg = tl_exp(gc[static_cast<size_t>(t) * nv + h]);
            const float v = bf(qkv[static_cast<size_t>(t) * ld_qkv + 2 * nk * D + h * D + d]);
            const float k = bf(qk[(static_cast<size_t>(t) * 2 * nk + nk + kh) * D + d]);
            vb = static_cast<unsigned short>(bits(tobf(__fmul_rn(v, be))));
            kb = static_cast<unsigned short>(bits(tobf(__fmul_rn(__fmul_rn(k, be), eg))));
        }
        sV[r][d] = vb;
        sK[r][d] = kb;
    }
    __syncthreads();
    for (int tile = warp; tile < 2 * (BT / 16) * (D / 8); tile += 4) {
        const bool is_w = tile >= (BT / 16) * (D / 8);
        const int local = is_w ? tile - (BT / 16) * (D / 8) : tile;
        const int mt = local / (D / 8), nt = local % (D / 8);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        if (is_w) {
            mma_bf16_chain(
                c, [&](int r, int k) { return static_cast<unsigned>(sAb[mt * 16 + r][k]); },
                [&](int k, int n) { return static_cast<unsigned>(sK[k][nt * 8 + n]); }, BT / 16);
        } else {
            mma_bf16_chain(
                c, [&](int r, int k) { return static_cast<unsigned>(sAb[mt * 16 + r][k]); },
                [&](int k, int n) { return static_cast<unsigned>(sV[k][nt * 8 + n]); }, BT / 16);
        }
        __nv_bfloat16* out = is_w ? w : u;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = t0 + mt * 16 + c_row(e);
            if (t < rows) out[(static_cast<size_t>(t) * nv + h) * D + nt * 8 + c_col(e)] = tobf(c[e]);
        }
    }
}

// 5. chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (fla/chunk_delta_h.py), BV 32, fp32 state slice in shared
// memory for the whole segment. S = state_in; per chunk: h[it] = bf16(S); b_v = u - w @ bf16(S)^T; vn = bf16(b_v);
// b_v *= safe_exp(g_last - g); S = fma(S, exp(g_last), k^T @ bf16(b_v)). The final S goes to state_out; state_in is
// never written, so the two must be distinct. Grid (128 / 32, NV); block 128.
// ABI 'p p p p p p p p i i i' = qk w u gc state_in state_out h vn rows nk nv.
extern "C" __global__ void __launch_bounds__(128) drama_gdn_chunk_h(const __nv_bfloat16* qk, const __nv_bfloat16* w,
                                                                  const __nv_bfloat16* u, const float* gc,
                                                                  const float* state_in, float* state_out,
                                                                  __nv_bfloat16* hout, __nv_bfloat16* vn, int rows,
                                                                  int nk, int nv) {
    using namespace drama_gdn_chunk;
    grid_dependency();
    constexpr int BV = 32;
    __shared__ float sS[BV][D];
    __shared__ unsigned short sH[BV][D];
    __shared__ unsigned short sVb[BT][BV];
    __shared__ float sG[BT];
    const int v0 = blockIdx.x * BV, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5;
    const int kh = h / (nv / nk), NT = (rows + BT - 1) / BT;
    const float* s_in = state_in + static_cast<size_t>(h) * D * D;
    float* s_out = state_out + static_cast<size_t>(h) * D * D;
    for (int idx = tid; idx < BV * D; idx += blockDim.x) {
        const int vr = idx / D, k = idx % D;
        sS[vr][k] = __fadd_rn(0.f, s_in[static_cast<size_t>(v0 + vr) * D + k]);  // b_h = zeros; b_h += load
    }
    for (int it = 0; it < NT; ++it) {
        const int t0 = it * BT;
        __syncthreads();
        for (int idx = tid; idx < BV * D; idx += blockDim.x) {
            const int vr = idx / D, k = idx % D;
            const __nv_bfloat16 hb = tobf(sS[vr][k]);
            sH[vr][k] = static_cast<unsigned short>(bits(hb));
            hout[((static_cast<size_t>(it) * nv + h) * D + v0 + vr) * D + k] = hb;
        }
        for (int i = tid; i < BT; i += blockDim.x) {
            const int t = t0 + i;
            sG[i] = t < rows ? gc[static_cast<size_t>(t) * nv + h] : 0.f;
        }
        __syncthreads();
        const float glast = gc[static_cast<size_t>(min(t0 + BT, rows) - 1) * nv + h];
        const float eg = tl_exp(glast);
        // b_v [64 t][32 v] = w @ bf16(S)^T, k ascending 0..127 (chunk_delta_h.py:213-231, fused accumulators)
        for (int tile = warp; tile < (BT / 16) * (BV / 8); tile += 4) {
            const int mt = tile / (BV / 8), nt = tile % (BV / 8);
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16_chain(
                c,
                [&](int r, int k) {
                    const int t = t0 + mt * 16 + r;
                    return t < rows ? bits(w[(static_cast<size_t>(t) * nv + h) * D + k]) : 0u;
                },
                [&](int k, int n) { return static_cast<unsigned>(sH[nt * 8 + n][k]); }, D / 16);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = mt * 16 + c_row(e), vv = nt * 8 + c_col(e), t = t0 + rr;
                const bool valid = t < rows;
                const float uu = valid ? bf(u[(static_cast<size_t>(t) * nv + h) * D + v0 + vv]) : 0.f;
                float bv = __fsub_rn(uu, c[e]);
                if (valid) vn[(static_cast<size_t>(t) * nv + h) * D + v0 + vv] = tobf(bv);
                bv = __fmul_rn(bv, safe_exp(__fsub_rn(glast, sG[rr])));
                sVb[rr][vv] = static_cast<unsigned short>(bits(tobf(bv)));
            }
        }
        __syncthreads();
        // S[v][k] = fma(S, exp(g_last), sum_t k[t][k] * bf16(b_v)[t][v]), t ascending (chunk_delta_h.py:250-310)
        for (int tile = warp; tile < (D / 16) * (BV / 8); tile += 4) {
            const int mt = tile / (BV / 8), nt = tile % (BV / 8);
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16_chain(
                c,
                [&](int r, int tt) {
                    const int t = t0 + tt;
                    return t < rows ? bits(qk[(static_cast<size_t>(t) * 2 * nk + nk + kh) * D + mt * 16 + r]) : 0u;
                },
                [&](int tt, int n) { return static_cast<unsigned>(sVb[tt][nt * 8 + n]); }, BT / 16);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kd = mt * 16 + c_row(e), vv = nt * 8 + c_col(e);
                sS[vv][kd] = __fmaf_rn(sS[vv][kd], eg, c[e]);
            }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < BV * D; idx += blockDim.x) {
        const int vr = idx / D, k = idx % D;
        s_out[static_cast<size_t>(v0 + vr) * D + k] = sS[vr][k];
    }
}

// 6. chunk_fwd_kernel_o (fla/chunk_o.py), BV 64. b_o = (q @ h[it]^T) * exp(g); b_A = (q @ k^T) * safe_exp(g_i - g_j),
// j <= i, bf16; o = bf16(fma(b_o, scale, scale * (b_A @ vn))), the final dot over all 64 rows of the chunk (rows past
// the segment are zero), for every segment length. Only rows emit_from .. rows - 1 are written, to y row
// (t - emit_from); a warp with none of them skips its products. Grid (128 / 64, NT, NV); block 128 (warp w owns rows
// 16 w .. 16 w + 15 of its chunk). ABI 'p p p p p i i i i f' = qk vn h gc y rows emit_from nk nv scale.
extern "C" __global__ void __launch_bounds__(128) drama_gdn_chunk_o(const __nv_bfloat16* qk, const __nv_bfloat16* vn,
                                                                  const __nv_bfloat16* hin, const float* gc,
                                                                  __nv_bfloat16* y, int rows, int emit_from, int nk,
                                                                  int nv, float scale) {
    using namespace drama_gdn_chunk;
    grid_dependency();
    constexpr int BV = 64;
    __shared__ unsigned short sAo[BT][BT];
    __shared__ float sG[BT];
    const int vb0 = blockIdx.x * BV, it = blockIdx.y, h = blockIdx.z, tid = threadIdx.x, warp = tid >> 5;
    const int t0 = it * BT, kh = h / (nv / nk);
    const int w0 = t0 + warp * 16;
    const bool active = w0 < rows && w0 + 16 > emit_from;
    for (int i = tid; i < BT; i += blockDim.x) {
        const int t = t0 + i;
        sG[i] = t < rows ? gc[static_cast<size_t>(t) * nv + h] : 0.f;
    }
    __syncthreads();
    auto qbits = [&](int r, int d) -> unsigned {
        const int t = w0 + r;
        return t < rows ? bits(qk[(static_cast<size_t>(t) * 2 * nk + kh) * D + d]) : 0u;
    };
    float bo[BV / 8][4];
    if (active) {
        // b_A tiles of this warp's rows; masked, gated, bf16 into shared memory
        for (int nt = 0; nt < BT / 8; ++nt) {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            mma_bf16_chain(
                c, qbits,
                [&](int d, int n) {
                    const int t = t0 + nt * 8 + n;
                    return t < rows ? bits(qk[(static_cast<size_t>(t) * 2 * nk + nk + kh) * D + d]) : 0u;
                },
                D / 16);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = warp * 16 + c_row(e), j = nt * 8 + c_col(e);
                float v = __fmul_rn(c[e], safe_exp(__fsub_rn(sG[i], sG[j])));
                v = i >= j ? v : 0.f;
                sAo[i][j] = static_cast<unsigned short>(bits(tobf(v)));
            }
        }
        // b_o tiles (registers): q @ h^T, k ascending, then * exp(g)
        const __nv_bfloat16* h0 = hin + (static_cast<size_t>(it) * nv + h) * D * D;
        for (int nt = 0; nt < BV / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) bo[nt][e] = 0.f;
            mma_bf16_chain(
                bo[nt], qbits,
                [&](int d, int n) { return bits(h0[static_cast<size_t>(vb0 + nt * 8 + n) * D + d]); }, D / 16);
#pragma unroll
            for (int e = 0; e < 4; ++e) bo[nt][e] = __fmul_rn(bo[nt][e], tl_exp(sG[warp * 16 + c_row(e)]));
        }
    }
    __syncthreads();
    if (!active) return;
    for (int nt = 0; nt < BV / 8; ++nt) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        mma_bf16_chain(
            c, [&](int r, int j) { return static_cast<unsigned>(sAo[warp * 16 + r][j]); },
            [&](int j, int n) {
                const int t = t0 + j;
                return t < rows ? bits(vn[(static_cast<size_t>(t) * nv + h) * D + vb0 + nt * 8 + n]) : 0u;
            },
            BT / 16);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = w0 + c_row(e);
            const float o = __fmaf_rn(bo[nt][e], scale, __fmul_rn(scale, c[e]));
            if (t < rows && t >= emit_from)
                y[(static_cast<size_t>(t - emit_from) * nv + h) * D + vb0 + nt * 8 + c_col(e)] = tobf(o);
        }
    }
}
