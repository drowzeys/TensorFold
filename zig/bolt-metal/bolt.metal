// Nemotron 3.5 Bolt on Metal: the kernels of libbolt's Metal engine, ported from MiaAI-Lab's CUDA engine
// (Nemotron-3.5-Bolt-TensorFold-DGX-Spark, engine/bolt_kernels.cu and bolt_draft.cu).
// Weights stay packed (Q8_0, NVFP4, IQ2_XXS, bf16) and are decoded in registers; activations are fp32.
// The host prepends bolt_args.h and iq2_grid.h to this file before compiling it.
#include <metal_stdlib>
using namespace metal;


inline float bf(ushort v) { return as_type<float>(uint(v) << 16); }
inline ushort to_bf(float f) {
    uint u = as_type<uint>(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return ushort(u >> 16);
}
inline float e4m3(uint b) {
    uint e = (b >> 3) & 0xFu, m = b & 7u;
    float v = e == 0 ? float(m) * 0.001953125f : as_type<float>(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}
// e2m1 nibble -> float without a table: magnitudes 2..7 put their bits in the exponent and the mantissa's top bit
// (exponent biased by 126), 0 and 1 are 0 and 0.5 (no denormal step: fast math would flush it); bit 3 is the sign.
inline float fp4(uint nib) {
    const uint m = nib & 7u;
    const float v = m < 2u ? 0.5f * float(m) : as_type<float>((m << 22) + (126u << 23));
    return (nib & 8u) ? -v : v;
}
inline float silu(float v) { return v / (1.f + exp(-v)); }
inline float softplus(float v) { return v > 20.f ? v : log(1.f + exp(v)); }
inline bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

inline int seg_of(device const Ctl* ctl, int row) {
    int i = 0;
    while (i + 1 < ctl->nseg && row >= ctl->seg_row0[i + 1]) i++;
    return i;
}

// ---------------------------------------------------------------------------------------------------------
// Format policies. A chunk is 32 consecutive weights of one row; a lane owns every 32nd chunk.
// ---------------------------------------------------------------------------------------------------------

struct FQ8 {  // planar Q8_0: int8 [N][K], half [N][K/32]
    struct C { uint4 a, b; float d; };
    enum { grid = 0 };
    static C load(constant GemvArgs& g, int row, int c) {
        C ch;
        device const uint4* p = (device const uint4*)(g.w + (ulong)row * g.k + (ulong)c * 32);
        ch.a = p[0];
        ch.b = p[1];
        ch.d = float(((device const half*)g.s)[(ulong)row * (g.k >> 5) + c]);
        return ch;
    }
    static void deq(thread const C& ch, int sub, thread float4& lo, thread float4& hi, threadgroup const uint2*) {
        uint x = sub == 0 ? ch.a.x : sub == 1 ? ch.a.z : sub == 2 ? ch.b.x : ch.b.z;
        uint y = sub == 0 ? ch.a.y : sub == 1 ? ch.a.w : sub == 2 ? ch.b.y : ch.b.w;
        lo = float4(as_type<char4>(x)) * ch.d;
        hi = float4(as_type<char4>(y)) * ch.d;
    }
};

struct FNV4 {  // ModelOpt NVFP4: q [N][K/2] low nibble first, e4m3 scales [N][K/16], one fp32 tensor scale
    struct C { uint4 q; float s0, s1; };
    enum { grid = 0 };
    static C load(constant GemvArgs& g, int row, int c) {
        C ch;
        ch.q = *(device const uint4*)(g.w + (ulong)row * (g.k >> 1) + (ulong)c * 16);
        device const uchar* sp = g.s + (ulong)row * (g.k >> 4) + (ulong)c * 2;
        ch.s0 = e4m3(sp[0]) * g.s2;
        ch.s1 = e4m3(sp[1]) * g.s2;
        return ch;
    }
    static void deq(thread const C& ch, int sub, thread float4& lo, thread float4& hi, threadgroup const uint2*) {
        uint w = sub == 0 ? ch.q.x : sub == 1 ? ch.q.y : sub == 2 ? ch.q.z : ch.q.w;
        float s = sub < 2 ? ch.s0 : ch.s1;
        lo = float4(fp4(w), fp4(w >> 4), fp4(w >> 8), fp4(w >> 12)) * s;
        hi = float4(fp4(w >> 16), fp4(w >> 20), fp4(w >> 24), fp4(w >> 28)) * s;
    }
};

inline void iq2_deq(uint2 q, float db, int sub, thread float4& lo, thread float4& hi, threadgroup const uint2* grid) {
    uint2 gv = grid[(q.x >> (8 * sub)) & 0xffu];
    uint si = (q.y >> (7 * sub)) & 127u;
    uint sg = si | ((popcount(si) & 1u) << 7);
    float4 a = float4(as_type<uchar4>(gv.x)), b = float4(as_type<uchar4>(gv.y));
    float4 sa = float4((sg & 1u) ? -1.f : 1.f, (sg & 2u) ? -1.f : 1.f, (sg & 4u) ? -1.f : 1.f, (sg & 8u) ? -1.f : 1.f);
    float4 sb = float4((sg & 16u) ? -1.f : 1.f, (sg & 32u) ? -1.f : 1.f, (sg & 64u) ? -1.f : 1.f, (sg & 128u) ? -1.f : 1.f);
    lo = a * sa * db;
    hi = b * sb * db;
}

struct FIQ2 {  // planar IQ2_XXS: qs [N][K/256][64] bytes, half d [N][K/256]
    struct C { uint2 q; float db; };
    enum { grid = 1 };
    static C load(constant GemvArgs& g, int row, int c) {
        C ch;
        int nb = g.k >> 8;
        ch.q = ((device const uint2*)(g.w + (ulong)row * nb * 64))[c];
        float d = float(((device const half*)g.s)[(ulong)row * nb + (c >> 3)]);
        ch.db = d * (0.5f + float(ch.q.y >> 28)) * 0.25f;
        return ch;
    }
    static void deq(thread const C& ch, int sub, thread float4& lo, thread float4& hi, threadgroup const uint2* grid) {
        iq2_deq(ch.q, ch.db, sub, lo, hi, grid);
    }
};

struct FBF {  // row-major bf16
    struct C { uint4 a, b, c, d; };
    enum { grid = 0 };
    static C load(constant GemvArgs& g, int row, int c) {
        C ch;
        device const uint4* p = (device const uint4*)(g.w + ((ulong)row * g.k + (ulong)c * 32) * 2);
        ch.a = p[0]; ch.b = p[1]; ch.c = p[2]; ch.d = p[3];
        return ch;
    }
    static void deq(thread const C& ch, int sub, thread float4& lo, thread float4& hi, threadgroup const uint2*) {
        uint4 v = sub == 0 ? ch.a : sub == 1 ? ch.b : sub == 2 ? ch.c : ch.d;
        lo = float4(as_type<float>(v.x << 16), as_type<float>(v.x & 0xffff0000u), as_type<float>(v.y << 16), as_type<float>(v.y & 0xffff0000u));
        hi = float4(as_type<float>(v.z << 16), as_type<float>(v.z & 0xffff0000u), as_type<float>(v.w << 16), as_type<float>(v.w & 0xffff0000u));
    }
};

inline float epilogue(float v, int epi) {
    if (epi == 1) return v > 0.f ? v * v : 0.f;
    if (epi == 2) return silu(v);
    return v;
}

// T activation rows at a time, R weight rows per simdgroup, four simdgroups a threadgroup. The sum over K runs in
// one fixed order per row whatever T is, so a row's result does not depend on the rows beside it.
template <typename F, int TT, int R>
kernel void gemv(constant GemvArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]],
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                 uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint2 grid_s[256];
    if (F::grid) {
        for (uint i = tid; i < 256; i += 128) grid_s[i] = IQ2XXS_GRID[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int row0 = (int)tgp.x * 4 * R + (int)sg * R;
    if (row0 >= a.n) return;
    const int nchunk = a.k >> 5;
    for (int t0 = 0; t0 < a.T; t0 += TT) {
        float acc[R][TT];
        for (int r = 0; r < R; r++)
            for (int t = 0; t < TT; t++) acc[r][t] = 0.f;
        for (int c = lane; c < nchunk; c += 32) {
            typename F::C ch[R];
            for (int r = 0; r < R; r++) ch[r] = F::load(a, min(row0 + r, a.n - 1), c);
            for (int sub = 0; sub < 4; sub++) {
                float4 wl[R], wh[R];
                for (int r = 0; r < R; r++) F::deq(ch[r], sub, wl[r], wh[r], grid_s);
                for (int t = 0; t < TT; t++) {
                    const int tt = min(t0 + t, a.T - 1);
                    device const float4* xp = (device const float4*)(a.x + (ulong)tt * a.ldx + (ulong)c * 32 + sub * 8);
                    const float4 xa = xp[0], xb = xp[1];
                    for (int r = 0; r < R; r++) acc[r][t] += dot(wl[r], xa) + dot(wh[r], xb);
                }
            }
        }
        for (int r = 0; r < R; r++)
            for (int t = 0; t < TT; t++) {
                float v = simd_sum(acc[r][t]);
                const int row = row0 + r;
                if (lane == 0 && row < a.n && t0 + t < a.T) {
                    if (a.bias) v += a.bias[row];
                    v = epilogue(v, a.epi);
                    device float* yp = a.y + (ulong)(t0 + t) * a.ldy + row;
                    *yp = a.accum ? *yp + v : v;
                }
            }
    }
}

typedef decltype(gemv<FQ8, 1, 4>) gemv_t;
template [[host_name("gemv_q8_1")]] kernel gemv_t gemv<FQ8, 1, 4>;
template [[host_name("gemv_q8_2")]] kernel gemv_t gemv<FQ8, 2, 4>;
template [[host_name("gemv_q8_4")]] kernel gemv_t gemv<FQ8, 4, 2>;
template [[host_name("gemv_q8_8")]] kernel gemv_t gemv<FQ8, 8, 2>;
template [[host_name("gemv_nv4_1")]] kernel gemv_t gemv<FNV4, 1, 4>;
template [[host_name("gemv_nv4_2")]] kernel gemv_t gemv<FNV4, 2, 4>;
template [[host_name("gemv_nv4_4")]] kernel gemv_t gemv<FNV4, 4, 2>;
template [[host_name("gemv_nv4_8")]] kernel gemv_t gemv<FNV4, 8, 2>;
template [[host_name("gemv_iq2_1")]] kernel gemv_t gemv<FIQ2, 1, 4>;
template [[host_name("gemv_iq2_2")]] kernel gemv_t gemv<FIQ2, 2, 4>;
template [[host_name("gemv_iq2_4")]] kernel gemv_t gemv<FIQ2, 4, 2>;
template [[host_name("gemv_iq2_8")]] kernel gemv_t gemv<FIQ2, 8, 2>;
template [[host_name("gemv_bf_1")]] kernel gemv_t gemv<FBF, 1, 4>;
template [[host_name("gemv_bf_2")]] kernel gemv_t gemv<FBF, 2, 4>;
template [[host_name("gemv_bf_4")]] kernel gemv_t gemv<FBF, 4, 2>;
template [[host_name("gemv_bf_8")]] kernel gemv_t gemv<FBF, 8, 2>;
template [[host_name("gemv_q8_4r4")]] kernel gemv_t gemv<FQ8, 4, 4>;
template [[host_name("gemv_q8_8r4")]] kernel gemv_t gemv<FQ8, 8, 4>;
template [[host_name("gemv_q8_8r8")]] kernel gemv_t gemv<FQ8, 8, 8>;
template [[host_name("gemv_nv4_4r4")]] kernel gemv_t gemv<FNV4, 4, 4>;
template [[host_name("gemv_nv4_8r4")]] kernel gemv_t gemv<FNV4, 8, 4>;
template [[host_name("gemv_nv4_8r8")]] kernel gemv_t gemv<FNV4, 8, 8>;
template [[host_name("gemv_iq2_4r4")]] kernel gemv_t gemv<FIQ2, 4, 4>;
template [[host_name("gemv_iq2_8r4")]] kernel gemv_t gemv<FIQ2, 8, 4>;
template [[host_name("gemv_iq2_8r8")]] kernel gemv_t gemv<FIQ2, 8, 8>;
template [[host_name("gemv_bf_4r4")]] kernel gemv_t gemv<FBF, 4, 4>;
template [[host_name("gemv_bf_8r4")]] kernel gemv_t gemv<FBF, 8, 4>;
template [[host_name("gemv_bf_8r8")]] kernel gemv_t gemv<FBF, 8, 8>;

// ---------------------------------------------------------------------------------------------------------
// Routed experts
// ---------------------------------------------------------------------------------------------------------

// Group the (token, expert) pairs by expert and cut them into work items of at most item_b pairs. One threadgroup
// of 512 threads; experts < 512.
kernel void moe_group(constant GroupArgs& a [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_int cnt[512];
    threadgroup atomic_int fill[512];
    threadgroup int start[512], act[512], sstart[512], slen[512], sexp[512], nbat[512];
    atomic_store_explicit(&cnt[tid], 0, memory_order_relaxed);
    atomic_store_explicit(&fill[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = tid; p < a.pairs; p += 512) atomic_fetch_add_explicit(&cnt[a.ridx[p]], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int mycnt = atomic_load_explicit(&cnt[tid], memory_order_relaxed);
    start[tid] = mycnt;
    act[tid] = mycnt > 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < 512; off <<= 1) {
        int x = tid >= off ? start[tid - off] : 0, y = tid >= off ? act[tid - off] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        start[tid] += x;
        act[tid] += y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int pos0 = start[tid] - mycnt;
    const int slot = act[tid] - (mycnt > 0);
    const int nact = act[511];
    sstart[tid] = 0; slen[tid] = 0; sexp[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (mycnt > 0) { sstart[slot] = pos0; slen[slot] = mycnt; sexp[slot] = tid; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    nbat[tid] = (int)tid < nact ? (slen[tid] + a.item_b - 1) / a.item_b : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int mine = nbat[tid];
    for (uint off = 1; off < 512; off <<= 1) {
        int x = tid >= off ? nbat[tid - off] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nbat[tid] += x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if ((int)tid < nact) {
        int w0 = nbat[tid] - mine;
        for (int i = 0; i < mine; i++) {
            a.wfirst[w0 + i] = sstart[tid] + i * a.item_b;
            a.wcount[w0 + i] = min(a.item_b, slen[tid] - i * a.item_b);
            a.wexp[w0 + i] = sexp[tid];
        }
    }
    if (tid == 511) a.nwork[0] = nbat[511];
    start[tid] = pos0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the order of pairs within an expert varies run to run; each pair's sum does not depend on it
    for (int p = tid; p < a.pairs; p += 512) {
        int e = a.ridx[p];
        a.order[start[e] + atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed)] = p;
    }
}

// One work item per threadgroup row of the grid: expert e's rows in tiles of 4*R, the item's (up to TP) pairs.
template <int TP, int R>
kernel void moe_gemv(constant MoeArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                     uint tid [[thread_index_in_threadgroup]]) {
    if ((int)tgp.y >= a.nwork[0]) return;
    threadgroup uint2 grid_s[256];
    for (uint i = tid; i < 256; i += 128) grid_s[i] = IQ2XXS_GRID[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int e = a.wexp[tgp.y], first = a.wfirst[tgp.y], cnt = a.wcount[tgp.y];
    int pid[TP];
    for (int i = 0; i < TP; i++) pid[i] = a.order[first + min(i, cnt - 1)];
    const int row0 = (int)tgp.x * 4 * R + (int)sg * R;
    if (row0 >= a.n) return;
    const int nb = a.k >> 8, nchunk = a.k >> 5;
    device const uchar* qbase = a.qs + (ulong)e * a.qs_stride;
    device const half* dbase = a.d + (ulong)e * a.d_stride;
    float acc[R][TP];
    for (int r = 0; r < R; r++)
        for (int t = 0; t < TP; t++) acc[r][t] = 0.f;
    for (int c = lane; c < nchunk; c += 32) {
        uint2 q[R];
        float db[R];
        for (int r = 0; r < R; r++) {
            const int row = min(row0 + r, a.n - 1);
            q[r] = ((device const uint2*)(qbase + (ulong)row * nb * 64))[c];
            db[r] = float(dbase[(ulong)row * nb + (c >> 3)]) * (0.5f + float(q[r].y >> 28)) * 0.25f;
        }
        for (int sub = 0; sub < 4; sub++) {
            float4 wl[R], wh[R];
            for (int r = 0; r < R; r++) iq2_deq(q[r], db[r], sub, wl[r], wh[r], grid_s);
            for (int t = 0; t < TP; t++) {
                if (t >= cnt) break;
                const int xr = a.x_per_pair ? pid[t] : pid[t] / BOLT_TOPN;
                device const float4* xp = (device const float4*)(a.x + (ulong)xr * a.ldx + (ulong)c * 32 + sub * 8);
                const float4 xa = xp[0], xb = xp[1];
                for (int r = 0; r < R; r++) acc[r][t] += dot(wl[r], xa) + dot(wh[r], xb);
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int t = 0; t < TP; t++) {
            if (t >= cnt) break;
            float v = simd_sum(acc[r][t]);
            const int row = row0 + r;
            if (lane == 0 && row < a.n) a.y[(ulong)pid[t] * a.ldy + row] = epilogue(v, a.epi);
        }
}
typedef decltype(moe_gemv<1, 4>) moe_t;
template [[host_name("moe_gemv_1")]] kernel moe_t moe_gemv<1, 4>;
template [[host_name("moe_gemv_8")]] kernel moe_t moe_gemv<8, 2>;
template [[host_name("moe_gemv_8r4")]] kernel moe_t moe_gemv<8, 4>;

// ---------------------------------------------------------------------------------------------------------
// Block building blocks
// ---------------------------------------------------------------------------------------------------------

kernel void embed(constant EmbedArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int t = tgp.x;
    const int id = a.ctl->tokens[t];
    device const ushort* row = id < 0 ? a.extra + (ulong)t * BOLT_HID
                                      : a.table + (ulong)((uint)id >= (uint)BOLT_VOCAB ? 0 : id) * BOLT_HID;
    for (int i = tid; i < BOLT_HID; i += 256) a.x[(ulong)t * BOLT_HID + i] = bf(row[i]);
}

// x += r1 + r2 (when given), xn = rmsnorm(x) * w; cap (optional) receives the updated residual.
kernel void add_rms(constant AddRmsArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    const int t = tgp.x;
    device float4* xp = (device float4*)(a.x + (ulong)t * BOLT_HID);
    device const float4* p1 = a.r1 ? (device const float4*)(a.r1 + (ulong)t * BOLT_HID) : nullptr;
    device const float4* p2 = a.r2 ? (device const float4*)(a.r2 + (ulong)t * BOLT_HID) : nullptr;
    float4 v[4];
    float ss = 0.f;
    for (int j = 0; j < 4; j++) {
        int i = tid + j * 256;
        float4 u = xp[i];
        if (p1) u += p1[i];
        if (p2) u += p2[i];
        v[j] = u;
        ss += dot(u, u);
    }
    threadgroup float sm[8];
    ss = simd_sum(ss);
    if (lane == 0) sm[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int i = 0; i < 8; i++) tot += sm[i];
    const float inv = rsqrt(tot / float(BOLT_HID) + a.eps);
    device float4* np = (device float4*)(a.xn + (ulong)t * BOLT_HID);
    device const float4* wp = (device const float4*)a.w;
    device float4* cp = a.cap ? (device float4*)(a.cap + (ulong)t * a.cap_stride) : nullptr;
    for (int j = 0; j < 4; j++) {
        int i = tid + j * 256;
        xp[i] = v[j];
        np[i] = v[j] * inv * wp[i];
        if (cp) cp[i] = v[j];
    }
}

// Mamba conv1d (kernel 4, causal, SiLU), one thread per channel and segment. A record block writes the state after
// every row into the following ring slots (a window can be cut after any row), otherwise the final state only.
kernel void conv(constant ConvArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int c = tgp.x * 256 + tid;
    if (c >= 10240) return;
    const int sgi = tgp.y;
    device const Ctl* ctl = a.ctl;
    if (sgi >= ctl->nseg) return;
    const int r0 = ctl->seg_row0[sgi], r1 = ctl->seg_row0[sgi + 1];
    device float* ring = a.pool + (ulong)ctl->seg_seq[sgi] * BOLT_CONV_SLOTS * BOLT_CONV_ELEMS;
    const int cur = ctl->seg_conv[sgi];
    device const float* in = ring + (ulong)cur * BOLT_CONV_ELEMS + (ulong)c * 3;
    float s0 = in[0], s1 = in[1], s2 = in[2];
    const float w0 = a.w[c * 4], w1 = a.w[c * 4 + 1], w2 = a.w[c * 4 + 2], w3 = a.w[c * 4 + 3], b = a.b[c];
    for (int t = r0; t < r1; t++) {
        const float xin = a.proj[(ulong)t * a.ldp + 8192 + c];
        const float act = silu(b + w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin);
        a.xbc[(ulong)t * 10240 + c] = act;
        if (a.hist_x) {
            a.hist_x[(ulong)t * 10240 + c] = act;
            if (c < 128) a.hist_dt[t * 128 + c] = a.proj[(ulong)t * a.ldp + 18432 + c];
        }
        s0 = s1; s1 = s2; s2 = xin;
        if (a.record) {
            device float* o = ring + (ulong)((cur + 1 + (t - r0)) % BOLT_CONV_SLOTS) * BOLT_CONV_ELEMS + (ulong)c * 3;
            o[0] = s0; o[1] = s1; o[2] = s2;
        }
    }
    if (!a.record) {
        device float* o = ring + (ulong)((cur + 1) % BOLT_CONV_SLOTS) * BOLT_CONV_ELEMS + (ulong)c * 3;
        o[0] = s0; o[1] = s1; o[2] = s2;
    }
}

// Long single-segment blocks: no recurrence in a 4-tap filter, so rows are split over threadgroups (16 a group).
kernel void conv_long(constant ConvArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int c = tgp.x * 128 + tid;
    if (c >= 10240) return;
    device const Ctl* ctl = a.ctl;
    device float* ring = a.pool + (ulong)ctl->seg_seq[0] * BOLT_CONV_SLOTS * BOLT_CONV_ELEMS;
    const int cur = ctl->seg_conv[0];
    device const float* in = ring + (ulong)cur * BOLT_CONV_ELEMS + (ulong)c * 3;
    const int T = a.T;
    const int t0 = tgp.y * 16, t1 = min(T, t0 + 16);
    float v[3];
    for (int j = 0; j < 3; j++) {
        int t = t0 - 3 + j;
        v[j] = t >= 0 ? a.proj[(ulong)t * a.ldp + 8192 + c] : in[3 + t];
    }
    float s0 = v[0], s1 = v[1], s2 = v[2];
    const float w0 = a.w[c * 4], w1 = a.w[c * 4 + 1], w2 = a.w[c * 4 + 2], w3 = a.w[c * 4 + 3], b = a.b[c];
    for (int t = t0; t < t1; t++) {
        const float xin = a.proj[(ulong)t * a.ldp + 8192 + c];
        a.xbc[(ulong)t * 10240 + c] = silu(b + w0 * s0 + w1 * s1 + w2 * s2 + w3 * xin);
        s0 = s1; s1 = s2; s2 = xin;
    }
    if (t1 == T) {
        device float* o = ring + (ulong)((cur + 1) % BOLT_CONV_SLOTS) * BOLT_CONV_ELEMS + (ulong)c * 3;
        for (int j = 0; j < 3; j++) {
            int t = T - 3 + j;
            o[j] = t >= 0 ? a.proj[(ulong)t * a.ldp + 8192 + c] : in[3 + t];
        }
    }
}

// Mamba-2 selective scan, one simdgroup per (head, channel, segment): its 128 state entries sit 4 to a lane.
// The segment reads its SSM slot and writes the other one. dt is softplus(dt + bias), not clamped (the config's
// limit is 0).
kernel void scan(constant ScanArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]],
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device const Ctl* ctl = a.ctl;
    const int sgi = tgp.y;
    if (sgi >= ctl->nseg) return;
    const int h = tgp.x >> 3, d = (tgp.x & 7) * 8 + sg, g = h >> 4;
    const int r0 = ctl->seg_row0[sgi], r1 = ctl->seg_row0[sgi + 1];
    device float* st_pool = a.pool + (ulong)ctl->seg_seq[sgi] * 2 * BOLT_SSM_ELEMS;
    const int cur = ctl->seg_ssm[sgi];
    const ulong off = ((ulong)h * 64 + d) * 128 + 4 * lane;
    float4 st = *(device const float4*)(st_pool + (ulong)cur * BOLT_SSM_ELEMS + off);
    const float A = -exp(a.A_log[h]);
    const float bias = a.dt_bias[h], Dh = a.D[h];
    for (int t = r0; t < r1; t++) {
        const float sp = softplus(a.proj[(ulong)t * a.ldp + 18432 + h] + bias);
        const float dA = exp(sp * A);
        device const float* row = a.xbc + (ulong)t * 10240;
        const float xv = row[h * 64 + d];
        const float4 B = *(device const float4*)(row + 8192 + g * 128 + 4 * lane);
        const float4 C = *(device const float4*)(row + 9216 + g * 128 + 4 * lane);
        st = st * dA + (sp * xv) * B;
        const float part = simd_sum(dot(st, C));
        if (lane == 0) a.y[(ulong)t * 8192 + h * 64 + d] = part + Dh * xv;
    }
    *(device float4*)(st_pool + (ulong)(cur ^ 1) * BOLT_SSM_ELEMS + off) = st;
}

// Long single-segment blocks: a thread pair owns one channel, 64 state entries each; B, C, x and the decay of 16
// rows are staged in threadgroup memory. One threadgroup per head.
kernel void scan_long(constant ScanArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float Bs[16][128], Cs[16][128], xs[16][64];
    threadgroup float dAs[16], sps[16];
    const int half_ = tid & 1, d = tid >> 1;
    const int h = tgp.x, g = h >> 4;
    device const Ctl* ctl = a.ctl;
    device float* st_pool = a.pool + (ulong)ctl->seg_seq[0] * 2 * BOLT_SSM_ELEMS;
    const int cur = ctl->seg_ssm[0];
    float4 st[16];
    device const float4* src = (device const float4*)(st_pool + (ulong)cur * BOLT_SSM_ELEMS + ((ulong)h * 64 + d) * 128 + half_ * 64);
    for (int i = 0; i < 16; i++) st[i] = src[i];
    const float A = -exp(a.A_log[h]);
    const float bias = a.dt_bias[h], Dh = a.D[h];
    const int T = a.T;
    for (int c0 = 0; c0 < T; c0 += 16) {
        const int tc = min(16, T - c0);
        for (int idx = tid; idx < tc * 32; idx += 128) {
            int row = idx >> 5, q = idx & 31, n = 4 * q;
            device const float* r = a.xbc + (ulong)(c0 + row) * 10240;
            *(threadgroup float4*)&Bs[row][n] = *(device const float4*)(r + 8192 + g * 128 + n);
            *(threadgroup float4*)&Cs[row][n] = *(device const float4*)(r + 9216 + g * 128 + n);
        }
        for (int idx = tid; idx < tc * 16; idx += 128) {
            int row = idx >> 4, q = idx & 15;
            *(threadgroup float4*)&xs[row][4 * q] = *(device const float4*)(a.xbc + (ulong)(c0 + row) * 10240 + h * 64 + 4 * q);
        }
        if ((int)tid < tc) {
            const float sp = softplus(a.proj[(ulong)(c0 + tid) * a.ldp + 18432 + h] + bias);
            sps[tid] = sp;
            dAs[tid] = exp(sp * A);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int tl = 0; tl < tc; tl++) {
            const float dA = dAs[tl], xv = xs[tl][d], sx = sps[tl] * xv;
            threadgroup const float4* Bp = (threadgroup const float4*)&Bs[tl][half_ * 64];
            threadgroup const float4* Cp = (threadgroup const float4*)&Cs[tl][half_ * 64];
            float part = 0.f;
            for (int i = 0; i < 16; i++) {
                st[i] = st[i] * dA + sx * Bp[i];
                part += dot(st[i], Cp[i]);
            }
            part += simd_shuffle_xor(part, 1);
            if (half_ == 0) a.y[(ulong)(c0 + tl) * 8192 + h * 64 + d] = part + Dh * xv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4* dst = (device float4*)(st_pool + (ulong)(cur ^ 1) * BOLT_SSM_ELEMS + ((ulong)h * 64 + d) * 128 + half_ * 64);
    for (int i = 0; i < 16; i++) dst[i] = st[i];
}

// Rebuild the SSM state after the first n rows of a verify window from the pre-window state. grid.y: Mamba layer.
kernel void scan_replay(constant ReplayArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int L = tgp.y;
    const int h = tgp.x >> 3, d = (tgp.x & 7) * 8 + sg, g = h >> 4;
    const ulong off = ((ulong)h * 64 + d) * 128 + 4 * lane;
    device float* pool = a.ssm + ((ulong)L * a.nseq + a.seq) * 2 * BOLT_SSM_ELEMS;
    float4 st = *(device const float4*)(pool + (ulong)a.slot * BOLT_SSM_ELEMS + off);
    const float A = -exp(a.A_log[L * 128 + h]);
    const float bias = a.dt_bias[L * 128 + h];
    device const float* hx = a.hist_x + (ulong)L * BOLT_BATCH_MAX * 10240;
    device const float* hd = a.hist_dt + (ulong)L * BOLT_BATCH_MAX * 128;
    for (int t = a.row0; t < a.row0 + a.n; t++) {
        const float sp = softplus(hd[t * 128 + h] + bias);
        const float dA = exp(sp * A);
        device const float* row = hx + (ulong)t * 10240;
        const float4 B = *(device const float4*)(row + 8192 + g * 128 + 4 * lane);
        st = st * dA + (sp * row[h * 64 + d]) * B;
    }
    *(device float4*)(pool + (ulong)(a.slot ^ 1) * BOLT_SSM_ELEMS + off) = st;
}

// out = RMSNorm_group(y * silu(z)) * w over groups of 1024.
kernel void gate_rms(constant GateArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int t = tgp.x >> 3, g = tgp.x & 7;
    const float4 y = ((device const float4*)(a.y + (ulong)t * 8192 + g * 1024))[tid];
    const float4 z = ((device const float4*)(a.proj + (ulong)t * a.ldp + g * 1024))[tid];
    const float4 v = y * float4(silu(z.x), silu(z.y), silu(z.z), silu(z.w));
    threadgroup float sm[8];
    const float ss = simd_sum(dot(v, v));
    if (lane == 0) sm[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.f;
    for (int i = 0; i < 8; i++) tot += sm[i];
    const float inv = rsqrt(tot / 1024.f + a.eps);
    const float4 w = ((device const float4*)(a.w + g * 1024))[tid];
    ((device float4*)(a.out + (ulong)t * 8192 + g * 1024))[tid] = v * inv * w;
}

// Sigmoid router: the top 9 of sigmoid(logit) + bias (ties to the lower index); weights are the plain sigmoids
// normalised to one and scaled. One simdgroup per token; E <= 512.
kernel void route(constant RouteArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const int t = tgp.x;
    float sig[16], sel[16];
    for (int j = 0; j < 16; j++) {
        const int i = lane + 32 * j;
        if (i < a.E) {
            const float z = a.logits[(ulong)t * a.ldl + i];
            sig[j] = 1.f / (1.f + exp(-z));
            sel[j] = sig[j] + a.bias[i];
        } else {
            sig[j] = 0.f;
            sel[j] = -1e30f;
        }
    }
    float picked[BOLT_TOPN];
    int pid[BOLT_TOPN];
    for (int k = 0; k < BOLT_TOPN; k++) {
        float bv = -3e38f;
        int bi = 0x7fffffff;
        for (int j = 0; j < 16; j++) {
            if (sel[j] > bv) { bv = sel[j]; bi = lane + 32 * j; }
        }
        for (ushort m = 16; m > 0; m >>= 1) {
            const float ov = simd_shuffle_xor(bv, m);
            const int oi = simd_shuffle_xor(bi, m);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        pid[k] = bi;
        float ps = 0.f;
        for (int j = 0; j < 16; j++) {
            if (lane + 32 * j == bi) { ps = sig[j]; sel[j] = -1e30f; }
        }
        picked[k] = simd_sum(ps);
    }
    float sum = 1e-20f;
    for (int k = 0; k < BOLT_TOPN; k++) sum += picked[k];
    if (lane < BOLT_TOPN) {
        int idx = 0;
        float val = 0.f;
        for (int k = 0; k < BOLT_TOPN; k++) if (k == lane) { idx = pid[k]; val = picked[k]; }
        a.ridx[t * BOLT_TOPN + lane] = idx;
        a.rw[t * BOLT_TOPN + lane] = val / sum * a.scale;
    }
}

// out[t] = sum_k rw[t][k] * o[t * 9 + k], 1024 wide.
kernel void combine(constant CombineArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int t = tgp.x;
    float4 acc = 0.f;
    for (int k = 0; k < BOLT_TOPN; k++)
        acc += a.rw[t * BOLT_TOPN + k] * ((device const float4*)(a.o + (ulong)(t * BOLT_TOPN + k) * 1024))[tid];
    ((device float4*)(a.out + (ulong)t * 1024))[tid] = acc;
}

// ---------------------------------------------------------------------------------------------------------
// Attention: 32 query heads over 2 KV heads, no positional encoding, bf16 cache [seq][pos][2][128].
// ---------------------------------------------------------------------------------------------------------

kernel void kv_store(constant KvArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int t = tgp.x;
    device const Ctl* ctl = a.ctl;
    if (t >= ctl->nrows) return;
    const int sgi = seg_of(ctl, t);
    const ulong pos = (ulong)ctl->seg_seq[sgi] * a.ctx + (ulong)(ctl->seg_base[sgi] + t - ctl->seg_row0[sgi]);
    for (int i = tid; i < 256; i += 128) {
        a.kc[pos * 256 + i] = to_bf(a.k[t * 256 + i]);
        a.vc[pos * 256 + i] = to_bf(a.v[t * 256 + i]);
    }
}

// One split of the keys for one row and KV head: eight simdgroups, two query heads each; a lane scores one key of
// each 32-key tile, then the tile's values are summed in key order. A row's splits are fixed by its segment's
// chunk (seg_chunk), so its rounding does not depend on the other rows of the block.
kernel void attn_part(constant KvArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int sp = tgp.x, g = tgp.y, t = tgp.z;
    device const Ctl* ctl = a.ctl;
    const bool pad = t >= ctl->nrows;
    const int sgi = pad ? 0 : seg_of(ctl, t);
    const int len = pad ? 1 : ctl->seg_base[sgi] + (t - ctl->seg_row0[sgi]) + 1;
    const int chunk = ctl->seg_chunk[sgi];
    const int p0 = sp * chunk, p1 = min(len, p0 + chunk);
    if (p0 >= p1) return;
    const ulong seq = pad ? 0 : ctl->seg_seq[sgi];
    threadgroup float qs[16][128];
    const float qscale = rsqrt(128.f);
    for (int i = tid; i < 16 * 128; i += 256) qs[i >> 7][i & 127] = a.q[(ulong)t * 4096 + g * 2048 + i] * qscale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int h0 = sg * 2, h1 = h0 + 1;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;
    float4 o0 = 0.f, o1 = 0.f;
    device const ushort* kc = a.kc + (seq * a.ctx) * 256 + g * 128;
    device const ushort* vc = a.vc + (seq * a.ctx) * 256 + g * 128;
    for (int base = p0; base < p1; base += 32) {
        const int n = min(32, p1 - base);
        const int p = base + lane;
        float s0 = -1e30f, s1 = -1e30f;
        if ((int)lane < n) {
            device const ushort4* kr = (device const ushort4*)(kc + (ulong)p * 256);
            float d0 = 0.f, d1 = 0.f;
            for (int i = 0; i < 32; i++) {
                const ushort4 kb = kr[i];
                const float4 k4 = float4(bf(kb.x), bf(kb.y), bf(kb.z), bf(kb.w));
                d0 += dot(*(threadgroup const float4*)&qs[h0][4 * i], k4);
                d1 += dot(*(threadgroup const float4*)&qs[h1][4 * i], k4);
            }
            s0 = d0; s1 = d1;
        }
        const float n0 = max(m0, simd_max(s0)), n1 = max(m1, simd_max(s1));
        const float c0 = exp(m0 - n0), c1 = exp(m1 - n1);
        const float e0 = (int)lane < n ? exp(s0 - n0) : 0.f, e1 = (int)lane < n ? exp(s1 - n1) : 0.f;
        l0 = l0 * c0 + simd_sum(e0);
        l1 = l1 * c1 + simd_sum(e1);
        m0 = n0; m1 = n1;
        o0 *= c0; o1 *= c1;
        for (int j = 0; j < n; j++) {
            const float pj0 = simd_shuffle(e0, (ushort)j), pj1 = simd_shuffle(e1, (ushort)j);
            const ushort4 vb = ((device const ushort4*)(vc + (ulong)(base + j) * 256))[lane];
            const float4 v4 = float4(bf(vb.x), bf(vb.y), bf(vb.z), bf(vb.w));
            o0 += pj0 * v4;
            o1 += pj1 * v4;
        }
    }
    device float* out = a.part + ((ulong)(t * 2 + g) * BOLT_ATT_SPLITS + sp) * BOLT_ATT_PART;
    if (lane == 0) { out[h0] = m0; out[h1] = m1; out[16 + h0] = l0; out[16 + h1] = l1; }
    ((device float4*)(out + 32 + h0 * 128))[lane] = o0;
    ((device float4*)(out + 32 + h1 * 128))[lane] = o1;
}

// Merge a row's splits in order.
kernel void attn_comb(constant KvArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int t = tgp.x, h = tgp.y, g = h >> 4, hh = h & 15, d = tid;
    device const Ctl* ctl = a.ctl;
    const bool pad = t >= ctl->nrows;
    const int sgi = pad ? 0 : seg_of(ctl, t);
    const int len = pad ? 1 : ctl->seg_base[sgi] + (t - ctl->seg_row0[sgi]) + 1;
    const int chunk = ctl->seg_chunk[sgi];
    const int splits = (len + chunk - 1) / chunk;
    device const float* base = a.part + (ulong)((t * 2 + g) * BOLT_ATT_SPLITS) * BOLT_ATT_PART;
    float M = -1e30f;
    for (int s = 0; s < splits; s++) M = max(M, base[(ulong)s * BOLT_ATT_PART + hh]);
    float L = 0.f, O = 0.f;
    for (int s = 0; s < splits; s++) {
        device const float* p = base + (ulong)s * BOLT_ATT_PART;
        const float w = exp(p[hh] - M);
        L += w * p[16 + hh];
        O += w * p[32 + hh * 128 + d];
    }
    a.out[(ulong)t * 4096 + h * 128 + d] = O / L;
}

// ---------------------------------------------------------------------------------------------------------
// Row-wise top candidates of the logits. Stage one: 128 threadgroups per row, each its slice's best topk; stage
// two merges them. Ties go to the lower id.
// ---------------------------------------------------------------------------------------------------------

inline void tg_best(float bv, int bi, threadgroup float* wv, threadgroup int* wi, ushort sg, ushort lane, thread float& fv, thread int& fi) {
    for (ushort m = 16; m > 0; m >>= 1) {
        const float ov = simd_shuffle_xor(bv, m);
        const int oi = simd_shuffle_xor(bi, m);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { wv[sg] = bv; wi[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fv = wv[0];
    fi = wi[0];
    for (int w = 1; w < 8; w++) if (better(wv[w], wi[w], fv, fi)) { fv = wv[w]; fi = wi[w]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void cand1(constant CandArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int b = tgp.x, i = tgp.y;
    const int t = a.sel ? a.sel[i] : i;
    device const float* row = a.logits + (ulong)t * BOLT_VOCAB + (ulong)b * 1024;
    device const uint* mk = a.masks ? a.masks + (ulong)i * a.words : nullptr;
    float v[4];
    for (int j = 0; j < 4; j++) {
        const int tok = b * 1024 + tid + 256 * j;
        const bool ok = !mk || ((tok >> 5) < a.words && ((mk[tok >> 5] >> (tok & 31)) & 1u));
        v[j] = ok ? row[tid + 256 * j] : -3.4e38f;
    }
    threadgroup float wv[8];
    threadgroup int wi[8];
    for (int r = 0; r < a.topk; r++) {
        float bv = -3.4e38f;
        int bj = 0;
        for (int j = 0; j < 4; j++) if (v[j] > bv) { bv = v[j]; bj = j; }
        float fv; int fi;
        tg_best(bv, (int)tid + 256 * bj, wv, wi, sg, lane, fv, fi);
        if (tid == 0) {
            const ulong o = ((ulong)i * BOLT_CAND_BLOCKS + b) * BOLT_CAND + r;
            a.sv[o] = fv;
            a.si[o] = b * 1024 + fi;
        }
        if ((fi & 255) == (int)tid) v[fi >> 8] = -3.4e38f;
    }
}

kernel void cand2(constant CandArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int t = tgp.x;
    const int total = BOLT_CAND_BLOCKS * a.topk;
    float v[32];
    int id[32];
    for (int j = 0; j < 32; j++) {
        const int k = tid + 256 * j;
        if (k < total) {
            const int blk = k / a.topk, r = k % a.topk;
            const ulong o = ((ulong)t * BOLT_CAND_BLOCKS + blk) * BOLT_CAND + r;
            v[j] = a.sv[o];
            id[j] = a.si[o];
        } else {
            v[j] = -3.4e38f;
            id[j] = 0x7fffffff;
        }
    }
    threadgroup float wv[8];
    threadgroup int wi[8];
    for (int r = 0; r < a.topk; r++) {
        float bv = -3.4e38f;
        int bi = 0x7fffffff, bj = 0;
        for (int j = 0; j < 32; j++) if (better(v[j], id[j], bv, bi)) { bv = v[j]; bi = id[j]; bj = j; }
        float fv; int fi;
        tg_best(bv, bi, wv, wi, sg, lane, fv, fi);
        if (tid == 0) {
            a.vals[(ulong)t * BOLT_CAND + r] = fv;
            a.ids[(ulong)t * BOLT_CAND + r] = fi;
        }
        if (bi == fi && bv == fv) {
            for (int j = 0; j < 32; j++) if (j == bj) { v[j] = -3.4e38f; id[j] = 0x7fffffff; }
        }
    }
}

kernel void copy_f32(constant CopyArgs& a [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;
    if (i >= a.n) return;
    a.dst[i] = a.add ? a.dst[i] + a.src[i] : a.src[i];
}

// ---------------------------------------------------------------------------------------------------------
// LiLiCorr drafter: five DFlash layers (grouped conv, sliding-window sink attention, NVFP4 MLP) over six of the
// target's residuals, then the candidate lattice and its greedy walk.
// ---------------------------------------------------------------------------------------------------------

constant constexpr int DH = 4096, DHD = 128, DHEADS = 32, DKVH = 2;
constant constexpr int DBLOCK = 8, DSLOTS = 7, DTOPK = 8, DMASK = 990, DNODES = DSLOTS * DTOPK;
constant constexpr int DLH = 1024, DLHEADS = 8, DLDIM = 128;
constant constexpr int DRING = 2048, DWINDOW = 1024;
constant constexpr float DEPS = 1e-6f;

inline float block_sum(float v, threadgroup float* sm, ushort sg, ushort lane, uint nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sm[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.f;
    for (uint i = 0; i < nsg; i++) t += sm[i];
    return t;
}

// y = rmsnorm(x) * w (w optional), rows of n; one threadgroup of 256 per row.
kernel void d_rms(constant RmsArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sm[8];
    device const float* xr = a.x + (ulong)tgp.x * a.n;
    float ss = 0.f;
    for (int i = tid; i < a.n; i += 256) ss += xr[i] * xr[i];
    ss = block_sum(ss, sm, sg, lane, 8);
    const float inv = rsqrt(ss / float(a.n) + a.eps);
    device float* yr = a.y + (ulong)tgp.x * a.n;
    for (int i = tid; i < a.n; i += 256) yr[i] = xr[i] * inv * (a.w ? a.w[i] : 1.f);
}

// Grouped dynamic depthwise conv over each 8-row block: tap 0 the row itself, tap 1 the previous row of its block.
kernel void d_gconv(constant GconvArgs& a [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
    const int i = gid.x;
    if (i >= a.T * DH) return;
    const int t = i / DH, c = i % DH, g = c / 16;
    device const float* b = a.base + a.side * 2 * DH;
    float acc = 0.f;
    for (int tap = 0; tap < 2; tap++) {
        if ((t % DBLOCK) - tap < 0) continue;
        const float coef = b[tap * DH + c] + a.delta[(((ulong)t * 2 + a.side) * 2 + tap) * 256 + g];
        acc += coef * a.x[(ulong)(t - tap) * DH + c];
    }
    a.y[i] = acc;
}

// Per-head RMS norm then NeoX rope; 64 threads per (row, head).
kernel void d_qk_norm_rope(constant RopeArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int t = tgp.x, h = tgp.y;
    if (h >= a.heads) return;
    device float* r = a.x + ((ulong)t * a.heads + h) * DHD;
    threadgroup float sm[2];
    float v0 = r[tid], v1 = r[tid + 64];
    const float ss = block_sum(v0 * v0 + v1 * v1, sm, sg, lane, 2);
    const float inv = rsqrt(ss / float(DHD) + DEPS);
    v0 = v0 * inv * a.w[tid];
    v1 = v1 * inv * a.w[tid + 64];
    const int pos = a.mode == 2 ? a.rowpos[t] : a.mode == 1 ? a.posv[t / DBLOCK] + t % DBLOCK : a.pos0 + t;
    const float ang = float(pos) * pow(10000.f, -2.f * float(tid) / float(DHD));
    const float c = precise::cos(ang), s = precise::sin(ang);
    r[tid] = v0 * c - v1 * s;
    r[tid + 64] = v0 * s + v1 * c;
}

// Block attention with a sink: context keys from the sequence's ring (a sliding window of 1024), the block's own
// keys all visible. One threadgroup of 128 per (head, row).
kernel void d_attn(constant DattnArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int h = tgp.x, row = tgp.y;
    const int b = row / DBLOCK, qi = row % DBLOCK;
    const int seq = a.seqs[b];
    device const float* ck = a.ringk + ((ulong)seq * a.nlayers + a.layer) * DRING * DKVH * DHD;
    device const float* cv = a.ringv + ((ulong)seq * a.nlayers + a.layer) * DRING * DKVH * DHD;
    device const float* bk = a.bk + (ulong)b * DBLOCK * DKVH * DHD;
    device const float* bv = a.bv + (ulong)b * DBLOCK * DKVH * DHD;
    const int clen = a.clen[b], pos = a.pos[b];
    const int kh = h / (DHEADS / DKVH);
    threadgroup float qs[DHD];
    threadgroup float sc[DWINDOW + DBLOCK + 8];
    threadgroup float red[4];
    qs[tid] = a.q[((ulong)row * DHEADS + h) * DHD + tid] * rsqrt(float(DHD));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int lo = pos + qi - (DWINDOW - 1);
    if (lo < 0) lo = 0;
    int nctx = clen > lo ? clen - lo : 0;
    if (nctx > DWINDOW + 8) nctx = DWINDOW + 8;
    lo = clen - nctx;
    const int total = nctx + DBLOCK;
    float m = a.sink[h];
    for (int j = tid; j < total; j += 128) {
        device const float* kk = j < nctx ? ck + ((ulong)((lo + j) & (DRING - 1)) * DKVH + kh) * DHD : bk + ((ulong)(j - nctx) * DKVH + kh) * DHD;
        float d = 0.f;
        for (int i = 0; i < DHD; i += 4) d += dot(*(threadgroup const float4*)&qs[i], *(device const float4*)(kk + i));
        sc[j] = d;
        m = max(m, d);
    }
    m = simd_max(m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(red[0], red[1]), max(red[2], red[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float den = 0.f;
    for (int j = tid; j < total; j += 128) {
        const float e = exp(sc[j] - m);
        sc[j] = e;
        den += e;
    }
    den = block_sum(den, red, sg, lane, 4) + exp(a.sink[h] - m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.f;
    for (int j = 0; j < total; j++) {
        device const float* vv = j < nctx ? cv + ((ulong)((lo + j) & (DRING - 1)) * DKVH + kh) * DHD : bv + ((ulong)(j - nctx) * DKVH + kh) * DHD;
        acc += sc[j] * vv[tid];
    }
    a.out[((ulong)row * DHEADS + h) * DHD + tid] = acc / den;
}

// Rows to a sequence's ring: row t at position pos0 + t of `seq`, or (rowseq given) at rowpos[t] of rowseq[t].
kernel void d_write_kv(constant RingArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int t = tgp.x;
    const int q = a.rowseq ? a.rowseq[t] : a.seq;
    if (q < 0) return;
    const int pos = a.rowseq ? a.rowpos[t] : a.pos0 + t;
    const ulong ring = ((ulong)q * a.nlayers + a.layer) * DRING * DKVH * DHD;
    for (int i = tid; i < DKVH * DHD; i += 256) {
        const ulong o = ring + (ulong)(pos & (DRING - 1)) * DKVH * DHD + i;
        a.ringk[o] = a.k[(ulong)t * DKVH * DHD + i];
        a.ringv[o] = a.v[(ulong)t * DKVH * DHD + i];
    }
}

// Misc row kernels of the drafter, one threadgroup of 256 per row unless noted.
// d_block_embed: rows [anchor, mask x 7] of every stream (ids = anchors).
kernel void d_block_embed(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int r = tgp.x, row = r % DBLOCK;
    int id = row == 0 ? a.ids[r / DBLOCK] : DMASK;
    if ((uint)id >= (uint)BOLT_VOCAB) id = 0;
    device const ushort* src = a.table + (ulong)id * DH;
    for (int i = tid; i < DH; i += 256) a.y[(ulong)r * DH + i] = bf(src[i]);
}

// d_gather_cand: embeddings of each stream's 56 candidates (slot-major) from the candidate ids (ids = cids).
kernel void d_gather_cand(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int node = tgp.x, b = node / DNODES, li = node % DNODES;
    int id = a.ids[((ulong)(b * DBLOCK + 1 + li / DTOPK)) * BOLT_CAND + li % DTOPK];
    if ((uint)id >= (uint)BOLT_VOCAB) id = 0;
    device const ushort* src = a.table + (ulong)id * DH;
    for (int i = tid; i < DH; i += 256) a.y[(ulong)node * DH + i] = bf(src[i]);
}

// d_lse: log-sum-exp of a logits row (a = logits, y = lse).
kernel void d_lse(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    device const float* x = a.a + (ulong)tgp.x * BOLT_VOCAB;
    threadgroup float sm[8], mm[8];
    float m = -1e30f;
    for (int i = tid; i < BOLT_VOCAB; i += 256) m = max(m, x[i]);
    m = simd_max(m);
    if (lane == 0) mm[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = mm[0];
    for (int i = 1; i < 8; i++) m = max(m, mm[i]);
    float s = 0.f;
    for (int i = tid; i < BOLT_VOCAB; i += 256) s += exp(x[i] - m);
    s = block_sum(s, sm, sg, lane, 8);
    if (tid == 0) a.y[tgp.x] = m + log(s);
}

// d_features: features of candidate (slot, rank): [logp, p, logp - max, rank/7, is_top1], LayerNorm over the five.
// a = candidate values, b = lse, c = norm weight, d = norm bias. One threadgroup of 64 per stream.
kernel void d_features(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int b = tgp.x;
    const int node = tid;
    if (node >= DNODES) return;
    device const float* vals = a.a + (ulong)(b * DBLOCK + 1) * BOLT_CAND;
    device const float* lse = a.b + b * DBLOCK + 1;
    const int s = node / DTOPK, k = node % DTOPK;
    const float lp = vals[s * BOLT_CAND + k] - lse[s];
    const float mx = vals[s * BOLT_CAND] - lse[s];
    float f[5] = {lp, exp(lp), lp - mx, float(k) / 7.f, k == 0 ? 1.f : 0.f};
    float mean = 0.f;
    for (int i = 0; i < 5; i++) mean += f[i];
    mean /= 5.f;
    float var = 0.f;
    for (int i = 0; i < 5; i++) var += (f[i] - mean) * (f[i] - mean);
    const float inv = rsqrt(var / 5.f + 1e-5f);
    device float* feat = a.y + (ulong)b * DNODES * 5;
    for (int i = 0; i < 5; i++) feat[node * 5 + i] = (f[i] - mean) * inv * a.c[i] + a.d[i];
}

// d_feat_up: the feature MLP's up projection (5 -> 1024) with SiLU. table = weight (bf16 [1024][5]), a = bias, b = features.
kernel void d_feat_up(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int node = tgp.x;
    for (int j = tid; j < DLH; j += 256) {
        float v = a.a[j];
        for (int i = 0; i < 5; i++) v += bf(a.table[j * 5 + i]) * a.b[node * 5 + i];
        a.y[(ulong)node * DLH + j] = silu(v);
    }
}

// d_nodes: token state + pass state of its slot + features + slot and rank embeddings (a, b, c, d, e).
kernel void d_nodes(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int i = tgp.x, b = i / DNODES, li = i % DNODES;
    const int s = li / DTOPK, k = li % DTOPK;
    for (int c = tid; c < DLH; c += 256)
        a.y[(ulong)i * DLH + c] = a.a[(ulong)i * DLH + c] + a.b[(ulong)(b * DBLOCK + 1 + s) * DLH + c] + a.c[(ulong)i * DLH + c] +
                                  a.d[(ulong)s * DLH + c] + a.e[(ulong)k * DLH + c];
}

// d_lat_attn: lattice self-attention over each stream's 56 nodes with a per-head bias; qkv rows [q | k | v] (a),
// bias [heads][56][56] (b). One threadgroup of 128 per (head, node).
kernel void d_lat_attn(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int h = tgp.x, row = tgp.y;
    const int b = row / DNODES, qi = row % DNODES;
    threadgroup float qs[DLDIM];
    threadgroup float sc[DNODES];
    const ulong ld = 3 * DLH;
    qs[tid] = a.a[(ulong)row * ld + h * DLDIM + tid] * rsqrt(float(DLDIM));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)tid < DNODES) {
        device const float* kk = a.a + (ulong)(b * DNODES + tid) * ld + DLH + h * DLDIM;
        float d = 0.f;
        for (int i = 0; i < DLDIM; i++) d += qs[i] * kk[i];
        sc[tid] = d + a.b[((ulong)h * DNODES + qi) * DNODES + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -1e30f;
    for (int j = 0; j < DNODES; j++) m = max(m, sc[j]);
    float den = 0.f, acc = 0.f;
    for (int j = 0; j < DNODES; j++) {
        const float e = exp(sc[j] - m);
        den += e;
        acc += e * a.a[(ulong)(b * DNODES + j) * ld + 2 * DLH + h * DLDIM + tid];
    }
    a.y[(ulong)row * DLH + h * DLDIM + tid] = acc / den;
}

// d_cat3: [h, anchor, h * anchor] rows (a = h, b = anchor state per stream).
kernel void d_cat3(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int row = tgp.x;
    device const float* an = a.b + (ulong)(row / DNODES) * DLH;
    for (int c = tid; c < DLH; c += 256) {
        const float hv = a.a[(ulong)row * DLH + c], av = an[c];
        device float* o = a.y + (ulong)row * 3 * DLH;
        o[c] = hv; o[DLH + c] = av; o[2 * DLH + c] = hv * av;
    }
}

// d_l2norm: y /= max(|y|, eps) on rows of 1024.
kernel void d_l2norm(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sm[8];
    device float* r = a.y + (ulong)tgp.x * DLH;
    float ss = 0.f;
    for (int i = tid; i < DLH; i += 256) ss += r[i] * r[i];
    ss = block_sum(ss, sm, sg, lane, 8);
    const float inv = 1.f / max(sqrt(ss), a.eps);
    for (int i = tid; i < DLH; i += 256) r[i] *= inv;
}

// d_walk: greedy walk through each stream's lattice: slot 0 from the anchor, then each slot from the previous choice.
// a = anchor out [stream][1024], b = out edges, c = in edges [node][1024], ids = candidate ids, out = tokens [stream][7].
kernel void d_walk(constant DraftMiscArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const int b = tgp.x;
    device const float* anchor_out = a.a + (ulong)b * DLH;
    device const float* outv = a.b + (ulong)b * DNODES * DLH;
    device const float* inv = a.c + (ulong)b * DNODES * DLH;
    device const int* ids = a.ids + (ulong)(b * DBLOCK + 1) * BOLT_CAND;
    device int* result = a.out + b * DSLOTS;
    threadgroup float sc[DTOPK];
    threadgroup int prev;
    if (tid == 0) prev = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = -1; s < DSLOTS - 1; s++) {
        device const float* x = s < 0 ? anchor_out : outv + ((ulong)s * DTOPK + prev) * DLH;
        device const float* y = inv + ((ulong)(s + 1) * DTOPK + sg) * DLH;
        float acc = 0.f;
        for (int i = lane; i < DLH; i += 32) acc += x[i] * y[i];
        acc = simd_sum(acc);
        if (lane == 0) sc[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            int best = 0;
            for (int k = 1; k < DTOPK; k++) if (sc[k] > sc[best]) best = k;
            prev = best;
            result[s + 1] = ids[(s + 1) * BOLT_CAND + best];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// d_silu_mul: y = silu(a) * b, n elements (one thread each).
kernel void d_silu_mul(constant DraftMiscArgs& a [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
    const int i = gid.x;
    if (i >= a.n) return;
    a.y[i] = silu(a.a[i]) * a.b[i];
}

// ---------------------------------------------------------------------------------------------------------
// Prompt GEMMs: a threadgroup decodes a 64-row x 32-wide tile of the weights once into threadgroup memory and
// multiplies it against 32 activation rows on fp32 simdgroup matrices. Four simdgroups, 16 weight rows each.
// ---------------------------------------------------------------------------------------------------------

constant constexpr int GM = 64, GN = 32, GK = 64, GKP = GK + 8;

// 64 weight rows x 64 of K, decoded once into threadgroup memory as half: thread tid decodes one chunk
template <typename F>
inline void gemm_tile(constant GemvArgs& a, int row0, int c0, uint tid, threadgroup half* Ws, threadgroup const uint2* grid) {
    const int r = tid >> 1, part = tid & 1;
    const int row = min(row0 + r, a.n - 1);
    typename F::C ch = F::load(a, row, c0 + part);
    for (int sub = 0; sub < 4; sub++) {
        float4 lo, hi;
        F::deq(ch, sub, lo, hi, grid);
        threadgroup half* w = Ws + r * GKP + part * 32 + sub * 8;
        *(threadgroup half4*)w = half4(lo);
        *(threadgroup half4*)(w + 4) = half4(hi);
    }
}

inline void gemm_x(device const float* x, int ldx, int row, int c0, uint tid, threadgroup float* Xs) {
    const int t = tid >> 2, q = (tid & 3) * 16;
    device const float4* xp = (device const float4*)(x + (ulong)row * ldx + (ulong)c0 * 32 + q);
    for (int i = 0; i < 4; i++) *(threadgroup float4*)(Xs + t * (GK + 4) + q + 4 * i) = xp[i];
}

inline void gemm_mma(threadgroup const half* Ws, threadgroup const float* Xs, ushort sg, thread simdgroup_float8x8 (&acc)[2][4]) {
    for (int kk = 0; kk < GK; kk += 8) {
        simdgroup_half8x8 A[2];
        simdgroup_float8x8 B[4];
        for (int i = 0; i < 2; i++) simdgroup_load(A[i], Ws + (sg * 16 + i * 8) * GKP + kk, GKP);
        for (int j = 0; j < 4; j++) simdgroup_load(B[j], Xs + (j * 8) * (GK + 4) + kk, GK + 4, ulong2(0, 0), true);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
    }
}

template <typename F>
kernel void gemm(constant GemvArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint2 grid_s[256];
    threadgroup half Ws[GM * GKP];
    threadgroup float Xs[GN * (GK + 4)];
    threadgroup float Cs[4][16 * GN];
    if (F::grid) {
        for (uint i = tid; i < 256; i += 128) grid_s[i] = IQ2XXS_GRID[i];
    }
    const int row0 = tgp.x * GM, t0 = tgp.y * GN;
    simdgroup_float8x8 acc[2][4];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    const int nchunk = a.k >> 5;   // K is a multiple of 64 for every matrix of the model
    for (int c = 0; c < nchunk; c += 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_tile<F>(a, row0, c, tid, Ws, grid_s);
        gemm_x(a.x, a.ldx, min(t0 + (int)(tid >> 2), a.T - 1), c, tid, Xs);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_mma(Ws, Xs, sg, acc);
    }
    threadgroup float* C = Cs[sg];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) simdgroup_store(acc[i][j], C + (i * 8) * GN + j * 8, GN);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 16 * GN; e += 32) {
        const int r = e / GN, t = e % GN;
        const int row = row0 + sg * 16 + r, tok = t0 + t;
        if (row >= a.n || tok >= a.T) continue;
        float v = C[r * GN + t];
        if (a.bias) v += a.bias[row];
        v = epilogue(v, a.epi);
        device float* yp = a.y + (ulong)tok * a.ldy + row;
        *yp = a.accum ? *yp + v : v;
    }
}

typedef decltype(gemm<FQ8>) gemm_t;
template [[host_name("gemm_q8")]] kernel gemm_t gemm<FQ8>;
template [[host_name("gemm_nv4")]] kernel gemm_t gemm<FNV4>;
template [[host_name("gemm_iq2")]] kernel gemm_t gemm<FIQ2>;
template [[host_name("gemm_bf")]] kernel gemm_t gemm<FBF>;

// Routed experts for prompt chunks: one work item (one expert, up to 32 pairs) per threadgroup row of the grid,
// 64 of the expert's rows a threadgroup.
kernel void moe_gemm(constant MoeArgs& a [[buffer(0)]], uint3 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if ((int)tgp.y >= a.nwork[0]) return;
    threadgroup uint2 grid_s[256];
    threadgroup half Ws[GM * GKP];
    threadgroup float Xs[GN * (GK + 4)];
    threadgroup float Cs[4][16 * GN];
    threadgroup int pid[GN];
    for (uint i = tid; i < 256; i += 128) grid_s[i] = IQ2XXS_GRID[i];
    const int e = a.wexp[tgp.y], first = a.wfirst[tgp.y], cnt = a.wcount[tgp.y];
    if ((int)tid < GN) pid[tid] = a.order[first + min((int)tid, cnt - 1)];
    const int row0 = tgp.x * GM;
    const int nb = a.k >> 8;
    device const uchar* qbase = a.qs + (ulong)e * a.qs_stride;
    device const half* dbase = a.d + (ulong)e * a.d_stride;
    simdgroup_float8x8 acc[2][4];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int p = pid[tid >> 2];
    const int xr = a.x_per_pair ? p : p / BOLT_TOPN;
    const int nchunk = a.k >> 5;
    for (int c = 0; c < nchunk; c += 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const int r = tid >> 1, part = tid & 1;
            const int row = min(row0 + r, a.n - 1);
            const uint2 q = ((device const uint2*)(qbase + (ulong)row * nb * 64))[c + part];
            const float db = float(dbase[(ulong)row * nb + ((c + part) >> 3)]) * (0.5f + float(q.y >> 28)) * 0.25f;
            for (int sub = 0; sub < 4; sub++) {
                float4 lo, hi;
                iq2_deq(q, db, sub, lo, hi, grid_s);
                threadgroup half* w = Ws + r * GKP + part * 32 + sub * 8;
                *(threadgroup half4*)w = half4(lo);
                *(threadgroup half4*)(w + 4) = half4(hi);
            }
        }
        gemm_x(a.x, a.ldx, xr, c, tid, Xs);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_mma(Ws, Xs, sg, acc);
    }
    threadgroup float* C = Cs[sg];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) simdgroup_store(acc[i][j], C + (i * 8) * GN + j * 8, GN);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int x = lane; x < 16 * GN; x += 32) {
        const int r = x / GN, t = x % GN;
        const int row = row0 + sg * 16 + r;
        if (row >= a.n || t >= cnt) continue;
        a.y[(ulong)pid[t] * a.ldy + row] = epilogue(C[r * GN + t], a.epi);
    }
}
