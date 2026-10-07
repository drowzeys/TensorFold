//! GLM-5.3's fused forward for windows of 1 to 128 rows (fused.compute / layer / attention_part / attention_core /
//! ffn_part), one stream, world 1 (Phase 1). Every kernel computes each row alone, so a row's bits never depend on
//! the window: the per-layer outputs match the Python engine's bit for bit when the kernels and their launches do.
//!
//! Frozen as tools/glm53/oracle.py freezes the Python side: bf16 latent cache, DCP 1, one stream (TF_GLM53_SIDE=0:
//! the side stream gives the same bits), x3linear.plan tiles, TensorFold's expert layout, the shared expert folded
//! into the routed combine. Selection always goes through the radix select (topk.top_columns over _index_scores'
//! 4-byte order words): the same keys as torch.topk + sort, which the champion uses for decode-sized windows (the
//! oracle checks the two agree).
//! TODO(P2): TP4 (all-gather + rank-order sum or the exact reduce-scatter between attention / FFN and the residual),
//! prompt chunks over 128 rows (x3prefill GEMM, bmm absorb/expand), DCP, int8/int4 latent caches, CUDA graphs.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const exl3 = @import("exl3.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const W = @import("cuda_weights.zig");

/// fused.FAST_ROWS: windows up to this many rows use the decode attention tiling (non-decode Buffers).
pub const fast_rows: usize = 16;
/// fused.MAX_ROWS: the widest window of the row-invariant EXL3 linear (this port's limit for now).
pub const max_rows: usize = 128;

/// fused.Buffers: scratch for windows of up to `rows` rows (sliced [:R]); `score_cols` the widest index range.
pub const Buffers = struct {
    d: *const cuda.Driver,
    rows: usize,
    score_cols: usize,
    small: usize,
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: std.mem.Allocator,
    x: u64 = 0, // bf16 [rows, D]
    normed: u64 = 0,
    qa: u64 = 0, // bf16 [rows, q_lora]
    qn: u64 = 0,
    kva: u64 = 0, // bf16 [rows, kv_a.n]
    kva_n: usize = 0,
    q: u64 = 0, // bf16 [rows, H * (nope + rope)]
    qlat: u64 = 0, // bf16 [rows, H, kv_lora]
    qrot: u64 = 0, // bf16 [rows, H, rope]
    po: u64 = 0, // fp32 chunk partials
    pm: u64 = 0,
    pl: u64 = 0,
    ol: u64 = 0, // bf16 [rows, H, kv_lora]
    o: u64 = 0, // bf16 [rows, H * v]
    dummy: u64 = 0, // int32 [1] zero
    ik: u64 = 0, // fp32 [rows, index_dim]
    iw: u64 = 0, // fp32 [rows, index_heads]
    iq: u64 = 0, // bf16 [rows, index_heads * index_dim]
    sc: u64 = 0, // int64 [min(rows, 128) * score_cols] (int32 order words here)
    tok: u64 = 0, // int32 [rows, index_topk]
    g: u64 = 0, // bf16 [rows * width]
    u: u64 = 0,
    act: u64 = 0,
    mlog: u64 = 0, // fp32 [rows, E]
    pick: u64 = 0, // int32 [rows, top_k]
    wts: u64 = 0, // fp32 [rows, top_k]
    rpart: u64 = 0, // fp32 router partials [ROUTER_KS, rows, max(E, index_dim)]
    sy: u64 = 0, // fp32 [rows, D]
    part: u64 = 0, // fp32 [rows, D]
    lin: kern.LinearScratch = undefined,
    xs: ?kern.ExpertScratch = null,

    fn alloc(b: *Buffers, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(b.d, @max(bytes, 16));
        errdefer m.free();
        try m.fill8(0, null); // torch.zeros / empty: zero here, so stray reads are deterministic
        try b.mem.append(b.gpa, m);
        return m.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, rows: usize, score_cols: usize) !Buffers {
        const c = w.config;
        if (rows > max_rows) return error.WindowTooWide;
        var b: Buffers = .{ .d = d, .gpa = gpa, .rows = rows, .score_cols = score_cols, .small = fast_rows };
        errdefer b.deinit();
        const D = c.hidden;
        const H = w.heads;
        const lw = c.kv_lora;
        const rd = c.rope;
        const qd = c.qkDim();
        b.x = try b.alloc(rows * D * 2);
        b.normed = try b.alloc(rows * D * 2);
        b.qa = try b.alloc(rows * c.q_lora * 2);
        b.qn = try b.alloc(rows * c.q_lora * 2);
        b.kva_n = w.layers[0].kv_a.n;
        b.kva = try b.alloc(rows * b.kva_n * 2);
        b.q = try b.alloc(rows * H * qd * 2);
        b.qlat = try b.alloc(rows * H * lw * 2);
        b.qrot = try b.alloc(rows * H * rd * 2);
        const slots = @max(c.index_topk / tri.attn_decode.chk * @min(rows, b.small), c.index_topk / tri.attn_prompt.chk * rows);
        b.po = try b.alloc(slots * H * lw * 4);
        b.pm = try b.alloc(slots * H * 4);
        b.pl = try b.alloc(slots * H * 4);
        b.ol = try b.alloc(rows * H * lw * 2);
        b.o = try b.alloc(rows * H * c.v_dim * 2);
        b.dummy = try b.alloc(4);
        b.ik = try b.alloc(rows * c.index_dim * 4);
        b.iw = try b.alloc(rows * c.index_heads * 4);
        b.iq = try b.alloc(rows * c.index_heads * c.index_dim * 2);
        b.sc = try b.alloc(@min(rows, 128) * score_cols * 8);
        b.tok = try b.alloc(rows * c.index_topk * 4);
        const width = @max(c.dense_width, c.expert_width * @max(c.shared_experts, 1)) / w.world;
        b.g = try b.alloc(rows * width * 2);
        b.u = try b.alloc(rows * width * 2);
        b.act = try b.alloc(rows * width * 2);
        b.mlog = try b.alloc(rows * c.experts * 4);
        b.pick = try b.alloc(rows * c.top_k * 4);
        b.wts = try b.alloc(rows * c.top_k * 4);
        b.rpart = try b.alloc(tri.router_ks * rows * @max(c.experts, c.index_dim) * 4);
        b.sy = try b.alloc(rows * D * 4);
        b.part = try b.alloc(rows * D * 4);
        // EXL3 linear scratch: per job fp16 Xh [rows, K] and fp32 Z [SK, rows, N]
        const ext = w.linearExtents();
        b.lin.xh_bytes = rows * ext.k * 2;
        b.lin.z_bytes = rows * ext.zn * 4;
        for (0..exl3.max_jobs) |j| {
            b.lin.xh[j] = try b.alloc(b.lin.xh_bytes);
            b.lin.z[j] = try b.alloc(b.lin.z_bytes);
        }
        // experts.Scratch(ex, rows, slots = top_k)
        for (w.layers) |*L| {
            const ex = if (L.experts) |*e| e else continue;
            const P = rows * c.top_k;
            const gu = try exl3.defaultTiles(ex.dims, ex.width, true);
            const dn = try exl3.defaultTiles(ex.width, ex.dims, false);
            const maxu = @min(P, ex.count);
            const xs: kern.ExpertScratch = .{
                .xg = try b.alloc(P * ex.dims * 2),
                .xu = try b.alloc(P * ex.dims * 2),
                .xd = try b.alloc(P * ex.width * 2),
                .z = try b.alloc(@max(2 * gu.sk * ex.width, dn.sk * ex.dims) * P * 4),
                .y = try b.alloc(P * ex.dims * 4),
                .ids = try b.alloc(maxu * 4),
                .count = try b.alloc(4),
                .members = try b.alloc(maxu * rows * 4),
                .done_gu = try b.alloc(maxu * ((rows + 15) / 16) * @max(ex.width / 128, 1) * 4),
                .done_d = try b.alloc(rows * @max(ex.dims / 128, 1) * 4),
                .rows = rows,
                .slots = c.top_k,
                .gu = gu,
                .dn = dn,
            };
            // members_buf starts at -1 (torch.full(..., -1))
            try b.d.check(b.d.api.cuMemsetD32_v2(xs.members, 0xFFFFFFFF, maxu * rows), "cuMemsetD32");
            b.xs = xs;
            break; // one scratch serves every MoE layer (the same shapes)
        }
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        for (b.mem.items) |*m| m.free();
        b.mem.deinit(b.gpa);
        b.* = undefined;
    }
};

/// fused.State: a 576-wide latent row a token and layer, bf16 index keys on full-indexer layers, the device position.
pub const State = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    capacity: usize,
    kc: []u64, // per layer: bf16 [capacity + 1, latent]
    ic: []u64, // per layer: bf16 [capacity + 1, index_dim], 0 on layers without an indexer
    pos: u64, // int32 [1]
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize) !State {
        const c = w.config;
        var s: State = .{ .d = d, .gpa = gpa, .capacity = capacity, .kc = try gpa.alloc(u64, w.layers.len), .ic = try gpa.alloc(u64, w.layers.len), .pos = 0 };
        errdefer s.deinit();
        const rows = capacity + 1;
        for (w.layers, 0..) |*L, i| {
            s.kc[i] = try s.zeros(rows * c.latentWidth() * 2);
            s.ic[i] = if (L.indexer != null) try s.zeros(rows * c.index_dim * 2) else 0;
        }
        s.pos = try s.zeros(4);
        return s;
    }

    fn zeros(s: *State, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(s.d, bytes);
        errdefer m.free();
        try m.fill8(0, null);
        try s.mem.append(s.gpa, m);
        return m.ptr;
    }

    pub fn deinit(s: *State) void {
        for (s.mem.items) |*m| m.free();
        s.mem.deinit(s.gpa);
        s.gpa.free(s.kc);
        s.gpa.free(s.ic);
        s.* = undefined;
    }
};

pub const Forward = struct {
    w: *const W.Weights,
    k: *const kern.Kernels,
    s: cuda.Stream,
    inv: u64, // fp32 [rope / 2]: fused.inv_freq (the oracle's bytes in Phase 1)

    fn ops(f: *const Forward) kern.Ops {
        return .{ .k = f.k, .s = f.s };
    }

    fn t(f: *const Forward) tri.Tri {
        return .{ .set = &f.k.triton, .s = f.s };
    }

    fn mla(f: *const Forward) tri.Mla {
        const c = f.w.config;
        return .{ .heads = f.w.heads, .nope = c.nope, .rope = c.rope, .lw = c.kv_lora, .vd = c.v_dim, .topk = c.index_topk };
    }

    /// torch.index_select(embed, 0, ids): the window's token rows into b.x (row copies: exact).
    pub fn embed(f: *const Forward, b: *const Buffers, ids: []const u32) !void {
        const D = f.w.config.hidden;
        for (ids, 0..) |id, r| try f.ops().copy(b.x + r * D * 2, f.w.embed + @as(u64, id) * D * 2, D * 2);
    }

    /// st.pos.fill_(a): the window's first position.
    pub fn setPos(f: *const Forward, st: *const State, a: usize) !void {
        const v: i32 = @intCast(a);
        try f.ops().upload(st.pos, std.mem.asBytes(&v));
    }

    /// fused.layer for layer `li` over rows b.x[:R] at st.pos ..; `T`: the index range (null: no row selects).
    pub fn layer(f: *const Forward, b: *const Buffers, st: *const State, li: usize, R: usize, T: ?usize) !void {
        const c = f.w.config;
        const L = &f.w.layers[li];
        const D = c.hidden;
        const t_ = f.t();
        try t_.rmsnorm(b.x, D, L.input_norm, b.normed, D, R, D, c.eps);
        try f.attentionPart(b, st, L, li, R, T);
        try t_.residualAdd(b.x, b.part, R, D, f.w.world);
        try t_.rmsnorm(b.x, D, L.post_attn_norm, b.normed, D, R, D, c.eps);
        try f.ffnPart(b, L, R);
        try t_.residualAdd(b.x, b.part, R, D, f.w.world);
    }

    /// fused.attention_part (side "" and reuse 0) and attention_core: this rank's fp32 partial in b.part[:R].
    fn attentionPart(f: *const Forward, b: *const Buffers, st: *const State, L: *const W.Layer, li: usize, R: usize, T: ?usize) !void {
        const c = f.w.config;
        const o = f.ops();
        const t_ = f.t();
        const m = f.mla();
        const D = c.hidden;
        try o.linears(&.{ &L.q_a, &L.kv_a }, b.normed, .bf16, &.{ b.qa, b.kva }, &.{ .bf16, .bf16 }, R, b.lin);
        try t_.rmsnorm(b.qa, c.q_lora, L.q_a_norm, b.qn, c.q_lora, R, c.q_lora, c.eps);
        // keys(False): the latent and rope key, and on indexer layers the index key
        try t_.kvWrite(b.kva, b.kva_n, L.kv_a_norm, st.kc[li], st.pos, f.inv, c.eps, R, m);
        if (L.indexer) |*ix| {
            try t_.router(b.normed, D, ix.wk, b.ik, b.rpart, R, D, c.index_dim);
            try t_.ikWrite(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, st.ic[li], st.pos, R, c.index_dim, c.rope);
        }
        var q_done = false;
        if (L.indexer) |*ix| {
            if (T) |tt| {
                try t_.router(b.normed, D, ix.weights_proj, b.iw, b.rpart, R, D, c.index_heads);
                try o.linears(&.{ &ix.wq_b, &L.q_b }, b.qn, .bf16, &.{ b.iq, b.q }, &.{ .bf16, .bf16 }, R, b.lin);
                q_done = true;
                try t_.iqRope(b.iq, f.inv, st.pos, R, c.index_heads, c.index_dim, c.rope);
                try f.select(b, st.ic[li], st.pos, R, tt);
            }
        }
        // attention_core
        if (!q_done) try o.linear(&L.q_b, b.qn, .bf16, b.q, .bf16, R, b.lin);
        if (R > max_rows) return error.WindowTooWide; // TODO(P2): prompt chunks' bmm absorb / expand
        try t_.absorb(b.q, L.wk, f.inv, b.qlat, b.qrot, st.pos, R, m);
        const tiling = if (R <= b.small) tri.attn_decode else tri.attn_prompt;
        try t_.attention(b.qlat, b.qrot, st.kc[li], b.tok, st.pos, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
        try t_.expand(b.ol, L.wv, b.o, R, m);
        try o.linear(&L.o_proj, b.o, .bf16, b.part, .f32, R, b.lin);
    }

    /// fused.select (DCP 1) through the radix select for every window size: b.tok[:R] ascending.
    fn select(f: *const Forward, b: *const Buffers, ic: u64, pos: u64, R: usize, T: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        if (T > b.score_cols) return error.ScoreBufferTooSmall;
        if (T < c.index_topk) return error.Invalid;
        var r0: usize = 0;
        while (r0 < R) : (r0 += 128) {
            const n = @min(128, R - r0);
            const iq = b.iq + r0 * c.index_heads * c.index_dim * 2;
            const iw = b.iw + r0 * c.index_heads * 4;
            try t_.indexScores(iq, iw, ic, pos, b.sc, T, r0, n, c.index_heads, c.index_dim);
            try t_.selectKeys(b.sc, b.tok + r0 * c.index_topk * 4, T, n, c.index_topk);
        }
    }

    /// fused.ffn_part: the dense MLP, or the router, the shared expert (folded) and the routed experts.
    fn ffnPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        const D = c.hidden;
        if (L.experts) |*ex| {
            // route
            try t_.router(b.normed, D, L.router_w, b.mlog, b.rpart, R, D, c.experts);
            try t_.topk(b.mlog, L.router_bias, b.pick, b.wts, R, c.experts, c.top_k, c.routed_scaling, c.norm_topk);
            // experts_part, FOLD_SHARED: the shared expert first (into sy), then routed + sy
            try f.mlp(b, L, R, b.sy);
            const xs = b.xs orelse return error.NoExpertScratch;
            try f.ops().routed(b.normed, D, b.pick, b.wts, ex, xs, b.part, b.sy, R);
            return;
        }
        try f.mlp(b, L, R, b.part);
    }

    /// fused.mlp: gate and up (one launch), SwiGLU, down into `out` (fp32).
    fn mlp(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize, out: u64) !void {
        const o = f.ops();
        const width = L.mlp.gate.n;
        try o.linears(&.{ &L.mlp.gate, &L.mlp.up }, b.normed, .bf16, &.{ b.g, b.u }, &.{ .bf16, .bf16 }, R, b.lin);
        try f.t().swiglu2(b.g, b.u, b.act, R, width);
        try o.linear(&L.mlp.down, b.act, .bf16, out, .f32, R, b.lin);
    }
};

/// fused.inv_freq on the host (diagnostic: Phase 1 uploads the oracle's bytes, which torch computes on the GPU).
pub fn invFreq(out: []f32, dim: usize, theta: f64) void {
    for (out, 0..) |*v, i| {
        const x: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim));
        v.* = 1.0 / std.math.pow(f32, @floatCast(theta), x);
    }
}

test "inv_freq's first entries" {
    var v: [32]f32 = undefined;
    invFreq(&v, 64, 8000000.0);
    try std.testing.expectEqual(@as(f32, 1.0), v[0]);
    try std.testing.expect(v[31] < v[30] and v[31] > 0);
}
