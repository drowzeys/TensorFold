//! One Liquid stack on the device (LFM2 or LFM3 layers: gated short conv or attention, then SwiGLU), with its caches.
//! LFM3-Omni runs four: the speech encoder (LFM2), the backbone, the depth decoder and the codec decoder (LFM3).
const std = @import("std");
const cuda = @import("cuda");
const st = @import("core").safetensors;
const kern = @import("kernels.zig");
const spec = @import("spec.zig");

pub const Kind = spec.Kind;
pub const Rope = spec.Rope;
pub const Spec = spec.Spec;

pub const Layer = struct {
    kind: Kind,
    op_norm: u64,
    ffn_norm: u64,
    post_op_norm: u64 = 0,
    post_ffn_norm: u64 = 0,
    // conv
    in_proj: u64 = 0, // [3D, D]
    conv_w: u64 = 0, // [D, K]
    out_proj: u64 = 0, // [D, D]
    state: u64 = 0, // [D, K-1]
    // attention
    qkvg: u64 = 0, // [q | k | v | gate, D]
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    o_proj: u64 = 0, // [D, H * hd]
    kc: u64 = 0, // [max_ctx, Hkv * hd]
    vc: u64 = 0,
    // feed-forward
    w13: u64 = 0, // [2F, D]
    w2: u64 = 0, // [D, F]
};

pub const Stack = struct {
    s: Spec,
    layers: []Layer,
    final_norm: u64,
    inv_full: u64,
    inv_sliding: u64,
    // scratch
    xn: u64,
    proj: u64,
    y: u64,
    tmp: u64,
    states: u64,
    states_bytes: usize,
    /// Rows already in the caches (the next row's position), and the same on the device (i32) for replayed graphs.
    pos: usize = 0,
    dpos: u64 = 0,
    d: *const cuda.Driver,

    pub fn nq(st_: *const Stack) usize {
        return st_.s.heads * st_.s.head_dim;
    }
    pub fn nk(st_: *const Stack) usize {
        return st_.s.kv_heads * st_.s.head_dim;
    }
    fn qkvgCols(st_: *const Stack) usize {
        return st_.nq() + 2 * st_.nk() + if (st_.s.lfm3) st_.s.heads else 0;
    }

    /// Clears the conv states and the cache position (the caches' rows are overwritten as they fill).
    pub fn reset(x: *Stack, s: cuda.Stream) !void {
        x.pos = 0;
        try x.d.check(x.d.api.cuMemsetD8Async(x.dpos, 0, 4, s.handle), "cuMemsetD8Async");
        if (x.states_bytes > 0) try x.d.check(x.d.api.cuMemsetD8Async(x.states, 0, x.states_bytes, s.handle), "cuMemsetD8Async");
    }

    /// Runs `t` rows of `x` (the residual stream, [t, D], updated in place) at the cache position, then the final
    /// norm into `out` ([t, D]); the rows join the caches.
    pub fn forward(x_: *Stack, k: *const kern.Kernels, s: cuda.Stream, x: u64, t: usize, out: u64) !void {
        return x_.forwardAt(k, s, x, t, out, false);
    }

    /// `forward`; with `graph` one row at the device's position (the launches a captured graph replays).
    pub fn forwardAt(x_: *Stack, k: *const kern.Kernels, s: cuda.Stream, x: u64, t: usize, out: u64, graph: bool) !void {
        const sp = x_.s;
        if (graph and t != 1) return error.GraphRowsAreOne;
        const posp: u64 = if (graph) x_.dpos else 0;
        const D = sp.d;
        if (t == 0) return;
        if (t > sp.max_rows) return error.TooManyRows;
        if (x_.pos + t > sp.max_ctx) return error.ContextFull;
        const nq_ = x_.nq();
        const nk_ = x_.nk();
        const cols = x_.qkvgCols();
        var pending: u64 = 0; // the last block's output, added to the residual by the next norm
        for (x_.layers) |*l| {
            try k.rmsnorm(s, x, pending, l.op_norm, x_.xn, D, D, D, t, sp.eps, .lfm);
            switch (l.kind) {
                .conv => {
                    try k.linear(s, l.in_proj, x_.xn, D, D, 3 * D, x_.proj, 3 * D, t);
                    try k.shortConv(s, x_.proj, 3 * D, l.conv_w, l.state, x_.y, D, t, D, sp.conv_k);
                    try k.linear(s, l.out_proj, x_.y, D, D, D, x_.tmp, D, t);
                },
                .full, .sliding => {
                    try k.linear(s, l.qkvg, x_.xn, D, D, cols, x_.proj, cols, t);
                    try k.headNorm(s, x_.proj, l.q_norm, t, sp.heads, sp.head_dim, cols, sp.eps);
                    try k.headNorm(s, x_.proj + nq_ * 2, l.k_norm, t, sp.kv_heads, sp.head_dim, cols, sp.eps);
                    const inv = if (l.kind == .sliding) x_.inv_sliding else x_.inv_full;
                    try k.rope(s, x_.proj, inv, t, sp.heads, sp.head_dim, cols, x_.pos, sp.rope_mode, posp);
                    try k.rope(s, x_.proj + nq_ * 2, inv, t, sp.kv_heads, sp.head_dim, cols, x_.pos, sp.rope_mode, posp);
                    const row: usize = if (graph) 0 else x_.pos;
                    try k.copyRows(s, x_.proj + nq_ * 2, l.kc + row * nk_ * 2, t, nk_, cols, nk_, posp);
                    try k.copyRows(s, x_.proj + (nq_ + nk_) * 2, l.vc + row * nk_ * 2, t, nk_, cols, nk_, posp);
                    const windowed = l.kind == .sliding or (sp.window_all and sp.window > 0);
                    try k.attention(s, .{ .q = x_.proj, .ldq = cols, .kc = l.kc, .vc = l.vc, .ldkv = nk_, .o = x_.y, .ldo = nq_, .t = t, .heads = sp.heads, .kv_heads = sp.kv_heads, .hd = sp.head_dim, .pos0 = x_.pos, .kv_len = x_.pos + t, .window = if (windowed) sp.window else 0, .causal = sp.causal, .posp = posp, .max_ctx = sp.max_ctx });
                    if (sp.lfm3) try k.gate(s, x_.y, nq_, x_.proj + (nq_ + 2 * nk_) * 2, cols, t, sp.heads, sp.head_dim);
                    try k.linear(s, l.o_proj, x_.y, nq_, nq_, D, x_.tmp, D, t);
                },
            }
            if (sp.sandwich) try k.rmsnorm(s, x_.tmp, 0, l.post_op_norm, x_.tmp, D, D, D, t, sp.eps, .lfm);
            try k.rmsnorm(s, x, x_.tmp, l.ffn_norm, x_.xn, D, D, D, t, sp.eps, .lfm);
            try k.linear(s, l.w13, x_.xn, D, D, 2 * sp.ffn, x_.proj, 2 * sp.ffn, t);
            try k.swiglu(s, x_.proj, x_.y, t, sp.ffn);
            try k.linear(s, l.w2, x_.y, sp.ffn, sp.ffn, D, x_.tmp, D, t);
            if (sp.sandwich) try k.rmsnorm(s, x_.tmp, 0, l.post_ffn_norm, x_.tmp, D, D, D, t, sp.eps, .lfm);
            pending = x_.tmp;
        }
        try k.rmsnorm(s, x, pending, x_.final_norm, out, D, D, D, t, sp.eps, .lfm);
        x_.pos += t;
        if (graph) try k.pos(s, x_.dpos, 1, false) else try k.pos(s, x_.dpos, x_.pos, true);
    }
};

/// Device buffers a loader hands out and frees together.
pub const Arena = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn alloc(a: *Arena, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(a.d, n);
        errdefer b.free();
        try a.bufs.append(a.gpa, b);
        a.bytes += n;
        return b.ptr;
    }

    pub fn zeroed(a: *Arena, n: usize) !u64 {
        const p = try a.alloc(n);
        try a.d.check(a.d.api.cuMemsetD8_v2(p, 0, n), "cuMemsetD8");
        return p;
    }

    pub fn upload(a: *Arena, bytes: []const u8) !u64 {
        const p = try a.alloc(bytes.len);
        try a.bufs.items[a.bufs.items.len - 1].upload(0, bytes);
        return p;
    }

    pub fn deinit(a: *Arena) void {
        for (a.bufs.items) |*b| b.free();
        a.bufs.deinit(a.gpa);
    }
};

/// A bf16 tensor of the given shape, or a named refusal.
pub fn tensor(f: *const st.File, name: []const u8, shape: []const usize) ![]const u8 {
    const t = f.get(name) orelse {
        std.log.err("lfm3_omni: {s} is missing", .{name});
        return error.MissingTensor;
    };
    if (!t.is(.bf16, shape)) {
        std.log.err("lfm3_omni: {s} is {s} {any}, expected bf16 {any}", .{ name, @tagName(t.dtype), t.shape[0..t.rank], shape });
        return error.BadTensor;
    }
    return t.bytes;
}

fn one(a: *Arena, f: *const st.File, name: []const u8, shape: []const usize) !u64 {
    return a.upload(try tensor(f, name, shape));
}

/// Tensors of `k` columns stacked by rows into one matrix.
fn stacked(a: *Arena, f: *const st.File, names: []const []const u8, rows: []const usize, k: usize) !u64 {
    var total: usize = 0;
    for (rows) |r| total += r;
    const p = try a.alloc(total * k * 2);
    const buf = a.bufs.items[a.bufs.items.len - 1];
    var at: usize = 0;
    for (names, rows) |n, r| {
        try buf.upload(at * k * 2, try tensor(f, n, &.{ r, k }));
        at += r;
    }
    return p;
}

/// Inverse frequencies of `hd / 2` pairs, zero past the rotated share, as the reference computes them in fp32.
fn invFreq(gpa: std.mem.Allocator, a: *Arena, r: Rope, hd: usize) !u64 {
    const v = try gpa.alloc(f32, hd / 2);
    defer gpa.free(v);
    spec.invFreq(v, r, hd);
    return a.upload(std.mem.sliceAsBytes(v));
}

/// Loads a stack's weights from `f` and allocates its caches and scratch in `a`.
pub fn load(gpa: std.mem.Allocator, a: *Arena, f: *const st.File, sp: Spec) !Stack {
    const D = sp.d;
    const hd = sp.head_dim;
    var x: Stack = undefined;
    x.s = sp;
    x.d = a.d;
    x.pos = 0;
    x.layers = try gpa.alloc(Layer, sp.kinds.len);
    var nbuf: [5][256]u8 = undefined;
    const P = sp.prefix;
    var conv_layers: usize = 0;
    for (sp.kinds) |kd| conv_layers += @intFromBool(kd == .conv);
    x.states_bytes = conv_layers * D * (sp.conv_k - 1) * 2;
    x.states = if (x.states_bytes > 0) try a.zeroed(x.states_bytes) else 0;
    var conv_i: usize = 0;
    for (sp.kinds, 0..) |kd, i| {
        var l: Layer = .{ .kind = kd, .op_norm = 0, .ffn_norm = 0 };
        const n0 = try std.fmt.bufPrint(&nbuf[0], "{s}layers.{d}.", .{ P, i });
        l.op_norm = try one(a, f, try cat(&nbuf[1], n0, "operator_norm.weight"), &.{D});
        l.ffn_norm = try one(a, f, try cat(&nbuf[1], n0, "ffn_norm.weight"), &.{D});
        if (sp.sandwich) {
            l.post_op_norm = try one(a, f, try cat(&nbuf[1], n0, "post_operator_norm.weight"), &.{D});
            l.post_ffn_norm = try one(a, f, try cat(&nbuf[1], n0, "post_ffn_norm.weight"), &.{D});
        }
        switch (kd) {
            .conv => {
                l.in_proj = try one(a, f, try cat(&nbuf[1], n0, "conv.in_proj.weight"), &.{ 3 * D, D });
                l.conv_w = try one(a, f, try cat(&nbuf[1], n0, "conv.conv.weight"), &.{ D, 1, sp.conv_k });
                l.out_proj = try one(a, f, try cat(&nbuf[1], n0, "conv.out_proj.weight"), &.{ D, D });
                l.state = x.states + conv_i * D * (sp.conv_k - 1) * 2;
                conv_i += 1;
            },
            .full, .sliding => {
                const nq_ = sp.heads * hd;
                const nk_ = sp.kv_heads * hd;
                const q = try cat(&nbuf[1], n0, "self_attn.q_proj.weight");
                const kk = try cat(&nbuf[2], n0, "self_attn.k_proj.weight");
                const v = try cat(&nbuf[3], n0, "self_attn.v_proj.weight");
                if (sp.lfm3) {
                    const g = try cat(&nbuf[4], n0, "self_attn.gate_proj.weight");
                    l.qkvg = try stacked(a, f, &.{ q, kk, v, g }, &.{ nq_, nk_, nk_, sp.heads }, D);
                    l.q_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.q_norm.weight"), &.{hd});
                    l.k_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.k_norm.weight"), &.{hd});
                    l.o_proj = try one(a, f, try cat(&nbuf[1], n0, "self_attn.o_proj.weight"), &.{ D, nq_ });
                } else {
                    l.qkvg = try stacked(a, f, &.{ q, kk, v }, &.{ nq_, nk_, nk_ }, D);
                    l.q_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.q_layernorm.weight"), &.{hd});
                    l.k_norm = try one(a, f, try cat(&nbuf[1], n0, "self_attn.k_layernorm.weight"), &.{hd});
                    l.o_proj = try one(a, f, try cat(&nbuf[1], n0, "self_attn.out_proj.weight"), &.{ D, nq_ });
                }
                l.kc = try a.alloc(sp.max_ctx * nk_ * 2);
                l.vc = try a.alloc(sp.max_ctx * nk_ * 2);
            },
        }
        const w1 = try cat(&nbuf[1], n0, "feed_forward.w1.weight");
        const w3 = try cat(&nbuf[2], n0, "feed_forward.w3.weight");
        l.w13 = try stacked(a, f, &.{ w1, w3 }, &.{ sp.ffn, sp.ffn }, D);
        l.w2 = try one(a, f, try cat(&nbuf[1], n0, "feed_forward.w2.weight"), &.{ D, sp.ffn });
        x.layers[i] = l;
    }
    x.final_norm = try one(a, f, try cat(&nbuf[1], P, "embedding_norm.weight"), &.{D});
    x.dpos = try a.zeroed(4);
    x.inv_full = try invFreq(gpa, a, sp.rope_full, hd);
    x.inv_sliding = try invFreq(gpa, a, sp.rope_sliding, hd);
    const R = sp.max_rows;
    const cols = @max(3 * D, @max(sp.heads * hd + 2 * sp.kv_heads * hd + sp.heads, 2 * sp.ffn));
    x.xn = try a.alloc(R * D * 2);
    x.proj = try a.alloc(R * cols * 2);
    x.y = try a.alloc(R * @max(D, @max(sp.ffn, sp.heads * hd)) * 2);
    x.tmp = try a.alloc(R * D * 2);
    return x;
}

fn cat(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}
