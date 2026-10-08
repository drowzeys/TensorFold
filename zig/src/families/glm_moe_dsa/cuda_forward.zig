//! GLM-5.3's fused forward for windows of 1 to 128 rows (fused.compute / layer / attention_part / attention_core /
//! ffn_part / head), one stream, one rank of the TP world.
//!
//! Exchanges (fused.gather's all-gather branch): every rank's fp32 partial [world, R, D] in rank order, then
//! glue.residual_add sums the world slots in rank order (WORLD = world), so every rank gets the same bits.
//! Phase 2a: `Forward.comm` set - NCCL all-gather of the real partials on the compute stream (the Python engine's
//! all-gather path for every window: tools/glm53/reference.py gives its engine a communicator without all_reduce /
//! all_to_all, so prompt windows too take this branch). `comm` null - the Phase 1 identity communicator (oracle.py's
//! LocalComm): b.gath zeroed, this rank's partial in its slot. Every kernel computes each row alone, so a row's bits
//! never depend on the window: the outputs match the Python engine's bit for bit when the kernels and launches do.
//! The head (fused.head mode "argmax"): this rank's logits over its vocabulary share (glue.router), its first
//! maximum (host: torch.argmax's rule), the [value, id, 0, 0] records all-gathered, the lowest rank of the maxima.
//!
//! Frozen as tools/glm53/oracle.py freezes the Python side: bf16 latent cache, DCP 1, one stream (TF_GLM53_SIDE=0:
//! the side stream gives the same bits), x3linear.plan tiles, TensorFold's expert layout, the shared expert folded
//! into the routed combine. Selection always goes through the radix select (topk.top_columns over _index_scores'
//! 4-byte order words): the same keys as torch.topk + sort, which the champion uses for decode-sized windows (the
//! oracle checks the two agree).
//! Phase 2b: the device head (`headDevice`: fused.head "argmax" / "local" / "draft" with no host step, so a window
//! replays as a CUDA graph), ids on the device (b.ids, torch.index_select's rows), the MTP layer (`mtpCompute`,
//! fused.mtp_compute with TF_GLM53_MTP_REUSE's keep / reuse) and the decode-class buffers (`BufOpts.decode`).
//! TODO(P3): the RoCE one-shot / exact reduce-scatter exchanges, prompt chunks over 128 rows (x3prefill GEMM, bmm
//! absorb/expand), DCP, int8/int4 latent caches.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const exl3 = @import("exl3.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const W = @import("cuda_weights.zig");
const smp = @import("sampling.zig");

/// The widest TP world the head's exchange records hold.
pub const max_world: usize = 16;

/// One head pick: the token, this rank's local argmax and the gathered [world, 4] fp32 records (bit-compared with
/// the Python engine's b.amax_all).
pub const Pick = struct {
    token: u32 = 0,
    local: usize = 0,
    world: usize,
    gathered: [max_world * 4]f32 = @splat(0),
};

/// fused.FAST_ROWS: windows up to this many rows use the decode attention tiling (non-decode Buffers).
pub const fast_rows: usize = 16;
/// fused.MAX_ROWS: the widest window of the row-invariant EXL3 linear (this port's limit for now).
pub const max_rows: usize = 128;
/// fused.DECODE_ROWS: the widest decode-class buffer set.
pub const decode_rows: usize = 32;

pub const BufOpts = struct {
    /// fused.Buffers(decode=True): verify and MTP windows
    decode: bool = false,
    /// the MTP layer's scratch (hin, me, mh, mcat, mx32, eh_proj partials)
    mtp: bool = false,
};

/// What a window's head computes (fused.head's modes; `none`: no head).
pub const HeadMode = enum { none, argmax, local, draft };

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
    gath: u64 = 0, // fp32 [world * rows * D]: every rank's partial in rank order (fused.gather)
    // the head (allocated when the weights hold one): fused.Buffers fnormed / lpart / amax / amax_all
    head_rows: usize = 0, // min(rows, small)
    fnormed: u64 = 0, // bf16 [head_rows, D]
    lg: u64 = 0, // fp32 [head_rows, vocab_part]
    hpart: u64 = 0, // fp32 [ROUTER_KS, head_rows, vocab_part] (glue.router's partials)
    amax: u64 = 0, // fp32 [head_rows, 4]: [max logit, global id, 0, 0] a row
    amax_all: u64 = 0, // fp32 [world, head_rows, 4]
    // Phase 2b: device ids, the hidden rows the MTP layer reads and writes, its scratch, the device head
    ids: u64 = 0, // int64 [rows]
    hin: u64 = 0, // bf16 [rows, D]: the hidden rows the MTP layer reads
    hidden: u64 = 0, // bf16 [rows, D]: the window's final (target) or chain (MTP) hidden rows
    ht: u64 = 0, // bf16 [rows, D]: target_hidden_for_mtp of a prompt chunk
    me: u64 = 0, // bf16 [rows, D]
    mh: u64 = 0,
    mcat: u64 = 0, // bf16 [rows, 2 D]
    mx32: u64 = 0, // fp32 [rows, D]
    epart: u64 = 0, // fp32 [ROUTER_KS, rows, D]: eh_proj's router partials
    argmax: u64 = 0, // int64 [head_rows]
    gsum: u64 = 0, // fp32 [head_rows, D / 64]: the draft head's input group sums
    head_v: usize = 0, // b.lg holds head_rows rows of up to this many logits (max(vocab_part, draft rows))
    cand_words: usize = 0, // words a rank sends for sampled candidates: head_rows * 2 * max_candidates + 1
    cand_send: u64 = 0, // fp32 [cand_words]
    cand_all: u64 = 0, // fp32 [world * cand_words]
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
        return initWith(gpa, d, w, rows, score_cols, .{});
    }

    /// `opts.decode` (fused.Buffers(decode=True)): every window these buffers serve is a decode / verify window,
    /// whatever its width: the decode attention tiling and head rows up to max(rows, FAST_ROWS).
    pub fn initWith(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, rows: usize, score_cols: usize, opts: BufOpts) !Buffers {
        const c = w.config;
        if (rows > max_rows) return error.WindowTooWide;
        if (opts.decode and rows > decode_rows) return error.WindowTooWide;
        var b: Buffers = .{ .d = d, .gpa = gpa, .rows = rows, .score_cols = score_cols, .small = if (opts.decode) @max(rows, fast_rows) else fast_rows };
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
        b.gath = try b.alloc(w.world * rows * D * 4);
        if (w.lm_head != 0) {
            const hr = @min(rows, b.small);
            const V = @max(w.vocab_part, if (w.draft) |dh| dh.n else 0);
            b.head_rows = hr;
            b.head_v = V;
            b.fnormed = try b.alloc(hr * D * 2);
            b.lg = try b.alloc(hr * V * 4);
            b.hpart = try b.alloc(tri.router_ks * hr * V * 4);
            b.amax = try b.alloc(hr * 4 * 4);
            b.amax_all = try b.alloc(w.world * hr * 4 * 4);
            b.argmax = try b.alloc(hr * 8);
            b.gsum = try b.alloc(hr * (D / 64) * 4);
            b.cand_words = hr * 2 * smp.max_candidates + 1;
            b.cand_send = try b.alloc(b.cand_words * 4);
            b.cand_all = try b.alloc(w.world * b.cand_words * 4);
        }
        b.ids = try b.alloc(rows * 8);
        b.hidden = try b.alloc(rows * D * 2);
        if (opts.mtp) {
            b.hin = try b.alloc(rows * D * 2);
            b.ht = try b.alloc(rows * D * 2);
            b.me = try b.alloc(rows * D * 2);
            b.mh = try b.alloc(rows * D * 2);
            b.mcat = try b.alloc(rows * 2 * D * 2);
            b.mx32 = try b.alloc(rows * D * 4);
            b.epart = try b.alloc(tri.router_ks * rows * D * 4);
        }
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
    mkc: u64 = 0, // the MTP layer's latent rows (0: no MTP layer)
    mic: u64 = 0, // the MTP layer's index keys (0: no indexer there)
    mpos: u64 = 0, // int32 [1]: the MTP window's first position
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
        if (w.mtp) |*m| {
            s.mkc = try s.zeros(rows * c.latentWidth() * 2);
            s.mic = if (m.layer.indexer != null) try s.zeros(rows * c.index_dim * 2) else 0;
            s.mpos = try s.zeros(4);
        }
        return s;
    }

    /// fused.State.bytes_for: what the caches of `capacity` tokens take (the cache guard's figure).
    pub fn bytesFor(w: *const W.Weights, capacity: usize) usize {
        const c = w.config;
        var per: usize = 0;
        for (w.layers) |*L| per += c.latentWidth() * 2 + (if (L.indexer != null) c.index_dim * 2 else 0);
        if (w.mtp) |*m| per += c.latentWidth() * 2 + (if (m.layer.indexer != null) c.index_dim * 2 else 0);
        return per * (capacity + 1);
    }

    /// A fresh request: every cache row and the position zeroed (fused.State's torch.zeros).
    pub fn reset(s: *State) !void {
        for (s.mem.items) |*m| try m.fill8(0, null);
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
    inv: u64, // fp32 [rope / 2]: fused.inv_freq (Phase 1: the oracle's bytes; Phase 2a: Kernels.invFreq's)
    /// the world's NCCL communicator (rank and world as the weights' cut); null: the identity exchange of Phase 1
    comm: ?*const cuda.nccl.Communicator = null,

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
        return f.layerWith(b, &f.w.layers[li], st.kc[li], st.ic[li], st.pos, R, T, 0);
    }

    /// fused.layer for any layer (the MTP layer too): its caches kc / ic (0: no indexer), positions from `pos`;
    /// `reuse` (TF_GLM53_MTP_REUSE, one-row MTP draft steps): attend b.tok[0] instead of selecting.
    pub fn layerWith(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, T: ?usize, reuse: u32) !void {
        const c = f.w.config;
        const D = c.hidden;
        const t_ = f.t();
        try t_.rmsnorm(b.x, D, L.input_norm, b.normed, D, R, D, c.eps);
        try f.attentionPart(b, L, kc, ic, pos, R, T, reuse);
        try t_.residualAdd(b.x, try f.gather(b, R), R, D, f.w.world);
        try t_.rmsnorm(b.x, D, L.post_attn_norm, b.normed, D, R, D, c.eps);
        try f.ffnPart(b, L, R);
        try t_.residualAdd(b.x, try f.gather(b, R), R, D, f.w.world);
    }

    /// fused.gather: [world, R, D] fp32 partials in rank order for residual_add. World 1: b.part itself. NCCL:
    /// comm.all_gather(b.part[:R], b.gath[:world R D]) on the compute stream. Phase 1 (other ranks absent,
    /// LocalComm): zeros in every slot but this rank's, which holds b.part[:R].
    pub fn gather(f: *const Forward, b: *const Buffers, R: usize) !u64 {
        const world = f.w.world;
        if (world == 1) return b.part;
        const D = f.w.config.hidden;
        if (f.comm) |c| {
            try c.allGather(b.part, b.gath, R * D, .f32, f.s.handle);
            return b.gath;
        }
        const o = f.ops();
        try o.fill32(b.gath, 0, world * R * D);
        try o.copy(b.gath + f.w.rank * R * D * 4, b.part, R * D * 4);
        return b.gath;
    }

    /// fused.attention_part (side "") and attention_core: this rank's fp32 partial in b.part[:R]. `reuse` 1 / 2
    /// (an MTP draft step after its chain's first, one row): no index key written, no selection; the row attends
    /// b.tok[0] (mode 1 puts its own position in the list's last slot).
    fn attentionPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, T: ?usize, reuse: u32) !void {
        const c = f.w.config;
        const o = f.ops();
        const t_ = f.t();
        const m = f.mla();
        const D = c.hidden;
        if (reuse != 0 and (L.indexer == null or R != 1)) return error.Invalid;
        try o.linears(&.{ &L.q_a, &L.kv_a }, b.normed, .bf16, &.{ b.qa, b.kva }, &.{ .bf16, .bf16 }, R, b.lin);
        try t_.rmsnorm(b.qa, c.q_lora, L.q_a_norm, b.qn, c.q_lora, R, c.q_lora, c.eps);
        // keys(False): the latent and rope key, and on indexer layers (not under reuse) the index key
        try t_.kvWrite(b.kva, b.kva_n, L.kv_a_norm, kc, pos, f.inv, c.eps, R, m);
        if (L.indexer) |*ix| {
            if (reuse == 0) {
                try t_.router(b.normed, D, ix.wk, b.ik, b.rpart, R, D, c.index_dim);
                try t_.ikWrite(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, ic, pos, R, c.index_dim, c.rope);
            }
        }
        if (reuse == 1 and T != null) try o.copy(b.tok + (c.index_topk - 1) * 4, pos, 4); // b.tok[0, K-1:] = pos
        var q_done = false;
        if (L.indexer) |*ix| {
            if (T) |tt| {
                if (reuse == 0) {
                    try t_.router(b.normed, D, ix.weights_proj, b.iw, b.rpart, R, D, c.index_heads);
                    try o.linears(&.{ &ix.wq_b, &L.q_b }, b.qn, .bf16, &.{ b.iq, b.q }, &.{ .bf16, .bf16 }, R, b.lin);
                    q_done = true;
                    try t_.iqRope(b.iq, f.inv, pos, R, c.index_heads, c.index_dim, c.rope);
                    try f.select(b, ic, pos, R, tt);
                }
            }
        }
        // attention_core
        if (!q_done) try o.linear(&L.q_b, b.qn, .bf16, b.q, .bf16, R, b.lin);
        if (R > max_rows) return error.WindowTooWide; // TODO(P3): prompt chunks' bmm absorb / expand
        try t_.absorb(b.q, L.wk, f.inv, b.qlat, b.qrot, pos, R, m);
        const tiling = if (R <= b.small) tri.attn_decode else tri.attn_prompt;
        try t_.attention(b.qlat, b.qrot, kc, b.tok, pos, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
        try t_.expand(b.ol, L.wv, b.o, R, m);
        try o.linear(&L.o_proj, b.o, .bf16, b.part, .f32, R, b.lin);
    }

    // ------------------------------------------------------------------------------- Phase 2b: device windows ---

    /// fused.compute for rows b.ids[:R] (device ids) at st.pos ..: the target's caches written, its final hidden in
    /// b.hidden[:R], then `head` over every row (`last`: only row R - 1, as logits="last"). No host step: capturable.
    pub fn compute(f: *const Forward, b: *const Buffers, st: *const State, R: usize, T: ?usize, hm: HeadMode, last: bool) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        try o.embedRows(f.w.embed, b.ids, D, b.x, R, false);
        for (0..f.w.layers.len) |li| try f.layer(b, st, li, R, T);
        try o.copy(b.hidden, b.x, R * D * 2);
        if (hm == .none) return;
        if (last) return f.headDevice(b, b.x + (R - 1) * D * 2, f.w.final_norm, 1, hm);
        return f.headDevice(b, b.x, f.w.final_norm, R, hm);
    }

    /// fused.head on n rows from `x` (bf16, row stride D) normed with `norm`: "local" leaves this rank's logits in
    /// b.lg[:n] ([n, vocab_part], contiguous); "argmax" / "draft" leave the token in b.argmax[:n] (int64) after exchanging
    /// each rank's first maximum ([value, id, -, -] records, b.amax_all) and taking the lowest rank among the equal
    /// maxima. "draft": the 4-bit draft head over the reduced vocabulary (headq q4), ids through its table.
    pub fn headDevice(f: *const Forward, b: *const Buffers, x: u64, norm: u64, n: usize, mode: HeadMode) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        const t_ = f.t();
        const world = f.w.world;
        if (f.w.lm_head == 0 or b.lg == 0) return error.NoHead;
        if (n > b.head_rows) return error.WindowTooWide;
        try t_.rmsnorm(x, D, norm, b.fnormed, D, n, D, c.eps);
        const V = f.w.vocab_part;
        switch (mode) {
            .none => return,
            .local => return t_.router(b.fnormed, D, f.w.lm_head, b.lg, b.hpart, n, D, V),
            .argmax => {
                try t_.router(b.fnormed, D, f.w.lm_head, b.lg, b.hpart, n, D, V);
                try o.argmaxRec(b.lg, V, V, 0, f.w.vocab_off, b.amax, n);
            },
            .draft => {
                const dh = f.w.draft orelse return error.NoDraftHead;
                try t_.groupSums(b.fnormed, D, b.gsum, n, D);
                try o.q4Head(b.fnormed, b.gsum, dh.w, dh.scales, dh.biases, dh.n, dh.npad, dh.k, dh.sk, b.lg, n);
                try o.argmaxRec(b.lg, dh.n, dh.n, dh.ids, 0, b.amax, n);
            },
        }
        if (f.comm) |cm| {
            try cm.allGather(b.amax, b.amax_all, n * 4, .f32, f.s.handle);
        } else if (world > 1) {
            try o.fill32(b.amax_all, 0, world * n * 4);
            try o.copy(b.amax_all + f.w.rank * n * 16, b.amax, n * 16);
        } else {
            try o.copy(b.amax_all, b.amax, n * 16);
        }
        try o.pickRank(b.amax_all, world, n, b.argmax);
    }

    /// The MTP head's mode: the reduced 4-bit draft vocabulary when loaded, else the full share's argmax.
    pub fn draftMode(f: *const Forward) HeadMode {
        return if (f.w.draft != null) .draft else .argmax;
    }

    pub const MtpOpts = struct {
        /// head on the last row (logits="last"); false: logits="none" (prompt rows)
        head: bool = true,
        /// position 0's embedding zeroed (the first prompt chunk)
        zero_first: bool = false,
        /// TF_GLM53_MTP "x/normed": the chain passes on shared_head-normed rows
        chain_normed: bool = true,
        /// a chain's first step keeps its last row's selection in b.tok[0]
        keep_sel: bool = false,
        /// later one-row steps attend b.tok[0] (TF_GLM53_MTP_REUSE 1 / 2)
        reuse: u32 = 0,
    };

    /// fused.mtp_compute: rows (b.hin[:n] = the previous position's hidden, b.ids[:n] = the token) through the MTP
    /// layer at st.mpos ..; b.hidden[:n] the chain's rows; with `head` the last row's draft in b.argmax[0].
    pub fn mtpCompute(f: *const Forward, b: *const Buffers, st: *const State, n: usize, T: ?usize, opts: MtpOpts) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        const t_ = f.t();
        const m = f.w.mtp orelse return error.NoMtpLayer;
        if (b.mcat == 0) return error.NoMtpScratch;
        try o.embedRows(f.w.embed, b.ids, D, b.me, n, opts.zero_first);
        try t_.rmsnorm(b.me, D, m.enorm, b.mh, D, n, D, c.eps);
        try t_.rmsnorm(b.hin, D, m.hnorm, b.normed, D, n, D, c.eps);
        try t_.cat2(b.mh, b.normed, b.mcat, n, D);
        try t_.router(b.mcat, 2 * D, m.eh_proj, b.mx32, b.epart, n, 2 * D, D);
        try o.f32ToBf16(b.mx32, b.x, n * D);
        try f.layerWith(b, &f.w.mtp.?.layer, st.mkc, st.mic, st.mpos, n, T, opts.reuse);
        if (opts.keep_sel and T != null and n > 1) try o.copy(b.tok, b.tok + (n - 1) * c.index_topk * 4, c.index_topk * 4);
        if (opts.chain_normed) {
            try t_.rmsnorm(b.x, D, m.head_norm, b.hidden, D, n, D, c.eps);
        } else {
            try o.copy(b.hidden, b.x, n * D * 2);
        }
        if (opts.head) try f.headDevice(b, b.x + (n - 1) * D * 2, m.head_norm, 1, f.draftMode());
    }

    /// fused.target_hidden_for_mtp (normed): rmsnorm(src.hidden[row0 .. row0 + n], final_norm) into `out` rows.
    pub fn targetHidden(f: *const Forward, src: *const Buffers, row0: usize, n: usize, out: u64) !void {
        const c = f.w.config;
        const D = c.hidden;
        try f.t().rmsnorm(src.hidden + row0 * D * 2, D, f.w.final_norm, out, D, n, D, c.eps);
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

    /// fused.head(x, final_norm, R, slice(R - 1, R), mode="argmax") for the window's last row: the final RMSNorm,
    /// this rank's logits over its vocabulary share (glue.router into b.lg), its first maximum (on the host, from
    /// b.lg: torch.argmax's rule), the [value, global id, 0, 0] record all-gathered (b.amax_all) and the lowest rank
    /// among the equal maxima. `host`: at least vocab_part floats. Synchronizes the stream (the pick goes to the host).
    pub fn head(f: *const Forward, b: *const Buffers, R: usize, host: []f32) !Pick {
        const c = f.w.config;
        const D = c.hidden;
        const V = f.w.vocab_part;
        const world = f.w.world;
        if (f.w.lm_head == 0 or b.lg == 0) return error.NoHead;
        if (host.len < V or world > max_world) return error.Invalid;
        const t_ = f.t();
        const o = f.ops();
        try t_.rmsnorm(b.x + (R - 1) * D * 2, D, f.w.final_norm, b.fnormed, D, 1, D, c.eps);
        try t_.router(b.fnormed, D, f.w.lm_head, b.lg, b.hpart, 1, D, V);
        try o.download(std.mem.sliceAsBytes(host[0..V]), b.lg);
        try f.s.synchronize();
        const i = argmaxFirst(host[0..V]);
        const rec: [4]f32 = .{ host[i], @floatFromInt(i + f.w.vocab_off), 0, 0 };
        try o.upload(b.amax, std.mem.asBytes(&rec));
        if (f.comm) |cm| {
            try cm.allGather(b.amax, b.amax_all, 4, .f32, f.s.handle);
        } else {
            try o.fill32(b.amax_all, 0, world * 4);
            try o.copy(b.amax_all + f.w.rank * 16, b.amax, 16);
        }
        var p: Pick = .{ .world = world };
        try o.download(std.mem.sliceAsBytes(p.gathered[0 .. world * 4]), b.amax_all);
        try f.s.synchronize();
        var vals: [max_world]f32 = undefined;
        for (0..world) |r| vals[r] = p.gathered[r * 4];
        const best = argmaxFirst(vals[0..world]);
        const id = p.gathered[best * 4 + 1];
        if (!(id >= 0 and id < @as(f32, @floatFromInt(c.vocab)))) return error.BadPick;
        p.token = @intFromFloat(id);
        p.local = i;
        return p;
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

/// fused.inv_freq on the host: a diagnostic only. torch's bytes come from libdevice's powf on the GPU, which this
/// does not reproduce exactly; Kernels.invFreq runs the same powf (glm53_rope.cu) and is what the engine uses.
pub fn invFreq(out: []f32, dim: usize, theta: f64) void {
    for (out, 0..) |*v, i| {
        const x: f32 = @as(f32, @floatFromInt(2 * i)) * (@as(f32, 1.0) / @as(f32, @floatFromInt(dim)));
        const p: f32 = @floatCast(std.math.pow(f64, @as(f64, @as(f32, @floatCast(theta))), @as(f64, x)));
        v.* = 1.0 / p;
    }
}

/// torch.argmax's choice over one row: the first maximum, NaN above every number (the first NaN wins).
pub fn argmaxFirst(v: []const f32) usize {
    var best: usize = 0;
    for (v, 0..) |x, i| {
        if (i == 0) continue;
        if (std.math.isNan(v[best])) break;
        if (std.math.isNan(x) or x > v[best]) best = i;
    }
    return best;
}

test "argmax: first maximum, NaN first" {
    try std.testing.expectEqual(@as(usize, 1), argmaxFirst(&.{ 1, 3, 3, 2 }));
    try std.testing.expectEqual(@as(usize, 2), argmaxFirst(&.{ 1, 3, std.math.nan(f32), std.math.nan(f32) }));
    try std.testing.expectEqual(@as(usize, 0), argmaxFirst(&.{ 0.0, -0.0 }));
    try std.testing.expectEqual(@as(usize, 0), argmaxFirst(&.{5}));
}

test "inv_freq's first entries" {
    var v: [32]f32 = undefined;
    invFreq(&v, 64, 8000000.0);
    try std.testing.expectEqual(@as(f32, 1.0), v[0]);
    try std.testing.expect(v[31] < v[30] and v[31] > 0);
}
