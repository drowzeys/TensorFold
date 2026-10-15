//! Drama 3 generation on one GPU. A prompt (frame rows [main token, code 0..9]) runs through the backbone in chunks
//! of up to `max_rows` rows (cuBLAS GEMMs, the GDN recurrence, cuBLAS attention), then each chunk's rows through the
//! codebook tower with their next frame as label (AudioDecoder.prefill: the tower cache of every prompt row but the
//! last). A frame is then one step that replays as a CUDA graph: sample the main token and code 0 from the candidate
//! logits of the newest row (RAS on the last 16 main tokens), the 13 tower layers at that row's position with a
//! codebook head and a draw after each residual layer, record the frame, and feed it through the backbone at the next
//! position (one row, GEMV everywhere), whose final-norm hidden and candidate logits the next frame samples from.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");

const H = cfg.hidden;
const W = cfg.frame_width;

pub const Options = struct {
    max_ctx: usize = 4096, // positions (prompt rows + frames)
    max_rows: usize = 1024, // prompt rows a chunk
    attn_chunk: usize = 512, // query rows a prompt attention GEMM
    max_frames: usize = 2048,
    threads: usize = 16,
    pdl: bool = true, // GEMVs launched with programmatic dependent launch
    split: bool = true, // tower codes 2..9 on a second stream beside the next backbone row
    exact_attn_rows: usize = 0, // prompt chunks up to this many rows attend with the fp32 decode kernel
    fp8_act: bool = true, // the routed experts' activations FP8 per token and 128 inputs, as SGLang serves them
};

pub const Sampling = struct {
    temperature: f32 = 1.0,
    top_k: i32 = 30,
    top_p: f32 = 0.9,
    seed: u64 = 1,

    pub fn greedy(s: Sampling) bool {
        return s.temperature <= 0 or s.top_k == 1;
    }
};

/// Per-layer probe of a prompt pass (the oracle's dump names); called with device rows [t, width] bf16 or fp32.
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, f32_: bool) anyerror!void,
};

const Buf = cuda.DeviceBuffer;
const max_split_rows = 8; // decode rows the split-key attention takes

/// The scratch and stream one chain of launches uses (the backbone and the tower run concurrently in a frame).
const Lane = struct { s: cuda.Stream, xn: u64, tmp: u64, proj: u64, y: u64, q: u64, pos: u64, pml: u64, pacc: u64 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    o: Options,
    c: cfg.Config,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    stream2: cuda.Stream, // the tower's codes 2..9 in a frame, beside the next backbone row
    ev_fork: cuda.Event,
    ev_join: cuda.Event,
    k: kern.Kernels,
    w: *wts.Weights,
    bufs: std.ArrayList(Buf),
    // state
    kv_bb: [cfg.layers][2]u64, // full-attention layers: K, V [max_ctx, 2, 256]
    kv_tw: [cfg.tower_layers][2]u64,
    conv: [cfg.layers]u64, // GDN layers: [8192, 3] bf16
    gstate: [cfg.layers]u64, // GDN layers: [32, 128, 128] fp32
    state_bufs: [2]u64, // conv / gstate blocks (zeroed per request)
    state_bytes: [2]usize,
    // device scalars
    pos: u64, // i32: the newest row's position
    pos_bb: u64, // i32: the next backbone row's position (pos + 1 inside a frame)
    bb_input: u64, // i32 [11]: the next backbone row's frame (main, code 0, code 1)
    t_xn: u64, // the tower lane's scratch
    t_tmp: u64,
    t_proj: u64,
    t_y: u64,
    t_q: u64,
    pml: [2]u64, // split-key attention partials, a lane each
    pacc: [2]u64,
    step: u64, // u64: the sampler's step
    count: u64, // i32: frames made
    window: u64, // i32 [16]
    cur_input: u64, // i32 [11]: the frame the backbone takes next (at pos)
    frame: u64, // i32 [11]: the frame being sampled
    frames_out: u64, // i32 [max_frames, 11]
    pframes: u64, // i32 [max_ctx, 11]: the prompt rows
    // scratch (rows R = max_rows)
    x: u64, // residual stream [R, H] bf16
    xn: u64,
    tmp: u64, // [R, H]
    proj: u64, // [R, 12352] bf16
    conv_out: u64, // [R, 8192]
    o32: u64, // [R, 4096] fp32 (GDN core out)
    y: u64, // [R, 4096] bf16 (gated norm / attention out)
    q: u64, // [R, 16, 256]
    qg: u64,
    og: u64,
    scores: u64, // [2, 8 * attn_chunk, max_ctx] fp32
    vf: u64, // V cache rows in fp32 [max_ctx, 512] for the prompt PV GEMM
    sp: u64, // [R, 1288]
    sact: u64, // [R, 512]
    shout: u64, // [R, H]
    picks: u64, // i32 [R * 8]
    rw: u64, // f32 [R * 8]
    gu: u64, // [R * 8, 512] bf16
    down: u64, // [R * 8, H] fp32
    xq: u64, // [R, H] bf16 (e4m3 values)
    xsc: u64, // [R, H / 128] fp32
    gq: u64, // [R * 8, 512]
    gsc: u64, // [R * 8, 4]
    plan: cuda.grouped.Plan,
    hfin: u64, // [R, H] final-norm hidden
    th: [2]u64, // tower hidden ping-pong [R, H]
    mix: u64, // [R, 2H]
    mixed: u64, // [R, H]
    act: u64, // [R, 6144]
    hlast: u64, // [H] the newest row's final-norm hidden
    cand_logits: u64, // f32 [4097]
    code_logits: u64, // f32 [4096]
    // frame graph
    graph: ?cuda.graph.Exec = null,
    graph_sampling: Sampling = .{},
    sampling: Sampling = .{},
    probe_router: ?Probe = null, // diagnostics: every MoE layer's [shared gate | up | router | gate] rows
    rows_done: usize = 0, // positions in the cache (prompt rows; the frame step advances on the device)
    prompt_rows: usize = 0,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.o = o;
        e.graph = null;
        e.sampling = .{};
        e.graph_sampling = .{};
        e.rows_done = 0;
        e.prompt_rows = 0;
        e.probe_router = null;
        e.bufs = .empty;
        e.c = try cfg.Config.load(gpa, io, model_dir);
        e.driver = try cuda.Driver.open();
        e.ctx = try cuda.Context.init(&e.driver, 0);
        e.stream = try cuda.Stream.init(&e.driver, true);
        e.stream2 = try cuda.Stream.init(&e.driver, true);
        e.ev_fork = try cuda.Event.init(&e.driver, false);
        e.ev_join = try cuda.Event.init(&e.driver, false);
        e.k = try kern.Kernels.load(&e.driver, &e.ctx);
        e.k.rebind();
        e.k.pdl = o.pdl;
        e.w = try wts.load(gpa, io, &e.driver, model_dir, e.c, .{ .threads = o.threads });
        try e.allocState();
        return e;
    }

    pub fn destroy(e: *Engine) void {
        if (e.graph) |*g| g.deinit();
        for (e.bufs.items) |*b| b.free();
        e.bufs.deinit(e.gpa);
        e.w.deinit();
        e.gpa.destroy(e.w);
        e.k.blas.deinit();
        e.k.ws.free();
        e.k.blas_lib.close();
        e.k.fp8.unload();
        e.k.fp8a_mod.unload();
        e.k.experts_mod.unload();
        e.k.mod.unload();
        e.ev_fork.deinit();
        e.ev_join.deinit();
        e.stream2.deinit();
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    fn alloc(e: *Engine, bytes: usize) !u64 {
        const b = try Buf.alloc(&e.driver, @max(bytes, 256));
        try e.bufs.append(e.gpa, b);
        return b.ptr;
    }

    fn allocState(e: *Engine) !void {
        const R = e.o.max_rows;
        const ctxn = e.o.max_ctx;
        const kvb = ctxn * cfg.kv_heads * cfg.head_dim * 2;
        for (0..cfg.layers) |li| {
            if (e.c.full_attention[li]) {
                e.kv_bb[li] = .{ try e.alloc(kvb), try e.alloc(kvb) };
            } else e.kv_bb[li] = .{ 0, 0 };
        }
        for (0..cfg.tower_layers) |ti| e.kv_tw[ti] = .{ try e.alloc(kvb), try e.alloc(kvb) };
        var n_gdn: usize = 0;
        for (e.c.full_attention) |f| n_gdn += @intFromBool(!f);
        const conv_b = cfg.conv_channels * 3 * 2;
        const gs_b = cfg.lin_v_heads * cfg.lin_dim * cfg.lin_dim * 4;
        e.state_bytes = .{ n_gdn * conv_b, n_gdn * gs_b };
        e.state_bufs = .{ try e.alloc(e.state_bytes[0]), try e.alloc(e.state_bytes[1]) };
        var g: usize = 0;
        for (0..cfg.layers) |li| {
            if (e.c.full_attention[li]) {
                e.conv[li] = 0;
                e.gstate[li] = 0;
            } else {
                e.conv[li] = e.state_bufs[0] + g * conv_b;
                e.gstate[li] = e.state_bufs[1] + g * gs_b;
                g += 1;
            }
        }
        e.pos = try e.alloc(8);
        e.pos_bb = try e.alloc(8);
        e.bb_input = try e.alloc(W * 4);
        e.t_xn = try e.alloc(R * H * 2);
        e.t_tmp = try e.alloc(R * H * 2);
        e.t_proj = try e.alloc(R * cfg.gdn_proj * 2);
        e.t_y = try e.alloc(R * 4096 * 2);
        e.t_q = try e.alloc(R * 4096 * 2);
        for (0..2) |li| {
            e.pml[li] = try e.alloc(max_split_rows * 16 * kern.Kernels.attn_splits * 2 * 4);
            e.pacc[li] = try e.alloc(max_split_rows * 16 * kern.Kernels.attn_splits * 256 * 4);
        }
        e.step = try e.alloc(8);
        e.count = try e.alloc(8);
        e.window = try e.alloc(16 * 4);
        e.cur_input = try e.alloc(W * 4);
        e.frame = try e.alloc(W * 4);
        e.frames_out = try e.alloc(e.o.max_frames * W * 4);
        e.pframes = try e.alloc((ctxn + 1) * W * 4);
        e.x = try e.alloc(R * H * 2);
        e.xn = try e.alloc(R * H * 2);
        e.tmp = try e.alloc(R * H * 2);
        e.proj = try e.alloc(R * cfg.gdn_proj * 2);
        e.conv_out = try e.alloc(R * cfg.conv_channels * 2);
        e.o32 = try e.alloc(R * 4096 * 4);
        e.y = try e.alloc(R * 4096 * 2);
        e.q = try e.alloc(R * 4096 * 2);
        e.qg = try e.alloc(R * 4096 * 2);
        e.og = try e.alloc(R * 4096 * 4);
        const ac = @min(e.o.attn_chunk, R);
        e.scores = try e.alloc(2 * 8 * ac * (ctxn + 8) * 4);
        e.vf = try e.alloc((ctxn + 8) * 512 * 4);
        e.sp = try e.alloc(R * cfg.moe_proj * 2);
        e.sact = try e.alloc(R * cfg.shared_width * 2);
        e.shout = try e.alloc(R * H * 2);
        e.picks = try e.alloc(R * 8 * 4);
        e.rw = try e.alloc(R * 8 * 4);
        e.gu = try e.alloc(R * 8 * cfg.expert_width * 2);
        e.down = try e.alloc(R * 8 * H * 4);
        e.xq = try e.alloc(R * H * 2);
        e.xsc = try e.alloc(R * (H / 128) * 4);
        e.gq = try e.alloc(R * 8 * cfg.expert_width * 2);
        e.gsc = try e.alloc(R * 8 * (cfg.expert_width / 128) * 4);
        const pairs = R * 8;
        const items = cuda.grouped.maxItems(pairs, cfg.experts, 16);
        const nblk = (pairs + 1023) / 1024;
        e.plan = .{ .members = try e.alloc(pairs * 4), .items = try e.alloc(items * 3 * 4), .counts = try e.alloc(16), .rank = try e.alloc(pairs * 4), .hist = try e.alloc(nblk * cfg.experts * 4) };
        e.hfin = try e.alloc(R * H * 2);
        e.th = .{ try e.alloc(R * H * 2), try e.alloc(R * H * 2) };
        e.mix = try e.alloc(R * 2 * H * 2);
        e.mixed = try e.alloc(R * H * 2);
        e.act = try e.alloc(R * cfg.tower_ffn * 2);
        e.hlast = try e.alloc(H * 2);
        e.cand_logits = try e.alloc(e.w.n_cand * 4);
        e.code_logits = try e.alloc(cfg.codebook_size * 4);
    }

    fn buf(ptr: u64) Buf {
        return .{ .d = undefined, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    fn put(e: *Engine, dst: u64, bytes: []const u8) !void {
        var b = buf(dst);
        b.d = &e.driver;
        try b.uploadAsync(0, bytes, e.stream.handle);
    }

    fn get(e: *Engine, src: u64, out: []u8) !void {
        var b = buf(src);
        b.d = &e.driver;
        try b.downloadAsync(0, out, e.stream.handle);
        try e.stream.synchronize();
    }

    fn setPos(e: *Engine, p: usize) !void {
        const v: i32 = @intCast(p);
        try e.put(e.pos, std.mem.asBytes(&v));
        try e.stream.synchronize();
    }

    // ------------------------------------------------------------------------------------------------ forward

    const lay_bb: kern.Kernels.QkLayout = .{ .q_off = 0, .q_head_stride = 2 * cfg.head_dim, .k_off = 2 * cfg.heads * cfg.head_dim, .v_off = 2 * cfg.heads * cfg.head_dim + cfg.kv_heads * cfg.head_dim, .gate_off = cfg.head_dim, .gate_head_stride = 2 * cfg.head_dim };
    const lay_tw: kern.Kernels.QkLayout = .{ .q_off = 0, .q_head_stride = cfg.head_dim, .k_off = 2 * cfg.heads * cfg.head_dim, .v_off = 2 * cfg.heads * cfg.head_dim + cfg.kv_heads * cfg.head_dim, .gate_off = cfg.heads * cfg.head_dim, .gate_head_stride = cfg.head_dim };

    /// Gated attention of t rows at positions pos0 + i whose q / k / v are in e.proj (layout `lay`), K / V caches kc
    /// / vc; the gated output into e.y [t, 4096]. `start` is the first row's position (host side, for the GEMMs).
    fn attention(e: *Engine, ln: Lane, t: usize, start: usize, lay: kern.Kernels.QkLayout, qn: u64, kn: u64, kc: u64, vc: u64, interleaved: bool) !void {
        const k = &e.k;
        const s = ln.s;
        const ps = cfg.attn_proj;
        try k.qkRope(s, ln.proj, ps, lay, qn, kn, ln.pos, ln.q, kc, vc, t, interleaved, cfg.rope_theta, cfg.eps);
        if (t == 1 or t <= e.o.exact_attn_rows) {
            if (t <= max_split_rows) return k.attnSplit(s, ln.q, kc, vc, ln.pos, ln.pml, ln.pacc, ln.proj, ps, lay, ln.y, t);
            return k.attnDecode(s, ln.q, kc, vc, ln.pos, ln.proj, ps, lay, ln.y, t);
        }
        std.debug.assert(ln.pos == e.pos and ln.s.handle == e.stream.handle); // prompt rows: the main stream
        // prompt rows: per chunk of query rows, scores = q K^T / 16 (fp32), causal softmax (bf16), P V
        const ac = @min(e.o.attn_chunk, e.o.max_rows);
        var c0: usize = 0;
        while (c0 < t) : (c0 += ac) {
            const tc = @min(ac, t - c0);
            const L = start + c0 + tc;
            const ld = (L + 7) / 8 * 8;
            if (c0 > 0) try e.setPos(start + c0);
            const qrow = ln.q + c0 * 4096 * 2;
            try k.qGroup(s, qrow, e.qg, tc);
            const m8 = 8 * tc;
            try k.gemmEx(s, .t, .n, L, m8, 256, 0.0625, kc, 512, 256, e.qg, 256, m8 * 256, e.scores, ld, m8 * ld, 2, true);
            try k.causalSoftmax(s, e.scores, tc, ld, e.pos);
            try k.bf16F32(s, vc, e.vf, L * 512);
            try k.gemmTyped(s, .n, .n, 256, m8, L, 1.0, e.vf, 512, 256, e.scores, ld, m8 * ld, e.og, 256, m8 * 256, 2, .f32, .f32);
            try k.attnGate(s, e.og, ln.proj + c0 * ps * 2, ps, lay, ln.y + c0 * 4096 * 2, tc);
        }
        if (t > ac) try e.setPos(start);
    }

    /// One backbone layer over t rows: e.x (residual stream) in and out.
    fn laneBB(e: *Engine, pos: u64) Lane {
        return .{ .s = e.stream, .xn = e.xn, .tmp = e.tmp, .proj = e.proj, .y = e.y, .q = e.q, .pos = pos, .pml = e.pml[0], .pacc = e.pacc[0] };
    }

    fn laneTW(e: *Engine, s: cuda.Stream) Lane {
        return .{ .s = s, .xn = e.t_xn, .tmp = e.t_tmp, .proj = e.t_proj, .y = e.t_y, .q = e.t_q, .pos = e.pos, .pml = e.pml[1], .pacc = e.pacc[1] };
    }

    fn backboneLayer(e: *Engine, ln: Lane, li: usize, t: usize, start: usize) !void {
        const k = &e.k;
        const s = ln.s;
        const L = &e.w.layers[li];
        try k.rmsnorm(s, e.x, H, L.in_norm, e.xn, H, t, H, cfg.eps, true);
        if (L.full) {
            try k.linear(s, L.attn.qkv, e.xn, H, H, cfg.attn_proj, e.proj, cfg.attn_proj, t);
            try e.attention(ln, t, start, lay_bb, L.attn.q_norm, L.attn.k_norm, e.kv_bb[li][0], e.kv_bb[li][1], false);
            try k.linear(s, L.attn.o, e.y, 4096, 4096, H, e.tmp, H, t);
        } else {
            const ps = cfg.gdn_proj;
            try k.linear(s, L.gdn.proj, e.xn, H, H, ps, e.proj, ps, t);
            try k.convSilu(s, e.proj, ps, t, L.gdn.conv, e.conv[li], e.conv_out, cfg.conv_channels);
            try k.gdn(s, e.conv_out, e.proj, ps, t, L.gdn.a_log, L.gdn.dt_bias, e.gstate[li], e.o32);
            try k.gatedNorm(s, e.o32, e.proj, ps, L.gdn.norm, e.y, t, cfg.eps);
            try k.linear(s, L.gdn.out, e.y, 4096, 4096, H, e.tmp, H, t);
        }
        try k.addRmsnorm(s, e.x, e.tmp, H, L.post_norm, e.xn, t, H, cfg.eps);
        try e.moe(s, L, t);
        if (e.probe_router) |p| {
            var nb: [32]u8 = undefined;
            try p.f(p.ctx, try std.fmt.bufPrint(&nb, "router.layer{d:0>2}", .{li}), e.sp, t, cfg.moe_proj, false);
        }
    }

    fn moe(e: *Engine, s: cuda.Stream, L: *const wts.Layer, t: usize) !void {
        const k = &e.k;
        const mp = cfg.moe_proj;
        try k.linear(s, L.moe_in, e.xn, H, H, mp, e.sp, mp, t);
        try k.router(s, e.sp, mp, 2 * cfg.shared_width, e.picks, e.rw, t);
        try k.swiglu(s, e.sp, mp, 0, cfg.shared_width, cfg.shared_width, e.sact, t);
        try k.linear(s, L.shared_down, e.sact, cfg.shared_width, cfg.shared_width, H, e.shout, H, t);
        const pairs = t * cfg.top_k;
        const prompt = t > 2;
        const tile: usize = if (prompt) cuda.experts.promptTile(.fp8g) else 16;
        try k.plan.route(s, e.picks, pairs, cfg.experts, tile, e.plan);
        const items = cuda.grouped.maxItems(pairs, cfg.experts, tile);
        if (e.o.fp8_act) {
            try k.actQuant(s, e.xn, H, e.xq, H, e.xsc, t, H);
            try k.experts(s, true, prompt, e.xq, H, cfg.top_k, e.xsc, L.experts, e.plan, e.gu, items);
            try k.actQuant(s, e.gu, cfg.expert_width, e.gq, cfg.expert_width, e.gsc, pairs, cfg.expert_width);
            try k.experts(s, false, prompt, e.gq, cfg.expert_width, 0, e.gsc, L.experts, e.plan, e.down, items);
        } else {
            try k.experts(s, true, prompt, e.xn, H, cfg.top_k, 0, L.experts, e.plan, e.gu, items);
            try k.experts(s, false, prompt, e.gu, cfg.expert_width, 0, 0, L.experts, e.plan, e.down, items);
        }
        try k.moeCombine(s, e.x, e.down, e.rw, e.shout, e.sp, mp, 2 * cfg.shared_width + cfg.experts, H, t);
    }

    /// Tower layer ti over t rows: hidden hin -> hout; label frames lf (rows lfs ints apart), input frames inf.
    fn towerLayer(e: *Engine, ln: Lane, ti: usize, t: usize, start: usize, hin: u64, hout: u64, lf: u64, lfs: usize, inf: u64) !void {
        const k = &e.k;
        const s = ln.s;
        const T = &e.w.tower[ti];
        const code_col: usize = if (ti < cfg.shared_layers) 1 else 1 + (ti - cfg.shared_layers);
        try k.towerMix(s, lf, lfs, code_col, inf, ti == 0, T.table, e.w.cb_emb, hin, T.enorm, T.hnorm, e.mix, H, e.c.ids.sem_begin, e.c.ids.sem_end, cfg.codebook_size, cfg.eps, t);
        try k.linear(s, T.eh, e.mix, 2 * H, 2 * H, H, e.mixed, H, t);
        try k.rmsnorm(s, e.mixed, H, T.attn_norm, ln.xn, H, t, H, cfg.eps, true);
        try k.linear(s, T.qkv, ln.xn, H, H, cfg.attn_proj, ln.proj, cfg.attn_proj, t);
        try e.attention(ln, t, start, lay_tw, T.q_norm, T.k_norm, e.kv_tw[ti][0], e.kv_tw[ti][1], true);
        try k.linear(s, T.wo, ln.y, 4096, 4096, H, ln.tmp, H, t);
        try k.addRmsnorm(s, e.mixed, ln.tmp, H, T.ffn_norm, ln.xn, t, H, cfg.eps); // resid = o + mixed
        try k.linear(s, T.w13, ln.xn, H, H, 2 * cfg.tower_ffn, ln.proj, 2 * cfg.tower_ffn, t);
        try k.swiglu(s, ln.proj, 2 * cfg.tower_ffn, 0, cfg.tower_ffn, cfg.tower_ffn, e.act, t);
        try k.linear(s, T.w2, e.act, cfg.tower_ffn, cfg.tower_ffn, H, ln.tmp, H, t);
        try k.addRmsnorm(s, e.mixed, ln.tmp, H, T.final_norm, hout, t, H, cfg.eps);
    }

    // ------------------------------------------------------------------------------------------------ requests

    /// Zero the recurrent states and the frame counters; the caches need no clearing (rows are written before read).
    pub fn reset(e: *Engine) !void {
        for (0..2) |i| {
            var b = buf(e.state_bufs[i]);
            b.d = &e.driver;
            b.len = e.state_bytes[i];
            try b.fill8(0, e.stream.handle);
        }
        const zero: [2]u64 = .{ 0, 0 };
        try e.put(e.count, std.mem.asBytes(&zero));
        try e.put(e.step, std.mem.asBytes(&zero));
        e.rows_done = 0;
        e.prompt_rows = 0;
        try e.stream.synchronize();
    }

    /// Run the prompt rows (int32 [n][11]) through the backbone and the tower; leaves the frame step ready.
    pub fn prefill(e: *Engine, rows: []const [W]i32, probe: ?Probe) !void {
        const n = rows.len;
        if (n < 2) return error.PromptTooShort;
        if (n + 2 > e.o.max_ctx) return error.PromptTooLong;
        try e.reset();
        try e.put(e.pframes, std.mem.sliceAsBytes(rows));
        const k = &e.k;
        const s = e.stream;
        const R = e.o.max_rows;
        if (probe != null and n > R) return error.ProbeNeedsOneChunk;
        var c0: usize = 0;
        while (c0 < n) : (c0 += R) {
            const t = @min(R, n - c0);
            try e.setPos(c0);
            try k.embedBackbone(s, e.pframes + c0 * W * 4, e.w.embed, e.w.cb_emb, e.x, t, H, e.c.ids.sem_begin, e.c.ids.sem_end, cfg.codebook_size);
            if (probe) |p| try p.f(p.ctx, "backbone_input_embeds", e.x, t, H, false);
            for (0..cfg.layers) |li| {
                try e.backboneLayer(e.laneBB(e.pos), li, t, c0);
                if (probe) |p| {
                    var nb: [64]u8 = undefined;
                    try p.f(p.ctx, try std.fmt.bufPrint(&nb, "backbone_hidden.layer{d:0>2}", .{li}), e.x, t, H, false);
                }
            }
            try k.rmsnorm(s, e.x, H, e.w.final_norm, e.hfin, H, t, H, cfg.eps, true);
            if (probe) |p| try p.f(p.ctx, "backbone_final_norm", e.hfin, t, H, false);
            // the tower over this chunk's rows that have a next prompt row
            const tt = @min(c0 + t, n - 1) - c0;
            if (tt > 0) {
                var hin = e.hfin;
                for (0..cfg.tower_layers) |ti| {
                    const hout = e.th[ti % 2];
                    try e.towerLayer(e.laneTW(e.stream), ti, tt, c0, hin, hout, e.pframes + (c0 + 1) * W * 4, W, e.pframes + c0 * W * 4);
                    if (probe) |p| {
                        var nb: [64]u8 = undefined;
                        try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_prefill.layer{d:0>2}", .{ti}), hout, tt, H, false);
                    }
                    hin = hout;
                }
            }
            if (c0 + t == n) try e.copyRow(e.hfin + (t - 1) * H * 2, e.hlast, H * 2);
        }
        // decode state: the last prompt row is the backbone's newest row; RAS window = its last 16 main tokens
        try e.setPos(n - 1);
        try e.put(e.cur_input, std.mem.asBytes(&rows[n - 1]));
        var win: [16]i32 = @splat(-1);
        const m = @min(16, n);
        for (0..m) |i| win[16 - m + i] = rows[n - m + i][0];
        try e.put(e.window, std.mem.asBytes(&win));
        const one: u64 = 1;
        try e.put(e.step, std.mem.asBytes(&one));
        try e.stream.synchronize();
        e.prompt_rows = n;
        e.rows_done = n;
    }

    /// Parity: the tower prefill alone over prompt rows 0..n-2 from final-norm hidden rows hfin (host bf16 [n, H]),
    /// with the prompt rows prefill() left on the device (one chunk).
    pub fn towerOnly(e: *Engine, n: usize, hfin: []const u8, probe: ?Probe) !void {
        if (n > e.o.max_rows) return error.ProbeNeedsOneChunk;
        try e.put(e.hfin, hfin);
        try e.setPos(0);
        var hin = e.hfin;
        for (0..cfg.tower_layers) |ti| {
            const hout = e.th[ti % 2];
            try e.towerLayer(e.laneTW(e.stream), ti, n - 1, 0, hin, hout, e.pframes + W * 4, W, e.pframes);
            if (probe) |p| {
                var nb: [64]u8 = undefined;
                try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_prefill.layer{d:0>2}", .{ti}), hout, n - 1, H, false);
            }
            hin = hout;
        }
    }

    fn copyRow(e: *Engine, src: u64, dst: u64, bytes: usize) !void {
        var b = buf(dst);
        b.d = &e.driver;
        try b.copyFrom(0, src, bytes, e.stream.handle);
    }

    fn spOf(s: Sampling) kern.Sampling {
        return .{ .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .seed = s.seed };
    }

    /// The frame step's launches (one graph): sample, tower, record, backbone row at the next position.
    fn enqueueFrame(e: *Engine, smp: Sampling, probe: ?Probe) !void {
        return e.enqueueFrameForced(smp, probe, null);
    }

    /// Teacher forcing (parity checks): each pick goes to force.picks and is then replaced by force.frame's value
    /// (device int32 [11] each), so the tower and the next backbone row take the forced frame.
    pub const Force = struct { frame: u64, picks: u64 };

    fn enqueueFrameForced(e: *Engine, smp: Sampling, probe: ?Probe, force: ?Force) !void {
        const k = &e.k;
        const s = e.stream;
        const sem_begin = e.c.ids.sem_begin;
        const greedy = smp.greedy();
        const sp = spOf(smp);
        // the tail of the tower on the second stream (not under a probe: its reads would race)
        const split = e.o.split and probe == null;
        try k.gemv(s, e.w.cand_head, e.hlast, H, e.w.n_cand, 0, e.cand_logits);
        if (probe) |p| try p.f(p.ctx, "tower_decode0.candidate_logits", e.cand_logits, 1, e.w.n_cand, true);
        if (force) |fc| {
            try k.argmax(s, e.cand_logits, e.w.n_cand, e.w.cand_ids, fc.picks, sem_begin, cfg.codebook_size);
            try k.copyI32(s, fc.frame, e.frame, 2);
        } else if (greedy) {
            try k.argmax(s, e.cand_logits, e.w.n_cand, e.w.cand_ids, e.frame, sem_begin, cfg.codebook_size);
        } else {
            try k.sampleMain(s, e.cand_logits, e.w.n_cand, e.w.cand_ids, e.window, sp, e.step, e.frame, sem_begin, cfg.codebook_size);
        }
        var hin = e.hlast;
        var ts = s;
        for (0..cfg.tower_layers) |ti| {
            const hout = e.th[ti % 2];
            try e.towerLayer(e.laneTW(ts), ti, 1, 0, hin, hout, e.frame, W, e.cur_input);
            if (probe) |p| {
                var nb: [64]u8 = undefined;
                try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_decode0.layer{d:0>2}", .{ti}), hout, 1, H, false);
            }
            hin = hout;
            if (ti >= cfg.shared_layers) {
                const hi = ti - cfg.shared_layers;
                try k.gemv(ts, e.w.heads[hi], hout, H, cfg.codebook_size, 0, e.code_logits);
                if (probe) |p| {
                    var nb: [64]u8 = undefined;
                    try p.f(p.ctx, try std.fmt.bufPrint(&nb, "tower_decode0.head{d}_logits", .{hi}), e.code_logits, 1, cfg.codebook_size, true);
                }
                const out = e.frame + (2 + hi) * 4;
                if (force) |fc| {
                    try k.argmax(ts, e.code_logits, cfg.codebook_size, 0, fc.picks + (2 + hi) * 4, 0, 0);
                    try k.copyI32(ts, fc.frame + (2 + hi) * 4, out, 1);
                } else if (greedy) {
                    try k.argmax(ts, e.code_logits, cfg.codebook_size, 0, out, 0, 0);
                } else {
                    try k.sampleCode(ts, e.code_logits, cfg.codebook_size, sp, e.step, @intCast(2 + hi), out);
                }
                if (hi == 0) {
                    // main token, code 0 and code 1 drawn: the next backbone row can start
                    try k.nextInput(s, e.frame, e.bb_input, e.pos, e.pos_bb);
                    if (split) {
                        try e.ev_fork.record(s);
                        try e.stream2.wait(e.ev_fork);
                        ts = e.stream2;
                    }
                }
            }
        }
        if (split) try e.ev_join.record(e.stream2);
        // the backbone row of the new frame at pos + 1
        const bb = e.laneBB(e.pos_bb);
        try k.embedBackbone(s, e.bb_input, e.w.embed, e.w.cb_emb, e.x, 1, H, sem_begin, e.c.ids.sem_end, cfg.codebook_size);
        for (0..cfg.layers) |li| try e.backboneLayer(bb, li, 1, 0);
        try k.rmsnorm(s, e.x, H, e.w.final_norm, e.hlast, H, 1, H, cfg.eps, true);
        if (split) try s.wait(e.ev_join);
        try k.finishFrame(s, e.frame, e.frames_out, e.cur_input, e.window, e.pos, e.step, e.count);
    }

    /// One teacher-forced greedy frame, eager.
    pub fn frameForced(e: *Engine, force: Force) !void {
        try e.enqueueFrameForced(.{ .temperature = 0, .top_k = 1 }, null, force);
    }

    /// One frame (eager, or replaying the captured graph when `graphs`).
    pub fn frameStep(e: *Engine, smp: Sampling, graphs: bool, probe: ?Probe) !void {
        if (!graphs or probe != null) return e.enqueueFrame(smp, probe);
        if (e.graph == null or !std.meta.eql(e.graph_sampling, smp)) {
            if (e.graph) |*g| g.deinit();
            e.graph = null;
            try cuda.graph.beginCapture(e.stream, .thread_local);
            e.enqueueFrame(smp, null) catch |er| {
                var g = cuda.graph.endCapture(e.stream) catch return er;
                g.deinit();
                return er;
            };
            var g = try cuda.graph.endCapture(e.stream);
            defer g.deinit();
            e.graph = try g.instantiate();
            e.graph_sampling = smp;
        }
        try e.graph.?.launchOn(e.stream);
    }

    pub const Result = struct {
        frames: [][W]i32, // without the end-of-speech frame
        eos: bool,
        prefill_ms: f64,
        first_frame_ms: f64, // from the request's start to the first frame on the host
        decode_ms: f64,
    };

    /// Generate up to max_frames frames after prompt rows; stops at the end-of-speech frame.
    pub fn generate(e: *Engine, io: std.Io, rows: []const [W]i32, smp: Sampling, max_frames: usize, graphs: bool, sync_every: usize) !Result {
        const t0 = nowNs(io);
        try e.prefill(rows, null);
        const t1 = nowNs(io);
        const limit = @min(max_frames, e.o.max_frames, e.o.max_ctx - rows.len - 1);
        var made: usize = 0;
        var eos = false;
        var first_ms: f64 = 0;
        const every = @max(sync_every, 1);
        const host = try e.gpa.alloc([W]i32, limit);
        errdefer e.gpa.free(host);
        var checked: usize = 0;
        while (made < limit and !eos) {
            const burst = @min(if (made == 0) 1 else every, limit - made);
            for (0..burst) |_| try e.frameStep(smp, graphs, null);
            made += burst;
            try e.get(e.frames_out + checked * W * 4, std.mem.sliceAsBytes(host[checked..made]));
            if (checked == 0) first_ms = ms(t0, nowNs(io));
            for (host[checked..made], checked..) |f, i| if (f[0] == e.c.ids.eos) {
                eos = true;
                made = i;
                break;
            };
            checked = made;
        }
        const t2 = nowNs(io);
        return .{ .frames = host[0..made], .eos = eos, .prefill_ms = ms(t0, t1), .first_frame_ms = first_ms, .decode_ms = ms(t1, t2) };
    }

    pub fn download(e: *Engine, src: u64, out: []u8) !void {
        return e.get(src, out);
    }
};

pub fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn ms(a: i128, b: i128) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e6;
}
