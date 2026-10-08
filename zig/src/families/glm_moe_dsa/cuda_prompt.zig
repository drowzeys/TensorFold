//! GLM-5.3's served prompt path in Zig (Phase 3a): the pieces of fused.py a prompt chunk of thousands of rows takes,
//! with the Python engine's kernels and launch shapes, so a chunk's bits are the served engine's.
//!
//! `Wide` (windows over fused.MAX_ROWS = 128 rows): x3prefill.matmul - the input rotated (ext.rot_in), W_q decoded
//! once (ext.unpack; kept in an UnpackCache-like arena, TF_GLM53_UNPACK_CACHE_MB), the Triton prompt GEMM (_gemm,
//! captured from the reference) - and attention_core's torch.bmm absorb / expand through cuBLAS with torch's own
//! arguments (bgemm: cublasGemmStridedBatchedEx, bf16, fp32 compute, the default tensor-op algorithm), then the copy
//! into fused's [R, H, X] view and _qrope.
//!
//! `Sp` (TF_GLM53_PROMPT_SP=1, fused.compute_prompt_sp): a chunk as two micro-batches (rows [0, R/2) in b0, the rest
//! in b1), each split into `world` blocks of own rows. The main stream runs every row's head- and width-sharded work
//! (attention_core, the MLP / experts); a half's reduce-scatter (NCCL ring, bf16) -> own-row residual add, norms,
//! replicated projections (q_a, kv_a, the indexer's wk / weights_proj / wq_b, the router) and the indexer's top-k on
//! own rows -> in-place all-gathers chain runs on the comm stream under the other half's main-stream work. Every
//! kernel is the one the Python engine launches on the same rows, so the order of the two streams changes no bit.
//!
//! What is not row-exact (as in Python): the prompt GEMM's tiles, cuBLAS, NCCL's ring sums, the prompt experts. A
//! prompt's bits are reproducible run to run only with TF_EXL3_PROMPT_DET=slots16 (prompt_experts' slot-order sum),
//! which the Phase 3a gate runs both engines with.
const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("exl3.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const W = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");

fn tyName(y: exl3.DType) []const u8 {
    return switch (y) {
        .bf16 => "*bf16",
        .f32 => "*fp32",
        .f16 => "*fp16",
    };
}

/// x3prefill.UnpackCache: decoded W_q kept for the next calls of the same linear (the other half of a chunk),
/// oldest first out. A ring arena: an entry lands at the head (wrapping to 0), dropping the entries it overlaps.
/// Users run on one stream, so stream order keeps an overwritten entry's readers before the overwrite.
pub const UnpackCache = struct {
    const Entry = struct { key: u64, off: usize, len: usize };
    arena: cuda.DeviceBuffer,
    head: usize = 0,
    entries: std.ArrayList(Entry) = .empty,
    gpa: std.mem.Allocator,
    hits: usize = 0,
    misses: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) !UnpackCache {
        return .{ .arena = try cuda.DeviceBuffer.alloc(d, bytes), .gpa = gpa };
    }

    pub fn deinit(c: *UnpackCache) void {
        c.entries.deinit(c.gpa);
        c.arena.free();
    }

    /// The decoded W_q of `l` (decoded now on a miss); null when it is larger than the arena.
    pub fn get(c: *UnpackCache, o: kern.Ops, l: *const exl3.Linear) !?u64 {
        const len = std.mem.alignForward(usize, l.k * l.n * 2, 256);
        for (c.entries.items) |e| {
            if (e.key == l.words) {
                c.hits += 1;
                return c.arena.ptr + e.off;
            }
        }
        if (len > c.arena.len) return null;
        if (c.head + len > c.arena.len) c.head = 0;
        const lo = c.head;
        const hi = c.head + len;
        var i: usize = 0;
        while (i < c.entries.items.len) {
            const e = c.entries.items[i];
            if (e.off < hi and lo < e.off + e.len) {
                _ = c.entries.swapRemove(i);
            } else i += 1;
        }
        try c.entries.append(c.gpa, .{ .key = l.words, .off = lo, .len = len });
        c.head = hi;
        c.misses += 1;
        try o.unpack(l, c.arena.ptr + lo);
        return c.arena.ptr + lo;
    }
};

/// What windows over max_rows rows need on one stream: x3prefill.Workspace (the rotated input, a decoded W_q when
/// the cache does not take it, the Hadamard matrix), the main stream's UnpackCache and cuBLAS handle.
pub const Wide = struct {
    xh: cuda.DeviceBuffer, // fp16 [rows, K]
    w: cuda.DeviceBuffer, // fp16 [K, N]
    had: cuda.DeviceBuffer, // bf16 [128, 128]
    cache: ?*UnpackCache = null,
    blas: ?*const cuda.cublas.Blas = null,

    /// `rows`: the widest window; `cache` (main stream) / `blas` (main stream: attention_core runs there).
    pub fn init(d: *const cuda.Driver, k: *const kern.Kernels, s: cuda.Stream, w: *const W.Weights, rows: usize, cache: ?*UnpackCache, blas: ?*const cuda.cublas.Blas) !Wide {
        var kmax: usize = 0;
        var knmax: usize = 0;
        var all: std.ArrayList(*exl3.Linear) = .empty;
        defer all.deinit(w.gpa);
        try @import("tiles.zig").tunable(@constCast(w), &all, w.gpa);
        for (all.items) |l| {
            kmax = @max(kmax, l.k);
            knmax = @max(knmax, l.k * l.n);
        }
        var wd: Wide = .{ .xh = try cuda.DeviceBuffer.alloc(d, rows * kmax * 2), .w = undefined, .had = undefined, .cache = cache, .blas = blas };
        errdefer wd.xh.free();
        wd.w = try cuda.DeviceBuffer.alloc(d, knmax * 2);
        errdefer wd.w.free();
        wd.had = try cuda.DeviceBuffer.alloc(d, 128 * 128 * 2);
        errdefer wd.had.free();
        const o: kern.Ops = .{ .k = k, .s = s };
        try o.hadamard(wd.had.ptr);
        return wd;
    }

    pub fn deinit(wd: *Wide) void {
        wd.had.free();
        wd.w.free();
        wd.xh.free();
    }

    /// x3prefill.matmul: out [M, N] (row stride N, dtype y) = x [M, K] @ W (no bias) on f's stream.
    pub fn matmul(wd: *const Wide, f: *const fw.Forward, l: *const exl3.Linear, x: u64, M: usize, out: u64, y: exl3.DType) !void {
        const o: kern.Ops = .{ .k = f.k, .s = f.s };
        if (M * l.k * 2 > wd.xh.len) return error.ScratchTooSmall;
        try o.rotIn(x, .bf16, l.suh, wd.xh.ptr, M, l.k);
        var wq: u64 = 0;
        if (wd.cache) |c| wq = (try c.get(o, l)) orelse 0;
        if (wq == 0) {
            if (l.k * l.n * 2 > wd.w.len) return error.ScratchTooSmall;
            try o.unpack(l, wd.w.ptr);
            wq = wd.w.ptr;
        }
        const tl = exl3.prefillTiles(l.k, l.n);
        const t_: tri.Tri = .{ .set = &f.k.triton, .s = f.s };
        try t_.prefillGemm(wd.xh.ptr, wq, wd.had.ptr, l.svh, out, tyName(y), M, l.n, l.k, l.n, tl.bm, tl.bk, tl.group, tl.warps, tl.stages, exl3.had_scale);
    }

    /// attention_core's wide absorb: b.qlat[:R] = torch.bmm(q_nope [H, R, nope], wk [H, nope, lw]) copied into the
    /// [R, H, lw] view (the bmm result staged in b.ol, free until the attention writes it), then _qrope.
    pub fn absorb(wd: *const Wide, f: *const fw.Forward, b: *const fw.Buffers, L: *const W.Layer, pos: u64, R: usize) !void {
        const blas = wd.blas orelse return error.NoCublas;
        const c = f.w.config;
        const H = f.w.heads;
        const qd = c.qkDim();
        const lw = c.kv_lora;
        // baddbmm_out_cuda_impl, result [H, R, lw] contiguous -> transpose_result: m = lw, n = R, k = nope;
        // batch1_ = wk ('n', lda lw, stride nope lw), batch2_ = q's view ('n', ldb H qd, stride qd), ldc lw
        try blas.bgemmBf16(f.s.handle, .n, .n, lw, R, c.nope, L.wk, lw, c.nope * lw, b.q, H * qd, qd, b.ol, lw, R * lw, H);
        const o: kern.Ops = .{ .k = f.k, .s = f.s };
        try o.bhxToHbx(b.ol, b.qlat, H, R, lw);
        const t_: tri.Tri = .{ .set = &f.k.triton, .s = f.s };
        try t_.qrope(b.q, f.inv, b.qrot, pos, R, .{ .heads = H, .nope = c.nope, .rope = c.rope, .lw = lw, .vd = c.v_dim, .topk = c.index_topk });
    }

    /// attention_core's wide expand: b.o[:R] (as [R, H, v]) = torch.bmm(ol [H, R, lw], wv^T [H, lw, v]) transposed
    /// back (the bmm result staged in b.qlat, spent by now).
    pub fn expand(wd: *const Wide, f: *const fw.Forward, b: *const fw.Buffers, L: *const W.Layer, R: usize) !void {
        const blas = wd.blas orelse return error.NoCublas;
        const c = f.w.config;
        const H = f.w.heads;
        const lw = c.kv_lora;
        const vd = c.v_dim;
        // m = v, n = R, k = lw; batch1_ = wv^T ('t', lda lw, stride v lw), batch2_ = ol's view ('n', ldb H lw,
        // stride lw), ldc v
        try blas.bgemmBf16(f.s.handle, .t, .n, vd, R, lw, L.wv, lw, vd * lw, b.ol, H * lw, lw, b.qlat, vd, R * vd, H);
        const o: kern.Ops = .{ .k = f.k, .s = f.s };
        try o.bhxToHbx(b.qlat, b.o, H, R, vd);
    }
};

/// What a --prompt-prof timer counts. Main stream: attention_core's projections (q_b, o_proj: the prompt GEMM) and
/// its core (absorb / expand through cuBLAS, _qrope, the attention kernels), the routed experts (prompt_experts), the
/// shared expert (+ part += sy), the dense MLP (layers 0-2), the MTP layer's rows (all of it), the head. Comm stream
/// (the sequence-parallel chains): the bf16 reduce-scatter + residual add, the own-row fronts (norms, q_a / kv_a,
/// the indexer's wk / weights_proj / wq_b + rope, the router), the indexer's top-k, the all-gathers, the latent /
/// index key writes.
pub const Phase = enum { attn_proj, attn_core, routed, shared, dense, mtp, head, rs, front, select, ag, kv_write };
const n_phases = @typeInfo(Phase).@"enum".field_names.len;

/// --prompt-prof (default off): CUDA timing events around the prompt path's phases on the main and comm streams,
/// summed per chunk (`report`, after the caller synchronized) and per prompt (`total`). The streams overlap, so the
/// main and comm sums each stay under the chunk's wall time; "main idle" = wall - main busy (host gaps and waits for
/// the comm stream's chains). Recording events changes no kernel and no stream order: the bits are the same.
pub const Prof = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    main: cuda.abi.Stream,
    /// timing on (the caller turns it on after the prewarm's prompt)
    on: bool = false,
    /// inside a region timed as a whole (the MTP layer's rows): nested marks are not taken
    suspended: bool = false,
    pool: std.ArrayList(cuda.Event) = .empty,
    used: usize = 0,
    marks: std.ArrayList(Mark) = .empty,
    sum_main: [n_phases]f64 = @splat(0),
    sum_comm: [n_phases]f64 = @splat(0),
    sum_wall: f64 = 0,
    chunks: usize = 0,

    const Mark = struct { a: usize, b: usize, ph: Phase, comm: bool };

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, main: cuda.Stream) Prof {
        return .{ .gpa = gpa, .d = d, .main = main.handle };
    }

    pub fn deinit(p: *Prof) void {
        for (p.pool.items) |*e| e.deinit();
        p.pool.deinit(p.gpa);
        p.marks.deinit(p.gpa);
    }

    fn next(p: *Prof) !usize {
        if (p.used == p.pool.items.len) try p.pool.append(p.gpa, try cuda.Event.init(p.d, true));
        p.used += 1;
        return p.used - 1;
    }

    /// A timing mark on `s` (null when off or suspended).
    pub fn begin(p: *Prof, s: cuda.Stream) !?usize {
        if (!p.on or p.suspended) return null;
        const i = try p.next();
        try p.pool.items[i].record(s);
        return i;
    }

    /// The end of the region `i` began on `s`, counted under `ph`.
    pub fn end(p: *Prof, s: cuda.Stream, i: usize, ph: Phase) !void {
        if (!p.on) return;
        const j = try p.next();
        try p.pool.items[j].record(s);
        try p.marks.append(p.gpa, .{ .a = i, .b = j, .ph = ph, .comm = s.handle != p.main });
    }

    fn line(buf: []u8, label: []const u8, ms: *const [n_phases]f64) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        var busy: f64 = 0;
        for (ms) |v| busy += v;
        w.print("{s} {d:.1} ms:", .{ label, busy }) catch return buf[0..0];
        for (ms, 0..) |v, i| {
            if (v == 0) continue;
            w.print(" {t} {d:.1}", .{ @as(Phase, @enumFromInt(i)), v }) catch break;
        }
        return w.buffered();
    }

    /// One chunk's sums (every event recorded so far has completed: the caller synchronized), printed as
    /// `PROMPT-PROF rank R chunk I [a, e) rows R wall W ms | main ...: phase ms ... | comm ...: ... | main idle ...`.
    pub fn report(p: *Prof, rank: usize, chunk: usize, a: usize, e: usize, wall_s: f64) !void {
        if (!p.on) return;
        var main_ms: [n_phases]f64 = @splat(0);
        var comm_ms: [n_phases]f64 = @splat(0);
        for (p.marks.items) |m| {
            const ms: f64 = try cuda.Event.elapsedMs(p.pool.items[m.a], p.pool.items[m.b]);
            if (m.comm) comm_ms[@intFromEnum(m.ph)] += ms else main_ms[@intFromEnum(m.ph)] += ms;
        }
        var busy: f64 = 0;
        for (main_ms, 0..) |v, i| {
            busy += v;
            p.sum_main[i] += v;
            p.sum_comm[i] += comm_ms[i];
        }
        const wall = wall_s * 1e3;
        p.sum_wall += wall;
        p.chunks += 1;
        var b1: [512]u8 = undefined;
        var b2: [512]u8 = undefined;
        std.debug.print("PROMPT-PROF rank {d} chunk {d} [{d}, {d}) rows {d} wall {d:.1} ms | {s} | {s} | main idle {d:.1} ms\n", .{ rank, chunk, a, e, e - a, wall, line(&b1, "main", &main_ms), line(&b2, "comm", &comm_ms), wall - busy });
        p.used = 0;
        p.marks.clearRetainingCapacity();
    }

    /// The prompt's sums over its chunks (`PROMPT-PROF rank R total ...`), then zeroed for the next prompt.
    pub fn total(p: *Prof, rank: usize, prompt_len: usize) void {
        if (!p.on or p.chunks == 0) return;
        var busy: f64 = 0;
        for (p.sum_main) |v| busy += v;
        var b1: [512]u8 = undefined;
        var b2: [512]u8 = undefined;
        std.debug.print("PROMPT-PROF rank {d} total prompt {d} chunks {d} wall {d:.1} ms | {s} | {s} | main idle {d:.1} ms\n", .{ rank, prompt_len, p.chunks, p.sum_wall, line(&b1, "main", &p.sum_main), line(&b2, "comm", &p.sum_comm), p.sum_wall - busy });
        p.sum_main = @splat(0);
        p.sum_comm = @splat(0);
        p.sum_wall = 0;
        p.chunks = 0;
    }
};

/// fused.sp_fits: the padded halves fit the buffers (and the world splits rows at all). Under DCP (Forward.dcp > 1)
/// fused.sp_fits is False whatever the rows: the caller checks Forward.dcp first (Sp.chunk refuses it).
pub fn spFits(world: usize, rows0: usize, rows1: usize, R: usize) bool {
    if (world < 2) return false;
    const hR = R / 2;
    const p0 = (hR + world - 1) / world * world;
    const p1 = (R - hR + world - 1) / world * world;
    return p0 <= rows0 and p1 <= rows1;
}

/// The smallest chunk the two-micro-batch path takes (Runner.prefill_chunk: R >= 2 * MAX_ROWS + 2).
pub const min_overlap_rows: usize = 2 * fw.max_rows + 2;

const Half = struct {
    b: *const fw.Buffers,
    R: usize,
    pos: u64, // device int32: the half's first position (st.pos / pos1)
    spos: u64, // device int32: its first own row's position (b.spos)
    n: usize, // own rows a rank (padded: Rp = world n)
    Rp: usize,
    o: usize, // this rank's first own row
    nv: usize, // own rows that are real
    i: usize,
};

const Step = enum { start, after_attn, after_ffn };

/// fused.compute_prompt_sp over the target's layers for one chunk (the caller sets st.pos, pos1 and b0.ids).
pub const Sp = struct {
    main: *const fw.Forward,
    side: fw.Forward, // main's settings on the comm stream (its own prompt-GEMM workspace, no cache)
    comm: *const cuda.nccl.Communicator,
    ready: [2]cuda.Event,
    done: [2]cuda.Event,
    pos1: cuda.DeviceBuffer, // int32 [1]
    spos: cuda.DeviceBuffer, // int32 a half, 256 bytes apart (16-aligned pointers: Triton specializes on it)

    pub fn init(d: *const cuda.Driver, main: *const fw.Forward, side_stream: cuda.Stream, side_wide: *const Wide) !Sp {
        var sp: Sp = .{ .main = main, .side = main.*, .comm = main.comm orelse return error.NoCommunicator, .ready = undefined, .done = undefined, .pos1 = undefined, .spos = undefined };
        sp.side.s = side_stream;
        sp.side.wide = side_wide;
        sp.side.fast = null; // a prompt half's exchanges are NCCL's (never decode-sized)
        sp.side.side = null; // no decode window runs on the comm stream
        sp.side.l2pf = null;
        for (0..2) |i| {
            sp.ready[i] = try cuda.Event.init(d, false);
            sp.done[i] = try cuda.Event.init(d, false);
        }
        sp.pos1 = try cuda.DeviceBuffer.alloc(d, 4);
        sp.spos = try cuda.DeviceBuffer.alloc(d, 512); // one 256-byte slot a half: each pointer 16-aligned like b.spos
        return sp;
    }

    pub fn deinit(sp: *Sp) void {
        for (0..2) |i| {
            sp.ready[i].deinit();
            sp.done[i].deinit();
        }
        sp.pos1.free();
        sp.spos.free();
    }

    fn half(sp: *const Sp, b: *const fw.Buffers, R: usize, pos: u64, i: usize) Half {
        const world = sp.main.w.world;
        const n = (R + world - 1) / world;
        const o = sp.main.w.rank * n;
        return .{ .b = b, .R = R, .pos = pos, .spos = sp.spos.ptr + 256 * i, .n = n, .Rp = n * world, .o = o, .nv = if (R > o) @min(n, R - o) else 0, .i = i };
    }

    /// One chunk of R rows starting at host position `a`: b0.ids[:R] holds its ids, st.pos = a. Leaves every row's
    /// final hidden in b0.hidden[:R] (rows [R/2, R) also in b1.x) and the main stream waiting for every chain.
    pub fn chunk(sp: *Sp, st: *const fw.State, b0: *const fw.Buffers, b1: *const fw.Buffers, a: usize, R: usize, T: ?usize) !void {
        const f = sp.main;
        const D = f.w.config.hidden;
        const o = kern.Ops{ .k = f.k, .s = f.s };
        const hR = R / 2;
        if (f.dcp != 1) return error.SpWithDcp; // fused.sp_fits: no sequence-parallel chunk under DCP
        if (!spFits(f.w.world, b0.rows, b1.rows, R)) return error.WindowTooWide;
        try o.copy(b1.ids, b0.ids + hR * 8, (R - hR) * 8);
        try o.fill32(sp.pos1.ptr, @intCast(a + hR), 1);
        var hs = [2]Half{ sp.half(b0, hR, st.pos, 0), sp.half(b1, R - hR, sp.pos1.ptr, 1) };
        try o.fill32(hs[0].spos, @intCast(a + hs[0].o), 1);
        try o.fill32(hs[1].spos, @intCast(a + hR + hs[1].o), 1);
        for (&hs) |*h| try sp.chain(h, st, .start, 0, T);
        const n_layers = f.w.layers.len;
        for (0..n_layers) |li| {
            const L = &f.w.layers[li];
            for (&hs) |*h| {
                try f.s.wait(sp.done[h.i]);
                try f.attentionCore(h.b, L, st.kc[li], h.pos, h.R, false);
                try sp.chain(h, st, .after_attn, li, T);
            }
            for (&hs) |*h| {
                try f.s.wait(sp.done[h.i]);
                if (L.experts != null) {
                    try f.expertsPart(h.b, L, h.R);
                } else {
                    const p0 = try f.pbegin();
                    try f.mlp(h.b, L, h.R, h.b.part);
                    try f.pend(p0, .dense);
                }
                try sp.chain(h, st, .after_ffn, li, T);
            }
        }
        for (&hs) |*h| try f.s.wait(sp.done[h.i]);
        try o.copy(b0.hidden, b0.x, hR * D * 2);
        try o.copy(b0.hidden + hR * D * 2, b1.x, (R - hR) * D * 2);
        if (b0.taps != 0 and b1.taps != 0) { // b0.taps[hR:R].copy_(b1.taps[:R - hR])
            const row = b0.tap_n * D * 2;
            if (b1.tap_n != b0.tap_n) return error.Invalid;
            try o.copy(b0.taps + hR * row, b1.taps, (R - hR) * row);
        }
    }

    /// compute_prompt_sp's tap(h, i): when a drafter reads layer li, every row of the half (gathered from the ranks'
    /// own rows, fused._gather_x) into its column block of h.b.taps (comm stream).
    fn tap(sp: *Sp, h: *const Half, li: usize) !void {
        const f = sp.main;
        if (f.taps.n == 0 or h.b.taps == 0 or li >= f.taps.slot.len) return;
        const slot = f.taps.slot[li];
        if (slot < 0) return;
        const D = f.w.config.hidden;
        const b = h.b;
        const p0 = try sp.side.pbegin();
        try sp.comm.allGather(b.xo, b.gath, h.n * D, .bf16, sp.side.s.handle);
        const sl: usize = @intCast(slot);
        try sp.ops().copyRows(b.taps + sl * D * 2, b.tap_n * D * 2, b.gath, D * 2, D * 2, h.R);
        try sp.side.pend(p0, .ag);
    }

    /// fused._side: `step` on the comm stream after the main stream's work so far; done[h] marks its end.
    fn chain(sp: *Sp, h: *const Half, st: *const fw.State, step: Step, li: usize, T: ?usize) !void {
        try sp.ready[h.i].record(sp.main.s);
        try sp.side.s.wait(sp.ready[h.i]);
        const w = sp.main.w;
        switch (step) {
            .start => try sp.start(h, st, T),
            .after_attn => {
                try sp.rsResidual(h);
                try sp.frontFfn(h, &w.layers[li]);
            },
            .after_ffn => {
                try sp.rsResidual(h);
                try sp.tap(h, li);
                if (li + 1 < w.layers.len) {
                    try sp.frontAttn(h, st, li + 1, T);
                } else {
                    try sp.gatherX(h, h.b.x);
                }
            },
        }
        try sp.done[h.i].record(sp.side.s);
    }

    fn ops(sp: *const Sp) kern.Ops {
        return .{ .k = sp.side.k, .s = sp.side.s };
    }

    fn tr(sp: *const Sp) tri.Tri {
        return .{ .set = &sp.side.k.triton, .s = sp.side.s };
    }

    /// start: own rows' embeddings (padding rows zero), then layer 0's front.
    fn start(sp: *Sp, h: *const Half, st: *const fw.State, T: ?usize) !void {
        const D = sp.main.w.config.hidden;
        const b = h.b;
        if (h.nv > 0) try sp.ops().embedRows(sp.main.w.embed, b.ids + h.o * 8, D, b.xo, h.nv, false);
        if (h.n > h.nv) try sp.ops().fill32(b.xo + h.nv * D * 2, 0, (h.n - h.nv) * D / 2);
        try sp.frontAttn(h, st, 0, T);
    }

    /// fused._ag_rows: every rank's own rows of each buffer, in place (rank r's at rows [r n, (r + 1) n)).
    fn agRows(sp: *Sp, h: *const Half, bufs: []const u64, row_elems: []const usize, dts: []const cuda.nccl.DataType, eb: []const usize) !void {
        var counts: [6]usize = undefined;
        for (row_elems, 0..) |re, i| counts[i] = h.n * re;
        try sp.comm.allGatherInPlace(bufs, counts[0..bufs.len], dts, eb, sp.side.s.handle);
    }

    /// fused._front_attn: own rows' input norm, q_a / kv_a and the indexer's projections; gathered; every row's
    /// latent and index key into the caches; the selection of own rows (SP_SELECT), gathered.
    fn frontAttn(sp: *Sp, h: *const Half, st: *const fw.State, li: usize, T: ?usize) !void {
        const f = &sp.side;
        const w = f.w;
        const c = w.config;
        const D = c.hidden;
        const L = &w.layers[li];
        const b = h.b;
        const t_ = sp.tr();
        const o = h.o;
        const n = h.n;
        const ql = c.q_lora;
        const nx = b.normed + o * D * 2;
        const p0 = try f.pbegin();
        try t_.rmsnorm(b.xo, D, L.input_norm, nx, D, n, D, c.eps);
        try f.lin(b, &L.q_a, nx, n, b.qa + o * ql * 2, .bf16);
        try f.lin(b, &L.kv_a, nx, n, b.kva + o * b.kva_n * 2, .bf16);
        try t_.rmsnorm(b.qa + o * ql * 2, ql, L.q_a_norm, b.qn + o * ql * 2, ql, n, ql, c.eps);
        var bufs: [3]u64 = .{ b.qn, b.kva, b.ik };
        var re: [3]usize = .{ ql, b.kva_n, c.index_dim };
        var dts: [3]cuda.nccl.DataType = .{ .bf16, .bf16, .f32 };
        var eb: [3]usize = .{ 2, 2, 4 };
        var nb: usize = 2;
        if (L.indexer) |*ix| {
            try t_.router(nx, D, ix.wk, b.ik + o * c.index_dim * 4, b.rpart, n, D, c.index_dim);
            nb = 3;
            if (T != null) {
                try t_.router(nx, D, ix.weights_proj, b.iw + o * c.index_heads * 4, b.rpart, n, D, c.index_heads);
                const iq = b.iq + o * c.index_heads * c.index_dim * 2;
                try f.lin(b, &ix.wq_b, b.qn + o * ql * 2, n, iq, .bf16);
                try t_.iqRope(iq, f.inv, h.spos, n, c.index_heads, c.index_dim, c.rope);
            }
        }
        try f.pend(p0, .front);
        const p1 = try f.pbegin();
        try sp.agRows(h, bufs[0..nb], re[0..nb], dts[0..nb], eb[0..nb]);
        try f.pend(p1, .ag);
        const p2 = try f.pbegin();
        const m: tri.Mla = .{ .heads = w.heads, .nope = c.nope, .rope = c.rope, .lw = c.kv_lora, .vd = c.v_dim, .topk = c.index_topk };
        try t_.kvWrite(b.kva, b.kva_n, L.kv_a_norm, st.kc[li], h.pos, f.inv, c.eps, h.R, m);
        const ix = if (L.indexer) |*x| x else return f.pend(p2, .kv_write);
        try t_.ikWrite(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, st.ic[li], h.pos, h.R, c.index_dim, c.rope);
        try f.pend(p2, .kv_write);
        const tt = T orelse return;
        const p3 = try f.pbegin();
        try f.select(b, st.ic[li], h.pos, o, n, tt);
        try f.pend(p3, .select);
        bufs[0] = b.tok;
        re[0] = c.index_topk;
        dts[0] = .i32;
        eb[0] = 4;
        const p4 = try f.pbegin();
        try sp.agRows(h, bufs[0..1], re[0..1], dts[0..1], eb[0..1]);
        try f.pend(p4, .ag);
    }

    /// fused._front_ffn: own rows' post-attention norm and routing; the normed rows (and picks) gathered.
    fn frontFfn(sp: *Sp, h: *const Half, L: *const W.Layer) !void {
        const f = &sp.side;
        const c = f.w.config;
        const D = c.hidden;
        const b = h.b;
        const p0 = try f.pbegin();
        try sp.tr().rmsnorm(b.xo, D, L.post_attn_norm, b.normed + h.o * D * 2, D, h.n, D, c.eps);
        if (L.experts != null) try f.route(b, L, h.o, h.n);
        try f.pend(p0, .front);
        const p1 = try f.pbegin();
        if (L.experts != null) {
            try sp.agRows(h, &.{ b.normed, b.pick, b.wts }, &.{ D, c.top_k, c.top_k }, &.{ .bf16, .i32, .f32 }, &.{ 2, 4, 4 });
        } else {
            try sp.agRows(h, &.{b.normed}, &.{D}, &.{.bf16}, &.{2});
        }
        try f.pend(p1, .ag);
    }

    /// fused._rs_residual (PREFILL_REDUCE ring): b.part[:Rp] (padding rows zeroed) to bf16, NCCL's ring
    /// reduce-scatter of the rows -> own rows' sum (bf16), added to the own residual rows b.xo.
    fn rsResidual(sp: *Sp, h: *const Half) !void {
        const D = sp.main.w.config.hidden;
        const b = h.b;
        const o = sp.ops();
        const p0 = try sp.side.pbegin();
        if (h.Rp > h.R) try o.fill32(b.part + h.R * D * 4, 0, (h.Rp - h.R) * D);
        try o.f32ToBf16(b.part, b.bpart, h.Rp * D);
        try sp.comm.reduceScatter(b.bpart, b.bred, h.n * D, .bf16, .sum, sp.side.s.handle);
        try sp.tr().residualAddOf(b.xo, b.bred, "*bf16", h.n, D, 1);
        try sp.side.pend(p0, .rs);
    }

    /// fused._gather_x: every row's residual stream gathered from the ranks' own rows -> out [R, D].
    fn gatherX(sp: *Sp, h: *const Half, out: u64) !void {
        const D = sp.main.w.config.hidden;
        const b = h.b;
        const p0 = try sp.side.pbegin();
        try sp.comm.allGather(b.xo, b.gath, h.n * D, .bf16, sp.side.s.handle);
        try sp.ops().copy(out, b.gath, h.R * D * 2);
        try sp.side.pend(p0, .ag);
    }
};

/// Phase 3a speed (--mtp-dense 1): the MTP layer's prompt rows through cuBLAS. Its 8-bit (K2 16) routed experts
/// are not prompt_experts' widths, and the decode expert kernel re-decodes each expert's trellis for every 16 member
/// rows (~0.5 s an 8K chunk). The MTP layer only drafts - the target verifies every token, and the Python engine's
/// own path there (cuda-exl3's split-K grouped GEMM) is not reproducible run to run - so here each expert is kept a
/// second time as dense bf16 matrices with the EXL3 rotations folded in (built once at load from the same trellis:
/// unpack, the Hadamards, suh / svh), and a prompt chunk's MTP rows run:
///   picks to the host (one sync) -> pairs grouped by expert (counting sort, pair order within an expert) ->
///   gather of the rows (b.gath, idle while the MTP layer runs) -> a cuBLAS GEMM an expert for gate | up ->
///   SwiGLU in place -> a GEMM an expert for down (into the expert's own gathered rows) -> combine in slot order
///   with the shared expert's b.sy into b.part.
/// eh_proj of the prompt rows also goes through one cuBLAS GEMM (Forward.mtpCompute), which drops the router's
/// [ROUTER_KS, rows, D] fp32 partials from the prompt buffers.
/// Memory a rank: E * 3 * D * I * 2 bytes of weights (GLM-5.3 at world 4: 256 * 3 * 6144 * 512 * 2 = 4.83 GB) +
/// h (rows * K * 2 I * 2 = 134 MB at 8192 rows) - the eh_proj partials and fp32 rows it saves (1.79 GB at 8192 rows).
pub const MtpDense = struct {
    gpa: std.mem.Allocator,
    blas: *const cuda.cublas.Blas,
    wgu: cuda.DeviceBuffer, // bf16 [E, D, 2 I]: x @ wgu[e] = (gate | up) of expert e
    wd: cuda.DeviceBuffer, // bf16 [E, I, D]: a @ wd[e] = down of expert e
    h: cuda.DeviceBuffer, // bf16 [rows K, 2 I]: gate | up of the grouped pairs (SwiGLU in place into [:, :I])
    perm_dev: cuda.DeviceBuffer, // int32 [rows K]: segment position -> pair (row K + slot), segments one after another
    inv_dev: cuda.DeviceBuffer, // int32 [rows K]: pair -> position in its segment (-1: unrouted)
    pick: []i32, // host copies
    perm: []i32,
    inv: []i32,
    cnt: []usize, // a segment's pairs an expert
    off: []usize, // their first position
    fill: []usize,
    E: usize,
    D: usize,
    I: usize,
    K: usize,
    rows: usize,
    /// device bytes held (weights + scratch)
    bytes: usize,
    /// prompt chunks run, segments, cuBLAS GEMMs launched (stats)
    chunks: usize = 0,
    segments: usize = 0,
    gemms: usize = 0,

    /// Builds the dense experts of `w`'s MTP layer on `s` (synchronizes) for prompt windows of up to `rows` rows.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, k: *const kern.Kernels, s: cuda.Stream, w: *const W.Weights, blas: *const cuda.cublas.Blas, rows: usize) !MtpDense {
        const c = w.config;
        const m = w.mtp orelse return error.NoMtpLayer;
        const ex = if (m.layer.experts) |*e| e else return error.NoMtpExperts;
        const E = ex.count;
        const D = ex.dims;
        const I = ex.width;
        const K = c.top_k;
        if (K == 0 or K > 32 or D % 128 != 0 or I % 128 != 0) return error.Invalid;
        const o: kern.Ops = .{ .k = k, .s = s };
        // the trellis pointer and width tables (device int64 / int32 [E] each) to the host
        const ptrs = try gpa.alloc(u64, 3 * E);
        defer gpa.free(ptrs);
        const k2s = try gpa.alloc(i32, 3 * E);
        defer gpa.free(k2s);
        for ([_]u64{ ex.gate_ptr, ex.up_ptr, ex.down_ptr }, 0..) |tab, j| try o.download(std.mem.sliceAsBytes(ptrs[j * E ..][0..E]), tab);
        for ([_]u64{ ex.gate_k2, ex.up_k2, ex.down_k2 }, 0..) |tab, j| try o.download(std.mem.sliceAsBytes(k2s[j * E ..][0..E]), tab);
        try s.synchronize();
        var wgu = try cuda.DeviceBuffer.alloc(d, E * D * 2 * I * 2);
        errdefer wgu.free();
        var wd = try cuda.DeviceBuffer.alloc(d, E * I * D * 2);
        errdefer wd.free();
        {
            var wq = try cuda.DeviceBuffer.alloc(d, D * I * 2); // fp16 W_q of one matrix (every one is D x I or I x D)
            defer wq.free();
            var tt = try cuda.DeviceBuffer.alloc(d, D * I * 4);
            defer tt.free();
            for (0..E) |e| {
                const gu = wgu.ptr + e * D * 2 * I * 2;
                try o.unpackRaw(ptrs[e], @intCast(k2s[e]), D, I, wq.ptr);
                try o.mtpDense(wq.ptr, tt.ptr, ex.suh_g + e * D * 2, ex.svh_g + e * I * 2, gu, D, I, 2 * I);
                try o.unpackRaw(ptrs[E + e], @intCast(k2s[E + e]), D, I, wq.ptr);
                try o.mtpDense(wq.ptr, tt.ptr, ex.suh_u + e * D * 2, ex.svh_u + e * I * 2, gu + I * 2, D, I, 2 * I);
                try o.unpackRaw(ptrs[2 * E + e], @intCast(k2s[2 * E + e]), I, D, wq.ptr);
                try o.mtpDense(wq.ptr, tt.ptr, ex.suh_d + e * I * 2, ex.svh_d + e * D * 2, wd.ptr + e * I * D * 2, I, D, D);
            }
            try s.synchronize(); // wq / tt are freed at the end of this block
        }
        const P = rows * K;
        var h = try cuda.DeviceBuffer.alloc(d, P * 2 * I * 2);
        errdefer h.free();
        var perm_dev = try cuda.DeviceBuffer.alloc(d, P * 4);
        errdefer perm_dev.free();
        var inv_dev = try cuda.DeviceBuffer.alloc(d, P * 4);
        errdefer inv_dev.free();
        const pick = try gpa.alloc(i32, P);
        errdefer gpa.free(pick);
        const perm = try gpa.alloc(i32, P);
        errdefer gpa.free(perm);
        const inv = try gpa.alloc(i32, P);
        errdefer gpa.free(inv);
        const cnt = try gpa.alloc(usize, E);
        errdefer gpa.free(cnt);
        const off = try gpa.alloc(usize, E);
        errdefer gpa.free(off);
        const fill = try gpa.alloc(usize, E);
        errdefer gpa.free(fill);
        return .{
            .gpa = gpa,
            .blas = blas,
            .wgu = wgu,
            .wd = wd,
            .h = h,
            .perm_dev = perm_dev,
            .inv_dev = inv_dev,
            .pick = pick,
            .perm = perm,
            .inv = inv,
            .cnt = cnt,
            .off = off,
            .fill = fill,
            .E = E,
            .D = D,
            .I = I,
            .K = K,
            .rows = rows,
            .bytes = wgu.len + wd.len + h.len + perm_dev.len + inv_dev.len,
        };
    }

    pub fn deinit(md: *MtpDense) void {
        md.gpa.free(md.fill);
        md.gpa.free(md.off);
        md.gpa.free(md.cnt);
        md.gpa.free(md.inv);
        md.gpa.free(md.perm);
        md.gpa.free(md.pick);
        md.inv_dev.free();
        md.perm_dev.free();
        md.h.free();
        md.wd.free();
        md.wgu.free();
        md.* = undefined;
    }

    /// eh_proj of prompt rows: x [n, D] bf16 = mcat [n, 2 D] @ eh_proj^T (eh_proj bf16 [D, 2 D]), fp32 compute.
    pub fn ehProj(md: *MtpDense, f: *const fw.Forward, mcat: u64, eh_proj: u64, x: u64, n: usize) !void {
        const D = f.w.config.hidden;
        // column-major: x^T [D, n] = (eh_proj as [2D, D] col-major)^T @ mcat^T [2D, n]
        try md.blas.bgemmBf16(f.s.handle, .t, .n, D, n, 2 * D, eh_proj, 2 * D, 0, mcat, 2 * D, 0, x, D, 0, 1);
        md.gemms += 1;
    }

    /// Rows [r0, r0 + n) of md.pick: cnt / off (segment positions grouped by expert, ascending; pairs in order
    /// within an expert), perm[base ..] (position -> pair) and inv[r0 K ..] (pair -> position, -1 unrouted).
    /// Returns the routed pairs.
    fn plan(md: *MtpDense, r0: usize, n: usize, base: usize) usize {
        const K = md.K;
        const E: i32 = @intCast(md.E);
        @memset(md.cnt, 0);
        for (r0 * K..(r0 + n) * K) |p| {
            const e = md.pick[p];
            if (e >= 0 and e < E) md.cnt[@intCast(e)] += 1;
        }
        var acc: usize = 0;
        for (0..md.E) |e| {
            md.off[e] = acc;
            md.fill[e] = acc;
            acc += md.cnt[e];
        }
        for (r0 * K..(r0 + n) * K) |p| {
            const e = md.pick[p];
            if (e >= 0 and e < E) {
                const q = md.fill[@intCast(e)];
                md.fill[@intCast(e)] = q + 1;
                md.perm[base + q] = @intCast(p);
                md.inv[p] = @intCast(q);
            } else {
                md.inv[p] = -1;
            }
        }
        return acc;
    }

    /// The MTP layer's routed experts of prompt rows b.normed[:R] (picks b.pick, weights b.wts), plus b.sy (the
    /// shared expert, already queued), into b.part[:R] (fp32). One host sync (the picks); rows in segments that fit
    /// b.gath (one segment at 8192 rows, world 4, top-k 8) and md.h.
    pub fn routed(md: *MtpDense, f: *const fw.Forward, b: *const fw.Buffers, R: usize) !void {
        const o: kern.Ops = .{ .k = f.k, .s = f.s };
        const K = md.K;
        const D = md.D;
        const I = md.I;
        if (R > md.rows or b.gath == 0) return error.NoMtpScratch;
        // xs: the grouped rows, then each expert's down output over its own rows (bf16 [pairs, D]) - b.gath (fp32
        // [world, rows, D]) is idle while the MTP layer runs: prompt chunks reduce over the bf16 ring, and the next
        // chunk's comm stream waits for this chunk's main stream
        const room = f.w.world * b.rows * D * 4;
        const seg = @min(room / (K * D * 2), md.h.len / (K * 2 * I * 2));
        if (seg == 0) return error.ScratchTooSmall;
        const xs = b.gath;
        try o.download(std.mem.sliceAsBytes(md.pick[0 .. R * K]), b.pick);
        try f.s.synchronize(); // the picks on the host: a prompt chunk's MTP layer only
        md.chunks += 1;
        var r0: usize = 0;
        var base: usize = 0;
        while (r0 < R) : (r0 += seg) {
            const n = @min(seg, R - r0);
            const np = md.plan(r0, n, base);
            md.segments += 1;
            try o.upload(md.inv_dev.ptr + r0 * K * 4, std.mem.sliceAsBytes(md.inv[r0 * K ..][0 .. n * K]));
            if (np > 0) {
                const pd = md.perm_dev.ptr + base * 4;
                try o.upload(pd, std.mem.sliceAsBytes(md.perm[base..][0..np]));
                try o.mtpGather(b.normed, D, pd, K, xs, np, D);
                // gate | up: h[pos] = xs[pos] @ wgu[e] (column-major h^T [2I, cnt] = wgu[e]^T-view [2I, D] @ xs^T)
                for (0..md.E) |e| {
                    const cn = md.cnt[e];
                    if (cn == 0) continue;
                    const q = md.off[e];
                    try md.blas.bgemmBf16(f.s.handle, .n, .n, 2 * I, cn, D, md.wgu.ptr + e * D * 2 * I * 2, 2 * I, 0, xs + q * D * 2, D, 0, md.h.ptr + q * 2 * I * 2, 2 * I, 0, 1);
                    md.gemms += 1;
                }
                try o.mtpSwiglu(md.h.ptr, np, I);
                // down: xs[pos] = act[pos] @ wd[e] (over the expert's own gathered rows: spent by its gate | up)
                for (0..md.E) |e| {
                    const cn = md.cnt[e];
                    if (cn == 0) continue;
                    const q = md.off[e];
                    try md.blas.bgemmBf16(f.s.handle, .n, .n, D, cn, I, md.wd.ptr + e * I * D * 2, D, 0, md.h.ptr + q * 2 * I * 2, 2 * I, 0, xs + q * D * 2, D, 0, 1);
                    md.gemms += 1;
                }
            }
            try o.mtpCombine(xs, md.inv_dev.ptr + r0 * K * 4, b.wts + r0 * K * 4, b.sy + r0 * D * 4, b.part + r0 * D * 4, n, K, D);
            base += np;
        }
    }
};

test "sequence-parallel fits as fused.sp_fits" {
    try std.testing.expect(spFits(4, 8192, 4096, 8177)); // a 32K prompt's chunk: halves 4088 / 4089 -> 4088, 4092
    try std.testing.expect(!spFits(4, 8192, 4096, 8192 + 4)); // the second half pads to 4100 > 4096
    try std.testing.expect(spFits(4, 4096, 2048, 4096));
    try std.testing.expect(!spFits(1, 8192, 4096, 1000));
}
