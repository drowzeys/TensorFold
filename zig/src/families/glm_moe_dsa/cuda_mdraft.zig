//! Phase 4c: GLM-5.3's speculative drafters for concurrent streams (--parallel N, cuda_multi.zig).
//!
//! DFlash2 is dflash.MultiDrafter ported: one drafter's weights (the one-stream cuda_draft.Drafter's), a ring a stream
//! slot in a pool per layer ([kv heads, N x RING, head_dim]: slot i's ring at rows i x RING), the host's committed
//! length of every slot (`end`), every live DFlash2 stream's block [pending, mask x (block - 1)] in one pass
//! (_dconv_seg, _dattn_seg reading each block's slot and committed length from META = pending | slot | end) and every
//! stream's kept taps stacked into context updates of up to t_max rows (each row's position and slot from int64
//! tables: glm53_draft_rotary_rows / glm53_draft_scatter_rows). Each stream's candidates are merged and walked by the
//! one-stream drafter's host stage (the selector chain at the stream's committed length + 1, its keyed sampling, the
//! round's confidence floor), so a stream drafts what Python's MultiDrafter drafts for it.
//!
//! DSpark has no Python counterpart (engine.py refuses TF_GLM53_DSPARK with --parallel). It follows our vLLM DSpark
//! concurrency patches (Patch 1: the drafter's own KV keyed by the STABLE stream slot, never by the batch row; Patch 2:
//! one ragged batched draft pass over every live drafting stream, each stream's Markov / confidence cut and host stage
//! on its own): the pool is slot-major (slot i's ring [kv heads, RING, head_dim], the one-stream ring's layout), a
//! block pass runs the row-wise steps (norms, 4-bit matmuls, the row-parallel sums, the MLP, the head) over every
//! block at once and each block's attention part alone (_prep_kernel on its rows, its keys into its slot's ring,
//! _block_attn_kernel over that ring at its committed length) - the one-stream pass's kernels and shapes, so a block
//! computes what it computes alone (the passes are row-invariant). Its blocks run in slot order and its graphs are
//! keyed by the slot set (the rings' addresses are baked into a graph). Context updates as DFlash2's (slot-major
//! strides). Every stream walks MarkovChain.walk at its own committed length with its own sampling and policy.
//!
//! Drafts only propose: the target verifies every row and every emitted token is its own pick, so a stream's reply
//! under concurrency is its reply alone whatever the drafts; every rank holds the same gathered candidates and walks
//! the same chains, so every rank verifies the same window.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const kern = @import("cuda_kernels.zig");
const fw = @import("cuda_forward.zig");
const smp = @import("sampling.zig");
const draft = @import("cuda_draft.zig");
const host = @import("draft_host.zig");
const rn = @import("cuda_runner.zig");

pub const ring_rows: usize = draft.ring_rows;
/// The most stream slots a concurrent drafter serves (a block pass's blocks).
pub const max_streams: usize = draft.max_segs;
/// The widest pass (qmm_group's 4-bit tiles take up to 64 rows): N streams' blocks side by side, a context update.
pub const max_pass_rows: usize = draft.tap_rows;

/// Rows a stream's verify window takes with this drafter at N streams (multi.frows): its most drafts and the pending
/// token, at most a decode window's share (4 streams: 8 rows - 7 drafts).
pub fn frows(sp: draft.Spec, N: usize) usize {
    return @min(sp.maxDepth() + 1, fw.decode_rows / @max(N, 1));
}

/// Whether N streams can draft with this drafter: every block side by side in one pass, two window rows at least.
pub fn fits(sp: draft.Spec, N: usize) bool {
    return N >= 1 and N <= max_streams and N * sp.block <= max_pass_rows and frows(sp, N) >= 2;
}

/// The rows a context update takes at most (MultiDrafter.t_max: a block multiple, within one pass).
pub fn tapMax(block: usize) usize {
    return max_pass_rows / block * block;
}

/// The captured row bucket of a context update of `rows` stacked rows (the next block multiple).
pub fn tapBucket(rows: usize, block: usize) usize {
    return (rows + block - 1) / block * block;
}

/// One stream's ask of a round (multi._gpu's reqs): its slot, its pending token, the drafts it may take, its keyed
/// sampling (the chain's noise and filter) and its confidence floor.
pub const Ask = struct {
    slot: usize,
    pending: u32,
    depth: usize,
    sampling: ?smp.Sampling = null,
    conf: f64,
};

/// A stream's drafts of a round.
pub const Drafted = struct {
    n: usize = 0,
    ids: [fw.decode_rows]u32 = @splat(0),

    pub fn slice(x: *const Drafted) []const u32 {
        return x.ids[0..x.n];
    }
};

/// One stream's committed rows: its slot and `n` rows of a taps buffer from device address `src` on.
pub const Commit = struct { slot: usize, src: u64, n: usize };

/// A piece of a context update: `n` rows of one stream from `src`, at positions p0 ...
pub const Piece = struct { slot: usize, src: u64, n: usize, p0: usize };

/// MultiDrafter.propose's host tables of one block pass: the asks that draft (blocks: DFlash2 in ask order, as
/// Python; DSpark in slot order), META (pending | slot | committed length, pitch N), each block row's position and
/// slot, every block's ids [pending, mask ...], the slots' bits (DSpark's graph key).
pub const BlockPlan = struct {
    S: usize = 0,
    ask: [max_streams]usize = @splat(0),
    depth: [max_streams]usize = @splat(0),
    meta: [3 * max_streams]i64 = @splat(0),
    rpos: [max_pass_rows]i64 = @splat(0),
    rslot: [max_pass_rows]i64 = @splat(0),
    ids: [max_pass_rows]i32 = @splat(0),
    mask: u64 = 0,
};

/// The block pass of `asks` (MultiDrafter.propose's live list: a depth of at least one draft - at most `max_depth` -
/// and a context, end[slot] > 0) and its tables.
pub fn blockPlan(asks: []const Ask, end: []const usize, N: usize, block: usize, max_depth: usize, mask_id: u32, by_slot: bool) !BlockPlan {
    var bp: BlockPlan = .{};
    if (N == 0 or N > max_streams or N * block > max_pass_rows or end.len < N) return error.Invalid;
    for (asks, 0..) |a, i| {
        if (a.slot >= N) return error.Invalid;
        const d = @min(a.depth, max_depth);
        if (d < 1 or end[a.slot] == 0) continue;
        const bit = @as(u64, 1) << @intCast(a.slot);
        if (bp.mask & bit != 0) return error.Invalid; // a slot asks once a round
        bp.ask[bp.S] = i;
        bp.depth[bp.S] = d;
        bp.mask |= bit;
        bp.S += 1;
    }
    if (by_slot) { // insertion sort of the blocks by slot
        var s: usize = 1;
        while (s < bp.S) : (s += 1) {
            var j = s;
            while (j > 0 and asks[bp.ask[j - 1]].slot > asks[bp.ask[j]].slot) : (j -= 1) {
                std.mem.swap(usize, &bp.ask[j - 1], &bp.ask[j]);
                std.mem.swap(usize, &bp.depth[j - 1], &bp.depth[j]);
            }
        }
    }
    for (0..bp.S) |s| {
        const a = asks[bp.ask[s]];
        const e = end[a.slot];
        bp.meta[s] = a.pending;
        bp.meta[N + s] = @intCast(a.slot);
        bp.meta[2 * N + s] = @intCast(e);
        for (0..block) |j| {
            bp.rpos[s * block + j] = @intCast(e + j);
            bp.rslot[s * block + j] = @intCast(a.slot);
            bp.ids[s * block + j] = if (j == 0) @intCast(a.pending) else @intCast(mask_id);
        }
    }
    return bp;
}

/// A context update's tables (MultiDrafter._launch_taps): each stacked row's position and slot, the padding of the
/// captured bucket at slot -1 (nothing written). Returns the bucket's rows.
pub fn tapTables(pieces: []const Piece, block: usize, t_max: usize, pos: []i64, slot: []i64) !usize {
    var off: usize = 0;
    for (pieces) |pc| {
        if (off + pc.n > t_max) return error.Invalid;
        for (0..pc.n) |j| {
            pos[off + j] = @intCast(pc.p0 + j);
            slot[off + j] = @intCast(pc.slot);
        }
        off += pc.n;
    }
    const T = tapBucket(off, block);
    if (T > t_max or T == 0) return error.Invalid;
    for (off..T) |j| {
        pos[j] = 0;
        slot[j] = -1;
    }
    return T;
}

fn tapKey(which: usize, T: usize) u64 {
    return 7 | (@as(u64, which) << 4) | (@as(u64, T) << 8);
}

fn blockKey(which: usize, sel: u64) u64 {
    return 8 | (@as(u64, which) << 4) | (sel << 8);
}

/// The device tables' byte offsets: META [3 N] | block rows' positions [N n] | their slots [N n] | each block's
/// committed length 16 bytes apart (DSpark's _block_attn_kernel POS) | a context update's positions [t_max] | its
/// slots [t_max]. The pinned staging mirrors the block part, then the ids, then `updates` context updates' tables.
const Lay = struct {
    meta: usize,
    rpos: usize,
    rslot: usize,
    spos: usize,
    block_end: usize,
    tpos: usize,
    tslot: usize,
    total: usize,
    pin_ids: usize,
    pin_taps: usize,

    fn make(N: usize, n: usize, t_max: usize) Lay {
        var l: Lay = undefined;
        l.meta = 0;
        l.rpos = std.mem.alignForward(usize, 3 * N * 8, 16);
        l.rslot = std.mem.alignForward(usize, l.rpos + N * n * 8, 16);
        l.spos = std.mem.alignForward(usize, l.rslot + N * n * 8, 16);
        l.block_end = l.spos + N * 16;
        l.tpos = std.mem.alignForward(usize, l.block_end, 16);
        l.tslot = l.tpos + t_max * 8;
        l.total = l.tslot + t_max * 8;
        l.pin_ids = std.mem.alignForward(usize, l.block_end, 256);
        l.pin_taps = std.mem.alignForward(usize, l.pin_ids + N * n * 4, 256);
        return l;
    }

    fn pinBytes(l: Lay, t_max: usize, updates: usize) usize {
        return l.pin_taps + updates * 2 * t_max * 8;
    }
};

/// One drafter for N concurrent streams on one rank (every rank calls it alike: same slots, same order).
pub const MultiDrafter = struct {
    gpa: std.mem.Allocator,
    r: *rn.Runner,
    base: *draft.Drafter, // the weights, tap layout, settings and host stage (the one-stream drafter's)
    v: draft.Drafter, // the passes' view: base's weights, this pool's buffers
    which: usize, // 0 DFlash2, 1 DSpark (the runner's drafter order; graph keys)
    N: usize,
    n: usize, // block
    hr: usize, // head rows a block
    t_max: usize,
    send_words: usize,
    pl: draft.Drafter.PoolLayout,
    pool_rows: usize,
    kc: []u64,
    vc: []u64,
    mem: std.ArrayList(cuda.DeviceBuffer),
    tab: u64,
    lay: Lay,
    pin: cuda.HostBuffer,
    updates: usize, // context updates a commit stages before it synchronizes
    end: [max_streams]usize, // each slot's committed rows (host)
    strides: [max_streams + 1]usize, // a block pass of S blocks: the gather's words a rank
    inflight: bool, // a commit's tables may still be uploading from the pinned staging
    passes: usize,

    pub fn init(gpa: std.mem.Allocator, r: *rn.Runner, base: *draft.Drafter, which: usize, N: usize) !*MultiDrafter {
        const sp = base.sp;
        if (!fits(sp, N)) return error.DrafterTooWide;
        const d = base.d orelse return error.Invalid;
        const md = try gpa.create(MultiDrafter);
        errdefer gpa.destroy(md);
        md.* = undefined;
        md.gpa = gpa;
        md.r = r;
        md.base = base;
        md.which = which;
        md.N = N;
        md.n = sp.block;
        md.hr = sp.headRows();
        md.t_max = tapMax(sp.block);
        md.send_words = std.mem.alignForward(usize, N * md.hr * 2 * sp.top_k, 4);
        md.mem = .empty;
        md.end = @splat(0);
        md.strides = @splat(0);
        md.inflight = false;
        md.passes = 0;
        md.kc = try gpa.alloc(u64, sp.layers);
        errdefer gpa.free(md.kc);
        md.vc = try gpa.alloc(u64, sp.layers);
        errdefer gpa.free(md.vc);
        errdefer md.freeMem();
        const ring_bytes = sp.KV * ring_rows * sp.hd * 2; // one slot's ring of one layer
        for (0..sp.layers) |i| {
            md.kc[i] = try md.zeros(d, N * ring_bytes);
            md.vc[i] = try md.zeros(d, N * ring_bytes);
        }
        if (sp.kind == .dflash) { // dflash.MultiDrafter's pool: [kv heads, N x RING, hd] (_dattn_seg's POOL)
            md.pool_rows = N * ring_rows;
            md.pl = .{ .head_stride = N * ring_rows, .slot_stride = ring_rows };
        } else { // slot-major: slot i's ring [kv heads, RING, hd], the one-stream layout _block_attn_kernel reads
            md.pool_rows = ring_rows;
            md.pl = .{ .head_stride = ring_rows, .slot_stride = sp.KV * ring_rows };
        }
        md.lay = Lay.make(N, md.n, md.t_max);
        md.tab = try md.zeros(d, md.lay.total);
        md.updates = r.set.prompt_rows / md.t_max + 2;
        md.pin = try cuda.HostBuffer.alloc(d, md.lay.pinBytes(md.t_max, md.updates));
        errdefer md.pin.free();
        md.v = try base.initView(@max(md.t_max, N * md.n), N * md.hr, md.send_words, md.kc, md.vc);
        return md;
    }

    pub fn deinit(md: *MultiDrafter) void {
        md.v.freeView();
        md.pin.free();
        md.freeMem();
        md.gpa.free(md.kc);
        md.gpa.free(md.vc);
        md.gpa.destroy(md);
    }

    fn freeMem(md: *MultiDrafter) void {
        for (md.mem.items) |*m| m.free();
        md.mem.deinit(md.gpa);
        md.mem = .empty;
    }

    fn zeros(md: *MultiDrafter, d: *const cuda.Driver, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(d, @max(bytes, 16));
        errdefer m.free();
        try m.fill8(0, null);
        try md.mem.append(md.gpa, m);
        return m.ptr;
    }

    /// Device bytes of the rings, tables and the view's buffers (the weights are the one-stream drafter's).
    pub fn deviceBytes(md: *const MultiDrafter) usize {
        var b: usize = 0;
        for (md.mem.items) |m| b += m.len;
        for (md.v.mem.items) |m| b += m.len;
        return b;
    }

    fn ops(md: *const MultiDrafter) kern.Ops {
        return .{ .k = md.r.f.k, .s = md.r.f.s };
    }

    fn dev(md: *const MultiDrafter) draft.Dev {
        const f = &md.r.f;
        return .{ .t = .{ .set = &f.k.triton, .s = f.s }, .o = .{ .k = f.k, .s = f.s }, .f = f };
    }

    fn pinSlice(md: *const MultiDrafter, comptime T: type, off: usize, n: usize) []T {
        const p: [*]T = @ptrCast(@alignCast(md.pin.bytes.ptr + off));
        return p[0..n];
    }

    /// MultiDrafter.reset: slot's context starts at `at` (0: a new prompt; a resumed one without a kept ring window
    /// at its resume point - putWindow when the kept state has one).
    pub fn reset(md: *MultiDrafter, slot: usize, at: usize) void {
        md.end[slot] = at;
    }

    /// prefixes.slot_ring_window's rows: positions [lo, n) of the context a block at n still sees.
    pub fn windowLo(md: *const MultiDrafter, n: usize) usize {
        return n -| (md.base.sp.window + 1);
    }

    /// Bytes of a slot's ring window at n: [layers][k, v][kv heads][rows][hd] bf16.
    pub fn windowBytes(md: *const MultiDrafter, n: usize) usize {
        const sp = md.base.sp;
        return sp.layers * 2 * sp.KV * (n - md.windowLo(n)) * sp.hd * 2;
    }

    /// prefixes.slot_ring_window / put_slot_ring_window: slot's ring rows of positions [lo, n) to `buf` (save) or
    /// back from it (put), every layer's keys and values, each KV head's rows in at most two runs (the ring wraps).
    fn moveWindow(md: *MultiDrafter, slot: usize, n: usize, buf: u64, save: bool) !void {
        const sp = md.base.sp;
        const lo = md.windowLo(n);
        const rows = n - lo;
        if (rows == 0) return;
        const o = md.ops();
        const rb = sp.hd * 2; // one ring row of one head
        for (0..sp.layers) |i| {
            for ([2]u64{ md.kc[i], md.vc[i] }, 0..) |pool, c| {
                for (0..sp.KV) |h| {
                    const ring0 = pool + (h * md.pl.head_stride + slot * md.pl.slot_stride) * rb;
                    const out0 = buf + (((i * 2 + c) * sp.KV + h) * rows) * rb;
                    var p = lo;
                    while (p < n) {
                        const q = p % ring_rows;
                        const run = @min(n - p, ring_rows - q);
                        const ring_at = ring0 + q * rb;
                        const buf_at = out0 + (p - lo) * rb;
                        if (save) try o.copy(buf_at, ring_at, run * rb) else try o.copy(ring_at, buf_at, run * rb);
                        p += run;
                    }
                }
            }
        }
    }

    pub fn saveWindow(md: *MultiDrafter, slot: usize, n: usize, buf: u64) !void {
        try md.moveWindow(slot, n, buf, true);
    }

    /// The slot's context resumes at n with the kept window's rows (multi._queue's put_slot_ring_window).
    pub fn putWindow(md: *MultiDrafter, slot: usize, n: usize, buf: u64) !void {
        try md.moveWindow(slot, n, buf, false);
        md.end[slot] = n;
    }

    // ------------------------------------------------------------------------------------------- passes ---

    const BlockCtx = struct {
        md: *MultiDrafter,
        sg: draft.Drafter.Segs,
        pub fn go(c: @This()) !void {
            const st = try c.md.v.blockComputeSegs(c.md.dev(), c.sg, c.md.send_words);
            c.md.strides[c.sg.S] = st;
        }
    };

    const TapCtx = struct {
        md: *MultiDrafter,
        T: usize,
        pub fn go(c: @This()) !void {
            const m = c.md;
            try m.v.tapsComputeRows(m.dev(), c.T, m.pl, m.tab + m.lay.tslot, m.tab + m.lay.tpos);
        }
    };

    /// The block pass of `bp` (its tables and ids uploaded through the pinned staging, its graph replayed or
    /// captured). The caller synchronizes before the staging is written again.
    fn runBlock(md: *MultiDrafter, asks: []const Ask, bp: *const BlockPlan) !void {
        const o = md.ops();
        const l = md.lay;
        const N = md.N;
        const rows = bp.S * md.n;
        @memcpy(md.pinSlice(i64, l.meta, 3 * N), bp.meta[0 .. 3 * N]);
        @memcpy(md.pinSlice(i64, l.rpos, rows), bp.rpos[0..rows]);
        @memcpy(md.pinSlice(i64, l.rslot, rows), bp.rslot[0..rows]);
        const spos = md.pinSlice(i64, l.spos, 2 * N);
        @memset(spos, 0);
        for (0..bp.S) |s| spos[2 * s] = bp.meta[2 * N + s];
        try o.upload(md.tab, md.pin.bytes[0..l.block_end]);
        const ids = md.pinSlice(i32, l.pin_ids, rows);
        @memcpy(ids, bp.ids[0..rows]);
        try o.upload(md.v.b.ids, std.mem.sliceAsBytes(ids));
        var sg: draft.Drafter.Segs = .{ .S = bp.S, .St = N, .meta = md.tab + l.meta, .spos = md.tab + l.spos, .rpos = md.tab + l.rpos, .rslot = md.tab + l.rslot, .pl = md.pl, .pool_rows = md.pool_rows };
        for (0..bp.S) |s| sg.slots[s] = asks[bp.ask[s]].slot;
        const sel: u64 = if (md.v.sp.kind == .dspark) bp.mask else bp.S;
        try md.r.runGraph(blockKey(md.which, sel), BlockCtx{ .md = md, .sg = sg });
        md.passes += 1;
    }

    /// MultiDrafter.propose: every ask with a depth and a context drafts in one block pass (the others get none, as a
    /// lone drafter runs none for them); one read of every rank's candidates and the rows' extra words; each stream's
    /// candidates merged and walked by the host stage at its committed length + 1. `out[i]`: ask i's drafts.
    pub fn propose(md: *MultiDrafter, asks: []const Ask, out: []Drafted) !void {
        if (out.len < asks.len) return error.Invalid;
        for (out[0..asks.len]) |*x| x.n = 0;
        const sp = md.v.sp;
        const bp = try blockPlan(asks, md.end[0..md.N], md.N, md.n, sp.maxDepth(), sp.mask_id, sp.kind == .dspark);
        if (bp.S == 0) return;
        try md.runBlock(asks, &bp);
        const o = md.ops();
        const stride = md.strides[bp.S];
        const ranks = md.v.ranks;
        const e = sp.extra();
        const k = sp.top_k;
        const HR = bp.S * md.hr;
        const vpin = md.v.pin orelse return error.Invalid;
        const pf: [*]f32 = @ptrCast(@alignCast(vpin.bytes.ptr));
        const all = pf[0 .. ranks * stride];
        const ex = pf[sp.world * md.send_words ..][0 .. HR * e];
        try o.download(std.mem.sliceAsBytes(all), md.v.b.cand);
        try o.download(std.mem.sliceAsBytes(ex), md.v.b.cand + sp.world * md.send_words * 4);
        try o.s.synchronize();
        const b = md.base;
        for (0..bp.S) |s| {
            const a = asks[bp.ask[s]];
            const depth = bp.depth[s];
            host.mergeCandidates(all[s * md.hr * 2 * k ..], ranks, stride, depth, k, b.tokens, b.values, b.scratch);
            for (0..depth * e) |q| b.extra[q] = ex[s * md.hr * e + q];
            const got = b.walkConf(depth, a.pending, md.end[a.slot] + 1, a.sampling, a.conf, &out[bp.ask[s]].ids);
            out[bp.ask[s]].n = got.len;
        }
    }

    /// Rows [row0, row0 + n) of b.tap_in from a taps buffer (row pitch `pitch` bytes) in this drafter's column
    /// order (dspark_host.tap_runs: one copy a contiguous run).
    fn loadTaps(md: *const MultiDrafter, o: kern.Ops, src: u64, pitch: usize, n: usize, row0: usize) !void {
        const vw = &md.v;
        const D = vw.sp.D;
        const width = vw.sp.ntaps * D * 2;
        const dst = vw.b.tap_in + row0 * width;
        if (vw.identity) {
            if (pitch == width) return o.copy(dst, src, n * width);
            return o.copyRows(dst, width, src, pitch, width, n);
        }
        var runs: [host.max_taps][3]usize = undefined;
        for (host.tapRuns(vw.cols[0..vw.sp.ntaps], &runs)) |run| {
            try o.copyRows(dst + run[0] * D * 2, width, src + run[1] * D * 2, pitch, run[2] * D * 2, n);
        }
    }

    /// One context update (MultiDrafter._launch_taps): the pieces' rows stacked into b.tap_in, their tables through
    /// the next pinned staging slot, the bucket's graph; each piece's slot committed to its end.
    fn update(md: *MultiDrafter, pieces: []const Piece, pitch: usize, upd: *usize) !void {
        if (upd.* == md.updates) { // every staging slot of this commit used: the uploads must have run
            try md.r.f.s.synchronize();
            upd.* = 0;
        }
        const o = md.ops();
        const tm = md.t_max;
        const stage = md.pinSlice(i64, md.lay.pin_taps + upd.* * 2 * tm * 8, 2 * tm);
        upd.* += 1;
        const T = try tapTables(pieces, md.n, tm, stage[0..tm], stage[tm..]);
        var off: usize = 0;
        for (pieces) |pc| {
            try md.loadTaps(o, pc.src, pitch, pc.n, off);
            off += pc.n;
        }
        try o.upload(md.tab + md.lay.tpos, std.mem.sliceAsBytes(stage));
        try md.r.runGraph(tapKey(md.which, T), TapCtx{ .md = md, .T = T });
        for (pieces) |pc| md.end[pc.slot] = pc.p0 + pc.n;
    }

    /// MultiDrafter.commit: each item's rows into its slot's context from its committed length on, every item's rows
    /// stacked into context updates of up to t_max rows (each a captured bucket of block multiples). `pitch`: the
    /// taps buffer's row bytes.
    pub fn commit(md: *MultiDrafter, items: []const Commit, pitch: usize) !void {
        if (items.len > max_streams) return error.Invalid;
        if (md.inflight) try md.r.f.s.synchronize(); // the last commit's tables may still be uploading
        md.inflight = true;
        var pieces: [max_streams]Piece = undefined;
        var np: usize = 0;
        var used: usize = 0;
        var upd: usize = 0;
        for (items) |it| {
            if (it.slot >= md.N) return error.Invalid;
            var start: usize = 0;
            var at = md.end[it.slot];
            while (start < it.n) {
                const take = @min(it.n - start, md.t_max - used);
                pieces[np] = .{ .slot = it.slot, .src = it.src + start * pitch, .n = take, .p0 = at };
                np += 1;
                start += take;
                at += take;
                used += take;
                if (used == md.t_max) {
                    try md.update(pieces[0..np], pitch, &upd);
                    np = 0;
                    used = 0;
                }
            }
        }
        if (np > 0) try md.update(pieces[0..np], pitch, &upd);
    }

    /// A prompt chunk's rows [a, e) of a prompt of L0 tokens (`b`'s taps) into slot's context (multi._chunk's
    /// commit). Rows no block can read any more (older than the ring's reach at the prompt's end) only move the
    /// committed length: each row's K / V is its own (row-invariant passes), so the rows kept are the same bits.
    pub fn feedPrompt(md: *MultiDrafter, b: *const fw.Buffers, slot: usize, a: usize, e: usize, L0: usize) !void {
        if (b.taps == 0 or e <= a) return;
        const keep = host.ring - host.max_block - draft.tap_rows; // >= every drafter's window + block
        const from = @min(@max(L0 -| keep, a), e);
        md.end[slot] += from - a;
        if (e == from) return;
        const pitch = b.tap_n * md.v.sp.D * 2;
        const one = [1]Commit{.{ .slot = slot, .src = b.taps + (from - a) * pitch, .n = e - from }};
        try md.commit(&one, pitch);
    }

    /// MultiDrafter.capture (every rank together, no messages): context updates of every row bucket (all padding:
    /// nothing written), then block passes of 1..N blocks (DSpark: every slot set up to 6 slots, else the first S
    /// slots; other sets capture on first use) at committed length 1; every committed length back to 0 after.
    /// Returns the passes run.
    pub fn prewarm(md: *MultiDrafter) !usize {
        const o = md.ops();
        const tm = md.t_max;
        var count: usize = 0;
        const stage = md.pinSlice(i64, md.lay.pin_taps, 2 * tm);
        @memset(stage[0..tm], 0);
        @memset(stage[tm..], -1);
        try o.upload(md.tab + md.lay.tpos, std.mem.sliceAsBytes(stage));
        var T = md.n;
        while (T <= tm) : (T += md.n) {
            try md.r.runGraph(tapKey(md.which, T), TapCtx{ .md = md, .T = T });
            count += 1;
        }
        try md.r.f.s.synchronize();
        const N = md.N;
        md.end = @splat(0);
        for (0..N) |s| md.end[s] = 1;
        defer md.end = @splat(0);
        const every = md.v.sp.kind == .dspark and N <= 6;
        const last: u64 = if (every) (@as(u64, 1) << @intCast(N)) - 1 else N;
        var sel: u64 = 1;
        while (sel <= last) : (sel += 1) {
            const mask: u64 = if (every) sel else (@as(u64, 1) << @intCast(sel)) - 1;
            var asks: [max_streams]Ask = undefined;
            var na: usize = 0;
            for (0..N) |s| {
                if (mask & (@as(u64, 1) << @intCast(s)) == 0) continue;
                asks[na] = .{ .slot = s, .pending = 1000, .depth = md.v.sp.maxDepth(), .conf = 0 };
                na += 1;
            }
            const bp = try blockPlan(asks[0..na], md.end[0..N], N, md.n, md.v.sp.maxDepth(), md.v.sp.mask_id, md.v.sp.kind == .dspark);
            try md.runBlock(asks[0..na], &bp);
            try md.r.f.s.synchronize(); // the pinned tables are written again next
            count += 1;
        }
        return count;
    }
};

// ------------------------------------------------------------------------------------------------ coverage ---

/// Every Triton launch a concurrent drafter of shapes `sp` can make at N streams (cuda_coverage.zig): context updates
/// of every row bucket and block passes of 1..N blocks, through the compute functions themselves in probe mode.
pub fn enumerate(p: *aot.Probe, sp: draft.Spec, N: usize) !void {
    if (!fits(sp, N)) return error.DrafterTooWide;
    const dr = try draft.Drafter.probe(p.gpa, sp);
    defer dr.deinit();
    const dv: draft.Dev = .{ .t = .{ .set = null, .s = undefined, .probe = p } };
    const A: u64 = 1 << 20;
    const pl: draft.Drafter.PoolLayout = if (sp.kind == .dflash) .{ .head_stride = N * ring_rows, .slot_stride = ring_rows } else .{ .head_stride = ring_rows, .slot_stride = sp.KV * ring_rows };
    var T = sp.block;
    while (T <= tapMax(sp.block)) : (T += sp.block) try dr.tapsComputeRows(dv, T, pl, A, A);
    const send = std.mem.alignForward(usize, N * sp.headRows() * 2 * sp.top_k, 4);
    for (1..N + 1) |S| {
        var sg: draft.Drafter.Segs = .{ .S = S, .St = N, .meta = A, .spos = A, .rpos = A, .rslot = A, .pl = pl, .pool_rows = if (sp.kind == .dflash) N * ring_rows else ring_rows };
        for (0..S) |s| sg.slots[s] = N - 1 - s; // any slots: every ring address stays 16-aligned
        _ = try dr.blockComputeSegs(dv, sg, send);
    }
}

// ------------------------------------------------------------------------------------------------- tests ---

test "frows and fits: 4 streams take 8 rows (7 drafts) a window, N blocks fit one 64-row pass" {
    const df: draft.Spec = .{ .kind = .dflash, .D = 6144, .hd = 128, .H = 16, .KV = 2, .I = 3072, .layers = 6, .ntaps = 6, .block = 8, .top_k = 16, .V = 38720, .vocab_off = 0, .rsel = 256, .gs = 16, .eps = 1e-5, .window = 2047, .causal = false, .world = 4, .theta = 1e6, .mask_id = 154856 };
    var ds = df;
    ds.kind = .dspark;
    try std.testing.expectEqual(@as(usize, 8), frows(df, 4)); // block 8 = pending + 7 drafts
    try std.testing.expectEqual(@as(usize, 8), frows(ds, 4)); // block + 1 = 9, capped at 32 / 4
    try std.testing.expectEqual(@as(usize, 9), frows(ds, 2));
    try std.testing.expectEqual(@as(usize, 4), frows(df, 8));
    try std.testing.expect(fits(df, 8));
    try std.testing.expect(!fits(df, 9)); // 72 rows: past one pass
    try std.testing.expectEqual(@as(usize, 64), tapMax(8));
    try std.testing.expectEqual(@as(usize, 63), tapMax(7));
    try std.testing.expectEqual(@as(usize, 16), tapBucket(9, 8));
    try std.testing.expectEqual(@as(usize, 8), tapBucket(8, 8));
}

test "block plan as MultiDrafter.propose's tables: live asks only, DSpark in slot order" {
    const end = [_]usize{ 100, 0, 7, 40 };
    const asks = [_]Ask{
        .{ .slot = 3, .pending = 11, .depth = 5, .conf = 0.3 },
        .{ .slot = 1, .pending = 12, .depth = 7, .conf = 0.3 }, // no context yet: no block
        .{ .slot = 0, .pending = 13, .depth = 0, .conf = 0.3 }, // no depth: no block
        .{ .slot = 2, .pending = 14, .depth = 9, .conf = 0.3 }, // depth capped at 7
    };
    const f = try blockPlan(&asks, &end, 4, 8, 7, 999, false);
    try std.testing.expectEqual(@as(usize, 2), f.S);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 }, f.ask[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 5, 7 }, f.depth[0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 11, 14, 0, 0, 3, 2, 0, 0, 40, 7, 0, 0 }, f.meta[0..12]);
    try std.testing.expectEqualSlices(i64, &.{ 40, 41, 42, 43, 44, 45, 46, 47, 7, 8 }, f.rpos[0..10]);
    try std.testing.expectEqualSlices(i64, &.{ 3, 3, 3, 3, 3, 3, 3, 3, 2, 2 }, f.rslot[0..10]);
    try std.testing.expectEqualSlices(i32, &.{ 11, 999, 999, 999, 999, 999, 999, 999, 14, 999 }, f.ids[0..10]);
    try std.testing.expectEqual(@as(u64, 0b1100), f.mask);
    const s = try blockPlan(&asks, &end, 4, 8, 8, 999, true);
    try std.testing.expectEqualSlices(usize, &.{ 3, 0 }, s.ask[0..2]); // slot 2 first, then slot 3
    try std.testing.expectEqualSlices(usize, &.{ 8, 5 }, s.depth[0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 14, 11, 0, 0, 2, 3, 0, 0, 7, 40, 0, 0 }, s.meta[0..12]);
    try std.testing.expectEqual(f.mask, s.mask);
    const twice = [_]Ask{ .{ .slot = 0, .pending = 1, .depth = 2, .conf = 0 }, .{ .slot = 0, .pending = 1, .depth = 2, .conf = 0 } };
    try std.testing.expectError(error.Invalid, blockPlan(&twice, &end, 4, 8, 7, 999, false));
}

test "context update tables: stacked pieces, the bucket's padding writes nothing" {
    var pos: [64]i64 = undefined;
    var slot: [64]i64 = undefined;
    const pieces = [_]Piece{ .{ .slot = 2, .src = 0, .n = 3, .p0 = 50 }, .{ .slot = 0, .src = 0, .n = 6, .p0 = 9 } };
    const T = try tapTables(&pieces, 8, 64, &pos, &slot);
    try std.testing.expectEqual(@as(usize, 16), T);
    try std.testing.expectEqualSlices(i64, &.{ 50, 51, 52, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0, 0, 0, 0 }, pos[0..16]);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2, 2, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1 }, slot[0..16]);
    const wide = [_]Piece{.{ .slot = 1, .src = 0, .n = 65, .p0 = 0 }};
    try std.testing.expectError(error.Invalid, tapTables(&wide, 8, 64, &pos, &slot));
}

test "the device tables' offsets: META at the base, every Triton pointer 16-aligned, regions disjoint" {
    for ([_]usize{ 1, 2, 3, 4, 7, 8 }) |N| {
        const l = Lay.make(N, 8, 64);
        try std.testing.expectEqual(@as(usize, 0), l.meta);
        for ([_]usize{ l.rpos, l.rslot, l.spos, l.tpos }) |x| try std.testing.expectEqual(@as(usize, 0), x % 16);
        try std.testing.expect(l.rpos >= 3 * N * 8 and l.rslot >= l.rpos + N * 64 and l.spos >= l.rslot + N * 64);
        try std.testing.expect(l.tpos >= l.block_end and l.tslot == l.tpos + 64 * 8 and l.total == l.tslot + 64 * 8);
        try std.testing.expect(l.pin_ids >= l.block_end and l.pin_taps >= l.pin_ids + N * 8 * 4);
    }
}

test "graph keys of the concurrent drafter passes stay apart from each other and the runner's" {
    try std.testing.expect(tapKey(0, 8) != tapKey(1, 8));
    try std.testing.expect(blockKey(0, 3) != blockKey(1, 3));
    try std.testing.expect((tapKey(0, 8) & 15) == 7 and (blockKey(1, 0b1011) & 15) == 8);
}

test "concurrent drafter coverage reaches the segment kernels (DFlash2, DSpark ft2 shapes at 4 streams)" {
    const gpa = std.testing.allocator;
    var p = aot.Probe.init(gpa);
    defer p.deinit();
    const ds: draft.Spec = .{ .kind = .dspark, .D = 6144, .hd = 64, .H = 16, .KV = 16, .I = 3072, .layers = 3, .ntaps = 5, .block = 8, .top_k = 64, .V = 38720, .vocab_off = 0, .eps = 1e-5, .window = 2048, .causal = true, .world = 4, .theta = 8e6, .mask_id = 154856 };
    const df: draft.Spec = .{ .kind = .dflash, .D = 6144, .hd = 128, .H = 16, .KV = 2, .I = 3072, .layers = 6, .ntaps = 6, .block = 8, .top_k = 16, .V = 38720, .vocab_off = 0, .rsel = 256, .gs = 16, .eps = 1e-5, .window = 2047, .causal = false, .world = 4, .theta = 1e6, .mask_id = 154856 };
    try enumerate(&p, ds, 4);
    try enumerate(&p, df, 4);
    for ([_][]const u8{ "_dconv_seg", "_dattn_seg", "_block_attn_kernel", "_prep_kernel", "_rmsnorm", "_group_sums", "_embed_b16", "_swiglu", "_residual_add" }) |fname| {
        var hit = false;
        for (p.needs.values()) |nd| hit = hit or std.mem.eql(u8, nd.function, fname);
        if (!hit) std.debug.print("concurrent drafter coverage: {s} never enumerated\n", .{fname});
        try std.testing.expect(hit);
    }
    try std.testing.expectError(error.DrafterTooWide, enumerate(&p, df, 9));
}
