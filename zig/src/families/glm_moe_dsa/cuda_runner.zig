//! GLM-5.3's decoding in Zig (Phase 2b), runner.py's single-stream rounds: the prompt in equal chunks (prefixes.
//! even_chunks with TF_GLM53_PROMPT_ROWS = the window) through the target and the MTP layer, then MTP-drafted rounds -
//! the MTP layer takes the rows the last round accepted (normed target hidden, next token) and drafts k tokens (its
//! first step over those rows, k - 1 one-row steps chaining its own hidden and draft), the target verifies
//! [pending token, drafts] in one window - replayed as CUDA graphs keyed like GraphSet's ("tgt", R, T, pick) and
//! ("mtp", m, j, T, keep, reuse), captured on first use after an eager run (or all of them by `prewarm`).
//!
//! Picks: greedy - the device argmax head (fused.head "argmax"); sampled - the verify head "local" (this rank's
//! logits), then on the host each rank's top candidates (sampling.selectTop), one all-gather of them, and
//! sampling.choose at each row's position (Runner._sample_local; top_k 0: Glm53Engine._sample's 264-candidate rule).
//! A verify row computes what a serial step at its position computes and draws are keyed by (seed, position, id), so
//! a drafted reply equals the serial one (k = 0) and the Python engine's.
//!
//! Phase 3a (Settings.served): the served prompt path (chunks of PROMPT_ROWS / PROMPT_ROWS_SHORT rows, the two-micro-
//! batch sequence-parallel chunk of cuda_prompt.zig), the RoCE one-shot for decode windows through Forward.fast (the
//! head's records and the sampled candidates gathered over it too) and runner.healthy after each window's picks.
//! Phase 3b (`run` with a Gen): EOS stops a run (stop_eos), rank 0's stop wish rides on the verify window's own
//! exchange (StopVote: the argmax head's spare record column, the sampled candidates' spare word), prompt reuse's
//! cuts / keep points / resumed begin / replayed head (cuda_reuse.zig: prefixes.PromptReuse), decode context
//! parallelism (Settings.dcp: the caches interleaved over the ranks; no sequence-parallel prompt chunks then).
//! `generate` stays Phase 3a's schedule (caches zeroed first, EOS not a stop) for tf-glm53-generate --mode 3a.
//! Phase 4 (Gen.mode, Gen.copies): the drafters (cuda_draft.zig) - DSpark and DFlash2 rounds (Runner._generate_dflash:
//! the drafter proposes after the pending token, the target verifies [pending, drafts] in one window, the kept rows'
//! taps extend every loaded drafter's context), and copy drafts (copies.zig: Runner._copied, verify windows of 8 and
//! COPY_MAX + 1 rows besides the drafters', a copied MTP round leaving the MTP layer a backlog). The target's tap
//! layers' rows land in every Buffers' taps (Forward.taps); prompt chunks feed the drafters too.
//! Not here: the depth policy, auto mode, late tokens (each round's tokens reach the hook at once), concurrency.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const smp = @import("sampling.zig");
const W = @import("cuda_weights.zig");
const prompt = @import("cuda_prompt.zig");
const draft = @import("cuda_draft.zig");
const dhost = @import("draft_host.zig");
const copies = @import("copies.zig");

pub const Settings = struct {
    /// MTP drafts a round the buffers and graphs are built for (a request may ask fewer; 0: serial only)
    k: usize = 2,
    /// CUDA graphs for decode windows (false: every window eager)
    graphs: bool = true,
    /// prompt chunk rows (Runner.prompt_rows; both engines use TF_GLM53_PROMPT_ROWS = SHORT = this)
    window: usize = 128,
    /// TF_GLM53_MTP_REUSE (0, 1, 2): draft steps 2.. attend the first step's selection
    reuse: u32 = 2,
    /// TF_GLM53_MTP "normed/normed": the MTP layer reads final-normed target rows, chains shared_head-normed rows
    hid_normed: bool = true,
    chain_normed: bool = true,
    /// Phase 3a: the served prompt path - chunks of fused.PROMPT_ROWS (TF_GLM53_PROMPT_ROWS 8192; PROMPT_ROWS_SHORT
    /// 4096 under 3 x PROMPT_ROWS), the two-micro-batch sequence-parallel chunk where it fits (TF_GLM53_PROMPT_SP),
    /// wider windows through Forward.wide; false: Phase 2b's chunks of `window` rows
    served: bool = false,
    prompt_rows: usize = 8192,
    prompt_rows_short: usize = 4096,
    prompt_sp: bool = true,
    /// TF_GLM53_ROCE_HEALTH: after each window's picks are read, every rank agrees no RoCE runtime timed out
    roce_health: bool = true,
    /// the sequence-parallel chunk (its comm stream and events), owned by the caller; null: no overlap
    sp: ?*prompt.Sp = null,
    /// Phase 3a speed: sampled windows pick each rank's candidates on the device (the multi-block top-k over the
    /// local logits in sampling.topBetter's order, the Python engine's torch.topk) and read the gathered candidates
    /// once; false: the whole local logits rows go to the host for sampling.selectTop (the same candidates)
    device_cands: bool = true,
    /// Phase 3b: decode context parallelism (fused.Weights.dcp: 1 or the world; Glm53Engine turns it on past
    /// DCP_AUTO tokens of context): the target's and the MTP layer's cache rows interleaved over the ranks
    /// (position p on rank p % dcp at row p / dcp), so no sequence-parallel prompt chunks (compute_prompt_sp needs 1)
    dcp: usize = 1,
    /// Phase 4: TF_GLM53_VERIFY_ROWS (runner.vrows when a drafter taps the target: 8) and a drafter's own verify rows
    /// (DSpark: block + 1)
    verify_rows: usize = 8,
    draft_rows: usize = 0,
    /// Phase 4: copy drafts (TF_GLM53_COPY_DRAFTS / _MIN / _MAX)
    copy: copies.Settings = .{ .on = false, .match = 0, .most = 0 },
    /// Phase 4b (--parallel N): the caches hold this many streams' slots (fused.State(slots=N)); `run` uses slot 0,
    /// cuda_multi.zig drives the rest. DCP 1 only.
    slots: usize = 1,
};

/// What drafts a request's rounds (runner.set_mode: the MTP head, DSpark, DFlash2).
pub const Mode = enum { mtp, dspark, dflash };

/// What `run` calls back on every rank (followers' hooks may be no-ops).
pub const Hooks = struct {
    ctx: *anyopaque,
    /// Tokens the run committed, in order (the first pick alone, then each round's). Returns this rank's stop wish
    /// (sticky): only rank 0's counts - it rides on the next verify window's exchange to every rank (StopVote).
    tokens: *const fn (ctx: *anyopaque, toks: []const u32) bool,
    /// A prompt keep point (prefixes.PromptReuse.run's `keep(n, head)`): the caches hold the prompt's rows below `n`
    /// (queued on the compute stream), the MTP carry the target hidden n - 1; at the prompt's end (n = L0) also the
    /// head's local logits row (`head`: device fp32 [vocab_part]) its first token is picked from.
    keep: ?*const fn (ctx: *anyopaque, n: usize, head: ?u64) anyerror!void = null,
};

/// One request's schedule for `run` (Runner.generate's arguments).
pub const Gen = struct {
    max_tokens: usize,
    sampling: ?smp.Sampling = null,
    /// MTP drafts a round (0: the serial reference; at most Settings.k)
    k: usize = 2,
    /// tokens that end the run (stop_eos); empty: the run makes max_tokens picks
    eos: []const u32 = &.{},
    /// prompt reuse: the kept state the caches resume at (its rows, carry restored by the caller), the prompt's keep
    /// points past it (the chunks are cut there: prefixes.cut_chunks), and a kept head row (device fp32
    /// [vocab_part]) when begin == the prompt's length (a replay: nothing prefilled)
    begin: usize = 0,
    cuts: []const usize = &.{},
    replay: ?u64 = null,
    /// zero every cache row first (Phase 3a's reference schedule); served runs keep the live rows (prompt reuse)
    reset: bool = false,
    hooks: ?Hooks = null,
    /// Phase 4: the drafter (mtp: the MTP head with `k`) and copy drafts (Runner.generate's `copies`: drafted runs)
    mode: Mode = .mtp,
    copies: bool = false,
};

pub const Stats = struct {
    prompt_len: usize = 0,
    prefill_s: f64 = 0,
    decode_s: f64 = 0,
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    capture_s: f64 = 0, // graph captures during this request (prewarm's are not counted)
    graphs: usize = 0,
    // Phase 3b
    begin: usize = 0, // prompt tokens resumed from a kept state
    eos: bool = false, // the run ended at an EOS token
    stopped: bool = false, // rank 0's stop wish ended it (StopVote.agreed)
    // Phase 4
    mode: Mode = .mtp,
    copy_rounds: usize = 0,
    copy_drafted: usize = 0,
    copy_accepted: usize = 0,

    pub fn tokS(s: Stats, tokens: usize) f64 {
        return if (s.decode_s > 0 and tokens > 1) @as(f64, @floatFromInt(tokens - 1)) / s.decode_s else 0;
    }

    pub fn perRound(s: Stats, tokens: usize) f64 {
        return @as(f64, @floatFromInt(tokens -| 1)) / @as(f64, @floatFromInt(@max(s.rounds, 1)));
    }
};

/// prefixes.even_chunks: [begin, end) in equal chunks of at most `rows` (`short_rows` under 3 x rows).
pub fn evenChunks(gpa: std.mem.Allocator, begin: usize, end: usize, rows: usize, short_rows: usize) ![][2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    errdefer out.deinit(gpa);
    if (end > begin) {
        const L = end - begin;
        var r = rows;
        if (L < 3 * rows) r = @min(rows, short_rows);
        const n = (L + r - 1) / r;
        const step = (L + n - 1) / n;
        var a = begin;
        while (a < end) : (a += step) try out.append(gpa, .{ a, @min(a + step, end) });
    }
    return out.toOwnedSlice(gpa);
}

/// prefixes.cut_chunks (Runner.segments): [begin, end) cut at the `cuts` inside it (ascending, duplicates allowed),
/// each piece in even chunks - a prompt resumed at any of its cuts runs the chunks a fresh one runs past it.
pub fn cutChunks(gpa: std.mem.Allocator, begin: usize, end: usize, cuts: []const usize, rows: usize, short_rows: usize) ![][2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    errdefer out.deinit(gpa);
    const sorted = try gpa.dupe(usize, cuts);
    defer gpa.free(sorted);
    std.mem.sort(usize, sorted, {}, std.sort.asc(usize));
    var a = begin;
    var last: ?usize = null;
    for (sorted) |t| {
        if (t <= begin or t >= end) continue;
        if (last != null and last.? == t) continue; // set(): each cut once
        last = t;
        const piece = try evenChunks(gpa, a, t, rows, short_rows);
        defer gpa.free(piece);
        try out.appendSlice(gpa, piece);
        a = t;
    }
    const tail = try evenChunks(gpa, a, end, rows, short_rows);
    defer gpa.free(tail);
    try out.appendSlice(gpa, tail);
    return out.toOwnedSlice(gpa);
}

/// fused.bucket as the runner takes it for decode windows.
fn bucketOf(t: usize, topk: usize) ?usize {
    return tri.bucket(t, topk);
}

fn tIndex(T: ?usize) u64 {
    const t = T orelse return 0;
    return std.math.log2_int(usize, t) + 1;
}

const Pick = enum(u1) { argmax = 0, local = 1 };

pub const Runner = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    f: fw.Forward,
    w: *const W.Weights,
    set: Settings,
    capacity: usize,
    topk: usize,
    reuse: u32, // the MTP layer has an indexer: TF_GLM53_MTP_REUSE, else 0
    st: fw.State,
    pb: fw.Buffers, // prompt chunks
    vb: fw.Buffers, // verify windows (decode class, k + 1 rows)
    mb: ?fw.Buffers, // MTP windows (decode class, k + 1 rows)
    carry: cuda.DeviceBuffer, // bf16 [D]: the last prompt row's normed hidden (its MTP row waits for its token)
    pin: cuda.HostBuffer, // page-locked staging for every host <-> device copy of a round
    graphs: std.AutoHashMapUnmanaged(u64, cuda.graph.Exec) = .empty,
    capture_s: f64 = 0,
    // runGraph's outcomes since init: replays of a captured graph, eager runs (a miss: run, then captured)
    graph_runs: u64 = 0,
    eager_runs: u64 = 0,
    cands: []smp.Cand,
    // Phase 3a (Settings.served): the second micro-batch's buffers, the sequence-parallel chunk, the prompt ids on
    // the device (uploaded once a request), the RoCE health words
    pb1: ?fw.Buffers = null,
    sp: ?*prompt.Sp = null,
    ids_dev: ?cuda.DeviceBuffer = null, // int64 [capacity]
    health: ?cuda.DeviceBuffer = null, // int32 [1 + world]
    // Phase 4: the drafters (cuda_draft.zig; the engine attaches them), the verify widths (runner.widths: a bit a
    // width), the MTP windows' widest backlog, copy drafts
    dflash: ?*draft.Drafter = null,
    dspark: ?*draft.Drafter = null,
    widths: u64 = 0,
    vrows: usize = 1,
    mrows: usize = 1,
    copy_on: bool = false,
    /// Phase 4c: prompt chunks feed the one-stream drafters (false while cuda_multi.zig fills a stream's chunk: the
    /// stream's own concurrent drafter takes its taps)
    solo_feed: bool = true,

    fn pinLayout(r: *const Runner) struct { lg: usize, send: usize, all: usize, ids: usize, rd: usize, vote: usize, vids: usize, total: usize } {
        const V = r.w.vocab_part;
        const hr = r.vb.head_rows;
        const lg_b = std.mem.alignForward(usize, @max(hr, 1) * V * 4, 256);
        const words = std.mem.alignForward(usize, r.vb.cand_words, 4);
        const send_b = std.mem.alignForward(usize, words * 4, 256);
        const all_b = std.mem.alignForward(usize, r.w.world * words * 4, 256);
        const id_rows = if (r.set.served) r.capacity else @max(r.set.window, 64);
        const ids_b = std.mem.alignForward(usize, id_rows * 8, 256);
        const rd_b: usize = 2 * fw.decode_rows * 8;
        const vote_b: usize = 256; // the stop vote's word (StopVote), read with the window's picks
        const vids_b: usize = std.mem.alignForward(usize, fw.decode_rows * 8, 256); // a host-drafted window's ids
        const rd = lg_b + send_b + all_b + ids_b;
        return .{ .lg = 0, .send = lg_b, .all = lg_b + send_b, .ids = lg_b + send_b + all_b, .rd = rd, .vote = rd + rd_b, .vids = rd + rd_b + vote_b, .total = rd + rd_b + vote_b + vids_b };
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, f: fw.Forward, capacity: usize, set_in: Settings) !Runner {
        const w = f.w;
        const c = w.config;
        var set = set_in;
        if (set.dcp != 1 and set.dcp != w.world) return error.BadDcp;
        if (set.slots == 0 or (set.slots > 1 and set.dcp != 1)) return error.BadDcp; // concurrent streams: DCP off
        if (set.dcp > 1 and !set.served) return error.BadDcp; // the served prompt path only
        if (set.dcp > 1) { // Runner.__init__ with dcp > 1: prompt chunks of at most 1024 rows, no overlap
            set.prompt_rows = @min(set.prompt_rows, 1024);
            set.prompt_rows_short = @min(set.prompt_rows_short, set.prompt_rows);
        }
        if (set.k > 0 and w.mtp == null) return error.NoMtpLayer;
        if (set.window == 0 or set.window > fw.max_rows) return error.WindowTooWide;
        if (set.served and (set.prompt_rows == 0 or set.prompt_rows > fw.max_prompt_rows or set.prompt_rows_short == 0)) return error.WindowTooWide;
        const topk = c.index_topk;
        const cols = bucketOf(capacity, topk) orelse 1;
        // Phase 4 (Runner.__init__): verify windows of the drafters' widths and the copy widths
        const taps = f.taps.n;
        const vrows = @max(@max(set.k + 1, if (taps > 0) set.verify_rows else 1), set.draft_rows);
        const copy_on = set.copy.on and (set.k > 0 or taps > 0);
        const crows: usize = if (copy_on) set.copy.most + 1 else 1;
        var widths: u64 = 0;
        for (1..vrows + 1) |R| widths |= @as(u64, 1) << @intCast(R);
        if (copy_on) widths |= (@as(u64, 1) << @intCast(@min(8, crows))) | (@as(u64, 1) << @intCast(crows));
        const wrows: usize = 63 - @clz(widths);
        if (wrows > fw.decode_rows) return error.WindowTooWide;
        const mrows = @max(if (taps > 0) vrows else set.k + 1, crows);
        var r: Runner = undefined;
        r.gpa = gpa;
        r.io = io;
        r.d = d;
        r.f = f;
        r.w = w;
        r.set = set;
        r.capacity = capacity;
        r.topk = topk;
        r.reuse = if (set.k > 0 and w.mtp.?.layer.indexer != null) set.reuse else 0;
        r.graphs = .empty;
        r.capture_s = 0;
        r.graph_runs = 0;
        r.eager_runs = 0;
        r.pb1 = null;
        r.sp = null;
        r.ids_dev = null;
        r.health = null;
        r.dflash = null;
        r.dspark = null;
        r.widths = widths;
        r.vrows = vrows;
        r.mrows = mrows;
        r.copy_on = copy_on;
        r.solo_feed = true;
        const shared = f.shared_experts;
        r.st = try fw.State.initSlots(gpa, d, w, capacity, set.dcp, set.slots);
        errdefer r.st.deinit();
        if (set.served) {
            // Runner.__init__: pb (prompt_rows, exact key range), pb1 (the second micro-batch) when chunks overlap
            r.pb = try fw.Buffers.initWith(gpa, d, w, set.prompt_rows, @max(capacity, 1), .{ .mtp = set.k > 0, .prompt = true, .shared_experts = shared, .mtp_dense = set.k > 0 and f.mtp_dense != null, .dcp = set.dcp, .taps = taps });
        } else {
            r.pb = try fw.Buffers.initWith(gpa, d, w, set.window, @max(capacity, 1), .{ .mtp = set.k > 0, .taps = taps });
        }
        errdefer r.pb.deinit();
        if (set.served and set.prompt_sp and set.sp != null and w.world > 1 and set.dcp == 1) { // fused: sp needs dcp 1
            r.pb1 = try fw.Buffers.initWith(gpa, d, w, set.prompt_rows - set.prompt_rows / 2, @max(capacity, 1), .{ .prompt = true, .shared_experts = shared, .dcp = set.dcp, .taps = taps });
            r.sp = set.sp;
        }
        errdefer if (r.pb1) |*b| b.deinit();
        r.vb = try fw.Buffers.initWith(gpa, d, w, wrows, cols, .{ .decode = true, .shared_experts = shared, .dcp = set.dcp, .taps = taps });
        errdefer r.vb.deinit();
        r.mb = null;
        if (set.k > 0) r.mb = try fw.Buffers.initWith(gpa, d, w, @max(@max(set.k + 1, vrows), crows), cols, .{ .decode = true, .mtp = true, .shared_experts = shared, .dcp = set.dcp });
        errdefer if (r.mb) |*m| m.deinit();
        r.carry = try cuda.DeviceBuffer.alloc(d, c.hidden * 2);
        errdefer r.carry.free();
        try r.carry.fill8(0, null);
        const lay = r.pinLayout();
        r.pin = try cuda.HostBuffer.alloc(d, lay.total);
        errdefer r.pin.free();
        if (set.served) {
            r.ids_dev = try cuda.DeviceBuffer.alloc(d, capacity * 8);
            r.health = try cuda.DeviceBuffer.alloc(d, (1 + w.world) * 4);
        }
        errdefer if (r.ids_dev) |*b| b.free();
        errdefer if (r.health) |*b| b.free();
        r.cands = try gpa.alloc(smp.Cand, w.world * smp.max_candidates);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        var it = r.graphs.valueIterator();
        while (it.next()) |e| e.deinit();
        r.graphs.deinit(r.gpa);
        r.gpa.free(r.cands);
        if (r.ids_dev) |*b| b.free();
        if (r.health) |*b| b.free();
        if (r.pb1) |*b| b.deinit();
        r.pin.free();
        r.carry.free();
        if (r.mb) |*m| m.deinit();
        r.vb.deinit();
        r.pb.deinit();
        r.st.deinit();
        r.* = undefined;
    }

    fn ops(r: *const Runner) kern.Ops {
        return .{ .k = r.f.k, .s = r.f.s };
    }

    fn now(r: *const Runner) i128 {
        return @intCast(std.Io.Clock.awake.now(r.io).toNanoseconds());
    }

    fn since(r: *const Runner, t0: i128) f64 {
        return @as(f64, @floatFromInt(r.now() - t0)) / 1e9;
    }

    /// An int32 device scalar set on the stream (Tensor.fill_: no host sync, unlike a pageable copy).
    fn setI32(r: *const Runner, dst: u64, v: usize) !void {
        try r.ops().fill32(dst, @intCast(v), 1);
    }

    /// An int64 device id (< 2^31) set on the stream: its low word, then a zero high word.
    fn setId(r: *const Runner, dst: u64, v: u32) !void {
        try r.ops().fill32(dst, v, 1);
        try r.ops().fill32(dst + 4, 0, 1);
    }

    /// `n` values of T at byte `off` of the pinned staging (every region starts 256-byte aligned).
    fn pinSlice(r: *const Runner, comptime T: type, off: usize, n: usize) []T {
        const p: [*]T = @ptrCast(@alignCast(r.pin.bytes.ptr + off));
        return p[0..n];
    }

    /// int64 ids from the host through the pinned staging (async; the staging is reused only after a sync).
    pub fn uploadIds(r: *Runner, dst: u64, ids: []const u32) !void {
        const lay = r.pinLayout();
        if (ids.len * 8 > lay.rd - lay.ids) return error.WindowTooWide;
        const host = r.pinSlice(i64, lay.ids, ids.len);
        for (ids, 0..) |t, i| host[i] = t;
        try r.ops().upload(dst, std.mem.sliceAsBytes(host));
    }

    /// Every cache row and position zeroed on the compute stream (each request starts from zeros, both engines).
    pub fn resetState(r: *Runner) !void {
        for (r.st.mem.items) |*m| try m.fill8(0, r.f.s.handle);
        try r.f.s.synchronize();
    }

    // ------------------------------------------------------------------------------------------------ graphs ---

    /// GraphSet.run: replay `key`'s graph, or run `ctx.go()` eagerly (the step's real result) and capture it.
    pub fn runGraph(r: *Runner, key: u64, ctx: anytype) !void {
        if (r.graphs.get(key)) |e| {
            r.graph_runs += 1;
            return e.launchOn(r.f.s);
        }
        r.eager_runs += 1;
        try ctx.go();
        if (!r.set.graphs) return;
        const t0 = r.now();
        try r.f.s.synchronize();
        try cuda.graph.beginCapture(r.f.s, .thread_local);
        ctx.go() catch |e| {
            if (cuda.graph.endCapture(r.f.s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return e;
        };
        var g = try cuda.graph.endCapture(r.f.s);
        defer g.deinit();
        var ex = try g.instantiate();
        errdefer ex.deinit();
        try ex.upload(r.f.s);
        try r.f.s.synchronize();
        try r.graphs.put(r.gpa, key, ex);
        r.capture_s += r.since(t0);
    }

    const VerifyCtx = struct {
        r: *Runner,
        R: usize,
        T: ?usize,
        pick: Pick,
        fn go(c: @This()) !void {
            try c.r.f.compute(&c.r.vb, &c.r.st, c.R, c.T, if (c.pick == .argmax) .argmax else .local, false);
        }
    };

    /// Runner._verify: the target over vb.ids[:R] at P (logits "all", head `pick`).
    fn verify(r: *Runner, R: usize, P: usize, T: ?usize, pick: Pick) !void {
        try r.setI32(r.st.pos, P);
        const key: u64 = 1 | (@as(u64, R) << 4) | (tIndex(T) << 12) | (@as(u64, @intFromEnum(pick)) << 20);
        try r.runGraph(key, VerifyCtx{ .r = r, .R = R, .T = T, .pick = pick });
    }

    const MtpCtx = struct {
        r: *Runner,
        m: usize,
        j: usize,
        T: ?usize,
        keep: bool,
        reuse: u32,
        fn go(c: @This()) !void {
            const r = c.r;
            const mb = &r.mb.?;
            const D = r.w.config.hidden;
            try r.f.mtpCompute(mb, &r.st, c.m, c.T, .{ .head = true, .chain_normed = r.set.chain_normed, .keep_sel = c.keep, .reuse = c.reuse });
            const o = r.ops();
            try o.copy(r.vb.ids + c.j * 8, mb.argmax, 8); // vb.ids[j] = the draft
            try o.copy(mb.ids, mb.argmax, 8); // it feeds the next step
            try o.copy(mb.hin, mb.hidden + (c.m - 1) * D * 2, D * 2); // with this step's last hidden
        }
    };

    /// Runner._mtp: an MTP window of m rows at P0 ..; its last row's draft goes to verify slot j.
    fn mtp(r: *Runner, m: usize, j: usize, P0: usize, T: ?usize, reuse: u32) !void {
        const keep = j == 1 and r.reuse != 0 and T != null;
        try r.setI32(r.st.mpos, P0);
        const key: u64 = 2 | (@as(u64, m) << 4) | (@as(u64, j) << 8) | (tIndex(T) << 12) | (@as(u64, @intFromBool(keep)) << 20) | (@as(u64, reuse) << 21);
        try r.runGraph(key, MtpCtx{ .r = r, .m = m, .j = j, .T = T, .keep = keep, .reuse = reuse });
    }

    /// Runner.buckets: the index-key ranges a decode window can take up to the capacity (null: below index_topk).
    pub fn buckets(r: *const Runner, out: *[24]?usize) []?usize {
        var n: usize = 0;
        out[n] = null;
        n += 1;
        var t = 2 * r.topk;
        const top = @max(bucketOf(r.capacity, r.topk) orelse 0, 2 * r.topk);
        while (r.topk < r.capacity and t <= top and n < out.len) {
            out[n] = t;
            n += 1;
            t = bucketOf(t + 1, r.topk).?;
        }
        return out[0..n];
    }

    /// Runner.prewarm's captures: every decode graph (each bucket, windows of 1..k+1 rows, both picks, the MTP
    /// steps) up front, so no request pays a capture; the caches are zeroed after (they took scratch values).
    pub fn prewarm(r: *Runner) !usize {
        return r.prewarmWith(true);
    }

    /// Runner.prewarm(capture=...): `capture` false (--parallel N: cuda_multi.zig captures its own windows) runs only
    /// the served prompt path's first use and zeroes the caches after it.
    pub fn prewarmWith(r: *Runner, capture: bool) !usize {
        if (r.set.served) {
            // Runner.prewarm's first step: a prompt of a full chunk, a remainder and rows past index_topk
            // (1000 + 7919 i mod 150000), so the prompt path's first use (cuBLAS's heuristics, NCCL's buffers for
            // the prompt sizes, the unpack cache) is not inside a measured request, as in the Python engine
            const n = @min(r.capacity - 1, r.set.prompt_rows + r.topk + 301);
            const ids = try r.gpa.alloc(u32, n);
            defer r.gpa.free(ids);
            for (ids, 0..) |*v, i| v.* = @intCast(1000 + (i * 7919) % 150000);
            _ = try r.prefill(ids, null, 0, &.{}, null);
            try r.healthy();
        }
        if (!capture) {
            try r.f.s.synchronize();
            try r.resetState();
            return r.graphs.count();
        }
        var bs: [24]?usize = undefined;
        for (r.buckets(&bs)) |T| {
            for (1..fw.decode_rows + 1) |R| {
                if (!r.hasWidth(R)) continue;
                try r.verify(R, 0, T, .argmax);
                try r.verify(R, 0, T, .local);
            }
            if (r.set.k > 0) {
                for (1..r.mrows + 1) |m| try r.mtp(m, 1, 0, T, 0);
                var j: usize = 2;
                while (j <= r.set.k) : (j += 1) {
                    try r.mtp(1, j, 0, T, 0);
                    if (r.reuse != 0 and T != null) try r.mtp(1, j, 0, T, r.reuse);
                }
            }
        }
        // the drafters' passes (Drafter.capture: context updates of 1..block rows, the block pass; every rank
        // together - the block pass gathers), then DSpark's cost cut (Runner._calibrate_dspark)
        for (r.drafterList(), 0..) |maybe, which| {
            const dr = maybe orelse continue;
            try dr.resetAt(r.ops(), 0);
            for (1..dr.sp.block + 1) |n| try r.runGraph(tapKey(which, n), TapCtx{ .r = r, .dr = dr, .n = n });
            try dr.resetAt(r.ops(), 0);
            try r.runGraph(blockKey(which), BlockCtx{ .r = r, .dr = dr });
            try dr.resetAt(r.ops(), 0);
        }
        try r.calibrate();
        try r.f.s.synchronize();
        try r.resetState();
        return r.graphs.count();
    }

    // ------------------------------------------------------------------------------------------ Phase 4 ---

    /// runner.widths: whether verify windows of R rows are captured (a drafter's or a copy's width).
    pub fn hasWidth(r: *const Runner, R: usize) bool {
        return R < 64 and (r.widths >> @intCast(R)) & 1 == 1;
    }

    /// Runner.drafters: DFlash2, then DSpark (the tap plan's order).
    fn drafterList(r: *const Runner) [2]?*draft.Drafter {
        return .{ r.dflash, r.dspark };
    }

    fn tapKey(which: usize, n: usize) u64 {
        return 3 | (@as(u64, n) << 4) | (@as(u64, which) << 12);
    }

    fn blockKey(which: usize) u64 {
        return 4 | (@as(u64, which) << 12);
    }

    /// What the drafters' passes launch on (this runner's stream, its Forward's exchanges).
    fn dev(r: *const Runner) draft.Dev {
        return .{ .t = .{ .set = &r.f.k.triton, .s = r.f.s }, .o = r.ops(), .f = &r.f };
    }

    const TapCtx = struct {
        r: *Runner,
        dr: *draft.Drafter,
        n: usize,
        fn go(c: @This()) !void {
            try c.dr.tapsCompute(c.r.dev(), c.n);
        }
    };

    const BlockCtx = struct {
        r: *Runner,
        dr: *draft.Drafter,
        fn go(c: @This()) !void {
            try c.dr.blockCompute(c.r.dev());
        }
    };

    /// Drafter.add_taps for every loaded drafter: rows [row0, row0 + n) of b's taps into its context, `step` rows a
    /// pass at most (decode: a block, each pass a captured graph; prompts: tap_rows, eager past a block).
    fn feedDrafters(r: *Runner, b: *const fw.Buffers, row0: usize, n: usize, decode: bool) !void {
        if (b.taps == 0) return;
        const pitch = b.tap_n * r.w.config.hidden * 2;
        for (r.drafterList(), 0..) |maybe, which| {
            const dr = maybe orelse continue;
            const step = if (decode) dr.sp.block else draft.tap_rows;
            var a: usize = 0;
            while (a < n) : (a += step) {
                const m = @min(step, n - a);
                try dr.loadTaps(r.ops(), b.taps + (row0 + a) * pitch, pitch, m);
                if (m <= dr.sp.block) {
                    try r.runGraph(tapKey(which, m), TapCtx{ .r = r, .dr = dr, .n = m });
                } else {
                    try dr.tapsCompute(r.dev(), m);
                }
                dr.context_end += m;
            }
        }
    }

    /// A prompt chunk's rows [a, e) of L0 into the drafters' contexts (Runner.prefill_chunk's add_taps). Rows a block
    /// can never read (older than the ring's reach at the prompt's end) only move the position: each row's K / V is
    /// its own (row-invariant passes), so the rows kept are the same bits.
    fn feedPrompt(r: *Runner, b: *const fw.Buffers, a: usize, e: usize, L0: usize) !void {
        if (b.taps == 0) return;
        const keep = dhost.ring - dhost.max_block - draft.tap_rows; // >= every drafter's window + block
        const from = @min(@max(L0 -| keep, a), e);
        if (from > a) {
            for (r.drafterList()) |maybe| {
                const dr = maybe orelse continue;
                try dr.skip(r.ops(), from - a);
            }
        }
        if (e > from) try r.feedDrafters(b, from - a, e - from, false);
    }

    /// A fresh drafter context at `at` for every loaded drafter (Runner.prefill: reset at a new prompt; a resumed or
    /// replayed one starts at its resume point).
    fn resetDrafters(r: *Runner, at: usize) !void {
        for (r.drafterList()) |maybe| {
            const dr = maybe orelse continue;
            try dr.resetAt(r.ops(), at);
        }
    }

    /// Drafter.propose: up to `room` drafts after `tok` (pending at the drafter's context end): the block pass (its
    /// graph), one read, the host stage.
    fn propose(r: *Runner, dr: *draft.Drafter, which: usize, tok: u32, room: usize, s: ?smp.Sampling, out: []u32) ![]u32 {
        const depth = @min(room, dr.sp.maxDepth());
        if (depth == 0 or dr.context_end == 0) return out[0..0];
        try dr.setPending(r.ops(), tok);
        try r.runGraph(blockKey(which), BlockCtx{ .r = r, .dr = dr });
        try dr.readCandidates(r.ops(), depth);
        return dr.walkDrafts(depth, tok, dr.context_end + 1, s, out);
    }

    /// The drafts a drafter round asks for at most (runner: DSpark min(block, vrows - 1, its depth setting),
    /// DFlash2 min(block - 1, vrows - 1, TF_GLM53_DFLASH_DEPTH)).
    fn depthMax(r: *const Runner, dr: *const draft.Drafter) usize {
        const want = if (dr.sp.kind == .dspark) dr.dset.depth else dr.fset.depth;
        return @min(@min(dr.sp.maxDepth(), r.vrows - 1), want);
    }

    /// Runner._copied: this round's copy drafts, at most COPY_MAX, as many as `left` and the cache allow, in a captured
    /// window width - padded up to the next width by repeating the last draft when there is room, else cut to the
    /// widest width below it.
    fn copied(r: *const Runner, cx: *const copies.CopyIndex, left: usize, P: usize, out: []u32) []u32 {
        const room = @min(@min(r.set.copy.most, left), r.capacity -| (P + 1));
        if (room == 0) return out[0..0];
        const got = cx.propose(room, out);
        const R = got.len + 1;
        if (got.len == 0 or r.hasWidth(R)) return got;
        var up = R + 1;
        while (up <= fw.decode_rows and !r.hasWidth(up)) up += 1;
        if (up <= fw.decode_rows and up - 1 <= room and up - 1 <= out.len) {
            for (got.len..up - 1) |i| out[i] = got[got.len - 1];
            return out[0 .. up - 1];
        }
        var down = R - 1;
        while (down > 1 and !r.hasWidth(down)) down -= 1;
        return out[0 .. down - 1];
    }

    /// Runner._mtp_flush: an MTP backlog of m rows (positions P - m .. P - 1) written without drafting from it.
    fn mtpFlush(r: *Runner, m: usize, P: usize) !void {
        if (m > 0) try r.mtp(m, 1, P - m, bucketOf(P, r.topk), 0);
    }

    /// int64 ids of a host-drafted verify window through its own pinned staging (the round's MTP ids use `ids`).
    fn uploadVerify(r: *Runner, ids: []const u32) !void {
        const lay = r.pinLayout();
        if (ids.len > fw.decode_rows) return error.WindowTooWide;
        const hostv = r.pinSlice(i64, lay.vids, ids.len);
        for (ids, 0..) |t, i| hostv[i] = t;
        try r.ops().upload(r.vb.ids, std.mem.sliceAsBytes(hostv));
    }

    /// Runner._calibrate_dspark (policy cost): ms of a round with k = 0 .. block drafts - the verify window of k + 1
    /// rows (fastest of 7 graph replays, made non-decreasing) plus the block pass - the ranks' slowest, so every rank
    /// cuts alike; TF_GLM53_DSPARK_COSTS replaces the measurement. Startup only (the caches hold scratch rows).
    fn calibrate(r: *Runner) !void {
        const dr = r.dspark orelse return;
        if (dr.dset.policy != .cost) return;
        if (dr.dset.n_costs > 0) {
            @memcpy(dr.round_ms[0..dr.dset.n_costs], dr.dset.costs[0..dr.dset.n_costs]);
            dr.n_round_ms = dr.dset.n_costs;
            return;
        }
        const B = dr.sp.block;
        for (1..B + 2) |R| if (!r.hasWidth(R)) return;
        var mine: [dhost.max_block + 3]f32 = @splat(0);
        try r.setI32(r.st.pos, 0);
        for (1..B + 2) |R| {
            var best: f64 = std.math.inf(f64);
            for (0..7) |_| {
                try r.f.s.synchronize();
                const t0 = r.now();
                try r.verify(R, 0, null, .argmax);
                try r.f.s.synchronize();
                best = @min(best, r.since(t0));
            }
            mine[R - 1] = @floatCast(1e3 * best);
        }
        try dr.resetAt(r.ops(), 0);
        var best_b: f64 = std.math.inf(f64);
        for (0..7) |_| {
            try r.f.s.synchronize();
            const t0 = r.now();
            try r.runGraph(blockKey(1), BlockCtx{ .r = r, .dr = dr });
            try r.f.s.synchronize();
            best_b = @min(best_b, r.since(t0));
        }
        try dr.resetAt(r.ops(), 0);
        const n = B + 2;
        mine[n - 1] = @floatCast(1e3 * best_b);
        var got: [dhost.max_block + 3]f32 = mine;
        const world = r.w.world;
        if (world > 1) if (r.f.comm) |cm| {
            var dev_buf = try cuda.DeviceBuffer.alloc(r.d, (world + 1) * n * 4);
            defer dev_buf.free();
            try r.ops().upload(dev_buf.ptr, std.mem.sliceAsBytes(mine[0..n]));
            try cm.allGather(dev_buf.ptr, dev_buf.ptr + n * 4, n, .f32, r.f.s.handle);
            const every = try r.gpa.alloc(f32, world * n);
            defer r.gpa.free(every);
            try r.ops().download(std.mem.sliceAsBytes(every), dev_buf.ptr + n * 4);
            try r.f.s.synchronize();
            for (0..n) |i| {
                var mx: f32 = every[i];
                for (1..world) |q| mx = @max(mx, every[q * n + i]);
                got[i] = mx;
            }
        };
        var run_max: f64 = 0;
        for (0..B + 1) |i| {
            const v: f64 = got[i];
            run_max = if (i == 0) v else @max(run_max, v);
            const blk: f64 = got[n - 1];
            dr.round_ms[i] = @round((run_max + blk) * 1e3) / 1e3;
        }
        dr.n_round_ms = B + 1;
        if (r.w.rank == 0) std.log.info("glm53: DSpark cost cut: round ms with 0..{d} drafts {any} (block pass {d:.2} ms)", .{ B, dr.round_ms[0 .. B + 1], got[n - 1] });
    }

    // ------------------------------------------------------------------------------------------------ prompt ---

    /// Where a keep point's state is reported (Hooks.keep), and on what.
    const Keep = struct {
        hooks: Hooks,
        fn at(k: Keep, n: usize, head: ?u64) !void {
            const f = k.hooks.keep orelse return;
            try f(k.hooks.ctx, n, head);
        }
    };

    /// Runner.prefill + _mtp_prompt: the prompt's chunks through the target (and, when the runner drafts, the MTP
    /// layer's rows a - 1 .. e - 2); returns the first pick (position L0).
    /// Phase 2b: equal chunks of `window` rows. Served (Phase 3a): prefixes.even_chunks with PROMPT_ROWS /
    /// PROMPT_ROWS_SHORT; a chunk of at least 2 MAX_ROWS + 2 rows that sp_fits takes compute_prompt_sp (two micro-
    /// batches, the second's last row holds the head), any other chunk fused.compute; the prompt's ids go to the
    /// device once, and each chunk takes its rows there (no host step between chunks).
    /// Phase 3b: rows from `begin` (a kept state's rows and carry already in place), the chunks cut at `cuts`
    /// (prefixes.cut_chunks), `keep` called at each cut inside (begin, L0) once its chunk is queued and at L0 with the
    /// head's logits row. DCP > 1: never the sequence-parallel chunk (fused.compute_prompt_sp needs dcp 1).
    fn prefill(r: *Runner, prompt_ids: []const u32, s: ?smp.Sampling, begin: usize, cuts: []const usize, keep: ?Keep) !u32 {
        const L0 = prompt_ids.len;
        const rows = if (r.set.served) r.set.prompt_rows else r.set.window;
        const short = if (r.set.served) r.set.prompt_rows_short else r.set.window;
        if (begin >= L0) return error.Invalid;
        if (begin > 0 and !r.set.served) return error.Invalid; // resuming needs the device ids of the served path
        const chunks = try cutChunks(r.gpa, begin, L0, cuts, rows, short);
        defer r.gpa.free(chunks);
        const head: fw.HeadMode = if (s == null) .argmax else .local;
        var hb: *const fw.Buffers = &r.pb; // the buffers whose head picked the first token
        if (r.set.served) {
            const ids = r.ids_dev orelse return error.Invalid;
            try r.uploadIds(ids.ptr, prompt_ids);
        }
        // Runner._prefill: the drafters' contexts start at the prompt (a resumed one: at its resume point - the kept
        // ring rows below it are not restored, which only costs acceptance until the window refills)
        try r.resetDrafters(begin);
        for (chunks, 0..) |ch, ci| {
            const a = ch[0];
            const e = ch[1];
            if (try r.prefillChunk(&r.st, prompt_ids, a, e, ci, head)) |b| hb = b;
            // prefixes.PromptReuse.run's keep(e): a cut the prompt keeps (the caller filters); the state is queued
            if (e != L0) if (keep) |k| {
                for (cuts) |c| if (c == e) {
                    try k.at(e, null);
                    break;
                };
            };
        }
        if (r.f.prof) |p| {
            p.total(r.w.rank, L0 - begin);
            p.on = false; // the prompt only: decode windows are never timed (nor captured with events in them)
        }
        if (keep) |k| try k.at(L0, hb.lg); // Runner.generate: keep(L0, lg) before the first pick moves anything
        return r.pickFirst(hb, L0, s);
    }

    /// One prompt chunk [a, e) of prompt_ids (L0 = its length) into the caches of `st` (the Runner's State, or one
    /// stream's slot view: Runner.prefill_chunk(st, ...)) through the target and, when the runner drafts, the MTP
    /// layer; the served path takes the chunk's ids from the device copy of the prompt (uploaded by the caller). The
    /// last chunk (e == L0) runs `head` on its last row and returns the buffers holding that head row; else null.
    pub fn prefillChunk(r: *Runner, st: *const fw.State, prompt_ids: []const u32, a: usize, e: usize, ci: usize, head: fw.HeadMode) !?*const fw.Buffers {
        const L0 = prompt_ids.len;
        const D = r.w.config.hidden;
        if (a >= e or e > L0) return error.Invalid;
        const R = e - a;
        const t_chunk = r.now();
        const T: ?usize = if (e > r.topk) e else null;
        const last = e == L0;
        var hb: ?*const fw.Buffers = null;
        if (r.set.served) {
            try r.ops().copy(r.pb.ids, r.ids_dev.?.ptr + a * 8, R * 8);
        } else {
            try r.uploadIds(r.pb.ids, prompt_ids[a..e]);
        }
        try r.setI32(st.pos, a);
        const sp_ok = if (r.sp != null and r.pb1 != null and r.set.dcp == 1) R >= prompt.min_overlap_rows and prompt.spFits(r.w.world, r.pb.rows, r.pb1.?.rows, R) else false;
        if (sp_ok) {
            const b1 = &r.pb1.?;
            try r.sp.?.chunk(st, &r.pb, b1, a, R, T);
            if (last) {
                const hR = R / 2;
                const p0 = try r.f.pbegin();
                try r.f.headDevice(b1, b1.x + (R - hR - 1) * D * 2, r.w.final_norm, 1, head);
                try r.f.pend(p0, .head);
                hb = b1;
            }
        } else {
            try r.f.compute(&r.pb, st, R, T, if (last) head else .none, true);
            if (last) hb = &r.pb;
        }
        if (r.solo_feed) try r.feedPrompt(&r.pb, a, e, L0); // prefill_chunk: the drafters' contexts take this chunk's taps
        if (r.set.k > 0) try r.mtpPrompt(st, prompt_ids, a, e, T);
        if (!r.set.served) try r.f.s.synchronize(); // the pinned ids are reused by the next chunk
        if (r.f.prof) |p| {
            if (p.on) { // --prompt-prof: this chunk's phases (a sync a chunk: profiling runs only)
                try r.f.s.synchronize();
                if (r.sp) |sp| try sp.side.s.synchronize();
                try p.report(r.w.rank, ci, a, e, r.since(t_chunk));
            }
        }
        return hb;
    }

    /// The first pick from `hb`'s head row (row 0: the prompt's last row): greedy from its argmax records, sampled
    /// from its local logits (Runner.generate's `sample(lg, L0)`).
    pub fn pickFirst(r: *Runner, hb: *const fw.Buffers, L0: usize, s: ?smp.Sampling) !u32 {
        if (s) |smpl| {
            var pick: [1]u32 = undefined;
            _ = try r.sampleRows(hb, 1, L0, smpl, &pick, false);
            return pick[0];
        }
        const lay = r.pinLayout();
        const got = r.pinSlice(i64, lay.rd, 1);
        try r.ops().download(std.mem.sliceAsBytes(got), hb.argmax);
        try r.f.s.synchronize();
        return @intCast(got[0]);
    }

    /// Runner.generate's `head` replay: a kept prompt end's logits row (device fp32 [vocab_part]) back into the prompt
    /// buffers' head row, its first pick made from it as the prefill made it (greedy: fused.head's argmax records
    /// exchanged; sampled: the candidates) - nothing prefilled, the kept rows and carry in place.
    pub fn replayHead(r: *Runner, lg: u64, L0: usize, s: ?smp.Sampling) !u32 {
        const b = &r.pb;
        const V = r.w.vocab_part;
        const o = r.ops();
        if (b.lg == 0) return error.NoHead;
        try o.copy(b.lg, lg, V * 4);
        if (s == null) {
            // fused.head "argmax" past the router: this rank's first maximum, the records gathered, the lowest rank
            try o.argmaxRec(b.lg, V, V, 0, r.w.vocab_off, b.amax, 1);
            const world = r.w.world;
            if (r.f.fast != null and world > 1) {
                try r.f.fast.?.allGather(r.f.s, b.amax, b.amax_all, 16);
            } else if (r.f.comm) |cm| {
                try cm.allGather(b.amax, b.amax_all, 4, .f32, r.f.s.handle);
            } else {
                try o.copy(b.amax_all, b.amax, 16);
            }
            try o.pickRank(b.amax_all, if (r.f.comm != null or r.f.fast != null) world else 1, 1, b.argmax);
        }
        return r.pickFirst(b, L0, s);
    }

    /// Runner._mtp_prompt: MTP rows for positions a - 1 .. e - 2 (hidden q, token q + 1); row e - 1 waits (carry).
    /// Served chunks run the MTP layer over all their rows at once (Forward.wide past MAX_ROWS rows, NCCL's ring
    /// all-reduce past b.small).
    fn mtpPrompt(r: *Runner, st: *const fw.State, prompt_ids: []const u32, a: usize, e: usize, T: ?usize) !void {
        const D = r.w.config.hidden;
        const o = r.ops();
        const b = &r.pb;
        const R = e - a;
        if (r.set.hid_normed) {
            try r.f.targetHidden(b, 0, R, b.ht);
        } else {
            try o.copy(b.ht, b.hidden, R * D * 2);
        }
        const first = a == 0;
        const rows = if (first) R - 1 else R;
        if (rows > 0) {
            if (first) {
                try o.copy(b.hin, b.ht, rows * D * 2);
            } else {
                try o.copy(b.hin, r.carry.ptr, D * 2);
                if (rows > 1) try o.copy(b.hin + D * 2, b.ht, (rows - 1) * D * 2);
            }
            if (r.set.served) {
                const from: usize = if (first) a + 1 else a;
                try o.copy(b.ids, r.ids_dev.?.ptr + from * 8, rows * 8);
            } else {
                try r.f.s.synchronize(); // the pinned ids staging held this chunk's ids until its copy ran
                try r.uploadIds(b.ids, if (first) prompt_ids[a + 1 .. e] else prompt_ids[a..e]);
            }
            try r.setI32(st.mpos, if (first) 0 else a - 1);
            const p0 = try r.f.pbegin();
            if (r.f.prof) |p| p.suspended = true;
            defer if (r.f.prof) |p| {
                p.suspended = false;
            };
            try r.f.mtpCompute(b, st, rows, T, .{ .head = false, .zero_first = first, .chain_normed = r.set.chain_normed });
            if (r.f.prof) |p| p.suspended = false;
            try r.f.pend(p0, .mtp);
        }
        try o.copy(r.carry.ptr, b.ht + (R - 1) * D * 2, D * 2);
    }

    /// runner.healthy: every rank agrees no RoCE runtime timed out (after a window's picks are read, before a token
    /// leaves); nothing without RoCE. Synchronizes.
    pub fn healthy(r: *Runner) !void {
        const fast = r.f.fast orelse return;
        if (!r.set.roce_health) return;
        const hw = r.health orelse return;
        const comm = r.f.comm orelse return;
        try fast.healthyEverywhere(comm, r.f.s, hw.ptr, hw.ptr + 4);
    }

    // ---------------------------------------------------------------------------------------------- sampling ---

    /// Runner._sample_local for rows 0..R-1 of `b`'s local logits at positions pos0 + i: each rank's top candidates
    /// of its share, one all-gather, the keyed draw on the host. Synchronizes. `wish`: this rank's stop wish, one
    /// word more on the gather (StopVote; only rank 0's is read); returns rank 0's (null on one rank).
    fn sampleRows(r: *Runner, b: *const fw.Buffers, R: usize, pos0: usize, s: smp.Sampling, picks: []u32, wish: bool) !?bool {
        const wish_bits: u32 = if (wish and r.w.rank == 0) @bitCast(@as(f32, 1.0)) else 0;
        const V = r.w.vocab_part;
        const world = r.w.world;
        const cnt = @min(s.candidates(), V);
        const lay = r.pinLayout();
        const o = r.ops();
        if (R > r.vb.head_rows and R > 1) return error.WindowTooWide;
        if (r.set.device_cands and b.cand_cols != 0 and R <= b.head_rows and cnt <= smp.max_candidates) {
            // each rank's top `cnt` of its share on the device (Runner._sample_local's torch.topk; selectTop's set:
            // `choose` sorts the gathered candidates itself), one gather, one read
            const words = R * 2 * cnt + 1;
            if (words > b.cand_words) return error.Invalid;
            try o.topkColumns(b.lg, V, V, R, cnt, true, b.tsel, b.cand_cols, cnt);
            try o.cands(b.lg, V, b.cand_cols, cnt, r.w.vocab_off, b.cand_send, R);
            try o.fill32(b.cand_send + (words - 1) * 4, wish_bits, 1); // the stop vote's spare word
            const stride = try r.f.smallGather(b.cand_send, b.cand_all, words);
            const ranks = if (r.f.comm != null) world else 1;
            const all = r.pinSlice(f32, lay.all, ranks * stride);
            try o.download(std.mem.sliceAsBytes(all), b.cand_all);
            try r.f.s.synchronize();
            r.pickRows(all, ranks, stride, R, cnt, pos0, s, picks);
            return if (ranks > 1) all[words - 1] > 0 else null;
        }
        const lg = r.pinSlice(f32, lay.lg, R * V);
        try o.download(std.mem.sliceAsBytes(lg), b.lg); // head "local": [R, V] rows, contiguous
        try r.f.s.synchronize();
        const words = R * 2 * cnt + 1;
        if (words > b.cand_words or words > r.vb.cand_words) return error.Invalid;
        const send = r.pinSlice(f32, lay.send, words);
        for (0..R) |i| {
            const row = send[i * 2 * cnt ..][0 .. 2 * cnt];
            const ids: []i32 = @ptrCast(row[cnt..]);
            smp.selectTop(lg[i * V ..][0..V], r.w.vocab_off, cnt, row[0..cnt], ids);
        }
        send[words - 1] = @bitCast(wish_bits); // the stop vote's spare word
        try o.upload(b.cand_send, std.mem.sliceAsBytes(send));
        // fused.small_gather: over the RoCE gather when it is up (rows padded to 16-byte packs), else NCCL
        const stride = try r.f.smallGather(b.cand_send, b.cand_all, words);
        const ranks = if (r.f.comm != null) world else 1;
        const all = r.pinSlice(f32, lay.all, ranks * stride);
        try o.download(std.mem.sliceAsBytes(all), b.cand_all);
        try r.f.s.synchronize();
        r.pickRows(all, ranks, stride, R, cnt, pos0, s, picks);
        return if (ranks > 1) all[words - 1] > 0 else null;
    }

    /// The keyed draws of rows 0..R-1 from every rank's gathered candidates (`all`: [ranks, stride] words, a row's
    /// [values (cnt) ; ids (cnt)] at i * 2 cnt).
    fn pickRows(r: *Runner, all: []const f32, ranks: usize, stride: usize, R: usize, cnt: usize, pos0: usize, s: smp.Sampling, picks: []u32) void {
        for (0..R) |i| {
            var n: usize = 0;
            for (0..ranks) |k| {
                const row = all[k * stride + i * 2 * cnt ..][0 .. 2 * cnt];
                const ids: []const i32 = @ptrCast(row[cnt..]);
                for (0..cnt) |q| {
                    r.cands[n] = .{ .v = row[q], .id = ids[q] };
                    n += 1;
                }
            }
            picks[i] = smp.choose(r.cands[0..n], pos0 + i, s);
        }
    }

    // ------------------------------------------------------------------------------------------------ decode ---

    /// Runner.generate (MTP mode normed/normed, no copies, no depth policy): `max_tokens` picks after the prompt
    /// (EOS is not a stop), `k` drafts a round (0: serial; at most Settings.k), the caches zeroed first (Phase 3a's
    /// reference schedule). Tokens go to `out`.
    pub fn generate(r: *Runner, prompt_ids: []const u32, max_tokens: usize, s: ?smp.Sampling, k_req: usize, out: *std.ArrayList(u32)) !Stats {
        return r.run(prompt_ids, .{ .max_tokens = max_tokens, .sampling = s, .k = k_req, .reset = true }, out);
    }

    /// StopVote (--parallel 1): rank 0's wish (its hook asked to stop: the client has gone, a stop string) ends the
    /// run on every rank after the same round, read from the verify window's own exchange.
    const Vote = struct {
        mine: bool = false, // this rank's hook asked to stop (sticky)
        armed: bool = false, // the argmax head's spare column holds it (sticky)
        agreed: bool = false, // rank 0 asked, as of the last verify window read (sticky)

        fn settle(v: *Vote, voted: ?bool) bool {
            v.agreed = v.agreed or (if (voted) |x| x else v.mine);
            return v.agreed;
        }
    };

    fn isEos(eos: []const u32, t: u32) bool {
        return std.mem.indexOfScalar(u32, eos, t) != null;
    }

    /// Runner.generate (MTP mode normed/normed, no depth policy) for one request: the prompt from `g.begin` (or a
    /// replayed head), then drafted rounds until `g.max_tokens` picks, an EOS of `g.eos`, or rank 0's stop wish heard
    /// by every rank. Tokens go to `out` and, round by round, to `g.hooks`. Drafts: the MTP head (`g.mode` mtp, `g.k`
    /// steps; 0: serial), DSpark or DFlash2 (Runner._generate_dflash), and with `g.copies` copy drafts ahead of them.
    pub fn run(r: *Runner, prompt_ids: []const u32, g: Gen, out: *std.ArrayList(u32)) !Stats {
        const mode = g.mode;
        const k = if (mode == .mtp) @min(g.k, r.set.k) else 0;
        const drafter: ?*draft.Drafter = switch (mode) {
            .mtp => null,
            .dspark => r.dspark orelse return error.NoDrafter,
            .dflash => r.dflash orelse return error.NoDrafter,
        };
        const which: usize = if (mode == .dspark) 1 else 0;
        const D = r.w.config.hidden;
        const o = r.ops();
        const s = g.sampling;
        const max_tokens = g.max_tokens;
        var st: Stats = .{ .prompt_len = prompt_ids.len, .begin = g.begin, .mode = mode };
        if (prompt_ids.len == 0 or max_tokens == 0 or prompt_ids.len + max_tokens + k + 1 > r.capacity) return error.ContextTooSmall;
        if (g.replay != null and g.begin != prompt_ids.len) return error.Invalid; // a replay resumes at its prompt's end
        if (g.reset and g.begin > 0) return error.Invalid;
        const cap0 = r.capture_s;
        if (g.reset) try r.resetState();
        const t0 = r.now();
        const keep: ?Keep = if (g.hooks) |h| (if (h.keep != null) Keep{ .hooks = h } else null) else null;
        if (g.replay != null) try r.resetDrafters(prompt_ids.len);
        var tok = if (g.replay) |lg| try r.replayHead(lg, prompt_ids.len, s) else try r.prefill(prompt_ids, s, g.begin, g.cuts, keep);
        try r.healthy(); // short prompts' chunks reduce over RoCE too
        st.prefill_s = r.since(t0);
        out.clearRetainingCapacity();
        try out.append(r.gpa, tok);
        // the prompt's copy index (drafted runs only: the serial reference never copies)
        var cx: ?copies.CopyIndex = null;
        defer if (cx) |*c| c.deinit();
        if (g.copies and r.copy_on and (mode != .mtp or k > 0)) {
            cx = try copies.CopyIndex.init(r.gpa, prompt_ids, r.set.copy.match, r.set.copy.most);
            try cx.?.extend(&.{tok});
        }
        var vote: Vote = .{};
        // vb.amax[:, 2].zero_(): a previous run's vote (the head never writes the column)
        for (0..r.vb.head_rows) |i| try o.fill32(r.vb.amax + (i * 4 + 2) * 4, 0, 1);
        if (g.hooks) |h| {
            if (h.tokens(h.ctx, out.items[0..1])) vote.mine = true;
        }
        var P = prompt_ids.len;
        var m: usize = 1; // MTP rows pending: (carry = hidden P - 1, tok) at P - 1
        if (k > 0) {
            const mb = &r.mb.?;
            try o.copy(mb.hin, r.carry.ptr, D * 2);
            try r.setId(mb.ids, tok);
        }
        const depth_max: usize = if (drafter) |dr| r.depthMax(dr) else 0;
        const lay = r.pinLayout();
        var picks: [fw.decode_rows]u32 = undefined;
        var drafts: [fw.decode_rows]u32 = undefined;
        var window: [fw.decode_rows]u32 = undefined;
        var done = out.items.len >= max_tokens or isEos(g.eos, tok);
        st.eos = isEos(g.eos, tok);
        const multi = r.w.world > 1 and (r.f.comm != null or r.f.fast != null);
        const t1 = r.now();
        while (!done) {
            st.rounds += 1;
            const left = max_tokens - out.items.len - 1;
            var nd: usize = 0; // drafts this round
            var on_device = false; // the MTP steps wrote them into vb.ids
            var was_copied = false;
            if (cx) |*c| {
                nd = r.copied(c, left, P, &drafts).len;
                was_copied = nd > 0;
            }
            if (!was_copied) {
                if (drafter) |dr| {
                    const room = @min(depth_max, @min(left, r.capacity - P - 1));
                    if (room > 0) nd = (try r.propose(dr, which, tok, room, s, &drafts)).len;
                } else {
                    const room = @min(k, @min(left, r.capacity - P - 1));
                    try r.setId(r.vb.ids, tok);
                    if (room > 0) {
                        const Tm = bucketOf(P + room, r.topk);
                        const ru: u32 = if (Tm != null and P - 1 >= r.topk) r.reuse else 0;
                        try r.mtp(m, 1, P - m, Tm, 0);
                        var j: usize = 2;
                        while (j <= room) : (j += 1) try r.mtp(1, j, P + j - 2, Tm, ru);
                    }
                    nd = room;
                    on_device = true;
                }
            }
            const R = nd + 1;
            if (!on_device) { // vb.ids[:R] = [tok] + drafts
                window[0] = tok;
                @memcpy(window[1..R], drafts[0..nd]);
                try r.uploadVerify(window[0..R]);
            }
            const pick: Pick = if (s == null) .argmax else .local;
            // Runner._vote_arm: rank 0's wish into the argmax head's spare column (once: the head never writes it)
            if (pick == .argmax and multi and r.w.rank == 0 and vote.mine and !vote.armed) {
                for (0..r.vb.head_rows) |i| try o.fill32(r.vb.amax + (i * 4 + 2) * 4, @bitCast(@as(f32, 1.0)), 1);
                vote.armed = true;
            }
            try r.verify(R, P, bucketOf(P + R, r.topk), pick);
            // the window's drafts (vb.ids[1:R], when the MTP steps made them) and, greedy, its picks and the vote word,
            // in one read
            const rd = r.pinSlice(i64, lay.rd, 2 * fw.decode_rows);
            if (on_device and R > 1) try o.download(std.mem.sliceAsBytes(rd[0 .. R - 1]), r.vb.ids + 8);
            var voted: ?bool = null;
            if (s) |smpl| {
                voted = try r.sampleRows(&r.vb, R, P + 1, smpl, picks[0..R], vote.mine);
            } else {
                try o.download(std.mem.sliceAsBytes(rd[fw.decode_rows..][0..R]), r.vb.argmax);
                const word = r.pinSlice(f32, lay.vote, 1);
                if (multi) try o.download(std.mem.sliceAsBytes(word), r.vb.amax_all + 2 * 4); // rank 0's row 0, column 2
                try r.f.s.synchronize();
                for (0..R) |i| picks[i] = @intCast(rd[fw.decode_rows + i]);
                if (multi) voted = word[0] > 0;
            }
            if (on_device) { // both branches synchronized after the drafts' copy was queued
                for (0..R - 1) |i| drafts[i] = @intCast(rd[i]);
            }
            try r.healthy(); // every rank: no RoCE timeout behind these picks
            var n: usize = 0;
            while (n < R - 1 and drafts[n] == picks[n]) n += 1;
            const emit = picks[0 .. n + 1];
            st.drafted += R - 1;
            st.accepted += n;
            if (was_copied) {
                st.copy_rounds += 1;
                st.copy_drafted += R - 1;
                st.copy_accepted += n;
            }
            // the drafters' contexts: every round's kept rows (Runner._add_taps; drafter modes only, as Python's)
            if (drafter != null) try r.feedDrafters(&r.vb, 0, n + 1, true);
            const was = out.items.len;
            for (emit) |e| {
                try out.append(r.gpa, e);
                if (out.items.len >= max_tokens) {
                    done = true;
                    break;
                }
                if (isEos(g.eos, e)) {
                    done = true;
                    st.eos = true;
                    break;
                }
            }
            if (g.hooks) |h| {
                if (h.tokens(h.ctx, out.items[was..])) vote.mine = true;
            }
            if (cx) |*c| try c.extend(out.items[was..]);
            if (vote.settle(voted)) done = true; // rank 0 asked to stop: every rank ends after this round
            if (k > 0) {
                // next round's MTP rows: (target hidden P + i, token P + i + 1); a copied round's backlog grows (its
                // MTP step did not run), written first when it would outgrow the MTP window
                const mb = &r.mb.?;
                if (!was_copied) {
                    m = 0;
                } else if (m + n + 1 > mb.rows) {
                    try r.mtpFlush(m, P);
                    m = 0;
                }
                if (r.set.hid_normed) {
                    try r.f.targetHidden(&r.vb, 0, n + 1, mb.hin + m * D * 2);
                } else {
                    try o.copy(mb.hin + m * D * 2, r.vb.hidden, (n + 1) * D * 2);
                }
                try r.uploadIds(mb.ids + m * 8, emit);
                m += n + 1;
            }
            P += n + 1;
            tok = emit[emit.len - 1];
        }
        try r.f.s.synchronize();
        st.stopped = vote.agreed;
        st.decode_s = r.since(t1);
        st.capture_s = r.capture_s - cap0;
        st.graphs = r.graphs.count();
        return st;
    }
};

test "even chunks as prefixes.even_chunks" {
    const gpa = std.testing.allocator;
    const a = try evenChunks(gpa, 0, 4500, 128, 128);
    defer gpa.free(a);
    try std.testing.expectEqual(@as(usize, 36), a.len); // ceil(4500 / 128) = 36 chunks of ceil(4500 / 36) = 125
    try std.testing.expectEqual([2]usize{ 0, 125 }, a[0]);
    try std.testing.expectEqual([2]usize{ 4375, 4500 }, a[35]);
    const b = try evenChunks(gpa, 0, 300, 128, 128);
    defer gpa.free(b);
    try std.testing.expectEqual(@as(usize, 3), b.len);
    try std.testing.expectEqual([2]usize{ 200, 300 }, b[2]);
    const c = try evenChunks(gpa, 0, 20, 128, 128);
    defer gpa.free(c);
    try std.testing.expectEqual(@as(usize, 1), c.len);
}

test "cut chunks as prefixes.cut_chunks" {
    const gpa = std.testing.allocator;
    // a resumed prompt at 5000 runs the chunks a fresh one cut at 5000 runs past it
    const fresh = try cutChunks(gpa, 0, 9000, &.{ 5000, 300, 5000, 9000 }, 4096, 4096);
    defer gpa.free(fresh);
    const resumed = try cutChunks(gpa, 5000, 9000, &.{ 5000, 300, 9000 }, 4096, 4096);
    defer gpa.free(resumed);
    try std.testing.expectEqual([2]usize{ 0, 300 }, fresh[0]);
    try std.testing.expectEqual([2]usize{ 300, 2650 }, fresh[1]); // 4700 rows in two equal chunks
    try std.testing.expectEqual(fresh.len - 3, resumed.len);
    for (resumed, fresh[3..]) |x, y| try std.testing.expectEqual(y, x);
    try std.testing.expectEqual([2]usize{ 5000, 9000 }, resumed[0]);
}

test "graph keys: buckets get distinct indices" {
    try std.testing.expect(tIndex(null) != tIndex(4096));
    try std.testing.expect(tIndex(4096) != tIndex(8192));
}
