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
//! Not here (Phase 3b+): copy drafts, DFlash2, the depth policy, prompt reuse, stop votes, streaming (EOS is not a
//! stop: every request makes `max_tokens` picks), concurrency, DCP.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const smp = @import("sampling.zig");
const W = @import("cuda_weights.zig");
const prompt = @import("cuda_prompt.zig");

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
    cands: []smp.Cand,
    // Phase 3a (Settings.served): the second micro-batch's buffers, the sequence-parallel chunk, the prompt ids on
    // the device (uploaded once a request), the RoCE health words
    pb1: ?fw.Buffers = null,
    sp: ?*prompt.Sp = null,
    ids_dev: ?cuda.DeviceBuffer = null, // int64 [capacity]
    health: ?cuda.DeviceBuffer = null, // int32 [1 + world]

    fn pinLayout(r: *const Runner) struct { lg: usize, send: usize, all: usize, ids: usize, rd: usize, total: usize } {
        const V = r.w.vocab_part;
        const hr = r.vb.head_rows;
        const lg_b = std.mem.alignForward(usize, @max(hr, 1) * V * 4, 256);
        const words = std.mem.alignForward(usize, r.vb.cand_words, 4);
        const send_b = std.mem.alignForward(usize, words * 4, 256);
        const all_b = std.mem.alignForward(usize, r.w.world * words * 4, 256);
        const id_rows = if (r.set.served) r.capacity else @max(r.set.window, 64);
        const ids_b = std.mem.alignForward(usize, id_rows * 8, 256);
        const rd_b: usize = 2 * fw.decode_rows * 8;
        return .{ .lg = 0, .send = lg_b, .all = lg_b + send_b, .ids = lg_b + send_b + all_b, .rd = lg_b + send_b + all_b + ids_b, .total = lg_b + send_b + all_b + ids_b + rd_b };
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, f: fw.Forward, capacity: usize, set: Settings) !Runner {
        const w = f.w;
        const c = w.config;
        if (set.k > 0 and w.mtp == null) return error.NoMtpLayer;
        if (set.window == 0 or set.window > fw.max_rows) return error.WindowTooWide;
        if (set.served and (set.prompt_rows == 0 or set.prompt_rows > fw.max_prompt_rows or set.prompt_rows_short == 0)) return error.WindowTooWide;
        const topk = c.index_topk;
        const cols = bucketOf(capacity, topk) orelse 1;
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
        r.pb1 = null;
        r.sp = null;
        r.ids_dev = null;
        r.health = null;
        const shared = f.shared_experts;
        r.st = try fw.State.init(gpa, d, w, capacity);
        errdefer r.st.deinit();
        if (set.served) {
            // Runner.__init__: pb (prompt_rows, exact key range), pb1 (the second micro-batch) when chunks overlap
            r.pb = try fw.Buffers.initWith(gpa, d, w, set.prompt_rows, @max(capacity, 1), .{ .mtp = set.k > 0, .prompt = true, .shared_experts = shared, .mtp_dense = set.k > 0 and f.mtp_dense != null });
        } else {
            r.pb = try fw.Buffers.initWith(gpa, d, w, set.window, @max(capacity, 1), .{ .mtp = set.k > 0 });
        }
        errdefer r.pb.deinit();
        if (set.served and set.prompt_sp and set.sp != null and w.world > 1) {
            r.pb1 = try fw.Buffers.initWith(gpa, d, w, set.prompt_rows - set.prompt_rows / 2, @max(capacity, 1), .{ .prompt = true, .shared_experts = shared });
            r.sp = set.sp;
        }
        errdefer if (r.pb1) |*b| b.deinit();
        r.vb = try fw.Buffers.initWith(gpa, d, w, set.k + 1, cols, .{ .decode = true, .shared_experts = shared });
        errdefer r.vb.deinit();
        r.mb = null;
        if (set.k > 0) r.mb = try fw.Buffers.initWith(gpa, d, w, set.k + 1, cols, .{ .decode = true, .mtp = true, .shared_experts = shared });
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
    fn uploadIds(r: *Runner, dst: u64, ids: []const u32) !void {
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
    fn runGraph(r: *Runner, key: u64, ctx: anytype) !void {
        if (r.graphs.get(key)) |e| return e.launchOn(r.f.s);
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
        if (r.set.served) {
            // Runner.prewarm's first step: a prompt of a full chunk, a remainder and rows past index_topk
            // (1000 + 7919 i mod 150000), so the prompt path's first use (cuBLAS's heuristics, NCCL's buffers for
            // the prompt sizes, the unpack cache) is not inside a measured request, as in the Python engine
            const n = @min(r.capacity - 1, r.set.prompt_rows + r.topk + 301);
            const ids = try r.gpa.alloc(u32, n);
            defer r.gpa.free(ids);
            for (ids, 0..) |*v, i| v.* = @intCast(1000 + (i * 7919) % 150000);
            _ = try r.prefill(ids, null);
            try r.healthy();
        }
        var bs: [24]?usize = undefined;
        for (r.buckets(&bs)) |T| {
            for (1..r.set.k + 2) |R| {
                try r.verify(R, 0, T, .argmax);
                try r.verify(R, 0, T, .local);
            }
            if (r.set.k > 0) {
                for (1..r.set.k + 2) |m| try r.mtp(m, 1, 0, T, 0);
                var j: usize = 2;
                while (j <= r.set.k) : (j += 1) {
                    try r.mtp(1, j, 0, T, 0);
                    if (r.reuse != 0 and T != null) try r.mtp(1, j, 0, T, r.reuse);
                }
            }
        }
        try r.f.s.synchronize();
        try r.resetState();
        return r.graphs.count();
    }

    // ------------------------------------------------------------------------------------------------ prompt ---

    /// Runner.prefill + _mtp_prompt: the prompt's chunks through the target (and, when the runner drafts, the MTP
    /// layer's rows a - 1 .. e - 2); returns the first pick (position L0).
    /// Phase 2b: equal chunks of `window` rows. Served (Phase 3a): prefixes.even_chunks with PROMPT_ROWS /
    /// PROMPT_ROWS_SHORT; a chunk of at least 2 MAX_ROWS + 2 rows that sp_fits takes compute_prompt_sp (two micro-
    /// batches, the second's last row holds the head), any other chunk fused.compute; the prompt's ids go to the
    /// device once, and each chunk takes its rows there (no host step between chunks).
    fn prefill(r: *Runner, prompt_ids: []const u32, s: ?smp.Sampling) !u32 {
        const L0 = prompt_ids.len;
        const D = r.w.config.hidden;
        const rows = if (r.set.served) r.set.prompt_rows else r.set.window;
        const short = if (r.set.served) r.set.prompt_rows_short else r.set.window;
        const chunks = try evenChunks(r.gpa, 0, L0, rows, short);
        defer r.gpa.free(chunks);
        const head: fw.HeadMode = if (s == null) .argmax else .local;
        var hb: *const fw.Buffers = &r.pb; // the buffers whose head picked the first token
        if (r.set.served) {
            const ids = r.ids_dev orelse return error.Invalid;
            try r.uploadIds(ids.ptr, prompt_ids);
        }
        for (chunks, 0..) |ch, ci| {
            const a = ch[0];
            const e = ch[1];
            const R = e - a;
            const t_chunk = r.now();
            const T: ?usize = if (e > r.topk) e else null;
            const last = e == L0;
            if (r.set.served) {
                try r.ops().copy(r.pb.ids, r.ids_dev.?.ptr + a * 8, R * 8);
            } else {
                try r.uploadIds(r.pb.ids, prompt_ids[a..e]);
            }
            try r.setI32(r.st.pos, a);
            const sp_ok = if (r.sp != null and r.pb1 != null) R >= prompt.min_overlap_rows and prompt.spFits(r.w.world, r.pb.rows, r.pb1.?.rows, R) else false;
            if (sp_ok) {
                const b1 = &r.pb1.?;
                try r.sp.?.chunk(&r.st, &r.pb, b1, a, R, T);
                if (last) {
                    const hR = R / 2;
                    const p0 = try r.f.pbegin();
                    try r.f.headDevice(b1, b1.x + (R - hR - 1) * D * 2, r.w.final_norm, 1, head);
                    try r.f.pend(p0, .head);
                    hb = b1;
                }
            } else {
                try r.f.compute(&r.pb, &r.st, R, T, if (last) head else .none, true);
                if (last) hb = &r.pb;
            }
            if (r.set.k > 0) try r.mtpPrompt(prompt_ids, a, e, T);
            if (!r.set.served) try r.f.s.synchronize(); // the pinned ids are reused by the next chunk
            if (r.f.prof) |p| {
                if (p.on) { // --prompt-prof: this chunk's phases (a sync a chunk: profiling runs only)
                    try r.f.s.synchronize();
                    if (r.sp) |sp| try sp.side.s.synchronize();
                    try p.report(r.w.rank, ci, a, e, r.since(t_chunk));
                }
            }
        }
        if (r.f.prof) |p| {
            p.total(r.w.rank, L0);
            p.on = false; // the prompt only: decode windows are never timed (nor captured with events in them)
        }
        if (s) |smpl| {
            var pick: [1]u32 = undefined;
            try r.sampleRows(hb, 1, L0, smpl, &pick);
            return pick[0];
        }
        const lay = r.pinLayout();
        const got = r.pinSlice(i64, lay.rd, 1);
        try r.ops().download(std.mem.sliceAsBytes(got), hb.argmax);
        try r.f.s.synchronize();
        return @intCast(got[0]);
    }

    /// Runner._mtp_prompt: MTP rows for positions a - 1 .. e - 2 (hidden q, token q + 1); row e - 1 waits (carry).
    /// Served chunks run the MTP layer over all their rows at once (Forward.wide past MAX_ROWS rows, NCCL's ring
    /// all-reduce past b.small).
    fn mtpPrompt(r: *Runner, prompt_ids: []const u32, a: usize, e: usize, T: ?usize) !void {
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
            try r.setI32(r.st.mpos, if (first) 0 else a - 1);
            const p0 = try r.f.pbegin();
            if (r.f.prof) |p| p.suspended = true;
            defer if (r.f.prof) |p| {
                p.suspended = false;
            };
            try r.f.mtpCompute(b, &r.st, rows, T, .{ .head = false, .zero_first = first, .chain_normed = r.set.chain_normed });
            if (r.f.prof) |p| p.suspended = false;
            try r.f.pend(p0, .mtp);
        }
        try o.copy(r.carry.ptr, b.ht + (R - 1) * D * 2, D * 2);
    }

    /// runner.healthy: every rank agrees no RoCE runtime timed out (after a window's picks are read, before a token
    /// leaves); nothing without RoCE. Synchronizes.
    fn healthy(r: *Runner) !void {
        const fast = r.f.fast orelse return;
        if (!r.set.roce_health) return;
        const hw = r.health orelse return;
        const comm = r.f.comm orelse return;
        try fast.healthyEverywhere(comm, r.f.s, hw.ptr, hw.ptr + 4);
    }

    // ---------------------------------------------------------------------------------------------- sampling ---

    /// Runner._sample_local for rows 0..R-1 of `b`'s local logits at positions pos0 + i: each rank's top candidates
    /// of its share, one all-gather, the keyed draw on the host. Synchronizes.
    fn sampleRows(r: *Runner, b: *const fw.Buffers, R: usize, pos0: usize, s: smp.Sampling, picks: []u32) !void {
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
            try o.fill32(b.cand_send + (words - 1) * 4, 0, 1); // the stop vote's spare word (no stops here)
            const stride = try r.f.smallGather(b.cand_send, b.cand_all, words);
            const ranks = if (r.f.comm != null) world else 1;
            const all = r.pinSlice(f32, lay.all, ranks * stride);
            try o.download(std.mem.sliceAsBytes(all), b.cand_all);
            try r.f.s.synchronize();
            r.pickRows(all, ranks, stride, R, cnt, pos0, s, picks);
            return;
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
        send[words - 1] = 0; // the stop vote's spare word (no stops here)
        try o.upload(b.cand_send, std.mem.sliceAsBytes(send));
        // fused.small_gather: over the RoCE gather when it is up (rows padded to 16-byte packs), else NCCL
        const stride = try r.f.smallGather(b.cand_send, b.cand_all, words);
        const ranks = if (r.f.comm != null) world else 1;
        const all = r.pinSlice(f32, lay.all, ranks * stride);
        try o.download(std.mem.sliceAsBytes(all), b.cand_all);
        try r.f.s.synchronize();
        r.pickRows(all, ranks, stride, R, cnt, pos0, s, picks);
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
    /// (EOS is not a stop), `k` drafts a round (0: serial; at most Settings.k). Tokens go to `out`.
    pub fn generate(r: *Runner, prompt_ids: []const u32, max_tokens: usize, s: ?smp.Sampling, k_req: usize, out: *std.ArrayList(u32)) !Stats {
        const k = @min(k_req, r.set.k);
        const D = r.w.config.hidden;
        const o = r.ops();
        var st: Stats = .{ .prompt_len = prompt_ids.len };
        if (prompt_ids.len == 0 or prompt_ids.len + max_tokens + k + 1 > r.capacity) return error.ContextTooSmall;
        const cap0 = r.capture_s;
        try r.resetState();
        const t0 = r.now();
        var tok = try r.prefill(prompt_ids, s);
        try r.healthy(); // short prompts' chunks reduce over RoCE too
        st.prefill_s = r.since(t0);
        out.clearRetainingCapacity();
        try out.append(r.gpa, tok);
        var P = prompt_ids.len;
        var m: usize = 1; // MTP rows pending: (carry = hidden P - 1, tok) at P - 1
        if (k > 0) {
            const mb = &r.mb.?;
            try o.copy(mb.hin, r.carry.ptr, D * 2);
            try r.setId(mb.ids, tok);
        }
        const lay = r.pinLayout();
        var picks: [fw.decode_rows]u32 = undefined;
        var drafts: [fw.decode_rows]u32 = undefined;
        const t1 = r.now();
        while (out.items.len < max_tokens) {
            st.rounds += 1;
            const room = @min(k, @min(max_tokens - out.items.len - 1, r.capacity - P - 1));
            try r.setId(r.vb.ids, tok);
            if (room > 0) {
                const Tm = bucketOf(P + room, r.topk);
                const ru: u32 = if (Tm != null and P - 1 >= r.topk) r.reuse else 0;
                try r.mtp(m, 1, P - m, Tm, 0);
                var j: usize = 2;
                while (j <= room) : (j += 1) try r.mtp(1, j, P + j - 2, Tm, ru);
            }
            const R = room + 1;
            const pick: Pick = if (s == null) .argmax else .local;
            try r.verify(R, P, bucketOf(P + R, r.topk), pick);
            // the window's drafts (vb.ids[1:R]) and, greedy, its picks, in one read
            const rd = r.pinSlice(i64, lay.rd, 2 * fw.decode_rows);
            if (R > 1) try o.download(std.mem.sliceAsBytes(rd[0 .. R - 1]), r.vb.ids + 8);
            if (s) |smpl| {
                try r.sampleRows(&r.vb, R, P + 1, smpl, picks[0..R]);
            } else {
                try o.download(std.mem.sliceAsBytes(rd[fw.decode_rows..][0..R]), r.vb.argmax);
                try r.f.s.synchronize();
                for (0..R) |i| picks[i] = @intCast(rd[fw.decode_rows + i]);
            }
            for (0..R - 1) |i| drafts[i] = @intCast(rd[i]);
            try r.healthy(); // every rank: no RoCE timeout behind these picks
            var n: usize = 0;
            while (n < R - 1 and drafts[n] == picks[n]) n += 1;
            const emit = picks[0 .. n + 1];
            st.drafted += R - 1;
            st.accepted += n;
            for (emit) |e| {
                if (out.items.len >= max_tokens) break;
                try out.append(r.gpa, e);
            }
            if (k > 0) {
                // next round's MTP rows: (target hidden P + i, token P + i + 1)
                const mb = &r.mb.?;
                m = 0;
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

test "graph keys: buckets get distinct indices" {
    try std.testing.expect(tIndex(null) != tIndex(4096));
    try std.testing.expect(tIndex(4096) != tIndex(8192));
}
