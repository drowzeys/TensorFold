//! As-served scoring on the device (score.zig holds the formats and the math): fixed token rows through the served
//! engine's own prompt path, teacher-forced, no drafts, the head on EVERY row instead of the last one alone.
//!
//! A row runs as a served prompt runs: the Runner's prefillChunk over the chunks the server would cut (or chunks of
//! Options.chunk_rows rows: up to fused.FAST_ROWS = 16 rows a chunk takes the decode windows' kernels, so the rows
//! are computed as generated tokens are). After a chunk its final hidden rows are in the prompt buffers
//! (pb.hidden[:R]); the scorer runs fused.head's own two launches on them - the final RMSNorm and glue.router over
//! this rank's lm_head share - in its own scratch, `head_rows` rows a pass, copies the logits to the host and reduces
//! each row to score.Part. Both kernels compute each row alone, so a row's logits are the bits the served head gives
//! that row. Nothing is exchanged between the ranks: every rank writes its own partials file and score.combine adds
//! them up.
//!
//! The engine must be booted with no drafts (k 0, no drafter): the prompt chunks then feed nothing else. Nothing
//! here is reached by the server.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const fw = @import("cuda_forward.zig");
const rn = @import("cuda_runner.zig");
const sc = @import("score.zig");

pub const Options = struct {
    /// rows a head pass (their logits are copied to the host together: head_rows * vocab_part * 4 bytes)
    head_rows: usize = 64,
    /// rows a prompt chunk; 0: the served prompt path's chunks (prefixes.even_chunks of prompt_rows / _short)
    chunk_rows: usize = 0,
    /// rows whose positions' largest local logits are kept too (score.DumpRec, in Scorer.dump): debugging
    dump_rows: []const u32 = &.{},
    /// every MoE layer's routed weights and picks (fw.RouteTap); needs chunk_rows 1..16, the windows fused.ffn_part routes
    route: ?*fw.RouteTap = null,
};

pub const Scorer = struct {
    gpa: std.mem.Allocator,
    r: *rn.Runner,
    o: Options,
    fnormed: cuda.DeviceBuffer, // bf16 [head_rows, D]
    lg: cuda.DeviceBuffer, // fp32 [head_rows, vocab_part]
    hpart: cuda.DeviceBuffer, // fp32 [ROUTER_KS, head_rows, vocab_part]
    host: cuda.HostBuffer, // fp32 [head_rows, vocab_part], page-locked
    /// seconds in the engine (chunks and head passes) and in the host sums, positions scored
    engine_s: f64 = 0,
    host_s: f64 = 0,
    positions: usize = 0,
    dump: std.ArrayList(u8) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *rn.Runner, o: Options) !Scorer {
        const w = r.w;
        if (w.lm_head == 0) return error.NoHead;
        if (r.set.k != 0 or r.dspark != null or r.dflash != null or r.f.taps.n != 0) return error.ScoreWithDrafts;
        if (!r.set.served or r.set.slots != 1) return error.Invalid;
        if (o.head_rows == 0 or o.head_rows > fw.max_rows) return error.WindowTooWide; // the router's widest block
        if (o.chunk_rows > r.pb.rows) return error.WindowTooWide;
        if (o.route != null and (o.chunk_rows == 0 or o.chunk_rows > 16)) return error.Invalid;
        r.f.route_tap = o.route;
        const D = w.config.hidden;
        const V = w.vocab_part;
        var fnormed = try cuda.DeviceBuffer.alloc(d, o.head_rows * D * 2);
        errdefer fnormed.free();
        var lg = try cuda.DeviceBuffer.alloc(d, o.head_rows * V * 4);
        errdefer lg.free();
        var hpart = try cuda.DeviceBuffer.alloc(d, tri.router_ks * o.head_rows * V * 4);
        errdefer hpart.free();
        var host = try cuda.HostBuffer.alloc(d, o.head_rows * V * 4);
        errdefer host.free();
        return .{ .gpa = gpa, .r = r, .o = o, .fnormed = fnormed, .lg = lg, .hpart = hpart, .host = host };
    }

    pub fn deinit(s: *Scorer) void {
        s.dump.deinit(s.gpa);
        s.host.free();
        s.hpart.free();
        s.lg.free();
        s.fnormed.free();
    }

    fn now(s: *const Scorer) i128 {
        return @intCast(std.Io.Clock.awake.now(s.r.io).toNanoseconds());
    }

    fn since(s: *const Scorer, t0: i128) f64 {
        return @as(f64, @floatFromInt(s.now() - t0)) / 1e9;
    }

    /// One row (`ri` in the file): `ids` its tokens, `ref` and `first_pos` where its positions' reference records
    /// are, `out` its len - 1 positions' shares (position t: the head row of token t against the reference for
    /// token t + 1).
    pub fn row(s: *Scorer, ri: usize, ids: []const u32, ref: *const sc.Ref, first_pos: u64, out: []sc.Part) !void {
        const r = s.r;
        const w = r.w;
        const c = w.config;
        const D = c.hidden;
        const V = w.vocab_part;
        const L = ids.len;
        if (L < 2 or out.len != L - 1) return error.Invalid;
        if (L > r.capacity) return error.ContextTooLong;
        const topk: usize = ref.head.topk;
        const o = kern.Ops{ .k = r.f.k, .s = r.f.s };
        const t_: tri.Tri = .{ .set = &r.f.k.triton, .s = r.f.s };
        const rows = if (s.o.chunk_rows > 0) s.o.chunk_rows else r.set.prompt_rows;
        const short = if (s.o.chunk_rows > 0) s.o.chunk_rows else r.set.prompt_rows_short;
        const chunks = try rn.evenChunks(s.gpa, 0, L, rows, short);
        defer s.gpa.free(chunks);
        try r.resetState(); // each request starts from zeroed caches
        try r.uploadIds(r.ids_dev.?.ptr, ids);
        const host = std.mem.bytesAsSlice(f32, s.host.bytes);
        const dumped = std.mem.indexOfScalar(u32, s.o.dump_rows, @intCast(ri)) != null;
        for (chunks, 0..) |ch, ci| {
            const a = ch[0];
            const e = ch[1];
            const R = e - a;
            var t0 = s.now();
            _ = try r.prefillChunk(&r.st, ids, a, e, ci, .none);
            var i: usize = 0;
            while (i < R) : (i += s.o.head_rows) {
                const n = @min(s.o.head_rows, R - i);
                // fused.head "local": the final norm, then this rank's logits over its vocabulary share
                try t_.rmsnorm(r.pb.hidden + i * D * 2, D, w.final_norm, s.fnormed.ptr, D, n, D, c.eps);
                try t_.router(s.fnormed.ptr, D, w.lm_head, s.lg.ptr, s.hpart.ptr, n, D, V);
                try o.download(s.host.bytes[0 .. n * V * 4], s.lg.ptr);
                try r.f.s.synchronize();
                s.engine_s += s.since(t0);
                t0 = s.now();
                for (0..n) |j| {
                    const p = a + i + j;
                    if (p + 1 >= L) break; // the last token predicts nothing the row holds
                    const pos = sc.Pos.of(ref.posRec(first_pos + p), topk);
                    out[p] = sc.part(host[j * V ..][0..V], w.vocab_off, pos, topk, ids[p + 1]);
                    if (dumped) {
                        const d = sc.topLocal(host[j * V ..][0..V], w.vocab_off, @intCast(ri), @intCast(p));
                        try s.dump.appendSlice(s.gpa, std.mem.asBytes(&d));
                    }
                }
                s.host_s += s.since(t0);
                t0 = s.now();
            }
            s.engine_s += s.since(t0);
        }
        s.positions += L - 1;
    }

    /// Every row of `ref` into a partials file's bytes (score.PartHead, then the positions' shares in order).
    /// `progress` is called after each row with its index.
    pub fn run(s: *Scorer, ref: *const sc.Ref, progress: ?*const fn (row: usize, of: usize, s: *const Scorer) void) ![]u8 {
        const h = ref.head;
        const w = s.r.w;
        if (!ref.hasPositions()) return error.NoReference;
        if (h.vocab != w.config.vocab) return error.ReferenceVocab;
        const n_pos: usize = @intCast(h.n_pos);
        const bytes = try s.gpa.alloc(u8, @sizeOf(sc.PartHead) + n_pos * @sizeOf(sc.Part));
        errdefer s.gpa.free(bytes);
        const ph: sc.PartHead = .{ .rank = @intCast(w.rank), .world = @intCast(w.world), .vocab_off = @intCast(w.vocab_off), .vocab_part = @intCast(w.vocab_part), .topk = h.topk, .n_pos = h.n_pos, .tokens_digest = h.tokens_digest, .body_digest = h.body_digest, .chunk_rows = @intCast(s.o.chunk_rows) };
        @memcpy(bytes[0..@sizeOf(sc.PartHead)], std.mem.asBytes(&ph));
        var parts: std.ArrayList(sc.Part) = .empty;
        defer parts.deinit(s.gpa);
        for (0..h.n_rows) |ri| {
            const rw = ref.row(ri);
            const ids = try ref.tokens(s.gpa, ri);
            defer s.gpa.free(ids);
            try parts.resize(s.gpa, rw.len - 1);
            try s.row(ri, ids, ref, rw.first_pos, parts.items);
            const off = @sizeOf(sc.PartHead) + @as(usize, @intCast(rw.first_pos)) * @sizeOf(sc.Part);
            @memcpy(bytes[off..][0 .. parts.items.len * @sizeOf(sc.Part)], std.mem.sliceAsBytes(parts.items));
            if (progress) |f| f(ri, h.n_rows, s);
        }
        return bytes;
    }
};
