//! As-served scoring on the device (score.zig holds the formats and the math): fixed token rows through the served
//! engine's own prompt path, teacher-forced, no drafts, the head on EVERY row instead of the last one alone.
//!
//! A row runs as a served prompt runs: the Runner's prefillChunk over the chunks the server would cut (or chunks of
//! Options.chunk_rows rows: up to fused.FAST_ROWS = 16 rows a chunk takes the decode windows' kernels, so the rows
//! are computed as generated tokens are). After a chunk its final hidden rows are in the prompt buffers
//! (pb.hidden[:R]); the scorer runs fused.head's own two launches on them - the final RMSNorm and glue.router over
//! this rank's lm_head share, or the quantized head's launch (TF_GLM53_HEAD_Q, fw.headLogits) - in its own scratch, `head_rows` rows a pass, copies the logits to the host and reduces
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
const headq = @import("headq.zig");

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
    hpart: cuda.DeviceBuffer, // fp32 [ROUTER_KS, head_rows, vocab_part]; a quantized head: [head_rows, blocks * 128]
    hxh: cuda.DeviceBuffer, // fp16 [head_rows, D]: a quantized head's rotated input (16 bytes without one)
    host: cuda.HostBuffer, // fp32 [head_rows, vocab_part], page-locked
    /// seconds in the engine (chunks and head passes) and in the host sums, positions scored
    engine_s: f64 = 0,
    host_s: f64 = 0,
    positions: usize = 0,
    dump: std.ArrayList(u8) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, r: *rn.Runner, o: Options) !Scorer {
        const w = r.w;
        if (!w.hasHead()) return error.NoHead;
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
        var hpart = try cuda.DeviceBuffer.alloc(d, o.head_rows * 4 * if (w.head_q) |hq| @max(hq.lin.n, V) else tri.router_ks * V);
        errdefer hpart.free();
        var hxh = try cuda.DeviceBuffer.alloc(d, if (w.head_q != null) o.head_rows * D * 2 else 16);
        errdefer hxh.free();
        var host = try cuda.HostBuffer.alloc(d, o.head_rows * V * 4);
        errdefer host.free();
        return .{ .gpa = gpa, .r = r, .o = o, .fnormed = fnormed, .lg = lg, .hpart = hpart, .hxh = hxh, .host = host };
    }

    pub fn deinit(s: *Scorer) void {
        s.dump.deinit(s.gpa);
        s.host.free();
        s.hxh.free();
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

    /// --score-time N (a probe, not a score): N launches each of the bf16 head over this rank's vocabulary share and
    /// of some EXL3 linears, at 1..3 rows, in milliseconds a launch (the stream synchronized around each batch).
    pub fn timeHead(s: *Scorer, reps: usize) !void {
        const r = s.r;
        const w = r.w;
        const D = w.config.hidden;
        const V = w.vocab_part;
        const t_: tri.Tri = .{ .set = &r.f.k.triton, .s = r.f.s };
        const o = kern.Ops{ .k = r.f.k, .s = r.f.s };
        try o.fill32(s.fnormed.ptr, 0x3c003c00, s.o.head_rows * D / 2); // bf16 ~0.0078 everywhere
        try o.fill32(r.pb.normed, 0x3c003c00, 4 * D / 2);
        for (1..4) |n| {
            for (0..10) |_| try fw.headLogits(w, t_, o, s.fnormed.ptr, s.lg.ptr, s.hpart.ptr, s.hxh.ptr, n);
            try r.f.s.synchronize();
            const t0 = s.now();
            for (0..reps) |_| try fw.headLogits(w, t_, o, s.fnormed.ptr, s.lg.ptr, s.hpart.ptr, s.hxh.ptr, n);
            try r.f.s.synchronize();
            const ms = s.since(t0) * 1e3 / @as(f64, @floatFromInt(reps));
            if (w.head_q) |hq| {
                std.debug.print("TIME rank {d} head exl3 [{d} x {d}] {d} half-bits ({d} MB, its share copied out) rows {d}: {d:.3} ms\n", .{ w.rank, hq.lin.n, D, hq.lin.k2, hq.lin.wordBytes() >> 20, n, ms });
            } else {
                std.debug.print("TIME rank {d} head bf16 [{d} x {d}] ({d} MB) rows {d}: {d:.3} ms\n", .{ w.rank, V, D, V * D * 2 >> 20, n, ms });
            }
        }
        const L0 = &w.layers[0];
        const names = [_][]const u8{ "layer0 mlp.gate", "layer0 q_a", "layer0 kv_a" };
        const lins = [_]*const @TypeOf(L0.q_a){ &L0.mlp.gate, &L0.q_a, &L0.kv_a };
        const outs = [_]u64{ r.pb.g, r.pb.qa, r.pb.kva };
        for (names, lins, outs) |name, l, out| {
            for (1..4) |n| {
                for (0..10) |_| try r.f.lin(&r.pb, l, r.pb.normed, n, out, .bf16);
                try r.f.s.synchronize();
                const t0 = s.now();
                for (0..reps) |_| try r.f.lin(&r.pb, l, r.pb.normed, n, out, .bf16);
                try r.f.s.synchronize();
                std.debug.print("TIME rank {d} exl3 {s} [{d} x {d}] {d} half-bits ({d} MB) rows {d}: {d:.3} ms\n", .{ w.rank, name, l.n, l.k, l.k2, l.n * l.k * l.k2 / 16 >> 20, n, s.since(t0) * 1e3 / @as(f64, @floatFromInt(reps)) });
            }
        }
    }

    /// --score-head-check N (a gate, not a score; TF_GLM53_HEAD_Q): `n` fixed pseudo-random bf16 rows through the
    /// quantized head's launch on the device against headq.replay of this rank's part `h` on the CPU. The launch's
    /// tile steps are mma sums whose inner order the replay cannot know, so the bound is 1e-4 of the largest logit
    /// (the replay agrees with the numpy reference to 2e-5 on the host); each row's first maximum must be the same.
    pub fn checkHead(s: *Scorer, h: *const headq.Host, n: usize) !void {
        const r = s.r;
        const w = r.w;
        const D = w.config.hidden;
        const V = w.vocab_part;
        const hq = w.head_q orelse return error.NoHead;
        if (n == 0 or n > s.o.head_rows or h.k != D or h.cut.part != V or h.cut.n() != hq.lin.n or h.cut.off != hq.off or h.k2 != hq.lin.k2) return error.Invalid;
        const t_: tri.Tri = .{ .set = &r.f.k.triton, .s = r.f.s };
        const o = kern.Ops{ .k = r.f.k, .s = r.f.s };
        const x = try s.gpa.alloc(u16, n * D);
        defer s.gpa.free(x);
        var prng = std.Random.DefaultPrng.init(0x53 + w.rank);
        for (x) |*v| v.* = @import("cuda_weights.zig").bf16Bits(prng.random().floatNorm(f32)); // unit rows, as the final norm leaves them
        try o.upload(s.fnormed.ptr, std.mem.sliceAsBytes(x));
        try fw.headLogits(w, t_, o, s.fnormed.ptr, s.lg.ptr, s.hpart.ptr, s.hxh.ptr, n);
        try o.download(s.host.bytes[0 .. n * V * 4], s.lg.ptr);
        try r.f.s.synchronize();
        const got = std.mem.bytesAsSlice(f32, s.host.bytes)[0 .. n * V];
        const want = try s.gpa.alloc(f32, n * V);
        defer s.gpa.free(want);
        try headq.replay(s.gpa, h, x, n, want);
        var top: f32 = 0;
        var worst: f32 = 0;
        var same_bits: usize = 0;
        for (got, want) |g, e| {
            top = @max(top, @abs(e));
            worst = @max(worst, @abs(g - e));
            same_bits += @intFromBool(@as(u32, @bitCast(g)) == @as(u32, @bitCast(e)));
        }
        var picks: usize = 0;
        for (0..n) |i| picks += @intFromBool(fw.argmaxFirst(got[i * V ..][0..V]) == fw.argmaxFirst(want[i * V ..][0..V]));
        const ok = worst <= 1e-4 * top and picks == n;
        std.debug.print("HEADQ rank {d}: {d} rows x {d} logits, {d} half-bits: largest difference {e:.2} of {e:.2} (bound 1e-4), {d} of {d} values the replay's bits, {d} of {d} rows the same first maximum: {s}\n", .{ w.rank, n, V, hq.lin.k2, worst, top, same_bits, got.len, picks, n, if (ok) "OK" else "FAIL" });
        if (!ok) return error.HeadCheckFailed;
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
                try fw.headLogits(w, t_, o, s.fnormed.ptr, s.lg.ptr, s.hpart.ptr, s.hxh.ptr, n);
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
