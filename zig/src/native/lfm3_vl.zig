//! d1-2B (LFM3-VL) on CUDA: an engine that scores decisions through the score hook and writes no text. The bf16
//! release's text decoder runs on LFM3's CUDA forward, pictures on LFM3-VL's tower (lfm3_vision.zig); each question
//! is read after its trunk as one causal prompt.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lfm3 = @import("lfm3_cuda");
const lfm2 = @import("lfm2");
const vision = @import("lfm3_vision.zig");
const Allocator = std.mem.Allocator;
const c = lfm3.config;
const mdl = lfm3.model;
const core = @import("core");

pub const family: api.Family = .{ .model_type = "lfm3_vl", .formats = &.{"bf16"} };
const default_context = 32768;
/// Pictures a decision takes (their features live in one buffer of this many cap-size pictures).
const max_pictures = 16;
const feature_rows = max_pictures * vision.patches_most / 4;

const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: lfm3.kernels.Kernels,
    w: lfm3.weights.Weights,
    m: mdl.Model,
    row: []u16, // one logits row on the host (bf16 [vocab])
    temperature: f64 = 1, // calibration.json's default
    vw: ?vision.Weights = null,
    tower: ?vision.Vision = null,
    feats: ?cuda.DeviceBuffer = null, // bf16 [feature_rows, hidden]: a decision's pictures' rows, in prompt order
    set_size: usize = 0, // d1's `set` attention: questions packed after their state, this many a pass; 0: causal
    header: []const u32 = &.{}, // "QUESTION:\n": what each further question of a pack opens with
    set_fn: ?cuda.Function = null,
    groups: ?cuda.DeviceBuffer = null, // int32 [prompt_rows]: each packed row's question, -1 in the state
    rows: []u16 = &.{}, // the packed questions' logits rows on the host (bf16 [round_rows, vocab])
    window: u32,
    mutex: std.Io.Mutex = .init, // one prompt pass at a time: HTTP threads call score directly

    fn of(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .name = "lfm3_vl", .context_window = of(ctx).window, .thinking_budget = false, .decides = true, .pictures = of(ctx).tower != null, .max_pictures = max_pictures, .decision_lead = "<think></think>", .decision_temperature = of(ctx).temperature };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed; // the server refuses text requests before they reach a decision engine
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak };
    }

    /// `ids` from position 0 on slot 0 in prompt chunks, the picture rows of `runs` over their tokens; the last
    /// row's logits in `e.row`.
    fn pass(e: *Engine, ids: []const u32, runs: []const Run) !void {
        if (ids.len == 0 or ids.len > e.window) return error.BadWindow;
        try e.m.clearSlot(e.stream, 0);
        var at: usize = 0;
        var inserts: [2 * max_pictures]lfm3.forward.Insert = undefined;
        while (at < ids.len) {
            const n = @min(mdl.prompt_rows, ids.len - at);
            const last = at + n == ids.len;
            // each run's part inside this chunk
            var k: usize = 0;
            for (runs) |r| {
                const lo = @max(r.at, at);
                const hi = @min(r.at + r.count, at + n);
                if (lo >= hi) continue;
                inserts[k] = .{ .at = lo - at, .src = e.feats.?.ptr + (r.row + lo - r.at) * c.hidden * 2, .count = hi - lo };
                k += 1;
            }
            try lfm3.forward.forwardWith(&e.m, &e.k, e.stream, &.{.{ .slot = 0, .p0 = at, .tokens = ids[at..][0..n] }}, true, if (last) n - 1 else 0, @intFromBool(last), .{ .inserts = inserts[0..k], .gemm = &e.tower.?.g });
            at += n;
        }
        const d = &e.driver;
        try d.check(d.api.cuMemcpyDtoHAsync_v2(e.row.ptr, e.m.s.logits, e.row.len * 2, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
    }

    fn logit(e: *const Engine, id: u32) f64 {
        return @as(f32, @bitCast(@as(u32, e.row[id]) << 16));
    }

    /// One packed row under `set` attention: `ids` from position 0, `groups` a row, picture `runs`; the logits of
    /// rows `ends` (each question's last) in `e.rows`, in order.
    fn packPass(e: *Engine, ids: []const u32, groups: []const i32, state: usize, ends: []const usize, runs: []const Run) !void {
        if (ids.len > mdl.prompt_rows) return error.PackTooLong;
        try e.m.clearSlot(e.stream, 0);
        var inserts: [max_pictures]lfm3.forward.Insert = undefined;
        for (runs, 0..) |r, i| inserts[i] = .{ .at = r.at, .src = e.feats.?.ptr + r.row * c.hidden * 2, .count = r.count };
        const d = &e.driver;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(e.groups.?.ptr, groups.ptr, groups.len * 4, e.stream.handle), "cuMemcpyHtoDAsync");
        try lfm3.forward.forwardWith(&e.m, &e.k, e.stream, &.{.{ .slot = 0, .p0 = 0, .tokens = ids }}, true, 0, 0, .{
            .inserts = inserts[0..runs.len],
            .set = .{ .kernel = e.set_fn.?, .group = e.groups.?.ptr, .state = state },
            .heads_at = ends,
            .gemm = &e.tower.?.g,
        });
        try d.check(d.api.cuMemcpyDtoHAsync_v2(e.rows.ptr, e.m.s.logits, ends.len * c.vocab * 2, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
    }

    fn lse(e: *const Engine) f64 {
        var most: f64 = -std.math.inf(f64);
        for (0..e.row.len) |i| most = @max(most, e.logit(@intCast(i)));
        var sum: f64 = 0;
        for (0..e.row.len) |i| sum += @exp(e.logit(@intCast(i)) - most);
        return most + @log(sum);
    }

    fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
        const e = of(ctx);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        e.ctx.makeCurrent() catch return error.Failed;
        e.pass(prompt, &.{}) catch |err| {
            std.log.err("lfm3_vl decision pass failed: {s}", .{@errorName(err)});
            return error.Failed;
        };
        for (labels, logits) |id, *l| {
            if (id >= c.vocab) return error.Failed;
            l.* = e.logit(id);
        }
        return if (mass) e.lse() else 0;
    }

    /// Pictures decoded and prepared before the GPU is taken; then their features, and each branch read after the
    /// trunk as one causal prompt: trunk ++ branch, the labels at its last row.
    fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
        const e = of(ctx);
        const a = d.arena orelse return error.Failed;
        var trunk = d.trunk;
        const pics = a.alloc(vision.Picture, d.pictures.len) catch return error.Failed;
        var runs: std.ArrayList(Run) = .empty;
        var jobs: []Prep = &.{};
        // the pictures' host rows stay until the decision ends: their uploads are queued on the stream
        defer if (jobs.len > 0) {
            e.stream.synchronize() catch {};
            for (jobs) |*job| job.free();
        };
        if (d.pictures.len > 0) {
            if (e.tower == null) return refuse(d, "this checkpoint has no vision tower");
            if (d.pictures.len > max_pictures) return refuse(d, std.fmt.allocPrint(a, "at most {d} pictures a decision", .{max_pictures}) catch return error.Failed);
            var ids: std.ArrayList(u32) = .empty;
            ids.appendSlice(a, d.trunk[0..d.at]) catch return error.Failed;
            // each picture decoded, resized and cut into patches on its own thread, before the GPU is taken
            jobs = a.alloc(Prep, d.pictures.len) catch return error.Failed;
            for (jobs) |*job| job.* = .{ .io = e.io, .url = "" };
            const threads = a.alloc(?std.Thread, d.pictures.len) catch return error.Failed;
            for (d.pictures, jobs, threads) |url, *job, *th| {
                job.* = .{ .io = e.io, .url = url };
                th.* = std.Thread.spawn(.{}, Prep.run, .{job}) catch null;
                if (th.* == null) job.run();
            }
            for (threads) |th| if (th) |t| t.join();
            var row: usize = 0;
            for (jobs, pics) |*job, *pic| {
                if (job.problem.len > 0) return refuse(d, job.problem);
                pic.* = job.pic orelse return error.Failed;
                ids.append(a, vision.vision_start) catch return error.Failed;
                runs.append(a, .{ .at = ids.items.len, .count = pic.tokens(), .row = row }) catch return error.Failed;
                ids.appendNTimes(a, vision.image_token, pic.tokens()) catch return error.Failed;
                ids.append(a, vision.vision_end) catch return error.Failed;
                row += pic.tokens();
            }
            ids.appendSlice(a, d.trunk[d.at..]) catch return error.Failed;
            trunk = ids.items;
        }
        for (d.branches) |b| if (trunk.len + b.len > e.window)
            return refuse(d, std.fmt.allocPrint(a, "a decision sequence exceeds the {d}-token context", .{e.window}) catch return error.Failed);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        e.ctx.makeCurrent() catch return error.Failed;
        var row: usize = 0;
        for (pics) |*pic| {
            row += e.tower.?.features(e.stream, pic, e.feats.?.ptr + row * c.hidden * 2) catch |err| {
                std.log.err("lfm3_vl picture features failed: {s}", .{@errorName(err)});
                return error.Failed;
            };
        }
        d.read = trunk.len;
        if (e.set_size > 0) return e.decideSet(d, a, trunk, runs.items);
        var at: usize = 0;
        for (d.branches, d.counts) |b, n| {
            const ids = std.mem.concat(a, u32, &.{ trunk, b }) catch return error.Failed;
            e.pass(ids, runs.items) catch |err| {
                std.log.err("lfm3_vl decision pass failed: {s}", .{@errorName(err)});
                return error.Failed;
            };
            for (d.labels[at..][0..n], d.logits[at..][0..n]) |id, *l| {
                if (id >= c.vocab) return error.Failed;
                l.* = e.logit(id);
            }
            d.read += b.len;
            at += n;
        }
    }

    /// d1's `set` reading: up to `set_size` questions a pass after their state, the first as its lone prompt, each
    /// further one opened by the state's question header; every row of the state sees all, a question its state and
    /// itself (attention.py `pack_set`, `Sees`).
    fn decideSet(e: *Engine, d: *api.Decision, a: Allocator, trunk: []const u32, runs: []const Run) error{ Failed, Refused }!void {
        const hdr = if (std.mem.endsWith(u32, trunk, e.header)) e.header else &[_]u32{};
        var label_at: usize = 0;
        var start: usize = 0;
        while (start < d.branches.len) {
            const end = @min(d.branches.len, start + e.set_size);
            var ids: std.ArrayList(u32) = .empty;
            var groups: std.ArrayList(i32) = .empty;
            var ends: std.ArrayList(usize) = .empty;
            ids.appendSlice(a, trunk) catch return error.Failed;
            groups.appendNTimes(a, -1, trunk.len) catch return error.Failed;
            for (d.branches[start..end], start..) |b, k| {
                const own_hdr = if (k > start) hdr else &[_]u32{};
                ids.appendSlice(a, own_hdr) catch return error.Failed;
                ids.appendSlice(a, b) catch return error.Failed;
                groups.appendNTimes(a, @intCast(k - start + 1), own_hdr.len + b.len) catch return error.Failed;
                ends.append(a, ids.items.len - 1) catch return error.Failed;
                d.read += own_hdr.len + b.len;
            }
            if (ids.items.len > mdl.prompt_rows) return refuse(d, std.fmt.allocPrint(a, "a packed decision of {d} tokens is past the {d} rows one pass reads", .{ ids.items.len, mdl.prompt_rows }) catch return error.Failed);
            e.packPass(ids.items, groups.items, trunk.len, ends.items, runs) catch |err| {
                std.log.err("lfm3_vl decision pass failed: {s}", .{@errorName(err)});
                return error.Failed;
            };
            for (d.counts[start..end], 0..) |n, q| {
                const row = e.rows[q * c.vocab ..][0..c.vocab];
                for (d.labels[label_at..][0..n], d.logits[label_at..][0..n]) |id, *l| {
                    if (id >= c.vocab) return error.Failed;
                    l.* = @as(f32, @bitCast(@as(u32, row[id]) << 16));
                }
                label_at += n;
            }
            start = end;
        }
    }

    /// One picture's CPU work on its own thread, in a thread-safe allocator (freed after the decision).
    const Prep = struct {
        io: std.Io,
        url: []const u8,
        pic: ?vision.Picture = null,
        problem: []const u8 = "",
        failed: bool = false,

        fn run(p: *Prep) void {
            const a = std.heap.smp_allocator;
            var problem: []const u8 = "";
            const rgb = lfm2.image.load(p.io, a, p.url, &problem) catch {
                p.failed = true;
                return;
            } orelse {
                p.problem = problem;
                return;
            };
            defer a.free(rgb.pixels);
            p.pic = vision.prepare(a, rgb) catch |err| {
                if (err == error.PictureTooLarge) p.problem = "a picture past the tower's patch limit" else p.failed = true;
                return;
            };
        }

        fn free(p: *Prep) void {
            if (p.pic) |*pic| vision.free(std.heap.smp_allocator, pic);
        }
    };

    /// A picture's tokens in the prompt: `count` from position `at`, its rows from feature row `row`.
    const Run = struct { at: usize, count: usize, row: usize };

    fn refuse(d: *api.Decision, problem: []const u8) error{Refused} {
        d.problem = problem;
        return error.Refused;
    }

    fn close(ctx: *anyopaque) void {
        const e = of(ctx);
        e.ctx.makeCurrent() catch {};
        e.gpa.free(e.row);
        if (e.feats) |*f| f.free();
        if (e.groups) |*g| g.free();
        e.gpa.free(e.rows);
        e.gpa.free(e.header);
        if (e.tower) |*t| t.deinit();
        if (e.vw) |*v| v.deinit(e.gpa);
        e.m.deinit();
        e.w.deinit();
        e.k.deinit();
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .score = score, .decide = decide };
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const window: u32 = if (o.context) |cx| blk: {
        if (cx < 1 or cx > default_context) {
            problem.* = try std.fmt.allocPrint(a, "--context {d}: d1 reads 1 to {d} tokens a decision", .{ cx, default_context });
            return null;
        }
        break :blk @intCast(cx);
    } else default_context;
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .window = window, .driver = undefined, .ctx = undefined, .stream = undefined, .k = undefined, .w = undefined, .m = undefined, .row = &.{} };
    e.driver = cuda.Driver.open() catch |err| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(err)});
        return null;
    };
    errdefer e.driver.close();
    e.ctx = try cuda.Context.init(&e.driver, 0);
    errdefer e.ctx.deinit();
    try e.ctx.makeCurrent();
    e.stream = try cuda.Stream.init(&e.driver, false);
    errdefer e.stream.deinit();
    e.k = try lfm3.kernels.Kernels.load(&e.ctx);
    errdefer e.k.deinit();
    e.w = lfm3.weights.loadPlain(gpa, io, &e.driver, e.stream, &e.k, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "d1 (LFM3-VL): the bf16 text decoder did not load ({s})", .{@errorName(err)});
        return null;
    };
    errdefer e.w.deinit();
    if (e.w.config.kind != .vision_language) {
        problem.* = "lfm3_vl: this checkpoint is LFM3 text";
        return null;
    }
    e.m = try mdl.Model.init(gpa, &e.driver, e.stream, &e.w, &e.k, window, 1, false);
    errdefer e.m.deinit();
    e.row = try gpa.alloc(u16, c.vocab);
    e.vw = try vision.load(gpa, io, &e.driver, o.dir);
    e.tower = try vision.Vision.init(gpa, &e.driver, &e.vw.?);
    e.feats = try cuda.DeviceBuffer.alloc(&e.driver, feature_rows * c.hidden * 2);
    // d1's reading: `set` with its recorded set size unless TF_D1_ATTENTION=causal
    const mode = try d1Mode(a, io, o.dir);
    const forced = std.c.getenv("TF_D1_ATTENTION");
    const causal = forced != null and std.mem.eql(u8, std.mem.span(forced.?), "causal");
    if (mode.set and !causal) {
        e.set_size = mode.size;
        e.set_fn = try e.tower.?.glue.mod.function("tf_lfm3v_set_attn_tiled");
        e.groups = try cuda.DeviceBuffer.alloc(&e.driver, mdl.prompt_rows * 4);
        e.rows = try gpa.alloc(u16, mdl.round_rows * c.vocab);
        var tok = try core.tokenizer.loadTokenizer(io, gpa, try std.fs.path.join(a, &.{ o.dir, "tokenizer.json" }));
        defer tok.deinit();
        e.header = try gpa.dupe(u32, try tok.encode(a, "QUESTION:\n"));
    }
    e.temperature = calibration(a, io, o.dir) catch 1;
    std.log.info("d1 (LFM3-VL) on CUDA: decisions over a {d}-token window, up to {d} pictures each; {s} attention ({d} questions a pass)", .{ window, max_pictures, if (e.set_size > 0) "set" else "causal", @max(e.set_size, 1) });
    return .{ .engine = .{ .ctx = e, .vtable = &Engine.vtable }, .close = Engine.close, .ctx = e };
}

/// `calibration.json`'s default temperature, 1 without the file; a non-positive or non-finite one is refused.
fn calibration(a: Allocator, io: std.Io, dir: []const u8) !f64 {
    const path = try std.fs.path.join(a, &.{ dir, "calibration.json" });
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22)) catch return 1;
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (v != .object) return error.BadCalibration;
    const t = v.object.get("default") orelse return 1;
    const x: f64 = switch (t) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => return error.BadCalibration,
    };
    if (!(x > 0) or !std.math.isFinite(x)) return error.BadCalibration;
    return x;
}

/// config.json's `d1_attention` ("set" or "causal") and `d1_set_size` (2 to 16, 16 without it).
fn d1Mode(a: Allocator, io: std.Io, dir: []const u8) !struct { set: bool, size: usize } {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (v != .object) return error.BadLfm3Config;
    const att = v.object.get("d1_attention") orelse return .{ .set = false, .size = 0 };
    if (att != .string or !std.mem.eql(u8, att.string, "set")) return .{ .set = false, .size = 0 };
    const size: usize = if (v.object.get("d1_set_size")) |n| switch (n) {
        .integer => |i| std.math.cast(usize, i) orelse return error.BadLfm3Config,
        else => return error.BadLfm3Config,
    } else 16;
    if (size < 2 or size > 16) return error.BadLfm3Config;
    return .{ .set = true, .size = size };
}
