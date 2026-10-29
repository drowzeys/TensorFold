//! d1-2B (LFM3-VL) on CUDA: an engine that scores decisions through the score hook and writes no text. The bf16
//! release's text decoder runs on LFM3's CUDA forward; each question is read after its trunk as one causal prompt.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lfm3 = @import("lfm3_cuda");
const Allocator = std.mem.Allocator;
const c = lfm3.config;
const mdl = lfm3.model;

pub const family: api.Family = .{ .model_type = "lfm3_vl", .formats = &.{"bf16"} };
const default_context = 32768;

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
    window: u32,
    mutex: std.Io.Mutex = .init, // one prompt pass at a time: HTTP threads call score directly

    fn of(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .name = "lfm3_vl", .context_window = of(ctx).window, .thinking_budget = false, .decides = true, .pictures = false, .decision_lead = "<think></think>", .decision_temperature = of(ctx).temperature };
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

    /// `ids` from position 0 on slot 0 in prompt chunks; the last row's logits in `e.row`.
    fn pass(e: *Engine, ids: []const u32) !void {
        if (ids.len == 0 or ids.len > e.window) return error.BadWindow;
        try e.m.clearSlot(e.stream, 0);
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(mdl.prompt_rows, ids.len - at);
            const last = at + n == ids.len;
            try lfm3.forward.forward(&e.m, &e.k, e.stream, &.{.{ .slot = 0, .p0 = at, .tokens = ids[at..][0..n] }}, true, if (last) n - 1 else 0, @intFromBool(last));
            at += n;
        }
        const d = &e.driver;
        try d.check(d.api.cuMemcpyDtoHAsync_v2(e.row.ptr, e.m.s.logits, e.row.len * 2, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
    }

    fn logit(e: *const Engine, id: u32) f64 {
        return @as(f32, @bitCast(@as(u32, e.row[id]) << 16));
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
        e.pass(prompt) catch |err| {
            std.log.err("lfm3_vl decision pass failed: {s}", .{@errorName(err)});
            return error.Failed;
        };
        for (labels, logits) |id, *l| {
            if (id >= c.vocab) return error.Failed;
            l.* = e.logit(id);
        }
        return if (mass) e.lse() else 0;
    }

    /// Each branch read after the trunk as one causal prompt: trunk ++ branch, the labels at its last row.
    fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
        const e = of(ctx);
        const a = d.arena orelse return error.Failed;
        if (d.pictures.len > 0) return refuse(d, "LFM3-VL's vision tower is not built on CUDA in this engine yet");
        for (d.branches) |b| if (d.trunk.len + b.len > e.window)
            return refuse(d, std.fmt.allocPrint(a, "a decision sequence exceeds the {d}-token context", .{e.window}) catch return error.Failed);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        e.ctx.makeCurrent() catch return error.Failed;
        d.read = d.trunk.len;
        var at: usize = 0;
        for (d.branches, d.counts) |b, n| {
            const ids = std.mem.concat(a, u32, &.{ d.trunk, b }) catch return error.Failed;
            e.pass(ids) catch |err| {
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

    fn refuse(d: *api.Decision, problem: []const u8) error{Refused} {
        d.problem = problem;
        return error.Refused;
    }

    fn close(ctx: *anyopaque) void {
        const e = of(ctx);
        e.ctx.makeCurrent() catch {};
        e.gpa.free(e.row);
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
    e.w = lfm3.weights.load(gpa, io, &e.driver, e.stream, &e.k, o.dir, null) catch |err| {
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
    e.temperature = calibration(a, io, o.dir) catch 1;
    std.log.info("d1 (LFM3-VL) on CUDA: decisions over a {d}-token window; pictures are refused until the vision tower is built", .{window});
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
