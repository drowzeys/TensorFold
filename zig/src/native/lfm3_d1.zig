//! d1-2B (LFM3-VL) on Metal: decisions scored on LFM3's Metal forward and LFM3-VL's tower; it writes no text.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.lfm3;
const c = q.config;
const Allocator = std.mem.Allocator;

const default_context = 32768;
/// Rows one pass reads: a packed `set` decision (its state, pictures and questions) runs as one pass.
const pass_rows = 4096;
/// Pictures a decision takes, and image tokens each at most (d1_vision_max_pixels' 1280 x 1280).
const max_pictures = 16;
const picture_tokens = 1600;
/// Image tokens one tower pass takes: a decision's pictures go through it together, as many as fit.
const tower_tokens = 2048;
const cap_pixels = 1638400;
/// Projected picture rows a decision holds, in prompt order.
const feature_rows = pass_rows;
const vision_start: u32 = 125009;
const vision_end: u32 = 125010;
const image_token: u32 = 125030;

const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    model: *q.Model,
    scratch: q.state.Scratch,
    cache: q.state.Cache,
    tower: ?*q.vision.Vision = null,
    feats: ?mtl.Buffer = null, // bf16 [feature_rows, hidden]: a decision's pictures' rows, in prompt order
    copy: ?mtl.Pipeline = null,
    groups: mtl.Buffer, // int32 [pass_rows]: each packed row's question, -1 in the state
    temperature: f64 = 1, // calibration.json's default
    set_size: usize = 0, // d1's `set` attention: questions packed after their state, this many a pass; 0: causal
    window: u32,
    last: []const u16 = &.{}, // the final hidden row of `pass`'s last row (its view of the scratch)
    profile: bool = false, // TF_D1_PROFILE: each decision's tower and pass seconds in the log
    mutex: std.Io.Mutex = .init, // one prompt pass at a time: HTTP threads call score directly

    fn of(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        const e = of(ctx);
        return .{ .name = "lfm3_vl", .context_window = e.window, .thinking_budget = false, .decides = true, .pictures = e.tower != null, .max_pictures = max_pictures, .decision_lead = "<think></think>", .decision_temperature = e.temperature };
    }

    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed; // the server refuses text requests before they reach a decision engine
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(ctx: *anyopaque, _: bool) ?api.Memory {
        const dev = of(ctx).model.device;
        return .{ .active = dev.allocated() };
    }

    /// The cache emptied: no keys, every convolution's history zero.
    fn reset(e: *Engine) void {
        e.cache.len = 0;
        e.cache.last = null;
        for (e.cache.blocks) |b| if (b == .conv) @memset(b.conv.contents()[0..q.state.conv_bytes], 0);
    }

    /// The pictures' rows over their runs of `ids`, as the forward lays them over the embedding.
    fn images(e: *Engine, spans: []const q.image_rows.Span) ?q.image_rows.Images {
        if (spans.len == 0) return null;
        return .{ .buffer = e.feats.?, .pipeline = e.copy.?, .spans = spans };
    }

    /// `ids` from position 0, causal, in passes of `pass_rows`; the last row's logits in the cache.
    fn pass(e: *Engine, ids: []const u32, spans: []const q.image_rows.Span) !void {
        if (ids.len == 0 or ids.len > e.window) return error.BadWindow;
        e.reset();
        var at: usize = 0;
        while (at < ids.len) {
            const n: usize = @min(pass_rows, ids.len - at);
            if (at == 0) {
                try q.forward.runDecision(e.model, &e.scratch, &e.cache, ids[0..n], e.images(spans), .{});
            } else {
                try q.forward.runWithImages(e.model, &e.scratch, &.{.{ .cache = &e.cache, .rows = n }}, ids[at..][0..n], false, .last, e.images(spans));
            }
            at += n;
            e.last = e.scratch.x.slice(u16, n * c.hidden)[(n - 1) * c.hidden ..];
        }
    }

    /// Label `id`'s logit for a final hidden row: a bf16 head in f64 (no bf16 logit rounding), else the bf16 row.
    fn label(e: *const Engine, hidden: []const u16, logits: []const u16, id: u32) f64 {
        const head = e.model.weights.embedding;
        if (!q.metal_weights.isDense(head)) return @as(f32, @bitCast(@as(u32, logits[id]) << 16));
        const row: [*]const u16 = @ptrCast(@alignCast(head.w.contents() + head.w_off + @as(usize, id) * c.hidden * 2));
        var sum: f64 = 0;
        for (hidden, row[0..c.hidden]) |x, y| sum += @as(f64, @as(f32, @bitCast(@as(u32, x) << 16))) * @as(f32, @bitCast(@as(u32, y) << 16));
        return sum;
    }

    /// The last pass's final row (after the final norm) and its bf16 logits.
    fn logit(e: *const Engine, id: u32) f64 {
        return e.label(e.last, e.cache.logits.slice(u16, c.vocab), id);
    }

    fn lse(e: *const Engine) f64 {
        const row = e.cache.logits.slice(u16, c.vocab);
        var most: f64 = -std.math.inf(f64);
        for (row) |w| most = @max(most, @as(f32, @bitCast(@as(u32, w) << 16)));
        var sum: f64 = 0;
        for (row) |w| sum += @exp(@as(f64, @as(f32, @bitCast(@as(u32, w) << 16))) - most);
        return most + @log(sum);
    }

    fn score(ctx: *anyopaque, prompt: []const u32, labels: []const u32, logits: []f64, mass: bool) error{Failed}!f64 {
        const e = of(ctx);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
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

    /// Pictures prepared before the GPU is taken, then the branches read after the trunk (packed under `set`).
    fn decide(ctx: *anyopaque, d: *api.Decision) error{ Failed, Refused }!void {
        const e = of(ctx);
        const a = d.arena orelse return error.Failed;
        var trunk = d.trunk;
        var spans: std.ArrayList(q.image_rows.Span) = .empty;
        var jobs: []Prep = &.{};
        defer for (jobs) |*job| job.free();
        if (d.pictures.len > 0) {
            if (e.tower == null) return refuse(d, "this checkpoint has no vision tower");
            if (d.pictures.len > max_pictures) return refuse(d, std.fmt.allocPrint(a, "at most {d} pictures a decision", .{max_pictures}) catch return error.Failed);
            var ids: std.ArrayList(u32) = .empty;
            ids.appendSlice(a, d.trunk[0..d.at]) catch return error.Failed;
            // each picture decoded, resized and cut into patches on its own thread, before the GPU is taken
            jobs = a.alloc(Prep, d.pictures.len) catch return error.Failed;
            const threads = a.alloc(?std.Thread, d.pictures.len) catch return error.Failed;
            for (d.pictures, jobs, threads) |url, *job, *th| {
                job.* = .{ .io = e.io, .url = url };
                th.* = std.Thread.spawn(.{}, Prep.run, .{job}) catch null;
                if (th.* == null) job.run();
            }
            for (threads) |th| if (th) |t| t.join();
            var row: u32 = 0;
            for (jobs) |*job| {
                if (job.problem.len > 0) return refuse(d, job.problem);
                const pic = job.pic orelse return error.Failed;
                if (row + pic.tokens > feature_rows) return refuse(d, "the pictures' tokens exceed what one decision holds");
                ids.append(a, vision_start) catch return error.Failed;
                spans.append(a, .{ .at = @intCast(ids.items.len), .rows = pic.tokens, .source = row }) catch return error.Failed;
                ids.appendNTimes(a, image_token, pic.tokens) catch return error.Failed;
                ids.append(a, vision_end) catch return error.Failed;
                row += pic.tokens;
            }
            ids.appendSlice(a, d.trunk[d.at..]) catch return error.Failed;
            trunk = ids.items;
        }
        for (d.branches) |b| if (trunk.len + b.len > e.window)
            return refuse(d, std.fmt.allocPrint(a, "a decision sequence exceeds the {d}-token context", .{e.window}) catch return error.Failed);
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const t0 = std.Io.Clock.awake.now(e.io).toNanoseconds();
        defer if (e.profile) {
            const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(e.io).toNanoseconds() - t0)) / 1e6;
            std.log.info("d1 decision: {d} pictures, {d} rows, {d:.2} ms (gpu {d:.2} ms in the passes)", .{ jobs.len, d.read, ms, 1000 * q.forward.gpu_seconds });
            q.forward.gpu_seconds = 0;
        };
        var first: usize = 0;
        while (first < jobs.len) {
            var batch: [max_pictures]q.vision.Vision.Picture = undefined;
            var tokens: u32 = 0;
            var n: usize = 0;
            while (first + n < jobs.len and tokens + jobs[first + n].pic.?.tokens <= tower_tokens) : (n += 1) {
                const pic = jobs[first + n].pic.?;
                batch[n] = .{ .pixels = pic.pixels, .gh = pic.gh, .gw = pic.gw };
                tokens += pic.tokens;
            }
            const out: q.vision.Ref = .{ .buf = e.feats.?, .off = @as(usize, spans.items[first].source) * c.hidden * 2 };
            e.tower.?.images(e.model.queue, batch[0..n], out) catch |err| {
                std.log.err("lfm3_vl picture features failed: {s}", .{@errorName(err)});
                return error.Failed;
            };
            first += n;
        }
        if (e.profile and jobs.len > 0) std.log.info("d1 tower: {d:.2} ms", .{@as(f64, @floatFromInt(std.Io.Clock.awake.now(e.io).toNanoseconds() - t0)) / 1e6});
        d.read = trunk.len;
        if (e.set_size > 0) return e.decideSet(d, a, trunk, spans.items);
        var at: usize = 0;
        for (d.branches, d.counts) |b, n| {
            const ids = std.mem.concat(a, u32, &.{ trunk, b }) catch return error.Failed;
            e.pass(ids, spans.items) catch |err| {
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

    /// d1's set reading: up to `set_size` questions a pass after their state, each seeing its state and itself.
    fn decideSet(e: *Engine, d: *api.Decision, a: Allocator, trunk: []const u32, spans: []const q.image_rows.Span) error{ Failed, Refused }!void {
        const hdr = if (d.header.len > 0 and std.mem.endsWith(u32, trunk, d.header)) d.header else &[_]u32{};
        var label_at: usize = 0;
        var start: usize = 0;
        while (start < d.branches.len) {
            const end = @min(d.branches.len, start + e.set_size);
            var ids: std.ArrayList(u32) = .empty;
            var ends: std.ArrayList(usize) = .empty;
            ids.appendSlice(a, trunk) catch return error.Failed;
            for (d.branches[start..end], start..) |b, k| {
                if (k > start) ids.appendSlice(a, hdr) catch return error.Failed;
                ids.appendSlice(a, b) catch return error.Failed;
                ends.append(a, ids.items.len - 1) catch return error.Failed;
                d.read += b.len + if (k > start) hdr.len else 0;
            }
            if (ids.items.len > pass_rows) return refuse(d, std.fmt.allocPrint(a, "a packed decision of {d} tokens is past the {d} rows one pass reads", .{ ids.items.len, pass_rows }) catch return error.Failed);
            // each row's question: -1 in the state, then 1, 2, .. for the packed questions (a header joins its own)
            const groups = e.groups.slice(i32, ids.items.len);
            @memset(groups[0..trunk.len], -1);
            var from = trunk.len;
            for (ends.items, 1..) |last, g| {
                @memset(groups[from .. last + 1], @intCast(g));
                from = last + 1;
            }
            e.reset();
            q.forward.runDecision(e.model, &e.scratch, &e.cache, ids.items, e.images(spans), .{ .set = .{ .group = e.groups, .state = trunk.len }, .heads_at = ends.items }) catch |err| {
                std.log.err("lfm3_vl decision pass failed: {s}", .{@errorName(err)});
                return error.Failed;
            };
            const rows = e.scratch.logits.slice(u16, ends.items.len * c.vocab);
            const hidden = e.scratch.r.slice(u16, ends.items.len * c.hidden);
            for (d.counts[start..end], 0..) |n, qi| {
                const row = rows[qi * c.vocab ..][0..c.vocab];
                for (d.labels[label_at..][0..n], d.logits[label_at..][0..n]) |id, *l| {
                    if (id >= c.vocab) return error.Failed;
                    l.* = e.label(hidden[qi * c.hidden ..][0..c.hidden], row, id);
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
        pic: ?q.image.Prepared = null,
        problem: []const u8 = "",

        fn run(p: *Prep) void {
            const a = std.heap.smp_allocator;
            const bytes = dataBytes(a, p.url, &p.problem) orelse return;
            defer a.free(bytes);
            var rgb = tf.picture.decode(p.io, a, bytes) catch |err| {
                p.problem = tf.picture.decoderProblem(err) orelse "the picture did not decode";
                return;
            };
            defer rgb.deinit(a);
            p.pic = q.image.prepareCapped(a, rgb, cap_pixels, picture_tokens) catch |err| {
                p.problem = if (err == error.ImageTooLarge) "a picture past the tower's patch limit" else q.image.explain(err) orelse "the picture could not be prepared";
                return;
            };
        }

        fn free(p: *Prep) void {
            if (p.pic) |pic| std.heap.smp_allocator.free(pic.pixels);
        }
    };

    fn refuse(d: *api.Decision, problem: []const u8) error{Refused} {
        d.problem = problem;
        return error.Refused;
    }

    fn close(ctx: *anyopaque) void {
        const e = of(ctx);
        if (e.feats) |f| f.deinit();
        if (e.copy) |p| p.deinit();
        if (e.tower) |t| t.deinit();
        e.groups.deinit();
        e.cache.deinit();
        e.scratch.deinit();
        e.model.deinit();
        e.gpa.destroy(e);
    }

    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .score = score, .decide = decide };
};

/// A data URL's bytes (base64 or plain); null with `problem` set for anything else.
fn dataBytes(a: Allocator, url: []const u8, problem: *[]const u8) ?[]u8 {
    if (!std.mem.startsWith(u8, url, "data:")) {
        problem.* = "decision pictures are data URLs";
        return null;
    }
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse {
        problem.* = "invalid image data URL";
        return null;
    };
    const payload = url[comma + 1 ..];
    if (!std.mem.endsWith(u8, url[0..comma], ";base64")) return a.dupe(u8, payload) catch null;
    const dec = std.base64.standard.Decoder;
    const n = dec.calcSizeForSlice(payload) catch {
        problem.* = "invalid image data URL encoding";
        return null;
    };
    const out = a.alloc(u8, n) catch return null;
    dec.decode(out, payload) catch {
        a.free(out);
        problem.* = "invalid image data URL encoding";
        return null;
    };
    return out;
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const window: u32 = if (o.context) |cx| blk: {
        if (cx < 1 or cx > default_context) {
            problem.* = try std.fmt.allocPrint(a, "--context {d}: d1 reads 1 to {d} tokens a decision", .{ cx, default_context });
            return null;
        }
        break :blk @intCast(cx);
    } else default_context;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = q.Model.load(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "d1 (LFM3-VL): the text decoder did not load ({s})", .{@errorName(err)});
        return null;
    };
    if (model.config.kind != .vision_language) {
        model.deinit();
        problem.* = "lfm3_vl: this checkpoint is LFM3 text";
        return null;
    }
    errdefer model.deinit();
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .window = window, .model = model, .scratch = undefined, .cache = undefined, .groups = undefined };
    e.scratch = try q.state.Scratch.init(gpa, model.device, &model.config.layer_types, pass_rows);
    errdefer e.scratch.deinit();
    e.cache = try q.state.Cache.init(gpa, model.device, &model.config.layer_types, window);
    errdefer e.cache.deinit();
    e.groups = try model.device.buffer(pass_rows * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer e.groups.deinit();
    e.tower = try q.vision.Vision.load(gpa, model.device, &model.checkpoint, tower_tokens);
    errdefer e.tower.?.deinit();
    e.feats = try model.device.buffer(feature_rows * c.hidden * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer e.feats.?.deinit();
    e.copy = try q.image_rows.load(model.device);
    errdefer e.copy.?.deinit();
    // d1's reading: `set` with its recorded set size unless TF_D1_ATTENTION=causal
    const mode = try d1Mode(a, io, o.dir);
    const forced = std.c.getenv("TF_D1_ATTENTION");
    const causal = forced != null and std.mem.eql(u8, std.mem.span(forced.?), "causal");
    if (mode.set and !causal) {
        e.set_size = @min(mode.size, q.state.batch_rows);
    }
    e.temperature = calibration(a, io, o.dir) catch 1;
    e.profile = std.c.getenv("TF_D1_PROFILE") != null;
    std.log.info("d1 (LFM3-VL) on Metal: decisions over a {d}-token window, up to {d} pictures each; {s} attention ({d} questions a pass)", .{ window, max_pictures, if (e.set_size > 0) "set" else "causal", @max(e.set_size, 1) });
    return .{ .engine = .{ .ctx = e, .vtable = &Engine.vtable }, .close = Engine.close, .ctx = e };
}

/// A d1 checkpoint (its config.json names a `d1_` setting) decides; any other LFM3-VL chats on the lane engine.
pub fn decides(a: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return false;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22)) catch return false;
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return false;
    if (v != .object) return false;
    for (v.object.keys()) |k| if (std.mem.startsWith(u8, k, "d1_")) return true;
    return false;
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

test "a data URL's base64 payload decodes; other URLs are refused" {
    var problem: []const u8 = "";
    const bytes = dataBytes(std.testing.allocator, "data:image/png;base64,aGVsbG8=", &problem).?;
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualStrings("hello", bytes);
    try std.testing.expect(dataBytes(std.testing.allocator, "https://example.com/a.png", &problem) == null);
}
