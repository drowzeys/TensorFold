//! EmbeddingGemma-2's text encoder on Metal: OpenAI embeddings from the bf16 release; it writes no text.
const std = @import("std");
const api = @import("engine_api");
const g = @import("embgemma2_geometry.zig");
const em = @import("embgemma2_model.zig");
const cache_fit = @import("cache_fit.zig");
const Allocator = std.mem.Allocator;
const Model = em.Model;
const Attention = em.Attention;

pub const family = g.family;

test {
    _ = @import("embgemma2_metal_test.zig");
    _ = g;
}

/// A request's inputs waiting for the GPU: requests that arrive while a batch runs go together in the next one.
const Job = struct {
    ids: []const u32,
    ends: []const u32,
    out: []f32,
    done: bool = false,
    failed: bool = false,
};

/// The jobs from the front of `queue` that fit one batch (at least one), and their rows.
fn take(queue: []const *Job) struct { jobs: usize, rows: usize, inputs: usize } {
    var rows: usize = 0;
    var inputs: usize = 0;
    for (queue, 0..) |j, k| {
        if (k > 0 and (rows + j.ids.len > g.max_rows or inputs + j.ends.len > g.max_inputs)) return .{ .jobs = k, .rows = rows, .inputs = inputs };
        rows += j.ids.len;
        inputs += j.ends.len;
    }
    return .{ .jobs = queue.len, .rows = rows, .inputs = inputs };
}

test "embedding_gemma2 metal: waiting requests join one batch within its limits" {
    const ends = [_]u32{4096};
    const ids: [4096]u32 = @splat(2);
    var out: [g.dim]f32 = undefined;
    var a: Job = .{ .ids = &ids, .ends = &ends, .out = &out };
    var b = a;
    var c = a;
    const t = take(&.{ &a, &b, &c });
    try std.testing.expectEqual(@as(usize, 2), t.jobs); // 8,192 rows: the third waits for the next batch
    try std.testing.expectEqual(@as(usize, g.max_rows), t.rows);
}

const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    model: *Model,
    profile: bool = false, // TF_EG2_PROFILE: each batch's rows and GPU time on stderr
    mutex: std.Io.Mutex = .init, // the GPU: one batch at a time (HTTP threads call submit directly)
    qlock: std.Io.Mutex = .init, // the queue
    queue: std.ArrayList(*Job) = .empty,
    ids: []u32 = &.{}, // a joined batch's rows, ends and vectors
    ends: []u32 = &.{},
    out: []f32 = &.{},

    fn of(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .name = "embedding_gemma2", .text_generation = false, .thinking_budget = false, .context_window = g.context, .embeddings = .{ .dimensions = g.dim, .vocabulary = g.vocab, .max_batch_tokens = g.max_rows, .max_inputs = g.max_inputs, .mrl_dimensions = &g.mrl } };
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        return cache_fit.hostMemory(0, reset_peak);
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = of(ctx);
        const batch = r.embedding orelse return error.InvalidRequest;
        if (r.prompt.len == 0 or r.prompt.len > g.max_rows or batch.ends.len == 0 or batch.ends.len > g.max_inputs) return error.InvalidRequest;
        var job: Job = .{ .ids = r.prompt, .ends = batch.ends, .out = batch.output };
        {
            e.qlock.lockUncancelable(e.io);
            defer e.qlock.unlock(e.io);
            e.queue.append(e.gpa, &job) catch return error.InvalidRequest;
        }
        // whoever holds the GPU next runs every waiting job that fits, this one included or before it
        e.mutex.lockUncancelable(e.io);
        while (!job.done) e.next();
        e.mutex.unlock(e.io);
        const done: api.Event = if (!job.failed) .{ .finished = .{ .reason = .length } } else .{ .finished = .{ .reason = .failed, .message = "embedding inference failed" } };
        sink.event(sink.ctx, id, &done);
    }

    /// One batch of the queue's front jobs (the GPU lock held).
    fn next(e: *Engine) void {
        var jobs: [g.max_inputs]*Job = undefined;
        const t = blk: {
            e.qlock.lockUncancelable(e.io);
            defer e.qlock.unlock(e.io);
            const t = take(e.queue.items);
            @memcpy(jobs[0..t.jobs], e.queue.items[0..t.jobs]);
            e.queue.replaceRangeAssumeCapacity(0, t.jobs, &.{});
            break :blk t;
        };
        const ran = if (t.jobs == 1) e.model.run(jobs[0].ids, jobs[0].ends, jobs[0].out) else joined: {
            var rows: usize = 0;
            var inputs: usize = 0;
            for (jobs[0..t.jobs]) |j| {
                @memcpy(e.ids[rows..][0..j.ids.len], j.ids);
                for (j.ends, 0..) |end, k| e.ends[inputs + k] = @intCast(rows + end);
                rows += j.ids.len;
                inputs += j.ends.len;
            }
            e.model.run(e.ids[0..rows], e.ends[0..inputs], e.out) catch |err| break :joined err;
            inputs = 0;
            for (jobs[0..t.jobs]) |j| {
                @memcpy(j.out[0 .. j.ends.len * g.dim], e.out[inputs * g.dim ..][0 .. j.ends.len * g.dim]);
                inputs += j.ends.len;
            }
        };
        if (ran) |_| {} else |err| std.log.err("embedding_gemma2: the batch failed: {s}", .{@errorName(err)});
        if (e.profile) std.debug.print("embedding_gemma2: {d} request(s), {d} inputs, {d} rows, {d:.3} ms on the GPU\n", .{ t.jobs, t.inputs, t.rows, e.model.gpu_ms });
        for (jobs[0..t.jobs]) |j| {
            j.failed = if (ran) |_| false else |_| true;
            j.done = true;
        }
    }

    fn close(ctx: *anyopaque) void {
        const e = of(ctx);
        e.queue.deinit(e.gpa);
        e.gpa.free(e.ids);
        e.gpa.free(e.ends);
        e.gpa.free(e.out);
        e.model.close();
        e.gpa.destroy(e);
    }

    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory };
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.context) |cx| if (cx < 1 or cx > g.context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: EmbeddingGemma-2 reads 1 to {d} tokens an input", .{ cx, g.context });
        return null;
    };
    g.admit(a, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "embedding_gemma2: config.json is not the released text geometry ({s})", .{@errorName(err)});
        return null;
    };
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .model = undefined };
    e.model = Model.open(gpa, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "embedding_gemma2: the bf16 text encoder did not load on Metal ({s})", .{@errorName(err)});
        return null;
    };
    errdefer e.model.close();
    if (std.c.getenv("TF_EG2_ATTN")) |v| {
        if (std.meta.stringToEnum(Attention, std.mem.span(v))) |x| e.model.attention = x;
    }
    e.profile = std.c.getenv("TF_EG2_PROFILE") != null;
    e.ids = try gpa.alloc(u32, g.max_rows);
    errdefer gpa.free(e.ids);
    e.ends = try gpa.alloc(u32, g.max_inputs);
    errdefer gpa.free(e.ends);
    e.out = try gpa.alloc(f32, g.max_inputs * g.dim);
    errdefer gpa.free(e.out);
    try e.queue.ensureTotalCapacity(gpa, 64);
    if (std.c.getenv("TF_EG2_FEW_ROWS")) |v| e.model.few_rows = @min(em.matvec_rows, std.fmt.parseInt(usize, std.mem.span(v), 10) catch em.matvec_rows);
    std.log.info("EmbeddingGemma-2 on Metal ({s}): {d}-dimensional vectors (MRL {d}, {d}, {d}), up to {d} inputs and {d} tokens a batch, {d} MiB", .{ e.model.device.name(), g.dim, g.mrl[0], g.mrl[1], g.mrl[2], g.max_inputs, g.max_rows, e.model.bytes() >> 20 });
    return .{ .engine = .{ .ctx = e, .vtable = &Engine.vtable }, .close = Engine.close, .ctx = e };
}
