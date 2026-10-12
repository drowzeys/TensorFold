//! The served GLM-5.3 engine behind a C ABI (libtensorfold_glm53.so), for a server that is not TensorFold's own: one
//! rank of the world the TF_GLM53_* variables name. Rank 0 takes requests; every other rank follows it until it stops.
const std = @import("std");
const api = @import("engine_api");
const host = @import("glm53_host");

const gpa = std.heap.c_allocator;

pub const Sampling = extern struct { seed: u64, temperature: f64, top_k: u32, top_p: f64, min_p: f64 };

pub const kind_prefilled: c_int = 0;
pub const kind_tokens: c_int = 1;
pub const kind_finished: c_int = 2;

/// One event of a request; `tokens` and `message` are valid only during the call.
pub const Event = extern struct {
    kind: c_int,
    /// prefilled: prompt tokens that came from a kept prefix
    cached: u32 = 0,
    tokens: ?[*]const u32 = null,
    n: usize = 0,
    /// finished: api.Reason's value (stop, length, cancelled, failed)
    reason: c_int = 0,
    message: ?[*]const u8 = null,
    message_len: usize = 0,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    prefill_seconds: f64 = 0,
};

/// Called on the engine's thread: it must return at once and not call back into the engine.
pub const EventFn = *const fn (ctx: ?*anyopaque, id: u64, event: *const Event) callconv(.c) void;

const State = struct {
    threaded: std.Io.Threaded,
    arena: std.heap.ArenaAllocator,
    opened: api.Opened,
    info: api.Info,
};

/// A request's own copies, kept until its `finished` event.
const Job = struct {
    request: api.Request,
    prompt: []u32,
    eos: []u32,
    event: EventFn,
    ctx: ?*anyopaque,

    fn free(j: *Job) void {
        gpa.free(j.prompt);
        gpa.free(j.eos);
        gpa.destroy(j);
    }
};

fn say(problem: ?[*]u8, cap: usize, text: []const u8) void {
    const out = problem orelse return;
    if (cap == 0) return;
    const n = @min(text.len, cap - 1);
    @memcpy(out[0..n], text[0..n]);
    out[n] = 0;
}

/// Loads this rank (TF_GLM53_RANK of TF_GLM53_WORLD) from `dir` and agrees with the others; `context` 0: the engine's
/// default, `parallel` 0: TF_GLM53_PARALLEL. Null with the reason in `problem`.
export fn tfglm53_open(dir: [*:0]const u8, context: i64, parallel: u32, problem: ?[*]u8, problem_cap: usize) ?*anyopaque {
    const s = gpa.create(State) catch return null;
    s.threaded = .init(gpa, .{});
    s.arena = .init(gpa);
    var why: []const u8 = "";
    const o: api.Open = .{ .dir = std.mem.span(dir), .model_type = host.model_type, .context = if (context > 0) context else null, .lanes = @max(1, parallel), .lanes_fixed = parallel > 0 };
    const opened = host.open(s.arena.allocator(), gpa, s.threaded.io(), o, &why) catch |e| {
        say(problem, problem_cap, @errorName(e));
        return closeState(s);
    } orelse {
        say(problem, problem_cap, why);
        return closeState(s);
    };
    s.opened = opened;
    s.info = opened.engine.info();
    return s;
}

fn closeState(s: *State) ?*anyopaque {
    s.arena.deinit();
    s.threaded.deinit();
    gpa.destroy(s);
    return null;
}

/// 1 on rank 0 (it takes requests), 0 on a rank that follows.
export fn tfglm53_serves(p: *anyopaque) c_int {
    const s: *State = @ptrCast(@alignCast(p));
    return @intFromBool(s.opened.follow == null);
}

/// A following rank's whole life: returns 0 when rank 0 stopped, 1 when following failed.
export fn tfglm53_follow(p: *anyopaque) c_int {
    const s: *State = @ptrCast(@alignCast(p));
    const follow = s.opened.follow orelse return 1;
    follow(s.opened.ctx) catch |e| {
        std.log.err("glm53: following rank 0 stopped: {t}", .{e});
        return 1;
    };
    return 0;
}

export fn tfglm53_context(p: *anyopaque) u32 {
    const s: *State = @ptrCast(@alignCast(p));
    return s.info.context_window;
}

export fn tfglm53_streams(p: *anyopaque) u32 {
    const s: *State = @ptrCast(@alignCast(p));
    return s.info.lanes;
}

/// The widest top_k a sampled request may ask (0: no limit).
export fn tfglm53_top_k_most(p: *anyopaque) u32 {
    const s: *State = @ptrCast(@alignCast(p));
    return s.info.top_k_most;
}

fn onEvent(ctx: *anyopaque, id: api.Id, event: *const api.Event) void {
    const j: *Job = @ptrCast(@alignCast(ctx));
    switch (event.*) {
        .prefilled => |cached| j.event(j.ctx, id, &.{ .kind = kind_prefilled, .cached = cached }),
        .logprobs => {}, // never asked for: submit sets no `logprobs`
        .tokens => |t| j.event(j.ctx, id, &.{ .kind = kind_tokens, .tokens = t.ptr, .n = t.len }),
        .finished => |f| {
            j.event(j.ctx, id, &.{
                .kind = kind_finished,
                .reason = @intFromEnum(f.reason),
                .message = f.message.ptr,
                .message_len = f.message.len,
                .rounds = f.stats.rounds,
                .drafted = f.stats.drafted,
                .accepted = f.stats.accepted,
                .prefill_seconds = f.stats.prefill_seconds orelse 0,
            });
            j.free();
        },
    }
}

/// Queues a reply; its events end with one `finished`. `sampling` null decodes greedily, `drafts` 0 is the serial
/// reference. Returns 0, or 1 closed, 2 busy, 3 invalid, 4 out of memory (no event follows a nonzero return).
export fn tfglm53_submit(p: *anyopaque, id: u64, prompt: [*]const u32, n_prompt: usize, max_tokens: u32, sampling: ?*const Sampling, eos: ?[*]const u32, n_eos: usize, drafts: c_int, event: EventFn, ctx: ?*anyopaque) c_int {
    const s: *State = @ptrCast(@alignCast(p));
    const j = gpa.create(Job) catch return 4;
    const own_prompt = gpa.dupe(u32, prompt[0..n_prompt]) catch {
        gpa.destroy(j);
        return 4;
    };
    const own_eos = gpa.dupe(u32, if (eos) |e| e[0..n_eos] else &.{}) catch {
        gpa.free(own_prompt);
        gpa.destroy(j);
        return 4;
    };
    j.* = .{
        .prompt = own_prompt,
        .eos = own_eos,
        .event = event,
        .ctx = ctx,
        .request = .{
            .prompt = own_prompt,
            .max_tokens = max_tokens,
            .sampling = if (sampling) |x| .{ .seed = x.seed, .temperature = x.temperature, .top_k = x.top_k, .top_p = x.top_p, .min_p = x.min_p } else null,
            .eos = own_eos,
            .drafts = drafts != 0,
        },
    };
    s.opened.engine.submit(id, &j.request, .{ .ctx = j, .event = onEvent }) catch |e| {
        j.free();
        return switch (e) {
            error.Closed => 1,
            error.Busy => 2,
            error.InvalidSpans => 3,
        };
    };
    return 0;
}

/// Ends a request between rounds; its `finished` (cancelled) still arrives.
export fn tfglm53_cancel(p: *anyopaque, id: u64) void {
    const s: *State = @ptrCast(@alignCast(p));
    s.opened.engine.cancel(id);
}

export fn tfglm53_close(p: *anyopaque) void {
    const s: *State = @ptrCast(@alignCast(p));
    s.opened.close(s.opened.ctx);
    _ = closeState(s);
}
