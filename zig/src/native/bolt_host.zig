//! Nemotron 3.5 Bolt 74B-A9B on Metal: libbolt.dylib (zig/bolt-metal) under the lane host and scheduler, so the round
//! loop drives verify windows, keeps and LiLiCorr drafts as on CUDA. The recipe is MiaAI-Lab's
//! (Nemotron-3.5-Bolt-TensorFold-DGX-Spark); the backend is families/nemotron/bolt_metal.zig.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const bolt = tf.bolt;
const lanes = tf.lanes;
const cache_fit = @import("cache_fit.zig");

pub const model_type = bolt.model_type;

const Host = struct {
    gpa: std.mem.Allocator,
    loaded: bolt.Loaded,
    config: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    cache: ?api.prompt_cache.Store = null, // kept prompt states (engine thread only); null: a zero budget

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        if (h.cache) |*store| store.deinit();
        h.core.deinit();
        h.config.deinit(h.gpa);
        h.loaded.deinit(h.loaded.ctx);
        h.gpa.destroy(h);
    }
};

const Snaps = struct {
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
        return bolt.snap_save(ptr, owner, at);
    }
};

pub fn open(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.vision != null) {
        problem.* = "--vision: Bolt on Metal reads text only; serve it without --vision";
        return null;
    }
    const window: i64 = o.context orelse bolt.default_context;
    if (window < 1 or window > bolt.default_context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: Bolt serves 1 to {d} tokens a stream", .{ window, bolt.default_context });
        return null;
    }
    const lanes_n: u32 = @max(o.lanes, 1);
    if (lanes_n > 16) {
        problem.* = "--parallel: Bolt runs at most 16 streams together";
        return null;
    }
    const loaded = bolt.open(gpa, o.dir, .{ .context = @intCast(window), .drafts = o.drafts, .lanes = lanes_n }) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the Metal Bolt engine cannot load {s} ({s})", .{ o.dir, bolt.explain(null, e) orelse @errorName(e) });
        return null;
    };
    errdefer loaded.deinit(loaded.ctx);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.loaded = loaded;
    h.config = try lanes.Config.init(gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    errdefer h.config.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.config, loaded.backend, h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = lanes_n, .context_window = @intCast(window), .prefill_step = bolt.prompt_rows, .structures = true });
    h.cache = null;
    var why: []const u8 = "";
    const dev = try mtl.Device.init();
    defer dev.deinit();
    const plan = cache_fit.budget(o.prompt_cache_gib, o.prompt_cache_over_cap, dev.maxWorkingSet() -| dev.allocated() -| (8 << 30), a, &why, "") catch |e| {
        if (e == error.CacheOverCap) problem.* = why;
        return e;
    };
    h.host.info_.prompt_cache_plan = plan;
    if (plan.budget_bytes > 0) {
        h.cache = api.prompt_cache.Store.init(gpa, .{
            .ptr = loaded.target,
            .vtable = &.{ .bytes = bolt.snap_bytes, .save = Snaps.save, .restore = bolt.snap_restore, .drop = bolt.snap_drop },
        }, .{ .lookahead = bolt.snap_lookahead, .min_prompt = 0, .warm = .waits }, plan.budget_bytes);
        h.host.cache = &h.cache.?;
        h.host.info_.prompt_cache = true;
        // after a reply, the conversation and the next turn's opening prefill once the engine has been quiet 150 ms, so the
        // next turn resumes past the reply (the server's warm turns; CUDA Bolt has them through its family's warm_quiet)
        h.host.info_.warm_turns = true;
        h.host.warm_quiet = 150 * std.time.ns_per_ms;
    }
    errdefer if (h.cache) |*store| store.deinit();
    if (bolt.canTick(loaded.ctx)) h.host.keepalive_target = .{ .ctx = loaded.ctx, .tick = bolt.keepTick };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
