//! A compact checkpoint against the full one under the same mask: picks, weights, hidden and logits, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");

const Model = drama.backbone.Model;
const config = drama.backbone.config;
const top = config.top_k;
const prompt = 64;
const decodes = 16;

/// Every layer's picks and weights in run order, then each run's hidden rows and semantic logits.
const Record = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    routing: std.ArrayList(u8) = .empty,
    finals: std.ArrayList(u8) = .empty,

    fn deinit(r: *Record) void {
        r.routing.deinit(r.gpa);
        r.finals.deinit(r.gpa);
    }

    pub fn routed(r: *Record, s: cuda.Stream, _: usize, rt: drama.backbone.Routed) !void {
        try s.synchronize();
        for ([_]u64{ rt.picks, rt.weights }) |src| try r.take(&r.routing, src, rt.rows * top * 4);
    }

    fn take(r: *Record, to: *std.ArrayList(u8), src: u64, len: usize) !void {
        const view: cuda.DeviceBuffer = .{ .d = r.d, .ptr = src, .len = len, .borrowed = true };
        try view.download(0, try to.addManyAsSlice(r.gpa, len));
    }

    fn rows(r: *Record, out: drama.backbone.Rows) !void {
        try r.take(&r.finals, out.hidden.ptr, out.hidden.len);
        try r.take(&r.finals, out.semantic_logits.ptr, out.semantic_logits.len);
    }
};

/// `dir` loaded under `mask`, its device bytes, and a prompt pass of seeded text tokens then decode rows of them.
fn record(gpu: check.Gpu, dir: []const u8, mask: ?drama.expert_mask.Mask, r: *Record) !usize {
    const before = cuda.usage(false).device;
    // Model.init takes the context mutable; the runner's is only borrowed for this run
    var model = try Model.init(gpu.gpa, gpu.io, @constCast(gpu.ctx), dir, "", .{ .context = prompt + decodes, .max_rows = prompt, .expert_mask = mask });
    defer model.deinit();
    const bytes = cuda.usage(false).device - before;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    try model.reset(s);
    var prng = std.Random.DefaultPrng.init(0xC0A1_EC70);
    var tokens: [prompt + decodes]u32 = undefined;
    for (&tokens) |*t| t.* = prng.random().uintLessThan(u32, model.eos);
    try r.rows(try model.runTraced(s, .prefill, tokens[0..prompt], null, r));
    for (prompt..prompt + decodes) |i| try r.rows(try model.runTraced(s, .decode, tokens[i..][0..1], null, r));
    return bytes;
}

/// `drama-compact FULL COMPACT [MASK]`: FULL under MASK (none: every expert) against COMPACT unmasked.
pub fn run(gpu: check.Gpu, full: []const u8, compact: []const u8, mask_path: ?[]const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var problem: []const u8 = "";
    const mask = if (mask_path) |p| try drama.expert_mask.read(arena.allocator(), gpu.io, p, &problem) orelse {
        std.debug.print("FAIL {s}\n", .{problem});
        return error.TestFailed;
    } else null;
    var a: Record = .{ .gpa = gpu.gpa, .d = gpu.d };
    defer a.deinit();
    var b: Record = .{ .gpa = gpu.gpa, .d = gpu.d };
    defer b.deinit();
    const full_bytes = try record(gpu, full, mask, &a);
    const compact_bytes = try record(gpu, compact, null, &b);
    std.debug.print("RESULT drama-compact: {s} {d} device bytes{s}, {s} {d} device bytes\n", .{ full, full_bytes, if (mask != null) " masked" else "", compact, compact_bytes });
    try check.sameBytes("every layer's picks and routing weights", b.routing.items, a.routing.items);
    try check.sameBytes("hidden rows and semantic logits", b.finals.items, a.finals.items);
    check.pass("drama-compact: {d} rows route, and end, byte for byte as the full checkpoint {s}", .{ prompt + decodes, if (mask != null) "masked" else "does" });
}
