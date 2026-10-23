//! tf-drama-probe's statistics: the kernel against moe_stats.reference, and the real routed hook (synthetic).
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");

const stats = drama.moe_stats;
const config = drama.backbone.config;
const top = config.top_k;
const H = config.hidden;
const I = config.expert_inner;
const band = [2]i32{ 100, 200 };

/// One run's rows: tokens either side of the band, every row's slot 0 on expert 9, the rest spread and repeating.
const Run = struct {
    ids: []i32,
    picks: []i32,
    weights: []f32,
    input: []u16,
    act: []u16, // the SwiGLU activations Sums.add reads, or the hook's gate | up rows
    down: []u16,

    fn init(a: std.mem.Allocator, rng: std.Random, rows: usize) !Run {
        const r: Run = .{
            .ids = try a.alloc(i32, rows),
            .picks = try a.alloc(i32, rows * top),
            .weights = try a.alloc(f32, rows * top),
            .input = try a.alloc(u16, rows * H),
            .act = try a.alloc(u16, rows * top * I),
            .down = try a.alloc(u16, rows * top * H),
        };
        for (r.ids, 0..) |*id, i| id.* = if (i % 3 == 0) 50 else 100 + @as(i32, @intCast(i * 17 % 101));
        for (r.picks, r.weights, 0..) |*p, *w, i| {
            p.* = if (i % top == 0) 9 else @intCast((i / top * 37 + i % top * 29) % config.experts);
            w.* = rng.float(f32);
        }
        for ([_][]u16{ r.input, r.act, r.down }) |values| for (values) |*v| {
            v.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * 16)) >> 16);
        };
        return r;
    }

    fn routed(r: Run, gpu: check.Gpu, buffers: *[6]cuda.DeviceBuffer) !drama.backbone.Routed {
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(r.ids), std.mem.sliceAsBytes(r.picks), std.mem.sliceAsBytes(r.weights), std.mem.sliceAsBytes(r.input), std.mem.sliceAsBytes(r.act), std.mem.sliceAsBytes(r.down) };
        for (buffers, bytes) |*b, x| b.* = try cuda.DeviceBuffer.fromHost(gpu.d, x);
        return .{ .rows = r.ids.len, .ids = buffers[0].ptr, .picks = buffers[1].ptr, .weights = buffers[2].ptr, .input = buffers[3].ptr, .gate_up = buffers[4].ptr, .down = buffers[5].ptr };
    }
};

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng: std.Random.DefaultPrng = .init(20261018);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var sums = try stats.Sums.init(gpu.d, band[0], band[1]);
    defer sums.deinit();
    const layer = 7;
    const want = try a.alloc(f64, config.layers * stats.layer_len);
    @memset(want, 0);
    // a prompt chunk of 7 rows, then 3 generated rows, as the probe adds them
    for ([_]usize{ 7, 3 }, [_]bool{ false, true }) |rows, generated| {
        const r = try Run.init(a, prng.random(), rows);
        var buffers: [6]cuda.DeviceBuffer = undefined;
        const routed = try r.routed(gpu, &buffers);
        defer for (&buffers) |*b| b.free();
        try sums.add(stream, layer, routed, buffers[4].ptr, generated);
        try stream.synchronize();
        stats.reference(want[layer * stats.layer_len ..][0..stats.layer_len], r.ids, band, generated, r.picks, r.weights, r.input, r.act, r.down);
    }
    // an NVFP4 layer: unweighted fp32 down rows, each pair's bf16(w * d) formed by the hook as the combine rounds it
    const fp4_layer = 8;
    {
        const rng = prng.random();
        const r = try Run.init(a, rng, 5);
        const down = try a.alloc(f32, r.down.len);
        for (down, r.down, 0..) |*d, *b, i| {
            d.* = (rng.float(f32) - 0.5) * 16;
            b.* = toBf16(r.weights[i / H] * d.*);
        }
        var buffers: [6]cuda.DeviceBuffer = undefined;
        var routed = try r.routed(gpu, &buffers);
        defer for (&buffers) |*b| b.free();
        var f32_down = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(down));
        defer f32_down.free();
        var weighed = try cuda.DeviceBuffer.alloc(gpu.d, r.down.len * 2);
        defer weighed.free();
        routed.gate_up = 0;
        routed.down = 0;
        routed.act = buffers[4].ptr;
        routed.down_f32 = f32_down.ptr;
        try sums.addFp4(stream, fp4_layer, routed, weighed.ptr, false);
        try stream.synchronize();
        const rows = try check.download(gpu, weighed);
        defer gpu.gpa.free(rows);
        try check.sameBytes("NVFP4 pair rows bf16(w * d) against the host's", rows, std.mem.sliceAsBytes(r.down));
        stats.reference(want[fp4_layer * stats.layer_len ..][0..stats.layer_len], r.ids, band, false, r.picks, r.weights, r.input, r.act, r.down);
    }
    const got = try a.alloc(f64, want.len);
    @memset(got, 0);
    try sums.drain(stream, got, try a.alloc(f64, want.len));
    try check.sameBytes("statistics after two runs", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    const pairs = got[layer * stats.layer_len + stats.count_at ..][0 .. 3 * config.experts];
    var counted: f64 = 0;
    for (pairs) |c| counted += c;
    try check.expect(counted == 10 * top, "{d} routed pairs counted, expected {d}", .{ counted, 10 * top });
    const cleared = try check.download(gpu, sums.buf);
    defer gpu.gpa.free(cleared);
    try check.expect(std.mem.allEqual(u8, cleared, 0), "drain leaves the device sums nonzero", .{});
    check.pass("drama-probe-stats: 10 FP8-layer and 5 NVFP4-layer rows, every layer's sums bit-equal to the CPU reference; drain clears", .{});
    try hook(gpu);
    try hookFp4(gpu);
}

fn bf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Round to nearest even bf16 (finite inputs).
fn toBf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Tracer.routed on synthetic gate | up rows: activations and sums against the host's, forward buffers only read.
pub fn hook(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng: std.Random.DefaultPrng = .init(20261019);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var k = try drama.backbone.kernels.Kernels.load(gpu.gpa, gpu.io, gpu.d, "");
    defer k.unload();
    const rows = 5;
    const gu = drama.backbone.weights.gu_rows;
    var t = try stats.Tracer.init(gpu.d, &k, .{ band[0], band[1] }, rows);
    defer t.deinit();
    try t.act.fill8(0xff, null); // bf16 NaNs: an activation the hook never writes cannot match
    var r = try Run.init(a, prng.random(), rows);
    r.act = try a.alloc(u16, rows * top * gu);
    for (r.act) |*v| v.* = toBf16((prng.random().float(f32) - 0.5) * 12);
    var buffers: [6]cuda.DeviceBuffer = undefined;
    const routed = try r.routed(gpu, &buffers);
    defer for (&buffers) |*b| b.free();
    const layer = 3;
    try t.routed(stream, layer, routed);
    const got = try a.alloc(f64, config.layers * stats.layer_len);
    @memset(got, 0);
    try t.sums.drain(stream, got, try a.alloc(f64, got.len));

    const originals = [_][]const u8{ std.mem.sliceAsBytes(r.ids), std.mem.sliceAsBytes(r.picks), std.mem.sliceAsBytes(r.weights), std.mem.sliceAsBytes(r.input), std.mem.sliceAsBytes(r.act), std.mem.sliceAsBytes(r.down) };
    const names = [_][]const u8{ "ids", "picks", "routing weights", "MoE input", "gate | up", "down" };
    for (buffers, originals, names) |b, want, name| {
        const after = try check.download(gpu, b);
        defer gpu.gpa.free(after);
        try check.sameBytes(name, after, want);
    }
    // drama_silu_mul: h = bf16(silu(bf16 g) * bf16 u), silu(x) = x / (1 + exp(-x)) in f32
    const want_act = try a.alloc(u16, rows * top * I);
    for (0..rows * top) |p| for (0..I) |c| {
        const g = bf16(r.act[p * gu + c]);
        const u = bf16(r.act[p * gu + I + c]);
        want_act[p * I + c] = toBf16(g / (1 + @exp(-g)) * u);
    };
    const act_bytes = try check.download(gpu, .{ .d = gpu.d, .ptr = t.act.ptr, .len = want_act.len * 2, .borrowed = true });
    defer gpu.gpa.free(act_bytes);
    const got_act = std.mem.bytesAsSlice(u16, act_bytes);
    var exact: usize = 0;
    for (got_act, want_act) |x, y| {
        // the device's expf may differ from the host's by an ulp: at most one bf16 step, never a sign or a NaN
        try check.expect(x >> 15 == y >> 15 and @max(x, y) - @min(x, y) <= 1, "activation {x} against the host's {x}", .{ x, y });
        exact += @intFromBool(x == y);
    }
    const want = try a.alloc(f64, got.len);
    @memset(want, 0);
    stats.reference(want[layer * stats.layer_len ..][0..stats.layer_len], r.ids, band, false, r.picks, r.weights, r.input, want_act, r.down);
    const down = layer * stats.layer_len + stats.down_at;
    try check.sameBytes("count, mass, reap and gate_up sums", std.mem.sliceAsBytes(got[0..down]), std.mem.sliceAsBytes(want[0..down]));
    // one bf16 step changes a square by under 2^-6 of itself, so each column sum stays within 2^-6
    var worst: f64 = 0;
    for (got[down..][0 .. 3 * config.experts * I], want[down..][0 .. 3 * config.experts * I]) |x, y| {
        try check.expect(@abs(x - y) <= y / 64, "down.sumsq {d} against the host's {d}", .{ x, y });
        if (y > 0) worst = @max(worst, @abs(x - y) / y);
    }
    try check.sameBytes("other layers", std.mem.sliceAsBytes(got[down + 3 * config.experts * I ..]), std.mem.sliceAsBytes(want[down + 3 * config.experts * I ..]));
    check.pass("drama-probe-stats hook: forward buffers unchanged; {d} of {d} activations bit-equal to the host's, the rest one bf16 step; down.sumsq worst relative {e}", .{ exact, want_act.len, worst });
}

/// Tracer.routed on an NVFP4 layer's rows: its own SwiGLU and f32 down rows, sums bit-equal, forward buffers only read.
pub fn hookFp4(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng: std.Random.DefaultPrng = .init(20261020);
    const rng = prng.random();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var k = try drama.backbone.kernels.Kernels.load(gpu.gpa, gpu.io, gpu.d, "");
    defer k.unload();
    try k.loadFp4(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    const rows = 5;
    var t = try stats.Tracer.init(gpu.d, &k, .{ band[0], band[1] }, rows);
    defer t.deinit();
    try check.expect(t.weighed != null, "a tracer over NVFP4 kernels holds the weighed rows", .{});
    const r = try Run.init(a, rng, rows);
    const down = try a.alloc(f32, r.down.len);
    for (down, r.down, 0..) |*d, *b, i| {
        d.* = (rng.float(f32) - 0.5) * 16;
        b.* = toBf16(r.weights[i / H] * d.*);
    }
    var buffers: [6]cuda.DeviceBuffer = undefined;
    var routed = try r.routed(gpu, &buffers);
    defer for (&buffers) |*b| b.free();
    var f32_down = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(down));
    defer f32_down.free();
    routed.gate_up = 0;
    routed.down = 0;
    routed.act = buffers[4].ptr;
    routed.down_f32 = f32_down.ptr;
    const layer = 5;
    try t.routed(stream, layer, routed);
    const got = try a.alloc(f64, config.layers * stats.layer_len);
    @memset(got, 0);
    try t.sums.drain(stream, got, try a.alloc(f64, got.len));
    const originals = [_][]const u8{ std.mem.sliceAsBytes(r.ids), std.mem.sliceAsBytes(r.picks), std.mem.sliceAsBytes(r.weights), std.mem.sliceAsBytes(r.input), std.mem.sliceAsBytes(r.act) };
    const names = [_][]const u8{ "ids", "picks", "routing weights", "MoE input", "act" };
    for (buffers[0..5], originals, names) |b, want, name| {
        const after = try check.download(gpu, b);
        defer gpu.gpa.free(after);
        try check.sameBytes(name, after, want);
    }
    const after = try check.download(gpu, f32_down);
    defer gpu.gpa.free(after);
    try check.sameBytes("f32 down", after, std.mem.sliceAsBytes(down));
    const want = try a.alloc(f64, got.len);
    @memset(want, 0);
    stats.reference(want[layer * stats.layer_len ..][0..stats.layer_len], r.ids, band, false, r.picks, r.weights, r.input, r.act, r.down);
    try check.sameBytes("NVFP4 layer sums through the hook", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    check.pass("drama-probe-stats NVFP4 hook: forward buffers unchanged; sums of the act and bf16(w * d) rows bit-equal to the host's", .{});
}
