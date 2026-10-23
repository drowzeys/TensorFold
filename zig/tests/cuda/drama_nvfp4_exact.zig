//! Bit-exact NVFP4 MoE checks: Drama's SwiGLU epilogue on sparse exact projections, and the weighted combine.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");
const ref = @import("drama_nvfp4_ref.zig");
const h = @import("quant_ref_tests.zig").helpers;

const Model = drama.backbone.Model;
const H = ref.H;
const I = ref.I;
const E = ref.E;
const top = ref.top;
const rows = ref.rows;

/// Every expert alike: gate row j reads input 2j, up row j input 2j + 1, at weight 1; down is zero.
pub fn sparse(a: std.mem.Allocator, dense: ref.Synth) !ref.Synth {
    var f: [3]ref.Fp4 = .{ try .alloc(a, I, H), try .alloc(a, I, H), try .alloc(a, H, I) };
    for (&f) |*p| {
        @memset(p.codes, 0);
        @memset(p.scales, 0x38);
        @memset(p.global, 1);
    }
    for (0..E) |e| for (0..I) |j| {
        f[0].codes[(e * I + j) * (H / 2) + j] = 0x02; // input 2j: the low nibble
        f[1].codes[(e * I + j) * (H / 2) + j] = 0x20; // input 2j + 1: the high nibble
    };
    var s = dense;
    s.experts = .{ .fp4 = f };
    return s;
}

/// Drama's epilogue in f32: silu of the bf16 gate by exp, add and divide, times the bf16 up, before one bf16 rounding.
fn swiglu(g: f32, u: f32) f32 {
    return g / (1 + @exp(-g)) * u;
}

/// Whether `p` lies at least 64 f32 steps from a bf16 rounding midpoint: room for CUDA expf's 2-ulp error and the rest.
fn clear(p: f32) bool {
    const low: i32 = @intCast(@as(u32, @bitCast(p)) & 0xFFFF);
    return @abs(low - 0x8000) >= 64;
}

fn val(b: u16) f32 {
    return @floatCast(h.bfOf(b));
}

/// The sparse layer's 128-row replay: pair p's Scratch.act[j] must be bf16(swiglu(x[2j], x[2j + 1])) of its row.
pub fn epilogue(gpu: check.Gpu, a: std.mem.Allocator, m: *Model, s: cuda.Stream, layer: usize, io: *ref.Pinned, rng: std.Random) !void {
    // a reviewer's case first: 0.5 and 0.50390625 give Drama 0.1572265625, an extra silu rounding 0.15625
    try check.expect(clear(swiglu(0.5, 0.50390625)) and h.bf16Of(swiglu(0.5, 0.50390625)) == 0x3E21, "a reviewer's discriminator", .{});
    const x = try ref.bfs(a, rng, rows * H, 1.0);
    var generic: usize = 0;
    for (0..rows) |r| for (0..I) |j| {
        var g: u16 = 0x3F00;
        var u: u16 = 0x3F01;
        while (r + j > 0) {
            g = h.bf16Of((rng.float(f32) * 2 - 1) * 8);
            u = h.bf16Of((rng.float(f32) * 2 - 1) * 4);
            if (clear(swiglu(val(g), val(u)))) break;
        }
        x[r * H + 2 * j] = g;
        x[r * H + 2 * j + 1] = u;
        const extra = h.bf16Of(val(h.bf16Of(val(g) / (1 + @exp(-val(g))))) * val(u));
        generic += @intFromBool(extra != h.bf16Of(swiglu(val(g), val(u))));
    };
    @memcpy(io.in.bytes, std.mem.sliceAsBytes(x));
    try ref.replay(m, s, layer, &.{rows}, io.in.bytes, io.outs[0]);
    const got = try a.alloc(u16, rows * top * I);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = m.s.act, .len = got.len * 2, .borrowed = true }).download(0, std.mem.sliceAsBytes(got));
    var differ: usize = 0;
    var first: usize = got.len;
    for (got, 0..) |v, i| {
        const row = x[(i / (top * I)) * H ..];
        const j = i % I;
        const want = h.bf16Of(swiglu(val(row[2 * j]), val(row[2 * j + 1])));
        if (v != want) {
            differ += 1;
            first = @min(first, i);
        }
    }
    std.debug.print("RESULT NVFP4 epilogue: {d} of {d} activations differ from bf16(silu(bf16 g) * bf16 u) (first {d}); an extra silu rounding would change {d} of {d} (g, u)\n", .{ differ, got.len, first, generic, rows * I });
    try check.expect(differ == 0, "NVFP4 epilogue: Scratch.act is Drama's SwiGLU bit for bit", .{});
    try check.expect(generic > 0, "NVFP4 epilogue: the fixture tells Drama's rounding from the generic epilogue's", .{});
    check.pass("NVFP4 epilogue: {d} activations bit-equal to Drama's SwiGLU rounding", .{got.len});
}

/// drama_moe_combine_f32 on cancelling unequal-weight pair rows: bit-equal to the f32 emulation of its contract.
pub fn combine(gpu: check.Gpu, a: std.mem.Allocator, m: *Model, s: cuda.Stream, rng: std.Random) !void {
    const n = 16;
    const pairs = n * top;
    const down = try a.alloc(f32, pairs * H);
    const w = try a.alloc(f32, pairs);
    for (0..pairs) |p| {
        w[p] = if (p % 2 == 1) w[p - 1] * (1 + 0x1p-9 * rng.float(f32)) else 0.01 + rng.float(f32) * 0.5;
        for (down[p * H ..][0..H], 0..) |*d, c| d.* = if (p % 2 == 1)
            -down[(p - 1) * H + c] * (1 + 0x1p-12 * rng.float(f32)) // the pair before it, nearly cancelled
        else
            (rng.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), @as(i32, rng.intRangeAtMost(i32, -8, 8)));
    }
    const x = try ref.bfs(a, rng, n * H, 1.0);
    const shared = try ref.bfs(a, rng, n * H, 4.0);
    const wg = try a.alloc(u16, H);
    @memset(wg, 0); // gate = sigmoid(0) = 0.5 exactly: the shared addition is exact to emulate
    var bufs: [6]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for ([_][]const u8{ std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(x), std.mem.sliceAsBytes(wg), std.mem.sliceAsBytes(shared) }) |bytes| {
        bufs[made] = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
        made += 1;
    }
    bufs[made] = try cuda.DeviceBuffer.alloc(gpu.d, n * H * 2);
    made += 1;
    try m.k.combineWeighted(s, bufs[0].ptr, bufs[1].ptr, top, bufs[2].ptr, bufs[3].ptr, bufs[4].ptr, bufs[5].ptr, n, H);
    try s.synchronize();
    const got = try a.alloc(u16, n * H);
    try bufs[5].download(0, std.mem.sliceAsBytes(got));
    var differ: usize = 0;
    for (got, 0..) |v, i| {
        const r = i / H;
        const c = i % H;
        var acc: f32 = 0;
        for (0..top) |slot| acc += @floatCast(h.bfOf(h.bf16Of(w[r * top + slot] * down[(r * top + slot) * H + c])));
        const routed: f32 = @floatCast(h.bfOf(h.bf16Of(acc)));
        differ += @intFromBool(v != h.bf16Of(0.5 * @as(f32, @floatCast(h.bfOf(shared[i]))) + routed));
    }
    std.debug.print("RESULT NVFP4 combine: {d} of {d} outputs differ from f32 product, bf16 pair, f32 slot-order sum, bf16, shared add\n", .{ differ, got.len });
    try check.expect(differ == 0, "NVFP4 combine: bit-equal to its rounding contract", .{});
    check.pass("NVFP4 combine: {d} rows of cancelling unequal-weight pairs bit-equal to the contract", .{n});
}
