//! FP8 weights (e4m3 codes, a scale_inv a 128 x 128 block or a row's 64 inputs) on the host: codes, encoding, fp64.
const std = @import("std");
const types = @import("types.zig");
const convert = @import("convert.zig");

const Tensor = types.Tensor;

/// Rows and columns one scale_inv entry covers.
pub const block = 128;

/// Inputs of a row one per-row scale_inv entry covers (the fp8g lane's group).
pub const row_group = 64;

/// An e4m3 code's value; 0x7f and 0xff are NaN, as the GPU's conversion reads them.
pub fn e4m3(b: u8) f64 {
    if (b & 0x7f == 0x7f) return std.math.nan(f64);
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xF;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8.0 * std.math.pow(f64, 2, -6) else (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// x rounded to the nearest finite e4m3 value on its exact binary grid (ties to even), saturating at 448.
pub fn toE4m3(x: f64) f64 {
    std.debug.assert(!std.math.isNan(x));
    const a = @abs(x);
    if (a >= 448) return std.math.copysign(@as(f64, 448), x);
    if (a == 0) return x;
    // a's binade has the step 2^(e - 3); below 2^-6 the subnormals' 2^-9
    const e = @max(std.math.frexp(a).exponent - 1, -6);
    const t = std.math.ldexp(a, 3 - e); // exact, as is its fraction below
    const whole = @floor(t);
    const frac = t - whole;
    const q = if (frac > 0.5 or (frac == 0.5 and @mod(whole, 2) == 1)) whole + 1 else whole;
    return std.math.copysign(std.math.ldexp(q, e - 3), x);
}

/// The code of an e4m3 value (toE4m3's result), with no further rounding; both zeros are 0x00.
pub fn e4m3Code(v: f64) u8 {
    const a = @abs(v);
    if (a == 0) return 0;
    const e = @max(std.math.frexp(a).exponent - 1, -6);
    const m = std.math.ldexp(a, 3 - e);
    std.debug.assert(a <= 448 and m == @floor(m));
    const field: u8 = @intFromFloat(m);
    const bits = if (m < 8) field else @as(u8, @intCast(e + 7)) << 3 | (field - 8);
    return if (v < 0) 0x80 | bits else bits;
}

/// Stored e4m3 [..., N, K] and fp32/bf16/fp16 scales [..., ceil(N / 128), ceil(K / 128)].
pub const Host = struct { weight: Tensor, scale_inv: Tensor };

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel();
}

/// fp64 dequantization of `h`'s rows (a stack's leading dimensions folded), code * scale_inv, into `out` [rows, K].
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale_inv.rank != rank or h.weight.dtype != .f8_e4m3 or !convert.tableDtype(h.scale_inv.dtype)) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1];
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale_inv.shape[d]) return error.UnexpectedTensor;
    const nb = h.scale_inv.shape[rank - 2];
    const kb = h.scale_inv.shape[rank - 1];
    if (nb != (n + block - 1) / block or kb != (k + block - 1) / block or out.len != h.weight.numel()) return error.UnexpectedTensor;
    const stack = h.weight.numel() / (n * k);
    if (h.scale_inv.numel() != stack * nb * kb) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const at = (e * n + r) * k + c;
        const s = convert.load(h.scale_inv.dtype, h.scale_inv.bytes, (e * nb + r / block) * kb + c / block);
        out[at] = e4m3(h.weight.bytes[at]) * s;
    };
}

/// C6 over bf16 W [n][k] (`k` a multiple of row_group) into e4m3 codes [n][k] and scale_inv [n][k / row_group].
pub fn encodeRows(w: []const u8, n: usize, k: usize, codes: []u8, scale_inv: []f32) error{NonFiniteWeight}!void {
    std.debug.assert(k % row_group == 0 and w.len == 2 * n * k and codes.len == n * k and scale_inv.len == n * k / row_group);
    for (scale_inv, 0..) |*s, g| {
        const at = g * row_group;
        var a: f64 = 0;
        for (at..at + row_group) |i| {
            const x: f64 = convert.load(.bf16, w, i);
            if (!std.math.isFinite(x)) return error.NonFiniteWeight;
            a = @max(a, @abs(x));
        }
        s.* = if (a == 0) 1 else @floatCast(a / 448);
        // a finite bf16 amax over 448 stays a positive finite f32, the least a subnormal: no scale to refuse here
        std.debug.assert(std.math.isFinite(s.*) and s.* > 0);
        for (codes[at..][0..row_group], at..) |*c, i| c.* = e4m3Code(toE4m3(convert.load(.bf16, w, i) / @as(f64, s.*)));
    }
}

pub const RowError = error{ UnexpectedTensor, NanCode, BadScale };

/// fp64 dequantization of row-FP8 W (e4m3 [n][k], f32 scale_inv [n][k / row_group]) into `out` [n][k].
pub fn referenceRows(h: Host, out: []f64) RowError!void {
    if (h.weight.dtype != .f8_e4m3 or h.weight.rank != 2) return error.UnexpectedTensor;
    const n, const k = .{ h.weight.shape[0], h.weight.shape[1] };
    if (n == 0 or k == 0 or k % row_group != 0 or !h.scale_inv.is(.f32, &.{ n, k / row_group }) or out.len != n * k) return error.UnexpectedTensor;
    for (0..n * k / row_group) |g| {
        const s = convert.load(.f32, h.scale_inv.bytes, g);
        if (!std.math.isFinite(s) or s <= 0) return error.BadScale;
        for (g * row_group..(g + 1) * row_group) |i| {
            const b = h.weight.bytes[i];
            // the lane's conversion has no NaN branch: it would read these as +-480
            if (b & 0x7f == 0x7f) return error.NanCode;
            out[i] = e4m3(b) * s;
        }
    }
}

test "e4m3 codes and their rounding" {
    try std.testing.expectEqual(@as(f64, 1), e4m3(0x38));
    try std.testing.expectEqual(@as(f64, -2), e4m3(0xc0));
    try std.testing.expectEqual(@as(f64, 448), e4m3(0x7e));
    try std.testing.expectEqual(std.math.pow(f64, 2, -9), e4m3(0x01));
    try std.testing.expect(std.math.isNan(e4m3(0x7f)) and std.math.isNan(e4m3(0xff)));
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f) continue;
        try std.testing.expectEqual(e4m3(b), toE4m3(e4m3(b)));
    }
    try std.testing.expectEqual(@as(f64, 1), toE4m3(1.0625)); // halfway between 1 and 1.125: to even
    try std.testing.expectEqual(@as(f64, 1.25), toE4m3(1.1875));
    try std.testing.expectEqual(@as(f64, -448), toE4m3(-1e6));
}

/// The nearest finite code to x by brute force over all 254, ties to the even code, zero as 0x00: toE4m3's oracle.
fn nearest(x: f64) u8 {
    var best: u8 = 0;
    for (1..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f or b == 0x80) continue;
        const d, const bd = .{ @abs(x - e4m3(b)), @abs(x - e4m3(best)) };
        if (d < bd or (d == bd and b & 1 == 0 and best & 1 == 1)) best = b;
    }
    return best;
}

test "e4m3 rounding is the nearest finite code, ties to even, against a brute-force oracle" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f) continue;
        try std.testing.expectEqual(if (b == 0x80) 0 else b, e4m3Code(toE4m3(e4m3(b))));
    }
    // every midpoint between neighbours (464 between 448 and the 480 the NaN code would be), and the doubles beside it
    for (0..127) |i| {
        const lo = e4m3(@intCast(i));
        const mid = if (i == 126) 464 else (lo + e4m3(@intCast(i + 1))) / 2;
        for ([_]f64{ std.math.nextAfter(f64, mid, 0), mid, std.math.nextAfter(f64, mid, 1e9) }) |x| {
            try std.testing.expectEqual(nearest(x), e4m3Code(toE4m3(x)));
            try std.testing.expectEqual(nearest(-x), e4m3Code(toE4m3(-x)));
        }
    }
    try std.testing.expectEqual(@as(f64, 1.125), toE4m3(1.0625000000000036)); // an epsilon tie test made this 1
    var prng = std.Random.DefaultPrng.init(1);
    const rnd = prng.random();
    for (0..20000) |_| {
        const x = (rnd.float(f64) * 2 - 1) * std.math.ldexp(@as(f64, 1), rnd.intRangeAtMost(i32, -14, 10));
        try std.testing.expectEqual(nearest(x), e4m3Code(toE4m3(x)));
    }
    for ([_]f64{ 480, 1e6, std.math.inf(f64) }) |x| {
        try std.testing.expectEqual(@as(u8, 0x7e), e4m3Code(toE4m3(x)));
        try std.testing.expectEqual(@as(u8, 0xfe), e4m3Code(toE4m3(-x)));
    }
}

fn rowHost(codes: []const u8, inv: []const f32, n: usize, k: usize) Host {
    return .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = codes },
        .scale_inv = .{ .dtype = .f32, .rank = 2, .shape = .{ n, k / row_group, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(inv) },
    };
}

test "row FP8 reconstructs its stored-scale grid exactly and takes each code by brute force" {
    const n, const k = .{ 3, 2 * row_group };
    var prng = std.Random.DefaultPrng.init(2);
    const rnd = prng.random();
    var w: [2 * n * k]u8 = undefined;
    // row 0's groups lie on the grids of scales 2^-4 and 0.75 (amax 28 and 336); rows 1 and 2 are arbitrary bf16
    for (0..n * k) |i| {
        const s: f64 = if (i < row_group) 0.0625 else 0.75;
        const x: f64 = if (i >= k) rnd.floatNorm(f64) * 0.02 else if (i % row_group == 5) 448 * s else e4m3(rnd.uintLessThan(u8, 0x7f) | (rnd.int(u8) & 0x80)) * s;
        std.mem.writeInt(u16, w[2 * i ..][0..2], convert.bf16(@floatCast(x)), .little);
    }
    var codes: [n * k]u8 = undefined;
    var inv: [n * k / row_group]f32 = undefined;
    try encodeRows(&w, n, k, &codes, &inv);
    try std.testing.expectEqualSlices(f32, &.{ 0.0625, 0.75 }, inv[0..2]);
    var out: [n * k]f64 = undefined;
    try referenceRows(rowHost(&codes, &inv, n, k), &out);
    for (out[0..k], 0..) |q, i| try std.testing.expectEqual(@as(f64, convert.load(.bf16, &w, i)), q);
    for (k..n * k) |i| {
        const g = i / row_group;
        var a: f64 = 0;
        for (g * row_group..(g + 1) * row_group) |j| a = @max(a, @abs(convert.load(.bf16, &w, j)));
        try std.testing.expectEqual(@as(f32, @floatCast(a / 448)), inv[g]);
        try std.testing.expectEqual(nearest(convert.load(.bf16, &w, i) / @as(f64, inv[g])), codes[i]);
        try std.testing.expectEqual(e4m3(codes[i]) * inv[g], out[i]);
    }
}

/// One group whose first input is the bf16 `first`, the rest `rest`, encoded.
fn group(first: u16, rest: u16, codes: *[row_group]u8, inv: *[1]f32) !void {
    var w: [2 * row_group]u8 = undefined;
    for (0..row_group) |i| std.mem.writeInt(u16, w[2 * i ..][0..2], if (i == 0) first else rest, .little);
    try encodeRows(&w, 1, row_group, codes, inv);
}

test "row FP8 saturation, zero and extreme groups, and what it refuses" {
    var codes: [row_group]u8 = undefined;
    var inv: [1]f32 = undefined;
    var out: [row_group]f64 = undefined;
    // amax 1: the scale rounds up and 1 reconstructs as 1.0000000447; a scale rounded down saturates its amax to 0x7e
    try group(0x3f80, 0, &codes, &inv);
    try std.testing.expectEqual(@as(f32, 0.0022321429569274187), inv[0]);
    try std.testing.expectEqual(@as(u8, 0x7e), codes[0]);
    var bits: u16 = 0x3f81;
    while (true) : (bits += 1) {
        try group(bits, 0, &codes, &inv);
        if (convert.load(.bf16, std.mem.asBytes(&bits), 0) / @as(f64, inv[0]) > 448) break;
    }
    try std.testing.expectEqual(@as(u8, 0x7e), codes[0]);
    try group(bits | 0x8000, 0, &codes, &inv);
    try std.testing.expectEqual(@as(u8, 0xfe), codes[0]);
    try group(0x8000, 0, &codes, &inv);
    try std.testing.expectEqual(@as(f32, 1), inv[0]);
    try std.testing.expect(std.mem.allEqual(u8, &codes, 0));
    // the least positive bf16 alone gives a positive subnormal scale; the greatest a finite one, beside which 1 is 0
    for ([_][2]u16{ .{ 0x0001, 0 }, .{ 0x7f7f, 0x3f80 } }) |g| {
        try group(g[0], g[1], &codes, &inv);
        try std.testing.expect(inv[0] > 0 and std.math.isFinite(inv[0]));
        try referenceRows(rowHost(&codes, &inv, 1, row_group), &out);
        for (out) |x| try std.testing.expect(std.math.isFinite(x));
        try std.testing.expect(out[0] > 0);
        if (g[0] == 0x0001) try std.testing.expect(inv[0] < std.math.floatMin(f32)) else try std.testing.expect(out[1] == 0);
    }
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |b| try std.testing.expectError(error.NonFiniteWeight, group(0x3f80, b, &codes, &inv));
    try group(0x3f80, 0, &codes, &inv);
    for ([_]f32{ 0, -0.0, -1, std.math.inf(f32), std.math.nan(f32) }) |s| {
        try std.testing.expectError(error.BadScale, referenceRows(rowHost(&codes, &.{s}, 1, row_group), &out));
    }
    for ([_]u8{ 0x7f, 0xff }) |b| {
        codes[3] = b;
        try std.testing.expectError(error.NanCode, referenceRows(rowHost(&codes, &inv, 1, row_group), &out));
    }
    try std.testing.expectError(error.UnexpectedTensor, referenceRows(rowHost(&codes, &inv, 2, row_group / 2), out[0 .. row_group]));
}

test "a block-FP8 stack against hand-decoded values" {
    // two experts of [2, 130]: the last two columns fall in each row's second scale block
    var codes: [2 * 2 * 130]u8 = @splat(0x38);
    codes[129] = 0xc0;
    codes[260 + 130 + 128] = 0x40;
    const inv = [_]f32{ 0.5, 4, 1, 8 };
    const h: Host = .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 2, 130, 1, 1 }, .bytes = &codes },
        .scale_inv = .{ .dtype = .f32, .rank = 3, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = std.mem.sliceAsBytes(&inv) },
    };
    var out: [codes.len]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.5), out[0]);
    try std.testing.expectEqual(@as(f64, -8), out[129]);
    try std.testing.expectEqual(@as(f64, 1), out[260]);
    try std.testing.expectEqual(@as(f64, 16), out[260 + 130 + 128]);
}

test "a reference refuses an unsupported f64 scale before loading it" {
    const codes = [_]u8{0x38};
    const inv = [_]f64{1};
    const h: Host = .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &codes },
        .scale_inv = .{ .dtype = .f64, .rank = 2, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&inv) },
    };
    var out: [1]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}

test "a reference refuses an empty projection" {
    const h: Host = .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 0, 1, 1, 1, 1 }, .bytes = &.{} },
        .scale_inv = .{ .dtype = .f32, .rank = 2, .shape = .{ 0, 1, 1, 1, 1 }, .bytes = &.{} },
    };
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &.{}));
}
