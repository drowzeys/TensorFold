//! ModelOpt NVFP4 weights (e2m1 codes, an e4m3 scale every 16 inputs, a global scale) on the host: decode and encode.
const std = @import("std");
const types = @import("types.zig");
const fp8 = @import("fp8.zig");

const Tensor = types.Tensor;

/// Inputs one e4m3 scale covers.
pub const group = 16;

/// An e2m1 nibble's value.
pub fn e2m1(n: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * mag[n & 7];
}

/// As stored: `weight` u8 [..., N, K / 2] (even columns low nibbles), `scale` e4m3 [..., N, K / 16], `global` a matrix.
pub const Host = struct { weight: Tensor, scale: Tensor, global: []const f32 };

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel() * 2;
}

/// fp64 dequantization of every row of `h` (a stack's leading dimensions folded), code * scale * global, into `out`.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale.rank != rank or h.weight.dtype != .u8 or h.scale.dtype != .f8_e4m3) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1] * 2;
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale.shape[d]) return error.UnexpectedTensor;
    if (h.scale.shape[rank - 2] != n or h.scale.shape[rank - 1] * group != k or out.len != elements(h)) return error.UnexpectedTensor;
    const stack = elements(h) / (n * k);
    if (h.global.len != stack) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const row = e * n + r;
        const byte = h.weight.bytes[row * (k / 2) + c / 2];
        const nib: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
        out[row * k + c] = e2m1(nib) * fp8.e4m3(h.scale.bytes[row * (k / group) + c / group]) * h.global[e];
    };
}

/// The 126 positive finite e4m3 values, ascending with their codes 0x01..0x7E.
const scale_values = blk: {
    @setEvalBranchQuota(100_000);
    var v: [126]f64 = undefined;
    for (&v, 1..) |*x, c| x.* = fp8.e4m3(c);
    break :blk v;
};

const Lanes = @Vector(group, f64);
const Index = @Vector(group, u8);

/// C2's global of a matrix whose largest magnitude is `amax`: its largest block scale lands on e4m3's 448.
pub fn globalScale(amax: f64) f32 {
    return if (amax == 0) 1 else @floatCast(amax / (6.0 * 448.0));
}

/// The e2m1 index whose (q s) g is nearest each |W|, via exact midpoints (m s) g: ties to even, saturating at 6.
fn index(aw: Lanes, s: f64, g: f64) Index {
    const mid = [7]f64{ 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5 };
    var idx: Index = @splat(0);
    inline for (mid, 0..) |m, i| {
        const t: Lanes = @splat(m * s * g);
        const up = if (i % 2 == 1) aw >= t else aw > t; // the upper neighbour 2, 4 or 6 is even: a tie goes up
        idx += @select(u8, up, @as(Index, @splat(1)), @as(Index, @splat(0)));
    }
    return idx;
}

fn magnitude(idx: Index) Lanes {
    const f: Lanes = @floatFromInt(idx);
    const high = @select(f64, idx == @as(Index, @splat(7)), @as(Lanes, @splat(6)), f - @as(Lanes, @splat(2)));
    return @select(f64, idx <= @as(Index, @splat(4)), f * @as(Lanes, @splat(0.5)), high);
}

/// Σ h (|W| - (q s) g)² of one block, in column order (a strict-mode @reduce keeps the order and does not contract).
fn loss(aw: Lanes, h: Lanes, s: f64, g: f64) f64 {
    const d = aw - magnitude(index(aw, s, g)) * @as(Lanes, @splat(s)) * @as(Lanes, @splat(g));
    return @reduce(.Add, h * d * d);
}

/// One block's e4m3 scale code, its 16 nibbles (sign bit 3) and the importance-weighted loss of that choice.
pub const Block = struct { scale: u8, nibbles: [group]u8, loss: f64 };

/// C2 step 4: of all 126 e4m3 scale codes the least loss (ties: smaller code); an all-zero block takes scale 0.
pub fn encodeBlock(w: *const [group]f64, h: *const [group]f64, g: f32) Block {
    var aw: Lanes = undefined;
    inline for (0..group) |i| aw[i] = @abs(w[i]);
    if (@reduce(.Max, aw) == 0) return .{ .scale = 0, .nibbles = @splat(0), .loss = 0 };
    var best: Block = .{ .scale = 1, .nibbles = undefined, .loss = std.math.inf(f64) };
    for (scale_values, 1..) |s, code| {
        const e = loss(aw, h.*, s, g);
        if (e < best.loss) best = .{ .scale = @intCast(code), .nibbles = undefined, .loss = e };
    }
    best.nibbles = index(aw, scale_values[best.scale - 1], g);
    for (&best.nibbles, w) |*n, x| if (x < 0 and n.* != 0) {
        n.* |= 8;
    };
    return best;
}

/// A matrix's amax and global, and the sums behind C2 step 5's err_h and rel.
pub const Summary = struct {
    amax: f64 = 0,
    global: f32 = 1,
    err_h: f64 = 0,
    ref_h: f64 = 0,
    err: f64 = 0,
    ref: f64 = 0,

    /// sqrt(Σ h (W - Q)² / Σ h W²), a diagonal expert-input-error proxy; null when only the denominator is zero.
    pub fn errH(s: Summary) ?f64 {
        return ratio(s.err_h, s.ref_h);
    }

    /// ‖W - Q‖ / ‖W‖; null when only the denominator is zero.
    pub fn rel(s: Summary) ?f64 {
        return ratio(s.err, s.ref);
    }

    fn ratio(num: f64, den: f64) ?f64 {
        if (den == 0) return if (num == 0) 0 else null;
        return @sqrt(num / den);
    }
};

/// C2 over W [n][k] weighed by `h` [k] into codes [n][k / 2], scales [n][k / 16]; errors of what `reference` reads.
pub fn encode(w: []const f64, h: []const f64, n: usize, k: usize, codes: []u8, scales: []u8) Summary {
    std.debug.assert(k % group == 0 and w.len == n * k and h.len == k and codes.len == n * k / 2 and scales.len == n * k / group);
    var amax: f64 = 0;
    for (w) |x| amax = @max(amax, @abs(x));
    const g = globalScale(amax);
    for (0..n) |r| for (0..k / group) |b| {
        const at = r * k + b * group;
        const q = encodeBlock(w[at..][0..group], h[b * group ..][0..group], g);
        scales[at / group] = q.scale;
        for (0..group / 2) |j| codes[at / 2 + j] = q.nibbles[2 * j] | q.nibbles[2 * j + 1] << 4;
    };
    return measure(w, h, n, k, codes, scales, g);
}

/// C2 step 5 for any NVFP4 matrix: W [n][k] against what `reference` decodes, row-major, increasing columns.
pub fn measure(w: []const f64, h: []const f64, n: usize, k: usize, codes: []const u8, scales: []const u8, g: f32) Summary {
    std.debug.assert(k % group == 0 and w.len == n * k and h.len == k and codes.len == n * k / 2 and scales.len == n * k / group);
    var out: Summary = .{ .global = g };
    for (w) |x| out.amax = @max(out.amax, @abs(x));
    for (0..n) |r| for (0..k / group) |b| {
        const at = r * k + b * group;
        const s = fp8.e4m3(scales[at / group]);
        for (0..group) |i| {
            const x = w[at + i];
            const byte = codes[(at + i) / 2];
            const d = x - e2m1(if (i & 1 == 0) byte & 0xF else byte >> 4) * s * g;
            const hi = h[b * group + i];
            out.err_h += hi * d * d;
            out.ref_h += hi * x * x;
            out.err += d * d;
            out.ref += x * x;
        }
    };
    return out;
}

test "an NVFP4 stack against hand-decoded values" {
    // two experts of [1, 32]: two scale groups a row; byte 0 holds columns 0 (low) and 1 (high)
    var codes: [2 * 16]u8 = @splat(0);
    codes[0] = 0xF2; // column 0: e2m1 2 -> 1.0; column 1: e2m1 0xF -> -6
    codes[8] = 0x07; // column 16: 6
    codes[16 + 15] = 0x30; // expert 1, column 31: e2m1 3 -> 1.5
    const scales = [_]u8{ 0x38, 0x40, 0x38, 0x30 }; // 1, 2 | 1, 0.5
    const global = [_]f32{ 0.25, 2 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 16, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [64]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.25), out[0]);
    try std.testing.expectEqual(@as(f64, -1.5), out[1]);
    try std.testing.expectEqual(@as(f64, 3), out[16]);
    try std.testing.expectEqual(@as(f64, 1.5), out[32 + 31]);
    try std.testing.expectEqual(@as(f64, 0), out[2]);
}

test "a reference refuses mismatched expert stacks" {
    const codes: [16]u8 = @splat(0x22);
    const scales: [2]u8 = @splat(0x38);
    const global = [_]f32{ 1, 1 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 8, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [32]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}

fn decoded(out: []f64, codes: []const u8, scales: []const u8, g: f32, n: usize, k: usize) !void {
    try reference(.{
        .weight = .{ .dtype = .u8, .rank = 2, .shape = .{ n, k / 2, 1, 1, 1 }, .bytes = codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ n, k / group, 1, 1, 1 }, .bytes = scales },
        .global = &.{g},
    }, out);
}

test "representable weights encode with zero error and decode to themselves" {
    const n = 4;
    const k = 64;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const g: f32 = 1.0 / 1024.0;
    var w: [n * k]f64 = undefined;
    for (0..n * k / group) |b| {
        const s = fp8.e4m3(if (b == 0) 0x7E else rnd.intRangeAtMost(u8, 1, 0x7E));
        // block 0 holds the matrix amax, 6 at scale 448, so the encoder finds the global g itself
        for (w[b * group ..][0..group], 0..) |*x, i| x.* = e2m1(if (i == b % group) 7 else rnd.int(u4)) * s * g;
    }
    const h: [k]f64 = @splat(1);
    var codes: [n * k / 2]u8 = undefined;
    var scales: [n * k / group]u8 = undefined;
    const sum = encode(&w, &h, n, k, &codes, &scales);
    try std.testing.expectEqual(g, sum.global);
    try std.testing.expectEqual(@as(f64, 0), sum.err);
    try std.testing.expectEqual(@as(f64, 0), sum.err_h);
    var out: [n * k]f64 = undefined;
    try decoded(&out, &codes, &scales, g, n, k);
    for (w, out) |x, y| try std.testing.expectEqual(x, y);
}

/// An independent block encoder: every code's value from fp8.e4m3, then the nearest reconstruction by scalar distance.
fn brute(w: *const [group]f64, h: *const [group]f64, g: f32) Block {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    var best: Block = .{ .scale = 0, .nibbles = @splat(0), .loss = 0 };
    for (w) |x| {
        if (x != 0) break;
    } else return best;
    best.loss = std.math.inf(f64);
    for (1..127) |code| {
        const s = fp8.e4m3(@intCast(code));
        var q: Block = .{ .scale = @intCast(code), .nibbles = undefined, .loss = 0 };
        for (w, h, &q.nibbles) |x, hi, *nib| {
            const a = @abs(x);
            var i: u8 = 0;
            for (1..8) |j| {
                const dj = @abs(a - mag[j] * s * g);
                const di = @abs(a - mag[i] * s * g);
                if (dj < di or (dj == di and j % 2 == 0)) i = @intCast(j);
            }
            nib.* = if (x < 0 and i != 0) i | 8 else i;
            const d = a - mag[i] * s * g;
            q.loss += hi * d * d;
        }
        if (q.loss < best.loss) best = q;
    }
    return best;
}

test "the block search picks what an independent scalar search of all 126 scales picks on random blocks" {
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for (0..20000) |trial| {
        var w: [group]f64 = undefined;
        const base = std.math.exp2(rnd.float(f64) * 16 - 12);
        // f32 values, as an e4m3 code times an f32 scale_inv has at most 28 significant bits
        for (&w) |*x| x.* = if (rnd.uintLessThan(u8, 8) == 0) 0 else @as(f32, @floatCast(rnd.floatNorm(f64) * base));
        var h: [group]f64 = @splat(1);
        if (trial % 2 == 1) for (&h) |*x| {
            x.* = 0.25 + rnd.float(f64) * 3.75;
        };
        // one outlier the importance nearly ignores: the best scale may saturate it to resolve the rest
        if (trial % 5 == 2) {
            w[trial % group] *= 20;
            h[trial % group] = 1e-4;
        }
        var amax: f64 = 0;
        for (w) |x| amax = @max(amax, @abs(x));
        // matrix amax: the block's own (scale at most 448), up to 1000x above, or 1e5x above (subnormal scales win)
        const ratio: f64 = if (trial % 5 == 1) 1e5 + rnd.float(f64) * 1e5 else if (trial % 3 == 0) 1 else 1 + rnd.float(f64) * 999;
        const g = globalScale(amax * ratio);
        const got = encodeBlock(&w, &h, g);
        const want = brute(&w, &h, g);
        try std.testing.expectEqual(want.scale, got.scale);
        try std.testing.expectEqualSlices(u8, &want.nibbles, &got.nibbles);
        try std.testing.expectEqual(@as(u64, @bitCast(want.loss)), @as(u64, @bitCast(got.loss))); // the same sum order
    }
}

test "e2m1 ties go to the even index and saturate at six; equal block errors take the smaller scale" {
    const x = [group]f64{ 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7, 0.2500001, 0.7499999, 1.2500001, 1.7499999, 2.5000001, 3.4999999, 5.0000001, 0 };
    const want = [group]u8{ 0, 2, 2, 4, 4, 6, 6, 7, 1, 1, 3, 3, 5, 5, 7, 0 };
    try std.testing.expectEqual(want, @as([group]u8, index(x, 1, 1)));
    var w: [group]f64 = @splat(1);
    w[3] = -1e-6; // rounds to zero: a positive zero code, not -0
    const g = globalScale(1);
    const ones: [group]f64 = @splat(1);
    const q = encodeBlock(&w, &ones, g);
    try std.testing.expectEqual(@as(u8, 7), q.nibbles[0]);
    try std.testing.expectEqual(@as(u8, 0), q.nibbles[3]);
    const zero: [group]f64 = @splat(0);
    try std.testing.expectEqual(@as(u8, 1), encodeBlock(&w, &zero, g).scale); // every scale costs 0: the smallest code
    const none = encodeBlock(&@as([group]f64, @splat(-0.0)), &ones, g);
    try std.testing.expectEqual(@as(u8, 0), none.scale);
    try std.testing.expectEqualSlices(u8, &@as([group]u8, @splat(0)), &none.nibbles);
}
