//! drama-fp8-dense's scoring, std only so the host tests run it: the nearest bf16 and float64 error statistics.
const std = @import("std");

/// The bf16 nearest to `v`, ties to even; past the largest finite to infinity; zero keeps its sign.
pub fn bf16Nearest(v: f64) u16 {
    const sign: u16 = if (std.math.signbit(v)) 0x8000 else 0;
    if (std.math.isNan(v)) return 0x7FC0;
    const a = @abs(v);
    if (std.math.isInf(a)) return sign | 0x7F80;
    if (a == 0) return sign;
    // the grid step at a: 2^(e - 7) from 2^-126 up, the subnormals' 2^-133 below; a / step is exact
    const e = std.math.frexp(a).exponent - 1;
    const step = std.math.ldexp(@as(f64, 1), @max(e, -126) - 7);
    const q = a / step;
    var r = @floor(q);
    const frac = q - r;
    if (frac > 0.5 or (frac == 0.5 and @mod(r, 2) == 1)) r += 1;
    const x = r * step;
    if (x >= 0x1p128) return sign | 0x7F80;
    const bits: u32 = @bitCast(@as(f32, @floatCast(x))); // exact: x lies on the bf16 grid
    return sign | @as(u16, @intCast(bits >> 16));
}

pub fn bfOf(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

/// sqrt(err / ref): 0 only for 0 / 0; a nonzero error against a zero reference is infinite, so it fails every limit.
pub fn relRms(err: f64, ref: f64) f64 {
    if (ref == 0) return if (err == 0) 0 else std.math.inf(f64);
    return @sqrt(err / ref);
}

/// Float64 outputs and each one's tolerance, preregistered with the test before its first run.
pub const Ref = struct { want: []f64, tol: []f64 };

pub const Stats = struct {
    worst: f64 = 0, // the largest |got - want| / tol over the finite outputs
    at: usize = 0,
    nonfinite_got: usize = 0,
    nonfinite_want: usize = 0,
    exact: usize = 0,
    agg: f64 = 0, // relative RMS error over the gated columns
    col: f64 = 0, // the worst gated column's relative RMS error
    col_at: usize = 0,

    /// Both arms finite everywhere and every relative error defined: a usable measurement or baseline.
    pub fn valid(st: Stats) bool {
        return st.nonfinite_got == 0 and st.nonfinite_want == 0 and std.math.isFinite(st.agg) and std.math.isFinite(st.col);
    }

    /// (a)'s preregistered limits: within tolerance everywhere, relative RMS 2^-8 overall and 2^-7 in the worst column.
    pub fn within(st: Stats) bool {
        return st.valid() and st.worst <= 1 and st.agg <= 0x1p-8 and st.col <= 0x1p-7;
    }

    /// (b): the lane's error no worse than the BF16 path's, overall and in the worst column, both arms valid.
    pub fn noWorse(lane: Stats, bf16: Stats) bool {
        return lane.valid() and bf16.valid() and lane.agg <= bf16.agg and lane.col <= bf16.col;
    }
};

/// `got` (bf16 [m][n]) against `want`'s first m rows, with `tol` when given; RMS over the columns `gated` admits.
pub fn stats(a: std.mem.Allocator, got: []const u8, want: []const f64, tol: ?[]const f64, n: usize, gated: ?[]const bool) !Stats {
    var st: Stats = .{};
    const err = try a.alloc(f64, n);
    defer a.free(err);
    const sq = try a.alloc(f64, n);
    defer a.free(sq);
    @memset(err, 0);
    @memset(sq, 0);
    for (0..got.len / 2) |i| {
        const g = bfOf(std.mem.bytesToValue(u16, got[2 * i ..][0..2]));
        const w = want[i];
        st.nonfinite_got += @intFromBool(!std.math.isFinite(g));
        st.nonfinite_want += @intFromBool(!std.math.isFinite(w));
        if (!std.math.isFinite(g) or !std.math.isFinite(w)) continue;
        const d = g - w;
        st.exact += @intFromBool(d == 0);
        if (tol) |t| if (!(@abs(d) / t[i] <= st.worst)) {
            st.worst = @abs(d) / t[i];
            st.at = i;
        };
        err[i % n] += d * d;
        sq[i % n] += w * w;
    }
    var e: f64 = 0;
    var s: f64 = 0;
    for (err, sq, 0..) |ce, cs, c| {
        if (gated) |gt| if (!gt[c]) continue;
        e += ce;
        s += cs;
        const rel = relRms(ce, cs);
        if (!(rel <= st.col)) {
            st.col = rel;
            st.col_at = c;
        }
    }
    st.agg = relRms(e, s);
    return st;
}

/// The value of positive code b, infinity standing as 2^128: the codes 0..0x7F80 increase with their values.
fn codeValue(b: u16) f64 {
    return if (b == 0x7F80) 0x1p128 else bfOf(b);
}

/// The bf16 nearest to `v` by exhaustive search over every positive code, ties to the even code: the oracle.
fn bruteNearest(v: f64) u16 {
    const sign: u16 = if (std.math.signbit(v)) 0x8000 else 0;
    const a = @abs(v);
    var best: u16 = 0;
    var dist = std.math.inf(f64);
    var b: u16 = 0;
    while (b <= 0x7F80) : (b += 1) {
        const x = codeValue(b);
        const d = @abs(a - x);
        if (d < dist or (d == dist and b & 1 == 0)) {
            best = b;
            dist = d;
        }
    }
    return sign | best;
}

/// The oracle by bisection over the ordered codes, for the many random values: the nearer neighbour, ties to even.
fn searchNearest(v: f64) u16 {
    const sign: u16 = if (std.math.signbit(v)) 0x8000 else 0;
    const a = @abs(v);
    if (a >= codeValue(0x7F80)) return sign | 0x7F80;
    var lo: u16 = 0; // codeValue(lo) <= a < codeValue(hi)
    var hi: u16 = 0x7F80;
    while (hi - lo > 1) {
        const mid = lo + (hi - lo) / 2;
        if (codeValue(mid) <= a) lo = mid else hi = mid;
    }
    const below = a - codeValue(lo);
    const above = codeValue(hi) - a;
    return sign | if (below < above or (below == above and lo & 1 == 0)) lo else hi;
}

test "bf16Nearest against an exhaustive oracle: ties, subnormals, overflow, signed zero, a reviewer's counterexample" {
    const scale: f32 = @bitCast(@as(u32, 0x3f804ec5));
    const six = 1.625 * @as(f64, scale);
    try std.testing.expectEqual(@as(u16, 0x3fd1), bf16Nearest(six)); // through f32 first it was 0x3fd0
    try std.testing.expectEqual(@as(u16, 0x3fd1), bruteNearest(six));
    const max: f64 = bfOf(0x7F7F);
    const half = 0x1p119; // half the top binade's step 2^120: max + half ties to the even neighbour, infinity
    const cases = [_]struct { f64, u16 }{
        .{ 0.0, 0x0000 },                                 .{ -0.0, 0x8000 },
        .{ -0x1p-140, 0x8000 },                           .{ 0x1p-134, 0x0000 }, // half the least step: to even zero
        .{ 0x1p-133, 0x0001 },                            .{ 0x1.8p-133, 0x0002 }, // 1.5 steps: to even 2
        .{ 0x1.0000000000001p-134, 0x0001 },              .{ 2.05e-43, 0x0000 },
        .{ 1 + 0x1p-8, 0x3F80 },                          .{ 1 + 3 * 0x1p-8, 0x3F82 },
        .{ std.math.nextAfter(f64, 1 + 0x1p-8, 2), 0x3F81 }, .{ std.math.nextAfter(f64, 1 + 3 * 0x1p-8, 0), 0x3F81 },
        .{ -(1 + 0x1p-8), 0xBF80 },                       .{ max, 0x7F7F },
        .{ max + half, 0x7F80 },                          .{ std.math.nextAfter(f64, max + half, 0), 0x7F7F },
        .{ -(max + half), 0xFF80 },                       .{ 1e300, 0x7F80 },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c[1], bf16Nearest(c[0]));
        try std.testing.expectEqual(c[1], bruteNearest(c[0]));
    }
    var prng = std.Random.DefaultPrng.init(0xBF16);
    const rng = prng.random();
    for (0..4000) |i| {
        // random doubles over bf16's range, then e4m3 values times random positive f32 scales (the dequantized weights)
        const v: f64 = if (i % 2 == 0)
            std.math.ldexp(1 + rng.float(f64), rng.intRangeAtMost(i32, -140, 128)) * @as(f64, if (rng.boolean()) -1 else 1)
        else
            @as(f64, @floatFromInt(8 + rng.uintLessThan(u8, 8))) / 8 * std.math.ldexp(@as(f64, 1), rng.intRangeAtMost(i32, -9, 8)) *
                @as(f64, @as(f32, @bitCast(1 + rng.uintLessThan(u32, 0x7F7F_FFFF))));
        try std.testing.expectEqual(searchNearest(v), bf16Nearest(v));
        if (i < 64) try std.testing.expectEqual(bruteNearest(v), bf16Nearest(v));
    }
    for (cases) |c| try std.testing.expectEqual(c[1], searchNearest(c[0])); // the two oracles agree
}

test "relRms is 0 only for 0 / 0" {
    try std.testing.expectEqual(@as(f64, 0), relRms(0, 0));
    try std.testing.expect(std.math.isInf(relRms(1e-30, 0)));
    try std.testing.expectEqual(@as(f64, 2), relRms(4, 1));
}

test "stats: a non-finite arm or a nonzero error against a zero reference never passes" {
    const gpa = std.testing.allocator;
    const one = bf16Nearest(1);
    const two = bf16Nearest(2);
    var got = [_]u16{ one, two, one, two };
    var want = [_]f64{ 1, 2, 1, 2 };
    var tol = [_]f64{ 0.1, 0.1, 0.1, 0.1 };
    const good = try stats(gpa, std.mem.sliceAsBytes(&got), &want, &tol, 2, null);
    try std.testing.expect(good.within() and good.exact == 4);
    got[1] = 0x7FC0; // a NaN output
    const nan_got = try stats(gpa, std.mem.sliceAsBytes(&got), &want, &tol, 2, null);
    try std.testing.expect(nan_got.nonfinite_got == 1 and !nan_got.valid() and !nan_got.within());
    got[1] = two;
    want[2] = std.math.inf(f64); // a non-finite reference or baseline
    const inf_want = try stats(gpa, std.mem.sliceAsBytes(&got), &want, &tol, 2, null);
    try std.testing.expect(inf_want.nonfinite_want == 1 and !inf_want.valid());
    try std.testing.expect(!Stats.noWorse(good, inf_want) and !Stats.noWorse(inf_want, good));
    want = .{ 0, 2, 0, 2 }; // column 0's reference is zero, its outputs are not
    const zero_ref = try stats(gpa, std.mem.sliceAsBytes(&got), &want, null, 2, null);
    try std.testing.expect(std.math.isInf(zero_ref.col) and zero_ref.col_at == 0 and !zero_ref.valid() and !zero_ref.within());
    try std.testing.expect(!Stats.noWorse(good, zero_ref));
    got = .{ 0, two, 0, two }; // 0 / 0 in column 0 is no error
    const zeros = try stats(gpa, std.mem.sliceAsBytes(&got), &want, null, 2, null);
    try std.testing.expect(zeros.valid() and zeros.col == 0 and zeros.agg == 0);
    const gated = [_]bool{ false, true };
    got = .{ one, two, one, two };
    const masked = try stats(gpa, std.mem.sliceAsBytes(&got), &want, null, 2, &gated); // the gate leaves column 0 out
    try std.testing.expect(masked.valid() and masked.col == 0);
    var worse = good;
    worse.agg = 1e-3;
    try std.testing.expect(Stats.noWorse(good, worse) and !Stats.noWorse(worse, good));
}
