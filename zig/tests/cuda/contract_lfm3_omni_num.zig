//! LFM3-Omni's CUDA contract arithmetic: float64 at the reference's bf16 rounding points, as a value or an enclosure.
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;
const rows = core.contract.rows;
const bfr = core.bf16_round;

/// An enclosure of what an fp32 kernel may give at the reference's rounding points.
pub const I = struct { lo: f64, hi: f64 };

/// A replay's sums: exact float64 (the reference), each add rounded to `acc` in `order` (gate 4), or known exact.
pub const Mode = struct { acc: c.Accum = .none, order: rows.Order = .sequential, exact_sums: bool = false };

/// One fp32 rounding's reach from a value: half an ulp, taken as a whole one, and the subnormal floor.
fn reach(v: f64, ulps: f64) f64 {
    return @abs(v) * ulps * 0x1p-23 + 0x1p-140;
}

/// Float64 point values (iv false) or enclosures (iv true) through the same replay code.
pub fn Num(comptime iv: bool) type {
    return struct {
        pub const T = if (iv) I else f64;
        mode: Mode = .{},
        /// Extra fp32 roundings a kernel's own steps add before an output's last rounding (online softmax rescales).
        slack: f64 = 0,

        const Self = @This();

        pub fn k(_: Self, x: f64) T {
            return if (iv) .{ .lo = x, .hi = x } else x;
        }

        pub fn mid(_: Self, x: T) f64 {
            return if (iv) (x.lo + x.hi) / 2 else x;
        }

        /// One fp32 operation's rounding of a float64 result (`ulps` of its own error).
        pub fn op(_: Self, x: T, ulps: f64) T {
            if (!iv) return x;
            const lo = if (std.math.isFinite(x.lo)) x.lo - reach(x.lo, ulps) else x.lo;
            const hi = if (std.math.isFinite(x.hi)) x.hi + reach(x.hi, ulps) else x.hi;
            return .{ .lo = lo, .hi = hi };
        }

        pub fn add(s: Self, a: T, b: T) T {
            return if (iv) s.op(.{ .lo = a.lo + b.lo, .hi = a.hi + b.hi }, 1) else a + b;
        }

        pub fn sub(s: Self, a: T, b: T) T {
            return if (iv) s.op(.{ .lo = a.lo - b.hi, .hi = a.hi - b.lo }, 1) else a - b;
        }

        pub fn neg(_: Self, a: T) T {
            return if (iv) .{ .lo = -a.hi, .hi = -a.lo } else -a;
        }

        pub fn mul(s: Self, a: T, b: T) T {
            if (!iv) return a * b;
            const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
            return s.op(.{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) }, 1);
        }

        /// a / b for b of one sign.
        pub fn div(s: Self, a: T, b: T) T {
            if (!iv) return a / b;
            return s.mul(a, s.op(.{ .lo = 1 / b.hi, .hi = 1 / b.lo }, 0));
        }

        /// The reference's rounding point: the value rounded to bf16, ties to even.
        pub fn rb(_: Self, x: T) T {
            return if (iv) .{ .lo = bfr.bf16(x.lo), .hi = bfr.bf16(x.hi) } else bfr.bf16(x);
        }

        /// An output's last rounding to bf16, after the kernel's own slack.
        pub fn out(s: Self, x: T) T {
            return s.rb(s.op(x, s.slack));
        }

        /// A monotone function with `ulps` of fp32 error.
        pub fn mono(s: Self, x: T, f: *const fn (f64) f64, inc: bool, ulps: f64) T {
            if (!iv) return f(x);
            const a = f(x.lo);
            const b = f(x.hi);
            return s.op(if (inc) .{ .lo = a, .hi = b } else .{ .lo = b, .hi = a }, ulps);
        }

        /// A function of Lipschitz constant `lip`, with `ulps` of fp32 error.
        pub fn smooth(s: Self, x: T, f: *const fn (f64) f64, lip: f64, ulps: f64) T {
            if (!iv) return f(x);
            const m = (x.lo + x.hi) / 2;
            const r = (x.hi - x.lo) / 2 * lip;
            const v = f(m);
            return s.op(.{ .lo = v - r, .hi = v + r }, ulps);
        }

        pub fn max(_: Self, a: T, b: T) T {
            return if (iv) .{ .lo = @max(a.lo, b.lo), .hi = @max(a.hi, b.hi) } else @max(a, b);
        }

        /// A sum of terms as the replay's mode sums it; an enclosure widens by gamma(n) of the terms' magnitude.
        pub fn sum(s: Self, terms: []const T, buf: []f64) T {
            if (!iv) {
                if (s.mode.acc == .none) {
                    var t: f64 = 0;
                    for (terms) |x| t += x;
                    return t;
                }
                @memcpy(buf[0..terms.len], terms);
                return rows.sumIn(s.mode.acc, s.mode.order, buf[0..terms.len]);
            }
            var lo: f64 = 0;
            var hi: f64 = 0;
            var mag: f64 = 0;
            for (terms) |x| {
                lo += x.lo;
                hi += x.hi;
                mag += @max(@abs(x.lo), @abs(x.hi));
            }
            const g = if (s.mode.exact_sums) 0 else c.gamma(@floatFromInt(terms.len + 6), 0x1p-24) * mag;
            return s.op(.{ .lo = lo - g, .hi = hi + g }, 1);
        }
    };
}

pub fn exp(x: f64) f64 {
    return @exp(x);
}

/// erf in float64 (Abramowitz and Stegun 7.1.26 is too coarse): the series below 3, else 1 less the tail.
pub fn erf(x: f64) f64 {
    const a = @abs(x);
    var r: f64 = undefined;
    if (a < 3) {
        var term = a;
        var total = a;
        var n: f64 = 0;
        while (@abs(term) > 1e-18 * @abs(total)) {
            n += 1;
            term *= -a * a / n;
            total += term / (2 * n + 1);
        }
        r = total * 2 / @sqrt(std.math.pi);
    } else r = 1 - @exp(-a * a) / (a * @sqrt(std.math.pi)) * (1 - 1 / (2 * a * a) + 3 / (4 * a * a * a * a));
    return std.math.copysign(r, x);
}

pub fn cos(x: f64) f64 {
    return @cos(x);
}

pub fn sin(x: f64) f64 {
    return @sin(x);
}

pub fn rsqrt(x: f64) f64 {
    return 1 / @sqrt(x);
}

pub fn sqrt(x: f64) f64 {
    return @sqrt(x);
}

pub fn log10(x: f64) f64 {
    return @log10(x);
}

pub fn log(x: f64) f64 {
    return @log(x);
}

test "erf matches known values and the enclosure holds its point" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.8427007929497149), erf(1), 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, -0.9999779095030014), erf(-3), 1e-12);
    const P = Num(false){};
    const Q = Num(true){};
    const x = 1.37;
    const p = P.rb(P.mul(P.k(x), P.mono(P.k(x), exp, true, 2)));
    const q = Q.rb(Q.mul(Q.k(x), Q.mono(Q.k(x), exp, true, 2)));
    try std.testing.expect(q.lo <= p and p <= q.hi);
}
