//! LFM3-Omni's CUDA kernels replayed in float64 at the reference's bf16 rounding points: values and enclosures.
const std = @import("std");
const num = @import("contract_lfm3_omni_num.zig");
const data = @import("contract_lfm3_omni_data.zig");

const Data = data.Data;

/// Every output of `d`'s kernel in buffer order, as points (iv false) or enclosures; `terms`/`buf` are scratch.
pub fn replay(comptime iv: bool, n: num.Num(iv), d: *const Data, out: []num.Num(iv).T, terms: []num.Num(iv).T, buf: []f64) void {
    const R = Replay(iv){ .n = n, .d = d, .out = out, .terms = terms, .buf = buf };
    R.run();
}

fn Replay(comptime iv: bool) type {
    return struct {
        n: num.Num(iv),
        d: *const Data,
        out: []T,
        terms: []T,
        buf: []f64,

        const T = num.Num(iv).T;
        const Self = @This();

        fn x(r: Self, b: usize, i: usize) T {
            return r.n.k(r.d.v(b, i));
        }

        fn sum(r: Self, len: usize) T {
            return r.n.sum(r.terms[0..len], r.buf);
        }

        /// max that keeps the other operand over a NaN, as fmaxf does.
        fn fmax(r: Self, a: T, b: T) T {
            if (!iv) {
                if (std.math.isNan(a)) return b;
                if (std.math.isNan(b)) return a;
            }
            return r.n.max(a, b);
        }

        /// 1 / (1 + expf(-v)), step by step as the kernel takes it (an expf of a large -v overflows to Inf).
        fn sigmoid(r: Self, v: T) T {
            const n = r.n;
            return n.div(n.k(1), n.add(n.k(1), r.exp32(n.neg(v))));
        }

        fn silu(r: Self, v: T) T {
            const n = r.n;
            return n.div(v, n.add(n.k(1), r.exp32(n.neg(v))));
        }

        /// 0.5 v (1 + erff(v c)): erff's error is absolute next to 1, so it is taken before the sum.
        fn gelu(r: Self, v: T) T {
            const n = r.n;
            const c: f32 = 0.70710678118654752;
            return n.mul(n.mul(n.k(0.5), v), n.add(n.k(1), n.smooth(n.mul(v, n.k(c)), num.erf, 1.13, 2)));
        }

        /// expf: two ulps, Inf past fp32's largest value.
        fn exp32(r: Self, v: T) T {
            const e = r.n.mono(v, num.exp, true, 2);
            const top: f64 = std.math.floatMax(f32);
            if (iv) return .{ .lo = if (e.lo > top) std.math.inf(f64) else e.lo, .hi = if (e.hi > top) std.math.inf(f64) else e.hi };
            return if (e > top) std.math.inf(f64) else e;
        }

        fn sq(r: Self, a: T) T {
            if (iv and a.lo < 0 and a.hi > 0) return r.n.op(.{ .lo = 0, .hi = @max(a.lo * a.lo, a.hi * a.hi) }, 1);
            return r.n.mul(a, a);
        }

        fn run(r: Self) void {
            const d = r.d;
            const n = r.n;
            const sh = d.shape;
            const T_ = sh.rows;
            switch (d.which) {
                .rmsnorm_res, .rmsnorm_rows => {
                    const D = sh.k;
                    const res = d.which == .rmsnorm_res;
                    const y = if (res) r.out[T_ * D ..] else r.out;
                    for (0..T_) |t| {
                        for (0..D) |i| {
                            const v = if (res) n.rb(n.add(r.x(0, t * D + i), r.x(1, t * D + i))) else r.x(0, t * D + i);
                            if (res) r.out[t * D + i] = v;
                            r.terms[i] = r.sq(v);
                        }
                        const inv = n.mono(n.add(n.div(r.sum(D), n.k(@floatFromInt(D))), n.k(data.eps)), num.rsqrt, false, 2);
                        for (0..D) |i| {
                            const v = if (res) r.out[t * D + i] else r.x(0, t * D + i);
                            const w = r.x(2, if (res) i else t * D + i);
                            y[t * D + i] = if (d.cfg.mode == 0) n.rb(n.mul(w, n.rb(n.mul(v, inv)))) else n.rb(n.mul(n.mul(v, inv), w));
                        }
                    }
                },
                .head_norm => {
                    const hd = sh.k;
                    for (0..T_ * sh.n) |row| {
                        for (0..hd) |i| r.terms[i] = r.sq(r.x(0, row * hd + i));
                        const inv = n.mono(n.add(n.div(r.sum(hd), n.k(@floatFromInt(hd))), n.k(data.eps)), num.rsqrt, false, 2);
                        for (0..hd) |i| r.out[row * hd + i] = n.rb(n.mul(r.x(1, i), n.rb(n.mul(r.x(0, row * hd + i), inv))));
                    }
                },
                .rope => {
                    const hd = sh.k;
                    const half = hd / 2;
                    const mode = d.cfg.mode;
                    for (0..T_) |t| for (0..sh.n) |h| for (0..half) |p| {
                        const at = (t * sh.n + h) * hd;
                        const fr: f32 = @as(f32, @floatFromInt(d.cfg.pos0 + t)) * d.fl(1)[p];
                        var co = n.smooth(n.k(fr), num.cos, 1, 2);
                        var si = n.smooth(n.k(fr), num.sin, 1, 2);
                        if (mode != 0) {
                            co = n.rb(co);
                            si = n.rb(si);
                        }
                        const a = r.x(0, at + p);
                        const z = r.x(0, at + p + half);
                        if (mode == 2) {
                            r.out[at + p] = n.rb(n.add(n.rb(n.mul(a, co)), n.rb(n.mul(n.neg(z), si))));
                            r.out[at + p + half] = n.rb(n.add(n.rb(n.mul(z, co)), n.rb(n.mul(a, si))));
                        } else {
                            r.out[at + p] = n.rb(n.sub(n.mul(a, co), n.mul(z, si)));
                            r.out[at + p + half] = n.rb(n.add(n.mul(z, co), n.mul(a, si)));
                        }
                    };
                },
                .copy_rows, .bf16_f32, .repeat => {
                    const rep = if (d.which == .repeat) sh.k else 1;
                    for (0..T_ * rep) |row| for (0..sh.n) |i| {
                        r.out[row * sh.n + i] = r.x(0, (row / rep) * sh.n + i);
                    };
                },
                .short_conv, .conv_rows, .conv_state => r.conv(),
                .attention, .attention_tiles => r.attention(),
                .gate => {
                    const hd = sh.k;
                    for (0..T_) |t| for (0..sh.n) |h| {
                        const gv = n.rb(r.sigmoid(r.x(1, t * sh.n + h)));
                        for (0..hd) |i| {
                            const at = (t * sh.n + h) * hd + i;
                            r.out[at] = n.rb(n.mul(r.x(0, at), gv));
                        }
                    };
                },
                .swiglu => for (0..T_) |t| for (0..sh.n) |i| {
                    const a = r.x(0, t * 2 * sh.n + i);
                    r.out[t * sh.n + i] = n.rb(n.mul(n.rb(r.silu(a)), r.x(0, t * 2 * sh.n + sh.n + i)));
                },
                .add => for (0..T_ * sh.n) |i| {
                    r.out[i] = n.rb(n.add(r.x(1, i), r.x(2, i)));
                },
                .bias_act => for (0..T_) |row| for (0..sh.n) |col| {
                    var v = n.rb(n.add(r.x(0, row * sh.n + col), r.x(1, if (d.cfg.per_row) row else col)));
                    if (d.cfg.act == 1) v = r.silu(v) else if (d.cfg.act == 2) v = r.gelu(v);
                    r.out[row * sh.n + col] = n.rb(v);
                },
                .layernorm => {
                    const D = sh.k;
                    const dk = n.k(@floatFromInt(D));
                    for (0..T_) |t| {
                        for (0..D) |i| r.terms[i] = r.x(0, t * D + i);
                        const mean = n.div(r.sum(D), dk);
                        for (0..D) |i| r.terms[i] = r.sq(n.sub(r.x(0, t * D + i), mean));
                        const rstd = n.mono(n.add(n.div(r.sum(D), dk), n.k(data.eps)), num.rsqrt, false, 2);
                        for (0..D) |i| r.out[t * D + i] = n.rb(n.add(n.mul(n.mul(n.sub(r.x(0, t * D + i), mean), rstd), r.x(1, i)), r.x(2, i)));
                    }
                },
                .embed => for (0..T_) |t| {
                    const id = d.ints(1)[t];
                    for (0..sh.n) |i| r.out[t * sh.n + i] = if (id < 0) r.x(2, t * sh.n + i) else r.x(0, @as(usize, @intCast(id)) * sh.n + i);
                },
                .embed_codes => for (0..T_) |t| for (0..sh.n) |i| {
                    for (0..data.nq) |q| r.terms[q] = r.x(0, (@as(usize, @intCast(d.ints(1)[t * data.nq + q])) + q * data.size) * sh.n + i);
                    r.out[t * sh.n + i] = n.rb(r.sum(data.nq));
                },
                .depth_in => for (0..data.nq) |q| for (0..sh.n) |i| {
                    const e = if (d.ints(4)[q] != 0) r.x(1, i) else r.x(2, (q * data.size + @as(usize, @intCast(d.ints(3)[q]))) * sh.n + i);
                    r.out[q * sh.n + i] = n.rb(n.add(r.x(0, q * sh.n + i), e));
                },
                .conv2d => {
                    const C = sh.rows;
                    const W2 = sh.n;
                    const H = d.cfg.height;
                    const W = 2 * W2;
                    for (0..C) |ch| for (0..H / 2) |h| for (0..W2) |w| {
                        const plane = (if (d.cfg.cin == 1) 0 else ch) * H * W;
                        var m: usize = 0;
                        for (0..3) |kh| for (0..3) |kw| {
                            const ih = @as(isize, @intCast(2 * h + kh)) - 1;
                            const iw = @as(isize, @intCast(2 * w + kw)) - 1;
                            if (ih < 0 or ih >= H or iw < 0 or iw >= W) continue;
                            r.terms[m] = n.mul(r.x(1, ch * 9 + kh * 3 + kw), r.x(0, plane + @as(usize, @intCast(ih)) * W + @as(usize, @intCast(iw))));
                            m += 1;
                        };
                        var v = n.rb(n.add(r.sum(m), r.x(2, ch)));
                        if (d.cfg.act != 0) v = n.rb(r.silu(v));
                        r.out[(ch * (H / 2) + h) * W2 + w] = v;
                    };
                },
                .f32_bf16 => for (0..T_ * sh.n) |i| {
                    r.out[i] = n.rb(r.x(0, i));
                },
                .fsq => {
                    const levels = [4]i32{ 8, 7, 6, 6 };
                    const basis = [4]i32{ 1, 8, 56, 336 };
                    for (0..T_ * sh.n) |i| for (0..4) |j| {
                        const half = @divTrunc(levels[j], 2);
                        const digit = @mod(@divTrunc(d.ints(0)[i], basis[j]), levels[j]) - half;
                        r.out[i * 4 + j] = n.rb(n.div(n.k(@floatFromInt(digit)), n.k(@floatFromInt(half))));
                    };
                },
                .polar => for (0..T_) |t| for (0..sh.n) |k| {
                    const m = n.mono(r.x(0, t * 2 * sh.n + k), num.exp, true, 2);
                    const a = r.x(0, t * 2 * sh.n + sh.n + k);
                    r.out[t * 2 * sh.n + k] = n.mul(m, n.smooth(a, num.cos, 1, 2));
                    r.out[t * 2 * sh.n + sh.n + k] = n.mul(m, n.smooth(a, num.sin, 1, 2));
                },
                .gemv => {
                    const K = sh.k;
                    for (0..T_) |t| for (0..sh.n) |col| {
                        for (0..K) |k| r.terms[k] = n.mul(r.x(0, col * K + k), r.x(1, t * K + k));
                        const s = r.sum(K);
                        r.out[t * sh.n + col] = if (d.cfg.f32_out) s else n.rb(s);
                    };
                },
                .mel_frames => {
                    const len = d.bufs[0].host.len / 4;
                    for (0..T_) |t| for (0..sh.n) |i| {
                        const s = t * sh.k + i;
                        r.out[t * sh.n + i] = if (s < len) n.mul(r.x(0, s), r.x(1, i)) else n.k(0);
                    };
                },
                .power => for (0..T_) |t| for (0..sh.n) |k| {
                    const re = r.x(0, t * 2 * sh.n + k);
                    const im = r.x(0, t * 2 * sh.n + sh.n + k);
                    const a = n.mono(n.add(n.mul(re, re), n.mul(im, im)), num.sqrt, true, 1);
                    r.out[t * sh.n + k] = n.mul(a, a);
                },
                .logmel => for (0..T_) |t| for (0..sh.n) |m| {
                    const v = r.x(0, t * sh.n + m);
                    const tiny = if (iv) v.hi < 1e-13 else (v < 1e-13 and !std.math.isNan(v));
                    const lg = if (tiny) n.k(-12) else r.fmax(n.mono(v, num.log10, true, 2), n.k(-12));
                    r.out[m * T_ + t] = n.div(n.add(lg, n.k(4)), n.k(4));
                },
                .pos, .argmax, .depth_pick, .frame_end, .hist_push => unreachable,
            }
        }

        /// The gated short conv: y = C * conv(B * x), the K-1 rows before the first from the state.
        fn conv(r: Self) void {
            const d = r.d;
            const n = r.n;
            const T_ = d.shape.rows;
            const D = d.shape.n;
            const K = d.shape.k;
            const st = (K - 1) * D;
            const y = if (d.which == .short_conv) r.out[st..] else r.out;
            for (0..D) |ch| {
                if (d.which != .conv_state) for (0..T_) |t| {
                    for (0..K) |j| {
                        const src = @as(isize, @intCast(t + j)) - @as(isize, @intCast(K - 1));
                        const v = if (src >= 0) r.bx(@intCast(src), ch) else r.x(2, ch * (K - 1) + @as(usize, @intCast(@as(isize, @intCast(K - 1)) + src)));
                        r.terms[j] = n.mul(r.x(1, ch * K + j), v);
                    }
                    y[t * D + ch] = n.rb(n.mul(r.x(0, t * 3 * D + D + ch), n.rb(r.sum(K))));
                };
                if (d.which == .conv_rows) continue;
                for (0..K - 1) |j| {
                    const src = @as(isize, @intCast(T_ + j)) - @as(isize, @intCast(K - 1));
                    r.out[ch * (K - 1) + j] = if (src >= 0) r.bx(@intCast(src), ch) else r.x(2, ch * (K - 1) + j + T_);
                }
            }
        }

        fn bx(r: Self, row: usize, ch: usize) T {
            const D = r.d.shape.n;
            return r.n.rb(r.n.mul(r.x(0, row * 3 * D + ch), r.x(0, row * 3 * D + 2 * D + ch)));
        }

        /// Softmax attention over the row's visible keys, as one pass or (tiles) online over 64-key tiles.
        fn attention(r: Self) void {
            const d = r.d;
            const cf = d.cfg;
            const T_ = d.shape.rows;
            const kv = d.shape.k;
            const H = cf.heads;
            const hd = cf.hd;
            const nk = cf.kv_heads * hd;
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
            const pos0 = kv - T_;
            var n = r.n;
            n.slack = if (d.which == .attention_tiles) 16 + 8 * @as(f64, @floatFromInt(kv / 64 + 1)) else 4;
            const sc = r.terms[0..kv];
            for (0..T_) |t| for (0..H) |h| {
                const kh = h / (H / cf.kv_heads);
                const pos = pos0 + t;
                const hi = if (cf.causal) @min(pos + 1, kv) else kv;
                const lo = if (cf.causal and cf.window > 0) (pos + 1) -| cf.window else 0;
                var m: T = n.k(-std.math.inf(f64));
                for (lo..hi) |k| {
                    var s: T = n.k(0);
                    if (iv) {
                        var tl: [128]T = undefined;
                        for (0..hd) |i| tl[i] = n.mul(r.x(0, (t * H + h) * hd + i), r.x(1, k * nk + kh * hd + i));
                        s = n.sum(tl[0..hd], r.buf);
                    } else {
                        var tl: [128]f64 = undefined;
                        for (0..hd) |i| tl[i] = r.x(0, (t * H + h) * hd + i) * r.x(1, k * nk + kh * hd + i);
                        s = n.sum(tl[0..hd], r.buf);
                    }
                    sc[k] = n.mul(s, n.k(scale));
                    m = r.fmax(m, sc[k]);
                }
                for (lo..hi) |k| {
                    const z = n.sub(sc[k], m);
                    sc[k] = n.mono(z, num.exp, true, 2 + 1.2 * @abs(n.mid(z)));
                }
                const inv = n.div(n.k(1), n.sum(sc[lo..hi], r.buf));
                for (0..hd) |i| {
                    var acc: T = n.k(0);
                    for (lo..hi) |k| acc = n.add(acc, n.mul(sc[k], r.x(2, k * nk + kh * hd + i)));
                    r.out[(t * H + h) * hd + i] = n.out(n.mul(acc, inv));
                }
            };
        }
    };
}
