//! LFM3-Omni's CUDA contract inputs: each kernel's buffers at a shape, filled for a gate, and its launch.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cases = @import("contract_lfm3_omni_cases.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;
const Which = cases.Which;

pub const Elem = enum { bf16, f32, i32 };
/// An input stays as uploaded; an output starts poisoned; an in-out output gets its prepared bytes before each run.
pub const Role = enum { in, out, inout };

pub const Buf = struct { elem: Elem, role: Role, host: []u8, dev: u64 = 0 };

pub const size = 2017;
pub const nq = 12;
pub const eps: f32 = 1e-5;

pub const Data = struct {
    which: Which,
    cfg: cases.Cfg,
    shape: c.Shape,
    fill: g.Fill,
    bufs: [8]Buf = undefined,
    nb: usize = 0,

    pub fn add(d: *Data, a: std.mem.Allocator, elem: Elem, role: Role, n: usize) ![]u8 {
        const words = try a.alloc(u32, n + 1);
        const host = std.mem.sliceAsBytes(words)[0 .. n * @as(usize, if (elem == .bf16) 2 else 4)];
        @memset(host, 0);
        d.bufs[d.nb] = .{ .elem = elem, .role = role, .host = host };
        d.nb += 1;
        return host;
    }

    pub fn bf(d: *const Data, b: usize) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, d.bufs[b].host));
    }

    pub fn fl(d: *const Data, b: usize) []f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, d.bufs[b].host));
    }

    pub fn ints(d: *const Data, b: usize) []i32 {
        return @alignCast(std.mem.bytesAsSlice(i32, d.bufs[b].host));
    }

    /// Element i of buffer b as prepared, in float64.
    pub fn v(d: *const Data, b: usize, i: usize) f64 {
        return switch (d.bufs[b].elem) {
            .bf16 => bfr.value(d.bf(b)[i]),
            .f32 => d.fl(b)[i],
            .i32 => @floatFromInt(d.ints(b)[i]),
        };
    }

    pub fn ptr(d: *const Data, b: usize) u64 {
        return d.bufs[b].dev;
    }
};

fn rand(rng: std.Random, lo: f32, hi: f32) f32 {
    return lo + rng.float(f32) * (hi - lo);
}

fn fillBf(rng: std.Random, xs: []u16, lo: f32, hi: f32) void {
    for (xs) |*x| x.* = bfr.bits(rand(rng, lo, hi));
}

/// Small integers, so every fp32 sum of their products is exact in any order (gate 4's inputs).
fn fillInt(rng: std.Random, xs: []u16, top: u32) void {
    for (xs) |*x| x.* = bfr.bits(@floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 2 * top + 1))) - @as(i32, @intCast(top))));
}

/// NaN, +Inf and -Inf at three places of a bf16 or fp32 buffer (one row each where rows are `row` apart).
fn plant(d: *Data, b: usize, row: usize) void {
    const n = d.bufs[b].host.len / @as(usize, if (d.bufs[b].elem == .bf16) 2 else 4);
    const at = [3]usize{ @min(n - 1, row + 1), @min(n - 1, 2 * row + 3), @min(n - 1, 3 * row + 5) };
    const vals = [3]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, vals) |i, x| switch (d.bufs[b].elem) {
        .bf16 => d.bf(b)[i] = bfr.bits(x),
        .f32 => d.fl(b)[i] = x,
        .i32 => {},
    };
}

/// The buffers of `d.which` at `d.shape`, filled for `d.fill`; false when the kernel cannot take the shape.
pub fn prepare(d: *Data, a: std.mem.Allocator, seed: u64) !bool {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const sh = d.shape;
    const T = sh.rows;
    const stress = d.fill == .stress;
    const nf = d.fill == .nonfinite;
    switch (d.which) {
        .rmsnorm_res, .rmsnorm_rows => {
            const D = sh.k;
            const res = d.which == .rmsnorm_res;
            _ = try d.add(a, .bf16, if (res) .inout else .in, T * D);
            _ = try d.add(a, .bf16, .in, if (res) T * D else 0);
            _ = try d.add(a, .bf16, .in, if (res) D else T * D);
            _ = try d.add(a, .bf16, .out, T * D);
            if (stress) {
                fillInt(rng, d.bf(0), 32);
                fillInt(rng, d.bf(1), 31);
            } else {
                fillBf(rng, d.bf(0), -3, 3);
                fillBf(rng, d.bf(1), -2, 2);
            }
            fillBf(rng, d.bf(2), 0.5, 1.5);
            if (nf) plant(d, 0, D);
        },
        .head_norm => {
            const n = T * sh.n * sh.k;
            _ = try d.add(a, .bf16, .inout, n);
            _ = try d.add(a, .bf16, .in, sh.k);
            if (stress) fillInt(rng, d.bf(0), 64) else fillBf(rng, d.bf(0), -4, 4);
            fillBf(rng, d.bf(1), 0.5, 1.5);
            if (nf) plant(d, 0, sh.k);
        },
        .rope => {
            if (sh.k % 2 != 0) return false;
            _ = try d.add(a, .bf16, .inout, T * sh.n * sh.k);
            _ = try d.add(a, .f32, .in, sh.k / 2);
            fillBf(rng, d.bf(0), -3, 3);
            const rot: usize = @as(usize, @intFromFloat(d.cfg.fraction * @as(f32, @floatFromInt(sh.k)))) / 2;
            for (d.fl(1), 0..) |*x, i| x.* = if (i < rot) 1.0 / std.math.pow(f32, d.cfg.theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(sh.k))) else 0;
            if (nf) plant(d, 0, sh.k);
        },
        .copy_rows, .bf16_f32 => {
            _ = try d.add(a, .bf16, .in, T * sh.n);
            _ = try d.add(a, if (d.which == .bf16_f32) .f32 else .bf16, .out, T * sh.n);
            fillBf(rng, d.bf(0), -100, 100);
            if (nf) plant(d, 0, sh.n);
        },
        .pos => {
            _ = try d.add(a, .i32, .inout, 1);
            d.ints(0)[0] = @intCast(rng.uintLessThan(u32, 4000));
        },
        .short_conv, .conv_rows, .conv_state => {
            const D = sh.n;
            const K = sh.k;
            if (K < 2 or K > 8) return false;
            _ = try d.add(a, .bf16, .in, T * 3 * D);
            _ = try d.add(a, .bf16, .in, D * K);
            _ = try d.add(a, .bf16, if (d.which == .conv_rows) .in else .inout, D * (K - 1));
            if (d.which != .conv_state) _ = try d.add(a, .bf16, .out, T * D);
            if (stress) {
                fillInt(rng, d.bf(0), 16);
                fillInt(rng, d.bf(1), 16);
                fillInt(rng, d.bf(2), 64);
            } else {
                fillBf(rng, d.bf(0), -2, 2);
                fillBf(rng, d.bf(1), -1, 1);
                fillBf(rng, d.bf(2), -2, 2);
            }
            if (nf) plant(d, 0, 3 * D);
        },
        .attention, .attention_tiles => {
            const H = d.cfg.heads;
            const hd = d.cfg.hd;
            const kv = sh.k;
            if (H != sh.n or T > kv) return false;
            _ = try d.add(a, .bf16, .in, T * H * hd);
            _ = try d.add(a, .bf16, .in, kv * d.cfg.kv_heads * hd);
            _ = try d.add(a, .bf16, .in, kv * d.cfg.kv_heads * hd);
            _ = try d.add(a, .bf16, .out, T * H * hd);
            if (stress) {
                fillInt(rng, d.bf(0), 2);
                fillInt(rng, d.bf(1), 2);
                fillInt(rng, d.bf(2), 64);
            } else {
                fillBf(rng, d.bf(0), -2, 2);
                fillBf(rng, d.bf(1), -2, 2);
                fillBf(rng, d.bf(2), -3, 3);
            }
            if (nf) {
                // the last keys, which every causal or sliding row of the shape sees
                const nk = d.cfg.kv_heads * hd;
                d.bf(2)[(kv - 1) * nk + 1] = bfr.bits(std.math.nan(f32));
                d.bf(2)[(kv - 2) * nk + 3] = bfr.bits(std.math.inf(f32));
                d.bf(2)[(kv - 3) * nk + 5] = bfr.bits(-std.math.inf(f32));
            }
        },
        .gate => {
            _ = try d.add(a, .bf16, .inout, T * sh.n * sh.k);
            _ = try d.add(a, .bf16, .in, T * sh.n);
            fillBf(rng, d.bf(0), -4, 4);
            fillBf(rng, d.bf(1), -6, 6);
            if (nf) plant(d, 1, sh.n);
        },
        .swiglu => {
            _ = try d.add(a, .bf16, .in, T * 2 * sh.n);
            _ = try d.add(a, .bf16, .out, T * sh.n);
            fillBf(rng, d.bf(0), -6, 6);
            if (nf) plant(d, 0, 2 * sh.n);
        },
        .add => {
            _ = try d.add(a, .bf16, .out, T * sh.n);
            _ = try d.add(a, .bf16, .in, T * sh.n);
            _ = try d.add(a, .bf16, .in, T * sh.n);
            fillBf(rng, d.bf(1), -300, 300);
            fillBf(rng, d.bf(2), -3, 3);
            if (nf) plant(d, 1, sh.n);
        },
        .bias_act => {
            _ = try d.add(a, .f32, .in, T * sh.n);
            _ = try d.add(a, .bf16, .in, if (d.cfg.per_row) T else sh.n);
            _ = try d.add(a, .bf16, .out, T * sh.n);
            for (d.fl(0)) |*x| x.* = rand(rng, -5, 5);
            fillBf(rng, d.bf(1), -1, 1);
            if (nf) plant(d, 0, sh.n);
        },
        .layernorm => {
            const D = sh.k;
            _ = try d.add(a, .bf16, .in, T * D);
            _ = try d.add(a, .bf16, .in, D);
            _ = try d.add(a, .bf16, .in, D);
            _ = try d.add(a, .bf16, .out, T * D);
            if (stress) fillInt(rng, d.bf(0), 64) else fillBf(rng, d.bf(0), -3, 5);
            fillBf(rng, d.bf(1), 0.5, 1.5);
            fillBf(rng, d.bf(2), -0.5, 0.5);
            if (nf) plant(d, 0, D);
        },
        .embed => {
            const V = 64;
            _ = try d.add(a, .bf16, .in, V * sh.n);
            _ = try d.add(a, .i32, .in, T);
            _ = try d.add(a, .bf16, .inout, T * sh.n);
            fillBf(rng, d.bf(0), -1, 1);
            fillBf(rng, d.bf(2), -9, 9);
            for (d.ints(1), 0..) |*x, t| x.* = if (t % 5 == 3) -1 else @intCast(rng.uintLessThan(u32, V));
            if (nf) {
                d.ints(1)[0] = 1;
                d.ints(1)[1] = 2;
                plant(d, 0, sh.n);
            }
        },
        .embed_codes, .depth_in => {
            const D = sh.n;
            const din = d.which == .depth_in;
            if (din) {
                _ = try d.add(a, .bf16, .in, nq * D);
                _ = try d.add(a, .bf16, .in, D);
            }
            const tb = d.nb;
            _ = try d.add(a, .bf16, .in, nq * size * D);
            const cb = d.nb;
            _ = try d.add(a, .i32, .in, (if (din) 1 else T) * nq);
            if (din) _ = try d.add(a, .i32, .in, nq);
            _ = try d.add(a, .bf16, .out, (if (din) nq else T) * D);
            for (d.ints(cb)) |*x| x.* = @intCast(rng.uintLessThan(u32, size));
            if (din) {
                fillBf(rng, d.bf(0), -2, 2);
                fillBf(rng, d.bf(1), -1, 1);
                for (d.ints(cb + 1), 0..) |*x, q| x.* = @intFromBool(q % 3 == 1);
            }
            if (stress) {
                // half the codebooks' rows large and the rest their negatives plus a little: fp16 overflows, bf16 loses
                const tab = d.bf(tb);
                for (0..T) |t| for (0..nq) |q| {
                    const row = (@as(usize, @intCast(d.ints(cb)[t * nq + q])) + q * size) * D;
                    for (0..D) |col| tab[row + col] = bfr.bits(if (q < nq / 2) @as(f32, @floatFromInt(30720 + 256 * ((col + q) % 8))) else -@as(f32, @floatFromInt(30720 + 256 * ((col + q - nq / 2) % 8))) + @as(f32, if (q == nq - 1) 256 else 0));
                };
            } else fillBf(rng, d.bf(tb), -1, 1);
            if (nf) {
                const tab = d.bf(tb);
                const row0 = @as(usize, @intCast(d.ints(cb)[0])) * D;
                tab[row0 + 1] = bfr.bits(std.math.nan(f32));
                tab[row0 + 3] = bfr.bits(std.math.inf(f32));
                tab[row0 + 5] = bfr.bits(-std.math.inf(f32));
            }
        },
        .conv2d => {
            const C = sh.rows;
            const W2 = sh.n;
            const H = d.cfg.height;
            const W = 2 * W2;
            _ = try d.add(a, .bf16, .in, (if (d.cfg.cin == 1) 1 else C) * H * W);
            _ = try d.add(a, .bf16, .in, C * 9);
            _ = try d.add(a, .bf16, .in, C);
            _ = try d.add(a, .bf16, .out, C * (H / 2) * W2);
            if (stress) {
                fillInt(rng, d.bf(0), 3);
                fillInt(rng, d.bf(1), 3);
            } else {
                fillBf(rng, d.bf(0), -2, 2);
                fillBf(rng, d.bf(1), -1, 1);
            }
            fillBf(rng, d.bf(2), -1, 1);
            if (nf) plant(d, 0, W);
        },
        .f32_bf16 => {
            _ = try d.add(a, .f32, .in, T * sh.n);
            _ = try d.add(a, .bf16, .out, T * sh.n);
            for (d.fl(0), 0..) |*x, i| {
                const v = bfr.bits(rand(rng, -4, 4));
                // every seventh an exact tie between two bf16 values
                x.* = if (i % 7 == 0) (bfr.value(v) + bfr.value(v + 1)) / 2 else rand(rng, -40, 4);
            }
            if (nf) plant(d, 0, sh.n);
        },
        .fsq => {
            _ = try d.add(a, .i32, .in, T * sh.n);
            _ = try d.add(a, .bf16, .out, T * sh.n * 4);
            for (d.ints(0)) |*x| x.* = @intCast(rng.uintLessThan(u32, 8 * 7 * 6 * 6));
        },
        .repeat => {
            _ = try d.add(a, .bf16, .in, T * sh.n);
            _ = try d.add(a, .bf16, .out, T * sh.n * sh.k);
            fillBf(rng, d.bf(0), -5, 5);
            if (nf) plant(d, 0, sh.n);
        },
        .polar => {
            _ = try d.add(a, .bf16, .in, T * 2 * sh.n);
            _ = try d.add(a, .f32, .out, T * 2 * sh.n);
            for (0..T) |t| {
                fillBf(rng, d.bf(0)[t * 2 * sh.n ..][0..sh.n], -6, 3);
                fillBf(rng, d.bf(0)[t * 2 * sh.n + sh.n ..][0..sh.n], -3.2, 3.2);
            }
            if (nf) plant(d, 0, 2 * sh.n);
        },
        .argmax => {
            _ = try d.add(a, .bf16, .in, sh.n);
            _ = try d.add(a, .i32, .in, 4);
            _ = try d.add(a, .i32, .out, 1);
            fillBf(rng, d.bf(0), -8, 8);
            const top = bfr.bits(30);
            // ties: the best value at several ids, the first suppressed
            for ([_]usize{ 777, 9000, 40000, 127999 }) |i| d.bf(0)[i % sh.n] = top;
            for (d.ints(1), 0..) |*x, i| x.* = @intCast((777 + 4096 * i) % sh.n);
            if (nf) {
                d.bf(0)[5] = bfr.bits(std.math.nan(f32));
                d.bf(0)[sh.n / 2] = bfr.bits(std.math.inf(f32));
                d.bf(0)[sh.n - 3] = bfr.bits(std.math.inf(f32));
            }
        },
        .gemv => {
            if (d.shape.rows > d.cfg.tm) d.shape.rows = d.cfg.tm;
            const Tg = d.shape.rows;
            const K = sh.k;
            if (K % 8 != 0 or Tg > d.cfg.tm or Tg * K * 2 > 96 << 10) return false;
            _ = try d.add(a, .bf16, .in, sh.n * K);
            _ = try d.add(a, .bf16, .in, Tg * K);
            _ = try d.add(a, if (d.cfg.f32_out) .f32 else .bf16, .out, Tg * sh.n);
            if (stress) {
                // first half's products all positive and large, the second half their negatives but a few
                const h = K / 2;
                for (0..Tg) |t| for (0..h) |i| {
                    const x: f32 = @floatFromInt(16 + rng.uintLessThan(u32, 16));
                    d.bf(1)[t * K + i] = bfr.bits(x);
                    d.bf(1)[t * K + h + i] = bfr.bits(x);
                };
                for (0..sh.n) |n| for (0..h) |i| {
                    const w: f32 = @floatFromInt(32 + rng.uintLessThan(u32, 32));
                    d.bf(0)[n * K + i] = bfr.bits(w);
                    d.bf(0)[n * K + h + i] = bfr.bits(-w + @as(f32, if (i % 5 == 0) 1 else 0));
                };
            } else {
                fillBf(rng, d.bf(0), -0.1, 0.1);
                fillBf(rng, d.bf(1), -2, 2);
            }
            if (nf) plant(d, 0, K);
        },
        .depth_pick => {
            _ = try d.add(a, .bf16, .in, nq * size);
            _ = try d.add(a, .i32, .in, nq);
            _ = try d.add(a, .i32, .in, 64 * nq);
            _ = try d.add(a, .i32, .inout, nq);
            _ = try d.add(a, .i32, .inout, nq);
            fillBf(rng, d.bf(0), -6, 6);
            for (d.ints(1), 0..) |*x, i| x.* = @intCast((i * 5) % nq);
            for (d.ints(2)) |*x| x.* = @intCast(rng.uintLessThan(u32, size));
            for (d.ints(4)) |*x| x.* = 1;
            // ties at each codebook's best, one of them in the repetition window
            for (0..nq) |q| for ([_]usize{ 100, 700, 1900 }) |i| {
                d.bf(0)[q * size + i] = bfr.bits(9);
            };
            d.ints(2)[0] = 100;
            if (nf) for (0..nq) |q| {
                d.bf(0)[q * size + 3] = bfr.bits(std.math.nan(f32));
                if (q % 2 == 0) d.bf(0)[q * size + 1500] = bfr.bits(std.math.inf(f32));
            };
        },
        .frame_end => {
            _ = try d.add(a, .i32, .inout, nq);
            for (d.ints(0)) |*x| x.* = @intCast(rng.uintLessThan(u32, size - 1));
            if (seed % 2 == 0) d.ints(0)[@intCast(seed % nq)] = size - 1;
        },
        .hist_push => {
            _ = try d.add(a, .i32, .in, nq);
            _ = try d.add(a, .i32, .inout, sh.k * nq);
            for (d.ints(0)) |*x| x.* = @intCast(rng.uintLessThan(u32, size));
            for (d.ints(1)) |*x| x.* = @intCast(rng.uintLessThan(u32, size));
        },
        .mel_frames => {
            const len = (T - 1) * sh.k + sh.n - 37;
            _ = try d.add(a, .f32, .in, len);
            _ = try d.add(a, .f32, .in, sh.n);
            _ = try d.add(a, .f32, .out, T * sh.n);
            for (d.fl(0)) |*x| x.* = rand(rng, -1, 1);
            for (d.fl(1), 0..) |*x, i| x.* = @floatCast(0.5 - 0.5 * @cos(2 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(sh.n))));
            if (nf) plant(d, 0, sh.k);
        },
        .power => {
            _ = try d.add(a, .f32, .in, T * 2 * sh.n);
            _ = try d.add(a, .f32, .out, T * sh.n);
            for (d.fl(0)) |*x| x.* = rand(rng, -30, 30);
            if (nf) plant(d, 0, 2 * sh.n);
        },
        .logmel => {
            _ = try d.add(a, .f32, .in, T * sh.n);
            _ = try d.add(a, .f32, .out, T * sh.n);
            for (d.fl(0), 0..) |*x, i| x.* = if (i % 9 == 0) 1e-14 * rng.float(f32) else std.math.pow(f32, 10, rand(rng, -11, 3));
            if (nf) plant(d, 0, sh.n);
        },
    }
    return true;
}

