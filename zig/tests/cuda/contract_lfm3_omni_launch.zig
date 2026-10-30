//! LFM3-Omni's CUDA contract launches: each kernel's grid, block, shared bytes and arguments as the engine makes them.
const cuda = @import("cuda");
const data = @import("contract_lfm3_omni_data.zig");

const Data = data.Data;
const eps = data.eps;
const nq = data.nq;
const size = data.size;

fn i(x: usize) i32 {
    return @intCast(x);
}

fn blocks(n: usize, per: usize) u32 {
    return @intCast((n + per - 1) / per);
}

pub const Launch = struct { grid: [2]u32, block: u32, shared: u32 = 0 };

/// The launch the engine makes for `d`, its arguments in `args`.
pub fn launch(d: *const Data, args: *cuda.Args) Launch {
    const sh = d.shape;
    const T = sh.rows;
    const cf = d.cfg;
    const p = struct {
        fn of(dd: *const Data, b: usize) u64 {
            return if (b < dd.nb) dd.ptr(b) else 0;
        }
    }.of;
    switch (d.which) {
        .rmsnorm_res, .rmsnorm_rows => {
            const D = sh.k;
            const res = d.which == .rmsnorm_res;
            for ([_]u64{ p(d, 0), if (res) p(d, 1) else 0, p(d, 2), p(d, 3) }) |x| args.add(x);
            for ([_]i32{ i(D), i(D), i(D) }) |x| args.add(x);
            args.add(eps);
            args.add(cf.mode);
            args.add(if (res) @as(i32, 0) else i(D));
            return .{ .grid = .{ @intCast(T), 1 }, .block = 256 };
        },
        .head_norm => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            for ([_]i32{ i(T), i(sh.n), i(sh.k), i(sh.n * sh.k) }) |x| args.add(x);
            args.add(eps);
            return .{ .grid = .{ blocks(T * sh.n * 32, 256), 1 }, .block = 256 };
        },
        .rope => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            for ([_]i32{ i(T), i(sh.n), i(sh.k), i(sh.n * sh.k), i(cf.pos0), cf.mode }) |x| args.add(x);
            args.add(@as(u64, 0));
            return .{ .grid = .{ blocks(T * sh.n * sh.k / 2, 256), 1 }, .block = 256 };
        },
        .copy_rows => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            for ([_]i32{ i(T), i(sh.n), i(sh.n), i(sh.n) }) |x| args.add(x);
            args.add(@as(u64, 0));
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .pos => {
            args.add(p(d, 0));
            args.add(@as(i32, 3));
            args.add(@as(i32, 0));
            return .{ .grid = .{ 1, 1 }, .block = 32 };
        },
        .short_conv, .conv_rows => {
            args.add(p(d, 0));
            args.add(i(3 * sh.n));
            args.add(p(d, 1));
            args.add(p(d, 2));
            args.add(p(d, 3));
            for ([_]i32{ i(sh.n), i(T), i(sh.n), i(sh.k) }) |x| args.add(x);
            return if (d.which == .short_conv) .{ .grid = .{ blocks(sh.n, 128), 1 }, .block = 128 } else .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .conv_state => {
            args.add(p(d, 0));
            args.add(i(3 * sh.n));
            args.add(p(d, 2));
            for ([_]i32{ i(T), i(sh.n), i(sh.k) }) |x| args.add(x);
            return .{ .grid = .{ blocks(sh.n, 128), 1 }, .block = 128 };
        },
        .attention, .attention_tiles => {
            const H = cf.heads;
            const hd = cf.hd;
            const nk = cf.kv_heads * hd;
            const kv = sh.k;
            args.add(p(d, 0));
            args.add(i(H * hd));
            args.add(p(d, 1));
            args.add(p(d, 2));
            args.add(i(nk));
            args.add(p(d, 3));
            args.add(i(H * hd));
            if (d.which == .attention_tiles) args.add(i(T));
            for ([_]i32{ i(H), i(cf.kv_heads), i(hd), i(kv - T), i(kv), i(cf.window), @intFromBool(cf.causal) }) |x| args.add(x);
            args.add(1.0 / @sqrt(@as(f32, @floatFromInt(hd))));
            if (d.which == .attention_tiles) return .{ .grid = .{ blocks(T, 16), @intCast(H) }, .block = 128, .shared = @intCast((16 * hd + 16 * 64) * 4 + 2 * 64 * (hd + 2) * 2) };
            args.add(@as(u64, 0));
            const span = if (cf.causal and cf.window > 0) @min(cf.window, kv) else kv;
            return .{ .grid = .{ @intCast(T * H), 1 }, .block = 128, .shared = @intCast((128 + span) * 4) };
        },
        .gate => {
            args.add(p(d, 0));
            args.add(i(sh.n * sh.k));
            args.add(p(d, 1));
            for ([_]i32{ i(sh.n), i(T), i(sh.n), i(sh.k) }) |x| args.add(x);
            return .{ .grid = .{ blocks(T * sh.n * sh.k, 256), 1 }, .block = 256 };
        },
        .swiglu => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            args.add(i(T));
            args.add(i(sh.n));
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .add => {
            for ([_]u64{ p(d, 0), p(d, 1), p(d, 2) }) |x| args.add(x);
            args.add(@as(i64, @intCast(T * sh.n)));
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .bias_act => {
            for ([_]u64{ p(d, 0), p(d, 1), p(d, 2) }) |x| args.add(x);
            for ([_]i32{ i(T), i(sh.n), @intFromBool(cf.per_row), cf.act }) |x| args.add(x);
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .layernorm => {
            for ([_]u64{ p(d, 0), p(d, 1), p(d, 2), p(d, 3) }) |x| args.add(x);
            args.add(i(sh.k));
            args.add(eps);
            return .{ .grid = .{ @intCast(T), 1 }, .block = 256 };
        },
        .embed => {
            for ([_]u64{ p(d, 0), p(d, 1), p(d, 2) }) |x| args.add(x);
            args.add(i(T));
            args.add(i(sh.n));
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .embed_codes => {
            for ([_]u64{ p(d, 0), p(d, 1), p(d, 2) }) |x| args.add(x);
            for ([_]i32{ i(T), i(sh.n), nq, size }) |x| args.add(x);
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .depth_in => {
            for ([_]u64{ p(d, 0), p(d, 1), p(d, 2), p(d, 3), p(d, 4), p(d, 5) }) |x| args.add(x);
            for ([_]i32{ nq, i(sh.n), size }) |x| args.add(x);
            return .{ .grid = .{ blocks(nq * sh.n, 256), 1 }, .block = 256 };
        },
        .conv2d => {
            const C = sh.rows;
            const H = cf.height;
            args.add(p(d, 0));
            for ([_]i32{ i(cf.cin), i(H), i(2 * sh.n) }) |x| args.add(x);
            args.add(p(d, 1));
            args.add(p(d, 2));
            args.add(p(d, 3));
            for ([_]i32{ i(C), i(H / 2), i(sh.n), cf.act }) |x| args.add(x);
            return .{ .grid = .{ blocks(C * (H / 2) * sh.n, 256), 1 }, .block = 256 };
        },
        .f32_bf16, .bf16_f32 => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            args.add(@as(i64, @intCast(T * sh.n)));
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .fsq => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            args.add(i(T));
            args.add(i(sh.n));
            return .{ .grid = .{ blocks(T * sh.n, 128), 1 }, .block = 128 };
        },
        .repeat => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            for ([_]i32{ i(T), i(sh.n), i(sh.k) }) |x| args.add(x);
            return .{ .grid = .{ blocks(T * sh.n * sh.k, 256), 1 }, .block = 256 };
        },
        .polar => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            args.add(i(T));
            args.add(i(sh.n));
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .argmax => {
            args.add(p(d, 0));
            args.add(i(sh.n));
            args.add(p(d, 1));
            args.add(@as(i32, 4));
            args.add(p(d, 2));
            return .{ .grid = .{ 1, 1 }, .block = 1024 };
        },
        .gemv => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            for ([_]i32{ i(sh.k), i(sh.k), i(sh.n), i(T) }) |x| args.add(x);
            args.add(if (cf.f32_out) @as(u64, 0) else p(d, 2));
            args.add(i(sh.n));
            args.add(if (cf.f32_out) p(d, 2) else @as(u64, 0));
            for ([_]i64{ 0, 0, 0 }) |x| args.add(x);
            return .{ .grid = .{ blocks(sh.n, 8), 1 }, .block = 256, .shared = @intCast(T * sh.k * 2) };
        },
        .depth_pick => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            args.add(@as(i32, 4));
            args.add(p(d, 2));
            args.add(i(sh.k));
            args.add(@as(f32, 1.2));
            args.add(cf.temp);
            args.add(@as(u64, 0x5eed));
            args.add(@as(i32, 17));
            args.add(p(d, 3));
            args.add(p(d, 4));
            args.add(@as(i32, nq));
            args.add(@as(i32, size));
            args.add(@as(u64, 0));
            return .{ .grid = .{ @intCast(T), 1 }, .block = 256, .shared = size };
        },
        .frame_end => {
            args.add(p(d, 0));
            args.add(@as(i32, nq));
            args.add(@as(i32, size - 1));
            return .{ .grid = .{ 1, 1 }, .block = 32 };
        },
        .hist_push => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            args.add(@as(i32, 5));
            args.add(@as(i32, nq));
            return .{ .grid = .{ 1, 1 }, .block = 32 };
        },
        .mel_frames => {
            args.add(p(d, 0));
            args.add(i(d.bufs[0].host.len / 4));
            args.add(p(d, 1));
            args.add(p(d, 2));
            for ([_]i32{ i(T), i(sh.n), i(sh.k) }) |x| args.add(x);
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
        .power, .logmel => {
            args.add(p(d, 0));
            args.add(p(d, 1));
            args.add(i(T));
            args.add(i(sh.n));
            if (d.which == .logmel) args.add(@as(f32, -12));
            return .{ .grid = .{ blocks(T * sh.n, 256), 1 }, .block = 256 };
        },
    }
}
