//! drama_codec_encode.cu's kernels, from the embedded fatbin, launched with their exact C argument types.

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("codec_kernels.zig");

pub const BiasMode = kern.BiasMode;
pub const Act = kern.Act;

/// drama_codec_encode.cu's EncodeGemmParams, field for field.
const GemmParams = extern struct {
    x: u64,
    w: u64,
    bias: u64,
    scale: u64,
    residual: u64,
    y: u64,
    m: i64,
    n: i32,
    cin: i32,
    taps: i32,
    dilation: i32,
    stride: i32,
    ldx: i32,
    ldy: i32,
    ldr: i32,
    bias_mode: i32,
    act: i32,
    flags: i32,
};

/// One GEMM over rows [0, m): BF16 x and blockedTaps w, FP32 scale, BF16 or FP32 residual and output.
pub const Gemm = struct {
    x: u64,
    w: u64,
    y: u64,
    m: usize,
    n: usize,
    cin: usize,
    ldx: usize,
    ldy: usize,
    taps: usize = 1,
    dilation: usize = 1,
    stride: usize = 1,
    bias: u64 = 0,
    bias_mode: BiasMode = .none,
    act: Act = .none,
    scale: u64 = 0,
    /// LayerScale's in-place mul_ on a BF16 tensor rounds; ConvNeXt's gamma * x promotes to FP32.
    round_scaled: bool = false,
    residual: u64 = 0,
    ldr: usize = 0,
    residual_f32: bool = false,
    out_f32: bool = false,
};

fn small(v: usize) !i32 {
    if (v > std.math.maxInt(i32)) return error.Invalid;
    return @intCast(v);
}

fn blocks(n: usize, per: usize) !u32 {
    const b = (n + per - 1) / per;
    if (b == 0 or b > std.math.maxInt(i32)) return error.Invalid;
    return @intCast(b);
}

pub const Kernels = struct {
    module: cuda.Module,
    conv_in: cuda.Function,
    snake_bf16: cuda.Function,
    snake_f32: cuda.Function,
    /// [input FP32][output FP32]
    rms_norm: [2][2]cuda.Function,
    rope: cuda.Function,
    attention: cuda.Function,
    silu_mul: cuda.Function,
    layer_norm: cuda.Function,
    dwconv7: cuda.Function,
    gemm_fn: cuda.Function,
    rvq: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var module = try cuda.Module.load(d, cuda.kernels.drama_codec_encode);
        errdefer module.unload();
        return .{
            .module = module,
            .conv_in = try module.function("drama_encode_conv_in"),
            .snake_bf16 = try module.function("drama_encode_snake"),
            .snake_f32 = try module.function("drama_encode_snake_f32"),
            .rms_norm = .{
                .{ try module.function("drama_encode_rms_norm"), try module.function("drama_encode_rms_norm_to_f32") },
                .{ try module.function("drama_encode_rms_norm_f32"), try module.function("drama_encode_rms_norm_f32_to_f32") },
            },
            .rope = try module.function("drama_encode_rope"),
            .attention = try module.function("drama_encode_window_attention"),
            .silu_mul = try module.function("drama_encode_silu_mul"),
            .layer_norm = try module.function("drama_encode_layer_norm"),
            .dwconv7 = try module.function("drama_encode_dwconv7"),
            .gemm_fn = try module.function("drama_encode_gemm"),
            .rvq = try module.function("drama_encode_rvq"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.module.unload();
        k.* = undefined;
    }

    fn run(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
    }

    /// audio [rows] FP32 -> y [rows][64] BF16.
    pub fn convIn(k: *const Kernels, s: cuda.Stream, audio: u64, w: u64, b: u64, y: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(audio);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(@as(i64, @intCast(rows)));
        try run(k.conv_in, s, .{ .x = try blocks(rows * 64, 256) }, 256, &a);
    }

    /// Snake over [rows][width] BF16 (or FP32 `x_f32`) into BF16; `fused` is TorchScript's FMA form.
    pub fn snake(k: *const Kernels, s: cuda.Stream, x: u64, x_f32: bool, alpha: u64, inv_alpha: u64, y: u64, rows: usize, width: usize, fused: bool) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(alpha);
        a.add(inv_alpha);
        a.add(y);
        a.add(@as(i64, @intCast(rows * width)));
        a.add(try small(width));
        a.add(@as(i32, @intFromBool(fused)));
        try run(if (x_f32) k.snake_f32 else k.snake_bf16, s, .{ .x = try blocks(rows * width, 256) }, 256, &a);
    }

    pub fn rmsNorm(k: *const Kernels, s: cuda.Stream, x: u64, x_f32: bool, w: u64, y: u64, y_f32: bool, rows: usize, width: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(y);
        a.add(try small(width));
        a.add(eps);
        try run(k.rms_norm[@intFromBool(x_f32)][@intFromBool(y_f32)], s, .{ .x = try blocks(rows, 1) }, 256, &a);
    }

    /// In place on the q and k thirds of qkv [rows][3 * width].
    pub fn ropeQk(k: *const Kernels, s: cuda.Stream, qkv: u64, table: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(table);
        a.add(try small(width));
        a.add(try small(3 * width));
        try run(k.rope, s, .{ .x = try blocks(rows, 1) }, 256, &a);
    }

    pub fn windowAttention(k: *const Kernels, s: cuda.Stream, qkv: u64, out: u64, rows: usize, heads: usize, window: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(out);
        a.add(@as(i32, 0));
        a.add(try small(heads));
        a.add(try small(window));
        a.add(scale);
        try run(k.attention, s, .{ .x = try blocks(rows, 1), .y = @intCast(heads) }, 128, &a);
    }

    pub fn siluMul(k: *const Kernels, s: cuda.Stream, gu: u64, out: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(gu);
        a.add(out);
        a.add(@as(i64, @intCast(rows)));
        a.add(try small(width));
        try run(k.silu_mul, s, .{ .x = try blocks(rows * width, 256) }, 256, &a);
    }

    pub fn layerNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(try small(width));
        a.add(eps);
        try run(k.layer_norm, s, .{ .x = try blocks(rows, 1) }, 256, &a);
    }

    pub fn dwconv(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(@as(i64, @intCast(rows)));
        a.add(try small(width));
        try run(k.dwconv7, s, .{ .x = try blocks(rows * width, 256) }, 256, &a);
    }

    pub fn gemm(k: *const Kernels, s: cuda.Stream, g: Gemm) !void {
        if (g.m == 0 or g.n == 0 or g.taps == 0 or g.cin == 0) return error.Invalid;
        if (g.cin % 32 != 0 or g.ldx % 8 != 0 or g.ldx < g.cin or g.ldy < g.n) return error.Invalid;
        if (g.x % 16 != 0 or g.w % 16 != 0) return error.Misaligned;
        if ((g.bias_mode != .none) != (g.bias != 0)) return error.Invalid;
        if (g.residual != 0 and g.ldr < g.n) return error.Invalid;
        const flags: i32 = @as(i32, @intFromBool(g.round_scaled)) | @as(i32, @intFromBool(g.residual_f32)) << 1 | @as(i32, @intFromBool(g.out_f32)) << 2;
        const p: GemmParams = .{
            .x = g.x,
            .w = g.w,
            .bias = g.bias,
            .scale = g.scale,
            .residual = g.residual,
            .y = g.y,
            .m = @intCast(g.m),
            .n = try small(g.n),
            .cin = try small(g.cin),
            .taps = try small(g.taps),
            .dilation = try small(g.dilation),
            .stride = try small(g.stride),
            .ldx = try small(g.ldx),
            .ldy = try small(g.ldy),
            .ldr = try small(g.ldr),
            .bias_mode = @backingInt(g.bias_mode),
            .act = @backingInt(g.act),
            .flags = flags,
        };
        _ = try small(g.taps * g.cin);
        var a: cuda.Args = .{};
        a.add(p);
        try run(k.gemm_fn, s, .{ .x = try blocks(g.m, 128), .y = try blocks(g.n, 128) }, 256, &a);
    }

    /// z [frames][1024] FP32 (consumed as the residual) -> codes [frames][10].
    pub fn rvqCodes(k: *const Kernels, s: cuda.Stream, z: u64, codes: u64, in_w: u64, in_b: u64, books: u64, books_sq: u64, books_raw: u64, out_w: u64, out_b: u64, frames: usize) !void {
        var a: cuda.Args = .{};
        a.add(z);
        a.add(codes);
        a.add(in_w);
        a.add(in_b);
        a.add(books);
        a.add(books_sq);
        a.add(books_raw);
        a.add(out_w);
        a.add(out_b);
        a.add(try small(frames));
        try run(k.rvq, s, .{ .x = try blocks(frames, 16) }, 128, &a);
    }
};
