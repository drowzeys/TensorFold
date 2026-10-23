//! The codec encoder's geometry and weight image: BF16 where autocast casts a conv or linear weight, else FP32.

const std = @import("std");
const load = @import("codec_load.zig");

const Allocator = std.mem.Allocator;

pub const latent = load.latent;
pub const codebook_dim = load.codebook_dim;
pub const codebooks = load.codebooks;
pub const semantic_size = load.semantic_size;
pub const residual_size = load.residual_size;
pub const ffn = load.ffn;
pub const convnext_hidden = load.convnext_hidden;
pub const dilations = load.dilations;
pub const in_channels = 64; // encoder_dim
pub const strides = [_]usize{ 2, 4, 8, 8 }; // encoder_rates
pub const encoder_layers = 4; // encoder_transformer_layers [0, 0, 0, 4]
pub const pre_layers = load.layers;
/// EncoderBlock's window_size: getattr on the transformer config partial falls back to 512.
pub const encoder_window = 512;
pub const pre_window = load.window;
pub const downsample_stages = load.upsample_stages;
pub const hop = blk: {
    var h: usize = 1;
    for (strides) |s| h *= s;
    break :blk h;
};
pub const samples_per_frame = load.samples_per_frame;
/// The VQ distance MMA's K: the codebook dimension zero-padded to 16.
pub const book_width = 16;
pub const book_rows = semantic_size + (codebooks - 1) * residual_size;

comptime {
    std.debug.assert(hop << downsample_stages == samples_per_frame);
}

pub const Snake = load.Snake;
pub const Unit = load.Unit;

/// EncoderBlock: three residual units at half width, Snake, then the strided conv to full width.
pub const Block = struct { units: [dilations.len]Unit, snake: Snake, conv: u64, conv_b: u64 };

/// A transformer layer: BF16 projections, FP32 norm weights and layer scales.
pub const Layer = load.Layer;

/// rvq.py's downsample stage: CausalConvNet(kernel 2, stride 2), then ConvNeXtBlock.
pub const Downsample = struct {
    conv: u64,
    conv_b: u64,
    dw: u64,
    dw_b: u64,
    norm_w: u64,
    norm_b: u64,
    pw1: u64,
    pw1_b: u64,
    pw2: u64,
    pw2_b: u64,
    gamma: u64,
};

pub const Rvq = struct { in_w: u64, in_b: u64, books: u64, books_sq: u64, books_raw: u64, out_w: u64, out_b: u64 };

/// Byte offsets into the one device weight buffer.
pub const Weights = struct {
    conv_in: u64,
    conv_in_b: u64,
    blocks: [strides.len]Block,
    layers: [encoder_layers]Layer,
    norm: u64,
    out_snake: Snake,
    out_conv: u64,
    out_conv_b: u64,
    downsample: [downsample_stages]Downsample,
    pre: [pre_layers]Layer,
    pre_norm: u64,
    rvq: Rvq,
    rope: u64,
};

/// Block of input channels cuDNN's BF16 implicit GEMM sums whole before the next tap.
pub const channel_block = 32;

/// Conv1d weight [cout][cin][k] as the encode GEMM's [cout][cin / 32][k][32]: its K order, as cuDNN sums it.
pub fn blockedTaps(gpa: Allocator, w: []const f32, cout: usize, cin: usize, k: usize) ![]f32 {
    if (w.len != cout * cin * k or cin % channel_block != 0) return error.Invalid;
    const out = try gpa.alloc(f32, w.len);
    for (0..cout) |co| {
        for (0..cin) |ci| {
            const block = ci / channel_block;
            for (0..k) |t| out[((co * (cin / channel_block) + block) * k + t) * channel_block + ci % channel_block] = w[(co * cin + ci) * k + t];
        }
    }
    return out;
}

/// F.normalize(book) and its sum of squares in torch's CUDA order: squares, then (i, i+4), (i, i+2), (0, 1).
pub fn normalizeRow(row: *const [codebook_dim]f32, out: *[codebook_dim]f32) f32 {
    const norm = @max(@sqrt(treeSquares(row)), 1e-12);
    for (out, row) |*o, v| o.* = v / norm;
    return treeSquares(out);
}

fn treeSquares(row: *const [codebook_dim]f32) f32 {
    var s: [codebook_dim]f32 = undefined;
    for (&s, row) |*o, v| o.* = v * v;
    var n: usize = codebook_dim / 2;
    while (n > 0) : (n /= 2) {
        for (s[0..n], s[n .. 2 * n]) |*a, b| a.* += b;
    }
    return s[0];
}

pub const Loader = struct {
    l: *load.Loader,

    /// A weight-normed causal conv (parametrizations.weight_norm) at `prefix`.conv, folded, as BF16 blocked taps.
    fn conv(e: Loader, prefix: []const u8, cout: usize, cin: usize, k: usize) !u64 {
        const l = e.l;
        const v = try l.folded(
            try l.name("{s}.conv.parametrizations.weight.original0", .{prefix}),
            try l.name("{s}.conv.parametrizations.weight.original1", .{prefix}),
            &[_]u64{ cout, cin, k },
        );
        defer l.gpa.free(v);
        const t = try blockedTaps(l.gpa, v, cout, cin, k);
        defer l.gpa.free(t);
        return l.stage.bfloats(t);
    }

    fn bfloat(e: Loader, tensor: []const u8, shape: []const u64) !u64 {
        const v = try e.l.read(tensor, shape);
        defer e.l.gpa.free(v);
        return e.l.stage.bfloats(v);
    }

    fn float(e: Loader, tensor: []const u8, shape: []const u64) !u64 {
        const v = try e.l.read(tensor, shape);
        defer e.l.gpa.free(v);
        return e.l.stage.floats(v);
    }

    fn layer(e: Loader, prefix: []const u8) !Layer {
        const l = e.l;
        const w13 = try l.gpa.alloc(f32, 2 * ffn * latent);
        defer l.gpa.free(w13);
        const w1 = try l.read(try l.name("{s}.feed_forward.w1.weight", .{prefix}), &[_]u64{ ffn, latent });
        defer l.gpa.free(w1);
        const w3 = try l.read(try l.name("{s}.feed_forward.w3.weight", .{prefix}), &[_]u64{ ffn, latent });
        defer l.gpa.free(w3);
        @memcpy(w13[0 .. ffn * latent], w1);
        @memcpy(w13[ffn * latent ..], w3);
        return .{
            .wqkv = try e.bfloat(try l.name("{s}.attention.wqkv.weight", .{prefix}), &[_]u64{ 3 * latent, latent }),
            .wo = try e.bfloat(try l.name("{s}.attention.wo.weight", .{prefix}), &[_]u64{ latent, latent }),
            .w13 = try l.stage.bfloats(w13),
            .w2 = try e.bfloat(try l.name("{s}.feed_forward.w2.weight", .{prefix}), &[_]u64{ latent, ffn }),
            .attention_norm = try e.float(try l.name("{s}.attention_norm.weight", .{prefix}), &[_]u64{latent}),
            .ffn_norm = try e.float(try l.name("{s}.ffn_norm.weight", .{prefix}), &[_]u64{latent}),
            .attention_scale = try e.float(try l.name("{s}.attention_layer_scale.gamma", .{prefix}), &[_]u64{latent}),
            .ffn_scale = try e.float(try l.name("{s}.ffn_layer_scale.gamma", .{prefix}), &[_]u64{latent}),
        };
    }

    fn downsample(e: Loader, prefix: []const u8) !Downsample {
        const l = e.l;
        const v = try l.read(try l.name("{s}.0.conv.weight", .{prefix}), &[_]u64{ latent, latent, 2 });
        defer l.gpa.free(v);
        const t = try blockedTaps(l.gpa, v, latent, latent, 2);
        defer l.gpa.free(t);
        return .{
            .conv = try l.stage.bfloats(t),
            .conv_b = try e.bfloat(try l.name("{s}.0.conv.bias", .{prefix}), &[_]u64{latent}),
            .dw = try e.bfloat(try l.name("{s}.1.dwconv.conv.weight", .{prefix}), &[_]u64{ latent, 1, 7 }),
            .dw_b = try e.bfloat(try l.name("{s}.1.dwconv.conv.bias", .{prefix}), &[_]u64{latent}),
            .norm_w = try e.float(try l.name("{s}.1.norm.weight", .{prefix}), &[_]u64{latent}),
            .norm_b = try e.float(try l.name("{s}.1.norm.bias", .{prefix}), &[_]u64{latent}),
            .pw1 = try e.bfloat(try l.name("{s}.1.pwconv1.weight", .{prefix}), &[_]u64{ convnext_hidden, latent }),
            .pw1_b = try e.bfloat(try l.name("{s}.1.pwconv1.bias", .{prefix}), &[_]u64{convnext_hidden}),
            .pw2 = try e.bfloat(try l.name("{s}.1.pwconv2.weight", .{prefix}), &[_]u64{ latent, convnext_hidden }),
            .pw2_b = try e.bfloat(try l.name("{s}.1.pwconv2.bias", .{prefix}), &[_]u64{latent}),
            .gamma = try e.float(try l.name("{s}.1.gamma", .{prefix}), &[_]u64{latent}),
        };
    }

    /// Each book's folded in_proj and out_proj (legacy weight_norm: weight_g, weight_v) and its codebook three ways.
    fn rvq(e: Loader) !Rvq {
        const l = e.l;
        const gpa = l.gpa;
        const in_w = try gpa.alloc(f32, codebooks * codebook_dim * latent);
        defer gpa.free(in_w);
        const in_b = try gpa.alloc(f32, codebooks * codebook_dim);
        defer gpa.free(in_b);
        const out_w = try gpa.alloc(f32, codebooks * latent * codebook_dim);
        defer gpa.free(out_w);
        const out_b = try gpa.alloc(f32, codebooks * latent);
        defer gpa.free(out_b);
        const raw = try gpa.alloc(f32, book_rows * codebook_dim);
        defer gpa.free(raw);
        const padded = try gpa.alloc(f32, book_rows * book_width);
        defer gpa.free(padded);
        @memset(padded, 0);
        const sq = try gpa.alloc(f32, book_rows);
        defer gpa.free(sq);
        var row: usize = 0;
        for (0..codebooks) |i| {
            const prefix = if (i == 0) "quantizer.semantic_quantizer.quantizers.0" else try l.name("quantizer.quantizer.quantizers.{d}", .{i - 1});
            const size: usize = if (i == 0) semantic_size else residual_size;
            const vi = try l.folded(try l.name("{s}.in_proj.weight_g", .{prefix}), try l.name("{s}.in_proj.weight_v", .{prefix}), &[_]u64{ codebook_dim, latent, 1 });
            defer gpa.free(vi);
            @memcpy(in_w[i * codebook_dim * latent ..][0..vi.len], vi);
            const bi = try l.read(try l.name("{s}.in_proj.bias", .{prefix}), &[_]u64{codebook_dim});
            defer gpa.free(bi);
            @memcpy(in_b[i * codebook_dim ..][0..codebook_dim], bi);
            const vo = try l.folded(try l.name("{s}.out_proj.weight_g", .{prefix}), try l.name("{s}.out_proj.weight_v", .{prefix}), &[_]u64{ latent, codebook_dim, 1 });
            defer gpa.free(vo);
            @memcpy(out_w[i * latent * codebook_dim ..][0..vo.len], vo);
            const bo = try l.read(try l.name("{s}.out_proj.bias", .{prefix}), &[_]u64{latent});
            defer gpa.free(bo);
            @memcpy(out_b[i * latent ..][0..latent], bo);
            const book = try l.read(try l.name("{s}.codebook.weight", .{prefix}), &[_]u64{ size, codebook_dim });
            defer gpa.free(book);
            @memcpy(raw[row * codebook_dim ..][0..book.len], book);
            for (0..size) |r| {
                const src = book[r * codebook_dim ..][0..codebook_dim];
                sq[row + r] = normalizeRow(src, padded[(row + r) * book_width ..][0..codebook_dim]);
            }
            row += size;
        }
        return .{
            .in_w = try l.stage.bfloats(in_w),
            .in_b = try l.stage.bfloats(in_b),
            .books = try l.stage.bfloats(padded),
            .books_sq = try l.stage.floats(sq),
            .books_raw = try l.stage.floats(raw),
            .out_w = try l.stage.bfloats(out_w),
            .out_b = try l.stage.bfloats(out_b),
        };
    }

    /// The encoder, downsampler, pre_module and quantizer weights; RoPE for `rows` positions of either transformer.
    pub fn weights(e: Loader, rows: usize) !Weights {
        const l = e.l;
        var w: Weights = undefined;
        {
            // one input channel: [64][1][7] is already the first conv kernel's [64][7]
            const v = try l.folded("encoder.block.0.conv.parametrizations.weight.original0", "encoder.block.0.conv.parametrizations.weight.original1", &[_]u64{ in_channels, 1, 7 });
            defer l.gpa.free(v);
            w.conv_in = try l.stage.bfloats(v);
        }
        w.conv_in_b = try e.bfloat("encoder.block.0.conv.bias", &[_]u64{in_channels});
        var ch: usize = in_channels;
        for (&w.blocks, strides, 0..) |*b, s, i| {
            const p = try l.name("encoder.block.{d}.block", .{i + 1});
            for (&b.units, 0..) |*u, j| {
                const q = try l.name("{s}.{d}.block", .{ p, j });
                u.snake1 = try l.snake(try l.name("{s}.0.alpha", .{q}), ch);
                u.conv7 = try e.conv(try l.name("{s}.1", .{q}), ch, ch, 7);
                u.conv7_b = try e.bfloat(try l.name("{s}.1.conv.bias", .{q}), &[_]u64{ch});
                u.snake2 = try l.snake(try l.name("{s}.2.alpha", .{q}), ch);
                u.conv1 = try e.conv(try l.name("{s}.3", .{q}), ch, ch, 1);
                u.conv1_b = try e.bfloat(try l.name("{s}.3.conv.bias", .{q}), &[_]u64{ch});
            }
            b.snake = try l.snake(try l.name("{s}.3.alpha", .{p}), ch);
            b.conv = try e.conv(try l.name("{s}.4", .{p}), 2 * ch, ch, 2 * s);
            b.conv_b = try e.bfloat(try l.name("{s}.4.conv.bias", .{p}), &[_]u64{2 * ch});
            ch *= 2;
        }
        const last = try l.name("encoder.block.{d}.block.5", .{strides.len});
        for (&w.layers, 0..) |*layer_w, i| layer_w.* = try e.layer(try l.name("{s}.layers.{d}", .{ last, i }));
        w.norm = try e.float(try l.name("{s}.norm.weight", .{last}), &[_]u64{latent});
        w.out_snake = try l.snake("encoder.block.5.alpha", latent);
        w.out_conv = try e.conv("encoder.block.6", latent, latent, 3);
        w.out_conv_b = try e.bfloat("encoder.block.6.conv.bias", &[_]u64{latent});
        for (&w.downsample, 0..) |*d, i| d.* = try e.downsample(try l.name("quantizer.downsample.{d}", .{i}));
        for (&w.pre, 0..) |*layer_w, i| layer_w.* = try e.layer(try l.name("quantizer.pre_module.layers.{d}", .{i}));
        w.pre_norm = try e.float("quantizer.pre_module.norm.weight", &[_]u64{latent});
        w.rvq = try e.rvq();
        const rope = try load.ropeTable(l.gpa, rows);
        defer l.gpa.free(rope);
        w.rope = try l.stage.bfloats(rope);
        return w;
    }
};

test "blocked taps walked as the encode GEMM walks K reproduce a causal Conv1d of kernel 2s and stride s" {
    // cin 64, stride 2, kernel 4 (left padding 2), small integers so every sum is exact in any order
    const cin = 64;
    const s = 2;
    const k = 4;
    const rows = 6;
    var w: [cin * k]f32 = undefined; // [cout 1][cin][k]
    for (&w, 0..) |*v, i| v.* = @floatFromInt(@as(isize, @intCast(i * 7 % 11)) - 5);
    var x: [rows * s][cin]f32 = undefined;
    for (&x, 0..) |*row, r| for (row, 0..) |*v, c| {
        v.* = @floatFromInt(@as(isize, @intCast((r * 13 + c * 5) % 9)) - 4);
    };
    const taps = try blockedTaps(std.testing.allocator, &w, 1, cin, k);
    defer std.testing.allocator.free(taps);
    for (0..rows) |m| {
        var want: f32 = 0;
        for (0..k) |kk| {
            const pos = @as(isize, @intCast(m * s + kk)) - (k - s);
            if (pos < 0) continue;
            for (0..cin) |ci| want += x[@intCast(pos)][ci] * w[ci * k + kk];
        }
        // drama_codec.cuh conv_gemm, Blocked: K index -> (block, tap, channel) and input row m * s + tap - k + s
        var got: f32 = 0;
        for (0..cin * k) |kk| {
            const block = kk / (channel_block * k);
            const within = kk % (channel_block * k);
            const tap = within / channel_block;
            const rin = @as(isize, @intCast(m * s + tap + s - 1)) - (k - 1);
            if (rin < 0) continue;
            got += x[@intCast(rin)][block * channel_block + within % channel_block] * taps[kk];
        }
        try std.testing.expectEqual(want, got);
    }
}

test "codebook rows normalise with torch's CUDA reduction order" {
    // torch 2.13 CUDA F.normalize and .pow(2).sum(1) of semantic book rows 9, 25, 26, where a sequential sum differs
    const raw = [_][codebook_dim]u32{
        .{ 0x3f66e2f4, 0x3d63d998, 0x406eb7cc, 0x40574f0e, 0x4087b117, 0x401ec536, 0xc0212e95, 0xc01d3320 },
        .{ 0x4092dc61, 0x407dc69c, 0xc052320b, 0x4085ed03, 0x40474bed, 0x402ff348, 0x400c7529, 0x408cb892 },
        .{ 0x40b626e7, 0xc0660219, 0x4067621c, 0x40a6598c, 0x3f3f39c4, 0x3fddf642, 0x40903e06, 0xbf0dc9c7 },
    };
    const normalized = [_][codebook_dim]u32{
        .{ 0x3de9885d, 0x3be67618, 0x3ef1742f, 0x3ed9c6c3, 0x3f093f34, 0x3ea09709, 0xbea3077c, 0xbe9f0058 },
        .{ 0x3ee3b218, 0x3ec4bab1, 0xbea2f206, 0x3ecfa417, 0x3e9a7f19, 0x3e8865f3, 0x3e59c49e, 0x3eda2d22 },
        .{ 0x3f0b3127, 0xbeafc2f2, 0x3eb0cfef, 0x3efe3bc4, 0x3d92202d, 0x3e299ce9, 0x3edc7247, 0xbd58b220 },
    };
    const squares = [_]u32{ 0x3f800000, 0x3f7fffff, 0x3f7ffffe };
    for (raw, normalized, squares) |r, n, sq| {
        var out: [codebook_dim]f32 = undefined;
        const got = normalizeRow(@ptrCast(&r), &out);
        try std.testing.expectEqualSlices(u32, &n, @ptrCast(&out));
        try std.testing.expectEqual(sq, @as(u32, @bitCast(got)));
    }
}
