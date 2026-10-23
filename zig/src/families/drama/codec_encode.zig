//! Drama 3's codec encoder on CUDA: 44.1 kHz clip samples to ten codes a 2048-sample frame, as DAC.encode in BF16.

const std = @import("std");
const cuda = @import("cuda");
const pth = @import("codec_pth.zig");
const load = @import("codec_load.zig");
const enc = @import("codec_encode_load.zig");
const kern = @import("codec_encode_kernels.zig");
pub const clip = @import("clip.zig");

const Allocator = std.mem.Allocator;

pub const samples_per_frame = enc.samples_per_frame;
pub const codebooks = enc.codebooks;
/// The frames of the longest clip the hook accepts.
pub const max_frames = (clip.max_samples + samples_per_frame - 1) / samples_per_frame;

const latent = enc.latent;
const ffn = enc.ffn;
const heads = load.heads;
const head_dim = load.head_dim;
const rms_eps = load.rms_eps;
const layer_norm_eps = load.layer_norm_eps;
/// The widest activation of the encode, in BF16 elements per input sample: the first block's 64 channels.
const halves_per_sample = enc.in_channels;

/// TorchScript runs Snake op by op on a new shape (two static, then dynamic) or dtype, fused after: in a fresh process.
fn fusedSnake(block: usize, unit: usize) bool {
    return !(unit == 0 and block < 3);
}

pub const Model = struct {
    ctx: *cuda.Context,
    kernels: kern.Kernels,
    weights: cuda.DeviceBuffer,
    w: enc.Weights,

    pub fn init(gpa: Allocator, io: std.Io, ctx: *cuda.Context, checkpoint_dir: []const u8) !Model {
        try ctx.makeCurrent();
        var kernels = try kern.Kernels.load(ctx.d);
        errdefer kernels.unload();
        const path = try std.fs.path.join(gpa, &.{ checkpoint_dir, "codec.pth" });
        defer gpa.free(path);
        var archive = try pth.Archive.open(gpa, io, path);
        defer archive.close();
        var names = std.heap.ArenaAllocator.init(gpa);
        defer names.deinit();
        var stage: load.Stage = .{ .gpa = gpa };
        defer stage.bytes.deinit(gpa);
        var loader: load.Loader = .{ .gpa = gpa, .names = names.allocator(), .archive = &archive, .stage = &stage };
        const w = try (enc.Loader{ .l = &loader }).weights(max_frames << enc.downsample_stages);
        return .{ .ctx = ctx, .kernels = kernels, .weights = try cuda.DeviceBuffer.fromHost(ctx.d, stage.bytes.items), .w = w };
    }

    pub fn deinit(self: *Model) void {
        self.ctx.makeCurrent() catch {};
        self.weights.free();
        self.kernels.unload();
        self.* = undefined;
    }

    /// Codes [frames][10] (caller frees) of clamped 44.1 kHz `samples`; frames = ceil(len / 2048) <= max_frames.
    pub fn encode(self: *Model, gpa: Allocator, stream: cuda.Stream, samples: []const f32) ![][codebooks]u32 {
        if (samples.len == 0) return error.NoSamples;
        const frames = (samples.len + samples_per_frame - 1) / samples_per_frame;
        if (frames > max_frames) return error.TooManyFrames;
        const rows = frames * samples_per_frame;
        try self.ctx.makeCurrent();
        const d = self.ctx.d;
        var audio = try cuda.DeviceBuffer.alloc(d, rows * @sizeOf(f32));
        defer audio.free();
        try audio.fill8(0, stream.handle);
        try audio.uploadAsync(0, std.mem.sliceAsBytes(samples), stream.handle);
        var acts: [3]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        defer for (acts[0..made]) |*b| b.free();
        for (&acts) |*b| {
            b.* = try cuda.DeviceBuffer.alloc(d, rows * halves_per_sample * 2);
            made += 1;
        }
        var codes_dev = try cuda.DeviceBuffer.alloc(d, frames * codebooks * @sizeOf(u32));
        defer codes_dev.free();
        try self.forward(stream, .{ acts[0].ptr, acts[1].ptr, acts[2].ptr }, audio.ptr, codes_dev.ptr, rows);
        const codes = try gpa.alloc([codebooks]u32, frames);
        errdefer gpa.free(codes);
        try codes_dev.downloadAsync(0, std.mem.sliceAsBytes(codes), stream.handle);
        try stream.synchronize();
        return codes;
    }

    fn wp(self: *const Model, offset: u64) u64 {
        return self.weights.ptr + offset;
    }

    fn forward(self: *Model, s: cuda.Stream, p: [3]u64, audio: u64, codes: u64, samples: usize) !void {
        const k = &self.kernels;
        const w = &self.w;
        const h = 2; // bytes a BF16 element

        // encoder.block[0], then the EncoderBlocks: x is the block's input, spare its two scratch buffers
        var rows = samples;
        try k.convIn(s, audio, self.wp(w.conv_in), self.wp(w.conv_in_b), p[0], rows);
        var x = p[0];
        var spare = [2]u64{ p[1], p[2] };
        var ch: usize = enc.in_channels;
        for (&w.blocks, enc.strides, 0..) |*b, stride, i| {
            // ResidualUnits: x + conv1(snake(conv7(snake(x))))
            for (&b.units, enc.dilations, 0..) |*u, dilation, j| {
                const sb = spare[0];
                const tb = spare[1];
                try k.snake(s, x, false, self.wp(u.snake1.alpha), self.wp(u.snake1.inv), sb, rows, ch, fusedSnake(i, j));
                try k.gemm(s, .{ .x = sb, .w = self.wp(u.conv7), .y = tb, .m = rows, .n = ch, .cin = ch, .ldx = ch, .ldy = ch, .taps = 7, .dilation = dilation, .bias = self.wp(u.conv7_b), .bias_mode = .separate });
                try k.snake(s, tb, false, self.wp(u.snake2.alpha), self.wp(u.snake2.inv), tb, rows, ch, true);
                try k.gemm(s, .{ .x = tb, .w = self.wp(u.conv1), .y = sb, .m = rows, .n = ch, .cin = ch, .ldx = ch, .ldy = ch, .bias = self.wp(u.conv1_b), .bias_mode = .separate, .residual = x, .ldr = ch });
                spare = .{ x, tb };
                x = sb;
            }
            // Snake, then the causal conv of kernel 2 * stride and that stride
            try k.snake(s, x, false, self.wp(b.snake.alpha), self.wp(b.snake.inv), spare[0], rows, ch, true);
            rows /= stride;
            try k.gemm(s, .{ .x = spare[0], .w = self.wp(b.conv), .y = spare[1], .m = rows, .n = 2 * ch, .cin = ch, .ldx = ch, .ldy = 2 * ch, .taps = 2 * stride, .stride = stride, .bias = self.wp(b.conv_b), .bias_mode = .separate });
            ch *= 2;
            const out = spare[1];
            spare[1] = x;
            x = out;
        }
        std.debug.assert(ch == latent);

        // the last block's transformer (BF16 residual stream), its FP32 norm, Snake, conv 3 -> the latent in p
        try self.transformer(s, x, false, &w.layers, enc.encoder_window, rows, spare[0]);
        const normed = spare[1];
        try k.rmsNorm(s, x, false, self.wp(w.norm), normed, true, rows, latent, rms_eps);
        try k.snake(s, normed, true, self.wp(w.out_snake.alpha), self.wp(w.out_snake.inv), spare[0], rows, latent, false);
        var lat = x;
        try k.gemm(s, .{ .x = spare[0], .w = self.wp(w.out_conv), .y = lat, .m = rows, .n = latent, .cin = latent, .ldx = latent, .ldy = latent, .taps = 3, .bias = self.wp(w.out_conv_b), .bias_mode = .separate });

        // quantizer.downsample: (causal conv k2 s2, ConvNeXtBlock) twice; the last keeps FP32 for pre_module
        var free = [2]u64{ spare[0], spare[1] };
        for (&w.downsample, 0..) |*u, i| {
            rows /= 2;
            const y = free[0];
            try k.gemm(s, .{ .x = lat, .w = self.wp(u.conv), .y = y, .m = rows, .n = latent, .cin = latent, .ldx = latent, .ldy = latent, .taps = 2, .stride = 2, .bias = self.wp(u.conv_b), .bias_mode = .separate });
            const dw = free[1];
            const normed_rows = free[1] + rows * latent * h;
            const hidden = lat;
            try k.dwconv(s, y, self.wp(u.dw), self.wp(u.dw_b), dw, rows, latent);
            try k.layerNorm(s, dw, self.wp(u.norm_w), self.wp(u.norm_b), normed_rows, rows, latent, layer_norm_eps);
            try k.gemm(s, .{ .x = normed_rows, .w = self.wp(u.pw1), .y = hidden, .m = rows, .n = enc.convnext_hidden, .cin = latent, .ldx = latent, .ldy = enc.convnext_hidden, .bias = self.wp(u.pw1_b), .bias_mode = .fused, .act = .gelu });
            // y + gamma * pwconv2(...): FP32, which the next conv reads as BF16
            const last = i + 1 == w.downsample.len;
            const out = free[1];
            try k.gemm(s, .{ .x = hidden, .w = self.wp(u.pw2), .y = out, .m = rows, .n = latent, .cin = enc.convnext_hidden, .ldx = enc.convnext_hidden, .ldy = latent, .bias = self.wp(u.pw2_b), .bias_mode = .fused, .scale = self.wp(u.gamma), .residual = y, .ldr = latent, .out_f32 = last });
            free = .{ lat, y };
            lat = out;
        }

        // pre_module over the FP32 residual stream, its final norm, then the RVQ of that FP32 latent
        try self.transformer(s, lat, true, &w.pre, enc.pre_window, rows, free[0]);
        const z = free[1];
        try k.rmsNorm(s, lat, true, self.wp(w.pre_norm), z, true, rows, latent, rms_eps);
        const r = &w.rvq;
        try k.rvqCodes(s, z, codes, self.wp(r.in_w), self.wp(r.in_b), self.wp(r.books), self.wp(r.books_sq), self.wp(r.books_raw), self.wp(r.out_w), self.wp(r.out_b), rows);
    }

    /// WindowLimitedTransformer's layers in place on x [rows][1024], BF16 or FP32; `scratch` holds 28 KiB a row.
    fn transformer(self: *Model, s: cuda.Stream, x: u64, x_f32: bool, layers: []const enc.Layer, window: usize, rows: usize, scratch: u64) !void {
        const k = &self.kernels;
        const h = 2;
        const xn = scratch;
        const qkv = xn + rows * latent * h;
        const att = qkv + rows * 3 * latent * h;
        const gu = att + rows * latent * h;
        const act = gu + rows * 2 * ffn * h;
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        for (layers) |*l| {
            // h = x + attention_layer_scale(wo(attention(attention_norm(x)))), the scale rounded as mul_ on BF16
            try k.rmsNorm(s, x, x_f32, self.wp(l.attention_norm), xn, false, rows, latent, rms_eps);
            try k.gemm(s, .{ .x = xn, .w = self.wp(l.wqkv), .y = qkv, .m = rows, .n = 3 * latent, .cin = latent, .ldx = latent, .ldy = 3 * latent });
            try k.ropeQk(s, qkv, self.wp(self.w.rope), rows, latent);
            try k.windowAttention(s, qkv, att, rows, heads, window, scale);
            try k.gemm(s, .{ .x = att, .w = self.wp(l.wo), .y = x, .m = rows, .n = latent, .cin = latent, .ldx = latent, .ldy = latent, .scale = self.wp(l.attention_scale), .round_scaled = true, .residual = x, .ldr = latent, .residual_f32 = x_f32, .out_f32 = x_f32 });
            try k.rmsNorm(s, x, x_f32, self.wp(l.ffn_norm), xn, false, rows, latent, rms_eps);
            try k.gemm(s, .{ .x = xn, .w = self.wp(l.w13), .y = gu, .m = rows, .n = 2 * ffn, .cin = latent, .ldx = latent, .ldy = 2 * ffn });
            try k.siluMul(s, gu, act, rows, ffn);
            try k.gemm(s, .{ .x = act, .w = self.wp(l.w2), .y = x, .m = rows, .n = latent, .cin = ffn, .ldx = ffn, .ldy = latent, .scale = self.wp(l.ffn_scale), .round_scaled = true, .residual = x, .ldr = latent, .residual_f32 = x_f32, .out_f32 = x_f32 });
        }
    }
};

comptime {
    std.debug.assert(max_frames == 1292);
    // the transformer's scratch, 28 KiB a row at one row per 512 samples, fits an activation buffer
    std.debug.assert(14 * latent * 2 <= 512 * halves_per_sample * 2);
}
