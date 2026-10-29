//! LFM3-VL's pictures on CUDA: the image processor on the CPU, then the vision tower and projector (lfm3v.cu).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lfm2 = @import("lfm2");
const Allocator = std.mem.Allocator;
const vit = cuda.vit;
const resample = core.picture.resample;

pub const Rgb = core.picture.Rgb;

// config.json's vision_config and projector (lfm3/config.zig admits exactly these)
const hidden = 1024;
const ffn = 2560;
const heads = 16;
const kv_heads = 8;
const dim = 64;
const layers = 14;
const patch = 16;
const merge = 2;
const temporal = 2;
const patch_width = 3 * temporal * patch * patch; // 1536
const text_hidden = 2048;
const proj_hidden = 4096;
const eps: f32 = 1e-5;
const proj_eps: f32 = 1e-6;
const theta: f64 = 1e6;
const attention_at = [layers]bool{ false, false, true, false, true, false, true, false, true, false, true, false, true, false };
// processor_config.json: size.shortest_edge, size.longest_edge; d1's d1_vision_max_pixels before it
const min_pixels = 65536;
const max_pixels = 16777216;
const cap_pixels = 1638400;
/// The largest picture a pass takes: 6,400 patches (1,600 tokens), the cap's 1280 x 1280.
pub const patches_most = 6400;

/// The ids around a picture's tokens (config.json).
pub const vision_start: u32 = 125009;
pub const vision_end: u32 = 125010;
pub const image_token: u32 = 125030;

/// A picture ready for the tower: bf16 patch rows in merge order, its grid, and its RoPE tables.
pub const Picture = struct {
    gh: usize,
    gw: usize,
    px: []u16, // bf16 [gh gw, 1536]
    cos: []f32, // [gh gw, 64]
    sin: []f32,

    pub fn rows(p: *const Picture) usize {
        return p.gh * p.gw;
    }

    /// The image tokens the projector makes: one a 2 x 2 block.
    pub fn tokens(p: *const Picture) usize {
        return p.rows() / (merge * merge);
    }

    /// <|vision_start|>, one <|image|> a token, <|vision_end|>.
    pub fn appendIds(p: *const Picture, a: Allocator, ids: *std.ArrayList(u32)) Allocator.Error!void {
        try ids.append(a, vision_start);
        try ids.appendNTimes(a, image_token, p.tokens());
        try ids.append(a, vision_end);
    }
};

/// round(x / factor) * factor, Python's round: halves go to even.
fn roundBy(x: usize, factor: usize) usize {
    const q = @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(factor));
    const r = if (q - @floor(q) == 0.5) 2 * @round(q / 2) else @round(q);
    return @as(usize, @intFromFloat(r)) * factor;
}

/// smart_resize(height, width, factor=32, min_pixels, max_pixels) as (width, height).
pub fn smartSize(w: usize, h: usize) [2]usize {
    const f = patch * merge;
    const fw: f64, const fh: f64 = .{ @floatFromInt(w), @floatFromInt(h) };
    var hb = roundBy(h, f);
    var wb = roundBy(w, f);
    if (hb * wb > max_pixels) {
        const beta = @sqrt(fh * fw / max_pixels);
        hb = @max(f, @as(usize, @intFromFloat(@floor(fh / beta / f))) * f);
        wb = @max(f, @as(usize, @intFromFloat(@floor(fw / beta / f))) * f);
    } else if (hb * wb < min_pixels) {
        const beta = @sqrt(@as(f64, min_pixels) / (fh * fw));
        hb = @as(usize, @intFromFloat(@ceil(fh * beta / f))) * f;
        wb = @as(usize, @intFromFloat(@ceil(fw * beta / f))) * f;
    }
    return .{ wb, hb };
}

/// bf16 of a finite f32, round to nearest even.
fn bf16(x: f32) u16 {
    return vit.bf16(x);
}

/// The d1 cap (Pillow's bicubic), smart_resize (torch's uint8 bicubic), patches and RoPE tables.
pub fn prepare(a: Allocator, rgb: Rgb) !Picture {
    return prepareUnder(a, rgb, cap_pixels);
}

/// `prepare` under a cap of `cap` pixels (d1's), or none (LFM3-VL's processor: smart_resize alone).
pub fn prepareUnder(a: Allocator, rgb: Rgb, cap: ?usize) !Picture {
    var src = rgb;
    var capped: ?[]u8 = null;
    defer if (capped) |c| a.free(c);
    if (cap != null and rgb.width * rgb.height > cap.?) {
        const s = @sqrt(@as(f64, @floatFromInt(cap.?)) / @as(f64, @floatFromInt(rgb.width * rgb.height)));
        const w = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(rgb.width)) * s)));
        const h = @max(1, @as(usize, @intFromFloat(@as(f64, @floatFromInt(rgb.height)) * s)));
        capped = try resample.resize(a, .pillow, .of(rgb.pixels, rgb.width, rgb.height), w, h);
        src = .{ .width = w, .height = h, .pixels = capped.? };
    }
    const sw, const sh = smartSize(src.width, src.height);
    const gh = sh / patch;
    const gw = sw / patch;
    if (gh * gw > patches_most) return error.PictureTooLarge;
    const same = sw == src.width and sh == src.height;
    const pixels = if (same) src.pixels else try resample.resize(a, .torch, .of(src.pixels, src.width, src.height), sw, sh);
    defer if (!same) a.free(pixels);
    const n = gh * gw;
    const px = try a.alloc(u16, n * patch_width);
    errdefer a.free(px);
    // (x / 255 - 0.5) / 0.5 in f32, as the processor's fused rescale and normalize, then the patchifier's bf16
    var table: [256]u16 = undefined;
    for (&table, 0..) |*v, i| v.* = bf16((@as(f32, @floatFromInt(i)) * @as(f32, 0.00392156862745098) - 0.5) / 0.5);
    for (0..n) |i| {
        const py, const pxl = spot(i, gw);
        // the row's (C, T, 16, 16) values: each channel's patch twice
        const row = px[i * patch_width ..][0..patch_width];
        for (0..3) |ch| for (0..temporal) |t| for (0..patch) |r| for (0..patch) |col| {
            const v = pixels[((py * patch + r) * sw + pxl * patch + col) * 3 + ch];
            row[((ch * temporal + t) * patch + r) * patch + col] = table[v];
        };
    }
    return withRope(a, px, gh, gw);
}

/// Merge-order row `i`'s patch (row, column) on a grid `gw` patches wide.
fn spot(i: usize, gw: usize) [2]usize {
    const block = i / 4;
    const inner = i % 4;
    return .{ (block / (gw / 2)) * 2 + inner / 2, (block % (gw / 2)) * 2 + inner % 2 };
}

/// A picture of patch rows `px` (kept on success) on a gh x gw grid, with its axial RoPE tables: (h, w freqs) twice.
fn withRope(a: Allocator, px: []u16, gh: usize, gw: usize) !Picture {
    const n = gh * gw;
    const cs = try a.alloc(f32, n * dim);
    errdefer a.free(cs);
    const sn = try a.alloc(f32, n * dim);
    errdefer a.free(sn);
    var inv: [dim / 4]f32 = undefined;
    for (&inv, 0..) |*v, j| {
        const e: f32 = @as(f32, @floatFromInt(2 * j)) / @as(f32, dim / 2);
        v.* = 1.0 / std.math.pow(f32, @floatCast(theta), e);
    }
    for (0..n) |i| {
        const py, const pxl = spot(i, gw);
        const c = cs[i * dim ..][0..dim];
        const s = sn[i * dim ..][0..dim];
        for (0..dim / 4) |j| {
            const fh = @as(f32, @floatFromInt(py)) * inv[j];
            const fw = @as(f32, @floatFromInt(pxl)) * inv[j];
            for ([_]usize{ 0, dim / 2 }) |base| {
                c[base + j] = @cos(fh);
                s[base + j] = @sin(fh);
                c[base + dim / 4 + j] = @cos(fw);
                s[base + dim / 4 + j] = @sin(fw);
            }
        }
    }
    return .{ .gh = gh, .gw = gw, .px = px, .cos = cs, .sin = sn };
}

/// A prepared picture's patch rows as f32 (exact: they are bf16), the form a server's image request carries.
pub fn widen(a: Allocator, p: *const Picture) ![]f32 {
    const out = try a.alloc(f32, p.px.len);
    for (out, p.px) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
    return out;
}

/// The picture a server's image request carries (`widen`'s rows on a gh x gw grid), its RoPE tables made again.
pub fn fromRows(a: Allocator, rows: []const f32, gh: usize, gw: usize) !Picture {
    if (gh % merge != 0 or gw % merge != 0 or gh * gw > patches_most or rows.len != gh * gw * patch_width) return error.BadPictureRows;
    const px = try a.alloc(u16, rows.len);
    errdefer a.free(px);
    for (px, rows) |*b, v| b.* = bf16(v);
    return withRope(a, px, gh, gw);
}

test "a server's rows give back the prepared picture, and RoPE tables the same" {
    const a = std.testing.allocator;
    const w = 64;
    const h = 96;
    const pixels = try a.alloc(u8, w * h * 3);
    defer a.free(pixels);
    for (pixels, 0..) |*v, i| v.* = @intCast((i * 37) % 251);
    var p = try prepareUnder(a, .{ .width = w, .height = h, .pixels = pixels }, null);
    defer free(a, &p);
    const rows = try widen(a, &p);
    defer a.free(rows);
    var q = try fromRows(a, rows, p.gh, p.gw);
    defer free(a, &q);
    try std.testing.expectEqualSlices(u16, p.px, q.px);
    try std.testing.expectEqualSlices(f32, p.cos, q.cos);
    try std.testing.expectEqualSlices(f32, p.sin, q.sin);
    try std.testing.expectError(error.BadPictureRows, fromRows(a, rows, p.gh + 1, p.gw));
}

pub fn free(a: Allocator, p: *Picture) void {
    a.free(p.px);
    a.free(p.cos);
    a.free(p.sin);
}

const Layer = struct {
    operator_norm: u64,
    ffn_norm: u64,
    gate_up: u64, // [2 ffn, hidden] (w1 | w3)
    down: u64, // [hidden, ffn]
    mix_in: u64, // conv in_proj [3 hidden, hidden], or q | k | v [3 hidden, hidden] with each kv head twice
    mix_out: u64, // conv out_proj, or the attention's out_proj [hidden, hidden]
    conv: u64 = 0, // [hidden, 9]
    qn: u64 = 0, // [64]
    kn: u64 = 0,
};

pub const Weights = struct {
    patchifier: u64,
    embedding_norm: u64,
    layers: [layers]Layer,
    ln_w: u64,
    ln_b: u64,
    l1_w: u64,
    l1_b: u64,
    l2_w: u64,
    l2_b: u64,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights, gpa: Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
    }
};

const Loader = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    ck: *core.checkpoint.Checkpoint,
    w: *Weights,

    fn up(x: *Loader, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(x.d, bytes);
        errdefer {
            var y = b;
            y.free();
        }
        try x.w.buffers.append(x.gpa, b);
        return b.ptr;
    }

    fn tensor(x: *Loader, name: []const u8, shape: []const usize) ![]const u8 {
        return (try x.ck.expect(name, .bf16, shape)).bytes;
    }

    fn one(x: *Loader, name: []const u8, shape: []const usize) !u64 {
        return x.up(try x.tensor(name, shape));
    }

    /// Row blocks of several tensors stacked into one device matrix.
    fn stacked(x: *Loader, parts: []const []const u8) !u64 {
        var n: usize = 0;
        for (parts) |p| n += p.len;
        const host = try x.gpa.alloc(u8, n);
        defer x.gpa.free(host);
        var at: usize = 0;
        for (parts) |p| {
            @memcpy(host[at..][0..p.len], p);
            at += p.len;
        }
        return x.up(host);
    }

    /// k or v [kv_heads 64, hidden] with each head's rows twice: query head h reads kv head h / 2.
    fn repeated(x: *Loader, rows: []const u8) ![]u8 {
        const head_bytes = dim * hidden * 2;
        const out = try x.gpa.alloc(u8, 2 * rows.len);
        for (0..heads) |h| @memcpy(out[h * head_bytes ..][0..head_bytes], rows[(h / 2) * head_bytes ..][0..head_bytes]);
        return out;
    }
};

pub fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var w: Weights = undefined;
    w.buffers = .empty;
    errdefer w.deinit(gpa);
    var x: Loader = .{ .gpa = gpa, .d = d, .ck = &ck, .w = &w };
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    var b3: [128]u8 = undefined;
    w.patchifier = try x.one("model.vision_tower.patchifier.weight", &.{ hidden, patch_width });
    w.embedding_norm = try x.one("model.vision_tower.embedding_norm.weight", &.{hidden});
    for (&w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.vision_tower.layers.{d}.", .{i});
        const n = struct {
            fn f(b: []u8, pre: []const u8, s: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "{s}{s}", .{ pre, s });
            }
        }.f;
        l.* = .{ .operator_norm = 0, .ffn_norm = 0, .gate_up = 0, .down = 0, .mix_in = 0, .mix_out = 0 };
        l.operator_norm = try x.one(try n(&b1, p, "operator_norm.weight"), &.{hidden});
        l.ffn_norm = try x.one(try n(&b1, p, "ffn_norm.weight"), &.{hidden});
        l.gate_up = try x.stacked(&.{ try x.tensor(try n(&b1, p, "feed_forward.w1.weight"), &.{ ffn, hidden }), try x.tensor(try n(&b2, p, "feed_forward.w3.weight"), &.{ ffn, hidden }) });
        l.down = try x.one(try n(&b1, p, "feed_forward.w2.weight"), &.{ hidden, ffn });
        if (attention_at[i]) {
            const kr = try x.repeated(try x.tensor(try n(&b2, p, "self_attn.k_proj.weight"), &.{ kv_heads * dim, hidden }));
            defer gpa.free(kr);
            const vr = try x.repeated(try x.tensor(try n(&b3, p, "self_attn.v_proj.weight"), &.{ kv_heads * dim, hidden }));
            defer gpa.free(vr);
            l.mix_in = try x.stacked(&.{ try x.tensor(try n(&b1, p, "self_attn.q_proj.weight"), &.{ hidden, hidden }), kr, vr });
            l.mix_out = try x.one(try n(&b1, p, "self_attn.out_proj.weight"), &.{ hidden, hidden });
            l.qn = try x.one(try n(&b1, p, "self_attn.q_layernorm.weight"), &.{dim});
            l.kn = try x.one(try n(&b1, p, "self_attn.k_layernorm.weight"), &.{dim});
        } else {
            l.mix_in = try x.one(try n(&b1, p, "conv.in_proj.weight"), &.{ 3 * hidden, hidden });
            l.mix_out = try x.one(try n(&b1, p, "conv.out_proj.weight"), &.{ hidden, hidden });
            l.conv = try x.one(try n(&b1, p, "conv.spatial_conv.weight"), &.{ hidden, 1, 3, 3 });
        }
    }
    w.ln_w = try x.one("model.multi_modal_projector.layer_norm.weight", &.{hidden});
    w.ln_b = try x.one("model.multi_modal_projector.layer_norm.bias", &.{hidden});
    w.l1_w = try x.one("model.multi_modal_projector.linear_1.weight", &.{ proj_hidden, 4 * hidden });
    w.l1_b = try x.one("model.multi_modal_projector.linear_1.bias", &.{proj_hidden});
    w.l2_w = try x.one("model.multi_modal_projector.linear_2.weight", &.{ text_hidden, proj_hidden });
    w.l2_b = try x.one("model.multi_modal_projector.linear_2.bias", &.{text_hidden});
    return w;
}

const Glue = struct {
    mod: cuda.Module,
    rms: cuda.Function,
    qk: cuda.Function,
    conv: cuda.Function,
    add: cuda.Function,
    silu: cuda.Function,

    fn load(d: *const cuda.Driver) !Glue {
        var mod = try cuda.Module.load(d, cuda.kernels.lfm3v);
        errdefer mod.unload();
        return .{ .mod = mod, .rms = try mod.function("tf_lfm3v_rms"), .qk = try mod.function("tf_lfm3v_qk"), .conv = try mod.function("tf_lfm3v_conv"), .add = try mod.function("tf_lfm3v_add"), .silu = try mod.function("tf_lfm3v_silu_mul") };
    }
};

fn grid(n: usize, by: usize) u32 {
    return @intCast((n + by - 1) / by);
}

pub const Vision = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    w: *const Weights,
    k: vit.Kernels,
    at: vit.Attention,
    g: vit.Gemms,
    glue: Glue,
    scratch: cuda.DeviceBuffer,
    b: Buffers,

    const Buffers = struct { px: u64, cs: u64, sn: u64, h: u64, x: u64, wide: u64, y: u64, a: u64, acc: u64 };

    pub fn init(gpa: Allocator, d: *const cuda.Driver, w: *const Weights) !Vision {
        var k = try vit.Kernels.load(d);
        errdefer k.unload();
        var at = try vit.Attention.load(d);
        errdefer at.unload();
        var g = try vit.Gemms.init(gpa, d);
        errdefer g.deinit();
        var glue = try Glue.load(d);
        errdefer glue.mod.unload();
        const n = patches_most;
        const sizes = [_]usize{
            2 * n * patch_width, // px
            4 * n * dim, // cos
            4 * n * dim, // sin
            2 * n * hidden, // h: the residual
            2 * n * hidden, // x: a norm's output, the projector's merged rows
            2 * n * @max(3 * hidden, 2 * ffn), // wide: in_proj or q|k|v, w1|w3
            2 * n * @max(hidden, ffn), // y: the conv's or attention's output, the SwiGLU
            2 * n * @max(@max(hidden, ffn), proj_hidden / 4), // a: the attention rows, the SwiGLU, the projector's hidden
            4 * n * @max(proj_hidden, text_hidden) / 4, // acc: f32 products where cuBLASLt does not fuse the bias
        };
        var total: usize = 0;
        for (sizes) |sz| total += std.mem.alignForward(usize, sz, 256);
        var scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer scratch.free();
        var ptrs: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &ptrs) |sz, *p| {
            p.* = scratch.ptr + off;
            off += std.mem.alignForward(usize, sz, 256);
        }
        return .{ .gpa = gpa, .d = d, .w = w, .k = k, .at = at, .g = g, .glue = glue, .scratch = scratch, .b = .{ .px = ptrs[0], .cs = ptrs[1], .sn = ptrs[2], .h = ptrs[3], .x = ptrs[4], .wide = ptrs[5], .y = ptrs[6], .a = ptrs[7], .acc = ptrs[8] } };
    }

    pub fn deinit(v: *Vision) void {
        v.scratch.free();
        v.glue.mod.unload();
        v.g.deinit();
        v.at.unload();
        v.k.unload();
    }

    fn rms(v: *Vision, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, w, y }) |p| a.add(p);
        a.add(@as(c_int, hidden));
        a.add(eps);
        try cuda.launch.launch(v.glue.rms, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    fn add(v: *Vision, s: cuda.Stream, x: u64, r: u64, y: u64, count: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, r, y }) |p| a.add(p);
        a.add(@as(c_longlong, @intCast(count)));
        try cuda.launch.launch(v.glue.add, .{ .grid = .{ .x = grid(count, 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The picture's projected rows, bf16 [tokens, 2048] at `out`; returns the token count.
    pub fn features(v: *Vision, s: cuda.Stream, p: *const Picture, out: u64) !usize {
        const n = p.rows();
        const b = v.b;
        const up = struct {
            fn f(d: *const cuda.Driver, st: cuda.Stream, dst: u64, bytes: []const u8) !void {
                try d.check(d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, st.handle), "cuMemcpyHtoDAsync");
            }
        }.f;
        try up(v.d, s, b.px, std.mem.sliceAsBytes(p.px));
        try up(v.d, s, b.cs, std.mem.sliceAsBytes(p.cos));
        try up(v.d, s, b.sn, std.mem.sliceAsBytes(p.sin));
        // patchifier, embedding norm
        try v.g.run(s, b.px, v.w.patchifier, b.y, n, hidden, patch_width, .bf16);
        try v.rms(s, b.y, v.w.embedding_norm, b.h, n);
        for (v.w.layers, attention_at) |l, attn| {
            try v.rms(s, b.h, l.operator_norm, b.x, n);
            try v.g.run(s, b.x, l.mix_in, b.wide, n, 3 * hidden, hidden, .bf16);
            if (attn) {
                var a: cuda.Args = .{};
                for ([_]u64{ b.wide, l.qn, l.kn, b.cs, b.sn }) |ptr| a.add(ptr);
                a.add(@as(c_int, @intCast(n)));
                a.add(@as(c_int, heads));
                a.add(eps);
                try cuda.launch.launch(v.glue.qk, .{ .grid = .{ .x = @intCast(n * heads * 2), .y = 1, .z = 1 }, .block = .{ .x = 32 } }, s, &a);
                try v.at.full(s, b.wide, b.a, n, heads, dim, 0.125);
                try v.g.run(s, b.a, l.mix_out, b.y, n, hidden, hidden, .bf16);
            } else {
                var a: cuda.Args = .{};
                for ([_]u64{ b.wide, l.conv, b.a }) |ptr| a.add(ptr);
                for ([_]usize{ p.gh, p.gw, hidden }) |m| a.add(@as(c_int, @intCast(m)));
                try cuda.launch.launch(v.glue.conv, .{ .grid = .{ .x = grid(n * hidden, 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
                try v.g.run(s, b.a, l.mix_out, b.y, n, hidden, hidden, .bf16);
            }
            try v.add(s, b.h, b.y, b.h, n * hidden);
            try v.rms(s, b.h, l.ffn_norm, b.x, n);
            try v.g.run(s, b.x, l.gate_up, b.wide, n, 2 * ffn, hidden, .bf16);
            var a: cuda.Args = .{};
            for ([_]u64{ b.wide, b.a }) |ptr| a.add(ptr);
            a.add(@as(c_int, ffn));
            a.add(@as(c_longlong, @intCast(n * ffn)));
            try cuda.launch.launch(v.glue.silu, .{ .grid = .{ .x = grid(n * ffn, 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
            try v.g.run(s, b.a, l.down, b.y, n, hidden, ffn, .bf16);
            try v.add(s, b.h, b.y, b.h, n * hidden);
        }
        // projector: LayerNorm a patch, 2 x 2 blocks side by side (merge order keeps them consecutive), GELU MLP
        const t = n / 4;
        try v.k.layerNorm(s, b.h, v.w.ln_w, v.w.ln_b, b.x, n, hidden, proj_eps);
        try vit.linearAct(&v.g, &v.k, s, b.acc, b.x, v.w.l1_w, v.w.l1_b, b.a, t, proj_hidden, 4 * hidden, null, b.a, .gelu);
        try vit.linear(&v.g, &v.k, s, b.acc, b.a, v.w.l2_w, v.w.l2_b, out, t, text_hidden, proj_hidden);
        return t;
    }
};
