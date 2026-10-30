//! LFM3-Omni's vision tower and projector on Metal: the LFM3-VL tower (14 layers, gated 3 x 3 convolutions and
//! bidirectional attention with axial RoPE over each picture's own patches, SwiGLU), the projector's per-patch
//! LayerNorm, the 2 x 2 merge, two linears with GELU between. The tower's own operators (omni_vision.metal, from the
//! LFM3-VL Metal port) and the family's products; one or more pictures a pass, their rows back to back.
const std = @import("std");
const st = @import("core").safetensors;
const gpu = @import("gpu.zig");
const kern = @import("kernels.zig");
const stack = @import("stack.zig");

const Arena = stack.Arena;
const width = 1024;
const ffn = 2560;
const depth = 14;
const kv_width = 512;
const patch_width = 1536;
const proj_hidden = 4096;
const text_hidden = 2048;

/// Each layer's mixer: conv everywhere but the even layers from 2 on (config.json's layer_types).
pub fn attentionAt(i: usize) bool {
    return i >= 2 and i % 2 == 0;
}

const Layer = struct {
    op_norm: u64,
    ffn_norm: u64,
    w1: u64,
    w3: u64,
    w2: u64,
    // conv
    in_proj: u64 = 0,
    out_proj: u64 = 0,
    spatial: u64 = 0,
    // attention
    q: u64 = 0,
    k: u64 = 0,
    v: u64 = 0,
    o: u64 = 0,
    q_norm: u64 = 0,
    k_norm: u64 = 0,
};

/// A picture's patches for one pass: bf16 rows [gh * gw, 1536] are written by `run` from these fp32 values.
pub const Picture = struct { pixels: []const f32, gh: u32, gw: u32 };

pub const Vision = struct {
    g: *gpu.Gpu,
    max_patches: usize,
    patch: u64,
    norm: u64,
    layers: [depth]Layer,
    ln_w: u64,
    ln_b: u64,
    l1_w: u64,
    l1_b: u64,
    l2_w: u64,
    l2_b: u64,
    // scratch
    px: u64,
    h: u64,
    x: u64,
    r: u64,
    y: u64,
    q: u64,
    k: u64,
    v: u64,
    bcx: u64,
    gate: u64,
    up: u64,
    act: u64,
    acc: u64,
    hid: u64,
    host_px: []u16,

    /// The tower's 138 tensors from `f` (opened with the `model.vision_tower.` and `model.multi_modal_projector.`
    /// prefixes) and scratch for `max_tokens` language tokens a pass (4 patches each).
    pub fn load(gpa: std.mem.Allocator, a: *Arena, f: *const st.File, max_tokens: usize) !Vision {
        var v: Vision = undefined;
        v.g = a.g;
        v.max_patches = max_tokens * 4;
        const t = "model.vision_tower.";
        v.patch = try load1(a, f, t ++ "patchifier.weight", &.{ width, patch_width });
        v.norm = try load1(a, f, t ++ "embedding_norm.weight", &.{width});
        for (&v.layers, 0..) |*l, i| {
            l.* = .{
                .op_norm = try lt(a, f, i, "operator_norm.weight", &.{width}),
                .ffn_norm = try lt(a, f, i, "ffn_norm.weight", &.{width}),
                .w1 = try lt(a, f, i, "feed_forward.w1.weight", &.{ ffn, width }),
                .w3 = try lt(a, f, i, "feed_forward.w3.weight", &.{ ffn, width }),
                .w2 = try lt(a, f, i, "feed_forward.w2.weight", &.{ width, ffn }),
            };
            if (attentionAt(i)) {
                l.q = try lt(a, f, i, "self_attn.q_proj.weight", &.{ width, width });
                l.k = try lt(a, f, i, "self_attn.k_proj.weight", &.{ kv_width, width });
                l.v = try lt(a, f, i, "self_attn.v_proj.weight", &.{ kv_width, width });
                l.o = try lt(a, f, i, "self_attn.out_proj.weight", &.{ width, width });
                l.q_norm = try lt(a, f, i, "self_attn.q_layernorm.weight", &.{64});
                l.k_norm = try lt(a, f, i, "self_attn.k_layernorm.weight", &.{64});
            } else {
                l.in_proj = try lt(a, f, i, "conv.in_proj.weight", &.{ 3 * width, width });
                l.out_proj = try lt(a, f, i, "conv.out_proj.weight", &.{ width, width });
                l.spatial = try lt(a, f, i, "conv.spatial_conv.weight", &.{ width, 1, 3, 3 });
            }
        }
        const pj = "model.multi_modal_projector.";
        v.ln_w = try load1(a, f, pj ++ "layer_norm.weight", &.{width});
        v.ln_b = try load1(a, f, pj ++ "layer_norm.bias", &.{width});
        v.l1_w = try load1(a, f, pj ++ "linear_1.weight", &.{ proj_hidden, 4 * width });
        v.l1_b = try load1(a, f, pj ++ "linear_1.bias", &.{proj_hidden});
        v.l2_w = try load1(a, f, pj ++ "linear_2.weight", &.{ text_hidden, proj_hidden });
        v.l2_b = try load1(a, f, pj ++ "linear_2.bias", &.{text_hidden});
        const P = v.max_patches;
        v.px = try a.alloc(P * patch_width * 2);
        inline for (.{ "h", "x", "r", "y", "q" }) |field| @field(v, field) = try a.alloc(P * width * 2);
        v.k = try a.alloc(P * kv_width * 2);
        v.v = try a.alloc(P * kv_width * 2);
        v.bcx = try a.alloc(P * 3 * width * 2);
        inline for (.{ "gate", "up", "act" }) |field| @field(v, field) = try a.alloc(P * ffn * 2);
        v.acc = try a.alloc(max_tokens * proj_hidden * 4);
        v.hid = try a.alloc(max_tokens * proj_hidden * 2);
        v.host_px = try gpa.alloc(u16, P * patch_width);
        return v;
    }

    pub fn deinit(v: *Vision, gpa: std.mem.Allocator) void {
        gpa.free(v.host_px);
    }

    fn load1(a: *Arena, f: *const st.File, name: []const u8, shape: []const usize) !u64 {
        return a.upload(try stack.tensor(f, name, shape));
    }

    /// Layer i's tensor `suffix`.
    fn lt(a: *Arena, f: *const st.File, i: usize, comptime suffix: []const u8, shape: []const usize) !u64 {
        var buf: [160]u8 = undefined;
        return load1(a, f, try std.fmt.bufPrint(&buf, "model.vision_tower.layers.{d}." ++ suffix, .{i}), shape);
    }

    /// Every picture's patches through the tower and projector: `out` receives one 2048-wide bf16 row a token
    /// (a picture's patches / 4, pictures in order). `tower_out`, if given, receives the tower's last hidden state
    /// (bf16 [patches, 1024]) for checks. Encoded on the device's open command buffer; the caller flushes.
    pub fn run(v: *Vision, k: *const kern.Kernels, pictures: []const Picture, out: u64) !usize {
        var patches: usize = 0;
        for (pictures) |p| {
            if (p.gh == 0 or p.gw == 0 or p.gh % 2 != 0 or p.gw % 2 != 0) return error.BadImageGrid;
            const n = @as(usize, p.gh) * p.gw;
            if (p.pixels.len != n * patch_width) return error.BadImageGrid;
            patches += n;
        }
        if (pictures.len == 0) return 0;
        if (patches > v.max_patches) return error.ImageTooLarge;
        const tokens = patches / 4;
        // the patches as bf16 (round to nearest even), as the reference casts its fp32 pixel values
        var at: usize = 0;
        for (pictures) |p| for (p.pixels) |x| {
            if (!std.math.isFinite(x)) return error.NonfiniteImage;
            v.host_px[at] = bf16(x);
            at += 1;
        };
        try v.g.put(v.px, std.mem.sliceAsBytes(v.host_px[0 .. patches * patch_width]));
        const g = v.g;
        k.linear(v.patch, v.px, patch_width, patch_width, width, v.r, width, patches);
        v.rms(v.r, v.norm, v.h, patches);
        for (&v.layers, 0..) |*l, i| {
            v.rms(v.h, l.op_norm, v.x, patches);
            if (attentionAt(i)) {
                k.linear(l.q, v.x, width, width, width, v.q, width, patches);
                k.linear(l.k, v.x, width, width, kv_width, v.k, kv_width, patches);
                k.linear(l.v, v.x, width, width, kv_width, v.v, kv_width, patches);
                var row: usize = 0;
                for (pictures) |p| {
                    const n = @as(usize, p.gh) * p.gw;
                    g.go(.l3v_qk, .{ 24, n, 1 }, 32, &.{ v.q + row * width * 2, v.k + row * kv_width * 2, l.q_norm, l.k_norm }, @as(u32, p.gw));
                    g.go(.l3v_attention_tiled, .{ (n + 7) / 8, 8, 1 }, 512, &.{ v.q + row * width * 2, v.k + row * kv_width * 2, v.v + row * kv_width * 2, v.y + row * width * 2 }, @as(u32, @intCast(n)));
                    row += n;
                }
                k.linear(l.o, v.y, width, width, width, v.r, width, patches);
            } else {
                k.linear(l.in_proj, v.x, width, width, 3 * width, v.bcx, 3 * width, patches);
                var row: usize = 0;
                for (pictures) |p| {
                    const n = @as(usize, p.gh) * p.gw;
                    const Grid = extern struct { gh: u32, gw: u32 };
                    g.go(.l3v_conv, .{ width / 256, n, 1 }, 256, &.{ v.bcx + row * 3 * width * 2, l.spatial, v.y + row * width * 2 }, Grid{ .gh = p.gh, .gw = p.gw });
                    row += n;
                }
                k.linear(l.out_proj, v.y, width, width, width, v.r, width, patches);
            }
            v.add(patches);
            v.rms(v.h, l.ffn_norm, v.x, patches);
            k.linear(l.w1, v.x, width, width, ffn, v.gate, ffn, patches);
            k.linear(l.w3, v.x, width, width, ffn, v.up, ffn, patches);
            const count: u32 = @intCast(patches * ffn);
            g.go(.l3v_swiglu, .{ (count + 255) / 256, 1, 1 }, 256, &.{ v.gate, v.up, v.act }, count);
            k.linear(l.w2, v.act, ffn, ffn, width, v.r, width, patches);
            v.add(patches);
        }
        // the projector: LayerNorm a patch, four merge-order patches a row, linear_1 (bf16), GELU, linear_2
        g.go(.l3v_layer_norm, .{ patches, 1, 1 }, 256, &.{ v.h, v.ln_w, v.ln_b, v.x }, {});
        k.linearF32(v.l1_w, v.x, 4 * width, 4 * width, proj_hidden, v.acc, tokens);
        k.biasAct(v.acc, v.l1_b, v.hid, tokens, proj_hidden, false, .gelu);
        k.linearF32(v.l2_w, v.hid, proj_hidden, proj_hidden, text_hidden, v.acc, tokens);
        k.biasAct(v.acc, v.l2_b, out, tokens, text_hidden, false, .none);
        return tokens;
    }

    /// The tower's last hidden state (before the projector) of the last `run`.
    pub fn towerOut(v: *const Vision) u64 {
        return v.h;
    }

    fn rms(v: *Vision, x: u64, w: u64, y: u64, rows: usize) void {
        v.g.go(.l3v_rms, .{ rows, 1, 1 }, 256, &.{ x, w, y }, {});
    }

    /// h += r, rounded.
    fn add(v: *Vision, rows: usize) void {
        const count: u32 = @intCast(rows * width);
        v.g.go(.l3v_add, .{ (count + 255) / 256, 1, 1 }, 256, &.{ v.r, v.h }, count);
    }
};

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
