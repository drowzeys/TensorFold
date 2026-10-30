//! LFM3-Omni's CUDA registrations: each kernel at the engine's shapes, and the planted controls that must go red.
const std = @import("std");
const core = @import("core");

const c = core.contract.spec;
const G = core.contract.gates.Gate;

pub const Which = enum { rmsnorm_res, rmsnorm_rows, head_norm, rope, copy_rows, pos, short_conv, attention, gate, swiglu, add, bias_act, layernorm, embed, embed_codes, depth_in, conv2d, f32_bf16, bf16_f32, fsq, repeat, polar, argmax, gemv, depth_pick, frame_end, hist_push, mel_frames, power, logmel, conv_rows, conv_state, attention_tiles };

/// One registration's settings beyond its shape: attention heads, rotary mode, activation, a control's gate.
pub const Cfg = struct {
    mode: i32 = 0,
    heads: usize = 0,
    kv_heads: usize = 0,
    hd: usize = 0,
    window: usize = 0,
    causal: bool = true,
    theta: f32 = 1e6,
    fraction: f32 = 1,
    pos0: usize = 0,
    act: i32 = 0,
    per_row: bool = false,
    cin: usize = 1,
    height: usize = 0,
    tm: usize = 1,
    f32_out: bool = false,
    temp: f32 = 0,
};

pub const Entry = struct {
    name: []const u8,
    which: Which,
    symbol: [:0]const u8,
    served: []const c.Shape,
    cfg: Cfg = .{},
    /// A control: its symbol in the controls image and the gate it must fail.
    ctl: ?[:0]const u8 = null,
    fails: G = .shapes,
};

fn s(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

const backbone: Cfg = .{ .mode = 1, .heads = 16, .kv_heads = 4, .hd = 128, .fraction = 0.5, .pos0 = 299 };
const sliding: Cfg = .{ .mode = 1, .heads = 16, .kv_heads = 4, .hd = 128, .window = 256, .theta = 10000 };
const encoder: Cfg = .{ .mode = 2, .heads = 8, .kv_heads = 4, .hd = 96, .window = 32 };
const depth: Cfg = .{ .mode = 1, .heads = 32, .kv_heads = 8, .hd = 32, .causal = false };
const codec: Cfg = .{ .mode = 0, .heads = 8, .kv_heads = 2, .hd = 64, .window = 30, .pos0 = 4000 };

const norm_shapes = [_]c.Shape{ s(1, 2048, 2048), s(512, 2048, 2048), s(384, 512, 512), s(40, 768, 768) };
const gemv1 = [_]c.Shape{ s(1, 2048, 2048), s(1, 6144, 2048), s(1, 2048, 7680), s(1, 2017, 1024) };

pub const entries = [_]Entry{
    .{ .name = "om_rmsnorm with its residual (bf16(w * bf16(x * rsqrt)))", .which = .rmsnorm_res, .symbol = "om_rmsnorm", .served = &norm_shapes },
    .{ .name = "om_rmsnorm a weight row a row (bf16(x * rsqrt * w))", .which = .rmsnorm_rows, .symbol = "om_rmsnorm", .served = &.{s(12, 1024, 1024)}, .cfg = .{ .mode = 1 } },
    .{ .name = "om_head_norm", .which = .head_norm, .symbol = "om_head_norm", .served = &.{ s(1, 16, 128), s(512, 4, 128), s(12, 32, 32), s(40, 8, 96), s(384, 8, 64) } },
    .{ .name = "om_rope bf16 cos/sin, half the head rotated", .which = .rope, .symbol = "om_rope", .served = &.{ s(1, 16, 128), s(512, 4, 128) }, .cfg = backbone },
    .{ .name = "om_rope in bf16 steps (speech encoder)", .which = .rope, .symbol = "om_rope", .served = &.{s(40, 8, 96)}, .cfg = encoder },
    .{ .name = "om_rope fp32 cos/sin (codec)", .which = .rope, .symbol = "om_rope", .served = &.{s(384, 8, 64)}, .cfg = codec },
    .{ .name = "om_copy_rows", .which = .copy_rows, .symbol = "om_copy_rows", .served = &.{ s(1, 512, 1), s(512, 512, 1) } },
    .{ .name = "om_pos", .which = .pos, .symbol = "om_pos", .served = &.{s(1, 1, 1)} },
    .{ .name = "om_short_conv (a few rows, the state kept)", .which = .short_conv, .symbol = "om_short_conv", .served = &.{ s(1, 2048, 7), s(4, 2048, 7), s(1, 512, 7), s(2, 768, 3) } },
    .{ .name = "om_attention backbone row, every key", .which = .attention, .symbol = "om_attention", .served = &.{ s(1, 16, 300), s(1, 16, 4096) }, .cfg = backbone },
    .{ .name = "om_attention backbone row, sliding 256", .which = .attention, .symbol = "om_attention", .served = &.{s(1, 16, 600)}, .cfg = sliding },
    .{ .name = "om_attention codec row, sliding 30", .which = .attention, .symbol = "om_attention", .served = &.{s(1, 8, 200)}, .cfg = codec },
    .{ .name = "om_gate", .which = .gate, .symbol = "om_gate", .served = &.{ s(1, 16, 128), s(512, 16, 128), s(12, 32, 32), s(384, 8, 64) } },
    .{ .name = "om_swiglu", .which = .swiglu, .symbol = "om_swiglu", .served = &.{ s(1, 7680, 1), s(512, 7680, 1), s(12, 2816, 1), s(40, 1792, 1), s(384, 2304, 1) } },
    .{ .name = "om_add", .which = .add, .symbol = "om_add", .served = &.{ s(1, 2048, 1), s(384, 512, 1) } },
    .{ .name = "om_bias_act a column's bias", .which = .bias_act, .symbol = "om_bias_act", .served = &.{ s(40, 768, 1), s(384, 1282, 1), s(1, 12288, 1) } },
    .{ .name = "om_bias_act a row's bias, SiLU", .which = .bias_act, .symbol = "om_bias_act", .served = &.{s(256, 1500, 1)}, .cfg = .{ .act = 1, .per_row = true } },
    .{ .name = "om_bias_act exact GELU", .which = .bias_act, .symbol = "om_bias_act", .served = &.{s(40, 2048, 1)}, .cfg = .{ .act = 2 } },
    .{ .name = "om_layernorm", .which = .layernorm, .symbol = "om_layernorm", .served = &.{ s(40, 768, 768), s(375, 768, 768) } },
    .{ .name = "om_embed", .which = .embed, .symbol = "om_embed", .served = &.{ s(1, 2048, 1), s(512, 2048, 1) } },
    .{ .name = "om_embed_codes (12 codebooks' rows summed)", .which = .embed_codes, .symbol = "om_embed_codes", .served = &.{s(1, 2048, 12)} },
    .{ .name = "om_depth_in", .which = .depth_in, .symbol = "om_depth_in", .served = &.{s(12, 1024, 1)} },
    .{ .name = "om_conv2d_s2 from one plane, SiLU", .which = .conv2d, .symbol = "om_conv2d_s2", .served = &.{s(256, 100, 9)}, .cfg = .{ .cin = 1, .height = 128, .act = 1 } },
    .{ .name = "om_conv2d_s2 depthwise", .which = .conv2d, .symbol = "om_conv2d_s2", .served = &.{ s(256, 50, 9), s(256, 25, 9) }, .cfg = .{ .cin = 256, .height = 64 } },
    .{ .name = "om_f32_bf16", .which = .f32_bf16, .symbol = "om_f32_bf16", .served = &.{s(128, 1000, 1)} },
    .{ .name = "om_bf16_f32", .which = .bf16_f32, .symbol = "om_bf16_f32", .served = &.{s(12, 2017, 1)} },
    .{ .name = "om_fsq_decode", .which = .fsq, .symbol = "om_fsq_decode", .served = &.{ s(1, 12, 1), s(64, 12, 1) } },
    .{ .name = "om_repeat_rows", .which = .repeat, .symbol = "om_repeat_rows", .served = &.{s(64, 512, 6)} },
    .{ .name = "om_polar", .which = .polar, .symbol = "om_polar", .served = &.{ s(384, 641, 1), s(6, 641, 1) } },
    .{ .name = "om_argmax", .which = .argmax, .symbol = "om_argmax", .served = &.{s(1, 128000, 1)} },
    .{ .name = "om_gemv_1 fp32 out", .which = .gemv, .symbol = "om_gemv_1", .served = &gemv1, .cfg = .{ .f32_out = true } },
    .{ .name = "om_gemv_1 bf16 out", .which = .gemv, .symbol = "om_gemv_1", .served = &gemv1 },
    .{ .name = "om_gemv_4", .which = .gemv, .symbol = "om_gemv_4", .served = &.{ s(4, 2048, 2048), s(3, 6144, 2048) }, .cfg = .{ .tm = 4, .f32_out = true } },
    .{ .name = "om_gemv_8", .which = .gemv, .symbol = "om_gemv_8", .served = &.{ s(8, 2048, 2048), s(6, 2048, 7680) }, .cfg = .{ .tm = 8, .f32_out = true } },
    .{ .name = "om_gemv_12", .which = .gemv, .symbol = "om_gemv_12", .served = &.{ s(12, 5632, 1024), s(12, 1024, 2816) }, .cfg = .{ .tm = 12, .f32_out = true } },
    .{ .name = "om_gemv_16", .which = .gemv, .symbol = "om_gemv_16", .served = &.{ s(16, 2048, 2048), s(13, 512, 512) }, .cfg = .{ .tm = 16, .f32_out = true } },
    .{ .name = "om_depth_pick greedy", .which = .depth_pick, .symbol = "om_depth_pick", .served = &.{ s(4, 2017, 16), s(4, 2017, 0) } },
    .{ .name = "om_depth_pick sampled", .which = .depth_pick, .symbol = "om_depth_pick", .served = &.{s(4, 2017, 16)}, .cfg = .{ .temp = 0.4 } },
    .{ .name = "om_frame_end", .which = .frame_end, .symbol = "om_frame_end", .served = &.{ s(1, 12, 1), s(2, 12, 1) } },
    .{ .name = "om_hist_push", .which = .hist_push, .symbol = "om_hist_push", .served = &.{s(1, 12, 64)} },
    .{ .name = "om_mel_frames", .which = .mel_frames, .symbol = "om_mel_frames", .served = &.{s(300, 400, 160)} },
    .{ .name = "om_power", .which = .power, .symbol = "om_power", .served = &.{s(300, 201, 1)} },
    .{ .name = "om_logmel", .which = .logmel, .symbol = "om_logmel", .served = &.{s(300, 128, 1)} },
    .{ .name = "om_short_conv_rows", .which = .conv_rows, .symbol = "om_short_conv_rows", .served = &.{ s(512, 2048, 7), s(40, 768, 3), s(384, 512, 7) } },
    .{ .name = "om_conv_state", .which = .conv_state, .symbol = "om_conv_state", .served = &.{ s(512, 2048, 7), s(40, 768, 3), s(2, 2048, 7) } },
    .{ .name = "om_attention_tiles backbone prompt, every key", .which = .attention_tiles, .symbol = "om_attention_tiles", .served = &.{ s(512, 16, 512), s(100, 16, 612) }, .cfg = backbone },
    .{ .name = "om_attention_tiles backbone prompt, sliding 256", .which = .attention_tiles, .symbol = "om_attention_tiles", .served = &.{s(512, 16, 512)}, .cfg = sliding },
    .{ .name = "om_attention_tiles depth decoder, both ways", .which = .attention_tiles, .symbol = "om_attention_tiles", .served = &.{s(12, 32, 12)}, .cfg = depth },
    .{ .name = "om_attention_tiles speech encoder, sliding 32", .which = .attention_tiles, .symbol = "om_attention_tiles", .served = &.{s(40, 8, 40)}, .cfg = encoder },
    .{ .name = "om_attention_tiles codec, sliding 30", .which = .attention_tiles, .symbol = "om_attention_tiles", .served = &.{s(384, 8, 384)}, .cfg = codec },
} ++ controls;

/// Each control: a copy of a kernel with one planted fault (contract_lfm3_omni_controls.cuh).
pub const controls = [_]Entry{
    ctl("control: om_rmsnorm without x * rsqrt's rounding", .rmsnorm_res, "tf_ctl_om_rmsnorm", .shapes, .{}),
    ctl("control: om_head_norm without x * rsqrt's rounding", .head_norm, "tf_ctl_om_head_norm", .shapes, .{}),
    ctl("control: om_rope bf16 steps without a product's rounding", .rope, "tf_ctl_om_rope", .shapes, encoder),
    ctl("control: om_copy_rows reading row 0 for every row", .copy_rows, "tf_ctl_om_copy_rows", .shapes, .{}),
    ctl("control: om_pos advancing twice", .pos, "tf_ctl_om_pos", .shapes, .{}),
    ctl("control: om_short_conv without B * x's rounding", .short_conv, "tf_ctl_om_short_conv", .shapes, .{}),
    ctl("control: om_attention sliding one key short", .attention, "tf_ctl_om_attention", .shapes, sliding),
    ctl("control: om_gate without the gate's rounding", .gate, "tf_ctl_om_gate", .shapes, .{}),
    ctl("control: om_swiglu without silu's rounding", .swiglu, "tf_ctl_om_swiglu", .shapes, .{}),
    ctl("control: om_add dropping every eighth residual", .add, "tf_ctl_om_add", .shapes, .{}),
    ctl("control: om_bias_act GELU without the bias sum's rounding", .bias_act, "tf_ctl_om_bias_act", .shapes, .{ .act = 2 }),
    ctl("control: om_layernorm with a bf16 mean", .layernorm, "tf_ctl_om_layernorm", .shapes, .{}),
    ctl("control: om_embed writing zeros for a placeholder row", .embed, "tf_ctl_om_embed", .shapes, .{}),
    ctl("control: om_embed_codes summed in bf16", .embed_codes, "tf_ctl_om_embed_codes", .sums, .{}),
    ctl("control: om_depth_in ignoring the mask", .depth_in, "tf_ctl_om_depth_in", .shapes, .{}),
    ctl("control: om_conv2d_s2 without the rounding before SiLU", .conv2d, "tf_ctl_om_conv2d_s2", .shapes, .{ .cin = 1, .height = 128, .act = 1 }),
    ctl("control: om_f32_bf16 truncating", .f32_bf16, "tf_ctl_om_f32_bf16", .shapes, .{}),
    ctl("control: om_bf16_f32 one element late", .bf16_f32, "tf_ctl_om_bf16_f32", .shapes, .{}),
    ctl("control: om_fsq_decode with 8 levels in the second digit", .fsq, "tf_ctl_om_fsq_decode", .shapes, .{}),
    ctl("control: om_repeat_rows repeating one too many", .repeat, "tf_ctl_om_repeat_rows", .shapes, .{}),
    ctl("control: om_polar with a bf16 magnitude", .polar, "tf_ctl_om_polar", .shapes, .{}),
    ctl("control: om_argmax tying to the higher id", .argmax, "tf_ctl_om_argmax", .shapes, .{}),
    ctl("control: om_gemv_1 summed in fp16", .gemv, "tf_ctl_om_gemv", .sums, .{ .f32_out = true }),
    ctl("control: om_depth_pick tying to the higher id", .depth_pick, "tf_ctl_om_depth_pick", .shapes, .{}),
    ctl("control: om_frame_end ending one code", .frame_end, "tf_ctl_om_frame_end", .shapes, .{}),
    ctl("control: om_hist_push into row 0", .hist_push, "tf_ctl_om_hist_push", .shapes, .{}),
    ctl("control: om_mel_frames reading past the clip", .mel_frames, "tf_ctl_om_mel_frames", .shapes, .{}),
    ctl("control: om_power without the imaginary part", .power, "tf_ctl_om_power", .shapes, .{}),
    ctl("control: om_logmel without its floor", .logmel, "tf_ctl_om_logmel", .shapes, .{}),
    ctl("control: om_short_conv_rows reading the state one tap off", .conv_rows, "tf_ctl_om_short_conv_rows", .shapes, .{}),
    ctl("control: om_conv_state keeping the state one row late", .conv_state, "tf_ctl_om_conv_state", .shapes, .{}),
    ctl("control: om_attention_tiles dropping a row's last key", .attention_tiles, "tf_ctl_om_attention_tiles", .shapes, backbone),
};

fn ctl(name: []const u8, w: Which, symbol: [:0]const u8, gate: G, cfg: Cfg) Entry {
    return .{ .name = name, .which = w, .symbol = symbol, .served = served(w), .cfg = cfg, .ctl = symbol, .fails = gate };
}

/// A control's shapes: its kernel's served ones where they are cheap, else a smaller served one.
fn served(w: Which) []const c.Shape {
    return switch (w) {
        .rmsnorm_res => &.{s(16, 2048, 2048)},
        .head_norm => &.{s(40, 8, 96)},
        .rope => &.{s(40, 8, 96)},
        .copy_rows => &.{s(16, 512, 1)},
        .pos => &.{s(1, 1, 1)},
        .short_conv => &.{s(4, 2048, 7)},
        .attention => &.{s(1, 16, 600)},
        .gate => &.{s(12, 32, 32)},
        .swiglu => &.{s(12, 2816, 1)},
        .add => &.{s(1, 2048, 1)},
        .bias_act => &.{s(40, 2048, 1)},
        .layernorm => &.{s(40, 768, 768)},
        .embed => &.{s(64, 2048, 1)},
        .embed_codes => &.{s(1, 2048, 12)},
        .depth_in => &.{s(12, 1024, 1)},
        .conv2d => &.{s(256, 100, 9)},
        .f32_bf16 => &.{s(128, 200, 1)},
        .bf16_f32 => &.{s(12, 2017, 1)},
        .fsq => &.{s(64, 12, 1)},
        .repeat => &.{s(16, 512, 6)},
        .polar => &.{s(6, 641, 1)},
        .argmax => &.{s(1, 128000, 1)},
        .gemv => &.{s(1, 2048, 2048)},
        .depth_pick => &.{s(4, 2017, 16)},
        .frame_end => &.{ s(1, 12, 1), s(2, 12, 1) },
        .hist_push => &.{s(1, 12, 64)},
        .mel_frames => &.{s(300, 400, 160)},
        .power => &.{s(300, 201, 1)},
        .logmel => &.{s(300, 128, 1)},
        .conv_rows => &.{s(64, 2048, 7)},
        .conv_state => &.{s(64, 2048, 7)},
        .attention_tiles => &.{s(100, 16, 612)},
        else => unreachable,
    };
}
