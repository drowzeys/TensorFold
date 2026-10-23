//! The Drama backbone's checkpoint tensors on the GPU, as stored: bf16 or FP8 (backbone_dense) dense weights, e4m3
//! experts with fp32 128 x 128 block scales or NVFP4 ones in the grouped kernels' blocks. None is requantized.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("backbone_config.zig");
const Stager = @import("stager.zig").Stager;
const source = @import("backbone_source.zig");
const stored_fp4 = @import("backbone_fp4.zig");
const compact = @import("compact.zig");
pub const dense = @import("backbone_dense.zig");

const Source = source.Source;
const DType = source.DType;

const Io = std.Io;

pub const prefix = "model.language_model.";
pub const head_name = "lm_head.weight";

/// Rows of the GDN input projection: in_proj_qkv (8192) | in_proj_z (4096) | in_proj_b (32) | in_proj_a (32).
pub const gdn_in_rows = cfg.conv_dim + cfg.value_dim + 2 * cfg.linear_value_heads;
pub const gdn_z_col = cfg.conv_dim;
pub const gdn_b_col = cfg.conv_dim + cfg.value_dim;
/// Rows of the attention input projection: q_proj (16 x [q 256 | gate 256]) | k_proj (512) | v_proj (512).
pub const attn_in_rows = 2 * cfg.query_heads * cfg.head_dim + 2 * cfg.kv_heads * cfg.head_dim;
/// Rows of the MoE input projection: shared gate_proj (512) | shared up_proj (512) | router gate (256).
pub const moe_in_rows = 2 * cfg.shared_inner + cfg.experts;
pub const router_col = 2 * cfg.shared_inner;
/// Semantic head rows: EOS, the 4096-code band, zero rows to a multiple of 8.
pub const head_cols = 1 + 4096;
pub const head_rows = (head_cols + 7) / 8 * 8;
/// One expert's gate | up rows and their scale blocks.
pub const gu_rows = 2 * cfg.expert_inner;
pub const gu_scales = (gu_rows / cfg.block) * (cfg.hidden / cfg.block);
pub const dn_scales = (cfg.hidden / cfg.block) * (cfg.expert_inner / cfg.block);
/// One projection's NVFP4 experts as stored, for the packer: codes [experts][n][k / 2], scales [experts][n][k / 16].
pub const fp4_part = cfg.experts * cfg.expert_inner * cfg.hidden * 9 / 16;
/// An NVFP4 layer's experts view, its four device pointers still to set: Drama's SwiGLU rounding, no clamp.
pub const fp4_layer: cuda.experts.Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .width = cfg.expert_inner, .dims = cfg.hidden, .experts = cfg.experts, .silu_f32 = true };

pub const Layer = struct {
    ln1: u64 = 0, // bf16 [hidden] input_layernorm (zero-centered)
    ln2: u64 = 0, // bf16 [hidden] post_attention_layernorm (zero-centered)
    // Gated DeltaNet
    gdn_in: u64 = 0, // bf16 [gdn_in_rows][hidden]
    conv: u64 = 0, // bf16 [conv_dim][4]
    a_log: u64 = 0, // f32 [32], widened from the stored bf16
    dt_bias: u64 = 0, // f32 [32], widened from the stored bf16
    gnorm: u64 = 0, // bf16 [128] RMSNormGated weight (plain, not zero-centered)
    gdn_out: u64 = 0, // bf16 [hidden][value_dim]
    // full attention
    attn_in: u64 = 0, // bf16 [attn_in_rows][hidden]
    q_norm: u64 = 0, // bf16 [256] (zero-centered)
    k_norm: u64 = 0, // bf16 [256] (zero-centered)
    attn_out: u64 = 0, // bf16 [hidden][query_heads * head_dim]
    // MoE
    moe_in: u64 = 0, // bf16 [moe_in_rows][hidden]
    shared_down: u64 = 0, // bf16 [hidden][shared_inner]
    shared_gate: u64 = 0, // bf16 [hidden]
    gu: u64 = 0, // e4m3 [held][gu_rows][hidden]
    gu_s: u64 = 0, // f32 [held][gu_rows / 128][hidden / 128]
    dn: u64 = 0, // e4m3 [held][hidden][expert_inner]
    dn_s: u64 = 0, // f32 [held][hidden / 128][expert_inner / 128]
    experts: usize = cfg.experts, // routed experts held: slot s holds the map's s-th original id
    nvfp4: ?cuda.experts.Layer = null, // the routed experts when stored NVFP4; gu .. dn_s are then 0
    fp8: dense.Groups = .{}, // dense groups stored FP8; each one's bf16 field above is then 0
    gdn_ba: u64 = 0, // bf16 [64][hidden] in_proj_b | in_proj_a when fp8.gdn_in holds qkv | z
    router: u64 = 0, // bf16 [experts][hidden] mlp.gate when fp8.shared_in holds the shared gate | up
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    embed: u64 = 0, // bf16 [vocab][hidden]
    final_norm: u64 = 0, // bf16 [hidden] (zero-centered)
    head: u64 = 0, // bf16 [head_rows][hidden]
    inv_freq: u64 = 0, // f32 [rotary_dim / 2]
    layers: [cfg.layers]Layer = @splat(.{}),
    held: ?compact.Held = null, // a compact checkpoint's expert map; null: every layer holds all of its experts
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.* = undefined;
    }

    pub fn nvfp4Layers(w: *const Weights) usize {
        var n: usize = 0;
        for (w.layers) |l| n += @intFromBool(l.nvfp4 != null);
        return n;
    }

    pub fn denseFp8(w: *const Weights) dense.Census {
        var c: dense.Census = .{};
        for (w.layers) |l| {
            var any = false;
            inline for (dense.group_names, 0..) |g, j| if (@field(l.fp8, g) != null) {
                c.groups[j] += 1;
                any = true;
            };
            c.layers += @intFromBool(any);
        }
        return c;
    }

    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !u64 {
        try w.buffers.ensureUnusedCapacity(w.gpa, 1);
        const b = try cuda.DeviceBuffer.alloc(d, bytes);
        w.buffers.appendAssumeCapacity(b);
        w.bytes += bytes;
        return b.ptr;
    }
};

fn widenBf16(out: []f32, bytes: []const u8) void {
    for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16);
}

const Loader = struct {
    src: *Source,
    st: *Stager,
    w: *Weights,
    d: *const cuda.Driver,
    name_buf: [192]u8 = undefined,
    fp4: ?Fp4Stage = null,
    fp8: ?dense.Stage = null,

    fn name(L: *Loader, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&L.name_buf, fmt, args);
    }

    /// A fresh device tensor holding `name` as stored.
    fn raw(L: *Loader, n: []const u8, dtype: DType, shape: []const usize) !u64 {
        const bytes = try L.src.get(n, dtype, shape);
        const p = try L.w.alloc(L.d, bytes.len);
        try L.st.put(p, bytes);
        return p;
    }

    /// `name` as stored at `dst` (part of a fused tensor).
    fn into(L: *Loader, dst: u64, n: []const u8, dtype: DType, shape: []const usize) !void {
        try L.st.put(dst, try L.src.get(n, dtype, shape));
    }

    fn layerRaw(L: *Loader, i: usize, comptime suffix: []const u8, dtype: DType, shape: []const usize) !u64 {
        return L.raw(try L.name(prefix ++ "layers.{d}." ++ suffix, .{i}), dtype, shape);
    }

    fn layerInto(L: *Loader, dst: u64, i: usize, comptime suffix: []const u8, dtype: DType, shape: []const usize) !void {
        try L.into(dst, try L.name(prefix ++ "layers.{d}." ++ suffix, .{i}), dtype, shape);
    }

    fn widened(L: *Loader, i: usize, comptime suffix: []const u8) !u64 {
        const n = cfg.linear_value_heads;
        const bytes = try L.src.get(try L.name(prefix ++ "layers.{d}." ++ suffix, .{i}), .bf16, &.{n});
        var host: [n]f32 = undefined;
        widenBf16(&host, bytes);
        const p = try L.w.alloc(L.d, n * 4);
        try L.st.put(p, std.mem.sliceAsBytes(&host));
        return p;
    }

    fn ropeFrequencies(L: *Loader) !void {
        const inv = ropeInvFreq();
        L.w.inv_freq = try L.w.alloc(L.d, inv.len * 4);
        try L.st.put(L.w.inv_freq, std.mem.sliceAsBytes(&inv));
    }

    fn fp4Stage(L: *Loader) !*Fp4Stage {
        if (L.fp4 == null) {
            var packer = try cuda.modelopt.Packer.load(L.d);
            errdefer packer.unload();
            L.fp4 = .{ .bytes = try cuda.DeviceBuffer.alloc(L.d, stored_fp4.parts * fp4_part), .packer = packer };
        }
        return &L.fp4.?;
    }

    /// Layer i's group of `stems` ([rows[j], k] each) as one lane weight when stored FP8; null when stored BF16.
    fn fp8Group(L: *Loader, i: usize, comptime stems: []const []const u8, rows: []const usize, k: usize) !?cuda.qlinear.Weight {
        const fp8 = groupFp8(L.src, i, stems) catch |err| {
            std.log.err("drama backbone: layer {d}'s {s} group mixes BF16 and FP8 parts", .{ i, stems[0] });
            return err;
        };
        if (!fp8) return null;
        var n: usize = 0;
        for (rows) |r| n += r;
        const kg = k / cuda.fp8.group;
        const codes = try L.w.alloc(L.d, n * k);
        const scales = try L.w.alloc(L.d, n * kg * 4);
        if (L.fp8 == null) L.fp8 = try dense.Stage.init(L.d);
        // each part's rows are whole 128-row blocks, so the fused pack is the concatenation of the parts' packs
        var at: usize = 0;
        inline for (stems, 0..) |stem, j| {
            const part = try L.fp8Part(i, stem, rows[j], k);
            try L.fp8.?.add(L.st, part.codes, codes + at * k, rows[j], k);
            const tiles = try dense.tiles(L.w.gpa, part.scales, rows[j], k);
            defer L.w.gpa.free(tiles);
            try L.st.put(scales + at * kg * 4, std.mem.sliceAsBytes(tiles));
            at += rows[j];
        }
        return .{ .fp8g = cuda.fp8.weight(codes, scales, n, k) };
    }

    /// Layer i's `stem` refused unless C5's codes [n, k] and row scales [n, k / 64] with C5's values.
    fn fp8Part(L: *Loader, i: usize, comptime stem: []const u8, n: usize, k: usize) !dense.Part {
        const codes = try L.name(prefix ++ "layers.{d}." ++ stem ++ ".weight", .{i});
        const c = try L.src.get(codes, .f8_e4m3, &.{ n, k });
        dense.checkCodes(c) catch |err| {
            std.log.err("drama backbone: {s} holds an e4m3 NaN code (0x7F or 0xFF)", .{codes});
            return err;
        };
        const scales = try L.name(prefix ++ "layers.{d}." ++ stem ++ ".weight_scale_inv", .{i});
        const s = try L.src.get(scales, .f32, &.{ n, k / cuda.fp8.group });
        dense.checkScales(s) catch |err| {
            std.log.err("drama backbone: {s} holds a scale that is not positive and finite", .{scales});
            return err;
        };
        return .{ .codes = c, .scales = s };
    }

    /// Frees the NVFP4 and FP8 staging once the stream has run every pack queued from it.
    fn release(L: *Loader) void {
        if (L.fp4 == null and L.fp8 == null) return;
        L.st.stream.synchronize() catch {};
        if (L.fp4) |*f| {
            f.bytes.free();
            f.packer.unload();
        }
        if (L.fp8) |*s| s.deinit();
    }
};

/// Whether layer i's group is stored FP8: its first part's dtype decides, and a group of both formats is refused (C8).
fn groupFp8(src: *const Source, i: usize, comptime stems: []const []const u8) !bool {
    var buf: [192]u8 = undefined;
    var fp8: ?bool = null;
    inline for (stems) |stem| {
        const e = src.names.get(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}." ++ stem ++ ".weight", .{i}));
        const this = e != null and e.?.dtype == .f8_e4m3;
        if (fp8 != null and fp8.? != this) return error.MixedDramaFormats;
        fp8 = this;
    }
    return fp8.?;
}

/// Device room for one layer's stored NVFP4 experts (gate, up, down parts) and the packer that reads it.
const Fp4Stage = struct { bytes: cuda.DeviceBuffer, packer: cuda.modelopt.Packer };

/// `l.experts` stored NVFP4 gate, up, down (fp4_part bytes each at `staged`) into `l`'s blocks, gate, up SwiGLU 0, 1.
pub fn packFp4(p: *const cuda.modelopt.Packer, s: cuda.Stream, staged: u64, l: cuda.experts.Layer) !void {
    for (stored_fp4.shapes, 0..) |shape, j| {
        const n, const k = shape;
        const codes = staged + j * fp4_part;
        const scales = codes + cfg.experts * n * k / 2;
        if (j < 2) try p.fp4Experts(s, codes, scales, l.up, l.experts, n, k, 0, k, 2, j) else try p.fp4Experts(s, codes, scales, l.down, l.experts, n, k, 0, k, 1, 0);
    }
}

const H = cfg.hidden;

fn loadLinear(L: *Loader, i: usize, l: *Layer) !void {
    const row = H * 2;
    const heads = cfg.linear_value_heads;
    l.fp8.gdn_in = try L.fp8Group(i, &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z" }, &.{ cfg.conv_dim, cfg.value_dim }, H);
    if (l.fp8.gdn_in != null) {
        l.gdn_ba = try L.w.alloc(L.d, 2 * heads * row);
        try L.layerInto(l.gdn_ba, i, "linear_attn.in_proj_b.weight", .bf16, &.{ heads, H });
        try L.layerInto(l.gdn_ba + heads * row, i, "linear_attn.in_proj_a.weight", .bf16, &.{ heads, H });
    } else {
        l.gdn_in = try L.w.alloc(L.d, gdn_in_rows * row);
        try L.layerInto(l.gdn_in, i, "linear_attn.in_proj_qkv.weight", .bf16, &.{ cfg.conv_dim, H });
        try L.layerInto(l.gdn_in + gdn_z_col * row, i, "linear_attn.in_proj_z.weight", .bf16, &.{ cfg.value_dim, H });
        try L.layerInto(l.gdn_in + gdn_b_col * row, i, "linear_attn.in_proj_b.weight", .bf16, &.{ heads, H });
        try L.layerInto(l.gdn_in + (gdn_b_col + heads) * row, i, "linear_attn.in_proj_a.weight", .bf16, &.{ heads, H });
    }
    l.conv = try L.layerRaw(i, "linear_attn.conv1d.weight", .bf16, &.{ cfg.conv_dim, 1, cfg.conv_taps });
    l.a_log = try L.widened(i, "linear_attn.A_log");
    l.dt_bias = try L.widened(i, "linear_attn.dt_bias");
    l.gnorm = try L.layerRaw(i, "linear_attn.norm.weight", .bf16, &.{cfg.linear_dim});
    l.fp8.gdn_out = try L.fp8Group(i, &.{"linear_attn.out_proj"}, &.{H}, cfg.value_dim);
    if (l.fp8.gdn_out == null) l.gdn_out = try L.layerRaw(i, "linear_attn.out_proj.weight", .bf16, &.{ H, cfg.value_dim });
}

fn loadFull(L: *Loader, i: usize, l: *Layer) !void {
    const row = H * 2;
    const q_rows = 2 * cfg.query_heads * cfg.head_dim;
    const kv_rows = cfg.kv_heads * cfg.head_dim;
    l.fp8.attn_in = try L.fp8Group(i, &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, &.{ q_rows, kv_rows, kv_rows }, H);
    if (l.fp8.attn_in == null) {
        l.attn_in = try L.w.alloc(L.d, attn_in_rows * row);
        try L.layerInto(l.attn_in, i, "self_attn.q_proj.weight", .bf16, &.{ q_rows, H });
        try L.layerInto(l.attn_in + q_rows * row, i, "self_attn.k_proj.weight", .bf16, &.{ kv_rows, H });
        try L.layerInto(l.attn_in + (q_rows + kv_rows) * row, i, "self_attn.v_proj.weight", .bf16, &.{ kv_rows, H });
    }
    l.q_norm = try L.layerRaw(i, "self_attn.q_norm.weight", .bf16, &.{cfg.head_dim});
    l.k_norm = try L.layerRaw(i, "self_attn.k_norm.weight", .bf16, &.{cfg.head_dim});
    l.fp8.attn_out = try L.fp8Group(i, &.{"self_attn.o_proj"}, &.{H}, cfg.query_heads * cfg.head_dim);
    if (l.fp8.attn_out == null) l.attn_out = try L.layerRaw(i, "self_attn.o_proj.weight", .bf16, &.{ H, cfg.query_heads * cfg.head_dim });
}

fn loadMoe(L: *Loader, i: usize, l: *Layer) !void {
    const row = H * 2;
    const I = cfg.expert_inner;
    l.fp8.shared_in = try L.fp8Group(i, &.{ "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj" }, &.{ cfg.shared_inner, cfg.shared_inner }, H);
    if (l.fp8.shared_in != null) {
        l.router = try L.layerRaw(i, "mlp.gate.weight", .bf16, &.{ cfg.experts, H });
    } else {
        l.moe_in = try L.w.alloc(L.d, moe_in_rows * row);
        try L.layerInto(l.moe_in, i, "mlp.shared_expert.gate_proj.weight", .bf16, &.{ cfg.shared_inner, H });
        try L.layerInto(l.moe_in + cfg.shared_inner * row, i, "mlp.shared_expert.up_proj.weight", .bf16, &.{ cfg.shared_inner, H });
        try L.layerInto(l.moe_in + router_col * row, i, "mlp.gate.weight", .bf16, &.{ cfg.experts, H });
    }
    l.fp8.shared_out = try L.fp8Group(i, &.{"mlp.shared_expert.down_proj"}, &.{H}, cfg.shared_inner);
    if (l.fp8.shared_out == null) l.shared_down = try L.layerRaw(i, "mlp.shared_expert.down_proj.weight", .bf16, &.{ H, cfg.shared_inner });
    l.shared_gate = try L.layerRaw(i, "mlp.shared_expert_gate.weight", .bf16, &.{ 1, H });
    const ids = (if (L.w.held) |*h| h else &full).layer(i);
    l.experts = ids.len;
    if (fp4Layer(L.src, i, ids)) return loadFp4(L, i, ids, l);
    const E = ids.len;
    l.gu = try L.w.alloc(L.d, E * gu_rows * H);
    l.gu_s = try L.w.alloc(L.d, E * gu_scales * 4);
    l.dn = try L.w.alloc(L.d, E * H * I);
    l.dn_s = try L.w.alloc(L.d, E * dn_scales * 4);
    try putFp8(L, i, ids, l);
}

/// Every expert of every layer, in place: a checkpoint without a map.
const full: compact.Held = .{};

/// Whether layer i's experts are NVFP4: its first held expert's gate decides; get() holds the rest to that format.
fn fp4Layer(src: *const Source, i: usize, ids: []const u8) bool {
    var buf: [192]u8 = undefined;
    const gate = src.names.get(std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ i, ids[0] }) catch unreachable);
    return gate != null and gate.?.dtype == DType.u8;
}

/// Layer i's held FP8 experts into `l`'s arrays: expert ids[s] at slot s.
fn putFp8(L: *Loader, i: usize, ids: []const u8, l: *const Layer) !void {
    const I = cfg.expert_inner;
    const gs = &[_]usize{ I / cfg.block, H / cfg.block };
    const ds = &[_]usize{ H / cfg.block, I / cfg.block };
    for (ids, 0..) |e, s| {
        const gu = l.gu + s * gu_rows * H;
        const gu_s = l.gu_s + s * gu_scales * 4;
        try L.into(gu, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ i, e }), .f8_e4m3, &.{ I, H });
        try L.into(gu + I * H, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.up_proj.weight", .{ i, e }), .f8_e4m3, &.{ I, H });
        try L.into(gu_s, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.gate_proj.weight_scale_inv", .{ i, e }), .f32, gs);
        try L.into(gu_s + gu_scales * 2, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.up_proj.weight_scale_inv", .{ i, e }), .f32, gs);
        try L.into(l.dn + s * H * I, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.down_proj.weight", .{ i, e }), .f8_e4m3, &.{ H, I });
        try L.into(l.dn_s + s * dn_scales * 4, try L.name(prefix ++ "layers.{d}.mlp.experts.{d}.down_proj.weight_scale_inv", .{ i, e }), .f32, ds);
    }
}

fn loadFp4(L: *Loader, i: usize, ids: []const u8, l: *Layer) !void {
    const E = ids.len;
    const I = cfg.expert_inner;
    const fp4 = cuda.experts.Nvfp4Experts;
    const staged = (try L.fp4Stage()).bytes.ptr;
    var globals: [stored_fp4.parts][cfg.experts]f32 = undefined;
    try putFp4(L, i, ids, staged, &globals);
    var ex = fp4_layer;
    ex.experts = E;
    ex.up = try L.w.alloc(L.d, fp4.bytes(E, I, H, 2));
    ex.down = try L.w.alloc(L.d, fp4.bytes(E, H, I, 1));
    try L.st.finish(); // the stored experts land before the packs read them; the next layer's copies queue after the packs
    try packFp4(&L.fp4.?.packer, L.st.stream, staged, ex);
    var up_scale: [cfg.experts][2]f32 = undefined;
    for (up_scale[0..E], globals[0][0..E], globals[1][0..E]) |*s, g, u| s.* = .{ g, u };
    ex.up_scale = try L.w.alloc(L.d, E * @sizeOf([2]f32));
    try L.st.put(ex.up_scale, std.mem.sliceAsBytes(up_scale[0..E]));
    ex.down_scale = try L.w.alloc(L.d, E * 4);
    try L.st.put(ex.down_scale, std.mem.sliceAsBytes(globals[2][0..E]));
    l.nvfp4 = ex;
}

/// Layer i's held NVFP4 experts into the packer's staging at `staged` and their globals: expert ids[s] at slot s.
fn putFp4(L: *Loader, i: usize, ids: []const u8, staged: u64, globals: *[stored_fp4.parts][cfg.experts]f32) !void {
    for (ids, 0..) |e, s| {
        const x = try stored_fp4.read(L.src, i, e);
        for (stored_fp4.shapes, 0..) |shape, j| {
            const n, const k = shape;
            const at = staged + j * fp4_part;
            try L.st.put(at + s * n * k / 2, x.codes[j]);
            try L.st.put(at + cfg.experts * n * k / 2 + s * n * k / 16, x.scales[j]);
            globals[j][s] = x.global[j];
        }
    }
}

/// The checkpoint's metadata.drama_experts, refused unless compact.parse admits it and it holds every stored expert.
pub fn expertMap(src: *Source) !?compact.Held {
    const value = src.metadata.get(compact.experts_key) orelse return null;
    var arena: std.heap.ArenaAllocator = .init(src.gpa);
    defer arena.deinit();
    var problem: []const u8 = "";
    const held = try compact.parse(arena.allocator(), value, &problem) orelse {
        std.log.err("drama backbone: metadata.{s}: {s}", .{ compact.experts_key, problem });
        return error.BadDramaExpertMap;
    };
    if (compact.stray(&held, src.names.keyIterator())) |name| {
        std.log.err("drama backbone: {s} belongs to an expert metadata.{s} does not hold", .{ name, compact.experts_key });
        return error.BadDramaExpertMap;
    }
    return held;
}

/// Internal test seam: loadMoe's puts of layer i's held `ids` (FP8 at `l`, NVFP4 at `staged`) on a driverless `st`.
pub fn putExperts(src: *Source, st: *Stager, i: usize, ids: []const u8, l: *const Layer, staged: u64) !?[stored_fp4.parts][cfg.experts]f32 {
    var L: Loader = .{ .src = src, .st = st, .w = undefined, .d = undefined };
    if (!fp4Layer(src, i, ids)) {
        try putFp8(&L, i, ids, l);
        return null;
    }
    var globals: [stored_fp4.parts][cfg.experts]f32 = undefined;
    try putFp4(&L, i, ids, staged, &globals);
    return globals;
}

/// The checkpoint folder's backbone tensors (and the semantic rows of lm_head) uploaded on `stream`; returns after
/// every copy has landed. Refuses missing, mis-shaped or unconsumed `model.language_model.*` tensors.
pub fn load(gpa: std.mem.Allocator, io: Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, c: cfg.Config) !Weights {
    return loadOf(gpa, io, d, stream, dir, c, null);
}

/// Internal test seam: layers `only` of a checkpoint folder holding no other tensor, each read as `load` reads it.
pub fn loadLayers(gpa: std.mem.Allocator, io: Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, only: []const usize) !Weights {
    return loadOf(gpa, io, d, stream, dir, null, only);
}

fn loadOf(gpa: std.mem.Allocator, io: Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, c: ?cfg.Config, only: ?[]const usize) !Weights {
    var src = try Source.open(gpa, io, dir);
    defer src.close();
    var st: Stager = .init(gpa, d, stream);
    defer st.deinit();
    for (src.shards.items) |sh| try st.addFile(sh.path, sh.map.memory);
    var w: Weights = .{ .gpa = gpa, .held = try expertMap(&src) };
    errdefer {
        stream.synchronize() catch {};
        w.deinit();
    }
    var L: Loader = .{ .src = &src, .st = &st, .w = &w, .d = d };
    defer L.release();
    if (c != null) {
        w.embed = try L.raw(prefix ++ "embed_tokens.weight", .bf16, &.{ cfg.vocab, H });
        w.final_norm = try L.raw(prefix ++ "norm.weight", .bf16, &.{H});
    }
    for (&w.layers, 0..) |*l, i| {
        if (only) |o| if (std.mem.indexOfScalar(usize, o, i) == null) continue;
        l.ln1 = try L.layerRaw(i, "input_layernorm.weight", .bf16, &.{H});
        l.ln2 = try L.layerRaw(i, "post_attention_layernorm.weight", .bf16, &.{H});
        if (cfg.linear(i)) try loadLinear(&L, i, l) else try loadFull(&L, i, l);
        try loadMoe(&L, i, l);
        if (L.fp8) |*s| try s.pack(&st);
    }
    if (c) |config| try loadHead(&L, config) else try L.ropeFrequencies(); // attention reads them in every layer
    if (src.used != src.backbone) {
        std.log.err("drama backbone: {d} of {d} {s}* tensors left unused", .{ src.backbone - src.used, src.backbone, prefix });
        return error.UnusedDramaTensors;
    }
    try st.finish();
    return w;
}

/// The semantic rows of lm_head (EOS, then semantic_start..semantic_end), zero padded, and RoPE's inverse frequencies.
fn loadHead(L: *Loader, c: cfg.Config) !void {
    const head = try L.src.get(head_name, .bf16, &.{ cfg.vocab, H });
    const row = H * 2;
    L.w.head = try L.w.alloc(L.d, head_rows * row);
    try L.st.zero(L.w.head, head_rows * row);
    try L.st.put(L.w.head, head[@as(usize, c.eos) * row ..][0..row]);
    try L.st.put(L.w.head + row, head[@as(usize, c.semantic_start) * row ..][0 .. (head_cols - 1) * row]);
    try L.ropeFrequencies();
}

/// RoPE inverse frequencies as SGLang computes them in fp32: 1 / theta^(2i / rotary_dim).
pub fn ropeInvFreq() [cfg.rotary_dim / 2]f32 {
    var inv: [cfg.rotary_dim / 2]f32 = undefined;
    for (&inv, 0..) |*f, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, cfg.rotary_dim);
        f.* = 1.0 / std.math.pow(f32, @floatCast(cfg.theta), e);
    }
    return inv;
}

test "C8: a group's first part decides its format; mixed fused groups, bad companions and strays are refused" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const S = cfg.shared_inner;
    const big = try gpa.alloc(u8, 2 * S * H);
    defer gpa.free(big);
    @memset(big, 0x38); // e4m3 1.0 codes, small bf16 values
    const ones = try gpa.alloc(f32, S * H / 64);
    defer gpa.free(ones);
    @memset(ones, 1);
    const scales = std.mem.sliceAsBytes(ones);
    const p = prefix ++ "layers.";
    const x = ".mlp.shared_expert.";
    const t = [_]core.shard_edit.New{
        // layer 0: shared gate | up FP8 beside a BF16 down that carries a stray companion
        .{ .name = p ++ "0" ++ x ++ "gate_proj.weight", .dtype = .f8_e4m3, .shape = &.{ S, H }, .bytes = big[0 .. S * H] },
        .{ .name = p ++ "0" ++ x ++ "gate_proj.weight_scale_inv", .dtype = .f32, .shape = &.{ S, H / 64 }, .bytes = scales },
        .{ .name = p ++ "0" ++ x ++ "up_proj.weight", .dtype = .f8_e4m3, .shape = &.{ S, H }, .bytes = big[0 .. S * H] },
        .{ .name = p ++ "0" ++ x ++ "up_proj.weight_scale_inv", .dtype = .f32, .shape = &.{ S, H / 64 }, .bytes = scales },
        .{ .name = p ++ "0" ++ x ++ "down_proj.weight", .dtype = .bf16, .shape = &.{ H, S }, .bytes = big },
        .{ .name = p ++ "0" ++ x ++ "down_proj.weight_scale_inv", .dtype = .f32, .shape = &.{ H, S / 64 }, .bytes = scales },
        // layer 1: an FP8 gate beside a BF16 up; layers 2, 3, 4: FP8 downs with a BF16, a short and no companion
        .{ .name = p ++ "1" ++ x ++ "gate_proj.weight", .dtype = .f8_e4m3, .shape = &.{ S, H }, .bytes = big[0 .. S * H] },
        .{ .name = p ++ "1" ++ x ++ "up_proj.weight", .dtype = .bf16, .shape = &.{ S, H }, .bytes = big },
        .{ .name = p ++ "2" ++ x ++ "down_proj.weight", .dtype = .f8_e4m3, .shape = &.{ H, S }, .bytes = big[0 .. S * H] },
        .{ .name = p ++ "2" ++ x ++ "down_proj.weight_scale_inv", .dtype = .bf16, .shape = &.{ H, S / 64 }, .bytes = scales[0 .. scales.len / 2] },
        .{ .name = p ++ "3" ++ x ++ "down_proj.weight", .dtype = .f8_e4m3, .shape = &.{ H, S }, .bytes = big[0 .. S * H] },
        .{ .name = p ++ "3" ++ x ++ "down_proj.weight_scale_inv", .dtype = .f32, .shape = &.{ H, S / 128 }, .bytes = scales[0 .. scales.len / 2] },
        .{ .name = p ++ "4" ++ x ++ "down_proj.weight", .dtype = .f8_e4m3, .shape = &.{ H, S }, .bytes = big[0 .. S * H] },
    };
    try source.writeCheckpoint(gpa, io, dir, &t);
    var src = try Source.open(gpa, io, dir);
    defer src.close();
    var L: Loader = .{ .src = &src, .st = undefined, .w = undefined, .d = undefined };
    const gate_up = [_][]const u8{ "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj" };
    const down = [_][]const u8{"mlp.shared_expert.down_proj"};
    try std.testing.expect(try groupFp8(&src, 0, &gate_up));
    try std.testing.expect(!try groupFp8(&src, 0, &down)); // groups of one layer take their own formats
    const gate = try L.fp8Part(0, gate_up[0], S, H);
    try std.testing.expect(gate.codes.len == S * H and gate.scales.len == scales.len);
    _ = try L.fp8Part(0, gate_up[1], S, H);
    _ = try src.get(t[4].name, .bf16, &.{ H, S });
    try std.testing.expectEqual(@as(usize, 5), src.used); // the stray companion stays unread: load refuses it
    try std.testing.expectError(error.MixedDramaFormats, groupFp8(&src, 1, &gate_up));
    try std.testing.expectError(error.MixedDramaFormats, groupFp8(&src, 1, &.{ gate_up[1], gate_up[0] })); // BF16 first
    for (2..5) |i| try std.testing.expect(try groupFp8(&src, i, &down));
    try std.testing.expect(!src.names.get(t[9].name).?.fits(.f32, &.{ H, S / 64 })); // get refuses a BF16 companion
    try std.testing.expect(!src.names.get(t[11].name).?.fits(.f32, &.{ H, S / 64 })); // and a short one
    try std.testing.expect(src.names.get(p ++ "4" ++ x ++ "down_proj.weight_scale_inv") == null); // MissingDramaTensor
}
