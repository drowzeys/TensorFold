//! The backbone's dense groups stored FP8 (C5): e4m3 codes [n, k] and fp32 scales a row and 64 inputs, for the lane.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("backbone_config.zig");
const Stager = @import("stager.zig").Stager;
const tower_fp8 = @import("tower_fp8.zig");

const H = cfg.hidden;
const group = cuda.fp8.group;

/// A layer's dense groups stored FP8, each a lane weight; a null group keeps its bf16 rows in the Layer.
pub const Groups = struct {
    gdn_in: ?cuda.qlinear.Weight = null, // in_proj_qkv | in_proj_z; in_proj_b | in_proj_a are then Layer.gdn_ba
    gdn_out: ?cuda.qlinear.Weight = null, // out_proj
    attn_in: ?cuda.qlinear.Weight = null, // q_proj | k_proj | v_proj
    attn_out: ?cuda.qlinear.Weight = null, // o_proj
    shared_in: ?cuda.qlinear.Weight = null, // shared gate_proj | up_proj; mlp.gate is then Layer.router
    shared_out: ?cuda.qlinear.Weight = null, // shared down_proj
};

pub const group_names = @typeInfo(Groups).@"struct".field_names;

/// Layers storing any group FP8, and per group (Groups' order) the layers storing it so.
pub const Census = struct {
    layers: usize = 0,
    groups: [group_names.len]usize = @splat(0),

    /// `N of 40 backbone layers (gdn_in a, ...)`: the backbone's part of the startup line's dense FP8 count.
    pub fn write(c: Census, w: *std.Io.Writer, layers: usize) !void {
        try w.print("{d} of {d} backbone layers (", .{ c.layers, layers });
        inline for (group_names, 0..) |g, j| try w.print("{s}{s} {d}", .{ if (j == 0) "" else ", ", g, c.groups[j] });
        try w.writeAll(")");
    }
};

/// The startup line's dense FP8 count (C8), empty when neither the backbone nor the tower stores a projection FP8.
pub fn note(buf: []u8, backbone: Census, backbone_layers: usize, tower: tower_fp8.Census, tower_layers: usize) ![]const u8 {
    if (backbone.layers == 0 and tower.layers == 0) return "";
    var w: std.Io.Writer = .fixed(buf);
    try w.writeAll("; dense FP8 in ");
    try backbone.write(&w, backbone_layers);
    try w.writeAll(", ");
    try tower.write(&w, tower_layers);
    return w.buffered();
}

/// One stored FP8 projection: codes [n, k] and row scales [n, k / 64], as mapped.
pub const Part = struct { codes: []const u8, scales: []const u8 };

/// The lane reads the e4m3 NaN codes 0x7F and 0xFF as ±480, the host decoder as NaN: C5 stores neither.
pub fn checkCodes(bytes: []const u8) !void {
    for (bytes) |b| if (b & 0x7f == 0x7f) return error.BadDramaFp8Code;
}

/// C5's scales are positive and finite; positive f32 subnormals are admitted.
pub fn checkScales(bytes: []const u8) !void {
    var i: usize = 0;
    while (i < bytes.len) : (i += 4) {
        const v: f32 = @bitCast(std.mem.readInt(u32, bytes[i..][0..4], .little));
        if (!(v > 0 and std.math.isFinite(v))) return error.BadDramaFp8Scale;
    }
}

/// Row scales as stored (f32 [n][k / 64], any alignment) in the lane's tiles; the caller frees them.
pub fn tiles(gpa: std.mem.Allocator, scales: []const u8, n: usize, k: usize) ![]f32 {
    const inv = try gpa.alloc(f32, scales.len / 4);
    defer gpa.free(inv);
    @memcpy(std.mem.sliceAsBytes(inv), scales);
    const out = try gpa.alloc(f32, cuda.fp8.padded(n) * (k / group));
    cuda.fp8.packRowScales(out, inv, n, k);
    return out;
}

/// Codes of every group one layer may store FP8: a DeltaNet layer's or an attention layer's, and the shared expert's.
const room_bytes = @max((cfg.conv_dim + 2 * cfg.value_dim) * H, (2 * cfg.query_heads + 2 * cfg.kv_heads + cfg.query_heads) * cfg.head_dim * H) + 3 * cfg.shared_inner * H;
/// Parts of one layer's groups: q, k, v, o and the shared gate, up and down at most.
const max_parts = 7;

/// One layer's FP8 codes staged as stored, packed once landed; its own packer: dense FP8 needs no routed NVFP4 layer.
pub const Stage = struct {
    room: cuda.DeviceBuffer,
    packer: cuda.modelopt.Packer,
    used: usize = 0,
    packs: [max_parts]Pack = undefined,
    count: usize = 0,

    const Pack = struct { at: usize, dst: u64, n: usize, k: usize };

    pub fn init(d: *const cuda.Driver) !Stage {
        var packer = try cuda.modelopt.Packer.load(d);
        errdefer packer.unload();
        return .{ .room = try cuda.DeviceBuffer.alloc(d, room_bytes), .packer = packer };
    }

    pub fn deinit(s: *Stage) void {
        s.room.free();
        s.packer.unload();
    }

    /// `codes` ([n, k] as stored) for `dst` in the lane's order: copied at the stager's next finish, packed by `pack`.
    pub fn add(s: *Stage, st: *Stager, codes: []const u8, dst: u64, n: usize, k: usize) !void {
        try st.put(s.room.ptr + s.used, codes);
        s.packs[s.count] = .{ .at = s.used, .dst = dst, .n = n, .k = k };
        s.used += codes.len;
        s.count += 1;
    }

    /// Lands every staged code, then queues the packs; the next layer's copies queue after them on the same stream.
    pub fn pack(s: *Stage, st: *Stager) !void {
        if (s.count == 0) return;
        try st.finish();
        for (s.packs[0..s.count]) |p| try s.packer.fp8Lane(st.stream, s.room.ptr + p.at, p.dst, p.n, p.k);
        s.used = 0;
        s.count = 0;
    }
};

test "C5 content checks: NaN codes, non-positive or non-finite scales; subnormal scales admitted" {
    try checkCodes(&.{ 0x00, 0x01, 0x07, 0x38, 0x7E, 0x80, 0x87, 0xFE });
    for ([_]u8{ 0x7F, 0xFF }) |b| try std.testing.expectError(error.BadDramaFp8Code, checkCodes(&.{ 0x38, b }));
    const good = [_]f32{ 0x1p-149, 2.05e-43, 0x1p-126, 1, 256, 3.0e38 };
    try checkScales(std.mem.sliceAsBytes(&good));
    for ([_]f32{ 0, -0.0, -0x1p-149, -1, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |v| {
        const row = [_]f32{ 1, v };
        try std.testing.expectError(error.BadDramaFp8Scale, checkScales(std.mem.sliceAsBytes(&row)));
    }
}

test "stored row scales at any alignment become the lane's tiles" {
    const gpa = std.testing.allocator;
    const n = 128;
    const k = 128;
    var bytes: [1 + n * (k / group) * 4]u8 = undefined;
    for (0..n * (k / group)) |i| std.mem.writeInt(u32, bytes[1 + 4 * i ..][0..4], @bitCast(@as(f32, @floatFromInt(i + 1))), .little);
    const t = try tiles(gpa, bytes[1..], n, k);
    defer gpa.free(t);
    try std.testing.expectEqual(@as(f32, 2 * 70 + 1 + 1), t[(1 * 2 + 1) * 64 + 6]); // tile 1, group 1, row 64 + 6
}

test "the startup line counts the backbone's and the tower's dense FP8 layers and classes together" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("", try note(&buf, .{}, 40, .{}, 13));
    const both = "; dense FP8 in 40 of 40 backbone layers (gdn_in 30, gdn_out 30, attn_in 10, attn_out 10, shared_in 40, shared_out 39), 13 of 13 tower layers (tower_qkv 13, tower_o 12, tower_gate_up 13, tower_down 13)";
    try std.testing.expectEqualStrings(both, try note(&buf, .{ .layers = 40, .groups = .{ 30, 30, 10, 10, 40, 39 } }, 40, .{ .layers = 13, .kinds = .{ 13, 12, 13, 13 } }, 13));
    const tower_only = "; dense FP8 in 0 of 40 backbone layers (gdn_in 0, gdn_out 0, attn_in 0, attn_out 0, shared_in 0, shared_out 0), 2 of 13 tower layers (tower_qkv 0, tower_o 0, tower_gate_up 2, tower_down 1)";
    try std.testing.expectEqualStrings(tower_only, try note(&buf, .{}, 40, .{ .layers = 2, .kinds = .{ 0, 0, 2, 1 } }, 13));
}
