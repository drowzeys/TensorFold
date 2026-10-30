// EmbeddingGemma-2's released text geometry, shared by the CUDA and Metal engines.
const std = @import("std");
const api = @import("engine_api");

pub const family: api.Family = .{ .model_type = "embedding_gemma2", .formats = &.{"bf16"} };

// config.json's text_config (checked at load)
pub const hidden = 512;
pub const inter = 2048;
pub const layers = 24;
pub const per_layer = 512; // hidden_size_per_layer_input
pub const heads = 4;
pub const vocab = 262144;
pub const dim = 768; // embedding_dim
pub const window = 512; // sliding_window: |q - k| <= 512 on the sliding layers
pub const eps: f32 = 1e-6;
pub const context = 8192;
/// Tokens a batch at most (every input's rows together).
pub const max_rows = 8192;
pub const max_inputs = 128;
pub const mrl = [_]u32{ 512, 256, 128 };
/// The token rows' scale: sqrt(512) in bf16.
pub const embed_scale: f32 = 22.625;

/// Layers 5, 11, 17 and 23 attend over the whole input with 512-wide heads and one kv head; the rest slide.
pub fn full(i: usize) bool {
    return i % 6 == 5;
}
pub fn headDim(i: usize) usize {
    return if (full(i)) 512 else 256;
}
pub fn kvHeads(i: usize) usize {
    return if (full(i)) 1 else 2;
}
pub fn qkvWidth(i: usize) usize {
    return (heads + 2 * kvHeads(i)) * headDim(i);
}
/// RoPE's base: 1e6 on the full layers, 1e4 on the sliding ones.
pub fn theta(i: usize) f32 {
    return if (full(i)) 1e6 else 1e4;
}

/// Layer `i`'s tensor `s` in the checkpoint.
pub fn key(b: []u8, i: usize, s: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "language_model.layers.{d}.{s}", .{ i, s });
}

pub fn bf16f(bytes: []const u8) f32 {
    const bits: u32 = @as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16;
    return @bitCast(bits);
}

/// config.json must be this geometry: the kernels and buffers are sized for it.
pub fn admit(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    try admitJson(a, bytes);
}

pub fn admitJson(a: std.mem.Allocator, bytes: []const u8) !void {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (v != .object) return error.EmbeddingGemma2Geometry;
    const tc = v.object.get("text_config") orelse return error.NoTextConfig;
    if (tc != .object) return error.EmbeddingGemma2Geometry;
    const t = tc.object;
    const want = .{ .{ "hidden_size", hidden }, .{ "intermediate_size", inter }, .{ "num_hidden_layers", layers }, .{ "num_attention_heads", heads }, .{ "num_key_value_heads", 2 }, .{ "head_dim", 256 }, .{ "vocab_size", vocab }, .{ "embedding_dim", dim }, .{ "sliding_window", window }, .{ "hidden_size_per_layer_input", per_layer } };
    inline for (want) |p| {
        const got = t.get(p[0]) orelse return error.EmbeddingGemma2Geometry;
        if (got != .integer or got.integer != p[1]) return error.EmbeddingGemma2Geometry;
    }
    const lt = t.get("layer_types") orelse return error.EmbeddingGemma2Geometry;
    if (lt != .array or lt.array.items.len != layers) return error.EmbeddingGemma2Geometry;
    for (lt.array.items, 0..) |ty, i| {
        if (ty != .string or !std.mem.eql(u8, ty.string, if (full(i)) "full_attention" else "sliding_attention")) return error.EmbeddingGemma2Geometry;
    }
}

// A packed batch's rows: position in the input and the keys read, [lo, hi), full and sliding (|q - k| <= 512).
pub fn rows(ends: []const u32, pos: []i32, full_lo: []i32, full_hi: []i32, slide_lo: []i32, slide_hi: []i32) void {
    var at: u32 = 0;
    for (ends) |end| {
        for (at..end) |r| {
            const ri: i32 = @intCast(r);
            pos[r] = @intCast(r - at);
            full_lo[r] = @intCast(at);
            full_hi[r] = @intCast(end);
            slide_lo[r] = @max(full_lo[r], ri - window);
            slide_hi[r] = @min(full_hi[r], ri + window + 1);
        }
        at = end;
    }
}

test "embedding_gemma2 geometry: layer kinds, packed rows' key ranges and the config gate" {
    try std.testing.expect(full(5) and full(23) and !full(0) and !full(22));
    try std.testing.expectEqual(@as(usize, 2048), qkvWidth(0));
    try std.testing.expectEqual(@as(usize, 3072), qkvWidth(5));
    var pos: [1100]i32 = undefined;
    var fl: [1100]i32 = undefined;
    var fh: [1100]i32 = undefined;
    var sl: [1100]i32 = undefined;
    var sh: [1100]i32 = undefined;
    rows(&.{ 3, 1100 }, &pos, &fl, &fh, &sl, &sh);
    try std.testing.expectEqual(@as(i32, 2), pos[2]);
    try std.testing.expectEqual(@as(i32, 0), pos[3]);
    try std.testing.expectEqual(@as(i32, 3), fl[3]);
    try std.testing.expectEqual(@as(i32, 1100), fh[3]);
    try std.testing.expectEqual(@as(i32, 3), sl[3]);
    try std.testing.expectEqual(@as(i32, 3 + 513), sh[3]); // |q - k| <= 512
    try std.testing.expectEqual(@as(i32, 1099 - 512), sl[1099]);
    try std.testing.expectEqual(@as(i32, 1100), sh[1099]);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.NoTextConfig, admitJson(arena.allocator(), "{}"));
    try std.testing.expectError(error.EmbeddingGemma2Geometry, admitJson(arena.allocator(), "{\"text_config\":{\"hidden_size\":640}}"));
}
