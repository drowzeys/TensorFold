//! drama-fp8-dense (e)'s source folder as a checkpoint lays it out: shards, their index and config.json (no GPU).
const std = @import("std");
const core = @import("core");
const drama = @import("drama");

/// One shard's tensors, written as `name` in the folder.
pub const Shard = struct { name: []const u8, tensors: []const core.shard_edit.New };

/// `shards` under `dir`, their index and config.json, whose digest tf-drama-quant records in its provenance.
pub fn write(gpa: std.mem.Allocator, a: std.mem.Allocator, io: std.Io, dir: []const u8, shards: []const Shard) !void {
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var index: std.Io.Writer.Allocating = .init(a);
    try index.writer.writeAll("{\"weight_map\": {");
    var sep: []const u8 = "";
    for (shards) |s| {
        try core.shard_edit.create(gpa, try std.fs.path.join(a, &.{ dir, s.name }), &.{}, s.tensors);
        for (s.tensors) |t| {
            try index.writer.print("{s}\"{s}\": \"{s}\"", .{ sep, t.name, s.name });
            sep = ", ";
        }
    }
    try index.writer.writeAll("}}");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), .data = index.written() });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = "{\"model_type\": \"fish_omni\"}\n" });
}

test "a folder written as (e) writes it converts through tf-drama-quant's library entry" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    const w = try a.alloc(u16, 128 * 64);
    for (w, 0..) |*v, i| v.* = @truncate(0x3C00 + i % 256); // small positive bf16
    const t = [_]core.shard_edit.New{.{ .name = "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", .dtype = .bf16, .shape = &.{ 128, 64 }, .bytes = std.mem.sliceAsBytes(w) }};
    const src = try std.fs.path.join(a, &.{ root, "src" });
    try write(gpa, a, io, src, &.{.{ .name = "layer-0.safetensors", .tensors = &t }});
    const dst = try std.fs.path.join(a, &.{ root, "q" });
    const s = try drama.quant_convert.convert(gpa, io, src, dst, .{ .experts = .keep, .dense_fp8 = "all" });
    try std.testing.expectEqual(@as(usize, 1), s.dense);
}
