//! tf-drama-quant's refusals: paths leaving SRC, corrupt statistics, unknown companions, symlinks, failed runs.
const std = @import("std");
const core = @import("core");
const convert = @import("quant_convert.zig");
const t = @import("quant_convert_test.zig");

const shard_edit = core.shard_edit;
const testing = std.testing;

const weight = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";

/// Neither DST nor its work directory exists.
fn nothingAt(f: *t.Fixture, dst: []const u8) !void {
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(testing.io, dst, .{}));
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(testing.io, try std.fmt.allocPrint(f.arena_state.allocator(), "{s}.partial", .{dst}), .{}));
}

fn sha(f: *t.Fixture, sub: []const u8) ![32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(try f.read(sub), &out, .{});
    return out;
}

test "index shard paths that are not plain names in SRC are refused before DST exists, their targets untouched" {
    var f = try t.Fixture.init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const io = testing.io;
    try t.single(&f, "trav", 0x38, 1, &.{});
    try f.tmp.dir.createDir(io, "outside", .default_dir);
    try f.tmp.dir.writeFile(io, .{ .sub_path = "outside/one.safetensors", .data = try f.read(try std.fs.path.join(arena, &.{ "trav", t.shard_names[0] })) });
    const before = try sha(&f, "outside/one.safetensors");
    const cases = [_]struct { []const u8, anyerror }{
        .{ "\"../outside/one.safetensors\"", error.BadShardName },
        .{ try std.fmt.allocPrint(arena, "\"{s}\"", .{try f.path("outside/one.safetensors")}), error.BadShardName },
        .{ "\"\"", error.BadShardName },
        .{ "\".safetensors\"", error.BadShardName },
        .{ "\"model.bin\"", error.BadShardName },
        .{ "\"sub/one.safetensors\"", error.BadShardName },
        .{ "7", error.BadIndex },
    };
    for (cases) |c| {
        const index = try std.fmt.allocPrint(arena, "{{\"weight_map\": {{\"{s}\": {s}, \"{s}_scale_inv\": {s}}}}}", .{ weight, c[0], weight, c[0] });
        try f.tmp.dir.writeFile(io, .{ .sub_path = "trav/model.safetensors.index.json", .data = index });
        try testing.expectError(c[1], convert.convert(testing.allocator, io, try f.path("trav"), try f.path("out"), .{}));
        try nothingAt(&f, "out");
    }
    try testing.expectEqual(before, try sha(&f, "outside/one.safetensors"));
}

/// Statistics for `single`'s one-expert layer 0: each class's count `n`, gate/up sums `gate`, down sums `down`.
fn stats(f: *t.Fixture, name: []const u8, n: f64, gate: f64, down: f64) ![]const u8 {
    const arena = f.arena_state.allocator();
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    for ([_][]const u8{ "prompt_text", "prompt_audio", "generated" }) |c| {
        const count = try arena.dupe(f64, &.{n});
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "layers.0.count.{s}", .{c}), .dtype = .f64, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(count) });
        for ([_][]const u8{ "gate_up", "down" }, [_]f64{ gate, down }) |p, v| {
            const sums = try arena.alloc(f64, 32);
            @memset(sums, v);
            try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "layers.0.{s}.sumsq.{s}", .{ p, c }), .dtype = .f64, .shape = &.{ 1, 32 }, .bytes = std.mem.sliceAsBytes(sums) });
        }
    }
    const path = try f.path(name);
    try shard_edit.create(arena, path, &.{}, tensors.items);
    return path;
}

test "a non-finite or negative count or sum refuses with no output, even where no pair was observed" {
    var f = try t.Fixture.init();
    defer f.deinit();
    const io = testing.io;
    const inf = std.math.inf(f64);
    const nan = std.math.nan(f64);
    try t.single(&f, "one", 0x38, 1, &.{});
    const bad = [_][3]f64{ .{ nan, 1, 1 }, .{ -1, 1, 1 }, .{ inf, 1, 1 }, .{ 1, inf, 1 }, .{ 1, 1, -1 }, .{ 0, nan, 1 }, .{ 0, 1, -0.5 } };
    for (bad, 0..) |v, i| {
        const path = try stats(&f, try std.fmt.allocPrint(f.arena_state.allocator(), "bad{d}.safetensors", .{i}), v[0], v[1], v[2]);
        try testing.expectError(error.BadStats, convert.convert(testing.allocator, io, try f.path("one"), try f.path("out"), .{ .stats = path }));
        try nothingAt(&f, "out");
    }
    // a valid zero count is the one way to the no-observation prior
    _ = try convert.convert(testing.allocator, io, try f.path("one"), try f.path("zero"), .{ .stats = try stats(&f, "zero.safetensors", 0, 0, 0) });
    _ = try convert.convert(testing.allocator, io, try f.path("one"), try f.path("seen"), .{ .stats = try stats(&f, "seen.safetensors", 2, 1, 1) });
}

test "an unknown companion of a converted projection is refused before writing; a kept layer keeps it" {
    var f = try t.Fixture.init();
    defer f.deinit();
    const io = testing.io;
    const bias = try f.arena_state.allocator().alloc(f32, 32);
    @memset(bias, 0.5);
    try t.single(&f, "bias", 0x38, 1, &.{.{ .name = "model.language_model.layers.0.mlp.experts.0.gate_proj.bias", .dtype = .f32, .shape = &.{32}, .bytes = std.mem.sliceAsBytes(bias) }});
    try testing.expectError(error.UnknownExpertCompanion, convert.convert(testing.allocator, io, try f.path("bias"), try f.path("out"), .{}));
    try nothingAt(&f, "out");
    _ = try convert.convert(testing.allocator, io, try f.path("bias"), try f.path("kept"), .{ .keep = &.{0} });
    var kept = try f.open("kept", 0);
    defer kept.close(io);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(bias), kept.get("model.language_model.layers.0.mlp.experts.0.gate_proj.bias").?.bytes);
}

test "a symlink in SRC is refused before writing" {
    var f = try t.Fixture.init();
    defer f.deinit();
    try f.tmp.dir.symLink(testing.io, "config.json", "src/link.json", .{});
    try testing.expectError(error.SymlinkInSource, convert.convert(testing.allocator, testing.io, try f.path("src"), try f.path("out"), .{}));
    try nothingAt(&f, "out");
}

test "a run that fails mid-way leaves neither DST nor DST.partial, and an existing DST.partial is refused" {
    var f = try t.Fixture.init();
    defer f.deinit();
    const io = testing.io;
    try t.single(&f, "nan", 0x7F, 1, &.{}); // refused while its shard encodes, after DST.partial exists
    try testing.expectError(error.NonFiniteWeight, convert.convert(testing.allocator, io, try f.path("nan"), try f.path("out"), .{}));
    try nothingAt(&f, "out");
    try f.tmp.dir.createDir(io, "busy.partial", .default_dir);
    try testing.expectError(error.PartialExists, convert.convert(testing.allocator, io, try f.path("src"), try f.path("busy"), .{}));
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(io, "busy", .{}));
}
