//! One NVFP4 routed expert of a Drama checkpoint as C1 stores it (ModelOpt names), read and content-checked.
const std = @import("std");
const cfg = @import("backbone_config.zig");
const source = @import("backbone_source.zig");
const prefix = @import("backbone_weights.zig").prefix;

const Source = source.Source;

pub const projections = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
pub const shapes = [3][2]usize{ .{ cfg.expert_inner, cfg.hidden }, .{ cfg.expert_inner, cfg.hidden }, .{ cfg.hidden, cfg.expert_inner } };
pub const parts = shapes.len;

/// One NVFP4 expert's stored gate, up and down: codes [n][k / 2], e4m3 scales [n][k / 16], fp32 globals.
pub const Expert = struct { codes: [parts][]const u8, scales: [parts][]const u8, global: [parts]f32 };

/// Expert e of NVFP4 layer i, refused unless all nine tensors have C1's dtypes and shapes and content checks pass.
pub fn read(src: *Source, i: usize, e: usize) !Expert {
    var buf: [192]u8 = undefined;
    var x: Expert = undefined;
    for (projections, shapes, 0..) |p, shape, j| {
        const n, const k = shape;
        x.codes[j] = try src.get(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}.weight", .{ i, e, p }), .u8, &.{ n, k / 2 });
        const scales = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}.weight_scale", .{ i, e, p });
        x.scales[j] = try src.get(scales, .f8_e4m3, &.{ n, k / 16 });
        checkScales(x.scales[j]) catch |err| {
            std.log.err("drama backbone: {s} holds a NaN or negative e4m3 scale", .{scales});
            return err;
        };
        const global = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}.weight_scale_2", .{ i, e, p });
        x.global[j] = globalOf(try src.get(global, .f32, &.{})) catch |err| {
            std.log.err("drama backbone: {s} is not a positive finite global scale", .{global});
            return err;
        };
    }
    return x;
}

/// The decode and prompt kernel forms read e4m3 NaN codes (0x7F, 0xFF) differently; negative scales are not C1's.
fn checkScales(bytes: []const u8) !void {
    for (bytes) |b| if (b == 0x7F or b > 0x80) return error.BadDramaNvfp4Scale;
}

fn globalOf(bytes: []const u8) !f32 {
    const g: f32 = @bitCast(std.mem.readInt(u32, bytes[0..4], .little));
    if (!(g > 0 and std.math.isFinite(g))) return error.BadDramaNvfp4Global;
    return g;
}

test "an NVFP4 expert reads as C1 stores it; get refuses other dtypes, shapes and formats, load counts leftovers" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const big = try gpa.alloc(u8, cfg.expert_inner * cfg.hidden);
    defer gpa.free(big);
    @memset(big, 0x38); // e4m3 1.0 as every scale; the codes' values do not matter here
    const g: f32 = 1.5;
    const e0 = prefix ++ "layers.0.mlp.experts.0.";
    const e1 = prefix ++ "layers.0.mlp.experts.1.";
    var t: [parts * 3 + 4]@import("core").shard_edit.New = undefined;
    inline for (projections, shapes, 0..) |p, shape, j| { // comptime shapes: the slices outlive the loop
        const n, const k = shape;
        t[3 * j] = .{ .name = e0 ++ p ++ ".weight", .dtype = .u8, .shape = &.{ n, k / 2 }, .bytes = big[0 .. n * k / 2] };
        t[3 * j + 1] = .{ .name = e0 ++ p ++ ".weight_scale", .dtype = .f8_e4m3, .shape = &.{ n, k / 16 }, .bytes = big[0 .. n * k / 16] };
        t[3 * j + 2] = .{ .name = e0 ++ p ++ ".weight_scale_2", .dtype = .f32, .shape = &.{}, .bytes = std.mem.asBytes(&g) };
    }
    // expert 1: an FP8 gate, fp32 up scales, a transposed down and no down global; expert 0 keeps an FP8 scale_inv
    t[9] = .{ .name = e1 ++ "gate_proj.weight", .dtype = .f8_e4m3, .shape = &.{ cfg.expert_inner, cfg.hidden }, .bytes = big };
    t[10] = .{ .name = e1 ++ "up_proj.weight_scale", .dtype = .f32, .shape = &.{ cfg.expert_inner, 32 }, .bytes = big[0 .. cfg.expert_inner * 128] };
    t[11] = .{ .name = e1 ++ "down_proj.weight", .dtype = .u8, .shape = &.{ cfg.expert_inner, cfg.hidden / 2 }, .bytes = big[0 .. cfg.expert_inner * cfg.hidden / 2] };
    t[12] = .{ .name = e0 ++ "gate_proj.weight_scale_inv", .dtype = .f32, .shape = &.{ 4, 16 }, .bytes = big[0..256] };
    try source.writeCheckpoint(gpa, io, dir, &t);
    var src = try Source.open(gpa, io, dir);
    defer src.close();
    const x = try read(&src, 0, 0);
    try std.testing.expectEqual([parts]f32{ 1.5, 1.5, 1.5 }, x.global);
    try std.testing.expectEqual(@as(usize, 512 * 1024), x.codes[2].len);
    try std.testing.expectEqual(src.backbone - 4, src.used); // the unread four: load refuses with UnusedDramaTensors
    const at = struct {
        fn entry(s: *const Source, name: []const u8) source.Entry {
            return s.names.get(name).?;
        }
    }.entry;
    try std.testing.expect(!at(&src, t[9].name).fits(.u8, &.{ cfg.expert_inner, cfg.hidden / 2 })); // FP8 beside NVFP4
    try std.testing.expect(!at(&src, t[0].name).fits(.f8_e4m3, &.{ cfg.expert_inner, cfg.hidden })); // and NVFP4 beside FP8
    try std.testing.expect(!at(&src, t[10].name).fits(.f8_e4m3, &.{ cfg.expert_inner, cfg.hidden / 16 }));
    try std.testing.expect(!at(&src, t[11].name).fits(.u8, &.{ cfg.hidden, cfg.expert_inner / 2 }));
    try std.testing.expect(!at(&src, t[2].name).fits(.f32, &.{1}));
    try std.testing.expect(src.names.get(e1 ++ "down_proj.weight_scale_2") == null); // get: MissingDramaTensor
}

test "NVFP4 content checks: 0x7F and 0xFF and negative scale codes, non-positive or non-finite globals" {
    try checkScales(&.{ 0x00, 0x01, 0x07, 0x38, 0x7E, 0x80 });
    for ([_]u8{ 0x7F, 0xFF, 0x81, 0xB8 }) |b| try std.testing.expectError(error.BadDramaNvfp4Scale, checkScales(&.{ 0x38, b }));
    for ([_]f32{ 0x1p-140, 1e-3, 3.0e38 }) |g| _ = try globalOf(std.mem.asBytes(&g));
    for ([_]f32{ 0, -0.0, -1, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |g| try std.testing.expectError(error.BadDramaNvfp4Global, globalOf(std.mem.asBytes(&g)));
}
