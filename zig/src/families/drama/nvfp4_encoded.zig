//! tf-drama-quant --encoded: layers another encoder (the GPTQ tool) wrote as C1 tensors, checked before any is taken.
const std = @import("std");
const core = @import("core");

const st = core.safetensors;
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const projections = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };

/// A checked `layer-LL.safetensors`: its path, SHA-256 and `__metadata__` as written (null if it has none).
pub const Layer = struct { path: []const u8, sha256: [64]u8, metadata: ?std.json.Value };

/// Source expert weight shapes [n, k] by full tensor name.
pub const Shapes = std.StringHashMapUnmanaged([2]usize);

/// The layers `dir` holds a file for; a `layer-*.safetensors` not named `layer-LL.safetensors` is refused.
pub fn list(arena: Allocator, io: Io, dir: []const u8) ![]const usize {
    var d = try Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var layers: std.ArrayList(usize) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (!std.mem.startsWith(u8, e.name, "layer-") or !std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const l = std.fmt.parseInt(usize, e.name["layer-".len .. e.name.len - ".safetensors".len], 10) catch return error.BadEncodedName;
        if (!std.mem.eql(u8, e.name, try fileName(arena, l))) return error.BadEncodedName;
        try layers.append(arena, l);
    }
    std.mem.sort(usize, layers.items, {}, std.sort.asc(usize));
    return layers.items;
}

pub fn fileName(arena: Allocator, layer: usize) ![]const u8 {
    return std.fmt.allocPrint(arena, "layer-{d:0>2}.safetensors", .{layer});
}

/// `dir`'s file of layer `l`: exactly C1's tensors of `experts` experts at the source's shapes; scales, globals valid.
pub fn check(gpa: Allocator, arena: Allocator, io: Io, dir: []const u8, l: usize, experts: usize, source: *const Shapes) !Layer {
    const path = try std.fs.path.join(arena, &.{ dir, try fileName(arena, l) });
    var file = try st.File.open(gpa, io, path);
    defer file.close(io);
    if (file.names.count() != experts * projections.len * 3) return error.EncodedTensorCount;
    for (0..experts) |e| for (projections) |p| {
        const base = try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.", .{ l, e, p });
        const n, const k = source.get(try std.fmt.allocPrint(arena, "{s}weight", .{base})) orelse return error.EncodedTensorMismatch;
        const codes = file.get(try std.fmt.allocPrint(arena, "{s}weight", .{base})) orelse return error.EncodedTensorMismatch;
        const scales = file.get(try std.fmt.allocPrint(arena, "{s}weight_scale", .{base})) orelse return error.EncodedTensorMismatch;
        const global = file.get(try std.fmt.allocPrint(arena, "{s}weight_scale_2", .{base})) orelse return error.EncodedTensorMismatch;
        if (!codes.is(.u8, &.{ n, k / 2 }) or !scales.is(.f8_e4m3, &.{ n, k / 16 }) or !global.is(.f32, &.{})) return error.EncodedTensorMismatch;
        for (scales.bytes) |b| if (b >= 0x7F) return error.BadEncodedScale; // 0x7F is NaN, 0x80 and up negative
        const g: f32 = @bitCast(std.mem.readInt(u32, global.bytes[0..4], .little));
        if (!std.math.isFinite(g) or g <= 0) return error.BadEncodedGlobal;
    };
    const header = try std.json.parseFromSliceLeaky(std.json.Value, arena, file.map.memory[8..file.data], .{ .allocate = .alloc_always });
    return .{ .path = path, .sha256 = sha256(file.map.memory), .metadata = header.object.get("__metadata__") };
}

fn sha256(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}
