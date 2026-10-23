//! A Drama checkpoint folder's safetensors shards mapped read-only: every header entry by name, read only as declared.
const std = @import("std");
const prefix = @import("backbone_weights.zig").prefix;
const shard_edit = @import("core").shard_edit;

const Io = std.Io;

pub const DType = enum { bf16, f32, f8_e4m3, u8 };

fn parseDType(s: []const u8) ?DType {
    if (std.mem.eql(u8, s, "BF16")) return .bf16;
    if (std.mem.eql(u8, s, "F32")) return .f32;
    if (std.mem.eql(u8, s, "F8_E4M3")) return .f8_e4m3;
    if (std.mem.eql(u8, s, "U8")) return .u8;
    return null;
}

fn size(t: DType) usize {
    return switch (t) {
        .bf16 => 2,
        .f32 => 4,
        .f8_e4m3, .u8 => 1,
    };
}

pub const Entry = struct {
    shard: usize,
    dtype: ?DType,
    rank: usize,
    shape: [3]usize,
    begin: usize,
    end: usize,

    /// Whether the entry is exactly `dtype` at `shape`, its byte count included.
    pub fn fits(e: Entry, dtype: DType, shape: []const usize) bool {
        var ok = e.dtype != null and e.dtype.? == dtype and e.rank == shape.len;
        var n: usize = size(dtype);
        for (shape, 0..) |d, i| {
            ok = ok and e.shape[i] == d;
            n *= d;
        }
        return ok and e.end - e.begin == n;
    }
};

const Shard = struct { path: [:0]const u8, file: Io.File, map: Io.File.MemoryMap, data: usize };

/// The checkpoint's shards mapped read-only, every header entry indexed by name.
pub const Source = struct {
    gpa: std.mem.Allocator,
    io: Io,
    arena: std.heap.ArenaAllocator,
    shards: std.ArrayList(Shard) = .empty,
    names: std.StringHashMapUnmanaged(Entry) = .empty,
    backbone: usize = 0, // entries under `prefix`
    used: usize = 0, // of those, consumed
    /// The index's `metadata` object (a compact checkpoint's maps); empty when it has none.
    metadata: std.json.ObjectMap = .empty,

    pub fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8) !Source {
        var s: Source = .{ .gpa = gpa, .io = io, .arena = .init(gpa) };
        errdefer s.close();
        const a = s.arena.allocator();
        const index = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = try Io.Dir.cwd().readFileAlloc(io, index, a, .limited(1 << 26));
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        if (doc != .object) return error.BadDramaCheckpoint;
        const map = doc.object.get("weight_map") orelse return error.BadDramaCheckpoint;
        if (map != .object) return error.BadDramaCheckpoint;
        if (doc.object.get("metadata")) |m| s.metadata = if (m == .object) m.object else return error.BadDramaCheckpoint;
        var files: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = map.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.BadDramaCheckpoint;
            try files.put(a, e.value_ptr.string, {});
        }
        for (files.keys(), 0..) |name, i| {
            const path = try std.fs.path.joinZ(a, &.{ dir, name });
            try s.mapShard(path, i);
        }
        // every name the index lists must be in its shard's header
        it = map.object.iterator();
        while (it.next()) |e| {
            const got = s.names.get(e.key_ptr.*) orelse return error.BadDramaCheckpoint;
            if (!std.mem.eql(u8, files.keys()[got.shard], e.value_ptr.string)) return error.BadDramaCheckpoint;
        }
        return s;
    }

    fn mapShard(s: *Source, path: [:0]const u8, index: usize) !void {
        const a = s.arena.allocator();
        // the file and its map belong to `shards` once appended; the errdefers cover only the steps before
        const sh = blk: {
            var file = try Io.Dir.cwd().openFile(s.io, path, .{});
            errdefer file.close(s.io);
            const len: usize = @intCast(try file.length(s.io));
            if (len < 8) return error.BadDramaCheckpoint;
            var map = try Io.File.MemoryMap.create(s.io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
            errdefer map.destroy(s.io);
            const header_len: usize = @intCast(std.mem.readInt(u64, map.memory[0..8], .little));
            if (header_len > len - 8) return error.BadDramaCheckpoint;
            try s.shards.append(s.gpa, .{ .path = path, .file = file, .map = map, .data = 8 + header_len });
            break :blk s.shards.items[s.shards.items.len - 1];
        };
        const data_len = sh.map.memory.len - sh.data;
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, sh.map.memory[8..sh.data], .{ .allocate = .alloc_always });
        if (doc != .object) return error.BadDramaCheckpoint;
        var it = doc.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const o = if (kv.value_ptr.* == .object) kv.value_ptr.object else return error.BadDramaCheckpoint;
            const dt = o.get("dtype") orelse return error.BadDramaCheckpoint;
            const shape = o.get("shape") orelse return error.BadDramaCheckpoint;
            const offs = o.get("data_offsets") orelse return error.BadDramaCheckpoint;
            if (dt != .string or shape != .array or offs != .array or offs.array.items.len != 2) return error.BadDramaCheckpoint;
            var e: Entry = .{ .shard = index, .dtype = parseDType(dt.string), .rank = shape.array.items.len, .shape = @splat(1), .begin = 0, .end = 0 };
            if (e.rank > 3) e.dtype = null; // not a backbone layout; refused if anything asks for it
            for (shape.array.items, 0..) |d, i| {
                if (d != .integer or d.integer < 0) return error.BadDramaCheckpoint;
                if (i < 3) e.shape[i] = @intCast(d.integer);
            }
            for (offs.array.items, 0..) |d, i| {
                if (d != .integer or d.integer < 0) return error.BadDramaCheckpoint;
                if (i == 0) e.begin = @intCast(d.integer) else e.end = @intCast(d.integer);
            }
            if (e.end < e.begin or e.end > data_len) return error.BadDramaCheckpoint;
            if (std.mem.startsWith(u8, kv.key_ptr.*, prefix)) s.backbone += 1;
            const got = try s.names.getOrPut(s.gpa, kv.key_ptr.*);
            if (got.found_existing) return error.BadDramaCheckpoint;
            got.value_ptr.* = e;
        }
    }

    pub fn close(s: *Source) void {
        for (s.shards.items) |*sh| {
            sh.map.destroy(s.io);
            sh.file.close(s.io);
        }
        s.shards.deinit(s.gpa);
        s.names.deinit(s.gpa);
        s.arena.deinit();
        s.* = undefined;
    }

    /// The bytes of `name`, refused unless its dtype and shape are exactly these.
    pub fn get(s: *Source, name: []const u8, dtype: DType, shape: []const usize) ![]const u8 {
        const e = s.names.get(name) orelse {
            std.log.err("drama backbone: checkpoint has no {s}", .{name});
            return error.MissingDramaTensor;
        };
        if (!e.fits(dtype, shape)) {
            std.log.err("drama backbone: {s} is not {s} {any}", .{ name, @tagName(dtype), shape });
            return error.UnexpectedDramaTensor;
        }
        if (std.mem.startsWith(u8, name, prefix)) s.used += 1;
        const sh = s.shards.items[e.shard];
        return sh.map.memory[sh.data + e.begin .. sh.data + e.end];
    }
};

/// A one-shard checkpoint at `dir` holding `tensors`, with the index Source.open reads (for tests).
pub fn writeCheckpoint(gpa: std.mem.Allocator, io: Io, dir: []const u8, tensors: []const shard_edit.New) !void {
    const shard = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(shard);
    try shard_edit.create(gpa, shard, &.{}, tensors);
    var index: std.Io.Writer.Allocating = .init(gpa);
    defer index.deinit();
    try index.writer.writeAll("{\"weight_map\": {");
    for (tensors, 0..) |t, i| try index.writer.print("{s}\"{s}\": \"model.safetensors\"", .{ if (i == 0) "" else ", ", t.name });
    try index.writer.writeAll("}}");
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(path);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = index.written() });
}
