//! tf-drama-quant's files: the shard names it admits, SRC's other files, and output that never replaces an entry.
const std = @import("std");

const Allocator = std.mem.Allocator;
const Io = std.Io;

/// An index shard name that can only name a file directly inside SRC (and DST): `NAME.safetensors`, no separators.
pub fn shardName(name: []const u8) bool {
    const ext = ".safetensors";
    return name.len > ext.len and std.mem.endsWith(u8, name, ext) and std.mem.indexOfAny(u8, name, "/\\") == null;
}

/// SRC's directories and non-safetensors files by relative path, parents first, the files DST generates excluded.
pub const Ancillary = struct { dirs: []const []const u8, files: []const []const u8 };

/// Walks SRC; a symlink anywhere is refused, since copying through it could reach outside SRC.
pub fn ancillary(gpa: Allocator, arena: Allocator, io: Io, src: []const u8, generated: []const []const u8) !Ancillary {
    var dir = try Io.Dir.cwd().openDir(io, src, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    var dirs: std.ArrayList([]const u8) = .empty;
    var files: std.ArrayList([]const u8) = .empty;
    while (try walker.next(io)) |e| switch (e.kind) {
        .directory => try dirs.append(arena, try arena.dupe(u8, e.path)),
        .file => {
            if (std.mem.endsWith(u8, e.basename, ".safetensors")) continue;
            for (generated) |g| {
                if (std.mem.eql(u8, e.path, g)) break;
            } else try files.append(arena, try arena.dupe(u8, e.path));
        },
        .sym_link => return error.SymlinkInSource,
        else => return error.UnsupportedSourceEntry,
    };
    return .{ .dirs = dirs.items, .files = files.items };
}

/// `a` from `src` into `out`, each file created anew (an existing entry is an error, never replaced).
pub fn copy(io: Io, src: Io.Dir, out: Io.Dir, a: Ancillary) !void {
    for (a.dirs) |d| try out.createDir(io, d, .default_dir);
    for (a.files) |f| try src.copyFile(f, out, f, io, .{ .replace = false });
}

/// A new JSON file in `dir` (never over an existing one).
pub fn emit(arena: Allocator, io: Io, dir: Io.Dir, name: []const u8, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(arena, value, .{ .whitespace = .indent_2 });
    try dir.writeFile(io, .{ .sub_path = name, .data = try std.mem.concat(arena, u8, &.{ text, "\n" }), .flags = .{ .exclusive = true } });
}

pub fn fileSha256(io: Io, path: []const u8) ![64]u8 {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var buffer: [1 << 16]u8 = undefined;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var offset: u64 = 0;
    while (true) {
        const got = try file.readPositionalAll(io, &buffer, offset);
        hash.update(buffer[0..got]);
        offset += got;
        if (got < buffer.len) break;
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

test "shard names stay inside their directory" {
    try std.testing.expect(shardName("model-00001-of-00008.safetensors"));
    for ([_][]const u8{ "", ".safetensors", "../x.safetensors", "/tmp/x.safetensors", "a/b.safetensors", "a\\b.safetensors", "model.bin" }) |bad| {
        try std.testing.expect(!shardName(bad));
    }
}
