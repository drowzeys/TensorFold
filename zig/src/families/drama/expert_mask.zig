//! --expert-mask: the routed experts each backbone layer may pick, {"layer": [kept ids]}; unlisted layers keep all.
const std = @import("std");
const config = @import("backbone_config.zig");
const Allocator = std.mem.Allocator;

pub const words = config.experts / 32;

/// Per layer, bit e of word e / 32 set: expert e is eligible. Uploaded as is for drama_topk.
pub const Mask = [config.layers][words]u32;

/// The mask in the file at `path`, or null with `problem` set to why it is refused.
pub fn read(a: Allocator, io: std.Io, path: []const u8, problem: *[]const u8) Allocator.Error!?Mask {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "--expert-mask {s}: cannot read it ({s})", .{ path, @errorName(e) });
        return null;
    };
    var why: []const u8 = "";
    const mask = try parse(a, text, &why) orelse {
        problem.* = try std.fmt.allocPrint(a, "--expert-mask {s}: {s}", .{ path, why });
        return null;
    };
    return mask;
}

/// The mask a file's text names, or null with `problem` set to why it is refused.
pub fn parse(a: Allocator, text: []const u8, problem: *[]const u8) Allocator.Error!?Mask {
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "not a JSON object of layers ({s})", .{@errorName(e)});
        return null;
    };
    if (root != .object) return fail(a, problem, "not a JSON object of layers", .{});
    var mask: Mask = @splat(@splat(std.math.maxInt(u32)));
    for (root.object.keys(), root.object.values()) |key, ids| {
        const layer = std.fmt.parseInt(u8, key, 10) catch return fail(a, problem, "layer \"{s}\" is not a layer index", .{key});
        if (layer >= config.layers or !std.mem.eql(u8, key, try std.fmt.allocPrint(a, "{d}", .{layer})))
            return fail(a, problem, "layer \"{s}\" is not a layer index from 0 to {d}", .{ key, config.layers - 1 });
        if (ids != .array) return fail(a, problem, "layer {d}: a list of kept expert ids", .{layer});
        var row: [words]u32 = @splat(0);
        for (ids.array.items) |id| {
            if (id != .integer or id.integer < 0 or id.integer >= config.experts)
                return fail(a, problem, "layer {d}: expert ids run from 0 to {d}", .{ layer, config.experts - 1 });
            const e: usize = @intCast(id.integer);
            const bit = @as(u32, 1) << @intCast(e % 32);
            if (row[e / 32] & bit != 0) return fail(a, problem, "layer {d}: expert {d} is listed twice", .{ layer, e });
            row[e / 32] |= bit;
        }
        // drama_topk picks top_k experts a row; the reference refuses a layer pruned below that too
        if (kept(row) < config.top_k) return fail(a, problem, "layer {d} keeps {d} experts, under the {d} a row picks", .{ layer, kept(row), config.top_k });
        mask[layer] = row;
    }
    return mask;
}

/// The experts a layer's row keeps.
pub fn kept(row: [words]u32) usize {
    var n: usize = 0;
    for (row) |w| n += @popCount(w);
    return n;
}

/// The experts the whole mask keeps.
pub fn total(mask: *const Mask) usize {
    var n: usize = 0;
    for (mask) |row| n += kept(row);
    return n;
}

fn fail(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Allocator.Error!?Mask {
    problem.* = try std.fmt.allocPrint(a, fmt, args);
    return null;
}

test "a mask keeps exactly the listed experts of the listed layers, and every expert of the others" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const mask = (try parse(a, "{\"3\": [0, 1, 2, 31, 32, 100, 254, 255], \"39\": [9, 8, 7, 6, 5, 4, 3, 2, 1]}", &problem)).?;
    try std.testing.expectEqual([words]u32{ 0x80000007, 1, 0, 1 << 4, 0, 0, 0, 0xc0000000 }, mask[3]);
    try std.testing.expectEqual(@as(usize, 9), kept(mask[39]));
    try std.testing.expectEqual(@as(u32, 0x3fe), mask[39][0]);
    for (mask, 0..) |row, l| if (l != 3 and l != 39) try std.testing.expectEqual(@as(usize, config.experts), kept(row));
    try std.testing.expectEqual(@as(usize, 38 * config.experts + 8 + 9), total(&mask));
    // an empty object keeps everything
    try std.testing.expectEqual(@as(usize, config.layers * config.experts), total(&(try parse(a, "{}", &problem)).?));
}

test "a mask file is refused when a layer or an id is not one, an id repeats, or a layer keeps fewer than top_k" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const eight = "[0, 1, 2, 3, 4, 5, 6, 7]";
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "[1, 2]", "not a JSON object of layers" },
        .{ "{\"3\": [0, 1", "not a JSON object of layers (" },
        .{ "{\"40\": " ++ eight ++ "}", "layer \"40\" is not a layer index from 0 to 39" },
        .{ "{\"03\": " ++ eight ++ "}", "layer \"03\" is not a layer index from 0 to 39" },
        .{ "{\"x\": " ++ eight ++ "}", "layer \"x\" is not a layer index" },
        .{ "{\"3\": 8}", "layer 3: a list of kept expert ids" },
        .{ "{\"3\": [0, 1, 2, 3, 4, 5, 6, 256]}", "layer 3: expert ids run from 0 to 255" },
        .{ "{\"3\": [0, 1, 2, 3, 4, 5, 6, 7.0]}", "layer 3: expert ids run from 0 to 255" },
        .{ "{\"3\": [0, 1, 2, 3, 4, 5, 6, 6]}", "layer 3: expert 6 is listed twice" },
        .{ "{\"3\": [0, 1, 2, 3, 4, 5, 6]}", "layer 3 keeps 7 experts, under the 8 a row picks" },
        .{ "{\"3\": " ++ eight ++ ", \"3\": " ++ eight ++ "}", "not a JSON object of layers (" },
    };
    for (cases) |case| {
        var problem: []const u8 = "";
        try std.testing.expect(try parse(a, case[0], &problem) == null);
        try std.testing.expectStringStartsWith(problem, case[1]);
    }
}
