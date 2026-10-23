//! A compact Drama 3 checkpoint's maps in its index `metadata`: held experts and lm_head rows, each by original id.
const std = @import("std");
const cfg = @import("backbone_config.zig");
const expert_mask = @import("expert_mask.zig");
const Allocator = std.mem.Allocator;

/// `{"layer": [held expert ids, ascending]}` for every layer; a layer's experts keep their original tensor names.
pub const experts_key = "drama_experts";
/// `[eos, semantic_start, ..., semantic_end]`: the vocabulary row of each row lm_head holds, in order.
pub const head_key = "drama_head_rows";

const Row = [expert_mask.words]u32;

/// Per layer, the original ids of the experts it holds, ascending: the layer's slot s holds expert `ids[s]`.
pub const Held = struct {
    counts: [cfg.layers]u16 = @splat(cfg.experts),
    ids: [cfg.layers][cfg.experts]u8 = @splat(identity),

    const identity = blk: {
        var ids: [cfg.experts]u8 = undefined;
        for (&ids, 0..) |*id, e| id.* = @intCast(e);
        break :blk ids;
    };

    pub fn layer(h: *const Held, l: usize) []const u8 {
        return h.ids[l][0..h.counts[l]];
    }

    /// The layer's held ids as drama_topk's keep row.
    pub fn row(h: *const Held, l: usize) Row {
        var r: Row = @splat(0);
        for (h.layer(l)) |e| r[e / 32] |= @as(u32, 1) << @intCast(e % 32);
        return r;
    }

    pub fn total(h: *const Held) usize {
        var n: usize = 0;
        for (h.counts) |c| n += c;
        return n;
    }
};

/// drama_topk's per-layer tables: the experts a row may pick, and on a compact checkpoint each original id's slot.
pub const Tables = struct {
    keep: expert_mask.Mask,
    /// The held slot of each original id, -1 where the layer does not hold it; null on a full checkpoint.
    slots: ?[cfg.layers][cfg.experts]i32,
};

/// drama_topk's tables for `held` (null: all) under --expert-mask `mask` (inside held, see `unheld`); null for neither.
pub fn tables(held: ?*const Held, mask: ?*const expert_mask.Mask) ?Tables {
    if (held == null and mask == null) return null;
    var t: Tables = .{ .keep = if (mask) |m| m.* else @splat(@splat(std.math.maxInt(u32))), .slots = null };
    const h = held orelse return t;
    t.slots = @splat(@splat(-1));
    for (&t.keep, &t.slots.?, 0..) |*keep, *slots, l| {
        if (expert_mask.kept(keep.*) == cfg.experts) keep.* = h.row(l); // a layer the mask leaves out keeps all held
        for (h.layer(l), 0..) |e, s| slots[e] = @intCast(s);
    }
    return t;
}

/// The first expert `mask` keeps that the checkpoint does not hold, or null.
pub fn unheld(held: *const Held, mask: *const expert_mask.Mask) ?struct { layer: usize, expert: usize } {
    for (mask, 0..) |keep, l| {
        if (expert_mask.kept(keep) == cfg.experts) continue;
        const have = held.row(l);
        for (keep, have, 0..) |k, x, w| if (k & ~x != 0) return .{ .layer = l, .expert = w * 32 + @ctz(k & ~x) };
    }
    return null;
}

/// The map `value`, or null with `problem` set: all 40 layers, each at least top_k strictly ascending ids below 256.
pub fn parse(a: Allocator, value: std.json.Value, problem: *[]const u8) Allocator.Error!?Held {
    if (value != .object) return fail(a, problem, "not a JSON object of layers", .{});
    var h: Held = .{};
    var listed: [cfg.layers]bool = @splat(false);
    for (value.object.keys(), value.object.values()) |key, ids| {
        const l = std.fmt.parseInt(u8, key, 10) catch return fail(a, problem, "layer \"{s}\" is not a layer index", .{key});
        if (l >= cfg.layers or !std.mem.eql(u8, key, try std.fmt.allocPrint(a, "{d}", .{l})))
            return fail(a, problem, "layer \"{s}\" is not a layer index from 0 to {d}", .{ key, cfg.layers - 1 });
        listed[l] = true;
        if (ids != .array) return fail(a, problem, "layer {d}: a list of held expert ids", .{l});
        const items = ids.array.items;
        if (items.len < cfg.top_k or items.len > cfg.experts) return fail(a, problem, "layer {d} holds {d} experts, not {d} to {d}", .{ l, items.len, cfg.top_k, cfg.experts });
        for (items, 0..) |id, s| {
            if (id != .integer or id.integer < 0 or id.integer >= cfg.experts)
                return fail(a, problem, "layer {d}: expert ids run from 0 to {d}", .{ l, cfg.experts - 1 });
            const e: u8 = @intCast(id.integer);
            if (s > 0 and e == h.ids[l][s - 1]) return fail(a, problem, "layer {d}: expert {d} is listed twice", .{ l, e });
            if (s > 0 and e < h.ids[l][s - 1]) return fail(a, problem, "layer {d}: expert ids are not ascending", .{l});
            h.ids[l][s] = e;
        }
        h.counts[l] = @intCast(items.len);
    }
    for (listed, 0..) |x, l| if (!x) return fail(a, problem, "layer {d} is not listed", .{l});
    return h;
}

/// The routed expert a backbone tensor belongs to (`model.language_model.layers.L.mlp.experts.E.*`), or null.
pub fn expertOf(name: []const u8) ?struct { layer: usize, expert: usize } {
    const pre = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return null;
    var it = std.mem.splitScalar(u8, name[pre.len..], '.');
    const l = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "mlp") or !std.mem.eql(u8, it.next() orelse return null, "experts")) return null;
    return .{ .layer = l, .expert = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null };
}

/// The first of `names` that belongs to an expert `held` does not list, or null: a map that leaves out a stored expert.
pub fn stray(held: *const Held, names: anytype) ?[]const u8 {
    var it = names;
    while (it.next()) |name| {
        const x = expertOf(name.*) orelse continue;
        if (x.layer >= cfg.layers or std.mem.indexOfScalar(u8, held.layer(x.layer), std.math.cast(u8, x.expert) orelse return name.*) == null) return name.*;
    }
    return null;
}

fn fail(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Allocator.Error!?Held {
    problem.* = try std.fmt.allocPrint(a, fmt, args);
    return null;
}

/// A map's JSON text: `layers` as given, every other layer holding all of its experts.
fn mapText(a: Allocator, layers: []const struct { usize, []const u8 }) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, '{');
    for (0..cfg.layers) |l| {
        try out.print(a, "{s}\"{d}\": ", .{ if (l == 0) "" else ", ", l });
        for (layers) |x| {
            if (x[0] != l) continue;
            try out.appendSlice(a, x[1]);
            break;
        } else {
            try out.append(a, '[');
            for (0..cfg.experts) |e| try out.print(a, "{s}{d}", .{ if (e == 0) "" else ", ", e });
            try out.append(a, ']');
        }
    }
    try out.append(a, '}');
    return out.items;
}

fn parseText(a: Allocator, text: []const u8, problem: *[]const u8) !?Held {
    return parse(a, try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}), problem);
}

test "a map holds each layer's listed experts in slot order; tables keep them and slot them, narrowed by a mask" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const h = (try parseText(a, try mapText(a, &.{.{ 3, "[1, 2, 31, 32, 100, 200, 254, 255]" }}), &problem)).?;
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 31, 32, 100, 200, 254, 255 }, h.layer(3));
    try std.testing.expectEqual(@as(usize, 39 * cfg.experts + 8), h.total());
    try std.testing.expectEqual(Row{ 0x80000006, 1, 0, 1 << 4, 0, 0, 1 << 8, 0xc0000000 }, h.row(3));
    const t = tables(&h, null).?;
    try std.testing.expectEqual(h.row(3), t.keep[3]);
    for (t.slots.?[3], 0..) |s, e| try std.testing.expectEqual(if (std.mem.indexOfScalar(u8, h.layer(3), @intCast(e))) |i| @as(i32, @intCast(i)) else -1, s);
    for (t.slots.?[4], 0..) |s, e| try std.testing.expectEqual(@as(i32, @intCast(e)), s);
    // a mask narrows a held layer and prunes a full one; its rows are the keep rows, the slots stay the checkpoint's
    const m = (try expert_mask.parse(a, "{\"3\": [1, 2, 31, 32, 100, 200, 254, 255], \"5\": [0, 1, 2, 3, 4, 5, 6, 9]}", &problem)).?;
    const both = tables(&h, &m).?;
    try std.testing.expectEqual(m[5], both.keep[5]);
    try std.testing.expectEqual(t.slots.?, both.slots.?);
    try std.testing.expect(unheld(&h, &m) == null);
    const beyond = (try expert_mask.parse(a, "{\"3\": [1, 2, 31, 32, 100, 200, 254, 3]}", &problem)).?;
    try std.testing.expectEqual(@as(usize, 3), unheld(&h, &beyond).?.expert);
    // the mask alone: its rows, no slots; neither: no tables
    const alone = tables(null, &m).?;
    try std.testing.expectEqual(m, alone.keep);
    try std.testing.expect(alone.slots == null and tables(null, null) == null);
}

test "a map is refused unless every layer lists at least top_k distinct ascending ids from 0 to 255" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const eight = "[0, 1, 2, 3, 4, 5, 6, 7]";
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "[1, 2]", "not a JSON object of layers" },
        .{ "{\"3\": " ++ eight ++ "}", "layer 0 is not listed" },
        .{ "{\"x\": " ++ eight ++ "}", "layer \"x\" is not a layer index" },
        .{ "{\"40\": " ++ eight ++ "}", "layer \"40\" is not a layer index from 0 to 39" },
        .{ "{\"03\": " ++ eight ++ "}", "layer \"03\" is not a layer index from 0 to 39" },
    };
    for (cases) |c| {
        var problem: []const u8 = "";
        try std.testing.expect(try parseText(a, c[0], &problem) == null);
        try std.testing.expectStringStartsWith(problem, c[1]);
    }
    const layers = [_]struct { []const u8, []const u8 }{
        .{ "8", "layer 3: a list of held expert ids" },
        .{ "[0, 1, 2, 3, 4, 5, 6]", "layer 3 holds 7 experts, not 8 to 256" },
        .{ "[0, 1, 2, 3, 4, 5, 6, 256]", "layer 3: expert ids run from 0 to 255" },
        .{ "[-1, 1, 2, 3, 4, 5, 6, 7]", "layer 3: expert ids run from 0 to 255" },
        .{ "[0, 1, 2, 3, 4, 5, 6, 7.0]", "layer 3: expert ids run from 0 to 255" },
        .{ "[0, 1, 2, 3, 4, 5, 6, 6]", "layer 3: expert 6 is listed twice" },
        .{ "[0, 1, 2, 3, 4, 5, 7, 6]", "layer 3: expert ids are not ascending" },
    };
    for (layers) |c| {
        var problem: []const u8 = "";
        try std.testing.expect(try parseText(a, try mapText(a, &.{.{ 3, c[0] }}), &problem) == null);
        try std.testing.expectStringStartsWith(problem, c[1]);
    }
    // a key twice fails the JSON parse itself
    try std.testing.expectError(error.DuplicateField, std.json.parseFromSliceLeaky(std.json.Value, a, "{\"3\": " ++ eight ++ ", \"3\": " ++ eight ++ "}", .{}));
}

test "a stored expert the map does not hold is a stray; other tensors are not experts" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const h = (try parseText(a, try mapText(a, &.{.{ 3, "[1, 2, 3, 4, 5, 6, 7, 8]" }}), &problem)).?;
    const pre = "model.language_model.layers.";
    var fine = [_][]const u8{ pre ++ "3.mlp.experts.8.down_proj.weight", pre ++ "3.mlp.gate.weight", pre ++ "4.mlp.experts.0.up_proj.weight_scale", "lm_head.weight" };
    var it: SliceIter = .{ .items = &fine };
    try std.testing.expect(stray(&h, it) == null);
    for ([_][]const u8{ pre ++ "3.mlp.experts.0.gate_proj.weight", pre ++ "3.mlp.experts.9.down_proj.weight_scale_inv", pre ++ "40.mlp.experts.0.up_proj.weight", pre ++ "4.mlp.experts.256.up_proj.weight" }) |bad| {
        var names = fine ++ [_][]const u8{bad};
        it = .{ .items = &names };
        try std.testing.expectEqualStrings(bad, stray(&h, it).?);
    }
    try std.testing.expectEqual(@as(usize, 7), expertOf(pre ++ "12.mlp.experts.7.gate_proj.weight").?.expert);
    try std.testing.expect(expertOf(pre ++ "12.mlp.shared_expert.gate_proj.weight") == null);
}

/// An iterator over names like a hash map's key iterator: `next` yields pointers.
const SliceIter = struct {
    items: []const []const u8,
    at: usize = 0,

    fn next(it: *SliceIter) ?*const []const u8 {
        if (it.at == it.items.len) return null;
        it.at += 1;
        return &it.items[it.at - 1];
    }
};
