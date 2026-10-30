//! A released drafter context keeps its stream tokens; a later stream repeating them keeps its rows.
const std = @import("std");
const lanes = @import("lanes");
const Stream = lanes.Stream;

/// TF_DRAFT_RESUME=0: every stream's context starts empty, as before.
fn off() bool {
    const v = std.c.getenv("TF_DRAFT_RESUME") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

/// Rows of `ctx` a stream with prompt `prompt` may keep: the kept tokens it repeats, within the rows the ring holds.
pub fn usable(begin: u64, end: u64, kept: []const u32, prompt: []const u32) u64 {
    const n = @min(@min(kept.len, prompt.len), end);
    var same: usize = 0;
    while (same < n and kept[same] == prompt[same]) same += 1;
    return if (same > begin) same else 0;
}

/// The pooled context whose rows the stream repeats most, out of the pool (null: none repeats any).
pub fn take(d: anytype, s: *const Stream) ?@TypeOf(d.pool.items[0]) {
    if (off()) return null;
    var best: ?usize = null;
    var most: u64 = 0;
    for (d.pool.items, 0..) |ctx, i| {
        const n = usable(ctx.begin, ctx.end, ctx.toks.items, s.prompt());
        if (n > most) {
            most = n;
            best = i;
        }
    }
    return d.pool.swapRemove(best orelse return null);
}

/// The context reset for a new stream, keeping its rows the stream's prompt repeats (absorb() cuts at its first row).
pub fn open(ctx: anytype, s: *const Stream) void {
    const rows = if (off()) 0 else usable(ctx.begin, ctx.end, ctx.toks.items, s.prompt());
    const begin = ctx.begin;
    ctx.* = .{ .ring = ctx.ring, .toks = ctx.toks };
    if (rows > 0) {
        ctx.begin = begin;
        ctx.end = rows;
    }
}

/// A released context keeps its stream's tokens (none when they do not fit).
pub fn keep(gpa: std.mem.Allocator, ctx: anytype, s: *const Stream) void {
    ctx.toks.clearRetainingCapacity();
    if (off()) return;
    ctx.toks.appendSlice(gpa, s.context.items) catch ctx.toks.clearRetainingCapacity();
}

test "a stream keeps the rows its prompt repeats, within the ring's rows" {
    const kept = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try std.testing.expectEqual(@as(u64, 4), usable(0, 6, &kept, &.{ 1, 2, 3, 4, 9, 9, 9 }));
    try std.testing.expectEqual(@as(u64, 5), usable(2, 5, &kept, &.{ 1, 2, 3, 4, 5, 6, 7 })); // to the ring's end
    try std.testing.expectEqual(@as(u64, 0), usable(3, 6, &kept, &.{ 1, 2, 9 })); // nothing past the ring's first row
    try std.testing.expectEqual(@as(u64, 0), usable(0, 6, &kept, &.{ 7, 2, 3 }));
}
