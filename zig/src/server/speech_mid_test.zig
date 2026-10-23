//! --silence-mid-s on the wire, against the scripted engine: a draw it ended is a silence, counted in its own header.
const std = @import("std");
const server = @import("root.zig");
const drama_long = @import("drama_long.zig");
const speech_redraw = @import("speech_redraw.zig");
const harness = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");
const Turns = scripted.Turns;

const gpa = std.testing.allocator;
/// A 3.2 s hole at the scripted engine's 1 kHz.
const hole = 3200;

fn config(mid: bool, policy: speech_redraw.Policy, long_form: ?drama_long.Options) server.Config {
    var cfg = scripted.config(long_form);
    cfg.silence_redraw = policy;
    cfg.silence_mid = mid;
    return cfg;
}

/// Served once, with `deadline` as X-Deadline-S if given; the reply's head and the whole reply.
fn serve(t: *Turns, cfg: server.Config, deadline: ?[]const u8, request: []const u8) !struct { s: *harness.Served, head: []const u8, all: []const u8 } {
    var line: [96]u8 = undefined;
    const s = try gpa.create(harness.Served);
    const first = if (deadline) |d| try std.fmt.bufPrint(&line, "POST /v1/audio/speech HTTP/1.1\r\nX-Deadline-S: {s}", .{d}) else "POST /v1/audio/speech HTTP/1.1";
    try s.open(t.engine(), cfg, first, request, 2000);
    _ = try s.readHead();
    s.route.join();
    _ = try s.readRest();
    // read whole first: reading more can move the bytes a head slice points into
    const all = s.got.items;
    return .{ .s = s, .head = all[0 .. std.mem.indexOf(u8, all, "\r\n\r\n").? + 4], .all = all };
}

fn has(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

test "a draw --silence-mid-s ended is redrawn under the silence redraws' rules and counted in X-Silence-Mid-Trips" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // a 150 ms draw: 0.25 s leaves no room for a redraw, 5 s does
    for ([_]struct { deadline: []const u8, redrawn: bool }{ .{ .deadline = "5", .redrawn = true }, .{ .deadline = "0.25", .redrawn = false } }) |case| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence, .stall_ms = 150, .hollow = hole }, .{ .frames = 4 } } };
        defer t.deinit();
        const r = try serve(&t, config(true, .{ .max = 2 }, null), case.deadline, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, if (case.redrawn) 2 else 1), t.seen);
        try std.testing.expect(has(r.head, if (case.redrawn) "X-Finish-Reason: stop\r\nX-Silence-Redraws: 1\r\nX-Silence-Mid-Trips: 1\r\n" else "X-Finish-Reason: silence\r\nX-Silence-Redraws: 0\r\nX-Silence-Mid-Trips: 1\r\n"));
    }
}

test "a short stop the mid rule ended is one silence trip, not also a short one: one redraw, one mid trip" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // 240 characters are 15.4 s expected: at a ratio of 0.5, 5 frames are short and 900 (9 s) are not
    const sentence = "Sentence one is here and it is plain enough to read aloud, nothing more. ";
    const input = sentence ++ sentence ++ sentence ++ "Then a last sentence.";
    var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence, .hollow = hole }, .{ .frames = 900 }, .{ .frames = 900 } } };
    defer t.deinit();
    const r = try serve(&t, config(true, .{ .max = 2, .short_ratio = 0.5 }, null), null, try scripted.body(a, input, false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: stop\r\nX-Silence-Redraws: 1\r\nX-Silence-Mid-Trips: 1\r\n"));
}

test "a long-form segment the mid rule ended is redrawn in place, and the whole reply counts it" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence, .hollow = hole }, .{ .frames = 5 }, .{ .frames = 4, .reason = .silence, .hollow = hole }, .{ .frames = 4 } } };
    defer t.deinit();
    const r = try serve(&t, config(true, .{ .max = 2 }, .{ .chars = 20 }), null, try scripted.body(a, try scripted.sentences(a, 2), false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 4), t.seen);
    try std.testing.expect(has(r.head, "X-Silence-Redraws: 2\r\nX-Silence-Mid-Trips: 2\r\n") and has(r.head, "X-Long-Form-Segments: 2;"));
}

test "with --silence-mid-s off a reply is byte for byte as before; on, an untripped one differs only by its header" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var replies: [2][]const u8 = undefined;
    for ([_]bool{ false, true }, &replies) |mid, *reply| {
        var t: Turns = .{ .turns = &.{.{ .frames = 4 }} };
        defer t.deinit();
        const r = try serve(&t, config(mid, .{ .max = 2 }, null), null, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        reply.* = try a.dupe(u8, r.all[std.mem.indexOf(u8, r.all, "Content-Type").?..]);
    }
    try std.testing.expect(!has(replies[0], "X-Silence-Mid-Trips"));
    try std.testing.expectEqualStrings(replies[0], try std.mem.replaceOwned(u8, a, replies[1], "X-Silence-Mid-Trips: 0\r\n", ""));
}
