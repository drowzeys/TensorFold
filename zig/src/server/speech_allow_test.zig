//! --allowed-controls on the wire, against the scripted engine: what every draw speaks, and the count sent.
const std = @import("std");
const server = @import("root.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_long = @import("drama_long.zig");
const harness = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");
const Turn = scripted.Turn;
const Turns = scripted.Turns;

const gpa = std.testing.allocator;
const allowed: []const []const u8 = &.{ "concerned", "curious", "small laugh" };

fn config(list: ?[]const []const u8, long_form: ?drama_long.Options) server.Config {
    var cfg = scripted.config(long_form);
    cfg.allowed_controls = list;
    return cfg;
}

/// The whole reply from its Content-Type on, the route joined.
fn reply(a: std.mem.Allocator, t: *Turns, cfg: server.Config, input: []const u8, stream: bool) ![]const u8 {
    const s = try gpa.create(harness.Served);
    defer scripted.done(s);
    try s.open(t.engine(), cfg, "POST /v1/audio/speech HTTP/1.1", try scripted.body(a, input, stream), 2000);
    _ = try s.readHead();
    s.route.join();
    // all of it read first: reading more can move the buffer a head slice points into
    _ = try s.readRest();
    const got = s.got.items;
    return a.dupe(u8, got[std.mem.indexOf(u8, got, "Content-Type").?..]);
}

fn has(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

test "controls off the list are gone from the first draw on; listed ones and brackets that are not controls stay" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const cases = [_]struct { input: []const u8, spoken: []const u8, stripped: []const u8 }{
        .{ .input = "[PLEASED!] It works. [Concerned] But <SOFT>quietly</soft> [warmly] so, [John].", .spoken = "It works. [concerned] But quietly so, [John].", .stripped = "4" },
        .{ .input = "See [1] and [https://x.io] [small laugh] here, x[sighs].", .spoken = "See [1] and [https://x.io] [small laugh] here, x[sighs].", .stripped = "0" },
        .{ .input = "A [note <SOFT>quietly</SOFT>] works.", .spoken = "A [note quietly] works.", .stripped = "2" },
        .{ .input = "A [see [note] [sighs]] works. [sighs] Done.", .spoken = "A [see [note] [sighs]] works. Done.", .stripped = "1" },
    };
    for (cases) |case| for ([_]bool{ false, true }) |stream| {
        var t: Turns = .{ .turns = &.{.{ .frames = 5 }} };
        defer t.deinit();
        const got = try reply(a, &t, config(allowed, null), case.input, stream);
        try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), case.spoken), t.prompts.items[0]);
        try std.testing.expect(has(got, try std.fmt.allocPrint(a, "X-Controls-Stripped: {s}\r\n", .{case.stripped})));
    };
}

test "unset, an input with controls is served byte for byte as before, with no count" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "[pleased] It works. <soft>quietly</soft> so.";
    var t: Turns = .{ .turns = &.{.{ .frames = 5 }} };
    defer t.deinit();
    const got = try reply(a, &t, config(null, null), input, false);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), input), t.prompts.items[0]);
    var want: std.ArrayList(u8) = .empty;
    try want.print(a, "Content-Type: audio/pcm\r\nContent-Length: 100\r\nX-Finish-Reason: stop\r\n\r\n", .{});
    try scripted.turnPcm(&want, a, .{ .frames = 5 }, 1);
    try std.testing.expectEqualStrings(want.items, got);
}

test "a plain redraw starts from the filtered input and drops the listed controls too" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence }, .{ .frames = 4 } } };
    defer t.deinit();
    var cfg = config(allowed, null);
    cfg.silence_redraw = .{ .max = 1, .plain = true, .plain_scope = .controls };
    const got = try reply(a, &t, cfg, "[pleased] It works. [concerned] Next.", false);
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "It works. [concerned] Next."), t.prompts.items[0]);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "It works. Next."), t.prompts.items[1]);
    try std.testing.expect(has(got, "X-Silence-Redraws: 1\r\nX-Controls-Stripped: 1\r\n"));
}

test "controls go before the marks are read: a mark after a control off the list still cuts" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 4 } } };
    defer t.deinit();
    const got = try reply(a, &t, config(allowed, .{ .shift_pause_s = 0.85 }), "One. [pleased] \u{b6} Two.", false);
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "One."), t.prompts.items[0]);
    try std.testing.expect(has(got, "X-Pause-Spans: ") and has(got, "X-Controls-Stripped: 1\r\n"));
}

test "the short-reply rule counts the input without the controls taken out" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // 209 characters as sent, 152 filtered: under the rule's 200, a short stop is redrawn only unfiltered
    const xs: [150]u8 = @splat('x');
    const input = xs ++ ". [softly, quietly, almost whispering, with a little smile]";
    for ([_]?[]const []const u8{ null, allowed }, [_]usize{ 2, 1 }) |list, draws| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 5 } } };
        defer t.deinit();
        var cfg = config(list, null);
        cfg.silence_redraw = .{ .max = 1, .short_ratio = 0.5 };
        _ = try reply(a, &t, cfg, input, false);
        try std.testing.expectEqual(draws, t.seen);
    }
}
