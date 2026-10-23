//! Guidance on the wire, against the scripted engine: what each draw carries to the engine, and the γ a reply says.
const std = @import("std");
const server = @import("root.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_long = @import("drama_long.zig");
const drama_cfg = @import("drama_cfg.zig");
const harness = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");
const Turn = scripted.Turn;
const Turns = scripted.Turns;

const gpa = std.testing.allocator;

/// One two-frame reference, its codes at both ends of [0, 4096).
const reference = "[{\"text\": \"Hi there.\", \"vq_codes\": [[0, 4095], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]}]";

fn config(cfg: ?drama_cfg.Options, long_form: ?drama_long.Options) server.Config {
    var c = scripted.config(long_form);
    c.cfg = cfg;
    return c;
}

/// A request for `input`, with the reference when `voiced`.
fn body(a: std.mem.Allocator, input: []const u8, stream: bool, voiced: bool) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"{s}\",\"response_format\":\"pcm\",\"stream\":{}{s}}}", .{ input, stream, if (voiced) ",\"references\":" ++ reference else "" });
}

/// The reply from its Content-Type on, sent with `gamma` as X-Cfg-Gamma if given, the route joined.
fn reply(a: std.mem.Allocator, t: *Turns, cfg: server.Config, gamma: ?[]const u8, request: []const u8) ![]const u8 {
    var line: [96]u8 = undefined;
    const first = if (gamma) |g| try std.fmt.bufPrint(&line, "POST /v1/audio/speech HTTP/1.1\r\nX-Cfg-Gamma: {s}", .{g}) else "POST /v1/audio/speech HTTP/1.1";
    const s = try gpa.create(harness.Served);
    defer scripted.done(s);
    try s.open(t.engine(), cfg, first, request, 2000);
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

/// The prompt rows of `input` spoken after the reference.
fn referenced(a: std.mem.Allocator, text: *harness.Bytes, input: []const u8) !drama_prompt.Rows {
    const frames = [_]drama_prompt.Codes{ .{ 0, 1, 3, 5, 7, 9, 11, 13, 15, 17 }, .{ 4095, 2, 4, 6, 8, 10, 12, 14, 16, 18 } };
    return drama_prompt.encodeReferenced(a, text.text(), input, &.{.{ .text = "Hi there.", .frames = &frames }});
}

test "off, a referenced request is served byte for byte as before: unguided, X-Cfg-Gamma neither read nor sent" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "It works.";
    const rows = try referenced(a, &text, input);
    for ([_]bool{ false, true }) |stream| for ([_]?[]const u8{ null, "3" }) |gamma| {
        var t: Turns = .{ .turns = &.{.{ .frames = 5 }} };
        defer t.deinit();
        const got = try reply(a, &t, config(null, null), gamma, try body(a, input, stream, true));
        try std.testing.expectEqualSlices(u32, rows.tokens, t.prompts.items[0]);
        try std.testing.expectEqualSlices(drama_prompt.Codes, rows.codes, t.codes.items[0]);
        try std.testing.expectEqual(@as(usize, 0), t.unconditioned.items[0].len);
        try std.testing.expectEqual(@as(f32, 1), t.gammas.items[0]);
        var want: std.ArrayList(u8) = .empty;
        if (stream) {
            try want.print(a, "Content-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\nX-Sample-Rate: 1000\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\nTrailer: X-Finish-Reason\r\n\r\n64\r\n", .{});
            try scripted.turnPcm(&want, a, .{ .frames = 5 }, 1);
            try want.appendSlice(a, "\r\n0\r\n\r\n");
        } else {
            try want.print(a, "Content-Type: audio/pcm\r\nContent-Length: 100\r\nX-Finish-Reason: stop\r\n\r\n", .{});
            try scripted.turnPcm(&want, a, .{ .frames = 5 }, 1);
        }
        try std.testing.expectEqualStrings(want.items, got);
    };
}

test "--cfg-gamma: a referenced draw carries its input's plain prompt, the built-in voice's; a plain one goes unguided" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "It works.";
    for ([_]bool{ false, true }) |stream| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 5 } } };
        defer t.deinit();
        const guided = try reply(a, &t, config(.{ .gamma = 2 }, null), "3", try body(a, input, stream, true));
        try std.testing.expectEqualSlices(u32, (try referenced(a, &text, input)).tokens, t.prompts.items[0]);
        try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), input), t.unconditioned.items[0]);
        try std.testing.expectEqual(@as(f32, 2), t.gammas.items[0]);
        try std.testing.expect(has(guided, "X-Cfg-Gamma: 2\r\n"));
        // the built-in voice's prompt is the plain prompt already
        const plain = try reply(a, &t, config(.{ .gamma = 2 }, null), null, try body(a, input, stream, false));
        try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), input), t.prompts.items[1]);
        try std.testing.expectEqual(@as(usize, 0), t.unconditioned.items[1].len);
        try std.testing.expectEqual(@as(f32, 1), t.gammas.items[1]);
        try std.testing.expect(has(plain, "X-Cfg-Gamma: 2\r\n"));
    }
}

test "--cfg-allow-request-gamma: X-Cfg-Gamma sets a request's γ, 1 leaves it unguided, outside 1 to 5 is refused" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "It works.";
    const plain = try drama_prompt.encode(a, text.text(), input);
    const open: drama_cfg.Options = .{ .request_gamma = true };
    const cases = [_]struct { cfg: drama_cfg.Options, sent: ?[]const u8, gamma: f32, says: []const u8 }{
        .{ .cfg = open, .sent = "1.5", .gamma = 1.5, .says = "1.5" },
        .{ .cfg = open, .sent = "3", .gamma = 3, .says = "3" },
        .{ .cfg = open, .sent = "1", .gamma = 1, .says = "1" },
        .{ .cfg = open, .sent = null, .gamma = 1, .says = "1" },
        .{ .cfg = .{ .gamma = 2, .request_gamma = true }, .sent = null, .gamma = 2, .says = "2" },
        .{ .cfg = .{ .gamma = 2 }, .sent = "3", .gamma = 2, .says = "2" },
    };
    for (cases) |case| {
        var t: Turns = .{ .turns = &.{.{ .frames = 5 }} };
        defer t.deinit();
        const got = try reply(a, &t, config(case.cfg, null), case.sent, try body(a, input, false, true));
        try std.testing.expectEqual(case.gamma, t.gammas.items[0]);
        try std.testing.expectEqualSlices(u32, if (case.gamma == 1) &.{} else plain, t.unconditioned.items[0]);
        try std.testing.expect(has(got, try std.fmt.allocPrint(a, "X-Cfg-Gamma: {s}\r\n", .{case.says})));
    }
    for ([_][]const u8{ "0.5", "6", "x", "nan" }) |bad| {
        var t: Turns = .{ .turns = &.{.{ .frames = 5 }} };
        defer t.deinit();
        const got = try reply(a, &t, config(open, null), bad, try body(a, input, false, true));
        try std.testing.expectEqual(@as(usize, 0), t.seen);
        try std.testing.expect(has(got, "X-Cfg-Gamma must be a number from 1 (unguided) to 5"));
    }
}

test "--cfg-force-guided: at γ 1 a referenced draw carries its plain prompt all the same; a built-in voice's none" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "It works.";
    const plain = try drama_prompt.encode(a, text.text(), input);
    const cases = [_]struct { cfg: drama_cfg.Options, sent: ?[]const u8 }{
        .{ .cfg = .{ .force_guided = true }, .sent = null },
        .{ .cfg = .{ .request_gamma = true, .force_guided = true }, .sent = "1" },
    };
    for (cases) |case| for ([_]bool{ false, true }) |stream| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 5 } } };
        defer t.deinit();
        const got = try reply(a, &t, config(case.cfg, null), case.sent, try body(a, input, stream, true));
        try std.testing.expectEqualSlices(u32, (try referenced(a, &text, input)).tokens, t.prompts.items[0]);
        try std.testing.expectEqualSlices(u32, plain, t.unconditioned.items[0]);
        try std.testing.expectEqual(@as(f32, 1), t.gammas.items[0]);
        try std.testing.expect(has(got, "X-Cfg-Gamma: 1\r\n"));
        _ = try reply(a, &t, config(case.cfg, null), case.sent, try body(a, input, stream, false));
        try std.testing.expectEqual(@as(usize, 0), t.unconditioned.items[1].len);
    };
}

test "a plain redraw is guided by the plain prompt of what it speaks" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence }, .{ .frames = 4 } } };
    defer t.deinit();
    var cfg = config(.{ .gamma = 2 }, null);
    cfg.silence_redraw = .{ .max = 1, .plain = true, .plain_scope = .controls };
    _ = try reply(a, &t, cfg, null, try body(a, "[pleased] It works.", false, true));
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expectEqualSlices(u32, (try referenced(a, &text, "It works.")).tokens, t.prompts.items[1]);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "[pleased] It works."), t.unconditioned.items[0]);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "It works."), t.unconditioned.items[1]);
}

test "long-form: each segment with audio rows, the voice's or earlier segments', is guided by its own plain prompt" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = try scripted.sentences(a, 2);
    const segments = try drama_long.split(a, input, 20);
    try std.testing.expectEqual(@as(usize, 2), segments.len);
    for ([_]bool{ true, false }) |voiced| for ([_]bool{ false, true }) |stream| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 3 } } };
        defer t.deinit();
        const got = try reply(a, &t, config(.{ .gamma = 3 }, .{ .chars = 20 }), null, try body(a, input, stream, voiced));
        try std.testing.expectEqual(@as(usize, 2), t.seen);
        try std.testing.expect(has(got, "X-Cfg-Gamma: 3\r\n"));
        for (segments, 0..) |segment, i| {
            // the built-in voice's first segment hears nothing before it: its prompt is the plain one
            const guided = voiced or i > 0;
            try std.testing.expectEqual(guided, t.codes.items[i].len > 0);
            try std.testing.expectEqualSlices(u32, if (guided) try drama_prompt.encode(a, text.text(), segment) else &.{}, t.unconditioned.items[i]);
            try std.testing.expectEqual(@as(f32, if (guided) 3 else 1), t.gammas.items[i]);
        }
    };
}
