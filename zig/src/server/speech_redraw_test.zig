//! --silence-redraws on the wire, against the scripted engine: what is redrawn, with which prompt and seed, or not.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_long = @import("drama_long.zig");
const speech_redraw = @import("speech_redraw.zig");
const harness = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");
const Turn = scripted.Turn;
const Turns = scripted.Turns;

const gpa = std.testing.allocator;

fn redrawing(policy: speech_redraw.Policy, long_form: ?drama_long.Options) server.Config {
    var cfg = scripted.config(long_form);
    cfg.silence_redraw = policy;
    return cfg;
}

/// Served once, with `deadline` as X-Deadline-S if given, the route joined; the reply head and the served request.
fn serve(t: *Turns, cfg: server.Config, deadline: ?[]const u8, request: []const u8) !struct { s: *harness.Served, head: []const u8 } {
    var line: [96]u8 = undefined;
    const s = try gpa.create(harness.Served);
    const first = if (deadline) |d| try std.fmt.bufPrint(&line, "POST /v1/audio/speech HTTP/1.1\r\nX-Deadline-S: {s}", .{d}) else "POST /v1/audio/speech HTTP/1.1";
    try s.open(t.engine(), cfg, first, request, 2000);
    const head = try s.readHead();
    s.route.join();
    return .{ .s = s, .head = head };
}

fn has(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

fn pcm(a: std.mem.Allocator, turn: Turn) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try scripted.turnPcm(&out, a, turn, 1);
    return out.items;
}

test "a whole reply the guard silenced is drawn again with the next salt and its cues, and the new draw is sent" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // another request takes the engine between the draws: the redraw waits for it rather than refuse the reply
    const turns = [_]Turn{ .{ .frames = 5, .reason = .silence, .level = 0.3 }, .{ .frames = 4, .busy = 2 } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2 }, null), null, try scripted.body(a, "One. [pause] Two.", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: stop\r\nX-Silence-Redraws: 1\r\n"));
    try std.testing.expectEqualSlices(u32, t.prompts.items[0], t.prompts.items[1]);
    for (0..2) |k| try std.testing.expectEqual(api.seedFor(t.prompts.items[0], @intCast(k)), t.seeds.items[k]);
    try std.testing.expectEqualSlices(u8, try pcm(a, turns[1]), try r.s.readRest());
}

test "redraws stop at --silence-redraws: the last draw is sent with its silence" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const turns = [_]Turn{ .{ .frames = 5, .reason = .silence }, .{ .frames = 6, .reason = .silence }, .{ .frames = 7, .reason = .silence } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2 }, null), null, try scripted.body(a, "One. Two.", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 3), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: silence\r\nX-Silence-Redraws: 2\r\n"));
    for (0..3) |k| try std.testing.expectEqual(api.seedFor(t.prompts.items[0], @intCast(k)), t.seeds.items[k]);
    try std.testing.expectEqualSlices(u8, try pcm(a, turns[2]), try r.s.readRest());
}

test "a redraw is started only if t + d fits the budget, the lower of X-Deadline-S and the server's" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // a 150 ms draw silenced: t + d is at least 0.3 s, so a 0.25 s budget fails closed though t alone fits it
    const cases = [_]struct { budget_s: f64, deadline: ?[]const u8, redrawn: bool }{
        .{ .budget_s = 120, .deadline = "0.25", .redrawn = false },
        .{ .budget_s = 0.25, .deadline = "5", .redrawn = false },
        .{ .budget_s = 0.25, .deadline = null, .redrawn = false },
        .{ .budget_s = 120, .deadline = "5", .redrawn = true },
    };
    for (cases) |case| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence, .stall_ms = 150 }, .{ .frames = 4 } } };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{ .max = 2, .budget_s = case.budget_s }, null), case.deadline, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, if (case.redrawn) 2 else 1), t.seen);
        try std.testing.expect(has(r.head, if (case.redrawn) "X-Finish-Reason: stop\r\nX-Silence-Redraws: 1\r\n" else "X-Finish-Reason: silence\r\nX-Silence-Redraws: 0\r\n"));
    }
}

test "with redraws off a reply is as before: X-Deadline-S unread, no header, the silenced draw final" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence }, .{ .frames = 4 } } };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{}, null), "soon", try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, 1), t.seen);
        try std.testing.expect(has(r.head, "HTTP/1.1 200") and has(r.head, "X-Finish-Reason: silence\r\n") and !has(r.head, "X-Silence-Redraws"));
    }
    // a draw not silenced is the same request and reply with redraws on, but for the header
    var replies: [2][]const u8 = undefined;
    var seeds: [2]u64 = undefined;
    for ([_]u8{ 0, 2 }, 0..) |max, i| {
        var t: Turns = .{ .turns = &.{.{ .frames = 4 }} };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{ .max = max }, null), null, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        seeds[i] = t.seeds.items[0];
        // all of it read first: reading more can move the buffer `r.head` points into
        _ = try r.s.readRest();
        const got = r.s.got.items;
        replies[i] = try a.dupe(u8, got[std.mem.indexOf(u8, got, "Content-Type").?..]);
    }
    try std.testing.expectEqual(seeds[0], seeds[1]);
    try std.testing.expectEqualStrings(replies[0], try std.mem.replaceOwned(u8, a, replies[1], "X-Silence-Redraws: 0\r\n", ""));
    try std.testing.expect(has(replies[1], "X-Silence-Redraws: 0\r\n"));
}

test "a stream is never redrawn: its first audio is out before the guard can trip" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]?drama_long.Options{ null, .{ .chars = 20 } }) |long_form| {
        const turns: []const Turn = if (long_form == null) &.{.{ .frames = 5, .reason = .silence }} else &.{ .{ .frames = 5 }, .{ .frames = 3, .reason = .silence } };
        var t: Turns = .{ .turns = turns };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{ .max = 2 }, long_form), null, try scripted.body(a, try scripted.sentences(a, turns.len), true));
        defer scripted.done(r.s);
        try std.testing.expectEqual(turns.len, t.seen);
        try std.testing.expect(has(r.head, "X-Silence-Redraws: 0\r\n"));
        try std.testing.expect(has(try r.s.readRest(), "\r\n0\r\nX-Finish-Reason: silence\r\n"));
    }
}

test "a long-form whole reply redraws a silenced segment, one count for the reply, later segments hearing the redraw" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const turns = [_]Turn{ .{ .frames = 5 }, .{ .frames = 3, .reason = .silence }, .{ .frames = 4 }, .{ .frames = 6, .reason = .silence } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 1 }, .{ .chars = 20 }), null, try scripted.body(a, try scripted.sentences(a, 3), false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 4), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: silence\r\n") and has(r.head, "X-Silence-Redraws: 1\r\n"));
    try std.testing.expectEqualSlices(u32, t.prompts.items[1], t.prompts.items[2]);
    try std.testing.expectEqual(api.seedFor(t.prompts.items[2], 1), t.seeds.items[2]);
    try std.testing.expectEqual(api.seedFor(t.prompts.items[3], 0), t.seeds.items[3]);
    // the third segment hears the second as redrawn (the third submission's codes), not the silenced draw
    var redrawn = false;
    for (t.codes.items[3]) |codes| {
        try std.testing.expect(!std.meta.eql(codes, @as(drama_prompt.Codes, @splat(200))));
        redrawn = redrawn or std.meta.eql(codes, @as(drama_prompt.Codes, @splat(300)));
    }
    try std.testing.expect(redrawn);
}

test "--silence-redraw-plain: every redraw speaks the input without its cues, seeded salt + k; no cue, no change" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    for ([_][]const u8{ "Done. [pause] Next one. [sighs] Fine.", "Done. Next one. Fine." }) |input| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence }, .{ .frames = 6, .reason = .silence }, .{ .frames = 4 } } };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{ .max = 2, .plain = true }, null), null, try scripted.body(a, input, false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, 3), t.seen);
        try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), input), t.prompts.items[0]);
        const plain = try drama_prompt.encode(a, text.text(), "Done. Next one. Fine.");
        for (1..3) |k| {
            try std.testing.expectEqualSlices(u32, plain, t.prompts.items[k]);
            try std.testing.expectEqual(api.seedFor(plain, @intCast(k)), t.seeds.items[k]);
        }
    }
    // a long-form segment redrawn plain: only that segment's text loses its cues
    const input = "Sentence 00 is ok. Sentence [sighs] ok. Sentence 02 is ok.";
    const segments = try drama_long.split(a, input, 20);
    var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 3, .reason = .silence }, .{ .frames = 4 }, .{ .frames = 4 } } };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 1, .plain = true }, .{ .chars = 20, .history = 0 }), null, try scripted.body(a, input, false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 4), t.seen);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), segments[1]), t.prompts.items[1]);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), (try speech_redraw.plain(a, segments[1], .cues)).?), t.prompts.items[2]);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), segments[2]), t.prompts.items[3]);
}

test "--silence-redraw-plain never speaks the cues again: an input of cues alone is not redrawn" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const turns = [_]Turn{ .{ .frames = 5, .reason = .silence }, .{ .frames = 4 } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2, .plain = true }, null), null, try scripted.body(a, "[pause] [sighs]", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 1), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: silence\r\nX-Silence-Redraws: 0\r\n"));
    try std.testing.expectEqualSlices(u8, try pcm(a, turns[0]), try r.s.readRest());
}

test "with redraws on, a refusal carries X-Silence-Redraws too: before any draw, and after a redraw" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const valid = try scripted.body(a, "One. Two.", false);
    const cases = [_]struct { max: u8, deadline: ?[]const u8 = null, body: []const u8, turns: []const Turn, status: []const u8, header: ?[]const u8 }{
        .{ .max = 2, .deadline = "soon", .body = valid, .turns = &.{}, .status = " 400 ", .header = "X-Silence-Redraws: 0\r\n" },
        .{ .max = 2, .body = "{\"model\":\"m\"}", .turns = &.{}, .status = " 400 ", .header = "X-Silence-Redraws: 0\r\n" },
        .{ .max = 2, .body = valid, .turns = &.{.{ .frames = 4, .busy = 1 }}, .status = " 503 ", .header = "X-Silence-Redraws: 0\r\n" },
        .{ .max = 2, .body = valid, .turns = &.{ .{ .frames = 5, .reason = .silence }, .{ .frames = 0, .reason = .failed } }, .status = " 500 ", .header = "X-Silence-Redraws: 1\r\n" },
        .{ .max = 0, .body = valid, .turns = &.{.{ .frames = 4, .busy = 1 }}, .status = " 503 ", .header = null },
    };
    for (cases) |case| {
        var t: Turns = .{ .turns = case.turns };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{ .max = case.max }, null), case.deadline, case.body);
        defer scripted.done(r.s);
        try std.testing.expect(has(r.head, case.status));
        if (case.header) |h| try std.testing.expect(has(r.head, h)) else try std.testing.expect(!has(r.head, "X-Silence-Redraws"));
    }
    var t: Turns = .{ .turns = &.{} };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2 }, null), "soon", valid);
    defer scripted.done(r.s);
    try std.testing.expect(has(try r.s.readRest(), "X-Deadline-S must be a positive number of seconds"));
}

test "a redraw waiting on a busy engine gives up once it could no longer end within the budget" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // t + d is about 0.3 s, so the redraw is due; admitted later than 0.6 - d s from receipt, it would end too late
    const turns = [_]Turn{ .{ .frames = 5, .reason = .silence, .stall_ms = 150 }, .{ .frames = 4, .busy = 255 } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2, .budget_s = 0.6 }, null), null, try scripted.body(a, "One. Two.", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 1), t.seen);
    try std.testing.expect(t.refused > 0 and t.refused < 255);
    try std.testing.expect(has(r.head, "X-Finish-Reason: silence\r\nX-Silence-Redraws: 0\r\n"));
    try std.testing.expectEqualSlices(u8, try pcm(a, turns[0]), try r.s.readRest());
}

test "a long-form segment redrawn plain is heard without its cues by the segment after it" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "Sentence 00 is ok. Sentence [sighs] ok. Sentence 02 is ok.";
    const segments = try drama_long.split(a, input, 20);
    var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 3, .reason = .silence }, .{ .frames = 4 }, .{ .frames = 6 } } };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 1, .plain = true }, .{ .chars = 20 }), null, try scripted.body(a, input, false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 4), t.seen);
    // the redraw is the third submission: its codes are 300 + frame
    const codes = try a.alloc(drama_prompt.Codes, 4);
    for (codes, 0..) |*c, f| c.* = @splat(@intCast(300 + f));
    const heard = [_]drama_prompt.Reference{.{ .text = (try speech_redraw.plain(a, segments[1], .cues)).?, .frames = codes }};
    const want = try drama_prompt.encodeReferenced(a, text.text(), segments[2], &heard);
    try std.testing.expectEqualSlices(u32, want.tokens, t.prompts.items[3]);
    try std.testing.expectEqualSlices(drama_prompt.Codes, want.codes, t.codes.items[3]);
}

test "a redraw is never admitted past its latest start, though the engine would take it then" {
    const speech = @import("speech.zig");
    const clock = @import("clock.zig");
    const Conn = @import("http_conn.zig").Conn;
    const Mailbox = @import("mailbox.zig").Mailbox;
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var c = try Conn.init(gpa, pair[1], "test");
    defer c.deinit();
    var samples: [40]f32 = undefined;
    // the bound already past; or crossed in the 20 ms wait after a Busy that took 30 ms, the engine free by then
    for ([_]struct { after_ms: i96, turn: Turn }{
        .{ .after_ms = -1, .turn = .{ .frames = 4 } },
        .{ .after_ms = 40, .turn = .{ .frames = 4, .busy = 1, .stall_ms = 30 } },
    }) |case| {
        var t: Turns = .{ .turns = &.{case.turn} };
        defer t.deinit();
        var text: harness.Bytes = .{};
        const srv = try server.Server.init(gpa, std.testing.io, t.engine(), text.text(), redrawing(.{ .max = 1 }, null), null);
        defer srv.deinit();
        var result: api.SpeechResult = .{};
        const request: api.SpeechRequest = .{ .prompt = &.{1}, .pcm = &samples, .result = &result, .seed = 0, .max_frames = 4 };
        var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
        defer box.deinit();
        const start_by = clock.nowNs(srv.io) + case.after_ms * std.time.ns_per_ms;
        try std.testing.expectError(error.Busy, speech.submit(srv, &c, 1, &request, &box, start_by));
        try std.testing.expectEqual(@as(usize, 0), t.seen);
    }
}

test "--short-reply-ratio: a stop far short of its input is redrawn like a silence, a normal stop is not" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    // 246 characters are 15.8 s expected: under a ratio of 0.5, 5 frames (50 ms) are short and 900 (9 s) are not
    const input = try scripted.sentences(a, 13);
    const short = Turn{ .frames = 5 };
    const normal = Turn{ .frames = 900, .level = 0.3 };
    const cases = [_]struct { ratio: f64, input: []const u8, turns: []const Turn, sent: Turn, head: []const u8 }{
        .{ .ratio = 0.5, .input = input, .turns = &.{ short, normal }, .sent = normal, .head = "Content-Length: 18000\r\nX-Finish-Reason: stop\r\nX-Silence-Redraws: 1\r\n" },
        .{ .ratio = 0.5, .input = input, .turns = &.{normal}, .sent = normal, .head = "Content-Length: 18000\r\nX-Finish-Reason: stop\r\nX-Silence-Redraws: 0\r\n" },
        .{ .ratio = 0, .input = input, .turns = &.{short}, .sent = short, .head = "Content-Length: 100\r\nX-Finish-Reason: stop\r\nX-Silence-Redraws: 0\r\n" },
        // 151 characters returning 1 s: a ratio of 0.10, but under 200 characters a reply is not judged
        .{ .ratio = 0.5, .input = try scripted.sentences(a, 8), .turns = &.{.{ .frames = 100 }}, .sent = .{ .frames = 100 }, .head = "Content-Length: 2000\r\nX-Finish-Reason: stop\r\nX-Silence-Redraws: 0\r\n" },
    };
    for (cases) |case| {
        var t: Turns = .{ .turns = case.turns };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{ .max = 2, .short_ratio = case.ratio }, null), null, try scripted.body(a, case.input, false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(case.turns.len, t.seen);
        for (t.prompts.items, t.seeds.items, 0..) |prompt, seed, k| try std.testing.expectEqual(api.seedFor(prompt, @intCast(k)), seed);
        // all of it read first: reading more can move the buffer `r.head` points into
        _ = try r.s.readRest();
        const got = r.s.got.items;
        const want = try std.mem.concat(a, u8, &.{ "Content-Type: audio/pcm\r\n", case.head, "\r\n", try pcm(a, case.sent) });
        try std.testing.expectEqualStrings(want, got[std.mem.indexOf(u8, got, "Content-Type").?..]);
    }
    // a short trip redraws plain as a silence does
    var t: Turns = .{ .turns = &.{ short, normal } };
    defer t.deinit();
    const cued = try std.mem.concat(a, u8, &.{ input, " [pause] Done." });
    const r = try serve(&t, redrawing(.{ .max = 2, .plain = true, .short_ratio = 0.5 }, null), null, try scripted.body(a, cued, false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), try std.mem.concat(a, u8, &.{ input, " Done." })), t.prompts.items[1]);
}

test "--short-reply-ratio in a long-form whole reply: a short segment is redrawn in place" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // two segments of 246 characters, 15.8 s expected each: 5 frames are short, 900 are not
    var t: Turns = .{ .turns = &.{ .{ .frames = 900 }, .{ .frames = 5 }, .{ .frames = 900 } } };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2, .short_ratio = 0.5 }, .{ .chars = 250 }), null, try scripted.body(a, try scripted.sentences(a, 26), false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 3), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: stop\r\nX-Long-Form-Segments: 2;") and has(r.head, "X-Silence-Redraws: 1\r\n"));
    try std.testing.expectEqualSlices(u32, t.prompts.items[1], t.prompts.items[2]);
    try std.testing.expectEqual(api.seedFor(t.prompts.items[2], 1), t.seeds.items[2]);
}

test "--short-reply-ratio judges a plain redraw by the cue-free text it spoke" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // 221 characters with the cues, 189 without: the cue-free redraw is under 200 and is not judged
    const input = try std.mem.concat(a, u8, &.{ try scripted.sentences(a, 10), " [pause] [sighs] [laughs] [pause]" });
    var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 5 }, .{ .frames = 5 } } };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2, .plain = true, .short_ratio = 0.5 }, null), null, try scripted.body(a, input, false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: stop\r\nX-Silence-Redraws: 1\r\n"));
}

test "--silence-redraw-plain-scope controls: redraws speak the text without any control; nothing left, no redraw" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "Direction: warm\nLine: [pleased] It <emphasis>works</emphasis> now.";
    const escaped = "Direction: warm\\nLine: [pleased] It <emphasis>works</emphasis> now.";
    for ([_]speech_redraw.Scope{ .cues, .controls }) |scope| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5, .reason = .silence }, .{ .frames = 4 } } };
        defer t.deinit();
        const r = try serve(&t, redrawing(.{ .max = 2, .plain = true, .plain_scope = scope }, null), null, try scripted.body(a, escaped, false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, 2), t.seen);
        // with cues, these controls are no cue: the redraw speaks the input as sent
        const said = if (scope == .cues) input else "It works now.";
        try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), said), t.prompts.items[1]);
        try std.testing.expectEqual(api.seedFor(t.prompts.items[1], 1), t.seeds.items[1]);
    }
    const turns = [_]Turn{ .{ .frames = 5, .reason = .silence }, .{ .frames = 4 } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2, .plain = true, .plain_scope = .controls }, null), null, try scripted.body(a, "[pleased] <soft></soft>", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 1), t.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: silence\r\nX-Silence-Redraws: 0\r\n"));
    try std.testing.expectEqualSlices(u8, try pcm(a, turns[0]), try r.s.readRest());
}

/// The prompt a long-form reply's plain redraw of segment 2 sent, when segment 2 is silenced once (no history heard).
fn redrawnSecond(a: std.mem.Allocator, escaped: []const u8, chars: u32) ![]const u32 {
    var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 3, .reason = .silence }, .{ .frames = 4 }, .{ .frames = 4 } } };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 1, .plain = true, .plain_scope = .controls }, .{ .chars = chars, .history = 0 }), null, try scripted.body(a, escaped, false));
    defer scripted.done(r.s);
    try std.testing.expect(t.seen >= 3);
    return a.dupe(u32, t.prompts.items[2]);
}

test "long-form controls: a later segment that opens like a director's note keeps it, as the whole input would" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "Sentence 00 is ok, and it is long enough.\nDirection: d\nLine: Sentence 01 [pause] ok.";
    const segments = try drama_long.split(a, input, 45);
    try std.testing.expectEqual(@as(usize, 2), segments.len);
    try std.testing.expect(std.mem.startsWith(u8, segments[1], "Direction: d\nLine: "));
    const want = try drama_prompt.encode(a, text.text(), "Direction: d Line: Sentence 01 ok.");
    try std.testing.expectEqualSlices(u32, want, try redrawnSecond(a, "Sentence 00 is ok, and it is long enough.\\nDirection: d\\nLine: Sentence 01 [pause] ok.", 45));
}

test "long-form controls: a bracket span that a segment cut crosses is dropped whole, from both segments" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "Sentence 00 is ok. [warmly. then] Sentence 01 ok.";
    const segments = try drama_long.split(a, input, 30);
    try std.testing.expectEqual(@as(usize, 2), segments.len);
    try std.testing.expect(std.mem.endsWith(u8, segments[0], "[warmly.") and std.mem.startsWith(u8, segments[1], "then]"));
    const want = try drama_prompt.encode(a, text.text(), "Sentence 01 ok.");
    try std.testing.expectEqualSlices(u32, want, try redrawnSecond(a, input, 30));
}

test "long-form controls: the input's spans, found at the first plain redraw, serve every later segment's too" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = "Sentence 00 [pause] ok. Sentence 01 <soft>is</soft> ok.";
    const segments = try drama_long.split(a, input, 35);
    try std.testing.expectEqual(@as(usize, 2), segments.len);
    var t: Turns = .{ .turns = &.{ .{ .frames = 3, .reason = .silence }, .{ .frames = 4 }, .{ .frames = 3, .reason = .silence }, .{ .frames = 4 } } };
    defer t.deinit();
    const r = try serve(&t, redrawing(.{ .max = 2, .plain = true, .plain_scope = .controls }, .{ .chars = 35, .history = 0 }), null, try scripted.body(a, input, false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 4), t.seen);
    try std.testing.expect(has(r.head, "X-Silence-Redraws: 2\r\n"));
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "Sentence 00 ok."), t.prompts.items[1]);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "Sentence 01 is ok."), t.prompts.items[3]);
}
