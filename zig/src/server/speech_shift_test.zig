//! --shift-pause-s on the wire, against the scripted engine: `¶` cuts, the pauses made there and the spans sent.
const std = @import("std");
const server = @import("root.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_long = @import("drama_long.zig");
const speech_redraw = @import("speech_redraw.zig");
const harness = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");
const Turn = scripted.Turn;
const Turns = scripted.Turns;

const gpa = std.testing.allocator;
/// 0.85 s at the scripted 1 kHz: 850 samples of quiet at each shift.
const shifting: drama_long.Options = .{ .shift_pause_s = 0.85 };

fn config(long_form: ?drama_long.Options, redraw: speech_redraw.Policy) server.Config {
    var cfg = scripted.config(long_form);
    cfg.silence_redraw = redraw;
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

/// No prompt holds a byte of the mark (U+00B6 is C2 B6; the test tokenizer's ids are bytes).
fn expectNoMark(t: *const Turns) !void {
    for (t.prompts.items) |prompt| for (prompt) |id| try std.testing.expect(id != 0xc2 and id != 0xb6);
}

test "shifts off: an input with marks is served byte for byte as the one without them, whole or long-form" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pairs = [_][2][]const u8{
        .{ "One. \u{b6} Two \u{b6} three.", "One. Two three." },
        .{ "Sentence 00 is ok. \u{b6} \u{b6} Sentence 01 \u{b6}is ok.", "Sentence 00 is ok. Sentence 01 is ok." },
    };
    for ([_]?drama_long.Options{ null, .{ .chars = 20 } }) |long_form| for ([_]bool{ false, true }) |stream| for (pairs) |pair| {
        var replies: [2][]const u8 = undefined;
        var prompts: [2][]const []const u32 = undefined;
        for (pair, 0..) |input, k| {
            var t: Turns = .{ .turns = &.{ .{ .frames = 5, .tail = 2 }, .{ .frames = 4, .lead = 1 } } };
            defer t.deinit();
            replies[k] = try reply(a, &t, config(long_form, .{}), input, stream);
            // the turns' arena goes with them: the prompts are copied out
            const copied = try a.alloc([]const u32, t.prompts.items.len);
            for (copied, t.prompts.items) |*c, p| c.* = try a.dupe(u32, p);
            prompts[k] = copied;
            try expectNoMark(&t);
        }
        try std.testing.expectEqualStrings(replies[1], replies[0]);
        try std.testing.expectEqual(prompts[1].len, prompts[0].len);
        for (prompts[0], prompts[1]) |got, want| try std.testing.expectEqualSlices(u32, want, got);
        try std.testing.expect(!has(replies[0], "X-Pause-Spans"));
    };
}

test "shifts on: a mark cuts only after a sentence end and before words, even in an input that fits one segment" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 4 } } };
    defer t.deinit();
    const got = try reply(a, &t, config(shifting, .{}), "One \u{b6} two. \u{b6} Three. \u{b6}", false);
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "One two."), t.prompts.items[0]);
    try std.testing.expect(has(got, "X-Long-Form-Segments: 2;") and has(got, "X-Pause-Spans: "));
    // no mark it honours: one synthesis, as sent without them
    var one: Turns = .{ .turns = &.{.{ .frames = 5 }} };
    defer one.deinit();
    const single = try reply(a, &one, config(shifting, .{}), "One, \u{b6} two. Three \u{b6} four.", false);
    try std.testing.expectEqual(@as(usize, 1), one.seen);
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), "One, two. Three four."), one.prompts.items[0]);
    try std.testing.expect(!has(single, "X-Pause-Spans") and !has(single, "X-Long-Form-Segments"));
}

/// Two shifts: the first segment ends just under the floor (-42.5 dBFS), the second is loud at both seams.
const three = [_]Turn{ .{ .frames = 30, .lead = 12, .tail = 12, .hush = 0.0075 }, .{ .frames = 30 }, .{ .frames = 20 } };
const three_input = "One. \u{b6} Two. \u{b6} Three.";

/// The value of header (or trailer) `name` in `got`.
fn field(got: []const u8, name: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, got, name).? + name.len + 2;
    return got[at..std.mem.indexOfPos(u8, got, at, "\r\n").?];
}

/// X-Pause-Spans' ``start-end`` pairs.
fn spans(a: std.mem.Allocator, value: []const u8) ![]const [2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    var it = std.mem.splitScalar(u8, value, ',');
    while (it.next()) |span| {
        const dash = std.mem.indexOfScalar(u8, span, '-').?;
        try out.append(a, .{ try std.fmt.parseInt(usize, span[0..dash], 10), try std.fmt.parseInt(usize, span[dash + 1 ..], 10) });
    }
    return out.items;
}

/// The PCM16 sample at `i` of a whole PCM reply's body.
fn sample(body: []const u8, i: usize) i16 {
    return std.mem.readInt(i16, body[2 * i ..][0..2], .little);
}

/// The longest run of silent 10-sample frames (the 1 kHz test rate's 10 ms) inside [from, to), framed from the start.
fn longestSilent(body: []const u8, from: usize, to: usize) usize {
    var best: usize = 0;
    var run: usize = 0;
    var k = (from + 9) / 10;
    while ((k + 1) * 10 <= to) : (k += 1) {
        var sum: f64 = 0;
        for (k * 10..k * 10 + 10) |i| {
            const v = @as(f64, @floatFromInt(sample(body, i))) / 32767.0;
            sum += v * v;
        }
        run = if (10 * std.math.log10(sum / 10) < -42) run + 10 else 0;
        best = @max(best, run);
    }
    return best;
}

/// A whole PCM reply's body, after its head.
fn bodyOf(got: []const u8) []const u8 {
    return got[std.mem.indexOf(u8, got, "\r\n\r\n").? + 4 ..];
}

test "shifts on: in the audio as sent, each span holds T of silent frames, though the seams are loud and the tone near the floor" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var t: Turns = .{ .turns = &three };
    defer t.deinit();
    const got = try reply(a, &t, config(shifting, .{}), three_input, false);
    const body = bodyOf(got);
    const starts = field(got, "X-Long-Form-Segments")["3;".len..];
    const second = try std.fmt.parseInt(usize, starts[0..std.mem.indexOfScalar(u8, starts, ',').?], 10);
    const third = try std.fmt.parseInt(usize, starts[std.mem.indexOfScalar(u8, starts, ',').? + 1 ..], 10);
    const found = try spans(a, field(got, "X-Pause-Spans"));
    try std.testing.expectEqual(@as(usize, 2), found.len);
    for (found) |span| try std.testing.expect(longestSilent(body, span[0], span[1]) >= 850);
    // each segment as made, the fills between them never a zero sample
    for ([_]usize{ 0, second, third }, three) |start, turn| {
        var want: std.ArrayList(u8) = .empty;
        try scripted.turnPcm(&want, a, turn, 1);
        try std.testing.expectEqualSlices(u8, want.items, body[2 * start ..][0..want.items.len]);
    }
    for ([_][2]usize{ .{ 300, second }, .{ second + 300, third } }) |fill| for (fill[0]..fill[1]) |i| try std.testing.expect(sample(body, i) != 0);
    // the tone is the first segment's near-floor quiet, brought 6 dB under the floor at its peak (PCM16 130)
    for ([_]usize{ (300 + second) / 2, (second + 300 + third) / 2 }) |middle| try std.testing.expectEqual(@as(i16, 130), sample(body, middle));
    try std.testing.expectEqual(2 * (third + 200), body.len);
}

test "shifts on: with no quiet before a mark the tone is the next segment's, and with none anywhere low noise, never zeros" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 12, 0 }) |lead| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 30 }, .{ .frames = 30, .lead = lead } } };
        defer t.deinit();
        const got = try reply(a, &t, config(shifting, .{}), "One. \u{b6} Two.", false);
        const body = bodyOf(got);
        const span = (try spans(a, field(got, "X-Pause-Spans")))[0];
        try std.testing.expect(longestSilent(body, span[0], span[1]) >= 850);
        const second = try std.fmt.parseInt(usize, field(got, "X-Long-Form-Segments")["2;".len..], 10);
        // noise may quantize a sample to zero, never a frame of them
        var zeros: usize = 0;
        for (300..second) |i| {
            zeros = if (sample(body, i) == 0) zeros + 1 else 0;
            try std.testing.expect(zeros < 10);
        }
        // the next segment's quiet head (0.0005, PCM16 16), or uniform noise at -65 dBFS (peaks under PCM16 32)
        const middle = sample(body, (300 + second) / 2);
        try std.testing.expect(if (lead > 0) middle == 16 else @abs(middle) < 32);
    }
}

test "shifts on: a stream declares X-Pause-Spans and ends with the whole reply's spans beside its segments" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var whole: Turns = .{ .turns = &three };
    defer whole.deinit();
    const sent = try reply(a, &whole, config(shifting, .{}), three_input, false);
    var t: Turns = .{ .turns = &three };
    defer t.deinit();
    const got = try reply(a, &t, config(shifting, .{}), three_input, true);
    try std.testing.expect(has(got, "Trailer: X-Finish-Reason, X-Long-Form-Segments, X-Pause-Spans\r\n"));
    const end = try std.fmt.allocPrint(a, "\r\n0\r\nX-Long-Form-Segments: {s}\r\nX-Pause-Spans: {s}\r\n\r\n", .{ field(sent, "X-Long-Form-Segments"), field(sent, "X-Pause-Spans") });
    try std.testing.expect(std.mem.endsWith(u8, got, end));
    // shifts on, an input with no mark: the trailer is declared and not sent
    var plain: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 3 } } };
    defer plain.deinit();
    var cfg = config(shifting, .{});
    cfg.long_form.?.chars = 20;
    const unmarked = try reply(a, &plain, cfg, try scripted.sentences(a, 2), true);
    try std.testing.expect(std.mem.endsWith(u8, unmarked, "\r\n0\r\nX-Long-Form-Segments: 2;470\r\n\r\n"));
}

test "shifts on: a segment after a mark redrawn plain speaks no mark and no cue, and its span follows the final take" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var t: Turns = .{ .turns = &.{ .{ .frames = 30, .tail = 12 }, .{ .frames = 30, .reason = .silence }, .{ .frames = 20, .lead = 3 } } };
    defer t.deinit();
    const got = try reply(a, &t, config(shifting, .{ .max = 1, .plain = true }), "One. \u{b6} Two [sighs] more.", false);
    try std.testing.expectEqual(@as(usize, 3), t.seen);
    try expectNoMark(&t);
    const said: []const u32 = &.{ 'T', 'w', 'o', ' ', 'm', 'o', 'r', 'e', '.' };
    try std.testing.expect(std.mem.indexOf(u32, t.prompts.items[2], said) != null);
    // the redraw's 30 leading quiet samples count: 850 - 120 - 30 added, the span to its speech
    try std.testing.expect(has(got, "X-Long-Form-Segments: 2;1000\r\nX-Pause-Spans: 180-1030\r\nX-Silence-Redraws: 1\r\n"));
}

test "the short-reply rule counts the input without its marks" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // 199 characters without the mark, 201 with it: under the rule's 200, so a short stop is not redrawn
    for ([_]usize{ 97, 98 }, [_]usize{ 1, 2 }) |xs, draws| {
        var text: std.ArrayList(u8) = .empty;
        try text.appendNTimes(a, 'x', xs);
        try text.appendSlice(a, ". \u{b6} ");
        try text.appendNTimes(a, 'y', 99);
        try text.append(a, '.');
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 5 } } };
        defer t.deinit();
        _ = try reply(a, &t, config(null, .{ .max = 1, .short_ratio = 0.5 }), text.items, false);
        try std.testing.expectEqual(draws, t.seen);
    }
}

test "shifts on: quiet already long enough is kept whole, and a reply cut short before a mark sends no spans" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // 900 samples of quiet end the first segment: nothing is added, and the span is that quiet
    var t: Turns = .{ .turns = &.{ .{ .frames = 100, .tail = 90 }, .{ .frames = 20 } } };
    defer t.deinit();
    const got = try reply(a, &t, config(shifting, .{}), "One. \u{b6} Two.", false);
    try std.testing.expect(has(got, "Content-Length: 2400\r\nX-Finish-Reason: stop\r\nX-Long-Form-Segments: 2;1000\r\nX-Pause-Spans: 100-1000\r\n"));
    var cut: Turns = .{ .turns = &.{ .{ .frames = 30, .reason = .silence }, .{ .frames = 20 } } };
    defer cut.deinit();
    const short = try reply(a, &cut, config(shifting, .{}), "One. \u{b6} Two.", false);
    try std.testing.expectEqual(@as(usize, 1), cut.seen);
    try std.testing.expect(has(short, "X-Finish-Reason: silence\r\nX-Long-Form-Segments: 1;\r\n\r\n") and !has(short, "X-Pause-Spans"));
}
