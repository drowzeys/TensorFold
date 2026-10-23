//! Long-form speech on the wire, against an engine that scripts each segment's frames and finish reason.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_long = @import("drama_long.zig");
const harness = @import("speech_stream_test.zig");
const Served = harness.Served;

const gpa = std.testing.allocator;
/// 1 kHz audio in 10-sample frames: a codec frame is one 10 ms join frame, and a sentence's pause is 420 samples.
const rate = 1000;
const frame = 10;
const quiet: f32 = 0.0005;

/// A segment: `frames` of audio (`lead`, `tail` at `hush`, the rest at `level`); `busy` refusals; `stall_ms`; `hollow`.
pub const Turn = struct { frames: usize, reason: api.Reason = .stop, level: f32 = 0.5, lead: usize = 0, tail: usize = 0, hush: f32 = quiet, busy: u8 = 0, stall_ms: u32 = 0, hollow: usize = 0 };

/// One synthesis a turn, run inside submit; turn n's codes are n * 100 + frame.
pub const Turns = struct {
    turns: []const Turn,
    max_prompt: u32 = 4096,
    seen: usize = 0,
    refused: u8 = 0,
    arena: std.heap.ArenaAllocator = .init(gpa),
    prompts: std.ArrayList([]const u32) = .empty,
    codes: std.ArrayList([]const drama_prompt.Codes) = .empty,
    seeds: std.ArrayList(u64) = .empty,
    limits: std.ArrayList(u32) = .empty,
    generated: std.ArrayList(usize) = .empty,
    streamed: std.ArrayList(bool) = .empty,
    unconditioned: std.ArrayList([]const u32) = .empty,
    gammas: std.ArrayList(f32) = .empty,

    pub fn engine(t: *Turns) api.Engine {
        return .{ .ctx = t, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .submit_speech = speech } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const t: *Turns = @ptrCast(@alignCast(ctx));
        return .{ .text_generation = false, .speech = .{ .sample_rate = rate, .frame_length = frame, .max_frames = 1024, .max_prompt_tokens = t.max_prompt } };
    }
    fn speech(ctx: *anyopaque, id: api.Id, request: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const t: *Turns = @ptrCast(@alignCast(ctx));
        if (t.seen == t.turns.len) return error.InvalidRequest;
        const turn = t.turns[t.seen];
        std.Io.sleep(std.testing.io, .fromMilliseconds(turn.stall_ms), .awake) catch {};
        if (t.refused < turn.busy) {
            t.refused += 1;
            return error.Busy;
        }
        t.refused = 0;
        t.seen += 1;
        const a = t.arena.allocator();
        t.prompts.append(a, a.dupe(u32, request.prompt) catch unreachable) catch unreachable;
        t.codes.append(a, a.dupe(drama_prompt.Codes, request.codes) catch unreachable) catch unreachable;
        t.seeds.append(a, request.seed) catch unreachable;
        t.limits.append(a, request.max_frames) catch unreachable;
        t.generated.append(a, request.generated.len) catch unreachable;
        t.streamed.append(a, request.stream) catch unreachable;
        t.unconditioned.append(a, a.dupe(u32, request.unconditioned) catch unreachable) catch unreachable;
        t.gammas.append(a, request.cfg_gamma) catch unreachable;
        const frames = @min(turn.frames, request.max_frames);
        fill(request.pcm[0 .. frames * frame], turn);
        if (request.generated.len > 0) for (request.generated[0..frames], 0..) |*codes, f| {
            codes.* = @splat(@intCast(t.seen * 100 + f));
        };
        if (request.stream and frames > 0) sink.event(sink.ctx, id, &.{ .audio = frames * frame });
        const reason: api.Reason = if (turn.reason == .stop and frames == request.max_frames) .length else turn.reason;
        if (reason == .stop or reason == .length or reason == .silence) request.result.samples = frames * frame;
        request.result.hollow = turn.hollow;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = reason, .message = if (reason == .failed) "scripted failure" else "" } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Unsupported;
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    pub fn deinit(t: *Turns) void {
        t.arena.deinit();
    }
};

fn fill(pcm: []f32, turn: Turn) void {
    @memset(pcm, turn.level);
    @memset(pcm[0..@min(pcm.len, turn.lead * frame)], turn.hush);
    @memset(pcm[pcm.len -| turn.tail * frame..], turn.hush);
}

pub fn config(long_form: ?drama_long.Options) server.Config {
    return .{ .served_name = "m", .model_ids = &.{"m"}, .long_form = long_form };
}

/// `n` sentences of 18 characters: with 20-character segments, one a segment.
pub fn sentences(a: std.mem.Allocator, n: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (0..n) |i| try out.print(a, "{s}Sentence {d:0>2} is ok.", .{ if (i == 0) "" else " ", i });
    return out.items;
}

pub fn body(a: std.mem.Allocator, input: []const u8, stream: bool) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"{s}\",\"response_format\":\"pcm\",\"stream\":{}}}", .{ input, stream });
}

/// Served once, the route joined; the reply head and the bytes after it.
fn serve(t: *Turns, cfg: server.Config, request: []const u8) !struct { s: *Served, head: []const u8 } {
    const s = try gpa.create(Served);
    try s.startWith(t.engine(), cfg, "HTTP/1.1", request, 2000);
    const head = try s.readHead();
    s.route.join();
    return .{ .s = s, .head = head };
}

pub fn done(s: *Served) void {
    s.deinit();
    gpa.destroy(s);
}

fn appendPcm(out: *std.ArrayList(u8), a: std.mem.Allocator, pcm: []const f32) !void {
    for (pcm) |x| try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(i16, @as(i16, @intFromFloat(x * 32767.0)))));
}

/// The PCM16 of `turn`'s audio scaled by `gain`.
pub fn turnPcm(out: *std.ArrayList(u8), a: std.mem.Allocator, turn: Turn, gain: f32) !void {
    const pcm = try a.alloc(f32, turn.frames * frame);
    fill(pcm, turn);
    for (pcm) |*x| x.* *= gain;
    try appendPcm(out, a, pcm);
}

pub fn zeros(out: *std.ArrayList(u8), a: std.mem.Allocator, samples: usize) !void {
    try out.appendNTimes(a, 0, 2 * samples);
}

/// Segment `i`'s expected prompt: the earlier segments' text and scripted codes as references, latest last.
fn expected(a: std.mem.Allocator, text: *harness.Bytes, segments: []const []const u8, frames: []const usize, i: usize, keep: usize) !drama_prompt.Rows {
    if (keep == 0) return .{ .tokens = try drama_prompt.encode(a, text.text(), segments[i]), .codes = &.{} };
    var heard: std.ArrayList(drama_prompt.Reference) = .empty;
    for (i - keep..i) |j| {
        const codes = try a.alloc(drama_prompt.Codes, frames[j]);
        for (codes, 0..) |*c, f| c.* = @splat(@intCast((j + 1) * 100 + f));
        try heard.append(a, .{ .text = segments[j], .frames = codes });
    }
    return drama_prompt.encodeReferenced(a, text.text(), segments[i], heard.items);
}

test "long-form off: a long input is one synthesis, and an input of one segment is that same synthesis with it on" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    for ([_][]const u8{ try sentences(a, 3), "Short enough." }) |input| {
        var replies: [2][]const u8 = undefined;
        var prompts: [2][]const u32 = undefined;
        for ([_]?drama_long.Options{ null, .{ .chars = 300 } }, 0..) |lf, k| {
            var t: Turns = .{ .turns = &.{.{ .frames = 5, .level = 0.3, .tail = 2 }} };
            defer t.deinit();
            const r = try serve(&t, config(lf), try body(a, input, false));
            defer done(r.s);
            try std.testing.expectEqual(@as(usize, 1), t.seen);
            try std.testing.expect(std.mem.indexOf(u8, r.head, "X-Long-Form-Segments") == null);
            try std.testing.expectEqual(@as(usize, 0), t.generated.items[0]);
            try std.testing.expectEqual(api.seedFor(t.prompts.items[0], 0), t.seeds.items[0]);
            prompts[k] = try a.dupe(u32, t.prompts.items[0]);
            replies[k] = try std.mem.concat(a, u8, &.{ r.head[std.mem.indexOf(u8, r.head, "Content-Type").?..], try r.s.readRest() });
        }
        try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text.text(), input), prompts[0]);
        try std.testing.expectEqualSlices(u32, prompts[0], prompts[1]);
        try std.testing.expectEqualStrings(replies[0], replies[1]);
    }
}

test "segments are spoken in order, each hearing the one before; joins without silence get the whole pause" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = try sentences(a, 3);
    const segments = try drama_long.split(a, input, 20);
    const turns = [_]Turn{ .{ .frames = 5 }, .{ .frames = 3 }, .{ .frames = 7 } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, config(.{ .chars = 20 }), try body(a, input, false));
    defer done(r.s);
    try std.testing.expect(std.mem.indexOf(u8, r.head, "HTTP/1.1 200") != null);
    // speech to the last sample has no quiet tail: each sentence pause is 420 samples of zeros
    try std.testing.expect(std.mem.indexOf(u8, r.head, "X-Finish-Reason: stop\r\nX-Long-Form-Segments: 3;470,920\r\n") != null);
    var want: std.ArrayList(u8) = .empty;
    for (turns, 0..) |turn, i| {
        if (i > 0) try zeros(&want, a, 420);
        try turnPcm(&want, a, turn, 1);
    }
    try std.testing.expectEqualSlices(u8, want.items, try r.s.readRest());
    try std.testing.expectEqual(@as(usize, 3), t.seen);
    const frames = [_]usize{ 5, 3, 7 };
    for (0..3) |i| {
        const rows = try expected(a, &text, segments, &frames, i, @min(i, 1));
        try std.testing.expectEqualSlices(u32, rows.tokens, t.prompts.items[i]);
        try std.testing.expectEqualSlices(drama_prompt.Codes, rows.codes, t.codes.items[i]);
        try std.testing.expectEqual(api.seedFor(rows.tokens, 0), t.seeds.items[i]);
        try std.testing.expect(t.generated.items[i] >= t.limits.items[i]);
    }
}

test "a join adds only the shortfall, as room tone from the segment before; a long silence is kept whole" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // 150 ms of tail and 50 ms of head silence: 220 samples of the first segment's tail tone are added
    const turns = [_]Turn{ .{ .frames = 30, .tail = 15 }, .{ .frames = 70, .lead = 5, .tail = 60 }, .{ .frames = 10 } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, config(.{ .chars = 20 }), try body(a, try sentences(a, 3), false));
    defer done(r.s);
    // the second join has 600 ms of tail: nothing is added before the third segment
    try std.testing.expect(std.mem.indexOf(u8, r.head, "X-Long-Form-Segments: 3;520,1220\r\n") != null);
    var want: std.ArrayList(u8) = .empty;
    try turnPcm(&want, a, turns[0], 1);
    const tone: [220]f32 = @splat(quiet);
    try appendPcm(&want, a, &tone);
    try turnPcm(&want, a, turns[1], 1);
    try turnPcm(&want, a, turns[2], 1);
    try std.testing.expectEqualSlices(u8, want.items, try r.s.readRest());
}

test "each join's pause comes from its punctuation or a turn in the text, a blank line's the longest" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { input: []const u8, chars: u32, want: []const u8 }{
        // a question or a sentence opening a new move is a turn in the text: a 0.60 s pause
        .{ .input = "Is that right, then? Yes, it is.", .chars = 20, .want = "2;610" },
        .{ .input = "Sentence 00 is ok. So it is ok, fine.", .chars = 20, .want = "2;610" },
        .{ .input = "And so it goes\u{2026} Then more.", .chars = 16, .want = "2;560" },
        // the paragraph ends the first segment though the next sentence would fit after it
        .{ .input = "Sentence 00 is ok.\\n\\nSentence 01 is ok. Sentence 02 is ok.", .chars = 40, .want = "2;710" },
        .{ .input = "one two three four five six", .chars = 14, .want = "2;210" },
    };
    for (cases) |case| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 1 }, .{ .frames = 1 } } };
        defer t.deinit();
        const r = try serve(&t, config(.{ .chars = case.chars }), try body(a, case.input, false));
        defer done(r.s);
        std.testing.expect(std.mem.indexOf(u8, r.head, try std.fmt.allocPrint(a, "X-Long-Form-Segments: {s}\r\n", .{case.want})) != null) catch |e| {
            std.debug.print("{s}\n{s}\n", .{ case.input, r.head });
            return e;
        };
    }
}

test "a segment off the earlier level gets one gain, clamped to 3 dB" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // -6 dB from the first: raised by the 3 dB clamp; -1 dB from the median of the first two: left alone
    const turns = [_]Turn{ .{ .frames = 4, .level = 0.5 }, .{ .frames = 4, .level = 0.25 }, .{ .frames = 4, .level = 0.4 } };
    var t: Turns = .{ .turns = &turns };
    defer t.deinit();
    const r = try serve(&t, config(.{ .chars = 20 }), try body(a, try sentences(a, 3), false));
    defer done(r.s);
    var want: std.ArrayList(u8) = .empty;
    try turnPcm(&want, a, turns[0], 1);
    try zeros(&want, a, 420);
    try turnPcm(&want, a, turns[1], std.math.pow(f32, 10, 3.0 / 20.0));
    try zeros(&want, a, 420);
    try turnPcm(&want, a, turns[2], 1);
    try std.testing.expectEqualSlices(u8, want.items, try r.s.readRest());
}

test "history 2 hears two earlier segments, dropping the oldest when the prompt would not fit" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var text: harness.Bytes = .{};
    const input = try sentences(a, 3);
    const segments = try drama_long.split(a, input, 20);
    const frames = [_]usize{ 5, 3, 7 };
    const two = try expected(a, &text, segments, &frames, 2, 2);
    const one = try expected(a, &text, segments, &frames, 2, 1);
    for ([_]u32{ 4096, @intCast(two.tokens.len - 1) }, [_]drama_prompt.Rows{ two, one }) |max_prompt, want| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 3 }, .{ .frames = 7 } }, .max_prompt = max_prompt };
        defer t.deinit();
        const r = try serve(&t, config(.{ .chars = 20, .history = 2 }), try body(a, input, false));
        defer done(r.s);
        try std.testing.expectEqual(@as(usize, 3), t.seen);
        try std.testing.expectEqualSlices(u32, want.tokens, t.prompts.items[2]);
    }
}

test "a clipped or silenced segment ends the reply with its reason; later segments are not spoken" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]api.Reason{ .length, .silence }) |cut| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 3, .reason = cut }, .{ .frames = 7 } } };
        defer t.deinit();
        const r = try serve(&t, config(.{ .chars = 20 }), try body(a, try sentences(a, 3), false));
        defer done(r.s);
        try std.testing.expectEqual(@as(usize, 2), t.seen);
        const head = try std.fmt.allocPrint(a, "X-Finish-Reason: {s}\r\nX-Long-Form-Segments: 2;470\r\n", .{@tagName(cut)});
        try std.testing.expect(std.mem.indexOf(u8, r.head, head) != null);
        try std.testing.expectEqual(@as(usize, 2 * (frame * 8 + 420)), (try r.s.readRest()).len);
    }
}

test "the frame budget: each segment gets what is left, and a reply that spends it ends with length" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const thousand: Turn = .{ .frames = 1000 };
    const cases = [_]struct { turns: []const Turn, seen: usize, last_limit: u32 }{
        .{ .turns = &.{ thousand, thousand, thousand, thousand, thousand, thousand, thousand }, .seen = 6, .last_limit = 1000 },
        .{ .turns = &.{ thousand, thousand, thousand, thousand, thousand, .{ .frames = 1500 }, thousand }, .seen = 6, .last_limit = 1000 },
    };
    for (cases) |case| {
        var t: Turns = .{ .turns = case.turns };
        defer t.deinit();
        const r = try serve(&t, config(.{ .chars = 20 }), try body(a, try sentences(a, 7), false));
        defer done(r.s);
        try std.testing.expectEqual(case.seen, t.seen);
        try std.testing.expectEqual(@as(u32, 1024), t.limits.items[0]);
        try std.testing.expectEqual(case.last_limit, t.limits.items[t.seen - 1]);
        try std.testing.expect(std.mem.indexOf(u8, r.head, "X-Finish-Reason: length\r\n") != null);
        try std.testing.expectEqual(@as(usize, 2 * (frame * drama_long.max_frames + 5 * 420)), (try r.s.readRest()).len);
    }
}

test "more than 12 segments are refused before any synthesis" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var t: Turns = .{ .turns = &.{} };
    defer t.deinit();
    const r = try serve(&t, config(.{ .chars = 20 }), try body(a, try sentences(a, 13), false));
    defer done(r.s);
    try std.testing.expect(std.mem.indexOf(u8, r.head, " 400 ") != null);
    try std.testing.expect(std.mem.indexOf(u8, try r.s.readRest(), "Drama 3 request: a long-form input in 13 segments of at most 20 characters exceeds the maximum of 12 segments") != null);
    try std.testing.expectEqual(@as(usize, 0), t.seen);
}

test "a long-form stream sends the first segment as it is made, then each later one with its pause, then trailers" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]api.Reason{ .stop, .length, .silence }) |last| {
        const turns = [_]Turn{ .{ .frames = 5 }, .{ .frames = 3, .reason = last } };
        var t: Turns = .{ .turns = &turns };
        defer t.deinit();
        const r = try serve(&t, config(.{ .chars = 20 }), try body(a, try sentences(a, 2), true));
        defer done(r.s);
        try std.testing.expect(t.streamed.items[0] and !t.streamed.items[1]);
        try std.testing.expect(std.mem.indexOf(u8, r.head, "Trailer: X-Finish-Reason, X-Long-Form-Segments\r\n") != null);
        var want: std.ArrayList(u8) = .empty;
        try want.print(a, "{x}\r\n", .{2 * frame * 5});
        try turnPcm(&want, a, turns[0], 1);
        try want.print(a, "\r\n{x}\r\n", .{2 * (420 + frame * 3)});
        try zeros(&want, a, 420);
        try turnPcm(&want, a, turns[1], 1);
        const reason = if (last == .stop) "" else try std.fmt.allocPrint(a, "X-Finish-Reason: {s}\r\n", .{@tagName(last)});
        try want.print(a, "\r\n0\r\n{s}X-Long-Form-Segments: 2;470\r\n\r\n", .{reason});
        try std.testing.expectEqualSlices(u8, want.items, try r.s.readRest());
    }
}

test "a failure after the first segment ends a stream incomplete and a whole reply with an error" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ true, false }) |stream| {
        var t: Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 0, .reason = .failed } } };
        defer t.deinit();
        const r = try serve(&t, config(.{ .chars = 20 }), try body(a, try sentences(a, 2), stream));
        defer done(r.s);
        const rest = try r.s.readRest();
        if (stream) {
            try std.testing.expect(std.mem.indexOf(u8, r.head, "HTTP/1.1 200") != null and std.mem.indexOf(u8, rest, "\r\n0\r\n") == null);
        } else {
            try std.testing.expect(std.mem.indexOf(u8, r.head, "HTTP/1.1 500") != null and std.mem.indexOf(u8, rest, "scripted failure") != null);
        }
    }
}
