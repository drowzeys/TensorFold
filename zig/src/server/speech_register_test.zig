//! --register-redraw-st on the wire, against an engine scripting each draw's audio: what trips, redraws and is sent.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const speech_redraw = @import("speech_redraw.zig");
const pcm16 = @import("speech.zig").pcm16;
const speech_register = @import("speech_register.zig");
const harness = @import("speech_stream_test.zig");
const drama_register = @import("drama_register.zig");
const scripted = @import("speech_long_test.zig");
const tone = @import("drama_register_test.zig");

const gpa = std.testing.allocator;
const rate = 16000;
const frame = 160;
const on: speech_register.Policy = .{ .st = 4 };

/// A draw: its audio and finish; refused busy `busy` times first, each try after `stall_ms`; `hollow`: its mid hole.
const Draw = struct { pcm: []const f32, reason: api.Reason = .stop, busy: u8 = 0, stall_ms: u32 = 0, hollow: usize = 0 };

/// One synthesis a draw, run inside submit; each draw's prompt and seed kept in `a`.
const Draws = struct {
    a: std.mem.Allocator,
    draws: []const Draw,
    seen: usize = 0,
    refused: u8 = 0,
    prompts: [4][]const u32 = undefined,
    seeds: [4]u64 = undefined,

    fn engine(d: *Draws) api.Engine {
        return .{ .ctx = d, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .submit_speech = speech } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .text_generation = false, .speech = .{ .sample_rate = rate, .frame_length = frame, .max_frames = 1000, .max_prompt_tokens = 4096 } };
    }
    fn speech(ctx: *anyopaque, id: api.Id, request: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const d: *Draws = @ptrCast(@alignCast(ctx));
        if (d.seen == d.draws.len) return error.InvalidRequest;
        const draw = d.draws[d.seen];
        std.Io.sleep(std.testing.io, .fromMilliseconds(draw.stall_ms), .awake) catch {};
        if (d.refused < draw.busy) {
            d.refused += 1;
            return error.Busy;
        }
        d.refused = 0;
        d.prompts[d.seen] = d.a.dupe(u32, request.prompt) catch unreachable;
        d.seeds[d.seen] = request.seed;
        d.seen += 1;
        @memcpy(request.pcm[0..draw.pcm.len], draw.pcm);
        if (request.stream) sink.event(sink.ctx, id, &.{ .audio = draw.pcm.len });
        request.result.samples = draw.pcm.len;
        request.result.hollow = draw.hollow;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = draw.reason, .message = "" } });
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
};

/// Four 1.2 s phrases near 180 Hz, the third opening at `third_hz`: 178 level, 195 under 4 st up, 240 and 260 past it.
fn reply(a: std.mem.Allocator, third_hz: f64) ![]const f32 {
    return tone.tones(a, rate, &.{ .{ 1.2, 180, 150 }, .{ 0.3, 0, 0 }, .{ 1.2, 178, 148 }, .{ 0.3, 0, 0 }, .{ 1.2, third_hz, third_hz - 35 }, .{ 0.3, 0, 0 }, .{ 1.2, 182, 152 } }, tone.plain);
}

/// The reply body `pcm` is sent as.
fn body(a: std.mem.Allocator, pcm: []const f32) ![]const u8 {
    const out = try a.alloc(u8, 2 * pcm.len);
    for (pcm, 0..) |x, i| std.mem.writeInt(i16, out[2 * i ..][0..2], try pcm16(x), .little);
    return out;
}

const Reply = struct { s: *harness.Served, head: []const u8, body: []const u8 };

/// Served once, with `deadline` as X-Deadline-S if given; the reply's head and body.
fn serve(d: *Draws, cfg: server.Config, deadline: ?[]const u8, request: []const u8) !Reply {
    var line: [96]u8 = undefined;
    const s = try gpa.create(harness.Served);
    const first = if (deadline) |x| try std.fmt.bufPrint(&line, "POST /v1/audio/speech HTTP/1.1\r\nX-Deadline-S: {s}", .{x}) else "POST /v1/audio/speech HTTP/1.1";
    try s.open(d.engine(), cfg, first, request, 2000);
    // the body's read may move the bytes the head lies in
    const head = try d.a.dupe(u8, try s.readHead());
    s.route.join();
    return .{ .s = s, .head = head, .body = try s.readRest() };
}

fn config(register: speech_register.Policy, silence: speech_redraw.Policy) server.Config {
    var cfg = scripted.config(null);
    cfg.register = register;
    cfg.silence_redraw = silence;
    return cfg;
}

fn has(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

test "a jump trips one redraw at the next salt; the calmer redraw is sent, marked X-Register-Trip 1 and Redraws 1" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const calm = try reply(a, 178);
    var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = try reply(a, 240) }, .{ .pcm = calm } } };
    const r = try serve(&d, config(on, .{}), null, try scripted.body(a, "One. Two.", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 2), d.seen);
    try std.testing.expect(has(r.head, "X-Finish-Reason: stop\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n"));
    try std.testing.expectEqualSlices(u8, try body(a, calm), r.body);
    for (0..2) |k| try std.testing.expectEqual(api.seedFor(d.prompts[0], @intCast(k)), d.seeds[k]);
    try std.testing.expectEqualSlices(u32, d.prompts[0], d.prompts[1]);
}

test "an untripped reply is drawn once and sent byte for byte as with the guard off, marked Trip 0 and Redraws 0" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const small = try reply(a, 195);
    var bodies: [2][]const u8 = undefined;
    for ([_]speech_register.Policy{ .{}, on }, &bodies) |policy, *sent| {
        var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = small }, .{ .pcm = small } } };
        const r = try serve(&d, config(policy, .{}), null, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, 1), d.seen);
        try std.testing.expectEqual(policy.st > 0, has(r.head, "X-Finish-Reason: stop\r\nX-Register-Trip: 0\r\nX-Register-Redraws: 0\r\n"));
        sent.* = try a.dupe(u8, r.body);
    }
    try std.testing.expectEqualSlices(u8, try body(a, small), bodies[0]);
    try std.testing.expectEqualSlices(u8, bodies[0], bodies[1]);
}

test "with the guard off a jump is sent as drawn: one draw, X-Deadline-S unread, no register headers" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const jump = try reply(a, 240);
    var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = jump }, .{ .pcm = jump } } };
    const r = try serve(&d, config(.{}, .{}), "not seconds", try scripted.body(a, "One. Two.", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 1), d.seen);
    try std.testing.expect(has(r.head, "HTTP/1.1 200 ") and !has(r.head, "X-Register"));
    try std.testing.expectEqualSlices(u8, try body(a, jump), r.body);
}

test "of a jump and its redraw the one with the smaller jump is sent; never a redraw that ended in silence" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { first: f64, redraw: f64, reason: api.Reason = .stop, sent: enum { first, redraw } }{
        .{ .first = 260, .redraw = 240, .sent = .redraw },
        .{ .first = 240, .redraw = 260, .sent = .first },
        .{ .first = 240, .redraw = 195, .sent = .redraw },
        .{ .first = 240, .redraw = 178, .reason = .silence, .sent = .first },
    };
    for (cases) |case| {
        const first = try reply(a, case.first);
        const redraw = try reply(a, case.redraw);
        // a third draw stands ready: a second register redraw would take it
        var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = first }, .{ .pcm = redraw, .reason = case.reason }, .{ .pcm = first } } };
        const r = try serve(&d, config(on, .{}), null, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, 2), d.seen);
        try std.testing.expect(has(r.head, "X-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n"));
        try std.testing.expectEqualSlices(u8, try body(a, if (case.sent == .first) first else redraw), r.body);
    }
}

test "a redraw that jumps exactly as far as the first draw is not sent: a tie keeps the first" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const first = try reply(a, 240);
    // the same phrases and 0.2 s of silence after them: the same jump in other bytes
    const redraw = try a.alloc(f32, first.len + 20 * frame);
    @memcpy(redraw[0..first.len], first);
    @memset(redraw[first.len..], 0);
    var jumps: [2]f64 = undefined;
    for ([_][]const f32{ first, redraw }, &jumps) |pcm, *j| j.* = drama_register.jump((try drama_register.measure(a, pcm, rate, 210)).phrases, 4, 0.2).?.excess.?;
    try std.testing.expectEqual(jumps[0], jumps[1]);
    var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = first }, .{ .pcm = redraw } } };
    const r = try serve(&d, config(on, .{}), null, try scripted.body(a, "One. Two.", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 2), d.seen);
    try std.testing.expectEqualSlices(u8, try body(a, first), r.body);
}

test "silence and short redraws: a draw they trip is not measured; a jump after one still gets its own redraw" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const jump = try reply(a, 240);
    const small = try reply(a, 195);
    const calm = try reply(a, 178);
    // 240 characters are 15.4 s expected: at a ratio of 0.3 a 4 s stop is short, a 5.7 s one is not
    const sentence = "Sentence one is here and it is plain enough to read aloud, nothing more. ";
    const long_input = sentence ++ sentence ++ sentence ++ "Then a last sentence.";
    const cases = [_]struct { draws: []const Draw, silence: speech_redraw.Policy, input: []const u8, sent: []const f32, head: []const u8 }{
        // the onset guard's silence (or any silence) is redrawn as before; the redraw's jump then gets its own redraw
        .{ .draws = &.{ .{ .pcm = jump, .reason = .silence }, .{ .pcm = jump }, .{ .pcm = calm } }, .silence = .{ .max = 2 }, .input = "One. Two.", .sent = calm, .head = "X-Silence-Redraws: 1\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n" },
        // a register redraw that ends in silence is redrawn as a silence, and the silence redraw may be sent
        .{ .draws = &.{ .{ .pcm = jump }, .{ .pcm = small, .reason = .silence }, .{ .pcm = calm } }, .silence = .{ .max = 2 }, .input = "One. Two.", .sent = calm, .head = "X-Silence-Redraws: 1\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n" },
        // a short stop is redrawn without measuring its jump
        .{ .draws = &.{ .{ .pcm = jump[0 .. 4 * rate] }, .{ .pcm = calm } }, .silence = .{ .max = 1, .short_ratio = 0.3 }, .input = long_input, .sent = calm, .head = "X-Silence-Redraws: 1\r\nX-Register-Trip: 0\r\nX-Register-Redraws: 0\r\n" },
    };
    for (cases) |case| {
        var d: Draws = .{ .a = a, .draws = case.draws };
        const r = try serve(&d, config(on, case.silence), null, try scripted.body(a, case.input, false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(case.draws.len, d.seen);
        // every draw, silence, short or register redraw alike, takes the next salt
        for (0..d.seen) |k| try std.testing.expectEqual(api.seedFor(d.prompts[k], @intCast(k)), d.seeds[k]);
        try std.testing.expect(has(r.head, case.head));
        try std.testing.expectEqualSlices(u8, try body(a, case.sent), r.body);
    }
}

test "a register redraw starts only if t + d fits the budget; else, or if the engine stays busy, the first is sent" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const jump = try reply(a, 240);
    const calm = try reply(a, 178);
    // a 150 ms draw: t + d is at least 0.3 s, so 0.25 s fails closed though t alone fits it
    const cases = [_]struct { deadline: ?[]const u8, busy: u8 = 0, redrawn: bool }{
        .{ .deadline = "0.25", .redrawn = false },
        .{ .deadline = "5", .redrawn = true },
        .{ .deadline = "1", .busy = 200, .redrawn = false },
    };
    for (cases) |case| {
        var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = jump, .stall_ms = 150 }, .{ .pcm = calm, .busy = case.busy } } };
        const r = try serve(&d, config(on, .{}), case.deadline, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, if (case.redrawn) 2 else 1), d.seen);
        try std.testing.expect(has(r.head, if (case.redrawn) "X-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n" else "X-Register-Trip: 1\r\nX-Register-Redraws: 0\r\n"));
        try std.testing.expectEqualSlices(u8, try body(a, if (case.redrawn) calm else jump), r.body);
    }
}

test "under a caller's deadline a 400 to 700 character reply fits its redraw and a 2000 character one does not" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const jump = try reply(a, 240);
    const calm = try reply(a, 178);
    for ([_]struct { chars: f64, redrawn: bool }{ .{ .chars = 400, .redrawn = true }, .{ .chars = 700, .redrawn = true }, .{ .chars = 2000, .redrawn = false } }) |case| {
        // at 1/100 time: the wrapper's X-Deadline-S, and a draw of chars / 15.6 s of audio at the live 0.8 of real time
        const deadline = (@min(20 + 0.15 * case.chars, 150) - 3) / 100;
        const draw_ms: u32 = @intFromFloat(case.chars / speech_redraw.chars_per_second * 0.8 * 10);
        var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = jump, .stall_ms = draw_ms }, .{ .pcm = calm } } };
        const r = try serve(&d, config(on, .{}), try std.fmt.allocPrint(a, "{d}", .{deadline}), try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, if (case.redrawn) 2 else 1), d.seen);
        try std.testing.expect(has(r.head, if (case.redrawn) "X-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n" else "X-Register-Trip: 1\r\nX-Register-Redraws: 0\r\n"));
    }
}

test "a stream is not guarded: one draw, no register headers" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const jump = try reply(a, 240);
    var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = jump }, .{ .pcm = jump } } };
    const r = try serve(&d, config(on, .{}), null, try scripted.body(a, "One. Two.", true));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 1), d.seen);
    try std.testing.expect(has(r.head, "HTTP/1.1 200 ") and !has(r.head, "X-Register"));
}

test "a NaN or infinite sample fails the reply before any measure, as the guard off fails it" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const jump = try reply(a, 240);
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| for ([_]speech_register.Policy{ .{}, on }) |policy| {
        const pcm = try a.dupe(f32, jump);
        pcm[8000] = bad;
        var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = pcm }, .{ .pcm = jump } } };
        const r = try serve(&d, config(policy, .{}), null, try scripted.body(a, "One. Two.", false));
        defer scripted.done(r.s);
        try std.testing.expectEqual(@as(usize, 1), d.seen);
        try std.testing.expect(has(r.head, "HTTP/1.1 500 ") and !has(r.head, "X-Register"));
        try std.testing.expect(has(r.body, "the synthesized audio has non-finite samples"));
    };
}

test "a long-form reply of more than one segment is not guarded: no register headers" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var t: scripted.Turns = .{ .turns = &.{ .{ .frames = 5 }, .{ .frames = 4 } } };
    defer t.deinit();
    var cfg = scripted.config(.{ .chars = 20 });
    cfg.register = on;
    const s = try gpa.create(harness.Served);
    defer scripted.done(s);
    try s.startWith(t.engine(), cfg, "HTTP/1.1", try scripted.body(a, try scripted.sentences(a, 2), false), 2000);
    const head = try a.dupe(u8, try s.readHead());
    s.route.join();
    try std.testing.expectEqual(@as(usize, 2), t.seen);
    try std.testing.expect(has(head, "X-Long-Form-Segments: 2;") and !has(head, "X-Register"));
}

test "a draw --silence-mid-s ended is redrawn as a silence, unmeasured; the redraw's jump then gets its register redraw" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const jump = try reply(a, 240);
    const calm = try reply(a, 178);
    var d: Draws = .{ .a = a, .draws = &.{ .{ .pcm = jump, .reason = .silence, .hollow = 3 * rate }, .{ .pcm = jump }, .{ .pcm = calm } } };
    var cfg = config(on, .{ .max = 2 });
    cfg.silence_mid = true;
    const r = try serve(&d, cfg, null, try scripted.body(a, "One. Two.", false));
    defer scripted.done(r.s);
    try std.testing.expectEqual(@as(usize, 3), d.seen);
    for (0..3) |k| try std.testing.expectEqual(api.seedFor(d.prompts[k], @intCast(k)), d.seeds[k]);
    try std.testing.expect(has(r.head, "X-Silence-Redraws: 1\r\nX-Silence-Mid-Trips: 1\r\nX-Register-Trip: 1\r\nX-Register-Redraws: 1\r\n"));
    try std.testing.expectEqualSlices(u8, try body(a, calm), r.body);
}
