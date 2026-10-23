//! /v1/audio/speech with --long-form: a long input spoken segment by segment, each hearing the audio of those before.
const std = @import("std");
const api = @import("engine_api");
const errors = @import("errors.zig");
const speech = @import("speech.zig");
const speech_redraw = @import("speech_redraw.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_long = @import("drama_long.zig");
const drama_joins = @import("drama_joins.zig");
const drama_shift = @import("drama_shift.zig");
const drama_cfg = @import("drama_cfg.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = errors.Cx;
const Reference = drama_prompt.Reference;
const Codes = drama_prompt.Codes;

/// A long-form reply's segments of `input`, the references each hears first, how many earlier ones after, and the `¶`s.
pub const Long = struct {
    input: []const u8,
    segments: []const []const u8,
    shifts: []const bool,
    references: []const Reference,
    history: u8,
    shift_pause_s: f32,

    fn shifted(l: Long, i: usize) bool {
        return i < l.shifts.len and l.shifts[i];
    }
};

/// The stream's trailers: the finish reason when cut short, where the segments start, and with shifts on, their pauses.
const trailer = "X-Finish-Reason, X-Long-Form-Segments";
const shift_trailer = trailer ++ ", X-Pause-Spans";

/// The input's segments, cut at each of `marked`'s cuts, each refused now as a whole input would be; null: one reply.
pub fn plan(srv: *Server, cx: *Cx, sp: api.SpeechInfo, marked: drama_shift.Marked, references: []const Reference, options: drama_long.Options) errors.Refused!?Long {
    const input = marked.text;
    if (marked.cuts.len == 0 and (std.unicode.utf8CountCodepoints(input) catch unreachable) <= options.chars) return null;
    const split = try drama_shift.split(cx.a, marked, options.chars);
    const segments = split.segments;
    if (segments.len <= 1) return null;
    if (segments.len > drama_long.max_segments)
        return cx.fail(.request, drama_prompt.error_prefix ++ " a long-form input in {d} segments of at most {d} characters exceeds the maximum of {d} segments", .{ segments.len, options.chars, drama_long.max_segments });
    for (segments) |segment| {
        if (try drama_prompt.problem(cx.a, segment)) |message| return cx.refuse(message);
        const rows = drama_prompt.promptRows(cx.a, srv.text, segment, references) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Template => cx.other("the tokenizer cannot encode this prompt"),
        };
        if (rows.tokens.len > sp.max_prompt_tokens)
            return cx.fail(.request, drama_prompt.error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ rows.tokens.len, sp.max_prompt_tokens });
    }
    return .{ .input = input, .segments = segments, .shifts = split.shifts, .references = references, .history = options.history, .shift_pause_s = options.shift_pause_s };
}

/// Speaks every segment into one reply: a WAV or PCM body, or a PCM stream from the first segment's first audio.
pub fn synthesize(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, long: Long, format: speech.Format, stream: bool, received: i96, redraws: *speech_redraw.Redraws, stripped: ?usize, gamma: ?f32) void {
    if (sp.frame_length == 0 or sp.max_frames == 0 or sp.sample_rate == 0 or sp.sample_rate > std.math.maxInt(u32) / 2)
        return speech.fail(c, cx, id, redraws, .server, "the engine advertises no usable speech output");
    const segment_samples = @as(usize, sp.max_frames) * sp.frame_length;
    // a shift's join can take more than a pause (drama_shift.room)
    const pause = @max(@as(usize, @intFromFloat(@ceil(drama_joins.max_pause_seconds * @as(f32, @floatFromInt(sp.sample_rate))))), drama_shift.room(long.shift_pause_s, sp.sample_rate));
    const out_len = drama_long.max_frames * @as(usize, sp.frame_length) + (drama_long.max_segments - 1) * pause;
    const pcm = srv.gpa.alloc(f32, out_len + segment_samples) catch return speech.fail(c, cx, id, redraws, .server, "out of memory");
    defer srv.gpa.free(pcm);
    const bytes = srv.gpa.alloc(u8, 2 * (segment_samples + pause)) catch return speech.fail(c, cx, id, redraws, .server, "out of memory");
    defer srv.gpa.free(bytes);
    const generated = srv.gpa.alloc(Codes, drama_long.max_frames) catch return speech.fail(c, cx, id, redraws, .server, "out of memory");
    defer srv.gpa.free(generated);
    if (srv.keepalive) |k| k.begin();
    defer if (srv.keepalive) |k| k.end();
    var r: Reply = .{ .srv = srv, .c = c, .cx = cx, .id = id, .sp = sp, .long = long, .stream = stream, .received = received, .redraws = redraws, .stripped = stripped, .gamma = gamma, .out = pcm[0..out_len], .scratch = pcm[out_len..], .bytes = bytes, .generated = generated };
    for (0..long.segments.len) |i| {
        if (r.frames == drama_long.max_frames) {
            r.reason = .length;
            break;
        }
        if (!r.speak(i)) return;
        // a clipped or silenced segment's audio would be the next one's carry-over: the reply ends with it
        if (r.reason != .stop) break;
    }
    r.finish(format);
}

/// A finished draw of a segment: the text it spoke, its audio's length in `scratch`, and its finish.
const Drawn = struct { text: []const u8, samples: usize, reason: api.Reason };

const Reply = struct {
    srv: *Server,
    c: *Conn,
    cx: *Cx,
    id: []const u8,
    sp: api.SpeechInfo,
    long: Long,
    stream: bool,
    received: i96,
    /// A whole reply redraws a segment the silence guard ended, as one synthesis does; a stream never does.
    redraws: *speech_redraw.Redraws,
    /// The controls --allowed-controls took out of the input, null without it.
    stripped: ?usize,
    /// The guidance weight with guidance on (1: unguided), null without.
    gamma: ?f32,
    /// The reply's audio: the segments with the pauses made up between them.
    out: []f32,
    /// Where each segment is synthesized before it joins `out`.
    scratch: []f32,
    /// A stream chunk's PCM16, converted from audio that stays float for the joins.
    bytes: []u8,
    generated: []Codes,
    heard: std.ArrayList(Reference) = .empty,
    starts: std.ArrayList(usize) = .empty,
    /// Each shift's quiet stretch in `out`, [start, end) samples, fill included.
    spans: std.ArrayList([2]usize) = .empty,
    levels: std.ArrayList(f32) = .empty,
    samples: usize = 0,
    frames: usize = 0,
    rows: usize = 0,
    added: usize = 0,
    reason: api.Reason = .stop,
    open: bool = false,
    /// What a plain redraw drops from the whole input, found at the reply's first plain redraw; empty: nothing.
    dropped: ?[]const speech_redraw.Span = null,

    /// The prompt for `text`: its references, then the latest segments heard while it fits (bare, `plan` checked it).
    fn prompt(r: *Reply, text: []const u8) !drama_prompt.Rows {
        var keep: usize = @min(r.long.history, r.heard.items.len);
        while (true) : (keep -= 1) {
            const rows = try drama_prompt.promptRows(r.cx.a, r.srv.text, text, try drama_long.carried(r.cx.a, r.long.references, r.heard.items, keep));
            if (rows.tokens.len <= r.sp.max_prompt_tokens or keep == 0) return rows;
        }
    }

    /// Synthesizes segment `i` and joins it to the reply; false when the reply has ended (refused, cut short, or left).
    fn speak(r: *Reply, i: usize) bool {
        const fl: usize = r.sp.frame_length;
        const room = @min(r.sp.max_frames, drama_long.max_frames - r.frames);
        var text = r.long.segments[i];
        var rows = r.prompt(text) catch |e| return r.end(.other, if (e == error.OutOfMemory) "out of memory" else "the tokenizer cannot encode this prompt");
        // only the first segment streams as it is made: a later one's level and pause need all of it
        const live = r.stream and i == 0;
        // after the first segment another request's synthesis is waited out, so a reply is not cut between segments
        var start_by: ?i96 = if (i > 0) std.math.maxInt(i96) else null;
        var next: ?speech_redraw.Spoken = null; // what redraws speak, made at the first trip
        var last: Drawn = undefined; // the tripped draw a redraw replaces, still in `scratch` if the redraw never starts
        var k: u8 = 0;
        const drawn: Drawn = while (true) : (k += 1) {
            var result: api.SpeechResult = .{};
            const guided = drama_cfg.guide(r.cx.a, r.srv.text, r.srv.config.cfg, r.gamma orelse 1, text, rows) catch |e| return r.end(.other, if (e == error.OutOfMemory) "out of memory" else "the tokenizer cannot encode this prompt");
            const request: api.SpeechRequest = .{
                .prompt = rows.tokens,
                .codes = rows.codes,
                .generated = r.generated[r.frames..][0..room],
                .pcm = r.scratch[0 .. room * fl],
                .result = &result,
                .seed = api.seedFor(rows.tokens, speech_redraw.Redraws.salt(r.srv.config.seed_salt, k)),
                .max_frames = @intCast(room),
                .temperature = drama_prompt.temperature,
                .top_p = drama_prompt.top_p,
                .top_k = drama_prompt.top_k,
                .unconditioned = guided.unconditioned,
                .cfg_gamma = guided.gamma,
                .stream = live,
            };
            var box: Mailbox = .{ .io = r.srv.io, .gpa = r.srv.gpa };
            defer box.deinit();
            const engine_id = r.srv.next_id.fetchAdd(1, .monotonic);
            const started = clock.nowNs(r.srv.io);
            speech.submit(r.srv, r.c, engine_id, &request, &box, start_by) catch |e| {
                if (e == error.Busy and k > 0) {
                    r.redraws.stand(r.id, "the engine stayed busy past the latest start that fits the budget");
                    break last;
                }
                const refused = speech.submitFailure(e);
                return r.end(refused.kind, refused.message);
            };
            // The engine holds `request` and its buffers until finished: no path below returns before it arrives.
            if (k > 0) r.redraws.used += 1;
            r.rows += rows.tokens.len;
            if (live) {
                const sent = r.stream1(&box, engine_id, request.pcm) orelse return false;
                const mismatch: ?[]const u8 = if (result.samples != sent) "the finished synthesis does not match the audio it published" else null;
                if (speech.why(&box, result.samples, request.pcm.len) orelse mismatch) |message| return r.end(.server, message);
            } else {
                if (speech.settle(r.srv, r.c, &box, engine_id, null) == null) return r.gone();
                if (speech.why(&box, result.samples, request.pcm.len)) |message| return r.end(.server, message);
            }
            r.redraws.hollowed(r.id, result.hollow, r.sp.sample_rate);
            const this: Drawn = .{ .text = text, .samples = result.samples, .reason = box.reason };
            // a stream has sent audio before its guard can trip, so only a whole reply is redrawn
            if (r.stream or r.redraws.policy.max == 0) break this;
            const cause = r.redraws.trip(this.reason, this.samples, r.sp.sample_rate, text) orelse break this;
            if (next == null) next = r.redrawText(i, rows) orelse break this;
            start_by = r.redraws.again(r.srv.io, r.id, started, cause) orelse break this;
            text = next.?.text;
            rows = next.?.rows;
            last = this;
        };
        const frames = drawn.samples / fl;
        const audio = r.scratch[0..drawn.samples];
        const joined = r.join(i, audio) catch return r.end(.server, "out of memory");
        if (r.stream and !live) r.send(r.out[joined..r.samples]) catch |e| return if (e == error.Closed) r.gone() else r.end(.server, "the synthesized audio has non-finite samples");
        r.heard.append(r.cx.a, .{ .text = drawn.text, .frames = r.generated[r.frames..][0..frames] }) catch return r.end(.server, "out of memory");
        r.frames += frames;
        if (drawn.reason != .stop) r.reason = drawn.reason;
        return true;
    }

    /// What redraws of segment `i` speak: its `rows`, or its plain text's (--silence-redraw-plain); null if neither.
    fn redrawText(r: *Reply, i: usize, rows: drama_prompt.Rows) ?speech_redraw.Spoken {
        const segment = r.long.segments[i];
        if (!r.redraws.policy.plain) return .{ .text = segment, .rows = rows };
        const why_not = why: {
            const a = r.cx.a;
            // the whole input's controls, so one a cut crosses goes whole and only the input's own note head counts
            const spans = r.dropped orelse spans: {
                const found = (speech_redraw.dropped(a, r.long.input, r.redraws.policy.plain_scope) catch break :why "out of memory") orelse &.{};
                r.dropped = found;
                break :spans found;
            };
            const start = @intFromPtr(segment.ptr) - @intFromPtr(r.long.input.ptr);
            const text = (speech_redraw.heard(a, r.long.input, spans, start, start + segment.len) catch break :why "out of memory") orelse return .{ .text = segment, .rows = rows };
            if (speech_redraw.problem(a, text)) |message| break :why message;
            const cue_free = r.prompt(text) catch break :why "the tokenizer cannot encode the segment without its cues";
            if (cue_free.tokens.len > r.sp.max_prompt_tokens) break :why "the segment without its cues does not fit the prompt";
            log.line("speech {s} long-form segment {d}: redraws speak it without its {s}", .{ r.id, i + 1, @tagName(r.redraws.policy.plain_scope) });
            return .{ .text = text, .rows = cue_free };
        };
        r.redraws.stand(r.id, why_not);
        return null;
    }

    /// Levels segment `i`'s `audio`, makes up the pause before it, appends both to `out`; where the pause starts.
    fn join(r: *Reply, i: usize, audio: []f32) !usize {
        const rate = r.sp.sample_rate;
        const before = r.samples;
        if (i > 0) {
            const earlier = drama_joins.median(try r.cx.a.dupe(f32, r.levels.items));
            const g = drama_joins.gain(try drama_joins.loudness(r.cx.a, audio, rate), earlier, drama_joins.peak(audio));
            if (g != 1) for (audio) |*v| {
                v.* *= g;
            };
            const prev = r.out[r.starts.items[i - 1]..r.samples];
            const left = r.long.segments[i - 1];
            const right = r.long.segments[i];
            const b = drama_joins.boundary(left, drama_long.gapBetween(left, right), right);
            const n = if (r.long.shifted(i)) try r.shift(prev, audio, @max(r.long.shift_pause_s, b.seconds())) else plain: {
                const short = drama_joins.shortfall(prev, audio, rate, b);
                drama_joins.roomTone(prev, rate, r.out[r.samples..][0..short]);
                break :plain short;
            };
            r.samples += n;
            r.added += n;
        }
        try r.starts.append(r.cx.a, r.samples);
        @memcpy(r.out[r.samples..][0..audio.len], audio);
        r.samples += audio.len;
        if (try drama_joins.loudness(r.cx.a, audio, rate)) |level| try r.levels.append(r.cx.a, level);
        return before;
    }

    /// A shift's join: room tone until the silent frames around it, on the reply's grid, last `seconds`; its span.
    fn shift(r: *Reply, prev: []const f32, audio: []const f32, seconds: f32) !usize {
        const rate = r.sp.sample_rate;
        const target: usize = @intFromFloat(@round(seconds * @as(f32, @floatFromInt(rate))));
        const quiet_before = drama_joins.trailingSilence(prev, rate);
        const quiet_after = drama_joins.leadingSilence(audio, rate);
        var tone: ?[]f32 = null;
        var n = target -| (quiet_before + quiet_after);
        // with nothing to add, the silent frames around the middle of the quiet already there
        var run = if (n == 0) drama_shift.quietRun(r.out[0..r.samples], audio, rate, r.samples - quiet_before + (quiet_before + quiet_after) / 2) else try r.place(prev, audio, &tone, n);
        // the fades or the grid can leave it short; once the tone alone outlasts it by two frames it is long enough
        while (run[1] - run[0] < target) {
            n += target - (run[1] - run[0]) + drama_joins.frameLen(rate);
            run = try r.place(prev, audio, &tone, n);
        }
        try r.spans.append(r.cx.a, .{ @min(r.samples - quiet_before, run[0]), @max(r.samples + n + quiet_after, run[1]) });
        return n;
    }

    /// `n` samples of fill after the reply so far, then the silent frames around its middle with `audio` after it.
    fn place(r: *Reply, prev: []const f32, audio: []const f32, tone: *?[]f32, n: usize) ![2]usize {
        const rate = r.sp.sample_rate;
        // the earlier segment's last second, then everything before it, then the next segment: never zeros
        tone.* = tone.* orelse try drama_shift.tone(r.cx.a, &.{ prev[prev.len - @min(prev.len, rate) ..], r.out[0..r.samples], audio }, rate);
        drama_shift.fill(prev, audio, tone.*.?, rate, r.out[r.samples..][0..n]);
        return drama_shift.quietRun(r.out[0 .. r.samples + n], audio, rate, r.samples + n / 2);
    }

    /// Streams the first segment as it is published; the samples sent, or null when the reply has ended.
    fn stream1(r: *Reply, box: *Mailbox, engine_id: api.Id, pcm: []const f32) ?usize {
        var sent: usize = 0;
        while (true) {
            const now = speech.settle(r.srv, r.c, box, engine_id, sent) orelse {
                _ = r.gone();
                return null;
            };
            if (now == sent) return sent;
            r.send(pcm[sent..now]) catch |e| {
                _ = speech.abandon(r.srv, box, engine_id);
                _ = if (e == error.Closed) r.gone() else r.end(.server, "the synthesized audio has non-finite samples");
                return null;
            };
            sent = now;
        }
    }

    /// One chunk of `audio` as PCM16, the stream opened first if it is not yet.
    fn send(r: *Reply, audio: []const f32) error{ Closed, NonFinite }!void {
        if (audio.len == 0) return;
        for (audio, 0..) |x, k| std.mem.writeInt(i16, r.bytes[2 * k ..][0..2], try speech.pcm16(x), .little);
        if (!r.open) {
            var n: [4]u8 = undefined;
            var count: [20]u8 = undefined;
            var guidance: [24]u8 = undefined;
            const heads: speech.Heads = .{ .redraws = r.redraws.header(&n), .stripped = speech.strippedHeader(&count, r.stripped), .cfg_gamma = drama_cfg.header(&guidance, r.gamma) };
            try speech.startStream(r.c, r.sp.sample_rate, if (r.long.shift_pause_s > 0) shift_trailer else trailer, heads);
            r.open = true;
        }
        var size: [20]u8 = undefined;
        r.c.writeAll(std.fmt.bufPrint(&size, "{x}\r\n", .{2 * audio.len}) catch unreachable) catch return error.Closed;
        r.c.writeAll(r.bytes[0 .. 2 * audio.len]) catch return error.Closed;
        r.c.writeAll("\r\n") catch return error.Closed;
    }

    /// An error reply before any audio; after a stream's first chunk, no terminator, so the body is incomplete.
    fn end(r: *Reply, kind: errors.Kind, message: []const u8) bool {
        if (!r.open) {
            speech.fail(r.c, r.cx, r.id, r.redraws, kind, message);
            return false;
        }
        r.c.close = true;
        log.line("speech {s} long-form stream ended incomplete after segments={d} samples={d}: {s}", .{ r.id, r.starts.items.len, r.samples, message });
        return false;
    }

    fn gone(r: *Reply) bool {
        speech.left(r.srv, r.c, r.id, r.rows, r.received);
        return false;
    }

    /// The whole body with the segments and pause headers, or the stream's terminating chunk with them as trailers.
    fn finish(r: *Reply, format: speech.Format) void {
        // fixed buffers, never allocation: a stream already open must end with its trailers
        var header: [drama_long.segments_header_max]u8 = undefined;
        const segments = drama_long.segmentsHeader(&header, r.starts.items);
        var pauses: [drama_shift.spans_header_max]u8 = undefined;
        const spans: ?[]const u8 = if (r.spans.items.len > 0) drama_shift.spansHeader(&pauses, r.spans.items) else null;
        if (r.stream) {
            const reason = switch (r.reason) {
                .length => "X-Finish-Reason: length\r\n",
                .silence => "X-Finish-Reason: silence\r\n",
                else => "",
            };
            var buf: [drama_long.segments_header_max + drama_shift.spans_header_max + 96]u8 = undefined;
            const last = std.fmt.bufPrint(&buf, "0\r\n{s}X-Long-Form-Segments: {s}\r\n{s}{s}{s}\r\n", .{ reason, segments, if (spans != null) "X-Pause-Spans: " else "", spans orelse "", if (spans != null) "\r\n" else "" }) catch unreachable;
            r.c.writeAll(last) catch {
                r.c.close = true;
                return;
            };
        } else {
            const data = speech.quantizeInPlace(r.out[0..r.samples]) catch return speech.fail(r.c, r.cx, r.id, r.redraws, .server, "the synthesized audio has non-finite samples");
            if (data.len > std.math.maxInt(u32) - 36) return speech.fail(r.c, r.cx, r.id, r.redraws, .server, "the synthesized audio is too long for one reply");
            var n: [4]u8 = undefined;
            var trips: [4]u8 = undefined;
            var count: [20]u8 = undefined;
            var guidance: [24]u8 = undefined;
            const heads: speech.Heads = .{ .segments = segments, .spans = spans, .redraws = r.redraws.header(&n), .mid = r.redraws.hollowHeader(&trips), .stripped = speech.strippedHeader(&count, r.stripped), .cfg_gamma = drama_cfg.header(&guidance, r.gamma) };
            speech.sendAudio(r.c, format, r.sp.sample_rate, data, r.reason, heads) catch {
                r.c.close = true;
                return;
            };
        }
        log.line("speech {s} long_form segments={d}/{d} prompt={d} samples={d} pauses_added={d} format={s} stream={} finish={s} seconds={d:.2}", .{
            r.id, r.starts.items.len, r.long.segments.len, r.rows, r.samples, r.added, @tagName(format), r.stream, @tagName(r.reason), clock.seconds(clock.nowNs(r.srv.io) - r.received),
        });
        if (spans) |value| log.line("speech {s} pause spans {s}", .{ r.id, value });
    }
};
