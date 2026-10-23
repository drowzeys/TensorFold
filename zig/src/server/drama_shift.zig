//! Pauses at topic shifts: the wrapper's `¶` marks one, and with --shift-pause-s a long-form cut and pause go there.
const std = @import("std");
const drama_prompt = @import("drama_prompt.zig");
const drama_long = @import("drama_long.zig");
const drama_joins = @import("drama_joins.zig");
const unicode = @import("tokenizer").unicode;
const Allocator = std.mem.Allocator;

/// U+00B6 PILCROW SIGN, which the wrapper writes for a blank line between topics.
pub const mark = "\u{b6}";

/// An input without its marks: `text`, the byte offsets in it of the honoured marks, and how many marks were dropped.
pub const Marked = struct { text: []const u8, cuts: []const usize = &.{}, dropped: usize = 0 };

/// `input` without its marks and the white space after each; with `honour`, its cuts (`read`'s rule, in the docs).
pub fn read(a: Allocator, input: []const u8, honour: bool) Allocator.Error!Marked {
    if (std.mem.indexOf(u8, input, mark) == null) return .{ .text = input };
    var text: std.ArrayList(u8) = .empty;
    var cuts: std.ArrayList(usize) = .empty;
    var dropped: usize = 0;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, input, at, mark)) |found| {
        try text.appendSlice(a, input[at..found]);
        var marks: usize = 0;
        at = found;
        while (std.mem.startsWith(u8, input[at..], mark)) {
            marks += 1;
            at = afterWhiteSpace(input, at + mark.len);
        }
        if (!honour) continue;
        // the segments either side: from the cut before, and up to the next mark
        const before = text.items[if (cuts.items.len > 0) cuts.items[cuts.items.len - 1] else 0..];
        const after = input[at .. std.mem.indexOfPos(u8, input, at, mark) orelse input.len];
        if (drama_long.endsSentence(before) and hasWords(before) and hasWords(after)) try cuts.append(a, text.items.len) else dropped += marks;
    }
    try text.appendSlice(a, input[at..]);
    return .{ .text = text.items, .cuts = cuts.items, .dropped = dropped };
}

fn afterWhiteSpace(text: []const u8, from: usize) usize {
    var it = std.unicode.Utf8View.initUnchecked(text[from..]).iterator();
    while (it.peek(1).len > 0 and drama_prompt.isWhiteSpace(std.unicode.utf8Decode(it.peek(1)) catch unreachable)) _ = it.nextCodepoint();
    return from + it.i;
}

/// Whether `text` has a letter or number (general category L or N) outside its ``[...]`` and ``<...>`` controls.
fn hasWords(text: []const u8) bool {
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (it.nextCodepoint()) |cp| {
        const close: ?u8 = switch (cp) {
            '[' => ']',
            '<' => '>',
            else => null,
        };
        if (close) |c| if (std.mem.indexOfScalarPos(u8, text, it.i, c)) |end| {
            it.i = end + 1;
            continue;
        };
        if (unicode.bit(unicode.category(cp)) & (unicode.letter | unicode.number) != 0) return true;
    }
    return false;
}

/// A long-form input's segments, and `shifts[i]` (empty without a cut): a mark stands right before segment i.
pub const Segments = struct { segments: []const []const u8, shifts: []const bool = &.{} };

/// `marked`'s segments of at most `chars`.
pub fn split(a: Allocator, marked: Marked, chars: u32) Allocator.Error!Segments {
    if (marked.cuts.len == 0) return .{ .segments = try drama_long.split(a, marked.text, chars) };
    var segments: std.ArrayList([]const u8) = .empty;
    var shifts: std.ArrayList(bool) = .empty;
    var from: usize = 0;
    for (0..marked.cuts.len + 1) |k| {
        const to = if (k < marked.cuts.len) marked.cuts[k] else marked.text.len;
        for (try drama_long.split(a, marked.text[from..to], chars), 0..) |segment, j| {
            try segments.append(a, segment);
            try shifts.append(a, j == 0 and k > 0);
        }
        from = to;
    }
    return .{ .segments = segments.items, .shifts = shifts.items };
}

/// Below this a source holds digital silence, not a room's tone.
const least_tone_db: f32 = -90;
/// The generated tone's level when no audio has a room's tone.
const noise_db: f32 = -65;
/// Every sample of a shift's tone stays 6 dB under the floor, so any 10 ms of it is silent however it is played.
const tone_peak = std.math.pow(f32, 10, (drama_joins.floor_db - 6) / 20);
/// Equal-power crossfades into and out of the tone, each up to 15 ms.
const crossfade_seconds: f32 = 0.015;

/// A shift's tone under `tone_peak`: the first of `sources`' quietest silent 100 ms, never digital silence; else noise.
pub fn tone(a: Allocator, sources: []const []const f32, rate: u32) Allocator.Error![]f32 {
    const out = try a.alloc(f32, 10 * drama_joins.frameLen(rate));
    const found = for (sources) |s| {
        if (drama_joins.quietest(s, rate, least_tone_db)) |window| break window;
    } else null;
    if (found) |window| @memcpy(out, window) else noise(out);
    const top = drama_joins.peak(out);
    if (top > tone_peak) for (out) |*v| {
        v.* *= tone_peak / top;
    };
    return out;
}

/// Uniform white noise at `noise_db` RMS, the same on every call.
fn noise(out: []f32) void {
    const amplitude = @sqrt(3.0) * std.math.pow(f32, 10, noise_db / 20);
    var state: u32 = 0x2545f491;
    for (out) |*v| {
        state = state *% 1664525 +% 1013904223;
        v.* = amplitude * (@as(f32, @floatFromInt(state >> 8)) / (1 << 23) - 1);
    }
}

/// A shift's fill in `out`: `tone_` back and forth, crossfaded from `left`'s end into `right`'s start.
pub fn fill(left: []const f32, right: []const f32, tone_: []const f32, rate: u32, out: []f32) void {
    for (out, 0..) |*v, i| v.* = drama_joins.play(tone_, i);
    const fade = @min(@as(usize, @intFromFloat(crossfade_seconds * @as(f32, @floatFromInt(rate)))), out.len / 2, left.len, right.len);
    // each segment mirrored at its seam, so neither changes: a stream has sent the earlier one already
    for (0..fade) |j| {
        const theta = std.math.pi / 2.0 * (@as(f32, @floatFromInt(j)) + 0.5) / @as(f32, @floatFromInt(fade));
        const head = &out[j];
        head.* = left[left.len - 1 - j] * @cos(theta) + head.* * @sin(theta);
        const tail = &out[out.len - 1 - j];
        tail.* = right[j] * @cos(theta) + tail.* * @sin(theta);
    }
}

/// The silent 10 ms frames around sample `probe` of `head` then `tail`, framed from `head`'s start: [start, end).
pub fn quietRun(head: []const f32, tail: []const f32, rate: u32, probe: usize) [2]usize {
    const n = drama_joins.frameLen(rate);
    const frames = (head.len + tail.len) / n;
    var first = probe / n;
    if (first >= frames or !silentFrame(head, tail, first * n, n)) return .{ probe, probe };
    var end = first + 1;
    while (first > 0 and silentFrame(head, tail, (first - 1) * n, n)) first -= 1;
    while (end < frames and silentFrame(head, tail, end * n, n)) end += 1;
    return .{ first * n, end * n };
}

fn silentFrame(head: []const f32, tail: []const f32, at: usize, n: usize) bool {
    var sum: f64 = 0;
    for (at..at + n) |i| {
        const v: f64 = if (i < head.len) head[i] else tail[i - head.len];
        sum += v * v;
    }
    return 10 * std.math.log10(sum / @as(f64, @floatFromInt(n))) < drama_joins.floor_db;
}

/// Samples one shift's join may add at most: twice its pause, three frames and both crossfades (`Reply.shift`).
pub fn room(seconds: f32, rate: u32) usize {
    if (seconds == 0) return 0;
    const target: usize = @intFromFloat(@ceil(@max(seconds, drama_joins.max_pause_seconds) * @as(f32, @floatFromInt(rate))));
    return 2 * target + 3 * drama_joins.frameLen(rate) + 2 * @as(usize, @intFromFloat(crossfade_seconds * @as(f32, @floatFromInt(rate))));
}

/// Bytes the X-Pause-Spans value takes at most: a ``start-end`` of 20 digits each per join, commas between.
pub const spans_header_max = (drama_long.max_segments - 1) * (2 * 20 + 1) + (drama_long.max_segments - 2);

/// The X-Pause-Spans value in `buf`: each shift's quiet stretch as ``start-end`` samples, end exclusive.
pub fn spansHeader(buf: *[spans_header_max]u8, spans: []const [2]usize) []const u8 {
    std.debug.assert(spans.len < drama_long.max_segments);
    var w: std.Io.Writer = .fixed(buf);
    // the buffer holds the largest value
    for (spans, 0..) |s, i| w.print("{s}{d}-{d}", .{ if (i == 0) "" else ",", s[0], s[1] }) catch unreachable;
    return w.buffered();
}

fn expectRead(input: []const u8, text: []const u8, cuts: []const usize, dropped: usize) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const got = try read(arena.allocator(), input, true);
    try std.testing.expectEqualStrings(text, got.text);
    try std.testing.expectEqualSlices(usize, cuts, got.cuts);
    try std.testing.expectEqual(dropped, got.dropped);
    // off, the marks go the same way and nothing is cut
    const off = try read(arena.allocator(), input, false);
    try std.testing.expectEqualStrings(text, off.text);
    try std.testing.expect(off.cuts.len == 0 and off.dropped == 0);
}

test "a mark is honoured right after . ! ? or \u{2026}, closing quotes between, and only with words after it" {
    try expectRead("One. \u{b6} Two.", "One. Two.", &.{5}, 0);
    for ([_][]const u8{ "!", "?", "\u{2026}", "...", ".\"", "?\u{201d})" }) |end| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const head = try std.mem.concat(a, u8, &.{ "One", end, " " });
        try expectRead(try std.mem.concat(a, u8, &.{ head, "\u{b6} Two." }), try std.mem.concat(a, u8, &.{ head, "Two." }), &.{head.len}, 0);
    }
    // a run of marks is one; no words after, no sentence end before, or nothing before: dropped
    try expectRead("One. \u{b6} \u{b6}\u{b6} Two.", "One. Two.", &.{5}, 0);
    try expectRead("One, \u{b6} two. \u{b6} Three \u{b6} four.", "One, two. Three four.", &.{10}, 2);
    try expectRead("\u{b6} One. \u{b6} \u{b6} ", "One. ", &.{}, 3);
    try expectRead("Pi is 3.\u{b6}14.", "Pi is 3.14.", &.{8}, 0);
    try expectRead("No marks. At all.", "No marks. At all.", &.{}, 0);
    try expectRead("Caf\u{e9}. \u{b6} \u{c9}t\u{e9}.", "Caf\u{e9}. \u{c9}t\u{e9}.", &.{7}, 0);
}

test "a mark needs words on both sides: punctuation or cues alone are no segment" {
    try expectRead("One. \u{b6} !!!", "One. !!!", &.{}, 1);
    try expectRead("One. \u{b6} ...\u{201d})", "One. ...\u{201d})", &.{}, 1);
    try expectRead("One. \u{b6} [laughs]", "One. [laughs]", &.{}, 1);
    try expectRead("!!! \u{b6} Two.", "!!! Two.", &.{}, 1);
    // a mark before punctuation alone is dropped; the next one, with words since the last cut and after it, is kept
    try expectRead("One. \u{b6} !!! \u{b6} Two.", "One. !!! Two.", &.{9}, 1);
    // a cue may open the segment after a mark, before its words
    try expectRead("One. \u{b6} [small laugh] Two.", "One. [small laugh] Two.", &.{5}, 0);
}

test "words are Unicode letters and numbers; full-width punctuation and symbols are none" {
    // full-width marks, math and currency signs, an emoji, Latin-1's inverted marks
    for ([_][]const u8{ "\u{ff01}\u{ff01}\u{ff01}", "\u{d7}", "\u{20ac} \u{2192}", "\u{1f600}", "\u{bf}\u{a1}", "\u{3002}\u{300c}\u{300d}" }) |rest| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        try expectRead(try std.mem.concat(a, u8, &.{ "One. \u{b6} ", rest }), try std.mem.concat(a, u8, &.{ "One. ", rest }), &.{}, 1);
    }
    // CJK and accented letters, full-width digits
    try expectRead("One. \u{b6} \u{6b21}\u{3078}\u{3002}", "One. \u{6b21}\u{3078}\u{3002}", &.{5}, 0);
    try expectRead("One. \u{b6} \u{c9}t\u{e9}.", "One. \u{c9}t\u{e9}.", &.{5}, 0);
    try expectRead("One. \u{b6} \u{ff12}.", "One. \u{ff12}.", &.{5}, 0);
    // punctuation alone before a mark is no segment either
    try expectRead("\u{2026} \u{b6} Two.", "\u{2026} Two.", &.{}, 1);
}

test "a cue that trails a sentence goes after the mark: after the cue, the mark is dropped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectRead("Done now. [small laugh] \u{b6} Next up.", "Done now. [small laugh] Next up.", &.{}, 1);
    const kept = try read(a, "Done now. \u{b6} [small laugh] Next up.", true);
    const got = try split(a, kept, 500);
    try std.testing.expectEqualStrings("Done now.", got.segments[0]);
    try std.testing.expectEqualStrings("[small laugh] Next up.", got.segments[1]);
}

test "a part longer than the segment limit is split as long-form splits it; the shifts mark the honoured cuts only" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const marked = try read(a, "Aaaa. Bbbb. Cccc. \u{b6} Dddd. \u{b6} Eeee.", true);
    const got = try split(a, marked, 12);
    try std.testing.expectEqual(@as(usize, 4), got.segments.len);
    for ([_][]const u8{ "Aaaa.", "Bbbb. Cccc.", "Dddd.", "Eeee." }, got.segments) |want, s| try std.testing.expectEqualStrings(want, s);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, got.shifts);
    // the segments are the text's own slices, as the redraws' offsets need
    try std.testing.expectEqual(@intFromPtr(marked.text.ptr) + 18, @intFromPtr(got.segments[2].ptr));
    // short enough for one segment, a mark still cuts it; without a cut, the long-form split as it is
    try std.testing.expectEqual(@as(usize, 2), (try split(a, try read(a, "One. \u{b6} Two.", true), 500)).segments.len);
    const plain = try split(a, .{ .text = "One. Two." }, 5);
    try std.testing.expect(plain.segments.len == 2 and plain.shifts.len == 0);
}

test "a shift's tone: the first source with a room's quiet, never digital silence, else noise, every sample under the peak" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rate = 1000;
    const loud: [300]f32 = @splat(0.5);
    var zeros: [300]f32 = @splat(0.5);
    @memset(zeros[100..250], 0);
    var hushed: [300]f32 = @splat(0.5);
    for (hushed[150..250], 0..) |*v, i| v.* = if (i % 2 == 0) 0.0075 else -0.0075; // -42.5 dBFS: silent, peaks over
    // the tail is loud, the earlier audio digital silence, a short source none: the next segment's quiet is used
    const found = try tone(a, &.{ &loud, &zeros, hushed[0..90], &hushed }, rate);
    try std.testing.expectEqual(@as(usize, 100), found.len);
    for (found) |v| try std.testing.expectApproxEqAbs(tone_peak, @abs(v), 1e-7);
    // nothing with a room's quiet: noise at -65 dBFS, never a zero stretch
    const made = try tone(a, &.{ &loud, &zeros }, rate);
    try std.testing.expectApproxEqAbs(@as(f32, -65), drama_joins.levelDb(made), 1.5);
    for (made) |v| try std.testing.expect(@abs(v) < tone_peak);
    try std.testing.expect(drama_joins.peak(made) > 0);
}

test "a shift's fill is the tone crossfaded at equal power, mirrored at both seams, never changing either segment" {
    const rate = 1000; // a 15-sample crossfade
    var left: [40]f32 = undefined;
    for (&left, 0..) |*v, i| v.* = 0.01 * @as(f32, @floatFromInt(i));
    const right: [40]f32 = @splat(-0.3);
    const t: [10]f32 = @splat(0.002);
    var out: [100]f32 = undefined;
    fill(&left, &right, &t, rate, &out);
    // continuous at both seams: the first sample is mostly left's last, the last mostly right's first
    try std.testing.expectApproxEqAbs(left[39], out[0], 0.002);
    try std.testing.expectApproxEqAbs(right[0], out[99], 0.003);
    for (out[15..85]) |v| try std.testing.expectEqual(@as(f32, 0.002), v);
    for (0..15) |j| {
        const theta = std.math.pi / 2.0 * (@as(f32, @floatFromInt(j)) + 0.5) / 15.0;
        try std.testing.expectApproxEqAbs(left[39 - j] * @cos(theta) + 0.002 * @sin(theta), out[j], 1e-6);
        try std.testing.expectApproxEqAbs(right[j] * @cos(theta) + 0.002 * @sin(theta), out[99 - j], 1e-6);
    }
    // a fill shorter than two crossfades splits itself between them
    var short: [6]f32 = undefined;
    fill(&left, &right, &t, rate, &short);
    try std.testing.expectApproxEqAbs(left[39] * @cos(std.math.pi / 12.0) + 0.002 * @sin(std.math.pi / 12.0), short[0], 1e-6);
}

test "the quiet run around a sample spans head and tail on one 10 ms grid" {
    const rate = 1000;
    var head: [105]f32 = @splat(0.5);
    @memset(head[40..], 0.001);
    var tail: [60]f32 = @splat(0.001);
    @memset(tail[25..], 0.5);
    // frames 4 to 12 are silent (40 to 130); frame 13 (130-140) holds 125-130's quiet and 130's speech
    try std.testing.expectEqual([2]usize{ 40, 120 + 10 }, quietRun(&head, &tail, rate, 104));
    try std.testing.expectEqual([2]usize{ 5, 5 }, quietRun(&head, &tail, rate, 5));
    try std.testing.expectEqual(@as(usize, 0), room(0, 44100));
    try std.testing.expectEqual(2 * 37485 + 3 * 441 + 2 * 661, room(0.85, 44100));
}

test "the spans header lists each start-end, and its largest value fits its buffer" {
    var buf: [spans_header_max]u8 = undefined;
    try std.testing.expectEqualStrings("2048-40000,90000-131072", spansHeader(&buf, &.{ .{ 2048, 40000 }, .{ 90000, 131072 } }));
    const widest: [drama_long.max_segments - 1][2]usize = @splat(@splat(std.math.maxInt(u64)));
    try std.testing.expectEqual(spans_header_max, spansHeader(&buf, &widest).len);
}
