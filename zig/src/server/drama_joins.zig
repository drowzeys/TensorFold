//! Long-form joins: the pause a join's text asks for, made up from room tone, and one speech level across segments.
const std = @import("std");
const drama_long = @import("drama_long.zig");

/// Frames quieter than this (10 ms, RMS dBFS) are silence; louder ones are speech.
pub const floor_db: f32 = -42;
/// The most one join may add, a paragraph's pause, in seconds.
pub const max_pause_seconds: f32 = 0.70;
/// How far a segment's speech level may sit from the earlier segments' median before it is moved, and how far it is.
const level_slack_db: f32 = 1.5;
const max_gain_db: f32 = 3;
const peak_limit: f32 = 0.99;

/// What ends the text before a join, which sets the pause the join should have.
pub const Boundary = enum {
    sentence,
    ellipsis,
    topic,
    paragraph,
    clause,

    pub fn seconds(b: Boundary) f32 {
        return switch (b) {
            .sentence => 0.42,
            .ellipsis => 0.55,
            .topic => 0.60,
            .paragraph => 0.70,
            .clause => 0.20,
        };
    }
};

/// The join between segments `left` and `right` (`gap` between): a blank line, a turn, else the punctuation.
pub fn boundary(left: []const u8, gap: []const u8, right: []const u8) Boundary {
    if (drama_long.paragraph(gap)) return .paragraph;
    if (drama_long.turns(left, gap, right)) return .topic;
    var end = left.len;
    while (end > 0) {
        const at = lastStart(left[0..end]);
        const cp = std.unicode.utf8Decode(left[at..end]) catch break;
        switch (cp) {
            '"', '\'', ')', ']', 0x201d, 0x2019, 0xbb => end = at,
            0x2026 => return .ellipsis,
            '.' => return if (std.mem.endsWith(u8, left[0..end], "...")) .ellipsis else .sentence,
            '!' => return .sentence,
            else => break,
        }
    }
    return if (std.mem.indexOfAny(u8, gap, "\n\r") != null) .sentence else .clause;
}

/// The start of the last code point of valid UTF-8 `text`.
fn lastStart(text: []const u8) usize {
    var at = text.len - 1;
    while (at > 0 and text[at] & 0xc0 == 0x80) at -= 1;
    return at;
}

/// RMS level of `x` in dBFS; -inf for digital silence.
pub fn levelDb(x: []const f32) f32 {
    var sum: f64 = 0;
    for (x) |v| sum += @as(f64, v) * v;
    return @floatCast(10 * std.math.log10(sum / @as(f64, @floatFromInt(x.len))));
}

pub fn frameLen(rate: u32) usize {
    return @max(1, rate / 100);
}

/// Samples of whole 10 ms silent frames counted back from the end of `pcm`.
pub fn trailingSilence(pcm: []const f32, rate: u32) usize {
    const n = frameLen(rate);
    var silent: usize = 0;
    while ((silent + 1) * n <= pcm.len and levelDb(pcm[pcm.len - (silent + 1) * n ..][0..n]) < floor_db) silent += 1;
    return silent * n;
}

/// Samples of whole 10 ms silent frames from the start of `pcm`.
pub fn leadingSilence(pcm: []const f32, rate: u32) usize {
    const n = frameLen(rate);
    var silent: usize = 0;
    while ((silent + 1) * n <= pcm.len and levelDb(pcm[silent * n ..][0..n]) < floor_db) silent += 1;
    return silent * n;
}

/// Samples to add between `left` and `right` so their silence reaches `b`'s pause; existing silence is never cut.
pub fn shortfall(left: []const f32, right: []const f32, rate: u32, b: Boundary) usize {
    const target: usize = @intFromFloat(@round(b.seconds() * @as(f32, @floatFromInt(rate))));
    return target -| (trailingSilence(left, rate) + leadingSilence(right, rate));
}

/// `out` filled back and forth from `left`'s quietest 100 ms in its last second; zeros if none or none is silent.
pub fn roomTone(left: []const f32, rate: u32, out: []f32) void {
    const tone = quietest(left[left.len - @min(left.len, 100 * frameLen(rate)) ..], rate, -std.math.inf(f32)) orelse return @memset(out, 0);
    for (out, 0..) |*v, i| v.* = play(tone, i);
}

/// `pcm`'s quietest silent 100 ms of `least_db` or more on the 10 ms grid, its last 100 ms first; null if none.
pub fn quietest(pcm: []const f32, rate: u32, least_db: f32) ?[]const f32 {
    const step = frameLen(rate);
    const width = 10 * step;
    if (pcm.len < width) return null;
    var best: ?usize = null;
    var best_db = floor_db;
    var at = pcm.len - width;
    var next: usize = 0;
    while (true) {
        const db = levelDb(pcm[at..][0..width]);
        if (db < best_db and db >= least_db) {
            best = at;
            best_db = db;
        }
        if (next + width > pcm.len) break;
        at = next;
        next += step;
    }
    return pcm[(best orelse return null)..][0..width];
}

/// Sample `i` of `tone` played back and forth.
pub fn play(tone: []const f32, i: usize) f32 {
    const p = i % (2 * tone.len);
    return tone[if (p < tone.len) p else 2 * tone.len - 1 - p];
}

/// Median dBFS of `pcm`'s 10 ms speech frames, or null when it has none.
pub fn loudness(a: std.mem.Allocator, pcm: []const f32, rate: u32) std.mem.Allocator.Error!?f32 {
    const n = frameLen(rate);
    var levels: std.ArrayList(f32) = .empty;
    defer levels.deinit(a);
    var at: usize = 0;
    while (at + n <= pcm.len) : (at += n) {
        const db = levelDb(pcm[at..][0..n]);
        if (db >= floor_db) try levels.append(a, db);
    }
    return median(levels.items);
}

/// The median of `values` (sorted in place), or null when empty.
pub fn median(values: []f32) ?f32 {
    if (values.len == 0) return null;
    std.mem.sort(f32, values, {}, std.sort.asc(f32));
    const mid = values.len / 2;
    return if (values.len % 2 == 1) values[mid] else (values[mid - 1] + values[mid]) / 2;
}

/// A segment's gain against the earlier median: none within 1.5 dB, else to 1.5 dB, at most 3 dB, peak under 0.99.
pub fn gain(level: ?f32, earlier: ?f32, top: f32) f32 {
    const l = level orelse return 1;
    const m = earlier orelse return 1;
    const d = l - m;
    if (@abs(d) <= level_slack_db) return 1;
    const db = std.math.clamp(-(d - std.math.sign(d) * level_slack_db), -max_gain_db, max_gain_db);
    const linear = std.math.pow(f32, 10, db / 20);
    if (linear > 1 and top * linear > peak_limit) return @max(1, peak_limit / top);
    return linear;
}

/// The largest magnitude in `pcm`.
pub fn peak(pcm: []const f32) f32 {
    var p: f32 = 0;
    for (pcm) |v| p = @max(p, @abs(v));
    return p;
}

test "a join's pause: a blank line, then a turn in the text, then the punctuation before it" {
    const cases = [_]struct { left: []const u8, gap: []const u8, right: []const u8 = "and on.", want: Boundary }{
        .{ .left = "It ends.", .gap = " ", .want = .sentence },
        .{ .left = "It ends!", .gap = " ", .want = .sentence },
        .{ .left = "Does it?", .gap = " ", .want = .topic },
        .{ .left = "Wait!?", .gap = " ", .want = .topic },
        .{ .left = "It ends.", .gap = " ", .right = "So, next.", .want = .topic },
        .{ .left = "And then\u{2026}", .gap = " ", .right = "Anyway, on.", .want = .topic },
        .{ .left = "And then\u{2026}", .gap = " ", .want = .ellipsis },
        .{ .left = "And then...", .gap = " ", .want = .ellipsis },
        .{ .left = "He said \"stop.\"", .gap = " ", .want = .sentence },
        .{ .left = "Is it \u{201c}so?\u{201d}", .gap = " ", .want = .topic },
        .{ .left = "It ends.", .gap = "\n\n", .want = .paragraph },
        .{ .left = "Does it?", .gap = " \r\n \r\n", .right = "So.", .want = .paragraph },
        .{ .left = "A line", .gap = "\n", .want = .sentence },
        .{ .left = "first, second;", .gap = " ", .want = .clause },
        .{ .left = "cut at a space", .gap = " ", .want = .clause },
    };
    for (cases) |c| try std.testing.expectEqual(c.want, boundary(c.left, c.gap, c.right));
    const seconds = [_]f32{ 0.42, 0.55, 0.60, 0.70, 0.20 };
    for (std.enums.values(Boundary), seconds) |b, s| try std.testing.expectEqual(s, b.seconds());
}

test "only the shortfall to the target is added; silence already there is never cut" {
    const rate = 1000; // 10-sample frames: the sentence target is 420 samples
    var speech: [100]f32 = @splat(0.5);
    var quiet: [100]f32 = @splat(0.5);
    @memset(quiet[70..], 0.001); // 3 silent frames at the end
    var head: [100]f32 = @splat(0.5);
    @memset(head[0..25], 0); // 2 whole silent frames at the start; the third is half speech
    try std.testing.expectEqual(@as(usize, 30), trailingSilence(&quiet, rate));
    try std.testing.expectEqual(@as(usize, 20), leadingSilence(&head, rate));
    try std.testing.expectEqual(@as(usize, 420), shortfall(&speech, &speech, rate, .sentence));
    try std.testing.expectEqual(@as(usize, 420 - 50), shortfall(&quiet, &head, rate, .sentence));
    try std.testing.expectEqual(@as(usize, 700 - 50), shortfall(&quiet, &head, rate, .paragraph));
    var long_quiet: [800]f32 = @splat(0);
    try std.testing.expectEqual(@as(usize, 0), shortfall(&long_quiet, &speech, rate, .paragraph));
}

test "room tone is the quietest tail stretch played back and forth; zeros without a quiet tail" {
    const rate = 1000;
    var left: [300]f32 = @splat(0.5);
    for (left[150..250], 0..) |*v, i| v.* = 0.001 * @as(f32, @floatFromInt(i % 7)); // the quiet stretch
    var out: [250]f32 = undefined;
    roomTone(&left, rate, &out);
    for (0..100) |i| try std.testing.expectEqual(left[150 + i], out[i]);
    for (0..100) |i| try std.testing.expectEqual(left[249 - i], out[100 + i]);
    try std.testing.expectEqual(left[150], out[200]);
    const loud: [300]f32 = @splat(0.5);
    for ([_][]const f32{ left[0..50], &loud }) |no_tail| {
        @memset(&out, 1);
        roomTone(no_tail, rate, &out);
        for (out) |v| try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "one gain a segment: none within 1.5 dB, else to 1.5 dB, clamped to 3 dB and kept under a 0.99 peak" {
    try std.testing.expectEqual(@as(f32, 1), gain(-20, -21.4, 0.5));
    try std.testing.expectEqual(@as(f32, 1), gain(null, -20, 0.5));
    try std.testing.expectEqual(@as(f32, 1), gain(-20, null, 0.5));
    try std.testing.expectApproxEqAbs(std.math.pow(f32, 10, -1.0 / 20.0), gain(-17.5, -20, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(std.math.pow(f32, 10, 1.0 / 20.0), gain(-22.5, -20, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(std.math.pow(f32, 10, -3.0 / 20.0), gain(-10, -20, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(std.math.pow(f32, 10, 3.0 / 20.0), gain(-30, -20, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.99 / 0.9), gain(-30, -20, 0.9), 1e-6);
    try std.testing.expectEqual(@as(f32, 1), gain(-30, -20, 1.0));
    var a: [4]f32 = .{ -3, -1, -4, -2 };
    try std.testing.expectEqual(@as(f32, -2.5), median(&a).?);
    var segment: [40]f32 = @splat(0.5);
    @memset(segment[30..], 0);
    const level = try loudness(std.testing.allocator, &segment, 1000);
    try std.testing.expectApproxEqAbs(@as(f32, 20 * std.math.log10(0.5)), level.?, 1e-4);
}

test "the quietest silent 100 ms of a stretch, if any, above a lower bound when one is given" {
    const rate = 1000;
    var pcm: [400]f32 = @splat(0.5);
    for (pcm[120..220], 0..) |*v, i| v.* = 0.001 * @as(f32, @floatFromInt(i % 5));
    const found = quietest(&pcm, rate, -std.math.inf(f32)).?;
    try std.testing.expectEqual(@intFromPtr(&pcm[120]), @intFromPtr(found.ptr));
    try std.testing.expectEqual(@as(usize, 100), found.len);
    // a stretch with no quiet 100 ms, or shorter than 100 ms, has none
    try std.testing.expect(quietest(pcm[300..], rate, -std.math.inf(f32)) == null and quietest(pcm[0..99], rate, -std.math.inf(f32)) == null);
    // digital silence is the quietest of all, unless the bound passes over it
    @memset(pcm[300..400], 0);
    try std.testing.expectEqual(@intFromPtr(&pcm[300]), @intFromPtr(quietest(&pcm, rate, -std.math.inf(f32)).?.ptr));
    try std.testing.expectEqual(@intFromPtr(&pcm[120]), @intFromPtr(quietest(&pcm, rate, -90).?.ptr));
}
