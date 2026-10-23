//! --silence-redraws: a whole reply the silence guard ended, or that stopped far short, is drawn again in time.
const std = @import("std");
const api = @import("engine_api");
const reply_text = @import("reply_text.zig");
const drama_prompt = @import("drama_prompt.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Allocator = std.mem.Allocator;

/// At most `max` redraws, each started only if it fits `budget_s`; `plain`: see `plain_scope`; `short_ratio`: `trip`.
pub const Policy = struct { max: u8 = 0, budget_s: f64 = 120, plain: bool = false, plain_scope: Scope = .cues, short_ratio: f64 = 0 };

/// What a plain redraw drops: the five cues the wrapper wrote, or every Drama 3 control (the scorer's ``spoken()``).
pub const Scope = enum { cues, controls };

/// Characters an input speaks per second of audio, as round 4's replies measured; a reply's expected length.
pub const chars_per_second = 15.6;

/// Inputs shorter than this never trip as short: a reply's fixed lead-in and tail make their rate noisy.
pub const short_min_chars = 200;

/// Why a draw is redrawn: the guard's silence, or a stop far short of its input (a stall, then end of speech).
pub const Trip = enum { silence, short };

/// The text a draw speaks, and its prompt.
pub const Spoken = struct { text: []const u8, rows: drama_prompt.Rows };

/// The cues a caller's wrapper wrote and stripped on its own retry, before the server redrew.
const cues = [_][]const u8{ "[sighs]", "[laughs]", "[small laugh]", "[pause]", "[hesitates]" };

/// Drama 3's documented scoped tags: `controls` drops their opening and closing tags and keeps the text between.
pub const scoped_tags = [_][]const u8{ "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful" };

/// One reply's redraws: the budget from its receipt, and how many were admitted to the engine.
pub const Redraws = struct {
    policy: Policy,
    budget_s: f64,
    received: i96,
    used: u8 = 0,
    /// --silence-mid-s on: the draws it ended as silence; null when it is off.
    hollow: ?u8 = null,

    pub fn init(policy: Policy, received: i96) Redraws {
        return .{ .policy = policy, .budget_s = policy.budget_s, .received = received };
    }

    /// Lowers the budget to the client's X-Deadline-S.
    pub fn limit(r: *Redraws, deadline_s: f64) void {
        r.budget_s = @min(deadline_s, r.budget_s);
    }

    /// The header value every response carries when redraws are on; null when they are off, so replies are unchanged.
    pub fn header(r: *const Redraws, buf: []u8) ?[]const u8 {
        if (r.policy.max == 0) return null;
        return std.fmt.bufPrint(buf, "{d}", .{r.used}) catch unreachable;
    }

    /// Counts and logs a draw --silence-mid-s ended as silence, end of speech in a hole of `samples` at `rate`.
    pub fn hollowed(r: *Redraws, id: []const u8, samples: usize, rate: u32) void {
        if (samples == 0) return;
        if (r.hollow) |*n| {
            n.* +|= 1;
            log.line("speech {s} silence mid trip: hole={d:.2}s trips={d}", .{ id, @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(rate)), n.* });
        }
    }

    /// The X-Silence-Mid-Trips value whole replies carry with --silence-mid-s on; null when it is off.
    pub fn hollowHeader(r: *const Redraws, buf: []u8) ?[]const u8 {
        return std.fmt.bufPrint(buf, "{d}", .{r.hollow orelse return null}) catch unreachable;
    }

    /// The seed salt of a synthesis's attempt `k` (0 first, then its redraws): the server's salt plus `k`.
    pub fn salt(base: i64, k: u8) i64 {
        return base +% k;
    }

    /// Whether a draw of `text` that ended `reason` with `samples` at `rate` trips: silence, or a short stop.
    pub fn trip(r: *const Redraws, reason: api.Reason, samples: usize, rate: u32, text: []const u8) ?Trip {
        if (reason == .silence) return .silence;
        // bytes bound code points from above: a short text or an off ratio is rejected before any scan
        if (reason != .stop or r.policy.short_ratio == 0 or text.len < short_min_chars) return null;
        const chars = std.unicode.utf8CountCodepoints(text) catch return null;
        if (chars < short_min_chars) return null;
        const seconds = @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(rate));
        const expected = @as(f64, @floatFromInt(chars)) / chars_per_second;
        return if (seconds < r.policy.short_ratio * expected) .short else null;
    }

    /// After a draw begun at `started` tripped, logged: if a redraw is due, the latest time it may start at.
    pub fn again(r: *const Redraws, io: std.Io, id: []const u8, started: i96, cause: Trip) ?i96 {
        if (r.policy.max == 0) return null;
        const f = r.fit(io, started);
        const go = r.used < r.policy.max and f.start_by != null;
        const why = if (r.used == r.policy.max) "no redraws left" else if (go) "redraw" else "t+d past the budget";
        log.line("speech {s} silence redraw decision: trip={s} attempt={d} t={d:.2}s d={d:.2}s t+d={d:.2}s budget={d:.2}s redraws={d}/{d}: {s}", .{ id, @tagName(cause), r.used, f.t, f.d, f.t + f.d, r.budget_s, r.used, r.policy.max, why });
        return if (go) f.start_by else null;
    }

    /// Now `t` s from receipt after a `d` s draw begun at `started`: a redraw as long fits if t + d is in the budget.
    pub fn fit(r: *const Redraws, io: std.Io, started: i96) struct { t: f64, d: f64, start_by: ?i96 } {
        const now = clock.nowNs(io);
        const t = clock.seconds(now - r.received);
        const d = clock.seconds(now - started);
        const by = r.received + @as(i96, @intFromFloat(r.budget_s * std.time.ns_per_s)) - (now - started);
        return .{ .t = t, .d = d, .start_by = if (t + d <= r.budget_s) by else null };
    }

    /// A redraw not taken after all, logged: the tripped draw is the reply.
    pub fn stand(r: *const Redraws, id: []const u8, why: []const u8) void {
        log.line("speech {s} silence redraw not taken after {d}/{d}: {s}", .{ id, r.used, r.policy.max, why });
    }
};

/// The request's X-Deadline-S: a positive, finite number of seconds; null when absent; error when malformed.
pub fn deadline(value: ?[]const u8) error{Malformed}!?f64 {
    const text = std.mem.trim(u8, value orelse return null, " \t");
    const s = std.fmt.parseFloat(f64, text) catch return error.Malformed;
    if (!std.math.isFinite(s) or s <= 0) return error.Malformed;
    return s;
}

/// Why `text`, an input without its cues, cannot be spoken; null when it can.
pub fn problem(a: Allocator, text: []const u8) ?[]const u8 {
    if (text.len == 0) return "nothing is left of the input without its cues";
    return drama_prompt.problem(a, text) catch "out of memory";
}

/// `input` heard without `scope` (``re.sub(r'\s+', ' ', s).strip()`` after each is a space), maybe empty; null: none.
pub fn plain(a: Allocator, input: []const u8, scope: Scope) Allocator.Error!?[]const u8 {
    const spans = try dropped(a, input, scope) orelse return null;
    return try heard(a, input, spans, 0, input.len);
}

/// A stretch of the input a plain redraw drops.
pub const Span = struct { start: usize, end: usize };

/// What `scope` drops from the whole `input`, in order; null: nothing. Long-form segments read their range of these.
pub fn dropped(a: Allocator, input: []const u8, scope: Scope) Allocator.Error!?[]const Span {
    var spans: std.ArrayList(Span) = .empty;
    var i: usize = 0;
    if (scope == .controls) {
        i = directionHead(input);
        if (i > 0) try spans.append(a, .{ .start = 0, .end = i });
    }
    // once no ']' follows, no later '[' can close either: each byte is scanned at most twice
    var closes = true;
    while (i < input.len) {
        const n = controlAt(scope, input[i..], &closes);
        if (n == 0) {
            i += 1;
            continue;
        }
        try spans.append(a, .{ .start = i, .end = i + n });
        i += n;
    }
    return if (spans.items.len == 0) null else spans.items;
}

/// `input[start..end]` heard with `spans` dropped, as `plain` makes it; null when no span reaches into the range.
pub fn heard(a: Allocator, input: []const u8, spans: []const Span, start: usize, end: usize) Allocator.Error!?[]const u8 {
    var k: usize = 0;
    while (k < spans.len and spans[k].end <= start) k += 1;
    if (k == spans.len or spans[k].start >= end) return null;
    // a dropped span is white space, as the space it becomes; each run is one space between words, none at either end
    var out: std.ArrayList(u8) = .empty;
    var gap = false;
    var i = start;
    while (i < end) {
        if (k < spans.len and spans[k].start <= i) {
            gap = true;
            i = @min(spans[k].end, end);
            k += 1;
            continue;
        }
        const n = reply_text.spaceAt(input, i);
        if (n > 0) {
            gap = true;
            i += n;
            continue;
        }
        if (gap and out.items.len > 0) try out.append(a, ' ');
        gap = false;
        try out.append(a, input[i]);
        i += 1;
    }
    return out.items;
}

/// The bytes of the control `rest` opens with, 0 when none: a cue; or a [..] span (if a ']' may follow), a scoped tag.
fn controlAt(scope: Scope, rest: []const u8, closes: *bool) usize {
    switch (scope) {
        .cues => for (cues) |cue| {
            if (std.mem.startsWith(u8, rest, cue)) return cue.len;
        },
        .controls => if (rest[0] == '[' and closes.*) {
            if (std.mem.indexOfScalar(u8, rest, ']')) |end| return end + 1;
            closes.* = false;
        } else if (rest[0] == '<') {
            const name = rest[@as(usize, 1) + @intFromBool(rest.len > 1 and rest[1] == '/') ..];
            for (scoped_tags) |tag| if (std.mem.startsWith(u8, name, tag) and name.len > tag.len and name[tag.len] == '>')
                return rest.len - name.len + tag.len + 1;
        },
    }
    return 0;
}

/// The bytes of a director's note head (``^\s*Direction:.*\nLine:``): the first line, then its 'Line:'; 0 when none.
fn directionHead(input: []const u8) usize {
    var i: usize = 0;
    while (i < input.len) {
        const n = reply_text.spaceAt(input, i);
        if (n == 0) break;
        i += n;
    }
    if (!std.mem.startsWith(u8, input[i..], "Direction:")) return 0;
    const end = std.mem.indexOfScalarPos(u8, input, i, '\n') orelse return 0;
    return if (std.mem.startsWith(u8, input[end + 1 ..], "Line:")) end + 1 + "Line:".len else 0;
}

test "the plain input drops exactly the five cues, as the wrapper's regex does" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("done. Next", (try plain(a, "done. [pause] Next", .cues)).?);
    try std.testing.expectEqualStrings("word .", (try plain(a, "word [sighs].", .cues)).?);
    try std.testing.expectEqualStrings("a b c", (try plain(a, " [laughs]a\t[small laugh]\n b\u{3000}[hesitates]c ", .cues)).?);
    // other brackets, case and spacing inside the cue are text, as \[(?:sighs|...)\] matches them
    try std.testing.expect(try plain(a, "[Sighs] [ pause] [cough] [whispers] no cue", .cues) == null);
    // controls that are not cues stay: today's plain redraw, as it was
    try std.testing.expect(try plain(a, "[pleased] <emphasis>x</emphasis>\nDirection: d\nLine: y", .cues) == null);
    try std.testing.expectEqualStrings("<soft>x</soft> [pleased] y", (try plain(a, "<soft>x</soft> [pleased] [pause] y", .cues)).?);
    // cues and nothing else: an empty text, not the input as it is
    try std.testing.expectEqualStrings("", (try plain(a, " [pause] [sighs] ", .cues)).?);
}

test "plain controls drop what the scorer's spoken() drops, scoped tags only from Drama 3's documented set" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // each expected text is Python's re.sub(r'\s+', ' ', CONTROLS.sub(' ', s)).strip() with the tag set below
    const cases = [_]struct { []const u8, ?[]const u8 }{
        .{ "[pleased] Good news. <emphasis>It works</emphasis>, finally.", "Good news. It works , finally." },
        .{ "Direction: warm, then [slower]\nLine: [warmly] Hello <soft>there</soft>.", "Hello there ." },
        .{ " \n Direction: x\nLine:Hi", "Hi" },
        .{ "<fast>quick</fast><slow>slow</slow><stress>hit</stress><forceful>no</forceful>", "quick slow hit no" },
        .{ "<emphasis>a</emphasis", "a</emphasis" },
        .{ "[unclosed and <whisper>text</whisper>", "[unclosed and text" },
        .{ "Say: Direction: x\nLine: y [pause]", "Say: Direction: x Line: y" },
        .{ "a [b\n c] d", "a d" },
        // the first ']' closes; once none follows, a '[' is text and later controls still go
        .{ "[a [b] c]", "c]" },
        .{ "[[[[ x <soft>y</soft> [z", "[[[[ x y [z" },
        .{ "[pleased] <soft></soft> ", "" },
        // no control: a note whose 'Line:' is not on its second line, other or capitalised tags
        .{ "Direction: x\nLater\nLine: no", null },
        .{ "<b>bold</b> <french>bonjour</french> <Soft>x</Soft>", null },
    };
    for (cases) |case| {
        const got = try plain(a, case[0], .controls);
        if (case[1]) |want| try std.testing.expectEqualStrings(want, got.?) else try std.testing.expect(got == null);
    }
}

test "a range of the input is heard with the whole input's controls: one crossing its edge, a head only at the start" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = "Direction: d\nLine: One [warmly. two] three. Direction: e\nLine: four";
    const spans = (try dropped(a, input, .controls)).?;
    const cut = std.mem.indexOf(u8, input, "two").?;
    try std.testing.expectEqualStrings("One", (try heard(a, input, spans, 0, cut)).?);
    try std.testing.expectEqualStrings("three. Direction: e Line: four", (try heard(a, input, spans, cut, input.len)).?);
    // a range no span reaches is spoken as it is
    const four = std.mem.indexOf(u8, input, "four").?;
    try std.testing.expect(try heard(a, input, spans, four, input.len) == null);
}

test "X-Deadline-S is a positive number of seconds; the budget is the lower of it and the server's" {
    try std.testing.expectEqual(@as(?f64, 96), try deadline("96"));
    try std.testing.expectEqual(@as(?f64, 2.5), try deadline(" 2.5 "));
    try std.testing.expectEqual(@as(?f64, null), try deadline(null));
    for ([_][]const u8{ "", "0", "-3", "nan", "inf", "soon" }) |bad| try std.testing.expectError(error.Malformed, deadline(bad));
    for ([_]struct { f64, f64 }{ .{ 30, 30 }, .{ 600, 120 }, .{ 120, 120 } }) |case| {
        var r: Redraws = .init(.{ .max = 2, .budget_s = 120 }, 0);
        r.limit(case[0]);
        try std.testing.expectEqual(case[1], r.budget_s);
    }
}

test "a stop of 200 characters or more trips as short only under the ratio of its expected length; silence always" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const repeat = struct {
        fn f(al: Allocator, unit: []const u8, n: usize) ![]const u8 {
            const out = try al.alloc(u8, unit.len * n);
            for (0..n) |i| @memcpy(out[i * unit.len ..][0..unit.len], unit);
            return out;
        }
    }.f;
    // 312 characters are 20 s expected: at a ratio of 0.5, a stop under 10 s of audio is short
    const r: Redraws = .init(.{ .max = 1, .short_ratio = 0.5 }, 0);
    const long = try repeat(a, "a", 312);
    try std.testing.expectEqual(@as(?Trip, .short), r.trip(.stop, 9999, 1000, long));
    try std.testing.expectEqual(@as(?Trip, null), r.trip(.stop, 10_000, 1000, long));
    try std.testing.expectEqual(@as(?Trip, null), r.trip(.length, 10, 1000, long));
    try std.testing.expectEqual(@as(?Trip, .silence), r.trip(.silence, 60_000, 1000, long));
    // a 150-character reply of 1 s is a ratio of 0.10, but too short an input to judge; 200 characters are judged
    try std.testing.expectEqual(@as(?Trip, null), r.trip(.stop, 1000, 1000, try repeat(a, "a", 150)));
    try std.testing.expectEqual(@as(?Trip, .short), r.trip(.stop, 1000, 1000, try repeat(a, "a", 200)));
    try std.testing.expectEqual(@as(?Trip, .silence), r.trip(.silence, 1000, 1000, try repeat(a, "a", 150)));
    // characters are code points: 199 of two bytes each are too few, 200 are enough
    try std.testing.expectEqual(@as(?Trip, null), r.trip(.stop, 1000, 1000, try repeat(a, "é", 199)));
    try std.testing.expectEqual(@as(?Trip, .short), r.trip(.stop, 1000, 1000, try repeat(a, "é", 200)));
    const off: Redraws = .init(.{ .max = 1 }, 0);
    try std.testing.expectEqual(@as(?Trip, null), off.trip(.stop, 1, 1000, long));
}
