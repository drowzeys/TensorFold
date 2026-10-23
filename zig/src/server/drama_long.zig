//! Drama 3 long-form speech: an input cut at sentence ends into segments, each spoken after hearing those before it.
const std = @import("std");
const drama_prompt = @import("drama_prompt.zig");
const Reference = drama_prompt.Reference;
const Allocator = std.mem.Allocator;

/// --long-form's segments of at most `chars`, each hearing `history` before it; --shift-pause-s: seconds at each `¶`.
pub const Options = struct { chars: u32 = 500, history: u8 = 1, shift_pause_s: f32 = 0 };

pub const max_history = 3;
pub const max_segments = 12;
/// Frames one long-form reply may take in all; each segment gets what is left, up to the engine's own limit.
pub const max_frames = 6000;

/// Where text may be cut, coarsest first.
const Level = enum { sentence, clause, space, any };

/// The input's segments: the fewest that fit `chars`, balanced, ending where the text turns (see `balance`).
pub fn split(a: Allocator, input: []const u8, chars: u32) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var run: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (nextPiece(input, &at, .sentence, chars)) |piece| {
        if (count(piece) <= chars) {
            try run.append(a, piece);
            continue;
        }
        try balance(a, &out, run.items, chars);
        run.clearRetainingCapacity();
        // a sentence over `chars` is cut at ``; : ,``, then spaces, then anywhere, into segments of its own
        try pack(a, &out, piece, .clause, chars);
    }
    try balance(a, &out, run.items, chars);
    return out.items;
}

/// A cut where the text turns may sit about 15% of a segment off a two-way balanced split and still be chosen.
const turn_bonus = 0.3;

/// The fewest segments of `sentences` fitting `chars`: cuts minimizing relative length deviation less `turn_bonus`s.
fn balance(a: Allocator, out: *std.ArrayList([]const u8), sentences: []const []const u8, chars: u32) Allocator.Error!void {
    const k = sentences.len;
    if (k == 0) return;
    // code point offsets of each sentence's start and end within the run
    const starts = try a.alloc(usize, k);
    const ends = try a.alloc(usize, k);
    var cp: usize = 0;
    for (sentences, 0..) |s, i| {
        cp += if (i == 0) 0 else count(sentences[i - 1].ptr[sentences[i - 1].len .. @intFromPtr(s.ptr) - @intFromPtr(sentences[i - 1].ptr)]);
        starts[i] = cp;
        cp += count(s);
        ends[i] = cp;
    }
    // the fewest segments: greedy packing is optimal for a length cap
    var greedy: std.ArrayList(usize) = .empty;
    var first: usize = 0;
    while (first < k) {
        try greedy.append(a, first);
        var last = first;
        while (last + 1 < k and ends[last + 1] - starts[first] <= chars) last += 1;
        first = last + 1;
    }
    const n = greedy.items.len;
    // one segment has nothing to balance; past the segment cap the reply is refused, so any cuts will do
    if (n == 1 or out.items.len + n > max_segments) {
        try greedy.append(a, k);
        for (greedy.items[0..n], greedy.items[1..]) |head, next| try out.append(a, span(sentences[head], sentences[next - 1]));
        return;
    }
    const mean = @as(f64, @floatFromInt(ends[k - 1])) / @as(f64, @floatFromInt(n));
    // cost[m * (k + 1) + j]: sentences 0..j in m segments; from: where the last of them starts
    const cost = try a.alloc(f64, (n + 1) * (k + 1));
    const from = try a.alloc(usize, (n + 1) * (k + 1));
    @memset(cost, std.math.inf(f64));
    cost[0] = 0;
    for (1..n + 1) |m| for (m..k + 1) |j| {
        var i = j;
        while (i > m - 1) {
            i -= 1;
            if (ends[j - 1] - starts[i] > chars) break;
            const before = cost[(m - 1) * (k + 1) + i];
            if (before == std.math.inf(f64)) continue;
            const length = @as(f64, @floatFromInt(ends[j - 1] - starts[i]));
            var c = before + @abs(length - mean) / mean;
            if (i > 0 and turns(sentences[i - 1], gapBetween(sentences[i - 1], sentences[i]), sentences[i])) c -= turn_bonus;
            if (c < cost[m * (k + 1) + j] or (c == cost[m * (k + 1) + j] and i < from[m * (k + 1) + j])) {
                cost[m * (k + 1) + j] = c;
                from[m * (k + 1) + j] = i;
            }
        }
    };
    const cuts = try a.alloc(usize, n + 1);
    cuts[n] = k;
    var m = n;
    while (m > 0) : (m -= 1) cuts[m - 1] = from[m * (k + 1) + cuts[m]];
    for (cuts[0..n], cuts[1..]) |head, next| try out.append(a, span(sentences[head], sentences[next - 1]));
}

/// The input from the start of `head` to the end of `tail`.
fn span(head: []const u8, tail: []const u8) []const u8 {
    return head.ptr[0 .. @intFromPtr(tail.ptr) + tail.len - @intFromPtr(head.ptr)];
}

/// The input between two of its slices, `left` before `right`.
pub fn gapBetween(left: []const u8, right: []const u8) []const u8 {
    return left.ptr[left.len .. @intFromPtr(right.ptr) - @intFromPtr(left.ptr)];
}

/// Packs `text`'s pieces at `level` greedily into segments of at most `chars`; a longer piece is cut one level finer.
fn pack(a: Allocator, out: *std.ArrayList([]const u8), text: []const u8, level: Level, chars: u32) Allocator.Error!void {
    var current: ?[]const u8 = null;
    var at: usize = 0;
    while (nextPiece(text, &at, level, chars)) |piece| {
        if (count(piece) > chars) {
            if (current) |c| try out.append(a, c);
            current = null;
            try pack(a, out, piece, @fromBackingInt(@intCast(@backingInt(level) + 1)), chars);
        } else if (current) |c| {
            const joined = c.ptr[0 .. @intFromPtr(piece.ptr) + piece.len - @intFromPtr(c.ptr)];
            if (count(joined) <= chars) {
                current = joined;
            } else {
                try out.append(a, c);
                current = piece;
            }
        } else current = piece;
    }
    if (current) |c| try out.append(a, c);
}

/// Words that open a new move when they start a sentence (any case, a whole word).
const move_words = [_][]const u8{ "so", "okay", "ok", "now", "right", "anyway", "also", "next", "finally", "but", "here's", "here\u{2019}s" };
const number_words = [_][]const u8{ "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten" };

/// The text turns between `left` and `right`: a blank line, a question, or `right` opening a new move.
pub fn turns(left: []const u8, gap: []const u8, right: []const u8) bool {
    return paragraph(gap) or question(left) or opensMove(right);
}

/// `text` ends in a question mark, closing quotes and brackets aside.
pub fn question(text: []const u8) bool {
    return lastMark(text) == '?';
}

/// `text`, trailing white space, closing quotes and brackets aside, ends in `.` `!` `?` or `…`.
pub fn endsSentence(text: []const u8) bool {
    return terminal(lastMark(trim(text)) orelse return false);
}

/// The last code point of `text` before its closing quotes and brackets; null when there is none.
fn lastMark(text: []const u8) ?u21 {
    var end = text.len;
    while (end > 0) {
        var at = end - 1;
        while (at > 0 and text[at] & 0xc0 == 0x80) at -= 1;
        const cp = std.unicode.utf8Decode(text[at..end]) catch return null;
        if (!closer(cp)) return cp;
        end = at;
    }
    return null;
}

/// `text`, past a leading speaker tag and opening quotes, starts with a move word or a number and ``thing(s)``.
pub fn opensMove(text: []const u8) bool {
    var rest = drama_prompt.trimStart(text);
    if (std.mem.startsWith(u8, rest, "<|speaker:")) if (std.mem.indexOf(u8, rest, "|>")) |end| {
        rest = drama_prompt.trimStart(rest[end + 2 ..]);
    };
    while (for ([_][]const u8{ "\"", "'", "(", "\u{201c}", "\u{2018}" }) |q| {
        if (std.mem.startsWith(u8, rest, q)) break q.len;
    } else null) |skip| rest = rest[skip..];
    for (move_words) |w| if (wordAt(rest, w)) return true;
    const number = for (number_words) |w| {
        if (wordAt(rest, w)) break w.len;
    } else digits(rest);
    if (number == 0) return false;
    const after = drama_prompt.trimStart(rest[number..]);
    if (after.len == rest.len - number) return false;
    return wordAt(after, "thing") or wordAt(after, "things");
}

/// `text` starts with `word` (ASCII case-insensitive) followed by something other than a letter.
fn wordAt(text: []const u8, word: []const u8) bool {
    if (text.len < word.len or !std.ascii.eqlIgnoreCase(text[0..word.len], word)) return false;
    return text.len == word.len or !std.ascii.isAlphabetic(text[word.len]);
}

fn digits(text: []const u8) usize {
    var n: usize = 0;
    while (n < text.len and std.ascii.isDigit(text[n])) n += 1;
    return if (n < text.len and std.ascii.isAlphabetic(text[n])) 0 else n;
}

/// A blank line: two line breaks, or a paragraph separator.
pub fn paragraph(gap: []const u8) bool {
    return std.mem.count(u8, gap, "\n") >= 2 or std.mem.indexOf(u8, gap, "\u{2029}") != null;
}

fn count(text: []const u8) usize {
    return std.unicode.utf8CountCodepoints(text) catch unreachable;
}

/// Ends a sentence when followed by white space or the end; closing quotes and brackets stay with it.
fn terminal(cp: u21) bool {
    return cp == '.' or cp == '!' or cp == '?' or cp == 0x2026;
}

fn closer(cp: u21) bool {
    return switch (cp) {
        '"', '\'', ')', ']', 0x201d, 0x2019, 0xbb => true,
        else => false,
    };
}

fn newline(cp: u21) bool {
    return cp == '\n' or cp == '\r' or cp == 0x2028 or cp == 0x2029;
}

/// The next non-empty piece of `text` from `at.*`, trimmed of white space, with `at.*` moved past it.
fn nextPiece(text: []const u8, at: *usize, level: Level, chars: u32) ?[]const u8 {
    while (at.* < text.len) {
        const begin = at.*;
        const end = cut(text, begin, level, chars);
        at.* = end;
        const piece = trim(text[begin..end]);
        if (piece.len > 0) return piece;
    }
    return null;
}

/// Where the piece starting at `begin` ends (its cut point, or the end of `text`).
fn cut(text: []const u8, begin: usize, level: Level, chars: u32) usize {
    var it = std.unicode.Utf8View.initUnchecked(text[begin..]).iterator();
    var seen: usize = 0;
    while (it.nextCodepoint()) |cp| {
        seen += 1;
        switch (level) {
            .sentence => {
                if (newline(cp)) return begin + it.i;
                if (!terminal(cp)) continue;
                var after = it;
                while (after.peek(1).len > 0) {
                    const next = std.unicode.utf8Decode(after.peek(1)) catch unreachable;
                    if (!terminal(next) and !closer(next)) break;
                    _ = after.nextCodepoint();
                }
                it = after;
                if (endsWord(text[begin..], it.i)) return begin + it.i;
            },
            .clause => if ((cp == ';' or cp == ':' or cp == ',') and endsWord(text[begin..], it.i)) return begin + it.i,
            .space => if (drama_prompt.isWhiteSpace(cp)) return begin + it.i,
            .any => if (seen == chars) return begin + it.i,
        }
    }
    return text.len;
}

/// True at the end of `text` or before white space.
fn endsWord(text: []const u8, i: usize) bool {
    if (i == text.len) return true;
    var it = std.unicode.Utf8View.initUnchecked(text[i..]).iterator();
    return drama_prompt.isWhiteSpace(it.nextCodepoint().?);
}

/// `text` without leading or trailing Unicode white space.
fn trim(text: []const u8) []const u8 {
    const body = drama_prompt.trimStart(text);
    var end: usize = 0;
    var it = std.unicode.Utf8View.initUnchecked(body).iterator();
    while (it.nextCodepoint()) |cp| if (!drama_prompt.isWhiteSpace(cp)) {
        end = it.i;
    };
    return body[0..end];
}

/// A segment's references: the request's and the voice's (`base`), then the last `keep` segments heard, latest last.
pub fn carried(a: Allocator, base: []const Reference, heard: []const Reference, keep: usize) Allocator.Error![]const Reference {
    return std.mem.concat(a, Reference, &.{ base, heard[heard.len - keep ..] });
}

/// Bytes the X-Long-Form-Segments value takes at most: the count, then each later start as 20 digits, commas between.
pub const segments_header_max = std.fmt.comptimePrint("{d};", .{max_segments}).len + (max_segments - 1) * 20 + (max_segments - 2);

/// The X-Long-Form-Segments value in `buf`: the segments spoken, then where each after the first starts, in samples.
pub fn segmentsHeader(buf: *[segments_header_max]u8, starts: []const usize) []const u8 {
    std.debug.assert(starts.len >= 1 and starts.len <= max_segments);
    var w: std.Io.Writer = .fixed(buf);
    w.print("{d};", .{starts.len}) catch unreachable; // the buffer holds the largest value
    for (starts[1..], 0..) |start, i| w.print("{s}{d}", .{ if (i == 0) "" else ",", start }) catch unreachable;
    return w.buffered();
}

fn expectSplit(input: []const u8, chars: u32, want: []const []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const got = try split(arena.allocator(), input, chars);
    std.testing.expectEqual(want.len, got.len) catch |e| {
        for (got) |s| std.debug.print("[{s}]\n", .{s});
        return e;
    };
    for (want, got) |w, g| try std.testing.expectEqualStrings(w, g);
}

test "sentences end at . ! ? \u{2026} and line breaks, keeping their punctuation" {
    try expectSplit("One. Two! Three? Four\u{2026} Five.", 10, &.{ "One. Two!", "Three?", "Four\u{2026}", "Five." });
    try expectSplit("One. Two! Three?", 16, &.{"One. Two! Three?"});
    try expectSplit("  First line\nSecond line.\n\n  Third  ", 12, &.{ "First line", "Second line.", "Third" });
    // a full stop inside a number, or before a closing quote, does not end the sentence there
    try expectSplit("Pi is 3.14 today. \"Stop.\" Then go.", 18, &.{ "Pi is 3.14 today.", "\"Stop.\" Then go." });
    try expectSplit("Wait... what?! Yes.", 7, &.{ "Wait...", "what?!", "Yes." });
    try expectSplit("ab", 1, &.{ "a", "b" });
    try expectSplit(" \n\t ", 5, &.{});
}

/// `n` sentences of 50 characters, sentence `opens` starting with `opener` and sentence `asks` ending in a question.
fn sample(a: Allocator, n: usize, opens: ?usize, opener: []const u8, asks: ?usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (0..n) |i| {
        if (i > 0) try out.append(a, ' ');
        const head = if (opens == i) opener else "";
        try out.appendSlice(a, head);
        try out.appendNTimes(a, 'x', 49 - count(head));
        try out.append(a, if (asks == i) '?' else '.');
    }
    return out.items;
}

fn lengths(a: Allocator, input: []const u8, chars: u32) ![]usize {
    const segments = try split(a, input, chars);
    const out = try a.alloc(usize, segments.len);
    for (segments, out) |seg, *n| n.* = count(seg);
    return out;
}

test "the fewest segments that fit, balanced: 713 characters under 500 are two of 356, not 458 and 254" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualSlices(usize, &.{ 356, 356 }, try lengths(a, try sample(a, 14, null, "", null), 500));
    try std.testing.expectEqualSlices(usize, &.{ 203, 254, 254 }, try lengths(a, try sample(a, 14, null, "", null), 300));
}

test "a cut where the text turns wins within the balance tolerance, not beyond it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // one sentence (51 characters) off balance: the turn wins; two sentences off: balance wins
    for ([_][]const u8{ "So, ", "Okay ", "Now ", "But ", "Here's ", "Here\u{2019}s ", "Two things: ", "3 things. ", "Finally ", "\"Anyway " }) |opener| {
        std.testing.expectEqualSlices(usize, &.{ 305, 407 }, try lengths(a, try sample(a, 14, 6, opener, null), 500)) catch |e| {
            std.debug.print("{s}\n", .{opener});
            return e;
        };
        try std.testing.expectEqualSlices(usize, &.{ 356, 356 }, try lengths(a, try sample(a, 14, 4, opener, null), 500));
    }
    try std.testing.expectEqualSlices(usize, &.{ 305, 407 }, try lengths(a, try sample(a, 14, null, "", 5), 500));
    for ([_][]const u8{ "Sooner ", "Butter ", "Nowhere ", "Ones thing ", "Twothings " }) |not_a_turn|
        try std.testing.expectEqualSlices(usize, &.{ 356, 356 }, try lengths(a, try sample(a, 14, 6, not_a_turn, null), 500));
}

test "the text turns at a blank line, a question, or a new move" {
    try std.testing.expect(turns("Done.", "\n\n", "and more."));
    try std.testing.expect(turns("Done.", "\r\n \r\n", "and more."));
    try std.testing.expect(turns("Is it \u{201c}so?\u{201d}", " ", "and more."));
    try std.testing.expect(turns("Done.", " ", "<|speaker:0|> So we go."));
    try std.testing.expect(turns("Done.", " ", "One thing matters."));
    try std.testing.expect(!turns("Done.", "\n", "and more."));
    try std.testing.expect(!turns("Done?!", " ", "and more."));
}

test "no segment is ever longer than the limit, and there are never more than needed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: std.ArrayList(u8) = .empty;
    var seed: u32 = 7;
    for (0..60) |i| {
        seed = seed *% 1103515245 +% 12345;
        const len = 20 + (seed >> 16) % 70;
        if (i > 0) try text.append(a, ' ');
        if (i % 9 == 4) try text.appendSlice(a, "So ");
        try text.appendNTimes(a, 'y', len);
        try text.append(a, if (i % 7 == 3) '?' else '.');
    }
    for ([_]u32{ 95, 120, 160, 230, 500, 1000 }) |chars| {
        const got = try split(a, text.items, chars);
        for (got) |seg| try std.testing.expect(count(seg) <= chars);
        // the greedy packing's count is the fewest possible
        var greedy: usize = 0;
        var current: usize = 0;
        var at: usize = 0;
        while (nextPiece(text.items, &at, .sentence, chars)) |piece| {
            const n = count(piece);
            if (current > 0 and current + 1 + n <= chars) current += 1 + n else {
                greedy += 1;
                current = n;
            }
        }
        try std.testing.expectEqual(greedy, got.len);
    }
}

test "a sentence longer than the limit is cut at clause marks, then spaces, then anywhere" {
    try expectSplit("alpha, beta; gamma: delta.", 13, &.{ "alpha, beta;", "gamma: delta." });
    // a colon inside a speaker tag is not a clause mark
    try expectSplit("<|speaker:0|> one two, three", 14, &.{ "<|speaker:0|>", "one two,", "three" });
    try expectSplit("aaaa bbbb cccc dddd.", 9, &.{ "aaaa bbbb", "cccc", "dddd." });
    try expectSplit("abcdefghij k", 4, &.{ "abcd", "efgh", "ij", "k" });
    try expectSplit("\u{3042}\u{3044}\u{3046}\u{3048}\u{304a}", 2, &.{ "\u{3042}\u{3044}", "\u{3046}\u{3048}", "\u{304a}" });
}

test "the segments header lists where each segment after the first starts, and its largest value fits its buffer" {
    var buf: [segments_header_max]u8 = undefined;
    try std.testing.expectEqualStrings("3;2048,6144", segmentsHeader(&buf, &.{ 0, 2048, 6144 }));
    try std.testing.expectEqualStrings("1;", segmentsHeader(&buf, &.{0}));
    var starts: [max_segments]usize = @splat(std.math.maxInt(usize));
    starts[0] = 0;
    try std.testing.expectEqual(segments_header_max, segmentsHeader(&buf, &starts).len);
}
