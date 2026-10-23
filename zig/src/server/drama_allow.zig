//! --allowed-controls: delivery controls off the list are taken out of an input before anything reads it.
const std = @import("std");
const speech_redraw = @import("speech_redraw.zig");
const unicode = @import("tokenizer").unicode;
const case = @import("template").unicode;
const Allocator = std.mem.Allocator;

/// The longest ``[...]`` read as a control; a longer one is text, and no scan for its ']' goes further.
const max_control = 120;

const Section = enum { cues, modifiers };

/// drama_cues.txt: every word of a control is one of these, and at least one is a cue.
const vocabulary = vocabulary: {
    @setEvalBranchQuota(200_000);
    const Entry = struct { []const u8, Section };
    var entries: []const Entry = &.{};
    var section: ?Section = null;
    var lines = std.mem.tokenizeScalar(u8, @embedFile("drama_cues.txt"), '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "## ")) {
            section = std.meta.stringToEnum(Section, line[3..]) orelse @compileError("unknown section " ++ line);
        } else if (line[0] != '#') entries = entries ++ &[_]Entry{.{ line, section orelse @compileError("a word before any section") }};
    }
    break :vocabulary std.StaticStringMap(Section).initComptime(entries);
};

/// An input with the controls off the list taken out, and how many were.
pub const Filtered = struct { text: []const u8, stripped: usize };

/// `input` without its scoped tags (text kept) and its ``[...]`` controls off `allowed`; listed ones normalized.
pub fn filter(a: Allocator, input: []const u8, allowed: []const []const u8) Allocator.Error!Filtered {
    var out: std.ArrayList(u8) = .empty;
    var done: usize = 0; // input[0..done] is in `out` or replaced there
    var stripped: usize = 0;
    var stacked: ?usize = null; // where the last candidate ended: a '[' right there stacks on it, not an index
    var text_until: usize = 0; // inside a bracket kept as text only scoped tags go: no '[' there opens a control
    var closes: ?[]const usize = null; // each '[''s matching ']', made at the first bracket kept as text
    var i: usize = 0;
    while (i < input.len) {
        const tag = tagAt(input[i..]);
        if (tag > 0) {
            try out.appendSlice(a, input[done..i]);
            done = drop(&out, input, i + tag);
            stripped += 1;
            i = done;
            continue;
        }
        const end = (if (i < text_until) null else candidateAt(input, i, stacked)) orelse {
            i += 1;
            continue;
        };
        stacked = end;
        const inner = input[i + 1 .. end - 1];
        const bare = try untagged(a, inner);
        const name = try normalize(a, bare.text);
        const listed = for (allowed) |kept| {
            if (std.mem.eql(u8, name, kept)) break true;
        } else false;
        if (listed and !std.mem.eql(u8, name, inner)) {
            try out.appendSlice(a, input[done..i]);
            try out.print(a, "[{s}]", .{name});
            done = end;
            stripped += bare.tags;
            i = done;
        } else if (!listed and isControl(name)) {
            try out.appendSlice(a, input[done..i]);
            done = drop(&out, input, end);
            stripped += 1 + bare.tags;
            i = done;
        } else {
            const close = (closes orelse made: {
                closes = try matching(a, input);
                break :made closes.?;
            })[i];
            // to its matching ']', or, unbalanced, its first: a stray '[' must not shield the rest of the input
            text_until = if (close < input.len) close + 1 else end;
            stacked = text_until;
            i += 1;
        }
    }
    if (done == 0) return .{ .text = input, .stripped = 0 };
    try out.appendSlice(a, input[done..]);
    return .{ .text = out.items, .stripped = stripped };
}

/// Where each '[' of `input` closes, by nesting depth; `input.len` for one that never does and for other bytes.
fn matching(a: Allocator, input: []const u8) Allocator.Error![]const usize {
    const closes = try a.alloc(usize, input.len);
    @memset(closes, input.len);
    var open: std.ArrayList(usize) = .empty;
    for (input, 0..) |c, k| switch (c) {
        '[' => try open.append(a, k),
        ']' => if (open.pop()) |start| {
            closes[start] = k;
        },
        else => {},
    };
    return closes;
}

/// `text` as controls and names compare: NFC, lower case, white space runs one space, trimmed, no trailing ``!?.…``.
pub fn normalize(a: Allocator, text: []const u8) Allocator.Error![]const u8 {
    const lower = try case.lower(a, try unicode.normalize(a, text, .nfc));
    var out: std.ArrayList(u8) = .empty;
    var space = false;
    var it = std.unicode.Utf8View.initUnchecked(lower).iterator();
    while (it.nextCodepointSlice()) |bytes| {
        if (unicode.isSpace(std.unicode.utf8Decode(bytes) catch unreachable)) {
            space = out.items.len > 0;
            continue;
        }
        if (space) try out.append(a, ' ');
        space = false;
        try out.appendSlice(a, bytes);
    }
    var end = out.items.len;
    while (end > 0) {
        if (std.mem.indexOfScalar(u8, "!?. ", out.items[end - 1]) != null) {
            end -= 1;
        } else if (std.mem.endsWith(u8, out.items[0..end], "\u{2026}")) {
            end -= "\u{2026}".len;
        } else break;
    }
    return out.items[0..end];
}

/// Whether a normalized ``[name]`` is a control: its words, split at `` ,-``, all in drama_cues.txt, one of them a cue.
pub fn isControl(name: []const u8) bool {
    var cue = false;
    var words = std.mem.tokenizeAny(u8, name, " ,-");
    while (words.next()) |word| switch (vocabulary.get(word) orelse return false) {
        .cues => cue = true,
        .modifiers => {},
    };
    return cue;
}

/// The end of the ``[...]`` opening at `i` that may be a control: not an index (``x[i]``), a link label or quoted code.
fn candidateAt(input: []const u8, i: usize, stacked: ?usize) ?usize {
    if (input[i] != '[') return null;
    if (i > 0 and stacked != i) {
        var start = i - 1;
        while (start > 0 and input[start] & 0xc0 == 0x80) start -= 1;
        const before = std.unicode.utf8Decode(input[start..i]) catch unreachable;
        if (before == ']' or before == '_' or before == '`') return null;
        if (unicode.bit(unicode.category(before)) & (unicode.letter | unicode.number) != 0) return null;
    }
    const close = std.mem.indexOfScalar(u8, input[i..@min(input.len, i + max_control)], ']') orelse return null;
    const end = i + close + 1;
    if (end < input.len and input[end] == '(') return null;
    return end;
}

/// The bytes of the scoped tag `rest` opens with, its name in any case (``<Whisper>``, ``</SOFT>``), 0 when none.
fn tagAt(rest: []const u8) usize {
    if (rest[0] != '<') return 0;
    const open: usize = if (rest.len > 1 and rest[1] == '/') 2 else 1;
    for (speech_redraw.scoped_tags) |tag| {
        const close = open + tag.len;
        if (rest.len > close and rest[close] == '>' and std.ascii.eqlIgnoreCase(rest[open..close], tag)) return close + 1;
    }
    return 0;
}

/// `inner` without its scoped tags, and how many there were: a control's name never holds them.
fn untagged(a: Allocator, inner: []const u8) Allocator.Error!struct { text: []const u8, tags: usize } {
    if (std.mem.indexOfScalar(u8, inner, '<') == null) return .{ .text = inner, .tags = 0 };
    var out: std.ArrayList(u8) = .empty;
    var tags: usize = 0;
    var i: usize = 0;
    while (i < inner.len) {
        const n = tagAt(inner[i..]);
        if (n > 0) tags += 1 else try out.append(a, inner[i]);
        i += @max(n, 1);
    }
    return .{ .text = out.items, .tags = tags };
}

/// Where the copy resumes after the control before `end`: of the white space around it, the run with more line breaks.
fn drop(out: *std.ArrayList(u8), input: []const u8, end: usize) usize {
    var after = end;
    while (after < input.len and std.ascii.isWhitespace(input[after])) after += 1;
    var before = out.items.len;
    while (before > 0 and std.ascii.isWhitespace(out.items[before - 1])) before -= 1;
    if (before == 0) {
        out.clearRetainingCapacity(); // nothing but white space before it: the input now starts after the control
        return after;
    }
    const lines_before = std.mem.count(u8, out.items[before..], "\n");
    if (std.mem.count(u8, input[end..after], "\n") > lines_before) {
        out.shrinkRetainingCapacity(before);
        return end;
    }
    return if (before < out.items.len) after else end;
}

test "controls off the list go whatever their case, spacing or end punctuation; listed ones stay, normalized" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const allowed: []const []const u8 = &.{ "warmly", "small laugh" };
    const cases = [_]struct { []const u8, []const u8, usize }{
        .{ "[Pleased!] It works.", "It works.", 1 },
        .{ "It [PLEASED] works.", "It works.", 1 },
        .{ "It [ softly,\twhispering ] works.", "It works.", 1 },
        .{ "[relieved, with a little smile] Done\u{2026} [takes a breath] [long-break] next.", "Done\u{2026} next.", 3 },
        .{ "[Warmly] Hi [ small   LAUGH! ] there.", "[warmly] Hi [small laugh] there.", 0 },
        // stacked controls go one by one; a modifier alone is no control
        .{ "[sad][whispering] I miss you [a] lot.", "I miss you [a] lot.", 2 },
        .{ "<WHISPER>Go on</Whisper> <soft>now</soft>.", "Go on now.", 4 },
    };
    for (cases) |c| {
        const got = try filter(a, c[0], allowed);
        try std.testing.expectEqualStrings(c[1], got.text);
        try std.testing.expectEqual(c[2], got.stripped);
    }
    // an empty list takes out every control
    try std.testing.expectEqualStrings("One. Two.", (try filter(a, "[warmly] One. [sighs] Two.", &.{})).text);
}

test "brackets that are not controls stay as sent: names, notes, links, indexes, quoted syntax, long or unclosed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const too_long = comptime long: {
        var text: []const u8 = "[";
        for (0..24) |_| text = text ++ "very ";
        break :long text ++ "sad]";
    };
    const kept = [_][]const u8{
        "Meet [John] on [Tuesday].",
        "As [see note 3] says.",
        "Read [the docs](http://x) or [sighs](http://y).",
        "Use x[i], a[pause] and m[1][sighs].",
        "[TODO: fix] the build.",
        "Write `[pleased]` before a line.",
        "A [softly and unclosed",
        too_long ++ " is too long.",
        "Two  spaces [1]\n\nstay.",
    };
    for (kept) |text| {
        const got = try filter(a, text, &.{});
        try std.testing.expectEqual(@as([*]const u8, text.ptr), got.text.ptr);
        try std.testing.expectEqual(@as(usize, 0), got.stripped);
    }
}

test "scoped tags inside a bracket go too, whether the bracket stays as text, is listed or is a control" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, []const u8, usize }{
        .{ "[note <SOFT>quietly</SOFT>] stays.", "[note quietly] stays.", 2 },
        .{ "[1 <soft>x</soft>]", "[1 x]", 2 },
        .{ "[<soft>Warmly</soft>] Hi.", "[warmly] Hi.", 2 },
        .{ "[softly <soft>] Hi.", "Hi.", 2 },
        // a bracket kept as text opens no control inside it: only its tags go
        .{ "[see [sighs] <fast>now</fast>] ok", "[see [sighs] now] ok", 2 },
        // protection runs to the matching ']': a nested bracket's close does not end it, an unbalanced '[' gives none
        .{ "[see [note] [sighs] <soft>x</soft>] stays. [sighs] Go.", "[see [note] [sighs] x] stays. Go.", 3 },
        .{ "[see [note] [sighs] Go.", "[see [note] Go.", 1 },
    };
    for (cases) |c| {
        const got = try filter(a, c[0], &.{"warmly"});
        try std.testing.expectEqualStrings(c[1], got.text);
        try std.testing.expectEqual(c[2], got.stripped);
    }
}

test "a control's white space: the run with more line breaks stays, so paragraphs and marks keep their place" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "Hi [sighs] there, [pleased]\n\nnext. \u{b6} [sighs] On.", "Hi there,\n\nnext. \u{b6} On." },
        .{ "Hi\n[sighs]\nthere word [sighs].", "Hi\nthere word ." },
        .{ "quietly</fast> so", "quietly so" },
        .{ " \t [sighs] Hi", "Hi" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], (try filter(a, c[0], &.{})).text);
}

test "names compare normalized: NFC, lower case in any script, white space and end punctuation folded" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("café crème", try normalize(a, " \tCAFE\u{301}\u{a0}\u{a0}CRÈME!?\u{2026} "));
    try std.testing.expectEqualStrings("", try normalize(a, " ... "));
    // the shipped vocabulary is normalized, single words
    for (vocabulary.keys()) |word| {
        try std.testing.expectEqualStrings(word, try normalize(a, word));
        try std.testing.expect(std.mem.indexOfAny(u8, word, " ,-") == null);
    }
    try std.testing.expect(isControl("in a hurry tone") and isControl("crying loudly") and !isControl("with a little") and !isControl(""));
}
