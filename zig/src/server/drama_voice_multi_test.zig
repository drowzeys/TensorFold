//! drama3-voice/2 files: several references in one voice, built into the prompt as a request's references are.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_voice = @import("drama_voice.zig");
const harness = @import("speech_stream_test.zig");
const Reference = drama_prompt.Reference;
const Codes = drama_prompt.Codes;
const Allocator = std.mem.Allocator;

const source = "{\"kind\": \"owner\", \"note\": \"my own voice\"}";
const clip = "a40ff3d5900fb7698b8c865041347cb49eccedc8f93945f89629ad104aaecce4";

/// Three passages with codes of their own, the first reaching both ends of [0, 4096).
fn passages() [3]Reference {
    const S = struct {
        const first = [_]Codes{ .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .{ 4095, 11, 12, 13, 14, 15, 16, 17, 18, 19 } };
        const second = [_]Codes{.{ 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 }};
        const own = [_]Codes{ .{ 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 }, .{ 40, 41, 42, 43, 44, 45, 46, 47, 48, 49 }, .{ 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 } };
    };
    return .{ .{ .text = "The first passage.", .frames = &S.first }, .{ .text = "A second one, then.", .frames = &S.second }, .{ .text = "And the voice's own.", .frames = &S.own } };
}

/// The [codebooks, frames] grid as a voice file writes it.
fn grid(a: Allocator, frames: []const Codes) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, '[');
    for (0..drama_prompt.codebooks) |book| {
        try out.appendSlice(a, if (book == 0) "[" else ", [");
        for (frames, 0..) |frame, f| try out.print(a, "{s}{d}", .{ if (f == 0) "" else ", ", frame[book] });
        try out.append(a, ']');
    }
    try out.append(a, ']');
    return out.items;
}

/// One reference's fields: a /1 file's top level, or an entry of a /2 file's list.
fn fields(a: Allocator, r: Reference, audio: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "\"text\": {s}, \"codes_sha256\": \"{s}\", \"audio_sha256\": \"{s}\", \"vq_codes\": {s}", .{ try json.quote(a, r.text, .{}), &std.fmt.bytesToHex(drama_voice.codesDigest(r), .lower), audio, try grid(a, r.frames) });
}

fn single(a: Allocator, r: Reference) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"format\": \"drama3-voice/1\", \"source\": {s}, {s}}}", .{ source, try fields(a, r, clip) });
}

/// A /2 file whose entries are `entries`, already written.
fn several(a: Allocator, entries: []const []const u8) ![]const u8 {
    var list: std.ArrayList(u8) = .empty;
    for (entries, 0..) |entry, i| try list.print(a, "{s}{{{s}}}", .{ if (i == 0) "" else ", ", entry });
    return std.fmt.allocPrint(a, "{{\"format\": \"drama3-voice/2\", \"source\": {s}, \"references\": [{s}]}}", .{ source, list.items });
}

fn load(a: Allocator, bytes: []const u8) ![]const Reference {
    var problem: []const u8 = "";
    const voice = try drama_voice.parse(a, "duet", bytes, &problem) orelse {
        std.debug.print("refused: {s}\n", .{problem});
        return error.TestUnexpectedResult;
    };
    return voice.references;
}

/// The prompt rows the speech route builds for `input`: the request's ``references`` then the voice's.
fn rows(a: Allocator, input: []const u8, sent: []const u8, voice: []const Reference) !drama_prompt.Rows {
    var text: harness.Bytes = .{};
    var cx: errors.Cx = .{ .a = a };
    const references = try drama_voice.lower(&cx, input, (try json.parse(a, sent)).ok, voice);
    return drama_prompt.encodeReferenced(a, text.text(), input, references);
}

fn expectRows(want: drama_prompt.Rows, got: drama_prompt.Rows) !void {
    try std.testing.expectEqualSlices(u32, want.tokens, got.tokens);
    try std.testing.expectEqualSlices(Codes, want.codes, got.codes);
}

test "a /2 voice builds the rows a request's references followed by a /1 voice build, and one reference is unchanged" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = passages();
    const input = "The next train departs at nine.";
    const entries = [_][]const u8{ try fields(a, p[0], clip), try fields(a, p[1], clip), try fields(a, p[2], clip) };
    // as a client sends the first two passages before the voice's own
    const sent = try std.fmt.allocPrint(a, "[{{\"text\": {s}, \"vq_codes\": {s}}}, {{\"text\": {s}, \"vq_codes\": {s}}}]", .{ try json.quote(a, p[0].text, .{}), try grid(a, p[0].frames), try json.quote(a, p[1].text, .{}), try grid(a, p[1].frames) });
    const own = try load(a, try single(a, p[2]));
    const combined = try load(a, try several(a, &entries));
    try std.testing.expectEqual(@as(usize, 3), combined.len);
    const request_side = try rows(a, input, sent, own);
    try expectRows(request_side, try rows(a, input, "null", combined));
    try std.testing.expect(std.mem.indexOf(u32, request_side.tokens, &[_]u32{ '[', 'R', 'e', 'f', 'e', 'r', 'e', 'n', 'c', 'e', ' ', '3', ']' }) != null);
    // one reference: a /1 file, a /2 file and the same passage sent inline give the same rows
    const inline_own = try std.fmt.allocPrint(a, "[{{\"text\": {s}, \"vq_codes\": {s}}}]", .{ try json.quote(a, p[2].text, .{}), try grid(a, p[2].frames) });
    try expectRows(try rows(a, input, inline_own, &.{}), try rows(a, input, "null", own));
    try expectRows(try rows(a, input, "null", own), try rows(a, input, "null", try load(a, try several(a, entries[2..]))));
}

test "a /2 file is refused for an empty list, a reference without its clip's digest, wrong codes or another speaker" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = passages();
    const good = try fields(a, p[0], clip);
    const wrong = try std.mem.replaceOwned(u8, a, try fields(a, p[1], clip), &std.fmt.bytesToHex(drama_voice.codesDigest(p[1]), .lower), &std.fmt.bytesToHex(drama_voice.codesDigest(p[0]), .lower));
    const cases = [_][2][]const u8{
        .{ try several(a, &.{}), "references must be a non-empty list of {text, codes_sha256, audio_sha256, vq_codes}" },
        .{ try several(a, &.{ good, try fields(a, p[1], "") }), "references.1: audio_sha256 is not a clip's sha256 (64 lowercase hex digits)" },
        .{ try several(a, &.{ good, try fields(a, p[1], clip[1..] ++ "A") }), "references.1: audio_sha256 is not a clip's sha256 (64 lowercase hex digits)" },
        .{ try several(a, &.{ good, wrong }), "references.1: codes_sha256 does not match vq_codes" },
        .{ try several(a, &.{try fields(a, .{ .text = "<|speaker:3|> x", .frames = p[0].frames }, clip)}), "references.0: Drama 3 request: only speaker 0 is supported; got speaker tags [3]" },
        .{ try std.mem.replaceOwned(u8, a, try several(a, &.{good}), "voice/2", "voice/3"), "format is not drama3-voice/1 or drama3-voice/2" },
        .{ try std.mem.replaceOwned(u8, a, try several(a, &.{good}), source, "null"), "source must say whose voice this is" },
    };
    var problem: []const u8 = "";
    for (cases) |case| {
        try std.testing.expect(try drama_voice.parse(a, "duet", case[0], &problem) == null);
        try std.testing.expectStringStartsWith(problem, case[1]);
    }
    // voice encode's check reads single-reference files only
    try std.testing.expect(try drama_voice.parseReference(a, try several(a, &.{good}), &problem) == null);
    try std.testing.expectEqualStrings("format is not drama3-voice/1", problem);
}

test "a voice whose references leave no prompt row for an input is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = passages();
    var text: harness.Bytes = .{};
    const voices = [_]drama_voice.Voice{
        .{ .name = "solo", .references = p[0..1], .source = .owner },
        .{ .name = "duet", .references = p[0..2], .source = .owner },
    };
    const needed: u32 = @intCast((try drama_prompt.encodeReferenced(a, text.text(), "", &.{ p[0], p[1] })).tokens.len);
    try std.testing.expect(try drama_voice.overflow(a, text.text(), &voices, needed + 1) == null);
    const why = (try drama_voice.overflow(a, text.text(), &voices, needed)).?;
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "voice references too long: duet.json's 2 references take {d} prompt rows before any input, and a prompt may take {d} (--max-frames and --context set it); drop a reference or raise --context", .{ needed, needed }), why);
}
