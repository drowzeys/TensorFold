//! Drama 3 voice references against the pinned reference (drama_voice_fixture.json, tools/zig/drama_voice_fixture.py).
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const hf_text = @import("hf_text.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_voice = @import("drama_voice.zig");
const drama_long = @import("drama_long.zig");
const drama_cfg = @import("drama_cfg.zig");
const harness = @import("speech_stream_test.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Reference = drama_prompt.Reference;

const fixture = @embedFile("drama_voice_fixture.json");

test "references are refused with the reference's messages, in its order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, fixture)).ok;
    const cases = doc.get("refusals").?.array;
    try std.testing.expect(cases.len >= 20);
    for (cases) |case| {
        var cx: Cx = .{ .a = a };
        const got = drama_voice.lower(&cx, case.strField("input").?, case.get("references").?, &.{});
        std.testing.expectError(error.Refused, got) catch |e| {
            std.debug.print("{s}: accepted\n", .{case.strField("name").?});
            return e;
        };
        std.testing.expectEqualStrings(case.strField("message").?, cx.message) catch |e| {
            std.debug.print("{s}\n", .{case.strField("name").?});
            return e;
        };
    }
}

test "the server's own refusals: audio it cannot encode, fields it would drop, text that is not UTF-8" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "[{\"text\": \"x\", \"audio_path\": \"/a.wav\"}]", "Drama 3 request: references.0: reference audio is not encoded by this server; send its vq_codes (tools/zig/drama_voice.py)" },
        .{ "[{\"text\": \"x\", \"vq_codes\": [[1]], \"speed\": 2}]", "unknown speech reference field: references.0.speed" },
        .{ "[{\"text\": \"\\ud800\", \"vq_codes\": [[1]]}]", "Drama 3 request: references.0.text must be valid UTF-8 text" },
    };
    for (cases) |case| {
        var cx: Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, drama_voice.lower(&cx, "hi", (try json.parse(a, case[0])).ok, &.{}));
        try std.testing.expectEqualStrings(case[1], cx.message);
    }
    // with codes, the audio fields are not read, as the reference reads vq_codes first
    var cx: Cx = .{ .a = a };
    const codes = "[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]";
    const both = try drama_voice.lower(&cx, "hi", (try json.parse(a, "[{\"vq_codes\": " ++ codes ++ ", \"audio_path\": \"/a.wav\"}]")).ok, &.{});
    try std.testing.expectEqual(@as(usize, 1), both.len);
    try std.testing.expectEqualStrings("", both[0].text);
    try std.testing.expectEqual(@as(u32, 10), both[0].frames[0][9]);
}

test "a request's references come before its voice's; no references and no voice is a plain prompt" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: Cx = .{ .a = a };
    const frames = [_]drama_prompt.Codes{@splat(1)};
    const voice = [_]Reference{.{ .text = "voice", .frames = &frames }};
    const sent = (try json.parse(a, "[{\"text\": \"sent\", \"vq_codes\": [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2]]}]")).ok;
    const both = try drama_voice.lower(&cx, "hi", sent, &voice);
    try std.testing.expectEqual(@as(usize, 2), both.len);
    try std.testing.expectEqualStrings("sent", both[0].text);
    try std.testing.expectEqualStrings("voice", both[1].text);
    try std.testing.expectEqual(@as(usize, 1), (try drama_voice.lower(&cx, "hi", .null, &voice)).len);
    for ([_][]const u8{ "null", "[]" }) |none| try std.testing.expectEqual(@as(usize, 0), (try drama_voice.lower(&cx, "<|voice|>", (try json.parse(a, none)).ok, &.{})).len);
    // the voice's transcript counts in the other-speaker check, as an uploaded voice's does
    const other = [_]Reference{.{ .text = "<|speaker:4|> voice", .frames = &frames }};
    try std.testing.expectError(error.Refused, drama_voice.lower(&cx, "hi", .null, &other));
    try std.testing.expectEqualStrings("Drama 3 request: only speaker 0 is supported; got speaker tags [4]", cx.message);
}

test "transcripts move a tag behind leading white space in front of it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "hi", "<|speaker:0|> hi" },
        .{ "", "<|speaker:0|> " },
        .{ "  <|speaker:0|> hi", "<|speaker:0|>   hi" },
        .{ "\u{3000}\n<|speaker:0|>\tx", "<|speaker:0|>\u{3000}\n\tx" },
        .{ "a <|speaker:0|> b", "<|speaker:0|> a <|speaker:0|> b" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], try drama_prompt.transcript(a, c[0]));
}

/// The rows as the fixture stores them: [token, code 0, ..., code 9] each.
fn rowsOf(a: std.mem.Allocator, rows: drama_prompt.Rows) ![]const [11]i64 {
    const out = try a.alloc([11]i64, rows.tokens.len);
    for (out, 0..) |*row, i| {
        row[0] = rows.tokens[i];
        for (row[1..], 0..) |*v, c| v.* = if (rows.codes.len == 0) 0 else rows.codes[i][c];
    }
    return out;
}

/// The real tokenizer at TF_DRAMA3_DIR, checked against the one the fixture was captured with; null: skipped.
fn realText(a: std.mem.Allocator, doc: Value) !?*hf_text.HfText {
    const dir = std.testing.environ.getPosix("TF_DRAMA3_DIR") orelse return null;
    const tokenizer = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fs.path.join(a, &.{ dir, "tokenizer.json" }), a, .limited(256 << 20));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(tokenizer, &digest, .{});
    try std.testing.expectEqualStrings(doc.strField("tokenizer_sha256").?, &std.fmt.bytesToHex(digest, .lower));
    var problem: []const u8 = "";
    return try hf_text.HfText.load(std.testing.allocator, std.testing.io, dir, null, a, &problem);
}

/// `rows` equal the fixture case's, by its digest and row by row.
fn expectRows(a: std.mem.Allocator, case: Value, rows: drama_prompt.Rows) !void {
    const got = try rowsOf(a, rows);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (got) |row| for (row) |v| h.update(&std.mem.toBytes(std.mem.nativeToLittle(i64, v)));
    const want = case.get("rows").?.array;
    std.testing.expectEqualStrings(case.strField("rows_sha256").?, &std.fmt.bytesToHex(h.finalResult(), .lower)) catch |e| {
        std.debug.print("{s}: {d} rows, the reference's {d}\n", .{ case.strField("name").?, got.len, want.len });
        return e;
    };
    for (want, got) |row, mine| for (row.array, mine) |v, m| try std.testing.expectEqual(v.int64().?, m);
}

test "prompt rows equal the reference's build_prompt_rows with the real tokenizer (TF_DRAMA3_DIR)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, fixture)).ok;
    const text = try realText(a, doc) orelse return error.SkipZigTest;
    defer text.deinit();
    for (doc.get("rows").?.array) |case| {
        var cx: Cx = .{ .a = a };
        const input = case.strField("input").?;
        const references = try drama_voice.lower(&cx, input, case.get("references").?, &.{});
        try expectRows(a, case, if (references.len == 0) .{ .tokens = try drama_prompt.encode(a, text.text(), input), .codes = &.{} } else try drama_prompt.encodeReferenced(a, text.text(), input, references));
    }
}

test "long-form segments' rows, the voice then earlier segments latest last, equal the reference's (TF_DRAMA3_DIR)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, fixture)).ok;
    const text = try realText(a, doc) orelse return error.SkipZigTest;
    defer text.deinit();
    const cases = doc.get("carry").?.array;
    try std.testing.expectEqual(@as(usize, 3), cases.len);
    for (cases) |case| {
        var cx: Cx = .{ .a = a };
        const input = case.strField("input").?;
        var voice_file = [_]Value{case.get("voice").?};
        const voice = if (voice_file[0] == .null) &.{} else try drama_voice.lower(&cx, input, .{ .array = &voice_file }, &.{});
        const heard = try drama_voice.lower(&cx, input, case.get("carry").?, &.{});
        const base = try drama_voice.lower(&cx, input, .null, voice);
        try expectRows(a, case, try drama_prompt.encodeReferenced(a, text.text(), input, try drama_long.carried(a, base, heard, heard.len)));
    }
}

test "a guided draw's unconditioned prompt is the reference's rows for its input without references (TF_DRAMA3_DIR)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, fixture)).ok;
    const text = try realText(a, doc) orelse return error.SkipZigTest;
    defer text.deinit();
    var guided: usize = 0;
    for (doc.get("rows").?.array) |case| {
        var cx: Cx = .{ .a = a };
        const input = case.strField("input").?;
        const references = try drama_voice.lower(&cx, input, case.get("references").?, &.{});
        const rows: drama_prompt.Rows = if (references.len == 0) .{ .tokens = try drama_prompt.encode(a, text.text(), input), .codes = &.{} } else try drama_prompt.encodeReferenced(a, text.text(), input, references);
        const g = try drama_cfg.guide(a, text.text(), .{ .gamma = 2 }, 2, input, rows);
        // a plain prompt is its own unconditioned prompt: its draws go unguided
        if (references.len == 0) {
            try std.testing.expectEqual(@as(usize, 0), g.unconditioned.len);
            try expectRows(a, case.get("unconditioned").?, rows);
            continue;
        }
        try expectRows(a, case.get("unconditioned").?, .{ .tokens = g.unconditioned, .codes = &.{} });
        guided += 1;
    }
    for (doc.get("carry").?.array) |case| {
        var cx: Cx = .{ .a = a };
        const input = case.strField("input").?;
        var voice_file = [_]Value{case.get("voice").?};
        const voice = if (voice_file[0] == .null) &.{} else try drama_voice.lower(&cx, input, .{ .array = &voice_file }, &.{});
        const heard = try drama_voice.lower(&cx, input, case.get("carry").?, &.{});
        const base = try drama_voice.lower(&cx, input, .null, voice);
        const rows = try drama_prompt.encodeReferenced(a, text.text(), input, try drama_long.carried(a, base, heard, heard.len));
        try expectRows(a, case.get("unconditioned").?, .{ .tokens = (try drama_cfg.guide(a, text.text(), .{ .gamma = 2 }, 2, input, rows)).unconditioned, .codes = &.{} });
        guided += 1;
    }
    try std.testing.expectEqual(@as(usize, 11), guided);
}

const owner = "{\"kind\": \"owner\", \"note\": \"my own voice\"}";
const unsourced = "source must say whose voice this is: {\"kind\": \"model-invented\" | \"owner\" | \"consented\", \"note\": \"...\"} (docs/drama-voices.md)";

/// A voice file with `text`, `source` (JSON, or empty for none) and a [10, 2] grid, its digest right or wrong.
fn voiceFile(a: std.mem.Allocator, text: []const u8, source: []const u8, right_digest: bool) ![]const u8 {
    const frames = [_]drama_prompt.Codes{ .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .{ 4095, 11, 12, 13, 14, 15, 16, 17, 18, 19 } };
    var digest = drama_voice.codesDigest(.{ .text = text, .frames = &frames });
    if (!right_digest) digest[0] +%= 1;
    var grid: std.ArrayList(u8) = .empty;
    for (0..10) |book| try grid.print(a, "{s}[{d}, {d}]", .{ if (book == 0) "" else ", ", frames[0][book], frames[1][book] });
    const sourced = if (source.len == 0) "" else try std.fmt.allocPrint(a, "\"source\": {s}, ", .{source});
    return std.fmt.allocPrint(a, "{{\"format\": \"drama3-voice/1\", \"text\": \"{s}\", {s}\"codes_sha256\": \"{s}\", \"vq_codes\": [{s}]}}", .{ text, sourced, &std.fmt.bytesToHex(digest, .lower), grid.items });
}

test "voice files: regular NAME.json files only, names, consent and digests checked, nothing outside the directory" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var outside = std.testing.tmpDir(.{});
    defer outside.cleanup();
    try outside.dir.writeFile(io, .{ .sub_path = "secret.json", .data = try voiceFile(a, "Outside.", owner, true) });
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try tmp.dir.writeFile(io, .{ .sub_path = "me.json", .data = try voiceFile(a, "Hello there.", owner, true) });
    try tmp.dir.writeFile(io, .{ .sub_path = "alto.json", .data = try voiceFile(a, "Invented.", "{\"kind\": \"model-invented\", \"note\": \"a model's own voice\"}", true) });
    try tmp.dir.writeFile(io, .{ .sub_path = "vera.json", .data = try voiceFile(a, "Agreed.", "{\"kind\": \"consented\", \"note\": \" signed release, consent/vera.pdf \"}", true) });
    try tmp.dir.writeFile(io, .{ .sub_path = ".hidden.json", .data = "not read" });
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "not read" });
    try tmp.dir.createDir(io, "folder.json", .default_dir);
    try tmp.dir.symLink(io, try std.fmt.allocPrint(a, "../{s}/secret.json", .{outside.sub_path}), "link.json", .{});
    var problem: []const u8 = "";
    const voices = (try drama_voice.load(a, io, path, &problem)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 3), voices.len);
    try std.testing.expectEqualStrings("alto", voices[0].name);
    try std.testing.expectEqual(drama_voice.Source.@"model-invented", voices[0].source);
    try std.testing.expectEqualStrings("me", voices[1].name);
    try std.testing.expectEqual(drama_voice.Source.owner, voices[1].source);
    try std.testing.expectEqual(@as(usize, 1), voices[1].references.len);
    try std.testing.expectEqualStrings("Hello there.", voices[1].references[0].text);
    try std.testing.expectEqual(@as(u32, 4095), voices[1].references[0].frames[1][0]);
    try std.testing.expectEqual(drama_voice.Source.consented, voices[2].source);
    try std.testing.expect(drama_voice.find(voices, "me") != null and drama_voice.find(voices, "Me") == null and drama_voice.find(voices, "../me") == null);
    const refused = [_]struct { file: []const u8, data: []const u8, problem: []const u8 }{
        .{ .file = "default.json", .data = try voiceFile(a, "x", owner, true), .problem = "a voice name is" },
        .{ .file = "-x.json", .data = try voiceFile(a, "x", owner, true), .problem = "a voice name is" },
        .{ .file = "bad.json", .data = try voiceFile(a, "x", owner, false), .problem = "codes_sha256 does not match vq_codes" },
        .{ .file = "tagged.json", .data = try voiceFile(a, "<|speaker:3|> x", owner, true), .problem = "only speaker 0 is supported" },
        .{ .file = "plain.json", .data = "nope", .problem = "not JSON" },
        .{ .file = "unsourced.json", .data = try voiceFile(a, "x", "", true), .problem = unsourced },
        .{ .file = "flat.json", .data = try voiceFile(a, "x", "\"owner\"", true), .problem = unsourced },
        .{ .file = "famous.json", .data = try voiceFile(a, "x", "{\"kind\": \"celebrity\", \"note\": \"from a film\"}", true), .problem = unsourced },
        .{ .file = "unnoted.json", .data = try voiceFile(a, "x", "{\"kind\": \"consented\", \"note\": \"\"}", true), .problem = unsourced },
        .{ .file = "blank.json", .data = try voiceFile(a, "x", "{\"kind\": \"consented\", \"note\": \" \\t\\n\"}", true), .problem = unsourced },
        .{ .file = "noteless.json", .data = try voiceFile(a, "x", "{\"kind\": \"owner\"}", true), .problem = unsourced },
        .{ .file = "numbered.json", .data = try voiceFile(a, "x", "{\"kind\": \"owner\", \"note\": 3}", true), .problem = unsourced },
    };
    for (refused) |case| {
        var each = std.testing.tmpDir(.{});
        defer each.cleanup();
        try each.dir.writeFile(io, .{ .sub_path = case.file, .data = case.data });
        try std.testing.expect(try drama_voice.load(a, io, try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{each.sub_path}), &problem) == null);
        try std.testing.expect(std.mem.indexOf(u8, problem, case.problem) != null);
    }
    try std.testing.expect(try drama_voice.load(a, io, ".zig-cache/tmp/no-such-voices", &problem) == null);
    const long: [65]u8 = @splat('x');
    for ([_][]const u8{ "", "..", "../x", "a/b", "default", "DEFAULT", ".x", "_x", &long, "a b" }) |name| try std.testing.expect(!drama_voice.validName(name));
    for ([_][]const u8{ "lighthouse-2", "Ada.v1_2", long[1..] }) |name| try std.testing.expect(drama_voice.validName(name));
}

test "GET /v1/voices lists the loaded voices with their references' length and source kind, OpenAI-list style" {
    const frames: [431]drama_prompt.Codes = @splat(@splat(7));
    const two = [_]Reference{ .{ .text = "a", .frames = frames[0..2] }, .{ .text = "c", .frames = frames[0..429] } };
    const voices = [_]drama_voice.Voice{
        .{ .name = "alto", .references = two[0..1], .source = .@"model-invented" },
        .{ .name = "duet", .references = &two, .source = .owner },
    };
    var script: harness.Script = .{ .steps = &.{} };
    var s: harness.Served = undefined;
    try s.open(script.engine(), .{ .served_name = "m", .model_ids = &.{"m"}, .voices = &voices }, "GET /v1/voices HTTP/1.1", "", 2000);
    defer s.deinit();
    try std.testing.expect(std.mem.startsWith(u8, try s.readHead(), "HTTP/1.1 200"));
    s.route.join();
    try std.testing.expectEqualStrings("{\"object\": \"list\", \"data\": [" ++
        "{\"object\": \"voice\", \"name\": \"alto\", \"references\": 1, \"frames\": 2, \"seconds\": 0.09, \"source\": {\"kind\": \"model-invented\"}}, " ++
        "{\"object\": \"voice\", \"name\": \"duet\", \"references\": 2, \"frames\": 431, \"seconds\": 20.02, \"source\": {\"kind\": \"owner\"}}]}", try s.readRest());
}
