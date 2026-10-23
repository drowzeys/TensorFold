//! ``tensorfold-native voice encode``: its arguments, and the voice file it writes once the loader accepts it.
const std = @import("std");
const json = @import("json.zig");
const drama_voice = @import("drama_voice.zig");
const drama_prompt = @import("drama_prompt.zig");
const Allocator = std.mem.Allocator;

pub const usage = "usage: tensorfold-native voice encode CLIP.wav --model MODEL --text TRANSCRIPT --name NAME --out NAME.json\n";

pub const Args = struct { clip: []const u8, model: []const u8, text: []const u8, name: []const u8, out: []const u8 };

/// The arguments after ``voice``, or null with `problem` set; --text loses outer white space, --out is NAME.json.
pub fn parse(argv: []const []const u8, problem: *[]const u8) ?Args {
    if (argv.len == 0 or !std.mem.eql(u8, argv[0], "encode")) {
        problem.* = "the voice command is encode";
        return null;
    }
    var clip: ?[]const u8 = null;
    var flags: [4]?[]const u8 = @splat(null);
    const names = [_][]const u8{ "--model", "--text", "--name", "--out" };
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (!std.mem.startsWith(u8, arg, "--")) {
            if (clip != null) {
                problem.* = "one clip at a time";
                return null;
            }
            clip = arg;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, arg, '=');
        const name = arg[0 .. eq orelse arg.len];
        const slot = for (names, 0..) |n, j| {
            if (std.mem.eql(u8, n, name)) break j;
        } else {
            problem.* = "unknown flag; the flags are --model, --text, --name and --out";
            return null;
        };
        if (eq) |e| flags[slot] = arg[e + 1 ..] else if (i + 1 < argv.len) {
            i += 1;
            flags[slot] = argv[i];
        } else {
            problem.* = "a flag without its value";
            return null;
        }
    }
    const args: Args = .{
        .clip = clip orelse return refuse(problem, "no clip"),
        .model = flags[0] orelse return refuse(problem, "no --model (the Drama 3 checkpoint holding codec.pth)"),
        .text = std.mem.trim(u8, flags[1] orelse return refuse(problem, "no --text (the clip's exact transcript)"), " \t\r\n\x0b\x0c"),
        .name = flags[2] orelse return refuse(problem, "no --name"),
        .out = flags[3] orelse return refuse(problem, "no --out"),
    };
    if (args.text.len == 0) return refuse(problem, "--text is empty");
    if (!drama_voice.validName(args.name)) return refuse(problem, "--name: a voice name is 1..64 of A-Z a-z 0-9 . _ - led by a letter or digit, not default");
    const base = std.fs.path.basename(args.out);
    if (!std.mem.endsWith(u8, base, ".json") or !std.mem.eql(u8, base[0 .. base.len - ".json".len], args.name))
        return refuse(problem, "--out must be NAME.json: --voices names a voice by its file's stem");
    return args;
}

fn refuse(problem: *[]const u8, why: []const u8) ?Args {
    problem.* = why;
    return null;
}

/// Clip `wav`'s voice file with `codes`, or null with `problem` set when ``drama_voice.parseReference`` refuses it.
pub fn voiceFile(a: Allocator, text: []const u8, codes: []const drama_prompt.Codes, wav: []const u8, encoder: json.Value, problem: *[]const u8) Allocator.Error!?[]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(wav, &digest, .{});
    const bytes = try drama_voice.file(a, .{ .text = text, .frames = codes }, try a.dupe(u8, &std.fmt.bytesToHex(digest, .lower)), encoder);
    var why: []const u8 = "";
    if (try drama_voice.parseReference(a, bytes, &why) == null) {
        problem.* = try std.fmt.allocPrint(a, "the voice file would not load: {s}", .{why});
        return null;
    }
    return bytes;
}

test "voice encode's arguments" {
    var problem: []const u8 = "";
    const ok = parse(&.{ "encode", "clip.wav", "--model", "/m", "--text=  Hello there.\n", "--name", "me", "--out", "voices/me.json" }, &problem).?;
    try std.testing.expectEqualStrings("clip.wav", ok.clip);
    try std.testing.expectEqualStrings("Hello there.", ok.text);
    const refusals = [_]struct { []const []const u8, []const u8 }{
        .{ &.{ "decode", "clip.wav" }, "the voice command is encode" },
        .{ &.{ "encode", "clip.wav", "--model", "/m", "--text", "t", "--name", "me", "--out", "voices/you.json" }, "--out must be NAME.json: --voices names a voice by its file's stem" },
        .{ &.{ "encode", "clip.wav", "--model", "/m", "--text", "t", "--name", "default", "--out", "default.json" }, "--name: a voice name is 1..64 of A-Z a-z 0-9 . _ - led by a letter or digit, not default" },
        .{ &.{ "encode", "clip.wav", "--model", "/m", "--text", " \n", "--name", "me", "--out", "me.json" }, "--text is empty" },
        .{ &.{ "encode", "clip.wav", "--model", "/m", "--name", "me", "--out", "me.json" }, "no --text (the clip's exact transcript)" },
        .{ &.{ "encode", "clip.wav", "--model", "/m", "--text", "t", "--name", "me", "--out" }, "a flag without its value" },
        .{ &.{ "encode", "a.wav", "b.wav" }, "one clip at a time" },
        .{ &.{ "encode", "a.wav", "--voices", "v" }, "unknown flag; the flags are --model, --text, --name and --out" },
    };
    for (refusals) |r| {
        try std.testing.expect(parse(r[0], &problem) == null);
        try std.testing.expectEqualStrings(r[1], problem);
    }
}

test "the voice file loads with the codes it was written with, and a transcript the loader refuses is not written" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const codes = [_]drama_prompt.Codes{ .{ 4095, 1, 2, 3, 4, 5, 6, 7, 8, 1023 }, .{ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 } };
    const encoder = try json.newObject(a);
    try encoder.put(a, "dtype", .{ .string = "bfloat16" });
    const bytes = (try voiceFile(a, "Wie spät ist es?", &codes, "RIFF", .{ .object = encoder }, &problem)).?;
    var why: []const u8 = "";
    const loaded = (try drama_voice.parseReference(a, bytes, &why)).?;
    try std.testing.expect(try drama_voice.parse(a, "me", bytes, &why) == null);
    try std.testing.expectStringStartsWith(why, "source must say whose voice this is");
    try std.testing.expectEqualStrings("Wie spät ist es?", loaded.text);
    try std.testing.expectEqualSlices(drama_prompt.Codes, &codes, loaded.frames);
    // codebook-major grid, the clip's digest, and json.dumps(ensure_ascii=False)'s bytes
    const doc = (try json.parse(a, bytes)).ok;
    try std.testing.expectEqualStrings("[4095, 0]", try json.stringify(a, doc.get("vq_codes").?.array[0], .{}));
    try std.testing.expectEqualStrings("[1023, 90]", try json.stringify(a, doc.get("vq_codes").?.array[9], .{}));
    try std.testing.expectEqualStrings("a40ff3d5900fb7698b8c865041347cb49eccedc8f93945f89629ad104aaecce4", doc.strField("audio_sha256").?);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "\"Wie spät ist es?\"") != null);
    try std.testing.expect(std.mem.endsWith(u8, bytes, "]]}\n"));
    try std.testing.expect(try voiceFile(a, "<|speaker:4|> hi", &codes, "RIFF", .{ .object = encoder }, &problem) == null);
    try std.testing.expectEqualStrings("the voice file would not load: Drama 3 request: only speaker 0 is supported; got speaker tags [4]", problem);
}
