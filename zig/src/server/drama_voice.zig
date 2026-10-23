//! Drama 3 voice references: a request's ``references`` and the ``--voices`` files, refused in the reference's order.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const drama_prompt = @import("drama_prompt.zig");
const model_text = @import("model_text.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Reference = drama_prompt.Reference;
const Codes = drama_prompt.Codes;
const Allocator = std.mem.Allocator;
const prefix = drama_prompt.error_prefix;

/// The protocol's ``SpeechReference`` fields; a reference carrying any other is refused, not dropped.
const reference_fields = [_][]const u8{ "text", "vq_codes", "audio_path", "ref_audio", "audio", "data", "media_type" };

/// The fields that send audio for the reference's codec encoder, which this server does not run.
const audio_fields = [_][]const u8{ "audio_path", "ref_audio", "audio", "data" };

/// What ``tools/zig/drama_voice.py`` and ``tensorfold-native voice encode`` write: one reference.
pub const voice_format = "drama3-voice/1";

/// What ``drama_voice.py combine`` writes: several references, spoken in order; a loader that reads only /1 refuses it.
pub const voices_format = "drama3-voice/2";

/// A voice file's ``source.kind``: whose voice it holds, the consent rule in docs/drama-voices.md.
pub const Source = enum { @"model-invented", owner, consented };

/// A voice file's references, in prompt order, named by its file's stem.
pub const Voice = struct { name: []const u8, references: []const Reference, source: Source };

/// A [codebooks, frames] grid of int64 codes, row-major.
const Grid = struct { rows: usize, cols: usize, values: []const i64 };

/// A request's ``references`` (null or a list) then `voice`'s, refused as ``Drama3SpeechPreprocessor`` refuses them.
pub fn lower(cx: *Cx, input: []const u8, sent: Value, voice: []const Reference) errors.Refused![]const Reference {
    const items: []const Value = switch (sent) {
        .null => &.{},
        .array => |x| x,
        else => return cx.refuse(prefix ++ " references must be a list of objects"),
    };
    if (items.len == 0 and voice.len == 0) return &.{};
    for (items, 0..) |item, i| {
        if (item != .object) return cx.refuse(prefix ++ " references must be a list of objects");
        for (item.object.keys()) |key| if (!listed(&reference_fields, key))
            return cx.fail(.request, "unknown speech reference field: references.{d}.{s}", .{ i, key });
    }
    const a = cx.a;
    const texts = try a.alloc([]const u8, 1 + items.len + voice.len);
    texts[0] = input;
    for (items, texts[1..][0..items.len], 0..) |item, *text, i| {
        // descriptor.get("text") or "": a falsy text is an empty transcript
        const t = item.get("text") orelse Value.null;
        text.* = if (!t.truthy()) "" else if (t == .string) t.string else return cx.fail(.request, prefix ++ " references.{d}.text must be a string", .{i});
        if (!std.unicode.utf8ValidateSlice(text.*)) return cx.fail(.request, prefix ++ " references.{d}.text must be valid UTF-8 text", .{i});
    }
    for (voice, texts[1 + items.len ..]) |v, *text| text.* = v.text;
    if (try drama_prompt.otherSpeakers(a, texts)) |message| return cx.refuse(message);
    const grids = try a.alloc(Grid, items.len);
    for (items, grids, 0..) |item, *g, i| {
        if (item.field("vq_codes")) |codes| {
            g.* = try grid(a, codes) orelse return cx.fail(.request, prefix ++ " references.{d}.vq_codes must be a [codebooks, frames] grid of int64 integers", .{i});
        } else for (audio_fields) |name| {
            if (item.field(name) != null) return cx.fail(.request, prefix ++ " references.{d}: reference audio is not encoded by this server; send its vq_codes (tools/zig/drama_voice.py)", .{i});
        } else return cx.fail(.request, prefix ++ " references.{d} needs audio or vq_codes", .{i});
    }
    const out = try a.alloc(Reference, items.len + voice.len);
    for (grids, out[0..items.len], 0..) |g, *reference, i| {
        const location = try std.fmt.allocPrint(a, "references.{d}", .{i});
        reference.* = .{ .text = texts[1 + i], .frames = undefined };
        if (try drama_prompt.callerText(a, try drama_prompt.tagged(a, reference.text), location)) |message| return cx.refuse(message);
        reference.frames = try frames(a, g) orelse return cx.refuse(try codesProblem(a, g, location));
    }
    @memcpy(out[items.len..], voice);
    if (try drama_prompt.callerText(a, input, "input")) |message| return cx.refuse(message);
    return out;
}

fn listed(names: []const []const u8, key: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, key)) return true;
    return false;
}

/// ``reference_codes``: a non-empty list of equally long, non-empty lists of int64 integers (bools are not), else null.
fn grid(a: Allocator, v: Value) Allocator.Error!?Grid {
    const rows = v.items() orelse return null;
    if (rows.len == 0) return null;
    const cols = (rows[0].items() orelse return null).len;
    if (cols == 0) return null;
    const values = try a.alloc(i64, rows.len * cols);
    for (rows, 0..) |row, r| {
        const codes = row.items() orelse return null;
        if (codes.len != cols) return null;
        for (codes, values[r * cols ..][0..cols]) |code, *out| out.* = code.int64() orelse return null;
    }
    return .{ .rows = rows.len, .cols = cols, .values = values };
}

/// The grid as frames when ``validate_audio_codes`` accepts it: [10, frames] codes in [0, 4096); else null.
fn frames(a: Allocator, g: Grid) Allocator.Error!?[]Codes {
    if (g.rows != drama_prompt.codebooks) return null;
    for (g.values) |code| if (code < 0 or code >= drama_prompt.codebook_size) return null;
    const out = try a.alloc(Codes, g.cols);
    for (out, 0..) |*frame, f| {
        for (frame, 0..) |*code, book| code.* = @intCast(g.values[book * g.cols + f]);
    }
    return out;
}

/// ``validate_audio_codes``'s refusal of a grid `frames` rejects.
fn codesProblem(a: Allocator, g: Grid, location: []const u8) Allocator.Error![]const u8 {
    if (g.rows != drama_prompt.codebooks)
        return std.fmt.allocPrint(a, prefix ++ " {s}: audio codes must be an int32 or int64 [{d}, frames] tensor with at least one frame; got torch.int64 ({d}, {d})", .{ location, drama_prompt.codebooks, g.rows, g.cols });
    return std.fmt.allocPrint(a, prefix ++ " {s}: audio codes must be in [0, {d})", .{ location, drama_prompt.codebook_size });
}

/// The digest a voice file carries: its [codebooks, frames] codes as little-endian int32, codebook-major.
pub fn codesDigest(reference: Reference) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..drama_prompt.codebooks) |book| for (reference.frames) |frame| {
        h.update(&std.mem.toBytes(std.mem.nativeToLittle(i32, @intCast(frame[book]))));
    };
    return h.finalResult();
}

/// A voice file as ``tools/zig/drama_voice.py`` writes it: ``json.dumps(voice, ensure_ascii=False)`` and a newline.
pub fn file(a: Allocator, reference: Reference, audio_sha256: []const u8, encoder: Value) Allocator.Error![]u8 {
    const voice = try json.newObject(a);
    try voice.put(a, "format", .{ .string = voice_format });
    try voice.put(a, "text", .{ .string = reference.text });
    try voice.put(a, "codes_sha256", .{ .string = try a.dupe(u8, &std.fmt.bytesToHex(codesDigest(reference), .lower)) });
    try voice.put(a, "frames", try json.intValue(a, reference.frames.len));
    try voice.put(a, "audio_sha256", .{ .string = audio_sha256 });
    try voice.put(a, "encoder", encoder);
    const rows = try a.alloc(Value, drama_prompt.codebooks);
    for (rows, 0..) |*row, book| {
        const codes = try a.alloc(Value, reference.frames.len);
        for (codes, reference.frames) |*c, frame| c.* = try json.intValue(a, frame[book]);
        row.* = .{ .array = codes };
    }
    try voice.put(a, "vq_codes", .{ .array = rows });
    return std.mem.concat(a, u8, &.{ try json.stringify(a, .{ .object = voice }, .{ .ascii = false }), "\n" });
}

/// A voice name: 1..64 of ``[A-Za-z0-9._-]`` led by a letter or digit; ``default`` names the built-in voice.
pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64 or !std.ascii.isAlphanumeric(name[0]) or std.ascii.eqlIgnoreCase(name, "default")) return false;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '.' and c != '_' and c != '-') return false;
    return true;
}

/// The voice file `name`, or null with `problem` set; its transcripts and codes pass the request's checks.
pub fn parse(a: Allocator, name: []const u8, bytes: []const u8, problem: *[]const u8) Allocator.Error!?Voice {
    const doc = try document(a, bytes, problem) orelse return null;
    const format = doc.strField("format") orelse "";
    const several = std.mem.eql(u8, format, voices_format);
    if (!several and !std.mem.eql(u8, format, voice_format)) {
        problem.* = "format is not " ++ voice_format ++ " or " ++ voices_format;
        return null;
    }
    const source = doc.get("source") orelse .null;
    const kind = std.meta.stringToEnum(Source, source.strField("kind") orelse "");
    if (kind == null or std.mem.trim(u8, json.strOr(source.get("note")), &std.ascii.whitespace).len == 0) {
        problem.* = "source must say whose voice this is: {\"kind\": \"model-invented\" | \"owner\" | \"consented\", \"note\": \"...\"} (docs/drama-voices.md)";
        return null;
    }
    const references = if (several) try fileReferences(a, doc, problem) orelse return null else try a.dupe(Reference, &.{try fileReference(a, doc, problem) orelse return null});
    return .{ .name = name, .references = references, .source = kind.? };
}

/// A single-reference voice file's reference as ``parse`` checks it, without the `source` ``drama_voice.py`` adds.
pub fn parseReference(a: Allocator, bytes: []const u8, problem: *[]const u8) Allocator.Error!?Reference {
    const doc = try document(a, bytes, problem) orelse return null;
    if (std.mem.eql(u8, doc.strField("format") orelse "", voice_format)) return fileReference(a, doc, problem);
    problem.* = "format is not " ++ voice_format;
    return null;
}

fn document(a: Allocator, bytes: []const u8, problem: *[]const u8) Allocator.Error!?Value {
    return switch (try json.parse(a, bytes)) {
        .ok => |v| v,
        else => {
            problem.* = "not JSON";
            return null;
        },
    };
}

/// One reference's ``text``, ``codes_sha256`` and ``vq_codes``: a /1 file's top level, or an entry of a /2 file's list.
fn fileReference(a: Allocator, doc: Value, problem: *[]const u8) Allocator.Error!?Reference {
    const text = doc.strField("text") orelse "";
    const digest = doc.strField("codes_sha256") orelse "";
    const g = try grid(a, doc.get("vq_codes") orelse .null);
    const codes = if (g) |x| try frames(a, x) else null;
    if (!std.unicode.utf8ValidateSlice(text)) {
        problem.* = "text is not UTF-8";
    } else if (try drama_prompt.otherSpeakers(a, &.{text}) orelse try drama_prompt.callerText(a, try drama_prompt.tagged(a, text), "text")) |message| {
        problem.* = message;
    } else if (codes == null) {
        problem.* = "vq_codes is not a [10, frames] grid of codes in [0, 4096)";
    } else {
        const reference: Reference = .{ .text = text, .frames = codes.? };
        if (std.mem.eql(u8, digest, &std.fmt.bytesToHex(codesDigest(reference), .lower))) return reference;
        problem.* = "codes_sha256 does not match vq_codes";
    }
    return null;
}

/// A /2 file's ``references``, in order: each checked as a /1 file is, and each naming its clip's ``audio_sha256``.
fn fileReferences(a: Allocator, doc: Value, problem: *[]const u8) Allocator.Error!?[]const Reference {
    const items = (doc.get("references") orelse Value{ .null = {} }).items() orelse &.{};
    if (items.len == 0) {
        problem.* = "references must be a non-empty list of {text, codes_sha256, audio_sha256, vq_codes}";
        return null;
    }
    const out = try a.alloc(Reference, items.len);
    for (items, out, 0..) |item, *reference, i| {
        var why: []const u8 = "audio_sha256 is not a clip's sha256 (64 lowercase hex digits)";
        const found = if (isDigest(json.strOr(item.get("audio_sha256")))) try fileReference(a, item, &why) else null;
        reference.* = found orelse {
            problem.* = try std.fmt.allocPrint(a, "references.{d}: {s}", .{ i, why });
            return null;
        };
    }
    return out;
}

fn isDigest(text: []const u8) bool {
    if (text.len != 64) return false;
    for (text) |c| if (!std.ascii.isDigit(c) and (c < 'a' or c > 'f')) return false;
    return true;
}

/// Null when every voice leaves room under `max_rows` for an input; else the named refusal of the first that does not.
pub fn overflow(a: Allocator, text: model_text.Text, voices: []const Voice, max_rows: u32) model_text.Error!?[]const u8 {
    for (voices) |v| {
        // the empty input still adds its speaker tag and the closing segments, so a real one needs more
        const rows = (try drama_prompt.encodeReferenced(a, text, "", v.references)).tokens.len;
        if (rows >= max_rows) return try std.fmt.allocPrint(a, "voice references too long: {s}.json's {d} references take {d} prompt rows before any input, and a prompt may take {d} (--max-frames and --context set it); drop a reference or raise --context", .{ v.name, v.references.len, rows, max_rows });
    }
    return null;
}

/// Every ``NAME.json`` regular file in `path`, by name, or null with `problem` set; other entries are skipped.
pub fn load(a: Allocator, io: std.Io, path: []const u8, problem: *[]const u8) Allocator.Error!?[]const Voice {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "--voices {s}: {s}", .{ path, @errorName(e) });
        return null;
    };
    defer dir.close(io);
    var voices: std.ArrayList(Voice) = .empty;
    var it = dir.iterate();
    while (it.next(io) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "--voices {s}: {s}", .{ path, @errorName(e) });
        return null;
    }) |entry| {
        // only regular files: a link could name a file outside the directory
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".json") or entry.name[0] == '.') continue;
        const name = try a.dupe(u8, entry.name[0 .. entry.name.len - ".json".len]);
        if (!validName(name)) {
            problem.* = try std.fmt.allocPrint(a, "--voices {s}: {s}: a voice name is 1..64 of A-Z a-z 0-9 . _ - led by a letter or digit, not default", .{ path, entry.name });
            return null;
        }
        const bytes = dir.readFileAlloc(io, entry.name, a, .limited(4 << 20)) catch |e| {
            problem.* = try std.fmt.allocPrint(a, "--voices {s}: {s}: {s}", .{ path, entry.name, @errorName(e) });
            return null;
        };
        var why: []const u8 = "";
        try voices.append(a, try parse(a, name, bytes, &why) orelse {
            problem.* = try std.fmt.allocPrint(a, "--voices {s}: {s}: {s}", .{ path, entry.name, why });
            return null;
        });
    }
    std.mem.sort(Voice, voices.items, {}, byName);
    return voices.items;
}

fn byName(_: void, x: Voice, y: Voice) bool {
    return std.mem.lessThan(u8, x.name, y.name);
}

/// The voice named `name` (names are case-sensitive), or null.
pub fn find(voices: []const Voice, name: []const u8) ?*const Voice {
    for (voices) |*v| if (std.mem.eql(u8, v.name, name)) return v;
    return null;
}
