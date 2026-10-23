//! tf-drama-probe's requests, one JSON object a line, lowered to prompt rows as POST /v1/audio/speech lowers input.
const std = @import("std");
const json = @import("json.zig");
const errors = @import("errors.zig");
const speech = @import("speech.zig");
const reply_text = @import("reply_text.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_voice = @import("drama_voice.zig");
const model_text = @import("model_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// A request's fields; salt and sampling default to what the server sends (seed salt 0, the model's defaults).
pub const Request = struct {
    /// Names the request's output files: a voice name's characters.
    id: []const u8,
    /// A voice file's name in the probe's voices directory; null or ``default``: the built-in voice, no references.
    voice: ?[]const u8,
    text: []const u8,
    salt: i64 = 0,
    temperature: f32 = drama_prompt.temperature,
    top_p: f32 = drama_prompt.top_p,
    top_k: u32 = drama_prompt.top_k,
    max_frames: u32,
};

const fields = [_][]const u8{ "id", "voice", "text", "salt", "temperature", "top_p", "top_k", "max_frames" };

/// The most frames a request may ask for: the engine's context (8192 rows) must keep a prompt row beside them.
pub const max_frames = 8191;

/// The request on `line`, or null with `problem` set; sampling is checked as the tower's sampler checks it.
pub fn parse(a: Allocator, line: []const u8, problem: *[]const u8) Allocator.Error!?Request {
    const doc = switch (try json.parse(a, line)) {
        .ok => |v| v,
        else => return fail(problem, "not a JSON object"),
    };
    if (doc != .object) return fail(problem, "not a JSON object");
    for (doc.object.keys()) |key| if (!listed(key)) {
        problem.* = try std.fmt.allocPrint(a, "unknown field {s}", .{key});
        return null;
    };
    var r: Request = .{
        .id = doc.strField("id") orelse return fail(problem, "id must be a string"),
        .voice = switch (doc.get("voice") orelse Value{ .bool = false }) {
            .null => null,
            .string => |s| s,
            else => return fail(problem, "voice must be a string or null"),
        },
        .text = doc.strField("text") orelse return fail(problem, "text must be a string"),
        .max_frames = 0,
    };
    if (!drama_voice.validName(r.id)) return fail(problem, "id must be 1..64 of A-Z a-z 0-9 . _ - led by a letter or digit, not default");
    if (doc.get("salt")) |v| r.salt = v.int64() orelse return fail(problem, "salt must be an integer");
    if (doc.get("temperature")) |v| r.temperature = @floatCast(number(v) orelse return fail(problem, "temperature must be a number"));
    if (doc.get("top_p")) |v| r.top_p = @floatCast(number(v) orelse return fail(problem, "top_p must be a number"));
    if (doc.get("top_k")) |v| r.top_k = std.math.cast(u32, v.int64() orelse 0) orelse 0;
    r.max_frames = std.math.cast(u32, (doc.get("max_frames") orelse Value{ .null = {} }).int64() orelse 0) orelse 0;
    if (!std.math.isFinite(r.temperature) or r.temperature < 0) return fail(problem, "temperature must be finite and at least 0");
    if (!std.math.isFinite(r.top_p) or r.top_p <= 0 or r.top_p > 1) return fail(problem, "top_p must be in (0, 1]");
    if (r.top_k == 0 or r.top_k > std.math.maxInt(i32)) return fail(problem, "top_k must be an integer in 1..2147483647");
    if (r.max_frames == 0 or r.max_frames > max_frames) return fail(problem, "max_frames must be an integer in 1..8191");
    return r;
}

fn fail(problem: *[]const u8, message: []const u8) ?Request {
    problem.* = message;
    return null;
}

fn listed(key: []const u8) bool {
    for (fields) |f| if (std.mem.eql(u8, f, key)) return true;
    return false;
}

/// A JSON number (bools are not).
fn number(v: Value) ?f64 {
    return if (v == .bool) null else v.float64();
}

/// The request's prompt rows, refused as the speech route refuses its input and voice (``cx.message`` says why).
pub fn prompt(cx: *errors.Cx, text: model_text.Text, voices: []const drama_voice.Voice, r: Request) errors.Refused!drama_prompt.Rows {
    if (!std.unicode.utf8ValidateSlice(r.text)) return cx.refuse("input must be valid UTF-8 text");
    if (reply_text.pyStrip(r.text).len == 0) return cx.refuse("input must be a non-empty string");
    if ((std.unicode.utf8CountCodepoints(r.text) catch unreachable) > speech.max_input_chars)
        return cx.fail(.request, "input must be at most {d} characters", .{speech.max_input_chars});
    const name = r.voice orelse "default";
    const voice: []const drama_prompt.Reference = if (std.ascii.eqlIgnoreCase(name, "default")) &.{} else if (drama_voice.find(voices, name)) |v| v.references else
        return cx.fail(.request, "unknown voice {s}: the voices directory holds no {s}.json", .{ name, name });
    const references = try drama_voice.lower(cx, r.text, .null, voice);
    if (references.len == 0) if (try drama_prompt.problem(cx.a, r.text)) |message| return cx.refuse(message);
    return drama_prompt.promptRows(cx.a, text, r.text, references) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.other("the tokenizer cannot encode this prompt"),
    };
}

test "a request line: required fields, the server's sampling defaults, and refusals naming the field" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const plain = (try parse(a, "{\"id\": \"en-1\", \"voice\": null, \"text\": \"Hello.\", \"max_frames\": 300}", &problem)).?;
    try std.testing.expectEqualDeep(Request{ .id = "en-1", .voice = null, .text = "Hello.", .max_frames = 300 }, plain);
    const full = (try parse(a, "{\"id\": \"b\", \"voice\": \"alto\", \"text\": \"x\", \"salt\": -3, \"temperature\": 0, \"top_p\": 1, \"top_k\": 1, \"max_frames\": 8191}", &problem)).?;
    try std.testing.expectEqualDeep(Request{ .id = "b", .voice = "alto", .text = "x", .salt = -3, .temperature = 0, .top_p = 1, .top_k = 1, .max_frames = 8191 }, full);
    const head = "{\"id\": \"a\", \"voice\": \"v\", \"text\": \"x\", ";
    const refused = [_][2][]const u8{
        .{ "[1]", "not a JSON object" },
        .{ "{\"id\": \"a\", \"voice\": \"v\", \"text\": \"x\", \"max_frames\": 9, \"seed\": 1}", "unknown field seed" },
        .{ "{\"voice\": \"v\", \"text\": \"x\", \"max_frames\": 9}", "id must be a string" },
        .{ "{\"id\": \"a\", \"text\": \"x\", \"max_frames\": 9}", "voice must be a string or null" },
        .{ "{\"id\": \"../a\", \"voice\": \"v\", \"text\": \"x\", \"max_frames\": 9}", "id must be 1..64 of A-Z a-z 0-9 . _ - led by a letter or digit, not default" },
        .{ "{\"id\": \"a\", \"voice\": \"v\", \"text\": 3, \"max_frames\": 9}", "text must be a string" },
        .{ head ++ "\"max_frames\": 0}", "max_frames must be an integer in 1..8191" },
        .{ head ++ "\"max_frames\": 8192}", "max_frames must be an integer in 1..8191" },
        .{ head ++ "\"text\": \"x\"}", "max_frames must be an integer in 1..8191" },
        .{ head ++ "\"max_frames\": 9, \"salt\": 1.5}", "salt must be an integer" },
        .{ head ++ "\"max_frames\": 9, \"temperature\": -1}", "temperature must be finite and at least 0" },
        .{ head ++ "\"max_frames\": 9, \"temperature\": true}", "temperature must be a number" },
        .{ head ++ "\"max_frames\": 9, \"top_p\": 0}", "top_p must be in (0, 1]" },
        .{ head ++ "\"max_frames\": 9, \"top_k\": 0}", "top_k must be an integer in 1..2147483647" },
        .{ head ++ "\"max_frames\": 9, \"top_k\": 2.0}", "top_k must be an integer in 1..2147483647" },
    };
    for (refused) |case| {
        std.testing.expect(try parse(a, case[0], &problem) == null) catch |e| {
            std.debug.print("accepted: {s}\n", .{case[0]});
            return e;
        };
        try std.testing.expectEqualStrings(case[1], problem);
    }
}

test "a request's rows are the speech route's: the built-in voice plain, a named voice with its references" {
    const harness = @import("speech_stream_test.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bytes: harness.Bytes = .{};
    const text = bytes.text();
    const frames = [_]drama_prompt.Codes{ .{ 5, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, .{ 4095, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    const references = [_]drama_prompt.Reference{.{ .text = "Hello there.", .frames = &frames }};
    const voices = [_]drama_voice.Voice{.{ .name = "alto", .references = &references, .source = .@"model-invented" }};
    var cx: errors.Cx = .{ .a = a };
    const plain = try prompt(&cx, text, &voices, .{ .id = "a", .voice = null, .text = "Say it.", .max_frames = 9 });
    try std.testing.expectEqualDeep(plain, try prompt(&cx, text, &voices, .{ .id = "a", .voice = "Default", .text = "Say it.", .max_frames = 9 }));
    try std.testing.expectEqualSlices(u32, try drama_prompt.encode(a, text, "Say it."), plain.tokens);
    try std.testing.expectEqual(@as(usize, 0), plain.codes.len);
    const voiced = try prompt(&cx, text, &voices, .{ .id = "a", .voice = "alto", .text = "Say it.", .max_frames = 9 });
    const want = try drama_prompt.encodeReferenced(a, text, "Say it.", &references);
    try std.testing.expectEqualSlices(u32, want.tokens, voiced.tokens);
    try std.testing.expectEqualSlices(drama_prompt.Codes, want.codes, voiced.codes);
    const refused = [_][2][]const u8{
        .{ "Alto", "unknown voice Alto: the voices directory holds no Alto.json" },
        .{ "default", "Drama 3 request: input: <|voice|> is written by the prompt builder and must not appear in caller text" },
    };
    for (refused) |case| {
        try std.testing.expectError(error.Refused, prompt(&cx, text, &voices, .{ .id = "a", .voice = case[0], .text = "a <|voice|> b", .max_frames = 9 }));
        try std.testing.expectEqualStrings(case[1], cx.message);
    }
    try std.testing.expectError(error.Refused, prompt(&cx, text, &voices, .{ .id = "a", .voice = "alto", .text = " \n", .max_frames = 9 }));
    try std.testing.expectEqualStrings("input must be a non-empty string", cx.message);
}
