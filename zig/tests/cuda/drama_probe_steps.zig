//! tf-drama-probe's frame loop and teacher runs apart from the GPU: what each step feeds, keeps and refuses.
const std = @import("std");
const server = @import("server");
const Codes = server.drama_prompt.Codes;
const Allocator = std.mem.Allocator;

pub const Frame = [11]u32;
pub const Finish = enum { eos, length };
/// Codebooks 1..9, one tower head each.
pub const heads = 9;

/// A teacher run's frames for one request: its codes and how it ended.
pub const Forced = struct { codes: []const Codes, finish: Finish };

/// A probe manifest's request entry, the fields forcing checks.
pub const Entry = struct { id: []const u8, prompt_rows: usize, prompt_sha256: []const u8, frames: usize, steps: usize, finish: Finish, codes_sha256: []const u8 };

/// A probe manifest, as forcing reads it.
pub const Teacher = struct { requests: []const Entry };

/// The steps a run of `frames` frames that ended `finish` took: the EOS decision is a step of its own.
pub fn stepsOf(finish: Finish, frames: usize) usize {
    return if (finish == .eos) frames + 1 else frames;
}

/// The teacher manifest in `bytes`, or null with `problem` set when it is not one a probe wrote whole.
pub fn teacher(a: Allocator, bytes: []const u8, problem: *[]const u8) !?Teacher {
    const t = std.json.parseFromSliceLeaky(Teacher, a, bytes, .{ .ignore_unknown_fields = true }) catch |e| {
        if (e == error.OutOfMemory) return e;
        problem.* = "the teacher's manifest.json is not a probe manifest";
        return null;
    };
    for (t.requests, 0..) |e, i| {
        for (t.requests[0..i]) |earlier| if (std.mem.eql(u8, earlier.id, e.id)) {
            problem.* = try std.fmt.allocPrint(a, "the teacher's manifest lists {s} twice", .{e.id});
            return null;
        };
        if (e.steps != stepsOf(e.finish, e.frames) or (e.finish == .length and e.frames == 0)) {
            problem.* = try std.fmt.allocPrint(a, "the teacher's {s}: {d} steps disagree with {d} frames ending {t}", .{ e.id, e.steps, e.frames, e.finish });
            return null;
        }
    }
    return t;
}

/// Request `id`'s teacher frames from `bytes`, or null with `problem` set where they differ from the teacher's.
pub fn forced(a: Allocator, t: Teacher, id: []const u8, prompt_rows: usize, prompt_sha256: []const u8, max_frames: usize, bytes: []const u8, problem: *[]const u8) !?Forced {
    const e = for (t.requests) |e| {
        if (std.mem.eql(u8, e.id, id)) break e;
    } else return refuse(problem, "the teacher's manifest has no such request");
    if (e.prompt_rows != prompt_rows) {
        problem.* = try std.fmt.allocPrint(a, "its prompt has {d} rows, the teacher's {d}", .{ prompt_rows, e.prompt_rows });
        return null;
    }
    if (!std.mem.eql(u8, e.prompt_sha256, prompt_sha256)) return refuse(problem, "its prompt is not the teacher's");
    if (bytes.len != e.frames * @sizeOf(Codes)) {
        problem.* = try std.fmt.allocPrint(a, "the teacher's codes file holds {d} bytes, not its manifest's {d} frames", .{ bytes.len, e.frames });
        return null;
    }
    if (e.frames > max_frames) return refuse(problem, "the teacher's frames exceed its max_frames");
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), e.codes_sha256)) return refuse(problem, "the teacher's codes file does not match its manifest's codes_sha256");
    const codes = try a.alloc(Codes, e.frames);
    @memcpy(std.mem.sliceAsBytes(codes), bytes);
    for (codes) |frame| if (std.mem.max(u32, &frame) >= server.drama_prompt.codebook_size) return refuse(problem, "the teacher's codes are not all below 4096");
    return .{ .codes = codes, .finish = e.finish };
}

fn refuse(problem: *[]const u8, why: []const u8) ?Forced {
    problem.* = why;
    return null;
}

/// The frame a teacher's codes feed back: build_prompt_rows' audio row, its semantic token then all ten codes.
pub fn teacherFrame(semantic_start: u32, codes: Codes) Frame {
    var frame: Frame = undefined;
    frame[0] = semantic_start + codes[0];
    frame[1..].* = codes;
    return frame;
}

/// The server's frame loop over `model` (prompt, save, sample, force, feed); forced runs follow the teacher's frames.
pub fn run(model: anytype, gpa: Allocator, semantic_start: u32, teacher_frames: ?Forced, max_frames: usize, codes: *std.ArrayList(Codes)) !Finish {
    codes.clearRetainingCapacity();
    try model.prompt();
    while (true) {
        try model.save();
        const frame: Frame = if (teacher_frames) |f| blk: {
            if (codes.items.len == f.codes.len) return .eos;
            const frame = teacherFrame(semantic_start, f.codes[codes.items.len]);
            try model.force(frame);
            break :blk frame;
        } else try model.sample() orelse return .eos;
        try codes.append(gpa, frame[1..].*);
        const cap = if (teacher_frames) |f| (if (f.finish == .length) f.codes.len else std.math.maxInt(usize)) else max_frames;
        if (codes.items.len == cap) return .length;
        try model.feed(frame);
    }
}

/// Every value finite: a raw semantic row the probe may keep.
pub fn finite(values: []const f32) bool {
    for (values) |v| if (!std.math.isFinite(v)) return false;
    return true;
}

/// The NLL of `code` under a head's bf16 logits at T = 1: logsumexp in f64 less the code's logit, stored f32.
pub fn headNll(logits: []const u16, code: u32) !f32 {
    var top = -std.math.inf(f64);
    for (logits) |bits| {
        const v: f64 = @as(f32, @bitCast(@as(u32, bits) << 16));
        if (!std.math.isFinite(v)) return error.NonFiniteHeadLogits;
        top = @max(top, v);
    }
    var sum: f64 = 0;
    for (logits) |bits| sum += @exp(@as(f64, @as(f32, @bitCast(@as(u32, bits) << 16))) - top);
    const nll: f32 = @floatCast(top + @log(sum) - @as(f32, @bitCast(@as(u32, logits[code]) << 16)));
    if (!std.math.isFinite(nll)) return error.NonFiniteHeadNll;
    return nll;
}

/// A model that records what the loop asks of it: `samples` are its sampled frames, null an EOS.
const Fake = struct {
    samples: []const ?Frame = &.{},
    sampled: usize = 0,
    saves: usize = 0,
    prompts: usize = 0,
    forced: std.ArrayList(Frame) = .empty,
    fed: std.ArrayList(Frame) = .empty,

    fn prompt(f: *Fake) !void {
        if (f.saves != 0 or f.fed.items.len != 0) return error.TestUnexpectedResult;
        f.prompts += 1;
    }
    fn save(f: *Fake) !void {
        if (f.saves != f.fed.items.len) return error.TestUnexpectedResult; // one logits row a backbone pass
        f.saves += 1;
    }
    fn sample(f: *Fake) !?Frame {
        f.sampled += 1;
        return f.samples[f.sampled - 1];
    }
    fn force(f: *Fake, frame: Frame) !void {
        try f.forced.append(std.testing.allocator, frame);
    }
    fn feed(f: *Fake, frame: Frame) !void {
        try f.fed.append(std.testing.allocator, frame);
    }
    fn deinit(f: *Fake) void {
        f.forced.deinit(std.testing.allocator);
        f.fed.deinit(std.testing.allocator);
    }
};

const start = 1000;

/// Three distinct teacher frames, every codebook different in each.
fn teacherCodes() [3]Codes {
    var codes: [3]Codes = undefined;
    for (&codes, 0..) |*frame, f| for (frame, 0..) |*c, book| {
        c.* = @intCast(100 * f + 7 * book + 1);
    };
    return codes;
}

test "a forced run feeds each teacher frame whole, in order, and ends as the teacher did" {
    const gpa = std.testing.allocator;
    const t = teacherCodes();
    var codes: std.ArrayList(Codes) = .empty;
    defer codes.deinit(gpa);
    // eos: three frames forced and fed, then the EOS step's logits; the student never samples
    var eos: Fake = .{};
    defer eos.deinit();
    try std.testing.expectEqual(Finish.eos, try run(&eos, gpa, start, .{ .codes = &t, .finish = .eos }, 9, &codes));
    try std.testing.expectEqual(@as(usize, 4), eos.saves);
    try std.testing.expectEqual(@as(usize, 0), eos.sampled);
    for (t, eos.forced.items, eos.fed.items) |want, forced_frame, fed| {
        try std.testing.expectEqualSlices(u32, &teacherFrame(start, want), &forced_frame);
        try std.testing.expectEqualSlices(u32, &forced_frame, &fed);
        try std.testing.expectEqual(start + want[0], fed[0]);
    }
    try std.testing.expectEqualSlices(Codes, &t, codes.items);
    // length: three frames forced, the last never fed, no EOS step
    var length: Fake = .{};
    defer length.deinit();
    try std.testing.expectEqual(Finish.length, try run(&length, gpa, start, .{ .codes = &t, .finish = .length }, 9, &codes));
    try std.testing.expectEqual(@as(usize, 3), length.saves);
    try std.testing.expectEqual(@as(usize, 3), length.forced.items.len);
    try std.testing.expectEqual(@as(usize, 2), length.fed.items.len);
    try std.testing.expectEqualSlices(u32, &teacherFrame(start, t[1]), &length.fed.items[1]);
    try std.testing.expectEqual(stepsOf(.length, 3), length.saves);
    try std.testing.expectEqual(stepsOf(.eos, 3), eos.saves);
}

test "a sampled run stops at EOS or at max_frames, keeping one logits row a step" {
    const gpa = std.testing.allocator;
    var codes: std.ArrayList(Codes) = .empty;
    defer codes.deinit(gpa);
    const t = teacherCodes();
    const frames = [_]?Frame{ teacherFrame(start, t[0]), teacherFrame(start, t[1]), null };
    var eos: Fake = .{ .samples = &frames };
    defer eos.deinit();
    try std.testing.expectEqual(Finish.eos, try run(&eos, gpa, start, null, 9, &codes));
    try std.testing.expectEqual(@as(usize, 3), eos.saves);
    try std.testing.expectEqual(@as(usize, 2), eos.fed.items.len);
    try std.testing.expectEqualSlices(u32, &frames[1].?, &eos.fed.items[1]);
    var capped: Fake = .{ .samples = &frames };
    defer capped.deinit();
    try std.testing.expectEqual(Finish.length, try run(&capped, gpa, start, null, 2, &codes));
    try std.testing.expectEqual(@as(usize, 2), capped.saves);
    try std.testing.expectEqual(@as(usize, 1), capped.fed.items.len);
    var silent: Fake = .{ .samples = &.{null} };
    defer silent.deinit();
    try std.testing.expectEqual(Finish.eos, try run(&silent, gpa, start, null, 9, &codes));
    try std.testing.expectEqual(@as(usize, 1), silent.saves);
    try std.testing.expectEqual(@as(usize, 0), codes.items.len);
}

test "a teacher manifest whose ids repeat or whose steps disagree with its finish is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const entry = "\"prompt_rows\": 5, \"prompt_sha256\": \"p\", \"codes_sha256\": \"c\"";
    const ok = try std.fmt.allocPrint(a, "{{\"requests\": [{{\"id\": \"a\", \"frames\": 2, \"steps\": 3, \"finish\": \"eos\", {s}}}, {{\"id\": \"b\", \"frames\": 2, \"steps\": 2, \"finish\": \"length\", {s}}}]}}", .{ entry, entry });
    try std.testing.expectEqual(@as(usize, 2), (try teacher(a, ok, &problem)).?.requests.len);
    const refused = [_][2][]const u8{
        .{ "{\"id\": \"a\", \"frames\": 2, \"steps\": 3, \"finish\": \"eos\", E}, {\"id\": \"a\", \"frames\": 2, \"steps\": 3, \"finish\": \"eos\", E}", "the teacher's manifest lists a twice" },
        .{ "{\"id\": \"a\", \"frames\": 2, \"steps\": 2, \"finish\": \"eos\", E}", "the teacher's a: 2 steps disagree with 2 frames ending eos" },
        .{ "{\"id\": \"a\", \"frames\": 2, \"steps\": 3, \"finish\": \"length\", E}", "the teacher's a: 3 steps disagree with 2 frames ending length" },
        .{ "{\"id\": \"a\", \"frames\": 0, \"steps\": 0, \"finish\": \"length\", E}", "the teacher's a: 0 steps disagree with 0 frames ending length" },
        .{ "{\"id\": \"a\", \"frames\": 2, \"finish\": \"eos\", E}", "the teacher's manifest.json is not a probe manifest" },
    };
    for (refused) |case| {
        const items = try std.mem.replaceOwned(u8, a, case[0], "E", entry);
        try std.testing.expect(try teacher(a, try std.fmt.allocPrint(a, "{{\"requests\": [{s}]}}", .{items}), &problem) == null);
        try std.testing.expectEqualStrings(case[1], problem);
    }
}

test "forcing refuses another prompt, a truncated or stale codes file and codes out of range" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const codes = teacherCodes();
    const bytes = std.mem.sliceAsBytes(&codes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const t: Teacher = .{ .requests = &.{.{ .id = "a", .prompt_rows = 5, .prompt_sha256 = "p", .frames = 3, .steps = 3, .finish = .length, .codes_sha256 = &hex }} };
    var problem: []const u8 = "";
    const got = (try forced(a, t, "a", 5, "p", 9, bytes, &problem)).?;
    try std.testing.expectEqualSlices(Codes, &codes, got.codes);
    try std.testing.expectEqual(Finish.length, got.finish);
    var stale = codes;
    stale[2][4] += 1; // same length, other history
    var high = codes;
    high[1][9] = 4096;
    const cases = [_]struct { id: []const u8 = "a", rows: usize = 5, sha: []const u8 = "p", max: usize = 9, bytes: []const u8, why: []const u8 }{
        .{ .id = "b", .bytes = bytes, .why = "the teacher's manifest has no such request" },
        .{ .rows = 6, .bytes = bytes, .why = "its prompt has 6 rows, the teacher's 5" },
        .{ .sha = "q", .bytes = bytes, .why = "its prompt is not the teacher's" },
        .{ .bytes = bytes[0 .. bytes.len - 40], .why = "the teacher's codes file holds 80 bytes, not its manifest's 3 frames" },
        .{ .bytes = bytes[0 .. bytes.len - 1], .why = "the teacher's codes file holds 119 bytes, not its manifest's 3 frames" },
        .{ .max = 2, .bytes = bytes, .why = "the teacher's frames exceed its max_frames" },
        .{ .bytes = std.mem.sliceAsBytes(&stale), .why = "the teacher's codes file does not match its manifest's codes_sha256" },
    };
    for (cases) |c| {
        try std.testing.expect(try forced(a, t, c.id, c.rows, c.sha, c.max, c.bytes, &problem) == null);
        try std.testing.expectEqualStrings(c.why, problem);
    }
    const high_bytes = std.mem.sliceAsBytes(&high);
    std.crypto.hash.sha2.Sha256.hash(high_bytes, &digest, .{});
    const high_hex = std.fmt.bytesToHex(digest, .lower);
    const t_high: Teacher = .{ .requests = &.{.{ .id = "a", .prompt_rows = 5, .prompt_sha256 = "p", .frames = 3, .steps = 3, .finish = .length, .codes_sha256 = &high_hex }} };
    try std.testing.expect(try forced(a, t_high, "a", 5, "p", 9, high_bytes, &problem) == null);
    try std.testing.expectEqualStrings("the teacher's codes are not all below 4096", problem);
}

test "head NLL is logsumexp less the code's logit, and refuses non-finite logits" {
    const bf = struct {
        fn of(x: f32) u16 {
            return @intCast(@as(u32, @bitCast(x)) >> 16);
        }
    }.of;
    var logits: [4096]u16 = @splat(0);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@log(4096.0))), try headNll(&logits, 7), 1e-6);
    logits[7] = bf(2);
    const sum = 4095.0 + @exp(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@log(sum) - 2.0)), try headNll(&logits, 7), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@log(sum))), try headNll(&logits, 8), 1e-6);
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        logits[4000] = bf(bad);
        try std.testing.expectError(error.NonFiniteHeadLogits, headNll(&logits, 7));
    }
    try std.testing.expect(finite(&.{ 1, -3e38, 0 }));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| try std.testing.expect(!finite(&.{ 1, bad }));
}
