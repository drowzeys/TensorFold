//! The register measure against the numpy one it was calibrated with: drama_register_fixture.json, register_fixture.py.
const std = @import("std");
const json = @import("json.zig");
const drama_register = @import("drama_register.zig");

const fixture = @embedFile("drama_register_fixture.json");

/// [seconds, start Hz, end Hz]: a falling glide of 4 harmonics at amplitude 0.3, or a rest of zeros when both are 0.
pub const Part = [3]f64;

/// Harmonic weights falling as 1/h, the generator's default.
pub const plain: [4]f64 = .{ 1, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0 };

/// `parts` as float32 samples at `rate`, as the fixture's generator writes them.
pub fn tones(a: std.mem.Allocator, rate: u32, parts: []const Part, harmonics: [4]f64) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    const sr: f64 = @floatFromInt(rate);
    for (parts) |part| {
        const n: usize = @intFromFloat(@round(part[0] * sr));
        for (0..n) |i| {
            const t = @as(f64, @floatFromInt(i)) / sr;
            const phase = part[1] * t + (part[2] - part[1]) * t * t / (2 * part[0]);
            var x: f64 = 0;
            if (part[1] != 0) for (harmonics, 1..) |w, h| {
                x += w * @sin(2 * std.math.pi * @as(f64, @floatFromInt(h)) * phase);
            };
            try out.append(a, @floatCast(0.3 * x));
        }
    }
    return out.items;
}

test "phrases, openings, excesses and high shares match the numpy reference; so does each trip" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = (try json.parse(a, fixture)).ok;
    const cases = doc.get("cases").?.array;
    try std.testing.expectEqual(@as(usize, 7), cases.len);
    for (cases) |case| {
        errdefer std.debug.print("case {s}\n", .{case.strField("name").?});
        var parts: std.ArrayList(Part) = .empty;
        for (case.get("parts").?.array) |p| try parts.append(a, .{ number(p.array[0]), number(p.array[1]), number(p.array[2]) });
        var harmonics: [4]f64 = undefined;
        for (&harmonics, case.get("harmonics").?.array) |*w, v| w.* = number(v);
        const rate: u32 = @intCast(case.get("rate").?.int64().?);
        const m = try drama_register.measure(a, try tones(a, rate, parts.items, harmonics), rate, 210);
        try std.testing.expectApproxEqAbs(number(case.get("median_hz").?), m.median_hz.?, 0.05);
        const want = case.get("phrases").?.array;
        try std.testing.expectEqual(want.len, m.phrases.len);
        for (want, m.phrases) |w, p| {
            try std.testing.expectEqual(w.get("start").?.int64().?, @as(i64, @intCast(p.start)));
            try std.testing.expectEqual(w.get("end").?.int64().?, @as(i64, @intCast(p.end)));
            try std.testing.expectApproxEqAbs(number(w.get("opening").?), p.opening.?, 0.011);
            try std.testing.expectApproxEqAbs(number(w.get("excess").?), p.excess.?, 0.011);
            try std.testing.expectApproxEqAbs(number(w.get("high").?), p.high.?, 0.0011);
        }
        try std.testing.expectEqual(case.get("trips").?.bool, drama_register.jump(m.phrases, 4, 0.2) != null);
    }
}

test "the trip is the phrase after the first with the largest excess that also holds enough frames over the line" {
    const P = drama_register.Phrase;
    const list = [_]P{
        .{ .start = 0, .end = 100, .opening = 6, .excess = 6, .high = 0.9 },
        .{ .start = 100, .end = 200, .opening = 4.5, .excess = 4.5, .high = 0.19 },
        .{ .start = 200, .end = 300, .opening = 4.1, .excess = 4.1, .high = 0.2 },
        .{ .start = 300, .end = 400, .opening = 4.3, .excess = 4.3, .high = 0.5 },
        .{ .start = 400, .end = 500, .opening = null, .excess = null, .high = null },
    };
    try std.testing.expectEqual(@as(usize, 300), drama_register.jump(&list, 4, 0.2).?.start);
    try std.testing.expectEqual(@as(usize, 200), drama_register.jump(list[0..3], 4, 0.2).?.start);
    try std.testing.expect(drama_register.jump(&list, 4.4, 0.2) == null and drama_register.jump(list[0..1], 0, 0) == null);
}

test "silence and audio too short for a frame measure nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const quiet: [44100]f32 = @splat(0);
    for ([_][]const f32{ &quiet, quiet[0..100], &.{} }) |pcm| {
        const m = try drama_register.measure(a, pcm, 44100, 210);
        try std.testing.expect(drama_register.jump(m.phrases, 4, 0.2) == null);
    }
}

test "a NaN or infinite sample in voiced audio is refused before any measure" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pcm = try tones(a, 16000, &.{ .{ 1.2, 180, 150 }, .{ 0.3, 0, 0 }, .{ 1.2, 240, 205 } }, plain);
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        pcm[8000] = bad;
        try std.testing.expectError(error.NonFinite, drama_register.measure(a, pcm, 16000, 210));
    }
}

fn number(v: json.Value) f64 {
    return v.float64().?;
}

test "private parity: a reply's trip as the numpy reference decides it, and the measure's cost (TF_REGISTER_CASES)" {
    const path = std.testing.environ.getPosix("TF_REGISTER_CASES") orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const doc = (try json.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)))).ok;
    var disagree: usize = 0;
    var seconds: f64 = 0;
    var spent: f64 = 0;
    for (doc.get("cases").?.array) |case| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, case.strField("raw").?, a, .limited(1 << 30));
        const pcm = std.mem.bytesAsSlice(f32, @as([]align(4) const u8, @alignCast(bytes)));
        const sample_rate: u32 = @intCast(case.get("rate").?.int64().?);
        const began = std.Io.Clock.awake.now(io).toNanoseconds();
        const m = try drama_register.measure(a, pcm, sample_rate, 210);
        const took = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - began)) / 1e6;
        const trip = drama_register.jump(m.phrases, 4, 0.2);
        const want = case.get("trips").?.bool;
        disagree += @intFromBool(want != (trip != null));
        seconds += @as(f64, @floatFromInt(pcm.len)) / @as(f64, @floatFromInt(sample_rate));
        spent += took;
        std.debug.print("{s} trips {} numpy {} excess {d:.2} numpy {d:.2} phrases {d} numpy {d} {d:.1} ms\n", .{
            case.strField("name").?, trip != null, want, if (trip) |t| t.excess.? else 0, case.get("excess").?.float64() orelse 0, m.phrases.len, case.get("phrases").?.int64().?, took,
        });
    }
    std.debug.print("{d} disagreements; {d:.1} s of audio measured in {d:.1} ms\n", .{ disagree, seconds, spent });
    try std.testing.expectEqual(@as(usize, 0), disagree);
}
