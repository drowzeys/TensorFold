//! What a Liquid lane adds for an image tower (LFM3-VL's) and for logprobs: picture rows, and rows read on the host.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");

pub const Image = lanes.stream.Image;
const Row = lanes.LogprobRow;
const max_top = lanes.logprob.max_top;

/// Images a prompt takes: a chunk's inserts are one array of this many.
pub const max_pictures = 16;

/// A family's image tower: an image's projected rows, bf16 [image.tokens, width] at `out`, queued on `s`.
pub const Tower = struct {
    ctx: *anyopaque,
    rows: *const fn (ctx: *anyopaque, s: cuda.Stream, image: *const Image, out: u64) anyerror!void,
};

/// An image's rows in a prompt: `count` from position `at`, the first at feature row `row`.
pub const Run = struct { at: usize, count: usize, row: usize };

/// One insert of a chunk: `count` rows from `src` over the embedding from chunk row `at`.
pub const Part = struct { at: usize, src: u64, count: usize };

/// A stream's pictures' rows on the device, made when its prompt pass starts and freed when the pass ends.
pub const Rows = struct {
    buf: ?cuda.DeviceBuffer = null,
    runs: std.ArrayList(Run) = .empty,

    /// The rows of every image not wholly before `from` (where a kept state ends), `width` bf16 values each.
    pub fn make(r: *Rows, gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, t: Tower, images: []const Image, from: usize, len: usize, width: usize) !void {
        r.free(gpa);
        if (images.len > max_pictures) return error.TooManyPictures;
        var total: usize = 0;
        for (images) |img| {
            if (img.tokens == 0 or img.at + img.tokens > len) return error.BadImageSpan;
            if (img.at + img.tokens > from) total += img.tokens;
        }
        if (total == 0) return;
        r.buf = try cuda.DeviceBuffer.alloc(d, total * width * 2);
        var next: usize = 0;
        for (images) |*img| {
            if (img.at + img.tokens <= from) continue;
            try t.rows(t.ctx, s, img, r.buf.?.ptr + next * width * 2);
            try r.runs.append(gpa, .{ .at = img.at, .count = img.tokens, .row = next });
            next += img.tokens;
        }
    }

    /// The runs' parts inside the chunk [at, end), in `out`.
    pub fn within(r: *const Rows, at: usize, end: usize, width: usize, out: *[max_pictures]Part) []Part {
        var k: usize = 0;
        for (r.runs.items) |run| {
            const lo = @max(run.at, at);
            const hi = @min(run.at + run.count, end);
            if (lo >= hi) continue;
            out[k] = .{ .at = lo - at, .src = r.buf.?.ptr + (run.row + lo - run.at) * width * 2, .count = hi - lo };
            k += 1;
        }
        return out[0..k];
    }

    pub fn free(r: *Rows, gpa: std.mem.Allocator) void {
        if (r.buf) |*b| b.free();
        r.buf = null;
        r.runs.clearAndFree(gpa);
    }
};

fn value(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The row's best token, the lower id on a tie.
pub fn argmax(logits: []const u16) u32 {
    var best: usize = 0;
    for (logits, 0..) |b, i| if (value(b) > value(logits[best])) {
        best = i;
    };
    return @intCast(best);
}

/// A logprob row from one bf16 logits row: `pick`'s, the `k` best (logit descending, ties by the lower id), the rank.
pub fn row(logits: []const u16, pick: u32, k: u8) Row {
    std.debug.assert(k <= max_top and pick < logits.len);
    var most: f32 = -std.math.inf(f32);
    for (logits) |b| most = @max(most, value(b));
    var sum: f64 = 0;
    for (logits) |b| sum += @exp(@as(f64, value(b)) - most);
    const lse: f64 = most + @log(sum);
    const at = value(logits[pick]);
    var r: Row = .{ .token = pick, .logprob = @floatCast(at - lse), .count = k };
    var best: [max_top]f32 = undefined;
    var n: usize = 0;
    var rank: u32 = 0;
    for (logits, 0..) |b, i| {
        const v = value(b);
        if (v >= at) rank += 1;
        if (k == 0 or (n == k and v <= best[k - 1])) continue;
        var j: usize = @min(n, k - 1);
        while (j > 0 and v > best[j - 1]) : (j -= 1) {
            best[j] = best[j - 1];
            r.ids[j] = r.ids[j - 1];
        }
        best[j] = v;
        r.ids[j] = @intCast(i);
        n = @min(n + 1, k);
    }
    for (0..k) |i| r.logprobs[i] = @floatCast(best[i] - lse);
    r.rank = rank;
    return r;
}

fn bf(x: f32) u16 {
    return @intCast(@as(u32, @bitCast(x)) >> 16);
}

test "a row names the pick's logprob, the best ids with ties by the lower id, and the rank" {
    const logits = [_]u16{ bf(1), bf(3), bf(2), bf(3), bf(-1) };
    const r = row(&logits, 2, 3);
    var sum: f64 = 0;
    for (logits) |b| sum += @exp(@as(f64, value(b)));
    const lse: f32 = @floatCast(@log(sum));
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 2 }, r.ids[0..3]);
    try std.testing.expectApproxEqAbs(@as(f32, 2) - lse, r.logprob, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3) - lse, r.logprobs[0], 1e-6);
    try std.testing.expectEqual(@as(u32, 3), r.rank);
    try std.testing.expectEqual(@as(u32, 1), argmax(&logits));
    try std.testing.expectEqual(@as(u8, 0), row(&logits, 4, 0).count);
}

test "a chunk takes each image's part inside it, from the image's own rows" {
    var r: Rows = .{};
    defer r.runs.deinit(std.testing.allocator);
    r.buf = .{ .d = undefined, .ptr = 1 << 20, .len = 0, .borrowed = true };
    try r.runs.append(std.testing.allocator, .{ .at = 10, .count = 300, .row = 0 });
    try r.runs.append(std.testing.allocator, .{ .at = 400, .count = 100, .row = 300 });
    var out: [max_pictures]Part = undefined;
    const p = r.within(256, 512, 4, &out);
    try std.testing.expectEqual(@as(usize, 2), p.len);
    try std.testing.expectEqual(Part{ .at = 0, .src = (1 << 20) + 246 * 8, .count = 54 }, p[0]);
    try std.testing.expectEqual(Part{ .at = 144, .src = (1 << 20) + 300 * 8, .count = 100 }, p[1]);
    try std.testing.expectEqual(@as(usize, 0), r.within(600, 700, 4, &out).len);
}
