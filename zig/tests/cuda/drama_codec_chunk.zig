//! GPU-only equality gate against the original absolute-row codec, including poisoned chunk scratch.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");

const frames = 305;
const samples = frames * 2048;

fn equal(expected: []const f32, actual: []const f32) !void {
    for (actual) |v| if (!std.math.isFinite(v)) return error.NonFiniteAudio;
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(actual))) return error.PcmBytesDiffer;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) {
        std.debug.print("usage: tf-drama-codec-chunk CHECKPOINT\n", .{});
        return 2;
    }
    var codes: [frames][10]u32 = undefined;
    for (&codes, 0..) |*row, f| {
        for (row, 0..) |*code, book| code.* = @intCast((f * 73 + book * 131 + f * f) % (if (book == 0) @as(usize, 4096) else 1024));
    }
    const expected = try init.gpa.alloc(f32, samples);
    defer init.gpa.free(expected);
    const actual = try init.gpa.alloc(f32, samples);
    defer init.gpa.free(actual);
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var reference = try drama.codec.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .max_frames = frames, .full_plane_reference = true });
    defer reference.deinit();
    var chunk = try drama.codec.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .max_frames = frames });
    defer chunk.deinit();
    _ = try reference.decode(stream, &codes, expected);
    _ = try chunk.decode(stream, &codes, actual);
    try equal(expected, actual);
    // Each utterance starts from poisoned QKV, carry and PCM: nothing may leak from the last one.
    for ([_]usize{ 1, 4, 16, 127, 300 }) |cadence| {
        chunk.begin();
        for ([_]cuda.DeviceBuffer{ chunk.kv, chunk.carry, chunk.audio_dev }) |b| try b.fill8(0xff, stream.handle);
        @memset(actual, std.math.nan(f32));
        var end: usize = 0;
        while (end < frames) {
            end = @min(frames, end + if (end == 0 and cadence != 16) 1 else cadence);
            for (chunk.acts) |b| try b.fill8(0xff, stream.handle);
            _ = try chunk.advance(stream, codes[0..end], actual);
            try equal(expected[0 .. end * 2048], actual[0 .. end * 2048]);
        }
        std.debug.print("frames={d} cadence={d}: chunk advance PCM bytes equal the full-plane decode\n", .{ frames, cadence });
    }
    return 0;
}
