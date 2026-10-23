//! `tensorfold-native voice encode`'s CUDA half: a WAV clip to Drama 3 voice codes with the native codec encoder.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const Allocator = std.mem.Allocator;

pub const Encoded = struct {
    /// [frames][10], frame-major.
    codes: []const [10]u32,
    sample_rate: u32 = drama.codec_encode.clip.sample_rate,
    frame_length: u32 = drama.codec_encode.samples_per_frame,
    device: []const u8,
    /// The DeviceBuffers' peak: weights and the clip's activations (the driver's own context memory is not counted).
    peak_device_bytes: u64,
    /// The encode alone, after the weights load.
    seconds: f64,
};

/// A WAV clip's codes on CUDA device `device` with the codec in `checkpoint_dir`; null with `problem` when refused.
pub fn encode(a: Allocator, gpa: Allocator, io: std.Io, checkpoint_dir: []const u8, wav: []const u8, device: u32, problem: *[]const u8) !?Encoded {
    const samples = try drama.codec_encode.clip.prepare(a, wav, problem) orelse return null;
    var driver = try cuda.Driver.open();
    defer driver.close();
    if (device >= try driver.deviceCount()) {
        problem.* = try std.fmt.allocPrint(a, "no CUDA device {d}", .{device});
        return null;
    }
    var ctx = try cuda.Context.init(&driver, @intCast(device));
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var model = try drama.codec_encode.Model.init(gpa, io, &ctx, checkpoint_dir);
    defer model.deinit();
    const start = std.Io.Clock.awake.now(io).toNanoseconds();
    const codes = try model.encode(a, stream, samples);
    const elapsed = std.Io.Clock.awake.now(io).toNanoseconds() - start;
    var name: [256]u8 = undefined;
    return .{
        .codes = codes,
        .device = try a.dupe(u8, ctx.name(&name) catch "GPU"),
        .peak_device_bytes = cuda.usage(false).peak,
        .seconds = @as(f64, @floatFromInt(elapsed)) / 1e9,
    };
}
