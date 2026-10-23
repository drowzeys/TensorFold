//! The engines this binary opens for a checkpoint: the backend module built in (Metal, or none for the tests).
const std = @import("std");
const api = @import("engine_api");
const native = @import("native_engines");
const cli = @import("cli.zig");
const Allocator = std.mem.Allocator;

pub const Opened = api.Opened;

/// The serve flags past the common ones the backend lists (tensor parallelism, MTP, kept entries, policy).
pub const serves: []const []const u8 = if (@hasDecl(native, "serves")) native.serves else &.{};

/// The backend's std options when it has any (HIP: no alternate signal stack for RCCL's threads); else the defaults.
pub const std_options: std.Options = @import("std_options.zig").of(native);

/// What ``capabilities --json`` reports: this release, the chip here, and what the built-in backend serves.
pub fn capabilities(a: Allocator) cli.Engines {
    return .{ .version = @import("build_options").version, .chip = if (native.families.len > 0) native.chip(a) else null, .backends = native.backends, .families = native.families };
}

/// The engine for the checkpoint in ``dir``, or null with ``problem`` set.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !?Opened {
    return native.open(a, gpa, io, try cli.request(a, dir, model_type, args), problem);
}

/// A tensor-parallel rank above 0 runs rank 0's steps until it stops; false with ``problem`` set when it cannot.
pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, model_type: []const u8, args: cli.Args, problem: *[]const u8) !bool {
    if (!@hasDecl(native, "follow")) {
        problem.* = "this build has no tensor-parallel ranks";
        return false;
    }
    return native.follow(a, gpa, io, try cli.request(a, dir, model_type, args), problem);
}

/// A clip's Drama 3 voice codes [frames][10], the codec's rate and frame, and the device, peak bytes and seconds.
pub const EncodedVoice = struct { codes: []const [10]u32, sample_rate: u32, frame_length: u32, device: []const u8, peak_device_bytes: u64, seconds: f64 };

/// ``voice encode``: the WAV clip `wav` through the backend's codec encoder, or null with ``problem`` set.
pub fn encodeVoice(a: Allocator, gpa: Allocator, io: std.Io, dir: []const u8, wav: []const u8, problem: *[]const u8) !?EncodedVoice {
    if (!@hasDecl(native, "encodeVoice")) {
        problem.* = "this build has no Drama 3 voice encoder; it runs on the CUDA backend";
        return null;
    }
    const e = try native.encodeVoice(a, gpa, io, dir, wav, problem) orelse return null;
    return .{ .codes = e.codes, .sample_rate = e.sample_rate, .frame_length = e.frame_length, .device = e.device, .peak_device_bytes = e.peak_device_bytes, .seconds = e.seconds };
}
