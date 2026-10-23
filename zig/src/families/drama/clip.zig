//! A clip's samples as FishReferenceEncodeHook hands them to the codec: WAV to FP32, channels averaged, 44.1 kHz.

const std = @import("std");

const Allocator = std.mem.Allocator;

/// DAC.sample_rate.
pub const sample_rate = 44100;
/// drama_voice.py's max_reference_audio_seconds: the hook refuses longer clips.
pub const max_seconds = 60;
pub const max_samples = max_seconds * sample_rate;
/// Every common rate up to 384 kHz fits; near-coprime rates (44056 Hz needs 121M coefficients) are refused.
const max_taps = 2048;
const max_kernel = 1 << 20;
/// KSDATAFORMAT_SUBTYPE_PCM's and _IEEE_FLOAT's GUID after the leading format tag they share.
const subtype_tail = "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71";

/// The clip's mono samples at 44.1 kHz, or null with `problem` set.
pub fn prepare(a: Allocator, bytes: []const u8, problem: *[]const u8) Allocator.Error!?[]f32 {
    const wav = try parse(a, bytes, problem) orelse return null;
    const kernel: Kernel = .init(wav.rate);
    if (kernel.taps > max_taps or kernel.new > max_kernel / kernel.taps) {
        problem.* = try std.fmt.allocPrint(a, "resampling {d} Hz to {d} Hz needs {d} phases of {d} taps; convert the clip to 48 kHz", .{ wav.rate, sample_rate, kernel.new, kernel.taps });
        return null;
    }
    // the hook measures the resampled clip, whose length the header alone gives
    const len = kernel.len(wav.frames());
    if (len == 0) {
        problem.* = "the clip holds no samples";
        return null;
    }
    if (len > max_samples) {
        const seconds = @as(f64, @floatFromInt(len)) / sample_rate;
        problem.* = try std.fmt.allocPrint(a, "reference audio lasts {d:.1} s, longer than the {d} s limit; shorten it", .{ seconds, max_seconds });
        return null;
    }
    const mono = try wav.decode(a);
    const samples = if (wav.rate == sample_rate) mono else try resample(a, mono, kernel);
    for (samples) |*s| s.* = std.math.clamp(s.*, -1.0, 1.0);
    return samples;
}

/// A PCM (8, 16, 24 or 32 bit) or IEEE float (32 or 64 bit) WAV file's format and whole sample frames.
const Wav = struct {
    float: bool,
    channels: u16,
    rate: u32,
    block: u16,
    data: []const u8 = &.{},

    fn frames(w: Wav) u32 {
        return @intCast(w.data.len / w.block);
    }

    /// The frames, channels averaged.
    fn decode(w: Wav, a: Allocator) Allocator.Error![]f32 {
        const width = w.block / w.channels;
        const out = try a.alloc(f32, w.frames());
        for (out, 0..) |*o, i| {
            var sum: f32 = 0;
            for (0..w.channels) |c| sum += sample(w.data[(i * w.channels + c) * width ..][0..width], w.float);
            o.* = sum / @as(f32, @floatFromInt(w.channels));
        }
        return out;
    }
};

/// The WAV file's format and data chunk, every chunk inside the RIFF chunk's declared size, or null with `problem` set.
fn parse(a: Allocator, bytes: []const u8, problem: *[]const u8) Allocator.Error!?Wav {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE") or std.mem.readInt(u32, bytes[4..8], .little) < 4) {
        problem.* = "the clip is not a RIFF/WAVE file";
        return null;
    }
    const riff = std.mem.readInt(u32, bytes[4..8], .little);
    if (riff > bytes.len - 8) {
        problem.* = try std.fmt.allocPrint(a, "the WAV file is truncated: its RIFF chunk declares {d} bytes, {d} are present", .{ riff, bytes.len - 8 });
        return null;
    }
    // bytes past the RIFF chunk are not the WAV's (taggers append ID3 there), so they are left unread
    const end = 8 + @as(usize, riff);
    var format: ?Wav = null;
    var data: ?[]const u8 = null;
    var at: usize = 12;
    while (at < end) {
        if (end - at < 8 or std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little) > end - at - 8) {
            problem.* = try std.fmt.allocPrint(a, "the WAV file is truncated: its chunk at byte {d} runs past the RIFF chunk's end", .{at});
            return null;
        }
        const id = bytes[at..][0..4];
        const size = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        const body = bytes[at + 8 ..][0..size];
        if (std.mem.eql(u8, id, "fmt ") and format == null) {
            format = try parseFormat(a, body, problem) orelse return null;
        } else if (std.mem.eql(u8, id, "data") and data == null) {
            if (format == null) break;
            data = body;
        }
        // odd chunks are padded to even, though a last chunk may end the RIFF chunk without its pad byte
        at += 8 + size + (size & 1);
    }
    const pcm = data orelse {
        problem.* = "the WAV file has no fmt chunk before its data chunk";
        return null;
    };
    var wav = format.?;
    if (pcm.len % wav.block != 0) {
        problem.* = try std.fmt.allocPrint(a, "the WAV data ends mid frame: {d} bytes are not whole {d}-byte frames", .{ pcm.len, wav.block });
        return null;
    }
    wav.data = pcm;
    return wav;
}

/// A fmt chunk's supported format, or null with `problem` set.
fn parseFormat(a: Allocator, body: []const u8, problem: *[]const u8) Allocator.Error!?Wav {
    if (body.len < 16) {
        problem.* = try std.fmt.allocPrint(a, "the WAV fmt chunk is {d} bytes, shorter than 16", .{body.len});
        return null;
    }
    var tag = std.mem.readInt(u16, body[0..2], .little);
    const channels = std.mem.readInt(u16, body[2..4], .little);
    const rate = std.mem.readInt(u32, body[4..8], .little);
    const block = std.mem.readInt(u16, body[12..14], .little);
    const bits = std.mem.readInt(u16, body[14..16], .little);
    if (tag == 0xFFFE) {
        const extension: u16 = if (body.len < 18) 0 else std.mem.readInt(u16, body[16..18], .little);
        if (body.len < 40 or extension < 22 or extension > body.len - 18) {
            problem.* = try std.fmt.allocPrint(a, "malformed WAVE_FORMAT_EXTENSIBLE: {d} extension bytes in a {d}-byte fmt chunk", .{ extension, body.len });
            return null;
        }
        if (!std.mem.eql(u8, body[26..40], subtype_tail)) {
            problem.* = try std.fmt.allocPrint(a, "unsupported WAV encoding: extensible subformat {x}; use PCM or IEEE float", .{body[24..40]});
            return null;
        }
        // the subformat GUID's first two bytes are the real format tag
        tag = std.mem.readInt(u16, body[24..26], .little);
        const valid = std.mem.readInt(u16, body[18..20], .little);
        if (valid == 0 or valid > bits or (tag == 3 and valid != bits)) {
            problem.* = try std.fmt.allocPrint(a, "malformed WAVE_FORMAT_EXTENSIBLE: {d} valid bits in {d}-bit samples", .{ valid, bits });
            return null;
        }
        // SPEAKER_RESERVED bits name no speaker, and a channel cannot feed two speakers
        const mask = std.mem.readInt(u32, body[20..24], .little);
        if (mask & 0x7FFC0000 != 0 or @popCount(mask) > channels) {
            problem.* = try std.fmt.allocPrint(a, "malformed WAVE_FORMAT_EXTENSIBLE: channel mask 0x{x} for {d} channels", .{ mask, channels });
            return null;
        }
    }
    const width: usize = bits / 8;
    const supported = (tag == 1 and (bits == 8 or bits == 16 or bits == 24 or bits == 32)) or (tag == 3 and (bits == 32 or bits == 64));
    if (!supported or channels == 0 or rate == 0 or block != channels * width) {
        problem.* = try std.fmt.allocPrint(a, "unsupported WAV encoding: format tag {d}, {d} bits; use PCM or IEEE float", .{ tag, bits });
        return null;
    }
    return .{ .float = tag == 3, .channels = channels, .rate = rate, .block = block };
}

/// One sample as libsndfile reads it to float: integers scaled by 2^-(bits-1) (8-bit unsigned around 128).
fn sample(b: []const u8, float: bool) f32 {
    return switch (b.len) {
        1 => @as(f32, @floatFromInt(@as(i32, b[0]) - 128)) / 128.0,
        2 => @as(f32, @floatFromInt(std.mem.readInt(i16, b[0..2], .little))) / 32768.0,
        3 => @as(f32, @floatFromInt(std.mem.readInt(i24, b[0..3], .little))) / 8388608.0,
        4 => if (float) @bitCast(std.mem.readInt(u32, b[0..4], .little)) else @as(f32, @floatFromInt(std.mem.readInt(i32, b[0..4], .little))) / 2147483648.0,
        8 => @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, b[0..8], .little)))),
        else => unreachable,
    };
}

/// torchaudio's sinc_interp_hann kernel geometry from `from` Hz to 44.1 kHz: `new` phases of `taps` coefficients.
const Kernel = struct {
    orig: usize,
    new: usize,
    base: f64,
    width: usize,
    taps: usize,

    const zeros = 6;

    fn init(from: u32) Kernel {
        const g = std.math.gcd(from, sample_rate);
        const orig: usize = from / g;
        const new: usize = sample_rate / g;
        const base: f64 = @as(f64, @floatFromInt(@min(orig, new))) * 0.99;
        const width: usize = @intFromFloat(@ceil(zeros * @as(f64, @floatFromInt(orig)) / base));
        return .{ .orig = orig, .new = new, .base = base, .width = width, .taps = 2 * width + orig };
    }

    /// The resampled length of `frames` frames; `new` <= 44100 keeps the product far inside 64 bits.
    fn len(k: Kernel, frames: u32) usize {
        return (k.new * frames + k.orig - 1) / k.orig;
    }
};

/// torchaudio.functional.resample at its defaults: the sinc_interp_hann kernel built in FP32, then the strided conv.
fn resample(a: Allocator, x: []const f32, k: Kernel) Allocator.Error![]f32 {
    const zeros = Kernel.zeros;
    const kernel = try a.alloc(f32, k.new * k.taps);
    defer a.free(kernel);
    const pi: f32 = std.math.pi;
    const scale: f32 = @floatCast(k.base / @as(f64, @floatFromInt(k.orig)));
    for (0..k.new) |j| {
        for (0..k.taps) |i| {
            const idx = @as(f32, @floatFromInt(@as(isize, @intCast(i)) - @as(isize, @intCast(k.width)))) / @as(f32, @floatFromInt(k.orig));
            var t = -@as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(k.new)) + idx;
            t = std.math.clamp(t * @as(f32, @floatCast(k.base)), -zeros, zeros);
            const c = @cos(t * pi / zeros / 2);
            const window = c * c;
            t *= pi;
            const sinc: f32 = if (t == 0) 1 else @sin(t) / t;
            kernel[j * k.taps + i] = sinc * (window * scale);
        }
    }
    const out = try a.alloc(f32, k.len(@intCast(x.len)));
    for (out, 0..) |*o, n| {
        const frame = n / k.new;
        const j = n % k.new;
        var acc: f32 = 0;
        for (kernel[j * k.taps ..][0..k.taps], 0..) |w, i| {
            const at = @as(isize, @intCast(frame * k.orig + i)) - @as(isize, @intCast(k.width));
            if (at >= 0 and at < x.len) acc += x[@intCast(at)] * w;
        }
        o.* = acc;
    }
    return out;
}

const TestFormat = struct { tag: u16 = 1, channels: u16 = 1, rate: u32 = sample_rate, bits: u16 = 16, extensible: bool = false };

/// A canonical WAV of `data`; `extensible` writes WAVE_FORMAT_EXTENSIBLE with `tag` as its subformat.
fn testWav(a: Allocator, f: TestFormat, data: []const u8) ![]u8 {
    const fmt_len: u32 = if (f.extensible) 40 else 16;
    const out = try a.alloc(u8, 28 + fmt_len + data.len);
    @memcpy(out[0..4], "RIFF");
    std.mem.writeInt(u32, out[4..8], @intCast(out.len - 8), .little);
    @memcpy(out[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, out[16..20], fmt_len, .little);
    const fmt = out[20..][0..fmt_len];
    std.mem.writeInt(u16, fmt[0..2], if (f.extensible) 0xFFFE else f.tag, .little);
    std.mem.writeInt(u16, fmt[2..4], f.channels, .little);
    std.mem.writeInt(u32, fmt[4..8], f.rate, .little);
    std.mem.writeInt(u32, fmt[8..12], @truncate(@as(u64, f.rate) * f.channels * f.bits / 8), .little);
    std.mem.writeInt(u16, fmt[12..14], f.channels * f.bits / 8, .little);
    std.mem.writeInt(u16, fmt[14..16], f.bits, .little);
    if (f.extensible) {
        std.mem.writeInt(u16, fmt[16..18], 22, .little);
        std.mem.writeInt(u16, fmt[18..20], f.bits, .little);
        std.mem.writeInt(u32, fmt[20..24], (@as(u32, 1) << @intCast(f.channels)) - 1, .little);
        std.mem.writeInt(u16, fmt[24..26], f.tag, .little);
        @memcpy(fmt[26..40], subtype_tail);
    }
    const chunk = out[20 + fmt_len ..];
    @memcpy(chunk[0..4], "data");
    std.mem.writeInt(u32, chunk[4..8], @intCast(data.len), .little);
    @memcpy(chunk[8..], data);
    return out;
}

test "WAV samples read as soundfile reads them, channels averaged, clamped" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    // stereo 16-bit: (1 + 2) / 32768 / 2 and (-32768 + 32767) / 32768 / 2
    const pcm16 = try testWav(a, .{ .channels = 2 }, &.{ 1, 0, 2, 0, 0x00, 0x80, 0xff, 0x7f });
    try std.testing.expectEqualSlices(f32, &.{ 3.0 / 65536.0, -1.0 / 65536.0 }, (try prepare(a, pcm16, &problem)).?);
    // 24-bit -2^23 is -1; 8-bit 255 is 127 / 128
    const pcm24 = try testWav(a, .{ .bits = 24 }, &.{ 0, 0, 0x80, 0xff, 0xff, 0x7f });
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 8388607.0 / 8388608.0 }, (try prepare(a, pcm24, &problem)).?);
    const pcm8 = try testWav(a, .{ .bits = 8 }, &.{ 255, 128 });
    try std.testing.expectEqualSlices(f32, &.{ 127.0 / 128.0, 0 }, (try prepare(a, pcm8, &problem)).?);
    // float beyond full scale is clamped as the hook clamps it under autocast
    const float = try testWav(a, .{ .tag = 3, .bits = 32 }, std.mem.asBytes(&[_]f32{ 1.5, -0.25 }));
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -0.25 }, (try prepare(a, float, &problem)).?);
}

test "accepted clips decode to the samples they always have" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var bytes: [1152]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @truncate(i * i * 31 + i * 7);
    var floats: [96]f32 = undefined;
    for (&floats, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast(i * 37 % 101)) - 50)) / 40.0;
    var doubles: [48]f64 = undefined;
    for (&doubles, 0..) |*v, i| v.* = @as(f64, @floatFromInt(@as(isize, @intCast(i * 53 % 89)) - 44)) / 50.0;
    const clips = [_][]u8{
        try testWav(a, .{ .channels = 2, .rate = 48000 }, &bytes),
        try testWav(a, .{ .bits = 24, .rate = 22050, .extensible = true }, &bytes),
        try testWav(a, .{ .channels = 3, .bits = 8, .rate = 16000 }, &bytes),
        try testWav(a, .{ .tag = 3, .bits = 32 }, std.mem.sliceAsBytes(&floats)),
        try testWav(a, .{ .tag = 3, .bits = 64, .channels = 2, .rate = 96000, .extensible = true }, std.mem.sliceAsBytes(&doubles)),
    };
    // the samples' SHA-256 as the reader gave them before it checked containers and resampling bounds
    var sha: std.crypto.hash.sha2.Sha256 = .init(.{});
    var problem: []const u8 = "";
    for (clips) |clip| sha.update(std.mem.sliceAsBytes((try prepare(a, clip, &problem)).?));
    try std.testing.expectEqualStrings("37ff4ef0b173ed55dc27844ad6a2e631ca6fc3d9e56745203981685640642393", &std.fmt.bytesToHex(sha.finalResult(), .lower));
}

test "unreadable and over-long clips are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    try std.testing.expect(try prepare(a, "ID3\x04 not a wav", &problem) == null);
    try std.testing.expectEqualStrings("the clip is not a RIFF/WAVE file", problem);
    const mp3 = try testWav(a, .{ .tag = 0x55 }, &.{ 0, 0 });
    try std.testing.expect(try prepare(a, mp3, &problem) == null);
    try std.testing.expectEqualStrings("unsupported WAV encoding: format tag 85, 16 bits; use PCM or IEEE float", problem);
    const silence = try a.alloc(u8, max_samples + 1);
    @memset(silence, 128);
    try std.testing.expect(try prepare(a, try testWav(a, .{ .bits = 8 }, silence), &problem) == null);
    try std.testing.expectEqualStrings("reference audio lasts 60.0 s, longer than the 60 s limit; shorten it", problem);
    const exact = try testWav(a, .{ .bits = 8 }, silence[0..max_samples]);
    try std.testing.expectEqual(@as(usize, max_samples), (try prepare(a, exact, &problem)).?.len);
}

test "resampling geometry and duration are refused from the header, before any sample is read" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const silence = try a.alloc(u8, 60_001);
    @memset(silence, 128);
    // far below one second of output: these refusals must neither decode nor resample
    var budget: [16 << 10]u8 = undefined;
    const cases = [_]struct { u32, usize, []const u8 }{
        .{ 44101, 2, "resampling 44101 Hz to 44100 Hz needs 44100 phases of 44115 taps; convert the clip to 48 kHz" },
        .{ 768000, 2, "resampling 768000 Hz to 44100 Hz needs 147 phases of 2772 taps; convert the clip to 48 kHz" },
        .{ 2003, 2, "resampling 2003 Hz to 44100 Hz needs 44100 phases of 2017 taps; convert the clip to 48 kHz" },
        .{ std.math.maxInt(u32), 2, "resampling 4294967295 Hz to 44100 Hz needs 2940 phases of 287511657 taps; convert the clip to 48 kHz" },
        .{ 48000, 0, "the clip holds no samples" },
        .{ 1000, 60_001, "reference audio lasts 60.0 s, longer than the 60 s limit; shorten it" },
    };
    var problem: []const u8 = "";
    for (cases) |c| {
        var fba: std.heap.FixedBufferAllocator = .init(&budget);
        try std.testing.expect(try prepare(fba.allocator(), try testWav(a, .{ .rate = c[0], .bits = 8 }, silence[0..c[1]]), &problem) == null);
        try std.testing.expectEqualStrings(c[2], problem);
    }
    for ([_]u32{ 1, 8000, 11025, 16000, 22050, 24000, 32000, 48000, 88200, 96000, 176400, 192000, 352800, 384000 }) |rate| {
        try std.testing.expect(try prepare(a, try testWav(a, .{ .rate = rate, .bits = 8 }, silence[0..1]), &problem) != null);
    }
}

test "truncated chunks, inconsistent RIFF sizes and partial frames are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pcm = [_]u8{ 1, 0, 2, 0, 3, 0, 4, 0 };
    // 52 bytes: the fmt chunk at 12, the data chunk at 36
    const wav = try testWav(a, .{ .channels = 2 }, &pcm);
    var problem: []const u8 = "";
    const want = (try prepare(a, wav, &problem)).?;
    const patches = [_]struct { usize, []const u8, []const u8 }{
        .{ 4, "\x2d\x00\x00\x00", "the WAV file is truncated: its RIFF chunk declares 45 bytes, 44 are present" },
        .{ 4, "\x03\x00\x00\x00", "the clip is not a RIFF/WAVE file" },
        .{ 4, "\x20\x00\x00\x00", "the WAV file is truncated: its chunk at byte 36 runs past the RIFF chunk's end" },
        .{ 16, "\xff\xff\xff\xff", "the WAV file is truncated: its chunk at byte 12 runs past the RIFF chunk's end" },
        .{ 16, "\x0e\x00\x00\x00", "the WAV fmt chunk is 14 bytes, shorter than 16" },
        .{ 40, "\x09\x00\x00\x00", "the WAV file is truncated: its chunk at byte 36 runs past the RIFF chunk's end" },
        .{ 40, "\x06\x00\x00\x00", "the WAV file is truncated: its chunk at byte 50 runs past the RIFF chunk's end" },
        .{ 32, "\x02\x00", "unsupported WAV encoding: format tag 1, 16 bits; use PCM or IEEE float" },
        .{ 36, "list", "the WAV file has no fmt chunk before its data chunk" },
    };
    for (patches) |p| {
        const bad = try a.dupe(u8, wav);
        @memcpy(bad[p[0]..][0..p[1].len], p[1]);
        try std.testing.expect(try prepare(a, bad, &problem) == null);
        try std.testing.expectEqualStrings(p[2], problem);
    }
    try std.testing.expect(try prepare(a, try testWav(a, .{ .channels = 2 }, pcm[0..6]), &problem) == null);
    try std.testing.expectEqualStrings("the WAV data ends mid frame: 6 bytes are not whole 4-byte frames", problem);
    const after = try std.mem.concat(a, u8, &.{ wav, "LIST\x09\x00\x00\x00abcd" });
    std.mem.writeInt(u32, after[4..8], @intCast(after.len - 8), .little);
    try std.testing.expect(try prepare(a, after, &problem) == null);
    try std.testing.expectEqualStrings("the WAV file is truncated: its chunk at byte 52 runs past the RIFF chunk's end", problem);
    // bytes past the RIFF chunk are left unread, an odd chunk's pad byte is skipped, and a last one's may be missing
    const tagged = try std.mem.concat(a, u8, &.{ wav, "TAG\x00\x01" });
    try std.testing.expectEqualSlices(f32, want, (try prepare(a, tagged, &problem)).?);
    const padded = try std.mem.concat(a, u8, &.{ wav[0..36], "LIST\x03\x00\x00\x00abc\x00", wav[36..] });
    std.mem.writeInt(u32, padded[4..8], @intCast(padded.len - 8), .little);
    try std.testing.expectEqualSlices(f32, want, (try prepare(a, padded, &problem)).?);
    const unpadded = try testWav(a, .{ .bits = 8 }, &.{ 0, 128, 255 });
    try std.testing.expectEqualSlices(f32, &.{ -1, 0, 127.0 / 128.0 }, (try prepare(a, unpadded, &problem)).?);
}

test "WAVE_FORMAT_EXTENSIBLE is read only whole, consistent and PCM or IEEE float" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pcm = [_]u8{ 1, 0, 2, 0, 3, 0, 4, 0 };
    var problem: []const u8 = "";
    const want = (try prepare(a, try testWav(a, .{ .channels = 2 }, &pcm), &problem)).?;
    // fmt body at 20: cbSize at 36, valid bits at 38, channel mask at 40, subformat GUID at 44
    const wav = try testWav(a, .{ .channels = 2, .extensible = true }, &pcm);
    try std.testing.expectEqualSlices(f32, want, (try prepare(a, wav, &problem)).?);
    const patches = [_]struct { usize, []const u8, []const u8 }{
        .{ 16, "\x12\x00\x00\x00", "malformed WAVE_FORMAT_EXTENSIBLE: 22 extension bytes in a 18-byte fmt chunk" },
        .{ 36, "\x00\x00", "malformed WAVE_FORMAT_EXTENSIBLE: 0 extension bytes in a 40-byte fmt chunk" },
        .{ 36, "\x17\x00", "malformed WAVE_FORMAT_EXTENSIBLE: 23 extension bytes in a 40-byte fmt chunk" },
        .{ 38, "\x00\x00", "malformed WAVE_FORMAT_EXTENSIBLE: 0 valid bits in 16-bit samples" },
        .{ 38, "\x11\x00", "malformed WAVE_FORMAT_EXTENSIBLE: 17 valid bits in 16-bit samples" },
        .{ 40, "\x07", "malformed WAVE_FORMAT_EXTENSIBLE: channel mask 0x7 for 2 channels" },
        .{ 42, "\x04", "malformed WAVE_FORMAT_EXTENSIBLE: channel mask 0x40003 for 2 channels" },
        .{ 59, "\x72", "unsupported WAV encoding: extensible subformat 0100000000001000800000aa00389b72; use PCM or IEEE float" },
        .{ 44, "\x55", "unsupported WAV encoding: format tag 85, 16 bits; use PCM or IEEE float" },
    };
    for (patches) |p| {
        const bad = try a.dupe(u8, wav);
        @memcpy(bad[p[0]..][0..p[1].len], p[1]);
        try std.testing.expect(try prepare(a, bad, &problem) == null);
        try std.testing.expectEqualStrings(p[2], problem);
    }
    // 20 valid bits ride left-justified in a 24-bit container, as libsndfile reads them; float has no narrower form
    const pcm20 = try testWav(a, .{ .bits = 24, .extensible = true }, &.{ 0x00, 0x10, 0x00 });
    pcm20[38] = 20;
    try std.testing.expectEqualSlices(f32, &.{1.0 / 2048.0}, (try prepare(a, pcm20, &problem)).?);
    const float24 = try testWav(a, .{ .tag = 3, .bits = 32, .extensible = true }, std.mem.asBytes(&@as(f32, 0.5)));
    float24[38] = 24;
    try std.testing.expect(try prepare(a, float24, &problem) == null);
    try std.testing.expectEqualStrings("malformed WAVE_FORMAT_EXTENSIBLE: 24 valid bits in 32-bit samples", problem);
}

test "every truncation and single-byte corruption is refused or read within the kernel bound" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const wav = try testWav(a, .{ .channels = 2, .extensible = true }, &.{ 1, 0, 2, 0, 3, 0, 4, 0 });
    // the largest kernel the bounds allow plus a few frames' output: no header can ask for more
    const budget = try std.testing.allocator.alloc(u8, max_kernel * @sizeOf(f32) + (1 << 20));
    defer std.testing.allocator.free(budget);
    var problem: []const u8 = "";
    for (0..wav.len) |n| {
        var fba: std.heap.FixedBufferAllocator = .init(budget);
        try std.testing.expect(try prepare(fba.allocator(), wav[0..n], &problem) == null);
    }
    const bad = try a.dupe(u8, wav);
    for (bad, wav) |*b, good| {
        for ([_]u8{ 0, 1, 0x7f, 0x80, 0xff }) |v| {
            b.* = v;
            var fba: std.heap.FixedBufferAllocator = .init(budget);
            _ = try prepare(fba.allocator(), bad, &problem);
        }
        b.* = good;
    }
}

test "resampling matches torchaudio's sinc_interp_hann kernel" {
    // torchaudio 2.11 resample of x[i] = ((37 i) mod 101 - 50) / 64, i < 200, from 48 and 22.05 kHz to 44.1 kHz
    var x: [200]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast(i * 37 % 101)) - 50)) / 64.0;
    const cases = .{
        .{ 48000, 184, [_]usize{ 0, 1, 2, 5, 50, 91, 150, 182, 183 }, [_]u32{ 0xbf467ca1, 0xbdb199cf, 0x3e6fc71f, 0x3dd1dff0, 0x3d4ac4dc, 0xbe877cf1, 0x3e1dedf1, 0x3e2774ee, 0x3f0f4c8c } },
        .{ 22050, 400, [_]usize{ 0, 1, 2, 3, 100, 201, 398, 399 }, [_]u32{ 0xbf47e3d1, 0xbf3a092a, 0xbe4efb70, 0x3eabd58c, 0xbe8a8bb7, 0xbe1fc8a7, 0x3f2411b7, 0x3ea99231 } },
    };
    inline for (cases) |c| {
        const y = try resample(std.testing.allocator, &x, .init(c[0]));
        defer std.testing.allocator.free(y);
        try std.testing.expectEqual(@as(usize, c[1]), y.len);
        // torch's vectorised sin/cos and convolution order differ from these in the last bits only
        for (c[2], c[3]) |i, bits| try std.testing.expectApproxEqAbs(@as(f32, @bitCast(bits)), y[i], 1e-6);
    }
}
