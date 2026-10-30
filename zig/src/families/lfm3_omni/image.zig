//! LFM3-Omni's pictures on the host: Lfm3VlImageProcessor as the checkpoint's processor_config.json sets it (the
//! LFM3-VL processor with Omni's bounds, 65,536 to 2,097,152 pixels): smart_resize to multiples of 32, torch's
//! antialiased uint8 bicubic, (x - 127.5) / 127.5 in fp32 (the fused rescale and normalize), and the patches in merge
//! order, each channel's 16 x 16 patch twice (the temporal pair). The tower takes them as bf16.
const std = @import("std");
const core = @import("core");
const picture = core.picture;
const geometry = core.qwen_vl.geometry;

pub const patch = 16;
pub const merge = 2;
pub const temporal = 2;
pub const patch_width = 3 * temporal * patch * patch; // 1536
pub const min_pixels = 65536;
pub const max_pixels = 2097152;
/// The most tokens one picture makes at the processor's bounds (2,097,152 pixels / 32^2).
pub const max_tokens = max_pixels / (patch * merge * patch * merge);

/// One picture ready for the tower: fp32 patches [gh * gw, 1536] in merge order, the grid, its language tokens.
pub const Prepared = struct {
    pixels: []f32,
    gh: u32,
    gw: u32,

    pub fn patches(p: Prepared) usize {
        return @as(usize, p.gh) * p.gw;
    }

    pub fn tokens(p: Prepared) usize {
        return p.patches() / (merge * merge);
    }

    pub fn deinit(p: *Prepared, a: std.mem.Allocator) void {
        a.free(p.pixels);
    }
};

/// A PNG, JPEG or WebP file's picture (the shared decoder; JPEG and WebP need their pinned libraries).
pub fn prepare(io: std.Io, a: std.mem.Allocator, bytes: []const u8) !Prepared {
    var rgb = try picture.decode(io, a, bytes);
    defer rgb.deinit(a);
    return prepareRgb(a, rgb);
}

pub fn prepareRgb(a: std.mem.Allocator, rgb: picture.Rgb) !Prepared {
    if (rgb.width == 0 or rgb.height == 0) return error.EmptyImage;
    const size = try geometry.smartResize(@intCast(rgb.height), @intCast(rgb.width), patch * merge, min_pixels, max_pixels);
    // torch's bicubic at the same size is the identity: a picture already on the grid skips it
    const same = size.w == rgb.width and size.h == rgb.height;
    const resized = if (same) rgb.pixels else try picture.resample.resize(a, .torch, .of(rgb.pixels, rgb.width, rgb.height), size.w, size.h);
    defer if (!same) a.free(resized);
    return .{ .pixels = try patchify(a, .{ .pixels = resized, .width = size.w, .height = size.h }), .gh = size.h / patch, .gw = size.w / patch };
}

/// An RGB picture whose sides are multiples of 32 as merge-order patches, normalized as the processor does.
pub fn patchify(a: std.mem.Allocator, rgb: picture.Rgb) ![]f32 {
    if (rgb.width % (patch * merge) != 0 or rgb.height % (patch * merge) != 0 or rgb.width == 0 or rgb.height == 0)
        return error.BadPatchGrid;
    const gh = rgb.height / patch;
    const gw = rgb.width / patch;
    const pixels = try a.alloc(f32, gh * gw * patch_width);
    // transformers fuses rescale and normalize: mean and std over the rescale factor, (x - 127.5) / 127.5 in fp32
    var lut: [256]f32 = undefined;
    for (&lut, 0..) |*value, i| value.* = (@as(f32, @floatFromInt(i)) - 127.5) / 127.5;
    var row: usize = 0;
    for (0..gh / merge) |by| for (0..gw / merge) |bx| for (0..merge) |dy| for (0..merge) |dx| {
        const dst = pixels[row * patch_width ..][0..patch_width];
        for (0..3) |channel| for (0..temporal) |frame| for (0..patch) |y| for (0..patch) |x| {
            const iy = (by * merge + dy) * patch + y;
            const ix = (bx * merge + dx) * patch + x;
            dst[((channel * temporal + frame) * patch + y) * patch + x] = lut[rgb.pixels[(iy * rgb.width + ix) * 3 + channel]];
        };
        row += 1;
    };
    return pixels;
}

/// Words for a refusal of a picture.
pub fn explain(err: anyerror) []const u8 {
    if (geometry.problem(err)) |p| return p;
    return switch (err) {
        error.Invalid => "the picture is not a PNG, JPEG or WebP file",
        error.Oversized => "the picture is larger than the decoder takes",
        error.EmptyImage => "the picture is empty",
        else => "the picture could not be read",
    };
}

test "patches keep merge order, channel order and the repeated temporal frame" {
    var pixels: [32 * 64 * 3]u8 = undefined;
    for (0..32) |y| for (0..64) |x| {
        pixels[(y * 64 + x) * 3] = @intCast(y / 16 * 4 + x / 16);
        pixels[(y * 64 + x) * 3 + 1] = 255;
        pixels[(y * 64 + x) * 3 + 2] = 0;
    };
    const out = try patchify(std.testing.allocator, .{ .pixels = &pixels, .width = 64, .height = 32 });
    defer std.testing.allocator.free(out);
    const order = [_]u8{ 0, 1, 4, 5, 2, 3, 6, 7 };
    for (order, 0..) |red, row| {
        const expected = (@as(f32, @floatFromInt(red)) - 127.5) / 127.5;
        try std.testing.expectEqual(expected, out[row * patch_width]);
        try std.testing.expectEqual(expected, out[row * patch_width + 256]);
        try std.testing.expectEqual(@as(f32, 1), out[row * patch_width + 512]);
        try std.testing.expectEqual(@as(f32, -1), out[row * patch_width + 1024]);
    }
}

test "a camera frame of 640 x 480 is already on the grid: 40 x 30 patches, 300 tokens" {
    const size = try geometry.smartResize(480, 640, 32, min_pixels, max_pixels);
    try std.testing.expectEqual(@as(u32, 640), size.w);
    try std.testing.expectEqual(@as(u32, 480), size.h);
    const big = try geometry.smartResize(3000, 4000, 32, min_pixels, max_pixels);
    try std.testing.expect(@as(u64, big.w) * big.h <= max_pixels);
}
