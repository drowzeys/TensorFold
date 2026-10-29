//! LFM3-VL for chat on CUDA: LFM3's lane engine on the bf16 release, LFM3-VL's tower laying images over the prompt.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const core = @import("core");
const lfm3 = @import("lfm3_cuda");
const vision = @import("lfm3_vision.zig");
const Allocator = std.mem.Allocator;

const N = lfm3.native;
const Liquid = lfm3.backend.Liquid;
const Tower = @typeInfo(@FieldType(Liquid, "tower")).optional.child;

pub const model_type = "lfm3_vl";
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const explains_open = true;
/// A camera frame of 640 x 480 is 300 tokens; the tower holds a picture of 1,600 (1280 x 1280) at most.
pub const default_image_tokens: u32 = vision.patches_most / 4;
pub const prompt_rows = N.prompt_rows;
pub const snap_lookahead = N.snap_lookahead;
pub const window_rows = N.window_rows;
pub const Room = N.Room;
pub const snap_bytes = N.snap_bytes;
pub const snap_ready = N.snap_ready;
pub const snap_charged = N.snap_charged;
pub const snap_spare = N.snap_spare;
pub const snap_reuses = N.snap_reuses;
pub const snap_save = N.snap_save;
pub const snap_restore = N.snap_restore;
pub const snap_drop = N.snap_drop;

/// --vision's limits as the server set them (cuda_options.zig).
pub const Limits = struct { image_tokens: u32, max_images: u32 };
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null, vision: ?Limits = null };

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *Liquid,
    layout: @FieldType(N.Loaded, "layout") = null,
    engine_paged: bool = true,
    vision: ?*Eyes = null,
};

/// What the server's image requests reach: LFM3-VL's processor (its smart_resize, no cap) and the tower's rows.
pub const Eyes = struct {
    /// The template writes <|vision_start|><|image|><|vision_end|> for an "image" part itself.
    pub const template_wraps = true;
    pub const part_type = "image";
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    w: vision.Weights,
    tower: vision.Vision,
    marks: struct { open: u32 = vision.vision_start, image: u32 = vision.image_token, close: u32 = vision.vision_end } = .{},
    mark_ids: [3]u32 = .{ vision.vision_start, vision.image_token, vision.vision_end },
    limits: Limits,

    /// An image's bytes as the tower's patch rows (f32, exactly bf16), at most `max_tokens` tokens.
    pub fn prepare(e: *Eyes, a: Allocator, bytes: []const u8, max_tokens: u32) !api.PreparedImage {
        const rgb = try core.picture.decode(e.io, a, bytes);
        defer a.free(rgb.pixels);
        var p = try vision.prepareUnder(a, rgb, null);
        defer vision.free(a, &p);
        if (p.tokens() > max_tokens) return error.PictureTooLarge;
        const id = core.picture.identity;
        const hash = id.withGrid(id.ofBytes(tag, bytes), .{ @intCast(p.gh), @intCast(p.gw) });
        return .{ .pixels = try vision.widen(a, &p), .gh = @intCast(p.gh), .gw = @intCast(p.gw), .tokens = @intCast(p.tokens()), .hash = hash };
    }

    pub fn explain(err: anyerror) ?[]const u8 {
        if (err == error.PictureTooLarge) return "the image is past LFM3-VL's 1,600 tokens a picture on CUDA (1280 x 1280): send a smaller one";
        return core.picture.decoderProblem(err);
    }

    /// The lane's hook: an image's projected rows, bf16 [tokens, 2048] at `out`.
    fn rows(ctx: *anyopaque, s: cuda.Stream, img: *const lanes.stream.Image, out: u64) anyerror!void {
        const e: *Eyes = @ptrCast(@alignCast(ctx));
        var p = try vision.fromRows(e.gpa, img.pixels, img.gh, img.gw);
        defer vision.free(e.gpa, &p);
        if (p.tokens() != img.tokens) return error.BadPictureRows;
        _ = try e.tower.features(s, &p, out);
        try s.synchronize(); // the host rows are freed as this returns
    }

    fn deinit(e: *Eyes) void {
        e.tower.deinit();
        e.w.deinit(e.gpa);
        e.gpa.destroy(e);
    }
};

/// The image identity's family tag ("lfm3vl").
const tag: u64 = 0x6c_66_6d_33_76_6c;

/// The lane engine's state and the tower's, behind one pointer.
const Own = struct { gpa: Allocator, inner: *anyopaque, close: *const fn (*anyopaque) void, eyes: ?*Eyes };

pub fn open(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    const in = try N.open(gpa, io, ctx, dir, kernels, .{ .context = o.context, .drafts = o.drafts, .segments = o.segments, .slide = o.slide, .lanes = o.lanes, .lanes_fixed = o.lanes_fixed, .room = o.room, .prompt_gemm = true });
    errdefer in.deinit(in.ctx);
    if (in.target.m.w.config.kind != .vision_language) return error.Lfm3VlTextOnly;
    const own = try gpa.create(Own);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .inner = in.ctx, .close = in.deinit, .eyes = null };
    if (o.vision) |lim| {
        const e = try gpa.create(Eyes);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .io = io, .d = ctx.d, .w = undefined, .tower = undefined, .limits = .{ .image_tokens = lim.image_tokens, .max_images = @min(lim.max_images, 16) } };
        e.w = try vision.load(gpa, io, ctx.d, dir);
        errdefer e.w.deinit(gpa);
        e.tower = try vision.Vision.init(gpa, ctx.d, &e.w);
        own.eyes = e;
        in.target.tower = Tower{ .ctx = e, .rows = Eyes.rows };
        std.log.info("LFM3-VL on CUDA: chat from the bf16 release, images through its tower (up to {d} a request)", .{e.limits.max_images});
    }
    return .{ .backend = in.backend, .facts = in.facts, .rows = in.rows, .stream_bytes = in.stream_bytes, .free_streams = in.free_streams, .ctx = own, .deinit = release, .target = in.target, .layout = in.layout, .engine_paged = in.engine_paged, .vision = own.eyes };
}

fn release(p: *anyopaque) void {
    const own: *Own = @ptrCast(@alignCast(p));
    own.close(own.inner);
    if (own.eyes) |e| e.deinit();
    own.gpa.destroy(own);
}

pub fn attach(p: *anyopaque, streams: u32, n: usize) !void {
    const own: *Own = @ptrCast(@alignCast(p));
    return N.attach(own.inner, streams, n);
}

pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *const Own = @ptrCast(@alignCast(p));
    return N.fits(own.inner, prompt, max_tokens);
}

pub fn explain(ctx: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Lfm3VlTextOnly => "this lfm3_vl checkpoint has no LFM3-VL vision config",
        error.Lfm3TextOnly => "this request has images: serve LFM3-VL with --vision to read them",
        error.TooManyPictures => "LFM3-VL on CUDA reads at most 16 images a request",
        error.UnsupportedLfm3Quantization => "LFM3-VL on CUDA reads the bf16 release (LiquidAI/LFM3-VL-2B)",
        else => N.explain(ctx, err),
    };
}

test "the eyes' words name their refusals, the rest are LFM3's" {
    try std.testing.expect(explain(null, error.Lfm3TextOnly) != null);
    try std.testing.expect(explain(null, error.PastContext) != null);
    try std.testing.expect(Eyes.explain(error.PictureTooLarge) != null);
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}
