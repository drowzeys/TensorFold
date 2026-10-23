//! Classifier-free guidance on Drama 3's semantic logits: l_c + (γ - 1)(l_c - l_u), l_u from a second sequence.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const drama = @import("drama");
const Allocator = std.mem.Allocator;

const Frame = drama.tower.Frame;
/// Semantic logits a backbone row carries: EOS first, then the 4096 codes.
const cols = drama.backbone.weights.head_cols;

/// --cfg-every's largest: the frames the unconditioned sequence may hold back, staged in one tower pass.
pub const max_every = 16;

/// `cond` guided away from `uncond` by `gamma`, in place; written so that γ = 1 gives l_c's values whatever l_u is.
pub fn mix(cond: []f32, uncond: []const f32, gamma: f32) void {
    const w = gamma - 1;
    for (cond, uncond) |*c, u| c.* += w * (c.* - u);
}

/// Whether an engine with guidance `on` admits `r`'s: a plain prompt (text rows only) within `max_prompt`, finite γ.
pub fn admits(r: *const api.SpeechRequest, on: bool, max_prompt: u32, semantic_start: u32, semantic_end: u32) bool {
    if (!on or r.unconditioned.len > max_prompt or !std.math.isFinite(r.cfg_gamma)) return false;
    for (r.unconditioned) |t| if (t >= drama.backbone.config.vocab or (t >= semantic_start and t <= semantic_end)) return false;
    return true;
}

/// The sampled frames the unconditioned sequence has not read yet.
pub const Lag = struct {
    every: usize,
    held: [max_every]Frame = undefined,
    count: usize = 0,

    /// Holds `frame`; once `every` are held, all of them, oldest first, for one run (valid until the next push).
    pub fn push(l: *Lag, frame: Frame) ?[]const Frame {
        l.held[l.count] = frame;
        l.count += 1;
        if (l.count < l.every) return null;
        const n = l.count;
        l.count = 0;
        return l.held[0..n];
    }
};

/// A guided synthesis's state, worker-thread only; `h` below is the host: backbone, tower, codec, stream, now, check.
pub const Guide = struct {
    sequence: drama.backbone.Sequence,
    guided: cuda.DeviceBuffer, // f32 [4097]: the logits the tower draws the main token from
    lag: Lag,
    bytes: usize, // what the sequence and the buffer took on the device
    batched: bool, // --cfg-batched: each frame's two runs are one batch, which writes l_u into `guided` first
    gamma: f32 = 1,
    cond: [cols]f32 = undefined,
    uncond: [cols]f32 = undefined,
    tally: Tally = .{},
    words: [128]u8 = undefined,

    /// One synthesis's guidance: the plain prompt's rows, the later runs and their rows, the time all of it took.
    const Tally = struct { prompt: usize = 0, runs: usize = 0, rows: usize = 0, ns: i96 = 0 };

    /// A sequence of `b`'s context and the guided logits' buffer, the sequence to run every `every` frames.
    pub fn create(gpa: Allocator, b: *drama.backbone.Model, every: u32, batched: bool) !*Guide {
        if (every == 0 or every > max_every or (batched and every != 1)) return error.BadDramaGuidance;
        if (batched) try b.loadPair();
        const held = cuda.usage(false).device;
        const g = try gpa.create(Guide);
        errdefer gpa.destroy(g);
        var sequence = try b.sequence();
        errdefer sequence.free();
        const guided = try cuda.DeviceBuffer.alloc(b.d, cols * 4);
        g.* = .{ .sequence = sequence, .guided = guided, .lag = .{ .every = every }, .bytes = cuda.usage(false).device -| held, .batched = batched };
        return g;
    }

    pub fn destroy(g: *Guide, gpa: Allocator) void {
        g.guided.free();
        g.sequence.free();
        gpa.destroy(g);
    }

    /// The startup line's words: the device memory guidance took, --cfg-every and --cfg-batched.
    pub fn describe(g: *Guide) []const u8 {
        const gib = @as(f64, @floatFromInt(g.bytes)) / (1 << 30);
        return std.fmt.bufPrint(&g.words, "; guidance on: a second backbone sequence of {d:.3} GiB, run every {d} frames{s}", .{ gib, g.lag.every, if (g.batched) " in one batch with the first" else "" }) catch unreachable;
    }

    /// Prefills `r`'s plain prompt and keeps its last logits: first, as every run writes the rows the tower reads.
    pub fn prefill(g: *Guide, h: anytype, r: *const api.SpeechRequest) !*Guide {
        const began = h.now();
        try g.sequence.reset(h.stream);
        g.lag.count = 0;
        g.gamma = r.cfg_gamma;
        g.tally = .{ .prompt = r.unconditioned.len };
        const step = h.backbone.options.max_rows;
        var last: drama.backbone.Rows = undefined;
        var start: usize = 0;
        while (start < r.unconditioned.len) : (start += step) {
            try h.check();
            // text rows only, which take no codebook-1 embedding
            last = try h.backbone.runOn(&g.sequence, h.stream, .prefill, r.unconditioned[start..@min(start + step, r.unconditioned.len)], null);
        }
        try last.semantic_logits.download(0, std.mem.sliceAsBytes(&g.uncond));
        g.tally.ns = h.now() - began;
        return g;
    }

    /// The conditioned decode of `frame` (one), the unconditioned sequence reading it too: in one batch if `batched`.
    pub fn decode(g: *Guide, h: anytype, frame: []const Frame) !drama.backbone.Rows {
        if (!g.batched) return h.backbone.run(h.stream, .decode, frame[0][0..1], try g.follow(h, frame));
        const rows = try h.backbone.decodePair(&g.sequence, h.stream, frame[0][0], try h.tower.setInput(h.stream, frame), g.guided);
        const began = h.now();
        try g.guided.download(0, std.mem.sliceAsBytes(&g.uncond));
        g.tally.runs += 1;
        g.tally.rows += 1;
        g.tally.ns += h.now() - began;
        return rows;
    }

    /// Stages `frame` (one) for the conditioned run, the unconditioned sequence run first once it is due; the staging.
    fn follow(g: *Guide, h: anytype, frame: []const Frame) !?cuda.DeviceBuffer {
        const due = g.lag.push(frame[0]) orelse return h.tower.setInput(h.stream, frame);
        if (due.len == 1) {
            const extra = try h.tower.setInput(h.stream, frame);
            try g.run(h, due, extra);
            return extra;
        }
        try g.run(h, due, try h.tower.setInput(h.stream, due));
        return h.tower.setInput(h.stream, frame);
    }

    /// The unconditioned sequence reads `frames`, their codebook-1 rows staged as `extra`; its logits are kept.
    fn run(g: *Guide, h: anytype, frames: []const Frame, extra: ?cuda.DeviceBuffer) !void {
        const began = h.now();
        var tokens: [max_every]u32 = undefined;
        for (frames, tokens[0..frames.len]) |frame, *token| token.* = frame[0];
        // frames held back for --cfg-every go in as one prompt chunk
        const phase: drama.backbone.Phase = if (frames.len == 1) .decode else .prefill;
        const rows = try h.backbone.runOn(&g.sequence, h.stream, phase, tokens[0..frames.len], extra);
        try rows.semantic_logits.download(0, std.mem.sliceAsBytes(&g.uncond));
        g.tally.runs += 1;
        g.tally.rows += frames.len;
        g.tally.ns += h.now() - began;
    }

    /// The conditioned `logits` guided away from the unconditioned sequence's latest, uploaded for the tower's draw.
    pub fn mixed(g: *Guide, h: anytype, logits: cuda.DeviceBuffer) !cuda.DeviceBuffer {
        const began = h.now();
        try logits.download(0, std.mem.sliceAsBytes(&g.cond));
        mix(&g.cond, &g.uncond, g.gamma);
        try g.guided.upload(0, std.mem.sliceAsBytes(&g.cond));
        g.tally.ns += h.now() - began;
        return g.guided;
    }

    /// A synthesis's RTF line with guidance on, guided or not: its frames and time, and what guidance took of it.
    pub fn note(g: *const Guide, h: anytype, id: api.Id, guided: bool, frames: usize, began: i96) void {
        const t: Tally = if (guided) g.tally else .{};
        const seconds = @as(f64, @floatFromInt(h.now() - began)) / 1e9;
        const audio = @as(f64, @floatFromInt(frames * h.codec.frame_length)) / @as(f64, @floatFromInt(h.codec.sample_rate));
        const rtf = if (frames > 0) seconds / audio else 0;
        std.log.info("drama: synthesis {d} cfg_gamma={d:.2} cfg_every={d} cfg_batched={} frames={d} audio_s={d:.3} seconds={d:.3} rtf={d:.3} unconditioned_prompt={d} unconditioned_runs={d} unconditioned_rows={d} guidance_s={d:.3}", .{
            id, if (guided) g.gamma else 1, g.lag.every, g.batched, frames, audio, seconds, rtf, t.prompt, t.runs, t.rows, @as(f64, @floatFromInt(t.ns)) / 1e9,
        });
    }
};

test "at gamma 1 the guided logits are the conditioned ones, whatever the unconditioned ones are" {
    const cond = [_]f32{ 3.5, -2.25, 0, -0.0, 17, -40.125, 1e-30, 0.1 };
    const uncond = [_]f32{ -1, 8, 0.5, -3, 17, 2, -1e30, 0.3 };
    var guided = cond;
    mix(&guided, &uncond, 1);
    for (cond, guided) |c, g| try std.testing.expect(c == g);
}

test "past gamma 1 the logits move away from the unconditioned ones: gamma 2 is 2 l_c - l_u" {
    const cond = [_]f32{ 1, 2, -3, 0 };
    const uncond = [_]f32{ 1, 0, -1, 4 };
    var guided = cond;
    mix(&guided, &uncond, 2);
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, -5, -4 }, &guided);
    guided = cond;
    mix(&guided, &uncond, 1.5);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, -4, -2 }, &guided);
}

test "the unconditioned sequence reads every frame once, in order, every K at a time" {
    for ([_]usize{ 1, 3, max_every }) |every| {
        var lag: Lag = .{ .every = every };
        var read: std.ArrayList(u32) = .empty;
        defer read.deinit(std.testing.allocator);
        var runs: usize = 0;
        for (0..40) |i| {
            var frame: Frame = @splat(0);
            frame[0] = @intCast(i);
            const due = lag.push(frame) orelse continue;
            try std.testing.expectEqual(every, due.len);
            for (due) |f| try read.append(std.testing.allocator, f[0]);
            runs += 1;
        }
        try std.testing.expectEqual(40 / every, runs);
        for (read.items, 0..) |token, i| try std.testing.expectEqual(@as(u32, @intCast(i)), token);
        try std.testing.expectEqual(40 % every, lag.count);
    }
}

test "a guided request is admitted only with guidance on, its plain prompt of text rows within the prompt limit, γ finite" {
    var result: api.SpeechResult = .{};
    const start = 248099;
    const end = 252194;
    var r: api.SpeechRequest = .{ .prompt = &.{ 1, start + 7 }, .pcm = &.{}, .result = &result, .seed = 0, .unconditioned = &.{ 1, 2, 3 }, .cfg_gamma = 2 };
    try std.testing.expect(admits(&r, true, 4096, start, end));
    try std.testing.expect(!admits(&r, false, 4096, start, end));
    try std.testing.expect(!admits(&r, true, 2, start, end));
    // --cfg-force-guided's: run at γ 1, where the mix leaves l_c as it is
    r.cfg_gamma = 1;
    try std.testing.expect(admits(&r, true, 4096, start, end));
    r.unconditioned = &.{ 1, start + 7 };
    try std.testing.expect(!admits(&r, true, 4096, start, end));
    r.unconditioned = &.{ 1, drama.backbone.config.vocab };
    try std.testing.expect(!admits(&r, true, 4096, start, end));
    r.unconditioned = &.{ 1, 2 };
    r.cfg_gamma = std.math.nan(f32);
    try std.testing.expect(!admits(&r, true, 4096, start, end));
}
