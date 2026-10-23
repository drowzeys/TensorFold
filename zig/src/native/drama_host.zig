//! Drama 3 speech behind the native server: one synthesis at a time on a single worker thread that owns the CUDA
//! context, stream, text backbone, codebook tower and DAC codec. Prompts carry the built-in voice or a voice reference.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const drama = @import("drama");
const budget = @import("cuda_memory.zig");
const guide = @import("drama_guide.zig");
const generation = @import("drama_generate.zig");
const Guide = guide.Guide;
const Allocator = std.mem.Allocator;

const Frame = drama.tower.Frame;
const Codes = [10]u32;

/// Backbone and tower rows a prompt pass takes at most.
pub const chunk_rows = 128;
/// Generation steps a request may take at most without --max-frames, the end-of-speech step included.
pub const default_max_frames = 1024;
/// Prompt tokens a request may carry at most, whatever the context.
const prompt_cap = 4096;
/// The tower's caches hold at most this many rows (its BadTowerCapacity).
const max_context = 8192;
/// The tower's hidden width, which every backbone row it reads must have.
const tower_hidden = 2048;
/// Semantic logits a backbone row carries: EOS first, then the 4096 codes.
const logit_cols = drama.backbone.weights.head_cols;
const sample_rate = 44100;
const frame_length = 2048;
/// Frames a streamed request decodes together after its first, which is decoded alone for the earliest audio.
const stream_chunk_frames = 4;
/// Frames a whole reply decodes together while it generates, so its silence guard hears the audio as it is made.
const whole_chunk_frames = 16;

pub const Options = struct {
    context: usize = 8192,
    /// Frames one synthesis may generate; the context holds them beside the prompt.
    max_frames: usize = default_max_frames,
    /// --silence-onset-s: seconds of silence that end a synthesis voiced for less than 1 s; 0: off.
    silence_onset_s: f64 = 0,
    /// --silence-mid-s: seconds of silence that make end of speech after 1 s voiced a silence; 0: off.
    silence_mid_s: f64 = 0,
    cfg_every: ?u32 = null, // --cfg-every: frames between a guided request's second-sequence runs; null: no guidance
    cfg_batched: bool = false, // --cfg-batched: a guided frame's two sequences decode as one batch
    device: u32 = 0,
    /// --expert-mask: the routed experts each backbone layer may pick; null: all of them.
    expert_mask: ?drama.expert_mask.Mask = null,
};

const Job = struct {
    id: api.Id,
    request: *const api.SpeechRequest,
    sink: api.Sink,
};

const Host = struct {
    gpa: Allocator,
    io: std.Io,
    context: usize,
    silence_onset_s: f64 = 0,
    silence_mid_s: f64 = 0,
    driver: cuda.Driver = undefined,
    ctx: cuda.Context = undefined,
    stream: cuda.Stream = undefined,
    backbone: drama.backbone.Model = undefined,
    tower: drama.tower.Model = undefined,
    codec: drama.codec.Model = undefined,
    guide: ?*Guide = null, // made at open with --cfg-every; null: guided requests are refused
    info_: api.Info = .{},
    startup: []u8 = &.{},
    previous_allocation_limit: ?u64 = null,
    semantic_start: u32 = 0,
    semantic_end: u32 = 0,
    eos: u32 = 0,
    thread: ?std.Thread = null,

    // Worker-thread only: the prompt pass's frames and the generated codes, sized once at open.
    frames: [chunk_rows + 1]Frame = undefined,
    codes: []Codes = &.{},

    // Under `mutex`: the one admitted request and what status reports of it.
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    slot: ?Job = null,
    running: bool = false,
    cancelled: bool = false,
    closing: bool = false,
    live_prompt: u64 = 0,
    live_frames: u64 = 0,
    decode_began: ?i96 = null,
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,

    const window_ns: i96 = 2 * std.time.ns_per_s;

    const vtable: api.Engine.VTable = .{
        .info = infoFn,
        .submit = submitFn,
        .cancel = cancelFn,
        .status = statusFn,
        .memory = memoryFn,
        .submit_speech = submitSpeechFn,
    };

    fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &vtable };
    }

    fn self(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }

    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }

    pub fn now(h: *Host) i96 {
        return std.Io.Clock.awake.now(h.io).toNanoseconds();
    }

    fn infoFn(ctx: *anyopaque) api.Info {
        return self(ctx).info_;
    }

    /// Speech-only: the text completion routes are refused.
    fn submitFn(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Unsupported;
    }

    /// Admits `request` when it fits the advertised limits and no other synthesis is admitted.
    fn submitSpeechFn(ctx: *anyopaque, id: api.Id, request: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const h = self(ctx);
        if (!h.valid(request)) return error.InvalidRequest;
        h.lock();
        defer h.unlock();
        if (h.closing) return error.Closed;
        if (h.slot != null) return error.Busy;
        h.slot = .{ .id = id, .request = request, .sink = sink };
        h.cancelled = false;
        h.wake.signal(h.io);
    }

    /// Sizes, sampling and prompt rows a synthesis may use (immutable host fields only).
    fn valid(h: *const Host, r: *const api.SpeechRequest) bool {
        const speech = h.info_.speech.?;
        if (r.prompt.len == 0 or r.prompt.len > speech.max_prompt_tokens) return false;
        if (r.max_frames == 0 or r.max_frames > speech.max_frames) return false;
        if (r.pcm.len < @as(usize, r.max_frames) * speech.frame_length) return false;
        // the tower's sampler accepts exactly these (tower.generate's BadSpeechSampling)
        if (!std.math.isFinite(r.temperature) or r.temperature < 0) return false;
        if (!std.math.isFinite(r.top_p) or r.top_p <= 0 or r.top_p > 1) return false;
        if (r.top_k == 0 or r.top_k > std.math.maxInt(i32)) return false;
        if (r.codes.len != 0 and r.codes.len != r.prompt.len) return false;
        if (r.generated.len != 0 and r.generated.len < r.max_frames) return false;
        for (r.prompt, 0..) |t, i| {
            if (t >= drama.backbone.config.vocab) return false;
            const audio = t >= h.semantic_start and t <= h.semantic_end;
            if (r.codes.len == 0) {
                if (audio) return false; // plain prompts carry no audio rows
                continue;
            }
            // build_prompt_rows: an audio row's token is semantic_start plus its code 0; a text row's codes are zero
            const codes = r.codes[i];
            if (!audio) {
                if (!std.mem.allEqual(u32, &codes, 0)) return false;
            } else if (codes[0] != t - h.semantic_start or std.mem.max(u32, &codes) >= 4096) return false;
        }
        if (r.unconditioned.len > 0 and !guide.admits(r, h.guide != null, speech.max_prompt_tokens, h.semantic_start, h.semantic_end)) return false;
        return true;
    }

    /// Ends the admitted request at its next prompt chunk, frame or codec boundary; unknown ids are ignored.
    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const job = h.slot orelse return;
        if (job.id != id) return;
        h.cancelled = true;
        h.wake.signal(h.io);
    }

    /// Host counters only: HTTP threads never touch the CUDA context.
    fn statusFn(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const t = h.now();
        var rate: f64 = 0;
        if (h.running) if (h.decode_began) |began| if (t > began) {
            rate = @as(f64, @floatFromInt(h.live_frames)) / (@as(f64, @floatFromInt(t - began)) / 1e9);
        };
        var streams: usize = 0;
        if (h.running and stream_tokens.len > 0) {
            stream_tokens[0] = @intCast(h.live_prompt + h.live_frames);
            streams = 1;
        }
        out.* = .{
            .running = @intFromBool(h.running),
            .waiting = @intFromBool(h.slot != null and !h.running),
            .decode_tokens_per_second = rate,
            .prefill_tokens_per_second = if (t - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = streams,
            .generation_tokens = if (h.running) h.live_frames else 0,
        };
    }

    /// The process's CUDA allocation counters (atomics, not a driver query).
    fn memoryFn(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .cache = 0, .peak = u.peak, .pinned = u.host };
    }

    /// The worker: takes the admitted request, synthesizes it, finishes it; on close, cancels what is admitted.
    fn run(h: *Host) void {
        // CUDA binds per thread; synthesize binds again and reports a failure there
        h.ctx.makeCurrent() catch |e| std.log.err("drama: cuCtxSetCurrent on the worker thread: {s}", .{@errorName(e)});
        while (true) {
            h.lock();
            while (h.slot == null and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
            const job = h.slot orelse {
                h.unlock();
                return;
            };
            h.running = true;
            h.live_prompt = job.request.prompt.len;
            h.live_frames = 0;
            h.decode_began = null;
            h.unlock();
            h.serve(job);
        }
    }

    /// True when the admitted request was cancelled or the engine is closing (read under the lock).
    fn stopped(h: *Host) bool {
        h.lock();
        defer h.unlock();
        return h.cancelled or h.closing;
    }

    pub fn check(h: *Host) error{Cancelled}!void {
        if (h.stopped()) return error.Cancelled;
    }

    /// One synthesis to its single terminal event; nothing touches the request after `finished`.
    fn serve(h: *Host, job: Job) void {
        const r = job.request;
        var stats: api.Stats = .{};
        const began = h.now();
        const done = h.synthesize(job, &stats, began) catch |e| {
            h.stream.synchronize() catch {}; // no device work outlives the request it served
            if (e == error.Cancelled) return h.finish(job, .cancelled, stats, "");
            std.log.warn("drama: synthesis {d} failed: {s}", .{ job.id, @errorName(e) });
            return h.finish(job, .failed, stats, words(e));
        };
        r.result.samples = done.samples; // published only after the full decode and the final cancellation check
        r.result.hollow = done.hollow;
        h.finish(job, done.reason, stats, "");
    }

    /// Frees the slot (a new request may be admitted while this one's event is out), then sends `finished`.
    fn finish(h: *Host, job: Job, reason: api.Reason, stats: api.Stats, message: []const u8) void {
        h.lock();
        h.slot = null;
        h.running = false;
        h.cancelled = false;
        h.live_frames = 0;
        h.decode_began = null;
        h.unlock();
        const event: api.Event = .{ .finished = .{ .reason = reason, .stats = stats, .message = message } };
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// Prompt pass, then the frame loop (`generation.run`) decoding into the request's PCM as it goes. A synthesis
    /// stopped by max_frames (`length`) or by its silence guard (`silence`) keeps the audio it made.
    fn synthesize(h: *Host, job: Job, stats: *api.Stats, began: i96) !Done {
        const r = job.request;
        try h.check();
        try h.ctx.makeCurrent(); // CUDA binds per thread: the worker's calls need the context current
        const guided = if (r.unconditioned.len > 0) try h.guide.?.prefill(h, r) else null;
        const last = try prefill(&h.backbone, &h.tower, h.stream, &h.frames, r.prompt, r.codes, chunk_rows, r.seed, {}, h);
        const prefilled = h.now();
        stats.prefill_seconds = @as(f64, @floatFromInt(prefilled - began)) / 1e9;
        h.notePrefill(r.prompt.len + r.unconditioned.len, began, prefilled);

        // Generation: the last row's hidden state and semantic logits give a frame; a frame that is not EOS is the
        // next backbone row. Steps are bounded by max_frames, the EOS step included.
        var steps: Steps = .{
            .h = h,
            .job = job,
            .stats = stats,
            .sampling = .{ .temperature = r.temperature, .top_p = r.top_p, .top_k = r.top_k },
            .row = last,
            .guide = guided,
        };
        // a caller's generated buffer takes the codes in place of the host's own: nothing is copied either way
        const codes: []Codes = if (r.generated.len > 0) r.generated else h.codes;
        const cadence: generation.Cadence = if (r.stream) .{ .first = 1, .chunk = stream_chunk_frames } else .{ .first = whole_chunk_frames, .chunk = whole_chunk_frames };
        h.codec.begin();
        const ending = try generation.run(&steps, codes[0..r.max_frames], cadence, .init(sample_rate, h.silence_onset_s, h.silence_mid_s), frame_length);
        if (h.guide) |g| g.note(h, job.id, guided != null, ending.frames, began);
        if (ending.frames == 0) return error.EmptySpeech;
        return .{ .samples = ending.frames * frame_length, .reason = ending.reason, .hollow = ending.hollow };
    }

    /// What a finished synthesis produced: its samples, why it ended (stop, length or silence), its mid hole's samples.
    const Done = struct { samples: usize, reason: api.Reason, hollow: usize };

    /// The frame loop's source: the backbone and tower's next frame, and the codec's decode of the frames so far.
    const Steps = struct {
        h: *Host,
        job: Job,
        stats: *api.Stats,
        sampling: drama.tower.Sampling,
        row: Row,
        count: usize = 0,
        guide: ?*Guide = null, // a guided request's; null: the main token is drawn from the backbone's logits as they are

        /// The next frame's codes, or null at end of speech; each frame after the first feeds the one before it in.
        pub fn next(s: *Steps) !?Codes {
            const h = s.h;
            const b = &h.backbone;
            const t = &h.tower;
            try h.check();
            if (s.count > 0) {
                if (s.guide) |g| {
                    const rows = try g.decode(h, h.frames[0..1]);
                    if (rows.rows != 1) return error.DramaRowsMismatch;
                    s.row = .{ .hidden = try lastRow(rows.hidden, 1, b.hidden * 2), .logits = try lastRow(rows.semantic_logits, 1, logit_cols * 4) };
                } else s.row = try feed(b, t, h.stream, h.frames[0..1], {});
            }
            const logits = if (s.guide) |g| try g.mixed(h, s.row.logits) else s.row.logits;
            const frame = try t.generate(h.stream, logits, s.row.hidden, s.sampling);
            s.stats.rounds += 1;
            if (try speechEnds(b, frame)) return null;
            h.frames[0] = frame;
            s.count += 1;
            h.noteFrames(s.count);
            return frame[1..11].*;
        }

        /// Decodes the frames not yet decoded and checks them; their PCM, still the host's until `publish`.
        pub fn decode(s: *Steps, codes: []const Codes) ![]const f32 {
            const h = s.h;
            try h.check();
            const r = s.job.request;
            const from = h.codec.decoded * frame_length;
            const samples = try h.codec.advance(h.stream, codes, r.pcm);
            if (samples != codes.len * frame_length) return error.CodecSampleCount;
            for (r.pcm[from..samples]) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
            try h.check();
            return r.pcm[from..samples];
        }

        /// Hands a stream's caller pcm[0..frames * 2048] (an `audio` event); a whole reply's audio waits for its end.
        pub fn publish(s: *Steps, frames: usize) !void {
            if (!s.job.request.stream) return;
            const event: api.Event = .{ .audio = frames * frame_length };
            s.job.sink.event(s.job.sink.ctx, s.job.id, &event);
        }
    };

    fn notePrefill(h: *Host, tokens: usize, began: i96, done: i96) void {
        h.lock();
        defer h.unlock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(tokens)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        h.decode_began = done;
    }

    fn noteFrames(h: *Host, frames: usize) void {
        h.lock();
        defer h.unlock();
        h.live_frames = frames;
    }
};

/// A step's source: the last row's hidden state and semantic logits, borrowed until the next backbone run.
pub const Row = struct { hidden: cuda.DeviceBuffer, logits: cuda.DeviceBuffer };

/// Resets both models for a synthesis seeded `seed`, then the prompt pass in chunks of `chunk` (up to chunk_rows) rows,
/// each staged in the tower, run through the backbone (runTraced), all but the last row teacher-forced; check() each.
pub fn prefill(b: *drama.backbone.Model, t: *drama.tower.Model, s: cuda.Stream, frames: *[chunk_rows + 1]Frame, prompt: []const u32, codes: []const Codes, chunk: usize, seed: u64, trace: anytype, check: anytype) !Row {
    if (chunk == 0 or chunk > chunk_rows) return error.BadDramaChunk;
    try b.reset(s);
    try t.reset(s, prompt, seed);
    var last: drama.backbone.Rows = undefined;
    var start: usize = 0;
    while (start < prompt.len) {
        if (@TypeOf(check) != void) try check.check();
        const end = @min(start + chunk, prompt.len);
        const n = end - start;
        const look = @min(end + 1, prompt.len); // this chunk's frames and the next row's, for its last label
        for (frames[0 .. look - start], start..) |*frame, i| frame.* = promptFrame(prompt, codes, i);
        const extra = try t.setInput(s, frames[0..n]);
        last = try b.runTraced(s, .prefill, prompt[start..end], extra, trace);
        if (last.rows != n) return error.DramaRowsMismatch;
        const taught = look - start - 1; // n inside the prompt; n - 1 in its final chunk
        if (taught > 0) try t.prefill(s, last.hidden, frames[1 .. 1 + taught]);
        // the final row's tower pass is generate's: it stages that row alone when prefill consumed the chunk
        if (end == prompt.len and taught > 0) _ = try t.setInput(s, frames[n - 1 .. n]);
        start = end;
    }
    return .{ .hidden = try lastRow(last.hidden, last.rows, b.hidden * 2), .logits = try lastRow(last.semantic_logits, 1, logit_cols * 4) };
}

/// Feeds `frame` to the backbone as the next row, its codes staged in the tower: the next step's source.
pub fn feed(b: *drama.backbone.Model, t: *drama.tower.Model, s: cuda.Stream, frame: *const [1]Frame, trace: anytype) !Row {
    const extra = try t.setInput(s, frame);
    const rows = try b.runTraced(s, .decode, frame[0][0..1], extra, trace);
    if (rows.rows != 1) return error.DramaRowsMismatch;
    return .{ .hidden = try lastRow(rows.hidden, 1, b.hidden * 2), .logits = try lastRow(rows.semantic_logits, 1, logit_cols * 4) };
}

/// True when `frame` (the tower's) ends speech; a token outside the semantic band is an error.
pub fn speechEnds(b: *const drama.backbone.Model, frame: Frame) !bool {
    if (frame[0] == b.eos) return true;
    if (frame[0] < b.semantic_start or frame[0] > b.semantic_end) return error.BadSpeechFrame;
    return false;
}

/// Prompt row `i`'s frame: its token, then its codes when the prompt carries them.
fn promptFrame(prompt: []const u32, codes: []const Codes, i: usize) Frame {
    var frame: Frame = @splat(0);
    frame[0] = prompt[i];
    if (codes.len > 0) frame[1..].* = codes[i];
    return frame;
}

/// A borrowed view of the last row; the backbone's semantic logits contain exactly one row.
fn lastRow(buffer: cuda.DeviceBuffer, rows: usize, row_bytes: usize) !cuda.DeviceBuffer {
    if (rows == 0 or buffer.len != rows * row_bytes) return error.DramaRowsMismatch;
    const at = (rows - 1) * row_bytes;
    return .{ .d = buffer.d, .ptr = buffer.ptr + at, .len = row_bytes, .borrowed = true };
}

/// The failure message a finished event carries (static text: it outlives the sink call).
fn words(err: anyerror) []const u8 {
    return switch (err) {
        error.EmptySpeech => "the model ended speech before its first audio frame",
        error.NonFiniteSpeechLogits => "the model produced non-finite speech logits",
        error.NonFiniteAudio => "the codec produced non-finite audio",
        error.BadSpeechFrame => "the tower sampled a token outside the semantic band",
        error.CodecSampleCount => "the codec returned a sample count other than frames x frame_length",
        error.DramaRowsMismatch => "the backbone returned rows other than those it was given",
        error.DramaContextOverflow => "the prompt and frames do not fit the engine's context",
        error.OutOfDeviceMemory => "the GPU ran out of memory during synthesis",
        else => @errorName(err),
    };
}

/// codec.Model.init run on its own thread (it makes the context current there); `model` is set before the thread ends.
const CodecInit = struct {
    gpa: Allocator,
    io: std.Io,
    ctx: *cuda.Context,
    checkpoint_dir: []const u8,
    kernel_dir: []const u8,
    max_frames: usize,
    model: anyerror!drama.codec.Model = undefined,

    fn run(c: *CodecInit) void {
        c.model = drama.codec.Model.init(c.gpa, c.io, c.ctx, c.checkpoint_dir, c.kernel_dir, .{ .max_frames = c.max_frames });
    }
};

/// What requests may ask of `context` rows when a synthesis may generate `frames`; null when the two do not fit.
pub fn speechInfo(context: usize, frames: usize) ?api.SpeechInfo {
    if (frames == 0 or context <= frames or context > max_context) return null;
    return .{ .sample_rate = sample_rate, .frame_length = frame_length, .max_frames = @intCast(frames), .max_prompt_tokens = @intCast(@min(prompt_cap, context - frames)) };
}

/// Drama 3 at `checkpoint_dir` with kernels from `kernel_dir`, served on its own worker thread; `close` joins it.
pub fn open(gpa: Allocator, io: std.Io, checkpoint_dir: []const u8, kernel_dir: []const u8, options: Options) !api.Opened {
    const speech = speechInfo(options.context, options.max_frames) orelse return error.BadDramaContext;
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .context = options.context, .silence_onset_s = options.silence_onset_s, .silence_mid_s = options.silence_mid_s };
    h.codes = try gpa.alloc(Codes, options.max_frames);
    errdefer gpa.free(h.codes);

    h.driver = try cuda.Driver.open();
    errdefer h.driver.close();
    const count = try h.driver.deviceCount();
    if (count <= 0 or options.device >= @as(u32, @intCast(count))) return error.NoSuchCudaDevice;
    h.ctx = try cuda.Context.init(&h.driver, @intCast(options.device));
    errdefer h.ctx.deinit();
    h.stream = try cuda.Stream.init(&h.driver, true);
    errdefer h.stream.deinit();
    var admission_arena: std.heap.ArenaAllocator = .init(gpa);
    defer admission_arena.deinit();
    var admission_problem: []const u8 = "";
    const plan = try budget.readPool(admission_arena.allocator(), io, &h.ctx, &admission_problem) orelse {
        std.log.err("Drama memory admission: {s}", .{admission_problem});
        return error.DramaMemoryAdmission;
    };
    const used = cuda.usage(false);
    const held = used.device + used.host;
    h.previous_allocation_limit = try cuda.setAllocationLimit(held + plan.room(held));
    errdefer _ = cuda.setAllocationLimit(h.previous_allocation_limit) catch unreachable;

    const held0 = cuda.usage(false).device;
    // The codec builds its weight image from codec.pth on the CPU; on its own thread that overlaps the backbone and
    // tower reads, which leave the CPU idle.
    var codec: CodecInit = .{ .gpa = gpa, .io = io, .ctx = &h.ctx, .checkpoint_dir = checkpoint_dir, .kernel_dir = kernel_dir, .max_frames = options.max_frames };
    var codec_thread: ?std.Thread = try std.Thread.spawn(.{}, CodecInit.run, .{&codec});
    errdefer if (codec_thread) |t| {
        t.join();
        if (codec.model) |*m| m.deinit() else |_| {}
    };
    h.backbone = try drama.backbone.Model.init(gpa, io, &h.ctx, checkpoint_dir, kernel_dir, .{ .context = options.context, .max_rows = chunk_rows, .expert_mask = options.expert_mask });
    errdefer h.backbone.deinit();
    h.tower = try drama.tower.Model.init(gpa, io, &h.ctx, checkpoint_dir, kernel_dir, .{ .context = options.context, .max_rows = chunk_rows });
    errdefer h.tower.deinit();
    codec_thread.?.join();
    codec_thread = null;
    h.codec = try codec.model;
    errdefer h.codec.deinit();
    const model_bytes = cuda.usage(false).device -| held0;

    if (h.backbone.hidden != tower_hidden) return error.DramaHiddenMismatch;
    if (h.codec.sample_rate != sample_rate or h.codec.frame_length != frame_length or h.codec.num_codebooks != 10) return error.UnsupportedDramaCodec;
    h.semantic_start = h.backbone.semantic_start;
    h.semantic_end = h.backbone.semantic_end;
    h.eos = h.backbone.eos;
    if (options.cfg_every) |every| h.guide = try Guide.create(gpa, &h.backbone, every, options.cfg_batched);
    errdefer if (h.guide) |g| g.destroy(gpa);

    var name_buf: [256]u8 = undefined;
    const name = h.ctx.name(&name_buf) catch "GPU";
    const capability = try h.ctx.capability();
    const fp4 = h.backbone.w.nvfp4Layers();
    var fp4_buf: [64]u8 = undefined;
    const fp4_note = if (fp4 == 0) "" else try std.fmt.bufPrint(&fp4_buf, "; routed experts NVFP4 in {d} of {d} layers", .{ fp4, h.backbone.w.layers.len });
    var dense_buf: [256]u8 = undefined;
    const dense_note = try drama.backbone.weights.dense.note(&dense_buf, h.backbone.w.denseFp8(), h.backbone.w.layers.len, h.tower.denseFp8(), drama.tower.layer_count);
    const all = drama.backbone.config.layers * drama.backbone.config.experts;
    var mask_buf: [96]u8 = undefined;
    const mask = if (options.expert_mask) |*m| std.fmt.bufPrint(&mask_buf, "; routed experts masked to {d} of {d}", .{ drama.expert_mask.total(m), all }) catch unreachable else "";
    var held_buf: [96]u8 = undefined;
    const held_note = if (h.backbone.w.held) |*x| std.fmt.bufPrint(&held_buf, "; routed experts held {d} of {d}", .{ x.total(), all }) catch unreachable else "";
    h.startup = try std.fmt.allocPrint(gpa, "Drama 3 on CUDA sm_{d} device {d} ({s}): models {d:.2} GiB; context {d} tokens, prompts up to {d} tokens in passes of {d} rows, up to {d} frames of {d} samples at {d} Hz{s}{s}; one synthesis at a time{s}{s}{s}", .{
        capability, options.device, name, @as(f64, @floatFromInt(model_bytes)) / (1 << 30), options.context, speech.max_prompt_tokens, chunk_rows, speech.max_frames, frame_length, sample_rate, held_note, mask, fp4_note, dense_note, if (h.guide) |g| g.describe() else "",
    });
    errdefer gpa.free(h.startup);
    h.info_ = .{
        .name = "drama-3",
        .text_generation = false,
        .speech = speech,
        .lanes = 1,
        .context_window = @intCast(options.context),
        .thinking_budget = false,
        .plain_only = true,
        .startup = h.startup,
    };
    h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, Host.run, .{h});
    return .{ .engine = h.engine(), .close = close, .ctx = h };
}

/// Stops admitting, cancels the admitted request (its finished event still arrives), joins the worker, frees all.
fn close(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.lock();
    h.closing = true;
    h.wake.broadcast(h.io);
    h.unlock();
    if (h.thread) |t| t.join();
    h.thread = null;
    h.ctx.makeCurrent() catch {}; // the closing thread frees device memory
    h.stream.synchronize() catch {};
    h.codec.deinit();
    h.tower.deinit();
    if (h.guide) |g| g.destroy(h.gpa);
    h.backbone.deinit();
    h.stream.deinit();
    _ = cuda.setAllocationLimit(h.previous_allocation_limit) catch |e| blk: {
        std.log.err("Drama memory budget restore: {s}", .{@errorName(e)});
        break :blk null;
    };
    h.ctx.deinit();
    h.driver.close();
    h.gpa.free(h.startup);
    h.gpa.free(h.codes);
    h.gpa.destroy(h);
}

test {
    _ = generation;
    _ = guide;
}

test "without --max-frames the limits are as before; a larger limit takes its rows from the prompt's share" {
    try std.testing.expectEqual(api.SpeechInfo{ .sample_rate = 44100, .frame_length = 2048, .max_frames = 1024, .max_prompt_tokens = 4096 }, speechInfo(8192, default_max_frames).?);
    try std.testing.expectEqual(@as(u32, 1025), speechInfo(2049, 1024).?.max_prompt_tokens);
    try std.testing.expectEqual(@as(u32, 4096), speechInfo(8192, 4096).?.max_prompt_tokens);
    try std.testing.expectEqual(@as(u32, 1192), speechInfo(8192, 7000).?.max_prompt_tokens);
    try std.testing.expectEqual(@as(u32, 1), speechInfo(4097, 4096).?.max_prompt_tokens);
    for ([_][2]usize{ .{ 4096, 4096 }, .{ 8192, 0 }, .{ 8193, 1024 }, .{ 16384, 8192 } }) |bad| try std.testing.expect(speechInfo(bad[0], bad[1]) == null);
}

test "admission holds a request to the advertised frame limit, with pcm and code buffers for all of it" {
    const gpa = std.testing.allocator;
    const h = try gpa.create(Host);
    defer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = std.testing.io, .context = 8192, .info_ = .{ .speech = speechInfo(8192, 4096).? }, .semantic_start = 248099, .semantic_end = 252194 };
    const pcm = try gpa.alloc(f32, 4096 * frame_length);
    defer gpa.free(pcm);
    const generated = try gpa.alloc(Codes, 4096);
    defer gpa.free(generated);
    var result: api.SpeechResult = .{};
    const ok: api.SpeechRequest = .{ .prompt = &.{ 1, 2 }, .generated = generated, .pcm = pcm, .result = &result, .seed = 0, .max_frames = 4096 };
    try std.testing.expect(h.valid(&ok));
    var r = ok;
    r.max_frames = 4097;
    try std.testing.expect(!h.valid(&r));
    r = ok;
    r.pcm = pcm[0 .. pcm.len - 1];
    try std.testing.expect(!h.valid(&r));
    r = ok;
    r.generated = generated[0..4095];
    try std.testing.expect(!h.valid(&r));
    h.info_.speech = speechInfo(8192, default_max_frames).?;
    try std.testing.expect(!h.valid(&ok));
    r = ok;
    r.max_frames = default_max_frames;
    try std.testing.expect(h.valid(&r));
}
