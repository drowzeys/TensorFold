//! LFM3-Omni-2B speech in, speech and text out, on one GPU: the log-mel front end (host), the speech encoder and its
//! projector, the LFM3 backbone, the depth decoder that writes each speech frame's 12 codes, and the Liquid codec's
//! decoder streaming 24 kHz audio. bf16 weights as released; one request at a time.
const std = @import("std");
const cuda = @import("cuda");
const st = @import("core").safetensors;
const kern = @import("kernels.zig");
const stack = @import("stack.zig");
const melmod = @import("mel.zig");
const cfg = @import("config.zig");

const Stack = stack.Stack;
const Arena = stack.Arena;

pub const Options = struct {
    device: usize = 0,
    /// Longest input clip (seconds of 16 kHz audio) the encoder's buffers take.
    max_input_s: usize = 60,
    /// Backbone rows: the window (prompt and reply) and a prompt chunk.
    max_ctx: usize = 4096,
    prefill_rows: usize = 512,
    /// Longest reply in speech frames (12.5 a second) and the codec's largest chunk.
    max_frames: usize = 1024,
    codec_chunk: usize = 64,
    /// Replay the backbone's row and the depth decoder's frame as CUDA graphs.
    graphs: bool = true,
};

pub const Sampling = struct {
    /// 0 picks the most likely code; the model card's setting is 0.4.
    audio_temperature: f32 = 0.4,
    audio_top_k: usize = 0,
    audio_repetition_penalty: f32 = 1.2,
    audio_repetition_window: usize = 64,
    /// Reveal the codebooks in order 0..11 instead of a random order (the reference's randperm).
    identity_order: bool = false,
    seed: u64 = 0,
    /// Generated slots (a text token or a speech frame each).
    max_new: usize = 512,
};

/// Where a reply goes as it is made.
pub const Sink = struct {
    ctx: *anyopaque,
    text: ?*const fn (*anyopaque, u32) anyerror!void = null,
    frame: ?*const fn (*anyopaque, *const [cfg.codebooks]i32) anyerror!void = null,
    pcm: ?*const fn (*anyopaque, []const f32) anyerror!void = null,
    /// After every backbone row (the engine's `lastHidden` reads it).
    hidden: ?*const fn (*anyopaque) anyerror!void = null,
};

/// When the codec runs while speech frames arrive: `first` frames for the first audio, then every `chunk`.
pub const Streaming = struct {
    first: usize = 1,
    chunk: usize = 8,
    /// Decode audio at all (frames only otherwise).
    audio: bool = true,
};

/// Teacher forcing: the reference reply's slots and frames are fed whatever the engine picks; the picks are scored.
pub const Forced = struct {
    seq: []const u32,
    frames: []const [cfg.codebooks]i32,
    text_hits: usize = 0,
    text_total: usize = 0,
    code_hits: usize = 0,
    code_total: usize = 0,
    frame_hits: usize = 0,
    frame_total: usize = 0,
};

pub const Stats = struct {
    mel_ms: f64 = 0,
    encode_ms: f64 = 0,
    prefill_ms: f64 = 0,
    first_text_ms: f64 = 0,
    first_frame_ms: f64 = 0,
    first_pcm_ms: f64 = 0,
    total_ms: f64 = 0,
    step_ms: f64 = 0,
    depth_ms: f64 = 0,
    codec_ms: f64 = 0,
    text_tokens: usize = 0,
    frames: usize = 0,
    samples: usize = 0,
    prompt_rows: usize = 0,
    audio_tokens: usize = 0,
};

pub fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn msSince(io: std.Io, t0: i128) f64 {
    return @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e6;
}

const Mode = enum { text, audio, interleaved };

/// Frames the device's repetition window holds (the reference's default window).
const hist_cap = 64;

const Downsample = struct {
    conv_in_w: u64,
    conv_in_b: u64,
    conv_mid_w: u64,
    conv_mid_b: u64,
    lin_mid_w: u64,
    lin_mid_b: u64,
    conv_end_w: u64,
    conv_end_b: u64,
    lin_end_w: u64,
    lin_end_b: u64,
    lin_out_w: u64,
    lin_out_b: u64,
};

/// The codec's spectrum to windowed frames, [bins re | bins im] x n_fft: irfft (norm "backward") times the window.
fn synthesisBasis(gpa: std.mem.Allocator, window: []const f32) ![]f32 {
    const n = cfg.n_fft;
    const nb = cfg.spec_bins;
    const m = try gpa.alloc(f32, 2 * nb * n);
    for (0..nb) |k| {
        const c: f64 = if (k == 0 or k == nb - 1) 1.0 else 2.0;
        for (0..n) |t| {
            const a = 2.0 * std.math.pi * @as(f64, @floatFromInt((k * t) % n)) / @as(f64, @floatFromInt(n));
            const w: f64 = window[t];
            m[k * n + t] = @floatCast(c * @cos(a) / @as(f64, @floatFromInt(n)) * w);
            m[(nb + k) * n + t] = @floatCast(-c * @sin(a) / @as(f64, @floatFromInt(n)) * w);
        }
    }
    return m;
}

/// The codec's overlap-add on the host: frames in, finished samples out (held back where later frames still add).
pub const OverlapAdd = struct {
    gpa: std.mem.Allocator,
    window_sq: [cfg.n_fft]f32,
    acc: std.ArrayList(f32) = .empty,
    env: std.ArrayList(f32) = .empty,
    rows: usize = 0,
    emitted: usize = cfg.istft_trim,
    out: std.ArrayList(f32) = .empty,

    pub fn init(gpa: std.mem.Allocator, window: []const f32) OverlapAdd {
        var o: OverlapAdd = .{ .gpa = gpa, .window_sq = undefined };
        for (&o.window_sq, window) |*q, w| q.* = w * w;
        return o;
    }

    pub fn deinit(o: *OverlapAdd) void {
        o.acc.deinit(o.gpa);
        o.env.deinit(o.gpa);
        o.out.deinit(o.gpa);
    }

    /// Adds windowed frames [n, n_fft]; returns the samples now final (valid until the next call).
    pub fn add(o: *OverlapAdd, frames: []const f32, final: bool) ![]const f32 {
        const n = frames.len / cfg.n_fft;
        for (0..n) |j| {
            const base = (o.rows + j) * cfg.hop;
            const need = base + cfg.n_fft;
            while (o.acc.items.len < need) {
                try o.acc.append(o.gpa, 0);
                try o.env.append(o.gpa, 0);
            }
            const fr = frames[j * cfg.n_fft ..][0..cfg.n_fft];
            for (0..cfg.n_fft) |t| {
                o.acc.items[base + t] += fr[t];
                o.env.items[base + t] += o.window_sq[t];
            }
        }
        o.rows += n;
        const end = if (final) o.rows * cfg.hop + cfg.istft_trim else o.rows * cfg.hop;
        o.out.clearRetainingCapacity();
        while (o.emitted < end and o.emitted < o.acc.items.len) : (o.emitted += 1) try o.out.append(o.gpa, o.acc.items[o.emitted] / o.env.items[o.emitted]);
        return o.out.items;
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    o: Options,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: kern.Kernels,
    a: Arena,
    mel: *melmod.Mel,
    window: [cfg.n_fft]f32,

    enc: Stack,
    bb: Stack,
    depth: Stack,
    codec: Stack,
    ds: Downsample,
    // projector
    ln_w: u64,
    ln_b: u64,
    l1_w: u64,
    l1_b: u64,
    l2_w: u64,
    l2_b: u64,
    // backbone tables
    embed: u64,
    cb_embed: u64,
    // depth decoder
    depth_w: u64,
    depth_b: u64,
    mask_emb: u64,
    depth_tables: u64, // [12 * 2017, 1024]
    depth_norms: u64, // [12, 1024]
    // codec
    po_w: u64,
    po_b: u64,
    fq_w: u64,
    fq_b: u64,
    lin_w: u64,
    lin_b: u64,
    basis: u64, // f32 [1282, 1280]

    // the log-mel front end on the device
    mel_pcm: u64, // f32 [samples]
    mel_win: u64, // f32 [400]
    mel_dft: u64, // f32 [400, 2 * 201]: cos | sin
    mel_fb: u64, // f32 [201, 128]
    mel_fr: u64, // f32 [frames, 400]
    mel_s: u64, // f32 [frames, 402]
    mel_p: u64, // f32 [frames, 201]
    mel_m: u64, // f32 [frames, 128]
    // the device's frame: order | codes | masked (i32 [3 * 12]) and the repetition window's frames (a ring)
    dframe: u64,
    hist: u64,
    step_graph: ?cuda.graph.Exec = null,
    frame_graph: ?cuda.graph.Exec = null,
    /// The sampling settings the frame graph was captured with (its kernels' arguments).
    frame_smp: Sampling = .{},
    // scratch
    mel_f32: u64,
    mel_bf: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    acc: u64, // f32
    enc_x: u64,
    enc_out: u64,
    ln_out: u64,
    h1: u64,
    audio_emb: u64,
    audio_tokens: usize = 0,
    /// The newest backbone row's hidden state (bf16 on the device).
    last: u64 = 0,
    ids: u64, // i32 [prefill_rows]
    bb_x: u64,
    bb_out: u64,
    logits: u64, // bf16 [vocab]
    pick: u64, // i32
    sup: u64, // i32 [4]
    din_f32: u64,
    din: u64,
    dx: u64,
    dout: u64,
    dz: u64,
    dlog: u64,
    dlog_f32: u64,
    dcodes: u64,
    dmask: u64,
    c_codes: u64,
    c_lat: u64,
    c_p1: u64,
    c_p2: u64,
    c_up: u64,
    c_out: u64,
    c_f32: u64,
    c_lin: u64,
    c_spec: u64,
    c_frames: u64,
    host_frames: []f32,
    host_logits: []f32,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpa = gpa;
        e.io = io;
        e.o = o;
        try cfg.check(gpa, io, model_dir);
        e.driver = try cuda.Driver.open();
        errdefer e.driver.close();
        e.ctx = try cuda.Context.init(&e.driver, std.math.cast(c_int, o.device) orelse return error.NoSuchCudaDevice);
        errdefer e.ctx.deinit();
        // a blocking stream: the synchronous copies of the legacy stream order with its work
        e.stream = try cuda.Stream.init(&e.driver, false);
        errdefer e.stream.deinit();
        e.k = try kern.Kernels.load(&e.driver);
        errdefer e.k.unload();
        e.a = .{ .gpa = gpa, .d = &e.driver };
        errdefer e.a.deinit();
        e.mel = try gpa.create(melmod.Mel);
        errdefer gpa.destroy(e.mel);
        e.mel.init();
        e.step_graph = null;
        e.frame_graph = null;
        e.frame_smp = .{};
        try e.loadModel(model_dir);
        try e.allocScratch();
        return e;
    }

    pub fn destroy(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.step_graph) |*g| g.deinit();
        if (e.frame_graph) |*g| g.deinit();
        e.gpa.free(e.host_frames);
        e.gpa.free(e.host_logits);
        for ([_]*Stack{ &e.enc, &e.bb, &e.depth, &e.codec }) |s| e.gpa.free(s.layers);
        e.a.deinit();
        e.k.unload();
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e.mel);
        e.gpa.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) usize {
        return e.a.bytes;
    }

    fn loadModel(e: *Engine, dir: []const u8) !void {
        const gpa = e.gpa;
        const mpath = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
        defer gpa.free(mpath);
        var f = try st.File.openPrefixes(gpa, e.io, mpath, &.{ "model.audio_encoder.", "model.audio_projector.", "model.language_model.", "model.codebook_embedding.", "model.depth_decoder." });
        defer f.close(e.io);
        const a = &e.a;
        const max_tokens = melmod.tokens(melmod.frames(e.o.max_input_s * melmod.sample_rate));
        e.enc = try stack.load(gpa, a, &f, cfg.encoderSpec(max_tokens));
        e.bb = try stack.load(gpa, a, &f, cfg.backboneSpec(e.o.prefill_rows, e.o.max_ctx));
        e.depth = try stack.load(gpa, a, &f, cfg.depthSpec());
        const p = "model.audio_encoder.downsample.";
        const C = cfg.conv_channels;
        e.ds = .{
            .conv_in_w = try up(a, &f, p ++ "conv_in.weight", &.{ C, 1, 3, 3 }),
            .conv_in_b = try up(a, &f, p ++ "conv_in.bias", &.{C}),
            .conv_mid_w = try up(a, &f, p ++ "conv_mid.weight", &.{ C, 1, 3, 3 }),
            .conv_mid_b = try up(a, &f, p ++ "conv_mid.bias", &.{C}),
            .lin_mid_w = try up(a, &f, p ++ "lin_mid.weight", &.{ C, C, 1, 1 }),
            .lin_mid_b = try up(a, &f, p ++ "lin_mid.bias", &.{C}),
            .conv_end_w = try up(a, &f, p ++ "conv_end.weight", &.{ C, 1, 3, 3 }),
            .conv_end_b = try up(a, &f, p ++ "conv_end.bias", &.{C}),
            .lin_end_w = try up(a, &f, p ++ "lin_end.weight", &.{ C, C, 1, 1 }),
            .lin_end_b = try up(a, &f, p ++ "lin_end.bias", &.{C}),
            .lin_out_w = try up(a, &f, p ++ "lin_out.weight", &.{ cfg.enc_d, C * cfg.mel_bins / 8, 1 }),
            .lin_out_b = try up(a, &f, p ++ "lin_out.bias", &.{cfg.enc_d}),
        };
        const q = "model.audio_projector.";
        e.ln_w = try up(a, &f, q ++ "layer_norm.weight", &.{cfg.enc_d});
        e.ln_b = try up(a, &f, q ++ "layer_norm.bias", &.{cfg.enc_d});
        e.l1_w = try up(a, &f, q ++ "linear_1.weight", &.{ cfg.hidden, cfg.enc_d });
        e.l1_b = try up(a, &f, q ++ "linear_1.bias", &.{cfg.hidden});
        e.l2_w = try up(a, &f, q ++ "linear_2.weight", &.{ cfg.hidden, cfg.hidden });
        e.l2_b = try up(a, &f, q ++ "linear_2.bias", &.{cfg.hidden});
        e.embed = try up(a, &f, "model.language_model.embed_tokens.weight", &.{ cfg.vocab, cfg.hidden });
        e.cb_embed = try up(a, &f, "model.codebook_embedding.weight", &.{ cfg.codebooks * cfg.codebook_size, cfg.hidden });
        const dd = "model.depth_decoder.";
        e.depth_w = try up(a, &f, dd ++ "depth_linear.weight", &.{ cfg.codebooks * cfg.depth_d, cfg.hidden });
        e.depth_b = try up(a, &f, dd ++ "depth_linear.bias", &.{cfg.codebooks * cfg.depth_d});
        e.mask_emb = try up(a, &f, dd ++ "mask_embedding", &.{cfg.depth_d});
        const tbytes = cfg.codebook_size * cfg.depth_d * 2;
        e.depth_tables = try a.alloc(cfg.codebooks * tbytes);
        e.depth_norms = try a.alloc(cfg.codebooks * cfg.depth_d * 2);
        const tbuf = a.bufs.items[a.bufs.items.len - 2];
        const nbuf = a.bufs.items[a.bufs.items.len - 1];
        var name: [128]u8 = undefined;
        for (0..cfg.codebooks) |i| {
            try tbuf.upload(i * tbytes, try stack.tensor(&f, try std.fmt.bufPrint(&name, dd ++ "depth_embeddings.{d}.embedding.weight", .{i}), &.{ cfg.codebook_size, cfg.depth_d }));
            try nbuf.upload(i * cfg.depth_d * 2, try stack.tensor(&f, try std.fmt.bufPrint(&name, dd ++ "depth_embeddings.{d}.embedding_norm.weight", .{i}), &.{cfg.depth_d}));
        }
        f.release();

        const cpath = try std.fs.path.join(gpa, &.{ dir, "audio_tokenizer", "model.safetensors" });
        defer gpa.free(cpath);
        var c = try st.File.openPrefixes(gpa, e.io, cpath, &.{ "decoder.", "quantizer.", "from_quantizer." });
        defer c.close(e.io);
        e.codec = try stack.load(gpa, a, &c, cfg.codecSpec(6 * e.o.codec_chunk, 6 * e.o.max_frames + 6 * e.o.codec_chunk));
        e.po_w = try up(a, &c, "quantizer.project_out.weight", &.{ cfg.quantizer_dim, cfg.codebooks * 4 });
        e.po_b = try up(a, &c, "quantizer.project_out.bias", &.{cfg.quantizer_dim});
        e.fq_w = try up(a, &c, "from_quantizer.weight", &.{ cfg.codec_d, cfg.quantizer_dim });
        e.fq_b = try up(a, &c, "from_quantizer.bias", &.{cfg.codec_d});
        e.lin_w = try up(a, &c, "decoder.lin.weight", &.{ 2 * cfg.spec_bins, cfg.codec_d });
        e.lin_b = try up(a, &c, "decoder.lin.bias", &.{2 * cfg.spec_bins});
        const wt = c.get("decoder.istft.window") orelse return error.MissingTensor;
        if (!wt.is(.f32, &.{cfg.n_fft})) return error.BadTensor;
        @memcpy(std.mem.asBytes(&e.window), wt.bytes);
        const basis = try synthesisBasis(gpa, &e.window);
        defer gpa.free(basis);
        e.basis = try a.upload(std.mem.sliceAsBytes(basis));
        c.release();
    }

    fn up(a: *Arena, f: *const st.File, name: []const u8, shape: []const usize) !u64 {
        return a.upload(try stack.tensor(f, name, shape));
    }

    fn allocScratch(e: *Engine) !void {
        const a = &e.a;
        const F = melmod.frames(e.o.max_input_s * melmod.sample_rate);
        const T = melmod.tokens(F);
        const C = cfg.conv_channels;
        const w1 = (F + 1) / 2;
        const w2 = (w1 + 1) / 2;
        e.mel_f32 = try a.alloc(cfg.mel_bins * F * 4);
        e.mel_pcm = try a.alloc((e.o.max_input_s * melmod.sample_rate + melmod.n_fft) * 4);
        e.mel_fr = try a.alloc(F * melmod.n_fft * 4);
        e.mel_s = try a.alloc(F * 2 * melmod.bins * 4);
        e.mel_p = try a.alloc(F * melmod.bins * 4);
        e.mel_m = try a.alloc(F * melmod.mels * 4);
        {
            const gpa = e.gpa;
            const win = try gpa.alloc(f32, melmod.n_fft);
            defer gpa.free(win);
            for (win, e.mel.window) |*w, x| w.* = @floatCast(x);
            e.mel_win = try a.upload(std.mem.sliceAsBytes(win));
            const dft = try gpa.alloc(f32, melmod.n_fft * 2 * melmod.bins);
            defer gpa.free(dft);
            for (0..melmod.n_fft) |n| for (0..melmod.bins) |kk| {
                dft[n * 2 * melmod.bins + kk] = @floatCast(e.mel.cos[n][kk]);
                dft[n * 2 * melmod.bins + melmod.bins + kk] = @floatCast(e.mel.sin[n][kk]);
            };
            e.mel_dft = try a.upload(std.mem.sliceAsBytes(dft));
            e.mel_fb = try a.upload(std.mem.sliceAsBytes(&e.mel.filters));
        }
        e.dframe = try a.alloc((3 * cfg.codebooks + 2) * 4);
        e.hist = try a.zeroed(hist_cap * cfg.codebooks * 4);
        e.mel_bf = try a.alloc(cfg.mel_bins * F * 2);
        e.a1 = try a.alloc(C * 64 * w1 * 2);
        e.a2 = try a.alloc(C * 32 * w2 * 2);
        e.a3 = try a.alloc(C * 32 * w2 * 2);
        e.acc = try a.alloc(@max(C * 32 * w2, T * cfg.hidden) * 4);
        e.enc_x = try a.alloc(T * cfg.enc_d * 2);
        e.enc_out = try a.alloc(T * cfg.enc_d * 2);
        e.ln_out = try a.alloc(T * cfg.enc_d * 2);
        e.h1 = try a.alloc(T * cfg.hidden * 2);
        e.audio_emb = try a.alloc(T * cfg.hidden * 2);
        const R = e.o.prefill_rows;
        e.ids = try a.alloc(R * 4);
        e.bb_x = try a.alloc(R * cfg.hidden * 2);
        e.bb_out = try a.alloc(R * cfg.hidden * 2);
        e.logits = try a.alloc(cfg.vocab * 2);
        e.pick = try a.alloc(4);
        const sup = [_]i32{ cfg.audio_token_id, cfg.audio_out_token_id, cfg.image_token_id, cfg.video_token_id };
        e.sup = try a.upload(std.mem.sliceAsBytes(&sup));
        const NQ = cfg.codebooks;
        e.din_f32 = try a.alloc(NQ * cfg.depth_d * 4);
        e.din = try a.alloc(NQ * cfg.depth_d * 2);
        e.dx = try a.alloc(NQ * cfg.depth_d * 2);
        e.dout = try a.alloc(NQ * cfg.depth_d * 2);
        e.dz = try a.alloc(NQ * cfg.depth_d * 2);
        e.dlog = try a.alloc(NQ * cfg.codebook_size * 2);
        e.dlog_f32 = try a.alloc(NQ * cfg.codebook_size * 4);
        e.dcodes = try a.alloc(NQ * 4);
        e.dmask = try a.alloc(NQ * 4);
        const CC = e.o.codec_chunk;
        e.c_codes = try a.alloc(NQ * CC * 4);
        e.c_lat = try a.alloc(CC * NQ * 4 * 2);
        e.c_p1 = try a.alloc(CC * cfg.quantizer_dim * 2);
        e.c_p2 = try a.alloc(CC * cfg.codec_d * 2);
        e.c_up = try a.alloc(6 * CC * cfg.codec_d * 2);
        e.c_out = try a.alloc(6 * CC * cfg.codec_d * 2);
        e.c_f32 = try a.alloc(6 * CC * 2 * cfg.spec_bins * 4);
        e.c_lin = try a.alloc(6 * CC * 2 * cfg.spec_bins * 2);
        e.c_spec = try a.alloc(6 * CC * 2 * cfg.spec_bins * 4);
        e.c_frames = try a.alloc(6 * CC * cfg.n_fft * 4);
        e.host_frames = try e.gpa.alloc(f32, 6 * CC * cfg.n_fft);
        e.host_logits = try e.gpa.alloc(f32, NQ * cfg.codebook_size);
    }

    // ---- copies through the legacy stream (synchronous; our stream is blocking, so they order with its work)

    pub fn h2d(e: *Engine, dst: u64, bytes: []const u8) !void {
        try e.driver.check(e.driver.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    pub fn d2h(e: *Engine, out: []u8, src: u64) !void {
        try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(out.ptr, src, out.len), "cuMemcpyDtoH");
    }

    fn d2d(e: *Engine, dst: u64, src: u64, n: usize) !void {
        try e.driver.check(e.driver.api.cuMemcpyDtoDAsync_v2(dst, src, n, e.stream.handle), "cuMemcpyDtoDAsync");
    }

    // ---- speech in

    /// Log-mel features of 16 kHz `pcm` into `out` (host, [128, frames]).
    pub fn features(e: *Engine, pcm: []const f32, out: []f32) void {
        e.mel.compute(pcm, out);
    }

    /// Log-mel features of 16 kHz `pcm` on the device (into `mel_f32`, [128, frames]); returns the frames.
    pub fn featuresDevice(e: *Engine, pcm: []const f32) !usize {
        const k = &e.k;
        const s = e.stream;
        const nf = melmod.frames(pcm.len);
        if (nf > melmod.frames(e.o.max_input_s * melmod.sample_rate)) return error.InputTooLong;
        try e.h2d(e.mel_pcm, std.mem.sliceAsBytes(pcm));
        try k.melFrames(s, e.mel_pcm, pcm.len, e.mel_win, e.mel_fr, nf, melmod.n_fft, melmod.hop);
        try k.matmul(s, false, false, nf, 2 * melmod.bins, melmod.n_fft, e.mel_fr, e.mel_dft, e.mel_s, .f32, .f32);
        try k.power(s, e.mel_s, e.mel_p, nf, melmod.bins);
        try k.matmul(s, false, false, nf, melmod.mels, melmod.bins, e.mel_p, e.mel_fb, e.mel_m, .f32, .f32);
        try k.logmel(s, e.mel_m, e.mel_f32, nf, melmod.mels, -12.0);
        return nf;
    }

    /// The speech encoder and projector over host features [128, nf] (null: `featuresDevice`'s, on the device):
    /// `audio_emb` holds one row per 8 frames.
    pub fn encode(e: *Engine, feats: ?[]const f32, nf: usize) !usize {
        const k = &e.k;
        const s = e.stream;
        const C = cfg.conv_channels;
        if (nf > melmod.frames(e.o.max_input_s * melmod.sample_rate)) return error.InputTooLong;
        if (feats) |fh| try e.h2d(e.mel_f32, std.mem.sliceAsBytes(fh));
        try k.f32ToBf16(s, e.mel_f32, e.mel_bf, cfg.mel_bins * nf);
        const w1 = (nf + 1) / 2;
        const w2 = (w1 + 1) / 2;
        const w3 = (w2 + 1) / 2;
        try k.conv2dS2(s, e.mel_bf, 1, cfg.mel_bins, nf, e.ds.conv_in_w, e.ds.conv_in_b, e.a1, C, 64, w1, true);
        try k.conv2dS2(s, e.a1, C, 64, w1, e.ds.conv_mid_w, e.ds.conv_mid_b, e.a2, C, 32, w2, false);
        try k.matmul(s, false, false, C, 32 * w2, C, e.ds.lin_mid_w, e.a2, e.acc, .bf16, .f32);
        try k.biasAct(s, e.acc, e.ds.lin_mid_b, e.a3, C, 32 * w2, true, .silu);
        try k.conv2dS2(s, e.a3, C, 32, w2, e.ds.conv_end_w, e.ds.conv_end_b, e.a2, C, 16, w3, false);
        try k.matmul(s, false, false, C, 16 * w3, C, e.ds.lin_end_w, e.a2, e.acc, .bf16, .f32);
        try k.biasAct(s, e.acc, e.ds.lin_end_b, e.a1, C, 16 * w3, true, .silu);
        // [C * 16, w3] rows to tokens: Z [w3, 768] = A^T Wo^T
        try k.matmul(s, true, true, w3, cfg.enc_d, C * 16, e.a1, e.ds.lin_out_w, e.acc, .bf16, .f32);
        try k.biasAct(s, e.acc, e.ds.lin_out_b, e.enc_x, w3, cfg.enc_d, false, .none);
        try e.enc.reset(s);
        try e.enc.forward(k, s, e.enc_x, w3, e.enc_out);
        try k.layernorm(s, e.enc_out, e.ln_w, e.ln_b, e.ln_out, cfg.enc_d, w3, 1e-5);
        try k.linearF32(s, e.l1_w, e.ln_out, cfg.enc_d, cfg.enc_d, cfg.hidden, e.acc, w3);
        try k.biasAct(s, e.acc, e.l1_b, e.h1, w3, cfg.hidden, false, .gelu);
        try k.linearF32(s, e.l2_w, e.h1, cfg.hidden, cfg.hidden, cfg.hidden, e.acc, w3);
        try k.biasAct(s, e.acc, e.l2_b, e.audio_emb, w3, cfg.hidden, false, .none);
        e.audio_tokens = w3;
        return w3;
    }

    // ---- the backbone

    /// Prompt rows through the backbone from an empty window: token ids, with the `audio_tokens` placeholder rows
    /// (contiguous, from `audio_at`) taken from the encoder. The last row's hidden state is left in `bb_out` row 0
    /// of the last chunk (`lastHidden`). `hidden_out`, if given, receives every row's state (fp32).
    pub fn prefill(e: *Engine, ids: []const u32, audio_at: usize, hidden_out: ?[]f32) !void {
        const k = &e.k;
        const s = e.stream;
        const R = e.o.prefill_rows;
        try e.bb.reset(s);
        var at: usize = 0;
        const ibuf = try e.gpa.alloc(i32, R);
        defer e.gpa.free(ibuf);
        const hb = try e.gpa.alloc(u16, R * cfg.hidden);
        defer e.gpa.free(hb);
        while (at < ids.len) {
            const t = @min(R, ids.len - at);
            for (0..t) |j| {
                const g = at + j;
                const audio = g >= audio_at and g < audio_at + e.audio_tokens;
                ibuf[j] = if (audio) -1 else @intCast(ids[g]);
            }
            try e.h2d(e.ids, std.mem.sliceAsBytes(ibuf[0..t]));
            try k.embed(s, e.embed, e.ids, e.bb_x, t, cfg.hidden);
            // the encoder's rows over this chunk's share of the placeholders
            const lo = @max(at, audio_at);
            const hi = @min(at + t, audio_at + e.audio_tokens);
            if (hi > lo) try e.d2d(e.bb_x + (lo - at) * cfg.hidden * 2, e.audio_emb + (lo - audio_at) * cfg.hidden * 2, (hi - lo) * cfg.hidden * 2);
            try e.bb.forward(k, s, e.bb_x, t, e.bb_out);
            if (hidden_out) |ho| {
                try e.d2h(std.mem.sliceAsBytes(hb[0 .. t * cfg.hidden]), e.bb_out);
                for (0..t * cfg.hidden) |j| ho[at * cfg.hidden + j] = bf16(hb[j]);
            }
            e.last = e.bb_out + (t - 1) * cfg.hidden * 2;
            at += t;
        }
    }

    /// One backbone row from the embedding in `bb_x` row 0; the state lands in `bb_out` row 0.
    fn stepRow(e: *Engine) !void {
        e.last = e.bb_out;
        if (!e.o.graphs) return e.bb.forward(&e.k, e.stream, e.bb_x, 1, e.bb_out);
        if (e.bb.pos + 1 > e.bb.s.max_ctx) return error.ContextFull;
        if (e.step_graph == null) {
            const saved = e.bb.pos;
            try cuda.graph.beginCapture(e.stream, .thread_local);
            e.bb.forwardAt(&e.k, e.stream, e.bb_x, 1, e.bb_out, true) catch |er| {
                var g = cuda.graph.endCapture(e.stream) catch return er;
                g.deinit();
                return er;
            };
            var g = try cuda.graph.endCapture(e.stream);
            defer g.deinit();
            e.step_graph = try g.instantiate();
            e.bb.pos = saved;
        }
        try e.step_graph.?.launchOn(e.stream);
        e.bb.pos += 1;
    }

    /// The greedy text token from the last hidden state (placeholders excluded), left on the device in `pick`.
    fn textPick(e: *Engine) !u32 {
        try e.k.linear(e.stream, e.embed, e.last, cfg.hidden, cfg.hidden, cfg.vocab, e.logits, cfg.vocab, 1);
        try e.k.argmax(e.stream, e.logits, cfg.vocab, e.sup, 4, e.pick);
        var v: i32 = 0;
        try e.d2h(std.mem.asBytes(&v), e.pick);
        return @intCast(v);
    }

    /// Text logits (fp32 on the host) from the last hidden state.
    pub fn textLogits(e: *Engine, out: []f32) !void {
        try e.k.linear(e.stream, e.embed, e.last, cfg.hidden, cfg.hidden, cfg.vocab, e.logits, cfg.vocab, 1);
        const hb = try e.gpa.alloc(u16, cfg.vocab);
        defer e.gpa.free(hb);
        try e.d2h(std.mem.sliceAsBytes(hb), e.logits);
        for (out, hb) |*o, h| o.* = bf16(h);
    }

    /// Sets the newest hidden state from the host (bf16-exact fp32 values), for checks of the heads alone.
    pub fn setHidden(e: *Engine, h: []const f32) !void {
        var hb: [cfg.hidden]u16 = undefined;
        for (&hb, h) |*o, v| o.* = @intCast(@as(u32, @bitCast(v)) >> 16);
        try e.h2d(e.bb_out, std.mem.sliceAsBytes(&hb));
        e.last = e.bb_out;
    }

    pub fn lastHidden(e: *Engine, out: []f32) !void {
        var hb: [cfg.hidden]u16 = undefined;
        try e.d2h(std.mem.sliceAsBytes(&hb), e.last);
        for (out, hb) |*o, h| o.* = bf16(h);
    }

    // ---- the depth decoder

    /// The depth decoder's slots from the last hidden state (run once a frame).
    pub fn depthStart(e: *Engine) !void {
        try e.k.linearF32(e.stream, e.depth_w, e.last, cfg.hidden, cfg.hidden, cfg.codebooks * cfg.depth_d, e.din_f32, 1);
        try e.k.biasAct(e.stream, e.din_f32, e.depth_b, e.din, 1, cfg.codebooks * cfg.depth_d, false, .none);
    }

    /// One pass: every codebook's logits (host, fp32 [12, 2017]) given the known codes and the masked set.
    pub fn depthPass(e: *Engine, codes: *const [cfg.codebooks]i32, masked: *const [cfg.codebooks]bool) ![]f32 {
        const k = &e.k;
        const s = e.stream;
        var m: [cfg.codebooks]i32 = undefined;
        for (&m, masked) |*x, y| x.* = @intFromBool(y);
        try e.h2d(e.dcodes, std.mem.sliceAsBytes(codes));
        try e.h2d(e.dmask, std.mem.sliceAsBytes(&m));
        try k.depthIn(s, e.din, e.mask_emb, e.depth_tables, e.dcodes, e.dmask, e.dx, cfg.codebooks, cfg.depth_d, cfg.codebook_size);
        try e.depth.reset(s);
        try e.depth.forward(k, s, e.dx, cfg.codebooks, e.dout);
        try k.rmsnormRows(s, e.dout, 0, e.depth_norms, cfg.depth_d, e.dz, cfg.depth_d, cfg.depth_d, cfg.depth_d, cfg.codebooks, 1e-5, .head);
        try k.linearBatched(s, e.depth_tables, cfg.codebook_size * cfg.depth_d, e.dz, cfg.depth_d, cfg.depth_d, cfg.codebook_size, e.dlog, cfg.codebook_size, 1, cfg.codebooks);
        try k.bf16ToF32(s, e.dlog, e.dlog_f32, cfg.codebooks * cfg.codebook_size);
        try e.d2h(std.mem.sliceAsBytes(e.host_logits), e.dlog_f32);
        return e.host_logits;
    }

    /// A speech frame's codes from the last hidden state: the codebooks revealed over the mask schedule's passes.
    pub fn sampleFrame(e: *Engine, smp: Sampling, rng: std.Random, history: []const [cfg.codebooks]i32) ![cfg.codebooks]i32 {
        try e.depthStart();
        var codes: [cfg.codebooks]i32 = @splat(0);
        var masked: [cfg.codebooks]bool = @splat(true);
        var order: [cfg.codebooks]usize = undefined;
        for (&order, 0..) |*x, i| x.* = i;
        if (!smp.identity_order) rng.shuffle(usize, &order);
        var at: usize = 0;
        for (cfg.maskgit_schedule) |n| {
            const logits = try e.depthPass(&codes, &masked);
            for (order[at .. at + n]) |q| codes[q] = pickCode(logits[q * cfg.codebook_size ..][0..cfg.codebook_size], q, smp, rng, history);
            for (order[at .. at + n]) |q| masked[q] = false;
            at += n;
        }
        return codes;
    }

    /// A speech frame on the device, from the last hidden state: the passes of the mask schedule each pick their
    /// codebooks (`om_depth_pick`); the frame (all end-of-speech if any code is) is left in `dframe`'s codes and fed
    /// to the backbone's next row in `bb_x`. `hn` frames of the repetition window are in `hist`.
    fn frameDevice(e: *Engine, smp: Sampling, order: *const [cfg.codebooks]i32, frame: usize, hn: usize) !void {
        const NQ = cfg.codebooks;
        var init: [3 * NQ + 2]i32 = undefined;
        @memcpy(init[0..NQ], order);
        @memset(init[NQ .. 2 * NQ], 0);
        @memset(init[2 * NQ .. 3 * NQ], 1);
        init[3 * NQ] = @intCast(hn);
        init[3 * NQ + 1] = @intCast(frame);
        try e.h2d(e.dframe, std.mem.sliceAsBytes(&init));
        // the graph reads the hidden state from `bb_out` row 0
        if (e.last != e.bb_out) {
            try e.d2d(e.bb_out, e.last, cfg.hidden * 2);
            e.last = e.bb_out;
        }
        if (!e.o.graphs) return e.frameLaunches(smp, 0, frame, hn);
        const same = e.frame_smp.audio_repetition_penalty == smp.audio_repetition_penalty and e.frame_smp.audio_temperature == smp.audio_temperature and e.frame_smp.seed == smp.seed;
        if (e.frame_graph != null and !same) {
            e.frame_graph.?.deinit();
            e.frame_graph = null;
        }
        if (e.frame_graph == null) {
            try cuda.graph.beginCapture(e.stream, .thread_local);
            e.frameLaunches(smp, e.dframe + 3 * NQ * 4, 0, 0) catch |er| {
                var g = cuda.graph.endCapture(e.stream) catch return er;
                g.deinit();
                return er;
            };
            var g = try cuda.graph.endCapture(e.stream);
            defer g.deinit();
            e.frame_graph = try g.instantiate();
            e.frame_smp = smp;
        }
        try e.frame_graph.?.launchOn(e.stream);
    }

    /// The frame's launches: `meta` (the window's frames and the frame number on the device) or `frame` and `hn`.
    fn frameLaunches(e: *Engine, smp: Sampling, meta: u64, frame: usize, hn: usize) !void {
        const k = &e.k;
        const s = e.stream;
        const NQ = cfg.codebooks;
        const codes = e.dframe + NQ * 4;
        const masked = e.dframe + 2 * NQ * 4;
        try e.depthStart();
        var at: usize = 0;
        for (cfg.maskgit_schedule) |n| {
            try k.depthIn(s, e.din, e.mask_emb, e.depth_tables, codes, masked, e.dx, NQ, cfg.depth_d, cfg.codebook_size);
            try e.depth.reset(s);
            try e.depth.forward(k, s, e.dx, NQ, e.dout);
            try k.rmsnormRows(s, e.dout, 0, e.depth_norms, cfg.depth_d, e.dz, cfg.depth_d, cfg.depth_d, cfg.depth_d, NQ, 1e-5, .head);
            try k.linearBatched(s, e.depth_tables, cfg.codebook_size * cfg.depth_d, e.dz, cfg.depth_d, cfg.depth_d, cfg.codebook_size, e.dlog, cfg.codebook_size, 1, NQ);
            try k.depthPick(s, .{ .logits = e.dlog, .order = e.dframe, .p0 = at, .n = n, .hist = e.hist, .hn = hn, .penalty = smp.audio_repetition_penalty, .temperature = smp.audio_temperature, .seed = smp.seed, .frame = frame, .codes = codes, .masked = masked, .nq = NQ, .size = cfg.codebook_size, .meta = meta });
            at += n;
        }
        try k.frameEnd(s, codes, NQ, cfg.audio_eos_code);
    }

    // ---- the codec

    /// Decodes `frames` (each 12 codes) after those already decoded; windowed frames for the overlap-add (host).
    pub fn codecChunk(e: *Engine, frames: []const [cfg.codebooks]i32, lin_out: ?[]f32) ![]const f32 {
        const k = &e.k;
        const s = e.stream;
        const n = frames.len;
        if (n == 0) return &.{};
        if (n > e.o.codec_chunk) return error.CodecChunkTooLong;
        try e.h2d(e.c_codes, std.mem.sliceAsBytes(frames));
        try k.fsqDecode(s, e.c_codes, e.c_lat, n, cfg.codebooks);
        try k.linearF32(s, e.po_w, e.c_lat, cfg.codebooks * 4, cfg.codebooks * 4, cfg.quantizer_dim, e.c_f32, n);
        try k.biasAct(s, e.c_f32, e.po_b, e.c_p1, n, cfg.quantizer_dim, false, .none);
        try k.linearF32(s, e.fq_w, e.c_p1, cfg.quantizer_dim, cfg.quantizer_dim, cfg.codec_d, e.c_f32, n);
        try k.biasAct(s, e.c_f32, e.fq_b, e.c_p2, n, cfg.codec_d, false, .none);
        const r = cfg.frame_stack * n;
        try k.repeatRows(s, e.c_p2, e.c_up, n, cfg.codec_d, cfg.frame_stack);
        try e.codec.forward(k, s, e.c_up, r, e.c_out);
        try k.linearF32(s, e.lin_w, e.c_out, cfg.codec_d, cfg.codec_d, 2 * cfg.spec_bins, e.c_f32, r);
        try k.biasAct(s, e.c_f32, e.lin_b, e.c_lin, r, 2 * cfg.spec_bins, false, .none);
        if (lin_out) |lo| {
            const hb = try e.gpa.alloc(u16, r * 2 * cfg.spec_bins);
            defer e.gpa.free(hb);
            try e.d2h(std.mem.sliceAsBytes(hb), e.c_lin);
            for (lo[0..hb.len], hb) |*o, h| o.* = bf16(h);
        }
        try k.polar(s, e.c_lin, e.c_spec, r, cfg.spec_bins);
        try k.matmul(s, false, false, r, cfg.n_fft, 2 * cfg.spec_bins, e.c_spec, e.basis, e.c_frames, .f32, .f32);
        const out = e.host_frames[0 .. r * cfg.n_fft];
        try e.d2h(std.mem.sliceAsBytes(out), e.c_frames);
        return out;
    }

    pub fn codecReset(e: *Engine) !void {
        try e.codec.reset(e.stream);
    }

    // ---- a reply

    /// Generates a reply after `prefill` (the prompt ends on the interleaved marker): text slots and speech frames in
    /// the model's interleaved schedule, until <|im_end|> in text or `max_new` slots. Text and frames go to `sink` as
    /// they are made; audio through the codec per `stream`.
    pub fn reply(e: *Engine, smp: Sampling, stream: Streaming, sink: Sink, forced: ?*Forced, st_: *Stats, t_req: i128) !void {
        var prng = std.Random.DefaultPrng.init(smp.seed);
        const rng = prng.random();
        const io = e.io;
        var ola = OverlapAdd.init(e.gpa, &e.window);
        defer ola.deinit();
        if (stream.audio) try e.codecReset();
        var history: std.ArrayList([cfg.codebooks]i32) = .empty;
        defer history.deinit(e.gpa);
        var pending: std.ArrayList([cfg.codebooks]i32) = .empty;
        defer pending.deinit(e.gpa);
        var decoded: usize = 0;
        var mode: Mode = .interleaved;
        var speaking = false;
        var text_done = false;
        var slots_left: usize = cfg.interleaved_n_text;
        var fi: usize = 0; // forced: the next reference frame
        var si: usize = 1; // forced: the next reference slot (0 is the interleaved marker)
        var step: usize = 0;
        // the device path samples frames on the GPU (one copy back a frame); teacher forcing and top-k use the host's
        const device = forced == null and smp.audio_top_k == 0;
        var hist_n: usize = 0;
        while (step < smp.max_new) : (step += 1) {
            const interleaving = mode == .interleaved;
            const last_step = step + 1 == smp.max_new;
            var finished = false;
            if (mode == .audio or (interleaving and speaking)) {
                const t0 = nowNs(io);
                var codes: [cfg.codebooks]i32 = undefined;
                if (device) {
                    var order: [cfg.codebooks]i32 = undefined;
                    var perm: [cfg.codebooks]usize = undefined;
                    for (&perm, 0..) |*x, i| x.* = i;
                    if (!smp.identity_order) rng.shuffle(usize, &perm);
                    for (&order, perm) |*o, x| o.* = @intCast(x);
                    try e.frameDevice(smp, &order, step, @min(hist_n, hist_cap));
                    try e.d2h(std.mem.sliceAsBytes(&codes), e.dframe + cfg.codebooks * 4);
                } else {
                    const lo = history.items.len -| smp.audio_repetition_window;
                    codes = try e.sampleFrame(smp, rng, history.items[lo..]);
                }
                st_.depth_ms += msSince(io, t0);
                if (forced) |fc| {
                    // the reference's returned slots leave out its end-of-speech frames: a text slot where a frame
                    // is due is one
                    var mine_eos = false;
                    for (codes) |c| mine_eos = mine_eos or c == cfg.audio_eos_code;
                    if (si < fc.seq.len and fc.seq[si] == cfg.audio_out_token_id and fi < fc.frames.len) {
                        const ref = fc.frames[fi];
                        var all = true;
                        for (codes, ref) |c, r| {
                            fc.code_hits += @intFromBool(c == r);
                            all = all and c == r;
                        }
                        fc.code_total += cfg.codebooks;
                        fc.frame_hits += @intFromBool(all);
                        codes = ref;
                        fi += 1;
                        si += 1;
                    } else {
                        fc.frame_hits += @intFromBool(mine_eos);
                        codes = @splat(cfg.audio_eos_code);
                    }
                    fc.frame_total += 1;
                }
                var eos = false;
                for (codes) |c| eos = eos or c == cfg.audio_eos_code;
                if (eos) {
                    codes = @splat(cfg.audio_eos_code);
                    mode = .text;
                } else {
                    if (device) {
                        try e.k.histPush(e.stream, e.dframe + cfg.codebooks * 4, e.hist, hist_n % hist_cap, cfg.codebooks);
                        hist_n += 1;
                    } else try history.append(e.gpa, codes);
                    st_.frames += 1;
                    if (st_.frames == 1) st_.first_frame_ms = msSince(io, t_req);
                    if (sink.frame) |f| try f(sink.ctx, &codes);
                    try pending.append(e.gpa, codes);
                }
                if (!device) try e.h2d(e.dframe + cfg.codebooks * 4, std.mem.sliceAsBytes(&codes));
                try e.k.embedCodes(e.stream, e.cb_embed, e.dframe + cfg.codebooks * 4, e.bb_x, 1, cfg.hidden, cfg.codebooks, cfg.codebook_size);
            } else {
                var tok = try e.textPick();
                if (forced) |fc| {
                    if (si >= fc.seq.len) break;
                    const ref = fc.seq[si];
                    si += 1;
                    fc.text_hits += @intFromBool(tok == ref);
                    fc.text_total += 1;
                    tok = ref;
                }
                st_.text_tokens += 1;
                if (st_.text_tokens == 1) st_.first_text_ms = msSince(io, t_req);
                if (sink.text) |f| try f(sink.ctx, tok);
                if (tok == cfg.text_end_token_id) {
                    if (interleaving) {
                        text_done = true;
                        speaking = true;
                    } else finished = true;
                } else if (tok == cfg.eos_token_id) {
                    finished = true;
                } else if (interleaving) {} else if (tok == cfg.audio_start_token_id) {
                    mode = .audio;
                } else if (tok == cfg.mixed_start_token_id) {
                    mode = .interleaved;
                    slots_left = cfg.interleaved_n_text;
                    speaking = false;
                    text_done = false;
                }
                if (forced != null) {
                    const id: i32 = @intCast(tok);
                    try e.h2d(e.pick, std.mem.asBytes(&id));
                }
                try e.k.embed(e.stream, e.embed, e.pick, e.bb_x, 1, cfg.hidden);
            }
            if (interleaving and !text_done) {
                slots_left -= 1;
                if (slots_left == 0) {
                    speaking = !speaking;
                    slots_left = if (speaking) cfg.interleaved_n_audio else cfg.interleaved_n_text;
                }
            }
            if (forced) |fc| if (si >= fc.seq.len and mode != .audio and !(mode == .interleaved and speaking)) {
                finished = true;
            };
            // the codec before the next backbone row: the first frames as soon as they exist, then whole chunks
            var frames_out: []const f32 = &.{};
            if (stream.audio) {
                const want: usize = if (decoded == 0) stream.first else stream.chunk;
                if (pending.items.len >= want or (pending.items.len > 0 and (finished or mode == .text))) {
                    const t2 = nowNs(io);
                    frames_out = try e.codecChunk(pending.items, null);
                    decoded += pending.items.len;
                    pending.clearRetainingCapacity();
                    st_.codec_ms += msSince(io, t2);
                }
            }
            const go_on = !(finished or last_step);
            const t1 = nowNs(io);
            if (go_on) try e.stepRow();
            // the overlap-add on the host while the row runs
            if (frames_out.len > 0) {
                const pcm = try ola.add(frames_out, false);
                if (pcm.len > 0) {
                    if (st_.samples == 0) st_.first_pcm_ms = msSince(io, t_req);
                    st_.samples += pcm.len;
                    if (sink.pcm) |f| try f(sink.ctx, pcm);
                }
            }
            if (!go_on) break;
            if (sink.hidden) |f| try f(sink.ctx);
            st_.step_ms += msSince(io, t1);
        }
        if (stream.audio) try e.flushCodec(&ola, &pending, &decoded, sink, st_, t_req, true);
        st_.total_ms = msSince(io, t_req);
    }

    fn flushCodec(e: *Engine, ola: *OverlapAdd, pending: *std.ArrayList([cfg.codebooks]i32), decoded: *usize, sink: Sink, st_: *Stats, t_req: i128, final: bool) !void {
        const t0 = nowNs(e.io);
        var at: usize = 0;
        while (at < pending.items.len or (final and at == 0)) {
            const n = @min(e.o.codec_chunk, pending.items.len - at);
            const frames = try e.codecChunk(pending.items[at .. at + n], null);
            const last = final and at + n >= pending.items.len;
            const pcm = try ola.add(frames, last);
            if (pcm.len > 0) {
                if (st_.samples == 0) st_.first_pcm_ms = msSince(e.io, t_req);
                st_.samples += pcm.len;
                if (sink.pcm) |f| try f(sink.ctx, pcm);
            }
            decoded.* += n;
            at += n;
            if (n == 0) break;
        }
        pending.clearRetainingCapacity();
        st_.codec_ms += msSince(e.io, t0);
    }
};

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// One codebook's code: the repetition penalty over its recent codes, then greedy or tempered sampling.
fn pickCode(logits: []f32, q: usize, smp: Sampling, rng: std.Random, history: []const [cfg.codebooks]i32) i32 {
    const p = smp.audio_repetition_penalty;
    if (p != 1.0) {
        var seen: [cfg.codebook_size]bool = @splat(false);
        for (history) |fr| {
            const c: usize = @intCast(fr[q]);
            if (seen[c]) continue;
            seen[c] = true;
            logits[c] = if (logits[c] < 0) logits[c] * p else logits[c] / p;
        }
    }
    if (smp.audio_temperature == 0) {
        var best: usize = 0;
        for (logits, 0..) |v, i| if (v > logits[best]) {
            best = i;
        };
        return @intCast(best);
    }
    const t = smp.audio_temperature;
    var m: f32 = -std.math.inf(f32);
    for (logits) |*v| {
        v.* /= t;
        m = @max(m, v.*);
    }
    if (smp.audio_top_k > 0 and smp.audio_top_k < logits.len) {
        // the k-th largest as a floor
        var tmp: [cfg.codebook_size]f32 = undefined;
        @memcpy(tmp[0..logits.len], logits);
        std.mem.sort(f32, tmp[0..logits.len], {}, std.sort.desc(f32));
        const floor = tmp[smp.audio_top_k - 1];
        for (logits) |*v| if (v.* < floor) {
            v.* = -std.math.inf(f32);
        };
    }
    var sum: f64 = 0;
    for (logits) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    var u = rng.float(f64) * sum;
    for (logits, 0..) |v, i| {
        u -= v;
        if (u <= 0) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

test "overlap-add: streamed chunks give the samples of one whole decode" {
    const gpa = std.testing.allocator;
    var window: [cfg.n_fft]f32 = undefined;
    for (&window, 0..) |*w, i| w.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / cfg.n_fft));
    const rows = 37;
    const frames = try gpa.alloc(f32, rows * cfg.n_fft);
    defer gpa.free(frames);
    var prng = std.Random.DefaultPrng.init(7);
    for (frames) |*x| x.* = prng.random().float(f32) - 0.5;
    var whole = OverlapAdd.init(gpa, &window);
    defer whole.deinit();
    const a = try gpa.dupe(f32, try whole.add(frames, true));
    defer gpa.free(a);
    try std.testing.expectEqual(@as(usize, rows * cfg.hop), a.len);
    var parts = OverlapAdd.init(gpa, &window);
    defer parts.deinit();
    var b: std.ArrayList(f32) = .empty;
    defer b.deinit(gpa);
    var at: usize = 0;
    for ([_]usize{ 6, 6, 12, 1, 12 }) |n| {
        try b.appendSlice(gpa, try parts.add(frames[at * cfg.n_fft .. (at + n) * cfg.n_fft], at + n == rows));
        at += n;
    }
    try std.testing.expectEqualSlices(f32, a, b.items);
}

test "synthesis basis: a cosine bin comes back as the windowed cosine" {
    const gpa = std.testing.allocator;
    var window: [cfg.n_fft]f32 = @splat(1);
    const m = try synthesisBasis(gpa, &window);
    defer gpa.free(m);
    // the spectrum of cos(2 pi 3 t / n) is n / 2 at bin 3 (real)
    for (0..cfg.n_fft) |t| {
        const want = @cos(2.0 * std.math.pi * 3.0 * @as(f64, @floatFromInt(t)) / cfg.n_fft);
        const got = @as(f64, m[3 * cfg.n_fft + t]) * cfg.n_fft / 2.0;
        try std.testing.expectApproxEqAbs(want, got, 1e-5);
    }
}
