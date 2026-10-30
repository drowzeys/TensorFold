//! LFM3-Omni-2B on one CUDA GPU in bf16 as released, one request at a time: speech in, text and 24 kHz speech out.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kernels.zig");
const stack = @import("stack.zig");
const load = @import("load.zig");
const melmod = @import("../mel.zig");
const cfg = @import("../config.zig");
const common = @import("../common.zig");

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

pub const Sampling = common.Sampling;
pub const Sink = common.Sink;
pub const Streaming = common.Streaming;
pub const Forced = common.Forced;
pub const Stats = common.Stats;
pub const OverlapAdd = common.OverlapAdd;
pub const nowNs = common.nowNs;
const bf16 = common.bf16;
const pickCode = common.pickCode;

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
        try load.model(e, model_dir);
        try load.scratch(e);
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

    /// The speech encoder and projector over features [128, nf] (null: `featuresDevice`'s), a row of `audio_emb` per 8.
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

    /// Prompt rows from an empty window, the audio rows from `audio_at` from the encoder; `hidden_out` gets every row.
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
    pub fn stepRow(e: *Engine) !void {
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
    pub fn textPick(e: *Engine) !u32 {
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

    /// A speech frame picked on the device into `dframe`'s codes over `hn` frames of `hist`, all EOS if any code is.
    pub fn frameDevice(e: *Engine, smp: Sampling, order: *const [cfg.codebooks]i32, frame: usize, hn: usize) !void {
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

    /// A reply after `prefill`, until <|im_end|> in text or `max_new` slots (reply.zig).
    pub const reply = @import("reply.zig").reply;
};
