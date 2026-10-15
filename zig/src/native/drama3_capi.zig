//! libtensorfold_drama3.so: Fish Audio Drama 3 TTS (families/fish_drama3) behind a C ABI, for the Studio and other
//! hosts. One handle owns one GPU engine; calls on a handle must not overlap.
//!
//!   void*   tf_drama3_open(const char* model_dir, int max_ctx, int max_frames);   // NULL on failure
//!   int     tf_drama3_prompt(void* h, const char* text, const char* ref_text, const int32_t* ref_codes,
//!                            int ref_frames, int32_t* rows_out, int max_rows);       // rows [n, 11]; n or -1
//!   int     tf_drama3_generate(void* h, const int32_t* rows, int n_rows, float temperature, int top_k,
//!                              float top_p, uint64_t seed, int max_frames, int32_t* frames_out,
//!                              double* stats_out /* [4]: prefill ms, first frame ms, decode ms, frames/s */);
//!                              // frames [n, 11] (main token, codes 0..9), without the end frame; n or -1
//!   const char* tf_drama3_error(void);                                              // the last failure
//!   void    tf_drama3_close(void* h);
//! ref_codes: int32 [ref_frames, 10] (the codec's codes, frame-major), or NULL.
//!
//! The native codec (drama3-codec-v1 safetensors from tools/drama3/codec_convert.py), audio 44.1 kHz mono fp32:
//!   void*   tf_drama3_open2(const char* model_dir, const char* codec_path, int max_ctx, int max_frames);
//!   int     tf_drama3_encode_wav(void* h, const char* wav_path, int32_t* codes_out, int max_frames);  // [n, 10]
//!   int     tf_drama3_encode_pcm(void* h, const float* pcm, int64_t n, int rate, int32_t* codes_out, int max_frames);
//!   int64_t tf_drama3_decode(void* h, const int32_t* frames, int n, float* pcm_out);  // n * 2048 samples
//!   int     tf_drama3_speak(void* h, const int32_t* rows, int n_rows, float temperature, int top_k, float top_p,
//!                           uint64_t seed, int max_frames, int first_chunk, int chunk,
//!                           void (*on_audio)(void* user, const float* pcm, int n), void* user,
//!                           int32_t* frames_out /* [max_frames, 11] or NULL */,
//!                           double* stats_out /* [6]: prefill ms, first audio ms, total ms, frames/s, audio s,
//!                                                eos (1/0) */);                    // frames, or -1
//!   int     tf_drama3_speak_wav(void* h, const char* text, const char* ref_wav, const char* ref_text,
//!                               const char* out_wav, float temperature, int top_k, float top_p, uint64_t seed,
//!                               int max_frames, double* stats_out /* [6] */);     // frames, or -1
//! on_audio runs on the calling thread as each chunk is decoded (first chunk 4 frames = 186 ms, then growing to 16).
const std = @import("std");
const core = @import("core");
const d3 = @import("drama3");

const W = d3.config.frame_width;

const Handle = struct {
    threaded: std.Io.Threaded,
    engine: *d3.Engine,
    tok: core.tokenizer.Tokenizer,
    tts: ?*d3.tts.Tts = null,
};

var err_buf: [512]u8 = undefined;
var err_len: usize = 0;

fn fail(comptime f: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(&err_buf, f, args) catch err_buf[0..0];
    err_len = s.len;
}

export fn tf_drama3_error() [*:0]const u8 {
    if (err_len >= err_buf.len) err_len = err_buf.len - 1;
    err_buf[err_len] = 0;
    return @ptrCast(&err_buf);
}

export fn tf_drama3_open(model_dir: [*:0]const u8, max_ctx: c_int, max_frames: c_int) ?*anyopaque {
    return tf_drama3_open2(model_dir, null, max_ctx, max_frames);
}

export fn tf_drama3_open2(model_dir: [*:0]const u8, codec_path: ?[*:0]const u8, max_ctx: c_int, max_frames: c_int) ?*anyopaque {
    const gpa = std.heap.c_allocator;
    const h = gpa.create(Handle) catch return null;
    h.threaded = .init(gpa, .{});
    const io = h.threaded.io();
    const dir = std.mem.span(model_dir);
    var o: d3.Options = .{};
    if (max_ctx > 0) o.max_ctx = @intCast(max_ctx);
    if (max_frames > 0) o.max_frames = @intCast(max_frames);
    h.engine = d3.Engine.create(gpa, io, dir, o) catch |e| {
        fail("tf_drama3_open: {s}", .{@errorName(e)});
        h.threaded.deinit();
        gpa.destroy(h);
        return null;
    };
    const tpath = std.fs.path.join(gpa, &.{ dir, "tokenizer.json" }) catch {
        h.engine.destroy();
        h.threaded.deinit();
        gpa.destroy(h);
        return null;
    };
    defer gpa.free(tpath);
    h.tok = core.tokenizer.loadTokenizer(io, gpa, tpath) catch |e| {
        fail("tf_drama3_open: tokenizer: {s}", .{@errorName(e)});
        h.engine.destroy();
        h.threaded.deinit();
        gpa.destroy(h);
        return null;
    };
    h.tts = null;
    if (codec_path) |cp| {
        h.tts = d3.tts.Tts.create(gpa, io, h.engine, std.mem.span(cp), 32) catch |e| {
            fail("tf_drama3_open: codec: {s}", .{@errorName(e)});
            h.tok.deinit();
            h.engine.destroy();
            h.threaded.deinit();
            gpa.destroy(h);
            return null;
        };
    }
    return h;
}

export fn tf_drama3_close(p: ?*anyopaque) void {
    const h: *Handle = @ptrCast(@alignCast(p orelse return));
    if (h.tts) |t| t.destroy();
    h.tok.deinit();
    h.engine.destroy();
    h.threaded.deinit();
    std.heap.c_allocator.destroy(h);
}

export fn tf_drama3_prompt(p: ?*anyopaque, text: [*:0]const u8, ref_text: ?[*:0]const u8, ref_codes: ?[*]const i32, ref_frames: c_int, rows_out: [*]i32, max_rows: c_int) c_int {
    const h: *Handle = @ptrCast(@alignCast(p orelse return -1));
    const gpa = std.heap.c_allocator;
    var refs: [1]d3.prompt.Reference = undefined;
    var n: usize = 0;
    if (ref_codes) |codes| if (ref_frames > 0) {
        const cs: [*]const [10]i32 = @ptrCast(codes);
        refs[0] = .{ .transcript = if (ref_text) |t| std.mem.span(t) else "", .codes = cs[0..@intCast(ref_frames)] };
        n = 1;
    };
    const rows = d3.prompt.build(gpa, &h.tok, std.mem.span(text), refs[0..n]) catch |e| {
        fail("tf_drama3_prompt: {s}", .{@errorName(e)});
        return -1;
    };
    defer gpa.free(rows);
    if (rows.len > @as(usize, @intCast(max_rows))) {
        fail("tf_drama3_prompt: {d} rows, room for {d}", .{ rows.len, max_rows });
        return -1;
    }
    const flat = std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(rows));
    @memcpy(rows_out[0..flat.len], flat);
    return @intCast(rows.len);
}

export fn tf_drama3_generate(p: ?*anyopaque, rows: [*]const i32, n_rows: c_int, temperature: f32, top_k: c_int, top_p: f32, seed: u64, max_frames: c_int, frames_out: [*]i32, stats_out: ?[*]f64) c_int {
    const h: *Handle = @ptrCast(@alignCast(p orelse return -1));
    const rs: [*]const [W]i32 = @ptrCast(rows);
    const smp: d3.Sampling = .{ .temperature = temperature, .top_k = top_k, .top_p = top_p, .seed = seed };
    const res = h.engine.generate(h.threaded.io(), rs[0..@intCast(n_rows)], smp, @intCast(max_frames), true, 8) catch |e| {
        fail("tf_drama3_generate: {s}", .{@errorName(e)});
        return -1;
    };
    defer std.heap.c_allocator.free(res.frames.ptr[0..res.frames.len]);
    const flat = std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(res.frames));
    @memcpy(frames_out[0..flat.len], flat);
    if (stats_out) |s| {
        s[0] = res.prefill_ms;
        s[1] = res.first_frame_ms;
        s[2] = res.decode_ms;
        s[3] = @as(f64, @floatFromInt(res.frames.len)) / @max(res.decode_ms / 1e3, 1e-9);
    }
    return @intCast(res.frames.len);
}

fn ttsOf(p: ?*anyopaque) ?*Handle {
    const h: *Handle = @ptrCast(@alignCast(p orelse return null));
    if (h.tts == null) {
        fail("no codec: open the handle with tf_drama3_open2 and a codec path", .{});
        return null;
    }
    return h;
}

fn putCodes(codes: []const [10]i32, out: [*]i32, max_frames: c_int) c_int {
    defer std.heap.c_allocator.free(codes);
    if (codes.len > @as(usize, @intCast(max_frames))) {
        fail("reference of {d} frames, room for {d}", .{ codes.len, max_frames });
        return -1;
    }
    const flat = std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(codes));
    @memcpy(out[0..flat.len], flat);
    return @intCast(codes.len);
}

export fn tf_drama3_encode_wav(p: ?*anyopaque, wav_path: [*:0]const u8, codes_out: [*]i32, max_frames: c_int) c_int {
    const h = ttsOf(p) orelse return -1;
    const gpa = std.heap.c_allocator;
    const bytes = std.Io.Dir.cwd().readFileAlloc(h.threaded.io(), std.mem.span(wav_path), gpa, .limited(1 << 30)) catch |e| {
        fail("tf_drama3_encode_wav: {s}", .{@errorName(e)});
        return -1;
    };
    defer gpa.free(bytes);
    const codes = h.tts.?.encodeWav(bytes) catch |e| {
        fail("tf_drama3_encode_wav: {s}", .{@errorName(e)});
        return -1;
    };
    return putCodes(codes, codes_out, max_frames);
}

export fn tf_drama3_encode_pcm(p: ?*anyopaque, pcm: [*]const f32, n: i64, rate: c_int, codes_out: [*]i32, max_frames: c_int) c_int {
    const h = ttsOf(p) orelse return -1;
    const codes = h.tts.?.encodeSamples(pcm[0..@intCast(n)], @intCast(rate)) catch |e| {
        fail("tf_drama3_encode_pcm: {s}", .{@errorName(e)});
        return -1;
    };
    return putCodes(codes, codes_out, max_frames);
}

export fn tf_drama3_decode(p: ?*anyopaque, frames: [*]const i32, n: c_int, pcm_out: [*]f32) i64 {
    const h = ttsOf(p) orelse return -1;
    const fs: [*]const [W]i32 = @ptrCast(frames);
    const nn: usize = @intCast(n);
    h.tts.?.decodeAll(fs[0..nn], pcm_out[0 .. nn * 2048]) catch |e| {
        fail("tf_drama3_decode: {s}", .{@errorName(e)});
        return -1;
    };
    return @intCast(nn * 2048);
}

const Callback = struct {
    f: *const fn (user: ?*anyopaque, pcm: [*]const f32, n: c_int) callconv(.c) void,
    user: ?*anyopaque,
    fn put(ctx: *anyopaque, x: []const f32) anyerror!void {
        const cb: *Callback = @ptrCast(@alignCast(ctx));
        cb.f(cb.user, x.ptr, @intCast(x.len));
    }
};

fn stats(s: d3.tts.Stats, out: ?[*]f64) void {
    const o = out orelse return;
    o[0] = s.prefill_ms;
    o[1] = s.first_audio_ms;
    o[2] = s.total_ms;
    o[3] = s.fps();
    o[4] = s.audio_s;
    o[5] = if (s.eos) 1 else 0;
}

export fn tf_drama3_speak(p: ?*anyopaque, rows: [*]const i32, n_rows: c_int, temperature: f32, top_k: c_int, top_p: f32, seed: u64, max_frames: c_int, first_chunk: c_int, chunk: c_int, on_audio: ?*const fn (user: ?*anyopaque, pcm: [*]const f32, n: c_int) callconv(.c) void, user: ?*anyopaque, frames_out: ?[*]i32, stats_out: ?[*]f64) c_int {
    const h = ttsOf(p) orelse return -1;
    const rs: [*]const [W]i32 = @ptrCast(rows);
    var cb: Callback = .{ .f = on_audio orelse &noop, .user = user };
    const smp: d3.Sampling = .{ .temperature = temperature, .top_k = top_k, .top_p = top_p, .seed = seed };
    const so: d3.tts.StreamOptions = .{ .first_chunk = if (first_chunk > 0) @intCast(first_chunk) else 4, .chunk = if (chunk > 0) @intCast(chunk) else 16 };
    const r = h.tts.?.speak(h.threaded.io(), rs[0..@intCast(n_rows)], smp, @intCast(max_frames), so, .{ .ctx = &cb, .audio = Callback.put }) catch |e| {
        fail("tf_drama3_speak: {s}", .{@errorName(e)});
        return -1;
    };
    defer std.heap.c_allocator.free(r.frames.ptr[0..r.frames.len]);
    if (frames_out) |fo| {
        const flat = std.mem.bytesAsSlice(i32, std.mem.sliceAsBytes(r.frames));
        @memcpy(fo[0..flat.len], flat);
    }
    stats(r.stats, stats_out);
    return @intCast(r.frames.len);
}

fn noop(_: ?*anyopaque, _: [*]const f32, _: c_int) callconv(.c) void {}

const WavSink = struct {
    gpa: std.mem.Allocator,
    pcm: std.ArrayList(f32) = .empty,
    fn put(ctx: *anyopaque, x: []const f32) anyerror!void {
        const w: *WavSink = @ptrCast(@alignCast(ctx));
        try w.pcm.appendSlice(w.gpa, x);
    }
};

export fn tf_drama3_speak_wav(p: ?*anyopaque, text: [*:0]const u8, ref_wav: ?[*:0]const u8, ref_text: ?[*:0]const u8, out_wav: [*:0]const u8, temperature: f32, top_k: c_int, top_p: f32, seed: u64, max_frames: c_int, stats_out: ?[*]f64) c_int {
    const h = ttsOf(p) orelse return -1;
    const gpa = std.heap.c_allocator;
    const io = h.threaded.io();
    var refs: [1]d3.prompt.Reference = undefined;
    var nref: usize = 0;
    var codes: ?[][10]i32 = null;
    defer if (codes) |c| gpa.free(c);
    if (ref_wav) |rw| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(rw), gpa, .limited(1 << 30)) catch |e| {
            fail("tf_drama3_speak_wav: reference: {s}", .{@errorName(e)});
            return -1;
        };
        defer gpa.free(bytes);
        codes = h.tts.?.encodeWav(bytes) catch |e| {
            fail("tf_drama3_speak_wav: encode: {s}", .{@errorName(e)});
            return -1;
        };
        refs[0] = .{ .transcript = if (ref_text) |t| std.mem.span(t) else "", .codes = codes.? };
        nref = 1;
    }
    const rows = d3.prompt.build(gpa, &h.tok, std.mem.span(text), refs[0..nref]) catch |e| {
        fail("tf_drama3_speak_wav: prompt: {s}", .{@errorName(e)});
        return -1;
    };
    defer gpa.free(rows);
    var sink: WavSink = .{ .gpa = gpa };
    defer sink.pcm.deinit(gpa);
    const smp: d3.Sampling = .{ .temperature = temperature, .top_k = top_k, .top_p = top_p, .seed = seed };
    const r = h.tts.?.speak(io, rows, smp, @intCast(max_frames), .{}, .{ .ctx = &sink, .audio = WavSink.put }) catch |e| {
        fail("tf_drama3_speak_wav: {s}", .{@errorName(e)});
        return -1;
    };
    defer gpa.free(r.frames.ptr[0..r.frames.len]);
    const bytes = d3.wav.encode16(gpa, sink.pcm.items, 44100) catch return -1;
    defer gpa.free(bytes);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(out_wav), .data = bytes }) catch |e| {
        fail("tf_drama3_speak_wav: write: {s}", .{@errorName(e)});
        return -1;
    };
    stats(r.stats, stats_out);
    return @intCast(r.frames.len);
}
