//! drama3-generate: Fish Audio Drama 3 on one CUDA GPU (fish_drama3), text (and a voice reference) in, audio out.
//!
//!   text -> WAV (no Python; the codec from tools/drama3/codec_convert.py):
//!   drama3-generate --model DIR --codec CODEC.safetensors --text "..." --wav OUT.wav
//!                   [--ref-audio REF.wav --ref-text "..."] [--ref-codes FILE] [--pcm-out PATH|-]
//!                   [--temperature 1.0] [--top-k 30] [--top-p 0.9] [--seed 1] [--greedy] [--max-frames 1500]
//!                   [--first-chunk 4] [--chunk 16] [--chunks-log 1] [--out FRAMES] [--rows-out FILE]
//!   one process, many requests (JSON lines on stdin, one JSON result a line):
//!   drama3-generate --model DIR --codec CODEC --serve 1
//!   audio -> codes:  drama3-generate --codec CODEC --encode OUT.codes --ref-audio REF.wav
//!   checks: --check DUMPDIR [--check-frames 40]  /  --codec-check REF --codec CODEC --frames F [--ref-audio WAV]
//!   engine knobs: [--max-ctx 4096] [--max-rows 1024] [--threads 16] [--repeat N] [--graphs 1] [--split 1] [--pdl 1]
//!                 [--fp8-act 1] [--fast-sampler 1] [--fp8-dense 0] [--fp8-tower 0] [--codec-tf32 0]
//!
//! Audio: 16-bit PCM mono 44.1 kHz WAV (--wav, header patched at the end); --pcm-out streams raw s16le 44.1 kHz mono
//! as chunks decode ("-" = stdout). The last stderr line is `RESULT drama3 OK frames=.. audio_s=.. first_audio_ms=..`.
//! Reference WAVs: PCM 8/16/24/32-bit or float32, any rate and channel count (mixed to mono, resampled to 44.1 kHz as
//! torchaudio does). --rows: prompt rows int32 [n, 11]; --ref-codes: int32 [frames, 10]; --out: frames int32 [n, 11].
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const d3 = @import("drama3");

const W = d3.config.frame_width;
const H = d3.config.hidden;

const Args = struct {
    model: []const u8 = "",
    text: ?[]const u8 = null,
    ref_text: ?[]const u8 = null,
    ref_codes: ?[]const u8 = null,
    rows: ?[]const u8 = null,
    rows_out: ?[]const u8 = null,
    out: ?[]const u8 = null,
    check: ?[]const u8 = null,
    check_frames: usize = 40,
    smp: d3.Sampling = .{},
    max_frames: usize = 1500,
    graphs: bool = true,
    sync_every: usize = 8,
    opts: d3.Options = .{},
    repeat: usize = 1,
    codec: ?[]const u8 = null,
    codec_check: ?[]const u8 = null,
    frames: ?[]const u8 = null,
    ref_audio: ?[]const u8 = null,
    first_chunk: usize = 4,
    chunk: usize = 16,
    wav: ?[]const u8 = null,
    pcm: ?[]const u8 = null,
    encode_out: ?[]const u8 = null,
    serve: bool = false,
    verbose_chunks: bool = false,
    codec_tf32: bool = false,
};

const usage =
    \\usage: drama3-generate --model DIR --codec CODEC --text "..." --wav OUT.wav [--ref-audio REF.wav --ref-text "..."]
    \\       drama3-generate --model DIR --codec CODEC --serve 1        (JSON lines on stdin)
    \\       drama3-generate --codec CODEC --encode OUT.codes --ref-audio REF.wav
    \\       drama3-generate --model DIR --text "..." [--ref-text "..." --ref-codes FILE] [--rows FILE]
    \\                       [--temperature 1.0] [--top-k 30] [--top-p 0.9] [--seed 1] [--greedy]
    \\                       [--max-frames 1500] [--graphs 1] [--sync-every 8] [--out FILE] [--rows-out FILE]
    \\                       [--max-ctx 4096] [--max-rows 1024] [--threads 16] [--repeat N] [--fp8-act 1]
    \\       drama3-generate --model DIR --check DUMPDIR [--check-frames 40]
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (std.mem.eql(u8, k, "--greedy")) {
            a.smp.temperature = 0;
            a.smp.top_k = 1;
            continue;
        }
        if (i + 1 >= argv.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = argv[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--model")) a.model = v else if (std.mem.eql(u8, k, "--text")) a.text = v else if (std.mem.eql(u8, k, "--ref-text")) a.ref_text = v else if (std.mem.eql(u8, k, "--ref-codes")) a.ref_codes = v else if (std.mem.eql(u8, k, "--rows")) a.rows = v else if (std.mem.eql(u8, k, "--rows-out")) a.rows_out = v else if (std.mem.eql(u8, k, "--out")) a.out = v else if (std.mem.eql(u8, k, "--check")) a.check = v else if (std.mem.eql(u8, k, "--check-frames")) a.check_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--temperature")) a.smp.temperature = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--top-k")) a.smp.top_k = try std.fmt.parseInt(i32, v, 10) else if (std.mem.eql(u8, k, "--top-p")) a.smp.top_p = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--seed")) a.smp.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--max-frames")) a.max_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--graphs")) a.graphs = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--sync-every")) a.sync_every = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-ctx")) a.opts.max_ctx = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-rows")) a.opts.max_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--threads")) a.opts.threads = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--repeat")) a.repeat = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--wav")) a.wav = v else if (std.mem.eql(u8, k, "--pcm-out")) a.pcm = v else if (std.mem.eql(u8, k, "--encode")) a.encode_out = v else if (std.mem.eql(u8, k, "--chunk")) a.chunk = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--serve")) a.serve = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--chunks-log")) a.verbose_chunks = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--codec")) a.codec = v else if (std.mem.eql(u8, k, "--codec-check")) a.codec_check = v else if (std.mem.eql(u8, k, "--frames")) a.frames = v else if (std.mem.eql(u8, k, "--ref-audio")) a.ref_audio = v else if (std.mem.eql(u8, k, "--first-chunk")) a.first_chunk = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--codec-tf32")) a.codec_tf32 = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--fp8-tower")) a.opts.fp8_tower = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--fp8-dense")) a.opts.fp8_dense = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--fast-sampler")) a.opts.fast_sampler = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--pdl")) a.opts.pdl = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--split")) a.opts.split = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--exact-attn")) a.opts.exact_attn_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--fp8-act")) a.opts.fp8_act = !std.mem.eql(u8, v, "0") else {
            std.debug.print("unknown flag {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.codec_check) |ref| return codecCheck(gpa, io, a, ref);
    if (a.model.len == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    a.opts.max_frames = @max(a.opts.max_frames, a.max_frames);
    if (a.check) |dir| return check(gpa, io, a, dir);

    // audio file -> codes (the codec alone)
    if (a.encode_out) |out| {
        const wp = a.ref_audio orelse return error.RefAudioRequired;
        var driver = try cuda.Driver.open();
        var ctx = try cuda.Context.init(&driver, 0);
        const stream = try cuda.Stream.init(&driver, true);
        const k = try gpa.create(d3.kernels.Kernels);
        k.* = try d3.kernels.Kernels.load(&driver, &ctx);
        k.rebind();
        const c = try d3.codec.Codec.load(gpa, io, &driver, k, a.codec orelse return error.CodecRequired, .{});
        defer c.destroy();
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, wp, gpa, .limited(1 << 30));
        const au = try d3.wav.parse(gpa, bytes);
        const x44 = try d3.wav.resample(gpa, au.samples, au.rate, d3.codec.sample_rate);
        const t0 = d3.engine.nowNs(io);
        const codes = try c.encode(stream, x44);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = std.mem.sliceAsBytes(codes) });
        std.debug.print("RESULT drama3 encode OK frames={d} seconds={d:.2} ms={d:.1} out={s}\n", .{ codes.len, @as(f64, @floatFromInt(au.samples.len)) / @as(f64, @floatFromInt(au.rate)), @as(f64, @floatFromInt(d3.engine.nowNs(io) - t0)) / 1e6, out });
        return 0;
    }

    const tpath = try std.fs.path.join(arena, &.{ a.model, "tokenizer.json" });
    var tok = try core.tokenizer.loadTokenizer(io, gpa, tpath);
    defer tok.deinit();
    const t_load = d3.engine.nowNs(io);
    const e = try d3.Engine.create(gpa, io, a.model, a.opts);
    defer e.destroy();
    var tts: ?*d3.tts.Tts = null;
    if (a.codec) |cp| tts = try d3.tts.Tts.create(gpa, io, e, cp, @max(a.chunk, a.first_chunk));
    defer if (tts) |t| t.destroy();
    const load_s = @as(f64, @floatFromInt(d3.engine.nowNs(io) - t_load)) / 1e9;
    std.log.info("drama3: loaded in {d:.1} s (codec {s})", .{ load_s, if (tts != null) "native" else "none" });
    if (a.serve) return serve(gpa, io, a, e, tts orelse return error.CodecRequired, &tok);
    const req: Request = .{ .text = a.text, .rows = a.rows, .ref_text = a.ref_text, .ref_codes = a.ref_codes, .ref_audio = a.ref_audio, .wav = a.wav, .pcm = a.pcm, .frames_out = a.out, .rows_out = a.rows_out, .smp = a.smp, .max_frames = a.max_frames };
    var r: usize = 0;
    while (r < a.repeat) : (r += 1) try runRequest(gpa, io, a, e, tts, &tok, req, false);
    return 0;
}

const Request = struct {
    text: ?[]const u8 = null,
    rows: ?[]const u8 = null,
    ref_text: ?[]const u8 = null,
    ref_codes: ?[]const u8 = null,
    ref_audio: ?[]const u8 = null,
    wav: ?[]const u8 = null,
    pcm: ?[]const u8 = null, // raw s16le 44.1 kHz mono as it streams ("-" = stdout)
    frames_out: ?[]const u8 = null,
    rows_out: ?[]const u8 = null,
    smp: d3.Sampling = .{},
    max_frames: usize = 1500,
};

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(p: [*]const u8, size: usize, n: usize, f: *anyopaque) usize;
extern "c" fn fseek(f: *anyopaque, off: c_long, whence: c_int) c_int;
extern "c" fn fflush(f: ?*anyopaque) c_int;
extern "c" fn fclose(f: *anyopaque) c_int;
extern "c" fn fdopen(fd: c_int, mode: [*:0]const u8) ?*anyopaque;

/// Where the samples go as they stream: a WAV file (header patched at the end) and / or raw PCM.
const AudioOut = struct {
    gpa: std.mem.Allocator,
    wav: ?*anyopaque = null,
    pcm: ?*anyopaque = null,
    samples: usize = 0,
    chunks: usize = 0,
    t0: i128 = 0,
    io: std.Io,
    verbose: bool,
    buf: []u8 = &.{},

    fn open(gpa: std.mem.Allocator, io: std.Io, wav_path: ?[]const u8, pcm_path: ?[]const u8, verbose: bool) !AudioOut {
        var o: AudioOut = .{ .gpa = gpa, .io = io, .verbose = verbose, .t0 = d3.engine.nowNs(io) };
        if (wav_path) |p| {
            const z = try gpa.dupeSentinel(u8, p, 0);
            defer gpa.free(z);
            o.wav = fopen(z.ptr, "wb") orelse return error.WavOpen;
            var h: [44]u8 = undefined;
            d3.wav.header(&h, 44100, 0x7FFFFFF0 / 2);
            _ = fwrite(&h, 1, 44, o.wav.?);
        }
        if (pcm_path) |p| {
            if (std.mem.eql(u8, p, "-")) o.pcm = fdopen(1, "wb") else {
                const z = try gpa.dupeSentinel(u8, p, 0);
                defer gpa.free(z);
                o.pcm = fopen(z.ptr, "wb") orelse return error.PcmOpen;
            }
        }
        return o;
    }

    fn put(ctx: *anyopaque, x: []const f32) anyerror!void {
        const o: *AudioOut = @ptrCast(@alignCast(ctx));
        if (o.buf.len < x.len * 2) {
            o.gpa.free(o.buf);
            o.buf = try o.gpa.alloc(u8, x.len * 2);
        }
        const b = o.buf[0 .. x.len * 2];
        d3.wav.pcm16(x, b);
        if (o.wav) |f| _ = fwrite(b.ptr, 1, b.len, f);
        if (o.pcm) |f| {
            _ = fwrite(b.ptr, 1, b.len, f);
            _ = fflush(f);
        }
        o.samples += x.len;
        o.chunks += 1;
        if (o.verbose) std.debug.print("CHUNK {d} samples={d} total_s={d:.3} at_ms={d:.1}\n", .{ o.chunks, x.len, @as(f64, @floatFromInt(o.samples)) / 44100.0, @as(f64, @floatFromInt(d3.engine.nowNs(o.io) - o.t0)) / 1e6 });
    }

    fn close(o: *AudioOut) void {
        if (o.wav) |f| {
            var h: [44]u8 = undefined;
            d3.wav.header(&h, 44100, o.samples);
            _ = fseek(f, 0, 0);
            _ = fwrite(&h, 1, 44, f);
            _ = fclose(f);
        }
        if (o.pcm) |f| _ = fflush(f);
        o.gpa.free(o.buf);
    }
};

fn runRequest(gpa: std.mem.Allocator, io: std.Io, a: Args, e: *d3.Engine, tts: ?*d3.tts.Tts, tok: *const core.tokenizer.Tokenizer, q: Request, json: bool) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const t_req = d3.engine.nowNs(io);
    // prompt rows (the reference encoded natively when it is audio)
    var rows: []const [W]i32 = undefined;
    var ref_frames: usize = 0;
    var encode_ms: f64 = 0;
    if (q.rows) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, ar, .limited(1 << 30));
        if (bytes.len % (W * 4) != 0) return error.BadRows;
        rows = @alignCast(std.mem.bytesAsSlice([W]i32, bytes));
    } else {
        const text = q.text orelse return error.TextRequired;
        var refs: [1]d3.prompt.Reference = undefined;
        var nref: usize = 0;
        if (q.ref_codes) |p| {
            refs[0] = .{ .transcript = q.ref_text orelse return error.RefTextRequired, .codes = try readCodes(ar, io, p) };
            nref = 1;
        } else if (q.ref_audio) |p| {
            const t = tts orelse return error.CodecRequired;
            const e0 = d3.engine.nowNs(io);
            const codes = try t.encodeWav(try std.Io.Dir.cwd().readFileAlloc(io, p, ar, .limited(1 << 30)));
            encode_ms = @as(f64, @floatFromInt(d3.engine.nowNs(io) - e0)) / 1e6;
            refs[0] = .{ .transcript = q.ref_text orelse return error.RefTextRequired, .codes = codes };
            nref = 1;
        }
        if (nref > 0) ref_frames = refs[0].codes.len;
        rows = try d3.prompt.build(ar, tok, text, refs[0..nref]);
    }
    if (q.rows_out) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = std.mem.sliceAsBytes(rows) });
    var frames: []const [W]i32 = undefined;
    var st: d3.tts.Stats = undefined;
    if (tts) |t| {
        var out = try AudioOut.open(ar, io, q.wav, q.pcm, a.verbose_chunks);
        out.t0 = t_req;
        defer out.close();
        const r = try t.speak(io, rows, q.smp, q.max_frames, .{ .first_chunk = a.first_chunk, .chunk = a.chunk }, .{ .ctx = &out, .audio = AudioOut.put });
        frames = r.frames;
        st = r.stats;
    } else {
        const res = try e.generate(io, rows, q.smp, q.max_frames, a.graphs, a.sync_every);
        frames = res.frames;
        st = .{ .frames = res.frames.len, .eos = res.eos, .prompt_rows = rows.len, .prefill_ms = res.prefill_ms, .first_audio_ms = 0, .total_ms = res.prefill_ms + res.decode_ms, .audio_s = @as(f64, @floatFromInt(res.frames.len)) * 2048.0 / 44100.0 };
    }
    defer gpa.free(frames.ptr[0..frames.len]);
    if (q.frames_out) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = std.mem.sliceAsBytes(frames) });
    const total_ms = @as(f64, @floatFromInt(d3.engine.nowNs(io) - t_req)) / 1e6;
    const mu = cuda.usage(false);
    if (json) {
        std.debug.print("{{\"ok\":true,\"frames\":{d},\"eos\":{},\"audio_s\":{d:.3},\"prompt_rows\":{d},\"ref_frames\":{d},\"encode_ms\":{d:.1},\"prefill_ms\":{d:.1},\"first_audio_ms\":{d:.1},\"total_ms\":{d:.1},\"fps\":{d:.2},\"wav\":\"{s}\"}}\n", .{ st.frames, st.eos, st.audio_s, rows.len, ref_frames, encode_ms, st.prefill_ms, st.first_audio_ms + encode_ms, total_ms, st.fps(), q.wav orelse "" });
    } else {
        std.debug.print("RESULT drama3 OK frames={d} eos={} audio_s={d:.2} prompt_rows={d} ref_frames={d} encode_ms={d:.1} prefill_ms={d:.1} first_audio_ms={d:.1} total_ms={d:.1} fps={d:.2} rtf={d:.2} device_peak_gb={d:.2}\n", .{ st.frames, st.eos, st.audio_s, rows.len, ref_frames, encode_ms, st.prefill_ms, st.first_audio_ms + encode_ms, total_ms, st.fps(), st.fps() / 21.533, @as(f64, @floatFromInt(mu.peak)) / 1e9 });
    }
}

extern "c" fn getline(line: *?[*]u8, n: *usize, f: *anyopaque) isize;
extern "c" fn free(p: ?*anyopaque) void;

/// --serve: one JSON request a line on stdin, one JSON result a line on stdout (the engine and codec stay loaded).
/// {"text": "...", "wav": "out.wav", "ref_audio": "ref.wav", "ref_text": "...", "ref_codes": "f", "pcm": "f",
///  "temperature": 1.0, "top_k": 30, "top_p": 0.9, "seed": 1, "max_frames": 1500}
fn serve(gpa: std.mem.Allocator, io: std.Io, a: Args, e: *d3.Engine, tts: *d3.tts.Tts, tok: *const core.tokenizer.Tokenizer) !u8 {
    const in = fdopen(0, "r") orelse return error.Stdin;
    std.debug.print("{{\"ready\":true}}\n", .{});
    var line: ?[*]u8 = null;
    var cap: usize = 0;
    defer free(line);
    while (true) {
        const n = getline(&line, &cap, in);
        if (n <= 0) break;
        const text = std.mem.trim(u8, line.?[0..@intCast(n)], " \r\n");
        if (text.len == 0) continue;
        var parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch {
            std.debug.print("{{\"ok\":false,\"error\":\"bad json\"}}\n", .{});
            continue;
        };
        defer parsed.deinit();
        const o = parsed.value.object;
        var q: Request = .{ .smp = a.smp, .max_frames = a.max_frames };
        const str = struct {
            fn get(m: std.json.ObjectMap, k: []const u8) ?[]const u8 {
                const v = m.get(k) orelse return null;
                return if (v == .string) v.string else null;
            }
            fn num(m: std.json.ObjectMap, k: []const u8) ?f64 {
                const v = m.get(k) orelse return null;
                return switch (v) {
                    .float => |x| x,
                    .integer => |x| @floatFromInt(x),
                    else => null,
                };
            }
        };
        q.text = str.get(o, "text");
        q.wav = str.get(o, "wav");
        q.pcm = str.get(o, "pcm");
        q.ref_audio = str.get(o, "ref_audio");
        q.ref_text = str.get(o, "ref_text");
        q.ref_codes = str.get(o, "ref_codes");
        q.frames_out = str.get(o, "frames_out");
        if (str.num(o, "temperature")) |x| q.smp.temperature = @floatCast(x);
        if (str.num(o, "top_k")) |x| q.smp.top_k = @intFromFloat(x);
        if (str.num(o, "top_p")) |x| q.smp.top_p = @floatCast(x);
        if (str.num(o, "seed")) |x| q.smp.seed = @intFromFloat(x);
        if (str.num(o, "max_frames")) |x| q.max_frames = @intFromFloat(x);
        runRequest(gpa, io, a, e, tts, tok, q, true) catch |er| {
            std.debug.print("{{\"ok\":false,\"error\":\"{s}\"}}\n", .{@errorName(er)});
        };
    }
    return 0;
}

/// Reference codes: raw int32 [frames, 10].
fn readCodes(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]const [10]i32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    if (!std.mem.endsWith(u8, path, ".npy")) {
        if (bytes.len % 40 != 0) return error.BadCodes;
        return @alignCast(std.mem.bytesAsSlice([10]i32, bytes));
    }
    return error.NpyCodesUnsupported; // tools/drama3/encode_reference.py writes the raw form
}

// ------------------------------------------------------------------------------------------------ oracle check

const Check = struct {
    gpa: std.mem.Allocator,
    e: *d3.Engine,
    ref: *core.safetensors.File,
    worst: f64 = 1.0,
    n: usize = 0,
    fails: usize = 0,
    // frame 0's heads (0: the candidate logits, 1 + i: codebook head i): both sides' top two
    tops: [10]?[2]Top2 = @splat(null),

    fn refF32(c: *Check, name: []const u8, rows: usize, width: usize) !?[]f32 {
        const t = c.ref.get(name) orelse return null;
        const n = t.numel();
        // the oracle's decode0 rows are [width]; prompt rows [rows, width]: compare the last `rows * width` values
        if (n < rows * width) return null;
        const out = try c.gpa.alloc(f32, rows * width);
        const skip = n - rows * width;
        switch (t.dtype) {
            .bf16 => {
                const v = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, t.bytes));
                for (out, 0..) |*o, j| o.* = @bitCast(@as(u32, v[skip + j]) << 16);
            },
            .f32 => {
                const v = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, t.bytes));
                for (out, 0..) |*o, j| o.* = v[skip + j];
            },
            else => return null,
        }
        return out;
    }

    fn probe(ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, is_f32: bool) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        const mine = try c.gpa.alloc(f32, rows * width);
        defer c.gpa.free(mine);
        if (is_f32) {
            try c.e.download(ptr, std.mem.sliceAsBytes(mine));
        } else {
            const raw = try c.gpa.alloc(u16, rows * width);
            defer c.gpa.free(raw);
            try c.e.download(ptr, std.mem.sliceAsBytes(raw));
            for (mine, raw) |*m, r| m.* = @bitCast(@as(u32, r) << 16);
        }
        if (std.mem.startsWith(u8, name, "dbg.")) {
            for ([_]usize{ 100, 140, 147, 149, 153 }) |r| if (r < rows) {
                var ss: f64 = 0;
                for (mine[r * width ..][0..width]) |x| ss += @as(f64, x) * x;
                std.debug.print("  {s} row {d} |x| {d:.3}\n", .{ name, r, @sqrt(ss) });
            };
            return;
        }
        if (std.mem.startsWith(u8, name, "router.")) {
            // the 8th / 9th routing probabilities of each row: a near tie flips an expert
            for (0..rows) |r| {
                const lg = mine[r * width + 1024 ..][0..256];
                var mx: f32 = -std.math.inf(f32);
                for (lg) |x| mx = @max(mx, x);
                var p: [256]f32 = undefined;
                var sum: f32 = 0;
                for (lg, 0..) |x, j| {
                    p[j] = @exp(x - mx);
                    sum += p[j];
                }
                std.mem.sort(f32, &p, {}, std.sort.desc(f32));
                const gap = (p[7] - p[8]) / sum;
                const li = std.fmt.parseInt(usize, name[12..14], 10) catch 0;
                const want_row = if (std.c.getenv("D3_ROW")) |w| std.fmt.parseInt(usize, std.mem.span(w), 10) catch rows - 2 else rows - 2;
                if (r == want_row and li >= 30) std.debug.print("  {s} row {d}: 8th / 9th routing probability {d:.5} / {d:.5} (gap {d:.5})\n", .{ name, r, p[7] / sum, p[8] / sum, gap });
            }
            return;
        }
        const ref = (try c.refF32(name, rows, width)) orelse {
            std.debug.print("  {s:<36} (no oracle tensor)\n", .{name});
            return;
        };
        defer c.gpa.free(ref);
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        var maxd: f64 = 0;
        var maxr: f64 = 0;
        for (mine, ref) |x, y| {
            dot += @as(f64, x) * y;
            na += @as(f64, x) * x;
            nb += @as(f64, y) * y;
            maxd = @max(maxd, @abs(@as(f64, x) - y));
            maxr = @max(maxr, @abs(@as(f64, y)));
        }
        // the worst row's cosine too
        var worst_row: f64 = 1;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
            }
            if (a2 > 0 and b2 > 0) worst_row = @min(worst_row, d / @sqrt(a2 * b2));
        }
        const cos = if (na > 0 and nb > 0) dot / @sqrt(na * nb) else 1.0;
        if (std.mem.startsWith(u8, name, "tower_decode0.head") or std.mem.eql(u8, name, "tower_decode0.candidate_logits")) {
            const hi: usize = if (name[18] >= '0' and name[18] <= '9') 1 + (name[18] - '0') else 0;
            c.tops[hi] = .{ top2(mine), top2(ref) };
        }
        c.n += 1;
        c.worst = @min(c.worst, worst_row);
        var wr: usize = 0;
        var wv: f64 = 2;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
            }
            const cr = if (a2 > 0 and b2 > 0) d / @sqrt(a2 * b2) else 1.0;
            if (cr < wv) {
                wv = cr;
                wr = r;
            }
            if (std.c.getenv("D3_ROW")) |want| if (std.fmt.parseInt(usize, std.mem.span(want), 10) catch 99999 == r) std.debug.print("    ROW {s} row {d} cos {d:.6}\n", .{ name, r, cr });
            if (cr < 0.99 and std.c.getenv("D3_ROWS") != null) std.debug.print("    {s} row {d} cos {d:.4}\n", .{ name, r, cr });
        }
        const ok = worst_row >= 0.999;
        if (!ok) c.fails += 1;
        std.debug.print("  {s:<36} rows {d:>4} cos {d:.6} worst-row {d:.6} (row {d}) max|d| {d:.4} (max|ref| {d:.2}) {s}\n", .{ name, rows, cos, worst_row, wr, maxd, maxr, if (ok) "ok" else "LOW" });
    }
};

fn check(gpa: std.mem.Allocator, io: std.Io, a: Args, dir: []const u8) !u8 {
    const vpath = try std.fs.path.join(gpa, &.{ dir, "vectors.safetensors" });
    var ref = try core.safetensors.File.open(gpa, io, vpath);
    defer ref.close(io);
    const pr = ref.get("prompt_rows") orelse return error.NoPromptRows;
    const n = pr.dim(0);
    if (pr.dim(1) != W or pr.dtype != .i64) return error.BadPromptRows;
    const rows = try gpa.alloc([W]i32, n);
    defer gpa.free(rows);
    const v = std.mem.bytesAsSlice(i64, @as([]align(1) const u8, pr.bytes));
    for (rows, 0..) |*r, j| for (r, 0..) |*x, c| {
        x.* = @intCast(v[j * W + c]);
    };
    var opts = a.opts;
    opts.max_rows = @max(opts.max_rows, n);
    const e = try d3.Engine.create(gpa, io, a.model, opts);
    defer e.destroy();
    var c: Check = .{ .gpa = gpa, .e = e, .ref = &ref };
    const probe: d3.engine.Probe = .{ .ctx = &c, .f = Check.probe };
    std.debug.print("drama3 check: {d} prompt rows from {s}\n", .{ n, vpath });
    e.probe_router = probe;
    try e.prefill(rows, probe);
    e.probe_router = null;
    if (ref.get("backbone_final_norm")) |hf| if (hf.dtype == .bf16 and hf.dim(0) == n) {
        std.debug.print("  -- the tower prefill again from the oracle's backbone_final_norm rows:\n", .{});
        try e.towerOnly(n, hf.bytes, probe);
        try e.prefill(rows, null); // restore my own state for the decode comparison
    };
    // frame 0 with the probe (tower_decode0.*), then greedy frames against the oracle's
    const smp: d3.Sampling = .{ .temperature = 0, .top_k = 1 };
    const oref = ref.get("frames_all") orelse ref.get("frames") orelse return error.NoFrames;
    const nref = oref.dim(0);
    const ov = std.mem.bytesAsSlice(i64, @as([]align(1) const u8, oref.bytes));
    const cand_ref = ref.get("decode.candidate_logits");
    const nc = e.w.n_cand;
    const mine_logits = try gpa.alloc(f32, nc);
    defer gpa.free(mine_logits);
    var same: usize = 0;
    var diverged = false;
    const want = @min(a.check_frames, nref + 1);
    var f: usize = 0;
    var got: [W]i32 = undefined;
    while (f < want) : (f += 1) {
        // my candidate logits for this frame (the step recomputes them; read them back after it)
        try e.frameStep(smp, false, if (f == 0) probe else null);
        try e.download(e.cand_logits, std.mem.sliceAsBytes(mine_logits));
        try e.download(e.frames_out + f * W * 4, std.mem.asBytes(&got));
        if (f >= nref) {
            std.debug.print("  frame {d}: mine {d} (the oracle stopped at {d} frames: eos expected)\n", .{ f, got[0], nref });
            if (got[0] == e.c.ids.eos) same += 1;
            break;
        }
        var match = true;
        for (0..W) |j| match = match and got[j] == ov[f * W + j];
        if (match) {
            same += 1;
            continue;
        }
        diverged = true;
        std.debug.print("  frame {d} differs: mine [", .{f});
        for (got) |x| std.debug.print(" {d}", .{x});
        std.debug.print(" ] oracle [", .{});
        for (0..W) |j| std.debug.print(" {d}", .{ov[f * W + j]});
        std.debug.print(" ]\n", .{});
        var col: usize = 0;
        while (col < W and got[col] == ov[f * W + col]) col += 1;
        const head: usize = if (col <= 1) 0 else col - 1;
        if (f == 0) if (c.tops[head]) |t| {
            std.debug.print("    first difference: column {d} ({s} {d}); mine top {d} ({d:.4}) second {d} ({d:.4}) margin {d:.4}; oracle top {d} ({d:.4}) second {d} ({d:.4}) margin {d:.4}\n", .{ col, if (head == 0) "candidate logits" else "codebook head", if (head == 0) 0 else head - 1, t[0].i1, t[0].v1, t[0].i2, t[0].v2, t[0].v1 - t[0].v2, t[1].i1, t[1].v1, t[1].i2, t[1].v2, t[1].v1 - t[1].v2 });
        };
        // the main token's margin on both sides (next frame's candidate logits)
        const mm = top2(mine_logits);
        std.debug.print("    my candidate logits: top {d} ({d:.4}) second {d} ({d:.4})\n", .{ mm.i1, mm.v1, mm.i2, mm.v2 });
        if (cand_ref) |cr| if (f < cr.dim(0)) {
            const rv = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, cr.bytes))[f * nc ..][0..nc];
            const rcopy = try gpa.alloc(f32, nc);
            defer gpa.free(rcopy);
            @memcpy(rcopy, rv);
            const rm = top2(rcopy);
            std.debug.print("    oracle candidate logits: top {d} ({d:.4}) second {d} ({d:.4}); mine at the oracle's top: {d:.4}\n", .{ rm.i1, rm.v1, rm.i2, rm.v2, mine_logits[rm.i1] });
        };
        break;
    }
    _ = &diverged;
    // teacher forcing: the oracle's frames fed back; my greedy picks against them, my decode rows' final-norm hidden
    // against the oracle's (decode.backbone_final_norm, its first dump-frames rows)
    var tf_agree: [W]usize = @splat(0);
    var tf_frames: usize = 0;
    var tf_worst: f64 = 1;
    {
        try e.prefill(rows, null);
        const dev = try cuda.DeviceBuffer.alloc(&e.driver, 2 * W * 4);
        var devb = dev;
        defer devb.free();
        const hidden_ref = ref.get("decode.backbone_final_norm");
        const hbuf = try gpa.alloc(u16, H);
        defer gpa.free(hbuf);
        const nf = @min(nref, a.check_frames * 5);
        for (0..nf) |fi| {
            if (hidden_ref) |hr| if (fi < hr.dim(0)) {
                try e.download(e.hlast, std.mem.sliceAsBytes(hbuf));
                const rv = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, hr.bytes))[fi * H ..][0..H];
                var d: f64 = 0;
                var a2: f64 = 0;
                var b2: f64 = 0;
                for (hbuf, rv) |x16, y16| {
                    const x: f64 = @as(f32, @bitCast(@as(u32, x16) << 16));
                    const y: f64 = @as(f32, @bitCast(@as(u32, y16) << 16));
                    d += x * y;
                    a2 += x * x;
                    b2 += y * y;
                }
                tf_worst = @min(tf_worst, d / @sqrt(a2 * b2));
            };
            var fr: [W]i32 = undefined;
            for (0..W) |j| fr[j] = @intCast(ov[fi * W + j]);
            try devb.upload(0, std.mem.asBytes(&fr));
            try e.frameForced(.{ .frame = dev.ptr, .picks = dev.ptr + W * 4 });
            var picks: [W]i32 = undefined;
            try e.download(dev.ptr + W * 4, std.mem.asBytes(&picks));
            for (0..W) |j| tf_agree[j] += @intFromBool(picks[j] == fr[j]);
            tf_frames += 1;
        }
        std.debug.print("  teacher-forced {d} frames: my greedy picks equal the oracle's: main {d}", .{ tf_frames, tf_agree[0] });
        for (2..W) |j| std.debug.print(" c{d} {d}", .{ j - 1, tf_agree[j] });
        std.debug.print("; decode-row final-norm hidden worst cos {d:.6} over the oracle's dumped rows\n", .{tf_worst});
    }
    const pass = c.fails == 0 and (same >= @min(20, want));
    var tf_total: usize = 0;
    for (tf_agree) |x| tf_total += x;
    std.debug.print("RESULT drama3 check {s}: {d} probes, {d} below cos 0.999 (worst row {d:.6}); greedy frames identical {d} of {d} (oracle made {d}); teacher-forced picks {d:.2}% of {d}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.worst, same, want, nref, 100.0 * @as(f64, @floatFromInt(tf_total - tf_agree[1])) / @as(f64, @floatFromInt(@max(1, tf_frames * (W - 1)))), tf_frames * (W - 1) });
    return if (pass) 0 else 1;
}

const Top2 = struct { i1: usize, v1: f32, i2: usize, v2: f32 };

fn top2(x: []const f32) Top2 {
    var r: Top2 = .{ .i1 = 0, .v1 = -std.math.inf(f32), .i2 = 0, .v2 = -std.math.inf(f32) };
    for (x, 0..) |v, j| {
        if (v > r.v1) {
            r.i2 = r.i1;
            r.v2 = r.v1;
            r.i1 = j;
            r.v1 = v;
        } else if (v > r.v2) {
            r.i2 = j;
            r.v2 = v;
        }
    }
    return r;
}

// ------------------------------------------------------------------------------------------------ codec check

fn refTensor(gpa: std.mem.Allocator, f: *const core.safetensors.File, name: []const u8) !?[]f32 {
    const t = f.get(name) orelse return null;
    if (t.dtype != .f32) return error.BadRef;
    const out = try gpa.alloc(f32, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn snrDb(ref: []const f32, got: []const f32) f64 {
    var sig: f64 = 0;
    var err: f64 = 0;
    for (ref, got) |r, g| {
        sig += @as(f64, r) * r;
        err += (@as(f64, r) - g) * (@as(f64, r) - g);
    }
    return 10.0 * std.math.log10(sig / @max(err, 1e-30));
}

/// --codec-check REF.safetensors (codec_convert.py --ref) --codec CODEC --frames FRAMES.int32 [--ref-audio WAV]:
/// the streamed decode (a first chunk of --first-chunk frames, then max_chunk) against the PyTorch codec's waveform
/// (SNR), and the reference clip's codes (my WAV reader, resampler and encoder) against its codes.
fn codecCheck(gpa: std.mem.Allocator, io: std.Io, a: Args, ref_path: []const u8) !u8 {
    var ref = try core.safetensors.File.open(gpa, io, ref_path);
    defer ref.close(io);
    var driver = try cuda.Driver.open();
    var ctx = try cuda.Context.init(&driver, 0);
    var stream = try cuda.Stream.init(&driver, true);
    const k = try gpa.create(d3.kernels.Kernels);
    k.* = try d3.kernels.Kernels.load(&driver, &ctx);
    k.rebind();
    const t0 = d3.engine.nowNs(io);
    const c = try d3.codec.Codec.load(gpa, io, &driver, k, a.codec orelse return error.CodecRequired, .{ .tf32_decode = a.codec_tf32 });
    defer c.destroy();
    std.debug.print("codec: {d:.2} GB of weights in {d:.1} s\n", .{ @as(f64, @floatFromInt(c.weight_bytes)) / 1e9, @as(f64, @floatFromInt(d3.engine.nowNs(io) - t0)) / 1e9 });
    var pass = true;
    if (a.frames) |fp| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fp, gpa, .limited(1 << 28));
        const nf = bytes.len / 44;
        var fb = try cuda.DeviceBuffer.alloc(&driver, bytes.len);
        defer fb.free();
        try fb.upload(0, bytes);
        var ab = try cuda.DeviceBuffer.alloc(&driver, nf * 2048 * 4);
        defer ab.free();
        for (0..2) |rep| {
            try c.reset(stream);
            try stream.synchronize();
            const s0 = d3.engine.nowNs(io);
            const first = @min(a.first_chunk, nf);
            try c.decode(stream, fb.ptr, first, ab.ptr);
            try stream.synchronize();
            const s1 = d3.engine.nowNs(io);
            try c.decode(stream, fb.ptr + first * 44, nf - first, ab.ptr + first * 2048 * 4);
            try stream.synchronize();
            const s2 = d3.engine.nowNs(io);
            const audio = try gpa.alloc(f32, nf * 2048);
            defer gpa.free(audio);
            try ab.download(0, std.mem.sliceAsBytes(audio));
            const secs = @as(f64, @floatFromInt(nf * 2048)) / 44100.0;
            std.debug.print("decode run {d}: {d} frames ({d:.2} s): first chunk of {d} in {d:.1} ms, all in {d:.1} ms = {d:.0}x real time\n", .{ rep, nf, secs, first, @as(f64, @floatFromInt(s1 - s0)) / 1e6, @as(f64, @floatFromInt(s2 - s0)) / 1e6, secs / (@as(f64, @floatFromInt(s2 - s0)) / 1e9) });
            if (rep == 1) if (try refTensor(gpa, &ref, "dec.audio")) |r| {
                defer gpa.free(r);
                const n = @min(r.len, audio.len);
                const snr = snrDb(r[0..n], audio[0..n]);
                std.debug.print("  waveform vs the PyTorch codec (fp32, no TF32): SNR {d:.2} dB over {d} samples\n", .{ snr, n });
                pass = pass and snr >= 35;
                const wavb = try d3.wav.encode16(gpa, audio, 44100);
                defer gpa.free(wavb);
                try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = "codec_check.wav", .data = wavb });
            };
        }
    }
    if (a.ref_audio) |wp| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, wp, gpa, .limited(1 << 30));
        const au = try d3.wav.parse(gpa, bytes);
        const x44 = try d3.wav.resample(gpa, au.samples, au.rate, 44100);
        if (try refTensor(gpa, &ref, "enc.audio44k")) |r| {
            const n = @min(r.len, x44.len);
            std.debug.print("  resampled clip ({d} Hz -> 44.1 kHz, {d} vs {d} samples): SNR {d:.2} dB vs torchaudio\n", .{ au.rate, x44.len, r.len, snrDb(r[0..n], x44[0..n]) });
        }
        const P = struct {
            gpa: std.mem.Allocator,
            ref: *core.safetensors.File,
            d: *const cuda.Driver,
            fn f(pc: *anyopaque, name: []const u8, ptr: u64, rows: usize, cols: usize) anyerror!void {
                const p: *@This() = @ptrCast(@alignCast(pc));
                const r = (try refTensor(p.gpa, p.ref, name)) orelse return;
                const mine = try p.gpa.alloc(f32, rows * cols);
                var b: cuda.DeviceBuffer = .{ .d = p.d, .ptr = ptr, .len = rows * cols * 4, .borrowed = true };
                try b.download(0, std.mem.sliceAsBytes(mine));
                const n = @min(r.len, mine.len);
                std.debug.print("  {s}: {d} vs {d} values, SNR {d:.2} dB\n", .{ name, mine.len, r.len, snrDb(r[0..n], mine[0..n]) });
            }
        };
        var pctx: P = .{ .gpa = gpa, .ref = &ref, .d = &driver };
        c.probe = .{ .ctx = &pctx, .f = P.f };
        const e0 = d3.engine.nowNs(io);
        const codes = try c.encode(stream, x44);
        c.probe = null;
        const e1 = d3.engine.nowNs(io);
        std.debug.print("encode: {d} frames in {d:.1} ms\n", .{ codes.len, @as(f64, @floatFromInt(e1 - e0)) / 1e6 });
        if (ref.get("enc.codes")) |rc| {
            const rv = std.mem.bytesAsSlice(i32, @as([]align(1) const u8, rc.bytes));
            const nf = @min(codes.len, rc.dim(0));
            var same: [10]usize = @splat(0);
            for (0..nf) |t| for (0..10) |j| {
                same[j] += @intFromBool(codes[t][j] == rv[t * 10 + j]);
            };
            var tot: usize = 0;
            std.debug.print("  codes vs the PyTorch codec ({d} vs {d} frames): per codebook", .{ codes.len, rc.dim(0) });
            for (same) |x| {
                tot += x;
                std.debug.print(" {d}", .{x});
            }
            const pct = 100.0 * @as(f64, @floatFromInt(tot)) / @as(f64, @floatFromInt(nf * 10));
            std.debug.print(" of {d}: {d:.2}%\n", .{ nf, pct });
            pass = pass and pct >= 97;
        }
    }
    std.debug.print("RESULT drama3 codec-check {s}\n", .{if (pass) "PASS" else "FAIL"});
    return if (pass) 0 else 1;
}
