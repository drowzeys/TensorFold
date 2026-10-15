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
const std = @import("std");
const core = @import("core");
const d3 = @import("drama3");

const W = d3.config.frame_width;

const Handle = struct {
    threaded: std.Io.Threaded,
    engine: *d3.Engine,
    tok: core.tokenizer.Tokenizer,
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
    return h;
}

export fn tf_drama3_close(p: ?*anyopaque) void {
    const h: *Handle = @ptrCast(@alignCast(p orelse return));
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
