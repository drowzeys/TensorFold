//! The backbone over 2..511 voice-reference prompt rows (11 LE u32 each), staged as drama_host.zig stages them.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");
const Frame = drama.tower.Frame;
const width = 2048;
const candidate_count = 4097;
const context = 512;
const max_rows = 128;

fn runChunks(model: *drama.backbone.Model, tower: *drama.tower.Model, stream: cuda.Stream, frames: []const Frame, tokens: []const u32, chunk: usize, hidden: []u16, logits: []f32) !void {
    try model.reset(stream);
    var at: usize = 0;
    while (at < frames.len) {
        const end = @min(at + chunk, frames.len);
        const extra = try tower.setInput(stream, frames[at..end]);
        const output = try model.run(stream, .prefill, tokens[at..end], extra);
        try output.hidden.download(0, std.mem.sliceAsBytes(hidden[at * width .. end * width]));
        if (end == frames.len) try output.semantic_logits.download(0, std.mem.sliceAsBytes(logits));
        at = end;
    }
    for (hidden) |bits| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonFiniteBackboneHidden;
    }
    for (logits) |value| if (!std.math.isFinite(value)) return error.NonFiniteBackboneLogits;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-reference-prefill CHECKPOINT ROWS_U32LE OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    var bytes: [(context - 1) * @sizeOf(Frame) + 1]u8 align(@alignOf(Frame)) = undefined;
    const file = try std.Io.Dir.cwd().openFile(init.io, args[2], .{});
    defer file.close(init.io);
    const size = try file.readPositionalAll(init.io, &bytes, 0);
    if (size < 2 * @sizeOf(Frame) or size >= bytes.len or size % @sizeOf(Frame) != 0) return error.ExpectedTwoTo511Rows;
    const frames = std.mem.bytesAsSlice(Frame, @as([]align(@alignOf(Frame)) u8, bytes[0..size]));
    const tokens = try init.gpa.alloc(u32, frames.len);
    defer init.gpa.free(tokens);
    for (frames, tokens) |frame, *token| token.* = frame[0];
    var out = try std.Io.Dir.cwd().openDir(init.io, args[3], .{});
    defer out.close(init.io);
    const baseline = try init.gpa.alloc(u16, frames.len * width);
    defer init.gpa.free(baseline);
    const actual = try init.gpa.alloc(u16, frames.len * width);
    defer init.gpa.free(actual);
    var baseline_logits: [candidate_count]f32 = undefined;
    var logits: [candidate_count]f32 = undefined;
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var model = try drama.backbone.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = context, .max_rows = max_rows });
    defer model.deinit();
    var tower = try drama.tower.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = context, .max_rows = max_rows });
    defer tower.deinit();
    var audio_rows: usize = 0;
    for (frames) |frame| {
        if (frame[0] >= drama.backbone.config.vocab) return error.InvalidPromptToken;
        if (frame[0] < model.semantic_start or frame[0] > model.semantic_end) {
            if (!std.mem.allEqual(u32, frame[1..], 0)) return error.TextRowWithCodes;
            continue;
        }
        if (frame[1] != frame[0] - model.semantic_start or std.mem.max(u32, frame[1..]) >= 4096) return error.BadAudioRow;
        audio_rows += 1;
    }
    var all_equal = true;
    var name_buffer: [128]u8 = undefined;
    for ([_]usize{ 1, 7, max_rows }) |chunk| {
        const started = std.Io.Clock.awake.now(init.io).toNanoseconds();
        try runChunks(&model, &tower, stream, frames, tokens, chunk, actual, &logits);
        const elapsed = std.Io.Clock.awake.now(init.io).toNanoseconds() - started;
        var name = try std.fmt.bufPrint(&name_buffer, "native-chunk-{d}-hidden.bf16", .{chunk});
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(actual) });
        name = try std.fmt.bufPrint(&name_buffer, "native-chunk-{d}-semantic.f32", .{chunk});
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(&logits) });
        if (chunk == 1) {
            @memcpy(baseline, actual);
            baseline_logits = logits;
        }
        var hidden_differences: usize = 0;
        for (baseline, actual) |a, b| if (a != b) {
            hidden_differences += 1;
        };
        const head_equal = std.mem.eql(u8, std.mem.sliceAsBytes(&baseline_logits), std.mem.sliceAsBytes(&logits));
        all_equal = all_equal and hidden_differences == 0 and head_equal;
        std.debug.print("rows={d} audio_rows={d} chunk={d} elapsed_s={d:.6} hidden_bit_differences={d} semantic_logits_bit_equal={}\n", .{
            frames.len, audio_rows, chunk, @as(f64, @floatFromInt(elapsed)) / 1e9, hidden_differences, head_equal,
        });
    }
    // Repeat the exact last execution after reset; chunk consistency and reset are separate invariants.
    @memcpy(baseline, actual);
    baseline_logits = logits;
    try runChunks(&model, &tower, stream, frames, tokens, max_rows, actual, &logits);
    const reset_equal = std.mem.eql(u16, baseline, actual) and std.mem.eql(u8, std.mem.sliceAsBytes(&baseline_logits), std.mem.sliceAsBytes(&logits));
    var metadata: [512]u8 = undefined;
    const text = try std.fmt.bufPrint(&metadata, "{{\"rows\":{d},\"audio_rows\":{d},\"hidden_width\":{d},\"eos\":{d},\"semantic_start\":{d},\"semantic_end\":{d},\"chunk_bit_exact\":{},\"reset_bit_exact\":{}}}\n", .{ frames.len, audio_rows, width, model.eos, model.semantic_start, model.semantic_end, all_equal, reset_equal });
    try out.writeFile(init.io, .{ .sub_path = "native-metadata.json", .data = text });
    const used = cuda.usage(false);
    std.debug.print("reset_bit_exact={} cuda_bytes={d} pinned_bytes={d}\n", .{ reset_equal, used.device, used.host });
    return if (all_equal and reset_equal) 0 else 1;
}
