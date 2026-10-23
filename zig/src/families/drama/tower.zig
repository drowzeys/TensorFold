// SPDX-License-Identifier: Apache-2.0
//! Drama 3's four shared and nine residual codebook layers, on TensorFold CUDA.
//! The tower cache is temporal: every layer retains all preceding text/audio rows.
const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("core").checkpoint;
const Stager = @import("stager.zig").Stager;
const tower_fp8 = @import("tower_fp8.zig");
const Projection = tower_fp8.Projection;
const Buffer = cuda.DeviceBuffer;
const Linear = cuda.cublaslt.Linear;
const Allocator = std.mem.Allocator;

pub const Frame = [11]u32;
pub const Options = struct { context: usize = 8192, max_rows: usize = 128 };
pub const Sampling = struct { temperature: f32 = 1, top_p: f32 = 0.9, top_k: u32 = 30 };
const hidden = 2048;
const intermediate = 6144;
const query_width = 4096;
const kv_width = 512;
const qkv_width = 9216;
pub const layer_count = 13;
const workspace_bytes = 8 << 20;
const sample_shared_bytes = 65536;
/// drama_tower.cu's attention_split: the CTAs of drama_attention_decode's cluster for one query head.
const attention_split = 4;
const projections = [_][2]usize{
    .{ hidden, 2 * hidden },       .{ qkv_width, hidden },    .{ hidden, query_width },
    .{ 2 * intermediate, hidden }, .{ hidden, intermediate }, .{ 4096, hidden },
};

/// Diagnostic capture points of `Model.prefillTraced`, in execution order (temporary; tf-drama-tower-trace is the
/// only tracer). Every capture is a BF16 row-major [rows][stageWidth(stage)] tensor; row r is request position
/// Model.position + r. shared_input is captured once, before layer 0; every other stage once per layer.
/// shared_input: setInput's codebook 2..9 embedding sum (layer 0's mix extra input). joined: drama_mix output,
/// [enorm(code embedding (+ shared_input)), hnorm(layer input)]. mixed: eh_proj output. attention_input: input
/// RMSNorm of mixed. qkv: wqkv output in checkpoint column order [query | gate | key | value]. query: q_norm then
/// interleaved rope. key / value: the layer's KV cache rows just written (k_norm then rope / raw value).
/// attended: attention times sigmoid(gate), the wo input. attention_output: wo output. mlp_input / mlp_residual:
/// post-attention RMSNorm of attention_output + mixed and that BF16 sum. gate_up: [gate | up] projection.
/// activated: silu(gate) * up. mlp_output: w2 output. output: final RMSNorm of mlp_output + mlp_residual, the
/// layer output (layer 12: the hidden the codebook heads read).
pub const Stage = enum {
    shared_input,
    joined,
    mixed,
    attention_input,
    qkv,
    query,
    key,
    value,
    attended,
    attention_output,
    mlp_input,
    mlp_residual,
    gate_up,
    activated,
    mlp_output,
    output,
};

pub fn stageWidth(stage: Stage) usize {
    return switch (stage) {
        .shared_input, .mixed, .attention_input, .attention_output, .mlp_input, .mlp_residual, .mlp_output, .output => hidden,
        .joined => 2 * hidden,
        .qkv => qkv_width,
        .query, .attended => query_width,
        .key, .value => kv_width,
        .gate_up => 2 * intermediate,
        .activated => intermediate,
    };
}

/// Hashed by the trace driver so a capture names the exact tower it observed; unreferenced in production.
pub const trace_source = @embedFile("tower.zig");

const Config = struct {
    model_type: []const u8,
    semantic_start_token_id: u32,
    semantic_end_token_id: u32,
    eos_token_id: u32,
    audio_decoder_config: struct {
        decoder_type: []const u8,
        dim: usize,
        intermediate_size: usize,
        n_head: usize,
        n_local_heads: usize,
        head_dim: usize,
        shared_enrichment_layers: usize,
        num_codebooks: usize,
        vocab_size: usize,
        vq_backbone_codebooks: usize,
        rotary_percent: f32,
        rotary_interleaved: bool,
        rope_base: f32,
        norm_eps: f32,
        attention_output_gate: bool,
        layernorm_zero_centered_gamma: bool,
    },
};

const Layer = struct {
    embedding: Buffer,
    enorm: Buffer,
    hnorm: Buffer,
    eh: Buffer,
    input_norm: Buffer,
    qkv: Projection,
    qnorm: Buffer,
    knorm: Buffer,
    out: Projection,
    post_norm: Buffer,
    gate_up: Projection,
    down: Projection,
    final_norm: Buffer,
};
const Kernels = struct {
    embeddings: cuda.Function,
    mix: cuda.Function,
    norm: cuda.Function,
    rope: cuda.Function,
    attention: cuda.Function,
    attention_decode: cuda.Function,
    silu: cuda.Function,
    sample: cuda.Function,
    fn resolve(m: cuda.Module) !Kernels {
        return .{
            .embeddings = try m.function("drama_frame_embeddings"),
            .mix = try m.function("drama_mix"),
            .norm = try m.function("drama_norm"),
            .rope = try m.function("drama_rope_cache"),
            .attention = try m.function("drama_attention"),
            .attention_decode = try m.function("drama_attention_decode"),
            .silu = try m.function("drama_silu_mul"),
            .sample = try m.function("drama_sample"),
        };
    }
};

const Impl = struct {
    gpa: Allocator,
    driver: *const cuda.Driver,
    options: Options,
    semantic_start: u32,
    semantic_end: u32,
    eos: u32,
    eps: f32,
    rope_base: f32,
    buffers: std.ArrayList(Buffer) = .empty,
    plans: std.ArrayList(Linear) = .empty,
    module: ?cuda.Module = null,
    gemv_module: ?cuda.Module = null,
    fp8: tower_fp8.Lane = .{},
    gemv: cuda.Function = undefined,
    /// Per projection kind: qualifyGemv found drama_tower_gemv bit-identical to its one-row plan.
    gemv_exact: [projections.len]bool = @splat(false),
    library: cuda.cublaslt.Library = undefined,
    library_open: bool = false,
    kernels: Kernels = undefined,
    layers: [layer_count]Layer = undefined,
    heads: [9]Buffer = undefined,
    embeddings: Buffer = undefined,
    workspace: Buffer = undefined,
    cache: Buffer = undefined,
    input_frames: Buffer = undefined,
    labels: Buffer = undefined,
    extra: Buffer = undefined,
    shared: Buffer = undefined,
    joined: Buffer = undefined,
    mixed: Buffer = undefined,
    normalized: Buffer = undefined,
    qkv: Buffer = undefined,
    queries: Buffer = undefined,
    attended: Buffer = undefined,
    branch: Buffer = undefined,
    residual: Buffer = undefined,
    gate_up: Buffer = undefined,
    activated: Buffer = undefined,
    output_hidden: Buffer = undefined,
    head_logits: Buffer = undefined,
    result: Buffer = undefined,
    position: usize = 0,
    input_rows: usize = 0,
    step: u64 = 0,
    seed: u64 = 0,

    fn release(p: *Impl) void {
        for (p.plans.items) |*plan| plan.deinit();
        p.plans.deinit(p.gpa);
        if (p.library_open) p.library.close();
        for (p.buffers.items) |*buffer| buffer.free();
        p.buffers.deinit(p.gpa);
        if (p.module) |*module| module.unload();
        if (p.gemv_module) |*module| module.unload();
        p.fp8.unload();
        const gpa = p.gpa;
        gpa.destroy(p);
    }

    /// pub for tower_fp8.zig, which allocates the weights it loads and the lane's scratch here.
    pub fn allocate(p: *Impl, bytes: usize) !Buffer {
        var owned = try Buffer.alloc(p.driver, bytes);
        errdefer owned.free();
        try p.buffers.append(p.gpa, owned);
        return view(owned, 0, bytes);
    }

    fn weight(p: *Impl, st: *Stager, ck: *checkpoint.Checkpoint, name: []const u8, shape: []const usize) !Buffer {
        const t = try ck.expect(name, .bf16, shape);
        const out = try p.allocate(t.bytes.len);
        try st.put(out.ptr, t.bytes);
        return out;
    }

    fn layerWeight(p: *Impl, st: *Stager, ck: *checkpoint.Checkpoint, prefix: []const u8, suffix: []const u8, shape: []const usize) !Buffer {
        var name: [256]u8 = undefined;
        return p.weight(st, ck, try std.fmt.bufPrint(&name, "{s}.{s}.weight", .{ prefix, suffix }), shape);
    }

    fn loadLayer(p: *Impl, st: *Stager, ck: *checkpoint.Checkpoint, staging: *tower_fp8.Staging, i: usize) !Layer {
        var name: [128]u8 = undefined;
        const prefix = if (i < 4)
            try std.fmt.bufPrint(&name, "audio_decoder_mtp_shared.layers.{d}", .{i})
        else
            try std.fmt.bufPrint(&name, "audio_decoder_mtp.layers.{d}", .{i - 4});
        const gate_up = try tower_fp8.load(p, st, ck, staging, prefix, &.{ "feed_forward.w1", "feed_forward.w3" }, projections[3]);
        return .{
            .embedding = try p.layerWeight(st, ck, prefix, if (i < 4) "semantic_embedding" else "codebook_embedding", &.{ 4096, hidden }),
            .enorm = try p.layerWeight(st, ck, prefix, "enorm", &.{hidden}),
            .hnorm = try p.layerWeight(st, ck, prefix, "hnorm", &.{hidden}),
            .eh = try p.layerWeight(st, ck, prefix, "eh_proj", &.{ hidden, 2 * hidden }),
            .input_norm = try p.layerWeight(st, ck, prefix, "transformer_layer.attention_norm", &.{hidden}),
            .qkv = try tower_fp8.load(p, st, ck, staging, prefix, &.{"attention.wqkv"}, projections[1]),
            .qnorm = try p.layerWeight(st, ck, prefix, "transformer_layer.attention.q_norm", &.{256}),
            .knorm = try p.layerWeight(st, ck, prefix, "transformer_layer.attention.k_norm", &.{256}),
            .out = try tower_fp8.load(p, st, ck, staging, prefix, &.{"attention.wo"}, projections[2]),
            .post_norm = try p.layerWeight(st, ck, prefix, "transformer_layer.ffn_norm", &.{hidden}),
            .gate_up = gate_up,
            .down = try tower_fp8.load(p, st, ck, staging, prefix, &.{"feed_forward.w2"}, projections[4]),
            .final_norm = try p.layerWeight(st, ck, prefix, "final_layernorm", &.{hidden}),
        };
    }

    /// pub for tf-cuda-test drama-fp8-tower, which holds the lane to this path's BF16 arithmetic.
    pub fn project(p: *Impl, stream: cuda.Stream, rows: usize, kind: usize, x: Buffer, weights: Projection, out: Buffer) !void {
        const w = switch (weights) {
            .fp8g => |q| return p.fp8.matmul(stream, q, x, out, rows),
            .bf16 => |b| b,
        };
        if (rows == 1 and p.gemv_exact[kind]) return p.gemvRow(stream, kind, x, w, out);
        const plan = &p.plans.items[(rows - 1) * projections.len + kind];
        try plan.run(x.ptr, w.ptr, out.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
    }

    fn gemvRow(p: *Impl, stream: cuda.Stream, kind: usize, x: Buffer, w: Buffer, out: Buffer) !void {
        comptime for (projections) |shape| std.debug.assert(shape[0] % 8 == 0 and shape[1] % 256 == 0);
        var a: cuda.Args = .{};
        a.add(x.ptr);
        a.add(w.ptr);
        a.add(out.ptr);
        a.add(@as(i32, @intCast(projections[kind][1])));
        try launch(p.gemv, stream, projections[kind][0] / 8, 32, 0, &a);
    }

    /// drama_tower_gemv sums in the order of the gemvx kernel cuBLASLt 13.1.1 picks for these plans on GB10, but a
    /// plan's arithmetic is the heuristic's choice. So a projection switches only once its own one-row plan, run as
    /// project() runs it, matched the kernel bit for bit on rows that separate association orders (probeWeights);
    /// any other device, library or result keeps cuBLASLt, and an error commits nothing (qualifyAll).
    fn qualifyGemv(p: *Impl, io: std.Io, ctx: *cuda.Context, kernel_dir: []const u8) !void {
        const capability = try ctx.capability();
        const version = p.library.api.cublasLtGetVersion();
        if (capability != 121 or version != 130101) {
            std.log.info("drama tower: one-row projections on cuBLASLt: drama_tower_gemv is qualified on sm 121 with cuBLASLt 130101, not sm {d} with {d}", .{ capability, version });
            return;
        }
        p.gemv_module = try loadModule(p.gpa, io, p.driver, kernel_dir, "drama_tower_gemv.fatbin", cuda.kernels.drama_tower_gemv);
        errdefer {
            p.gemv_module.?.unload();
            p.gemv_module = null;
        }
        p.gemv = try p.gemv_module.?.function("drama_tower_gemv");
        const most = comptime blk: {
            var m: struct { weights: usize = 0, n: usize = 0, k: usize = 0 } = .{};
            for (projections) |shape| {
                m.weights = @max(m.weights, shape[0] * shape[1]);
                m.n = @max(m.n, shape[0]);
                m.k = @max(m.k, shape[1]);
            }
            break :blk m;
        };
        const host = try p.gpa.alloc(u16, @max(probe_rows * most.k, 2 * most.n));
        defer p.gpa.free(host);
        var w = try Buffer.alloc(p.driver, most.weights * 2);
        defer w.free();
        var x = try Buffer.alloc(p.driver, most.k * 2);
        defer x.free();
        var planned = try Buffer.alloc(p.driver, most.n * 2);
        defer planned.free();
        var kernel = try Buffer.alloc(p.driver, most.n * 2);
        defer kernel.free();
        // blocking: its work waits for the uploads, which may return before a pageable copy has landed
        var stream = try cuda.Stream.init(p.driver, false);
        defer stream.deinit();
        // drained before the scratch is freed or the image unloaded, on every path
        defer stream.synchronize() catch {};
        @memset(host[0..most.k], 0x3f80); // x = 1
        try x.upload(0, std.mem.sliceAsBytes(host[0..most.k]));
        try qualifyAll(&p.gemv_exact, GemvProbe{ .p = p, .stream = stream, .host = host, .w = w, .x = x, .planned = planned, .kernel = kernel });
        const e = p.gemv_exact;
        std.log.info("drama tower: one-row projections on drama_tower_gemv (1) or cuBLASLt (0): eh {d} wqkv {d} wo {d} gate_up {d} down {d} head {d}", .{
            @intFromBool(e[0]), @intFromBool(e[1]), @intFromBool(e[2]), @intFromBool(e[3]), @intFromBool(e[4]), @intFromBool(e[5]),
        });
    }

    fn norm(p: *Impl, stream: cuda.Stream, rows: usize, x: Buffer, residual: ?Buffer, w: Buffer, out: Buffer, residual_out: ?Buffer) !void {
        var a: cuda.Args = .{};
        a.add(x.ptr);
        a.add(if (residual) |r| r.ptr else @as(u64, 0));
        a.add(w.ptr);
        a.add(out.ptr);
        a.add(if (residual_out) |r| r.ptr else @as(u64, 0));
        a.add(@as(i32, hidden));
        a.add(p.eps);
        try launch(p.kernels.norm, stream, rows, 256, 0, &a);
    }

    /// One layer. `trace` is `{}` in production: every `stage` call below compiles to nothing.
    fn layer(p: *Impl, stream: cuda.Stream, rows: usize, i: usize, input: Buffer, labels: Buffer, trace: anytype) !void {
        const w = p.layers[i];
        var a: cuda.Args = .{};
        a.add(input.ptr);
        a.add(w.embedding.ptr);
        a.add(labels.ptr);
        a.add(if (i == 0) p.shared.ptr else @as(u64, 0));
        a.add(w.enorm.ptr);
        a.add(w.hnorm.ptr);
        a.add(p.joined.ptr);
        a.add(@as(i32, hidden));
        a.add(@as(i32, @intCast(if (i < 4) 0 else i - 4)));
        a.add(@as(i32, @intCast(p.semantic_start)));
        a.add(@as(i32, @intCast(p.semantic_end)));
        a.add(p.eps);
        try launch(p.kernels.mix, stream, rows, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .joined);
        try p.project(stream, rows, 0, p.joined, .{ .bf16 = w.eh }, p.mixed);
        try p.stage(trace, stream, i, rows, .mixed);
        try p.norm(stream, rows, p.mixed, null, w.input_norm, p.normalized, null);
        try p.stage(trace, stream, i, rows, .attention_input);
        try p.project(stream, rows, 1, p.normalized, w.qkv, p.qkv);
        try p.stage(trace, stream, i, rows, .qkv);
        const cache_bytes = p.options.context * kv_width * 2;
        const keys = view(p.cache, i * 2 * cache_bytes, cache_bytes);
        const values = view(p.cache, (i * 2 + 1) * cache_bytes, cache_bytes);
        a = .{};
        a.add(p.qkv.ptr);
        a.add(w.qnorm.ptr);
        a.add(w.knorm.ptr);
        a.add(p.queries.ptr);
        a.add(keys.ptr);
        a.add(values.ptr);
        a.add(@as(i32, @intCast(p.position)));
        a.add(@as(i32, 16));
        a.add(@as(i32, 2));
        a.add(@as(i32, 256));
        a.add(@as(i32, 64));
        a.add(p.rope_base);
        a.add(p.eps);
        try launch(p.kernels.rope, stream, rows * 18, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .query);
        try p.stage(trace, stream, i, rows, .key);
        try p.stage(trace, stream, i, rows, .value);
        a = .{};
        a.add(p.queries.ptr);
        a.add(keys.ptr);
        a.add(values.ptr);
        a.add(p.qkv.ptr);
        a.add(p.attended.ptr);
        a.add(@as(i32, @intCast(p.position)));
        a.add(@as(i32, 16));
        a.add(@as(i32, 2));
        a.add(@as(i32, 256));
        if (rows == 1) try launch(p.kernels.attention_decode, stream, 16 * attention_split, 256, 0, &a) else try launch(p.kernels.attention, stream, rows * 16, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .attended);
        try p.project(stream, rows, 2, p.attended, w.out, p.branch);
        try p.stage(trace, stream, i, rows, .attention_output);
        try p.norm(stream, rows, p.branch, p.mixed, w.post_norm, p.normalized, p.residual);
        try p.stage(trace, stream, i, rows, .mlp_input);
        try p.stage(trace, stream, i, rows, .mlp_residual);
        try p.project(stream, rows, 3, p.normalized, w.gate_up, p.gate_up);
        try p.stage(trace, stream, i, rows, .gate_up);
        a = .{};
        a.add(p.gate_up.ptr);
        a.add(p.activated.ptr);
        a.add(@as(i32, intermediate));
        a.add(@as(i32, @intCast(rows)));
        try launch(p.kernels.silu, stream, (rows * intermediate + 255) / 256, 256, 0, &a);
        try p.stage(trace, stream, i, rows, .activated);
        try p.project(stream, rows, 4, p.activated, w.down, p.branch);
        try p.stage(trace, stream, i, rows, .mlp_output);
        try p.norm(stream, rows, p.branch, p.residual, w.final_norm, p.output_hidden, null);
        try p.stage(trace, stream, i, rows, .output);
    }

    /// One capture point. With a void trace the comptime-false branch is never analyzed and emits nothing; call
    /// sites pass only values already in hand, so production evaluates nothing for it. Otherwise the tracer gets
    /// the stage's live scratch tensor (key / value: this layer's cache rows from position on), first `rows` rows,
    /// right after the producing kernel was queued on `stream`. It must finish reading them on `stream` before
    /// returning, because the next kernel may overwrite them, and must not queue other device work.
    inline fn stage(p: *const Impl, trace: anytype, stream: cuda.Stream, i: usize, rows: usize, comptime which: Stage) !void {
        if (@TypeOf(trace) != void) if (@hasDecl(@typeInfo(@TypeOf(trace)).pointer.child, "capture")) {
            const width = comptime stageWidth(which);
            const cache_bytes = p.options.context * kv_width * 2;
            const data = switch (which) {
                .shared_input => p.shared,
                .joined => p.joined,
                .mixed => p.mixed,
                .attention_input, .mlp_input => p.normalized,
                .qkv => p.qkv,
                .query => p.queries,
                .key => view(p.cache, i * 2 * cache_bytes + p.position * width * 2, cache_bytes - p.position * width * 2),
                .value => view(p.cache, (i * 2 + 1) * cache_bytes + p.position * width * 2, cache_bytes - p.position * width * 2),
                .attended => p.attended,
                .attention_output, .mlp_output => p.branch,
                .mlp_residual => p.residual,
                .gate_up => p.gate_up,
                .activated => p.activated,
                .output => p.output_hidden,
            };
            try trace.capture(stream, i, which, view(data, 0, rows * width * 2), p.position);
        };
    }

    fn sample(p: *Impl, stream: cuda.Stream, logits: Buffer, is_bf16: bool, output_index: usize, sampling: Sampling) !void {
        var a: cuda.Args = .{};
        a.add(logits.ptr);
        a.add(@as(i32, @intFromBool(is_bf16)));
        a.add(@as(i32, if (output_index == 0) 4097 else 4096));
        a.add(sampling.temperature);
        a.add(@as(i32, @intCast(sampling.top_k)));
        a.add(sampling.top_p);
        a.add(p.seed);
        a.add(p.step);
        a.add(@as(i32, @intCast(output_index)));
        a.add(@as(i32, @intCast(p.semantic_start)));
        a.add(@as(i32, @intCast(p.eos)));
        a.add(p.result.ptr + 48);
        a.add(p.result.ptr);
        a.add(p.result.ptr + 44);
        try launch(p.kernels.sample, stream, 1, 1024, sample_shared_bytes, &a);
    }
};

fn view(buffer: Buffer, offset: usize, bytes: usize) Buffer {
    std.debug.assert(offset <= buffer.len and bytes <= buffer.len - offset);
    return .{ .d = buffer.d, .ptr = buffer.ptr + offset, .len = bytes, .borrowed = true };
}
fn launch(function: cuda.Function, stream: cuda.Stream, blocks: usize, threads: u32, shared: u32, args: *cuda.Args) !void {
    try cuda.launch.launch(function, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = threads, .y = 1, .z = 1 }, .shared = shared }, stream, args);
}

/// `file` from `kernel_dir`, else the embedded image.
fn loadModule(gpa: Allocator, io: std.Io, d: *const cuda.Driver, kernel_dir: []const u8, file: []const u8, embedded: []const u8) !cuda.Module {
    if (kernel_dir.len == 0) return cuda.Module.load(d, embedded);
    const path = try std.fs.path.join(gpa, &.{ kernel_dir, file });
    defer gpa.free(path);
    const image = try std.Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .limited(1 << 26), .@"16", null);
    defer gpa.free(image);
    return cuda.Module.load(d, image);
}

/// Distinct probe rows: qualifyGemv repeats them down each weight matrix (every output row is summed alone).
const probe_rows = 256;

/// Each projection's decision from `probe.bitIdentical(kind)`, committed to `exact` only once all of them ran, so an
/// error, even after earlier projections passed, leaves every projection on its plan.
fn qualifyAll(exact: *[projections.len]bool, probe: anytype) !void {
    var staged: [projections.len]bool = undefined;
    for (&staged, 0..) |*s, kind| s.* = try probe.bitIdentical(kind);
    exact.* = staged;
}

/// qualifyGemv's scratch, on its blocking stream.
const GemvProbe = struct {
    p: *Impl,
    stream: cuda.Stream,
    host: []u16,
    w: Buffer,
    x: Buffer,
    planned: Buffer,
    kernel: Buffer,

    /// The projection's one-row plan, through project() while it is not yet switched, against drama_tower_gemv on
    /// probe rows; different output fills, so an output either side failed to write cannot match.
    fn bitIdentical(g: GemvProbe, kind: usize) !bool {
        const n = projections[kind][0];
        const k = projections[kind][1];
        probeWeights(g.host[0 .. probe_rows * k], k);
        try g.w.upload(0, std.mem.sliceAsBytes(g.host[0 .. probe_rows * k]));
        var filled: usize = probe_rows;
        while (filled < n) {
            const more = @min(filled, n - filled);
            try g.w.copyFrom(filled * k * 2, g.w.ptr, more * k * 2, g.stream.handle);
            filled += more;
        }
        try g.planned.fill8(0xff, g.stream.handle);
        try g.kernel.fill8(0x00, g.stream.handle);
        try g.p.project(g.stream, 1, kind, g.x, .{ .bf16 = g.w }, g.planned);
        try g.p.gemvRow(g.stream, kind, g.x, g.w, g.kernel);
        try g.stream.synchronize();
        try g.planned.download(0, std.mem.sliceAsBytes(g.host[0..n]));
        try g.kernel.download(0, std.mem.sliceAsBytes(g.host[n .. 2 * n]));
        return std.mem.eql(u16, g.host[0..n], g.host[n .. 2 * n]);
    }
};

/// qualifyGemv's weights, read with x = 1. Row r pairs column c with c ^ m (m random a row, below 2048, so the pair
/// stays inside K) and gives the pair an exactly cancelling ±2^0..2^20 value, or, one pair in eight, two independent
/// ±2^-7..2^0 values. What survives of the small terms depends on which chains the large ones accumulated in, in
/// which order, and the tree joining the chains, so another association order changes most rows (test below).
fn probeWeights(w: []u16, k: usize) void {
    for (0..w.len / k) |r| {
        const h = splitmix(r);
        const m: usize = @intCast(1 + h % 2047);
        const row = w[r * k ..][0..k];
        for (0..k) |c| {
            const partner = c ^ m;
            if (partner < c) continue;
            const q = splitmix(h ^ @as(u64, c));
            const small = (q & 7) == 0;
            row[c] = probeValue(q, small);
            row[partner] = if (small) probeValue(splitmix(q), true) else row[c] ^ 0x8000;
        }
    }
}

fn probeValue(q: u64, small: bool) u16 {
    const exponent: u64 = if (small) 120 + (q >> 8) % 8 else 127 + (q >> 8) % 21;
    return @intCast((q >> 24 & 1) << 15 | exponent << 7 | (q >> 16 & 0x7f));
}

fn splitmix(z0: u64) u64 {
    var z = z0 +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

pub const Model = struct {
    impl: *Impl,

    pub fn init(gpa: Allocator, io: std.Io, ctx: *cuda.Context, dir: []const u8, kernel_dir: []const u8, options: Options) !Model {
        if (options.context == 0 or options.context > 8192 or options.max_rows == 0 or options.max_rows > 128) return error.BadTowerCapacity;
        const config_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(config_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(Config, gpa, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const c = parsed.value.audio_decoder_config;
        if (!std.mem.eql(u8, parsed.value.model_type, "fish_omni") or !std.mem.eql(u8, c.decoder_type, "codebook_mtp_shared") or
            c.dim != hidden or c.intermediate_size != intermediate or c.n_head != 16 or c.n_local_heads != 2 or c.head_dim != 256 or
            c.shared_enrichment_layers != 4 or c.num_codebooks != 10 or c.vocab_size != 4096 or c.vq_backbone_codebooks != 1 or
            c.rotary_percent != 0.25 or !c.rotary_interleaved or !c.attention_output_gate or !c.layernorm_zero_centered_gamma or
            c.rope_base != 10000000 or c.norm_eps != 1e-6 or parsed.value.semantic_end_token_id -| parsed.value.semantic_start_token_id != 4095 or
            parsed.value.eos_token_id >= parsed.value.semantic_start_token_id) return error.UnsupportedTowerConfig;
        const p = try gpa.create(Impl);
        p.* = .{ .gpa = gpa, .driver = ctx.d, .options = options, .semantic_start = parsed.value.semantic_start_token_id, .semantic_end = parsed.value.semantic_end_token_id, .eos = parsed.value.eos_token_id, .eps = c.norm_eps, .rope_base = c.rope_base };
        errdefer p.release();
        p.module = try loadModule(gpa, io, ctx.d, kernel_dir, "drama_tower.fatbin", cuda.kernels.drama_tower);
        p.kernels = try Kernels.resolve(p.module.?);
        try p.kernels.sample.allowDynamicShared(sample_shared_bytes);
        p.library = try cuda.cublaslt.Library.open();
        p.library_open = true;
        try p.plans.ensureTotalCapacityPrecise(gpa, options.max_rows * projections.len);
        for (1..options.max_rows + 1) |rows| for (projections) |shape| {
            var plan = try Linear.init(&p.library, rows, shape[0], shape[1], .bf16, workspace_bytes);
            errdefer plan.deinit();
            try p.plans.append(gpa, plan);
        };
        var ck = try checkpoint.Checkpoint.openModelPrefix(gpa, io, dir, "audio_decoder");
        defer ck.close();
        var stream = try cuda.Stream.init(ctx.d, true);
        defer stream.deinit();
        var st: Stager = .init(gpa, ctx.d, stream);
        defer st.deinit();
        for (ck.files.items) |*f| try st.addFile(f.path, f.map.memory);
        var staging: tower_fp8.Staging = .{ .d = ctx.d, .bytes = tower_fp8.codeBytes(projections[1..5]) };
        defer staging.deinit(stream);
        p.embeddings = try p.weight(&st, &ck, "audio_decoder.codebook_embeddings.weight", &.{ 40960, hidden });
        for (0..layer_count) |i| {
            p.layers[i] = try p.loadLayer(&st, &ck, &staging, i);
            try staging.pack(&st);
        }
        for (&p.heads, 0..) |*head, i| {
            var name: [128]u8 = undefined;
            head.* = try p.weight(&st, &ck, try std.fmt.bufPrint(&name, "audio_decoder_mtp.heads.{d}.weight", .{i}), &.{ 4096, hidden });
        }
        if (ck.unused() != 0) return error.UnusedTowerWeights;
        try st.finish();
        const rows = options.max_rows;
        p.workspace = try p.allocate(workspace_bytes);
        p.cache = try p.allocate(layer_count * 2 * options.context * kv_width * 2);
        p.input_frames = try p.allocate(rows * @sizeOf(Frame));
        p.labels = try p.allocate(rows * @sizeOf(Frame));
        p.extra = try p.allocate(rows * hidden * 2);
        p.shared = try p.allocate(rows * hidden * 2);
        p.joined = try p.allocate(rows * 2 * hidden * 2);
        p.mixed = try p.allocate(rows * hidden * 2);
        p.normalized = try p.allocate(rows * hidden * 2);
        p.qkv = try p.allocate(rows * qkv_width * 2);
        p.queries = try p.allocate(rows * query_width * 2);
        p.attended = try p.allocate(rows * query_width * 2);
        p.branch = try p.allocate(rows * hidden * 2);
        p.residual = try p.allocate(rows * hidden * 2);
        p.gate_up = try p.allocate(rows * 2 * intermediate * 2);
        p.activated = try p.allocate(rows * intermediate * 2);
        p.output_hidden = try p.allocate(rows * hidden * 2);
        p.head_logits = try p.allocate(4096 * 2);
        p.result = try p.allocate(48 + 16 * 4);
        try p.fp8.prepare(p, ctx);
        // The probe is an optimization: a failure that leaves the context usable keeps every projection on its plan
        // (qualifyGemv commits nothing and unloads its image). A sticky fault poisons the context for the whole
        // model, so that one stays fatal.
        p.qualifyGemv(io, ctx, kernel_dir) catch |e| {
            try ctx.synchronize();
            std.log.warn("drama tower: one-row projections on cuBLASLt: drama_tower_gemv qualification failed: {s}", .{@errorName(e)});
        };
        return .{ .impl = p };
    }

    pub fn deinit(self: *Model) void {
        self.impl.release();
        self.* = undefined;
    }

    pub fn denseFp8(self: *const Model) tower_fp8.Census {
        return tower_fp8.census(&self.impl.layers);
    }

    pub fn reset(self: *Model, stream: cuda.Stream, prompt_tail: []const u32, seed: u64) !void {
        const p = self.impl;
        try stream.synchronize();
        try p.result.fill8(0, null);
        var history: [16]u32 = @splat(std.math.maxInt(u32));
        const n = @min(prompt_tail.len, history.len);
        @memcpy(history[history.len - n ..], prompt_tail[prompt_tail.len - n ..]);
        try p.result.upload(48, std.mem.sliceAsBytes(&history));
        p.position = 0;
        p.input_rows = 0;
        p.step = 0;
        p.seed = seed;
        // All subsequently read KV positions are rewritten; clearing the entire
        // cache would be wasted work and is not needed for request isolation.
    }

    /// Stage input frames once, before the backbone consumes the same rows.
    /// The returned BF16 codebook-1 embedding is borrowed until the next setInput.
    pub fn setInput(self: *Model, stream: cuda.Stream, frames: []const Frame) !?Buffer {
        const p = self.impl;
        if (frames.len == 0 or frames.len > p.options.max_rows) return error.BadTowerRows;
        var has_audio = false;
        for (frames) |frame| {
            if (frame[0] >= p.semantic_start and frame[0] <= p.semantic_end) has_audio = true;
            for (frame[1..]) |v| if (v >= 4096) return error.InvalidCodebookIndex;
        }
        p.input_rows = frames.len;
        if (!has_audio) {
            try view(p.shared, 0, frames.len * hidden * 2).fill8(0, stream.handle);
            return null;
        }
        try p.input_frames.upload(0, std.mem.sliceAsBytes(frames));
        var a: cuda.Args = .{};
        a.add(p.input_frames.ptr);
        a.add(p.embeddings.ptr);
        a.add(p.extra.ptr);
        a.add(p.shared.ptr);
        a.add(@as(i32, hidden));
        a.add(@as(i32, @intCast(p.semantic_start)));
        a.add(@as(i32, @intCast(p.semantic_end)));
        try launch(p.kernels.embeddings, stream, frames.len, 256, 0, &a);
        return view(p.extra, 0, frames.len * hidden * 2);
    }

    /// Fill every teacher-forced prompt row except the final prompt row, whose
    /// tower pass instead uses the first sampled semantic token in generate().
    pub fn prefill(self: *Model, stream: cuda.Stream, hidden_rows: Buffer, next_frames: []const Frame) !void {
        return self.prefillTraced(stream, hidden_rows, next_frames, {});
    }

    /// `prefill` with stage captures. `trace` is `{}` (void: every capture site is compiled out, so `prefill` is
    /// unchanged) or a pointer to a tracer with `head(t, stream, head: usize, logits: Buffer) !void`, capture, or both:
    ///     capture(t, stream: cuda.Stream, layer: usize, stage: Stage, rows: cuda.DeviceBuffer, row0: usize) !void
    /// called after the stage's producing kernel is queued on `stream` (see `Stage`; shared_input as layer 0). `rows`
    /// borrows live BF16 [n][stageWidth(stage)] device memory, valid only in stream order: read it on `stream` before
    /// returning, queueing no device work; row0 is the first row's request position. `head`, for one row, reads head
    /// h's BF16 [4096] logits, generate's projection of layer 4 + h's output, the same way; nothing else changes.
    pub fn prefillTraced(self: *Model, stream: cuda.Stream, hidden_rows: Buffer, next_frames: []const Frame, trace: anytype) !void {
        const p = self.impl;
        const n = next_frames.len;
        if (n == 0 or n > p.input_rows or n > p.options.max_rows or p.position + n > p.options.context or hidden_rows.len < n * hidden * 2) return error.BadTowerRows;
        try p.labels.upload(0, std.mem.sliceAsBytes(next_frames));
        try p.stage(trace, stream, 0, n, .shared_input);
        for (0..layer_count) |i| {
            try p.layer(stream, n, i, if (i == 0) hidden_rows else p.output_hidden, p.labels, trace);
            if (@TypeOf(trace) != void) if (@hasDecl(@typeInfo(@TypeOf(trace)).pointer.child, "head")) if (i >= 4) {
                if (n != 1) return error.BadTowerRows; // head_logits holds one row, as generate projects one
                try p.project(stream, 1, 5, p.output_hidden, .{ .bf16 = p.heads[i - 4] }, p.head_logits);
                try trace.head(stream, i - 4, p.head_logits);
            };
        }
        try stream.synchronize();
        p.position += n;
        p.input_rows = 0;
    }

    pub fn generate(self: *Model, stream: cuda.Stream, semantic_logits: Buffer, hidden_row: Buffer, sampling: Sampling) !Frame {
        const p = self.impl;
        if (p.input_rows != 1 or p.position >= p.options.context or hidden_row.len < hidden * 2 or semantic_logits.len < 4097 * 4) return error.BadTowerRows;
        if (!std.math.isFinite(sampling.temperature) or sampling.temperature < 0 or !std.math.isFinite(sampling.top_p) or
            sampling.top_p <= 0 or sampling.top_p > 1 or sampling.top_k == 0 or sampling.top_k > std.math.maxInt(i32)) return error.BadSpeechSampling;
        try p.sample(stream, semantic_logits, false, 0, sampling);
        for (0..layer_count) |i| {
            try p.layer(stream, 1, i, if (i == 0) hidden_row else p.output_hidden, p.result, {});
            if (i >= 4) {
                try p.project(stream, 1, 5, p.output_hidden, .{ .bf16 = p.heads[i - 4] }, p.head_logits);
                try p.sample(stream, p.head_logits, true, i - 2, sampling);
            }
        }
        try stream.synchronize();
        var result: [12]u32 = undefined;
        try p.result.download(0, std.mem.sliceAsBytes(&result));
        if (result[11] != 0) return error.NonFiniteSpeechLogits;
        p.position += 1;
        p.step += 1;
        p.input_rows = 0;
        return result[0..11].*;
    }
};

/// One output of a bf16 row times x = 1 summed as `chains` fp32 fma chains (strided by `chains`, or contiguous
/// slices), joined by a halving or a sequential tree, then cuBLASLt's epilogue and the bf16 rounding.
fn probeSum(row: []const u16, chains: usize, strided: bool, halving: bool) u16 {
    var p: [32]f32 = @splat(0);
    const len = row.len / chains;
    for (0..chains) |t| for (0..len) |j| {
        const v: f32 = @bitCast(@as(u32, row[if (strided) t + j * chains else t * len + j]) << 16);
        p[t] = @mulAdd(f32, v, 1, p[t]);
    };
    if (halving) {
        var h = chains / 2;
        while (h >= 1) : (h /= 2) for (0..h) |t| {
            p[t] = p[t] + p[t + h];
        };
    } else for (1..chains) |t| {
        p[0] = p[0] + p[t];
    }
    const u: u32 = @bitCast(@mulAdd(f32, 1, p[0], 0) + 0);
    return @truncate((u + 0x7fff + (u >> 16 & 1)) >> 16);
}

test "the gemv probe separates the one-row plans' summation order from its neighbours" {
    const rows = probe_rows;
    const w = try std.testing.allocator.alloc(u16, rows * intermediate);
    defer std.testing.allocator.free(w);
    // (chains, strided, halving tree); the first is cuBLASLt's order on GB10, the others what another kernel might do
    const orders = [_]struct { usize, bool, bool }{
        .{ 16, true, true }, .{ 1, true, true }, .{ 16, true, false }, .{ 8, true, true }, .{ 32, true, true }, .{ 16, false, true },
    };
    for (projections) |shape| {
        const k = shape[1];
        probeWeights(w[0 .. rows * k], k);
        var differ: [orders.len]usize = @splat(0);
        for (0..rows) |r| {
            const row = w[r * k ..][0..k];
            const model = probeSum(row, orders[0][0], orders[0][1], orders[0][2]);
            for (orders[1..], differ[1..]) |o, *d| d.* += @intFromBool(probeSum(row, o[0], o[1], o[2]) != model);
        }
        // measured: each wrong order changes 211..256 of these rows, so half is a floor with margin
        for (differ[1..]) |d| try std.testing.expect(d >= rows / 2);
    }
}

test "gemv qualification commits its decisions only when every projection was probed" {
    // Stands in for GemvProbe: `fail_at` injects the error any of its steps (upload, copy, fill, launch,
    // synchronize, download) would return, at that projection.
    const Fake = struct {
        fail_at: usize,
        mismatch: usize,
        probed: *usize,
        fn bitIdentical(f: @This(), kind: usize) !bool {
            if (kind == f.fail_at) return error.InjectedProbeFailure;
            f.probed.* += 1;
            return kind != f.mismatch;
        }
    };
    const none = projections.len;
    var probed: usize = 0;
    var exact: [projections.len]bool = @splat(false);
    try qualifyAll(&exact, Fake{ .fail_at = none, .mismatch = 2, .probed = &probed });
    try std.testing.expectEqual([_]bool{ true, true, false, true, true, true }, exact);
    for (0..projections.len) |kind| {
        exact = @splat(false);
        probed = 0;
        try std.testing.expectError(error.InjectedProbeFailure, qualifyAll(&exact, Fake{ .fail_at = kind, .mismatch = none, .probed = &probed }));
        try std.testing.expectEqual(kind, probed); // every projection before it passed...
        try std.testing.expectEqual(@as([projections.len]bool, @splat(false)), exact); // ...and none switched
    }
}
