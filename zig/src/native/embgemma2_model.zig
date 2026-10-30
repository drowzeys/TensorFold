//! EmbeddingGemma-2's text encoder on Metal: the bf16 release's weights, its kernels, and one batch a command buffer.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = @import("embgemma2_geometry.zig");
const Allocator = std.mem.Allocator;

/// Key rows a tiled attention block may read past the batch, and query rows a tile past it: the scratch's padding.
const pad_rows = 256;

/// A tensor's place in the weights buffer.
const Layer = struct {
    qkv: usize, // q | k | v stacked: one GEMM
    o: usize,
    gate_up: usize, // gate | up stacked
    down: usize,
    ple_gate: usize,
    ple_proj: usize,
    input_ln: usize,
    post_attn: usize,
    pre_ff: usize,
    post_ff: usize,
    post_ple: usize,
    q_norm: usize,
    k_norm: usize,
    scalar: f32,
};

const Weights = struct {
    buf: mtl.Buffer,
    embed: usize,
    ple_model: usize,
    ple_norm: usize,
    norm: usize,
    projection: usize,
    layers: [g.layers]Layer,
};

/// The checkpoint's text tensors, bf16 as stored, one after another in one shared buffer (q|k|v and gate|up adjacent).
fn load(gpa: Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var ck = try tf.Checkpoint.openModelPrefix(gpa, io, dir, "language_model.");
    defer ck.close();
    const Part = struct { name: []const u8, shape: [2]usize, rank: usize };
    var parts: std.ArrayList(Part) = .empty;
    defer {
        for (parts.items) |p| gpa.free(p.name);
        parts.deinit(gpa);
    }
    const add = struct {
        fn f(a: Allocator, list: *std.ArrayList(Part), name: []const u8, shape: []const usize) !void {
            var p: Part = .{ .name = try a.dupe(u8, name), .shape = .{ 1, 1 }, .rank = shape.len };
            errdefer a.free(p.name);
            @memcpy(p.shape[0..shape.len], shape);
            try list.append(a, p);
        }
    }.f;
    var kb: [96]u8 = undefined;
    try add(gpa, &parts, "language_model.embed_tokens.weight", &.{ g.vocab, g.hidden });
    try add(gpa, &parts, "language_model.ple.per_layer_model_projection.weight", &.{ g.layers * g.per_layer, g.hidden });
    try add(gpa, &parts, "language_model.ple.per_layer_projection_norm.weight", &.{g.per_layer});
    try add(gpa, &parts, "language_model.norm.weight", &.{g.hidden});
    try add(gpa, &parts, "language_model.embedding_projection.weight", &.{ g.dim, g.hidden });
    const per_layer_names = [_][]const u8{ "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "ple_block.per_layer_input_gate.weight", "ple_block.per_layer_projection.weight", "ple_block.post_per_layer_input_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar" };
    for (0..g.layers) |i| {
        const hd = g.headDim(i);
        const kv = g.kvHeads(i);
        const shapes = [_][]const usize{ &.{ g.heads * hd, g.hidden }, &.{ kv * hd, g.hidden }, &.{ kv * hd, g.hidden }, &.{ g.hidden, g.heads * hd }, &.{hd}, &.{hd}, &.{ g.inter, g.hidden }, &.{ g.inter, g.hidden }, &.{ g.hidden, g.inter }, &.{ g.per_layer, g.hidden }, &.{ g.hidden, g.per_layer }, &.{g.hidden}, &.{g.hidden}, &.{g.hidden}, &.{g.hidden}, &.{g.hidden}, &.{1} };
        for (per_layer_names, shapes) |n, s| try add(gpa, &parts, try g.key(&kb, i, n), s);
    }
    var total: usize = 0;
    for (parts.items) |p| total += std.mem.alignForward(usize, p.shape[0] * p.shape[1] * 2, 256);
    var w: Weights = undefined;
    w.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer w.buf.deinit();
    const base = w.buf.contents();
    var offs = try gpa.alloc(usize, parts.items.len);
    defer gpa.free(offs);
    var at: usize = 0;
    for (parts.items, 0..) |p, k| {
        const t = try ck.expect(p.name, .bf16, p.shape[0..p.rank]);
        @memcpy(base[at..][0..t.bytes.len], t.bytes);
        offs[k] = at;
        if (std.mem.endsWith(u8, p.name, "mlp.up_proj.weight")) interleave(base[offs[k - 1]..][0 .. 2 * t.bytes.len], gpa) catch |err| return err;
        // q|k|v and gate|up are read as one matrix: no padding between their parts (each part's bytes are 256-aligned)
        at += std.mem.alignForward(usize, t.bytes.len, 256);
    }
    w.embed = offs[0];
    w.ple_model = offs[1];
    w.ple_norm = offs[2];
    w.norm = offs[3];
    w.projection = offs[4];
    for (&w.layers, 0..) |*l, i| {
        const o = offs[5 + i * per_layer_names.len ..];
        l.* = .{ .qkv = o[0], .o = o[3], .q_norm = o[4], .k_norm = o[5], .gate_up = o[6], .down = o[8], .ple_gate = o[9], .ple_proj = o[10], .post_ple = o[11], .input_ln = o[12], .post_attn = o[13], .pre_ff = o[14], .post_ff = o[15], .scalar = g.bf16f(base[o[16]..][0..2]) };
        // stacked reads need each part to end where the next begins
        const hd = g.headDim(i);
        std.debug.assert(o[1] == o[0] + g.heads * hd * g.hidden * 2 and o[2] == o[1] + g.kvHeads(i) * hd * g.hidden * 2);
        std.debug.assert(o[7] == o[6] + g.inter * g.hidden * 2);
    }
    return w;
}

/// Batches of up to this many rows take the few-row products (a simdgroup a column), more the tensor-unit GEMM.
pub const matvec_rows = 32;
pub const matvec_widths = [_]usize{ 4, 8, 16, 32 };

// Gate and up rows interleaved 16 at a time: a 32-column block of the product holds matching gate and up columns.
fn interleave(gu: []u8, gpa: Allocator) !void {
    const row = g.hidden * 2;
    const copy = try gpa.dupe(u8, gu);
    defer gpa.free(copy);
    for (0..g.inter / 16) |b| {
        @memcpy(gu[(32 * b) * row ..][0 .. 16 * row], copy[(16 * b) * row ..][0 .. 16 * row]);
        @memcpy(gu[(32 * b + 16) * row ..][0 .. 16 * row], copy[(g.inter + 16 * b) * row ..][0 .. 16 * row]);
    }
}

test "embedding_gemma2 metal: gate and up interleave 16 rows at a time" {
    const row = g.hidden * 2;
    const gu = try std.testing.allocator.alloc(u8, 2 * g.inter * row);
    defer std.testing.allocator.free(gu);
    for (0..2 * g.inter) |r| @memset(gu[r * row ..][0..row], @truncate(r % 251));
    try interleave(gu, std.testing.allocator);
    try std.testing.expectEqual(@as(u8, 15), gu[15 * row]); // gate row 15
    try std.testing.expectEqual(@as(u8, @truncate(g.inter % 251)), gu[16 * row]); // up row 0
    try std.testing.expectEqual(@as(u8, 16), gu[32 * row]); // gate row 16
    try std.testing.expectEqual(@as(u8, @truncate((g.inter + 16) % 251)), gu[48 * row]); // up row 16
}

pub const Epilogue = enum(u2) { plain, glu, ple };

pub const Pipes = struct {
    gemm: [3][2]mtl.Pipeline, // [epilogue][M % 64 == 0]
    matvec: [3][matvec_widths.len]mtl.Pipeline, // [epilogue][rows]
    embed: mtl.Pipeline,
    rms: mtl.Pipeline,
    qkv: mtl.Pipeline,
    attn256: mtl.Pipeline,
    attn512: mtl.Pipeline,
    attn256_tiled: mtl.Pipeline,
    attn512_tiled: mtl.Pipeline,
    attn256_short: mtl.Pipeline,
    attn512_short: mtl.Pipeline,
    rope_table: mtl.Pipeline,
    add_norm: mtl.Pipeline,
    pool: mtl.Pipeline,

    const names = .{ .{ "embed", "eg2_embed" }, .{ "rms", "eg2_rms" }, .{ "qkv", "eg2_qkv" }, .{ "attn256", "eg2_attn256" }, .{ "attn512", "eg2_attn512" }, .{ "attn256_tiled", "eg2_attn256_tiled" }, .{ "attn512_tiled", "eg2_attn512_tiled" }, .{ "attn256_short", "eg2_attn256_short" }, .{ "attn512_short", "eg2_attn512_short" }, .{ "rope_table", "eg2_rope_table" }, .{ "add_norm", "eg2_add_norm" }, .{ "pool", "eg2_pool" } };

    pub fn load(device: mtl.Device, lib: mtl.Library) !Pipes {
        var p: Pipes = undefined;
        var made: usize = 0;
        errdefer p.release(made);
        const epi = [_][]const u8{ "", "glu_", "ple_" };
        for (epi, 0..) |x, k| for (0..2) |m| {
            var name: [32]u8 = undefined;
            p.gemm[k][m] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name, "eg2_gemm_{s}{d}", .{ x, m }), false);
            made += 1;
        };
        for (epi, 0..) |x, k| for (matvec_widths, 0..) |r, i| {
            var name: [32]u8 = undefined;
            p.matvec[k][i] = try mtl.Pipeline.init(device, lib, try std.fmt.bufPrint(&name, "eg2_matvec_{s}{d}", .{ x, r }), false);
            made += 1;
        };
        inline for (names) |kv| {
            @field(p, kv[0]) = try mtl.Pipeline.init(device, lib, kv[1], false);
            made += 1;
        }
        return p;
    }

    /// The first `n` pipelines in `load`'s order.
    pub fn release(p: *Pipes, n: usize) void {
        var i: usize = 0;
        for (&p.gemm) |*row| for (row) |q| {
            if (i == n) return;
            q.deinit();
            i += 1;
        };
        for (&p.matvec) |*row| for (row) |q| {
            if (i == n) return;
            q.deinit();
            i += 1;
        };
        inline for (names) |kv| {
            if (i == n) return;
            @field(p, kv[0]).deinit();
            i += 1;
        }
    }
};

/// The library: gemm_nax, then eg2.metal, nax.h inlined for this GPU (TF_SIMD_FRAGS without tensor units).
pub fn library(gpa: Allocator, device: mtl.Device, head: []const u8) !mtl.Library {
    const body = try tf.frags.source(device, gpa, tf.embgemma2_kernels);
    defer gpa.free(body);
    const src = try std.mem.concat(gpa, u8, &.{ head, body });
    defer gpa.free(src);
    return mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
}

/// The scratch's parts, in bytes (each 256-aligned when laid out), for `R` rows.
fn scratchParts(R: usize) [17]usize {
    return .{
        4 * R, 4 * R, 4 * R, 4 * R, 4 * R, 4 * R, 4 * g.max_inputs, // ids, pos, full lo/hi, sliding lo/hi, ends
        16 * max_tiles, // the attention tiles: first row, rows, the input's first row
        2 * R * g.hidden, // h: the residual
        2 * R * g.hidden, // xn: the next sub-block's normed input
        2 * R * g.layers * g.per_layer, // pli: the per-layer inputs
        2 * R * (g.heads + 2) * 512, // qkv: the widest layer's q | k | v
        2 * R * g.heads * 512, // att
        2 * R * g.hidden, // o: a sub-block's output before its post norm
        2 * R * g.inter, // m: rbf(gelu(gate)) * up
        2 * R * g.per_layer, // gp: the per-layer input gate's product
        4 * g.max_inputs * g.dim, // out: f32 vectors
    };
}

/// Attention tiles a batch at most: 32 rows of one input each.
const max_tiles = g.max_rows / 32 + g.max_inputs;

pub const pipe_count = 6 + 3 * matvec_widths.len + Pipes.names.len;

const Buffers = struct { ids: usize, pos: usize, flo: usize, fhi: usize, slo: usize, shi: usize, ends: usize, tiles: usize, h: usize, xn: usize, pli: usize, qkv: usize, att: usize, o: usize, m: usize, gp: usize, out: usize };

// auto: fused short attention up to 32 rows, else tiled; TF_EG2_ATTN picks one kind (t256/t512: comparisons).
pub const Attention = enum { auto, tiled, row, short, t256, t512 };

/// The encoder on one GPU queue: weights, kernels and scratch for up to max_rows tokens and max_inputs inputs.
pub const Model = struct {
    gpa: Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    p: Pipes,
    w: Weights,
    scratch: mtl.Buffer,
    b: Buffers,
    rope: mtl.Buffer, // RoPE's cos | sin tables at every position: the sliding layers' (hd 256), then the full ones'
    rope_slide: usize = 0,
    rope_full: usize = 0,
    attention: Attention = .auto,
    few_rows: usize = matvec_rows, // TF_EG2_FEW_ROWS=0: the GEMM at every size (a row's bits then never depend on its batch)
    gpu_ms: f64 = 0, // the last batch's GPU time
    tiles: usize = 0, // the batch's attention tiles

    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8) !*Model {
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .device = undefined, .queue = undefined, .lib = undefined, .p = undefined, .w = undefined, .scratch = undefined, .b = undefined, .rope = undefined };
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.lib = try library(gpa, m.device, "");
        errdefer m.lib.deinit();
        m.p = try Pipes.load(m.device, m.lib);
        errdefer m.p.release(pipe_count);
        m.w = try load(gpa, io, m.device, dir);
        errdefer m.w.buf.deinit();
        const parts = scratchParts(g.max_rows + pad_rows);
        var total: usize = 0;
        for (parts) |n| total += std.mem.alignForward(usize, n, 256);
        m.scratch = try m.device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer m.scratch.deinit();
        m.rope_full = 2 * g.context * 128 * 2;
        m.rope = try m.device.buffer(m.rope_full + 2 * g.context * 256 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer m.rope.deinit();
        try m.ropeTables();
        @memset(m.scratch.contents()[0..total], 0); // the padding rows tiles read past a batch hold finite values
        var offs: [parts.len]usize = undefined;
        var at: usize = 0;
        for (parts, &offs) |n, *o| {
            o.* = at;
            at += std.mem.alignForward(usize, n, 256);
        }
        m.b = .{ .ids = offs[0], .pos = offs[1], .flo = offs[2], .fhi = offs[3], .slo = offs[4], .shi = offs[5], .ends = offs[6], .tiles = offs[7], .h = offs[8], .xn = offs[9], .pli = offs[10], .qkv = offs[11], .att = offs[12], .o = offs[13], .m = offs[14], .gp = offs[15], .out = offs[16] };
        return m;
    }

    /// RoPE's tables on the GPU, once: theta 1e4 at head width 256, then 1e6 at 512.
    fn ropeTables(m: *Model) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(m.p.rope_table);
        for ([_]struct { usize, usize, f32 }{ .{ m.rope_slide, 256, 1e4 }, .{ m.rope_full, 512, 1e6 } }) |t| {
            e.setBuffer(m.rope, t[0], 0);
            e.setValue(extern struct { hd: u32, theta: f32, positions: u32 }{ .hd = @intCast(t[1]), .theta = t[2], .positions = g.context }, 1);
            e.dispatchThreads(mtl.Size.of(t[1] / 2, g.context, 1), mtl.Size.of(128, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("embedding_gemma2: RoPE's tables failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
    }

    pub fn close(m: *Model) void {
        m.rope.deinit();
        m.scratch.deinit();
        m.w.buf.deinit();
        m.p.release(pipe_count);
        m.lib.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }

    pub fn bytes(m: *const Model) usize {
        return m.w.buf.length() + m.scratch.length() + m.rope.length();
    }

    fn ints(m: *Model, off: usize, n: usize) []i32 {
        const p: [*]i32 = @ptrCast(@alignCast(m.scratch.contents() + off));
        return p[0..n];
    }

    /// The encoder over `ids` (inputs ending at `ends`), each input's normalised mean in `out` [inputs, 768].
    pub fn run(m: *Model, ids: []const u32, ends: []const u32, out: []f32) !void {
        const n = ids.len;
        if (n == 0 or n > g.max_rows or ends.len == 0 or ends.len > g.max_inputs or ends[ends.len - 1] != n) return error.InvalidBatch;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const b = m.b;
        for (m.ints(b.ids, n), ids) |*d, s| d.* = @intCast(s);
        g.rows(ends, m.ints(b.pos, n), m.ints(b.flo, n), m.ints(b.fhi, n), m.ints(b.slo, n), m.ints(b.shi, n));
        for (m.ints(b.ends, ends.len), ends) |*d, s| d.* = @intCast(s);
        m.tiles = tileRows(ends, m.ints(b.tiles, 4 * max_tiles));
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        m.encode(e, n, ends.len);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("embedding_gemma2: the batch failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        m.gpu_ms = cb.gpuSeconds() * 1000;
        const v: [*]const f32 = @ptrCast(@alignCast(m.scratch.contents() + b.out));
        @memcpy(out[0 .. ends.len * g.dim], v[0 .. ends.len * g.dim]);
    }

    fn set(m: *Model, e: mtl.ComputeEncoder, bufs: []const struct { bool, usize }) void {
        for (bufs, 0..) |x, i| e.setBuffer(if (x[0]) m.w.buf else m.scratch, x[1], i);
    }

    /// `set` from index 5.
    fn set5(m: *Model, e: mtl.ComputeEncoder, bufs: []const struct { bool, usize }) void {
        for (bufs, 5..) |x, i| e.setBuffer(if (x[0]) m.w.buf else m.scratch, x[1], i);
    }

    fn W(off: usize) struct { bool, usize } {
        return .{ true, off };
    }
    fn S(off: usize) struct { bool, usize } {
        return .{ false, off };
    }

    // y [M, N] = x [M, K] w^T through `epi`: a simdgroup a column for a few rows, else gemm_nax's tiles.
    fn gemm(m: *Model, e: mtl.ComputeEncoder, x: usize, w: usize, y: usize, M: usize, N: usize, K: usize, epi: Epilogue, layer: usize) void {
        const k = @backingInt(epi);
        const ldy = if (epi == .glu) N / 2 else N;
        const ldp = g.layers * g.per_layer;
        e.setBuffer(m.scratch, x, 0);
        e.setBuffer(m.w.buf, w, 1);
        e.setBuffer(m.scratch, y, 3);
        e.setBuffer(m.scratch, m.b.pli, 4);
        if (M <= m.few_rows) {
            const at = for (matvec_widths, 0..) |r, i| {
                if (M <= r) break i;
            } else unreachable;
            e.setPipeline(m.p.matvec[k][at]);
            e.setValue(extern struct { rows: u32, n: u32, k: u32, ldp: u32, poff: u32 }{ .rows = @intCast(M), .n = @intCast(ldy), .k = @intCast(K), .ldp = ldp, .poff = @intCast(layer * g.per_layer) }, 2);
            e.dispatchGroups(mtl.Size.of((ldy + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(m.p.gemm[k][@intFromBool(M % 64 == 0)]);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, K, K, ldy, tn, tm, 2, K / 256, ldp, layer * g.per_layer };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.dispatchThreads(mtl.Size.of((tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
    }

    fn rms(m: *Model, e: mtl.ComputeEncoder, x: usize, w: ?usize, y: usize, rows: usize, n: usize, pre: f32) void {
        e.setPipeline(m.p.rms);
        m.set(e, &.{ S(x), W(w orelse 0), S(y) });
        e.setValue(extern struct { n: u32, eps: f32, pre: f32, scaled: u32 }{ .n = @intCast(n), .eps = g.eps, .pre = pre, .scaled = @intFromBool(w != null) }, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(128, 1, 1));
    }

    fn addNorm(m: *Model, e: mtl.ComputeEncoder, x: usize, wp: usize, wn: ?usize, rows: usize, scalar: f32) void {
        e.setPipeline(m.p.add_norm);
        m.set(e, &.{ S(x), W(wp), S(m.b.h), W(wn orelse 0), S(m.b.xn) });
        e.setValue(extern struct { n: u32, eps: f32, scalar: f32, next: u32 }{ .n = g.hidden, .eps = g.eps, .scalar = scalar, .next = @intFromBool(wn != null) }, 5);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(128, 1, 1));
    }

    fn encode(m: *Model, e: mtl.ComputeEncoder, n: usize, inputs: usize) void {
        const b = m.b;
        const w = m.w;
        e.setPipeline(m.p.embed);
        m.set(e, &.{ S(b.ids), W(w.embed), S(b.h) });
        e.setValue(extern struct { n: u32, scale: f32 }{ .n = g.hidden, .scale = g.embed_scale }, 3);
        e.dispatchThreads(mtl.Size.of(g.hidden, n, 1), mtl.Size.of(256, 1, 1));
        // the per-layer inputs: one projection of the embeddings, scaled by 512^-0.5 and normed per layer slice
        m.gemm(e, b.h, w.ple_model, b.pli, n, g.layers * g.per_layer, g.hidden, .plain, 0);
        m.rms(e, b.pli, w.ple_norm, b.pli, n * g.layers, g.per_layer, 1.0 / @sqrt(@as(f32, g.hidden)));
        m.rms(e, b.h, w.layers[0].input_ln, b.xn, n, g.hidden, 1);
        for (w.layers, 0..) |l, i| {
            const hd = g.headDim(i);
            const kv = g.kvHeads(i);
            m.gemm(e, b.xn, l.qkv, b.qkv, n, g.qkvWidth(i), g.hidden, .plain, 0);
            const lo, const hi = if (g.full(i)) .{ b.flo, b.fhi } else .{ b.slo, b.shi };
            const mode: Attention = switch (m.attention) {
                .auto => if (n <= m.few_rows) .short else .tiled,
                .short => if (n <= matvec_rows) .short else .tiled, // a row reads at most 32 keys there
                .t256 => if (hd == 256) .tiled else .row,
                .t512 => if (hd == 512) .tiled else .row,
                else => m.attention,
            };
            const rope = if (g.full(i)) m.rope_full else m.rope_slide;
            if (mode != .short) {
                e.setPipeline(m.p.qkv);
                m.set(e, &.{ S(b.qkv), W(l.q_norm), W(l.k_norm), S(b.pos) });
                e.setValue(extern struct { hq: u32, hkv: u32, hd: u32, positions: u32, eps: f32 }{ .hq = g.heads, .hkv = @intCast(kv), .hd = @intCast(hd), .positions = g.context, .eps = g.eps }, 4);
                e.setBuffer(m.rope, rope, 5);
                e.dispatchGroups(mtl.Size.of(n, g.heads + 2 * kv, 1), mtl.Size.of(128, 1, 1));
            }
            m.set(e, &.{ S(b.qkv), S(b.att), S(lo), S(hi) });
            switch (mode) {
                .short => {
                    e.setPipeline(if (hd == 512) m.p.attn512_short else m.p.attn256_short);
                    e.setValue(extern struct { rows: u32, hq: u32, hkv: u32, positions: u32, eps: f32 }{ .rows = @intCast(n), .hq = g.heads, .hkv = @intCast(kv), .positions = g.context, .eps = g.eps }, 4);
                    m.set5(e, &.{ S(b.pos), W(l.q_norm), W(l.k_norm) });
                    e.setBuffer(m.rope, rope, 8);
                    e.dispatchGroups(mtl.Size.of((n * g.heads + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
                },
                .row, .t256, .t512, .auto => {
                    e.setPipeline(if (hd == 512) m.p.attn512 else m.p.attn256);
                    e.setValue(extern struct { rows: u32, hq: u32, hkv: u32 }{ .rows = @intCast(n), .hq = g.heads, .hkv = @intCast(kv) }, 4);
                    e.dispatchGroups(mtl.Size.of((n * g.heads + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
                },
                .tiled => {
                    e.setPipeline(if (hd == 512) m.p.attn512_tiled else m.p.attn256_tiled);
                    e.setValue(extern struct { rows: u32, hq: u32, hkv: u32 }{ .rows = @intCast(n), .hq = g.heads, .hkv = @intCast(kv) }, 4);
                    e.setBuffer(m.scratch, b.tiles, 5);
                    e.setThreadgroupMemory(attnTgBytes(hd), 0);
                    e.dispatchGroups(mtl.Size.of(m.tiles, g.heads, 1), mtl.Size.of(32 * hd / 64, 1, 1));
                },
            }
            m.gemm(e, b.att, l.o, b.o, n, g.hidden, g.heads * hd, .plain, 0);
            m.addNorm(e, b.o, l.post_attn, l.pre_ff, n, 1);
            m.gemm(e, b.xn, l.gate_up, b.m, n, 2 * g.inter, g.hidden, .glu, 0);
            m.gemm(e, b.m, l.down, b.o, n, g.hidden, g.inter, .plain, 0);
            m.addNorm(e, b.o, l.post_ff, null, n, 1);
            m.gemm(e, b.h, l.ple_gate, b.gp, n, g.per_layer, g.hidden, .ple, i);
            m.gemm(e, b.gp, l.ple_proj, b.o, n, g.hidden, g.per_layer, .plain, 0);
            const next = if (i + 1 < g.layers) w.layers[i + 1].input_ln else w.norm;
            m.addNorm(e, b.o, l.post_ple, next, n, l.scalar);
        }
        e.setPipeline(m.p.pool);
        m.set(e, &.{ S(b.xn), S(b.ends), W(w.projection), S(b.out) });
        e.setValue(extern struct { n: u32, dim: u32 }{ .n = g.hidden, .dim = g.dim }, 4);
        e.dispatchGroups(mtl.Size.of(inputs, 1, 1), mtl.Size.of(256, 1, 1));
    }
};

/// The tiled attention's threadgroup memory: 32 rows of 136 fp32 probabilities, then a row max per simdgroup.
pub fn attnTgBytes(hd: usize) usize {
    return 4 * (32 * 136 + hd / 64 * 32);
}

// A packed batch's attention tiles (first row, rows, input start, 0): 32 rows of one input each; their count.
pub fn tileRows(ends: []const u32, out: []i32) usize {
    var n: usize = 0;
    var at: u32 = 0;
    for (ends) |end| {
        var r = at;
        while (r < end) : (r += 32) {
            out[4 * n ..][0..4].* = .{ @intCast(r), @intCast(@min(32, end - r)), @intCast(at), 0 };
            n += 1;
        }
        at = end;
    }
    return n;
}

test "embedding_gemma2 metal: attention tiles start at each input's first row" {
    var t: [4 * 8]i32 = undefined;
    try std.testing.expectEqual(@as(usize, 4), tileRows(&.{ 3, 40, 72 }, &t));
    try std.testing.expectEqualSlices(i32, &.{ 0, 3, 0, 0, 3, 32, 3, 0, 35, 5, 3, 0, 40, 32, 40, 0 }, t[0..16]);
}
