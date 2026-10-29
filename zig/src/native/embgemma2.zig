//! EmbeddingGemma-2's text encoder on CUDA: OpenAI embeddings from the bf16 release; it writes no text.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const core = @import("core");
const Allocator = std.mem.Allocator;
const vit = cuda.vit;

pub const family: api.Family = .{ .model_type = "embedding_gemma2", .formats = &.{"bf16"} };

// config.json's text_config (checked at load)
const hidden = 512;
const inter = 2048;
const layers = 24;
const per_layer = 512; // hidden_size_per_layer_input
const heads = 4;
const vocab = 262144;
const dim = 768; // embedding_dim
const window = 512; // sliding_window: |q - k| <= 512 on the sliding layers
const eps: f32 = 1e-6;
const context = 8192;
/// Tokens a batch at most (every input's rows together).
const max_rows = 8192;
const max_inputs = 128;
const mrl = [_]u32{ 512, 256, 128 };

/// Layers 5, 11, 17 and 23 attend over the whole input with 512-wide heads and one kv head; the rest slide.
fn full(i: usize) bool {
    return i % 6 == 5;
}
fn headDim(i: usize) usize {
    return if (full(i)) 512 else 256;
}
fn kvHeads(i: usize) usize {
    return if (full(i)) 1 else 2;
}
fn qkvWidth(i: usize) usize {
    return (heads + 2 * kvHeads(i)) * headDim(i);
}

const Layer = struct {
    qkv: u64,
    o: u64,
    gate_up: u64,
    down: u64,
    ple_gate: u64,
    ple_proj: u64,
    input_ln: u64,
    post_attn: u64,
    pre_ff: u64,
    post_ff: u64,
    post_ple: u64,
    q_norm: u64,
    k_norm: u64,
    scalar: f32,
};

const Weights = struct {
    embed: u64,
    ple_model: u64,
    ple_norm: u64,
    norm: u64,
    projection: u64,
    layers: [layers]Layer,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn deinit(w: *Weights, gpa: Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
    }
};

const Loader = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    ck: *core.checkpoint.Checkpoint,
    w: *Weights,

    fn up(x: *Loader, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(x.d, bytes);
        errdefer {
            var y = b;
            y.free();
        }
        try x.w.buffers.append(x.gpa, b);
        return b.ptr;
    }

    fn tensor(x: *Loader, name: []const u8, shape: []const usize) ![]const u8 {
        return (try x.ck.expect(name, .bf16, shape)).bytes;
    }

    fn one(x: *Loader, name: []const u8, shape: []const usize) !u64 {
        return x.up(try x.tensor(name, shape));
    }

    /// Row blocks of several [rows, hidden] tensors stacked into one device matrix (one GEMM for all of them).
    fn stacked(x: *Loader, parts: []const []const u8) !u64 {
        var n: usize = 0;
        for (parts) |p| n += p.len;
        const host = try x.gpa.alloc(u8, n);
        defer x.gpa.free(host);
        var at: usize = 0;
        for (parts) |p| {
            @memcpy(host[at..][0..p.len], p);
            at += p.len;
        }
        return x.up(host);
    }
};

fn key(b: []u8, i: usize, s: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "language_model.layers.{d}.{s}", .{ i, s });
}

fn bf16f(bytes: []const u8) f32 {
    const bits: u32 = @as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16;
    return @bitCast(bits);
}

fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var w: Weights = undefined;
    w.buffers = .empty;
    errdefer w.deinit(gpa);
    var x: Loader = .{ .gpa = gpa, .d = d, .ck = &ck, .w = &w };
    w.embed = try x.one("language_model.embed_tokens.weight", &.{ vocab, hidden });
    w.ple_model = try x.one("language_model.ple.per_layer_model_projection.weight", &.{ layers * per_layer, hidden });
    w.ple_norm = try x.one("language_model.ple.per_layer_projection_norm.weight", &.{per_layer});
    w.norm = try x.one("language_model.norm.weight", &.{hidden});
    w.projection = try x.one("language_model.embedding_projection.weight", &.{ dim, hidden });
    var b1: [96]u8 = undefined;
    var b2: [96]u8 = undefined;
    var b3: [96]u8 = undefined;
    for (&w.layers, 0..) |*l, i| {
        const hd = headDim(i);
        const kv = kvHeads(i);
        l.qkv = try x.stacked(&.{
            try x.tensor(try key(&b1, i, "self_attn.q_proj.weight"), &.{ heads * hd, hidden }),
            try x.tensor(try key(&b2, i, "self_attn.k_proj.weight"), &.{ kv * hd, hidden }),
            try x.tensor(try key(&b3, i, "self_attn.v_proj.weight"), &.{ kv * hd, hidden }),
        });
        l.o = try x.one(try key(&b1, i, "self_attn.o_proj.weight"), &.{ hidden, heads * hd });
        l.q_norm = try x.one(try key(&b1, i, "self_attn.q_norm.weight"), &.{hd});
        l.k_norm = try x.one(try key(&b1, i, "self_attn.k_norm.weight"), &.{hd});
        l.gate_up = try x.stacked(&.{
            try x.tensor(try key(&b1, i, "mlp.gate_proj.weight"), &.{ inter, hidden }),
            try x.tensor(try key(&b2, i, "mlp.up_proj.weight"), &.{ inter, hidden }),
        });
        l.down = try x.one(try key(&b1, i, "mlp.down_proj.weight"), &.{ hidden, inter });
        l.ple_gate = try x.one(try key(&b1, i, "ple_block.per_layer_input_gate.weight"), &.{ per_layer, hidden });
        l.ple_proj = try x.one(try key(&b1, i, "ple_block.per_layer_projection.weight"), &.{ hidden, per_layer });
        l.post_ple = try x.one(try key(&b1, i, "ple_block.post_per_layer_input_norm.weight"), &.{hidden});
        l.input_ln = try x.one(try key(&b1, i, "input_layernorm.weight"), &.{hidden});
        l.post_attn = try x.one(try key(&b1, i, "post_attention_layernorm.weight"), &.{hidden});
        l.pre_ff = try x.one(try key(&b1, i, "pre_feedforward_layernorm.weight"), &.{hidden});
        l.post_ff = try x.one(try key(&b1, i, "post_feedforward_layernorm.weight"), &.{hidden});
        l.scalar = bf16f(try x.tensor(try key(&b1, i, "layer_scalar"), &.{1}));
    }
    return w;
}

/// config.json must be this geometry: the kernels and buffers are sized for it.
fn admit(a: Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    const t = (v.object.get("text_config") orelse return error.NoTextConfig).object;
    const want = .{ .{ "hidden_size", hidden }, .{ "intermediate_size", inter }, .{ "num_hidden_layers", layers }, .{ "num_attention_heads", heads }, .{ "num_key_value_heads", 2 }, .{ "head_dim", 256 }, .{ "vocab_size", vocab }, .{ "embedding_dim", dim }, .{ "sliding_window", window }, .{ "hidden_size_per_layer_input", per_layer } };
    inline for (want) |p| {
        const got = t.get(p[0]) orelse return error.EmbeddingGemma2Geometry;
        if (got != .integer or got.integer != p[1]) return error.EmbeddingGemma2Geometry;
    }
    const types = (t.get("layer_types") orelse return error.EmbeddingGemma2Geometry).array.items;
    if (types.len != layers) return error.EmbeddingGemma2Geometry;
    for (types, 0..) |ty, i| if (!std.mem.eql(u8, ty.string, if (full(i)) "full_attention" else "sliding_attention")) return error.EmbeddingGemma2Geometry;
}

const Glue = struct {
    mod: cuda.Module,
    embed: cuda.Function,
    rms: cuda.Function,
    qkv: cuda.Function,
    attn256: cuda.Function,
    attn512: cuda.Function,
    add_norm: cuda.Function,
    gelu_mul: cuda.Function,
    ple_gate: cuda.Function,
    pool: cuda.Function,

    fn load(d: *const cuda.Driver) !Glue {
        var mod = try cuda.Module.load(d, cuda.kernels.eg2);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .embed = try mod.function("tf_eg2_embed"),
            .rms = try mod.function("tf_eg2_rms"),
            .qkv = try mod.function("tf_eg2_qkv"),
            .attn256 = try mod.function("tf_eg2_attn256"),
            .attn512 = try mod.function("tf_eg2_attn512"),
            .add_norm = try mod.function("tf_eg2_add_norm"),
            .gelu_mul = try mod.function("tf_eg2_gelu_mul"),
            .ple_gate = try mod.function("tf_eg2_ple_gate"),
            .pool = try mod.function("tf_eg2_pool"),
        };
    }
};

fn grid(n: usize, by: usize) c_uint {
    return @intCast((n + by - 1) / by);
}

const Buffers = struct { ids: u64, pos: u64, lo: u64, hi: u64, ends: u64, h: u64, xn: u64, pli: u64, qkv: u64, att: u64, o: u64, gu: u64, m: u64, g: u64, p: u64, out: u64 };

const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    w: Weights,
    glue: Glue,
    gemm: vit.Gemms,
    scratch: cuda.DeviceBuffer,
    b: Buffers,
    host: []i32, // ids | pos | lo | hi for max_rows, then ends
    mutex: std.Io.Mutex = .init, // one batch at a time: HTTP threads call submit directly

    fn of(ctx: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .name = "embedding_gemma2", .text_generation = false, .thinking_budget = false, .context_window = context, .embeddings = .{ .dimensions = dim, .vocabulary = vocab, .max_batch_tokens = max_rows, .max_inputs = max_inputs, .mrl_dimensions = &mrl } };
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .peak = u.peak };
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = of(ctx);
        const batch = r.embedding orelse return error.InvalidRequest;
        if (r.prompt.len == 0 or r.prompt.len > max_rows or batch.ends.len > max_inputs) return error.InvalidRequest;
        e.mutex.lockUncancelable(e.io);
        const ran = e.run(r.prompt, batch.ends, batch.output);
        e.mutex.unlock(e.io);
        const done: api.Event = if (ran) |_| .{ .finished = .{ .reason = .length } } else |err| blk: {
            std.log.err("embedding_gemma2: the batch failed: {s}", .{@errorName(err)});
            break :blk .{ .finished = .{ .reason = .failed, .message = "embedding inference failed" } };
        };
        sink.event(sink.ctx, id, &done);
    }

    fn launch(e: *Engine, f: cuda.Function, g: cuda.launch.Dim3, block: c_uint, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = g, .block = .{ .x = block } }, e.stream, a);
    }

    fn rms(e: *Engine, x: u64, w: u64, y: u64, rows: usize, n: usize, pre: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, w, y }) |p| a.add(p);
        a.add(@as(c_int, @intCast(n)));
        a.add(eps);
        a.add(pre);
        try e.launch(e.glue.rms, .{ .x = @intCast(rows) }, 128, &a);
    }

    fn addNorm(e: *Engine, x: u64, wp: u64, wn: u64, rows: usize, scalar: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, wp, e.b.h, wn, e.b.xn }) |p| a.add(p);
        a.add(@as(c_int, hidden));
        a.add(eps);
        a.add(scalar);
        try e.launch(e.glue.add_norm, .{ .x = @intCast(rows) }, 128, &a);
    }

    /// The encoder over `ids` (inputs ending at `ends`), each input's normalised mean in `out` [inputs, 768].
    fn run(e: *Engine, ids: []const u32, ends: []const u32, out: []f32) !void {
        try e.ctx.makeCurrent();
        const n = ids.len;
        const b = e.b;
        // host rows: token, position in its input, its input's first and end rows
        const hid = e.host[0..n];
        const hpos = e.host[max_rows..][0..n];
        const hlo = e.host[2 * max_rows ..][0..n];
        const hhi = e.host[3 * max_rows ..][0..n];
        const hend = e.host[4 * max_rows ..][0..ends.len];
        var at: u32 = 0;
        for (ends, 0..) |end, k| {
            for (at..end) |r| {
                hid[r] = @intCast(ids[r]);
                hpos[r] = @intCast(r - at);
                hlo[r] = @intCast(at);
                hhi[r] = @intCast(end);
            }
            hend[k] = @intCast(end);
            at = end;
        }
        const d = &e.driver;
        inline for (.{ .{ b.ids, hid }, .{ b.pos, hpos }, .{ b.lo, hlo }, .{ b.hi, hhi }, .{ b.ends, hend } }) |c|
            try d.check(d.api.cuMemcpyHtoDAsync_v2(c[0], c[1].ptr, c[1].len * 4, e.stream.handle), "cuMemcpyHtoDAsync");
        {
            var a: cuda.Args = .{};
            for ([_]u64{ b.ids, e.w.embed, b.h }) |p| a.add(p);
            a.add(@as(c_int, hidden));
            a.add(@as(f32, 22.625)); // sqrt(512) in bf16
            try e.launch(e.glue.embed, .{ .x = @intCast(n) }, 128, &a);
        }
        // the per-layer inputs: one projection of the embeddings, scaled by 512^-0.5 and normed per layer slice
        try e.gemm.run(e.stream, b.h, e.w.ple_model, b.pli, n, layers * per_layer, hidden, .bf16);
        try e.rms(b.pli, e.w.ple_norm, b.pli, n * layers, per_layer, 1.0 / @sqrt(@as(f32, hidden)));
        try e.rms(b.h, e.w.layers[0].input_ln, b.xn, n, hidden, 1);
        for (e.w.layers, 0..) |l, i| {
            const hd = headDim(i);
            const kv = kvHeads(i);
            try e.gemm.run(e.stream, b.xn, l.qkv, b.qkv, n, qkvWidth(i), hidden, .bf16);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ b.qkv, l.q_norm, l.k_norm, b.pos }) |p| a.add(p);
                a.add(@as(c_int, heads));
                a.add(@as(c_int, @intCast(kv)));
                a.add(@as(c_int, @intCast(hd)));
                a.add(@as(f32, if (full(i)) 1e6 else 1e4));
                a.add(eps);
                try e.launch(e.glue.qkv, .{ .x = @intCast(n), .y = @intCast(heads + 2 * kv) }, 128, &a);
            }
            {
                var a: cuda.Args = .{};
                for ([_]u64{ b.qkv, b.att, b.lo, b.hi }) |p| a.add(p);
                a.add(@as(c_int, @intCast(n)));
                a.add(@as(c_int, heads));
                a.add(@as(c_int, @intCast(kv)));
                a.add(@as(c_int, if (full(i)) 0 else window));
                try e.launch(if (hd == 512) e.glue.attn512 else e.glue.attn256, .{ .x = grid(n * heads * 32, 128) }, 128, &a);
            }
            try e.gemm.run(e.stream, b.att, l.o, b.o, n, hidden, heads * hd, .bf16);
            try e.addNorm(b.o, l.post_attn, l.pre_ff, n, 1);
            try e.gemm.run(e.stream, b.xn, l.gate_up, b.gu, n, 2 * inter, hidden, .bf16);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ b.gu, b.m }) |p| a.add(p);
                a.add(@as(c_int, inter));
                a.add(@as(i64, @intCast(n * inter)));
                try e.launch(e.glue.gelu_mul, .{ .x = grid(n * inter, 256) }, 256, &a);
            }
            try e.gemm.run(e.stream, b.m, l.down, b.o, n, hidden, inter, .bf16);
            try e.addNorm(b.o, l.post_ff, 0, n, 1);
            try e.gemm.run(e.stream, b.h, l.ple_gate, b.g, n, per_layer, hidden, .bf16);
            {
                var a: cuda.Args = .{};
                for ([_]u64{ b.g, b.pli, b.p }) |p| a.add(p);
                a.add(@as(c_int, per_layer));
                a.add(@as(c_int, layers));
                a.add(@as(c_int, @intCast(i)));
                a.add(@as(i64, @intCast(n * per_layer)));
                try e.launch(e.glue.ple_gate, .{ .x = grid(n * per_layer, 256) }, 256, &a);
            }
            try e.gemm.run(e.stream, b.p, l.ple_proj, b.o, n, hidden, per_layer, .bf16);
            const next = if (i + 1 < layers) e.w.layers[i + 1].input_ln else e.w.norm;
            try e.addNorm(b.o, l.post_ple, next, n, l.scalar);
        }
        {
            var a: cuda.Args = .{};
            for ([_]u64{ b.xn, b.ends, e.w.projection, b.out }) |p| a.add(p);
            a.add(@as(c_int, hidden));
            a.add(@as(c_int, dim));
            try e.launch(e.glue.pool, .{ .x = @intCast(ends.len) }, 256, &a);
        }
        try d.check(d.api.cuMemcpyDtoHAsync_v2(out.ptr, b.out, ends.len * dim * 4, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
    }

    fn close(ctx: *anyopaque) void {
        const e = of(ctx);
        e.ctx.makeCurrent() catch {};
        e.gpa.free(e.host);
        e.scratch.free();
        e.gemm.deinit();
        e.glue.mod.unload();
        e.w.deinit(e.gpa);
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    const vtable: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory };
};

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (o.context) |cx| if (cx < 1 or cx > context) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: EmbeddingGemma-2 reads 1 to {d} tokens an input", .{ cx, context });
        return null;
    };
    admit(a, io, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "embedding_gemma2: config.json is not the released text geometry ({s})", .{@errorName(err)});
        return null;
    };
    if (!cuda.kernels.available) {
        problem.* = "embedding_gemma2 needs the embedded CUDA kernel images";
        return null;
    }
    const e = try gpa.create(Engine);
    errdefer gpa.destroy(e);
    e.* = .{ .gpa = gpa, .io = io, .driver = undefined, .ctx = undefined, .stream = undefined, .w = undefined, .glue = undefined, .gemm = undefined, .scratch = undefined, .b = undefined, .host = &.{} };
    e.driver = cuda.Driver.open() catch |err| {
        problem.* = try std.fmt.allocPrint(a, "no CUDA driver ({s})", .{@errorName(err)});
        return null;
    };
    errdefer e.driver.close();
    e.ctx = try cuda.Context.init(&e.driver, 0);
    errdefer e.ctx.deinit();
    try e.ctx.makeCurrent();
    e.stream = try cuda.Stream.init(&e.driver, false);
    errdefer e.stream.deinit();
    e.w = load(gpa, io, &e.driver, o.dir) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "embedding_gemma2: the bf16 text encoder did not load ({s})", .{@errorName(err)});
        return null;
    };
    errdefer e.w.deinit(gpa);
    e.glue = try Glue.load(&e.driver);
    errdefer e.glue.mod.unload();
    e.gemm = try vit.Gemms.init(gpa, &e.driver);
    errdefer e.gemm.deinit();
    const r = max_rows;
    const sizes = [_]usize{
        4 * r, 4 * r, 4 * r, 4 * r, 4 * max_inputs, // ids, pos, lo, hi, ends
        2 * r * hidden, // h: the residual
        2 * r * hidden, // xn: the next sub-block's normed input
        2 * r * layers * per_layer, // pli: the per-layer inputs
        2 * r * (heads + 2) * 512, // qkv: the widest layer's q | k | v
        2 * r * heads * 512, // att
        2 * r * hidden, // o: a sub-block's output before its post norm
        2 * r * 2 * inter, // gu: gate | up
        2 * r * inter, // m
        2 * r * per_layer, // g
        2 * r * per_layer, // p
        4 * max_inputs * dim, // out: f32 vectors
    };
    var total: usize = 0;
    for (sizes) |sz| total += std.mem.alignForward(usize, sz, 256);
    e.scratch = try cuda.DeviceBuffer.alloc(&e.driver, total);
    errdefer e.scratch.free();
    var ptrs: [sizes.len]u64 = undefined;
    var off: usize = 0;
    for (sizes, &ptrs) |sz, *p| {
        p.* = e.scratch.ptr + off;
        off += std.mem.alignForward(usize, sz, 256);
    }
    e.b = .{ .ids = ptrs[0], .pos = ptrs[1], .lo = ptrs[2], .hi = ptrs[3], .ends = ptrs[4], .h = ptrs[5], .xn = ptrs[6], .pli = ptrs[7], .qkv = ptrs[8], .att = ptrs[9], .o = ptrs[10], .gu = ptrs[11], .m = ptrs[12], .g = ptrs[13], .p = ptrs[14], .out = ptrs[15] };
    e.host = try gpa.alloc(i32, 4 * max_rows + max_inputs);
    std.log.info("EmbeddingGemma-2 on CUDA: {d}-dimensional vectors (MRL {d}, {d}, {d}), up to {d} inputs and {d} tokens a batch", .{ dim, mrl[0], mrl[1], mrl[2], max_inputs, max_rows });
    return .{ .engine = .{ .ctx = e, .vtable = &Engine.vtable }, .close = Engine.close, .ctx = e };
}
