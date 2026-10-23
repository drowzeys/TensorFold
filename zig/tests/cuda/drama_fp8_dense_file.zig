//! drama-fp8-dense CHECKPOINT LAYER [SOURCE] (f): a layer as the production loader stored it, under (a) to (d) rules.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const drama = @import("drama");
const check = @import("check.zig");
const ref = @import("drama_fp8_dense_ref.zig");

const backbone = drama.backbone;
const config = backbone.config;
const weights = backbone.weights;
const Model = backbone.Model;
const Checkpoint = core.checkpoint.Checkpoint;
const Desc = ref.Desc;
const read = ref.read;
const bf16Rows = ref.bf16Rows;

const H = config.hidden;
const rows = 128;

fn widen(a: std.mem.Allocator, v: []const u16) ![]f64 {
    const out = try a.alloc(f64, v.len);
    for (out, v) |*o, b| o.* = @as(f32, @bitCast(@as(u32, b) << 16));
    return out;
}

pub fn run(gpu: check.Gpu, dir: []const u8, layer_arg: []const u8, source: ?[]const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const layer = try std.fmt.parseInt(usize, layer_arg, 10);
    if (layer >= config.layers) return error.Usage;
    // Model.init takes the context mutable; the runner's is only borrowed for this run
    var model = try Model.init(gpu.gpa, gpu.io, @constCast(gpu.ctx), dir, "", .{ .context = 2 * rows, .max_rows = rows });
    defer model.deinit();
    var note: [160]u8 = undefined;
    var w: std.Io.Writer = .fixed(&note);
    try model.w.denseFp8().write(&w, config.layers);
    std.debug.print("RESULT drama-fp8-dense: {s} loaded, dense FP8 in {s}\n", .{ dir, w.buffered() });
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xF11E_0000 + layer);
    var r: ref.Rig = .{ .gpu = gpu, .a = a, .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var ck = try Checkpoint.openModel(a, gpu.io, dir);
    defer ck.close();
    var src: ?Checkpoint = if (source) |p| try Checkpoint.openModel(a, gpu.io, p) else null;
    defer if (src) |*c| c.close();
    const l = model.w.layers[layer];
    if (config.linear(layer)) {
        inline for (ref.mixers[0]) |d| try group(&r, &model, &ck, if (src) |*c| c else null, layer, l, d);
    } else inline for (ref.mixers[1]) |d| try group(&r, &model, &ck, if (src) |*c| c else null, layer, l, d);
    inline for (ref.shared) |d| try group(&r, &model, &ck, if (src) |*c| c else null, layer, l, d);
    check.pass("drama-fp8-dense {s} layer {d}: every group as stored, FP8 groups within (a), (b), (c) and (d)", .{ dir, layer });
}

fn group(r: *ref.Rig, model: *Model, ck: *Checkpoint, src: ?*Checkpoint, layer: usize, l: weights.Layer, comptime d: Desc) !void {
    const a = r.a;
    var n: usize = 0;
    for (d.parts) |p| n += p;
    var what_buf: [96]u8 = undefined;
    const what = try std.fmt.bufPrint(&what_buf, "layer {d} {s} [{d}, {d}]", .{ layer, d.name, n, d.k });
    const extra = try bf16Rows(a, try read(a, ck, layer, d.extra, d.extra_rows, d.k, ".weight", .bf16));
    const q = @field(l.fp8, d.name);
    const fused: []const u16 = if (q) |w| blk: {
        // the loader's packing against the host's, byte for byte
        const g = try ref.Group.alloc(a, n, d.k);
        @memcpy(g.codes, try read(a, ck, layer, d.stems, d.parts, d.k, ".weight", .f8_e4m3));
        @memcpy(std.mem.sliceAsBytes(g.inv), try read(a, ck, layer, d.stems, d.parts, d.k / 64, ".weight_scale_inv", .f32));
        const p = try g.toLane(a);
        try check.sameBytes(what, try r.back(w.fp8g.codes, p.codes.len), p.codes);
        try check.sameBytes(what, try r.back(w.fp8g.scales, p.tiles.len * 4), std.mem.sliceAsBytes(p.tiles));
        if (comptime d.extra.len > 0) try check.sameBytes(what, try r.back(@field(l, d.split), extra.len * 2), std.mem.sliceAsBytes(extra));
        try lane(r, model, src, layer, what, g, w, d);
        break :blk try g.bf16(a);
    } else blk: {
        const stored = try bf16Rows(a, try read(a, ck, layer, d.stems, d.parts, d.k, ".weight", .bf16));
        try check.sameBytes(what, try r.back(@field(l, d.bf16), (stored.len + extra.len) * 2), std.mem.sliceAsBytes(try std.mem.concat(a, u16, &.{ stored, extra })));
        std.debug.print("RESULT {s}: BF16, on the device as stored\n", .{what});
        break :blk stored;
    };
    if (comptime d.extra.len > 0) try split(r, &model.k, what, fused, extra, n);
}

/// (a), (b), (c) on the stored weights, and with a source checkpoint both paths' distance to its BF16 weights.
fn lane(r: *ref.Rig, model: *Model, src: ?*Checkpoint, layer: usize, what: []const u8, g: ref.Group, w: cuda.qlinear.Weight, comptime d: Desc) !void {
    const a = r.a;
    const x = try r.bfs(rows * g.k, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const want = try ref.reference(a, try g.dequant(a), g.n, g.k, x, rows);
    const lin: cuda.qlinear.Linear = .{ .lane = &model.k.lane.? };
    const got = try ref.rowCounts(r, lin, w, dx, rows, model.s.part, what);
    const fp8 = try ref.stats(a, got, want, g.n, null);
    try ref.gate(what, g.n, fp8);
    const wb = try r.dev(std.mem.sliceAsBytes(try g.bf16(a)));
    const yb = try r.zeros(rows * g.n * 2);
    var name: [128]u8 = undefined;
    for ([_]usize{ 1, 7, rows }) |chunk| {
        try ref.gemmChunked(&model.k, r.s, dx, g.k, rows, chunk, wb, g.n, yb);
        try ref.noWorse(try std.fmt.bufPrint(&name, "{s} in calls of {d} rows", .{ what, chunk }), fp8, try ref.stats(a, try r.back(yb, rows * g.n * 2), want, g.n, null));
    }
    const c = src orelse return;
    const source = try bf16Rows(a, try read(a, c, layer, d.stems, d.parts, d.k, ".weight", .bf16));
    const truth = try ref.reference(a, try widen(a, source), g.n, g.k, x, rows);
    try ref.measured(try std.fmt.bufPrint(&name, "{s} FP8 lane against the source's BF16 weights", .{what}), g.n, try ref.stats(a, got, truth, g.n, null));
    try model.k.gemm(r.s, dx, g.k, rows, try r.dev(std.mem.sliceAsBytes(source)), g.n, g.k, yb, g.n);
    try ref.measured(try std.fmt.bufPrint(&name, "{s} source BF16 gemm against its own weights", .{what}), g.n, try ref.stats(a, try r.back(yb, rows * g.n * 2), truth, g.n, null));
}

/// (d) on the layer's weights: the split-off rows' own gemm against the fused BF16 gemm's columns, bit for bit.
fn split(r: *ref.Rig, k: *const backbone.kernels.Kernels, what: []const u8, fused: []const u16, extra: []const u16, n: usize) !void {
    const a = r.a;
    const m = extra.len / H;
    const all = n + m;
    const fw = try r.dev(std.mem.sliceAsBytes(try std.mem.concat(a, u16, &.{ fused, extra })));
    const ew = try r.dev(std.mem.sliceAsBytes(extra));
    const dx = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * H, 1.0)));
    for ([_]usize{ 1, 7, rows }) |count| {
        const yf = try r.zeros(count * all * 2);
        try k.gemm(r.s, dx, H, count, fw, all, H, yf, all);
        const ye = try r.zeros(count * m * 2);
        try k.gemm(r.s, dx, H, count, ew, m, H, ye, m);
        const f = try r.back(yf, count * all * 2);
        const e = try r.back(ye, count * m * 2);
        for (0..count) |row| try check.sameBytes(what, e[row * m * 2 ..][0 .. m * 2], f[(row * all + n) * 2 ..][0 .. m * 2]);
    }
    check.pass("{s}: its {d} split-off rows equal the fused gemm's columns at 1, 7 and {d} rows", .{ what, m, rows });
}
