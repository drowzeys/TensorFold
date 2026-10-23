//! Drama's MoE on synthetic NVFP4 experts: float64 envelope, held-out sets, exact epilogue and combine, FP8 beside.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");
const ref = @import("drama_nvfp4_ref.zig");
const exact = @import("drama_nvfp4_exact.zig");
const h = @import("quant_ref_tests.zig").helpers;

const backbone = drama.backbone;
const weights = backbone.weights;
const Kernels = backbone.kernels.Kernels;
const Model = backbone.Model;

const H = ref.H;
const I = ref.I;
const E = ref.E;
const top = ref.top;
const rows = ref.rows;
const fp8_layer = 0;
const nvfp4_layer = 1;
const sparse_layer = 2;
const held_layer = 3; // the held-out sets' layers: 3, 4, 5

/// Block-FP8 experts for the layer beside, in the device layout: e4m3 [E][gu_rows][H] and [E][H][I], fp32 scales.
const Fp8Layer = struct {
    gu: []u8,
    gu_s: []f32,
    dn: []u8,
    dn_s: []f32,

    fn random(a: std.mem.Allocator, rng: std.Random) !Fp8Layer {
        return .{ .gu = try codes(a, rng, E * weights.gu_rows * H), .gu_s = try scales(a, rng, E * weights.gu_scales), .dn = try codes(a, rng, E * H * I), .dn_s = try scales(a, rng, E * weights.dn_scales) };
    }

    fn codes(a: std.mem.Allocator, rng: std.Random, n: usize) ![]u8 {
        const out = try a.alloc(u8, n);
        rng.bytes(out);
        for (out) |*b| if (b.* & 0x7f == 0x7f) {
            b.* -= 1; // e4m3 NaN codes out: a checkpoint holds none
        };
        return out;
    }

    fn scales(a: std.mem.Allocator, rng: std.Random, n: usize) ![]f32 {
        const out = try a.alloc(f32, n);
        for (out) |*v| v.* = 1e-4 * (1 + rng.float(f32));
        return out;
    }
};

/// Device copies the test owns (the models it builds borrow them).
const Bufs = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn dev(b: *Bufs, bytes: []const u8) !u64 {
        const buf = try cuda.DeviceBuffer.fromHost(b.d, bytes);
        try b.list.append(b.gpa, buf);
        return buf.ptr;
    }

    fn room(b: *Bufs, bytes: usize) !u64 {
        const buf = try cuda.DeviceBuffer.alloc(b.d, bytes);
        try b.list.append(b.gpa, buf);
        return buf.ptr;
    }

    fn deinit(b: *Bufs) void {
        for (b.list.items) |*buf| buf.free();
        b.list.deinit(b.gpa);
    }
};

fn dense(b: *Bufs, sy: ref.Synth) !weights.Layer {
    return .{
        .moe_in = try b.dev(std.mem.sliceAsBytes(sy.moe_in)),
        .shared_down = try b.dev(std.mem.sliceAsBytes(sy.shared_down)),
        .shared_gate = try b.dev(std.mem.sliceAsBytes(sy.shared_gate)),
    };
}

/// The NVFP4 layer of `sy` on the device, staged as stored and packed by the loader's packFp4.
fn fp4Layer(gpu: check.Gpu, b: *Bufs, s: cuda.Stream, packer: *const cuda.modelopt.Packer, sy: ref.Synth) !weights.Layer {
    const f = sy.experts.fp4;
    var staged = try cuda.DeviceBuffer.alloc(gpu.d, 3 * weights.fp4_part);
    defer staged.free();
    for (f, 0..) |p, j| {
        try staged.upload(j * weights.fp4_part, p.codes);
        try staged.upload(j * weights.fp4_part + p.codes.len, p.scales);
    }
    const W = cuda.experts.Nvfp4Experts;
    var up_scale: [E][2]f32 = undefined;
    for (&up_scale, f[0].global, f[1].global) |*u, g, v| u.* = .{ g, v };
    var ex = weights.fp4_layer;
    ex.up = try b.room(W.bytes(E, I, H, 2));
    ex.down = try b.room(W.bytes(E, H, I, 1));
    ex.up_scale = try b.dev(std.mem.asBytes(&up_scale));
    ex.down_scale = try b.dev(std.mem.sliceAsBytes(f[2].global));
    try weights.packFp4(packer, s, staged.ptr, ex);
    try s.synchronize();
    var l = try dense(b, sy);
    l.nvfp4 = ex;
    return l;
}

/// A model on `w` (placeholder layers past the ones set), with the NVFP4 kernels when a layer needs them.
fn model(gpu: check.Gpu, w: weights.Weights) !Model {
    var k = try Kernels.load(gpu.gpa, gpu.io, gpu.d, "");
    errdefer k.unload();
    if (w.nvfp4Layers() > 0) try k.loadFp4(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    return Model.assemble(gpu.gpa, gpu.d, k, w, .{ .eos = 0, .semantic_start = 1, .semantic_end = 4096 }, .{ .context = 2 * rows, .max_rows = rows });
}

/// Held-out fixtures, frozen with the envelope before their first run: own seeds, plain, cancelling, exact sums.
const Held = enum { plain, cancelling, exact_sums };
const held_seeds = [3]u64{ 0x4E1D_0A11, 0x4E1D_0B0B, 0x4E1D_0C0D };

/// Held-out weights: `cancelling` odd experts' down negates the even one's; `exact_sums` 2^k globals (few-bit sums).
fn heldWeights(a: std.mem.Allocator, rng: std.Random, which: Held) !ref.Synth {
    const sy = try ref.Synth.random(a, rng, which == .exact_sums);
    if (which == .cancelling) {
        const f = sy.experts.fp4;
        for (0..E / 2) |pair| {
            const even = 2 * pair;
            for (f[0..2]) |p| {
                const n = p.n * p.k / 2;
                @memcpy(p.codes[(even + 1) * n ..][0..n], p.codes[even * n ..][0..n]);
                @memcpy(p.scales[(even + 1) * n / 8 ..][0 .. n / 8], p.scales[even * n / 8 ..][0 .. n / 8]);
                p.global[even + 1] = p.global[even];
            }
            const d = f[2];
            const n = d.n * d.k / 2;
            for (d.codes[(even + 1) * n ..][0..n], d.codes[even * n ..][0..n]) |*odd, v| odd.* = v ^ 0x88; // both signs flipped
            @memcpy(d.scales[(even + 1) * n / 8 ..][0 .. n / 8], d.scales[even * n / 8 ..][0 .. n / 8]);
            d.global[even + 1] = d.global[even];
        }
    }
    return sy;
}

/// Held-out rows: `cancelling` odd router inputs a bf16 step off their even partner's; `exact_sums` ±{.5, .75, 1, 1.5}.
fn heldRows(a: std.mem.Allocator, rng: std.Random, which: Held) ![]u16 {
    const x = try ref.bfs(a, rng, rows * H, 1.0);
    for (0..rows) |r| switch (which) {
        .plain => {},
        .cancelling => for (0..E / 2) |pair| {
            x[r * H + 2 * pair + 1] = x[r * H + 2 * pair] + 1;
        },
        .exact_sums => for (x[r * H ..][0..H]) |*v| {
            const mag = [4]u16{ 0x3F00, 0x3F40, 0x3F80, 0x3FC0 };
            v.* = mag[rng.uintLessThan(usize, 4)] | (@as(u16, rng.uintLessThan(u16, 2)) << 15);
        },
    };
    return x;
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xD4A3);
    const rng = prng.random();
    const sy = try ref.Synth.random(a, rng, false);
    const f8 = try Fp8Layer.random(a, rng);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var b: Bufs = .{ .d = gpu.d, .gpa = gpu.gpa };
    defer b.deinit();
    var packer = try cuda.modelopt.Packer.load(gpu.d);
    defer packer.unload();

    var w: weights.Weights = .{ .gpa = gpu.gpa };
    w.layers[fp8_layer] = try dense(&b, sy);
    w.layers[fp8_layer].gu = try b.dev(f8.gu);
    w.layers[fp8_layer].gu_s = try b.dev(std.mem.sliceAsBytes(f8.gu_s));
    w.layers[fp8_layer].dn = try b.dev(f8.dn);
    w.layers[fp8_layer].dn_s = try b.dev(std.mem.sliceAsBytes(f8.dn_s));
    w.layers[nvfp4_layer] = try fp4Layer(gpu, &b, s, &packer, sy);
    const sparse = try exact.sparse(a, sy);
    w.layers[sparse_layer] = try fp4Layer(gpu, &b, s, &packer, sparse);
    var held: [3]ref.Synth = undefined;
    var held_x: [3][]u16 = undefined;
    for (&held, &held_x, held_seeds, 0..) |*hs, *hx, seed, i| {
        var hp = std.Random.DefaultPrng.init(seed);
        hs.* = try heldWeights(a, hp.random(), @enumFromInt(i));
        hx.* = try heldRows(a, hp.random(), @enumFromInt(i));
        w.layers[held_layer + i] = try fp4Layer(gpu, &b, s, &packer, hs.*);
    }
    var mixed = try model(gpu, w);
    defer mixed.deinit();
    var io = try ref.Pinned.init(gpu.d);
    defer io.deinit();

    const x = try ref.bfs(a, rng, rows * H, 1.0);
    try ref.scenario(gpu, a, &mixed, s, nvfp4_layer, sy, null, x, "NVFP4 spread routing", &io, true);
    // every row to experts 0..7 (128 pairs each, the subnormal-scaled expert 7 among them), slots in per-row orders
    const crowd = try a.dupe(u16, x);
    for (0..rows) |r| for (0..ref.heavy) |e| {
        crowd[r * H + e] = h.bf16Of(if (e < top) 0.5 + rng.float(f32) * 0.5 else -rng.float(f32));
    };
    try ref.scenario(gpu, a, &mixed, s, nvfp4_layer, sy, null, crowd, "NVFP4 all pairs to 8 experts", &io, true);
    try exact.epilogue(gpu, a, &mixed, s, sparse_layer, &io, rng);
    try exact.combine(gpu, a, &mixed, s, rng);

    // the FP8 layer beside, then the same layer in a model with no NVFP4 layer (no NVFP4 scratch or kernels)
    @memcpy(io.in.bytes, std.mem.sliceAsBytes(x));
    try ref.rowCounts(&mixed, s, fp8_layer, "FP8 layer beside NVFP4", io.in.bytes, io.outs);
    const beside = try a.dupe(u8, io.outs[0]);
    var plain_w: weights.Weights = .{ .gpa = gpu.gpa };
    plain_w.layers[fp8_layer] = w.layers[fp8_layer];
    var plain = try model(gpu, plain_w);
    defer plain.deinit();
    try check.expect(plain.k.fp4 == null, "an FP8-only model loads no NVFP4 kernels", .{});
    try ref.replay(&plain, s, fp8_layer, &.{rows}, io.in.bytes, io.outs[0]);
    try check.sameBytes("FP8 layer: beside an NVFP4 layer against an FP8-only model", beside, io.outs[0]);
    check.pass("FP8 layer: bit-equal beside an NVFP4 layer and in an FP8-only model", .{});

    // held out: every set runs and reports before the command fails on any of them
    var failed: usize = 0;
    for (held, held_x, 0..) |hs, hx, i| {
        var name: [64]u8 = undefined;
        const what = try std.fmt.bufPrint(&name, "held-out {t}", .{@as(Held, @enumFromInt(i))});
        ref.scenario(gpu, a, &mixed, s, held_layer + i, hs, null, hx, what, &io, true) catch |err| {
            std.debug.print("FAIL {s}: {t}\n", .{ what, err });
            failed += 1;
        };
    }
    try check.expect(failed == 0, "{d} of 3 held-out sets outside the envelope", .{failed});
}
