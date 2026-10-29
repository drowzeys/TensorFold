//! Affine-4 prompt rows on qmm_midrow's tiles: every tile and each pick byte-equal to qmm_prefill's 128-row tile.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const qlinear = cuda.qlinear;
const mid = qlinear.affine4_mid;
const Rig = quant_ref.RigT;

/// Qwen3.8-27B's prompt projections, a partial 64-column tile and a short K; rows across the served prompt sizes.
const Shape = struct { n: usize, k: usize, rows: []const usize };
const wide_rows = &[_]usize{ 17, 100, 160, 300 };
const all_rows = &[_]usize{ 17, 48, 100, 160, 192, 224, 257, 300, 513 };
const shapes = [_]Shape{
    .{ .n = 10240, .k = 5120, .rows = wide_rows }, .{ .n = 6144, .k = 5120, .rows = wide_rows },
    .{ .n = 48, .k = 5120, .rows = all_rows },     .{ .n = 5120, .k = 6144, .rows = wide_rows },
    .{ .n = 12288, .k = 5120, .rows = wide_rows }, .{ .n = 1024, .k = 5120, .rows = all_rows },
    .{ .n = 17408, .k = 5120, .rows = wide_rows }, .{ .n = 5120, .k = 17408, .rows = wide_rows },
    .{ .n = 200, .k = 512, .rows = all_rows },     .{ .n = 2600, .k = 1088, .rows = all_rows },
};

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(31);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const kk = cuda.kernels;
    const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack, kk.qmm_midrow };
    var mods: [images.len]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*md| md.unload();
    for (images, &mods) |img, *md| {
        md.* = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var a4 = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, false);
    defer a4.deinit();
    const tiles = try qlinear.MidRows.resolve(mods[4], sms);
    var with_mid = a4;
    with_mid.mid = tiles;
    var name: [160]u8 = undefined;
    var launches: usize = 0;
    for (shapes) |sh| {
        const mark = r.bufs.items.len;
        defer {
            for (r.bufs.items[mark..]) |*b| b.free();
            r.bufs.shrinkRetainingCapacity(mark);
        }
        const most = std.mem.max(usize, sh.rows);
        const x = try r.bfs(most * sh.k, 1.0);
        // a NaN, +inf and -inf in the last row: their columns' classes must come out as the 128-row tile's
        x[(most - 1) * sh.k + 3] = 0x7fc0;
        x[(most - 1) * sh.k + 70] = 0x7f80;
        x[(most - 1) * sh.k + 140] = 0xff80;
        const dx = try r.dev(std.mem.sliceAsBytes(x));
        const words = try r.a.alloc(u32, sh.n * sh.k / 8);
        for (words) |*w| w.* = r.rng.int(u32);
        const scales = try r.a.alloc(u16, sh.n * sh.k / 64);
        const biases = try r.a.alloc(u16, sh.n * sh.k / 64);
        for (scales, biases) |*sv, *bv| {
            sv.* = quant_ref.helpers.bf16Of((r.rng.float(f32) * 0.9 + 0.1) * 0.0025);
            bv.* = quant_ref.helpers.bf16Of(-(r.rng.float(f32) * 0.4 + 0.8) * 0.02);
        }
        const lay = qlinear.Affine4.layout(sh.n, sh.k);
        const q: qlinear.Affine4 = .{ .w = try r.zeros(lay.words), .s = try r.zeros(lay.scales), .b = try r.zeros(lay.scales), .n = sh.n, .k = sh.k, .npad = lay.npad };
        try a4.pack(r.s, try r.dev(std.mem.sliceAsBytes(words)), try r.dev(std.mem.sliceAsBytes(scales)), try r.dev(std.mem.sliceAsBytes(biases)), q);
        const o = try r.zeros(most * sh.n * 2);
        const buf: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = o, .len = most * sh.n * 2 };
        for (sh.rows) |rows| {
            const bytes = rows * sh.n * 2;
            try buf.fill8(0xff, r.s.handle);
            try a4.prompt(r.s, dx, q, o, rows);
            const want = try r.back(o, bytes);
            for (0..mid.tiles.len) |i| {
                try buf.fill8(0xff, r.s.handle);
                try tiles.launchTile(i, r.s, dx, q, o, rows);
                const t = mid.tiles[i];
                try check.sameBytes(try std.fmt.bufPrint(&name, "mid tile {d}x{d} [{d}, {d}] at {d} rows equals the 128-row tile", .{ t.rows, t.cols, sh.n, sh.k, rows }), try r.back(o, bytes), want);
                launches += 1;
            }
            try buf.fill8(0xff, r.s.handle);
            try with_mid.prompt(r.s, dx, q, o, rows);
            const t = if (mid.pick(rows, sh.n, sms)) |i| mid.tiles[i] else mid.base;
            try check.sameBytes(try std.fmt.bufPrint(&name, "prompt() on its {d}x{d} pick [{d}, {d}] at {d} rows equals the 128-row tile", .{ t.rows, t.cols, sh.n, sh.k, rows }), try r.back(o, bytes), want);
        }
    }
    check.pass("affine4-mid: {d} shapes, {d} tile launches and every pick byte-equal to qmm_prefill's 128-row tile (NaN and infinities included)", .{ shapes.len, launches });
}
