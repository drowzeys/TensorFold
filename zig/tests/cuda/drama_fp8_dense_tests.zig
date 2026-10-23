//! drama-fp8-dense: dense FP8 groups on the lane against float64 and BF16, splits, mixed formats, a checkpoint's layer.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");
const ref = @import("drama_fp8_dense_ref.zig");
const mix = @import("drama_fp8_dense_mix.zig");

const weights = drama.backbone.weights;
const Kernels = drama.backbone.kernels.Kernels;
const qmmf = cuda.qmmf;
const qlinear = cuda.qlinear;
const H = ref.H;

/// The server's prompt pass, and a count past qmmf.fused_rows: capacities above 128 take the fused form from 256 rows.
const rows = 128;
const wide = 300;

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xF8DE);
    var r: ref.Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const free = (try gpu.ctx.memInfo()).free;
    var lane = try qmmf.Lane.load(gpu.d, try gpu.ctx.capability() / 10);
    defer lane.unload();
    std.debug.print("RESULT the backbone's lane module took {d} bytes of device memory (free before minus after)\n", .{free -| (try gpu.ctx.memInfo()).free});
    var reduce = lane; // the slice reduce of GPUs before sm_90, which Model.init sizes `part` for
    reduce.clusters = false;
    var k = try Kernels.load(gpu.gpa, gpu.io, gpu.d, "");
    defer k.unload();
    for (ref.shapes) |shape| try group(&r, &k, &lane, &reduce, shape[0], shape[1], shape[2]);
    try splits(&r, &k);
    try mix.run(gpu, &r);
    check.pass("drama-fp8-dense: every group within its float64 tolerance, no worse than BF16, bits at every row count", .{});
}

/// The BF16 path's call sizes (b) compares at: 1 row (its one-row gemv), 7 (the r1 tiling) and 128 rows, then 300.
const chunks = [_]usize{ 1, 7, rows };

/// (a), (b), (c) for one group shape on synthetic weights.
fn group(r: *ref.Rig, k: *const Kernels, lane: *const qmmf.Lane, reduce: *const qmmf.Lane, name: []const u8, n: usize, kd: usize) !void {
    const a = r.a;
    const g = try ref.synthetic(a, r.rng, n, kd);
    const x = try r.bfs(wide * kd, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const want = try ref.reference(a, try g.dequant(a), n, kd, x, wide);
    const gated = try a.alloc(bool, n);
    for (gated, 0..) |*o, j| o.* = !ref.subnormalRow(j); // subnormal-scale outputs: finite and within tolerance only
    const w = try g.upload(r);
    const part = try r.zeros(reduce.partBytes(qmmf.fused_rows - 1, w.fp8g));
    var name_buf: [128]u8 = undefined;
    var wide_buf: [160]u8 = undefined;
    // (b)'s baselines: the BF16 path on the nearest-bf16 weights, at each call size, against the same reference
    const wb = try r.dev(std.mem.sliceAsBytes(try g.bf16(a)));
    var bf: [chunks.len + 1]ref.Stats = undefined;
    for (chunks ++ [_]usize{wide}, &bf) |c, *st| {
        const m: usize = if (c == wide) wide else rows;
        const yb = try r.zeros(m * n * 2);
        try ref.gemmChunked(k, r.s, dx, kd, m, c, wb, n, yb);
        st.* = try ref.stats(a, try r.back(yb, m * n * 2), want, n, gated);
        try ref.measured(try std.fmt.bufPrint(&name_buf, "{s} [{d}, {d}] bf16 gemm in calls of {d} rows", .{ name, n, kd, c }), n, st.*);
    }
    var at_rows: []u8 = &.{};
    const forms = [_]struct { []const u8, *const qmmf.Lane }{ .{ "cluster", lane }, .{ "reduce", reduce } };
    for (forms) |f| {
        if (!f[1].clusters and std.mem.eql(u8, f[0], "cluster")) continue; // no clusters before sm_90: one form
        const lin: qlinear.Linear = .{ .lane = f[1] };
        const what = try std.fmt.bufPrint(&name_buf, "{s} [{d}, {d}] {s}", .{ name, n, kd, f[0] });
        const got = try ref.rowCounts(r, lin, w, dx, rows, part, what); // the 1-, 7- and 128-row calls' common bytes
        const lane_rows = try ref.stats(a, got, want, n, gated);
        try ref.gate(what, n, lane_rows);
        for (chunks, bf[0..chunks.len]) |c, b| try ref.noWorse(try std.fmt.bufPrint(&wide_buf, "{s} in calls of {d} rows", .{ what, c }), lane_rows, b);
        const y = try r.zeros(wide * n * 2);
        try lin.decode(r.s, w, .{ .x = dx, .part = part }, y, wide);
        const fused = try r.back(y, wide * n * 2);
        try check.sameBytes(try std.fmt.bufPrint(&wide_buf, "{s}: the {d}-row call's first {d} rows", .{ what, wide, rows }), fused[0 .. rows * n * 2], got);
        const lane_wide = try ref.stats(a, fused, want, n, gated);
        try ref.noWorse(try std.fmt.bufPrint(&wide_buf, "{s} in one call of {d} rows", .{ what, wide }), lane_wide, bf[chunks.len]);
        try ref.gate(try std.fmt.bufPrint(&name_buf, "{s} [{d}, {d}] {s}, {d} rows", .{ name, n, kd, f[0], wide }), n, lane_wide);
        if (at_rows.len == 0) at_rows = got else std.debug.print("RESULT {s} [{d}, {d}]: cluster and reduce forms {s}\n", .{ name, n, kd, if (std.mem.eql(u8, at_rows, got)) "bit-equal" else "differ" });
    }
}

/// (d): the BF16 router and b | a rows an FP8 group splits off give the fused BF16 product's columns, bit for bit.
fn splits(r: *ref.Rig, k: *const Kernels) !void {
    const cases = [_]struct { []const u8, usize, usize, usize }{
        .{ "router", weights.moe_in_rows, weights.router_col, drama.backbone.config.experts },
        .{ "b|a", weights.gdn_in_rows, weights.gdn_b_col, 2 * drama.backbone.config.linear_value_heads },
    };
    const x = try r.bfs(wide * H, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    var name: [96]u8 = undefined;
    for (cases) |c| {
        const what, const fused_n, const col, const n = c;
        const w = try r.bfs(fused_n * H, 0.05);
        const fused_w = try r.dev(std.mem.sliceAsBytes(w));
        const own_w = try r.dev(std.mem.sliceAsBytes(w[col * H ..][0 .. n * H])); // its own allocation, as the loader's
        for ([_]usize{ 1, 7, rows, wide }) |m| {
            const yf = try r.zeros(m * fused_n * 2);
            try k.gemm(r.s, dx, H, m, fused_w, fused_n, H, yf, fused_n);
            const ys = try r.zeros(m * n * 2);
            try k.gemm(r.s, dx, H, m, own_w, n, H, ys, n);
            const f = try r.back(yf, m * fused_n * 2);
            const own = try r.back(ys, m * n * 2);
            for (0..m) |row| try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: {d} rows, row {d}", .{ what, m, row }), own[row * n * 2 ..][0 .. n * 2], f[(row * fused_n + col) * 2 ..][0 .. n * 2]);
        }
        check.pass("{s} [{d}, {d}] equals columns {d}..{d} of the fused [{d}, {d}] gemm at 1, 7, {d}, {d} rows", .{ what, n, H, col, col + n - 1, fused_n, H, rows, wide });
    }
}

/// (f) `drama-fp8-dense CHECKPOINT LAYER [SOURCE]`: the layer's groups as the production loader stored them.
pub const file = @import("drama_fp8_dense_file.zig").run;
