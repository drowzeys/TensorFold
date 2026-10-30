//! EmbeddingGemma-2's Metal kernels on this GPU: batch invariance, both projection paths, epilogues, a planted control.
const std = @import("std");
const mtl = @import("metal");
const em = @import("embgemma2_model.zig");
const t = std.testing;

const Rig = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    p: em.Pipes,
    bufs: [40]mtl.Buffer = undefined,
    n: usize = 0,

    fn init(head: []const u8) !?Rig {
        const device = mtl.Device.init() catch return null;
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        const lib = try em.library(t.allocator, device, head);
        errdefer lib.deinit();
        return .{ .device = device, .queue = queue, .lib = lib, .p = try em.Pipes.load(device, lib) };
    }

    fn deinit(r: *Rig) void {
        for (r.bufs[0..r.n]) |b| b.deinit();
        r.p.release(em.pipe_count);
        r.lib.deinit();
        r.queue.deinit();
        r.device.deinit();
    }

    fn buf(r: *Rig, bytes: usize) !mtl.Buffer {
        if (r.n == r.bufs.len) return error.TooManyBuffers;
        const b = try r.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(b.contents()[0..bytes], 0);
        r.bufs[r.n] = b;
        r.n += 1;
        return b;
    }

    fn run(r: *Rig, ctx: anytype, comptime f: fn (@TypeOf(ctx), mtl.ComputeEncoder) void) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        f(ctx, e);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailure;
    }
};

fn bf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
fn fl(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
fn fill(b: mtl.Buffer, n: usize, seed: u64, scale: f32) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const p: [*]u16 = @ptrCast(@alignCast(b.contents()));
    for (0..n) |i| p[i] = bf((rng.random().float(f32) * 2 - 1) * scale);
}
fn halves(b: mtl.Buffer, off: usize, n: usize) []u16 {
    const p: [*]u16 = @ptrCast(@alignCast(b.contents() + off));
    return p[0..n];
}
/// Whether a and b agree to `ulps` units of bf16's last place (relative to the larger, with a floor at 2^-12).
fn close(a: []const u16, b: []const u16, ulps: f32) bool {
    for (a, b) |x, y| {
        const u = fl(x);
        const v = fl(y);
        if (!std.math.isFinite(u) or !std.math.isFinite(v)) return false;
        if (@abs(u - v) > ulps * @max(@max(@abs(u), @abs(v)), 1.0 / 4096.0) / 128) return false;
    }
    return true;
}

/// The largest difference between a and b is at most `tol` (TF_EG2_TEST_VERBOSE prints it).
fn near(a: []const u16, b: []const u16, tol: f32) bool {
    var worst: f32 = 0;
    for (a, b) |x, y| worst = @max(worst, @abs(fl(x) - fl(y)));
    if (std.c.getenv("TF_EG2_TEST_VERBOSE") != null) std.debug.print("largest difference {d}\n", .{worst});
    return std.math.isFinite(worst) and worst <= tol;
}

const Proj = struct { r: *Rig, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, pli: mtl.Buffer, m: usize, n: usize, k: usize, epi: em.Epilogue, rows: usize };

fn gemm(c: Proj, e: mtl.ComputeEncoder) void {
    const ldy = if (c.epi == .glu) c.n / 2 else c.n;
    for ([_]mtl.Buffer{ c.x, c.w }, 0..) |b, i| e.setBuffer(b, 0, i);
    e.setBuffer(c.y, 0, 3);
    e.setBuffer(c.pli, 0, 4);
    if (c.rows > 0) {
        const at = std.mem.indexOfScalar(usize, &em.matvec_widths, c.rows).?;
        e.setPipeline(c.r.p.matvec[@backingInt(c.epi)][at]);
        e.setValue(extern struct { rows: u32, n: u32, k: u32, ldp: u32, poff: u32 }{ .rows = @intCast(c.m), .n = @intCast(ldy), .k = @intCast(c.k), .ldp = @intCast(c.n), .poff = 0 }, 2);
        e.dispatchGroups(mtl.Size.of((ldy + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        return;
    }
    const tn = c.n / 128;
    const tm = (c.m + 63) / 64;
    e.setPipeline(c.r.p.gemm[@backingInt(c.epi)][@intFromBool(c.m % 64 == 0)]);
    var p: [16]i32 = @splat(0);
    for ([_]usize{ c.m, c.n, c.k, c.k, c.k, ldy, tn, tm, 2, c.k / 256, c.n, 0 }, 0..) |v, i| p[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&p), 2);
    e.dispatchThreads(mtl.Size.of((tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}

test "embedding_gemma2 metal: projections, gemm rows batch-invariant, few-row path close, epilogues as glue" {
    var r = (try Rig.init("")) orelse return error.SkipZigTest;
    defer r.deinit();
    const K = 512;
    const N = 256;
    const x = try r.buf(130 * K * 2 + 64 * K * 2);
    const w = try r.buf(N * K * 2);
    const pli = try r.buf(130 * N * 2);
    fill(x, 130 * K, 1, 2);
    fill(w, N * K, 2, 0.05);
    fill(pli, 130 * N, 3, 1);
    const y70 = try r.buf(130 * N * 2);
    const y130 = try r.buf(130 * N * 2);
    const y4 = try r.buf(64 * N * 2);
    const y32 = try r.buf(64 * N * 2);
    inline for (.{ em.Epilogue.plain, em.Epilogue.glu, em.Epilogue.ple }) |epi| {
        const ldy = if (epi == .glu) N / 2 else N;
        const base: Proj = .{ .r = &r, .x = x, .w = w, .y = y70, .pli = pli, .m = 70, .n = N, .k = K, .epi = epi, .rows = 0 };
        try r.run(base, gemm);
        var c = base;
        c.y = y130;
        c.m = 130;
        try r.run(c, gemm);
        // a row's bits do not depend on the rest of the batch
        try t.expectEqualSlices(u16, halves(y70, 0, 70 * ldy), halves(y130, 0, 70 * ldy));
        c = base;
        c.y = y4;
        c.m = 3;
        c.rows = 4;
        try r.run(c, gemm);
        c.y = y32;
        c.rows = 32;
        try r.run(c, gemm);
        // the few-row path: the same bits at every width, within 2 bf16 places of the tensor-unit GEMM
        try t.expectEqualSlices(u16, halves(y4, 0, 3 * ldy), halves(y32, 0, 3 * ldy));
        try t.expect(close(halves(y32, 0, 3 * ldy), halves(y70, 0, 3 * ldy), 2));
    }
    // the GLU epilogue against the plain product and the glue's formula on the host (gate, up interleaved by 16)
    const plain: Proj = .{ .r = &r, .x = x, .w = w, .y = y70, .pli = pli, .m = 70, .n = N, .k = K, .epi = .plain, .rows = 0 };
    try r.run(plain, gemm);
    var glu = plain;
    glu.y = y130;
    glu.epi = .glu;
    try r.run(glu, gemm);
    const yp = halves(y70, 0, 70 * N);
    const yg = halves(y130, 0, 70 * N / 2);
    var want: [N / 2]u16 = undefined;
    for (0..70) |row| {
        for (0..N / 2) |j| {
            const col = 32 * (j / 16) + j % 16;
            const gt = fl(yp[row * N + col]);
            const up = fl(yp[row * N + col + 16]);
            const ge = 0.5 * gt * (1 + std.math.tanh(0.7978845608028654 * (gt + 0.044715 * gt * gt * gt)));
            want[j] = bf(fl(bf(ge)) * up);
        }
        try t.expect(close(yg[row * N / 2 ..][0 .. N / 2], &want, 1));
    }
}

const Attn = struct { r: *Rig, qkv: mtl.Buffer, out: mtl.Buffer, ints: mtl.Buffer, rope: mtl.Buffer, w: mtl.Buffer, rows: usize, hd: usize, hkv: usize, tiles: usize, kind: em.Attention };

/// ints: lo [0, 64), hi [64, 128), tiles [128, 128 + 4 * 64), pos [384, 448) as i32.
fn attend(c: Attn, e: mtl.ComputeEncoder) void {
    const p = &c.r.p;
    if (c.kind == .row) { // q, k, v norms and RoPE first, then the plain kernel (.row); .t256 the plain kernel alone
        e.setPipeline(p.qkv);
        for ([_]struct { mtl.Buffer, usize }{ .{ c.qkv, 0 }, .{ c.w, 0 }, .{ c.w, 1024 }, .{ c.ints, 384 * 4 } }, 0..) |b, i| e.setBuffer(b[0], b[1], i);
        e.setValue(extern struct { hq: u32, hkv: u32, hd: u32, positions: u32, eps: f32 }{ .hq = 4, .hkv = @intCast(c.hkv), .hd = @intCast(c.hd), .positions = 64, .eps = 1e-6 }, 4);
        e.setBuffer(c.rope, 0, 5);
        e.dispatchGroups(mtl.Size.of(c.rows, 4 + 2 * c.hkv, 1), mtl.Size.of(128, 1, 1));
    }
    for ([_]struct { mtl.Buffer, usize }{ .{ c.qkv, 0 }, .{ c.out, 0 }, .{ c.ints, 0 }, .{ c.ints, 64 * 4 } }, 0..) |b, i| e.setBuffer(b[0], b[1], i);
    switch (c.kind) {
        .short => {
            e.setPipeline(if (c.hd == 512) p.attn512_short else p.attn256_short);
            e.setValue(extern struct { rows: u32, hq: u32, hkv: u32, positions: u32, eps: f32 }{ .rows = @intCast(c.rows), .hq = 4, .hkv = @intCast(c.hkv), .positions = 64, .eps = 1e-6 }, 4);
            for ([_]struct { mtl.Buffer, usize }{ .{ c.ints, 384 * 4 }, .{ c.w, 0 }, .{ c.w, 1024 }, .{ c.rope, 0 } }, 5..) |b, i| e.setBuffer(b[0], b[1], i);
            e.dispatchGroups(mtl.Size.of((c.rows * 4 + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
        },
        .tiled => {
            e.setPipeline(if (c.hd == 512) p.attn512_tiled else p.attn256_tiled);
            e.setValue(extern struct { rows: u32, hq: u32, hkv: u32 }{ .rows = @intCast(c.rows), .hq = 4, .hkv = @intCast(c.hkv) }, 4);
            e.setBuffer(c.ints, 128 * 4, 5);
            e.setThreadgroupMemory(em.attnTgBytes(c.hd), 0);
            e.dispatchGroups(mtl.Size.of(c.tiles, 4, 1), mtl.Size.of(32 * c.hd / 64, 1, 1));
        },
        else => {
            e.setPipeline(if (c.hd == 512) p.attn512 else p.attn256);
            e.setValue(extern struct { rows: u32, hq: u32, hkv: u32 }{ .rows = @intCast(c.rows), .hq = 4, .hkv = @intCast(c.hkv) }, 4);
            e.dispatchGroups(mtl.Size.of((c.rows * 4 + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
        },
    }
}

fn ropeTable(c: Attn, e: mtl.ComputeEncoder) void {
    e.setPipeline(c.r.p.rope_table);
    e.setBuffer(c.rope, 0, 0);
    e.setValue(extern struct { hd: u32, theta: f32, positions: u32 }{ .hd = @intCast(c.hd), .theta = 1e4, .positions = 64 }, 1);
    e.dispatchThreads(mtl.Size.of(c.hd / 2, 64, 1), mtl.Size.of(128, 1, 1));
}

/// Inputs ending at `ends` packed from row 0: their key ranges, tiles and positions in `ints`; the tile count.
fn pack(ints: mtl.Buffer, ends: []const u32) usize {
    const p: [*]i32 = @ptrCast(@alignCast(ints.contents()));
    var at: u32 = 0;
    for (ends) |end| {
        for (at..end) |row| {
            p[row] = @intCast(at);
            p[64 + row] = @intCast(end);
            p[384 + row] = @intCast(row - at);
        }
        at = end;
    }
    return em.tileRows(ends, p[128..384]);
}

test "embedding_gemma2 metal: attention alone and in a batch, tiled against plain, the fused short kernel" {
    var r = (try Rig.init("")) orelse return error.SkipZigTest;
    defer r.deinit();
    for ([_][2]usize{ .{ 256, 2 }, .{ 512, 1 } }) |geo| {
        const hd = geo[0];
        const S = (4 + 2 * geo[1]) * hd;
        const rows = 7 + 40;
        const raw = try r.buf((rows + 300) * S * 2);
        fill(raw, rows * S, hd, 1);
        const alone = try r.buf((40 + 300) * S * 2);
        @memcpy(halves(alone, 0, 40 * S), halves(raw, 7 * S * 2, 40 * S));
        const out_b = try r.buf(rows * 4 * hd * 2);
        const out_a = try r.buf(40 * 4 * hd * 2);
        const ib = try r.buf(448 * 4);
        const ia = try r.buf(448 * 4);
        const rope = try r.buf(2 * 64 * hd / 2 * 2);
        const w = try r.buf(2048);
        fill(w, 1024, 5, 1);
        var c: Attn = .{ .r = &r, .qkv = raw, .out = out_b, .ints = ib, .rope = rope, .w = w, .rows = rows, .hd = hd, .hkv = geo[1], .tiles = pack(ib, &.{ 7, 47 }), .kind = .tiled };
        try r.run(c, ropeTable);
        try r.run(c, attend);
        var a = c;
        a.qkv = alone;
        a.out = out_a;
        a.ints = ia;
        a.rows = 40;
        a.tiles = pack(ia, &.{40});
        try r.run(a, attend);
        // the second input's rows, alone and after another input: the same bits
        try t.expectEqualSlices(u16, halves(out_a, 0, 40 * 4 * hd), halves(out_b, 7 * 4 * hd * 2, 40 * 4 * hd));
        // tiled (tensor units, fp32 probabilities) against the plain kernel, values in [-1, 1]: within 2^-7
        const plain = try r.buf(rows * 4 * hd * 2);
        c.out = plain;
        c.kind = .t256; // the plain kernel on normed rows: attend's default branch
        try r.run(c, attend);
        try t.expect(near(halves(plain, 0, rows * 4 * hd), halves(out_b, 0, rows * 4 * hd), 1.0 / 128.0));
        // the fused short kernel on raw rows (inputs of 5 and 9) against norms, RoPE, then the plain kernel
        const short_raw = try r.buf((14 + 300) * S * 2);
        fill(short_raw, 14 * S, hd + 1, 3);
        const normed = try r.buf((14 + 300) * S * 2);
        @memcpy(halves(normed, 0, 14 * S), halves(short_raw, 0, 14 * S));
        const is = try r.buf(448 * 4);
        const o1 = try r.buf(14 * 4 * hd * 2);
        const o2 = try r.buf(14 * 4 * hd * 2);
        var s: Attn = .{ .r = &r, .qkv = short_raw, .out = o1, .ints = is, .rope = rope, .w = w, .rows = 14, .hd = hd, .hkv = geo[1], .tiles = pack(is, &.{ 5, 14 }), .kind = .short };
        try r.run(s, attend);
        s.qkv = normed;
        s.out = o2;
        s.kind = .row;
        try r.run(s, attend);
        try t.expect(near(halves(o1, 0, 14 * 4 * hd), halves(o2, 0, 14 * 4 * hd), 1.0 / 32.0));
    }
}

const Norm = struct { r: *Rig, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, rows: usize };

fn norm(c: Norm, e: mtl.ComputeEncoder) void {
    e.setPipeline(c.r.p.rms);
    for ([_]mtl.Buffer{ c.x, c.w, c.y }, 0..) |b, i| e.setBuffer(b, 0, i);
    e.setValue(extern struct { n: u32, eps: f32, pre: f32, scaled: u32 }{ .n = 512, .eps = 1e-6, .pre = 1, .scaled = 1 }, 3);
    e.dispatchGroups(mtl.Size.of(c.rows, 1, 1), mtl.Size.of(128, 1, 1));
}

/// RMSNorm of rows of 512 values up to 160 (past half's range in a thread's sum) on `head`'s build, against fp64.
fn normMatches(head: []const u8) !?bool {
    var r = (try Rig.init(head)) orelse return null;
    defer r.deinit();
    const x = try r.buf(8 * 512 * 2);
    const w = try r.buf(512 * 2);
    const y = try r.buf(8 * 512 * 2);
    fill(x, 8 * 512, 9, 160);
    fill(w, 512, 10, 1);
    try r.run(Norm{ .r = &r, .x = x, .w = w, .y = y, .rows = 8 }, norm);
    var want: [512]u16 = undefined;
    for (0..8) |row| {
        const xs = halves(x, row * 1024, 512);
        var s: f64 = 0;
        for (xs) |v| s += @as(f64, fl(v)) * fl(v);
        const inv = 1 / @sqrt(s / 512 + 1e-6);
        for (xs, halves(w, 0, 512), &want) |v, k, *o| o.* = bf(@floatCast(fl(v) * inv * fl(k)));
        if (!close(halves(y, row * 1024, 512), &want, 1)) return false;
    }
    return true;
}

test "embedding_gemma2 metal: RMSNorm against fp64, and the planted half-accumulator control goes red" {
    try t.expect((try normMatches("")) orelse return error.SkipZigTest);
    try t.expect(!((try normMatches("#define EG2_ACC half\n")).?));
}
