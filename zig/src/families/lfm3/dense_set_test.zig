//! Device checks: the bf16 release's projections and d1's set attention against float64 on the host.
const std = @import("std");
const mtl = @import("metal");
const kern = @import("kernels.zig");

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

fn value(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const opts = mtl.ResourceOptions.shared;

fn fill(rng: std.Random, out: []u16, scale: f32) void {
    for (out) |*v| v.* = bf16((rng.float(f32) * 2 - 1) * scale);
}

/// Within the output's half ulp of float64 plus fp32 sums' bound over `terms` products of magnitude `mag`.
fn near(got: f64, want: f64, terms: usize, mag: f64) bool {
    return @abs(got - want) <= @abs(want) * 0x1p-8 + @as(f64, @floatFromInt(terms)) * 0x1p-23 * mag + 1e-30;
}

test "LFM3 bf16 projections: register rows to 32, the tensor-unit GEMM past them, against float64" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var d = try kern.Dense.load(device);
    defer d.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const k = 512;
    for ([_]usize{ 1, 3, 32, 33, 100 }) |rows| for ([_]usize{ 16, 264 }) |n| {
        const x = try device.buffer(rows * k * 2, opts);
        defer x.deinit();
        const w = try device.buffer(n * k * 2, opts);
        defer w.deinit();
        const y = try device.buffer(rows * n * 2, opts);
        defer y.deinit();
        fill(rng, x.slice(u16, rows * k), 1);
        fill(rng, w.slice(u16, n * k), 0.1);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        if (rows <= 32) {
            const v = for (kern.Dense.most_rows, 0..) |most, i| {
                if (rows <= most) break i;
            } else unreachable;
            e.setPipeline(d.matvec[v]);
            e.setBuffer(x, 0, 0);
            e.setBuffer(w, 0, 1);
            e.setValue(extern struct { rows: u32, n: u32, k: u32, ldy: u32 }{ .rows = @intCast(rows), .n = @intCast(n), .k = k, .ldy = @intCast(n) }, 2);
            e.setBuffer(y, 0, 3);
            e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        } else kern.gemm(e, &d.gemm, x, 0, w, 0, y, 0, rows, n, k);
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const xs = x.slice(u16, rows * k);
        const ws = w.slice(u16, n * k);
        const ys = y.slice(u16, rows * n);
        for (0..rows) |r| for (0..n) |j| {
            var want: f64 = 0;
            var mag: f64 = 0;
            for (0..k) |i| {
                const p = value(xs[r * k + i]) * value(ws[j * k + i]);
                want += p;
                mag += @abs(p);
            }
            if (!near(value(ys[r * n + j]), want, k, mag)) {
                std.debug.print("rows {d} n {d}: [{d}][{d}] {d} want {d}\n", .{ rows, n, r, j, value(ys[r * n + j]), want });
                return error.TestUnexpectedResult;
            }
        };
    };
}

test "d1's set attention: the state sees everything, a question its state and itself, against float64" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ks = try kern.load(device);
    defer ks.deinit();
    const rows = 39;
    const state = 21;
    const cap = 64;
    var groups: [rows]i32 = undefined;
    for (&groups, 0..) |*g, r| g.* = if (r < state) -1 else if (r < 30) 1 else if (r < 33) 2 else 3;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const qb = try device.buffer(rows * 16 * 128 * 2, opts);
    defer qb.deinit();
    const kb = try device.buffer(4 * cap * 128 * 2, opts);
    defer kb.deinit();
    const vb = try device.buffer(4 * cap * 128 * 2, opts);
    defer vb.deinit();
    const gb = try device.buffer(rows * 4, opts);
    defer gb.deinit();
    const ob = try device.buffer(rows * 16 * 128 * 2, opts);
    defer ob.deinit();
    fill(rng, qb.slice(u16, rows * 16 * 128), 1);
    fill(rng, kb.slice(u16, 4 * cap * 128), 1);
    fill(rng, vb.slice(u16, 4 * cap * 128), 1);
    @memcpy(gb.slice(i32, rows), &groups);
    const scale: f32 = 1.0 / @sqrt(128.0);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(ks.get("lfm3_set_attention"));
    e.setBuffer(qb, 0, 0);
    e.setBuffer(kb, 0, 1);
    e.setBuffer(vb, 0, 2);
    e.setBuffer(gb, 0, 3);
    e.setValue(kern.SetArgs{ .rows = rows, .cap = cap, .state = state, .scale = scale }, 4);
    e.setBuffer(ob, 0, 5);
    e.dispatchGroups(mtl.Size.of((rows + 3) / 4, 4, 1), mtl.Size.of(512, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
    const q = qb.slice(u16, rows * 16 * 128);
    const kk = kb.slice(u16, 4 * cap * 128);
    const vv = vb.slice(u16, 4 * cap * 128);
    const o = ob.slice(u16, rows * 16 * 128);
    for (0..rows) |r| for (0..16) |h| {
        const kh = h / 4;
        var scores: [rows]f64 = undefined;
        var most: f64 = -std.math.inf(f64);
        for (0..rows) |j| {
            const seen = r < state or j < state or groups[j] == groups[r];
            var s: f64 = -std.math.inf(f64);
            if (seen) {
                s = 0;
                for (0..128) |i| s += value(q[(r * 16 + h) * 128 + i]) * value(kk[(kh * cap + j) * 128 + i]);
                s *= scale;
            }
            scores[j] = s;
            most = @max(most, s);
        }
        var total: f64 = 0;
        for (&scores) |*s| {
            s.* = @exp(s.* - most);
            total += s.*;
        }
        for (0..128) |i| {
            var want: f64 = 0;
            for (0..rows) |j| want += scores[j] / total * value(vv[(kh * cap + j) * 128 + i]);
            const got = value(o[(r * 16 + h) * 128 + i]);
            if (@abs(got - want) > @abs(want) * 0x1p-8 + 1e-4) {
                std.debug.print("row {d} head {d} dim {d}: {d} want {d}\n", .{ r, h, i, got, want });
                return error.TestUnexpectedResult;
            }
        }
    };
}

test "LFM3-VL's tiled tower attention equals its row loop bit for bit" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var ks = try @import("vision_kernels.zig").Kernels.load(std.testing.allocator, device);
    defer ks.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    for ([_]usize{ 4, 37, 300 }) |patches| {
        const q = try device.buffer(patches * 16 * 64 * 2, opts);
        defer q.deinit();
        const k = try device.buffer(patches * 8 * 64 * 2, opts);
        defer k.deinit();
        const v = try device.buffer(patches * 8 * 64 * 2, opts);
        defer v.deinit();
        const a = try device.buffer(patches * 16 * 64 * 2, opts);
        defer a.deinit();
        const b = try device.buffer(patches * 16 * 64 * 2, opts);
        defer b.deinit();
        fill(rng, q.slice(u16, patches * 16 * 64), 2);
        fill(rng, k.slice(u16, patches * 8 * 64), 2);
        fill(rng, v.slice(u16, patches * 8 * 64), 1);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for ([_][]const u8{ "l3v_attention", "l3v_attention_tiled" }, [_]mtl.Buffer{ a, b }) |name, out| {
            e.setPipeline(if (name.len == 13) ks.get("l3v_attention") else ks.get("l3v_attention_tiled"));
            e.setBuffer(q, 0, 0);
            e.setBuffer(k, 0, 1);
            e.setBuffer(v, 0, 2);
            e.setBuffer(out, 0, 3);
            e.setValue(@as(u32, @intCast(patches)), 4);
            if (name.len == 13) e.dispatchGroups(mtl.Size.of(16, patches, 1), mtl.Size.of(32, 1, 1)) else e.dispatchGroups(mtl.Size.of((patches + 7) / 8, 8, 1), mtl.Size.of(512, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        try std.testing.expectEqualSlices(u16, a.slice(u16, patches * 16 * 64), b.slice(u16, patches * 16 * 64));
    }
}
