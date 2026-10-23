//! Drama's MoE in float64 from a layer's source tensors, and the harness both NVFP4 commands run the model through.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const drama = @import("drama");
const check = @import("check.zig");
const h = @import("quant_ref_tests.zig").helpers;

const backbone = drama.backbone;
const config = backbone.config;
const weights = backbone.weights;
const Model = backbone.Model;

pub const H = config.hidden;
pub const I = config.expert_inner;
pub const E = config.experts;
pub const top = config.top_k;
pub const rows = 128; // drama_host's prompt pass
pub const heavy = 16; // synthetic router: experts whose logit is 16 x (not 1 x) an input value take many pairs each

/// One NVFP4 projection as a checkpoint stores it: codes [E][n][k / 2], e4m3 scales [E][n][k / 16], globals [E].
pub const Fp4 = struct {
    n: usize,
    k: usize,
    codes: []u8,
    scales: []u8,
    global: []f32,

    pub fn alloc(a: std.mem.Allocator, n: usize, k: usize) !Fp4 {
        return .{ .n = n, .k = k, .codes = try a.alloc(u8, E * n * k / 2), .scales = try a.alloc(u8, E * n * k / 16), .global = try a.alloc(f32, E) };
    }

    /// Random signed nibbles; each 8th expert's scales subnormal (128 x global), else 1 in 32 zero; exact: 2^k globals
    pub fn random(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize, exact: bool) !Fp4 {
        const f = try alloc(a, n, k);
        rng.bytes(f.codes);
        const blocks = n * k / 16;
        for (0..E) |e| {
            const sub = e % 8 == 7;
            for (f.scales[e * blocks ..][0..blocks]) |*v| v.* = if (sub) 1 + rng.uintLessThan(u8, 7) else if (rng.uintLessThan(u8, 32) == 0) 0 else 0x20 + rng.uintLessThan(u8, 0x28);
            const g: f32 = if (exact) 0x1p-8 else 0.002 + rng.float(f32) * 0.004;
            f.global[e] = g * @as(f32, if (sub) 128 else 1);
        }
        return f;
    }

    fn dequant(f: Fp4, e: usize, out: []f64) void {
        const g: f64 = f.global[e];
        for (0..f.n) |r| for (0..f.k) |c| {
            const byte = f.codes[(e * f.n + r) * (f.k / 2) + c / 2];
            const nib: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
            out[r * f.k + c] = h.e2m1Of(nib) * h.e4m3Of(f.scales[(e * f.n + r) * (f.k / 16) + c / 16]) * g;
        };
    }
};

/// One block-FP8 projection as a checkpoint stores it: e4m3 [E][n][k], fp32 scale_inv [E][n / 128][k / 128].
pub const Fp8 = struct {
    n: usize,
    k: usize,
    codes: []u8,
    inv: []f32,

    fn dequant(f: Fp8, e: usize, out: []f64) void {
        const kb = f.k / 128;
        for (0..f.n) |r| for (0..f.k) |c| {
            out[r * f.k + c] = h.e4m3Of(f.codes[(e * f.n + r) * f.k + c]) * f.inv[(e * (f.n / 128) + r / 128) * kb + c / 128];
        };
    }
};

/// A layer's routed experts in either stored format: gate, up, down.
pub const Experts = union(enum) {
    fp4: [3]Fp4,
    fp8: [3]Fp8,

    /// Expert e's projection j as float64 [n][k].
    fn dequant(x: Experts, j: usize, e: usize, out: []f64) void {
        switch (x) {
            inline else => |p| p[j].dequant(e, out),
        }
    }
};

const projections = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
const shapes = [3][2]usize{ .{ I, H }, .{ I, H }, .{ H, I } };

/// One MoE layer's source tensors: bf16 dense weights and the routed experts.
pub const Synth = struct {
    moe_in: []u16, // [moe_in_rows][H]: shared gate | shared up | router
    shared_down: []u16, // [H][I]
    shared_gate: []u16, // [H]
    experts: Experts,

    /// Synthetic: one-hot router rows (logit e = bf16(x[e] * 16 or 1), exact) and random NVFP4 experts.
    pub fn random(a: std.mem.Allocator, rng: std.Random, exact: bool) !Synth {
        var s: Synth = undefined;
        s.moe_in = try bfs(a, rng, weights.moe_in_rows * H, 0.066); // shared gate and up sums near unit variance
        const router = s.moe_in[weights.router_col * H ..];
        @memset(router, 0);
        for (0..E) |e| router[e * H + e] = if (e < heavy) 0x4180 else 0x3F80;
        s.shared_down = try bfs(a, rng, H * I, 0.1);
        s.shared_gate = try bfs(a, rng, H, 0.02);
        s.experts = .{ .fp4 = .{ try .random(a, rng, I, H, exact), try .random(a, rng, I, H, exact), try .random(a, rng, H, I, exact) } };
        return s;
    }

    /// Layer `layer` of a checkpoint as stored, its experts in the format expert 0's gate_proj.weight names (C1).
    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, layer: usize) !Synth {
        var ck = try core.checkpoint.Checkpoint.openModel(a, io, dir);
        defer ck.close();
        var s: Synth = undefined;
        s.moe_in = try a.alloc(u16, weights.moe_in_rows * H);
        const at = [_]usize{ 0, config.shared_inner, weights.router_col };
        for ([_][]const u8{ "shared_expert.gate_proj", "shared_expert.up_proj", "gate" }, at, [_]usize{ config.shared_inner, config.shared_inner, E }) |p, row, n|
            try get(&ck, std.mem.sliceAsBytes(s.moe_in[row * H ..][0 .. n * H]), "{d}.mlp.{s}.weight", .{ layer, p }, .bf16, &.{ n, H });
        s.shared_down = try a.alloc(u16, H * I);
        try get(&ck, std.mem.sliceAsBytes(s.shared_down), "{d}.mlp.shared_expert.down_proj.weight", .{layer}, .bf16, &.{ H, I });
        s.shared_gate = try a.alloc(u16, H);
        try get(&ck, std.mem.sliceAsBytes(s.shared_gate), "{d}.mlp.shared_expert_gate.weight", .{layer}, .bf16, &.{ 1, H });
        var name: [128]u8 = undefined;
        const gate0 = try ck.get(try std.fmt.bufPrint(&name, weights.prefix ++ "layers.{d}.mlp.experts.0.gate_proj.weight", .{layer}));
        if (gate0.dtype == .u8) {
            var f: [3]Fp4 = undefined;
            for (&f, projections, shapes) |*p, proj, nk| {
                const n, const k = nk;
                p.* = try .alloc(a, n, k);
                for (0..E) |e| {
                    try get(&ck, p.codes[e * n * k / 2 ..][0 .. n * k / 2], "{d}.mlp.experts.{d}.{s}.weight", .{ layer, e, proj }, .u8, &.{ n, k / 2 });
                    try get(&ck, p.scales[e * n * k / 16 ..][0 .. n * k / 16], "{d}.mlp.experts.{d}.{s}.weight_scale", .{ layer, e, proj }, .f8_e4m3, &.{ n, k / 16 });
                    try get(&ck, std.mem.asBytes(&p.global[e]), "{d}.mlp.experts.{d}.{s}.weight_scale_2", .{ layer, e, proj }, .f32, &.{});
                }
            }
            s.experts = .{ .fp4 = f };
        } else {
            var f: [3]Fp8 = undefined;
            for (&f, projections, shapes) |*p, proj, nk| {
                const n, const k = nk;
                const blocks = (n / 128) * (k / 128);
                p.* = .{ .n = n, .k = k, .codes = try a.alloc(u8, E * n * k), .inv = try a.alloc(f32, E * blocks) };
                for (0..E) |e| {
                    try get(&ck, p.codes[e * n * k ..][0 .. n * k], "{d}.mlp.experts.{d}.{s}.weight", .{ layer, e, proj }, .f8_e4m3, &.{ n, k });
                    try get(&ck, std.mem.sliceAsBytes(p.inv[e * blocks ..][0..blocks]), "{d}.mlp.experts.{d}.{s}.weight_scale_inv", .{ layer, e, proj }, .f32, &.{ n / 128, k / 128 });
                }
            }
            s.experts = .{ .fp8 = f };
        }
        return s;
    }
};

fn get(ck: *core.checkpoint.Checkpoint, out: []u8, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape: []const usize) !void {
    var name: [128]u8 = undefined;
    @memcpy(out, (try ck.expect(try std.fmt.bufPrint(&name, weights.prefix ++ "layers." ++ fmt, args), dtype, shape)).bytes);
}

pub fn bfs(a: std.mem.Allocator, rng: std.Random, n: usize, scale: f32) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*v| v.* = h.bf16Of((rng.float(f32) * 2 - 1) * scale);
    return out;
}

fn bfr(v: f64) f64 {
    return h.bfOf(h.bf16Of(@floatCast(v)));
}

fn silu(v: f64) f64 {
    return v / (1 + @exp(-v));
}

/// A bf16 step at |v|: 2^(floor(log2 |v|) - 7).
fn ulp(v: f64) f64 {
    if (v == 0) return 0x1p-133;
    return std.math.ldexp(@as(f64, 1), std.math.frexp(v).exponent - 8);
}

fn dot(a: []const f64, b: []const f64) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

fn absDot(a: []const f64, b: []const f64) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += @abs(x * y);
    return sum;
}

fn widen(a: std.mem.Allocator, v: []const u16) ![]f64 {
    const out = try a.alloc(f64, v.len);
    for (out, v) |*o, b| o.* = h.bfOf(b);
    return out;
}

/// Softmax top-8 of the router's bf16 logits, lower expert first on ties, renormalized (drama_topk's rule).
fn route(logits: []const f64, picks: []usize, wts: []f64) void {
    var p: [E]f64 = undefined;
    const most = std.mem.max(f64, logits);
    var sum: f64 = 0;
    for (&p, logits) |*v, l| {
        v.* = @exp(l - most);
        sum += v.*;
    }
    var taken: [E]bool = @splat(false);
    var kept: f64 = 0;
    for (picks, wts) |*pick, *w| {
        var best: usize = E;
        for (0..E) |e| if (!taken[e] and (best == E or p[e] > p[best])) {
            best = e;
        };
        taken[best] = true;
        pick.* = best;
        w.* = p[best] / sum;
        kept += w.*;
    }
    for (wts) |*w| w.* /= kept;
}

/// Drama's MoE in float64, bf16-rounded where the kernels store or cast bf16, with each output's envelope in `tol`.
fn reference(a: std.mem.Allocator, sy: Synth, xs: []const u16, picks: []usize, want: []f64, tol: []f64) !void {
    const x = try widen(a, xs);
    const moe_in = try widen(a, sy.moe_in);
    const wts = try a.alloc(f64, rows * top);
    var logits: [E]f64 = undefined;
    for (0..rows) |r| {
        for (&logits, 0..) |*l, e| l.* = bfr(dot(x[r * H ..][0..H], moe_in[(weights.router_col + e) * H ..][0..H]));
        route(&logits, picks[r * top ..][0..top], wts[r * top ..][0..top]);
    }
    // routed pairs expert by expert: Drama's SwiGLU rounding, then unweighted down rows and their sums of |terms|
    const down = try a.alloc(f64, rows * top * H);
    const down_abs = try a.alloc(f64, rows * top * H);
    const wg = try a.alloc(f64, I * H);
    const wu = try a.alloc(f64, I * H);
    const wd = try a.alloc(f64, H * I);
    var act: [I]f64 = undefined;
    for (0..E) |e| {
        if (std.mem.indexOfScalar(usize, picks, e) == null) continue;
        sy.experts.dequant(0, e, wg);
        sy.experts.dequant(1, e, wu);
        sy.experts.dequant(2, e, wd);
        for (picks, 0..) |p, pair| {
            if (p != e) continue;
            const xr = x[(pair / top) * H ..][0..H];
            for (&act, 0..) |*v, j| v.* = bfr(silu(bfr(dot(xr, wg[j * H ..][0..H]))) * bfr(dot(xr, wu[j * H ..][0..H])));
            for (0..H) |c| {
                down[pair * H + c] = dot(&act, wd[c * I ..][0..I]);
                down_abs[pair * H + c] = absDot(&act, wd[c * I ..][0..I]);
            }
        }
    }
    const sd = try widen(a, sy.shared_down);
    const sg = try widen(a, sy.shared_gate);
    var hs: [I]f64 = undefined;
    for (0..rows) |r| {
        const xr = x[r * H ..][0..H];
        for (&hs, 0..) |*v, j| v.* = bfr(silu(bfr(dot(xr, moe_in[j * H ..][0..H]))) * bfr(dot(xr, moe_in[(config.shared_inner + j) * H ..][0..H])));
        const gate = 1 / (1 + @exp(-dot(xr, sg)));
        for (0..H) |c| {
            const shared = bfr(dot(&hs, sd[c * I ..][0..I]));
            var sum: f64 = 0;
            var drift: f64 = 0;
            for (0..top) |slot| {
                const pair = r * top + slot;
                const product = bfr(wts[pair] * down[pair * H + c]);
                sum += product;
                drift += ulp(product) + 0x1p-11 * wts[pair] * down_abs[pair * H + c];
            }
            const routed = bfr(sum);
            const out = bfr(gate * shared + routed);
            want[r * H + c] = out;
            // empirical envelope (no proved bound; fixed after a failing v1 run): a bf16 step a rounding, 2^-11 |terms|
            tol[r * H + c] = ulp(out) + gate * (ulp(shared) + 0x1p-11 * absDot(&hs, sd[c * I ..][0..I])) + ulp(routed) + drift;
        }
    }
}

/// Every output and reference value finite, routing mask or not; the rows the mask leaves out are reported apart.
fn finite(what: []const u8, got: []const u8, want: []const f64, same: []const bool) !void {
    var nonfinite: usize = 0;
    for (want, 0..) |w, i| nonfinite += @intFromBool(!std.math.isFinite(h.bfOf(std.mem.bytesToValue(u16, got[2 * i ..][0..2]))) or !std.math.isFinite(w));
    const excluded = std.mem.count(bool, same, &.{false});
    std.debug.print("RESULT {s}: {d} non-finite outputs or references over all {d} rows; {d} rows routed unlike the reference left out below\n", .{ what, nonfinite, same.len, excluded });
    try check.expect(nonfinite == 0, "{s}: {d} non-finite outputs or references", .{ what, nonfinite });
}

/// `got` (bf16 [rows][H]) on rows routed alike: finite; envelope limits 1 an output, 2^-9 relative RMS, 0.5 a column.
fn compare(what: []const u8, got: []const u8, want: []const f64, tol: []const f64, same: []const bool) !void {
    try finite(what, got, want, same);
    var col_err: [H]f64 = @splat(0);
    var col_ref: [H]f64 = @splat(0);
    var col_q: [H]f64 = @splat(0);
    var worst: f64 = 0;
    var at: usize = 0;
    var exact: usize = 0;
    var n: usize = 0;
    for (want, tol, 0..) |w, t, i| {
        if (!same[i / H]) continue;
        n += 1;
        const g = h.bfOf(std.mem.bytesToValue(u16, got[2 * i ..][0..2]));
        const d = g - w;
        if (d == 0) exact += 1;
        col_err[i % H] += d * d;
        col_ref[i % H] += w * w;
        col_q[i % H] += (d / t) * (d / t);
        if (!(@abs(d) / t <= worst)) {
            worst = @abs(d) / t;
            at = i;
        }
    }
    var err: f64 = 0;
    var ref: f64 = 0;
    var col_rel: f64 = 0;
    var col_tol: f64 = 0;
    var rel_at: usize = 0;
    var tol_at: usize = 0;
    const m: f64 = @floatFromInt(n / H);
    for (col_err, col_ref, col_q, 0..) |e, r, q, c| {
        err += e;
        ref += r;
        if (!(@sqrt(e / r) <= col_rel)) {
            col_rel = @sqrt(e / r);
            rel_at = c;
        }
        if (!(@sqrt(q / m) <= col_tol)) {
            col_tol = @sqrt(q / m);
            tol_at = c;
        }
    }
    const agg = @sqrt(err / ref);
    std.debug.print("RESULT {s}: {d} of {d} outputs bit-equal to float64; worst {d:.3} of its envelope (row {d} column {d})\n", .{ what, exact, n, worst, at / H, at % H });
    std.debug.print("RESULT {s}: relative RMS error {e:.3} overall; worst column {d} {e:.3}; worst column RMS in envelopes {d} {d:.3}\n", .{ what, agg, rel_at, col_rel, tol_at, col_tol });
    try check.expect(worst <= 1 and agg <= 0x1p-9 and col_tol <= 0.5, "{s}: envelope limits are 1 an output, 2^-9 relative RMS, 0.5 RMS a column", .{what});
}

/// `got` against a reference it is not expected to meet (another format's weights): the distance, reported only.
fn distance(what: []const u8, got: []const u8, want: []const f64, same: []const bool) !void {
    try finite(what, got, want, same);
    var col_err: [H]f64 = @splat(0);
    var col_ref: [H]f64 = @splat(0);
    for (want, 0..) |w, i| {
        if (!same[i / H]) continue;
        const d = h.bfOf(std.mem.bytesToValue(u16, got[2 * i ..][0..2])) - w;
        col_err[i % H] += d * d;
        col_ref[i % H] += w * w;
    }
    var err: f64 = 0;
    var ref: f64 = 0;
    var col: f64 = 0;
    for (col_err, col_ref) |e, r| {
        err += e;
        ref += r;
        col = @max(col, @sqrt(e / r));
    }
    std.debug.print("RESULT {s}: relative RMS distance {e:.3} overall, {e:.3} worst column\n", .{ what, @sqrt(err / ref), col });
}

/// One layer's production moe on every input row in `chunks`, into `out` (bf16 [rows][H]).
pub fn replay(m: *Model, s: cuda.Stream, layer: usize, chunks: []const usize, in: []const u8, out: []u8) !void {
    try m.reset(s);
    try m.replayOperator(s, layer, .moe, chunks, in, out);
}

const ones: [rows]usize = @splat(1);
const sevens = @as([rows / 7]usize, @splat(7)) ++ [_]usize{rows % 7};

/// The layer's 128-row replay (into outs[0]) against its 1-row and 7-row replays of the same rows, byte for byte.
pub fn rowCounts(m: *Model, s: cuda.Stream, layer: usize, what: []const u8, in: []const u8, outs: [3][]u8) !void {
    try replay(m, s, layer, &ones, in, outs[1]);
    try replay(m, s, layer, &sevens, in, outs[2]);
    try replay(m, s, layer, &.{rows}, in, outs[0]);
    var name: [128]u8 = undefined;
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: 1-row replays against the 128-row one", .{what}), outs[1], outs[0]);
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: 7-row replays against the 128-row one", .{what}), outs[2], outs[0]);
    check.pass("{s}: {d} rows in passes of 1, 7 and 128 rows are bit-equal", .{ what, rows });
}

/// Pinned input rows and three pinned output sets for the replays.
pub const Pinned = struct {
    in: cuda.HostBuffer,
    host: [3]cuda.HostBuffer,
    outs: [3][]u8,

    pub fn init(d: *const cuda.Driver) !Pinned {
        var io: Pinned = undefined;
        io.in = try cuda.HostBuffer.alloc(d, rows * H * 2);
        errdefer io.in.free();
        var made: usize = 0;
        errdefer for (io.host[0..made]) |*o| o.free();
        while (made < io.host.len) : (made += 1) io.host[made] = try cuda.HostBuffer.alloc(d, rows * H * 2);
        for (&io.outs, io.host) |*o, hb| o.* = hb.bytes;
        return io;
    }

    pub fn deinit(io: *Pinned) void {
        for (&io.host) |*o| o.free();
        io.in.free();
    }
};

/// Row counts, routing against the reference's, then outputs: in the envelope against `sy`, measured against `other`.
pub fn scenario(gpu: check.Gpu, a: std.mem.Allocator, m: *Model, s: cuda.Stream, layer: usize, sy: Synth, other: ?Synth, x: []const u16, what: []const u8, io: *Pinned, exact_routing: bool) !void {
    @memcpy(io.in.bytes, std.mem.sliceAsBytes(x));
    try rowCounts(m, s, layer, what, io.in.bytes, io.outs);
    const gpu_picks = try a.alloc(i32, rows * top);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = m.s.picks, .len = rows * top * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(gpu_picks));
    const picks = try a.alloc(usize, rows * top);
    const want = try a.alloc(f64, rows * H);
    const tol = try a.alloc(f64, rows * H);
    try reference(a, sy, x, picks, want, tol);
    const same = try a.alloc(bool, rows);
    var differ: usize = 0;
    for (same, 0..) |*ok, r| {
        ok.* = true;
        for (0..top) |j| ok.* = ok.* and gpu_picks[r * top + j] == @as(i32, @intCast(picks[r * top + j]));
        differ += @intFromBool(!ok.*);
    }
    std.debug.print("RESULT {s}: {d} of {d} rows route to other experts or slots than the float64 reference's\n", .{ what, differ, rows });
    try check.expect(differ == 0 or !exact_routing, "{s}: exact router logits route every row as the reference", .{what});
    try check.expect(differ < rows, "{s}: no row routes as the reference", .{what});
    if (sy.experts == .fp4) try compare(what, io.outs[0], want, tol, same) else try distance(what, io.outs[0], want, same);
    if (other) |o| {
        try reference(a, o, x, picks, want, tol);
        var name: [128]u8 = undefined;
        try distance(try std.fmt.bufPrint(&name, "{s} against the {t} source's weights", .{ what, std.meta.activeTag(o.experts) }), io.outs[0], want, same);
    }
}

test "a non-finite output or reference on a row the routing mask leaves out still fails, checked or measured" {
    var got: [2 * H]u16 = @splat(0x3F80); // 1.0
    var want: [2 * H]f64 = @splat(1.0);
    const tol: [2 * H]f64 = @splat(1.0);
    const same = [2]bool{ true, false };
    try compare("finite", std.mem.sliceAsBytes(&got), &want, &tol, &same);
    try distance("finite", std.mem.sliceAsBytes(&got), &want, &same);
    got[H + 5] = 0x7FC0; // NaN on row 1, which routed unlike the reference
    try std.testing.expectError(error.TestFailed, compare("masked NaN output", std.mem.sliceAsBytes(&got), &want, &tol, &same));
    try std.testing.expectError(error.TestFailed, distance("masked NaN output", std.mem.sliceAsBytes(&got), &want, &same));
    got[H + 5] = 0x3F80;
    want[H + 5] = std.math.inf(f64);
    try std.testing.expectError(error.TestFailed, compare("masked infinite reference", std.mem.sliceAsBytes(&got), &want, &tol, &same));
    try std.testing.expectError(error.TestFailed, distance("masked infinite reference", std.mem.sliceAsBytes(&got), &want, &same));
}
