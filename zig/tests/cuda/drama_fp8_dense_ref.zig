//! drama-fp8-dense's parts: an FP8 group as C5 stores it, its float64 reference, preregistered criteria, lane runs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const drama = @import("drama");
const check = @import("check.zig");
const h = @import("quant_ref_tests.zig").helpers;
pub const score = @import("drama_fp8_dense_score.zig");

const fp8 = cuda.fp8;
const qlinear = cuda.qlinear;
const config = drama.backbone.config;
const weights = drama.backbone.weights;
const Checkpoint = core.checkpoint.Checkpoint;

pub const H = config.hidden;
pub const Rig = @import("quant_ref_tests.zig").RigT;

/// The six backbone groups as the lane sees them, fused rows by kd inputs.
pub const shapes = [_]struct { []const u8, usize, usize }{
    .{ "gdn_in", 12288, H },
    .{ "gdn_out", H, 4096 },
    .{ "attn_in", 9216, H },
    .{ "attn_out", H, 4096 },
    .{ "shared_in", 1024, H },
    .{ "shared_out", H, 512 },
};

/// A group: its parts, the Layer field holding it as BF16, and the rows kept BF16 beside it (`split` when FP8).
pub const Desc = struct {
    name: []const u8,
    stems: []const []const u8,
    parts: []const usize,
    k: usize,
    bf16: []const u8,
    extra: []const []const u8 = &.{},
    extra_rows: []const usize = &.{},
    split: []const u8 = "",
};

const heads = config.linear_value_heads;
pub const mixers = [2][2]Desc{ .{
    .{ .name = "gdn_in", .stems = &.{ "linear_attn.in_proj_qkv", "linear_attn.in_proj_z" }, .parts = &.{ config.conv_dim, config.value_dim }, .k = H, .bf16 = "gdn_in", .extra = &.{ "linear_attn.in_proj_b", "linear_attn.in_proj_a" }, .extra_rows = &.{ heads, heads }, .split = "gdn_ba" },
    .{ .name = "gdn_out", .stems = &.{"linear_attn.out_proj"}, .parts = &.{H}, .k = config.value_dim, .bf16 = "gdn_out" },
}, .{
    .{ .name = "attn_in", .stems = &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }, .parts = &.{ 2 * config.query_heads * config.head_dim, config.kv_heads * config.head_dim, config.kv_heads * config.head_dim }, .k = H, .bf16 = "attn_in" },
    .{ .name = "attn_out", .stems = &.{"self_attn.o_proj"}, .parts = &.{H}, .k = config.query_heads * config.head_dim, .bf16 = "attn_out" },
} };
pub const shared = [2]Desc{
    .{ .name = "shared_in", .stems = &.{ "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj" }, .parts = &.{ config.shared_inner, config.shared_inner }, .k = H, .bf16 = "moe_in", .extra = &.{"mlp.gate"}, .extra_rows = &.{config.experts}, .split = "router" },
    .{ .name = "shared_out", .stems = &.{"mlp.shared_expert.down_proj"}, .parts = &.{H}, .k = config.shared_inner, .bf16 = "shared_down" },
};

/// Layer `layer`'s four groups in Layer order: its mixer's input and output, the shared expert's input and output.
pub fn groupsOf(layer: usize) [4]Desc {
    const m = mixers[@intFromBool(!config.linear(layer))];
    return .{ m[0], m[1], shared[0], shared[1] };
}

/// `stems`' parts [parts[j], k] of `dtype` in layer `layer`, concatenated; `suffix` names the tensor beside `.weight`.
pub fn read(a: std.mem.Allocator, ck: *Checkpoint, layer: usize, stems: []const []const u8, parts: []const usize, k: usize, suffix: []const u8, dtype: core.checkpoint.DType) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var name: [160]u8 = undefined;
    for (stems, parts) |stem, n| {
        const t = try ck.expect(try std.fmt.bufPrint(&name, "{s}layers.{d}.{s}{s}", .{ weights.prefix, layer, stem, suffix }), dtype, &.{ n, k });
        try out.appendSlice(a, t.bytes);
    }
    return out.items;
}

pub fn bf16Rows(a: std.mem.Allocator, bytes: []const u8) ![]u16 {
    const out = try a.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

/// One FP8 group as C5 stores it, its parts' rows concatenated: codes [n][k], row scales [n][k / 64].
pub const Group = struct {
    n: usize,
    k: usize,
    codes: []u8,
    inv: []f32,

    pub fn alloc(a: std.mem.Allocator, n: usize, k: usize) !Group {
        return .{ .n = n, .k = k, .codes = try a.alloc(u8, n * k), .inv = try a.alloc(f32, n * (k / 64)) };
    }

    pub fn value(g: Group, row: usize, col: usize) f64 {
        return h.e4m3Of(g.codes[row * g.k + col]) * g.inv[row * (g.k / 64) + col / 64];
    }

    pub fn dequant(g: Group, a: std.mem.Allocator) ![]f64 {
        const out = try a.alloc(f64, g.n * g.k);
        for (0..g.n) |r| for (0..g.k) |c| {
            out[r * g.k + c] = g.value(r, c);
        };
        return out;
    }

    /// The bf16 nearest each stored value (direct f64 RNE): the BF16 path's weights for the same matrix.
    pub fn bf16(g: Group, a: std.mem.Allocator) ![]u16 {
        const out = try a.alloc(u16, g.n * g.k);
        for (0..g.n) |r| for (0..g.k) |c| {
            out[r * g.k + c] = score.bf16Nearest(g.value(r, c));
        };
        return out;
    }

    /// The host packing of the loader's GPU one: codes in the lane's order, then the scale tiles.
    pub fn toLane(g: Group, a: std.mem.Allocator) !struct { codes: []u8, tiles: []f32 } {
        const codes = try a.alloc(u8, fp8.padded(g.n) * g.k);
        fp8.packCodes(codes, g.codes, g.n, g.k);
        const tiles = try a.alloc(f32, fp8.padded(g.n) * (g.k / 64));
        fp8.packRowScales(tiles, g.inv, g.n, g.k);
        return .{ .codes = codes, .tiles = tiles };
    }

    pub fn upload(g: Group, r: *Rig) !qlinear.Weight {
        const p = try g.toLane(r.a);
        return .{ .fp8g = fp8.weight(try r.dev(p.codes), try r.dev(std.mem.sliceAsBytes(p.tiles)), g.n, g.k) };
    }
};

/// Synthetic weights by output row j % 16: 0, 1, 4 hold subnormal or 2^8 scales, 2 saturated codes, 3 subnormal codes.
pub fn synthetic(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize) !Group {
    const g = try Group.alloc(a, n, k);
    const kg = k / 64;
    for (0..n) |j| {
        const base = std.math.exp2(-16 * rng.float(f32));
        for (g.codes[j * k ..][0..k]) |*c| {
            const sign: u8 = if (rng.boolean()) 0x80 else 0;
            c.* = switch (j % 16) {
                2 => 0x7E | sign,
                3 => rng.uintLessThan(u8, 8) | sign,
                else => code(rng),
            };
        }
        for (g.inv[j * kg ..][0..kg], 0..) |*s, i| s.* = switch (j % 16) {
            0 => if (j == 0) 2.05e-43 else subnormal(rng),
            1 => 256,
            4 => if (i % 2 == 0) 256 else subnormal(rng),
            else => base * (0.5 + 1.5 * rng.float(f32)),
        };
    }
    return g;
}

pub fn subnormalRow(j: usize) bool {
    return j % 16 == 0;
}

/// A random e4m3 code that is not NaN.
pub fn code(rng: std.Random) u8 {
    while (true) {
        const b = rng.int(u8);
        if (b & 0x7f != 0x7f) return b;
    }
}

fn subnormal(rng: std.Random) f32 {
    return @bitCast(1 + rng.uintLessThan(u32, 0x7F_FFFF)); // a positive f32 subnormal, 2^-149 to 2^-126
}

pub const Ref = score.Ref;

/// y = x w^T in float64 for m rows of x (bf16 [m][k]) and w [n][k], with each output's preregistered tolerance.
pub fn reference(a: std.mem.Allocator, w: []const f64, n: usize, k: usize, x: []const u16, m: usize) !Ref {
    const ref: Ref = .{ .want = try a.alloc(f64, m * n), .tol = try a.alloc(f64, m * n) };
    const xf = try a.alloc(f64, m * k);
    for (xf, x[0 .. m * k]) |*o, v| o.* = h.bfOf(v);
    const Work = struct {
        fn rows(r: Ref, wd: []const f64, xd: []const f64, nn: usize, kk: usize, first: usize, step: usize) void {
            var row = first;
            while (row < xd.len / kk) : (row += step) for (0..nn) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (xd[row * kk ..][0..kk], wd[j * kk ..][0..kk]) |xv, wv| {
                    sum += xv * wv;
                    abs += @abs(xv * wv);
                }
                r.want[row * nn + j] = sum;
                // bf16 out 2^-7 |y|, fp32 sums 2^-12 sum |x w|, bf16's least step, a subnormal fma step a group
                const groups: f64 = @floatFromInt(kk / 64);
                r.tol[row * nn + j] = @abs(sum) * 0x1p-7 + abs * 0x1p-12 + 0x1p-133 + groups * 0x1p-149;
            };
        }
    };
    var threads: [16]std.Thread = undefined;
    const t = @min(threads.len, m);
    for (threads[0..t], 0..) |*th, i| th.* = try std.Thread.spawn(.{}, Work.rows, .{ ref, w, xf, n, k, i, t });
    for (threads[0..t]) |th| th.join();
    return ref;
}

pub const Stats = score.Stats;

/// `got` (bf16 [m][n]) against `ref`'s first m rows; the RMS figures over the columns `gated` admits (null: all).
pub fn stats(a: std.mem.Allocator, got: []const u8, ref: Ref, n: usize, gated: ?[]const bool) !Stats {
    return score.stats(a, got, ref.want, ref.tol, n, gated);
}

pub fn report(what: []const u8, n: usize, st: Stats) void {
    std.debug.print("RESULT {s}: {d} outputs bit-equal to float64, {d} non-finite outputs, {d} non-finite references; worst {d:.3} of its tolerance" ++
        " (row {d} column {d}); relative RMS {e:.3} overall, worst column {d} {e:.3}\n", .{ what, st.exact, st.nonfinite_got, st.nonfinite_want, st.worst, st.at / n, st.at % n, st.agg, st.col_at, st.col });
}

/// (a)'s preregistered criteria (Stats.within): finite, within tolerance, gated relative RMS 2^-8, 2^-7 a column.
pub fn gate(what: []const u8, n: usize, st: Stats) !void {
    report(what, n, st);
    try check.expect(st.valid(), "{s}: non-finite outputs or references, or a nonzero error against a zero reference", .{what});
    try check.expect(st.within(), "{s}: limits are 1 tolerance an output, 2^-8 relative RMS, 2^-7 a column", .{what});
}

/// A comparison reported, not gated by its magnitude: both arms must still be finite and every error defined.
pub fn measured(what: []const u8, n: usize, st: Stats) !void {
    report(what, n, st);
    try check.expect(st.valid(), "{s}: non-finite outputs or references, or a nonzero error against a zero reference", .{what});
}

/// (b)'s preregistered criterion (Stats.noWorse): both arms valid, the lane no worse overall and in the worst column.
pub fn noWorse(what: []const u8, lane: Stats, bf16: Stats) !void {
    std.debug.print("RESULT {s}: lane relative RMS {e:.3} (worst column {e:.3}), bf16 gemm {e:.3} (worst column {e:.3})\n", .{ what, lane.agg, lane.col, bf16.agg, bf16.col });
    try check.expect(Stats.noWorse(lane, bf16), "{s}: both arms valid and the lane's error no worse than the BF16 gemm's", .{what});
}

/// Rows 0..m of x (packed kd wide) through the BF16 path's k.gemm in calls of `chunk` rows into y (packed n wide).
pub fn gemmChunked(k: *const drama.backbone.kernels.Kernels, s: cuda.Stream, x: u64, kd: usize, m: usize, chunk: usize, w: u64, n: usize, y: u64) !void {
    var r: usize = 0;
    while (r < m) : (r += chunk) try k.gemm(s, x + r * kd * 2, kd, @min(chunk, m - r), w, n, kd, y + r * n * 2, n);
}

/// Rows 0..m of x (packed k wide) through `lin` in calls of `chunk` rows into y (packed n wide).
pub fn chunked(lin: qlinear.Linear, s: cuda.Stream, w: qlinear.Weight, x: u64, m: usize, chunk: usize, y: u64, part: u64) !void {
    var r: usize = 0;
    while (r < m) : (r += chunk) {
        const rows = @min(chunk, m - r);
        try lin.decode(s, w, .{ .x = x + r * w.inputs() * 2, .part = part }, y + r * w.outputs() * 2, rows);
    }
}

/// One wide call at `rows` (c): 1-row and 7-row calls of the same rows must give its bytes; returns them.
pub fn rowCounts(r: *Rig, lin: qlinear.Linear, w: qlinear.Weight, x: u64, rows: usize, part: u64, what: []const u8) ![]u8 {
    const bytes = rows * w.outputs() * 2;
    const wide = try r.zeros(bytes);
    try lin.decode(r.s, w, .{ .x = x, .part = part }, wide, rows);
    const got = try r.back(wide, bytes);
    var name: [128]u8 = undefined;
    for ([_]usize{ 1, 7 }) |chunk| {
        const y = try r.zeros(bytes);
        try chunked(lin, r.s, w, x, rows, chunk, y, part);
        try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: {d}-row calls against the {d}-row one", .{ what, chunk, rows }), try r.back(y, bytes), got);
    }
    check.pass("{s}: {d} rows in calls of 1, 7 and {d} rows are bit-equal", .{ what, rows, rows });
    return got;
}
