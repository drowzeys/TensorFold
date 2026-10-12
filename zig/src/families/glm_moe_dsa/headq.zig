//! A quantized output head (TF_GLM53_HEAD_Q, off by default): lm_head as an EXL3 linear instead of bf16 rows.
//!
//! The head stays split over the ranks by vocabulary share (rows rank * V / world ..), but an EXL3 linear can only be
//! cut along whole 128-wide Hadamard blocks of its outputs and GLM-5.3's share (154880 / 4 = 38720 = 302.5 blocks) is
//! not a whole number of them. So a rank holds the blocks that cover its share - 303 of them, one shared with a
//! neighbour - runs the ordinary linear launch over those (rot_in, linear_kernel) and copies its share out of the
//! result (Cut.off values in). An output's value depends on its own block's tiles and scales alone, so a token's logit
//! is the same bits whichever rank computes it and the two holders of a shared block agree.
//!
//! The tensors (lm_head.trellis int16 [hidden/16, vocab/16, 16 * bits], lm_head.suh fp16 [hidden], lm_head.svh fp16
//! [vocab], lm_head.mul1) come from a safetensors file of their own (the setting's value) or, with the value "pack",
//! from the checkpoint. `replay` is the launch on the CPU; tools/glm53/head_q_ref.py is its numpy reference.
const std = @import("std");
const core = @import("core");
const exl3 = @import("exl3.zig");
const fmt = core.exl3_format;
const st = core.safetensors;
pub const core_st = st;
pub const core_ck = core.Checkpoint;

pub const had = 128;

/// TF_GLM53_HEAD_Q / --head-q: unset, "", "0" or "off": the bf16 head; "pack": lm_head.* of the checkpoint; else
/// the path of a safetensors file (or of a directory holding lm_head.safetensors).
pub const Setting = union(enum) {
    off,
    pack,
    file: []const u8,

    pub fn parse(v: ?[]const u8) Setting {
        const t = std.mem.trim(u8, v orelse return .off, " ");
        if (t.len == 0 or std.mem.eql(u8, t, "0") or std.ascii.eqlIgnoreCase(t, "off")) return .off;
        if (std.ascii.eqlIgnoreCase(t, "pack") or std.mem.eql(u8, t, "1")) return .pack;
        return .{ .file = t };
    }

    pub fn on(s: Setting) bool {
        return s != .off;
    }
};

/// The Hadamard blocks of the head a rank holds and where its vocabulary share starts inside them.
pub const Cut = struct {
    b0: usize, // first block
    nb: usize, // blocks
    off: usize, // the share's first output, counted from block b0's first
    part: usize, // the share: vocab / world outputs

    pub fn of(vocab: usize, world: usize, rank: usize) !Cut {
        if (world == 0 or rank >= world or vocab % world != 0 or vocab % had != 0) return error.UnevenVocab;
        const part = vocab / world;
        const v0 = rank * part;
        const b0 = v0 / had;
        const off = v0 - b0 * had;
        // the copy out of the launch's result moves 16-byte words (glm53_copy_rows) of fp32
        if (off % 4 != 0 or part % 4 != 0) return error.UnevenVocab;
        return .{ .b0 = b0, .nb = (v0 + part + had - 1) / had - b0, .off = off, .part = part };
    }

    /// outputs the rank's launch computes
    pub fn n(c: Cut) usize {
        return c.nb * had;
    }
};

/// Tile columns [8 b0, 8 (b0 + nb)) of every tile row: raw [kt, nt, tile_bytes] -> out [kt, 8 nb, tile_bytes].
pub fn cutTrellis(out: []u8, raw: []const u8, kt: usize, nt: usize, tile_bytes: usize, c: Cut) void {
    const take = c.nb * 8 * tile_bytes;
    std.debug.assert(raw.len == kt * nt * tile_bytes and out.len == kt * take and (c.b0 + c.nb) * 8 <= nt);
    for (0..kt) |r| @memcpy(out[r * take ..][0..take], raw[(r * nt + c.b0 * 8) * tile_bytes ..][0..take]);
}

/// Where the head's tensors are read from: a file of their own, or the checkpoint.
pub const Source = struct {
    file: ?*const st.File = null,
    ck: ?*const core.Checkpoint = null,

    pub fn get(s: Source, name: []const u8) ?st.Tensor {
        if (s.file) |f| return f.get(name);
        if (s.ck) |ck| for (ck.files.items) |*f| if (f.get(name)) |t| return t;
        return null;
    }
};

/// The setting's file: the path itself, or lm_head.safetensors inside it when it is a directory.
pub fn openFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !st.File {
    return st.File.open(gpa, io, path) catch |e| {
        const inner = try std.fs.path.join(gpa, &.{ path, "lm_head.safetensors" });
        defer gpa.free(inner);
        return st.File.open(gpa, io, inner) catch {
            if (!@import("builtin").is_test) std.log.err("glm53: the quantized head {s} does not open ({t})", .{ path, e });
            return error.NoQuantizedHead;
        };
    };
}

/// The head's shape, checked against the model: (half-bits a value, the three tensors).
pub const Parts = struct {
    k2: usize,
    trellis: st.Tensor,
    suh: st.Tensor,
    svh: st.Tensor,

    pub fn of(s: Source, vocab: usize, hidden: usize) !Parts {
        const t = s.get("lm_head.trellis") orelse return fail("no lm_head.trellis");
        const suh = s.get("lm_head.suh") orelse return fail("no lm_head.suh");
        const svh = s.get("lm_head.svh") orelse return fail("no lm_head.svh");
        if (s.get("lm_head.mul1") == null) return fail("no lm_head.mul1 (only the mul1 codebook has kernels here)");
        if (hidden % had != 0 or vocab % had != 0) return fail("the model's hidden size or vocabulary is not a multiple of 128");
        if (t.dtype != .i16 or t.rank != 3 or t.shape[0] * 16 != hidden or t.shape[1] * 16 != vocab or t.shape[2] % 8 != 0) return fail("lm_head.trellis is not int16 [hidden / 16, vocab / 16, 16 * bits]");
        if (!suh.is(.f16, &.{hidden}) or !svh.is(.f16, &.{vocab})) return fail("lm_head.suh / svh are not fp16 [hidden] / [vocab] (packed sign words are not read)");
        const k2 = t.shape[2] / 8;
        if (std.mem.indexOfScalar(usize, &exl3.linear_k2s, k2) == null) {
            std.log.err("glm53: a head of {d}.{d} bits has no kernel (3, 4, 5, 6 and 8 bits do)", .{ k2 / 2, (k2 % 2) * 5 });
            return error.UnsupportedExl3Width;
        }
        return .{ .k2 = k2, .trellis = t, .suh = suh, .svh = svh };
    }

    fn fail(comptime what: []const u8) error{BadQuantizedHead} {
        if (!@import("builtin").is_test) std.log.err("glm53: quantized head: " ++ what, .{});
        return error.BadQuantizedHead;
    }

    /// What every rank must agree on: the width, both scale vectors and the trellis' two ends.
    pub fn identity(p: Parts) u64 {
        var h = std.hash.Wyhash.init(0x4ead);
        const dims = [_]u64{ p.k2, p.trellis.shape[0], p.trellis.shape[1] };
        h.update(std.mem.asBytes(&dims));
        h.update(p.suh.bytes);
        h.update(p.svh.bytes);
        const edge = @min(p.trellis.bytes.len, 1 << 16);
        h.update(p.trellis.bytes[0..edge]);
        h.update(p.trellis.bytes[p.trellis.bytes.len - edge ..]);
        return h.final();
    }
};

/// The setting's identity for the ranks' start record (0: off). Opens the file's header and reads about 0.5 MB.
pub fn identityOf(gpa: std.mem.Allocator, io: std.Io, s: Setting, model: []const u8, vocab: usize, hidden: usize) !u64 {
    switch (s) {
        .off => return 0,
        .file => |path| {
            var f = try openFile(gpa, io, path);
            defer f.close(io);
            return (try Parts.of(.{ .file = &f }, vocab, hidden)).identity() | 1;
        },
        .pack => {
            var ck = try core.Checkpoint.openModel(gpa, io, model);
            defer ck.close();
            return (try Parts.of(.{ .ck = &ck }, vocab, hidden)).identity() | 1;
        },
    }
}

/// A rank's part of the head on the host: its blocks' tiles (row order, as stored) and scales.
pub const Host = struct {
    gpa: std.mem.Allocator,
    k: usize,
    k2: usize,
    cut: Cut,
    trellis: []align(2) u8, // int16 [k / 16, 8 nb, 8 k2]
    suh: []align(2) u8, // fp16 [k]
    svh: []align(2) u8, // fp16 [128 nb]

    pub fn load(gpa: std.mem.Allocator, s: Source, vocab: usize, hidden: usize, world: usize, rank: usize) !Host {
        const p = try Parts.of(s, vocab, hidden);
        const c = try Cut.of(vocab, world, rank);
        const kt = hidden / 16;
        const tb = 16 * p.k2;
        const trellis = try gpa.alignedAlloc(u8, .@"2", kt * c.nb * 8 * tb);
        errdefer gpa.free(trellis);
        cutTrellis(trellis, p.trellis.bytes, kt, vocab / 16, tb, c);
        const suh = try gpa.alignedAlloc(u8, .@"2", hidden * 2);
        errdefer gpa.free(suh);
        @memcpy(suh, p.suh.bytes);
        const svh = try gpa.alignedAlloc(u8, .@"2", c.n() * 2);
        errdefer gpa.free(svh);
        @memcpy(svh, p.svh.bytes[c.b0 * had * 2 ..][0 .. c.n() * 2]);
        return .{ .gpa = gpa, .k = hidden, .k2 = p.k2, .cut = c, .trellis = trellis, .suh = suh, .svh = svh };
    }

    pub fn deinit(h: *Host) void {
        h.gpa.free(h.trellis);
        h.gpa.free(h.suh);
        h.gpa.free(h.svh);
        h.* = undefined;
    }

    /// The device layout's words (exl3.strips) of the rank's tiles.
    pub fn strips(h: *const Host, out: []u8) void {
        exl3.strips(out, h.trellis, h.k, h.cut.n(), 16 * h.k2);
    }
};

const had_scale: f32 = 0.08838834764831845; // 1 / sqrt(128), decode.cuh HAD_SCALE

/// decode.cuh fwht128: the unscaled Walsh-Hadamard transform of 128 fp32 values, strides 1, 2, ... 64 (the kernel's
/// lanes hold 4 values each: strides 1 and 2 inside a lane, then the shuffles), every sum the kernel's own.
pub fn fwht128(v: *[had]f32) void {
    var s: usize = 1;
    while (s < had) : (s *= 2) {
        for (0..had) |i| {
            if (i & s != 0) continue;
            const a = v[i];
            const b = v[i + s];
            v[i] = a + b;
            v[i + s] = a - b;
        }
    }
}

fn bf16Value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// rot_in_kernel on one bf16 row: xh = fp16(((x suh) H) / sqrt(128)), the kernel's fp32 steps.
pub fn rotIn(x: []const u16, suh: []const f16, xh: []f16) void {
    std.debug.assert(x.len == suh.len and xh.len == x.len and x.len % had == 0);
    var blk: usize = 0;
    while (blk < x.len) : (blk += had) {
        var v: [had]f32 = undefined;
        for (0..had) |i| v[i] = bf16Value(x[blk + i]) * @as(f32, suh[blk + i]);
        fwht128(&v);
        for (0..had) |i| xh[blk + i] = @floatCast(v[i] * had_scale);
    }
}

/// The rank's launch on the CPU for `m` bf16 rows `x` [m, k]: rot_in, linear_kernel over the rank's blocks with the
/// K splits and warps exl3.plan gives the shape (a tile step's 16 products are summed in float64 and added to the
/// warp's fp32 sum: the mma's own order inside a step is not specified, so the device may differ in the last bits),
/// the output Hadamard and svh, then the share copied out. `out` fp32 [m, cut.part].
pub fn replay(gpa: std.mem.Allocator, h: *const Host, x: []const u16, m: usize, out: []f32) !void {
    const K = h.k;
    const N = h.cut.n();
    if (x.len != m * K or out.len != m * h.cut.part) return error.Invalid;
    const wq = try gpa.alloc(f16, K * N);
    defer gpa.free(wq);
    fmt.unpack(.{ .halves = @intCast(h.k2) }, .mul1, std.mem.bytesAsSlice(i16, h.trellis), K / 16, N / 16, wq);
    const xh = try gpa.alloc(f16, K);
    defer gpa.free(xh);
    const sk_wk = exl3.plan(K, N);
    const sk = sk_wk[0];
    const wk = sk_wk[1];
    if ((K / 16) % (sk * wk) != 0) return error.BadSplit;
    const per_warp = K / 16 / sk / wk;
    const suh = std.mem.bytesAsSlice(f16, h.suh);
    const svh = std.mem.bytesAsSlice(f16, h.svh);
    const row = try gpa.alloc(f32, N);
    defer gpa.free(row);
    for (0..m) |r| {
        rotIn(x[r * K ..][0..K], suh, xh);
        for (0..N) |col| {
            var total: f32 = 0;
            for (0..sk) |split| {
                var z: f32 = 0;
                for (0..wk) |warp| {
                    var acc: f32 = 0;
                    const kt0 = split * per_warp * wk + warp * per_warp;
                    for (kt0..kt0 + per_warp) |kt| {
                        var d: f64 = 0;
                        for (0..16) |i| d += @as(f64, xh[kt * 16 + i]) * @as(f64, wq[(kt * 16 + i) * N + col]);
                        acc += @floatCast(d);
                    }
                    z = if (warp == 0) acc else z + acc;
                }
                total = if (split == 0) z else total + z;
            }
            row[col] = total;
        }
        var blk: usize = 0;
        while (blk < N) : (blk += had) {
            const v: *[had]f32 = row[blk..][0..had];
            fwht128(v);
            for (0..had) |i| v[i] = v[i] * had_scale * @as(f32, svh[blk + i]);
        }
        @memcpy(out[r * h.cut.part ..][0..h.cut.part], row[h.cut.off..][0..h.cut.part]);
    }
}

/// The Walsh-Hadamard transform of `v`'s 128 values, float64, strides 1 .. 64 (unscaled; the natural order).
fn fwht128f64(v: *[had]f64) void {
    var s: usize = 1;
    while (s < had) : (s *= 2) {
        for (0..had) |i| {
            if (i & s != 0) continue;
            const a = v[i];
            const b = v[i + s];
            v[i] = a + b;
            v[i + s] = a - b;
        }
    }
}

/// The dense matrix of an EXL3 linear (mul1): W = diag(suh) H_K W_q H_N diag(svh) with y = x W, in float64 steps
/// (core.exl3_format.dequantize's, the transforms as butterflies), as fp32 rows by output: out[o * K + i] = W[i, o],
/// the layout of a checkpoint's `.weight` [N, K]. TF_GLM53_KVB_Q decodes an EXL3 kv_b with it at load.
pub fn dense(gpa: std.mem.Allocator, k2: usize, trellis: []const i16, K: usize, N: usize, suh: []const f16, svh: []const f16, out: []f32) !void {
    if (K % had != 0 or N % had != 0 or trellis.len != K / 16 * (N / 16) * 8 * k2 or suh.len != K or svh.len != N or out.len != K * N) return error.Invalid;
    const wq = try gpa.alloc(f16, K * N);
    defer gpa.free(wq);
    fmt.unpack(.{ .halves = @intCast(k2) }, .mul1, trellis, K / 16, N / 16, wq);
    const m = try gpa.alloc(f64, K * N);
    defer gpa.free(m);
    const inv = 1.0 / @sqrt(@as(f64, had));
    var col: [had]f64 = undefined;
    // H_K down every column, 128 rows a block, then suh
    for (0..N) |n| {
        var kb: usize = 0;
        while (kb < K) : (kb += had) {
            for (0..had) |i| col[i] = wq[(kb + i) * N + n];
            fwht128f64(&col);
            for (0..had) |i| m[(kb + i) * N + n] = col[i] * inv * @as(f64, suh[kb + i]);
        }
    }
    // H_N along every row, 128 columns a block, then svh; stored by output
    for (0..K) |k| {
        var nb: usize = 0;
        while (nb < N) : (nb += had) {
            const v: *[had]f64 = m[k * N + nb ..][0..had];
            fwht128f64(v);
            for (0..had) |i| out[(nb + i) * K + k] = @floatCast(v[i] * inv * @as(f64, svh[nb + i]));
        }
    }
}

// ------------------------------------------------------------------------------------------------- tests ---

test "a rank's blocks cover its vocabulary share" {
    // GLM-5.3: 154880 tokens, 1210 blocks, four shares of 302.5 blocks
    const want = [_][3]usize{ .{ 0, 303, 0 }, .{ 302, 303, 64 }, .{ 605, 303, 0 }, .{ 907, 303, 64 } };
    for (want, 0..) |w, r| {
        const c = try Cut.of(154880, 4, r);
        try std.testing.expectEqual(w, [3]usize{ c.b0, c.nb, c.off });
        try std.testing.expectEqual(@as(usize, 38720), c.part);
        try std.testing.expect(c.b0 * had + c.off == r * 38720 and c.off + c.part <= c.n() and (c.b0 + c.nb) * had <= 154880);
    }
    const one = try Cut.of(154880, 1, 0);
    try std.testing.expectEqual([3]usize{ 0, 1210, 0 }, [3]usize{ one.b0, one.nb, one.off });
    const two = try Cut.of(154880, 2, 1);
    try std.testing.expectEqual([3]usize{ 605, 605, 0 }, [3]usize{ two.b0, two.nb, two.off });
    try std.testing.expectError(error.UnevenVocab, Cut.of(154880, 3, 0));
    try std.testing.expectError(error.UnevenVocab, Cut.of(1000, 4, 0));
}

test "the setting" {
    try std.testing.expect(Setting.parse(null) == .off);
    try std.testing.expect(Setting.parse(" ") == .off);
    try std.testing.expect(Setting.parse("0") == .off);
    try std.testing.expect(Setting.parse("off") == .off);
    try std.testing.expect(Setting.parse("pack") == .pack);
    try std.testing.expect(Setting.parse("1") == .pack);
    try std.testing.expectEqualStrings("/work/head/lm_head.safetensors", Setting.parse(" /work/head/lm_head.safetensors ").file);
    try std.testing.expect(!Setting.parse(null).on() and Setting.parse("pack").on());
}

test "tile columns of a rank's blocks" {
    // 2 tile rows, 24 tile columns (3 blocks), 2-byte tiles; blocks 1..2
    var raw: [2 * 24 * 2]u8 = undefined;
    for (&raw, 0..) |*b, i| b.* = @intCast(i);
    var out: [2 * 16 * 2]u8 = undefined;
    cutTrellis(&out, &raw, 2, 24, 2, .{ .b0 = 1, .nb = 2, .off = 0, .part = 256 });
    try std.testing.expectEqualSlices(u8, raw[16..48], out[0..32]);
    try std.testing.expectEqualSlices(u8, raw[48 + 16 .. 96], out[32..64]);
}

const cases = @embedFile("headq_cases.bin");

const Case = struct {
    k2: usize,
    hidden: usize,
    vocab: usize,
    world: usize,
    rows: usize,
    trellis: []const u8,
    suh: []const u8,
    svh: []const u8,
    x: []const u8,
    logits: []const u8,
    next: usize,

    fn at(start: usize) Case {
        var f: [5]u32 = undefined;
        for (&f, 0..) |*v, i| v.* = std.mem.readInt(u32, cases[start + 4 * i ..][0..4], .little);
        var c: Case = .{ .k2 = f[0], .hidden = f[1], .vocab = f[2], .world = f[3], .rows = f[4], .trellis = &.{}, .suh = &.{}, .svh = &.{}, .x = &.{}, .logits = &.{}, .next = 0 };
        var o = start + 20;
        const lens = [_]usize{ c.hidden / 16 * (c.vocab / 16) * 16 * c.k2, c.hidden * 2, c.vocab * 2, c.rows * c.hidden * 2, c.rows * c.vocab * 4 };
        const dst = [_]*[]const u8{ &c.trellis, &c.suh, &c.svh, &c.x, &c.logits };
        for (lens, dst) |n, d| {
            d.* = cases[o..][0..n];
            o += n;
        }
        c.next = o;
        return c;
    }
};

/// The case's tensors as a rank's Host (what Host.load cuts from a file).
fn hostOf(gpa: std.mem.Allocator, c: Case, world: usize, rank: usize) !Host {
    const cut = try Cut.of(c.vocab, world, rank);
    const kt = c.hidden / 16;
    const tb = 16 * c.k2;
    const trellis = try gpa.alignedAlloc(u8, .@"2", kt * cut.nb * 8 * tb);
    errdefer gpa.free(trellis);
    cutTrellis(trellis, c.trellis, kt, c.vocab / 16, tb, cut);
    const suh = try gpa.alignedAlloc(u8, .@"2", c.hidden * 2);
    errdefer gpa.free(suh);
    @memcpy(suh, c.suh);
    const svh = try gpa.alignedAlloc(u8, .@"2", cut.n() * 2);
    @memcpy(svh, c.svh[cut.b0 * had * 2 ..][0 .. cut.n() * 2]);
    return .{ .gpa = gpa, .k = c.hidden, .k2 = c.k2, .cut = cut, .trellis = trellis, .suh = suh, .svh = svh };
}

test "a rank's launch replayed on the CPU gives the numpy reference's logits of its share" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqualStrings("GLM53HQ1", cases[0..8]);
    const count = std.mem.readInt(u32, cases[8..12], .little);
    var o: usize = 12;
    var seen12 = false;
    for (0..count) |_| {
        const c = Case.at(o);
        o = c.next;
        seen12 = seen12 or c.k2 == 12;
        const x = try gpa.alloc(u16, c.rows * c.hidden);
        defer gpa.free(x);
        @memcpy(std.mem.sliceAsBytes(x), c.x);
        const want = try gpa.alloc(f32, c.rows * c.vocab);
        defer gpa.free(want);
        @memcpy(std.mem.sliceAsBytes(want), c.logits);
        var top: f32 = 0;
        for (want) |v| top = @max(top, @abs(v));
        // the whole head in one launch (world 1): every logit, whatever K splits its shape takes
        const all = try gpa.alloc(f32, c.rows * c.vocab);
        defer gpa.free(all);
        {
            var h = try hostOf(gpa, c, 1, 0);
            defer h.deinit();
            try replay(gpa, &h, x, c.rows, all);
        }
        for (all, want) |g, w| try std.testing.expect(@abs(g - w) <= 2e-5 * top);
        // each rank's own blocks: its share of the reference
        const part = c.vocab / c.world;
        const got = try gpa.alloc(f32, c.rows * part);
        defer gpa.free(got);
        for (0..c.world) |rank| {
            var h = try hostOf(gpa, c, c.world, rank);
            defer h.deinit();
            try replay(gpa, &h, x, c.rows, got);
            for (0..c.rows) |r| for (0..part) |j| {
                const w = want[r * c.vocab + rank * part + j];
                try std.testing.expect(@abs(got[r * part + j] - w) <= 2e-5 * top);
            };
            // a row's logits do not depend on the rows beside it
            if (c.rows > 1) {
                const alone = try gpa.alloc(f32, part);
                defer gpa.free(alone);
                try replay(gpa, &h, x[(c.rows - 1) * c.hidden ..], 1, alone);
                try std.testing.expectEqualSlices(f32, got[(c.rows - 1) * part ..], alone);
            }
        }
    }
    try std.testing.expect(seen12); // a 6-bit head is among the cases
    try std.testing.expectEqual(cases.len, o);
}

test "a block two ranks hold gives both the same bits" {
    // the same K splits on both ranks (the launch's shape is the same): the shared block's outputs are equal bit for bit
    const gpa = std.testing.allocator;
    const c = Case.at(12);
    try std.testing.expect(c.world == 4 and (c.vocab / c.world) % had == had / 2);
    const x = try gpa.alloc(u16, c.rows * c.hidden);
    defer gpa.free(x);
    @memcpy(std.mem.sliceAsBytes(x), c.x);
    var h0 = try hostOf(gpa, c, c.world, 0);
    defer h0.deinit();
    var h1 = try hostOf(gpa, c, c.world, 1);
    defer h1.deinit();
    try std.testing.expectEqual(h0.cut.b0 + h0.cut.nb - 1, h1.cut.b0); // rank 0's last block is rank 1's first
    try std.testing.expectEqual(h0.cut.n(), h1.cut.n());
    // widen both shares to the whole launch (part = n, off = 0) and compare the shared block
    var w0 = h0;
    var w1 = h1;
    w0.cut = .{ .b0 = h0.cut.b0, .nb = h0.cut.nb, .off = 0, .part = h0.cut.n() };
    w1.cut = .{ .b0 = h1.cut.b0, .nb = h1.cut.nb, .off = 0, .part = h1.cut.n() };
    const a = try gpa.alloc(f32, c.rows * w0.cut.part);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, c.rows * w1.cut.part);
    defer gpa.free(b);
    try replay(gpa, &w0, x, c.rows, a);
    try replay(gpa, &w1, x, c.rows, b);
    const n = w0.cut.part;
    for (0..c.rows) |r| try std.testing.expectEqualSlices(f32, a[r * n + n - had ..][0..had], b[r * n ..][0..had]);
}

test "the head's shape is checked and its identity follows its bytes" {
    const c = Case.at(12);
    const tshape = [_]usize{ c.hidden / 16, c.vocab / 16, 8 * c.k2, 0, 0 };
    var p: Parts = .{
        .k2 = c.k2,
        .trellis = .{ .dtype = .i16, .rank = 3, .shape = tshape, .bytes = c.trellis },
        .suh = .{ .dtype = .f16, .rank = 1, .shape = .{ c.hidden, 0, 0, 0, 0 }, .bytes = c.suh },
        .svh = .{ .dtype = .f16, .rank = 1, .shape = .{ c.vocab, 0, 0, 0, 0 }, .bytes = c.svh },
    };
    const id = p.identity();
    try std.testing.expectEqual(id, p.identity());
    var other = try std.testing.allocator.dupe(u8, c.svh);
    defer std.testing.allocator.free(other);
    other[c.svh.len - 1] ^= 0x80;
    p.svh.bytes = other;
    try std.testing.expect(p.identity() != id);
}

test "the dense matrix of an EXL3 linear is the reference decoder's" {
    const gpa = std.testing.allocator;
    const c = Case.at(12);
    const K = c.hidden;
    const N: usize = 384; // the head's first three blocks as a linear of their own
    const kt = K / 16;
    const tb = 16 * c.k2;
    const tr = try gpa.alignedAlloc(u8, .@"2", kt * (N / 16) * tb);
    defer gpa.free(tr);
    cutTrellis(tr, c.trellis, kt, c.vocab / 16, tb, .{ .b0 = 0, .nb = N / had, .off = 0, .part = N });
    const suh = try gpa.alloc(f16, K);
    defer gpa.free(suh);
    @memcpy(std.mem.sliceAsBytes(suh), c.suh);
    const svh = try gpa.alloc(f16, N);
    defer gpa.free(svh);
    @memcpy(std.mem.sliceAsBytes(svh), c.svh[0 .. N * 2]);
    const got = try gpa.alloc(f32, K * N);
    defer gpa.free(got);
    try dense(gpa, c.k2, std.mem.bytesAsSlice(i16, tr), K, N, suh, svh, got);
    const want = try gpa.alloc(f64, K * N);
    defer gpa.free(want);
    const scratch = try gpa.alloc(f64, 2 * K * N);
    defer gpa.free(scratch);
    fmt.dequantize(.{ .halves = @intCast(c.k2) }, .mul1, std.mem.bytesAsSlice(i16, tr), kt, N / 16, suh, svh, want, scratch);
    var top: f64 = 0;
    for (want) |v| top = @max(top, @abs(v));
    for (0..K) |k| for (0..N) |n| try std.testing.expect(@abs(@as(f64, got[n * K + k]) - want[k * N + n]) <= 1e-6 * top);
    // x @ W through the dense rows is the numpy reference's logits up to the launch's fp16 input rounding
    const x = try gpa.alloc(u16, c.rows * K);
    defer gpa.free(x);
    @memcpy(std.mem.sliceAsBytes(x), c.x);
    var ltop: f64 = 0;
    for (0..N) |n| ltop = @max(ltop, @abs(@as(f64, @as(f32, @bitCast(std.mem.readInt(u32, c.logits[n * 4 ..][0..4], .little))))));
    for (0..N) |n| {
        var d: f64 = 0;
        for (0..K) |k| d += @as(f64, bf16Value(x[k])) * @as(f64, got[n * K + k]);
        const ref: f64 = @as(f32, @bitCast(std.mem.readInt(u32, c.logits[n * 4 ..][0..4], .little)));
        try std.testing.expect(@abs(d - ref) <= 5e-3 * ltop);
    }
}

/// A safetensors image of a case's head (what the converter writes: suh, svh, trellis, the mul1 marker).
fn fileImage(gpa: std.mem.Allocator, c: Case) ![]u8 {
    const a = c.suh.len;
    const b = a + c.svh.len;
    const e = b + c.trellis.len;
    var hb: [512]u8 = undefined;
    var json = try std.fmt.bufPrint(&hb, "{{\"lm_head.suh\":{{\"dtype\":\"F16\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}},\"lm_head.svh\":{{\"dtype\":\"F16\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}},\"lm_head.trellis\":{{\"dtype\":\"I16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}},\"lm_head.mul1\":{{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[{d},{d}]}}}}", .{ c.hidden, a, c.vocab, a, b, c.hidden / 16, c.vocab / 16, 8 * c.k2, b, e, e, e + 4 });
    while (json.len % 8 != 0) : (json.len += 1) hb[json.len] = ' ';
    const out = try gpa.alloc(u8, 8 + json.len + e + 4);
    std.mem.writeInt(u64, out[0..8], json.len, .little);
    @memcpy(out[8..][0..json.len], json);
    var o: usize = 8 + json.len;
    for ([_][]const u8{ c.suh, c.svh, c.trellis, &[_]u8{ 0, 0, 0, 0 } }) |part| {
        @memcpy(out[o..][0..part.len], part);
        o += part.len;
    }
    return out;
}

test "a head file loads as a rank's part, by its path or its directory's, and names itself" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const c = Case.at(12);
    const image = try fileImage(gpa, c);
    defer gpa.free(image);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "lm_head.safetensors", .data = image });
    var pb: [96]u8 = undefined;
    const dir = try std.fmt.bufPrint(&pb, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var fb: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&fb, "{s}/lm_head.safetensors", .{dir});
    const id = try identityOf(gpa, io, .{ .file = path }, "", c.vocab, c.hidden);
    try std.testing.expect(id != 0);
    try std.testing.expectEqual(id, try identityOf(gpa, io, .{ .file = dir }, "", c.vocab, c.hidden));
    try std.testing.expectEqual(@as(u64, 0), try identityOf(gpa, io, .off, "", c.vocab, c.hidden));
    // another model's shape is refused
    try std.testing.expectError(error.BadQuantizedHead, identityOf(gpa, io, .{ .file = path }, "", c.vocab + 128, c.hidden));
    try std.testing.expectError(error.NoQuantizedHead, identityOf(gpa, io, .{ .file = ".zig-cache/tmp/no-such-head" }, "", c.vocab, c.hidden));
    var f = try openFile(gpa, io, dir);
    defer f.close(io);
    for (0..c.world) |rank| {
        var got = try Host.load(gpa, .{ .file = &f }, c.vocab, c.hidden, c.world, rank);
        defer got.deinit();
        var want = try hostOf(gpa, c, c.world, rank);
        defer want.deinit();
        try std.testing.expectEqual(want.cut, got.cut);
        try std.testing.expectEqual(c.k2, got.k2);
        try std.testing.expectEqualSlices(u8, want.trellis, got.trellis);
        try std.testing.expectEqualSlices(u8, want.suh, got.suh);
        try std.testing.expectEqualSlices(u8, want.svh, got.svh);
        // the device layout's words: tile (kt, nt) of the rank's blocks at block (nt / 8), row kt, slot nt % 8
        const words = try gpa.alloc(u8, got.trellis.len);
        defer gpa.free(words);
        got.strips(words);
        const tb = 16 * c.k2;
        const kt = c.hidden / 16;
        const nt = got.cut.nb * 8;
        const t0 = (3 * nt + 9) * tb; // tile row 3, tile column 9: block 1, slot 1
        try std.testing.expectEqualSlices(u8, got.trellis[t0..][0..tb], words[((1 * kt + 3) * 8 + 1) * tb ..][0..tb]);
    }
}
