//! A quantized Transcribe release's FP8 MTP head, 4-bit as Lightning's ModelOpt head (default) or bf16.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const W = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const modelopt = @import("cuda_modelopt.zig");
const salm = @import("cuda_salm.zig");
const Config = @import("config.zig").Config;
const draft_ids = @import("draft_ids.zig");
const host4 = core.affine4_host;

const Loader = W.Loader;
const Tensor = core.checkpoint.Tensor;
const Bf16 = cuda.experts.Bf16Experts;

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

const p0 = "mtp.layers.0.";
const p1 = "mtp.layers.1.mixer.";
const att = p0 ++ "mixer.";

/// The head under "llm.", at c.mtp_bits (0: bf16, 4: affine-4); lm_head's draft rows from the release's top level.
pub fn load(L: *Loader, ck: *core.Checkpoint, c: Config, with_mtp: bool) !void {
    const scope = ck.scope;
    ck.scope = "llm.";
    defer ck.scope = scope;
    if (!with_mtp) return ck.skip("mtp.");
    const small = c.mtp_bits == 4;
    const router = try modelopt.router(L, "mtp.moe.router", try ck.get(p1 ++ "gate.weight"), c);
    const bias = try ck.expect(p1 ++ "gate.e_score_correction_bias", .f32, &.{c.experts});
    const qkv = [3][]const u8{ att ++ "q_proj", att ++ "k_proj", att ++ "v_proj" };
    L.w.mtp = .{
        .enorm = try L.raw("mtp.enorm", try ck.expect(p0 ++ "enorm.weight", .bf16, &.{c.hidden})),
        .hnorm = try L.raw("mtp.hnorm", try ck.expect(p0 ++ "hnorm.weight", .bf16, &.{c.hidden})),
        .eh_proj = if (small) try tile4(L, ck, "mtp.eh_proj", &.{p0 ++ "eh_proj"}) else try modelopt.dense(L, ck, "mtp.eh_proj", &.{p0 ++ "eh_proj"}),
        .attn_norm = try L.raw("mtp.attn_norm", try ck.expect(p0 ++ "norm.weight", .bf16, &.{c.hidden})),
        .attn = .{
            .qkv = if (small) try tile4(L, ck, "mtp.attn.qkv", &qkv) else try widenedDense(L, ck, "mtp.attn.qkv", &qkv),
            .o = if (small) try tile4(L, ck, "mtp.attn.o", &.{att ++ "o_proj"}) else try modelopt.dense(L, ck, "mtp.attn.o", &.{att ++ "o_proj"}),
        },
        .moe_norm = try L.raw("mtp.moe_norm", try ck.expect("mtp.layers.1.norm.weight", .bf16, &.{c.hidden})),
        .moe = .{ .router = router, .bias = try L.raw("mtp.moe.bias", bias), .experts = if (small) try experts4(L, ck, c) else try experts16(L, ck, c) },
        .final_norm = try L.raw("mtp.final_norm", try ck.expect("mtp.layers.1.final_layernorm.weight", .bf16, &.{c.hidden})),
    };
    ck.scope = scope;
    if (small) try draftHead4(L, ck, c) else try salm.draftHead(L, ck, c);
}

/// An e4m3 code's value (fn: no infinities, 0x7F and 0xFF NaN).
pub fn e4m3(b: u8) f32 {
    const mag = b & 0x7F;
    if (mag == 0x7F) return std.math.nan(f32);
    const e: i32 = mag >> 3;
    const m: f32 = @floatFromInt(mag & 7);
    const v: f32 = if (e == 0) m * std.math.ldexp(@as(f32, 1), -9) else (1 + m / 8) * std.math.ldexp(@as(f32, 1), e - 7);
    return if (b & 0x80 != 0) -v else v;
}

/// f32 to bf16, round to nearest even (NaN stays NaN).
pub fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// An FP8 part's values (e4m3 codes times its scale) in fp32, row-major as stored.
fn values(L: *Loader, p: modelopt.Part, out: []f32) !void {
    if (p.kind != .fp8 or out.len != p.n * p.k) return error.UnexpectedTensor;
    const codes = try L.src.view(p.w.bytes);
    for (out, codes[0..out.len]) |*o, b| o.* = e4m3(b) * p.scale;
}

/// Row-stacked FP8 parts with scales of their own (q, k, v) widened to one bf16 lane weight.
fn widenedDense(L: *Loader, ck: *core.Checkpoint, label: []const u8, prefixes: []const []const u8) !kern.QLinear {
    const m = if (L.ops.k.modelopt) |*x| x else return error.ModeloptNotLoaded;
    var parts: [3]modelopt.Part = undefined;
    var n: usize = 0;
    for (prefixes, 0..) |pre, i| {
        parts[i] = try modelopt.part(ck, pre);
        if (parts[i].k != parts[0].k) return error.UnexpectedTensor;
        n += parts[i].n;
    }
    const k = parts[0].k;
    const f = try L.gpa.alloc(f32, n * k);
    defer L.gpa.free(f);
    var at: usize = 0;
    for (parts[0..prefixes.len]) |p| {
        try values(L, p, f[at .. at + p.n * k]);
        at += p.n * k;
    }
    const host = std.mem.bytesAsSlice(u16, try L.staging(n * k * 2));
    for (host, f) |*h, v| h.* = bf16Bits(v);
    const base = try L.tmp(n * k * 2);
    try L.ops.upload(base, std.mem.sliceAsBytes(host));
    var buf: [128]u8 = undefined;
    const out = try L.alloc(try name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k * 2);
    try m.pack.bf16Lane(L.ops.s, base, out, n, k);
    return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
}

/// The expert names in order: routed experts 0..E, then the shared expert.
fn partName(buf: []u8, e: usize, E: usize, up: bool) ![]const u8 {
    const leaf: []const u8 = if (up) "up_proj" else "down_proj";
    if (e == E) return name(buf, "{s}shared_experts.{s}", .{ p1, leaf });
    return name(buf, "{s}experts.{d}.{s}", .{ p1, e, leaf });
}

/// FP8 routed experts and the FP8 shared expert's halves (experts E and E + 1) widened into the bf16 experts' blocks.
fn experts16(L: *Loader, ck: *core.Checkpoint, c: Config) !kern.Experts {
    const k = &(L.ops.k.bf16x orelse return error.Bf16ExpertsNotLoaded);
    const E = c.experts;
    const w = c.expert_width;
    const D = c.hidden;
    var layer: kern.Experts = .{ .format = .bf16, .up = 0, .down = 0, .width = w, .dims = D, .experts = E + 2, .relu2 = true };
    var b: [192]u8 = undefined;
    const f = try L.gpa.alloc(f32, 2 * w * D);
    defer L.gpa.free(f);
    for ([2]bool{ true, false }) |up| {
        const n = if (up) w else D;
        const kk = if (up) D else w;
        const one = n * kk;
        const base = try L.tmp((E + 2) * one * 2);
        for (0..E + 1) |e| {
            const p = try modelopt.part(ck, try partName(&b, e, E, up));
            const want = if (e < E) [2]usize{ n, kk } else if (up) [2]usize{ 2 * w, D } else [2]usize{ D, 2 * w };
            if (p.n != want[0] or p.k != want[1]) return error.UnexpectedTensor;
            const v = f[0 .. p.n * p.k];
            try values(L, p, v);
            const host = std.mem.bytesAsSlice(u16, try L.staging(v.len * 2));
            for (host, v) |*h, x| h.* = bf16Bits(x);
            try L.ops.upload(base + e * one * 2, std.mem.sliceAsBytes(host));
        }
        const out = try L.alloc(if (up) "mtp.moe.experts.up" else "mtp.moe.experts.down", Bf16.bytes(E + 2, n, kk));
        const s = L.ops.s;
        const per = Bf16.bytes(1, n, kk);
        try k.pack(s, base, .{ .stride = one, .ld = kk, .in_out = false }, out, E, n, kk);
        const sh = base + E * one * 2; // the shared expert: up [2W, D] rows, down [D, 2W] columns
        for (0..2) |h| {
            const from: Bf16.Source = if (up) .{ .ld = D, .n0 = h * w, .in_out = false } else .{ .ld = 2 * w, .k0 = h * w, .in_out = false };
            try k.pack(s, sh, from, out + (E + h) * per, 1, n, kk);
        }
        if (up) layer.up = out else layer.down = out;
    }
    return layer;
}

/// `values` [n, k] made MLX affine-4 into words, scales and biases kept until the load's uploads finish.
fn four(L: *Loader, v: []const f32, n: usize, k: usize) ![3]Tensor {
    if (k % host4.group != 0) return error.UnexpectedTensor;
    const words = try L.keep(u32, n * k / 8);
    const sc = try L.keep(u16, n * k / host4.group);
    const bi = try L.keep(u16, n * k / host4.group);
    host4.quantize(v, words, sc, bi);
    return .{ W.made(.u32, n, k / 8, std.mem.sliceAsBytes(words)), W.made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(sc)), W.made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(bi)) };
}

/// FP8 parts (one projection, or q, k, v) made affine-4 and tiled as one lane weight.
fn tile4(L: *Loader, ck: *core.Checkpoint, label: []const u8, prefixes: []const []const u8) !kern.QLinear {
    var parts: [3][3]Tensor = undefined;
    for (prefixes, 0..) |pre, i| {
        const p = try modelopt.part(ck, pre);
        const v = try L.gpa.alloc(f32, p.n * p.k);
        defer L.gpa.free(v);
        try values(L, p, v);
        parts[i] = try four(L, v, p.n, p.k);
    }
    return .{ .affine4 = try L.tile(label, parts[0..prefixes.len]) };
}

/// FP8 routed experts made affine-4 and stacked as the MLX loader reads them, the shared expert's halves last.
fn experts4(L: *Loader, ck: *core.Checkpoint, c: Config) !kern.Experts {
    const E = c.experts;
    const w = c.expert_width;
    const D = c.hidden;
    var out: kern.Experts = .{ .format = .affine4, .up = 0, .down = 0, .experts = E + 2, .width = w, .dims = D };
    var b: [192]u8 = undefined;
    for ([2]bool{ true, false }) |up| {
        const n = if (up) w else D;
        const kk = if (up) D else w;
        const per = n * kk;
        const words = try L.keep(u32, E * per / 8);
        const sc = try L.keep(u16, E * per / host4.group);
        const bi = try L.keep(u16, E * per / host4.group);
        const v = try L.gpa.alloc(f32, 2 * per);
        defer L.gpa.free(v);
        for (0..E) |e| {
            const p = try modelopt.part(ck, try partName(&b, e, E, up));
            if (p.n != n or p.k != kk) return error.UnexpectedTensor;
            try values(L, p, v[0..per]);
            host4.quantize(v[0..per], words[e * per / 8 ..][0 .. per / 8], sc[e * per / host4.group ..][0 .. per / host4.group], bi[e * per / host4.group ..][0 .. per / host4.group]);
        }
        const routed = [3]Tensor{ W.made(.u32, E * n, kk / 8, std.mem.sliceAsBytes(words)), W.made(.bf16, E * n, kk / host4.group, std.mem.sliceAsBytes(sc)), W.made(.bf16, E * n, kk / host4.group, std.mem.sliceAsBytes(bi)) };
        const sp = try modelopt.part(ck, try partName(&b, E, E, up));
        if (sp.n * sp.k != 2 * per) return error.SharedExpertWidth;
        try values(L, sp, v);
        const shared = try four(L, v, sp.n, sp.k);
        const label: []const u8 = if (up) "mtp.moe.experts.up" else "mtp.moe.experts.down";
        const ptr = try L.packExperts(label, routed, shared, !up, E + 2, n, kk);
        if (up) out.up = ptr else out.down = ptr;
    }
    return out;
}

/// head_rows: lm_head's bf16 rows for the draft ids, made affine-4 as a head of their own.
fn draftHead4(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const ids = try draft_ids.load(L.gpa, c.vocab);
    defer L.gpa.free(ids);
    const p = try modelopt.part(ck, "lm_head");
    if (p.kind != .bf16 or p.n != c.vocab or p.k != c.hidden) return error.UnexpectedTensor;
    const rows = try L.gpa.alloc(u8, ids.len * p.k * 2);
    defer L.gpa.free(rows);
    try W.gatherRows(L, p.w, ids, rows);
    const v = try L.gpa.alloc(f32, ids.len * p.k);
    defer L.gpa.free(v);
    for (v, std.mem.bytesAsSlice(u16, rows)) |*o, x| o.* = host4.f32of(x);
    L.w.draft_head = .{ .affine4 = try L.tile("draft_head", &.{try four(L, v, ids.len, p.k)}) };
    try W.draftIds(L, ids);
}

test "e4m3 decodes as the fn format: subnormals, normals, the largest and NaN" {
    const t = std.testing;
    try t.expectEqual(@as(f32, 0), e4m3(0));
    try t.expectEqual(std.math.ldexp(@as(f32, 1), -9), e4m3(1));
    try t.expectEqual(@as(f32, 1), e4m3(0x38));
    try t.expectEqual(@as(f32, -1.5), e4m3(0xBC));
    try t.expectEqual(@as(f32, 448), e4m3(0x7E));
    try t.expect(std.math.isNan(e4m3(0x7F)) and std.math.isNan(e4m3(0xFF)));
    try t.expectEqual(@as(u16, 0x3F80), bf16Bits(1));
    try t.expectEqual(@as(u16, 0x3F80), bf16Bits(@bitCast(@as(u32, 0x3F808000)))); // a tie to even
    try t.expectEqual(@as(u16, 0x3F82), bf16Bits(@bitCast(@as(u32, 0x3F818000))));
}
