//! Full GLM-5.3 over `world` ranks (the Python engine's cuda/split.py): which part of each checkpoint tensor a rank
//! holds. EXL3 linears are cut along whole 16x16 tiles and whole 128-wide Hadamard blocks: output-split linears
//! (gate/up, q_b) take tile columns and svh, input-split linears (down, o_proj) take tile rows and suh; q_a, kv_a and
//! the indexer's wq_b are replicated. Phase 1 loads rank 0 of world 4 with these cuts.
const std = @import("std");

pub const Kind = enum { rep, row, dim1 };

pub const had = 128;
pub const tile = 16;

const exl3_parts = [_][]const u8{ "trellis", "suh", "svh", "mul1", "mcg" };

/// The projection an EXL3 tensor name belongs to (split.EXL3_LINEAR), or null for other tensors.
fn exl3Proj(name: []const u8) ?struct { proj: []const u8, part: []const u8 } {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return null;
    const part = name[dot + 1 ..];
    var is_part = false;
    for (exl3_parts) |p| is_part = is_part or std.mem.eql(u8, p, part);
    if (!is_part) return null;
    const stem = name[0..dot];
    const pd = std.mem.lastIndexOfScalar(u8, stem, '.') orelse return null;
    const proj = stem[pd + 1 ..];
    const owner = stem[0..pd];
    // .mlp.experts.<e>.{gate,up,down}_proj | .mlp.shared_experts.X_proj | .mlp.X_proj
    if (std.mem.eql(u8, proj, "gate_proj") or std.mem.eql(u8, proj, "up_proj") or std.mem.eql(u8, proj, "down_proj")) {
        const ok = std.mem.endsWith(u8, owner, ".mlp") or std.mem.endsWith(u8, owner, ".mlp.shared_experts") or isExpert(owner);
        if (!ok) return null;
        return .{ .proj = proj[0 .. proj.len - "_proj".len], .part = part };
    }
    if (std.mem.endsWith(u8, owner, ".self_attn")) {
        for ([_][]const u8{ "q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "o_proj" }) |p| if (std.mem.eql(u8, proj, p)) return .{ .proj = proj, .part = part };
        return null;
    }
    if (std.mem.endsWith(u8, owner, ".self_attn.indexer") and std.mem.eql(u8, proj, "wq_b")) return .{ .proj = proj, .part = part };
    return null;
}

fn isExpert(owner: []const u8) bool {
    // "...mlp.experts.<digits>"
    const d = std.mem.lastIndexOfScalar(u8, owner, '.') orelse return false;
    const num = owner[d + 1 ..];
    if (num.len == 0) return false;
    for (num) |c| if (c < '0' or c > '9') return false;
    return std.mem.endsWith(u8, owner[0..d], ".mlp.experts");
}

fn partRule(part: []const u8, out_split: bool) Kind {
    if (out_split) {
        if (std.mem.eql(u8, part, "trellis")) return .dim1;
        if (std.mem.eql(u8, part, "svh")) return .row;
        return .rep;
    }
    if (std.mem.eql(u8, part, "trellis") or std.mem.eql(u8, part, "suh")) return .row;
    return .rep;
}

/// split.rule: rep | row (leading axis) | dim1 (second axis) for one checkpoint tensor.
pub fn rule(name: []const u8) !Kind {
    return known(name) orelse {
        if (!@import("builtin").is_test) std.log.err("{s}: no split rule (or more than one)", .{name}); // the test expects it
        return error.NoSplitRule;
    };
}

/// `rule` without the error log: null for a tensor no rule covers (the prefetcher leaves those to the mapped path).
pub fn known(name: []const u8) ?Kind {
    if (exl3Proj(name)) |x| {
        if (std.mem.eql(u8, x.proj, "q_a_proj") or std.mem.eql(u8, x.proj, "kv_a_proj_with_mqa") or std.mem.eql(u8, x.proj, "wq_b")) return .rep;
        const out_split = std.mem.eql(u8, x.proj, "gate") or std.mem.eql(u8, x.proj, "up") or std.mem.eql(u8, x.proj, "q_b_proj");
        return partRule(x.part, out_split);
    }
    const row = std.mem.endsWith(u8, name, ".self_attn.kv_b_proj.weight") or std.mem.eql(u8, name, "lm_head.weight");
    const rep_suffix = [_][]const u8{ "_layernorm.weight", ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias", ".self_attn.indexer.wk.weight", ".self_attn.indexer.weights_proj.weight", ".self_attn.indexer.k_norm.weight", ".self_attn.indexer.k_norm.bias", ".eh_proj.weight", ".enorm.weight", ".hnorm.weight", ".shared_head.norm.weight" };
    var rep = std.mem.startsWith(u8, name, "model.embed_tokens.") or std.mem.eql(u8, name, "model.norm.weight");
    for (rep_suffix) |s| rep = rep or std.mem.endsWith(u8, name, s);
    if (row == rep) return null;
    return if (row) .row else .rep;
}

/// split._check_cut: an even cut that keeps whole Hadamard blocks.
pub fn checkCut(name: []const u8, kind: Kind, shape: []const usize, world: usize) !void {
    if (kind == .rep) return;
    const axis: usize = if (kind == .row) 0 else 1;
    if (shape.len <= axis or shape[axis] % world != 0) return error.UnevenCut;
    if (std.mem.endsWith(u8, name, ".trellis")) {
        if ((shape[axis] / world) % (had / tile) != 0) return error.CutStraddlesHadamard;
    } else if (std.mem.endsWith(u8, name, ".suh") or std.mem.endsWith(u8, name, ".svh")) {
        if ((shape[0] / world) % had != 0) return error.CutStraddlesHadamard;
    }
}

/// Rank `rank`'s part of a tensor's bytes into `out` (host): the whole tensor, a row range, or second-axis columns.
pub fn cutBytes(out: []u8, raw: []const u8, shape: []const usize, itemsize: usize, kind: Kind, rank: usize, world: usize) !void {
    switch (kind) {
        .rep => {
            if (out.len != raw.len) return error.Invalid;
            @memcpy(out, raw);
        },
        .row => {
            const per = raw.len / shape[0];
            const n = shape[0] / world;
            if (out.len != n * per) return error.Invalid;
            @memcpy(out, raw[rank * n * per ..][0 .. n * per]);
        },
        .dim1 => {
            var inner: usize = itemsize;
            for (shape[2..]) |d| inner *= d;
            const n = shape[1] / world;
            const row_bytes = shape[1] * inner;
            const take = n * inner;
            if (out.len != shape[0] * take) return error.Invalid;
            for (0..shape[0]) |r| @memcpy(out[r * take ..][0..take], raw[r * row_bytes + rank * take ..][0..take]);
        },
    }
}

test "split rules of GLM-5.3's tensors" {
    const T = struct { n: []const u8, k: Kind };
    const cases = [_]T{
        .{ .n = "model.layers.3.mlp.experts.17.gate_proj.trellis", .k = .dim1 },
        .{ .n = "model.layers.3.mlp.experts.17.gate_proj.svh", .k = .row },
        .{ .n = "model.layers.3.mlp.experts.17.gate_proj.suh", .k = .rep },
        .{ .n = "model.layers.3.mlp.experts.17.down_proj.trellis", .k = .row },
        .{ .n = "model.layers.3.mlp.experts.17.down_proj.suh", .k = .row },
        .{ .n = "model.layers.3.mlp.experts.17.down_proj.svh", .k = .rep },
        .{ .n = "model.layers.3.mlp.shared_experts.up_proj.mul1", .k = .rep },
        .{ .n = "model.layers.0.mlp.down_proj.trellis", .k = .row },
        .{ .n = "model.layers.0.self_attn.q_a_proj.trellis", .k = .rep },
        .{ .n = "model.layers.0.self_attn.kv_a_proj_with_mqa.svh", .k = .rep },
        .{ .n = "model.layers.0.self_attn.q_b_proj.trellis", .k = .dim1 },
        .{ .n = "model.layers.0.self_attn.o_proj.suh", .k = .row },
        .{ .n = "model.layers.0.self_attn.indexer.wq_b.trellis", .k = .rep },
        .{ .n = "model.layers.0.self_attn.kv_b_proj.weight", .k = .row },
        .{ .n = "lm_head.weight", .k = .row },
        .{ .n = "model.embed_tokens.weight", .k = .rep },
        .{ .n = "model.layers.0.input_layernorm.weight", .k = .rep },
        .{ .n = "model.layers.3.mlp.gate.e_score_correction_bias", .k = .rep },
        .{ .n = "model.layers.0.self_attn.indexer.k_norm.bias", .k = .rep },
    };
    for (cases) |c| {
        const got = try rule(c.n);
        if (got != c.k) {
            std.debug.print("{s}: {t}, want {t}\n", .{ c.n, got, c.k });
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expectError(error.NoSplitRule, rule("model.layers.0.something.weight"));
}

test "cuts: rows and second-axis columns" {
    // [2, 4, 2] bytes, dim1 over 2 ranks: rank 1 takes columns 2..3 of each row
    var raw: [16]u8 = undefined;
    for (&raw, 0..) |*b, i| b.* = @intCast(i);
    var out: [8]u8 = undefined;
    try cutBytes(&out, &raw, &.{ 2, 4, 2 }, 1, .dim1, 1, 2);
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6, 7, 12, 13, 14, 15 }, &out);
    try cutBytes(&out, &raw, &.{ 2, 4, 2 }, 1, .row, 1, 2);
    try std.testing.expectEqualSlices(u8, raw[8..16], &out);
    try checkCut("x.trellis", .dim1, &.{ 384, 1024, 80 }, 4);
    try std.testing.expectError(error.CutStraddlesHadamard, checkCut("x.trellis", .dim1, &.{ 384, 40, 80 }, 4));
}
