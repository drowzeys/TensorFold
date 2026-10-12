//! Routed-expert pruning (lossy, off unless TF_GLM53_EXPERT_TOPP is set): after the router's top-k a row drops the
//! routed experts that carry little of its weight, so a decode window reads fewer expert bytes. The rule is the one
//! Jay Leaton's DeepSeek-V4.1-Flash family of TensorFold uses (its prune.py, MIT):
//!
//! A row's routed weights w_0..w_{K-1} (fp32, slot = pick order) give fractions f_i = w_i / sum w, the sum in slot
//! order. By f descending (ties: the lower slot first) a slot is kept while the fraction mass before it is below
//! `topp`, and the first `min_k` slots always. A row that keeps every slot is untouched, bit for bit. Otherwise a
//! dropped slot gets weight 0 and the pick `drop` (the expert count: every expert kernel skips a pick outside
//! [0, E), so nothing of it is read), and a kept slot keeps its router weight (`orig`: the layer's routed output
//! shrinks by the dropped share) or gets scale * (w_i / the kept weights' sum in slot order) (`kept`).
//!
//! A row's result depends on its own picks and weights only, in a fixed order, so batched rows equal lone rows and
//! every rank (each routes its own copy of the rows) prunes alike. `row` is the kernel's arithmetic on the host
//! (glm53_expert_prune, glm53_decode.cu), checked against tools/glm53/prune_ref.py's numpy cases.
//!
//! Settings: TF_GLM53_EXPERT_TOPP (0 < p < 1; unset or 0: off), TF_GLM53_EXPERT_MIN_K (default 3),
//! TF_GLM53_EXPERT_RENORM (orig | kept, default orig), TF_GLM53_EXPERT_PRUNE_PROMPT (default 0: only windows the
//! decode kernels run; 1: prompt chunks too, which changes the cached keys and values of a prompt).
const std = @import("std");

pub const max_k = 16;

pub const Renorm = enum(u32) { orig = 0, kept = 1 };

pub const Rule = struct {
    topp: f32 = 0,
    min_k: u32 = 3,
    renorm: Renorm = .orig,
    /// prompt chunks prune too (else only the windows the decode kernels run)
    prompt: bool = false,

    pub fn on(r: Rule) bool {
        return r.topp > 0 and r.topp < 1;
    }

    /// The rule for a layer of `top_k` routed slots, or null when it cannot drop a slot there.
    pub fn of(r: Rule, top_k: usize) ?Rule {
        return if (r.on() and r.min_k < top_k and top_k <= max_k) r else null;
    }

    /// The settings' strings (null: unset). A value out of range is an error, not a silent default.
    pub fn parse(topp: ?[]const u8, min_k: ?[]const u8, renorm: ?[]const u8, prompt: ?[]const u8) !Rule {
        var r: Rule = .{};
        if (topp) |v| {
            const x = std.fmt.parseFloat(f32, v) catch return error.BadSetting;
            if (!(x >= 0 and x < 1)) return error.BadSetting;
            r.topp = x;
        }
        if (min_k) |v| {
            r.min_k = std.fmt.parseInt(u32, v, 10) catch return error.BadSetting;
            if (r.min_k == 0) return error.BadSetting;
        }
        if (renorm) |v| r.renorm = std.meta.stringToEnum(Renorm, v) orelse return error.BadSetting;
        if (prompt) |v| {
            if (std.mem.eql(u8, v, "1")) r.prompt = true else if (!std.mem.eql(u8, v, "0")) return error.BadSetting;
        }
        return r;
    }

    /// What the ranks compare before serving (0: off).
    pub fn digest(r: Rule) u64 {
        if (!r.on()) return 0;
        const words = [_]u32{ @bitCast(r.topp), r.min_k, @intFromEnum(r.renorm), @intFromBool(r.prompt) };
        return std.hash.Wyhash.hash(0x9e11, std.mem.asBytes(&words));
    }
};

/// One row in place: w fp32 [K] and pick int32 [K]; returns the slots kept. The kernel's statements in its order.
pub fn row(w: []f32, pick: []i32, topp: f32, min_k: u32, renorm: Renorm, scale: f32, drop: i32) usize {
    const K = w.len;
    std.debug.assert(K <= max_k and pick.len == K);
    var total: f32 = 0;
    for (w) |x| total += x;
    var f: [max_k]f32 = undefined;
    for (w, 0..) |x, i| f[i] = x / total;
    var rank: [max_k]u32 = undefined;
    for (0..K) |i| {
        var n: u32 = 0;
        for (0..K) |j| n += @intFromBool(f[j] > f[i] or (f[j] == f[i] and j < i));
        rank[i] = n;
    }
    var excl: [max_k]f32 = undefined;
    var acc: f32 = 0;
    for (0..K) |r| {
        var fr: f32 = 0;
        for (0..K) |i| {
            if (rank[i] == r) {
                excl[i] = acc;
                fr = f[i];
            }
        }
        acc += fr;
    }
    var keep: [max_k]bool = undefined;
    var kept: usize = 0;
    for (0..K) |i| {
        keep[i] = excl[i] < topp or rank[i] < min_k;
        kept += @intFromBool(keep[i]);
    }
    if (kept == K) return K;
    var ks: f32 = 0;
    for (0..K) |i| ks += if (keep[i]) w[i] else 0;
    for (0..K) |i| {
        if (keep[i]) {
            if (renorm == .kept) w[i] = scale * (w[i] / ks);
        } else {
            w[i] = 0;
            pick[i] = drop;
        }
    }
    return kept;
}

const cases = @embedFile("prune_cases.bin");

test "the rule equals numpy's on prune_ref.py's cases, bit for bit" {
    try std.testing.expectEqualSlices(u8, "GLM53PR1", cases[0..8]);
    const n = std.mem.readInt(u32, cases[8..12], .little);
    const K: usize = std.mem.readInt(u32, cases[12..16], .little);
    try std.testing.expect(n >= 200 and K == 8);
    var at: usize = 16;
    var dropped: usize = 0;
    for (0..n) |_| {
        const topp: f32 = @bitCast(std.mem.readInt(u32, cases[at..][0..4], .little));
        const min_k = std.mem.readInt(u32, cases[at + 4 ..][0..4], .little);
        const renorm: Renorm = @enumFromInt(std.mem.readInt(u32, cases[at + 8 ..][0..4], .little));
        const scale: f32 = @bitCast(std.mem.readInt(u32, cases[at + 12 ..][0..4], .little));
        const drop = std.mem.readInt(i32, cases[at + 16 ..][0..4], .little);
        at += 20;
        var w: [max_k]f32 = undefined;
        var p: [max_k]i32 = undefined;
        for (0..K) |i| {
            w[i] = @bitCast(std.mem.readInt(u32, cases[at + 4 * i ..][0..4], .little));
            p[i] = std.mem.readInt(i32, cases[at + 4 * (K + i) ..][0..4], .little);
        }
        const kept = row(w[0..K], p[0..K], topp, min_k, renorm, scale, drop);
        var gone: usize = 0;
        for (0..K) |i| {
            const ew = std.mem.readInt(u32, cases[at + 4 * (2 * K + i) ..][0..4], .little);
            const ep = std.mem.readInt(i32, cases[at + 4 * (3 * K + i) ..][0..4], .little);
            try std.testing.expectEqual(ew, @as(u32, @bitCast(w[i])));
            try std.testing.expectEqual(ep, p[i]);
            gone += @intFromBool(ep == drop);
        }
        try std.testing.expectEqual(K - gone, kept);
        try std.testing.expect(kept >= @min(min_k, K));
        dropped += gone;
        at += 16 * K;
    }
    try std.testing.expectEqual(cases.len, at);
    try std.testing.expect(dropped > n);
}

test "a row that keeps every slot is untouched, and min_k holds on a peaked row" {
    var w = [_]f32{ 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125 };
    var p = [_]i32{ 7, 3, 250, 11, 0, 90, 41, 5 };
    const before = w;
    // equal fractions of 1/8: the mass before the last slot is 0.875 < 0.9
    try std.testing.expectEqual(@as(usize, 8), row(&w, &p, 0.9, 3, .kept, 2.5, 256));
    try std.testing.expectEqualSlices(f32, &before, &w);
    // ... and 0.85 keeps 7: ties go to the lower slot, so the last slot is the one dropped
    try std.testing.expectEqual(@as(usize, 7), row(&w, &p, 0.85, 3, .orig, 2.5, 256));
    try std.testing.expectEqual(@as(i32, 256), p[7]);
    try std.testing.expectEqual(@as(f32, 0), w[7]);
    try std.testing.expectEqual(before[0], w[0]);
    var w2 = [_]f32{ 2.4, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01 };
    var p2 = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try std.testing.expectEqual(@as(usize, 3), row(&w2, &p2, 0.5, 3, .kept, 2.5, 256));
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 256, 256, 256, 256, 256 }, &p2);
    var sum: f32 = 0;
    for (w2) |x| sum += x;
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), sum, 1e-5);
}

test "settings" {
    try std.testing.expect(!(try Rule.parse(null, null, null, null)).on());
    try std.testing.expect(!(try Rule.parse("0", null, null, null)).on());
    const r = try Rule.parse("0.85", "3", "kept", "1");
    try std.testing.expect(r.on() and r.renorm == .kept and r.prompt and r.min_k == 3);
    try std.testing.expect(r.of(8) != null and r.of(3) == null and r.of(17) == null);
    try std.testing.expect(r.digest() != 0 and r.digest() != (try Rule.parse("0.9", "3", "kept", "1")).digest());
    try std.testing.expectEqual(@as(u64, 0), (Rule{}).digest());
    try std.testing.expectError(error.BadSetting, Rule.parse("1", null, null, null));
    try std.testing.expectError(error.BadSetting, Rule.parse("0.85", "0", null, null));
    try std.testing.expectError(error.BadSetting, Rule.parse("0.85", null, "half", null));
    try std.testing.expectError(error.BadSetting, Rule.parse("0.85", null, null, "yes"));
}
