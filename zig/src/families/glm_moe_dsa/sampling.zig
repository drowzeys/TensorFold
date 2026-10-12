//! The Python engine's keyed sampling on the host, bit for bit (engine/exact_sampling.py + cuda/sampling.py as the
//! GLM-5.3 runner uses them): seed_for(prompt ids), uniform(seed, position, id) (splitmix64), choose_rows' float64
//! rule (temperature, top_k, top_p, min_p, Gumbel race keyed by position and token id).
//!
//! Which candidates a draw sees (runner.py / engine.py, one stream):
//!   * top_k > 0 (the server's default 20): every rank sends its own vocabulary share's top_k + MARGIN logits (value
//!     desc, lower id on ties: torch.topk's set) and choose_rows takes the window's top_k of all of them
//!     (Runner._sample_local, head "local"); the first token after the prompt (Glm53Engine._sample over the gathered
//!     logits, torch.topk(top_k + MARGIN)) sees the same top_k.
//!   * top_k == 0 (top_p alone): Runner._sampled_pick takes the "full" head and Glm53Engine._sample's rule for every
//!     row: torch.topk(256 + MARGIN) of the whole vocabulary, then choose_rows with k = that width. Every rank's top
//!     264 of its share holds the vocabulary's top 264, so the merged, ordered union cut to 264 is torch's set.
//! Either way `choose` sorts the gathered candidates (numpy lexsort: value desc, id asc), keeps the first
//! `candidates()` of them and runs choose_rows on those - the bits of both Python paths.
//!
//! float64 steps as numpy runs them on aarch64: division, subtraction and the cumulative sum are IEEE; exp and log are
//! glibc's (numpy's scalar loops call libm there; this binary calls the same libm in the same image); a row's sum is
//! numpy's pairwise sum (8 accumulators up to 128 values, halves above).
const std = @import("std");

/// exact_sampling.MARGIN: candidates beyond top_k read from the GPU so ties at the cut resolve by id.
pub const margin: usize = 8;
/// Glm53Engine._sample's candidate count for top_k 0 (before MARGIN).
pub const top_k_off_width: usize = 256;
/// The widest candidate list `choose` handles (256 + MARGIN).
pub const max_candidates: usize = top_k_off_width + margin;

pub const Sampling = struct {
    seed: u64,
    temperature: f64 = 1.0,
    top_k: usize = 20,
    top_p: f64 = 0.95,
    min_p: f64 = 0.0,

    /// The candidates each rank sends a row (and the width choose_rows sees): top_k + MARGIN, or 256 + MARGIN.
    pub fn candidates(s: Sampling) usize {
        return (if (s.top_k > 0) s.top_k else top_k_off_width) + margin;
    }
};

const libm = struct {
    const exp = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "exp" });
    const log = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "log" });
};

/// exact_sampling.seed_for: sha256 of "id,id,...|salt", its first 8 bytes little-endian, top bit cleared.
pub fn seedFor(ids: []const u32, salt: i64) u64 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [32]u8 = undefined;
    for (ids, 0..) |t, i| {
        if (i > 0) h.update(",");
        h.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
    }
    h.update(std.fmt.bufPrint(&buf, "|{d}", .{salt}) catch unreachable);
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.mem.readInt(u64, d[0..8], .little) & ((@as(u64, 1) << 63) - 1);
}

fn mix(x0: u64) u64 {
    var x = x0;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// exact_sampling.uniform: a (0, 1) double from splitmix64 of (seed, position, token id).
pub fn uniform(seed: u64, position: u64, id: u64) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

/// numpy's DOUBLE_pairwise_sum (add.reduce over a contiguous row, identity 0.0 first).
pub fn pairwiseSum(a: []const f64) f64 {
    const n = a.len;
    if (n < 8) {
        var res: f64 = 0.0;
        for (a) |v| res += v;
        return res;
    }
    if (n <= 128) {
        var r: [8]f64 = a[0..8].*;
        var i: usize = 8;
        while (i < n - (n % 8)) : (i += 8) {
            for (0..8) |j| r[j] += a[i + j];
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < n) : (i += 1) res += a[i];
        return res;
    }
    var n2 = n / 2;
    n2 -= n2 % 8;
    return pairwiseSum(a[0..n2]) + pairwiseSum(a[n2..]);
}

/// A candidate logit and its global token id.
pub const Cand = struct { v: f32, id: i64 };

/// numpy lexsort((ids, -values)): ascending -value (NaN last), then ascending id.
fn lexLess(_: void, a: Cand, b: Cand) bool {
    const na = -a.v;
    const nb = -b.v;
    const an = std.math.isNan(na);
    const bn = std.math.isNan(nb);
    if (an != bn) return bn;
    if (!an and na != nb) return na < nb;
    return a.id < b.id;
}

/// torch.topk's choice of a row's best: NaN above every number, then the larger value, ties to the lower index.
pub fn topBetter(va: f32, ia: usize, vb: f32, ib: usize) bool {
    const an = std.math.isNan(va);
    const bn = std.math.isNan(vb);
    if (an != bn) return an;
    if (an) return ia < ib;
    return va > vb or (va == vb and ia < ib);
}

/// A row's `want` best logits (topBetter order, best first) into vals / ids (global id = index + off).
pub fn selectTop(row: []const f32, off: usize, want: usize, vals: []f32, ids: []i32) void {
    std.debug.assert(want > 0 and want <= vals.len and want <= ids.len and want <= row.len);
    var n: usize = 0;
    var idx: [max_candidates]usize = undefined;
    std.debug.assert(want <= max_candidates);
    for (row, 0..) |v, i| {
        if (n == want) {
            // worse than the current last: skip (the common case)
            if (!topBetter(v, i, vals[n - 1], idx[n - 1])) continue;
            n -= 1;
        }
        // insertion from the end
        var j = n;
        while (j > 0 and topBetter(v, i, vals[j - 1], idx[j - 1])) : (j -= 1) {
            vals[j] = vals[j - 1];
            idx[j] = idx[j - 1];
        }
        vals[j] = v;
        idx[j] = i;
        n += 1;
    }
    for (0..want) |j| ids[j] = @intCast(idx[j] + off);
}

/// choose_rows for one row: `cands` (any order, every rank's; reordered in place) at `position` -> the token.
pub fn choose(cands: []Cand, position: u64, s: Sampling) u32 {
    return chooseKeep(cands, position, s, null);
}

/// `choose`, also reporting the nucleus size (choose_rows' `keep`; the width when top_p is off) for the checks.
pub fn chooseKeep(cands: []Cand, position: u64, s: Sampling, keep_out: ?*usize) u32 {
    std.sort.pdq(Cand, cands, {}, lexLess);
    const width = @min(cands.len, s.candidates());
    const want: usize = if (s.top_k > 0) s.top_k else width;
    const k = @max(1, @min(want, width));
    const temp = @max(s.temperature, 1e-6);
    var scaled: [max_candidates]f64 = undefined;
    var score: [max_candidates]f64 = undefined;
    for (0..k) |i| {
        scaled[i] = @as(f64, cands[i].v) / temp;
        const u = uniform(s.seed, position, @bitCast(cands[i].id));
        score[i] = scaled[i] - libm.log(-libm.log(u));
    }
    var keep: usize = k;
    if (0.0 < s.top_p and s.top_p < 1.0) {
        var mx = scaled[0];
        for (scaled[1..k]) |v| mx = @max(mx, v);
        var e: [max_candidates]f64 = undefined;
        for (0..k) |i| e[i] = libm.exp(scaled[i] - mx);
        const total = pairwiseSum(e[0..k]);
        var cum: f64 = 0.0;
        var below: usize = 0;
        for (0..k) |i| {
            cum += e[i] / total;
            if (cum < s.top_p) below += 1;
        }
        keep = below + 1;
        if (keep < k) {
            for (keep..k) |i| score[i] = -std.math.inf(f64);
        }
    }
    if (s.min_p > 0.0) {
        const floor = scaled[0] + libm.log(s.min_p);
        for (0..k) |i| {
            if (scaled[i] < floor) score[i] = -std.math.inf(f64);
        }
    }
    // numpy argmax: the first maximum (a NaN wins at once)
    var best: usize = 0;
    for (1..k) |i| {
        if (std.math.isNan(score[best])) break;
        if (std.math.isNan(score[i]) or score[i] > score[best]) best = i;
    }
    if (keep_out) |p| p.* = @min(keep, k);
    return @intCast(cands[best].id);
}

test "uniform stays inside (0, 1)" {
    var lo: f64 = 1;
    var hi: f64 = 0;
    for (0..1000) |i| {
        const u = uniform(12345, i, i * 7919);
        lo = @min(lo, u);
        hi = @max(hi, u);
    }
    try std.testing.expect(lo > 0 and hi < 1);
    try std.testing.expectEqual(uniform(1, 2, 3), uniform(1, 2, 3));
    try std.testing.expect(uniform(1, 2, 3) != uniform(1, 3, 2));
}

test "pairwise sum: numpy's association" {
    var a: [20]f64 = undefined;
    for (&a, 0..) |*v, i| v.* = 1.0 / @as(f64, @floatFromInt(i + 3));
    var r: [8]f64 = a[0..8].*;
    for (0..8) |j| r[j] += a[8 + j];
    var want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (16..20) |i| want += a[i];
    try std.testing.expectEqual(want, pairwiseSum(&a));
    var b: [300]f64 = undefined;
    for (&b, 0..) |*v, i| v.* = @as(f64, @floatFromInt(i % 17)) * 0.1;
    try std.testing.expectEqual(pairwiseSum(b[0..144]) + pairwiseSum(b[144..]), pairwiseSum(&b));
    // runtime f64 adds (literal arithmetic would be comptime_float, exact: 0.6); numpy: 0.6000000000000001
    var x = [_]f64{ 0.1, 0.2, 0.3 };
    _ = &x;
    try std.testing.expectEqual((x[0] + x[1]) + x[2], pairwiseSum(&x));
    try std.testing.expectEqual(@as(f64, 0.6000000000000001), pairwiseSum(&x));
}

test "seed_for is sha256 of the decimal ids" {
    // sha256("1,2,3|0") computed by the same rule; the cluster check compares against Python's seed_for
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("1,2,3|0", &d, .{});
    const want = std.mem.readInt(u64, d[0..8], .little) & ((@as(u64, 1) << 63) - 1);
    try std.testing.expectEqual(want, seedFor(&.{ 1, 2, 3 }, 0));
}

test "selectTop: value order, lower index on ties, NaN first" {
    const row = [_]f32{ 1, 5, 3, 5, std.math.nan(f32), 2, 5 };
    var v: [4]f32 = undefined;
    var id: [4]i32 = undefined;
    selectTop(&row, 100, 4, &v, &id);
    try std.testing.expectEqualSlices(i32, &.{ 104, 101, 103, 106 }, &id);
    try std.testing.expectEqual(@as(f32, 5), v[1]);
}

test "choose: the top_k cut, the top_p nucleus" {
    var c = [_]Cand{ .{ .v = 1, .id = 7 }, .{ .v = 9, .id = 3 }, .{ .v = 8, .id = 2 }, .{ .v = -4, .id = 11 } };
    try std.testing.expectEqual(@as(u32, 3), choose(&c, 10, .{ .seed = 42, .top_k = 1, .top_p = 1.0 }));
    var keep: usize = 0;
    var d = [_]Cand{ .{ .v = 10, .id = 1 }, .{ .v = 0, .id = 2 }, .{ .v = 0, .id = 3 } };
    _ = chooseKeep(&d, 5, .{ .seed = 1, .top_k = 3, .top_p = 0.95 }, &keep);
    try std.testing.expectEqual(@as(usize, 1), keep);
}
