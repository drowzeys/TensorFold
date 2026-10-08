//! Phase 4: prompt-lookup ("copy") drafts, cuda/copies.py: when the reply's last `match` tokens (the pending one
//! included) occurred before - in the prompt or earlier in the reply - the tokens that followed that occurrence are
//! proposed as the round's drafts, ahead of (instead of) the MTP head's or a drafter's. The same choice of occurrence
//! (the latest one with k tokens after it, else the earliest, which has the most) and a hashed index (the prompt's
//! n-grams sorted once, the reply's in a map as they arrive), so a round without a match costs a lookup.
//!
//! Exact: drafts only propose - the target verifies every row and keeps its own samples, so replies equal serial ones
//! bit for bit. Every rank holds the same prompt and samples the same tokens, so every rank proposes the same drafts
//! with no exchange; TF_GLM53_COPY_* must agree across ranks (the startup digest).
//! TF_GLM53_COPY_DRAFTS (default 1): 0 off. TF_GLM53_COPY_MIN (default 8, 2..64): the match length.
//! TF_GLM53_COPY_MAX (default 15, 1..DECODE_ROWS - 1): drafts a copied round proposes; the runner verifies copies in
//! windows of 8 and COPY_MAX + 1 rows besides the drafters' and pads a copy up to the next of them.
const std = @import("std");

pub const match_default: usize = 8;
pub const most_default: usize = 15;
pub const widest: usize = 32; // fused.DECODE_ROWS
const B: u64 = 0x9E3779B97F4A7C15; // the rolling hash's base (odd), mod 2**64

/// copies.settings: (on, match, most) from the TF_GLM53_COPY_* strings (null: unset).
pub const Settings = struct {
    on: bool = true,
    match: usize = match_default,
    most: usize = most_default,

    pub fn parse(on: ?[]const u8, min: ?[]const u8, max: ?[]const u8) !Settings {
        var s: Settings = .{};
        if (on) |v| {
            const t = std.mem.trim(u8, v, " ");
            if (std.mem.eql(u8, t, "0")) return .{ .on = false, .match = 0, .most = 0 };
            if (!(t.len == 0 or std.mem.eql(u8, t, "1"))) return error.BadSetting;
        }
        s.match = try number(min, match_default, 2, 64);
        s.most = try number(max, most_default, 1, widest - 1);
        return s;
    }

    fn number(text: ?[]const u8, default: usize, low: usize, high: usize) !usize {
        const t = std.mem.trim(u8, text orelse return default, " ");
        if (t.len == 0) return default;
        const v = std.fmt.parseInt(usize, t, 10) catch return error.BadSetting;
        if (v < low or v > high) return error.BadSetting;
        return v;
    }
};

/// copies._hashes' n-gram hash of `toks` (Horner over uint64, wrapping).
fn hashOf(toks: []const u32) u64 {
    var h: u64 = 0;
    for (toks) |t| h = h *% B +% t;
    return h;
}

fn powB(n: usize) u64 {
    var x: u64 = 1;
    for (0..n) |_| x *%= B;
    return x;
}

/// copies.CopyIndex: one request's context (prompt, then the reply's tokens with the pending one) and its proposals.
pub const CopyIndex = struct {
    gpa: std.mem.Allocator,
    match: usize,
    most: usize,
    buf: std.ArrayList(u32) = .empty,
    /// the prompt's n-gram starts sorted by hash (equal hashes: ascending starts) and their hashes
    order: []usize = &.{},
    keys: []u64 = &.{},
    /// hash -> ascending starts of the reply's n-grams
    tail: std.AutoHashMapUnmanaged(u64, std.ArrayList(usize)) = .empty,
    /// the hash of the context's last `match` tokens
    h: ?u64 = null,
    lead: u64,

    pub fn init(gpa: std.mem.Allocator, prompt: []const u32, match: usize, most: usize) !CopyIndex {
        if (match < 1 or most < 1) return error.Invalid;
        var c: CopyIndex = .{ .gpa = gpa, .match = match, .most = most, .lead = powB(match - 1) };
        errdefer c.deinit();
        try c.buf.ensureTotalCapacity(gpa, @max(1024, 2 * prompt.len));
        c.buf.appendSliceAssumeCapacity(prompt);
        if (prompt.len >= match) {
            const m = prompt.len - match + 1;
            c.order = try gpa.alloc(usize, m);
            c.keys = try gpa.alloc(u64, m);
            const hs = try gpa.alloc(u64, m);
            defer gpa.free(hs);
            // rolling as _hashes computes them (each n-gram's Horner sum)
            hs[0] = hashOf(prompt[0..match]);
            for (1..m) |i| hs[i] = (hs[i - 1] -% @as(u64, prompt[i - 1]) *% c.lead) *% B +% prompt[i + match - 1];
            for (c.order, 0..) |*o, i| o.* = i;
            std.sort.pdq(usize, c.order, hs, struct {
                fn lt(h: []u64, a: usize, b: usize) bool {
                    return h[a] < h[b] or (h[a] == h[b] and a < b);
                }
            }.lt);
            for (c.order, c.keys) |o, *k| k.* = hs[o];
            c.h = hs[m - 1];
        }
        return c;
    }

    pub fn deinit(c: *CopyIndex) void {
        var it = c.tail.valueIterator();
        while (it.next()) |l| l.deinit(c.gpa);
        c.tail.deinit(c.gpa);
        c.gpa.free(c.order);
        c.gpa.free(c.keys);
        c.buf.deinit(c.gpa);
    }

    pub fn len(c: *const CopyIndex) usize {
        return c.buf.items.len;
    }

    /// CopyIndex.extend: tokens that joined the context (the reply's; the last is the next round's pending token).
    pub fn extend(c: *CopyIndex, toks: []const u32) !void {
        const n = c.match;
        for (toks) |t| {
            try c.buf.append(c.gpa, t);
            const L = c.buf.items.len;
            if (L < n) continue;
            const s = L - n; // the n-gram that ends with this token
            const items = c.buf.items;
            const h = if (c.h == null or s == 0) hashOf(items[s..L]) else (c.h.? -% @as(u64, items[s - 1]) *% c.lead) *% B +% items[L - 1];
            c.h = h;
            const gop = try c.tail.getOrPut(c.gpa, h);
            if (!gop.found_existing) gop.value_ptr.* = .empty;
            try gop.value_ptr.append(c.gpa, s);
        }
    }

    fn same(c: *const CopyIndex, s: usize) bool {
        const L = c.buf.items.len;
        const n = c.match;
        return std.mem.eql(u32, c.buf.items[s .. s + n], c.buf.items[L - n .. L]);
    }

    /// The prompt's starts with hash `h` (ascending): keys' equal range.
    fn head(c: *const CopyIndex, h: u64) []const usize {
        return c.order[bound(c.keys, h, false)..bound(c.keys, h, true)];
    }

    /// numpy searchsorted over ascending keys: side "left" (first >= key) or "right" (first > key).
    fn bound(keys: []const u64, key: u64, right: bool) usize {
        var lo: usize = 0;
        var hi: usize = keys.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const below = if (right) keys[mid] <= key else keys[mid] < key;
            if (below) lo = mid + 1 else hi = mid;
        }
        return lo;
    }

    /// CopyIndex.find: the start of the earlier occurrence of the context's last `match` tokens to copy `k` drafts
    /// from - the latest with k tokens after it, else the earliest (the most after it); null when there is none.
    pub fn find(c: *const CopyIndex, k: usize) ?usize {
        const L = c.buf.items.len;
        const n = c.match;
        if (L <= n) return null;
        const h = c.h orelse return null;
        const hd = c.head(h);
        const tl: []const usize = if (c.tail.get(h)) |l| l.items else &.{};
        if (hd.len == 0 and tl.len == 0) return null;
        const last: i64 = @as(i64, @intCast(L)) - @as(i64, @intCast(n)) - @as(i64, @intCast(k)); // latest start leaving k after its match
        // the reply's starts at or below `last`, latest first
        var i: usize = tl.len;
        while (i > 0) {
            i -= 1;
            if (@as(i64, @intCast(tl[i])) > last) continue;
            if (c.same(tl[i])) return tl[i];
        }
        i = hd.len;
        while (i > 0) {
            i -= 1;
            if (@as(i64, @intCast(hd[i])) > last) continue;
            if (c.same(hd[i])) return hd[i];
        }
        for ([_][]const usize{ hd, tl }) |part| for (part) |s| {
            if (s >= L - n) return null; // the context's own last n tokens (and later)
            if (c.same(s)) return s;
        };
        return null;
    }

    /// CopyIndex.propose: up to min(most, room) drafts - what followed the chosen occurrence; empty when none.
    pub fn propose(c: *const CopyIndex, room: usize, out: []u32) []u32 {
        const k = @min(c.most, room);
        if (k < 1) return out[0..0];
        const s = c.find(k) orelse return out[0..0];
        const a = s + c.match;
        const e = @min(a + k, c.buf.items.len);
        const n = @min(e - a, out.len);
        @memcpy(out[0..n], c.buf.items[a .. a + n]);
        return out[0..n];
    }
};

/// copies.reference: MiaAI-Lab 0007's scan (every start compared, no index) - what propose(k) must return.
pub fn reference(ctx: []const u32, match: usize, k: usize, out: []u32) []u32 {
    const L = ctx.len;
    const n = match;
    if (L <= n or k < 1) return out[0..0];
    const q = ctx[L - n ..];
    var first: ?usize = null;
    var full: ?usize = null;
    var s: usize = 0;
    while (s < L - n) : (s += 1) {
        if (!std.mem.eql(u32, ctx[s .. s + n], q)) continue;
        if (first == null) first = s;
        if (s + k <= L - n) full = s;
    }
    const pick = full orelse first orelse return out[0..0];
    const e = @min(pick + n + k, L);
    @memcpy(out[0 .. e - pick - n], ctx[pick + n .. e]);
    return out[0 .. e - pick - n];
}

test "copy settings" {
    const d = try Settings.parse(null, null, null);
    try std.testing.expect(d.on and d.match == 8 and d.most == 15);
    try std.testing.expect(!(try Settings.parse("0", null, null)).on);
    try std.testing.expectError(error.BadSetting, Settings.parse("2", null, null));
    try std.testing.expectError(error.BadSetting, Settings.parse(null, "1", null));
    try std.testing.expectError(error.BadSetting, Settings.parse(null, null, "32"));
}

test "copy index proposes what the reference scan proposes" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var trial: usize = 0;
    while (trial < 60) : (trial += 1) {
        // a small alphabet so n-grams repeat; match 2..4
        const match = 2 + trial % 3;
        const alpha: u32 = 3 + @as(u32, @intCast(trial % 4));
        var prompt: [80]u32 = undefined;
        const plen = 10 + trial % 60;
        for (prompt[0..plen]) |*t| t.* = r.uintLessThan(u32, alpha);
        var ci = try CopyIndex.init(gpa, prompt[0..plen], match, 6);
        defer ci.deinit();
        var ctx: std.ArrayList(u32) = .empty;
        defer ctx.deinit(gpa);
        try ctx.appendSlice(gpa, prompt[0..plen]);
        var step: usize = 0;
        while (step < 40) : (step += 1) {
            for ([_]usize{ 1, 3, 6, 9 }) |room| {
                var a: [16]u32 = undefined;
                var b: [16]u32 = undefined;
                const got = ci.propose(room, &a);
                const want = reference(ctx.items, match, @min(6, room), &b);
                try std.testing.expectEqualSlices(u32, want, got);
            }
            const t = r.uintLessThan(u32, alpha);
            try ci.extend(&.{t});
            try ctx.append(gpa, t);
        }
    }
}

test "copy index: a quoted run is proposed, a prompt shorter than the match is fine" {
    const gpa = std.testing.allocator;
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var ci = try CopyIndex.init(gpa, &prompt, 3, 4);
    defer ci.deinit();
    try ci.extend(&.{ 50, 2, 3, 4 });
    var out: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 7, 8 }, ci.propose(15, &out));
    var short = try CopyIndex.init(gpa, &.{ 1, 2 }, 3, 4);
    defer short.deinit();
    try short.extend(&.{ 3, 1, 2, 3 });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, short.propose(4, &out));
}
