//! GLM-5.3's decode windows on two more streams (Phase 3a speed), as the served Python engine runs them:
//!
//! `Side` (TF_GLM53_SIDE, default 1 = "af"; fused.py `_SideWindow` / `_fork` / `_join`): inside a decode window
//! (fused.compute / mtp_compute up to min(b.small, MAX_ROWS) rows, never a prompt chunk's halves) independent work
//! queues on a second stream. "a": the key path (kv_a out of q_a's launch, the latent / rope key write, the indexer's
//! wk and index key write) beside the query path, joined before select (or before the attention kernels when no row
//! selects); "w": the same with kv_a left in q_a's launch; "f": the shared expert beside the router and the routed
//! experts, joined right before the routed down + combine launch that adds its output (TF_GLM53_FOLD_SHARED). Every
//! kernel computes what it computes on one stream (its own EXL3 / router scratch on the side, as torch's per-stream
//! allocator gives the Python engine), so the bits are the one-stream bits.
//!
//! `L2pf` (TF_GLM53_L2PF, default bulk, 8 MiB a site, sites "afo"; MiaAI-Lab patch 0046's l2pf.py): in the target's
//! decode windows (fused._compute_pf: up to min(b.small, 32) rows; never the MTP layer or prompt chunks) three sites a
//! layer fork a third stream that asks L2 for the weights the main stream reads next - "a" after the attention
//! partial (the post-attention norm, the router and its bias, the shared expert / dense MLP), "f" after the FFN
//! partial (the next layer's input norm, q_a + kv_a, their norms, the indexer's small projections), "o" after absorb
//! (expand's kv_b value half, o_proj). Within an EXL3 linear its scales first, then its words whole when they fit,
//! else the head of every warp's k range. The prefetch only reads: no kernel's inputs, outputs or order change.
//! The table is l2pf.Prefetch's, built the same way from the same tensors (sizes and tiles), so the same bytes.
//!
//! Both streams fork from the compute stream with an event and join it before the window ends, so a captured decode
//! graph holds them as branches that rejoin before its end.
const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("exl3.zig");
const kern = @import("cuda_kernels.zig");
const W = @import("cuda_weights.zig");

/// TF_GLM53_SIDE's letters (fused._side_letters).
pub const SideLetters = struct {
    a: bool = true,
    w: bool = false,
    f: bool = true,

    pub fn any(l: SideLetters) bool {
        return l.a or l.w or l.f;
    }
};

/// fused._side_letters: "0" / "off" -> none, "1" / "on" -> "af", else letters of "awf" ("a" and "w" exclude each other).
pub fn parseSide(v: []const u8) !SideLetters {
    if (v.len == 0 or std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "on")) return .{};
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return .{ .a = false, .w = false, .f = false };
    var l: SideLetters = .{ .a = false, .w = false, .f = false };
    for (v) |ch| switch (ch) {
        'a' => l.a = true,
        'w' => l.w = true,
        'f' => l.f = true,
        else => return error.BadSideLetters,
    };
    if (l.a and l.w) return error.BadSideLetters;
    return l;
}

/// The decode windows' second stream (fused.Buffers.side) with its fork / join events.
pub const Side = struct {
    s: cuda.Stream,
    fork_ev: cuda.Event,
    join_ev: cuda.Event,
    letters: SideLetters,
    /// a decode window is running (fused.Buffers.side_on non-empty)
    on: bool = false,
    /// work queued on the side stream since the last join (fused.Buffers.forked)
    forked: bool = false,

    pub fn init(d: *const cuda.Driver, letters: SideLetters) !Side {
        var s = try cuda.Stream.init(d, true);
        errdefer s.deinit();
        var f = try cuda.Event.init(d, false);
        errdefer f.deinit();
        const j = try cuda.Event.init(d, false);
        return .{ .s = s, .fork_ev = f, .join_ev = j, .letters = letters };
    }

    pub fn deinit(sd: *Side) void {
        sd.join_ev.deinit();
        sd.fork_ev.deinit();
        sd.s.deinit();
    }

    /// fused._fork: the side stream queues after everything `main` has queued so far.
    pub fn fork(sd: *Side, main: cuda.Stream) !void {
        try sd.fork_ev.record(main);
        try sd.s.wait(sd.fork_ev);
        sd.forked = true;
    }

    /// An event marking the side stream's work so far (experts.routed's sy_ready waits on it); the side counts as
    /// joined once the caller made the compute stream wait on it.
    pub fn mark(sd: *Side) !cuda.Event {
        try sd.join_ev.record(sd.s);
        return sd.join_ev;
    }

    /// fused._join: `main` waits for the side stream's work so far (nothing when nothing was forked).
    pub fn join(sd: *Side, main: cuda.Stream) !void {
        if (!sd.forked) return;
        try sd.join_ev.record(sd.s);
        try main.wait(sd.join_ev);
        sd.forked = false;
    }
};

/// l2pf.Settings (TF_GLM53_L2PF / _MB / _SITES / _ROWS / _BLOCKS / _THREADS / _KB).
pub const L2pfSettings = struct {
    /// 0 bulk (cp.async.bulk.prefetch.L2), 1 lines, 2 touch
    mode: c_int = 0,
    mb: f64 = 8.0,
    site_a: bool = true,
    site_f: bool = true,
    site_o: bool = true,
    rows: usize = 32,
    blocks: u32 = 0,
    threads: u32 = 128,
    piece: usize = 32 << 10,
};

pub const SiteName = enum(u2) { a = 0, f = 1, o = 2 };

const Entry = struct { first: u32 = 0, count: u32 = 0, grid: u32 = 0 };

const align_bytes: u64 = 16; // cp.async.bulk wants 16-byte addresses and sizes
const min_head: i64 = 512; // a warp chunk's head under this many bytes is not worth a piece

/// One thing a site covers, in the order the main stream reads it: a tensor (address, bytes), an EXL3 linear, or
/// EXL3 linears of one launch.
const Item = union(enum) {
    t: [2]u64,
    lin: *const exl3.Linear,
    grp: [2]*const exl3.Linear,
};

/// l2pf.ranges: (address, bytes) spans of `items` up to `budget` bytes, in order.
const Plan = struct {
    gpa: std.mem.Allocator,
    out: *std.ArrayList([2]u64),
    left: i64,

    /// l2pf._span: the leading min(size, left) bytes of [a, a + n), widened to 16-byte bounds.
    fn span(a: u64, n: u64, left: i64) ?[2]u64 {
        if (left <= 0) return null;
        const lo = a - a % align_bytes;
        var hi = @min(a + n, lo + @as(u64, @intCast(left)));
        hi += (align_bytes - hi % align_bytes) % align_bytes;
        return if (hi > lo) .{ lo, hi - lo } else null;
    }

    fn take(p: *Plan, a: u64, n: u64) !void {
        if (a == 0 or n == 0 or p.left < @as(i64, @intCast(align_bytes))) return;
        if (span(a, n, p.left)) |sp| {
            try p.out.append(p.gpa, sp);
            p.left -= @intCast(sp[1]);
        }
    }

    /// l2pf.exl3_heads: the first budget / chunks bytes of every warp's contiguous k range (strips layout).
    fn heads(p: *Plan, l: *const exl3.Linear, budget: i64) !bool {
        const kt = l.k / 16;
        const chunks = l.sk * l.wk;
        if (chunks == 0 or kt % chunks != 0 or budget <= 0) return false;
        const step: u64 = 8 * 4 * l.k2 * 4; // bytes of a k step of one strip
        const per: u64 = kt / chunks * step; // a warp's chunk
        const nb = l.n / 128;
        const each = @divFloor(budget, @as(i64, @intCast(nb * chunks)));
        const head: i64 = @divFloor(@min(@as(i64, @intCast(per)), each), 16) * 16;
        if (head < min_head or l.words % align_bytes != 0) return false;
        const strip = kt * step;
        for (0..nb) |b| for (0..chunks) |c| {
            try p.out.append(p.gpa, .{ l.words + b * strip + c * per, @intCast(head) });
        };
        p.left -= head * @as(i64, @intCast(nb * chunks));
        return true;
    }

    fn words(p: *Plan, lins: []const *const exl3.Linear) !void {
        var total: u64 = 0;
        for (lins) |l| total += l.wordBytes();
        if (@as(i64, @intCast(total)) <= p.left) {
            for (lins) |l| try p.take(l.words, l.wordBytes());
            return;
        }
        const share = p.left;
        if (share <= 0) return;
        for (lins) |l| {
            const mine = @divFloor(share * @as(i64, @intCast(l.wordBytes())), @as(i64, @intCast(@max(total, 1))));
            if (try p.heads(l, mine)) continue;
            if (span(l.words, l.wordBytes(), @divFloor(mine, 16) * 16)) |sp| {
                if (@as(i64, @intCast(sp[1])) <= p.left) {
                    try p.out.append(p.gpa, sp);
                    p.left -= @intCast(sp[1]);
                }
            }
        }
    }

    fn small(p: *Plan, l: *const exl3.Linear) !void {
        try p.take(l.suh, l.k * 2); // fp16 [K]
        try p.take(l.svh, l.n * 2); // fp16 [N]
    }

    fn run(p: *Plan, items: []const Item) !void {
        for (items) |it| {
            if (p.left < @as(i64, @intCast(align_bytes))) break;
            switch (it) {
                .t => |t| try p.take(t[0], t[1]),
                .lin => |l| {
                    try p.small(l);
                    try p.words(&.{l});
                },
                .grp => |g| {
                    for (g) |l| try p.small(l);
                    try p.words(&g);
                },
            }
        }
    }
};

/// l2pf.Prefetch: the sites' pieces for one rank's weights (one device table; (first, count, grid) a site), its
/// stream and events. Build it after the tiles are final (tiles.load) and before any graph capture.
pub const L2pf = struct {
    gpa: std.mem.Allocator,
    set: L2pfSettings,
    s: cuda.Stream,
    fork_ev: cuda.Event,
    join_ev: cuda.Event,
    table: cuda.DeviceBuffer,
    sink: cuda.DeviceBuffer,
    entries: []Entry, // [layers * 3]
    layers: usize,
    site_bytes: [3]u64 = @splat(0),
    site_count: [3]usize = @splat(0),
    /// a target decode window is running its sites (l2pf.Prefetch.active)
    active: bool = false,
    forked: bool = false,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, set: L2pfSettings) !L2pf {
        const c = w.config;
        const D = c.hidden;
        const budget: i64 = @intFromFloat(set.mb * (1 << 20));
        var pieces: std.ArrayList([2]u64) = .empty;
        defer pieces.deinit(gpa);
        var spans: std.ArrayList([2]u64) = .empty;
        defer spans.deinit(gpa);
        const n = w.layers.len;
        const entries = try gpa.alloc(Entry, n * 3);
        errdefer gpa.free(entries);
        @memset(entries, .{});
        var pf: L2pf = .{ .gpa = gpa, .set = set, .s = undefined, .fork_ev = undefined, .join_ev = undefined, .table = undefined, .sink = undefined, .entries = entries, .layers = n };
        var items: std.ArrayList(Item) = .empty;
        defer items.deinit(gpa);
        for (w.layers, 0..) |*L, i| {
            for ([_]SiteName{ .a, .f, .o }) |name| {
                const on = switch (name) {
                    .a => set.site_a,
                    .f => set.site_f,
                    .o => set.site_o,
                };
                if (!on) continue;
                items.clearRetainingCapacity();
                switch (name) {
                    .a => { // l2pf.site_a (TF_GLM53_FOLD_SHARED: the shared expert before the routed ones)
                        try items.append(gpa, .{ .t = .{ L.post_attn_norm, D * 2 } });
                        if (L.router_w != 0) {
                            try items.append(gpa, .{ .t = .{ L.router_w, c.experts * D * 2 } });
                            try items.append(gpa, .{ .t = .{ L.router_bias, c.experts * 4 } });
                        }
                        try items.append(gpa, .{ .grp = .{ &L.mlp.gate, &L.mlp.up } });
                        try items.append(gpa, .{ .lin = &L.mlp.down });
                    },
                    .f => { // l2pf.site_f: the next layer's attention_part inputs (none after the last layer)
                        if (i + 1 >= n) continue;
                        const nx = &w.layers[i + 1];
                        try items.append(gpa, .{ .t = .{ nx.input_norm, D * 2 } });
                        try items.append(gpa, .{ .grp = .{ &nx.q_a, &nx.kv_a } });
                        try items.append(gpa, .{ .t = .{ nx.q_a_norm, c.q_lora * 2 } });
                        try items.append(gpa, .{ .t = .{ nx.kv_a_norm, c.kv_lora * 2 } });
                        if (nx.indexer) |*ix| {
                            try items.append(gpa, .{ .t = .{ ix.wk, c.index_dim * D * 2 } });
                            try items.append(gpa, .{ .t = .{ ix.k_norm_w, c.index_dim * 2 } });
                            try items.append(gpa, .{ .t = .{ ix.k_norm_b, c.index_dim * 2 } });
                            try items.append(gpa, .{ .t = .{ ix.weights_proj, c.index_heads * D * 2 } });
                        }
                    },
                    .o => { // l2pf.site_o: expand's kv_b value half, then o_proj
                        try items.append(gpa, .{ .t = .{ L.wv, w.heads * c.v_dim * c.kv_lora * 2 } });
                        try items.append(gpa, .{ .lin = &L.o_proj });
                    },
                }
                if (items.items.len == 0) continue;
                spans.clearRetainingCapacity();
                var plan: Plan = .{ .gpa = gpa, .out = &spans, .left = budget };
                try plan.run(items.items);
                const first = pieces.items.len;
                var bytes: u64 = 0;
                for (spans.items) |sp| {
                    bytes += sp[1];
                    var o: u64 = 0;
                    while (o < sp[1]) : (o += set.piece) try pieces.append(gpa, .{ sp[0] + o, @min(set.piece, sp[1] - o) });
                }
                const count = pieces.items.len - first;
                if (count == 0) continue;
                entries[i * 3 + @intFromEnum(name)] = .{ .first = @intCast(first), .count = @intCast(count), .grid = pf.grid(count) };
                pf.site_bytes[@intFromEnum(name)] += bytes;
                pf.site_count[@intFromEnum(name)] += 1;
            }
        }
        // the device table: int64 (address, bytes) pairs ((0, 0) when empty)
        const rows = @max(pieces.items.len, 1);
        const host = try gpa.alloc(i64, rows * 2);
        defer gpa.free(host);
        @memset(host, 0);
        for (pieces.items, 0..) |pc, j| {
            host[2 * j] = @intCast(pc[0]);
            host[2 * j + 1] = @intCast(pc[1]);
        }
        pf.table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
        errdefer pf.table.free();
        pf.sink = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer pf.sink.free();
        try pf.sink.fill8(0, null);
        pf.s = try cuda.Stream.init(d, true);
        errdefer pf.s.deinit();
        pf.fork_ev = try cuda.Event.init(d, false);
        errdefer pf.fork_ev.deinit();
        pf.join_ev = try cuda.Event.init(d, false);
        return pf;
    }

    pub fn deinit(pf: *L2pf) void {
        pf.join_ev.deinit();
        pf.fork_ev.deinit();
        pf.s.deinit();
        pf.sink.free();
        pf.table.free();
        pf.gpa.free(pf.entries);
    }

    fn grid(pf: *const L2pf, n: usize) u32 {
        if (pf.set.blocks > 0) return pf.set.blocks;
        const per: usize = if (pf.set.mode == 0) pf.set.threads else pf.set.threads / 32;
        return @intCast(@max(1, @min(48, (n + per - 1) / per)));
    }

    /// l2pf.Prefetch.site: fork the prefetch stream at this point of `main` and prefetch the site's pieces there
    /// (nothing outside an active decode window, past the target's layers, or for an empty site).
    pub fn site(pf: *L2pf, k: *const kern.Kernels, main: cuda.Stream, layer: usize, name: SiteName) !void {
        if (!pf.active or layer >= pf.layers) return;
        const e = pf.entries[layer * 3 + @intFromEnum(name)];
        if (e.count == 0) return;
        try pf.fork_ev.record(main);
        try pf.s.wait(pf.fork_ev);
        const o: kern.Ops = .{ .k = k, .s = pf.s };
        try o.l2pf(pf.table.ptr + @as(u64, e.first) * 16, e.count, pf.set.mode, e.grid, pf.set.threads, pf.sink.ptr);
        pf.forked = true;
    }

    /// l2pf.Prefetch.join: `main` waits for the prefetch stream (a captured graph's branch rejoins before its end).
    pub fn join(pf: *L2pf, main: cuda.Stream) !void {
        if (!pf.forked) return;
        try pf.join_ev.record(pf.s);
        try main.wait(pf.join_ev);
        pf.forked = false;
    }

    /// l2pf.Prefetch.summary.
    pub fn summary(pf: *const L2pf, buf: []u8) []const u8 {
        const mode = switch (pf.set.mode) {
            0 => "bulk",
            1 => "lines",
            else => "touch",
        };
        const mib = struct {
            fn avg(b: u64, n: usize) f64 {
                return if (n == 0) 0 else @as(f64, @floatFromInt(b)) / @as(f64, @floatFromInt(n)) / (1 << 20);
            }
        };
        return std.fmt.bufPrint(buf, "L2 prefetch in decode windows ({s}, up to {d} MiB a site): a {d} sites {d:.1} MiB, f {d} sites {d:.1} MiB, o {d} sites {d:.1} MiB", .{ mode, pf.set.mb, pf.site_count[0], mib.avg(pf.site_bytes[0], pf.site_count[0]), pf.site_count[1], mib.avg(pf.site_bytes[1], pf.site_count[1]), pf.site_count[2], mib.avg(pf.site_bytes[2], pf.site_count[2]) }) catch buf[0..0];
    }
};

test "TF_GLM53_SIDE letters as fused._side_letters" {
    const d = try parseSide("1");
    try std.testing.expect(d.a and d.f and !d.w);
    const z = try parseSide("0");
    try std.testing.expect(!z.any());
    const wf = try parseSide("wf");
    try std.testing.expect(wf.w and wf.f and !wf.a);
    try std.testing.expectError(error.BadSideLetters, parseSide("aw"));
    try std.testing.expectError(error.BadSideLetters, parseSide("x"));
}

test "l2pf spans: 16-byte bounds, budget cut" {
    try std.testing.expectEqual([2]u64{ 1024, 64 }, Plan.span(1024, 60, 1 << 20).?);
    try std.testing.expectEqual([2]u64{ 1024, 32 }, Plan.span(1024, 1000, 32).?);
    try std.testing.expectEqual(@as(?[2]u64, null), Plan.span(1024, 1000, 0));
}
