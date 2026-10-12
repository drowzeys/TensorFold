//! Decode context parallelism (Phase 3b): the Python engine's DCP = 4 for contexts past ~200K tokens (Glm53Engine:
//! TF_GLM53_DCP unset turns it on past DCP_AUTO), its cache sharding and index math, and NCCL's all-to-all for the
//! prompt chunks' head exchange (fused._attention_dcp's comm.all_to_all, which nccl.zig does not wrap: ncclSend /
//! ncclRecv resolved here from the communicator's own library).
//!
//! The sharding (fused.State): position p lives on rank p % dcp at local row p // dcp; a rank's caches hold
//! ceil(capacity / dcp) + 1 rows. Attention (fused._attention_dcp): every rank's absorbed queries are gathered, each
//! rank attends ALL heads over its own keys, the normalized partials and log-sum-exps go to the heads' owners
//! (decode windows: one all-gather each; prompt chunks: all-to-all), which merge them in rank order. Selection
//! (fused.select, DCP): each rank scores its own slots, keeps its top index_topk packed keys, the candidates are
//! gathered and every rank takes the same global top index_topk; its share as local slots (b.tok) and a count (b.cnt).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const fw = @import("cuda_forward.zig");
const W = @import("cuda_weights.zig");

/// engine.DCP_AUTO: contexts past this many tokens interleave the KV cache over the ranks (bf16 latent cache).
pub const dcp_auto_tokens: usize = 200_000;

/// engine.dcp_auto for the bf16 latent (the only cache this port has): the context past which TF_GLM53_DCP unset turns
/// decode context parallelism on.
pub fn dcpAuto(cfg: Config) usize {
    _ = cfg;
    return dcp_auto_tokens;
}

/// Glm53Engine's choice: TF_GLM53_DCP when set (1 or world), else world past dcpAuto, else 1.
pub fn choose(cfg: Config, context: usize, world: usize, env: ?usize) error{BadDcp}!usize {
    if (env) |v| {
        if (v != 0) {
            if (v != 1 and v != world) return error.BadDcp;
            return v;
        }
    }
    return if (context > dcpAuto(cfg)) world else 1;
}

/// fused.State.bytes_for with DCP: the caches of `capacity` tokens a rank holds.
pub fn bytesForDcp(w: *const W.Weights, capacity: usize, dcp: usize) usize {
    return fw.State.bytesForDcp(w, capacity, dcp);
}

/// ceil(a / b).
pub fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// fused.State's `local`: rows a rank's caches hold for `capacity` tokens.
pub fn localCapacity(capacity: usize, dcp: usize) usize {
    return cdiv(capacity, dcp) + 1;
}

/// The rank that holds position p, and its row there.
pub fn owner(p: usize, dcp: usize) struct { rank: usize, row: usize } {
    return .{ .rank = p % dcp, .row = p / dcp };
}

/// _attn_dcp's key count of a row at position p < index_topk on `rank`: its slots of positions 0..p.
pub fn keysBelow(p: usize, dcp: usize, rank: usize) usize {
    return if (p >= rank) (p - rank) / dcp + 1 else 0;
}

/// fused.select's Tl: the slots a rank scores for a key range of T positions.
pub fn localCols(T: usize, dcp: usize) usize {
    return cdiv(T, dcp);
}

/// _attention_dcp's exchange layout for R rows of H own heads and lw latent columns over G ranks: where this rank's
/// heads sit in the received partials (element offsets o0 / l0) and the per-source strides (ss / ssl). `small`:
/// decode windows (all-gather of every rank's whole send buffer), else prompt chunks (all-to-all of head blocks).
pub const Exchange = struct { o0: usize, l0: usize, ss: usize, ssl: usize };

pub fn exchange(small: bool, G: usize, rank: usize, R: usize, H: usize, lw: usize) Exchange {
    if (small) return .{ .o0 = rank * R * H * lw, .l0 = rank * R * H, .ss = G * R * H * lw, .ssl = G * R * H };
    return .{ .o0 = 0, .l0 = 0, .ss = R * H * lw, .ssl = R * H };
}

// ------------------------------------------------------------- several streams under DCP (this port's own) ---
// docs/glm53-dcp-streams.md. The Python engine refuses --parallel N with DCP on (fused.attention_part: "several
// streams in one window need decode context parallelism off"): its ROWS kernels ignore DCP and _attn_dcp takes one
// position. Nothing in the method needs one stream: a stream's shard of a rank is `localCapacity` rows, the slots sit
// back to back, and a row of stream s at position p is the row BASE + p / dcp on rank p % dcp with BASE = s * local.

/// The most tokens of cache N streams may hold in all with the cache split over the ranks: the one-stream shape that
/// has been served (1M x 1). Nothing larger has ever been allocated on a node.
pub const streams_tokens: usize = 1 << 20;
/// GiB that must stay free on a rank after the slots' caches (TF_GLM53_DCP_STREAMS_RESERVE_GB): what the runner's
/// buffers, the drafters' rings, the decode graphs and the host still take after the check, with room to spare.
pub const streams_reserve_gb: f64 = 20.0; // for two streams, scaled by streams / 2 (cuda_engine); measured 2026-10-12 at 250K x 2: 14 to 21 GiB go after this check (graphs, windows, drafter); 8 let 250K x 4 start and the memory guard stopped it

pub fn streamsTokensOk(context: usize, streams: usize) bool {
    return streams >= 1 and context <= streams_tokens / streams;
}

/// Row r of a window: stream `slot` at position p -> (the rank that stores its keys, the cache row there).
pub fn slotOwner(p: usize, dcp: usize, slot: usize, local: usize) struct { rank: usize, row: usize } {
    return .{ .rank = p % dcp, .row = slot * local + p / dcp };
}

/// Bytes of cache one token of context takes over all ranks: a latent row a layer, an index key a full-indexer
/// layer, and the MTP layer's (fw.State.bytesForDcp's `per`).
pub fn tokenBytes(layers: usize, full_indexers: usize, latent: usize, index_dim: usize, mtp: bool, mtp_indexer: bool) usize {
    var per = layers * latent * 2 + full_indexers * index_dim * 2;
    if (mtp) per += latent * 2 + (if (mtp_indexer) index_dim * 2 else 0);
    return per;
}

/// What `streams` slots of `context` tokens take on one rank (capacity = context + k + 1, as the engine sizes it).
pub fn rankCacheBytes(token_bytes: usize, context: usize, k: usize, streams: usize, dcp: usize) usize {
    return token_bytes * localCapacity(context + k + 1, dcp) * streams;
}

pub const StreamsFit = struct {
    need: u64,
    free: u64,
    reserve: u64,
    ok: bool,
    streams: u64,
    dcp_rows: u64,

    /// The most context a stream that would fit (0: none), for the refusal's message.
    pub fn fit_tokens(f: StreamsFit, capacity: usize) u64 {
        if (f.free <= f.reserve or f.need == 0) return 0;
        const room = f.free - f.reserve;
        if (room >= f.need) return capacity;
        return @intFromFloat(@as(f64, @floatFromInt(capacity)) * @as(f64, @floatFromInt(room)) / @as(f64, @floatFromInt(f.need)) * 0.98);
    }
};

/// The boot check of several streams under DCP: `per` = a rank's bytes for one stream (fw.State.bytesForDcp), `free`
/// what the device reports before the caches allocate. Unlike cacheFit the reserve is whole: never halved.
pub fn streamsFit(per: u64, streams: u64, free: u64, reserve_gb: f64) StreamsFit {
    const need = per * streams;
    const reserve: u64 = @intFromFloat(@max(reserve_gb, 0) * (1 << 30));
    return .{ .need = need, .free = free, .reserve = reserve, .ok = need + reserve <= free, .streams = streams, .dcp_rows = 0 };
}

// ---------------------------------------------------------------------------------------- NCCL all-to-all ---

const SendFn = *const fn (u64, usize, cuda.nccl.DataType, c_int, cuda.nccl.Comm, cuda.abi.Stream) callconv(.c) cuda.nccl.Result;
const RecvFn = *const fn (u64, usize, cuda.nccl.DataType, c_int, cuda.nccl.Comm, cuda.abi.Stream) callconv(.c) cuda.nccl.Result;

/// ncclSend / ncclRecv of the communicator's library, resolved once (the engine thread is the only caller).
var p2p_lib: ?*const cuda.nccl.Library = null;
var p2p_send: ?SendFn = null;
var p2p_recv: ?RecvFn = null;

fn p2p(lib: *const cuda.nccl.Library) !struct { send: SendFn, recv: RecvFn } {
    if (p2p_lib != lib) {
        const dl = @constCast(&lib.lib);
        p2p_send = dl.lookup(SendFn, "ncclSend") orelse return error.MissingSymbol;
        p2p_recv = dl.lookup(RecvFn, "ncclRecv") orelse return error.MissingSymbol;
        p2p_lib = lib;
    }
    return .{ .send = p2p_send.?, .recv = p2p_recv.? };
}

/// comm.all_to_all(send.view(G, n), recv.view(G, n)): block k of `send` (count elements of `dt`, `elem` bytes each)
/// to rank k, rank k's block for this rank into block k of `recv`; grouped sends and receives on `stream` (what
/// torch's all_to_all_single does). Moves words only.
pub fn allToAll(comm: *const cuda.nccl.Communicator, send: u64, recv: u64, count: usize, dt: cuda.nccl.DataType, elem: usize, stream: cuda.abi.Stream) !void {
    const f = try p2p(comm.lib);
    const lib = comm.lib;
    try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
    var failed: ?anyerror = null;
    for (0..comm.world) |k| {
        const off = k * count * elem;
        lib.check(f.send(send + off, count, dt, @intCast(k), comm.comm, stream), "ncclSend") catch |e| {
            failed = e;
            break;
        };
        lib.check(f.recv(recv + off, count, dt, @intCast(k), comm.comm, stream), "ncclRecv") catch |e| {
            failed = e;
            break;
        };
    }
    try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    if (failed) |e| return e;
}

test "positions shard round robin: owner, local rows, slots below a position" {
    try std.testing.expectEqual(@as(usize, 250_001), localCapacity(1_000_000, 4));
    try std.testing.expectEqual(@as(usize, 1_000_001), localCapacity(1_000_000, 1));
    const o = owner(830_001, 4);
    try std.testing.expectEqual(@as(usize, 1), o.rank);
    try std.testing.expectEqual(@as(usize, 207_500), o.row);
    // position 5 on 4 ranks: rank 0 holds 0, 4; rank 1 holds 1, 5; rank 2 holds 2; rank 3 holds 3
    try std.testing.expectEqual(@as(usize, 2), keysBelow(5, 4, 0));
    try std.testing.expectEqual(@as(usize, 2), keysBelow(5, 4, 1));
    try std.testing.expectEqual(@as(usize, 1), keysBelow(5, 4, 2));
    try std.testing.expectEqual(@as(usize, 1), keysBelow(5, 4, 3));
    try std.testing.expectEqual(@as(usize, 0), keysBelow(2, 4, 3));
    var total: usize = 0;
    for (0..4) |r| total += keysBelow(1234, 4, r);
    try std.testing.expectEqual(@as(usize, 1235), total);
}

test "select's local key range and the exchange layout" {
    try std.testing.expectEqual(@as(usize, 768), localCols(3072, 4));
    try std.testing.expectEqual(@as(usize, 262_144), localCols(1 << 20, 4));
    try std.testing.expectEqual(@as(usize, 1), localCols(1, 4));
    const s = exchange(true, 4, 2, 3, 16, 512);
    try std.testing.expectEqual(@as(usize, 2 * 3 * 16 * 512), s.o0);
    try std.testing.expectEqual(@as(usize, 4 * 3 * 16 * 512), s.ss);
    try std.testing.expectEqual(@as(usize, 2 * 3 * 16), s.l0);
    const p = exchange(false, 4, 2, 1024, 16, 512);
    try std.testing.expectEqual(@as(usize, 0), p.o0);
    try std.testing.expectEqual(@as(usize, 1024 * 16 * 512), p.ss);
}

test "DCP turns on past DCP_AUTO, TF_GLM53_DCP 1 or the world" {
    const cfg: Config = undefined;
    try std.testing.expectEqual(@as(usize, 1), try choose(cfg, 200_000, 4, null));
    try std.testing.expectEqual(@as(usize, 4), try choose(cfg, 200_001, 4, null));
    try std.testing.expectEqual(@as(usize, 1), try choose(cfg, 1 << 20, 4, 1));
    try std.testing.expectEqual(@as(usize, 4), try choose(cfg, 4096, 4, 4));
    try std.testing.expectEqual(@as(usize, 4), try choose(cfg, 1 << 20, 4, 0));
    try std.testing.expectError(error.BadDcp, choose(cfg, 1 << 20, 4, 2));
}

// ----------------------------------------------------- host model: several streams under DCP = each alone ---
// A CPU model of the position-dependent kernels in both forms - one stream (POS a first position; fused._kv_write /
// _index_scores / _attn_dcp with DCP, RANK) and several (POS / BASE tables; the ROWS forms of
// tools/glm53/patch_fused_dcp_rows.py) - over toy keys. It checks the layout and the composition (what each rank
// stores, scores, selects, attends and how the ranks' partials merge), not the device kernels' arithmetic.

const Model = struct {
    const K: usize = 8; // index_topk
    const G: usize = 4; // ranks
    gpa: std.mem.Allocator,
    local: usize,
    slots: usize,
    kc: [G][]f64, // latent cache a rank: [slots * local]
    ic: [G][]f64, // index keys a rank

    fn init(gpa: std.mem.Allocator, capacity: usize, slots: usize) !Model {
        var m: Model = .{ .gpa = gpa, .local = localCapacity(capacity, G), .slots = slots, .kc = undefined, .ic = undefined };
        for (0..G) |r| {
            m.kc[r] = try gpa.alloc(f64, m.local * slots);
            m.ic[r] = try gpa.alloc(f64, m.local * slots);
            @memset(m.kc[r], 0);
            @memset(m.ic[r], 0);
        }
        return m;
    }

    fn deinit(m: *Model) void {
        for (0..G) |r| {
            m.gpa.free(m.kc[r]);
            m.gpa.free(m.ic[r]);
        }
    }

    fn mix(a: u64, b: u64, c: u64) f64 {
        var h = std.hash.Wyhash.init(a *% 0x9E3779B97F4A7C15 +% c);
        h.update(std.mem.asBytes(&b));
        return @as(f64, @floatFromInt(h.final() % 2001)) / 1000.0 - 1.0;
    }
    fn latentOf(stream: usize, p: usize) f64 {
        return mix(stream, p, 1);
    }
    fn indexOf(stream: usize, p: usize) f64 {
        return mix(stream, p, 2);
    }
    fn queryOf(stream: usize, p: usize, head: usize) f64 {
        return mix(stream, p * 64 + head, 3);
    }

    /// _kv_write / _ik_write: `base` null - one stream, rows at pos0 + r; else row r at pos[r] of the stream at base[r].
    fn write(m: *Model, rank: usize, pos: []const usize, base: ?[]const usize, lat: []const f64, idx: []const f64) void {
        for (pos, 0..) |pg, r| {
            if (pg % G != rank) continue;
            const row = (if (base) |b| b[r] else 0) + pg / G;
            m.kc[rank][row] = lat[r];
            m.ic[rank][row] = idx[r];
        }
    }

    /// _index_scores (DCP, PACK): row r's packed keys over this rank's slots t < Tl of its stream.
    fn scores(m: *const Model, rank: usize, p: usize, base: usize, q: f64, Tl: usize, out: []i64) void {
        for (0..Tl) |t| {
            const g = t * G + rank;
            if (g <= p) {
                const sc: f32 = @floatCast(@max(q * m.ic[rank][base + t], 0));
                const bits: i32 = @bitCast(sc);
                const key: i64 = if (bits >= 0) bits else bits ^ 0x7FFFFFFF;
                out[t] = (key << 32) | @as(i64, @intCast(0x7FFFFFFF - g));
            } else out[t] = -9223372036854775807;
        }
    }

    fn desc(_: void, a: i64, b: i64) bool {
        return a > b;
    }

    /// fused.select's DCP branch for one row: every rank's top min(K, Tl) keys, gathered, the global top K; each
    /// rank's share as local slots ascending (tok) and its count.
    fn select(m: *const Model, p: usize, base: usize, q: f64, T: usize, tok: *[G][K]usize, cnt: *[G]usize) !void {
        const Tl = localCols(T, G);
        const kk = @min(K, Tl);
        var cand: [G * K]i64 = @splat(-9223372036854775807);
        const sc = try m.gpa.alloc(i64, Tl);
        defer m.gpa.free(sc);
        for (0..G) |rank| {
            m.scores(rank, p, base, q, Tl, sc);
            std.mem.sort(i64, sc, {}, desc);
            @memcpy(cand[rank * K ..][0..kk], sc[0..kk]);
        }
        std.mem.sort(i64, &cand, {}, desc);
        cnt.* = @splat(0);
        for (cand[0..K]) |c| {
            const gpos: usize = @intCast(0x7FFFFFFF - (c & 0xFFFFFFFF));
            const rank = gpos % G;
            tok[rank][cnt[rank]] = gpos / G;
            cnt[rank] += 1;
        }
        for (0..G) |rank| std.mem.sort(usize, tok[rank][0..cnt[rank]], {}, std.sort.asc(usize));
    }

    /// _attn_dcp for one row and head on every rank (its slots 0..p while p < K, else its share of the selection, read
    /// from its stream's rows at `base`), _merge_lse and _dcp_combine in rank order.
    fn attend(m: *const Model, p: usize, base: usize, q: f64, tok: *const [G][K]usize, cnt: *const [G]usize) f64 {
        var o: [G]f64 = undefined;
        var l: [G]f64 = undefined; // log-sum-exp a rank (-inf: no keys)
        for (0..G) |rank| {
            const n = if (p < K) keysBelow(p, G, rank) else cnt[rank];
            var mx = -std.math.inf(f64);
            var sum: f64 = 0;
            var acc: f64 = 0;
            for (0..n) |i| {
                const key = if (p < K) i else tok[rank][i];
                const kv = m.kc[rank][base + key];
                const s = q * kv;
                const next = @max(mx, s);
                const alpha = if (mx == -std.math.inf(f64)) 0 else @exp(mx - next);
                const pr = @exp(s - next);
                acc = acc * alpha + pr * kv;
                sum = sum * alpha + pr;
                mx = next;
            }
            o[rank] = if (sum > 0) acc / sum else 0;
            l[rank] = if (sum > 0) mx + @log(sum) else -std.math.inf(f64);
        }
        var top = -std.math.inf(f64);
        for (l) |x| top = @max(top, x);
        var acc: f64 = 0;
        var den: f64 = 0;
        for (0..G) |rank| {
            const w = if (l[rank] == -std.math.inf(f64)) 0 else @exp(l[rank] - top);
            acc += w * o[rank];
            den += w;
        }
        return acc / den;
    }
};

/// One stream's `n` tokens decoded in windows of `win` rows on a model of its own (slot 0): every row's attention
/// output of head 0 and 1, in position order.
fn modelAlone(gpa: std.mem.Allocator, stream: usize, n: usize, win: usize, capacity: usize, out: []f64, caches: ?*Model) !void {
    var own = try Model.init(gpa, capacity, 1);
    defer own.deinit();
    const m = caches orelse &own;
    var a: usize = 0;
    while (a < n) : (a += win) {
        const R = @min(win, n - a);
        var pos: [16]usize = undefined;
        var lat: [16]f64 = undefined;
        var idx: [16]f64 = undefined;
        for (0..R) |r| {
            pos[r] = a + r;
            lat[r] = Model.latentOf(stream, a + r);
            idx[r] = Model.indexOf(stream, a + r);
        }
        for (0..Model.G) |rank| m.write(rank, pos[0..R], null, lat[0..R], idx[0..R]);
        const T = a + R; // the window's last position + 1 (what the host passes when it is past index_topk)
        for (0..R) |r| {
            var tok: [Model.G][Model.K]usize = undefined;
            var cnt: [Model.G]usize = @splat(0);
            for (0..2) |h| {
                const p = a + r;
                if (T > Model.K) try m.select(p, 0, Model.queryOf(stream, p, 7), T, &tok, &cnt);
                out[p * 2 + h] = m.attend(p, 0, Model.queryOf(stream, p, h), &tok, &cnt);
            }
        }
    }
}

test "several streams under DCP: each stream's rows, selections and attention are what it has alone" {
    const gpa = std.testing.allocator;
    const capacity: usize = 96;
    const lens = [_]usize{ 61, 5, 90, 33 }; // one stream stays below index_topk, the others pass it at different rounds
    const S = lens.len;
    var m = try Model.init(gpa, capacity, S);
    defer m.deinit();
    var alone: [S][]f64 = undefined;
    var together: [S][]f64 = undefined;
    for (0..S) |s| {
        alone[s] = try gpa.alloc(f64, lens[s] * 2);
        together[s] = try gpa.alloc(f64, lens[s] * 2);
    }
    defer for (0..S) |s| {
        gpa.free(alone[s]);
        gpa.free(together[s]);
    };
    // each stream alone, in windows of 3 rows (its pending token and two drafts)
    for (0..S) |s| try modelAlone(gpa, s + 1, lens[s], 3, capacity, alone[s], null);
    // together: a round takes up to 3 rows of every stream that has rows left, in one window with POS / BASE tables
    var at: [S]usize = @splat(0);
    while (true) {
        var pos: [16]usize = undefined;
        var base: [16]usize = undefined;
        var who: [16]usize = undefined;
        var lat: [16]f64 = undefined;
        var idx: [16]f64 = undefined;
        var R: usize = 0;
        var T: usize = 0; // the host's bound: the window's largest position + 1 over every stream
        var tmax: [S]usize = @splat(0);
        for (0..S) |s| {
            const n = @min(3, lens[s] - at[s]);
            for (0..n) |j| {
                pos[R] = at[s] + j;
                base[R] = s * m.local;
                who[R] = s;
                lat[R] = Model.latentOf(s + 1, at[s] + j);
                idx[R] = Model.indexOf(s + 1, at[s] + j);
                R += 1;
            }
            tmax[s] = at[s] + n;
            T = @max(T, at[s] + n);
        }
        if (R == 0) break;
        for (0..Model.G) |rank| m.write(rank, pos[0..R], base[0..R], lat[0..R], idx[0..R]);
        for (0..R) |r| {
            const s = who[r];
            var tok: [Model.G][Model.K]usize = undefined;
            var cnt: [Model.G]usize = @splat(0);
            for (0..2) |h| {
                // the window's T is the largest of its streams': a row's own stream may be far shorter (keys past its
                // position are masked by g <= p, and its slot's later rows hold nothing it may read)
                if (T > Model.K) try m.select(pos[r], base[r], Model.queryOf(s + 1, pos[r], 7), T, &tok, &cnt);
                together[s][pos[r] * 2 + h] = m.attend(pos[r], base[r], Model.queryOf(s + 1, pos[r], h), &tok, &cnt);
            }
        }
        for (0..S) |s| at[s] = tmax[s];
    }
    for (0..S) |s| try std.testing.expectEqualSlices(f64, alone[s], together[s]);
    // every slot holds exactly the rows its stream holds alone, and nothing of its neighbours
    for (0..S) |s| {
        var own = try Model.init(gpa, capacity, 1);
        defer own.deinit();
        const scratch = try gpa.alloc(f64, lens[s] * 2);
        defer gpa.free(scratch);
        try modelAlone(gpa, s + 1, lens[s], 3, capacity, scratch, &own);
        for (0..Model.G) |rank| {
            try std.testing.expectEqualSlices(f64, own.kc[rank], m.kc[rank][s * m.local ..][0..m.local]);
            try std.testing.expectEqualSlices(f64, own.ic[rank], m.ic[rank][s * m.local ..][0..m.local]);
        }
    }
}

test "a slot's rows under DCP: the owner's row, no two streams on one row, the last position inside the slot" {
    const cap: usize = 250_003;
    const local = localCapacity(cap, 4);
    try std.testing.expectEqual(@as(usize, 62_502), local);
    const a = slotOwner(cap - 1, 4, 0, local);
    const b = slotOwner(0, 4, 1, local);
    try std.testing.expect(a.row < b.row);
    try std.testing.expectEqual(@as(usize, 2), a.rank);
    const c = slotOwner(249_999, 4, 3, local);
    try std.testing.expectEqual(@as(usize, 3), c.rank);
    try std.testing.expectEqual(@as(usize, 3 * local + 62_499), c.row);
    try std.testing.expect(c.row < 4 * local);
    // DCP 1: the Phase 4b layout (slot * local + p)
    try std.testing.expectEqual(@as(usize, 2 * 100 + 7), slotOwner(7, 1, 2, 100).row);
}

test "cache bytes a rank: N streams with the cache split take what the one-stream shape of N x the tokens takes" {
    // GLM-5.3: 78 layers, 21 full indexers, a 576-wide latent, 128-wide index keys, an MTP layer with an indexer
    const tb = tokenBytes(78, 21, 576, 128, true, true);
    try std.testing.expectEqual(@as(usize, 96_640), tb);
    const gib: f64 = 1 << 30;
    const one_m = rankCacheBytes(tb, 1_000_000, 2, 1, 4);
    const two = rankCacheBytes(tb, 500_000, 2, 2, 4);
    const four = rankCacheBytes(tb, 250_000, 2, 4, 4);
    const base32 = rankCacheBytes(tb, 32_768, 2, 4, 1);
    const base64 = rankCacheBytes(tb, 65_536, 2, 2, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 22.50), @as(f64, @floatFromInt(one_m)) / gib, 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 11.80), @as(f64, @floatFromInt(base32)) / gib, 0.01);
    try std.testing.expectApproxEqAbs(@as(f64, 11.80), @as(f64, @floatFromInt(base64)) / gib, 0.01);
    // within a few rows a slot of the 1M shape
    try std.testing.expect(two >= one_m and two - one_m < 1 << 20);
    try std.testing.expect(four >= one_m and four - one_m < 1 << 20);
    // with DCP off the same tokens would take four times as much: what took four Sparks down at 4 x 140K
    try std.testing.expectApproxEqAbs(@as(f64, 50.4), @as(f64, @floatFromInt(rankCacheBytes(tb, 140_000, 2, 4, 1))) / gib, 0.1);
}

test "the boot check of several streams under DCP: a whole reserve, the token cap, a fitting context in the refusal" {
    const gib: u64 = 1 << 30;
    try std.testing.expect(streamsTokensOk(500_000, 2));
    try std.testing.expect(streamsTokensOk(250_000, 4));
    try std.testing.expect(streamsTokensOk(262_144, 4));
    try std.testing.expect(!streamsTokensOk(262_145, 4));
    try std.testing.expect(!streamsTokensOk(1_000_000, 2));
    const per = 6 * gib; // a stream's caches on this rank
    const fits = streamsFit(per, 4, 33 * gib, 8);
    try std.testing.expect(fits.ok);
    try std.testing.expectEqual(24 * gib, fits.need);
    // cacheFit would halve the reserve when little is free; this one never does
    const tight = streamsFit(per, 4, 31 * gib, 8);
    try std.testing.expect(!tight.ok);
    const fit = tight.fit_tokens(250_003);
    try std.testing.expect(fit < 250_003 and fit > 230_000);
    try std.testing.expectEqual(@as(u64, 0), streamsFit(per, 4, 7 * gib, 8).fit_tokens(250_003));
}
