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
