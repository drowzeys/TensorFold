//! Full GLM-5.3's concurrent streams in Zig (Phase 4b, `--parallel N`): multi.py's GlmMultiDecoder - up to N requests
//! decoded together, MTP drafts, every stream's reply token-identical to the same request alone.
//!
//! Each stream owns a slot of the Runner's caches (fw.State.initSlots: N streams' rows back to back). A decode round
//! packs every live stream's [pending token, drafts] into one verify window (up to N x (k + 1) rows) of decode-class
//! buffers (`vb`: fused.Buffers(decode=True)), and the position-dependent kernels read each row's position and cache
//! base from int32 device tables (Forward.base, cuda_triton *Rows: fused.Rows), so a row computes exactly what it
//! computes alone. The MTP chain runs batched: the first step takes every MTP stream's backlog rows (the target
//! hiddens of the rows its last round kept, `bh` / `bt`), steps 2..k one row a drafting stream, each step's drafts
//! scattered into the verify window's ids on the device. Windows and steps are CUDA graphs keyed by shape (rows,
//! streams, MTP step, key bucket, pick); the tables are static buffers, so one graph serves every position mix.
//!
//! Prompts fill one chunk a round between decode rounds (oldest prompt first), with the chunks a lone request takes
//! (Runner.prefillChunk on the stream's slot view: the one-stream kernels on its rows); after a round's fill, queued
//! prompts whose rest fits TF_GLM53_QUICK_ROWS rows together fill to their first token (TF_GLM53_BATCH_FILL). Not
//! ported: TF_GLM53_FILL_LAYERS (a long chunk paused between layers while others decode): a chunk here always fills
//! whole - the same bits, a longer stall for the decoding streams while a long prompt fills.
//!
//! Draft cut (TF_GLM53_DRAFT_CUT 0.6 from TF_GLM53_DRAFT_CUT_STREAMS 4 streams): after the MTP chain every rank
//! computes each stream's chain probability from the gathered pick logits and log-sum-exps (glm53_cut_stats) and
//! drops the drafts past the cut from the verify window before it runs. Drafts only propose: replies are the same.
//!
//! Picks: greedy windows take the device argmax head; a window with a sampled stream takes the "local" head and each
//! rank's candidates (the device top-k, one gather): sampled rows draw with their stream's keyed sampling at the
//! row's position, greedy rows take the best candidate (the gathered argmax). Every rank computes the same picks, so
//! the followers need no results message: rank 0's next plan carries each stream's position, pending token and backlog
//! and a follower that disagrees stops (ranks out of step).
//!
//! Prompt reuse (TF_GLM53_PROMPT_REUSE=1, prefixes.SlotReuse): a stream's kept states stay in its slot's rows
//! (cuda_reuse.Store in slot mode); a new request resumes from one in place (that slot free) or from a copy of its rows
//! into its own slot (that slot busy: a decoding stream never writes rows before its prompt's end); a slot taking
//! another conversation forgets the states it overwrites.
//!
//! Roles: the lead (rank 0 of a server, every rank of tf-glm53-generate --mode 4b) decides admissions, fills, rounds
//! and endings; a server's followers apply rank 0's messages (cuda_engine / native/glm53_cuda.zig carry them).
//!
//! Phase 4c: DFlash2 and DSpark streams (cuda_mdraft.zig; multi.py's DFLASH mode, and DSpark the same way): a drafter
//! stream's round asks its drafter for up to min(the drafter's most, the window's share - 1, its depth setting) drafts
//! (4 streams: 7), every drafting stream's block in one pass, each stream's host stage on its own; its window rows are
//! [pending, drafts] from the host; after the picks every drafter stream's kept rows' taps (vb.taps) extend its slot's
//! drafter context, and each prompt chunk's taps do too. From TF_GLM53_DRAFT_CUT_STREAMS streams their confidence floor
//! rises to TF_GLM53_DRAFT_CUT_DFLASH (default: TF_GLM53_DRAFT_CUT). Verify windows hold N x max(k + 1, the drafters'
//! rows) rows. A drafter stream's kept prompt state keeps its slot's ring window, and a stream resumed from it gets the
//! window in its own slot (prefixes.slot_ring_window / put_slot_ring_window). Not here: auto mode (per-stream MTP /
//! DFlash2 by rate), copy drafts, DCP (refused at boot).
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const smp = @import("sampling.zig");
const W = @import("cuda_weights.zig");
const rn = @import("cuda_runner.zig");
const rz = @import("cuda_reuse.zig");
const draft = @import("cuda_draft.zig");
const mdraft = @import("cuda_mdraft.zig");

/// The most streams a decoder takes (--parallel); decode windows hold at most fw.decode_rows rows anyway.
pub const max_streams: usize = 16;
/// The most MTP drafts a round (the tables' steps).
pub const max_k: usize = 8;
/// Ints a round's plan carries a stream (multi.ITEM).
pub const item_words: usize = 9;

/// A stream's drafts (multi.SERIAL / MTP / DFLASH; 3 is Python's AUTO, not ported; 4: DSpark, Phase 4c's own).
pub const Mode = enum(u8) { serial = 0, mtp = 1, dflash = 2, dspark = 4 };
/// A stream's drafts this round (multi.NONE / ARM_M / ARM_F: its drafter's, DFlash2 or DSpark by its mode).
pub const arm_none: u32 = 0;
pub const arm_mtp: u32 = 1;
pub const arm_draft: u32 = 2;

/// A request's mode from the engine's draft mode (a drafted request; "draft": false is serial).
pub fn modeOf(m: rn.Mode) Mode {
    return switch (m) {
        .mtp => .mtp,
        .dflash => .dflash,
        .dspark => .dspark,
    };
}

/// A plan item's mode word as a Mode (null: not one).
pub fn modeOfWord(w: u32) ?Mode {
    return switch (w) {
        0 => .serial,
        1 => .mtp,
        2 => .dflash,
        4 => .dspark,
        else => null,
    };
}

/// The drafter a mode drafts with (the runner's order: 0 DFlash2, 1 DSpark); null: none.
pub fn drafterOf(m: Mode) ?usize {
    return switch (m) {
        .dflash => 0,
        .dspark => 1,
        .serial, .mtp => null,
    };
}

fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    return if (t.len == 0) null else t;
}

/// The concurrent decoder's settings (every rank the same: they are in the boot's settings digest).
pub const Settings = struct {
    /// TF_GLM53_DRAFT_CUT (0: off) and TF_GLM53_DRAFT_CUT_STREAMS (at least 2)
    draft_cut: f64 = 0.6,
    cut_streams: usize = 4,
    /// TF_GLM53_DRAFT_CUT_DFLASH: a drafter stream's confidence floor from cut_streams streams (null: draft_cut)
    draft_cut_dflash: ?f64 = null,
    /// TF_GLM53_BATCH_FILL / TF_GLM53_QUICK_ROWS
    batch_fill: bool = true,
    quick_rows: usize = 1024,
    /// TF_GLM53_PRECAPTURE: prewarm also captures every verify width with both picks in every key bucket
    precapture: bool = true,

    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        if (env("TF_GLM53_DRAFT_CUT")) |v| s.draft_cut = std.fmt.parseFloat(f64, v) catch return error.BadSetting;
        if (env("TF_GLM53_DRAFT_CUT_STREAMS")) |v| s.cut_streams = @max(2, std.fmt.parseInt(usize, v, 10) catch return error.BadSetting);
        if (env("TF_GLM53_DRAFT_CUT_DFLASH")) |v| s.draft_cut_dflash = std.fmt.parseFloat(f64, v) catch return error.BadSetting;
        if (env("TF_GLM53_BATCH_FILL")) |v| s.batch_fill = !std.mem.eql(u8, v, "0");
        if (env("TF_GLM53_QUICK_ROWS")) |v| s.quick_rows = std.fmt.parseInt(usize, v, 10) catch return error.BadSetting;
        if (env("TF_GLM53_PRECAPTURE")) |v| s.precapture = !std.mem.eql(u8, v, "0");
        if (s.draft_cut < 0) s.draft_cut = 0;
        return s;
    }

    /// multi.DRAFT_CUT_DFLASH: the drafter streams' confidence floor in a cut round (0: none).
    pub fn cutDFlash(s: Settings) f64 {
        return @max(0, s.draft_cut_dflash orelse s.draft_cut);
    }

    pub fn digest(s: Settings, h: *std.hash.Wyhash) void {
        const words = [_]u64{ @bitCast(s.draft_cut), s.cut_streams, @intFromBool(s.batch_fill), s.quick_rows, @intFromBool(s.precapture), @bitCast(s.cutDFlash()) };
        h.update(std.mem.asBytes(&words));
    }
};

/// N streams x (k + 1) rows: the widest verify window, refused past a decode window (multi.GlmMultiDecoder).
pub fn windowRows(n: usize, k: usize) !usize {
    return windowRowsWith(n, k, 0);
}

/// Phase 4c: N streams x max(k + 1, frows) rows (frows: the rows a drafter stream's window takes, 0: no drafter).
pub fn windowRowsWith(n: usize, k: usize, frows: usize) !usize {
    if (n == 0 or n > max_streams or k > max_k) return error.Invalid;
    const rows = n * @max(k + 1, frows);
    if (rows > fw.decode_rows) return error.WindowTooWide;
    return rows;
}

/// Phase 4c: the rows a drafter stream's window takes with these drafters at N streams (multi.frows: the widest of
/// them; 0 with none) - or the refusal of a drafter whose N blocks do not fit one pass.
pub fn drafterRows(n: usize, specs: []const draft.Spec) !usize {
    var fr: usize = 0;
    for (specs) |sp| {
        if (!mdraft.fits(sp, n)) return error.DrafterTooWide;
        fr = @max(fr, mdraft.frows(sp, n));
    }
    return fr;
}

// --------------------------------------------------------------------------------------------- round plans ---

/// One stream's part of a round (multi.py's plan item): (sid, slot, P, tok, m, mode, arm, room, sampled).
pub const Item = struct {
    sid: u32,
    slot: u32,
    P: u32, // the pending token's position
    tok: u32, // the pending token
    m: u32, // MTP backlog rows (the rows the last round kept)
    mode: u32,
    arm: u32,
    room: u32, // drafts asked this round
    sampled: u32,
};

/// Rank 0's round: its streams, the draft cut (0: off) and the drafter streams' confidence floor (0: none).
pub const Plan = struct {
    n: usize = 0,
    cut: f64 = 0,
    dcut: f64 = 0,
    items: [max_streams]Item = undefined,

    /// The plan as words: [n, cut's low bits, cut's high bits, dcut's low bits, dcut's high bits, items x item_words].
    pub fn encode(p: *const Plan, out: []u32) ![]u32 {
        const need = plan_head + p.n * item_words;
        if (out.len < need) return error.Invalid;
        const bits: u64 = @bitCast(p.cut);
        const dbits: u64 = @bitCast(p.dcut);
        out[0] = @intCast(p.n);
        out[1] = @truncate(bits);
        out[2] = @truncate(bits >> 32);
        out[3] = @truncate(dbits);
        out[4] = @truncate(dbits >> 32);
        for (p.items[0..p.n], 0..) |it, i| {
            const o = out[plan_head + i * item_words ..][0..item_words];
            o.* = .{ it.sid, it.slot, it.P, it.tok, it.m, it.mode, it.arm, it.room, it.sampled };
        }
        return out[0..need];
    }

    pub fn decode(words: []const u32) !Plan {
        if (words.len < plan_head) return error.OutOfStep;
        const n: usize = words[0];
        if (n > max_streams or words.len != plan_head + n * item_words) return error.OutOfStep;
        var p: Plan = .{ .n = n, .cut = @bitCast(@as(u64, words[1]) | (@as(u64, words[2]) << 32)), .dcut = @bitCast(@as(u64, words[3]) | (@as(u64, words[4]) << 32)) };
        for (0..n) |i| {
            const o = words[plan_head + i * item_words ..][0..item_words];
            p.items[i] = .{ .sid = o[0], .slot = o[1], .P = o[2], .tok = o[3], .m = o[4], .mode = o[5], .arm = o[6], .room = o[7], .sampled = o[8] };
        }
        return p;
    }
};

/// A plan's header words and the most words a plan takes (the server's ROUND message).
pub const plan_head: usize = 5;
pub const plan_words: usize = plan_head + max_streams * item_words;

/// Phase 4c: each plan item's drafts from its drafter (arm_draft items; the host stage's, every rank alike).
pub const Drafts = struct {
    n: [max_streams]usize = @splat(0),
    ids: [max_streams][fw.decode_rows]u32 = @splat(@splat(0)),
};

/// A round's host tables (multi._gpu): the verify window's rows (position, cache base, ids), each MTP step's rows
/// (position, base, the rows whose picks it drafts, where the drafts go in the verify window) and the backlog rows
/// step 1 gathers.
pub const Tables = struct {
    R: usize = 0, // verify rows
    M: usize = 0, // MTP backlog rows (step 1's rows)
    S: usize = 0, // drafting streams
    offs: [max_streams]usize = @splat(0), // each item's first verify row
    ds: [max_streams]usize = @splat(0), // each item's drafts
    drafting: [max_streams]usize = @splat(0), // the plan index of each drafting stream (step row order)
    vpos: [fw.decode_rows]i32 = @splat(0),
    vbase: [fw.decode_rows]i32 = @splat(0),
    vids: [fw.decode_rows]i64 = @splat(0),
    mpos: [max_k][fw.decode_rows]i32 = @splat(@splat(0)),
    mbase: [max_k][fw.decode_rows]i32 = @splat(@splat(0)),
    last: [max_k][max_streams]i64 = @splat(@splat(0)),
    vdst: [max_k][max_streams]i64 = @splat(@splat(0)),
    src: [fw.decode_rows]i64 = @splat(0),
    keep_end: ?usize = null, // max P of the MTP-keeping streams + k (the MTP steps' key range end)
    end: usize = 0, // max P + 1 + d (the verify window's key range end)
};

/// multi._gpu's tables for `plan` (k MTP drafts a round, BL backlog rows a slot, `local` cache rows a slot; `dd`
/// the drafter streams' drafts, null: none).
pub fn tables(plan: *const Plan, k: usize, BL: usize, local: usize, dd: ?*const Drafts) !Tables {
    var t: Tables = .{};
    var r: usize = 0;
    for (plan.items[0..plan.n], 0..) |it, i| {
        const d: usize = switch (it.arm) {
            arm_mtp => it.room,
            arm_draft => if (dd) |x| x.n[i] else 0,
            else => 0,
        };
        if (it.arm == arm_mtp and d > k) return error.Invalid;
        if (it.arm == arm_draft and d > it.room) return error.Invalid;
        if (r + 1 + d > fw.decode_rows) return error.WindowTooWide;
        t.offs[i] = r;
        t.ds[i] = d;
        for (0..1 + d) |j| {
            t.vpos[r + j] = @intCast(it.P + j);
            t.vbase[r + j] = @intCast(it.slot * local);
        }
        t.vids[r] = it.tok;
        if (it.arm == arm_draft) if (dd) |x| {
            for (0..d) |j| t.vids[r + 1 + j] = x.ids[i][j];
        };
        r += 1 + d;
        t.end = @max(t.end, it.P + 1 + d);
    }
    t.R = r;
    if (k == 0) return t;
    var M: usize = 0;
    var S: usize = 0;
    for (plan.items[0..plan.n], 0..) |it, q| {
        if (it.mode != @intFromEnum(Mode.mtp)) continue; // multi._keeps_mtp
        if (it.m == 0 or it.m > BL or it.m > it.P) return error.Invalid;
        for (0..it.m) |i| {
            if (M >= fw.decode_rows) return error.WindowTooWide;
            t.mpos[0][M] = @intCast(it.P - it.m + i);
            t.mbase[0][M] = @intCast(it.slot * local);
            t.src[M] = @intCast(it.slot * BL + i);
            M += 1;
        }
        t.keep_end = @max(t.keep_end orelse 0, it.P + k);
        if (it.arm != arm_mtp or it.room == 0) continue;
        const off = t.offs[q];
        t.drafting[S] = q;
        t.last[0][S] = @intCast(M - 1);
        t.vdst[0][S] = @intCast(off + 1);
        var j: usize = 2;
        while (j <= k) : (j += 1) {
            t.mpos[j - 1][S] = @intCast(it.P + j - 2);
            t.mbase[j - 1][S] = @intCast(it.slot * local);
            t.last[j - 1][S] = @intCast(S);
            t.vdst[j - 1][S] = @intCast(off + j);
        }
        S += 1;
    }
    t.M = M;
    t.S = S;
    return t;
}

const libm = struct {
    const exp = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "exp" });
    const log = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "log" });
};

/// multi._cut on the host: each drafting stream s (plan index t.drafting[s]) keeps its drafts while their chain
/// probability - the MTP head's probabilities of its drafts so far, multiplied - stays at or above `cut`.
/// `pick[j * N + s]`: step j's picked logit (the gathered maximum); `parts[r * rank_stride + j * N + s]`: rank r's
/// log-sum-exp over its vocabulary share (N: the decoder's streams, the tables' row pitch). The new drafts of every
/// plan item go to `out` (items that do not draft keep theirs).
pub fn cutDecide(t: *const Tables, k: usize, N: usize, cut: f64, pick: []const f32, parts: []const f32, ranks: usize, rank_stride: usize, out: *[max_streams]usize) void {
    out.* = t.ds;
    for (0..t.S) |s| {
        const q = t.drafting[s];
        var chain: f64 = 1.0;
        var keep: usize = 0;
        for (0..@min(k, t.ds[q])) |j| {
            var top: f64 = -std.math.inf(f64);
            for (0..ranks) |r| top = @max(top, @as(f64, parts[r * rank_stride + j * N + s]));
            var sum: f64 = 0;
            for (0..ranks) |r| sum += libm.exp(@as(f64, parts[r * rank_stride + j * N + s]) - top);
            const total = top + libm.log(sum);
            chain *= libm.exp(@min(0.0, @as(f64, pick[j * N + s]) - total));
            if (chain < cut) break;
            keep = j + 1;
        }
        out[q] = keep;
    }
}

/// multi._cut's close-up: the verify window with `ds` drafts an item - new row offsets, positions, bases and the old
/// rows each new row takes its id from. Returns the new row count.
pub fn compact(plan: *const Plan, t: *Tables, ds: *const [max_streams]usize, local: usize, src_rows: *[fw.decode_rows]usize) usize {
    var r: usize = 0;
    for (plan.items[0..plan.n], 0..) |it, i| {
        const off = t.offs[i];
        t.offs[i] = r;
        for (0..1 + ds[i]) |j| {
            src_rows[r] = off + j;
            t.vpos[r] = @intCast(it.P + j);
            t.vbase[r] = @intCast(it.slot * local);
            r += 1;
        }
        t.ds[i] = ds[i];
    }
    t.R = r;
    return r;
}

/// A round's outcome for one stream: drafts kept (n) and the tokens emitted (n + 1: the picks along them).
pub const Result = struct {
    n: usize = 0,
    len: usize = 0,
    emit: [fw.decode_rows]u32 = @splat(0),

    pub fn tokens(x: *const Result) []const u32 {
        return x.emit[0..x.len];
    }
};

/// The kept drafts of a row run: drafts[0..d] against the target's picks along them (multi._picks' n).
pub fn accepted(drafts: []const u32, picks: []const u32) usize {
    var n: usize = 0;
    while (n < drafts.len and n + 1 < picks.len and drafts[n] == picks[n]) n += 1;
    return n;
}

/// A greedy row of a sampled window: the best of every rank's candidates - the largest value, the lowest id among
/// equal ones (the argmax head's choice: each rank's first maximum, the lowest rank among equal maxima).
pub fn bestCandidate(cands: []const smp.Cand) u32 {
    var best: usize = 0;
    for (cands, 0..) |c, i| {
        const b = cands[best];
        if (c.v > b.v or (c.v == b.v and c.id < b.id)) best = i;
    }
    return @intCast(cands[best].id);
}

// ------------------------------------------------------------------------------------------------- streams ---

/// What the lead's caller hears from a stream (rank 0's server: the reply's events; returns true to stop it).
pub const Hook = struct {
    ctx: *anyopaque,
    tokens: *const fn (ctx: *anyopaque, toks: []const u32) bool,
};

/// One request on every rank (streams.Stream as multi.py uses it).
pub const Stream = struct {
    sid: u32,
    slot: usize,
    mode: Mode,
    sampling: ?smp.Sampling,
    count: usize, // tokens to make, the first one included
    prompt: []u32,
    eos: []u32, // tokens that end it (empty: ignore_eos)
    named: []u32, // rank 0's keep points: n * 2 + shared
    chunks: [][2]usize,
    ci: usize = 0,
    begin: usize = 0, // prompt tokens resumed from a kept state
    carry0: bool = false, // a resumed prompt: its MTP carry waits in the decoder's carry0 slot row
    head0: bool = false, // a replay: its kept head row waits in head0
    started: bool = false,
    P: usize = 0,
    tok: u32 = 0,
    m: usize = 1,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    eos_hit: bool = false,
    stopped: bool = false, // the hook asked
    rounds: usize = 0,
    drafted: usize = 0,
    accepted_n: usize = 0,
    min_rows: usize = 0,
    prefill_s: f64 = 0,
    t_admit: i128 = 0,
    t_first: i128 = 0,
    t_start: i128 = 0,
    t_end: i128 = 0,
    hook: ?Hook = null,
    // the lead's per-request timing (rank 0's done line): its rounds' applyRound time, the host time before each of
    // them since the previous round ended (hooks, admissions, fills, the round's message), the graph replays and
    // eager runs (misses) in its rounds
    round_ns: u64 = 0,
    gap_ns: u64 = 0,
    graph_runs: u64 = 0,
    eager_runs: u64 = 0,

    fn isEnd(s: *const Stream, t: u32) bool {
        return std.mem.indexOfScalar(u32, s.eos, t) != null;
    }

    pub fn decodeS(s: *const Stream) f64 {
        if (s.t_end <= s.t_start) return 0;
        return @as(f64, @floatFromInt(s.t_end - s.t_start)) / 1e9;
    }

    pub fn tokS(s: *const Stream) f64 {
        const d = s.decodeS();
        return if (d > 0 and s.out.items.len > 1) @as(f64, @floatFromInt(s.out.items.len - 1)) / d else 0;
    }
};

/// What a lead's admission needs (every rank applies the same: rank 0's choices carried to the followers).
pub const Admit = struct {
    sid: u32,
    slot: usize,
    mode: Mode,
    sampling: ?smp.Sampling = null,
    count: usize,
    prompt: []const u32,
    eos: []const u32 = &.{},
    begin: usize = 0,
    src: ?usize = null, // the slot the resumed rows are copied from (null: in place / nothing resumed)
    stops: []const usize = &.{},
    keeps: []const u32 = &.{},
    flush: bool = false,
    hook: ?Hook = null,
};

/// SlotReuse.choose's answer (rank 0): the slot, the resume point and where its rows are, the prompt's cuts and keeps.
pub const Choice = struct {
    slot: usize,
    begin: usize = 0,
    src: ?usize = null,
    stops: []usize = &.{},
    keeps: []u32 = &.{},

    pub fn deinit(c: Choice, gpa: std.mem.Allocator) void {
        gpa.free(c.stops);
        gpa.free(c.keeps);
    }
};

/// The lead's caller, told when a round's GPU work is queued and the host is about to wait for it (rank 0's server:
/// the previous round's tokens go to the replies then, off the path between a round's picks and the next round).
pub const Idle = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque) void,
};

/// What the lead tells the followers between its decisions (rank 0's server sends them; the CLI needs none).
pub const Link = struct {
    ctx: *anyopaque,
    fill: *const fn (ctx: *anyopaque, sid: u32) anyerror!void,
    round: *const fn (ctx: *anyopaque, plan: *const Plan) anyerror!void,
};

/// The device tables' byte offsets (every sub-table 16-byte aligned: the ROWS kernels' BASE / POS pointers).
const Layout = struct {
    vpos: usize = 0,
    vbase: usize = 0,
    mpos: [max_k]usize = @splat(0),
    mbase: [max_k]usize = @splat(0),
    last: [max_k]usize = @splat(0),
    vdst: [max_k]usize = @splat(0),
    src: usize = 0,
    plan_end: usize = 0,
    cidx: usize = 0,
    cdst: usize = 0,
    ctok: usize = 0,
    commit_end: usize = 0,
    lse: usize = 0, // fp32 [k, N]: this rank's log-sum-exps (glm53_cut_stats)
    top: usize = 0, // fp32 [k, N]: the picked drafts' logits
    lse_all: usize = 0, // fp32 [world, pad4(k N)]: every rank's
    total: usize = 0,

    fn make(rows: usize, N: usize, k: usize, world: usize) Layout {
        var l: Layout = .{};
        var at: usize = 0;
        const Next = struct {
            fn take(p: *usize, bytes: usize) usize {
                const o = p.*;
                p.* = std.mem.alignForward(usize, o + @max(bytes, 4), 16);
                return o;
            }
        };
        l.vpos = Next.take(&at, rows * 4);
        l.vbase = Next.take(&at, rows * 4);
        for (0..k) |j| {
            l.mpos[j] = Next.take(&at, rows * 4);
            l.mbase[j] = Next.take(&at, rows * 4);
            l.last[j] = Next.take(&at, N * 8);
            l.vdst[j] = Next.take(&at, N * 8);
        }
        l.src = Next.take(&at, rows * 8);
        l.plan_end = at;
        l.cidx = Next.take(&at, rows * 8);
        l.cdst = Next.take(&at, rows * 8);
        l.ctok = Next.take(&at, rows * 8);
        l.commit_end = at;
        const kn = @max(k, 1) * N;
        l.lse = Next.take(&at, kn * 4);
        l.top = Next.take(&at, kn * 4);
        l.lse_all = Next.take(&at, world * std.mem.alignForward(usize, kn, 4) * 4);
        l.total = at;
        return l;
    }
};

/// The pinned staging's extra regions past the tables' mirror.
const PinLayout = struct { vids: usize, rd: usize, cand: usize, cut: usize, total: usize };

fn tIndex(T: ?usize) u64 {
    const t = T orelse return 0;
    return std.math.log2_int(usize, t) + 1;
}

/// TF_GLM53_ROUND_PROFILE=1: each round's phases timed on every rank (the stream synchronized at each phase's end,
/// so the phases do not overlap), averaged over windows of rounds with the same streams and picks and logged per rank.
pub const Prof = struct {
    pub const names = [_][]const u8{ "propose", "mtp", "cut", "verify", "picks", "commit" };
    on: bool = false,
    key: u64 = 0,
    n: u64 = 0,
    rows: u64 = 0,
    acc: [names.len]u64 = @splat(0),
    total: u64 = 0,
    wait: u64 = 0, // the host time before the round since the previous one ended (rank 0: plan, hooks, message; others: the message)
    last: i128 = 0,
};

/// The concurrent decoder on one rank.
pub const Multi = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    r: *rn.Runner,
    w: *const W.Weights,
    set: Settings,
    lead: bool,
    N: usize,
    k: usize,
    BL: usize,
    MR: usize,
    rows: usize,
    local: usize,
    D: usize,
    V: usize, // vocab_part
    hv: usize, // the MTP head's logits a row (the draft head's rows, else vocab_part)
    world: usize,
    cut_on: bool,
    vb: fw.Buffers,
    mb: ?fw.Buffers,
    bh: cuda.DeviceBuffer, // bf16 [MR, D]: each slot's MTP backlog hiddens
    bt: cuda.DeviceBuffer, // int64 [MR]: ... and tokens
    tab: cuda.DeviceBuffer,
    carry0: cuda.DeviceBuffer, // bf16 [N, D]: a resumed prompt's MTP carry, until its first fill step
    head0: cuda.DeviceBuffer, // fp32 [N, V]: a replayed prompt's kept head row
    pin: cuda.HostBuffer,
    lay: Layout,
    play: PinLayout,
    views: []fw.State,
    // Phase 4c: the drafters for concurrent streams (0 DFlash2, 1 DSpark; cuda_mdraft.zig) and the rows a drafter
    // stream's window takes (multi.frows; 0: none loaded)
    md: [2]?*mdraft.MultiDrafter = .{ null, null },
    frows: usize = 0,
    busy: [max_streams]bool = @splat(false),
    streams: std.ArrayList(*Stream) = .empty, // decoding, in the order they started
    filling: std.ArrayList(*Stream) = .empty, // admitted, prompt chunks left (oldest first)
    next_sid: u32 = 0,
    // prompt reuse in the slots (prefixes.SlotReuse), when on
    store: ?rz.Store = null,
    pool: ?*rz.Pool = null,
    plan_rz: rz.Plan = .{ .ids = .{} },
    // counters
    rounds: usize = 0,
    cuts: usize = 0,
    widest: usize = 0,
    capture_s: f64 = 0,
    broken: bool = false,
    // the lead: when its last round ended (0: no round since the streams went idle), and its idle hook (Idle)
    t_round_end: i128 = 0,
    idle: ?Idle = null,
    pf: Prof = .{},

    pub fn init(gpa: std.mem.Allocator, io: std.Io, r: *rn.Runner, n: usize, set: Settings, lead: bool) !*Multi {
        const w = r.w;
        const c = w.config;
        const k = r.set.k;
        if (r.set.slots != n or r.set.dcp != 1 or !r.set.served) return error.Invalid;
        if (r.copy_on) return error.CopiesNeedOneStream;
        // Phase 4c: the runner's drafters serve the streams too (MultiDrafter); their rows widen the windows
        const bases = [2]?*draft.Drafter{ r.dflash, r.dspark };
        var specs: [2]draft.Spec = undefined;
        var n_specs: usize = 0;
        for (bases) |maybe| {
            const bd = maybe orelse continue;
            specs[n_specs] = bd.sp;
            n_specs += 1;
        }
        const fr = try drafterRows(n, specs[0..n_specs]);
        const rows = try windowRowsWith(n, k, fr);
        const mu = try gpa.create(Multi);
        errdefer gpa.destroy(mu);
        mu.* = undefined;
        mu.gpa = gpa;
        mu.io = io;
        mu.r = r;
        mu.w = w;
        mu.set = set;
        mu.lead = lead;
        mu.N = n;
        mu.k = k;
        mu.frows = fr;
        mu.md = .{ null, null };
        mu.BL = k + 1;
        mu.MR = n * (k + 1);
        mu.rows = rows;
        mu.local = r.st.local;
        mu.D = c.hidden;
        mu.V = w.vocab_part;
        mu.hv = if (w.draft) |dh| dh.n else w.vocab_part;
        mu.world = w.world;
        mu.cut_on = set.draft_cut > 0 and k > 0;
        mu.busy = @splat(false);
        mu.streams = .empty;
        mu.filling = .empty;
        mu.next_sid = 0;
        mu.store = null;
        mu.pool = null;
        mu.plan_rz = .{ .ids = .{} };
        mu.rounds = 0;
        mu.cuts = 0;
        mu.widest = 0;
        mu.capture_s = 0;
        mu.broken = false;
        mu.t_round_end = 0;
        mu.idle = null;
        mu.pf = .{};
        if (env("TF_GLM53_ROUND_PROFILE")) |v| mu.pf.on = !std.mem.eql(u8, v, "0");
        mu.lay = Layout.make(rows, n, k, w.world);
        const cols = tri.bucket(r.capacity, r.topk) orelse 1;
        const shared = r.f.shared_experts;
        mu.vb = try fw.Buffers.initWith(gpa, r.d, w, rows, cols, .{ .decode = true, .shared_experts = shared, .taps = r.f.taps.n });
        errdefer mu.vb.deinit();
        mu.mb = null;
        if (k > 0) mu.mb = try fw.Buffers.initWith(gpa, r.d, w, @max(mu.MR, rows), cols, .{ .decode = true, .mtp = true, .shared_experts = shared });
        errdefer if (mu.mb) |*b| b.deinit();
        mu.bh = try zeroed(r.d, @max(mu.MR, 1) * mu.D * 2);
        errdefer mu.bh.free();
        mu.bt = try zeroed(r.d, @max(mu.MR, 1) * 8);
        errdefer mu.bt.free();
        mu.tab = try zeroed(r.d, mu.lay.total);
        errdefer mu.tab.free();
        mu.carry0 = try zeroed(r.d, n * mu.D * 2);
        errdefer mu.carry0.free();
        mu.head0 = try zeroed(r.d, n * mu.V * 4);
        errdefer mu.head0.free();
        const words = std.mem.alignForward(usize, mu.vb.cand_words, 4);
        var at = std.mem.alignForward(usize, mu.lay.total, 256);
        const vids_at = at;
        at = std.mem.alignForward(usize, at + rows * 8, 256);
        const rd_at = at;
        at = std.mem.alignForward(usize, at + 2 * rows * 8, 256);
        const cand_at = at;
        at = std.mem.alignForward(usize, at + w.world * words * 4, 256);
        const cut_at = at;
        const kn = @max(k, 1) * n;
        at = std.mem.alignForward(usize, at + (kn + w.world * std.mem.alignForward(usize, kn, 4)) * 4, 256);
        mu.play = .{ .vids = vids_at, .rd = rd_at, .cand = cand_at, .cut = cut_at, .total = at };
        mu.pin = try cuda.HostBuffer.alloc(r.d, mu.play.total);
        errdefer mu.pin.free();
        mu.views = try gpa.alloc(fw.State, n);
        var made: usize = 0;
        errdefer {
            for (mu.views[0..made]) |*v| v.freeView();
            gpa.free(mu.views);
        }
        for (0..n) |s| {
            mu.views[s] = try r.st.slotView(w, s);
            made += 1;
        }
        errdefer {
            for (mu.md) |maybe| {
                if (maybe) |x| x.deinit();
            }
        }
        for (bases, 0..) |maybe, which| {
            const bd = maybe orelse continue;
            mu.md[which] = try mdraft.MultiDrafter.init(gpa, r, bd, which, n);
        }
        return mu;
    }

    pub fn deinit(mu: *Multi) void {
        mu.r.f.s.synchronize() catch {};
        mu.dropAll();
        mu.streams.deinit(mu.gpa);
        mu.filling.deinit(mu.gpa);
        if (mu.store) |*st| st.deinit();
        if (mu.pool) |p| {
            p.deinit();
            mu.gpa.destroy(p);
        }
        for (mu.md) |maybe| {
            if (maybe) |x| x.deinit();
        }
        for (mu.views) |*v| v.freeView();
        mu.gpa.free(mu.views);
        mu.pin.free();
        mu.head0.free();
        mu.carry0.free();
        mu.tab.free();
        mu.bt.free();
        mu.bh.free();
        if (mu.mb) |*b| b.deinit();
        mu.vb.deinit();
        mu.gpa.destroy(mu);
    }

    fn zeroed(d: *const cuda.Driver, bytes: usize) !cuda.DeviceBuffer {
        var b = try cuda.DeviceBuffer.alloc(d, @max(bytes, 16));
        errdefer b.free();
        try b.fill8(0, null);
        return b;
    }

    fn ops(mu: *const Multi) kern.Ops {
        return .{ .k = mu.r.f.k, .s = mu.r.f.s };
    }

    pub fn now(mu: *const Multi) i128 {
        return @intCast(std.Io.Clock.awake.now(mu.io).toNanoseconds());
    }

    fn pinSlice(mu: *const Multi, comptime T: type, off: usize, n: usize) []T {
        const p: [*]T = @ptrCast(@alignCast(mu.pin.bytes.ptr + off));
        return p[0..n];
    }

    /// The caches' bytes a stream holds (multi.nbytes_per_stream: the State's share, the drafters' rings).
    pub fn bytesPerStream(mu: *const Multi) usize {
        var b = fw.State.bytesForDcp(mu.w, mu.r.capacity, 1);
        for (mu.md) |maybe| {
            if (maybe) |x| b += x.deviceBytes() / mu.N;
        }
        return b;
    }

    /// The concurrent drafter of a mode (null: MTP / serial, or none loaded).
    fn mdOf(mu: *const Multi, m: Mode) ?*mdraft.MultiDrafter {
        const which = drafterOf(m) orelse return null;
        return mu.md[which];
    }

    /// The drafts a drafter stream may ask a round (multi._dflash_cfg's depth): the drafter's most, its window rows
    /// less the pending token, the settings' depth (TF_GLM53_DFLASH_DEPTH / TF_GLM53_DSPARK_DEPTH).
    fn depthOf(mu: *const Multi, which: usize) usize {
        const x = mu.md[which] orelse return 0;
        const b = x.base;
        const want = if (b.sp.kind == .dspark) b.dset.depth else b.fset.depth;
        return @min(@min(b.sp.maxDepth(), mdraft.frows(b.sp, mu.N) - 1), want);
    }

    /// A drafter stream's confidence floor this round: the settings' (TF_GLM53_DFLASH_CONFIDENCE /
    /// TF_GLM53_DSPARK_CONFIDENCE), raised to the plan's `dcut` in a cut round.
    fn confOf(mu: *const Multi, which: usize, dcut: f64) f64 {
        const b = (mu.md[which] orelse return 0).base;
        const c = if (b.sp.kind == .dspark) b.dset.confidence else b.fset.confidence;
        return if (dcut > 0) @max(c, dcut) else c;
    }

    /// Streams admitted and not finished (multi.live).
    pub fn live(mu: *const Multi) usize {
        return mu.streams.items.len + mu.filling.items.len;
    }

    pub fn freeSlots(mu: *const Multi) usize {
        var n: usize = 0;
        for (mu.busy[0..mu.N]) |b| n += @intFromBool(!b);
        return n;
    }

    pub fn find(mu: *const Multi, sid: u32) ?*Stream {
        for (mu.streams.items) |s| if (s.sid == sid) return s;
        for (mu.filling.items) |s| if (s.sid == sid) return s;
        return null;
    }

    // ------------------------------------------------------------------------------------- prompt reuse ---

    /// prefixes.SlotReuse in the slots (every rank alike, the budget the ranks' least).
    pub fn enableReuse(mu: *Multi, plan: rz.Plan, budget: u64, entries: usize, loose: bool) !void {
        if (mu.store != null) return error.Invalid;
        const pool = try mu.gpa.create(rz.Pool);
        errdefer mu.gpa.destroy(pool);
        pool.* = .{ .gpa = mu.gpa, .d = mu.r.d };
        mu.pool = pool;
        const c = mu.w.config;
        mu.store = rz.Store.init(mu.gpa, budget, entries, loose, c.hidden * 2, c.vocab * 4, pool.release());
        try mu.store.?.setSlots(mu.N);
        mu.plan_rz = plan;
    }

    /// Every rank: forget every kept state (REUSE_FLUSH).
    pub fn flushReuse(mu: *Multi) !void {
        const st = if (mu.store) |*x| x else return;
        try mu.r.f.s.synchronize();
        st.clear();
        try mu.pool.?.trim(0, mu.r.f.s);
    }

    /// Rank 0: SlotReuse.choose - (slot, resume point, its rows' slot, cuts, keeps) for a request (`resume_` false:
    /// "draft": false, the cold reference: cut alike, nothing resumed or kept). No reuse: the first free slot.
    pub fn choose(mu: *Multi, prompt: []const u32, resume_: bool) !Choice {
        var first_free: ?usize = null;
        for (mu.busy[0..mu.N], 0..) |b, i| if (!b) {
            first_free = i;
            break;
        };
        const free0 = first_free orelse return error.NoFreeSlot;
        const st = if (mu.store) |*x| x else return .{ .slot = free0 };
        const pts = try mu.plan_rz.points(mu.gpa, prompt);
        defer mu.gpa.free(pts);
        const hit = if (resume_) st.best(prompt, pts, 0) else null;
        var slot = free0;
        var src: ?usize = null;
        if (hit != null and !mu.busy[hit.?.slot]) {
            slot = hit.?.slot;
        } else {
            // the free slot whose states are worth least (none first): min (newest tick, slot)
            var best_tick: u64 = std.math.maxInt(u64);
            for (mu.busy[0..mu.N], 0..) |b, i| {
                if (b) continue;
                const t = st.slotTick(i);
                if (t < best_tick) {
                    best_tick = t;
                    slot = i;
                }
            }
            if (hit) |h| src = h.slot;
        }
        const begin = if (hit) |h| h.n() else 0;
        const c = try rz.cutAndKeep(mu.gpa, mu.plan_rz, st, prompt, pts, begin, resume_);
        return .{ .slot = slot, .begin = begin, .src = src, .stops = c.stops, .keeps = c.keeps };
    }

    /// prefixes.copy_slot_rows: slot src's rows of a state of n tokens into slot dst (target < n, MTP < n - 1).
    fn copySlotRows(mu: *Multi, src: usize, dst: usize, n: usize) !void {
        if (n == 0 or src == dst) return;
        const c = mu.w.config;
        const o = mu.ops();
        const st = &mu.r.st;
        const lat = c.latentWidth() * 2;
        const idx = c.index_dim * 2;
        for (st.kc) |p| try o.copy(p + dst * mu.local * lat, p + src * mu.local * lat, n * lat);
        for (st.ic) |p| if (p != 0) try o.copy(p + dst * mu.local * idx, p + src * mu.local * idx, n * idx);
        if (st.mkc != 0 and n > 1) {
            try o.copy(st.mkc + dst * mu.local * lat, st.mkc + src * mu.local * lat, (n - 1) * lat);
            if (st.mic != 0) try o.copy(st.mic + dst * mu.local * idx, st.mic + src * mu.local * idx, (n - 1) * idx);
        }
    }

    /// SlotReuse.admit (every rank): forget the slot's states this request overwrites, bring the resumed state's rows
    /// into it, its carry (and, for a replay, its head row) into the slot's waiting rows.
    fn admitReuse(mu: *Multi, s: *Stream, a: Admit) !?*rz.Kept {
        const st = if (mu.store) |*x| x else {
            if (a.begin != 0) return error.ReuseOff;
            return null;
        };
        var hit: ?*rz.Kept = null;
        if (a.begin > 0) {
            hit = st.namedIn(a.prompt, a.begin, a.src orelse a.slot) orelse {
                if (!@import("builtin").is_test) std.log.err("rank {d}: no kept prompt state of the {d} tokens rank 0 resumes from in slot {d}", .{ mu.w.rank, a.begin, a.src orelse a.slot });
                return error.ReuseOutOfStep;
            };
        }
        var pts: []const usize = &.{};
        defer if (pts.len > 0) mu.gpa.free(pts);
        var same: ?rz.Same = null;
        if (a.src != null and hit != null) {
            pts = try mu.plan_rz.points(mu.gpa, a.prompt);
            same = .{ .mode = hit.?.mode, .points = pts, .begin = a.begin };
        }
        const over = try st.overwrittenSlot(mu.gpa, a.slot, a.prompt, a.begin, same);
        defer mu.gpa.free(over);
        for (over) |e| {
            if (hit != null and e == hit.?) continue;
            st.drop(e);
        }
        if (a.src) |from| try mu.copySlotRows(from, a.slot, a.begin);
        try st.setLiveSlot(a.slot, a.prompt[0..a.begin]);
        const h = hit orelse return null;
        st.touch(h);
        const o = mu.ops();
        try o.copy(mu.carry0.ptr + a.slot * mu.D * 2, h.carry.ptr, mu.D * 2);
        s.carry0 = true;
        if (a.begin == a.prompt.len) {
            const head = h.head orelse return error.ReplayWithoutHead;
            try o.copy(mu.head0.ptr + a.slot * mu.V * 4, head.ptr, mu.V * 4);
            s.head0 = true;
        }
        return h;
    }

    /// SlotReuse.keep (every rank): the stream's state at prompt position n (its chunks reached n) in its slot.
    fn keepState(mu: *Multi, s: *Stream, n: usize, head: ?u64) !void {
        const st = if (mu.store) |*x| x else return;
        const pool = mu.pool.?;
        const e = try mu.gpa.create(rz.Kept);
        e.* = .{ .ids = try mu.gpa.dupe(u32, s.prompt[0..n]), .slot = s.slot, .shared = sharedOf(s.named, n) };
        const o = mu.ops();
        e.carry = try pool.get(mu.D * 2);
        try o.copy(e.carry.ptr, mu.r.carry.ptr, mu.D * 2);
        if (head) |lg| {
            const h = try pool.get(mu.V * 4);
            e.head = h;
            try o.copy(h.ptr, lg, mu.V * 4);
        }
        // multi._keep: a drafter stream's ring window at n (its prompt's taps are in: feedPrompt came first)
        if (mu.mdOf(s.mode)) |x| {
            const rg = try pool.get(x.windowBytes(n));
            e.ring = rg;
            try x.saveWindow(s.slot, n, rg.ptr);
        }
        if (!st.remember(e, &.{})) st.destroy(e);
    }

    fn sharedOf(named: []const u32, n: usize) bool {
        for (named) |x| if (x >> 1 == n) return (x & 1) != 0;
        return false;
    }

    fn isNamed(named: []const u32, n: usize) bool {
        for (named) |x| if (x >> 1 == n) return true;
        return false;
    }

    // ----------------------------------------------------------------------------------------- admission ---

    /// Every rank: multi.admit / _queue - a request queued in its slot (rank 0 chose it), its prompt to fill a chunk
    /// a round.
    pub fn admit(mu: *Multi, a: Admit) !*Stream {
        if (mu.broken) return error.RanksOutOfStep;
        if (a.slot >= mu.N or mu.busy[a.slot]) return error.NoFreeSlot;
        const L = a.prompt.len;
        if (L == 0 or L + 1 >= mu.r.capacity - mu.k) return error.PromptTooLong;
        if (a.count == 0) return error.Invalid;
        const mode: Mode = switch (a.mode) {
            .serial => .serial,
            .mtp => if (mu.k == 0) .serial else .mtp,
            .dflash, .dspark => if (mu.mdOf(a.mode) != null) a.mode else return error.NoDrafter,
        };
        if (a.flush) try mu.flushReuse();
        const s = try mu.gpa.create(Stream);
        errdefer mu.gpa.destroy(s);
        s.* = .{ .sid = a.sid, .slot = a.slot, .mode = mode, .sampling = a.sampling, .count = a.count, .prompt = &.{}, .eos = &.{}, .named = &.{}, .chunks = &.{}, .begin = a.begin, .hook = a.hook, .t_admit = mu.now() };
        errdefer s.out.deinit(mu.gpa);
        s.prompt = try mu.gpa.dupe(u32, a.prompt);
        errdefer mu.gpa.free(s.prompt);
        s.eos = try mu.gpa.dupe(u32, a.eos);
        errdefer mu.gpa.free(s.eos);
        s.named = try mu.gpa.dupe(u32, a.keeps);
        errdefer mu.gpa.free(s.named);
        const rows = mu.r.set.prompt_rows;
        const short = mu.r.set.prompt_rows_short;
        if (a.begin >= L) {
            if (a.begin != L) return error.Invalid;
            s.chunks = try mu.gpa.alloc([2]usize, 1); // (L, L): a replay - the kept head row, nothing filled
            s.chunks[0] = .{ L, L };
        } else {
            s.chunks = try rn.cutChunks(mu.gpa, a.begin, L, a.stops, rows, short);
        }
        errdefer mu.gpa.free(s.chunks);
        const hit = try mu.admitReuse(s, a);
        // the stream's drafter context starts at its prompt; a resumed one at its resume point with the kept state's
        // ring window (multi._queue: put_slot_ring_window), else empty there (MultiDrafter.reset)
        if (mu.mdOf(mode)) |x| {
            const ring = if (hit) |h| h.ring else null;
            if (ring) |rg| try x.putWindow(a.slot, a.begin, rg.ptr) else x.reset(a.slot, a.begin);
        }
        try mu.filling.append(mu.gpa, s);
        mu.busy[a.slot] = true;
        if (a.sid >= mu.next_sid) mu.next_sid = a.sid + 1;
        return s;
    }

    /// Every rank: multi._finish - a stream forgotten, its slot free (its kept states stay while the slot's rows hold
    /// them).
    pub fn finish(mu: *Multi, sid: u32) void {
        const s = mu.take(sid) orelse return;
        mu.busy[s.slot] = false;
        mu.destroy(s);
    }

    fn take(mu: *Multi, sid: u32) ?*Stream {
        for (mu.streams.items, 0..) |s, i| if (s.sid == sid) {
            _ = mu.streams.orderedRemove(i);
            return s;
        };
        for (mu.filling.items, 0..) |s, i| if (s.sid == sid) {
            _ = mu.filling.orderedRemove(i);
            return s;
        };
        return null;
    }

    fn destroy(mu: *Multi, s: *Stream) void {
        s.out.deinit(mu.gpa);
        mu.gpa.free(s.prompt);
        mu.gpa.free(s.eos);
        mu.gpa.free(s.named);
        mu.gpa.free(s.chunks);
        mu.gpa.destroy(s);
    }

    /// After an error in a round (multi.drop): every stream forgotten; the ranks may no longer agree, so the decoder
    /// refuses further work (restart all ranks).
    pub fn dropAll(mu: *Multi) void {
        while (mu.streams.items.len > 0) {
            const s = mu.streams.pop().?;
            mu.busy[s.slot] = false;
            mu.destroy(s);
        }
        while (mu.filling.items.len > 0) {
            const s = mu.filling.pop().?;
            mu.busy[s.slot] = false;
            mu.destroy(s);
        }
    }

    // -------------------------------------------------------------------------------------------- fills ---

    /// Every rank: multi._fill / _chunk - the next chunk of the oldest queued prompt (whole: TF_GLM53_FILL_LAYERS is
    /// not ported), on its slot; at the prompt's end its first token (returned), the stream decoding from then on.
    pub fn applyFill(mu: *Multi, sid: u32) !?u32 {
        if (mu.broken) return error.RanksOutOfStep;
        if (mu.filling.items.len == 0) return error.OutOfStep;
        const s = mu.filling.items[0];
        if (s.sid != sid) return error.OutOfStep;
        const t0 = mu.now();
        defer s.prefill_s += @as(f64, @floatFromInt(mu.now() - t0)) / 1e9;
        const r = mu.r;
        const L = s.prompt.len;
        const ch = s.chunks[s.ci];
        const a = ch[0];
        const e = ch[1];
        const head: fw.HeadMode = if (s.sampling == null) .argmax else .local;
        if (a == e) { // a replay of a kept prompt end: its head row, nothing filled
            if (a != L or !s.head0) return error.Invalid;
            if (s.carry0) try mu.ops().copy(r.carry.ptr, mu.carry0.ptr + s.slot * mu.D * 2, mu.D * 2);
            s.carry0 = false;
            s.ci += 1;
            const tok = try r.replayHead(mu.head0.ptr + s.slot * mu.V * 4, L, s.sampling);
            try mu.start(s, tok);
            return tok;
        }
        if (s.ci == 0) try r.uploadIds(r.ids_dev.?.ptr, s.prompt); // the prompt's ids on the device (fills go in order)
        if (s.carry0 and a == s.begin) { // a resumed prompt: the MTP carry at its point
            try mu.ops().copy(r.carry.ptr, mu.carry0.ptr + s.slot * mu.D * 2, mu.D * 2);
            s.carry0 = false;
        }
        const was = r.solo_feed;
        r.solo_feed = false; // the one-stream drafters stay out of it: the stream's own drafter takes the taps
        defer r.solo_feed = was;
        const hb = try r.prefillChunk(&mu.views[s.slot], s.prompt, a, e, s.ci, head);
        if (mu.mdOf(s.mode)) |x| try x.feedPrompt(&r.pb, s.slot, a, e, L); // multi._chunk: dr.commit(the chunk's taps)
        s.ci += 1;
        if (mu.store != null and s.mode != .serial and (isNamed(s.named, e) or e == L)) {
            try mu.keepState(s, e, if (e == L) hb.?.lg else null);
        }
        if (e != L) return null;
        const tok = try r.pickFirst(hb.?, L, s.sampling);
        try r.healthy();
        try mu.start(s, tok);
        return tok;
    }

    /// multi._start: the prompt done - the pending token at P = len(prompt), the MTP backlog (carry = hidden P - 1).
    fn start(mu: *Multi, s: *Stream, tok: u32) !void {
        const L = s.prompt.len;
        s.P = L;
        s.tok = tok;
        s.m = 1;
        if (mu.k > 0) {
            const o = mu.ops();
            const i = s.slot * mu.BL;
            try o.copy(mu.bh.ptr + i * mu.D * 2, mu.r.carry.ptr, mu.D * 2);
            try o.fill32(mu.bt.ptr + i * 8, tok, 1);
            try o.fill32(mu.bt.ptr + i * 8 + 4, 0, 1);
        }
        if (mu.store) |*st| try st.setLiveSlot(s.slot, s.prompt);
        s.started = true;
        s.t_start = mu.now();
        for (mu.filling.items, 0..) |x, i| if (x == s) {
            _ = mu.filling.orderedRemove(i);
            break;
        };
        try mu.streams.append(mu.gpa, s);
    }

    /// The lead: a fill step of the oldest queued prompt (rank 0 tells the followers first); its first token taken.
    fn leadFill(mu: *Multi, link: ?Link) !void {
        const s = mu.filling.items[0];
        if (link) |l| try l.fill(l.ctx, s.sid);
        const got = try mu.applyFill(s.sid);
        if (got) |tok| {
            s.t_first = mu.now();
            const one = [1]u32{tok};
            mu.takeTokens(s, &one);
        }
    }

    /// multi._quick_fills: queued prompts whose rest fits quick_rows rows (together) fill to their first token now,
    /// each in the chunks it takes alone.
    fn quickFills(mu: *Multi, link: ?Link) !void {
        var used: usize = 0;
        while (mu.filling.items.len > 0) {
            const s = mu.filling.items[0];
            const a = s.chunks[s.ci][0];
            const rest = s.prompt.len - a;
            if (used + rest > mu.set.quick_rows) break;
            used += rest;
            while (mu.filling.items.len > 0 and mu.filling.items[0] == s) try mu.leadFill(link);
        }
    }

    /// Stream.take: a round's tokens appended and heard; the stream ends at its count, an end token or a stop.
    fn takeTokens(mu: *Multi, s: *Stream, new: []const u32) void {
        if (new.len == 0) return;
        s.out.appendSlice(mu.gpa, new) catch {};
        s.accepted_n += new.len - 1;
        var stop = false;
        if (s.hook) |h| stop = h.tokens(h.ctx, new);
        if (stop) s.stopped = true;
        if (s.isEnd(new[new.len - 1])) s.eos_hit = true;
        if (stop or s.out.items.len >= s.count or s.eos_hit) {
            s.done = true;
            if (s.t_end == 0) s.t_end = mu.now();
        }
    }

    /// The lead: the streams that finished (their sids into `out`), to report and `finish` on every rank.
    pub fn doneList(mu: *const Multi, out: []u32) []u32 {
        var n: usize = 0;
        for (mu.streams.items) |s| if (s.done and n < out.len) {
            out[n] = s.sid;
            n += 1;
        };
        for (mu.filling.items) |s| if (s.done and n < out.len) {
            out[n] = s.sid;
            n += 1;
        };
        return out[0..n];
    }

    // -------------------------------------------------------------------------------------------- rounds ---

    /// The lead: multi.round - a fill step for the oldest queued prompt, quick fills, then one decode round over the
    /// decoding streams. Streams that finished have `done` set (the caller finishes them and tells the followers).
    pub fn step(mu: *Multi, link: ?Link) !void {
        if (mu.broken) return error.RanksOutOfStep;
        errdefer mu.broken = mu.world > 1;
        if (mu.filling.items.len > 0) try mu.leadFill(link);
        if (mu.set.batch_fill) try mu.quickFills(link);
        var plan: Plan = .{};
        var n_live: usize = 0;
        for (mu.streams.items) |s| n_live += @intFromBool(!s.done);
        if (n_live == 0) {
            mu.t_round_end = 0;
            return;
        }
        plan.cut = if (n_live >= mu.set.cut_streams and mu.cut_on) mu.set.draft_cut else 0;
        const drafters = mu.md[0] != null or mu.md[1] != null;
        plan.dcut = if (n_live >= mu.set.cut_streams and drafters) mu.set.cutDFlash() else 0;
        for (mu.streams.items) |s| {
            if (s.done) continue;
            const arm: u32 = switch (s.mode) {
                .mtp => arm_mtp,
                .dflash, .dspark => arm_draft,
                .serial => arm_none,
            };
            // multi._round's room: MTP k; a drafter stream min(its depth, the tokens left - 1, the cache left - 1)
            const room: usize = switch (arm) {
                arm_mtp => mu.k,
                arm_draft => @min(mu.depthOf(drafterOf(s.mode).?), @min(s.count -| s.out.items.len -| 1, mu.r.capacity -| s.P -| 1)),
                else => 0,
            };
            plan.items[plan.n] = .{ .sid = s.sid, .slot = @intCast(s.slot), .P = @intCast(s.P), .tok = s.tok, .m = @intCast(s.m), .mode = @intFromEnum(s.mode), .arm = arm, .room = @intCast(room), .sampled = @intFromBool(s.sampling != null) };
            plan.n += 1;
        }
        if (link) |l| try l.round(l.ctx, &plan);
        var res: [max_streams]Result = undefined;
        var ds: [max_streams]usize = undefined;
        const g0 = mu.r.graph_runs;
        const e0 = mu.r.eager_runs;
        const t0 = mu.now();
        try mu.applyRound(&plan, &res, &ds);
        const t1 = mu.now();
        // the host time before this round's work since the previous round ended (the CLI: the plan; the server:
        // also the reply hooks, the loop, admissions, fills and the round's message)
        const gap: u64 = if (mu.t_round_end > 0 and t0 > mu.t_round_end) @intCast(t0 - mu.t_round_end) else 0;
        const took: u64 = @intCast(@max(0, t1 - t0));
        mu.t_round_end = t1;
        mu.profWait(gap);
        for (plan.items[0..plan.n], 0..) |it, i| {
            const s = mu.find(it.sid).?;
            s.round_ns += took;
            s.gap_ns += gap;
            s.graph_runs += mu.r.graph_runs - g0;
            s.eager_runs += mu.r.eager_runs - e0;
            const x = &res[i];
            var new: [fw.decode_rows]u32 = undefined;
            var nn: usize = 0;
            for (x.tokens()) |t| { // a lone request's room: its count, its end tokens
                if (s.out.items.len + nn >= s.count) break;
                new[nn] = t;
                nn += 1;
                if (s.isEnd(t)) break;
            }
            mu.takeTokens(s, new[0..nn]);
        }
    }

    /// Every rank: one decode round of `plan` (multi._gpu, _picks, _commit): the round's tables, the batched MTP
    /// chain, the draft cut, the verify window, every rank's picks, the next backlogs. Each stream's position, pending
    /// token and backlog move on; `res[i]` holds item i's kept drafts and emitted tokens, `ds_out[i]` its drafts.
    pub fn applyRound(mu: *Multi, plan: *const Plan, res: *[max_streams]Result, ds_out: *[max_streams]usize) !void {
        if (mu.broken) return error.RanksOutOfStep;
        errdefer mu.broken = mu.world > 1;
        var ss: [max_streams]*Stream = undefined;
        for (plan.items[0..plan.n], 0..) |it, i| {
            const s = mu.find(it.sid) orelse return error.OutOfStep;
            if (!s.started or s.slot != it.slot or s.P != it.P or s.tok != it.tok or s.m != it.m or @intFromEnum(s.mode) != it.mode) {
                if (!@import("builtin").is_test) std.log.err("rank {d}: stream {d} is at ({d}, {d}, {d}), rank 0's round has ({d}, {d}, {d}): the ranks are out of step; restart all ranks", .{ mu.w.rank, s.sid, s.P, s.tok, s.m, it.P, it.tok, it.m });
                return error.OutOfStep;
            }
            ss[i] = s;
        }
        const pt0 = if (mu.pf.on) try mu.profBegin(plan) else 0;
        const t = try mu.gpu(plan);
        try mu.picks(plan, &t, res);
        try mu.mark(4);
        try mu.commit(plan, &t, res);
        try mu.mark(5);
        if (mu.pf.on) mu.profEnd(pt0, t.R);
        mu.rounds += 1;
        for (plan.items[0..plan.n], 0..) |_, i| {
            const s = ss[i];
            const x = &res[i];
            s.rounds += 1;
            s.drafted += t.ds[i];
            s.min_rows = if (s.min_rows == 0) 1 + t.ds[i] else @min(s.min_rows, 1 + t.ds[i]);
            s.P += x.n + 1;
            s.tok = x.emit[x.len - 1];
            s.m = x.n + 1;
            ds_out[i] = t.ds[i];
        }
    }

    /// TF_GLM53_ROUND_PROFILE: the host time before this round (since the previous round ended), from the caller.
    pub fn profWait(mu: *Multi, ns: u64) void {
        if (mu.pf.on) mu.pf.wait += ns;
    }

    fn profBegin(mu: *Multi, plan: *const Plan) !i128 {
        var sampled: u64 = 0;
        for (plan.items[0..plan.n]) |it| sampled |= it.sampled;
        var drafting: u64 = 0;
        for (plan.items[0..plan.n]) |it| drafting += @intFromBool(it.room > 0);
        const key = @as(u64, plan.n) | (sampled << 8) | (drafting << 16) | (@as(u64, @intFromBool(plan.cut > 0)) << 24);
        if (mu.pf.n > 0 and (key != mu.pf.key or mu.pf.n >= 64)) mu.profFlush();
        mu.pf.key = key;
        try mu.r.f.s.synchronize();
        mu.pf.last = mu.now();
        return mu.pf.last;
    }

    /// Phase `i` ends (profiling: the stream drained, the time since the previous mark).
    fn mark(mu: *Multi, i: usize) !void {
        if (!mu.pf.on) return;
        try mu.r.f.s.synchronize();
        const t = mu.now();
        mu.pf.acc[i] += @intCast(@max(0, t - mu.pf.last));
        mu.pf.last = t;
    }

    fn profEnd(mu: *Multi, t0: i128, R: usize) void {
        mu.pf.total += @intCast(@max(0, mu.now() - t0));
        mu.pf.rows += R;
        mu.pf.n += 1;
    }

    fn profFlush(mu: *Multi) void {
        const p = &mu.pf;
        if (p.n == 0) return;
        const n: f64 = @floatFromInt(p.n);
        const ms = struct {
            fn f(v: u64, c: f64) f64 {
                return @as(f64, @floatFromInt(v)) / 1e6 / c;
            }
        }.f;
        if (!@import("builtin").is_test) std.log.info("glm53 prof rank {d}: rounds={d} streams={d} sampled={d} drafting={d} cut={d} rows={d:.2} | propose={d:.2} mtp={d:.2} cut={d:.2} verify={d:.2} picks={d:.2} commit={d:.2} total={d:.2} wait={d:.2} ms/round", .{ mu.w.rank, p.n, p.key & 0xff, (p.key >> 8) & 0xff, (p.key >> 16) & 0xff, (p.key >> 24) & 0xff, @as(f64, @floatFromInt(p.rows)) / n, ms(p.acc[0], n), ms(p.acc[1], n), ms(p.acc[2], n), ms(p.acc[3], n), ms(p.acc[4], n), ms(p.acc[5], n), ms(p.total, n), ms(p.wait, n) });
        const on = p.on;
        p.* = .{};
        p.on = on;
    }

    /// The round's tables to the device (one upload of the plan region; vb.ids from their own staging).
    fn upload(mu: *Multi, t: *const Tables) !void {
        const o = mu.ops();
        const l = mu.lay;
        @memcpy(mu.pinSlice(i32, l.vpos, t.R), t.vpos[0..t.R]);
        @memcpy(mu.pinSlice(i32, l.vbase, t.R), t.vbase[0..t.R]);
        for (0..mu.k) |j| {
            const rows_j = if (j == 0) t.M else t.S;
            @memcpy(mu.pinSlice(i32, l.mpos[j], rows_j), t.mpos[j][0..rows_j]);
            @memcpy(mu.pinSlice(i32, l.mbase[j], rows_j), t.mbase[j][0..rows_j]);
            @memcpy(mu.pinSlice(i64, l.last[j], t.S), t.last[j][0..t.S]);
            @memcpy(mu.pinSlice(i64, l.vdst[j], t.S), t.vdst[j][0..t.S]);
        }
        @memcpy(mu.pinSlice(i64, l.src, t.M), t.src[0..t.M]);
        try o.upload(mu.tab.ptr, mu.pin.bytes[0..l.plan_end]);
        const ids = mu.pinSlice(i64, mu.play.vids, t.R);
        @memcpy(ids, t.vids[0..t.R]);
        try o.upload(mu.vb.ids, std.mem.sliceAsBytes(ids));
    }

    /// Every rank (multi._gpu): the tables, the MTP chain (step 1 over every MTP stream's backlog, steps 2..k one
    /// row a drafting stream), the draft cut, the verify window. Returns the round's tables (the cut applied).
    fn gpu(mu: *Multi, plan: *const Plan) !Tables {
        var dd: Drafts = .{};
        try mu.propose(plan, &dd);
        try mu.mark(0);
        var t = try tables(plan, mu.k, mu.BL, mu.local, &dd);
        if (t.R == 0) return error.Invalid;
        try mu.upload(&t);
        const topk = mu.r.topk;
        if (t.M > 0) {
            const mb = &mu.mb.?;
            const o = mu.ops();
            const src = mu.tab.ptr + mu.lay.src;
            try o.rowsGather(mb.hin, mu.bh.ptr, src, t.M, mu.D * 2);
            try o.rowsGather(mb.ids, mu.bt.ptr, src, t.M, 8);
            const Tm = tri.bucket(t.keep_end.?, topk);
            if (t.S == 0) return error.Invalid; // MTP streams always draft (no DFlash2 / auto rounds here)
            for (1..mu.k + 1) |j| try mu.mtpStep(j, if (j == 1) t.M else t.S, t.S, Tm);
            try mu.mark(1);
            if (plan.cut > 0) try mu.cutWindow(plan, &t);
            try mu.mark(2);
        } else {
            try mu.mark(1);
        }
        mu.widest = @max(mu.widest, t.R);
        const T = tri.bucket(t.end, topk);
        var sampled = false;
        for (plan.items[0..plan.n]) |it| sampled = sampled or it.sampled != 0;
        try mu.verify(t.R, T, if (sampled) .local else .argmax);
        try mu.mark(3);
        return t;
    }

    /// Every rank (multi._gpu's dr.propose): each drafter's streams of the plan with room ask it for drafts - one
    /// block pass a drafter, each stream's host stage with its sampling and the round's confidence floor.
    fn propose(mu: *Multi, plan: *const Plan, dd: *Drafts) !void {
        for (mu.md, 0..) |maybe, which| {
            const x = maybe orelse continue;
            var asks: [max_streams]mdraft.Ask = undefined;
            var at: [max_streams]usize = undefined;
            var na: usize = 0;
            for (plan.items[0..plan.n], 0..) |it, i| {
                if (it.arm != arm_draft or it.room == 0) continue;
                const m = modeOfWord(it.mode) orelse return error.OutOfStep;
                if ((drafterOf(m) orelse return error.OutOfStep) != which) continue;
                const s = mu.find(it.sid) orelse return error.OutOfStep;
                asks[na] = .{ .slot = it.slot, .pending = it.tok, .depth = it.room, .sampling = s.sampling, .conf = mu.confOf(which, plan.dcut) };
                at[na] = i;
                na += 1;
            }
            if (na == 0) continue;
            var got: [max_streams]mdraft.Drafted = undefined;
            try x.propose(asks[0..na], got[0..na]);
            for (0..na) |q| {
                const i = at[q];
                dd.n[i] = got[q].n;
                @memcpy(dd.ids[i][0..got[q].n], got[q].slice());
            }
        }
    }

    fn verifyKey(R: usize, T: ?usize, pick: fw.HeadMode) u64 {
        const pk: u64 = if (pick == .argmax) 0 else 1;
        return 5 | (@as(u64, R) << 4) | (tIndex(T) << 12) | (pk << 20);
    }

    /// multi.prewarm (+ TF_GLM53_PRECAPTURE's verify widths), every rank alike and with no messages: in every key
    /// bucket, the rounds of 1..N MTP-drafting streams with every backlog total (sampled mixes with the "local"
    /// head), then every verify width with both picks. The caches take scratch values at the positions used and are
    /// zeroed after (every request writes a position before reading it). Returns the graphs captured.
    pub fn prewarm(mu: *Multi) !usize {
        const r = mu.r;
        const before = r.graphs.count();
        const t0 = mu.now();
        var bs: [24]?usize = undefined;
        const bks = r.buckets(&bs);
        const cap = r.capacity - @max(mu.k, mu.frows) - 2;
        for (bks, 0..) |T, bi| {
            const prev: ?usize = if (bi > 0) bks[bi - 1] else null;
            const P: usize = if (T == null) 10 else @min((prev orelse r.topk) + 100, cap);
            if (mu.k > 0 and sameBucket(tri.bucket(P + mu.k + 1, r.topk), T)) {
                for (1..mu.N + 1) |S| {
                    for (S..S * (mu.k + 1) + 1) |M| {
                        const variants: usize = if (M == S) 2 else 1;
                        for (0..variants) |sampled| {
                            var plan: Plan = .{ .n = S };
                            for (0..S) |i| {
                                const ms = M / S + @intFromBool(i < M % S);
                                plan.items[i] = .{ .sid = @intCast(i), .slot = @intCast(i), .P = @intCast(P), .tok = 1000, .m = @intCast(ms), .mode = @intFromEnum(Mode.mtp), .arm = arm_mtp, .room = @intCast(mu.k), .sampled = @intCast(sampled) };
                            }
                            _ = try mu.gpu(&plan);
                            try r.f.s.synchronize(); // the pinned tables are written again next
                        }
                    }
                }
            }
            if (!mu.set.precapture and mu.frows == 0) continue; // drafters: every verify width, as multi.prewarm
            // multi._scratch: verify row i in slot i // width at P + i % width
            const width = mu.rows / mu.N;
            var t: Tables = .{ .R = mu.rows };
            for (0..mu.rows) |i| {
                t.vpos[i] = @intCast(P + i % width);
                t.vbase[i] = @intCast((i / width) * mu.local);
                t.vids[i] = 1000;
            }
            try mu.upload(&t);
            for ([_]fw.HeadMode{ .argmax, .local }) |pick| {
                for (1..mu.rows + 1) |R| {
                    if (r.graphs.contains(verifyKey(R, T, pick))) continue;
                    try mu.verify(R, T, pick);
                }
            }
            try r.f.s.synchronize();
        }
        // Phase 4c: the drafters' passes (MultiDrafter.capture: every context-update bucket, the block passes)
        for (mu.md) |maybe| {
            if (maybe) |x| _ = try x.prewarm();
        }
        try r.f.s.synchronize();
        try r.resetState();
        mu.pf = .{ .on = mu.pf.on }; // the prewarm's rounds are not profiled
        try mu.bh.fill8(0, r.f.s.handle);
        try mu.bt.fill8(0, r.f.s.handle);
        try r.f.s.synchronize();
        const n = r.graphs.count() - before;
        if (mu.w.rank == 0 and !@import("builtin").is_test) std.log.info("glm53: {d} concurrent decode graphs captured in {d:.1} s ({d} streams, windows up to {d} rows)", .{ n, @as(f64, @floatFromInt(mu.now() - t0)) / 1e9, mu.N, mu.rows });
        return n;
    }

    fn sameBucket(a: ?usize, b: ?usize) bool {
        if (a == null or b == null) return a == null and b == null;
        return a.? == b.?;
    }

    const VerifyCtx = struct {
        mu: *Multi,
        R: usize,
        T: ?usize,
        pick: fw.HeadMode,
        pub fn go(c: @This()) !void {
            const m = c.mu;
            var g = m.r.f;
            g.base = m.tab.ptr + m.lay.vbase;
            var sv = m.r.st;
            sv.pos = m.tab.ptr + m.lay.vpos;
            try g.compute(&m.vb, &sv, c.R, c.T, c.pick, false);
        }
    };

    /// multi._verify: the target over vb.ids[:R] at the rows' positions (logits "all", head `pick`).
    fn verify(mu: *Multi, R: usize, T: ?usize, pick: fw.HeadMode) !void {
        try mu.r.runGraph(verifyKey(R, T, pick), VerifyCtx{ .mu = mu, .R = R, .T = T, .pick = pick });
    }

    const StepCtx = struct {
        mu: *Multi,
        j: usize,
        n: usize,
        S: usize,
        T: ?usize,
        pub fn go(c: @This()) !void {
            const m = c.mu;
            const mb = &m.mb.?;
            const o = m.ops();
            const l = m.lay;
            var g = m.r.f;
            g.base = m.tab.ptr + l.mbase[c.j - 1];
            var sv = m.r.st;
            sv.mpos = m.tab.ptr + l.mpos[c.j - 1];
            const last = m.tab.ptr + l.last[c.j - 1];
            try g.mtpCompute(mb, &sv, c.n, c.T, .{ .head = true, .chain_normed = m.r.set.chain_normed, .last = last, .last_n = c.S });
            if (m.cut_on) try o.cutStats(mb.lg, m.hv, m.hv, mb.amax_all, m.world, c.S, m.tab.ptr + l.lse + (c.j - 1) * m.N * 4, m.tab.ptr + l.top + (c.j - 1) * m.N * 4);
            try o.rowsScatter(m.vb.ids, m.tab.ptr + l.vdst[c.j - 1], mb.argmax, c.S, 8); // vb.ids.index_copy_(0, vdst, ...)
            try o.copy(mb.ids, mb.argmax, c.S * 8); // the drafts feed the next step
            try o.rowsGather(mb.hin, mb.hidden, last, c.S, m.D * 2); // with the rows' chain hiddens
        }
    };

    /// multi._mtp_step: MTP step j over n rows (step 1: the backlog rows; later: one a drafting stream), its S picks
    /// into the verify window's ids.
    fn mtpStep(mu: *Multi, j: usize, n: usize, S: usize, T: ?usize) !void {
        const key: u64 = 6 | (@as(u64, n) << 4) | (@as(u64, S) << 12) | (@as(u64, j) << 20) | (tIndex(T) << 28);
        try mu.r.runGraph(key, StepCtx{ .mu = mu, .j = j, .n = n, .S = S, .T = T });
    }

    /// multi._cut on every rank: every rank's log-sum-exps gathered, each drafting stream's chain probability, the
    /// drafts past the cut dropped and the verify window closed up (positions, bases, ids).
    fn cutWindow(mu: *Multi, plan: *const Plan, t: *Tables) !void {
        const o = mu.ops();
        const l = mu.lay;
        const kn = mu.k * mu.N;
        const stride = try mu.r.f.smallGather(mu.tab.ptr + l.lse, mu.tab.ptr + l.lse_all, kn);
        const ranks: usize = if (mu.r.f.comm != null or mu.r.f.fast != null) mu.world else 1;
        const top = mu.pinSlice(f32, mu.play.cut, kn);
        const all = mu.pinSlice(f32, mu.play.cut + kn * 4, ranks * stride);
        try o.download(std.mem.sliceAsBytes(top), mu.tab.ptr + l.top);
        try o.download(std.mem.sliceAsBytes(all), mu.tab.ptr + l.lse_all);
        const ids = mu.pinSlice(i64, mu.play.rd, t.R);
        try o.download(std.mem.sliceAsBytes(ids), mu.vb.ids);
        try mu.r.f.s.synchronize();
        var nds: [max_streams]usize = undefined;
        cutDecide(t, mu.k, mu.N, plan.cut, top, all, ranks, stride, &nds);
        if (std.mem.eql(usize, nds[0..plan.n], t.ds[0..plan.n])) return;
        for (0..plan.n) |i| mu.cuts += t.ds[i] - nds[i];
        var old: [fw.decode_rows]i64 = undefined;
        @memcpy(old[0..t.R], ids);
        var src_rows: [fw.decode_rows]usize = undefined;
        const R = compact(plan, t, &nds, mu.local, &src_rows);
        for (0..R) |i| t.vids[i] = old[src_rows[i]];
        @memcpy(mu.pinSlice(i32, l.vpos, R), t.vpos[0..R]);
        @memcpy(mu.pinSlice(i32, l.vbase, R), t.vbase[0..R]);
        try o.upload(mu.tab.ptr + l.vpos, mu.pin.bytes[l.vpos .. l.vbase + R * 4]);
        const vids = mu.pinSlice(i64, mu.play.vids, R);
        @memcpy(vids, t.vids[0..R]);
        try o.upload(mu.vb.ids, std.mem.sliceAsBytes(vids));
        var end: usize = 0;
        for (plan.items[0..plan.n], 0..) |it, i| end = @max(end, it.P + 1 + t.ds[i]);
        t.end = end;
    }

    /// Every rank (multi._picks): each stream's target picks along its drafts -> (drafts kept, tokens emitted).
    /// Greedy windows: the argmax head's picks; sampled windows: every rank's candidates, gathered once.
    fn picks(mu: *Multi, plan: *const Plan, t: *const Tables, res: *[max_streams]Result) !void {
        const o = mu.ops();
        const R = t.R;
        const vb = &mu.vb;
        var sampled = false;
        var cnt: usize = 1;
        for (plan.items[0..plan.n]) |it| {
            if (it.sampled == 0) continue;
            sampled = true;
            const s = mu.find(it.sid) orelse return error.OutOfStep;
            const sm = s.sampling orelse return error.OutOfStep;
            cnt = @max(cnt, @min(sm.candidates(), mu.V));
        }
        const ids = mu.pinSlice(i64, mu.play.rd, R);
        const amax = mu.pinSlice(i64, mu.play.rd + mu.rows * 8, R);
        try o.download(std.mem.sliceAsBytes(ids), vb.ids);
        var all: []f32 = &.{};
        var stride: usize = 0;
        var ranks: usize = 1;
        if (!sampled) {
            try o.download(std.mem.sliceAsBytes(amax), vb.argmax);
            if (mu.idle) |x| x.f(x.ctx); // the window is queued: the lead's caller works while it runs
            try mu.r.f.s.synchronize();
        } else {
            if (vb.cand_cols == 0 or cnt > smp.max_candidates) return error.Invalid;
            const words = R * 2 * cnt + 1;
            if (words > vb.cand_words) return error.Invalid;
            try o.topkColumns(vb.lg, mu.V, mu.V, R, cnt, true, vb.tsel, vb.cand_cols, cnt);
            try o.cands(vb.lg, mu.V, vb.cand_cols, cnt, mu.w.vocab_off, vb.cand_send, R);
            try o.fill32(vb.cand_send + (words - 1) * 4, 0, 1);
            stride = try mu.r.f.smallGather(vb.cand_send, vb.cand_all, words);
            ranks = if (mu.r.f.comm != null or mu.r.f.fast != null) mu.world else 1;
            all = mu.pinSlice(f32, mu.play.cand, ranks * stride);
            try o.download(std.mem.sliceAsBytes(all), vb.cand_all);
            if (mu.idle) |x| x.f(x.ctx);
            try mu.r.f.s.synchronize();
        }
        try mu.r.healthy(); // every rank: no RoCE timeout behind these picks
        var cands: [max_streams * smp.max_candidates]smp.Cand = undefined;
        for (plan.items[0..plan.n], 0..) |it, i| {
            const off = t.offs[i];
            const d = t.ds[i];
            var drafts: [fw.decode_rows]u32 = undefined;
            for (0..d) |j| drafts[j] = @intCast(ids[off + 1 + j]);
            var pk: [fw.decode_rows]u32 = undefined;
            var np: usize = 0;
            const s = mu.find(it.sid).?;
            for (0..1 + d) |j| {
                if (!sampled) {
                    pk[j] = @intCast(amax[off + j]);
                } else {
                    var nc: usize = 0;
                    for (0..ranks) |q| {
                        const row = all[q * stride + (off + j) * 2 * cnt ..][0 .. 2 * cnt];
                        const cid: []const i32 = @ptrCast(row[cnt..]);
                        for (0..cnt) |c| {
                            cands[nc] = .{ .v = row[c], .id = cid[c] };
                            nc += 1;
                        }
                    }
                    pk[j] = if (s.sampling) |sm| smp.choose(cands[0..nc], it.P + j + 1, sm) else bestCandidate(cands[0..nc]);
                }
                np += 1;
                if (j >= d or drafts[j] != pk[j]) break; // multi._picks: a row past the first mismatch is never read
            }
            const n = accepted(drafts[0..d], pk[0..np]);
            res[i] = .{ .n = n, .len = n + 1 };
            @memcpy(res[i].emit[0 .. n + 1], pk[0 .. n + 1]);
        }
    }

    /// Every rank (multi._commit's dr.commit): each drafter stream's kept rows' taps (vb.taps at its window rows)
    /// into its slot's drafter context, every stream of a drafter in one stacked update.
    fn commitTaps(mu: *Multi, plan: *const Plan, t: *const Tables, res: *const [max_streams]Result) !void {
        const vb = &mu.vb;
        if (vb.taps == 0) return;
        const pitch = vb.tap_n * mu.D * 2;
        for (mu.md, 0..) |maybe, which| {
            const x = maybe orelse continue;
            var items: [max_streams]mdraft.Commit = undefined;
            var ni: usize = 0;
            for (plan.items[0..plan.n], 0..) |it, i| {
                const m = modeOfWord(it.mode) orelse return error.OutOfStep;
                if ((drafterOf(m) orelse continue) != which) continue;
                items[ni] = .{ .slot = it.slot, .src = vb.taps + t.offs[i] * pitch, .n = res[i].n + 1 };
                ni += 1;
            }
            if (ni > 0) try x.commit(items[0..ni], pitch);
        }
    }

    /// Every rank (multi._commit): each MTP stream's next backlog - the kept rows' target hiddens (final-normed as the
    /// MTP layer reads them) and tokens into its slot's backlog rows.
    fn commit(mu: *Multi, plan: *const Plan, t: *const Tables, res: *const [max_streams]Result) !void {
        try mu.commitTaps(plan, t, res);
        if (mu.k == 0) return;
        const o = mu.ops();
        const l = mu.lay;
        var c: usize = 0;
        const idx = mu.pinSlice(i64, l.cidx, mu.rows);
        const dst = mu.pinSlice(i64, l.cdst, mu.rows);
        const tok = mu.pinSlice(i64, l.ctok, mu.rows);
        for (plan.items[0..plan.n], 0..) |it, i| {
            if (it.mode != @intFromEnum(Mode.mtp)) continue;
            const x = &res[i];
            for (0..x.n + 1) |q| {
                idx[c] = @intCast(t.offs[i] + q);
                dst[c] = @intCast(it.slot * mu.BL + q);
                tok[c] = x.emit[q];
                c += 1;
            }
        }
        if (c == 0) return;
        try o.upload(mu.tab.ptr + l.cidx, mu.pin.bytes[l.cidx..l.commit_end]);
        const vb = &mu.vb;
        // the kept rows' hiddens (vb.x is free once the window's picks are read), final-normed (target_hidden_for_mtp)
        try o.rowsGather(vb.x, vb.hidden, mu.tab.ptr + l.cidx, c, mu.D * 2);
        var rows_from = vb.x;
        if (mu.r.set.hid_normed) {
            const tt: tri.Tri = .{ .set = &mu.r.f.k.triton, .s = mu.r.f.s };
            try tt.rmsnorm(vb.x, mu.D, mu.w.final_norm, vb.normed, mu.D, c, mu.D, mu.w.config.eps);
            rows_from = vb.normed;
        }
        try o.rowsScatter(mu.bh.ptr, mu.tab.ptr + l.cdst, rows_from, c, mu.D * 2);
        try o.rowsScatter(mu.bt.ptr, mu.tab.ptr + l.cdst, mu.tab.ptr + l.ctok, c, 8);
    }
};

// ------------------------------------------------------------------------------------------------------ tests ---

test "window rows: N streams x (k + 1), at most a decode window" {
    try std.testing.expectEqual(@as(usize, 12), try windowRows(4, 2));
    try std.testing.expectEqual(@as(usize, 32), try windowRows(8, 3));
    try std.testing.expectError(error.WindowTooWide, windowRows(8, 4));
    try std.testing.expectError(error.Invalid, windowRows(0, 2));
}

test "a round's plan as words and back" {
    var p: Plan = .{ .n = 2, .cut = 0.6 };
    p.items[0] = .{ .sid = 3, .slot = 1, .P = 4000, .tok = 151000, .m = 2, .mode = 1, .arm = arm_mtp, .room = 2, .sampled = 1 };
    p.items[1] = .{ .sid = 9, .slot = 0, .P = 77, .tok = 5, .m = 1, .mode = 0, .arm = arm_none, .room = 0, .sampled = 0 };
    p.dcut = 0.45;
    var buf: [plan_words]u32 = undefined;
    const w = try p.encode(&buf);
    try std.testing.expectEqual(@as(usize, plan_head + 2 * item_words), w.len);
    const q = try Plan.decode(w);
    try std.testing.expectEqual(p.n, q.n);
    try std.testing.expectEqual(p.cut, q.cut);
    try std.testing.expectEqual(p.dcut, q.dcut);
    try std.testing.expectEqual(p.items[0], q.items[0]);
    try std.testing.expectEqual(p.items[1], q.items[1]);
    try std.testing.expectError(error.OutOfStep, Plan.decode(w[0 .. w.len - 1]));
}

test "round tables as multi._gpu builds them" {
    var p: Plan = .{ .n = 3 };
    p.items[0] = .{ .sid = 0, .slot = 1, .P = 50, .tok = 7, .m = 2, .mode = 1, .arm = arm_mtp, .room = 2, .sampled = 0 };
    p.items[1] = .{ .sid = 1, .slot = 0, .P = 20, .tok = 9, .m = 1, .mode = 0, .arm = arm_none, .room = 0, .sampled = 0 };
    p.items[2] = .{ .sid = 2, .slot = 2, .P = 30, .tok = 5, .m = 3, .mode = 1, .arm = arm_mtp, .room = 2, .sampled = 0 };
    const t = try tables(&p, 2, 3, 100, null);
    try std.testing.expectEqual(@as(usize, 7), t.R);
    try std.testing.expectEqualSlices(i32, &.{ 50, 51, 52, 20, 30, 31, 32 }, t.vpos[0..7]);
    try std.testing.expectEqualSlices(i32, &.{ 100, 100, 100, 0, 200, 200, 200 }, t.vbase[0..7]);
    try std.testing.expectEqualSlices(i64, &.{ 7, 0, 0, 9, 5, 0, 0 }, t.vids[0..7]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 4 }, t.offs[0..3]);
    try std.testing.expectEqualSlices(usize, &.{ 2, 0, 2 }, t.ds[0..3]);
    try std.testing.expectEqual(@as(usize, 53), t.end);
    try std.testing.expectEqual(@as(usize, 5), t.M);
    try std.testing.expectEqual(@as(usize, 2), t.S);
    try std.testing.expectEqualSlices(i32, &.{ 48, 49, 27, 28, 29 }, t.mpos[0][0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 100, 100, 200, 200, 200 }, t.mbase[0][0..5]);
    try std.testing.expectEqualSlices(i64, &.{ 3, 4, 6, 7, 8 }, t.src[0..5]);
    try std.testing.expectEqualSlices(i64, &.{ 1, 4 }, t.last[0][0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 1, 5 }, t.vdst[0][0..2]);
    try std.testing.expectEqualSlices(i32, &.{ 50, 30 }, t.mpos[1][0..2]);
    try std.testing.expectEqualSlices(i32, &.{ 100, 200 }, t.mbase[1][0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 0, 1 }, t.last[1][0..2]);
    try std.testing.expectEqualSlices(i64, &.{ 2, 6 }, t.vdst[1][0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, t.drafting[0..2]);
    try std.testing.expectEqual(@as(?usize, 52), t.keep_end);
}

test "the draft cut keeps drafts while the chain probability holds, and the window closes up" {
    var p: Plan = .{ .n = 2 };
    p.items[0] = .{ .sid = 0, .slot = 0, .P = 10, .tok = 1, .m = 1, .mode = 1, .arm = arm_mtp, .room = 2, .sampled = 0 };
    p.items[1] = .{ .sid = 1, .slot = 1, .P = 20, .tok = 2, .m = 1, .mode = 1, .arm = arm_mtp, .room = 2, .sampled = 0 };
    var t = try tables(&p, 2, 3, 64, null);
    const N = 4;
    // stream 0: p1 = exp(1.5 - (1 + ln 2)) = 0.824 (kept), p2 = 0.824 * exp(1 - (2 + ln(1 + e^-2))) = 0.267 (cut)
    // stream 1: both picks at their total: p = 1, 1 (both kept)
    var pick: [2 * N]f32 = @splat(0);
    var parts: [2 * 2 * N]f32 = @splat(0);
    pick[0 * N + 0] = 1.5;
    pick[1 * N + 0] = 1.0;
    parts[0 * 2 * N + 0 * N + 0] = 1.0;
    parts[1 * 2 * N + 0 * N + 0] = 1.0;
    parts[0 * 2 * N + 1 * N + 0] = 2.0;
    parts[1 * 2 * N + 1 * N + 0] = 0.0;
    // stream 1: one rank holds everything (the other at -inf): total = its value
    pick[0 * N + 1] = 3.0;
    pick[1 * N + 1] = -1.0;
    parts[0 * 2 * N + 0 * N + 1] = 3.0;
    parts[1 * 2 * N + 0 * N + 1] = -std.math.inf(f32);
    parts[0 * 2 * N + 1 * N + 1] = -1.0;
    parts[1 * 2 * N + 1 * N + 1] = -std.math.inf(f32);
    var nds: [max_streams]usize = undefined;
    cutDecide(&t, 2, N, 0.6, &pick, &parts, 2, 2 * N, &nds);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, nds[0..2]);
    var src_rows: [fw.decode_rows]usize = undefined;
    const R = compact(&p, &t, &nds, 64, &src_rows);
    try std.testing.expectEqual(@as(usize, 5), R);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 4, 5 }, src_rows[0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 20, 21, 22 }, t.vpos[0..5]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 64, 64, 64 }, t.vbase[0..5]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, t.offs[0..2]);
}

test "kept drafts and a sampled window's greedy row" {
    try std.testing.expectEqual(@as(usize, 2), accepted(&.{ 4, 5 }, &.{ 4, 5, 6 }));
    try std.testing.expectEqual(@as(usize, 1), accepted(&.{ 4, 5 }, &.{ 4, 7 }));
    try std.testing.expectEqual(@as(usize, 0), accepted(&.{ 4, 5 }, &.{3}));
    try std.testing.expectEqual(@as(usize, 0), accepted(&.{}, &.{3}));
    const cands = [_]smp.Cand{ .{ .v = 1.0, .id = 30 }, .{ .v = 2.5, .id = 90 }, .{ .v = 2.5, .id = 12 }, .{ .v = -1, .id = 0 } };
    try std.testing.expectEqual(@as(u32, 12), bestCandidate(&cands));
}

test "the device tables' offsets are 16-byte aligned and disjoint" {
    const l = Layout.make(12, 4, 2, 4);
    const offs = [_]usize{ l.vpos, l.vbase, l.mpos[0], l.mbase[0], l.last[0], l.vdst[0], l.mpos[1], l.mbase[1], l.last[1], l.vdst[1], l.src, l.cidx, l.cdst, l.ctok, l.lse, l.top, l.lse_all };
    for (offs, 0..) |o, i| {
        try std.testing.expectEqual(@as(usize, 0), o % 16);
        if (i > 0) try std.testing.expect(o > offs[i - 1]);
    }
    try std.testing.expect(l.vbase - l.vpos >= 12 * 4);
    try std.testing.expect(l.plan_end <= l.cidx and l.commit_end <= l.lse and l.total > l.lse_all);
}

test "Phase 4c: drafter streams' windows take the host's drafts; MTP streams keep their backlog rows" {
    var p: Plan = .{ .n = 3 };
    p.items[0] = .{ .sid = 0, .slot = 2, .P = 40, .tok = 7, .m = 1, .mode = @intFromEnum(Mode.dflash), .arm = arm_draft, .room = 7, .sampled = 0 };
    p.items[1] = .{ .sid = 1, .slot = 0, .P = 20, .tok = 9, .m = 2, .mode = @intFromEnum(Mode.mtp), .arm = arm_mtp, .room = 2, .sampled = 0 };
    p.items[2] = .{ .sid = 2, .slot = 1, .P = 30, .tok = 5, .m = 1, .mode = @intFromEnum(Mode.dspark), .arm = arm_draft, .room = 3, .sampled = 1 };
    var dd: Drafts = .{};
    dd.n[0] = 4;
    dd.ids[0][0..4].* = .{ 100, 101, 102, 103 };
    dd.n[2] = 0; // its drafter declined: the pending token alone
    const t = try tables(&p, 2, 3, 1000, &dd);
    try std.testing.expectEqual(@as(usize, 9), t.R);
    try std.testing.expectEqualSlices(usize, &.{ 0, 5, 8 }, t.offs[0..3]);
    try std.testing.expectEqualSlices(usize, &.{ 4, 2, 0 }, t.ds[0..3]);
    try std.testing.expectEqualSlices(i64, &.{ 7, 100, 101, 102, 103, 9, 0, 0, 5 }, t.vids[0..9]);
    try std.testing.expectEqualSlices(i32, &.{ 40, 41, 42, 43, 44, 20, 21, 22, 30 }, t.vpos[0..9]);
    try std.testing.expectEqualSlices(i32, &.{ 2000, 2000, 2000, 2000, 2000, 0, 0, 0, 1000 }, t.vbase[0..9]);
    try std.testing.expectEqual(@as(usize, 45), t.end);
    // only the MTP stream keeps a backlog: rows P - m .. P - 1 of slot 0
    try std.testing.expectEqual(@as(usize, 2), t.M);
    try std.testing.expectEqual(@as(usize, 1), t.S);
    try std.testing.expectEqualSlices(i32, &.{ 18, 19 }, t.mpos[0][0..2]);
    try std.testing.expectEqualSlices(i64, &.{6}, t.vdst[0][0..1]);
    // more drafts than the room asked: out of step
    dd.n[0] = 8;
    try std.testing.expectError(error.Invalid, tables(&p, 2, 3, 1000, &dd));
}

test "Phase 4c: modes, the drafter rows and the window with drafters" {
    try std.testing.expectEqual(Mode.dflash, modeOf(.dflash));
    try std.testing.expectEqual(Mode.dspark, modeOf(.dspark));
    try std.testing.expectEqual(@as(?usize, 0), drafterOf(.dflash));
    try std.testing.expectEqual(@as(?usize, 1), drafterOf(.dspark));
    try std.testing.expectEqual(@as(?usize, null), drafterOf(.mtp));
    try std.testing.expectEqual(@as(?Mode, .dspark), modeOfWord(@intFromEnum(Mode.dspark)));
    try std.testing.expectEqual(@as(?Mode, null), modeOfWord(3));
    const df: draft.Spec = .{ .kind = .dflash, .D = 6144, .hd = 128, .H = 16, .KV = 2, .I = 3072, .layers = 6, .ntaps = 6, .block = 8, .top_k = 16, .V = 38720, .vocab_off = 0, .rsel = 256, .gs = 16, .eps = 1e-5, .window = 2047, .causal = false, .world = 4, .theta = 1e6, .mask_id = 154856 };
    var ds = df;
    ds.kind = .dspark;
    try std.testing.expectEqual(@as(usize, 8), try drafterRows(4, &.{ df, ds }));
    try std.testing.expectEqual(@as(usize, 0), try drafterRows(4, &.{}));
    try std.testing.expectEqual(@as(usize, 32), try windowRowsWith(4, 2, 8)); // 4 x max(3, 8)
    try std.testing.expectEqual(@as(usize, 12), try windowRowsWith(4, 2, 0));
    try std.testing.expectEqual(@as(usize, 32), try windowRowsWith(8, 2, 4)); // 8 streams: 3 drafts each
    try std.testing.expectError(error.DrafterTooWide, drafterRows(9, &.{df}));
    var set: Settings = .{};
    try std.testing.expectEqual(@as(f64, 0.6), set.cutDFlash());
    set.draft_cut_dflash = 0.5;
    try std.testing.expectEqual(@as(f64, 0.5), set.cutDFlash());
}
