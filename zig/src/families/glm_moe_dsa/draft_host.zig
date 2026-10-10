//! Phase 4: the speculative drafters' host side, NumPy-free ports of the Python engine's pure parts (no CUDA here, so
//! `zig build test-glm53` runs all of it):
//!   * DSpark (cuda/dspark_host.py): the checkpoint's config.json (DSparkConfig.parse / check), the settings every rank
//!     shares (TF_GLM53_DSPARK_*), the tap plan of every loaded drafter (tap_plan / tap_runs), the sequential stage -
//!     the Markov chain over each slot's merged candidates, the learned confidences and the cut (MarkovChain.walk,
//!     target_kept, best_depth, linear_costs);
//!   * DFlash2 (glm5_next/cuda/dflash2.py + cuda/dflash.py): its config.json and the selector chain (Drafter.chain);
//!   * the candidates' merge (dflash.merge_candidates): every rank's top-k of its vocabulary share, by value then id.
//! Drafts only propose: the target verifies every row and every emitted token is its own pick, so nothing here can
//! change a reply - only how many rows a round keeps. Every rank runs these on the same gathered words, so every rank
//! drafts alike. float64 as NumPy; exp / log are libm's (the Python engine's NumPy calls the same libm in the image).
const std = @import("std");

/// Config errors are logged for users; under test the negative cases expect them (a logged error fails a Zig test).
fn logErr(comptime fmt: []const u8, args: anytype) void {
    if (!@import("builtin").is_test) std.log.err(fmt, args);
}
const smp = @import("sampling.zig");

const libm = struct {
    const exp = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "exp" });
    const log = @extern(*const fn (f64) callconv(.c) f64, .{ .name = "log" });
};

/// The widest tap list a drafter may name (DSpark's deep drafters: up to a dozen taps).
pub const max_taps: usize = 16;
/// The widest block (one tile of DSpark's block attention; DFlash2's block 8).
pub const max_block: usize = 16;
/// dspark_host.TOP_K / NOISE / ROW_COST.
pub const dspark_top_k: usize = 64;
pub const dspark_noise: f64 = 0.7;
pub const dspark_row_cost: f64 = 0.06;
/// dflash2.EDGE / NOISE.
pub const dflash_edge: f64 = 0.6;
pub const dflash_noise: f64 = 0.7;
/// dflash.RING: the drafters' KV slots (a sliding window of 2047 + a block + a tap batch of 64, rounded up).
pub const ring: usize = 4096;
/// Drafter.tap_in's rows: the widest context update a call takes.
pub const tap_rows: usize = 64;

pub const Policy = enum { cost, confidence, fixed };

fn jint(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| i,
        .float => |f| @as(i64, @intFromFloat(f)),
        else => null,
    };
}

fn jfloat(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @as(f64, @floatFromInt(i)),
        .float => |f| f,
        else => null,
    };
}

fn jbool(v: ?std.json.Value, default: bool) bool {
    const x = v orelse return default;
    return switch (x) {
        .bool => |b| b,
        .integer => |i| i != 0,
        else => default,
    };
}

fn jstr(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return switch (x) {
        .string => |s| s,
        else => null,
    };
}

fn jobj(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return switch (x) {
        .object => |o| o,
        else => null,
    };
}

fn need(v: ?i64) !usize {
    const x = v orelse return error.BadDrafterConfig;
    if (x < 0) return error.BadDrafterConfig;
    return @intCast(x);
}

/// zlib.crc32(config.json) & 0x7FFFFFFF: the ranks compare it (dspark_host.words).
pub fn configCrc(text: []const u8) u32 {
    return std.hash.Crc32.hash(text) & 0x7FFFFFFF;
}

// -------------------------------------------------------------------------------------------------- DSpark ---

/// dspark_host.DSparkConfig.
pub const DSparkConfig = struct {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    inter: usize,
    layers: usize,
    eps: f64,
    theta: f64,
    window: usize, // context rows a block sees before its anchor (sliding_window); 0: unbounded
    block: usize,
    mask_id: u32,
    aux: [max_taps]usize = @splat(0), // aux_hidden_state_layer_ids: the INPUTS of those target layers
    n_aux: usize,
    vocab: usize,
    markov_rank: usize,
    confidence: bool,
    confidence_markov: bool,
    causal: bool,
    crc: u32 = 0,

    /// The engine's tap points (layer OUTPUT rows): aux id i is the input of target layer i, layer i - 1's output.
    pub fn tapLayers(c: *const DSparkConfig, out: *[max_taps]usize) []usize {
        for (c.aux[0..c.n_aux], 0..) |a, i| out[i] = a - 1;
        return out[0..c.n_aux];
    }

    /// DSparkConfig.parse over config.json's text.
    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !DSparkConfig {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const cfg = jobj(parsed.value) orelse return error.BadDrafterConfig;
        var kind = jstr(cfg.get("speculators_model_type"));
        if (kind == null) if (jobj(cfg.get("speculators_config"))) |sc| {
            kind = jstr(sc.get("algorithm"));
        };
        if (kind == null or !std.mem.eql(u8, kind.?, "dspark")) {
            logErr("not a DSpark speculator (speculators_model_type {?s})", .{kind});
            return error.NotDSpark;
        }
        const t = jobj(cfg.get("transformer_layer_config")) orelse return error.BadDrafterConfig;
        const heads = try need(jint(t.get("num_attention_heads")));
        const hidden = try need(jint(t.get("hidden_size")));
        const hd: usize = if (jint(t.get("head_dim"))) |x| @intCast(x) else hidden / heads;
        var sliding = jint(t.get("sliding_window")) != null and jbool(t.get("use_sliding_window"), true);
        if (t.get("layer_types")) |lt| if (lt == .array and lt.array.items.len > 0) {
            var any_sliding = false;
            var any_other = false;
            for (lt.array.items) |x| {
                const s = jstr(x) orelse "";
                if (std.mem.eql(u8, s, "sliding_attention")) any_sliding = true else any_other = true;
            }
            if (any_other) {
                if (any_sliding) {
                    logErr("mixed sliding / full DSpark draft layers are not ported", .{});
                    return error.UnsupportedDrafter;
                }
                sliding = false;
            }
        };
        const mtype = jstr(cfg.get("markov_head_type")) orelse "vanilla";
        const rank: usize = if (jint(cfg.get("markov_rank"))) |x| @intCast(@max(x, 0)) else 0;
        if (rank > 0 and !std.mem.eql(u8, mtype, "vanilla")) {
            logErr("only the vanilla Markov head is ported, not {s}", .{mtype});
            return error.UnsupportedDrafter;
        }
        if (!jbool(cfg.get("sample_from_anchor"), true)) {
            logErr("only sample_from_anchor=true is ported", .{});
            return error.UnsupportedDrafter;
        }
        for ([_][]const u8{ "t2d", "d2t" }) |k| if (cfg.get(k)) |v| switch (v) {
            .array, .object, .string, .integer => {
                logErr("a reduced draft vocabulary (t2d / d2t) is not ported", .{});
                return error.UnsupportedDrafter;
            },
            else => {},
        };
        var theta: f64 = 10000.0;
        if (jobj(t.get("rope_parameters"))) |rp| {
            if (jfloat(rp.get("rope_theta"))) |x| theta = x;
            if (jstr(rp.get("rope_type"))) |rt| if (!std.mem.eql(u8, rt, "default")) {
                logErr("only default RoPE is ported", .{});
                return error.UnsupportedDrafter;
            };
        } else if (jfloat(t.get("rope_theta"))) |x| theta = x;
        const conf = jbool(cfg.get("enable_confidence_head"), false);
        const aux_v = cfg.get("aux_hidden_state_layer_ids") orelse return error.BadDrafterConfig;
        if (aux_v != .array or aux_v.array.items.len == 0 or aux_v.array.items.len > max_taps) return error.BadDrafterConfig;
        var c: DSparkConfig = .{
            .hidden = hidden,
            .heads = heads,
            .kv_heads = try need(jint(t.get("num_key_value_heads"))),
            .head_dim = hd,
            .inter = try need(jint(t.get("intermediate_size"))),
            .layers = try need(jint(t.get("num_hidden_layers"))),
            .eps = jfloat(t.get("rms_norm_eps")) orelse return error.BadDrafterConfig,
            .theta = theta,
            .window = if (sliding) try need(jint(t.get("sliding_window"))) else 0,
            .block = try need(jint(cfg.get("block_size"))),
            .mask_id = @intCast(try need(jint(cfg.get("mask_token_id")))),
            .n_aux = aux_v.array.items.len,
            .vocab = if (jint(cfg.get("draft_vocab_size"))) |x| @intCast(x) else try need(jint(t.get("vocab_size"))),
            .markov_rank = rank,
            .confidence = conf,
            .confidence_markov = conf and jbool(cfg.get("confidence_head_with_markov"), false),
            .causal = sliding and !jbool(cfg.get("sliding_window_non_causal"), false),
            .crc = configCrc(text),
        };
        for (aux_v.array.items, 0..) |x, i| c.aux[i] = try need(jint(x));
        return c;
    }

    /// dspark_host.check: refuse a speculator this engine cannot run against this target at `world` ranks.
    pub fn check(c: *const DSparkConfig, world: usize, hidden: usize, vocab: usize, layers: usize) !void {
        if (c.hidden != hidden) {
            logErr("DSpark hidden size {d} != the target's {d}", .{ c.hidden, hidden });
            return error.DrafterMismatch;
        }
        if (c.vocab != vocab) {
            logErr("DSpark draft vocabulary {d} != the target's {d}", .{ c.vocab, vocab });
            return error.DrafterMismatch;
        }
        if (c.heads != c.kv_heads) return error.UnsupportedDrafter; // the block attention is multi-head
        if (c.heads % world != 0 or c.inter % world != 0 or (c.inter / world) % 64 != 0) return error.DrafterSplit;
        if (c.head_dim != 64) return error.UnsupportedDrafter;
        if (c.block == 0 or c.block > max_block) return error.UnsupportedDrafter;
        for (c.aux[0..c.n_aux]) |i| if (i < 1 or i > layers) return error.DrafterMismatch;
        // the context rows a block sees, the block and a tap batch inside the ring (SparkDrafter.__init__)
        if (c.span() + c.block + tap_rows > ring) return error.UnsupportedDrafter;
        if (hidden % 64 != 0 or (c.heads / world * c.head_dim) % 64 != 0) return error.DrafterSplit;
    }

    /// SparkDrafter.span: context rows a block sees (the training mask's window).
    pub fn span(c: *const DSparkConfig) usize {
        return if (c.window > 0) c.window else ring - c.block - tap_rows;
    }
};

/// dspark_host.settings (TF_GLM53_DSPARK_*), every rank alike; the raw environment strings in, null: unset.
pub const DSparkEnv = struct {
    depth: ?[]const u8 = null,
    policy: ?[]const u8 = null,
    confidence: ?[]const u8 = null,
    top_k: ?[]const u8 = null,
    noise: ?[]const u8 = null,
    filter: ?[]const u8 = null,
    row_cost: ?[]const u8 = null,
    quant: ?[]const u8 = null,
    costs: ?[]const u8 = null,
};

pub const DSparkSettings = struct {
    depth: usize,
    policy: Policy = .cost,
    confidence: f64 = 0.3,
    top_k: usize = dspark_top_k,
    noise: f64 = dspark_noise,
    filter: bool = true,
    row_cost: f64 = dspark_row_cost,
    /// TF_GLM53_DSPARK_COSTS: a round's ms with 0, 1, ... drafts (instead of the startup measurement)
    costs: [max_block + 2]f64 = @splat(0),
    n_costs: usize = 0,

    pub fn parse(block: usize, e: DSparkEnv) !DSparkSettings {
        var s: DSparkSettings = .{ .depth = block };
        if (e.depth) |v| s.depth = std.fmt.parseInt(usize, v, 10) catch return error.BadSetting;
        if (e.policy) |v| s.policy = std.meta.stringToEnum(Policy, v) orelse {
            logErr("TF_GLM53_DSPARK_POLICY={s}: cost, confidence or fixed", .{v});
            return error.BadSetting;
        };
        if (e.confidence) |v| s.confidence = std.fmt.parseFloat(f64, v) catch return error.BadSetting;
        if (e.top_k) |v| s.top_k = std.fmt.parseInt(usize, v, 10) catch return error.BadSetting;
        if (e.noise) |v| s.noise = std.fmt.parseFloat(f64, v) catch return error.BadSetting;
        if (e.filter) |v| s.filter = !std.mem.eql(u8, v, "0");
        if (e.row_cost) |v| s.row_cost = std.fmt.parseFloat(f64, v) catch return error.BadSetting;
        if (e.quant) |v| if (!std.mem.eql(u8, v, "q4")) {
            logErr("TF_GLM53_DSPARK_QUANT={s}: the Zig engine keeps DSpark's weights 4-bit (q4) only", .{v});
            return error.BadSetting;
        };
        if (e.costs) |v| {
            var it = std.mem.tokenizeScalar(u8, v, ',');
            while (it.next()) |x| {
                if (s.n_costs == s.costs.len) return error.BadSetting;
                const ms = std.fmt.parseFloat(f64, std.mem.trim(u8, x, " ")) catch return error.BadSetting;
                if (!(ms > 0)) return error.BadSetting;
                s.costs[s.n_costs] = ms;
                s.n_costs += 1;
            }
            if (s.n_costs < 2) return error.BadSetting;
        }
        s.depth = @min(s.depth, block);
        if (s.top_k < 1 or s.top_k > 256) return error.BadSetting;
        return s;
    }

    /// The words every rank must share (dspark_host.words' fields), hashed into the startup digest.
    pub fn digest(s: *const DSparkSettings, h: *std.hash.Wyhash) void {
        const words = [_]u64{ s.depth, @intFromEnum(s.policy), @intFromFloat(@round(s.confidence * 1e6)), s.top_k, @intFromFloat(@round(s.noise * 1e6)), @intFromBool(s.filter), @intFromFloat(@round(s.row_cost * 1e6)), s.n_costs };
        h.update(std.mem.asBytes(&words));
        h.update(std.mem.sliceAsBytes(s.costs[0..s.n_costs]));
    }
};

// ------------------------------------------------------------------------------------------------- DFlash2 ---

/// glm5_next dflash2.Drafter's config.json fields (and dflash.GlmDrafter's).
pub const DFlashConfig = struct {
    hidden: usize,
    head_dim: usize,
    heads: usize,
    kv_heads: usize,
    inter: usize,
    layers: usize,
    eps: f64,
    theta: f64,
    mask_id: u32,
    gs: usize, // conv_group_size
    block: usize,
    top_k: usize, // selector_top_k
    taps: [max_taps]usize = @splat(0), // target_layer_ids: the OUTPUTS of those layers
    n_taps: usize,
    window: usize, // sliding_window - 1
    causal: bool,
    crc: u32 = 0,

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !DFlashConfig {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const cfg = jobj(parsed.value) orelse return error.BadDrafterConfig;
        const dc = jobj(cfg.get("dflash_config")) orelse {
            logErr("not a DFlash2 drafter (no dflash_config)", .{});
            return error.NotDFlash;
        };
        if ((jint(dc.get("conv_kernel_size")) orelse 0) != 2) {
            logErr("only two-tap DFlash2 convolutions are ported", .{});
            return error.UnsupportedDrafter;
        }
        const rp = jobj(cfg.get("rope_parameters")) orelse return error.BadDrafterConfig;
        const tl = dc.get("target_layer_ids") orelse return error.BadDrafterConfig;
        if (tl != .array or tl.array.items.len == 0 or tl.array.items.len > max_taps) return error.BadDrafterConfig;
        const sw = try need(jint(cfg.get("sliding_window")));
        if (sw == 0) return error.BadDrafterConfig;
        var c: DFlashConfig = .{
            .hidden = try need(jint(cfg.get("hidden_size"))),
            .head_dim = try need(jint(cfg.get("head_dim"))),
            .heads = try need(jint(cfg.get("num_attention_heads"))),
            .kv_heads = try need(jint(cfg.get("num_key_value_heads"))),
            .inter = try need(jint(cfg.get("intermediate_size"))),
            .layers = try need(jint(cfg.get("num_hidden_layers"))),
            .eps = jfloat(cfg.get("rms_norm_eps")) orelse return error.BadDrafterConfig,
            .theta = jfloat(rp.get("rope_theta")) orelse return error.BadDrafterConfig,
            .mask_id = @intCast(try need(jint(dc.get("mask_token_id")))),
            .gs = try need(jint(dc.get("conv_group_size"))),
            .block = try need(jint(dc.get("block_size"))),
            .top_k = try need(jint(dc.get("selector_top_k"))),
            .n_taps = tl.array.items.len,
            .window = sw - 1,
            .causal = jbool(cfg.get("is_causal"), true),
            .crc = configCrc(text),
        };
        for (tl.array.items, 0..) |x, i| c.taps[i] = try need(jint(x));
        return c;
    }

    pub fn check(c: *const DFlashConfig, world: usize, hidden: usize, layers: usize) !void {
        if (c.hidden != hidden) return error.DrafterMismatch;
        if (c.heads % world != 0 or c.kv_heads % world != 0 or c.inter % world != 0 or (c.inter / world) % 64 != 0) return error.DrafterSplit;
        if (c.block < 2 or c.block > max_block) return error.UnsupportedDrafter;
        if (c.gs == 0 or c.hidden % c.gs != 0) return error.BadDrafterConfig;
        if (c.window + c.block + tap_rows > ring) return error.UnsupportedDrafter;
        if (c.top_k == 0 or c.top_k > 256) return error.UnsupportedDrafter;
        // _dattn_ring: G * N query rows a program (a power of two for tl.arange), head_dim a power of two
        const g = (c.heads / world) / (c.kv_heads / world);
        if (!std.math.isPowerOfTwo(g * c.block) or !std.math.isPowerOfTwo(c.head_dim)) return error.UnsupportedDrafter;
        for (c.taps[0..c.n_taps]) |i| if (i >= layers) return error.DrafterMismatch;
    }
};

/// runner._dflash_cfg (TF_GLM53_DFLASH_DEPTH 7, TF_GLM53_DFLASH_CONFIDENCE 0.3).
pub const DFlashSettings = struct {
    depth: usize = 7,
    confidence: f64 = 0.3,

    pub fn digest(s: *const DFlashSettings, h: *std.hash.Wyhash) void {
        const words = [_]u64{ s.depth, @intFromFloat(@round(s.confidence * 1e6)) };
        h.update(std.mem.asBytes(&words));
    }
};

// ---------------------------------------------------------------------------------------------------- taps ---

/// dspark_host.tap_plan: each loaded drafter's tap layers (null: not loaded) -> the engine's slot of every tapped
/// layer (first-seen order: the first drafter keeps the layout it has alone; a layer two drafters share is tapped
/// once) and each drafter's column blocks in its own layer order.
pub const TapPlan = struct {
    /// tapped layers in slot order
    layers: [2 * max_taps]usize = @splat(0),
    n: usize = 0,
    /// per drafter (DFlash2, DSpark): its column blocks of the taps buffer in its own order
    cols: [2][max_taps]usize = @splat(@splat(0)),
    n_cols: [2]usize = .{ 0, 0 },

    pub fn slotOf(p: *const TapPlan, layer: usize) ?usize {
        for (p.layers[0..p.n], 0..) |l, i| if (l == layer) return i;
        return null;
    }

    pub fn make(wanted: [2]?[]const usize) !TapPlan {
        var p: TapPlan = .{};
        for (wanted) |w| {
            const ls = w orelse continue;
            for (ls) |l| if (p.slotOf(l) == null) {
                if (p.n == p.layers.len) return error.TooManyTaps;
                p.layers[p.n] = l;
                p.n += 1;
            };
        }
        for (wanted, 0..) |w, d| {
            const ls = w orelse continue;
            for (ls, 0..) |l, j| p.cols[d][j] = p.slotOf(l).?;
            p.n_cols[d] = ls.len;
        }
        return p;
    }

    /// Whether drafter d's column blocks are the buffer's own order (one contiguous copy).
    pub fn identity(p: *const TapPlan, d: usize) bool {
        if (p.n_cols[d] != p.n) return false;
        for (p.cols[d][0..p.n_cols[d]], 0..) |c, j| if (c != j) return false;
        return true;
    }
};

/// dspark_host.tap_runs: column blocks as runs (own first block, source first block, width).
pub fn tapRuns(cols: []const usize, out: *[max_taps][3]usize) [][3]usize {
    var n: usize = 0;
    for (cols, 0..) |s, j| {
        if (n > 0 and out[n - 1][1] + out[n - 1][2] == s) {
            out[n - 1][2] += 1;
        } else {
            out[n] = .{ j, s, 1 };
            n += 1;
        }
    }
    return out[0..n];
}

// ------------------------------------------------------------------------------------------- candidates ---

/// dflash.merge_candidates: `g` [ranks][rows_all * 2 k] words (each rank's rows: k values, then k ids as int32 bits;
/// row stride 2 k, `stride` words a rank) -> rows 0..depth-1 of `tokens` / `values` [depth][k]: every rank's
/// candidates by value descending, then id ascending (numpy lexsort((tokens, -values))), the first k. The Zig top-k
/// leaves a rank's k in column order, so one rank sorts too (torch.topk's order is that order on distinct values).
pub fn mergeCandidates(g: []const f32, ranks: usize, stride: usize, depth: usize, k: usize, tokens: []i64, values: []f64, scratch: []smp.Cand) void {
    std.debug.assert(scratch.len >= ranks * k);
    for (0..depth) |d| {
        var n: usize = 0;
        for (0..ranks) |r| {
            const row = g[r * stride + d * 2 * k ..][0 .. 2 * k];
            const ids: []const i32 = @ptrCast(row[k..]);
            for (0..k) |q| {
                scratch[n] = .{ .v = row[q], .id = ids[q] };
                n += 1;
            }
        }
        std.sort.pdq(smp.Cand, scratch[0..n], {}, candLess);
        for (0..k) |q| {
            tokens[d * k + q] = scratch[q].id;
            values[d * k + q] = scratch[q].v;
        }
    }
}

fn candLess(_: void, a: smp.Cand, b: smp.Cand) bool {
    // numpy lexsort((ids, -values)): -value ascending (NaN last), then id ascending
    const na = -a.v;
    const nb = -b.v;
    const an = std.math.isNan(na);
    const bn = std.math.isNan(nb);
    if (an != bn) return bn;
    if (!an and na != nb) return na < nb;
    return a.id < b.id;
}

// -------------------------------------------------------------------------------- the sequential stages ---

/// bf16 bits -> float64 (bf16 is float32's top half: exact).
pub fn bf16f(bits: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, bits) << 16)));
}

/// numpy argmax: the first maximum (a NaN wins at once).
fn argmax(v: []const f64) usize {
    var best: usize = 0;
    for (1..v.len) |i| {
        if (std.math.isNan(v[best])) break;
        if (std.math.isNan(v[i]) or v[i] > v[best]) best = i;
    }
    return best;
}

/// exact_sampling.uniform_rows' Gumbel noise of one slot: -log(-log(u(seed, position, id))) for its candidates.
fn gumbelRow(seed: u64, position: u64, ids: []const i64, out: []f64) void {
    for (ids, 0..) |id, j| out[j] = -libm.log(-libm.log(smp.uniform(seed, position, @bitCast(id))));
}

/// The pick's probability among the candidates: exp(score - max)[j] / sum (the confidence without a head).
fn softmaxAt(score: []const f64, j: usize) f64 {
    var mx = score[0];
    for (score[1..]) |v| mx = @max(mx, v);
    var total: f64 = 0;
    var at: f64 = 0;
    for (score, 0..) |v, i| {
        const e = libm.exp(v - mx);
        total += e;
        if (i == j) at = e;
    }
    return at / total;
}

/// dspark_host.target_kept: which of a slot's candidates the target's sampling rule keeps, judged on the drafter's
/// scores (already over the temperature): top_k, the top_p prefix, min_p, in choose_rows' order.
pub fn targetKept(score: []const f64, ids: []const i64, s: smp.Sampling, kept: []bool, order: []usize) void {
    const n = score.len;
    for (0..n) |i| order[i] = i;
    const Ctx = struct {
        sc: []const f64,
        id: []const i64,
        fn lt(c: @This(), a: usize, b: usize) bool {
            const na = -c.sc[a];
            const nb = -c.sc[b];
            const an = std.math.isNan(na);
            const bn = std.math.isNan(nb);
            if (an != bn) return bn;
            if (!an and na != nb) return na < nb;
            return c.id[a] < c.id[b];
        }
    };
    std.sort.pdq(usize, order[0..n], Ctx{ .sc = score, .id = ids }, Ctx.lt);
    var k = if (s.top_k > 0) @min(s.top_k, n) else n;
    k = @max(k, 1);
    if (0.0 < s.top_p and s.top_p < 1.0) {
        const mx = score[order[0]];
        var total: f64 = 0;
        for (order[0..k]) |i| total += libm.exp(score[i] - mx);
        var cum: f64 = 0;
        var below: usize = 0;
        for (order[0..k]) |i| {
            cum += libm.exp(score[i] - mx) / total;
            if (cum < s.top_p) below += 1;
        }
        k = @min(k, below + 1);
    }
    @memset(kept[0..n], false);
    const floor = if (s.min_p > 0.0) score[order[0]] + libm.log(s.min_p) else -std.math.inf(f64);
    for (order[0..k]) |i| {
        if (score[i] >= floor) kept[i] = true;
    }
}

/// dspark_host.best_depth: the k (0 .. conf.len) maximizing E[tokens | k drafts] / round_ms[k], E = 1 + sum_{j <= k}
/// prod_{i <= j} conf_i; round_ms shorter than conf + 1: the last value's slope continues.
pub fn bestDepth(conf: []const f64, round_ms: []const f64) usize {
    var ms: [max_block + 2]f64 = undefined;
    const want = conf.len + 1;
    var n: usize = @min(round_ms.len, want);
    @memcpy(ms[0..n], round_ms[0..n]);
    if (n == 0) {
        ms[0] = 1.0;
        n = 1;
    }
    while (n < want) : (n += 1) {
        ms[n] = ms[n - 1] + (if (n > 1) ms[n - 1] - ms[n - 2] else ms[n - 1] * dspark_row_cost);
    }
    var alive: f64 = 1.0;
    var expect: f64 = 1.0;
    var best: usize = 0;
    var best_v = expect / ms[0];
    for (conf, 0..) |c, i| {
        alive *= c;
        expect += alive;
        const v = expect / ms[i + 1];
        if (v > best_v) {
            best_v = v;
            best = i + 1;
        }
    }
    return best;
}

/// dspark_host.linear_costs: 1 + row_cost x k for k = 0 .. depth.
pub fn linearCosts(depth: usize, row_cost: f64, out: []f64) []f64 {
    for (0..depth + 1) |k| out[k] = 1.0 + row_cost * @as(f64, @floatFromInt(k));
    return out[0 .. depth + 1];
}

/// What a walk needs besides the candidates.
pub const WalkOpts = struct {
    sampling: ?smp.Sampling = null,
    policy: Policy = .cost,
    confidence: f64 = 0.3,
    round_ms: ?[]const f64 = null,
    noise: f64 = dspark_noise,
    filtered: bool = true,
    row_cost: f64 = dspark_row_cost,
};

/// Scratch a walk uses (K candidates a slot).
pub const WalkScratch = struct {
    score: []f64,
    pick: []f64,
    gumbel: []f64,
    kept: []bool,
    order: []usize,
    emb: []f64, // [rank]

    pub fn init(gpa: std.mem.Allocator, k: usize, rank: usize) !WalkScratch {
        return .{ .score = try gpa.alloc(f64, k), .pick = try gpa.alloc(f64, k), .gumbel = try gpa.alloc(f64, k), .kept = try gpa.alloc(bool, k), .order = try gpa.alloc(usize, k), .emb = try gpa.alloc(f64, @max(rank, 1)) };
    }

    pub fn deinit(w: *WalkScratch, gpa: std.mem.Allocator) void {
        gpa.free(w.score);
        gpa.free(w.pick);
        gpa.free(w.gumbel);
        gpa.free(w.kept);
        gpa.free(w.order);
        gpa.free(w.emb);
    }
};

/// dspark_host.MarkovChain over host copies of the Markov tables (bf16 bits [vocab, rank]) and the confidence head's
/// Markov part.
pub const MarkovChain = struct {
    w1: ?[]const u16 = null,
    w2: ?[]const u16 = null,
    rank: usize = 0,
    conf_m: ?[]const f64 = null,
    conf_b: f64 = 0,
    confident: bool = false,
    /// the last walk's per-slot confidences
    last: [max_block]f64 = @splat(0),
    n_last: usize = 0,

    fn embed(m: *const MarkovChain, token: i64, out: []f64) bool {
        const w1 = m.w1 orelse return false;
        const row = w1[@as(usize, @intCast(token)) * m.rank ..][0..m.rank];
        for (row, 0..) |b, i| out[i] = bf16f(b);
        return true;
    }

    /// W2[cand] . emb in float64.
    fn bias(m: *const MarkovChain, cand: i64, emb: []const f64) f64 {
        const w2 = m.w2 orelse return 0;
        const row = w2[@as(usize, @intCast(cand)) * m.rank ..][0..m.rank];
        var acc: f64 = 0;
        for (row, emb[0..m.rank]) |b, e| acc += bf16f(b) * e;
        return acc;
    }

    /// MarkovChain.walk: drafts for slots 0 .. depth - 1 (candidates `tokens` [depth][k] with base logits `values`,
    /// merged by value then id; `hconf` [depth] the confidence head's hidden part), left to right from `anchor`;
    /// `first`: the position of slot 0's token (keyed noise). Cut by the policy. Returns the drafts in `out`.
    pub fn walk(m: *MarkovChain, tokens: []const i64, values: []const f64, hconf: []const f64, depth: usize, k: usize, anchor: i64, first: u64, o: WalkOpts, sc: *WalkScratch, out: []u32) []u32 {
        const sampled = o.sampling != null and o.sampling.?.temperature > 0;
        const temp: f64 = if (sampled) o.sampling.?.temperature else 1.0;
        var prev = anchor;
        var alive: f64 = 1.0;
        var n: usize = 0;
        m.n_last = 0;
        for (0..depth) |d| {
            const has_emb = m.embed(prev, sc.emb);
            const cand = tokens[d * k ..][0..k];
            const vals = values[d * k ..][0..k];
            for (0..k) |j| {
                const b = if (has_emb) m.bias(cand[j], sc.emb) else 0;
                sc.score[j] = (vals[j] + b) / temp;
                sc.pick[j] = sc.score[j];
            }
            if (sampled) {
                const s = o.sampling.?;
                gumbelRow(s.seed, first + d, cand, sc.gumbel[0..k]);
                for (0..k) |j| sc.pick[j] = sc.score[j] + o.noise * sc.gumbel[j];
                if (o.filtered) {
                    targetKept(sc.score[0..k], cand, s, sc.kept, sc.order);
                    for (0..k) |j| if (!sc.kept[j]) {
                        sc.pick[j] = -std.math.inf(f64);
                    };
                }
            }
            const j = argmax(sc.pick[0..k]);
            var c: f64 = undefined;
            if (m.confident) {
                var z = hconf[d] + m.conf_b;
                if (m.conf_m != null and has_emb) {
                    for (m.conf_m.?, sc.emb[0..m.rank]) |a, e| z += a * e;
                }
                c = if (z > -700) 1.0 / (1.0 + libm.exp(-z)) else 0.0;
            } else {
                c = softmaxAt(sc.score[0..k], j);
            }
            if (o.policy == .confidence and o.confidence > 0) {
                alive *= c;
                if (d > 0 and alive < o.confidence) break;
            }
            m.last[n] = c;
            prev = cand[j];
            out[n] = @intCast(prev);
            n += 1;
        }
        m.n_last = n;
        if (o.policy == .cost and n > 0) {
            var lin: [max_block + 2]f64 = undefined;
            const rms = o.round_ms orelse linearCosts(depth, o.row_cost, &lin);
            n = @min(n, bestDepth(m.last[0..n], rms));
        }
        return out[0..n];
    }
};

/// dflash2.Drafter.chain: candidates by selector edges (succ[t] . (pred[prev] * proj_d)) and the target's keyed noise;
/// stop below the cumulative confidence, the first draft always kept. `pred` / `succ` [vocab, rank] fp32, `proj`
/// [depth, rank].
pub fn selectorChain(pred: []const f32, succ: []const f32, rank: usize, tokens: []const i64, values: []const f64, proj: []const f64, depth: usize, k: usize, anchor: i64, first: u64, sampling: ?smp.Sampling, confidence: f64, sc: *WalkScratch, out: []u32) []u32 {
    const sampled = sampling != null and sampling.?.temperature > 0;
    const temp: f64 = if (sampled) sampling.?.temperature else 1.0;
    var prev = anchor;
    var chain: f64 = 1.0;
    var n: usize = 0;
    for (0..depth) |d| {
        const cand = tokens[d * k ..][0..k];
        const vals = values[d * k ..][0..k];
        const pr = pred[@as(usize, @intCast(prev)) * rank ..][0..rank];
        const pj = proj[d * rank ..][0..rank];
        for (0..rank) |i| sc.emb[i] = @as(f64, pr[i]) * pj[i];
        for (0..k) |j| {
            const su = succ[@as(usize, @intCast(cand[j])) * rank ..][0..rank];
            var edge: f64 = 0;
            for (su, sc.emb[0..rank]) |a, b| edge += @as(f64, a) * b;
            sc.score[j] = (vals[j] + dflash_edge * edge) / temp;
            sc.pick[j] = sc.score[j];
        }
        if (sampled) {
            gumbelRow(sampling.?.seed, first + d, cand, sc.gumbel[0..k]);
            for (0..k) |j| sc.pick[j] = sc.score[j] + dflash_noise * sc.gumbel[j];
        }
        const j = argmax(sc.pick[0..k]);
        if (confidence > 0) {
            chain *= softmaxAt(sc.score[0..k], j);
            if (d > 0 and chain < confidence) break;
        }
        prev = cand[j];
        out[n] = @intCast(prev);
        n += 1;
    }
    return out[0..n];
}

// ------------------------------------------------------------------------------------------------- tests ---

const ft2_config =
    \\{"aux_hidden_state_layer_ids": [2, 20, 39, 58, 75], "block_size": 8, "confidence_head_with_markov": true,
    \\ "draft_vocab_size": 154880, "enable_confidence_head": true, "markov_head_type": "vanilla", "markov_rank": 256,
    \\ "mask_token_id": 154856, "sample_from_anchor": true, "sliding_window_non_causal": false,
    \\ "speculators_config": {"algorithm": "dspark"}, "speculators_model_type": "dspark",
    \\ "transformer_layer_config": {"head_dim": 64, "hidden_size": 6144, "intermediate_size": 12288,
    \\   "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention"], "num_attention_heads": 64,
    \\   "num_hidden_layers": 3, "num_key_value_heads": 64, "rms_norm_eps": 1e-05,
    \\   "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"}, "sliding_window": 2048,
    \\   "use_sliding_window": true, "vocab_size": 154880}}
;

const dflash2_config =
    \\{"dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 154856,
    \\   "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61, 75]},
    \\ "head_dim": 128, "hidden_size": 6144, "intermediate_size": 12288, "is_causal": false,
    \\ "num_attention_heads": 64, "num_hidden_layers": 6, "num_key_value_heads": 8, "rms_norm_eps": 1e-05,
    \\ "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}, "sliding_window": 2048, "vocab_size": 154880}
;

test "DSpark config: the published ft2 drafter's fields and checks" {
    const c = try DSparkConfig.parse(std.testing.allocator, ft2_config);
    try std.testing.expectEqual(@as(usize, 3), c.layers);
    try std.testing.expectEqual(@as(usize, 2048), c.window);
    try std.testing.expect(c.causal and c.confidence and c.confidence_markov);
    try std.testing.expectEqual(@as(f64, 8000000), c.theta);
    var tl: [max_taps]usize = undefined;
    try std.testing.expectEqualSlices(usize, &.{ 1, 19, 38, 57, 74 }, c.tapLayers(&tl));
    try c.check(4, 6144, 154880, 78);
    try std.testing.expectError(error.DrafterMismatch, c.check(4, 6144, 151552, 78));
    try std.testing.expectError(error.NotDSpark, DSparkConfig.parse(std.testing.allocator, dflash2_config));
}

test "DFlash2 config: the DFlash2 drafter's fields" {
    const c = try DFlashConfig.parse(std.testing.allocator, dflash2_config);
    try std.testing.expectEqual(@as(usize, 6), c.n_taps);
    try std.testing.expectEqual(@as(usize, 2047), c.window);
    try std.testing.expect(!c.causal);
    try std.testing.expectEqual(@as(usize, 16), c.top_k);
    try c.check(4, 6144, 78);
}

test "DSpark settings: env strings, the champion's confidence policy" {
    const s = try DSparkSettings.parse(8, .{ .policy = "confidence", .confidence = "0.3" });
    try std.testing.expectEqual(Policy.confidence, s.policy);
    try std.testing.expectEqual(@as(usize, 8), s.depth);
    try std.testing.expectError(error.BadSetting, DSparkSettings.parse(8, .{ .policy = "greedy" }));
    try std.testing.expectError(error.BadSetting, DSparkSettings.parse(8, .{ .quant = "bf16" }));
    const t = try DSparkSettings.parse(8, .{ .depth = "12", .costs = "40,42.5,45" });
    try std.testing.expectEqual(@as(usize, 8), t.depth);
    try std.testing.expectEqual(@as(usize, 3), t.n_costs);
}

test "tap plan as dspark_host.tap_plan" {
    // DFlash2 alone: its own order
    const a = try TapPlan.make(.{ &.{ 5, 19, 33, 47, 61, 75 }, null });
    try std.testing.expectEqual(@as(usize, 6), a.n);
    try std.testing.expect(a.identity(0));
    // both: DSpark's layers after DFlash2's, 75 shared
    const b = try TapPlan.make(.{ &.{ 5, 19, 33, 47, 61, 75 }, &.{ 1, 19, 38, 57, 74 } });
    try std.testing.expectEqual(@as(usize, 10), b.n);
    try std.testing.expectEqualSlices(usize, &.{ 6, 1, 7, 8, 9 }, b.cols[1][0..5]);
    try std.testing.expect(!b.identity(1));
    var runs: [max_taps][3]usize = undefined;
    const r = tapRuns(&.{ 0, 1, 2, 7 }, &runs);
    try std.testing.expectEqual(@as(usize, 2), r.len);
    try std.testing.expectEqual([3]usize{ 0, 0, 3 }, r[0]);
    try std.testing.expectEqual([3]usize{ 3, 7, 1 }, r[1]);
}

test "best depth and linear costs as dspark_host" {
    var lin: [max_block + 2]f64 = undefined;
    // certain drafts: deeper is better while the row cost is small
    try std.testing.expectEqual(@as(usize, 3), bestDepth(&.{ 1, 1, 1 }, linearCosts(3, 0.06, &lin)));
    // hopeless drafts: none
    try std.testing.expectEqual(@as(usize, 0), bestDepth(&.{ 0.01, 0.01 }, linearCosts(2, 0.06, &lin)));
    // round_ms shorter than the conf: the slope continues (10, 11 -> 12, 13)
    try std.testing.expectEqual(@as(usize, 3), bestDepth(&.{ 0.9, 0.9, 0.9 }, &.{ 10, 11 }));
}

test "merge candidates: value descending, id ascending over the ranks" {
    // two ranks, one slot, k = 2: rank 0 (5 @ 10, 3 @ 11), rank 1 (5 @ 2, 4 @ 3)
    var g: [8]f32 = undefined;
    const ids0 = [_]i32{ 10, 11 };
    const ids1 = [_]i32{ 2, 3 };
    g[0] = 5;
    g[1] = 3;
    g[2] = @bitCast(ids0[0]);
    g[3] = @bitCast(ids0[1]);
    g[4] = 5;
    g[5] = 4;
    g[6] = @bitCast(ids1[0]);
    g[7] = @bitCast(ids1[1]);
    var tok: [2]i64 = undefined;
    var val: [2]f64 = undefined;
    var scratch: [4]smp.Cand = undefined;
    mergeCandidates(&g, 2, 4, 1, 2, &tok, &val, &scratch);
    try std.testing.expectEqualSlices(i64, &.{ 2, 10 }, &tok);
    try std.testing.expectEqual(@as(f64, 5), val[1]);
}

test "Markov walk: greedy picks, the confidence cut keeps the first draft" {
    const gpa = std.testing.allocator;
    var sc = try WalkScratch.init(gpa, 3, 2);
    defer sc.deinit(gpa);
    // no Markov tables, a confidence head with no hidden part: z = hconf + b
    var m: MarkovChain = .{ .confident = true, .conf_b = 0 };
    const tokens = [_]i64{ 7, 8, 9, 4, 5, 6, 1, 2, 3 };
    const values = [_]f64{ 1, 3, 2, 9, 0, 0, 0, 0, 5 };
    const hconf = [_]f64{ -3, -3, 5 }; // sigmoid(-3) ~ 0.047: the product falls under 0.3 at slot 1
    var out: [8]u32 = undefined;
    const got = m.walk(&tokens, &values, &hconf, 3, 3, 0, 100, .{ .policy = .confidence, .confidence = 0.3 }, &sc, &out);
    try std.testing.expectEqualSlices(u32, &.{8}, got);
    const all = m.walk(&tokens, &values, &hconf, 3, 3, 0, 100, .{ .policy = .fixed }, &sc, &out);
    try std.testing.expectEqualSlices(u32, &.{ 8, 4, 3 }, all);
}

test "target kept: top_k cut and min_p" {
    var kept: [4]bool = undefined;
    var order: [4]usize = undefined;
    targetKept(&.{ 1, 4, 3, 2 }, &.{ 10, 11, 12, 13 }, .{ .seed = 1, .top_k = 2, .top_p = 1.0 }, &kept, &order);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, &kept);
    targetKept(&.{ 1, 4, 3, 2 }, &.{ 10, 11, 12, 13 }, .{ .seed = 1, .top_k = 0, .top_p = 1.0, .min_p = 0.5 }, &kept, &order);
    // ln 0.5 ~ -0.69: only the top (4) and nothing within 0.69 of it
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, &kept);
}

test "selector chain: edges move the pick" {
    const gpa = std.testing.allocator;
    var sc = try WalkScratch.init(gpa, 2, 1);
    defer sc.deinit(gpa);
    // vocab 4, rank 1: pred[0] = 1; succ[2] = 10, succ[3] = 0; proj 1 -> edge favours token 2
    const pred = [_]f32{ 1, 1, 1, 1 };
    const succ = [_]f32{ 0, 0, 10, 0 };
    const tokens = [_]i64{ 3, 2 };
    const values = [_]f64{ 1, 0 };
    const proj = [_]f64{1};
    var out: [4]u32 = undefined;
    const got = selectorChain(&pred, &succ, 1, &tokens, &values, &proj, 1, 2, 0, 5, null, 0, &sc, &out);
    try std.testing.expectEqualSlices(u32, &.{2}, got);
}
