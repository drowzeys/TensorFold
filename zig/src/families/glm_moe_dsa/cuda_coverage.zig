//! Triton AOT coverage: every launch shape the GLM-5.3 engine can ask cuda_triton.Tri for, enumerated with Tri itself
//! in probe mode (no GPU: each launch becomes an aot.Need - function, launch options, constexprs, each runtime
//! argument's specialization class), so the bucket choices (glue.router's BM, the absorb / expand RBK, rmsnorm's
//! warps, the attention tilings, ...) are the launch code's own, not a copy of them. `enumerate` drives every Tri
//! method over a superset of what the call sites in cuda_forward.zig / cuda_prompt.zig pass:
//!   * window rows R = 1 .. max_rows (every decode / verify / MTP / resumed-reuse window and every prompt chunk or
//!     sequence-parallel half up to the served prompt rows), each R through every per-window launch, both attention
//!     tilings (decode and prompt: which one is chosen by R <= b.small) and both residual sums (world 1: RoCE / ring
//!     / one rank; world W: the NCCL all-gather);
//!   * the prompt GEMM (x3prefill._gemm) for every EXL3 linear shape of every rank (the tile tables) and both output
//!     dtypes, at every R above fw.max_rows;
//!   * the indexer's selection blocks n = 1 .. 128 over index ranges T that reach every specialization class of T and
//!     of DCP's ceil(T / dcp) (decode buckets and arbitrary prompt-chunk ends), row offsets R0 of every class;
//!   * DCP 1 and DCP = world with every rank's RANK constexpr;
//!   * Phase 4b (`rows_windows` > 0, --parallel N): the concurrent decode windows of 1 .. N x (k + 1) rows with per-row
//!     positions and cache bases (fused.Rows: _kv_write / _ik_write / _iq_rope / _absorb / _attn_chunks /
//!     _index_scores with ROWS and a BASE table), their selections over the decode key buckets;
//!   * Phase 4c (`streams` > 1 with drafters): the concurrent drafters' passes (cuda_mdraft.enumerate: context updates
//!     of every row bucket, block passes of 1 .. N blocks - DFlash2's _dconv_seg / _dattn_seg, DSpark's per-block
//!     _prep_kernel / _block_attn_kernel and the row-wise launches over every block).
//! Arguments that reach a kernel only as runtime ints (rows, T, R0, strides) select a variant by class alone (the value
//! 1, a multiple of 16, other): aot.pickVariant's own rule, so walking every class is walking every variant. Pointers are
//! 16-aligned here: every call site's pointer is a buffer base plus whole rows of 16-byte multiples (D, q_lora,
//! index_dim, experts, top_k, index_topk rows), which the tests below pin for GLM-5.3's dimensions.
//! The source test checks every Triton function cuda_triton.zig launches is reached by `enumerate`.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const Config = @import("config.zig").Config;
const exl3 = @import("exl3.zig");
const tri = @import("cuda_triton.zig");
const dcpm = @import("cuda_dcp.zig");
const draft = @import("cuda_draft.zig");
const mdraft = @import("cuda_mdraft.zig");

/// The engine's own constants (cuda_forward.zig): kept equal by the test below.
pub const max_window_rows: usize = 128; // fw.max_rows
pub const sel_rows: usize = 128; // fused.SEL_ROWS (select's row blocks)

/// What the enumeration needs to know about one served configuration.
pub const Space = struct {
    cfg: Config,
    world: usize,
    /// this rank's attention heads (cfg.heads / world)
    heads: usize,
    /// each rank's lm_head rows (cfg.vocab / world)
    vocab_part: usize,
    /// kv_a's stored outputs (kva row stride)
    kva_n: usize,
    /// the widest window / prompt chunk (the served prompt rows)
    max_rows: usize,
    /// the largest context (index ranges up to its power of two)
    capacity: usize,
    /// DCP degrees to cover (1 and/or world)
    dcps: []const usize,
    /// SwiGLU widths (dense MLP and shared expert gate outputs, every rank)
    widths: []const usize,
    /// (k, n) of every EXL3 linear of every rank (the prompt GEMM's shapes)
    gemms: []const [2]usize,
    mtp: bool = true,
    draft: bool = true,
    /// Phase 4: the loaded drafters' shapes (cuda_draft.Spec): their tap passes and block pass
    drafters: []const draft.Spec = &.{},
    /// Phase 4b: the widest concurrent decode window (--parallel N: N x max(k + 1, drafter rows) rows); 0: one stream
    rows_windows: usize = 0,
    /// Phase 4c: --parallel N (> 1): the drafters' concurrent passes for N streams too
    streams: usize = 0,
};

const A: u64 = 1 << 20; // a 16-aligned device address (no memory is touched)

fn mla(sp: Space) tri.Mla {
    const c = sp.cfg;
    return .{ .heads = sp.heads, .nope = c.nope, .rope = c.rope, .lw = c.kv_lora, .vd = c.v_dim, .topk = c.index_topk };
}

fn ranksOf(dcp: usize) usize {
    return if (dcp > 1) dcp else 1;
}

/// Every launch of the space into `p`.
pub fn enumerate(p: *aot.Probe, sp: Space) !void {
    const t: tri.Tri = .{ .set = null, .s = undefined, .probe = p };
    for (1..sp.max_rows + 1) |R| try window(t, sp, R);
    try selects(t, sp);
    for (1..sp.rows_windows + 1) |R| try rowsWindow(t, sp, R);
    for (sp.drafters) |ds| try draft.enumerate(p, ds);
    if (sp.streams > 1) for (sp.drafters) |ds| try mdraft.enumerate(p, ds, sp.streams);
}

/// One window of R rows: every per-window launch of fused.layer (target and MTP layers), the head, the MTP input
/// and the prompt path's sequence-parallel extras.
fn window(t: tri.Tri, sp: Space, R: usize) !void {
    const c = sp.cfg;
    const D = c.hidden;
    const ql = c.q_lora;
    const m = mla(sp);
    // norms: input / post-attention / final / MTP enorm, hnorm, head_norm, target_hidden (x_stride D) and q_a_norm
    try t.rmsnorm(A, D, A, A, D, R, D, c.eps);
    try t.rmsnorm(A, ql, A, A, ql, R, ql, c.eps);
    // residual sums: world 1 (RoCE / ring / one rank) and the NCCL all-gather's world slots; the SP halves' bf16
    for ([_]usize{ 1, sp.world }) |w| try t.residualAdd(A, A, R, D, w);
    try t.residualAddOf(A, A, "*bf16", R, D, 1);
    // glue.router: the indexer's weights_proj and wk, the MoE router, the head (lm_head share), the MTP eh_proj
    for ([_]usize{ c.index_heads, c.index_dim, c.experts, sp.vocab_part }) |ne| try t.router(A, D, A, A, A, R, D, ne);
    if (sp.mtp) try t.router(A, 2 * D, A, A, A, R, 2 * D, D);
    try t.topk(A, A, A, A, R, c.experts, c.top_k, c.routed_scaling, c.norm_topk);
    try t.iqRope(A, A, A, R, c.index_heads, c.index_dim, c.rope);
    for (sp.dcps) |dcp| {
        for (0..ranksOf(dcp)) |rank| {
            try t.kvWriteDcp(A, sp.kva_n, A, A, A, A, c.eps, R, m, dcp, rank);
            try t.ikWriteDcp(A, A, A, A, A, A, R, c.index_dim, c.rope, dcp, rank);
        }
    }
    // the attention core: absorb / expand (row-block kernels; wide windows take cuBLAS and _qrope), both tilings
    try t.absorb(A, A, A, A, A, A, R, m);
    try t.expand(A, A, A, R, m);
    try t.qrope(A, A, A, A, R, m);
    for ([_]tri.AttnTiling{ tri.attn_decode, tri.attn_prompt }) |tl| {
        for (sp.dcps) |dcp| {
            if (dcp <= 1) {
                try t.attention(A, A, A, A, A, A, A, A, A, A, R, m, tl);
                continue;
            }
            const nch = @max(1, m.topk / tl.chk);
            for (0..dcp) |rank| {
                try t.attnDcp(A, A, A, A, A, A, A, A, R, m, dcp, rank, tl);
                try t.mergeLse(A, A, A, A, A, R, dcp, m, nch);
                for ([_]bool{ true, false }) |small| {
                    const ex = dcpm.exchange(small, dcp, rank, R, m.heads, m.lw);
                    try t.dcpCombine(A + ex.o0 * 2, A + ex.l0 * 4, A, R, ex.ss, ex.ssl, m, dcp);
                }
            }
        }
    }
    if (sp.mtp) try t.cat2(A, A, A, R, D);
    if (sp.draft) try t.groupSums(A, D, A, R, D);
    for (sp.widths) |w| try t.swiglu2(A, A, A, R, w);
    // the prompt GEMM: every linear above max_window_rows rows (fused.lin), into bf16 or fp32 outputs
    if (R > max_window_rows) {
        for (sp.gemms) |kn| {
            const tl = exl3.prefillTiles(kn[0], kn[1]);
            for ([_][]const u8{ "*bf16", "*fp32" }) |ty| {
                try t.prefillGemm(A, A, A, A, A, ty, R, kn[1], kn[0], kn[1], tl.bm, tl.bk, tl.group, tl.warps, tl.stages, exl3.had_scale);
            }
        }
    }
}

/// Phase 4b: a concurrent decode window of R rows (fused.Rows): the position-dependent launches with ROWS (the other
/// launches are a one-stream window's, enumerated by `window`), the decode attention tiling (every concurrent window
/// is a decode window), the selections of its R rows (row offset 0) over the decode key buckets.
fn rowsWindow(t: tri.Tri, sp: Space, R: usize) !void {
    const c = sp.cfg;
    const m = mla(sp);
    try t.kvWriteRows(A, sp.kva_n, A, A, A, A, A, c.eps, R, m);
    try t.ikWriteRows(A, A, A, A, A, A, A, R, c.index_dim, c.rope);
    try t.iqRopeRows(A, A, A, R, c.index_heads, c.index_dim, c.rope);
    try t.absorbRows(A, A, A, A, A, A, R, m);
    try t.attentionRows(A, A, A, A, A, A, A, A, A, A, A, R, m, tri.attn_decode);
    // decode key buckets: powers of two from 2 index_topk up to the capacity's (one class: multiples of 16)
    const K = c.index_topk;
    const top = std.math.ceilPowerOfTwo(usize, @max(sp.capacity, 2 * K)) catch return error.Overflow;
    var T: usize = 2 * K;
    while (T <= top) : (T *= 2) try t.indexScoresRows(A, A, A, A, A, A, T, 0, R, c.index_heads, c.index_dim);
}

/// Index ranges reaching every class of T and of ceil(T / dcp) for dcp in sp.dcps: decode buckets (powers of two
/// from 2 index_topk) and arbitrary prompt-chunk ends (base + 0 .. 32 above each power of two and above dcp * K).
pub fn indexRanges(gpa: std.mem.Allocator, sp: Space) ![]usize {
    const K = sp.cfg.index_topk;
    const top = std.math.ceilPowerOfTwo(usize, @max(sp.capacity, 2 * K)) catch return error.Overflow;
    var out: std.ArrayList(usize) = .empty;
    errdefer out.deinit(gpa);
    var bases: std.ArrayList(usize) = .empty;
    defer bases.deinit(gpa);
    var b: usize = K;
    while (b <= top) : (b *= 2) try bases.append(gpa, b);
    for (sp.dcps) |dcp| try bases.append(gpa, dcp * K);
    for (bases.items) |base| {
        for (0..33) |d| {
            const T = base + d;
            if (T <= K or T > top) continue;
            if (std.mem.indexOfScalar(usize, out.items, T) == null) try out.append(gpa, T);
        }
    }
    return out.toOwnedSlice(gpa);
}

/// fused.select: _index_scores and the radix selects of every block of n <= SEL_ROWS rows, at row offsets R0 of
/// every class (0, 1, multiples of 16, other: sequence-parallel halves start at a rank's own row).
fn selects(t: tri.Tri, sp: Space) !void {
    const c = sp.cfg;
    const K = c.index_topk;
    const gpa = t.probe.?.gpa;
    const Ts = try indexRanges(gpa, sp);
    defer gpa.free(Ts);
    for (Ts) |T| {
        for ([_]usize{ 0, 1, 16, 17 }) |r0| {
            for (1..sel_rows + 1) |n| {
                for (sp.dcps) |dcp| {
                    if (dcp <= 1) {
                        try t.indexScores(A, A, A, A, A, T, r0, n, c.index_heads, c.index_dim);
                        try t.selectKeys(A, A, T, n, K);
                        continue;
                    }
                    const Tl = dcpm.localCols(T, dcp);
                    const kk = @min(K, Tl);
                    for (0..dcp) |rank| {
                        try t.indexScoresDcp(A, A, A, A, A, Tl, r0, n, c.index_heads, c.index_dim, dcp, rank);
                        // Forward.selectDcp copies the rows when kk == Tl (no launch)
                        if (kk < Tl) try t.topKeys(A, A, Tl, n, kk);
                    }
                }
            }
        }
    }
}

// ---- building a Space on the host: config.json + the reference's tile tables ----

/// The (k, n) of every row of a tiles.py table (TF_GLM53_TILES=save:...): [k, n, bits, codebook, layout, sk, wk].
pub fn tileShapes(gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList([2]usize)) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const rows = (parsed.value.object.get("linears") orelse return error.BadTileTable).array.items;
    for (rows) |rv| {
        const r = rv.array.items;
        if (r.len < 2) return error.BadTileTable;
        try out.append(gpa, .{ @intCast(r[0].integer), @intCast(r[1].integer) });
    }
}

/// SwiGLU widths in a tile table's row order: a gate, up, down triple is (D, w), (D, w), (w, D).
pub fn swigluWidths(shapes: []const [2]usize, D: usize, out: *std.ArrayList(usize), gpa: std.mem.Allocator) !void {
    if (shapes.len < 3) return;
    for (0..shapes.len - 2) |i| {
        const g = shapes[i];
        const u = shapes[i + 1];
        const d = shapes[i + 2];
        if (g[0] == D and u[0] == D and g[1] == u[1] and d[0] == g[1] and d[1] == D) {
            if (std.mem.indexOfScalar(usize, out.items, g[1]) == null) try out.append(gpa, g[1]);
        }
    }
}

pub fn addUnique(gpa: std.mem.Allocator, list: *std.ArrayList([2]usize), x: [2]usize) !void {
    for (list.items) |y| if (y[0] == x[0] and y[1] == x[1]) return;
    try list.append(gpa, x);
}

test "the engine constants the enumeration mirrors" {
    const fw = @import("cuda_forward.zig");
    try std.testing.expectEqual(fw.max_rows, max_window_rows);
}

fn testConfig() Config {
    // GLM-5.3's dimensions (config.json of the 2.75-bit checkpoint)
    return .{ .hidden = 6144, .layers = 78, .first_dense = 3, .dense_width = 12288, .expert_width = 2048, .experts = 256, .shared_experts = 1, .top_k = 8, .routed_scaling = 2.5, .norm_topk = true, .heads = 64, .q_lora = 2048, .kv_lora = 512, .nope = 192, .rope = 64, .v_dim = 256, .rope_theta = 1e6, .index_heads = 32, .index_dim = 128, .index_topk = 2048, .eps = 1e-5, .vocab = 154880, .mtp_layers = 1 };
}

test "every Triton function cuda_triton.zig launches is reached, a 41-row window's router included" {
    const gpa = std.testing.allocator;
    const c = testConfig();
    var p = aot.Probe.init(gpa);
    defer p.deinit();
    const gemms = [_][2]usize{ .{ 6144, 2048 }, .{ 6144, 640 }, .{ 2048, 4096 }, .{ 4096, 6144 }, .{ 6144, 3072 }, .{ 3072, 6144 }, .{ 6144, 512 }, .{ 512, 6144 } };
    const drafters = [_]draft.Spec{
        .{ .kind = .dspark, .D = 6144, .hd = 64, .H = 16, .KV = 16, .I = 3072, .layers = 3, .ntaps = 5, .block = 8, .top_k = 64, .V = c.vocab / 4, .vocab_off = 0, .eps = 1e-5, .window = 2048, .causal = true, .world = 4, .theta = 8e6, .mask_id = 154856 },
        .{ .kind = .dflash, .D = 6144, .hd = 128, .H = 16, .KV = 2, .I = 3072, .layers = 6, .ntaps = 6, .block = 8, .top_k = 16, .V = c.vocab / 4, .vocab_off = 0, .rsel = 256, .gs = 16, .eps = 1e-5, .window = 2047, .causal = false, .world = 4, .theta = 1e6, .mask_id = 154856 },
    };
    const sp: Space = .{ .cfg = c, .world = 4, .heads = c.heads / 4, .vocab_part = c.vocab / 4, .kva_n = 640, .max_rows = 160, .capacity = 4096, .dcps = &.{ 1, 4 }, .widths = &.{ 3072, 512 }, .gemms = &gemms, .drafters = &drafters };
    try enumerate(&p, sp);
    // the concurrent drafters' launches (_dconv_seg / _dattn_seg) exist only with streams > 1
    var sp4 = sp;
    sp4.dcps = &.{1};
    sp4.rows_windows = 32;
    sp4.streams = 4;
    try enumerate(&p, sp4);
    // the function names in cuda_triton.zig's launches: t.run("_name", ...)
    const src = @embedFile("cuda_triton.zig");
    var seen: usize = 0;
    var it = std.mem.splitSequence(u8, src, "t.run(\"");
    _ = it.next();
    while (it.next()) |rest| {
        const end = std.mem.indexOfScalar(u8, rest, '"') orelse continue;
        const name = rest[0..end];
        var hit = false;
        for (p.needs.values()) |n| hit = hit or std.mem.eql(u8, n.function, name);
        if (!hit) std.debug.print("cuda_coverage: {s} is launched by cuda_triton.zig but never enumerated\n", .{name});
        try std.testing.expect(hit);
        seen += 1;
    }
    try std.testing.expect(seen >= 22);
    // run 2's miss: _router_part for M 41 (BM 64, M not a multiple of 16), warps 4, stages 3
    var found = false;
    for (p.needs.values()) |n| {
        if (!std.mem.eql(u8, n.function, "_router_part")) continue;
        var bm: ?i64 = null;
        for (n.consts) |k| if (std.mem.eql(u8, k.name, "BM")) {
            bm = k.int;
        };
        var m_ok = false;
        for (n.args) |a| if (std.mem.eql(u8, a.name, "M")) {
            m_ok = @mod(a.value.i32, 16) != 0 and a.value.i32 != 1;
        };
        found = found or ((bm orelse 0) == 64 and m_ok and (n.opts.num_warps orelse 0) == 4 and (n.opts.num_stages orelse 0) == 3);
    }
    try std.testing.expect(found);
}

test "index ranges reach every class of T and of DCP's local columns" {
    const gpa = std.testing.allocator;
    const c = testConfig();
    const sp: Space = .{ .cfg = c, .world = 4, .heads = 16, .vocab_part = c.vocab / 4, .kva_n = 640, .max_rows = 1, .capacity = 1000000, .dcps = &.{ 1, 4 }, .widths = &.{}, .gemms = &.{} };
    const Ts = try indexRanges(gpa, sp);
    defer gpa.free(Ts);
    var t16 = false;
    var tn = false;
    var l16_big = false;
    var ln_big = false;
    for (Ts) |T| {
        try std.testing.expect(T > c.index_topk);
        if (T % 16 == 0) t16 = true else tn = true;
        const Tl = dcpm.localCols(T, 4);
        if (Tl > c.index_topk) {
            if (Tl % 16 == 0) l16_big = true else ln_big = true;
        }
    }
    try std.testing.expect(t16 and tn and l16_big and ln_big);
    // the decode buckets up to 1M context are all there
    var b: usize = 2 * c.index_topk;
    while (b <= 1 << 20) : (b *= 2) try std.testing.expect(std.mem.indexOfScalar(usize, Ts, b) != null);
}

test "call-site row offsets stay 16-byte aligned for GLM-5.3 (pointers are taken as aligned)" {
    const c = testConfig();
    // b.normed / b.x + r * D * 2, b.mlog + r * experts * 4, b.pick / b.wts + r * top_k * 4, b.iq + r * ih * id * 2,
    // b.iw + r * ih * 4, b.ik + r * id * 4, b.qa / b.qn + r * q_lora * 2, b.tok + r * index_topk * 4, b.kva + r * kva_n * 2
    for ([_]usize{ c.hidden * 2, c.experts * 4, c.top_k * 4, c.index_heads * c.index_dim * 2, c.index_heads * 4, c.index_dim * 4, c.q_lora * 2, c.index_topk * 4, 640 * 2 }) |row| {
        try std.testing.expectEqual(@as(usize, 0), row % 16);
    }
}

test "Phase 4b: concurrent windows enumerate the ROWS launches with a BASE table" {
    const gpa = std.testing.allocator;
    const c = testConfig();
    var p = aot.Probe.init(gpa);
    defer p.deinit();
    const sp: Space = .{ .cfg = c, .world = 4, .heads = c.heads / 4, .vocab_part = c.vocab / 4, .kva_n = 640, .max_rows = 4, .capacity = 32771, .dcps = &.{1}, .widths = &.{}, .gemms = &.{}, .rows_windows = 12 };
    try enumerate(&p, sp);
    const fns = [_][]const u8{ "_kv_write", "_ik_write", "_iq_rope", "_absorb", "_attn_chunks", "_index_scores" };
    for (fns) |name| {
        var rows_true = false;
        for (p.needs.values()) |n| {
            if (!std.mem.eql(u8, n.function, name)) continue;
            for (n.consts) |k| if (std.mem.eql(u8, k.name, "ROWS") and (k.int orelse 0) == 1) {
                rows_true = true;
            };
        }
        if (!rows_true) std.debug.print("cuda_coverage: no ROWS launch of {s}\n", .{name});
        try std.testing.expect(rows_true);
    }
    // the BASE table is a runtime pointer of the ROWS variants
    var base_seen = false;
    for (p.needs.values()) |n| for (n.args) |a| {
        if (std.mem.eql(u8, a.name, "BASE")) base_seen = true;
    };
    try std.testing.expect(base_seen);
}

test "Phase 4c: --parallel 4 with both drafters reaches the concurrent drafter kernels and 32-row windows" {
    const gpa = std.testing.allocator;
    const c = testConfig();
    var p = aot.Probe.init(gpa);
    defer p.deinit();
    const drafters = [_]draft.Spec{
        .{ .kind = .dspark, .D = 6144, .hd = 64, .H = 16, .KV = 16, .I = 3072, .layers = 3, .ntaps = 5, .block = 8, .top_k = 64, .V = c.vocab / 4, .vocab_off = 0, .eps = 1e-5, .window = 2048, .causal = true, .world = 4, .theta = 8e6, .mask_id = 154856 },
        .{ .kind = .dflash, .D = 6144, .hd = 128, .H = 16, .KV = 2, .I = 3072, .layers = 6, .ntaps = 6, .block = 8, .top_k = 16, .V = c.vocab / 4, .vocab_off = 0, .rsel = 256, .gs = 16, .eps = 1e-5, .window = 2047, .causal = false, .world = 4, .theta = 1e6, .mask_id = 154856 },
    };
    const sp: Space = .{ .cfg = c, .world = 4, .heads = c.heads / 4, .vocab_part = c.vocab / 4, .kva_n = 640, .max_rows = 4, .capacity = 32779, .dcps = &.{1}, .widths = &.{}, .gemms = &.{}, .drafters = &drafters, .rows_windows = 32, .streams = 4 };
    try enumerate(&p, sp);
    for ([_][]const u8{ "_dconv_seg", "_dattn_seg" }) |name| {
        var hit = false;
        for (p.needs.values()) |n| hit = hit or std.mem.eql(u8, n.function, name);
        try std.testing.expect(hit);
    }
    // the concurrent windows' ROWS absorb
    var rows_seen = false;
    for (p.needs.values()) |n| {
        if (!std.mem.eql(u8, n.function, "_absorb")) continue;
        for (n.consts) |k| if (std.mem.eql(u8, k.name, "ROWS") and (k.int orelse 0) == 1) {
            rows_seen = true;
        };
    }
    try std.testing.expect(rows_seen);
}
