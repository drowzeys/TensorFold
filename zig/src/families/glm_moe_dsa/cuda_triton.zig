//! GLM-5.3's Triton kernels from the captured cubins: each launch has the Python call site's grid, constexprs and
//! launch options (fused.py, glm5_next glue.py / latent.py, topk.py). Only constexprs the call site passes are
//! matched (defaults it leaves out are not: aot.find ignores constexprs a launch does not name).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const p = aot.ptr;

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

pub fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, n) catch unreachable;
}

/// A Python float constexpr as Triton stores it: the fp32 nearest the fp64 value.
pub fn f32of(x: f64) f32 {
    return @floatCast(x);
}

/// fused.ATTN_DECODE / ATTN_PROMPT: (keys a chunk program, keys a tile, warps, stages) by window class.
pub const AttnTiling = struct { chk: usize, kt: usize, nw: u32, ns: u32 };
pub const attn_decode: AttnTiling = .{ .chk = 256, .kt = 32, .nw = 4, .ns = 2 };
pub const attn_prompt: AttnTiling = .{ .chk = 2048, .kt = 64, .nw = 8, .ns = 2 };

/// fused.RB: rows a program in the absorb / expand kernels for wide windows.
pub const rb: usize = 16;
/// fused.BT: keys a scoring program.
pub const bt: usize = 128;
/// glue.ROUTER_KS: the router's K slices.
pub const router_ks: usize = 8;

/// The MLA shapes every attention launch compiles in (this rank's heads).
pub const Mla = struct { heads: usize, nope: usize, rope: usize, lw: usize, vd: usize, topk: usize };

pub const Tri = struct {
    set: ?*const aot.Set,
    s: cuda.Stream,
    /// Coverage (cuda_coverage.zig, no GPU): record each launch's variant key here instead of launching.
    probe: ?*aot.Probe = null,

    fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const, opts: aot.Launch) !void {
        if (t.probe) |pr| return pr.add(name, args, consts, opts);
        const set = t.set orelse return error.NoTritonSet;
        try set.runWith(t.s, name, .{ u(grid[0]), u(grid[1]), u(grid[2]) }, args, consts, opts);
    }

    /// glue.rmsnorm (no group sums): out = bf16(w * bf16(x * rinv)).
    pub fn rmsnorm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, rows: usize, d: usize, eps: f32) !void {
        const block = pow2(d);
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ p("X", "*bf16", x), int("x_stride", x_stride), p("W", "*bf16", w), p("OUT", "*bf16", out), int("o_stride", o_stride), p("XS", "*bf16", out), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", block), cb("SUMS", false) }, .{ .num_warps = if (block <= 2048) 4 else 8 });
    }

    /// glue.router: OUT[m, e] = fp32 x[m] . w[e] over ROUTER_KS K slices, the slices added in order. `part`: fp32
    /// scratch of ROUTER_KS * m * ne.
    pub fn router(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, d: usize, ne: usize) !void {
        if (d % (router_ks * 64) != 0) return error.RouterUnsliced; // TODO: glue's single-pass _router branch
        const bm: usize = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
        try t.run("_router_part", .{ cdiv(m, bm), cdiv(ne, 32), router_ks }, &.{ p("X", "*bf16", x), p("W", "*bf16", w), p("PART", "*fp32", part), int("M", m), int("x_stride", x_stride) }, &.{ ci("D", d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", 32), ci("BK", 64), ci("KS", router_ks) }, .{ .num_warps = 4, .num_stages = 3 });
        const total = m * ne;
        try t.run("_router_sum", .{ cdiv(total, 1024), 1, 1 }, &.{ p("PART", "*fp32", part), p("OUT", "*fp32", out), int("total", total) }, &.{ ci("KS", router_ks), ci("BLOCK", 1024) }, .{ .num_warps = 4 });
    }

    /// fused.route's glue._topk: sigmoid scores + bias pick TOPK experts (lower id on ties), weights normalized and
    /// scaled; SLOTS = TOPK (no shared slot).
    pub fn topk(t: Tri, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, ne: usize, k: usize, scale: f32, norm: bool) !void {
        try t.run("_topk", .{ rows, 1, 1 }, &.{ p("L", "*fp32", logits), p("BIAS", "*fp32", bias), p("PICK", "*i32", pick), p("WTS", "*fp32", wts), aot.float("scale", scale) }, &.{ ci("NE", ne), ci("TOPK", k), ci("SLOTS", k), ci("BLOCK", pow2(ne + 1)), ci("SLOTP", pow2(k + 1)), cb("NORM", norm) }, .{ .num_warps = 4 });
    }

    /// glue.residual_add: X = bf16(X + bf16(the world partials summed rank 0 first)); g [world, rows, d] fp32.
    pub fn residualAdd(t: Tri, x: u64, g: u64, rows: usize, d: usize, world: usize) !void {
        return t.residualAddOf(x, g, "*fp32", rows, d, world);
    }

    /// glue.residual_add with the partials' dtype named: "*fp32" (gathered partials, the RoCE / ring sums in b.red) or
    /// "*bf16" (Phase 3a: a prompt half's bf16 ring all-reduce / reduce-scatter, b.hred).
    pub fn residualAddOf(t: Tri, x: u64, g: u64, g_ty: []const u8, rows: usize, d: usize, world: usize) !void {
        const block = @min(1024, d);
        try t.run("_residual_add", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", x), p("G", g_ty, g), int("RS", rows * d) }, &.{ ci("D", d), ci("WORLD", world), ci("BLOCK", block) }, .{ .num_warps = 4 });
    }

    /// fused._qrope: q_rot of every row at POS + r (the wide window's companion of the batched absorb).
    pub fn qrope(t: Tri, q: u64, inv: u64, qr: u64, pos: u64, rows: usize, m: Mla) !void {
        try t.run("_qrope", .{ rows, m.heads, 1 }, &.{ p("Q", "*bf16", q), p("INV", "*fp32", inv), p("QR", "*bf16", qr), p("POS", "*i32", pos) }, &.{ ci("H", m.heads), ci("QD", m.nope + m.rope), ci("NOPE", m.nope), ci("RD", m.rope), cb("ROWS", false) }, .{ .num_warps = 1 });
    }

    /// x3prefill._gemm: OUT[m, block] = ((xh[m] @ W_q[:, block]) @ H) * SCALE * svh (no bias), a 128-column block a
    /// program; tiles from x3prefill.tiles (the shape's alone). `out_ty` "*bf16" or "*fp32" (the out tensor's dtype).
    pub fn prefillGemm(t: Tri, xh: u64, wq: u64, h: u64, svh: u64, out: u64, out_ty: []const u8, M: usize, o_stride: usize, K: usize, N: usize, bm: usize, bk: usize, group: usize, warps: u32, stages: u32, scale: f32) !void {
        try t.run("_gemm", .{ cdiv(M, bm) * (N / 128), 1, 1 }, &.{ p("X", "*fp16", xh), p("W", "*fp16", wq), p("H", "*bf16", h), p("SVH", "*fp16", svh), p("BIAS", "*fp16", svh), p("OUT", out_ty, out), int("M", M), int("o_stride", o_stride) }, &.{ ci("K", K), ci("N", N), ci("BM", bm), ci("BK", bk), ci("GROUP", group), cb("HAS_BIAS", false), aot.cf("SCALE", scale) }, .{ .num_warps = warps, .num_stages = stages });
    }

    /// fused._kv_write (bf16 cache, DCP 1, one stream): row r's latent RMSNorm and rotated key into cache row POS + r.
    pub fn kvWrite(t: Tri, kva: u64, kva_stride: usize, nw: u64, cache: u64, pos: u64, inv: u64, eps: f32, rows: usize, m: Mla) !void {
        return t.kvWriteDcp(kva, kva_stride, nw, cache, pos, inv, eps, rows, m, 1, 0);
    }

    /// fused.kv_write's _kv_write with DCP / RANK (Phase 3b): only the owner of position p (p % DCP == RANK) stores
    /// its row, at local row p // DCP (DCP 1, RANK 0: every row at p, as kvWrite).
    pub fn kvWriteDcp(t: Tri, kva: u64, kva_stride: usize, nw: u64, cache: u64, pos: u64, inv: u64, eps: f32, rows: usize, m: Mla, dcp: usize, rank: usize) !void {
        try t.run("_kv_write", .{ rows, 1, 1 }, &.{ p("KVA", "*bf16", kva), int("kva_stride", kva_stride), p("NW", "*bf16", nw), p("LC", "*bf16", cache), p("POS", "*i32", pos), p("INV", "*fp32", inv), aot.float("eps", eps) }, &.{ ci("LW", m.lw), ci("RD", m.rope), ci("DCP", dcp), ci("RANK", rank), cb("ROWS", false) }, .{ .num_warps = 4 });
    }

    /// fused._ik_write: the index key's LayerNorm (fp32, bias) and RoPE on its first RD dims into IC[POS + r].
    pub fn ikWrite(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, rows: usize, d: usize, rd: usize) !void {
        return t.ikWriteDcp(ik, nw, nb, inv, ic, pos, rows, d, rd, 1, 0);
    }

    /// _ik_write with DCP / RANK (Phase 3b): the owner's local row p // DCP only.
    pub fn ikWriteDcp(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, rows: usize, d: usize, rd: usize, dcp: usize, rank: usize) !void {
        try t.run("_ik_write", .{ rows, 1, 1 }, &.{ p("IK", "*fp32", ik), p("NW", "*bf16", nw), p("NB", "*bf16", nb), p("INV", "*fp32", inv), p("IC", "*bf16", ic), p("POS", "*i32", pos), aot.float("eps", 1e-6) }, &.{ ci("D", d), ci("RD", rd), ci("DCP", dcp), ci("RANK", rank), cb("ROWS", false) }, .{ .num_warps = 4 });
    }

    /// fused._iq_rope: the index query's first RD dims rotated in place.
    pub fn iqRope(t: Tri, q: u64, inv: u64, pos: u64, rows: usize, nh: usize, d: usize, rd: usize) !void {
        try t.run("_iq_rope", .{ rows, nh, 1 }, &.{ p("Q", "*bf16", q), p("INV", "*fp32", inv), p("POS", "*i32", pos) }, &.{ ci("NH", nh), ci("D", d), ci("RD", rd), cb("ROWS", false) }, .{ .num_warps = 1 });
    }

    /// fused._index_scores with PACK False (the radix select's 4-byte order words; later keys 0): OUT [n, T] int32.
    pub fn indexScores(t: Tri, q: u64, w: u64, ic: u64, pos: u64, out: u64, T: usize, r0: usize, n: usize, nh: usize, d: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        try t.run("_index_scores", .{ n, cdiv(T, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*bf16", ic), p("POS", "*i32", pos), p("OUT", "*i32", out), int("T", T), int("R0", r0) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), ci("DCP", 1), ci("RANK", 0), cb("PACK", false), cb("ROWS", false) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// fused._index_scores, DCP > 1 (Phase 3b; PACK True: fused.select packs whenever DCP > 1): row r's score of this
    /// rank's slots t < Tl (global position t * DCP + RANK) packed with the position into one int64 that orders by
    /// score, then lower position; later positions int64 min + 1. OUT [n, Tl] int64.
    pub fn indexScoresDcp(t: Tri, q: u64, w: u64, ic: u64, pos: u64, out: u64, Tl: usize, r0: usize, n: usize, nh: usize, d: usize, dcp: usize, rank: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        try t.run("_index_scores", .{ n, cdiv(Tl, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*bf16", ic), p("POS", "*i32", pos), p("OUT", "*i64", out), int("T", Tl), int("R0", r0) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), ci("DCP", dcp), ci("RANK", rank), cb("PACK", true), cb("ROWS", false) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// topk.top_keys: row r's k best packed keys of S [rows, nc] int64 (high word order, ties to the lower column),
    /// int64 in column order -> OUT [rows, k] (the radix branch of fused.select under DCP).
    pub fn topKeys(t: Tri, sc: u64, out: u64, nc: usize, rows: usize, k: usize) !void {
        try t.run("_select_keys", .{ rows, 1, 1 }, &.{ p("S", "*i64", sc), p("OUT", "*i64", out), int("NC", nc) }, &.{ ci("K", k), ci("BLOCK", 1024), cb("KEYS", true) }, .{ .num_warps = 4 });
    }

    /// fused._attn_dcp (bf16 cache; Phase 3b): program (row, chunk, head group g) - rank g's gathered queries (QALL
    /// [G, R, H, LW + RD]) over this rank's keys of row r (its slots of positions 0..p while p < K, else its share
    /// of the selection, TOK / CNT); partials at head g * H + h of [nch, R, G * H].
    pub fn attnDcp(t: Tri, qall: u64, cache: u64, tok: u64, cnt: u64, pos: u64, po: u64, pm: u64, pl: u64, rows: usize, m: Mla, g: usize, rank: usize, tl: AttnTiling) !void {
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        try t.run("_attn_dcp", .{ rows, nch, g }, &.{ p("QALL", "*bf16", qall), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("CNT", "*i32", cnt), p("POS", "*i32", pos), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &.{ ci("H", m.heads), ci("G", g), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", tl.kt), aot.cf("SCALE", scale), ci("DCP", g), ci("RANK", rank) }, .{ .num_warps = tl.nw, .num_stages = tl.ns });
    }

    /// fused._merge_lse: program (row, head of all HT = G * H): this rank's chunk partials in chunk order -> the
    /// normalized output (bf16) and its log-sum-exp at [destination rank = head // H, row, head % H].
    pub fn mergeLse(t: Tri, po: u64, pm: u64, pl: u64, osend: u64, lsend: u64, rows: usize, g: usize, m: Mla, nch: usize) !void {
        try t.run("_merge_lse", .{ rows, g * m.heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OSEND", "*bf16", osend), p("LSEND", "*fp32", lsend), int("R", rows) }, &.{ ci("HT", g * m.heads), ci("H", m.heads), ci("LW", m.lw), ci("NCH", nch) }, .{ .num_warps = 4 });
    }

    /// fused._dcp_combine: program (row, own head) - every rank's normalized partial merged in rank order by their
    /// log-sum-exps (ORECV[src] at stride ss elements, LRECV[src] at stride ssl) -> OUT [R, H, LW] bf16.
    pub fn dcpCombine(t: Tri, orecv: u64, lrecv: u64, out: u64, rows: usize, ss: usize, ssl: usize, m: Mla, world: usize) !void {
        try t.run("_dcp_combine", .{ rows, m.heads, 1 }, &.{ p("ORECV", "*bf16", orecv), p("LRECV", "*fp32", lrecv), p("OUT", "*bf16", out), int("R", rows), int("SS", ss), int("SSL", ssl) }, &.{ ci("H", m.heads), ci("LW", m.lw), ci("WORLD", world) }, .{ .num_warps = 4 });
    }

    /// topk.top_columns over int32 order words: row r's K best columns, ascending, ties to the lower column.
    pub fn selectKeys(t: Tri, sc: u64, out: u64, nc: usize, rows: usize, k: usize) !void {
        try t.run("_select_keys", .{ rows, 1, 1 }, &.{ p("S", "*i32", sc), p("OUT", "*i32", out), int("NC", nc) }, &.{ ci("K", k), ci("BLOCK", 1024), cb("KEYS", false), cb("U32", true) }, .{ .num_warps = 4 });
    }

    /// fused._absorb: QA[r, h] = q_nope[r, h] . WK[h] and the rotated q_rot (row-block grid).
    pub fn absorb(t: Tri, q: u64, wk: u64, inv: u64, qa: u64, qr: u64, pos: u64, rows: usize, m: Mla) !void {
        const rbk = if (rows > rb) rb else rows;
        try t.run("_absorb", .{ m.heads, m.lw / 32, cdiv(rows, rbk) }, &.{ p("Q", "*bf16", q), p("WK", "*bf16", wk), p("INV", "*fp32", inv), p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("POS", "*i32", pos), int("R", rows) }, &.{ ci("H", m.heads), ci("QD", m.nope + m.rope), ci("NOPE", m.nope), ci("NOPE_P", pow2(m.nope)), ci("RD", m.rope), ci("LW", m.lw), ci("BN", 32), ci("RBK", rbk), cb("ROWS", false) }, .{ .num_warps = 4 });
    }

    /// fused._attention_local (bf16 cache, one stream): one pass straight into `ol` when one chunk covers index_topk
    /// keys, else chunk partials and latent._merge.
    pub fn attention(t: Tri, qa: u64, qr: u64, cache: u64, tok: u64, pos: u64, po: u64, pm: u64, pl: u64, ol: u64, dummy: u64, rows: usize, m: Mla, tl: AttnTiling) !void {
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        const opts: aot.Launch = .{ .num_warps = tl.nw, .num_stages = tl.ns };
        const common = [_]aot.Const{ ci("H", m.heads), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", tl.kt), aot.cf("SCALE", scale), cb("ROWS", false) };
        if (nch == 1) {
            var direct: [common.len + 1]aot.Const = undefined;
            @memcpy(direct[0..common.len], &common);
            direct[common.len] = cb("DIRECT", true);
            try t.run("_attn_chunks", .{ rows, 1, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*bf16", ol), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &direct, opts);
            return;
        }
        try t.run("_attn_chunks", .{ rows, nch, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &common, opts);
        try t.run("_merge", .{ rows, m.heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OUT", "*bf16", ol), p("CNT", "*i32", dummy), int("R", rows) }, &.{ ci("H", m.heads), ci("LW", m.lw), ci("NCH", nch), cb("SPARSE", false) }, .{ .num_warps = 4 });
    }

    /// fused._expand: OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k].
    pub fn expand(t: Tri, ol: u64, wv: u64, out: u64, rows: usize, m: Mla) !void {
        const rbk = if (rows > rb) rb else rows;
        try t.run("_expand", .{ m.heads, m.vd / 16, cdiv(rows, rbk) }, &.{ p("OL", "*bf16", ol), p("WV", "*bf16", wv), p("OUT", "*bf16", out), int("R", rows) }, &.{ ci("H", m.heads), ci("DV", m.vd), ci("LW", m.lw), ci("BN", 16), ci("RBK", rbk) }, .{ .num_warps = 4 });
    }

    // ---- Phase 4b: several streams in one decode window (fused.Rows: ROWS=True, BASE an int32 table) ----------
    // Row r sits at position POS[r] (an int32 table, not the window's first position) of the stream whose cache rows
    // start at row BASE[r]; every row computes what it computes alone (DCP 1, multi.GlmMultiDecoder's windows).

    /// _kv_write ROWS: row r's latent and rotated key into cache row BASE[r] + POS[r].
    pub fn kvWriteRows(t: Tri, kva: u64, kva_stride: usize, nw: u64, cache: u64, pos: u64, base: u64, inv: u64, eps: f32, rows: usize, m: Mla) !void {
        try t.run("_kv_write", .{ rows, 1, 1 }, &.{ p("KVA", "*bf16", kva), int("kva_stride", kva_stride), p("NW", "*bf16", nw), p("LC", "*bf16", cache), p("POS", "*i32", pos), p("INV", "*fp32", inv), aot.float("eps", eps), p("BASE", "*i32", base) }, &.{ ci("LW", m.lw), ci("RD", m.rope), ci("DCP", 1), ci("RANK", 0), cb("ROWS", true) }, .{ .num_warps = 4 });
    }

    /// _ik_write ROWS: row r's index key into IC row BASE[r] + POS[r].
    pub fn ikWriteRows(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, base: u64, rows: usize, d: usize, rd: usize) !void {
        try t.run("_ik_write", .{ rows, 1, 1 }, &.{ p("IK", "*fp32", ik), p("NW", "*bf16", nw), p("NB", "*bf16", nb), p("INV", "*fp32", inv), p("IC", "*bf16", ic), p("POS", "*i32", pos), aot.float("eps", 1e-6), p("BASE", "*i32", base) }, &.{ ci("D", d), ci("RD", rd), ci("DCP", 1), ci("RANK", 0), cb("ROWS", true) }, .{ .num_warps = 4 });
    }

    /// _iq_rope ROWS: the index query of row r rotated at POS[r].
    pub fn iqRopeRows(t: Tri, q: u64, inv: u64, pos: u64, rows: usize, nh: usize, d: usize, rd: usize) !void {
        try t.run("_iq_rope", .{ rows, nh, 1 }, &.{ p("Q", "*bf16", q), p("INV", "*fp32", inv), p("POS", "*i32", pos) }, &.{ ci("NH", nh), ci("D", d), ci("RD", rd), cb("ROWS", true) }, .{ .num_warps = 1 });
    }

    /// _index_scores ROWS (PACK False): window row R0 + r scores the keys t <= POS[R0 + r] of its stream's index-key
    /// rows from BASE[R0 + r]; OUT [n, T] int32 order words.
    pub fn indexScoresRows(t: Tri, q: u64, w: u64, ic: u64, pos: u64, base: u64, out: u64, T: usize, r0: usize, n: usize, nh: usize, d: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        try t.run("_index_scores", .{ n, cdiv(T, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*bf16", ic), p("POS", "*i32", pos), p("OUT", "*i32", out), int("T", T), int("R0", r0), p("BASE", "*i32", base) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), ci("DCP", 1), ci("RANK", 0), cb("PACK", false), cb("ROWS", true) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// _absorb ROWS: q_rot of row r at its own position POS[r].
    pub fn absorbRows(t: Tri, q: u64, wk: u64, inv: u64, qa: u64, qr: u64, pos: u64, rows: usize, m: Mla) !void {
        const rbk = if (rows > rb) rb else rows;
        try t.run("_absorb", .{ m.heads, m.lw / 32, cdiv(rows, rbk) }, &.{ p("Q", "*bf16", q), p("WK", "*bf16", wk), p("INV", "*fp32", inv), p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("POS", "*i32", pos), int("R", rows) }, &.{ ci("H", m.heads), ci("QD", m.nope + m.rope), ci("NOPE", m.nope), ci("NOPE_P", pow2(m.nope)), ci("RD", m.rope), ci("LW", m.lw), ci("BN", 32), ci("RBK", rbk), cb("ROWS", true) }, .{ .num_warps = 4 });
    }

    /// fused._attention_local ROWS: row r at POS[r] over its stream's cache rows from BASE[r] (its key list: keys
    /// 0..p while p < K, else TOK[r]); one pass when a chunk covers index_topk keys, else partials + latent._merge.
    pub fn attentionRows(t: Tri, qa: u64, qr: u64, cache: u64, tok: u64, pos: u64, base: u64, po: u64, pm: u64, pl: u64, ol: u64, dummy: u64, rows: usize, m: Mla, tl: AttnTiling) !void {
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        const opts: aot.Launch = .{ .num_warps = tl.nw, .num_stages = tl.ns };
        const common = [_]aot.Const{ ci("H", m.heads), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", tl.kt), aot.cf("SCALE", scale), cb("ROWS", true) };
        if (nch == 1) {
            var direct: [common.len + 1]aot.Const = undefined;
            @memcpy(direct[0..common.len], &common);
            direct[common.len] = cb("DIRECT", true);
            try t.run("_attn_chunks", .{ rows, 1, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*bf16", ol), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows), p("BASE", "*i32", base) }, &direct, opts);
            return;
        }
        try t.run("_attn_chunks", .{ rows, nch, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows), p("BASE", "*i32", base) }, &common, opts);
        try t.run("_merge", .{ rows, m.heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OUT", "*bf16", ol), p("CNT", "*i32", dummy), int("R", rows) }, &.{ ci("H", m.heads), ci("LW", m.lw), ci("NCH", nch), cb("SPARSE", false) }, .{ .num_warps = 4 });
    }

    /// fused._cat2: OUT[r] = [A[r] ; B[r]] (bf16 rows of D each), the MTP layer's [enorm(embedding) ; hnorm(hidden)].
    pub fn cat2(t: Tri, a: u64, b: u64, out: u64, rows: usize, d: usize) !void {
        try t.run("_cat2", .{ rows, d / 1024, 1 }, &.{ p("A", "*bf16", a), p("B", "*bf16", b), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("BLOCK", 1024) }, .{ .num_warps = 4 });
    }

    /// glm5_next qmm.group_sums: XS[m, g] = fp32 sum of x[m, 64 g .. 64 g + 63] (the draft head's input sums).
    pub fn groupSums(t: Tri, x: u64, x_stride: usize, xs: u64, rows: usize, k: usize) !void {
        try t.run("_group_sums", .{ rows, cdiv(k / 64, 16), 1 }, &.{ p("X", "*bf16", x), p("XS", "*fp32", xs), int("x_stride", x_stride) }, &.{ ci("K", k), ci("GB", 16) }, .{ .num_warps = 2 });
    }

    // ---- Phase 4: the drafters (glm5_next glue / dflash2, cuda/dspark.py, cuda/dflash.py) ----------------------

    /// glue.rmsnorm with group sums (the drafters' _norm): out = bf16(w * bf16(x * rinv)) and XS[r, g] = fp32 sums of
    /// out's 64-groups (the next 4-bit matmul's input sums).
    pub fn rmsnormSums(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: u64, rows: usize, d: usize, eps: f32) !void {
        const block = pow2(d);
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ p("X", "*bf16", x), int("x_stride", x_stride), p("W", "*bf16", w), p("OUT", "*bf16", out), int("o_stride", o_stride), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", block), cb("SUMS", true) }, .{ .num_warps = if (block <= 2048) 4 else 8 });
    }

    /// glue.embed over a BF16 table (_embed_b16): rows of the verifier's embedding for int32 ids, one copy.
    pub fn embedB16(t: Tri, ids: u64, table: u64, out: u64, rows: usize, d: usize) !void {
        try t.run("_embed_b16", .{ rows, d / 64, 1 }, &.{ p("IDS", "*i32", ids), p("W", "*bf16", table), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("COPIES", 1) }, .{ .num_warps = 1 });
    }

    /// glue.swiglu: [gate | up] (bf16) -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)) and its 64-group sums.
    pub fn swigluSums(t: Tri, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, width: usize) !void {
        const blk = std.math.gcd(@as(usize, 512), width);
        if (blk < 64) return error.Invalid;
        try t.run("_swiglu", .{ rows, width / blk, 1 }, &.{ p("GU", "*bf16", gu), p("OUT", "*bf16", out), p("XS", "*fp32", xs), aot.float("LIMIT", limit) }, &.{ ci("W", width), ci("BLOCK", blk) }, .{ .num_warps = 4 });
    }

    /// dflash2._prep_kernel: q / k normed (q_norm / k_norm) and rotated, v copied, out of [q heads | k heads | v heads]
    /// rows (row stride `stride`) into [heads, rows, head_dim]; heads 0 (context rows): QO is the input itself.
    pub fn prep(t: Tri, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: usize, stride: usize, eps: f32, heads: usize, kvh: usize, half: usize) !void {
        try t.run("_prep_kernel", .{ rows, heads + 2 * kvh, 1 }, &.{ p("QKV", "*bf16", qkv), p("QN", "*bf16", qn), p("KN", "*bf16", kn), p("COS", "*fp32", cos), p("SIN", "*fp32", sin), p("QO", "*bf16", qo), p("KO", "*bf16", ko), p("VO", "*bf16", vo), int("L", rows), int("stride", stride), aot.float("eps", eps) }, &.{ ci("H", heads), ci("HKV", kvh), ci("HALF", half) }, .{ .num_warps = 1 });
    }

    /// dflash2._dconv_kernel: the two-tap grouped dynamic convolution over a block (row r mixes rows r and r - 1),
    /// plus the residual when `has_res`.
    pub fn dconv(t: Tri, x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: usize, d: usize, gs: usize, branch: usize, has_res: bool) !void {
        try t.run("_dconv_kernel", .{ rows, cdiv(d, 1024), 1 }, &.{ p("X", "*bf16", x), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", res), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("G", d / gs), ci("GS", gs), ci("BRANCH", branch), cb("HAS_RES", has_res), ci("BLOCK", 1024) }, .{ .num_warps = 4 });
    }

    /// dspark._block_attn_kernel: one program a head - the block's N queries over the context rows [s - W, s) and the
    /// block's keys (causal: up to their own slot), keys at ring slot position % RING_.
    pub fn blockAttn(t: Tri, q: u64, k: u64, v: u64, out: u64, pos: u64, w: usize, scale: f32, n: usize, heads: usize, hd: usize, ring_rows: usize, causal: bool) !void {
        try t.run("_block_attn_kernel", .{ heads, 1, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", k), p("V", "*bf16", v), p("OUT", "*bf16", out), p("POS", "*i64", pos), int("W", w), aot.float("scale", scale) }, &.{ ci("N", n), ci("MP", @max(16, pow2(n))), ci("NH", heads), ci("HD", hd), ci("RING_", ring_rows), ci("BK", 64), cb("CAUSAL", causal) }, .{ .num_warps = 4 });
    }

    /// dflash._dattn_ring: one program a KV head - its G query heads' N rows over the sliding window of committed
    /// context and the block, keys at ring slot position % RING_.
    pub fn dattnRing(t: Tri, q: u64, k: u64, v: u64, out: u64, pos: u64, window: usize, scale: f32, n: usize, g: usize, heads: usize, hd: usize, ring_rows: usize, kvh: usize, causal: bool) !void {
        try t.run("_dattn_ring", .{ kvh, 1, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", k), p("V", "*bf16", v), p("OUT", "*bf16", out), p("POS", "*i64", pos), int("window", window), aot.float("scale", scale) }, &.{ ci("N", n), ci("G", g), ci("NH", heads), ci("HD", hd), ci("RING_", ring_rows), ci("BK", 64), cb("CAUSAL", causal) }, .{ .num_warps = 4 });
    }

    /// Phase 4c: dflash._dconv_seg - _dconv_kernel over blocks of SEG rows side by side (a block's first row mixes no
    /// row before it: one block computes the solo kernel's bits), plus the residual when `has_res`.
    pub fn dconvSeg(t: Tri, x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: usize, d: usize, gs: usize, branch: usize, has_res: bool, seg: usize) !void {
        try t.run("_dconv_seg", .{ rows, cdiv(d, 1024), 1 }, &.{ p("X", "*bf16", x), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", res), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("G", d / gs), ci("GS", gs), ci("BRANCH", branch), cb("HAS_RES", has_res), ci("BLOCK", 1024), ci("SEG", seg) }, .{ .num_warps = 4 });
    }

    /// Phase 4c: dflash._dattn_seg - _dattn_ring for `segs` blocks of N rows side by side (program 1: the block): its
    /// queries at rows seg * N of q [heads, L, hd], its ring at slot META[St + seg] (rows slot * RING_ of the pooled
    /// k / v [kv heads, POOL, hd]), its committed length META[2 St + seg] (int64).
    pub fn dattnSeg(t: Tri, q: u64, k: u64, v: u64, out: u64, meta: u64, st: usize, window: usize, scale: f32, L: usize, pool: usize, n: usize, g: usize, heads: usize, hd: usize, ring_rows: usize, kvh: usize, segs: usize, causal: bool) !void {
        try t.run("_dattn_seg", .{ kvh, segs, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", k), p("V", "*bf16", v), p("OUT", "*bf16", out), p("META", "*i64", meta), int("St", st), int("window", window), aot.float("scale", scale), int("L", L), int("POOL", pool) }, &.{ ci("N", n), ci("G", g), ci("NH", heads), ci("HD", hd), ci("RING_", ring_rows), ci("BK", 64), cb("CAUSAL", causal) }, .{ .num_warps = 4 });
    }

    /// glue.residual_add with the gathered slots `rs` words apart (fused.small_gather's padded row stride).
    pub fn residualAddStride(t: Tri, x: u64, g: u64, rows: usize, d: usize, world: usize, rs: usize) !void {
        const block = @min(1024, d);
        try t.run("_residual_add", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", x), p("G", "*fp32", g), int("RS", rs) }, &.{ ci("D", d), ci("WORLD", world), ci("BLOCK", block) }, .{ .num_warps = 4 });
    }

    /// fused._swiglu2: bf16(bf16(silu(g)) * u) for separate gate and up rows of `width`.
    pub fn swiglu2(t: Tri, g: u64, up: u64, out: u64, rows: usize, width: usize) !void {
        const blk = std.math.gcd(@as(usize, 512), width);
        try t.run("_swiglu2", .{ rows, width / blk, 1 }, &.{ p("G", "*bf16", g), p("U", "*bf16", up), p("OUT", "*bf16", out) }, &.{ ci("W", width), ci("BLOCK", blk) }, .{ .num_warps = 4 });
    }
};

/// fused.bucket: the indexer's key range for a window whose last row sits at t - 1 (INDEX_SPLIT 1); null while
/// t <= topk (no selection).
pub fn bucket(t: usize, topk: usize) ?usize {
    if (t <= topk) return null;
    const pw = std.math.ceilPowerOfTwo(usize, t) catch unreachable; // 1 << (t - 1).bit_length()
    return @max(2 * topk, pw);
}

test "index buckets as fused.bucket (INDEX_SPLIT 1)" {
    try std.testing.expectEqual(@as(?usize, null), bucket(2048, 2048));
    try std.testing.expectEqual(@as(?usize, 4096), bucket(2049, 2048));
    try std.testing.expectEqual(@as(?usize, 32768), bucket(20001, 2048));
    try std.testing.expectEqual(@as(?usize, 32768), bucket(32768, 2048));
    try std.testing.expectEqual(@as(?usize, 65536), bucket(32769, 2048));
}

test "constexpr floats: the fp32 nearest Python's float" {
    try std.testing.expectEqual(@as(u32, 0x3d800000), @as(u32, @bitCast(f32of(std.math.pow(f64, 256.0, -0.5)))));
    try std.testing.expectEqual(@as(u32, 0x3e3504f3), @as(u32, @bitCast(f32of(std.math.pow(f64, 32.0, -0.5)))));
}
