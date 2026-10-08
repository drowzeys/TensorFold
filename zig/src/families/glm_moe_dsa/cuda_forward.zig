//! GLM-5.3's fused forward for windows of 1 to 128 rows (fused.compute / layer / attention_part / attention_core /
//! ffn_part / head), one stream, one rank of the TP world.
//!
//! Exchanges (fused.gather's all-gather branch): every rank's fp32 partial [world, R, D] in rank order, then
//! glue.residual_add sums the world slots in rank order (WORLD = world), so every rank gets the same bits.
//! Phase 2a: `Forward.comm` set - NCCL all-gather of the real partials on the compute stream (the Python engine's
//! all-gather path for every window: tools/glm53/reference.py gives its engine a communicator without all_reduce /
//! all_to_all, so prompt windows too take this branch). `comm` null - the Phase 1 identity communicator (oracle.py's
//! LocalComm): b.gath zeroed, this rank's partial in its slot. Every kernel computes each row alone, so a row's bits
//! never depend on the window: the outputs match the Python engine's bit for bit when the kernels and launches do.
//! The head (fused.head mode "argmax"): this rank's logits over its vocabulary share (glue.router), its first
//! maximum (host: torch.argmax's rule), the [value, id, 0, 0] records all-gathered, the lowest rank of the maxima.
//!
//! Frozen as tools/glm53/oracle.py freezes the Python side: bf16 latent cache, DCP 1, one stream (TF_GLM53_SIDE=0:
//! the side stream gives the same bits), x3linear.plan tiles, TensorFold's expert layout, the shared expert folded
//! into the routed combine. Selection always goes through the radix select (topk.top_columns over _index_scores'
//! 4-byte order words): the same keys as torch.topk + sort, which the champion uses for decode-sized windows (the
//! oracle checks the two agree).
//! Phase 2b: the device head (`headDevice`: fused.head "argmax" / "local" / "draft" with no host step, so a window
//! replays as a CUDA graph), ids on the device (b.ids, torch.index_select's rows), the MTP layer (`mtpCompute`,
//! fused.mtp_compute with TF_GLM53_MTP_REUSE's keep / reuse) and the decode-class buffers (`BufOpts.decode`).
//! Phase 3a: the served configuration's exchanges - decode windows over the RoCE one-shot (`Forward.fast`: the
//! rank-order sum into b.red, the head's records and sampled candidates gathered over it), wider windows over NCCL's
//! bf16 ring all-reduce (`Forward.ring`, TF_GLM53_PREFILL_REDUCE=ring) - and windows over 128 rows (`Forward.wide`:
//! the prompt GEMM, torch.bmm absorb / expand, prompt experts); the sequence-parallel prompt chunk itself is
//! cuda_prompt.zig's.
//! Phase 3b: decode context parallelism (`Forward.dcp`, cuda_dcp.zig; fused._attention_dcp and select's DCP branch):
//! the caches sharded round robin over the ranks (State.initDcp), each rank's keys written by their owner only
//! (_kv_write / _ik_write DCP, RANK), every rank's queries gathered and all heads attended over this rank's keys, the
//! partials merged by log-sum-exp at the heads' owners; the indexer's per-rank top-k, gathered candidates and one
//! global top-k. DCP 1 keeps every Phase 3a path unchanged. TODO: int8/int4 latent caches, the exact reduce-scatter
//! (TF_GLM53_PREFILL_REDUCE=rs).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const exl3 = @import("exl3.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const W = @import("cuda_weights.zig");
const smp = @import("sampling.zig");
const roce = @import("cuda_roce.zig");
const prompt = @import("cuda_prompt.zig");
const dec = @import("cuda_decode.zig");
const dcpm = @import("cuda_dcp.zig");

/// The widest TP world the head's exchange records hold.
pub const max_world: usize = 16;

/// One head pick: the token, this rank's local argmax and the gathered [world, 4] fp32 records (bit-compared with
/// the Python engine's b.amax_all).
pub const Pick = struct {
    token: u32 = 0,
    local: usize = 0,
    world: usize,
    gathered: [max_world * 4]f32 = @splat(0),
};

/// fused.FAST_ROWS: windows up to this many rows use the decode attention tiling (non-decode Buffers).
pub const fast_rows: usize = 16;
/// fused.MAX_ROWS: the widest window of the row-invariant EXL3 linear; wider prompt chunks take the prompt GEMM.
pub const max_rows: usize = 128;
/// fused.PROMPT_ROWS' default: the widest prompt chunk the served path's buffers take.
pub const max_prompt_rows: usize = 8192;
/// fused.DECODE_ROWS: the widest decode-class buffer set.
pub const decode_rows: usize = 32;
/// fused.RADIX_MIN_ROWS: selections of fewer rows (decode windows) take the multi-block top-k (glm53_decode.cu) - the
/// Python engine's torch.topk + sort branch there; the same columns as the one-program radix select.
pub const radix_min_rows: usize = 32;

pub const BufOpts = struct {
    /// fused.Buffers(decode=True): verify and MTP windows
    decode: bool = false,
    /// the MTP layer's scratch (hin, me, mh, mcat, mx32, eh_proj partials)
    mtp: bool = false,
    /// Phase 3a: prompt chunks of more than max_rows rows (the served path's 8192 / 4096-row buffers): the EXL3
    /// linear scratch stays at max_rows rows (wider calls take the prompt GEMM's workspace), plus the sequence-
    /// parallel and ring-reduce buffers (bpart, bred, xo)
    prompt: bool = false,
    /// weights.EXPERTS_IMPL == "shared" (experts_cx.SharedExperts): the decode experts' scratch holds min(rows,
    /// MAX_ROWS) rows and wider windows take the prompt experts kernel (prompt_experts.cu); false ("tf"): the decode
    /// kernel at every width, its scratch for every row
    shared_experts: bool = false,
    /// Phase 3a speed (cuda_prompt.MtpDense, --mtp-dense 1): the MTP layer's prompt rows take cuBLAS for eh_proj and
    /// the dense experts, so the eh_proj router partials / fp32 rows stay at max_rows rows and no decode-kernel
    /// blocks are carved (xs_big)
    mtp_dense: bool = false,
    /// Phase 3b: decode context parallelism over this many ranks (fused.Buffers with w.dcp > 1): partial slots for
    /// every rank's heads, the query / partial exchange buffers, the selection's candidates; the score scratch holds a
    /// rank's ceil(score_cols / dcp) slots
    dcp: usize = 1,
    /// Phase 4: fused.Buffers.taps - the drafters' tap layers' output rows, [rows, taps * D] bf16 (0: no drafter)
    taps: usize = 0,
};

/// Phase 4: fused.Weights.tap_slot - which target layers' output rows the loaded drafters read and where in
/// Buffers.taps (draft_host.TapPlan's slots); n 0: no drafter.
pub const Taps = struct {
    slot: [256]i16 = @splat(-1),
    n: usize = 0,

    pub fn of(plan: anytype) !Taps {
        var t: Taps = .{};
        for (plan.layers[0..plan.n], 0..) |l, i| {
            if (l >= t.slot.len) return error.Invalid;
            t.slot[l] = @intCast(i);
        }
        t.n = plan.n;
        return t;
    }
};

/// What a window's head computes (fused.head's modes; `none`: no head).
pub const HeadMode = enum { none, argmax, local, draft };

/// fused.Buffers: scratch for windows of up to `rows` rows (sliced [:R]); `score_cols` the widest index range.
pub const Buffers = struct {
    d: *const cuda.Driver,
    rows: usize,
    score_cols: usize,
    small: usize,
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: std.mem.Allocator,
    x: u64 = 0, // bf16 [rows, D]
    normed: u64 = 0,
    qa: u64 = 0, // bf16 [rows, q_lora]
    qn: u64 = 0,
    kva: u64 = 0, // bf16 [rows, kv_a.n]
    kva_n: usize = 0,
    q: u64 = 0, // bf16 [rows, H * (nope + rope)]
    qlat: u64 = 0, // bf16 [rows, H, kv_lora]
    qrot: u64 = 0, // bf16 [rows, H, rope]
    po: u64 = 0, // fp32 chunk partials
    pm: u64 = 0,
    pl: u64 = 0,
    ol: u64 = 0, // bf16 [rows, H, kv_lora]
    o: u64 = 0, // bf16 [rows, H * v]
    dummy: u64 = 0, // int32 [1] zero
    ik: u64 = 0, // fp32 [rows, index_dim]
    iw: u64 = 0, // fp32 [rows, index_heads]
    iq: u64 = 0, // bf16 [rows, index_heads * index_dim]
    sc: u64 = 0, // int64 [min(rows, 128) * score_cols] (int32 order words here)
    tok: u64 = 0, // int32 [rows, index_topk]
    g: u64 = 0, // bf16 [rows * width]
    u: u64 = 0,
    act: u64 = 0,
    mlog: u64 = 0, // fp32 [rows, E]
    pick: u64 = 0, // int32 [rows, top_k]
    wts: u64 = 0, // fp32 [rows, top_k]
    rpart: u64 = 0, // fp32 router partials [ROUTER_KS, rows, max(E, index_dim)]
    sy: u64 = 0, // fp32 [rows, D]
    part: u64 = 0, // fp32 [rows, D]
    gath: u64 = 0, // fp32 [world * rows * D]: every rank's partial in rank order (fused.gather)
    // the head (allocated when the weights hold one): fused.Buffers fnormed / lpart / amax / amax_all
    head_rows: usize = 0, // min(rows, small)
    fnormed: u64 = 0, // bf16 [head_rows, D]
    lg: u64 = 0, // fp32 [head_rows, vocab_part]
    hpart: u64 = 0, // fp32 [ROUTER_KS, head_rows, vocab_part] (glue.router's partials)
    amax: u64 = 0, // fp32 [head_rows, 4]: [max logit, global id, 0, 0] a row
    amax_all: u64 = 0, // fp32 [world, head_rows, 4]
    // Phase 2b: device ids, the hidden rows the MTP layer reads and writes, its scratch, the device head
    ids: u64 = 0, // int64 [rows]
    hin: u64 = 0, // bf16 [rows, D]: the hidden rows the MTP layer reads
    hidden: u64 = 0, // bf16 [rows, D]: the window's final (target) or chain (MTP) hidden rows
    ht: u64 = 0, // bf16 [rows, D]: target_hidden_for_mtp of a prompt chunk
    me: u64 = 0, // bf16 [rows, D]
    mh: u64 = 0,
    mcat: u64 = 0, // bf16 [rows, 2 D]
    mx32: u64 = 0, // fp32 [rows, D]
    epart: u64 = 0, // fp32 [ROUTER_KS, rows, D]: eh_proj's router partials
    erows: usize = 0, // rows mx32 / epart hold (BufOpts.mtp_dense: min(rows, max_rows))
    argmax: u64 = 0, // int64 [head_rows]
    gsum: u64 = 0, // fp32 [head_rows, D / 64]: the draft head's input group sums
    head_v: usize = 0, // b.lg holds head_rows rows of up to this many logits (max(vocab_part, draft rows))
    cand_words: usize = 0, // words a rank sends for sampled candidates: head_rows * 2 * max_candidates + 1
    cand_send: u64 = 0, // fp32 [cand_words rounded up to 4]
    cand_all: u64 = 0, // fp32 [world * cand_words rounded up to 4]
    lin: kern.LinearScratch = undefined,
    xs: ?kern.ExpertScratch = null,
    // Phase 3a
    red: u64 = 0, // fp32 [rows, D]: the RoCE sum (decode windows) / the ring all-reduce's bf16 sum widened (prompts)
    bpart: u64 = 0, // bf16 [rows, D]: a prompt chunk's partial for the bf16 ring all-reduce / reduce-scatter
    bred: u64 = 0, // bf16 [rows, D]
    xo: u64 = 0, // bf16 [ceil(rows / world), D]: sequence-parallel own rows of the residual stream
    xs_rows: usize = 0, // rows the decode experts' scratch holds
    pe: ?kern.PromptExpertScratch = null, // prompt experts (SharedExperts.prefill), prompt buffers of the shared impl
    // Phase 3a speed
    slin: kern.LinearScratch = undefined, // TF_GLM53_SIDE: the side stream's EXL3 scratch (min(rows, small) rows)
    srpart: u64 = 0, // the side stream's router partials
    tsel: kern.TopkScratch = .{}, // the multi-block top-k (decode-sized selections, sampled candidates)
    cand_cols: u64 = 0, // int32 [head_rows, max_candidates]: a sampled window's candidate columns
    /// the MTP layer's prompt rows (8-bit experts prompt_experts does not take): the decode kernel in blocks of
    /// xs_big.rows rows, its big buffers inside b.gath (idle while the MTP layer runs: prompt chunks reduce over the
    /// bf16 ring); null: blocks of xs_rows
    xs_big: ?kern.ExpertScratch = null,
    // Phase 3b: decode context parallelism (fused.Buffers with w.dcp > 1; the exchange buffers 0 at DCP 1)
    dcp: usize = 1,
    qpack: u64 = 0, // bf16 [rows, H, lw + rd]: this rank's absorbed queries, packed
    qall: u64 = 0, // bf16 [G, rows, H, lw + rd]: every rank's
    osend: u64 = 0, // bf16 [G, rows, H, lw]: normalized partials by destination rank
    lsend: u64 = 0, // fp32 [G, rows, H]: their log-sum-exps
    orecv: u64 = 0, // bf16 [fan, G, rows, H, lw] (fan G for decode windows' all-gather, 1 for the all-to-all)
    lrecv: u64 = 0, // fp32 [fan, G, rows, H]
    cnt: u64 = 0, // int32 [rows]: this rank's share of each row's selection
    cand: u64 = 0, // int64 [G, min(rows, 128), index_topk]: every rank's candidate keys
    dmine: u64 = 0, // int64 [min(rows, 128), index_topk]: this rank's candidates (int64 min + 1 padded)
    dsel: u64 = 0, // int64 [min(rows, 128), index_topk]: topk.top_keys' output (radix branch)
    // Phase 4: the drafters' taps (fused.Buffers.taps)
    taps: u64 = 0, // bf16 [rows, tap_n * D]
    tap_n: usize = 0,

    fn alloc(b: *Buffers, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(b.d, @max(bytes, 16));
        errdefer m.free();
        try m.fill8(0, null); // torch.zeros / empty: zero here, so stray reads are deterministic
        try b.mem.append(b.gpa, m);
        return m.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, rows: usize, score_cols: usize) !Buffers {
        return initWith(gpa, d, w, rows, score_cols, .{});
    }

    /// `opts.decode` (fused.Buffers(decode=True)): every window these buffers serve is a decode / verify window,
    /// whatever its width: the decode attention tiling and head rows up to max(rows, FAST_ROWS).
    pub fn initWith(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, rows: usize, score_cols: usize, opts: BufOpts) !Buffers {
        const c = w.config;
        if (rows > max_rows and !opts.prompt) return error.WindowTooWide;
        if (rows > max_prompt_rows) return error.WindowTooWide;
        if (opts.decode and rows > decode_rows) return error.WindowTooWide;
        var b: Buffers = .{ .d = d, .gpa = gpa, .rows = rows, .score_cols = score_cols, .small = if (opts.decode) @max(rows, fast_rows) else fast_rows };
        errdefer b.deinit();
        const D = c.hidden;
        const H = w.heads;
        const lw = c.kv_lora;
        const rd = c.rope;
        const qd = c.qkDim();
        b.x = try b.alloc(rows * D * 2);
        b.normed = try b.alloc(rows * D * 2);
        b.qa = try b.alloc(rows * c.q_lora * 2);
        b.qn = try b.alloc(rows * c.q_lora * 2);
        b.kva_n = w.layers[0].kv_a.n;
        b.kva = try b.alloc(rows * b.kva_n * 2);
        b.q = try b.alloc(rows * H * qd * 2);
        b.qlat = try b.alloc(rows * H * lw * 2);
        b.qrot = try b.alloc(rows * H * rd * 2);
        const G = @max(opts.dcp, 1);
        b.dcp = G;
        // DCP: partials for every rank's heads
        const slots = @max(c.index_topk / tri.attn_decode.chk * @min(rows, b.small), c.index_topk / tri.attn_prompt.chk * rows) * G;
        if (G > 1) {
            b.qpack = try b.alloc(rows * H * (lw + rd) * 2);
            b.qall = try b.alloc(G * rows * H * (lw + rd) * 2);
            b.osend = try b.alloc(G * rows * H * lw * 2);
            b.lsend = try b.alloc(G * rows * H * 4);
            const fan: usize = if (rows <= b.small) G else 1; // decode windows exchange by all-gather (G x the bytes)
            b.orecv = try b.alloc(fan * G * rows * H * lw * 2);
            b.lrecv = try b.alloc(fan * G * rows * H * 4);
            b.cnt = try b.alloc(rows * 4);
            const sel_rows = @min(rows, 128);
            b.cand = try b.alloc(G * sel_rows * c.index_topk * 8);
            b.dmine = try b.alloc(sel_rows * c.index_topk * 8);
            b.dsel = try b.alloc(sel_rows * c.index_topk * 8);
        }
        b.po = try b.alloc(slots * H * lw * 4);
        b.pm = try b.alloc(slots * H * 4);
        b.pl = try b.alloc(slots * H * 4);
        b.ol = try b.alloc(rows * H * lw * 2);
        b.o = try b.alloc(rows * H * c.v_dim * 2);
        b.dummy = try b.alloc(4);
        b.ik = try b.alloc(rows * c.index_dim * 4);
        b.iw = try b.alloc(rows * c.index_heads * 4);
        b.iq = try b.alloc(rows * c.index_heads * c.index_dim * 2);
        b.sc = try b.alloc(@min(rows, 128) * dcpm.cdiv(score_cols, G) * 8); // Buffers.score_len
        b.tok = try b.alloc(rows * c.index_topk * 4);
        const width = @max(c.dense_width, c.expert_width * @max(c.shared_experts, 1)) / w.world;
        b.g = try b.alloc(rows * width * 2);
        b.u = try b.alloc(rows * width * 2);
        b.act = try b.alloc(rows * width * 2);
        b.mlog = try b.alloc(rows * c.experts * 4);
        b.pick = try b.alloc(rows * c.top_k * 4);
        b.wts = try b.alloc(rows * c.top_k * 4);
        b.rpart = try b.alloc(tri.router_ks * rows * @max(c.experts, c.index_dim) * 4);
        b.sy = try b.alloc(rows * D * 4);
        b.part = try b.alloc(rows * D * 4);
        b.gath = try b.alloc(w.world * rows * D * 4);
        if (w.lm_head != 0) {
            const hr = @min(rows, b.small);
            const V = @max(w.vocab_part, if (w.draft) |dh| dh.n else 0);
            b.head_rows = hr;
            b.head_v = V;
            b.fnormed = try b.alloc(hr * D * 2);
            b.lg = try b.alloc(hr * V * 4);
            b.hpart = try b.alloc(tri.router_ks * hr * V * 4);
            b.amax = try b.alloc(hr * 4 * 4);
            b.amax_all = try b.alloc(w.world * hr * 4 * 4);
            b.argmax = try b.alloc(hr * 8);
            b.gsum = try b.alloc(hr * (D / 64) * 4);
            b.cand_words = hr * 2 * smp.max_candidates + 1;
            const padded = std.mem.alignForward(usize, b.cand_words, 4); // fused.small_gather's 16-byte RoCE packs
            b.cand_send = try b.alloc(padded * 4);
            b.cand_all = try b.alloc(w.world * padded * 4);
        }
        // TF_GLM53_SIDE: the side stream's own EXL3 and router scratch (decode windows: up to min(rows, small) rows)
        const srows = @min(@min(rows, b.small), max_rows);
        {
            const sext = w.linearExtents();
            b.slin.xh_bytes = srows * sext.k * 2;
            b.slin.z_bytes = srows * sext.zn * 4;
            for (0..exl3.max_jobs) |j| {
                b.slin.xh[j] = try b.alloc(b.slin.xh_bytes);
                b.slin.z[j] = try b.alloc(b.slin.z_bytes);
            }
            b.srpart = try b.alloc(tri.router_ks * srows * @max(c.experts, c.index_dim) * 4);
        }
        // the multi-block top-k: selections of fewer than radix_min_rows rows over score_cols keys, and the sampled
        // candidates of the head rows over the vocabulary share
        {
            const trows = @max(@min(rows, radix_min_rows), b.head_rows);
            const cols = @max(score_cols, b.head_v);
            const tblocks = (cols + kern.topk_chunk - 1) / kern.topk_chunk;
            b.tsel = .{ .hist = try b.alloc(kern.TopkScratch.histBytes(trows)), .state = try b.alloc(trows * 16), .cnt = try b.alloc(trows * tblocks * 8), .rows = trows, .blocks = tblocks };
            if (b.head_rows > 0) b.cand_cols = try b.alloc(b.head_rows * smp.max_candidates * 4);
        }
        b.red = try b.alloc(rows * D * 4);
        if (rows > b.small) {
            b.bpart = try b.alloc(rows * D * 2);
            b.bred = try b.alloc(rows * D * 2);
            if (opts.prompt) b.xo = try b.alloc((rows + w.world - 1) / w.world * D * 2);
        }
        b.ids = try b.alloc(rows * 8);
        b.hidden = try b.alloc(rows * D * 2);
        if (opts.taps > 0) {
            b.taps = try b.alloc(rows * opts.taps * D * 2);
            b.tap_n = opts.taps;
        }
        if (opts.mtp) {
            b.hin = try b.alloc(rows * D * 2);
            b.ht = try b.alloc(rows * D * 2);
            b.me = try b.alloc(rows * D * 2);
            b.mh = try b.alloc(rows * D * 2);
            b.mcat = try b.alloc(rows * 2 * D * 2);
            b.erows = if (opts.mtp_dense) @min(rows, max_rows) else rows;
            b.mx32 = try b.alloc(b.erows * D * 4);
            b.epart = try b.alloc(tri.router_ks * b.erows * D * 4);
        }
        // EXL3 linear scratch: per job fp16 Xh [rows, K] and fp32 Z [SK, rows, N] (the row-invariant kernel: up to
        // max_rows rows; prompt chunks' wider calls take the prompt GEMM's workspace)
        const ext = w.linearExtents();
        const lrows = @min(rows, max_rows);
        b.lin.xh_bytes = lrows * ext.k * 2;
        b.lin.z_bytes = lrows * ext.zn * 4;
        for (0..exl3.max_jobs) |j| {
            b.lin.xh[j] = try b.alloc(b.lin.xh_bytes);
            b.lin.z[j] = try b.alloc(b.lin.z_bytes);
        }
        // experts.Scratch(ex, rows, slots = top_k); SharedExperts.scratch: min(rows, MAX_ROWS) rows
        const xrows = if (opts.shared_experts) @min(rows, max_rows) else rows;
        b.xs_rows = xrows;
        for (w.layers) |*L| {
            const ex = if (L.experts) |*e| e else continue;
            const P = xrows * c.top_k;
            const gu = try exl3.defaultTiles(ex.dims, ex.width, true);
            const dn = try exl3.defaultTiles(ex.width, ex.dims, false);
            const maxu = @min(P, ex.count);
            const xs: kern.ExpertScratch = .{
                .xg = try b.alloc(P * ex.dims * 2),
                .xu = try b.alloc(P * ex.dims * 2),
                .xd = try b.alloc(P * ex.width * 2),
                .z = try b.alloc(@max(2 * gu.sk * ex.width, dn.sk * ex.dims) * P * 4),
                .y = try b.alloc(P * ex.dims * 4),
                .ids = try b.alloc(maxu * 4),
                .count = try b.alloc(4),
                .members = try b.alloc(maxu * xrows * 4),
                .done_gu = try b.alloc(maxu * ((xrows + 15) / 16) * @max(ex.width / 128, 1) * 4),
                .done_d = try b.alloc(xrows * @max(ex.dims / 128, 1) * 4),
                .rows = xrows,
                .slots = c.top_k,
                .gu = gu,
                .dn = dn,
            };
            // members_buf starts at -1 (torch.full(..., -1))
            try b.d.check(b.d.api.cuMemsetD32_v2(xs.members, 0xFFFFFFFF, maxu * xrows), "cuMemsetD32");
            b.xs = xs;
            // prompt_experts.PromptScratch(ex, min(rows, CHUNK_ROWS), top_k) + the slot-order sums' rows
            if (opts.shared_experts and rows > xrows) {
                const pr = @min(rows, exl3.pe_chunk_rows);
                const n = pr * c.top_k;
                const ipm: usize = 112; // prompt_experts.cu IPM (TF_PE_GPM); Kernels.pe.ipm checks it at launch
                const max_items = n / ipm + @min(n, ex.count);
                b.pe = .{
                    .rows = pr,
                    .slots = c.top_k,
                    .max_items = max_items,
                    .sorted = try b.alloc(n * 4),
                    .items = try b.alloc(3 * max_items * 4),
                    .count = try b.alloc(4),
                    .xd = try b.alloc(n * ex.width * 2),
                    .pairs = try b.alloc(n * ex.dims * 2),
                    .acc = try b.alloc(pr * ex.dims * 4),
                    .out16 = try b.alloc(pr * ex.dims * 2),
                };
            }
            // the MTP layer's prompt rows in bigger blocks of the decode kernel (row-invariant: the same bits as
            // blocks of xs_rows), xg / xu / y / z carved out of b.gath (world * rows * D fp32); the group kernel
            // stages a block's picks in 48 KiB of shared memory (1536 rows of 8 slots)
            if (opts.prompt and opts.mtp and !opts.mtp_dense and opts.shared_experts and rows > xrows and b.gath != 0) {
                const zcols = @max(2 * gu.sk * ex.width, dn.sk * ex.dims);
                const per_slot = std.mem.alignForward(usize, ex.dims * 2, 256) * 2 + ex.dims * 4 + zcols * 4;
                const room = w.world * rows * D * 4;
                var rb = room / (per_slot * c.top_k + 4 * 256);
                rb = @min(@min(rb, rows), 49152 / (4 * c.top_k));
                rb = rb / 16 * 16;
                if (rb > xrows) {
                    const Pb = rb * c.top_k;
                    const maxub = @min(Pb, ex.count);
                    // carved in order: xg, xu, z, y (each 256-byte aligned)
                    var off: usize = 0;
                    var region: [4]u64 = undefined;
                    for ([_]usize{ Pb * ex.dims * 2, Pb * ex.dims * 2, zcols * Pb * 4, Pb * ex.dims * 4 }, 0..) |bytes, j| {
                        region[j] = b.gath + off;
                        off = std.mem.alignForward(usize, off + bytes, 256);
                    }
                    const big: kern.ExpertScratch = .{
                        .xg = region[0],
                        .xu = region[1],
                        .xd = try b.alloc(Pb * ex.width * 2),
                        .z = region[2],
                        .y = region[3],
                        .ids = try b.alloc(maxub * 4),
                        .count = try b.alloc(4),
                        .members = try b.alloc(maxub * rb * 4),
                        .done_gu = try b.alloc(maxub * ((rb + 15) / 16) * @max(ex.width / 128, 1) * 4),
                        .done_d = try b.alloc(rb * @max(ex.dims / 128, 1) * 4),
                        .rows = rb,
                        .slots = c.top_k,
                        .gu = gu,
                        .dn = dn,
                    };
                    if (off > room) return error.ScratchTooSmall;
                    try b.d.check(b.d.api.cuMemsetD32_v2(big.members, 0xFFFFFFFF, maxub * rb), "cuMemsetD32");
                    b.xs_big = big;
                }
            }
            break; // one scratch serves every MoE layer (the same shapes)
        }
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        for (b.mem.items) |*m| m.free();
        b.mem.deinit(b.gpa);
        b.* = undefined;
    }
};

/// fused.State: a 576-wide latent row a token and layer, bf16 index keys on full-indexer layers, the device position.
/// DCP (Phase 3b): position p on rank p % dcp at local row p // dcp; every cache holds `local` = ceil(capacity / dcp)
/// + 1 rows (capacity + 1 at DCP 1).
pub const State = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    capacity: usize,
    dcp: usize = 1,
    local: usize = 0,
    kc: []u64, // per layer: bf16 [local, latent]
    ic: []u64, // per layer: bf16 [local, index_dim], 0 on layers without an indexer
    pos: u64, // int32 [1]
    mkc: u64 = 0, // the MTP layer's latent rows (0: no MTP layer)
    mic: u64 = 0, // the MTP layer's index keys (0: no indexer there)
    mpos: u64 = 0, // int32 [1]: the MTP window's first position
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Phase 4b (fused.State(slots=N)): each cache holds `slots` streams' rows back to back, `local` rows each (slot s
    /// at rows [s local, (s + 1) local)); `slotView` is one stream alone (the one-stream kernels on its rows). A view
    /// owns only its kc / ic arrays (`view` true: `freeView`, never `deinit`).
    slots: usize = 1,
    view: bool = false,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize) !State {
        return initDcp(gpa, d, w, capacity, 1);
    }

    /// fused.State(w, capacity) with w.dcp = `dcp`: this rank's shard, ceil(capacity / dcp) + 1 rows a cache.
    pub fn initDcp(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize, dcp: usize) !State {
        return initSlots(gpa, d, w, capacity, dcp, 1);
    }

    /// fused.State(w, capacity, slots): `slots` streams of ceil(capacity / dcp) + 1 rows each (slots > 1: DCP 1).
    pub fn initSlots(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize, dcp: usize, slots: usize) !State {
        const c = w.config;
        if (dcp == 0 or (dcp > 1 and dcp != w.world)) return error.Invalid;
        if (slots == 0 or (slots > 1 and dcp > 1)) return error.Invalid; // concurrent streams need DCP off
        const rows = dcpm.localCapacity(capacity, dcp);
        const all = rows * slots;
        var s: State = .{ .d = d, .gpa = gpa, .capacity = capacity, .dcp = dcp, .local = rows, .slots = slots, .kc = try gpa.alloc(u64, w.layers.len), .ic = try gpa.alloc(u64, w.layers.len), .pos = 0 };
        errdefer s.deinit();
        for (w.layers, 0..) |*L, i| {
            s.kc[i] = try s.zeros(all * c.latentWidth() * 2);
            s.ic[i] = if (L.indexer != null) try s.zeros(all * c.index_dim * 2) else 0;
        }
        s.pos = try s.zeros(4);
        if (w.mtp) |*m| {
            s.mkc = try s.zeros(all * c.latentWidth() * 2);
            s.mic = if (m.layer.indexer != null) try s.zeros(all * c.index_dim * 2) else 0;
            s.mpos = try s.zeros(4);
        }
        return s;
    }

    /// fused.SlotView: stream `slot` alone - every cache pointer moved to its first row, the positions shared with
    /// the whole State (each window sets them before it runs). Free with `freeView`.
    pub fn slotView(s: *const State, w: *const W.Weights, slot: usize) !State {
        if (slot >= s.slots) return error.Invalid;
        const c = w.config;
        const lat = s.local * c.latentWidth() * 2;
        const idx = s.local * c.index_dim * 2;
        var v = s.*;
        v.view = true;
        v.mem = .empty;
        v.slots = 1;
        v.kc = try s.gpa.alloc(u64, s.kc.len);
        errdefer s.gpa.free(v.kc);
        v.ic = try s.gpa.alloc(u64, s.ic.len);
        for (s.kc, v.kc) |a, *b| b.* = a + slot * lat;
        for (s.ic, v.ic) |a, *b| b.* = if (a != 0) a + slot * idx else 0;
        if (s.mkc != 0) v.mkc = s.mkc + slot * lat;
        if (s.mic != 0) v.mic = s.mic + slot * idx;
        return v;
    }

    pub fn freeView(v: *State) void {
        std.debug.assert(v.view);
        v.gpa.free(v.kc);
        v.gpa.free(v.ic);
        v.* = undefined;
    }

    /// fused.State.bytes_for: what the caches of `capacity` tokens take (the cache guard's figure).
    pub fn bytesFor(w: *const W.Weights, capacity: usize) usize {
        return bytesForDcp(w, capacity, 1);
    }

    /// Runner._check_cache_fits' figure: the target layers' bf16 latent rows (index keys and the MTP layer not
    /// counted, as Python counts) of `slots` streams of `capacity` tokens on this rank.
    pub fn latentBytes(w: *const W.Weights, capacity: usize, dcp: usize, slots: usize) usize {
        const c = w.config;
        return w.layers.len * c.latentWidth() * 2 * dcpm.localCapacity(capacity, @max(dcp, 1)) * slots;
    }

    /// fused.State.bytes_for with w.dcp: a rank's shard of the caches.
    pub fn bytesForDcp(w: *const W.Weights, capacity: usize, dcp: usize) usize {
        const c = w.config;
        var per: usize = 0;
        for (w.layers) |*L| per += c.latentWidth() * 2 + (if (L.indexer != null) c.index_dim * 2 else 0);
        if (w.mtp) |*m| per += c.latentWidth() * 2 + (if (m.layer.indexer != null) c.index_dim * 2 else 0);
        return per * dcpm.localCapacity(capacity, @max(dcp, 1));
    }

    /// prefixes.row_views' m: the local rows a state of n tokens reads on this rank (every rank counts the most any
    /// rank holds, ceil(n / dcp), so every rank saves the same bytes).
    pub fn localRows(s: *const State, n: usize) usize {
        return dcpm.cdiv(n, s.dcp);
    }

    /// prefixes.row_views' mm: the MTP layer's local rows of a state of n tokens (positions < n - 1).
    pub fn localMtpRows(s: *const State, n: usize) usize {
        return dcpm.cdiv(n -| 1, s.dcp);
    }

    /// A fresh request: every cache row and the position zeroed (fused.State's torch.zeros).
    pub fn reset(s: *State) !void {
        for (s.mem.items) |*m| try m.fill8(0, null);
    }

    fn zeros(s: *State, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(s.d, bytes);
        errdefer m.free();
        try m.fill8(0, null);
        try s.mem.append(s.gpa, m);
        return m.ptr;
    }

    pub fn deinit(s: *State) void {
        for (s.mem.items) |*m| m.free();
        s.mem.deinit(s.gpa);
        s.gpa.free(s.kc);
        s.gpa.free(s.ic);
        s.* = undefined;
    }
};

pub const Forward = struct {
    w: *const W.Weights,
    k: *const kern.Kernels,
    s: cuda.Stream,
    inv: u64, // fp32 [rope / 2]: fused.inv_freq (Phase 1: the oracle's bytes; Phase 2a: Kernels.invFreq's)
    /// the world's NCCL communicator (rank and world as the weights' cut); null: the identity exchange of Phase 1
    comm: ?*const cuda.nccl.Communicator = null,
    /// Phase 3a: the RoCE one-shot runtime (roce.RoceReduce, w.fast): decode windows (R <= b.small) reduce over it,
    /// and the head's records and sampled candidates gather over it; null: NCCL
    fast: ?*const roce.Roce = null,
    /// Phase 3a: TF_GLM53_PREFILL_REDUCE=ring - windows wider than b.small (prompt chunks) reduce with NCCL's bf16
    /// ring all-reduce (fused.gather's ring branch); false: the all-gather + rank-order sum of Phase 2
    ring: bool = false,
    /// Phase 3a: what windows over max_rows rows need (the prompt GEMM's workspace on this stream, cuBLAS for the
    /// batched absorb / expand); null: no window over max_rows rows
    wide: ?*const prompt.Wide = null,
    /// weights.EXPERTS_IMPL == "shared": windows over b.xs_rows rows take prompt_experts (SharedExperts.prefill)
    shared_experts: bool = false,
    /// TF_EXL3_PROMPT_DET=slots16 (reproducible prompt experts); false: fp32 red.add in arrival order (the default)
    pe_det16: bool = true,
    /// Phase 3a speed. TF_GLM53_SIDE: the decode windows' second stream (cuda_decode.Side); null: one stream
    side: ?*dec.Side = null,
    /// TF_GLM53_L2PF: the L2 prefetch of the target's decode windows (cuda_decode.L2pf); null: off
    l2pf: ?*dec.L2pf = null,
    /// decode-sized selections (fewer than radix_min_rows rows) through the multi-block top-k; false: the
    /// one-program radix select for every block (the same columns)
    multi_select: bool = true,
    /// the prompt path's per-phase timers (cuda_prompt.Prof, --prompt-prof); null: none
    prof: ?*prompt.Prof = null,
    /// Phase 3a speed (--mtp-dense 1): the MTP layer's prompt rows (windows over max_rows rows) through cuBLAS -
    /// eh_proj and its routed experts as dense bf16 GEMMs (cuda_prompt.MtpDense); null: the router kernel and the
    /// decode expert kernel in blocks. Drafts only: the target verifies every token, so no output token moves.
    mtp_dense: ?*prompt.MtpDense = null,
    /// Phase 3b: decode context parallelism (w.dcp: 1, or the world - every rank of the TP world is one of the DCP
    /// group, its rank w.rank). The State must be State.initDcp(.., dcp) and every Buffers BufOpts.dcp = dcp.
    dcp: usize = 1,
    /// Phase 4: the target layers whose output rows the drafters read (fused.Weights.tap_slot): every target window
    /// copies them into its buffers' taps (Buffers with BufOpts.taps = taps.n); n 0: none
    taps: Taps = .{},
    /// Phase 4b (--parallel N, fused.Rows): several streams in one window - the `pos` a layer is given is then an int32
    /// table of each row's own position and this an int32 table of each row's cache base row (its stream's slot
    /// start), every position-dependent kernel launched with ROWS (cuda_triton *Rows); 0: one stream, `pos` the
    /// window's first position. DCP 1 only (concurrent streams need it off).
    base: u64 = 0,

    fn ops(f: *const Forward) kern.Ops {
        return .{ .k = f.k, .s = f.s };
    }

    fn t(f: *const Forward) tri.Tri {
        return .{ .set = &f.k.triton, .s = f.s };
    }

    fn mla(f: *const Forward) tri.Mla {
        const c = f.w.config;
        return .{ .heads = f.w.heads, .nope = c.nope, .rope = c.rope, .lw = c.kv_lora, .vd = c.v_dim, .topk = c.index_topk };
    }

    /// The side stream while a decode window runs on it (fused.Buffers.side_on), else null.
    fn sideOn(f: *const Forward) ?*dec.Side {
        const sd = f.side orelse return null;
        return if (sd.on) sd else null;
    }

    /// fused._fork: this Forward on the side stream, queued after everything f.s has queued so far.
    fn forkSide(f: *const Forward, sd: *dec.Side) !Forward {
        try sd.fork(f.s);
        var g = f.*;
        g.s = sd.s;
        g.side = null;
        g.l2pf = null;
        g.prof = null;
        return g;
    }

    /// `b` with the side stream's own EXL3 and router scratch (what torch's per-stream allocator gives fused).
    fn sideBufs(b: *const Buffers) Buffers {
        var s = b.*;
        s.lin = b.slin;
        s.rpart = b.srpart;
        return s;
    }

    /// TF_GLM53_L2PF: prefetch site `name` of layer L (nothing outside the target's decode windows).
    fn site(f: *const Forward, L: *const W.Layer, name: dec.SiteName) !void {
        const pf = f.l2pf orelse return;
        try pf.site(f.k, f.s, L.index, name);
    }

    /// --prompt-prof: a timing mark on this stream (null when the timers are off or suspended).
    pub fn pbegin(f: *const Forward) !?usize {
        const p = f.prof orelse return null;
        return p.begin(f.s);
    }

    /// --prompt-prof: the end of the region `m` opened, counted under `ph`.
    pub fn pend(f: *const Forward, m: ?usize, ph: prompt.Phase) !void {
        const p = f.prof orelse return;
        const i = m orelse return;
        try p.end(f.s, i, ph);
    }

    /// torch.index_select(embed, 0, ids): the window's token rows into b.x (row copies: exact).
    pub fn embed(f: *const Forward, b: *const Buffers, ids: []const u32) !void {
        const D = f.w.config.hidden;
        for (ids, 0..) |id, r| try f.ops().copy(b.x + r * D * 2, f.w.embed + @as(u64, id) * D * 2, D * 2);
    }

    /// st.pos.fill_(a): the window's first position.
    pub fn setPos(f: *const Forward, st: *const State, a: usize) !void {
        const v: i32 = @intCast(a);
        try f.ops().upload(st.pos, std.mem.asBytes(&v));
    }

    /// fused.layer for layer `li` over rows b.x[:R] at st.pos ..; `T`: the index range (null: no row selects).
    pub fn layer(f: *const Forward, b: *const Buffers, st: *const State, li: usize, R: usize, T: ?usize) !void {
        return f.layerWith(b, &f.w.layers[li], st.kc[li], st.ic[li], st.pos, R, T, 0);
    }

    /// fused.layer for any layer (the MTP layer too): its caches kc / ic (0: no indexer), positions from `pos`;
    /// `reuse` (TF_GLM53_MTP_REUSE, one-row MTP draft steps): attend b.tok[0] instead of selecting.
    pub fn layerWith(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, T: ?usize, reuse: u32) !void {
        const c = f.w.config;
        const D = c.hidden;
        const t_ = f.t();
        try t_.rmsnorm(b.x, D, L.input_norm, b.normed, D, R, D, c.eps);
        try f.attentionPart(b, L, kc, ic, pos, R, T, reuse);
        try f.site(L, .a); // the FFN's first weights, during the all-reduce
        const g1 = try f.gather(b, R);
        try t_.residualAdd(b.x, g1.ptr, R, D, g1.world);
        try t_.rmsnorm(b.x, D, L.post_attn_norm, b.normed, D, R, D, c.eps);
        try f.ffnPart(b, L, R);
        try f.site(L, .f); // the next layer's input projections, during the all-reduce
        const g2 = try f.gather(b, R);
        try t_.residualAdd(b.x, g2.ptr, R, D, g2.world);
    }

    /// What residual_add sums: `world` fp32 slots of [R, D] in rank order.
    pub const Gathered = struct { ptr: u64, world: usize };

    /// fused.gather. World 1: b.part itself. RoCE (decode windows, R <= b.small): the one-shot rank-order sum into
    /// b.red ([1, R, D]). Ring (Phase 3a, wider windows): b.part to bf16, NCCL's bf16 ring all-reduce, widened into
    /// b.red ([1, R, D]). Otherwise NCCL comm.all_gather(b.part[:R], b.gath[:world R D]) on the compute stream ([world,
    /// R, D], summed in rank order by residual_add). Phase 1 (other ranks absent, LocalComm): zeros in every slot but
    /// this rank's, which holds b.part[:R].
    pub fn gather(f: *const Forward, b: *const Buffers, R: usize) !Gathered {
        const world = f.w.world;
        if (world == 1) return .{ .ptr = b.part, .world = 1 };
        const D = f.w.config.hidden;
        const o = f.ops();
        if (f.fast) |r| {
            if (R <= b.small) {
                try r.allReduce(f.s, b.part, b.red, R * D * 4);
                return .{ .ptr = b.red, .world = 1 };
            }
        }
        if (f.comm) |c| {
            if (f.ring and R > b.small) {
                if (b.bpart == 0) return error.NoRingBuffers;
                try o.f32ToBf16(b.part, b.bpart, R * D);
                try c.allReduce(b.bpart, b.bred, R * D, .bf16, .sum, f.s.handle);
                try o.bf16ToF32(b.bred, b.red, R * D);
                return .{ .ptr = b.red, .world = 1 };
            }
            try c.allGather(b.part, b.gath, R * D, .f32, f.s.handle);
            return .{ .ptr = b.gath, .world = world };
        }
        try o.fill32(b.gath, 0, world * R * D);
        try o.copy(b.gath + f.w.rank * R * D * 4, b.part, R * D * 4);
        return .{ .ptr = b.gath, .world = world };
    }

    /// fused.lin: an EXL3 linear of x [R, K] (bf16, contiguous rows) into out [R, N] of dtype y - the row-invariant
    /// kernel up to max_rows rows, the prompt GEMM (x3prefill.matmul, weights decoded once) above.
    pub fn lin(f: *const Forward, b: *const Buffers, l: *const exl3.Linear, x: u64, R: usize, out: u64, y: exl3.DType) !void {
        if (R <= max_rows) return f.ops().linear(l, x, .bf16, out, y, R, b.lin);
        const wd = f.wide orelse return error.WindowTooWide;
        return wd.matmul(f, l, x, R, out, y);
    }

    /// fused.lins: linears of one input in one launch when groupable (up to max_rows rows), else one `lin` each.
    pub fn lins(f: *const Forward, b: *const Buffers, ls: []const *const exl3.Linear, x: u64, R: usize, outs: []const u64, ys: []const exl3.DType) !void {
        if (R <= max_rows) return f.ops().linears(ls, x, .bf16, outs, ys, R, b.lin);
        for (ls, outs, ys) |l, out, y| try f.lin(b, l, x, R, out, y);
    }

    /// fused.attention_part and attention_core: this rank's fp32 partial in b.part[:R]. `reuse` 1 / 2 (an MTP draft
    /// step after its chain's first, one row): no index key written, no selection; the row attends b.tok[0] (mode 1
    /// puts its own position in the list's last slot). TF_GLM53_SIDE "a" / "w" inside a decode window: the key path
    /// (keys) on the side stream beside the query path, joined before select or the attention kernels.
    fn attentionPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, T: ?usize, reuse: u32) !void {
        const c = f.w.config;
        const o = f.ops();
        const t_ = f.t();
        const D = c.hidden;
        if (reuse != 0 and (L.indexer == null or R != 1)) return error.Invalid;
        if (f.base != 0 and (reuse != 0 or f.dcp > 1)) return error.Invalid; // several streams: no MTP index reuse, DCP 1
        const sd = f.sideOn();
        const side_a = if (sd) |x| x.letters.a else false;
        const side_w = if (sd) |x| x.letters.w else false;
        if (side_a) {
            const g = try f.forkSide(sd.?);
            const bs = sideBufs(b);
            try g.lin(&bs, &L.kv_a, b.normed, R, b.kva, .bf16); // the bits of its share of the grouped launch
            try g.keys(&bs, L, kc, ic, pos, R, reuse);
            try f.lin(b, &L.q_a, b.normed, R, b.qa, .bf16);
        } else {
            try f.lins(b, &.{ &L.q_a, &L.kv_a }, b.normed, R, &.{ b.qa, b.kva }, &.{ .bf16, .bf16 });
            if (side_w) {
                const g = try f.forkSide(sd.?);
                const bs = sideBufs(b);
                try g.keys(&bs, L, kc, ic, pos, R, reuse);
            }
        }
        try t_.rmsnorm(b.qa, c.q_lora, L.q_a_norm, b.qn, c.q_lora, R, c.q_lora, c.eps);
        if (!side_a and !side_w) try f.keys(b, L, kc, ic, pos, R, reuse);
        // b.tok[0, K-1:] = pos (DCP 1 only: under DCP the row's own key is in its owner's share already)
        if (reuse == 1 and T != null and f.dcp == 1) try o.copy(b.tok + (c.index_topk - 1) * 4, pos, 4);
        var q_done = false;
        if (L.indexer) |*ix| {
            if (T) |tt| {
                if (reuse == 0) {
                    try t_.router(b.normed, D, ix.weights_proj, b.iw, b.rpart, R, D, c.index_heads);
                    try f.lins(b, &.{ &ix.wq_b, &L.q_b }, b.qn, R, &.{ b.iq, b.q }, &.{ .bf16, .bf16 });
                    q_done = true;
                    if (f.base != 0) {
                        try t_.iqRopeRows(b.iq, f.inv, pos, R, c.index_heads, c.index_dim, c.rope);
                    } else {
                        try t_.iqRope(b.iq, f.inv, pos, R, c.index_heads, c.index_dim, c.rope);
                    }
                    if (sd) |x| try x.join(f.s); // the window's index keys are in
                    try f.select(b, ic, pos, 0, R, tt);
                }
            }
        }
        try f.attentionCore(b, L, kc, pos, R, q_done);
    }

    /// attention_part's keys(): the window's latent and rope key and, on indexer layers (not under reuse), the index
    /// key into the caches (kv_a already in b.kva).
    fn keys(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, reuse: u32) !void {
        const c = f.w.config;
        const t_ = f.t();
        const rank = f.dcpRank();
        if (f.base != 0) {
            try t_.kvWriteRows(b.kva, b.kva_n, L.kv_a_norm, kc, pos, f.base, f.inv, c.eps, R, f.mla());
        } else {
            try t_.kvWriteDcp(b.kva, b.kva_n, L.kv_a_norm, kc, pos, f.inv, c.eps, R, f.mla(), f.dcp, rank);
        }
        if (L.indexer) |*ix| {
            if (reuse == 0) {
                try t_.router(b.normed, c.hidden, ix.wk, b.ik, b.rpart, R, c.hidden, c.index_dim);
                if (f.base != 0) {
                    try t_.ikWriteRows(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, ic, pos, f.base, R, c.index_dim, c.rope);
                } else {
                    try t_.ikWriteDcp(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, ic, pos, R, c.index_dim, c.rope, f.dcp, rank);
                }
            }
        }
    }

    /// fused.attention_core for rows b.qn[:R] (selection in b.tok): q_b (unless attention_part ran it grouped with
    /// wq_b), absorb, attention over the cache, expand, o_proj -> this rank's fp32 partial b.part[:R]. Windows over
    /// max_rows rows (prompt chunks): q_b / o_proj through the prompt GEMM, absorb and expand as torch.bmm (cuBLAS,
    /// tensor cores: not row-exact, as the Python engine's) with _qrope beside.
    pub fn attentionCore(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, pos: u64, R: usize, q_done: bool) !void {
        const t_ = f.t();
        const m = f.mla();
        if (!q_done) {
            const p0 = try f.pbegin();
            try f.lin(b, &L.q_b, b.qn, R, b.q, .bf16);
            try f.pend(p0, .attn_proj);
        }
        const p1 = try f.pbegin();
        const wide = R > max_rows;
        if (wide) {
            if (f.base != 0) return error.WindowTooWide; // several streams: decode windows only
            const wd = f.wide orelse return error.WindowTooWide;
            try wd.absorb(f, b, L, pos, R);
        } else {
            if (f.base != 0) {
                try t_.absorbRows(b.q, L.wk, f.inv, b.qlat, b.qrot, pos, R, m);
            } else {
                try t_.absorb(b.q, L.wk, f.inv, b.qlat, b.qrot, pos, R, m);
            }
            try f.site(L, .o); // expand's and o_proj's weights, during the attention core
        }
        const tiling = if (R <= b.small) tri.attn_decode else tri.attn_prompt;
        if (f.sideOn()) |sd| try sd.join(f.s); // TF_GLM53_SIDE: the window's latents are in the cache
        if (f.dcp > 1) {
            try f.attentionDcp(b, kc, pos, R, tiling);
        } else if (f.base != 0) {
            try t_.attentionRows(b.qlat, b.qrot, kc, b.tok, pos, f.base, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
        } else {
            try t_.attention(b.qlat, b.qrot, kc, b.tok, pos, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
        }
        if (wide) {
            try f.wide.?.expand(f, b, L, R);
        } else {
            try t_.expand(b.ol, L.wv, b.o, R, m);
        }
        try f.pend(p1, .attn_core);
        const p2 = try f.pbegin();
        try f.lin(b, &L.o_proj, b.o, R, b.part, .f32);
        try f.pend(p2, .attn_proj);
    }

    // ------------------------------------------------------------------------------- Phase 2b: device windows ---

    /// fused.compute for rows b.ids[:R] (device ids) at st.pos ..: the target's caches written, its final hidden in
    /// b.hidden[:R], then `head` over every row (`last`: only row R - 1, as logits="last"). No host step: capturable.
    pub fn compute(f: *const Forward, b: *const Buffers, st: *const State, R: usize, T: ?usize, hm: HeadMode, last: bool) !void {
        // fused.compute's _SideWindow (TF_GLM53_SIDE: windows up to min(b.small, MAX_ROWS) rows) and _compute_pf
        // (TF_GLM53_L2PF: up to min(b.small, L2PF_ROWS) rows); both streams rejoin before the window ends
        const sd = f.sideWindow(b, R);
        var pf: ?*dec.L2pf = null;
        if (f.l2pf) |p| {
            if (!p.active and R <= @min(b.small, p.set.rows)) pf = p;
        }
        if (sd) |x| x.on = true;
        if (pf) |p| p.active = true;
        defer {
            if (sd) |x| x.on = false;
            if (pf) |p| p.active = false;
        }
        try f.computeWindow(b, st, R, T, hm, last);
        if (pf) |p| try p.join(f.s);
        if (sd) |x| try x.join(f.s);
    }

    /// The side stream a window of R rows of `b` runs beside, if any (_SideWindow: not already inside one).
    fn sideWindow(f: *const Forward, b: *const Buffers, R: usize) ?*dec.Side {
        const sd = f.side orelse return null;
        if (sd.on or !sd.letters.any() or R > @min(b.small, max_rows)) return null;
        return sd;
    }

    fn computeWindow(f: *const Forward, b: *const Buffers, st: *const State, R: usize, T: ?usize, hm: HeadMode, last: bool) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        try o.embedRows(f.w.embed, b.ids, D, b.x, R, false);
        for (0..f.w.layers.len) |li| {
            try f.layer(b, st, li, R, T);
            try f.tapRows(b, li, b.x, R);
        }
        try o.copy(b.hidden, b.x, R * D * 2);
        if (hm == .none) return;
        if (last) return f.headDevice(b, b.x + (R - 1) * D * 2, f.w.final_norm, 1, hm);
        return f.headDevice(b, b.x, f.w.final_norm, R, hm);
    }

    /// Phase 4 (fused._compute's tap): layer li's output rows x [R, D] into b.taps' column block of its slot, when a
    /// drafter reads that layer (and `b` holds taps).
    pub fn tapRows(f: *const Forward, b: *const Buffers, li: usize, x: u64, R: usize) !void {
        if (f.taps.n == 0 or b.taps == 0 or li >= f.taps.slot.len) return;
        const slot = f.taps.slot[li];
        if (slot < 0) return;
        const D = f.w.config.hidden;
        const sl: usize = @intCast(slot);
        try f.ops().copyRows(b.taps + sl * D * 2, b.tap_n * D * 2, x, D * 2, D * 2, R);
    }

    /// fused.head on n rows from `x` (bf16, row stride D) normed with `norm`: "local" leaves this rank's logits in
    /// b.lg[:n] ([n, vocab_part], contiguous); "argmax" / "draft" leave the token in b.argmax[:n] (int64) after exchanging
    /// each rank's first maximum ([value, id, -, -] records, b.amax_all) and taking the lowest rank among the equal
    /// maxima. "draft": the 4-bit draft head over the reduced vocabulary (headq q4), ids through its table.
    pub fn headDevice(f: *const Forward, b: *const Buffers, x: u64, norm: u64, n: usize, mode: HeadMode) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        const t_ = f.t();
        const world = f.w.world;
        if (f.w.lm_head == 0 or b.lg == 0) return error.NoHead;
        if (n > b.head_rows) return error.WindowTooWide;
        try t_.rmsnorm(x, D, norm, b.fnormed, D, n, D, c.eps);
        const V = f.w.vocab_part;
        switch (mode) {
            .none => return,
            .local => return t_.router(b.fnormed, D, f.w.lm_head, b.lg, b.hpart, n, D, V),
            .argmax => {
                try t_.router(b.fnormed, D, f.w.lm_head, b.lg, b.hpart, n, D, V);
                try o.argmaxRec(b.lg, V, V, 0, f.w.vocab_off, b.amax, n);
            },
            .draft => {
                const dh = f.w.draft orelse return error.NoDraftHead;
                try t_.groupSums(b.fnormed, D, b.gsum, n, D);
                try o.q4Head(b.fnormed, b.gsum, dh.w, dh.scales, dh.biases, dh.n, dh.npad, dh.k, dh.sk, b.lg, n);
                try o.argmaxRec(b.lg, dh.n, dh.n, dh.ids, 0, b.amax, n);
            },
        }
        if (f.fast != null and world > 1) {
            try f.fast.?.allGather(f.s, b.amax, b.amax_all, n * 16); // fused.head: w.fast.all_gather(a, g)
        } else if (f.comm) |cm| {
            try cm.allGather(b.amax, b.amax_all, n * 4, .f32, f.s.handle);
        } else if (world > 1) {
            try o.fill32(b.amax_all, 0, world * n * 4);
            try o.copy(b.amax_all + f.w.rank * n * 16, b.amax, n * 16);
        } else {
            try o.copy(b.amax_all, b.amax, n * 16);
        }
        try o.pickRank(b.amax_all, world, n, b.argmax);
    }

    /// fused.small_gather: every rank's fp32 words x [n] -> out [world, n_padded] in rank order over the RoCE gather
    /// when it is up and the whole result fits its buffers (n padded to whole 16-byte packs; the pad words are this
    /// rank's zeros), else NCCL ([world, n]). Returns the row stride in words. A gather only moves words.
    pub fn smallGather(f: *const Forward, x: u64, out: u64, n: usize) !usize {
        const world = f.w.world;
        const np = std.mem.alignForward(usize, n, 4);
        if (f.fast) |r| {
            if (world > 1 and world * np * 4 <= roce.max_bytes) {
                if (np > n) try f.ops().fill32(x + n * 4, 0, np - n);
                try r.allGather(f.s, x, out, np * 4);
                return np;
            }
        }
        if (f.comm) |cm| {
            try cm.allGather(x, out, n, .f32, f.s.handle);
        } else {
            try f.ops().copy(out, x, n * 4);
        }
        return n;
    }

    /// The MTP head's mode: the reduced 4-bit draft vocabulary when loaded, else the full share's argmax.
    pub fn draftMode(f: *const Forward) HeadMode {
        return if (f.w.draft != null) .draft else .argmax;
    }

    pub const MtpOpts = struct {
        /// head on the last row (logits="last"); false: logits="none" (prompt rows)
        head: bool = true,
        /// position 0's embedding zeroed (the first prompt chunk)
        zero_first: bool = false,
        /// TF_GLM53_MTP "x/normed": the chain passes on shared_head-normed rows
        chain_normed: bool = true,
        /// a chain's first step keeps its last row's selection in b.tok[0]
        keep_sel: bool = false,
        /// later one-row steps attend b.tok[0] (TF_GLM53_MTP_REUSE 1 / 2)
        reuse: u32 = 0,
        /// Phase 4b (mtp_compute's `last`, several streams): the head over `last_n` rows of the layer's output picked
        /// by the int64 device table `last` (each drafting stream's last row) into b.me first; 0: the window's last row
        last: u64 = 0,
        last_n: usize = 0,
    };

    /// fused.mtp_compute: rows (b.hin[:n] = the previous position's hidden, b.ids[:n] = the token) through the MTP
    /// layer at st.mpos ..; b.hidden[:n] the chain's rows; with `head` the last row's draft in b.argmax[0].
    pub fn mtpCompute(f: *const Forward, b: *const Buffers, st: *const State, n: usize, T: ?usize, opts: MtpOpts) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        const t_ = f.t();
        const m = f.w.mtp orelse return error.NoMtpLayer;
        if (b.mcat == 0) return error.NoMtpScratch;
        try o.embedRows(f.w.embed, b.ids, D, b.me, n, opts.zero_first);
        try t_.rmsnorm(b.me, D, m.enorm, b.mh, D, n, D, c.eps);
        try t_.rmsnorm(b.hin, D, m.hnorm, b.normed, D, n, D, c.eps);
        try t_.cat2(b.mh, b.normed, b.mcat, n, D);
        if (f.mtp_dense != null and n > max_rows) {
            // prompt rows: one cuBLAS bf16 GEMM into b.x (bf16 of fp32 sums, as the router's split-K partials + cast)
            try f.mtp_dense.?.ehProj(f, b.mcat, m.eh_proj, b.x, n);
        } else {
            if (n > b.erows) return error.NoMtpScratch;
            try t_.router(b.mcat, 2 * D, m.eh_proj, b.mx32, b.epart, n, 2 * D, D);
            try o.f32ToBf16(b.mx32, b.x, n * D);
        }
        {
            // TF_GLM53_SIDE: draft and MTP refresh windows too (no L2 prefetch: l2pf skips the MTP passes)
            const sd = f.sideWindow(b, n);
            if (sd) |x| x.on = true;
            defer if (sd) |x| {
                x.on = false;
            };
            try f.layerWith(b, &f.w.mtp.?.layer, st.mkc, st.mic, st.mpos, n, T, opts.reuse);
            if (sd) |x| try x.join(f.s);
        }
        if (opts.keep_sel and T != null and n > 1) {
            try o.copy(b.tok, b.tok + (n - 1) * c.index_topk * 4, c.index_topk * 4);
            if (f.dcp > 1) try o.copy(b.cnt, b.cnt + (n - 1) * 4, 4); // b.cnt[0:1].copy_(b.cnt[n - 1:n])
        }
        if (opts.chain_normed) {
            try t_.rmsnorm(b.x, D, m.head_norm, b.hidden, D, n, D, c.eps);
        } else {
            try o.copy(b.hidden, b.x, n * D * 2);
        }
        if (!opts.head) return;
        if (opts.last != 0) {
            // several streams: index_select(x, 0, last, out=b.me[:S]) (the embeddings are spent by now), then the
            // head over those S rows
            if (opts.last_n == 0 or opts.last_n > n) return error.Invalid;
            try o.rowsGather(b.me, b.x, opts.last, opts.last_n, D * 2);
            return f.headDevice(b, b.me, m.head_norm, opts.last_n, f.draftMode());
        }
        try f.headDevice(b, b.x + (n - 1) * D * 2, m.head_norm, 1, f.draftMode());
    }

    /// fused.target_hidden_for_mtp (normed): rmsnorm(src.hidden[row0 .. row0 + n], final_norm) into `out` rows.
    pub fn targetHidden(f: *const Forward, src: *const Buffers, row0: usize, n: usize, out: u64) !void {
        const c = f.w.config;
        const D = c.hidden;
        try f.t().rmsnorm(src.hidden + row0 * D * 2, D, f.w.final_norm, out, D, n, D, c.eps);
    }

    /// fused.select (DCP 1) through the radix select for every window size: b.tok[row0 : row0 + R] ascending, window
    /// rows row0 .. (row0 > 0: a sequence-parallel prompt half's own rows) in blocks of SEL_ROWS = 128 from row0.
    pub fn select(f: *const Forward, b: *const Buffers, ic: u64, pos: u64, row0: usize, R: usize, T: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        if (T > b.score_cols) return error.ScoreBufferTooSmall;
        if (T < c.index_topk) return error.Invalid;
        if (f.dcp > 1) {
            if (f.base != 0) return error.Invalid;
            return f.selectDcp(b, ic, pos, row0, R, T);
        }
        var r0: usize = row0;
        while (r0 < row0 + R) : (r0 += 128) {
            const n = @min(128, row0 + R - r0);
            const iq = b.iq + r0 * c.index_heads * c.index_dim * 2;
            const iw = b.iw + r0 * c.index_heads * 4;
            if (f.base != 0) {
                try t_.indexScoresRows(iq, iw, ic, pos, f.base, b.sc, T, r0, n, c.index_heads, c.index_dim);
            } else {
                try t_.indexScores(iq, iw, ic, pos, b.sc, T, r0, n, c.index_heads, c.index_dim);
            }
            const out = b.tok + r0 * c.index_topk * 4;
            const blocks = (T + kern.topk_chunk - 1) / kern.topk_chunk;
            if (f.multi_select and n < radix_min_rows and n <= b.tsel.rows and blocks <= b.tsel.blocks) {
                // fused.select below RADIX_MIN_ROWS (torch.topk + sort there): the same K columns, ascending
                try f.ops().topkColumns(b.sc, T, T, n, c.index_topk, false, b.tsel, out, c.index_topk);
            } else {
                try t_.selectKeys(b.sc, out, T, n, c.index_topk);
            }
        }
    }

    // ---------------------------------------------------------------- Phase 3b: decode context parallelism ---

    /// This rank in the DCP group (fused: `w.rank if w.dcp > 1 else 0`).
    fn dcpRank(f: *const Forward) usize {
        return if (f.dcp > 1) f.w.rank else 0;
    }

    /// fused.dcp_gather: every rank's `bytes` at x into out [world, bytes] in rank order - the RoCE one-shot gather for
    /// decode-window exchanges (`small`) when it is up and the whole result fits its buffers, else NCCL (`count`
    /// elements of `dt`). A gather only moves words: both give the same bits.
    fn dcpGather(f: *const Forward, x: u64, out: u64, bytes: usize, count: usize, dt: cuda.nccl.DataType, small: bool) !void {
        const world = f.w.world;
        if (small) if (f.fast) |r| {
            if (bytes % 16 == 0 and world * bytes <= roce.max_bytes) return r.allGather(f.s, x, out, bytes);
        };
        const cm = f.comm orelse return error.NoCommunicator;
        try cm.allGather(x, out, count, dt, f.s.handle);
    }

    /// fused.select's DCP branch for window rows row0 .. row0 + R - 1 (blocks of SEL_ROWS = 128): this rank's slots
    /// t < Tl = ceil(T / dcp) scored as packed int64 keys (_index_scores DCP, RANK, PACK), its top kk = min(K, Tl) of
    /// them padded to K (int64 min + 1), every rank's candidates gathered (RoCE for decode windows), then the K best
    /// of the dcp * K - by the whole key (torch.topk) below RADIX_MIN_ROWS rows, by the high word with ties to the
    /// lower column at or above it (topk.top_keys: the Python radix branch's own tie rule) - and this rank's share as
    /// local slots ascending in b.tok, its count in b.cnt. The per-rank top kk is one set however it is found
    /// (scores tie-free by position): the multi-block top-k over the keys' high words (glm53_topk_*, mode 2) where its
    /// scratch fits, else topk.top_keys.
    fn selectDcp(f: *const Forward, b: *const Buffers, ic: u64, pos: u64, row0: usize, R: usize, T: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        const o = f.ops();
        const G = f.dcp;
        const rank = f.dcpRank();
        const K = c.index_topk;
        if (b.dcp != G or b.cand == 0) return error.NoDcpBuffers;
        const Tl = dcpm.localCols(T, G);
        const kk = @min(K, Tl);
        var r0: usize = row0;
        while (r0 < row0 + R) : (r0 += 128) {
            const n = @min(128, row0 + R - r0);
            const iq = b.iq + r0 * c.index_heads * c.index_dim * 2;
            const iw = b.iw + r0 * c.index_heads * 4;
            const radix = n >= radix_min_rows; // fused.RADIX and n >= RADIX_MIN_ROWS (dcp > 1)
            try t_.indexScoresDcp(iq, iw, ic, pos, b.sc, Tl, r0, n, c.index_heads, c.index_dim, G, rank);
            const tok = b.tok + r0 * K * 4;
            const blocks = (Tl + kern.topk_chunk - 1) / kern.topk_chunk;
            if (n <= b.tsel.rows and blocks <= b.tsel.blocks) {
                // this rank's kk best columns ascending (b.tok's rows as scratch), then their keys
                try o.topkColumnsMode(b.sc, Tl, Tl, n, kk, 2, b.tsel, tok, K);
                try o.dcpCands(b.sc, Tl, tok, K, kk, K, false, b.dmine, n);
            } else if (kk == Tl) {
                // topk.top_keys(sc, kk) with kk == Tl (Tl <= K: short prompt chunks under DCP) keeps every key in
                // column order - the score rows themselves, bit for bit. Copied rather than launched: _select_keys'
                // K constexpr would be a different variant for each such Tl (none of them in a finite AOT set).
                try o.copy(b.dsel, b.sc, n * Tl * 8);
                try o.dcpCands(b.sc, Tl, b.dsel, kk, kk, K, true, b.dmine, n);
            } else {
                try t_.topKeys(b.sc, b.dsel, Tl, n, kk);
                try o.dcpCands(b.sc, Tl, b.dsel, kk, kk, K, true, b.dmine, n);
            }
            try f.dcpGather(b.dmine, b.cand, n * K * 8, n * K, .i64, n <= b.small);
            try o.dcpMerge(b.cand, n, K, G, rank, radix, tok, b.cnt + r0 * 4);
        }
    }

    /// fused._attention_dcp: this rank's absorbed queries packed ([R, H, lw + rd]) and gathered from every rank, all
    /// G * H heads attended over this rank's keys (_attn_dcp: chunk partials), merged per head into a normalized
    /// output and log-sum-exp by destination rank (_merge_lse), exchanged (decode windows: all-gather of everything,
    /// read our block; prompt chunks: NCCL all-to-all of head blocks) and combined in rank order (_dcp_combine) ->
    /// b.ol.
    fn attentionDcp(f: *const Forward, b: *const Buffers, kc: u64, pos: u64, R: usize, tl: tri.AttnTiling) !void {
        const c = f.w.config;
        const t_ = f.t();
        const o = f.ops();
        const m = f.mla();
        const G = f.dcp;
        const rank = f.dcpRank();
        const H = m.heads;
        const lw = m.lw;
        const rd = m.rope;
        if (b.dcp != G or b.qall == 0) return error.NoDcpBuffers;
        const small = R <= b.small;
        try o.dcpQpack(b.qlat, b.qrot, b.qpack, R * H, lw, rd);
        try f.dcpGather(b.qpack, b.qall, R * H * (lw + rd) * 2, R * H * (lw + rd), .bf16, small);
        const nch = @max(1, c.index_topk / tl.chk);
        try t_.attnDcp(b.qall, kc, b.tok, b.cnt, pos, b.po, b.pm, b.pl, R, m, G, rank, tl);
        try t_.mergeLse(b.po, b.pm, b.pl, b.osend, b.lsend, R, G, m, nch);
        if (small) {
            try f.dcpGather(b.osend, b.orecv, G * R * H * lw * 2, G * R * H * lw, .bf16, true);
            try f.dcpGather(b.lsend, b.lrecv, G * R * H * 4, G * R * H, .f32, true);
        } else {
            const cm = f.comm orelse return error.NoCommunicator;
            try dcpm.allToAll(cm, b.osend, b.orecv, R * H * lw, .bf16, 2, f.s.handle);
            try dcpm.allToAll(cm, b.lsend, b.lrecv, R * H, .f32, 4, f.s.handle);
        }
        const ex = dcpm.exchange(small, G, rank, R, H, lw);
        try t_.dcpCombine(b.orecv + ex.o0 * 2, b.lrecv + ex.l0 * 4, b.ol, R, ex.ss, ex.ssl, m, G);
    }

    /// fused.head(x, final_norm, R, slice(R - 1, R), mode="argmax") for the window's last row: the final RMSNorm,
    /// this rank's logits over its vocabulary share (glue.router into b.lg), its first maximum (on the host, from
    /// b.lg: torch.argmax's rule), the [value, global id, 0, 0] record all-gathered (b.amax_all) and the lowest rank
    /// among the equal maxima. `host`: at least vocab_part floats. Synchronizes the stream (the pick goes to the host).
    pub fn head(f: *const Forward, b: *const Buffers, R: usize, host: []f32) !Pick {
        const c = f.w.config;
        const D = c.hidden;
        const V = f.w.vocab_part;
        const world = f.w.world;
        if (f.w.lm_head == 0 or b.lg == 0) return error.NoHead;
        if (host.len < V or world > max_world) return error.Invalid;
        const t_ = f.t();
        const o = f.ops();
        try t_.rmsnorm(b.x + (R - 1) * D * 2, D, f.w.final_norm, b.fnormed, D, 1, D, c.eps);
        try t_.router(b.fnormed, D, f.w.lm_head, b.lg, b.hpart, 1, D, V);
        try o.download(std.mem.sliceAsBytes(host[0..V]), b.lg);
        try f.s.synchronize();
        const i = argmaxFirst(host[0..V]);
        const rec: [4]f32 = .{ host[i], @floatFromInt(i + f.w.vocab_off), 0, 0 };
        try o.upload(b.amax, std.mem.asBytes(&rec));
        if (f.comm) |cm| {
            try cm.allGather(b.amax, b.amax_all, 4, .f32, f.s.handle);
        } else {
            try o.fill32(b.amax_all, 0, world * 4);
            try o.copy(b.amax_all + f.w.rank * 16, b.amax, 16);
        }
        var p: Pick = .{ .world = world };
        try o.download(std.mem.sliceAsBytes(p.gathered[0 .. world * 4]), b.amax_all);
        try f.s.synchronize();
        var vals: [max_world]f32 = undefined;
        for (0..world) |r| vals[r] = p.gathered[r * 4];
        const best = argmaxFirst(vals[0..world]);
        const id = p.gathered[best * 4 + 1];
        if (!(id >= 0 and id < @as(f32, @floatFromInt(c.vocab)))) return error.BadPick;
        p.token = @intFromFloat(id);
        p.local = i;
        return p;
    }

    /// fused.ffn_part: the dense MLP, or the router (fused.route) and experts_part.
    /// TF_GLM53_SIDE "f" inside a decode window: the shared expert on the side stream beside the router and the
    /// routed experts (fused.ffn_part), joined right before the routed down + combine launch reads b.sy.
    fn ffnPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize) !void {
        if (L.experts != null) {
            var sy_ev: ?cuda.Event = null;
            var sd_used: ?*dec.Side = null;
            if (f.sideOn()) |sd| {
                if (sd.letters.f and (!f.shared_experts or R <= b.xs_rows)) {
                    const g = try f.forkSide(sd);
                    const bs = sideBufs(b);
                    try g.mlp(&bs, L, R, b.sy);
                    sy_ev = try sd.mark();
                    sd_used = sd;
                }
            }
            try f.route(b, L, 0, R);
            try f.expertsPartWith(b, L, R, sy_ev);
            if (sd_used) |sd| sd.forked = false; // joined inside the routed chain (experts.routed's sy_ready)
            return;
        }
        const p0 = try f.pbegin();
        try f.mlp(b, L, R, b.part);
        try f.pend(p0, .dense);
    }

    /// fused.route: router logits and top-k of rows b.normed[r0 : r0 + n] -> b.pick / b.wts rows r0 ..
    pub fn route(f: *const Forward, b: *const Buffers, L: *const W.Layer, r0: usize, n: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        const D = c.hidden;
        try t_.router(b.normed + r0 * D * 2, D, L.router_w, b.mlog + r0 * c.experts * 4, b.rpart, n, D, c.experts);
        try t_.topk(b.mlog + r0 * c.experts * 4, L.router_bias, b.pick + r0 * c.top_k * 4, b.wts + r0 * c.top_k * 4, n, c.experts, c.top_k, c.routed_scaling, c.norm_topk);
    }

    /// fused.experts_part (shared expert not on a side stream): routed experts (picks in b.pick) + the shared expert
    /// of rows b.normed[:R] -> fp32 partial b.part[:R]. FOLD_SHARED with the decode kernel (the tf impl at every width,
    /// the shared impl up to its scratch's rows): the shared expert into b.sy first, the routed combine adds it.
    /// Wider windows of the shared impl (prompt chunks): SharedExperts.prefill (prompt_experts.cu) into b.part, then
    /// the shared expert into b.sy and b.part += b.sy.
    pub fn expertsPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize) !void {
        return f.expertsPartWith(b, L, R, null);
    }

    /// expertsPart; `sy_ev` (TF_GLM53_SIDE "f"): the shared expert is already queued on the side stream into b.sy,
    /// this event marks its end (the routed chain waits on it right before its down + combine launch).
    pub fn expertsPartWith(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize, sy_ev: ?cuda.Event) !void {
        const D = f.w.config.hidden;
        const ex = if (L.experts) |*e| e else return error.Invalid;
        if (!f.shared_experts or R <= b.xs_rows) {
            if (sy_ev == null) try f.mlp(b, L, R, b.sy); // first: its weights may still be in L2 (TF_GLM53_L2PF)
            const xs = b.xs orelse return error.NoExpertScratch;
            return f.ops().routed(b.normed, D, b.pick, b.wts, ex, xs, b.part, b.sy, R, sy_ev);
        }
        if (b.pe != null and kern.Ops.promptSupported(ex)) {
            if (f.k.pe.ipm != 112) return error.PromptExpertsUnsupported; // Buffers sized PromptScratch for IPM 112
            const p0 = try f.pbegin();
            try f.ops().promptRouted(b.normed, D, b.pick, b.wts, ex, b.pe.?, b.part, R, f.pe_det16);
            try f.pend(p0, .routed);
            const p1 = try f.pbegin();
            try f.mlp(b, L, R, b.sy);
            try f.ops().addF32(b.part, b.sy, R * D);
            try f.pend(p1, .shared);
            return;
        }
        // widths prompt_experts does not take (the MTP layer's 8-bit experts): the Python engine runs cuda-exl3's
        // grouped GEMM (not ported; its split-K atomics make it irreproducible run to run anyway). Here: the decode
        // kernel in blocks of its scratch's rows, every row its one-row bits. Only MTP prompt rows take this: drafts
        // only propose and the target verifies each, so no token depends on it (acceptance may move a little).
        // --mtp-dense 1 (the MTP layer): the shared expert into b.sy, then the dense bf16 experts through cuBLAS, the
        // combine adding b.sy (cuda_prompt.MtpDense.routed)
        if (f.mtp_dense) |md| {
            if (L.index == f.w.config.layers) {
                try f.mlp(b, L, R, b.sy);
                return md.routed(f, b, R);
            }
        }
        // blocks of xs_big.rows rows when the prompt buffers carved one (Phase 3a speed: each block re-reads every
        // routed expert it touches, so 10x fewer blocks read ~3x fewer bytes), else of the decode scratch's rows
        const xs = b.xs_big orelse (b.xs orelse return error.NoExpertScratch);
        const K = f.w.config.top_k;
        try f.mlp(b, L, R, b.sy);
        var r0: usize = 0;
        while (r0 < R) : (r0 += xs.rows) {
            const n = @min(xs.rows, R - r0);
            try f.ops().routed(b.normed + r0 * D * 2, D, b.pick + r0 * K * 4, b.wts + r0 * K * 4, ex, xs, b.part + r0 * D * 4, b.sy + r0 * D * 4, n, null);
        }
    }

    /// fused.mlp: gate and up (one launch up to max_rows rows, else one prompt GEMM each), SwiGLU, down into `out`.
    pub fn mlp(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize, out: u64) !void {
        const width = L.mlp.gate.n;
        try f.lins(b, &.{ &L.mlp.gate, &L.mlp.up }, b.normed, R, &.{ b.g, b.u }, &.{ .bf16, .bf16 });
        try f.t().swiglu2(b.g, b.u, b.act, R, width);
        try f.lin(b, &L.mlp.down, b.act, R, out, .f32);
    }
};

/// fused.inv_freq on the host: a diagnostic only. torch's bytes come from libdevice's powf on the GPU, which this
/// does not reproduce exactly; Kernels.invFreq runs the same powf (glm53_rope.cu) and is what the engine uses.
pub fn invFreq(out: []f32, dim: usize, theta: f64) void {
    for (out, 0..) |*v, i| {
        const x: f32 = @as(f32, @floatFromInt(2 * i)) * (@as(f32, 1.0) / @as(f32, @floatFromInt(dim)));
        const p: f32 = @floatCast(std.math.pow(f64, @as(f64, @as(f32, @floatCast(theta))), @as(f64, x)));
        v.* = 1.0 / p;
    }
}

/// torch.argmax's choice over one row: the first maximum, NaN above every number (the first NaN wins).
pub fn argmaxFirst(v: []const f32) usize {
    var best: usize = 0;
    for (v, 0..) |x, i| {
        if (i == 0) continue;
        if (std.math.isNan(v[best])) break;
        if (std.math.isNan(x) or x > v[best]) best = i;
    }
    return best;
}

test "argmax: first maximum, NaN first" {
    try std.testing.expectEqual(@as(usize, 1), argmaxFirst(&.{ 1, 3, 3, 2 }));
    try std.testing.expectEqual(@as(usize, 2), argmaxFirst(&.{ 1, 3, std.math.nan(f32), std.math.nan(f32) }));
    try std.testing.expectEqual(@as(usize, 0), argmaxFirst(&.{ 0.0, -0.0 }));
    try std.testing.expectEqual(@as(usize, 0), argmaxFirst(&.{5}));
}

test "inv_freq's first entries" {
    var v: [32]f32 = undefined;
    invFreq(&v, 64, 8000000.0);
    try std.testing.expectEqual(@as(f32, 1.0), v[0]);
    try std.testing.expect(v[31] < v[30] and v[31] > 0);
}
