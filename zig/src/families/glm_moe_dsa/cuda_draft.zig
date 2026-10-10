//! Phase 4: GLM-5.3's speculative drafters on the device - DSpark (cuda/dspark.py SparkDrafter, the publishable
//! drowzeys/keys-GLM-5.3-speculator.dspark-ft2) and DFlash2 (cuda/dflash.py GlmDrafter over glm5_next dflash2.Drafter;
//! any DFlash2 checkpoint, loaded from a path; no drafter weights are shipped here).
//!
//! Both are a block pass over a sliding-window context the target's tap rows build:
//!   * context (`tapsCompute`): rows of the target's tap layers [n, taps * D] -> c = hidden_norm(fc(taps)); every
//!     draft layer's K = RoPE(k_norm(k c)), V = v c into its ring of RING slots (slot = position % RING), the drafter's
//!     position advanced by n. Row-invariant (each row alone), so passes of any size give the same rows;
//!   * block (`blockCompute`): [pending, mask x (block - 1)] at the committed length through the draft layers
//!     (DSpark: plain Qwen3 layers, causal block attention over [s - W, s); DFlash2: two-tap dynamic convolutions
//!     around attention and MLP, its sliding-window ring attention), the final norm, each rank's top-k of its share of
//!     the verifier's head (its 4-bit copy, headq q4), gathered (fused.small_gather: RoCE when it fits) with the rows'
//!     extra words (DSpark: the confidence head's hidden part; DFlash2: the selector's projected rows), read once;
//!   * host (draft_host.zig): DSpark's Markov chain, confidences and cut; DFlash2's selector chain.
//! Tensor parallelism as the Python drafters: query / KV heads and MLP width split evenly over the ranks; o_proj /
//! down_proj row-parallel partials gathered and summed in rank order (glue.residual_add / the rank-order rowsum); fc,
//! norms, the confidence head's hidden part, the selector replicated. Weights 4-bit (qmm.quantize4 on the host with
//! torch's float32 rounding, shared qmm.pack), as both Python drafters' defaults.
//! The Triton kernels are the Python drafters' own (AOT: glue _rmsnorm SUMS / _embed_b16 / _swiglu, glm5_next
//! qmm _group_sums, dflash2 _prep_kernel / _dconv_kernel, dspark _block_attn_kernel, dflash _dattn_ring, glue
//! _residual_add); the 4-bit matmuls qmm_group's tiles; torch's small steps glm53_draft.cu. `enumerate` drives the same
//! compute functions in probe mode (no GPU) so cuda_coverage.zig lists every launch shape a drafter can make.
//!
//! Exact: drafts only propose. The target verifies every row and every emitted token is its own pick (greedy, or the
//! keyed sample), so drafted replies equal serial ones whatever the drafts are; every rank holds the same gathered
//! candidates and runs the same host stage, so every rank verifies the same window.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const aot = cuda.aot;
const W = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const fw = @import("cuda_forward.zig");
const smp = @import("sampling.zig");
const host = @import("draft_host.zig");

const Q4 = kern.Q4;
pub const ring_rows: usize = host.ring;
pub const tap_rows: usize = host.tap_rows;
/// Phase 4c: the most blocks a concurrent block pass takes side by side (cuda_mdraft.zig: one a stream slot).
pub const max_segs: usize = 16;
/// glm5_next dflash2.NO_LIMIT: SwiGLU's clamp (none).
const no_limit: f32 = 1e30;

pub const Kind = enum { dspark, dflash };

/// A drafter's shapes on one rank (everything the launches depend on).
pub const Spec = struct {
    kind: Kind,
    D: usize,
    hd: usize,
    H: usize, // query heads this rank
    KV: usize, // KV heads this rank
    I: usize, // MLP width this rank
    layers: usize,
    ntaps: usize, // this drafter's taps (fc's input width / D)
    block: usize,
    top_k: usize, // candidates a rank sends a slot
    V: usize, // the verifier's vocabulary share
    vocab_off: usize,
    rsel: usize = 0, // DFlash2: the selector's rank
    gs: usize = 0, // DFlash2: conv_group_size
    eps: f32,
    window: usize, // DSpark: the context rows a block sees (W); DFlash2: sliding_window - 1
    causal: bool,
    world: usize,
    theta: f64,
    mask_id: u32,

    pub fn ofDSpark(c: *const host.DSparkConfig, s: *const host.DSparkSettings, world: usize, vocab_part: usize, vocab_off: usize) Spec {
        return .{ .kind = .dspark, .D = c.hidden, .hd = c.head_dim, .H = c.heads / world, .KV = c.kv_heads / world, .I = c.inter / world, .layers = c.layers, .ntaps = c.n_aux, .block = c.block, .top_k = s.top_k, .V = vocab_part, .vocab_off = vocab_off, .eps = @floatCast(c.eps), .window = c.span(), .causal = c.causal, .world = world, .theta = c.theta, .mask_id = c.mask_id };
    }

    pub fn ofDFlash(c: *const host.DFlashConfig, world: usize, vocab_part: usize, vocab_off: usize, rsel: usize) Spec {
        return .{ .kind = .dflash, .D = c.hidden, .hd = c.head_dim, .H = c.heads / world, .KV = c.kv_heads / world, .I = c.inter / world, .layers = c.layers, .ntaps = c.n_taps, .block = c.block, .top_k = c.top_k, .V = vocab_part, .vocab_off = vocab_off, .rsel = rsel, .gs = c.gs, .eps = @floatCast(c.eps), .window = c.window, .causal = c.causal, .world = world, .theta = c.theta, .mask_id = c.mask_id };
    }

    /// Rows of the block pass's head: DSpark every slot (sample_from_anchor), DFlash2 slots 1.. (x[1:]).
    pub fn headRows(s: Spec) usize {
        return if (s.kind == .dspark) s.block else s.block - 1;
    }

    /// The most drafts a block gives (DSpark: block; DFlash2: block - 1).
    pub fn maxDepth(s: Spec) usize {
        return s.headRows();
    }

    pub fn qkvN(s: Spec) usize {
        return (s.H + 2 * s.KV) * s.hd;
    }

    pub fn kvN(s: Spec) usize {
        return 2 * s.KV * s.hd;
    }

    /// The widest pass (a tap batch or a block).
    pub fn rows(s: Spec) usize {
        return @max(tap_rows, s.block);
    }

    /// Extra fp32 words a head row adds to the read: DSpark one (the confidence logit's hidden part), DFlash2 the
    /// selector's rank.
    pub fn extra(s: Spec) usize {
        return if (s.kind == .dspark) 1 else s.rsel;
    }

    /// Words a rank sends: every head row's top_k values and ids, padded to whole 16-byte packs.
    pub fn sendWords(s: Spec) usize {
        return std.mem.alignForward(usize, s.headRows() * 2 * s.top_k, 4);
    }

    pub fn scale(s: Spec) f32 {
        return tri.f32of(std.math.pow(f64, @floatFromInt(s.hd), -0.5));
    }
};

pub const Layer = struct {
    in_norm: u64 = 0,
    post_norm: u64 = 0,
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    qkv: Q4 = .{}, // this rank's [q heads | k heads | v heads]
    kv: Q4 = .{}, // this rank's [k heads | v heads] (context rows)
    o: Q4 = .{}, // [D, this rank's heads * head_dim]: a row-parallel partial
    gu: Q4 = .{}, // this rank's [gate | up]
    down: Q4 = .{}, // [D, this rank's MLP width]: a row-parallel partial
    // DFlash2's dynamic convolutions
    a_base: u64 = 0, // bf16 [2, 2, D]
    a_kp: Q4 = .{}, // [2 * 2 * D / gs, D]
    m_base: u64 = 0,
    m_kp: Q4 = .{},
};

/// Static device buffers (graphs replay over them).
const Bufs = struct {
    x: u64 = 0, // bf16 [R, D]
    normed: u64 = 0, // bf16 [R, D]
    xs: u64 = 0, // fp32 [R, K / 64] (K up to taps * D)
    t0: u64 = 0, // bf16 [R, max(D, qkv)]: fc / qkv / kv outputs
    q: u64 = 0, // bf16 [H, R, hd]
    k: u64 = 0, // bf16 [KV, R, hd]
    v: u64 = 0,
    att: u64 = 0, // bf16 [R, H hd]
    part: u64 = 0, // fp32 [R, D]
    gath: u64 = 0, // fp32 [world, R D (+ pad)]
    rsum: u64 = 0, // bf16 [R, D]: DFlash2's row-parallel sums
    gu: u64 = 0, // bf16 [R, 2 I]
    act: u64 = 0, // bf16 [R, I]
    axs: u64 = 0, // fp32 [R, I / 64]
    dyn: u64 = 0, // bf16 [R, 4 D / gs]
    conv: u64 = 0, // bf16 [R, D]
    cos: u64 = 0, // fp32 [R, hd / 2]
    sin: u64 = 0,
    logits: u64 = 0, // fp32 [head rows, V]
    tsel: kern.TopkScratch = .{},
    cand_cols: u64 = 0, // int32 [head rows, top_k]
    send: u64 = 0, // fp32 [sendWords]
    cand: u64 = 0, // fp32 [world, sendWords] then the extra words [head rows, extra]
    ids: u64 = 0, // int32 [block]: [pending, mask ...]
    pos: u64 = 0, // int64 [1]: the committed length (positions the context holds)
    inv: u64 = 0, // fp32 [hd / 2]
    tap_in: u64 = 0, // bf16 [tap_rows, taps * D]
};

/// What a compute pass runs on: Triton through `t`, everything else through `o` / `f` (null: coverage's probe, no
/// GPU - only the Triton launches are recorded).
pub const Dev = struct {
    t: tri.Tri,
    o: ?kern.Ops = null,
    f: ?*const fw.Forward = null,
};

pub const Drafter = struct {
    sp: Spec,
    gpa: std.mem.Allocator,
    d: ?*const cuda.Driver = null,
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,
    fc: Q4 = .{},
    hidden_norm: u64 = 0,
    norm: u64 = 0,
    layers: []Layer = &.{},
    hproj: u64 = 0, // DFlash2: bf16 [rsel, D]
    conf_h: u64 = 0, // DSpark: fp32 [D] (0: no confidence head)
    head: Q4 = .{}, // the verifier's 4-bit draft head over this rank's share (headq q4, shared)
    embed: u64 = 0, // the verifier's embedding
    b: Bufs = .{},
    kc: []u64 = &.{},
    vc: []u64 = &.{},
    /// committed positions the context holds (the pending token sits at context_end)
    context_end: usize = 0,
    // the tap layout: this drafter's column blocks of the target's Buffers.taps ([rows, slots * D])
    cols: [host.max_taps]usize = @splat(0),
    slots: usize = 0,
    identity: bool = true,
    // the host stage
    chain: host.MarkovChain = .{},
    w1: []u16 = &.{},
    w2: []u16 = &.{},
    conf_m: []f64 = &.{},
    pred: []f32 = &.{}, // DFlash2 [vocab, rsel]
    succ: []f32 = &.{},
    dset: host.DSparkSettings = .{ .depth = 0 },
    fset: host.DFlashSettings = .{},
    pin: ?cuda.HostBuffer = null,
    tokens: []i64 = &.{},
    values: []f64 = &.{},
    extra: []f64 = &.{},
    scratch: []smp.Cand = &.{},
    walk: ?host.WalkScratch = null,
    /// the gathered candidates' layout of the last read
    ranks: usize = 1,
    stride: usize = 0,
    /// TF_GLM53_DSPARK_COSTS or the startup measurement: a round's ms with 0, 1, .. drafts (policy cost)
    round_ms: [host.max_block + 2]f64 = @splat(0),
    n_round_ms: usize = 0,
    /// stats
    proposals: usize = 0,

    pub fn deinit(dr: *Drafter) void {
        for (dr.mem.items) |*m| m.free();
        dr.mem.deinit(dr.gpa);
        if (dr.pin) |*p| p.free();
        dr.gpa.free(dr.layers);
        dr.gpa.free(dr.kc);
        dr.gpa.free(dr.vc);
        dr.gpa.free(dr.w1);
        dr.gpa.free(dr.w2);
        dr.gpa.free(dr.conf_m);
        dr.gpa.free(dr.pred);
        dr.gpa.free(dr.succ);
        dr.gpa.free(dr.tokens);
        dr.gpa.free(dr.values);
        dr.gpa.free(dr.extra);
        dr.gpa.free(dr.scratch);
        if (dr.walk) |*w| w.deinit(dr.gpa);
        dr.gpa.destroy(dr);
    }

    /// Device bytes the drafter holds (weights, rings, buffers).
    pub fn deviceBytes(dr: *const Drafter) usize {
        var n: usize = 0;
        for (dr.mem.items) |m| n += m.len;
        return n;
    }

    fn zeros(dr: *Drafter, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(dr.d.?, @max(bytes, 16));
        errdefer m.free();
        try m.fill8(0, null);
        try dr.mem.append(dr.gpa, m);
        return m.ptr;
    }

    fn upload(dr: *Drafter, bytes: []const u8) !u64 {
        var m = try cuda.DeviceBuffer.fromHost(dr.d.?, bytes);
        errdefer m.free();
        try dr.mem.append(dr.gpa, m);
        return m.ptr;
    }

    /// The tap layout: column blocks `cols` (this drafter's layer order) of a taps buffer `slots` blocks wide.
    pub fn setTaps(dr: *Drafter, cols: []const usize, slots: usize) !void {
        if (cols.len != dr.sp.ntaps) return error.Invalid;
        @memcpy(dr.cols[0..cols.len], cols);
        dr.slots = slots;
        dr.identity = cols.len == slots;
        for (cols, 0..) |c, j| dr.identity = dr.identity and c == j;
    }

    // ------------------------------------------------------------------------------------------- buffers ---

    fn allocBufs(dr: *Drafter, k: *const kern.Kernels, s: cuda.Stream) !void {
        const sp = dr.sp;
        const R = sp.rows();
        const D = sp.D;
        const hr = sp.headRows();
        const kmax = @max(@max(sp.ntaps * D, D), @max(sp.I, sp.H * sp.hd));
        dr.b.x = try dr.zeros(R * D * 2);
        dr.b.normed = try dr.zeros(R * D * 2);
        dr.b.xs = try dr.zeros(R * (kmax / 64) * 4);
        dr.b.t0 = try dr.zeros(R * @max(D, sp.qkvN()) * 2);
        dr.b.q = try dr.zeros(sp.H * R * sp.hd * 2);
        dr.b.k = try dr.zeros(sp.KV * R * sp.hd * 2);
        dr.b.v = try dr.zeros(sp.KV * R * sp.hd * 2);
        dr.b.att = try dr.zeros(R * sp.H * sp.hd * 2);
        dr.b.part = try dr.zeros(std.mem.alignForward(usize, R * D, 4) * 4);
        dr.b.gath = try dr.zeros(sp.world * std.mem.alignForward(usize, R * D, 4) * 4);
        dr.b.gu = try dr.zeros(R * 2 * sp.I * 2);
        dr.b.act = try dr.zeros(R * sp.I * 2);
        dr.b.axs = try dr.zeros(R * (sp.I / 64) * 4);
        if (sp.kind == .dflash) {
            dr.b.rsum = try dr.zeros(R * D * 2);
            dr.b.dyn = try dr.zeros(R * 4 * (D / sp.gs) * 2);
            dr.b.conv = try dr.zeros(R * D * 2);
        }
        dr.b.cos = try dr.zeros(R * (sp.hd / 2) * 4);
        dr.b.sin = try dr.zeros(R * (sp.hd / 2) * 4);
        dr.b.logits = try dr.zeros(hr * sp.V * 4);
        const blocks = (sp.V + kern.topk_chunk - 1) / kern.topk_chunk;
        dr.b.tsel = .{ .hist = try dr.zeros(kern.TopkScratch.histBytes(hr)), .state = try dr.zeros(hr * 16), .cnt = try dr.zeros(hr * blocks * 8), .rows = hr, .blocks = blocks };
        dr.b.cand_cols = try dr.zeros(hr * sp.top_k * 4);
        dr.b.send = try dr.zeros(sp.sendWords() * 4);
        dr.b.cand = try dr.zeros((sp.world * sp.sendWords() + hr * sp.extra()) * 4);
        {
            const ids = try dr.gpa.alloc(i32, sp.block);
            defer dr.gpa.free(ids);
            @memset(ids, @intCast(sp.mask_id));
            var m = try cuda.DeviceBuffer.fromHost(dr.d.?, std.mem.sliceAsBytes(ids));
            errdefer m.free();
            try dr.mem.append(dr.gpa, m);
            dr.b.ids = m.ptr;
        }
        dr.b.pos = try dr.zeros(8);
        dr.b.inv = try dr.zeros((sp.hd / 2) * 4);
        try k.invFreq(s, dr.b.inv, sp.hd, sp.theta); // 1 / theta ** (arange(hd / 2) * 2 / hd), torch's float32 steps
        dr.b.tap_in = try dr.zeros(tap_rows * sp.ntaps * D * 2);
        dr.kc = try dr.gpa.alloc(u64, sp.layers);
        dr.vc = try dr.gpa.alloc(u64, sp.layers);
        for (0..sp.layers) |i| {
            dr.kc[i] = try dr.zeros(sp.KV * ring_rows * sp.hd * 2);
            dr.vc[i] = try dr.zeros(sp.KV * ring_rows * sp.hd * 2);
        }
        dr.pin = try cuda.HostBuffer.alloc(dr.d.?, (sp.world * sp.sendWords() + hr * sp.extra()) * 4);
        dr.tokens = try dr.gpa.alloc(i64, hr * sp.top_k);
        dr.values = try dr.gpa.alloc(f64, hr * sp.top_k);
        dr.extra = try dr.gpa.alloc(f64, hr * sp.extra());
        dr.scratch = try dr.gpa.alloc(smp.Cand, sp.world * sp.top_k);
        dr.walk = try host.WalkScratch.init(dr.gpa, sp.top_k, @max(dr.chain.rank, sp.rsel));
    }

    /// A probe drafter for coverage: the shapes of `sp`, every pointer a 16-aligned fake address (no memory).
    pub fn probe(gpa: std.mem.Allocator, sp: Spec) !*Drafter {
        const A: u64 = 1 << 20;
        const dr = try gpa.create(Drafter);
        errdefer gpa.destroy(dr);
        dr.* = .{ .sp = sp, .gpa = gpa };
        dr.layers = try gpa.alloc(Layer, sp.layers);
        const D = sp.D;
        const g4 = if (sp.gs > 0) 4 * (D / sp.gs) else 64;
        for (dr.layers) |*L| L.* = .{ .in_norm = A, .post_norm = A, .q_norm = A, .k_norm = A, .qkv = fake(sp.qkvN(), D), .kv = fake(sp.kvN(), D), .o = fake(D, sp.H * sp.hd), .gu = fake(2 * sp.I, D), .down = fake(D, sp.I), .a_base = A, .a_kp = fake(g4, D), .m_base = A, .m_kp = fake(g4, D) };
        dr.kc = try gpa.alloc(u64, sp.layers);
        dr.vc = try gpa.alloc(u64, sp.layers);
        @memset(dr.kc, A);
        @memset(dr.vc, A);
        dr.fc = fake(D, sp.ntaps * D);
        dr.head = fake(sp.V, D);
        dr.hidden_norm = A;
        dr.norm = A;
        dr.embed = A;
        dr.hproj = A;
        dr.conf_h = A;
        const binfo = @typeInfo(Bufs).@"struct";
        inline for (binfo.field_names, binfo.field_types) |fname, FT| {
            if (FT == u64) @field(dr.b, fname) = A;
        }
        dr.b.tsel = .{ .hist = A, .state = A, .cnt = A, .rows = sp.headRows(), .blocks = (sp.V + kern.topk_chunk - 1) / kern.topk_chunk };
        return dr;
    }

    /// A probe's 4-bit matrix [n, k] at a fake address (its K picks the group sums' variant).
    fn fake(n: usize, k: usize) Q4 {
        const A: u64 = 1 << 20;
        return .{ .w = A, .scales = A, .biases = A, .n = n, .npad = (n + 127) / 128 * 128, .k = k, .sk = W.q4SplitK(n, k) };
    }

    // -------------------------------------------------------------------------------------------- pieces ---

    /// Drafter._norm: rmsnorm with the output's group sums (into b.xs).
    fn rmsNorm(dr: *const Drafter, dv: Dev, x: u64, w: u64, out: u64, rows: usize) !void {
        const D = dr.sp.D;
        try dv.t.rmsnormSums(x, D, w, out, D, dr.b.xs, rows, D, dr.sp.eps);
    }

    /// glm5_next qmm.matmul(x, q, xs): xs null -> group_sums(x) first (into b.xs).
    fn mm(dr: *const Drafter, dv: Dev, x: u64, xs: ?u64, q: Q4, out: u64, rows: usize, f32_out: bool) !void {
        const sums = xs orelse blk: {
            try dv.t.groupSums(x, q.k, dr.b.xs, rows, q.k);
            break :blk dr.b.xs;
        };
        if (dv.o) |o| try o.q4Mm(x, sums, q, out, rows, f32_out);
    }

    /// Drafter._prep: qkv rows (stride `stride`) -> q [heads, rows, hd] (heads 0: none), k / v [KV, rows, hd].
    fn prepRows(dr: *const Drafter, dv: Dev, qkv: u64, stride: usize, L: *const Layer, heads: usize, rows: usize) !void {
        const sp = dr.sp;
        const qo = if (heads > 0) dr.b.q else qkv;
        try dv.t.prep(qkv, L.q_norm, L.k_norm, dr.b.cos, dr.b.sin, qo, dr.b.k, dr.b.v, rows, stride, sp.eps, heads, sp.KV, sp.hd / 2);
    }

    /// kc[i] / vc[i].index_copy_(1, (pos + arange(rows)) % RING, k / v).
    fn scatter(dr: *const Drafter, dv: Dev, i: usize, rows: usize) !void {
        const o = dv.o orelse return;
        const sp = dr.sp;
        try o.draftScatter(dr.b.k, dr.kc[i], dr.b.pos, sp.KV, rows, sp.hd, ring_rows);
        try o.draftScatter(dr.b.v, dr.vc[i], dr.b.pos, sp.KV, rows, sp.hd, ring_rows);
    }

    /// Drafter._rotary: cos / sin of positions pos .. pos + rows - 1.
    fn rotary(dr: *const Drafter, dv: Dev, rows: usize) !void {
        const o = dv.o orelse return;
        try o.draftRotary(dr.b.pos, dr.b.inv, dr.sp.hd / 2, dr.b.cos, dr.b.sin, rows);
    }

    /// Drafter._row's partials: x [rows, K] @ q -> fp32 [rows, D] gathered in rank order (fused.small_gather);
    /// returns the slots' stride (words) and their count.
    fn rowGather(dr: *const Drafter, dv: Dev, x: u64, xs: ?u64, q: Q4, rows: usize) !struct { stride: usize, ranks: usize } {
        try dr.mm(dv, x, xs, q, dr.b.part, rows, true);
        const n = rows * dr.sp.D;
        const f = dv.f orelse return .{ .stride = n, .ranks = dr.sp.world };
        const stride = try f.smallGather(dr.b.part, dr.b.gath, n);
        const ranks = if (f.w.world > 1 and (f.comm != null or f.fast != null)) f.w.world else 1;
        return .{ .stride = stride, .ranks = ranks };
    }

    /// DSpark: x += row(x_in @ q) (bf16(x + bf16(the rank-order sum)), glue.residual_add's bits).
    fn rowResidual(dr: *const Drafter, dv: Dev, x_in: u64, xs: ?u64, q: Q4, rows: usize) !void {
        const g = try dr.rowGather(dv, x_in, xs, q, rows);
        try dv.t.residualAddStride(dr.b.x, dr.b.gath, rows, dr.sp.D, g.ranks, g.stride);
    }

    /// DFlash2: rsum = bf16(the rank-order sum of x_in @ q) alone.
    fn rowSum(dr: *const Drafter, dv: Dev, x_in: u64, xs: ?u64, q: Q4, rows: usize) !void {
        const g = try dr.rowGather(dv, x_in, xs, q, rows);
        const o = dv.o orelse return;
        try o.draftRowsum(dr.b.gath, g.stride, g.ranks, rows * dr.sp.D, dr.b.rsum);
    }

    // ------------------------------------------------------------------------------------------- layers ---

    /// SparkDrafter._layer over b.x [rows, D] (rows = the block).
    fn sparkLayer(dr: *const Drafter, dv: Dev, i: usize, rows: usize) !void {
        const sp = dr.sp;
        const L = &dr.layers[i];
        try dr.rmsNorm(dv, dr.b.x, L.in_norm, dr.b.normed, rows);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.qkv, dr.b.t0, rows, false);
        try dr.prepRows(dv, dr.b.t0, sp.qkvN(), L, sp.H, rows);
        try dr.scatter(dv, i, rows);
        try dv.t.blockAttn(dr.b.q, dr.kc[i], dr.vc[i], dr.b.att, dr.b.pos, sp.window, sp.scale(), rows, sp.H, sp.hd, ring_rows, sp.causal);
        try dr.rowResidual(dv, dr.b.att, null, L.o, rows);
        try dr.rmsNorm(dv, dr.b.x, L.post_norm, dr.b.normed, rows);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.gu, dr.b.gu, rows, false);
        try dv.t.swigluSums(dr.b.gu, dr.b.act, dr.b.axs, no_limit, rows, sp.I);
        try dr.rowResidual(dv, dr.b.act, dr.b.axs, L.down, rows);
    }

    /// GlmDrafter._layer over b.x [rows, D].
    fn flashLayer(dr: *const Drafter, dv: Dev, i: usize, rows: usize) !void {
        const sp = dr.sp;
        const L = &dr.layers[i];
        const D = sp.D;
        try dr.rmsNorm(dv, dr.b.x, L.in_norm, dr.b.normed, rows);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.a_kp, dr.b.dyn, rows, false);
        try dv.t.dconv(dr.b.normed, dr.b.dyn, L.a_base, dr.b.normed, dr.b.conv, rows, D, sp.gs, 0, false);
        try dr.mm(dv, dr.b.conv, null, L.qkv, dr.b.t0, rows, false);
        try dr.prepRows(dv, dr.b.t0, sp.qkvN(), L, sp.H, rows);
        try dr.scatter(dv, i, rows);
        try dv.t.dattnRing(dr.b.q, dr.kc[i], dr.vc[i], dr.b.att, dr.b.pos, sp.window, sp.scale(), rows, sp.H / sp.KV, sp.H, sp.hd, ring_rows, sp.KV, sp.causal);
        try dr.rowSum(dv, dr.b.att, null, L.o, rows);
        try dv.t.dconv(dr.b.rsum, dr.b.dyn, L.a_base, dr.b.x, dr.b.x, rows, D, sp.gs, 1, true);
        try dr.rmsNorm(dv, dr.b.x, L.post_norm, dr.b.normed, rows);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.m_kp, dr.b.dyn, rows, false);
        try dv.t.dconv(dr.b.normed, dr.b.dyn, L.m_base, dr.b.normed, dr.b.conv, rows, D, sp.gs, 0, false);
        try dr.mm(dv, dr.b.conv, null, L.gu, dr.b.gu, rows, false);
        try dv.t.swigluSums(dr.b.gu, dr.b.act, dr.b.axs, no_limit, rows, sp.I);
        try dr.rowSum(dv, dr.b.act, dr.b.axs, L.down, rows);
        try dv.t.dconv(dr.b.rsum, dr.b.dyn, L.m_base, dr.b.x, dr.b.x, rows, D, sp.gs, 1, true);
    }

    // ------------------------------------------------------------------------------------------- passes ---

    /// Drafter._taps_compute(n): b.tap_in[:n] -> the context's K / V rows at pos .. pos + n - 1; pos += n. No
    /// collective: each rank computes its own heads' rows.
    pub fn tapsCompute(dr: *const Drafter, dv: Dev, n: usize) !void {
        const sp = dr.sp;
        if (n == 0 or n > tap_rows) return error.Invalid;
        try dr.mm(dv, dr.b.tap_in, null, dr.fc, dr.b.t0, n, false);
        try dr.rmsNorm(dv, dr.b.t0, dr.hidden_norm, dr.b.normed, n); // ctx (its sums unused: _mm(ctx) sums again)
        try dr.rotary(dv, n);
        for (dr.layers, 0..) |*L, i| {
            try dr.mm(dv, dr.b.normed, null, L.kv, dr.b.t0, n, false);
            try dr.prepRows(dv, dr.b.t0, sp.kvN(), L, 0, n);
            try dr.scatter(dv, i, n);
        }
        if (dv.o) |o| try o.draftPosAdd(dr.b.pos, n);
    }

    /// _block_compute: [pending, mask x (block - 1)] at the committed length -> each rank's top_k of its head share for
    /// every head row, gathered into b.cand [ranks, stride], and the rows' extra words behind them. Static buffers
    /// only (the block graph replays it); the block's ring rows sit past the context (the next update overwrites them).
    pub fn blockCompute(dr: *Drafter, dv: Dev) !void {
        const sp = dr.sp;
        const D = sp.D;
        const n = sp.block;
        const hr = sp.headRows();
        try dv.t.embedB16(dr.b.ids, dr.embed, dr.b.x, n, D);
        try dr.rotary(dv, n);
        for (0..sp.layers) |i| {
            if (sp.kind == .dspark) try dr.sparkLayer(dv, i, n) else try dr.flashLayer(dv, i, n);
        }
        const h_in = if (sp.kind == .dspark) dr.b.x else dr.b.x + D * 2; // DFlash2: x[1:]
        try dr.rmsNorm(dv, h_in, dr.norm, dr.b.normed, hr);
        try dr.mm(dv, dr.b.normed, dr.b.xs, dr.head, dr.b.logits, hr, true); // headq.logits (q4, fp32)
        const extra_at = dr.b.cand + sp.world * sp.sendWords() * 4;
        if (dv.o) |oc| {
            try oc.topkColumns(dr.b.logits, sp.V, sp.V, hr, sp.top_k, true, dr.b.tsel, dr.b.cand_cols, sp.top_k);
            try oc.cands(dr.b.logits, sp.V, dr.b.cand_cols, sp.top_k, sp.vocab_off, dr.b.send, hr);
        }
        const words = hr * 2 * sp.top_k;
        if (dv.f) |f| {
            dr.stride = try f.smallGather(dr.b.send, dr.b.cand, words);
            dr.ranks = if (f.w.world > 1 and (f.comm != null or f.fast != null)) f.w.world else 1;
        } else {
            dr.stride = words;
            dr.ranks = sp.world;
        }
        const o = dv.o orelse return;
        if (sp.kind == .dspark) {
            if (dr.conf_h != 0) {
                try o.draftDot(dr.b.normed, dr.conf_h, D, extra_at, hr);
            } else {
                try o.fill32(extra_at, 0, hr);
            }
        } else {
            try o.draftLinear(dr.b.normed, dr.hproj, D, sp.rsel, extra_at, hr);
        }
    }

    /// Copies rows [0, n) of a taps buffer (`src`, row pitch `pitch` bytes) into b.tap_in in this drafter's column
    /// order (dspark_host.tap_runs: one copy a contiguous run).
    pub fn loadTaps(dr: *const Drafter, o: kern.Ops, src: u64, pitch: usize, n: usize) !void {
        const D = dr.sp.D;
        const width = dr.sp.ntaps * D * 2;
        if (dr.identity) {
            if (pitch == width) return o.copy(dr.b.tap_in, src, n * width);
            return o.copyRows(dr.b.tap_in, width, src, pitch, width, n);
        }
        var runs: [host.max_taps][3]usize = undefined;
        for (host.tapRuns(dr.cols[0..dr.sp.ntaps], &runs)) |run| {
            try o.copyRows(dr.b.tap_in + run[0] * D * 2, width, src + run[1] * D * 2, pitch, run[2] * D * 2, n);
        }
    }

    /// The pending token into ids[0] (on the stream, before the block pass).
    pub fn setPending(dr: *const Drafter, o: kern.Ops, pending: u32) !void {
        try o.fill32(dr.b.ids, pending, 1);
    }

    /// A fresh context at position `at` (0: a new prompt; > 0: a resumed one - the ring's rows below `at` are not
    /// restored, which only costs acceptance until the window refills).
    pub fn resetAt(dr: *Drafter, o: kern.Ops, at: usize) !void {
        dr.context_end = at;
        try o.fill32(dr.b.pos, @truncate(at), 1);
        try o.fill32(dr.b.pos + 4, @truncate(at >> 32), 1);
    }

    /// `n` committed rows that never need computing (older than any block will read): the position moves on.
    pub fn skip(dr: *Drafter, o: kern.Ops, n: usize) !void {
        if (n == 0) return;
        try o.draftPosAdd(dr.b.pos, n);
        dr.context_end += n;
    }

    /// The block pass's words (b.cand) read to the host (one pinned copy, one sync) and merged: tokens / values
    /// [depth, top_k], extra [depth, extra].
    pub fn readCandidates(dr: *Drafter, o: kern.Ops, depth: usize) !void {
        const sp = dr.sp;
        const pin = dr.pin orelse return error.Invalid;
        const total = sp.world * sp.sendWords() + sp.headRows() * sp.extra();
        const host_f: [*]f32 = @ptrCast(@alignCast(pin.bytes.ptr));
        const all = host_f[0..total];
        try o.download(std.mem.sliceAsBytes(all), dr.b.cand);
        try o.s.synchronize();
        host.mergeCandidates(all[0 .. dr.ranks * dr.stride], dr.ranks, dr.stride, depth, sp.top_k, dr.tokens, dr.values, dr.scratch);
        const ex = all[sp.world * sp.sendWords() ..];
        for (0..depth * sp.extra()) |i| dr.extra[i] = ex[i];
    }

    /// The host stage on the merged candidates: up to `depth` drafts after `pending` (slot 0's token at `first`).
    pub fn walkDrafts(dr: *Drafter, depth: usize, pending: u32, first: u64, sampling: ?smp.Sampling, out: []u32) []u32 {
        return dr.walkConf(depth, pending, first, sampling, null, out);
    }

    /// walkDrafts with the confidence floor `conf` instead of the settings' (null: the settings'; Phase 4c: the
    /// concurrent rounds' draft cut raises it, multi.DRAFT_CUT_DFLASH).
    pub fn walkConf(dr: *Drafter, depth: usize, pending: u32, first: u64, sampling: ?smp.Sampling, conf: ?f64, out: []u32) []u32 {
        const sp = dr.sp;
        const ws = &dr.walk.?;
        dr.proposals += 1;
        if (sp.kind == .dspark) {
            const s = dr.dset;
            const rms: ?[]const f64 = if (dr.n_round_ms > 0) dr.round_ms[0..dr.n_round_ms] else null;
            return dr.chain.walk(dr.tokens, dr.values, dr.extra, depth, sp.top_k, pending, first, .{ .sampling = sampling, .policy = s.policy, .confidence = conf orelse s.confidence, .round_ms = rms, .noise = s.noise, .filtered = s.filter, .row_cost = s.row_cost }, ws, out);
        }
        return host.selectorChain(dr.pred, dr.succ, sp.rsel, dr.tokens, dr.values, dr.extra, depth, sp.top_k, pending, first, sampling, conf orelse dr.fset.confidence, ws, out);
    }

    // ------------------------------------------------------------------- Phase 4c: concurrent streams ---

    /// Phase 4c (cuda_mdraft.zig, dflash.MultiDrafter): a view of this drafter for concurrent streams - its weights,
    /// tap layout and settings, with static buffers of its own for passes of up to `rows` rows (every live stream's
    /// block side by side, or a stacked context update), `head_rows` head rows and `send_words` words a rank sends;
    /// `kc` / `vc` (each layer's pooled rings, owned by the caller) as its rings. The host stage stays the one-stream
    /// drafter's (walkConf on it). Never `deinit` a view: `freeView`.
    pub fn initView(base: *const Drafter, rows: usize, head_rows: usize, send_words: usize, kc: []u64, vc: []u64) !Drafter {
        var v: Drafter = base.*;
        v.mem = .empty;
        v.pin = null;
        v.b = .{ .inv = base.b.inv };
        v.kc = kc;
        v.vc = vc;
        errdefer v.freeView();
        try v.allocView(rows, head_rows, send_words);
        return v;
    }

    /// A view's own device buffers and pinned staging (not the weights, rings or host tables: the base's / caller's).
    pub fn freeView(v: *Drafter) void {
        for (v.mem.items) |*m| m.free();
        v.mem.deinit(v.gpa);
        v.mem = .empty;
        if (v.pin) |*p| p.free();
        v.pin = null;
    }

    fn allocView(v: *Drafter, rows: usize, head_rows: usize, send_words: usize) !void {
        const sp = v.sp;
        const R = rows;
        const D = sp.D;
        const hr = head_rows;
        if (R == 0 or R > tap_rows or hr == 0) return error.Invalid;
        const kmax = @max(@max(sp.ntaps * D, D), @max(sp.I, sp.H * sp.hd));
        v.b.x = try v.zeros(R * D * 2);
        v.b.normed = try v.zeros(R * D * 2);
        v.b.xs = try v.zeros(R * (kmax / 64) * 4);
        v.b.t0 = try v.zeros(R * @max(D, sp.qkvN()) * 2);
        v.b.q = try v.zeros(sp.H * R * sp.hd * 2);
        v.b.k = try v.zeros(sp.KV * R * sp.hd * 2);
        v.b.v = try v.zeros(sp.KV * R * sp.hd * 2);
        v.b.att = try v.zeros(R * sp.H * sp.hd * 2);
        v.b.part = try v.zeros(std.mem.alignForward(usize, R * D, 4) * 4);
        v.b.gath = try v.zeros(sp.world * std.mem.alignForward(usize, R * D, 4) * 4);
        v.b.gu = try v.zeros(R * 2 * sp.I * 2);
        v.b.act = try v.zeros(R * sp.I * 2);
        v.b.axs = try v.zeros(R * (sp.I / 64) * 4);
        v.b.conv = try v.zeros(R * D * 2); // DFlash2's convolutions; both: a block pass's head rows side by side
        if (sp.kind == .dflash) {
            v.b.rsum = try v.zeros(R * D * 2);
            v.b.dyn = try v.zeros(R * 4 * (D / sp.gs) * 2);
        }
        v.b.cos = try v.zeros(R * (sp.hd / 2) * 4);
        v.b.sin = try v.zeros(R * (sp.hd / 2) * 4);
        v.b.logits = try v.zeros(hr * sp.V * 4);
        const blocks = (sp.V + kern.topk_chunk - 1) / kern.topk_chunk;
        v.b.tsel = .{ .hist = try v.zeros(kern.TopkScratch.histBytes(hr)), .state = try v.zeros(hr * 16), .cnt = try v.zeros(hr * blocks * 8), .rows = hr, .blocks = blocks };
        v.b.cand_cols = try v.zeros(hr * sp.top_k * 4);
        v.b.send = try v.zeros(send_words * 4);
        v.b.cand = try v.zeros((sp.world * send_words + hr * sp.extra()) * 4);
        {
            const ids = try v.gpa.alloc(i32, R);
            defer v.gpa.free(ids);
            @memset(ids, @intCast(sp.mask_id));
            var m = try cuda.DeviceBuffer.fromHost(v.d.?, std.mem.sliceAsBytes(ids));
            errdefer m.free();
            try v.mem.append(v.gpa, m);
            v.b.ids = m.ptr;
        }
        v.b.tap_in = try v.zeros(R * sp.ntaps * D * 2);
        v.pin = try cuda.HostBuffer.alloc(v.d.?, (sp.world * send_words + hr * sp.extra()) * 4);
    }

    /// Where a concurrent pass's keys and values go: the pooled rings' layout (rows of head_dim).
    pub const PoolLayout = struct {
        head_stride: usize, // ring rows between two KV heads (DFlash2: slots x RING; DSpark: RING)
        slot_stride: usize, // ring rows between two stream slots (DFlash2: RING; DSpark: KV heads x RING)
    };

    /// A concurrent block pass's blocks (cuda_mdraft.zig's tables on the device).
    pub const Segs = struct {
        S: usize, // blocks side by side
        St: usize, // META's pitch (the decoder's stream slots)
        meta: u64, // int64 [3 St]: each block's pending | slot | committed length (MultiDrafter.b_meta)
        spos: u64, // int64 each block's committed length again, 16 bytes apart (DSpark: _block_attn_kernel's POS)
        rpos: u64, // int64 [S block]: each row's position
        rslot: u64, // int64 [S block]: each row's slot
        slots: [max_segs]usize = @splat(0), // each block's slot (DSpark: its ring, baked into the graph)
        pl: PoolLayout,
        pool_rows: usize, // DFlash2: _dattn_seg's POOL (ring rows a KV head)
    };

    /// kc[i] / vc[i].index_copy_(1, idx, k / v) into the pooled rings: row r to slot[r]'s ring at pos[r] % RING.
    fn scatterRows(dr: *const Drafter, dv: Dev, i: usize, rows: usize, pl: PoolLayout, slot: u64, pos: u64) !void {
        const o = dv.o orelse return;
        const sp = dr.sp;
        try o.draftScatterRows(dr.b.k, dr.kc[i], slot, pos, sp.KV, rows, sp.hd, pl.head_stride, pl.slot_stride, ring_rows);
        try o.draftScatterRows(dr.b.v, dr.vc[i], slot, pos, sp.KV, rows, sp.hd, pl.head_stride, pl.slot_stride, ring_rows);
    }

    /// MultiDrafter._taps_compute(T): b.tap_in[:T] (several streams' committed rows stacked) -> each row's context K /
    /// V into its slot's ring at its position (int64 tables `slot` / `pos`; a negative slot: the padding of a captured
    /// row bucket, nothing written). tapsCompute's arithmetic row by row (the passes are row-invariant).
    pub fn tapsComputeRows(dr: *const Drafter, dv: Dev, T: usize, pl: PoolLayout, slot: u64, pos: u64) !void {
        const sp = dr.sp;
        if (T == 0 or T > tap_rows) return error.Invalid;
        try dr.mm(dv, dr.b.tap_in, null, dr.fc, dr.b.t0, T, false);
        try dr.rmsNorm(dv, dr.b.t0, dr.hidden_norm, dr.b.normed, T); // ctx (its sums unused: _mm(ctx) sums again)
        if (dv.o) |o| try o.draftRotaryRows(pos, dr.b.inv, sp.hd / 2, dr.b.cos, dr.b.sin, T);
        for (dr.layers, 0..) |*L, i| {
            try dr.mm(dv, dr.b.normed, null, L.kv, dr.b.t0, T, false);
            try dr.prepRows(dv, dr.b.t0, sp.kvN(), L, 0, T);
            try dr.scatterRows(dv, i, T, pl, slot, pos);
        }
    }

    /// MultiDrafter._layer for DFlash2: S blocks side by side through one layer (_dconv_seg, the pooled rings,
    /// _dattn_seg reading each block's slot and committed length from META).
    fn flashLayerSegs(dr: *const Drafter, dv: Dev, i: usize, sg: Segs) !void {
        const sp = dr.sp;
        const L = &dr.layers[i];
        const D = sp.D;
        const n = sp.block;
        const R = sg.S * n;
        try dr.rmsNorm(dv, dr.b.x, L.in_norm, dr.b.normed, R);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.a_kp, dr.b.dyn, R, false);
        try dv.t.dconvSeg(dr.b.normed, dr.b.dyn, L.a_base, dr.b.normed, dr.b.conv, R, D, sp.gs, 0, false, n);
        try dr.mm(dv, dr.b.conv, null, L.qkv, dr.b.t0, R, false);
        try dr.prepRows(dv, dr.b.t0, sp.qkvN(), L, sp.H, R);
        try dr.scatterRows(dv, i, R, sg.pl, sg.rslot, sg.rpos);
        try dv.t.dattnSeg(dr.b.q, dr.kc[i], dr.vc[i], dr.b.att, sg.meta, sg.St, sp.window, sp.scale(), R, sg.pool_rows, n, sp.H / sp.KV, sp.H, sp.hd, ring_rows, sp.KV, sg.S, sp.causal);
        try dr.rowSum(dv, dr.b.att, null, L.o, R);
        try dv.t.dconvSeg(dr.b.rsum, dr.b.dyn, L.a_base, dr.b.x, dr.b.x, R, D, sp.gs, 1, true, n);
        try dr.rmsNorm(dv, dr.b.x, L.post_norm, dr.b.normed, R);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.m_kp, dr.b.dyn, R, false);
        try dv.t.dconvSeg(dr.b.normed, dr.b.dyn, L.m_base, dr.b.normed, dr.b.conv, R, D, sp.gs, 0, false, n);
        try dr.mm(dv, dr.b.conv, null, L.gu, dr.b.gu, R, false);
        try dv.t.swigluSums(dr.b.gu, dr.b.act, dr.b.axs, no_limit, R, sp.I);
        try dr.rowSum(dv, dr.b.act, dr.b.axs, L.down, R);
        try dv.t.dconvSeg(dr.b.rsum, dr.b.dyn, L.m_base, dr.b.x, dr.b.x, R, D, sp.gs, 1, true, n);
    }

    /// SparkDrafter._layer for S blocks side by side: the row-wise steps (norms, the 4-bit matmuls, the row-parallel
    /// sums, the MLP) over every block at once; each block's attention part alone, as the one-stream pass runs it -
    /// _prep_kernel on its rows, its keys / values into its slot's ring (slot-major pool), _block_attn_kernel over that
    /// ring at its committed length (`spos`).
    fn sparkLayerSegs(dr: *const Drafter, dv: Dev, i: usize, sg: Segs) !void {
        const sp = dr.sp;
        const L = &dr.layers[i];
        const n = sp.block;
        const R = sg.S * n;
        const half = sp.hd / 2;
        const qkv_n = sp.qkvN();
        try dr.rmsNorm(dv, dr.b.x, L.in_norm, dr.b.normed, R);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.qkv, dr.b.t0, R, false);
        for (0..sg.S) |s| {
            const r0 = s * n;
            const q = dr.b.q + s * sp.H * n * sp.hd * 2;
            const k = dr.b.k + s * sp.KV * n * sp.hd * 2;
            const v = dr.b.v + s * sp.KV * n * sp.hd * 2;
            const pos = sg.spos + s * 16;
            const ring_at = sg.slots[s] * sg.pl.slot_stride * sp.hd * 2;
            try dv.t.prep(dr.b.t0 + r0 * qkv_n * 2, L.q_norm, L.k_norm, dr.b.cos + r0 * half * 4, dr.b.sin + r0 * half * 4, q, k, v, n, qkv_n, sp.eps, sp.H, sp.KV, half);
            if (dv.o) |o| {
                try o.draftScatter(k, dr.kc[i] + ring_at, pos, sp.KV, n, sp.hd, ring_rows);
                try o.draftScatter(v, dr.vc[i] + ring_at, pos, sp.KV, n, sp.hd, ring_rows);
            }
            try dv.t.blockAttn(q, dr.kc[i] + ring_at, dr.vc[i] + ring_at, dr.b.att + r0 * sp.H * sp.hd * 2, pos, sp.window, sp.scale(), n, sp.H, sp.hd, ring_rows, sp.causal);
        }
        try dr.rowResidual(dv, dr.b.att, null, L.o, R);
        try dr.rmsNorm(dv, dr.b.x, L.post_norm, dr.b.normed, R);
        try dr.mm(dv, dr.b.normed, dr.b.xs, L.gu, dr.b.gu, R, false);
        try dv.t.swigluSums(dr.b.gu, dr.b.act, dr.b.axs, no_limit, R, sp.I);
        try dr.rowResidual(dv, dr.b.act, dr.b.axs, L.down, R);
    }

    /// MultiDrafter._block_compute(S): S blocks [pending, mask x (block - 1)] side by side (b.ids, the caller's) at
    /// their streams' committed lengths -> every head row's top_k of each rank's head share, gathered into b.cand
    /// [ranks, stride] (block s's head rows at s x headRows), the rows' extra words after `send_words` words a rank.
    /// Returns the gather's stride (words a rank).
    pub fn blockComputeSegs(dr: *Drafter, dv: Dev, sg: Segs, send_words: usize) !usize {
        const sp = dr.sp;
        const D = sp.D;
        const n = sp.block;
        const R = sg.S * n;
        const hr = sp.headRows();
        const HR = sg.S * hr;
        if (sg.S == 0 or sg.S > max_segs or R > tap_rows) return error.Invalid;
        try dv.t.embedB16(dr.b.ids, dr.embed, dr.b.x, R, D);
        if (dv.o) |oc| try oc.draftRotaryRows(sg.rpos, dr.b.inv, sp.hd / 2, dr.b.cos, dr.b.sin, R);
        for (0..sp.layers) |i| {
            if (sp.kind == .dspark) try dr.sparkLayerSegs(dv, i, sg) else try dr.flashLayerSegs(dv, i, sg);
        }
        var h_in = dr.b.x; // DSpark: every slot's row (sample_from_anchor)
        if (sp.kind == .dflash) { // x.view(S, n, D)[:, 1:]: each block's rows 1.. side by side
            if (dv.o) |oc| try oc.copyRows(dr.b.conv, hr * D * 2, dr.b.x + D * 2, n * D * 2, hr * D * 2, sg.S);
            h_in = dr.b.conv;
        }
        try dr.rmsNorm(dv, h_in, dr.norm, dr.b.normed, HR);
        try dr.mm(dv, dr.b.normed, dr.b.xs, dr.head, dr.b.logits, HR, true); // headq.logits (q4, fp32)
        if (dv.o) |oc| {
            try oc.topkColumns(dr.b.logits, sp.V, sp.V, HR, sp.top_k, true, dr.b.tsel, dr.b.cand_cols, sp.top_k);
            try oc.cands(dr.b.logits, sp.V, dr.b.cand_cols, sp.top_k, sp.vocab_off, dr.b.send, HR);
        }
        const words = HR * 2 * sp.top_k;
        var stride = words;
        if (dv.f) |f| {
            stride = try f.smallGather(dr.b.send, dr.b.cand, words);
            dr.ranks = if (f.w.world > 1 and (f.comm != null or f.fast != null)) f.w.world else 1;
        } else {
            dr.ranks = sp.world;
        }
        const extra_at = dr.b.cand + sp.world * send_words * 4;
        const o = dv.o orelse return stride;
        if (sp.kind == .dspark) {
            if (dr.conf_h != 0) {
                try o.draftDot(dr.b.normed, dr.conf_h, D, extra_at, HR);
            } else {
                try o.fill32(extra_at, 0, HR);
            }
        } else {
            try o.draftLinear(dr.b.normed, dr.hproj, D, sp.rsel, extra_at, HR);
        }
        return stride;
    }

    // -------------------------------------------------------------------------------------------- load ---

    /// The drafter of `dir` on this rank: DSpark (`kind` dspark: config `dc` with settings `ds`) or DFlash2.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const kern.Kernels, s: cuda.Stream, w: *const W.Weights, dir: []const u8, kind: Kind, ds: host.DSparkSettings, fs: host.DFlashSettings) !*Drafter {
        const cfg_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(cfg_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20));
        defer gpa.free(text);
        const c = w.config;
        const world = w.world;
        var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        const dr = try gpa.create(Drafter);
        dr.* = .{ .sp = undefined, .gpa = gpa, .d = d, .dset = ds, .fset = fs };
        errdefer dr.deinit();
        dr.embed = w.embed;
        dr.head = w.draft_lm orelse return error.NoDraftHead;
        switch (kind) {
            .dspark => {
                const dc = try host.DSparkConfig.parse(gpa, text);
                try dc.check(world, c.hidden, c.vocab, c.layers);
                dr.sp = Spec.ofDSpark(&dc, &ds, world, w.vocab_part, w.vocab_off);
                try dr.loadSpark(&ck, &dc, w.rank);
            },
            .dflash => {
                const fc = try host.DFlashConfig.parse(gpa, text);
                try fc.check(world, c.hidden, c.layers);
                const hp = try ck.get("candidate_selector.hidden_projection.weight");
                if (hp.rank != 2 or hp.shape[1] != c.hidden) return error.UnexpectedTensor;
                dr.sp = Spec.ofDFlash(&fc, world, w.vocab_part, w.vocab_off, hp.shape[0]);
                try dr.loadFlash(&ck, w.rank);
            },
        }
        if (dr.sp.V % 4 != 0 or dr.sp.top_k > dr.sp.V) return error.Invalid;
        try dr.allocBufs(k, s);
        try s.synchronize();
        return dr;
    }

    fn rowsOf(dr: *Drafter, t: core.checkpoint.Tensor, r0: usize, r1: usize, c0: usize, c1: usize, out: []u16) !void {
        _ = dr;
        if (t.rank != 2 and !(t.rank == 1 and c0 == 0)) return error.UnexpectedTensor;
        const cols = if (t.rank == 2) t.shape[1] else t.shape[0];
        const n_cols = c1 - c0;
        if (out.len != (r1 - r0) * n_cols) return error.Invalid;
        for (r0..r1) |r| {
            const dst = out[(r - r0) * n_cols ..][0..n_cols];
            switch (t.dtype) {
                .bf16 => @memcpy(std.mem.sliceAsBytes(dst), t.bytes[(r * cols + c0) * 2 ..][0 .. n_cols * 2]),
                .f32 => for (dst, 0..) |*v, j| {
                    const at = (r * cols + c0 + j) * 4;
                    v.* = W.bf16Bits(@bitCast(std.mem.readInt(u32, t.bytes[at..][0..4], .little)));
                },
                else => return error.UnexpectedTensor,
            }
        }
    }

    /// A tensor (any 1-D / 2-D bf16 or fp32) as bf16 bits: rows [r0, r1), columns [c0, c1).
    fn slice(dr: *Drafter, ck: *core.checkpoint.Checkpoint, name: []const u8, r0: usize, r1: usize, c0: usize, c1: usize) ![]u16 {
        const t = try ck.get(name);
        const out = try dr.gpa.alloc(u16, (r1 - r0) * (c1 - c0));
        errdefer dr.gpa.free(out);
        try dr.rowsOf(t, r0, r1, c0, c1, out);
        return out;
    }

    fn whole(dr: *Drafter, ck: *core.checkpoint.Checkpoint, name: []const u8) ![]u16 {
        const t = try ck.get(name);
        return switch (t.rank) {
            1 => dr.slice(ck, name, 0, 1, 0, t.shape[0]),
            2 => dr.slice(ck, name, 0, t.shape[0], 0, t.shape[1]),
            3 => blk: { // [2, 2, D]: flattened rows
                const n = t.numel();
                const out = try dr.gpa.alloc(u16, n);
                errdefer dr.gpa.free(out);
                switch (t.dtype) {
                    .bf16 => @memcpy(std.mem.sliceAsBytes(out), t.bytes[0 .. n * 2]),
                    .f32 => for (out, 0..) |*v, j| {
                        v.* = W.bf16Bits(@bitCast(std.mem.readInt(u32, t.bytes[j * 4 ..][0..4], .little)));
                    },
                    else => return error.UnexpectedTensor,
                }
                break :blk out;
            },
            else => error.UnexpectedTensor,
        };
    }

    /// gpu(t): a bf16 vector / small tensor on the device.
    fn vec(dr: *Drafter, ck: *core.checkpoint.Checkpoint, name: []const u8) !u64 {
        const v = try dr.whole(ck, name);
        defer dr.gpa.free(v);
        return dr.upload(std.mem.sliceAsBytes(v));
    }

    /// quantize4 (qmm.quantize4's float32 steps) + shared qmm.pack on the host, then on the device.
    fn q4(dr: *Drafter, rows: []const u16, n: usize, k: usize) !Q4 {
        const hq = try quantizePack(dr.gpa, rows, n, k);
        defer hq.free(dr.gpa);
        return .{ .w = try dr.upload(std.mem.sliceAsBytes(hq.w)), .scales = try dr.upload(std.mem.sliceAsBytes(hq.s)), .biases = try dr.upload(std.mem.sliceAsBytes(hq.b)), .n = n, .npad = hq.npad, .k = k, .sk = W.q4SplitK(n, k) };
    }

    /// Rows [a, b) of a [N, K] tensor's ranks' cut, concatenated: the caller's parts in order.
    const Part = struct { name: []const u8, r0: usize, r1: usize, c0: usize = 0, c1: usize = 0 };

    fn q4Parts(dr: *Drafter, ck: *core.checkpoint.Checkpoint, parts: []const Part) !Q4 {
        var n: usize = 0;
        var k: usize = 0;
        for (parts) |pt| {
            const t = try ck.get(pt.name);
            if (t.rank != 2) return error.UnexpectedTensor;
            const kk = if (pt.c1 > pt.c0) pt.c1 - pt.c0 else t.shape[1];
            if (k != 0 and kk != k) return error.UnexpectedTensor;
            if (pt.r1 > t.shape[0]) return error.UnexpectedTensor;
            k = kk;
            n += pt.r1 - pt.r0;
        }
        if (k % 64 != 0) return error.UnexpectedTensor;
        const rows = try dr.gpa.alloc(u16, n * k);
        defer dr.gpa.free(rows);
        var at: usize = 0;
        for (parts) |pt| {
            const t = try ck.get(pt.name);
            const c0 = pt.c0;
            const c1 = if (pt.c1 > pt.c0) pt.c1 else t.shape[1];
            const m = pt.r1 - pt.r0;
            try dr.rowsOf(t, pt.r0, pt.r1, c0, c1, rows[at * k ..][0 .. m * k]);
            at += m;
        }
        return dr.q4(rows, n, k);
    }

    fn layerName(buf: []u8, i: usize, suffix: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, suffix });
    }

    /// SparkDrafter.__init__'s tensors on this rank (the q4 kind; host Markov tables and the confidence head).
    fn loadSpark(dr: *Drafter, ck: *core.checkpoint.Checkpoint, dc: *const host.DSparkConfig, rank: usize) !void {
        const sp = dr.sp;
        const D = sp.D;
        const hd = sp.hd;
        const H = sp.H;
        const q0 = rank * H;
        const m0 = rank * sp.I;
        {
            const fc = try ck.get("fc.weight");
            if (!fc.is(fc.dtype, &.{ D, sp.ntaps * D })) return error.UnexpectedTensor;
            dr.fc = try dr.q4Parts(ck, &.{.{ .name = "fc.weight", .r0 = 0, .r1 = D }});
        }
        dr.hidden_norm = try dr.vec(ck, "hidden_norm.weight");
        dr.norm = try dr.vec(ck, "norm.weight");
        dr.layers = try dr.gpa.alloc(Layer, sp.layers);
        @memset(dr.layers, .{});
        var nb: [8][96]u8 = undefined;
        for (dr.layers, 0..) |*L, i| {
            const qn = try layerName(&nb[0], i, "self_attn.q_proj.weight");
            const kn = try layerName(&nb[1], i, "self_attn.k_proj.weight");
            const vn = try layerName(&nb[2], i, "self_attn.v_proj.weight");
            const on = try layerName(&nb[3], i, "self_attn.o_proj.weight");
            const gn = try layerName(&nb[4], i, "mlp.gate_proj.weight");
            const un = try layerName(&nb[5], i, "mlp.up_proj.weight");
            const dn = try layerName(&nb[6], i, "mlp.down_proj.weight");
            L.qkv = try dr.q4Parts(ck, &.{ .{ .name = qn, .r0 = q0 * hd, .r1 = (q0 + H) * hd }, .{ .name = kn, .r0 = q0 * hd, .r1 = (q0 + H) * hd }, .{ .name = vn, .r0 = q0 * hd, .r1 = (q0 + H) * hd } });
            L.kv = try dr.q4Parts(ck, &.{ .{ .name = kn, .r0 = q0 * hd, .r1 = (q0 + H) * hd }, .{ .name = vn, .r0 = q0 * hd, .r1 = (q0 + H) * hd } });
            L.o = try dr.q4Parts(ck, &.{.{ .name = on, .r0 = 0, .r1 = D, .c0 = q0 * hd, .c1 = (q0 + H) * hd }});
            L.gu = try dr.q4Parts(ck, &.{ .{ .name = gn, .r0 = m0, .r1 = m0 + sp.I }, .{ .name = un, .r0 = m0, .r1 = m0 + sp.I } });
            L.down = try dr.q4Parts(ck, &.{.{ .name = dn, .r0 = 0, .r1 = D, .c0 = m0, .c1 = m0 + sp.I }});
            L.in_norm = try dr.vec(ck, try layerName(&nb[7], i, "input_layernorm.weight"));
            L.post_norm = try dr.vec(ck, try layerName(&nb[7], i, "post_attention_layernorm.weight"));
            L.q_norm = try dr.vec(ck, try layerName(&nb[7], i, "self_attn.q_norm.weight"));
            L.k_norm = try dr.vec(ck, try layerName(&nb[7], i, "self_attn.k_norm.weight"));
        }
        // the sequential stage's tables on the host, every rank alike (bf16 words; rows read per candidate)
        const rk = dc.markov_rank;
        if (rk > 0) {
            dr.w1 = try dr.slice(ck, "markov_head.markov_w1.weight", 0, dc.vocab, 0, rk);
            dr.w2 = try dr.slice(ck, "markov_head.markov_w2.weight", 0, dc.vocab, 0, rk);
        }
        var conf_b: f64 = 0;
        if (dc.confidence) {
            const pw = try ck.get("confidence_head.proj.weight");
            const width = D + (if (dc.confidence_markov) rk else 0);
            if (pw.rank != 2 or pw.shape[0] != 1 or pw.shape[1] != width) return error.UnexpectedTensor;
            const cw = try dr.gpa.alloc(f32, width);
            defer dr.gpa.free(cw);
            try floats(pw, cw);
            dr.conf_h = try dr.upload(std.mem.sliceAsBytes(cw[0..D]));
            if (dc.confidence_markov) {
                dr.conf_m = try dr.gpa.alloc(f64, rk);
                for (dr.conf_m, cw[D..]) |*m, v| m.* = v;
            }
            const pb = try ck.get("confidence_head.proj.bias");
            var bv: [1]f32 = undefined;
            try floats(pb, &bv);
            conf_b = bv[0];
        }
        dr.chain = .{ .w1 = if (rk > 0) dr.w1 else null, .w2 = if (rk > 0) dr.w2 else null, .rank = rk, .conf_m = if (dr.conf_m.len > 0) dr.conf_m else null, .conf_b = conf_b, .confident = dc.confidence };
    }

    /// dflash2.Drafter.__init__'s tensors on this rank (every matrix 4-bit; the selector's codebooks on the host).
    fn loadFlash(dr: *Drafter, ck: *core.checkpoint.Checkpoint, rank: usize) !void {
        const sp = dr.sp;
        const D = sp.D;
        const hd = sp.hd;
        const H = sp.H;
        const KV = sp.KV;
        dr.fc = try dr.q4Parts(ck, &.{.{ .name = "fc.weight", .r0 = 0, .r1 = D }});
        dr.hidden_norm = try dr.vec(ck, "hidden_norm.weight");
        dr.norm = try dr.vec(ck, "norm.weight");
        dr.hproj = try dr.vec(ck, "candidate_selector.hidden_projection.weight");
        const pt = try ck.get("candidate_selector.predecessor_codebook");
        const st = try ck.get("candidate_selector.successor_codebook");
        if (pt.rank != 2 or st.rank != 2 or pt.shape[1] != sp.rsel or st.shape[1] != sp.rsel or pt.shape[0] != st.shape[0]) return error.UnexpectedTensor;
        dr.pred = try dr.gpa.alloc(f32, pt.numel());
        dr.succ = try dr.gpa.alloc(f32, st.numel());
        try floats(pt, dr.pred);
        try floats(st, dr.succ);
        dr.layers = try dr.gpa.alloc(Layer, sp.layers);
        @memset(dr.layers, .{});
        var nb: [8][96]u8 = undefined;
        for (dr.layers, 0..) |*L, i| {
            const qn = try layerName(&nb[0], i, "self_attn.q_proj.weight");
            const kn = try layerName(&nb[1], i, "self_attn.k_proj.weight");
            const vn = try layerName(&nb[2], i, "self_attn.v_proj.weight");
            const on = try layerName(&nb[3], i, "self_attn.o_proj.weight");
            const gn = try layerName(&nb[4], i, "mlp.gate_proj.weight");
            const un = try layerName(&nb[5], i, "mlp.up_proj.weight");
            const dn = try layerName(&nb[6], i, "mlp.down_proj.weight");
            const r0 = rank * H * hd;
            const k0 = rank * KV * hd;
            const m0 = rank * sp.I;
            L.qkv = try dr.q4Parts(ck, &.{ .{ .name = qn, .r0 = r0, .r1 = r0 + H * hd }, .{ .name = kn, .r0 = k0, .r1 = k0 + KV * hd }, .{ .name = vn, .r0 = k0, .r1 = k0 + KV * hd } });
            L.kv = try dr.q4Parts(ck, &.{ .{ .name = kn, .r0 = k0, .r1 = k0 + KV * hd }, .{ .name = vn, .r0 = k0, .r1 = k0 + KV * hd } });
            L.o = try dr.q4Parts(ck, &.{.{ .name = on, .r0 = 0, .r1 = D, .c0 = r0, .c1 = r0 + H * hd }});
            L.gu = try dr.q4Parts(ck, &.{ .{ .name = gn, .r0 = m0, .r1 = m0 + sp.I }, .{ .name = un, .r0 = m0, .r1 = m0 + sp.I } });
            L.down = try dr.q4Parts(ck, &.{.{ .name = dn, .r0 = 0, .r1 = D, .c0 = m0, .c1 = m0 + sp.I }});
            L.in_norm = try dr.vec(ck, try layerName(&nb[7], i, "input_layernorm.weight"));
            L.post_norm = try dr.vec(ck, try layerName(&nb[7], i, "post_attention_layernorm.weight"));
            L.q_norm = try dr.vec(ck, try layerName(&nb[7], i, "self_attn.q_norm.weight"));
            L.k_norm = try dr.vec(ck, try layerName(&nb[7], i, "self_attn.k_norm.weight"));
            for ([_][]const u8{ "attention_conv", "mlp_conv" }, 0..) |conv, j| {
                var bb: [128]u8 = undefined;
                const base_name = try std.fmt.bufPrint(&bb, "layers.{d}.{s}.base_kernel", .{ i, conv });
                const bt = try ck.get(base_name);
                if (!bt.is(bt.dtype, &.{ 2, 2, D })) return error.UnexpectedTensor;
                const base = try dr.vec(ck, base_name);
                var kb: [128]u8 = undefined;
                const kp_name = try std.fmt.bufPrint(&kb, "layers.{d}.{s}.kernel_projection.weight", .{ i, conv });
                const kpt = try ck.get(kp_name);
                if (kpt.rank != 2 or kpt.shape[0] != 4 * (D / sp.gs) or kpt.shape[1] != D) return error.UnexpectedTensor;
                const kp = try dr.q4Parts(ck, &.{.{ .name = kp_name, .r0 = 0, .r1 = kpt.shape[0] }});
                if (j == 0) {
                    L.a_base = base;
                    L.a_kp = kp;
                } else {
                    L.m_base = base;
                    L.m_kp = kp;
                }
            }
        }
    }
};

/// A 1-D / 2-D tensor's values as fp32 (bf16 widened exactly, fp32 copied).
fn floats(t: core.checkpoint.Tensor, out: []f32) !void {
    if (t.numel() != out.len) return error.UnexpectedTensor;
    switch (t.dtype) {
        .bf16 => for (out, 0..) |*v, i| {
            v.* = @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)) << 16);
        },
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes[0 .. out.len * 4]),
        else => return error.UnexpectedTensor,
    }
}

// ------------------------------------------------------------------------------- host quantization (q4) ---

const HostQ4 = struct {
    w: []u32,
    s: []u16,
    b: []u16,
    npad: usize,

    fn free(h: HostQ4, gpa: std.mem.Allocator) void {
        gpa.free(h.w);
        gpa.free(h.s);
        gpa.free(h.b);
    }
};

const QJob = struct {
    rows: []const u16,
    k: usize,
    words: []u32,
    sc: []u16,
    bi: []u16,
    r0: usize,
    r1: usize,
    // the pack step
    out: []u32 = &.{},
    so: []u16 = &.{},
    bo: []u16 = &.{},
    n: usize = 0,

    fn quant(j: *const QJob) void {
        const kg = j.k / 64;
        const m = j.r1 - j.r0;
        W.quantize4(j.rows[j.r0 * j.k ..][0 .. m * j.k], m, j.k, j.words[j.r0 * (j.k / 8) ..][0 .. m * (j.k / 8)], j.sc[j.r0 * kg ..][0 .. m * kg], j.bi[j.r0 * kg ..][0 .. m * kg]);
    }

    fn pack(j: *const QJob) void {
        W.pack4Range(j.words, j.sc, j.bi, j.n, j.k, j.r0, j.r1, j.out, j.so, j.bo);
    }
};

const max_threads: usize = 8;

fn parallel(jobs: []QJob, comptime f: fn (*const QJob) void) void {
    var threads: [max_threads]?std.Thread = @splat(null);
    for (jobs, 0..) |*j, i| {
        threads[i] = std.Thread.spawn(.{}, f, .{j}) catch null;
        if (threads[i] == null) f(j);
    }
    for (threads[0..jobs.len]) |t| if (t) |th| th.join();
}

/// qmm.quantize4 + shared qmm.pack of bf16 rows [n, k] on the host, rows / columns split over threads (each row's
/// groups and each column's tiles are independent: the same bytes as one pass).
pub fn quantizePack(gpa: std.mem.Allocator, rows: []const u16, n: usize, k: usize) !HostQ4 {
    if (k % 64 != 0 or rows.len != n * k) return error.Invalid;
    const kg = k / 64;
    const npad = (n + 127) / 128 * 128;
    const words = try gpa.alloc(u32, n * k / 8);
    defer gpa.free(words);
    const sc = try gpa.alloc(u16, n * kg);
    defer gpa.free(sc);
    const bi = try gpa.alloc(u16, n * kg);
    defer gpa.free(bi);
    const out = try gpa.alloc(u32, npad / 64 * kg * 8 * 32 * 2);
    errdefer gpa.free(out);
    const so = try gpa.alloc(u16, kg * npad);
    errdefer gpa.free(so);
    const bo = try gpa.alloc(u16, kg * npad);
    errdefer gpa.free(bo);
    @memset(out, 0);
    @memset(so, 0);
    @memset(bo, 0);
    const nt = @max(1, @min(max_threads, n / 64));
    var jobs: [max_threads]QJob = undefined;
    for (0..nt) |t| {
        const r0 = n * t / nt;
        const r1 = n * (t + 1) / nt;
        jobs[t] = .{ .rows = rows, .k = k, .words = words, .sc = sc, .bi = bi, .r0 = r0, .r1 = r1, .out = out, .so = so, .bo = bo, .n = n };
    }
    parallel(jobs[0..nt], QJob.quant);
    parallel(jobs[0..nt], QJob.pack);
    return .{ .w = out, .s = so, .b = bo, .npad = npad };
}

/// headq.prepare's draft_lm_head (TF_GLM53_DRAFT_HEAD=q4): this rank's lm_head share [vocab_part, D] quantized to
/// 4 bits (qmm.quantize4 on the host, torch's float32 steps) for the drafters' block passes; the verify windows keep
/// the bf16 share. Appended to the weights' own buffers.
pub fn buildDraftHead(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *W.Weights) !Q4 {
    const n = w.vocab_part;
    const k = w.config.hidden;
    if (w.lm_head == 0) return error.NoHead;
    const rows = try gpa.alloc(u16, n * k);
    defer gpa.free(rows);
    try d.check(d.api.cuMemcpyDtoH_v2(std.mem.sliceAsBytes(rows).ptr, w.lm_head, n * k * 2), "cuMemcpyDtoH");
    const hq = try quantizePack(gpa, rows, n, k);
    defer hq.free(gpa);
    return .{ .w = try keep(d, w, std.mem.sliceAsBytes(hq.w)), .scales = try keep(d, w, std.mem.sliceAsBytes(hq.s)), .biases = try keep(d, w, std.mem.sliceAsBytes(hq.b)), .n = n, .npad = hq.npad, .k = k, .sk = W.q4SplitK(n, k) };
}

/// Host bytes on the device, owned by the weights.
fn keep(d: *const cuda.Driver, w: *W.Weights, bytes: []const u8) !u64 {
    var m = try cuda.DeviceBuffer.fromHost(d, bytes);
    errdefer m.free();
    try w.buffers.append(w.gpa, m);
    return m.ptr;
}

// ------------------------------------------------------------------------------------------------ coverage ---

/// Every Triton launch a drafter of shapes `sp` can make (cuda_coverage.zig): tap passes of 1 .. tap_rows rows and
/// the block pass, through the compute functions themselves in probe mode.
pub fn enumerate(p: *aot.Probe, sp: Spec) !void {
    const dr = try Drafter.probe(p.gpa, sp);
    defer dr.deinit();
    const dv: Dev = .{ .t = .{ .set = null, .s = undefined, .probe = p } };
    for (1..tap_rows + 1) |n| try dr.tapsCompute(dv, n);
    try dr.blockCompute(dv);
}

test "q4 quantize + pack over threads equals one pass" {
    const gpa = std.testing.allocator;
    const n = 300;
    const k = 128;
    const rows = try gpa.alloc(u16, n * k);
    defer gpa.free(rows);
    for (rows, 0..) |*x, i| x.* = W.bf16Bits(@as(f32, @floatFromInt(@as(i64, @intCast((i * 7919) % 211)) - 105)) / 37.0);
    const got = try quantizePack(gpa, rows, n, k);
    defer got.free(gpa);
    const kg = k / 64;
    const words = try gpa.alloc(u32, n * k / 8);
    defer gpa.free(words);
    const sc = try gpa.alloc(u16, n * kg);
    defer gpa.free(sc);
    const bi = try gpa.alloc(u16, n * kg);
    defer gpa.free(bi);
    W.quantize4(rows, n, k, words, sc, bi);
    const npad = (n + 127) / 128 * 128;
    const out = try gpa.alloc(u32, npad / 64 * kg * 8 * 32 * 2);
    defer gpa.free(out);
    const so = try gpa.alloc(u16, kg * npad);
    defer gpa.free(so);
    const bo = try gpa.alloc(u16, kg * npad);
    defer gpa.free(bo);
    W.pack4(words, sc, bi, n, k, out, so, bo);
    try std.testing.expectEqualSlices(u32, out, got.w);
    try std.testing.expectEqualSlices(u16, so, got.s);
    try std.testing.expectEqualSlices(u16, bo, got.b);
}

test "drafter coverage reaches every drafter kernel (DSpark ft2, DFlash2 shapes)" {
    const gpa = std.testing.allocator;
    var p = aot.Probe.init(gpa);
    defer p.deinit();
    const ds: Spec = .{ .kind = .dspark, .D = 6144, .hd = 64, .H = 16, .KV = 16, .I = 3072, .layers = 3, .ntaps = 5, .block = 8, .top_k = 64, .V = 38720, .vocab_off = 0, .eps = 1e-5, .window = 2048, .causal = true, .world = 4, .theta = 8e6, .mask_id = 154856 };
    const df: Spec = .{ .kind = .dflash, .D = 6144, .hd = 128, .H = 16, .KV = 2, .I = 3072, .layers = 6, .ntaps = 6, .block = 8, .top_k = 16, .V = 38720, .vocab_off = 0, .rsel = 256, .gs = 16, .eps = 1e-5, .window = 2047, .causal = false, .world = 4, .theta = 1e6, .mask_id = 154856 };
    try enumerate(&p, ds);
    try enumerate(&p, df);
    for ([_][]const u8{ "_rmsnorm", "_group_sums", "_embed_b16", "_swiglu", "_prep_kernel", "_dconv_kernel", "_block_attn_kernel", "_dattn_ring", "_residual_add" }) |fname| {
        var hit = false;
        for (p.needs.values()) |nd| hit = hit or std.mem.eql(u8, nd.function, fname);
        if (!hit) std.debug.print("drafter coverage: {s} never enumerated\n", .{fname});
        try std.testing.expect(hit);
    }
}
