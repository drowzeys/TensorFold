//! Full GLM-5.3 served from Zig (Phase 3b): what Glm53Engine sets up and runs a request with, shared by the native
//! server (native/glm53_cuda.zig: rank 0's HTTP side, ranks 1..3 following its requests) and tf-glm53-generate
//! --mode 3b (every rank reading the same prompts file).
//!
//! `Boot`: one rank of the TP world in its served configuration (Phase 3a's run3a setup as one object: NCCL through
//! the TCP rendezvous, the weights, the reference boot's tiles, kernels, the RoCE one-shot, cuBLAS as torch sets it
//! up, the prompt GEMM's workspaces, the MTP layer's dense prompt experts, the decode side stream and L2 prefetch,
//! the sequence-parallel chunk, the Runner at the context's capacity (limit + k + 1), its graphs prewarmed). Decode
//! context parallelism (DCP = the world) turns on past DCP_AUTO tokens of context unless TF_GLM53_DCP says.
//! `Session`: a request on every rank alike (Glm53Engine._run): prompt reuse (cuda_reuse.zig) around Runner.run,
//! and the learned states' files (--learn: each rank writes and reads its own shard of a kept state).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const W = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const rn = @import("cuda_runner.zig");
const smp = @import("sampling.zig");
const prompt = @import("cuda_prompt.zig");
const dec = @import("cuda_decode.zig");
const roce = @import("cuda_roce.zig");
const tiles = @import("tiles.zig");
const reuse = @import("cuda_reuse.zig");
const draft = @import("cuda_draft.zig");
const dhost = @import("draft_host.zig");
const copies = @import("copies.zig");
const multi = @import("cuda_multi.zig");
const mdraft = @import("cuda_mdraft.zig");
const dcpm = @import("cuda_dcp.zig");

pub const world_default: usize = 4;
/// engine.DCP_AUTO: contexts past this interleave the KV cache over the ranks (bf16 latent)
pub const dcp_auto: usize = 200_000;

fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    return if (t.len == 0) null else t;
}

fn envFlag(name: [:0]const u8, default: bool) !bool {
    const v = env(name) orelse return default;
    if (std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "true") or std.ascii.eqlIgnoreCase(v, "on")) return true;
    if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "off")) return false;
    std.log.err("{s}={s}: 0 or 1", .{ name, v });
    return error.BadSetting;
}

fn envInt(comptime T: type, name: [:0]const u8, default: T) !T {
    const v = env(name) orelse return default;
    return std.fmt.parseInt(T, v, 10) catch {
        std.log.err("{s}={s}: a whole number", .{ name, v });
        return error.BadSetting;
    };
}

fn envFloat(name: [:0]const u8, default: f64) !f64 {
    const v = env(name) orelse return default;
    return std.fmt.parseFloat(f64, v) catch {
        std.log.err("{s}={s}: a number", .{ name, v });
        return error.BadSetting;
    };
}

/// TF_GLM53_L2PF: -1 off, 0 bulk, 1 lines, 2 touch (l2pf.MODES).
pub fn l2pfMode(v: []const u8) !c_int {
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return -1;
    if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "on") or std.mem.eql(u8, v, "bulk")) return 0;
    if (std.mem.eql(u8, v, "lines")) return 1;
    if (std.mem.eql(u8, v, "touch")) return 2;
    return error.BadL2pfMode;
}

/// roce.hca_names: "mlx5_0,mlx5_1" (an NCCL_IB_HCA spelling: =^ prefixes, :port suffixes) -> at most two names.
pub fn parseHcas(text: []const u8, out: *[2][]const u8) [][]const u8 {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |raw| {
        if (n == out.len) break;
        var name = std.mem.trim(u8, raw, " =^");
        if (std.mem.indexOfScalar(u8, name, ':')) |c| name = name[0..c];
        if (name.len == 0) continue;
        out[n] = name;
        n += 1;
    }
    return out[0..n];
}

/// One rank's served configuration: tf-glm53-generate --mode 3a's flags, or the TF_GLM53_* variables the server
/// reads (`fromEnv`); every rank must be started with the same ones (checked at the rendezvous).
pub const Options = struct {
    rank: usize = 0,
    world: usize = world_default,
    master: []const u8 = "127.0.0.1:29581",
    model: []const u8 = "",
    aot: []const u8 = "",
    nccl_lib: []const u8 = "libnccl.so.2",
    cublas_lib: []const u8 = "libcublas.so.13",
    cublas_ws: ?usize = null,
    cublas_math: c_int = 0,
    /// the longest prompt plus reply a request may use (Glm53Engine.limit); the runner holds limit + k + 1
    context: usize = 32768,
    k: usize = 2,
    draft_vocab: usize = 32768,
    mtp_reuse: u32 = 2,
    graphs: bool = true,
    prewarm: bool = true,
    fast_load: bool = true,
    roce: bool = true,
    hcas: []const u8 = "",
    gid: u32 = 3,
    roce_health: bool = true,
    spin_limit: u32 = 20_000_000,
    tiles: ?[]const u8 = null,
    inv_ref: ?[]const u8 = null,
    bmm_probe: ?[]const u8 = null,
    prompt_rows: usize = 8192,
    prompt_rows_short: usize = 4096,
    prompt_sp: bool = true,
    experts: []const u8 = "shared",
    pe_det: []const u8 = "slots16",
    unpack_mb: usize = 384,
    side: []const u8 = "af",
    l2pf: []const u8 = "1",
    l2pf_mb: f64 = 8.0,
    multi_select: bool = true,
    device_cands: bool = true,
    mtp_dense: bool = true,
    /// TF_GLM53_DCP: 0 auto (the world past dcp_auto tokens of context, else 1), 1, or the world
    dcp: usize = 0,
    timeout_s: f64 = 5400,
    reuse: reuse.Settings = .{},
    /// Phase 4: the drafters (checkpoint directories, "" none: TF_GLM53_DSPARK / TF_GLM53_DFLASH), DSpark's settings
    /// (TF_GLM53_DSPARK_*), DFlash2's (TF_GLM53_DFLASH_DEPTH / _CONFIDENCE), copy drafts (TF_GLM53_COPY_*), the
    /// default draft mode (TF_GLM53_MTP: dspark | dflash; unset: a drafter when --mtp-drafts 0 (k 0) loads one, else
    /// the MTP head), the drafters' verify rows (TF_GLM53_VERIFY_ROWS)
    dspark: []const u8 = "",
    dflash: []const u8 = "",
    dspark_env: dhost.DSparkEnv = .{},
    dflash_set: dhost.DFlashSettings = .{},
    copy: copies.Settings = .{ .on = false, .match = 0, .most = 0 },
    mode: ?[]const u8 = null,
    verify_rows: usize = 8,
    /// Phase 4b: concurrent streams (--parallel N / TF_GLM53_PARALLEL; 1: one request at a time) and the concurrent
    /// decoder's settings (TF_GLM53_DRAFT_CUT / _STREAMS, TF_GLM53_BATCH_FILL, TF_GLM53_QUICK_ROWS,
    /// TF_GLM53_PRECAPTURE)
    parallel: usize = 1,
    multi_set: multi.Settings = .{},

    /// The server's settings: TF_GLM53_RANK / _WORLD / _MASTER / _AOT / ... (the script's names), `context` from
    /// --context. Unset variables keep tf-glm53-generate --mode 3a's defaults (the image's served ones).
    pub fn fromEnv(model: []const u8, context: usize) !Options {
        var o: Options = .{ .model = model, .context = context };
        o.rank = try envInt(usize, "TF_GLM53_RANK", 0);
        o.world = try envInt(usize, "TF_GLM53_WORLD", world_default);
        o.master = env("TF_GLM53_MASTER") orelse o.master;
        o.aot = env("TF_GLM53_AOT") orelse env("TENSORFOLD_CUDA_KERNELS") orelse "";
        o.nccl_lib = env("TF_GLM53_NCCL_LIB") orelse o.nccl_lib;
        o.cublas_lib = env("TF_GLM53_CUBLAS_LIB") orelse o.cublas_lib;
        if (env("TF_GLM53_CUBLAS_WS")) |_| o.cublas_ws = try envInt(usize, "TF_GLM53_CUBLAS_WS", 0);
        o.k = try envInt(usize, "TF_GLM53_K", o.k);
        o.draft_vocab = try envInt(usize, "TF_GLM53_DRAFT_VOCAB", o.draft_vocab);
        o.mtp_reuse = try envInt(u32, "TF_GLM53_MTP_REUSE", o.mtp_reuse);
        o.graphs = try envFlag("TF_GLM53_GRAPHS", true);
        o.prewarm = try envFlag("TF_GLM53_PREWARM", true);
        o.fast_load = try envFlag("TF_GLM53_FAST_LOAD", true);
        o.roce = try envFlag("TF_GLM53_ROCE", true);
        o.hcas = env("TF_GLM53_HCAS") orelse env("TF_GLM53_ROCE_HCA") orelse "";
        o.gid = try envInt(u32, "TF_GLM53_GID", o.gid);
        o.roce_health = try envFlag("TF_GLM53_ROCE_HEALTH", true);
        o.tiles = env("TF_GLM53_TILES");
        if (o.tiles) |t| if (std.mem.startsWith(u8, t, "load:")) {
            o.tiles = t[5..];
        };
        o.inv_ref = env("TF_GLM53_INV_REF");
        o.bmm_probe = env("TF_GLM53_BMM_PROBE");
        o.prompt_rows = try envInt(usize, "TF_GLM53_PROMPT_ROWS", o.prompt_rows);
        o.prompt_rows_short = try envInt(usize, "TF_GLM53_PROMPT_ROWS_SHORT", o.prompt_rows_short);
        o.prompt_sp = try envFlag("TF_GLM53_PROMPT_SP", true);
        o.experts = env("TF_GLM53_EXPERTS") orelse o.experts;
        o.pe_det = env("TF_GLM53_PE_DET") orelse env("TF_EXL3_PROMPT_DET") orelse o.pe_det;
        o.unpack_mb = try envInt(usize, "TF_GLM53_UNPACK_MB", o.unpack_mb);
        o.side = env("TF_GLM53_SIDE") orelse o.side;
        o.l2pf = env("TF_GLM53_L2PF") orelse o.l2pf;
        o.l2pf_mb = try envFloat("TF_GLM53_L2PF_MB", o.l2pf_mb);
        o.multi_select = try envFlag("TF_GLM53_MULTI_SELECT", true);
        o.device_cands = try envFlag("TF_GLM53_DEVICE_CANDS", true);
        o.mtp_dense = try envFlag("TF_GLM53_MTP_DENSE", true);
        o.dcp = try envInt(usize, "TF_GLM53_DCP", 0);
        o.timeout_s = try envFloat("TF_GLM53_TIMEOUT", o.timeout_s);
        o.reuse = try reuse.Settings.fromEnv();
        o.dspark = env("TF_GLM53_DSPARK") orelse "";
        o.dflash = env("TF_GLM53_DFLASH") orelse "";
        o.dspark_env = .{ .depth = env("TF_GLM53_DSPARK_DEPTH"), .policy = env("TF_GLM53_DSPARK_POLICY"), .confidence = env("TF_GLM53_DSPARK_CONFIDENCE"), .top_k = env("TF_GLM53_DSPARK_TOPK"), .noise = env("TF_GLM53_DSPARK_NOISE"), .filter = env("TF_GLM53_DSPARK_FILTER"), .row_cost = env("TF_GLM53_DSPARK_ROW_COST"), .quant = env("TF_GLM53_DSPARK_QUANT"), .costs = env("TF_GLM53_DSPARK_COSTS") };
        o.dflash_set = .{ .depth = try envInt(usize, "TF_GLM53_DFLASH_DEPTH", 7), .confidence = try envFloat("TF_GLM53_DFLASH_CONFIDENCE", 0.3) };
        o.copy = copies.Settings.parse(env("TF_GLM53_COPY_DRAFTS"), env("TF_GLM53_COPY_MIN"), env("TF_GLM53_COPY_MAX")) catch {
            std.log.err("TF_GLM53_COPY_DRAFTS (0 / 1), _MIN (2..64), _MAX (1..31)", .{});
            return error.BadSetting;
        };
        o.mode = env("TF_GLM53_MTP");
        o.verify_rows = try envInt(usize, "TF_GLM53_VERIFY_ROWS", 8);
        o.parallel = @max(1, try envInt(usize, "TF_GLM53_PARALLEL", 1));
        // Phase 4c: TF_GLM53_CONC_MODE (engine.CONC_MODE) - the drafts of concurrent requests (dflash, dspark, ...)
        if (o.parallel > 1) if (env("TF_GLM53_CONC_MODE")) |m| {
            o.mode = m;
        };
        o.multi_set = multi.Settings.fromEnv() catch {
            std.log.err("TF_GLM53_DRAFT_CUT (a number), _STREAMS / TF_GLM53_QUICK_ROWS (whole numbers)", .{});
            return error.BadSetting;
        };
        if (o.aot.len == 0) {
            std.log.err("glm53: no Triton kernel pack: set TF_GLM53_AOT to the directory with aot.json", .{});
            return error.NoKernelPack;
        }
        return o;
    }

    /// The draft mode of a request without its own (Glm53Engine: TF_GLM53_MTP; a drafter alone with --mtp-drafts 0).
    pub fn defaultMode(o: Options) !rn.Mode {
        if (o.mode) |m| {
            if (std.mem.eql(u8, m, "dspark")) return if (o.dspark.len > 0) .dspark else error.NoDrafter;
            if (std.mem.eql(u8, m, "dflash")) return if (o.dflash.len > 0) .dflash else error.NoDrafter;
            return .mtp; // an MTP input mode (normed/normed, ...): the MTP head
        }
        if (o.k == 0 and o.dspark.len > 0) return .dspark;
        if (o.k == 0 and o.dflash.len > 0) return .dflash;
        return .mtp;
    }

    /// The decode context parallelism a context gets (Glm53Engine: TF_GLM53_DCP, else the world past DCP_AUTO).
    pub fn dcpFor(o: Options) !usize {
        const d = if (o.dcp != 0) o.dcp else if (o.context > dcp_auto) o.world else 1;
        if (d != 1 and d != o.world) return error.BadDcp;
        return d;
    }
};

/// Runner._check_cache_fits: the target layers' bf16 latent rows of `slots` streams of `capacity` tokens (`need`
/// bytes, a drafter's `extra` included) against what is free less the reserve (TF_GLM53_CACHE_RESERVE_GB, default 6,
/// at most half of what is free): on GB10's unified memory an overcommitted cache swaps the node instead of failing.
/// Null when it fits, else the most context a stream that would fit (Python's "--context <= fit").
pub const CacheFit = struct { need: u64, free: u64, spare: u64, fit: u64 };

pub fn cacheFit(row: u64, local: u64, slots: u64, dcp: u64, extra: u64, free: u64, reserve_gb: f64) ?CacheFit {
    const need = row * local * slots + extra;
    const spare: u64 = @intFromFloat(@min(reserve_gb * (1 << 30), @as(f64, @floatFromInt(free)) / 2));
    if (need + spare <= free) return null;
    const room = free -| spare -| extra;
    return .{ .need = need, .free = free, .spare = spare, .fit = room / (row * slots) * dcp };
}

/// The settings every rank must share, hashed (the rendezvous compares them before anything loads).
fn settingsDigest(o: Options, dcp: usize, drafts: u64) u64 {
    var h = std.hash.Wyhash.init(0x6a3b);
    h.update(std.mem.asBytes(&drafts));
    const cw = [_]usize{ @intFromBool(o.copy.on), o.copy.match, o.copy.most, o.verify_rows, o.parallel };
    h.update(std.mem.asBytes(&cw));
    if (o.parallel > 1) o.multi_set.digest(&h);
    const shape = [_]usize{ o.world, o.context, o.k, o.draft_vocab, o.mtp_reuse, @intFromBool(o.graphs), @intFromBool(o.prewarm), @intFromBool(o.roce), o.prompt_rows, o.prompt_rows_short, @intFromBool(o.prompt_sp), o.unpack_mb, @intFromBool(o.multi_select), @intFromBool(o.device_cands), @intFromBool(o.mtp_dense), dcp, @intFromBool(o.roce_health), @intFromBool(o.reuse.on), o.reuse.gap, o.reuse.entries, @intFromBool(o.reuse.loose), @intFromFloat(o.l2pf_mb * 1024) };
    h.update(std.mem.asBytes(&shape));
    for ([_][]const u8{ o.experts, o.pe_det, o.side, o.l2pf }) |t| {
        h.update(t);
        h.update("|");
    }
    return h.final();
}

fn now(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn since(io: std.Io, t0: i128) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e9;
}

/// One rank in its served configuration (run3a's setup), heap-held: the Forward and the Runner point into it.
pub const Boot = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    o: Options,
    dcp: usize,
    driver: cuda.Driver,
    ctx: cuda.Context,
    lib: cuda.nccl.Library,
    rdv: cuda.rendezvous.Rendezvous,
    comm: cuda.nccl.Communicator,
    cfg: Config,
    w: W.Weights,
    k: kern.Kernels,
    stream: cuda.Stream,
    side_stream: cuda.Stream,
    inv: cuda.DeviceBuffer,
    fast: ?*roce.Roce = null,
    blas_lib: cuda.cublas.Library,
    ws: cuda.DeviceBuffer,
    blas: cuda.cublas.Blas,
    cache: ?prompt.UnpackCache = null,
    main_wide: prompt.Wide,
    side_wide: prompt.Wide,
    mtp_dense: ?prompt.MtpDense = null,
    side: ?dec.Side = null,
    l2pf: ?dec.L2pf = null,
    sp: ?prompt.Sp = null,
    fwd: fw.Forward, // the sequence-parallel chunk points at it (Sp.main); the Runner holds a copy
    r: rn.Runner,
    // Phase 4: the drafters' configs (read and checked before anything loads), their tap plan, the drafters, the
    // default draft mode
    dspark_cfg: ?dhost.DSparkConfig = null,
    dflash_cfg: ?dhost.DFlashConfig = null,
    dspark_set: dhost.DSparkSettings = .{ .depth = 0 },
    plan: dhost.TapPlan = .{},
    dspark_dr: ?*draft.Drafter = null,
    dflash_dr: ?*draft.Drafter = null,
    /// Phase 4b: the concurrent decoder (--parallel N > 1; cuda_multi.zig), on every rank
    multi: ?*multi.Multi = null,
    mode: rn.Mode = .mtp,
    load_s: f64 = 0,
    prewarm_graphs: usize = 0,
    prewarm_s: f64 = 0,
    nccl_version: c_int = 0,
    stage: u8 = 0, // how far init got (deinit undoes that much)

    /// Every rank of the world loaded and warmed, in step with the others (rendezvous barriers between stages).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, o: Options) !*Boot {
        const b = try gpa.create(Boot);
        errdefer gpa.destroy(b);
        b.* = undefined;
        b.gpa = gpa;
        b.io = io;
        b.o = o;
        b.stage = 0;
        b.fast = null;
        b.cache = null;
        b.mtp_dense = null;
        b.side = null;
        b.l2pf = null;
        b.sp = null;
        b.dspark_cfg = null;
        b.dflash_cfg = null;
        b.dspark_set = .{ .depth = 0 };
        b.plan = .{};
        b.dspark_dr = null;
        b.dflash_dr = null;
        b.multi = null;
        b.mode = try o.defaultMode();
        b.dcp = try o.dcpFor();
        errdefer b.undo();
        try b.bringUp();
        return b;
    }

    fn bringUp(b: *Boot) !void {
        const gpa = b.gpa;
        const io = b.io;
        const o = b.o;
        if (o.rank >= o.world) return error.BadRank;
        try checkParallel(o, b.dcp);
        const det16 = std.mem.eql(u8, o.pe_det, "slots16");
        if (!det16 and !std.mem.eql(u8, o.pe_det, "0")) return error.UnsupportedPromptDet;
        const shared = std.mem.eql(u8, o.experts, "shared");
        if (!shared and !std.mem.eql(u8, o.experts, "tf")) return error.UnsupportedExperts;
        b.driver = try cuda.Driver.open();
        b.stage = 1;
        b.ctx = try cuda.Context.init(&b.driver, 0);
        b.stage = 2;
        const master = try cuda.rendezvous.parseMaster(o.master);
        b.lib = try cuda.nccl.Library.openPath(o.nccl_lib);
        b.stage = 3;
        b.nccl_version = try b.lib.version();
        b.rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, o.rank, o.world, o.timeout_s);
        b.stage = 4;
        var uid: cuda.nccl.UniqueId = undefined;
        if (o.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&b.lib);
        try b.rdv.broadcast(std.mem.asBytes(&uid));
        b.comm = try cuda.nccl.Communicator.init(&b.lib, uid, o.rank, o.world);
        b.stage = 5;
        std.log.info("glm53 rank {d} of {d}: NCCL {d} up; context {d}, MTP drafts {d}, DCP {d}, prompt reuse {s}", .{ o.rank, o.world, b.nccl_version, o.context, o.k, b.dcp, if (o.reuse.on) "on" else "off" });
        b.cfg = try Config.load(gpa, io, o.model);
        // Phase 4: the drafters' configs, checked against the target before anything big loads; the tap plan
        // (dspark_host.tap_plan: DFlash2's layers first, DSpark's after)
        var dh = std.hash.Wyhash.init(0x44);
        var dspark_layers: [dhost.max_taps]usize = undefined;
        var want: [2]?[]const usize = .{ null, null };
        if (o.dflash.len > 0) {
            const text = try readText(gpa, io, o.dflash, "config.json");
            defer gpa.free(text);
            const c = try dhost.DFlashConfig.parse(gpa, text);
            try c.check(o.world, b.cfg.hidden, b.cfg.layers);
            b.dflash_cfg = c;
            want[0] = b.dflash_cfg.?.taps[0..c.n_taps];
            dh.update(std.mem.asBytes(&c.crc));
            o.dflash_set.digest(&dh);
        }
        if (o.dspark.len > 0) {
            const text = try readText(gpa, io, o.dspark, "config.json");
            defer gpa.free(text);
            const c = try dhost.DSparkConfig.parse(gpa, text);
            try c.check(o.world, b.cfg.hidden, b.cfg.vocab, b.cfg.layers);
            b.dspark_cfg = c;
            b.dspark_set = try dhost.DSparkSettings.parse(c.block, o.dspark_env);
            want[1] = c.tapLayers(&dspark_layers);
            dh.update(std.mem.asBytes(&c.crc));
            b.dspark_set.digest(&dh);
        }
        b.plan = try dhost.TapPlan.make(want);
        try checkParallelDrafters(o.parallel, if (b.dflash_cfg) |c| c.block else null, if (b.dspark_cfg) |c| c.block else null);
        const mode_word: u64 = @intFromEnum(b.mode);
        dh.update(std.mem.asBytes(&mode_word));
        try b.rdv.sync("settings", settingsDigest(o, b.dcp, dh.final()), o.timeout_s);

        const t_load = now(io);
        b.w = try W.load(gpa, io, &b.driver, o.model, b.cfg, .{ .layers = b.cfg.layers, .rank = o.rank, .world = o.world, .head = true, .log_every = 10, .fast = o.fast_load, .mtp = o.k > 0, .draft_vocab = if (o.k > 0) o.draft_vocab else 0 });
        b.stage = 6;
        if (o.tiles) |path| {
            const t = try tiles.load(gpa, io, &b.w, path);
            std.log.info("glm53 rank {d}: tiles from {s}: {d} linears, {d} differ from x3linear.plan", .{ o.rank, path, t.linears, t.changed });
        }
        if (b.plan.n > 0) { // headq.prepare(drafts=True): the 4-bit draft copy of this rank's head share
            b.w.draft_lm = try draft.buildDraftHead(gpa, &b.driver, &b.w);
        }
        b.load_s = since(io, t_load);
        try b.rdv.sync("loaded", 0, o.timeout_s);

        b.k = try kern.Kernels.load(gpa, io, &b.ctx, o.aot);
        b.stage = 7;
        b.stream = try cuda.Stream.init(&b.driver, true);
        b.stage = 8;
        const half = b.cfg.rope / 2;
        b.inv = try cuda.DeviceBuffer.alloc(&b.driver, half * 4);
        b.stage = 9;
        try b.k.invFreq(b.stream, b.inv.ptr, b.cfg.rope, b.cfg.rope_theta);
        try b.stream.synchronize();
        if (o.inv_ref) |path| {
            const ref = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
            defer gpa.free(ref);
            const mine = try gpa.alloc(u8, half * 4);
            defer gpa.free(mine);
            try b.inv.download(0, mine);
            if (ref.len != mine.len) return error.BadInvReference;
            if (!std.mem.eql(u8, ref, mine)) {
                std.log.warn("glm53: inv_freq differs from torch's (the reference's bytes are used)", .{});
                try b.inv.upload(0, ref);
            }
        }
        if (o.roce and o.world > 1) {
            var hb: [2][]const u8 = undefined;
            const hcas = parseHcas(o.hcas, &hb);
            b.fast = try roce.Roce.start(gpa, &b.driver, &b.rdv, o.rank, o.world, .{ .hcas = hcas, .gid_index = o.gid, .spin_limit = o.spin_limit });
            if (!try b.fast.?.verify(gpa, &b.comm, b.stream, b.cfg.hidden, fw.decode_rows)) return error.RoceCheckFailed;
        }
        b.stage = 10;
        b.blas_lib = try cuda.cublas.Library.openPath(o.cublas_lib);
        b.stage = 11;
        const major: c_int = @intCast((try b.ctx.capability()) / 10);
        const ws_bytes = o.cublas_ws orelse cuda.cublas.torchWorkspaceBytes(major, null);
        b.ws = try cuda.DeviceBuffer.alloc(&b.driver, @max(ws_bytes, 16));
        b.stage = 12;
        b.blas = try cuda.cublas.Blas.init(&b.blas_lib, b.ws.ptr, ws_bytes, o.cublas_math);
        b.stage = 13;
        if (o.bmm_probe) |path| if (!try bmmProbe(gpa, io, &b.driver, b.stream, &b.blas, path)) return error.BmmBitsDiffer;
        if (o.unpack_mb > 0) b.cache = try prompt.UnpackCache.init(gpa, &b.driver, o.unpack_mb << 20);
        b.main_wide = try prompt.Wide.init(&b.driver, &b.k, b.stream, &b.w, o.prompt_rows, if (b.cache) |*c| c else null, &b.blas);
        b.stage = 14;
        b.side_stream = try cuda.Stream.init(&b.driver, true);
        b.stage = 15;
        b.side_wide = try prompt.Wide.init(&b.driver, &b.k, b.side_stream, &b.w, o.prompt_rows / o.world + 1, null, null);
        b.stage = 16;
        try b.stream.synchronize();
        try b.side_stream.synchronize();
        if (o.mtp_dense and o.k > 0 and shared and b.w.mtp != null) {
            if (prompt.MtpDense.init(gpa, &b.driver, &b.k, b.stream, &b.w, &b.blas, o.prompt_rows)) |x| {
                b.mtp_dense = x;
            } else |e| std.log.warn("glm53 rank {d}: MTP prompt rows dense unavailable ({t})", .{ o.rank, e });
        }
        const letters = try dec.parseSide(o.side);
        if (letters.any()) b.side = try dec.Side.init(&b.driver, letters);
        const pf_mode = try l2pfMode(o.l2pf);
        if (pf_mode >= 0) {
            if (!(o.l2pf_mb > 0 and o.l2pf_mb <= 64)) return error.BadL2pfBudget;
            b.l2pf = try dec.L2pf.init(gpa, &b.driver, &b.w, .{ .mode = pf_mode, .mb = o.l2pf_mb });
        }
        b.fwd = .{ .w = &b.w, .k = &b.k, .s = b.stream, .inv = b.inv.ptr, .comm = &b.comm, .fast = b.fast, .ring = true, .wide = &b.main_wide, .shared_experts = shared, .pe_det16 = det16, .side = if (b.side) |*x| x else null, .l2pf = if (b.l2pf) |*x| x else null, .multi_select = o.multi_select, .prof = null, .mtp_dense = if (b.mtp_dense) |*x| x else null, .dcp = b.dcp, .taps = try fw.Taps.of(&b.plan) };
        if (o.prompt_sp and o.world > 1 and b.dcp == 1) b.sp = try prompt.Sp.init(&b.driver, &b.fwd, b.side_stream, &b.side_wide);
        b.stage = 17;
        const capacity = o.context + o.k + 1; // Glm53Engine: Runner(limit + k + 1)
        try b.checkCacheFits(capacity);
        // --parallel N: N slots of the caches; copy drafts are one stream's (Runner: slots == 1)
        const copy_set: copies.Settings = if (o.parallel > 1) .{ .on = false, .match = 0, .most = 0 } else o.copy;
        b.r = try rn.Runner.init(gpa, io, &b.driver, b.fwd, capacity, .{ .k = o.k, .graphs = o.graphs, .window = 128, .reuse = o.mtp_reuse, .served = true, .prompt_rows = o.prompt_rows, .prompt_rows_short = o.prompt_rows_short, .prompt_sp = o.prompt_sp, .roce_health = o.roce_health, .sp = if (b.sp) |*x| x else null, .device_cands = o.device_cands, .dcp = b.dcp, .verify_rows = o.verify_rows, .draft_rows = if (b.dspark_cfg) |c| c.block + 1 else 0, .copy = copy_set, .slots = o.parallel });
        b.stage = 18;
        // Phase 4: the drafters on every rank (each its heads / MLP share), attached to the runner before its prewarm
        if (b.dflash_cfg != null) {
            b.dflash_dr = try draft.Drafter.load(gpa, io, &b.driver, &b.k, b.stream, &b.w, o.dflash, .dflash, b.dspark_set, o.dflash_set);
            try b.dflash_dr.?.setTaps(b.plan.cols[0][0..b.plan.n_cols[0]], b.plan.n);
            b.r.dflash = b.dflash_dr;
            std.log.info("glm53 rank {d}: DFlash2 drafter {s} (block {d}, taps after layers {any}, {d:.0} MiB on the device)", .{ o.rank, std.fs.path.basename(o.dflash), b.dflash_dr.?.sp.block, b.dflash_cfg.?.taps[0..b.dflash_cfg.?.n_taps], @as(f64, @floatFromInt(b.dflash_dr.?.deviceBytes())) / (1 << 20) });
        }
        if (b.dspark_cfg != null) {
            b.dspark_dr = try draft.Drafter.load(gpa, io, &b.driver, &b.k, b.stream, &b.w, o.dspark, .dspark, b.dspark_set, o.dflash_set);
            try b.dspark_dr.?.setTaps(b.plan.cols[1][0..b.plan.n_cols[1]], b.plan.n);
            b.r.dspark = b.dspark_dr;
            std.log.info("glm53 rank {d}: DSpark drafter {s} (block {d}, policy {t}, confidence {d}, depth {d}, {d:.0} MiB on the device)", .{ o.rank, std.fs.path.basename(o.dspark), b.dspark_cfg.?.block, b.dspark_set.policy, b.dspark_set.confidence, b.dspark_set.depth, @as(f64, @floatFromInt(b.dspark_dr.?.deviceBytes())) / (1 << 20) });
        }
        if (o.parallel > 1) {
            b.multi = try multi.Multi.init(gpa, io, &b.r, o.parallel, o.multi_set, o.rank == 0);
            const per = b.multi.?.bytesPerStream();
            std.log.info("glm53 rank {d}: {d} concurrent streams (MTP drafts {d}, windows up to {d} rows), {d:.2} GiB of caches each ({d} tokens); draft cut {d} from {d} streams", .{ o.rank, o.parallel, o.k, b.multi.?.rows, @as(f64, @floatFromInt(per)) / (1 << 30), o.context, o.multi_set.draft_cut, o.multi_set.cut_streams });
            if (b.multi.?.frows > 0) std.log.info("glm53 rank {d}: concurrent drafters: DFlash2 {s}, DSpark {s}; up to {d} drafts a stream, confidence floor {d} from {d} streams; requests draft with {t}", .{ o.rank, if (b.dflash_dr != null) "on" else "off", if (b.dspark_dr != null) "on" else "off", b.multi.?.frows - 1, o.multi_set.cutDFlash(), o.multi_set.cut_streams, b.mode });
        }
        b.stage = 19;
        if (o.prewarm) {
            const tp = now(io);
            if (b.multi) |mu| {
                // Runner.prewarm(capture=False) then the concurrent windows (multi.prewarm)
                _ = try b.r.prewarmWith(false);
                b.prewarm_graphs = try mu.prewarm();
            } else {
                b.prewarm_graphs = try b.r.prewarm();
            }
            b.prewarm_s = since(io, tp);
        }
        try b.rdv.sync("warm", b.prewarm_graphs, o.timeout_s);
        std.log.info("glm53 rank {d}: loaded in {d:.0} s, {d} decode graphs captured in {d:.1} s{s}", .{ o.rank, b.load_s, b.prewarm_graphs, b.prewarm_s, if (b.dcp > 1) "; decode context parallel" else "" });
    }

    /// Runner._check_cache_fits before the caches allocate (every rank; Python's message on the refusing rank).
    fn checkCacheFits(b: *Boot, capacity: usize) !void {
        const mem = try b.ctx.memInfo();
        const reserve_gb: f64 = if (std.c.getenv("TF_GLM53_CACHE_RESERVE_GB")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 6) else 6;
        const row: u64 = b.cfg.layers * b.cfg.latentWidth() * 2;
        const local: u64 = dcpm.localCapacity(capacity, b.dcp);
        const x = cacheFit(row, local, b.o.parallel, b.dcp, 0, mem.free, reserve_gb) orelse return;
        const gib = 1 << 30;
        std.log.err("context {d} x {d} streams needs {d:.1} GiB of caches a rank, {d:.1} GiB is free (keeping {d:.0} GiB spare): use --context <= {d} with --parallel {d}, or fewer streams", .{ capacity, b.o.parallel, @as(f64, @floatFromInt(x.need)) / gib, @as(f64, @floatFromInt(x.free)) / gib, @as(f64, @floatFromInt(x.spare)) / gib, x.fit, b.o.parallel });
        return error.CacheDoesNotFit;
    }

    fn undo(b: *Boot) void {
        const s = b.stage;
        if (b.multi) |x| x.deinit();
        if (b.dspark_dr) |x| x.deinit();
        if (b.dflash_dr) |x| x.deinit();
        if (s >= 18) b.r.deinit();
        if (s >= 17) if (b.sp) |*x| x.deinit();
        if (s >= 16) {
            if (b.l2pf) |*x| x.deinit();
            if (b.side) |*x| x.deinit();
            if (b.mtp_dense) |*x| x.deinit();
            b.side_wide.deinit();
        }
        if (s >= 15) b.side_stream.deinit();
        if (s >= 14) b.main_wide.deinit();
        if (s >= 13) {
            if (b.cache) |*c| c.deinit();
            b.blas.deinit();
        }
        if (s >= 12) b.ws.free();
        if (s >= 11) b.blas_lib.close();
        if (s >= 10) if (b.fast) |rc| rc.deinit(b.gpa);
        if (s >= 9) b.inv.free();
        if (s >= 8) b.stream.deinit();
        if (s >= 7) b.k.deinit();
        if (s >= 6) b.w.deinit();
        if (s >= 5) b.comm.deinit();
        if (s >= 4) b.rdv.close();
        if (s >= 3) b.lib.close();
        if (s >= 2) b.ctx.deinit();
        if (s >= 1) b.driver.close();
    }

    pub fn deinit(b: *Boot) void {
        b.undo();
        b.gpa.destroy(b);
    }
};

/// Glm53Engine's refusals of --parallel N > 1 (before anything loads): decode context parallelism must be off, the
/// verify window of N x (k + 1) rows must be a decode window. Phase 4c: DSpark and DFlash2 draft concurrent streams
/// too (cuda_mdraft.zig; the drafters' own check, N blocks in one pass, follows once their configs are read).
pub fn checkParallel(o: Options, dcp: usize) !void {
    if (o.parallel <= 1) return;
    const quiet = @import("builtin").is_test;
    if (dcp > 1) {
        if (!quiet) std.log.err("--parallel {d} needs decode context parallelism off: a context of at most {d} tokens a stream (--context), TF_GLM53_DCP unset or 1", .{ o.parallel, dcp_auto });
        return error.ParallelNeedsDcpOff;
    }
    _ = multi.windowRows(o.parallel, o.k) catch {
        if (!quiet) std.log.err("{d} streams x {d} rows = {d} > {d}: a window that wide is not a decode window (lower --parallel or --mtp-drafts)", .{ o.parallel, o.k + 1, o.parallel * (o.k + 1), fw.decode_rows });
        return error.WindowTooWide;
    };
}

/// Phase 4c: a loaded drafter drafts N concurrent streams when N of its blocks fit one pass (cuda_mdraft.fits:
/// N x block <= 64 rows; block 8: up to 8 streams).
pub fn checkParallelDrafters(parallel: usize, dflash_block: ?usize, dspark_block: ?usize) !void {
    if (parallel <= 1) return;
    const quiet = @import("builtin").is_test;
    for ([_]?usize{ dflash_block, dspark_block }, [_][]const u8{ "TF_GLM53_DFLASH", "TF_GLM53_DSPARK" }) |maybe, name| {
        const block = maybe orelse continue;
        if (parallel * block > mdraft.max_pass_rows or parallel > mdraft.max_streams) {
            if (!quiet) std.log.err("{s} with --parallel {d}: {d} streams x a block of {d} rows > {d} rows a draft pass; use --parallel <= {d}", .{ name, parallel, parallel, block, mdraft.max_pass_rows, mdraft.max_pass_rows / block });
            return error.DrafterTooWide;
        }
    }
}

/// --bmm-probe FILE (tf-glm53-generate's): torch.bmm's bits on fused.attention_core's wide absorb / expand shapes
/// against cuBLAS here with torch's arguments; true when every case matches.
fn bmmProbe(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, blas: *const cuda.cublas.Blas, path: []const u8) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var all_ok = true;
    for ((parsed.value.object.get("cases") orelse return error.BadProbe).array.items) |cv| {
        const o = cv.object;
        const kind = o.get("kind").?.string;
        const H: usize = @intCast(o.get("H").?.integer);
        const R: usize = @intCast(o.get("R").?.integer);
        const nope: usize = @intCast(o.get("nope").?.integer);
        const qd: usize = @intCast(o.get("qd").?.integer);
        const lw: usize = @intCast(o.get("lw").?.integer);
        const vd: usize = @intCast(o.get("vd").?.integer);
        const want = o.get("sha256").?.string;
        const absorb = std.mem.eql(u8, kind, "absorb");
        const na = if (absorb) R * H * qd else R * H * lw;
        const nb = if (absorb) H * nope * lw else H * vd * lw;
        const nc = if (absorb) H * R * lw else H * R * vd;
        const ha = try gpa.alloc(u16, na);
        defer gpa.free(ha);
        const hb = try gpa.alloc(u16, nb);
        defer gpa.free(hb);
        probeFill(ha, if (absorb) 1 else 3);
        probeFill(hb, if (absorb) 2 else 4);
        var da = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ha));
        defer da.free();
        var db = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(hb));
        defer db.free();
        var dc = try cuda.DeviceBuffer.alloc(d, nc * 2);
        defer dc.free();
        if (absorb) {
            try blas.bgemmBf16(s.handle, .n, .n, lw, R, nope, db.ptr, lw, nope * lw, da.ptr, H * qd, qd, dc.ptr, lw, R * lw, H);
        } else {
            try blas.bgemmBf16(s.handle, .t, .n, vd, R, lw, db.ptr, lw, vd * lw, da.ptr, H * lw, lw, dc.ptr, vd, R * vd, H);
        }
        try s.synchronize();
        const hc = try gpa.alloc(u8, nc * 2);
        defer gpa.free(hc);
        try dc.download(0, hc);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(hc, &digest, .{});
        const got = std.fmt.bytesToHex(digest, .lower);
        const ok = std.mem.eql(u8, &got, want);
        all_ok = all_ok and ok;
        std.log.info("glm53 bmm probe {s} H {d} R {d}: {s}", .{ kind, H, R, if (ok) "equal" else "DIFFERS" });
    }
    return all_ok;
}

fn probeFill(out: []u16, salt: u64) void {
    for (out, 0..) |*v, i| {
        const h: u64 = ((@as(u64, i) +% salt *% 1000003) *% 2654435761) & 0xFFFFFFFF;
        const x: f32 = @as(f32, @floatFromInt(@as(i64, @intCast((h >> 8) % 4001)) - 2000)) / @as(f32, 2000.0);
        v.* = W.bf16Bits(x);
    }
}

/// The model's end ids as the native server takes them (hf_text.eosIds): config.json's eos_token_id (an id or a
/// list; text_config's when the top level names none), plus tokenizer_config.json's eos_token when it is an added
/// token of tokenizer.json.
pub fn eosIds(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (readJson(a, io, dir, "config.json")) |doc| if (doc == .object) {
        const holders = [_]?std.json.Value{ doc.object.get("eos_token_id"), if (doc.object.get("text_config")) |tc| (if (tc == .object) tc.object.get("eos_token_id") else null) else null };
        for (holders) |h| {
            const v = h orelse continue;
            switch (v) {
                .integer => |i| try out.append(gpa, @intCast(i)),
                .array => |list| for (list.items) |x| if (x == .integer) try out.append(gpa, @intCast(x.integer)),
                else => {},
            }
            if (out.items.len > 0) break;
        }
    };
    if (readJson(a, io, dir, "tokenizer_config.json")) |doc| if (doc == .object) if (doc.object.get("eos_token")) |e| {
        const piece = switch (e) {
            .string => |s| s,
            .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else "") else "",
            else => "",
        };
        if (piece.len > 0) if (readJson(a, io, dir, "tokenizer.json")) |tok| if (tok == .object) if (tok.object.get("added_tokens")) |added| if (added == .array) {
            for (added.array.items) |t| {
                if (t != .object) continue;
                const c = t.object.get("content") orelse continue;
                const id = t.object.get("id") orelse continue;
                if (c == .string and id == .integer and std.mem.eql(u8, c.string, piece)) {
                    const v: u32 = @intCast(id.integer);
                    if (std.mem.indexOfScalar(u32, out.items, v) == null) try out.append(gpa, v);
                }
            }
        };
    };
    return out.toOwnedSlice(gpa);
}

fn readJson(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ?std.json.Value {
    const path = std.fs.path.join(a, &.{ dir, name }) catch return null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(256 << 20)) catch return null;
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch null;
}

/// One request as every rank runs it (Glm53Engine.generate's header, after rank 0's choices).
pub const Exec = struct {
    prompt: []const u32,
    max_tokens: usize,
    sampling: ?smp.Sampling = null,
    /// false: "draft": false - serial (k = 0), the cold reference: cut alike, never resumed, keeps nothing
    draft: bool = true,
    /// tokens that end the reply (empty: ignore_eos)
    eos: []const u32 = &.{},
    /// prompt reuse (rank 0's PromptReuse.choose): resume point, cuts, kept points (n * 2 + shared), flags
    begin: usize = 0,
    stops: []const usize = &.{},
    keeps: []const u32 = &.{},
    flags: u32 = 0,
    hooks: ?rn.Hooks = null,
    /// Phase 4: the draft mode (null: the boot's default) and copy drafts (drafted requests only)
    mode: ?rn.Mode = null,
    copies: bool = true,
    /// rn.Gen.prompt_votes (the server at --parallel 1, every rank alike)
    prompt_votes: bool = false,
};

pub const Done = struct {
    stats: rn.Stats,
    kept: []usize = &.{}, // keep points this request kept (gpa-owned)
    replay: bool = false,
};

/// A request on one rank: prompt reuse (when on) around Runner.run, as Glm53Engine._run.
pub const Session = struct {
    gpa: std.mem.Allocator,
    boot: *Boot,
    eos: []u32, // the model's end ids (eosIds)
    reuse: ?*reuse.Reuse = null,
    plan: reuse.Plan,

    /// The session over a booted rank: its prompt reuse with the ranks' least budget (every rank saves and drops
    /// alike: the settings were compared at the rendezvous, the budget is agreed here).
    pub fn init(gpa: std.mem.Allocator, boot: *Boot) !Session {
        const o = boot.o;
        var s: Session = .{ .gpa = gpa, .boot = boot, .eos = try eosIds(gpa, boot.io, o.model), .plan = .{ .ids = reuse.specialIds(gpa, boot.io, o.model), .gap = o.reuse.gap } };
        errdefer gpa.free(s.eos);
        const wanted: u64 = @intFromFloat(o.reuse.gib * (1 << 30));
        var mine: u64 = 0;
        if (o.reuse.on) {
            const mem = try boot.ctx.memInfo();
            mine = reuse.Reuse.keptBudget(mem.free, wanted) >> 20;
        }
        var all: [cuda.rendezvous.max_world * 8]u8 = undefined;
        try boot.rdv.allGather(std.mem.asBytes(&mine), all[0 .. boot.o.world * 8]);
        var least: u64 = std.math.maxInt(u64);
        for (0..boot.o.world) |i| least = @min(least, std.mem.readInt(u64, all[i * 8 ..][0..8], .little));
        if (o.reuse.on and boot.multi != null) {
            // --parallel N: the kept states live in the streams' slots (prefixes.SlotReuse); the one-stream store
            // never runs
            try boot.multi.?.enableReuse(s.plan, least << 20, o.reuse.entries, o.reuse.loose);
            if (o.rank == 0) std.log.info("glm53: prompt reuse on in the {d} streams' slots (never saved out of them): up to {d} entries, keep points every >= {d} tokens", .{ o.parallel, o.reuse.entries, o.reuse.gap });
        } else if (o.reuse.on) {
            s.reuse = try reuse.Reuse.init(gpa, &boot.r, s.plan, least << 20, o.reuse.entries, o.reuse.loose);
            if (o.rank == 0) std.log.info("glm53: prompt reuse on: kept prompt states up to {d:.2} GiB a rank and {d} entries, keep points every >= {d} tokens at assistant openers; <|user|> {?d}, <|assistant|> {?d}, <think> {?d}", .{ @as(f64, @floatFromInt(least << 20)) / (1 << 30), o.reuse.entries, o.reuse.gap, s.plan.ids.user, s.plan.ids.assistant, s.plan.ids.think });
        }
        return s;
    }

    pub fn deinit(s: *Session) void {
        if (s.reuse) |r| r.deinit();
        s.gpa.free(s.eos);
    }

    /// Glm53Engine.generate's clamps: the prompt under the context, max_tokens within what is left.
    pub fn clampTokens(s: *const Session, prompt_len: usize, max_tokens: usize) !usize {
        const limit = s.boot.o.context;
        if (prompt_len == 0 or prompt_len >= limit) return error.PromptTooLong;
        return @max(1, @min(max_tokens, limit - prompt_len));
    }

    /// Rank 0: PromptReuse.choose for a request (no reuse: nothing resumed, nothing cut).
    pub fn choose(s: *Session, prompt_ids: []const u32, drafting: bool) !reuse.Choice {
        const pr = s.reuse orelse return .{ .begin = 0, .stops = &.{}, .keeps = &.{} };
        return reuse.choose(s.gpa, s.plan, &pr.store, prompt_ids, drafting, 0);
    }

    /// Glm53Engine._run on this rank: the request through prompt reuse (when on) and Runner.run. Tokens go to `out`.
    pub fn exec(s: *Session, x: Exec, out: *std.ArrayList(u32)) !Done {
        const k = if (x.draft) s.boot.o.k else 0;
        const mode: rn.Mode = if (x.draft) x.mode orelse s.boot.mode else .mtp;
        const g: rn.Gen = .{ .max_tokens = x.max_tokens, .sampling = x.sampling, .k = k, .eos = x.eos, .hooks = x.hooks, .mode = mode, .copies = x.draft and x.copies, .prompt_votes = x.prompt_votes };
        if (s.reuse) |pr| {
            const res = try pr.run(x.prompt, x.begin, x.stops, x.keeps, x.flags, x.draft, 0, g, out);
            return .{ .stats = res.stats, .kept = res.kept, .replay = res.replay };
        }
        if (x.begin != 0) return error.ReuseOff;
        return .{ .stats = try s.boot.r.run(x.prompt, g, out) };
    }

    // ---------------------------------------------------------------------------------- learned states (--learn) ---

    /// A learned state's file for this rank: <dir>/<key>.r<rank>.bin.
    pub fn statePath(buf: []u8, dir: []const u8, key: u64, rank: usize) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.r{d}.bin", .{ dir, key, rank }, 0);
    }

    const file_magic: u32 = 0x4c333547; // "G53L"
    const file_version: u32 = 1;
    pub const Head = extern struct { magic: u32, version: u32, rank: u32, world: u32, dcp: u32, n: u32, key: u64, identity: u64, views: u32, carry: u32, bytes: u64 };

    /// This rank's shard of kept state `e` (its rows - saved, or still live - and its carry) to `file` through a
    /// temporary file renamed into place; returns the file's bytes. The rows go through a pinned staging buffer.
    pub fn writeState(s: *Session, e: *reuse.Kept, file: [:0]const u8, key: u64, identity: u64) !u64 {
        const pr = s.reuse orelse return error.ReuseOff;
        const b = s.boot;
        const D = b.cfg.hidden;
        const vs = try pr.views(e.n());
        defer s.gpa.free(vs);
        var src_offs: ?[]usize = null;
        defer if (src_offs) |x| s.gpa.free(x);
        var src_base: u64 = 0;
        if (e.saved) |sv| {
            // the copy's parts sit at its own (longer) state's offsets
            const big = try pr.views(sv.n);
            defer s.gpa.free(big);
            src_offs = try s.gpa.alloc(usize, big.len);
            _ = reuse.savedSize(big, src_offs.?);
            src_base = sv.mem.ptr;
        } else if (!pr.store.inLive(e)) return error.StateGone;
        var total: u64 = 0;
        for (vs) |v| total += v.row * v.rows;
        const head: Head = .{ .magic = file_magic, .version = file_version, .rank = @intCast(b.o.rank), .world = @intCast(b.o.world), .dcp = @intCast(b.dcp), .n = @intCast(e.n()), .key = key, .identity = identity, .views = @intCast(vs.len), .carry = @intCast(D * 2), .bytes = total };
        var tmp_buf: [1200]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
        const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.StateWrite;
        var ok = false;
        defer if (!ok) {
            _ = std.c.unlink(tmp);
        };
        {
            defer _ = std.c.close(fd);
            try putAll(fd, std.mem.asBytes(&head));
            for (vs) |v| {
                const sz = [2]u64{ v.row, v.rows };
                try putAll(fd, std.mem.sliceAsBytes(&sz));
            }
            try b.r.f.s.synchronize(); // the copies into the saved rows / the live rows' last chunk have landed
            var stage = try cuda.HostBuffer.alloc(&b.driver, 64 << 20);
            defer stage.free();
            for (vs, 0..) |v, i| {
                const src = if (src_offs) |offs| src_base + offs[i] else v.ptr;
                var done: usize = 0;
                const n = v.row * v.rows;
                while (done < n) {
                    const part = @min(n - done, stage.bytes.len);
                    try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, src + done, part), "cuMemcpyDtoH");
                    try putAll(fd, stage.bytes[0..part]);
                    done += part;
                }
            }
            try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, e.carry.ptr, D * 2), "cuMemcpyDtoH");
            try putAll(fd, stage.bytes[0 .. D * 2]);
            if (std.c.fsync(fd) != 0) return error.StateWrite;
        }
        if (std.c.rename(tmp, file) != 0) return error.StateWrite;
        ok = true;
        return @sizeOf(Head) + vs.len * 16 + total + D * 2;
    }

    /// This rank's shard of learned state `ids` read back from `file` as a new shared kept state (not yet an entry:
    /// the caller remembers it once every rank has read its own). A file of another rank, world, DCP, size, key or
    /// identity is refused.
    pub fn readState(s: *Session, ids: []const u32, file: [:0]const u8, key: u64, identity: u64) !*reuse.Kept {
        const pr = s.reuse orelse return error.ReuseOff;
        const b = s.boot;
        const D = b.cfg.hidden;
        const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.StateRead;
        defer _ = std.c.close(fd);
        var head: Head = undefined;
        try getAll(fd, std.mem.asBytes(&head), 0);
        const vs = try pr.views(ids.len);
        defer s.gpa.free(vs);
        var total: u64 = 0;
        for (vs) |v| total += v.row * v.rows;
        if (head.magic != file_magic or head.version != file_version or head.rank != b.o.rank or head.world != b.o.world or head.dcp != b.dcp or head.n != ids.len or head.key != key or head.identity != identity or head.views != vs.len or head.carry != D * 2 or head.bytes != total) return error.StateRefused;
        const sizes = try s.gpa.alloc([2]u64, vs.len);
        defer s.gpa.free(sizes);
        try getAll(fd, std.mem.sliceAsBytes(sizes), @sizeOf(Head));
        for (vs, sizes) |v, z| if (z[0] != v.row or z[1] != v.rows) return error.StateRefused;
        var stage = try cuda.HostBuffer.alloc(&b.driver, 64 << 20);
        defer stage.free();
        const data_at: u64 = @sizeOf(Head) + vs.len * 16;
        const Fill = struct {
            d: *const cuda.Driver,
            fd: c_int,
            at: u64,
            stage: []u8,
            pub fn fill(f: *const @This(), views: []const reuse.View, offs: []const usize, base: u64) !void {
                var pos = f.at;
                for (views, offs) |v, off| {
                    const n = v.row * v.rows;
                    var done: usize = 0;
                    while (done < n) {
                        const part = @min(n - done, f.stage.len);
                        try getAll(f.fd, f.stage[0..part], pos);
                        try f.d.check(f.d.api.cuMemcpyHtoD_v2(base + off + done, f.stage.ptr, part), "cuMemcpyHtoD");
                        done += part;
                        pos += part;
                    }
                }
            }
        };
        const filler: Fill = .{ .d = &b.driver, .fd = fd, .at = data_at, .stage = stage.bytes };
        const carry = try s.gpa.alloc(u8, D * 2);
        defer s.gpa.free(carry);
        try getAll(fd, carry, data_at + total);
        try b.r.f.s.synchronize(); // pooled buffers may still be read by queued copies
        return pr.adopt(ids, carry, &filler);
    }

    /// The learned states' identity probe (prompt_imprint's: a fixed prompt pass whose state is hashed): a 4,143-token
    /// prompt (a full chunk past the sparse index's reach, a middle and a short one) prefilled cold, this rank's rows
    /// and carry hashed, every rank's hashes combined - two builds share learned states only when every rank's prompt
    /// pass gives equal bits. Leaves the live caches holding the probe (nothing kept refers to them).
    pub fn probe(s: *Session) !u64 {
        const b = s.boot;
        var toks: [4096 + 40 + 7]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
        const len = @min(toks.len, b.o.context - 2);
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(s.gpa);
        var cuts = [_]usize{ 4096, 4136 };
        _ = try b.r.run(toks[0..len], .{ .max_tokens = 1, .k = 0, .cuts = &cuts }, &out);
        if (s.reuse) |pr| try pr.store.setLive(&.{});
        const vs = try reuse.rowViews(s.gpa, &b.r.st, b.cfg, len, b.dcp);
        defer s.gpa.free(vs);
        var h = std.hash.Wyhash.init(0x70);
        var stage = try cuda.HostBuffer.alloc(&b.driver, 64 << 20);
        defer stage.free();
        for (vs) |v| {
            const n = v.row * v.rows;
            var done: usize = 0;
            while (done < n) {
                const part = @min(n - done, stage.bytes.len);
                try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, v.ptr + done, part), "cuMemcpyDtoH");
                h.update(stage.bytes[0..part]);
                done += part;
            }
        }
        try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, b.r.carry.ptr, b.cfg.hidden * 2), "cuMemcpyDtoH");
        h.update(stage.bytes[0 .. b.cfg.hidden * 2]);
        h.update(std.mem.sliceAsBytes(out.items));
        const mine = h.final();
        var all: [cuda.rendezvous.max_world * 8]u8 = undefined;
        try b.rdv.allGather(std.mem.asBytes(&mine), all[0 .. b.o.world * 8]);
        return std.hash.Wyhash.hash(0x71, all[0 .. b.o.world * 8]);
    }
};

/// A small file of `dir` (a drafter's config.json).
fn readText(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
}

fn putAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
        if (n <= 0) return error.StateWrite;
        done += @intCast(n);
    }
}

fn getAll(fd: c_int, bytes: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return error.StateRead;
        done += @intCast(n);
    }
}

test "hca names as roce.hca_names" {
    var buf: [2][]const u8 = undefined;
    const got = parseHcas("=mlx5_0:1,^mlx5_1, ,mlx5_2", &buf);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("mlx5_0", got[0]);
    try std.testing.expectEqualStrings("mlx5_1", got[1]);
}

test "DCP turns on past DCP_AUTO unless named" {
    try std.testing.expectEqual(@as(usize, 1), try (Options{ .context = 131072 }).dcpFor());
    try std.testing.expectEqual(@as(usize, 4), try (Options{ .context = 1048576 }).dcpFor());
    try std.testing.expectEqual(@as(usize, 1), try (Options{ .context = 1048576, .dcp = 1 }).dcpFor());
    try std.testing.expectError(error.BadDcp, (Options{ .dcp = 2 }).dcpFor());
}

test "the cache check refuses a context x streams that leaves less than the reserve free (Python's figures)" {
    // 78 layers x 576 x 2 bytes a row; 4 streams x 140K context (~47 GiB) on 50 GiB free: refused, fit ~131K
    const row: u64 = 78 * 576 * 2;
    const gib: u64 = 1 << 30;
    const x = cacheFit(row, 140_000 + 4, 4, 1, 0, 50 * gib, 6).?;
    try std.testing.expect(x.need > 45 * gib);
    try std.testing.expectEqual(@as(u64, 6 * gib), x.spare);
    try std.testing.expect(x.fit < 140_000 and x.fit > 100_000);
    // 4 x 32K fits easily
    try std.testing.expectEqual(@as(?CacheFit, null), cacheFit(row, 32768 + 4, 4, 1, 0, 60 * gib, 6));
    // the reserve is at most half of what is free
    try std.testing.expectEqual(@as(?CacheFit, null), cacheFit(row, 1000, 1, 1, 0, 4 * gib, 6));
}

test "--parallel refusals: DCP, a window wider than a decode window; the drafters draft concurrent streams" {
    try checkParallel(.{ .parallel = 1, .dspark = "/x" }, 4);
    try checkParallel(.{ .parallel = 4, .k = 2 }, 1);
    try std.testing.expectError(error.ParallelNeedsDcpOff, checkParallel(.{ .parallel = 4 }, 4));
    try checkParallel(.{ .parallel = 4, .dspark = "/x" }, 1);
    try checkParallel(.{ .parallel = 2, .dflash = "/x", .mode = "dflash" }, 1);
    try checkParallel(.{ .parallel = 4, .dspark = "/x", .mode = "dspark", .k = 0 }, 1);
    try std.testing.expectError(error.WindowTooWide, checkParallel(.{ .parallel = 16, .k = 2 }, 1));
}

test "Phase 4c: a drafter drafts N streams when N blocks fit one pass" {
    try checkParallelDrafters(1, 64, null);
    try checkParallelDrafters(4, 8, 8);
    try checkParallelDrafters(8, 8, null);
    try std.testing.expectError(error.DrafterTooWide, checkParallelDrafters(9, 8, null));
    try std.testing.expectError(error.DrafterTooWide, checkParallelDrafters(4, null, 17));
}
