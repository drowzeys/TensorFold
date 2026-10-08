//! GLM-5.3's CUDA kernels: the EXL3 linear and routed-expert fatbins (our device-only copies of cuda/exl3) with the
//! Python launchers' geometry, and the captured Triton set (tools/glm53/oracle.py --record). Phase 2b: glm53_head.cu
//! (the torch steps of a decode window on the device) and qmm_group's fp32-out tile for the 4-bit draft head.
const std = @import("std");
const cuda = @import("cuda");
const exl3 = @import("exl3.zig");

const n_k2 = exl3.linear_k2s.len;

pub const Kernels = struct {
    d: *const cuda.Driver,
    linear_mod: cuda.Module,
    experts_mod: cuda.Module,
    triton: cuda.aot.Set,
    rot_in: cuda.Function,
    /// linear_kernel<K2, 2 (mul1), WK, G, V> by [K2 index][WK 2 / 4 / 8][G 1 / 2]
    linear: [n_k2][3][2]cuda.Function,
    /// Phase 3a: unpack_kernel<K2, 2> by K2 index (x3prefill.matmul's W_q decode)
    unpack: [n_k2]cuda.Function,
    prompt_mod: cuda.Module, // glm53_prompt.cu: transposed copies, bf16 -> fp32, add_, the Hadamard matrix
    bhx_to_hbx: cuda.Function,
    bf16_f32: cuda.Function,
    add_f32: cuda.Function,
    hadamard: cuda.Function,
    pe_mod: cuda.Module, // glm53_prompt_experts.cu (prompt_experts.cu's device part)
    pe_route: cuda.Function,
    pe_gateup: cuda.Function,
    pe_down: cuda.Function,
    pe_slot_sum16: cuda.Function,
    pe: PeConsts,
    group: cuda.Function,
    expert_rot_in: cuda.Function,
    gateup_epilogue: cuda.Function,
    /// grouped_kernel<2, 8, 4, 1, 1, LO, HI> for (8, 8), (2, 10), (2, 16)
    grouped: [3]cuda.Function,
    rope_mod: cuda.Module,
    inv_freq: cuda.Function, // glm53_rope.cu: fused.inv_freq as torch computes it
    head_mod: cuda.Module,
    argmax_rec: cuda.Function,
    pick_rank: cuda.Function,
    embed_rows: cuda.Function,
    f32_bf16: cuda.Function,
    qmm_mod: cuda.Module,
    q4_group: cuda.Function, // qmm_group tile 2 with fp32 out: the draft head (headq q4)
    gb10: bool, // sm_121: qmm_group launches with programmatic dependent launch (qmm_group_cuda's `early`)

    /// Loads both fatbins and resolves every kernel; `triton_dir` holds the captured aot.json and cubins.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, triton_dir: []const u8) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        k.linear_mod = try cuda.Module.load(d, cuda.kernels.glm53_exl3_linear);
        errdefer k.linear_mod.unload();
        k.experts_mod = try cuda.Module.load(d, cuda.kernels.glm53_exl3_experts);
        errdefer k.experts_mod.unload();
        k.rot_in = try k.linear_mod.function(exl3.rot_in_symbol);
        var buf: [256]u8 = undefined;
        for (exl3.linear_k2s, 0..) |k2, i| for ([_]usize{ 2, 4, 8 }, 0..) |wk, j| for ([_]usize{ 1, 2 }, 0..) |g, h| {
            k.linear[i][j][h] = try k.linear_mod.function(try exl3.linearSymbol(&buf, k2, .mul1, wk, g, exl3.vecOk(k2)));
        };
        for (exl3.linear_k2s, 0..) |k2, i| k.unpack[i] = try k.linear_mod.function(try exl3.unpackSymbol(&buf, k2, .mul1));
        k.group = try k.experts_mod.function(exl3.group_symbol);
        k.expert_rot_in = try k.experts_mod.function(exl3.expert_rot_in_symbol);
        k.gateup_epilogue = try k.experts_mod.function(exl3.gateup_epilogue_symbol);
        for ([_][2]usize{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } }, 0..) |r, i| {
            k.grouped[i] = try k.experts_mod.function(try exl3.groupedSymbol(&buf, .mul1, r[0], r[1]));
        }
        k.rope_mod = try cuda.Module.load(d, cuda.kernels.glm53_rope);
        errdefer k.rope_mod.unload();
        k.inv_freq = try k.rope_mod.function("glm53_inv_freq");
        k.head_mod = try cuda.Module.load(d, cuda.kernels.glm53_head);
        errdefer k.head_mod.unload();
        k.argmax_rec = try k.head_mod.function("glm53_argmax_rec");
        k.pick_rank = try k.head_mod.function("glm53_pick_rank");
        k.embed_rows = try k.head_mod.function("glm53_embed_rows");
        k.f32_bf16 = try k.head_mod.function("glm53_f32_bf16");
        k.qmm_mod = try cuda.Module.load(d, cuda.kernels.qmm_group);
        errdefer k.qmm_mod.unload();
        k.q4_group = try k.qmm_mod.function(q4_group_symbol);
        try k.q4_group.allowDynamicShared(q4_group_smem);
        k.gb10 = (try ctx.capability()) == 121;
        // Phase 3a: the prompt path's own kernels and the device copy of prompt_experts.cu
        k.prompt_mod = try cuda.Module.load(d, cuda.kernels.glm53_prompt);
        errdefer k.prompt_mod.unload();
        k.bhx_to_hbx = try k.prompt_mod.function("glm53_bhx_to_hbx");
        k.bf16_f32 = try k.prompt_mod.function("glm53_bf16_f32");
        k.add_f32 = try k.prompt_mod.function("glm53_add_f32");
        k.hadamard = try k.prompt_mod.function("glm53_hadamard");
        k.pe_mod = try cuda.Module.load(d, cuda.kernels.glm53_prompt_experts);
        errdefer k.pe_mod.unload();
        k.pe_route = try k.pe_mod.function(exl3.pe_route_symbol);
        k.pe_gateup = try k.pe_mod.function(exl3.pe_gateup_symbol);
        k.pe_down = try k.pe_mod.function(exl3.pe_down_symbol);
        k.pe_slot_sum16 = try k.pe_mod.function(exl3.pe_slot_sum16_symbol);
        k.pe = try PeConsts.read(d, k.pe_mod);
        try k.pe_gateup.allowDynamicShared(k.pe.gu_smem);
        try k.pe_down.allowDynamicShared(k.pe.dn_smem);
        k.triton = try cuda.aot.Set.load(gpa, io, d, ctx.device, triton_dir);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.triton.deinit();
        k.pe_mod.unload();
        k.prompt_mod.unload();
        k.qmm_mod.unload();
        k.head_mod.unload();
        k.rope_mod.unload();
        k.experts_mod.unload();
        k.linear_mod.unload();
    }

    /// fused.inv_freq(dim, theta): fp32 [dim / 2] into `out` on `s`, the bytes torch computes on the GPU
    /// (glm53_rope.cu: 1 / powf(theta, 2i * (1 / dim))).
    pub fn invFreq(k: *const Kernels, s: cuda.Stream, out: u64, dim: usize, theta: f64) !void {
        const half = dim / 2;
        var a: cuda.Args = .{};
        a.add(out);
        a.add(int(half));
        a.add(int(dim));
        a.add(@as(f32, 1.0) / @as(f32, @floatFromInt(dim)));
        a.add(@as(f32, @floatCast(theta)));
        try cuda.launch.launch(k.inv_freq, .{ .grid = .{ .x = u((half + 127) / 128) }, .block = .{ .x = 128 } }, s, &a);
    }

    fn linearFn(k: *const Kernels, k2: usize, wk: usize, g: usize) !cuda.Function {
        const i = std.mem.indexOfScalar(usize, &exl3.linear_k2s, k2) orelse return error.UnsupportedExl3Width;
        const j: usize = switch (wk) {
            2 => 0,
            4 => 1,
            8 => 2,
            else => return error.UnsupportedExl3Warps,
        };
        return k.linear[i][j][g - 1];
    }

    fn groupedFn(k: *const Kernels, lo: usize, hi: usize) cuda.Function {
        const r = exl3.k2Range(lo, hi);
        if (r[0] == 8) return k.grouped[0];
        if (r[1] == 10) return k.grouped[1];
        return k.grouped[2];
    }
};

/// prompt_experts.cu's launch constants (glm53_pe_consts): dynamic shared memory and threads of its two GEMM kernels,
/// routed pairs an item.
pub const PeConsts = struct {
    gu_smem: u32,
    dn_smem: u32,
    gu_threads: u32,
    dn_threads: u32,
    ipm: u32,

    fn read(d: *const cuda.Driver, m: cuda.Module) !PeConsts {
        const f = try m.function("glm53_pe_consts");
        var buf = try cuda.DeviceBuffer.alloc(d, 8 * 4);
        defer buf.free();
        var a: cuda.Args = .{};
        a.add(buf.ptr);
        const s: cuda.Stream = .{ .d = d, .handle = null };
        try cuda.launch.launch(f, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, s, &a);
        try s.synchronize();
        var v: [8]i32 = undefined;
        try buf.download(0, std.mem.sliceAsBytes(&v));
        return .{ .gu_smem = @intCast(v[0]), .dn_smem = @intCast(v[1]), .gu_threads = @intCast(v[2]), .dn_threads = @intCast(v[3]), .ipm = @intCast(v[4]) };
    }
};

/// prompt_experts.PromptScratch for `rows` rows of `slots` slots (TF_EXL3_PROMPT_DET=slots16: fp16 pair rows, summed
/// in slot order), plus the fp32 sums and their bf16 rounding (SharedExperts._prefill_tf's bf16 `out`).
pub const PromptExpertScratch = struct {
    rows: usize,
    slots: usize,
    max_items: usize,
    sorted: u64, // int32 [rows * slots]
    items: u64, // int32 [3 * max_items]
    count: u64, // int32 [1]
    xd: u64, // fp16 [rows * slots, I]
    pairs: u64, // fp16 [rows * slots, D] (mode 4) / unused in mode 0
    acc: u64, // fp32 [rows, D]
    out16: u64, // bf16 [rows, D]
};

/// qmm_group.cu's group_kernel<64, 16, 64, 1, 4, 8, F32 = true> (tile 2, the instantiation added for the draft head).
pub const q4_group_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
/// LaneTile<64, 16, 64, 1, 4, 8>::SMEM (Nemotron's group_smem: the same tile).
pub const q4_group_smem: u32 = 35328;

/// qmm_group.cu's Part and Parts, passed by value (Nemotron's cuda_kernels.zig has the same layout).
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232);
}

/// Device scratch the EXL3 launches write before they read it (fresh torch.empty tensors in Python).
pub const LinearScratch = struct {
    xh: [exl3.max_jobs]u64, // fp16 [rows, K] a job
    z: [exl3.max_jobs]u64, // fp32 [SK, rows, N] a job (split-K partials)
    xh_bytes: usize,
    z_bytes: usize,
};

/// The routed experts of one layer: trellis pointer tables and widths per projection, stacked scales (experts.py).
pub const Experts = struct {
    gate_ptr: u64, // int64 [E]
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]
    svh_d: u64, // fp16 [E, D]
    count: usize,
    dims: usize,
    width: usize,
    k2_gu: [2]usize, // (min, max)
    k2_d: [2]usize,
};

/// experts.Scratch for `rows` rows of `slots` slots.
pub const ExpertScratch = struct {
    xg: u64,
    xu: u64,
    xd: u64,
    z: u64,
    y: u64,
    ids: u64,
    count: u64,
    members: u64, // int32 [maxu * rows], -1 at first
    done_gu: u64,
    done_d: u64,
    rows: usize,
    slots: usize,
    gu: exl3.Tiles,
    dn: exl3.Tiles,
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// Launch helpers on one stream, each mirroring the Python wrapper (exl3 linear.py / experts.py) it replaces.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    /// x3linear.group / Exl3Linear.__call__: rot_in, then one linear launch for 1 to 3 linears of input x [m, K].
    /// outs[j] [m, N_j] of dtype ys[j]; the launch gives each output the bits of its own call.
    pub fn linears(o: Ops, lins: []const *const exl3.Linear, x: u64, x_dtype: exl3.DType, outs: []const u64, ys: []const exl3.DType, m: usize, sc: LinearScratch) !void {
        if (lins.len == 0 or lins.len > exl3.max_jobs or outs.len != lins.len or ys.len != lins.len) return error.Invalid;
        if (m < 1 or m > 128) return error.WindowTooWide;
        if (!exl3.groupable(lins)) {
            // x3linear.group falls back to one call a linear
            for (lins, outs, ys) |l, out, y| try o.launchGroup(&.{l}, x, x_dtype, &.{out}, &.{y}, m, sc);
            return;
        }
        try o.launchGroup(lins, x, x_dtype, outs, ys, m, sc);
    }

    fn launchGroup(o: Ops, lins: []const *const exl3.Linear, x: u64, x_dtype: exl3.DType, outs: []const u64, ys: []const exl3.DType, m: usize, sc: LinearScratch) !void {
        const a = lins[0];
        const kk = a.k;
        const pdl = ((exl3.linear_loads >> 5) & 1) == 1;
        var rj: exl3.RotJobs = std.mem.zeroes(exl3.RotJobs);
        for (lins, 0..) |l, i| {
            if (m * kk * 2 > sc.xh_bytes) return error.ScratchTooSmall;
            rj.suh[i] = l.suh;
            rj.xh[i] = sc.xh[i];
        }
        var ra: cuda.Args = .{};
        ra.add(x);
        ra.add(@as(c_int, @intFromEnum(x_dtype)));
        ra.add(rj);
        ra.add(int(kk));
        try cuda.launch.launch(o.k.rot_in, .{ .grid = .{ .x = u((kk / 128 + 3) / 4), .y = u(m), .z = u(lins.len) }, .block = .{ .x = 128 }, .pdl = pdl }, o.s, &ra);

        var jobs: exl3.Jobs = std.mem.zeroes(exl3.Jobs);
        jobs.n = int(lins.len);
        var blocks: usize = 0;
        for (lins, outs, ys, 0..) |l, out, y, i| {
            if ((kk / 16) % (l.sk * a.wk) != 0) return error.BadSplit;
            if (l.sk > 1 and l.sk * m * l.n * 4 > sc.z_bytes) return error.ScratchTooSmall;
            const st = l.strides();
            jobs.j[i] = .{
                .xh = sc.xh[i],
                .t = l.words,
                .stride_k = st[0],
                .stride_nb = st[1],
                .svh = l.svh,
                .bias = 0,
                .y = out,
                .z = if (l.sk > 1) sc.z[i] else 0,
                .counters = l.counters,
                .y_dtype = @intFromEnum(y),
                .n = int(l.n),
                .sk = int(l.sk),
                .start = int(blocks),
            };
            blocks += (l.n / 128) * l.sk;
        }
        const v = exl3.vecOk(a.k2) and (exl3.linear_loads & 3) == 3;
        if (!v and exl3.vecOk(a.k2)) return error.UnsupportedLoads;
        const g = exl3.linearG(m);
        const f = try o.k.linearFn(a.k2, a.wk, g);
        const smem = exl3.linearSmem(a.wk, m, a.k2, v);
        if (smem > 48 * 1024) try f.allowDynamicShared(u(smem));
        var la: cuda.Args = .{};
        la.add(jobs);
        la.add(int(m));
        la.add(int(kk));
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(blocks) }, .block = .{ .x = u(a.wk * 32) }, .shared = u(smem), .pdl = pdl }, o.s, &la);
    }

    pub fn linear(o: Ops, l: *const exl3.Linear, x: u64, x_dtype: exl3.DType, out: u64, y: exl3.DType, m: usize, sc: LinearScratch) !void {
        return o.linears(&.{l}, x, x_dtype, &.{out}, &.{y}, m, sc);
    }

    /// experts.routed with wts and sy (TF_EXL3_EXPERTS_FUSE 1, LOADS 1, PDL 0, fp32 SwiGLU, no limit):
    /// out [R, D] fp32 = the wts-weighted sum of the routed slots' outputs + sy.
    pub fn routed(o: Ops, x: u64, x_stride: usize, pick: u64, wts: u64, ex: *const Experts, s: ExpertScratch, out: u64, sy: u64, rows: usize) !void {
        if (rows > s.rows) return error.WindowTooWide;
        const D = ex.dims;
        const I = ex.width;
        const E = ex.count;
        const slots = s.slots;
        const P = rows * slots;
        const maxu = @min(P, E);
        const maxm = rows; // members view [maxu, R]
        // 1. grouping: distinct experts in id order, their members row * 32 + slot
        var ga: cuda.Args = .{};
        for ([_]u64{ pick, s.ids, s.count, s.members }) |p| ga.add(p);
        for ([_]usize{ rows, slots, E, maxm }) |v| ga.add(int(v));
        try cuda.launch.launch(o.k.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = u(P * 4) }, o.s, &ga);
        // 2. Xh of gate and up for every routed slot
        var ra: cuda.Args = .{};
        ra.add(x);
        ra.add(int(x_stride));
        ra.add(pick);
        for ([_]u64{ ex.suh_g, ex.suh_u, s.xg, s.xu }) |p| ra.add(p);
        for ([_]usize{ D, slots, E }) |v| ra.add(int(v));
        try cuda.launch.launch(o.k.expert_rot_in, .{ .grid = .{ .x = u(P), .y = u(D / 128), .z = 2 }, .block = .{ .x = 32 } }, o.s, &ra);
        // 3. gate and up: split-K partials into Z (fuse 0)
        const gu = s.gu;
        try o.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, s, 2, D, I, P, gu, maxu, maxm, ex.k2_gu, .{});
        // 4. the gate/up epilogue: splits summed in order, SwiGLU (fp32), Xd for down
        var ea: cuda.Args = .{};
        for ([_]u64{ s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd }) |p| ea.add(p);
        for ([_]usize{ P, I, gu.sk, E }) |v| ea.add(int(v));
        ea.add(std.math.inf(f32));
        ea.add(exl3.act_f32);
        try cuda.launch.launch(o.k.gateup_epilogue, .{ .grid = .{ .x = u(P), .y = u(I / 128) }, .block = .{ .x = 32 } }, o.s, &ea);
        // 5. down with its epilogue and the weighted combine (+ sy) fused
        const dn = s.dn;
        if (dn.nt != 8 or dn.sk != 1) return error.UnfusedDownUnsupported; // TODO: down_combine path (other tiles)
        try o.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, s, 1, I, D, P, dn, maxu, maxm, ex.k2_d, .{
            .fuse = 1,
            .pick = pick,
            .e = int(E),
            .limit = 0.0,
            .act_mode = exl3.act_f32,
            .svh_d = ex.svh_d,
            .y = s.y,
            .wts = wts,
            .out = out,
            .sy = sy,
            .done = s.done_d,
        });
    }

    /// grouped_launch: grid (distinct experts bound, N / (16 nt), mats * SK * member tiles), ns 0 (contiguous trellises).
    fn grouped(o: Ops, x0: u64, x1: u64, tp0: u64, tp1: u64, k20: u64, k21: u64, s: ExpertScratch, mats: usize, K: usize, N: usize, P: usize, t: exl3.Tiles, maxu: usize, maxm: usize, k2r: [2]usize, ep: exl3.Epi) !void {
        if (t.nt != 8 or t.w != 4) return error.UnsupportedExpertTiles; // GLM's tiles: the 16-byte load path (V 1)
        if (K % (16 * t.sk * t.w) != 0 or N % (16 * t.nt) != 0) return error.BadSplit;
        const mt = (maxm + 15) / 16;
        var a: cuda.Args = .{};
        for ([_]u64{ x0, x1, tp0, tp1, k20, k21, s.ids, s.count, s.members, s.z }) |p| a.add(p);
        for ([_]usize{ K, N, P, t.sk, maxm, s.slots, 0 }) |v| a.add(int(v));
        a.add(ep);
        const f = o.k.groupedFn(k2r[0], k2r[1]);
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(maxu), .y = u(N / (16 * t.nt)), .z = u(mats * t.sk * mt) }, .block = .{ .x = u(t.w * 32) } }, o.s, &a);
    }

    /// glm53_argmax_rec: rows x [V] fp32 (row stride ld floats) -> amax [rows, 4] records (value, float id); ids
    /// are idmap[i] (int64 table) when idmap != 0, else i + off.
    pub fn argmaxRec(o: Ops, lg: u64, V: usize, ld: usize, idmap: u64, off: usize, amax: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(lg);
        a.add(int(V));
        a.add(int(ld));
        a.add(idmap);
        a.add(@as(i64, @intCast(off)));
        a.add(amax);
        try cuda.launch.launch(o.k.argmax_rec, .{ .grid = .{ .x = u(rows) }, .block = .{ .x = 1024 } }, o.s, &a);
    }

    /// glm53_pick_rank: g [world, rows, 4] -> out int64 [rows] (the lowest rank among equal maxima).
    pub fn pickRank(o: Ops, g: u64, world: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(g);
        a.add(int(world));
        a.add(int(rows));
        a.add(out);
        try cuda.launch.launch(o.k.pick_rank, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, o.s, &a);
    }

    /// glm53_embed_rows: out[r] = table[ids[r]] (bf16 rows of D) for r < rows; zero_first: row 0 zeroed.
    pub fn embedRows(o: Ops, table: u64, ids: u64, D: usize, out: u64, rows: usize, zero_first: bool) !void {
        var a: cuda.Args = .{};
        a.add(table);
        a.add(ids);
        a.add(int(D));
        a.add(out);
        a.add(@as(c_int, @intFromBool(zero_first)));
        try cuda.launch.launch(o.k.embed_rows, .{ .grid = .{ .x = u(rows) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// glm53_f32_bf16: y[:n] = bfloat16(x[:n]) (Tensor.copy_).
    pub fn f32ToBf16(o: Ops, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(int(n));
        try cuda.launch.launch(o.k.f32_bf16, .{ .grid = .{ .x = u((n + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// shared qmm.matmul on sm_12x for the draft head (qmm_group_cuda, tile 2, f32): x [rows <= 16, K] bf16 with its
    /// 64-input group sums xs -> out [rows, n] fp32; a cluster of `sk` CTAs a 64-column tile.
    pub fn q4Head(o: Ops, x: u64, xs: u64, w: u64, scales: u64, biases: u64, n: usize, npad: usize, K: usize, sk: usize, out: u64, rows: usize) !void {
        if (rows > 16) return error.WindowTooWide;
        const tiles = (n + 63) / 64;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = w, .scales = scales, .biases = biases, .out = out, .n = int(n), .npad = int(npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        const rows_t = (rows + 15) / 16;
        const clusters = rows_t * tiles; // C == sk: one tile a cluster
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(parts);
        for ([_]usize{ rows, K, K, rows_t, sk }) |v| a.add(int(v)); // M, K, ldx (contiguous rows), row tiles, C
        try cuda.launch.launch(o.k.q4_group, .{
            .grid = .{ .x = u(clusters * sk) },
            .block = .{ .x = 128 },
            .shared = q4_group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = o.k.gb10,
        }, o.s, &a);
    }

    // ---- Phase 3a: the prompt path ----------------------------------------------------------------------------

    /// x3prefill.matmul's ext.rot_in(x, [suh], [xh], pdl=0): xh fp16 [m, K] = fp16(((x suh) H) / sqrt(128)).
    pub fn rotIn(o: Ops, x: u64, x_dtype: exl3.DType, suh: u64, xh: u64, m: usize, kk: usize) !void {
        var rj: exl3.RotJobs = std.mem.zeroes(exl3.RotJobs);
        rj.suh[0] = suh;
        rj.xh[0] = xh;
        var ra: cuda.Args = .{};
        ra.add(x);
        ra.add(@as(c_int, @intFromEnum(x_dtype)));
        ra.add(rj);
        ra.add(int(kk));
        try cuda.launch.launch(o.k.rot_in, .{ .grid = .{ .x = u((kk / 128 + 3) / 4), .y = u(m), .z = 1 }, .block = .{ .x = 128 } }, o.s, &ra);
    }

    /// Exl3Linear.unpack (ext.unpack): W_q [K, N] fp16 decoded from the strips words, one warp a 16 x 16 tile.
    pub fn unpack(o: Ops, l: *const exl3.Linear, w: u64) !void {
        const i = std.mem.indexOfScalar(usize, &exl3.linear_k2s, l.k2) orelse return error.UnsupportedExl3Width;
        if (l.cb != .mul1) return error.UnsupportedExl3Codebook;
        const st = l.strides();
        var a: cuda.Args = .{};
        a.add(l.words);
        a.add(w);
        a.add(int(l.n));
        a.add(st[0]);
        a.add(st[1]);
        try cuda.launch.launch(o.k.unpack[i], .{ .grid = .{ .x = u(l.n / 16), .y = u(l.k / 16) }, .block = .{ .x = 32 } }, o.s, &a);
    }

    /// out [R, H, X] = in [H, R, X] (bf16): the copy_ of a torch.bmm result into fused's row-major view.
    pub fn bhxToHbx(o: Ops, in: u64, out: u64, H: usize, R: usize, X: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(int(H));
        a.add(int(R));
        a.add(int(X));
        try cuda.launch.launch(o.k.bhx_to_hbx, .{ .grid = .{ .x = u(R), .y = u(H) }, .block = .{ .x = 128 } }, o.s, &a);
    }

    /// out[:n] = float(in[:n]) (bf16 -> fp32, exact).
    pub fn bf16ToF32(o: Ops, in: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(o.k.bf16_f32, .{ .grid = .{ .x = u((n + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// a[:n] += b[:n] (fp32: Tensor.add_).
    pub fn addF32(o: Ops, a_ptr: u64, b_ptr: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(a_ptr);
        a.add(b_ptr);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(o.k.add_f32, .{ .grid = .{ .x = u((n + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// x3prefill.Workspace.hadamard: the 128 x 128 +-1 matrix (bf16) the prompt GEMM's epilogue multiplies by.
    pub fn hadamard(o: Ops, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(out);
        try cuda.launch.launch(o.k.hadamard, .{ .grid = .{ .x = 128 }, .block = .{ .x = 128 } }, o.s, &a);
    }

    /// prompt_experts.supported: gate and up at one width, widths 2 / 3 / 4 bits, I % 128, D % 256.
    pub fn promptSupported(ex: *const Experts) bool {
        const ok = struct {
            fn k2(r: [2]usize) bool {
                return r[0] >= 4 and r[1] <= 8 and r[0] % 2 == 0;
            }
        };
        return ok.k2(ex.k2_gu) and ok.k2(ex.k2_d) and ex.width % 128 == 0 and ex.dims % 256 == 0;
    }

    /// SharedExperts._prefill_tf -> prompt_experts.prompt_routed (TF_EXL3_PROMPT_DET=slots16, act fp32, no limit,
    /// codebook 2): the routed experts of rows x [R, D] (bf16, row stride x_stride) with picks [R, S] int32 and
    /// weights [R, S] fp32 -> out [R, D] fp32 = float(bf16(the slots' outputs summed in slot order)) (b.part.copy_ of
    /// the bf16 result). Chunks of up to CHUNK_ROWS rows; each row's bits never depend on its chunk.
    pub fn promptRouted(o: Ops, x: u64, x_stride: usize, pick: u64, wts: u64, ex: *const Experts, sc: PromptExpertScratch, out: u64, R: usize, det16: bool) !void {
        const D = ex.dims;
        const I = ex.width;
        const S = sc.slots;
        const E = ex.count;
        const pe = o.k.pe;
        var r0: usize = 0;
        const step = @min(sc.rows, exl3.pe_chunk_rows);
        while (r0 < R) : (r0 += step) {
            const n = @min(step, R - r0);
            const pairs_n = n * S;
            const items = @min(sc.max_items, pairs_n / pe.ipm + @min(pairs_n, E));
            const xr = x + r0 * x_stride * 2;
            const pk = pick + r0 * S * 4;
            const wr = wts + r0 * S * 4;
            // route: one block, pairs grouped by expert into items of <= IPM
            var ra: cuda.Args = .{};
            ra.add(pk);
            ra.add(int(pairs_n));
            ra.add(int(E));
            ra.add(sc.sorted);
            ra.add(sc.items);
            ra.add(sc.count);
            ra.add(int(items));
            ra.add(@as(c_int, 0)); // TF_EXL3_PROMPT_ORDER 0
            try cuda.launch.launch(o.k.pe_route, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .shared = u(3 * E * 4) }, o.s, &ra);
            const mode: c_int = if (det16) 4 else 0;
            const dst = if (det16) sc.pairs else sc.acc;
            // gate|up (+ SwiGLU and down's input rotation into xd; mode 0 also zeroes the fp32 sums)
            var ga: cuda.Args = .{};
            ga.add(xr);
            ga.add(int(x_stride));
            ga.add(sc.sorted);
            ga.add(sc.items);
            ga.add(sc.count);
            for ([_]u64{ ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.suh_g, ex.suh_u, ex.svh_g, ex.svh_u, ex.suh_d, sc.xd }) |p| ga.add(p);
            for ([_]usize{ D, I, I / 16, S }) |v| ga.add(int(v));
            ga.add(std.math.inf(f32));
            ga.add(exl3.act_f32);
            ga.add(dst);
            ga.add(@as(i64, if (det16) 0 else @intCast(n * D * 4 / 16)));
            try cuda.launch.launch(o.k.pe_gateup, .{ .grid = .{ .x = u(I / 128), .y = u(items) }, .block = .{ .x = pe.gu_threads }, .shared = pe.gu_smem }, o.s, &ga);
            // down (+ svh, routing weight): mode 4 a pair's own fp16 row, mode 0 red.add into the fp32 sums
            const ncb = exl3.autoNcb(n, D, 4);
            var da: cuda.Args = .{};
            for ([_]u64{ sc.xd, sc.sorted, sc.items, sc.count, ex.down_ptr, ex.down_k2, ex.svh_d, wr, dst }) |p| da.add(p);
            for ([_]usize{ I, D, S, ncb }) |v| da.add(int(v));
            da.add(mode);
            try cuda.launch.launch(o.k.pe_down, .{ .grid = .{ .x = u(items), .y = u(D / 256 / ncb) }, .block = .{ .x = pe.dn_threads }, .shared = pe.dn_smem }, o.s, &da);
            if (det16) {
                // slot_sum16: a row's fp16 pair rows added in fp32 in slot order (picks outside [0, E) skipped)
                const n4 = n * D / 4;
                var sa: cuda.Args = .{};
                sa.add(sc.pairs);
                sa.add(pk);
                sa.add(sc.acc);
                sa.add(@as(i64, @intCast(n4)));
                sa.add(int(S));
                sa.add(int(D / 4));
                sa.add(int(E));
                try cuda.launch.launch(o.k.pe_slot_sum16, .{ .grid = .{ .x = u((n4 + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &sa);
            }
            // out.copy_(fp32 sums) into the bf16 result, then b.part.copy_(bf16) into fp32
            try o.f32ToBf16(sc.acc, sc.out16, n * D);
            try o.bf16ToF32(sc.out16, out + r0 * D * 4, n * D);
        }
    }

    pub fn copy(o: Ops, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, o.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn upload(o: Ops, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, o.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn download(o: Ops, dst: []u8, src: u64) !void {
        if (dst.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, o.s.handle), "cuMemcpyDtoHAsync");
    }

    pub fn fill32(o: Ops, dst: u64, value: u32, words: usize) !void {
        if (words == 0) return;
        try o.k.d.check(o.k.d.api.cuMemsetD32Async(dst, value, words, o.s.handle), "cuMemsetD32Async");
    }
};
