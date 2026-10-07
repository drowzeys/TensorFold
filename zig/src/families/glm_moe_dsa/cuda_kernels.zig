//! GLM-5.3's CUDA kernels: the EXL3 linear and routed-expert fatbins (our device-only copies of cuda/exl3) with the
//! Python launchers' geometry, and the captured Triton set (tools/glm53/oracle.py --record).
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
    /// linear_kernel<K2, 2 (mul1), WK, G, V> by [K2 index][WK 4 / 8][G 1 / 2]
    linear: [n_k2][2][2]cuda.Function,
    group: cuda.Function,
    expert_rot_in: cuda.Function,
    gateup_epilogue: cuda.Function,
    /// grouped_kernel<2, 8, 4, 1, 1, LO, HI> for (8, 8), (2, 10), (2, 16)
    grouped: [3]cuda.Function,

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
        for (exl3.linear_k2s, 0..) |k2, i| for ([_]usize{ 4, 8 }, 0..) |wk, j| for ([_]usize{ 1, 2 }, 0..) |g, h| {
            k.linear[i][j][h] = try k.linear_mod.function(try exl3.linearSymbol(&buf, k2, .mul1, wk, g, exl3.vecOk(k2)));
        };
        k.group = try k.experts_mod.function(exl3.group_symbol);
        k.expert_rot_in = try k.experts_mod.function(exl3.expert_rot_in_symbol);
        k.gateup_epilogue = try k.experts_mod.function(exl3.gateup_epilogue_symbol);
        for ([_][2]usize{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } }, 0..) |r, i| {
            k.grouped[i] = try k.experts_mod.function(try exl3.groupedSymbol(&buf, .mul1, r[0], r[1]));
        }
        k.triton = try cuda.aot.Set.load(gpa, io, d, ctx.device, triton_dir);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.triton.deinit();
        k.experts_mod.unload();
        k.linear_mod.unload();
    }

    fn linearFn(k: *const Kernels, k2: usize, wk: usize, g: usize) !cuda.Function {
        const i = std.mem.indexOfScalar(usize, &exl3.linear_k2s, k2) orelse return error.UnsupportedExl3Width;
        const j: usize = switch (wk) {
            4 => 0,
            8 => 1,
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
