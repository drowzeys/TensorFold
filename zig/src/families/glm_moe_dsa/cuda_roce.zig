//! GLM-5.3's decode-window reductions over RoCE (Phase 3a): b12x RoCEnante's one-shot all-reduce and small all-gather
//! (roce.py RoceReduce), in Zig. The proxy is b12x's own C (roce_proxy.c, compiled by Zig against libibverbs); the two
//! kernels are zig/kernels/cuda/glm53_roce.cu (the CuTe DSL kernels in plain CUDA, same protocol and PTX). One pinned
//! host region a rank (cuMemHostAlloc, device-mapped: GB10's GPU reads it in place and the ConnectX-7 RDMA-writes it):
//! recv[src][slot], flag[src][slot][hca], send[slot], ctrl. One launch is one collective; the sequence number lives in
//! device memory (the epoch), so CUDA graphs replay them.
//!
//! The sum is fixed rank order in fp32 (rank 0's value, then + rank 1, ...): the bits of fused.gather's all-gather +
//! glue.residual_add rank-order sum, which `verify` checks against NCCL at startup (roce.py RoceReduce._check: windows
//! of 3 and 32 rows of 6144, every rank's sum bit-equal and equal to the rank-order sum, else every rank refuses).
//! Fail-stop as b12x: a wait past the spin limit poisons the runtime (later launches do nothing); `healthyEverywhere`
//! (roce.py healthy_everywhere, TF_GLM53_ROCE_HEALTH) all-gathers every rank's poison flag over NCCL after each
//! window's picks are read, and every rank fails together before a token from stale sums leaves.
//! Setup runs over the tensor-parallel world's TCP rendezvous (b12x: gloo): every rank's configuration and connection
//! record, then every rank's verdict - all ranks get RoCE or none (start_everywhere).
const std = @import("std");
const cuda = @import("cuda");

// ---- the proxy (roce_proxy.c, ROCE_ABI_VERSION 3) ------------------------------------------------------------------
const Ctx = opaque {};
extern fn roce_abi_version() c_int;
extern fn roce_layout(world: c_int, slot_bytes: u64, out: *[7]u64) c_int;
extern fn roce_blob_bytes() u64;
extern fn roce_create(world: c_int, rank: c_int, names: [*]const [*:0]const u8, n_hca: c_int, gid_index: c_int, region: ?*anyopaque, region_bytes: u64, slot_bytes: u64, err: [*]u8, err_len: u64) ?*Ctx;
extern fn roce_local_blob(c: *Ctx, out: *anyopaque, out_len: u64) c_int;
extern fn roce_connect(c: *Ctx, blobs: *const anyopaque, blobs_len: u64) c_int;
extern fn roce_start(c: *Ctx) c_int;
extern fn roce_stop(c: *Ctx) void;
extern fn roce_failed(c: *Ctx) c_int;
extern fn roce_error(c: *Ctx) [*:0]const u8;
extern fn roce_stat(c: *Ctx, which: c_int) u64;
extern fn roce_destroy(c: ?*Ctx) void;

pub const abi_version: c_int = 3;
pub const max_hcas = 2;
pub const pack_bytes: usize = 16;

/// roce.MAX_BYTES: decode windows up to 32 rows x 6144 fp32 = 768 KB; the head's records and sampled candidates fit.
pub const max_bytes: usize = 1 << 20;

pub const Settings = struct {
    /// RDMA device names in rail order (TF_GLM53_ROCE_HCA / NCCL_IB_HCA: roce.hca_names), at most two
    hcas: []const []const u8,
    /// their RoCE v2 GID index (one for all, as b12x takes it)
    gid_index: u32 = 3,
    /// all-reduce messages and gather shards up to this many bytes (b12x max_size / max_gather_bytes)
    max_bytes: usize = max_bytes,
    /// b12x DEFAULT_THREADS / DEFAULT_BLOCKS / DEFAULT_SPIN_LIMIT (B12X_ROCE_SPIN_LIMIT)
    threads: u32 = 512,
    blocks: u32 = 8,
    spin_limit: u32 = 20_000_000,
};

/// roce_oneshot._grid_blocks: a power-of-two grid with about two 16-byte packs a thread.
pub fn gridBlocks(size_packs: usize, threads: usize, max_blocks: usize) usize {
    const per = 2 * threads;
    const required = @max(1, (size_packs + per - 1) / per);
    return @min(std.math.ceilPowerOfTwo(usize, required) catch max_blocks, max_blocks);
}

/// Bytes `n` rounded up to a multiple of `a`.
fn alignUp(n: usize, a: usize) usize {
    return (n + a - 1) / a * a;
}

pub const Layout = struct { recv: u64, flag: u64, send: u64, ctrl: u64, total: u64, flag_stride: u64, slots: u64 };

pub fn layout(world: usize, slot_bytes: usize) !Layout {
    var o: [7]u64 = undefined;
    if (roce_layout(@intCast(world), slot_bytes, &o) != 0) return error.BadRoceGeometry;
    return .{ .recv = o[0], .flag = o[1], .send = o[2], .ctrl = o[3], .total = o[4], .flag_stride = o[5], .slots = o[6] };
}

/// The configuration every rank must agree on (roce_oneshot's `config` dict), exchanged with the connection record.
const Config = extern struct {
    slot_bytes: u64,
    slots: u64,
    flag_stride: u64,
    max_bytes: u64,
    abi: u32,
    world: u32,
    hcas: u32,
    spin_limit: u32,
    threads: u32,
    blocks: u32,
    ok: u32, // this rank's proxy came up (0: setup failed here)
    pad: u32 = 0, // no padding bytes: the record is compared field by field after the exchange
};

comptime {
    std.debug.assert(@sizeOf(Config) == 64);
}

pub const Roce = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    kernel: cuda.Function,
    region: cuda.HostBuffer, // pinned, device-mapped, zeroed: flags and ctrl start at 0, the first sequence is 1
    dev: u64, // the region's device address (GB10: the host address)
    lay: Layout,
    counters: cuda.DeviceBuffer, // int32 [1 + 2 classes + 1]: epoch, stage[c], tail[c], poison
    classes: usize,
    ctx: ?*Ctx,
    rank: usize,
    world: usize,
    n_hca: usize,
    slot_bytes: usize,
    set: Settings,
    names: [max_hcas][64:0]u8 = @splat(@splat(0)),

    /// Every rank calls it at the same point (the TP world's rendezvous carries the setup); error.RoceUnavailable on
    /// every rank when any rank failed (the caller keeps NCCL, as start_everywhere).
    pub fn start(gpa: std.mem.Allocator, d: *const cuda.Driver, rdv: *cuda.rendezvous.Rendezvous, rank: usize, world: usize, set: Settings) !*Roce {
        if (world < 2 or world > 16) return error.RoceUnavailable;
        const self = try gpa.create(Roce);
        errdefer gpa.destroy(self);
        self.* = .{ .d = d, .module = undefined, .kernel = undefined, .region = undefined, .dev = 0, .lay = undefined, .counters = undefined, .classes = 0, .ctx = null, .rank = rank, .world = world, .n_hca = @min(set.hcas.len, max_hcas), .slot_bytes = alignUp(set.max_bytes, 4096), .set = set };
        var err_buf: [512]u8 = @splat(0);
        const built = if (self.setup(&err_buf)) true else |e| blk: {
            const why = if (err_buf[0] != 0) std.mem.sliceTo(&err_buf, 0) else @errorName(e);
            std.log.err("rank {d}: RoCE setup failed: {s}", .{ rank, why });
            break :blk false;
        };
        errdefer if (built) self.release();
        // every rank's configuration and connection record (b12x: _exchange((error, blob, config)))
        const blob_n: usize = @intCast(roce_blob_bytes());
        const rec_n = @sizeOf(Config) + blob_n;
        const mine = try gpa.alloc(u8, rec_n);
        defer gpa.free(mine);
        @memset(mine, 0);
        var ok = built;
        if (ok and roce_local_blob(self.ctx.?, mine[@sizeOf(Config)..].ptr, blob_n) != 0) ok = false;
        const cfg: Config = .{ .abi = @intCast(roce_abi_version()), .world = @intCast(world), .hcas = @intCast(self.n_hca), .slot_bytes = self.slot_bytes, .slots = if (built) self.lay.slots else 0, .flag_stride = if (built) self.lay.flag_stride else 0, .max_bytes = set.max_bytes, .spin_limit = set.spin_limit, .threads = set.threads, .blocks = set.blocks, .ok = @intFromBool(ok) };
        @memcpy(mine[0..@sizeOf(Config)], std.mem.asBytes(&cfg));
        const all = try gpa.alloc(u8, world * rec_n);
        defer gpa.free(all);
        try rdv.allGather(mine, all);
        var agree = true;
        const ref = std.mem.bytesToValue(Config, all[0..@sizeOf(Config)]);
        for (0..world) |r| {
            const c = std.mem.bytesToValue(Config, all[r * rec_n ..][0..@sizeOf(Config)]);
            if (c.ok == 0) {
                std.log.err("RoCE: rank {d} could not set up its runtime", .{r});
                agree = false;
                continue;
            }
            var cc = c;
            cc.ok = ref.ok;
            if (!std.meta.eql(cc, ref)) {
                std.log.err("RoCE: rank {d}'s configuration differs from rank 0's", .{r});
                agree = false;
            }
        }
        // connect every queue pair from the records in rank order, start the proxy thread, then every rank's verdict
        var connected = agree;
        if (agree) {
            const blobs = try gpa.alloc(u8, world * blob_n);
            defer gpa.free(blobs);
            for (0..world) |r| @memcpy(blobs[r * blob_n ..][0..blob_n], all[r * rec_n + @sizeOf(Config) ..][0..blob_n]);
            if (roce_connect(self.ctx.?, blobs.ptr, blobs.len) != 0 or roce_start(self.ctx.?) != 0) {
                std.log.err("rank {d}: RoCE connect / start failed: {s}", .{ rank, std.mem.span(roce_error(self.ctx.?)) });
                connected = false;
            }
        }
        var v: [1]u8 = .{@intFromBool(connected)};
        const verdicts = try gpa.alloc(u8, world);
        defer gpa.free(verdicts);
        try rdv.allGather(&v, verdicts);
        for (verdicts) |x| {
            if (x == 0) return error.RoceUnavailable; // errdefer: this rank's runtime released
        }
        return self;
    }

    /// Everything `setup` built (the proxy stopped and destroyed first: no NIC write lands in freed memory).
    fn release(self: *Roce) void {
        _ = self.d.api.cuCtxSynchronize();
        self.teardown();
        self.counters.free();
        self.region.free();
        self.module.unload();
    }

    fn setup(self: *Roce, err: *[512]u8) !void {
        const d = self.d;
        if (self.n_hca == 0) return error.NoRdmaDevice;
        self.lay = try layout(self.world, self.slot_bytes);
        self.module = try cuda.Module.load(d, cuda.kernels.glm53_roce);
        errdefer self.module.unload();
        self.kernel = try self.module.function("glm53_roce_oneshot");
        self.region = try cuda.HostBuffer.allocMapped(d, self.lay.total);
        errdefer self.region.free();
        @memset(self.region.bytes, 0);
        self.dev = try self.region.device();
        self.classes = std.math.log2_int(usize, self.set.blocks) + 1;
        self.counters = try cuda.DeviceBuffer.alloc(d, (2 + 2 * self.classes) * 4);
        errdefer self.counters.free();
        try self.counters.fill8(0, null);
        var ptrs: [max_hcas][*:0]const u8 = undefined;
        for (0..self.n_hca) |h| {
            const name = self.set.hcas[h];
            if (name.len >= 64) return error.BadRdmaDevice;
            @memcpy(self.names[h][0..name.len], name);
            self.names[h][name.len] = 0;
            ptrs[h] = &self.names[h];
        }
        self.ctx = roce_create(@intCast(self.world), @intCast(self.rank), &ptrs, @intCast(self.n_hca), @intCast(self.set.gid_index), self.region.bytes.ptr, self.lay.total, self.slot_bytes, err, err.len);
        if (self.ctx == null) return error.RoceProxyFailed;
    }

    fn teardown(self: *Roce) void {
        if (self.ctx) |c| roce_destroy(c);
        self.ctx = null;
    }

    pub fn deinit(self: *Roce, gpa: std.mem.Allocator) void {
        self.release(); // after a device sync: no kernel may still ring the doorbell
        gpa.destroy(self);
    }

    fn ctrlWord(self: *const Roce, i: usize) u32 {
        const off: usize = @intCast(self.lay.ctrl);
        const p: *volatile u32 = @ptrCast(@alignCast(self.region.bytes.ptr + off + 4 * i));
        return p.*;
    }

    /// The proxy stopped on an error, or a kernel's wait timed out (roce_oneshot.poisoned).
    pub fn poisoned(self: *const Roce) bool {
        if (self.ctx) |c| if (roce_failed(c) != 0) return true;
        return self.ctrlWord(2) != 0;
    }

    /// b12x check_health's reason, logged (call when `poisoned`).
    pub fn logHealth(self: *const Roce) void {
        if (self.ctx) |c| if (roce_failed(c) != 0) std.log.err("rank {d}: RoCE proxy failed: {s}", .{ self.rank, std.mem.span(roce_error(c)) });
        const seq = self.ctrlWord(2);
        if (seq != 0) std.log.err("rank {d}: RoCE collective timed out waiting for rank {d}, HCA {d}, at sequence {d}; the runtime is poisoned", .{ self.rank, self.ctrlWord(3), self.ctrlWord(6), seq });
    }

    /// Posted ops, completed writes, last sequence (diagnostics).
    pub fn stats(self: *const Roce) [3]u64 {
        const c = self.ctx orelse return .{ 0, 0, 0 };
        return .{ roce_stat(c, 0), roce_stat(c, 1), roce_stat(c, 2) };
    }

    fn launch(self: *const Roce, s: cuda.Stream, in: u64, out: u64, nbytes: usize, row_packs: usize, mode: c_int) !void {
        if (nbytes == 0 or nbytes % pack_bytes != 0 or nbytes > self.set.max_bytes) return error.BadRoceMessage;
        if (in % pack_bytes != 0 or out % pack_bytes != 0) return error.BadRoceAlignment;
        const packs = nbytes / pack_bytes;
        const blocks = gridBlocks(packs, self.set.threads, self.set.blocks);
        const class = std.math.log2_int(usize, blocks);
        const base = self.counters.ptr;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(c_int, @intCast(packs)));
        a.add(@as(c_int, @intCast(nbytes)));
        a.add(@as(c_int, @intCast(row_packs)));
        a.add(mode);
        a.add(self.dev + self.lay.recv);
        a.add(self.dev + self.lay.flag);
        a.add(self.dev + self.lay.send);
        a.add(self.dev + self.lay.ctrl);
        a.add(@as(u64, self.slot_bytes));
        a.add(base); // epoch
        a.add(base + @as(u64, @intCast(4 * (1 + class)))); // stage arrivals of this grid size
        a.add(base + @as(u64, @intCast(4 * (1 + self.classes + class)))); // tail arrivals
        a.add(base + @as(u64, @intCast(4 * (1 + 2 * self.classes)))); // poison
        a.add(self.set.spin_limit);
        a.add(@as(c_int, @intCast(self.world)));
        a.add(@as(c_int, @intCast(self.rank)));
        a.add(@as(c_int, @intCast(self.n_hca)));
        try cuda.launch.launch(self.kernel, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = self.set.threads } }, s, &a);
    }

    /// out [n fp32] = every rank's in [n] summed in rank order (RoceReduce.all_reduce); nbytes = 4 n.
    pub fn allReduce(self: *const Roce, s: cuda.Stream, in: u64, out: u64, nbytes: usize) !void {
        return self.launch(s, in, out, nbytes, 0, 0);
    }

    /// out [world * nbytes] = every rank's in [nbytes] in rank order (RoceReduce.all_gather, dim 0).
    pub fn allGather(self: *const Roce, s: cuda.Stream, in: u64, out: u64, nbytes: usize) !void {
        return self.launch(s, in, out, nbytes, nbytes / pack_bytes, 1);
    }

    /// roce.healthy_everywhere: this rank's poison flag all-gathered over NCCL (4 bytes), every rank fails together.
    /// `flag` / `every`: int32 device words [1] and [world]; synchronizes `s`.
    pub fn healthyEverywhere(self: *const Roce, comm: *const cuda.nccl.Communicator, s: cuda.Stream, flag: u64, every: u64) !void {
        const mine: u32 = @intFromBool(self.poisoned());
        try self.d.check(self.d.api.cuMemsetD32Async(flag, mine, 1, s.handle), "cuMemsetD32Async");
        try comm.allGather(flag, every, 1, .i32, s.handle);
        var got: [16]u32 = @splat(0);
        try self.d.check(self.d.api.cuMemcpyDtoHAsync_v2(&got, every, self.world * 4, s.handle), "cuMemcpyDtoHAsync");
        try s.synchronize();
        var bad = false;
        for (got[0..self.world], 0..) |v, r| {
            if (v != 0) {
                std.log.err("RoCE reductions failed on rank {d}: tokens of this window dropped", .{r});
                bad = true;
            }
        }
        if (bad) {
            if (mine != 0) self.logHealth();
            return error.RocePoisoned;
        }
    }

    /// RoceReduce._check: windows of 3 and `wide` rows of `d` fp32 (the decode reduction's shapes), every rank's
    /// input scaled 10^(rank - 1); every rank's sum must carry the same bits (compared through NCCL) and the bits of
    /// the rank-order sum of the NCCL-gathered inputs. Returns false on every rank when either fails anywhere.
    pub fn verify(self: *const Roce, gpa: std.mem.Allocator, comm: *const cuda.nccl.Communicator, s: cuda.Stream, d: usize, wide: usize) !bool {
        const world = self.world;
        var same = true;
        var order = true;
        for ([_]usize{ 3, wide }) |rows| {
            const n = rows * d;
            const x = try gpa.alloc(f32, n);
            defer gpa.free(x);
            var rng = std.Random.DefaultPrng.init(1000 + self.rank + rows);
            const scale = std.math.pow(f32, 10.0, @as(f32, @floatFromInt(self.rank)) - 1.0);
            for (x) |*v| v.* = (rng.random().floatNorm(f32)) * scale;
            var bx = try cuda.DeviceBuffer.fromHost(self.d, std.mem.sliceAsBytes(x));
            defer bx.free();
            var bs = try cuda.DeviceBuffer.alloc(self.d, n * 4);
            defer bs.free();
            var ball = try cuda.DeviceBuffer.alloc(self.d, world * n * 4);
            defer ball.free();
            var bparts = try cuda.DeviceBuffer.alloc(self.d, world * n * 4);
            defer bparts.free();
            try self.allReduce(s, bx.ptr, bs.ptr, n * 4);
            try comm.allGather(bs.ptr, ball.ptr, n, .f32, s.handle);
            try comm.allGather(bx.ptr, bparts.ptr, n, .f32, s.handle);
            try s.synchronize();
            const allv = try gpa.alloc(f32, world * n);
            defer gpa.free(allv);
            const parts = try gpa.alloc(f32, world * n);
            defer gpa.free(parts);
            try ball.download(0, std.mem.sliceAsBytes(allv));
            try bparts.download(0, std.mem.sliceAsBytes(parts));
            for (0..n) |i| {
                var ref = parts[i];
                for (1..world) |r| ref = ref + parts[r * n + i];
                const r0: u32 = @bitCast(allv[i]);
                if (r0 != @as(u32, @bitCast(ref))) order = false;
                for (1..world) |r| {
                    if (@as(u32, @bitCast(allv[r * n + i])) != r0) same = false;
                }
            }
        }
        if (self.poisoned()) {
            self.logHealth();
            return false;
        }
        std.log.info("rank {d}: RoCE one-shot reduce on {d} rails (ranks bit-equal: {}; equals the rank-order sum: {}; windows of 3 and {d} rows)", .{ self.rank, self.n_hca, same, order, wide });
        return same and order;
    }
};

/// Microseconds a collective, measured as roce.py's callers see them: `iters` launches captured in one CUDA graph
/// and replayed (decode rounds are graphs), the best of `reps` replays, on every rank in lockstep.
pub const BenchRow = struct { rows: usize, bytes: usize, roce_us: f64, roce_eager_us: f64, nccl_gather_us: f64 };

pub fn bench(r: *const Roce, comm: *const cuda.nccl.Communicator, s: cuda.Stream, d: usize, rows: usize, iters: usize, reps: usize) !BenchRow {
    const dr = r.d;
    const n = rows * d;
    var bx = try cuda.DeviceBuffer.alloc(dr, n * 4);
    defer bx.free();
    try bx.fill32(0x3f800000, s.handle);
    var bo = try cuda.DeviceBuffer.alloc(dr, n * 4);
    defer bo.free();
    var bg = try cuda.DeviceBuffer.alloc(dr, r.world * n * 4);
    defer bg.free();
    var e0 = try cuda.Event.init(dr, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(dr, true);
    defer e1.deinit();
    const Timed = struct {
        fn graphUs(rr: *const Roce, cm: *const cuda.nccl.Communicator, st: cuda.Stream, a: *cuda.Event, b: *cuda.Event, x: u64, o: u64, g: u64, nb: usize, cnt: usize, rp: usize, which: u8) !f64 {
            try st.synchronize();
            try cuda.graph.beginCapture(st, .thread_local);
            var i: usize = 0;
            while (i < cnt) : (i += 1) {
                if (which == 0) try rr.allReduce(st, x, o, nb) else try cm.allGather(x, g, nb / 4, .f32, st.handle);
            }
            var gr = try cuda.graph.endCapture(st);
            defer gr.deinit();
            var ex = try gr.instantiate();
            defer ex.deinit();
            try ex.upload(st);
            try ex.launchOn(st); // warm
            try st.synchronize();
            var best: f64 = std.math.inf(f64);
            for (0..rp) |_| {
                try a.record(st);
                try ex.launchOn(st);
                try b.record(st);
                try b.synchronize();
                const ms = try cuda.Event.elapsedMs(a.*, b.*);
                best = @min(best, @as(f64, ms) * 1000.0 / @as(f64, @floatFromInt(cnt)));
            }
            return best;
        }
    };
    const roce_us = try Timed.graphUs(r, comm, s, &e0, &e1, bx.ptr, bo.ptr, bg.ptr, n * 4, iters, reps, 0);
    const nccl_us = try Timed.graphUs(r, comm, s, &e0, &e1, bx.ptr, bo.ptr, bg.ptr, n * 4, iters, reps, 1);
    // eager launches (no graph): the host's launch cost included
    try s.synchronize();
    try e0.record(s);
    for (0..iters) |_| try r.allReduce(s, bx.ptr, bo.ptr, n * 4);
    try e1.record(s);
    try e1.synchronize();
    const eager = @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / @as(f64, @floatFromInt(iters));
    if (r.poisoned()) {
        r.logHealth();
        return error.RocePoisoned;
    }
    return .{ .rows = rows, .bytes = n * 4, .roce_us = roce_us, .roce_eager_us = eager, .nccl_gather_us = nccl_us };
}

test "grid blocks as roce_oneshot._grid_blocks" {
    try std.testing.expectEqual(@as(usize, 1), gridBlocks(1, 512, 8));
    try std.testing.expectEqual(@as(usize, 1), gridBlocks(1024, 512, 8));
    try std.testing.expectEqual(@as(usize, 2), gridBlocks(1025, 512, 8));
    try std.testing.expectEqual(@as(usize, 8), gridBlocks(6144 * 3 / 4, 512, 8)); // a 3-row window: 4608 packs
    try std.testing.expectEqual(@as(usize, 8), gridBlocks(1 << 20, 512, 8));
}

test "the pinned region's layout (roce_layout)" {
    const l = try layout(4, alignUp(max_bytes, 4096));
    try std.testing.expectEqual(@as(u64, 0), l.recv);
    try std.testing.expectEqual(@as(u64, 4 * 2 * (1 << 20)), l.flag);
    try std.testing.expectEqual(l.flag + 4 * 2 * 2 * 128, l.send);
    try std.testing.expectEqual(l.send + 2 * (1 << 20), l.ctrl);
    try std.testing.expectEqual(l.ctrl + 128, l.total);
    try std.testing.expectError(error.BadRoceGeometry, layout(1, 4096));
}
