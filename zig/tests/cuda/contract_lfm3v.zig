//! lfm3v.cu under the contract (LFM3-VL's tower glue, d1's set attention): RMS norm and q/k rows here, each controlled.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const pointwise = @import("contract_lfm3v_pointwise.zig");
const conv_k = @import("contract_lfm3v_conv.zig");
const set_k = @import("contract_lfm3v_set.zig");
const vit_k = @import("contract_lfm3v_vit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;

/// lfm3_vision.zig's config: hidden 1024, 16 heads of 64, eps 1e-5.
pub const eps: f32 = 1e-5;
const dim = 64;

/// Gate 3's rule of every exact rig here: the host's replay of the kernel's fp32 and bf16 steps in its order.
pub const replay_rule = "the host's replay of each output's fp32 and bf16 steps in the kernel's order, bit for bit (NaN as NaN)";
/// Gate 4: the reference replays every step after the sums, so a bf16 sum shows as a few output ulps at most.
const sums_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps (gate 3 replays the fp32 order)";

pub fn val(h: u16) f32 {
    return c.fromBf16(h);
}

pub fn rbf(x: f32) f32 {
    return val(c.toBf16(x));
}

pub fn same(got: u16, want: u16) bool {
    return got == want or (std.math.isNan(val(got)) and std.math.isNan(val(want)));
}

pub fn at16(bytes: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
}

/// lfm3v.cu's warp_sum: the xor butterfly 16, 8, 4, 2, 1, each lane adding its partner (every lane ends equal).
pub fn warpSum(lanes: *[32]f32) f32 {
    var o: usize = 16;
    while (o > 0) : (o >>= 1) {
        var t: [32]f32 = undefined;
        for (&t, 0..) |*v, l| v.* = lanes[l] + lanes[l ^ o];
        lanes.* = t;
    }
    return lanes[0];
}

/// lfm3v.cu's inv_rms: 1 / sqrt(sum / n + eps), each step rounded to fp32.
fn invRms(sum: f32, n: usize, e: f32) f32 {
    return 1.0 / @sqrt(sum / @as(f32, @floatFromInt(n)) + e);
}

/// Gate 1's places in x [rows, w]: NaN, +inf, -inf, then both infinities in row 3 (rows that exist).
pub fn plant(x: []u16, rows: usize, w: usize, at0: usize) void {
    const at = [_][2]usize{ .{ 0, w / 3 }, .{ 1, w / 2 }, .{ 2, w - 1 }, .{ 3, 0 }, .{ 3, w - 1 } };
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (at, put) |p, v| if (p[0] < rows) {
        x[at0 + p[0] * w + p[1]] = c.toBf16(v);
    };
}

/// The row's sum: exact (f64, the reference under gate 4's integer rows), in an accumulator, or the kernel's own.
const Sum = union(enum) { exact, acc: struct { a: c.Accum, o: g.Order } };

fn sumOf(how: Sum, t: []const f64) f32 {
    return switch (how) {
        .exact => blk: {
            var s: f64 = 0;
            for (t) |v| s += v;
            break :blk @floatCast(s);
        },
        .acc => |x| @floatCast(rows_.sumIn(x.a, x.o, t)),
    };
}

fn rowsDepth(_: c.Shape) usize {
    return 8 + 5 + 5; // a thread's 8 squares, its warp's butterfly, the 8 warps' butterfly
}

fn headDepth(_: c.Shape) usize {
    return 1 + 5; // a lane's two squares, the butterfly
}

pub fn rmsSpec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: lfm3v RMS norm with x r left in fp32" else "cuda lfm3v RMS norm tf_lfm3v_rms (bf16(w bf16(x r)), a block a row)",
        .accum = .f32,
        .depth = rowsDepth,
        .out = .bf16,
        .nan = .{ .stated = replay_rule },
        .tiles = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 1 } },
        // pictures of 280, 256 and 6400 patches (the cap): every norm of the tower is 1024 wide
        .served = &.{ .{ .rows = 280, .n = 1024, .k = 1024 }, .{ .rows = 256, .n = 1024, .k = 1024 }, .{ .rows = 6400, .n = 1024, .k = 1024 } },
        .sums = .{ .bf16 = 0, .why = sums_why },
    };
}

/// x [rows, k] bf16 normed by its rows' RMS, times w [k]: torch's LFM3-VL RMSNorm at its bf16 points.
pub const Rms = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    dev: [3]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Rms) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Rms) g.Kernel {
        return .{ .spec = rmsSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Rms);

    pub fn prepare(self: *Rms, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.k > 2048) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const k = shape.k;
        self.x = try a.alloc(u16, shape.rows * k);
        self.w = try a.alloc(u16, k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1));
        for (self.w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        // gate 4: integers 64 to 127 of either sign, squares below 2^24 summed over 1024 in any order; fp16 overflows
        if (fill == .stress) for (self.x) |*v| {
            const m: f32 = @floatFromInt(64 + rng.uintLessThan(u32, 64));
            v.* = c.toBf16(if (rng.boolean()) m else -m);
        };
        if (fill == .nonfinite) plant(self.x, shape.rows, k, 0);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.dev[2] = try self.rig.output(shape.rows * k * 2);
        return true;
    }

    pub fn run(self: *Rms) !void {
        var a: cuda.Args = .{};
        for (self.dev) |p| a.add(p);
        a.add(@as(c_int, @intCast(self.shape.k)));
        a.add(eps);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rms) usize {
        return self.shape.rows * self.shape.k;
    }

    /// The kernel's sum of squares: each thread its columns t + 256 i in order, its warp's butterfly, the warps'.
    fn kernelSum(self: *const Rms, row: usize) f32 {
        const k = self.shape.k;
        var part: [32]f32 = @splat(0);
        for (0..8) |wp| {
            var lanes: [32]f32 = undefined;
            for (&lanes, 0..) |*s, l| {
                s.* = 0;
                for (0..8) |i| {
                    const col = wp * 32 + l + 256 * i;
                    const v: f32 = if (col < k) val(self.x[row * k + col]) else 0;
                    s.* += v * v;
                }
            }
            part[wp] = warpSum(&lanes);
        }
        return warpSum(&part);
    }

    /// Row `row`'s outputs from its sum: r = inv_rms, then bf16(w bf16(x r)).
    fn finish(self: *const Rms, row: usize, sum: f32, col: usize) u16 {
        const r = invRms(sum, self.shape.k, eps);
        return c.toBf16(val(self.w[col]) * rbf(val(self.x[row * self.shape.k + col]) * r));
    }

    fn sumHow(self: *const Rms, row: usize, how: Sum, t: []f64) f32 {
        for (t, self.x[row * self.shape.k ..][0..self.shape.k]) |*o, h| o.* = @as(f64, val(h)) * val(h);
        return sumOf(how, t);
    }

    fn outputs(self: *Rms, how: Sum, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (0..self.shape.rows) |row| {
            const s = self.sumHow(row, how, t);
            for (0..self.shape.k) |col| out[row * self.shape.k + col] = val(self.finish(row, s, col));
        }
    }

    pub fn reference(self: *Rms, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        return self.outputs(.exact, want);
    }

    pub fn simulate(self: *Rms, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.outputs(.{ .acc = .{ .a = acc, .o = order } }, out);
    }

    pub fn check(self: *Rms, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const k = self.shape.k;
        var nonfinite: usize = 0;
        for (0..self.shape.rows) |row| {
            const s = self.kernelSum(row);
            for (0..k) |col| {
                const want = self.finish(row, s, col);
                const got = at16(out, row * k + col);
                if (!std.math.isFinite(val(want))) nonfinite += 1;
                if (!same(got, want)) return try std.fmt.bufPrint(&self.msg, "row {d} column {d}: 0x{x:0>4} where the replay gives 0x{x:0>4}", .{ row, col, got, want });
            }
        }
        if (at == .nonfinite and nonfinite == 0) return "the planted values make no output non-finite: no proof";
        return null;
    }
};

pub fn qkSpec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: lfm3v q/k rows rotating the normed head in fp32" else "cuda lfm3v q/k head RMS norm and axial RoPE tf_lfm3v_qk (in place, v untouched)",
        .accum = .f32,
        .depth = headDepth,
        .out = .bf16,
        .in_place = true,
        .nan = .{ .stated = replay_rule },
        // qkv [patches, 3, 16, 64], each kv head written for both its query heads
        .served = &.{ .{ .rows = 280, .n = 16, .k = dim }, .{ .rows = 256, .n = 16, .k = dim }, .{ .rows = 6400, .n = 16, .k = dim } },
        .sums = .{ .bf16 = 0, .why = sums_why },
    };
}

/// qkv [rows, 3, H, 64] in place: q and k heads RMS-normed (qw, kw), then rotated by the rows' cos and sin tables.
pub const Qk = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    qkv: []u16 = &.{},
    w: []u16 = &.{}, // qw then kw
    cs: []f32 = &.{},
    sn: []f32 = &.{},
    dev: [5]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Qk) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Qk) g.Kernel {
        return .{ .spec = qkSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Qk);

    pub fn prepare(self: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != dim) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const n = shape.rows * 3 * shape.n * dim;
        self.qkv = try a.alloc(u16, n);
        self.w = try a.alloc(u16, 2 * dim);
        self.cs = try a.alloc(f32, shape.rows * dim);
        self.sn = try a.alloc(f32, shape.rows * dim);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.qkv) |*v| v.* = c.toBf16(rng.floatNorm(f32) * 2);
        for (self.w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        tables(shape.rows, self.cs, self.sn);
        // gate 4: integers 128 to 255, squares 16384 and up: fp16 sums overflow, 64 of them stay below 2^24
        if (fill == .stress) for (self.qkv) |*v| {
            const m: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
            v.* = c.toBf16(if (rng.boolean()) m else -m);
        };
        if (fill == .nonfinite) plant(self.qkv, shape.rows, 3 * shape.n * dim, 0);
        self.dev[0] = try self.rig.output(n * 2);
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.w[0..dim]));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.w[dim..]));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.cs));
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(self.sn));
        return true;
    }

    /// lfm3_vision.zig's axial tables: rows in merge order on a 20-wide grid, (h freqs, w freqs) twice over the head.
    fn tables(rows: usize, cs: []f32, sn: []f32) void {
        var inv: [dim / 4]f32 = undefined;
        for (&inv, 0..) |*v, j| v.* = 1.0 / std.math.pow(f32, 1e6, @as(f32, @floatFromInt(2 * j)) / @as(f32, dim / 2));
        for (0..rows) |i| {
            const py: f32 = @floatFromInt((i / 4 / 10) * 2 + i % 4 / 2);
            const px: f32 = @floatFromInt((i / 4 % 10) * 2 + i % 2);
            for (0..dim / 4) |j| for ([_]usize{ 0, dim / 2 }) |base| {
                cs[i * dim + base + j] = @cos(py * inv[j]);
                sn[i * dim + base + j] = @sin(py * inv[j]);
                cs[i * dim + base + dim / 4 + j] = @cos(px * inv[j]);
                sn[i * dim + base + dim / 4 + j] = @sin(px * inv[j]);
            };
        }
    }

    pub fn run(self: *Qk) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev[0], .len = self.qkv.len * 2 };
        try b.upload(0, std.mem.sliceAsBytes(self.qkv)); // in place: the rows go in fresh each run
        var a: cuda.Args = .{};
        for (self.dev) |p| a.add(p);
        a.add(@as(c_int, @intCast(self.shape.rows)));
        a.add(@as(c_int, @intCast(self.shape.n)));
        a.add(eps);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows * self.shape.n * 2), .y = 1, .z = 1 }, .block = .{ .x = 32 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Qk) usize {
        return self.qkv.len;
    }

    fn head(self: *const Qk, row: usize, part: usize, h: usize) []const u16 {
        return self.qkv[((row * 3 + part) * self.shape.n + h) * dim ..][0..dim];
    }

    /// The kernel's sum: lane l's a a + b b (dims l and l + 32), then the butterfly.
    fn kernelSum(x: []const u16) f32 {
        var lanes: [32]f32 = undefined;
        for (&lanes, 0..) |*s, l| s.* = val(x[l]) * val(x[l]) + val(x[l + 32]) * val(x[l + 32]);
        return warpSum(&lanes);
    }

    /// The head's 64 outputs from its sum: the normed head rounded as torch hands it on, then rotate_half in fp32.
    fn finish(self: *const Qk, row: usize, part: usize, x: []const u16, sum: f32, out: *[dim]u16) void {
        const r = invRms(sum, dim, eps);
        const w = self.w[part * dim ..][0..dim];
        const cs = self.cs[row * dim ..][0..dim];
        const sn = self.sn[row * dim ..][0..dim];
        for (0..32) |l| {
            const na = rbf(val(w[l]) * rbf(val(x[l]) * r));
            const nb = rbf(val(w[l + 32]) * rbf(val(x[l + 32]) * r));
            out[l] = c.toBf16(na * cs[l] + -nb * sn[l]);
            out[l + 32] = c.toBf16(nb * cs[l + 32] + na * sn[l + 32]);
        }
    }

    fn outputs(self: *Qk, how: ?Sum, out: []f64) void {
        var t: [dim]f64 = undefined;
        var y: [dim]u16 = undefined;
        for (0..self.shape.rows) |row| for (0..3) |part| for (0..self.shape.n) |h| {
            const x = self.head(row, part, h);
            const o = out[((row * 3 + part) * self.shape.n + h) * dim ..][0..dim];
            if (part == 2) {
                for (o, x) |*v, b| v.* = val(b);
                continue;
            }
            for (&t, x) |*v, b| v.* = @as(f64, val(b)) * val(b);
            self.finish(row, part, x, if (how) |hw| sumOf(hw, &t) else kernelSum(x), &y);
            for (o, y) |*v, b| v.* = val(b);
        };
    }

    pub fn reference(self: *Qk, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        self.outputs(.exact, want);
    }

    pub fn simulate(self: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        self.outputs(.{ .acc = .{ .a = acc, .o = order } }, out);
    }

    pub fn check(self: *Qk, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        var y: [dim]u16 = undefined;
        var nonfinite: usize = 0;
        for (0..self.shape.rows) |row| for (0..3) |part| for (0..self.shape.n) |h| {
            const x = self.head(row, part, h);
            if (part < 2) self.finish(row, part, x, kernelSum(x), &y) else @memcpy(&y, x);
            const base = ((row * 3 + part) * self.shape.n + h) * dim;
            for (y, 0..) |want, e| {
                const got = at16(out, base + e);
                if (!std.math.isFinite(val(want))) nonfinite += 1;
                if (!same(got, want)) return try std.fmt.bufPrint(&self.msg, "row {d} {s} head {d} dim {d}: 0x{x:0>4} where the replay gives 0x{x:0>4}", .{ row, ([_][]const u8{ "q", "k", "v" })[part], h, e, got, want });
            }
        };
        if (at == .nonfinite and nonfinite == 0) return "the planted values make no output non-finite: no proof";
        return null;
    }
};

/// Every lfm3v kernel and its control, registered onto `kernels`; the controls from the contract_controls image `ctl`.
pub const Set = struct {
    const parts = .{ "rms", "qk", "residual", "silu", "conv", "set", "vit" };

    mod: ?cuda.Module = null,
    pa: ?cuda.Module = null, // prefill_attention: vision_kernel<64, 8>
    rms: [2]Rms = undefined,
    qk: [2]Qk = undefined,
    residual: [2]pointwise.Add = undefined,
    silu: [2]pointwise.Silu = undefined,
    conv: [2]conv_k.Conv = undefined,
    set: [4]set_k.Attn = undefined,
    vit: [2]vit_k.Vit = undefined,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.lfm3v);
        const m = self.mod.?;
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const pick = struct {
            fn f(mm: cuda.Module, cm: cuda.Module, ctrl: bool, comptime name: []const u8) !cuda.Function {
                return if (ctrl) cm.function("tf_ctl_" ++ name) else mm.function("tf_" ++ name);
            }
        }.f;
        const both = [_]bool{ false, true };
        for (&self.rms, both) |*x, ctrl| x.* = .{ .rig = rig, .f = try pick(m, ctl, ctrl, "lfm3v_rms"), .control = ctrl, .arena = .init(gpu.gpa) };
        for (&self.qk, both) |*x, ctrl| x.* = .{ .rig = rig, .f = try pick(m, ctl, ctrl, "lfm3v_qk"), .control = ctrl, .arena = .init(gpu.gpa) };
        for (&self.residual, both) |*x, ctrl| x.* = .{ .rig = rig, .f = try pick(m, ctl, ctrl, "lfm3v_add"), .control = ctrl, .arena = .init(gpu.gpa) };
        for (&self.silu, both) |*x, ctrl| x.* = .{ .rig = rig, .f = try pick(m, ctl, ctrl, "lfm3v_silu_mul"), .control = ctrl, .arena = .init(gpu.gpa) };
        for (&self.conv, both) |*x, ctrl| x.* = .{ .rig = rig, .f = try pick(m, ctl, ctrl, "lfm3v_conv"), .control = ctrl, .arena = .init(gpu.gpa) };
        const plain = try m.function("tf_lfm3v_set_attn");
        for (&self.set, set_k.forms) |*x, form| x.* = .{ .rig = rig, .form = form, .f = try (if (form.control) ctl else m).function(form.sym), .plain = plain, .arena = .init(gpu.gpa) };
        self.pa = try cuda.Module.load(gpu.d, cuda.kernels.prefill_attention);
        for (&self.vit, vit_k.forms) |*x, form| {
            const f = try (if (form.control) ctl else self.pa.?).function(form.sym);
            try f.allowDynamicShared(vit_k.smem);
            x.* = .{ .rig = rig, .form = form, .f = f, .arena = .init(gpu.gpa) };
        }
        inline for (parts) |name| for (&@field(self, name)) |*x| try kernels.append(gpu.gpa, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        var m = self.mod orelse return;
        inline for (parts) |name| for (&@field(self, name)) |*x| x.deinit();
        if (self.pa) |*p| p.unload();
        m.unload();
    }
};
