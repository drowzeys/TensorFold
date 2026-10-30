//! lfm3v.cu's pointwise rows under the contract: the residual add (exact replay), SwiGLU (float64 at its bf16 points).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const l3 = @import("contract_lfm3v.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const bfr = core.bf16_round;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Edges of the width: a partial 256-thread block at 255 and 257 wide (280 rows), whole blocks at the served widths.
const width_tile: c.Tile = .{ .size = 256, .step = 1, .min = 1 };

pub fn addSpec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: lfm3v residual add with whole blocks and no tail" else "cuda lfm3v residual add tf_lfm3v_add (bf16(a + b), in place over a)",
        .accum = .none,
        .depth = noSums,
        .out = .bf16,
        .in_place = true,
        .nan = .{ .stated = l3.replay_rule },
        .tiles = .{ .{}, width_tile, .{} },
        // the tower's residual rows: h = bf16(h + y) over pictures of 280, 256 and 6400 patches
        .served = &.{ .{ .rows = 280, .n = 1024, .k = 1 }, .{ .rows = 256, .n = 1024, .k = 1 }, .{ .rows = 6400, .n = 1024, .k = 1 } },
    };
}

fn launch1d(f: cuda.Function, s: cuda.Stream, n: usize, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((n + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, a);
    try s.synchronize();
}

/// h [rows, n] += y as lfm3_vision.zig calls it (a and the output one buffer), each bf16(a + b) in fp32.
pub const Add = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    a: []u16 = &.{},
    b: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Add) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Add) g.Kernel {
        return .{ .spec = addSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Add);

    pub fn prepare(self: *Add, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const al = self.arena.allocator();
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.a = try al.alloc(u16, n);
        self.b = try al.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.a) |*v| v.* = c.toBf16(rng.floatNorm(f32) * 8);
        for (self.b) |*v| v.* = c.toBf16(rng.floatNorm(f32));
        if (fill == .nonfinite) {
            l3.plant(self.a, shape.rows, shape.n, 0);
            if (shape.rows > 1) self.b[shape.n + shape.n / 2] = c.toBf16(-std.math.inf(f32)); // meets a's +inf: NaN
        }
        self.dev[0] = try self.rig.output(n * 2);
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.b));
        return true;
    }

    pub fn run(self: *Add) !void {
        const buf: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.dev[0], .len = self.a.len * 2 };
        try buf.upload(0, std.mem.sliceAsBytes(self.a)); // in place: a goes in fresh each run
        var args: cuda.Args = .{};
        for ([_]u64{ self.dev[0], self.dev[1], self.dev[0] }) |p| args.add(p);
        args.add(@as(c_longlong, @intCast(self.a.len)));
        try launch1d(self.f, self.rig.s, self.a.len, &args);
    }

    pub fn count(self: *Add) usize {
        return self.a.len;
    }

    pub fn reference(_: *Add, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Add, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        var nonfinite: usize = 0;
        for (self.a, self.b, 0..) |x, y, i| {
            const want = c.toBf16(l3.val(x) + l3.val(y));
            const got = l3.at16(out, i);
            if (!std.math.isFinite(l3.val(want))) nonfinite += 1;
            if (!l3.same(got, want)) return try std.fmt.bufPrint(&self.msg, "element {d} of {d}: 0x{x:0>4} where the replay gives 0x{x:0>4}", .{ i, self.a.len, got, want });
        }
        if (at == .nonfinite and nonfinite == 0) return "the planted values make no output non-finite: no proof";
        return null;
    }
};

/// expf within 2 ulps (2^-22 of its value), 1 + e's and the division's roundings within 2^-23: corners at these.
const exp_spread = 0x1p-21;
const div_spread = 0x1p-22;

/// The output at float64 with expf off by (1 + te) and the quotient by (1 + tq): bf16(bf16(silu(g)) u).
fn siluPoint(gate: f64, up: f64, te: f64, tq: f64) f64 {
    const s = bfr.bf16(gate / (1 + @exp(-gate) * (1 + te)) * (1 + tq));
    return bfr.bf16(s * up);
}

pub fn siluSpec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: lfm3v SwiGLU with silu left in fp32" else "cuda lfm3v SwiGLU tf_lfm3v_silu_mul (bf16(bf16(silu(g)) u), w1 | w3 rows)",
        .accum = .none,
        .depth = noSums,
        .out = .bf16,
        .nan = .{ .stated = "float64 at its bf16 points within expf's ulps; NaN and infinities as float64's point" },
        .tiles = .{ .{}, width_tile, .{} },
        // the tower's MLP: 2560 wide over pictures of 280, 256 and 6400 patches
        .served = &.{ .{ .rows = 280, .n = 2560, .k = 1 }, .{ .rows = 256, .n = 2560, .k = 1 }, .{ .rows = 6400, .n = 2560, .k = 1 } },
    };
}

/// gu [rows, 2n] (w1's n gates, then w3's n ups) to y [rows, n].
pub const Silu = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Silu) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Silu) g.Kernel {
        return .{ .spec = siluSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Silu);

    pub fn prepare(self: *Silu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.gu = try self.arena.allocator().alloc(u16, shape.rows * 2 * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.gu) |*v| v.* = c.toBf16(rng.floatNorm(f32) * 3);
        if (fill == .nonfinite) plant(self.gu, shape.rows, shape.n);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.gu));
        self.dev[1] = try self.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    /// Gates at NaN, +inf and -inf (silu(-inf) is -inf / inf: NaN), ups at +inf and -inf.
    fn plant(gu: []u16, rows: usize, n: usize) void {
        const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
        const at = [_][2]usize{ .{ 0, n / 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, n }, .{ 3, 2 * n - 1 } };
        for (at, put) |p, v| if (p[0] < rows) {
            gu[p[0] * 2 * n + p[1]] = c.toBf16(v);
        };
    }

    pub fn run(self: *Silu) !void {
        const n = self.shape.rows * self.shape.n;
        var a: cuda.Args = .{};
        for (self.dev) |p| a.add(p);
        a.add(@as(c_int, @intCast(self.shape.n)));
        a.add(@as(c_longlong, @intCast(n)));
        try launch1d(self.f, self.rig.s, n, &a);
    }

    pub fn count(self: *Silu) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(_: *Silu, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Each output within its float64 ends over expf's and the division's spread; non-finite ones by class.
    pub fn check(self: *Silu, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const n = self.shape.n;
        var nonfinite: usize = 0;
        var ranged: usize = 0;
        for (0..self.shape.rows) |row| for (0..n) |i| {
            const gate: f64 = l3.val(self.gu[row * 2 * n + i]);
            const up: f64 = l3.val(self.gu[row * 2 * n + n + i]);
            const got: f64 = l3.val(l3.at16(out, row * n + i));
            var lo = std.math.inf(f64);
            var hi = -std.math.inf(f64);
            var finite = true;
            for ([_]f64{ -exp_spread, exp_spread }) |te| for ([_]f64{ -div_spread, div_spread }) |tq| {
                const v = siluPoint(gate, up, te, tq);
                if (!std.math.isFinite(v)) finite = false;
                lo = @min(lo, v);
                hi = @max(hi, v);
            };
            const ok = if (finite) std.math.isFinite(got) and lo <= got and got <= hi else blk: {
                nonfinite += 1;
                break :blk c.classOf(got) == c.classOf(siluPoint(gate, up, 0, 0));
            };
            if (lo != hi) ranged += 1;
            if (!ok) return try std.fmt.bufPrint(&self.msg, "row {d} out[{d}] = {e} outside float64's [{e}, {e}] (g {e}, u {e})", .{ row, i, got, lo, hi, gate, up });
        };
        if (at == .nonfinite and nonfinite == 0) return "the planted values make no output non-finite: no proof";
        if (ranged * 8 > self.count()) return try std.fmt.bufPrint(&self.msg, "{d} of {d} outputs only within a range: too loose to prove the points", .{ ranged, self.count() });
        return null;
    }
};
