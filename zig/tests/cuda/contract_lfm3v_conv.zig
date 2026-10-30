//! lfm3v.cu's gated 3x3 depthwise conv (LFM3-VL) under the contract: an exact host replay, its raster-order control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const l3 = @import("contract_lfm3v.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A shape is a picture: rows = gh gw patches in merge order (2 x 2 blocks), n channels, k = gw (both sides even).
pub fn spec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: lfm3v 3x3 depthwise conv reading neighbours in raster order" else "cuda lfm3v gated 3x3 depthwise conv tf_lfm3v_conv (merge-order rows, zero edges)",
        .accum = .none,
        .depth = noSums,
        .out = .bf16,
        .nan = .{ .stated = l3.replay_rule },
        .tiles = .{ .{}, .{ .size = 256, .step = 64, .min = 64 }, .{} },
        // first a 4 x 2 picture: the single-shape gates clamp the first shape's rows into 5 to 8 and 1 to 64
        .served = &.{ .{ .rows = 8, .n = 1024, .k = 2 }, .{ .rows = 256, .n = 1024, .k = 16 }, .{ .rows = 280, .n = 1024, .k = 20 }, .{ .rows = 4480, .n = 1024, .k = 80 }, .{ .rows = 6400, .n = 1024, .k = 80 } },
    };
}

/// bcx [P, 3C] (B, C, x thirds of in_proj's rows), w [C, 9] to out [P, C] = bf16(C bf16(dwconv3x3(bf16(B x)))).
pub const Conv = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    p: []u16 = &.{},
    w: []u16 = &.{},
    dev: [3]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Conv) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Conv) g.Kernel {
        return .{ .spec = spec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Conv);

    pub fn prepare(self: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gw = shape.k;
        if (shape.n == 0 or gw < 2 or gw % 2 != 0 or shape.rows == 0 or shape.rows % (2 * gw) != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const r = shape.rows;
        const d = shape.n;
        self.p = try a.alloc(u16, r * 3 * d);
        self.w = try a.alloc(u16, d * 9);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.p) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (self.w) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, d / 3 }, .{ 1, d + 1 }, .{ 2, 2 * d + 5 }, .{ 3, 0 }, .{ 3, 3 * d - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |q, v| if (q[0] < r) {
                self.p[q[0] * 3 * d + q[1]] = c.toBf16(v);
            };
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.p));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.dev[2] = try self.rig.output(r * d * 2);
        return true;
    }

    /// As lfm3_vision.zig launches it: a thread a (row, channel), 256 a block.
    pub fn run(self: *Conv) !void {
        const n = self.shape.rows * self.shape.n;
        var a: cuda.Args = .{};
        for (self.dev) |p| a.add(p);
        for ([_]usize{ self.shape.rows / self.shape.k, self.shape.k, self.shape.n }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((n + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Conv) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(_: *Conv, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The merge-order row at grid position (y, x), merge 2.
    pub fn rowAt(y: usize, x: usize, gw: usize) usize {
        return ((y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
    }

    /// Taps in (dy, dx) order, each gated value rounded to bf16 and fused into the fp32 sum (fmaf), edges skipped.
    pub fn check(self: *Conv, at: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const d = self.shape.n;
        const gw = self.shape.k;
        const gh = self.shape.rows / gw;
        var nonfinite: usize = 0;
        for (0..gh) |y| for (0..gw) |x| {
            const row = rowAt(y, x, gw);
            for (0..d) |ch| {
                var acc: f32 = 0;
                for (0..3) |ky| for (0..3) |kx| {
                    if (y + ky < 1 or y + ky > gh or x + kx < 1 or x + kx > gw) continue;
                    const m = rowAt(y + ky - 1, x + kx - 1, gw);
                    const bx = l3.rbf(l3.val(self.p[m * 3 * d + ch]) * l3.val(self.p[m * 3 * d + 2 * d + ch]));
                    acc = @mulAdd(f32, l3.val(self.w[ch * 9 + ky * 3 + kx]), bx, acc);
                };
                const want = c.toBf16(l3.val(self.p[row * 3 * d + d + ch]) * l3.rbf(acc));
                const got = l3.at16(out, row * d + ch);
                if (!std.math.isFinite(l3.val(want))) nonfinite += 1;
                if (!l3.same(got, want)) return try std.fmt.bufPrint(&self.msg, "grid ({d}, {d}) channel {d}: 0x{x:0>4} where the replay gives 0x{x:0>4}", .{ y, x, ch, got, want });
            }
        };
        if (at == .nonfinite and nonfinite == 0) return "the planted values make no output non-finite: no proof";
        return null;
    }
};

test "the rig's merge-order rows: block by block, each block's two rows" {
    try std.testing.expectEqual(@as(usize, 0), Conv.rowAt(0, 0, 4));
    try std.testing.expectEqual(@as(usize, 1), Conv.rowAt(0, 1, 4));
    try std.testing.expectEqual(@as(usize, 2), Conv.rowAt(1, 0, 4));
    try std.testing.expectEqual(@as(usize, 4), Conv.rowAt(0, 2, 4));
    try std.testing.expectEqual(@as(usize, 8), Conv.rowAt(2, 0, 4));
}
