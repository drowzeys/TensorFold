//! qmm_midrow's prompt tiles under the contract on CUDA: each bit-equal to qmm_prefill's 128-row tile it replaces.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const check = @import("check.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const qlinear = cuda.qlinear;
const mid = qlinear.affine4_mid;

/// One fp32 chain over K in 16-deep MMA steps.
fn mmaChain(s: c.Shape) usize {
    return s.k / 16 + 16;
}

/// A short projection first (the single-shape gates take it), then Qwen3.8-27B's projections, then a ragged one.
const served = [_]c.Shape{
    .{ .rows = 17, .n = 200, .k = 512 },    .{ .rows = 100, .n = 10240, .k = 5120 }, .{ .rows = 160, .n = 6144, .k = 5120 },
    .{ .rows = 192, .n = 48, .k = 5120 },   .{ .rows = 224, .n = 5120, .k = 6144 },  .{ .rows = 257, .n = 12288, .k = 5120 },
    .{ .rows = 300, .n = 1024, .k = 5120 }, .{ .rows = 64, .n = 17408, .k = 5120 },  .{ .rows = 129, .n = 5120, .k = 17408 },
    .{ .rows = 513, .n = 2600, .k = 1088 },
};

const names = blk: {
    var out: [mid.tiles.len][]const u8 = undefined;
    for (mid.tiles, &out) |t, *n| n.* = std.fmt.comptimePrint("cuda qmm_midrow {d}x{d} prompt tile", .{ t.rows, t.cols });
    break :blk out;
};

/// The control's tile: tf_ctl_qmm_mid_reverse is the 80x256 tile walking K's groups last to first.
pub const reverse_tile = 7;

pub fn spec(tile: usize, control: bool) c.Spec {
    const t = mid.tiles[tile];
    return .{
        .name = if (control) "control: qmm_midrow 80x256 tile, K groups in reverse" else names[tile],
        .accum = .f32,
        .depth = mmaChain,
        .out = .bf16,
        .replaces = "qmm_prefill's 128-row prompt tile",
        .tiles = .{ .{ .size = t.rows }, .{ .size = t.cols }, .{ .size = 64 * @as(usize, t.stages), .step = 64, .min = 64 } },
        .served = &served,
    };
}

pub const Tile = struct {
    rig: rig_.Rig,
    tile: usize,
    affine4: *const qlinear.Affine4Kernels, // without its mid tiles: the replaced 128-row tile
    mids: *const qlinear.MidRows,
    control: ?cuda.Function = null,
    f: ?Affine = null,
    x: u64 = 0,
    q: qlinear.Affine4 = undefined,
    base: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Tile) void {
        if (self.f) |f| f.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Tile) g.Kernel {
        return .{ .spec = spec(self.tile, self.control != null), .ctx = self, .vt = &vt, .control = if (self.control != null) .shapes else null };
    }

    const vt = g.hooks(Tile);

    pub fn prepare(self: *Tile, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.f) |f| f.deinit();
        self.f = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        var f = try Affine.init(a, shape, 4, 64);
        f.form = .dequantized; // each weight s * q + b, then the dots
        self.f = f;
        try f.fill(fill, seed);
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        defer a.free(words);
        f.words(words);
        self.x = try self.rig.input(std.mem.sliceAsBytes(f.x));
        const lay = qlinear.Affine4.layout(shape.n, shape.k);
        const zeros = try a.alloc(u8, lay.words);
        defer a.free(zeros);
        @memset(zeros, 0);
        self.q = .{ .w = try self.rig.input(zeros), .s = try self.rig.input(zeros[0..lay.scales]), .b = try self.rig.input(zeros[0..lay.scales]), .n = shape.n, .k = shape.k, .npad = lay.npad };
        const src_w = try self.rig.scratch(words.len * 4);
        const src_s = try self.rig.scratch(f.scales.len * 2);
        const src_b = try self.rig.scratch(f.biases.len * 2);
        for ([_]u64{ src_w, src_s, src_b }, [_][]const u8{ std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases) }) |p, h| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = p, .len = h.len };
            try b.upload(0, h);
        }
        try self.affine4.pack(self.rig.s, src_w, src_s, src_b, self.q);
        try self.rig.s.synchronize();
        self.base = try self.rig.scratch(shape.rows * shape.n * 2);
        self.out = try self.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Tile) !void {
        const rows = self.f.?.shape.rows;
        if (self.control) |f| {
            try mid.launchWith(f, mid.tiles[reverse_tile], self.rig.s, self.x, self.q, self.out, rows);
        } else try self.mids.launchTile(self.tile, self.rig.s, self.x, self.q, self.out, rows);
        try self.rig.s.synchronize();
    }

    /// qmm_prefill's 128-row tile on the same inputs.
    pub fn replaced(self: *Tile) ![]const u8 {
        try self.affine4.prompt(self.rig.s, self.x, self.q, self.base, self.f.?.shape.rows);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn count(self: *Tile) usize {
        return self.f.?.outputs();
    }

    pub fn reference(self: *Tile, want: []f64, mag: []f64) !void {
        return self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Tile, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.f.?.simulate(acc, order, out);
    }
};

/// Every tile and the control, registered together (contract_tests.zig's one call).
pub const Set = struct {
    gpa: std.mem.Allocator,
    mod: ?cuda.Module = null,
    mids: qlinear.MidRows = undefined,
    list: [mid.tiles.len + 1]Tile = undefined,
    made: usize = 0,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, affine4: *const qlinear.Affine4Kernels, ctl_mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.qmm_midrow);
        self.mids = try qlinear.MidRows.resolve(self.mod.?, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
        const reverse = try ctl_mod.function("tf_ctl_qmm_mid_reverse");
        try reverse.allowDynamicShared(mid.tiles[reverse_tile].smem());
        for (&self.list, 0..) |*t, i| {
            const control = i == mid.tiles.len;
            t.* = .{ .rig = .{ .gpu = gpu, .s = s }, .tile = if (control) reverse_tile else i, .affine4 = affine4, .mids = &self.mids, .control = if (control) reverse else null };
            self.made += 1;
        }
        for (&self.list) |*t| try kernels.append(gpu.gpa, t.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.list[0..self.made]) |*t| t.deinit();
        if (self.mod) |*m| m.unload();
    }

    /// A set on `gpa`, its tiles and control registered (destroy() after the gates).
    pub fn create(gpa: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, affine4: *const qlinear.Affine4Kernels, ctl_mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpa.create(Set);
        set.* = .{ .gpa = gpa };
        errdefer set.destroy();
        try set.add(gpu, s, affine4, ctl_mod, kernels);
        return set;
    }

    pub fn destroy(self: *Set) void {
        self.deinit();
        self.gpa.destroy(self);
    }
};
