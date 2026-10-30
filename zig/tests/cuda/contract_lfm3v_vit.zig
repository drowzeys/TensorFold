//! LFM3-VL's tower attention, vision_kernel<64, 8> (prefill_attention.cu), under attention_ref's bound, controlled.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;

// lfm3_vision.zig: 16 heads of 64 (its kv heads repeated in the rows), qkv [patches, 3, 16, 64], scale 64^-1/2
const h = 16;
const d = 64;
const scale: f32 = 0.125;
/// vit.zig's Full.smem for vision_kernel: two buffers of 64 K rows and 64 V rows, each row d / 8 + 1 uint4s.
pub const smem: u32 = 2 * 2 * 64 * (d / 8 + 1) * 16;

/// A registration: the served instance or its planted twin (contract_controls.cu).
pub const Form = struct { name: []const u8, sym: [:0]const u8, control: bool = false };

const vision64 = "_ZN20tf_prefill_attention13vision_kernelILi64ELi8EEEvPK13__nv_bfloat16PS1_iif";

pub const forms = [_]Form{
    .{ .name = "cuda vit vision_kernel<64, 8> (LFM3-VL's tower, 16 heads of 64, all patches)", .sym = vision64 },
    .{
        .name = "control: vision_kernel<64, 8> reading its values from the neighbouring 8 dims",
        .sym = "tf_ctl_vision_64",
        .control = true,
    },
};

/// A row's P x V chain: every patch it sees, a 64-key tile, then the fold.
fn keyChain(s: c.Shape) usize {
    return s.k + 64 + 64;
}

/// rows = the query rows checked (spread, with the first and last), n = heads, k = the picture's patches.
pub fn spec(f: Form) c.Spec {
    return .{
        .name = f.name,
        .accum = .f32,
        .depth = keyChain,
        .ops = 1,
        .out = .bf16,
        // 128 rows a block, 64 keys a tile: 127, 128 and 129 patches cross both
        .tiles = .{ .{}, .{}, .{ .size = 128 } },
        // a 16 x 16 picture (the least pixels), 14 x 20, 32 x 32 (256 tokens), 80 x 80 (patches_most)
        .served = &.{
            .{ .rows = 256, .n = h, .k = 256 },   .{ .rows = 280, .n = h, .k = 280 },
            .{ .rows = 1024, .n = h, .k = 1024 }, .{ .rows = 6400, .n = h, .k = 6400 },
        },
        .sums = .{ .why = attn.why },
    };
}

pub const Vit = struct {
    rig: rig_.Rig,
    form: Form,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    r: ?attn.Ref = null,
    w: usize = 0,
    picked: []usize = &.{},
    qkv: []u16 = &.{},
    sel: []u8 = &.{},
    dev: [2]u64 = @splat(0),

    pub fn deinit(self: *Vit) void {
        if (self.r) |*r| r.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Vit) g.Kernel {
        const control: ?g.Gate = if (self.form.control) .shapes else null;
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Vit);

    /// n rows of w, spread evenly with the first and the last.
    pub fn spread(out: []usize, w: usize) void {
        for (out, 0..) |*x, i| x.* = if (out.len == 1) 0 else i * (w - 1) / (out.len - 1);
    }

    fn at(row: usize, part: usize, head: usize) usize {
        return ((row * 3 + part) * h + head) * d;
    }

    pub fn prepare(self: *Vit, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != h or shape.k == 0 or shape.rows == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.w = shape.k;
        self.picked = try a.alloc(usize, @min(shape.rows, shape.k));
        spread(self.picked, self.w);
        self.qkv = try a.alloc(u16, self.w * 3 * h * d);
        self.sel = try a.alloc(u8, self.picked.len * h * d * 2);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.qkv) |*e| e.* = attn.normal(rng);
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(self.rig.gpu.gpa, d, self.w);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.qkv));
        self.dev[1] = try self.rig.output(self.w * h * d * 2);
        return true;
    }

    /// A NaN query (row 0, head 0); +inf, -inf values at patch 0 (every row sees it), +inf at patch 1; a NaN key.
    fn plant(self: *Vit) void {
        self.qkv[at(0, 0, 0) + 3] = c.toBf16(std.math.nan(f32));
        self.qkv[at(0, 2, 0) + 7] = c.toBf16(std.math.inf(f32));
        self.qkv[at(0, 2, 1) + 9] = c.toBf16(-std.math.inf(f32));
        self.qkv[at(@min(1, self.w - 1), 2, 1) + 9] = c.toBf16(std.math.inf(f32));
        self.qkv[at(0, 1, 2)] = c.toBf16(std.math.nan(f32));
    }

    /// Gate 4: q 0, so every key scores alike (each p is 1); each value column mirrored over the patches.
    fn stress(self: *Vit, rng: std.Random) !void {
        for (0..self.w) |row| @memset(self.qkv[at(row, 0, 0)..][0 .. h * d], 0);
        const col = try self.arena.allocator().alloc(f32, self.w);
        for (0..h) |head| for (0..d) |e| {
            attn.mirrored(rng, col);
            for (col, 0..) |y, j| self.qkv[at(j, 2, head) + e] = c.toBf16(y);
        };
    }

    /// As vit.zig's Attention.full launches it: a block of 8 warps a head and 128 rows.
    pub fn run(self: *Vit) !void {
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(self.dev[1]);
        for ([_]usize{ self.w, h }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(scale);
        const grid: cuda.launch.Dim3 = .{ .x = @intCast((self.w + 127) / 128), .y = h, .z = 1 };
        const cfg: cuda.launch.Config = .{ .grid = grid, .block = .{ .x = 256 }, .shared = smem };
        try cuda.launch.launch(self.f, cfg, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Vit) usize {
        return self.picked.len * h * d;
    }

    /// Checked head i (picked row i / 16, head i % 16): its query over every patch's key and value.
    pub fn load(self: *Vit, i: usize, hd: *attn.Head) !void {
        const row = self.picked[i / h];
        const head = i % h;
        for (hd.q, 0..) |*x, e| x.* = c.fromBf16(self.qkv[at(row, 0, head) + e]);
        for (0..self.w) |j| for (0..d) |e| {
            hd.k[j * d + e] = c.fromBf16(self.qkv[at(j, 1, head) + e]);
            hd.v[j * d + e] = c.fromBf16(self.qkv[at(j, 2, head) + e]);
        };
        hd.n = self.w;
    }

    pub fn check(self: *Vit, gate: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const row_bytes = h * d * 2;
        for (self.picked, 0..) |row, i| {
            @memcpy(self.sel[i * row_bytes ..][0..row_bytes], out[row * row_bytes ..][0..row_bytes]);
        }
        return self.r.?.check(self, self.picked.len * h, scale, self.sel, gate);
    }

    pub fn reference(self: *Vit, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.picked.len * h, scale, want, mag);
    }

    pub fn simulate(self: *Vit, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.picked.len * h, scale, acc, order, out);
    }
};

test "checked rows: spread over the picture, its first and its last" {
    var r: [4]usize = undefined;
    Vit.spread(&r, 280);
    try std.testing.expectEqualSlices(usize, &.{ 0, 93, 186, 279 }, &r);
    var one: [1]usize = undefined;
    Vit.spread(&one, 7);
    try std.testing.expectEqual(@as(usize, 0), one[0]);
}
