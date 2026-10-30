//! d1's set attention (lfm3v.cu) under the contract: the warp form against float64, the tiled form bit-equal to it.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;

// LFM3's full-attention layers (lfm3/config.zig): 16 query heads over 4 KV heads of 128
const h = 16;
const hk = 4;
const d = 128;

/// A registration: the warp form (a warp a row and head) or the tiled one, and whether it is the planted twin.
pub const Form = struct { name: []const u8, sym: [:0]const u8, tiled: bool, control: bool = false };

pub const forms = [_]Form{
    .{ .name = "cuda lfm3v d1 set attention tf_lfm3v_set_attn (a warp a row and head, 16/4 heads of 128)", .sym = "tf_lfm3v_set_attn", .tiled = false },
    .{ .name = "control: lfm3v set attention reading kv head h % 4 for query head h", .sym = "tf_ctl_lfm3v_set_attn", .tiled = false, .control = true },
    .{ .name = "cuda lfm3v d1 set attention tf_lfm3v_set_attn_tiled (4 rows x a kv head's 4 query heads, 32-key tiles)", .sym = "tf_lfm3v_set_attn_tiled", .tiled = true },
    .{ .name = "control: lfm3v tiled set attention skipping a tile by its first key's group alone", .sym = "tf_ctl_lfm3v_set_attn_tiled", .tiled = true, .control = true },
};

/// A row's chain: one fma a key into its P V sum (the keys it sees, at most all of the pass), its score's butterfly.
fn keyChain(s: c.Shape) usize {
    return s.k + 4 + 5;
}

const why = "P is rounded to bf16 before P V and exp is __expf: gates 1 and 3 hold each output to attention_ref's float64 bound";

/// rows = questions after their state, n = query heads, k = the pass's rows (each a key), as lfm3_vl's decideSet packs.
pub fn spec(f: Form) c.Spec {
    return .{
        .name = f.name,
        .accum = .f32,
        .depth = keyChain,
        .ops = 1,
        .out = .bf16,
        .replaces = if (f.tiled) "tf_lfm3v_set_attn" else null,
        .tiles = .{ .{}, .{}, .{ .size = 32, .step = 1, .min = 12 } },
        // d1_set_size up to 16 questions a pass of at most 1024 rows (prompt_rows), a third of them questions
        .served = &.{ .{ .rows = 4, .n = h, .k = 300 }, .{ .rows = 16, .n = h, .k = 1024 }, .{ .rows = 1, .n = h, .k = 97 }, .{ .rows = 8, .n = h, .k = 611 } },
        .sums = .{ .why = why },
    };
}

/// q [R, 16, 128], kv [R, 2, 4, 128] (a row's keys, then its values), group [R] (-1 in the state) to out [R, 16, 128].
pub const Attn = struct {
    rig: rig_.Rig,
    form: Form,
    f: cuda.Function,
    plain: cuda.Function, // tf_lfm3v_set_attn: the tiled forms' baseline
    arena: std.heap.ArenaAllocator,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    state: usize = 0,
    q: []u16 = &.{},
    kv: []u16 = &.{},
    group: []i32 = &.{},
    dev: [4]u64 = @splat(0),

    pub fn deinit(self: *Attn) void {
        if (self.r) |*r| r.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Attn) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }

    const vt = g.hooks(Attn);

    fn scale() f32 {
        return 1.0 / @sqrt(@as(f32, d));
    }

    /// The pass as decideSet packs it: the state's rows (-1), then the questions 1.. in order, a third of the rows.
    pub fn groups(out: []i32, questions: usize) usize {
        const rows = out.len;
        const qt = rows / 3;
        const state = rows - qt;
        @memset(out[0..state], -1);
        var at = state;
        for (0..questions) |k| {
            const len = qt / questions + @intFromBool(k < qt % questions);
            @memset(out[at..][0..len], @intCast(k + 1));
            at += len;
        }
        return state;
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const rows = shape.k;
        if (shape.n != h or shape.rows == 0 or rows / 3 < shape.rows) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        self.q = try a.alloc(u16, rows * h * d);
        self.kv = try a.alloc(u16, rows * 2 * hk * d);
        self.group = try a.alloc(i32, rows);
        self.state = groups(self.group, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.q) |*e| e.* = attn.normal(rng);
        for (self.kv) |*e| e.* = attn.normal(rng);
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(self.rig.gpu.gpa, d, rows);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.kv));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.group));
        self.dev[3] = try self.rig.output(rows * h * d * 2);
        return true;
    }

    fn keyAt(j: usize, kh: usize) usize {
        return j * 2 * hk * d + kh * d;
    }

    fn valueAt(j: usize, kh: usize) usize {
        return keyAt(j, kh) + hk * d;
    }

    /// A NaN query (row 0, head 0); +inf, -inf values at key 0 (every row sees it), +inf at key 1; a NaN key.
    fn plant(self: *Attn) void {
        self.q[3] = c.toBf16(std.math.nan(f32));
        self.kv[valueAt(0, 0) + 7] = c.toBf16(std.math.inf(f32));
        self.kv[valueAt(0, 1) + 9] = c.toBf16(-std.math.inf(f32));
        self.kv[valueAt(1, 1) + 9] = c.toBf16(std.math.inf(f32));
        self.kv[keyAt(0, 2)] = c.toBf16(std.math.nan(f32));
    }

    /// Gate 4: q 0, so every key scores alike (each p is 1); each value column mirrored over the pass's keys.
    fn stress(self: *Attn, rng: std.Random) !void {
        @memset(self.q, 0);
        const rows = self.shape.k;
        const col = try self.arena.allocator().alloc(f32, rows);
        for (0..hk) |kh| for (0..d) |e| {
            attn.mirrored(rng, col);
            for (col, 0..) |y, j| self.kv[valueAt(j, kh) + e] = c.toBf16(y);
        };
    }

    fn launch(self: *Attn, f: cuda.Function, tiled: bool, out: u64) !void {
        const rows = self.shape.k;
        var a: cuda.Args = .{};
        for ([_]u64{ self.dev[0], self.dev[1], self.dev[2], out }) |p| a.add(p);
        for ([_]usize{ rows, h, hk, self.state }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(scale());
        // the tiled form as cuda_forward.zig launches it: 4 rows x a kv head's 4 query heads, 16 warps a block
        const cfg: cuda.launch.Config = if (tiled) .{ .grid = .{ .x = @intCast((rows + 3) / 4), .y = hk, .z = 1 }, .block = .{ .x = 512 } } else .{ .grid = .{ .x = @intCast(rows), .y = h, .z = 1 }, .block = .{ .x = 32 } };
        try cuda.launch.launch(f, cfg, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Attn) !void {
        try self.launch(self.f, self.form.tiled, self.dev[3]);
    }

    /// The warp form's bytes on the same inputs.
    pub fn replaced(self: *Attn) ![]const u8 {
        const len = self.shape.k * h * d * 2;
        const out = try self.rig.scratch(len);
        try self.launch(self.plain, false, out);
        return self.rig.bytes(out, len);
    }

    pub fn count(self: *Attn) usize {
        return self.shape.k * h * d;
    }

    /// Output head i (row i / 16, head i % 16): the keys its row sees (the state, or its own question) in order.
    pub fn load(self: *Attn, i: usize, hd: *attn.Head) !void {
        const row = i / h;
        const head = i % h;
        const kh = head / (h / hk);
        for (hd.q, 0..) |*x, e| x.* = c.fromBf16(self.q[(row * h + head) * d + e]);
        var n: usize = 0;
        for (0..self.shape.k) |j| {
            if (!(row < self.state or j < self.state or self.group[j] == self.group[row])) continue;
            for (0..d) |e| {
                hd.k[n * d + e] = c.fromBf16(self.kv[keyAt(j, kh) + e]);
                hd.v[n * d + e] = c.fromBf16(self.kv[valueAt(j, kh) + e]);
            }
            n += 1;
        }
        hd.n = n;
    }

    pub fn check(self: *Attn, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.shape.k * h, scale(), try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.shape.k * h, scale(), want, mag);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.shape.k * h, scale(), acc, order, out);
    }
};

test "a pass's groups: the state's two thirds, then each question's run in order" {
    var gr: [31]i32 = undefined;
    const state = Attn.groups(&gr, 4);
    try std.testing.expectEqual(@as(usize, 21), state);
    try std.testing.expectEqualSlices(i32, &.{ -1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4 }, gr[20..]);
}
