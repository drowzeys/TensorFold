//! LFM3-Omni's CUDA kernels (lfm3_omni.cu) under the contract: a registration each, every control red.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const cases = @import("contract_lfm3_omni_cases.zig");
const data = @import("contract_lfm3_omni_data.zig");
const ref = @import("contract_lfm3_omni_ref.zig");
const num = @import("contract_lfm3_omni_num.zig");
const own = @import("contract_lfm3_omni_check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Which = cases.Which;

fn depth(s: c.Shape) usize {
    return s.k + 16;
}

fn checked(w: Which) bool {
    return switch (w) {
        .pos, .fsq, .argmax, .depth_pick, .frame_end, .hist_push => true,
        else => false,
    };
}

fn sums(w: Which) bool {
    return switch (w) {
        .rmsnorm_res, .rmsnorm_rows, .head_norm, .short_conv, .conv_rows, .attention, .attention_tiles, .layernorm, .embed_codes, .conv2d, .gemv => true,
        else => false,
    };
}

/// Stress inputs whose fp32 sums are exact in any order (integers): the enclosure needs no sum error there.
fn exactStress(w: Which) bool {
    return sums(w) and w != .attention and w != .attention_tiles and w != .layernorm;
}

pub fn spec(e: cases.Entry) c.Spec {
    const w = e.which;
    const out: c.Elem = if (checked(w)) .raw else switch (w) {
        .bf16_f32, .polar, .mel_frames, .power, .logmel => .f32,
        .gemv => if (e.cfg.f32_out) .f32 else .bf16,
        else => .bf16,
    };
    const proof: c.Sums = switch (w) {
        .gemv, .embed_codes => .{},
        .rmsnorm_res, .rmsnorm_rows, .head_norm => .{ .bf16 = 0, .why = "a bf16 sum of squares moves the rsqrt by under the output's own ulp" },
        .short_conv, .conv_rows => .{ .f16 = 0, .bf16 = 0, .why = "a sum of at most 7 taps" },
        .conv2d => .{ .f16 = 0, .bf16 = 0, .why = "a sum of at most 9 taps" },
        else => .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "softmax weights and centred squares: no exact-sum inputs" },
    };
    const nan: c.Nan = switch (w) {
        .logmel => .{ .stated = "a NaN logarithm gives the floor, as fmaxf takes the other operand" },
        .argmax => .{ .stated = "a NaN logit never wins; ties to the lower id; suppressed ids skipped" },
        .depth_pick => .{ .stated = "a NaN logit never wins; ties to the lower id; the penalty divides or multiplies by sign" },
        else => .propagate,
    };
    const tiles: [3]c.Tile = switch (w) {
        .rmsnorm_res, .layernorm => .{ .{}, .{}, .{ .size = 256, .step = 8, .min = 8 } },
        .gemv => .{ .{}, .{ .size = 8 }, .{ .size = 1024, .step = 8, .min = 8 } },
        .attention_tiles => .{ .{ .size = 16 }, .{}, .{} },
        else => .{ .{}, .{}, .{} },
    };
    return .{
        .name = e.name,
        .accum = if (sums(w)) .f32 else .none,
        .depth = depth,
        .out = out,
        .nan = nan,
        .tiles = tiles,
        .served = e.served,
        .sums = proof,
        .replaces = if (w == .gemv and e.cfg.tm > 1) "om_gemv_1 fp32 out" else null,
    };
}

/// One registration's buffers on the device, its float64 values and enclosures for the prepared inputs.
pub const Case = struct {
    rig: rig_.Rig,
    entry: cases.Entry,
    f: cuda.Function,
    base: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    d: data.Data = undefined,
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Case) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Case);

    pub fn kernel(self: *Case) g.Kernel {
        const table = if (checked(self.entry.which)) &checked_vt else &vt;
        return .{ .spec = spec(self.entry), .ctx = self, .vt = table, .control = if (self.entry.ctl != null) self.entry.fails else null };
    }

    pub fn alloc(self: *Case) std.mem.Allocator {
        return self.arena.?.allocator();
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        self.lo = &.{};
        self.hi = &.{};
        self.d = .{ .which = self.entry.which, .cfg = self.entry.cfg, .shape = shape, .fill = fill };
        if (!try data.prepare(&self.d, self.alloc(), seed)) return false;
        for (self.d.bufs[0..self.d.nb]) |*b| b.dev = switch (b.role) {
            .in => try self.rig.input(b.host),
            .out, .inout => try self.rig.output(b.host.len),
        };
        return true;
    }

    pub fn run(self: *Case) !void {
        for (self.d.bufs[0..self.d.nb]) |b| if (b.role == .inout) {
            const buf: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = b.dev, .len = b.host.len };
            try buf.upload(0, b.host);
        };
        var args: cuda.Args = .{};
        const l = @import("contract_lfm3_omni_launch.zig").launch(&self.d, &args);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = l.grid[0], .y = l.grid[1] }, .block = .{ .x = l.block }, .shared = l.shared }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        var n: usize = 0;
        for (self.d.bufs[0..self.d.nb]) |b| if (b.role != .in) {
            n += if (checked(self.d.which)) b.host.len else b.host.len / @as(usize, if (b.elem == .bf16) 2 else 4);
        };
        return n;
    }

    fn scratch(self: *Case, comptime iv: bool) !struct { []num.Num(iv).T, []f64 } {
        const s = self.d.shape;
        const len = @max(@max(s.k, s.n), 8192) + 64;
        return .{ try self.alloc().alloc(num.Num(iv).T, len), try self.alloc().alloc(f64, len) };
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const p = try self.scratch(false);
        ref.replay(false, .{}, &self.d, want, p[0], p[1]);
        if (self.d.fill == .nonfinite) return;
        const n = want.len;
        const iv = try self.alloc().alloc(num.I, n);
        const q = try self.scratch(true);
        const mode: num.Mode = .{ .exact_sums = self.d.fill == .stress and exactStress(self.d.which) };
        ref.replay(true, .{ .mode = mode }, &self.d, iv, q[0], q[1]);
        self.lo = try self.alloc().alloc(f64, n);
        self.hi = try self.alloc().alloc(f64, n);
        for (iv, self.lo, self.hi, want) |x, *l, *h, w| {
            l.* = @min(x.lo, w);
            h.* = @max(x.hi, w);
        }
    }

    /// The enclosure's reach on either side of the float64 value: the shapes and sums gates' allowance.
    pub fn bounds(self: *Case, want: []const f64, _: []const f64, allow: []f64) !void {
        if (self.lo.len != want.len) return;
        for (allow, want, self.lo, self.hi) |*a, w, l, h| a.* = @max(h - w, w - l) + 0x1p-140;
    }

    pub fn intervals(self: *Case, low: []f64, high: []f64) !void {
        if (self.lo.len != low.len) {
            @memset(low, -std.math.floatMax(f64));
            @memset(high, std.math.floatMax(f64));
            return;
        }
        @memcpy(low, self.lo);
        @memcpy(high, self.hi);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const p = try self.scratch(false);
        ref.replay(false, .{ .mode = .{ .acc = acc, .order = order } }, &self.d, out, p[0], p[1]);
        if (self.d.which == .gemv and self.d.cfg.f32_out) for (out) |*o| {
            o.* = @as(f32, @floatCast(o.*));
        };
    }

    /// om_gemv_1 row by row on the same inputs: the bytes a few-row form must equal.
    pub fn replaced(self: *Case) ![]const u8 {
        const sh = self.d.shape;
        const yf = try self.rig.scratch(sh.rows * sh.n * 4);
        for (0..sh.rows) |t| {
            var a: cuda.Args = .{};
            a.add(self.d.ptr(0));
            a.add(self.d.ptr(1) + t * sh.k * 2);
            for ([_]i32{ @intCast(sh.k), @intCast(sh.k), @intCast(sh.n), 1 }) |x| a.add(x);
            a.add(@as(u64, 0));
            a.add(@as(i32, @intCast(sh.n)));
            a.add(yf + t * sh.n * 4);
            for ([_]i64{ 0, 0, 0 }) |x| a.add(x);
            try cuda.launch.launch(self.base.?, .{ .grid = .{ .x = @intCast((sh.n + 7) / 8) }, .block = .{ .x = 256 }, .shared = @intCast(sh.k * 2) }, self.rig.s, &a);
        }
        return self.rig.bytes(yf, sh.rows * sh.n * 4);
    }
};

/// A registration whose outputs are ids, codes or exact values: its own check at every gate, no enclosure.
const checked_vt: g.VTable = blk: {
    var v = g.hooks(Case);
    v.check = checkHook;
    v.bounds = null;
    v.intervals = null;
    v.simulate = null;
    v.replaced = null;
    break :blk v;
};

fn checkHook(ctx: *anyopaque, _: g.Gate) anyerror!?[]const u8 {
    return own.verify(@ptrCast(@alignCast(ctx)));
}

/// Every registration and control onto `kernels`; the controls' symbols come from the contract_controls image.
pub const Set = struct {
    mod: ?cuda.Module = null,
    list: []Case = &.{},
    gpa: ?std.mem.Allocator = null,

    pub fn add(self: *Set, gpu: check_.Gpu, s: cuda.Stream, ctl_mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        const a = gpu.gpa;
        self.gpa = a;
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.lfm3_omni);
        const m = self.mod.?;
        self.list = try a.alloc(Case, cases.entries.len);
        const base = try m.function("om_gemv_1");
        try base.allowDynamicShared(96 << 10);
        for (cases.entries, self.list) |e, *x| {
            const f = if (e.ctl) |sym| try ctl_mod.function(sym) else try m.function(e.symbol);
            switch (e.which) {
                .gemv => try f.allowDynamicShared(96 << 10),
                .attention, .attention_tiles => try f.allowDynamicShared(64 << 10),
                else => {},
            }
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .f = f, .base = base };
        }
        for (self.list) |*x| try kernels.append(a, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.list) |*x| x.deinit();
        if (self.gpa) |a| a.free(self.list);
        if (self.mod) |*m| m.unload();
    }

    /// A set on the heap, its registrations already on `kernels`.
    pub fn create(gpu: check_.Gpu, s: cuda.Stream, ctl_mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Set {
        const self = try gpu.gpa.create(Set);
        self.* = .{};
        errdefer gpu.gpa.destroy(self);
        try self.add(gpu, s, ctl_mod, kernels);
        return self;
    }

    pub fn destroy(self: *Set) void {
        const a = self.gpa.?;
        self.deinit();
        a.destroy(self);
    }
};

test {
    _ = num;
}
