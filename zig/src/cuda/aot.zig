//! A captured set of Triton cubins (aot.json + cubins/): each launch picks the variant Triton itself would have picked.
//! Variants of one function with the same constexprs and specialization but other launch options (num_warps,
//! num_stages: Python passes them per call site, e.g. GLM-5.3's decode vs prompt attention tilings) are told apart by
//! the caller's `Launch`; a launch that leaves them open while captured variants differ in them is refused.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const triton = @import("triton.zig");

const ParamJson = struct { name: []const u8, type: []const u8, div16: bool, nospec: bool };
const ConstJson = struct { int: ?i64 = null, f32: ?u32 = null };
const KernelJson = struct {
    @"fn": []const u8,
    hash: []const u8,
    name: []const u8,
    num_warps: u32,
    num_stages: u32 = 0, // 0: not recorded (sets packed before num_stages was kept); such a variant matches any pin
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch: u32 = 0,
    global_align: u32 = 1,
    profile_scratch: u32 = 0,
    pdl: bool = false,
    params: []ParamJson,
    consts: std.json.ArrayHashMap(ConstJson),
};
const SetJson = struct { kernels: []KernelJson };

/// One argument of a launch, by the kernel's parameter name.
pub const Arg = struct {
    name: []const u8,
    value: Value,

    pub const Value = union(enum) { ptr: struct { addr: u64, ty: []const u8 }, i32: i32, f32: f32, u64: u64 };
};

/// A constexpr the call site compiled the kernel with (Python's keyword arguments): ints, bools, fp32 bits.
pub const Const = struct { name: []const u8, int: ?i64 = null, f32: ?f32 = null };

pub fn ptr(name: []const u8, ty: []const u8, addr: u64) Arg {
    return .{ .name = name, .value = .{ .ptr = .{ .addr = addr, .ty = ty } } };
}
pub fn int(name: []const u8, v: i32) Arg {
    return .{ .name = name, .value = .{ .i32 = v } };
}
pub fn float(name: []const u8, v: f32) Arg {
    return .{ .name = name, .value = .{ .f32 = v } };
}
pub fn word(name: []const u8, v: u64) Arg {
    return .{ .name = name, .value = .{ .u64 = v } };
}
pub fn ci(name: []const u8, v: i64) Const {
    return .{ .name = name, .int = v };
}
pub fn cf(name: []const u8, v: f32) Const {
    return .{ .name = name, .f32 = v };
}

/// The launch options Python passes beside the arguments (`num_warps=`, `num_stages=`); null: whatever was captured.
pub const Launch = struct {
    num_warps: ?u32 = null,
    num_stages: ?u32 = null,

    fn admits(self: Launch, k: KernelJson) bool {
        if (self.num_warps) |w| if (k.num_warps != w) return false;
        if (self.num_stages) |s| if (k.num_stages != 0 and k.num_stages != s) return false;
        return true;
    }
};

pub const PickError = error{ MissingTritonVariant, AmbiguousTritonVariant };

/// Index into `specs` of `function`'s variant for this launch: constexprs, specialization and the launch options.
/// Several matches that differ in num_warps or num_stages are refused unless `opts` pins the one meant (the first
/// match is no answer: it is whichever sorted first in aot.json).
pub fn pick(specs: []const KernelJson, function: []const u8, args: []const Arg, consts: []const Const, opts: Launch) PickError!usize {
    var found: ?usize = null;
    for (specs, 0..) |k, i| {
        if (!std.mem.eql(u8, k.@"fn", function)) continue;
        if (!opts.admits(k) or !matches(k, args, consts)) continue;
        if (found) |f| {
            const a = specs[f];
            if (a.num_warps != k.num_warps or a.num_stages != k.num_stages) return error.AmbiguousTritonVariant;
            continue;
        }
        found = i;
    }
    return found orelse error.MissingTritonVariant;
}

const Variant = struct { spec: KernelJson, kernel: triton.Kernel };

pub const Set = struct {
    parsed: std.json.Parsed(SetJson),
    variants: []Variant,
    gpa: std.mem.Allocator,

    /// Loads every cubin listed in `dir`/aot.json into its own module.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, device: abi.Device, dir: []const u8) !Set {
        const path = try std.fs.path.join(gpa, &.{ dir, "aot.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(SetJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer parsed.deinit();
        const variants = try gpa.alloc(Variant, parsed.value.kernels.len);
        var n: usize = 0;
        errdefer {
            for (variants[0..n]) |*v| v.kernel.unload();
            gpa.free(variants);
        }
        for (parsed.value.kernels) |k| {
            if (k.global_scratch != 0 or k.profile_scratch != 0) return error.ScratchUnsupported;
            const file = try std.fmt.allocPrint(gpa, "{s}/cubins/{s}.cubin", .{ dir, k.hash });
            defer gpa.free(file);
            const cubin = try std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(1 << 26), .@"16", null);
            defer gpa.free(cubin);
            const name_z = try gpa.dupeSentinel(u8, k.name, 0);
            defer gpa.free(name_z);
            const meta: triton.Meta = .{ .name = k.name, .num_warps = k.num_warps, .num_ctas = k.num_ctas, .shared = k.shared, .launch_pdl = k.pdl };
            variants[n] = .{ .spec = k, .kernel = try triton.Kernel.load(d, device, cubin, meta, name_z) };
            n += 1;
        }
        return .{ .parsed = parsed, .variants = variants, .gpa = gpa };
    }

    pub fn deinit(self: *Set) void {
        for (self.variants) |*v| v.kernel.unload();
        self.gpa.free(self.variants);
        self.parsed.deinit();
        self.* = undefined;
    }

    /// The variant of `function` compiled for these constexprs, this specialization and these launch options.
    pub fn find(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const) !*const Variant {
        return self.findWith(function, args, consts, .{});
    }

    pub fn findWith(self: *const Set, function: []const u8, args: []const Arg, consts: []const Const, opts: Launch) !*const Variant {
        // variants[i] was loaded from parsed.value.kernels[i]
        if (pick(self.parsed.value.kernels, function, args, consts, opts)) |i| return &self.variants[i] else |e| {
            if (e == error.AmbiguousTritonVariant) {
                std.log.err("several captured Triton variants of {s} match; pin num_warps/num_stages (Launch)", .{function});
                return e;
            }
        }
        std.log.err("no captured Triton variant of {s} for this launch (warps {?d}, stages {?d}):", .{ function, opts.num_warps, opts.num_stages });
        for (args) |a| switch (a.value) {
            .ptr => |p| std.log.err("  {s}: {s} at {x} (16-aligned {})", .{ a.name, p.ty, p.addr, p.addr % 16 == 0 }),
            .i32 => |x| std.log.err("  {s}: i32 {d}", .{ a.name, x }),
            .f32 => |x| std.log.err("  {s}: fp32 {d}", .{ a.name, x }),
            .u64 => |x| std.log.err("  {s}: u64 {d}", .{ a.name, x }),
        };
        return error.MissingTritonVariant;
    }

    /// The smallest value of constexpr `name` at or above `at_least` among `function`'s variants.
    pub fn smallestConst(self: *const Set, function: []const u8, name: []const u8, at_least: i64) ?i64 {
        var best: ?i64 = null;
        for (self.variants) |v| {
            if (!std.mem.eql(u8, v.spec.@"fn", function)) continue;
            const c = (v.spec.consts.map.get(name) orelse continue).int orelse continue;
            if (c >= at_least and (best == null or c < best.?)) best = c;
        }
        return best;
    }

    /// Launches `function` on `grid` as Triton's launcher would: runtime arguments in the variant's order, scratch null.
    pub fn run(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const) !void {
        return self.runWith(stream, function, grid, args, consts, .{});
    }

    /// `run` with the call site's launch options (num_warps / num_stages) pinned.
    pub fn runWith(self: *const Set, stream: Stream, function: []const u8, grid: [3]u32, args: []const Arg, consts: []const Const, opts: Launch) !void {
        const v = try self.findWith(function, args, consts, opts);
        var packed_args: launch.Args = .{};
        for (v.spec.params) |p| {
            const a = lookup(args, p.name).?;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        try v.kernel.launchOn(.{ .x = grid[0], .y = grid[1], .z = grid[2] }, stream, &packed_args, .{}, &.{});
    }
};

fn lookup(args: []const Arg, name: []const u8) ?Arg {
    for (args) |a| if (std.mem.eql(u8, a.name, name)) return a;
    return null;
}

fn matches(k: KernelJson, args: []const Arg, consts: []const Const) bool {
    for (consts) |c| {
        const got = k.consts.map.get(c.name) orelse return false;
        if (c.int) |x| if (got.int == null or got.int.? != x) return false;
        if (c.f32) |x| if (got.f32 == null or got.f32.? != @as(u32, @bitCast(x))) return false;
    }
    var runtime: usize = 0;
    for (args) |a| {
        const param = for (k.params) |p| {
            if (std.mem.eql(u8, p.name, a.name)) break p;
        } else null;
        switch (a.value) {
            .i32 => |x| {
                if (param == null) {
                    // an int Triton folded: only the value 1 is ever specialized to a constexpr
                    const got = k.consts.map.get(a.name) orelse return false;
                    if (x != 1 or got.int == null or got.int.? != 1) return false;
                    continue;
                }
                const p = param.?;
                if (!std.mem.eql(u8, p.type, "i32")) return false;
                if (!p.nospec and x == 1) return false;
                if (p.div16 != (!p.nospec and @mod(x, 16) == 0)) return false;
            },
            .ptr => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, x.ty) or p.div16 != (x.addr % 16 == 0)) return false;
            },
            .f32 => {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "fp32")) return false;
            },
            .u64 => |x| {
                const p = param orelse return false;
                if (!std.mem.eql(u8, p.type, "u64") or p.div16 != (x % 16 == 0)) return false;
            },
        }
        runtime += 1;
    }
    return runtime == k.params.len;
}

// ---- host tests (no GPU): variant choice from an aot.json as aot_pack.py writes it ----

const test_set =
    \\{"kernels": [
    \\ {"fn": "_attn", "hash": "a", "name": "_attn", "num_warps": 4, "num_stages": 2,
    \\  "params": [{"name": "Q", "type": "*bf16", "div16": true, "nospec": false},
    \\             {"name": "R", "type": "i32", "div16": false, "nospec": false}],
    \\  "consts": {"CHK": {"int": 256}, "SCALE": {"f32": 1031798784}, "BASE": {"none": true}}},
    \\ {"fn": "_attn", "hash": "b", "name": "_attn", "num_warps": 8, "num_stages": 2,
    \\  "params": [{"name": "Q", "type": "*bf16", "div16": true, "nospec": false},
    \\             {"name": "R", "type": "i32", "div16": false, "nospec": false}],
    \\  "consts": {"CHK": {"int": 256}, "SCALE": {"f32": 1031798784}, "BASE": {"none": true}}},
    \\ {"fn": "_attn", "hash": "c", "name": "_attn", "num_warps": 8, "num_stages": 1,
    \\  "params": [{"name": "Q", "type": "*bf16", "div16": true, "nospec": false},
    \\             {"name": "R", "type": "i32", "div16": false, "nospec": false}],
    \\  "consts": {"CHK": {"int": 256}, "SCALE": {"f32": 1031798784}, "BASE": {"none": true}}},
    \\ {"fn": "_attn", "hash": "d", "name": "_attn", "num_warps": 4, "num_stages": 2,
    \\  "params": [{"name": "Q", "type": "*bf16", "div16": true, "nospec": false}],
    \\  "consts": {"CHK": {"int": 256}, "SCALE": {"f32": 1031798784}, "R": {"int": 1}, "BASE": {"none": true}}},
    \\ {"fn": "_norm", "hash": "e", "name": "_norm", "num_warps": 8,
    \\  "params": [{"name": "X", "type": "*bf16", "div16": true, "nospec": false}],
    \\  "consts": {"D": {"int": 6144}}}
    \\]}
;

fn parseTestSet() !std.json.Parsed(SetJson) {
    return std.json.parseFromSlice(SetJson, std.testing.allocator, test_set, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

test "variants that differ only in num_warps / num_stages are told apart by the launch options" {
    const parsed = try parseTestSet();
    defer parsed.deinit();
    const ks = parsed.value.kernels;
    const args = [_]Arg{ ptr("Q", "*bf16", 0x1000), int("R", 3) };
    const consts = [_]Const{ ci("CHK", 256), cf("SCALE", 0.0625) };
    try std.testing.expectEqual(@as(usize, 0), try pick(ks, "_attn", &args, &consts, .{ .num_warps = 4, .num_stages = 2 }));
    try std.testing.expectEqual(@as(usize, 1), try pick(ks, "_attn", &args, &consts, .{ .num_warps = 8, .num_stages = 2 }));
    try std.testing.expectEqual(@as(usize, 2), try pick(ks, "_attn", &args, &consts, .{ .num_warps = 8, .num_stages = 1 }));
    // warps alone: 4 has one variant here, 8 has two (stages 2 and 1)
    try std.testing.expectEqual(@as(usize, 0), try pick(ks, "_attn", &args, &consts, .{ .num_warps = 4 }));
    try std.testing.expectError(error.AmbiguousTritonVariant, pick(ks, "_attn", &args, &consts, .{ .num_warps = 8 }));
    // no pin at all: the old first-match answer is refused
    try std.testing.expectError(error.AmbiguousTritonVariant, pick(ks, "_attn", &args, &consts, .{}));
    try std.testing.expectError(error.MissingTritonVariant, pick(ks, "_attn", &args, &consts, .{ .num_warps = 2 }));
    try std.testing.expectError(error.MissingTritonVariant, pick(ks, "_attn", &args, &consts, .{ .num_warps = 4, .num_stages = 3 }));
}

test "the specialization still decides: R == 1 folds to a constexpr, a wrong constexpr misses" {
    const parsed = try parseTestSet();
    defer parsed.deinit();
    const ks = parsed.value.kernels;
    const consts = [_]Const{ ci("CHK", 256), cf("SCALE", 0.0625) };
    const one = [_]Arg{ ptr("Q", "*bf16", 0x1000), int("R", 1) };
    try std.testing.expectEqual(@as(usize, 3), try pick(ks, "_attn", &one, &consts, .{ .num_warps = 4, .num_stages = 2 }));
    const unaligned = [_]Arg{ ptr("Q", "*bf16", 0x1008), int("R", 3) };
    try std.testing.expectError(error.MissingTritonVariant, pick(ks, "_attn", &unaligned, &consts, .{ .num_warps = 4, .num_stages = 2 }));
    const other = [_]Const{ ci("CHK", 2048), cf("SCALE", 0.0625) };
    const three = [_]Arg{ ptr("Q", "*bf16", 0x1000), int("R", 3) };
    try std.testing.expectError(error.MissingTritonVariant, pick(ks, "_attn", &three, &other, .{ .num_warps = 4, .num_stages = 2 }));
}

test "a single captured variant needs no pin, and a set without num_stages matches any stage pin" {
    const parsed = try parseTestSet();
    defer parsed.deinit();
    const ks = parsed.value.kernels;
    const x = [_]Arg{ptr("X", "*bf16", 0x2000)};
    const d = [_]Const{ci("D", 6144)};
    try std.testing.expectEqual(@as(usize, 4), try pick(ks, "_norm", &x, &d, .{}));
    try std.testing.expectEqual(@as(usize, 4), try pick(ks, "_norm", &x, &d, .{ .num_warps = 8, .num_stages = 3 }));
    try std.testing.expectError(error.MissingTritonVariant, pick(ks, "_norm", &x, &d, .{ .num_warps = 4 }));
    try std.testing.expectError(error.MissingTritonVariant, pick(ks, "_nope", &x, &d, .{}));
}
