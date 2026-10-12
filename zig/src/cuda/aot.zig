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

/// Where a family's glue kernels come from: our own fatbins, or a captured set in a folder.
pub const Pick = union(enum) { native, captured: []const u8, missing: []const u8 };

/// `explicit` (folder or native), else the binary's share/tensorfold/cuda/sm<capability> when it has aot.json.
pub fn pick(a: std.mem.Allocator, io: std.Io, explicit: ?[]const u8, capability: u32) !Pick {
    if (explicit) |dir| {
        if (std.mem.eql(u8, dir, "native")) return .native;
        return if (has(a, io, dir)) .{ .captured = dir } else .{ .missing = dir };
    }
    const exe = try std.process.executableDirPathAlloc(io, a);
    const dir = try std.fs.path.join(a, &.{ exe, "..", "share", "tensorfold", "cuda", try std.fmt.allocPrint(a, "sm{d}", .{capability}) });
    return if (has(a, io, dir)) .{ .captured = dir } else .native;
}

fn has(a: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "aot.json" }) catch return false;
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

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
pub fn pickVariant(specs: []const KernelJson, function: []const u8, args: []const Arg, consts: []const Const, opts: Launch) PickError!usize {
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
        if (pickVariant(self.parsed.value.kernels, function, args, consts, opts)) |i| return &self.variants[i] else |e| {
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

// ---- coverage (no GPU): the launches an engine can make, checked against an aot.json ----

/// aot.json alone (no cubins loaded): what `pickVariant` chooses from, for host-side coverage checks.
pub const Catalog = std.json.Parsed(SetJson);

pub fn loadCatalog(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Catalog {
    const path = try std.fs.path.join(gpa, &.{ dir, "aot.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    return std.json.parseFromSlice(SetJson, gpa, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

/// One launch shape an engine can request, reduced to what `pickVariant` matches on: the function, its launch options, the
/// constexprs it names and each runtime argument's type and specialization class (an int's value 1 / a multiple of
/// 16 / other, a pointer's or u64's 16-byte alignment). `args` keeps the first launch seen with this key (its values
/// are a valid representative of the class: tools/glm53/sweep_aot.py compiles the variant from them).
pub const Need = struct {
    function: []const u8,
    opts: Launch,
    args: []const Arg,
    consts: []const Const,
    count: usize = 1,
};

/// Collects `Need`s instead of launching (cuda_triton.Tri with .probe set): every distinct launch key once.
pub const Probe = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    needs: std.StringArrayHashMapUnmanaged(Need) = .empty,
    key: std.ArrayList(u8) = .empty,
    calls: usize = 0,

    pub fn init(gpa: std.mem.Allocator) Probe {
        return .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(p: *Probe) void {
        p.needs.deinit(p.gpa);
        p.key.deinit(p.gpa);
        p.arena.deinit();
        p.* = undefined;
    }

    fn class(x: i64) u8 {
        if (x == 1) return '1';
        return if (@mod(x, 16) == 0) 'D' else 'N';
    }

    /// Records one launch (what Set.runWith would have been asked for).
    pub fn add(p: *Probe, function: []const u8, args: []const Arg, consts: []const Const, opts: Launch) !void {
        p.calls += 1;
        const g = p.gpa;
        p.key.clearRetainingCapacity();
        try p.key.print(g, "{s}|w{?d}|s{?d}", .{ function, opts.num_warps, opts.num_stages });
        for (consts) |c| {
            if (c.int) |x| try p.key.print(g, "|{s}=i{d}", .{ c.name, x });
            if (c.f32) |x| try p.key.print(g, "|{s}=f{x}", .{ c.name, @as(u32, @bitCast(x)) });
        }
        for (args) |a| switch (a.value) {
            .ptr => |x| try p.key.print(g, "|{s}:{s}:{c}", .{ a.name, x.ty, @as(u8, if (x.addr % 16 == 0) 'D' else 'N') }),
            .i32 => |x| try p.key.print(g, "|{s}:i32:{c}", .{ a.name, class(x) }),
            .f32 => try p.key.print(g, "|{s}:fp32", .{a.name}),
            .u64 => |x| try p.key.print(g, "|{s}:u64:{c}", .{ a.name, @as(u8, if (x % 16 == 0) 'D' else 'N') }),
        };
        const gop = try p.needs.getOrPut(g, p.key.items);
        if (gop.found_existing) {
            gop.value_ptr.count += 1;
            return;
        }
        const ar = p.arena.allocator();
        gop.key_ptr.* = try ar.dupe(u8, p.key.items);
        gop.value_ptr.* = .{ .function = try ar.dupe(u8, function), .opts = opts, .args = try ar.dupe(Arg, args), .consts = try ar.dupe(Const, consts) };
    }

    /// Indices (into `needs`) of the launches `specs` has no single variant for: missing or ambiguous.
    pub fn missing(p: *const Probe, gpa: std.mem.Allocator, specs: []const KernelJson) ![]usize {
        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(gpa);
        for (p.needs.values(), 0..) |n, i| {
            _ = pickVariant(specs, n.function, n.args, n.consts, n.opts) catch {
                try out.append(gpa, i);
                continue;
            };
        }
        return out.toOwnedSlice(gpa);
    }

    /// The needs as JSON ({"needs": [...]}; "have": whether `specs` holds the variant, when given).
    pub fn json(p: *const Probe, gpa: std.mem.Allocator, specs: ?[]const KernelJson) ![]u8 {
        var o: std.ArrayList(u8) = .empty;
        errdefer o.deinit(gpa);
        try o.print(gpa, "{{\"generator\": \"tf-glm53-generate --mode list-variants\", \"calls\": {d}, \"needs\": [", .{p.calls});
        for (p.needs.values(), 0..) |n, i| {
            try o.print(gpa, "{s}\n {{\"fn\": \"{s}\", \"count\": {d}, ", .{ if (i > 0) "," else "", n.function, n.count });
            if (n.opts.num_warps) |w| try o.print(gpa, "\"num_warps\": {d}, ", .{w}) else try o.appendSlice(gpa, "\"num_warps\": null, ");
            if (n.opts.num_stages) |s| try o.print(gpa, "\"num_stages\": {d}, ", .{s}) else try o.appendSlice(gpa, "\"num_stages\": null, ");
            if (specs) |ks| {
                const have = if (pickVariant(ks, n.function, n.args, n.consts, n.opts)) |_| true else |_| false;
                try o.print(gpa, "\"have\": {}, ", .{have});
            }
            try o.appendSlice(gpa, "\"consts\": {");
            for (n.consts, 0..) |c, j| {
                const sep = if (j > 0) ", " else "";
                if (c.int) |x| try o.print(gpa, "{s}\"{s}\": {{\"int\": {d}}}", .{ sep, c.name, x });
                if (c.f32) |x| try o.print(gpa, "{s}\"{s}\": {{\"f32\": {d}}}", .{ sep, c.name, @as(u32, @bitCast(x)) });
            }
            try o.appendSlice(gpa, "}, \"args\": [");
            for (n.args, 0..) |a, j| {
                const sep = if (j > 0) ", " else "";
                switch (a.value) {
                    .ptr => |x| try o.print(gpa, "{s}{{\"name\": \"{s}\", \"kind\": \"ptr\", \"type\": \"{s}\", \"align16\": {}}}", .{ sep, a.name, x.ty, x.addr % 16 == 0 }),
                    .i32 => |x| try o.print(gpa, "{s}{{\"name\": \"{s}\", \"kind\": \"i32\", \"value\": {d}}}", .{ sep, a.name, x }),
                    .f32 => |x| try o.print(gpa, "{s}{{\"name\": \"{s}\", \"kind\": \"f32\", \"bits\": {d}}}", .{ sep, a.name, @as(u32, @bitCast(x)) }),
                    .u64 => |x| try o.print(gpa, "{s}{{\"name\": \"{s}\", \"kind\": \"u64\", \"value\": {d}}}", .{ sep, a.name, x }),
                }
            }
            try o.appendSlice(gpa, "]}");
        }
        try o.appendSlice(gpa, "\n]}\n");
        return o.toOwnedSlice(gpa);
    }
};

test "the probe keeps one need per specialization class and checks it against a set" {
    const parsed = try parseTestSet();
    defer parsed.deinit();
    var p = Probe.init(std.testing.allocator);
    defer p.deinit();
    const consts = [_]Const{ ci("CHK", 256), cf("SCALE", 0.0625) };
    // R 3 and 5 are one class (not 1, not a multiple of 16); R 32 another; R 1 a third
    for ([_]i32{ 3, 5, 32, 1 }) |r| try p.add("_attn", &.{ ptr("Q", "*bf16", 0x1000), int("R", r) }, &consts, .{ .num_warps = 4, .num_stages = 2 });
    try std.testing.expectEqual(@as(usize, 3), p.needs.count());
    try std.testing.expectEqual(@as(usize, 2), p.needs.values()[0].count);
    const miss = try p.missing(std.testing.allocator, parsed.value.kernels);
    defer std.testing.allocator.free(miss);
    // R 3: variant a; R 32: no variant with R divisible by 16 captured; R 1: variant d
    try std.testing.expectEqual(@as(usize, 1), miss.len);
    try std.testing.expectEqual(@as(i32, 32), p.needs.values()[miss[0]].args[1].value.i32);
    const j = try p.json(std.testing.allocator, parsed.value.kernels);
    defer std.testing.allocator.free(j);
    const doc = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, j, .{});
    defer doc.deinit();
    try std.testing.expectEqual(@as(usize, 3), doc.value.object.get("needs").?.array.items.len);
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
    try std.testing.expectEqual(@as(usize, 0), try pickVariant(ks, "_attn", &args, &consts, .{ .num_warps = 4, .num_stages = 2 }));
    try std.testing.expectEqual(@as(usize, 1), try pickVariant(ks, "_attn", &args, &consts, .{ .num_warps = 8, .num_stages = 2 }));
    try std.testing.expectEqual(@as(usize, 2), try pickVariant(ks, "_attn", &args, &consts, .{ .num_warps = 8, .num_stages = 1 }));
    // warps alone: 4 has one variant here, 8 has two (stages 2 and 1)
    try std.testing.expectEqual(@as(usize, 0), try pickVariant(ks, "_attn", &args, &consts, .{ .num_warps = 4 }));
    try std.testing.expectError(error.AmbiguousTritonVariant, pickVariant(ks, "_attn", &args, &consts, .{ .num_warps = 8 }));
    // no pin at all: the old first-match answer is refused
    try std.testing.expectError(error.AmbiguousTritonVariant, pickVariant(ks, "_attn", &args, &consts, .{}));
    try std.testing.expectError(error.MissingTritonVariant, pickVariant(ks, "_attn", &args, &consts, .{ .num_warps = 2 }));
    try std.testing.expectError(error.MissingTritonVariant, pickVariant(ks, "_attn", &args, &consts, .{ .num_warps = 4, .num_stages = 3 }));
}

test "the specialization still decides: R == 1 folds to a constexpr, a wrong constexpr misses" {
    const parsed = try parseTestSet();
    defer parsed.deinit();
    const ks = parsed.value.kernels;
    const consts = [_]Const{ ci("CHK", 256), cf("SCALE", 0.0625) };
    const one = [_]Arg{ ptr("Q", "*bf16", 0x1000), int("R", 1) };
    try std.testing.expectEqual(@as(usize, 3), try pickVariant(ks, "_attn", &one, &consts, .{ .num_warps = 4, .num_stages = 2 }));
    const unaligned = [_]Arg{ ptr("Q", "*bf16", 0x1008), int("R", 3) };
    try std.testing.expectError(error.MissingTritonVariant, pickVariant(ks, "_attn", &unaligned, &consts, .{ .num_warps = 4, .num_stages = 2 }));
    const other = [_]Const{ ci("CHK", 2048), cf("SCALE", 0.0625) };
    const three = [_]Arg{ ptr("Q", "*bf16", 0x1000), int("R", 3) };
    try std.testing.expectError(error.MissingTritonVariant, pickVariant(ks, "_attn", &three, &other, .{ .num_warps = 4, .num_stages = 2 }));
}

test "a single captured variant needs no pin, and a set without num_stages matches any stage pin" {
    const parsed = try parseTestSet();
    defer parsed.deinit();
    const ks = parsed.value.kernels;
    const x = [_]Arg{ptr("X", "*bf16", 0x2000)};
    const d = [_]Const{ci("D", 6144)};
    try std.testing.expectEqual(@as(usize, 4), try pickVariant(ks, "_norm", &x, &d, .{}));
    try std.testing.expectEqual(@as(usize, 4), try pickVariant(ks, "_norm", &x, &d, .{ .num_warps = 8, .num_stages = 3 }));
    try std.testing.expectError(error.MissingTritonVariant, pickVariant(ks, "_norm", &x, &d, .{ .num_warps = 4 }));
    try std.testing.expectError(error.MissingTritonVariant, pickVariant(ks, "_nope", &x, &d, .{}));
}
