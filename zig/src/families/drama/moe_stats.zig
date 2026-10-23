//! tf-drama-probe's routed-expert statistics: device fp64 sums per layer (drama_probe.cu), its CPU reference, the file.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const backbone = @import("backbone.zig");
const config = backbone.config;
const Allocator = std.mem.Allocator;

/// A row's class: prompt rows by whether their token is a semantic (audio) id, then every row fed after the prompt.
pub const Class = enum { prompt_text, prompt_audio, generated };
const classes = @typeInfo(Class).@"enum".field_names.len;

const E = config.experts;
const H = config.hidden;
const I = config.expert_inner;
const top = config.top_k;
const threads = 256; // drama_probe_moe_stats's block

/// A layer's fp64 sums: count, mass, reap [class][expert], then gate_up [class][expert][hidden], down [..][inner].
pub const count_at = 0;
pub const mass_at = classes * E;
pub const reap_at = 2 * classes * E;
pub const gate_up_at = 3 * classes * E;
pub const down_at = gate_up_at + classes * E * H;
pub const layer_len = down_at + classes * E * I;

/// Every layer's sums on the device, cleared by `drain`.
pub const Sums = struct {
    module: cuda.Module,
    add_fn: cuda.Function,
    weigh_fn: cuda.Function,
    buf: cuda.DeviceBuffer, // f64 [layers][layer_len]
    semantic_start: u32,
    semantic_end: u32,

    pub fn init(d: *const cuda.Driver, semantic_start: u32, semantic_end: u32) !Sums {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var module = try cuda.Module.load(d, cuda.kernels.drama_probe);
        errdefer module.unload();
        var buf = try cuda.DeviceBuffer.alloc(d, config.layers * layer_len * 8);
        errdefer buf.free();
        try buf.fill8(0, null);
        return .{ .module = module, .add_fn = try module.function("drama_probe_moe_stats"), .weigh_fn = try module.function("drama_probe_weigh"), .buf = buf, .semantic_start = semantic_start, .semantic_end = semantic_end };
    }

    pub fn deinit(m: *Sums) void {
        m.buf.free();
        m.module.unload();
        m.* = undefined;
    }

    /// Adds layer `layer`'s pairs of `r`, with `act` (bf16 [rows * top_k][expert_inner]) their SwiGLU activations.
    pub fn add(m: *const Sums, s: cuda.Stream, layer: usize, r: backbone.Routed, act: u64, generated: bool) !void {
        var a: cuda.Args = .{};
        a.add(r.ids);
        a.add(@as(i32, @intCast(m.semantic_start)));
        a.add(@as(i32, @intCast(m.semantic_end)));
        a.add(@as(i32, @intFromBool(generated)));
        a.add(r.picks);
        a.add(r.weights);
        a.add(r.input);
        a.add(act);
        a.add(r.down);
        a.add(@as(i32, @intCast(r.rows)));
        a.add(m.buf.ptr + layer * layer_len * 8);
        try cuda.launch.launch(m.add_fn, .{ .grid = .{ .x = E }, .block = .{ .x = threads } }, s, &a);
    }

    /// `add` for an NVFP4 layer: its own SwiGLU rows, and the combine's bf16(w * down) pair rows made into `weighed`.
    pub fn addFp4(m: *const Sums, s: cuda.Stream, layer: usize, r: backbone.Routed, weighed: u64, generated: bool) !void {
        var a: cuda.Args = .{};
        a.add(r.down_f32);
        a.add(r.weights);
        a.add(weighed);
        try cuda.launch.launch(m.weigh_fn, .{ .grid = .{ .x = @intCast(r.rows * top) }, .block = .{ .x = threads } }, s, &a);
        var w = r;
        w.down = weighed;
        try m.add(s, layer, w, r.act, generated);
    }

    /// Adds the device sums to `totals` (f64 [layers][layer_len]) through `staging` (as long), then clears them.
    pub fn drain(m: *const Sums, s: cuda.Stream, totals: []f64, staging: []f64) !void {
        try s.synchronize();
        try m.buf.download(0, std.mem.sliceAsBytes(staging));
        for (totals, staging) |*t, x| t.* += x;
        try m.buf.fill8(0, s.handle);
        try s.synchronize();
    }
};

/// `Model.runTraced`'s `routed` hook: each layer's SwiGLU activations by the production kernel, then `Sums.add`.
pub const Tracer = struct {
    k: *const backbone.kernels.Kernels,
    sums: Sums,
    act: cuda.DeviceBuffer, // bf16 [max_rows * top_k][expert_inner]
    weighed: ?cuda.DeviceBuffer, // bf16 [max_rows * top_k][hidden], when a layer is NVFP4
    /// Rows fed after the prompt are generated, whatever their tokens.
    generated: bool = false,

    /// For runs of up to `max_rows` rows; `semantic` is the inclusive band of audio tokens.
    pub fn init(d: *const cuda.Driver, k: *const backbone.kernels.Kernels, semantic: [2]u32, max_rows: usize) !Tracer {
        var sums = try Sums.init(d, semantic[0], semantic[1]);
        errdefer sums.deinit();
        const pairs = max_rows * top;
        var act = try cuda.DeviceBuffer.alloc(d, pairs * I * 2);
        errdefer act.free();
        // NVFP4 kernels are loaded only for a model with an NVFP4 layer, the one the weighed rows serve
        const weighed = if (k.fp4 != null) try cuda.DeviceBuffer.alloc(d, pairs * H * 2) else null;
        return .{ .k = k, .sums = sums, .act = act, .weighed = weighed };
    }

    pub fn deinit(t: *Tracer) void {
        if (t.weighed) |*w| w.free();
        t.act.free();
        t.sums.deinit();
        t.* = undefined;
    }

    pub fn routed(t: *Tracer, s: cuda.Stream, layer: usize, r: backbone.Routed) !void {
        if (r.act != 0) return t.sums.addFp4(s, layer, r, t.weighed.?.ptr, t.generated);
        // routed 0: every pair takes the shared expert's bf16 store, the value the FP8 path quantizes
        try t.k.siluMul(s, 0, 0, 0, 0, 0, r.gate_up, backbone.weights.gu_rows, t.act.ptr, r.rows * top, I);
        try t.sums.add(s, layer, r, t.act.ptr, t.generated);
    }
};

fn sq(v: u16) f64 {
    const x: f64 = @as(f32, @bitCast(@as(u32, v) << 16));
    return x * x;
}

/// `Sums.add` on host rows into one layer's `sums`, in drama_probe.cu's order: the device's sums bit for bit.
pub fn reference(sums: []f64, ids: []const i32, semantic: [2]i32, generated: bool, picks: []const i32, weights: []const f32, input: []const u16, act: []const u16, down: []const u16) void {
    for (0..E) |e| for (0..classes) |c| {
        var gu: [H]f64 = @splat(0);
        var dn: [I]f64 = @splat(0);
        var count: f64 = 0;
        var mass: f64 = 0;
        var reap: f64 = 0;
        for (picks, 0..) |pick, p| {
            const r = p / top;
            const class: usize = if (generated) 2 else @intFromBool(ids[r] >= semantic[0] and ids[r] <= semantic[1]);
            if (pick != e or class != c) continue;
            var partial: [threads]f64 = undefined;
            for (&partial, 0..) |*norm, t| {
                norm.* = 0;
                for (0..H / threads) |j| {
                    gu[t + j * threads] += sq(input[r * H + t + j * threads]);
                    norm.* += sq(down[p * H + t + j * threads]);
                }
            }
            for (&dn, act[p * I ..][0..I]) |*x, v| x.* += sq(v);
            var half: usize = threads / 2;
            while (half > 0) : (half /= 2) for (0..half) |t| {
                partial[t] += partial[t + half];
            };
            count += 1;
            mass += weights[p];
            reap += @sqrt(partial[0]);
        }
        if (count == 0) continue;
        for (sums[gate_up_at + (c * E + e) * H ..][0..H], gu) |*x, v| x.* += v;
        for (sums[down_at + (c * E + e) * I ..][0..I], dn) |*x, v| x.* += v;
        sums[count_at + c * E + e] += count;
        sums[mass_at + c * E + e] += mass;
        sums[reap_at + c * E + e] += reap;
    };
}

/// `totals` (f64 [layers][layer_len]) as the statistics safetensors at `path`; `metadata` goes in its header.
pub fn write(gpa: Allocator, path: []const u8, totals: []const f64, metadata: []const [2][]const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    for (0..totals.len / layer_len) |l| {
        const sums = totals[l * layer_len ..][0..layer_len];
        for (std.meta.tags(Class), 0..) |c, i| {
            for ([_][]const u8{ "count", "mass", "reap" }, [_]usize{ count_at, mass_at, reap_at }) |name, at| try tensors.append(a, .{
                .name = try std.fmt.allocPrint(a, "layers.{d}.{s}.{t}", .{ l, name, c }),
                .dtype = .f64,
                .shape = &.{E},
                .bytes = std.mem.sliceAsBytes(sums[at + i * E ..][0..E]),
            });
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "layers.{d}.gate_up.sumsq.{t}", .{ l, c }), .dtype = .f64, .shape = &.{ E, H }, .bytes = std.mem.sliceAsBytes(sums[gate_up_at + i * E * H ..][0 .. E * H]) });
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "layers.{d}.down.sumsq.{t}", .{ l, c }), .dtype = .f64, .shape = &.{ E, I }, .bytes = std.mem.sliceAsBytes(sums[down_at + i * E * I ..][0 .. E * I]) });
        }
    }
    try core.shard_edit.create(gpa, path, metadata, tensors.items);
}

/// Each layer's routing mass, every class summed, as ``{"layer": [256 masses]}`` (drama_expert_mask.py's input).
pub fn massJson(a: Allocator, totals: []const f64) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, '{');
    for (0..totals.len / layer_len) |l| {
        try out.print(a, "{s}\"{d}\": [", .{ if (l == 0) "" else ", ", l });
        for (0..E) |e| {
            var m: f64 = 0;
            for (0..classes) |c| m += totals[l * layer_len + mass_at + c * E + e];
            try out.print(a, "{s}{d}", .{ if (e == 0) "" else ", ", m });
        }
        try out.append(a, ']');
    }
    try out.appendSlice(a, "}\n");
    return out.items;
}

test "the reference sums a tiny run by expert and class: counts, mass, mass times down norm, squared columns" {
    const gpa = std.testing.allocator;
    const sums = try gpa.alloc(f64, layer_len);
    defer gpa.free(sums);
    @memset(sums, 0);
    const bf = struct {
        fn of(x: f32) u16 {
            return @intCast(@as(u32, @bitCast(x)) >> 16);
        }
    }.of;
    // row 0 is text (token 7), row 1 audio (token 120 of band 100..200); expert 5 takes slot 0 of both rows
    const ids = [_]i32{ 7, 120 };
    var picks: [2 * top]i32 = undefined;
    for (&picks, 0..) |*p, i| p.* = @intCast(i % top + 10);
    picks[0] = 5;
    picks[top] = 5;
    var weights: [2 * top]f32 = @splat(0.125);
    weights[0] = 0.5;
    weights[top] = 0.25;
    var input: [2 * H]u16 = @splat(0);
    input[3] = bf(2);
    input[H + 3] = bf(-0.5);
    var act: [2 * top * I]u16 = @splat(0);
    act[1] = bf(3);
    act[top * I + 1] = bf(1);
    var down: [2 * top * H]u16 = @splat(0);
    down[0] = bf(3);
    down[1] = bf(4); // pair 0's norm 5
    down[top * H + 300] = bf(-6);
    down[top * H + 2047] = bf(8); // pair 8's norm 10
    reference(sums, &ids, .{ 100, 200 }, false, &picks, &weights, &input, &act, &down);
    const at = struct {
        fn of(base: usize, c: Class, e: usize) usize {
            return base + @as(usize, @intFromEnum(c)) * E + e;
        }
    }.of;
    const col = struct {
        fn of(base: usize, width: usize, c: Class, e: usize, column: usize) usize {
            return base + (@as(usize, @intFromEnum(c)) * E + e) * width + column;
        }
    }.of;
    try std.testing.expectEqual(@as(f64, 1), sums[at(count_at, .prompt_text, 5)]);
    try std.testing.expectEqual(@as(f64, 1), sums[at(count_at, .prompt_audio, 5)]);
    try std.testing.expectEqual(@as(f64, 0.5), sums[at(mass_at, .prompt_text, 5)]);
    try std.testing.expectEqual(@as(f64, 0.25), sums[at(mass_at, .prompt_audio, 5)]);
    try std.testing.expectEqual(@as(f64, 5), sums[at(reap_at, .prompt_text, 5)]);
    try std.testing.expectEqual(@as(f64, 10), sums[at(reap_at, .prompt_audio, 5)]);
    try std.testing.expectEqual(@as(f64, 4), sums[col(gate_up_at, H, .prompt_text, 5, 3)]);
    try std.testing.expectEqual(@as(f64, 0.25), sums[col(gate_up_at, H, .prompt_audio, 5, 3)]);
    try std.testing.expectEqual(@as(f64, 9), sums[col(down_at, I, .prompt_text, 5, 1)]);
    try std.testing.expectEqual(@as(f64, 1), sums[col(down_at, I, .prompt_audio, 5, 1)]);
    try std.testing.expectEqual(@as(f64, 1), sums[at(count_at, .prompt_text, 11)]);
    try std.testing.expectEqual(@as(f64, 0.125), sums[at(mass_at, .prompt_audio, 11)]);
    try std.testing.expectEqual(@as(f64, 0.25), sums[col(gate_up_at, H, .prompt_audio, 11, 3)]);
    try std.testing.expectEqual(@as(f64, 0), sums[at(count_at, .generated, 5)]);
    // the same rows fed after the prompt land in the generated class, added to what is there
    reference(sums, &ids, .{ 100, 200 }, true, &picks, &weights, &input, &act, &down);
    try std.testing.expectEqual(@as(f64, 2), sums[at(count_at, .generated, 5)]);
    try std.testing.expectEqual(@as(f64, 15), sums[at(reap_at, .generated, 5)]);
    try std.testing.expectEqual(@as(f64, 4.25), sums[col(gate_up_at, H, .generated, 5, 3)]);
    try std.testing.expectEqual(@as(f64, 10), sums[col(down_at, I, .generated, 5, 1)]);
    try std.testing.expectEqual(@as(f64, 4), sums[col(gate_up_at, H, .prompt_text, 5, 3)]);
    var total: f64 = 0;
    for (0..classes) |c| for (0..E) |e| {
        total += sums[count_at + c * E + e];
    };
    try std.testing.expectEqual(@as(f64, 4 * top), total);
}

test "the statistics file names each layer's tensors by class, and mass.json sums the classes" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const totals = try gpa.alloc(f64, 3 * layer_len);
    defer gpa.free(totals);
    @memset(totals, 0);
    totals[2 * layer_len + mass_at + 2 * E + 255] = 1.5;
    totals[2 * layer_len + mass_at + 255] = 0.25;
    totals[2 * layer_len + down_at + (2 * E + 3) * I + 7] = 9;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/stats.safetensors", .{tmp.sub_path});
    defer gpa.free(path);
    try write(gpa, path, totals, &.{.{ "rows", "12" }});
    var file = try core.safetensors.File.open(gpa, io, path);
    defer file.close(io);
    try std.testing.expectEqual(@as(usize, 3 * 15), file.names.count());
    const down = file.get("layers.2.down.sumsq.generated").?;
    try std.testing.expect(down.is(.f64, &.{ E, I }));
    try std.testing.expectEqual(@as(f64, 9), std.mem.bytesAsSlice(f64, down.bytes)[3 * I + 7]);
    try std.testing.expect(file.get("layers.2.gate_up.sumsq.prompt_audio").?.is(.f64, &.{ E, H }));
    try std.testing.expect(file.get("layers.0.mass.generated").?.is(.f64, &.{E}));
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const mass = try massJson(arena.allocator(), totals);
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), mass, .{});
    try std.testing.expectEqual(@as(usize, 3), parsed.object.count());
    try std.testing.expectEqual(@as(f64, 1.75), parsed.object.get("2").?.array.items[255].float);
}
