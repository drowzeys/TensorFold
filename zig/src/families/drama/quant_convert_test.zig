//! tf-drama-quant on a synthetic FP8 checkpoint (generic shapes, two shards), and the encoder through packOne.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const convert = @import("quant_convert.zig");

const st = core.safetensors;
const shard_edit = core.shard_edit;
const nvfp4 = core.quant.nvfp4;
const fp8 = core.quant.fp8;
const Allocator = std.mem.Allocator;
const testing = std.testing;

pub const experts = 3;
pub const hidden = 160; // five k32 groups: no dimension is a power of two
pub const inner = 64;
pub const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
pub const shard_names = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };

/// Layer 0 and experts 0-1 of layer 1 in the first shard; expert 2 of layer 1 and layer 2 in the second.
fn shardOf(layer: usize, expert: usize) usize {
    return @intFromBool(layer * experts + expert >= experts + 2);
}

fn fill(rnd: std.Random, arena: Allocator, n: usize, fp8_codes: bool) ![]u8 {
    const b = try arena.alloc(u8, n);
    for (b) |*x| x.* = if (fp8_codes) rnd.uintLessThan(u8, 0x7F) | (rnd.int(u8) & 0x80) else rnd.int(u8);
    return b;
}

fn scaleInv(rnd: std.Random, arena: Allocator, n: usize) ![]u8 {
    const v = try arena.alloc(f32, n);
    for (v) |*x| x.* = @floatCast(1e-4 + rnd.float(f64) * 1e-2);
    return std.mem.sliceAsBytes(v);
}

/// SRC: two shards whose experts interleave with BF16 tensors, an index, config.json, a tokenizer and a cache dir.
fn fixture(arena: Allocator, io: std.Io, tmp: std.Io.Dir, dir: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    var shards: [2]std.ArrayList(shard_edit.New) = .{ .empty, .empty };
    try shards[0].append(arena, .{ .name = "model.language_model.embed_tokens.weight", .dtype = .bf16, .shape = &.{ 8, hidden }, .bytes = try fill(rnd, arena, 16 * hidden, false) });
    for (0..3) |l| {
        const pre = try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.mlp.", .{l});
        const router = try std.fmt.allocPrint(arena, "{s}gate.weight", .{pre});
        try shards[shardOf(l, 0)].append(arena, .{ .name = router, .dtype = .bf16, .shape = &.{ experts, hidden }, .bytes = try fill(rnd, arena, 2 * experts * hidden, false) });
        for (0..experts) |e| for (projs) |p| {
            const down = p[0] == 'd';
            const n: usize, const k: usize = if (down) .{ hidden, inner } else .{ inner, hidden };
            const name = try std.fmt.allocPrint(arena, "{s}experts.{d}.{s}.weight", .{ pre, e, p });
            const inv = try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name});
            const nb, const kb = .{ (n + 127) / 128, (k + 127) / 128 };
            const s = &shards[shardOf(l, e)];
            try s.append(arena, .{ .name = name, .dtype = .f8_e4m3, .shape = try arena.dupe(usize, &.{ n, k }), .bytes = try fill(rnd, arena, n * k, true) });
            try s.append(arena, .{ .name = inv, .dtype = .f32, .shape = try arena.dupe(usize, &.{ nb, kb }), .bytes = try scaleInv(rnd, arena, nb * kb) });
            // an FP8 activation scale C1 has no place for: dropped where converted, kept where the layer stays FP8
            if (e == 0 and p[0] == 'g') try s.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}experts.0.gate_proj.input_scale", .{pre}), .dtype = .f32, .shape = &.{}, .bytes = try scaleInv(rnd, arena, 1) });
        };
        const shared = try std.fmt.allocPrint(arena, "{s}shared_expert.gate_proj.weight", .{pre});
        try shards[shardOf(l, experts - 1)].append(arena, .{ .name = shared, .dtype = .bf16, .shape = &.{ inner, hidden }, .bytes = try fill(rnd, arena, 2 * inner * hidden, false) });
    }
    try shards[1].append(arena, .{ .name = "lm_head.weight", .dtype = .bf16, .shape = &.{ 4, hidden }, .bytes = try fill(rnd, arena, 8 * hidden, false) });
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(arena, "{\"metadata\": {\"total_size\": 1}, \"weight_map\": {");
    for (shards, shard_names, 0..) |s, file, i| {
        try shard_edit.create(arena, try std.fs.path.join(arena, &.{ dir, file }), &.{.{ "format", "pt" }}, s.items);
        for (s.items, 0..) |t, j| try index.print(arena, "{s}\"{s}\": \"{s}\"", .{ if (i + j == 0) "" else ", ", t.name, file });
    }
    try index.appendSlice(arena, "}}\n");
    try tmp.writeFile(io, .{ .sub_path = "src/model.safetensors.index.json", .data = index.items });
    try tmp.writeFile(io, .{ .sub_path = "src/config.json", .data = "{\"model_type\": \"fish_omni\"}\n" });
    try tmp.writeFile(io, .{ .sub_path = "src/tokenizer.json", .data = "{\"tokens\": []}\n" });
    try tmp.createDirPath(io, "src/.cache/huggingface");
    try tmp.writeFile(io, .{ .sub_path = "src/.cache/huggingface/x.metadata", .data = "copied at its relative path\n" });
}

pub const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    root: []const u8,

    pub fn init() !Fixture {
        var f: Fixture = .{ .arena_state = .init(testing.allocator), .tmp = testing.tmpDir(.{}), .root = undefined };
        errdefer f.deinit();
        const arena = f.arena_state.allocator();
        f.root = try f.tmp.dir.realPathFileAlloc(testing.io, ".", arena);
        try f.tmp.dir.createDir(testing.io, "src", .default_dir);
        try fixture(arena, testing.io, f.tmp.dir, try f.path("src"));
        return f;
    }

    pub fn deinit(f: *Fixture) void {
        f.tmp.cleanup();
        f.arena_state.deinit();
    }

    pub fn path(f: *Fixture, sub: []const u8) ![]const u8 {
        return std.fs.path.join(f.arena_state.allocator(), &.{ f.root, sub });
    }

    pub fn read(f: *Fixture, sub: []const u8) ![]u8 {
        return f.tmp.dir.readFileAlloc(testing.io, sub, f.arena_state.allocator(), .limited(1 << 24));
    }

    pub fn json(f: *Fixture, sub: []const u8) !std.json.Value {
        return std.json.parseFromSliceLeaky(std.json.Value, f.arena_state.allocator(), try f.read(sub), .{});
    }

    pub fn open(f: *Fixture, dir: []const u8, shard: usize) !st.File {
        return st.File.open(testing.allocator, testing.io, try std.fs.path.join(f.arena_state.allocator(), &.{ f.root, dir, shard_names[shard] }));
    }
};

/// A shard's tensor names in data order.
pub fn ordered(arena: Allocator, file: *const st.File) ![]const []const u8 {
    const names = try arena.dupe([]const u8, file.names.keys());
    std.mem.sort([]const u8, names, file, struct {
        fn lt(f: *const st.File, a: []const u8, b: []const u8) bool {
            return f.names.get(a).?.begin < f.names.get(b).?.begin;
        }
    }.lt);
    return names;
}

/// The SRC projection as C2 step 1 decodes it: fp8.reference, the FP8 value times its scale_inv in f64.
pub fn dequant(arena: Allocator, w: st.Tensor, inv: st.Tensor) ![]f64 {
    const out = try arena.alloc(f64, w.numel());
    try fp8.reference(.{ .weight = w, .scale_inv = inv }, out);
    return out;
}

fn isExpertWeight(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".mlp.experts.") != null and std.mem.endsWith(u8, name, ".weight");
}

/// SHA-256 of every file SRC holds at its top level.
fn sources(f: *Fixture) ![5][32]u8 {
    const files = shard_names ++ [_][]const u8{ "model.safetensors.index.json", "config.json", "tokenizer.json" };
    var out: [5][32]u8 = undefined;
    for (&out, files) |*h, name| std.crypto.hash.sha2.Sha256.hash(try f.read(try std.fs.path.join(f.arena_state.allocator(), &.{ "src", name })), h, .{});
    return out;
}

/// The NVFP4 triple of `name` in `b` against the encoder run on its SRC dequant with importance `h` (null: 1).
fn expectEncoded(arena: Allocator, a: *const st.File, b: *const st.File, name: []const u8, h: ?[]const f64) !void {
    const w = a.get(name).?;
    const n, const k = .{ w.shape[0], w.shape[1] };
    const ones = try arena.alloc(f64, k);
    @memset(ones, 1);
    const codes = try arena.alloc(u8, n * k / 2);
    const scales = try arena.alloc(u8, n * k / 16);
    const sum = nvfp4.encode(try dequant(arena, w, a.get(try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name})).?), h orelse ones, n, k, codes, scales);
    const base = name[0 .. name.len - "weight".len];
    const got = b.get(name).?;
    try testing.expect(got.is(.u8, &.{ n, k / 2 }));
    try testing.expectEqualSlices(u8, codes, got.bytes);
    const s = b.get(try std.fmt.allocPrint(arena, "{s}weight_scale", .{base})).?;
    try testing.expect(s.is(.f8_e4m3, &.{ n, k / 16 }));
    try testing.expectEqualSlices(u8, scales, s.bytes);
    const g2 = b.get(try std.fmt.allocPrint(arena, "{s}weight_scale_2", .{base})).?;
    try testing.expect(g2.is(.f32, &.{}));
    try testing.expectEqualSlices(u8, std.mem.asBytes(&sum.global), g2.bytes);
}

test "a synthetic FP8 checkpoint converts: order, untouched bytes, encoded experts, index, files and provenance" {
    var f = try Fixture.init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const io = testing.io;
    const before = try sources(&f);
    const s = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("dst"), .{ .threads = 3 });
    try testing.expectEqual(convert.Summary{ .shards = 2, .converted = 27, .dense = 0, .copied = 3, .bytes = s.bytes }, s);
    const index = try f.json("dst/model.safetensors.index.json");
    const map = index.object.get("weight_map").?.object;
    var listed: usize = 0;
    var total: u64 = 0;
    for (0..2) |i| {
        var a = try f.open("src", i);
        defer a.close(io);
        var b = try f.open("dst", i);
        defer b.close(io);
        const meta = try std.json.parseFromSliceLeaky(std.json.Value, arena, b.map.memory[8..b.data], .{});
        try testing.expectEqualStrings("pt", meta.object.get("__metadata__").?.object.get("format").?.string);
        var want: std.ArrayList([]const u8) = .empty;
        for (try ordered(arena, &a)) |name| {
            if (std.mem.endsWith(u8, name, "_scale_inv") or std.mem.endsWith(u8, name, ".input_scale")) continue;
            try want.append(arena, name);
            if (!isExpertWeight(name)) continue;
            try want.append(arena, try std.fmt.allocPrint(arena, "{s}_scale", .{name}));
            try want.append(arena, try std.fmt.allocPrint(arena, "{s}_scale_2", .{name}));
            try expectEncoded(arena, &a, &b, name, null);
        }
        const got = try ordered(arena, &b);
        try testing.expectEqual(want.items.len, got.len);
        for (want.items, got) |x, y| {
            try testing.expectEqualStrings(x, y);
            try testing.expectEqualStrings(shard_names[i], map.get(y).?.string);
            const t = b.get(y).?;
            total += t.bytes.len;
            if (std.mem.indexOf(u8, y, ".mlp.experts.") != null) try testing.expect(isExpertWeight(y) or std.mem.endsWith(u8, y, ".weight_scale") or std.mem.endsWith(u8, y, ".weight_scale_2"));
            if (isExpertWeight(y)) continue;
            if (a.get(y)) |src| try testing.expect(src.is(t.dtype, t.shape[0..t.rank]) and std.mem.eql(u8, src.bytes, t.bytes));
        }
        listed += got.len;
    }
    try testing.expectEqual(listed, map.count());
    try testing.expectEqual(@as(i64, @intCast(total)), index.object.get("metadata").?.object.get("total_size").?.integer);
    try testing.expectEqual(total, s.bytes);
    try testing.expectEqualStrings(try f.read("src/config.json"), try f.read("dst/config.json"));
    try testing.expectEqualStrings(try f.read("src/tokenizer.json"), try f.read("dst/tokenizer.json"));
    try testing.expectEqualStrings(try f.read("src/.cache/huggingface/x.metadata"), try f.read("dst/.cache/huggingface/x.metadata"));
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(io, "dst.partial", .{}));
    const prov = (try f.json("dst/quant-provenance.json")).object.get("experts").?.object;
    try testing.expectEqual(std.json.Value.null, prov.get("stats").?);
    try testing.expectEqual(@as(usize, 3), prov.get("converted_layers").?.array.items.len);
    const tensors = prov.get("tensors").?.array.items;
    try testing.expectEqual(@as(usize, 27), tensors.len);
    for (tensors) |t| {
        const rel = t.object.get("rel").?.float;
        try testing.expect(rel > 0 and rel < 0.2 and rel == t.object.get("err_h").?.float and t.object.get("global").?.float > 0);
    }
    const groups = prov.get("projections").?.array.items;
    try testing.expectEqual(@as(usize, 3 * projs.len), groups.len);
    for (groups) |g| try testing.expectEqual(@as(i64, experts), g.object.get("defined").?.integer);
    try testing.expectEqual(before, try sources(&f));
}

test "any thread count writes the same shards, and --keep-fp8 layers keep every byte" {
    var f = try Fixture.init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const io = testing.io;
    _ = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("one"), .{ .threads = 1 });
    _ = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("four"), .{ .threads = 4 });
    for (shard_names) |name| try testing.expectEqualSlices(u8, try f.read(try std.fs.path.join(arena, &.{ "one", name })), try f.read(try std.fs.path.join(arena, &.{ "four", name })));
    _ = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("kept"), .{ .threads = 2, .keep = &.{1} });
    var kept: usize = 0;
    for (0..2) |i| {
        var a = try f.open("src", i);
        defer a.close(io);
        var b = try f.open("kept", i);
        defer b.close(io);
        for (a.names.keys()) |name| {
            const t = a.get(name).?;
            if (std.mem.startsWith(u8, name, "model.language_model.layers.1.")) {
                const got = b.get(name).?;
                try testing.expect(got.is(t.dtype, t.shape[0..t.rank]) and std.mem.eql(u8, t.bytes, got.bytes));
                kept += 1;
            } else if (isExpertWeight(name)) {
                try testing.expectEqual(st.DType.u8, b.get(name).?.dtype);
            } else if (std.mem.endsWith(u8, name, ".input_scale")) try testing.expect(b.get(name) == null);
        }
    }
    try testing.expectEqual(@as(usize, 2 + 2 * experts * projs.len + 1), kept); // with its input_scale
    const prov = (try f.json("kept/quant-provenance.json")).object.get("experts").?.object;
    try testing.expectEqual(@as(i64, 1), prov.get("kept_fp8_layers").?.array.items[0].integer);
    try testing.expectEqual(@as(usize, 2), prov.get("converted_layers").?.array.items.len);
}

test "a DST that exists, is SRC or lies inside it, and an unknown kept layer are refused before writing" {
    var f = try Fixture.init();
    defer f.deinit();
    const io = testing.io;
    const before = try sources(&f);
    const gpa = testing.allocator;
    try f.tmp.dir.createDir(io, "taken", .default_dir);
    try testing.expectError(error.DestinationExists, convert.convert(gpa, io, try f.path("src"), try f.path("taken"), .{}));
    try testing.expectError(error.DestinationInSource, convert.convert(gpa, io, try f.path("src"), try f.path("src"), .{}));
    try testing.expectError(error.DestinationInSource, convert.convert(gpa, io, try f.path("src"), try f.path("src/out"), .{}));
    try testing.expectError(error.KeptLayerHasNoExperts, convert.convert(gpa, io, try f.path("src"), try f.path("new"), .{ .keep = &.{3} }));
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(io, "new", .{}));
    try testing.expectEqual(before, try sources(&f));
}

fn f64Bytes(arena: Allocator, v: []const f64, part: f64) ![]const u8 {
    const out = try arena.alloc(f64, v.len);
    for (out, v) |*o, x| o.* = x * part;
    return std.mem.sliceAsBytes(out);
}

test "statistics change a block's scale only where its importance is not uniform, as C3 weighs it" {
    var f = try Fixture.init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const io = testing.io;
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    // counts by class; expert 1 never routed. Block 0 of gate/up and all of expert 0's down sum to a uniform h.
    const by_class = [3][experts]f64{ .{ 3, 0, 1 }, .{ 5, 0, 0 }, .{ 8, 0, 2 } };
    const n = [experts]f64{ 16, 0, 3 };
    var gate_up: [3][experts * hidden]f64 = undefined;
    var down: [3][experts * inner]f64 = undefined;
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    for (0..3) |l| {
        for (0..experts) |e| {
            for (gate_up[l][e * hidden ..][0..hidden], 0..) |*x, c| x.* = if (c < 16) 2 * n[e] else rnd.float(f64) * 50 * n[e];
            for (down[l][e * inner ..][0..inner]) |*x| x.* = if (e == 0) 4 * n[e] else rnd.float(f64) * 50 * n[e];
        }
        // each class holds 1/2, 1/4 and 1/4 of the sums: pooled in class order they add back exactly
        for ([_][]const u8{ "prompt_text", "prompt_audio", "generated" }, by_class, [_]f64{ 0.5, 0.25, 0.25 }) |c, v, part| {
            const pre = try std.fmt.allocPrint(arena, "layers.{d}.", .{l});
            try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}count.{s}", .{ pre, c }), .dtype = .f64, .shape = &.{experts}, .bytes = try f64Bytes(arena, &v, 1) });
            try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}gate_up.sumsq.{s}", .{ pre, c }), .dtype = .f64, .shape = &.{ experts, hidden }, .bytes = try f64Bytes(arena, &gate_up[l], part) });
            try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}down.sumsq.{s}", .{ pre, c }), .dtype = .f64, .shape = &.{ experts, inner }, .bytes = try f64Bytes(arena, &down[l], part) });
        }
    }
    try shard_edit.create(arena, try f.path("stats.safetensors"), &.{.{ "tool_commit", "test" }}, tensors.items);
    _ = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("plain"), .{ .threads = 2 });
    _ = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("weighed"), .{ .threads = 2, .stats = try f.path("stats.safetensors") });
    var moved = [2]usize{ 0, 0 };
    for (0..2) |i| {
        var src = try f.open("src", i);
        defer src.close(io);
        var a = try f.open("plain", i);
        defer a.close(io);
        var b = try f.open("weighed", i);
        defer b.close(io);
        for (a.names.keys()) |name| {
            if (!std.mem.endsWith(u8, name, ".weight_scale")) continue;
            const x, const y = .{ a.get(name).?.bytes, b.get(name).?.bytes };
            const down_proj = std.mem.indexOf(u8, name, "down_proj") != null;
            const expert = name[std.mem.indexOf(u8, name, "experts.").? + "experts.".len] - '0';
            if (down_proj and expert < 2) {
                try testing.expectEqualSlices(u8, x, y); // h = 4 everywhere, or n_e = 0: h = 1
                continue;
            }
            const groups: usize = if (down_proj) inner / 16 else hidden / 16;
            for (x, y, 0..) |p, q, j| {
                if (!down_proj and j % groups == 0) try testing.expectEqual(p, q) else moved[@intFromBool(down_proj)] += @intFromBool(p != q);
            }
        }
        // expert 2 of layer 2 (second shard) against C3's formulas evaluated here
        if (i == 1) for ([_][]const u8{ "gate_proj", "down_proj" }) |p| {
            const gu = p[0] == 'g';
            const k: usize = if (gu) hidden else inner;
            const h = try arena.alloc(f64, k);
            const sums = if (gu) gate_up[2][2 * hidden ..][0..hidden] else down[2][2 * inner ..][0..inner];
            var mean: f64 = 0;
            for (sums) |x| mean += x;
            mean = mean / @as(f64, @floatFromInt(k)) / n[2];
            for (h, sums, 0..) |*x, s, c| {
                const m = (gate_up[2][c] + gate_up[2][hidden + c] + gate_up[2][2 * hidden + c]) / (n[0] + n[1] + n[2]);
                x.* = (s + 64 * (if (gu) m else mean)) / (n[2] + 64);
            }
            try expectEncoded(arena, &src, &b, try std.fmt.allocPrint(arena, "model.language_model.layers.2.mlp.experts.2.{s}.weight", .{p}), h);
        };
    }
    try testing.expect(moved[0] > 0 and moved[1] > 0);
    const prov = (try f.json("weighed/quant-provenance.json")).object.get("experts").?.object;
    try testing.expectEqual(@as(usize, 64), prov.get("stats_sha256").?.string.len);
    const pairs = prov.get("stats_pairs").?.object;
    try testing.expectEqual(@as(i64, 3 * 4), pairs.get("prompt_text").?.integer);
    try testing.expectEqual(@as(i64, 3 * 10), pairs.get("generated").?.integer);
}

test "the encoder's bytes, packed as the expert kernels read them, decode to the encoder's own values" {
    const gpa = testing.allocator;
    const n = 64;
    const k = hidden;
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var w: [n * k]f64 = undefined;
    for (&w) |*x| x.* = rnd.floatNorm(f64) * 0.02;
    var h: [k]f64 = undefined;
    for (&h) |*x| x.* = 0.5 + rnd.float(f64) * 1.5;
    var codes: [n * k / 2]u8 = undefined;
    var scales: [n * k / 16]u8 = undefined;
    const sum = nvfp4.encode(&w, &h, n, k, &codes, &scales);
    const g = sum.global;
    var q: [n * k]f64 = undefined;
    try nvfp4.reference(.{
        .weight = .{ .dtype = .u8, .rank = 2, .shape = .{ n, k / 2, 1, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ n, k / 16, 1, 1, 1 }, .bytes = &scales },
        .global = &.{g},
    }, &q);
    var again: nvfp4.Summary = .{};
    for (w, q, 0..) |x, y, i| {
        const d = x - y;
        again.err_h += h[i % k] * d * d;
        again.err += d * d;
    }
    try testing.expectEqual(@as(u64, @bitCast(sum.err_h)), @as(u64, @bitCast(again.err_h)));
    try testing.expectEqual(@as(u64, @bitCast(sum.err)), @as(u64, @bitCast(again.err)));
    const Nv = cuda.experts.Nvfp4Experts;
    const words = try gpa.alloc(u32, Nv.bytes(1, n, k, 1) / 4);
    defer gpa.free(words);
    Nv.packOne(words, &codes, &scales, n, k);
    for (0..n) |r| for (0..k) |c| {
        // block (r / 32, c / 32): lane (gq, t), word j holds row 8j + gq, input 16hh + 4t + qq at nibble slot(hh, qq)
        const block = words[((r / 32) * (k / 32) + c / 32) * Nv.words ..][0..Nv.words];
        const j, const gq, const hh, const t, const qq = .{ r % 32 / 8, r % 8, c % 32 / 16, c % 16 / 4, c % 4 };
        const nib: u8 = @truncate(block[(gq * 4 + t) * 4 + j] >> @intCast(4 * (2 * hh + qq / 2 + 4 * (qq % 2))));
        const row = r % 32;
        const scale = std.mem.sliceAsBytes(block[128..])[((row % 8 / 2 * 2 + hh) * 4 + row / 8) * 2 + row % 2];
        const v = nvfp4.e2m1(nib & 0xF) * fp8.e4m3(scale) * g;
        try testing.expectEqual(@as(u64, @bitCast(q[r * k + c])), @as(u64, @bitCast(v)));
    };
}

/// A one-projection checkpoint at `sub` whose first FP8 code and only scale_inv are given, plus `extra` tensors.
pub fn single(f: *Fixture, sub: []const u8, code: u8, inv: f32, extra: []const shard_edit.New) !void {
    const arena = f.arena_state.allocator();
    try f.tmp.dir.createDir(testing.io, sub, .default_dir);
    const name = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";
    const w = try arena.alloc(u8, 32 * 32);
    @memset(w, 0x38);
    w[0] = code;
    const s = try arena.dupe(f32, &.{inv});
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    try tensors.appendSlice(arena, &.{
        .{ .name = name, .dtype = .f8_e4m3, .shape = &.{ 32, 32 }, .bytes = w },
        .{ .name = name ++ "_scale_inv", .dtype = .f32, .shape = &.{ 1, 1 }, .bytes = std.mem.sliceAsBytes(s) },
    });
    try tensors.appendSlice(arena, extra);
    try shard_edit.create(arena, try f.path(try std.fs.path.join(arena, &.{ sub, shard_names[0] })), &.{}, tensors.items);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(arena, "{\"weight_map\": {");
    for (tensors.items, 0..) |t, i| try index.print(arena, "{s}\"{s}\": \"{s}\"", .{ if (i == 0) "" else ", ", t.name, shard_names[0] });
    try index.appendSlice(arena, "}}");
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(arena, &.{ sub, "model.safetensors.index.json" }), .data = index.items });
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(arena, &.{ sub, "config.json" }), .data = "{}" });
}

test "a non-finite FP8 value, a non-positive scale_inv and negative statistics are refused" {
    var f = try Fixture.init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const gpa = testing.allocator;
    const io = testing.io;
    try single(&f, "nan", 0x7F, 1, &.{});
    try testing.expectError(error.NonFiniteWeight, convert.convert(gpa, io, try f.path("nan"), try f.path("nan.out"), .{}));
    try single(&f, "zero", 0x38, 0, &.{});
    try testing.expectError(error.BadSourceScale, convert.convert(gpa, io, try f.path("zero"), try f.path("zero.out"), .{}));
    try single(&f, "fine", 0x38, 1, &.{});
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    for ([_][]const u8{ "prompt_text", "prompt_audio", "generated" }) |c| {
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "layers.0.count.{s}", .{c}), .dtype = .f64, .shape = &.{1}, .bytes = try f64Bytes(arena, &.{1}, 1) });
        for ([_][]const u8{ "gate_up", "down" }) |p| {
            const sums = try arena.alloc(f64, 32);
            @memset(sums, -1);
            try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "layers.0.{s}.sumsq.{s}", .{ p, c }), .dtype = .f64, .shape = &.{ 1, 32 }, .bytes = try f64Bytes(arena, sums, 1) });
        }
    }
    try shard_edit.create(arena, try f.path("negative.safetensors"), &.{}, tensors.items);
    try testing.expectError(error.BadStats, convert.convert(gpa, io, try f.path("fine"), try f.path("fine.out"), .{ .stats = try f.path("negative.safetensors") }));
    _ = try convert.convert(gpa, io, try f.path("fine"), try f.path("fine.ok"), .{});
}
