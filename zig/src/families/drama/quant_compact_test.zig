//! tf-drama-quant --compact-experts (generic shapes): FP8 and NVFP4 sources, the map, encoding, refused masks.
const std = @import("std");
const core = @import("core");
const quant = @import("quant_convert.zig");
const compact = @import("compact.zig");

const st = core.safetensors;
const shard_edit = core.shard_edit;
const Allocator = std.mem.Allocator;
const testing = std.testing;

const experts = 12;
const hidden = 160;
const inner = 64;
const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
const shard = "model.safetensors";
/// Layer 0 keeps 8 of its 12 experts, not expert 0; layer 1 is left out, so it keeps all 12.
const mask = "{\"0\": [11, 1, 3, 4, 5, 6, 7, 9]}";
const layer0 = [_]i64{ 1, 3, 4, 5, 6, 7, 9, 11 };

const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    root: []const u8,

    /// SRC: two layers of FP8 experts between BF16 tensors, less layer 1's `hole`, layer 0 expert 3's up_proj `odd`.
    fn init(hole: ?usize, odd: Odd) !Fixture {
        var f: Fixture = .{ .arena_state = .init(testing.allocator), .tmp = testing.tmpDir(.{}), .root = undefined };
        errdefer f.deinit();
        const a = f.arena_state.allocator();
        f.root = try f.tmp.dir.realPathFileAlloc(testing.io, ".", a);
        var prng = std.Random.DefaultPrng.init(11);
        const rnd = prng.random();
        var tensors: std.ArrayList(shard_edit.New) = .empty;
        try tensors.append(a, .{ .name = "model.language_model.embed_tokens.weight", .dtype = .bf16, .shape = &.{ 4, hidden }, .bytes = try fill(a, rnd, 8 * hidden, false) });
        for (0..2) |l| {
            const pre = try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.", .{l});
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}gate.weight", .{pre}), .dtype = .bf16, .shape = &.{ experts, hidden }, .bytes = try fill(a, rnd, 2 * experts * hidden, false) });
            for (0..experts) |e| if (l != 1 or e != hole) for (projs) |p| {
                const n: usize, const k: usize = if (p[0] == 'd') .{ hidden, inner } else .{ inner, hidden };
                const name = try std.fmt.allocPrint(a, "{s}experts.{d}.{s}.weight", .{ pre, e, p });
                const inv = try a.alloc(f32, ((n + 127) / 128) * ((k + 127) / 128));
                for (inv) |*x| x.* = @floatCast(1e-4 + rnd.float(f64) * 1e-2);
                const wrong = l == 0 and e == 3 and p[0] == 'u' and odd != .none;
                if (wrong and odd == .fp4_names) { // an FP8 weight beside NVFP4 scale names
                    try tensors.append(a, .{ .name = name, .dtype = .f8_e4m3, .shape = try a.dupe(usize, &.{ n, k }), .bytes = try fill(a, rnd, n * k, true) });
                    try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}_scale", .{name}), .dtype = .f8_e4m3, .shape = try a.dupe(usize, &.{ n, k / 16 }), .bytes = try fill(a, rnd, n * k / 16, true) });
                    try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}_scale_2", .{name}), .dtype = .f32, .shape = &.{}, .bytes = std.mem.sliceAsBytes(inv[0..1]) });
                    continue;
                }
                if (wrong) { // an NVFP4 weight beside an FP8 scale_inv
                    try tensors.append(a, .{ .name = name, .dtype = .u8, .shape = try a.dupe(usize, &.{ n, k / 2 }), .bytes = try fill(a, rnd, n * k / 2, false) });
                } else try tensors.append(a, .{ .name = name, .dtype = .f8_e4m3, .shape = try a.dupe(usize, &.{ n, k }), .bytes = try fill(a, rnd, n * k, true) });
                try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}_scale_inv", .{name}), .dtype = .f32, .shape = try a.dupe(usize, &.{ (n + 127) / 128, (k + 127) / 128 }), .bytes = std.mem.sliceAsBytes(inv) });
                if (e == 0 and p[0] == 'g') try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}experts.0.gate_proj.input_scale", .{pre}), .dtype = .f32, .shape = &.{}, .bytes = std.mem.sliceAsBytes(inv[0..1]) });
            };
        }
        try tensors.append(a, .{ .name = "lm_head.weight", .dtype = .bf16, .shape = &.{ 4, hidden }, .bytes = try fill(a, rnd, 8 * hidden, false) });
        try f.tmp.dir.createDir(testing.io, "src", .default_dir);
        try writeCheckpoint(&f, "src", tensors.items);
        try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "mask.json", .data = mask });
        return f;
    }

    const Odd = enum { none, fp4_names, fp8_names };

    fn deinit(f: *Fixture) void {
        f.tmp.cleanup();
        f.arena_state.deinit();
    }

    fn path(f: *Fixture, sub: []const u8) ![]const u8 {
        return std.fs.path.join(f.arena_state.allocator(), &.{ f.root, sub });
    }

    fn read(f: *Fixture, sub: []const u8) ![]u8 {
        return f.tmp.dir.readFileAlloc(testing.io, sub, f.arena_state.allocator(), .limited(1 << 24));
    }

    fn metadata(f: *Fixture, dir: []const u8) !std.json.ObjectMap {
        const a = f.arena_state.allocator();
        const index = try std.json.parseFromSliceLeaky(std.json.Value, a, try f.read(try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" })), .{});
        return index.object.get("metadata").?.object;
    }

    fn open(f: *Fixture, dir: []const u8) !st.File {
        return st.File.open(testing.allocator, testing.io, try std.fs.path.join(f.arena_state.allocator(), &.{ f.root, dir, shard }));
    }

    fn convert(f: *Fixture, src: []const u8, dst: []const u8, o: quant.Options) !quant.Summary {
        return quant.convert(testing.allocator, testing.io, try f.path(src), try f.path(dst), o);
    }
};

fn fill(a: Allocator, rnd: std.Random, n: usize, fp8_codes: bool) ![]u8 {
    const b = try a.alloc(u8, n);
    for (b) |*x| x.* = if (fp8_codes) rnd.uintLessThan(u8, 0x7F) | (rnd.int(u8) & 0x80) else rnd.int(u8);
    return b;
}

/// One shard and its index at `sub`.
fn writeCheckpoint(f: *Fixture, sub: []const u8, tensors: []const shard_edit.New) !void {
    const a = f.arena_state.allocator();
    try shard_edit.create(a, try f.path(try std.fs.path.join(a, &.{ sub, shard })), &.{.{ "format", "pt" }}, tensors);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"metadata\": {\"total_size\": 1}, \"weight_map\": {");
    for (tensors, 0..) |t, i| try index.print(a, "{s}\"{s}\": \"{s}\"", .{ if (i == 0) "" else ", ", t.name, shard });
    try index.appendSlice(a, "}}\n");
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ sub, "model.safetensors.index.json" }), .data = index.items });
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ sub, "config.json" }), .data = "{}" });
}

/// Whether the mask holds expert `e` of `layer`.
fn held(layer: usize, e: usize) bool {
    return layer != 0 or std.mem.indexOfScalar(i64, &layer0, @intCast(e)) != null;
}

/// The expert a routed-expert tensor name belongs to, or null.
fn expertOf(name: []const u8) ?[2]usize {
    const pre = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, name, pre)) return null;
    var it = std.mem.splitScalar(u8, name[pre.len..], '.');
    const layer = std.fmt.parseInt(usize, it.next().?, 10) catch return null;
    if (!std.mem.eql(u8, it.next() orelse "", "mlp") or !std.mem.eql(u8, it.next() orelse "", "experts")) return null;
    return .{ layer, std.fmt.parseInt(usize, it.next().?, 10) catch return null };
}

/// DST holds exactly SRC's tensors less the unheld experts' (every companion included), in SRC's order and bytes.
fn expectCompacted(f: *Fixture, src: []const u8, dst: []const u8) !void {
    const a = f.arena_state.allocator();
    var from = try f.open(src);
    defer from.close(testing.io);
    var to = try f.open(dst);
    defer to.close(testing.io);
    var want: std.ArrayList([]const u8) = .empty;
    for (try ordered(a, &from)) |name| if (expertOf(name)) |x| {
        if (held(x[0], x[1])) try want.append(a, name);
    } else try want.append(a, name);
    const got = try ordered(a, &to);
    try testing.expectEqual(want.items.len, got.len);
    for (want.items, got) |x, y| {
        try testing.expectEqualStrings(x, y);
        const s, const d = .{ from.get(x).?, to.get(y).? };
        try testing.expect(s.is(d.dtype, d.shape[0..d.rank]) and std.mem.eql(u8, s.bytes, d.bytes));
    }
}

fn ordered(a: Allocator, file: *const st.File) ![]const []const u8 {
    const names = try a.dupe([]const u8, file.names.keys());
    std.mem.sort([]const u8, names, file, struct {
        fn lt(x: *const st.File, l: []const u8, r: []const u8) bool {
            return x.names.get(l).?.begin < x.names.get(r).?.begin;
        }
    }.lt);
    return names;
}

/// The index map: layer 0's eight held ids ascending, layer 1's twelve.
fn expectMap(f: *Fixture, dst: []const u8) !void {
    const map = (try f.metadata(dst)).get(compact.experts_key).?.object;
    try testing.expectEqual(@as(usize, 2), map.count());
    const ids0 = map.get("0").?.array.items;
    try testing.expectEqual(layer0.len, ids0.len);
    for (ids0, layer0) |v, e| try testing.expectEqual(e, v.integer);
    const ids1 = map.get("1").?.array.items;
    try testing.expectEqual(@as(usize, experts), ids1.len);
    for (ids1, 0..) |v, e| try testing.expectEqual(@as(i64, @intCast(e)), v.integer);
}

test "FP8 experts compact byte for byte: only the held ones' tensors, in SRC's order, and the index maps them" {
    var f = try Fixture.init(null, .none);
    defer f.deinit();
    const s = try f.convert("src", "dst", .{ .experts = .keep, .compact_experts = try f.path("mask.json") });
    try expectCompacted(&f, "src", "dst");
    try expectMap(&f, "dst");
    try testing.expectEqual(@as(i64, @intCast(s.bytes)), (try f.metadata("dst")).get("total_size").?.integer);
    const prov = try std.json.parseFromSliceLeaky(std.json.Value, f.arena_state.allocator(), try f.read("dst/quant-provenance.json"), .{});
    const record = prov.object.get("compact_experts").?.object;
    try testing.expectEqual(@as(i64, 8 + experts), record.get("held").?.integer);
    try testing.expectEqual(@as(i64, 2 * experts), record.get("of").?.integer);
    try testing.expectEqualStrings(try f.path("mask.json"), record.get("file").?.string);
}

test "NVFP4 experts compact byte for byte, and compacting while encoding writes the full encoding's held tensors" {
    var f = try Fixture.init(null, .none);
    defer f.deinit();
    _ = try f.convert("src", "nvfp4", .{ .threads = 2 });
    _ = try f.convert("nvfp4", "dst", .{ .experts = .keep, .compact_experts = try f.path("mask.json") });
    try expectCompacted(&f, "nvfp4", "dst");
    try expectMap(&f, "dst");
    // each projection encodes alone, so encoding only the held experts writes what encoding all of them did
    const s = try f.convert("src", "encoded", .{ .threads = 2, .compact_experts = try f.path("mask.json") });
    try testing.expectEqual(@as(usize, (layer0.len + experts) * projs.len), s.converted);
    try expectCompacted(&f, "nvfp4", "encoded");
    try expectMap(&f, "encoded");
}

test "a mask that keeps every expert writes SRC's shards unchanged, and a later run carries the map on" {
    var f = try Fixture.init(null, .none);
    defer f.deinit();
    const a = f.arena_state.allocator();
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "all.json", .data = "{}" });
    _ = try f.convert("src", "plain", .{ .experts = .keep });
    _ = try f.convert("src", "all", .{ .experts = .keep, .compact_experts = try f.path("all.json") });
    try testing.expectEqualSlices(u8, try f.read("plain/" ++ shard), try f.read("all/" ++ shard));
    try testing.expect((try f.metadata("plain")).get(compact.experts_key) == null);
    for ((try f.metadata("all")).get(compact.experts_key).?.object.values()) |ids| try testing.expectEqual(@as(usize, experts), ids.array.items.len);
    _ = try f.convert("src", "dst", .{ .experts = .keep, .compact_experts = try f.path("mask.json") });
    _ = try f.convert("dst", "again", .{ .experts = .keep });
    const map = try std.json.Stringify.valueAlloc(a, (try f.metadata("dst")).get(compact.experts_key).?, .{});
    try testing.expectEqualStrings(map, try std.json.Stringify.valueAlloc(a, (try f.metadata("again")).get(compact.experts_key).?, .{}));
}

test "a malformed mask, a layer or an expert SRC lacks, and an already compact SRC are refused before writing" {
    var f = try Fixture.init(null, .none);
    defer f.deinit();
    const a = f.arena_state.allocator();
    _ = try f.convert("src", "dst", .{ .experts = .keep, .compact_experts = try f.path("mask.json") });
    const cases = [_]struct { []const u8, anyerror }{
        .{ "{\"0\": [1, 3, 4, 5, 6, 7, 9, 9]}", error.BadCompactExperts }, // a duplicate
        .{ "{\"0\": [1, 3, 4, 5, 6, 7, 9, 256]}", error.BadCompactExperts }, // past 255
        .{ "{\"0\": [1, 3, 4, 5, 6, 7, 9]}", error.BadCompactExperts }, // under top_k
        .{ "{\"0\": [1, 3, 4, 5, 6, 7, 9, -1]}", error.BadCompactExperts },
        .{ "{\"40\": [1, 3, 4, 5, 6, 7, 9, 11]}", error.BadCompactExperts }, // no such layer
        .{ "{\"0\": [1, 3, 4, 5, 6, 7, 9, 12]}", error.CompactExpertNotInSource }, // SRC's layer 0 has 12
        .{ "{\"2\": [1, 3, 4, 5, 6, 7, 9, 11]}", error.CompactLayerHasNoExperts },
    };
    for (cases, 0..) |c, i| {
        const file = try std.fmt.allocPrint(a, "bad{d}.json", .{i});
        try f.tmp.dir.writeFile(testing.io, .{ .sub_path = file, .data = c[0] });
        try testing.expectError(c[1], f.convert("src", "out", .{ .experts = .keep, .compact_experts = try f.path(file) }));
        try testing.expectError(error.FileNotFound, f.tmp.dir.access(testing.io, "out", .{}));
    }
    try testing.expectError(error.AlreadyCompact, f.convert("dst", "out", .{ .experts = .keep, .compact_experts = try f.path("mask.json") }));
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(testing.io, "out", .{}));
}

test "an expert missing below a layer's highest id is not held: a mask keeping it, or its whole layer, is refused" {
    var f = try Fixture.init(5, .none);
    defer f.deinit();
    try testing.expectError(error.CompactExpertNotInSource, f.convert("src", "out", .{ .experts = .keep, .compact_experts = try f.path("mask.json") }));
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "hole.json", .data = "{\"1\": [0, 1, 2, 3, 4, 5, 6, 7]}" });
    try testing.expectError(error.CompactExpertNotInSource, f.convert("src", "out", .{ .experts = .keep, .compact_experts = try f.path("hole.json") }));
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "around.json", .data = "{\"0\": [0, 1, 2, 3, 4, 5, 6, 7], \"1\": [0, 1, 2, 3, 4, 6, 7, 8]}" });
    _ = try f.convert("src", "out", .{ .experts = .keep, .compact_experts = try f.path("around.json") });
}

/// `dir`'s index with `name` listed in its one shard (`present`) or left out.
fn setIndex(f: *Fixture, dir: []const u8, name: []const u8, present: bool) !void {
    const a = f.arena_state.allocator();
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
    var index = try std.json.parseFromSliceLeaky(std.json.Value, a, try f.read(path), .{});
    const map = &index.object.getPtr("weight_map").?.object;
    if (present) try map.put(a, name, .{ .string = shard }) else try testing.expect(map.orderedRemove(name));
    try f.tmp.dir.deleteFile(testing.io, path);
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = path, .data = try std.json.Stringify.valueAlloc(a, index, .{}) });
}

/// Compacting `src` by mask.json fails as SRC not storing a held expert, and leaves no output.
fn expectRefused(f: *Fixture, src: []const u8) !void {
    try testing.expectError(error.CompactExpertNotInSource, f.convert(src, "out", .{ .experts = .keep, .compact_experts = try f.path("mask.json") }));
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(testing.io, "out", .{}));
}

test "a held expert without its FP8 scale_inv, or its NVFP4 weight_scale_2, is refused before writing" {
    var f = try Fixture.init(null, .none);
    defer f.deinit();
    _ = try f.convert("src", "nvfp4", .{ .threads = 2 });
    try setIndex(&f, "src", "model.language_model.layers.0.mlp.experts.3.up_proj.weight_scale_inv", false);
    try setIndex(&f, "nvfp4", "model.language_model.layers.0.mlp.experts.9.down_proj.weight_scale_2", false);
    for ([_][]const u8{ "src", "nvfp4" }) |dir| try expectRefused(&f, dir);
}

test "a held expert its index lists in full but its shard lacks is refused before writing" {
    var f = try Fixture.init(5, .none);
    defer f.deinit();
    const a = f.arena_state.allocator();
    for (projs) |p| for ([_][]const u8{ "", "_scale_inv" }) |suffix| {
        try setIndex(&f, "src", try std.fmt.allocPrint(a, "model.language_model.layers.1.mlp.experts.5.{s}.weight{s}", .{ p, suffix }), true);
    };
    try expectRefused(&f, "src");
}

test "a held FP8 weight with NVFP4 scale names instead of its scale_inv is refused before writing" {
    var f = try Fixture.init(null, .fp4_names);
    defer f.deinit();
    try expectRefused(&f, "src");
}

test "a held NVFP4 weight with an FP8 scale_inv instead of its scales is refused before writing" {
    var f = try Fixture.init(null, .fp8_names);
    defer f.deinit();
    try expectRefused(&f, "src");
}
