//! tf-drama-quant --dense-fp8 on a synthetic BF16 + FP8 checkpoint: selectors, both expert modes, provenance, refusals.
const std = @import("std");
const core = @import("core");
const convert = @import("quant_convert.zig");
const dense = @import("quant_dense.zig");
const t = @import("quant_convert_test.zig");

const st = core.safetensors;
const shard_edit = core.shard_edit;
const fp8 = core.quant.fp8;
const Allocator = std.mem.Allocator;
const testing = std.testing;

const hidden = 128;
const experts = 2;
const Spec = struct { []const u8, []const usize };

/// SRC at `sub`: backbone layer 0 GDN and layer 1 attention (each with a MoE), tower layers t:0 and t:4; FP8 experts.
fn source(f: *t.Fixture, sub: []const u8) !void {
    const arena = f.arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(21);
    const rnd = prng.random();
    var shards: [2]std.ArrayList(shard_edit.New) = .{ .empty, .empty };
    const Add = struct {
        fn bf16(a: Allocator, r: std.Random, s: *std.ArrayList(shard_edit.New), name: []const u8, shape: []const usize) !void {
            var n: usize = 1;
            for (shape) |d| n *= d;
            const b = try a.alloc(u16, n);
            for (b) |*x| x.* = core.quant.convert.bf16(@floatCast(r.floatNorm(f64) * 0.02));
            try s.append(a, .{ .name = name, .dtype = .bf16, .shape = try a.dupe(usize, shape), .bytes = std.mem.sliceAsBytes(b) });
        }
    };
    try Add.bf16(arena, rnd, &shards[0], "model.language_model.embed_tokens.weight", &.{ 8, hidden });
    for (0..2) |l| {
        const s = &shards[l];
        const pre = try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.", .{l});
        const own = if (l == 0)
            [_]Spec{ .{ "linear_attn.in_proj_qkv", &.{ 256, hidden } }, .{ "linear_attn.in_proj_z", &.{ 128, hidden } }, .{ "linear_attn.in_proj_b", &.{ 2, hidden } }, .{ "linear_attn.in_proj_a", &.{ 2, hidden } }, .{ "linear_attn.out_proj", &.{ hidden, 128 } } }
        else
            [_]Spec{ .{ "self_attn.q_proj", &.{ 256, hidden } }, .{ "self_attn.k_proj", &.{ 128, hidden } }, .{ "self_attn.v_proj", &.{ 128, hidden } }, .{ "self_attn.q_norm", &.{64} }, .{ "self_attn.o_proj", &.{ hidden, 128 } } };
        for (own) |p| try Add.bf16(arena, rnd, s, try std.fmt.allocPrint(arena, "{s}{s}.weight", .{ pre, p[0] }), p[1]);
        try Add.bf16(arena, rnd, s, try std.fmt.allocPrint(arena, "{s}mlp.gate.weight", .{pre}), &.{ experts, hidden });
        for (0..experts) |e| for (t.projs) |p| {
            const n: usize, const k: usize = if (p[0] == 'd') .{ hidden, 64 } else .{ 64, hidden };
            const name = try std.fmt.allocPrint(arena, "{s}mlp.experts.{d}.{s}.weight", .{ pre, e, p });
            const codes = try arena.alloc(u8, n * k);
            for (codes) |*x| x.* = rnd.uintLessThan(u8, 0x7F) | (rnd.int(u8) & 0x80);
            const inv = try arena.dupe(f32, &.{@floatCast(1e-3 + rnd.float(f64) * 1e-2)});
            try s.append(arena, .{ .name = name, .dtype = .f8_e4m3, .shape = try arena.dupe(usize, &.{ n, k }), .bytes = codes });
            try s.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name}), .dtype = .f32, .shape = &.{ 1, 1 }, .bytes = std.mem.sliceAsBytes(inv) });
        };
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |p| try Add.bf16(arena, rnd, s, try std.fmt.allocPrint(arena, "{s}mlp.shared_expert.{s}.weight", .{ pre, p }), &.{ 128, hidden });
        try Add.bf16(arena, rnd, s, try std.fmt.allocPrint(arena, "{s}mlp.shared_expert_gate.weight", .{pre}), &.{ 1, hidden });
    }
    for ([_][]const u8{ "audio_decoder_mtp_shared.layers.0.", "audio_decoder_mtp.layers.0." }) |pre| {
        try Add.bf16(arena, rnd, &shards[1], try std.fmt.allocPrint(arena, "{s}eh_proj.weight", .{pre}), &.{ hidden, 2 * hidden });
        for ([_][]const u8{ "attention.wqkv", "attention.wo", "feed_forward.w1", "feed_forward.w3", "feed_forward.w2", "attention_norm" }) |p| {
            const shape: []const usize = if (std.mem.eql(u8, p, "attention_norm")) &.{hidden} else &.{ 128, hidden };
            try Add.bf16(arena, rnd, &shards[1], try std.fmt.allocPrint(arena, "{s}transformer_layer.{s}.weight", .{ pre, p }), shape);
        }
    }
    try Add.bf16(arena, rnd, &shards[1], "lm_head.weight", &.{ 8, hidden });
    try f.tmp.dir.createDir(testing.io, sub, .default_dir);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(arena, "{\"metadata\": {\"total_size\": 1}, \"weight_map\": {");
    for (shards, t.shard_names, 0..) |s, file, i| {
        try shard_edit.create(arena, try f.path(try std.fs.path.join(arena, &.{ sub, file })), &.{.{ "format", "pt" }}, s.items);
        for (s.items, 0..) |x, j| try index.print(arena, "{s}\"{s}\": \"{s}\"", .{ if (i + j == 0) "" else ", ", x.name, file });
    }
    try index.appendSlice(arena, "}}\n");
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(arena, &.{ sub, "model.safetensors.index.json" }), .data = index.items });
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(arena, &.{ sub, "config.json" }), .data = "{\"model_type\": \"fish_omni\"}\n" });
}

fn init() !t.Fixture {
    var f = try t.Fixture.init();
    errdefer f.deinit();
    try source(&f, "d");
    return f;
}

/// `dst` against `src`: the `converted` tensors as encodeRows makes them, each before its scale_inv; the rest kept.
fn expectConverted(f: *t.Fixture, src: []const u8, dst: []const u8, converted: []const []const u8) !void {
    const arena = f.arena_state.allocator();
    var listed: usize = 0;
    var total: u64 = 0;
    const map = (try f.json(try std.fs.path.join(arena, &.{ dst, "model.safetensors.index.json" }))).object.get("weight_map").?.object;
    for (0..2) |i| {
        var a = try f.open(src, i);
        defer a.close(testing.io);
        var b = try f.open(dst, i);
        defer b.close(testing.io);
        var want: std.ArrayList([]const u8) = .empty;
        for (try t.ordered(arena, &a)) |name| {
            try want.append(arena, name);
            const x = a.get(name).?;
            const y = b.get(name).?;
            for (converted) |c| {
                if (!std.mem.eql(u8, c, name)) continue;
                const n, const k = .{ x.shape[0], x.shape[1] };
                const codes = try arena.alloc(u8, n * k);
                const inv = try arena.alloc(f32, n * k / fp8.row_group);
                try fp8.encodeRows(x.bytes, n, k, codes, inv);
                const scale = try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name});
                try want.append(arena, scale);
                try testing.expect(y.is(.f8_e4m3, &.{ n, k }) and std.mem.eql(u8, codes, y.bytes));
                try testing.expect(b.get(scale).?.is(.f32, &.{ n, k / fp8.row_group }) and std.mem.eql(u8, std.mem.sliceAsBytes(inv), b.get(scale).?.bytes));
                break;
            } else try testing.expect(y.is(x.dtype, x.shape[0..x.rank]) and std.mem.eql(u8, x.bytes, y.bytes));
        }
        const got = try t.ordered(arena, &b);
        try testing.expectEqual(want.items.len, got.len);
        for (want.items, got) |x, y| {
            try testing.expectEqualStrings(x, y);
            try testing.expectEqualStrings(t.shard_names[i], map.get(y).?.string);
            total += b.get(y).?.bytes.len;
        }
        listed += got.len;
    }
    try testing.expectEqual(listed, map.count());
    const prov = (try f.json(try std.fs.path.join(arena, &.{ dst, "quant-provenance.json" }))).object.get("dense_fp8").?.object;
    const tensors = prov.get("tensors").?.array.items;
    try testing.expectEqual(converted.len, tensors.len);
    for (converted, tensors) |c, x| {
        try testing.expectEqualStrings(c, x.object.get("name").?.string);
        const rel = x.object.get("rel").?.float;
        try testing.expect(rel > 0 and rel < 0.1 and rel <= x.object.get("worst_row_rel").?.float);
    }
}

/// SRC's dense projections the selection converts, in SRC's order.
fn resolve(f: *t.Fixture, src: []const u8, s: *const dense.Selection) ![]const []const u8 {
    const arena = f.arena_state.allocator();
    var out: std.ArrayList([]const u8) = .empty;
    for (0..2) |i| {
        var a = try f.open(src, i);
        defer a.close(testing.io);
        for (try t.ordered(arena, &a)) |name| if (dense.parse(name)) |at| if (s.converts(at)) try out.append(arena, try arena.dupe(u8, name));
    }
    return out.items;
}

test "--experts keep --dense-fp8 all converts every C5 group and keeps every expert byte, from a source without provenance" {
    var f = try init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const s = try convert.convert(testing.allocator, testing.io, try f.path("d"), try f.path("all"), .{ .experts = .keep, .dense_fp8 = "all", .threads = 3 });
    const all = try dense.Selection.init(arena, "all", null, .{ .backbone = 2, .tower = 5 });
    const names = try resolve(&f, "d", &all);
    // GDN qkv, z, out; q, k, v, o; 2 x shared gate, up, down; 2 x tower wqkv, wo, w1, w3, w2
    try testing.expectEqual(@as(usize, 3 + 4 + 6 + 10), names.len);
    try testing.expectEqual(convert.Summary{ .shards = 2, .converted = 0, .dense = names.len, .copied = 1, .bytes = s.bytes }, s);
    try expectConverted(&f, "d", "all", names);
    const prov = (try f.json("all/quant-provenance.json")).object;
    try testing.expectEqualStrings("tf-drama-quant", prov.get("producer").?.string);
    try testing.expectEqualStrings("keep", prov.get("experts").?.object.get("mode").?.string);
    try testing.expectEqual(std.json.Value.null, prov.get("source_provenance").?);
    try testing.expectEqual(@as(usize, 0), prov.get("stats_consumed_by").?.array.items.len);
    const d = prov.get("dense_fp8").?.object;
    try testing.expectEqualStrings("all", d.get("selectors").?.object.get("dense_fp8").?.string);
    try testing.expectEqual(std.json.Value.null, d.get("selectors").?.object.get("keep_bf16").?);
    try testing.expectEqual(@as(usize, 10), d.get("classes").?.array.items.len);
    _ = try convert.convert(testing.allocator, testing.io, try f.path("d"), try f.path("one"), .{ .experts = .keep, .dense_fp8 = "all", .threads = 1 });
    for (t.shard_names) |name| try testing.expectEqualSlices(u8, try f.read(try std.fs.path.join(arena, &.{ "one", name })), try f.read(try std.fs.path.join(arena, &.{ "all", name })));
}

test "--keep-bf16 subtracts whole groups by class and by namespaced layer from an all-selection" {
    var f = try init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const cases = [_][2][]const u8{ .{ "all", "shared_in,b:1,t:4" }, .{ "all", "t:0,4,gdn_in" }, .{ "gdn_in,attn_in,tower_gate_up", "b:0" } };
    for (cases, 0..) |c, i| {
        const dst = try std.fmt.allocPrint(arena, "keep{d}", .{i});
        _ = try convert.convert(testing.allocator, testing.io, try f.path("d"), try f.path(dst), .{ .experts = .keep, .dense_fp8 = c[0], .keep_bf16 = c[1], .threads = 2 });
        const s = try dense.Selection.init(arena, c[0], c[1], .{ .backbone = 2, .tower = 5 });
        const names = try resolve(&f, "d", &s);
        for (names) |name| {
            const at = dense.parse(name).?;
            try testing.expect(switch (i) {
                0 => at.class != .shared_in and !(!at.layer.tower and at.layer.index == 1) and !(at.layer.tower and at.layer.index == 4),
                1 => !at.layer.tower and at.class != .gdn_in,
                else => at.class == .attn_in or at.class == .tower_gate_up,
            });
        }
        try testing.expectEqual(@as(usize, switch (i) {
            0 => 3 + 1 + 5,
            1 => 1 + 4 + 6,
            else => 3 + 4,
        }), names.len);
        try expectConverted(&f, "d", dst, names);
        const selectors = (try f.json(try std.fs.path.join(arena, &.{ dst, "quant-provenance.json" }))).object.get("dense_fp8").?.object.get("selectors").?.object;
        try testing.expectEqualStrings(c[1], selectors.get("keep_bf16").?.string);
    }
}

/// Statistics for both layers: count 1 an expert and class, gate/up sums 1 + c % 7, down sums 2.
fn stats(f: *t.Fixture) ![]const u8 {
    const arena = f.arena_state.allocator();
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    for (0..2) |l| for ([_][]const u8{ "prompt_text", "prompt_audio", "generated" }) |c| {
        const count = try arena.alloc(f64, experts);
        @memset(count, 1);
        const gate_up = try arena.alloc(f64, experts * hidden);
        for (gate_up, 0..) |*x, i| x.* = @floatFromInt(1 + i % hidden % 7);
        const down = try arena.alloc(f64, experts * 64);
        @memset(down, 2);
        const pre = try std.fmt.allocPrint(arena, "layers.{d}.", .{l});
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}count.{s}", .{ pre, c }), .dtype = .f64, .shape = &.{experts}, .bytes = std.mem.sliceAsBytes(count) });
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}gate_up.sumsq.{s}", .{ pre, c }), .dtype = .f64, .shape = &.{ experts, hidden }, .bytes = std.mem.sliceAsBytes(gate_up) });
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}down.sumsq.{s}", .{ pre, c }), .dtype = .f64, .shape = &.{ experts, 64 }, .bytes = std.mem.sliceAsBytes(down) });
    };
    const path = try f.path("stats.safetensors");
    try shard_edit.create(arena, path, &.{}, tensors.items);
    return path;
}

test "dense statistics weigh shared_in's err_h only and leave a kept expert's bytes alone" {
    var f = try init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const path = try stats(&f);
    _ = try convert.convert(testing.allocator, testing.io, try f.path("d"), try f.path("w"), .{ .experts = .keep, .dense_fp8 = "all", .stats = path, .threads = 2 });
    const all = try dense.Selection.init(arena, "all", null, .{ .backbone = 2, .tower = 5 });
    try expectConverted(&f, "d", "w", try resolve(&f, "d", &all));
    const prov = (try f.json("w/quant-provenance.json")).object;
    const consumers = prov.get("stats_consumed_by").?.array.items;
    try testing.expectEqual(@as(usize, 1), consumers.len);
    try testing.expectEqualStrings("dense_fp8", consumers[0].string);
    const d = prov.get("dense_fp8").?.object;
    try testing.expectEqual(@as(usize, 64), d.get("stats_sha256").?.string.len);
    var a = try f.open("d", 0);
    defer a.close(testing.io);
    var b = try f.open("w", 0);
    defer b.close(testing.io);
    for (d.get("tensors").?.array.items) |x| {
        const name = x.object.get("name").?.string;
        const err_h = x.object.get("err_h").?;
        if (dense.parse(name).?.class != .shared_in) {
            try testing.expectEqual(std.json.Value.null, err_h);
            continue;
        }
        const w = a.get(name) orelse continue;
        // h = the routed experts' gate/up sums over classes: 3 x 2 x (1 + c % 7)
        const q = try arena.alloc(f64, 128 * hidden);
        try fp8.referenceRows(.{ .weight = b.get(name).?, .scale_inv = b.get(try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name})).? }, q);
        var num: f64 = 0;
        var den: f64 = 0;
        for (q, 0..) |y, i| {
            const v: f64 = core.quant.convert.load(.bf16, w.bytes, i);
            const h: f64 = @floatFromInt(6 * (1 + i % hidden % 7));
            num += h * (v - y) * (v - y);
            den += h * v * v;
        }
        try testing.expectEqual(@sqrt(num / den), err_h.float);
    }
}

/// SHA-256 of `sub`'s shards, index and config.
fn digests(f: *t.Fixture, sub: []const u8) ![4][32]u8 {
    var out: [4][32]u8 = undefined;
    for (&out, t.shard_names ++ [_][]const u8{ "model.safetensors.index.json", "config.json" }) |*h, name| {
        std.crypto.hash.sha2.Sha256.hash(try f.read(try std.fs.path.join(f.arena_state.allocator(), &.{ sub, name })), h, .{});
    }
    return out;
}

test "a Phase 1 source, legacy or current, keeps its NVFP4 experts and its provenance verbatim under the dense run's" {
    var f = try init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const io = testing.io;
    _ = try convert.convert(testing.allocator, io, try f.path("d"), try f.path("p1"), .{ .dense_fp8 = "attn_in", .threads = 2 });
    const p1 = (try f.json("p1/quant-provenance.json")).object;
    try testing.expectEqualStrings("nvfp4", p1.get("experts").?.object.get("mode").?.string);
    try testing.expectEqual(@as(usize, 2 * experts * t.projs.len), p1.get("experts").?.object.get("tensors").?.array.items.len);
    try testing.expectEqual(@as(usize, 3), p1.get("dense_fp8").?.object.get("tensors").?.array.items.len);
    try testing.expectError(error.ExpectedFp8Expert, convert.convert(testing.allocator, io, try f.path("p1"), try f.path("again"), .{}));
    try testing.expectError(error.ExpectedBf16Dense, convert.convert(testing.allocator, io, try f.path("p1"), try f.path("again"), .{ .experts = .keep, .dense_fp8 = "attn_in" }));
    _ = try convert.convert(testing.allocator, io, try f.path("d"), try f.path("legacy"), .{ .threads = 2 });
    try f.tmp.dir.rename("legacy/quant-provenance.json", f.tmp.dir, "legacy/nvfp4-provenance.json", io);
    for ([_][2][]const u8{ .{ "p1", "quant-provenance.json" }, .{ "legacy", "nvfp4-provenance.json" } }) |c| {
        const before = try digests(&f, c[0]);
        const record = try f.read(try std.fs.path.join(arena, &.{ c[0], c[1] }));
        const dst = try std.fmt.allocPrint(arena, "{s}-fp8", .{c[0]});
        _ = try convert.convert(testing.allocator, io, try f.path(c[0]), try f.path(dst), .{ .experts = .keep, .dense_fp8 = "gdn_in,tower_o", .threads = 2 });
        const s = try dense.Selection.init(arena, "gdn_in,tower_o", null, .{ .backbone = 2, .tower = 5 });
        try expectConverted(&f, c[0], dst, try resolve(&f, c[0], &s));
        const prov = (try f.json(try std.fs.path.join(arena, &.{ dst, "quant-provenance.json" }))).object;
        try testing.expectEqualStrings("keep", prov.get("experts").?.object.get("mode").?.string);
        const from = prov.get("source_provenance").?.object;
        try testing.expectEqualStrings(c[1], from.get("file").?.string);
        try testing.expectEqualStrings(record, from.get("text").?.string);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(record, &digest, .{});
        try testing.expectEqualStrings(&std.fmt.bytesToHex(digest, .lower), from.get("sha256").?.string);
        const kept = try std.json.parseFromSliceLeaky(std.json.Value, arena, record, .{});
        try testing.expectEqualStrings("nvfp4", kept.object.get("experts").?.object.get("mode").?.string);
        // a legacy record stays beside the new one; a current one is replaced, its text kept above
        const copied = f.read(try std.fs.path.join(arena, &.{ dst, "nvfp4-provenance.json" }));
        if (c[1][0] == 'n') try testing.expectEqualStrings(record, try copied) else try testing.expectError(error.FileNotFound, copied);
        try testing.expectEqual(before, try digests(&f, c[0]));
    }
}

test "the selectors and the expert modes refuse what they cannot mean, before DST exists" {
    var f = try init();
    defer f.deinit();
    const gpa = testing.allocator;
    const io = testing.io;
    const cases = [_]struct { convert.Options, anyerror }{
        .{ .{ .experts = .keep, .keep = &.{0} }, error.ExpertsKept },
        .{ .{ .experts = .keep, .encoded = "enc" }, error.ExpertsKept },
        .{ .{ .dense_fp8 = "gdn" }, error.UnknownDenseClass },
        .{ .{ .dense_fp8 = "all,gdn_in" }, error.UnknownDenseClass },
        .{ .{ .dense_fp8 = "" }, error.UnknownDenseClass },
        .{ .{ .dense_fp8 = "model.language_model.layers.1.self_attn.q_proj" }, error.TensorSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "model.language_model.layers.0.linear_attn.out_proj.weight" }, error.TensorSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "1" }, error.UnnamespacedLayer },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "shared_in,1" }, error.UnnamespacedLayer },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:" }, error.BadLayerSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:x" }, error.BadLayerSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:2" }, error.LayerOutOfRange },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "t:5" }, error.LayerOutOfRange },
        .{ .{ .keep_bf16 = "gdn_in" }, error.KeepWithoutDense },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:+0" }, error.BadLayerSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:-0" }, error.BadLayerSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:0_0" }, error.BadLayerSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:01" }, error.BadLayerSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "t:+4" }, error.BadLayerSelector },
        .{ .{ .dense_fp8 = "all", .keep_bf16 = "b:0,0_1" }, error.BadLayerSelector },
    };
    for (cases) |c| {
        try testing.expectError(c[1], convert.convert(gpa, io, try f.path("d"), try f.path("out"), c[0]));
        try testing.expectError(error.FileNotFound, f.tmp.dir.access(io, "out", .{}));
        try testing.expectError(error.FileNotFound, f.tmp.dir.access(io, "out.partial", .{}));
    }
}

test "statistics no operation reads are recorded unconsumed, and a fully subtracted selection keeps its dense_fp8 record" {
    var f = try init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const path = try stats(&f);
    const runs = [_]struct { []const u8, convert.Options }{
        .{ "kept", .{ .keep = &.{ 0, 1 }, .stats = path } }, // the Phase 1 invocation with every layer's experts kept FP8
        .{ "gdn", .{ .experts = .keep, .dense_fp8 = "gdn_in", .stats = path } },
        .{ "none", .{ .experts = .keep, .dense_fp8 = "gdn_in", .keep_bf16 = "b:0" } },
    };
    for (runs) |r| {
        _ = try convert.convert(testing.allocator, testing.io, try f.path("d"), try f.path(r[0]), r[1]);
        const prov = (try f.json(try std.fs.path.join(arena, &.{ r[0], "quant-provenance.json" }))).object;
        try testing.expectEqual(@as(usize, 0), prov.get("stats_consumed_by").?.array.items.len);
        const s = prov.get("stats").?;
        if (r[1].stats == null) try testing.expectEqual(std.json.Value.null, s) else {
            try testing.expectEqualStrings(path, s.object.get("path").?.string);
            try testing.expectEqual(@as(usize, 64), s.object.get("sha256").?.string.len);
            try testing.expect(!s.object.get("consumed").?.bool);
        }
    }
    const kept = (try f.json("kept/quant-provenance.json")).object;
    try testing.expectEqual(@as(usize, 0), kept.get("experts").?.object.get("converted_layers").?.array.items.len);
    try testing.expectEqual(std.json.Value.null, kept.get("dense_fp8").?);
    const gdn = try dense.Selection.init(arena, "gdn_in", null, .{ .backbone = 2, .tower = 5 });
    try expectConverted(&f, "d", "gdn", try resolve(&f, "d", &gdn));
    try expectConverted(&f, "d", "none", &.{});
    const selectors = (try f.json("none/quant-provenance.json")).object.get("dense_fp8").?.object.get("selectors").?.object;
    try testing.expectEqualStrings("b:0", selectors.get("keep_bf16").?.string);
}
