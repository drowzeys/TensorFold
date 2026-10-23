//! Compact checkpoints on the CPU reference path: the loader's puts and host routing against full and masked loads.
const std = @import("std");
const core = @import("core");
const cfg = @import("backbone_config.zig");
const weights = @import("backbone_weights.zig");
const compact = @import("compact.zig");
const expert_mask = @import("expert_mask.zig");
const quant = @import("quant_convert.zig");
const Source = @import("backbone_source.zig").Source;
const Stager = @import("stager.zig").Stager;

const fp8 = core.quant.fp8;
const Allocator = std.mem.Allocator;
const testing = std.testing;

const H = cfg.hidden;
const I = cfg.expert_inner;
const E = cfg.experts;
const top = cfg.top_k;
const layer = 3;
const pre = "model.language_model.layers.3.mlp.";
const router_name = pre ++ "gate.weight";
/// Layer 3's kept experts: not expert 0, so the first held expert, not expert 0, decides the compact layer's format.
const kept = [_]u8{ 2, 9, 17, 40, 41, 77, 128, 130, 199, 200, 254, 255 };

/// A tensor of a windowed shard: `len` bytes at `at` of the data.
const Win = struct { name: []const u8, dtype: []const u8, shape: []const usize, at: usize, len: usize };

/// One tensor per window of `data`: windows may overlap, so 256 experts at the real geometry take a few MiB.
fn writeShard(a: Allocator, dir: std.Io.Dir, sub: []const u8, data: []const u8, wins: []const Win, meta: []const u8) !void {
    var header: std.ArrayList(u8) = .empty;
    try header.append(a, '{');
    for (wins, 0..) |w, i| {
        try header.print(a, "{s}\"{s}\": {{\"dtype\": \"{s}\", \"shape\": [", .{ if (i == 0) "" else ", ", w.name, w.dtype });
        for (w.shape, 0..) |d, j| try header.print(a, "{s}{d}", .{ if (j == 0) "" else ", ", d });
        try header.print(a, "], \"data_offsets\": [{d}, {d}]}}", .{ w.at, w.at + w.len });
    }
    try header.append(a, '}');
    while (header.items.len % 8 != 0) try header.append(a, ' ');
    var file: std.ArrayList(u8) = .empty;
    try file.appendSlice(a, std.mem.asBytes(&std.mem.nativeToLittle(u64, header.items.len)));
    try file.appendSlice(a, header.items);
    try file.appendSlice(a, data);
    try dir.createDirPath(testing.io, sub);
    try dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ sub, "model.safetensors" }), .data = file.items });
    try writeIndex(a, dir, sub, wins, meta);
    try dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ sub, "config.json" }), .data = "{}" });
}

fn writeIndex(a: Allocator, dir: std.Io.Dir, sub: []const u8, wins: []const Win, meta: []const u8) !void {
    var index: std.ArrayList(u8) = .empty;
    try index.print(a, "{{\"metadata\": {{\"total_size\": 1{s}}}, \"weight_map\": {{", .{meta});
    for (wins, 0..) |w, i| try index.print(a, "{s}\"{s}\": \"model.safetensors\"", .{ if (i == 0) "" else ", ", w.name });
    try index.appendSlice(a, "}}");
    try dir.writeFile(testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ sub, "model.safetensors.index.json" }), .data = index.items });
}

/// `, "drama_experts": {...}`: layer 3 holding `ids` (null: all), the other 39 layers all 256.
fn mapMeta(a: Allocator, ids: ?[]const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, ", \"{s}\": {{", .{compact.experts_key});
    for (0..cfg.layers) |l| {
        try out.print(a, "{s}\"{d}\": [", .{ if (l == 0) "" else ", ", l });
        const held: []const u8 = if (l == layer and ids != null) ids.? else &full_ids;
        for (held, 0..) |e, j| try out.print(a, "{s}{d}", .{ if (j == 0) "" else ", ", e });
        try out.append(a, ']');
    }
    try out.append(a, '}');
    return out.items;
}

const full_ids = blk: {
    var ids: [E]u8 = undefined;
    for (&ids, 0..) |*x, e| x.* = @intCast(e);
    break :blk ids;
};

/// Layer 3's 256 FP8 experts and router: 4 KiB-apart code windows, f32-apart scale windows (2^-9..2^-6), bf16 router.
fn fp8Layer(a: Allocator) !struct { data: []u8, wins: []Win } {
    var prng = std.Random.DefaultPrng.init(0xF8);
    const rnd = prng.random();
    const codes_len = I * H + 3 * E * 4096;
    const scales_len = (16 * 4 + 3 * E) * 4;
    const data = try a.alloc(u8, codes_len + scales_len + E * H * 2);
    for (data[0..codes_len]) |*b| b.* = rnd.uintLessThan(u8, 0x7F) | (rnd.int(u8) & 0x80);
    const scales = std.mem.bytesAsSlice(f32, data[codes_len..][0..scales_len]);
    for (scales) |*s| s.* = std.math.ldexp(@as(f32, 1) + rnd.float(f32), -9 + @as(i32, rnd.uintLessThan(u8, 3)));
    const router = std.mem.bytesAsSlice(u16, data[codes_len + scales_len ..]);
    for (router) |*r| r.* = bf16(@floatCast((rnd.float(f64) - 0.5) / 16));
    var wins: std.ArrayList(Win) = .empty;
    try wins.append(a, .{ .name = router_name, .dtype = "BF16", .shape = &.{ E, H }, .at = codes_len + scales_len, .len = E * H * 2 });
    for (0..E) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| {
        const name = try std.fmt.allocPrint(a, pre ++ "experts.{d}.{s}.weight", .{ e, p });
        const shape: []const usize = if (j < 2) &.{ I, H } else &.{ H, I };
        const blocks: []const usize = if (j < 2) &.{ I / 128, H / 128 } else &.{ H / 128, I / 128 };
        try wins.append(a, .{ .name = name, .dtype = "F8_E4M3", .shape = shape, .at = (e * 3 + j) * 4096, .len = I * H });
        try wins.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}_scale_inv", .{name}), .dtype = "F32", .shape = blocks, .at = codes_len + (e * 3 + j) * 4, .len = 16 * 4 * 4 });
    };
    return .{ .data = data, .wins = wins.items };
}

/// Layer 3 with 256 NVFP4 experts: codes, valid e4m3 scales and positive globals in overlapping windows.
fn fp4Layer(a: Allocator) !struct { data: []u8, wins: []Win } {
    var prng = std.Random.DefaultPrng.init(0xF4);
    const rnd = prng.random();
    const codes_len = I * H / 2 + 3 * E * 4096;
    const scales_len = I * H / 16 + 3 * E * 64;
    const data = try a.alloc(u8, codes_len + scales_len + 3 * E * 4);
    rnd.bytes(data[0..codes_len]);
    for (data[codes_len..][0..scales_len]) |*b| b.* = rnd.uintLessThan(u8, 0x7F);
    for (std.mem.bytesAsSlice(f32, data[codes_len + scales_len ..])) |*g| g.* = 1e-3 + rnd.float(f32);
    var wins: std.ArrayList(Win) = .empty;
    for (0..E) |e| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| {
        const n: usize, const k: usize = if (j < 2) .{ I, H } else .{ H, I };
        const at = e * 3 + j;
        const base = try std.fmt.allocPrint(a, pre ++ "experts.{d}.{s}.weight", .{ e, p });
        try wins.append(a, .{ .name = base, .dtype = "U8", .shape = try a.dupe(usize, &.{ n, k / 2 }), .at = at * 4096, .len = n * k / 2 });
        try wins.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}_scale", .{base}), .dtype = "F8_E4M3", .shape = try a.dupe(usize, &.{ n, k / 16 }), .at = codes_len + at * 64, .len = n * k / 16 });
        try wins.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}_scale_2", .{base}), .dtype = "F32", .shape = &.{}, .at = codes_len + scales_len + at * 4, .len = 4 });
    };
    return .{ .data = data, .wins = wins.items };
}

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

fn widen(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// A checkpoint mapped, with a driverless Stager recording the loader's puts from its shard.
const Loaded = struct {
    src: Source,
    st: Stager,
    held: ?compact.Held,

    fn open(a: Allocator, root: []const u8, sub: []const u8) !Loaded {
        var x: Loaded = .{ .src = try Source.open(testing.allocator, testing.io, try std.fs.path.join(a, &.{ root, sub })), .st = .init(testing.allocator, undefined, undefined), .held = null };
        for (x.src.shards.items) |sh| try x.st.addFile(sh.path, sh.map.memory);
        x.held = try weights.expertMap(&x.src);
        return x;
    }

    /// Stager.deinit would synchronize its stream, which a driverless one does not have.
    fn close(x: *Loaded) void {
        x.st.files.deinit(testing.allocator);
        x.st.copies.deinit(testing.allocator);
        x.st.made.deinit();
        x.src.close();
    }

    fn ids(x: *const Loaded) []const u8 {
        return if (x.held) |*h| h.layer(layer) else &full_ids;
    }

    /// The bytes the loader put at `dst`.
    fn at(x: *const Loaded, dst: u64) []const u8 {
        for (x.st.copies.items) |c| if (c.dst == dst) {
            const f = x.st.files.items[c.file];
            return @as([*]const u8, @ptrFromInt(f.base))[c.offset..][0..c.len];
        };
        std.debug.panic("nothing put at 0x{x}", .{dst});
    }
};

/// Fake device addresses for one layer's FP8 arrays; put only records them.
const device: weights.Layer = .{ .gu = 1 << 40, .gu_s = 2 << 40, .dn = 3 << 40, .dn_s = 4 << 40 };
const staged: u64 = 5 << 40;

/// drama_topk on the host: softmax over the experts `keep` allows, the top 8 (lower id on ties), renormalized.
const Picks = struct { ids: [top]i32, wts: [top]f32 };

fn route(z: *const [E]f32, keep: ?[expert_mask.words]u32) Picks {
    const allowed = struct {
        fn f(k: ?[expert_mask.words]u32, e: usize) bool {
            return if (k) |m| m[e / 32] >> @intCast(e % 32) & 1 == 1 else true;
        }
    }.f;
    var mx = -std.math.inf(f32);
    for (z, 0..) |x, e| if (allowed(keep, e)) {
        mx = @max(mx, x);
    };
    var p: [E]f32 = undefined;
    var sum: f32 = 0;
    for (&p, z, 0..) |*q, x, e| {
        q.* = if (allowed(keep, e)) @exp(x - mx) else 0;
        sum += q.*;
    }
    var taken: [E]bool = undefined;
    for (&taken, 0..) |*t, e| t.* = !allowed(keep, e);
    var out: Picks = undefined;
    var s: f32 = 0;
    for (0..top) |k| {
        var best: usize = E;
        for (0..E) |e| if (!taken[e] and (best == E or p[e] / sum > p[best] / sum)) {
            best = e;
        };
        taken[best] = true;
        out.ids[k] = @intCast(best);
        out.wts[k] = p[best] / sum;
        s += out.wts[k];
    }
    for (&out.wts) |*w| w.* /= s;
    return out;
}

/// Router logits of row `x`: bf16(x . router[e]), from the checkpoint's own router rows.
fn logits(src: *Source, x: []const u16) ![E]f32 {
    const rows = std.mem.bytesAsSlice(u16, try src.get(router_name, .bf16, &.{ E, H }));
    var out: [E]f32 = undefined;
    for (&out, 0..) |*o, e| {
        var acc: f64 = 0;
        for (x, rows[e * H ..][0..H]) |a, r| acc += @as(f64, widen(a)) * widen(r);
        o.* = widen(bf16(@floatCast(acc)));
    }
    return out;
}

/// The routed experts' output for row `x` in float64, each pick read from the slot `slot(id)` the loader put it in.
fn experts(l: *const Loaded, x: []const u16, picks: Picks, slots: ?*const [E]i32) [H]f64 {
    var table: [256]f64 = undefined;
    for (&table, 0..) |*t, b| t.* = fp8.e4m3(@intCast(b));
    var y: [H]f64 = @splat(0);
    for (picks.ids, picks.wts) |id, w| {
        const s: usize = @intCast(if (slots) |t| t[@intCast(id)] else id);
        const gu = l.at(device.gu + s * weights.gu_rows * H);
        const up = l.at(device.gu + s * weights.gu_rows * H + I * H);
        const gs = std.mem.bytesAsSlice(f32, l.at(device.gu_s + s * weights.gu_scales * 4));
        const us = std.mem.bytesAsSlice(f32, l.at(device.gu_s + s * weights.gu_scales * 4 + weights.gu_scales * 2));
        const dn = l.at(device.dn + s * H * I);
        const ds = std.mem.bytesAsSlice(f32, l.at(device.dn_s + s * weights.dn_scales * 4));
        var h: [I]f64 = undefined;
        for (&h, 0..) |*o, j| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (0..H) |c| {
                const v: f64 = widen(x[c]);
                g += table[gu[j * H + c]] * gs[(j / 128) * (H / 128) + c / 128] * v;
                u += table[up[j * H + c]] * us[(j / 128) * (H / 128) + c / 128] * v;
            }
            o.* = g / (1 + @exp(-g)) * u;
        }
        for (&y, 0..) |*o, d| {
            var acc: f64 = 0;
            for (h, 0..) |v, j| acc += table[dn[d * I + j]] * ds[(d / 128) * (I / 128) + j / 128] * v;
            o.* += w * acc;
        }
    }
    return y;
}

fn rowsOf(a: Allocator, n: usize) ![]u16 {
    var prng = std.Random.DefaultPrng.init(0x0C0A);
    const x = try a.alloc(u16, n * H);
    for (x) |*v| v.* = bf16(@floatCast(prng.random().floatNorm(f64)));
    return x;
}

const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    root: []const u8,

    fn init() !Fixture {
        var f: Fixture = .{ .arena_state = .init(testing.allocator), .tmp = testing.tmpDir(.{}), .root = undefined };
        f.root = try f.tmp.dir.realPathFileAlloc(testing.io, ".", f.arena_state.allocator());
        return f;
    }

    fn deinit(f: *Fixture) void {
        f.tmp.cleanup();
        f.arena_state.deinit();
    }

    fn path(f: *Fixture, sub: []const u8) ![]const u8 {
        return std.fs.path.join(f.arena_state.allocator(), &.{ f.root, sub });
    }

    /// `src` compacted to the kept experts by tf-drama-quant, its map relisted for 40 layers as a full run writes it.
    fn compacted(f: *Fixture, src: []const u8, dst: []const u8) !void {
        const a = f.arena_state.allocator();
        var mask: std.ArrayList(u8) = .empty;
        try mask.print(a, "{{\"{d}\": [", .{layer});
        for (kept, 0..) |e, i| try mask.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", e });
        try mask.appendSlice(a, "]}");
        try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "mask.json", .data = mask.items });
        _ = try quant.convert(testing.allocator, testing.io, try f.path(src), try f.path(dst), .{ .experts = .keep, .compact_experts = try f.path("mask.json") });
        const index_path = try std.fs.path.join(a, &.{ dst, "model.safetensors.index.json" });
        const index = try std.json.parseFromSliceLeaky(std.json.Value, a, try f.tmp.dir.readFileAlloc(testing.io, index_path, a, .limited(1 << 24)), .{});
        const map = index.object.get("metadata").?.object.get(compact.experts_key).?.object;
        try testing.expectEqual(@as(usize, 1), map.count());
        const ids = map.get("3").?.array.items;
        try testing.expectEqual(kept.len, ids.len);
        for (ids, kept) |v, e| try testing.expectEqual(@as(i64, e), v.integer);
        var wins: std.ArrayList(Win) = .empty;
        for (index.object.get("weight_map").?.object.keys()) |name| try wins.append(a, .{ .name = name, .dtype = "", .shape = &.{}, .at = 0, .len = 0 });
        try writeIndexOnly(a, f.tmp.dir, dst, wins.items, try mapMeta(a, &kept));
    }
};

fn writeIndexOnly(a: Allocator, dir: std.Io.Dir, sub: []const u8, wins: []const Win, meta: []const u8) !void {
    try dir.deleteFile(testing.io, try std.fs.path.join(a, &.{ sub, "model.safetensors.index.json" }));
    try writeIndex(a, dir, sub, wins, meta);
}

test "FP8: every expert held puts the full load's bytes; held experts route, and compute, as the masked full layer" {
    var f = try Fixture.init();
    defer f.deinit();
    const a = f.arena_state.allocator();
    const layer3 = try fp8Layer(a);
    try writeShard(a, f.tmp.dir, "full", layer3.data, layer3.wins, "");
    try writeShard(a, f.tmp.dir, "all", layer3.data, layer3.wins, try mapMeta(a, null));
    try f.compacted("full", "small");
    var full = try Loaded.open(a, f.root, "full");
    defer full.close();
    var all = try Loaded.open(a, f.root, "all");
    defer all.close();
    var small = try Loaded.open(a, f.root, "small");
    defer small.close();
    try testing.expect(full.held == null and all.held != null);
    try testing.expectEqualSlices(u8, &kept, small.ids());
    for ([_]*Loaded{ &full, &all, &small }) |l| try testing.expect(try weights.putExperts(&l.src, &l.st, layer, l.ids(), &device, staged) == null);
    // every expert held: the same puts, destination and bytes, as the map-less load
    try testing.expectEqual(full.st.copies.items.len, all.st.copies.items.len);
    for (full.st.copies.items) |c| try testing.expectEqualSlices(u8, full.at(c.dst), all.at(c.dst));
    // the compact layer puts each held expert's six tensors in its slot, from bytes equal to the full layer's
    try testing.expectEqual(kept.len * 6, small.st.copies.items.len);
    for (kept, 0..) |e, s| for ([_]u64{ device.gu, device.gu + I * H, device.dn }, [_]usize{ weights.gu_rows * H, weights.gu_rows * H, H * I }) |base, stride| {
        try testing.expectEqualSlices(u8, full.at(base + e * stride), small.at(base + s * stride));
    };
    const tables = compact.tables(&small.held.?, null).?;
    const all_tables = compact.tables(&all.held.?, null).?;
    var mask: expert_mask.Mask = @splat(@splat(std.math.maxInt(u32)));
    mask[layer] = @splat(0);
    for (kept) |e| mask[layer][e / 32] |= @as(u32, 1) << @intCast(e % 32);
    // the compact layer's keep row is the mask's, so drama_topk reads the same row either way
    try testing.expectEqual(mask[layer], tables.keep[layer]);
    const x = try rowsOf(a, 1);
    var pruned: usize = 0;
    for (0..1) |r| {
        const row = x[r * H ..][0..H];
        const z = try logits(&full.src, row);
        try testing.expectEqual(z, try logits(&small.src, row));
        // every expert held against the full layer unmasked
        const p_full = route(&z, null);
        const p_all = route(&z, all_tables.keep[layer]);
        try testing.expectEqualSlices(u8, std.mem.asBytes(&p_full), std.mem.asBytes(&p_all));
        const y_full = experts(&full, row, p_full, null);
        try testing.expectEqualSlices(u8, std.mem.asBytes(&y_full), std.mem.asBytes(&experts(&all, row, p_all, &all_tables.slots.?[layer])));
        // the compact layer against the full one masked to its experts: picks, weights and outputs byte for byte
        const p_masked = route(&z, mask[layer]);
        const p_small = route(&z, tables.keep[layer]);
        try testing.expectEqualSlices(u8, std.mem.asBytes(&p_masked), std.mem.asBytes(&p_small));
        const y_masked = experts(&full, row, p_masked, null);
        try testing.expectEqualSlices(u8, std.mem.asBytes(&y_masked), std.mem.asBytes(&experts(&small, row, p_small, &tables.slots.?[layer])));
        for (p_full.ids) |id| pruned += @intFromBool(std.mem.indexOfScalar(u8, &kept, @intCast(id)) == null);
    }
    try testing.expect(pruned > 0); // the mask changed the routing, so the comparison covers it
}

test "NVFP4: the compact layer stages each held expert's codes, scales and globals in its slot, as the full layer has them" {
    var f = try Fixture.init();
    defer f.deinit();
    const a = f.arena_state.allocator();
    const layer3 = try fp4Layer(a);
    try writeShard(a, f.tmp.dir, "full", layer3.data, layer3.wins, "");
    try f.compacted("full", "small");
    var full = try Loaded.open(a, f.root, "full");
    defer full.close();
    var small = try Loaded.open(a, f.root, "small");
    defer small.close();
    const g_full = (try weights.putExperts(&full.src, &full.st, layer, full.ids(), &device, staged)).?;
    const g_small = (try weights.putExperts(&small.src, &small.st, layer, small.ids(), &device, staged)).?;
    try testing.expectEqual(kept.len * 6, small.st.copies.items.len);
    for (kept, 0..) |e, s| for ([_][2]usize{ .{ I, H }, .{ I, H }, .{ H, I } }, 0..) |shape, j| {
        const n, const k = shape;
        const at = staged + j * weights.fp4_part;
        try testing.expectEqualSlices(u8, full.at(at + e * n * k / 2), small.at(at + s * n * k / 2));
        try testing.expectEqualSlices(u8, full.at(at + E * n * k / 2 + e * n * k / 16), small.at(at + E * n * k / 2 + s * n * k / 16));
        try testing.expectEqual(g_full[j][e], g_small[j][s]);
    };
}

test "the loader refuses a map that leaves out a stored expert, and a mask beyond the held experts" {
    var f = try Fixture.init();
    defer f.deinit();
    const a = f.arena_state.allocator();
    const layer3 = try fp8Layer(a);
    // the full layer's tensors under a map that holds only the kept experts: the others are strays
    try writeShard(a, f.tmp.dir, "stray", layer3.data, layer3.wins, try mapMeta(a, &kept));
    var src = try Source.open(testing.allocator, testing.io, try f.path("stray"));
    defer src.close();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const held = (try compact.parse(arena.allocator(), src.metadata.get(compact.experts_key).?, &problem)).?;
    const name = compact.stray(&held, src.names.keyIterator()).?;
    const x = compact.expertOf(name).?;
    try testing.expect(x.layer == layer and std.mem.indexOfScalar(u8, &kept, @intCast(x.expert)) == null);
    var mask: expert_mask.Mask = @splat(@splat(std.math.maxInt(u32)));
    mask[layer] = held.row(layer);
    try testing.expect(compact.unheld(&held, &mask) == null);
    mask[layer][0] |= 1; // expert 0, which the map does not hold
    try testing.expectEqual(@as(usize, 0), compact.unheld(&held, &mask).?.expert);
}
