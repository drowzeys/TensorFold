//! The tower's projections stored fp8g (C5): e4m3 codes, an f32 scale per row and 64 inputs, checked and lane-packed.
const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("core").checkpoint;
const Stager = @import("stager.zig").Stager;
const Buffer = cuda.DeviceBuffer;
const fp8 = cuda.fp8;
const qmmf = cuda.qmmf;
const qlinear = cuda.qlinear;

/// A projection of kinds 1-4 (wqkv, wo, gate_up, down): the checkpoint's BF16 rows, or fp8g codes and row scales.
pub const Projection = union(enum) { bf16: Buffer, fp8g: qmmf.Weight };

fn kinds(layer: anytype) [4]Projection {
    return .{ layer.qkv, layer.out, layer.gate_up, layer.down };
}

/// Kind `shape` from `parts` at `prefix`, fused in order, as stored (C5, C8); BF16 allocates and copies as it did.
pub fn load(owner: anytype, st: *Stager, ck: *checkpoint.Checkpoint, staging: *Staging, prefix: []const u8, comptime parts: []const []const u8, shape: [2]usize) !Projection {
    const n = shape[0] / parts.len;
    const k = shape[1];
    var keys: [parts.len][256]u8 = undefined;
    var names: [parts.len][]const u8 = undefined;
    for (parts, &keys, &names) |part, *key, *name| name.* = try std.fmt.bufPrint(key, "{s}.transformer_layer.{s}.weight", .{ prefix, part });
    const found = format(ck, &names) catch |e| {
        std.log.err("drama tower: {s}: {t}", .{ names[0], e });
        return e;
    };
    switch (found) {
        .bf16 => {
            const out = try owner.allocate(parts.len * n * k * 2);
            for (names, 0..) |name, j| try st.put(out.ptr + j * n * k * 2, (try ck.expect(name, .bf16, &.{ n, k })).bytes);
            return .{ .bf16 = out };
        },
        .fp8g => {
            const kg = k / fp8.group;
            const codes = try owner.allocate(parts.len * n * k);
            const scales = try owner.allocate(parts.len * n * kg * 4);
            // fused parts are whole 128-row blocks, so their packs concatenate
            for (names, 0..) |name, j| {
                const s = stored(ck, name, n, k) catch |e| {
                    std.log.err("drama tower: {s}: {t}", .{ name, e });
                    return e;
                };
                try st.put(try staging.add(codes.ptr + j * n * k, n, k), s.codes);
                try putScales(owner.gpa, st, scales.ptr + j * n * kg * 4, s.scales, n, k);
            }
            return .{ .fp8g = fp8.weight(codes.ptr, scales.ptr, parts.len * n, k) };
        },
    }
}

/// The tower's own lane (D10) and its slice scratch, loaded only once a layer holds fp8g.
pub const Lane = struct {
    lane: ?qmmf.Lane = null,
    part: ?Buffer = null,

    pub fn unload(l: *Lane) void {
        if (l.lane) |*lane| lane.unload();
    }

    /// With fp8g layers: lane and `part` up to max_rows; a BF16-only tower returns before asking `ctx`.
    pub fn prepare(l: *Lane, owner: anytype, ctx: anytype) !void {
        const c = census(&owner.layers);
        if (c.layers == 0) return;
        if (l.lane == null) l.lane = try qmmf.Lane.load(owner.driver, (try ctx.capability()) / 10);
        var bytes: usize = 0;
        for (owner.layers) |layer| {
            const ws = kinds(layer);
            bytes = @max(bytes, partBytes(l.lane.?, &ws, owner.options.max_rows));
        }
        if (bytes > (if (l.part) |b| b.len else 0)) l.part = try owner.allocate(bytes);
    }

    /// y [rows, n] = x [rows, k] times an fp8g weight, one kernel form at every row count (C7).
    pub fn matmul(l: *const Lane, s: cuda.Stream, w: qmmf.Weight, x: Buffer, out: Buffer, rows: usize) !void {
        const lanes: qlinear.Linear = .{ .lane = if (l.lane) |*lane| lane else null };
        return lanes.decode(s, .{ .fp8g = w }, .{ .x = x.ptr, .part = if (l.part) |b| b.ptr else 0 }, out.ptr, rows);
    }
};

/// Layers holding any fp8g projection, and how many hold each of kinds 1-4 fp8g.
pub const Census = struct {
    layers: usize = 0,
    kinds: [4]usize = @splat(0),

    /// `M of 13 tower layers (tower_qkv a, ...)`, by C5 class: the tower's part of the startup line's dense FP8 count.
    pub fn write(c: Census, w: *std.Io.Writer, layers: usize) !void {
        try w.print("{d} of {d} tower layers (tower_qkv {d}, tower_o {d}, tower_gate_up {d}, tower_down {d})", .{ c.layers, layers, c.kinds[0], c.kinds[1], c.kinds[2], c.kinds[3] });
    }
};

pub fn census(layers: anytype) Census {
    var c: Census = .{};
    for (layers) |layer| {
        var any = false;
        for (kinds(layer), &c.kinds) |w, *count| if (w == .fp8g) {
            count.* += 1;
            any = true;
        };
        c.layers += @intFromBool(any);
    }
    return c;
}

/// The code bytes of one layer's kinds 1-4: the staging room.
pub fn codeBytes(shapes: []const [2]usize) usize {
    var bytes: usize = 0;
    for (shapes) |shape| bytes += shape[0] * shape[1];
    return bytes;
}

pub const Format = enum { bf16, fp8g };

/// A fused group's format from its tensors' dtypes, which must all agree (C5: one format per layer and group).
pub fn groupFormat(dtypes: []const checkpoint.DType) error{ UnexpectedTowerTensor, MixedTowerGroup }!Format {
    var found: ?Format = null;
    for (dtypes) |dtype| {
        const f: Format = switch (dtype) {
            .bf16 => .bf16,
            .f8_e4m3 => .fp8g,
            else => return error.UnexpectedTowerTensor,
        };
        if (found != null and found.? != f) return error.MixedTowerGroup;
        found = f;
    }
    return found.?;
}

/// The format of the group `names`, taking each tensor.
pub fn format(ck: *checkpoint.Checkpoint, names: []const []const u8) !Format {
    var dtypes: [2]checkpoint.DType = undefined;
    for (names, dtypes[0..names.len]) |name, *dtype| dtype.* = (try ck.get(name)).dtype;
    return groupFormat(dtypes[0..names.len]);
}

/// One fp8g part as C5 stores it: e4m3 codes [n][k] and their f32 `weight_scale_inv` [n][k/64].
pub const Stored = struct { codes: []const u8, scales: []const u8 };

/// `weight` and its companion, refused unless C5's dtypes and shapes, no NaN code and only positive finite scales.
pub fn stored(ck: *checkpoint.Checkpoint, weight: []const u8, n: usize, k: usize) !Stored {
    var name: [256]u8 = undefined;
    const companion = try std.fmt.bufPrint(&name, "{s}_scale_inv", .{weight});
    if (!ck.has(companion)) return error.MissingTowerScale;
    const codes = try ck.get(weight);
    const scales = try ck.get(companion);
    if (!codes.is(.f8_e4m3, &.{ n, k }) or !scales.is(.f32, &.{ n, k / fp8.group })) return error.UnexpectedTowerTensor;
    try checkCodes(codes.bytes);
    try checkScales(scales.bytes);
    return .{ .codes = codes.bytes, .scales = scales.bytes };
}

/// The lane reads 0x7F and 0xFF as ±480 (it has no NaN branch); the host decoder reads NaN.
fn checkCodes(codes: []const u8) error{BadTowerFp8Code}!void {
    var nan: u8 = 0; // or-reduced, not a branch a byte, so the scan of every code vectorizes
    for (codes) |c| nan |= @intFromBool(c & 0x7f == 0x7f);
    if (nan != 0) return error.BadTowerFp8Code;
}

/// Positive and finite; positive subnormals are admitted (C6.7).
fn checkScales(bytes: []const u8) error{BadTowerFp8Scale}!void {
    for (0..bytes.len / 4) |i| {
        const s: f32 = @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little));
        if (!(s > 0 and std.math.isFinite(s))) return error.BadTowerFp8Scale;
    }
}

/// `scales` as stored in the lane's tiles, for the stager to copy to `dst`.
pub fn putScales(gpa: std.mem.Allocator, st: *Stager, dst: u64, scales: []const u8, n: usize, k: usize) !void {
    const kg = k / fp8.group;
    const rows = try gpa.alloc(f32, n * kg);
    defer gpa.free(rows);
    @memcpy(std.mem.sliceAsBytes(rows), scales);
    const tiles = try gpa.alloc(f32, fp8.padded(n) * kg);
    defer gpa.free(tiles);
    fp8.packRowScales(tiles, rows, n, k);
    try st.put(dst, std.mem.sliceAsBytes(tiles));
}

/// The lane's slice partials for `ws` at every row count up to `max_rows`: 0 on the cluster path.
pub fn partBytes(lane: qmmf.Lane, ws: []const Projection, max_rows: usize) usize {
    var bytes: usize = 0;
    for (ws) |w| if (w == .fp8g) for (1..max_rows + 1) |m| {
        bytes = @max(bytes, lane.partBytes(m, w.fp8g));
    };
    return bytes;
}

/// One layer's stored codes on the device until packed into lane order; room and packer appear with the first fp8g.
pub const Staging = struct {
    d: *const cuda.Driver,
    bytes: usize,
    room: ?Room = null,
    used: usize = 0,
    packs: [max_parts]Pack = undefined,
    count: usize = 0,

    const Room = struct { raw: Buffer, packer: cuda.modelopt.Packer };
    const max_parts = 5; // wqkv, wo, w1, w3, w2
    const Pack = struct { at: u64, out: u64, n: usize, k: usize };

    /// Frees the room once `stream` has run every pack queued from it.
    pub fn deinit(s: *Staging, stream: cuda.Stream) void {
        if (s.room) |*r| {
            stream.synchronize() catch {};
            r.raw.free();
            r.packer.unload();
        }
    }

    /// Where an n x k part's stored codes go, packed into `out` by `pack`.
    fn add(s: *Staging, out: u64, n: usize, k: usize) !u64 {
        if (s.room == null) {
            var packer = try cuda.modelopt.Packer.load(s.d);
            errdefer packer.unload();
            s.room = .{ .raw = try Buffer.alloc(s.d, s.bytes), .packer = packer };
        }
        std.debug.assert(s.count < max_parts and s.used + n * k <= s.bytes);
        const at = s.room.?.raw.ptr + s.used;
        s.packs[s.count] = .{ .at = at, .out = out, .n = n, .k = k };
        s.count += 1;
        s.used += n * k;
        return at;
    }

    /// After a layer: lands its staged codes, then queues their packs, so the next layer's copies queue after them.
    pub fn pack(s: *Staging, st: *Stager) !void {
        if (s.count == 0) return;
        try st.finish();
        for (s.packs[0..s.count]) |p| try s.room.?.packer.fp8Lane(st.stream, p.at, p.out, p.n, p.k);
        s.count = 0;
        s.used = 0;
    }
};

test "the startup census counts layers holding fp8g and each kind's fp8g layers" {
    const T = struct { qkv: Projection, out: Projection, gate_up: Projection, down: Projection };
    const b: Projection = .{ .bf16 = .{ .d = undefined, .ptr = 0x1000, .len = 2, .borrowed = true } };
    const q: Projection = .{ .fp8g = fp8.weight(0, 0, 2048, 4096) };
    const plain: [13]T = @splat(.{ .qkv = b, .out = b, .gate_up = b, .down = b });
    try std.testing.expectEqual(Census{}, census(&plain));
    const layers = [_]T{
        .{ .qkv = q, .out = q, .gate_up = q, .down = q },
        .{ .qkv = b, .out = b, .gate_up = b, .down = b },
        .{ .qkv = q, .out = b, .gate_up = b, .down = q },
        .{ .qkv = b, .out = b, .gate_up = q, .down = b },
    };
    try std.testing.expectEqual(Census{ .layers = 3, .kinds = .{ 2, 1, 2, 2 } }, census(&layers));
}

test "a BF16-only tower never asks for the capability; an fp8g one asks before it loads the lane" {
    const T = struct { qkv: Projection, out: Projection, gate_up: Projection, down: Projection };
    const Ctx = struct {
        fn capability(_: @This()) !u32 {
            return error.CapabilityQueried;
        }
    };
    const Owner = struct {
        layers: [2]T,
        driver: *const cuda.Driver = undefined,
        options: struct { max_rows: usize = 1 } = .{},
        fn allocate(_: *@This(), _: usize) !Buffer {
            return error.Allocated;
        }
    };
    const b: Projection = .{ .bf16 = .{ .d = undefined, .ptr = 0x1000, .len = 2, .borrowed = true } };
    const q: Projection = .{ .fp8g = fp8.weight(0, 0, 2048, 4096) };
    const plain: T = .{ .qkv = b, .out = b, .gate_up = b, .down = b };
    var lane: Lane = .{};
    var bf16_only: Owner = .{ .layers = .{ plain, plain } };
    try lane.prepare(&bf16_only, Ctx{});
    try std.testing.expect(lane.lane == null and lane.part == null);
    var mixed: Owner = .{ .layers = .{ plain, .{ .qkv = b, .out = q, .gate_up = b, .down = b } } };
    try std.testing.expectError(error.CapabilityQueried, lane.prepare(&mixed, Ctx{}));
}

test "a fused group's format is its dtypes' and must be one; other dtypes are refused" {
    try std.testing.expectEqual(Format.bf16, try groupFormat(&.{ .bf16, .bf16 }));
    try std.testing.expectEqual(Format.fp8g, try groupFormat(&.{ .f8_e4m3, .f8_e4m3 }));
    try std.testing.expectEqual(Format.fp8g, try groupFormat(&.{.f8_e4m3}));
    try std.testing.expectError(error.MixedTowerGroup, groupFormat(&.{ .bf16, .f8_e4m3 }));
    try std.testing.expectError(error.MixedTowerGroup, groupFormat(&.{ .f8_e4m3, .bf16 }));
    for ([_]checkpoint.DType{ .f16, .f32, .u8, .f8_e5m2 }) |d| try std.testing.expectError(error.UnexpectedTowerTensor, groupFormat(&.{d}));
}

test "codes may be saturated or subnormal but never NaN; scales positive and finite, subnormals included" {
    try checkCodes(&.{ 0x00, 0x01, 0x07, 0x08, 0x7E, 0x80, 0x81, 0xFE });
    for ([_]u8{ 0x7F, 0xFF }) |b| try std.testing.expectError(error.BadTowerFp8Code, checkCodes(&.{ 0x38, 0x38, b, 0x38 }));
    // 146 x 2^-149: the stored scale of the smallest positive bf16 group (C6.7)
    for ([_]f32{ @bitCast(@as(u32, 146)), @bitCast(@as(u32, 1)), 0x1p-126, 1, 0x1p8, std.math.floatMax(f32) }) |s| try checkScales(std.mem.asBytes(&s));
    for ([_]f32{ 0, -0.0, -1, -0x1p-149, std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32) }) |s| {
        const row = [_]f32{ 1, s, 1 };
        try std.testing.expectError(error.BadTowerFp8Scale, checkScales(std.mem.sliceAsBytes(&row)));
    }
}

test "a stored part is read as C5 writes it; a wrong companion is refused, a stray one left for unused()" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    const n = 2;
    const k = 128;
    var codes: [n * k]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @truncate(i);
    for (&codes) |*c| if (c.* & 0x7f == 0x7f) {
        c.* -= 1;
    };
    const scales = [n * 2]f32{ @bitCast(@as(u32, 146)), 0x1p8, 1, 0x1p-20 };
    const bf16_bytes: [n * k * 2]u8 = @splat(0x3c);
    const shape = &[_]usize{ n, k };
    const scale_shape = &[_]usize{ n, 2 };
    const tensors = [_]@import("core").shard_edit.New{
        .{ .name = "good.weight", .dtype = .f8_e4m3, .shape = shape, .bytes = &codes },
        .{ .name = "good.weight_scale_inv", .dtype = .f32, .shape = scale_shape, .bytes = std.mem.sliceAsBytes(&scales) },
        .{ .name = "f16_scale.weight", .dtype = .f8_e4m3, .shape = shape, .bytes = &codes },
        .{ .name = "f16_scale.weight_scale_inv", .dtype = .f16, .shape = scale_shape, .bytes = std.mem.sliceAsBytes(&scales)[0..8] },
        .{ .name = "block_scale.weight", .dtype = .f8_e4m3, .shape = shape, .bytes = &codes },
        .{ .name = "block_scale.weight_scale_inv", .dtype = .f32, .shape = &.{ 1, 1 }, .bytes = std.mem.sliceAsBytes(&scales)[0..4] },
        .{ .name = "orphan.weight", .dtype = .f8_e4m3, .shape = shape, .bytes = &codes },
        .{ .name = "kept.weight", .dtype = .bf16, .shape = shape, .bytes = &bf16_bytes },
        .{ .name = "kept.weight_scale_inv", .dtype = .f32, .shape = scale_shape, .bytes = std.mem.sliceAsBytes(&scales) },
    };
    const shard = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(shard);
    try @import("core").shard_edit.create(gpa, shard, &.{}, &tensors);
    var ck = try checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const s = try stored(&ck, "good.weight", n, k);
    try std.testing.expectEqualSlices(u8, &codes, s.codes);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&scales), s.scales);
    try std.testing.expectError(error.UnexpectedTowerTensor, stored(&ck, "f16_scale.weight", n, k));
    try std.testing.expectError(error.UnexpectedTowerTensor, stored(&ck, "block_scale.weight", n, k));
    try std.testing.expectError(error.UnexpectedTowerTensor, stored(&ck, "good.weight", n, 2 * k));
    try std.testing.expectError(error.MissingTowerScale, stored(&ck, "orphan.weight", n, k));
    // a BF16 weight's companion is never taken, so the loader's unused() refuses the checkpoint over it
    try std.testing.expectEqual(Format.bf16, try format(&ck, &.{"kept.weight"}));
    try std.testing.expect(ck.used.contains("kept.weight") and !ck.used.contains("kept.weight_scale_inv"));
}

test "slice partials: none where slices meet in a cluster, else the largest reduce over fp8g parts and row counts" {
    var lane: qmmf.Lane = undefined;
    const bf16: Projection = .{ .bf16 = .{ .d = undefined, .ptr = 0x1000, .len = 2, .borrowed = true } };
    const ws = [_]Projection{
        .{ .fp8g = fp8.weight(0, 0, 9216, 2048) }, // 2 slices
        .{ .fp8g = fp8.weight(0, 0, 2048, 4096) }, // 8 slices
        .{ .fp8g = fp8.weight(0, 0, 12288, 2048) }, // 1 slice: no partials
        bf16,
    };
    lane.clusters = true;
    try std.testing.expectEqual(@as(usize, 0), partBytes(lane, &ws, 128));
    lane.clusters = false;
    try std.testing.expectEqual(@as(usize, 2 * 128 * 9216 * 4), partBytes(lane, &ws, 128));
    try std.testing.expectEqual(@as(usize, 8 * 7 * 2048 * 4), partBytes(lane, ws[1..], 7));
    try std.testing.expectEqual(@as(usize, 0), partBytes(lane, &.{ ws[2], bf16 }, 128));
}
