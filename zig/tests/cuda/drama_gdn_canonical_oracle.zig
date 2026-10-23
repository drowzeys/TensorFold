//! Private verification driver, not a gate by itself: runs the production Drama GDN controller
//! (`drama.backbone.gdn`: State, Workspace, Kernels) on deterministic raw fixtures and writes every output, every
//! boundary state and the controller metadata for an external checker to judge against the pinned reference.
//!
//! usage: tf-drama-gdn-canonical-oracle GLUE_FATBIN FROZEN_FATBIN CASES_DIR OUTPUT_DIR
//!
//! GLUE_FATBIN is the produced drama_backbone.fatbin (loaded with cuda.Module; gdn.Kernels.init resolves the
//! controller's kernels from it). FROZEN_FATBIN must be the frozen pre-chunk drama_backbone image
//! (SHA-256 pinned below): its unchanged `drama_gdn_recur` is the decode baseline and its `drama_gdn_recur_prefill`
//! the old sequential prefill control. CASES_DIR holds cases.json (schema drama-gdn-canonical-cases/1, written by
//! that checker) and the raw tensors it names; OUTPUT_DIR must exist and every output is created
//! exclusively. No model weights are read.
//!
//! 1. Fixtures (CPU only): cases.json is parsed strictly; every raw file must have its exact size and manifest
//!    SHA-256. Any failure exits 2 before the GPU is touched.
//! 2. Memory, once per geometry: a_log/dt_bias, the padded input streams, two layers' caller-owned regions
//!    (guard | bank0 | guard | bank1 | guard | qkv_cache | guard | ba_cache | guard), one workspace sized for the
//!    largest case (front and back guards), guarded output rows, the frozen controls' states. Nothing is allocated
//!    while a case runs; the counted device bytes are recorded and checked constant.
//! 3. Each case runs twice (pass 1 must reproduce pass 0 bit for bit). A pass starts by resetting both layers:
//!    zero banks and caches (or NaN caches for an explicit poison reset), State.resetMetadata(), then a nonzero
//!    initial state uploaded into bank0. Ops call exactly gdn.Kernels.prefill / gdn.Kernels.decode on the layer's
//!    State, the shared placed Workspace and the caller stream. Before every prefill the workspace and the output
//!    rows are filled with 0xFF (NaN); afterwards rows [0, n) must all be written and every guard byte must be intact.
//!    Inputs (streams, a_log, dt_bias) are compared byte for byte after every op. Each decode first copies
//!    State.current() and runs the frozen image's decode export on that copy and the same rows: output and state
//!    must be bit identical. The old sequential chain (frozen prefill/decode exports on its own state) runs next to
//!    it for the old-discrepancy control.
//! 4. result.json: per op metadata (pending/committed before and after, which bank current() returned), SHA-256 of
//!    every output and state, coverage, overrun, guard, input and repeat findings, timings, allocations, image and
//!    binary hashes. evidence_valid is false (exit 3) on any recorded error. A CUDA or controller error aborts.
//!
//! The caller owns the shared GPU lock, admission, process cap and lifetime monitoring.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const gdn = drama.backbone.gdn;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const producer = "tf-drama-gdn-canonical-oracle";
const cases_schema = "drama-gdn-canonical-cases/1";
const result_schema = "drama-gdn-canonical-native/1";
const frozen_image_sha256 = "60ec0492bd5926c12bec45b1a555022d080bfe187bb9180d89aec36a0ac43b7f";
const frozen_decode = "drama_gdn_recur";
const frozen_prefill = "drama_gdn_recur_prefill";
const dv = 128;
const chunk = gdn.chunk_rows;
const layer_count = 2;
const passes = 2;
const guard = 4096; // bytes; a multiple of 256 keeps every placed buffer aligned like cuMemAlloc
const y_guard_rows = 2;
const fill = 0xff; // bf16 0xFFFF and f32 0xFFFFFFFF are NaN: no finite result has these bytes
const device_limit = 1 << 30;
const max_file = 1 << 28;
/// Workspace fields read for the gate probe when the component's (free) representation has them as device
/// addresses: FP32 [capacity rows][nv] each, as the prototype workspace. Absent fields are reported, not invented.
const probe_fields = [_][]const u8{ "g", "gc", "beta" };
const probe_available = blk: {
    for (probe_fields) |f| {
        if (!@hasField(gdn.Workspace, f)) break :blk false;
        if (@FieldType(gdn.Workspace, f) != u64) break :blk false;
    }
    break :blk true;
};

comptime {
    std.debug.assert(chunk == 64);
}

// ---------------------------------------------------------------------------------------------------------------
// manifest

const BadCases = error{BadCases};

fn bad(comptime f: []const u8, args: anytype) BadCases {
    std.debug.print("cases manifest: " ++ f ++ "\n", args);
    return error.BadCases;
}

const File = struct { name: []const u8, sha: []const u8, bytes: []align(16) const u8 };

const Geometry = struct {
    name: []const u8,
    nk: usize,
    nv: usize,
    a_log: File,
    dt_bias: File,
    max_rows: usize = 0, // largest case max_rows: output rows allocated
    ws_bytes: usize = 0, // largest Workspace.layout size over this geometry's cases
    used: bool = false,

    fn width(g: Geometry) usize {
        return (2 * g.nk + g.nv) * dv;
    }
    fn stateBytes(g: Geometry) usize {
        return g.nv * dv * dv * 4;
    }
    fn rowBytes(g: Geometry) usize {
        return g.nv * dv * 2;
    }
    fn qkvCacheBytes(g: Geometry) usize {
        return chunk * g.width() * 2;
    }
    fn baCacheBytes(g: Geometry) usize {
        return chunk * 2 * g.nv * 2;
    }
};

const Stream = struct { name: []const u8, geometry: usize, rows: usize, ld_qkv: usize, ld_ba: usize, qkv: File, ba: File };
const Init = struct { name: []const u8, geometry: usize, file: File };
const OpKind = enum { prefill, decode, reset };
const Op = struct {
    kind: OpKind,
    layer: usize,
    stream: usize = 0,
    start: usize = 0,
    rows: usize = 0,
    probe: bool = false,
    init: ?usize = null, // reset: null = zero state
    poison_cache: bool = false,
};
const Case = struct {
    name: []const u8,
    geometry: usize,
    max_rows: usize,
    init: [layer_count]?usize,
    ops: []const Op,
    save_old_states: bool,
    last_op: [layer_count]?usize, // the layer's last prefill/decode op
};

const Cases = struct {
    sha: [64]u8,
    geometries: []Geometry,
    streams: []Stream,
    inits: []Init,
    cases: []Case,
};

fn get(o: std.json.ObjectMap, key: []const u8) BadCases!std.json.Value {
    return o.get(key) orelse bad("missing key {s}", .{key});
}

fn uintOf(o: std.json.ObjectMap, key: []const u8) BadCases!usize {
    const v = try get(o, key);
    if (v != .integer or v.integer < 0) return bad("{s} is not a non-negative integer", .{key});
    return @intCast(v.integer);
}

fn textOf(o: std.json.ObjectMap, key: []const u8) BadCases![]const u8 {
    const v = try get(o, key);
    if (v != .string) return bad("{s} is not a string", .{key});
    return v.string;
}

fn boolOr(o: std.json.ObjectMap, key: []const u8, default: bool) BadCases!bool {
    const v = o.get(key) orelse return default;
    if (v != .bool) return bad("{s} is not a bool", .{key});
    return v.bool;
}

fn objectOf(v: std.json.Value, what: []const u8) BadCases!std.json.ObjectMap {
    if (v != .object) return bad("{s} is not an object", .{what});
    return v.object;
}

fn arrayOf(o: std.json.ObjectMap, key: []const u8) BadCases![]std.json.Value {
    const v = try get(o, key);
    if (v != .array) return bad("{s} is not an array", .{key});
    return v.array.items;
}

/// Names become file names: lowercase letters, digits, '_', '-', '.', not leading '.', at most 96 bytes.
fn safeName(s: []const u8) BadCases![]const u8 {
    if (s.len == 0 or s.len > 96 or s[0] == '.') return bad("unsafe name '{s}'", .{s});
    for (s) |c| switch (c) {
        'a'...'z', '0'...'9', '_', '-', '.' => {},
        else => return bad("unsafe name '{s}'", .{s}),
    };
    return s;
}

fn isSha256Hex(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| switch (c) {
        '0'...'9', 'a'...'f' => {},
        else => return false,
    };
    return true;
}

fn loadFile(arena: Allocator, io: std.Io, dir: std.Io.Dir, v: std.json.Value, what: []const u8, size: usize) !File {
    const o = try objectOf(v, what);
    const name = try safeName(try textOf(o, "file"));
    const sha = try textOf(o, "sha256");
    if (!isSha256Hex(sha)) return bad("{s}: sha256 is not 64 lowercase hex digits", .{what});
    const bytes = dir.readFileAllocOptions(io, name, arena, .limited(max_file), .@"16", null) catch |err|
        return bad("{s}: {s} unreadable: {s}", .{ what, name, @errorName(err) });
    if (bytes.len != size) return bad("{s}: {s} has {d} bytes, expected {d}", .{ what, name, bytes.len, size });
    const got = sha256Hex(bytes);
    if (!std.mem.eql(u8, &got, sha)) return bad("{s}: {s} sha256 {s} is not the manifest's {s}", .{ what, name, got[0..], sha });
    return .{ .name = name, .sha = sha, .bytes = bytes };
}

fn findName(comptime T: type, items: []const T, name: []const u8, what: []const u8) BadCases!usize {
    for (items, 0..) |item, i| {
        if (std.mem.eql(u8, item.name, name)) return i;
    }
    return bad("unknown {s} '{s}'", .{ what, name });
}

fn workspaceCapacity(max_rows: usize) usize {
    return @max(chunk, max_rows / chunk * chunk);
}

fn parseCases(arena: Allocator, io: std.Io, dir: std.Io.Dir) !Cases {
    const text = dir.readFileAlloc(io, "cases.json", arena, .limited(1 << 24)) catch |err| return bad("cases.json unreadable: {s}", .{@errorName(err)});
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch |err| return bad("cases.json is not JSON: {s}", .{@errorName(err)});
    const root = try objectOf(doc, "cases.json");
    if (!std.mem.eql(u8, try textOf(root, "schema"), cases_schema)) return bad("schema is not {s}", .{cases_schema});

    const gs = try arrayOf(root, "geometries");
    const geometries = try arena.alloc(Geometry, gs.len);
    for (gs, geometries) |v, *g| {
        const o = try objectOf(v, "geometry");
        const nk = try uintOf(o, "nk");
        const nv = try uintOf(o, "nv");
        if (nk == 0 or nv == 0 or nv % nk != 0) return bad("geometry nk {d} nv {d}: nv must be a positive multiple of nk", .{ nk, nv });
        if ((try uintOf(o, "dv")) != dv) return bad("geometry dv must be {d}", .{dv});
        g.* = .{
            .name = try safeName(try textOf(o, "name")),
            .nk = nk,
            .nv = nv,
            .a_log = try loadFile(arena, io, dir, try get(o, "a_log"), "a_log", nv * 4),
            .dt_bias = try loadFile(arena, io, dir, try get(o, "dt_bias"), "dt_bias", nv * 4),
        };
    }
    for (geometries, 0..) |g, i| for (geometries[0..i]) |h| if (std.mem.eql(u8, g.name, h.name)) return bad("duplicate geometry {s}", .{g.name});

    const ss = try arrayOf(root, "streams");
    const streams = try arena.alloc(Stream, ss.len);
    for (ss, streams) |v, *s| {
        const o = try objectOf(v, "stream");
        const gi = try findName(Geometry, geometries, try textOf(o, "geometry"), "geometry");
        const g = geometries[gi];
        const rows = try uintOf(o, "rows");
        const ld_qkv = try uintOf(o, "ld_qkv");
        const ld_ba = try uintOf(o, "ld_ba");
        if (rows == 0) return bad("stream with zero rows", .{});
        if (ld_qkv < g.width() or ld_ba < 2 * g.nv) return bad("stream strides {d}/{d} below the row widths {d}/{d}", .{ ld_qkv, ld_ba, g.width(), 2 * g.nv });
        if (ld_qkv % 8 != 0 or ld_ba % 8 != 0) return bad("stream strides must keep 16-byte rows", .{});
        s.* = .{
            .name = try safeName(try textOf(o, "name")),
            .geometry = gi,
            .rows = rows,
            .ld_qkv = ld_qkv,
            .ld_ba = ld_ba,
            .qkv = try loadFile(arena, io, dir, try get(o, "qkv"), "qkv", rows * ld_qkv * 2),
            .ba = try loadFile(arena, io, dir, try get(o, "ba"), "ba", rows * ld_ba * 2),
        };
    }

    const is = try arrayOf(root, "states");
    const inits = try arena.alloc(Init, is.len);
    for (is, inits) |v, *s| {
        const o = try objectOf(v, "state");
        const gi = try findName(Geometry, geometries, try textOf(o, "geometry"), "geometry");
        s.* = .{
            .name = try safeName(try textOf(o, "name")),
            .geometry = gi,
            .file = try loadFile(arena, io, dir, try get(o, "data"), "state", geometries[gi].stateBytes()),
        };
    }

    const cs = try arrayOf(root, "cases");
    const cases = try arena.alloc(Case, cs.len);
    for (cs, cases) |v, *c| {
        const o = try objectOf(v, "case");
        const name = try safeName(try textOf(o, "name"));
        const gi = try findName(Geometry, geometries, try textOf(o, "geometry"), "geometry");
        const max_rows = try uintOf(o, "max_rows");
        if (max_rows == 0) return bad("{s}: max_rows 0", .{name});
        const init_list = try arrayOf(o, "init");
        if (init_list.len != layer_count) return bad("{s}: init must name {d} layer states", .{ name, layer_count });
        var init: [layer_count]?usize = undefined;
        for (init_list, &init) |iv, *slot| slot.* = try initRef(inits, iv, gi, name);
        const ov = try arrayOf(o, "ops");
        if (ov.len == 0) return bad("{s}: no ops", .{name});
        const ops = try arena.alloc(Op, ov.len);
        var last: [layer_count]?usize = @splat(null);
        for (ov, ops, 0..) |item, *op, oi| {
            const p = try objectOf(item, "op");
            const kind = std.meta.stringToEnum(OpKind, try textOf(p, "op")) orelse return bad("{s} op {d}: unknown op", .{ name, oi });
            const layer = try uintOf(p, "layer");
            if (layer >= layer_count) return bad("{s} op {d}: layer {d}", .{ name, oi, layer });
            op.* = .{ .kind = kind, .layer = layer };
            switch (kind) {
                .reset => {
                    op.init = try initRef(inits, try get(p, "state"), gi, name);
                    op.poison_cache = try boolOr(p, "poison_cache", false);
                },
                .prefill, .decode => {
                    op.stream = try findName(Stream, streams, try textOf(p, "stream"), "stream");
                    op.start = try uintOf(p, "start");
                    op.rows = try uintOf(p, "rows");
                    op.probe = try boolOr(p, "probe", false);
                    const s = streams[op.stream];
                    if (s.geometry != gi) return bad("{s} op {d}: stream {s} has another geometry", .{ name, oi, s.name });
                    if (op.rows == 0 or op.rows > max_rows) return bad("{s} op {d}: rows {d} outside 1..max_rows {d}", .{ name, oi, op.rows, max_rows });
                    if (op.start + op.rows > s.rows) return bad("{s} op {d}: rows {d}+{d} beyond stream {s}", .{ name, oi, op.start, op.rows, s.name });
                    if (op.probe and kind != .prefill) return bad("{s} op {d}: probe on a decode", .{ name, oi });
                    last[layer] = oi;
                },
            }
        }
        c.* = .{
            .name = name,
            .geometry = gi,
            .max_rows = max_rows,
            .init = init,
            .ops = ops,
            .save_old_states = try boolOr(o, "save_old_states", false),
            .last_op = last,
        };
        const g = &geometries[gi];
        g.used = true;
        g.max_rows = @max(g.max_rows, max_rows);
        _, const ws_bytes = gdn.Workspace.layout(max_rows, g.nk, g.nv);
        g.ws_bytes = @max(g.ws_bytes, ws_bytes);
    }
    for (cases, 0..) |c, i| for (cases[0..i]) |d| if (std.mem.eql(u8, c.name, d.name)) return bad("duplicate case {s}", .{c.name});
    return .{ .sha = sha256Hex(text), .geometries = geometries, .streams = streams, .inits = inits, .cases = cases };
}

/// A state reference: the string "zero" or the name of a state of geometry `gi`.
fn initRef(inits: []const Init, v: std.json.Value, gi: usize, case_name: []const u8) BadCases!?usize {
    if (v != .string) return bad("{s}: state reference is not a string", .{case_name});
    if (std.mem.eql(u8, v.string, "zero")) return null;
    const i = try findName(Init, inits, v.string, "state");
    if (inits[i].geometry != gi) return bad("{s}: state {s} has another geometry", .{ case_name, v.string });
    return i;
}

// ---------------------------------------------------------------------------------------------------------------
// recorded failures, hashing, JSON

const Errors = struct {
    gpa: Allocator,
    json: std.ArrayList(u8) = .empty,
    count: usize = 0,

    fn add(e: *Errors, comptime f: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(e.gpa, f, args);
        defer e.gpa.free(message);
        if (e.count < 200) std.debug.print("error: {s}\n", .{message});
        try e.json.appendSlice(e.gpa, if (e.count == 0) "\n    " else ",\n    ");
        try jsonString(&e.json, e.gpa, message);
        e.count += 1;
    }
};

fn sha256Hex(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

fn fileSha256(gpa: Allocator, io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const buffer = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buffer);
    var hash = Sha256.init(.{});
    var offset: u64 = 0;
    while (true) {
        const got = try file.readPositionalAll(io, buffer, offset);
        hash.update(buffer[0..got]);
        offset += got;
        if (got < buffer.len) break;
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

fn jsonString(list: *std.ArrayList(u8), gpa: Allocator, s: []const u8) !void {
    try list.append(gpa, '"');
    for (s) |c| switch (c) {
        '"', '\\' => {
            try list.append(gpa, '\\');
            try list.append(gpa, c);
        },
        0...0x1f => try list.print(gpa, "\\u{x:0>4}", .{c}),
        else => try list.append(gpa, c),
    };
    try list.append(gpa, '"');
}

fn nowNs(io: std.Io) i128 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn notFill(bytes: []const u8) usize {
    var n: usize = 0;
    for (bytes) |b| {
        if (b != fill) n += 1;
    }
    return n;
}

fn nonfiniteBf16(values: []const u16) usize {
    var n: usize = 0;
    for (values) |bits| {
        if ((bits & 0x7f80) == 0x7f80) n += 1;
    }
    return n;
}

fn unwrittenBf16(values: []const u16) usize {
    var n: usize = 0;
    for (values) |bits| {
        if (bits == 0xffff) n += 1;
    }
    return n;
}

fn nonfiniteF32(bytes: []align(16) const u8) usize {
    var n: usize = 0;
    for (std.mem.bytesAsSlice(u32, bytes)) |bits| {
        if ((bits & 0x7f800000) == 0x7f800000) n += 1;
    }
    return n;
}

fn bitsDifferent(comptime T: type, a: []align(16) const u8, b: []align(16) const u8) usize {
    var n: usize = 0;
    for (std.mem.bytesAsSlice(T, a), std.mem.bytesAsSlice(T, b)) |x, y| {
        if (x != y) n += 1;
    }
    return n;
}

fn int(v: usize) i32 {
    return @intCast(v);
}

fn sameBank(a: ?u1, b: ?u1) bool {
    if (a) |x| {
        if (b) |y| return x == y;
        return false;
    }
    return b == null;
}

// ---------------------------------------------------------------------------------------------------------------
// device

/// One layer's caller-owned controller memory with guards between every buffer, and its frozen-control states.
const LayerMem = struct {
    region: cuda.DeviceBuffer,
    old: cuda.DeviceBuffer, // old sequential chain state (frozen prefill/decode exports)
    state: gdn.State,
    bank_bytes: usize,
    qkv_cache_bytes: usize,
    ba_cache_bytes: usize,

    fn offsets(m: LayerMem) [4]usize {
        const b0 = guard;
        const b1 = b0 + m.bank_bytes + guard;
        const q = b1 + m.bank_bytes + guard;
        const a = q + m.qkv_cache_bytes + guard;
        return .{ b0, b1, q, a };
    }

    fn total(bank: usize, qkvc: usize, bac: usize) usize {
        return 5 * guard + 2 * bank + qkvc + bac;
    }

    /// [start, len) of the five guards.
    fn guards(m: LayerMem) [5][2]usize {
        const o = m.offsets();
        return .{ .{ 0, guard }, .{ o[0] + m.bank_bytes, guard }, .{ o[1] + m.bank_bytes, guard }, .{ o[2] + m.qkv_cache_bytes, guard }, .{ o[3] + m.ba_cache_bytes, guard } };
    }
};

const GeoMem = struct {
    a_log: cuda.DeviceBuffer,
    dt_bias: cuda.DeviceBuffer,
    layers: [layer_count]LayerMem,
    ws: cuda.DeviceBuffer, // guard | workspace (largest layout) | guard
    y: cuda.DeviceBuffer, // guard rows | max_rows | guard rows, bf16 rows of nv * 128
    y_old: cuda.DeviceBuffer,
    y_base: cuda.DeviceBuffer,
    base_state: cuda.DeviceBuffer,
};

fn view(d: *const cuda.Driver, ptr: u64, len: usize) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = ptr, .len = len, .borrowed = true };
}

const Recur = struct { qkv: u64, ld_qkv: usize, ba: u64, ld_ba: usize, a_log: u64, dt_bias: u64, state: u64, y: u64, rows: usize, nk: usize, nv: usize, scale: f32 };

/// The frozen image's exports with backbone-kernels-pre-chunk.zig's launch: grid (128 / 32, nv), 128 threads.
fn launchFrozen(f: cuda.Function, s: cuda.Stream, r: Recur) !void {
    var a: cuda.Args = .{};
    a.add(r.qkv);
    a.add(int(r.ld_qkv));
    a.add(r.ba);
    a.add(int(r.ld_ba));
    a.add(r.a_log);
    a.add(r.dt_bias);
    a.add(r.state);
    a.add(r.y);
    a.add(int(r.rows));
    a.add(int(r.nk));
    a.add(int(r.nv));
    a.add(r.scale);
    try cuda.launch.launch(f, .{ .grid = .{ .x = dv / 32, .y = @intCast(r.nv), .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

/// What one op observed in one pass.
const OpRecord = struct {
    pending_before: usize = 0,
    committed_before: u1 = 0,
    pending_after: usize = 0,
    committed_after: u1 = 0,
    current_bank: ?u1 = null,
    y_sha: [64]u8 = @splat('-'),
    state_sha: [64]u8 = @splat('-'),
    old_y_sha: [64]u8 = @splat('-'),
    old_state_sha: [64]u8 = @splat('-'),
    base_y_sha: [64]u8 = @splat('-'),
    base_state_sha: [64]u8 = @splat('-'),
    base_y_bits_different: usize = 0,
    base_state_bits_different: usize = 0,
    y_unwritten: usize = 0,
    y_nonfinite: usize = 0,
    y_guard_before_changed: usize = 0,
    y_guard_after_changed: usize = 0,
    state_nonfinite: usize = 0,
    ws_front_guard_changed: usize = 0,
    ws_tail_changed: usize = 0,
    layer_guards_changed: usize = 0,
    inputs_unchanged: bool = true,
    elapsed_ns: i128 = 0,
    probe_sha: [probe_fields.len][64]u8 = @splat(@splat('-')),
    probe_nonfinite: [probe_fields.len]usize = @splat(0),
    probe_in_layout: bool = false,

    fn sameOutputs(a: OpRecord, b: OpRecord) bool {
        return a.pending_after == b.pending_after and a.committed_after == b.committed_after and sameBank(a.current_bank, b.current_bank) and
            std.mem.eql(u8, &a.y_sha, &b.y_sha) and std.mem.eql(u8, &a.state_sha, &b.state_sha) and
            std.mem.eql(u8, &a.old_y_sha, &b.old_y_sha) and std.mem.eql(u8, &a.old_state_sha, &b.old_state_sha) and
            std.mem.eql(u8, &a.base_y_sha, &b.base_y_sha) and std.mem.eql(u8, &a.base_state_sha, &b.base_state_sha) and
            std.mem.eql(u8, std.mem.sliceAsBytes(&a.probe_sha), std.mem.sliceAsBytes(&b.probe_sha));
    }
};

const Runner = struct {
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const gdn.Kernels,
    frozen_decode_fn: cuda.Function,
    frozen_prefill_fn: cuda.Function,
    cs: *const Cases,
    stream_bufs: []const [2]cuda.DeviceBuffer, // per stream: qkv, ba
    out: std.Io.Dir,
    errors: *Errors,
    scale: f32,
    // host scratch, sized once for the largest geometry and stream
    scratch: []align(16) u8,
    scratch2: []align(16) u8,

    fn sync(r: *const Runner) !void {
        try r.stream.synchronize();
    }

    fn download(r: *const Runner, ptr: u64, len: usize, dst: []align(16) u8) ![]align(16) u8 {
        const buf = dst[0..len];
        try view(r.d, ptr, len).download(0, buf);
        return buf;
    }

    /// Zero (or NaN-poison) the layer's caches, zero both banks, resetMetadata, then a nonzero state into bank0.
    fn reset(r: *Runner, g: Geometry, m: *GeoMem, layer: usize, init: ?usize, poison_cache: bool, rec: *OpRecord) !void {
        const lm = &m.layers[layer];
        const o = lm.offsets();
        const base = lm.region.ptr;
        try view(r.d, base + o[0], lm.bank_bytes).fill8(0, r.stream.handle);
        try view(r.d, base + o[1], lm.bank_bytes).fill8(0, r.stream.handle);
        const cache_value: u8 = if (poison_cache) fill else 0;
        try view(r.d, base + o[2], lm.qkv_cache_bytes).fill8(cache_value, r.stream.handle);
        try view(r.d, base + o[3], lm.ba_cache_bytes).fill8(cache_value, r.stream.handle);
        try lm.old.fill8(0, r.stream.handle);
        rec.pending_before = lm.state.pending;
        rec.committed_before = lm.state.committed_bank;
        lm.state.resetMetadata();
        try r.sync();
        if (init) |i| {
            const bytes = r.cs.inits[i].file.bytes;
            try view(r.d, lm.state.banks[0], g.stateBytes()).upload(0, bytes);
            try lm.old.upload(0, bytes);
        }
        rec.pending_after = lm.state.pending;
        rec.committed_after = lm.state.committed_bank;
        rec.current_bank = bankOf(lm.state);
        if (lm.state.pending != 0 or lm.state.committed_bank != 0 or !sameBank(rec.current_bank, 0))
            try r.errors.add("layer {d}: after resetMetadata pending {d} committed_bank {d}, current() not bank0", .{ layer, lm.state.pending, lm.state.committed_bank });
        const cur = try r.download(lm.state.current(), g.stateBytes(), r.scratch);
        rec.state_sha = sha256Hex(cur);
        if (init) |i| {
            if (!std.mem.eql(u8, cur, r.cs.inits[i].file.bytes)) try r.errors.add("layer {d}: bank0 does not hold the uploaded initial state", .{layer});
        } else if (notZero(cur)) try r.errors.add("layer {d}: zeroed bank0 is not zero", .{layer});
    }

    fn notZero(bytes: []const u8) bool {
        for (bytes) |b| {
            if (b != 0) return true;
        }
        return false;
    }

    fn bankOf(s: gdn.State) ?u1 {
        const c = s.current();
        if (c == s.banks[0]) return 0;
        if (c == s.banks[1]) return 1;
        return null;
    }

    fn checkInputs(r: *Runner, g: Geometry, m: *GeoMem, op: Op) !bool {
        const s = r.cs.streams[op.stream];
        const bufs = r.stream_bufs[op.stream];
        var same = true;
        for ([_]struct { []align(16) const u8, cuda.DeviceBuffer }{ .{ s.qkv.bytes, bufs[0] }, .{ s.ba.bytes, bufs[1] }, .{ g.a_log.bytes, m.a_log }, .{ g.dt_bias.bytes, m.dt_bias } }) |pair| {
            const host, const dev = pair;
            const got = try r.download(dev.ptr, host.len, r.scratch2);
            if (!std.mem.eql(u8, got, host)) same = false;
        }
        return same;
    }

    fn checkLayerGuards(r: *Runner, m: *GeoMem) !usize {
        var changed: usize = 0;
        for (m.layers) |lm| {
            for (lm.guards()) |gr| changed += notFill(try r.download(lm.region.ptr + gr[0], gr[1], r.scratch2));
        }
        return changed;
    }

    /// Rows [0, n) of the guarded output buffer, after checking coverage and both guards.
    fn checkY(r: *Runner, g: Geometry, buf: cuda.DeviceBuffer, n: usize, rec: *OpRecord) ![]align(16) u8 {
        const rb = g.rowBytes();
        const all = try r.download(buf.ptr, buf.len, r.scratch2);
        rec.y_guard_before_changed = notFill(all[0 .. y_guard_rows * rb]);
        const rows: []align(16) u8 = @alignCast(all[y_guard_rows * rb ..][0 .. n * rb]);
        rec.y_guard_after_changed = notFill(all[(y_guard_rows + n) * rb ..]);
        const values = std.mem.bytesAsSlice(u16, rows);
        rec.y_unwritten = unwrittenBf16(values);
        rec.y_nonfinite = nonfiniteBf16(values);
        return rows;
    }

    fn runCase(r: *Runner, c: Case, m: *GeoMem, records: []OpRecord, pass: usize, ys: *[layer_count]std.ArrayList(u8), old_ys: *[layer_count]std.ArrayList(u8), base_ys: *[layer_count]std.ArrayList(u8)) !void {
        const g = r.cs.geometries[c.geometry];
        const ws_proto, const ws_bytes = gdn.Workspace.layout(c.max_rows, g.nk, g.nv);
        const ws_base = m.ws.ptr + guard;
        const ws = ws_proto.place(ws_base);
        const rb = g.rowBytes();
        const y_ptr = m.y.ptr + y_guard_rows * rb;
        var path_buffer: [256]u8 = undefined;

        // a pass starts from both layers' case initial states
        for (0..layer_count) |layer| {
            var rec: OpRecord = .{};
            try r.reset(g, m, layer, c.init[layer], false, &rec);
        }
        for (c.ops, records, 0..) |op, *rec, oi| {
            rec.* = .{};
            const lm = &m.layers[op.layer];
            if (op.kind == .reset) {
                try r.reset(g, m, op.layer, op.init, op.poison_cache, rec);
                rec.layer_guards_changed = try r.checkLayerGuards(m);
                continue;
            }
            const sb = r.stream_bufs[op.stream];
            const s = r.cs.streams[op.stream];
            const run: gdn.Run = .{
                .qkv = sb[0].ptr + op.start * s.ld_qkv * 2,
                .ld_qkv = s.ld_qkv,
                .ba = sb[1].ptr + op.start * s.ld_ba * 2,
                .ld_ba = s.ld_ba,
                .a_log = m.a_log.ptr,
                .dt_bias = m.dt_bias.ptr,
                .y = y_ptr,
                .rows = op.rows,
                .nk = g.nk,
                .nv = g.nv,
                .dv = dv,
                .scale = r.scale,
            };
            const frozen: Recur = .{ .qkv = run.qkv, .ld_qkv = run.ld_qkv, .ba = run.ba, .ld_ba = run.ld_ba, .a_log = run.a_log, .dt_bias = run.dt_bias, .state = lm.old.ptr, .y = m.y_old.ptr, .rows = op.rows, .nk = g.nk, .nv = g.nv, .scale = r.scale };
            rec.pending_before = lm.state.pending;
            rec.committed_before = lm.state.committed_bank;
            try m.y.fill8(fill, r.stream.handle);
            if (op.kind == .prefill) {
                try m.ws.fill8(fill, r.stream.handle);
                try r.sync();
                const t0 = nowNs(r.io);
                try r.kernels.prefill(r.stream, &lm.state, ws, run);
                try r.sync();
                rec.elapsed_ns = nowNs(r.io) - t0;
                rec.ws_front_guard_changed = notFill(try r.download(m.ws.ptr, guard, r.scratch2));
                rec.ws_tail_changed = notFill(try r.download(ws_base + ws_bytes, m.ws.len - guard - ws_bytes, r.scratch2));
                if (op.probe) try r.probe(c, g, ws, ws_base, ws_bytes, op, oi, rec, pass, &path_buffer);
            } else {
                // the unchanged frozen decode on a copy of the state decode must start from, the same rows
                try m.y_base.fill8(fill, r.stream.handle);
                try m.base_state.copyFrom(0, lm.state.current(), g.stateBytes(), r.stream.handle);
                try r.sync();
                const t0 = nowNs(r.io);
                try r.kernels.decode(r.stream, &lm.state, run);
                try r.sync();
                rec.elapsed_ns = nowNs(r.io) - t0;
                var base = frozen;
                base.state = m.base_state.ptr;
                base.y = m.y_base.ptr;
                try launchFrozen(r.frozen_decode_fn, r.stream, base);
                try r.sync();
            }
            rec.pending_after = lm.state.pending;
            rec.committed_after = lm.state.committed_bank;
            rec.current_bank = bankOf(lm.state);
            if (rec.current_bank == null) try r.errors.add("{s} op {d}: State.current() is neither bank", .{ c.name, oi });

            // the controller's outputs, then its state, before any control touches device memory again
            const y = try r.checkY(g, m.y, op.rows, rec);
            rec.y_sha = sha256Hex(y);
            if (pass == 0) try ys[op.layer].appendSlice(r.gpa, y);
            const y_copy = try r.gpa.alignedAlloc(u8, .@"16", y.len);
            defer r.gpa.free(y_copy);
            @memcpy(y_copy, y);
            const state = try r.download(lm.state.current(), g.stateBytes(), r.scratch);
            rec.state_sha = sha256Hex(state);
            rec.state_nonfinite = nonfiniteF32(state);
            if (pass == 0) {
                const name = try std.fmt.bufPrint(&path_buffer, "{s}/op-{d:0>4}.state.f32", .{ c.name, oi });
                try r.out.writeFile(r.io, .{ .sub_path = name, .data = state, .flags = .{ .exclusive = true } });
            }
            if (op.kind == .decode) {
                const by = try r.download(m.y_base.ptr, op.rows * rb, r.scratch2);
                rec.base_y_sha = sha256Hex(by);
                rec.base_y_bits_different = bitsDifferent(u16, by, y_copy);
                if (pass == 0) try base_ys[op.layer].appendSlice(r.gpa, by);
                const bs = try r.download(m.base_state.ptr, g.stateBytes(), r.scratch2);
                rec.base_state_sha = sha256Hex(bs);
                rec.base_state_bits_different = bitsDifferent(u32, bs, state);
                if (rec.base_y_bits_different != 0 or rec.base_state_bits_different != 0) {
                    try r.errors.add("{s} op {d}: decode differs from the frozen decode export on identical state and rows ({d} output, {d} state elements)", .{ c.name, oi, rec.base_y_bits_different, rec.base_state_bits_different });
                    if (pass == 0) {
                        const name = try std.fmt.bufPrint(&path_buffer, "{s}/op-{d:0>4}.decbase.state.f32", .{ c.name, oi });
                        try r.out.writeFile(r.io, .{ .sub_path = name, .data = bs, .flags = .{ .exclusive = true } });
                    }
                }
            }

            // the old sequential chain on its own state: frozen prefill export for prefill, frozen decode for decode
            try launchFrozen(if (op.kind == .prefill) r.frozen_prefill_fn else r.frozen_decode_fn, r.stream, frozen);
            try r.sync();
            const oy = try r.download(m.y_old.ptr, op.rows * rb, r.scratch2);
            rec.old_y_sha = sha256Hex(oy);
            if (pass == 0) try old_ys[op.layer].appendSlice(r.gpa, oy);
            const os = try r.download(lm.old.ptr, g.stateBytes(), r.scratch2);
            rec.old_state_sha = sha256Hex(os);
            if (pass == 0 and (c.save_old_states or c.last_op[op.layer] == oi)) {
                const name = try std.fmt.bufPrint(&path_buffer, "{s}/op-{d:0>4}.old.state.f32", .{ c.name, oi });
                try r.out.writeFile(r.io, .{ .sub_path = name, .data = os, .flags = .{ .exclusive = true } });
            }

            rec.inputs_unchanged = try r.checkInputs(g, m, op);
            rec.layer_guards_changed = try r.checkLayerGuards(m);
            var prefix_buffer: [192]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&prefix_buffer, "{s} pass {d} op {d} ({s} layer {d} rows {d})", .{ c.name, pass, oi, @tagName(op.kind), op.layer, op.rows });
            if (!rec.inputs_unchanged) try r.errors.add("{s}: inputs changed", .{prefix});
            if (rec.y_unwritten != 0 or rec.y_nonfinite != 0) try r.errors.add("{s}: output rows unwritten {d}, nonfinite {d}", .{ prefix, rec.y_unwritten, rec.y_nonfinite });
            if (rec.y_guard_before_changed != 0 or rec.y_guard_after_changed != 0) try r.errors.add("{s}: output overrun: {d} bytes before, {d} bytes after the rows changed", .{ prefix, rec.y_guard_before_changed, rec.y_guard_after_changed });
            if (rec.state_nonfinite != 0) try r.errors.add("{s}: {d} nonfinite state values", .{ prefix, rec.state_nonfinite });
            if (rec.ws_front_guard_changed != 0 or rec.ws_tail_changed != 0) try r.errors.add("{s}: workspace wrote outside its layout ({d} front-guard bytes, {d} bytes past layout {d})", .{ prefix, rec.ws_front_guard_changed, rec.ws_tail_changed, ws_bytes });
            if (rec.layer_guards_changed != 0) try r.errors.add("{s}: {d} bank/cache guard bytes changed", .{ prefix, rec.layer_guards_changed });
            if (rec.pending_after >= chunk) try r.errors.add("{s} pass {d} op {d}: pending {d} is not below {d}", .{ c.name, pass, oi, rec.pending_after, chunk });
        }
    }

    fn probe(r: *Runner, c: Case, g: Geometry, ws: gdn.Workspace, ws_base: u64, ws_bytes: usize, op: Op, oi: usize, rec: *OpRecord, pass: usize, path_buffer: *[256]u8) !void {
        if (comptime !probe_available) {
            if (pass == 0) try r.errors.add("{s} op {d}: gate probe requested but gdn.Workspace has no u64 fields g, gc and beta", .{ c.name, oi });
            return;
        } else {
            const bytes = op.rows * g.nv * 4;
            rec.probe_in_layout = true;
            inline for (probe_fields, 0..) |name, fi| {
                const ptr: u64 = @field(ws, name);
                if (ptr < ws_base or ptr + bytes > ws_base + ws_bytes) {
                    rec.probe_in_layout = false;
                    try r.errors.add("{s} op {d}: Workspace.{s} [{d} rows][{d}] f32 is not inside the placed layout", .{ c.name, oi, name, op.rows, g.nv });
                } else {
                    const got = try r.download(ptr, bytes, r.scratch2);
                    rec.probe_sha[fi] = sha256Hex(got);
                    rec.probe_nonfinite[fi] = nonfiniteF32(got);
                    if (pass == 0) {
                        const file = try std.fmt.bufPrint(path_buffer, "{s}/op-{d:0>4}.ws-{s}.f32", .{ c.name, oi, name });
                        try r.out.writeFile(r.io, .{ .sub_path = file, .data = got, .flags = .{ .exclusive = true } });
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------------------------

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 5) {
        std.debug.print("usage: tf-drama-gdn-canonical-oracle GLUE_FATBIN FROZEN_FATBIN CASES_DIR OUTPUT_DIR\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const started = nowNs(io);

    // 1. fixtures and images, CPU only
    const cases_dir = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer cases_dir.close(io);
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);
    const cs = parseCases(arena, io, cases_dir) catch |err| switch (err) {
        error.BadCases => return 2,
        else => return err,
    };
    const glue_image = try std.Io.Dir.cwd().readFileAllocOptions(io, args[1], arena, .limited(max_file), .@"16", null);
    const frozen_image = try std.Io.Dir.cwd().readFileAllocOptions(io, args[2], arena, .limited(max_file), .@"16", null);
    const glue_sha = sha256Hex(glue_image);
    const frozen_sha = sha256Hex(frozen_image);
    if (!std.mem.eql(u8, &frozen_sha, frozen_image_sha256)) {
        std.debug.print("FROZEN_FATBIN sha256 {s} is not the frozen image {s}\n", .{ frozen_sha[0..], frozen_image_sha256 });
        return 2;
    }
    if (std.mem.eql(u8, &glue_sha, frozen_image_sha256)) {
        std.debug.print("GLUE_FATBIN is the frozen pre-chunk image\n", .{});
        return 2;
    }

    var errors: Errors = .{ .gpa = gpa };
    defer errors.json.deinit(gpa);
    const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, dv))); // backbone.zig's deltaNet scale

    // 2. device memory, once
    _ = try cuda.setAllocationLimit(device_limit);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var name_buffer: [128]u8 = undefined;
    const device_name = try arena.dupe(u8, try ctx.name(&name_buffer));
    const capability = try ctx.capability();
    const driver_version = try driver.version();
    const usage_start = cuda.usage(true);

    var glue = try cuda.Module.load(&driver, glue_image);
    defer glue.unload();
    const kernels = try gdn.Kernels.init(&glue, false); // GLUE_FATBIN may be any image: plain launches
    var frozen = try cuda.Module.load(&driver, frozen_image);
    defer frozen.unload();
    const frozen_decode_fn = try frozen.function(frozen_decode);
    const frozen_prefill_fn = try frozen.function(frozen_prefill);

    const stream_bufs = try arena.alloc([2]cuda.DeviceBuffer, cs.streams.len);
    var streams_made: usize = 0;
    defer for (stream_bufs[0..streams_made]) |*p| {
        p[0].free();
        p[1].free();
    };
    var scratch_len: usize = 0;
    for (cs.streams, stream_bufs) |s, *p| {
        p[0] = try cuda.DeviceBuffer.fromHost(&driver, s.qkv.bytes);
        p[1] = cuda.DeviceBuffer.fromHost(&driver, s.ba.bytes) catch |e| {
            p[0].free();
            return e;
        };
        streams_made += 1;
        scratch_len = @max(scratch_len, s.qkv.bytes.len, s.ba.bytes.len);
    }

    const mems = try arena.alloc(?GeoMem, cs.geometries.len);
    @memset(mems, null);
    defer for (mems) |*slot| if (slot.*) |*m| {
        m.a_log.free();
        m.dt_bias.free();
        for (&m.layers) |*lm| {
            lm.region.free();
            lm.old.free();
        }
        m.ws.free();
        m.y.free();
        m.y_old.free();
        m.y_base.free();
        m.base_state.free();
        slot.* = null;
    };
    var geometry_json: std.ArrayList(u8) = .empty;
    defer geometry_json.deinit(gpa);
    for (cs.geometries, mems) |g, *slot| {
        if (!g.used) continue;
        const sb = g.stateBytes();
        const y_bytes = (g.max_rows + 2 * y_guard_rows) * g.rowBytes();
        var m: GeoMem = undefined;
        m.a_log = try cuda.DeviceBuffer.fromHost(&driver, g.a_log.bytes);
        m.dt_bias = try cuda.DeviceBuffer.fromHost(&driver, g.dt_bias.bytes);
        for (&m.layers) |*lm| {
            lm.bank_bytes = sb;
            lm.qkv_cache_bytes = g.qkvCacheBytes();
            lm.ba_cache_bytes = g.baCacheBytes();
            lm.region = try cuda.DeviceBuffer.alloc(&driver, LayerMem.total(sb, lm.qkv_cache_bytes, lm.ba_cache_bytes));
            try lm.region.fill8(fill, null);
            lm.old = try cuda.DeviceBuffer.alloc(&driver, sb);
            const o = lm.offsets();
            lm.state = .{ .banks = .{ lm.region.ptr + o[0], lm.region.ptr + o[1] }, .qkv_cache = lm.region.ptr + o[2], .ba_cache = lm.region.ptr + o[3] };
        }
        m.ws = try cuda.DeviceBuffer.alloc(&driver, guard + g.ws_bytes + guard);
        m.y = try cuda.DeviceBuffer.alloc(&driver, y_bytes);
        m.y_old = try cuda.DeviceBuffer.alloc(&driver, g.max_rows * g.rowBytes());
        m.y_base = try cuda.DeviceBuffer.alloc(&driver, g.max_rows * g.rowBytes());
        m.base_state = try cuda.DeviceBuffer.alloc(&driver, sb);
        slot.* = m;
        scratch_len = @max(scratch_len, y_bytes, sb, m.ws.len);
        try geometry_json.print(gpa, "{s}\n    {{\"name\": \"{s}\", \"nk\": {d}, \"nv\": {d}, \"dv\": {d}, \"width\": {d}, \"state_bytes\": {d}, \"qkv_cache_bytes\": {d}, \"ba_cache_bytes\": {d}, \"layer_region_bytes\": {d}, \"layers\": {d}, \"workspace_allocated_bytes\": {d}, \"workspace_largest_layout_bytes\": {d}, \"output_rows_allocated\": {d}, \"guard_bytes\": {d}, \"output_guard_rows\": {d}}}", .{
            if (geometry_json.items.len == 0) "" else ",", g.name, g.nk, g.nv, dv, g.width(), sb, g.qkvCacheBytes(), g.baCacheBytes(), m.layers[0].region.len, layer_count, m.ws.len, g.ws_bytes, g.max_rows, guard, y_guard_rows,
        });
    }
    try ctx.synchronize();
    const usage_after_setup = cuda.usage(false);
    const mem_after_setup = try ctx.memInfo();

    const scratch = try gpa.alignedAlloc(u8, .@"16", scratch_len);
    defer gpa.free(scratch);
    const scratch2 = try gpa.alignedAlloc(u8, .@"16", scratch_len);
    defer gpa.free(scratch2);

    var runner: Runner = .{
        .gpa = gpa,
        .io = io,
        .d = &driver,
        .stream = stream,
        .kernels = &kernels,
        .frozen_decode_fn = frozen_decode_fn,
        .frozen_prefill_fn = frozen_prefill_fn,
        .cs = &cs,
        .stream_bufs = stream_bufs,
        .out = out,
        .errors = &errors,
        .scale = scale,
        .scratch = scratch,
        .scratch2 = scratch2,
    };

    // 3. cases, two passes each
    var cases_json: std.ArrayList(u8) = .empty;
    defer cases_json.deinit(gpa);
    var path_buffer: [256]u8 = undefined;
    var max_ops: usize = 0;
    for (cs.cases) |c| max_ops = @max(max_ops, c.ops.len);
    const records = [passes][]OpRecord{ try gpa.alloc(OpRecord, max_ops), try gpa.alloc(OpRecord, max_ops) };
    defer for (records) |r| gpa.free(r);
    var total_ops: usize = 0;
    for (cs.cases) |c| {
        const m = &mems[c.geometry].?;
        const g = cs.geometries[c.geometry];
        try out.createDir(io, c.name, .default_dir);
        var ys: [layer_count]std.ArrayList(u8) = @splat(.empty);
        var old_ys: [layer_count]std.ArrayList(u8) = @splat(.empty);
        var base_ys: [layer_count]std.ArrayList(u8) = @splat(.empty);
        defer for (&ys, &old_ys, &base_ys) |*a, *b, *e| {
            a.deinit(gpa);
            b.deinit(gpa);
            e.deinit(gpa);
        };
        const case_start = nowNs(io);
        for (0..passes) |pass| try runner.runCase(c, m, records[pass][0..c.ops.len], pass, &ys, &old_ys, &base_ys);
        const case_ns = nowNs(io) - case_start;
        total_ops += c.ops.len;
        const usage_now = cuda.usage(false);
        if (usage_now.device != usage_after_setup.device or usage_now.host != usage_after_setup.host)
            try errors.add("{s}: counted allocations changed while running ({d} -> {d} device bytes)", .{ c.name, usage_after_setup.device, usage_now.device });

        // per layer concatenated outputs
        var files_json: std.ArrayList(u8) = .empty;
        defer files_json.deinit(gpa);
        for (0..layer_count) |layer| {
            for ([_]struct { []const u8, *std.ArrayList(u8) }{ .{ "y", &ys[layer] }, .{ "old.y", &old_ys[layer] }, .{ "decbase.y", &base_ys[layer] } }) |entry| {
                const kind, const list = entry;
                if (list.items.len == 0) continue;
                const name = try std.fmt.bufPrint(&path_buffer, "{s}/L{d}.{s}.bf16", .{ c.name, layer, kind });
                try out.writeFile(io, .{ .sub_path = name, .data = list.items, .flags = .{ .exclusive = true } });
                const file_sha = sha256Hex(list.items);
                try files_json.print(gpa, "{s}\"L{d}.{s}.bf16\": {{\"sha256\": \"{s}\", \"rows\": {d}}}", .{ if (files_json.items.len == 0) "" else ", ", layer, kind, file_sha[0..], list.items.len / g.rowBytes() });
            }
        }

        _, const ws_bytes = gdn.Workspace.layout(c.max_rows, g.nk, g.nv);
        try cases_json.print(gpa, "{s}\n    {{\"name\": \"{s}\", \"geometry\": \"{s}\", \"max_rows\": {d}, \"workspace_layout_bytes\": {d}, \"workspace_capacity_rows_contract\": {d}, \"init\": [", .{
            if (cases_json.items.len == 0) "" else ",", c.name, g.name, c.max_rows, ws_bytes, workspaceCapacity(c.max_rows),
        });
        for (c.init, 0..) |iv, i| try cases_json.print(gpa, "{s}\"{s}\"", .{ if (i == 0) "" else ", ", if (iv) |x| cs.inits[x].name else "zero" });
        try cases_json.print(gpa, "], \"wall_s\": {d:.6}, \"files\": {{{s}}}, \"ops\": [", .{ @as(f64, @floatFromInt(case_ns)) / 1e9, files_json.items });
        for (c.ops, 0..) |op, oi| {
            const a = records[0][oi];
            const b = records[1][oi];
            const repeat_equal = a.sameOutputs(b);
            if (!repeat_equal) try errors.add("{s} op {d}: pass 1 did not reproduce pass 0 (outputs, states, metadata or controls)", .{ c.name, oi });
            try cases_json.print(gpa, "{s}\n      {{\"index\": {d}, \"op\": \"{s}\", \"layer\": {d}", .{ if (oi == 0) "" else ",", oi, @tagName(op.kind), op.layer });
            if (op.kind == .reset) {
                try cases_json.print(gpa, ", \"state\": \"{s}\", \"poison_cache\": {}", .{ if (op.init) |x| cs.inits[x].name else "zero", op.poison_cache });
            } else {
                try cases_json.print(gpa, ", \"stream\": \"{s}\", \"start\": {d}, \"rows\": {d}, \"probe\": {}, \"y_sha256\": \"{s}\", \"old_y_sha256\": \"{s}\", \"old_state_sha256\": \"{s}\", \"y_unwritten\": {d}, \"y_nonfinite\": {d}, \"y_guard_before_changed\": {d}, \"y_guard_after_changed\": {d}, \"state_nonfinite\": {d}, \"ws_front_guard_changed\": {d}, \"ws_tail_changed\": {d}, \"inputs_unchanged\": {}, \"elapsed_s\": [{d:.9}, {d:.9}]", .{
                    cs.streams[op.stream].name, op.start, op.rows, op.probe, a.y_sha[0..], a.old_y_sha[0..], a.old_state_sha[0..], a.y_unwritten, a.y_nonfinite, a.y_guard_before_changed, a.y_guard_after_changed, a.state_nonfinite, a.ws_front_guard_changed, a.ws_tail_changed, a.inputs_unchanged and b.inputs_unchanged, @as(f64, @floatFromInt(a.elapsed_ns)) / 1e9, @as(f64, @floatFromInt(b.elapsed_ns)) / 1e9,
                });
                if (op.kind == .decode) try cases_json.print(gpa, ", \"decode_baseline\": {{\"y_sha256\": \"{s}\", \"state_sha256\": \"{s}\", \"y_bits_different\": {d}, \"state_bits_different\": {d}}}", .{ a.base_y_sha[0..], a.base_state_sha[0..], a.base_y_bits_different, a.base_state_bits_different });
                if (op.probe) {
                    try cases_json.print(gpa, ", \"gate_probe\": {{\"available\": {}, \"in_layout\": {}", .{ probe_available, a.probe_in_layout });
                    for (probe_fields, 0..) |f, fi| try cases_json.print(gpa, ", \"{s}\": {{\"sha256\": \"{s}\", \"nonfinite\": {d}}}", .{ f, a.probe_sha[fi][0..], a.probe_nonfinite[fi] });
                    try cases_json.appendSlice(gpa, "}");
                }
            }
            try cases_json.print(gpa, ", \"pending_before\": {d}, \"committed_bank_before\": {d}, \"pending_after\": {d}, \"committed_bank_after\": {d}, \"current_bank\": {s}, \"state_sha256\": \"{s}\", \"layer_guards_changed\": {d}, \"repeat_bit_exact\": {}}}", .{
                a.pending_before, a.committed_before, a.pending_after, a.committed_after, if (a.current_bank) |x| (if (x == 0) "0" else "1") else "null", a.state_sha[0..], a.layer_guards_changed, repeat_equal,
            });
        }
        try cases_json.appendSlice(gpa, "\n    ]}");
        std.debug.print("{s}: ops={d} wall_s={d:.3} errors_so_far={d}\n", .{ c.name, c.ops.len, @as(f64, @floatFromInt(case_ns)) / 1e9, errors.count });
    }
    const usage_end = cuda.usage(false);
    const mem_end = try ctx.memInfo();

    // 4. result.json
    var binary_json: std.ArrayList(u8) = .empty;
    defer binary_json.deinit(gpa);
    if (fileSha256(gpa, io, "/proc/self/exe")) |sha| {
        try binary_json.print(gpa, "\"{s}\"", .{sha[0..]});
    } else |err| {
        std.debug.print("binary hash unavailable: {s}\n", .{@errorName(err)});
        try binary_json.appendSlice(gpa, "null");
    }
    const source_sha = sha256Hex(@embedFile("drama_gdn_canonical_oracle.zig"));
    const evidence_valid = errors.count == 0;
    var mj: std.ArrayList(u8) = .empty;
    defer mj.deinit(gpa);
    try mj.print(gpa, "{{\n  \"schema\": \"{s}\",\n  \"producer\": \"{s}\",\n  \"glue_fatbin\": ", .{ result_schema, producer });
    try jsonString(&mj, gpa, args[1]);
    try mj.print(gpa, ",\n  \"glue_fatbin_sha256\": \"{s}\",\n  \"glue_fatbin_bytes\": {d},\n  \"frozen_fatbin\": ", .{ glue_sha[0..], glue_image.len });
    try jsonString(&mj, gpa, args[2]);
    try mj.print(gpa, ",\n  \"frozen_fatbin_sha256\": \"{s}\",\n  \"cases_dir\": ", .{frozen_sha[0..]});
    try jsonString(&mj, gpa, args[3]);
    try mj.print(gpa, ",\n  \"cases_sha256\": \"{s}\",\n  \"binary_sha256\": {s},\n  \"oracle_source_sha256\": \"{s}\",\n  \"device\": ", .{ cs.sha[0..], binary_json.items, source_sha[0..] });
    try jsonString(&mj, gpa, device_name);
    try mj.print(gpa, ",\n  \"capability\": {d},\n  \"driver_version\": {d},\n  \"scale_f32_bits\": {d},\n  \"chunk_rows\": {d},\n  \"passes\": {d},\n  \"layers\": {d}," ++
        "\n  \"gate_probe_fields_available\": {},\n  \"controller\": \"drama.backbone.gdn: State.resetMetadata/current, Workspace.layout/place, Kernels.init/prefill/decode; frozen controls: {s} (decode baseline, old decode) and {s} (old sequential prefill)\"," ++
        "\n  \"geometries\": [{s}\n  ],\n  \"cases\": [{s}\n  ]," ++
        "\n  \"allocations\": {{\"counted_device_bytes_start\": {d}, \"counted_device_bytes_after_setup\": {d}, \"counted_device_bytes_end\": {d}, \"counted_device_peak_bytes\": {d}, \"counted_pinned_host_bytes_end\": {d}, \"device_limit_bytes\": {d}, \"driver_free_bytes_after_setup\": {d}, \"driver_free_bytes_end\": {d}, \"driver_total_bytes\": {d}, \"host_scratch_bytes\": {d}}}," ++
        "\n  \"ops_per_pass\": {d},\n  \"wall_s\": {d:.6},\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        capability,               driver_version,   @as(u32, @bitCast(scale)), chunk,               passes,                                             layer_count,
        probe_available,          frozen_decode,    frozen_prefill,            geometry_json.items, cases_json.items,                                   usage_start.device,
        usage_after_setup.device, usage_end.device, usage_end.peak,            usage_end.host,      @as(u64, device_limit),                             mem_after_setup.free,
        mem_end.free,             mem_end.total,    2 * scratch_len,           total_ops,           @as(f64, @floatFromInt(nowNs(io) - started)) / 1e9, evidence_valid,
        errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "result.json", .data = mj.items, .flags = .{ .exclusive = true } });
    std.debug.print("cases={d} ops_per_pass={d} evidence_valid={} errors={d}\n", .{ cs.cases.len, total_ops, evidence_valid, errors.count });
    return if (evidence_valid) 0 else 3;
}
