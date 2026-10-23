//! drama_topk with no mask, an all-kept mask, a base image's router, -inf logits, a mask and a compact slot table.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const experts = 256;
const top = 8;
const ld = 272; // the router's columns sit inside a wider row, as they do in moe_in
const rows = 48;
const neg_inf: u16 = 0xff80; // bf16 -inf

fn bf16(value: f32) u16 {
    return @truncate(@as(u32, @bitCast(value)) >> 16);
}

/// Integer-eighth logits, exactly BF16; rows 40 on tie often; in row 47 every expert but 200 underflows to p = 0.
fn logits() [rows * ld]u16 {
    var x: [rows * ld]u16 = @splat(bf16(-1000));
    for (0..rows) |r| for (0..experts) |e| {
        const k: i32 = @intCast((e * 37 + r * 101 + (e * e) % 13) % 129);
        const v: f32 = if (r >= 40) @floatFromInt(@divTrunc(k, 16)) else @as(f32, @floatFromInt(k - 64)) / 8;
        x[r * ld + e] = bf16(v);
    };
    for (0..experts) |e| x[(rows - 1) * ld + e] = bf16(if (e == 200) 100 else -100);
    return x;
}

/// Layer 7 of a drama3-small-like mask: about 100 of 256 kept, expert 200 among them.
fn mask() [experts / 32]u32 {
    var m: [experts / 32]u32 = @splat(0);
    for (0..experts) |e| if ((e * 29 + 7) % 5 < 2 or e == 200) {
        m[e / 32] |= @as(u32, 1) << @intCast(e % 32);
    };
    return m;
}

fn kept(m: [experts / 32]u32, e: usize) bool {
    return m[e / 32] >> @intCast(e % 32) & 1 == 1;
}

const Out = struct { ids: [rows * top]i32, wts: [rows * top]f32, slots: [rows * top]i32 };

/// The kernel's picks for host logits `x`; `keep`, a slot `table` uploaded when given; `base`: a base image's 5 args.
fn route(gpu: check.Gpu, stream: cuda.Stream, kernel: cuda.Function, x: []const u16, keep: ?[experts / 32]u32, table: ?[experts]i32, base: bool) !Out {
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dk = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&(keep orelse @as([experts / 32]u32, @splat(0)))));
    defer dk.free();
    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&(table orelse @as([experts]i32, @splat(-1)))));
    defer dt.free();
    var ids = try cuda.DeviceBuffer.alloc(gpu.d, rows * top * 4);
    defer ids.free();
    var wts = try cuda.DeviceBuffer.alloc(gpu.d, rows * top * 4);
    defer wts.free();
    var slots = try cuda.DeviceBuffer.alloc(gpu.d, rows * top * 4);
    defer slots.free();
    try gpu.ctx.synchronize();
    var args: cuda.Args = .{};
    args.add(dx.ptr);
    args.add(@as(i32, ld));
    args.add(ids.ptr);
    args.add(wts.ptr);
    args.add(@as(i32, top));
    if (!base) {
        args.add(if (keep != null) dk.ptr else @as(u64, 0));
        args.add(if (table != null) dt.ptr else @as(u64, 0));
        args.add(slots.ptr);
    }
    try cuda.launch.launch(kernel, .{ .grid = .{ .x = rows }, .block = .{ .x = 256 } }, stream, &args);
    try stream.synchronize();
    var out: Out = undefined;
    try ids.download(0, std.mem.asBytes(&out.ids));
    try wts.download(0, std.mem.asBytes(&out.wts));
    try slots.download(0, std.mem.asBytes(&out.slots));
    return out;
}

fn same(what: []const u8, a: Out, b: Out) !void {
    try check.sameBytes(what, std.mem.asBytes(&a.ids), std.mem.asBytes(&b.ids));
    try check.sameBytes(what, std.mem.asBytes(&a.wts), std.mem.asBytes(&b.wts));
}

pub fn run(gpu: check.Gpu, base_image: ?[]const u8) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.drama_backbone);
    defer module.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const kernel = try module.function("drama_topk");
    const x = logits();
    const all: [experts / 32]u32 = @splat(std.math.maxInt(u32));
    const plain = try route(gpu, stream, kernel, &x, null, null, false);
    try same("an all-kept mask against no mask", try route(gpu, stream, kernel, &x, all, null, false), plain);
    check.pass("drama_topk: no mask and an all-kept mask route {d} rows to the same bytes", .{rows});
    if (base_image) |path| {
        const image = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(64 << 20));
        defer gpu.gpa.free(image);
        var base = try cuda.Module.load(gpu.d, image);
        defer base.unload();
        try same("no mask against the base image's router", plain, try route(gpu, stream, try base.function("drama_topk"), &x, null, null, true));
        check.pass("drama_topk: no mask routes {d} rows to the base image's bytes ({s})", .{ rows, path });
    }
    const m = mask();
    const masked = try route(gpu, stream, kernel, &x, m, null, false);
    for (masked.ids) |id| try check.expect(kept(m, @intCast(id)), "a masked expert {d} was picked", .{id});
    // -inf logits route the same, but for the underflowing last row, where they tie masked experts with kept ones
    var inf = x;
    for (0..rows) |r| for (0..experts) |e| if (!kept(m, e)) {
        inf[r * ld + e] = neg_inf;
    };
    const by_inf = try route(gpu, stream, kernel, &inf, null, null, false);
    const n = (rows - 1) * top;
    try check.sameBytes("a mask against -inf logits, ids", std.mem.sliceAsBytes(masked.ids[0..n]), std.mem.sliceAsBytes(by_inf.ids[0..n]));
    try check.sameBytes("a mask against -inf logits, weights", std.mem.sliceAsBytes(masked.wts[0..n]), std.mem.sliceAsBytes(by_inf.wts[0..n]));
    try check.expect(masked.ids[n] == 200 and masked.wts[n] == 1, "the underflowing row: expert {d} at {d}", .{ masked.ids[n], masked.wts[n] });
    var count: usize = 0;
    for (0..experts) |e| count += @intFromBool(kept(m, e));
    check.pass("drama_topk: a mask of {d} experts routes as -inf logits do, and never to a masked expert", .{count});
    // a compact layer holding the kept experts: its slot table leaves ids and weights as they are, and slots each pick
    var table: [experts]i32 = @splat(-1);
    var held: i32 = 0;
    for (0..experts) |e| if (kept(m, e)) {
        table[e] = held;
        held += 1;
    };
    const slotted = try route(gpu, stream, kernel, &x, m, table, false);
    try same("a mask with a slot table against the mask alone", slotted, masked);
    for (slotted.ids, slotted.slots) |id, s| try check.expect(s == table[@intCast(id)], "expert {d} slotted {d}, not {d}", .{ id, s, table[@intCast(id)] });
    check.pass("drama_topk: a slot table of {d} held experts routes as its mask does and slots each pick", .{held});
}
