//! The batched backbone GEMV against the one-row GEMV and the tiled GEMM, byte for byte, on random BF16 operands.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");

const Kernels = drama.backbone.kernels.Kernels;
const weights = drama.backbone.weights;
const config = drama.backbone.config;
const max_rows = drama.backbone.kernels.gemv_rows;
const sentinel: u16 = 0x7fc1;

/// The decode's products a batch of rows takes: (output columns, input width).
const shapes = [_][2]usize{
    .{ weights.gdn_in_rows, config.hidden },
    .{ config.hidden, config.value_dim },
    .{ weights.attn_in_rows, config.hidden },
    .{ weights.moe_in_rows, config.hidden },
    .{ config.hidden, config.shared_inner },
};

fn bf16s(random: std.Random, out: []u16) void {
    for (out) |*v| v.* = @truncate(@as(u32, @bitCast(random.float(f32) * 2 - 1)) >> 16);
}

fn case(gpu: check.Gpu, k: *const Kernels, stream: cuda.Stream, random: std.Random, n: usize, kdim: usize) !void {
    const ldx = kdim + 8; // rows apart by more than their width, as a batch's are not
    const x = try gpu.gpa.alloc(u16, max_rows * ldx);
    defer gpu.gpa.free(x);
    const w = try gpu.gpa.alloc(u16, n * kdim);
    defer gpu.gpa.free(w);
    bf16s(random, x);
    bf16s(random, w);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(w));
    defer dw.free();
    const fill = try gpu.gpa.alloc(u16, max_rows * n);
    defer gpu.gpa.free(fill);
    @memset(fill, sentinel);
    var batched = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(fill));
    defer batched.free();
    var single = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(fill));
    defer single.free();
    var tiled = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(fill));
    defer tiled.free();
    const got = try gpu.gpa.alloc(u16, max_rows * n);
    defer gpu.gpa.free(got);
    const want = try gpu.gpa.alloc(u16, max_rows * n);
    defer gpu.gpa.free(want);
    const also = try gpu.gpa.alloc(u16, max_rows * n);
    defer gpu.gpa.free(also);
    for (1..max_rows + 1) |m| {
        try batched.upload(0, std.mem.sliceAsBytes(fill));
        try tiled.upload(0, std.mem.sliceAsBytes(fill));
        try k.gemvRows(stream, dx.ptr, ldx, m, dw.ptr, n, kdim, batched.ptr, n);
        // one row at a time takes the one-row GEMV; the rows together, the tiled GEMM
        try k.gemm(stream, dx.ptr + (m - 1) * ldx * 2, ldx, 1, dw.ptr, n, kdim, single.ptr + (m - 1) * n * 2, n);
        try k.gemm(stream, dx.ptr, ldx, m, dw.ptr, n, kdim, tiled.ptr, n);
        try stream.synchronize();
        try batched.download(0, std.mem.sliceAsBytes(got));
        try single.download(0, std.mem.sliceAsBytes(want));
        try tiled.download(0, std.mem.sliceAsBytes(also));
        try check.sameBytes("batched GEMV rows against one-row GEMVs", std.mem.sliceAsBytes(got[0 .. m * n]), std.mem.sliceAsBytes(want[0 .. m * n]));
        if (m > 1) try check.sameBytes("batched GEMV rows against the tiled GEMM", std.mem.sliceAsBytes(got[0 .. m * n]), std.mem.sliceAsBytes(also[0 .. m * n]));
        for (got[m * n ..]) |v| try check.expect(v == sentinel, "batched GEMV of {d} rows wrote past them ({d} x {d})", .{ m, n, kdim });
    }
}

pub fn run(gpu: check.Gpu) !void {
    var k = try Kernels.load(gpu.gpa, gpu.io, gpu.d, "");
    defer k.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const early = if (k.gemvRows(stream, 0, 0, 1, 0, 8, 256, 0, 8)) |_| false else |e| e == error.DramaGemvRowsNotLoaded;
    try check.expect(early, "gemvRows ran before loadRows", .{});
    try k.loadRows();
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    for (shapes) |shape| try case(gpu, &k, stream, prng.random(), shape[0], shape[1]);
    check.pass("Drama batched GEMV: 1 to {d} rows equal one-row GEMVs and the tiled GEMM byte for byte at {d} decode shapes, nothing past them written", .{ max_rows, shapes.len });
}
