//! tf-drama-quant --encoded on the synthetic checkpoint: a taken layer's bytes, and the files it refuses.
const std = @import("std");
const core = @import("core");
const convert = @import("quant_convert.zig");
const t = @import("quant_convert_test.zig");

const st = core.safetensors;
const shard_edit = core.shard_edit;
const nvfp4 = core.quant.nvfp4;
const testing = std.testing;

const Fault = enum { none, missing, extra, dtype, shape, nan_scale, negative_scale, zero_global, nan_global };

/// `sub`/layer-01.safetensors: random C1 tensors of every expert of layer 1, expert 0's gate_proj spoiled by `fault`.
fn writeLayer(f: *t.Fixture, sub: []const u8, fault: Fault) !void {
    const arena = f.arena_state.allocator();
    try f.tmp.dir.createDir(testing.io, sub, .default_dir);
    var prng = std.Random.DefaultPrng.init(13);
    const rnd = prng.random();
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    for (0..t.experts) |e| for (t.projs) |p| {
        const n: usize, const k: usize = if (p[0] == 'd') .{ t.hidden, t.inner } else .{ t.inner, t.hidden };
        const base = try std.fmt.allocPrint(arena, "model.language_model.layers.1.mlp.experts.{d}.{s}.", .{ e, p });
        const spoil = e == 0 and p[0] == 'g';
        const codes = try arena.alloc(u8, n * k / 2);
        rnd.bytes(codes);
        const scales = try arena.alloc(u8, n * k / 16);
        for (scales) |*x| x.* = rnd.uintLessThan(u8, 0x7F);
        const global = try arena.create(f32);
        global.* = @floatCast(1e-3 + rnd.float(f64) * 1e-3);
        if (spoil) switch (fault) {
            .nan_scale => scales[5] = 0x7F,
            .negative_scale => scales[5] = 0x80,
            .zero_global => global.* = 0,
            .nan_global => global.* = std.math.nan(f32),
            else => {},
        };
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}weight", .{base}), .dtype = .u8, .shape = if (spoil and fault == .shape) try arena.dupe(usize, &.{ n / 2, k }) else try arena.dupe(usize, &.{ n, k / 2 }), .bytes = codes });
        try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}weight_scale", .{base}), .dtype = if (spoil and fault == .dtype) .u8 else .f8_e4m3, .shape = try arena.dupe(usize, &.{ n, k / 16 }), .bytes = scales });
        if (!(spoil and fault == .missing)) try tensors.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}weight_scale_2", .{base}), .dtype = .f32, .shape = &.{}, .bytes = std.mem.asBytes(global) });
    };
    if (fault == .extra) try tensors.append(arena, .{ .name = "model.language_model.layers.1.mlp.experts.0.gate_proj.input_scale", .dtype = .f32, .shape = &.{}, .bytes = std.mem.asBytes(&@as(f32, 1)) });
    const path = try std.fs.path.join(arena, &.{ try f.path(sub), "layer-01.safetensors" });
    try shard_edit.create(arena, path, &.{ .{ "tool", "drama_nvfp4_gptq.py" }, .{ "kappa", "256" } }, tensors.items);
}

test "an encoded layer goes into the checkpoint byte for byte and the other layers encode as without it" {
    var f = try t.Fixture.init();
    defer f.deinit();
    const arena = f.arena_state.allocator();
    const io = testing.io;
    try writeLayer(&f, "enc", .none);
    _ = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("plain"), .{ .threads = 2 });
    _ = try convert.convert(testing.allocator, io, try f.path("src"), try f.path("taken"), .{ .threads = 2, .encoded = try f.path("enc") });
    var enc = try st.File.open(testing.allocator, io, try f.path("enc/layer-01.safetensors"));
    defer enc.close(io);
    var from_file: usize = 0;
    for (0..2) |i| {
        var a = try f.open("plain", i);
        defer a.close(io);
        var b = try f.open("taken", i);
        defer b.close(io);
        const names = try t.ordered(arena, &b);
        try testing.expectEqualDeep(try t.ordered(arena, &a), names);
        for (names) |name| {
            const got = b.get(name).?;
            const want = if (std.mem.startsWith(u8, name, "model.language_model.layers.1.mlp.experts.")) enc.get(name).? else a.get(name).?;
            from_file += @intFromBool(std.mem.startsWith(u8, name, "model.language_model.layers.1.mlp.experts."));
            try testing.expect(got.is(want.dtype, want.shape[0..want.rank]) and std.mem.eql(u8, want.bytes, got.bytes));
        }
    }
    try testing.expectEqual(enc.names.count(), from_file);
    const prov = (try f.json("taken/quant-provenance.json")).object.get("experts").?.object;
    const sources = prov.get("layer_sources").?.array.items;
    try testing.expectEqualStrings("rtn", sources[0].object.get("source").?.string);
    const one = sources[1].object;
    try testing.expectEqualStrings("encoded", one.get("source").?.string);
    try testing.expectEqualStrings("drama_nvfp4_gptq.py", one.get("metadata").?.object.get("tool").?.string);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(try f.read("enc/layer-01.safetensors"), &digest, .{});
    try testing.expectEqualStrings(&std.fmt.bytesToHex(digest, .lower), one.get("sha256").?.string);
    // a taken tensor's errors are measured against the FP8 source, as an encoded one's are
    var src = try f.open("src", 0);
    defer src.close(io);
    const name = "model.language_model.layers.1.mlp.experts.0.down_proj.weight";
    const w = src.get(name).?;
    const h = try arena.alloc(f64, t.inner);
    @memset(h, 1);
    const g: f32 = @bitCast(std.mem.readInt(u32, enc.get(name ++ "_scale_2").?.bytes[0..4], .little));
    const sum = nvfp4.measure(try t.dequant(arena, w, src.get(name ++ "_scale_inv").?), h, t.hidden, t.inner, enc.get(name).?.bytes, enc.get(name ++ "_scale").?.bytes, g);
    for (prov.get("tensors").?.array.items) |x| if (std.mem.eql(u8, x.object.get("name").?.string, name)) {
        try testing.expectEqual(sum.rel().?, x.object.get("rel").?.float);
    };
}

test "malformed, misnamed, kept or expertless encoded layers are refused before anything is written" {
    var f = try t.Fixture.init();
    defer f.deinit();
    const io = testing.io;
    const gpa = testing.allocator;
    const cases = [_]struct { Fault, anyerror }{
        .{ .missing, error.EncodedTensorCount },          .{ .extra, error.EncodedTensorCount },
        .{ .dtype, error.EncodedTensorMismatch },         .{ .shape, error.EncodedTensorMismatch },
        .{ .nan_scale, error.BadEncodedScale },           .{ .negative_scale, error.BadEncodedScale },
        .{ .zero_global, error.BadEncodedGlobal },        .{ .nan_global, error.BadEncodedGlobal },
    };
    for (cases) |c| {
        try writeLayer(&f, @tagName(c[0]), c[0]);
        try testing.expectError(c[1], convert.convert(gpa, io, try f.path("src"), try f.path("out"), .{ .encoded = try f.path(@tagName(c[0])) }));
    }
    try writeLayer(&f, "none", .none);
    try testing.expectError(error.EncodedLayerKept, convert.convert(gpa, io, try f.path("src"), try f.path("out"), .{ .encoded = try f.path("none"), .keep = &.{1} }));
    try f.tmp.dir.createDir(io, "misnamed", .default_dir);
    try f.tmp.dir.writeFile(io, .{ .sub_path = "misnamed/layer-1.safetensors", .data = "x" });
    try testing.expectError(error.BadEncodedName, convert.convert(gpa, io, try f.path("src"), try f.path("out"), .{ .encoded = try f.path("misnamed") }));
    try f.tmp.dir.createDir(io, "expertless", .default_dir);
    try f.tmp.dir.writeFile(io, .{ .sub_path = "expertless/layer-07.safetensors", .data = "x" });
    try testing.expectError(error.EncodedLayerHasNoExperts, convert.convert(gpa, io, try f.path("src"), try f.path("out"), .{ .encoded = try f.path("expertless") }));
    try testing.expectError(error.FileNotFound, f.tmp.dir.access(io, "out", .{}));
}
