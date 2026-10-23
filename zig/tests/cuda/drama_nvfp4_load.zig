//! A checkpoint through the production loader (Model.init) into replayed MoE layers, against its own files in float64.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");
const check = @import("check.zig");
const ref = @import("drama_nvfp4_ref.zig");

const Model = drama.backbone.Model;

/// `drama-nvfp4-load CHECKPOINT LAYERS [SOURCE]`: LAYERS like 0,1,2,39; SOURCE's weights are measured, not checked.
pub fn run(gpu: check.Gpu, dir: []const u8, list: []const u8, source: ?[]const u8) !void {
    const before = cuda.usage(false).device;
    // Model.init takes the context mutable; the runner's is only borrowed for this run
    var model = Model.init(gpu.gpa, gpu.io, @constCast(gpu.ctx), dir, "", .{ .context = 2 * ref.rows, .max_rows = ref.rows }) catch |err| {
        std.debug.print("RESULT drama-nvfp4-load refused {s}: {t}; device bytes held after the refusal {d} (before {d})\n", .{ dir, err, cuda.usage(false).device, before });
        return err;
    };
    defer model.deinit();
    std.debug.print("RESULT drama-nvfp4-load: {s} loaded, routed experts NVFP4 in {d} of {d} layers, {d} device bytes\n", .{ dir, model.w.nvfp4Layers(), model.w.layers.len, cuda.usage(false).device - before });
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var io = try ref.Pinned.init(gpu.d);
    defer io.deinit();
    var it = std.mem.splitScalar(u8, list, ',');
    var failed: usize = 0;
    while (it.next()) |field| {
        const layer = try std.fmt.parseInt(usize, field, 10);
        if (layer >= model.w.layers.len) return error.Usage;
        var layer_arena: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer layer_arena.deinit();
        const a = layer_arena.allocator();
        const own = try ref.Synth.read(a, gpu.io, dir, layer);
        const other = if (source) |src| try ref.Synth.read(a, gpu.io, src, layer) else null;
        var prng = std.Random.DefaultPrng.init(0x10AD_0000 + layer);
        const x = try ref.bfs(a, prng.random(), ref.rows * ref.H, 1.0);
        var name: [64]u8 = undefined;
        const what = try std.fmt.bufPrint(&name, "layer {d} ({t})", .{ layer, std.meta.activeTag(own.experts) });
        ref.scenario(gpu, a, &model, s, layer, own, other, x, what, &io, false) catch |err| {
            std.debug.print("FAIL {s}: {t}\n", .{ what, err });
            failed += 1;
        };
    }
    try check.expect(failed == 0, "{d} layers failed", .{failed});
}
