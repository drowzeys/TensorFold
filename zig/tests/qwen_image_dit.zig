//! Qwen-Image-2.1's native transformer against a case from tools/zig/qwen_image_case.py: the first velocity, then
//! every denoising step, compared with the Python family's rows and timed; the final latents are written raw.
const std = @import("std");
const mtl = @import("metal");
const qi = @import("qwen_image");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;

const Match = struct { cosine: f64, relative: f64, worst_row: f64 };

/// Cosine and relative error over everything, and the largest relative error of one row of `width`.
fn match(ours: []const f32, theirs: []const f32, width: usize) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var dd: f64 = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |a, b| {
            dot += @as(f64, a) * b;
            aa += @as(f64, a) * a;
            bb += @as(f64, b) * b;
            rd += (@as(f64, a) - b) * (@as(f64, a) - b);
            rb += @as(f64, b) * b;
        }
        dd += rd;
        worst = @max(worst, @sqrt(rd / @max(rb, 1e-30)));
    }
    return .{ .cosine = dot / @sqrt(aa * bb), .relative = @sqrt(dd / bb), .worst_row = worst };
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4 and args.len != 5) {
        std.debug.print("usage: tf-qwen-image-dit weights.safetensors case.safetensors latents.out [float | int8-float-attention]\n", .{});
        return error.BadArguments;
    }
    const gpa = init.gpa;
    const device = try mtl.Device.init();
    var case = try qi.Tensors.open(gpa, device, args[2]);
    defer case.deinit();
    const sig_e = case.names.get("sigmas") orelse return error.MissingTensor;
    const lat_e = case.names.get("latents") orelse return error.MissingTensor;
    const key_e = case.names.get("text_k.0") orelse return error.MissingTensor;
    const steps = sig_e.shape[0] - 1;
    const rows = lat_e.shape[0];
    const text = key_e.shape[2];
    const count = rows * qi.channels;
    const sigmas = try case.copy(f32, gpa, try case.at("sigmas", .f32, steps + 1), steps + 1);
    const start = try case.copy(f32, gpa, try case.at("latents", .f32, count), count);
    const expected = try case.copy(f32, gpa, try case.at("velocity", .f32, count), count);
    const final = try case.copy(f32, gpa, try case.at("final", .f32, count), count);

    const loading = mtl.clock.seconds();
    const int8 = args.len == 4 or !std.mem.eql(u8, args[4], "float");
    const model = try qi.Model.load(gpa, args[1], rows, text, int8);
    if (args.len == 5 and std.mem.eql(u8, args[4], "int8-float-attention")) model.attention8 = false;
    const half = rows * qi.head_dim / 2;
    model.setRotary(try case.copy(f32, gpa, try case.at("cos", .f32, half), half), try case.copy(f32, gpa, try case.at("sin", .f32, half), half));
    var name: [32]u8 = undefined;
    const prompt = qi.heads * qi.head_dim * text;
    for (0..qi.layers) |i| {
        const k = try case.at(try std.fmt.bufPrint(&name, "text_k.{d}", .{i}), .bf16, prompt);
        const v = try case.at(try std.fmt.bufPrint(&name, "text_v.{d}", .{i}), .bf16, prompt);
        const keys = try case.copy(u16, gpa, k, prompt);
        defer gpa.free(keys);
        const vals = try case.copy(u16, gpa, v, prompt);
        defer gpa.free(vals);
        model.setPrompt(i, keys, vals);
    }
    std.debug.print("{s}: {d} image rows after {d} prompt rows, {d} steps, {s} projections, loaded in {d:.2} s\n", .{ device.name(), rows, text, steps, if (int8) "int8" else "float", mtl.clock.seconds() - loading });

    const x = try gpa.alloc(f32, count);
    defer gpa.free(x);
    const velocity = try gpa.alloc(f32, count);
    defer gpa.free(velocity);
    @memcpy(x, start);
    for (0..steps) |step| {
        const took = try model.forward(x, sigmas[step], velocity);
        if (step == 0) {
            const m = match(velocity, expected, qi.channels);
            std.debug.print("first velocity against the reference: cosine {d:.6}, relative error {d:.5}, worst row {d:.5}\n", .{ m.cosine, m.relative, m.worst_row });
        }
        const by = sigmas[step + 1] - sigmas[step];
        for (x, velocity) |*o, v| o.* += v * by;
        std.debug.print("step {d}/{d}: {d:.3} s wall, {d:.3} s on the GPU\n", .{ step + 1, steps, took.wall, took.gpu });
    }
    const m = match(x, final, qi.channels);
    std.debug.print("final latents against the reference: cosine {d:.6}, relative error {d:.5}, worst row {d:.5}\n", .{ m.cosine, m.relative, m.worst_row });
    model.profile = true;
    _ = try model.forward(start, sigmas[0], velocity);
    std.debug.print("one forward by stage, each in its own command buffer:\n", .{});
    for (qi.stages, model.spent) |stage, seconds| std.debug.print("  {s}: {d:.3} s\n", .{ stage, seconds });
    const file = fopen(args[3], "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(x.ptr, 4, count, file) != count) return error.CannotWrite;
}
