//! tf-drama-quant: Drama 3's routed experts to NVFP4 and dense projections to FP8, on the CPU (docs/quantization.md).
const std = @import("std");
const builtin = @import("builtin");
const drama = @import("drama");

const usage = "usage: tf-drama-quant SRC_CHECKPOINT DST_DIR [--experts nvfp4|keep] [--stats FILE] [--keep-fp8 LAYERS] [--encoded DIR]\n" ++
    "                      [--dense-fp8 CLASS[,CLASS]|all] [--keep-bf16 CLASS|b:L[,L]|t:L[,L][,...]] [--compact-experts MASK]\n";

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var paths: [2][]const u8 = undefined;
    var given: usize = 0;
    var options: drama.quant_convert.Options = .{ .commit = std.mem.trimEnd(u8, @embedFile("tool_commit"), "\n") };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--stats")) {
            i += 1;
            if (i == args.len) return fail();
            options.stats = args[i];
        } else if (std.mem.eql(u8, a, "--encoded")) {
            i += 1;
            if (i == args.len) return fail();
            options.encoded = args[i];
        } else if (std.mem.eql(u8, a, "--keep-fp8")) {
            i += 1;
            if (i == args.len) return fail();
            var keep: std.ArrayList(usize) = .empty;
            var it = std.mem.splitScalar(u8, args[i], ',');
            while (it.next()) |l| try keep.append(arena, std.fmt.parseInt(usize, l, 10) catch return fail());
            options.keep = keep.items;
        } else if (std.mem.eql(u8, a, "--experts")) {
            i += 1;
            if (i == args.len) return fail();
            options.experts = std.meta.stringToEnum(drama.quant_convert.Experts, args[i]) orelse return fail();
        } else if (std.mem.eql(u8, a, "--dense-fp8")) {
            i += 1;
            if (i == args.len) return fail();
            options.dense_fp8 = args[i];
        } else if (std.mem.eql(u8, a, "--keep-bf16")) {
            i += 1;
            if (i == args.len) return fail();
            options.keep_bf16 = args[i];
        } else if (std.mem.eql(u8, a, "--compact-experts")) {
            i += 1;
            if (i == args.len) return fail();
            options.compact_experts = args[i];
        } else if (given < paths.len and !std.mem.startsWith(u8, a, "--")) {
            paths[given] = a;
            given += 1;
        } else return fail();
    }
    if (given != paths.len) return fail();
    options.threads = std.Thread.getCpuCount() catch 1;
    const s = drama.quant_convert.convert(init.gpa, init.io, paths[0], paths[1], options) catch |e| {
        if (e == error.BadCompactExperts) { // the converter refuses with a name; the mask's own parse says why
            const mask = options.compact_experts.?;
            var problem: []const u8 = "";
            _ = try drama.expert_mask.parse(arena, try std.Io.Dir.cwd().readFileAlloc(init.io, mask, arena, .limited(1 << 20)), &problem);
            std.debug.print("tf-drama-quant: --compact-experts {s}: {s}\n", .{ mask, problem });
        }
        return e;
    };
    std.debug.print("tf-drama-quant: {d} shards, {d} expert projections NVFP4, {d} dense projections FP8, {d} files copied, {d} tensor bytes, {d} threads\n", .{ s.shards, s.converted, s.dense, s.copied, s.bytes, options.threads });
    return 0;
}

fn fail() u8 {
    std.debug.print(usage, .{});
    return 2;
}
