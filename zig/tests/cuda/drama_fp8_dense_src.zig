//! drama-fp8-dense (e)'s fixture: BF16-dense source layers, experts from payloads made once, through tf-drama-quant.
const std = @import("std");
const core = @import("core");
const drama = @import("drama");
const check = @import("check.zig");
const ref = @import("drama_fp8_dense_ref.zig");
const nv = @import("drama_nvfp4_ref.zig");
const h = @import("quant_ref_tests.zig").helpers;
const folder = @import("drama_fp8_dense_folder.zig");

const config = drama.backbone.config;
const weights = drama.backbone.weights;
const quant = drama.quant_convert;

const H = config.hidden;
const E = config.experts;
const I = config.expert_inner;

/// A case's layer: its four groups (ref.groupsOf order) planned FP8 or not, its experts' format, C6.7's extreme rows.
pub const Plan = struct { layer: usize, fp8: [4]bool, nvfp4: bool, extreme: bool = false };

/// Routed experts made once and written into every layer of their format: block FP8 (C0) or NVFP4 (C1).
pub const Payload = struct {
    codes: [3][]u8, // e4m3 [E][n][k] gate, up, down
    inv: [3][]f32, // f32 [E][n / 128][k / 128]
    fp4: [3]nv.Fp4,

    pub fn init(a: std.mem.Allocator, rng: std.Random) !Payload {
        var p: Payload = undefined;
        for (shapes, 0..) |nk, j| {
            const n, const k = nk;
            p.codes[j] = try a.alloc(u8, E * n * k);
            rng.bytes(p.codes[j]);
            for (p.codes[j]) |*b| if (b.* & 0x7f == 0x7f) {
                b.* -= 1; // e4m3 NaN codes out: a checkpoint holds none
            };
            p.inv[j] = try a.alloc(f32, E * (n / 128) * (k / 128));
            for (p.inv[j]) |*v| v.* = 1e-4 * (1 + rng.float(f32));
            p.fp4[j] = try nv.Fp4.random(a, rng, n, k, false);
        }
        return p;
    }
};

const projections = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
const shapes = [3][2]usize{ .{ I, H }, .{ I, H }, .{ H, I } };

/// A source layer's dense rows: each group's parts concatenated, and the BF16 rows written beside it (b | a, router).
pub const Source = struct { groups: [4][]u16, extras: [4][]u16 };

const Writer = struct {
    a: std.mem.Allocator,
    rng: std.Random,
    layer: usize,
    tensors: std.ArrayList(core.shard_edit.New) = .empty,

    fn add(w: *Writer, suffix: []const u8, dtype: core.checkpoint.DType, shape: []const usize, bytes: []const u8) !void {
        const name = try std.fmt.allocPrint(w.a, "{s}layers.{d}.{s}", .{ weights.prefix, w.layer, suffix });
        try w.tensors.append(w.a, .{ .name = name, .dtype = dtype, .shape = try w.a.dupe(usize, shape), .bytes = bytes });
    }

    fn bf16(w: *Writer, suffix: []const u8, shape: []const usize, scale: f32) ![]u16 {
        var n: usize = 1;
        for (shape) |d| n *= d;
        const v = try nv.bfs(w.a, w.rng, n, scale);
        try w.add(suffix, .bf16, shape, std.mem.sliceAsBytes(v));
        return v;
    }

    fn emit(w: *Writer, p: Plan, pay: *const Payload) !Source {
        var s: Source = undefined;
        _ = try w.bf16("input_layernorm.weight", &.{H}, 0.1);
        _ = try w.bf16("post_attention_layernorm.weight", &.{H}, 0.1);
        for (ref.groupsOf(p.layer), 0..) |d, j| {
            var n: usize = 0;
            for (d.parts) |r| n += r;
            // spread near unit-variance sums
            s.groups[j] = try nv.bfs(w.a, w.rng, n * d.k, @floatCast(@sqrt(3.0 / @as(f64, @floatFromInt(d.k)))));
            if (p.extreme) extremes(s.groups[j], d.k, j % 2 == 0, w.rng);
            var at: usize = 0;
            for (d.stems, d.parts) |stem, rows| {
                try w.add(try std.fmt.allocPrint(w.a, "{s}.weight", .{stem}), .bf16, &.{ rows, d.k }, std.mem.sliceAsBytes(s.groups[j][at * d.k ..][0 .. rows * d.k]));
                at += rows;
            }
            var extra: std.ArrayList(u16) = .empty;
            for (d.extra, d.extra_rows) |stem, rows|
                try extra.appendSlice(w.a, try w.bf16(try std.fmt.allocPrint(w.a, "{s}.weight", .{stem}), &.{ rows, d.k }, if (j == 0) 0.04 else 0.05));
            s.extras[j] = extra.items;
        }
        if (config.linear(p.layer)) {
            const heads = config.linear_value_heads;
            _ = try w.bf16("linear_attn.conv1d.weight", &.{ config.conv_dim, 1, config.conv_taps }, 0.5);
            const a_log = try w.a.alloc(u16, heads);
            for (a_log) |*v| v.* = h.bf16Of(@log(1 + 15 * w.rng.float(f32)));
            try w.add("linear_attn.A_log", .bf16, &.{heads}, std.mem.sliceAsBytes(a_log));
            _ = try w.bf16("linear_attn.dt_bias", &.{heads}, 1);
            const gnorm = try w.a.alloc(u16, config.linear_dim);
            for (gnorm) |*v| v.* = h.bf16Of(0.9 + 0.2 * w.rng.float(f32));
            try w.add("linear_attn.norm.weight", .bf16, &.{config.linear_dim}, std.mem.sliceAsBytes(gnorm));
        } else {
            _ = try w.bf16("self_attn.q_norm.weight", &.{config.head_dim}, 0.1);
            _ = try w.bf16("self_attn.k_norm.weight", &.{config.head_dim}, 0.1);
        }
        _ = try w.bf16("mlp.shared_expert_gate.weight", &.{ 1, H }, 0.02);
        for (projections, shapes, 0..) |proj, nk, j| {
            const n, const k = nk;
            for (0..E) |e| {
                const stem = try std.fmt.allocPrint(w.a, "mlp.experts.{d}.{s}.", .{ e, proj });
                if (p.nvfp4) {
                    const f = pay.fp4[j];
                    try w.add(try std.mem.concat(w.a, u8, &.{ stem, "weight" }), .u8, &.{ n, k / 2 }, f.codes[e * n * k / 2 ..][0 .. n * k / 2]);
                    try w.add(try std.mem.concat(w.a, u8, &.{ stem, "weight_scale" }), .f8_e4m3, &.{ n, k / 16 }, f.scales[e * n * k / 16 ..][0 .. n * k / 16]);
                    try w.add(try std.mem.concat(w.a, u8, &.{ stem, "weight_scale_2" }), .f32, &.{}, std.mem.asBytes(&f.global[e]));
                } else {
                    const blocks = (n / 128) * (k / 128);
                    try w.add(try std.mem.concat(w.a, u8, &.{ stem, "weight" }), .f8_e4m3, &.{ n, k }, pay.codes[j][e * n * k ..][0 .. n * k]);
                    try w.add(try std.mem.concat(w.a, u8, &.{ stem, "weight_scale_inv" }), .f32, &.{ n / 128, k / 128 }, std.mem.sliceAsBytes(pay.inv[j][e * blocks ..][0..blocks]));
                }
            }
        }
        return s;
    }
};

/// C6.7's extreme 64-input groups in rows 0..4; an input group's huge ones on inputs 0..63, which its replay zeroes.
fn extremes(v: []u16, k: usize, input: bool, rng: std.Random) void {
    const sign = struct {
        fn of(r: std.Random, bits: u16) u16 {
            return bits | (@as(u16, @intFromBool(r.boolean())) << 15);
        }
    }.of;
    for (0..64) |c| {
        if (input) v[c] = sign(rng, 0x7F7F); // the largest finite bf16: a scale near 2^119
        if (input) v[3 * k + c] = if (c == 0) 0x7F7F else sign(rng, 0x0001); // beside subnormals that become zero codes
        v[k + 64 + c] = sign(rng, 0x0001); // the least subnormal bf16: C6.7's subnormal f32 scale, 2.05e-43
        v[2 * k + c] = 0;
        v[4 * k + 64 + c] = sign(rng, 0x0080); // the least normal bf16
    }
}

/// `plans`' layers as a BF16-dense source checkpoint (a shard a layer) under `dir`; returns their dense rows.
pub fn write(gpu: check.Gpu, a: std.mem.Allocator, rng: std.Random, dir: []const u8, plans: []const Plan, pay: *const Payload) ![]Source {
    const out = try a.alloc(Source, plans.len);
    const shards = try a.alloc(folder.Shard, plans.len);
    for (plans, out, shards) |p, *s, *sh| {
        var w: Writer = .{ .a = a, .rng = rng, .layer = p.layer };
        s.* = try w.emit(p, pay);
        sh.* = .{ .name = try std.fmt.allocPrint(a, "layer-{d}.safetensors", .{p.layer}), .tensors = w.tensors.items };
    }
    try folder.write(gpu.gpa, a, gpu.io, dir, shards);
    return out;
}

/// tf-drama-quant's conversion of SRC into DST, with the CLI's thread count.
pub fn convert(gpu: check.Gpu, src: []const u8, dst: []const u8, o: quant.Options) !void {
    var opt = o;
    opt.threads = std.Thread.getCpuCount() catch 1;
    opt.commit = "tf-cuda-test drama-fp8-dense";
    const s = try quant.convert(gpu.gpa, gpu.io, src, dst, opt);
    std.debug.print("RESULT tf-drama-quant {s} -> {s} (experts {t}, dense-fp8 {s}, keep-bf16 {s}): {d} expert projections NVFP4, {d} dense FP8\n", .{ src, dst, opt.experts, opt.dense_fp8 orelse "-", opt.keep_bf16 orelse "-", s.converted, s.dense });
}

/// One run a layer with FP8 groups, each reading the last: `--experts keep --dense-fp8 all --keep-bf16 BF16S,b:OTHERS`.
pub fn convertCase(gpu: check.Gpu, a: std.mem.Allocator, root: []const u8, plans: []const Plan) ![]const u8 {
    var from = try std.fs.path.join(a, &.{ root, "src" });
    for (plans, 0..) |p, i| {
        if (std.mem.indexOfScalar(bool, &p.fp8, true) == null) continue;
        var keep: std.ArrayList(u8) = .empty;
        for (ref.groupsOf(p.layer), p.fp8) |d, f| if (!f) try keep.print(a, "{s},", .{d.name});
        var space: []const u8 = "b:";
        for (plans) |o| if (o.layer != p.layer) {
            try keep.print(a, "{s}{d},", .{ space, o.layer });
            space = "";
        };
        const to = try std.fmt.allocPrint(a, "{s}/q{d}", .{ root, i });
        const kept = if (keep.items.len == 0) null else keep.items[0 .. keep.items.len - 1];
        try convert(gpu, from, to, .{ .experts = .keep, .dense_fp8 = "all", .keep_bf16 = kept });
        try std.Io.Dir.cwd().deleteTree(gpu.io, from); // an earlier step's output, this run's own files
        from = to;
    }
    return from;
}
