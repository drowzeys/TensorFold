//! drama-fp8-dense (e): C8's matrix as tf-drama-quant emits it, through the production loader into operator replays.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const drama = @import("drama");
const check = @import("check.zig");
const ref = @import("drama_fp8_dense_ref.zig");
const src = @import("drama_fp8_dense_src.zig");
const nv = @import("drama_nvfp4_ref.zig");

const backbone = drama.backbone;
const config = backbone.config;
const weights = backbone.weights;
const Kernels = backbone.kernels.Kernels;
const Model = backbone.Model;
const ReplayOperator = backbone.ReplayOperator;
const Checkpoint = core.checkpoint.Checkpoint;
const Plan = src.Plan;

const H = config.hidden;
const rows = nv.rows;

/// Layout j of a layer: its mixer's and its shared expert's input groups FP8 when j & 1, output groups when j & 2.
fn plan(layer: usize, j: usize, nvfp4: bool) Plan {
    const in = j & 1 != 0;
    const out = j & 2 != 0;
    return .{ .layer = layer, .fp8 = .{ in, out, in, out }, .nvfp4 = nvfp4 };
}

/// Case k: layouts k..k+3 (mod 4) on layers 0, 1 (DeltaNet; FP8, NVFP4 experts) and 3, 7 (attention; FP8, NVFP4).
const cases = blk: {
    var c: [4][4]Plan = undefined;
    for (&c, 0..) |*p, k| p.* = .{ plan(0, k, false), plan(1, (k + 1) % 4, true), plan(3, (k + 2) % 4, false), plan(7, (k + 3) % 4, true) };
    break :blk c;
};

/// C6.7: every group FP8 from extreme BF16 sources; `--experts nvfp4 --keep-fp8 0` makes layer 3's experts NVFP4.
const extreme = [2]Plan{
    .{ .layer = 0, .fp8 = @splat(true), .nvfp4 = false, .extreme = true },
    .{ .layer = 3, .fp8 = @splat(true), .nvfp4 = true, .extreme = true },
};

/// A dense group as tf-drama-quant wrote it.
const Dense = union(enum) {
    fp8: ref.Group,
    bf16: []const u16,

    /// The BF16 path's weights: the stored bf16, or the bf16 nearest each FP8 value.
    fn bf16Rows(d: Dense, a: std.mem.Allocator) ![]const u16 {
        return switch (d) {
            .fp8 => |g| g.bf16(a),
            .bf16 => |w| w,
        };
    }
};

fn readGroup(a: std.mem.Allocator, ck: *Checkpoint, layer: usize, d: ref.Desc) !Dense {
    var name: [160]u8 = undefined;
    const first = try ck.get(try std.fmt.bufPrint(&name, "{s}layers.{d}.{s}.weight", .{ weights.prefix, layer, d.stems[0] }));
    if (first.dtype != .f8_e4m3) return .{ .bf16 = try ref.bf16Rows(a, try ref.read(a, ck, layer, d.stems, d.parts, d.k, ".weight", .bf16)) };
    var n: usize = 0;
    for (d.parts) |p| n += p;
    const g = try ref.Group.alloc(a, n, d.k);
    @memcpy(g.codes, try ref.read(a, ck, layer, d.stems, d.parts, d.k, ".weight", .f8_e4m3));
    @memcpy(std.mem.sliceAsBytes(g.inv), try ref.read(a, ck, layer, d.stems, d.parts, d.k / 64, ".weight_scale_inv", .f32));
    return .{ .fp8 = g };
}

/// Group j of a loaded layer: its lane weight when FP8, else its bf16 rows, and the rows split off it.
const Slot = struct { q: ?cuda.qlinear.Weight, bf16: u64, split: u64 };

fn slot(l: weights.Layer, layer: usize, j: usize) Slot {
    const linear = config.linear(layer);
    return switch (j) {
        0 => if (linear) .{ .q = l.fp8.gdn_in, .bf16 = l.gdn_in, .split = l.gdn_ba } else .{ .q = l.fp8.attn_in, .bf16 = l.attn_in, .split = 0 },
        1 => if (linear) .{ .q = l.fp8.gdn_out, .bf16 = l.gdn_out, .split = 0 } else .{ .q = l.fp8.attn_out, .bf16 = l.attn_out, .split = 0 },
        2 => .{ .q = l.fp8.shared_in, .bf16 = l.moe_in, .split = l.router },
        else => .{ .q = l.fp8.shared_out, .bf16 = l.shared_down, .split = 0 },
    };
}

fn setBf16(t: *weights.Layer, layer: usize, j: usize, p: u64) void {
    const linear = config.linear(layer);
    switch (j) {
        0 => if (linear) {
            t.gdn_in = p;
        } else {
            t.attn_in = p;
        },
        1 => if (linear) {
            t.gdn_out = p;
        } else {
            t.attn_out = p;
        },
        2 => t.moe_in = p,
        else => t.shared_down = p,
    }
}

/// What the loader put on the device, byte for byte: FP8 groups in the lane's layout, BF16 rows as written.
fn sameGroup(r: *ref.Rig, what: []const u8, s: Slot, d: Dense, extra: []const u16) !void {
    var name: [96]u8 = undefined;
    try check.expect((s.q != null) == (d == .fp8), "{s}: loaded as {t}", .{ what, std.meta.activeTag(d) });
    switch (d) {
        .fp8 => |g| {
            const p = try g.toLane(r.a);
            const w = s.q.?.fp8g;
            try check.sameBytes(try std.fmt.bufPrint(&name, "{s} codes", .{what}), try r.back(w.codes, p.codes.len), p.codes);
            try check.sameBytes(try std.fmt.bufPrint(&name, "{s} scale tiles", .{what}), try r.back(w.scales, p.tiles.len * 4), std.mem.sliceAsBytes(p.tiles));
            if (extra.len > 0) try check.sameBytes(try std.fmt.bufPrint(&name, "{s} split rows", .{what}), try r.back(s.split, extra.len * 2), std.mem.sliceAsBytes(extra));
        },
        .bf16 => |w| {
            const want = try std.mem.concat(r.a, u16, &.{ w, extra });
            try check.sameBytes(what, try r.back(s.bf16, want.len * 2), std.mem.sliceAsBytes(want));
        },
    }
}

/// A model on `w` with the kernels Model.init loads for its formats.
fn model(gpu: check.Gpu, w: weights.Weights) !Model {
    var k = try Kernels.load(gpu.gpa, gpu.io, gpu.d, "");
    errdefer k.unload();
    if (w.nvfp4Layers() > 0) try k.loadFp4(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    if (w.denseFp8().layers > 0) try k.loadLane(gpu.d, try gpu.ctx.capability() / 10);
    return Model.assemble(gpu.gpa, gpu.d, k, w, .{ .eos = 0, .semantic_start = 1, .semantic_end = 4096 }, .{ .context = 2 * rows, .max_rows = rows });
}

fn replay(m: *Model, s: cuda.Stream, layer: usize, op: ReplayOperator, chunks: []const usize, in: []const u8, out: []u8) !void {
    try m.reset(s);
    try m.replayOperator(s, layer, op, chunks, in, out);
}

const ones: [rows]usize = @splat(1);
const sevens = @as([rows / 7]usize, @splat(7)) ++ [_]usize{rows % 7};

/// Model A's rows against its BF16 twin's, preregistered: both finite, relative RMS 2^-6 overall, 2^-4 every column.
fn near(a: std.mem.Allocator, what: []const u8, got: []const u8, twin_rows: []const u8) !void {
    const want = try a.alloc(f64, twin_rows.len / 2);
    for (want, 0..) |*w, i| w.* = ref.score.bfOf(std.mem.bytesToValue(u16, twin_rows[2 * i ..][0..2]));
    const st = try ref.score.stats(a, got, want, null, H, null);
    std.debug.print("RESULT {s}: {d} non-finite outputs, {d} non-finite twin outputs; relative RMS distance to the BF16 twin {e:.3} overall, {e:.3} worst column\n", .{ what, st.nonfinite_got, st.nonfinite_want, st.agg, st.col });
    try check.expect(st.valid() and st.agg <= 0x1p-6 and st.col <= 0x1p-4, "{s}: both finite, limits 2^-6 overall and 2^-4 a column", .{what});
}

pub fn run(gpu: check.Gpu, r: *ref.Rig) !void {
    const root = try std.fmt.allocPrint(r.a, "/tmp/tf-drama-fp8-dense-{x}", .{r.rng.int(u64)});
    defer std.Io.Dir.cwd().deleteTree(gpu.io, root) catch {}; // this run's own files
    const pay = try src.Payload.init(r.a, r.rng);
    var name: [32]u8 = undefined;
    for (cases, 0..) |plans, k| {
        var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
        defer arena.deinit();
        const dir = try std.fmt.allocPrint(arena.allocator(), "{s}/case{d}", .{ root, k });
        const sources = try src.write(gpu, arena.allocator(), r.rng, try std.fs.path.join(arena.allocator(), &.{ dir, "src" }), &plans, &pay);
        const final = try src.convertCase(gpu, arena.allocator(), dir, &plans);
        try case(gpu, r, arena.allocator(), try std.fmt.bufPrint(&name, "case {d}", .{k}), final, &plans, sources);
        try std.Io.Dir.cwd().deleteTree(gpu.io, dir);
    }
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try std.fmt.allocPrint(a, "{s}/extreme", .{root});
    const from = try std.fs.path.join(a, &.{ dir, "src" });
    var source_plans = extreme;
    for (&source_plans) |*p| p.nvfp4 = false; // the converter makes layer 3's FP8 experts NVFP4
    const sources = try src.write(gpu, a, r.rng, from, &source_plans, &pay);
    const to = try std.fs.path.join(a, &.{ dir, "q" });
    try src.convert(gpu, from, to, .{ .experts = .nvfp4, .keep = &.{0}, .dense_fp8 = "all" });
    try case(gpu, r, a, "C6.7 extremes", to, &extreme, sources);
    try std.Io.Dir.cwd().deleteTree(gpu.io, dir);
}

/// One converted case: formats as planned, loader bytes, the attention prerequisite, replays against the BF16 twin.
fn case(gpu: check.Gpu, parent: *ref.Rig, a: std.mem.Allocator, what: []const u8, final: []const u8, plans: []const Plan, sources: []const src.Source) !void {
    var r: ref.Rig = .{ .gpu = gpu, .a = a, .s = parent.s, .rng = parent.rng };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var ck = try Checkpoint.openModel(a, gpu.io, final);
    defer ck.close();
    const dense = try a.alloc([4]Dense, plans.len);
    var name: [128]u8 = undefined;
    var want: weights.dense.Census = .{};
    for (plans, sources, dense) |p, s, *ds| {
        for (ref.groupsOf(p.layer), p.fp8, s.groups, ds) |d, f, rows_in, *dd| {
            dd.* = try readGroup(a, &ck, p.layer, d);
            try check.expect((dd.* == .fp8) == f, "{s}: tf-drama-quant wrote layer {d} {s} as {t}", .{ what, p.layer, d.name, std.meta.activeTag(dd.*) });
            if (dd.* == .bf16) try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: layer {d} {s} kept", .{ what, p.layer, d.name }), std.mem.sliceAsBytes(dd.bf16), std.mem.sliceAsBytes(rows_in));
        }
        const gate0 = try ck.get(try std.fmt.bufPrint(&name, "{s}layers.{d}.mlp.experts.0.gate_proj.weight", .{ weights.prefix, p.layer }));
        try check.expect((gate0.dtype == .u8) == p.nvfp4, "{s}: layer {d}'s experts", .{ what, p.layer });
        var any = false;
        for (ref.groupsOf(p.layer), p.fp8) |d, f| if (f) {
            for (weights.dense.group_names, 0..) |g, at| want.groups[at] += @intFromBool(std.mem.eql(u8, g, d.name));
            any = true;
        };
        want.layers += @intFromBool(any);
    }
    var layer_ids: [4]usize = undefined;
    for (plans, 0..) |p, i| layer_ids[i] = p.layer;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const wa = try weights.loadLayers(gpu.gpa, gpu.io, gpu.d, stream, final, layer_ids[0..plans.len]);
    var ma = try model(gpu, wa);
    defer ma.deinit();
    try check.expect(std.meta.eql(ma.w.denseFp8(), want), "{s}: the loader's census {any} is the planned {any}", .{ what, ma.w.denseFp8(), want });
    try check.expect(ma.w.inv_freq != 0, "{s}: the loaded weights hold RoPE frequencies, which attention reads", .{what});
    const inv = weights.ropeInvFreq();
    try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: RoPE frequencies", .{what}), try r.back(ma.w.inv_freq, inv.len * 4), std.mem.sliceAsBytes(&inv));
    for (plans, sources, dense) |p, s, ds| for (0..4) |j| {
        const g = ref.groupsOf(p.layer)[j];
        try sameGroup(&r, try std.fmt.bufPrint(&name, "{s}: layer {d} {s}", .{ what, p.layer, g.name }), slot(ma.w.layers[p.layer], p.layer, j), ds[j], s.extras[j]);
    };
    if (plans[0].extreme) try extremes(&r, &ma, what, plans, dense);
    check.pass("{s}: every group converted as planned, on the device as written; census {any}", .{ what, want.groups });

    // the twin: each FP8 group as the bf16 nearest its values, fused as BF16; experts and frequencies borrowed
    var wb: weights.Weights = .{ .gpa = gpu.gpa, .inv_freq = ma.w.inv_freq };
    for (plans, sources, dense) |p, s, ds| {
        var t = ma.w.layers[p.layer];
        t.fp8 = .{};
        t.gdn_ba = 0;
        t.router = 0;
        for (ds, s.extras, 0..) |d, extra, j| setBf16(&t, p.layer, j, try r.dev(std.mem.sliceAsBytes(try std.mem.concat(a, u16, &.{ try d.bf16Rows(a), extra }))));
        wb.layers[p.layer] = t;
    }
    var mb = try model(gpu, wb);
    defer mb.deinit(); // before ma's: the twin borrows ma's experts and frequencies
    try check.expect(mb.k.lane == null and mb.w.inv_freq == ma.w.inv_freq, "{s}: the twin loads no lane and reads the frequencies", .{what});
    var io = try nv.Pinned.init(gpu.d);
    defer io.deinit();
    for (plans) |p| for ([_]ReplayOperator{ .attention, .moe }) |op| {
        const label = try std.fmt.bufPrint(&name, "{s}: layer {d} {t} ({s} experts)", .{ what, p.layer, op, if (p.nvfp4) "NVFP4" else "FP8" });
        const x = try nv.bfs(a, r.rng, rows * H, 1.0);
        if (p.extreme) for (0..rows) |row| @memset(x[row * H ..][0..64], 0); // the inputs of the largest groups
        @memcpy(io.in.bytes, std.mem.sliceAsBytes(x));
        try replay(&ma, r.s, p.layer, op, &ones, io.in.bytes, io.outs[1]);
        try replay(&ma, r.s, p.layer, op, &sevens, io.in.bytes, io.outs[2]);
        try replay(&ma, r.s, p.layer, op, &.{rows}, io.in.bytes, io.outs[0]);
        try check.sameBytes(label, io.outs[1], io.outs[0]);
        try check.sameBytes(label, io.outs[2], io.outs[0]);
        const got = try a.dupe(u8, io.outs[0]);
        try replay(&mb, r.s, p.layer, op, &.{rows}, io.in.bytes, io.outs[0]);
        const fp8 = switch (op) {
            .attention => p.fp8[0] or p.fp8[1],
            .moe => p.fp8[2] or p.fp8[3],
        };
        if (fp8) try near(a, label, got, io.outs[0]) else try check.sameBytes(label, got, io.outs[0]);
        check.pass("{s}: 1, 7 and {d}-row passes bit-equal; {s} the BF16 twin", .{ label, rows, if (fp8) "near" else "bit-equal to" });
    };
}

/// C6.7's runtime checks: the converted scale domain, each group's projection on inputs scaled by 2^-24, and alone.
fn extremes(r: *ref.Rig, ma: *Model, what: []const u8, plans: []const Plan, dense: []const [4]Dense) !void {
    var name: [128]u8 = undefined;
    for (plans, dense) |p, ds| for (ds, 0..) |d, j| {
        const g = d.fp8;
        var subnormal = false;
        var most: f32 = 0;
        for (g.inv) |s| {
            subnormal = subnormal or s < std.math.floatMin(f32);
            most = @max(most, s);
        }
        const label = try std.fmt.bufPrint(&name, "{s}: layer {d} {s}", .{ what, p.layer, ref.groupsOf(p.layer)[j].name });
        try check.expect(subnormal and (j % 2 == 1 or most >= 0x1p119), "{s}: a subnormal f32 scale and, in an input group, one of 2^119 or more (largest {e})", .{ label, most });
        const x = try r.bfs(rows * g.k, 1.0);
        for (x) |*v| v.* = ref.score.bf16Nearest(ref.score.bfOf(v.*) * 0x1p-24);
        const want = try ref.reference(r.a, try g.dequant(r.a), g.n, g.k, x, rows);
        const lin: cuda.qlinear.Linear = .{ .lane = &ma.k.lane.? };
        const got = try ref.rowCounts(r, lin, slot(ma.w.layers[p.layer], p.layer, j).q.?, try r.dev(std.mem.sliceAsBytes(x)), rows, ma.s.part, label);
        const st = try ref.stats(r.a, got, want, g.n, null);
        ref.report(label, g.n, st);
        try check.expect(st.nonfinite_got == 0 and st.nonfinite_want == 0 and st.worst <= 1, "{s}: finite and within every output's tolerance", .{label});
        try subnormalProbe(r, lin, slot(ma.w.layers[p.layer], p.layer, j).q.?, ma.s.part, g, label);
    };
}

/// Row 1's least-subnormal group alone, ±1 inputs as its codes' signs: output 1 is ~63.9x the 2^-133 floor, never 0.
fn subnormalProbe(r: *ref.Rig, lin: cuda.qlinear.Linear, w: cuda.qlinear.Weight, part: u64, g: ref.Group, label: []const u8) !void {
    var name: [160]u8 = undefined;
    const what = try std.fmt.bufPrint(&name, "{s}, row 1's subnormal group alone", .{label});
    const x = try r.a.alloc(u16, rows * g.k);
    @memset(x, 0);
    for (0..rows) |row| for (64..128) |c| {
        x[row * g.k + c] = if (g.codes[g.k + c] & 0x80 != 0) 0xBF80 else 0x3F80;
    };
    const want = try ref.reference(r.a, try g.dequant(r.a), g.n, g.k, x, rows);
    const got = try ref.rowCounts(r, lin, w, try r.dev(std.mem.sliceAsBytes(x)), rows, part, what);
    const st = try ref.stats(r.a, got, want, g.n, null);
    ref.report(what, g.n, st);
    try check.expect(st.nonfinite_got == 0 and st.nonfinite_want == 0 and st.worst <= 1, "{s}: finite and within every output's tolerance", .{what});
    var zero: usize = 0;
    for (0..rows) |row| {
        const out = std.mem.bytesToValue(u16, got[2 * (row * g.n + 1) ..][0..2]);
        zero += @intFromBool(out & 0x7FFF == 0 or !(want.want[row * g.n + 1] > 0));
    }
    std.debug.print("RESULT {s}: output 1 is {e} (float64 {e}); {d} of {d} rows zero\n", .{ what, ref.score.bfOf(std.mem.bytesToValue(u16, got[2..4])), want.want[1], zero, rows });
    try check.expect(zero == 0, "{s}: output 1, the group's aligned sum, positive in float64 and nonzero in every row", .{what});
}
