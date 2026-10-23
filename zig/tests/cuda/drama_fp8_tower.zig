//! tf-cuda-test drama-fp8-tower: the tower's fp8g projections on the production tower, held to `criteria`.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const drama = @import("drama");
const check = @import("check.zig");

const tower = drama.tower;
const Impl = @typeInfo(@FieldType(tower.Model, "impl")).pointer.child;
const Layer = @typeInfo(@FieldType(Impl, "layers")).array.child;
const Projection = @FieldType(Layer, "qkv");
const Buffer = cuda.DeviceBuffer;
const fp8 = cuda.fp8;
const e4m3 = core.quant.fp8.e4m3;
const Allocator = std.mem.Allocator;

pub const criteria =
    \\drama-fp8-tower criteria, fixed before the first run. For each tower kind (wqkv, wo, gate_up, down) and rows
    \\m in {1, 7, 128}, on the same bf16 rows x:
    \\  y = sum_k x[r][k] e4m3(code[c][k]) scale[c][k/64], in float64 from the stored codes and scales;
    \\  tol = 2^-7 |y| + 2^-12 sum_k |x w| + 2^-133 (a bf16 ulp, fp32 sums, the bf16 subnormal step);
    \\  column error = max over the call's rows of |got - y| / tol; worst column = its max over columns;
    \\  aggregate = ||got - y||_2 / ||y||_2 over the call's m x n outputs; when ||y||_2 = 0 (an all-zero reference):
    \\  0 if ||got - y||_2 = 0, else +inf (added after review, before any run).
    \\  lane = project() on the fp8g weight; comparator = project() on the bf16-rounded dequantized weight (each
    \\  float64 product rounded once to bf16, nearest-even): the qualified gemv at m = 1 when gemv_exact, else the
    \\  m-row cuBLASLt plan.
    \\F every output of both finite. B the lane's worst column <= 1. W the lane's worst column <= the comparator's.
    \\A the lane's aggregate finite and <= the comparator's. R the lane's rows at m = 1 and 7 byte-equal to its m = 128.
    \\L (stored layers) the loaded format is the stored one; packed codes and scale tiles equal the host packers'.
    \\M (mixed kinds and layers) in prefills of 1, 7 and 128 rows, every projection's output equals project() of
    \\that layer's weight on the captured input, and each fp8g one a direct lane call, byte for byte.
    \\Synthetic weights: random codes without 0x7F/0xFF, row 3 all +-448, row 5 subnormal codes; scales 2^e (1 + u),
    \\e in [-20, 8]; rows 7, 68, .. (every 61st) positive f32 subnormal scales from 146 x 2^-149 to below 2^-126.
    \\Z (all-zero reference) wo with every code 0x00: F, B, W, A and R as above; A so requires the lane's output 0.
;

/// Kinds 1-4 of tower.zig's projections: shape, the checkpoint parts fused into it, its trace stages.
const Shape = struct { kind: usize, field: []const u8, n: usize, k: usize, parts: []const []const u8, input: tower.Stage, output: tower.Stage };
const shapes = [_]Shape{
    .{ .kind = 1, .field = "qkv", .n = 9216, .k = 2048, .parts = &.{"attention.wqkv"}, .input = .attention_input, .output = .qkv },
    .{ .kind = 2, .field = "out", .n = 2048, .k = 4096, .parts = &.{"attention.wo"}, .input = .attended, .output = .attention_output },
    .{ .kind = 3, .field = "gate_up", .n = 12288, .k = 2048, .parts = &.{ "feed_forward.w1", "feed_forward.w3" }, .input = .mlp_input, .output = .gate_up },
    .{ .kind = 4, .field = "down", .n = 2048, .k = 6144, .parts = &.{"feed_forward.w2"}, .input = .activated, .output = .mlp_output },
};
const max_rows = 128;
const row_counts = [_]usize{ 1, 7, max_rows };
const hidden = 2048;
const max_k = 6144;
const max_n = 12288;
/// Per layer, F synthetic fp8g, B its dequantized bf16, . as loaded: fp8g and BF16 meet within layers and kinds.
const mixed = [_]struct { usize, *const [4]u8 }{ .{ 0, "FFFF" }, .{ 2, "FBBF" }, .{ 5, "..FB" }, .{ 12, "BF.." } };

fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// x rounded once to bf16, nearest-even: through f32 rounded to odd, which keeps the tie information.
fn bf16Of(x: f64) u16 {
    const f: f32 = @floatCast(x);
    var u: u32 = @bitCast(f);
    if (@as(f64, f) != x) {
        if (@abs(@as(f64, f)) > @abs(x)) u -= 1;
        u |= 1;
    }
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// One kind's stored codes [n][k] and row scales [n][k/64].
const Stored = struct {
    codes: []u8,
    scales: []f32,

    fn synthetic(a: Allocator, rng: std.Random, n: usize, k: usize, extreme: bool) !Stored {
        const codes = try a.alloc(u8, n * k);
        rng.bytes(codes);
        for (codes) |*c| if (c.* & 0x7f == 0x7f) {
            c.* -= 1; // e4m3 NaN codes out: C5 has none
        };
        if (extreme) for (0..k) |i| {
            codes[3 * k + i] = if (i % 2 == 0) 0x7E else 0xFE;
            codes[5 * k + i] = (1 + rng.uintLessThan(u8, 7)) | (rng.int(u8) & 0x80);
        };
        const kg = k / fp8.group;
        const scales = try a.alloc(f32, n * kg);
        for (scales, 0..) |*s, i| {
            const r = i / kg;
            s.* = if (extreme and r % 61 == 7)
                @bitCast(if (i % kg == 0) @as(u32, 146) else 1 + rng.uintLessThan(u32, 0x7fffff))
            else
                std.math.ldexp(1 + rng.float(f32), if (extreme) rng.intRangeAtMost(i32, -20, 8) else rng.intRangeAtMost(i32, -16, -12));
        }
        return .{ .codes = codes, .scales = scales };
    }

    /// Layer `layer`'s stored parts of `s`, fused, or null when the checkpoint holds them BF16.
    fn fromCheckpoint(a: Allocator, ck: *core.checkpoint.Checkpoint, layer: usize, s: Shape) !?Stored {
        const n = s.n / s.parts.len;
        const kg = s.k / fp8.group;
        const out: Stored = .{ .codes = try a.alloc(u8, s.n * s.k), .scales = try a.alloc(f32, s.n * kg) };
        for (s.parts, 0..) |part, j| {
            var name: [160]u8 = undefined;
            const weight = if (layer < 4)
                try std.fmt.bufPrint(&name, "audio_decoder_mtp_shared.layers.{d}.transformer_layer.{s}.weight", .{ layer, part })
            else
                try std.fmt.bufPrint(&name, "audio_decoder_mtp.layers.{d}.transformer_layer.{s}.weight", .{ layer - 4, part });
            const t = try ck.get(weight);
            if (t.dtype != .f8_e4m3) return null;
            @memcpy(out.codes[j * n * s.k ..][0 .. n * s.k], t.bytes);
            var companion: [192]u8 = undefined;
            const scales = try ck.get(try std.fmt.bufPrint(&companion, "{s}_scale_inv", .{weight}));
            @memcpy(std.mem.sliceAsBytes(out.scales[j * n * kg ..][0 .. n * kg]), scales.bytes);
        }
        return out;
    }

    /// The lane's packed layout of these bytes, as host packers write it.
    fn forLane(st: Stored, a: Allocator, n: usize, k: usize) !struct { codes: []u8, tiles: []f32 } {
        const codes = try a.alloc(u8, fp8.padded(n) * k);
        fp8.packCodes(codes, st.codes, n, k);
        const tiles = try a.alloc(f32, fp8.padded(n) * (k / fp8.group));
        fp8.packRowScales(tiles, st.scales, n, k);
        return .{ .codes = codes, .tiles = tiles };
    }
};

/// The float64 reference of `x` (max_rows rows) through `st`, its tolerance, and the bf16-rounded weight.
const Reference = struct {
    want: []f64,
    tol: []f64,
    dequantized: []u16,

    fn of(a: Allocator, x: []const u16, st: Stored, n: usize, k: usize) !Reference {
        const r: Reference = .{ .want = try a.alloc(f64, max_rows * n), .tol = try a.alloc(f64, max_rows * n), .dequantized = try a.alloc(u16, n * k) };
        const xs = try a.alloc(f64, max_rows * k);
        defer a.free(xs);
        for (xs, x) |*d, v| d.* = bf(v);
        const w = try a.alloc(f64, k);
        defer a.free(w);
        for (0..n) |c| {
            for (w, 0..) |*v, i| {
                v.* = e4m3(st.codes[c * k + i]) * @as(f64, st.scales[c * (k / fp8.group) + i / fp8.group]);
                r.dequantized[c * k + i] = bf16Of(v.*);
            }
            for (0..max_rows) |row| {
                var sum: f64 = 0;
                var mag: f64 = 0;
                for (xs[row * k ..][0..k], w) |xv, wv| {
                    sum += xv * wv;
                    mag += @abs(xv * wv);
                }
                r.want[row * n + c] = sum;
                r.tol[row * n + c] = 0x1p-7 * @abs(sum) + 0x1p-12 * mag + 0x1p-133;
            }
        }
        return r;
    }

    const Error = struct { finite: bool, worst: f64, column: usize, aggregate: f64 };

    fn measure(r: Reference, got: []const u16, m: usize, n: usize) Error {
        var e: Error = .{ .finite = true, .worst = 0, .column = 0, .aggregate = 0 };
        var diff2: f64 = 0;
        var want2: f64 = 0;
        for (0..n) |c| {
            var column: f64 = 0;
            for (0..m) |row| {
                const g = bf(got[row * n + c]);
                const y = r.want[row * n + c];
                e.finite = e.finite and std.math.isFinite(g);
                column = @max(column, @abs(g - y) / r.tol[row * n + c]);
                diff2 += (g - y) * (g - y);
                want2 += y * y;
            }
            if (column > e.worst) {
                e.worst = column;
                e.column = c;
            }
        }
        e.aggregate = if (want2 > 0) @sqrt(diff2 / want2) else if (diff2 == 0) 0 else std.math.inf(f64);
        return e;
    }
};

test "error measures: an all-zero reference gives 0 for exact zeros, +inf otherwise; nonfinite output is flagged" {
    var want = [_]f64{ 0, 0, 0, 0 };
    var tol = [_]f64{ 0x1p-133, 0x1p-133, 0x1p-133, 0x1p-133 };
    const zero: Reference = .{ .want = &want, .tol = &tol, .dequantized = &.{} };
    const exact = zero.measure(&.{ 0x0000, 0x8000, 0x0000, 0x0000 }, 2, 2);
    try std.testing.expect(exact.finite and exact.worst == 0 and exact.aggregate == 0);
    const off = zero.measure(&.{ 0x0000, 0x0000, 0x3f80, 0x0000 }, 2, 2); // 1.0 at row 1, column 0
    try std.testing.expect(off.finite and off.worst > 1 and off.column == 0 and std.math.isPositiveInf(off.aggregate));
    try std.testing.expect(!zero.measure(&.{ 0x0000, 0x7fc0, 0x0000, 0x0000 }, 2, 2).finite);
    want = .{ 3, 0, 4, 0 };
    tol = .{ 1, 1, 1, 1 };
    const plain: Reference = .{ .want = &want, .tol = &tol, .dequantized = &.{} };
    const near = plain.measure(&.{ 0x4040, 0x0000, 0x4080, 0x3fa0 }, 2, 2); // 3, 0, 4, 1.25: 1.25 / 5 off in L2
    try std.testing.expect(near.finite and near.worst == 1.25 and near.column == 1 and near.aggregate == 0.25);
}

const Run = struct {
    gpu: check.Gpu,
    a: Allocator,
    s: cuda.Stream,
    p: *Impl,
    rng: std.Random,
    bufs: std.ArrayList(Buffer) = .empty,
    failed: usize = 0,

    fn dev(r: *Run, bytes: []const u8) !Buffer {
        const b = try Buffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b;
    }

    fn room(r: *Run, bytes: usize) !Buffer {
        const b = try Buffer.alloc(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b;
    }

    /// Records a failed criterion and goes on, so every comparison reports before the command fails.
    fn expect(r: *Run, ok: bool, comptime fmt: []const u8, args: anytype) void {
        if (ok) return;
        std.debug.print("FAIL " ++ fmt ++ "\n", args);
        r.failed += 1;
    }

    fn back(r: *Run, b: Buffer, rows: usize, width: usize) ![]u16 {
        const out = try r.a.alloc(u16, rows * width);
        try r.s.synchronize();
        try b.download(0, std.mem.sliceAsBytes(out));
        return out;
    }

    /// A packed fp8g weight on the device from `st`.
    fn upload(r: *Run, st: Stored, n: usize, k: usize) !Projection {
        const p = try st.forLane(r.a, n, k);
        return .{ .fp8g = fp8.weight((try r.dev(p.codes)).ptr, (try r.dev(std.mem.sliceAsBytes(p.tiles))).ptr, n, k) };
    }

    /// B, W, A, F and R for `lane` (an fp8g weight of `st`) against project() on its bf16 dequantization.
    fn compare(r: *Run, what: []const u8, s: Shape, st: Stored, lane: Projection) !void {
        const x = try r.a.alloc(u16, max_rows * s.k);
        for (x) |*v| v.* = bf16Of(r.rng.float(f64) * 2 - 1);
        const ref = try Reference.of(r.a, x, st, s.n, s.k);
        const dx = try r.dev(std.mem.sliceAsBytes(x));
        const bf16w: Projection = .{ .bf16 = try r.dev(std.mem.sliceAsBytes(ref.dequantized)) };
        var lanes: [row_counts.len][]u16 = undefined;
        for (row_counts, &lanes) |m, *lane_rows| {
            const got = try r.room(m * s.n * 2);
            const want = try r.room(m * s.n * 2);
            try got.fill8(0xff, r.s.handle); // a bf16 NaN: a call that writes nothing cannot pass
            try want.fill8(0xff, r.s.handle);
            try r.p.project(r.s, m, s.kind, dx, lane, got);
            try r.p.project(r.s, m, s.kind, dx, bf16w, want);
            lane_rows.* = try r.back(got, m, s.n);
            const l = ref.measure(lane_rows.*, m, s.n);
            const c = ref.measure(try r.back(want, m, s.n), m, s.n);
            const path = if (m == 1 and r.p.gemv_exact[s.kind]) "gemv" else "cuBLASLt";
            std.debug.print("RESULT {s} {s} [{d}, {d}] m={d}: lane worst column {e:.3} (column {d}), aggregate {e:.3}; " ++
                "project() on {s} worst column {e:.3} (column {d}), aggregate {e:.3}\n", .{
                what, s.field, s.n, s.k, m, l.worst, l.column, l.aggregate, path, c.worst, c.column, c.aggregate,
            });
            r.expect(l.finite and c.finite, "F {s} {s} m={d}: lane finite {} comparator finite {}", .{ what, s.field, m, l.finite, c.finite });
            r.expect(l.worst <= 1, "B {s} {s} m={d}: lane worst column {e:.3} of its tolerance", .{ what, s.field, m, l.worst });
            r.expect(l.worst <= c.worst, "W {s} {s} m={d}: lane worst column {e:.3} above the comparator's {e:.3}", .{ what, s.field, m, l.worst, c.worst });
            r.expect(std.math.isFinite(l.aggregate) and l.aggregate <= c.aggregate, "A {s} {s} m={d}: lane aggregate {e:.3}, the comparator's {e:.3}", .{ what, s.field, m, l.aggregate, c.aggregate });
        }
        for (row_counts[0 .. row_counts.len - 1], lanes[0 .. lanes.len - 1]) |m, rows| {
            const wide = lanes[lanes.len - 1][0 .. m * s.n];
            r.expect(std.mem.eql(u16, rows, wide), "R {s} {s}: the lane's {d}-row call differs from its {d}-row call's rows", .{ what, s.field, m, max_rows });
        }
    }

    /// L for a loaded fp8g weight: its device codes and scale tiles against the host packers on the stored bytes.
    fn loaded(r: *Run, layer: usize, s: Shape, st: Stored, w: Projection) !void {
        const p = try st.forLane(r.a, s.n, s.k);
        const q = w.fp8g;
        const codes = try r.a.alloc(u8, p.codes.len);
        const tiles = try r.a.alloc(f32, p.tiles.len);
        try r.s.synchronize();
        try (Buffer{ .d = r.gpu.d, .ptr = q.codes, .len = codes.len, .borrowed = true }).download(0, codes);
        try (Buffer{ .d = r.gpu.d, .ptr = q.scales, .len = tiles.len * 4, .borrowed = true }).download(0, std.mem.sliceAsBytes(tiles));
        r.expect(q.n == s.n and q.k == s.k and std.mem.eql(u8, codes, p.codes) and std.mem.eql(f32, tiles, p.tiles), "L layer {d} {s}: packed weight differs from the host packers'", .{ layer, s.field });
    }
};

/// prefillTraced's tracer: each layer's projection inputs and outputs, copied to the host.
const Tracer = struct {
    a: Allocator,
    rows: [13][shapes.len][2][]u16 = undefined,

    pub fn capture(t: *Tracer, stream: cuda.Stream, layer: usize, stage: tower.Stage, src: Buffer, row0: usize) !void {
        _ = row0;
        inline for (shapes, 0..) |s, j| inline for (.{ s.input, s.output }, 0..) |which, side| if (stage == which) {
            const out = try t.a.alloc(u16, src.len / 2);
            try src.downloadAsync(0, std.mem.sliceAsBytes(out), stream.handle);
            try stream.synchronize();
            t.rows[layer][j][side] = out;
        };
    }
};

/// M: prefills of the mixed layers, each projection replayed on its captured input.
fn mixedLayers(r: *Run, model: *tower.Model) !void {
    const p = r.p;
    const hidden_rows = try r.a.alloc(u16, max_rows * hidden);
    for (hidden_rows) |*v| v.* = bf16Of(r.rng.float(f64) * 2 - 1);
    const dh = try r.dev(std.mem.sliceAsBytes(hidden_rows));
    var frames: [max_rows + 1]tower.Frame = undefined;
    for (&frames) |*f| {
        f[0] = p.semantic_start + r.rng.uintLessThan(u32, 4096);
        for (f[1..]) |*c| c.* = r.rng.uintLessThan(u32, 4096);
    }
    const x = try r.room(max_rows * max_k * 2);
    const y = try r.room(max_rows * max_n * 2);
    const direct = try r.room(max_rows * max_n * 2);
    const lane: cuda.qlinear.Linear = .{ .lane = &p.fp8.lane.? };
    for (row_counts) |n| {
        var t: Tracer = .{ .a = r.a };
        try model.reset(r.s, &.{}, 0);
        _ = try model.setInput(r.s, frames[0..n]);
        try model.prefillTraced(r.s, .{ .d = dh.d, .ptr = dh.ptr, .len = n * hidden * 2, .borrowed = true }, frames[1 .. n + 1], &t);
        var fp8g: usize = 0;
        for (p.layers, 0..) |l, i| inline for (shapes, 0..) |s, j| {
            const w: Projection = @field(l, s.field);
            const in = t.rows[i][j][0];
            const out = t.rows[i][j][1];
            try x.upload(0, std.mem.sliceAsBytes(in));
            try y.fill8(0xff, r.s.handle);
            try p.project(r.s, n, s.kind, x, w, y);
            var finite = true;
            for (out) |v| finite = finite and std.math.isFinite(bf(v));
            r.expect(finite and std.mem.eql(u16, out, try r.back(y, n, s.n)), "M rows {d} layer {d} {s} ({s}): finite {}, or project() on its input differs", .{ n, i, s.field, @tagName(w), finite });
            if (w == .fp8g) {
                fp8g += 1;
                try direct.fill8(0xff, r.s.handle);
                try lane.decode(r.s, .{ .fp8g = w.fp8g }, .{ .x = x.ptr, .part = if (p.fp8.part) |b| b.ptr else 0 }, direct.ptr, n);
                r.expect(std.mem.eql(u16, out, try r.back(direct, n, s.n)), "M rows {d} layer {d} {s}: the stage differs from a direct lane call", .{ n, i, s.field });
            }
        };
        std.debug.print("RESULT mixed rows {d}: {d} projections replayed, {d} of them fp8g\n", .{ n, p.layers.len * shapes.len, fp8g });
    }
}

pub fn run(gpu: check.Gpu, checkpoint: []const u8, layers: ?[]const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    std.debug.print("{s}\n", .{criteria});
    const capability = try gpu.ctx.capability();
    // D10: the lane module's own device memory, by the driver's free count around one load
    const free0 = (try gpu.ctx.memInfo()).free;
    var probe = try cuda.qmmf.Lane.load(gpu.d, capability / 10);
    const free1 = (try gpu.ctx.memInfo()).free;
    probe.unload();
    std.debug.print("RESULT qmmf lane module: {d} bytes of device memory (cuMemGetInfo free before - after load)\n", .{free0 -| free1});

    var model = try tower.Model.init(gpu.gpa, gpu.io, @constCast(gpu.ctx), checkpoint, "", .{ .context = 2 * max_rows, .max_rows = max_rows });
    defer model.deinit();
    var s = try cuda.Stream.init(gpu.d, false); // blocking: its work waits for the synchronous pageable uploads
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xF8_70E5);
    var r: Run = .{ .gpu = gpu, .a = a, .s = s, .p = model.impl, .rng = prng.random() };
    defer {
        s.synchronize() catch {};
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const e = r.p.gemv_exact;
    std.debug.print("RESULT gemv_exact wqkv {} wo {} gate_up {} down {}\n", .{ e[1], e[2], e[3], e[4] });

    if (layers) |list| {
        var ck = try core.checkpoint.Checkpoint.openModelPrefix(gpu.gpa, gpu.io, checkpoint, "audio_decoder");
        defer ck.close();
        var it = std.mem.tokenizeScalar(u8, list, ',');
        while (it.next()) |text| {
            const layer = try std.fmt.parseInt(usize, text, 10);
            if (layer >= r.p.layers.len) return error.BadTowerLayer;
            inline for (shapes) |sh| {
                const w: Projection = @field(r.p.layers[layer], sh.field);
                var name: [32]u8 = undefined;
                const what = try std.fmt.bufPrint(&name, "layer {d}", .{layer});
                if (try Stored.fromCheckpoint(a, &ck, layer, sh)) |st| {
                    r.expect(w == .fp8g, "L layer {d} {s}: stored F8_E4M3, loaded {s}", .{ layer, sh.field, @tagName(w) });
                    if (w == .fp8g) {
                        try r.loaded(layer, sh, st, w);
                        try r.compare(what, sh, st, w);
                    }
                } else {
                    r.expect(w == .bf16, "L layer {d} {s}: stored BF16, loaded {s}", .{ layer, sh.field, @tagName(w) });
                    std.debug.print("SKIP layer {d} {s}: stored BF16\n", .{ layer, sh.field });
                }
            }
        }
    }

    // synthetic weights at the four shapes; the plain set also goes into the mixed layers
    var plain: [shapes.len]struct { fp8g: Projection, bf16: Projection } = undefined;
    inline for (shapes, &plain) |sh, *w| {
        const st = try Stored.synthetic(a, r.rng, sh.n, sh.k, false);
        w.fp8g = try r.upload(st, sh.n, sh.k);
        const dq = try a.alloc(u16, sh.n * sh.k);
        for (dq, 0..) |*v, i| v.* = bf16Of(e4m3(st.codes[i]) * @as(f64, st.scales[i / fp8.group]));
        w.bf16 = .{ .bf16 = try r.dev(std.mem.sliceAsBytes(dq)) };
    }
    for (mixed) |m| inline for (shapes, 0..) |sh, j| switch (m[1][j]) {
        'F' => @field(r.p.layers[m[0]], sh.field) = plain[j].fp8g,
        'B' => @field(r.p.layers[m[0]], sh.field) = plain[j].bf16,
        else => {},
    };
    try r.p.fp8.prepare(r.p, gpu.ctx);
    std.debug.print("RESULT lane slice scratch: {d} bytes\n", .{if (r.p.fp8.part) |b| b.len else 0});
    inline for (shapes) |sh| {
        const st = try Stored.synthetic(a, r.rng, sh.n, sh.k, true);
        try r.compare("synthetic", sh, st, try r.upload(st, sh.n, sh.k));
    }
    const zero = try Stored.synthetic(a, r.rng, shapes[1].n, shapes[1].k, false);
    @memset(zero.codes, 0);
    try r.compare("all-zero", shapes[1], zero, try r.upload(zero, shapes[1].n, shapes[1].k));
    try mixedLayers(&r, &model);
    try check.expect(r.failed == 0, "drama-fp8-tower: {d} criteria failed", .{r.failed});
    check.pass("drama-fp8-tower: lane within float64 tolerance and no worse than project(), row counts bit-equal, mixed layers exact", .{});
}
