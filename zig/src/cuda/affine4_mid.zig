//! qmm_midrow.cu's prompt tiles: qmm_prefill's bits a projection on tiles sized to the prompt's rows and columns.
const std = @import("std");
const linear = @import("qlinear.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch_ = @import("launch.zig");
const Affine4 = linear.Affine4;

/// One tile: rows and columns a CTA, its threads and ring stages, and an output's relative time on GB10.
pub const Tile = struct {
    rows: u16,
    cols: u16,
    threads: u16,
    stages: u16,
    cost: u16 = 0, // a wide tile's time an output, fitted on GB10 to Qwen3.8-27B's projections; 0: not a wide tile

    /// Mid<BM, BN, WM, WN, STAGES>::SMEM: inputs, weights, scales and biases a 64-input stage.
    pub fn smem(t: Tile) u32 {
        return @as(u32, t.stages) * (@as(u32, t.rows) * 128 + @as(u32, t.cols) * 32 + @as(u32, t.cols) * 4);
    }
};

/// qmm_midrow.cu's TF_QMM_MID lines, in its order.
pub const tiles = [_]Tile{
    .{ .rows = 16, .cols = 32, .threads = 128, .stages = 10 },
    .{ .rows = 16, .cols = 64, .threads = 128, .stages = 8 },
    .{ .rows = 32, .cols = 64, .threads = 128, .stages = 6 },
    .{ .rows = 32, .cols = 128, .threads = 128, .stages = 4 },
    .{ .rows = 32, .cols = 256, .threads = 256, .stages = 3, .cost = 320 },
    .{ .rows = 48, .cols = 256, .threads = 256, .stages = 3, .cost = 230 },
    .{ .rows = 64, .cols = 128, .threads = 256, .stages = 4, .cost = 290 },
    .{ .rows = 80, .cols = 256, .threads = 256, .stages = 2, .cost = 200 },
    .{ .rows = 96, .cols = 128, .threads = 256, .stages = 2, .cost = 220 },
    .{ .rows = 128, .cols = 256, .threads = 256, .stages = 2, .cost = 235 },
};

fn at(rows: u16, cols: u16) usize {
    for (tiles, 0..) |t, i| if (t.rows == rows and t.cols == cols) return i;
    unreachable;
}

/// qmm_prefill's 128 by 128 tile at the same fitted cost: pick keeps it where no wide tile is quicker.
pub const base: Tile = .{ .rows = 128, .cols = 128, .threads = 256, .stages = 3, .cost = 235 };

/// The wide tiles, tallest first: on equal cost the taller tile wins.
const wide = [_]usize{ at(128, 256), at(96, 128), at(80, 256), at(64, 128), at(48, 256), at(32, 256) };

/// Waves of one CTA an SM (co-resident CTAs share it) times a CTA's outputs and its tile's cost.
fn estimate(t: Tile, rows: usize, n: usize, sms: usize) usize {
    const ctas = (rows + t.rows - 1) / t.rows * ((n + t.cols - 1) / t.cols);
    return (ctas + sms - 1) / sms * @as(usize, t.rows) * t.cols * t.cost;
}

/// The tile for `rows` rows of `n` columns on `sms` SMs, null where qmm_prefill's 128-row tile is as quick.
pub fn pick(rows: usize, n: usize, sms: usize) ?usize {
    if (n <= 128) return if (rows <= 768) at(16, 32) else at(32, 64);
    if (n <= 2048 and rows <= 384) return if (rows <= 64) at(16, 64) else if (rows <= 192) at(32, 64) else at(32, 128);
    var best = wide[0];
    var least: usize = std.math.maxInt(usize);
    for (wide) |i| {
        const cost = estimate(tiles[i], rows, n, sms);
        if (cost < least) {
            best = i;
            least = cost;
        }
    }
    return if (least < estimate(base, rows, n, sms)) best else null;
}

/// The loaded tiles; Affine4Kernels.prompt takes them when an engine attaches them.
pub const Mid = struct {
    fns: [tiles.len]Function,
    sms: usize,

    pub fn resolve(mod: Module, sms: usize) !Mid {
        var m: Mid = undefined;
        m.sms = sms;
        for (tiles, &m.fns) |t, *f| {
            var name: [32]u8 = undefined;
            f.* = try mod.function(try std.fmt.bufPrintSentinel(&name, "tf_qmm_mid_{d}x{d}", .{ t.rows, t.cols }, 0));
            try f.allowDynamicShared(t.smem());
        }
        return m;
    }

    /// Affine4Kernels.prompt on the picked tile; false where the 128-row tile should run instead.
    pub fn prompt(m: *const Mid, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !bool {
        const i = pick(rows, q.n, m.sms) orelse return false;
        try m.launchTile(i, s, x, q, out, rows);
        return true;
    }

    /// One tile's launch, whatever the rows (tests run every tile at every shape).
    pub fn launchTile(m: *const Mid, i: usize, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !void {
        return launchWith(m.fns[i], tiles[i], s, x, q, out, rows);
    }
};

/// `f` (a tile's kernel, or a control built on its shape) over `t`'s grid and row bands.
pub fn launchWith(f: Function, t: Tile, s: Stream, x: u64, q: Affine4, out: u64, rows: usize) !void {
    const rows_t = (rows + t.rows - 1) / t.rows;
    const band = (12 << 20) / (@as(usize, t.rows) * q.k * 2);
    const group = @max(1, @min(rows_t, band));
    var g: launch_.Args = .{};
    for ([_]u64{ x, q.w, q.s, q.b, out }) |v| g.add(v);
    for ([_]usize{ rows, q.n, q.k, q.npad, q.k, group }) |v| g.add(@as(c_int, @intCast(v)));
    const grid = rows_t * ((q.n + t.cols - 1) / t.cols);
    try launch_.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = t.threads }, .shared = t.smem() }, s, &g);
}

test "mid tiles: qmm_midrow.cu's shared bytes and the picks measured on GB10" {
    const smem = [_]u32{ 32000, 34816, 38400, 34816, 39936, 46080, 51200, 38912, 33792, 51200 };
    for (tiles, smem) |t, want| try std.testing.expectEqual(want, t.smem());
    const Case = struct { rows: usize, n: usize, tile: ?[2]u16 };
    const cases = [_]Case{
        .{ .rows = 160, .n = 48, .tile = .{ 16, 32 } },       .{ .rows = 2048, .n = 48, .tile = .{ 32, 64 } },
        .{ .rows = 48, .n = 1024, .tile = .{ 16, 64 } },      .{ .rows = 160, .n = 1024, .tile = .{ 32, 64 } },
        .{ .rows = 300, .n = 1024, .tile = .{ 32, 128 } },    .{ .rows = 512, .n = 1024, .tile = .{ 96, 128 } },
        .{ .rows = 17, .n = 17408, .tile = .{ 32, 256 } },    .{ .rows = 64, .n = 5120, .tile = .{ 64, 128 } },
        .{ .rows = 96, .n = 10240, .tile = .{ 96, 128 } },    .{ .rows = 128, .n = 17408, .tile = null },
        .{ .rows = 128, .n = 5120, .tile = null },            .{ .rows = 160, .n = 17408, .tile = .{ 80, 256 } },
        .{ .rows = 160, .n = 5120, .tile = .{ 80, 256 } },    .{ .rows = 192, .n = 6144, .tile = .{ 96, 128 } },
        .{ .rows = 224, .n = 12288, .tile = .{ 80, 256 } },   .{ .rows = 224, .n = 5120, .tile = null },
        .{ .rows = 288, .n = 17408, .tile = .{ 96, 128 } },   .{ .rows = 320, .n = 5120, .tile = .{ 80, 256 } },
        .{ .rows = 2048, .n = 248320, .tile = .{ 80, 256 } },
    };
    for (cases) |c| {
        const got = pick(c.rows, c.n, 48);
        if (c.tile) |want| {
            try std.testing.expectEqual(want[0], tiles[got.?].rows);
            try std.testing.expectEqual(want[1], tiles[got.?].cols);
        } else try std.testing.expectEqual(@as(?usize, null), got);
    }
}
