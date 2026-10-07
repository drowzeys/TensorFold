//! tf-glm53-layers: GLM-5.3's first N layers in Zig against the Python engine's fixtures (tools/glm53/oracle.py),
//! bit for bit, as rank `rank` of `world` (meta.json) with the other ranks absent (see cuda_forward.zig). Usage: tf-glm53-layers <model dir> <aot dir> <fixtures dir> [--layers N] [--chain short|long]
//!
//! For each chain of windows in meta.json (short: a 24-row prompt window then 1- and 3-row decode windows from an
//! empty state; long: the oracle's 20,000-token caches loaded, then 1- and 3-row decode windows that select keys),
//! every layer's output rows, the cache rows it wrote, its index keys and its selection must equal the oracle's
//! bytes (gating); the end-of-layer scratch is compared as diagnostics (the first differing buffer says where a
//! mismatch starts). The prepared weights' digests are checked against the oracle's too (gating).
//! Prints PASS / FAIL lines and a final `RESULT glm53-layers PASS|FAIL ...`; exit 0 only on PASS.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm53");

const usage = "usage: tf-glm53-layers <model dir> <aot dir> <fixtures dir> [--layers N] [--chain short|long]\n";

const Tally = struct {
    gated: usize = 0,
    gated_bad: usize = 0,
    diag: usize = 0,
    diag_bad: usize = 0,
};

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var layers_arg: ?usize = null;
    var only: ?[]const u8 = null;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--layers") and i + 1 < args.len) {
            layers_arg = try std.fmt.parseInt(usize, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--chain") and i + 1 < args.len) {
            only = args[i + 1];
            i += 1;
        } else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    const ok = run(init.gpa, init.io, args[1], args[2], args[3], layers_arg, only) catch |e| {
        std.debug.print("RESULT glm53-layers FAIL error {t}\n", .{e});
        return 1;
    };
    return if (ok) 0 else 1;
}

fn readJson(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    return std.json.parseFromSlice(std.json.Value, gpa, text, .{});
}

fn openFixture(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !core.safetensors.File {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return core.safetensors.File.open(gpa, io, path);
}

fn jint(v: std.json.Value) !usize {
    return switch (v) {
        .integer => |x| std.math.cast(usize, x) orelse error.BadMeta,
        else => error.BadMeta,
    };
}

fn run(gpa: std.mem.Allocator, io: std.Io, model: []const u8, aot_dir: []const u8, fx_dir: []const u8, layers_arg: ?usize, only: ?[]const u8) !bool {
    var meta = try readJson(gpa, io, fx_dir, "meta.json");
    defer meta.deinit();
    const mo = meta.value.object;
    const meta_layers = try jint(mo.get("layers") orelse return error.BadMeta);
    const n_layers = @min(layers_arg orelse meta_layers, meta_layers);
    const world = try jint(mo.get("world") orelse return error.BadMeta);
    const rank = try jint(mo.get("rank") orelse return error.BadMeta);
    if (rank >= world) return error.BadMeta;
    std.debug.print("rank {d} of world {d}, the other ranks absent (local comm: own partial, zeros elsewhere)\n", .{ rank, world });
    if (mo.get("triton")) |v| if (v == .string) std.debug.print("oracle: torch {s}, triton {s}\n", .{ (mo.get("torch") orelse v).string, v.string });
    if (mo.get("radix_equal")) |v| if (v == .bool and !v.bool) std.debug.print("WARN the oracle's radix and torch.topk selections differ (meta.radix_differs)\n", .{});
    if (mo.get("tiles_are_plan")) |v| if (v == .bool and !v.bool) std.debug.print("WARN the oracle ran EXL3 tiles other than x3linear.plan's (meta.tiles)\n", .{});

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    std.debug.print("device: {s} sm_{d}\n", .{ try ctx.name(&name_buf), try ctx.capability() });

    const cfg = try glm.Config.load(gpa, io, model);
    var digests: ?*const std.json.ObjectMap = null;
    if (mo.getPtr("weights_sha256")) |v| {
        if (v.* == .object) digests = &v.object;
    }
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var w = try glm.weights.load(gpa, io, &driver, model, cfg, .{ .layers = n_layers, .digests = digests, .rank = rank, .world = world });
    defer w.deinit();
    const load_s: f64 = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
    var tally: Tally = .{};
    tally.gated += 1;
    if (w.digest_bad > 0 or (digests != null and w.digest_checked == 0)) {
        tally.gated_bad += 1;
        std.debug.print("FAIL weights: {d} of {d} digests differ from the oracle's\n", .{ w.digest_bad, w.digest_checked });
    } else std.debug.print("PASS weights: {d} digests equal ({d} layers loaded in {d:.1}s)\n", .{ w.digest_checked, n_layers, load_s });

    var k = try glm.kernels.Kernels.load(gpa, io, &ctx, aot_dir);
    defer k.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();

    // fused.inv_freq: the oracle's bytes (torch computes them on the GPU); the host formula as a diagnostic
    var common = try openFixture(gpa, io, fx_dir, "common.safetensors");
    defer common.close(io);
    const inv_t = common.get("inv") orelse return error.BadFixture;
    var inv_buf = try cuda.DeviceBuffer.fromHost(&driver, inv_t.bytes);
    defer inv_buf.free();
    {
        var host: [64]f32 = undefined;
        const n = cfg.rope / 2;
        glm.forward.invFreq(host[0..n], cfg.rope, cfg.rope_theta);
        tally.diag += 1;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(host[0..n]), inv_t.bytes)) {
            tally.diag_bad += 1;
            std.debug.print("diag inv_freq: the host formula differs from torch's bytes (the oracle's are used)\n", .{});
        }
    }
    const fwd: glm.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv_buf.ptr };

    const chains = (mo.get("chains") orelse return error.BadMeta).object;
    for ([_][]const u8{ "short", "long" }) |chain_name| {
        if (only) |o| if (!std.mem.eql(u8, o, chain_name)) continue;
        const ch = (chains.get(chain_name) orelse continue).object;
        try runChain(gpa, io, &driver, &fwd, &w, fx_dir, chain_name, ch, n_layers, mo, &tally);
    }

    const pass = tally.gated_bad == 0 and tally.gated > 1;
    std.debug.print("RESULT glm53-layers {s} gated {d}/{d} equal, diagnostics {d}/{d} equal, layers {d}\n", .{ if (pass) "PASS" else "FAIL", tally.gated - tally.gated_bad, tally.gated, tally.diag - tally.diag_bad, tally.diag, n_layers });
    return pass;
}

fn runChain(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, fwd: *const glm.Forward, w: *const glm.weights.Weights, fx_dir: []const u8, chain: []const u8, ch: std.json.ObjectMap, n_layers: usize, mo: std.json.ObjectMap, tally: *Tally) !void {
    const c = w.config;
    var file = try openFixture(gpa, io, fx_dir, (ch.get("file") orelse return error.BadMeta).string);
    defer file.close(io);
    const cap = try jint(ch.get("capacity") orelse return error.BadMeta);
    const rows = try jint(ch.get("buffer_rows") orelse return error.BadMeta);
    const score_cols = try jint(ch.get("score_cols") orelse return error.BadMeta);
    var st = try glm.State.init(gpa, d, w, cap);
    defer st.deinit();
    var b = try glm.Buffers.init(gpa, d, w, rows, score_cols);
    defer b.deinit();
    const ops: glm.kernels.Ops = .{ .k = fwd.k, .s = fwd.s };

    // the long chain starts from the oracle's caches of its prefix
    if (ch.get("prefix")) |pv| {
        const P = try jint(pv);
        var nb: [64]u8 = undefined;
        for (0..n_layers) |li| {
            const kc = file.get(try std.fmt.bufPrint(&nb, "in.kc.{d}", .{li})) orelse return error.BadFixture;
            if (kc.bytes.len != P * c.latentWidth() * 2) return error.BadFixture;
            try ops.upload(st.kc[li], kc.bytes);
            if (st.ic[li] != 0) {
                const ic = file.get(try std.fmt.bufPrint(&nb, "in.ic.{d}", .{li})) orelse return error.BadFixture;
                try ops.upload(st.ic[li], ic.bytes);
            }
        }
        try fwd.s.synchronize();
        std.debug.print("{s}: loaded the oracle's caches of {d} positions\n", .{ chain, P });
    }
    const ids_t = file.get("ids") orelse return error.BadFixture;
    if (ids_t.dtype != .i64) return error.BadFixture;
    const n_ids = ids_t.numel();
    const ids = try gpa.alloc(u32, n_ids);
    defer gpa.free(ids);
    for (ids, 0..) |*x, j| x.* = @intCast(std.mem.readInt(i64, ids_t.bytes[8 * j ..][0..8], .little));

    const diag_names = (mo.get("diag") orelse return error.BadMeta).array.items;
    const windows = (ch.get("windows") orelse return error.BadMeta).array.items;
    for (windows) |wv| {
        const wo = wv.object;
        const name = (wo.get("name") orelse return error.BadMeta).string;
        const a = try jint(wo.get("a") orelse return error.BadMeta);
        const e = try jint(wo.get("e") orelse return error.BadMeta);
        const t_meta = try jint(wo.get("T") orelse return error.BadMeta);
        const T: ?usize = if (t_meta == 0) null else t_meta;
        if (!std.meta.eql(T, glm.triton.bucket(e, c.index_topk))) return error.BucketDiffers;
        const R = e - a;
        if (R > b.rows) return error.WindowTooWide;
        try fwd.embed(&b, ids[a..e]);
        try fwd.setPos(&st, a);
        var win_bad: usize = 0;
        for (0..n_layers) |li| {
            try fwd.layer(&b, &st, li, R, T);
            try fwd.s.synchronize();
            const lat = c.latentWidth();
            const g = [_]struct { what: []const u8, ptr: u64 }{
                .{ .what = "x", .ptr = b.x },
                .{ .what = "kc", .ptr = st.kc[li] + a * lat * 2 },
                .{ .what = "ic", .ptr = if (st.ic[li] != 0) st.ic[li] + a * c.index_dim * 2 else 0 },
                .{ .what = "tok", .ptr = b.tok },
            };
            for (g) |item| {
                if (item.ptr == 0) continue;
                var kb: [96]u8 = undefined;
                const key = try std.fmt.bufPrint(&kb, "{s}.{s}.{d}", .{ name, item.what, li });
                const want = file.get(key) orelse continue; // tok: only layers that selected; ic: indexer layers
                tally.gated += 1;
                if (!try same(gpa, ops, item.ptr, want, key)) {
                    tally.gated_bad += 1;
                    win_bad += 1;
                }
            }
            for (diag_names) |dn| {
                const dname = dn.string;
                const ptr = diagPtr(&b, dname) orelse continue;
                var kb: [96]u8 = undefined;
                const key = try std.fmt.bufPrint(&kb, "{s}.diag.{s}.{d}", .{ name, dname, li });
                const want = file.get(key) orelse continue;
                tally.diag += 1;
                if (!try same(gpa, ops, ptr, want, key)) tally.diag_bad += 1;
            }
        }
        std.debug.print("{s} {s}/{s}: rows {d} at {d}, T {?d}: {d} layers, {d} gated mismatches\n", .{ if (win_bad == 0) "PASS" else "FAIL", chain, name, R, a, T, n_layers, win_bad });
    }
}

fn diagPtr(b: *const glm.Buffers, name: []const u8) ?u64 {
    const map = .{
        .{ "normed", "normed" }, .{ "qa", "qa" },     .{ "qn", "qn" },   .{ "kva", "kva" },   .{ "q", "q" },
        .{ "qlat", "qlat" },     .{ "qrot", "qrot" }, .{ "ol", "ol" },   .{ "o", "o" },       .{ "ik", "ik" },
        .{ "iw", "iw" },         .{ "iq", "iq" },     .{ "mlog", "mlog" }, .{ "pick", "pick" }, .{ "wts", "wts" },
        .{ "sy", "sy" },         .{ "part", "part" }, .{ "g", "g" },     .{ "u", "u" },       .{ "act", "act" },
    };
    inline for (map) |m| if (std.mem.eql(u8, name, m[0])) return @field(b, m[1]);
    return null;
}

/// The device bytes at `ptr` against the fixture tensor; on a mismatch, where it starts and how far it goes.
fn same(gpa: std.mem.Allocator, ops: glm.kernels.Ops, ptr: u64, want: core.safetensors.Tensor, key: []const u8) !bool {
    const got = try gpa.alloc(u8, want.bytes.len);
    defer gpa.free(got);
    try ops.download(got, ptr);
    try ops.s.synchronize();
    if (std.mem.eql(u8, got, want.bytes)) return true;
    const es = want.dtype.size();
    var first: ?usize = null;
    var n_diff: usize = 0;
    var max_abs: f64 = 0;
    var j: usize = 0;
    while (j < got.len) : (j += es) {
        if (std.mem.eql(u8, got[j..][0..es], want.bytes[j..][0..es])) continue;
        n_diff += 1;
        if (first == null) first = j / es;
        const gv = value(want.dtype, got[j..][0..es]);
        const wv = value(want.dtype, want.bytes[j..][0..es]);
        if (gv != null and wv != null) max_abs = @max(max_abs, @abs(gv.? - wv.?));
    }
    std.debug.print("  differs {s}: {d} of {d} elements ({t}), first at {d}, max |diff| {e}\n", .{ key, n_diff, got.len / es, want.dtype, first.?, max_abs });
    return false;
}

fn value(dt: core.safetensors.DType, b: []const u8) ?f64 {
    return switch (dt) {
        .bf16 => @as(f64, @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, b[0..2], .little)) << 16))),
        .f16 => @as(f64, @as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little)))),
        .f32 => @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, b[0..4], .little)))),
        .i32 => @floatFromInt(std.mem.readInt(i32, b[0..4], .little)),
        .i64 => @floatFromInt(std.mem.readInt(i64, b[0..8], .little)),
        else => null,
    };
}
