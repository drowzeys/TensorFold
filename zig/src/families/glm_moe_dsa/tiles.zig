//! The served engine's EXL3 tile table (tiles.py, TF_GLM53_TILES=save:PATH): the (K splits, warps) each dense EXL3
//! linear of a rank runs with. The image default times tiles at every boot (fused.tune_linears / tune_groups, rank
//! 0's picks shared); a different pick sums in another order, so the Zig engine runs the reference boot's own table
//! to get its bits. Rows in fused.Weights.tunable order: every layer, then the MTP layer - q_a, kv_a, q_b, o_proj,
//! the MLP's (or shared expert's) gate, up, down, and the indexer's wq_b where there is one. Each row is
//! [k, n, bits, codebook, layout, K splits, warps]; a table whose shapes, widths or codebooks differ is refused.
const std = @import("std");
const exl3 = @import("exl3.zig");
const W = @import("cuda_weights.zig");

/// The weights' linears in tiles.rows order.
pub fn tunable(w: *W.Weights, out: *std.ArrayList(*exl3.Linear), gpa: std.mem.Allocator) !void {
    const Add = struct {
        fn layer(L: *W.Layer, o: *std.ArrayList(*exl3.Linear), g: std.mem.Allocator) !void {
            for ([_]*exl3.Linear{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down }) |l| try o.append(g, l);
            if (L.indexer) |*ix| try o.append(g, &ix.wq_b);
        }
    };
    for (w.layers) |*L| try Add.layer(L, out, gpa);
    if (w.mtp) |*m| try Add.layer(&m.layer, out, gpa);
}

fn codebookName(cb: exl3.Codebook) []const u8 {
    return switch (cb) {
        .@"3inst" => "3inst",
        .mcg => "mcg",
        .mul1 => "mul1",
    };
}

pub const Loaded = struct { linears: usize, changed: usize };

/// tiles.load: every linear's (sk, wk) from the table at `path`; nothing changes unless all rows match.
pub fn load(gpa: std.mem.Allocator, io: std.Io, w: *W.Weights, path: []const u8) !Loaded {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const rows = (parsed.value.object.get("linears") orelse return error.BadTileTable).array.items;
    var lins: std.ArrayList(*exl3.Linear) = .empty;
    defer lins.deinit(gpa);
    try tunable(w, &lins, gpa);
    if (rows.len != lins.items.len) {
        std.log.err("{s}: a table of {d} linears, this engine has {d}: refusing it", .{ path, rows.len, lins.items.len });
        return error.BadTileTable;
    }
    var tiles = try gpa.alloc([2]usize, rows.len);
    defer gpa.free(tiles);
    for (rows, lins.items, 0..) |rv, l, i| {
        const r = rv.array.items;
        if (r.len != 7) return error.BadTileTable;
        const k: usize = @intCast(r[0].integer);
        const n: usize = @intCast(r[1].integer);
        const bits: f64 = switch (r[2]) {
            .float => |f| f,
            .integer => |x| @floatFromInt(x),
            else => return error.BadTileTable,
        };
        const cb = r[3].string;
        if (k != l.k or n != l.n or @as(usize, @intFromFloat(bits * 2.0)) != l.k2 or !std.mem.eql(u8, cb, codebookName(l.cb))) {
            std.log.err("{s}: linear {d} is [{d}, {d}, {d}, {s}] in the table and [{d}, {d}, {d}, {s}] here: refusing it", .{ path, i, k, n, bits, cb, l.k, l.n, @as(f64, @floatFromInt(l.k2)) / 2.0, codebookName(l.cb) });
            return error.BadTileTable;
        }
        const sk: usize = @intCast(r[5].integer);
        const wk: usize = @intCast(r[6].integer);
        if (wk != 2 and wk != 4 and wk != 8) return error.UnsupportedExl3Warps;
        if ((k / 16) % (sk * wk) != 0) return error.BadSplit;
        tiles[i] = .{ sk, wk };
    }
    var changed: usize = 0;
    for (lins.items, tiles) |l, t| {
        if (l.sk != t[0] or l.wk != t[1]) changed += 1;
        l.sk = t[0];
        l.wk = t[1];
    }
    return .{ .linears = lins.items.len, .changed = changed };
}
