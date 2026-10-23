//! C3 per-input importance of a routed expert's projection, and C6's shared-expert proxy, from tf-drama-probe's sums.
const std = @import("std");
const core = @import("core");

const st = core.safetensors;
const Allocator = std.mem.Allocator;

/// Pseudo-rows of prior: a rarely routed expert's importance leans on its prior, a busy one on its own sums.
pub const kappa = 64;
pub const classes = [_][]const u8{ "prompt_text", "prompt_audio", "generated" };

/// One layer's routed pairs n_e and, per class, its experts' sums of squares of the gate/up and the down inputs.
const Layer = struct {
    n: []f64,
    /// m_c = Σ_e S_e,c / Σ_e n_e of the shared gate/up input; null when no pair reached the layer.
    mean: ?[]f64,
    gate_up: [classes.len]st.Tensor,
    down: [classes.len]st.Tensor,
};

pub const Importance = struct {
    file: st.File,
    layers: []?Layer,
    /// Routed pairs per class over the layers read: the mixture the pooled sums stand for.
    pairs: [classes.len]f64,

    /// The statistics of every layer `wanted` marks, checked to cover `experts[l]` experts; names live in `arena`.
    pub fn open(gpa: Allocator, arena: Allocator, io: std.Io, path: []const u8, wanted: []const bool, experts: []const usize) !Importance {
        var file = try st.File.open(gpa, io, path);
        errdefer file.close(io);
        const layers = try arena.alloc(?Layer, wanted.len);
        @memset(layers, null);
        var pairs: [classes.len]f64 = @splat(0);
        for (wanted, experts, 0..) |want, e, l| if (want) {
            var layer: Layer = .{ .n = try arena.alloc(f64, e), .mean = null, .gate_up = undefined, .down = undefined };
            @memset(layer.n, 0);
            for (classes, &pairs, &layer.gate_up, &layer.down) |c, *p, *gu, *dn| {
                const count = try get(&file, arena, l, "count", c);
                if (!count.is(.f64, &.{e})) return error.StatsShape;
                try valid(count);
                for (layer.n, 0..) |*x, i| x.* += at(count, i);
                for (0..e) |i| p.* += at(count, i);
                gu.* = try get(&file, arena, l, "gate_up.sumsq", c);
                dn.* = try get(&file, arena, l, "down.sumsq", c);
                if (gu.rank != 2 or gu.shape[0] != e or dn.rank != 2 or dn.shape[0] != e) return error.StatsShape;
                if (gu.shape[1] != layer.gate_up[0].shape[1] or dn.shape[1] != layer.down[0].shape[1]) return error.StatsShape;
                try valid(gu.*);
                try valid(dn.*);
            }
            for (layer.n) |x| if (!std.math.isFinite(x)) return error.BadStats;
            var total: f64 = 0;
            for (layer.n) |x| total += x;
            if (!std.math.isFinite(total)) return error.BadStats;
            if (total > 0) {
                const k = layer.gate_up[0].shape[1];
                const m = try arena.alloc(f64, k);
                for (m, 0..) |*x, c| {
                    var s: f64 = 0;
                    for (0..e) |i| s += pooled(layer.gate_up, i * k + c);
                    x.* = s / total;
                    if (!std.math.isFinite(x.*)) return error.BadStats;
                }
                layer.mean = m;
            }
            layers[l] = layer;
        };
        for (pairs) |p| if (!std.math.isFinite(p)) return error.BadStats;
        return .{ .file = file, .layers = layers, .pairs = pairs };
    }

    pub fn close(self: *Importance, io: std.Io) void {
        self.file.close(io);
    }

    /// Whether `fill` can serve layer `l`'s `k`-input gate/up (or down) projections.
    pub fn covers(self: *const Importance, l: usize, gate_up: bool, k: usize) bool {
        const layer = self.layers[l] orelse return false;
        return (if (gate_up) layer.gate_up else layer.down)[0].shape[1] == k;
    }

    /// h [k] of expert `e`'s gate/up (shared input, layer-mean prior) or down (its own uniform prior) projection.
    pub fn fill(self: *const Importance, l: usize, e: usize, gate_up: bool, h: []f64) void {
        const layer = self.layers[l].?;
        const sums = if (gate_up) layer.gate_up else layer.down;
        const k = h.len;
        const n = layer.n[e];
        if (gate_up) {
            const mean = layer.mean orelse return @memset(h, 1);
            for (h, mean, 0..) |*x, m, c| x.* = (pooled(sums, e * k + c) + kappa * m) / (n + kappa);
            return;
        }
        if (n == 0) return @memset(h, 1);
        var s: f64 = 0;
        for (0..k) |c| s += pooled(sums, e * k + c);
        const prior = s / @as(f64, @floatFromInt(k)) / n;
        for (h, 0..) |*x, c| x.* = (pooled(sums, e * k + c) + kappa * prior) / (n + kappa);
    }

    /// h [k] of layer `l`'s shared expert gate/up: its routed experts' gate/up sums, which read the same input row.
    pub fn shared(self: *const Importance, l: usize, h: []f64) void {
        const layer = self.layers[l].?;
        const k = h.len;
        for (h, 0..) |*x, c| {
            x.* = 0;
            for (0..layer.n.len) |e| x.* += pooled(layer.gate_up, e * k + c);
        }
    }
};

fn get(file: *const st.File, arena: Allocator, l: usize, what: []const u8, class: []const u8) !st.Tensor {
    const t = file.get(try std.fmt.allocPrint(arena, "layers.{d}.{s}.{s}", .{ l, what, class })) orelse return error.StatsMissingLayer;
    return if (t.dtype == .f64) t else error.StatsShape;
}

/// Every element finite and non-negative: a corrupt count or sum must refuse, not turn into the no-observation prior.
fn valid(t: st.Tensor) !void {
    for (0..t.bytes.len / 8) |i| {
        const x = at(t, i);
        if (!std.math.isFinite(x) or x < 0) return error.BadStats;
    }
}

/// Element i of an F64 tensor; the data region need not be 8-byte aligned.
fn at(t: st.Tensor, i: usize) f64 {
    return @bitCast(std.mem.readInt(u64, t.bytes[i * 8 ..][0..8], .little));
}

/// Element i summed over the classes in their listed order: pairs pool by count.
fn pooled(ts: [classes.len]st.Tensor, i: usize) f64 {
    var s: f64 = 0;
    for (ts) |t| s += at(t, i);
    return s;
}
