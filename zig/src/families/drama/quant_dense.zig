//! tf-drama-quant --dense-fp8: C5's classes, the --dense-fp8 and --keep-bf16 selectors, and C6 with its errors.
const std = @import("std");
const core = @import("core");

const st = core.safetensors;
const fp8 = core.quant.fp8;
const Allocator = std.mem.Allocator;

/// C5's classes, each one fused group of a layer: a group is converted or kept whole.
pub const Class = enum { gdn_in, gdn_out, attn_in, attn_out, shared_in, shared_out, tower_qkv, tower_o, tower_gate_up, tower_down };

/// Tower layers t:0-3 are audio_decoder_mtp_shared.layers.0-3, t:4 on audio_decoder_mtp.layers.0 on.
pub const tower_shared = 4;

/// A layer of the backbone (b:L) or of the tower (t:L), and the rest of a tensor name after its prefix.
pub const Layer = struct { tower: bool, index: usize, rest: []const u8 };

pub fn layerOf(name: []const u8) ?Layer {
    const prefixes = [_]struct { []const u8, bool, usize }{
        .{ "model.language_model.layers.", false, 0 },
        .{ "audio_decoder_mtp_shared.layers.", true, 0 },
        .{ "audio_decoder_mtp.layers.", true, tower_shared },
    };
    for (prefixes) |p| if (std.mem.startsWith(u8, name, p[0])) {
        const tail = name[p[0].len..];
        const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return null;
        const index = std.fmt.parseInt(usize, tail[0..dot], 10) catch return null;
        if (p[1] and p[2] == 0 and index >= tower_shared) return null;
        return .{ .tower = p[1], .index = p[2] + index, .rest = tail[dot + 1 ..] };
    };
    return null;
}

const Projection = struct { []const u8, Class };
const backbone_projections = [_]Projection{
    .{ "linear_attn.in_proj_qkv", .gdn_in },           .{ "linear_attn.in_proj_z", .gdn_in },            .{ "linear_attn.out_proj", .gdn_out },
    .{ "self_attn.q_proj", .attn_in },                 .{ "self_attn.k_proj", .attn_in },                .{ "self_attn.v_proj", .attn_in },
    .{ "self_attn.o_proj", .attn_out },                .{ "mlp.shared_expert.gate_proj", .shared_in },   .{ "mlp.shared_expert.up_proj", .shared_in },
    .{ "mlp.shared_expert.down_proj", .shared_out },
};
const tower_projections = [_]Projection{
    .{ "transformer_layer.attention.wqkv", .tower_qkv },        .{ "transformer_layer.attention.wo", .tower_o },
    .{ "transformer_layer.feed_forward.w1", .tower_gate_up },   .{ "transformer_layer.feed_forward.w3", .tower_gate_up },
    .{ "transformer_layer.feed_forward.w2", .tower_down },
};

/// A convertible projection's weight (C5 names): its class and layer.
pub const Name = struct { class: Class, layer: Layer };

pub fn parse(name: []const u8) ?Name {
    const l = layerOf(name) orelse return null;
    if (!std.mem.endsWith(u8, l.rest, ".weight")) return null;
    const projection = l.rest[0 .. l.rest.len - ".weight".len];
    const table: []const Projection = if (l.tower) &tower_projections else &backbone_projections;
    for (table) |p| if (std.mem.eql(u8, projection, p[0])) return .{ .class = p[1], .layer = l };
    return null;
}

/// Layers in SRC per namespace: the range a layer selector must fall in.
pub const Counts = struct { backbone: usize = 0, tower: usize = 0 };

/// Which projections become FP8: --dense-fp8's classes less --keep-bf16's classes and layers, as given and resolved.
pub const Selection = struct {
    dense: []const u8,
    keep: ?[]const u8,
    classes: std.EnumSet(Class),
    kept_classes: std.EnumSet(Class) = .empty,
    kept_backbone: []const bool,
    kept_tower: []const bool,

    /// `dense` is CLASS[,CLASS] or all; `keep` CLASS, b:L[,L] or t:L[,L] items, a bare L belonging to the item before.
    pub fn init(arena: Allocator, dense: []const u8, keep: ?[]const u8, counts: Counts) !Selection {
        var s: Selection = .{ .dense = dense, .keep = keep, .classes = .empty, .kept_backbone = undefined, .kept_tower = undefined };
        if (std.mem.eql(u8, dense, "all")) s.classes = .full else {
            var it = std.mem.splitScalar(u8, dense, ',');
            while (it.next()) |token| s.classes.insert(try class(token));
        }
        const backbone = try arena.alloc(bool, counts.backbone);
        const tower = try arena.alloc(bool, counts.tower);
        @memset(backbone, false);
        @memset(tower, false);
        s.kept_backbone = backbone;
        s.kept_tower = tower;
        var it = std.mem.splitScalar(u8, keep orelse return s, ',');
        var space: ?[]bool = null;
        while (it.next()) |token| {
            var number = token;
            if (std.mem.startsWith(u8, token, "b:") or std.mem.startsWith(u8, token, "t:")) {
                space = if (token[0] == 'b') backbone else tower;
                number = token[2..];
            } else if (token.len == 0 or !std.ascii.isDigit(token[0])) {
                s.kept_classes.insert(try class(token));
                space = null;
                continue;
            }
            const layers = space orelse return error.UnnamespacedLayer;
            const l = decimal(number) orelse return error.BadLayerSelector;
            if (l >= layers.len) return error.LayerOutOfRange;
            layers[l] = true;
        }
        return s;
    }

    pub fn converts(s: *const Selection, n: Name) bool {
        const kept = if (n.layer.tower) s.kept_tower else s.kept_backbone;
        return s.classes.contains(n.class) and !s.kept_classes.contains(n.class) and !kept[n.layer.index];
    }
};

/// A layer number written canonically: decimal digits only, with no sign, `_` or leading 0.
fn decimal(text: []const u8) ?usize {
    if (text.len == 0 or (text[0] == '0' and text.len > 1)) return null;
    for (text) |c| if (!std.ascii.isDigit(c)) return null;
    return std.fmt.parseInt(usize, text, 10) catch null;
}

fn class(token: []const u8) !Class {
    if (std.mem.indexOfScalar(u8, token, '.') != null) return error.TensorSelector;
    return std.meta.stringToEnum(Class, token) orelse error.UnknownDenseClass;
}

pub fn count(names: []const []const u8) Counts {
    var c: Counts = .{};
    for (names) |name| if (layerOf(name)) |l| {
        const n = if (l.tower) &c.tower else &c.backbone;
        n.* = @max(n.*, l.index + 1);
    };
    return c;
}

/// SRC's selected projections from its shard headers, each as C5 converts it: BF16 [n, k], n % 128 = k % 64 = 0.
pub fn admit(gpa: Allocator, arena: Allocator, io: std.Io, src: []const u8, shards: []const []const u8, index: std.json.ObjectMap, s: *const Selection) ![]const Name {
    var out: std.ArrayList(Name) = .empty;
    for (shards) |shard| {
        var file = try st.File.open(gpa, io, try std.fs.path.join(arena, &.{ src, shard }));
        defer file.close(io);
        for (file.names.keys(), file.names.values()) |name, e| {
            const at = parse(name) orelse continue;
            if (!index.contains(name)) return error.BadIndex;
            if (!s.converts(at)) continue;
            if (e.dtype != .bf16 or e.rank != 2 or e.shape[0] % 128 != 0 or e.shape[1] % fp8.row_group != 0) return error.ExpectedBf16Dense;
            if (index.contains(try std.fmt.allocPrint(arena, "{s}_scale_inv", .{name}))) return error.DenseCompanionExists;
            try out.append(arena, at);
        }
    }
    return out.items;
}

/// C6 step 5 sums of one tensor, f64 in row-major increasing order.
pub const Errors = struct {
    err: f64 = 0,
    ref: f64 = 0,
    err_h: f64 = 0,
    ref_h: f64 = 0,
    max_abs: f64 = 0,
    worst_row: ?f64 = 0,

    pub fn rel(e: Errors) ?f64 {
        return ratio(e.err, e.ref);
    }

    pub fn errH(e: Errors) ?f64 {
        return ratio(e.err_h, e.ref_h);
    }
};

/// sqrt(num / den): 0 for 0 / 0, null (reported "undefined") for nonzero / 0.
fn ratio(num: f64, den: f64) ?f64 {
    if (den == 0) return if (num == 0) 0 else null;
    return @sqrt(num / den);
}

pub const Failure = error{NonFiniteWeight} || fp8.RowError;

/// One BF16 projection to encode, the slices its codes and scales land in, and its errors.
pub const Job = struct {
    name: []const u8,
    at: Name,
    weight: st.Tensor,
    /// shared_in's diagonal input importance (with --stats), weighing err_h.
    h: ?[]const f64 = null,
    codes: []u8 = &.{},
    scales: []f32 = &.{},
    errors: Errors = .{},
    failure: ?Failure = null,

    pub fn rows(j: Job) usize {
        return j.weight.shape[0];
    }

    pub fn cols(j: Job) usize {
        return j.weight.shape[1];
    }

    /// C6, then step 5 against what fp8.referenceRows reads back, a row at a time into `row` [k].
    pub fn encode(j: *Job, row: []f64) Failure!void {
        const n, const k = .{ j.rows(), j.cols() };
        try fp8.encodeRows(j.weight.bytes, n, k, j.codes, j.scales);
        const kg = k / fp8.row_group;
        for (0..n) |r| {
            try fp8.referenceRows(.{
                .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 1, k, 1, 1, 1 }, .bytes = j.codes[r * k ..][0..k] },
                .scale_inv = .{ .dtype = .f32, .rank = 2, .shape = .{ 1, kg, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(j.scales[r * kg ..][0..kg]) },
            }, row[0..k]);
            var row_err: f64 = 0;
            var row_ref: f64 = 0;
            const e = &j.errors;
            for (row[0..k], 0..) |q, c| {
                const x: f64 = core.quant.convert.load(.bf16, j.weight.bytes, r * k + c);
                const d = x - q;
                e.err += d * d;
                e.ref += x * x;
                row_err += d * d;
                row_ref += x * x;
                e.max_abs = @max(e.max_abs, @abs(d));
                if (j.h) |h| {
                    e.err_h += h[c] * d * d;
                    e.ref_h += h[c] * x * x;
                }
            }
            if (e.worst_row) |w| e.worst_row = if (ratio(row_err, row_ref)) |x| @max(w, x) else null;
        }
    }
};

/// A converted projection's C6 errors for the provenance file; err_h only when `weighed`.
pub const Report = struct { name: []const u8, at: Name, errors: Errors, weighed: bool };
