//! quant-provenance.json: what a tf-drama-quant run did to the experts and the dense projections, and SRC's own record.
const std = @import("std");
const convert_ = @import("quant_convert.zig");
const importance_ = @import("nvfp4_importance.zig");
const encoded_ = @import("nvfp4_encoded.zig");
const dense_ = @import("quant_dense.zig");
const files = @import("quant_files.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;
const Importance = importance_.Importance;
const projections = encoded_.projections;

pub const name = "quant-provenance.json";
/// The legacy Phase 1 converter's provenance file.
const legacy_name = "nvfp4-provenance.json";

/// SRC's own provenance file: its name, SHA-256 and exact text.
pub const Source = struct { file: []const u8, sha256: []const u8, text: []const u8 };

/// --compact-experts: the mask file, and the routed experts DST holds of SRC's.
pub const Compact = struct { file: []const u8, sha256: []const u8, held: usize, of: usize };

/// SRC's quant-provenance.json, else its nvfp4-provenance.json, else null.
pub fn source(arena: Allocator, io: Io, from: []const u8) !?Source {
    for ([_][]const u8{ name, legacy_name }) |file| {
        const path = try std.fs.path.join(arena, &.{ from, file });
        const text = Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 28)) catch |e| if (e == error.FileNotFound) continue else return e;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
        return .{ .file = file, .sha256 = try arena.dupe(u8, &std.fmt.bytesToHex(digest, .lower)), .text = text };
    }
    return null;
}

/// Everything a run's record is made of.
pub const Run = struct {
    from: []const u8,
    o: convert_.Options,
    convert_layer: []const bool,
    sources: []const ?encoded_.Layer,
    weigh: ?*const Importance,
    selection: ?*const dense_.Selection,
    experts: []const convert_.Report,
    dense: []const dense_.Report,
    source: ?Source,
    compact_experts: ?Compact,
    start: Io.Timestamp,
};

pub fn build(arena: Allocator, io: Io, r: Run) !Provenance {
    const cwd = Io.Dir.cwd();
    const wall_s = @as(f64, @floatFromInt(r.start.untilNow(io, .awake).nanoseconds)) / 1e9;
    const stats_path: ?[]const u8 = if (r.o.stats) |p| try cwd.realPathFileAlloc(io, p, arena) else null;
    const stats_sha: ?[]const u8 = if (stats_path) |p| try arena.dupe(u8, &try files.fileSha256(io, p)) else null;
    const config_sha = try arena.dupe(u8, &try files.fileSha256(io, try std.fs.path.join(arena, &.{ r.from, "config.json" })));
    var consumers: std.ArrayList([]const u8) = .empty;
    const encodes = r.o.experts == .nvfp4 and std.mem.indexOfScalar(bool, r.convert_layer, true) != null;
    if (r.weigh != null and encodes) try consumers.append(arena, "experts");
    var weighed = false;
    for (r.dense) |d| weighed = weighed or d.weighed;
    if (weighed) try consumers.append(arena, "dense_fp8");
    return .{
        .source = r.from,
        .stats = if (stats_path) |p| .{ .path = p, .sha256 = stats_sha.?, .consumed = consumers.items.len > 0 } else null,
        .stats_consumed_by = consumers.items,
        .experts = switch (r.o.experts) {
            .keep => .{ .keep = .{} },
            .nvfp4 => .{ .nvfp4 = try experts(arena, r, config_sha, stats_path, stats_sha, wall_s) },
        },
        .source_provenance = r.source,
        .compact_experts = r.compact_experts,
        .dense_fp8 = if (r.selection) |s| try dense(arena, io, r, s, config_sha, if (weighed) stats_path else null, if (weighed) stats_sha else null, wall_s) else null,
    };
}

fn experts(arena: Allocator, r: Run, config_sha: []const u8, stats_path: ?[]const u8, stats_sha: ?[]const u8, wall_s: f64) !Experts {
    var converted: std.ArrayList(usize) = .empty;
    var layer_sources: std.ArrayList(Experts.LayerSource) = .empty;
    for (r.convert_layer, r.sources, 0..) |c, *from_file, l| if (c) {
        try converted.append(arena, l);
        try layer_sources.append(arena, if (from_file.*) |*f| .{ .layer = l, .source = "encoded", .file = f.path, .sha256 = &f.sha256, .metadata = f.metadata } else .{ .layer = l, .source = "rtn" });
    };
    const kept = try arena.dupe(usize, r.o.keep);
    std.mem.sort(usize, kept, {}, std.sort.asc(usize));
    const tensors = try arena.alloc(Experts.Tensor, r.experts.len);
    const groups = try arena.alloc(Experts.Group, r.convert_layer.len * projections.len);
    for (groups, 0..) |*g, i| g.* = .{ .layer = i / projections.len, .projection = projections[i % projections.len] };
    for (tensors, r.experts) |*t, e| {
        const x = e.summary;
        t.* = .{ .name = e.name, .amax = x.amax, .global = x.global, .rel = ratio(x.rel()), .err_h = ratio(x.errH()) };
        const g = &groups[e.at.layer * projections.len + e.at.proj];
        g.tensors += 1;
        const rel = x.rel() orelse continue;
        const err_h = x.errH() orelse continue;
        g.defined += 1;
        g.rel_mean += rel;
        g.rel_max = @max(g.rel_max, rel);
        g.err_h_mean += err_h;
        g.err_h_max = @max(g.err_h_max, err_h);
    }
    var summaries: std.ArrayList(Experts.Group) = .empty;
    for (groups) |g| if (g.tensors > 0) try summaries.append(arena, .{
        .layer = g.layer,
        .projection = g.projection,
        .tensors = g.tensors,
        .defined = g.defined,
        .rel_mean = if (g.defined == 0) 0 else g.rel_mean / @as(f64, @floatFromInt(g.defined)),
        .rel_max = g.rel_max,
        .err_h_mean = if (g.defined == 0) 0 else g.err_h_mean / @as(f64, @floatFromInt(g.defined)),
        .err_h_max = g.err_h_max,
    });
    var pairs: ?std.json.Value = null;
    if (r.weigh) |imp| {
        var m: std.json.ObjectMap = .empty;
        for (importance_.classes, imp.pairs) |c, n| try m.put(arena, c, .{ .float = n });
        pairs = .{ .object = m };
    }
    return .{
        .source = r.from,
        .config_sha256 = config_sha,
        .tool_commit = r.o.commit,
        .stats = stats_path,
        .stats_sha256 = stats_sha,
        .stats_pairs = pairs,
        .threads = r.o.threads,
        .converted_layers = converted.items,
        .layer_sources = layer_sources.items,
        .kept_fp8_layers = kept,
        .wall_s = wall_s,
        .projections = summaries.items,
        .tensors = tensors,
    };
}

fn dense(arena: Allocator, io: Io, r: Run, s: *const dense_.Selection, config_sha: []const u8, stats_path: ?[]const u8, stats_sha: ?[]const u8, wall_s: f64) !Dense {
    var classes: std.ArrayList([]const u8) = .empty;
    var it = s.classes.iterator();
    while (it.next()) |c| try classes.append(arena, @tagName(c));
    const tensors = try arena.alloc(Dense.Tensor, r.dense.len);
    for (tensors, r.dense) |*t, d| {
        const e = d.errors;
        const space = if (d.at.layer.tower) "t" else "b";
        t.* = .{
            .name = d.name,
            .class = @tagName(d.at.class),
            .layer = try std.fmt.allocPrint(arena, "{s}:{d}", .{ space, d.at.layer.index }),
            .rel = ratio(e.rel()),
            .worst_row_rel = ratio(e.worst_row),
            .max_abs = e.max_abs,
            .err_h = if (d.weighed) ratio(e.errH()) else .null,
        };
    }
    return .{
        .source = r.from,
        .index_sha256 = try arena.dupe(u8, &try files.fileSha256(io, try std.fs.path.join(arena, &.{ r.from, "model.safetensors.index.json" }))),
        .config_sha256 = config_sha,
        .tool_commit = r.o.commit,
        .classes = classes.items,
        .selectors = .{ .dense_fp8 = s.dense, .keep_bf16 = s.keep },
        .stats = stats_path,
        .stats_sha256 = stats_sha,
        .threads = r.o.threads,
        .wall_s = wall_s,
        .tensors = tensors,
    };
}

fn ratio(v: ?f64) std.json.Value {
    return if (v) |x| .{ .float = x } else .{ .string = "undefined" };
}

pub const Provenance = struct {
    schema_version: u32 = 1,
    producer: []const u8 = "tf-drama-quant",
    source: []const u8,
    /// The --stats file given and whether any operation read it; null without --stats.
    stats: ?struct { path: []const u8, sha256: []const u8, consumed: bool },
    /// The operations that read --stats: experts (C3 importance) and dense_fp8 (shared_in's err_h).
    stats_consumed_by: []const []const u8,
    experts: union(enum) {
        nvfp4: Experts,
        keep: struct {
            mode: []const u8 = "keep",
            description: []const u8 = "every routed-expert tensor and companion DST holds copied from SRC byte for byte; none encoded by this run",
        },

        /// The entry alone, its mode a field of its own.
        pub fn jsonStringify(self: @This(), jw: anytype) !void {
            switch (self) {
                inline else => |v| try jw.write(v),
            }
        }
    },
    /// SRC's own provenance file, verbatim; null when SRC has none.
    source_provenance: ?Source,
    /// null without --compact-experts; DST's index maps the ids each layer holds (`drama_experts`).
    compact_experts: ?Compact,
    /// null when --dense-fp8 was not given; a selection --keep-bf16 empties keeps its record with no tensors.
    dense_fp8: ?Dense,
};

/// The routed experts this run encoded: where the checkpoint came from and each converted projection's C2 errors.
const Experts = struct {
    mode: []const u8 = "nvfp4",
    /// NVFP4 experts are a lossy repack of the FP8 checkpoint, not the faithful port.
    lossy: bool = true,
    source: []const u8,
    config_sha256: []const u8,
    tool_commit: []const u8,
    stats: ?[]const u8,
    stats_sha256: ?[]const u8,
    /// Routed pairs per class behind the pooled sums (null without stats).
    stats_pairs: ?std.json.Value,
    kappa: u32 = importance_.kappa,
    threads: usize,
    converted_layers: []const usize,
    /// How each converted layer was encoded: here (rtn, C2) or taken from an --encoded file.
    layer_sources: []const LayerSource,
    kept_fp8_layers: []const usize,
    /// Seconds from start to this file, everything but its own write.
    wall_s: f64,
    err_h: []const u8 = "sqrt(sum h (W - Q)^2 / sum h W^2): a diagonal expert-input-error proxy, not the routed layer-output error",
    rel: []const u8 = "||W - Q|| / ||W||; either is \"undefined\" when only its denominator is 0",
    projections: []const Group,
    tensors: []const Tensor,

    const LayerSource = struct { layer: usize, source: []const u8, file: ?[]const u8 = null, sha256: ?[]const u8 = null, metadata: ?std.json.Value = null };
    const Tensor = struct { name: []const u8, amax: f64, global: f32, rel: std.json.Value, err_h: std.json.Value };
    /// One (layer, projection) over its experts; means and maxima over the `defined` ones.
    const Group = struct {
        layer: usize,
        projection: []const u8,
        tensors: usize = 0,
        defined: usize = 0,
        rel_mean: f64 = 0,
        rel_max: f64 = 0,
        err_h_mean: f64 = 0,
        err_h_max: f64 = 0,
    };
};

/// The dense projections this run made FP8 (C5, C6): SRC's identity, the selectors and each tensor's errors.
const Dense = struct {
    lossy: bool = true,
    source: []const u8,
    index_sha256: []const u8,
    config_sha256: []const u8,
    tool_commit: []const u8,
    /// --dense-fp8's classes resolved; --keep-bf16 then removed whole fused groups.
    classes: []const []const u8,
    selectors: struct { dense_fp8: []const u8, keep_bf16: ?[]const u8 },
    scale_rule: []const u8 = "per row and 64 inputs: scale_inv = f32(amax / 448) rounded once from f64, 1.0 for an all-zero group; " ++
        "code = w / scale_inv rounded to the nearest e4m3fn value, ties to even, saturating at 448",
    /// The statistics shared_in's err_h weighs by (null when none was weighed).
    stats: ?[]const u8,
    stats_sha256: ?[]const u8,
    threads: usize,
    /// Seconds from start to this file, everything but its own write.
    wall_s: f64,
    rel: []const u8 = "||W - Q|| / ||W|| over the tensor, worst_row_rel the largest over its rows; \"undefined\" when only its denominator is 0",
    max_abs: []const u8 = "max |W - Q|",
    err_h: []const u8 = "shared_in with --stats: sqrt(sum h (W - Q)^2 / sum h W^2), h the layer's routed gate/up input sums; else null",
    /// The converted tensors, in output order: each `X.weight` is F8_E4M3 beside an F32 `X.weight_scale_inv`.
    tensors: []const Tensor,

    const Tensor = struct { name: []const u8, class: []const u8, layer: []const u8, rel: std.json.Value, worst_row_rel: std.json.Value, max_abs: f64, err_h: std.json.Value };
};
