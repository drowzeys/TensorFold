//! tf-drama-quant: a new Drama 3 checkpoint, its routed experts NVFP4 or kept and chosen dense projections FP8 (C5).
const std = @import("std");
const core = @import("core");
const importance_ = @import("nvfp4_importance.zig");
const encoded_ = @import("nvfp4_encoded.zig");
const dense_ = @import("quant_dense.zig");
const files = @import("quant_files.zig");
const provenance_ = @import("quant_provenance.zig");
const expert_mask = @import("expert_mask.zig");
const compact = @import("compact.zig");
const cfg = @import("backbone_config.zig");

const st = core.safetensors;
const shard_edit = core.shard_edit;
const nvfp4 = core.quant.nvfp4;
const fp8 = core.quant.fp8;
const Allocator = std.mem.Allocator;
const Io = std.Io;

const Importance = importance_.Importance;
pub const kappa = importance_.kappa;
const index_name = "model.safetensors.index.json";

pub const Experts = enum { nvfp4, keep };

pub const Options = struct {
    /// nvfp4 encodes the routed experts, which must be FP8; keep copies every expert tensor byte for byte.
    experts: Experts = .nvfp4,
    /// tf-drama-probe's C3 statistics; null weighs every input alike.
    stats: ?[]const u8 = null,
    /// Layers whose experts stay FP8, byte for byte.
    keep: []const usize = &.{},
    /// A directory of layer-LL.safetensors (C1 tensors another encoder wrote) taken instead of encoding those layers.
    encoded: ?[]const u8 = null,
    /// C5 classes whose BF16 projections become FP8: CLASS[,CLASS] or all; null converts none.
    dense_fp8: ?[]const u8 = null,
    /// Fused groups of those that stay BF16: CLASS, b:L[,L] (backbone layers) or t:L[,L] (tower layers) items.
    keep_bf16: ?[]const u8 = null,
    /// An --expert-mask file: only its experts are written, under their own names, before any other expert operation.
    compact_experts: ?[]const u8 = null,
    threads: usize = 1,
    commit: []const u8 = "unknown",
};

pub const Summary = struct { shards: usize, converted: usize, dense: usize, copied: usize, bytes: u64 };

const projections = encoded_.projections;

/// A routed expert projection's weight: `model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`.
pub const Name = struct {
    layer: usize,
    expert: usize,
    proj: usize,

    /// gate and up read the MoE input row (one importance per layer), down the expert's own activation.
    fn gateUp(n: Name) bool {
        return n.proj < 2;
    }
};

/// A routed expert projection's tensor: the projection and the field after it (`weight`, `weight_scale_inv`, ...).
fn parseField(name: []const u8) ?struct { at: Name, field: []const u8 } {
    const prefix = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    var it = std.mem.splitScalar(u8, name[prefix.len..], '.');
    const layer = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    if (!std.mem.eql(u8, it.next() orelse return null, "mlp") or !std.mem.eql(u8, it.next() orelse return null, "experts")) return null;
    const expert = std.fmt.parseInt(usize, it.next() orelse return null, 10) catch return null;
    const proj = it.next() orelse return null;
    const field = it.rest();
    if (field.len == 0) return null;
    for (projections, 0..) |p, i| if (std.mem.eql(u8, proj, p)) return .{ .at = .{ .layer = layer, .expert = expert, .proj = i }, .field = field };
    return null;
}

fn parse(name: []const u8) ?Name {
    const x = parseField(name) orelse return null;
    return if (std.mem.eql(u8, x.field, "weight")) x.at else null;
}

/// A converted projection's tensor: its weight, its consumed scale_inv, or an input_scale C1 drops; others refused.
const Role = enum { weight, scale_inv, dropped };

fn role(field: []const u8) !Role {
    if (std.mem.eql(u8, field, "weight")) return .weight;
    if (std.mem.eql(u8, field, "weight_scale_inv")) return .scale_inv;
    if (std.mem.eql(u8, field, "input_scale")) return .dropped;
    if (std.mem.eql(u8, field, "weight_scale") or std.mem.eql(u8, field, "weight_scale_2")) return error.ExpectedFp8Expert; // NVFP4 already
    return error.UnknownExpertCompanion;
}

const Failure = error{ NonFiniteWeight, BadSourceScale, BadImportance, BadGlobal };

/// One projection to encode and the slices its three NVFP4 tensors land in.
const Job = struct {
    name: []const u8,
    at: Name,
    weight: st.Tensor,
    scale_inv: st.Tensor,
    codes: []u8 = &.{},
    scales: []u8 = &.{},
    /// An --encoded layer's tensors as its file holds them: measured against the source, never re-encoded.
    taken: ?struct { codes: []const u8, scales: []const u8, global: f32 } = null,
    summary: nvfp4.Summary = .{},
    failure: ?Failure = null,

    fn rows(j: Job) usize {
        return j.weight.shape[0];
    }

    fn cols(j: Job) usize {
        return j.weight.shape[1];
    }
};

const Work = struct {
    jobs: []Job,
    dense: []dense_.Job,
    importance: ?*const Importance,
    next: std.atomic.Value(usize) = .init(0),

    fn run(work: *Work, w: []f64, h: []f64) void {
        while (true) {
            const i = work.next.fetchAdd(1, .monotonic);
            if (i < work.jobs.len) {
                work.encode(&work.jobs[i], w, h) catch |e| {
                    work.jobs[i].failure = e;
                };
            } else if (i - work.jobs.len < work.dense.len) {
                const j = &work.dense[i - work.jobs.len];
                j.encode(w) catch |e| {
                    j.failure = e;
                };
            } else return;
        }
    }

    /// C2 steps 1-2, then the encoder or a taken layer's errors: the source in f64 as fp8.reference decodes it.
    fn encode(work: *const Work, j: *Job, w_buf: []f64, h_buf: []f64) Failure!void {
        const n, const k = .{ j.rows(), j.cols() };
        for (0..j.scale_inv.numel()) |i| {
            const s: f32 = @bitCast(std.mem.readInt(u32, j.scale_inv.bytes[i * 4 ..][0..4], .little));
            if (!std.math.isFinite(s) or s <= 0) return error.BadSourceScale;
        }
        const w = w_buf[0 .. n * k];
        fp8.reference(.{ .weight = j.weight, .scale_inv = j.scale_inv }, w) catch unreachable; // shapes checked on admission
        for (w) |x| if (!std.math.isFinite(x)) return error.NonFiniteWeight;
        const h = h_buf[0..k];
        if (work.importance) |imp| imp.fill(j.at.layer, j.at.expert, j.at.gateUp(), h) else @memset(h, 1);
        for (h) |x| if (!std.math.isFinite(x) or x < 0) return error.BadImportance;
        if (j.taken) |t| {
            j.summary = nvfp4.measure(w, h, n, k, t.codes, t.scales, t.global);
            return;
        }
        j.summary = nvfp4.encode(w, h, n, k, j.codes, j.scales);
        const g = j.summary.global;
        if (j.summary.amax > 0 and !(std.math.isFinite(g) and g > 0)) return error.BadGlobal; // amax / 2688 left f32
    }
};

/// Encodes the jobs on `threads` threads; each job's bytes depend on its inputs alone, so any count writes the same.
fn encodeAll(gpa: Allocator, jobs: []Job, dense: []dense_.Job, importance: ?*const Importance, threads: usize) !void {
    var most: [2]usize = .{ 0, 0 };
    for (jobs) |j| most = .{ @max(most[0], j.rows() * j.cols()), @max(most[1], j.cols()) };
    for (dense) |j| most[0] = @max(most[0], j.cols());
    const t = @max(1, @min(threads, jobs.len + dense.len));
    const w = try gpa.alloc(f64, t * most[0]);
    defer gpa.free(w);
    const h = try gpa.alloc(f64, t * most[1]);
    defer gpa.free(h);
    var work: Work = .{ .jobs = jobs, .dense = dense, .importance = importance };
    const pool = try gpa.alloc(std.Thread, t - 1);
    defer gpa.free(pool);
    {
        var spawned: usize = 0;
        defer for (pool[0..spawned]) |th| th.join();
        for (pool, 1..) |*th, i| {
            th.* = try std.Thread.spawn(.{}, Work.run, .{ &work, w[i * most[0] ..][0..most[0]], h[i * most[1] ..][0..most[1]] });
            spawned += 1;
        }
        work.run(w[0..most[0]], h[0..most[1]]);
    }
    for (jobs) |j| if (j.failure) |e| return e; // the first in tensor order, whichever thread met it
    for (dense) |j| if (j.failure) |e| return e;
}

/// A tensor of the output shard: SRC's own bytes, or a job's NVFP4 triple or FP8 pair in the weight's place.
const Item = union(enum) { keep: struct { name: []const u8, e: st.Entry }, job: usize, dense: usize };

/// A converted projection's amax, global and errors (C2 step 5), for the provenance file.
pub const Report = struct { name: []const u8, at: Name, summary: nvfp4.Summary };

/// What a run does in every shard: experts it holds, expert layers it encodes and from where, dense projections.
const Plan = struct {
    held: ?*const expert_mask.Mask,
    convert_layer: []const bool,
    sources: []const ?encoded_.Layer,
    importance: ?*const Importance,
    dense: ?*const dense_.Selection,
    threads: usize,
};

const Reports = struct { experts: std.ArrayList(Report) = .empty, dense: std.ArrayList(dense_.Report) = .empty };

const Shard = struct { names: []const []const u8, bytes: u64 };

/// One SRC shard rewritten at `out` in its tensor order; returns its tensor names and data bytes.
fn convertShard(gpa: Allocator, arena: Allocator, io: Io, src: []const u8, out: []const u8, plan: Plan, reports: *Reports) !Shard {
    const convert_layer, const sources, const importance = .{ plan.convert_layer, plan.sources, plan.importance };
    var file = try st.File.open(gpa, io, src);
    defer file.close(io);
    var taken: std.AutoArrayHashMapUnmanaged(usize, st.File) = .empty;
    defer {
        for (taken.values()) |*f| f.close(io);
        taken.deinit(gpa);
    }
    var scratch_state = std.heap.ArenaAllocator.init(gpa);
    defer scratch_state.deinit();
    const scratch = scratch_state.allocator();
    const raw = file.map.memory;
    const header = try std.json.parseFromSliceLeaky(std.json.Value, scratch, raw[8..file.data], .{});
    var meta: std.ArrayList([2][]const u8) = .empty;
    if (header.object.get("__metadata__")) |m| {
        if (m != .object) return error.BadSafetensors;
        var it = m.object.iterator();
        while (it.next()) |kv| try meta.append(scratch, .{ kv.key_ptr.*, if (kv.value_ptr.* == .string) kv.value_ptr.string else return error.BadSafetensors });
    }
    const keys = file.names.keys();
    const entries = file.names.values();
    const order = try scratch.alloc(usize, entries.len);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, @as([]const st.Entry, entries), struct {
        fn lt(e: []const st.Entry, a: usize, b: usize) bool {
            return e[a].begin < e[b].begin;
        }
    }.lt);
    var items: std.ArrayList(Item) = .empty;
    var jobs: std.ArrayList(Job) = .empty;
    var dense: std.ArrayList(dense_.Job) = .empty;
    var encoded: usize = 0;
    var dense_codes: usize = 0;
    for (order) |i| {
        const name = keys[i];
        if (plan.dense) |selection| if (dense_.parse(name)) |at| if (selection.converts(at)) {
            const weight = file.get(name).?; // its dtype and shape were admitted
            const k = weight.shape[1];
            try items.append(scratch, .{ .dense = dense.items.len });
            const job = try dense.addOne(scratch);
            job.* = .{ .name = try arena.dupe(u8, name), .at = at, .weight = weight };
            if (at.class == .shared_in) if (importance) |imp| {
                if (!imp.covers(at.layer.index, true, k)) return error.StatsShape;
                const h = try scratch.alloc(f64, k);
                imp.shared(at.layer.index, h);
                job.h = h;
            };
            dense_codes += weight.shape[0] * k;
            continue;
        };
        const x = parseField(name) orelse {
            try items.append(scratch, .{ .keep = .{ .name = name, .e = entries[i] } });
            continue;
        };
        const at = x.at;
        if (plan.held) |held| if (!holds(held, at.layer, at.expert)) continue; // every tensor of a dropped expert
        if (at.layer >= convert_layer.len or !convert_layer[at.layer]) {
            try items.append(scratch, .{ .keep = .{ .name = name, .e = entries[i] } });
            continue;
        }
        switch (try role(x.field)) {
            .weight => {},
            .scale_inv => {
                if (!file.names.contains(try std.fmt.allocPrint(scratch, "{s}weight", .{name[0 .. name.len - x.field.len]}))) return error.SplitExpert;
                continue;
            },
            .dropped => continue,
        }
        const weight = file.get(name).?;
        const scale_inv = file.get(try std.fmt.allocPrint(scratch, "{s}_scale_inv", .{name})) orelse return error.SplitExpert;
        if (weight.dtype != .f8_e4m3 or weight.rank != 2 or weight.shape[1] % nvfp4.group != 0) return error.ExpectedFp8Expert;
        const n, const k = .{ weight.shape[0], weight.shape[1] };
        if (!scale_inv.is(.f32, &.{ (n + fp8.block - 1) / fp8.block, (k + fp8.block - 1) / fp8.block })) return error.ExpectedFp8Expert;
        if (importance) |imp| if (!imp.covers(at.layer, at.gateUp(), k)) return error.StatsShape;
        try items.append(scratch, .{ .job = jobs.items.len });
        const job = try jobs.addOne(scratch);
        job.* = .{ .name = try arena.dupe(u8, name), .at = at, .weight = weight, .scale_inv = scale_inv };
        const layer = sources[at.layer] orelse {
            encoded += n * k / 2 + n * k / nvfp4.group;
            continue;
        };
        if (!taken.contains(at.layer)) {
            var f = try st.File.open(gpa, io, layer.path);
            errdefer f.close(io);
            try taken.put(gpa, at.layer, f);
        }
        const f = taken.getPtr(at.layer).?;
        const base = name[0 .. name.len - "weight".len];
        const g = f.get(try std.fmt.allocPrint(scratch, "{s}weight_scale_2", .{base})).?.bytes;
        job.taken = .{ // check() admitted every tensor of the layer
            .codes = f.get(name).?.bytes,
            .scales = f.get(try std.fmt.allocPrint(scratch, "{s}weight_scale", .{base})).?.bytes,
            .global = @bitCast(std.mem.readInt(u32, g[0..4], .little)),
        };
    }
    const buffer = try gpa.alloc(u8, encoded + dense_codes);
    defer gpa.free(buffer);
    const dense_scales = try gpa.alloc(f32, dense_codes / fp8.row_group);
    defer gpa.free(dense_scales);
    var used: usize = 0;
    for (jobs.items) |*j| if (j.taken == null) {
        const nk = j.rows() * j.cols();
        j.codes = buffer[used..][0 .. nk / 2];
        j.scales = buffer[used + nk / 2 ..][0 .. nk / nvfp4.group];
        used += nk / 2 + nk / nvfp4.group;
    };
    var scaled: usize = 0;
    for (dense.items) |*j| {
        const nk = j.rows() * j.cols();
        j.codes = buffer[used..][0..nk];
        j.scales = dense_scales[scaled..][0 .. nk / fp8.row_group];
        used += nk;
        scaled += nk / fp8.row_group;
    }
    try encodeAll(gpa, jobs.items, dense.items, importance, plan.threads);
    var tensors: std.ArrayList(shard_edit.New) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    var bytes: u64 = 0;
    for (items.items) |*item| switch (item.*) {
        .keep => |*x| try tensors.append(scratch, .{ .name = x.name, .dtype = x.e.dtype, .shape = x.e.shape[0..x.e.rank], .bytes = raw[file.data + x.e.begin .. file.data + x.e.end] }),
        .job => |ji| {
            const j = &jobs.items[ji];
            const n, const k = .{ j.rows(), j.cols() };
            const base = j.name[0 .. j.name.len - "weight".len];
            try tensors.appendSlice(scratch, &.{
                .{ .name = j.name, .dtype = .u8, .shape = try scratch.dupe(usize, &.{ n, k / 2 }), .bytes = if (j.taken) |t| t.codes else j.codes },
                .{ .name = try std.fmt.allocPrint(scratch, "{s}weight_scale", .{base}), .dtype = .f8_e4m3, .shape = try scratch.dupe(usize, &.{ n, k / nvfp4.group }), .bytes = if (j.taken) |t| t.scales else j.scales },
                .{ .name = try std.fmt.allocPrint(scratch, "{s}weight_scale_2", .{base}), .dtype = .f32, .shape = &.{}, .bytes = std.mem.asBytes(&j.summary.global) },
            });
        },
        .dense => |di| {
            const j = &dense.items[di];
            try tensors.appendSlice(scratch, &.{
                .{ .name = j.name, .dtype = .f8_e4m3, .shape = try scratch.dupe(usize, &.{ j.rows(), j.cols() }), .bytes = j.codes },
                .{ .name = try std.fmt.allocPrint(scratch, "{s}_scale_inv", .{j.name}), .dtype = .f32, .shape = try scratch.dupe(usize, &.{ j.rows(), j.cols() / fp8.row_group }), .bytes = std.mem.sliceAsBytes(j.scales) },
            });
        },
    };
    for (tensors.items) |t| {
        try names.append(arena, try arena.dupe(u8, t.name));
        bytes += t.bytes.len;
    }
    if (Io.Dir.cwd().access(io, out, .{})) |_| return error.OutputExists else |e| if (e != error.FileNotFound) return e;
    try shard_edit.create(gpa, out, meta.items, tensors.items);
    for (jobs.items) |j| try reports.experts.append(arena, .{ .name = j.name, .at = j.at, .summary = j.summary });
    for (dense.items) |j| try reports.dense.append(arena, .{ .name = j.name, .at = j.at, .errors = j.errors, .weighed = j.h != null });
    return .{ .names = names.items, .bytes = bytes };
}

/// SRC rewritten at DST (new, outside SRC): experts NVFP4 or kept, chosen dense projections FP8, all else as it was.
pub fn convert(gpa: Allocator, io: Io, src: []const u8, dst: []const u8, o: Options) !Summary {
    const start = Io.Timestamp.now(io, .awake);
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cwd = Io.Dir.cwd();
    const from = try cwd.realPathFileAlloc(io, src, arena);
    const parent = try cwd.realPathFileAlloc(io, std.fs.path.dirname(dst) orelse ".", arena);
    const to = try std.fs.path.join(arena, &.{ parent, std.fs.path.basename(dst) });
    if (std.mem.startsWith(u8, to, from) and (to.len == from.len or to[from.len] == '/')) return error.DestinationInSource;

    // checked before staging: index paths, layer selections, encoded tensors, stats values, ancillary file kinds
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, try cwd.readFileAlloc(io, try join(arena, from, index_name), arena, .limited(1 << 28)), .{});
    const map = if (index == .object) index.object.get("weight_map") orelse return error.BadIndex else return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var shards: std.StringArrayHashMapUnmanaged(void) = .empty;
    var experts: std.ArrayList(usize) = .empty;
    for (map.object.keys(), map.object.values()) |name, shard| {
        if (shard != .string) return error.BadIndex;
        if (!files.shardName(shard.string)) return error.BadShardName;
        try shards.put(arena, shard.string, {});
        const at = parse(name) orelse continue;
        if (at.layer >= experts.items.len) try experts.appendNTimes(arena, 0, at.layer + 1 - experts.items.len);
        experts.items[at.layer] = @max(experts.items[at.layer], at.expert + 1);
    }
    const source_meta = index.object.get("metadata") orelse std.json.Value{ .object = .empty };
    if (source_meta != .object) return error.BadIndex;
    const held: ?expert_mask.Mask = if (o.compact_experts) |path| try heldExperts(arena, io, path, experts.items) else null;
    if (held != null and source_meta.object.contains(compact.experts_key)) return error.AlreadyCompact;
    if (held) |*h| try storedHeld(gpa, arena, io, from, map.object, h, experts.items);
    const shard_names = shards.keys();
    std.mem.sort([]const u8, shard_names, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    if (o.experts == .keep and (o.keep.len > 0 or o.encoded != null)) return error.ExpertsKept;
    const convert_layer = try arena.alloc(bool, experts.items.len);
    for (convert_layer, experts.items) |*c, e| c.* = o.experts == .nvfp4 and e > 0;
    for (o.keep) |l| {
        if (l >= experts.items.len or experts.items[l] == 0) return error.KeptLayerHasNoExperts;
        convert_layer[l] = false;
    }
    for (map.object.keys()) |name| if (parseField(name)) |x| if (x.at.layer < convert_layer.len and convert_layer[x.at.layer]) {
        _ = try role(x.field);
    };
    const sources = try arena.alloc(?encoded_.Layer, experts.items.len);
    @memset(sources, null);
    if (o.encoded) |dir| {
        const listed = try encoded_.list(arena, io, dir);
        const wanted = try arena.alloc(bool, experts.items.len);
        @memset(wanted, false);
        for (listed) |l| {
            if (l >= experts.items.len or experts.items[l] == 0) return error.EncodedLayerHasNoExperts;
            if (!convert_layer[l]) return error.EncodedLayerKept;
            wanted[l] = true;
        }
        const shapes = try sourceShapes(gpa, arena, io, from, shard_names, wanted);
        for (listed) |l| sources[l] = try encoded_.check(gpa, arena, io, dir, l, experts.items[l], &shapes);
    }
    const selection: ?dense_.Selection = if (o.dense_fp8) |classes|
        try dense_.Selection.init(arena, classes, o.keep_bf16, dense_.count(map.object.keys()))
    else if (o.keep_bf16 != null) return error.KeepWithoutDense else null;
    const select: ?*const dense_.Selection = if (selection) |*s| s else null;
    // statistics serve the layers whose experts encode and those whose shared_in converts (its err_h)
    const wanted = try arena.dupe(bool, convert_layer);
    if (select) |s| for (try dense_.admit(gpa, arena, io, from, shard_names, map.object, s)) |at| if (o.stats != null and at.class == .shared_in) {
        if (at.layer.index >= experts.items.len or experts.items[at.layer.index] == 0) return error.StatsShape;
        wanted[at.layer.index] = true;
    };
    var importance: ?Importance = if (o.stats) |p| try Importance.open(gpa, arena, io, p, wanted, experts.items) else null;
    defer if (importance) |*imp| imp.close(io);
    const weigh: ?*const Importance = if (importance) |*imp| imp else null;
    const others = try files.ancillary(gpa, arena, io, from, &.{ index_name, provenance_.name });
    const source_provenance = try provenance_.source(arena, io, from);

    // written into DST.partial, which becomes DST only once complete: a failed run leaves no DST behind
    if (cwd.access(io, to, .{})) |_| return error.DestinationExists else |e| if (e != error.FileNotFound) return e;
    const partial = try std.fmt.allocPrint(arena, "{s}.partial", .{to});
    cwd.createDir(io, partial, .default_dir) catch |e| return if (e == error.PathAlreadyExists) error.PartialExists else e;
    errdefer cwd.deleteTree(io, partial) catch {};
    {
        var out_dir = try cwd.openDir(io, partial, .{});
        defer out_dir.close(io);
        var weight_map: std.json.ObjectMap = .empty;
        var reports: Reports = .{};
        const plan: Plan = .{ .held = if (held) |*h| h else null, .convert_layer = convert_layer, .sources = sources, .importance = weigh, .dense = select, .threads = o.threads };
        var total: u64 = 0;
        for (shard_names) |shard| {
            const s = try convertShard(gpa, arena, io, try join(arena, from, shard), try join(arena, partial, shard), plan, &reports);
            for (s.names) |n| try weight_map.put(arena, n, .{ .string = shard });
            total += s.bytes;
        }
        var meta: std.json.ObjectMap = .empty;
        try meta.put(arena, "total_size", .{ .integer = @intCast(total) });
        var compact_record: ?provenance_.Compact = null;
        if (held) |*h| {
            var layers: std.json.ObjectMap = .empty;
            var count: usize = 0;
            var of: usize = 0;
            for (experts.items, 0..) |n, l| if (n > 0) {
                var ids: std.json.Array = .init(arena);
                for (0..n) |e| if (holds(h, l, e)) try ids.append(.{ .integer = @intCast(e) });
                try layers.put(arena, try std.fmt.allocPrint(arena, "{d}", .{l}), .{ .array = ids });
                count += ids.items.len;
                of += n;
            };
            try meta.put(arena, compact.experts_key, .{ .object = layers });
            const path = try cwd.realPathFileAlloc(io, o.compact_experts.?, arena);
            compact_record = .{ .file = path, .sha256 = try arena.dupe(u8, &try files.fileSha256(io, path)), .held = count, .of = of };
        } else if (source_meta.object.get(compact.experts_key)) |m| try meta.put(arena, compact.experts_key, m); // SRC's map holds on
        var root: std.json.ObjectMap = .empty;
        try root.put(arena, "metadata", .{ .object = meta });
        try root.put(arena, "weight_map", .{ .object = weight_map });
        try files.emit(arena, io, out_dir, index_name, std.json.Value{ .object = root });
        var src_dir = try cwd.openDir(io, from, .{});
        defer src_dir.close(io);
        try files.copy(io, src_dir, out_dir, others);
        try files.emit(arena, io, out_dir, provenance_.name, try provenance_.build(arena, io, .{
            .from = from,
            .o = o,
            .convert_layer = convert_layer,
            .sources = sources,
            .weigh = weigh,
            .selection = select,
            .experts = reports.experts.items,
            .dense = reports.dense.items,
            .source = source_provenance,
            .compact_experts = compact_record,
            .start = start,
        }));
        cwd.renamePreserve(partial, cwd, to, io) catch |e| return if (e == error.PathAlreadyExists) error.DestinationExists else e;
        return .{ .shards = shard_names.len, .converted = reports.experts.items.len, .dense = reports.dense.items.len, .copied = others.files.len, .bytes = total };
    }
}

/// Whether a run compacting to `held` writes expert `e` of `layer`: a layer past the mask's keeps all of its experts.
fn holds(held: *const expert_mask.Mask, layer: usize, e: usize) bool {
    if (layer >= held.len) return true;
    return e < cfg.experts and held[layer][e / 32] >> @intCast(e % 32) & 1 == 1;
}

/// Each held projection in its mapped shard, with the companions the loader reads for the weight's stored dtype.
fn storedHeld(gpa: Allocator, arena: Allocator, io: Io, from: []const u8, map: std.json.ObjectMap, held: *const expert_mask.Mask, experts: []const usize) !void {
    var stored: Stored = .{ .gpa = gpa, .arena = arena, .io = io, .dir = from, .map = map };
    defer stored.close();
    for (experts, 0..) |count, l| for (0..count) |e| if (holds(held, l, e)) for (projections) |p| {
        const w = try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.weight", .{ l, e, p });
        const weight = try stored.get(w, "");
        if (weight.rank != 2) return error.CompactExpertNotInSource;
        const n, const k = .{ weight.shape[0], weight.shape[1] };
        const ok = switch (weight.dtype) {
            .f8_e4m3 => (try stored.get(w, "_scale_inv")).is(.f32, &.{ (n + fp8.block - 1) / fp8.block, (k + fp8.block - 1) / fp8.block }),
            .u8 => (try stored.get(w, "_scale")).is(.f8_e4m3, &.{ n, k * 2 / nvfp4.group }) and (try stored.get(w, "_scale_2")).is(.f32, &.{}),
            else => false,
        };
        if (!ok) return error.CompactExpertNotInSource;
    };
}

/// SRC's tensors where its index maps them, each shard opened once.
const Stored = struct {
    gpa: Allocator,
    arena: Allocator,
    io: Io,
    dir: []const u8,
    map: std.json.ObjectMap,
    files: std.StringArrayHashMapUnmanaged(st.File) = .empty,

    /// `base` ++ `suffix`, refused if the index does not list it or its shard does not hold it.
    fn get(s: *Stored, base: []const u8, suffix: []const u8) !st.Tensor {
        const name = try std.mem.concat(s.arena, u8, &.{ base, suffix });
        const shard = s.map.get(name) orelse return error.CompactExpertNotInSource;
        if (shard != .string) return error.BadIndex;
        const at = try s.files.getOrPut(s.gpa, shard.string);
        if (!at.found_existing) at.value_ptr.* = st.File.open(s.gpa, s.io, try join(s.arena, s.dir, shard.string)) catch |e| {
            s.files.swapRemoveAt(at.index);
            return e;
        };
        return at.value_ptr.get(name) orelse error.CompactExpertNotInSource;
    }

    fn close(s: *Stored) void {
        for (s.files.values()) |*f| f.close(s.io);
        s.files.deinit(s.gpa);
    }
};

/// --compact-experts' mask, refused if it does not parse, prunes a layer SRC lacks or keeps an id SRC lacks.
fn heldExperts(arena: Allocator, io: Io, path: []const u8, experts: []const usize) !expert_mask.Mask {
    const text = try Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 20));
    var problem: []const u8 = "";
    const mask = try expert_mask.parse(arena, text, &problem) orelse return error.BadCompactExperts;
    for (mask, 0..) |row, l| {
        if (expert_mask.kept(row) == cfg.experts) continue; // all of them, as for a layer the file leaves out
        const n = if (l < experts.len) experts[l] else 0;
        if (n == 0) return error.CompactLayerHasNoExperts;
        for (n..cfg.experts) |e| if (holds(&mask, l, e)) return error.CompactExpertNotInSource;
    }
    return mask;
}

/// The source's FP8 expert weight shapes [n, k] of the layers `wanted` marks, from the shards' headers.
fn sourceShapes(gpa: Allocator, arena: Allocator, io: Io, src: []const u8, shards: []const []const u8, wanted: []const bool) !encoded_.Shapes {
    var out: encoded_.Shapes = .empty;
    for (shards) |shard| {
        var file = try st.File.open(gpa, io, try join(arena, src, shard));
        defer file.close(io);
        for (file.names.keys(), file.names.values()) |name, e| {
            const at = parse(name) orelse continue;
            if (at.layer >= wanted.len or !wanted[at.layer]) continue;
            if (e.dtype != .f8_e4m3 or e.rank != 2) return error.ExpectedFp8Expert;
            try out.put(arena, try arena.dupe(u8, name), .{ e.shape[0], e.shape[1] });
        }
    }
    return out;
}

fn join(arena: Allocator, dir: []const u8, name: []const u8) ![]u8 {
    return std.fs.path.join(arena, &.{ dir, name });
}
