//! tf-drama-probe: requests through the production prompt pass and frame loop, every step's semantic logits kept.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const core = @import("core");
const drama = @import("drama");
const api = @import("engine_api");
const host = @import("drama_host");
const server = @import("server");
const Rows = @import("drama_probe_rows.zig").Rows;
const steps = @import("drama_probe_steps.zig");

const usage = "usage: tf-drama-probe CHECKPOINT VOICES_DIR REQUESTS.jsonl OUT_DIR [--stats] [--rows DIR] [--force CODES_DIR] [--prompt-chunk N]\n";
const tool_commit = std.mem.trim(u8, @embedFile("tool_commit"), " \n");
/// Production's context: every request's prompt and frames fit, as the server admits them.
const context = 8192;
const cols = drama.backbone.weights.head_cols;
const layers = drama.backbone.config.layers;
const Frame = steps.Frame;
const Codes = server.drama_prompt.Codes;
const Finish = steps.Finish;
const Forced = steps.Forced;
const Allocator = std.mem.Allocator;

comptime {
    std.debug.assert(builtin.cpu.arch.endian() == .little); // the output files are little-endian
    std.debug.assert(Frame == drama.tower.Frame);
}

/// A request lowered before the model loads: its rows and seed as the server makes them, and any forced frames.
const Job = struct { r: server.drama_probe.Request, rows: server.drama_prompt.Rows, prompt_sha256: []const u8, seed: u64, forced: ?Forced };

/// One request's manifest entry.
const Result = struct {
    id: []const u8,
    voice: ?[]const u8,
    salt: i64,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    max_frames: u32,
    prompt_rows: usize,
    prompt_sha256: []const u8,
    frames: usize,
    steps: usize,
    seed: u64,
    finish: Finish,
    forced_from: ?[]const u8,
    codes_sha256: []const u8,
    logits_sha256: []const u8,
    tower_sha256: ?[]const u8,
    seconds: f64,
};

/// The backbone's tracer when --stats or --rows asks for one: each routed hook goes to what was asked.
const Hooks = struct {
    stats: ?*drama.moe_stats.Tracer,
    rows: ?*Rows,

    fn generated(h: *Hooks, on: bool) void {
        if (h.stats) |t| t.generated = on;
        if (h.rows) |r| r.ledger.generated = on;
    }

    pub fn routed(h: *Hooks, s: cuda.Stream, layer: usize, r: drama.backbone.Routed) !void {
        if (h.stats) |t| try t.routed(s, layer, r);
        if (h.rows) |w| try w.routed(s, layer, r);
    }
};

/// The tower's tracer under --force: each head's NLL of the frame's forced code, T = 1, from its bf16 logits in f64.
const Scorer = struct {
    frame: Frame = undefined,
    nll: [steps.heads]f32 = undefined,
    logits: [4096]u16 = undefined,

    pub fn head(sc: *Scorer, s: cuda.Stream, h: usize, logits: cuda.DeviceBuffer) !void {
        try s.synchronize();
        try logits.download(0, std.mem.asBytes(&sc.logits));
        sc.nll[h] = try steps.headNll(&sc.logits, sc.frame[2 + h]); // head h reads frame[1 + h], predicts frame[2 + h]
    }
};

/// steps.run's model: the production backbone, tower and host step functions, with the run's hooks.
const Probe = struct {
    gpa: Allocator,
    backbone: *drama.backbone.Model,
    tower: *drama.tower.Model,
    stream: cuda.Stream,
    chunk: usize,
    hooks: ?*Hooks,
    scorer: Scorer = .{},
    job: Job = undefined,
    row: host.Row = undefined,
    frames: [host.chunk_rows + 1]Frame = undefined,
    codes: std.ArrayList(Codes) = .empty,
    logits: std.ArrayList(f32) = .empty,
    nll: std.ArrayList([steps.heads]f32) = .empty,
    /// Rows fed after the prompt in the last run.
    fed: usize = 0,

    fn run(p: *Probe, job: Job) !Finish {
        p.job = job;
        p.logits.clearRetainingCapacity();
        p.nll.clearRetainingCapacity();
        p.fed = 0;
        return steps.run(p, p.gpa, p.backbone.semantic_start, job.forced, job.r.max_frames, &p.codes);
    }

    pub fn prompt(p: *Probe) !void {
        const r = p.job.rows;
        if (p.hooks) |h| {
            h.generated(false);
            p.row = try host.prefill(p.backbone, p.tower, p.stream, &p.frames, r.tokens, r.codes, p.chunk, p.job.seed, h, {});
        } else p.row = try host.prefill(p.backbone, p.tower, p.stream, &p.frames, r.tokens, r.codes, p.chunk, p.job.seed, {}, {});
    }

    pub fn save(p: *Probe) !void {
        const row = try p.logits.addManyAsSlice(p.gpa, cols);
        try p.row.logits.download(0, std.mem.sliceAsBytes(row));
        if (!steps.finite(row)) return error.NonFiniteSemanticLogits; // forcing never reaches the sampler's check
    }

    pub fn sample(p: *Probe) !?Frame {
        const sampling: drama.tower.Sampling = .{ .temperature = p.job.r.temperature, .top_p = p.job.r.top_p, .top_k = p.job.r.top_k };
        const frame = try p.tower.generate(p.stream, p.row.logits, p.row.hidden, sampling);
        return if (try host.speechEnds(p.backbone, frame)) null else frame;
    }

    pub fn force(p: *Probe, frame: Frame) !void {
        p.frames[0] = frame;
        p.scorer.frame = frame;
        try p.tower.prefillTraced(p.stream, p.row.hidden, p.frames[0..1], &p.scorer);
        try p.nll.append(p.gpa, p.scorer.nll);
    }

    pub fn feed(p: *Probe, frame: Frame) !void {
        p.frames[0] = frame;
        if (p.hooks) |h| {
            h.generated(true);
            p.row = try host.feed(p.backbone, p.tower, p.stream, p.frames[0..1], h);
        } else p.row = try host.feed(p.backbone, p.tower, p.stream, p.frames[0..1], {});
        p.fed += 1;
    }
};

fn fail(comptime fmt: []const u8, args: anytype) u8 {
    std.debug.print("tf-drama-probe: " ++ fmt ++ "\n", args);
    return 2;
}

fn sha256Hex(a: Allocator, parts: []const []const u8) ![]const u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (parts) |part| h.update(part);
    return a.dupe(u8, &std.fmt.bytesToHex(h.finalResult(), .lower));
}

/// Every request line lowered, or null once one is refused (said on stderr): nothing loads for a bad file.
fn lower(a: Allocator, io: std.Io, text: server.model_text.Text, voices: []const server.drama_voice.Voice, lines: []const u8, force: ?[]const u8) !?[]Job {
    var problem: []const u8 = "";
    const teacher: ?steps.Teacher = if (force) |dir| blk: {
        const path = try std.fs.path.join(a, &.{ std.fs.path.dirname(dir) orelse ".", "manifest.json" });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
        break :blk try steps.teacher(a, bytes, &problem) orelse {
            std.debug.print("tf-drama-probe: --force {s}: {s}\n", .{ dir, problem });
            return null;
        };
    } else null;
    var jobs: std.ArrayList(Job) = .empty;
    var ids: std.StringHashMapUnmanaged(void) = .empty;
    var it = std.mem.splitScalar(u8, lines, '\n');
    var number: usize = 0;
    while (it.next()) |line| {
        number += 1;
        if (std.mem.trim(u8, line, " \t\r").len == 0) continue;
        problem = "";
        const r = try server.drama_probe.parse(a, line, &problem) orelse return refuse(number, problem);
        if ((try ids.getOrPut(a, r.id)).found_existing) return refuse(number, "id repeats an earlier request's");
        var cx: server.errors.Cx = .{ .a = a };
        const rows = server.drama_probe.prompt(&cx, text, voices, r) catch |e| switch (e) {
            error.Refused => return refuse(number, cx.message),
            else => |x| return x,
        };
        const cap = host.speechInfo(context, r.max_frames).?.max_prompt_tokens;
        if (rows.tokens.len > cap) return refuse(number, try std.fmt.allocPrint(a, "{s} a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ server.drama_prompt.error_prefix, rows.tokens.len, cap }));
        const prompt_sha256 = try sha256Hex(a, &.{ std.mem.sliceAsBytes(rows.tokens), std.mem.sliceAsBytes(rows.codes) });
        var forced: ?Forced = null;
        if (teacher) |t| {
            const path = try std.fmt.allocPrint(a, "{s}/{s}.u32", .{ force.?, r.id });
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(@as(usize, r.max_frames) * @sizeOf(Codes) + 1));
            forced = try steps.forced(a, t, r.id, rows.tokens.len, prompt_sha256, r.max_frames, bytes, &problem) orelse return refuse(number, problem);
        }
        try jobs.append(a, .{ .r = r, .rows = rows, .prompt_sha256 = prompt_sha256, .seed = api.seedFor(rows.tokens, r.salt), .forced = forced });
    }
    if (jobs.items.len == 0) return refuse(number, "no requests");
    return jobs.items;
}

fn refuse(line: usize, why: []const u8) ?[]Job {
    std.debug.print("tf-drama-probe: REQUESTS line {d}: {s}\n", .{ line, why });
    return null;
}

/// The routed experts' format of each layer, by the C1 rule: expert 0's gate_proj dtype.
fn expertFormats(a: Allocator, io: std.Io, dir: []const u8) ![layers][]const u8 {
    var ck = try core.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    var out: [layers][]const u8 = undefined;
    for (&out, 0..) |*f, l| {
        const t = try ck.get(try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.experts.0.gate_proj.weight", .{l}));
        f.* = switch (t.dtype) {
            .f8_e4m3 => "fp8",
            .u8 => "nvfp4",
            else => @tagName(t.dtype),
        };
    }
    return out;
}

/// Each named voice file's sha256, as a JSON object.
fn voiceDigests(a: Allocator, io: std.Io, dir: []const u8, jobs: []const Job) ![]const u8 {
    var out: std.json.ObjectMap = .empty;
    for (jobs) |job| {
        const name = job.r.voice orelse continue;
        if (std.ascii.eqlIgnoreCase(name, "default") or out.contains(name)) continue;
        const path = try std.fmt.allocPrint(a, "{s}/{s}.json", .{ dir, name });
        try out.put(a, name, .{ .string = try sha256Hex(a, &.{try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20))}) });
    }
    return std.json.Stringify.valueAlloc(a, std.json.Value{ .object = out }, .{});
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    var stats = false;
    var rows_dir: ?[]const u8 = null;
    var force: ?[]const u8 = null;
    var chunk: usize = host.chunk_rows;
    var paths: [4][]const u8 = undefined;
    var given: usize = 0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        const value = i + 1 < args.len;
        if (std.mem.eql(u8, arg, "--stats")) {
            stats = true;
        } else if (value and std.mem.eql(u8, arg, "--rows")) {
            i += 1;
            rows_dir = args[i];
        } else if (value and std.mem.eql(u8, arg, "--force")) {
            i += 1;
            force = args[i];
        } else if (value and std.mem.eql(u8, arg, "--prompt-chunk")) {
            i += 1;
            chunk = std.fmt.parseInt(usize, args[i], 10) catch 0;
            if (chunk == 0 or chunk > host.chunk_rows) return fail("--prompt-chunk must be 1..{d}", .{host.chunk_rows});
        } else if (given < paths.len and !std.mem.startsWith(u8, arg, "--")) {
            paths[given] = arg;
            given += 1;
        } else return fail("{s}", .{usage});
    }
    if (given != paths.len) return fail("{s}", .{usage});
    const checkpoint, const voices_dir, const requests_path, const out_path = paths;
    const mode = if (force == null) "generate" else "force";

    const requests = try std.Io.Dir.cwd().readFileAlloc(io, requests_path, a, .limited(64 << 20));
    var problem: []const u8 = "";
    const voices = try server.drama_voice.load(a, io, voices_dir, &problem) orelse return fail("{s}", .{problem});
    const text = server.hf_text.HfText.load(gpa, io, checkpoint, null, a, &problem) catch |e| return fail("tokenizer: {s}", .{if (problem.len > 0) problem else @errorName(e)});
    defer text.deinit();
    const jobs = try lower(a, io, text.text(), voices, requests, force) orelse return 2;
    const formats = try expertFormats(a, io, checkpoint);
    const config_path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
    const config_sha = try sha256Hex(a, &.{try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(1 << 20))});
    const requests_sha = try sha256Hex(a, &.{requests});
    try std.Io.Dir.cwd().createDir(io, out_path, .default_dir);
    var out = try std.Io.Dir.cwd().openDir(io, out_path, .{});
    defer out.close(io);
    const subdirs = [_][]const u8{ "codes", "logits", "tower" };
    const outputs: usize = if (force == null) 2 else 3; // tower NLL only under --force
    for (subdirs[0..outputs]) |sub| try out.createDir(io, sub, .default_dir);

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var backbone = try drama.backbone.Model.init(gpa, io, &ctx, checkpoint, "", .{ .context = context, .max_rows = host.chunk_rows });
    defer backbone.deinit();
    var tower = try drama.tower.Model.init(gpa, io, &ctx, checkpoint, "", .{ .context = context, .max_rows = host.chunk_rows });
    defer tower.deinit();
    var tracer: ?drama.moe_stats.Tracer = if (stats) try drama.moe_stats.Tracer.init(&driver, &backbone.k, .{ backbone.semantic_start, backbone.semantic_end }, host.chunk_rows) else null;
    defer if (tracer) |*t| t.deinit();
    var dump: ?Rows = if (rows_dir) |dir| try Rows.create(gpa, io, &driver, dir, host.chunk_rows, .{ backbone.semantic_start, backbone.semantic_end }) else null;
    defer if (dump) |*r| r.deinit();
    var hooks: Hooks = .{ .stats = if (tracer) |*t| t else null, .rows = if (dump) |*r| r else null };
    const totals = try a.alloc(f64, if (stats) layers * drama.moe_stats.layer_len else 0);
    @memset(totals, 0);
    const staging = try a.alloc(f64, totals.len);

    var probe: Probe = .{ .gpa = gpa, .backbone = &backbone, .tower = &tower, .stream = stream, .chunk = chunk, .hooks = if (stats or rows_dir != null) &hooks else null };
    defer {
        probe.codes.deinit(gpa);
        probe.logits.deinit(gpa);
        probe.nll.deinit(gpa);
    }
    var results: std.ArrayList(Result) = .empty;
    var spans: std.ArrayList(struct { id: []const u8, first: usize, rows: usize, prompt_rows: usize, prompt_sha256: []const u8 }) = .empty;
    var sampling: std.ArrayList(struct { id: []const u8, salt: i64, seed: u64, temperature: f32, top_p: f32, top_k: u32 }) = .empty;
    var class_rows: [3]usize = @splat(0);
    for (jobs) |job| {
        const began = std.Io.Clock.awake.now(io).toNanoseconds();
        if (dump) |*r| r.ledger.begin(job.r.id);
        const finish = try probe.run(job);
        if (tracer) |*t| try t.sums.drain(stream, totals, staging);
        const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - began)) / 1e9;
        if (dump) |r| try spans.append(a, .{ .id = job.r.id, .first = r.ledger.first, .rows = r.ledger.rows - r.ledger.first, .prompt_rows = job.rows.tokens.len, .prompt_sha256 = job.prompt_sha256 });
        try sampling.append(a, .{ .id = job.r.id, .salt = job.r.salt, .seed = job.seed, .temperature = job.r.temperature, .top_p = job.r.top_p, .top_k = job.r.top_k });
        for (job.rows.tokens) |t| class_rows[@intFromBool(t >= backbone.semantic_start and t <= backbone.semantic_end)] += 1;
        class_rows[2] += probe.fed;
        const files = [_]struct { dir: []const u8, ext: []const u8, bytes: []const u8 }{
            .{ .dir = "codes", .ext = "u32", .bytes = std.mem.sliceAsBytes(probe.codes.items) },
            .{ .dir = "logits", .ext = "f32", .bytes = std.mem.sliceAsBytes(probe.logits.items) },
            .{ .dir = "tower", .ext = "f32", .bytes = std.mem.sliceAsBytes(probe.nll.items) },
        };
        var digests: [files.len][]const u8 = undefined;
        for (files[0..outputs], digests[0..outputs]) |f, *digest| {
            try out.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.{s}", .{ f.dir, job.r.id, f.ext }), .data = f.bytes, .flags = .{ .exclusive = true } });
            digest.* = try sha256Hex(a, &.{f.bytes});
        }
        const frames = probe.codes.items.len;
        try results.append(a, .{
            .id = job.r.id,
            .voice = job.r.voice,
            .salt = job.r.salt,
            .temperature = job.r.temperature,
            .top_p = job.r.top_p,
            .top_k = job.r.top_k,
            .max_frames = job.r.max_frames,
            .prompt_rows = job.rows.tokens.len,
            .prompt_sha256 = job.prompt_sha256,
            .frames = frames,
            .steps = probe.logits.items.len / cols,
            .seed = job.seed,
            .finish = finish,
            .forced_from = force,
            .codes_sha256 = digests[0],
            .logits_sha256 = digests[1],
            .tower_sha256 = if (force == null) null else digests[2],
            .seconds = seconds,
        });
        std.debug.print("{s}: prompt {d} rows, {d} frames, {d} steps, {t}, {d:.1} s\n", .{ job.r.id, job.rows.tokens.len, frames, probe.logits.items.len / cols, finish, seconds });
    }

    const identity = .{ .tool_commit = tool_commit, .checkpoint = checkpoint, .config_sha256 = config_sha, .requests_file = requests_path, .requests_sha256 = requests_sha, .mode = mode, .prompt_chunk = chunk };
    if (stats) {
        const metadata = [_][2][]const u8{
            .{ "tool_commit", tool_commit },
            .{ "checkpoint", checkpoint },
            .{ "requests_sha256", requests_sha },
            .{ "requests", try std.fmt.allocPrint(a, "{d}", .{jobs.len}) },
            .{ "rows_prompt_text", try std.fmt.allocPrint(a, "{d}", .{class_rows[0]}) },
            .{ "rows_prompt_audio", try std.fmt.allocPrint(a, "{d}", .{class_rows[1]}) },
            .{ "rows_generated", try std.fmt.allocPrint(a, "{d}", .{class_rows[2]}) },
            .{ "mode", mode },
            .{ "sampling", try std.json.Stringify.valueAlloc(a, sampling.items, .{}) },
            .{ "voices", try voiceDigests(a, io, voices_dir, jobs) },
        };
        try drama.moe_stats.write(gpa, try std.fs.path.join(a, &.{ out_path, "stats.safetensors" }), totals, &metadata);
        try out.writeFile(io, .{ .sub_path = "mass.json", .data = try drama.moe_stats.massJson(a, totals), .flags = .{ .exclusive = true } });
    }
    if (dump) |r| {
        const rows_json = try std.json.Stringify.valueAlloc(a, .{
            .schema = "drama-probe-rows/1",
            .identity = identity,
            .rows = r.ledger.rows,
            .layers = layers,
            .class = .{ .file = "class.u8", .dtype = "u8", .shape = .{r.ledger.rows}, .values = "0 prompt_text, 1 prompt_audio, 2 generated" },
            .x = .{ .file = "layer-LL.x.bf16", .dtype = "bf16", .shape = .{ r.ledger.rows, drama.backbone.config.hidden }, .source = "Scratch.normed, the MoE input" },
            .picks = .{ .file = "layer-LL.picks.i32", .dtype = "i32", .shape = .{ r.ledger.rows, drama.backbone.config.top_k } },
            .w = .{ .file = "layer-LL.w.f32", .dtype = "f32", .shape = .{ r.ledger.rows, drama.backbone.config.top_k }, .source = "renormalized routing weights, slot order" },
            .requests = spans.items,
            .chunks = r.ledger.chunks.items,
        }, .{ .whitespace = .indent_2 });
        try r.dir.writeFile(io, .{ .sub_path = "rows.json", .data = rows_json, .flags = .{ .exclusive = true } });
    }
    const manifest = try std.json.Stringify.valueAlloc(a, .{
        .schema = "drama-probe/1",
        .producer = "tf-drama-probe",
        .identity = identity,
        .expert_formats = formats,
        .voices = voices_dir,
        .forced_from = force,
        .stats = if (stats) "stats.safetensors" else null,
        .rows = rows_dir,
        .codes = "codes/ID.u32: u32le [frames][10], codebook 0 first",
        .logits = "logits/ID.f32: f32le [steps][4097], EOS then the 4096 semantic ids; steps = frames + 1 on eos, frames on length",
        .tower = if (force == null) null else "tower/ID.f32: f32le [frames][9], NLL of codebooks 1..9 under their heads at T = 1",
        .requests = results.items,
    }, .{ .whitespace = .indent_2 });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest, .flags = .{ .exclusive = true } });
    return 0;
}

test {
    _ = steps;
    _ = @import("drama_probe_rows.zig");
}
