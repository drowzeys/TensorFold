//! tf-glm53-generate: full GLM-5.3 at tensor parallelism in Zig (Phase 2a), one process a rank: every layer, the final
//! norm and the vocabulary-sharded argmax head; NCCL all-gathers between ranks; GREEDY decode without drafts. For each
//! prompt (token ids, e.g. the chat template's with thinking on): the prompt in windows of up to --window rows (the
//! last window's last row picks the first token), then one-row decode windows, --tokens picks in all (EOS is not a
//! stop: every run makes the same number of picks). Each pick's gathered [world, 4] fp32 records (every rank's max
//! logit and its id) are kept bit for bit, so tools/glm53/compare_phase2a.py can check per-step logit-argmax equality
//! with tools/glm53/reference.py (the Python engine on the same schedule) besides the tokens.
//!
//!
//! --mode 2b (Phase 2b, cuda_runner.zig): the Python Runner's schedule - the prompt in equal chunks of up to --window
//! rows through the target and the MTP layer, then MTP-drafted rounds (--k drafts, verify windows of k + 1 rows) as
//! CUDA graphs (prewarmed: every bucket's windows captured before the first request), greedy or keyed sampled draws.
//! Each prompt of the prompts file names its runs ("g:K" greedy, "s<top_k>:K" sampled at the file's temperature /
//! top_p / min_p with the prompt's seed; K drafts, 0 = serial) and its token count. Prints per run the prefill time,
//! decode tok/s, tokens a round and acceptance; writes them with the tokens to --out for compare_phase2b.py. The
//! weights load with O_DIRECT through io_uring (--fast-load 1), the MTP layer and the 4-bit draft head
//! (--draft-vocab 32768, fused.DRAFT_VOCAB) included.
//! --sampler-check FILE (no GPU): sampling.zig's draws and seeds against cases tools/glm53/sampler_cases.py wrote
//! with the Python engine's exact_sampling; prints `SAMPLER PASS|FAIL ...`.
//!
//! usage: tf-glm53-generate --rank R --world W --master IP:PORT --model DIR --aot DIR --prompts FILE --tokens N
//!                          --out FILE [--context 32768] [--window 128] [--layers N] [--inv-ref FILE]
//!                          [--nccl-lib PATH] [--timeout S] [--mode 2a|2b] [--k 2] [--prewarm 1] [--graphs 1]
//!                          [--fast-load 0|1] [--draft-vocab 32768] [--mtp-reuse 2]
//!        tf-glm53-generate --sampler-check FILE
//! Prints `RESULT glm53-generate OK|FAIL ...`; exit 0 only on OK.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm53");

const usage =
    \\usage: tf-glm53-generate --rank R --world W --master IP:PORT --model DIR --aot DIR --prompts FILE --tokens N
    \\                         --out FILE [--context 32768] [--window 128] [--layers N] [--inv-ref FILE]
    \\                         [--nccl-lib PATH] [--timeout S] [--mode 2a|2b] [--k 2] [--prewarm 1] [--graphs 1]
    \\                         [--fast-load 0|1] [--draft-vocab 32768] [--mtp-reuse 2]
    \\       tf-glm53-generate --sampler-check FILE
    \\
;

const Args = struct {
    rank: usize = 0,
    world: usize = 1,
    master: []const u8 = "127.0.0.1:29581",
    model: []const u8 = "",
    aot: []const u8 = "",
    prompts: []const u8 = "",
    out: []const u8 = "",
    tokens: usize = 128,
    context: usize = 32768,
    window: usize = 128,
    layers: ?usize = null,
    inv_ref: ?[]const u8 = null,
    nccl_lib: []const u8 = "libnccl.so.2",
    timeout_s: f64 = 3600,
    mode: []const u8 = "2a",
    k: usize = 2,
    prewarm: bool = true,
    graphs: bool = true,
    fast_load: ?bool = null, // default: on in 2b, off in 2a
    draft_vocab: usize = 32768,
    mtp_reuse: u32 = 2,
    sampler_check: ?[]const u8 = null,
};

fn flag(v: []const u8) !bool {
    if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true")) return true;
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "false")) return false;
    return error.BadFlag;
}

const Prompt = struct { name: []const u8, ids: []u32 };

const Result = struct {
    tokens: std.ArrayList(u32) = .empty,
    records: std.ArrayList([glm.forward.max_world * 4]u32) = .empty,
    prefill_s: f64 = 0,
    decode_s: f64 = 0,
};

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (i + 1 >= argv.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = argv[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--rank")) {
            a.rank = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--world")) {
            a.world = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--master")) {
            a.master = v;
        } else if (std.mem.eql(u8, k, "--model")) {
            a.model = v;
        } else if (std.mem.eql(u8, k, "--aot")) {
            a.aot = v;
        } else if (std.mem.eql(u8, k, "--prompts")) {
            a.prompts = v;
        } else if (std.mem.eql(u8, k, "--out")) {
            a.out = v;
        } else if (std.mem.eql(u8, k, "--tokens")) {
            a.tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--context")) {
            a.context = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--window")) {
            a.window = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--layers")) {
            a.layers = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--inv-ref")) {
            a.inv_ref = v;
        } else if (std.mem.eql(u8, k, "--nccl-lib")) {
            a.nccl_lib = v;
        } else if (std.mem.eql(u8, k, "--timeout")) {
            a.timeout_s = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, k, "--mode")) {
            a.mode = v;
        } else if (std.mem.eql(u8, k, "--k")) {
            a.k = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--prewarm")) {
            a.prewarm = try flag(v);
        } else if (std.mem.eql(u8, k, "--graphs")) {
            a.graphs = try flag(v);
        } else if (std.mem.eql(u8, k, "--fast-load")) {
            a.fast_load = try flag(v);
        } else if (std.mem.eql(u8, k, "--draft-vocab")) {
            a.draft_vocab = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--mtp-reuse")) {
            a.mtp_reuse = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--sampler-check")) {
            a.sampler_check = v;
        } else {
            std.debug.print("unknown argument {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.sampler_check) |path| {
        const good = samplerCheck(init.gpa, init.io, path) catch |e| {
            std.debug.print("SAMPLER FAIL error {t}\n", .{e});
            return 1;
        };
        return if (good) 0 else 1;
    }
    if (a.model.len == 0 or a.aot.len == 0 or a.prompts.len == 0 or a.out.len == 0 or a.rank >= a.world or a.tokens == 0 or a.window == 0 or a.window > glm.forward.max_rows) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const two_b = std.mem.eql(u8, a.mode, "2b");
    if (!two_b and !std.mem.eql(u8, a.mode, "2a")) {
        std.debug.print("--mode 2a or 2b\n{s}", .{usage});
        return 2;
    }
    if (two_b) {
        const ok2 = run2b(init.gpa, init.io, a) catch |e| {
            std.debug.print("RESULT glm53-generate FAIL rank {d}: error {t}\n", .{ a.rank, e });
            return 1;
        };
        return if (ok2) 0 else 1;
    }
    const ok = run(init.gpa, init.io, a) catch |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: error {t}\n", .{ a.rank, e });
        return 1;
    };
    return if (ok) 0 else 1;
}

fn now(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn since(io: std.Io, t0: i128) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e9;
}

fn readPrompts(gpa: std.mem.Allocator, io: std.Io, path: []const u8, text_out: *[]u8) ![]Prompt {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    text_out.* = text;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const list = (parsed.value.object.get("prompts") orelse return error.BadPrompts).array.items;
    const out = try gpa.alloc(Prompt, list.len);
    for (list, out) |pv, *p| {
        const o = pv.object;
        const ids = (o.get("ids") orelse return error.BadPrompts).array.items;
        p.name = try gpa.dupe(u8, (o.get("name") orelse return error.BadPrompts).string);
        p.ids = try gpa.alloc(u32, ids.len);
        for (ids, p.ids) |x, *y| y.* = switch (x) {
            .integer => |n| std.math.cast(u32, n) orelse return error.BadPrompts,
            else => return error.BadPrompts,
        };
        if (p.ids.len == 0) return error.BadPrompts;
    }
    return out;
}

fn run(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    var prompts_text: []u8 = &.{};
    const prompts = try readPrompts(gpa, io, a.prompts, &prompts_text);
    defer {
        for (prompts) |p| {
            gpa.free(p.name);
            gpa.free(p.ids);
        }
        gpa.free(prompts);
        gpa.free(prompts_text);
    }
    std.debug.print("rank {d} of {d}: {d} prompts x {d} picks, context {d}, windows of {d} rows\n", .{ a.rank, a.world, prompts.len, a.tokens, a.context, a.window });

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    std.debug.print("device: {s} sm_{d}\n", .{ try ctx.name(&name_buf), try ctx.capability() });

    // -- the world: TCP rendezvous (rank 0's NCCL id), then ncclCommInitRank, before the load (as comm.NCCL) ------
    const master = try cuda.rendezvous.parseMaster(a.master);
    var lib = try cuda.nccl.Library.openPath(a.nccl_lib);
    defer lib.close();
    const nccl_version = try lib.version();
    std.debug.print("nccl: {s} version {d}\n", .{ a.nccl_lib, nccl_version });
    var rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, a.rank, a.world, a.timeout_s);
    defer rdv.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (a.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&lib);
    try rdv.broadcast(std.mem.asBytes(&uid));
    var comm = try cuda.nccl.Communicator.init(&lib, uid, a.rank, a.world);
    defer comm.deinit();
    std.debug.print("rank {d}: NCCL communicator up (world {d})\n", .{ a.rank, a.world });

    const cfg = try glm.Config.load(gpa, io, a.model);
    const n_layers = a.layers orelse cfg.layers;
    var settings = std.hash.Wyhash.init(0x6a1b);
    const shape = [_]usize{ a.world, n_layers, a.context, a.tokens, a.window };
    settings.update(std.mem.asBytes(&shape));
    settings.update(prompts_text);
    try rdv.sync("settings", settings.final(), a.timeout_s);

    // -- weights: every layer, the head; layer by layer with the page cache dropped -----------------------------
    const t_load = now(io);
    var w = try glm.weights.load(gpa, io, &driver, a.model, cfg, .{ .layers = n_layers, .rank = a.rank, .world = a.world, .head = true, .log_every = 10, .fast = a.fast_load orelse false });
    defer w.deinit();
    const load_s = since(io, t_load);
    std.debug.print("rank {d}: {d} layers + head loaded in {d:.0} s ({d:.1} GiB of weights; vocabulary rows {d}..{d})\n", .{ a.rank, n_layers, load_s, @as(f64, @floatFromInt(w.device_bytes)) / (1 << 30), w.vocab_off, w.vocab_off + w.vocab_part });
    try rdv.sync("loaded", 0, a.timeout_s);

    var k = try glm.kernels.Kernels.load(gpa, io, &ctx, a.aot);
    defer k.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();

    // -- fused.inv_freq: torch's bytes, computed the way torch computes them; the reference's sidecar checks them --
    const half = cfg.rope / 2;
    var inv_buf = try cuda.DeviceBuffer.alloc(&driver, half * 4);
    defer inv_buf.free();
    try k.invFreq(stream, inv_buf.ptr, cfg.rope, cfg.rope_theta);
    try stream.synchronize();
    var inv_status: []const u8 = "device (no reference to check)";
    if (a.inv_ref) |path| {
        const ref = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
        defer gpa.free(ref);
        const mine = try gpa.alloc(u8, half * 4);
        defer gpa.free(mine);
        try inv_buf.download(0, mine);
        if (ref.len != mine.len) return error.BadInvReference;
        if (std.mem.eql(u8, ref, mine)) {
            inv_status = "device == reference";
        } else {
            var bad: usize = 0;
            for (0..half) |j| {
                if (!std.mem.eql(u8, ref[4 * j ..][0..4], mine[4 * j ..][0..4])) bad += 1;
            }
            std.debug.print("WARN inv_freq: {d} of {d} entries differ from torch's (the reference's bytes are used)\n", .{ bad, half });
            try inv_buf.upload(0, ref);
            inv_status = "DIFFERS: reference bytes used";
        }
    }
    std.debug.print("inv_freq: {s}\n", .{inv_status});

    const fwd: glm.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv_buf.ptr, .comm = if (a.world > 1) &comm else null };
    const topk = cfg.index_topk;
    const score_cols = glm.triton.bucket(a.context, topk) orelse @max(a.context, 64);
    var b = try glm.Buffers.init(gpa, &driver, &w, a.window, score_cols);
    defer b.deinit();
    var st = try glm.State.init(gpa, &driver, &w, a.context);
    defer st.deinit();
    const host = try gpa.alloc(f32, w.vocab_part);
    defer gpa.free(host);

    const results = try gpa.alloc(Result, prompts.len);
    defer {
        for (results) |*r| {
            r.tokens.deinit(gpa);
            r.records.deinit(gpa);
        }
        gpa.free(results);
    }
    for (results) |*r| r.* = .{};
    var all_tokens = std.hash.Wyhash.init(0);
    var decode_steps: usize = 0;
    var decode_s: f64 = 0;
    for (prompts, results) |p, *res| {
        const P = p.ids.len;
        if (P + a.tokens > a.context) {
            std.log.err("{s}: {d} prompt tokens + {d} picks exceed the context {d}", .{ p.name, P, a.tokens, a.context });
            return error.ContextTooSmall;
        }
        try st.reset();
        const t_pf = now(io);
        var pick: glm.forward.Pick = undefined;
        var s0: usize = 0;
        while (s0 < P) {
            const e = @min(s0 + a.window, P);
            const R = e - s0;
            try fwd.embed(&b, p.ids[s0..e]);
            try fwd.setPos(&st, s0);
            const T = glm.triton.bucket(e, topk);
            for (0..n_layers) |li| try fwd.layer(&b, &st, li, R, T);
            if (e == P) pick = try fwd.head(&b, R, host);
            s0 = e;
        }
        res.prefill_s = since(io, t_pf);
        try keep(gpa, res, pick);
        const t_dec = now(io);
        var pos = P;
        while (res.tokens.items.len < a.tokens) : (pos += 1) {
            const last = [1]u32{res.tokens.items[res.tokens.items.len - 1]};
            try fwd.embed(&b, &last);
            try fwd.setPos(&st, pos);
            const T = glm.triton.bucket(pos + 1, topk);
            for (0..n_layers) |li| try fwd.layer(&b, &st, li, 1, T);
            pick = try fwd.head(&b, 1, host);
            try keep(gpa, res, pick);
        }
        res.decode_s = since(io, t_dec);
        decode_steps += a.tokens - 1;
        decode_s += res.decode_s;
        all_tokens.update(std.mem.sliceAsBytes(res.tokens.items));
        std.debug.print("rank {d} {s}: prompt {d} tokens in {d:.1} s, {d} picks in {d:.1} s; first picks {any}\n", .{ a.rank, p.name, P, res.prefill_s, a.tokens, res.decode_s, res.tokens.items[0..@min(8, res.tokens.items.len)] });
    }

    try writeOut(gpa, io, a, &w, n_layers, nccl_version, inv_status, load_s, prompts, results);
    // every rank picks from the same gathered records: the token streams must agree
    const agree = rdv.sync("done", all_tokens.final(), a.timeout_s);
    const tps = if (decode_s > 0) @as(f64, @floatFromInt(decode_steps)) / decode_s else 0;
    if (agree) {
        std.debug.print("RESULT glm53-generate OK rank {d}: {d} prompts x {d} picks, eager decode {d:.2} tok/s, inv_freq {s}\n", .{ a.rank, prompts.len, a.tokens, tps, inv_status });
        return true;
    } else |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: the ranks' token streams differ ({t})\n", .{ a.rank, e });
        return false;
    }
}

fn keep(gpa: std.mem.Allocator, res: *Result, p: glm.forward.Pick) !void {
    try res.tokens.append(gpa, p.token);
    var bits: [glm.forward.max_world * 4]u32 = @splat(0);
    for (0..p.world * 4) |j| bits[j] = @bitCast(p.gathered[j]);
    try res.records.append(gpa, bits);
}

fn writeOut(gpa: std.mem.Allocator, io: std.Io, a: Args, w: *const glm.weights.Weights, n_layers: usize, nccl_version: c_int, inv_status: []const u8, load_s: f64, prompts: []const Prompt, results: []const Result) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const head = try std.fmt.allocPrint(gpa, "{{\"tool\": \"tf-glm53-generate\", \"rank\": {d}, \"world\": {d}, \"layers\": {d}, \"context\": {d}, \"window\": {d}, \"tokens\": {d}, \"vocab_off\": {d}, \"vocab_part\": {d}, \"nccl_version\": {d}, \"inv_freq\": \"{s}\", \"load_s\": {d:.1}, \"prompts\": [", .{ a.rank, a.world, n_layers, a.context, a.window, a.tokens, w.vocab_off, w.vocab_part, nccl_version, inv_status, load_s });
    defer gpa.free(head);
    try out.appendSlice(gpa, head);
    for (prompts, results, 0..) |p, r, pi| {
        if (pi > 0) try out.appendSlice(gpa, ", ");
        const ph = try std.fmt.allocPrint(gpa, "\n {{\"name\": \"{s}\", \"prompt_len\": {d}, \"prefill_s\": {d:.3}, \"decode_s\": {d:.3}, \"tokens\": [", .{ p.name, p.ids.len, r.prefill_s, r.decode_s });
        defer gpa.free(ph);
        try out.appendSlice(gpa, ph);
        for (r.tokens.items, 0..) |t, j| {
            var buf: [16]u8 = undefined;
            try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}{d}", .{ if (j > 0) ", " else "", t }));
        }
        try out.appendSlice(gpa, "],\n  \"records\": [");
        for (r.records.items, 0..) |rec, j| {
            try out.appendSlice(gpa, if (j > 0) ",\n   \"" else "\n   \"");
            for (0..a.world * 4) |q| {
                var buf: [16]u8 = undefined;
                try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}{x:0>8}", .{ if (q > 0) "," else "", rec[q] }));
            }
            try out.appendSlice(gpa, "\"");
        }
        try out.appendSlice(gpa, "]}");
    }
    try out.appendSlice(gpa, "\n]}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = out.items });
    std.debug.print("rank {d}: wrote {s}\n", .{ a.rank, a.out });
}

// ------------------------------------------------------------------------------------------------- Phase 2b ---

/// A run of a prompt: greedy or sampled (top_k; 0 = top_p alone), with `k` drafts a round (0: serial).
const RunSpec = struct { greedy: bool, top_k: usize, k: usize, label: []const u8 };

const Prompt2b = struct {
    name: []const u8,
    ids: []u32,
    tokens: usize,
    seed: u64,
    seed_from_prompt: bool,
    runs: []RunSpec,
};

fn tf(b: bool) []const u8 {
    return if (b) "true" else "false";
}

const Sampling2b = struct { temperature: f64 = 1.0, top_p: f64 = 0.95, min_p: f64 = 0.0 };

fn parseRun(gpa: std.mem.Allocator, text: []const u8) !RunSpec {
    const colon = std.mem.indexOfScalar(u8, text, ':') orelse return error.BadRun;
    const mode = text[0..colon];
    const k = try std.fmt.parseInt(usize, text[colon + 1 ..], 10);
    const label = try gpa.dupe(u8, text);
    if (std.mem.eql(u8, mode, "g")) return .{ .greedy = true, .top_k = 0, .k = k, .label = label };
    if (mode.len >= 2 and mode[0] == 's') return .{ .greedy = false, .top_k = try std.fmt.parseInt(usize, mode[1..], 10), .k = k, .label = label };
    return error.BadRun;
}

fn jsonFloat(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => error.BadPrompts,
    };
}

fn jsonU64(v: std.json.Value) !u64 {
    return switch (v) {
        .integer => |n| std.math.cast(u64, n) orelse error.BadPrompts,
        .number_string => |t| std.fmt.parseInt(u64, t, 10),
        .string => |t| std.fmt.parseInt(u64, t, 10),
        else => error.BadPrompts,
    };
}

fn readPrompts2b(gpa: std.mem.Allocator, io: std.Io, path: []const u8, default_tokens: usize, text_out: *[]u8, smp_out: *Sampling2b) ![]Prompt2b {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    text_out.* = text;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    if (root.get("sampling")) |sv| {
        const so = sv.object;
        if (so.get("temperature")) |x| smp_out.temperature = try jsonFloat(x);
        if (so.get("top_p")) |x| smp_out.top_p = try jsonFloat(x);
        if (so.get("min_p")) |x| smp_out.min_p = try jsonFloat(x);
    }
    const list = (root.get("prompts") orelse return error.BadPrompts).array.items;
    const out = try gpa.alloc(Prompt2b, list.len);
    for (list, out) |pv, *p| {
        const o = pv.object;
        const ids = (o.get("ids") orelse return error.BadPrompts).array.items;
        p.name = try gpa.dupe(u8, (o.get("name") orelse return error.BadPrompts).string);
        p.ids = try gpa.alloc(u32, ids.len);
        for (ids, p.ids) |x, *y| y.* = switch (x) {
            .integer => |n| std.math.cast(u32, n) orelse return error.BadPrompts,
            else => return error.BadPrompts,
        };
        if (p.ids.len == 0) return error.BadPrompts;
        p.tokens = if (o.get("tokens")) |t| @intCast(try jsonU64(t)) else default_tokens;
        p.seed_from_prompt = if (o.get("seed_from_prompt")) |b| b == .bool and b.bool else false;
        p.seed = if (o.get("seed")) |sd| try jsonU64(sd) else glm.sampling.seedFor(p.ids, 0);
        const runs = if (o.get("runs")) |r| r.array.items else &[_]std.json.Value{};
        p.runs = try gpa.alloc(RunSpec, runs.len);
        for (runs, p.runs) |rv, *rs| rs.* = try parseRun(gpa, rv.string);
    }
    return out;
}

const Done = struct {
    prompt: usize,
    run: usize,
    tokens: []u32,
    st: glm.runner.Stats,
    seed_ok: bool,
};

fn run2b(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    var prompts_text: []u8 = &.{};
    var sset: Sampling2b = .{};
    const prompts = try readPrompts2b(gpa, io, a.prompts, a.tokens, &prompts_text, &sset);
    defer gpa.free(prompts_text);
    defer {
        for (prompts) |p| {
            gpa.free(p.name);
            gpa.free(p.ids);
            for (p.runs) |rs| gpa.free(rs.label);
            gpa.free(p.runs);
        }
        gpa.free(prompts);
    }
    var n_runs: usize = 0;
    for (prompts) |p| n_runs += p.runs.len;
    std.debug.print("rank {d} of {d}: phase 2b, {d} prompts, {d} runs, context {d}, prompt chunks of up to {d} rows, MTP drafts {d}, graphs {s}, prewarm {s}\n", .{ a.rank, a.world, prompts.len, n_runs, a.context, a.window, a.k, tf(a.graphs), tf(a.prewarm) });

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    std.debug.print("device: {s} sm_{d}\n", .{ try ctx.name(&name_buf), try ctx.capability() });

    const master = try cuda.rendezvous.parseMaster(a.master);
    var lib = try cuda.nccl.Library.openPath(a.nccl_lib);
    defer lib.close();
    const nccl_version = try lib.version();
    std.debug.print("nccl: {s} version {d}\n", .{ a.nccl_lib, nccl_version });
    var rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, a.rank, a.world, a.timeout_s);
    defer rdv.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (a.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&lib);
    try rdv.broadcast(std.mem.asBytes(&uid));
    var comm = try cuda.nccl.Communicator.init(&lib, uid, a.rank, a.world);
    defer comm.deinit();
    std.debug.print("rank {d}: NCCL communicator up (world {d})\n", .{ a.rank, a.world });

    const cfg = try glm.Config.load(gpa, io, a.model);
    const n_layers = a.layers orelse cfg.layers;
    var settings = std.hash.Wyhash.init(0x6a2b);
    const shape = [_]usize{ a.world, n_layers, a.context, a.tokens, a.window, a.k, a.draft_vocab, a.mtp_reuse, @intFromBool(a.graphs), @intFromBool(a.prewarm) };
    settings.update(std.mem.asBytes(&shape));
    settings.update(prompts_text);
    try rdv.sync("settings", settings.final(), a.timeout_s);

    // -- weights: every layer, the head, the MTP layer and the 4-bit draft head; O_DIRECT reads a layer ahead ------
    const fast = a.fast_load orelse true;
    const t_load = now(io);
    var w = try glm.weights.load(gpa, io, &driver, a.model, cfg, .{ .layers = n_layers, .rank = a.rank, .world = a.world, .head = true, .log_every = 10, .fast = fast, .mtp = a.k > 0, .draft_vocab = if (a.k > 0) a.draft_vocab else 0 });
    defer w.deinit();
    const load_s = since(io, t_load);
    const read_gib = @as(f64, @floatFromInt(w.read_bytes)) / (1 << 30);
    std.debug.print("rank {d}: {d} layers{s} + head loaded in {d:.0} s ({d:.1} GiB of weights; {s}: {d:.1} GiB read, {d} files direct, {d} fallback reads; draft head {d} rows)\n", .{ a.rank, n_layers, if (w.mtp != null) " + MTP layer" else "", load_s, @as(f64, @floatFromInt(w.device_bytes)) / (1 << 30), if (fast) "O_DIRECT + io_uring" else "mapped reads", read_gib, w.direct_files, w.read_fallbacks, if (w.draft) |dh| dh.n else 0 });
    try rdv.sync("loaded", 0, a.timeout_s);

    var k = try glm.kernels.Kernels.load(gpa, io, &ctx, a.aot);
    defer k.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();

    const half = cfg.rope / 2;
    var inv_buf = try cuda.DeviceBuffer.alloc(&driver, half * 4);
    defer inv_buf.free();
    try k.invFreq(stream, inv_buf.ptr, cfg.rope, cfg.rope_theta);
    try stream.synchronize();
    var inv_status: []const u8 = "device (no reference to check)";
    if (a.inv_ref) |path| {
        const ref = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
        defer gpa.free(ref);
        const mine = try gpa.alloc(u8, half * 4);
        defer gpa.free(mine);
        try inv_buf.download(0, mine);
        if (ref.len != mine.len) return error.BadInvReference;
        if (std.mem.eql(u8, ref, mine)) {
            inv_status = "device == reference";
        } else {
            std.debug.print("WARN inv_freq differs from torch's (the reference's bytes are used)\n", .{});
            try inv_buf.upload(0, ref);
            inv_status = "DIFFERS: reference bytes used";
        }
    }
    std.debug.print("inv_freq: {s}\n", .{inv_status});

    const fwd: glm.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv_buf.ptr, .comm = if (a.world > 1) &comm else null };
    const capacity = a.context + a.k + 1; // Glm53Engine: Runner(limit + k + 1)
    var r = try glm.Runner.init(gpa, io, &driver, fwd, capacity, .{ .k = a.k, .graphs = a.graphs, .window = a.window, .reuse = a.mtp_reuse });
    defer r.deinit();
    var prewarm_s: f64 = 0;
    var prewarm_graphs: usize = 0;
    if (a.prewarm) {
        const tp = now(io);
        prewarm_graphs = try r.prewarm();
        prewarm_s = since(io, tp);
        std.debug.print("rank {d}: {d} decode graphs captured in {d:.1} s\n", .{ a.rank, prewarm_graphs, prewarm_s });
    }
    try rdv.sync("warm", prewarm_graphs, a.timeout_s);

    var done: std.ArrayList(Done) = .empty;
    defer {
        for (done.items) |d| gpa.free(d.tokens);
        done.deinit(gpa);
    }
    var toks: std.ArrayList(u32) = .empty;
    defer toks.deinit(gpa);
    var all_tokens = std.hash.Wyhash.init(0);
    var seeds_ok = true;
    for (prompts, 0..) |p, pi| {
        const seed_ok = !p.seed_from_prompt or glm.sampling.seedFor(p.ids, 0) == p.seed;
        if (!seed_ok) {
            std.debug.print("WARN {s}: seed_for(prompt) {d} here, {d} in the prompts file\n", .{ p.name, glm.sampling.seedFor(p.ids, 0), p.seed });
            seeds_ok = false;
        }
        for (p.runs, 0..) |rs, ri| {
            const s: ?glm.sampling.Sampling = if (rs.greedy) null else .{ .seed = p.seed, .temperature = sset.temperature, .top_k = rs.top_k, .top_p = sset.top_p, .min_p = sset.min_p };
            if (rs.k > a.k) return error.TooManyDrafts;
            const st = try r.generate(p.ids, p.tokens, s, rs.k, &toks);
            all_tokens.update(std.mem.sliceAsBytes(toks.items));
            const n = toks.items.len;
            try done.append(gpa, .{ .prompt = pi, .run = ri, .tokens = try gpa.dupe(u32, toks.items), .st = st, .seed_ok = seed_ok });
            std.debug.print("rank {d} {s} {s}: prompt {d} in {d:.1} s; {d} tokens, decode {d:.2} tok/s, {d:.3} tokens/round ({d} rounds), accept {d:.3}, capture {d:.1} s; first {any}\n", .{ a.rank, p.name, rs.label, p.ids.len, st.prefill_s, n, st.tokS(n), st.perRound(n), st.rounds, if (st.drafted > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0, st.capture_s, toks.items[0..@min(8, n)] });
        }
    }

    try writeOut2b(gpa, io, a, &w, n_layers, capacity, nccl_version, inv_status, load_s, read_gib, prewarm_graphs, prewarm_s, prompts, done.items);
    const agree = rdv.sync("done", all_tokens.final(), a.timeout_s);
    if (agree) {
        std.debug.print("RESULT glm53-generate {s} rank {d}: phase 2b, {d} runs, load {d:.0} s, inv_freq {s}{s}\n", .{ if (seeds_ok) "OK" else "FAIL", a.rank, done.items.len, load_s, inv_status, if (seeds_ok) "" else ", seed_for differs from Python's" });
        return seeds_ok;
    } else |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: the ranks' token streams differ ({t})\n", .{ a.rank, e });
        return false;
    }
}

fn writeOut2b(gpa: std.mem.Allocator, io: std.Io, a: Args, w: *const glm.weights.Weights, n_layers: usize, capacity: usize, nccl_version: c_int, inv_status: []const u8, load_s: f64, read_gib: f64, prewarm_graphs: usize, prewarm_s: f64, prompts: []const Prompt2b, done: []const Done) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const head = try std.fmt.allocPrint(gpa, "{{\"tool\": \"tf-glm53-generate\", \"mode\": \"2b\", \"rank\": {d}, \"world\": {d}, \"layers\": {d}, \"context\": {d}, \"capacity\": {d}, \"window\": {d}, \"k\": {d}, \"draft_vocab\": {d}, \"mtp_reuse\": {d}, \"graphs\": {s}, \"vocab_off\": {d}, \"vocab_part\": {d}, \"nccl_version\": {d}, \"inv_freq\": \"{s}\", \"load_s\": {d:.1}, \"read_gib\": {d:.2}, \"direct_files\": {d}, \"read_fallbacks\": {d}, \"prewarm_graphs\": {d}, \"prewarm_s\": {d:.1}, \"runs\": [", .{ a.rank, a.world, n_layers, a.context, capacity, a.window, a.k, a.draft_vocab, a.mtp_reuse, tf(a.graphs), w.vocab_off, w.vocab_part, nccl_version, inv_status, load_s, read_gib, w.direct_files, w.read_fallbacks, prewarm_graphs, prewarm_s });
    defer gpa.free(head);
    try out.appendSlice(gpa, head);
    for (done, 0..) |d, i| {
        const p = prompts[d.prompt];
        const rs = p.runs[d.run];
        const n = d.tokens.len;
        const acc: f64 = if (d.st.drafted > 0) @as(f64, @floatFromInt(d.st.accepted)) / @as(f64, @floatFromInt(d.st.drafted)) else 0;
        const ph = try std.fmt.allocPrint(gpa, "{s}\n {{\"name\": \"{s}\", \"run\": \"{s}\", \"prompt_len\": {d}, \"seed\": \"{d}\", \"seed_ok\": {s}, \"k\": {d}, \"greedy\": {s}, \"top_k\": {d}, \"prefill_s\": {d:.3}, \"decode_s\": {d:.3}, \"tok_s\": {d:.3}, \"rounds\": {d}, \"tokens_per_round\": {d:.4}, \"accept\": {d:.4}, \"capture_s\": {d:.2}, \"graphs\": {d}, \"tokens\": [", .{ if (i > 0) "," else "", p.name, rs.label, p.ids.len, p.seed, tf(d.seed_ok), rs.k, tf(rs.greedy), rs.top_k, d.st.prefill_s, d.st.decode_s, d.st.tokS(n), d.st.rounds, d.st.perRound(n), acc, d.st.capture_s, d.st.graphs });
        defer gpa.free(ph);
        try out.appendSlice(gpa, ph);
        for (d.tokens, 0..) |t, j| {
            var buf: [16]u8 = undefined;
            try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}{d}", .{ if (j > 0) ", " else "", t }));
        }
        try out.appendSlice(gpa, "]}");
    }
    try out.appendSlice(gpa, "\n]}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = out.items });
    std.debug.print("rank {d}: wrote {s}\n", .{ a.rank, a.out });
}

/// --sampler-check: sampling.zig against the Python engine's exact_sampling on generated cases (no GPU).
fn samplerCheck(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    var bad: usize = 0;
    var bad_keep: usize = 0;
    var total: usize = 0;
    var cands: std.ArrayList(glm.sampling.Cand) = .empty;
    defer cands.deinit(gpa);
    for ((root.get("cases") orelse return error.BadCases).array.items) |cv| {
        const o = cv.object;
        const vals = o.get("values").?.array.items;
        const ids = o.get("ids").?.array.items;
        if (vals.len != ids.len) return error.BadCases;
        cands.clearRetainingCapacity();
        for (vals, ids) |v, id| try cands.append(gpa, .{ .v = @floatCast(try jsonFloat(v)), .id = id.integer });
        const s: glm.sampling.Sampling = .{
            .seed = try jsonU64(o.get("seed").?),
            .temperature = try jsonFloat(o.get("temperature").?),
            .top_k = @intCast(o.get("top_k").?.integer),
            .top_p = try jsonFloat(o.get("top_p").?),
            .min_p = try jsonFloat(o.get("min_p").?),
        };
        var nk: usize = 0;
        const got = glm.sampling.chooseKeep(cands.items, @intCast(o.get("position").?.integer), s, &nk);
        const want: u32 = @intCast(o.get("pick").?.integer);
        total += 1;
        if (got != want) {
            bad += 1;
            if (bad <= 10) std.debug.print("case {d}: zig {d}, python {d} (top_k {d}, top_p {d}, {d} candidates)\n", .{ total - 1, got, want, s.top_k, s.top_p, vals.len });
        }
        if (o.get("keep")) |kv| {
            if (nk != @as(usize, @intCast(kv.integer))) bad_keep += 1;
        }
    }
    var seed_bad: usize = 0;
    var seed_total: usize = 0;
    if (root.get("seeds")) |sv| {
        for (sv.array.items) |e| {
            const o = e.object;
            const list = o.get("ids").?.array.items;
            const ids = try gpa.alloc(u32, list.len);
            defer gpa.free(ids);
            for (list, ids) |x, *y| y.* = @intCast(x.integer);
            seed_total += 1;
            if (glm.sampling.seedFor(ids, o.get("salt").?.integer) != try jsonU64(o.get("seed").?)) seed_bad += 1;
        }
    }
    const ok = bad == 0 and bad_keep == 0 and seed_bad == 0 and total > 0;
    std.debug.print("SAMPLER {s} picks {d}/{d} nucleus sizes {d}/{d} seeds {d}/{d}\n", .{ if (ok) "PASS" else "FAIL", total - bad, total, total - bad_keep, total, seed_total - seed_bad, seed_total });
    return ok;
}
