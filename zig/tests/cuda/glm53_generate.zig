//! tf-glm53-generate: full GLM-5.3 at tensor parallelism in Zig (Phase 2a), one process a rank: every layer, the final
//! norm and the vocabulary-sharded argmax head; NCCL all-gathers between ranks; GREEDY decode without drafts. For each
//! prompt (token ids, e.g. the chat template's with thinking on): the prompt in windows of up to --window rows (the
//! last window's last row picks the first token), then one-row decode windows, --tokens picks in all (EOS is not a
//! stop: every run makes the same number of picks). Each pick's gathered [world, 4] fp32 records (every rank's max
//! logit and its id) are kept bit for bit, so tools/glm53/compare_phase2a.py can check per-step logit-argmax equality
//! with tools/glm53/reference.py (the Python engine on the same schedule) besides the tokens.
//!
//! usage: tf-glm53-generate --rank R --world W --master IP:PORT --model DIR --aot DIR --prompts FILE --tokens N
//!                          --out FILE [--context 32768] [--window 128] [--layers N] [--inv-ref FILE]
//!                          [--nccl-lib PATH] [--timeout S]
//! Prints `RESULT glm53-generate OK|FAIL ...`; exit 0 only on OK.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm53");

const usage =
    \\usage: tf-glm53-generate --rank R --world W --master IP:PORT --model DIR --aot DIR --prompts FILE --tokens N
    \\                         --out FILE [--context 32768] [--window 128] [--layers N] [--inv-ref FILE]
    \\                         [--nccl-lib PATH] [--timeout S]
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
};

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
        } else {
            std.debug.print("unknown argument {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.model.len == 0 or a.aot.len == 0 or a.prompts.len == 0 or a.out.len == 0 or a.rank >= a.world or a.tokens == 0 or a.window == 0 or a.window > glm.forward.max_rows) {
        std.debug.print("{s}", .{usage});
        return 2;
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
    var w = try glm.weights.load(gpa, io, &driver, a.model, cfg, .{ .layers = n_layers, .rank = a.rank, .world = a.world, .head = true, .log_every = 10 });
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
