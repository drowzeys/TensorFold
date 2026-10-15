//! drama3-generate: Fish Audio Drama 3 on one CUDA GPU (fish_drama3), text (and voice-reference codes) in, frames out.
//!
//!   drama3-generate --model DIR --text "..." [--ref-text "..." --ref-codes FILE] [--rows FILE]
//!                   [--temperature 1.0] [--top-k 30] [--top-p 0.9] [--seed 1] [--greedy]
//!                   [--max-frames 1500] [--graphs 1] [--sync-every 8] [--out FILE] [--rows-out FILE]
//!                   [--max-ctx 4096] [--max-rows 1024] [--threads 16] [--repeat N] [--fp8-act 1]
//!   drama3-generate --model DIR --check DUMPDIR [--check-frames 40]
//!
//! --ref-codes: int32 little-endian [frames, 10] (or .npy int32/int64 [10, frames] / [frames, 10]); --rows: prompt
//! rows int32 [n, 11] that override --text. --out: the frames, int32 [n, 11] (main token, codes 0..9), for the codec
//! (tools/drama3/frames_to_wav.py). --check: the oracle's test vectors (DUMPDIR/vectors.safetensors, written by
//! drama3_ref.py --dump --greedy): its prompt rows run with a probe on every layer (cosine and max |diff| against
//! the oracle's tensors), then greedy frames against its frames (first divergence with the oracle's logit margin).
//! Prints `RESULT drama3 ...`.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const d3 = @import("drama3");

const W = d3.config.frame_width;
const H = d3.config.hidden;

const Args = struct {
    model: []const u8 = "",
    text: ?[]const u8 = null,
    ref_text: ?[]const u8 = null,
    ref_codes: ?[]const u8 = null,
    rows: ?[]const u8 = null,
    rows_out: ?[]const u8 = null,
    out: ?[]const u8 = null,
    check: ?[]const u8 = null,
    check_frames: usize = 40,
    smp: d3.Sampling = .{},
    max_frames: usize = 1500,
    graphs: bool = true,
    sync_every: usize = 8,
    opts: d3.Options = .{},
    repeat: usize = 1,
};

const usage =
    \\usage: drama3-generate --model DIR --text "..." [--ref-text "..." --ref-codes FILE] [--rows FILE]
    \\                       [--temperature 1.0] [--top-k 30] [--top-p 0.9] [--seed 1] [--greedy]
    \\                       [--max-frames 1500] [--graphs 1] [--sync-every 8] [--out FILE] [--rows-out FILE]
    \\                       [--max-ctx 4096] [--max-rows 1024] [--threads 16] [--repeat N] [--fp8-act 1]
    \\       drama3-generate --model DIR --check DUMPDIR [--check-frames 40]
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (std.mem.eql(u8, k, "--greedy")) {
            a.smp.temperature = 0;
            a.smp.top_k = 1;
            continue;
        }
        if (i + 1 >= argv.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = argv[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--model")) a.model = v else if (std.mem.eql(u8, k, "--text")) a.text = v else if (std.mem.eql(u8, k, "--ref-text")) a.ref_text = v else if (std.mem.eql(u8, k, "--ref-codes")) a.ref_codes = v else if (std.mem.eql(u8, k, "--rows")) a.rows = v else if (std.mem.eql(u8, k, "--rows-out")) a.rows_out = v else if (std.mem.eql(u8, k, "--out")) a.out = v else if (std.mem.eql(u8, k, "--check")) a.check = v else if (std.mem.eql(u8, k, "--check-frames")) a.check_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--temperature")) a.smp.temperature = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--top-k")) a.smp.top_k = try std.fmt.parseInt(i32, v, 10) else if (std.mem.eql(u8, k, "--top-p")) a.smp.top_p = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, k, "--seed")) a.smp.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--max-frames")) a.max_frames = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--graphs")) a.graphs = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--sync-every")) a.sync_every = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-ctx")) a.opts.max_ctx = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--max-rows")) a.opts.max_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--threads")) a.opts.threads = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--repeat")) a.repeat = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--exact-attn")) a.opts.exact_attn_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, k, "--fp8-act")) a.opts.fp8_act = !std.mem.eql(u8, v, "0") else {
            std.debug.print("unknown flag {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.model.len == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    a.opts.max_frames = @max(a.opts.max_frames, a.max_frames);
    if (a.check) |dir| return check(gpa, io, a, dir);

    // prompt rows
    var rows: [][W]i32 = undefined;
    if (a.rows) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
        if (bytes.len % (W * 4) != 0) return error.BadRows;
        rows = @alignCast(std.mem.bytesAsSlice([W]i32, bytes));
    } else {
        const text = a.text orelse {
            std.debug.print("--text or --rows is required\n{s}", .{usage});
            return 2;
        };
        const tpath = try std.fs.path.join(arena, &.{ a.model, "tokenizer.json" });
        var tok = try core.tokenizer.loadTokenizer(io, gpa, tpath);
        defer tok.deinit();
        var refs: [1]d3.prompt.Reference = undefined;
        var nref: usize = 0;
        if (a.ref_codes) |p| {
            const codes = try readCodes(gpa, io, p);
            refs[0] = .{ .transcript = a.ref_text orelse return error.RefTextRequired, .codes = codes };
            nref = 1;
        }
        rows = try d3.prompt.build(gpa, &tok, text, refs[0..nref]);
    }
    if (a.rows_out) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = std.mem.sliceAsBytes(rows) });
    std.log.info("drama3: {d} prompt rows", .{rows.len});

    const t_load = d3.engine.nowNs(io);
    const e = try d3.Engine.create(gpa, io, a.model, a.opts);
    defer e.destroy();
    const load_s = @as(f64, @floatFromInt(d3.engine.nowNs(io) - t_load)) / 1e9;
    const mi = try e.ctx.memInfo();
    std.log.info("drama3: loaded in {d:.1} s; device free {d:.1} GB of {d:.1} GB", .{ load_s, @as(f64, @floatFromInt(mi.free)) / 1e9, @as(f64, @floatFromInt(mi.total)) / 1e9 });
    var last: ?d3.engine.Engine.Result = null;
    for (0..a.repeat) |r| {
        const res = try e.generate(io, rows, a.smp, a.max_frames, a.graphs, a.sync_every);
        const fps = @as(f64, @floatFromInt(res.frames.len)) / (res.decode_ms / 1e3);
        std.log.info("drama3 run {d}: {d} frames ({s}), prefill {d:.1} ms, first frame {d:.1} ms, decode {d:.1} ms = {d:.2} frames/s ({d:.2}x real time)", .{ r, res.frames.len, if (res.eos) "eos" else "length", res.prefill_ms, res.first_frame_ms, res.decode_ms, fps, fps / 21.533 });
        if (last) |l| gpa.free(l.frames.ptr[0..l.frames.len]);
        last = res;
    }
    const res = last.?;
    if (a.out) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = std.mem.sliceAsBytes(res.frames) });
    const fps = @as(f64, @floatFromInt(res.frames.len)) / (res.decode_ms / 1e3);
    const mu = cuda.usage(false);
    std.debug.print("RESULT drama3 OK frames={d} eos={} prompt_rows={d} prefill_ms={d:.1} first_frame_ms={d:.1} fps={d:.2} rtf={d:.2} device_peak_gb={d:.2}\n", .{ res.frames.len, res.eos, rows.len, res.prefill_ms, res.first_frame_ms, fps, fps / 21.533, @as(f64, @floatFromInt(mu.peak)) / 1e9 });
    return 0;
}

/// Reference codes: raw int32 [frames, 10].
fn readCodes(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]const [10]i32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    if (!std.mem.endsWith(u8, path, ".npy")) {
        if (bytes.len % 40 != 0) return error.BadCodes;
        return @alignCast(std.mem.bytesAsSlice([10]i32, bytes));
    }
    return error.NpyCodesUnsupported; // tools/drama3/encode_reference.py writes the raw form
}

// ------------------------------------------------------------------------------------------------ oracle check

const Check = struct {
    gpa: std.mem.Allocator,
    e: *d3.Engine,
    ref: *core.safetensors.File,
    worst: f64 = 1.0,
    n: usize = 0,
    fails: usize = 0,
    // frame 0's heads (0: the candidate logits, 1 + i: codebook head i): both sides' top two
    tops: [10]?[2]Top2 = @splat(null),

    fn refF32(c: *Check, name: []const u8, rows: usize, width: usize) !?[]f32 {
        const t = c.ref.get(name) orelse return null;
        const n = t.numel();
        // the oracle's decode0 rows are [width]; prompt rows [rows, width]: compare the last `rows * width` values
        if (n < rows * width) return null;
        const out = try c.gpa.alloc(f32, rows * width);
        const skip = n - rows * width;
        switch (t.dtype) {
            .bf16 => {
                const v = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, t.bytes));
                for (out, 0..) |*o, j| o.* = @bitCast(@as(u32, v[skip + j]) << 16);
            },
            .f32 => {
                const v = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, t.bytes));
                for (out, 0..) |*o, j| o.* = v[skip + j];
            },
            else => return null,
        }
        return out;
    }

    fn probe(ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, is_f32: bool) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        const mine = try c.gpa.alloc(f32, rows * width);
        defer c.gpa.free(mine);
        if (is_f32) {
            try c.e.download(ptr, std.mem.sliceAsBytes(mine));
        } else {
            const raw = try c.gpa.alloc(u16, rows * width);
            defer c.gpa.free(raw);
            try c.e.download(ptr, std.mem.sliceAsBytes(raw));
            for (mine, raw) |*m, r| m.* = @bitCast(@as(u32, r) << 16);
        }
        if (std.mem.startsWith(u8, name, "dbg.")) {
            for ([_]usize{ 100, 140, 147, 149, 153 }) |r| if (r < rows) {
                var ss: f64 = 0;
                for (mine[r * width ..][0..width]) |x| ss += @as(f64, x) * x;
                std.debug.print("  {s} row {d} |x| {d:.3}\n", .{ name, r, @sqrt(ss) });
            };
            return;
        }
        if (std.mem.startsWith(u8, name, "router.")) {
            // the 8th / 9th routing probabilities of each row: a near tie flips an expert
            for (0..rows) |r| {
                const lg = mine[r * width + 1024 ..][0..256];
                var mx: f32 = -std.math.inf(f32);
                for (lg) |x| mx = @max(mx, x);
                var p: [256]f32 = undefined;
                var sum: f32 = 0;
                for (lg, 0..) |x, j| {
                    p[j] = @exp(x - mx);
                    sum += p[j];
                }
                std.mem.sort(f32, &p, {}, std.sort.desc(f32));
                const gap = (p[7] - p[8]) / sum;
                const li = std.fmt.parseInt(usize, name[12..14], 10) catch 0;
                if (r == rows - 2 and li >= 30) std.debug.print("  {s} row {d}: 8th / 9th routing probability {d:.5} / {d:.5} (gap {d:.5})\n", .{ name, r, p[7] / sum, p[8] / sum, gap });
            }
            return;
        }
        const ref = (try c.refF32(name, rows, width)) orelse {
            std.debug.print("  {s:<36} (no oracle tensor)\n", .{name});
            return;
        };
        defer c.gpa.free(ref);
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        var maxd: f64 = 0;
        var maxr: f64 = 0;
        for (mine, ref) |x, y| {
            dot += @as(f64, x) * y;
            na += @as(f64, x) * x;
            nb += @as(f64, y) * y;
            maxd = @max(maxd, @abs(@as(f64, x) - y));
            maxr = @max(maxr, @abs(@as(f64, y)));
        }
        // the worst row's cosine too
        var worst_row: f64 = 1;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
            }
            if (a2 > 0 and b2 > 0) worst_row = @min(worst_row, d / @sqrt(a2 * b2));
        }
        const cos = if (na > 0 and nb > 0) dot / @sqrt(na * nb) else 1.0;
        if (std.mem.startsWith(u8, name, "tower_decode0.head") or std.mem.eql(u8, name, "tower_decode0.candidate_logits")) {
            const hi: usize = if (name[18] >= '0' and name[18] <= '9') 1 + (name[18] - '0') else 0;
            c.tops[hi] = .{ top2(mine), top2(ref) };
        }
        c.n += 1;
        c.worst = @min(c.worst, worst_row);
        var wr: usize = 0;
        var wv: f64 = 2;
        for (0..rows) |r| {
            var d: f64 = 0;
            var a2: f64 = 0;
            var b2: f64 = 0;
            for (mine[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
                d += @as(f64, x) * y;
                a2 += @as(f64, x) * x;
                b2 += @as(f64, y) * y;
            }
            const cr = if (a2 > 0 and b2 > 0) d / @sqrt(a2 * b2) else 1.0;
            if (cr < wv) {
                wv = cr;
                wr = r;
            }
            if (cr < 0.99 and std.c.getenv("D3_ROWS") != null) std.debug.print("    {s} row {d} cos {d:.4}\n", .{ name, r, cr });
        }
        const ok = worst_row >= 0.999;
        if (!ok) c.fails += 1;
        std.debug.print("  {s:<36} rows {d:>4} cos {d:.6} worst-row {d:.6} (row {d}) max|d| {d:.4} (max|ref| {d:.2}) {s}\n", .{ name, rows, cos, worst_row, wr, maxd, maxr, if (ok) "ok" else "LOW" });
    }
};

fn check(gpa: std.mem.Allocator, io: std.Io, a: Args, dir: []const u8) !u8 {
    const vpath = try std.fs.path.join(gpa, &.{ dir, "vectors.safetensors" });
    var ref = try core.safetensors.File.open(gpa, io, vpath);
    defer ref.close(io);
    const pr = ref.get("prompt_rows") orelse return error.NoPromptRows;
    const n = pr.dim(0);
    if (pr.dim(1) != W or pr.dtype != .i64) return error.BadPromptRows;
    const rows = try gpa.alloc([W]i32, n);
    defer gpa.free(rows);
    const v = std.mem.bytesAsSlice(i64, @as([]align(1) const u8, pr.bytes));
    for (rows, 0..) |*r, j| for (r, 0..) |*x, c| {
        x.* = @intCast(v[j * W + c]);
    };
    var opts = a.opts;
    opts.max_rows = @max(opts.max_rows, n);
    const e = try d3.Engine.create(gpa, io, a.model, opts);
    defer e.destroy();
    var c: Check = .{ .gpa = gpa, .e = e, .ref = &ref };
    const probe: d3.engine.Probe = .{ .ctx = &c, .f = Check.probe };
    std.debug.print("drama3 check: {d} prompt rows from {s}\n", .{ n, vpath });
    e.probe_router = probe;
    try e.prefill(rows, probe);
    e.probe_router = null;
    if (ref.get("backbone_final_norm")) |hf| if (hf.dtype == .bf16 and hf.dim(0) == n) {
        std.debug.print("  -- the tower prefill again from the oracle's backbone_final_norm rows:\n", .{});
        try e.towerOnly(n, hf.bytes, probe);
        try e.prefill(rows, null); // restore my own state for the decode comparison
    };
    // frame 0 with the probe (tower_decode0.*), then greedy frames against the oracle's
    const smp: d3.Sampling = .{ .temperature = 0, .top_k = 1 };
    const oref = ref.get("frames_all") orelse ref.get("frames") orelse return error.NoFrames;
    const nref = oref.dim(0);
    const ov = std.mem.bytesAsSlice(i64, @as([]align(1) const u8, oref.bytes));
    const cand_ref = ref.get("decode.candidate_logits");
    const nc = e.w.n_cand;
    const mine_logits = try gpa.alloc(f32, nc);
    defer gpa.free(mine_logits);
    var same: usize = 0;
    var diverged = false;
    const want = @min(a.check_frames, nref + 1);
    var f: usize = 0;
    var got: [W]i32 = undefined;
    while (f < want) : (f += 1) {
        // my candidate logits for this frame (the step recomputes them; read them back after it)
        try e.frameStep(smp, false, if (f == 0) probe else null);
        try e.download(e.cand_logits, std.mem.sliceAsBytes(mine_logits));
        try e.download(e.frames_out + f * W * 4, std.mem.asBytes(&got));
        if (f >= nref) {
            std.debug.print("  frame {d}: mine {d} (the oracle stopped at {d} frames: eos expected)\n", .{ f, got[0], nref });
            if (got[0] == e.c.ids.eos) same += 1;
            break;
        }
        var match = true;
        for (0..W) |j| match = match and got[j] == ov[f * W + j];
        if (match) {
            same += 1;
            continue;
        }
        diverged = true;
        std.debug.print("  frame {d} differs: mine [", .{f});
        for (got) |x| std.debug.print(" {d}", .{x});
        std.debug.print(" ] oracle [", .{});
        for (0..W) |j| std.debug.print(" {d}", .{ov[f * W + j]});
        std.debug.print(" ]\n", .{});
        var col: usize = 0;
        while (col < W and got[col] == ov[f * W + col]) col += 1;
        const head: usize = if (col <= 1) 0 else col - 1;
        if (f == 0) if (c.tops[head]) |t| {
            std.debug.print("    first difference: column {d} ({s} {d}); mine top {d} ({d:.4}) second {d} ({d:.4}) margin {d:.4}; oracle top {d} ({d:.4}) second {d} ({d:.4}) margin {d:.4}\n", .{ col, if (head == 0) "candidate logits" else "codebook head", if (head == 0) 0 else head - 1, t[0].i1, t[0].v1, t[0].i2, t[0].v2, t[0].v1 - t[0].v2, t[1].i1, t[1].v1, t[1].i2, t[1].v2, t[1].v1 - t[1].v2 });
        };
        // the main token's margin on both sides (next frame's candidate logits)
        const mm = top2(mine_logits);
        std.debug.print("    my candidate logits: top {d} ({d:.4}) second {d} ({d:.4})\n", .{ mm.i1, mm.v1, mm.i2, mm.v2 });
        if (cand_ref) |cr| if (f < cr.dim(0)) {
            const rv = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, cr.bytes))[f * nc ..][0..nc];
            const rcopy = try gpa.alloc(f32, nc);
            defer gpa.free(rcopy);
            @memcpy(rcopy, rv);
            const rm = top2(rcopy);
            std.debug.print("    oracle candidate logits: top {d} ({d:.4}) second {d} ({d:.4}); mine at the oracle's top: {d:.4}\n", .{ rm.i1, rm.v1, rm.i2, rm.v2, mine_logits[rm.i1] });
        };
        break;
    }
    _ = &diverged;
    // teacher forcing: the oracle's frames fed back; my greedy picks against them, my decode rows' final-norm hidden
    // against the oracle's (decode.backbone_final_norm, its first dump-frames rows)
    var tf_agree: [W]usize = @splat(0);
    var tf_frames: usize = 0;
    var tf_worst: f64 = 1;
    {
        try e.prefill(rows, null);
        const dev = try cuda.DeviceBuffer.alloc(&e.driver, 2 * W * 4);
        var devb = dev;
        defer devb.free();
        const hidden_ref = ref.get("decode.backbone_final_norm");
        const hbuf = try gpa.alloc(u16, H);
        defer gpa.free(hbuf);
        const nf = @min(nref, a.check_frames * 5);
        for (0..nf) |fi| {
            if (hidden_ref) |hr| if (fi < hr.dim(0)) {
                try e.download(e.hlast, std.mem.sliceAsBytes(hbuf));
                const rv = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, hr.bytes))[fi * H ..][0..H];
                var d: f64 = 0;
                var a2: f64 = 0;
                var b2: f64 = 0;
                for (hbuf, rv) |x16, y16| {
                    const x: f64 = @as(f32, @bitCast(@as(u32, x16) << 16));
                    const y: f64 = @as(f32, @bitCast(@as(u32, y16) << 16));
                    d += x * y;
                    a2 += x * x;
                    b2 += y * y;
                }
                tf_worst = @min(tf_worst, d / @sqrt(a2 * b2));
            };
            var fr: [W]i32 = undefined;
            for (0..W) |j| fr[j] = @intCast(ov[fi * W + j]);
            try devb.upload(0, std.mem.asBytes(&fr));
            try e.frameForced(.{ .frame = dev.ptr, .picks = dev.ptr + W * 4 });
            var picks: [W]i32 = undefined;
            try e.download(dev.ptr + W * 4, std.mem.asBytes(&picks));
            for (0..W) |j| tf_agree[j] += @intFromBool(picks[j] == fr[j]);
            tf_frames += 1;
        }
        std.debug.print("  teacher-forced {d} frames: my greedy picks equal the oracle's: main {d}", .{ tf_frames, tf_agree[0] });
        for (2..W) |j| std.debug.print(" c{d} {d}", .{ j - 1, tf_agree[j] });
        std.debug.print("; decode-row final-norm hidden worst cos {d:.6} over the oracle's dumped rows\n", .{tf_worst});
    }
    const pass = c.fails == 0 and (same >= @min(20, want));
    var tf_total: usize = 0;
    for (tf_agree) |x| tf_total += x;
    std.debug.print("RESULT drama3 check {s}: {d} probes, {d} below cos 0.999 (worst row {d:.6}); greedy frames identical {d} of {d} (oracle made {d}); teacher-forced picks {d:.2}% of {d}\n", .{ if (pass) "PASS" else "FAIL", c.n, c.fails, c.worst, same, want, nref, 100.0 * @as(f64, @floatFromInt(tf_total - tf_agree[1])) / @as(f64, @floatFromInt(@max(1, tf_frames * (W - 1)))), tf_frames * (W - 1) });
    return if (pass) 0 else 1;
}

const Top2 = struct { i1: usize, v1: f32, i2: usize, v2: f32 };

fn top2(x: []const f32) Top2 {
    var r: Top2 = .{ .i1 = 0, .v1 = -std.math.inf(f32), .i2 = 0, .v2 = -std.math.inf(f32) };
    for (x, 0..) |v, j| {
        if (v > r.v1) {
            r.i2 = r.i1;
            r.v2 = r.v1;
            r.i1 = j;
            r.v1 = v;
        } else if (v > r.v2) {
            r.i2 = j;
            r.v2 = v;
        }
    }
    return r;
}
