//! A guided draw batched (decodePair) against unbatched, byte for byte; each sequence against itself run alone; and, the
//! draws fixed, one row's prompt shortened: the other row's logits and DeltaNet state must not move. A run makes STEPS
//! logit observations (the prompts' first) from STEPS - 1 paired decodes; a sampled EOS is fed on as a token, no exit.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");

const Frame = drama.tower.Frame;
const H = drama.backbone.config.hidden;
const cols = drama.backbone.weights.head_cols;
const context = 4096;
const max_rows = 128;
const gamma: f32 = 2;
const seed = 7;
/// Observations a run makes at least, so at least min_steps - 1 of them come from a decodePair; an EOS draw is fed on.
const min_steps = 8;

/// A pair run over fixed frames: l_c and l_u a step, and both sequences' DeltaNet state buffers' digests at the end.
const Fixed = struct {
    cond: std.ArrayList(f32) = .empty,
    uncond: std.ArrayList(f32) = .empty,
    cond_state: [32]u8 = undefined,
    uncond_state: [32]u8 = undefined,
};

/// What a decode gave at each step: the frame drawn, l_c, l_u and the conditioned sequence's hidden row.
const Trace = struct {
    frames: std.ArrayList(Frame) = .empty,
    cond: std.ArrayList(f32) = .empty,
    uncond: std.ArrayList(f32) = .empty,
    hidden: std.ArrayList(u16) = .empty,
};

const Rig = struct {
    a: std.mem.Allocator,
    stream: cuda.Stream,
    backbone: *drama.backbone.Model,
    tower: *drama.tower.Model,
    other: *drama.backbone.Sequence,
    guided: cuda.DeviceBuffer,

    fn lastRow(rows: drama.backbone.Rows) cuda.DeviceBuffer {
        return .{ .d = rows.hidden.d, .ptr = rows.hidden.ptr + (rows.rows - 1) * H * 2, .len = H * 2, .borrowed = true };
    }

    /// The prompt pass as drama_host stages it: one chunk, the tower taught every row but the last.
    fn prompt(r: *Rig, rows: []const Frame, tokens: []u32) !drama.backbone.Rows {
        for (rows, tokens) |row, *t| t.* = row[0];
        const out = try r.backbone.run(r.stream, .prefill, tokens, try r.tower.setInput(r.stream, rows));
        if (rows.len > 1) {
            try r.tower.prefill(r.stream, out.hidden, rows[1..]);
            _ = try r.tower.setInput(r.stream, rows[rows.len - 1 ..]);
        }
        return out;
    }

    /// A guided draw of up to `steps` frames, as drama_host and drama_guide make it, batched or not.
    fn draw(r: *Rig, cond: []const Frame, uncond: []const u32, batched: bool, steps: usize) !Trace {
        var t: Trace = .{};
        try r.backbone.reset(r.stream);
        try r.other.reset(r.stream);
        const tokens = try r.a.alloc(u32, cond.len);
        for (cond, tokens) |row, *tok| tok.* = row[0];
        try r.tower.reset(r.stream, tokens, seed);
        // the plain prompt first: every run writes the rows the tower reads
        var l_u: [cols]f32 = undefined;
        try (try r.backbone.runOn(r.other, r.stream, .prefill, uncond, null)).semantic_logits.download(0, std.mem.sliceAsBytes(&l_u));
        var rows = try r.prompt(cond, tokens);
        var frame: Frame = undefined;
        for (0..steps) |i| {
            if (i > 0) {
                const extra = try r.tower.setInput(r.stream, @as(*const [1]Frame, &frame));
                if (batched) {
                    rows = try r.backbone.decodePair(r.other, r.stream, frame[0], extra, r.guided);
                    try r.guided.download(0, std.mem.sliceAsBytes(&l_u));
                } else {
                    try (try r.backbone.runOn(r.other, r.stream, .decode, frame[0..1], extra)).semantic_logits.download(0, std.mem.sliceAsBytes(&l_u));
                    rows = try r.backbone.run(r.stream, .decode, frame[0..1], extra);
                }
            }
            const l_c = try t.cond.addManyAsSlice(r.a, cols);
            try rows.semantic_logits.download(0, std.mem.sliceAsBytes(l_c));
            try t.uncond.appendSlice(r.a, &l_u);
            try lastRow(rows).download(0, std.mem.sliceAsBytes(try t.hidden.addManyAsSlice(r.a, H)));
            var mixed: [cols]f32 = undefined;
            for (&mixed, l_c, l_u) |*m, c, u| m.* = c + (gamma - 1) * (c - u);
            try r.guided.upload(0, std.mem.sliceAsBytes(&mixed));
            frame = try r.tower.generate(r.stream, r.guided, lastRow(rows), .{});
            try t.frames.append(r.a, frame);
        }
        return t;
    }

    /// Both prompts, then every frame but the last as a decodePair (what `draw` did, its draws fixed): logits, states.
    fn pairFixed(r: *Rig, cond: []const Frame, uncond: []const u32, frames: []const Frame) !Fixed {
        var f: Fixed = .{};
        try r.backbone.reset(r.stream);
        try r.other.reset(r.stream);
        try r.tower.reset(r.stream, uncond, seed);
        try (try r.backbone.runOn(r.other, r.stream, .prefill, uncond, null)).semantic_logits.download(0, std.mem.sliceAsBytes(try f.uncond.addManyAsSlice(r.a, cols)));
        var rows = try r.prompt(cond, try r.a.alloc(u32, cond.len));
        try rows.semantic_logits.download(0, std.mem.sliceAsBytes(try f.cond.addManyAsSlice(r.a, cols)));
        for (frames[0 .. frames.len - 1]) |frame| {
            rows = try r.backbone.decodePair(r.other, r.stream, frame[0], try r.tower.setInput(r.stream, @as(*const [1]Frame, &frame)), r.guided);
            try rows.semantic_logits.download(0, std.mem.sliceAsBytes(try f.cond.addManyAsSlice(r.a, cols)));
            try r.guided.download(0, std.mem.sliceAsBytes(try f.uncond.addManyAsSlice(r.a, cols)));
        }
        f.cond_state = try digest(r.backbone.seq.state_buf);
        f.uncond_state = try digest(r.other.state_buf);
        return f;
    }

    /// sha256 of a device buffer: a sequence's DeltaNet conv and recurrent states and row caches.
    fn digest(buffer: cuda.DeviceBuffer) ![32]u8 {
        const bytes = try std.heap.page_allocator.alloc(u8, buffer.len);
        defer std.heap.page_allocator.free(bytes);
        try buffer.download(0, bytes);
        var out: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
        return out;
    }

    /// One sequence alone on the Model's own state, its prompt then each drawn frame but the last: logits, hidden rows.
    fn alone(r: *Rig, cond: ?[]const Frame, uncond: []const u32, frames: []const Frame, hidden: ?*std.ArrayList(u16)) !std.ArrayList(f32) {
        var logits: std.ArrayList(f32) = .empty;
        try r.backbone.reset(r.stream);
        try r.tower.reset(r.stream, uncond, seed);
        var rows = if (cond) |c| try r.prompt(c, try r.a.alloc(u32, c.len)) else try r.backbone.run(r.stream, .prefill, uncond, null);
        for (0..frames.len) |i| {
            if (i > 0) rows = try r.backbone.run(r.stream, .decode, frames[i - 1][0..1], try r.tower.setInput(r.stream, frames[i - 1 .. i]));
            try rows.semantic_logits.download(0, std.mem.sliceAsBytes(try logits.addManyAsSlice(r.a, cols)));
            if (hidden) |h| try lastRow(rows).download(0, std.mem.sliceAsBytes(try h.addManyAsSlice(r.a, H)));
        }
        return logits;
    }
};

fn same(what: []const u8, got: []const u8, want: []const u8, failures: *usize) void {
    if (std.mem.eql(u8, got, want)) return std.debug.print("PASS {s}: {d} bytes equal\n", .{ what, got.len });
    const at = if (got.len == want.len) std.mem.indexOfDiff(u8, got, want).? else @min(got.len, want.len);
    std.debug.print("FAIL {s}: {d} against {d} bytes, first difference at byte {d}\n", .{ what, got.len, want.len, at });
    failures.* += 1;
}

fn fixtureRows(fixture: std.json.Value, name: []const u8, which: enum { cond, uncond }, a: std.mem.Allocator) ![]Frame {
    for (fixture.object.get("rows").?.array.items) |case| {
        if (!std.mem.eql(u8, case.object.get("name").?.string, name)) continue;
        const list = if (which == .cond) case.object.get("rows").? else case.object.get("unconditioned").?.object.get("rows").?;
        const out = try a.alloc(Frame, list.array.items.len);
        for (out, list.array.items) |*f, row| for (f, row.array.items) |*v, x| {
            v.* = @intCast(x.integer);
        };
        return out;
    }
    return error.NoSuchFixtureCase;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) {
        std.debug.print("usage: tf-drama-cfg-pair CHECKPOINT FIXTURE_JSON CASE STEPS   (STEPS at least {d} observations, STEPS - 1 paired decodes)\n", .{min_steps});
        return 2;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 26));
    const fixture = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const cond = try fixtureRows(fixture, args[3], .cond, a);
    const plain = try fixtureRows(fixture, args[3], .uncond, a);
    const uncond = try a.alloc(u32, plain.len);
    for (plain, uncond) |row, *t| t.* = row[0];
    const steps = try std.fmt.parseInt(usize, args[4], 10);
    if (cond.len > max_rows or uncond.len > max_rows or steps < min_steps or cond.len + steps > context) return error.CaseOutOfRange;

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var backbone = try drama.backbone.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = context, .max_rows = max_rows });
    defer backbone.deinit();
    try backbone.loadPair();
    var tower = try drama.tower.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = context, .max_rows = max_rows });
    defer tower.deinit();
    var other = try backbone.sequence();
    defer other.free();
    var guided = try cuda.DeviceBuffer.alloc(&driver, cols * 4);
    defer guided.free();
    var rig: Rig = .{ .a = a, .stream = stream, .backbone = &backbone, .tower = &tower, .other = &other, .guided = guided };

    const now = std.Io.Clock.awake;
    var began = now.now(init.io).toNanoseconds();
    const unbatched = try rig.draw(cond, uncond, false, steps);
    const unbatched_s = @as(f64, @floatFromInt(now.now(init.io).toNanoseconds() - began)) / 1e9;
    began = now.now(init.io).toNanoseconds();
    const batched = try rig.draw(cond, uncond, true, steps);
    const batched_s = @as(f64, @floatFromInt(now.now(init.io).toNanoseconds() - began)) / 1e9;
    const again = try rig.draw(cond, uncond, true, steps);
    var hidden: std.ArrayList(u16) = .empty;
    const cond_alone = try rig.alone(cond, uncond, unbatched.frames.items, &hidden);
    const uncond_alone = try rig.alone(null, uncond, unbatched.frames.items, null);
    // the draws fixed, one row's prompt shortened by a token: the other row's logits and state must not move
    const fixed = try rig.pairFixed(cond, uncond, unbatched.frames.items);
    const other_moved = try rig.pairFixed(cond, uncond[0 .. uncond.len - 1], unbatched.frames.items);
    const own_moved = try rig.pairFixed(cond[1..], uncond, unbatched.frames.items);

    var failures: usize = 0;
    const bytes = std.mem.sliceAsBytes;
    same("batched frames = unbatched", bytes(batched.frames.items), bytes(unbatched.frames.items), &failures);
    same("batched l_c = unbatched", bytes(batched.cond.items), bytes(unbatched.cond.items), &failures);
    same("batched l_u = unbatched", bytes(batched.uncond.items), bytes(unbatched.uncond.items), &failures);
    same("batched hidden = unbatched", bytes(batched.hidden.items), bytes(unbatched.hidden.items), &failures);
    same("batched l_c again after reset = batched", bytes(again.cond.items), bytes(batched.cond.items), &failures);
    same("batched l_u again after reset = batched", bytes(again.uncond.items), bytes(batched.uncond.items), &failures);
    same("conditioned alone l_c = guided", bytes(cond_alone.items), bytes(unbatched.cond.items), &failures);
    same("conditioned alone hidden = guided", bytes(hidden.items), bytes(unbatched.hidden.items), &failures);
    same("unconditioned alone l_u = guided", bytes(uncond_alone.items), bytes(unbatched.uncond.items), &failures);
    same("pair over the fixed draws l_c = unbatched", bytes(fixed.cond.items), bytes(unbatched.cond.items), &failures);
    same("pair over the fixed draws l_u = unbatched", bytes(fixed.uncond.items), bytes(unbatched.uncond.items), &failures);
    same("plain prompt shortened: l_c unchanged", bytes(other_moved.cond.items), bytes(fixed.cond.items), &failures);
    same("plain prompt shortened: the conditioned state unchanged", &other_moved.cond_state, &fixed.cond_state, &failures);
    same("referenced prompt shortened: l_u unchanged", bytes(own_moved.uncond.items), bytes(fixed.uncond.items), &failures);
    same("referenced prompt shortened: the unconditioned state unchanged", &own_moved.uncond_state, &fixed.uncond_state, &failures);
    for ([_][2][]const u8{ .{ bytes(other_moved.uncond.items), bytes(fixed.uncond.items) }, .{ bytes(own_moved.cond.items), bytes(fixed.cond.items) } }, 0..) |pair, i| {
        if (!std.mem.eql(u8, pair[0], pair[1])) continue;
        std.debug.print("FAIL shortening the {s} prompt left its own row's logits unchanged: the perturbation did nothing\n", .{if (i == 0) "plain" else "referenced"});
        failures += 1;
    }
    if (unbatched.frames.items.len != steps) {
        std.debug.print("FAIL {d} frames decoded, {d} asked\n", .{ unbatched.frames.items.len, steps });
        failures += 1;
    }
    const n: f64 = @floatFromInt(unbatched.frames.items.len);
    std.debug.print("RESULT case={s} prompt={d} plain={d} observations={d} paired_decodes={d} unbatched_ms_per_frame={d:.2} batched_ms_per_frame={d:.2} failures={d}\n", .{
        args[3], cond.len, uncond.len, unbatched.frames.items.len, unbatched.frames.items.len - 1, 1000 * unbatched_s / n, 1000 * batched_s / n, failures,
    });
    return if (failures == 0) 0 else 1;
}
