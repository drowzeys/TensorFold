//! The speech flags and their values: voices, long-form, silence and register redraws, frame limit, onset guard.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const drama_allow = @import("drama_allow.zig");
const drama_cfg = @import("drama_cfg.zig");
const Allocator = std.mem.Allocator;

/// The speech flags, the tail of ``cli.flags``.
pub const flags = [_]cli.Flag{
    // the native server's own: Drama 3 voice files (tools/zig/drama_voice.py), and the one ``default`` names
    .{ .name = "--voices", .native = true },
    .{ .name = "--default-voice", .native = true },
    // the native server's own: long speech inputs spoken in segments that hear the ones before them
    .{ .name = "--long-form", .kind = .store_true, .native = true },
    .{ .name = "--long-form-chars", .native = true },
    .{ .name = "--long-form-history", .native = true },
    .{ .name = "--shift-pause-s", .native = true },
    // the native server's own: a whole speech reply the silence guard ended is drawn again while its deadline allows
    .{ .name = "--silence-redraws", .native = true },
    .{ .name = "--silence-redraw-budget-s", .native = true },
    .{ .name = "--silence-redraw-plain", .kind = .store_true, .native = true },
    .{ .name = "--silence-redraw-plain-scope", .choices = &.{ "cues", "controls" }, .native = true },
    .{ .name = "--short-reply-ratio", .native = true },
    // the native server's own: a whole speech reply with a phrase opening far above the rest is drawn once more
    .{ .name = "--register-redraw-st", .native = true },
    .{ .name = "--register-redraw-high", .native = true },
    .{ .name = "--register-high-hz", .native = true },
    // the native server's own: the delivery controls a speech input may keep, every other one taken out first
    .{ .name = "--allowed-controls", .native = true },
    // the CUDA build's own: the frames one Drama 3 synthesis may generate, and the silences that end one
    .{ .name = "--max-frames", .native = cli.cuda_build },
    .{ .name = "--silence-onset-s", .native = cli.cuda_build },
    .{ .name = "--silence-mid-s", .native = cli.cuda_build },
    .{ .name = "--cfg-gamma", .native = cli.cuda_build },
    .{ .name = "--cfg-every", .native = cli.cuda_build },
    .{ .name = "--cfg-allow-request-gamma", .kind = .store_true, .native = cli.cuda_build },
    .{ .name = "--cfg-force-guided", .kind = .store_true, .native = cli.cuda_build },
    .{ .name = "--cfg-batched", .kind = .store_true, .native = cli.cuda_build },
};

/// --cfg-gamma (0: off), --cfg-every (null: not given), and the store-true --cfg-* flags as given.
pub const Guidance = struct { gamma: f32 = 0, every: ?u32 = null, request_gamma: bool = false, force_guided: bool = false, batched: bool = false };

/// Sets the speech flag `name` from `v`; false for any other flag.
pub fn flag(a: Allocator, out: *cli.Args, name: []const u8, v: []const u8, u: *cli.Usage) error{ Usage, OutOfMemory }!bool {
    const is = std.mem.eql;
    if (is(u8, name, "--voices")) out.voices = v else if (is(u8, name, "--default-voice")) out.default_voice = v else if (is(u8, name, "--long-form")) out.long_form = true else if (is(u8, name, "--long-form-chars")) {
        const n = try cli.int(u, a, name, v);
        if (n < 1 or n > 4096) return cli.fail(u, a, "argument --long-form-chars: characters a segment holds, from 1 to 4096: '{s}'", .{v});
        out.long_form_chars = @intCast(n);
        out.long_form_set = true;
    } else if (is(u8, name, "--long-form-history")) {
        const n = try cli.int(u, a, name, v);
        if (n < 0 or n > 3) return cli.fail(u, a, "argument --long-form-history: earlier segments a segment hears, from 0 to 3: '{s}'", .{v});
        out.long_form_history = @intCast(n);
        out.long_form_set = true;
    } else if (is(u8, name, "--shift-pause-s")) {
        const s = try cli.float(u, a, name, v);
        if (!(s >= 0 and s <= 3)) return cli.fail(u, a, "argument --shift-pause-s: seconds of quiet at a topic shift, from 0 (off) to 3: '{s}'", .{v});
        out.shift_pause_s = @floatCast(s);
        out.long_form_set = true;
    } else if (is(u8, name, "--silence-redraws")) {
        const n = try cli.int(u, a, name, v);
        if (n < 0 or n > 8) return cli.fail(u, a, "argument --silence-redraws: redraws a reply may take, from 0 to 8: '{s}'", .{v});
        out.silence_redraw.max = @intCast(n);
    } else if (is(u8, name, "--silence-redraw-budget-s")) {
        const s = try cli.float(u, a, name, v);
        if (!(s > 0 and s <= 3600)) return cli.fail(u, a, "argument --silence-redraw-budget-s: seconds from receipt, above 0 to 3600: '{s}'", .{v});
        out.silence_redraw.budget_s = s;
        out.silence_redraw_set = true;
    } else if (is(u8, name, "--silence-redraw-plain")) {
        out.silence_redraw.plain = true;
        out.silence_redraw_set = true;
    } else if (is(u8, name, "--silence-redraw-plain-scope")) {
        out.silence_redraw.plain_scope = std.meta.stringToEnum(@TypeOf(out.silence_redraw.plain_scope), v) orelse unreachable; // parse checked the flag's choices
        out.silence_redraw_set = true;
    } else if (is(u8, name, "--short-reply-ratio")) {
        const r = try cli.float(u, a, name, v);
        if (!(r >= 0 and r < 1)) return cli.fail(u, a, "argument --short-reply-ratio: a share of the expected length, from 0 (off) to below 1: '{s}'", .{v});
        out.silence_redraw.short_ratio = r;
        out.silence_redraw_set = true;
    } else if (is(u8, name, "--register-redraw-st")) {
        const s = try cli.float(u, a, name, v);
        if (!(s >= 0 and s <= 24)) return cli.fail(u, a, "argument --register-redraw-st: semitones a phrase opens above the others, from 0 (off) to 24: '{s}'", .{v});
        out.register.st = s;
    } else if (is(u8, name, "--register-redraw-high")) {
        const p = try cli.float(u, a, name, v);
        if (!(p > 0 and p <= 100)) return cli.fail(u, a, "argument --register-redraw-high: percent of a phrase's voiced frames over the high line, above 0 to 100: '{s}'", .{v});
        out.register.high = p / 100;
        out.register_set = true;
    } else if (is(u8, name, "--register-high-hz")) {
        const h = try cli.float(u, a, name, v);
        if (!(h >= 70 and h <= 400)) return cli.fail(u, a, "argument --register-high-hz: the high line, from 70 to 400 Hz: '{s}'", .{v});
        out.register.high_hz = h;
        out.register_set = true;
    } else if (is(u8, name, "--max-frames")) {
        const n = try cli.int(u, a, name, v);
        if (n < 1 or n > 8191) return cli.fail(u, a, "argument --max-frames: frames one synthesis may generate, from 1 to 8191: '{s}'", .{v});
        out.max_frames = @intCast(n);
    } else if (is(u8, name, "--silence-onset-s")) {
        // past 0, at least 1 s: a normal draw takes up to 0.93 s to voice its first frame
        const s = try cli.float(u, a, name, v);
        if (!(s == 0 or (s >= 1 and s < 10))) return cli.fail(u, a, "argument --silence-onset-s: seconds of silence that end a synthesis voiced for less than 1 s, 0 (off) or 1 to below 10: '{s}'", .{v});
        out.silence_onset_s = s;
    } else if (is(u8, name, "--silence-mid-s")) {
        // past 0, at least 3 s: replayed draws ending in silence held 1.0-2.65 s; every judged one at 3.2 s+ lost words
        const s = try cli.float(u, a, name, v);
        if (!(s == 0 or (s >= 3 and s < 10))) return cli.fail(u, a, "argument --silence-mid-s: seconds of silence end of speech must fall in to end a synthesis as silence, 0 (off) or 3 to below 10: '{s}'", .{v});
        out.silence_mid_s = s;
    } else if (is(u8, name, "--cfg-gamma")) {
        const g = try cli.float(u, a, name, v);
        if (!(g == 0 or drama_cfg.valid(g))) return cli.fail(u, a, "argument --cfg-gamma: the guidance weight on a referenced prompt's semantic logits, 0 (off) or 1 (unguided) to 5: '{s}'", .{v});
        out.cfg.gamma = @floatCast(g);
    } else if (is(u8, name, "--cfg-every")) {
        const n = try cli.int(u, a, name, v);
        // a K > 1 run catches up K frames as one prefill chunk: tested for order, not yet for its numbers
        if (n != 1) return cli.fail(u, a, "argument --cfg-every: frames between the unconditioned sequence's runs, only 1 until a catch-up run of more is validated: '{s}'", .{v});
        out.cfg.every = @intCast(n);
    } else if (is(u8, name, "--cfg-allow-request-gamma")) {
        out.cfg.request_gamma = true;
    } else if (is(u8, name, "--cfg-force-guided")) {
        out.cfg.force_guided = true;
    } else if (is(u8, name, "--cfg-batched")) {
        out.cfg.batched = true;
    } else if (is(u8, name, "--allowed-controls")) {
        if (!std.unicode.utf8ValidateSlice(v)) return cli.fail(u, a, "argument --allowed-controls: not UTF-8 text", .{});
        var names: std.ArrayList([]const u8) = .empty;
        var it = std.mem.splitScalar(u8, v, ',');
        while (it.next()) |part| {
            if (std.mem.indexOfAny(u8, part, "[]") != null) return cli.fail(u, a, "argument --allowed-controls: name a control without its brackets: '{s}'", .{part});
            const control = try drama_allow.normalize(a, part);
            if (control.len > 0) try names.append(a, control);
        }
        out.allowed_controls = names.items;
    } else if (is(u8, name, "--expert-mask")) {
        out.expert_mask = v;
    } else return false;
    return true;
}

/// Guidance as the flags set it: --cfg-gamma's γ (1 when 0 with --cfg-allow-request-gamma); null: off.
pub fn guidance(args: cli.Args) ?drama_cfg.Options {
    const c = args.cfg;
    if (c.gamma == 0 and !c.request_gamma) return null;
    return .{ .gamma = if (c.gamma == 0) 1 else c.gamma, .request_gamma = c.request_gamma, .force_guided = c.force_guided };
}

/// `o` with what the engine reads of guidance: --cfg-every (1 if not given) and --cfg-batched; as it is when off.
pub fn withGuidance(args: cli.Args, o: api.Open) api.Open {
    if (guidance(args) == null) return o;
    var out = o;
    out.cfg_every = args.cfg.every orelse 1;
    out.cfg_batched = args.cfg.batched;
    return out;
}

test "silence redraws: off by default; budget, plain and short replies need a redraw count; values checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    const off = try cli.parse(a, &.{"m"}, &u);
    try std.testing.expect(off.silence_redraw.max == 0 and off.silence_redraw.budget_s == 120 and !off.silence_redraw.plain and off.silence_redraw.plain_scope == .cues and off.silence_redraw.short_ratio == 0);
    const on = try cli.parse(a, &.{ "m", "--silence-redraws", "2", "--silence-redraw-budget-s=96.5", "--silence-redraw-plain", "--silence-redraw-plain-scope", "controls", "--short-reply-ratio", "0.5" }, &u);
    try std.testing.expect(on.silence_redraw.max == 2 and on.silence_redraw.budget_s == 96.5 and on.silence_redraw.plain and on.silence_redraw.plain_scope == .controls and on.silence_redraw.short_ratio == 0.5);
    for ([_][]const []const u8{
        &.{ "m", "--silence-redraw-plain" },
        &.{ "m", "--silence-redraw-budget-s", "60" },
        &.{ "m", "--short-reply-ratio", "0.5" },
        &.{ "m", "--silence-redraws", "0", "--silence-redraw-plain" },
        &.{ "m", "--silence-redraws", "9" },
        &.{ "m", "--silence-redraws", "-1" },
        &.{ "m", "--silence-redraws", "1", "--silence-redraw-budget-s", "0" },
        &.{ "m", "--silence-redraws", "1", "--silence-redraw-budget-s", "nan" },
        &.{ "m", "--silence-redraws", "1", "--short-reply-ratio", "1" },
        &.{ "m", "--silence-redraws", "1", "--short-reply-ratio", "-0.1" },
        &.{ "m", "--silence-redraws", "1", "--short-reply-ratio", "nan" },
        &.{ "m", "--silence-redraws", "1", "--silence-redraw-plain-scope", "controls" },
        &.{ "m", "--silence-redraws", "1", "--silence-redraw-plain", "--silence-redraw-plain-scope", "all" },
        &.{ "m", "--silence-redraw-plain-scope", "cues" },
    }) |bad| try std.testing.expectError(error.Usage, cli.parse(a, bad, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try cli.capabilities(&out.writer, .{ .version = "0.6.5" });
    for ([_][]const u8{ "\"--silence-redraws\": {}", "\"--silence-redraw-budget-s\": {}", "\"--silence-redraw-plain\": {}", "\"--short-reply-ratio\": {}", "\"--silence-redraw-plain-scope\": {\"values\": [\"cues\", \"controls\"]}" }) |key| try std.testing.expect(std.mem.indexOf(u8, out.written(), key) != null);
}

test "register redraws: off by default; the high share and line need a jump; values checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    const off = (try cli.parse(a, &.{"m"}, &u)).register;
    try std.testing.expect(off.st == 0 and off.high == 0.2 and off.high_hz == 210);
    const on = (try cli.parse(a, &.{ "m", "--register-redraw-st", "4", "--register-redraw-high=25", "--register-high-hz", "190" }, &u)).register;
    try std.testing.expect(on.st == 4 and on.high == 0.25 and on.high_hz == 190);
    for ([_][]const []const u8{
        &.{ "m", "--register-redraw-high", "20" },
        &.{ "m", "--register-redraw-st", "0", "--register-high-hz", "210" },
        &.{ "m", "--register-redraw-st", "-1" },
        &.{ "m", "--register-redraw-st", "nan" },
        &.{ "m", "--register-redraw-st", "4", "--register-redraw-high", "0" },
        &.{ "m", "--register-redraw-st", "4", "--register-redraw-high", "101" },
        &.{ "m", "--register-redraw-st", "4", "--register-high-hz", "69" },
        &.{ "m", "--register-redraw-st", "4", "--register-high-hz", "401" },
    }) |bad| try std.testing.expectError(error.Usage, cli.parse(a, bad, &u));
}

test "--shift-pause-s: off by default, 0 to 3 seconds, and only with --long-form" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    try std.testing.expectEqual(@as(f32, 0), (try cli.parse(a, &.{ "m", "--long-form" }, &u)).shift_pause_s);
    try std.testing.expectEqual(@as(f32, 0.85), (try cli.parse(a, &.{ "m", "--long-form", "--shift-pause-s", "0.85" }, &u)).shift_pause_s);
    for ([_][]const []const u8{
        &.{ "m", "--shift-pause-s", "0.85" },
        &.{ "m", "--long-form", "--shift-pause-s", "-0.1" },
        &.{ "m", "--long-form", "--shift-pause-s", "3.5" },
        &.{ "m", "--long-form", "--shift-pause-s", "nan" },
    }) |bad| try std.testing.expectError(error.Usage, cli.parse(a, bad, &u));
}

test "--silence-onset-s: off by default, CUDA builds only, 0 or 1 to below 10 s, and it reaches the engine" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    try std.testing.expectEqual(@as(f64, 0), (try cli.request(a, "d", "t", try cli.parse(a, &.{"m"}, &u))).silence_onset_s);
    if (@import("builtin").os.tag != .linux) return std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--silence-onset-s", "3" }, &u));
    try std.testing.expectEqual(@as(f64, 3), (try cli.request(a, "d", "t", try cli.parse(a, &.{ "m", "--silence-onset-s=3" }, &u))).silence_onset_s);
    try std.testing.expectEqual(@as(f64, 0), (try cli.parse(a, &.{ "m", "--silence-onset-s", "0" }, &u)).silence_onset_s);
    for ([_][]const u8{ "0.5", "-1", "10", "nan", "3s" }) |bad| try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--silence-onset-s", bad }, &u));
}

test "--silence-mid-s: off by default, CUDA builds only, 0 or 3 to below 10 s, and it reaches the engine" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    try std.testing.expectEqual(@as(f64, 0), (try cli.request(a, "d", "t", try cli.parse(a, &.{"m"}, &u))).silence_mid_s);
    if (@import("builtin").os.tag != .linux) return std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--silence-mid-s", "3" }, &u));
    try std.testing.expectEqual(@as(f64, 3.5), (try cli.request(a, "d", "t", try cli.parse(a, &.{ "m", "--silence-mid-s=3.5" }, &u))).silence_mid_s);
    try std.testing.expectEqual(@as(f64, 0), (try cli.parse(a, &.{ "m", "--silence-mid-s", "0" }, &u)).silence_mid_s);
    for ([_][]const u8{ "2.9", "-1", "10", "nan", "3s" }) |bad| try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--silence-mid-s", bad }, &u));
}

test "--allowed-controls: unset by default; a comma list of names normalized as controls are, brackets refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    try std.testing.expect((try cli.parse(a, &.{"m"}, &u)).allowed_controls == null);
    const got = (try cli.parse(a, &.{ "m", "--allowed-controls", "Concerned, curious!,\tsmall   LAUGH ,," }, &u)).allowed_controls.?;
    try std.testing.expectEqual(@as(usize, 3), got.len);
    for ([_][]const u8{ "concerned", "curious", "small laugh" }, got) |want, name| try std.testing.expectEqualStrings(want, name);
    try std.testing.expectEqual(@as(usize, 0), (try cli.parse(a, &.{ "m", "--allowed-controls", "" }, &u)).allowed_controls.?.len);
    for ([_][]const u8{ "sighs,[pause]", "warmly]", "\xff" }) |bad| try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--allowed-controls", bad }, &u));
}

test "guidance: off by default, CUDA builds only; --cfg-gamma 0 or 1 to 5, --cfg-every only 1, it and the diagnostics only with guidance on" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    const off = try cli.parse(a, &.{"m"}, &u);
    try std.testing.expect(guidance(off) == null and (try cli.request(a, "d", "t", off)).cfg_every == null and !(try cli.request(a, "d", "t", off)).cfg_batched);
    if (@import("builtin").os.tag != .linux) return std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--cfg-gamma", "2" }, &u));
    try std.testing.expect(guidance(try cli.parse(a, &.{ "m", "--cfg-gamma", "0" }, &u)) == null);
    const fixed = try cli.parse(a, &.{ "m", "--cfg-gamma=1.5" }, &u);
    try std.testing.expectEqual(drama_cfg.Options{ .gamma = 1.5 }, guidance(fixed).?);
    try std.testing.expectEqual(@as(?u32, 1), (try cli.request(a, "d", "t", fixed)).cfg_every);
    try std.testing.expect(!(try cli.request(a, "d", "t", fixed)).cfg_batched);
    const batched = try cli.request(a, "d", "t", try cli.parse(a, &.{ "m", "--cfg-allow-request-gamma", "--cfg-batched" }, &u));
    try std.testing.expect(batched.cfg_batched and batched.cfg_every.? == 1);
    const open = try cli.parse(a, &.{ "m", "--cfg-allow-request-gamma", "--cfg-every", "1" }, &u);
    try std.testing.expectEqual(drama_cfg.Options{ .gamma = 1, .request_gamma = true }, guidance(open).?);
    try std.testing.expectEqual(@as(?u32, 1), (try cli.request(a, "d", "t", open)).cfg_every);
    try std.testing.expectEqual(drama_cfg.Options{ .gamma = 1, .force_guided = true }, guidance(try cli.parse(a, &.{ "m", "--cfg-gamma", "1", "--cfg-force-guided" }, &u)).?);
    for ([_][]const []const u8{
        &.{ "m", "--cfg-every", "1" },
        &.{ "m", "--cfg-gamma", "0", "--cfg-every", "1" },
        &.{ "m", "--cfg-gamma", "2", "--cfg-every", "0" },
        &.{ "m", "--cfg-gamma", "2", "--cfg-every", "2" },
        &.{ "m", "--cfg-allow-request-gamma", "--cfg-every", "16" },
        &.{ "m", "--cfg-gamma", "0.5" },
        &.{ "m", "--cfg-gamma", "5.5" },
        &.{ "m", "--cfg-gamma", "nan" },
        &.{ "m", "--cfg-allow-request-gamma=1" },
        &.{ "m", "--cfg-force-guided" },
        &.{ "m", "--cfg-gamma", "0", "--cfg-force-guided" },
        &.{ "m", "--cfg-batched" },
        &.{ "m", "--cfg-gamma", "0", "--cfg-batched" },
    }) |bad| try std.testing.expectError(error.Usage, cli.parse(a, bad, &u));
}

test "--expert-mask reaches the engine on CUDA builds; absent, every routed expert is eligible" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    try std.testing.expectEqual(@as(?[]const u8, null), (try cli.request(a, "d", "t", try cli.parse(a, &.{"m"}, &u))).expert_mask);
    if (@import("builtin").os.tag != .linux) return std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--expert-mask", "k.json" }, &u));
    try std.testing.expectEqualStrings("k.json", (try cli.request(a, "d", "t", try cli.parse(a, &.{ "m", "--expert-mask=k.json" }, &u))).expert_mask.?);
}
