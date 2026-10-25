//! A server's life: keys, the listening socket, the startup lines, the live line, and a clean stop on SIGTERM or SIGINT.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const auth = @import("auth.zig");
const cli = @import("cli.zig");
const log = @import("log.zig");
const live = @import("live.zig");
const listener_mod = @import("listener.zig");
const model_text = @import("model_text.zig");
const server_mod = @import("server.zig");
const drama_voice = @import("drama_voice.zig");
const Allocator = std.mem.Allocator;
const posix = std.posix;

var stop_requested: std.atomic.Value(bool) = .init(false);

fn onStop(_: posix.SIG) callconv(.c) void {
    stop_requested.store(true, .release);
}

/// SIGTERM and SIGINT stop the server cleanly; SIGHUP rereads the key file; SIGPIPE is ignored, as Python ignores it.
pub fn installSignals(keys: bool) void {
    const stop: posix.Sigaction = .{ .handler = .{ .handler = onStop }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.TERM, &stop, null);
    posix.sigaction(.INT, &stop, null);
    const ignore: posix.Sigaction = .{ .handler = .{ .handler = posix.SIG.IGN }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.PIPE, &ignore, null);
    if (keys) auth.installHangup();
}

pub fn stopping() bool {
    return stop_requested.load(.acquire);
}

/// The hard limit macOS reads as unlimited.
const rlim_infinity: u64 = std.c.RLIM.INFINITY;

/// The targets to try, in order: the hard limit when finite, else the fallbacks 65536 then 10240, first that works; never an unlimited soft limit, since a loop over every descriptor would do absurd work.
fn fillTargets(max: u64, out: *[2]u64) []const u64 {
    if (max != rlim_infinity) {
        out[0] = max;
        return out[0..1];
    }
    out[0] = 65536;
    out[1] = 10240;
    return out;
}

/// Best-effort raise of the soft open-file limit so idle keep-alives cannot exhaust a default 1024; a failure keeps the current limits, and null (caller logs) means nothing moved.
fn raiseOpenFileLimit() ?[2]u64 {
    const limit = posix.getrlimit(.NOFILE) catch return null;
    const cur: u64 = @intCast(limit.cur);
    if (cur == rlim_infinity) return null; // already unlimited
    var buf: [2]u64 = undefined;
    for (fillTargets(@intCast(limit.max), &buf)) |target| {
        if (target <= cur) continue;
        posix.setrlimit(.NOFILE, .{ .cur = @intCast(target), .max = limit.max }) catch continue;
        return .{ cur, target };
    }
    return null;
}

/// What a server needs besides its flags.
pub const Setup = struct {
    engine: api.Engine,
    text: model_text.Text,
    served: []const u8,
    /// The checkpoint's folder (structured output reads its tokenizer.json and config.json, /slide shows its shards).
    model_dir: []const u8 = "",
    /// Sampling defaults from generation_config.json and the flags, as a JSON object; null: greedy.
    sampling: ?json.Value = null,
    environ: ?*const std.process.Environ.Map = null,
    started: i96 = 0,
    /// Called once the socket listens, with its port (tests read it).
    on_listen: ?*const fn (port: u16) void = null,
};

fn env(s: Setup, name: []const u8) ?[]const u8 {
    const m = s.environ orelse return null;
    return m.get(name);
}

/// Serves until a stop signal; the process exit status.
pub fn run(gpa: Allocator, io: std.Io, args: cli.Args, s: Setup) u8 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var store = auth.Store.init(gpa, args.api_key, env(s, "TENSORFOLD_API_KEY") orelse "", args.api_key_file, args.metrics_open, &problem) catch {
        std.debug.print("tensorfold: {s}\n", .{if (problem.len > 0) problem else "API key file cannot be opened safely"});
        return 1;
    };
    defer store.deinit();
    if (!store.enabled() and !auth.loopback(args.host)) log.line("warning: non-loopback server has no API key; requests are open", .{});
    var ids: std.ArrayList([]const u8) = .empty;
    for ([_][]const []const u8{ &.{s.served}, args.alias }) |group| for (group) |raw| {
        const id = std.mem.trim(u8, raw, " \t\r\n");
        if (id.len == 0) continue;
        for (ids.items) |seen| {
            if (std.mem.eql(u8, seen, id)) break;
        } else ids.append(a, id) catch return 1;
    };
    const salt = if (env(s, "TENSORFOLD_SEED_SALT")) |t| std.fmt.parseInt(i64, std.mem.trim(u8, t, " "), 10) catch {
        std.debug.print("tensorfold: TENSORFOLD_SEED_SALT={s}: an integer\n", .{t});
        return 1;
    } else 0;
    if ((args.voices != null or args.long_form) and s.engine.info().speech == null) {
        std.debug.print("tensorfold: {s} needs a speech model\n", .{if (args.long_form) "--long-form" else "--voices"});
        return 2;
    }
    const voices: []const drama_voice.Voice = if (args.voices) |dir| (drama_voice.load(a, io, dir, &problem) catch return 1) orelse {
        std.debug.print("tensorfold: {s}\n", .{problem});
        return 1;
    } else &.{};
    if (s.engine.info().speech) |sp| {
        if (drama_voice.overflow(a, s.text, voices, sp.max_prompt_tokens) catch return 1) |why| {
            std.debug.print("tensorfold: --voices {s}: {s}\n", .{ args.voices.?, why });
            return 2;
        }
    }
    const default_voice = if (args.default_voice) |name| drama_voice.find(voices, name) orelse {
        std.debug.print("tensorfold: --default-voice {s}: --voices holds no {s}.json\n", .{ name, name });
        return 2;
    } else null;
    const config: server_mod.Config = .{
        .served_name = s.served,
        .model_dir = s.model_dir,
        .model_ids = ids.items,
        .default_max_tokens = args.max_tokens,
        .enable_thinking = args.thinking,
        .reasoning_effort = args.reasoning_effort,
        .thinking_budget = args.thinking_budget,
        .loop_guard = args.loop_guard,
        .keep_warm_s = args.keep_warm,
        .default_sampling = s.sampling,
        .use_drafts = !args.no_drafts,
        .seed_salt = salt,
        .request_log = env(s, "TENSORFOLD_REQUEST_LOG"),
        .dashboard = args.dashboard,
        .compact_at = if (args.compact_auto) .{ .auto = {} } else if (args.compact_fraction) |f| .{ .fraction = f } else null,
        .compact_keep = args.compact_keep,
        .compact_memory = args.compact_memory,
        .slide_graph = args.slide_graph,
        .cortheon = args.cortheon,
        .cortheon_search_key = env(s, "BRAVE_SEARCH_API_KEY"),
        .voices = voices,
        .default_voice = default_voice,
        .long_form = if (args.long_form) .{ .chars = args.long_form_chars, .history = args.long_form_history, .shift_pause_s = args.shift_pause_s } else null,
        .silence_redraw = args.silence_redraw,
        .register = args.register,
        .silence_mid = args.silence_mid_s > 0,
        .allowed_controls = args.allowed_controls,
        .cfg = @import("cli_speech.zig").guidance(args),
    };
    const srv = server_mod.Server.init(gpa, io, s.engine, s.text, config, if (store.enabled()) &store else null) catch {
        std.debug.print("tensorfold: the server could not start\n", .{});
        return 1;
    };
    defer srv.deinit();
    if (args.loop_guard and !srv.info.loop_guard) {
        std.debug.print("tensorfold: --loop-guard is not supported by this native engine\n", .{});
        return 2;
    }
    if (raiseOpenFileLimit()) |r| log.line("open-file limit raised: soft {d} -> {d}", .{ r[0], r[1] });
    const address = resolve(io, args.host, args.port) orelse {
        std.debug.print("tensorfold: cannot resolve --host {s}\n", .{args.host});
        return 1;
    };
    const lis = listener_mod.Listener.open(address) catch |e| {
        std.debug.print("tensorfold: cannot listen on {s}:{d} ({s})\n", .{ args.host, args.port, @errorName(e) });
        return 1;
    };
    installSignals(store.enabled());
    const port = lis.port();
    if (s.on_listen) |f| f(port);
    const window = srv.info.context_window;
    if (srv.info.startup.len > 0) log.line("{s}", .{srv.info.startup});
    const now = std.Io.Clock.awake.now(io).toNanoseconds();
    const loaded = if (s.started > 0) @as(f64, @floatFromInt(now - s.started)) / 1e9 else 0;
    var window_text: [24]u8 = undefined;
    log.line("serving {s} at http://{s}:{d}/v1 (sampling: {s}; drafts: {s}; context: {s}; loaded in {d:.1}s)", .{
        s.served,                                                   args.host,                                                                                   port,   shownSampling(a, s.sampling),
        if (args.no_drafts or srv.info.plain_only) "off" else "on", if (window > 0) std.fmt.bufPrint(&window_text, "{d}", .{window}) catch "?" else "unlimited", loaded,
    });
    if (args.thinking and std.mem.indexOf(u8, s.text.templateSource(), "enable_thinking") != null)
        log.line("thinking on (the chat template's default): replies reason in reasoning_content before the answer in content, and max_tokens counts both. --no-thinking turns it off; a request can send chat_template_kwargs {{\"enable_thinking\": false}}", .{});
    if (srv.cortheon) |c| log.line("cortheon on: chat requests get Cortheon's reasoning program, web research before building, and cortheon_search/cortheon_fetch run here ({s}); a request can send \"cortheon\": false", .{c.engine});
    if (args.voices) |dir| log.line("voices: {d} from {s}; default voice: {s}", .{ voices.len, dir, if (default_voice) |v| v.name else "built-in" });
    if (args.long_form) log.line("long-form speech: segments of up to {d} characters, each hearing {d} before it; at most 12 segments and 6000 frames a reply", .{ args.long_form_chars, args.long_form_history });
    if (args.shift_pause_s > 0) log.line("shift pauses: a \u{b6} after a sentence end cuts a segment there, and the join gets at least {d:.2}s of quiet, its span sent as X-Pause-Spans", .{args.shift_pause_s});
    if (args.allowed_controls) |list| log.line("allowed controls: {d} ({s}); every other [...] delivery control (drama_cues.txt) and scoped tag is taken out of a speech input first, the count sent as X-Controls-Stripped", .{ list.len, std.mem.join(a, ", ", list) catch "" });
    if (args.silence_redraw.max > 0) log.line("silence redraws: up to {d} a whole reply, each started only if it should end within {d:.1}s of receipt (or X-Deadline-S if sooner); redraws speak the input {s}{s}; a stop under {d:.2} of the input's expected length is redrawn too (0: off)", .{ args.silence_redraw.max, args.silence_redraw.budget_s, if (args.silence_redraw.plain) "without its " else "as sent", if (args.silence_redraw.plain) @tagName(args.silence_redraw.plain_scope) else "", args.silence_redraw.short_ratio });
    if (args.register.st > 0) log.line("register redraws: a whole reply is drawn once more if a phrase after the first opens {d:.2} st or more above the median of its other phrase openings with {d:.0}% or more of its voiced frames over {d:.0} Hz, started only if it should end within {d:.1}s of receipt (or X-Deadline-S if sooner); the draw with the smaller jump is sent", .{ args.register.st, args.register.high * 100, args.register.high_hz, args.silence_redraw.budget_s });
    if (cli.cuda_build and s.engine.info().speech != null) log.line("drama3: silence guard on: a reply ends after {d} s below {d} dBFS{s}", .{ args.silence_seconds orelse 10, args.silence_floor_dbfs orelse -40, if (args.silence_onset_s > 0) std.fmt.allocPrint(a, ", or after {d} s below it with under 1 s voiced", .{args.silence_onset_s}) catch "" else "" });
    if (args.silence_mid_s > 0) log.line("silence mid trips: end of speech inside {d:.2}s or more of silence after {d} s voiced ends a synthesis as silence, redrawn as a silence is; whole replies say how often in X-Silence-Mid-Trips", .{ args.silence_mid_s, args.silence_mid_min_voiced_s orelse 1 });
    if (config.cfg) |g| log.line("guidance: a referenced speech prompt's main token is drawn from l_c + (gamma - 1)(l_c - l_u), l_u from the same turn as a plain prompt, gamma {d}{s}{s}, the unconditioned sequence run every {d} frames{s}; each reply says its gamma in X-Cfg-Gamma", .{ g.gamma, if (g.request_gamma) " unless the request's X-Cfg-Gamma sets it (1 to 5)" else "", if (g.force_guided) ", gamma 1 guided too (--cfg-force-guided, a diagnostic: it must draw as unguided)" else "", args.cfg.every orelse 1, if (args.cfg.batched) " in one batch with the referenced one (--cfg-batched)" else "" });
    live.columns_env = env(s, "COLUMNS");
    var ticker: live.Ticker = .{ .engine = s.engine };
    const drawing = live.wanted(io, if (env(s, "TENSORFOLD_NO_LIVE")) |v| std.mem.eql(u8, v, "1") else false);
    if (drawing) ticker.start();
    const accept = std.Thread.spawn(.{}, server_mod.Server.serve, .{ srv, lis, &stop_requested }) catch return 1;
    accept.join();
    lis.close();
    if (drawing) ticker.finish();
    var waited: u32 = 0;
    while (srv.open_connections.load(.acquire) > 0 and waited < 40) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    if (srv.open_connections.load(.acquire) > 0) std.process.exit(0); // replies still open: end without freeing what they read
    return 0;
}

/// ``--host`` as an address: an IP literal, else a name the resolver knows (localhost first).
fn resolve(io: std.Io, host: []const u8, port: u16) ?std.Io.net.IpAddress {
    if (std.Io.net.IpAddress.parse(host, port)) |ip| return ip else |_| {}
    if (std.ascii.eqlIgnoreCase(host, "localhost")) return .{ .ip4 = .loopback(port) };
    return std.Io.net.IpAddress.resolve(io, host, port) catch null;
}

/// The serving line's sampling: greedy, or each default as ``name value``.
fn shownSampling(a: Allocator, sampling: ?json.Value) []const u8 {
    const v = sampling orelse return "greedy";
    if (v != .object) return "greedy";
    const t = v.get("temperature") orelse return "greedy";
    const temp: f64 = switch (t) {
        .float => |f| f,
        .int => |i| std.fmt.parseFloat(f64, i) catch 0,
        else => 0,
    };
    if (temp <= 0) return "greedy";
    var parts: std.ArrayList([]const u8) = .empty;
    for (v.object.keys(), v.object.values()) |k, x| {
        var buf: [40]u8 = undefined;
        const text = switch (x) {
            .float => |f| json.floatRepr(&buf, f),
            .int => |i| i,
            else => "?",
        };
        parts.append(a, std.fmt.allocPrint(a, "{s} {s}", .{ k, text }) catch return "greedy") catch return "greedy";
    }
    return std.mem.join(a, ", ", parts.items) catch "greedy";
}

test "the targets are the hard limit, or the fallbacks under an unlimited one" {
    var buf: [2]u64 = undefined;
    try std.testing.expectEqualSlices(u64, &.{524288}, fillTargets(524288, &buf));
    try std.testing.expectEqualSlices(u64, &.{ 65536, 10240 }, fillTargets(rlim_infinity, &buf));
}

test "the raise lifts a low soft limit, and a second raise does nothing" {
    const before = try posix.getrlimit(.NOFILE);
    defer posix.setrlimit(.NOFILE, before) catch {};
    if (@as(u64, @intCast(before.max)) <= 256) return error.SkipZigTest;
    try posix.setrlimit(.NOFILE, .{ .cur = 256, .max = before.max });
    const raised = raiseOpenFileLimit() orelse return error.TestExpectedRaise;
    try std.testing.expectEqual(@as(u64, 256), raised[0]);
    try std.testing.expectEqual(raised[1], @as(u64, @intCast((try posix.getrlimit(.NOFILE)).cur)));
    try std.testing.expect(raiseOpenFileLimit() == null);
}
