//! POST /v1/audio/speech: OpenAI's speech request, sent as WAV or raw PCM whole, or with ``stream`` as raw PCM chunks.
//! Only a complete synthesis is audio: a failure is an error reply, or after a stream's first chunk no terminator.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const openai = @import("openai.zig");
const pyrepr = @import("pyrepr.zig");
const reply_text = @import("reply_text.zig");
const drama_prompt = @import("drama_prompt.zig");
const drama_voice = @import("drama_voice.zig");
const drama_shift = @import("drama_shift.zig");
const drama_allow = @import("drama_allow.zig");
const drama_cfg = @import("drama_cfg.zig");
const model_text = @import("model_text.zig");
const speech_long = @import("speech_long.zig");
const speech_redraw = @import("speech_redraw.zig");
const speech_register = @import("speech_register.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// ``MAX_SPEECH_INPUT_CHARS``: the most characters (code points) one input may hold.
pub const max_input_chars = 4096;

/// The request fields served: the OpenAI ones (``speaker`` is ``voice``'s alias); ``stream*`` are checked apart.
const served_fields = [_][]const u8{ "model", "input", "voice", "speaker", "response_format" };

/// The rest of the speech protocol's fields: synthesis modifiers and audio references this route does not honor.
const modifier_fields = [_][]const u8{ "speed", "task_type", "language", "instructions", "ref_audio", "ref_text", "x_vector_only_mode", "stream_codec_output", "suppress_bootstrap_silence", "token_count", "duration_tokens", "initial_codec_chunk_frames", "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed", "stage_params" };

pub const Format = enum {
    wav,
    pcm,

    fn contentType(f: Format) []const u8 {
        return switch (f) {
            .wav => "audio/wav",
            .pcm => "audio/pcm",
        };
    }
};

/// A synthesis's prompt and sources, or a long-form reply; `dropped`: `¶` marks not honoured; `stripped`: controls cut.
const Plan = struct { prompt: []const u32, codes: []const drama_prompt.Codes, input: []const u8, references: []const drama_prompt.Reference, format: Format, stream: bool, long: ?speech_long.Long = null, dropped: usize = 0, stripped: ?usize = null };

pub fn post(srv: *Server, c: *Conn, a: Allocator) !void {
    const sp = srv.info.speech orelse {
        _ = try http_body.read(c, a, http_body.limit); // dropped, so it cannot reach the next request on this connection
        return routes.unknown(c, a);
    };
    var cx: Cx = .{ .a = a };
    const id = try ids.make(a, "speech-", 32);
    const received = clock.nowNs(srv.io);
    var redraws: speech_redraw.Redraws = .init(srv.config.silence_redraw, received);
    if (srv.config.silence_mid) redraws.hollow = 0;
    const raw = http_body.readJson(c, &cx) catch |e| return refuse(c, &cx, id, &redraws, e);
    const plan = prepare(srv, &cx, sp, raw) catch |e| return refuse(c, &cx, id, &redraws, e);
    // only chunked framing tells a client that a stream which ends early is incomplete
    if (plan.stream and !std.mem.eql(u8, c.version, "HTTP/1.1")) return fail(c, &cx, id, &redraws, .request, "stream=true requires HTTP/1.1");
    // read only when a redraw is on: with them off, a request is served exactly as before
    if (redraws.policy.max > 0 or srv.config.register.st > 0) if (speech_redraw.deadline(c.header("X-Deadline-S")) catch
        return fail(c, &cx, id, &redraws, .request, "X-Deadline-S must be a positive number of seconds")) |seconds| redraws.limit(seconds);
    // read only with guidance on: with it off, a request is served exactly as before
    const gamma: ?f32 = if (srv.config.cfg) |options| drama_cfg.gamma(options, c.header("X-Cfg-Gamma")) orelse
        return fail(c, &cx, id, &redraws, .request, "X-Cfg-Gamma must be a number from 1 (unguided) to 5") else null;
    if (plan.dropped > 0) log.line("speech {s} dropped {d} \u{b6} marks that follow no sentence end or precede no words", .{ id, plan.dropped });
    if (plan.stripped) |n| log.line("speech {s} took out {d} delivery controls off --allowed-controls", .{ id, n });
    if (plan.long) |long| return speech_long.synthesize(srv, c, &cx, id, sp, long, plan.format, plan.stream, received, &redraws, plan.stripped, gamma);
    synthesize(srv, c, &cx, id, sp, plan, gamma, received, &redraws);
}

fn listed(names: []const []const u8, key: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, key)) return true;
    return false;
}

/// Refuses what this route would drop: unknown fields, set modifiers, non-strings, a bad ``stream``/``stream_format``.
fn checkFields(cx: *Cx, body: Value) errors.Refused!void {
    for (body.object.keys(), body.object.values()) |key, value| {
        if (listed(&served_fields, key)) {
            if (value != .string) return cx.fail(.request, "{s} must be a string", .{key});
        } else if (std.mem.eql(u8, key, "stream")) {
            if (value != .bool and value != .null) return cx.refuse("stream must be a boolean");
        } else if (std.mem.eql(u8, key, "stream_format")) {
            if (value == .null or (value == .string and std.mem.eql(u8, value.string, "audio"))) continue;
            if (value == .string and std.mem.eql(u8, value.string, "sse")) return cx.refuse("stream_format sse is not supported by this model");
            return cx.refuse("stream_format must be one of: audio, sse");
        } else if (std.mem.eql(u8, key, "references")) {
            // lowered with the voice, after the input's checks (drama_voice.lower)
        } else if (listed(&modifier_fields, key)) {
            if (value != .null) return cx.fail(.request, "{s} is not supported by this model", .{key});
        } else return cx.fail(.request, "unknown speech request field: {s}", .{key});
    }
}

/// ``response_format.strip().lower()``, which must name WAV or raw PCM.
fn responseFormat(cx: *Cx, value: []const u8) errors.Refused!Format {
    const name = reply_text.pyStrip(value);
    if (std.ascii.eqlIgnoreCase(name, "wav")) return .wav;
    if (std.ascii.eqlIgnoreCase(name, "pcm")) return .pcm;
    return cx.refuse("response_format must be one of: pcm, wav");
}

/// Every check that can refuse the request, then its prompt ids, before anything reaches the engine.
fn prepare(srv: *Server, cx: *Cx, sp: api.SpeechInfo, raw: Value) errors.Refused!Plan {
    if (raw != .object) return cx.refuse("speech request body must be a JSON object");
    try checkFields(cx, raw);
    const model = raw.strField("model") orelse return cx.fail(.request, "model is required: send {s}", .{srv.config.served_name});
    if (!listed(srv.config.model_ids, model))
        return cx.fail(.request, "model {s} is not served here: send {s}", .{ try pyrepr.repr(cx.a, .{ .string = model }), srv.config.served_name });
    const sent = raw.strField("input") orelse return cx.refuse("input must be a non-empty string");
    if (!std.unicode.utf8ValidateSlice(sent)) return cx.refuse("input must be valid UTF-8 text");
    // controls off the list go first, so every draw, every count and the marks read the input without them
    const allowed = if (srv.config.allowed_controls) |list| try drama_allow.filter(cx.a, sent, list) else null;
    // no path sees a mark: off, the input is the one sent without them; on, they cut long-form segments (drama_shift)
    const shifting = if (srv.config.long_form) |options| options.shift_pause_s > 0 else false;
    const marked = try drama_shift.read(cx.a, if (allowed) |f| f.text else sent, shifting);
    const stripped = if (allowed) |f| f.stripped else null;
    const input = marked.text;
    if (reply_text.pyStrip(input).len == 0) return cx.refuse("input must be a non-empty string");
    if ((std.unicode.utf8CountCodepoints(input) catch unreachable) > max_input_chars)
        return cx.fail(.request, "input must be at most {d} characters", .{max_input_chars});
    const format = try responseFormat(cx, raw.strField("response_format") orelse "wav");
    const stream = if (raw.get("stream")) |s| s == .bool and s.bool else false;
    if (stream and format != .pcm) return cx.refuse("stream=true requires response_format='pcm'");
    const voice = raw.strField("voice") orelse raw.strField("speaker") orelse "default";
    const references = try drama_voice.lower(cx, input, raw.get("references") orelse .null, try voiceReferences(srv, cx, voice));
    if (references.len == 0) if (try drama_prompt.problem(cx.a, input)) |message| return cx.refuse(message);
    if (srv.config.long_form) |options| if (try speech_long.plan(srv, cx, sp, marked, references, options)) |long|
        return .{ .prompt = &.{}, .codes = &.{}, .input = input, .references = references, .format = format, .stream = stream, .long = long, .dropped = marked.dropped, .stripped = stripped };
    const rows = drama_prompt.promptRows(cx.a, srv.text, input, references) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.other("the tokenizer cannot encode this prompt"),
    };
    if (rows.tokens.len > sp.max_prompt_tokens)
        return cx.fail(.request, drama_prompt.error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ rows.tokens.len, sp.max_prompt_tokens });
    return .{ .prompt = rows.tokens, .codes = rows.codes, .input = input, .references = references, .format = format, .stream = stream, .dropped = marked.dropped, .stripped = stripped };
}

/// What ``voice`` names: the built-in voice (none), --default-voice's for ``default``, else a --voices file's.
fn voiceReferences(srv: *Server, cx: *Cx, voice: []const u8) errors.Refused![]const drama_prompt.Reference {
    const chosen = if (voice.len == 0 or std.ascii.eqlIgnoreCase(voice, "default")) srv.config.default_voice orelse return &.{} else drama_voice.find(srv.config.voices, voice) orelse {
        const hint = if (srv.config.voices.len == 0) "only the default voice is supported" else "use default or a voice the server loaded with --voices";
        return cx.fail(.request, drama_prompt.error_prefix ++ " unknown voice {s}; {s}", .{ try pyrepr.repr(cx.a, .{ .string = voice }), hint });
    };
    return chosen.references;
}

/// One synthesis, sent whole or streamed; a whole reply that trips (`speech_redraw.Trip`) is redrawn while allowed.
fn synthesize(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, gamma: ?f32, received: i96, redraws: *speech_redraw.Redraws) void {
    const pcm_len = std.math.mul(usize, sp.max_frames, sp.frame_length) catch 0;
    if (pcm_len == 0 or sp.sample_rate == 0 or sp.sample_rate > std.math.maxInt(u32) / 2)
        return fail(c, cx, id, redraws, .server, "the engine advertises no usable speech output");
    const pcm = srv.gpa.alloc(f32, pcm_len) catch return fail(c, cx, id, redraws, .server, "out of memory");
    defer srv.gpa.free(pcm);
    if (srv.keepalive) |k| k.begin();
    defer if (srv.keepalive) |k| k.end();
    var spoken: speech_redraw.Spoken = .{ .text = plan.input, .rows = .{ .tokens = plan.prompt, .codes = plan.codes } };
    const rows = &spoken.rows;
    var next: ?speech_redraw.Spoken = null; // what redraws speak, made at the first trip
    var start_by: ?i96 = null; // a redraw's latest start: a busy engine is waited out until then
    var last: Drawn = undefined; // the tripped draw a redraw replaces, still in its buffer if the redraw never starts
    var guard: speech_register.Guard = .{ .policy = srv.config.register };
    var held: ?Drawn = null; // the draw a register redraw may replace, kept in `pcm` while later draws use `spare`
    var spare: []f32 = &.{};
    defer srv.gpa.free(spare);
    var register_draw = false;
    var k: u8 = 0;
    const drawn: Drawn = while (true) : (k += 1) {
        const audio = if (held == null) pcm else spare;
        var result: api.SpeechResult = .{};
        const guided = drama_cfg.guide(cx.a, srv.text, srv.config.cfg, gamma orelse 1, spoken.text, rows.*) catch |e| return fail(c, cx, id, redraws, .other, if (e == error.OutOfMemory) "out of memory" else "the tokenizer cannot encode this prompt");
        const request: api.SpeechRequest = .{
            .prompt = rows.tokens,
            .codes = rows.codes,
            .pcm = audio,
            .result = &result,
            .seed = api.seedFor(rows.tokens, speech_redraw.Redraws.salt(srv.config.seed_salt, k)),
            .max_frames = sp.max_frames,
            .temperature = drama_prompt.temperature,
            .top_p = drama_prompt.top_p,
            .top_k = drama_prompt.top_k,
            .unconditioned = guided.unconditioned,
            .cfg_gamma = guided.gamma,
            .stream = plan.stream,
        };
        var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
        defer box.deinit();
        const engine_id = srv.next_id.fetchAdd(1, .monotonic);
        const started = clock.nowNs(srv.io);
        submit(srv, c, engine_id, &request, &box, start_by) catch |e| {
            if (e == error.Busy and k > 0) {
                if (register_draw) log.line("speech {s} register redraw not taken: the engine stayed busy past its latest start", .{id}) else redraws.stand(id, "the engine stayed busy past the latest start that fits the budget");
                break last;
            }
            const refused = submitFailure(e);
            return fail(c, cx, id, redraws, refused.kind, refused.message);
        };
        // The engine holds `request`, its prompt, `audio` and `result` until finished: no path below returns before it arrives.
        if (k > 0) {
            if (register_draw) guard.used += 1 else redraws.used += 1;
        }
        register_draw = false;
        if (plan.stream) return streamReply(srv, c, cx, id, sp, plan, gamma, received, &box, engine_id, pcm, &result, redraws);
        if (settle(srv, c, &box, engine_id, null) == null) return left(srv, c, id, rows.tokens.len, received);
        redraws.hollowed(id, result.hollow, sp.sample_rate);
        if (why(&box, result.samples, audio.len)) |message| return fail(c, cx, id, redraws, .server, message);
        var this: Drawn = .{ .samples = result.samples, .reason = box.reason, .prompt = rows.tokens.len, .pcm = audio };
        this.trip = if (redraws.policy.max == 0) null else redraws.trip(this.reason, this.samples, sp.sample_rate, spoken.text);
        const cause = this.trip orelse {
            if (!guard.on()) break this;
            this.jump = guard.measure(srv.gpa, srv.io, id, audio[0..this.samples], sp.sample_rate) catch |e| return fail(c, cx, id, redraws, .server, switch (e) {
                error.NonFinite => "the synthesized audio has non-finite samples",
                error.OutOfMemory => "out of memory",
            });
            if (this.jump == null) break this;
            start_by = guard.again(redraws, srv.io, id, started) orelse break this;
            spare = srv.gpa.alloc(f32, pcm_len) catch {
                log.line("speech {s} register redraw not taken: out of memory", .{id});
                break this;
            };
            held = this;
            register_draw = true;
            last = this;
            continue;
        };
        if (next == null) next = redrawSpoken(srv, cx.a, sp, id, plan, spoken, redraws) orelse break this;
        start_by = redraws.again(srv.io, id, started, cause) orelse break this;
        spoken = next.?;
        last = this;
    };
    const sent = if (held) |h| keep(id, drawn, h) else drawn;
    const data = quantizeInPlace(sent.pcm[0..sent.samples]) catch return fail(c, cx, id, redraws, .server, "the synthesized audio has non-finite samples");
    if (data.len > std.math.maxInt(u32) - 36) return fail(c, cx, id, redraws, .server, "the synthesized audio is too long for one reply");
    var buf: [4]u8 = undefined;
    var trips: [4]u8 = undefined;
    var count: [20]u8 = undefined;
    var register_trip: [1]u8 = undefined;
    var register_redraws: [1]u8 = undefined;
    var guidance: [24]u8 = undefined;
    const heads: Heads = .{ .redraws = redraws.header(&buf), .mid = redraws.hollowHeader(&trips), .stripped = strippedHeader(&count, plan.stripped), .register = guard.headers(&register_trip, &register_redraws), .cfg_gamma = drama_cfg.header(&guidance, gamma) };
    sendAudio(c, plan.format, sp.sample_rate, data, sent.reason, heads) catch {
        c.close = true; // a head already started or a client gone: nothing more can be sent on this connection
        return;
    };
    log.line("speech {s} prompt={d} samples={d} format={s} finish={s} seconds={d:.2}", .{ id, sent.prompt, sent.samples, @tagName(plan.format), @tagName(sent.reason), clock.seconds(clock.nowNs(srv.io) - received) });
}

/// Of `held` and the draw after it, the one to send: the later only if it ended clean and jumps less.
fn keep(id: []const u8, later: Drawn, held: Drawn) Drawn {
    if (later.pcm.ptr == held.pcm.ptr) return held; // the redraw never started
    const better = later.trip == null and later.reason != .silence and (later.jump orelse -std.math.inf(f64)) < held.jump.?;
    log.line("speech {s} register: sent={s} excess={d:.2}st redraw_excess={d:.2}st redraw_finish={s} redraw_trip={s}", .{ id, if (better) "redraw" else "first", held.jump.?, later.jump orelse 0, @tagName(later.reason), if (later.trip) |t| @tagName(t) else "none" });
    return if (better) later else held;
}

/// A whole reply's finished draw: buffer `pcm`, audio length, finish, prompt rows, redraw trip, register jump.
const Drawn = struct { samples: usize, reason: api.Reason, prompt: usize, pcm: []f32, trip: ?speech_redraw.Trip = null, jump: ?f64 = null };

/// What a redraw speaks: `spoken`, or the cue-free input (--silence-redraw-plain); null, logged, if that cannot be.
fn redrawSpoken(srv: *Server, a: Allocator, sp: api.SpeechInfo, id: []const u8, plan: Plan, spoken: speech_redraw.Spoken, redraws: *const speech_redraw.Redraws) ?speech_redraw.Spoken {
    if (!redraws.policy.plain) return spoken;
    const why_not = why: {
        const text = (speech_redraw.plain(a, plan.input, redraws.policy.plain_scope) catch break :why "out of memory") orelse return spoken;
        if (speech_redraw.problem(a, text)) |message| break :why message;
        const cue_free = drama_prompt.promptRows(a, srv.text, text, plan.references) catch break :why "the tokenizer cannot encode the input without its cues";
        if (cue_free.tokens.len > sp.max_prompt_tokens) break :why "the input without its cues does not fit the prompt";
        log.line("speech {s} redraws speak the input without its {s}", .{ id, @tagName(redraws.policy.plain_scope) });
        return .{ .text = text, .rows = cue_free };
    };
    redraws.stand(id, why_not);
    return null;
}

/// Submits, waiting out another request's synthesis until `start_by` if given; Busy, unsubmitted, once it has passed.
pub fn submit(srv: *Server, c: *Conn, engine_id: api.Id, request: *const api.SpeechRequest, box: *Mailbox, start_by: ?i96) api.SpeechSubmitError!void {
    while (true) {
        if (start_by) |by| if (clock.nowNs(srv.io) > by) return error.Busy;
        srv.engine.submitSpeech(engine_id, request, .{ .ctx = box, .event = Mailbox.onEvent }) catch |e| {
            if (e != error.Busy or start_by == null or c.peerGone()) return e;
            std.Io.sleep(srv.io, .fromMilliseconds(20), .awake) catch {};
            continue;
        };
        return;
    }
}

/// Why the engine refused a submission, as the reply's kind and message.
pub fn submitFailure(e: api.SpeechSubmitError) struct { kind: errors.Kind, message: []const u8 } {
    return switch (e) {
        error.Busy => .{ .kind = .capacity, .message = "the engine is busy; retry shortly" },
        error.Closed => .{ .kind = .other, .message = "the scheduler is closed" },
        error.InvalidRequest => .{ .kind = .request, .message = "the engine refused this speech request's prompt, frame limit or sampling" },
        error.Unsupported => .{ .kind = .request, .message = "this model does not support speech synthesis" },
    };
}

/// The client left before the reply ended; the synthesis was cancelled and waited out.
pub fn left(srv: *Server, c: *Conn, id: []const u8, prompt_len: usize, received: i96) void {
    c.close = true;
    srv.metrics.disconnected(srv.io);
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, null, prompt_len, 0, clock.seconds(clock.nowNs(srv.io) - received))});
}

/// Why a finished synthesis has no complete audio, or null. Audio cut short by the frame limit (`length`) or silence
/// guard (`silence`) is returned, marked by X-Finish-Reason (a whole reply's header, a stream's trailer).
pub fn why(box: *const Mailbox, samples: usize, capacity: usize) ?[]const u8 {
    return switch (box.reason) {
        .stop, .length, .silence => if (samples == 0 or samples > capacity) "the engine finished without complete audio" else null,
        .cancelled => "speech synthesis was cancelled",
        .failed => if (box.message.len > 0) box.message else "speech synthesis failed",
    };
}

/// An error reply when synthesis ends before any audio; else the chunks, and the terminator only after a complete one.
fn streamReply(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, gamma: ?f32, received: i96, box: *Mailbox, engine_id: api.Id, pcm: []f32, result: *const api.SpeechResult, redraws: *const speech_redraw.Redraws) void {
    var buf: [4]u8 = undefined;
    var count: [20]u8 = undefined;
    var guidance: [24]u8 = undefined;
    const heads: Heads = .{ .redraws = redraws.header(&buf), .stripped = strippedHeader(&count, plan.stripped), .cfg_gamma = drama_cfg.header(&guidance, gamma) };
    const sent = streamAudio(srv, c, sp, box, engine_id, pcm, "X-Finish-Reason", heads) orelse return left(srv, c, id, plan.prompt.len, received);
    const mismatch: ?[]const u8 = if (result.samples != sent.samples) "the finished synthesis does not match the audio it published" else null;
    const problem = why(box, result.samples, pcm.len) orelse mismatch;
    if (sent.samples == 0) return fail(c, cx, id, redraws, .server, problem orelse "the engine finished without publishing audio");
    if (problem) |message| {
        c.close = true; // no terminating chunk: the client's HTTP/1.1 parser reports an incomplete body
        log.line("speech {s} stream ended incomplete after samples={d}: {s}", .{ id, sent.samples, message });
        return;
    }
    c.writeAll(lastChunk(box.reason)) catch {
        c.close = true;
        return;
    };
    log.line("speech {s} prompt={d} samples={d} format=pcm stream=true finish={s} first_audio_ready_seconds={d:.3} first_frame_written_seconds={d:.3} seconds={d:.2}", .{
        id, plan.prompt.len, sent.samples, @tagName(box.reason), clock.seconds(sent.first_ready - received), clock.seconds(sent.first_written - received), clock.seconds(clock.nowNs(srv.io) - received),
    });
}

/// What a stream sent, when this thread first saw published audio, and when frame 0's own bytes reached the socket.
pub const Sent = struct { samples: usize = 0, first_ready: i96 = 0, first_written: i96 = 0 };

/// Sends what is published as chunks, converted in place; `start`: the Trailer to open the stream with, if not open.
pub fn streamAudio(srv: *Server, c: *Conn, sp: api.SpeechInfo, box: *Mailbox, engine_id: api.Id, pcm: []f32, start: ?[]const u8, heads: Heads) ?Sent {
    var sent: Sent = .{};
    var size: [20]u8 = undefined;
    while (true) {
        const now = settle(srv, c, box, engine_id, sent.samples) orelse return null;
        if (now == sent.samples) return sent; // finished, and everything published is sent
        const first = sent.samples == 0;
        if (first) {
            sent.first_ready = clock.nowNs(srv.io);
            if (start) |trailer| startStream(c, sp.sample_rate, trailer, heads) catch return abandon(srv, box, engine_id);
        }
        // the audio event made pcm[..now] final: this thread converts it in place
        const bytes = quantizeInPlace(pcm[sent.samples..now]) catch return abandon(srv, box, engine_id);
        // frame 0's bytes go apart, so their hand-off time does not depend on the chunk's size
        const split: usize = if (first) @min(2 * sp.frame_length, bytes.len) else 0;
        c.writeAll(std.fmt.bufPrint(&size, "{x}\r\n", .{bytes.len}) catch unreachable) catch return abandon(srv, box, engine_id);
        c.writeAll(bytes[0..split]) catch return abandon(srv, box, engine_id);
        if (first) sent.first_written = clock.nowNs(srv.io);
        c.writeAll(bytes[split..]) catch return abandon(srv, box, engine_id);
        c.writeAll("\r\n") catch return abandon(srv, box, engine_id);
        sent.samples = now;
    }
}

/// Cancels a synthesis whose client is gone, then waits for its finished event; null, for the callers' `return`.
pub fn abandon(srv: *Server, box: *Mailbox, engine_id: api.Id) ?Sent {
    srv.engine.cancel(engine_id);
    box.mutex.lockUncancelable(box.io);
    defer box.mutex.unlock(box.io);
    while (!box.finished) box.cond.waitUncancelable(box.io, &box.mutex);
    return null;
}

/// Waits for `finished` or (with `sent`) samples past it, cancelling if the client leaves; samples so far, or null.
pub fn settle(srv: *Server, c: *Conn, box: *Mailbox, engine_id: api.Id, sent: ?usize) ?usize {
    while (true) {
        box.mutex.lockUncancelable(box.io);
        const idle = if (sent) |n| box.audio == n else true;
        if (!box.finished and idle) box.cond.waitTimeout(box.io, &box.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
        const finished = box.finished;
        const published = box.audio;
        box.mutex.unlock(box.io);
        if (finished or (if (sent) |n| published > n else false)) return published;
        if (c.peerGone()) {
            _ = abandon(srv, box, engine_id);
            return null;
        }
    }
}

/// The optional headers a reply's head carries, each when its option is on, in this order.
pub const Heads = struct { segments: ?[]const u8 = null, spans: ?[]const u8 = null, redraws: ?[]const u8 = null, mid: ?[]const u8 = null, stripped: ?[]const u8 = null, register: ?[2][]const u8 = null, cfg_gamma: ?[]const u8 = null };

fn addHeads(c: *Conn, heads: Heads) error{OutOfMemory}!void {
    if (heads.segments) |value| try c.addHeader("X-Long-Form-Segments", value);
    if (heads.spans) |value| try c.addHeader("X-Pause-Spans", value);
    if (heads.redraws) |n| try c.addHeader("X-Silence-Redraws", n);
    if (heads.mid) |n| try c.addHeader("X-Silence-Mid-Trips", n);
    if (heads.stripped) |n| try c.addHeader("X-Controls-Stripped", n);
    if (heads.register) |r| {
        try c.addHeader("X-Register-Trip", r[0]);
        try c.addHeader("X-Register-Redraws", r[1]);
    }
    if (heads.cfg_gamma) |g| try c.addHeader("X-Cfg-Gamma", g);
}

/// The X-Controls-Stripped value, with --allowed-controls on.
pub fn strippedHeader(buf: *[20]u8, stripped: ?usize) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{d}", .{stripped orelse return null}) catch unreachable;
}

/// 200 with raw PCM16LE in HTTP/1.1 chunks, under the reference's sample-format headers, then `heads`.
pub fn startStream(c: *Conn, sample_rate: u32, trailer: []const u8, heads: Heads) error{Closed}!void {
    c.startResponse(200, null) catch return error.Closed;
    c.addHeader("Content-Type", Format.pcm.contentType()) catch return error.Closed;
    c.addHeader("Transfer-Encoding", "chunked") catch return error.Closed;
    var rate: [16]u8 = undefined;
    c.addHeader("X-Sample-Rate", std.fmt.bufPrint(&rate, "{d}", .{sample_rate}) catch unreachable) catch return error.Closed;
    c.addHeader("X-Channels", "1") catch return error.Closed;
    c.addHeader("X-Bit-Depth", "16") catch return error.Closed;
    c.addHeader("Trailer", trailer) catch return error.Closed;
    addHeads(c, heads) catch return error.Closed;
    if (c.close) c.addHeader("Connection", "close") catch return error.Closed;
    // each chunk leaves at once rather than waiting on the client's acknowledgement of the last (no effect off TCP)
    std.posix.setsockopt(c.fd, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &std.mem.toBytes(@as(c_int, 1))) catch {};
    // a blocking write into a full socket returns after the stall limit, so writeAll's poll can end a stalled stream
    const stall: std.posix.timeval = .{ .sec = @divTrunc(c.timeouts.write_ms, 1000), .usec = @rem(c.timeouts.write_ms, 1000) * 1000 };
    std.posix.setsockopt(c.fd, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&stall)) catch {};
    return c.finish("");
}

/// The terminating chunk after a complete synthesis's last samples; audio cut short (clipped at the frame limit, or
/// ended by the silence guard) ends with the X-Finish-Reason trailer instead.
fn lastChunk(reason: api.Reason) []const u8 {
    return switch (reason) {
        .length => "0\r\nX-Finish-Reason: length\r\n\r\n",
        .silence => "0\r\nX-Finish-Reason: silence\r\n\r\n",
        else => "0\r\n\r\n",
    };
}

fn refuse(c: *Conn, cx: *Cx, id: []const u8, redraws: *const speech_redraw.Redraws, e: errors.Refused) void {
    if (e == error.OutOfMemory) {
        cx.kind = .server;
        cx.message = "out of memory";
    }
    send(c, cx, id, redraws);
}

pub fn fail(c: *Conn, cx: *Cx, id: []const u8, redraws: *const speech_redraw.Redraws, kind: errors.Kind, message: []const u8) void {
    cx.kind = kind;
    cx.message = message;
    send(c, cx, id, redraws);
}

/// OpenAI's error object, logged first: the access line shows only the status; with redraws on, the count taken.
fn send(c: *Conn, cx: *Cx, id: []const u8, redraws: *const speech_redraw.Redraws) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, cx.message)});
    const body = openai.errorBody(cx.a, cx, null) catch return;
    const text = json.stringify(cx.a, openai.wrapError(cx.a, body) catch return, .{}) catch return;
    var n: [4]u8 = undefined;
    c.sendJsonWith(cx.status(), text, if (redraws.header(&n)) |value| .{ .name = "X-Silence-Redraws", .value = value } else null);
}

/// The head, then the WAV header and the samples where they lie: the audio is never copied to prepend a header.
pub fn sendAudio(c: *Conn, format: Format, sample_rate: u32, data: []const u8, reason: api.Reason, heads: Heads) error{ Closed, OutOfMemory }!void {
    const header: ?[wav_header_len]u8 = if (format == .wav) wavHeader(sample_rate, @intCast(data.len)) else null;
    const header_len: usize = if (header != null) wav_header_len else 0;
    try c.startResponse(200, null);
    try c.addHeader("Content-Type", format.contentType());
    var len: [24]u8 = undefined;
    try c.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{header_len + data.len}) catch unreachable);
    try c.addHeader("X-Finish-Reason", @tagName(reason));
    try addHeads(c, heads);
    if (c.close) try c.addHeader("Connection", "close");
    defer c.out.clearRetainingCapacity();
    if (!std.mem.eql(u8, c.version, "HTTP/0.9")) try c.out.appendSlice(c.gpa, "\r\n");
    if (header) |h| try c.out.appendSlice(c.gpa, &h);
    try c.writeAll(c.out.items);
    try c.writeAll(data);
}

pub const wav_header_len = 44;

/// A canonical 44-byte RIFF/WAVE header for mono 16-bit PCM of ``data_bytes`` bytes.
pub fn wavHeader(sample_rate: u32, data_bytes: u32) [wav_header_len]u8 {
    var h: [wav_header_len]u8 = undefined;
    @memcpy(h[0..4], "RIFF");
    std.mem.writeInt(u32, h[4..8], 36 + data_bytes, .little);
    @memcpy(h[8..12], "WAVE");
    @memcpy(h[12..16], "fmt ");
    std.mem.writeInt(u32, h[16..20], 16, .little); // fmt chunk size
    std.mem.writeInt(u16, h[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, h[22..24], 1, .little); // mono
    std.mem.writeInt(u32, h[24..28], sample_rate, .little);
    std.mem.writeInt(u32, h[28..32], sample_rate * 2, .little); // byte rate
    std.mem.writeInt(u16, h[32..34], 2, .little); // block align
    std.mem.writeInt(u16, h[34..36], 16, .little); // bits a sample
    @memcpy(h[36..40], "data");
    std.mem.writeInt(u32, h[40..44], data_bytes, .little);
    return h;
}

/// The reference's WAV and PCM quantization, ``(np.clip(x, -1, 1) * 32767.0).astype(np.int16)`` (toward zero),
/// written as PCM16LE over the samples' own memory: each sample is read before its bytes or later ones are overwritten.
pub fn quantizeInPlace(pcm: []f32) error{NonFinite}![]u8 {
    const bytes = std.mem.sliceAsBytes(pcm);
    for (0..pcm.len) |i| std.mem.writeInt(i16, bytes[2 * i ..][0..2], try pcm16(pcm[i]), .little);
    return bytes[0 .. 2 * pcm.len];
}

/// One sample as the reference quantizes it.
pub fn pcm16(x: f32) error{NonFinite}!i16 {
    if (!std.math.isFinite(x)) return error.NonFinite;
    return @intFromFloat(std.math.clamp(x, -1.0, 1.0) * 32767.0);
}

test "quantization clips, truncates toward zero and stays in place" {
    var pcm = [_]f32{ 0, 1, -1, 1.5, -7, 0.5, -0.5, 0.25, 1e-6, -1e-6 };
    const want = [_]i16{ 0, 32767, -32767, 32767, -32767, 16383, -16383, 8191, 0, 0 };
    const bytes = try quantizeInPlace(&pcm);
    try std.testing.expectEqual(@as(usize, 2 * want.len), bytes.len);
    try std.testing.expectEqual(@intFromPtr(&pcm), @intFromPtr(bytes.ptr));
    for (want, 0..) |w, i| try std.testing.expectEqual(w, std.mem.readInt(i16, bytes[2 * i ..][0..2], .little));
}

test "non-finite audio is refused" {
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        var pcm = [_]f32{ 0.1, bad, 0.2 };
        try std.testing.expectError(error.NonFinite, quantizeInPlace(&pcm));
    }
}

test "the WAV header describes mono 16-bit PCM at the advertised rate" {
    const h = wavHeader(44100, 4096);
    try std.testing.expectEqualStrings("RIFF", h[0..4]);
    try std.testing.expectEqual(@as(u32, 36 + 4096), std.mem.readInt(u32, h[4..8], .little));
    try std.testing.expectEqualStrings("WAVEfmt ", h[8..16]);
    try std.testing.expectEqual(@as(u32, 16), std.mem.readInt(u32, h[16..20], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, h[20..22], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, h[22..24], .little));
    try std.testing.expectEqual(@as(u32, 44100), std.mem.readInt(u32, h[24..28], .little));
    try std.testing.expectEqual(@as(u32, 88200), std.mem.readInt(u32, h[28..32], .little));
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, h[32..34], .little));
    try std.testing.expectEqual(@as(u16, 16), std.mem.readInt(u16, h[34..36], .little));
    try std.testing.expectEqualStrings("data", h[36..40]);
    try std.testing.expectEqual(@as(u32, 4096), std.mem.readInt(u32, h[40..44], .little));
}

test "fields: unknown ones are refused even when null, modifiers when set" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"voice\": \"default\", \"response_format\": \"wav\", \"speed\": null, \"references\": null}", "" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream_audio\": null}", "unknown speech request field: stream_audio" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"speed\": 1.0}", "speed is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream\": false, \"stream_format\": \"audio\"}", "" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream\": null, \"stream_format\": null}", "" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream\": \"yes\"}", "stream must be a boolean" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream_format\": \"sse\"}", "stream_format sse is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream_format\": \"mp3\"}", "stream_format must be one of: audio, sse" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"temperature\": 0.7}", "temperature is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"ref_audio\": \"x.wav\"}", "ref_audio is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": 3}", "input must be a string" },
        .{ "{\"model\": null, \"input\": \"hi\"}", "model must be a string" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"voice\": null}", "voice must be a string" },
    };
    for (cases) |case| {
        var cx: Cx = .{ .a = a };
        const body = (try json.parse(a, case[0])).ok;
        if (case[1].len == 0) {
            try checkFields(&cx, body);
        } else {
            try std.testing.expectError(error.Refused, checkFields(&cx, body));
            try std.testing.expectEqualStrings(case[1], cx.message);
        }
    }
}

test "response_format is WAV or PCM, stripped and case-folded as Python does" {
    var cx: Cx = .{ .a = std.testing.allocator };
    try std.testing.expectEqual(Format.wav, try responseFormat(&cx, " WAV\n"));
    try std.testing.expectEqual(Format.pcm, try responseFormat(&cx, "\u{3000}Pcm\x1c"));
    for ([_][]const u8{ "mp3", "", "wave", "pcm16" }) |bad| try std.testing.expectError(error.Refused, responseFormat(&cx, bad));
}
