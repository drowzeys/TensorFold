//! Full GLM-5.3 (glm_moe_dsa) at TP4 behind the native server: rank 0 serves the OpenAI routes and shares each
//! request with ranks 1..3, which mirror it (Glm53Engine.generate / follow, with the request's header over the TCP
//! rendezvous instead of the all-gather). One request at a time (--parallel 1), its tokens streamed round by round;
//! EOS and rank 0's stop (a stop string, a cancel) end it on every rank after the same round (cuda_runner.run).
//!
//! Prompt reuse (TF_GLM53_PROMPT_REUSE=1: cuda_reuse.zig) keeps prompt states in memory; --learn (upstream 1.0.2's
//! learned states, prompt_imprint.zig) also keeps the shared ones (system prompt + tools) on disk: every rank writes
//! and reads its own shard of a state (<learn-dir>/<identity>/<key>.r<rank>.bin, the same key), rank 0 decides hit or
//! miss and sends the command, every rank answers and all agree before any resumes. The identity covers the
//! checkpoint, the split, the kernel pack and a startup probe's prompt-pass bits on every rank, so a state is only
//! read by a build that computes the same bits.
//!
//! Ranks: TF_GLM53_RANK / TF_GLM53_WORLD / TF_GLM53_MASTER and the other TF_GLM53_* variables (cuda_engine.Options
//! .fromEnv); every rank runs the same `tensorfold-native serve` command. Ranks 1..3 never listen: `Opened.follow`.
//!
//! Phase 4b, `--parallel N` (N > 1; cuda_multi.zig, multi.GlmMultiDecoder + the Python Scheduler): rank 0's engine
//! thread admits waiting requests into free stream slots (the rest queue, oldest first), then runs rounds - a prompt
//! chunk for the oldest filling prompt, quick fills, one shared decode round over every decoding stream - and ends the
//! streams that finished (count, EOS, a stop string, a cancel). Ranks 1..3 apply rank 0's messages in order: ADMIT
//! (the request, its slot and prompt-reuse choice), FILL, ROUND (each stream's position, pending token and backlog:
//! a follower that disagrees stops), FINISH; every rank computes the picks itself, so no results message is needed.
//! Prompt reuse keeps states in the streams' slots; --learn needs --parallel 1.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const glm = @import("glm53");
const eng = glm.engine;
const smp = glm.sampling;
const Allocator = std.mem.Allocator;
const Imprint = api.prompt_imprint.Imprint;

pub const model_type = "glm_moe_dsa";
pub const formats: []const []const u8 = &.{"exl3"};
/// Glm53Engine.DEFAULT_CONTEXT
pub const default_context: i64 = 32768;

// ------------------------------------------------------------------------------------------------- protocol ---

const magic: u32 = 0x47353348; // "G53H"

const Kind = enum(u32) { request = 1, load = 2, persist = 3, forget = 4, bye = 5, admit = 6, fill = 7, round = 8, finish = 9 };

const flag_flush: u32 = 1; // PromptReuse.FLUSH
const flag_draft: u32 = 2; // "draft": true (drafted, resumable, keeps)
const flag_greedy: u32 = 4;

/// Rank 0's command, then its payload (u32 words: the prompt, stops, keeps, eos for a request; a state's tokens for
/// load / persist). Fixed size, little-endian (every rank is the same machine type).
const Head = extern struct {
    magic: u32 = magic,
    kind: u32,
    seq: u64 = 0,
    max_tokens: u32 = 0,
    flags: u32 = 0,
    top_k: u32 = 0,
    prompt_len: u32 = 0,
    seed: u64 = 0,
    temperature: f64 = 0,
    top_p: f64 = 1,
    min_p: f64 = 0,
    begin: u32 = 0,
    n_stops: u32 = 0,
    n_keeps: u32 = 0,
    n_eos: u32 = 0,
    key: u64 = 0,
    at: u32 = 0,
    pad: u32 = 0,
    /// the seal: a hash of this header (with sum 0) and its payload, checked by every follower (sealed by send)
    sum: u64 = 0,
};

/// A message's seal: the header with `sum` zeroed, then its payload words.
fn sealOf(head: *const Head, payload: []const u32) u64 {
    var bare = head.*;
    bare.sum = 0;
    var x = std.hash.Wyhash.init(magic);
    x.update(std.mem.asBytes(&bare));
    x.update(std.mem.sliceAsBytes(payload));
    return x.final();
}

// ---------------------------------------------------------------------------------------------------- host ---

const Job = struct {
    id: api.Id,
    request: *const api.Request,
    sink: api.Sink,
    submitted: i96,
};

pub const Host = struct {
    gpa: Allocator,
    io: std.Io,
    boot: *eng.Boot,
    session: eng.Session,
    rank: usize,
    info_: api.Info = .{},
    startup: []u8 = &.{},
    // --learn
    learned: ?Imprint = null,
    identity: u64 = 0,
    // rank 0's engine thread and its queue
    thread: ?std.Thread = null,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queue: std.ArrayList(Job) = .empty,
    cancels: std.ArrayList(api.Id) = .empty,
    closing: bool = false,
    running: ?api.Id = null,
    seq: u64 = 0,
    flush_file: ?[]const u8 = null,
    /// TF_GLM53_FAIL_FILE (a test switch, --parallel N): when the file appears rank 0 takes it and marks the world
    /// broken, as a failed round does
    fail_file: ?[]const u8 = null,
    /// --parallel 1: the stop vote between prompt chunks (TF_GLM53_PROMPT_VOTES=0 on every rank: off)
    prompt_votes: bool = true,
    dump: ?[]const u8 = null,
    out: std.ArrayList(u32) = .empty,
    last_rate: f64 = 0,
    last_prefill_rate: f64 = 0,
    // --parallel N > 1: the concurrent decoder (boot.multi), its live requests (rank 0), the multi messages' number
    parallel: usize = 1,
    lives: std.ArrayList(*Live) = .empty,
    mseq: u64 = 0,
    streaming: u32 = 0, // rank 0: streams admitted and not finished (for /status)
    // --parallel N: a round's tokens reach the replies once the next round's GPU work is queued (multi.Idle), not
    // between a round's picks and the next round's message, where the followers and every GPU wait on rank 0
    // (TF_GLM53_DEFER_EMITS=0: at once, as before)
    defer_emits: bool = false,
    // --parallel N: requests that reach an idle engine together are admitted together (TF_GLM53_ADMIT_QUIET_MS,
    // default 5: the engine waits until no request has come for that long, at most TF_GLM53_ADMIT_CAP_MS, default
    // 30, or until N wait; 0: the first request at once). Requests sent together then start decoding in the same
    // round, as tf-glm53-generate --mode 4b admits a group: identical prompts decode in lockstep, their verify rows
    // routing to the same experts (Python's slower admission gets the same grouping by its message latency)
    admit_quiet_ns: i96 = 0,
    admit_cap_ns: i96 = 0,
    // the watchdog thread, on every rank (watch): a rank whose connection closed ends this one, and
    // the stall limit (TF_GLM53_STALL_S; default by context, defaultStallS; 0: off): while this rank has a request in
    // hand (busy) and nothing moved it on (beat: a round's tokens, a --parallel step or message) for that long - a rank
    // or a collective is stuck - the process exits, so a wedged world fails loudly instead of waiting for good
    stall_ns: i64 = 0,
    beat_ns: std.atomic.Value(i64) = .init(0),
    busy: std.atomic.Value(bool) = .init(false),
    stop_watch: std.atomic.Value(bool) = .init(false),
    watcher: ?std.Thread = null,

    fn lock(h: *Host) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *Host) void {
        h.mutex.unlock(h.io);
    }

    fn now(h: *const Host) i96 {
        return std.Io.Clock.awake.now(h.io).toNanoseconds();
    }

    // ----------------------------------------------------------------------------------------- the engine API ---

    pub fn engine(h: *Host) api.Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn } };
    }

    fn self(ctx: *anyopaque) *Host {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) api.Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const h = self(ctx);
        if (h.rank != 0) return error.Closed;
        h.lock();
        defer h.unlock();
        if (h.closing) return error.Closed;
        h.queue.append(h.gpa, .{ .id = id, .request = request, .sink = sink, .submitted = h.now() }) catch return error.Busy;
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: api.Id) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn statusFn(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        const running: u32 = if (h.parallel > 1) h.streaming else @intFromBool(h.running != null);
        out.* = .{ .running = running, .waiting = @intCast(h.queue.items.len), .decode_tokens_per_second = if (running > 0) h.last_rate else 0, .prefill_tokens_per_second = h.last_prefill_rate };
    }

    fn memoryFn(_: *anyopaque, reset_peak: bool) ?api.Memory {
        const u = cuda.usage(reset_peak);
        return .{ .active = u.device, .cache = 0, .peak = u.peak };
    }

    fn cancelled(h: *Host, id: api.Id) bool {
        h.lock();
        defer h.unlock();
        return h.closing or std.mem.indexOfScalar(api.Id, h.cancels.items, id) != null;
    }

    // ------------------------------------------------------------------------------------- the stall watchdog ---

    fn beat(h: *Host) void {
        h.beat_ns.store(@intCast(h.now()), .release);
    }

    fn setBusy(h: *Host, on: bool) void {
        if (on) h.beat();
        h.busy.store(on, .release);
    }

    /// Every rank's watchdog thread. A rank that has gone (its rendezvous connection closed: the process died or
    /// left) ends this one too - rank 0 whenever it sees one, a follower while it has a request in hand (idle, its
    /// own loop reads rank 0's goodbye or its closed connection and leaves cleanly) - seen on two checks a second
    /// apart, so a goodbye in flight is read first. And a request in hand with no progress for stall_ns ends the
    /// process (a rank that stands still without closing: paused, or stuck in a collective).
    fn watch(h: *Host) void {
        var gone: u32 = 0;
        while (!h.stop_watch.load(.acquire)) {
            std.Io.sleep(h.io, .fromMilliseconds(1000), .awake) catch {};
            if (h.stop_watch.load(.acquire)) return;
            const busy = h.busy.load(.acquire);
            if (h.boot.rdv.peerClosed()) |r| {
                gone = if (h.rank == 0 or busy) gone + 1 else 0;
                if (gone >= 2) {
                    std.log.err("glm53 rank {d}: rank {d} has gone (its connection closed){s}: the world cannot go on; exiting so the ranks can be restarted instead of hanging", .{ h.rank, r, if (busy) " with a request in hand" else "" });
                    leaveNow(4);
                }
            } else gone = 0;
            if (!busy or h.stall_ns <= 0) continue;
            const idle: i64 = @as(i64, @intCast(h.now())) - h.beat_ns.load(.acquire);
            if (idle <= h.stall_ns) continue;
            std.log.err("glm53 rank {d}: no progress on the request in hand for {d} s (TF_GLM53_STALL_S): a rank or a collective is stuck; exiting so the ranks can be restarted instead of hanging", .{ h.rank, @divTrunc(idle, std.time.ns_per_s) });
            leaveNow(3);
        }
    }

    /// A follower's tokens hook: progress for the watchdog (a follower's stop wish is never read).
    fn followerBeat(ctx: *anyopaque, _: []const u32) bool {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.beat();
        return false;
    }

    /// A follower between two prompt chunks: progress (rank 0's wish reaches it through the vote).
    fn followerChunk(ctx: *anyopaque) bool {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.beat();
        return false;
    }

    // --------------------------------------------------------------------------------------- rank 0's thread ---

    fn loop(h: *Host) void {
        h.boot.ctx.makeCurrent() catch |e| std.log.err("glm53: cuCtxSetCurrent on the engine thread: {t}", .{e});
        if (h.parallel > 1) return h.loopMulti();
        while (true) {
            h.lock();
            while (h.queue.items.len == 0 and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
            if (h.closing) {
                var left = h.queue;
                h.queue = .empty;
                h.unlock();
                for (left.items) |j| emit(j, .{ .finished = .{ .reason = .cancelled } });
                left.deinit(h.gpa);
                return;
            }
            const job = h.queue.orderedRemove(0);
            // a cancel that arrived while it waited
            var gone = false;
            for (h.cancels.items, 0..) |c, i| if (c == job.id) {
                _ = h.cancels.swapRemove(i);
                gone = true;
                break;
            };
            if (!gone) h.running = job.id;
            h.unlock();
            if (gone) {
                emit(job, .{ .finished = .{ .reason = .cancelled } });
                continue;
            }
            h.serveJob(job);
            h.lock();
            h.running = null;
            var i: usize = 0;
            while (i < h.cancels.items.len) {
                if (h.cancels.items[i] == job.id) _ = h.cancels.swapRemove(i) else i += 1;
            }
            h.unlock();
        }
    }

    fn emit(job: Job, event: api.Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// The tokens hook on rank 0: the prefilled event before the first token, each round's tokens to the sink, the
    /// request's stop strings and its cancel as rank 0's stop wish.
    const Live = struct {
        h: *Host,
        job: Job,
        emitted: std.ArrayList(u32) = .empty,
        begin: u32 = 0,
        first: ?i96 = null,
        stop_hit: bool = false,
        cancel_hit: bool = false,
        started: i96,
        // --parallel N: the stream's id, what the dump records
        sid: u32 = 0,
        sampling: ?smp.Sampling = null,
        draft: bool = true,
        // the emitted tokens the reply has heard (Host.defer_emits: the rest wait for flush)
        sent: usize = 0,

        fn tokens(ctx: *anyopaque, toks: []const u32) bool {
            const lv: *Live = @ptrCast(@alignCast(ctx));
            lv.h.beat();
            const first = lv.first == null;
            if (first) {
                lv.first = lv.h.now();
                emit(lv.job, .{ .prefilled = lv.begin });
            }
            if (lv.emitted.appendSlice(lv.h.gpa, toks)) {
                // the first token at once (time to first token); later rounds' wait for Host.flushEmits
                if (first or !lv.h.defer_emits) lv.flush();
            } else |_| {
                lv.flush();
                emit(lv.job, .{ .tokens = toks });
            }
            if (lv.job.request.stop) |s| {
                if (s.check(s.ctx, lv.emitted.items)) lv.stop_hit = true;
            }
            if (lv.h.cancelled(lv.job.id)) lv.cancel_hit = true;
            return lv.stop_hit or lv.cancel_hit;
        }

        /// Rank 0 between two prompt chunks (--parallel 1): progress, and the request's cancel as its stop wish - the
        /// client left while its prompt fills, so every rank stops before the next chunk.
        fn chunk(ctx: *anyopaque) bool {
            const lv: *Live = @ptrCast(@alignCast(ctx));
            lv.h.beat();
            if (lv.h.cancelled(lv.job.id)) lv.cancel_hit = true;
            return lv.cancel_hit;
        }

        /// The emitted tokens the reply has not heard, in one event.
        fn flush(lv: *Live) void {
            if (lv.emitted.items.len <= lv.sent) return;
            emit(lv.job, .{ .tokens = lv.emitted.items[lv.sent..] });
            lv.sent = lv.emitted.items.len;
        }
    };

    /// Rank 0 (--parallel N): every live reply's unheard tokens.
    fn flushEmits(h: *Host) void {
        for (h.lives.items) |lv| lv.flush();
    }

    /// multi.Idle on rank 0: a round's window is queued and the engine thread is about to wait for it.
    fn idleFn(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.flushEmits();
    }

    fn serveJob(h: *Host, job: Job) void {
        const r = job.request;
        const started = h.now();
        var live: Live = .{ .h = h, .job = job, .started = started };
        defer live.emitted.deinit(h.gpa);
        h.setBusy(true);
        defer h.setBusy(false);
        const outcome = h.runRequest(job, &live) catch |e| {
            if (e == error.PromptStopped) {
                // the client left while its prompt filled: every rank stopped at the same chunk
                std.log.info("glm53: done req-{d} prompt={d} tokens=0 finish=cancelled (during the prompt, after {d:.2}s) seq={d}", .{ job.id, job.request.prompt.len, @as(f64, @floatFromInt(h.now() - started)) / 1e9, h.seq });
                emit(job, .{ .prefilled = 0 });
                emit(job, .{ .finished = .{ .reason = .cancelled } });
                return;
            }
            std.log.err("glm53: request {d} failed: {t}", .{ job.id, e });
            if (live.first == null) emit(job, .{ .prefilled = 0 });
            emit(job, .{ .finished = .{ .reason = .failed, .message = words(e) } });
            return;
        };
        defer h.gpa.free(outcome.kept);
        const st = outcome.stats;
        const reason: api.Reason = if (live.cancel_hit) .cancelled else if (st.eos or live.stop_hit) .stop else .length;
        h.lock();
        h.last_rate = st.tokS(h.out.items.len);
        if (st.prefill_s > 0) h.last_prefill_rate = @as(f64, @floatFromInt(r.prompt.len - st.begin)) / st.prefill_s;
        h.unlock();
        if (h.dump) |path| h.writeDump(path, job, &live, outcome, h.out.items) catch |e| std.log.warn("glm53: TF_GLM53_DUMP {s}: {t}", .{ path, e });
        if (!@import("builtin").is_test) {
            const ttft: f64 = if (live.first) |f| @as(f64, @floatFromInt(f - started)) / 1e9 else 0;
            const n_out = h.out.items.len;
            std.log.info("glm53: done req-{d} prompt={d} cached={d} tokens={d} finish={t} tok/s={d:.1} ttft={d:.2}s prefill={d:.2}s decode={d:.2}s rounds={d} accepted={d}/{d} sampled={s} seq={d}", .{ job.id, r.prompt.len, st.begin, n_out, reason, st.tokS(n_out), ttft, st.prefill_s, st.decode_s, st.rounds, st.accepted, st.drafted, if (outcome.sampling != null) "true" else "false", h.seq });
        }
        // --learn: the request's shared states reach the disk before its reply ends (a client that stops the server
        // right after the reply still finds them learned)
        h.afterRequest();
        emit(job, .{ .finished = .{ .reason = reason, .stats = .{ .rounds = st.rounds, .drafted = st.drafted, .accepted = st.accepted, .min_rows = 1, .prefill_seconds = st.prefill_s } } });
    }

    const Outcome = struct { stats: glm.runner.Stats, kept: []usize, learned: usize, sampling: ?smp.Sampling, draft: bool, begin: usize };

    /// Rank 0: Glm53Engine.generate - the clamps, the flush, a learned state read back when it resumes more than any
    /// kept one, PromptReuse.choose, the header to every rank, then the run.
    fn runRequest(h: *Host, job: Job, live: *Live) !Outcome {
        const r = job.request;
        const sess = &h.session;
        const max_tokens = try sess.clampTokens(r.prompt.len, r.max_tokens);
        const s = try samplingOf(r);
        const draft = r.drafts;
        var flags: u32 = if (draft) flag_draft else 0;
        if (s == null) flags |= flag_greedy;
        if (h.flush_file) |path| if (sess.reuse != null and takeFlush(path)) {
            flags |= flag_flush;
            try sess.reuse.?.flush(); // rank 0 forgets before it chooses; the others at the request
        };
        var learned: usize = 0;
        if (draft and h.learned != null and sess.reuse != null) learned = try h.loadLearned(r.prompt);
        const c = try sess.choose(r.prompt, draft);
        defer if (sess.reuse != null) c.deinit(h.gpa);
        live.begin = @intCast(c.begin);
        // the request's number is taken when its header goes out (below): a request that fails before that - no
        // memory for its payload - must not leave rank 0 a number ahead of the followers
        var head: Head = .{ .kind = @intFromEnum(Kind.request), .seq = h.seq + 1, .max_tokens = @intCast(max_tokens), .flags = flags, .prompt_len = @intCast(r.prompt.len), .begin = @intCast(c.begin), .n_stops = @intCast(c.stops.len), .n_keeps = @intCast(c.keeps.len), .n_eos = @intCast(r.eos.len) };
        if (s) |x| {
            head.seed = x.seed;
            head.temperature = x.temperature;
            head.top_k = @intCast(x.top_k);
            head.top_p = x.top_p;
            head.min_p = x.min_p;
        }
        const words_n = r.prompt.len + c.stops.len + c.keeps.len + r.eos.len;
        const payload = try h.gpa.alloc(u32, words_n);
        defer h.gpa.free(payload);
        @memcpy(payload[0..r.prompt.len], r.prompt);
        for (c.stops, 0..) |p, i| payload[r.prompt.len + i] = @intCast(p);
        @memcpy(payload[r.prompt.len + c.stops.len ..][0..c.keeps.len], c.keeps);
        @memcpy(payload[r.prompt.len + c.stops.len + c.keeps.len ..], r.eos);
        h.seq += 1;
        try h.send(&head, payload);
        const done = try sess.exec(.{ .prompt = r.prompt, .max_tokens = max_tokens, .sampling = s, .draft = draft, .eos = r.eos, .begin = c.begin, .stops = c.stops, .keeps = c.keeps, .flags = flags & flag_flush, .hooks = .{ .ctx = live, .tokens = Live.tokens, .stop = Live.chunk }, .prompt_votes = h.prompt_votes }, &h.out);
        return .{ .stats = done.stats, .kept = done.kept, .learned = learned, .sampling = s, .draft = draft, .begin = c.begin };
    }

    /// Rank 0's command and payload to ranks 1..
    fn send(h: *Host, head: *Head, payload: []const u32) !void {
        if (h.boot.o.world == 1) return;
        head.sum = sealOf(head, payload);
        try h.boot.rdv.broadcast(std.mem.asBytes(head));
        if (payload.len > 0) try h.boot.rdv.broadcast(@constCast(std.mem.sliceAsBytes(payload)));
    }

    fn takeFlush(path: []const u8) bool {
        var buf: [1100]u8 = undefined;
        const z = std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0) catch return false;
        return std.c.unlink(z) == 0;
    }

    // ----------------------------------------------------------------------------- --parallel N (rank 0) ---

    /// Rank 0's engine thread with --parallel N (the Python Scheduler's loop): idle until a request waits; admit the
    /// waiting ones into free slots (oldest first), run a round (multi.round), end the streams that finished.
    fn loopMulti(h: *Host) void {
        const mu = h.boot.multi.?;
        if (h.defer_emits) mu.idle = .{ .ctx = h, .f = idleFn };
        var admit_buf: [glm.multi.max_streams]Job = undefined;
        var gone_buf: [64]Job = undefined;
        while (true) {
            h.lock();
            while (h.queue.items.len == 0 and mu.live() == 0 and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
            if (mu.live() == 0 and h.queue.items.len > 0) h.gatherArrivals();
            if (h.closing) {
                var left = h.queue;
                h.queue = .empty;
                h.unlock();
                for (left.items) |j| emit(j, .{ .finished = .{ .reason = .cancelled } });
                left.deinit(h.gpa);
                h.endAll(.cancelled, "the server is stopping");
                return;
            }
            var n_admit: usize = 0;
            var n_gone: usize = 0;
            while (h.queue.items.len > 0 and mu.live() + n_admit < h.parallel and n_gone < gone_buf.len) {
                const job = h.queue.orderedRemove(0);
                var gone = false;
                for (h.cancels.items, 0..) |c, i| if (c == job.id) {
                    _ = h.cancels.swapRemove(i);
                    gone = true;
                    break;
                };
                if (gone) {
                    gone_buf[n_gone] = job;
                    n_gone += 1;
                } else {
                    admit_buf[n_admit] = job;
                    n_admit += 1;
                }
            }
            h.unlock();
            for (gone_buf[0..n_gone]) |j| emit(j, .{ .finished = .{ .reason = .cancelled } });
            // an admission is a request in hand for the watchdog: its message (the prompt: megabytes at 1M tokens) to
            // a follower that stands still would otherwise block an idle rank 0 for good, unwatched
            if (n_admit > 0) h.setBusy(true);
            for (admit_buf[0..n_admit]) |j| h.admitJob(j);
            if (h.fail_file) |path| if (mu.live() > 0 and takeFlush(path)) {
                std.log.err("glm53: TF_GLM53_FAIL_FILE {s}: this world is marked broken (a test of the restart path)", .{path});
                mu.broken = h.boot.o.world > 1;
            };
            h.leaveIfBroken();
            h.setStreaming(mu.live());
            if (mu.live() == 0) continue;
            h.endCancelledFills() catch |e| {
                std.log.err("glm53: ending cancelled streams failed: {t}", .{e});
                h.endAll(.failed, words(e));
                h.leaveIfBroken();
                continue;
            };
            if (mu.live() == 0) continue;
            const link: glm.multi.Link = .{ .ctx = h, .fill = linkFill, .round = linkRound };
            mu.step(link) catch |e| {
                std.log.err("glm53: a concurrent round failed: {t} (every live request fails; restart all ranks if the ranks are out of step)", .{e});
                h.endAll(.failed, words(e));
                h.leaveIfBroken();
                continue;
            };
            h.endDone() catch |e| {
                std.log.err("glm53: ending finished streams failed: {t}", .{e});
                h.endAll(.failed, words(e));
                h.leaveIfBroken();
            };
            h.setStreaming(mu.live());
        }
    }

    /// Rank 0 (--parallel N): a world whose ranks are out of step (a failed round or admission: the decoder is
    /// `broken`) can serve nothing more, yet /health stayed green and every later request failed with "restart the
    /// server". It ends here instead: the live requests fail, then rank 0 exits with code 5, the followers leave at
    /// its closed connection (the watchdog's `has gone`), and a restart policy sees the world down.
    fn leaveIfBroken(h: *Host) void {
        const mu = h.boot.multi.?;
        if (!mu.broken or @import("builtin").is_test) return;
        if (h.lives.items.len > 0) h.endAll(.failed, words(error.RanksOutOfStep));
        std.log.err("glm53 rank 0: the ranks are out of step after a failed round or admission: this world can serve nothing more; exiting (code 5) so the ranks can be restarted instead of failing every request", .{});
        std.Io.sleep(h.io, .fromMilliseconds(1000), .awake) catch {}; // the failed replies leave first
        leaveNow(5);
    }

    /// Rank 0, the lock held, the engine idle with a request waiting: wait (the lock released) for the requests sent
    /// with it - until none has come for admit_quiet_ns, admit_cap_ns since the first or --parallel N wait.
    fn gatherArrivals(h: *Host) void {
        if (h.admit_quiet_ns <= 0) return;
        const t_first = h.now();
        var t_last = t_first;
        var seen = h.queue.items.len;
        while (!h.closing and h.queue.items.len < h.parallel) {
            h.unlock();
            std.Io.sleep(h.io, .fromMicroseconds(200), .awake) catch {};
            h.lock();
            const t = h.now();
            if (h.queue.items.len != seen) {
                seen = h.queue.items.len;
                t_last = t;
            }
            if (t - t_last >= h.admit_quiet_ns or t - t_first >= h.admit_cap_ns) break;
        }
        if (h.boot.multi.?.pf.on and h.queue.items.len > 0) {
            var lo: i96 = h.queue.items[0].submitted;
            var hi: i96 = lo;
            for (h.queue.items) |j| {
                lo = @min(lo, j.submitted);
                hi = @max(hi, j.submitted);
            }
            std.log.info("glm53: {d} requests admitted together (arrivals {d:.2} ms apart, waited {d:.2} ms)", .{ h.queue.items.len, @as(f64, @floatFromInt(hi - lo)) / 1e6, @as(f64, @floatFromInt(h.now() - t_first)) / 1e6 });
        }
    }

    fn setStreaming(h: *Host, n: usize) void {
        h.setBusy(n > 0); // each call follows an admission or a step: progress
        h.lock();
        h.streaming = @intCast(n);
        h.unlock();
    }

    /// Rank 0: one waiting request admitted (Glm53Engine.generate's clamps, SlotReuse.choose, ADMIT to every rank);
    /// a request the engine refuses fails alone.
    fn admitJob(h: *Host, job: Job) void {
        h.admitOne(job) catch |e| {
            std.log.err("glm53: request {d} not admitted: {t}", .{ job.id, e });
            emit(job, .{ .prefilled = 0 });
            emit(job, .{ .finished = .{ .reason = .failed, .message = words(e) } });
        };
    }

    fn admitOne(h: *Host, job: Job) !void {
        const mu = h.boot.multi.?;
        const r = job.request;
        const sess = &h.session;
        const max_tokens = try sess.clampTokens(r.prompt.len, r.max_tokens);
        const s = try samplingOf(r);
        const draft = r.drafts;
        var flags: u32 = if (draft) flag_draft else 0;
        if (s == null) flags |= flag_greedy;
        if (h.flush_file) |path| if (mu.store != null and takeFlush(path)) {
            flags |= flag_flush;
            try mu.flushReuse(); // rank 0 forgets before it chooses; the others at the admission
        };
        const c = try mu.choose(r.prompt, draft);
        defer c.deinit(h.gpa);
        const sid = mu.next_sid;
        const lv = try h.gpa.create(Live);
        errdefer h.gpa.destroy(lv);
        lv.* = .{ .h = h, .job = job, .started = job.submitted, .begin = @intCast(c.begin), .sid = sid, .sampling = s, .draft = draft };
        // the message's number is taken when it goes out (below), as a --parallel 1 request's
        var head: Head = .{ .kind = @intFromEnum(Kind.admit), .seq = h.mseq + 1, .max_tokens = @intCast(max_tokens), .flags = flags, .prompt_len = @intCast(r.prompt.len), .begin = @intCast(c.begin), .n_stops = @intCast(c.stops.len), .n_keeps = @intCast(c.keeps.len), .n_eos = @intCast(r.eos.len), .key = sid, .at = @intCast(c.slot), .pad = if (c.src) |x| @intCast(x + 1) else 0 };
        if (s) |x| {
            head.seed = x.seed;
            head.temperature = x.temperature;
            head.top_k = @intCast(x.top_k);
            head.top_p = x.top_p;
            head.min_p = x.min_p;
        }
        const n_words = r.prompt.len + c.stops.len + c.keeps.len + r.eos.len;
        const payload = try h.gpa.alloc(u32, n_words);
        defer h.gpa.free(payload);
        @memcpy(payload[0..r.prompt.len], r.prompt);
        for (c.stops, 0..) |p, i| payload[r.prompt.len + i] = @intCast(p);
        @memcpy(payload[r.prompt.len + c.stops.len ..][0..c.keeps.len], c.keeps);
        @memcpy(payload[r.prompt.len + c.stops.len + c.keeps.len ..], r.eos);
        h.mseq += 1;
        try h.sendOne(&head, payload);
        _ = mu.admit(.{ .sid = sid, .slot = c.slot, .mode = if (draft) glm.multi.modeOf(h.boot.mode) else .serial, .sampling = s, .count = max_tokens, .prompt = r.prompt, .eos = r.eos, .begin = c.begin, .src = c.src, .stops = c.stops, .keeps = c.keeps, .flush = false, .hook = .{ .ctx = lv, .tokens = Live.tokens } }) catch |e| {
            mu.broken = h.boot.o.world > 1; // the followers took the admission
            return e;
        };
        try h.lives.append(h.gpa, lv);
    }

    fn linkFill(ctx: *anyopaque, sid: u32) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.mseq += 1;
        var head: Head = .{ .kind = @intFromEnum(Kind.fill), .seq = h.mseq, .key = sid };
        try h.sendOne(&head, &.{});
        h.flushEmits(); // a prompt chunk runs next: the decoding replies hear their last round first
    }

    fn linkRound(ctx: *anyopaque, plan: *const glm.multi.Plan) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        var buf: [glm.multi.plan_words]u32 = undefined;
        const w = try plan.encode(&buf);
        h.mseq += 1;
        var head: Head = .{ .kind = @intFromEnum(Kind.round), .seq = h.mseq, .at = @intCast(w.len) };
        try h.sendOne(&head, w);
    }

    /// Rank 0's command and payload to ranks 1.. in one write (a round's message must not wait on the previous one's
    /// acknowledgement).
    fn sendOne(h: *Host, head: *Head, payload: []const u32) !void {
        if (h.boot.o.world == 1) return;
        head.sum = sealOf(head, payload);
        // no memory for the one buffer: the same bytes in two writes (the message's number is already taken, so a
        // failure here would leave rank 0 a number ahead of the followers)
        const bytes = h.gpa.alloc(u8, @sizeOf(Head) + payload.len * 4) catch return h.send(head, payload);
        defer h.gpa.free(bytes);
        @memcpy(bytes[0..@sizeOf(Head)], std.mem.asBytes(head));
        @memcpy(bytes[@sizeOf(Head)..], std.mem.sliceAsBytes(payload));
        try h.boot.rdv.broadcast(bytes);
    }

    fn liveOf(h: *Host, sid: u32) ?usize {
        for (h.lives.items, 0..) |lv, i| if (lv.sid == sid) return i;
        return null;
    }

    /// Rank 0: the streams that finished - FINISH to every rank, the replies' last events.
    fn endDone(h: *Host) !void {
        const mu = h.boot.multi.?;
        var buf: [glm.multi.max_streams]u32 = undefined;
        const done = mu.doneList(&buf);
        if (done.len == 0) return;
        h.mseq += 1;
        var head: Head = .{ .kind = @intFromEnum(Kind.finish), .seq = h.mseq, .at = @intCast(done.len) };
        try h.sendOne(&head, done);
        for (done) |sid| {
            const st = mu.find(sid) orelse continue;
            if (h.liveOf(sid)) |i| {
                const lv = h.lives.swapRemove(i);
                h.endStream(lv, st);
            }
            mu.finish(sid);
        }
    }

    /// Rank 0: the streams whose client left while their prompt still fills end now, before the next fill step - a
    /// decoding stream hears its cancel in its hook after each round, a filling one only at its first token, so a long
    /// prompt would otherwise fill to the end for no one. FINISH tells every rank, as for any finished stream.
    fn endCancelledFills(h: *Host) !void {
        const mu = h.boot.multi.?;
        var any = false;
        for (mu.filling.items) |st| {
            if (st.done) continue;
            const i = h.liveOf(st.sid) orelse continue;
            const lv = h.lives.items[i];
            if (!h.cancelled(lv.job.id)) continue;
            lv.cancel_hit = true;
            st.stopped = true;
            st.done = true;
            any = true;
        }
        if (any) try h.endDone();
    }

    /// A finished stream's reply: its reason and stats (the dump when TF_GLM53_DUMP), then the Live goes.
    fn endStream(h: *Host, lv: *Live, st: *const glm.multi.Stream) void {
        defer {
            lv.emitted.deinit(h.gpa);
            h.gpa.destroy(lv);
        }
        const reason: api.Reason = if (lv.cancel_hit) .cancelled else if (st.eos_hit or lv.stop_hit) .stop else .length;
        if (lv.first == null) emit(lv.job, .{ .prefilled = lv.begin });
        lv.flush();
        logMulti(lv, st, reason);
        h.lock();
        h.last_rate = st.tokS();
        if (st.prefill_s > 0) h.last_prefill_rate = @as(f64, @floatFromInt(st.prompt.len - st.begin)) / st.prefill_s;
        var i: usize = 0;
        while (i < h.cancels.items.len) {
            if (h.cancels.items[i] == lv.job.id) _ = h.cancels.swapRemove(i) else i += 1;
        }
        h.unlock();
        if (h.dump) |path| {
            const rs: glm.runner.Stats = .{ .prompt_len = st.prompt.len, .prefill_s = st.prefill_s, .decode_s = st.decodeS(), .rounds = st.rounds, .drafted = st.drafted, .accepted = st.accepted_n, .begin = st.begin, .eos = st.eos_hit, .stopped = st.stopped };
            const oc: Outcome = .{ .stats = rs, .kept = &.{}, .learned = 0, .sampling = lv.sampling, .draft = lv.draft, .begin = st.begin };
            h.writeDump(path, lv.job, lv, oc, st.out.items) catch |e| std.log.warn("glm53: TF_GLM53_DUMP {s}: {t}", .{ path, e });
        }
        emit(lv.job, .{ .finished = .{ .reason = reason, .stats = .{ .rounds = st.rounds, .drafted = st.drafted, .accepted = st.accepted_n, .min_rows = @intCast(@max(st.min_rows, 1)), .prefill_seconds = st.prefill_s } } });
    }

    /// Rank 0's line for a finished --parallel stream, as the Python server's `done req-...`: the engine's own tok/s
    /// (tokens after the first over its decode time), TTFT from submission, and per round on rank 0 the round's time
    /// (applyRound), the host time before it since the previous round ended (hooks, loop, admissions, fills, the
    /// message) and the graph replays / eager runs in its rounds.
    fn logMulti(lv: *const Live, st: *const glm.multi.Stream, reason: api.Reason) void {
        if (@import("builtin").is_test) return;
        const ttft: f64 = if (lv.first) |f| @as(f64, @floatFromInt(f - lv.started)) / 1e9 else 0;
        const rounds_f: f64 = @floatFromInt(@max(st.rounds, 1));
        const round_ms = @as(f64, @floatFromInt(st.round_ns)) / 1e6 / rounds_f;
        const gap_ms = @as(f64, @floatFromInt(st.gap_ns)) / 1e6 / rounds_f;
        std.log.info("glm53: done req-{d} prompt={d} cached={d} tokens={d} finish={t} tok/s={d:.1} ttft={d:.2}s prefill={d:.2}s decode={d:.2}s rounds={d} accepted={d}/{d} round_ms={d:.1} gap_ms={d:.2} graphs={d} eager={d} sampled={s} sid={d}", .{ lv.job.id, st.prompt.len, st.begin, st.out.items.len, reason, st.tokS(), ttft, st.prefill_s, st.decodeS(), st.rounds, st.accepted_n, st.drafted, round_ms, gap_ms, st.graph_runs, st.eager_runs, if (st.sampling != null) "true" else "false", lv.sid });
    }

    /// Rank 0 after a failed round (multi.drop) or at shutdown: every live request ends with `reason`; the decoder
    /// forgets its streams (a failed round leaves the ranks out of step: the decoder refuses further work).
    fn endAll(h: *Host, reason: api.Reason, message: []const u8) void {
        const mu = h.boot.multi.?;
        for (h.lives.items) |lv| {
            if (lv.first == null) emit(lv.job, .{ .prefilled = 0 });
            lv.flush();
            emit(lv.job, .{ .finished = .{ .reason = reason, .message = message } });
            lv.emitted.deinit(h.gpa);
            h.gpa.destroy(lv);
        }
        h.lives.clearRetainingCapacity();
        mu.dropAll();
    }

    // ----------------------------------------------------------------------------------------- the followers ---

    /// Ranks 1..: Glm53Engine.follow - every command rank 0 sends, until it says goodbye or its connection closes.
    pub fn follow(h: *Host) !void {
        std.log.info("glm53 rank {d}: following rank 0", .{h.rank});
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(h.gpa);
        while (true) {
            var head: Head = undefined;
            if (!try h.boot.rdv.awaitBroadcast(std.mem.asBytes(&head))) return; // rank 0 has gone
            if (head.magic != magic) return error.ProtocolOutOfStep;
            const kind = std.enums.fromInt(Kind, head.kind) orelse return error.ProtocolOutOfStep;
            switch (kind) {
                .bye => return h.verify(&head, &.{}),
                .request => try h.followRequest(&head, &out),
                .load => {
                    const ids = try h.recvWords(head.at);
                    defer h.gpa.free(ids);
                    try h.verify(&head, ids);
                    _ = try h.learnLoad(head.key, ids);
                },
                .persist => {
                    const ids = try h.recvWords(head.at);
                    defer h.gpa.free(ids);
                    try h.verify(&head, ids);
                    _ = try h.learnPersist(head.key, ids);
                },
                .forget => {
                    try h.verify(&head, &.{});
                    h.learnForget(head.key);
                },
                .admit, .fill, .round, .finish => try h.followMulti(&head),
            }
        }
    }

    /// Ranks 1..: one of rank 0's --parallel messages, applied in order (multi._apply).
    fn followMulti(h: *Host, head: *const Head) !void {
        const mu = h.boot.multi orelse return error.ProtocolOutOfStep;
        h.beat();
        defer h.setBusy(mu.live() > 0);
        h.mseq += 1;
        if (head.seq != h.mseq) {
            std.log.err("glm53 rank {d}: expected rank 0's message {d}, received number {d}: the ranks are out of step; restart all ranks", .{ h.rank, h.mseq, head.seq });
            return error.ProtocolOutOfStep;
        }
        switch (std.enums.fromInt(Kind, head.kind).?) {
            .admit => {
                const n = @as(usize, head.prompt_len) + head.n_stops + head.n_keeps + head.n_eos;
                const ws = try h.recvWords(n);
                defer h.gpa.free(ws);
                try h.verify(head, ws);
                const prompt = ws[0..head.prompt_len];
                const stops = try h.gpa.alloc(usize, head.n_stops);
                defer h.gpa.free(stops);
                for (stops, ws[head.prompt_len..][0..head.n_stops]) |*d, v| d.* = v;
                const keeps = ws[head.prompt_len + head.n_stops ..][0..head.n_keeps];
                const eos = ws[head.prompt_len + head.n_stops + head.n_keeps ..][0..head.n_eos];
                const s: ?smp.Sampling = if (head.flags & flag_greedy != 0) null else .{ .seed = head.seed, .temperature = head.temperature, .top_k = head.top_k, .top_p = head.top_p, .min_p = head.min_p };
                _ = try mu.admit(.{ .sid = @intCast(head.key), .slot = head.at, .mode = if (head.flags & flag_draft != 0) glm.multi.modeOf(h.boot.mode) else .serial, .sampling = s, .count = head.max_tokens, .prompt = prompt, .eos = eos, .begin = head.begin, .src = if (head.pad > 0) head.pad - 1 else null, .stops = stops, .keeps = keeps, .flush = head.flags & flag_flush != 0 });
            },
            .fill => {
                try h.verify(head, &.{});
                _ = try mu.applyFill(@intCast(head.key));
            },
            .round => {
                const ws = try h.recvWords(head.at);
                defer h.gpa.free(ws);
                try h.verify(head, ws);
                const plan = try glm.multi.Plan.decode(ws);
                var res: [glm.multi.max_streams]glm.multi.Result = undefined;
                var ds: [glm.multi.max_streams]usize = undefined;
                // TF_GLM53_ROUND_PROFILE: a follower's wait for this round's message since its previous round ended
                const t_in = mu.now();
                if (mu.t_round_end > 0 and t_in > mu.t_round_end and t_in - mu.t_round_end < std.time.ns_per_s) mu.profWait(@intCast(t_in - mu.t_round_end));
                try mu.applyRound(&plan, &res, &ds);
                mu.t_round_end = mu.now();
            },
            .finish => {
                const ws = try h.recvWords(head.at);
                defer h.gpa.free(ws);
                try h.verify(head, ws);
                for (ws) |sid| {
                    // the follower's side of rank 0's done line (its sid=): where the stream stood when rank 0 ended it
                    if (mu.find(sid)) |st| std.log.info("glm53 rank {d}: stream {d} finished by rank 0 ({s} at position {d}, {d} rounds)", .{ h.rank, sid, if (st.started) "decoding" else "filling", st.P, st.rounds });
                    mu.finish(sid);
                }
            },
            else => return error.ProtocolOutOfStep,
        }
    }

    /// Ranks 1..: rank 0's message checks against its seal, else the ranks are out of step (a misread length, a
    /// message lost or doubled) and this rank stops loudly rather than run the wrong collectives.
    fn verify(h: *Host, head: *const Head, payload: []const u32) !void {
        if (sealOf(head, payload) == head.sum) return;
        std.log.err("glm53 rank {d}: rank 0's message (kind {d}, number {d}) does not match its seal: the ranks are out of step; restart all ranks", .{ h.rank, head.kind, head.seq });
        return error.ProtocolOutOfStep;
    }

    fn recvWords(h: *Host, n: usize) ![]u32 {
        const w = try h.gpa.alloc(u32, n);
        errdefer h.gpa.free(w);
        if (n > 0) try h.boot.rdv.broadcast(std.mem.sliceAsBytes(w));
        return w;
    }

    fn followRequest(h: *Host, head: *const Head, out: *std.ArrayList(u32)) !void {
        const n = @as(usize, head.prompt_len) + head.n_stops + head.n_keeps + head.n_eos;
        const words_ = try h.recvWords(n);
        defer h.gpa.free(words_);
        try h.verify(head, words_);
        h.seq += 1;
        if (head.seq != h.seq) {
            std.log.err("glm53 rank {d}: expected rank 0's request {d}, received number {d}: the ranks are out of step; restart all ranks", .{ h.rank, h.seq, head.seq });
            return error.ProtocolOutOfStep;
        }
        h.setBusy(true);
        defer h.setBusy(false);
        const prompt = words_[0..head.prompt_len];
        const stops = try h.gpa.alloc(usize, head.n_stops);
        defer h.gpa.free(stops);
        for (stops, words_[head.prompt_len..][0..head.n_stops]) |*d, v| d.* = v;
        const keeps = words_[head.prompt_len + head.n_stops ..][0..head.n_keeps];
        const eos = words_[head.prompt_len + head.n_stops + head.n_keeps ..][0..head.n_eos];
        const s: ?smp.Sampling = if (head.flags & flag_greedy != 0) null else .{ .seed = head.seed, .temperature = head.temperature, .top_k = head.top_k, .top_p = head.top_p, .min_p = head.min_p };
        // a request that fails here fails on rank 0 too, at the same point (the same checks on the same arguments), and
        // rank 0 serves on: so does this rank - leaving would strand rank 0's next request in its collectives
        const done = h.session.exec(.{ .prompt = prompt, .max_tokens = head.max_tokens, .sampling = s, .draft = head.flags & flag_draft != 0, .eos = eos, .begin = head.begin, .stops = stops, .keeps = keeps, .flags = head.flags & flag_flush, .hooks = .{ .ctx = h, .tokens = followerBeat, .stop = followerChunk }, .prompt_votes = h.prompt_votes }, out) catch |e| {
            if (e == error.PromptStopped) {
                std.log.info("glm53 rank {d}: request {d} ended: stopped by rank 0 during the prompt", .{ h.rank, head.seq });
                return;
            }
            std.log.err("glm53 rank {d}: request {d} failed: {t} (following on, as rank 0 serves on)", .{ h.rank, head.seq, e });
            return;
        };
        h.gpa.free(done.kept);
        // the follower's side of rank 0's done line (its seq=): the vote ends a cancelled or stop-string reply on every rank
        std.log.info("glm53 rank {d}: request {d} ended: {s} tokens={d} rounds={d}", .{ h.rank, head.seq, if (done.stats.stopped) "stopped by rank 0" else if (done.stats.eos) "eos" else "length", out.items.len, done.stats.rounds });
        h.afterRequest();
    }

    /// After a request on every rank: nothing to do but rank 0's learning (it sends the commands).
    fn afterRequest(h: *Host) void {
        if (h.rank != 0 or h.learned == null) return;
        h.persistShared() catch |e| std.log.warn("glm53: --learn: a shared prompt state was not learned ({t})", .{e});
    }

    // ------------------------------------------------------------------------------------------------ --learn ---

    fn dir(h: *const Host) []const u8 {
        return h.learned.?.dir;
    }

    /// Rank 0: the longest learned state the prompt resumes past every kept one, read back on every rank (each its
    /// shard) and kept as a shared state - or forgotten everywhere when any rank cannot read it. Returns its tokens
    /// (0: none).
    fn loadLearned(h: *Host, prompt: []const u32) !usize {
        const pr = h.session.reuse.?;
        const m = &h.learned.?;
        const pts = try h.session.plan.points(h.gpa, prompt);
        defer h.gpa.free(pts);
        const starts = try h.gpa.alloc(u32, pts.len);
        defer h.gpa.free(starts);
        for (pts, starts) |p, *x| x.* = @intCast(p);
        const in_memory = if (pr.store.best(prompt, pts, 0)) |e| e.n() else 0;
        const best = m.best(prompt, starts, true, @intCast(in_memory)) orelse return 0;
        const key = best.key;
        const at = best.at;
        var head: Head = .{ .kind = @intFromEnum(Kind.load), .key = key, .at = at };
        try h.send(&head, prompt[0..at]);
        if (try h.learnLoad(key, prompt[0..at])) {
            m.touch(key);
            return at;
        }
        var forget: Head = .{ .kind = @intFromEnum(Kind.forget), .key = key };
        try h.send(&forget, &.{});
        h.learnForget(key);
        return 0;
    }

    /// Every rank: its shard of learned state `key` read back; kept (on every rank alike) only when every rank read
    /// its own.
    fn learnLoad(h: *Host, key: u64, ids: []const u32) !bool {
        const pr = h.session.reuse orelse return false;
        var buf: [1200]u8 = undefined;
        const file = try eng.Session.statePath(&buf, h.dir(), key, h.rank);
        const got = h.session.readState(ids, file, key, h.identity) catch |e| blk: {
            std.log.warn("glm53 rank {d}: learned state {x:0>16} not read ({t})", .{ h.rank, key, e });
            break :blk null;
        };
        const all = try h.boot.rdv.agree(got != null);
        if (!all) {
            if (got) |e| pr.store.destroy(e);
            return false;
        }
        if (!pr.store.remember(got.?, &.{})) {
            pr.store.destroy(got.?);
            return false;
        }
        return true;
    }

    /// Rank 0: every shared kept state not learned yet written to disk (each rank its shard), within --learn-gib.
    fn persistShared(h: *Host) !void {
        const pr = h.session.reuse orelse return;
        const m = &h.learned.?;
        var i: usize = 0;
        while (i < pr.store.entries.items.len) : (i += 1) {
            const e = pr.store.entries.items[i];
            if (!e.shared or e.learned) continue;
            const key = Imprint.keyOf(e.ids);
            if (m.has(key)) {
                e.learned = true;
                continue;
            }
            if (!pr.store.usable(e)) continue;
            const bytes = try h.stateBytes(e.n());
            while (!m.fits(bytes)) {
                const v = m.victim() orelse return; // nothing more to give
                var forget: Head = .{ .kind = @intFromEnum(Kind.forget), .key = v };
                try h.send(&forget, &.{});
                h.learnForget(v);
            }
            var head: Head = .{ .kind = @intFromEnum(Kind.persist), .key = key, .at = @intCast(e.n()) };
            const ids = try h.gpa.dupe(u32, e.ids);
            defer h.gpa.free(ids);
            try h.send(&head, ids);
            _ = try h.learnPersist(key, ids);
        }
    }

    /// A learned state's file bytes on a rank (every rank the same: its row views are sized alike).
    fn stateBytes(h: *Host, n: usize) !u64 {
        const pr = h.session.reuse.?;
        const vs = try pr.views(n);
        defer h.gpa.free(vs);
        var total: u64 = 0;
        for (vs) |v| total += v.row * v.rows;
        return @sizeOf(eng.Session.Head) + vs.len * 16 + total + h.boot.cfg.hidden * 2;
    }

    /// Every rank: its shard of kept state `ids` written; recorded in the index (every rank alike) only when every
    /// rank wrote its own, else every rank's file is removed.
    fn learnPersist(h: *Host, key: u64, ids: []const u32) !bool {
        const pr = h.session.reuse orelse return false;
        var buf: [1200]u8 = undefined;
        const file = try eng.Session.statePath(&buf, h.dir(), key, h.rank);
        var bytes: u64 = 0;
        const e = pr.store.named(ids, ids.len);
        if (e) |x| {
            bytes = h.session.writeState(x, file, key, h.identity) catch |err| blk: {
                std.log.warn("glm53 rank {d}: learned state {x:0>16} not written ({t})", .{ h.rank, key, err });
                break :blk 0;
            };
        }
        const all = try h.boot.rdv.agree(bytes > 0);
        if (!all) {
            _ = std.c.unlink(file);
            return false;
        }
        const pts = try h.session.plan.points(h.gpa, ids);
        defer h.gpa.free(pts);
        const starts = try h.gpa.alloc(u32, pts.len);
        defer h.gpa.free(starts);
        for (pts, starts) |p, *x| x.* = @intCast(p);
        try h.learned.?.add(key, @intCast(ids.len), ids, starts, bytes);
        e.?.learned = true;
        if (h.rank == 0) std.log.info("glm53: learned a shared prompt state of {d} tokens ({d:.2} GiB a rank)", .{ ids.len, @as(f64, @floatFromInt(bytes)) / (1 << 30) });
        return true;
    }

    /// Every rank: learned state `key` forgotten (its index record and this rank's file).
    fn learnForget(h: *Host, key: u64) void {
        const m = if (h.learned) |*x| x else return;
        m.remove(key) catch |e| std.log.warn("glm53: --learn index: {t}", .{e});
        var buf: [1200]u8 = undefined;
        const file = eng.Session.statePath(&buf, m.dir, key, h.rank) catch return;
        _ = std.c.unlink(file);
    }

    /// Learned states' identity: the checkpoint (config and weight index), the split and the served settings, the
    /// kernel pack, the driver, and the probe's prompt-pass bits on every rank.
    fn identityOf(h: *Host) !u64 {
        const b = h.boot;
        const o = b.o;
        var x = std.hash.Wyhash.init(0x67353362);
        for ([_][]const u8{ "config.json", "model.safetensors.index.json" }) |name| {
            const path = try std.fs.path.join(h.gpa, &.{ o.model, name });
            defer h.gpa.free(path);
            if (std.Io.Dir.cwd().readFileAlloc(h.io, path, h.gpa, .limited(64 << 20))) |bytes| {
                defer h.gpa.free(bytes);
                x.update(bytes);
            } else |_| {}
        }
        const aot = try std.fs.path.join(h.gpa, &.{ o.aot, "aot.json" });
        defer h.gpa.free(aot);
        if (std.Io.Dir.cwd().readFileAlloc(h.io, aot, h.gpa, .limited(64 << 20))) |bytes| {
            defer h.gpa.free(bytes);
            x.update(bytes);
        } else |_| {}
        const shape = [_]u64{ o.world, b.dcp, o.k, o.draft_vocab, o.mtp_reuse, o.prompt_rows, o.prompt_rows_short, @intFromBool(o.prompt_sp), o.unpack_mb, @intFromBool(o.mtp_dense), @intCast(try b.driver.version()), @intCast(b.nccl_version), o.reuse.gap };
        x.update(std.mem.asBytes(&shape));
        x.update(o.experts);
        x.update(o.pe_det);
        const probe = try h.session.probe();
        x.update(std.mem.asBytes(&probe));
        return x.final();
    }

    fn writeDump(h: *Host, path: []const u8, job: Job, live: *const Live, oc: Outcome, tokens: []const u32) !void {
        var w: std.Io.Writer.Allocating = .init(h.gpa);
        defer w.deinit();
        const o = &w.writer;
        const st = oc.stats;
        try o.print("{{\"id\": {d}, \"prompt_ids\": [", .{job.id});
        for (job.request.prompt, 0..) |t, i| try o.print("{s}{d}", .{ if (i > 0) "," else "", t });
        try o.writeAll("], \"tokens\": [");
        for (tokens, 0..) |t, i| try o.print("{s}{d}", .{ if (i > 0) "," else "", t });
        const finish = if (live.cancel_hit) "cancelled" else if (st.eos or live.stop_hit) "stop" else "length";
        try o.print("], \"finish\": \"{s}\", \"begin\": {d}, \"learned\": {d}, \"kept\": [", .{ finish, oc.begin, oc.learned });
        for (oc.kept, 0..) |n, i| try o.print("{s}{d}", .{ if (i > 0) "," else "", n });
        try o.writeAll("], \"sampling\": ");
        if (oc.sampling) |s| {
            try o.print("{{\"greedy\": false, \"seed\": {d}, \"temperature\": {d}, \"top_k\": {d}, \"top_p\": {d}, \"min_p\": {d}}}", .{ s.seed, s.temperature, s.top_k, s.top_p, s.min_p });
        } else try o.writeAll("{\"greedy\": true}");
        try o.writeAll(", \"eos\": [");
        for (job.request.eos, 0..) |t, i| try o.print("{s}{d}", .{ if (i > 0) "," else "", t });
        const ttft: f64 = if (live.first) |f| @as(f64, @floatFromInt(f - live.started)) / 1e9 else 0;
        try o.print("], \"drafts\": {s}, \"k\": {d}, \"ttft_s\": {d:.4}, \"prefill_s\": {d:.4}, \"decode_s\": {d:.4}, \"tok_s\": {d:.3}, \"rounds\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"replay\": {s}}}\n", .{ if (oc.draft) "true" else "false", if (oc.draft) h.boot.o.k else 0, ttft, st.prefill_s, st.decode_s, st.tokS(tokens.len), st.rounds, st.drafted, st.accepted, if (st.begin == job.request.prompt.len) "true" else "false" });
        var pbuf: [1100]u8 = undefined;
        const z = try std.fmt.bufPrintSentinel(&pbuf, "{s}", .{path}, 0);
        const fd = std.c.open(z, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.DumpOpen;
        defer _ = std.c.close(fd);
        try api.prompt_imprint.writeAll(fd, w.written());
    }
};

/// A request's keyed sampling (null: greedy). A top_k past the candidates a rank sends a row (256 + MARGIN) is
/// refused here, before any rank hears of the request: every rank would fail it at its first sampled window.
fn samplingOf(r: *const api.Request) !?smp.Sampling {
    const x = r.sampling orelse return null;
    if (!(x.temperature > 0)) return null;
    const s: smp.Sampling = .{ .seed = x.seed & 0x7FFFFFFFFFFFFFFF, .temperature = x.temperature, .top_k = x.top_k, .top_p = x.top_p, .min_p = x.min_p };
    if (s.candidates() > smp.max_candidates) return error.TopKTooWide;
    return s;
}

/// The watchdog's exit: at once, without libc's exit handlers. With a rank stuck or gone the CUDA and NCCL teardown
/// those handlers run waits on the very collective that will never finish - the process logged its exit and stayed
/// (seen on the cluster: every pause and kill case with a request in hand).
fn leaveNow(code: u8) noreturn {
    std.c._exit(code);
}

/// The words for a request the engine refuses.
fn words(err: anyerror) []const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt does not fit this server's --context",
        error.ContextTooSmall => "the prompt and max_tokens do not fit this server's --context",
        error.ReuseOutOfStep => "the ranks' kept prompt states are out of step (restart the server)",
        error.OutOfStep, error.RanksOutOfStep => "the ranks are out of step after a failed round (restart the server)",
        error.NoFreeSlot => "no free stream slot (the server admits at most --parallel streams)",
        error.TopKTooWide => "top_k above 256 is not supported by this engine: send top_k 256 or less (0: top_p alone)",
        else => @errorName(err),
    };
}

fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return words(err);
}

fn closeFn(ctx: *anyopaque) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    // the watchdog first: the followers leave at rank 0's goodbye, which must not read as a rank that has gone
    h.stop_watch.store(true, .release);
    if (h.watcher) |t| t.join();
    if (h.rank == 0) {
        h.lock();
        h.closing = true;
        h.wake.broadcast(h.io);
        h.unlock();
        if (h.thread) |t| t.join();
        var bye: Head = .{ .kind = @intFromEnum(Kind.bye) };
        h.send(&bye, &.{}) catch {};
    }
    h.queue.deinit(h.gpa);
    h.cancels.deinit(h.gpa);
    h.out.deinit(h.gpa);
    for (h.lives.items) |lv| {
        lv.emitted.deinit(h.gpa);
        h.gpa.destroy(lv);
    }
    h.lives.deinit(h.gpa);
    if (h.learned) |*m| m.deinit();
    h.session.deinit();
    h.boot.deinit();
    h.gpa.free(h.startup);
    h.gpa.destroy(h);
}

/// The stall limit in seconds when TF_GLM53_STALL_S names none: 900 (the Python server's), or longer where the
/// context is - a --parallel 1 prompt pass makes no progress the watchdog sees until its first token, and a full 1M
/// prompt took 2,749 s (304 tok/s, 2026-10-08): 5.4 ms a context token, 5400 s at 1M, 900 s up to 166K.
fn defaultStallS(context: usize) f64 {
    return @max(900.0, @as(f64, @floatFromInt(context)) * 0.0054);
}

fn followFn(ctx: *anyopaque) anyerror!void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    return h.follow();
}

/// The model's window (config.json's max_position_embeddings), 0 when it names none.
fn modelContext(a: Allocator, io: std.Io, dir_: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir_, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const limit = doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// The GLM-5.3 engine for `o.dir` on this rank (TF_GLM53_RANK): loads, warms and agrees with the other ranks; rank 0
/// returns the served engine, ranks 1..3 an engine whose `follow` mirrors rank 0 until it stops.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const native = modelContext(a, io, o.dir);
    var window: i64 = o.context orelse default_context;
    if (window == 0) window = native;
    if (window <= 0 or (native > 0 and window > native)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d}: a token count within this model's {d}-token window", .{ window, native });
        return null;
    }
    var opts = eng.Options.fromEnv(o.dir, @intCast(window)) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "glm53 settings: {t} (TF_GLM53_* variables)", .{e});
        return null;
    };
    // --parallel N (named) or TF_GLM53_PARALLEL: N concurrent streams (cuda_multi.zig); unnamed: one at a time
    if (o.lanes_fixed) opts.parallel = @max(1, o.lanes);
    if (opts.parallel > glm.multi.max_streams) {
        problem.* = try std.fmt.allocPrint(a, "--parallel {d}: at most {d} concurrent streams", .{ opts.parallel, glm.multi.max_streams });
        return null;
    }
    if (o.learn != null and opts.parallel > 1) {
        problem.* = "--learn keeps one stream's prompt states: serve --parallel 1 with it";
        return null;
    }
    if (o.learn != null and !opts.reuse.on) {
        problem.* = "--learn keeps prompt states that prompt reuse made: set TF_GLM53_PROMPT_REUSE=1 on every rank";
        return null;
    }
    const boot = eng.Boot.init(gpa, io, opts) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "glm53 rank {d}: the engine did not start ({t})", .{ opts.rank, e });
        return null;
    };
    var ok = false;
    defer if (!ok) boot.deinit();
    const h = try gpa.create(Host);
    defer if (!ok) gpa.destroy(h);
    h.* = .{ .gpa = gpa, .io = io, .boot = boot, .session = undefined, .rank = opts.rank, .parallel = opts.parallel };
    h.session = try eng.Session.init(gpa, boot);
    defer if (!ok) h.session.deinit();
    // --learn on every rank or none: the ranks agree, then the probe gives the identity
    const all_learn = try boot.rdv.agree(o.learn != null);
    const none_learn = try boot.rdv.agree(o.learn == null);
    if (!all_learn and !none_learn) {
        problem.* = "--learn: start every rank with it (or none)";
        return null;
    }
    const learning = all_learn;
    defer if (!ok) if (h.learned) |*m| m.deinit();
    if (learning) {
        h.identity = try h.identityOf();
        h.learned = try Imprint.open(gpa, o.learn.?, h.identity, @intFromFloat(o.learn_gib * (1 << 30)));
        if (h.rank == 0) std.log.info("glm53: learned prompt states in {s} ({d} known, {d} of {d} MiB)", .{ h.learned.?.dir, h.learned.?.metas.items.len, h.learned.?.total() >> 20, @as(u64, @intFromFloat(o.learn_gib * 1024)) });
    }
    h.flush_file = if (std.c.getenv("TF_GLM53_FLUSH_FILE")) |v| std.mem.span(v) else null;
    h.fail_file = if (std.c.getenv("TF_GLM53_FAIL_FILE")) |v| std.mem.span(v) else null;
    h.prompt_votes = if (std.c.getenv("TF_GLM53_PROMPT_VOTES")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    h.dump = if (std.c.getenv("TF_GLM53_DUMP")) |v| std.mem.span(v) else null;
    const defer_on = if (std.c.getenv("TF_GLM53_DEFER_EMITS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    h.defer_emits = opts.parallel > 1 and defer_on;
    const quiet_ms = if (std.c.getenv("TF_GLM53_ADMIT_QUIET_MS")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 5 else 5;
    const cap_ms = if (std.c.getenv("TF_GLM53_ADMIT_CAP_MS")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 30 else 30;
    h.admit_quiet_ns = if (opts.parallel > 1) @intFromFloat(@max(0, quiet_ms) * 1e6) else 0;
    h.admit_cap_ns = @intFromFloat(@max(0, cap_ms) * 1e6);
    h.startup = try std.fmt.allocPrint(gpa, "GLM-5.3 TP{d} rank 0 of {d}: context {d}, MTP drafts {d}, DCP {d}, {d} concurrent stream{s}, prompt reuse {s}{s}; loaded in {d:.0} s, {d} decode graphs", .{ opts.world, opts.world, opts.context, opts.k, boot.dcp, opts.parallel, if (opts.parallel > 1) "s" else "", if (opts.reuse.on) "on" else "off", if (learning) ", --learn" else "", boot.load_s, boot.prewarm_graphs });
    h.info_ = .{ .name = "glm_moe_dsa-cuda-tp", .lanes = @intCast(opts.parallel), .context_window = @intCast(opts.context), .startup = h.startup, .top_k_most = @intCast(smp.max_candidates - smp.margin) };
    const stall_default = defaultStallS(opts.context);
    const stall_s = if (std.c.getenv("TF_GLM53_STALL_S")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch stall_default else stall_default;
    h.stall_ns = @intFromFloat(@max(0, stall_s) * 1e9);
    std.log.info("glm53 rank {d}: stall limit {d:.0} s (TF_GLM53_STALL_S; 0: off), a rank that goes ends this one", .{ h.rank, @max(0, stall_s) });
    if (h.rank == 0) h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, Host.loop, .{h});
    ok = true;
    // a world of one has no rank to lose: its watchdog runs for the stall limit alone
    if (h.stall_ns > 0 or opts.world > 1) h.watcher = std.Thread.spawn(.{}, Host.watch, .{h}) catch |e| blk: {
        std.log.warn("glm53 rank {d}: no watchdog ({t})", .{ h.rank, e });
        break :blk null;
    };
    return .{ .engine = h.engine(), .close = closeFn, .ctx = h, .follow = if (h.rank == 0) null else followFn };
}

test "the stall limit is 900 s up to 166K of context and 5400 s at 1M" {
    try std.testing.expectEqual(@as(f64, 900), defaultStallS(32768));
    try std.testing.expectEqual(@as(f64, 900), defaultStallS(131072));
    try std.testing.expectApproxEqAbs(@as(f64, 5400), defaultStallS(1_000_000), 1e-6);
}

test "the header is a fixed little block" {
    try std.testing.expectEqual(@as(usize, 104), @sizeOf(Head));
}

test "a sealed message checks, a changed one does not" {
    var head: Head = .{ .kind = @intFromEnum(Kind.request), .seq = 7, .prompt_len = 3 };
    const words_ = [_]u32{ 1, 2, 3 };
    head.sum = sealOf(&head, &words_);
    try std.testing.expectEqual(head.sum, sealOf(&head, &words_));
    try std.testing.expect(head.sum != sealOf(&head, &[_]u32{ 1, 2, 4 }));
    var later = head;
    later.seq = 8;
    try std.testing.expect(head.sum != sealOf(&later, &words_));
}
