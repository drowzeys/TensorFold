//! Native CUDA Drama 3 text backbone (Qwen3.5-35B-A3B hybrid: 30 Gated DeltaNet + 10 gated full-attention layers,
//! 256-expert top-8 block-FP8 MoE with a gated shared expert); a request's recurrent and KV state is a Sequence.
//! run appends rows and returns every row's final-norm hidden state plus the last row's semantic logits (EOS, codes).
const std = @import("std");
const cuda = @import("cuda");

pub const config = @import("backbone_config.zig");
pub const weights = @import("backbone_weights.zig");
pub const kernels = @import("backbone_kernels.zig");
pub const gdn = @import("backbone_gdn.zig");
const scratch = @import("backbone_scratch.zig");
const expert_mask = @import("expert_mask.zig");
const compact = @import("compact.zig");

const H = config.hidden;
const top = config.top_k;
const Scratch = scratch.Scratch;
const tile = scratch.tile;
const gdnWorkspace = scratch.gdnWorkspace;

pub const Options = struct {
    context: usize = 8192,
    max_rows: usize = 128,
    /// --expert-mask: the routed experts each layer may pick; null: all of them, as the checkpoint routes.
    expert_mask: ?expert_mask.Mask = null,
};

/// Prefill is the chunked GDN algorithm (gdn), on 64-token chunks counted from the last reset or decode however runs
/// split the prompt, with normalized q/k in BF16; packed autoregressive decode keeps them in FP32.
/// A one-row prompt chunk is still prefill. Callers must select the phase, never infer it from row count.
pub const Phase = enum { prefill, decode };

pub const Rows = struct {
    // Borrowed views, valid until next run/reset/deinit. BF16 row-major [rows, hidden].
    hidden: cuda.DeviceBuffer,
    // F32 [4097] for the LAST row only: EOS first, then semantic_start..semantic_end.
    semantic_logits: cuda.DeviceBuffer,
    rows: usize,
};

/// Diagnostic per-layer capture points of `forward` (temporary; tf-drama-backbone-trace is the only tracer).
/// attention_input / attention_residual: the input RMSNorm's normalized output and its updated residual (layer 0:
/// the embedding). attention_output: the DeltaNet or attention out projection, before the second add-norm.
/// mlp_input / mlp_residual: the post-attention RMSNorm's normalized output and updated residual. mlp_output: the
/// complete MoE output (routed experts plus the gated shared expert). All BF16 [rows, hidden] scratch tensors.
pub const Stage = enum { attention_input, attention_residual, attention_output, mlp_input, mlp_residual, mlp_output };

/// One layer's routed experts as `runTraced`'s `routed` hook sees them: scratch addresses, in its format's forms.
pub const Routed = struct {
    rows: usize,
    ids: u64, // i32 [rows] the rows' tokens
    picks: u64, // i32 [rows][top_k] experts, descending probability
    weights: u64, // f32 [rows][top_k] routing weights after renormalization
    input: u64, // bf16 [rows][hidden] the MoE input (Scratch.normed)
    gate_up: u64 = 0, // FP8: bf16 [rows * top_k][gu_rows] each pair's gate | up, the SwiGLU input
    down: u64 = 0, // FP8: bf16 [rows * top_k][hidden] each pair's down output, scaled by its routing weight before the store
    act: u64 = 0, // NVFP4: bf16 [rows * top_k][expert_inner] each pair's SwiGLU, the gate-up epilogue's
    down_f32: u64 = 0, // NVFP4: f32 [rows * top_k][hidden] each pair's unweighted down output
};

/// Hashed by the trace driver so a capture names the exact forward it observed; unreferenced in production.
pub const trace_source = @embedFile("backbone.zig");

/// Diagnostic operator selector for `Model.replayOperator` (temporary; tf-drama-backbone-replay is the only caller).
/// attention: the layer's production `deltaNet` or `fullAttention`, by its kind. moe: the layer's production `moe`, on
/// the input's FP8 groups from `actQuant` (bit-identical to the ones the post-attention addNorm writes in `forward`).
pub const ReplayOperator = enum { attention, moe };

const conv_state_bytes = (config.conv_taps - 1) * config.conv_dim * 2; // bf16 [3][conv_dim] a DeltaNet layer
const rec_state_bytes = config.linear_value_heads * config.linear_dim * config.linear_dim * 4; // f32 [32][128][128]
const gdn_qkv_cache_bytes = gdn.chunk_rows * config.conv_dim * 2; // bf16 [64][conv_dim] the unfinished chunk's rows
const gdn_ba_cache_bytes = gdn.chunk_rows * 2 * config.linear_value_heads * 2; // bf16 [64][b 32 | a 32]
const gdn_state_bytes = 2 * rec_state_bytes + gdn_qkv_cache_bytes + gdn_ba_cache_bytes; // a DeltaNet layer's gdn.State

/// DeltaNet layer `slot`'s recurrent state in the state buffer at `base`: two FP32 banks, then the row caches.
fn gdnState(base: u64, slot: usize) gdn.State {
    const at = base + config.linear_layers * conv_state_bytes + slot * gdn_state_bytes;
    return .{
        .banks = .{ at, at + rec_state_bytes },
        .qkv_cache = at + 2 * rec_state_bytes,
        .ba_cache = at + 2 * rec_state_bytes + gdn_qkv_cache_bytes,
    };
}

/// A request's DeltaNet states, attention KV cache and position; a Model holds one and makes more on its weights.
pub const Sequence = struct {
    context: usize,
    state_buf: cuda.DeviceBuffer, // DeltaNet conv states, then per DeltaNet layer its gdn.State banks and row caches
    kv_buf: cuda.DeviceBuffer, // per attention layer: k cache then v cache, bf16 [context][kv_heads][head_dim]
    gdn_states: [config.linear_layers]gdn.State, // per DeltaNet layer: device addresses and chunk metadata
    pos: usize = 0,
    /// Set while a run's device work is in flight; a failed run leaves it set and only reset clears it.
    poisoned: bool = false,

    const state_bytes = config.linear_layers * (conv_state_bytes + gdn_state_bytes);

    /// The device bytes a sequence of `context` rows holds.
    pub fn bytes(context: usize) usize {
        return state_bytes + config.full_layers * 2 * kvBytes(context);
    }

    fn alloc(d: *const cuda.Driver, context: usize) !Sequence {
        var state_buf = try cuda.DeviceBuffer.alloc(d, state_bytes);
        errdefer state_buf.free();
        try state_buf.fill8(0, null);
        var gdn_states: [config.linear_layers]gdn.State = undefined;
        for (&gdn_states, 0..) |*st, slot| st.* = gdnState(state_buf.ptr, slot);
        const kv_buf = try cuda.DeviceBuffer.alloc(d, config.full_layers * 2 * kvBytes(context));
        return .{ .context = context, .state_buf = state_buf, .kv_buf = kv_buf, .gdn_states = gdn_states };
    }

    pub fn free(q: *Sequence) void {
        q.kv_buf.free();
        q.state_buf.free();
        q.* = undefined;
    }

    /// Clears the request: DeltaNet conv and recurrent states and row caches to zero, every layer's chunk metadata,
    /// position 0 (the KV cache is overwritten).
    pub fn reset(q: *Sequence, stream: cuda.Stream) !void {
        try q.state_buf.fill8(0, stream.handle);
        try stream.synchronize();
        for (&q.gdn_states) |*st| st.resetMetadata();
        q.pos = 0;
        q.poisoned = false;
    }

    fn kvBytes(context: usize) usize {
        return context * config.kv_heads * config.head_dim * 2;
    }

    /// The state after this sequence's first `rows` rows, copied out: DeltaNet conv and recurrent banks with their
    /// row caches and chunk metadata, and each attention layer's first `rows` KV rows.
    pub const Prefix = struct {
        rows: usize,
        state: cuda.DeviceBuffer,
        kv: cuda.DeviceBuffer, // per attention layer: k rows then v rows, bf16 [rows][kv_heads][head_dim]
        meta: [config.linear_layers]Meta,

        const Meta = struct { pending: usize, committed_bank: u1 };

        pub fn free(p: *Prefix) void {
            p.kv.free();
            p.state.free();
            p.* = undefined;
        }
    };

    /// Copies the state now (after `q.pos` rows) into new buffers, in `stream` order.
    pub fn savePrefix(q: *const Sequence, stream: cuda.Stream) !Prefix {
        const d = q.state_buf.d;
        const rows = q.pos;
        var state = try cuda.DeviceBuffer.alloc(d, state_bytes);
        errdefer state.free();
        var kv = try cuda.DeviceBuffer.alloc(d, config.full_layers * 2 * kvBytes(rows));
        errdefer kv.free();
        try state.copyFrom(0, q.state_buf.ptr, state_bytes, stream.handle);
        for (0..config.full_layers) |slot| {
            try kv.copyFrom((2 * slot) * kvBytes(rows), q.kCache(slot), kvBytes(rows), stream.handle);
            try kv.copyFrom((2 * slot + 1) * kvBytes(rows), q.vCache(slot), kvBytes(rows), stream.handle);
        }
        var p: Prefix = .{ .rows = rows, .state = state, .kv = kv, .meta = undefined };
        for (&p.meta, q.gdn_states) |*m, st| m.* = .{ .pending = st.pending, .committed_bank = st.committed_bank };
        return p;
    }

    /// Puts a saved prefix back: the sequence continues at row p.rows, as if it had just run them.
    pub fn loadPrefix(q: *Sequence, p: *const Prefix, stream: cuda.Stream) !void {
        if (p.rows > q.context) return error.DramaContextOverflow;
        try q.state_buf.copyFrom(0, p.state.ptr, state_bytes, stream.handle);
        for (0..config.full_layers) |slot| {
            try q.kv_buf.copyFrom((2 * slot) * kvBytes(q.context), p.kv.ptr + (2 * slot) * kvBytes(p.rows), kvBytes(p.rows), stream.handle);
            try q.kv_buf.copyFrom((2 * slot + 1) * kvBytes(q.context), p.kv.ptr + (2 * slot + 1) * kvBytes(p.rows), kvBytes(p.rows), stream.handle);
        }
        for (&q.gdn_states, p.meta) |*st, m| {
            st.pending = m.pending;
            st.committed_bank = m.committed_bank;
        }
        q.pos = p.rows;
        q.poisoned = false;
    }

    fn convState(q: *const Sequence, slot: usize) u64 {
        return q.state_buf.ptr + slot * conv_state_bytes;
    }

    fn kCache(q: *const Sequence, slot: usize) u64 {
        return q.kv_buf.ptr + (2 * slot) * kvBytes(q.context);
    }

    fn vCache(q: *const Sequence, slot: usize) u64 {
        return q.kv_buf.ptr + (2 * slot + 1) * kvBytes(q.context);
    }
};

pub const Model = struct {
    hidden: usize,
    semantic_start: u32,
    semantic_end: u32,
    eos: u32,

    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    options: Options,
    k: kernels.Kernels,
    w: weights.Weights,
    scratch_buf: cuda.DeviceBuffer,
    hidden_buf: cuda.DeviceBuffer, // bf16 [max_rows][H]
    logits_buf: cuda.DeviceBuffer, // f32 [4097], the last row's semantic logits
    tokens: cuda.HostBuffer, // pinned i32 [max_rows]
    keep_buf: ?cuda.DeviceBuffer, // u32 [layers][words] drama_topk's eligible experts (held, --expert-mask); null: all
    slot_buf: ?cuda.DeviceBuffer, // i32 [layers][experts] each original id's held slot, on a compact checkpoint
    s: Scratch,
    gdn_ws: gdn.Workspace, // placed in scratch_buf
    seq: Sequence, // the Model's own sequence: run, runTraced, reset and replayOperator use it

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *cuda.Context, checkpoint_dir: []const u8, kernel_dir: []const u8, options: Options) !Model {
        if (options.max_rows == 0 or options.context == 0 or options.context > 262144) return error.BadDramaOptions;
        if (try ctx.capability() < 89) return error.DramaNeedsFp8Mma; // the experts' e4m3 mma is sm_89+
        const d = ctx.d;
        const c = try config.Config.read(gpa, io, checkpoint_dir);
        var k = try kernels.Kernels.load(gpa, io, d, kernel_dir);
        errdefer k.unload();
        var w = blk: {
            var stream = try cuda.Stream.init(d, true);
            defer stream.deinit();
            break :blk try weights.load(gpa, io, d, stream, checkpoint_dir, c);
        };
        errdefer w.deinit();
        if (w.nvfp4Layers() > 0) try k.loadFp4(d, @intCast(try ctx.attribute(.multiprocessor_count)));
        if (w.denseFp8().layers > 0) try k.loadLane(d, try ctx.capability() / 10);
        return assemble(gpa, d, k, w, c, options);
    }

    /// Internal test seam, init minus its checks: valid options and config, format kernels loaded, weights outlive it.
    pub fn assemble(gpa: std.mem.Allocator, d: *const cuda.Driver, k: kernels.Kernels, w: weights.Weights, c: config.Config, options: Options) !Model { // owns k and w only once it returns
        const R = options.max_rows;
        const held = if (w.held) |*h| h else null;
        const mask = if (options.expert_mask) |*m| m else null;
        if (held != null and mask != null) if (compact.unheld(held.?, mask.?)) |x| {
            std.log.err("drama backbone: --expert-mask keeps expert {d} of layer {d}, which the checkpoint does not hold", .{ x.expert, x.layer });
            return error.DramaMaskNotHeld;
        };
        const tables = compact.tables(held, mask);
        var s, const scratch_bytes = Scratch.layout(R, scratch.extra(&k, &w, R));
        var scratch_buf = try cuda.DeviceBuffer.alloc(d, scratch_bytes);
        errdefer scratch_buf.free();
        try scratch_buf.fill8(0, null);
        s.place(scratch_buf.ptr);
        const gdn_ws = gdnWorkspace(R)[0].place(s.gdn);
        var seq = try Sequence.alloc(d, options.context);
        errdefer seq.free();
        var hidden_buf = try cuda.DeviceBuffer.alloc(d, R * H * 2);
        errdefer hidden_buf.free();
        var logits_buf = try cuda.DeviceBuffer.alloc(d, weights.head_cols * 4);
        errdefer logits_buf.free();
        var keep_buf = if (tables) |*t| try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&t.keep)) else null;
        errdefer if (keep_buf) |*b| b.free();
        var slot_buf = if (tables) |*t| if (t.slots) |*slots| try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(slots)) else null else null;
        errdefer if (slot_buf) |*b| b.free();
        const tokens = try cuda.HostBuffer.alloc(d, R * 4);
        return .{
            .hidden = H,
            .semantic_start = c.semantic_start,
            .semantic_end = c.semantic_end,
            .eos = c.eos,
            .gpa = gpa,
            .d = d,
            .options = options,
            .k = k,
            .w = w,
            .scratch_buf = scratch_buf,
            .hidden_buf = hidden_buf,
            .logits_buf = logits_buf,
            .tokens = tokens,
            .keep_buf = keep_buf,
            .slot_buf = slot_buf,
            .s = s,
            .gdn_ws = gdn_ws,
            .seq = seq,
        };
    }

    pub fn deinit(self: *Model) void {
        self.tokens.free();
        if (self.keep_buf) |*b| b.free();
        if (self.slot_buf) |*b| b.free();
        self.logits_buf.free();
        self.hidden_buf.free();
        self.seq.free();
        self.scratch_buf.free();
        self.w.deinit();
        self.k.unload();
        self.* = undefined;
    }

    /// Clears the Model's own sequence (`Sequence.reset`).
    pub fn reset(self: *Model, stream: cuda.Stream) !void {
        return self.seq.reset(stream);
    }

    /// Another sequence of the Model's context, sharing its weights and scratch.
    pub fn sequence(self: *const Model) !Sequence {
        return Sequence.alloc(self.d, self.options.context);
    }

    // Append contiguous rows to this request's state. Reject overflow before mutation.
    // extra is null for plain text; otherwise BF16 [tokens.len, hidden], codebook-1 embedding.
    // Semantic rows combine token + extra, scaled by 1/sqrt(2); ordinary rows do not.
    // tokens and extra may be released after run returns; run must complete its stream work.
    pub fn run(self: *Model, stream: cuda.Stream, phase: Phase, tokens: []const u32, extra: ?cuda.DeviceBuffer) !Rows {
        return self.advance(&.{&self.seq}, &.{self.logits_buf.ptr}, stream, phase, tokens, extra, {});
    }

    /// `run` on sequence `q` instead of the Model's own. The rows it returns are the same buffers every run writes.
    pub fn runOn(self: *Model, q: *Sequence, stream: cuda.Stream, phase: Phase, tokens: []const u32, extra: ?cuda.DeviceBuffer) !Rows {
        return self.advance(&.{q}, &.{self.logits_buf.ptr}, stream, phase, tokens, extra, {});
    }

    /// Readies `decodePair`, whose kernel `init` leaves out: images older than it lack it.
    pub fn loadPair(self: *Model) !void {
        try self.k.loadRows();
    }

    /// One decode of `token` on the own sequence and `other` as a batch of two; `other`'s logits go to `other_logits`.
    pub fn decodePair(self: *Model, other: *Sequence, stream: cuda.Stream, token: u32, extra: ?cuda.DeviceBuffer, other_logits: cuda.DeviceBuffer) !Rows {
        if (other == &self.seq) return error.DramaPairOfOneSequence;
        if (other_logits.len < weights.head_cols * 4) return error.DramaLogitsShort;
        return self.advance(&.{ &self.seq, other }, &.{ self.logits_buf.ptr, other_logits.ptr }, stream, .decode, &.{token}, extra, {});
    }

    /// `run` with stage captures. `trace` is `{}` (void: every capture site is compiled out, so `run` is unchanged)
    /// or a pointer to a tracer with `routed(t, stream, layer, r: Routed) !void`, this capture hook, or both:
    ///     capture(t, stream: cuda.Stream, layer: usize, stage: Stage, rows: cuda.DeviceBuffer, row0: usize) !void
    /// called after the stage's producing kernel is queued on `stream`. `rows` borrows the live scratch tensor,
    /// BF16 [n][hidden]; it holds the stage only in stream order, and the next kernel queued may overwrite it, so the
    /// tracer must read it on `stream` and complete the read before returning; row0 is the first row's position.
    /// `routed`, after the down projection, may queue kernels on `stream` reading `r`; `capture` queues no device work.
    pub fn runTraced(self: *Model, stream: cuda.Stream, phase: Phase, tokens: []const u32, extra: ?cuda.DeviceBuffer, trace: anytype) !Rows {
        return self.advance(&.{&self.seq}, &.{self.logits_buf.ptr}, stream, phase, tokens, extra, trace);
    }

    /// Each sequence in `qs` reads `tokens` and `extra` as rows of one batch, its last row's logits to its `heads`.
    fn advance(self: *Model, qs: []const *Sequence, heads: []const u64, stream: cuda.Stream, phase: Phase, tokens: []const u32, extra: ?cuda.DeviceBuffer, trace: anytype) !Rows {
        for (qs) |q| if (q.poisoned) return error.DramaStateNeedsReset;
        const n = tokens.len;
        if (n == 0) return error.EmptyDramaRun;
        if (n * qs.len > self.options.max_rows) return error.TooManyDramaRows;
        for (qs) |q| if (q.pos + n > q.context) return error.DramaContextOverflow;
        for (tokens) |t| if (t >= config.vocab) return error.DramaTokenOutOfRange;
        if (extra) |e| if (e.len < n * H * 2) return error.DramaExtraRowsShort;
        for (qs) |q| q.poisoned = true;
        errdefer stream.synchronize() catch {}; // queued kernels may still read tokens/extra; drain before returning
        const ids = self.tokens.slice(i32)[0 .. n * qs.len];
        for (ids, 0..) |*o, i| o.* = @intCast(tokens[i % n]);
        try self.d.check(self.d.api.cuMemcpyHtoDAsync_v2(self.s.ids, ids.ptr, ids.len * 4, stream.handle), "cuMemcpyHtoDAsync");
        try self.forward(qs, heads, stream, phase, n, extra, trace);
        try stream.synchronize();
        for (qs) |q| {
            q.pos += n;
            q.poisoned = false;
        }
        return .{
            .hidden = .{ .d = self.d, .ptr = self.hidden_buf.ptr, .len = n * H * 2, .borrowed = true },
            .semantic_logits = .{ .d = self.d, .ptr = heads[0], .len = weights.head_cols * 4, .borrowed = true },
            .rows = n,
        };
    }

    fn forward(self: *Model, qs: []const *Sequence, heads: []const u64, s: cuda.Stream, phase: Phase, n: usize, extra: ?cuda.DeviceBuffer, trace: anytype) !void {
        const k = &self.k;
        const sc = &self.s;
        const m = n * qs.len; // each sequence's n rows in turn
        const row0 = qs[0].pos; // a traced run's, which holds one sequence
        try k.embed.bf16(s, sc.ids, self.w.embed, sc.x, m, H);
        if (extra) |e| for (0..qs.len) |r| try k.semanticMix(s, sc.ids + r * n * 4, sc.x + r * n * H * 2, e.ptr, n, H, self.semantic_start, self.semantic_end);
        for (&self.w.layers, 0..) |*l, i| {
            try k.addNorm(s, sc.x, sc.residual, l.ln1, sc.normed, null, null, m, H, config.eps, i == 0);
            try self.stage(trace, s, i, .attention_input, sc.normed, n, row0);
            try self.stage(trace, s, i, .attention_residual, sc.residual, n, row0);
            if (config.linear(i)) try self.deltaNet(qs, s, phase, l, config.slot(i), n) else try self.fullAttention(qs, s, l, config.slot(i), n);
            try self.stage(trace, s, i, .attention_output, sc.x, n, row0);
            const fp8 = l.nvfp4 == null; // NVFP4 experts read the bf16 rows, never their FP8 groups
            try k.addNorm(s, sc.x, sc.residual, l.ln2, sc.normed, if (fp8) sc.xq else null, if (fp8) sc.xs else null, m, H, config.eps, false);
            try self.stage(trace, s, i, .mlp_input, sc.normed, n, row0);
            try self.stage(trace, s, i, .mlp_residual, sc.residual, n, row0);
            try self.moe(s, l, i, m, qs.len, trace);
            try self.stage(trace, s, i, .mlp_output, sc.x, n, row0);
        }
        try k.addNorm(s, sc.x, sc.residual, self.w.final_norm, self.hidden_buf.ptr, null, null, m, H, config.eps, false);
        for (heads, 1..) |logits, r| try k.head(s, self.hidden_buf.ptr + (r * n - 1) * H * 2, self.w.head, weights.head_cols, H, logits);
    }

    /// `k.gemm`, but a batch's rows (one a sequence) take `gemvRows`, which streams the weights once for all of them.
    fn product(self: *const Model, batched: bool, s: cuda.Stream, fp8: ?cuda.qlinear.Weight, x: u64, ldx: usize, m: usize, w: u64, n: usize, kdim: usize, y: u64, ldy: usize) !void {
        if (fp8) |f| return self.k.linear(s, f, x, m, y, self.s.part);
        if (batched) return self.k.gemvRows(s, x, ldx, m, w, n, kdim, y, ldy);
        return self.k.gemm(s, x, ldx, m, w, n, kdim, y, ldy);
    }

    /// One capture point; with a void trace, or one without `capture`, the comptime-false branch emits nothing.
    inline fn stage(self: *const Model, trace: anytype, s: cuda.Stream, layer: usize, comptime which: Stage, ptr: u64, n: usize, row0: usize) !void {
        if (@TypeOf(trace) != void) if (@hasDecl(@typeInfo(@TypeOf(trace)).pointer.child, "capture"))
            try trace.capture(s, layer, which, cuda.DeviceBuffer{ .d = self.d, .ptr = ptr, .len = n * H * 2, .borrowed = true }, row0);
    }

    /// The routed-expert hook, compiled out like `stage` unless the tracer has `routed`.
    inline fn route(self: *const Model, trace: anytype, s: cuda.Stream, layer: usize, n: usize, comptime fp4: bool) !void {
        if (@TypeOf(trace) != void) if (@hasDecl(@typeInfo(@TypeOf(trace)).pointer.child, "routed")) {
            const sc = &self.s;
            var r: Routed = .{ .rows = n, .ids = sc.ids, .picks = sc.picks, .weights = sc.routed, .input = sc.normed };
            if (fp4) {
                r.act = sc.act;
                r.down_f32 = sc.down_f32;
            } else {
                r.gate_up = sc.gu;
                r.down = sc.down;
            }
            try trace.routed(s, layer, r);
        };
    }

    /// Diagnostic replay (temporary; tf-drama-backbone-replay is the only caller): runs one layer's production
    /// `deltaNet`/`fullAttention` (`.attention`) or `moe` on caller-supplied rows instead of the layers below it.
    /// Needs a fresh reset (position 0, not poisoned). Each chunk in order is uploaded on `stream` into
    /// `Scratch.normed`, the operators' production input; the production method runs at request position `row0`, so
    /// attention carries the layer's own DeltaNet conv/recurrent state or KV cache across chunks exactly as `forward`
    /// does; then `Scratch.x`, the operators' production output, is downloaded to `output` at the same rows.
    /// `input` and `output` are BF16 [sum(chunks)][hidden] host bytes (pinned, for stream-ordered copies) and must
    /// not be touched until the call returns, which drains the stream first.
    /// Only this layer's state advances, so the model stays poisoned even on success: run and replayOperator both
    /// fail with DramaStateNeedsReset until reset. Unreferenced by production, so it is never analyzed there.
    pub fn replayOperator(self: *Model, stream: cuda.Stream, layer: usize, operator: ReplayOperator, chunks: []const usize, input: []const u8, output: []u8) !void {
        const q = &self.seq;
        if (q.poisoned) return error.DramaStateNeedsReset;
        if (q.pos != 0) return error.DramaReplayNeedsReset;
        if (layer >= self.w.layers.len) return error.DramaReplayLayerOutOfRange;
        var rows: usize = 0;
        for (chunks) |n| {
            if (n == 0) return error.EmptyDramaRun;
            if (n > self.options.max_rows) return error.TooManyDramaRows;
            rows += n;
        }
        if (rows == 0) return error.EmptyDramaRun;
        if (rows > self.options.context) return error.DramaContextOverflow;
        if (input.len != rows * H * 2 or output.len != rows * H * 2) return error.DramaReplayRowsMismatch;
        q.poisoned = true; // never cleared here: the other layers' state no longer matches pos
        errdefer stream.synchronize() catch {}; // queued copies may still touch input/output; drain before returning
        const l = &self.w.layers[layer];
        var row0: usize = 0;
        for (chunks) |n| {
            const at = row0 * H * 2;
            const bytes = n * H * 2;
            const normed: cuda.DeviceBuffer = .{ .d = self.d, .ptr = self.s.normed, .len = bytes, .borrowed = true };
            try normed.uploadAsync(0, input[at..][0..bytes], stream.handle);
            q.pos = row0;
            switch (operator) {
                .attention => if (config.linear(layer))
                    try self.deltaNet(&.{q}, stream, .prefill, l, config.slot(layer), n)
                else
                    try self.fullAttention(&.{q}, stream, l, config.slot(layer), n),
                .moe => {
                    try self.k.actQuant(stream, self.s.normed, H, self.s.xq, self.s.xs, n, H);
                    try self.moe(stream, l, layer, n, 1, {});
                },
            }
            const x: cuda.DeviceBuffer = .{ .d = self.d, .ptr = self.s.x, .len = bytes, .borrowed = true };
            try x.downloadAsync(0, output[at..][0..bytes], stream.handle);
            row0 += n;
        }
        try stream.synchronize();
        q.pos = row0;
    }

    /// Qwen3.5 GatedDeltaNet: in projection, causal conv + silu, gated delta rule, RMSNormGated * silu(z), out_proj.
    fn deltaNet(self: *Model, qs: []const *Sequence, s: cuda.Stream, phase: Phase, l: *const weights.Layer, slot: usize, n: usize) !void {
        const k = &self.k;
        const sc = &self.s;
        const fp8_in = l.fp8.gdn_in != null;
        // an FP8 qkv | z lands in packed rows: b | a then come from their own bf16 rows, as the lane has no row stride
        const ld: usize = if (fp8_in) weights.gdn_b_col else weights.gdn_in_rows;
        const ba = 2 * config.linear_value_heads;
        const m = n * qs.len;
        const batched = qs.len > 1;
        try self.product(batched, s, l.fp8.gdn_in, sc.normed, H, m, l.gdn_in, ld, H, sc.proj, ld);
        if (fp8_in) try self.product(batched, s, null, sc.normed, H, m, l.gdn_ba, ba, H, sc.ba, ba);
        // a batch shares the projections; each sequence keeps its own conv and recurrent state
        for (qs, 0..) |q, r| {
            const proj = sc.proj + r * n * ld * 2;
            const conv = sc.conv + r * n * config.conv_dim * 2;
            const recurrence: gdn.Run = .{
                .qkv = conv,
                .ld_qkv = config.conv_dim,
                .ba = if (fp8_in) sc.ba + r * n * ba * 2 else proj + weights.gdn_b_col * 2,
                .ld_ba = if (fp8_in) ba else ld,
                .a_log = l.a_log,
                .dt_bias = l.dt_bias,
                .y = sc.core + r * n * config.value_dim * 2,
                .rows = n,
                .nk = config.linear_key_heads,
                .nv = config.linear_value_heads,
                .dv = config.linear_dim,
                .scale = @floatCast(1.0 / @sqrt(@as(f64, config.linear_dim))),
            };
            const state = &q.gdn_states[slot];
            switch (phase) {
                .prefill => {
                    try k.convPrefill(s, proj, ld, l.conv, q.convState(slot), conv, n, config.conv_dim);
                    try k.gdn.prefill(s, state, self.gdn_ws, recurrence);
                },
                .decode => {
                    try k.conv(s, proj, ld, l.conv, q.convState(slot), conv, n, config.conv_dim);
                    try k.gdn.decode(s, state, recurrence);
                },
            }
        }
        try k.gatedRms(s, sc.core, config.value_dim, sc.proj + weights.gdn_z_col * 2, ld, l.gnorm, sc.gated, m, config.eps);
        try self.product(batched, s, l.fp8.gdn_out, sc.gated, config.value_dim, m, l.gdn_out, H, config.value_dim, sc.x, H);
    }

    /// Gated full attention: q|gate, k, v projection; q/k Gemma norm + partial RoPE; causal GQA; sigmoid gate; o_proj.
    fn fullAttention(self: *Model, qs: []const *Sequence, s: cuda.Stream, l: *const weights.Layer, slot: usize, n: usize) !void {
        const k = &self.k;
        const sc = &self.s;
        const ld = weights.attn_in_rows;
        const width = config.query_heads * config.head_dim;
        const m = n * qs.len;
        try self.product(qs.len > 1, s, l.fp8.attn_in, sc.normed, H, m, l.attn_in, ld, H, sc.proj, ld);
        for (qs, 0..) |q, r| try k.attention(s, .{
            .qkv = sc.proj + r * n * ld * 2,
            .ld_qkv = ld,
            .q_norm = l.q_norm,
            .k_norm = l.k_norm,
            .inv_freq = self.w.inv_freq,
            .q = sc.q + r * n * width * 2,
            .kc = q.kCache(slot),
            .vc = q.vCache(slot),
            .out = sc.gated + r * n * width * 2,
            .pos0 = q.pos,
            .rows = n,
            .heads = config.query_heads,
            .kv_heads = config.kv_heads,
            .eps = config.eps,
            .scale = 1.0 / 16.0, // head_dim ** -0.5
        });
        try self.product(qs.len > 1, s, l.fp8.attn_out, sc.gated, width, m, l.attn_out, H, width, sc.x, H);
    }

    /// Qwen2MoeSparseMoeBlock with bsgemm-moe experts: softmax top-8 renormalized routing, W8A8 block-FP8 experts
    /// (dynamic 1 x 128 activation scales) or NVFP4 (fp4Experts), SwiGLU, weighted down, plus the gated shared expert.
    /// Input: Scratch.normed, and for FP8 layers its FP8 groups in xq / xs (the post-attention addNorm writes both).
    fn moe(self: *Model, s: cuda.Stream, l: *const weights.Layer, layer: usize, n: usize, batch: usize, trace: anytype) !void {
        const k = &self.k;
        const sc = &self.s;
        const pairs = n * top;
        const ld = sharedLd(l);
        const I = config.expert_inner;
        comptime std.debug.assert(config.shared_inner == I); // one SwiGLU launch serves both experts
        const keep = if (self.keep_buf) |b| b.ptr + layer * expert_mask.words * 4 else 0;
        const slot = if (self.slot_buf) |b| b.ptr + layer * config.experts * 4 else 0;
        const batched = batch > 1;
        try self.product(batched, s, l.fp8.shared_in, sc.normed, H, n, l.moe_in, ld, H, sc.moe_in, ld);
        if (l.fp8.shared_in != null) {
            try self.product(batched, s, null, sc.normed, H, n, l.router, config.experts, H, sc.router, config.experts);
            try k.topk(s, sc.router, config.experts, sc.picks, sc.routed, n, top, keep, slot, sc.slots);
        } else try k.topk(s, sc.moe_in + weights.router_col * 2, ld, sc.picks, sc.routed, n, top, keep, slot, sc.slots);
        const plan: cuda.grouped.Plan = .{ .members = sc.members, .items = sc.items, .counts = sc.counts, .rank = sc.rank, .hist = sc.hist };
        // a batch's rows route independently: an expert both pick is read once, so two rows read 8 to 16 experts
        try k.router.route(s, if (slot != 0) sc.slots else sc.picks, pairs, config.experts, tile, plan);
        const items = cuda.grouped.maxItems(pairs, config.experts, tile);
        if (l.nvfp4) |ex| return self.fp4Experts(s, l, layer, ex, plan, items, n, batch, trace);
        try k.expert(s, plan, pairs, .{
            .a = sc.xq,
            .a_div = top,
            .a_scales = sc.xs,
            .w = l.gu,
            .w_scales = l.gu_s,
            .n = weights.gu_rows,
            .kdim = H,
            .max_items = items,
            .routed = null,
            .y = sc.gu,
        });
        try k.siluMul(s, sc.gu, weights.gu_rows, sc.hq, sc.hs, pairs, sc.moe_in, ld, sc.shared_h, n, I);
        try k.expert(s, plan, pairs, .{
            .a = sc.hq,
            .a_div = 1,
            .a_scales = sc.hs,
            .w = l.dn,
            .w_scales = l.dn_s,
            .n = H,
            .kdim = I,
            .max_items = items,
            .routed = sc.routed,
            .y = sc.down,
        });
        try self.route(trace, s, layer, n / batch, false);
        try self.sharedDown(s, l, n, batched);
        try k.combine(s, sc.down, top, sc.normed, l.shared_gate, sc.shared_out, sc.x, n, H);
    }

    /// moe's experts for an NVFP4 layer, in the decode form at any row count: a pair's bits never depend on its item.
    fn fp4Experts(self: *Model, s: cuda.Stream, l: *const weights.Layer, layer: usize, ex: cuda.experts.Layer, plan: cuda.grouped.Plan, items: usize, n: usize, batch: usize, trace: anytype) !void {
        const k = &self.k;
        const sc = &self.s;
        const g: cuda.experts.Grouped = .{ .nvfp4 = if (k.fp4) |*f| &f.experts else null };
        try g.decode(s, .up, .{ .x = sc.normed, .stride = H, .slots = top }, ex, plan, sc.act, items, -1);
        try k.siluMul(s, 0, 0, 0, 0, 0, sc.moe_in, sharedLd(l), sc.shared_h, n, config.shared_inner); // shared rows only
        try g.decode(s, .down_f32, .{ .x = sc.act, .stride = config.expert_inner }, ex, plan, sc.down_f32, items, -1);
        try self.route(trace, s, layer, n / batch, true);
        try self.sharedDown(s, l, n, batch > 1);
        try k.combineWeighted(s, sc.down_f32, sc.routed, top, sc.normed, l.shared_gate, sc.shared_out, sc.x, n, H);
    }

    /// The shared expert's down projection, from either stored format.
    fn sharedDown(self: *Model, s: cuda.Stream, l: *const weights.Layer, n: usize, batched: bool) !void {
        const sc = &self.s;
        try self.product(batched, s, l.fp8.shared_out, sc.shared_h, config.shared_inner, n, l.shared_down, H, config.shared_inner, sc.shared_out, H);
    }
};

/// The row stride of the shared gate | up outputs: packed when the lane writes them, else the fused MoE input's.
fn sharedLd(l: *const weights.Layer) usize {
    return if (l.fp8.shared_in != null) weights.router_col else weights.moe_in_rows;
}

test "a sequence holds its 10 attention layers' KV caches and 30 DeltaNet states: 0.33 GB at context 8192" {
    const deltanet = 30 * (3 * 8192 * 2 + 2 * 32 * 128 * 128 * 4 + 64 * 8192 * 2 + 64 * 2 * 32 * 2); // 159_006_720
    const kv = 10 * 2 * 8192 * 2 * 256 * 2; // 167_772_160
    try std.testing.expectEqual(@as(usize, deltanet + kv), Sequence.bytes(8192));
    try std.testing.expectEqual(@as(usize, deltanet + kv / 2), Sequence.bytes(4096));
}

test "a batch's product takes 1 to 8 rows of 256-wide k chunks, 16-byte aligned, and refuses the rest before any launch" {
    var k: kernels.Kernels = undefined;
    k.gemv_rows_fn = null;
    const s: cuda.Stream = undefined;
    try std.testing.expectError(error.DramaGemvRowsNotLoaded, k.gemvRows(s, 0, H, 2, 0, H, H, 0, H));
    k.gemv_rows_fn = .{ .d = undefined, .handle = null };
    // (rows, ldx, k, x): none, past the kernel's 8 lane groups, a k off the 256 chunk, an ldx off 8, an unaligned x
    for ([_][4]usize{ .{ 0, H, H, 0 }, .{ kernels.gemv_rows + 1, H, H, 0 }, .{ 2, H, H + 32, 0 }, .{ 2, H + 4, H, 0 }, .{ 2, H, H, 8 } }) |c|
        try std.testing.expectError(error.DramaGemmShape, k.gemvRows(s, c[3], c[1], c[0], 0, H, c[2], 0, H));
}
