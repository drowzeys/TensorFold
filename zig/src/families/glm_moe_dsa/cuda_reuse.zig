//! Prompt-state reuse for full GLM-5.3 on CUDA (one stream): the Python engine's TF_GLM53_PROMPT_REUSE
//! (families/glm_moe_dsa/cuda/prefixes.py: PromptPlan, cut_and_keep, KeptPrompts, PromptReuse), kept prompt states by
//! token ids, resumed instead of prefilled.
//!
//! The prompt path is not chunk-invariant, so a kept state is only resumed where every prompt that shares it is cut
//! alike: keep points are a function of the tokens before them (`Plan.points`: the end of the system block, then
//! assistant openers at least `gap` tokens apart), a prompt prefills to each of its points as a prompt of that length
//! would (cuda_runner.cutChunks), so a fresh prompt and one resumed at any of its points run the same chunks - resumed
//! == fresh bit for bit (TF_EXL3_PROMPT_DET=slots16).
//!
//! What a state at n keeps: the target's latent and index-key rows [0, n) and the MTP layer's [0, n - 1) - left in the
//! live caches, copied out only before another conversation overwrites them (one copy a conversation; its shorter
//! states read the first rows of it), the MTP carry (the target hidden n - 1), and at a prompt's own end the head's
//! logits row (a replay: an identical prompt picks its first token with nothing prefilled). DCP: a rank holds
//! ceil(n / dcp) rows of each cache (prefixes.row_views), so every rank counts the same bytes.
//!
//! Every rank keeps the same store: rank 0 picks the resume point and the keep points (`choose`) and shares them with
//! the request; every rank then saves, drops and keeps alike (`run`): the byte counts do not depend on the rank and
//! the budget is the ranks' least. The bookkeeping is host-only (`Store`, unit-tested); `Rows` does the device copies.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const rn = @import("cuda_runner.zig");
const W = @import("cuda_weights.zig");

pub const gap_default: usize = 1024; // TF_GLM53_REUSE_GAP
pub const system_min: usize = 256; // the system block's end is a keep point this far in or more
pub const keep_most: usize = 3; // keep points a request adds besides its own end
pub const entries_default: usize = 32; // TF_GLM53_CACHE_ENTRIES
pub const cache_gib_default: f64 = 4.0; // TF_GLM53_CACHE_GIB
pub const save_align: usize = 256; // prefixes.Saved.ALIGN

/// The reuse knobs every rank must share (prefixes.settings + enabled).
pub const Settings = struct {
    on: bool = false,
    gap: usize = gap_default,
    entries: usize = entries_default,
    gib: f64 = cache_gib_default,
    loose: bool = false,

    fn env(name: [:0]const u8) ?[]const u8 {
        const v = std.c.getenv(name) orelse return null;
        return std.mem.trim(u8, std.mem.span(v), " ");
    }

    /// TF_GLM53_PROMPT_REUSE (0 | 1 | on | off), TF_GLM53_REUSE_GAP, TF_GLM53_CACHE_ENTRIES, TF_GLM53_CACHE_GIB,
    /// TF_GLM53_REUSE_LOOSE.
    pub fn fromEnv() !Settings {
        var s: Settings = .{};
        if (env("TF_GLM53_PROMPT_REUSE")) |v| {
            if (std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "on")) {
                s.on = true;
            } else if (!(v.len == 0 or std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off"))) return error.BadPromptReuse;
        }
        if (env("TF_GLM53_REUSE_GAP")) |v| s.gap = std.fmt.parseInt(usize, v, 10) catch return error.BadPromptReuse;
        if (env("TF_GLM53_CACHE_ENTRIES")) |v| s.entries = std.fmt.parseInt(usize, v, 10) catch return error.BadPromptReuse;
        if (env("TF_GLM53_CACHE_GIB")) |v| s.gib = std.fmt.parseFloat(f64, v) catch return error.BadPromptReuse;
        if (env("TF_GLM53_REUSE_LOOSE")) |v| s.loose = std.mem.eql(u8, v, "1");
        if (s.gap < 1 or s.entries < 1 or s.gib < 0) return error.BadPromptReuse;
        return s;
    }
};

/// GLM's <|user|>, <|assistant|> and <think> ids from the checkpoint's tokenizer.json (null where absent).
pub const Specials = struct { user: ?u32 = null, assistant: ?u32 = null, think: ?u32 = null };

/// prefixes.special_ids: the added tokens of `dir`/tokenizer.json.
pub fn specialIds(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) Specials {
    var out: Specials = .{};
    const path = std.fs.path.join(gpa, &.{ dir, "tokenizer.json" }) catch return out;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(256 << 20)) catch return out;
    defer gpa.free(text);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch return out;
    if (doc != .object) return out;
    const added = doc.object.get("added_tokens") orelse return out;
    if (added != .array) return out;
    for (added.array.items) |t| {
        if (t != .object) continue;
        const c = t.object.get("content") orelse continue;
        const id = t.object.get("id") orelse continue;
        if (c != .string or id != .integer) continue;
        const v: u32 = std.math.cast(u32, id.integer) orelse continue;
        if (std.mem.eql(u8, c.string, "<|user|>")) out.user = v;
        if (std.mem.eql(u8, c.string, "<|assistant|>")) out.assistant = v;
        if (std.mem.eql(u8, c.string, "<think>")) out.think = v;
    }
    return out;
}

/// prefixes.PromptPlan: where a prompt's keep points are, from its tokens alone - prompts that agree up to a point
/// agree on every point up to it, so they cut their prompt chunks alike there.
pub const Plan = struct {
    ids: Specials,
    gap: usize = gap_default,
    system_min: usize = system_min,

    /// PromptPlan.points: the first <|user|> when at least system_min in; then, at least `gap` after the last point,
    /// the position after an <|assistant|> opener (after its <think> when one follows). Ascending.
    pub fn points(p: Plan, gpa: std.mem.Allocator, prompt: []const u32) ![]usize {
        var out: std.ArrayList(usize) = .empty;
        errdefer out.deinit(gpa);
        const L = prompt.len;
        var last: usize = 0;
        if (p.ids.user) |u| {
            if (std.mem.indexOfScalar(u32, prompt, u)) |at| {
                if (at >= p.system_min) {
                    last = at;
                    try out.append(gpa, at);
                }
            }
        }
        if (p.ids.assistant) |asst| {
            for (prompt, 0..) |t, i| {
                if (t != asst) continue;
                var c = i + 1;
                if (p.ids.think) |th| {
                    if (c < L and prompt[c] == th) c += 1;
                }
                if (c >= last and c - last >= p.gap) {
                    try out.append(gpa, c);
                    last = c;
                }
            }
        }
        return out.toOwnedSlice(gpa);
    }

    /// PromptPlan.system_point: the system block's point among `pts` (the first <|user|>), or null.
    pub fn systemPoint(p: Plan, pts: []const usize, prompt: []const u32) ?usize {
        if (pts.len == 0) return null;
        const u = p.ids.user orelse return null;
        const at = pts[0];
        return if (at < prompt.len and prompt[at] == u) at else null;
    }
};

fn contains(xs: []const usize, v: usize) bool {
    return std.mem.indexOfScalar(usize, xs, v) != null;
}

/// prefixes.common_prefix.
pub fn commonPrefix(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) i += 1;
    return i;
}

fn isPrefix(short: []const u32, long: []const u32) bool {
    return short.len <= long.len and std.mem.eql(u32, long[0..short.len], short);
}

/// Whether optional state `p` is `e`.
fn isKept(p: ?*Kept, e: *const Kept) bool {
    const q = p orelse return false;
    return @intFromPtr(q) == @intFromPtr(e);
}

/// Whether `e` reads saved copy `sv`.
fn savedIs(e: *const Kept, sv: *const Saved) bool {
    const x = e.saved orelse return false;
    return @intFromPtr(x) == @intFromPtr(sv);
}

/// A device allocation the store's entries hold (the pool hands it back; tests use fake ones).
pub const Handle = struct { ptr: u64 = 0, len: usize = 0 };

/// prefixes.Saved: one copy of a state's cache rows (row views) in one allocation; a conversation's shorter states
/// read its first rows.
pub const Saved = struct {
    mem: Handle,
    nbytes: usize, // Saved.size of the views it was copied from (each part ALIGN-aligned)
    n: usize, // the tokens of the state it was copied from
    refs: usize = 0,
};

/// prefixes.Kept: a prompt state at n = ids.len. `saved` null: the target / MTP rows sit in the live caches.
pub const Kept = struct {
    ids: []u32,
    mode: u8 = 0, // the MTP input its carry and MTP rows were made with (0: "normed", the only mode here)
    carry: Handle = .{}, // the target hidden n - 1 as the MTP layer reads it: bf16 [hidden]
    head: ?Handle = null, // at a prompt's end: the head's local logits row (fp32 [vocab_part])
    shared: bool = false, // a system block or fork point: outlives its conversation's turns
    saved: ?*Saved = null,
    tick: u64 = 0,
    learned: bool = false, // --learn: written to disk (or read from it)
    slot: usize = 0, // Phase 4b (prefixes.SlotPrompts, e.extra["slot"]): the stream slot whose rows hold it
    ring: ?Handle = null, // Phase 4c (Kept.ring): a drafter stream's ring window at n (MultiDrafter.saveWindow)

    pub fn n(e: *const Kept) usize {
        return e.ids.len;
    }
};

/// What the store's frees go to (the device pool); the store itself never touches the device.
pub const Release = struct {
    ctx: *anyopaque,
    free: *const fn (ctx: *anyopaque, h: Handle) void,
};

/// prefixes.KeptPrompts (one stream): the kept states' host bookkeeping. Every rank runs the same calls in the same
/// order, so every rank holds, saves and drops the same entries.
pub const Store = struct {
    gpa: std.mem.Allocator,
    budget: u64,
    cap: usize,
    loose: bool,
    /// the bytes a carry and a head count (prefixes.Kept.held: bf16 [hidden], fp32 [1, vocab] as Python holds it)
    carry_bytes: usize,
    head_bytes: usize,
    release: Release,
    entries: std.ArrayList(*Kept) = .empty,
    live: std.ArrayList(u32) = .empty, // the ids whose rows the live caches hold
    clock: u64 = 0,
    /// Phase 4b (prefixes.SlotPrompts, --parallel N): lives[slot] = the ids whose rows that slot holds; states stay in
    /// their slot's rows (never saved out of them) and are usable while the slot still holds their ids. Empty: one
    /// stream (`live`).
    lives: []std.ArrayList(u32) = &.{},

    pub fn init(gpa: std.mem.Allocator, budget: u64, entries: usize, loose: bool, carry_bytes: usize, head_bytes: usize, release: Release) Store {
        return .{ .gpa = gpa, .budget = budget, .cap = @max(1, entries), .loose = loose, .carry_bytes = carry_bytes, .head_bytes = head_bytes, .release = release };
    }

    pub fn deinit(s: *Store) void {
        s.clear();
        s.entries.deinit(s.gpa);
        s.live.deinit(s.gpa);
        for (s.lives) |*l| l.deinit(s.gpa);
        if (s.lives.len > 0) s.gpa.free(s.lives);
    }

    /// Slot mode (prefixes.SlotPrompts) over `n` stream slots, every slot holding nothing yet.
    pub fn setSlots(s: *Store, n: usize) !void {
        if (s.lives.len > 0 or n == 0) return error.Invalid;
        s.lives = try s.gpa.alloc(std.ArrayList(u32), n);
        for (s.lives) |*l| l.* = .empty;
    }

    /// SlotPrompts.lives[slot] = ids (the slot's rows hold these ids now).
    pub fn setLiveSlot(s: *Store, slot: usize, ids: []const u32) !void {
        if (slot >= s.lives.len) return error.Invalid;
        s.lives[slot].clearRetainingCapacity();
        try s.lives[slot].appendSlice(s.gpa, ids);
    }

    /// SlotPrompts.named_in: the usable state of `slot` with prompt[:n] (followers find rank 0's resume point).
    pub fn namedIn(s: *const Store, prompt: []const u32, n: usize, slot: usize) ?*Kept {
        for (s.entries.items) |e| {
            if (e.slot == slot and e.n() == n and n <= prompt.len and s.usable(e) and std.mem.eql(u32, prompt[0..n], e.ids)) return e;
        }
        return null;
    }

    /// The newest tick among `slot`'s states (0: none) - SlotReuse.choose takes the free slot whose states are worth
    /// least.
    pub fn slotTick(s: *const Store, slot: usize) u64 {
        var t: u64 = 0;
        for (s.entries.items) |e| if (e.slot == slot) {
            t = @max(t, e.tick);
        };
        return t;
    }

    /// SlotPrompts.overwritten_slot: the states of `slot` a request resumed there at `begin` overwrites - all but the
    /// prefixes of its prompt no longer than `begin` (and, when rows copied from another slot replace this slot's,
    /// only those `same` accepts).
    pub fn overwrittenSlot(s: *const Store, gpa: std.mem.Allocator, slot: usize, prompt: []const u32, begin: usize, same: ?Same) ![]*Kept {
        var out: std.ArrayList(*Kept) = .empty;
        errdefer out.deinit(gpa);
        for (s.entries.items) |e| {
            if (e.slot != slot) continue;
            if (e.n() <= begin and e.n() <= prompt.len and std.mem.eql(u32, prompt[0..e.n()], e.ids) and (same == null or same.?.accepts(e))) continue;
            try out.append(gpa, e);
        }
        return out.toOwnedSlice(gpa);
    }

    fn heldOne(s: *const Store, e: *const Kept) u64 {
        return s.carry_bytes + (if (e.head != null) s.head_bytes else 0);
    }

    /// KeptPrompts.held: bytes the given entries (null: all) hold, their own tensors and each saved copy once.
    pub fn heldOf(s: *const Store, es: []const *Kept) u64 {
        var total: u64 = 0;
        for (es, 0..) |e, i| {
            total += s.heldOne(e);
            if (e.saved) |sv| {
                var first = true;
                for (es[0..i]) |x| if (savedIs(x, sv)) {
                    first = false;
                    break;
                };
                if (first) total += sv.nbytes;
            }
        }
        return total;
    }

    pub fn held(s: *const Store) u64 {
        return s.heldOf(s.entries.items);
    }

    pub fn touch(s: *Store, e: *Kept) void {
        s.clock += 1;
        e.tick = s.clock;
    }

    pub fn clear(s: *Store) void {
        while (s.entries.items.len > 0) s.drop(s.entries.items[s.entries.items.len - 1]);
    }

    fn indexOf(s: *const Store, e: *const Kept) ?usize {
        for (s.entries.items, 0..) |x, i| if (x == e) return i;
        return null;
    }

    pub fn has(s: *const Store, e: *const Kept) bool {
        return s.indexOf(e) != null;
    }

    /// Unshare `e`'s saved copy (the copy goes with its last state).
    pub fn unsave(s: *Store, e: *Kept) void {
        const sv = e.saved orelse return;
        e.saved = null;
        sv.refs -= 1;
        if (sv.refs == 0) {
            s.release.free(s.release.ctx, sv.mem);
            s.gpa.destroy(sv);
        }
    }

    /// A state's own tensors released and the state freed (it is no entry, or no longer one).
    pub fn destroy(s: *Store, e: *Kept) void {
        s.unsave(e);
        if (e.carry.len > 0) s.release.free(s.release.ctx, e.carry);
        if (e.head) |h| s.release.free(s.release.ctx, h);
        if (e.ring) |h| s.release.free(s.release.ctx, h);
        s.gpa.free(e.ids);
        s.gpa.destroy(e);
    }

    /// KeptPrompts.drop.
    pub fn drop(s: *Store, e: *Kept) void {
        if (s.indexOf(e)) |i| _ = s.entries.orderedRemove(i);
        s.destroy(e);
    }

    pub fn inLive(s: *const Store, e: *const Kept) bool {
        if (s.lives.len > 0) {
            if (e.slot >= s.lives.len) return false;
            const live = s.lives[e.slot].items;
            return e.n() <= live.len and std.mem.eql(u32, live[0..e.n()], e.ids);
        }
        return e.n() <= s.live.items.len and std.mem.eql(u32, s.live.items[0..e.n()], e.ids);
    }

    /// Saved, or still in the live caches (a request that failed mid-prefill left later rows unknown); slot mode: in
    /// its slot's rows only.
    pub fn usable(s: *const Store, e: *const Kept) bool {
        if (s.lives.len > 0) return s.inLive(e);
        return e.saved != null or s.inLive(e);
    }

    /// KeptPrompts.best (rank 0): the longest state `prompt` resumes from - a strict prefix at one of the prompt's own
    /// points (any kept prefix when loose), or the whole prompt when it kept its head row (a replay).
    pub fn best(s: *const Store, prompt: []const u32, pts: []const usize, mode: u8) ?*Kept {
        const L = prompt.len;
        var out: ?*Kept = null;
        for (s.entries.items) |e| {
            const n = e.n();
            if (e.mode != mode or n > L or (out != null and n <= out.?.n()) or !s.usable(e)) continue;
            const ok = if (n == L) e.head != null else (contains(pts, n) or s.loose);
            if (ok and std.mem.eql(u32, prompt[0..n], e.ids)) out = e;
        }
        return out;
    }

    /// KeptPrompts.named: the state rank 0 resumes from (followers).
    pub fn named(s: *const Store, prompt: []const u32, n: usize) ?*Kept {
        for (s.entries.items) |e| {
            if (e.n() == n and n <= prompt.len and s.usable(e) and std.mem.eql(u32, prompt[0..n], e.ids)) return e;
        }
        return null;
    }

    /// The most leading tokens `prompt` shares with a kept state.
    pub fn longestShared(s: *const Store, prompt: []const u32) usize {
        var m: usize = 0;
        for (s.entries.items) |e| m = @max(m, commonPrefix(prompt, e.ids));
        return m;
    }

    /// A strict prefix of another kept state: an earlier point of a conversation that went on.
    pub fn superseded(s: *const Store, e: *const Kept) bool {
        for (s.entries.items) |f| {
            if (f != e and f.n() > e.n() and isPrefix(e.ids, f.ids)) return true;
        }
        return false;
    }

    fn protectedBy(e: *const Kept, protect: []const ?*Kept) bool {
        for (protect) |p| if (isKept(p, e)) return true;
        return false;
    }

    /// KeptPrompts.victim (0063): a superseded unshared state (oldest first), then a superseded shared one, then the
    /// least recently used.
    pub fn victim(s: *const Store, protect: []const ?*Kept) ?*Kept {
        var oldest: ?*Kept = null;
        var sup_any: ?*Kept = null;
        var sup_unshared: ?*Kept = null;
        for (s.entries.items) |e| {
            if (protectedBy(e, protect)) continue;
            if (oldest == null or e.tick < oldest.?.tick) oldest = e;
            if (!s.superseded(e)) continue;
            if (sup_any == null or e.tick < sup_any.?.tick) sup_any = e;
            if (!e.shared and (sup_unshared == null or e.tick < sup_unshared.?.tick)) sup_unshared = e;
        }
        return sup_unshared orelse sup_any orelse oldest;
    }

    /// KeptPrompts.fit: drop states until `need` more bytes fit the budget; false (nothing dropped) when even
    /// dropping every unprotected state would not make room.
    pub fn fit(s: *Store, need: u64, protect: []const ?*Kept) bool {
        var floor_list: std.ArrayList(*Kept) = .empty;
        defer floor_list.deinit(s.gpa);
        for (s.entries.items) |e| if (protectedBy(e, protect)) floor_list.append(s.gpa, e) catch return false;
        if (s.heldOf(floor_list.items) + need > s.budget) return false;
        while (s.held() + need > s.budget) {
            const v = s.victim(protect) orelse return false;
            s.drop(v);
        }
        return true;
    }

    /// KeptPrompts.remember: keep `e` (newest), replacing a state of the same ids, within the entry cap and the
    /// budget. False: `e` was not kept (the caller destroys it).
    pub fn remember(s: *Store, e: *Kept, protect_in: []const ?*Kept) bool {
        var i: usize = 0;
        while (i < s.entries.items.len) {
            const x = s.entries.items[i];
            // KeptPrompts.same_place: one stream - always; slot mode - the same slot
            const same_place = s.lives.len == 0 or x.slot == e.slot;
            if (same_place and x.n() == e.n() and std.mem.eql(u32, x.ids, e.ids)) {
                s.drop(x);
                continue;
            }
            i += 1;
        }
        var protect_buf: [8]?*Kept = @splat(null);
        const np = @min(protect_in.len, protect_buf.len - 1);
        @memcpy(protect_buf[0..np], protect_in[0..np]);
        protect_buf[np] = e;
        const protect = protect_buf[0 .. np + 1];
        while (s.entries.items.len >= s.cap) {
            const v = s.victim(protect) orelse return false;
            s.drop(v);
        }
        var own = s.heldOne(e);
        if (e.saved) |sv| {
            var shared_copy = false;
            for (s.entries.items) |x| if (savedIs(x, sv)) {
                shared_copy = true;
                break;
            };
            if (!shared_copy) own += sv.nbytes;
        }
        if (!s.fit(own, protect)) return false;
        s.touch(e);
        s.entries.append(s.gpa, e) catch return false;
        return true;
    }

    /// KeptPrompts.overwritten: the live states a request resumed at `begin` overwrites (newest first): all but
    /// prefixes of its prompt no longer than `begin` - and, when saved rows replace the live ones, only those `same`
    /// accepts. `same`: null, or (the hit's mode, the prompt's points, begin).
    pub fn overwritten(s: *const Store, gpa: std.mem.Allocator, prompt: []const u32, begin: usize, same: ?Same) ![]*Kept {
        var out: std.ArrayList(*Kept) = .empty;
        errdefer out.deinit(gpa);
        for (s.entries.items) |e| {
            if (e.saved != null) continue;
            if (e.n() <= begin and std.mem.eql(u32, prompt[0..e.n()], e.ids) and (same == null or same.?.accepts(e))) continue;
            try out.append(gpa, e);
        }
        std.mem.sort(*Kept, out.items, {}, struct {
            fn newer(_: void, a: *Kept, b: *Kept) bool {
                return a.tick > b.tick;
            }
        }.newer);
        return out.toOwnedSlice(gpa);
    }

    pub fn setLive(s: *Store, ids: []const u32) !void {
        s.live.clearRetainingCapacity();
        try s.live.appendSlice(s.gpa, ids);
    }
};

/// PromptReuse.run's `same`: a live prefix keeps its bits only where it was cut alike.
pub const Same = struct {
    mode: u8,
    points: []const usize,
    begin: usize,
    fn accepts(x: Same, e: *const Kept) bool {
        return e.mode == x.mode and (contains(x.points, e.n()) or e.n() == x.begin);
    }
};

/// cut_and_keep's result: the prompt's points past `begin` short of its end (its chunk cuts), and the ones kept
/// (n * 2 + shared).
pub const Choice = struct {
    begin: usize,
    stops: []usize,
    keeps: []u32,

    pub fn deinit(c: Choice, gpa: std.mem.Allocator) void {
        gpa.free(c.stops);
        gpa.free(c.keeps);
    }
};

/// prefixes.cut_and_keep (rank 0): the stops past `begin` short of the end, and the kept ones: the last one, the
/// system block's end, and where the prompt stops sharing a kept state (0015), at most keep_most. `resume` false
/// (the serial reference): cut alike, nothing kept.
pub fn cutAndKeep(gpa: std.mem.Allocator, plan: Plan, store: *const Store, prompt: []const u32, pts: []const usize, begin: usize, resume_: bool) !Choice {
    const L = prompt.len;
    var stops: std.ArrayList(usize) = .empty;
    errdefer stops.deinit(gpa);
    for (pts) |p| if (begin < p and p < L) try stops.append(gpa, p);
    var keeps: std.ArrayList(u32) = .empty;
    errdefer keeps.deinit(gpa);
    if (resume_) {
        // an insertion-ordered dict {n: shared}
        var kn: [4]usize = undefined;
        var ks: [4]bool = undefined;
        var count: usize = 0;
        const Put = struct {
            fn put(n_: []usize, s_: []bool, c: *usize, n: usize, sh: bool) void {
                for (n_[0..c.*], 0..) |x, i| if (x == n) {
                    s_[i] = sh;
                    return;
                };
                n_[c.*] = n;
                s_[c.*] = sh;
                c.* += 1;
            }
        };
        if (stops.items.len > 0) Put.put(&kn, &ks, &count, stops.items[stops.items.len - 1], false);
        if (plan.systemPoint(pts, prompt)) |sp| {
            if (begin < sp and sp < L) Put.put(&kn, &ks, &count, sp, true);
        }
        const shared = store.longestShared(prompt);
        var fork: ?usize = null;
        for (stops.items) |p| if (p <= shared) {
            fork = if (fork) |f| @max(f, p) else p;
        };
        if (fork) |f| {
            if (!contains(kn[0..count], f)) Put.put(&kn, &ks, &count, f, true);
        }
        for (0..@min(count, keep_most)) |i| try keeps.append(gpa, @intCast(kn[i] * 2 + @intFromBool(ks[i])));
        std.mem.sort(u32, keeps.items, {}, std.sort.asc(u32));
    }
    return .{ .begin = begin, .stops = try stops.toOwnedSlice(gpa), .keeps = try keeps.toOwnedSlice(gpa) };
}

/// PromptReuse.choose (rank 0): (resume point, the prompt's cuts past it, the kept ones).
pub fn choose(gpa: std.mem.Allocator, plan: Plan, store: *const Store, prompt: []const u32, resume_: bool, mode: u8) !Choice {
    const pts = try plan.points(gpa, prompt);
    defer gpa.free(pts);
    const hit = if (resume_) store.best(prompt, pts, mode) else null;
    const begin = if (hit) |h| h.n() else 0;
    return cutAndKeep(gpa, plan, store, prompt, pts, begin, resume_);
}

// --------------------------------------------------------------------------------------------- device rows ---

/// One cache plane's rows a state reads: rows [0, rows) of `row` bytes each from `ptr`.
pub const View = struct { ptr: u64, row: usize, rows: usize };

/// prefixes.row_views: the cache rows a state of n tokens reads - every layer's latent rows of positions < n, then
/// the index keys of the full-indexer layers, then the MTP layer's latent and index keys of positions < n - 1. DCP: a
/// rank holds positions p % dcp == rank at p / dcp; every rank takes ceil(n / dcp) rows (the most any rank has).
pub fn rowViews(gpa: std.mem.Allocator, st: *const fw.State, c: anytype, n: usize, dcp: usize) ![]View {
    const m = (n + dcp - 1) / dcp;
    const n1 = if (n > 0) n - 1 else 0;
    const mm = (n1 + dcp - 1) / dcp;
    var out: std.ArrayList(View) = .empty;
    errdefer out.deinit(gpa);
    const lat = c.latentWidth() * 2;
    const idx = c.index_dim * 2;
    for (st.kc) |p| try out.append(gpa, .{ .ptr = p, .row = lat, .rows = m });
    for (st.ic) |p| if (p != 0) try out.append(gpa, .{ .ptr = p, .row = idx, .rows = m });
    if (st.mkc != 0) {
        try out.append(gpa, .{ .ptr = st.mkc, .row = lat, .rows = mm });
        if (st.mic != 0) try out.append(gpa, .{ .ptr = st.mic, .row = idx, .rows = mm });
    }
    return out.toOwnedSlice(gpa);
}

/// prefixes.Saved.size: (total bytes, each part's offset) of parts each starting ALIGN-aligned.
pub fn savedSize(views: []const View, offs: ?[]usize) usize {
    var o: usize = 0;
    for (views, 0..) |v, i| {
        if (offs) |x| x[i] = o;
        o += std.mem.alignForward(usize, v.row * v.rows, save_align);
    }
    return o;
}

/// Device buffers the store's states hold, handed back to the next copy instead of freed (torch's caching pool: a
/// cuMemAlloc of GiBs maps fresh pages of GB10's unified memory and a free synchronizes); `trim` gives idle ones
/// back past a byte limit (PromptReuse._trim).
pub const Pool = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    idle: std.ArrayList(cuda.DeviceBuffer) = .empty,
    idle_bytes: usize = 0,

    pub fn get(p: *Pool, len: usize) !Handle {
        // best fit: the smallest idle buffer that holds `len` and is at most twice it
        var pick: ?usize = null;
        for (p.idle.items, 0..) |b, i| {
            if (b.len < len or b.len > 2 * len + (1 << 20)) continue;
            if (pick == null or b.len < p.idle.items[pick.?].len) pick = i;
        }
        if (pick) |i| {
            const b = p.idle.swapRemove(i);
            p.idle_bytes -= b.len;
            return .{ .ptr = b.ptr, .len = b.len };
        }
        const b = try cuda.DeviceBuffer.alloc(p.d, @max(len, 16));
        return .{ .ptr = b.ptr, .len = b.len };
    }

    pub fn put(p: *Pool, h: Handle) void {
        if (h.len == 0) return;
        p.idle.append(p.gpa, .{ .d = p.d, .ptr = h.ptr, .len = h.len }) catch {
            var b: cuda.DeviceBuffer = .{ .d = p.d, .ptr = h.ptr, .len = h.len };
            b.free();
            return;
        };
        p.idle_bytes += h.len;
    }

    fn releaseFn(ctx: *anyopaque, h: Handle) void {
        const p: *Pool = @ptrCast(@alignCast(ctx));
        p.put(h);
    }

    pub fn release(p: *Pool) Release {
        return .{ .ctx = p, .free = releaseFn };
    }

    /// Idle buffers freed while they hold more than `limit` bytes (after `s` drains: queued copies may read them).
    pub fn trim(p: *Pool, limit: usize, s: cuda.Stream) !void {
        if (p.idle_bytes <= limit) return;
        try s.synchronize();
        for (p.idle.items) |*b| b.free();
        p.idle.clearRetainingCapacity();
        p.idle_bytes = 0;
    }

    pub fn deinit(p: *Pool) void {
        for (p.idle.items) |*b| b.free();
        p.idle.deinit(p.gpa);
    }
};

/// prefixes.PromptReuse on the runner's caches: rank 0 picks (`choose`); every rank runs (`run`).
pub const Reuse = struct {
    gpa: std.mem.Allocator,
    r: *rn.Runner,
    plan: Plan,
    store: Store,
    pool: *Pool,
    dcp: usize,

    pub const flush_flag: u32 = 1; // header flag: forget every kept state first

    pub fn init(gpa: std.mem.Allocator, r: *rn.Runner, plan: Plan, budget: u64, entries: usize, loose: bool) !*Reuse {
        const pr = try gpa.create(Reuse);
        errdefer gpa.destroy(pr);
        const pool = try gpa.create(Pool);
        pool.* = .{ .gpa = gpa, .d = r.d };
        const c = r.w.config;
        pr.* = .{ .gpa = gpa, .r = r, .plan = plan, .pool = pool, .dcp = r.set.dcp, .store = Store.init(gpa, budget, entries, loose, c.hidden * 2, c.vocab * 4, pool.release()) };
        return pr;
    }

    pub fn deinit(pr: *Reuse) void {
        pr.r.f.s.synchronize() catch {};
        pr.store.deinit();
        pr.pool.deinit();
        pr.gpa.destroy(pr.pool);
        pr.gpa.destroy(pr);
    }

    fn ops(pr: *Reuse) Ops {
        return .{ .d = pr.r.d, .s = pr.r.f.s };
    }

    const Ops = struct {
        d: *const cuda.Driver,
        s: cuda.Stream,
        fn copy(o: Ops, dst: u64, src: u64, bytes: usize) !void {
            if (bytes == 0) return;
            try o.d.check(o.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, o.s.handle), "cuMemcpyDtoDAsync");
        }
    };

    pub fn views(pr: *Reuse, n: usize) ![]View {
        return rowViews(pr.gpa, &pr.r.st, pr.r.w.config, n, pr.dcp);
    }

    /// prefixes.row_bytes.
    pub fn rowBytes(pr: *Reuse, n: usize) !usize {
        const vs = try pr.views(n);
        defer pr.gpa.free(vs);
        return savedSize(vs, null);
    }

    /// prefixes.save_rows: `e`'s rows copied out of the live caches into one new copy (queued on the stream).
    fn saveRows(pr: *Reuse, e: *Kept) !*Saved {
        const vs = try pr.views(e.n());
        defer pr.gpa.free(vs);
        const offs = try pr.gpa.alloc(usize, vs.len);
        defer pr.gpa.free(offs);
        const total = savedSize(vs, offs);
        const mem = try pr.pool.get(total);
        const sv = try pr.gpa.create(Saved);
        sv.* = .{ .mem = mem, .nbytes = total, .n = e.n(), .refs = 1 };
        const o = pr.ops();
        for (vs, offs) |v, off| try o.copy(mem.ptr + off, v.ptr, v.row * v.rows);
        e.saved = sv;
        return sv;
    }

    /// prefixes.share_rows: `e` (a live prefix of the state `sv` was copied from) reads its rows there.
    fn shareRows(pr: *Reuse, sv: *Saved, e: *Kept) void {
        _ = pr;
        std.debug.assert(e.n() <= sv.n);
        sv.refs += 1;
        e.saved = sv;
    }

    /// prefixes.load_rows: `e`'s saved rows back into the live caches; `e` is live again.
    fn loadRows(pr: *Reuse, e: *Kept) !void {
        const sv = e.saved orelse return error.NotSaved;
        // the parts sit where the copy's own (longer) state put them: offsets from its views
        const big = try pr.views(sv.n);
        defer pr.gpa.free(big);
        const offs = try pr.gpa.alloc(usize, big.len);
        defer pr.gpa.free(offs);
        _ = savedSize(big, offs);
        const vs = try pr.views(e.n());
        defer pr.gpa.free(vs);
        if (vs.len != big.len) return error.SavedRowsMismatch;
        const o = pr.ops();
        for (vs, offs) |v, off| try o.copy(v.ptr, sv.mem.ptr + off, v.row * v.rows);
        pr.store.unsave(e);
    }

    /// The device side of a state read back from disk (--learn): its rows into a new saved copy, its carry, as one
    /// shared state of `ids`. `fill(ctx, views, offs, base, carry)` writes the bytes (the engine's file reader).
    pub fn adopt(pr: *Reuse, ids: []const u32, carry_host: []const u8, rows: anytype) !*Kept {
        const e = try pr.gpa.create(Kept);
        errdefer pr.gpa.destroy(e);
        e.* = .{ .ids = try pr.gpa.dupe(u32, ids), .shared = true, .learned = true };
        errdefer pr.gpa.free(e.ids);
        const vs = try pr.views(ids.len);
        defer pr.gpa.free(vs);
        const offs = try pr.gpa.alloc(usize, vs.len);
        defer pr.gpa.free(offs);
        const total = savedSize(vs, offs);
        const mem = try pr.pool.get(total);
        const sv = try pr.gpa.create(Saved);
        sv.* = .{ .mem = mem, .nbytes = total, .n = ids.len, .refs = 1 };
        e.saved = sv;
        e.carry = try pr.pool.get(carry_host.len);
        try rows.fill(vs, offs, mem.ptr);
        try pr.r.d.check(pr.r.d.api.cuMemcpyHtoD_v2(e.carry.ptr, carry_host.ptr, carry_host.len), "cuMemcpyHtoD");
        return e;
    }

    pub fn flush(pr: *Reuse) !void {
        try pr.r.f.s.synchronize();
        pr.store.clear();
        try pr.pool.trim(0, pr.r.f.s); // torch.cuda.empty_cache()
    }

    /// PromptReuse._take_over: the live states this request overwrites (but `renew`s) - the longest one that fits
    /// the budget copied out once, the shorter ones reading that copy; longer ones that do not fit and states no
    /// longer live dropped. At most one copy a request.
    fn takeOver(pr: *Reuse, prompt: []const u32, begin: usize, hit: ?*Kept, same: ?Same, renew: []const *Kept) !void {
        const store = &pr.store;
        const over = try store.overwritten(pr.gpa, prompt, begin, same);
        defer pr.gpa.free(over);
        var chain: std.ArrayList(*Kept) = .empty;
        defer chain.deinit(pr.gpa);
        for (over) |e| {
            if (isKept(hit, e) or std.mem.indexOfScalar(*Kept, renew, e) != null) continue;
            if (store.inLive(e)) try chain.append(pr.gpa, e) else store.drop(e);
        }
        std.mem.sort(*Kept, chain.items, {}, struct {
            fn longer(_: void, a: *Kept, b: *Kept) bool {
                return a.n() > b.n();
            }
        }.longer);
        var copy: ?*Saved = null;
        for (chain.items) |e| {
            if (!store.has(e)) continue; // dropped making room for the copy
            if (copy) |sv| {
                pr.shareRows(sv, e);
                continue;
            }
            const need = try pr.rowBytes(e.n());
            var protect: std.ArrayList(?*Kept) = .empty;
            defer protect.deinit(pr.gpa);
            try protect.append(pr.gpa, hit);
            try protect.append(pr.gpa, e);
            for (renew) |x| try protect.append(pr.gpa, x);
            if (store.fit(need, protect.items)) {
                copy = try pr.saveRows(e);
            } else {
                store.drop(e);
            }
        }
    }

    /// What `run` reports.
    pub const Result = struct { stats: rn.Stats, kept: []usize, replay: bool };

    const KeepCtx = struct {
        pr: *Reuse,
        prompt: []const u32,
        named: []const u32, // n * 2 + shared
        renewed: []const [2]usize, // (n, shared)
        keep_end: bool,
        mode: u8,
        hit: ?*Kept,
        added: *std.ArrayList(usize),
        inner: ?rn.Hooks,
        failed: ?anyerror = null,

        fn tokens(ctx: *anyopaque, toks: []const u32) bool {
            const k: *KeepCtx = @ptrCast(@alignCast(ctx));
            const h = k.inner orelse return false;
            return h.tokens(h.ctx, toks);
        }

        fn sharedOf(k: *const KeepCtx, n: usize) ?bool {
            var out: ?bool = null;
            for (k.named) |x| if (x >> 1 == n) {
                out = (x & 1) != 0;
            };
            for (k.renewed) |x| if (x[0] == n) {
                out = (out orelse false) or x[1] != 0;
            };
            return out;
        }

        fn renewedHas(k: *const KeepCtx, n: usize) bool {
            for (k.renewed) |x| if (x[0] == n) return true;
            return false;
        }

        fn namedHas(k: *const KeepCtx, n: usize) bool {
            for (k.named) |x| if (x >> 1 == n) return true;
            return false;
        }

        /// PromptReuse.run's keep(n, head).
        fn keep(ctx: *anyopaque, n: usize, head: ?u64) anyerror!void {
            const k: *KeepCtx = @ptrCast(@alignCast(ctx));
            const pr = k.pr;
            const L = k.prompt.len;
            if (n < L and !k.namedHas(n) and !k.renewedHas(n)) return;
            if (n == L and !k.keep_end and !k.renewedHas(n)) return;
            const c = pr.r.w.config;
            const e = try pr.gpa.create(Kept);
            e.* = .{ .ids = try pr.gpa.dupe(u32, k.prompt[0..n]), .mode = k.mode, .shared = k.sharedOf(n) orelse false };
            const o = pr.ops();
            e.carry = try pr.pool.get(c.hidden * 2);
            try o.copy(e.carry.ptr, pr.r.carry.ptr, c.hidden * 2);
            if (head != null and n == L) {
                const V = pr.r.w.vocab_part;
                const h = try pr.pool.get(V * 4);
                e.head = h;
                try o.copy(h.ptr, head.?, V * 4);
            }
            const protect = [_]?*Kept{k.hit};
            if (pr.store.remember(e, &protect)) {
                try k.added.append(pr.gpa, n);
            } else {
                pr.store.destroy(e);
            }
        }
    };

    /// PromptReuse.run (every rank): take the live caches over for this prompt, restore its resume point, run the
    /// request (Runner.run from `begin`, cut at `stops`) keeping the states rank 0 named and, with `keep_end`, the
    /// prompt's end. Kept states of this prompt's own ids at its cut points past `begin` (and at its end) are
    /// rewritten by it, cut alike: it keeps them again from its own rows (renew) - neither saved nor dropped - so a
    /// cold reference ("draft": false) of a kept prompt leaves it resumable and replayable.
    pub fn run(pr: *Reuse, prompt: []const u32, begin: usize, stops: []const usize, keeps: []const u32, flags: u32, keep_end: bool, mode: u8, base: rn.Gen, out: *std.ArrayList(u32)) !Result {
        const store = &pr.store;
        if (flags & flush_flag != 0) try pr.flush();
        const L = prompt.len;
        var hit: ?*Kept = null;
        if (begin > 0) {
            hit = store.named(prompt, begin) orelse {
                std.log.err("rank {d} has no kept prompt state of the {d} tokens rank 0 resumes from", .{ pr.r.w.rank, begin });
                return error.ReuseOutOfStep;
            };
        }
        const pts = try pr.plan.points(pr.gpa, prompt);
        defer pr.gpa.free(pts);
        const same: ?Same = if (hit != null and hit.?.saved != null) .{ .mode = hit.?.mode, .points = pts, .begin = begin } else null;
        // cut: the stops inside (begin, L), and L itself when anything is prefilled
        var renew: std.ArrayList(*Kept) = .empty;
        defer renew.deinit(pr.gpa);
        var renewed: std.ArrayList([2]usize) = .empty;
        defer renewed.deinit(pr.gpa);
        for (store.entries.items) |e| {
            if (isKept(hit, e)) continue;
            const n = e.n();
            const in_cut = (begin < n and n < L and contains(stops, n)) or (n == L and begin < L);
            if (in_cut and n <= L and std.mem.eql(u32, prompt[0..n], e.ids)) {
                try renew.append(pr.gpa, e);
                try renewed.append(pr.gpa, .{ n, @intFromBool(e.shared) });
            }
        }
        try pr.takeOver(prompt, begin, hit, same, renew.items);
        if (hit) |h| {
            if (h.saved) |copy| {
                try pr.loadRows(h);
                // its shorter states that shared the copy: live again
                for (store.entries.items) |e| {
                    if (savedIs(e, copy) and e.n() <= begin) store.unsave(e);
                }
            }
            try pr.ops().copy(pr.r.carry.ptr, h.carry.ptr, pr.r.w.config.hidden * 2);
            store.touch(h);
        }
        try store.setLive(prompt[0..begin]); // rows past it are about to change
        try pr.pool.trim(@max(@as(usize, @intCast(store.budget)), 1 << 30), pr.r.f.s);
        var added: std.ArrayList(usize) = .empty;
        defer added.deinit(pr.gpa);
        var kc: KeepCtx = .{ .pr = pr, .prompt = prompt, .named = keeps, .renewed = renewed.items, .keep_end = keep_end, .mode = mode, .hit = hit, .added = &added, .inner = base.hooks };
        const replay = hit != null and begin == L;
        var g = base;
        g.begin = begin;
        g.cuts = stops;
        g.replay = if (replay) hit.?.head.?.ptr else null;
        g.reset = false;
        g.hooks = .{ .ctx = &kc, .tokens = KeepCtx.tokens, .keep = KeepCtx.keep };
        const st = try pr.r.run(prompt, g, out);
        try store.setLive(prompt);
        return .{ .stats = st, .kept = try pr.gpa.dupe(usize, added.items), .replay = replay };
    }

    /// Bytes a rank may give kept states (Runner.kept_budget): at most `wanted`, at most half of what is free past
    /// the reserve (TF_GLM53_CACHE_RESERVE_GB, default 6, at most half of what is free) now.
    pub fn keptBudget(free: u64, wanted: u64) u64 {
        const reserve_gb: f64 = if (std.c.getenv("TF_GLM53_CACHE_RESERVE_GB")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 6) else 6;
        const reserve = @min(reserve_gb * (1 << 30), @as(f64, @floatFromInt(free)) / 2);
        const room = (@as(f64, @floatFromInt(free)) - reserve) / 2;
        return @intFromFloat(@max(0, @min(@as(f64, @floatFromInt(wanted)), room)));
    }
};

// ------------------------------------------------------------------------------------------------------ tests ---

const TestRelease = struct {
    freed: usize = 0,
    fn free(ctx: *anyopaque, h: Handle) void {
        const t: *TestRelease = @ptrCast(@alignCast(ctx));
        t.freed += h.len;
    }
};

fn testKept(gpa: std.mem.Allocator, ids: []const u32, shared: bool) !*Kept {
    const e = try gpa.create(Kept);
    e.* = .{ .ids = try gpa.dupe(u32, ids), .carry = .{ .ptr = 1, .len = 8 }, .shared = shared };
    return e;
}

test "keep points as PromptPlan.points" {
    const gpa = std.testing.allocator;
    const U = 9;
    const A = 7;
    const T = 5;
    const plan: Plan = .{ .ids = .{ .user = U, .assistant = A, .think = T }, .gap = 4, .system_min = 3 };
    //            0  1  2  3  4  5  6  7  8  9 10 11 12 13
    const p = [_]u32{ 1, 2, 3, U, 4, A, T, 6, U, 1, 2, A, T, 3 };
    const pts = try plan.points(gpa, &p);
    defer gpa.free(pts);
    // the first <|user|> at 3; an opener after it at 5 (+<think> -> 7): 7 - 3 = 4 >= gap; at 11 -> 13: 6 >= 4
    try std.testing.expectEqualSlices(usize, &.{ 3, 7, 13 }, pts);
    try std.testing.expectEqual(@as(?usize, 3), plan.systemPoint(pts, &p));
    const short: Plan = .{ .ids = plan.ids, .gap = 4, .system_min = 5 };
    const pts2 = try short.points(gpa, &p);
    defer gpa.free(pts2);
    try std.testing.expectEqualSlices(usize, &.{ 7, 13 }, pts2); // the system block is too short to be a point
    try std.testing.expectEqual(@as(?usize, null), short.systemPoint(pts2, &p));
}

test "cut_and_keep: the last stop, the system block and the fork, sorted as n * 2 + shared" {
    const gpa = std.testing.allocator;
    var rel: TestRelease = .{};
    var store = Store.init(gpa, 1 << 30, 32, false, 8, 16, .{ .ctx = &rel, .free = TestRelease.free });
    defer store.deinit();
    const U = 9;
    const A = 7;
    const plan: Plan = .{ .ids = .{ .user = U, .assistant = A }, .gap = 2, .system_min = 2 };
    // 8, not 7, before the last opener: token 7 is A itself (Python agrees: [..., 7, A] cuts at 14)
    const p = [_]u32{ 1, 2, U, 3, A, 4, 4, U, 5, A, 6, 6, U, 8, A };
    const pts = try plan.points(gpa, &p);
    defer gpa.free(pts);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5, 10, 15 }, pts);
    // a kept state sharing the first 11 tokens: the fork is the last stop <= 11 (10), already the last stop
    const e = try testKept(gpa, p[0..11], false);
    try std.testing.expect(store.remember(e, &.{}));
    const c = try cutAndKeep(gpa, plan, &store, &p, pts, 0, true);
    defer c.deinit(gpa);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5, 10 }, c.stops);
    try std.testing.expectEqualSlices(u32, &.{ 2 * 2 + 1, 10 * 2 }, c.keeps);
    const cold = try cutAndKeep(gpa, plan, &store, &p, pts, 0, false);
    defer cold.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), cold.keeps.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 5, 10 }, cold.stops);
}

test "the store: best resumes at points, replays need a head, victims are superseded first" {
    const gpa = std.testing.allocator;
    var rel: TestRelease = .{};
    var store = Store.init(gpa, 1000, 3, false, 100, 200, .{ .ctx = &rel, .free = TestRelease.free });
    defer store.deinit();
    const p = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try store.setLive(&p);
    const a = try testKept(gpa, p[0..3], true);
    const b = try testKept(gpa, p[0..6], false);
    try std.testing.expect(store.remember(a, &.{}));
    try std.testing.expect(store.remember(b, &.{}));
    try std.testing.expectEqual(b, store.best(&p, &.{ 3, 6 }, 0).?);
    try std.testing.expectEqual(a, store.best(&p, &.{3}, 0).?); // 6 is not one of this prompt's points
    try std.testing.expectEqual(@as(?*Kept, null), store.best(p[0..6], &.{6}, 0)); // whole prompt: no head
    try std.testing.expect(store.superseded(a));
    try std.testing.expectEqual(a, store.victim(&.{})); // superseded (shared: still before the LRU rule)
    const c = try testKept(gpa, &.{ 9, 9 }, false);
    try std.testing.expect(store.remember(c, &.{}));
    const d = try testKept(gpa, &.{ 9, 8 }, false);
    try std.testing.expect(store.remember(d, &.{})); // the cap (3): the superseded `a` goes
    try std.testing.expect(!store.has(a));
    try std.testing.expectEqual(@as(usize, 3), store.entries.items.len);
    // live prefix check
    try std.testing.expect(store.usable(b));
    try store.setLive(&.{ 1, 2 });
    try std.testing.expect(!store.usable(b));
}

test "overwritten: saved states stay, prefixes up to begin stay" {
    const gpa = std.testing.allocator;
    var rel: TestRelease = .{};
    var store = Store.init(gpa, 1 << 20, 8, false, 8, 8, .{ .ctx = &rel, .free = TestRelease.free });
    defer store.deinit();
    const p = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const a = try testKept(gpa, p[0..2], true);
    const b = try testKept(gpa, p[0..4], false);
    const c = try testKept(gpa, &.{ 7, 7, 7 }, false);
    try std.testing.expect(store.remember(a, &.{}));
    try std.testing.expect(store.remember(b, &.{}));
    try std.testing.expect(store.remember(c, &.{}));
    const over = try store.overwritten(gpa, &p, 2, null);
    defer gpa.free(over);
    try std.testing.expectEqual(@as(usize, 2), over.len);
    try std.testing.expectEqual(c, over[0]); // newest first
    try std.testing.expectEqual(b, over[1]);
}

test "slot mode (SlotPrompts): states live in their slot's rows, one slot's ids never stand for another's" {
    const gpa = std.testing.allocator;
    var rel: TestRelease = .{};
    var store = Store.init(gpa, 1 << 20, 8, false, 8, 8, .{ .ctx = &rel, .free = TestRelease.free });
    defer store.deinit();
    try store.setSlots(3);
    const p = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try store.setLiveSlot(0, &p);
    try store.setLiveSlot(2, p[0..4]);
    const a = try testKept(gpa, p[0..4], false);
    a.slot = 0;
    const b = try testKept(gpa, p[0..4], false);
    b.slot = 2;
    try std.testing.expect(store.remember(a, &.{}));
    try std.testing.expect(store.remember(b, &.{})); // the same ids in another slot: a second state
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
    try std.testing.expectEqual(b, store.namedIn(&p, 4, 2).?);
    try std.testing.expectEqual(a, store.namedIn(&p, 4, 0).?);
    try std.testing.expectEqual(@as(?*Kept, null), store.namedIn(&p, 4, 1));
    try std.testing.expect(store.slotTick(2) > store.slotTick(0));
    try std.testing.expectEqual(@as(u64, 0), store.slotTick(1));
    // slot 0 takes another conversation: its state is no longer in its rows (never saved out of them)
    try store.setLiveSlot(0, &.{ 9, 9 });
    try std.testing.expect(!store.usable(a));
    try std.testing.expect(store.usable(b));
    try std.testing.expectEqual(b, store.best(&p, &.{4}, 0).?);
    // what a request resumed in slot 2 at 4 overwrites there: nothing (its own prefix); at 2: the state at 4
    const keep4 = try store.overwrittenSlot(gpa, 2, &p, 4, null);
    defer gpa.free(keep4);
    try std.testing.expectEqual(@as(usize, 0), keep4.len);
    const over2 = try store.overwrittenSlot(gpa, 2, &p, 2, null);
    defer gpa.free(over2);
    try std.testing.expectEqual(@as(usize, 1), over2.len);
    try std.testing.expectEqual(b, over2[0]);
    // the same state again in the same slot replaces it
    const b2 = try testKept(gpa, p[0..4], false);
    b2.slot = 2;
    try std.testing.expect(store.remember(b2, &.{}));
    try std.testing.expect(!store.has(b));
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
}
