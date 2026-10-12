//! As-served scoring of GLM-5.3 against a reference model: KL(reference || served) at temperature 1 and top-1
//! agreement over fixed token rows, per row and per named set. Host code only (no device): the file formats, each
//! rank's share of a position's sums, and the merge of the ranks' shares. cuda_score.zig runs the rows through the
//! engine; tools/glm53/score_reference.py writes the reference.
//!
//! The head is split over the ranks (rank r holds the logits of ids [vocab_off, vocab_off + vocab_part)), and no rank
//! ever sees the others' logits. So each rank writes, for every position, a few sums over its own share (`Part`), and
//! `combine` adds the four files up. Nothing is exchanged between the ranks while scoring.
//!
//! The reference keeps, for every position, its `topk` most likely tokens (id and log-probability, most likely
//! first) and the probability mass of all the others together (`tail`). With p the reference and q the served
//! distribution, P and Q their masses outside the kept tokens:
//!     kl_lo = sum over kept (p ln p/q) + P ln(P/Q)        the KL of the two distributions with the tail merged
//!     kl_hi = sum over kept (p ln p/q) + P (ln p_K - ln q_min)
//! kl_lo <= KL(p || q) <= kl_hi. The lower bound is the log-sum inequality (merging outcomes never raises a KL);
//! the upper bound holds because no tail token is likelier than the last kept one (p_K) and none is less likely
//! under q than q's least likely token (q_min). topk = the vocabulary makes both exact.
//!
//! Files (little endian):
//!   rows / reference  "GLM53RF1": Head, sets [n_sets] SetRec, rows [n_rows] RowRec, tokens [sum of len] u32, then
//!                     (topk > 0) for every position PosHead + ids [topk] u32 + logp [topk] f32. A row of len tokens
//!                     has len - 1 positions: position t is the distribution of token t + 1 given tokens 0..t.
//!                     topk = 0: the rows alone (what the reference is computed from).
//!   partials          "GLM53PT1": PartHead, then [n_pos] Part.
const std = @import("std");

pub const ref_magic = "GLM53RF1".*;
pub const part_magic = "GLM53PT1".*;
pub const version: u32 = 1;

pub const Head = extern struct {
    magic: [8]u8 = ref_magic,
    version: u32 = version,
    vocab: u32,
    topk: u32,
    n_rows: u32,
    n_sets: u32,
    reserved: u32 = 0,
    n_pos: u64,
    /// CRC-32 (zlib's) of the tokens' bytes: ties partials and references to the rows they were made from
    tokens_digest: u64,
    /// CRC-32 of the position records' bytes (0 when topk = 0)
    body_digest: u64,
    pad: [8]u8 = @splat(0),
};

pub const SetRec = extern struct { name: [24]u8, first_row: u32, n_rows: u32 };
pub const RowRec = extern struct { len: u32, set: u32, first_pos: u64 };
/// What a position keeps besides its top tokens: the reference mass outside them, the reference log-probability of
/// the row's real next token, the reference entropy in nats.
pub const PosHead = extern struct { tail: f32, label_logp: f32, entropy: f32, reserved: u32 = 0 };

pub const PartHead = extern struct {
    magic: [8]u8 = part_magic,
    version: u32 = version,
    rank: u32,
    world: u32,
    vocab_off: u32,
    vocab_part: u32,
    topk: u32,
    n_pos: u64,
    tokens_digest: u64,
    body_digest: u64,
    /// rows a prompt chunk (0: the served prompt path's own chunks)
    chunk_rows: u32,
    pad: [12]u8 = @splat(0),
};

/// One rank's share of one position. m is the largest logit of the share; every sum is over the share's ids.
pub const Part = extern struct {
    /// sum of exp(logit - m)
    s: f64 = 0,
    /// sum over the kept reference tokens in the share of p * logit
    a: f64 = 0,
    /// sum over the kept reference tokens in the share of exp(logit - m)
    t: f64 = 0,
    m: f32 = -std.math.inf(f32),
    /// the smallest logit of the share
    lo: f32 = std.math.inf(f32),
    /// the id of the first largest logit
    arg: u32 = 0,
    /// the real next token's logit when the share holds it, else NaN
    label: f32 = std.math.nan(f32),
};

comptime {
    std.debug.assert(@sizeOf(Head) == 64);
    std.debug.assert(@sizeOf(SetRec) == 32);
    std.debug.assert(@sizeOf(RowRec) == 16);
    std.debug.assert(@sizeOf(PosHead) == 16);
    std.debug.assert(@sizeOf(PartHead) == 72);
    std.debug.assert(@sizeOf(Part) == 40);
}

/// zlib.crc32 of `bytes` (what the Python writer can compute over half a gigabyte in a second).
pub fn digest(bytes: []const u8) u64 {
    return std.hash.Crc32.hash(bytes);
}

pub fn posBytes(topk: usize) usize {
    return @sizeOf(PosHead) + topk * 8;
}

fn rd(comptime T: type, bytes: []const u8, off: usize) T {
    var v: T = undefined;
    @memcpy(std.mem.asBytes(&v), bytes[off..][0..@sizeOf(T)]);
    return v;
}

/// A rows or reference file held in memory (`bytes` is the whole file).
pub const Ref = struct {
    bytes: []const u8,
    head: Head,
    sets_off: usize,
    rows_off: usize,
    tokens_off: usize,
    pos_off: usize,

    pub const Error = error{ BadReference, ReferenceDamaged };

    /// Checks the layout and both digests (`check_body` false skips the position records' digest: half a gigabyte
    /// hashed at every open is only worth it once).
    pub fn open(bytes: []const u8, check_body: bool) Error!Ref {
        if (bytes.len < @sizeOf(Head)) return error.BadReference;
        const h = rd(Head, bytes, 0);
        if (!std.mem.eql(u8, &h.magic, &ref_magic) or h.version != version) return error.BadReference;
        if (h.vocab == 0 or h.topk > h.vocab or h.n_sets == 0 or h.n_rows == 0) return error.BadReference;
        const sets_off: usize = @sizeOf(Head);
        const rows_off = sets_off + @as(usize, h.n_sets) * @sizeOf(SetRec);
        const tokens_off = rows_off + @as(usize, h.n_rows) * @sizeOf(RowRec);
        if (bytes.len < tokens_off) return error.BadReference;
        var r: Ref = .{ .bytes = bytes, .head = h, .sets_off = sets_off, .rows_off = rows_off, .tokens_off = tokens_off, .pos_off = 0 };
        var n_tok: usize = 0;
        var n_pos: u64 = 0;
        for (0..h.n_rows) |i| {
            const rw = r.row(i);
            if (rw.len < 2 or rw.set >= h.n_sets or rw.first_pos != n_pos) return error.BadReference;
            n_tok += rw.len;
            n_pos += rw.len - 1;
        }
        if (n_pos != h.n_pos) return error.BadReference;
        var covered: usize = 0;
        for (0..h.n_sets) |i| {
            const s = r.set(i);
            if (s.first_row != covered or s.n_rows == 0) return error.BadReference;
            for (s.first_row..s.first_row + s.n_rows) |ri| {
                if (ri >= h.n_rows or r.row(ri).set != i) return error.BadReference;
            }
            covered += s.n_rows;
        }
        if (covered != h.n_rows) return error.BadReference;
        r.pos_off = tokens_off + n_tok * 4;
        const want = r.pos_off + if (h.topk > 0) @as(usize, @intCast(n_pos)) * posBytes(h.topk) else 0;
        if (bytes.len != want) return error.BadReference;
        if (digest(bytes[tokens_off..r.pos_off]) != h.tokens_digest) return error.ReferenceDamaged;
        for (0..n_tok) |i| if (rd(u32, bytes, tokens_off + i * 4) >= h.vocab) return error.BadReference;
        if (h.topk > 0 and check_body and digest(bytes[r.pos_off..]) != h.body_digest) return error.ReferenceDamaged;
        return r;
    }

    pub fn set(r: *const Ref, i: usize) SetRec {
        return rd(SetRec, r.bytes, r.sets_off + i * @sizeOf(SetRec));
    }

    pub fn setName(s: *const SetRec) []const u8 {
        return std.mem.sliceTo(&s.name, 0);
    }

    pub fn row(r: *const Ref, i: usize) RowRec {
        return rd(RowRec, r.bytes, r.rows_off + i * @sizeOf(RowRec));
    }

    /// Row i's tokens, copied out (the file's bytes are not aligned).
    pub fn tokens(r: *const Ref, gpa: std.mem.Allocator, i: usize) ![]u32 {
        const rw = r.row(i);
        const out = try gpa.alloc(u32, rw.len);
        const off = r.tokens_off + @as(usize, @intCast(rw.first_pos + i)) * 4; // tokens before row i: positions + rows
        @memcpy(std.mem.sliceAsBytes(out), r.bytes[off..][0 .. out.len * 4]);
        return out;
    }

    pub fn hasPositions(r: *const Ref) bool {
        return r.head.topk > 0;
    }

    /// Position `pos`'s record (PosHead, ids, logp) as raw bytes.
    pub fn posRec(r: *const Ref, pos: u64) []const u8 {
        const n = posBytes(r.head.topk);
        return r.bytes[r.pos_off + @as(usize, @intCast(pos)) * n ..][0..n];
    }
};

/// A position record's fields.
pub const Pos = struct {
    head: PosHead,
    ids: []const u8, // [topk] u32
    logp: []const u8, // [topk] f32

    pub fn of(rec: []const u8, topk: usize) Pos {
        return .{ .head = rd(PosHead, rec, 0), .ids = rec[@sizeOf(PosHead)..][0 .. topk * 4], .logp = rec[@sizeOf(PosHead) + topk * 4 ..][0 .. topk * 4] };
    }

    pub fn id(p: Pos, j: usize) u32 {
        return rd(u32, p.ids, j * 4);
    }

    pub fn lp(p: Pos, j: usize) f32 {
        return rd(f32, p.logp, j * 4);
    }
};

/// One rank's share of a position: `lg` its logits (ids vocab_off ..), `pos` the reference's record, `label` the
/// row's real next token.
pub fn part(lg: []const f32, vocab_off: usize, pos: Pos, topk: usize, label: u32) Part {
    var p: Part = .{};
    var arg: usize = 0;
    for (lg, 0..) |v, i| {
        if (v > p.m or (std.math.isNan(v) and !std.math.isNan(p.m))) { // the first largest; a NaN above every number
            p.m = v;
            arg = i;
        }
        if (v < p.lo) p.lo = v;
    }
    p.arg = @intCast(vocab_off + arg);
    const m: f64 = p.m;
    for (lg) |v| {
        const d = @as(f64, v) - m;
        if (d > -80) p.s += @exp(d); // exp(-80) is 2e-35 of the largest term: under an f64's last bit of the sum
    }
    for (0..topk) |j| {
        const tok = pos.id(j);
        if (tok < vocab_off or tok >= vocab_off + lg.len) continue;
        const v: f64 = lg[tok - vocab_off];
        p.a += @exp(@as(f64, pos.lp(j))) * v;
        p.t += @exp(v - m);
    }
    if (label >= vocab_off and label < vocab_off + lg.len) p.label = lg[label - vocab_off];
    return p;
}

/// Debugging: a rank's `dump_n` largest logits of a position (ids global, largest first, ties to the lower id), for
/// the rows a run asks to dump. A dump file is these records back to back; tools/glm53/score_dump.py merges the ranks'.
pub const dump_n = 16;
pub const DumpRec = extern struct { row: u32, pos: u32, ids: [dump_n]u32, lg: [dump_n]f32 };

pub fn topLocal(lg: []const f32, vocab_off: usize, row: u32, pos: u32) DumpRec {
    var d: DumpRec = .{ .row = row, .pos = pos, .ids = @splat(0), .lg = @splat(-std.math.inf(f32)) };
    for (lg, 0..) |v, i| {
        if (!(v > d.lg[dump_n - 1])) continue;
        var j: usize = dump_n - 1;
        while (j > 0 and v > d.lg[j - 1]) : (j -= 1) {
            d.lg[j] = d.lg[j - 1];
            d.ids[j] = d.ids[j - 1];
        }
        d.lg[j] = v;
        d.ids[j] = @intCast(vocab_off + i);
    }
    return d;
}

/// One position's outcome from every rank's share.
pub const Outcome = struct {
    kl_lo: f64,
    kl_hi: f64,
    /// the served engine's most likely token (the lowest rank among equal maxima, as the head picks)
    served_top: u32,
    agree: bool,
    /// ln q(real next token); NaN when no rank's share held it
    served_label_logp: f64,
    tail: f64,
};

pub fn outcome(parts: []const Part, pos: Pos, topk: usize) Outcome {
    var best: usize = 0;
    var lo: f64 = std.math.inf(f64);
    for (parts, 0..) |p, r| {
        if (p.m > parts[best].m) best = r;
        if (p.lo < lo) lo = p.lo;
    }
    const big: f64 = parts[best].m;
    var z: f64 = 0;
    var a: f64 = 0;
    var t: f64 = 0;
    var label = std.math.nan(f64);
    for (parts) |p| {
        const scale = @exp(@as(f64, p.m) - big);
        z += p.s * scale;
        t += p.t * scale;
        a += p.a;
        if (!std.math.isNan(p.label)) label = p.label;
    }
    const log_z = big + @log(z);
    var p_top: f64 = 0;
    var plogp: f64 = 0;
    for (0..topk) |j| {
        const l: f64 = pos.lp(j);
        const pj = @exp(l);
        p_top += pj;
        plogp += pj * l;
    }
    const kept = plogp - (a - p_top * log_z);
    const tail: f64 = @max(@as(f64, pos.head.tail), 0);
    var kl_lo = kept;
    var kl_hi = kept;
    if (tail > 0) {
        const q_tail = @max(1 - t / z, 1e-300);
        kl_lo += tail * @log(tail / q_tail);
        const last: f64 = if (topk > 0) pos.lp(topk - 1) else 0;
        kl_hi += tail * (last - (lo - log_z));
        kl_hi = @max(kl_hi, kl_lo);
    }
    const top = parts[best].arg;
    return .{ .kl_lo = kl_lo, .kl_hi = kl_hi, .served_top = top, .agree = topk > 0 and top == pos.id(0), .served_label_logp = label - log_z, .tail = tail };
}

/// A set's (or the whole run's) totals.
pub const Totals = struct {
    n: usize = 0,
    kl_lo: f64 = 0,
    kl_hi: f64 = 0,
    agree: usize = 0,
    served_nll: f64 = 0,
    ref_nll: f64 = 0,
    tail: f64 = 0,
    median: f64 = 0,
    p90: f64 = 0,

    pub fn mean(t: Totals, x: f64) f64 {
        return if (t.n == 0) 0 else x / @as(f64, @floatFromInt(t.n));
    }
};

pub const Summary = struct {
    all: Totals,
    sets: []Totals,
    /// kl_lo of every row (mean over its positions)
    rows: []f64,

    pub fn deinit(s: *Summary, gpa: std.mem.Allocator) void {
        gpa.free(s.sets);
        gpa.free(s.rows);
    }
};

pub const CombineError = error{ BadPartials, PartialsMismatch, OutOfMemory };

/// The ranks' partial files (any order; each rank once) against the reference.
pub fn combine(gpa: std.mem.Allocator, ref: *const Ref, files: []const []const u8) CombineError!Summary {
    const h = ref.head;
    if (!ref.hasPositions() or files.len == 0) return error.BadPartials;
    const world = files.len;
    const order = try gpa.alloc(usize, world);
    defer gpa.free(order);
    @memset(order, world);
    var covered: usize = 0;
    var chunk_rows: u32 = 0;
    for (files, 0..) |f, i| {
        if (f.len < @sizeOf(PartHead)) return error.BadPartials;
        const ph = rd(PartHead, f, 0);
        if (!std.mem.eql(u8, &ph.magic, &part_magic) or ph.version != version) return error.BadPartials;
        if (ph.world != world or ph.rank >= world or order[ph.rank] != world) return error.PartialsMismatch;
        if (ph.n_pos != h.n_pos or ph.topk != h.topk or ph.tokens_digest != h.tokens_digest or ph.body_digest != h.body_digest) return error.PartialsMismatch;
        if (f.len != @sizeOf(PartHead) + @as(usize, @intCast(h.n_pos)) * @sizeOf(Part)) return error.BadPartials;
        if (i == 0) chunk_rows = ph.chunk_rows else if (ph.chunk_rows != chunk_rows) return error.PartialsMismatch;
        order[ph.rank] = i;
        covered += ph.vocab_part;
    }
    for (order, 0..) |fi, r| { // the shares tile the vocabulary in rank order
        const ph = rd(PartHead, files[fi], 0);
        var off: usize = 0;
        for (order[0..r]) |fj| off += rd(PartHead, files[fj], 0).vocab_part;
        if (ph.vocab_off != off) return error.PartialsMismatch;
    }
    if (covered != h.vocab) return error.PartialsMismatch;

    const parts = try gpa.alloc(Part, world);
    defer gpa.free(parts);
    const sets = try gpa.alloc(Totals, h.n_sets);
    errdefer gpa.free(sets);
    @memset(sets, .{});
    const rows = try gpa.alloc(f64, h.n_rows);
    errdefer gpa.free(rows);
    const kls = try gpa.alloc(f64, @intCast(h.n_pos));
    defer gpa.free(kls);
    var all: Totals = .{};
    for (0..h.n_rows) |ri| {
        const rw = ref.row(ri);
        const st = &sets[rw.set];
        var row_kl: f64 = 0;
        for (0..rw.len - 1) |j| {
            const pi: usize = @intCast(rw.first_pos + j);
            for (parts, order) |*p, fi| p.* = rd(Part, files[fi], @sizeOf(PartHead) + pi * @sizeOf(Part));
            const pos = Pos.of(ref.posRec(pi), h.topk);
            const o = outcome(parts, pos, h.topk);
            kls[pi] = o.kl_lo;
            row_kl += o.kl_lo;
            for ([_]*Totals{ st, &all }) |t| {
                t.n += 1;
                t.kl_lo += o.kl_lo;
                t.kl_hi += o.kl_hi;
                t.agree += @intFromBool(o.agree);
                t.served_nll -= o.served_label_logp;
                t.ref_nll -= pos.head.label_logp;
                t.tail += o.tail;
            }
        }
        rows[ri] = row_kl / @as(f64, @floatFromInt(rw.len - 1));
    }
    for (0..h.n_sets) |si| { // a set's positions are contiguous: its rows are
        const s = ref.set(si);
        const first: usize = @intCast(ref.row(s.first_row).first_pos);
        quantiles(kls[first..][0..sets[si].n], &sets[si]);
    }
    quantiles(kls, &all); // last: it sorts across the sets
    return .{ .all = all, .sets = sets, .rows = rows };
}

fn quantiles(kl: []f64, t: *Totals) void {
    if (kl.len == 0) return;
    std.mem.sort(f64, kl, {}, std.sort.asc(f64));
    t.median = kl[(kl.len - 1) / 2];
    t.p90 = kl[@min(kl.len - 1, kl.len * 9 / 10)];
}

/// The summary as text (one line a set) and, appended to `json`, as one JSON object.
pub fn report(gpa: std.mem.Allocator, ref: *const Ref, s: *const Summary, text: *std.ArrayList(u8), json: *std.ArrayList(u8)) !void {
    try text.print(gpa, "{s:<12} {s:>8} {s:>10} {s:>10} {s:>8} {s:>10} {s:>10} {s:>9} {s:>9} {s:>10}\n", .{ "set", "tokens", "KL", "KL upper", "top-1", "median", "p90", "ppl", "ref ppl", "tail mass" });
    try json.print(gpa, "{{\"topk\": {d}, \"vocab\": {d}, \"sets\": [", .{ ref.head.topk, ref.head.vocab });
    for (0..s.sets.len + 1) |i| {
        const all = i == s.sets.len;
        const t = if (all) s.all else s.sets[i];
        var sr: SetRec = undefined;
        if (!all) sr = ref.set(i);
        const name: []const u8 = if (all) "all" else Ref.setName(&sr);
        const top1 = t.mean(@floatFromInt(t.agree));
        try text.print(gpa, "{s:<12} {d:>8} {d:>10.6} {d:>10.6} {d:>8.4} {d:>10.6} {d:>10.6} {d:>9.4} {d:>9.4} {e:>10.2}\n", .{ name, t.n, t.mean(t.kl_lo), t.mean(t.kl_hi), top1, t.median, t.p90, @exp(t.mean(t.served_nll)), @exp(t.mean(t.ref_nll)), t.mean(t.tail) });
        try json.print(gpa, "{s}{{\"set\": \"{s}\", \"tokens\": {d}, \"kl\": {d:.8}, \"kl_upper\": {d:.8}, \"top1\": {d:.6}, \"median\": {d:.8}, \"p90\": {d:.8}, \"ppl\": {d:.6}, \"ref_ppl\": {d:.6}, \"tail\": {e:.4}}}", .{ if (i > 0) ", " else "", name, t.n, t.mean(t.kl_lo), t.mean(t.kl_hi), top1, t.median, t.p90, @exp(t.mean(t.served_nll)), @exp(t.mean(t.ref_nll)), t.mean(t.tail) });
    }
    try json.appendSlice(gpa, "], \"rows\": [");
    for (s.rows, 0..) |v, i| try json.print(gpa, "{s}{d:.8}", .{ if (i > 0) ", " else "", v });
    try json.appendSlice(gpa, "]}\n");
}

/// A rows / reference file from its parts (the tools' and the tests' writer). `pos` is every position's record back
/// to back (empty with topk 0).
pub fn build(gpa: std.mem.Allocator, vocab: u32, topk: u32, set_names: []const []const u8, set_of_row: []const u32, row_tokens: []const []const u32, pos: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var n_pos: u64 = 0;
    for (row_tokens) |t| n_pos += t.len - 1;
    var h: Head = .{ .vocab = vocab, .topk = topk, .n_rows = @intCast(row_tokens.len), .n_sets = @intCast(set_names.len), .n_pos = n_pos, .tokens_digest = 0, .body_digest = 0 };
    try out.appendSlice(gpa, std.mem.asBytes(&h));
    var first: u32 = 0;
    for (set_names, 0..) |name, si| {
        var s: SetRec = .{ .name = @splat(0), .first_row = first, .n_rows = 0 };
        if (name.len == 0 or name.len > 23) return error.BadReference;
        @memcpy(s.name[0..name.len], name);
        for (set_of_row) |x| s.n_rows += @intFromBool(x == si);
        first += s.n_rows;
        try out.appendSlice(gpa, std.mem.asBytes(&s));
    }
    var at: u64 = 0;
    for (row_tokens, set_of_row) |t, si| {
        const r: RowRec = .{ .len = @intCast(t.len), .set = si, .first_pos = at };
        at += t.len - 1;
        try out.appendSlice(gpa, std.mem.asBytes(&r));
    }
    const tok0 = out.items.len;
    for (row_tokens) |t| try out.appendSlice(gpa, std.mem.sliceAsBytes(t));
    h.tokens_digest = digest(out.items[tok0..]);
    if (topk > 0) {
        if (pos.len != n_pos * posBytes(topk)) return error.BadReference;
        h.body_digest = digest(pos);
        try out.appendSlice(gpa, pos);
    }
    @memcpy(out.items[0..@sizeOf(Head)], std.mem.asBytes(&h));
    return out.toOwnedSlice(gpa);
}

/// A reference position record from full log-probabilities (the tests' reference; the Python writer does the same):
/// the topk likeliest ids (ties to the lower id), the rest as tail mass.
pub fn posFromLogp(gpa: std.mem.Allocator, logp: []const f64, topk: usize, label: u32, out: *std.ArrayList(u8)) !void {
    const idx = try gpa.alloc(u32, logp.len);
    defer gpa.free(idx);
    for (idx, 0..) |*x, i| x.* = @intCast(i);
    const Ctx = struct {
        lp: []const f64,
        fn less(c: @This(), a: u32, b: u32) bool {
            return c.lp[a] > c.lp[b] or (c.lp[a] == c.lp[b] and a < b);
        }
    };
    std.mem.sort(u32, idx, Ctx{ .lp = logp }, Ctx.less);
    var tail: f64 = 0;
    var entropy: f64 = 0;
    for (idx, 0..) |v, j| {
        const p = @exp(logp[v]);
        if (j >= topk) tail += p;
        if (p > 0) entropy -= p * logp[v];
    }
    const ph: PosHead = .{ .tail = @floatCast(tail), .label_logp = @floatCast(logp[label]), .entropy = @floatCast(entropy) };
    try out.appendSlice(gpa, std.mem.asBytes(&ph));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(idx[0..topk]));
    for (idx[0..topk]) |v| {
        const f: f32 = @floatCast(logp[v]);
        try out.appendSlice(gpa, std.mem.asBytes(&f));
    }
}

// ------------------------------------------------------------------------------------------------------ tests ---

const testing = std.testing;

fn logSoftmax(gpa: std.mem.Allocator, lg: []const f32) ![]f64 {
    const out = try gpa.alloc(f64, lg.len);
    var m: f64 = -std.math.inf(f64);
    for (lg) |v| m = @max(m, v);
    var z: f64 = 0;
    for (lg) |v| z += @exp(@as(f64, v) - m);
    for (lg, out) |v, *o| o.* = @as(f64, v) - m - @log(z);
    return out;
}

/// A toy run: `world` ranks, `vocab` ids, rows of the given lengths in two sets; reference and served logits drawn
/// from `seed`, the served ones the reference's plus noise. Returns the mean exact KL and the top-1 agreement
/// computed directly, and fills the reference file and the ranks' partial files.
const Toy = struct {
    ref: []u8,
    files: [][]u8,
    kl: f64,
    agree: f64,
    row_kl: []f64,

    fn make(gpa: std.mem.Allocator, seed: u64, world: usize, vocab: usize, topk: usize, lens: []const u32, noise: f32) !Toy {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const share = vocab / world;
        const toks = try gpa.alloc([]u32, lens.len);
        defer {
            for (toks) |t| gpa.free(t);
            gpa.free(toks);
        }
        const set_of = try gpa.alloc(u32, lens.len);
        defer gpa.free(set_of);
        var n_pos: usize = 0;
        for (lens, 0..) |n, i| {
            toks[i] = try gpa.alloc(u32, n);
            for (toks[i]) |*t| t.* = rnd.uintLessThan(u32, @intCast(vocab));
            set_of[i] = if (i < lens.len / 2) 0 else 1;
            n_pos += n - 1;
        }
        var pos: std.ArrayList(u8) = .empty;
        defer pos.deinit(gpa);
        const files = try gpa.alloc([]u8, world);
        const lists = try gpa.alloc(std.ArrayList(u8), world);
        defer gpa.free(lists);
        for (lists) |*l| l.* = .empty;
        errdefer for (lists) |*l| l.deinit(gpa);
        const ref_lg = try gpa.alloc(f32, vocab);
        defer gpa.free(ref_lg);
        const srv_lg = try gpa.alloc(f32, vocab);
        defer gpa.free(srv_lg);
        const row_kl = try gpa.alloc(f64, lens.len);
        var kl_sum: f64 = 0;
        var agree: usize = 0;
        // the records need the finished reference's digests: the partials' heads are written after it is built
        var recs: std.ArrayList(Part) = .empty;
        defer recs.deinit(gpa);
        for (lens, 0..) |n, ri| {
            var rk: f64 = 0;
            for (0..n - 1) |j| {
                for (ref_lg, srv_lg) |*a, *b| {
                    a.* = rnd.floatNorm(f32) * 3;
                    b.* = a.* + rnd.floatNorm(f32) * noise;
                }
                const lp = try logSoftmax(gpa, ref_lg);
                defer gpa.free(lp);
                const lq = try logSoftmax(gpa, srv_lg);
                defer gpa.free(lq);
                var kl: f64 = 0;
                var bp: usize = 0;
                var bq: usize = 0;
                for (lp, lq, 0..) |a, b, v| {
                    kl += @exp(a) * (a - b);
                    if (a > lp[bp]) bp = v;
                    if (srv_lg[v] > srv_lg[bq]) bq = v;
                }
                rk += kl;
                agree += @intFromBool(bp == bq);
                const label = toks[ri][j + 1];
                const at = pos.items.len;
                try posFromLogp(gpa, lp, topk, label, &pos);
                const p = Pos.of(pos.items[at..], topk);
                for (0..world) |r| try recs.append(gpa, part(srv_lg[r * share ..][0..share], r * share, p, topk, label));
            }
            kl_sum += rk;
            row_kl[ri] = rk / @as(f64, @floatFromInt(n - 1));
        }
        const ref = try build(gpa, @intCast(vocab), @intCast(topk), &.{ "first", "second" }, set_of, toks, pos.items);
        const h = rd(Head, ref, 0);
        for (lists, 0..) |*l, r| {
            const ph: PartHead = .{ .rank = @intCast(r), .world = @intCast(world), .vocab_off = @intCast(r * share), .vocab_part = @intCast(share), .topk = @intCast(topk), .n_pos = n_pos, .tokens_digest = h.tokens_digest, .body_digest = h.body_digest, .chunk_rows = 0 };
            try l.appendSlice(gpa, std.mem.asBytes(&ph));
            for (0..n_pos) |i| try l.appendSlice(gpa, std.mem.asBytes(&recs.items[i * world + r]));
            files[r] = try l.toOwnedSlice(gpa);
        }
        return .{ .ref = ref, .files = files, .kl = kl_sum / @as(f64, @floatFromInt(n_pos)), .agree = @as(f64, @floatFromInt(agree)) / @as(f64, @floatFromInt(n_pos)), .row_kl = row_kl };
    }

    fn deinit(t: *Toy, gpa: std.mem.Allocator) void {
        gpa.free(t.ref);
        for (t.files) |f| gpa.free(f);
        gpa.free(t.files);
        gpa.free(t.row_kl);
    }
};

test "with the whole vocabulary kept, the ranks' shares add up to the exact KL and top-1 agreement" {
    const gpa = testing.allocator;
    var toy = try Toy.make(gpa, 11, 4, 64, 64, &.{ 9, 5, 7, 12 }, 0.7);
    defer toy.deinit(gpa);
    const ref = try Ref.open(toy.ref, true);
    var s = try combine(gpa, &ref, toy.files);
    defer s.deinit(gpa);
    try testing.expectEqual(@as(usize, 8 + 4 + 6 + 11), s.all.n);
    try testing.expectApproxEqAbs(toy.kl, s.all.mean(s.all.kl_lo), 1e-6); // the reference's log-probabilities are f32
    try testing.expectApproxEqAbs(toy.kl, s.all.mean(s.all.kl_hi), 1e-6);
    try testing.expectApproxEqAbs(toy.agree, s.all.mean(@floatFromInt(s.all.agree)), 1e-12);
    for (toy.row_kl, s.rows) |a, b| try testing.expectApproxEqAbs(a, b, 1e-6);
    try testing.expectEqual(s.all.n, s.sets[0].n + s.sets[1].n);
    try testing.expectApproxEqAbs(s.all.kl_lo, s.sets[0].kl_lo + s.sets[1].kl_lo, 1e-9);
    try testing.expect(s.all.median <= s.all.p90);
    // any order of the files is the same run
    const swapped = [_][]const u8{ toy.files[2], toy.files[0], toy.files[3], toy.files[1] };
    var s2 = try combine(gpa, &ref, &swapped);
    defer s2.deinit(gpa);
    try testing.expectEqual(s.all.kl_lo, s2.all.kl_lo);
}

test "with only the top tokens kept, the exact KL lies between the two bounds, and the bounds close as more are kept" {
    const gpa = testing.allocator;
    var gap_before: f64 = std.math.inf(f64);
    for ([_]usize{ 4, 16, 48 }) |topk| {
        var toy = try Toy.make(gpa, 23, 4, 64, topk, &.{ 6, 6 }, 1.0);
        defer toy.deinit(gpa);
        const ref = try Ref.open(toy.ref, true);
        var s = try combine(gpa, &ref, toy.files);
        defer s.deinit(gpa);
        const lo = s.all.mean(s.all.kl_lo);
        const hi = s.all.mean(s.all.kl_hi);
        try testing.expect(lo <= toy.kl + 1e-6);
        try testing.expect(hi >= toy.kl - 1e-6);
        try testing.expect(hi - lo < gap_before);
        gap_before = hi - lo;
        try testing.expectApproxEqAbs(toy.agree, s.all.mean(@floatFromInt(s.all.agree)), 1e-12);
    }
}

test "a served model equal to the reference scores zero, and its perplexity is the reference's" {
    const gpa = testing.allocator;
    var toy = try Toy.make(gpa, 5, 2, 32, 32, &.{ 10, 10 }, 0);
    defer toy.deinit(gpa);
    const ref = try Ref.open(toy.ref, true);
    var s = try combine(gpa, &ref, toy.files);
    defer s.deinit(gpa);
    try testing.expectApproxEqAbs(@as(f64, 0), s.all.mean(s.all.kl_lo), 1e-6);
    try testing.expectEqual(s.all.n, s.all.agree);
    try testing.expectApproxEqAbs(s.all.ref_nll, s.all.served_nll, 1e-4);
}

test "topLocal keeps the largest logits in order, the lower id first among equals" {
    var lg: [40]f32 = undefined;
    for (&lg, 0..) |*v, i| v.* = @floatFromInt((i * 7) % 20); // each value 0..19 twice
    const d = topLocal(&lg, 100, 3, 9);
    try testing.expectEqual(@as(u32, 3), d.row);
    try testing.expectEqual(@as(f32, 19), d.lg[0]);
    try testing.expectEqual(@as(f32, 19), d.lg[1]);
    try testing.expect(d.ids[0] < d.ids[1]);
    try testing.expectEqual(@as(f32, 12), d.lg[dump_n - 1]);
    for (d.ids, d.lg) |id, v| try testing.expectEqual(v, lg[id - 100]);
}

test "digest is zlib's crc32" {
    try testing.expectEqual(@as(u64, 0xcbf43926), digest("123456789"));
}

test "part: the first largest logit, a label outside the share, far smaller logits" {
    var rec: [posBytes(2)]u8 = undefined;
    const ph: PosHead = .{ .tail = 0.25, .label_logp = -1, .entropy = 1 };
    @memcpy(rec[0..16], std.mem.asBytes(&ph));
    const ids = [2]u32{ 5, 100 };
    const lps = [2]f32{ @log(0.5), @log(0.25) };
    @memcpy(rec[16..24], std.mem.sliceAsBytes(&ids));
    @memcpy(rec[24..32], std.mem.sliceAsBytes(&lps));
    const pos = Pos.of(&rec, 2);
    const lg = [_]f32{ 1, 3, -200, 3, 0, 2, -1, 0 }; // ids 4 .. 11: id 5 is lg[1]
    const p = part(&lg, 4, pos, 2, 3);
    try testing.expectEqual(@as(f32, 3), p.m);
    try testing.expectEqual(@as(u32, 5), p.arg); // the first of the two 3s
    try testing.expectEqual(@as(f32, -200), p.lo);
    try testing.expect(std.math.isNan(p.label));
    try testing.expectApproxEqAbs(0.5 * 3.0, p.a, 1e-7); // id 100 is another rank's
    try testing.expectApproxEqAbs(@as(f64, 1), p.t, 1e-12);
    var want: f64 = 0;
    for (lg) |v| want += @exp(@as(f64, v) - 3);
    try testing.expectApproxEqRel(want, p.s, 1e-12);
    try testing.expectEqual(@as(f32, 2), part(&lg, 4, pos, 2, 9).label);
}

test "open refuses a damaged or misdescribed file, combine a wrong set of partials" {
    const gpa = testing.allocator;
    var toy = try Toy.make(gpa, 3, 4, 64, 8, &.{ 5, 4 }, 0.5);
    defer toy.deinit(gpa);
    const ref = try Ref.open(toy.ref, true);
    try testing.expectEqual(@as(u32, 2), ref.head.n_rows);
    const t1 = try ref.tokens(gpa, 1);
    defer gpa.free(t1);
    try testing.expectEqual(@as(usize, 4), t1.len);
    try testing.expectEqualStrings("second", Ref.setName(&ref.set(1)));

    const bad = try gpa.dupe(u8, toy.ref);
    defer gpa.free(bad);
    bad[bad.len - 3] ^= 1; // a position record
    try testing.expectError(error.ReferenceDamaged, Ref.open(bad, true));
    _ = try Ref.open(bad, false);
    bad[bad.len - 3] ^= 1;
    bad[ref.tokens_off] ^= 1; // a token
    try testing.expectError(error.ReferenceDamaged, Ref.open(bad, true));
    try testing.expectError(error.BadReference, Ref.open(toy.ref[0 .. toy.ref.len - 1], true));

    try testing.expectError(error.PartialsMismatch, combine(gpa, &ref, toy.files[0..3])); // a rank missing
    const twice = [_][]const u8{ toy.files[0], toy.files[1], toy.files[2], toy.files[2] };
    try testing.expectError(error.PartialsMismatch, combine(gpa, &ref, &twice));
    var other = try Toy.make(gpa, 4, 4, 64, 8, &.{ 5, 4 }, 0.5); // other rows: other digests
    defer other.deinit(gpa);
    const mixed = [_][]const u8{ toy.files[0], toy.files[1], toy.files[2], other.files[3] };
    try testing.expectError(error.PartialsMismatch, combine(gpa, &ref, &mixed));
    const cut = [_][]const u8{ toy.files[0], toy.files[1], toy.files[2], toy.files[3][0 .. toy.files[3].len - 8] };
    try testing.expectError(error.BadPartials, combine(gpa, &ref, &cut));
}

test "a rows file (no positions) opens, and report prints every set and the total" {
    const gpa = testing.allocator;
    const a = [_]u32{ 1, 2, 3 };
    const b = [_]u32{ 4, 5 };
    const rows = try build(gpa, 10, 0, &.{ "wiki", "code" }, &.{ 0, 1 }, &.{ &a, &b }, &.{});
    defer gpa.free(rows);
    const r = try Ref.open(rows, true);
    try testing.expect(!r.hasPositions());
    try testing.expectEqual(@as(u64, 3), r.head.n_pos);

    var toy = try Toy.make(gpa, 9, 4, 64, 64, &.{ 5, 5 }, 0.3);
    defer toy.deinit(gpa);
    const ref = try Ref.open(toy.ref, true);
    var s = try combine(gpa, &ref, toy.files);
    defer s.deinit(gpa);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try report(gpa, &ref, &s, &text, &json);
    try testing.expect(std.mem.indexOf(u8, text.items, "first") != null);
    try testing.expect(std.mem.indexOf(u8, text.items, "all") != null);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json.items, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 3), parsed.value.object.get("sets").?.array.items.len);
}
