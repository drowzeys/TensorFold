//! The register guard's measure, ported from the numpy one it was calibrated with and kept to its decisions.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The reference measures at 16 kHz, in 10 ms frames of 40 ms Hann windows, for pitch from 70 to 400 Hz.
const rate = 16000;
const hop = 160;
const size = 640;
const lag_lo = rate / 400;
const lag_hi = rate / 70 + 1;
/// A frame is voiced when its normalised autocorrelation peak reaches this, and within `quiet_db` of the speech level.
const voiced_r = 0.45;
/// A peak whose half lag scores this share of it is taken at the half lag: no sub-octave picks.
const octave_r = 0.9;
const quiet_db = 25.0;
/// Rests of 10 frames (0.10 s) or more part phrases; a phrase under 50 frames (0.5 s) joins the one before.
const rest_frames = 10;
const min_phrase_frames = 50;
/// A phrase's opening is the median pitch of its first quarter, and at least its first 40 frames (0.4 s).
const open_frames = 40;

/// Frames [start, end); opening and its excess over the others' openings in st (to 0.01); share over the line (0.001).
pub const Phrase = struct { start: usize, end: usize, opening: ?f64, excess: ?f64, high: ?f64 };

/// A reply's measure: its median voiced pitch (null when nothing is voiced) and its phrases.
pub const Measure = struct { median_hz: ?f64, phrases: []const Phrase };

/// The measure of `pcm` at `sample_rate`, `high_hz` its high line; NonFinite for audio with a NaN or infinite sample.
pub fn measure(a: Allocator, pcm: []const f32, sample_rate: u32, high_hz: f64) (Allocator.Error || error{NonFinite})!Measure {
    for (pcm) |x| if (!std.math.isFinite(x)) return error.NonFinite;
    const y = try resample(a, pcm, sample_rate);
    const db = try framesDb(a, y);
    if (db.len == 0) return .{ .median_hz = null, .phrases = &.{} };
    const level = try speechLevel(a, db);
    const f0 = try pitch(a, y, db, level);
    return describe(a, f0, try phrases(a, db, level), high_hz);
}

/// Of the phrases after the first with an excess of `st` and a high share of `high` or more, the largest jump; or null.
pub fn jump(phrases_: []const Phrase, st: f64, high: f64) ?Phrase {
    var found: ?Phrase = null;
    if (phrases_.len < 2) return null;
    for (phrases_[1..]) |p| {
        const excess = p.excess orelse continue;
        if (excess < st or (p.high orelse continue) < high) continue;
        if (found == null or excess > found.?.excess.?) found = p;
    }
    return found;
}

/// load16: a 3-tap (rate / 16 kHz, half to even) boxcar, then linear interpolation at 16 kHz, as numpy computes them.
fn resample(a: Allocator, pcm: []const f32, sample_rate: u32) Allocator.Error![]f64 {
    const sr: f64 = @floatFromInt(sample_rate);
    const k: usize = @max(1, @as(usize, @intFromFloat(roundHalfEven(sr / rate))));
    const smooth = try a.alloc(f64, pcm.len);
    defer a.free(smooth);
    // np.convolve(x, ones(k) / k, 'same'): sample i averages x[i + (k - 1) / 2 - (k - 1) ..][0..k], zeros outside
    const w = 1.0 / @as(f64, @floatFromInt(k));
    for (smooth, 0..) |*s, i| {
        var sum: f64 = 0;
        const last = i + (k - 1) / 2;
        for (0..k) |j| if (last >= j and last - j < pcm.len) {
            sum += @as(f64, pcm[last - j]) * w;
        };
        s.* = sum;
    }
    const step = 1.0 / @as(f64, rate);
    const n: usize = @intFromFloat(@ceil((@as(f64, @floatFromInt(pcm.len)) / sr) / step));
    const y = try a.alloc(f64, n);
    var m: usize = 0;
    for (y, 0..) |*out, j| {
        const t = @as(f64, @floatFromInt(j)) * step;
        while (m + 1 < smooth.len and @as(f64, @floatFromInt(m + 1)) / sr <= t) m += 1;
        if (m + 1 >= smooth.len) {
            out.* = smooth[smooth.len - 1];
            continue;
        }
        const x0 = @as(f64, @floatFromInt(m)) / sr;
        const x1 = @as(f64, @floatFromInt(m + 1)) / sr;
        out.* = if (t == x0) smooth[m] else (smooth[m + 1] - smooth[m]) / (x1 - x0) * (t - x0) + smooth[m];
    }
    return y;
}

fn roundHalfEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

/// Each 10 ms frame's power in dB.
fn framesDb(a: Allocator, y: []const f64) Allocator.Error![]f64 {
    const db = try a.alloc(f64, y.len / hop);
    for (db, 0..) |*d, i| {
        var sum: f64 = 0;
        for (y[i * hop ..][0..hop]) |x| sum += x * x;
        d.* = 10 * std.math.log10(sum / hop + 1e-10);
    }
    return db;
}

/// The mean power of the frames within 20 dB of the 95th percentile, in dB.
fn speechLevel(a: Allocator, db: []const f64) Allocator.Error!f64 {
    const sorted = try a.dupe(f64, db);
    defer a.free(sorted);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const top = percentile(sorted, 0.95) - 20;
    var sum: f64 = 0;
    var n: usize = 0;
    for (db) |d| if (d >= top) {
        sum += std.math.pow(f64, 10, d / 10);
        n += 1;
    };
    return 10 * std.math.log10(sum / @as(f64, @floatFromInt(n)));
}

/// np.percentile's linear method, its two-sided lerp included.
fn percentile(sorted: []const f64, q: f64) f64 {
    const at = q * @as(f64, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(at));
    const hi = @min(lo + 1, sorted.len - 1);
    const t = at - @floor(at);
    const d = sorted[hi] - sorted[lo];
    return if (t >= 0.5) sorted[hi] - d * (1 - t) else sorted[lo] + d * t;
}

/// Hz per frame, NaN where unvoiced: the autocorrelation of each Hann-windowed frame centred on it.
fn pitch(a: Allocator, y: []const f64, db: []const f64, level: f64) Allocator.Error![]f64 {
    var window: [size]f64 = undefined;
    for (&window, 0..) |*w, j| w.* = 0.5 - 0.5 * @cos(2 * std.math.pi * @as(f64, @floatFromInt(j)) / (size - 1));
    var norm: [lag_hi + 1]f64 = undefined;
    for (&norm, 0..) |*n, l| n.* = dot(window[0 .. size - l], window[l..]);
    const f0 = try a.alloc(f64, db.len);
    var frame: [size]f64 = undefined;
    var r: [lag_hi + 1]f64 = undefined;
    for (f0, db, 0..) |*f, d, i| {
        f.* = std.math.nan(f64);
        // only a frame within quiet_db of the speech level can be voiced
        if (d < level - quiet_db) continue;
        for (&frame, &window, 0..) |*x, w, j| {
            const at = i * hop + j;
            x.* = if (at >= size / 2 and at - size / 2 < y.len) y[at - size / 2] * w else 0;
        }
        const zero = @max(dot(&frame, &frame), 1e-12);
        r[0] = 1;
        for (lag_lo - 1..lag_hi + 1) |l| r[l] = dot(frame[0 .. size - l], frame[l..]) / zero / (norm[l] / norm[0]);
        var lag: usize = lag_lo;
        for (lag_lo + 1..lag_hi) |l| if (r[l] > r[lag]) {
            lag = l;
        };
        const half = lag / 2;
        if (half >= lag_lo and r[half] >= octave_r * r[lag]) lag = half;
        if (r[lag] < voiced_r) continue;
        const l = std.math.clamp(lag, 1, lag_hi - 1);
        const bend = r[l - 1] - 2 * r[l] + r[l + 1];
        const shift = if (@abs(bend) > 1e-12) 0.5 * (r[l - 1] - r[l + 1]) / bend else 0;
        f.* = rate / (@as(f64, @floatFromInt(l)) + std.math.clamp(shift, -0.5, 0.5));
    }
    return f0;
}

fn dot(x: []const f64, y: []const f64) f64 {
    const V = @Vector(4, f64);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 4 <= x.len) : (i += 4) acc += @as(V, x[i..][0..4].*) * @as(V, y[i..][0..4].*);
    var sum = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) sum += x[i] * y[i];
    return sum;
}

/// Speech between rests of `rest_frames` or more, a phrase under `min_phrase_frames` joined to the one before.
fn phrases(a: Allocator, db: []const f64, level: f64) Allocator.Error![]Phrase {
    var out: std.ArrayList(Phrase) = .empty;
    // a frame rests when under `quiet`, as the reference's `db < level - 25` decides it: a NaN never rests
    const quiet = level - quiet_db;
    var first: ?usize = null;
    var last: usize = 0;
    for (db, 0..) |d, i| if (!(d < quiet)) {
        if (first == null) first = i;
        last = i;
    };
    const begin = first orelse return out.items;
    var start = begin;
    var i = begin;
    while (i <= last) {
        var end = i;
        while (end <= last and db[end] < quiet) end += 1;
        if (end - i >= rest_frames) {
            try join(a, &out, start, i);
            start = end;
        }
        i = @max(end, i + 1);
    }
    try join(a, &out, start, last + 1);
    if (out.items.len > 1 and out.items[0].end - out.items[0].start < min_phrase_frames) {
        out.items[1].start = out.items[0].start;
        _ = out.orderedRemove(0);
    }
    return out.items;
}

fn join(a: Allocator, out: *std.ArrayList(Phrase), start: usize, end: usize) Allocator.Error!void {
    if (out.items.len > 0 and end - start < min_phrase_frames) {
        out.items[out.items.len - 1].end = end;
    } else try out.append(a, .{ .start = start, .end = end, .opening = null, .excess = null, .high = null });
}

/// Each phrase's opening, excess and high share, rounded as the reference rounds the values its decision reads.
fn describe(a: Allocator, f0: []const f64, list: []Phrase, high_hz: f64) Allocator.Error!Measure {
    var voiced: std.ArrayList(f64) = .empty;
    defer voiced.deinit(a);
    for (list) |p| for (f0[p.start..p.end]) |f| if (!std.math.isNan(f)) try voiced.append(a, f);
    const median = try medianOf(a, voiced.items) orelse return .{ .median_hz = null, .phrases = list };
    var scratch: std.ArrayList(f64) = .empty;
    defer scratch.deinit(a);
    for (list) |*p| {
        scratch.clearRetainingCapacity();
        const head_end = @min(p.start + @max((p.end - p.start) / 4, open_frames), f0.len);
        for (f0[p.start..head_end]) |f| if (!std.math.isNan(f)) try scratch.append(a, f);
        if (scratch.items.len >= 5) p.opening = round(12 * std.math.log2((try medianOf(a, scratch.items)).? / median), 100);
        var count: usize = 0;
        var over: usize = 0;
        for (f0[p.start..p.end]) |f| if (!std.math.isNan(f)) {
            count += 1;
            over += @intFromBool(f > high_hz);
        };
        if (count >= 5) p.high = round(@as(f64, @floatFromInt(over)) / @as(f64, @floatFromInt(count)), 1000);
    }
    for (list, 0..) |*p, i| {
        const opening = p.opening orelse continue;
        scratch.clearRetainingCapacity();
        for (list, 0..) |q, j| if (j != i) if (q.opening) |o| try scratch.append(a, o);
        if (try medianOf(a, scratch.items)) |others| p.excess = round(opening - others, 100);
    }
    return .{ .median_hz = median, .phrases = list };
}

/// Python's round(x, 2) and round(x, 3), ties aside: the reference compares the rounded values with its thresholds.
fn round(x: f64, scale: f64) f64 {
    return @round(x * scale) / scale;
}

fn medianOf(a: Allocator, values: []const f64) Allocator.Error!?f64 {
    if (values.len == 0) return null;
    const sorted = try a.dupe(f64, values);
    defer a.free(sorted);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const mid = sorted.len / 2;
    return if (sorted.len % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2;
}

test "the phrase scan steps past every frame, a NaN one too, which never rests" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    // 60 loud frames, a NaN, 60 loud, a 12-frame rest, 60 loud: two phrases, the NaN inside the first
    var db: [193]f64 = @splat(-10);
    db[60] = std.math.nan(f64);
    @memset(db[121..133], -100);
    const list = try phrases(arena.allocator(), &db, -10);
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expect(list[0].start == 0 and list[0].end == 121 and list[1].start == 133 and list[1].end == 193);
}
