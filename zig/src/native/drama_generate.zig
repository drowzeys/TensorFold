//! Drama 3's frame loop: frames until end of speech, the frame cap or a collapse into silence, decoded as they come.
const std = @import("std");
const api = @import("engine_api");

pub const Codes = [10]u32;

/// Seconds of silent frames that end a synthesis: one stalled 7.4 s and recovered; collapses ran 37 s and 64 s.
pub const seconds = 10;
/// A frame below -40 dBFS (an RMS of 0.01) is silent: collapses sat at -95 to -50, digital or a voice's noise floor.
const floor_power = 1e-4;

/// Voiced seconds after which only the 10 s rule applies: 42 of 189 measured collapses voiced less before collapsing.
pub const onset_voiced_seconds = 1;

/// The samples of the current run of silent frames, and of the voiced frames so far.
pub const Guard = struct {
    limit: usize,
    /// --silence-onset-s in samples: a silent run this long ends a synthesis voiced less than `voiced_limit`; 0: off.
    onset: usize = 0,
    /// --silence-mid-s in samples: end of speech in a silent run this long, after `voiced_limit`, is silence; 0: off.
    mid: usize = 0,
    voiced_limit: usize,
    run: usize = 0,
    voiced: usize = 0,

    pub fn init(sample_rate: u32, onset_s: f64, mid_s: f64) Guard {
        const per_s: f64 = @floatFromInt(sample_rate);
        return .{
            .limit = seconds * @as(usize, sample_rate),
            .onset = @intFromFloat(@round(onset_s * per_s)),
            .mid = @intFromFloat(@round(mid_s * per_s)),
            .voiced_limit = onset_voiced_seconds * @as(usize, sample_rate),
        };
    }

    /// The samples of the --silence-mid-s hole end of speech now falls in, or 0: in replays every such draw lost words.
    pub fn hollow(g: *const Guard) usize {
        return if (g.mid > 0 and g.run >= g.mid and g.voiced >= g.voiced_limit) g.run else 0;
    }

    /// Hears whole frames of `frame_length` samples; the frames up to the one ending a run that ends it, or null.
    pub fn hear(g: *Guard, pcm: []const f32, frame_length: usize) ?usize {
        var frames = std.mem.window(f32, pcm, frame_length, frame_length);
        var heard: usize = 0;
        while (frames.next()) |samples| {
            heard += 1;
            var power: f32 = 0;
            for (samples) |x| power += x * x;
            if (power < floor_power * @as(f32, @floatFromInt(samples.len))) {
                g.run += samples.len;
            } else {
                g.run = 0;
                g.voiced += samples.len;
            }
            // later speech in the same chunk does not undo the collapse
            if (g.run >= g.limit or (g.onset > 0 and g.run >= g.onset and g.voiced < g.voiced_limit)) return heard;
        }
        return null;
    }
};

/// When a synthesis decodes while it generates: once it has `first` frames, then every `chunk` more.
pub const Cadence = struct { first: usize, chunk: usize };

/// How a synthesis ended, and the frames it made; `hollow`: the samples of a --silence-mid-s hole that made it silence.
pub const Ending = struct { frames: usize, reason: api.Reason, hollow: usize = 0 };

/// Fills `codes` (the cap) from `source.next()` (null: EOS), decoding at `cadence`; a collapse `guard` hears cuts.
pub fn run(source: anytype, codes: []Codes, cadence: Cadence, guard_init: Guard, frame_length: usize) !Ending {
    var guard = guard_init;
    var count: usize = 0;
    var decoded: usize = 0;
    var reason: api.Reason = .stop;
    while (try source.next()) |made| {
        codes[count] = made;
        count += 1;
        if (count == codes.len) {
            reason = .length;
            break;
        }
        if (count - decoded == if (decoded == 0) cadence.first else cadence.chunk) {
            if (try hearNew(source, &guard, codes[0..count], decoded, frame_length)) |cut| return .{ .frames = cut, .reason = .silence };
            decoded = count;
        }
    }
    // silence heard in the last frames outranks end of speech and the cap: the reply collapsed however it stopped
    if (count > decoded) if (try hearNew(source, &guard, codes[0..count], decoded, frame_length)) |cut| return .{ .frames = cut, .reason = .silence };
    // the frame cap is not end of speech: a reply cut there was still talking or still stalled
    if (reason == .stop and guard.hollow() > 0) return .{ .frames = count, .reason = .silence, .hollow = guard.hollow() };
    return .{ .frames = count, .reason = reason };
}

/// Decodes the frames past `decoded`, then hands them to `source.publish` up to any cut; the cut, or null.
fn hearNew(source: anytype, guard: *Guard, codes: []const Codes, decoded: usize, frame_length: usize) !?usize {
    // heard before it is handed over: a stream's owner rewrites its audio in place (PCM16) as soon as it has it
    const cut = if (guard.hear(try source.decode(codes), frame_length)) |n| decoded + n else null;
    try source.publish(cut orelse codes.len);
    return cut;
}

const rate = 44100;
const frame = 2048;
const speech: f32 = 0.25;
const collapse: f32 = 4e-5; // -88 dBFS: a collapse whose samples still quantize to +-1
const stream: Cadence = .{ .first = 1, .chunk = 4 };
const whole: Cadence = .{ .first = 16, .chunk = 16 };

/// One sign-alternating level a frame, then EOS; `quantizes`: handed over as a stream's, its owner rewrites it at once.
const Script = struct {
    levels: []const f32,
    pcm: []f32,
    quantizes: bool = false,
    asked: usize = 0,
    decoded: usize = 0,
    published: usize = 0,

    fn next(s: *Script) !?Codes {
        if (s.asked == s.levels.len) return null;
        s.asked += 1;
        return @splat(0);
    }

    fn decode(s: *Script, codes: []const Codes) ![]const f32 {
        const from = s.decoded;
        for (s.levels[from..codes.len], from..) |level, f| {
            for (s.pcm[f * frame ..][0..frame], 0..) |*x, i| x.* = if (i % 2 == 0) level else -level;
        }
        s.decoded = codes.len;
        return s.pcm[from * frame .. codes.len * frame];
    }

    fn publish(s: *Script, frames: usize) !void {
        if (frames < s.published or frames > s.decoded) return error.TestUnexpectedResult;
        if (s.quantizes) quantize(s.pcm[s.published * frame .. frames * frame]);
        s.published = frames;
    }
};

/// What the stream route does to audio it owns: PCM16 in place, over the first half of the same bytes.
fn quantize(pcm: []f32) void {
    const bytes = std.mem.sliceAsBytes(pcm);
    for (0..pcm.len) |i| std.mem.writeInt(i16, bytes[2 * i ..][0..2], @intFromFloat(std.math.clamp(pcm[i], -1, 1) * 32767), .little);
}

/// How the loop ends `levels` under frame cap `cap`, the frames it asked the model for, and the frames it handed over.
fn ends(levels: []const f32, cap: usize, cadence: Cadence) !struct { Ending, usize, usize } {
    return onsetEnds(levels, cap, cadence, 0);
}

/// As `ends`, with --silence-onset-s `onset_s`.
fn onsetEnds(levels: []const f32, cap: usize, cadence: Cadence, onset_s: f64) !struct { Ending, usize, usize } {
    return guardEnds(levels, cap, cadence, .init(rate, onset_s, 0));
}

/// As `ends`, under `guard`.
fn guardEnds(levels: []const f32, cap: usize, cadence: Cadence, guard: Guard) !struct { Ending, usize, usize } {
    const gpa = std.testing.allocator;
    const pcm = try gpa.alloc(f32, levels.len * frame);
    defer gpa.free(pcm);
    const codes = try gpa.alloc(Codes, cap);
    defer gpa.free(codes);
    var s: Script = .{ .levels = levels, .pcm = pcm, .quantizes = cadence.chunk == stream.chunk };
    return .{ try run(&s, codes, cadence, guard, frame), s.asked, s.published };
}

/// `speaking` frames of speech, then `silent` of collapse.
fn collapsed(comptime speaking: usize, comptime silent: usize) [speaking + silent]f32 {
    var levels: [speaking + silent]f32 = undefined;
    @memset(levels[0..speaking], speech);
    @memset(levels[speaking..], collapse);
    return levels;
}

/// A voice's noise floor: frames at -62 and -48 dBFS by turns, straddling -60 dBFS as one cloned voice's collapse did.
fn hum(levels: []f32) void {
    for (levels, 0..) |*level, i| level.* = if (i % 2 == 0) 8e-4 else 4e-3;
}

test "a stream and a whole reply of the same frames end at the same sample, though the stream's owner rewrites its audio" {
    const levels = collapsed(140, 1400);
    const streamed, _, const handed = try ends(&levels, 4096, stream);
    try std.testing.expectEqual(Ending{ .frames = 140 + 216, .reason = .silence }, streamed);
    try std.testing.expectEqual(streamed.frames, handed);
    try std.testing.expectEqual(streamed, (try ends(&levels, 4096, whole))[0]);
}

test "a reply that collapses into silence ends at the frame that completes 10 s, asking for at most a chunk more" {
    const levels = collapsed(140, 1400); // 6.5 s of speech, then 65 s of collapse: what one render did
    const trip = 140 + 216; // 216 frames of 2048 samples are the first run past 10 s at 44.1 kHz
    for ([_]Cadence{ stream, whole }) |cadence| {
        const ending, const asked, _ = try ends(&levels, 4096, cadence);
        try std.testing.expectEqual(Ending{ .frames = trip, .reason = .silence }, ending);
        try std.testing.expect(asked < trip + cadence.chunk);
    }
}

test "a reply that collapses into its voice's noise floor ends at the frame that completes 10 s, though the floor sinks" {
    var levels: [28 + 646]f32 = undefined; // 1.3 s of speech, then 30 s of the cloned voice's floor: what one render did
    @memset(levels[0..28], speech);
    for ([_]usize{ levels.len, 28 + 331 }) |sinks| { // one reply's floor sank below -60 dBFS for good after 15.4 s
        hum(levels[28..sinks]);
        @memset(levels[sinks..], 7e-4); // -63 dBFS
        for ([_]Cadence{ stream, whole }) |cadence| {
            const ending, const asked, _ = try ends(&levels, 4096, cadence);
            try std.testing.expectEqual(Ending{ .frames = 28 + 216, .reason = .silence }, ending);
            try std.testing.expect(asked < 28 + 216 + cadence.chunk);
        }
    }
}

test "a 7.4 s stall that recovers into speech, as one measured reply did, does not trip it, in digital silence or noise" {
    var levels: [118 + 159 + 400]f32 = undefined; // 5.48 s of speech, a 7.40 s stall, then the rest of the text
    @memset(&levels, speech);
    for ([_]bool{ false, true }) |noise| {
        if (noise) hum(levels[118..][0..159]) else @memset(levels[118..][0..159], collapse);
        for ([_]Cadence{ stream, whole }) |cadence| {
            try std.testing.expectEqual(Ending{ .frames = levels.len, .reason = .stop }, (try ends(&levels, 4096, cadence))[0]);
        }
    }
}

test "pauses at the collapse's level never trip it short of 10 s; 10 s does" {
    var levels: [10 * (40 + 39)]f32 = undefined; // 1.86 s of speech, then a 1.81 s pause, ten times
    for (0..10) |i| {
        @memset(levels[i * 79 ..][0..40], speech);
        @memset(levels[i * 79 + 40 ..][0..39], collapse);
    }
    try std.testing.expectEqual(Ending{ .frames = levels.len, .reason = .stop }, (try ends(&levels, 4096, whole))[0]);
    var edge: [40 + 215 + 40]f32 = undefined; // 215 silent frames: 9.98 s
    @memset(&edge, speech);
    @memset(edge[40..][0..215], 0);
    const one: Cadence = .{ .first = 1, .chunk = 1 };
    try std.testing.expectEqual(Ending{ .frames = edge.len, .reason = .stop }, (try ends(&edge, 4096, one))[0]);
    edge[255] = 0;
    try std.testing.expectEqual(Ending{ .frames = 256, .reason = .silence }, (try ends(&edge, 4096, one))[0]);
}

test "a run that reaches 10 s inside a decode chunk trips it, though speech resumes before the chunk ends" {
    var levels: [20 + 216 + 30]f32 = undefined; // the 216th silent frame, 235, is mid-chunk for streams and whole replies
    @memset(&levels, speech);
    @memset(levels[20..][0..216], collapse);
    for ([_]Cadence{ stream, whole }) |cadence| {
        try std.testing.expectEqual(Ending{ .frames = 236, .reason = .silence }, (try ends(&levels, 4096, cadence))[0]);
    }
}

test "silence reached in the last frames outranks end of speech and the frame cap" {
    // whole replies decode at 352, streams at 353: 10 s of silence completes only in the final decode
    const whole_tail = collapsed(140, 216);
    const stream_tail = collapsed(138, 216);
    try std.testing.expectEqual(Ending{ .frames = 356, .reason = .silence }, (try ends(&whole_tail, 4096, whole))[0]);
    try std.testing.expectEqual(Ending{ .frames = 354, .reason = .silence }, (try ends(&stream_tail, 4096, stream))[0]);
    try std.testing.expectEqual(Ending{ .frames = 356, .reason = .silence }, (try ends(&whole_tail, 356, whole))[0]);
    const spoken = collapsed(205, 0);
    try std.testing.expectEqual(Ending{ .frames = 205, .reason = .stop }, (try ends(&spoken, 4096, whole))[0]);
    try std.testing.expectEqual(Ending{ .frames = 200, .reason = .length }, (try ends(&spoken, 200, whole))[0]);
}

test "quiet speech above -40 dBFS is not silence, nor are its pauses at its voice's noise floor" {
    var levels: [10 * (40 + 39)]f32 = undefined; // 1.86 s of speech at -39 dBFS, then a 1.81 s pause, ten times: 36.7 s
    for (0..10) |i| {
        @memset(levels[i * 79 ..][0..40], 0.011);
        hum(levels[i * 79 + 40 ..][0..39]);
    }
    try std.testing.expectEqual(Ending{ .frames = levels.len, .reason = .stop }, (try ends(&levels, 4096, whole))[0]);
    @memset(&levels, 0.011);
    try std.testing.expectEqual(Ending{ .frames = levels.len, .reason = .stop }, (try ends(&levels, 4096, whole))[0]);
}

test "the floor is -40 dBFS: 10 s just under it trips; just over it, or straddling it within each chunk, does not" {
    const under: f32 = 0.0099; // -40.09 dBFS
    const over: f32 = 0.0101; // -39.91 dBFS
    var levels: [28 + 300]f32 = undefined; // 1.3 s of speech, then 13.9 s at the floor
    @memset(levels[0..28], speech);
    for ([_]Cadence{ stream, whole }) |cadence| {
        @memset(levels[28..], under);
        try std.testing.expectEqual(Ending{ .frames = 28 + 216, .reason = .silence }, (try ends(&levels, 4096, cadence))[0]);
        @memset(levels[28..], over);
        try std.testing.expectEqual(Ending{ .frames = levels.len, .reason = .stop }, (try ends(&levels, 4096, cadence))[0]);
        for (levels[28..], 0..) |*level, i| level.* = if (i % 3 == 2) over else under; // both sides in every chunk
        try std.testing.expectEqual(Ending{ .frames = levels.len, .reason = .stop }, (try ends(&levels, 4096, cadence))[0]);
    }
}

const recommended = 3.0; // --silence-onset-s 3: a silent run of 65 frames (3.02 s) ends a synthesis not yet voiced 1 s
const onset_frames = 65;

test "with --silence-onset-s, 3 s of silence ends a synthesis that has voiced less than 1 s, as no words or one word did" {
    var levels: [21 + 400]f32 = undefined;
    for ([_]usize{ 0, 12, 21 }) |voiced| { // nothing; one word (0.56 s), as two observed collapses did; 21 frames: 0.98 s
        @memset(levels[0..voiced], speech);
        hum(levels[voiced..]);
        for ([_]Cadence{ stream, whole }) |cadence| {
            const ending, const asked, const handed = try onsetEnds(levels[0 .. voiced + 400], 4096, cadence, recommended);
            try std.testing.expectEqual(Ending{ .frames = voiced + onset_frames, .reason = .silence }, ending);
            try std.testing.expect(asked < voiced + onset_frames + cadence.chunk);
            if (cadence.chunk == stream.chunk) try std.testing.expectEqual(ending.frames, handed);
        }
    }
}

test "once 1 s is voiced, in one stretch or several, only the 10 s rule ends a synthesis" {
    var levels: [11 + 30 + 11 + 200 + 40]f32 = undefined; // two half-second words, a 1.4 s pause, a 9.3 s stall, speech
    @memset(&levels, speech);
    @memset(levels[11..][0..30], collapse);
    hum(levels[52..][0..200]);
    for ([_]Cadence{ stream, whole }) |cadence| {
        try std.testing.expectEqual(Ending{ .frames = levels.len, .reason = .stop }, (try onsetEnds(&levels, 4096, cadence, recommended))[0]);
        const long = collapsed(22, 300); // 22 frames: 1.02 s
        try std.testing.expectEqual(Ending{ .frames = 22 + 216, .reason = .silence }, (try onsetEnds(&long, 4096, cadence, recommended))[0]);
    }
}

test "--silence-onset-s leaves the mid-reply rule as it was: the 7.4 s stall recovers, a collapse ends at 10 s" {
    var stall: [118 + 159 + 400]f32 = undefined;
    @memset(&stall, speech);
    hum(stall[118..][0..159]);
    const late = collapsed(140, 1400);
    for ([_]Cadence{ stream, whole }) |cadence| {
        try std.testing.expectEqual(Ending{ .frames = stall.len, .reason = .stop }, (try onsetEnds(&stall, 4096, cadence, recommended))[0]);
        try std.testing.expectEqual((try ends(&late, 4096, cadence))[0], (try onsetEnds(&late, 4096, cadence, recommended))[0]);
    }
}

test "--silence-onset-s 0 is off: an unvoiced or one-word collapse ends at 10 s, as before" {
    var levels: [12 + 400]f32 = undefined;
    for ([_]usize{ 0, 12 }) |voiced| {
        @memset(levels[0..voiced], speech);
        hum(levels[voiced..]);
        for ([_]Cadence{ stream, whole }) |cadence| {
            const off, const asked, const handed = try ends(levels[0 .. voiced + 400], 4096, cadence);
            try std.testing.expectEqual(Ending{ .frames = voiced + 216, .reason = .silence }, off);
            try std.testing.expect(asked < voiced + 216 + cadence.chunk);
            if (cadence.chunk == stream.chunk) try std.testing.expectEqual(off.frames, handed);
        }
    }
}

test "each long-form segment is its own synthesis: one that starts silent trips, however much the one before it spoke" {
    const spoke = collapsed(400, 0); // 18.6 s of speech
    var silent: [400]f32 = undefined;
    hum(&silent);
    for ([_]Cadence{ stream, whole }) |cadence| {
        try std.testing.expectEqual(Ending{ .frames = spoke.len, .reason = .stop }, (try onsetEnds(&spoke, 4096, cadence, recommended))[0]);
        try std.testing.expectEqual(Ending{ .frames = onset_frames, .reason = .silence }, (try onsetEnds(&silent, 4096, cadence, recommended))[0]);
    }
}

const mid_frames = 65; // --silence-mid-s 3: a run of 65 frames (3.02 s)

test "with --silence-mid-s, end of speech in a 3 s hole after 1 s voiced is silence, its audio kept whole" {
    const lost = collapsed(140, mid_frames); // 6.5 s of speech, then the hole the reply ended in
    const short = collapsed(140, mid_frames - 1);
    for ([_]Cadence{ stream, whole }) |cadence| {
        const ending, _, const handed = try guardEnds(&lost, 4096, cadence, .init(rate, 0, recommended));
        try std.testing.expectEqual(Ending{ .frames = lost.len, .reason = .silence, .hollow = mid_frames * frame }, ending);
        if (cadence.chunk == stream.chunk) try std.testing.expectEqual(lost.len, handed);
        try std.testing.expectEqual(Ending{ .frames = short.len, .reason = .stop }, (try guardEnds(&short, 4096, cadence, .init(rate, 0, recommended)))[0]);
        try std.testing.expectEqual(Ending{ .frames = lost.len, .reason = .stop }, (try ends(&lost, 4096, cadence))[0]);
    }
}

test "--silence-mid-s leaves a hole that recovers, and a reply the frame cap ends in one, as they were" {
    var stall: [118 + 159 + 40]f32 = undefined; // the measured 7.4 s stall, then the rest of the text
    @memset(&stall, speech);
    hum(stall[118..][0..159]);
    const capped = collapsed(140, 100);
    for ([_]Cadence{ stream, whole }) |cadence| {
        try std.testing.expectEqual(Ending{ .frames = stall.len, .reason = .stop }, (try guardEnds(&stall, 4096, cadence, .init(rate, 0, recommended)))[0]);
        try std.testing.expectEqual(Ending{ .frames = 240, .reason = .length }, (try guardEnds(&capped, 240, cadence, .init(rate, 0, recommended)))[0]);
    }
}

test "under 1 s voiced a hole is the onset rule's, cut at 3 s; from 1 s voiced, end of speech in it is the mid rule's" {
    var levels: [22 + 70]f32 = undefined;
    for ([_]usize{ 21, 22 }) |voiced| { // 0.98 s and 1.02 s
        @memset(levels[0..voiced], speech);
        hum(levels[voiced..]);
        const draw = levels[0 .. voiced + 70];
        for ([_]Cadence{ stream, whole }) |cadence| {
            const both = (try guardEnds(draw, 4096, cadence, .init(rate, recommended, recommended)))[0];
            const mid_only = (try guardEnds(draw, 4096, cadence, .init(rate, 0, recommended)))[0];
            if (voiced < 22) {
                try std.testing.expectEqual(Ending{ .frames = voiced + onset_frames, .reason = .silence }, both);
                try std.testing.expectEqual(Ending{ .frames = draw.len, .reason = .stop }, mid_only);
            } else {
                try std.testing.expectEqual(Ending{ .frames = draw.len, .reason = .silence, .hollow = 70 * frame }, both);
                try std.testing.expectEqual(both, mid_only);
            }
        }
    }
}
