//! --register-redraw-st: a reply with a phrase opening far above the rest is drawn once more; the smaller jump is sent.
const std = @import("std");
const drama_register = @import("drama_register.zig");
const speech_redraw = @import("speech_redraw.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Allocator = std.mem.Allocator;

/// A phrase after the first trips at `st` over the others' median opening with a `high` share over `high_hz`; 0: off.
pub const Policy = struct { st: f64 = 0, high: f64 = 0.2, high_hz: f64 = 210 };

/// One reply's register guard: whether a draw tripped, and its one redraw, once admitted.
pub const Guard = struct {
    policy: Policy,
    tripped: bool = false,
    used: u8 = 0,

    pub fn on(g: *const Guard) bool {
        return g.policy.st > 0;
    }

    /// The jump in a draw's `pcm` at `rate`, in semitones of excess, logged; null when it does not trip.
    pub fn measure(g: *Guard, gpa: Allocator, io: std.Io, id: []const u8, pcm: []const f32, rate: u32) (Allocator.Error || error{NonFinite})!?f64 {
        // the measure's buffers, megabytes for a long reply, live only as long as it
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const began = clock.nowNs(io);
        const m = try drama_register.measure(arena.allocator(), pcm, rate, g.policy.high_hz);
        const ms = clock.seconds(clock.nowNs(io) - began) * 1000;
        const trip = drama_register.jump(m.phrases, g.policy.st, g.policy.high);
        // the largest excess after the first phrase, tripping or not: the reply's jump in numbers
        var top: ?drama_register.Phrase = null;
        if (m.phrases.len > 1) for (m.phrases[1..]) |p| if (p.excess) |e| if (top == null or e > top.?.excess.?) {
            top = p;
        };
        const shown: drama_register.Phrase = trip orelse top orelse .{ .start = 0, .end = 0, .opening = null, .excess = 0, .high = null };
        log.line("speech {s} register: trip={d} excess={d:.2}st high={d:.3} at={d:.2}s phrases={d} median={d:.1}Hz measured_ms={d:.1}", .{ id, @intFromBool(trip != null), shown.excess.?, shown.high orelse 0, @as(f64, @floatFromInt(shown.start)) / 100, m.phrases.len, m.median_hz orelse 0, ms });
        if (trip == null) return null;
        g.tripped = true;
        return trip.?.excess.?;
    }

    /// After a draw begun at `started` tripped: its one redraw's latest start if it is due and fits, logged; else null.
    pub fn again(g: *const Guard, redraws: *const speech_redraw.Redraws, io: std.Io, id: []const u8, started: i96) ?i96 {
        if (g.used > 0) {
            log.line("speech {s} register trip skipped: redraw used", .{id});
            return null;
        }
        const f = redraws.fit(io, started);
        if (f.start_by == null) {
            log.line("speech {s} register trip skipped: budget t={d:.2}s d={d:.2}s t+d={d:.2}s budget={d:.2}s", .{ id, f.t, f.d, f.t + f.d, redraws.budget_s });
            return null;
        }
        log.line("speech {s} register redraw: t={d:.2}s d={d:.2}s t+d={d:.2}s budget={d:.2}s", .{ id, f.t, f.d, f.t + f.d, redraws.budget_s });
        return f.start_by;
    }

    /// X-Register-Trip, whether any draw tripped, then X-Register-Redraws; null when the guard is off.
    pub fn headers(g: *const Guard, trip: *[1]u8, redraws: *[1]u8) ?[2][]const u8 {
        if (!g.on()) return null;
        trip[0] = '0' + @as(u8, @intFromBool(g.tripped));
        redraws[0] = '0' + g.used;
        return .{ trip, redraws };
    }
};
