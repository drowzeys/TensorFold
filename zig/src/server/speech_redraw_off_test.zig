//! Silence redraws off: silenced replies as sent before redraws existed (this file also builds against that server).
const std = @import("std");
const api = @import("engine_api");
const drama_long = @import("drama_long.zig");
const harness = @import("speech_stream_test.zig");
const scripted = @import("speech_long_test.zig");
const Turn = scripted.Turn;
const Turns = scripted.Turns;

const gpa = std.testing.allocator;
const head_pcm = "Content-Type: audio/pcm\r\n";
const head_stream = head_pcm ++ "Transfer-Encoding: chunked\r\nX-Sample-Rate: 1000\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n";

/// The whole reply from its Content-Type on, to a request sending an X-Deadline-S the server must not read.
fn reply(a: std.mem.Allocator, turns: []const Turn, long_form: ?drama_long.Options, input: []const u8, stream: bool) ![]const u8 {
    var t: Turns = .{ .turns = turns };
    defer t.deinit();
    const s = try gpa.create(harness.Served);
    defer scripted.done(s);
    try s.open(t.engine(), scripted.config(long_form), "POST /v1/audio/speech HTTP/1.1\r\nX-Deadline-S: soon", try scripted.body(a, input, stream), 2000);
    _ = try s.readHead();
    s.route.join();
    try std.testing.expectEqual(turns.len, t.seen);
    for (t.prompts.items, t.seeds.items) |prompt, seed| try std.testing.expectEqual(api.seedFor(prompt, 0), seed);
    // all of it read first: reading more can move the buffer a head slice points into
    _ = try s.readRest();
    const got = s.got.items;
    return a.dupe(u8, got[std.mem.indexOf(u8, got, "Content-Type").?..]);
}

test "redraws off: a silenced whole reply, single or long-form, is sent as before" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const one = Turn{ .frames = 5, .reason = .silence };
    var want: std.ArrayList(u8) = .empty;
    try want.print(a, head_pcm ++ "Content-Length: 100\r\nX-Finish-Reason: silence\r\n\r\n", .{});
    try scripted.turnPcm(&want, a, one, 1);
    try std.testing.expectEqualStrings(want.items, try reply(a, &.{one}, null, "One. Two.", false));

    const turns = [_]Turn{ .{ .frames = 5 }, .{ .frames = 3, .reason = .silence } };
    want.clearRetainingCapacity();
    try want.print(a, head_pcm ++ "Content-Length: {d}\r\nX-Finish-Reason: silence\r\nX-Long-Form-Segments: 2;470\r\n\r\n", .{2 * (10 * 8 + 420)});
    try scripted.turnPcm(&want, a, turns[0], 1);
    try scripted.zeros(&want, a, 420);
    try scripted.turnPcm(&want, a, turns[1], 1);
    try std.testing.expectEqualStrings(want.items, try reply(a, &turns, .{ .chars = 20 }, try scripted.sentences(a, 2), false));
}

test "redraws off: a silenced stream, single or long-form, is sent as before" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const one = Turn{ .frames = 5, .reason = .silence };
    var want: std.ArrayList(u8) = .empty;
    try want.print(a, head_stream ++ "Trailer: X-Finish-Reason\r\n\r\n64\r\n", .{});
    try scripted.turnPcm(&want, a, one, 1);
    try want.print(a, "\r\n0\r\nX-Finish-Reason: silence\r\n\r\n", .{});
    try std.testing.expectEqualStrings(want.items, try reply(a, &.{one}, null, "One. Two.", true));

    const turns = [_]Turn{ .{ .frames = 5 }, .{ .frames = 3, .reason = .silence } };
    want.clearRetainingCapacity();
    try want.print(a, head_stream ++ "Trailer: X-Finish-Reason, X-Long-Form-Segments\r\n\r\n64\r\n", .{});
    try scripted.turnPcm(&want, a, turns[0], 1);
    try want.print(a, "\r\n{x}\r\n", .{2 * (420 + 10 * 3)});
    try scripted.zeros(&want, a, 420);
    try scripted.turnPcm(&want, a, turns[1], 1);
    try want.print(a, "\r\n0\r\nX-Finish-Reason: silence\r\nX-Long-Form-Segments: 2;470\r\n\r\n", .{});
    try std.testing.expectEqualStrings(want.items, try reply(a, &turns, .{ .chars = 20 }, try scripted.sentences(a, 2), true));
}
