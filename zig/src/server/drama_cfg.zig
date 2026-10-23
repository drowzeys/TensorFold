//! Guidance as the server asks for it: --cfg-gamma's γ or a request's X-Cfg-Gamma, and each draw's plain prompt.
const std = @import("std");
const fields = @import("fields.zig");
const model_text = @import("model_text.zig");
const drama_prompt = @import("drama_prompt.zig");
const Allocator = std.mem.Allocator;

/// The largest γ: every logit moves (γ - 1) times its distance from the plain prompt's.
pub const max_gamma = 5;

/// γ (1: unguided), whether X-Cfg-Gamma may set it per request, and whether γ 1 is guided too (a diagnostic).
pub const Options = struct { gamma: f32 = 1, request_gamma: bool = false, force_guided: bool = false };

/// γ from 1 (unguided) to max_gamma.
pub fn valid(g: f64) bool {
    return g >= 1 and g <= max_gamma;
}

/// A reply's γ: X-Cfg-Gamma's value `sent` when the options let a request set it, else theirs; null: out of range.
pub fn gamma(options: Options, sent: ?[]const u8) ?f32 {
    if (!options.request_gamma) return options.gamma;
    const g = fields.pyFloat(sent orelse return options.gamma) orelse return null;
    return if (valid(g)) @floatCast(g) else null;
}

/// What a draw is guided by: the same turn as the built-in voice's plain prompt, and γ; empty: unguided.
pub const Guide = struct { unconditioned: []const u32 = &.{}, gamma: f32 = 1 };

/// The guide of a draw speaking `spoken` from `rows`: none at γ 1 unless forced, nor without audio rows (plain).
pub fn guide(a: Allocator, text: model_text.Text, cfg: ?Options, g: f32, spoken: []const u8, rows: drama_prompt.Rows) model_text.Error!Guide {
    const forced = if (cfg) |o| o.force_guided else false;
    if ((g == 1 and !forced) or rows.codes.len == 0) return .{};
    return .{ .unconditioned = try drama_prompt.encode(a, text, spoken), .gamma = g };
}

/// The X-Cfg-Gamma reply header's value, with guidance on.
pub fn header(buf: *[24]u8, g: ?f32) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{d}", .{g orelse return null}) catch unreachable;
}

test "γ is the server's unless it lets a request set it; a request's must be a number from 1 to 5" {
    const fixed: Options = .{ .gamma = 2 };
    try std.testing.expectEqual(@as(?f32, 2), gamma(fixed, "3"));
    try std.testing.expectEqual(@as(?f32, 2), gamma(fixed, "nonsense"));
    const open: Options = .{ .gamma = 2, .request_gamma = true };
    try std.testing.expectEqual(@as(?f32, 2), gamma(open, null));
    for ([_][]const u8{ "1", " 1.5 ", "3", "5" }, [_]f32{ 1, 1.5, 3, 5 }) |sent, want| try std.testing.expectEqual(@as(?f32, want), gamma(open, sent));
    for ([_][]const u8{ "0", "0.99", "5.01", "nan", "inf", "", "2x" }) |bad| try std.testing.expectEqual(@as(?f32, null), gamma(open, bad));
}

test "the header says γ as a number: 2, 1.5" {
    var buf: [24]u8 = undefined;
    try std.testing.expectEqualStrings("2", header(&buf, 2).?);
    try std.testing.expectEqualStrings("1.5", header(&buf, 1.5).?);
    try std.testing.expect(header(&buf, null) == null);
}
