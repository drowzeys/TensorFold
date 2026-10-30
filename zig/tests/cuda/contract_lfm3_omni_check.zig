//! LFM3-Omni's exact CUDA outputs checked on the host: positions, codes, greedy and sampled picks, frame windows.
const std = @import("std");
const core = @import("core");
const data = @import("contract_lfm3_omni_data.zig");
const ref = @import("contract_lfm3_omni_ref.zig");
const num = @import("contract_lfm3_omni_num.zig");
const Case = @import("contract_lfm3_omni.zig").Case;

const bfr = core.bf16_round;
const nq = data.nq;
const size = data.size;

/// What the last run wrote, against the kernel's rule on the prepared inputs: null when it holds.
pub fn verify(self: *Case) !?[]const u8 {
    const d = &self.d;
    const got = try self.rig.outputBytes();
    const ints: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, got[0 .. got.len / 4 * 4]));
    const m = &self.msg;
    switch (d.which) {
        .pos => {
            const want = d.ints(0)[0] + 3;
            if (ints[0] != want) return try std.fmt.bufPrint(m, "position {d}, expected {d}", .{ ints[0], want });
        },
        .frame_end => {
            const in = d.ints(0);
            var any = false;
            for (in) |x| any = any or x == size - 1;
            for (ints[0..nq], in, 0..) |x, y, q| if (x != (if (any) size - 1 else y))
                return try std.fmt.bufPrint(m, "code {d} is {d}, expected {d}", .{ q, x, if (any) size - 1 else y });
        },
        .hist_push => {
            const rows = d.ints(1);
            for (ints[0..rows.len], rows, 0..) |x, y, i| {
                const want = if (i / nq == 5) d.ints(0)[i % nq] else y;
                if (x != want) return try std.fmt.bufPrint(m, "window word {d} is {d}, expected {d}", .{ i, x, want });
            }
        },
        .fsq => {
            const out: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, got));
            const want = try self.alloc().alloc(f64, out.len);
            const t = try self.alloc().alloc(f64, 64);
            ref.replay(false, .{}, d, want, t, t);
            for (out, want, 0..) |x, w, i| if (bfr.value(x) != w)
                return try std.fmt.bufPrint(m, "latent {d} is {d}, expected {d}", .{ i, bfr.value(x), w });
        },
        .argmax => {
            const x = d.bf(0);
            const sup = d.ints(1);
            var best = -std.math.inf(f32);
            var at: i32 = std.math.maxInt(i32);
            for (x, 0..) |h, i| {
                if (std.mem.indexOfScalar(i32, sup, @intCast(i)) != null) continue;
                const v = bfr.value(h);
                if (v > best or (v == best and @as(i32, @intCast(i)) < at)) {
                    best = v;
                    at = @intCast(i);
                }
            }
            if (ints[0] != at) return try std.fmt.bufPrint(m, "picked {d}, expected {d}", .{ ints[0], at });
        },
        .depth_pick => return pick(self, ints),
        else => unreachable,
    }
    return null;
}

fn mix64(x: u64) u64 {
    var z = x +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Each picked code: greedy exactly, sampled within rounding of the best Gumbel score; other codebooks untouched.
fn pick(self: *Case, got: []const i32) !?[]const u8 {
    const d = &self.d;
    const m = &self.msg;
    const codes = got[0..nq];
    const masked = got[nq .. 2 * nq];
    const order = d.ints(1);
    const hist = d.ints(2);
    const temp = d.cfg.temp;
    const n = d.shape.rows;
    var picked: [nq]bool = @splat(false);
    for (0..n) |b| {
        const q: usize = @intCast(order[4 + b]);
        picked[q] = true;
        var seen: [size]bool = @splat(false);
        for (0..d.shape.k) |j| seen[@intCast(hist[j * nq + q])] = true;
        var best = -std.math.inf(f64);
        var at: i32 = std.math.maxInt(i32);
        var score_of_got: f64 = std.math.nan(f64);
        for (0..size) |i| {
            var v: f32 = bfr.value(d.bf(0)[q * size + i]);
            if (seen[i]) v = if (v < 0) v * 1.2 else v / 1.2;
            var s: f64 = v;
            if (temp > 0) {
                const h = mix64(0x5eed ^ mix64((@as(u64, 17) << 32) ^ (@as(u64, q) << 16) ^ @as(u64, i)));
                const u = (@as(f64, @floatFromInt(h >> 40)) + 0.5) / 16777216.0;
                s = @as(f64, v) / temp - @log(-@log(u));
            }
            if (@as(i32, @intCast(i)) == codes[q]) score_of_got = s;
            if (s > best or (s == best and @as(i32, @intCast(i)) < at)) {
                best = s;
                at = @intCast(i);
            }
        }
        if (masked[q] != 0) return try std.fmt.bufPrint(m, "codebook {d} picked but still masked", .{q});
        if (temp == 0 or !std.math.isFinite(best)) {
            if (codes[q] != at) return try std.fmt.bufPrint(m, "codebook {d} took {d}, expected {d}", .{ q, codes[q], at });
        } else if (!(score_of_got >= best - (1e-5 * @abs(best) + 1e-5)))
            return try std.fmt.bufPrint(m, "codebook {d} took {d} (score {d:.6}), the best is {d} ({d:.6})", .{ q, codes[q], score_of_got, at, best });
    }
    for (0..nq) |q| if (!picked[q] and (codes[q] != d.ints(3)[q] or masked[q] != d.ints(4)[q]))
        return try std.fmt.bufPrint(m, "codebook {d} was not this pass's but changed", .{q});
    return null;
}

test {
    _ = num;
}
