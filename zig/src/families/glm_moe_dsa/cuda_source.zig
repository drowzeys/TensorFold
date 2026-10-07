//! GLM-5.3's checkpoint bytes read with O_DIRECT through io_uring a unit ahead of the loader (Phase 2b: the fast
//! load, Nemotron's cuda_source.zig approach with the reads planned per layer). A unit is every tensor of one layer
//! ("model.layers.<i>."), or unit 0: the tensors outside the layers (embedding, final norm, lm_head). For each tensor
//! the span this rank needs is read: its row block for row cuts (split.rule .row: down / o_proj trellis, kv_b, the
//! lm_head share...), the whole tensor otherwise (second-axis cuts take every tile row's slice, which spans the
//! tensor, so the host cuts it as before). A background thread fills one of two arenas while the loader converts and
//! uploads the other, so the disk (here an NFS export) never waits for the GPU copies; nothing goes through the page
//! cache and no file is mapped for reading, which is what made the Phase 2a load (mapped reads, 688 s) slow.
//!
//! The loader asks `bytes(name)`: the prefetched span (whole tensor, or the rank's rows when `partial`), or null for
//! a tensor the plan did not cover (it then reads the mapping as before).
const std = @import("std");
const linux = std.os.linux;
const core = @import("core");
const split = @import("split.zig");
const dio = core.direct_io;

/// One read's span (an io_uring entry).
pub const chunk_bytes: usize = 4 << 20;
/// Reads in flight.
pub const depth: u16 = 32;

/// Where a tensor's bytes sit in its unit's arena; `partial`: only this rank's row block (split.rule .row).
pub const Entry = struct { at: usize, len: usize, partial: bool };

const Read = struct { file: usize, lo: u64, len: usize, need: usize, at: usize };

const idle: u32 = 0;
const requested: u32 = 1;
const ready: u32 = 2;

pub const Batch = struct {
    unit: usize = 0,
    names: std.StringHashMapUnmanaged(Entry) = .empty,
    reads: std.ArrayList(Read) = .empty,
    arena: []align(dio.alignment) u8 = &.{},
    used: usize = 0,
    state: std.atomic.Value(u32) = .init(idle),
    failed: bool = false,
};

const libc = struct {
    const usleep = @extern(*const fn (c_uint) callconv(.c) c_int, .{ .name = "usleep" });
    const posix_fadvise = @extern(*const fn (c_int, i64, i64, c_int) callconv(.c) c_int, .{ .name = "posix_fadvise" });
};

pub const Prefetch = struct {
    gpa: std.mem.Allocator,
    ck: *const core.Checkpoint,
    rank: usize,
    world: usize,
    direct: []dio.File, // per checkpoint file: O_DIRECT where the file system allows it
    plain: []dio.File, // per checkpoint file: through the page cache (short-read fallbacks only)
    sizes: []u64,
    batches: [2]Batch = .{ .{}, .{} },
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    read_bytes: std.atomic.Value(u64) = .init(0),
    fallbacks: std.atomic.Value(u64) = .init(0),
    uring: bool = false,
    direct_files: usize = 0,

    pub fn init(gpa: std.mem.Allocator, ck: *const core.Checkpoint, rank: usize, world: usize) !*Prefetch {
        const p = try gpa.create(Prefetch);
        errdefer gpa.destroy(p);
        const n = ck.files.items.len;
        p.* = .{ .gpa = gpa, .ck = ck, .rank = rank, .world = world, .direct = try gpa.alloc(dio.File, n), .plain = try gpa.alloc(dio.File, n), .sizes = try gpa.alloc(u64, n) };
        var opened: usize = 0;
        errdefer {
            for (0..opened) |i| {
                p.direct[i].close();
                p.plain[i].close();
            }
            gpa.free(p.direct);
            gpa.free(p.plain);
            gpa.free(p.sizes);
        }
        for (ck.files.items, 0..) |*f, i| {
            p.direct[i] = try dio.File.open(f.path);
            p.plain[i] = dio.File.open(f.path) catch |e| {
                p.direct[i].close();
                return e;
            };
            if (p.plain[i].direct) {
                // a second descriptor without O_DIRECT for the fallbacks
                p.plain[i].close();
                const fd = std.c.open(f.path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
                if (fd < 0) {
                    p.direct[i].close();
                    return error.FileNotFound;
                }
                p.plain[i] = .{ .fd = fd, .direct = false };
            }
            if (p.direct[i].direct) p.direct_files += 1;
            p.sizes[i] = f.map.memory.len;
            opened += 1;
        }
        p.thread = try std.Thread.spawn(.{}, worker, .{p});
        return p;
    }

    pub fn deinit(p: *Prefetch) void {
        p.stop.store(true, .release);
        if (p.thread) |t| t.join();
        for (&p.batches) |*b| {
            b.names.deinit(p.gpa);
            b.reads.deinit(p.gpa);
            if (b.arena.len > 0) p.gpa.free(b.arena);
        }
        for (p.direct, p.plain) |*a, *b| {
            a.close();
            b.close();
        }
        p.gpa.free(p.direct);
        p.gpa.free(p.plain);
        p.gpa.free(p.sizes);
        const gpa = p.gpa;
        gpa.destroy(p);
    }

    fn inUnit(unit: usize, name: []const u8) bool {
        if (unit == 0) return !std.mem.startsWith(u8, name, "model.layers.");
        var buf: [48]u8 = undefined;
        const pre = std.fmt.bufPrint(&buf, "model.layers.{d}.", .{unit - 1}) catch return false;
        return std.mem.startsWith(u8, name, pre);
    }

    /// Plans `unit` into batch `slot` and hands it to the reader; the loader alternates slots 0, 1, 0, ... in the
    /// order it loads units (the reader serves them in that order), and a slot must be released before its reuse.
    pub fn request(p: *Prefetch, slot: usize, unit: usize) !void {
        const b = &p.batches[slot % 2];
        if (b.state.load(.acquire) != idle) return error.PrefetchBusy;
        b.names.clearRetainingCapacity();
        b.reads.clearRetainingCapacity();
        b.used = 0;
        b.unit = unit;
        b.failed = false;
        for (p.ck.files.items, 0..) |*f, fi| {
            var it = f.names.iterator();
            while (it.next()) |kv| {
                const name = kv.key_ptr.*;
                if (!inUnit(unit, name)) continue;
                const kind = split.known(name) orelse continue;
                const e = kv.value_ptr.*;
                const total = e.end - e.begin;
                var lo_b: usize = 0;
                var len = total;
                var partial = false;
                if (kind == .row and p.world > 1 and e.rank > 0 and e.shape[0] % p.world == 0 and e.shape[0] > 0) {
                    const per = total / e.shape[0];
                    const n = e.shape[0] / p.world;
                    lo_b = p.rank * n * per;
                    len = n * per;
                    partial = true;
                }
                if (len == 0) continue;
                const abs: u64 = f.data + e.begin + lo_b;
                const lo = std.mem.alignBackward(u64, abs, dio.alignment);
                const hi = std.mem.alignForward(u64, abs + len, dio.alignment);
                const at = std.mem.alignForward(usize, b.used, dio.alignment);
                b.used = at + @as(usize, @intCast(hi - lo));
                try b.names.put(p.gpa, name, .{ .at = at + @as(usize, @intCast(abs - lo)), .len = len, .partial = partial });
                var s = lo;
                while (s < hi) : (s += chunk_bytes) {
                    const n: usize = @intCast(@min(chunk_bytes, hi - s));
                    const need: usize = @intCast(@min(@as(u64, n), p.sizes[fi] -| s));
                    try b.reads.append(p.gpa, .{ .file = fi, .lo = s, .len = n, .need = need, .at = at + @as(usize, @intCast(s - lo)) });
                }
            }
        }
        if (b.arena.len < b.used) {
            if (b.arena.len > 0) p.gpa.free(b.arena);
            b.arena = &.{};
            const want = std.mem.alignForward(usize, b.used, 256 << 20);
            b.arena = try p.gpa.alignedAlloc(u8, .fromByteUnits(dio.alignment), want);
        }
        b.state.store(requested, .release);
    }

    /// Waits for `unit`'s reads (requested into `slot`); its bytes stay valid until `release(slot)`.
    pub fn wait(p: *Prefetch, slot: usize, unit: usize) !*const Batch {
        const b = &p.batches[slot % 2];
        if (b.unit != unit or b.state.load(.acquire) == idle) return error.PrefetchNotRequested;
        while (b.state.load(.acquire) != ready) _ = libc.usleep(200);
        if (b.failed) return error.PrefetchReadFailed;
        return b;
    }

    /// The slot's arena goes back to the loader for its next unit.
    pub fn release(p: *Prefetch, slot: usize) void {
        p.batches[slot % 2].state.store(idle, .release);
    }

    /// A tensor's prefetched bytes in `b` (null: not planned).
    pub fn lookup(b: *const Batch, name: []const u8) ?struct { bytes: []const u8, partial: bool } {
        const e = b.names.get(name) orelse return null;
        return .{ .bytes = b.arena[e.at..][0..e.len], .partial = e.partial };
    }

    /// Bytes [offset, offset + out.len) of checkpoint file `fi` read now (one-off reads beside the units: the
    /// draft head's lm_head rows), through the plain descriptor.
    pub fn readNow(p: *Prefetch, fi: usize, offset: u64, out: []u8) !void {
        var got: usize = 0;
        while (got < out.len) {
            const n = std.c.pread(p.plain[fi].fd, out.ptr + got, out.len - got, @intCast(offset + got));
            if (n < 0) {
                if (std.c.errno(n) == .INTR) continue;
                return error.ReadFailed;
            }
            if (n == 0) return error.EndOfFile;
            got += @intCast(n);
        }
        _ = libc.posix_fadvise(p.plain[fi].fd, @intCast(offset), @intCast(out.len), 4); // POSIX_FADV_DONTNEED
        _ = p.read_bytes.fetchAdd(out.len, .monotonic);
    }

    fn worker(p: *Prefetch) void {
        var ring: ?linux.IoUring = linux.IoUring.init(depth, 0) catch null;
        defer if (ring) |*r| r.deinit();
        p.uring = ring != null;
        var next: usize = 0;
        while (true) {
            const b = &p.batches[next % 2];
            while (b.state.load(.acquire) != requested) {
                if (p.stop.load(.acquire)) return;
                _ = libc.usleep(200);
            }
            p.readAll(b, if (ring) |*r| r else null) catch |e| {
                std.log.err("glm53 prefetch of unit {d} failed: {t}", .{ b.unit, e });
                b.failed = true;
            };
            b.state.store(ready, .release);
            next += 1;
        }
    }

    fn readAll(p: *Prefetch, b: *Batch, ring: ?*linux.IoUring) !void {
        const reads = b.reads.items;
        const r = ring orelse {
            for (reads) |rd| try p.finish(b, rd, 0);
            return;
        };
        var next: usize = 0;
        var inflight: usize = 0;
        var cqes: [depth]linux.io_uring_cqe = undefined;
        while (next < reads.len or inflight > 0) {
            var queued: usize = 0;
            while (next < reads.len and inflight < depth) {
                const rd = reads[next];
                const fd = if (p.direct[rd.file].direct) p.direct[rd.file].fd else p.plain[rd.file].fd;
                _ = try r.read(next, fd, .{ .buffer = b.arena[rd.at..][0..rd.len] }, rd.lo);
                next += 1;
                inflight += 1;
                queued += 1;
            }
            if (queued > 0) _ = try r.submit();
            const n = try r.copy_cqes(&cqes, 1);
            for (cqes[0..n]) |c| {
                inflight -= 1;
                const rd = reads[@intCast(c.user_data)];
                const got: usize = if (c.res > 0) @intCast(c.res) else 0;
                if (c.res < 0) std.log.warn("glm53 prefetch: io_uring read at {d} failed ({t}); reading again", .{ rd.lo, c.err() });
                try p.finish(b, rd, got);
            }
        }
    }

    /// A read that returned `got` of its `need` bytes: the rest through the plain descriptor (EOF tails, errors).
    fn finish(p: *Prefetch, b: *Batch, rd: Read, got: usize) !void {
        _ = p.read_bytes.fetchAdd(@min(got, rd.need), .monotonic);
        // a file system without O_DIRECT read through the page cache: those pages go again at once (GB10's
        // page cache is the memory the weights need)
        if (!p.direct[rd.file].direct) _ = libc.posix_fadvise(p.plain[rd.file].fd, @intCast(rd.lo), @intCast(rd.len), 4);
        if (got >= rd.need) return;
        _ = p.fallbacks.fetchAdd(1, .monotonic);
        var done = got;
        while (done < rd.need) {
            const n = std.c.pread(p.plain[rd.file].fd, b.arena.ptr + rd.at + done, rd.need - done, @intCast(rd.lo + done));
            if (n < 0) {
                if (std.c.errno(n) == .INTR) continue;
                return error.ReadFailed;
            }
            if (n == 0) return error.EndOfFile;
            done += @intCast(n);
        }
        _ = p.read_bytes.fetchAdd(rd.need - got, .monotonic);
        _ = libc.posix_fadvise(p.plain[rd.file].fd, @intCast(rd.lo), @intCast(rd.len), 4);
    }
};

test "units: layer prefixes do not overlap" {
    try std.testing.expect(Prefetch.inUnit(8, "model.layers.7.mlp.gate.weight"));
    try std.testing.expect(!Prefetch.inUnit(8, "model.layers.78.eh_proj.weight"));
    try std.testing.expect(Prefetch.inUnit(79, "model.layers.78.eh_proj.weight"));
    try std.testing.expect(Prefetch.inUnit(0, "lm_head.weight"));
    try std.testing.expect(!Prefetch.inUnit(0, "model.layers.0.input_layernorm.weight"));
}
