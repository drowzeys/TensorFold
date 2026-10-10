//! A TCP rendezvous for the ranks of one tensor-parallel world: the part of torch's TCPStore that cuda/comm.py uses.
//! Rank 0 listens on the master's port, every other rank connects (retrying while rank 0 is not up yet), rank 0 hands
//! out NCCL's unique id, and the same connections then serve named barriers (`sync`), which also check that every
//! rank was started with the same settings (a 64-bit digest each rank passes). Plain blocking sockets with Nagle off
//! (TCP_NODELAY): GLM-5.3's --parallel N sends rank 0's small round messages on them, one a decode round, and a write
//! must not wait for the previous one's acknowledgement.
const std = @import("std");
const posix = std.posix;

pub const Error = error{ SocketFailed, BindFailed, ListenFailed, ConnectTimeout, PeerClosed, BadPeer, Disagree, Timeout, BadAddress, TooManyRanks };

const magic: u32 = 0x54465232; // "TFR2"
pub const max_world = 16;

/// `host:port` -> (IPv4 address, port); the master is given as a dotted address (the fabric address of rank 0).
pub fn parseMaster(text: []const u8) Error!struct { ip: u32, port: u16 } {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.BadAddress;
    const port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return error.BadAddress;
    return .{ .ip = try parseIp4(text[0..colon]), .port = port };
}

pub fn parseIp4(host: []const u8) Error!u32 {
    var it = std.mem.splitScalar(u8, host, '.');
    var v: u32 = 0;
    var n: usize = 0;
    while (it.next()) |part| {
        const b = std.fmt.parseInt(u8, part, 10) catch return error.BadAddress;
        v = (v << 8) | b;
        n += 1;
    }
    if (n != 4) return error.BadAddress;
    return v;
}

fn sockaddr(ip: u32, port: u16) posix.sockaddr.in {
    return .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, ip) };
}

/// Nagle off on a connection (best effort: a socket that refuses keeps working, only slower for small messages).
fn noDelay(fd: posix.socket_t) void {
    const one: c_int = 1;
    // IPPROTO_TCP (6) and TCP_NODELAY (1): the same numbers on Linux and macOS
    posix.setsockopt(fd, 6, 1, std.mem.asBytes(&one)) catch {};
}

fn seconds(io: std.Io, since: anytype) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - since)) / 1e9;
}

fn sendAll(fd: posix.socket_t, bytes: []const u8) Error!void {
    var sent: usize = 0;
    while (sent < bytes.len) {
        const n = posix.system.write(fd, bytes[sent..].ptr, bytes.len - sent);
        if (posix.errno(n) != .SUCCESS) return error.PeerClosed;
        if (n == 0) return error.PeerClosed;
        sent += @intCast(n);
    }
}

/// `buf.len` bytes, waiting at most `timeout_ms` for each part of them (error.Timeout: nothing arrived in time).
fn recvAll(fd: posix.socket_t, buf: []u8, timeout_ms: i32) Error!void {
    var got: usize = 0;
    while (got < buf.len) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, timeout_ms) catch return error.PeerClosed;
        if (ready == 0) return error.Timeout;
        const n = posix.read(fd, buf[got..]) catch return error.PeerClosed;
        if (n == 0) return error.PeerClosed;
        got += n;
    }
}

/// Linux's POLLRDHUP (the peer closed, or shut its writing half; std.posix.POLL does not name it), and what poll
/// reports for a connection that is gone: it, POLLHUP or POLLERR.
const poll_rdhup: i16 = 0x2000;
const poll_gone: i16 = 0x2000 | 0x0010 | 0x0008;

fn putU32(buf: []u8, v: u32) void {
    std.mem.writeInt(u32, buf[0..4], v, .little);
}

fn getU32(buf: []const u8) u32 {
    return std.mem.readInt(u32, buf[0..4], .little);
}

pub const Rendezvous = struct {
    io: std.Io,
    rank: usize,
    world: usize,
    /// rank 0: fds[r] is rank r's connection (fds[0] the listener, closed once every rank is in); other ranks: fds[0]
    /// is the connection to rank 0.
    fds: [max_world]posix.socket_t = @splat(-1),

    /// Every rank of `world` connected through `ip:port` (rank 0 listens on every address at that port). Waits up to
    /// `timeout_s` for the others (a rank whose container starts late is waited for, not failed).
    pub fn open(io: std.Io, ip: u32, port: u16, rank: usize, world: usize, timeout_s: f64) !Rendezvous {
        if (world > max_world or rank >= world) return error.TooManyRanks;
        var self: Rendezvous = .{ .io = io, .rank = rank, .world = world };
        errdefer self.close();
        if (world == 1) return self;
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        if (rank == 0) {
            const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
            if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
            const lfd: posix.socket_t = @intCast(rc);
            defer _ = posix.system.close(lfd);
            const one: c_int = 1;
            try posix.setsockopt(lfd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
            var addr = sockaddr(0, port); // every address
            if (posix.errno(posix.system.bind(lfd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
            if (posix.errno(posix.system.listen(lfd, @intCast(max_world))) != .SUCCESS) return error.ListenFailed;
            var joined: usize = 1;
            while (joined < world) {
                var fds = [_]posix.pollfd{.{ .fd = lfd, .events = posix.POLL.IN, .revents = 0 }};
                const ready = posix.poll(&fds, 5000) catch return error.SocketFailed;
                if (ready == 0) {
                    if (seconds(io, t0) > timeout_s) return error.ConnectTimeout;
                    continue;
                }
                const arc = posix.system.accept(lfd, null, null);
                if (posix.errno(arc) != .SUCCESS) continue;
                const fd: posix.socket_t = @intCast(arc);
                var hello: [12]u8 = undefined;
                recvAll(fd, &hello, 30_000) catch {
                    _ = posix.system.close(fd);
                    continue;
                };
                const r = getU32(hello[4..]);
                if (getU32(&hello) != magic or getU32(hello[8..]) != world or r == 0 or r >= world or self.fds[r] != -1) {
                    std.log.err("rendezvous: refused a peer (magic {x}, rank {d}, world {d})", .{ getU32(&hello), r, getU32(hello[8..]) });
                    _ = posix.system.close(fd);
                    continue;
                }
                noDelay(fd);
                self.fds[r] = fd;
                joined += 1;
                std.log.info("rendezvous: rank {d} joined ({d} of {d})", .{ r, joined, world });
            }
            return self;
        }
        var logged: f64 = 0;
        while (true) {
            const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
            if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
            const fd: posix.socket_t = @intCast(rc);
            var addr = sockaddr(ip, port);
            if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
                noDelay(fd);
                self.fds[0] = fd;
                break;
            }
            _ = posix.system.close(fd);
            const waited = seconds(io, t0);
            if (waited > timeout_s) return error.ConnectTimeout;
            if (waited - logged > 60) {
                logged = waited;
                std.log.info("rendezvous: rank {d} still waiting for rank 0 ({d:.0} s)", .{ rank, waited });
            }
            std.Io.sleep(io, .fromMilliseconds(500), .awake) catch {};
        }
        var hello: [12]u8 = undefined;
        putU32(hello[0..], magic);
        putU32(hello[4..], @intCast(rank));
        putU32(hello[8..], @intCast(world));
        try sendAll(self.fds[0], &hello);
        return self;
    }

    pub fn close(self: *Rendezvous) void {
        for (&self.fds) |*fd| {
            if (fd.* >= 0) _ = posix.system.close(fd.*);
            fd.* = -1;
        }
    }

    /// The first rank whose connection has closed or failed, without reading from it (rank 0: any other rank's; the
    /// others: rank 0's); null while every connection stands. Safe from another thread than the one that reads.
    pub fn peerClosed(self: *const Rendezvous) ?usize {
        if (self.world == 1) return null;
        const first: usize = if (self.rank == 0) 1 else 0;
        const last: usize = if (self.rank == 0) self.world else 1;
        for (first..last) |r| {
            if (self.fds[r] < 0) continue;
            var fds = [_]posix.pollfd{.{ .fd = self.fds[r], .events = poll_rdhup, .revents = 0 }};
            const ready = posix.poll(&fds, 0) catch continue;
            if (ready > 0 and fds[0].revents & poll_gone != 0) return r;
        }
        return null;
    }

    /// Rank 0's `bytes` into every other rank's `bytes` (NCCL's unique id).
    pub fn broadcast(self: *Rendezvous, bytes: []u8) !void {
        if (self.world == 1) return;
        if (self.rank == 0) {
            for (1..self.world) |r| try sendAll(self.fds[r], bytes);
            return;
        }
        try recvAll(self.fds[0], bytes, 600_000);
    }

    /// Ranks 1..: rank 0's next `bytes` (a served request's header), waiting as long as it takes - a follower idles
    /// between requests. False when rank 0 has closed the connection (the server stopped).
    pub fn awaitBroadcast(self: *Rendezvous, bytes: []u8) Error!bool {
        if (self.world == 1 or self.rank == 0) return error.BadPeer;
        while (true) {
            var fds = [_]posix.pollfd{.{ .fd = self.fds[0], .events = posix.POLL.IN, .revents = 0 }};
            const ready = posix.poll(&fds, 60_000) catch return error.PeerClosed;
            if (ready == 0) continue;
            recvAll(self.fds[0], bytes, 600_000) catch |e| switch (e) {
                error.PeerClosed => return false,
                else => return e,
            };
            return true;
        }
    }

    /// Whether every rank's `ok` holds (one byte each through rank 0); every rank gets the same answer.
    pub fn agree(self: *Rendezvous, ok: bool) Error!bool {
        var all: [max_world]u8 = undefined;
        const mine = [1]u8{@intFromBool(ok)};
        try self.allGather(&mine, all[0..self.world]);
        for (all[0..self.world]) |x| if (x == 0) return false;
        return true;
    }

    /// Every rank's `mine` (the same length on every rank) into `all` [world * mine.len] in rank order (the RoCE
    /// setup's exchange of connection records: b12x's dist.all_gather_object over gloo). Rank 0 collects, then sends
    /// the whole to every rank.
    pub fn allGather(self: *Rendezvous, mine: []const u8, all: []u8) !void {
        const n = mine.len;
        if (all.len != self.world * n) return error.BadPeer;
        @memcpy(all[self.rank * n ..][0..n], mine);
        if (self.world == 1) return;
        if (self.rank != 0) {
            try sendAll(self.fds[0], mine);
            try recvAll(self.fds[0], all, 600_000);
            return;
        }
        for (1..self.world) |r| try recvAll(self.fds[r], all[r * n ..][0..n], 600_000);
        for (1..self.world) |r| try sendAll(self.fds[r], all);
    }

    /// Every rank reaches `label` before any goes on; each passes `digest` (its settings) and all must agree, else
    /// error.Disagree on every rank. Waits up to `timeout_s`, naming the missing ranks every minute (loads finish
    /// minutes apart).
    pub fn sync(self: *Rendezvous, label: []const u8, digest: u64, timeout_s: f64) !void {
        if (self.world == 1) return;
        const tag: u32 = @truncate(std.hash.Wyhash.hash(0, label));
        const t0 = std.Io.Clock.awake.now(self.io).toNanoseconds();
        if (self.rank != 0) {
            var msg: [16]u8 = undefined;
            putU32(msg[0..], magic);
            putU32(msg[4..], tag);
            std.mem.writeInt(u64, msg[8..16], digest, .little);
            try sendAll(self.fds[0], &msg);
            var reply: [8]u8 = undefined;
            while (true) {
                recvAll(self.fds[0], &reply, 60_000) catch |e| switch (e) {
                    error.Timeout => {
                        const waited = seconds(self.io, t0);
                        if (waited > timeout_s) return error.Timeout;
                        std.log.info("rank {d} finished {s}; waiting for the others ({d:.0} s)", .{ self.rank, label, waited });
                        continue;
                    },
                    else => return e,
                };
                break;
            }
            if (getU32(&reply) != magic) return error.BadPeer;
            if (getU32(reply[4..]) != 0) {
                std.log.err("{s}: the ranks were started with different settings (rank 0 says so)", .{label});
                return error.Disagree;
            }
            return;
        }
        var bad = false;
        for (1..self.world) |r| {
            var msg: [16]u8 = undefined;
            while (true) {
                recvAll(self.fds[r], &msg, 60_000) catch |e| switch (e) {
                    error.Timeout => {
                        const waited = seconds(self.io, t0);
                        if (waited > timeout_s) return error.Timeout;
                        std.log.info("rank 0 finished {s}; waiting for rank {d} ({d:.0} s)", .{ label, r, waited });
                        continue;
                    },
                    else => return e,
                };
                break;
            }
            if (getU32(&msg) != magic or getU32(msg[4..]) != tag) {
                std.log.err("{s}: rank {d} is at another barrier", .{ label, r });
                return error.BadPeer;
            }
            const theirs = std.mem.readInt(u64, msg[8..16], .little);
            if (theirs != digest) {
                std.log.err("{s}: rank {d}'s settings digest {x} differs from rank 0's {x}", .{ label, r, theirs, digest });
                bad = true;
            }
        }
        var reply: [8]u8 = undefined;
        putU32(reply[0..], magic);
        putU32(reply[4..], @intFromBool(bad));
        for (1..self.world) |r| try sendAll(self.fds[r], &reply);
        if (bad) return error.Disagree;
    }
};

test "a world of one has no peer to lose" {
    const one: Rendezvous = .{ .io = std.testing.io, .rank = 0, .world = 1 };
    try std.testing.expectEqual(@as(?usize, null), one.peerClosed());
    // connections not made yet (fds -1) are not reported as closed
    const four: Rendezvous = .{ .io = std.testing.io, .rank = 0, .world = 4 };
    try std.testing.expectEqual(@as(?usize, null), four.peerClosed());
}

test "master addresses" {
    const m = try parseMaster("10.100.10.1:29571");
    try std.testing.expectEqual(@as(u32, 0x0a640a01), m.ip);
    try std.testing.expectEqual(@as(u16, 29571), m.port);
    try std.testing.expectError(error.BadAddress, parseMaster("10.100.10:1"));
    try std.testing.expectError(error.BadAddress, parseMaster("spark1"));
}
