//! NCCL opened at run time for tensor-parallel ranks: communicators and the collectives the lane engines use.

const std = @import("std");
const abi = @import("abi.zig");

pub const Result = c_int;
pub const Comm = ?*opaque {};
pub const UniqueId = extern struct { internal: [128]u8 };

pub const DataType = enum(c_int) { i8 = 0, u8 = 1, i32 = 2, u32 = 3, i64 = 4, u64 = 5, f16 = 6, f32 = 7, f64 = 8, bf16 = 9 };
pub const RedOp = enum(c_int) { sum = 0, prod = 1, max = 2, min = 3, avg = 4 };

pub const Error = error{ LibraryUnavailable, MissingSymbol, NcclFailed };

const R = Result;
const D = abi.DevicePtr;

pub const Api = struct {
    ncclGetVersion: *const fn (*c_int) callconv(.c) R,
    ncclGetUniqueId: *const fn (*UniqueId) callconv(.c) R,
    ncclCommInitRank: *const fn (*Comm, c_int, UniqueId, c_int) callconv(.c) R,
    ncclCommInitAll: *const fn ([*]Comm, c_int, ?[*]const c_int) callconv(.c) R,
    ncclCommDestroy: *const fn (Comm) callconv(.c) R,
    ncclGetErrorString: *const fn (R) callconv(.c) ?[*:0]const u8,
    ncclAllReduce: *const fn (D, D, usize, DataType, RedOp, Comm, abi.Stream) callconv(.c) R,
    ncclAllGather: *const fn (D, D, usize, DataType, Comm, abi.Stream) callconv(.c) R,
    ncclBroadcast: *const fn (D, D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclGroupStart: *const fn () callconv(.c) R,
    ncclGroupEnd: *const fn () callconv(.c) R,
};

pub const Library = struct {
    lib: std.DynLib,
    api: Api,

    pub fn open() Error!Library {
        return openPath("libnccl.so.2");
    }

    pub fn openPath(path: []const u8) Error!Library {
        var lib = std.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn check(self: *const Library, r: Result, what: []const u8) Error!void {
        if (r == 0) return;
        const s = self.api.ncclGetErrorString(r);
        std.log.err("{s}: {s} ({d})", .{ what, if (s) |p| std.mem.span(p) else "?", r });
        return error.NcclFailed;
    }

    /// NCCL's version as 10000 * major + 100 * minor + patch.
    pub fn version(self: *const Library) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.ncclGetVersion(&v), "ncclGetVersion");
        return v;
    }
};

/// One rank's NCCL communicator (ncclCommInitRank) and the collectives the GLM-5.3 engine runs on its compute stream:
/// every call is enqueued on `stream` (stream order is the only ordering; nothing here synchronizes).
pub const Communicator = struct {
    lib: *const Library,
    comm: Comm,
    rank: usize,
    world: usize,

    /// Rank 0 makes the id (ncclGetUniqueId) and hands its 128 bytes to the other ranks (rendezvous.zig).
    pub fn uniqueId(lib: *const Library) Error!UniqueId {
        var id: UniqueId = undefined;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        return id;
    }

    /// Joins the world: blocks until every rank has called it with the same id (the CUDA context must be current).
    pub fn init(lib: *const Library, id: UniqueId, rank: usize, world: usize) Error!Communicator {
        var c: Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }

    pub fn deinit(self: *Communicator) void {
        _ = self.lib.api.ncclCommDestroy(self.comm);
        self.* = undefined;
    }

    /// recv [world * count] <- every rank's send [count] in rank order (comm.NCCL.all_gather).
    pub fn allGather(self: *const Communicator, send: D, recv: D, count: usize, dt: DataType, stream: abi.Stream) Error!void {
        try self.lib.check(self.lib.api.ncclAllGather(send, recv, count, dt, self.comm, stream), "ncclAllGather");
    }

    /// recv <- the sum of every rank's send (NCCL's ring: ranks alike, the order depends on the size).
    pub fn allReduce(self: *const Communicator, send: D, recv: D, count: usize, dt: DataType, op: RedOp, stream: abi.Stream) Error!void {
        try self.lib.check(self.lib.api.ncclAllReduce(send, recv, count, dt, op, self.comm, stream), "ncclAllReduce");
    }
};
