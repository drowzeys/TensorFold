//! cuBLASLt opened at run time: descriptors, layouts, heuristics and matmul, with fp32 compute only (no TF32 or fast modes).

const std = @import("std");
const abi = @import("abi.zig");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const MatmulDesc = ?*opaque {};
pub const Layout = ?*opaque {};
pub const Preference = ?*opaque {};

pub const Algo = extern struct { data: [8]u64 };

pub const Heuristic = extern struct {
    algo: Algo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [4]c_int,
};

pub const DataType = enum(c_int) { f32 = 0, f16 = 2, bf16 = 14, e4m3 = 28 };
pub const compute_32f: c_int = 68;
pub const Op = enum(c_int) { n = 0, t = 1 };
pub const Order = enum(c_int) { col = 0, row = 1 };

pub const desc_transa: c_int = 3;
pub const desc_transb: c_int = 4;
pub const layout_order: c_int = 1;
pub const pref_max_workspace_bytes: c_int = 1;
pub const desc_a_scale_pointer: c_int = 17;
pub const desc_b_scale_pointer: c_int = 18;
pub const desc_a_scale_mode: c_int = 31;
pub const desc_b_scale_mode: c_int = 32;
/// CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 (MXFP8): one power-of-two scale a 32 values along K, in the 128 x 4 tiles
/// glm53_rows_mx8 writes. On a GB10 with cuBLASLt 13.1 the e4m3 modes with heuristics are this and one scalar a
/// tensor (per-row "outer vector" and the 128-block modes report NOT_SUPPORTED).
pub const scale_vec32_ue8m0: c_int = 2;

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasFailed, NoAlgorithm, Invalid };

const S = Status;
const CPtr = ?*const anyopaque;

/// Device pointers are declared as u64: the same register class as `void*` on every LP64 target.
pub const Api = struct {
    cublasLtCreate: *const fn (*Handle) callconv(.c) S,
    cublasLtDestroy: *const fn (Handle) callconv(.c) S,
    cublasLtGetVersion: *const fn () callconv(.c) usize,
    cublasLtGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasLtMatmulDescCreate: *const fn (*MatmulDesc, c_int, DataType) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (MatmulDesc) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (MatmulDesc, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Layout, DataType, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Layout) callconv(.c) S,
    cublasLtMatrixLayoutSetAttribute: *const fn (Layout, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulPreferenceCreate: *const fn (*Preference) callconv(.c) S,
    cublasLtMatmulPreferenceDestroy: *const fn (Preference) callconv(.c) S,
    cublasLtMatmulPreferenceSetAttribute: *const fn (Preference, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulAlgoGetHeuristic: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, Preference, c_int, [*]Heuristic, *c_int) callconv(.c) S,
    cublasLtMatmul: *const fn (Handle, MatmulDesc, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, ?*const Algo, abi.DevicePtr, usize, abi.Stream) callconv(.c) S,
};

pub const Library = struct {
    lib: std.DynLib,
    api: Api,

    /// The CUDA 13 soname first, then the unversioned link name.
    pub fn open() Error!Library {
        for ([_][]const u8{ "libcublasLt.so.13", "libcublasLt.so" }) |path| {
            return openPath(path) catch |e| switch (e) {
                error.LibraryUnavailable => continue,
                else => return e,
            };
        }
        return error.LibraryUnavailable;
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

    pub fn check(self: *const Library, s: Status, what: []const u8) Error!void {
        if (s == 0) return;
        const n = self.api.cublasLtGetStatusName(s);
        std.log.err("{s}: {s} ({d})", .{ what, if (n) |p| std.mem.span(p) else "?", s });
        return error.CublasFailed;
    }
};

/// One row-major bf16 product D[m,n] = X[m,k] . W[n,k]^T (a torch Linear), fp32 accumulate, bf16 or fp32 out.
pub const Linear = struct {
    lt: *const Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    pref: Preference,
    algo: Algo,
    workspace_need: usize,

    /// Picks the heuristic's first algorithm for this shape once; `workspace_limit` bounds what it may ask for.
    pub fn init(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize) Error!Linear {
        var self: Linear = undefined;
        self.lt = lt;
        try lt.check(lt.api.cublasLtCreate(&self.handle), "cublasLtCreate");
        errdefer _ = lt.api.cublasLtDestroy(self.handle);
        try lt.check(lt.api.cublasLtMatmulDescCreate(&self.desc, compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = lt.api.cublasLtMatmulDescDestroy(self.desc);
        const ta: c_int = @backingInt(Op.t);
        const tb: c_int = @backingInt(Op.n);
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transb, &tb, @sizeOf(c_int)), "set transb");
        // column-major view: D^T[n,m] = W[n,k] (stored k x n, transposed) . X^T[k,m]
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.a, .bf16, k, n, @intCast(k)), "layout W");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.a);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.b, .bf16, k, m, @intCast(k)), "layout X");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.b);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.c, out, n, m, @intCast(n)), "layout D");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.c);
        try lt.check(lt.api.cublasLtMatmulPreferenceCreate(&self.pref), "cublasLtMatmulPreferenceCreate");
        errdefer _ = lt.api.cublasLtMatmulPreferenceDestroy(self.pref);
        const limit: u64 = workspace_limit;
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        var found: [1]Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.c, self.c, self.pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        self.algo = found[0].algo;
        self.workspace_need = found[0].workspace_size;
        return self;
    }

    pub fn deinit(self: *Linear) void {
        const api = self.lt.api;
        _ = api.cublasLtMatmulPreferenceDestroy(self.pref);
        _ = api.cublasLtMatrixLayoutDestroy(self.c);
        _ = api.cublasLtMatrixLayoutDestroy(self.b);
        _ = api.cublasLtMatrixLayoutDestroy(self.a);
        _ = api.cublasLtMatmulDescDestroy(self.desc);
        _ = api.cublasLtDestroy(self.handle);
        self.* = undefined;
    }

    /// D = X . W^T on `stream`; the workspace must hold `workspace_need` bytes.
    pub fn run(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream) Error!void {
        if (workspace_len < self.workspace_need) return error.Invalid;
        const one: f32 = 1;
        const zero: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle, self.desc, &one, w, self.a, x, self.b, &zero, d, self.c, d, self.c, &self.algo, workspace, workspace_len, stream), "cublasLtMatmul");
    }
};

/// D[m,n] = X[m,k] . W[n,k]^T as `Linear`, for any input type cuBLASLt takes on this GPU: fp16 (W and X fp16, D fp32
/// or fp16) or e4m3 with MXFP8 scales (D bf16 or fp32). Keeps up to `max_algos` of the heuristic's algorithms; `pick`
/// chooses which one `run` uses (the caller times them: the heuristic's first is not always the fastest).
pub const Gemm = struct {
    pub const max_algos = 8;
    lt: *const Library,
    handle: Handle,
    desc: MatmulDesc,
    a: Layout,
    b: Layout,
    c: Layout,
    algos: [max_algos]Heuristic,
    count: usize,
    pick: usize = 0,
    mx: bool,

    /// `placeholder`: any device address, set as the scale pointers while the heuristic runs (it refuses e4m3 with
    /// block scales and no scale pointers: CUBLAS_STATUS_INVALID_VALUE); `run` sets the real ones.
    pub fn init(lt: *const Library, handle: Handle, m: u64, n: u64, k: u64, in: DataType, out: DataType, workspace_limit: usize, placeholder: abi.DevicePtr) Error!Gemm {
        var self: Gemm = undefined;
        self.lt = lt;
        self.handle = handle;
        self.pick = 0;
        self.mx = in == .e4m3;
        try lt.check(lt.api.cublasLtMatmulDescCreate(&self.desc, compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = lt.api.cublasLtMatmulDescDestroy(self.desc);
        const ta: c_int = @backingInt(Op.t);
        const tb: c_int = @backingInt(Op.n);
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transb, &tb, @sizeOf(c_int)), "set transb");
        if (self.mx) {
            const mode: c_int = scale_vec32_ue8m0;
            try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_a_scale_mode, &mode, @sizeOf(c_int)), "set A scale mode");
            try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_b_scale_mode, &mode, @sizeOf(c_int)), "set B scale mode");
            try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_a_scale_pointer, &placeholder, @sizeOf(u64)), "set A scale");
            try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_b_scale_pointer, &placeholder, @sizeOf(u64)), "set B scale");
        }
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.a, in, k, n, @intCast(k)), "layout W");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.a);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.b, in, k, m, @intCast(k)), "layout X");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.b);
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.c, out, n, m, @intCast(n)), "layout D");
        errdefer _ = lt.api.cublasLtMatrixLayoutDestroy(self.c);
        var pref: Preference = null;
        try lt.check(lt.api.cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
        defer _ = lt.api.cublasLtMatmulPreferenceDestroy(pref);
        const limit: u64 = workspace_limit;
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        var got: c_int = 0;
        try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(handle, self.desc, self.a, self.b, self.c, self.c, pref, max_algos, &self.algos, &got), "cublasLtMatmulAlgoGetHeuristic");
        self.count = 0;
        for (self.algos[0..@intCast(got)]) |h| {
            if (h.state != 0) continue;
            self.algos[self.count] = h;
            self.count += 1;
        }
        if (self.count == 0) return error.NoAlgorithm;
        return self;
    }

    pub fn deinit(self: *Gemm) void {
        const api = self.lt.api;
        _ = api.cublasLtMatrixLayoutDestroy(self.c);
        _ = api.cublasLtMatrixLayoutDestroy(self.b);
        _ = api.cublasLtMatrixLayoutDestroy(self.a);
        _ = api.cublasLtMatmulDescDestroy(self.desc);
        self.* = undefined;
    }

    /// D = X . W^T with algorithm `algo` (null: the picked one); `sw` / `sx` the MXFP8 scale factors (e4m3 only).
    pub fn run(self: *Gemm, x: abi.DevicePtr, w: abi.DevicePtr, sx: abi.DevicePtr, sw: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: abi.Stream, algo: ?usize) Error!void {
        const h = self.algos[algo orelse self.pick];
        if (workspace_len < h.workspace_size) return error.Invalid;
        if (self.mx) {
            try self.lt.check(self.lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_a_scale_pointer, &sw, @sizeOf(u64)), "set A scale");
            try self.lt.check(self.lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_b_scale_pointer, &sx, @sizeOf(u64)), "set B scale");
        }
        const one: f32 = 1;
        const zero: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle, self.desc, &one, w, self.a, x, self.b, &zero, d, self.c, d, self.c, &h.algo, workspace, workspace_len, stream), "cublasLtMatmul");
    }
};

comptime {
    std.debug.assert(@sizeOf(Algo) == 64 and @sizeOf(Heuristic) == 96);
}
