//! cuBLAS (the classic API) opened at run time, for the one call torch makes that the GLM-5.3 port must repeat bit
//! for bit: torch.bmm on bf16 (at::cuda::blas::bgemm -> cublasGemmStridedBatchedEx, fp32 compute, the default
//! tensor-op algorithm), fused.attention_core's absorb / expand for prompt chunks. The same library file, the same
//! handle settings (workspace size, math mode) and the same arguments give cuBLAS's heuristic the same inputs, so it
//! picks the same kernel; tools/glm53 checks the bits on a probe before any prompt (--bmm-probe).

const std = @import("std");
const abi = @import("abi.zig");

pub const Status = c_int;
pub const Handle = ?*opaque {};

pub const Op = enum(c_int) { n = 0, t = 1 };
/// cudaDataType_t
pub const DataType = enum(c_int) { f32 = 0, f16 = 2, bf16 = 14 };
/// cublasComputeType_t CUBLAS_COMPUTE_32F
pub const compute_32f: c_int = 68;
/// cublasGemmAlgo_t CUBLAS_GEMM_DEFAULT_TENSOR_OP (torch's choice for bgemm)
pub const gemm_default_tensor_op: c_int = 99;
/// cublasMath_t: CUBLAS_DEFAULT_MATH; | 16 = CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
pub const default_math: c_int = 0;
pub const disallow_reduced_precision: c_int = 16;

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasFailed };

const S = Status;
const CPtr = ?*const anyopaque;

pub const Api = struct {
    cublasCreate_v2: *const fn (*Handle) callconv(.c) S,
    cublasDestroy_v2: *const fn (Handle) callconv(.c) S,
    cublasGetVersion_v2: *const fn (Handle, *c_int) callconv(.c) S,
    cublasSetStream_v2: *const fn (Handle, abi.Stream) callconv(.c) S,
    cublasSetWorkspace_v2: *const fn (Handle, abi.DevicePtr, usize) callconv(.c) S,
    cublasSetMathMode: *const fn (Handle, c_int) callconv(.c) S,
    cublasGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasGemmStridedBatchedEx: *const fn (Handle, Op, Op, c_int, c_int, c_int, CPtr, abi.DevicePtr, DataType, c_int, c_longlong, abi.DevicePtr, DataType, c_int, c_longlong, CPtr, abi.DevicePtr, DataType, c_int, c_longlong, c_int, c_int, c_int) callconv(.c) S,
};

pub const Library = struct {
    lib: std.DynLib,
    api: Api,

    /// The CUDA 13 soname first, then the unversioned link name (tools/glm53 passes the file torch loaded).
    pub fn open() Error!Library {
        // absolute paths: std.DynLib without libc does not search LD_LIBRARY_PATH / ld.so.cache. torch's copy first
        // (the same cuBLAS as the Python engine: bit identity), then the toolkit's, then the bare names.
        for ([_][]const u8{ "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcublas.so.13",
            "/usr/local/cuda/lib64/libcublas.so.13", "libcublas.so.13", "libcublas.so" }) |path| {
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
        const name = self.api.cublasGetStatusName(s);
        std.log.err("{s}: {s} ({d})", .{ what, if (name) |p| std.mem.span(p) else "?", s });
        return error.CublasFailed;
    }
};

/// One handle set up as torch sets up its handle for a stream (getCurrentCUDABlasHandle): its workspace
/// (cublasSetWorkspace, torch's default size for this GPU unless CUBLAS_WORKSPACE_CONFIG says otherwise) and math mode.
pub const Blas = struct {
    lib: *const Library,
    h: Handle,
    workspace: abi.DevicePtr,
    workspace_bytes: usize,

    pub fn init(lib: *const Library, workspace: abi.DevicePtr, workspace_bytes: usize, math: c_int) Error!Blas {
        var h: Handle = null;
        try lib.check(lib.api.cublasCreate_v2(&h), "cublasCreate");
        errdefer _ = lib.api.cublasDestroy_v2(h);
        if (workspace_bytes > 0) try lib.check(lib.api.cublasSetWorkspace_v2(h, workspace, workspace_bytes), "cublasSetWorkspace");
        try lib.check(lib.api.cublasSetMathMode(h, math), "cublasSetMathMode");
        return .{ .lib = lib, .h = h, .workspace = workspace, .workspace_bytes = workspace_bytes };
    }

    pub fn deinit(self: *Blas) void {
        _ = self.lib.api.cublasDestroy_v2(self.h);
        self.* = undefined;
    }

    pub fn version(self: *const Blas) Error!c_int {
        var v: c_int = 0;
        try self.lib.check(self.lib.api.cublasGetVersion_v2(self.h, &v), "cublasGetVersion");
        return v;
    }

    /// at::cuda::blas::bgemm<at::BFloat16> (cuBLAS backend): column-major C[b] = op(A[b]) op(B[b]) for b < batch, bf16
    /// in and out, fp32 compute and scalars, CUBLAS_GEMM_DEFAULT_TENSOR_OP, on `stream`.
    pub fn bgemmBf16(self: *const Blas, stream: abi.Stream, ta: Op, tb: Op, m: usize, n: usize, k: usize, a: abi.DevicePtr, lda: usize, sa: usize, b: abi.DevicePtr, ldb: usize, sb: usize, c: abi.DevicePtr, ldc: usize, sc: usize, batch: usize) Error!void {
        try self.lib.check(self.lib.api.cublasSetStream_v2(self.h, stream), "cublasSetStream");
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        try self.lib.check(self.lib.api.cublasGemmStridedBatchedEx(self.h, ta, tb, @intCast(m), @intCast(n), @intCast(k), &alpha, a, .bf16, @intCast(lda), @intCast(sa), b, .bf16, @intCast(ldb), @intCast(sb), &beta, c, .bf16, @intCast(ldc), @intCast(sc), @intCast(batch), compute_32f, gemm_default_tensor_op), "cublasGemmStridedBatchedEx");
    }
};

/// torch's default cuBLAS workspace (CUDABlasHandlePool getDefaultWorkspaceSize): 4096 KiB * 8 on sm_90 / sm_100,
/// else 4096 KiB * 2 + 16 KiB * 8 (GB10, sm_121: 8,519,680 bytes). CUBLAS_WORKSPACE_CONFIG ":SIZE_KIB:COUNT..." overrides.
pub fn torchWorkspaceBytes(major: c_int, config: ?[]const u8) usize {
    if (config) |text| {
        var total: usize = 0;
        var it = std.mem.tokenizeScalar(u8, text, ':');
        while (it.next()) |size_s| {
            const count_s = it.next() orelse break;
            const size = std.fmt.parseInt(usize, size_s, 10) catch return 0;
            const count = std.fmt.parseInt(usize, count_s, 10) catch return 0;
            total += size * 1024 * count;
        }
        if (total > 0) return total;
    }
    if (major == 9 or major == 10) return 4096 * 1024 * 8;
    return 4096 * 1024 * 2 + 16 * 1024 * 8;
}

test "torch's cuBLAS workspace sizes" {
    try std.testing.expectEqual(@as(usize, 8519680), torchWorkspaceBytes(12, null));
    try std.testing.expectEqual(@as(usize, 33554432), torchWorkspaceBytes(9, null));
    try std.testing.expectEqual(@as(usize, 4096 * 1024 * 2 + 16 * 1024 * 8), torchWorkspaceBytes(12, ":4096:2:16:8"));
}
