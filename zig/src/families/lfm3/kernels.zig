//! LFM3's BF16 family operators and the shared MLX affine row projections.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const dual = @import("dual_projection.zig");
const row = @import("row_projection.zig");
const frags = @import("../../core/frags.zig");

pub const chunk_keys = 4096;
pub const max_chunks = 64;
pub const QkArgs = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32, rotary_pairs: u32 };
pub const AttentionArgs = extern struct { len: u32, cap: u32, scale: f32, window: u32, chunk_base: u32 = 0 };

pub const AttentionSegment = extern struct { keys: u64, values: u64, base: u32, rows: u32, args: AttentionArgs, pad: u32 = 0 };

pub const ConvSegment = extern struct { state: u64, base: u32, rows: u32 };
comptime {
    std.debug.assert(@sizeOf(ConvSegment) == 16 and @alignOf(ConvSegment) == 8);
    std.debug.assert(@offsetOf(ConvSegment, "state") == 0 and @offsetOf(ConvSegment, "base") == 8);
    std.debug.assert(@offsetOf(ConvSegment, "rows") == 12);
}

pub const names = [_][:0]const u8{
    "lfm3_embed_b4",      "lfm3_embed_b6",            "lfm3_embed_b8",   "lfm3_norm",     "lfm3_post_add",      "lfm3_add",    "lfm3_post_norm",
    "lfm3_attn_segments", "lfm3_attn_segments_merge", "lfm3_conv",       "lfm3_qk",       "lfm3_gate",          "lfm3_swiglu", "lfm3_attn_partial",
    "lfm3_attn_merge",    "lfm3_conv_segments",       "lfm3_embed_bf16", "lfm3_row_copy", "lfm3_set_attention",
};

/// d1's set attention constants (attention.metal's lfm3_set_args).
pub const SetArgs = extern struct { rows: u32, cap: u32, state: u32, scale: f32 };

/// The bf16 release's projections: Spark2.5's row kernels to 32 rows (sp_matvec), its tensor-unit GEMM past them.
pub const Dense = struct {
    matvec: [6]mtl.Pipeline, // 1, 2, 4, 8, 16 and 32 rows held in registers
    gemm: [2][2]mtl.Pipeline, // [M % 64 == 0][N % 128 == 0]

    pub const most_rows = [_]usize{ 1, 2, 4, 8, 16, 32 };

    pub fn load(device: mtl.Device) !Dense {
        var out: Dense = undefined;
        {
            const options: mtl.CompileOptions = .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
            const lib = try mtl.Library.fromSource(device, sources.spark25_glue, options);
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (out.matvec[0..made]) |p| p.deinit();
            inline for (most_rows, 0..) |n, i| {
                out.matvec[i] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("sp_matvec_{d}", .{n}), false);
                made += 1;
            }
        }
        errdefer for (out.matvec) |p| p.deinit();
        out.gemm = try loadGemm(device);
        return out;
    }

    pub fn deinit(self: *Dense) void {
        for (self.matvec) |p| p.deinit();
        for (self.gemm) |row_| for (row_) |p| p.deinit();
    }
};

/// Spark2.5's tensor-unit GEMM (MLX's gemm_nax, fp32 sums): bf16 x [M, K] by a stored W [N, K], by tile fit.
pub fn loadGemm(device: mtl.Device) ![2][2]mtl.Pipeline {
    const gpa = std.heap.c_allocator;
    const body = try frags.source(device, gpa, sources.spark25_gemm);
    defer gpa.free(body);
    const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var out: [2][2]mtl.Pipeline = undefined;
    var made: usize = 0;
    errdefer for (0..made) |i| out[i / 2][i % 2].deinit();
    inline for (0..4) |i| {
        out[i / 2][i % 2] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("sp_gemm_{d}{d}", .{ i / 2, i % 2 }), false);
        made += 1;
    }
    return out;
}

/// y [rows, n] = x [rows, k] W^T through `loadGemm`'s pipelines; rows `n` apart.
pub fn gemm(e: mtl.ComputeEncoder, g: *const [2][2]mtl.Pipeline, x: mtl.Buffer, x_off: usize, w: mtl.Buffer, w_off: usize, y: mtl.Buffer, y_off: usize, rows: usize, n: usize, k: usize) void {
    const tn = (n + 127) / 128;
    const tm = (rows + 63) / 64;
    e.setPipeline(g[@intFromBool(rows % 64 == 0)][@intFromBool(n % 128 == 0)]);
    e.setBuffer(x, x_off, 0);
    e.setBuffer(w, w_off, 1);
    var p: [16]i32 = @splat(0);
    const vals = [_]usize{ rows, n, k, k, k, n, tn, tm, 2, k / 256 };
    for (vals, 0..) |v, i| p[i] = @intCast(v);
    e.setBytes(std.mem.asBytes(&p), 2);
    e.setBuffer(y, y_off, 3);
    e.dispatchThreads(mtl.Size.of(tn * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
}

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    rows: row.Pipelines,
    dual: dual.Pipelines,
    dense: ?Dense = null,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        inline for (names, 0..) |name, i| if (comptime std.mem.eql(u8, name, key)) return self.pipelines[i];
        @compileError("no LFM3 kernel " ++ key);
    }

    pub fn embed(self: *const Kernels, bits: u8) mtl.Pipeline {
        return switch (bits) {
            4 => self.get("lfm3_embed_b4"),
            6 => self.get("lfm3_embed_b6"),
            8 => self.get("lfm3_embed_b8"),
            16 => self.get("lfm3_embed_bf16"),
            else => unreachable,
        };
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |pipeline| pipeline.deinit();
        self.rows.deinit();
        self.dual.deinit();
        if (self.dense) |*d| d.deinit();
    }
};

pub fn load(device: mtl.Device) !Kernels {
    const gpa = std.heap.c_allocator;
    const source = try std.mem.concat(gpa, u8, &.{ sources.lfm3, "\n", sources.lfm3_attention });
    defer gpa.free(source);
    const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer library.deinit();
    var out: Kernels = undefined;
    var loaded: usize = 0;
    errdefer for (out.pipelines[0..loaded]) |pipeline| pipeline.deinit();
    for (names, 0..) |name, i| {
        out.pipelines[i] = try mtl.Pipeline.init(device, library, name, false);
        loaded += 1;
    }
    out.rows = try row.Pipelines.load(device);
    errdefer out.rows.deinit();
    out.dual = try dual.Pipelines.load(device);
    out.dense = null;
    return out;
}
