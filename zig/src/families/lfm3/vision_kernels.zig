//! LFM3-VL BF16 image operators with FP32 reductions; an image's products on the prompt GEMM (tensor units if any).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
pub const Ref = @import("vision_weights.zig").Ref;
const text = @import("kernels.zig");

pub const names = [_][]const u8{
    "l3v_dense",
    "l3v_rms",
    "l3v_add",
    "l3v_conv",
    "l3v_qk",
    "l3v_attention",
    "l3v_swiglu",
    "l3v_layer_norm",
    "l3v_project",
    "l3v_gelu",
    "l3v_attention_tiled",
};

pub const Kernels = struct {
    p: [names.len]mtl.Pipeline,
    gemm: [2][2]mtl.Pipeline, // the tensor-unit GEMM the text decoder's bf16 release takes (kernels.zig)

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        _ = gpa;
        const library = try mtl.Library.fromSource(device, sources.lfm3_vision, mtl.CompileOptions.mlx());
        defer library.deinit();
        var out: Kernels = undefined;
        var loaded: usize = 0;
        errdefer for (out.p[0..loaded]) |p| p.deinit();
        for (names, 0..) |name, i| {
            out.p[i] = try mtl.Pipeline.init(device, library, name, false);
            loaded += 1;
        }
        out.gemm = try text.loadGemm(device);
        return out;
    }

    pub fn deinit(self: *Kernels) void {
        for (self.p) |p| p.deinit();
        for (self.gemm) |row| for (row) |p| p.deinit();
    }

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        inline for (names, 0..) |name, i| if (comptime std.mem.eql(u8, name, key)) return self.p[i];
        @compileError("no LFM3 vision pipeline " ++ key);
    }

    /// Dense BF16 x W^T with FP32 sums: an image's rows on the tensor-unit GEMM, a few on blocks of four.
    pub fn linear(self: *const Kernels, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, rows: usize, n: usize, k: usize) void {
        if (rows > 32) return text.gemm(e, &self.gemm, x.buf, x.off, w.buf, w.off, y.buf, y.off, rows, n, k);
        e.setPipeline(self.get("l3v_dense"));
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(y.buf, y.off, 2);
        e.setValue(extern struct { m: u32, n: u32, k: u32 }{ .m = @intCast(rows), .n = @intCast(n), .k = @intCast(k) }, 3);
        e.dispatchGroups(mtl.Size.of((n + 3) / 4, (rows + 3) / 4, 1), mtl.Size.of(32, 1, 1));
    }
};
