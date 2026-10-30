//! EmbeddingGemma-2's Metal launches (native/embgemma2_model.zig) reserve their widths in source.
const M = @import("limits.zig").MetalLaunch;

/// The tiled attention's threadgroup memory the encoder sets: 32 x 136 fp32 probabilities, a row max per simdgroup.
pub fn tiled(hd: u32) u32 {
    return 4 * (32 * 136 + hd / 64 * 32);
}

pub const metal = [_]M{
    .{ .name = "embgemma2 eg2_gemm_0", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_gemm_1", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_gemm_glu_0", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_gemm_glu_1", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_gemm_ple_0", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_gemm_ple_1", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_16", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_32", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_glu_4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_glu_8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_glu_16", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_glu_32", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_ple_4", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_ple_8", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_ple_16", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_matvec_ple_32", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "embgemma2 eg2_rope_table", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "embgemma2 eg2_rms", .threads = 128, .tg_memory = 16, .reserved = 128 },
    .{ .name = "embgemma2 eg2_qkv", .threads = 128, .tg_memory = 16, .reserved = 128 },
    .{ .name = "embgemma2 eg2_add_norm", .threads = 128, .tg_memory = 16, .reserved = 128 },
    .{ .name = "embgemma2 eg2_pool", .threads = 256, .tg_memory = 2080, .reserved = 256 },
    .{ .name = "embgemma2 eg2_attn256", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "embgemma2 eg2_attn512", .threads = 128, .tg_memory = 0, .reserved = 128 },
    .{ .name = "embgemma2 eg2_attn256_short", .threads = 128, .tg_memory = 4096, .reserved = 128 },
    .{ .name = "embgemma2 eg2_attn512_short", .threads = 128, .tg_memory = 8192, .reserved = 128 },
    .{ .name = "embgemma2 eg2_attn256_tiled", .threads = 128, .tg_memory = 0, .dynamic = tiled(256), .reserved = 128 },
    .{ .name = "embgemma2 eg2_attn512_tiled", .threads = 256, .tg_memory = 0, .dynamic = tiled(512), .reserved = 256 },
};

/// Planted: the tiled attention asking 36 KiB, and a few-row product at 1,024 threads its source does not reserve.
pub const controls = [_]M{
    .{ .name = "control: embgemma2 36 KiB eg2_attn512_tiled", .threads = 256, .tg_memory = 0, .dynamic = 36 * 1024, .reserved = 256 },
    .{ .name = "control: embgemma2 at 1024 threads eg2_matvec_32", .threads = 1024, .tg_memory = 0, .reserved = 256 },
};
