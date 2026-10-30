//! LFM3's CUDA records: head-128 paged prompt attention, the 7-tap Liquid conv, LFM3-VL's glue and tower attention.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;

const vision64 = "_ZN20tf_prefill_attention13vision_kernelILi64ELi8EEEvPK13__nv_bfloat16PS1_iif";
const pattn128x4 = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix";

pub const cuda = [_]R{
    .{ .launch = .{ .name = "liquid_conv tf_liquid_conv_7", .image = "lfm3", .symbol = "tf_liquid_conv_7", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "liquid_conv tf_liquid_commit_7", .image = "lfm3", .symbol = "tf_liquid_commit_7", .threads = 256 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "paging pattn128x4", .image = "prefill_attention", .symbol = pattn128x4, .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 206, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 }, .{ .regs = 231, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "lfm3v tf_lfm3v_rms", .image = "lfm3v", .symbol = "tf_lfm3v_rms", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 32, .bound = 256 }, .{ .regs = 29, .static = 32, .bound = 256 }, .{ .regs = 29, .static = 32, .bound = 256 }, .{ .regs = 34, .static = 32, .bound = 256 }, .{ .regs = 34, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "lfm3v tf_lfm3v_qk", .image = "lfm3v", .symbol = "tf_lfm3v_qk", .threads = 32 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 32 }, .{ .regs = 24, .static = 0, .bound = 32 }, .{ .regs = 24, .static = 0, .bound = 32 }, .{ .regs = 23, .static = 0, .bound = 32 }, .{ .regs = 23, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "lfm3v tf_lfm3v_conv", .image = "lfm3v", .symbol = "tf_lfm3v_conv", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "lfm3v tf_lfm3v_add", .image = "lfm3v", .symbol = "tf_lfm3v_add", .threads = 256 }, .usage = .{ .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "lfm3v tf_lfm3v_silu_mul", .image = "lfm3v", .symbol = "tf_lfm3v_silu_mul", .threads = 256 }, .usage = .{ .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "lfm3v tf_lfm3v_set_attn", .image = "lfm3v", .symbol = "tf_lfm3v_set_attn", .threads = 32 }, .usage = .{ .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 }, .{ .regs = 34, .static = 0, .bound = 32 } } },
    .{ .launch = .{ .name = "lfm3v tf_lfm3v_set_attn_tiled", .image = "lfm3v", .symbol = "tf_lfm3v_set_attn_tiled", .threads = 512 }, .usage = .{ .{ .regs = 32, .static = 16512, .bound = 512 }, .{ .regs = 37, .static = 16512, .bound = 512 }, .{ .regs = 37, .static = 16512, .bound = 512 }, .{ .regs = 40, .static = 16512, .bound = 512 }, .{ .regs = 40, .static = 16512, .bound = 512 } } },
    .{ .launch = .{ .name = "vit vision_kernel<64, 8> (LFM3-VL)", .image = "prefill_attention", .symbol = vision64, .threads = 256, .dynamic = 36864 }, .usage = .{ .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 } } },
};

/// nvcc 13.2's lfm3v and vision_kernel<64, 8> where it differs from 13.0 (nvcc -cubin per SM, cuobjdump -res-usage).
pub const cuda_132 = [_]registry.Override{
    .{ .name = "lfm3v tf_lfm3v_conv", .usage = .{ .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 } } },
    .{ .name = "lfm3v tf_lfm3v_set_attn_tiled", .usage = .{ .{ .regs = 32, .static = 16512, .bound = 512 }, .{ .regs = 37, .static = 16512, .bound = 512 }, .{ .regs = 37, .static = 16512, .bound = 512 }, .{ .regs = 64, .static = 16512, .bound = 512 }, .{ .regs = 64, .static = 16512, .bound = 512 } } },
    .{ .name = "vit vision_kernel<64, 8> (LFM3-VL)", .usage = .{ .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 } } },
};

test "LFM3 records: every launch fits every SM by its measured record" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
