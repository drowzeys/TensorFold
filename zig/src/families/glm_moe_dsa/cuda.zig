//! Full GLM-5.3 (glm_moe_dsa) on CUDA in Zig: the Python glm_moe_dsa engine's fused path (MLA latent attention, the DSA
//! indexer, EXL3 dense linears and routed experts), its kernels and layouts, so each layer matches it bit for bit.
//! Phase 1: the first N layers on one GB10 as rank 0 of world 4 with the other ranks absent (identity comm: own
//! partial, zeros elsewhere), zig/tests/cuda/glm53_layers.zig against tools/glm53/oracle.py.
//! Phase 2a: the full model at TP over NCCL (Forward.comm), the vocabulary-sharded argmax head (Forward.head), greedy,
//! zig/tests/cuda/glm53_generate.zig against tools/glm53/reference.py.
//! Phase 2b: the MTP layer, the 4-bit draft head, MTP-drafted rounds replayed as CUDA graphs and the Python engine's
//! keyed sampling (cuda_runner.zig, sampling.zig), the O_DIRECT / io_uring load (cuda_source.zig);
//! zig/tests/cuda/glm53_generate.zig --runner against tools/glm53/reference2b.py.
//! Phase 3a: the RoCE one-shot all-reduce / gather for decode windows (cuda_roce.zig over roce_proxy.c and
//! glm53_roce.cu), the served prompt path (cuda_prompt.zig: sequence-parallel chunks of thousands of rows, the prompt
//! GEMM, torch.bmm through cuBLAS, prompt experts) and the served tile table (tiles.zig); zig/tests/cuda/
//! glm53_generate.zig --mode 3a against tools/glm53/reference3a.py (the Python engine in its served configuration).
//! Phase 3a speed: the decode windows' side stream and L2 prefetch (cuda_decode.zig: TF_GLM53_SIDE / TF_GLM53_L2PF as
//! the served image runs them), the decode-sized multi-block top-k and device-side sampling candidates
//! (glm53_decode.cu), the MTP layer's prompt rows in bigger blocks, the prompt path's timers (cuda_prompt.Prof).
//! Not yet a native family (native/cuda.zig's registry): no engine or server here.

pub const Config = @import("config.zig").Config;
pub const exl3 = @import("exl3.zig");
pub const split = @import("split.zig");
pub const weights = @import("cuda_weights.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const triton = @import("cuda_triton.zig");
pub const forward = @import("cuda_forward.zig");
pub const Forward = forward.Forward;
pub const Buffers = forward.Buffers;
pub const State = forward.State;
pub const sampling = @import("sampling.zig");
pub const source = @import("cuda_source.zig");
pub const runner = @import("cuda_runner.zig");
pub const Runner = runner.Runner;
pub const roce = @import("cuda_roce.zig");
pub const prompt = @import("cuda_prompt.zig");
pub const tiles = @import("tiles.zig");
pub const decode = @import("cuda_decode.zig");

test {
    _ = @import("config.zig");
    _ = exl3;
    _ = split;
    _ = triton;
    _ = forward;
    _ = weights;
    _ = kernels;
    _ = sampling;
    _ = source;
    _ = runner;
    _ = roce;
    _ = prompt;
    _ = tiles;
    _ = decode;
}
