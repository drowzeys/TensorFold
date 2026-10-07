//! Qwen-Image-2.1's kernels: one Metal source, compiled when the model loads.
pub const source = @embedFile("qwen_image.metal");
pub const names = [_][:0]const u8{ "qi_gemm", "qi_tile_gemm", "qi_norm_scale", "qi_heads", "qi_softmax", "qi_rows", "qi_gate_add", "qi_swiglu" };
