//! LFM3-Omni-2B on Metal: speech in, interleaved text and speech out (LiquidAI's LFM Open License v1.0 weights).
//! The same module surface as omni.zig (the CUDA family), so generate.zig and serve.zig build against either.
pub const config = @import("config.zig");
pub const mel = @import("mel.zig");
pub const common = @import("common.zig");
pub const spec = @import("spec.zig");
pub const stack = @import("metal/stack.zig");
pub const kernels = @import("metal/kernels.zig");
pub const engine = @import("metal/engine.zig");
pub const prompt = @import("prompt.zig");
pub const serve = @import("serve.zig");
pub const wav = @import("wav");

pub const Engine = engine.Engine;

test {
    _ = mel;
    _ = prompt;
    _ = common;
    _ = kernels;
}
