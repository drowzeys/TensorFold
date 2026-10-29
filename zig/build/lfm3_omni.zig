//! LFM3-Omni's CUDA family and `zig build lfm3-omni-generate`; its host tests.
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    // cuBLAS opened at run time and the WAV reader / resampler, both Drama 3's
    const blas = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/cuda/blas.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "cuda", .module = cuda }} });
    const wav = b.createModule(.{ .root_source_file = b.path("zig/src/families/drama3/wav.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm3_omni/omni.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "blas", .module = blas }, .{ .name = "wav", .module = wav } } });
}

pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const family = module(b, target, optimize, cuda, core);
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm3_omni/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "lfm3_omni", .module = family } } });
    const exe = b.addExecutable(.{ .name = "lfm3-omni-generate", .root_module = m });
    b.step("lfm3-omni-generate", "LFM3-Omni speech in, text and speech out on CUDA, with its stage dumps").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const host = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug, cuda, core) }));
    b.step("test-lfm3-omni-host", "LFM3-Omni's mel front end and prompt on the host").dependOn(&host.step);
}
