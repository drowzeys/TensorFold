//! MiniMax Music 3 on one CUDA GPU (families/music3, ours, from music3-cuda-118): its kernel image (music3.fatbin),
//! `music3-generate` (caption and lyrics -> WAV, --serve, parity dumps) and `test-music3` (host tests, no GPU). The
//! executable installs into zig-out/bin (music3-generate, or music3-generate.exe on Windows).
const std = @import("std");
const fatbin = @import("cuda_fatbin.zig").fatbin;

/// The family's kernels: the AR GEMVs and sampler and the DiT / DAV glue. torch's default nvcc flags with -O3.
const images = [_][]const u8{"music3"};

/// `cuda`: the runtime (cuBLASLt, the driver); `nvcc` builds this family's image, else `prebuilt` holds it
/// (music3.fatbin), else the step fails with a message.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, cuda: *std.Build.Module, core: *std.Build.Module, nvcc: ?[]const u8, prebuilt: ?[]const u8, sms: []const u8) void {
    const gen_step = b.step("music3-generate", "MiniMax Music 3 lyrics and a caption to a song on one CUDA GPU");
    var files: [images.len]std.Build.LazyPath = undefined;
    if (nvcc) |tool| {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        const version = run.captureStdOut(.{});
        for (images, &files) |name, *f| f.* = fatbin(b, tool, version, .{ .name = name, .flags = &.{"-O3"} }, sms);
    } else if (prebuilt) |dir| {
        for (images, &files) |name, *f| f.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{name}) }));
    } else {
        gen_step.dependOn(&b.addFail("the music3 step needs -Dnvcc or -Dfatbins").step);
        return;
    }
    const family = module(b, target, optimize, cuda, core);
    for (images, files) |name, f| family.addAnonymousImport(b.fmt("fatbin_{s}", .{name}), .{ .root_source_file = f });
    const exe = b.addExecutable(.{ .name = "music3-generate", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/music3/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "music3", .module = family } } }) });
    gen_step.dependOn(&b.addInstallArtifact(exe, .{ .pdb_dir = .disabled }).step);
}

/// The music3 family over the given runtime and core (its kernel image is added by `targets`).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/music3/music3.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// `test-music3`: the family's host tests (config, prompt ids, WAV; no GPU, no kernel image).
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const m = module(b, b.graph.host, .debug, cuda, core);
    const empty = b.addWriteFiles().add("music3-host.fatbin", "");
    for (images) |name| m.addAnonymousImport(b.fmt("fatbin_{s}", .{name}), .{ .root_source_file = empty });
    b.step("test-music3", "Host-side tests of the music3 family (config, prompt ids, WAV; no GPU)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
}
