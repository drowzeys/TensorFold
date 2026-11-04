//! Fish Audio Drama 3 TTS on one CUDA GPU (families/fish_drama3, beside 1.2.2's own Drama family): its kernel images
//! (fish_drama3*.fatbin), `fish-drama3-generate` (drama3-generate: text -> WAV, serve, encode, checks), `fish-drama3-lib`
//! (libtensorfold_drama3.so / tensorfold_drama3.dll, the C ABI) and `test-fish-drama3` (host tests, no GPU). Both
//! artifacts install into zig-out/fish-drama3 (bin/ and lib/; the DLL beside the executable on Windows).
const std = @import("std");
const fatbin = @import("cuda_fatbin.zig").fatbin;

/// The family's kernels: backbone glue, the GDN recurrence, the codebook tower and samplers; the routed FP8 experts with
/// per-row activation scales; the codec (fp32 between its cuBLAS SGEMMs). torch's default nvcc flags with -O3.
const images = [_][]const u8{ "fish_drama3", "fish_drama3_fp8", "fish_drama3_codec" };

/// `cuda`: the runtime with every shared image (the family loads its experts image); `nvcc` builds this family's
/// images, else `prebuilt` holds them (fish_drama3*.fatbin), else the steps fail with a message.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, cuda: *std.Build.Module, core: *std.Build.Module, nvcc: ?[]const u8, prebuilt: ?[]const u8, sms: []const u8) void {
    const gen_step = b.step("fish-drama3-generate", "drama3-generate: Fish Audio Drama 3 TTS on one GPU, text in, WAV out (CUDA)");
    const lib_step = b.step("fish-drama3-lib", "libtensorfold_drama3 / tensorfold_drama3.dll: Fish Audio Drama 3 TTS behind a C ABI");
    var files: [images.len]std.Build.LazyPath = undefined;
    if (nvcc) |tool| {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        const version = run.captureStdOut(.{});
        for (images, &files) |name, *f| f.* = fatbin(b, tool, version, .{ .name = name, .flags = &.{"-O3"} }, sms);
    } else if (prebuilt) |dir| {
        for (images, &files) |name, *f| f.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{name}) }));
    } else {
        for ([_]*std.Build.Step{ gen_step, lib_step }) |s| s.dependOn(&b.addFail("the Fish Drama 3 steps need -Dnvcc or -Dfatbins").step);
        return;
    }
    const family = module(b, target, optimize, cuda, core);
    for (images, files) |name, f| family.addAnonymousImport(b.fmt("fatbin_{s}", .{name}), .{ .root_source_file = f });
    const dir: std.Build.InstallDir = .{ .custom = "fish-drama3" };
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "drama3", .module = family } };
    const exe = b.addExecutable(.{ .name = "drama3-generate", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/fish_drama3/generate.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = imports }) });
    gen_step.dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = dir }, .dest_sub_path = b.fmt("bin/{s}", .{exe.out_filename}), .pdb_dir = .disabled }).step);
    const lib = b.addLibrary(.{ .name = "tensorfold_drama3", .linkage = .dynamic, .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/families/fish_drama3/capi.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = imports }) });
    const lib_sub = if (target.result.os.tag == .windows) "bin" else "lib";
    lib_step.dependOn(&b.addInstallArtifact(lib, .{ .dest_dir = .{ .override = dir }, .dest_sub_path = b.fmt("{s}/{s}", .{ lib_sub, lib.out_filename }), .implib_dir = .disabled, .pdb_dir = .disabled }).step);
}

/// The fish_drama3 family over the given runtime and core (its kernel images are added by `targets`).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/fish_drama3/drama3.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// `test-fish-drama3`: the family's host tests (config, prompt rows; no GPU, no kernel images).
pub fn hostTests(b: *std.Build, cuda: *std.Build.Module, core: *std.Build.Module) void {
    const host = b.graph.host;
    const m = module(b, host, .debug, cuda, core);
    const empty = b.addWriteFiles().add("fish-drama3-host.fatbin", "");
    for (images) |name| m.addAnonymousImport(b.fmt("fatbin_{s}", .{name}), .{ .root_source_file = empty });
    b.step("test-fish-drama3", "Host-side tests of the Fish Drama 3 family (config, prompt rows; no GPU)").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
}
