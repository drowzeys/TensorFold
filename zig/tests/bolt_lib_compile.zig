//! Generate the libbolt lane backend's open and vtable bodies (zig/src/families/nemotron/bolt_native.zig) without a GPU.
const std = @import("std");
const cuda = @import("cuda");
const n = @import("nemotron");
export fn boltLibCompile(ctx: *const cuda.Context, io: *const std.Io, path: [*:0]const u8, images: bool, lanes: u32) void {
    const loaded = n.bolt.open(std.heap.page_allocator, io.*, ctx, std.mem.span(path), null, .{ .context = 256, .drafts = true, .lanes = lanes, .vision = if (images) .{ .image_tokens = 3328, .max_images = 1 } else null }) catch return;
    defer loaded.deinit(loaded.ctx);
    loaded.backend.vtable.verify(loaded.backend.ptr, &.{}, &.{}) catch {};
    _ = n.bolt.snap_bytes(loaded.target, 1);
    _ = n.bolt.explain(null, error.BoltLoad);
}
