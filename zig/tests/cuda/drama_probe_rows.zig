//! tf-drama-probe --rows: every layer's MoE input rows, picks and routing weights, appended per chunk in row order.
const std = @import("std");
const cuda = @import("cuda");
const drama = @import("drama");

const config = drama.backbone.config;
const H = config.hidden;
const top = config.top_k;
const kinds = [_][]const u8{ "x.bf16", "picks.i32", "w.f32" };
const row_bytes = [kinds.len]usize{ H * 2, top * 4, top * 4 };

/// One chunk's rows: its request, where they start in the files, and their positions in the request's input.
pub const Chunk = struct { request: []const u8, offset: usize, rows: usize, positions: [2]usize };

/// Which rows the files hold: each row's class and each chunk's place, as layer 0 sees the chunks arrive.
pub const Ledger = struct {
    gpa: std.mem.Allocator,
    semantic: [2]i32,
    /// Rows fed after the prompt are generated, whatever their tokens.
    generated: bool = false,
    /// Rows written; the chunk in flight starts at `row0`.
    rows: usize = 0,
    row0: usize = 0,
    chunks: std.ArrayList(Chunk) = .empty,
    /// The request whose rows run next, and its first row.
    request: []const u8 = "",
    first: usize = 0,

    /// The rows that follow are request `id`'s (borrowed until deinit), its prompt first.
    pub fn begin(l: *Ledger, id: []const u8) void {
        l.request = id;
        l.first = l.rows;
        l.generated = false;
    }

    /// Records the chunk of tokens `ids` and writes its classes: 0 prompt text, 1 prompt audio, 2 generated.
    pub fn note(l: *Ledger, ids: []align(1) const i32, classes: []u8) !void {
        for (classes, ids) |*c, id| c.* = if (l.generated) 2 else @intFromBool(id >= l.semantic[0] and id <= l.semantic[1]);
        l.row0 = l.rows;
        l.rows += ids.len;
        try l.chunks.append(l.gpa, .{ .request = l.request, .offset = l.row0, .rows = ids.len, .positions = .{ l.row0 - l.first, l.rows - l.first } });
    }
};

/// DIR/class.u8 [R]; per layer DIR/layer-LL.x.bf16 [R][hidden], layer-LL.picks.i32 and layer-LL.w.f32 [R][top_k].
pub const Rows = struct {
    io: std.Io,
    d: *const cuda.Driver,
    dir: std.Io.Dir,
    class: std.Io.File,
    files: [config.layers][kinds.len]std.Io.File,
    staging: []u8,
    ledger: Ledger,

    /// Creates `path`, which must not exist, for chunks of up to `max_rows` rows.
    pub fn create(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, path: []const u8, max_rows: usize, semantic: [2]u32) !Rows {
        try std.Io.Dir.cwd().createDir(io, path, .default_dir);
        const dir = try std.Io.Dir.cwd().openDir(io, path, .{});
        const ledger: Ledger = .{ .gpa = gpa, .semantic = .{ @intCast(semantic[0]), @intCast(semantic[1]) } };
        var r: Rows = .{ .io = io, .d = d, .dir = dir, .class = undefined, .files = undefined, .staging = try gpa.alloc(u8, max_rows * H * 2), .ledger = ledger };
        r.class = try dir.createFile(io, "class.u8", .{ .exclusive = true });
        var name: [32]u8 = undefined;
        for (&r.files, 0..) |*layer, l| for (layer, kinds) |*f, kind| {
            f.* = try dir.createFile(io, try std.fmt.bufPrint(&name, "layer-{d:0>2}.{s}", .{ l, kind }), .{ .exclusive = true });
        };
        return r;
    }

    pub fn deinit(r: *Rows) void {
        for (&r.files) |*layer| for (layer) |f| f.close(r.io);
        r.class.close(r.io);
        r.dir.close(r.io);
        r.ledger.chunks.deinit(r.ledger.gpa);
        r.ledger.gpa.free(r.staging);
        r.* = undefined;
    }

    /// Layer 0 also appends the chunk's classes, from its tokens; every layer appends its rows at the chunk's offset.
    pub fn routed(r: *Rows, s: cuda.Stream, layer: usize, rt: drama.backbone.Routed) !void {
        try s.synchronize();
        if (layer == 0) {
            const ids = r.staging[0 .. rt.rows * 4];
            try r.download(rt.ids, ids);
            const classes = r.staging[ids.len..][0..rt.rows];
            const at = r.ledger.rows;
            try r.ledger.note(std.mem.bytesAsSlice(i32, ids), classes);
            try r.class.writePositionalAll(r.io, classes, at);
        }
        for (r.files[layer], [_]u64{ rt.input, rt.picks, rt.weights }, row_bytes) |f, src, width| {
            const bytes = r.staging[0 .. rt.rows * width];
            try r.download(src, bytes);
            try f.writePositionalAll(r.io, bytes, r.ledger.row0 * width);
        }
    }

    fn download(r: *const Rows, src: u64, out: []u8) !void {
        const view: cuda.DeviceBuffer = .{ .d = r.d, .ptr = src, .len = out.len, .borrowed = true };
        try view.download(0, out);
    }
};

test "rows are classed by token in the prompt and generated after it; chunks keep request-local positions" {
    var l: Ledger = .{ .gpa = std.testing.allocator, .semantic = .{ 100, 200 } };
    defer l.chunks.deinit(std.testing.allocator);
    var classes: [3]u8 = undefined;
    l.begin("a");
    try l.note(&.{ 7, 100, 200 }, &classes); // prompt chunk: text, then the band's two ends
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 1 }, &classes);
    try l.note(&.{201}, classes[0..1]);
    try std.testing.expectEqual(@as(u8, 0), classes[0]);
    l.generated = true;
    try l.note(&.{150}, classes[0..1]); // a fed frame's row: generated, though its token is audio
    try std.testing.expectEqual(@as(u8, 2), classes[0]);
    try l.note(&.{7}, classes[0..1]);
    try std.testing.expectEqual(@as(u8, 2), classes[0]);
    l.begin("b");
    try l.note(&.{ 150, 7 }, classes[0..2]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 0 }, classes[0..2]);
    const want = [_]Chunk{
        .{ .request = "a", .offset = 0, .rows = 3, .positions = .{ 0, 3 } },
        .{ .request = "a", .offset = 3, .rows = 1, .positions = .{ 3, 4 } },
        .{ .request = "a", .offset = 4, .rows = 1, .positions = .{ 4, 5 } },
        .{ .request = "a", .offset = 5, .rows = 1, .positions = .{ 5, 6 } },
        .{ .request = "b", .offset = 6, .rows = 2, .positions = .{ 0, 2 } },
    };
    try std.testing.expectEqualDeep(@as([]const Chunk, &want), @as([]const Chunk, l.chunks.items));
    try std.testing.expectEqual(@as(usize, 8), l.rows);
}
