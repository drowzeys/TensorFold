//! MiniMax H3 / FastH3's transformer blocks over the CUDA runtime: the Metal family's int8 arithmetic
//! (families/h3/h3.zig) on the GB10's tensor cores. The caller owns everything around the blocks (text rows, patch
//! projections, modulation tables, the final layer) and hands over the packed stream; a forward runs every block
//! on it in place. Block projections are read from the checkpoint and quantized on the GPU at load.
const std = @import("std");
const cuda = @import("cuda");

pub const head_dim = 128;
pub const eps: f32 = 1e-5;
const slot_tile = 64;
/// Rows are padded to whole product tiles of this many.
const tile = 256;
const wide_group = 1024;
/// The six modulation tables of a block, in the order the checkpoint's projection splits them.
const shift_a = 0;
const scale_a = 1;
const gate_a = 2;
const shift_m = 3;
const scale_m = 4;
const gate_m = 5;

const image align(16) = @embedFile("fatbin_h3").*;
const kernel_names = [_][:0]const u8{ "h3_gemm", "h3_gemm_n256", "h3_gemm_swiglu", "h3_gemm_wide", "h3_quant_weight", "h3_norm_q8", "h3_gate_add", "h3_heads", "h3_pool", "h3_pool_quant", "h3_topk", "h3_mix_quant", "h3_quant_wide", "h3_attention" };

pub const stages = [_][]const u8{ "norms", "q k v gate", "heads", "routing", "attention", "gate mix", "attention out", "mlp in", "quantize", "mlp out" };

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, offset: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) ?*anyopaque;
extern "c" fn munmap(addr: ?*anyopaque, len: usize) c_int;
extern "c" fn pread(fd: c_int, buf: [*]u8, count: usize, offset: i64) isize;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;
extern "c" fn rename(from: [*:0]const u8, to: [*:0]const u8) c_int;
extern "c" fn clock_gettime(id: c_int, ts: *Timespec) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };

pub fn seconds() f64 {
    var ts: Timespec = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// One safetensors file mapped read-only; a tensor is found by its name in the header's text.
pub const Checkpoint = struct {
    bytes: []const u8,
    header: []const u8,
    data: usize,

    pub const Entry = struct { rows: usize, cols: usize, bytes: []const u8 };

    pub fn open_(path: [*:0]const u8) !Checkpoint {
        const fd = open(path, 0);
        if (fd < 0) return error.CannotOpen;
        defer _ = close(fd);
        const size = lseek(fd, 0, 2);
        if (size < 16) return error.CannotOpen;
        const len: usize = @intCast(size);
        const at = mmap(null, len, 1, 2, fd, 0) orelse return error.CannotMap;
        if (@intFromPtr(at) == std.math.maxInt(usize)) return error.CannotMap;
        const bytes = @as([*]const u8, @ptrCast(at))[0..len];
        const n: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        if (n + 8 > len) return error.BadHeader;
        return .{ .bytes = bytes, .header = bytes[8 .. 8 + n], .data = 8 + n };
    }

    pub fn deinit(self: *Checkpoint) void {
        _ = munmap(@constCast(@ptrCast(self.bytes.ptr)), self.bytes.len);
    }

    fn numbers(text: []const u8, comptime key: []const u8, out: []usize) !usize {
        const at = std.mem.indexOf(u8, text, key) orelse return error.BadHeader;
        const end = std.mem.indexOfScalarPos(u8, text, at, ']') orelse return error.BadHeader;
        var it = std.mem.tokenizeAny(u8, text[at + key.len .. end], ", ");
        var n: usize = 0;
        while (it.next()) |word| : (n += 1) {
            if (n == out.len) return error.BadHeader;
            out[n] = try std.fmt.parseInt(usize, word, 10);
        }
        return n;
    }

    /// The bf16 tensor `name`, a vector (one row) or a matrix.
    pub fn bf16(self: *const Checkpoint, name: []const u8) !Entry {
        var key: [160]u8 = undefined;
        const quoted = try std.fmt.bufPrint(&key, "\"{s}\":{{", .{name});
        const at = std.mem.indexOf(u8, self.header, quoted) orelse {
            std.debug.print("no tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        const end = std.mem.indexOfScalarPos(u8, self.header, at, '}') orelse return error.BadHeader;
        const text = self.header[at..end];
        if (std.mem.indexOf(u8, text, "\"BF16\"") == null) return error.WrongDType;
        var shape: [2]usize = undefined;
        var span: [2]usize = undefined;
        const dims = try numbers(text, "\"shape\":[", &shape);
        if (try numbers(text, "\"data_offsets\":[", &span) != 2 or dims == 0) return error.BadHeader;
        const rows = if (dims == 2) shape[0] else 1;
        const cols = shape[dims - 1];
        if (span[1] - span[0] != rows * cols * 2) return error.BadHeader;
        return .{ .rows = rows, .cols = cols, .bytes = self.bytes[self.data + span[0] .. self.data + span[1]] };
    }
};

/// An int8 projection (N, K) and its scale per output channel.
const Quant = struct { w: cuda.DeviceBuffer, s: cuda.DeviceBuffer };

/// `norms` holds norm1, norm2 (hidden each), then the heads' q and k norms (128 each), bf16.
const Block = struct { qkvc: Quant, out: Quant, fc1: Quant, fc2: Quant, norms: cuda.DeviceBuffer };

/// What depends on the packed sequence: allocated when its rows or tiles change.
const Geometry = struct {
    rows: usize = 0,
    tiles: usize = 0,
    prefix: usize = 0,
    keep: usize = 0,
    buffers: [40]cuda.DeviceBuffer = undefined,
    used: usize = 0,
    y: u64 = 0,
    q8: u64 = 0,
    xs: u64 = 0,
    qkvc: u64 = 0,
    tq: u64 = 0,
    tqs: u64 = 0,
    tk: u64 = 0,
    tvt: u64 = 0,
    ks: u64 = 0,
    vs: u64 = 0,
    qp: u64 = 0,
    kp: u64 = 0,
    vp: u64 = 0,
    scores: u64 = 0,
    chosen: u64 = 0,
    coarse: u64 = 0,
    // the tiles' means as a sequence of their own, in groups of 64 tiles
    pq: u64 = 0,
    pqs: u64 = 0,
    pk: u64 = 0,
    pks: u64 = 0,
    pvt: u64 = 0,
    pvs: u64 = 0,
    group_sizes: u64 = 0,
    group_rows: u64 = 0,
    ya: u64 = 0,
    a8: u64 = 0,
    as: u64 = 0,
    wide: u64 = 0,
    w8: u64 = 0,
    wxs: u64 = 0,
    row_of: u64 = 0,
    all: u64 = 0,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    module: cuda.Module,
    functions: [kernel_names.len]cuda.Function,
    stream: cuda.Stream,
    hidden: usize,
    inner: usize,
    heads: usize,
    mlp: usize,
    blocks: []Block,
    geometry: Geometry = .{},
    profile: bool = false,
    spent: [stages.len]f64 = @splat(0),
    /// Pinned host memory that uploads pass through: the driver copies pageable memory several times slower.
    staging: cuda.HostBuffer = undefined,
    loaded_seconds: f64 = 0,
    /// The int8 projections came from the copy saved by an earlier load.
    from_cache: bool = false,

    fn function(self: *const Model, comptime name: []const u8) cuda.Function {
        inline for (kernel_names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return self.functions[i];
        @compileError("no kernel " ++ name);
    }

    fn launch(self: *const Model, comptime name: []const u8, grid: cuda.Dim3, threads: u32, shared: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(self.function(name), .{ .grid = grid, .block = .{ .x = threads }, .shared = shared }, self.stream, args);
    }

    /// `path`: FastH3's transformer in one safetensors file (ComfyUI's layout: blocks.N.attn.qkv_proj.weight ...).
    pub fn load(gpa: std.mem.Allocator, path: [*:0]const u8) !*Model {
        const started = seconds();
        const self = try gpa.create(Model);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.geometry = .{};
        self.profile = false;
        self.spent = @splat(0);
        self.driver = try cuda.Driver.open();
        self.ctx = try cuda.Context.init(&self.driver, 0);
        const d = &self.driver;
        self.module = try cuda.Module.load(d, &image);
        inline for (kernel_names, 0..) |name, i| self.functions[i] = try self.module.function(name);
        // the products stage two K slices of their row and column tiles
                try self.function("h3_gemm_wide").allowDynamicShared(2 * (256 + 128) * 128);
        try self.function("h3_gemm_n256").allowDynamicShared(2 * (128 + 256) * 128);
                try self.function("h3_attention").allowDynamicShared(attention_shared);
        // the legacy default stream: work queued by the caller's framework comes first
        self.stream = .{ .d = d, .handle = null };

        self.staging = try cuda.HostBuffer.alloc(d, staging_bytes);
        defer self.staging.free();
        var ck = try Checkpoint.open_(path);
        defer ck.deinit();
        var cache_path: [1024]u8 = undefined;
        const cache: [*:0]const u8 = @ptrCast((try std.fmt.bufPrint(&cache_path, "{s}.tf-int8\x00", .{std.mem.span(path)})).ptr);
        var name: [128]u8 = undefined;
        const qkv0 = try ck.bf16("blocks.0.attn.qkv_proj.weight");
        const fc2_0 = try ck.bf16("blocks.0.mlp.fc2.weight");
        self.hidden = qkv0.cols;
        self.inner = qkv0.rows / 3;
        self.heads = self.inner / head_dim;
        self.mlp = fc2_0.cols;
        var layers: usize = 0;
        while (std.mem.indexOf(u8, ck.header, try std.fmt.bufPrint(&name, "\"blocks.{d}.norm1.weight\"", .{layers})) != null) layers += 1;
        const hidden = self.hidden;
        const inner = self.inner;

        self.blocks = try gpa.alloc(Block, layers);
        for (self.blocks) |*b| {
            b.qkvc = try self.quantAlloc(4 * inner, hidden);
            b.out = try self.quantAlloc(hidden, inner);
            b.fc1 = try self.quantAlloc(2 * self.mlp, hidden);
            b.fc2 = try self.quantAlloc(hidden, self.mlp);
            b.norms = try cuda.DeviceBuffer.alloc(d, (2 * hidden + 2 * head_dim) * 2);
        }
        self.from_cache = self.readCache(cache, ck.bytes.len) catch false;
        if (!self.from_cache) {
            var scratch = try cuda.DeviceBuffer.alloc(d, 4 * inner * hidden * 2);
            defer scratch.free();
            for (self.blocks, 0..) |*b, i| {
                try self.quantize(&scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.qkv_proj.weight", .{i})), b.qkvc, 0, 0);
                try self.quantize(&scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.to_gate_compress.weight", .{i})), b.qkvc, 3 * inner, 0);
                try self.quantize(&scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.out_proj.weight", .{i})), b.out, 0, 0);
                // the SwiGLU's first projection is stored [gate; value]: its rows are laid out value, gate, value, gate
                try self.quantize(&scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.mlp.fc1.weight", .{i})), b.fc1, 0, self.mlp);
                try self.quantize(&scratch, try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.mlp.fc2.weight", .{i})), b.fc2, 0, 0);
                try self.upload(b.norms, 0, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.norm1.weight", .{i}))).bytes);
                try self.upload(b.norms, hidden * 2, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.norm2.weight", .{i}))).bytes);
                try self.upload(b.norms, hidden * 4, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.q_norm.weight", .{i}))).bytes);
                try self.upload(b.norms, hidden * 4 + head_dim * 2, (try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.attn.k_norm.weight", .{i}))).bytes);
            }
            self.writeCache(cache, ck.bytes.len) catch |e| std.debug.print("tensorfold h3: the int8 copy was not saved ({t}); the next start converts again\n", .{e});
        }
        try self.ctx.synchronize();
        self.loaded_seconds = seconds() - started;
        return self;
    }

    const staging_bytes = 64 << 20;

    /// `bytes` into `to` from `offset`, through the pinned buffer.
    fn upload(self: *Model, to: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(staging_bytes, bytes.len - done);
            @memcpy(self.staging.bytes[0..n], bytes[done..][0..n]);
            try to.upload(offset + done, self.staging.bytes[0..n]);
            done += n;
        }
    }

    /// What the int8 copy starts with: its format, the checkpoint's size and the model's sizes.
    const CacheHeader = extern struct { magic: [8]u8 = "tfh3i8\x00\x01".*, source_bytes: u64, hidden: u32, inner: u32, mlp: u32, layers: u32 };

    fn cacheBuffers(b: *Block) [9]*cuda.DeviceBuffer {
        return .{ &b.qkvc.w, &b.qkvc.s, &b.out.w, &b.out.s, &b.fc1.w, &b.fc1.s, &b.fc2.w, &b.fc2.s, &b.norms };
    }

    fn header(self: *const Model, source_bytes: usize) CacheHeader {
        return .{ .source_bytes = source_bytes, .hidden = @intCast(self.hidden), .inner = @intCast(self.inner), .mlp = @intCast(self.mlp), .layers = @intCast(self.blocks.len) };
    }

    /// The int8 projections from an earlier load's copy beside the checkpoint; false when there is none that fits.
    fn readCache(self: *Model, path: [*:0]const u8, source_bytes: usize) !bool {
        const fd = open(path, 0);
        if (fd < 0) return false;
        defer _ = close(fd);
        const want = self.header(source_bytes);
        var total: usize = @sizeOf(CacheHeader);
        for (self.blocks) |*b| for (cacheBuffers(b)) |held| {
            total += held.len;
        };
        if (lseek(fd, 0, 2) != @as(i64, @intCast(total))) return false;
        var found: CacheHeader = undefined;
        if (pread(fd, @ptrCast(&found), @sizeOf(CacheHeader), 0) != @sizeOf(CacheHeader)) return false;
        if (!std.mem.eql(u8, std.mem.asBytes(&found), std.mem.asBytes(&want))) return false;
        var at: usize = @sizeOf(CacheHeader);
        for (self.blocks) |*b| for (cacheBuffers(b)) |held| {
            var done: usize = 0;
            while (done < held.len) {
                const n = @min(staging_bytes, held.len - done);
                if (pread(fd, self.staging.bytes.ptr, n, @intCast(at + done)) != @as(isize, @intCast(n))) return error.CannotRead;
                try held.upload(done, self.staging.bytes[0..n]);
                done += n;
            }
            at += held.len;
        };
        return true;
    }

    /// Written under a temporary name and renamed, so a stopped write leaves no half file to read.
    fn writeCache(self: *Model, path: [*:0]const u8, source_bytes: usize) !void {
        var tmp_path: [1040]u8 = undefined;
        const tmp: [*:0]const u8 = @ptrCast((try std.fmt.bufPrint(&tmp_path, "{s}.part\x00", .{std.mem.span(path)})).ptr);
        const file = fopen(tmp, "wb") orelse return error.CannotWrite;
        var open_file = true;
        defer if (open_file) {
            _ = fclose(file);
        };
        const head = self.header(source_bytes);
        if (fwrite(&head, 1, @sizeOf(CacheHeader), file) != @sizeOf(CacheHeader)) return error.CannotWrite;
        const host = try self.gpa.alloc(u8, 4 * self.inner * self.hidden);
        defer self.gpa.free(host);
        for (self.blocks) |*b| for (cacheBuffers(b)) |held| {
            try held.download(0, host[0..held.len]);
            if (fwrite(host.ptr, 1, held.len, file) != held.len) return error.CannotWrite;
        };
        open_file = false;
        if (fclose(file) != 0) return error.CannotWrite;
        if (rename(tmp, path) != 0) return error.CannotWrite;
    }

    const attention_shared = slot_tile * head_dim + slot_tile * slot_tile + 2 * 2 * slot_tile * head_dim;

    fn quantAlloc(self: *Model, n: usize, k: usize) !Quant {
        return .{ .w = try cuda.DeviceBuffer.alloc(&self.driver, n * k), .s = try cuda.DeviceBuffer.alloc(&self.driver, n * 4) };
    }

    /// `w` stored (N, K) in bf16 to int8 rows of `q` from `row0`, or interleaved around `half`.
    fn quantize(self: *Model, scratch: *cuda.DeviceBuffer, w: Checkpoint.Entry, q: Quant, row0: usize, half: usize) !void {
        try self.upload(scratch.*, 0, w.bytes);
        var args: cuda.Args = .{};
        args.add(scratch.ptr);
        args.add(q.w.ptr);
        args.add(q.s.ptr);
        args.add(@as(c_int, @intCast(w.cols)));
        args.add(@as(c_int, @intCast(row0)));
        args.add(@as(c_int, @intCast(half)));
        try self.launch("h3_quant_weight", .{ .x = @intCast(w.rows) }, 128, 0, &args);
        try self.stream.synchronize();
    }

    pub fn deinit(self: *Model) void {
        self.release();
        for (self.blocks) |*b| {
            inline for (.{ "qkvc", "out", "fc1", "fc2" }) |f| {
                @field(b, f).w.free();
                @field(b, f).s.free();
            }
            b.norms.free();
        }
        self.gpa.free(self.blocks);
        self.module.unload();
        self.ctx.deinit();
        self.driver.close();
        self.gpa.destroy(self);
    }

    fn release(self: *Model) void {
        const g = &self.geometry;
        for (g.buffers[0..g.used]) |*b| b.free();
        g.* = .{};
    }

    fn buffer(self: *Model, bytes: usize, zero: bool) !u64 {
        const g = &self.geometry;
        g.buffers[g.used] = try cuda.DeviceBuffer.alloc(&self.driver, bytes);
        g.used += 1;
        if (zero) try g.buffers[g.used - 1].fill8(0, null);
        return g.buffers[g.used - 1].ptr;
    }

    fn padded(rows: usize) usize {
        return (rows + tile - 1) / tile * tile;
    }

    /// Buffers for `rows` packed rows in `tiles` routing tiles; `slot` is each row's padded slot on the device.
    fn prepare(self: *Model, rows: usize, tiles: usize, prefix: usize, keep: usize, slot: u64) !void {
        const g = &self.geometry;
        if (g.rows == rows and g.tiles == tiles and g.prefix == prefix and g.keep == keep) return;
        self.release();
        const hidden = self.hidden;
        const inner = self.inner;
        const heads = self.heads;
        const slots = tiles * slot_tile;
        const mp = padded(rows);
        g.y = try self.buffer(rows * hidden * 2, false);
        g.q8 = try self.buffer(mp * hidden, false);
        g.xs = try self.buffer(mp * 4, false);
        g.qkvc = try self.buffer(rows * 4 * inner * 2, false);
        // slots past a tile's real rows are never written: they stay zero
        g.tq = try self.buffer(heads * slots * head_dim, true);
        g.tqs = try self.buffer(heads * slots * 4, true);
        g.tk = try self.buffer(heads * slots * head_dim, true);
        g.tvt = try self.buffer(heads * slots * head_dim, true);
        g.ks = try self.buffer(heads * tiles * 4, false);
        g.vs = try self.buffer(heads * tiles * 4, false);
        g.qp = try self.buffer(heads * tiles * head_dim * 4, false);
        g.kp = try self.buffer(heads * tiles * head_dim * 4, false);
        g.vp = try self.buffer(heads * tiles * head_dim * 4, false);
        const groups = (tiles + slot_tile - 1) / slot_tile;
        const span = groups * slot_tile;
        g.scores = try self.buffer(heads * span * span * 4, false);
        g.chosen = try self.buffer(heads * @max(tiles - prefix, 1) * (prefix + keep) * 4, false);
        g.coarse = try self.buffer(tiles * inner * 2, false);
        g.pq = try self.buffer(heads * span * head_dim, false);
        g.pqs = try self.buffer(heads * span * 4, false);
        g.pk = try self.buffer(heads * span * head_dim, false);
        g.pks = try self.buffer(heads * groups * 4, false);
        g.pvt = try self.buffer(heads * span * head_dim, false);
        g.pvs = try self.buffer(heads * groups * 4, false);
        {
            // each group's real tiles, and the tile at each of its places (or -1)
            const sizes = try self.gpa.alloc(i32, groups);
            defer self.gpa.free(sizes);
            for (sizes, 0..) |*n, i| n.* = @intCast(@min(slot_tile, tiles - i * slot_tile));
            const places = try self.gpa.alloc(i32, span);
            defer self.gpa.free(places);
            for (places, 0..) |*t, i| t.* = if (i < tiles) @intCast(i) else -1;
            g.group_sizes = try self.buffer(groups * 4, false);
            try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(sizes));
            g.group_rows = try self.buffer(span * 4, false);
            try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(places));
        }
        g.ya = try self.buffer(rows * inner * 2, false);
        g.a8 = try self.buffer(mp * inner, false);
        g.as = try self.buffer(mp * 4, false);
        g.wide = try self.buffer(rows * self.mlp * 2, false);
        g.w8 = try self.buffer(mp * self.mlp, false);
        g.wxs = try self.buffer(mp * (self.mlp / wide_group) * 4, false);
        // the row at each slot (or -1), and the list of every tile
        const row_of = try self.gpa.alloc(i32, slots);
        defer self.gpa.free(row_of);
        @memset(row_of, -1);
        const slot_host = try self.gpa.alloc(i32, rows);
        defer self.gpa.free(slot_host);
        try self.driver.check(self.driver.api.cuMemcpyDtoH_v2(@ptrCast(slot_host.ptr), slot, rows * 4), "cuMemcpyDtoH");
        for (slot_host, 0..) |at, row| {
            if (at < 0 or @as(usize, @intCast(at)) >= slots) return error.BadGeometry;
            row_of[@intCast(at)] = @intCast(row);
        }
        const all = try self.gpa.alloc(i32, tiles);
        defer self.gpa.free(all);
        for (all, 0..) |*t, i| t.* = @intCast(i);
        g.row_of = try self.buffer(slots * 4, false);
        try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(row_of));
        g.all = try self.buffer(tiles * 4, false);
        try g.buffers[g.used - 1].upload(0, std.mem.sliceAsBytes(all));
        g.rows = rows;
        g.tiles = tiles;
        g.prefix = prefix;
        g.keep = keep;
    }

    fn lap(self: *Model, comptime stage: []const u8, mark: *f64) !void {
        if (!self.profile) return;
        try self.stream.synchronize();
        const now = seconds();
        inline for (stages, 0..) |s, i| if (comptime std.mem.eql(u8, s, stage)) {
            self.spent[i] += now - mark.*;
        };
        mark.* = now;
    }

    fn norm(self: *Model, x: u64, weight: u64, gates: u64, tab: u64, line: u64, gate: c_int, scale: c_int, shift: c_int) !void {
        const g = &self.geometry;
        var args: cuda.Args = .{};
        args.add(x);
        args.add(g.y);
        args.add(weight);
        args.add(gates);
        args.add(tab);
        args.add(line);
        args.add(g.q8);
        args.add(g.xs);
        args.add(@as(c_int, @intCast(g.rows)));
        args.add(@as(c_int, @intCast(self.hidden)));
        args.add(gate);
        args.add(scale);
        args.add(shift);
        args.add(eps);
        try self.launch("h3_norm_q8", .{ .x = @intCast(padded(g.rows)) }, 128, 0, &args);
    }

    /// d = x w for int8 rows and an int8 projection to `n` outputs from `k` inputs. A CTA owns `bm` rows by `bn`
    /// outputs with `threads` threads and stages two K slices of `kc` bytes.
    fn product(self: *Model, comptime kernel: []const u8, bm: usize, bn: usize, kc: usize, threads: u32, x: u64, scales: u64, w: Quant, d: u64, n: usize, k: usize, group: c_int) !void {
        var args: cuda.Args = .{};
        self.productArgs(&args, x, scales, w, d, n, k, group);
        try self.launch(kernel, .{ .x = @intCast((self.geometry.rows + bm - 1) / bm * (n / bn)) }, threads, @intCast(2 * (bm + bn) * kc), &args);
    }

    fn productArgs(self: *Model, args: *cuda.Args, x: u64, scales: u64, w: Quant, d: u64, n: usize, k: usize, group: c_int) void {
        args.add(x);
        args.add(scales);
        args.add(w.w.ptr);
        args.add(w.s.ptr);
        args.add(d);
        args.add(@as(c_int, @intCast(self.geometry.rows)));
        args.add(@as(c_int, @intCast(n)));
        args.add(@as(c_int, @intCast(k)));
        args.add(group);
    }

    /// softmax(q k) v over tiles of the rows' sequence: the prefix tiles and each video tile's chosen ones.
    fn attend(self: *Model, list: u64, sizes: u64, queries: usize, keys: usize, first: usize, per_query: bool) !void {
        const g = &self.geometry;
        var args: cuda.Args = .{};
        args.add(g.tq);
        args.add(g.tqs);
        args.add(g.tk);
        args.add(g.ks);
        args.add(g.tvt);
        args.add(g.vs);
        args.add(list);
        args.add(sizes);
        args.add(g.row_of);
        args.add(g.ya);
        args.add(@as(c_int, @intCast(g.tiles * slot_tile)));
        args.add(@as(c_int, @intCast(g.tiles)));
        args.add(@as(c_int, @intCast(self.heads)));
        args.add(@as(c_int, @intCast(queries)));
        args.add(@as(c_int, @intCast(keys)));
        args.add(@as(c_int, @intCast(first)));
        args.add(@as(c_int, @intFromBool(per_query)));
        args.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
        args.add(@as(u64, 0));
        try self.launch("h3_attention", .{ .x = @intCast(queries), .y = @intCast(self.heads) }, 128, attention_shared, &args);
    }

    /// The same kernel over the tiles' means, every tile against every tile: its output is the pooled branch and
    /// its scores are what the video tiles choose their key tiles by.
    fn attendPooled(self: *Model) !void {
        const g = &self.geometry;
        const groups = (g.tiles + slot_tile - 1) / slot_tile;
        var args: cuda.Args = .{};
        args.add(g.pq);
        args.add(g.pqs);
        args.add(g.pk);
        args.add(g.pks);
        args.add(g.pvt);
        args.add(g.pvs);
        args.add(g.all);
        args.add(g.group_sizes);
        args.add(g.group_rows);
        args.add(g.coarse);
        args.add(@as(c_int, @intCast(groups * slot_tile)));
        args.add(@as(c_int, @intCast(groups)));
        args.add(@as(c_int, @intCast(self.heads)));
        args.add(@as(c_int, @intCast(groups)));
        args.add(@as(c_int, @intCast(groups)));
        args.add(@as(c_int, 0));
        args.add(@as(c_int, 0));
        args.add(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))));
        args.add(g.scores);
        try self.launch("h3_attention", .{ .x = @intCast(groups), .y = @intCast(self.heads) }, 128, attention_shared, &args);
    }

    pub const Inputs = extern struct {
        /// The packed stream, (rows, hidden) bf16, updated in place.
        x: u64,
        /// Every block's modulation tables: (blocks, lines, 6, hidden) float.
        tables: u64,
        /// Each row's line in the tables: (rows) int32.
        line: u64,
        /// Rotary cosines and sines: (rows, rot) float each.
        cos: u64,
        sin: u64,
        /// Each row's padded slot in tile order: (rows) int32. Each tile's real rows: (tiles) int32.
        slot: u64,
        sizes: u64,
        rows: u32,
        lines: u32,
        rot: u32,
        tiles: u32,
        /// Tiles ahead of the video's; they attend to and are seen by every tile.
        prefix_tiles: u32,
        /// Video tiles a video tile keeps; 0 for dense attention (every tile sees every tile, no pooled branch).
        keep: u32,
        /// Blocks to run, from the first; 0 for all.
        blocks: u32 = 0,
        /// When not 0: (rows, inner) bf16 that receives the last run block's attention output before its gate mix.
        attention_out: u64 = 0,
    };

    /// Every block over the stream, in place; returns once the GPU has finished.
    pub fn forward(self: *Model, in: *const Inputs) !f64 {
        const started = seconds();
        try self.ctx.makeCurrent();
        const rows: usize = in.rows;
        const tiles: usize = in.tiles;
        const dense = in.keep == 0 or in.prefix_tiles >= in.tiles;
        const prefix: usize = if (dense) tiles else in.prefix_tiles;
        const keep: usize = if (dense) 0 else in.keep;
        try self.prepare(rows, tiles, prefix, keep, in.slot);
        const g = &self.geometry;
        const hidden = self.hidden;
        const inner = self.inner;
        const heads = self.heads;
        const table_bytes = @as(usize, in.lines) * 6 * hidden * 4;
        const video_tiles = tiles - prefix;
        const count = if (in.blocks == 0) self.blocks.len else @min(in.blocks, self.blocks.len);
        var mark = seconds();
        for (self.blocks[0..count], 0..) |*b, index| {
            const tab = in.tables + index * table_bytes;
            // the block before left its MLP's rows in y: its gated add runs with this block's first norm
            if (index == 0) try self.norm(in.x, b.norms.ptr, tab, tab, in.line, -1, scale_a, shift_a) else try self.norm(in.x, b.norms.ptr, tab - table_bytes, tab, in.line, gate_m, scale_a, shift_a);
            try self.lap("norms", &mark);
            try self.product("h3_gemm_n256", 128, 256, 128, 256, g.q8, g.xs, b.qkvc, g.qkvc, 4 * inner, hidden, 16);
            try self.lap("q k v gate", &mark);
            {
                // q, k and v to int8 in tile order, the tile's k and v scales found on the way
                var args: cuda.Args = .{};
                args.add(g.qkvc);
                args.add(b.norms.ptr + hidden * 4);
                args.add(b.norms.ptr + hidden * 4 + head_dim * 2);
                args.add(in.cos);
                args.add(in.sin);
                args.add(g.row_of);
                args.add(g.tq);
                args.add(g.tqs);
                args.add(g.tk);
                args.add(g.tvt);
                args.add(g.ks);
                args.add(g.vs);
                args.add(@as(c_int, @intCast(heads)));
                args.add(@as(c_int, @intCast(4 * inner)));
                args.add(@as(c_int, @intCast(in.rot)));
                args.add(@as(c_int, @intCast(tiles * slot_tile)));
                args.add(@as(c_int, @intCast(tiles)));
                args.add(eps);
                try self.launch("h3_heads", .{ .x = @intCast(tiles) }, 256, 0, &args);
            }
            try self.lap("heads", &mark);
            if (dense) {
                try self.attend(g.all, in.sizes, tiles, tiles, 0, false);
                try self.lap("attention", &mark);
            } else {
                var args: cuda.Args = .{};
                args.add(g.tq);
                args.add(g.tqs);
                args.add(g.tk);
                args.add(g.ks);
                args.add(g.tvt);
                args.add(g.vs);
                args.add(in.sizes);
                args.add(g.qp);
                args.add(g.kp);
                args.add(g.vp);
                args.add(@as(c_int, @intCast(tiles * slot_tile)));
                args.add(@as(c_int, @intCast(tiles)));
                try self.launch("h3_pool", .{ .x = @intCast(tiles), .y = @intCast(heads) }, 32, 0, &args);
                const groups = (tiles + slot_tile - 1) / slot_tile;
                args = .{};
                args.add(g.qp);
                args.add(g.kp);
                args.add(g.vp);
                args.add(g.pq);
                args.add(g.pqs);
                args.add(g.pk);
                args.add(g.pks);
                args.add(g.pvt);
                args.add(g.pvs);
                args.add(@as(c_int, @intCast(tiles)));
                args.add(@as(c_int, @intCast(groups)));
                try self.launch("h3_pool_quant", .{ .x = @intCast(groups), .y = @intCast(heads) }, 64, 0, &args);
                try self.attendPooled();
                args = .{};
                args.add(g.scores);
                args.add(g.chosen);
                args.add(@as(c_int, @intCast(tiles)));
                args.add(@as(c_int, @intCast(groups * slot_tile)));
                args.add(@as(c_int, @intCast(prefix)));
                args.add(@as(c_int, @intCast(keep)));
                try self.launch("h3_topk", .{ .x = @intCast(video_tiles), .y = @intCast(heads) }, 32, 0, &args);
                try self.lap("routing", &mark);
                // the prefix's query tiles see every tile; a video tile sees the prefix and its chosen tiles
                try self.attend(g.all, in.sizes, prefix, tiles, 0, false);
                try self.attend(g.chosen, in.sizes, video_tiles, prefix + keep, prefix, true);
                try self.lap("attention", &mark);
            }
            if (in.attention_out != 0 and index == count - 1) try self.driver.check(self.driver.api.cuMemcpyDtoD_v2(in.attention_out, g.ya, rows * inner * 2), "cuMemcpyDtoD");
            {
                // the pooled branch gated in and the rows rounded to int8, in one pass
                var args: cuda.Args = .{};
                args.add(g.ya);
                args.add(g.qkvc + 3 * inner * 2);
                args.add(if (dense) @as(u64, 0) else g.coarse);
                args.add(in.slot);
                args.add(g.a8);
                args.add(g.as);
                args.add(@as(c_int, @intCast(rows)));
                args.add(@as(c_int, @intCast(inner)));
                args.add(@as(c_int, @intCast(4 * inner)));
                try self.launch("h3_mix_quant", .{ .x = @intCast(padded(rows)) }, 128, 0, &args);
            }
            try self.lap("gate mix", &mark);
            try self.product("h3_gemm", 128, 128, 64, 128, g.a8, g.as, b.out, g.y, hidden, inner, 12);
            try self.lap("attention out", &mark);
            try self.norm(in.x, b.norms.ptr + hidden * 2, tab, tab, in.line, gate_a, scale_m, shift_m);
            try self.lap("norms", &mark);
            try self.product("h3_gemm_swiglu", 128, 128, 64, 128, g.q8, g.xs, b.fc1, g.wide, 2 * self.mlp, hidden, 16);
            try self.lap("mlp in", &mark);
            {
                var args: cuda.Args = .{};
                args.add(g.wide);
                args.add(g.w8);
                args.add(g.wxs);
                args.add(@as(c_int, @intCast(rows)));
                args.add(@as(c_int, @intCast(self.mlp)));
                try self.launch("h3_quant_wide", .{ .x = @intCast(padded(rows)) }, 128, 0, &args);
            }
            try self.lap("quantize", &mark);
            try self.product("h3_gemm_wide", 256, 128, 128, 256, g.w8, g.wxs, b.fc2, g.y, hidden, self.mlp, 4);
            try self.lap("mlp out", &mark);
        }
        {
            var args: cuda.Args = .{};
            args.add(in.x);
            args.add(g.y);
            args.add(in.tables + (count - 1) * table_bytes);
            args.add(in.line);
            args.add(@as(c_int, @intCast(hidden)));
            args.add(@as(c_int, gate_m));
            try self.launch("h3_gate_add", .{ .x = @intCast(rows) }, 128, 0, &args);
        }
        try self.stream.synchronize();
        return seconds() - started;
    }
};
