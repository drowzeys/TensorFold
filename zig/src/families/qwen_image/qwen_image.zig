//! Qwen-Image-2.1's transformer over our Metal runtime: the denoising forward for the image rows.
//! The prompt's keys and values (one pass through the stack per prompt) are given; see tools/zig/qwen_image_case.py.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const kernels = @import("qwen_image_kernels");

pub const heads = 32;
pub const head_dim = 128;
pub const hidden = heads * head_dim;
pub const mlp_width = 3 * hidden;
pub const channels = 64;
pub const layers = 32;
pub const time_dim = 256;
pub const eps: f32 = 1e-6;
const tile = 128;

pub fn toBf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u +% (0x7fff + ((u >> 16) & 1))) >> 16);
}

pub fn fromBf(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn round(x: f32) f32 {
    return fromBf(toBf(x));
}

/// A safetensors file mapped once and handed to Metal whole; a tensor is an offset into that one buffer.
pub const Tensors = struct {
    map: mtl.MappedFile,
    buffer: mtl.Buffer,
    data: usize,
    names: st.Header,
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, device: mtl.Device, path: [*:0]const u8) !Tensors {
        const map = try mtl.MappedFile.open(path);
        errdefer map.deinit();
        if (map.size < 8) return error.BadSafetensors;
        const header_len: usize = @intCast(std.mem.readInt(u64, map.bytes[0..8], .little));
        if (header_len > map.size - 8) return error.BadSafetensors;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const names = try st.parseHeader(arena.allocator(), map.bytes[8..][0..header_len], map.size - 8 - header_len);
        const buffer = try device.bufferNoCopy(map.bytes.ptr, map.bytes.len, mtl.ResourceOptions.shared);
        return .{ .map = map, .buffer = buffer, .data = 8 + header_len, .names = names, .arena = arena };
    }

    pub fn deinit(self: *Tensors) void {
        self.buffer.deinit();
        self.arena.deinit();
        self.map.deinit();
    }

    /// The tensor `name` of `dtype` and `count` values: where it starts in the buffer.
    pub fn at(self: *const Tensors, name: []const u8, dtype: st.DType, count: usize) !Ref {
        const e = self.names.get(name) orelse {
            std.debug.print("no tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        const offset = self.data + e.begin;
        if (e.dtype != dtype or e.end - e.begin != count * dtype.size()) {
            std.debug.print("tensor {s}: unexpected type or size\n", .{name});
            return error.BadTensor;
        }
        return .{ .buffer = self.buffer, .offset = offset };
    }

    /// `at` for a tensor a kernel reads in place, which must start on a multiple of its value size.
    pub fn inPlace(self: *const Tensors, name: []const u8, dtype: st.DType, count: usize) !Ref {
        const ref = try self.at(name, dtype, count);
        if (ref.offset % dtype.size() != 0) {
            std.debug.print("tensor {s} is not aligned for the GPU\n", .{name});
            return error.BadTensor;
        }
        return ref;
    }

    pub fn values(self: *const Tensors, comptime T: type, ref: Ref, count: usize) []const T {
        return @as([*]const T, @ptrCast(@alignCast(self.map.bytes.ptr + ref.offset)))[0..count];
    }

    /// The tensor's values copied out, wherever it starts in the file.
    pub fn copy(self: *const Tensors, comptime T: type, gpa: std.mem.Allocator, ref: Ref, count: usize) ![]T {
        const out = try gpa.alloc(T, count);
        @memcpy(std.mem.sliceAsBytes(out), self.map.bytes[ref.offset..][0 .. count * @sizeOf(T)]);
        return out;
    }
};

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };

const Block = struct { to_q: Ref, to_k: Ref, to_v: Ref, to_out: Ref, norm_q: Ref, norm_k: Ref, gate: Ref, proj: Ref, out: Ref, keys: mtl.Buffer, vals: mtl.Buffer };

pub const Timing = struct { wall: f64, gpu: f64 };

/// Where a profiled forward's GPU time went.
pub const stages = [_][]const u8{ "projections", "norms", "heads", "scores", "softmax", "values", "rows", "gates", "mlp projections", "swiglu" };

const Pass = struct { cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pipelines: [kernels.names.len]mtl.Pipeline,
    weights: Tensors,
    rows: usize,
    text: usize,
    /// Prompt and image rows, padded to whole tiles; the padding holds zero keys and values that take no weight.
    keys: usize,
    blocks: [layers]Block,
    img_in: Ref,
    proj_out: Ref,
    time_1: Ref,
    time_2: Ref,
    modulation: Ref,
    norm_out: Ref,
    cos: mtl.Buffer,
    sin: mtl.Buffer,
    // rows of bf16: the stream, a normed copy, projections, head-major tiles, scores, and the MLP's wide rows
    x: mtl.Buffer,
    n: mtl.Buffer,
    qr: mtl.Buffer,
    kr: mtl.Buffer,
    vr: mtl.Buffer,
    qh: mtl.Buffer,
    scores: mtl.Buffer,
    oh: mtl.Buffer,
    y: mtl.Buffer,
    wide_a: mtl.Buffer,
    wide_b: mtl.Buffer,
    wide_h: mtl.Buffer,
    latents: mtl.Buffer,
    velocity: mtl.Buffer,
    scale_1: mtl.Buffer,
    gate_1: mtl.Buffer,
    scale_2: mtl.Buffer,
    gate_2: mtl.Buffer,
    scale_out: mtl.Buffer,
    /// With `profile` set each stage runs in a command buffer of its own and its GPU seconds add up in `spent`.
    profile: bool = false,
    /// Whole-tile products where the shapes allow; off, every product takes the fragment kernel.
    tiles: bool = true,
    spent: [stages.len]f64 = @splat(0),

    /// `weights_path`: the transposed projections. `rows` image rows after `text` prompt rows.
    pub fn load(gpa: std.mem.Allocator, weights_path: [*:0]const u8, rows: usize, text: usize) !*Model {
        const device = try mtl.Device.init();
        if (!device.tensorUnits()) return error.NeedsTensorUnits;
        const self = try gpa.create(Model);
        self.profile = false;
        self.tiles = true;
        self.spent = @splat(0);
        self.gpa = gpa;
        self.device = device;
        self.queue = try device.queue();
        const lib = try mtl.Library.fromSource(device, kernels.source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        for (kernels.names, 0..) |name, i| self.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
        self.weights = try Tensors.open(gpa, device, weights_path);
        self.rows = rows;
        self.text = text;
        self.keys = (text + rows + tile - 1) / tile * tile;
        const w = &self.weights;
        const square = hidden * hidden;
        var name: [96]u8 = undefined;
        for (&self.blocks, 0..) |*b, i| {
            b.to_q = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_q.weight", .{i}), .bf16, square);
            b.to_k = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_k.weight", .{i}), .bf16, square);
            b.to_v = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_v.weight", .{i}), .bf16, square);
            b.to_out = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.to_out.0.weight", .{i}), .bf16, square);
            b.norm_q = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_q.weight", .{i}), .bf16, head_dim);
            b.norm_k = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.attn.norm_k.weight", .{i}), .bf16, head_dim);
            b.gate = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.img_mlp.gate_layer.weight", .{i}), .bf16, hidden * mlp_width);
            b.proj = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.img_mlp.proj.weight", .{i}), .bf16, hidden * mlp_width);
            b.out = try w.inPlace(try std.fmt.bufPrint(&name, "transformer_blocks.{d}.img_mlp.out.weight", .{i}), .bf16, hidden * mlp_width);
            b.keys = try self.rowsOf(heads * head_dim * self.keys);
            b.vals = try self.rowsOf(heads * head_dim * self.keys);
        }
        self.img_in = try w.inPlace("img_in.weight", .bf16, channels * hidden);
        self.proj_out = try w.inPlace("proj_out.weight", .bf16, channels * hidden);
        self.time_1 = try w.inPlace("time_text_embed.timestep_embedder.linear_1.weight", .bf16, time_dim * hidden);
        self.time_2 = try w.inPlace("time_text_embed.timestep_embedder.linear_2.weight", .bf16, square);
        self.modulation = try w.inPlace("modulation.1.weight", .bf16, 4 * square);
        self.norm_out = try w.inPlace("norm_out.linear.weight", .bf16, square);
        self.cos = try device.buffer(rows * head_dim / 2 * 4, mtl.ResourceOptions.shared);
        self.sin = try device.buffer(rows * head_dim / 2 * 4, mtl.ResourceOptions.shared);
        // a tile kernel reads its left operand's rows up to the next whole tile, so every such buffer has a spare one
        inline for (.{ "x", "n", "qr", "kr", "vr", "qh", "oh", "y" }) |f| @field(self, f) = try self.rowsOf((rows + tile) * hidden);
        inline for (.{ "wide_a", "wide_b", "wide_h" }) |f| @field(self, f) = try self.rowsOf((rows + tile) * mlp_width);
        self.scores = try self.rowsOf((heads * rows + tile) * self.keys);
        self.latents = try self.rowsOf((rows + tile) * channels);
        self.velocity = try self.rowsOf(rows * channels);
        inline for (.{ "scale_1", "gate_1", "scale_2", "gate_2", "scale_out" }) |f| @field(self, f) = try self.rowsOf(hidden);
        return self;
    }

    fn rowsOf(self: *Model, count: usize) !mtl.Buffer {
        return self.device.buffer(count * 2, mtl.ResourceOptions.shared);
    }

    /// The image rows' rotary tables, (rows, 64) each.
    pub fn setRotary(self: *Model, cos: []const f32, sin: []const f32) void {
        @memcpy(self.cos.slice(f32, cos.len), cos);
        @memcpy(self.sin.slice(f32, sin.len), sin);
    }

    /// Block `index`'s prompt keys (heads, 128, text) and values (heads, text, 128), copied ahead of the image rows'.
    pub fn setPrompt(self: *Model, index: usize, keys: []const u16, vals: []const u16) void {
        const total = self.keys;
        const k = self.blocks[index].keys.slice(u16, heads * head_dim * total);
        const v = self.blocks[index].vals.slice(u16, heads * head_dim * total);
        for (0..heads) |h| {
            for (0..head_dim) |c| @memcpy(k[(h * head_dim + c) * total ..][0..self.text], keys[(h * head_dim + c) * self.text ..][0..self.text]);
            @memcpy(v[h * total * head_dim ..][0 .. self.text * head_dim], vals[h * self.text * head_dim ..][0 .. self.text * head_dim]);
        }
    }

    fn lap(self: *Model, pass: *Pass, comptime stage: []const u8) void {
        if (!self.profile) return;
        pass.enc.end();
        pass.cb.commit();
        pass.cb.wait();
        inline for (stages, 0..) |name, i| if (comptime std.mem.eql(u8, name, stage)) {
            self.spent[i] += pass.cb.gpuSeconds();
        };
        pass.cb = self.queue.commandBuffer();
        pass.enc = pass.cb.compute(.serial);
    }

    fn pipeline(self: *const Model, comptime name: []const u8) mtl.Pipeline {
        inline for (kernels.names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return self.pipelines[i];
        @compileError("no kernel " ++ name);
    }

    /// d[z] = a[z] b[z] for `batch` z: a (m, k), b (k, n); `ld` holds the leading dimensions, then the strides between z.
    fn gemm(self: *const Model, enc: mtl.ComputeEncoder, a: Ref, b: Ref, d: Ref, m: usize, n: usize, k: usize, ld: [6]i64, batch: usize) void {
        enc.setPipeline(self.pipeline("qi_gemm"));
        enc.setBuffer(a.buffer, a.offset, 0);
        enc.setBuffer(b.buffer, b.offset, 1);
        enc.setValue([3]i32{ @intCast(m), @intCast(n), @intCast(k) }, 2);
        enc.setValue(ld, 3);
        enc.setBuffer(d.buffer, d.offset, 4);
        enc.dispatchGroups(mtl.Size.of((n + 127) / 128, (m + 63) / 64, batch), mtl.Size.of(32, 4, 1));
    }

    /// `gemm` on whole tiles: n and k multiples of 128.
    fn tileGemm(self: *const Model, enc: mtl.ComputeEncoder, a: Ref, b: Ref, d: Ref, m: usize, n: usize, k: usize, ld: [6]i64, batch: usize) void {
        enc.setPipeline(self.pipeline("qi_tile_gemm"));
        enc.setBuffer(a.buffer, a.offset, 0);
        enc.setBuffer(b.buffer, b.offset, 1);
        enc.setValue([3]i32{ @intCast(m), @intCast(n), @intCast(k) }, 2);
        enc.setValue(ld, 3);
        enc.setBuffer(d.buffer, d.offset, 4);
        enc.dispatchGroups(mtl.Size.of(n / tile, (m + tile - 1) / tile, batch), mtl.Size.of(256, 1, 1));
    }

    /// d = a w for rows of `k` values and a projection stored (k, n).
    fn linear(self: *const Model, enc: mtl.ComputeEncoder, a: mtl.Buffer, w: Ref, d: mtl.Buffer, k: usize, n: usize) void {
        const ld = [6]i64{ @intCast(k), @intCast(n), @intCast(n), 0, 0, 0 };
        if (self.tiles and k % tile == 0 and n % tile == 0) {
            self.tileGemm(enc, .{ .buffer = a }, w, .{ .buffer = d }, self.rows, n, k, ld, 1);
        } else {
            self.gemm(enc, .{ .buffer = a }, w, .{ .buffer = d }, self.rows, n, k, ld, 1);
        }
    }

    fn normScale(self: *const Model, enc: mtl.ComputeEncoder, scale: mtl.Buffer) void {
        enc.setPipeline(self.pipeline("qi_norm_scale"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(scale, 0, 1);
        enc.setValue([1]i32{hidden}, 2);
        enc.setValue([1]f32{eps}, 3);
        enc.setBuffer(self.n, 0, 4);
        enc.dispatchGroups(mtl.Size.of(self.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }

    fn gateAdd(self: *const Model, enc: mtl.ComputeEncoder, gate: mtl.Buffer) void {
        enc.setPipeline(self.pipeline("qi_gate_add"));
        enc.setBuffer(self.x, 0, 0);
        enc.setBuffer(self.y, 0, 1);
        enc.setBuffer(gate, 0, 2);
        enc.setValue([2]i32{ @intCast(self.rows), hidden }, 3);
        enc.dispatchThreads(mtl.Size.of(hidden, self.rows, 1), mtl.Size.of(64, 16, 1));
    }

    /// y[n] = sum over k of x[k] w[k, n], accumulated in fp32 and rounded to bf16 as the projections' outputs are.
    fn project(self: *const Model, w: Ref, x: []const f32, y: []f32) void {
        const table = self.weights.values(u16, w, x.len * y.len);
        @memset(y, 0);
        for (x, 0..) |xv, k| {
            const row = table[k * y.len ..][0..y.len];
            for (y, row) |*o, v| o.* += xv * fromBf(v);
        }
        for (y) |*o| o.* = round(o.*);
    }

    fn silu(values: []f32) void {
        for (values) |*v| v.* = round(v.* * round(1.0 / (1.0 + @exp(-v.*))));
    }

    /// The modulation every block shares at noise level `sigma`, and the output norm's scale.
    fn modulate(self: *Model, sigma: f32) !void {
        const gpa = self.gpa;
        const emb = try gpa.alloc(f32, time_dim);
        defer gpa.free(emb);
        const a = try gpa.alloc(f32, hidden);
        defer gpa.free(a);
        const temb = try gpa.alloc(f32, hidden);
        defer gpa.free(temb);
        const table = try gpa.alloc(f32, 4 * hidden);
        defer gpa.free(table);
        const half = time_dim / 2;
        for (0..half) |i| {
            const freq = @exp(-@log(@as(f32, 10000.0)) * @as(f32, @floatFromInt(i)) / @as(f32, half));
            const angle = sigma * 1000.0 * freq;
            emb[i] = round(@cos(angle));
            emb[half + i] = round(@sin(angle));
        }
        self.project(self.time_1, emb, a);
        silu(a);
        self.project(self.time_2, a, temb);
        silu(temb);
        self.project(self.modulation, temb, table);
        const outs = .{ self.scale_1, self.gate_1, self.scale_2, self.gate_2 };
        inline for (outs, 0..) |buffer, part| {
            const to = buffer.slice(u16, hidden);
            for (to, table[part * hidden ..][0..hidden]) |*o, v| o.* = toBf(if (part % 2 == 0) 1.0 + v else std.math.tanh(v));
        }
        self.project(self.norm_out, temb, a);
        for (self.scale_out.slice(u16, hidden), a) |*o, v| o.* = toBf(1.0 + v);
    }

    /// The velocity (rows, 64) for `latents` (rows, 64) at noise level `sigma`.
    pub fn forward(self: *Model, latents: []const f32, sigma: f32, velocity: []f32) !Timing {
        const started = mtl.clock.seconds();
        try self.modulate(sigma);
        for (self.latents.slice(u16, latents.len), latents) |*o, v| o.* = toBf(v);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const rows = self.rows;
        const total = self.keys;
        var pass: Pass = undefined;
        pass.cb = self.queue.commandBuffer();
        pass.enc = pass.cb.compute(.serial);
        var gpu: f64 = 0;
        self.linear(pass.enc, self.latents, self.img_in, self.x, channels, hidden);
        for (&self.blocks) |*b| {
            self.normScale(pass.enc, self.scale_1);
            self.lap(&pass, "norms");
            self.linear(pass.enc, self.n, b.to_q, self.qr, hidden, hidden);
            self.linear(pass.enc, self.n, b.to_k, self.kr, hidden, hidden);
            self.linear(pass.enc, self.n, b.to_v, self.vr, hidden, hidden);
            self.lap(&pass, "projections");
            pass.enc.setPipeline(self.pipeline("qi_heads"));
            pass.enc.setBuffer(self.qr, 0, 0);
            pass.enc.setBuffer(self.kr, 0, 1);
            pass.enc.setBuffer(self.vr, 0, 2);
            pass.enc.setBuffer(b.norm_q.buffer, b.norm_q.offset, 3);
            pass.enc.setBuffer(b.norm_k.buffer, b.norm_k.offset, 4);
            pass.enc.setBuffer(self.cos, 0, 5);
            pass.enc.setBuffer(self.sin, 0, 6);
            pass.enc.setValue([4]i32{ @intCast(rows), heads, @intCast(self.text), @intCast(total) }, 7);
            pass.enc.setValue([1]f32{eps}, 8);
            pass.enc.setBuffer(self.qh, 0, 9);
            pass.enc.setBuffer(b.keys, 0, 10);
            pass.enc.setBuffer(b.vals, 0, 11);
            pass.enc.dispatchThreads(mtl.Size.of(rows, heads, 1), mtl.Size.of(32, 32, 1));
            self.lap(&pass, "heads");
            // scores = q k^T a head at a time, then probabilities in place, then probabilities times values
            const score_ld = [6]i64{ head_dim, @intCast(total), @intCast(total), @intCast(rows * head_dim), @intCast(head_dim * total), @intCast(rows * total) };
            if (self.tiles) self.tileGemm(pass.enc, .{ .buffer = self.qh }, .{ .buffer = b.keys }, .{ .buffer = self.scores }, rows, total, head_dim, score_ld, heads) else self.gemm(pass.enc, .{ .buffer = self.qh }, .{ .buffer = b.keys }, .{ .buffer = self.scores }, rows, total, head_dim, score_ld, heads);
            self.lap(&pass, "scores");
            pass.enc.setPipeline(self.pipeline("qi_softmax"));
            pass.enc.setBuffer(self.scores, 0, 0);
            pass.enc.setValue([2]i32{ @intCast(self.text + rows), @intCast(total) }, 1);
            pass.enc.setValue([1]f32{1.0 / @sqrt(@as(f32, head_dim))}, 2);
            pass.enc.dispatchGroups(mtl.Size.of(heads * rows, 1, 1), mtl.Size.of(32, 1, 1));
            self.lap(&pass, "softmax");
            const value_ld = [6]i64{ @intCast(total), head_dim, head_dim, @intCast(rows * total), @intCast(total * head_dim), @intCast(rows * head_dim) };
            if (self.tiles) self.tileGemm(pass.enc, .{ .buffer = self.scores }, .{ .buffer = b.vals }, .{ .buffer = self.oh }, rows, head_dim, total, value_ld, heads) else self.gemm(pass.enc, .{ .buffer = self.scores }, .{ .buffer = b.vals }, .{ .buffer = self.oh }, rows, head_dim, total, value_ld, heads);
            self.lap(&pass, "values");
            pass.enc.setPipeline(self.pipeline("qi_rows"));
            pass.enc.setBuffer(self.oh, 0, 0);
            pass.enc.setValue([2]i32{ @intCast(rows), heads }, 1);
            pass.enc.setBuffer(self.qr, 0, 2);
            pass.enc.dispatchThreads(mtl.Size.of(rows, heads, 1), mtl.Size.of(32, 32, 1));
            self.lap(&pass, "rows");
            self.linear(pass.enc, self.qr, b.to_out, self.y, hidden, hidden);
            self.lap(&pass, "projections");
            self.gateAdd(pass.enc, self.gate_1);
            self.lap(&pass, "gates");
            self.normScale(pass.enc, self.scale_2);
            self.lap(&pass, "norms");
            self.linear(pass.enc, self.n, b.gate, self.wide_a, hidden, mlp_width);
            self.linear(pass.enc, self.n, b.proj, self.wide_b, hidden, mlp_width);
            self.lap(&pass, "mlp projections");
            pass.enc.setPipeline(self.pipeline("qi_swiglu"));
            pass.enc.setBuffer(self.wide_a, 0, 0);
            pass.enc.setBuffer(self.wide_b, 0, 1);
            pass.enc.setValue([1]i64{@intCast(rows * mlp_width)}, 2);
            pass.enc.setBuffer(self.wide_h, 0, 3);
            pass.enc.dispatchThreads(mtl.Size.of(1024, (rows * mlp_width + 1023) / 1024, 1), mtl.Size.of(1024, 1, 1));
            self.lap(&pass, "swiglu");
            self.linear(pass.enc, self.wide_h, b.out, self.y, mlp_width, hidden);
            self.lap(&pass, "mlp projections");
            self.gateAdd(pass.enc, self.gate_2);
            self.lap(&pass, "gates");
        }
        self.normScale(pass.enc, self.scale_out);
        self.linear(pass.enc, self.n, self.proj_out, self.velocity, hidden, channels);
        pass.enc.end();
        pass.cb.commit();
        pass.cb.wait();
        gpu += pass.cb.gpuSeconds();
        if (pass.cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        for (velocity, self.velocity.slice(u16, velocity.len)) |*o, v| o.* = fromBf(v);
        return .{ .wall = mtl.clock.seconds() - started, .gpu = gpu };
    }
};

test "bf16 rounds to nearest even" {
    try std.testing.expectEqual(@as(f32, 1.0), fromBf(toBf(1.0)));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf(@bitCast(@as(u32, 0x3f818000))));
}
