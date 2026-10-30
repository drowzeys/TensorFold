//! Transcribe's perception tensors as stored, in one device allocation; the few the host reads are kept there too.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig");
const Allocator = std.mem.Allocator;
const st = core.safetensors;

const asr = "perception.encoder.asr_encoder.";
const sf = "perception.encoder.diarization_model.";
const heads_ = sf ++ "sortformer_modules.";

pub const Layer = struct {
    norm1: u64,
    norm1_bias: u64,
    qkv: u64, // [3 d, d], no bias: q rows, then k, then v, each head by head
    q_norm: u64 = 0, // [head dim] with its bias, when the stack has qk_norm
    q_norm_bias: u64 = 0,
    k_norm: u64 = 0,
    k_norm_bias: u64 = 0,
    out: u64,
    out_bias: u64,
    norm2: u64,
    norm2_bias: u64,
    fc1: u64, // [4 d, d]
    fc1_bias: u64,
    fc2: u64, // [d, 4 d]
    fc2_bias: u64,
    fp8: ?Fp8 = null, // a quantized release's FP8 projections (qkv, out, fc1, fc2 then read 0 above)
};

/// A ModelOpt FP8 release's four projections of a layer: e4m3 codes in the lane matmul's order, one scale each.
pub const Fp8 = struct { qkv: cuda.qmmf.Weight, out: cuda.qmmf.Weight, fc1: cuda.qmmf.Weight, fc2: cuda.qmmf.Weight };

/// One TransformerEncoder: the stacking projection, the embed norm, its layers and the final norm.
pub const Stack = struct {
    pre: u64, // [d, 8 * 128], no bias
    embed_norm: u64,
    embed_norm_bias: u64,
    final: u64,
    final_bias: u64,
    layers: []Layer,
};

pub const Weights = struct {
    memory: cuda.DeviceBuffer,
    asr: Stack,
    sf: Stack,
    encoder_proj: u64, // [192, 512]
    encoder_proj_bias: u64,
    upsample: u64, // the subpixel conv [1536, 3 * 192]: tap-major (stored [1536, 192, 3])
    upsample_bias: u64,
    hidden: u64, // first_hidden_to_hidden [192, 192]
    hidden_bias: u64,
    spks: u64, // single_hidden_to_spks [8, 192]
    spks_bias: u64,
    asr_norm: u64,
    asr_norm_bias: u64,
    proj: u64, // [text hidden, 1280]
    proj_bias: u64,
    host: Host,
    bytes: u64,
    fp8: bool, // the ASR encoder's projections are FP8 (the NVFP4 and FP8 releases)

    pub fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: Config) !Weights {
        const paths = try core.checkpoint.shardFiles(gpa, io, dir);
        defer core.checkpoint.freeShardFiles(gpa, paths);
        var files: std.ArrayList(st.File) = .empty;
        defer {
            for (files.items) |*f| {
                f.release();
                f.close(io);
            }
            files.deinit(gpa);
        }
        for (paths) |path| {
            var f = try st.File.openPrefix(gpa, io, path, "perception.");
            if (f.names.count() == 0) {
                f.close(io);
                continue;
            }
            files.append(gpa, f) catch |err| {
                f.close(io);
                return err;
            };
        }
        if (files.items.len == 0) return error.NoAudioEncoder;
        var p: Planner = .{ .gpa = gpa, .files = files.items };
        defer {
            for (p.pieces.items) |piece| if (piece.owned) gpa.free(piece.bytes);
            p.pieces.deinit(gpa);
        }
        var w: Weights = undefined;
        w.asr.layers = try gpa.alloc(Layer, c.asr.layers);
        errdefer gpa.free(w.asr.layers);
        w.sf.layers = try gpa.alloc(Layer, c.sf.layers);
        errdefer gpa.free(w.sf.layers);
        w.host = try Host.read(gpa, &p, c);
        errdefer w.host.deinit(gpa);
        try p.stack(&w.asr, asr, c.asr, true);
        w.fp8 = w.asr.layers.len > 0 and w.asr.layers[0].fp8 != null;
        try p.stack(&w.sf, sf ++ "encoder.", c.sf, false);
        const t = Config.tf_d;
        w.encoder_proj = try p.take(heads_ ++ "encoder_proj.weight", &.{ t, c.sf.d });
        w.encoder_proj_bias = try p.take(heads_ ++ "encoder_proj.bias", &.{t});
        w.upsample = try p.taps(heads_ ++ "subpixel_upsample.weight", Config.sub * t, t);
        w.upsample_bias = try p.take(heads_ ++ "subpixel_upsample.bias", &.{Config.sub * t});
        w.hidden = try p.take(heads_ ++ "first_hidden_to_hidden.weight", &.{ t, t });
        w.hidden_bias = try p.take(heads_ ++ "first_hidden_to_hidden.bias", &.{t});
        w.spks = try p.take(heads_ ++ "single_hidden_to_spks.weight", &.{ Config.speakers, t });
        w.spks_bias = try p.take(heads_ ++ "single_hidden_to_spks.bias", &.{Config.speakers});
        w.asr_norm = try p.take("perception.encoder.asr_norm.weight", &.{c.asr.d});
        w.asr_norm_bias = try p.take("perception.encoder.asr_norm.bias", &.{c.asr.d});
        w.proj = try p.take("perception.proj.weight", &.{ c.text_hidden, c.asr.d });
        w.proj_bias = try p.take("perception.proj.bias", &.{c.text_hidden});
        var memory = try cuda.DeviceBuffer.alloc(d, p.bytes);
        errdefer memory.free();
        for (p.pieces.items) |piece| try memory.upload(piece.dst, piece.bytes);
        w.memory = memory;
        w.bytes = p.bytes;
        shift(&w, memory.ptr);
        inline for (.{ &w.asr, &w.sf }) |s| {
            shift(s, memory.ptr);
            for (s.layers) |*l| {
                shift(l, memory.ptr);
                if (l.fp8) |*q| inline for (.{ &q.qkv, &q.out, &q.fc1, &q.fc2 }) |x| {
                    x.codes += memory.ptr;
                };
            }
        }
        return w;
    }

    pub fn deinit(w: *Weights, gpa: Allocator) void {
        w.memory.free();
        gpa.free(w.asr.layers);
        gpa.free(w.sf.layers);
        w.host.deinit(gpa);
        w.* = undefined;
    }
};

/// The tensors the host side reads, widened to f32 where it computes with them.
pub const Host = struct {
    window: []f32, // [400], bf16 values
    fb: []f32, // [128, 257], bf16 values
    sil: []u16, // learnable_sil_emb [512] bf16 bits
    diar_norm: [Config.speakers]f32,
    diar_norm_bias: [Config.speakers]f32,

    fn read(gpa: Allocator, p: *Planner, c: Config) !Host {
        const window = try widen(gpa, (try p.get("perception.preprocessor.featurizer.window", &.{Config.win})).bytes);
        errdefer gpa.free(window);
        const fb = try widen(gpa, (try p.get("perception.preprocessor.featurizer.fb", &.{ 1, Config.mels, Config.bins })).bytes);
        errdefer gpa.free(fb);
        const sil = try gpa.alloc(u16, c.sf.d);
        errdefer gpa.free(sil);
        for (sil, std.mem.bytesAsSlice(u16, (try p.get(heads_ ++ "learnable_sil_emb", &.{c.sf.d})).bytes)) |*o, v| o.* = v;
        var h: Host = .{ .window = window, .fb = fb, .sil = sil, .diar_norm = undefined, .diar_norm_bias = undefined };
        for ([2][]const u8{ "perception.encoder.diar_norm.weight", "perception.encoder.diar_norm.bias" }, [2]*[Config.speakers]f32{ &h.diar_norm, &h.diar_norm_bias }) |name, out| {
            const t = try p.get(name, &.{Config.speakers});
            for (out, std.mem.bytesAsSlice(u16, t.bytes)) |*o, v| o.* = bf(v);
        }
        return h;
    }

    fn deinit(h: *Host, gpa: Allocator) void {
        gpa.free(h.window);
        gpa.free(h.fb);
        gpa.free(h.sil);
    }
};

pub fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

fn widen(gpa: Allocator, bytes: []const u8) ![]f32 {
    const out = try gpa.alloc(f32, bytes.len / 2);
    for (out, std.mem.bytesAsSlice(u16, bytes)) |*o, v| o.* = bf(v);
    return out;
}

/// One tensor's bytes and where they go in the allocation; `owned` bytes were rearranged on the host.
const Piece = struct { bytes: []const u8, dst: u64, owned: bool = false };

const Planner = struct {
    gpa: Allocator,
    files: []const st.File,
    pieces: std.ArrayList(Piece) = .empty,
    bytes: u64 = 256, // nothing at offset 0: an offset of 0 is a tensor the stack does not have

    fn find(p: *Planner, name: []const u8) ?st.Tensor {
        for (p.files) |*f| if (f.get(name)) |t| return t;
        return null;
    }

    /// `{pre}.weight` [n, k] when the release stores it as FP8 (e4m3 codes, an fp32 weight_scale), packed for the lane
    /// matmul (fp8.packCodes on the host, as modelopt_pack.cu's fp8 lane packs it); null when it is bf16.
    fn fp8(p: *Planner, pre: []const u8, n: usize, k: usize) !?cuda.qmmf.Weight {
        var buf: [192]u8 = undefined;
        const t = p.find(try join(&buf, pre, ".weight")) orelse return null;
        if (t.dtype != .f8_e4m3) return null;
        if (!t.is(.f8_e4m3, &.{ n, k }) or k % cuda.qmmf.group != 0) {
            std.log.err("{s}.weight: {t} {any}, expected F8_E4M3 [{d}, {d}]", .{ pre, t.dtype, t.shape[0..t.rank], n, k });
            return error.UnexpectedTensor;
        }
        const sc = p.find(try join(&buf, pre, ".weight_scale")) orelse return error.MissingTensor;
        if (sc.dtype != .f32 or sc.bytes.len != 4) return error.UnexpectedTensor;
        const scale: f32 = @bitCast(std.mem.readInt(u32, sc.bytes[0..4], .little));
        const npad = cuda.modelopt.padded(n);
        const bytes = try p.gpa.alloc(u8, npad * k);
        errdefer p.gpa.free(bytes);
        cuda.fp8.packCodes(bytes, t.bytes, n, k);
        const at = try p.place(bytes, true);
        return cuda.modelopt.fp8Weight(at, scale, n, k);
    }

    fn get(p: *Planner, name: []const u8, shape: []const usize) !st.Tensor {
        for (p.files) |*f| if (f.get(name)) |t| {
            if (t.is(.bf16, shape)) return t;
            std.log.err("{s}: {t} {any}, expected bf16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        };
        std.log.err("checkpoint has no tensor {s}", .{name});
        return error.MissingTensor;
    }

    fn place(p: *Planner, bytes: []const u8, owned: bool) !u64 {
        const at = std.mem.alignForward(u64, p.bytes, 256);
        try p.pieces.append(p.gpa, .{ .bytes = bytes, .dst = at, .owned = owned });
        p.bytes = at + bytes.len;
        return at;
    }

    /// `name` placed as stored; its offset from the allocation's start.
    fn take(p: *Planner, name: []const u8, shape: []const usize) !u64 {
        return p.place((try p.get(name, shape)).bytes, false);
    }

    /// A conv1d weight [o, c, 3] as a GEMM's [o, 3 c]: tap j of channel c at column j c + c.
    fn taps(p: *Planner, name: []const u8, o: usize, c: usize) !u64 {
        const src = std.mem.bytesAsSlice(u16, (try p.get(name, &.{ o, c, 3 })).bytes);
        const bytes = try p.gpa.alloc(u8, o * 3 * c * 2); // bytes, as the pieces are freed
        errdefer p.gpa.free(bytes);
        const out = std.mem.bytesAsSlice(u16, bytes);
        for (0..o) |r| for (0..c) |ch| for (0..3) |j| {
            out[r * 3 * c + j * c + ch] = src[(r * c + ch) * 3 + j];
        };
        return p.place(bytes, true);
    }

    fn stack(p: *Planner, s: *Stack, pre: []const u8, c: Config.Encoder, qk_norm: bool) !void {
        var buf: [192]u8 = undefined;
        const d = c.d;
        const dh = d / c.heads;
        s.pre = try p.take(try join(&buf, pre, "pre_encode.proj.weight"), &.{ d, Config.sub * Config.mels });
        s.embed_norm = try p.take(try join(&buf, pre, "embed_norm.weight"), &.{d});
        s.embed_norm_bias = try p.take(try join(&buf, pre, "embed_norm.bias"), &.{d});
        s.final = try p.take(try join(&buf, pre, "final_norm.weight"), &.{d});
        s.final_bias = try p.take(try join(&buf, pre, "final_norm.bias"), &.{d});
        for (s.layers, 0..) |*l, i| {
            const n = struct {
                fn of(b: []u8, prefix: []const u8, li: usize, part: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, "{s}layers.{d}.{s}", .{ prefix, li, part });
                }
            }.of;
            var q: [4]?cuda.qmmf.Weight = undefined;
            for ([_][]const u8{ "attn.w_qkv", "attn.out_proj", "ffn.net.0", "ffn.net.3" }, [_][2]usize{ .{ 3 * d, d }, .{ d, d }, .{ 4 * d, d }, .{ d, 4 * d } }, &q) |part, nk, *x| {
                x.* = try p.fp8(try n(&buf, pre, i, part), nk[0], nk[1]);
            }
            const quantized = q[0] != null;
            for (q) |x| if ((x != null) != quantized) return error.MixedQuantization;
            l.* = .{
                .norm1 = try p.take(try n(&buf, pre, i, "norm1.weight"), &.{d}),
                .norm1_bias = try p.take(try n(&buf, pre, i, "norm1.bias"), &.{d}),
                .qkv = if (quantized) 0 else try p.take(try n(&buf, pre, i, "attn.w_qkv.weight"), &.{ 3 * d, d }),
                .out = if (quantized) 0 else try p.take(try n(&buf, pre, i, "attn.out_proj.weight"), &.{ d, d }),
                .out_bias = try p.take(try n(&buf, pre, i, "attn.out_proj.bias"), &.{d}),
                .norm2 = try p.take(try n(&buf, pre, i, "norm2.weight"), &.{d}),
                .norm2_bias = try p.take(try n(&buf, pre, i, "norm2.bias"), &.{d}),
                .fc1 = if (quantized) 0 else try p.take(try n(&buf, pre, i, "ffn.net.0.weight"), &.{ 4 * d, d }),
                .fc1_bias = try p.take(try n(&buf, pre, i, "ffn.net.0.bias"), &.{4 * d}),
                .fc2 = if (quantized) 0 else try p.take(try n(&buf, pre, i, "ffn.net.3.weight"), &.{ d, 4 * d }),
                .fc2_bias = try p.take(try n(&buf, pre, i, "ffn.net.3.bias"), &.{d}),
            };
            if (quantized) l.fp8 = .{ .qkv = q[0].?, .out = q[1].?, .fc1 = q[2].?, .fc2 = q[3].? };
            if (qk_norm) {
                l.q_norm = try p.take(try n(&buf, pre, i, "attn.q_norm.weight"), &.{dh});
                l.q_norm_bias = try p.take(try n(&buf, pre, i, "attn.q_norm.bias"), &.{dh});
                l.k_norm = try p.take(try n(&buf, pre, i, "attn.k_norm.weight"), &.{dh});
                l.k_norm_bias = try p.take(try n(&buf, pre, i, "attn.k_norm.bias"), &.{dh});
            }
        }
    }
};

fn join(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// Every placed u64 offset of `s` (not its byte count, nor a 0 that marks a tensor it lacks) moved by `base`.
fn shift(s: anytype, base: u64) void {
    const info = @typeInfo(@TypeOf(s.*)).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64 and comptime !std.mem.eql(u8, name, "bytes")) {
        if (@field(s, name) != 0) @field(s, name) += base;
    };
}
