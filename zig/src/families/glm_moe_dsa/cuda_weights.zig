//! GLM-5.3 layer slices on the GPU in the Python engine's layouts (cuda/weights.py load_layer + fused.Weights):
//! EXL3 linears in the strips layout, kv_b cut into the absorb (wk) and expand (wv) blocks, the indexer's bf16 parts,
//! the router, and the routed experts as experts.prepare tables (TF_GLM53_EXPERTS=tf: one trellis a matrix).
//! Reads go through mapped safetensors (core.Checkpoint), each rank's part cut as split.rule says (world 1: whole).
//! TODO(P2): O_DIRECT + io_uring streaming (nemotron/cuda_source.zig) instead of mapped reads; the cuda-exl3 shared
//! stacks (experts_cx) for prompt chunks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const exl3 = @import("exl3.zig");
const split = @import("split.zig");
const kern = @import("cuda_kernels.zig");

const Tensor = core.checkpoint.Tensor;
pub const Experts = kern.Experts;

pub const Mlp = struct { gate: exl3.Linear, up: exl3.Linear, down: exl3.Linear };

pub const Indexer = struct {
    wq_b: exl3.Linear,
    wk: u64, // bf16 [index_dim, D]
    weights_proj: u64, // bf16 [index_heads, D]
    k_norm_w: u64, // bf16 [index_dim]
    k_norm_b: u64,
};

pub const Layer = struct {
    index: usize,
    input_norm: u64,
    post_attn_norm: u64,
    q_a: exl3.Linear,
    q_a_norm: u64,
    kv_a: exl3.Linear, // its stored 640 outputs (the cache keeps 576)
    kv_a_norm: u64,
    q_b: exl3.Linear,
    wk: u64, // bf16 [H, nope, kv_lora]: kv_b's key block (absorb)
    wv: u64, // bf16 [H, v_dim, kv_lora]: kv_b's value block (expand)
    o_proj: exl3.Linear,
    indexer: ?Indexer,
    router_w: u64 = 0, // bf16 [E, D]
    router_bias: u64 = 0, // fp32 [E]
    experts: ?Experts = null,
    mlp: Mlp, // the dense MLP (layers < first_k_dense_replace) or the shared expert
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    rank: usize,
    world: usize,
    heads: usize, // this rank's attention heads
    embed: u64 = 0, // bf16 [V, D]
    layers: []Layer = &.{},
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    digest_checked: usize = 0,
    digest_bad: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    /// The widest EXL3 linear output the scratch must hold (rows x N), and the largest K, over every layer.
    pub fn linearExtents(w: *const Weights) struct { k: usize, zn: usize } {
        var k: usize = 0;
        var zn: usize = 0;
        for (w.layers) |*L| {
            const lins = [_]*const exl3.Linear{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down };
            for (lins) |l| {
                k = @max(k, l.k);
                zn = @max(zn, l.sk * l.n);
            }
            if (L.indexer) |*ix| {
                k = @max(k, ix.wq_b.k);
                zn = @max(zn, ix.wq_b.sk * ix.wq_b.n);
            }
        }
        return .{ .k = k, .zn = zn };
    }
};

pub const Options = struct {
    rank: usize = 0,
    world: usize = 1,
    layers: usize, // load layers 0..layers-1
    embed: bool = true,
    /// weights_sha256 of the oracle's meta.json: each prepared tensor's bytes are checked against it when given
    digests: ?*const std.json.ObjectMap = null,
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ck: *core.Checkpoint,
    w: *Weights,
    o: Options,
    host: std.ArrayList(u8) = .empty,
    host2: std.ArrayList(u8) = .empty,

    fn deinit(L: *Loader) void {
        L.host.deinit(L.gpa);
        L.host2.deinit(L.gpa);
    }

    fn has(L: *Loader, name: []const u8) bool {
        for (L.ck.files.items) |*f| if (f.names.get(name) != null) return true;
        return false;
    }

    /// The oracle's digest of a prepared tensor, compared with these bytes (a mismatch is logged and counted).
    fn check(L: *Loader, key: []const u8, bytes: []const u8) void {
        const want_map = L.o.digests orelse return;
        const want = want_map.get(key) orelse return;
        if (want != .string) return;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        L.w.digest_checked += 1;
        if (!std.mem.eql(u8, &hex, want.string)) {
            L.w.digest_bad += 1;
            std.log.err("weight digest differs from the oracle's: {s}", .{key});
        }
    }

    fn put(L: *Loader, key: []const u8, bytes: []const u8) !u64 {
        L.check(key, bytes);
        var b = try cuda.DeviceBuffer.fromHost(L.d, bytes);
        errdefer b.free();
        try L.w.buffers.append(L.gpa, b);
        return b.ptr;
    }

    fn zeros(L: *Loader, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(L.d, bytes);
        errdefer b.free();
        try b.fill8(0, null);
        try L.w.buffers.append(L.gpa, b);
        return b.ptr;
    }

    fn staging(list: *std.ArrayList(u8), gpa: std.mem.Allocator, n: usize) ![]u8 {
        try list.resize(gpa, n);
        return list.items;
    }

    /// This rank's part of a checkpoint tensor (split.rule): the mapped bytes themselves when whole, else a copy.
    fn part(L: *Loader, name: []const u8, shape_out: *[core.safetensors.max_rank]usize) ![]const u8 {
        const t = try L.ck.get(name);
        const kind = try split.rule(name);
        shape_out.* = t.shape;
        const shape = t.shape[0..t.rank];
        if (kind == .rep or L.o.world == 1) return t.bytes;
        try split.checkCut(name, kind, shape, L.o.world);
        const axis: usize = if (kind == .row) 0 else 1;
        shape_out[axis] /= L.o.world;
        const out = try staging(&L.host2, L.gpa, t.bytes.len / L.o.world);
        try split.cutBytes(out, t.bytes, shape, t.dtype.size(), kind, L.o.rank, L.o.world);
        return out;
    }

    /// weights.as_bf16: bf16 as stored; fp16 only when every value is exactly a bf16 value.
    fn bf16(L: *Loader, name: []const u8, key: []const u8) !u64 {
        const t = try L.ck.get(name);
        if (t.dtype == .bf16) {
            var shape: [core.safetensors.max_rank]usize = undefined;
            return L.put(key, try L.part(name, &shape));
        }
        if (t.dtype != .f16) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        const out = try staging(&L.host, L.gpa, raw.len);
        for (0..raw.len / 2) |i| {
            const h: f16 = @bitCast(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
            const f: f32 = @as(f32, h);
            const bits: u32 = @bitCast(f);
            if (bits & 0xFFFF != 0 or std.math.isNan(f)) {
                std.log.err("{s}: fp16 values that bf16 cannot hold exactly; refusing a rounding cast", .{name});
                return error.InexactBf16;
            }
            std.mem.writeInt(u16, out[2 * i ..][0..2], @intCast(bits >> 16), .little);
        }
        return L.put(key, out);
    }

    /// `.float()` of a bf16 / fp32 vector (the router's score-correction bias).
    fn f32vec(L: *Loader, name: []const u8, key: []const u8) !u64 {
        const t = try L.ck.get(name);
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        if (t.dtype == .f32) return L.put(key, raw);
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const out = try staging(&L.host, L.gpa, raw.len * 2);
        for (0..raw.len / 2) |i| {
            const v: u32 = @as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16;
            std.mem.writeInt(u32, out[4 * i ..][0..4], v, .little);
        }
        return L.put(key, out);
    }

    fn codebook(L: *Loader, prefix: []const u8) !exl3.Codebook {
        var buf: [256]u8 = undefined;
        if (L.has(try std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix}))) {
            _ = try L.ck.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix}));
            return .mul1;
        }
        if (L.has(try std.fmt.bufPrint(&buf, "{s}.mcg", .{prefix}))) {
            _ = try L.ck.get(try std.fmt.bufPrint(&buf, "{s}.mcg", .{prefix}));
            return .mcg;
        }
        return .@"3inst";
    }

    /// fp16 scales of `size` values (packed sign words are not in GLM-5.3's packs: refused).
    fn scales(L: *Loader, name: []const u8, key: []const u8, size: usize) !u64 {
        const t = try L.ck.get(name);
        if (t.dtype != .f16) {
            std.log.err("{s}: {t} scales (packed sign words are not implemented)", .{ name, t.dtype });
            return error.UnsupportedScales;
        }
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        if (raw.len != size * 2) return error.UnexpectedTensor;
        return L.put(key, raw);
    }

    /// weights.load_linear + Exl3Linear.from_tensors (strips layout) + x3linear.plan's tiles.
    fn linear(L: *Loader, prefix: []const u8, key: []const u8) !exl3.Linear {
        var nb: [256]u8 = undefined;
        var kb: [256]u8 = undefined;
        const tname = try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix});
        const t = try L.ck.get(tname);
        if (t.dtype != .i16 or t.rank != 3) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(tname, &shape);
        const last = shape[2];
        if (last % 8 != 0) return error.UnsupportedExl3Width;
        const k2 = last / 8;
        const kk = 16 * shape[0];
        const n = 16 * shape[1];
        if (kk % 128 != 0 or n % 128 != 0) return error.UnexpectedTensor;
        const cbk = try L.codebook(prefix);
        if (k2 % 2 != 0 and cbk != .mul1) return error.UnsupportedExl3Width;
        const words = try staging(&L.host, L.gpa, raw.len);
        exl3.strips(words, raw, kk, n, 2 * last); // a tile: 16 * bits int16 = 2 * last bytes
        const sk_wk = exl3.plan(kk, n);
        return .{
            .words = try L.put(try std.fmt.bufPrint(&kb, "{s}.words", .{key}), words),
            .suh = try L.scales(try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.suh", .{key}), kk),
            .svh = try L.scales(try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.svh", .{key}), n),
            .counters = try L.zeros(8 * (n / 128) * 4),
            .k = kk,
            .n = n,
            .k2 = k2,
            .cb = cbk,
            .sk = sk_wk[0],
            .wk = sk_wk[1],
        };
    }

    /// fused.Weights: kv_b [H (nope + v), kv_lora] -> wk [H, nope, kv_lora] and wv [H, v, kv_lora], contiguous.
    fn kvB(L: *Loader, name: []const u8, key: []const u8, c: Config, heads: usize) ![2]u64 {
        const t = try L.ck.get(name);
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        const per = c.nope + c.v_dim;
        const row = c.kv_lora * 2;
        if (shape[0] != heads * per or shape[1] != c.kv_lora) return error.UnexpectedTensor;
        // host2 may hold `raw` (a cut): copy it aside before the staging buffers are resized
        var own: []u8 = &.{};
        defer if (own.len > 0) L.gpa.free(own);
        var src = raw;
        if (L.o.world > 1) {
            own = try L.gpa.dupe(u8, raw);
            src = own;
        }
        const wk = try staging(&L.host, L.gpa, heads * c.nope * row);
        const wv = try staging(&L.host2, L.gpa, heads * c.v_dim * row);
        for (0..heads) |h| {
            @memcpy(wk[h * c.nope * row ..][0 .. c.nope * row], src[h * per * row ..][0 .. c.nope * row]);
            @memcpy(wv[h * c.v_dim * row ..][0 .. c.v_dim * row], src[(h * per + c.nope) * row ..][0 .. c.v_dim * row]);
        }
        var kb: [128]u8 = undefined;
        const a = try L.put(try std.fmt.bufPrint(&kb, "{s}.wk", .{key}), wk);
        const b = try L.put(try std.fmt.bufPrint(&kb, "{s}.wv", .{key}), wv);
        return .{ a, b };
    }

    /// weights.load_experts + experts.prepare: per-expert trellises (gate, up: output tiles; down: input tiles),
    /// int64 pointer tables, int32 widths, stacked fp16 scales; digests in prepare's order (gate, up, down).
    fn experts(L: *Loader, layer: usize, key: []const u8, c: Config) !Experts {
        const E = c.experts;
        const D = c.hidden;
        const I = c.expert_width / L.o.world;
        var nb: [256]u8 = undefined;
        var kb: [128]u8 = undefined;
        const projs = [_][]const u8{ "gate", "up", "down" };
        var ptrs: [3][]u64 = undefined;
        var k2s: [3][]i32 = undefined;
        for (0..3) |j| {
            ptrs[j] = try L.gpa.alloc(u64, E);
            k2s[j] = try L.gpa.alloc(i32, E);
        }
        defer for (0..3) |j| {
            L.gpa.free(ptrs[j]);
            L.gpa.free(k2s[j]);
        };
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var lo_gu: usize = 99;
        var hi_gu: usize = 0;
        var lo_d: usize = 99;
        var hi_d: usize = 0;
        var cbk: ?exl3.Codebook = null;
        for (projs, 0..) |pj, j| for (0..E) |e| {
            const pre = try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.{d}.{s}_proj", .{ layer, e, pj });
            if (cbk == null) cbk = try L.codebook(pre);
            var tb: [256]u8 = undefined;
            const tname = try std.fmt.bufPrint(&tb, "{s}.trellis", .{pre});
            const t = try L.ck.get(tname);
            if (t.dtype != .i16 or t.rank != 3) return error.UnexpectedTensor;
            var shape: [core.safetensors.max_rank]usize = undefined;
            const raw = try L.part(tname, &shape);
            const want_k = if (j == 2) I else D;
            const want_n = if (j == 2) D else I;
            if (shape[0] * 16 != want_k or shape[1] * 16 != want_n) return error.UnexpectedTensor;
            if (shape[2] % 8 != 0) return error.UnsupportedExl3Width;
            const k2 = shape[2] / 8;
            if (k2 < 2 or k2 > 16) return error.UnsupportedExl3Width;
            hash.update(raw);
            var b = try cuda.DeviceBuffer.fromHost(L.d, raw);
            errdefer b.free();
            try L.w.buffers.append(L.gpa, b);
            ptrs[j][e] = b.ptr;
            k2s[j][e] = @intCast(k2);
            if (j < 2) {
                lo_gu = @min(lo_gu, k2);
                hi_gu = @max(hi_gu, k2);
            } else {
                lo_d = @min(lo_d, k2);
                hi_d = @max(hi_d, k2);
            }
        };
        if (cbk.? != .mul1) return error.UnsupportedExpertCodebook; // the copy instantiates mul1 (CB 2) only
        if (L.o.digests) |dg| {
            if (dg.get(try std.fmt.bufPrint(&kb, "{s}.experts.trellis", .{key}))) |want| {
                var digest: [32]u8 = undefined;
                hash.final(&digest);
                const hex = std.fmt.bytesToHex(digest, .lower);
                L.w.digest_checked += 1;
                if (want == .string and !std.mem.eql(u8, &hex, want.string)) {
                    L.w.digest_bad += 1;
                    std.log.err("weight digest differs from the oracle's: {s}.experts.trellis", .{key});
                }
            }
        }
        var ex: Experts = undefined;
        ex.count = E;
        ex.dims = D;
        ex.width = I;
        ex.k2_gu = .{ lo_gu, hi_gu };
        ex.k2_d = .{ lo_d, hi_d };
        ex.gate_ptr = try L.put("", std.mem.sliceAsBytes(ptrs[0]));
        ex.up_ptr = try L.put("", std.mem.sliceAsBytes(ptrs[1]));
        ex.down_ptr = try L.put("", std.mem.sliceAsBytes(ptrs[2]));
        ex.gate_k2 = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.gate_k2", .{key}), std.mem.sliceAsBytes(k2s[0]));
        ex.up_k2 = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.up_k2", .{key}), std.mem.sliceAsBytes(k2s[1]));
        ex.down_k2 = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.down_k2", .{key}), std.mem.sliceAsBytes(k2s[2]));
        // stack(mats, j, n): [E, n] fp16 of each expert's suh (1) or svh (2)
        const Stack = struct { proj: []const u8, part: []const u8, n: usize, name: []const u8 };
        const stacks = [_]Stack{
            .{ .proj = "gate", .part = "suh", .n = D, .name = "suh_g" },
            .{ .proj = "up", .part = "suh", .n = D, .name = "suh_u" },
            .{ .proj = "gate", .part = "svh", .n = I, .name = "svh_g" },
            .{ .proj = "up", .part = "svh", .n = I, .name = "svh_u" },
            .{ .proj = "down", .part = "suh", .n = I, .name = "suh_d" },
            .{ .proj = "down", .part = "svh", .n = D, .name = "svh_d" },
        };
        var out: [stacks.len]u64 = undefined;
        for (stacks, 0..) |st, si| {
            const buf = try L.gpa.alloc(u8, E * st.n * 2);
            defer L.gpa.free(buf);
            for (0..E) |e| {
                const name = try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ layer, e, st.proj, st.part });
                const t = try L.ck.get(name);
                if (t.dtype != .f16) return error.UnsupportedScales; // .half() of fp16: as stored
                var shape: [core.safetensors.max_rank]usize = undefined;
                const raw = try L.part(name, &shape);
                if (raw.len != st.n * 2) return error.UnexpectedTensor;
                @memcpy(buf[e * st.n * 2 ..][0 .. st.n * 2], raw);
            }
            out[si] = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.{s}", .{ key, st.name }), buf);
        }
        ex.suh_g = out[0];
        ex.suh_u = out[1];
        ex.svh_g = out[2];
        ex.svh_u = out[3];
        ex.suh_d = out[4];
        ex.svh_d = out[5];
        return ex;
    }

    fn mlp(L: *Loader, prefix: []const u8, key: []const u8) !Mlp {
        var nb: [256]u8 = undefined;
        var kb: [128]u8 = undefined;
        return .{
            .gate = try L.linear(try std.fmt.bufPrint(&nb, "{s}.gate_proj", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.gate", .{key})),
            .up = try L.linear(try std.fmt.bufPrint(&nb, "{s}.up_proj", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.up", .{key})),
            .down = try L.linear(try std.fmt.bufPrint(&nb, "{s}.down_proj", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.down", .{key})),
        };
    }

    /// weights.load_layer for layer i.
    fn layer(L: *Loader, i: usize, c: Config) !Layer {
        var nb: [256]u8 = undefined;
        var kb: [128]u8 = undefined;
        var pb: [64]u8 = undefined;
        var key_buf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}", .{i});
        const key = try std.fmt.bufPrint(&key_buf, "layers.{d}", .{i});
        const at = struct {
            fn n(buf: []u8, pre: []const u8, s: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "{s}.{s}", .{ pre, s });
            }
        };
        var out: Layer = undefined;
        out.index = i;
        out.input_norm = try L.bf16(try at.n(&nb, p, "input_layernorm.weight"), try at.n(&kb, key, "input_norm"));
        out.post_attn_norm = try L.bf16(try at.n(&nb, p, "post_attention_layernorm.weight"), try at.n(&kb, key, "post_attn_norm"));
        out.q_a = try L.linear(try at.n(&nb, p, "self_attn.q_a_proj"), try at.n(&kb, key, "q_a"));
        out.q_a_norm = try L.bf16(try at.n(&nb, p, "self_attn.q_a_layernorm.weight"), try at.n(&kb, key, "q_a_norm"));
        out.kv_a = try L.linear(try at.n(&nb, p, "self_attn.kv_a_proj_with_mqa"), try at.n(&kb, key, "kv_a"));
        out.kv_a_norm = try L.bf16(try at.n(&nb, p, "self_attn.kv_a_layernorm.weight"), try at.n(&kb, key, "kv_a_norm"));
        out.q_b = try L.linear(try at.n(&nb, p, "self_attn.q_b_proj"), try at.n(&kb, key, "q_b"));
        const kvb = try L.kvB(try at.n(&nb, p, "self_attn.kv_b_proj.weight"), key, c, L.w.heads);
        out.wk = kvb[0];
        out.wv = kvb[1];
        out.o_proj = try L.linear(try at.n(&nb, p, "self_attn.o_proj"), try at.n(&kb, key, "o_proj"));
        out.indexer = null;
        if (L.has(try at.n(&nb, p, "self_attn.indexer.wq_b.trellis"))) {
            out.indexer = .{
                .wq_b = try L.linear(try at.n(&nb, p, "self_attn.indexer.wq_b"), try at.n(&kb, key, "wq_b")),
                .wk = try L.bf16(try at.n(&nb, p, "self_attn.indexer.wk.weight"), try at.n(&kb, key, "idx.wk")),
                .weights_proj = try L.bf16(try at.n(&nb, p, "self_attn.indexer.weights_proj.weight"), try at.n(&kb, key, "idx.weights_proj")),
                .k_norm_w = try L.bf16(try at.n(&nb, p, "self_attn.indexer.k_norm.weight"), try at.n(&kb, key, "idx.k_norm_w")),
                .k_norm_b = try L.bf16(try at.n(&nb, p, "self_attn.indexer.k_norm.bias"), try at.n(&kb, key, "idx.k_norm_b")),
            };
        }
        out.router_w = 0;
        out.router_bias = 0;
        out.experts = null;
        if (c.isMoe(i)) {
            out.router_w = try L.bf16(try at.n(&nb, p, "mlp.gate.weight"), try at.n(&kb, key, "router.weight"));
            out.router_bias = try L.f32vec(try at.n(&nb, p, "mlp.gate.e_score_correction_bias"), try at.n(&kb, key, "router.bias"));
            out.experts = try L.experts(i, key, c);
            out.mlp = try L.mlp(try at.n(&nb, p, "mlp.shared_experts"), try at.n(&kb, key, "mlp"));
        } else {
            out.mlp = try L.mlp(try at.n(&nb, p, "mlp"), try at.n(&kb, key, "mlp"));
        }
        return out;
    }
};

/// Layers 0..o.layers-1 of rank o.rank of o.world (and the embedding), from the checkpoint folder `dir`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: Config, o: Options) !Weights {
    if (c.heads % o.world != 0) return error.UnevenHeads;
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var w: Weights = .{ .gpa = gpa, .config = c, .rank = o.rank, .world = o.world, .heads = c.heads / o.world };
    errdefer w.deinit();
    var L: Loader = .{ .gpa = gpa, .d = d, .ck = &ck, .w = &w, .o = o };
    defer L.deinit();
    if (o.embed) {
        const t = try ck.expect("model.embed_tokens.weight", .bf16, &.{ c.vocab, c.hidden });
        w.embed = try L.put("embed", t.bytes);
    }
    const layers = try gpa.alloc(Layer, o.layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| {
        l.* = try L.layer(i, c);
        std.log.info("glm53: layer {d} loaded", .{i});
    }
    w.layers = layers;
    return w;
}
