//! LFM3-Omni's launches: lfm3_omni.cu by symbol and cuBLAS (classic API, fp32 compute) for every matrix product.
const std = @import("std");
const cuda = @import("cuda");
const blas = @import("blas");

pub const Op = blas.Op;

/// One field a kernel of lfm3_omni.cu, loaded by its field name.
const Fns = struct {
    om_rmsnorm: cuda.Function,
    om_head_norm: cuda.Function,
    om_rope: cuda.Function,
    om_copy_rows: cuda.Function,
    om_short_conv: cuda.Function,
    om_attention: cuda.Function,
    om_gate: cuda.Function,
    om_swiglu: cuda.Function,
    om_add: cuda.Function,
    om_bias_act: cuda.Function,
    om_layernorm: cuda.Function,
    om_embed: cuda.Function,
    om_embed_codes: cuda.Function,
    om_depth_in: cuda.Function,
    om_conv2d_s2: cuda.Function,
    om_f32_bf16: cuda.Function,
    om_bf16_f32: cuda.Function,
    om_fsq_decode: cuda.Function,
    om_repeat_rows: cuda.Function,
    om_polar: cuda.Function,
    om_argmax: cuda.Function,
    om_gemv_1: cuda.Function,
    om_gemv_4: cuda.Function,
    om_gemv_8: cuda.Function,
    om_gemv_12: cuda.Function,
    om_gemv_16: cuda.Function,
    om_depth_pick: cuda.Function,
    om_frame_end: cuda.Function,
    om_hist_push: cuda.Function,
    om_mel_frames: cuda.Function,
    om_power: cuda.Function,
    om_logmel: cuda.Function,
    om_short_conv_rows: cuda.Function,
    om_conv_state: cuda.Function,
    om_attention_tiles: cuda.Function,
    om_pos: cuda.Function,
};

/// Rows the few-row product takes; more go through cuBLAS.
pub const gemv_rows = 16;
const gemv_shared = 96 << 10;

pub const Act = enum(i32) { none = 0, silu = 1, gelu = 2 };
pub const Norm = enum(i32) { lfm = 0, head = 1 };

pub const Kernels = struct {
    mod: cuda.Module,
    fns: Fns,
    blas: blas.Blas,
    ws: cuda.DeviceBuffer,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mod = try cuda.Module.load(d, cuda.kernels.lfm3_omni);
        errdefer k.mod.unload();
        const info = @typeInfo(Fns).@"struct";
        inline for (info.field_names) |name| @field(k.fns, name) = try k.mod.function(name);
        try k.fns.om_attention.allowDynamicShared(64 << 10);
        try k.fns.om_attention_tiles.allowDynamicShared(64 << 10);
        inline for (.{ "om_gemv_1", "om_gemv_4", "om_gemv_8", "om_gemv_12", "om_gemv_16" }) |name| try @field(k.fns, name).allowDynamicShared(gemv_shared);
        const ws_bytes: usize = 32 << 20;
        k.ws = try cuda.DeviceBuffer.alloc(d, ws_bytes);
        errdefer k.ws.free();
        k.blas = try blas.Blas.open(k.ws.ptr, ws_bytes);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.blas.close();
        k.ws.free();
        k.mod.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: usize, block: usize, shared: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = @intCast(block) }, .shared = @intCast(shared) }, s, a);
    }

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    fn blocks(n: usize, per: usize) usize {
        return (n + per - 1) / per;
    }

    // ---- matrix products (row-major views over cuBLAS's column-major API)

    /// Whether `t` rows of `kdim` (rows `ldx` apart) take the few-row product.
    fn few(t: usize, kdim: usize, ldx: usize, x: u64) bool {
        return t <= gemv_rows and kdim % 8 == 0 and ldx % 8 == 0 and x % 16 == 0 and t * kdim * 2 <= gemv_shared;
    }

    /// The few-row product: y (bf16, rows `ldy` apart) and / or yf (fp32, rows n apart), `batch` products.
    pub fn gemv(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, ldx: usize, kdim: usize, n: usize, t: usize, y: u64, ldy: usize, yf: u64, batch: usize, sw: usize, sx: usize, sy: usize) !void {
        const f = if (t <= 1) k.fns.om_gemv_1 else if (t <= 4) k.fns.om_gemv_4 else if (t <= 8) k.fns.om_gemv_8 else if (t <= 12) k.fns.om_gemv_12 else k.fns.om_gemv_16;
        var a: cuda.Args = .{};
        a.add(w);
        a.add(x);
        a.add(i(ldx));
        a.add(i(kdim));
        a.add(i(n));
        a.add(i(t));
        a.add(y);
        a.add(i(ldy));
        a.add(yf);
        a.add(@as(i64, @intCast(sw)));
        a.add(@as(i64, @intCast(sx)));
        a.add(@as(i64, @intCast(sy)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(blocks(n, 8)), .y = @intCast(batch) }, .block = .{ .x = 256 }, .shared = @intCast(t * kdim * 2) }, s, &a);
    }

    /// y [t, n] = x [t, k] w^T with w [n, k]; x rows `ldx` apart, y rows `ldy` apart; bf16 out, fp32 accumulate.
    pub fn linear(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, ldx: usize, kdim: usize, n: usize, y: u64, ldy: usize, t: usize) !void {
        if (few(t, kdim, ldx, x)) return k.gemv(s, w, x, ldx, kdim, n, t, y, ldy, 0, 1, 0, 0, 0);
        try k.blas.gemm(s, .t, .n, n, t, kdim, 1.0, w, kdim, 0, x, ldx, 0, 0.0, y, ldy, 0, 1, .bf16, .bf16);
    }

    /// As `linear`, the result in fp32 (for a bias added before the one rounding).
    pub fn linearF32(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, ldx: usize, kdim: usize, n: usize, y: u64, t: usize) !void {
        if (few(t, kdim, ldx, x)) return k.gemv(s, w, x, ldx, kdim, n, t, 0, n, y, 1, 0, 0, 0);
        try k.blas.gemm(s, .t, .n, n, t, kdim, 1.0, w, kdim, 0, x, ldx, 0, 0.0, y, n, 0, 1, .bf16, .f32);
    }

    /// C [m, n] = A [m, kd] B [kd, n], any of them transposed in memory (`ta`: A stored [kd, m]), `in_t` to `out_t`.
    pub fn matmul(k: *const Kernels, s: cuda.Stream, ta: bool, tb: bool, m: usize, n: usize, kd: usize, a: u64, b: u64, c: u64, in_t: blas.DataType, out_t: blas.DataType) !void {
        // column-major C^T [n, m] = B^T A^T
        const opb: Op = if (tb) .t else .n;
        const opa: Op = if (ta) .t else .n;
        const ldb = if (tb) kd else n;
        const lda = if (ta) m else kd;
        try k.blas.gemm(s, opb, opa, n, m, kd, 1.0, b, ldb, 0, a, lda, 0, 0.0, c, n, 0, 1, in_t, out_t);
    }

    /// `batch` products y_j [t, n] = x_j [t, kd] w_j^T, the operands `sw`, `sx`, `sy` elements apart; bf16 out.
    pub fn linearBatched(k: *const Kernels, s: cuda.Stream, w: u64, sw: usize, x: u64, sx: usize, kd: usize, n: usize, y: u64, sy: usize, t: usize, batch: usize) !void {
        if (few(t, kd, kd, x) and sx % 8 == 0) return k.gemv(s, w, x, kd, kd, n, t, y, n, 0, batch, sw, sx, sy);
        try k.blas.gemm(s, .t, .n, n, t, kd, 1.0, w, kd, sw, x, kd, sx, 0.0, y, n, sy, batch, .bf16, .bf16);
    }

    // ---- elementwise and row kernels

    pub fn rmsnorm(k: *const Kernels, s: cuda.Stream, x: u64, r: u64, w: u64, y: u64, d: usize, ld: usize, ldy: usize, rows: usize, eps: f32, mode: Norm) !void {
        return k.rmsnormRows(s, x, r, w, 0, y, d, ld, ldy, rows, eps, mode);
    }

    /// As `rmsnorm` with a weight row a row (`ldw` elements apart).
    pub fn rmsnormRows(k: *const Kernels, s: cuda.Stream, x: u64, r: u64, w: u64, ldw: usize, y: u64, d: usize, ld: usize, ldy: usize, rows: usize, eps: f32, mode: Norm) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(r);
        a.add(w);
        a.add(y);
        a.add(i(d));
        a.add(i(ld));
        a.add(i(ldy));
        a.add(eps);
        a.add(@backingInt(mode));
        a.add(i(ldw));
        try go(k.fns.om_rmsnorm, s, rows, 256, 0, &a);
    }

    pub fn headNorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, t: usize, h: usize, hd: usize, ld: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(i(t));
        a.add(i(h));
        a.add(i(hd));
        a.add(i(ld));
        a.add(eps);
        try go(k.fns.om_head_norm, s, blocks(t * h * 32, 256), 256, 0, &a);
    }

    pub fn rope(k: *const Kernels, s: cuda.Stream, x: u64, inv: u64, t: usize, h: usize, hd: usize, ld: usize, pos0: usize, mode: u8, posp: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(inv);
        a.add(i(t));
        a.add(i(h));
        a.add(i(hd));
        a.add(i(ld));
        a.add(i(pos0));
        a.add(@as(i32, mode));
        a.add(posp);
        try go(k.fns.om_rope, s, blocks(t * h * hd / 2, 256), 256, 0, &a);
    }

    pub fn copyRows(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, t: usize, n: usize, lds: usize, ldd: usize, posp: u64) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(i(t));
        a.add(i(n));
        a.add(i(lds));
        a.add(i(ldd));
        a.add(posp);
        try go(k.fns.om_copy_rows, s, blocks(t * n, 256), 256, 0, &a);
    }

    /// The device position: set to `v`, or advanced by it.
    pub fn pos(k: *const Kernels, s: cuda.Stream, p: u64, v: usize, set: bool) !void {
        var a: cuda.Args = .{};
        a.add(p);
        a.add(i(v));
        a.add(@as(i32, @intFromBool(set)));
        try go(k.fns.om_pos, s, 1, 32, 0, &a);
    }

    pub fn shortConv(k: *const Kernels, s: cuda.Stream, bcx: u64, ld: usize, w: u64, state: u64, y: u64, ldy: usize, t: usize, d: usize, kw: usize) !void {
        if (t > 4) {
            var a: cuda.Args = .{};
            a.add(bcx);
            a.add(i(ld));
            a.add(w);
            a.add(state);
            a.add(y);
            a.add(i(ldy));
            a.add(i(t));
            a.add(i(d));
            a.add(i(kw));
            try go(k.fns.om_short_conv_rows, s, blocks(t * d, 256), 256, 0, &a);
            var c: cuda.Args = .{};
            c.add(bcx);
            c.add(i(ld));
            c.add(state);
            c.add(i(t));
            c.add(i(d));
            c.add(i(kw));
            return go(k.fns.om_conv_state, s, blocks(d, 128), 128, 0, &c);
        }
        var a: cuda.Args = .{};
        a.add(bcx);
        a.add(i(ld));
        a.add(w);
        a.add(state);
        a.add(y);
        a.add(i(ldy));
        a.add(i(t));
        a.add(i(d));
        a.add(i(kw));
        try go(k.fns.om_short_conv, s, blocks(d, 128), 128, 0, &a);
    }

    pub const Attn = struct {
        q: u64,
        ldq: usize,
        kc: u64,
        vc: u64,
        ldkv: usize,
        o: u64,
        ldo: usize,
        t: usize,
        heads: usize,
        kv_heads: usize,
        hd: usize,
        pos0: usize,
        kv_len: usize,
        window: usize,
        causal: bool,
        /// A replayed graph's row: the position on the device, the cache sized for `max_ctx` keys.
        posp: u64 = 0,
        max_ctx: usize = 0,
    };

    pub fn attention(k: *const Kernels, s: cuda.Stream, p: Attn) !void {
        if (p.t > 1 and p.posp == 0 and p.hd % 32 == 0 and p.hd <= 128) {
            const shared = (16 * p.hd + 16 * 64) * 4 + 2 * 64 * (p.hd + 2) * 2;
            var a: cuda.Args = .{};
            a.add(p.q);
            a.add(i(p.ldq));
            a.add(p.kc);
            a.add(p.vc);
            a.add(i(p.ldkv));
            a.add(p.o);
            a.add(i(p.ldo));
            a.add(i(p.t));
            a.add(i(p.heads));
            a.add(i(p.kv_heads));
            a.add(i(p.hd));
            a.add(i(p.pos0));
            a.add(i(p.kv_len));
            a.add(i(p.window));
            a.add(@as(i32, @intFromBool(p.causal)));
            a.add(1.0 / @sqrt(@as(f32, @floatFromInt(p.hd))));
            return cuda.launch.launch(k.fns.om_attention_tiles, .{ .grid = .{ .x = @intCast(blocks(p.t, 16)), .y = @intCast(p.heads) }, .block = .{ .x = 128 }, .shared = @intCast(shared) }, s, &a);
        }
        const keys = if (p.posp != 0) p.max_ctx else p.kv_len;
        const span = if (p.causal and p.window > 0) @min(p.window, keys) else keys;
        const shared = (128 + span) * 4;
        if (shared > 64 << 10) return error.AttentionSpanTooLong;
        var a: cuda.Args = .{};
        a.add(p.q);
        a.add(i(p.ldq));
        a.add(p.kc);
        a.add(p.vc);
        a.add(i(p.ldkv));
        a.add(p.o);
        a.add(i(p.ldo));
        a.add(i(p.heads));
        a.add(i(p.kv_heads));
        a.add(i(p.hd));
        a.add(i(p.pos0));
        a.add(i(p.kv_len));
        a.add(i(p.window));
        a.add(@as(i32, @intFromBool(p.causal)));
        a.add(1.0 / @sqrt(@as(f32, @floatFromInt(p.hd))));
        a.add(p.posp);
        try go(k.fns.om_attention, s, p.t * p.heads, 128, shared, &a);
    }

    pub fn gate(k: *const Kernels, s: cuda.Stream, o: u64, ldo: usize, g: u64, ldg: usize, t: usize, h: usize, hd: usize) !void {
        var a: cuda.Args = .{};
        a.add(o);
        a.add(i(ldo));
        a.add(g);
        a.add(i(ldg));
        a.add(i(t));
        a.add(i(h));
        a.add(i(hd));
        try go(k.fns.om_gate, s, blocks(t * h * hd, 256), 256, 0, &a);
    }

    pub fn swiglu(k: *const Kernels, s: cuda.Stream, ac: u64, y: u64, t: usize, f: usize) !void {
        var a: cuda.Args = .{};
        a.add(ac);
        a.add(y);
        a.add(i(t));
        a.add(i(f));
        try go(k.fns.om_swiglu, s, blocks(t * f, 256), 256, 0, &a);
    }

    pub fn add(k: *const Kernels, s: cuda.Stream, y: u64, x: u64, r: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(x);
        a.add(r);
        a.add(@as(i64, @intCast(n)));
        try go(k.fns.om_add, s, blocks(n, 256), 256, 0, &a);
    }

    pub fn biasAct(k: *const Kernels, s: cuda.Stream, acc: u64, bias: u64, y: u64, rows: usize, cols: usize, per_row: bool, act: Act) !void {
        var a: cuda.Args = .{};
        a.add(acc);
        a.add(bias);
        a.add(y);
        a.add(i(rows));
        a.add(i(cols));
        a.add(@as(i32, @intFromBool(per_row)));
        a.add(@backingInt(act));
        try go(k.fns.om_bias_act, s, blocks(rows * cols, 256), 256, 0, &a);
    }

    pub fn layernorm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, bias: u64, y: u64, d: usize, rows: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(bias);
        a.add(y);
        a.add(i(d));
        a.add(eps);
        try go(k.fns.om_layernorm, s, rows, 256, 0, &a);
    }

    pub fn embed(k: *const Kernels, s: cuda.Stream, table: u64, ids: u64, y: u64, t: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(table);
        a.add(ids);
        a.add(y);
        a.add(i(t));
        a.add(i(d));
        try go(k.fns.om_embed, s, blocks(t * d, 256), 256, 0, &a);
    }

    pub fn embedCodes(k: *const Kernels, s: cuda.Stream, table: u64, codes: u64, y: u64, t: usize, d: usize, nq: usize, size: usize) !void {
        var a: cuda.Args = .{};
        a.add(table);
        a.add(codes);
        a.add(y);
        a.add(i(t));
        a.add(i(d));
        a.add(i(nq));
        a.add(i(size));
        try go(k.fns.om_embed_codes, s, blocks(t * d, 256), 256, 0, &a);
    }

    pub fn depthIn(k: *const Kernels, s: cuda.Stream, din: u64, mask: u64, tables: u64, codes: u64, masked: u64, x: u64, nq: usize, d: usize, size: usize) !void {
        var a: cuda.Args = .{};
        a.add(din);
        a.add(mask);
        a.add(tables);
        a.add(codes);
        a.add(masked);
        a.add(x);
        a.add(i(nq));
        a.add(i(d));
        a.add(i(size));
        try go(k.fns.om_depth_in, s, blocks(nq * d, 256), 256, 0, &a);
    }

    pub fn conv2dS2(k: *const Kernels, s: cuda.Stream, x: u64, cin: usize, h: usize, w: usize, wt: u64, bias: u64, y: u64, c: usize, h2: usize, w2: usize, silu: bool) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(cin));
        a.add(i(h));
        a.add(i(w));
        a.add(wt);
        a.add(bias);
        a.add(y);
        a.add(i(c));
        a.add(i(h2));
        a.add(i(w2));
        a.add(@as(i32, @intFromBool(silu)));
        try go(k.fns.om_conv2d_s2, s, blocks(c * h2 * w2, 256), 256, 0, &a);
    }

    pub fn f32ToBf16(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try go(k.fns.om_f32_bf16, s, blocks(n, 256), 256, 0, &a);
    }

    pub fn bf16ToF32(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try go(k.fns.om_bf16_f32, s, blocks(n, 256), 256, 0, &a);
    }

    pub fn fsqDecode(k: *const Kernels, s: cuda.Stream, codes: u64, l: u64, n: usize, nq: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(l);
        a.add(i(n));
        a.add(i(nq));
        try go(k.fns.om_fsq_decode, s, blocks(n * nq, 128), 128, 0, &a);
    }

    pub fn repeatRows(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: usize, d: usize, rep: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(i(n));
        a.add(i(d));
        a.add(i(rep));
        try go(k.fns.om_repeat_rows, s, blocks(n * rep * d, 256), 256, 0, &a);
    }

    pub fn polar(k: *const Kernels, s: cuda.Stream, x: u64, spec: u64, t: usize, nb: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(spec);
        a.add(i(t));
        a.add(i(nb));
        try go(k.fns.om_polar, s, blocks(t * nb, 256), 256, 0, &a);
    }

    pub fn argmax(k: *const Kernels, s: cuda.Stream, x: u64, n: usize, sup: u64, ns: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(n));
        a.add(sup);
        a.add(i(ns));
        a.add(out);
        try go(k.fns.om_argmax, s, 1, 1024, 0, &a);
    }

    pub const Pick = struct {
        logits: u64,
        order: u64,
        p0: usize,
        n: usize,
        hist: u64,
        hn: usize,
        penalty: f32,
        temperature: f32,
        seed: u64,
        frame: usize,
        codes: u64,
        masked: u64,
        nq: usize,
        size: usize,
        /// [hn, frame] on the device (a replayed graph's), else the values above.
        meta: u64 = 0,
    };

    pub fn depthPick(k: *const Kernels, s: cuda.Stream, p: Pick) !void {
        var a: cuda.Args = .{};
        a.add(p.logits);
        a.add(p.order);
        a.add(i(p.p0));
        a.add(p.hist);
        a.add(i(p.hn));
        a.add(p.penalty);
        a.add(p.temperature);
        a.add(p.seed);
        a.add(i(p.frame));
        a.add(p.codes);
        a.add(p.masked);
        a.add(i(p.nq));
        a.add(i(p.size));
        a.add(p.meta);
        try go(k.fns.om_depth_pick, s, p.n, 256, p.size, &a);
    }

    pub fn frameEnd(k: *const Kernels, s: cuda.Stream, codes: u64, nq: usize, eos: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(i(nq));
        a.add(i(eos));
        try go(k.fns.om_frame_end, s, 1, 32, 0, &a);
    }

    pub fn histPush(k: *const Kernels, s: cuda.Stream, codes: u64, hist: u64, slot: usize, nq: usize) !void {
        var a: cuda.Args = .{};
        a.add(codes);
        a.add(hist);
        a.add(i(slot));
        a.add(i(nq));
        try go(k.fns.om_hist_push, s, 1, 32, 0, &a);
    }

    pub fn melFrames(k: *const Kernels, s: cuda.Stream, x: u64, len: usize, win: u64, fr: u64, nf: usize, nfft: usize, hop: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(len));
        a.add(win);
        a.add(fr);
        a.add(i(nf));
        a.add(i(nfft));
        a.add(i(hop));
        try go(k.fns.om_mel_frames, s, blocks(nf * nfft, 256), 256, 0, &a);
    }

    pub fn power(k: *const Kernels, s: cuda.Stream, spec: u64, p: u64, nf: usize, nb: usize) !void {
        var a: cuda.Args = .{};
        a.add(spec);
        a.add(p);
        a.add(i(nf));
        a.add(i(nb));
        try go(k.fns.om_power, s, blocks(nf * nb, 256), 256, 0, &a);
    }

    pub fn logmel(k: *const Kernels, s: cuda.Stream, m: u64, out: u64, nf: usize, mels: usize, floor: f32) !void {
        var a: cuda.Args = .{};
        a.add(m);
        a.add(out);
        a.add(i(nf));
        a.add(i(mels));
        a.add(floor);
        try go(k.fns.om_logmel, s, blocks(nf * mels, 256), 256, 0, &a);
    }
};
