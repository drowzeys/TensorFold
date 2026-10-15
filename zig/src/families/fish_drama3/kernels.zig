//! Drama 3's CUDA kernels: drama3.cu (ours) by symbol, the routed FP8 experts and their plan (fp8_experts.cu and
//! experts.cu through cuda/experts.zig and cuda/grouped.zig), and cuBLAS for prompt-row GEMMs. Each wrapper takes
//! device addresses (u64) and passes every argument with its kernel parameter's exact C type.
const std = @import("std");
const cuda = @import("cuda");

pub const Sampling = extern struct {
    temperature: f32,
    top_k: i32,
    top_p: f32,
    pad: i32 = 0,
    seed: u64,
};

const names = .{
    "d3_gemv",        "d3_rmsnorm",    "d3_add_rmsnorm", "d3_embed_backbone", "d3_conv_silu",
    "d3_gdn",         "d3_gated_norm", "d3_qk_rope",     "d3_attn_decode",    "d3_q_group",
    "d3_causal_softmax", "d3_attn_gate", "d3_router",   "d3_swiglu",         "d3_moe_combine",
    "d3_tower_mix",   "d3_sample_main", "d3_sample_code", "d3_argmax_main",   "d3_finish_frame",
    "d3_copy_i32",
};

pub const F = enum(u8) {
    gemv,
    rmsnorm,
    add_rmsnorm,
    embed_backbone,
    conv_silu,
    gdn,
    gated_norm,
    qk_rope,
    attn_decode,
    q_group,
    causal_softmax,
    attn_gate,
    router,
    swiglu,
    moe_combine,
    tower_mix,
    sample_main,
    sample_code,
    argmax_main,
    finish_frame,
    copy_i32,
};

const n_fns = @typeInfo(F).@"enum".field_names.len;

pub const Kernels = struct {
    d: *const cuda.Driver,
    mod: cuda.Module,
    fns: [n_fns]cuda.Function,
    experts_mod: cuda.Module,
    plan: cuda.grouped.Router,
    fp8: cuda.experts.Fp8Experts,
    blas_lib: cuda.cublas.Library,
    blas: cuda.cublas.Blas,
    ws: cuda.DeviceBuffer,
    sms: usize,

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        k.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        k.mod = try cuda.Module.load(d, cuda.kernels.drama3);
        inline for (names, 0..) |name, j| k.fns[j] = try k.mod.function(name);
        try k.fns[@intFromEnum(F.sample_main)].allowDynamicShared(8192 * 8);
        try k.fns[@intFromEnum(F.gemv)].allowDynamicShared(6144 * 2);
        k.experts_mod = try cuda.Module.load(d, cuda.kernels.experts);
        k.plan = try cuda.grouped.Router.resolve(k.experts_mod);
        k.fp8 = try cuda.experts.Fp8Experts.load(d, k.sms);
        k.blas_lib = try cuda.cublas.Library.open();
        const ws_bytes: usize = 32 << 20;
        k.ws = try cuda.DeviceBuffer.alloc(d, ws_bytes);
        k.blas = try cuda.cublas.Blas.init(&k.blas_lib, k.ws.ptr, ws_bytes, cuda.cublas.default_math);
        return k;
    }

    /// The blas handle keeps a pointer to the library: call after the Kernels value has its final address.
    pub fn rebind(k: *Kernels) void {
        k.blas.lib = &k.blas_lib;
    }

    fn f(k: *const Kernels, which: F) cuda.Function {
        return k.fns[@intFromEnum(which)];
    }

    fn go(k: *const Kernels, which: F, s: cuda.Stream, gx: usize, gy: usize, block: usize, shared: usize, a: *cuda.Args) !void {
        try cuda.launch.launch(k.f(which), .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block) }, .shared = @intCast(shared) }, s, a);
    }

    fn i(x: usize) i32 {
        return @intCast(x);
    }

    // ---- dense linear: y [T, N] = x [T, K] W^T, bf16 (decode rows through the GEMV, prompt rows through cuBLAS)

    /// One row: y (bf16, may be 0) and / or yf (fp32, may be 0).
    pub fn gemv(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, kdim: usize, n: usize, y: u64, yf: u64) !void {
        var a: cuda.Args = .{};
        a.add(w);
        a.add(x);
        a.add(i(kdim));
        a.add(i(n));
        a.add(y);
        a.add(yf);
        try k.go(.gemv, s, (n + 7) / 8, 1, 256, kdim * 2, &a);
    }

    /// Rows: x rows xs elements apart, y rows ys apart (bf16 out, fp32 accumulate).
    pub fn linear(k: *const Kernels, s: cuda.Stream, w: u64, x: u64, xs: usize, kdim: usize, n: usize, y: u64, ys: usize, t: usize) !void {
        if (t == 1 and xs >= kdim) return k.gemv(s, w, x, kdim, n, y, 0);
        try k.blas.bgemmBf16(s.handle, .t, .n, n, t, kdim, w, kdim, 0, x, xs, 0, y, ys, 0, 1);
    }

    /// Column-major C = alpha op(A) op(B), batched, bf16 in, fp32 or bf16 out (the attention score / value GEMMs).
    pub fn gemmEx(k: *const Kernels, s: cuda.Stream, ta: cuda.cublas.Op, tb: cuda.cublas.Op, m: usize, n: usize, kk: usize, alpha: f32, a_: u64, lda: usize, sa: usize, b_: u64, ldb: usize, sb: usize, c: u64, ldc: usize, sc: usize, batch: usize, out_f32: bool) !void {
        const lib = &k.blas_lib;
        try lib.check(lib.api.cublasSetStream_v2(k.blas.h, s.handle), "cublasSetStream");
        const beta: f32 = 0.0;
        try lib.check(lib.api.cublasGemmStridedBatchedEx(k.blas.h, ta, tb, @intCast(m), @intCast(n), @intCast(kk), &alpha, a_, .bf16, @intCast(lda), @intCast(sa), b_, .bf16, @intCast(ldb), @intCast(sb), &beta, c, if (out_f32) .f32 else .bf16, @intCast(ldc), @intCast(sc), @intCast(batch), cuda.cublas.compute_32f, cuda.cublas.gemm_default_tensor_op), "cublasGemmStridedBatchedEx");
    }

    // ---- norms and elementwise

    pub fn rmsnorm(k: *const Kernels, s: cuda.Stream, x: u64, xs: usize, w: u64, y: u64, ys: usize, rows: usize, dim: usize, eps: f32, zero_centred: bool) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(xs));
        a.add(w);
        a.add(y);
        a.add(i(ys));
        a.add(i(dim));
        a.add(eps);
        a.add(@as(i32, @intFromBool(zero_centred)));
        try k.go(.rmsnorm, s, rows, 1, 256, 0, &a);
    }

    pub fn addRmsnorm(k: *const Kernels, s: cuda.Stream, r: u64, add: u64, add_stride: usize, w: u64, y: u64, rows: usize, dim: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(r);
        a.add(add);
        a.add(i(add_stride));
        a.add(w);
        a.add(y);
        a.add(i(dim));
        a.add(eps);
        try k.go(.add_rmsnorm, s, rows, 1, 256, 0, &a);
    }

    pub fn embedBackbone(k: *const Kernels, s: cuda.Stream, frames: u64, embed: u64, cb_emb: u64, out: u64, rows: usize, dim: usize, sem_begin: i32, sem_end: i32, cb: i32) !void {
        var a: cuda.Args = .{};
        a.add(frames);
        a.add(embed);
        a.add(cb_emb);
        a.add(out);
        a.add(i(dim));
        a.add(sem_begin);
        a.add(sem_end);
        a.add(cb);
        try k.go(.embed_backbone, s, rows, 1, 256, 0, &a);
    }

    pub fn convSilu(k: *const Kernels, s: cuda.Stream, proj: u64, ps: usize, t: usize, w: u64, state: u64, out: u64, channels: usize) !void {
        var a: cuda.Args = .{};
        a.add(proj);
        a.add(i(ps));
        a.add(i(t));
        a.add(w);
        a.add(state);
        a.add(out);
        a.add(i(channels));
        try k.go(.conv_silu, s, (channels + 127) / 128, 1, 128, 0, &a);
    }

    pub fn gdn(k: *const Kernels, s: cuda.Stream, conv: u64, proj: u64, ps: usize, t: usize, a_log: u64, dt_bias: u64, state: u64, o: u64) !void {
        var a: cuda.Args = .{};
        a.add(conv);
        a.add(proj);
        a.add(i(ps));
        a.add(i(t));
        a.add(a_log);
        a.add(dt_bias);
        a.add(state);
        a.add(o);
        try k.go(.gdn, s, 32, 4, 32, 0, &a);
    }

    pub fn gatedNorm(k: *const Kernels, s: cuda.Stream, o: u64, proj: u64, ps: usize, w: u64, y: u64, t: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(o);
        a.add(proj);
        a.add(i(ps));
        a.add(w);
        a.add(y);
        a.add(eps);
        try k.go(.gated_norm, s, t, 4, 256, 0, &a);
    }

    pub const QkLayout = struct { q_off: usize, q_head_stride: usize, k_off: usize, v_off: usize, gate_off: usize, gate_head_stride: usize };

    pub fn qkRope(k: *const Kernels, s: cuda.Stream, proj: u64, ps: usize, lay: QkLayout, qn: u64, kn: u64, pos0: u64, qout: u64, kc: u64, vc: u64, t: usize, interleaved: bool, theta: f32, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(proj);
        a.add(i(ps));
        a.add(i(lay.q_off));
        a.add(i(lay.q_head_stride));
        a.add(i(lay.k_off));
        a.add(i(lay.v_off));
        a.add(qn);
        a.add(kn);
        a.add(pos0);
        a.add(qout);
        a.add(kc);
        a.add(vc);
        a.add(@as(i32, @intFromBool(interleaved)));
        a.add(theta);
        a.add(eps);
        try k.go(.qk_rope, s, t, 18, 256, 0, &a);
    }

    pub fn attnDecode(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, vc: u64, pos0: u64, proj: u64, ps: usize, lay: QkLayout, out: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(kc);
        a.add(vc);
        a.add(pos0);
        a.add(proj);
        a.add(i(ps));
        a.add(i(lay.gate_off));
        a.add(i(lay.gate_head_stride));
        a.add(out);
        try k.go(.attn_decode, s, t, 16, 256, 0, &a);
    }

    pub fn qGroup(k: *const Kernels, s: cuda.Stream, q: u64, qg: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(qg);
        a.add(i(t));
        try k.go(.q_group, s, t, 16, 256, 0, &a);
    }

    pub fn causalSoftmax(k: *const Kernels, s: cuda.Stream, sc: u64, p: u64, t: usize, ld: usize, pos0: u64) !void {
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(p);
        a.add(i(t));
        a.add(i(ld));
        a.add(pos0);
        try k.go(.causal_softmax, s, 2 * 8 * t, 1, 256, 0, &a);
    }

    pub fn attnGate(k: *const Kernels, s: cuda.Stream, og: u64, proj: u64, ps: usize, lay: QkLayout, out: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(og);
        a.add(proj);
        a.add(i(ps));
        a.add(i(lay.gate_off));
        a.add(i(lay.gate_head_stride));
        a.add(out);
        a.add(i(t));
        try k.go(.attn_gate, s, t, 16, 256, 0, &a);
    }

    pub fn router(k: *const Kernels, s: cuda.Stream, sp: u64, ss: usize, r_off: usize, picks: u64, weights: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(sp);
        a.add(i(ss));
        a.add(i(r_off));
        a.add(picks);
        a.add(weights);
        try k.go(.router, s, t, 1, 256, 0, &a);
    }

    pub fn swiglu(k: *const Kernels, s: cuda.Stream, x: u64, stride: usize, g_off: usize, u_off: usize, n: usize, y: u64, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(i(stride));
        a.add(i(g_off));
        a.add(i(u_off));
        a.add(i(n));
        a.add(y);
        try k.go(.swiglu, s, (n + 255) / 256, t, 256, 0, &a);
    }

    pub fn moeCombine(k: *const Kernels, s: cuda.Stream, r: u64, down: u64, w: u64, shared: u64, sp: u64, ss: usize, sg_off: usize, dim: usize, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(r);
        a.add(down);
        a.add(w);
        a.add(shared);
        a.add(sp);
        a.add(i(ss));
        a.add(i(sg_off));
        a.add(i(dim));
        try k.go(.moe_combine, s, t, 1, 256, 0, &a);
    }

    pub fn towerMix(k: *const Kernels, s: cuda.Stream, lf: u64, lfs: usize, code_col: usize, inf: u64, with_shared: bool, table: u64, cb_emb: u64, h: u64, enorm: u64, hnorm: u64, out: u64, dim: usize, sem_begin: i32, sem_end: i32, cb: i32, eps: f32, t: usize) !void {
        var a: cuda.Args = .{};
        a.add(lf);
        a.add(i(lfs));
        a.add(i(code_col));
        a.add(inf);
        a.add(@as(i32, @intFromBool(with_shared)));
        a.add(table);
        a.add(cb_emb);
        a.add(h);
        a.add(enorm);
        a.add(hnorm);
        a.add(out);
        a.add(i(dim));
        a.add(sem_begin);
        a.add(sem_end);
        a.add(cb);
        a.add(eps);
        try k.go(.tower_mix, s, t, 1, 256, 0, &a);
    }

    pub fn sampleMain(k: *const Kernels, s: cuda.Stream, logits: u64, count: usize, cand_ids: u64, window: u64, sp: Sampling, step: u64, frame: u64, sem_begin: i32, cb: i32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(i(count));
        a.add(cand_ids);
        a.add(window);
        a.add(sp);
        a.add(step);
        a.add(frame);
        a.add(sem_begin);
        a.add(cb);
        try k.go(.sample_main, s, 1, 1, 1024, 8192 * 8, &a);
    }

    pub fn sampleCode(k: *const Kernels, s: cuda.Stream, logits: u64, count: usize, sp: Sampling, step: u64, stream_id: i32, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(i(count));
        a.add(sp);
        a.add(step);
        a.add(stream_id);
        a.add(out);
        try k.go(.sample_code, s, 1, 1, 1024, 4096 * 8, &a);
    }

    /// Greedy: cand_ids non-zero picks a main token (frame[0..2]), zero a code (frame[0]).
    pub fn argmax(k: *const Kernels, s: cuda.Stream, logits: u64, count: usize, cand_ids: u64, frame: u64, sem_begin: i32, cb: i32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(i(count));
        a.add(cand_ids);
        a.add(frame);
        a.add(sem_begin);
        a.add(cb);
        try k.go(.argmax_main, s, 1, 1, 1024, 0, &a);
    }

    pub fn finishFrame(k: *const Kernels, s: cuda.Stream, frame: u64, frames_out: u64, next_input: u64, window: u64, pos: u64, step: u64, count: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ frame, frames_out, next_input, window, pos, step, count }) |v| a.add(v);
        try k.go(.finish_frame, s, 1, 1, 32, 0, &a);
    }

    pub fn copyI32(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(i(n));
        try k.go(.copy_i32, s, 1, 1, 64, 0, &a);
    }
};
