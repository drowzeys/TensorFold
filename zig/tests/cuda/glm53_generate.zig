//! tf-glm53-generate: full GLM-5.3 at tensor parallelism in Zig (Phase 2a), one process a rank: every layer, the final
//! norm and the vocabulary-sharded argmax head; NCCL all-gathers between ranks; GREEDY decode without drafts. For each
//! prompt (token ids, e.g. the chat template's with thinking on): the prompt in windows of up to --window rows (the
//! last window's last row picks the first token), then one-row decode windows, --tokens picks in all (EOS is not a
//! stop: every run makes the same number of picks). Each pick's gathered [world, 4] fp32 records (every rank's max
//! logit and its id) are kept bit for bit, so tools/glm53/compare_phase2a.py can check per-step logit-argmax equality
//! with tools/glm53/reference.py (the Python engine on the same schedule) besides the tokens.
//!
//!
//! --mode 2b (Phase 2b, cuda_runner.zig): the Python Runner's schedule - the prompt in equal chunks of up to --window
//! rows through the target and the MTP layer, then MTP-drafted rounds (--k drafts, verify windows of k + 1 rows) as
//! CUDA graphs (prewarmed: every bucket's windows captured before the first request), greedy or keyed sampled draws.
//! Each prompt of the prompts file names its runs ("g:K" greedy, "s<top_k>:K" sampled at the file's temperature /
//! top_p / min_p with the prompt's seed; K drafts, 0 = serial) and its token count. Prints per run the prefill time,
//! decode tok/s, tokens a round and acceptance; writes them with the tokens to --out for compare_phase2b.py. The
//! weights load with O_DIRECT through io_uring (--fast-load 1), the MTP layer and the 4-bit draft head
//! (--draft-vocab 32768, fused.DRAFT_VOCAB) included.
//! --sampler-check FILE (no GPU): sampling.zig's draws and seeds against cases tools/glm53/sampler_cases.py wrote
//! with the Python engine's exact_sampling; prints `SAMPLER PASS|FAIL ...`.
//! --mode 3a (Phase 3a): 2b's runs with the served engine's settings, against tools/glm53/reference3a.py (the
//! Python engine in its served configuration): the RoCE one-shot for decode windows (--roce 1 --hcas A,B --gid N), the
//! reference boot's EXL3 tiles (--tiles FILE), the served prompt path (--prompt-rows 8192 --prompt-rows-short 4096
//! --prompt-sp 1: sequence-parallel halves, the prompt GEMM, cuBLAS absorb / expand checked first on --bmm-probe FILE,
//! prompt experts --experts shared --pe-det slots16), NCCL's bf16 ring exchanges for prompt rows.
//! --mode 3b (Phase 3b): the served engine itself (cuda_engine.zig: Boot, Session - what tensorfold-native serve runs
//! on every rank) over a prompts file: EOS stops a run (--stop-eos 1), --reuse 1 cuts each prompt at the prompt-reuse
//! plan's keep points as the server cuts a cold prompt (every run starts cold: the kept states are forgotten first),
//! --dcp 0|1|4 (0: on past 200,000 tokens of context). Each prompt may name its own "temperature", "top_p", "min_p"
//! and "eos" (ids) over the file's. The served == CLI check: the server's requests (TF_GLM53_DUMP) run here.
//! --mode 4 (Phase 4): 3b's served engine with the speculative drafters: --dspark DIR (cuda_draft.zig, DSpark: its
//! settings --dspark-policy cost|confidence|fixed --dspark-confidence X --dspark-depth N --dspark-topk N) and / or
//! --dflash DIR (DFlash2: --dflash-depth 7 --dflash-confidence 0.3), copy drafts (--copy 0|1, default on when a run
//! asks for them; --copy-min 8 --copy-max 15). A run label names its drafts: "g:K" / "s<top_k>:K" the MTP head with K
//! steps (0: serial), "g:dspark" / "s20:dflash" a drafter (--mtp-drafts 0 loads no MTP layer), a "+c" suffix copy
//! drafts ahead of them ("s20:dspark+c"). The output adds each run's draft mode and copy counts.
//! --mode 4b (Phase 4b): the served engine with --parallel N (cuda_multi.zig: concurrent streams, shared decode
//! rounds): every rank runs the same schedule (the lead's, no messages) over the prompts file's groups - each group's
//! prompts admitted together and decoded concurrently, then each one alone through the concurrent decoder, then
//! (--single 1) each one through the one-stream path (Runner.run in slot 0, copy drafts off). Greedy or keyed
//! sampled; every stream's tokens must equal its alone run's and its one-stream run's. Prints aggregate and per-stream
//! tok/s and TTFT; `RESULT glm53-generate OK|FAIL` (FAIL: some stream differs). list-variants --parallel N adds the
//! concurrent windows' launches (ROWS kernels, N x (k + 1) rows).
//! Phase 4c: --mode 4b with --dspark DIR / --dflash DIR (and their settings as --mode 4) and --draft-mode
//! dflash|dspark|mtp (default: TF_GLM53_MTP's rule - a drafter alone with --mtp-drafts 0, else the MTP head): every
//! stream drafts with that drafter (cuda_mdraft.zig) - concurrently, alone through the concurrent decoder, and through
//! the one-stream path (Runner.run with the one-stream drafter); tokens must match. list-variants --parallel N with
//! --dspark / --dflash adds the concurrent drafters' passes and windows of N x max(k + 1, drafter rows) rows.
//! --mode roce-bench: the RoCE runtime alone (no model): bit check against NCCL, then microseconds a reduction of
//! [rows, --dims] fp32 for --bench-rows; prints `ROCE-BENCH PASS|FAIL ...` (decode windows <= 65 us).
//!
//! usage: tf-glm53-generate --rank R --world W --master IP:PORT --model DIR --aot DIR --prompts FILE --tokens N
//!                          --out FILE [--context 32768] [--window 128] [--layers N] [--inv-ref FILE]
//!                          [--nccl-lib PATH] [--timeout S] [--mode 2a|2b|3a] [--k 2] [--prewarm 1] [--graphs 1]
//!                          [--fast-load 0|1] [--draft-vocab 32768] [--mtp-reuse 2]
//!                          3a: [--roce 1] [--hcas A,B] [--gid 3] [--roce-health 1] [--spin-limit N] [--tiles FILE]
//!                          [--prompt-rows 8192] [--prompt-rows-short 4096] [--prompt-sp 1] [--experts shared|tf]
//!                          [--pe-det slots16|0] [--cublas-lib PATH] [--cublas-ws BYTES] [--cublas-math 0]
//!                          [--unpack-mb 384] [--bmm-probe FILE] [--side af|0] [--l2pf 1|0|lines|touch]
//!                          [--l2pf-mb 8] [--multi-select 1] [--device-cands 1] [--prompt-prof NAME]
//!                          [--mtp-dense 1]
//!        tf-glm53-generate --mode roce-bench --rank R --world W --master IP:PORT --hcas A,B --gid N
//!                          [--bench-rows 1,2,3,4,8,16,32] [--bench-iters 200] [--dims 6144] [--out FILE]
//!        tf-glm53-generate --sampler-check FILE
//!        tf-glm53-generate --mode list-variants --model DIR --world 4 --context N --tiles F0,F1,.. [--aot DIR]
//!                          [--prompt-rows 8192] [--dcp 0|1|4] [--out FILE] [--dspark DIR] [--dflash DIR]
//! --mode list-variants (no GPU): every Triton launch shape the engine can make for that configuration (cuda_coverage
//! .zig: Tri in probe mode over every window size, index range and DCP rank) as JSON for tools/glm53/sweep_aot.py;
//! with --aot, each marked "have" and `VARIANTS PASS|FAIL n needs, m missing` printed (exit 3 when any is missing).
//! Prints `RESULT glm53-generate OK|FAIL ...`; exit 0 only on OK.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm53");

const usage =
    \\usage: tf-glm53-generate --rank R --world W --master IP:PORT --model DIR --aot DIR --prompts FILE --tokens N
    \\                         --out FILE [--context 32768] [--window 128] [--layers N] [--inv-ref FILE]
    \\                         [--nccl-lib PATH] [--timeout S] [--mode 2a|2b|3a] [--k 2] [--prewarm 1] [--graphs 1]
    \\                         [--fast-load 0|1] [--draft-vocab 32768] [--mtp-reuse 2]
    \\                         3a: [--roce 1] [--hcas A,B] [--gid 3] [--roce-health 1] [--spin-limit N] [--tiles FILE]
    \\                         [--prompt-rows 8192] [--prompt-rows-short 4096] [--prompt-sp 1] [--experts shared|tf]
    \\                         [--pe-det slots16|0] [--cublas-lib PATH] [--cublas-ws BYTES] [--cublas-math 0]
    \\                         [--unpack-mb 384] [--bmm-probe FILE] [--side af|0] [--l2pf 1|0|lines|touch]
    \\                         [--l2pf-mb 8] [--multi-select 1] [--device-cands 1] [--prompt-prof NAME]
    \\                         [--mtp-dense 1]
    \\                         3b: 3a's flags and [--stop-eos 1] [--reuse 0|1] [--dcp 0|1|4]
    \\                         4: 3b's flags and [--dspark DIR] [--dspark-policy P] [--dspark-confidence X]
    \\                         [--dspark-depth N] [--dspark-topk N] [--dflash DIR] [--dflash-depth 7]
    \\                         [--dflash-confidence 0.3] [--copy 0|1] [--copy-min 8] [--copy-max 15]
    \\                         [--verify-rows 8] [--mtp-drafts K (= --k)]; runs "s20:dspark+c", "g:dflash", ...
    \\                         4b: 3b's flags and --parallel N [--single 1]; prompts {"prompts": [...], "groups": [...]}
    \\                         [--dspark DIR] [--dflash DIR] (+ their settings) [--draft-mode dflash|dspark|mtp]
    \\       tf-glm53-generate --mode roce-bench --rank R --world W --master IP:PORT --hcas A,B --gid N
    \\                         [--bench-rows 1,2,3,4,8,16,32] [--bench-iters 200] [--dims 6144] [--out FILE]
    \\       tf-glm53-generate --sampler-check FILE
    \\       tf-glm53-generate --mode list-variants --model DIR --world 4 --context N --tiles F0,F1,.. [--aot DIR]
    \\                         [--prompt-rows 8192] [--dcp 0|1|4] [--out FILE] [--parallel N]
    \\
;

const Args = struct {
    rank: usize = 0,
    world: usize = 1,
    master: []const u8 = "127.0.0.1:29581",
    model: []const u8 = "",
    aot: []const u8 = "",
    prompts: []const u8 = "",
    out: []const u8 = "",
    tokens: usize = 128,
    context: usize = 32768,
    window: usize = 128,
    layers: ?usize = null,
    inv_ref: ?[]const u8 = null,
    nccl_lib: []const u8 = "libnccl.so.2",
    timeout_s: f64 = 3600,
    mode: []const u8 = "2a",
    k: usize = 2,
    prewarm: bool = true,
    graphs: bool = true,
    fast_load: ?bool = null, // default: on in 2b, off in 2a
    draft_vocab: usize = 32768,
    mtp_reuse: u32 = 2,
    sampler_check: ?[]const u8 = null,
    // Phase 3a (--mode 3a | roce-bench)
    roce: bool = true,
    hcas: []const u8 = "",
    gid: u32 = 3,
    roce_health: bool = true,
    spin_limit: u32 = 20_000_000,
    tiles: ?[]const u8 = null,
    prompt_rows: usize = 8192,
    prompt_rows_short: usize = 4096,
    prompt_sp: bool = true,
    experts: []const u8 = "shared",
    pe_det: []const u8 = "slots16",
    cublas_lib: []const u8 = "libcublas.so.13",
    cublas_ws: ?usize = null,
    cublas_math: c_int = 0,
    unpack_mb: usize = 384,
    bmm_probe: ?[]const u8 = null,
    // Phase 3a speed: the served engine's decode-side defaults (image 2026-10-05: TF_GLM53_SIDE=1 -> "af",
    // TF_GLM53_L2PF=1 -> bulk, 8 MiB a site, sites "afo"), the decode-sized top-k, device candidates, prompt timers
    side: []const u8 = "af",
    l2pf: []const u8 = "1",
    l2pf_mb: f64 = 8.0,
    multi_select: bool = true,
    device_cands: bool = true,
    prompt_prof: []const u8 = "", // a prompt name prefix: its first run's prefill timed (--prompt-prof prose32k)
    // the MTP layer's prompt rows through cuBLAS (dense bf16 experts built at load, eh_proj): drafts only
    mtp_dense: bool = true,
    // Phase 3b (--mode 3b): EOS ends a run, the prompt-reuse cuts, decode context parallelism (0: auto)
    stop_eos: bool = true,
    reuse: bool = false,
    dcp: usize = 0,
    bench_rows: []const u8 = "1,2,3,4,8,16,32",
    bench_iters: usize = 200,
    dims: usize = 6144,
    // Phase 4 (--mode 4; list-variants: the drafters' launches too)
    dspark: []const u8 = "",
    dflash: []const u8 = "",
    dspark_policy: ?[]const u8 = null,
    dspark_confidence: ?[]const u8 = null,
    dspark_depth: ?[]const u8 = null,
    dspark_topk: ?[]const u8 = null,
    dflash_depth: usize = 7,
    dflash_confidence: f64 = 0.3,
    copy: ?bool = null,
    copy_min: usize = 8,
    copy_max: usize = 15,
    verify_rows: usize = 8,
    // Phase 4b (--mode 4b; list-variants: the concurrent windows too)
    parallel: usize = 1,
    single: bool = true,
    // Phase 4c (--mode 4b): the drafts of every stream (mtp | dflash | dspark; null: the engine's default mode)
    draft_mode: ?[]const u8 = null,
};

fn flag(v: []const u8) !bool {
    if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true")) return true;
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "false")) return false;
    return error.BadFlag;
}

/// TF_GLM53_L2PF: -1 off, 0 bulk, 1 lines, 2 touch (l2pf.MODES).
fn l2pfMode(v: []const u8) !c_int {
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return -1;
    if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "on") or std.mem.eql(u8, v, "bulk")) return 0;
    if (std.mem.eql(u8, v, "lines")) return 1;
    if (std.mem.eql(u8, v, "touch")) return 2;
    return error.BadL2pfMode;
}

const Prompt = struct { name: []const u8, ids: []u32 };

const Result = struct {
    tokens: std.ArrayList(u32) = .empty,
    records: std.ArrayList([glm.forward.max_world * 4]u32) = .empty,
    prefill_s: f64 = 0,
    decode_s: f64 = 0,
};

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const k = argv[i];
        if (i + 1 >= argv.len) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const v = argv[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--rank")) {
            a.rank = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--world")) {
            a.world = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--master")) {
            a.master = v;
        } else if (std.mem.eql(u8, k, "--model")) {
            a.model = v;
        } else if (std.mem.eql(u8, k, "--aot")) {
            a.aot = v;
        } else if (std.mem.eql(u8, k, "--prompts")) {
            a.prompts = v;
        } else if (std.mem.eql(u8, k, "--out")) {
            a.out = v;
        } else if (std.mem.eql(u8, k, "--tokens")) {
            a.tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--context")) {
            a.context = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--window")) {
            a.window = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--layers")) {
            a.layers = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--inv-ref")) {
            a.inv_ref = v;
        } else if (std.mem.eql(u8, k, "--nccl-lib")) {
            a.nccl_lib = v;
        } else if (std.mem.eql(u8, k, "--timeout")) {
            a.timeout_s = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, k, "--mode")) {
            a.mode = v;
        } else if (std.mem.eql(u8, k, "--k") or std.mem.eql(u8, k, "--mtp-drafts")) {
            a.k = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--prewarm")) {
            a.prewarm = try flag(v);
        } else if (std.mem.eql(u8, k, "--graphs")) {
            a.graphs = try flag(v);
        } else if (std.mem.eql(u8, k, "--fast-load")) {
            a.fast_load = try flag(v);
        } else if (std.mem.eql(u8, k, "--draft-vocab")) {
            a.draft_vocab = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--mtp-reuse")) {
            a.mtp_reuse = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--sampler-check")) {
            a.sampler_check = v;
        } else if (std.mem.eql(u8, k, "--roce")) {
            a.roce = try flag(v);
        } else if (std.mem.eql(u8, k, "--hcas")) {
            a.hcas = v;
        } else if (std.mem.eql(u8, k, "--gid")) {
            a.gid = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--roce-health")) {
            a.roce_health = try flag(v);
        } else if (std.mem.eql(u8, k, "--spin-limit")) {
            a.spin_limit = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, k, "--tiles")) {
            a.tiles = v;
        } else if (std.mem.eql(u8, k, "--prompt-rows")) {
            a.prompt_rows = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--prompt-rows-short")) {
            a.prompt_rows_short = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--prompt-sp")) {
            a.prompt_sp = try flag(v);
        } else if (std.mem.eql(u8, k, "--experts")) {
            a.experts = v;
        } else if (std.mem.eql(u8, k, "--pe-det")) {
            a.pe_det = v;
        } else if (std.mem.eql(u8, k, "--cublas-lib")) {
            a.cublas_lib = v;
        } else if (std.mem.eql(u8, k, "--cublas-ws")) {
            a.cublas_ws = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--cublas-math")) {
            a.cublas_math = try std.fmt.parseInt(c_int, v, 10);
        } else if (std.mem.eql(u8, k, "--unpack-mb")) {
            a.unpack_mb = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--bmm-probe")) {
            a.bmm_probe = v;
        } else if (std.mem.eql(u8, k, "--side")) {
            a.side = v;
        } else if (std.mem.eql(u8, k, "--l2pf")) {
            a.l2pf = v;
        } else if (std.mem.eql(u8, k, "--l2pf-mb")) {
            a.l2pf_mb = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, k, "--multi-select")) {
            a.multi_select = try flag(v);
        } else if (std.mem.eql(u8, k, "--device-cands")) {
            a.device_cands = try flag(v);
        } else if (std.mem.eql(u8, k, "--prompt-prof")) {
            a.prompt_prof = v;
        } else if (std.mem.eql(u8, k, "--mtp-dense")) {
            a.mtp_dense = try flag(v);
        } else if (std.mem.eql(u8, k, "--stop-eos")) {
            a.stop_eos = try flag(v);
        } else if (std.mem.eql(u8, k, "--reuse")) {
            a.reuse = try flag(v);
        } else if (std.mem.eql(u8, k, "--dcp")) {
            a.dcp = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--bench-rows")) {
            a.bench_rows = v;
        } else if (std.mem.eql(u8, k, "--bench-iters")) {
            a.bench_iters = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--dims")) {
            a.dims = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--dspark")) {
            a.dspark = v;
        } else if (std.mem.eql(u8, k, "--dflash")) {
            a.dflash = v;
        } else if (std.mem.eql(u8, k, "--dspark-policy")) {
            a.dspark_policy = v;
        } else if (std.mem.eql(u8, k, "--dspark-confidence")) {
            a.dspark_confidence = v;
        } else if (std.mem.eql(u8, k, "--dspark-depth")) {
            a.dspark_depth = v;
        } else if (std.mem.eql(u8, k, "--dspark-topk")) {
            a.dspark_topk = v;
        } else if (std.mem.eql(u8, k, "--dflash-depth")) {
            a.dflash_depth = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--dflash-confidence")) {
            a.dflash_confidence = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, k, "--copy")) {
            a.copy = try flag(v);
        } else if (std.mem.eql(u8, k, "--copy-min")) {
            a.copy_min = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--copy-max")) {
            a.copy_max = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--verify-rows")) {
            a.verify_rows = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--parallel")) {
            a.parallel = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--single")) {
            a.single = try flag(v);
        } else if (std.mem.eql(u8, k, "--draft-mode")) {
            a.draft_mode = v;
        } else {
            std.debug.print("unknown argument {s}\n{s}", .{ k, usage });
            return 2;
        }
    }
    if (a.sampler_check) |path| {
        const good = samplerCheck(init.gpa, init.io, path) catch |e| {
            std.debug.print("SAMPLER FAIL error {t}\n", .{e});
            return 1;
        };
        return if (good) 0 else 1;
    }
    if (std.mem.eql(u8, a.mode, "list-variants")) {
        return listVariants(init.gpa, init.io, a) catch |e| {
            std.debug.print("VARIANTS FAIL error {t}\n", .{e});
            return 1;
        };
    }
    if (std.mem.eql(u8, a.mode, "roce-bench")) {
        const okb = roceBench(init.gpa, init.io, a) catch |e| {
            std.debug.print("ROCE-BENCH FAIL rank {d}: error {t}\n", .{ a.rank, e });
            return 1;
        };
        return if (okb) 0 else 1;
    }
    if (a.model.len == 0 or a.aot.len == 0 or a.prompts.len == 0 or a.out.len == 0 or a.rank >= a.world or a.tokens == 0 or a.window == 0 or a.window > glm.forward.max_rows) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    if (std.mem.eql(u8, a.mode, "4b")) {
        const ok4b = run4b(init.gpa, init.io, a) catch |e| {
            std.debug.print("RESULT glm53-generate FAIL rank {d}: error {t}\n", .{ a.rank, e });
            return 1;
        };
        return if (ok4b) 0 else 1;
    }
    if (std.mem.eql(u8, a.mode, "3b") or std.mem.eql(u8, a.mode, "4")) {
        const ok3b = run3b(init.gpa, init.io, a) catch |e| {
            std.debug.print("RESULT glm53-generate FAIL rank {d}: error {t}\n", .{ a.rank, e });
            return 1;
        };
        return if (ok3b) 0 else 1;
    }
    if (std.mem.eql(u8, a.mode, "3a")) {
        const ok3 = run3a(init.gpa, init.io, a) catch |e| {
            std.debug.print("RESULT glm53-generate FAIL rank {d}: error {t}\n", .{ a.rank, e });
            return 1;
        };
        return if (ok3) 0 else 1;
    }
    const two_b = std.mem.eql(u8, a.mode, "2b");
    if (!two_b and !std.mem.eql(u8, a.mode, "2a")) {
        std.debug.print("--mode 2a, 2b, 3a, 3b or roce-bench\n{s}", .{usage});
        return 2;
    }
    if (two_b) {
        const ok2 = run2b(init.gpa, init.io, a) catch |e| {
            std.debug.print("RESULT glm53-generate FAIL rank {d}: error {t}\n", .{ a.rank, e });
            return 1;
        };
        return if (ok2) 0 else 1;
    }
    const ok = run(init.gpa, init.io, a) catch |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: error {t}\n", .{ a.rank, e });
        return 1;
    };
    return if (ok) 0 else 1;
}

fn now(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn since(io: std.Io, t0: i128) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e9;
}

fn readPrompts(gpa: std.mem.Allocator, io: std.Io, path: []const u8, text_out: *[]u8) ![]Prompt {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    text_out.* = text;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const list = (parsed.value.object.get("prompts") orelse return error.BadPrompts).array.items;
    const out = try gpa.alloc(Prompt, list.len);
    for (list, out) |pv, *p| {
        const o = pv.object;
        const ids = (o.get("ids") orelse return error.BadPrompts).array.items;
        p.name = try gpa.dupe(u8, (o.get("name") orelse return error.BadPrompts).string);
        p.ids = try gpa.alloc(u32, ids.len);
        for (ids, p.ids) |x, *y| y.* = switch (x) {
            .integer => |n| std.math.cast(u32, n) orelse return error.BadPrompts,
            else => return error.BadPrompts,
        };
        if (p.ids.len == 0) return error.BadPrompts;
    }
    return out;
}

fn run(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    var prompts_text: []u8 = &.{};
    const prompts = try readPrompts(gpa, io, a.prompts, &prompts_text);
    defer {
        for (prompts) |p| {
            gpa.free(p.name);
            gpa.free(p.ids);
        }
        gpa.free(prompts);
        gpa.free(prompts_text);
    }
    std.debug.print("rank {d} of {d}: {d} prompts x {d} picks, context {d}, windows of {d} rows\n", .{ a.rank, a.world, prompts.len, a.tokens, a.context, a.window });

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    std.debug.print("device: {s} sm_{d}\n", .{ try ctx.name(&name_buf), try ctx.capability() });

    // -- the world: TCP rendezvous (rank 0's NCCL id), then ncclCommInitRank, before the load (as comm.NCCL) ------
    const master = try cuda.rendezvous.parseMaster(a.master);
    var lib = try cuda.nccl.Library.openPath(a.nccl_lib);
    defer lib.close();
    const nccl_version = try lib.version();
    std.debug.print("nccl: {s} version {d}\n", .{ a.nccl_lib, nccl_version });
    var rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, a.rank, a.world, a.timeout_s);
    defer rdv.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (a.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&lib);
    try rdv.broadcast(std.mem.asBytes(&uid));
    var comm = try cuda.nccl.Communicator.init(&lib, uid, a.rank, a.world);
    defer comm.deinit();
    std.debug.print("rank {d}: NCCL communicator up (world {d})\n", .{ a.rank, a.world });

    const cfg = try glm.Config.load(gpa, io, a.model);
    const n_layers = a.layers orelse cfg.layers;
    var settings = std.hash.Wyhash.init(0x6a1b);
    const shape = [_]usize{ a.world, n_layers, a.context, a.tokens, a.window };
    settings.update(std.mem.asBytes(&shape));
    settings.update(prompts_text);
    try rdv.sync("settings", settings.final(), a.timeout_s);

    // -- weights: every layer, the head; layer by layer with the page cache dropped -----------------------------
    const t_load = now(io);
    var w = try glm.weights.load(gpa, io, &driver, a.model, cfg, .{ .layers = n_layers, .rank = a.rank, .world = a.world, .head = true, .log_every = 10, .fast = a.fast_load orelse false });
    defer w.deinit();
    const load_s = since(io, t_load);
    std.debug.print("rank {d}: {d} layers + head loaded in {d:.0} s ({d:.1} GiB of weights; vocabulary rows {d}..{d})\n", .{ a.rank, n_layers, load_s, @as(f64, @floatFromInt(w.device_bytes)) / (1 << 30), w.vocab_off, w.vocab_off + w.vocab_part });
    try rdv.sync("loaded", 0, a.timeout_s);

    var k = try glm.kernels.Kernels.load(gpa, io, &ctx, a.aot);
    defer k.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();

    // -- fused.inv_freq: torch's bytes, computed the way torch computes them; the reference's sidecar checks them --
    const half = cfg.rope / 2;
    var inv_buf = try cuda.DeviceBuffer.alloc(&driver, half * 4);
    defer inv_buf.free();
    try k.invFreq(stream, inv_buf.ptr, cfg.rope, cfg.rope_theta);
    try stream.synchronize();
    var inv_status: []const u8 = "device (no reference to check)";
    if (a.inv_ref) |path| {
        const ref = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
        defer gpa.free(ref);
        const mine = try gpa.alloc(u8, half * 4);
        defer gpa.free(mine);
        try inv_buf.download(0, mine);
        if (ref.len != mine.len) return error.BadInvReference;
        if (std.mem.eql(u8, ref, mine)) {
            inv_status = "device == reference";
        } else {
            var bad: usize = 0;
            for (0..half) |j| {
                if (!std.mem.eql(u8, ref[4 * j ..][0..4], mine[4 * j ..][0..4])) bad += 1;
            }
            std.debug.print("WARN inv_freq: {d} of {d} entries differ from torch's (the reference's bytes are used)\n", .{ bad, half });
            try inv_buf.upload(0, ref);
            inv_status = "DIFFERS: reference bytes used";
        }
    }
    std.debug.print("inv_freq: {s}\n", .{inv_status});

    const fwd: glm.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv_buf.ptr, .comm = if (a.world > 1) &comm else null };
    const topk = cfg.index_topk;
    const score_cols = glm.triton.bucket(a.context, topk) orelse @max(a.context, 64);
    var b = try glm.Buffers.init(gpa, &driver, &w, a.window, score_cols);
    defer b.deinit();
    var st = try glm.State.init(gpa, &driver, &w, a.context);
    defer st.deinit();
    const host = try gpa.alloc(f32, w.vocab_part);
    defer gpa.free(host);

    const results = try gpa.alloc(Result, prompts.len);
    defer {
        for (results) |*r| {
            r.tokens.deinit(gpa);
            r.records.deinit(gpa);
        }
        gpa.free(results);
    }
    for (results) |*r| r.* = .{};
    var all_tokens = std.hash.Wyhash.init(0);
    var decode_steps: usize = 0;
    var decode_s: f64 = 0;
    for (prompts, results) |p, *res| {
        const P = p.ids.len;
        if (P + a.tokens > a.context) {
            std.log.err("{s}: {d} prompt tokens + {d} picks exceed the context {d}", .{ p.name, P, a.tokens, a.context });
            return error.ContextTooSmall;
        }
        try st.reset();
        const t_pf = now(io);
        var pick: glm.forward.Pick = undefined;
        var s0: usize = 0;
        while (s0 < P) {
            const e = @min(s0 + a.window, P);
            const R = e - s0;
            try fwd.embed(&b, p.ids[s0..e]);
            try fwd.setPos(&st, s0);
            const T = glm.triton.bucket(e, topk);
            for (0..n_layers) |li| try fwd.layer(&b, &st, li, R, T);
            if (e == P) pick = try fwd.head(&b, R, host);
            s0 = e;
        }
        res.prefill_s = since(io, t_pf);
        try keep(gpa, res, pick);
        const t_dec = now(io);
        var pos = P;
        while (res.tokens.items.len < a.tokens) : (pos += 1) {
            const last = [1]u32{res.tokens.items[res.tokens.items.len - 1]};
            try fwd.embed(&b, &last);
            try fwd.setPos(&st, pos);
            const T = glm.triton.bucket(pos + 1, topk);
            for (0..n_layers) |li| try fwd.layer(&b, &st, li, 1, T);
            pick = try fwd.head(&b, 1, host);
            try keep(gpa, res, pick);
        }
        res.decode_s = since(io, t_dec);
        decode_steps += a.tokens - 1;
        decode_s += res.decode_s;
        all_tokens.update(std.mem.sliceAsBytes(res.tokens.items));
        std.debug.print("rank {d} {s}: prompt {d} tokens in {d:.1} s, {d} picks in {d:.1} s; first picks {any}\n", .{ a.rank, p.name, P, res.prefill_s, a.tokens, res.decode_s, res.tokens.items[0..@min(8, res.tokens.items.len)] });
    }

    try writeOut(gpa, io, a, &w, n_layers, nccl_version, inv_status, load_s, prompts, results);
    // every rank picks from the same gathered records: the token streams must agree
    const agree = rdv.sync("done", all_tokens.final(), a.timeout_s);
    const tps = if (decode_s > 0) @as(f64, @floatFromInt(decode_steps)) / decode_s else 0;
    if (agree) {
        std.debug.print("RESULT glm53-generate OK rank {d}: {d} prompts x {d} picks, eager decode {d:.2} tok/s, inv_freq {s}\n", .{ a.rank, prompts.len, a.tokens, tps, inv_status });
        return true;
    } else |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: the ranks' token streams differ ({t})\n", .{ a.rank, e });
        return false;
    }
}

fn keep(gpa: std.mem.Allocator, res: *Result, p: glm.forward.Pick) !void {
    try res.tokens.append(gpa, p.token);
    var bits: [glm.forward.max_world * 4]u32 = @splat(0);
    for (0..p.world * 4) |j| bits[j] = @bitCast(p.gathered[j]);
    try res.records.append(gpa, bits);
}

fn writeOut(gpa: std.mem.Allocator, io: std.Io, a: Args, w: *const glm.weights.Weights, n_layers: usize, nccl_version: c_int, inv_status: []const u8, load_s: f64, prompts: []const Prompt, results: []const Result) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const head = try std.fmt.allocPrint(gpa, "{{\"tool\": \"tf-glm53-generate\", \"rank\": {d}, \"world\": {d}, \"layers\": {d}, \"context\": {d}, \"window\": {d}, \"tokens\": {d}, \"vocab_off\": {d}, \"vocab_part\": {d}, \"nccl_version\": {d}, \"inv_freq\": \"{s}\", \"load_s\": {d:.1}, \"prompts\": [", .{ a.rank, a.world, n_layers, a.context, a.window, a.tokens, w.vocab_off, w.vocab_part, nccl_version, inv_status, load_s });
    defer gpa.free(head);
    try out.appendSlice(gpa, head);
    for (prompts, results, 0..) |p, r, pi| {
        if (pi > 0) try out.appendSlice(gpa, ", ");
        const ph = try std.fmt.allocPrint(gpa, "\n {{\"name\": \"{s}\", \"prompt_len\": {d}, \"prefill_s\": {d:.3}, \"decode_s\": {d:.3}, \"tokens\": [", .{ p.name, p.ids.len, r.prefill_s, r.decode_s });
        defer gpa.free(ph);
        try out.appendSlice(gpa, ph);
        for (r.tokens.items, 0..) |t, j| {
            var buf: [16]u8 = undefined;
            try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}{d}", .{ if (j > 0) ", " else "", t }));
        }
        try out.appendSlice(gpa, "],\n  \"records\": [");
        for (r.records.items, 0..) |rec, j| {
            try out.appendSlice(gpa, if (j > 0) ",\n   \"" else "\n   \"");
            for (0..a.world * 4) |q| {
                var buf: [16]u8 = undefined;
                try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}{x:0>8}", .{ if (q > 0) "," else "", rec[q] }));
            }
            try out.appendSlice(gpa, "\"");
        }
        try out.appendSlice(gpa, "]}");
    }
    try out.appendSlice(gpa, "\n]}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = out.items });
    std.debug.print("rank {d}: wrote {s}\n", .{ a.rank, a.out });
}

// ------------------------------------------------------------------------------------------------- Phase 2b ---

/// A run of a prompt: greedy or sampled (top_k; 0 = top_p alone), with `k` drafts a round (0: serial).
const RunSpec = struct { greedy: bool, top_k: usize, k: usize, label: []const u8, mode: glm.runner.Mode = .mtp, copies: bool = false };

const Prompt2b = struct {
    name: []const u8,
    ids: []u32,
    tokens: usize,
    seed: u64,
    seed_from_prompt: bool,
    runs: []RunSpec,
    // Phase 3b: this prompt's own sampling over the file's, and its end ids (null: the model's)
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    min_p: ?f64 = null,
    eos: ?[]u32 = null,
};

fn tf(b: bool) []const u8 {
    return if (b) "true" else "false";
}

const Sampling2b = struct { temperature: f64 = 1.0, top_p: f64 = 0.95, min_p: f64 = 0.0 };

/// "g:K" / "s<top_k>:K" (K MTP drafts, 0 serial); Phase 4: "g:dspark" / "s20:dflash" (a drafter's rounds) and a
/// "+c" suffix (copy drafts ahead of the round's drafter).
fn parseRun(gpa: std.mem.Allocator, text: []const u8) !RunSpec {
    const colon = std.mem.indexOfScalar(u8, text, ':') orelse return error.BadRun;
    const pick = text[0..colon];
    var rest = text[colon + 1 ..];
    const with_copies = std.mem.endsWith(u8, rest, "+c");
    if (with_copies) rest = rest[0 .. rest.len - 2];
    var dmode: glm.runner.Mode = .mtp;
    var k: usize = 0;
    if (std.mem.eql(u8, rest, "dspark")) {
        dmode = .dspark;
    } else if (std.mem.eql(u8, rest, "dflash")) {
        dmode = .dflash;
    } else {
        k = try std.fmt.parseInt(usize, rest, 10);
    }
    if (with_copies and dmode == .mtp and k == 0) return error.BadRun; // the serial reference never copies
    const label = try gpa.dupe(u8, text);
    if (std.mem.eql(u8, pick, "g")) return .{ .greedy = true, .top_k = 0, .k = k, .label = label, .mode = dmode, .copies = with_copies };
    if (pick.len >= 2 and pick[0] == 's') return .{ .greedy = false, .top_k = try std.fmt.parseInt(usize, pick[1..], 10), .k = k, .label = label, .mode = dmode, .copies = with_copies };
    return error.BadRun;
}

fn jsonFloat(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => error.BadPrompts,
    };
}

fn jsonU64(v: std.json.Value) !u64 {
    return switch (v) {
        .integer => |n| std.math.cast(u64, n) orelse error.BadPrompts,
        .number_string => |t| std.fmt.parseInt(u64, t, 10),
        .string => |t| std.fmt.parseInt(u64, t, 10),
        else => error.BadPrompts,
    };
}

fn readPrompts2b(gpa: std.mem.Allocator, io: std.Io, path: []const u8, default_tokens: usize, text_out: *[]u8, smp_out: *Sampling2b) ![]Prompt2b {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    text_out.* = text;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    if (root.get("sampling")) |sv| {
        const so = sv.object;
        if (so.get("temperature")) |x| smp_out.temperature = try jsonFloat(x);
        if (so.get("top_p")) |x| smp_out.top_p = try jsonFloat(x);
        if (so.get("min_p")) |x| smp_out.min_p = try jsonFloat(x);
    }
    const list = (root.get("prompts") orelse return error.BadPrompts).array.items;
    const out = try gpa.alloc(Prompt2b, list.len);
    for (list, out) |pv, *p| {
        const o = pv.object;
        const ids = (o.get("ids") orelse return error.BadPrompts).array.items;
        p.name = try gpa.dupe(u8, (o.get("name") orelse return error.BadPrompts).string);
        p.ids = try gpa.alloc(u32, ids.len);
        for (ids, p.ids) |x, *y| y.* = switch (x) {
            .integer => |n| std.math.cast(u32, n) orelse return error.BadPrompts,
            else => return error.BadPrompts,
        };
        if (p.ids.len == 0) return error.BadPrompts;
        p.tokens = if (o.get("tokens")) |t| @intCast(try jsonU64(t)) else default_tokens;
        p.seed_from_prompt = if (o.get("seed_from_prompt")) |b| b == .bool and b.bool else false;
        p.seed = if (o.get("seed")) |sd| try jsonU64(sd) else glm.sampling.seedFor(p.ids, 0);
        p.temperature = if (o.get("temperature")) |x| try jsonFloat(x) else null;
        p.top_p = if (o.get("top_p")) |x| try jsonFloat(x) else null;
        p.min_p = if (o.get("min_p")) |x| try jsonFloat(x) else null;
        p.eos = null;
        if (o.get("eos")) |ev| {
            const items = ev.array.items;
            const e = try gpa.alloc(u32, items.len);
            for (items, e) |x, *y| y.* = @intCast(try jsonU64(x));
            p.eos = e;
        }
        const runs = if (o.get("runs")) |r| r.array.items else &[_]std.json.Value{};
        p.runs = try gpa.alloc(RunSpec, runs.len);
        for (runs, p.runs) |rv, *rs| rs.* = try parseRun(gpa, rv.string);
    }
    return out;
}

const Done = struct {
    prompt: usize,
    run: usize,
    tokens: []u32,
    st: glm.runner.Stats,
    seed_ok: bool,
};

fn run2b(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    var prompts_text: []u8 = &.{};
    var sset: Sampling2b = .{};
    const prompts = try readPrompts2b(gpa, io, a.prompts, a.tokens, &prompts_text, &sset);
    defer gpa.free(prompts_text);
    defer {
        for (prompts) |p| {
            gpa.free(p.name);
            gpa.free(p.ids);
            for (p.runs) |rs| gpa.free(rs.label);
            gpa.free(p.runs);
        }
        gpa.free(prompts);
    }
    var n_runs: usize = 0;
    for (prompts) |p| n_runs += p.runs.len;
    std.debug.print("rank {d} of {d}: phase 2b, {d} prompts, {d} runs, context {d}, prompt chunks of up to {d} rows, MTP drafts {d}, graphs {s}, prewarm {s}\n", .{ a.rank, a.world, prompts.len, n_runs, a.context, a.window, a.k, tf(a.graphs), tf(a.prewarm) });

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    std.debug.print("device: {s} sm_{d}\n", .{ try ctx.name(&name_buf), try ctx.capability() });

    const master = try cuda.rendezvous.parseMaster(a.master);
    var lib = try cuda.nccl.Library.openPath(a.nccl_lib);
    defer lib.close();
    const nccl_version = try lib.version();
    std.debug.print("nccl: {s} version {d}\n", .{ a.nccl_lib, nccl_version });
    var rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, a.rank, a.world, a.timeout_s);
    defer rdv.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (a.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&lib);
    try rdv.broadcast(std.mem.asBytes(&uid));
    var comm = try cuda.nccl.Communicator.init(&lib, uid, a.rank, a.world);
    defer comm.deinit();
    std.debug.print("rank {d}: NCCL communicator up (world {d})\n", .{ a.rank, a.world });

    const cfg = try glm.Config.load(gpa, io, a.model);
    const n_layers = a.layers orelse cfg.layers;
    var settings = std.hash.Wyhash.init(0x6a2b);
    const shape = [_]usize{ a.world, n_layers, a.context, a.tokens, a.window, a.k, a.draft_vocab, a.mtp_reuse, @intFromBool(a.graphs), @intFromBool(a.prewarm) };
    settings.update(std.mem.asBytes(&shape));
    settings.update(prompts_text);
    try rdv.sync("settings", settings.final(), a.timeout_s);

    // -- weights: every layer, the head, the MTP layer and the 4-bit draft head; O_DIRECT reads a layer ahead ------
    const fast = a.fast_load orelse true;
    const t_load = now(io);
    var w = try glm.weights.load(gpa, io, &driver, a.model, cfg, .{ .layers = n_layers, .rank = a.rank, .world = a.world, .head = true, .log_every = 10, .fast = fast, .mtp = a.k > 0, .draft_vocab = if (a.k > 0) a.draft_vocab else 0 });
    defer w.deinit();
    const load_s = since(io, t_load);
    const read_gib = @as(f64, @floatFromInt(w.read_bytes)) / (1 << 30);
    std.debug.print("rank {d}: {d} layers{s} + head loaded in {d:.0} s ({d:.1} GiB of weights; {s}: {d:.1} GiB read, {d} files direct, {d} fallback reads; draft head {d} rows)\n", .{ a.rank, n_layers, if (w.mtp != null) " + MTP layer" else "", load_s, @as(f64, @floatFromInt(w.device_bytes)) / (1 << 30), if (fast) "O_DIRECT + io_uring" else "mapped reads", read_gib, w.direct_files, w.read_fallbacks, if (w.draft) |dh| dh.n else 0 });
    try rdv.sync("loaded", 0, a.timeout_s);

    var k = try glm.kernels.Kernels.load(gpa, io, &ctx, a.aot);
    defer k.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();

    const half = cfg.rope / 2;
    var inv_buf = try cuda.DeviceBuffer.alloc(&driver, half * 4);
    defer inv_buf.free();
    try k.invFreq(stream, inv_buf.ptr, cfg.rope, cfg.rope_theta);
    try stream.synchronize();
    var inv_status: []const u8 = "device (no reference to check)";
    if (a.inv_ref) |path| {
        const ref = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
        defer gpa.free(ref);
        const mine = try gpa.alloc(u8, half * 4);
        defer gpa.free(mine);
        try inv_buf.download(0, mine);
        if (ref.len != mine.len) return error.BadInvReference;
        if (std.mem.eql(u8, ref, mine)) {
            inv_status = "device == reference";
        } else {
            std.debug.print("WARN inv_freq differs from torch's (the reference's bytes are used)\n", .{});
            try inv_buf.upload(0, ref);
            inv_status = "DIFFERS: reference bytes used";
        }
    }
    std.debug.print("inv_freq: {s}\n", .{inv_status});

    const fwd: glm.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv_buf.ptr, .comm = if (a.world > 1) &comm else null };
    const capacity = a.context + a.k + 1; // Glm53Engine: Runner(limit + k + 1)
    var r = try glm.Runner.init(gpa, io, &driver, fwd, capacity, .{ .k = a.k, .graphs = a.graphs, .window = a.window, .reuse = a.mtp_reuse });
    defer r.deinit();
    var prewarm_s: f64 = 0;
    var prewarm_graphs: usize = 0;
    if (a.prewarm) {
        const tp = now(io);
        prewarm_graphs = try r.prewarm();
        prewarm_s = since(io, tp);
        std.debug.print("rank {d}: {d} decode graphs captured in {d:.1} s\n", .{ a.rank, prewarm_graphs, prewarm_s });
    }
    try rdv.sync("warm", prewarm_graphs, a.timeout_s);

    var done: std.ArrayList(Done) = .empty;
    defer {
        for (done.items) |d| gpa.free(d.tokens);
        done.deinit(gpa);
    }
    var toks: std.ArrayList(u32) = .empty;
    defer toks.deinit(gpa);
    var all_tokens = std.hash.Wyhash.init(0);
    var seeds_ok = true;
    for (prompts, 0..) |p, pi| {
        const seed_ok = !p.seed_from_prompt or glm.sampling.seedFor(p.ids, 0) == p.seed;
        if (!seed_ok) {
            std.debug.print("WARN {s}: seed_for(prompt) {d} here, {d} in the prompts file\n", .{ p.name, glm.sampling.seedFor(p.ids, 0), p.seed });
            seeds_ok = false;
        }
        for (p.runs, 0..) |rs, ri| {
            const s: ?glm.sampling.Sampling = if (rs.greedy) null else .{ .seed = p.seed, .temperature = sset.temperature, .top_k = rs.top_k, .top_p = sset.top_p, .min_p = sset.min_p };
            if (rs.k > a.k) return error.TooManyDrafts;
            if (rs.mode != .mtp or rs.copies) return error.DraftersNeedMode4;
            const st = try r.generate(p.ids, p.tokens, s, rs.k, &toks);
            all_tokens.update(std.mem.sliceAsBytes(toks.items));
            const n = toks.items.len;
            try done.append(gpa, .{ .prompt = pi, .run = ri, .tokens = try gpa.dupe(u32, toks.items), .st = st, .seed_ok = seed_ok });
            std.debug.print("rank {d} {s} {s}: prompt {d} in {d:.1} s; {d} tokens, decode {d:.2} tok/s, {d:.3} tokens/round ({d} rounds), accept {d:.3}, capture {d:.1} s; first {any}\n", .{ a.rank, p.name, rs.label, p.ids.len, st.prefill_s, n, st.tokS(n), st.perRound(n), st.rounds, if (st.drafted > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0, st.capture_s, toks.items[0..@min(8, n)] });
        }
    }

    try writeOut2b(gpa, io, a, &w, n_layers, capacity, nccl_version, inv_status, load_s, read_gib, prewarm_graphs, prewarm_s, prompts, done.items);
    const agree = rdv.sync("done", all_tokens.final(), a.timeout_s);
    if (agree) {
        std.debug.print("RESULT glm53-generate {s} rank {d}: phase 2b, {d} runs, load {d:.0} s, inv_freq {s}{s}\n", .{ if (seeds_ok) "OK" else "FAIL", a.rank, done.items.len, load_s, inv_status, if (seeds_ok) "" else ", seed_for differs from Python's" });
        return seeds_ok;
    } else |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: the ranks' token streams differ ({t})\n", .{ a.rank, e });
        return false;
    }
}

fn writeOut2b(gpa: std.mem.Allocator, io: std.Io, a: Args, w: *const glm.weights.Weights, n_layers: usize, capacity: usize, nccl_version: c_int, inv_status: []const u8, load_s: f64, read_gib: f64, prewarm_graphs: usize, prewarm_s: f64, prompts: []const Prompt2b, done: []const Done) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const head = try std.fmt.allocPrint(gpa, "{{\"tool\": \"tf-glm53-generate\", \"mode\": \"2b\", \"rank\": {d}, \"world\": {d}, \"layers\": {d}, \"context\": {d}, \"capacity\": {d}, \"window\": {d}, \"k\": {d}, \"draft_vocab\": {d}, \"mtp_reuse\": {d}, \"graphs\": {s}, \"vocab_off\": {d}, \"vocab_part\": {d}, \"nccl_version\": {d}, \"inv_freq\": \"{s}\", \"load_s\": {d:.1}, \"read_gib\": {d:.2}, \"direct_files\": {d}, \"read_fallbacks\": {d}, \"prewarm_graphs\": {d}, \"prewarm_s\": {d:.1}, \"runs\": [", .{ a.rank, a.world, n_layers, a.context, capacity, a.window, a.k, a.draft_vocab, a.mtp_reuse, tf(a.graphs), w.vocab_off, w.vocab_part, nccl_version, inv_status, load_s, read_gib, w.direct_files, w.read_fallbacks, prewarm_graphs, prewarm_s });
    defer gpa.free(head);
    try out.appendSlice(gpa, head);
    for (done, 0..) |d, i| {
        const p = prompts[d.prompt];
        const rs = p.runs[d.run];
        const n = d.tokens.len;
        const acc: f64 = if (d.st.drafted > 0) @as(f64, @floatFromInt(d.st.accepted)) / @as(f64, @floatFromInt(d.st.drafted)) else 0;
        const ph = try std.fmt.allocPrint(gpa, "{s}\n {{\"name\": \"{s}\", \"run\": \"{s}\", \"prompt_len\": {d}, \"seed\": \"{d}\", \"seed_ok\": {s}, \"k\": {d}, \"greedy\": {s}, \"top_k\": {d}, \"prefill_s\": {d:.3}, \"decode_s\": {d:.3}, \"tok_s\": {d:.3}, \"rounds\": {d}, \"tokens_per_round\": {d:.4}, \"accept\": {d:.4}, \"capture_s\": {d:.2}, \"graphs\": {d}, \"tokens\": [", .{ if (i > 0) "," else "", p.name, rs.label, p.ids.len, p.seed, tf(d.seed_ok), rs.k, tf(rs.greedy), rs.top_k, d.st.prefill_s, d.st.decode_s, d.st.tokS(n), d.st.rounds, d.st.perRound(n), acc, d.st.capture_s, d.st.graphs });
        defer gpa.free(ph);
        try out.appendSlice(gpa, ph);
        for (d.tokens, 0..) |t, j| {
            var buf: [16]u8 = undefined;
            try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}{d}", .{ if (j > 0) ", " else "", t }));
        }
        try out.appendSlice(gpa, "]}");
    }
    try out.appendSlice(gpa, "\n]}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = out.items });
    std.debug.print("rank {d}: wrote {s}\n", .{ a.rank, a.out });
}

// ------------------------------------------------------------------------------------------------- Phase 3a ---

/// The RDMA devices of a comma list (TF_GLM53_ROCE_HCA / NCCL_IB_HCA syntax: "=" and ":port" dropped), at most two.
fn parseHcas(text: []const u8, out: *[2][]const u8) [][]const u8 {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |raw| {
        if (n == out.len) break;
        var name = std.mem.trim(u8, raw, " =^");
        if (std.mem.indexOfScalar(u8, name, ':')) |c| name = name[0..c];
        if (name.len == 0) continue;
        out[n] = name;
        n += 1;
    }
    return out[0..n];
}

/// --mode list-variants: the Triton launches the engine can make at --world ranks, --context tokens, windows up to
/// max(--prompt-rows, --window) rows, DCP 1 and --world (--dcp 0) or only --dcp N, the EXL3 linear shapes of the tile
/// tables in --tiles (every rank's) - written to --out as JSON; with --aot, checked against that set (exit 3: some
/// launch has no variant there, the first 40 listed).
fn listVariants(gpa: std.mem.Allocator, io: std.Io, a: Args) !u8 {
    if (a.model.len == 0 or a.world == 0) return error.Usage;
    const cov = glm.coverage;
    const cfg = try glm.Config.load(gpa, io, a.model);
    if (cfg.heads % a.world != 0 or cfg.vocab % a.world != 0) return error.UnevenSplit;
    var gemms: std.ArrayList([2]usize) = .empty;
    defer gemms.deinit(gpa);
    var widths: std.ArrayList(usize) = .empty;
    defer widths.deinit(gpa);
    var kva_n: usize = 0;
    var tables: usize = 0;
    if (a.tiles) |list| {
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |path| {
            if (path.len == 0) continue;
            const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
            defer gpa.free(text);
            var shapes: std.ArrayList([2]usize) = .empty;
            defer shapes.deinit(gpa);
            try cov.tileShapes(gpa, text, &shapes);
            if (shapes.items.len < 2) return error.BadTileTable;
            kva_n = @max(kva_n, shapes.items[1][1]); // rows: layer 0's q_a, kv_a, ...
            for (shapes.items) |kn| try cov.addUnique(gpa, &gemms, kn);
            try cov.swigluWidths(shapes.items, cfg.hidden, &widths, gpa);
            tables += 1;
        }
    }
    if (tables == 0) return error.NoTileTables; // the prompt GEMM's shapes come from the reference's tables
    // the config's own widths too (dense MLP, shared expert) when they split evenly
    for ([_]usize{ cfg.dense_width, cfg.expert_width * cfg.shared_experts }) |w| {
        if (w % a.world == 0 and std.mem.indexOfScalar(usize, widths.items, w / a.world) == null) try widths.append(gpa, w / a.world);
    }
    var dbuf: [2]usize = .{ 1, a.world };
    const dcps: []const usize = switch (a.dcp) {
        0 => if (a.world > 1) dbuf[0..2] else dbuf[0..1],
        1 => dbuf[0..1],
        else => blk: {
            dbuf[0] = a.dcp;
            break :blk dbuf[0..1];
        },
    };
    // Phase 4: the drafters' tap passes and block passes (their configs; a DFlash2's selector rank is a CUDA launch)
    var dspecs: [2]glm.draft.Spec = undefined;
    var n_dspecs: usize = 0;
    if (a.dspark.len > 0) {
        const path = try std.fs.path.join(gpa, &.{ a.dspark, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        const dc = try glm.draft_host.DSparkConfig.parse(gpa, text);
        try dc.check(a.world, cfg.hidden, cfg.vocab, cfg.layers);
        const ds = try glm.draft_host.DSparkSettings.parse(dc.block, .{ .top_k = a.dspark_topk });
        dspecs[n_dspecs] = glm.draft.Spec.ofDSpark(&dc, &ds, a.world, cfg.vocab / a.world, 0);
        n_dspecs += 1;
    }
    if (a.dflash.len > 0) {
        const path = try std.fs.path.join(gpa, &.{ a.dflash, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        const fc = try glm.draft_host.DFlashConfig.parse(gpa, text);
        try fc.check(a.world, cfg.hidden, cfg.layers);
        dspecs[n_dspecs] = glm.draft.Spec.ofDFlash(&fc, a.world, cfg.vocab / a.world, 0, 0);
        n_dspecs += 1;
    }
    // verify windows up to the copy width and a DSpark block + 1 rows (all within the window rows enumerated)
    // Phase 4b: --parallel N's windows of up to N x (k + 1) rows with per-row positions and cache bases (ROWS);
    // Phase 4c: N x max(k + 1, the drafters' rows) with drafters, and their concurrent passes
    const fr: usize = if (a.parallel > 1) try glm.multi.drafterRows(a.parallel, dspecs[0..n_dspecs]) else 0;
    const rows_windows: usize = if (a.parallel > 1) try glm.multi.windowRowsWith(a.parallel, a.k, fr) else 0;
    const sp: cov.Space = .{ .cfg = cfg, .world = a.world, .heads = cfg.heads / a.world, .vocab_part = cfg.vocab / a.world, .kva_n = kva_n, .max_rows = @max(@max(a.prompt_rows, a.window), @max(a.k + 1, 32)), .capacity = a.context, .dcps = dcps, .widths = widths.items, .gemms = gemms.items, .mtp = cfg.mtp_layers > 0, .draft = a.draft_vocab > 0, .drafters = dspecs[0..n_dspecs], .rows_windows = rows_windows, .streams = a.parallel };
    var p = cuda.aot.Probe.init(gpa);
    defer p.deinit();
    try cov.enumerate(&p, sp);
    var cat: ?cuda.aot.Catalog = null;
    defer if (cat) |*c| c.deinit();
    if (a.aot.len > 0) cat = try cuda.aot.loadCatalog(gpa, io, a.aot);
    const specs = if (cat) |c| c.value.kernels else null;
    const doc = try p.json(gpa, specs);
    defer gpa.free(doc);
    if (a.out.len > 0) try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = doc });
    std.debug.print("variants: {d} launches enumerated, {d} distinct needs (rows 1..{d}, context {d}, dcp {any}, {d} tile tables, {d} GEMM shapes, SwiGLU widths {any}, concurrent windows up to {d} rows)\n", .{ p.calls, p.needs.count(), sp.max_rows, sp.capacity, dcps, tables, gemms.items.len, widths.items, rows_windows });
    const ks = specs orelse return 0;
    const miss = try p.missing(gpa, ks);
    defer gpa.free(miss);
    const needs = p.needs.values();
    for (miss, 0..) |i, j| {
        if (j >= 40) break;
        const n = needs[i];
        std.debug.print("  missing: {s} (warps {?d}, stages {?d})", .{ n.function, n.opts.num_warps, n.opts.num_stages });
        for (n.consts) |c| if (c.int) |x| std.debug.print(" {s}={d}", .{ c.name, x });
        for (n.args) |x| switch (x.value) {
            .i32 => |v| std.debug.print(" {s}:{d}", .{ x.name, v }),
            else => {},
        };
        std.debug.print("\n", .{});
    }
    std.debug.print("VARIANTS {s} {d} needs, {d} missing (aot {s})\n", .{ if (miss.len == 0) "PASS" else "FAIL", needs.len, miss.len, a.aot });
    return if (miss.len == 0) 0 else 3;
}

/// --mode roce-bench: the RoCE one-shot runtime alone on the TP world (no model): setup, the bit check against NCCL
/// (roce.RoceReduce._check: every rank's sum equal and equal to the rank-order sum), then microseconds a reduction
/// of the decode windows' shapes ([rows, --dims] fp32) replayed from a CUDA graph, eager, and NCCL's all-gather of the
/// same partials (the fallback's exchange). Prints `ROCE-BENCH rows ...` lines and, last, `ROCE-BENCH PASS|FAIL ...`
/// (PASS: bit check good and every window of up to 3 rows at most 65 us).
fn roceBench(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const master = try cuda.rendezvous.parseMaster(a.master);
    var lib = try cuda.nccl.Library.openPath(a.nccl_lib);
    defer lib.close();
    var rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, a.rank, a.world, a.timeout_s);
    defer rdv.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (a.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&lib);
    try rdv.broadcast(std.mem.asBytes(&uid));
    var comm = try cuda.nccl.Communicator.init(&lib, uid, a.rank, a.world);
    defer comm.deinit();
    var hb: [2][]const u8 = undefined;
    const hcas = parseHcas(a.hcas, &hb);
    std.debug.print("rank {d}: RoCE on {d} rails ({s}), GID index {d}\n", .{ a.rank, hcas.len, a.hcas, a.gid });
    const r = try glm.roce.Roce.start(gpa, &driver, &rdv, a.rank, a.world, .{ .hcas = hcas, .gid_index = a.gid, .spin_limit = a.spin_limit });
    defer r.deinit(gpa);
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const good = try r.verify(gpa, &comm, stream, a.dims, glm.forward.decode_rows);
    std.debug.print("rank {d}: RoCE bit check {s}\n", .{ a.rank, if (good) "good (bit-equal ranks, rank-order sum)" else "FAILED" });
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try out.appendSlice(gpa, "{\"tool\": \"tf-glm53-generate\", \"mode\": \"roce-bench\", \"rows\": [");
    var worst: f64 = 0;
    var it = std.mem.tokenizeScalar(u8, a.bench_rows, ',');
    var first = true;
    while (it.next()) |txt| {
        const rows = try std.fmt.parseInt(usize, txt, 10);
        try rdv.sync("bench", rows, a.timeout_s);
        const row = try glm.roce.bench(r, &comm, stream, a.dims, rows, a.bench_iters, 5);
        if (rows <= 3) worst = @max(worst, row.roce_us);
        std.debug.print("ROCE-BENCH rank {d} rows {d} bytes {d}: roce {d:.1} us (graph), {d:.1} us (eager), nccl all-gather {d:.1} us\n", .{ a.rank, rows, row.bytes, row.roce_us, row.roce_eager_us, row.nccl_gather_us });
        const line = try std.fmt.allocPrint(gpa, "{s}\n {{\"rows\": {d}, \"bytes\": {d}, \"roce_us\": {d:.2}, \"roce_eager_us\": {d:.2}, \"nccl_gather_us\": {d:.2}}}", .{ if (first) "" else ",", rows, row.bytes, row.roce_us, row.roce_eager_us, row.nccl_gather_us });
        defer gpa.free(line);
        try out.appendSlice(gpa, line);
        first = false;
    }
    const st = r.stats();
    const tail = try std.fmt.allocPrint(gpa, "\n], \"rank\": {d}, \"world\": {d}, \"hcas\": {d}, \"bit_check\": {s}, \"decode_worst_us\": {d:.2}, \"ops_posted\": {d}, \"writes_completed\": {d}}}\n", .{ a.rank, a.world, hcas.len, tf(good), worst, st[0], st[1] });
    defer gpa.free(tail);
    try out.appendSlice(gpa, tail);
    if (a.out.len > 0) try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = out.items });
    const pass = good and worst > 0 and worst <= 65.0;
    std.debug.print("ROCE-BENCH {s} rank {d}: bit check {s}, decode windows (1-3 rows) worst {d:.1} us (target <= 65)\n", .{ if (pass) "PASS" else "FAIL", a.rank, if (good) "good" else "FAILED", worst });
    return pass;
}

/// tools/glm53/reference3a.py's probe inputs: element i of a tensor salted s is bf16(((h >> 8) % 4001 - 2000) / 2000)
/// with h = (i + 1000003 s) * 2654435761 mod 2^32 (numpy's arithmetic on the same integers).
fn probeFill(out: []u16, salt: u64) void {
    for (out, 0..) |*v, i| {
        const h: u64 = ((@as(u64, i) +% salt *% 1000003) *% 2654435761) & 0xFFFFFFFF;
        const x: f32 = @as(f32, @floatFromInt(@as(i64, @intCast((h >> 8) % 4001)) - 2000)) / @as(f32, 2000.0);
        v.* = glm.weights.bf16Bits(x);
    }
}

/// --bmm-probe FILE: torch.bmm's bits on fused.attention_core's wide absorb / expand shapes (the reference hashed its
/// outputs) against cuBLAS here with torch's arguments; a cuBLAS that picks another kernel fails here, before any
/// prompt runs. Returns true when every case matches.
fn bmmProbe(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, blas: *const cuda.cublas.Blas, path: []const u8) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var all_ok = true;
    for ((parsed.value.object.get("cases") orelse return error.BadProbe).array.items) |cv| {
        const o = cv.object;
        const kind = o.get("kind").?.string;
        const H: usize = @intCast(o.get("H").?.integer);
        const R: usize = @intCast(o.get("R").?.integer);
        const nope: usize = @intCast(o.get("nope").?.integer);
        const qd: usize = @intCast(o.get("qd").?.integer);
        const lw: usize = @intCast(o.get("lw").?.integer);
        const vd: usize = @intCast(o.get("vd").?.integer);
        const want = o.get("sha256").?.string;
        const absorb = std.mem.eql(u8, kind, "absorb");
        const na = if (absorb) R * H * qd else R * H * lw; // q [R, H qd] / ol [R, H, lw]
        const nb = if (absorb) H * nope * lw else H * vd * lw; // wk [H, nope, lw] / wv [H, v, lw]
        const nc = if (absorb) H * R * lw else H * R * vd;
        const ha = try gpa.alloc(u16, na);
        defer gpa.free(ha);
        const hb = try gpa.alloc(u16, nb);
        defer gpa.free(hb);
        probeFill(ha, if (absorb) 1 else 3);
        probeFill(hb, if (absorb) 2 else 4);
        var da = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ha));
        defer da.free();
        var db = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(hb));
        defer db.free();
        var dc = try cuda.DeviceBuffer.alloc(d, nc * 2);
        defer dc.free();
        if (absorb) {
            try blas.bgemmBf16(s.handle, .n, .n, lw, R, nope, db.ptr, lw, nope * lw, da.ptr, H * qd, qd, dc.ptr, lw, R * lw, H);
        } else {
            try blas.bgemmBf16(s.handle, .t, .n, vd, R, lw, db.ptr, lw, vd * lw, da.ptr, H * lw, lw, dc.ptr, vd, R * vd, H);
        }
        try s.synchronize();
        const hc = try gpa.alloc(u8, nc * 2);
        defer gpa.free(hc);
        try dc.download(0, hc);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(hc, &digest, .{});
        const got = std.fmt.bytesToHex(digest, .lower);
        const ok = std.mem.eql(u8, &got, want);
        all_ok = all_ok and ok;
        std.debug.print("bmm probe {s} H {d} R {d}: {s}\n", .{ kind, H, R, if (ok) "equal" else "DIFFERS" });
    }
    return all_ok;
}

const Extra3a = struct {
    roce: bool,
    roce_ops: u64,
    tiles_changed: usize,
    unpack_hits: usize,
    unpack_misses: usize,
    bmm_probe: []const u8,
    cublas_version: c_int,
    experts: []const u8,
    pe_det: []const u8,
    mtp_dense: bool,
    mtp_dense_gb: f64,
    mtp_chunks: usize,
    mtp_gemms: usize,
};

/// --mode 3a: Phase 2b's runs with the served engine's settings - the RoCE one-shot for decode windows (setup and
/// bit check once every rank has loaded), the reference boot's tile table (--tiles), the served prompt path (chunks of
/// --prompt-rows / --prompt-rows-short, sequence-parallel halves, the prompt GEMM, cuBLAS absorb / expand, prompt
/// experts with --pe-det slots16), NCCL's bf16 ring all-reduce / reduce-scatter for prompt rows.
fn run3a(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    var prompts_text: []u8 = &.{};
    var sset: Sampling2b = .{};
    const prompts = try readPrompts2b(gpa, io, a.prompts, a.tokens, &prompts_text, &sset);
    defer gpa.free(prompts_text);
    defer {
        for (prompts) |p| {
            gpa.free(p.name);
            gpa.free(p.ids);
            for (p.runs) |rs| gpa.free(rs.label);
            gpa.free(p.runs);
        }
        gpa.free(prompts);
    }
    var n_runs: usize = 0;
    for (prompts) |p| n_runs += p.runs.len;
    const det16 = std.mem.eql(u8, a.pe_det, "slots16");
    if (!det16 and !std.mem.eql(u8, a.pe_det, "0")) return error.UnsupportedPromptDet;
    const shared = std.mem.eql(u8, a.experts, "shared");
    if (!shared and !std.mem.eql(u8, a.experts, "tf")) return error.UnsupportedExperts;
    std.debug.print("rank {d} of {d}: phase 3a, {d} prompts, {d} runs, context {d}, prompt chunks of {d} / {d} rows (sequence parallel {s}), MTP drafts {d}, RoCE {s}, experts {s}, prompt experts det {s}\n", .{ a.rank, a.world, prompts.len, n_runs, a.context, a.prompt_rows, a.prompt_rows_short, tf(a.prompt_sp), a.k, tf(a.roce), a.experts, a.pe_det });

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [128]u8 = undefined;
    std.debug.print("device: {s} sm_{d}\n", .{ try ctx.name(&name_buf), try ctx.capability() });

    const master = try cuda.rendezvous.parseMaster(a.master);
    var lib = try cuda.nccl.Library.openPath(a.nccl_lib);
    defer lib.close();
    const nccl_version = try lib.version();
    std.debug.print("nccl: {s} version {d}\n", .{ a.nccl_lib, nccl_version });
    var rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, a.rank, a.world, a.timeout_s);
    defer rdv.close();
    var uid: cuda.nccl.UniqueId = undefined;
    if (a.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&lib);
    try rdv.broadcast(std.mem.asBytes(&uid));
    var comm = try cuda.nccl.Communicator.init(&lib, uid, a.rank, a.world);
    defer comm.deinit();
    std.debug.print("rank {d}: NCCL communicator up (world {d})\n", .{ a.rank, a.world });

    const cfg = try glm.Config.load(gpa, io, a.model);
    const n_layers = a.layers orelse cfg.layers;
    var settings = std.hash.Wyhash.init(0x6a3a);
    const letters = try glm.decode.parseSide(a.side);
    const pf_mode = try l2pfMode(a.l2pf);
    if (pf_mode >= 0 and !(a.l2pf_mb > 0 and a.l2pf_mb <= 64)) return error.BadL2pfBudget;
    const shape = [_]usize{ a.world, n_layers, a.context, a.tokens, a.window, a.k, a.draft_vocab, a.mtp_reuse, @intFromBool(a.graphs), @intFromBool(a.prewarm), @intFromBool(a.roce), a.prompt_rows, a.prompt_rows_short, @intFromBool(a.prompt_sp), @intFromBool(shared), @intFromBool(det16), @intFromBool(a.roce_health), a.unpack_mb, @intFromBool(letters.a), @intFromBool(letters.w), @intFromBool(letters.f), @intCast(pf_mode + 1), @intFromFloat(a.l2pf_mb * 1024), @intFromBool(a.multi_select), @intFromBool(a.device_cands), @intFromBool(a.prompt_prof.len > 0) };
    settings.update(std.mem.asBytes(&shape));
    settings.update(prompts_text);
    try rdv.sync("settings", settings.final(), a.timeout_s);

    // -- weights (O_DIRECT + io_uring a layer ahead), then the reference boot's EXL3 tile table ---------------------
    const t_load = now(io);
    var w = try glm.weights.load(gpa, io, &driver, a.model, cfg, .{ .layers = n_layers, .rank = a.rank, .world = a.world, .head = true, .log_every = 10, .fast = a.fast_load orelse true, .mtp = a.k > 0, .draft_vocab = if (a.k > 0) a.draft_vocab else 0 });
    defer w.deinit();
    var tiles_changed: usize = 0;
    if (a.tiles) |path| {
        const t = try glm.tiles.load(gpa, io, &w, path);
        tiles_changed = t.changed;
        std.debug.print("rank {d}: tiles from {s}: {d} linears, {d} differ from x3linear.plan\n", .{ a.rank, path, t.linears, t.changed });
    }
    const load_s = since(io, t_load);
    const read_gib = @as(f64, @floatFromInt(w.read_bytes)) / (1 << 30);
    std.debug.print("rank {d}: {d} layers{s} + head loaded in {d:.0} s ({d:.1} GiB of weights, {d:.1} GiB read direct)\n", .{ a.rank, n_layers, if (w.mtp != null) " + MTP layer" else "", load_s, @as(f64, @floatFromInt(w.device_bytes)) / (1 << 30), read_gib });
    try rdv.sync("loaded", 0, a.timeout_s);

    var k = try glm.kernels.Kernels.load(gpa, io, &ctx, a.aot);
    defer k.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const half = cfg.rope / 2;
    var inv_buf = try cuda.DeviceBuffer.alloc(&driver, half * 4);
    defer inv_buf.free();
    try k.invFreq(stream, inv_buf.ptr, cfg.rope, cfg.rope_theta);
    try stream.synchronize();
    var inv_status: []const u8 = "device (no reference to check)";
    if (a.inv_ref) |path| {
        const ref = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
        defer gpa.free(ref);
        const mine = try gpa.alloc(u8, half * 4);
        defer gpa.free(mine);
        try inv_buf.download(0, mine);
        if (ref.len != mine.len) return error.BadInvReference;
        if (std.mem.eql(u8, ref, mine)) {
            inv_status = "device == reference";
        } else {
            std.debug.print("WARN inv_freq differs from torch's (the reference's bytes are used)\n", .{});
            try inv_buf.upload(0, ref);
            inv_status = "DIFFERS: reference bytes used";
        }
    }

    // -- RoCE: set up once every rank has loaded (start_everywhere), checked against NCCL's rank-order sum --------
    var fast: ?*glm.roce.Roce = null;
    defer if (fast) |rc| rc.deinit(gpa);
    if (a.roce and a.world > 1) {
        var hb: [2][]const u8 = undefined;
        const hcas = parseHcas(a.hcas, &hb);
        fast = try glm.roce.Roce.start(gpa, &driver, &rdv, a.rank, a.world, .{ .hcas = hcas, .gid_index = a.gid, .spin_limit = a.spin_limit });
        if (!try fast.?.verify(gpa, &comm, stream, cfg.hidden, glm.forward.decode_rows)) return error.RoceCheckFailed;
        std.debug.print("rank {d}: decode-window reductions: RoCE one-shot on {d} rails\n", .{ a.rank, hcas.len });
    }

    // -- cuBLAS as torch sets up its handle (workspace, math mode), checked on the reference's bmm probe ------------
    var blas_lib = try cuda.cublas.Library.openPath(a.cublas_lib);
    defer blas_lib.close();
    const major: c_int = @intCast((try ctx.capability()) / 10);
    const ws_bytes = a.cublas_ws orelse cuda.cublas.torchWorkspaceBytes(major, null);
    var ws = try cuda.DeviceBuffer.alloc(&driver, @max(ws_bytes, 16));
    defer ws.free();
    var blas = try cuda.cublas.Blas.init(&blas_lib, ws.ptr, ws_bytes, a.cublas_math);
    defer blas.deinit();
    const cublas_version = try blas.version();
    var probe: []const u8 = "not run";
    if (a.bmm_probe) |path| {
        if (!try bmmProbe(gpa, io, &driver, stream, &blas, path)) {
            std.debug.print("RESULT glm53-generate FAIL rank {d}: cuBLAS {d} (workspace {d} bytes, math {d}) does not give torch.bmm's bits\n", .{ a.rank, cublas_version, ws_bytes, a.cublas_math });
            return error.BmmBitsDiffer;
        }
        probe = "equal";
    }

    // -- the served prompt path: the prompt GEMM's workspaces (main: with the unpack cache and cuBLAS), the comm
    //    stream of the sequence-parallel chunk
    var cache: ?glm.prompt.UnpackCache = null;
    if (a.unpack_mb > 0) cache = try glm.prompt.UnpackCache.init(gpa, &driver, a.unpack_mb << 20);
    defer if (cache) |*c| c.deinit();
    var main_wide = try glm.prompt.Wide.init(&driver, &k, stream, &w, a.prompt_rows, if (cache) |*c| c else null, &blas);
    defer main_wide.deinit();
    var side_stream = try cuda.Stream.init(&driver, true);
    defer side_stream.deinit();
    var side_wide = try glm.prompt.Wide.init(&driver, &k, side_stream, &w, a.prompt_rows / a.world + 1, null, null);
    defer side_wide.deinit();
    try stream.synchronize();
    try side_stream.synchronize();

    // -- Phase 3a speed: the MTP layer's prompt rows as cuBLAS GEMMs over dense bf16 experts (--mtp-dense 1; drafts
    //    only, so no output token can move; built before the caches so its memory shows up at load)
    var mtp_dense: ?glm.prompt.MtpDense = null;
    if (a.mtp_dense and a.k > 0 and shared and w.mtp != null) {
        const tm = now(io);
        if (glm.prompt.MtpDense.init(gpa, &driver, &k, stream, &w, &blas, a.prompt_rows)) |x| {
            mtp_dense = x;
            std.debug.print("rank {d}: MTP prompt rows dense (cuBLAS): {d:.2} GB a rank (bf16 experts + scratch), built in {d:.1} s\n", .{ a.rank, @as(f64, @floatFromInt(x.bytes)) / 1e9, since(io, tm) });
        } else |e| {
            std.debug.print("WARN rank {d}: MTP prompt rows dense unavailable ({t}): the decode expert kernel in blocks\n", .{ a.rank, e });
        }
    }
    defer if (mtp_dense) |*x| x.deinit();

    // -- Phase 3a speed: the decode side stream (TF_GLM53_SIDE), the L2 prefetch (TF_GLM53_L2PF: after the tiles, before
    //    any capture), the prompt path's timers (--prompt-prof, on after the prewarm)
    var side_obj: ?glm.decode.Side = null;
    if (letters.any()) side_obj = try glm.decode.Side.init(&driver, letters);
    defer if (side_obj) |*x| x.deinit();
    var pf_obj: ?glm.decode.L2pf = null;
    if (pf_mode >= 0) {
        pf_obj = try glm.decode.L2pf.init(gpa, &driver, &w, .{ .mode = pf_mode, .mb = a.l2pf_mb });
        var sbuf: [256]u8 = undefined;
        std.debug.print("rank {d}: {s}\n", .{ a.rank, pf_obj.?.summary(&sbuf) });
    }
    defer if (pf_obj) |*x| x.deinit();
    var prof = glm.prompt.Prof.init(gpa, &driver, stream);
    defer prof.deinit();
    std.debug.print("rank {d}: decode side stream {s}{s}{s}, L2 prefetch {s}, decode top-k {s}, device candidates {s}, prompt timers {s}\n", .{ a.rank, if (letters.a) "a" else "", if (letters.w) "w" else "", if (letters.f) "f" else "", a.l2pf, if (a.multi_select) "multi-block" else "radix", tf(a.device_cands), if (a.prompt_prof.len > 0) a.prompt_prof else "off" });
    const fwd: glm.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv_buf.ptr, .comm = &comm, .fast = fast, .ring = true, .wide = &main_wide, .shared_experts = shared, .pe_det16 = det16, .side = if (side_obj) |*x| x else null, .l2pf = if (pf_obj) |*x| x else null, .multi_select = a.multi_select, .prof = if (a.prompt_prof.len > 0) &prof else null, .mtp_dense = if (mtp_dense) |*x| x else null };
    var sp: ?glm.prompt.Sp = null;
    if (a.prompt_sp and a.world > 1) sp = try glm.prompt.Sp.init(&driver, &fwd, side_stream, &side_wide);
    defer if (sp) |*x| x.deinit();
    const capacity = a.context + a.k + 1; // Glm53Engine: Runner(limit + k + 1)
    var r = try glm.Runner.init(gpa, io, &driver, fwd, capacity, .{ .k = a.k, .graphs = a.graphs, .window = a.window, .reuse = a.mtp_reuse, .served = true, .prompt_rows = a.prompt_rows, .prompt_rows_short = a.prompt_rows_short, .prompt_sp = a.prompt_sp, .roce_health = a.roce_health, .sp = if (sp) |*x| x else null, .device_cands = a.device_cands });
    defer r.deinit();
    var prewarm_s: f64 = 0;
    var prewarm_graphs: usize = 0;
    if (a.prewarm) {
        const tp = now(io);
        prewarm_graphs = try r.prewarm();
        prewarm_s = since(io, tp);
        std.debug.print("rank {d}: {d} decode graphs captured in {d:.1} s\n", .{ a.rank, prewarm_graphs, prewarm_s });
    }
    var profiled = false; // --prompt-prof: the first run of the named prompt only (the gate takes each prompt's best run)
    try rdv.sync("warm", prewarm_graphs, a.timeout_s);

    var done: std.ArrayList(Done) = .empty;
    defer {
        for (done.items) |dd| gpa.free(dd.tokens);
        done.deinit(gpa);
    }
    var toks: std.ArrayList(u32) = .empty;
    defer toks.deinit(gpa);
    var all_tokens = std.hash.Wyhash.init(0);
    var seeds_ok = true;
    for (prompts, 0..) |p, pi| {
        const seed_ok = !p.seed_from_prompt or glm.sampling.seedFor(p.ids, 0) == p.seed;
        if (!seed_ok) seeds_ok = false;
        for (p.runs, 0..) |rs, ri| {
            const s: ?glm.sampling.Sampling = if (rs.greedy) null else .{ .seed = p.seed, .temperature = sset.temperature, .top_k = rs.top_k, .top_p = sset.top_p, .min_p = sset.min_p };
            if (rs.k > a.k) return error.TooManyDrafts;
            if (rs.mode != .mtp or rs.copies) return error.DraftersNeedMode4;
            prof.on = a.prompt_prof.len > 0 and !profiled and std.mem.startsWith(u8, p.name, a.prompt_prof);
            if (prof.on) profiled = true;
            const st = try r.generate(p.ids, p.tokens, s, rs.k, &toks);
            prof.on = false;
            all_tokens.update(std.mem.sliceAsBytes(toks.items));
            const n = toks.items.len;
            try done.append(gpa, .{ .prompt = pi, .run = ri, .tokens = try gpa.dupe(u32, toks.items), .st = st, .seed_ok = seed_ok });
            std.debug.print("rank {d} {s} {s}: prompt {d} in {d:.2} s ({d:.0} tok/s); {d} tokens, decode {d:.2} tok/s, {d:.3} tokens/round ({d} rounds), accept {d:.3}; first {any}\n", .{ a.rank, p.name, rs.label, p.ids.len, st.prefill_s, @as(f64, @floatFromInt(p.ids.len)) / @max(st.prefill_s, 1e-9), n, st.tokS(n), st.perRound(n), st.rounds, if (st.drafted > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0, toks.items[0..@min(8, n)] });
        }
    }

    const ex: Extra3a = .{
        .roce = fast != null,
        .roce_ops = if (fast) |f| f.stats()[0] else 0,
        .tiles_changed = tiles_changed,
        .unpack_hits = if (cache) |c| c.hits else 0,
        .unpack_misses = if (cache) |c| c.misses else 0,
        .bmm_probe = probe,
        .cublas_version = cublas_version,
        .experts = a.experts,
        .pe_det = a.pe_det,
        .mtp_dense = mtp_dense != null,
        .mtp_dense_gb = if (mtp_dense) |x| @as(f64, @floatFromInt(x.bytes)) / 1e9 else 0,
        .mtp_chunks = if (mtp_dense) |x| x.chunks else 0,
        .mtp_gemms = if (mtp_dense) |x| x.gemms else 0,
    };
    try writeOut3a(gpa, io, a, &w, n_layers, capacity, nccl_version, inv_status, load_s, read_gib, prewarm_graphs, prewarm_s, prompts, done.items, ex);
    const agree = rdv.sync("done", all_tokens.final(), a.timeout_s);
    if (agree) {
        std.debug.print("RESULT glm53-generate {s} rank {d}: phase 3a, {d} runs, load {d:.0} s, RoCE {s}, inv_freq {s}\n", .{ if (seeds_ok) "OK" else "FAIL", a.rank, done.items.len, load_s, tf(fast != null), inv_status });
        return seeds_ok;
    } else |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: the ranks' token streams differ ({t})\n", .{ a.rank, e });
        return false;
    }
}

fn writeOut3a(gpa: std.mem.Allocator, io: std.Io, a: Args, w: *const glm.weights.Weights, n_layers: usize, capacity: usize, nccl_version: c_int, inv_status: []const u8, load_s: f64, read_gib: f64, prewarm_graphs: usize, prewarm_s: f64, prompts: []const Prompt2b, done: []const Done, ex: Extra3a) !void {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const head1 = try std.fmt.allocPrint(gpa, "{{\"tool\": \"tf-glm53-generate\", \"mode\": \"{s}\", \"rank\": {d}, \"world\": {d}, \"layers\": {d}, \"context\": {d}, \"capacity\": {d}, \"window\": {d}, \"k\": {d}, \"prompt_rows\": {d}, \"prompt_rows_short\": {d}, \"prompt_sp\": {s}, \"roce\": {s}, \"roce_ops\": {d}, \"tiles_changed\": {d}, \"unpack_hits\": {d}, \"unpack_misses\": {d}, \"bmm_probe\": \"{s}\", \"cublas_version\": {d}, \"experts\": \"{s}\", \"pe_det\": \"{s}\", \"mtp_dense\": {s}, ", .{ a.mode, a.rank, a.world, n_layers, a.context, capacity, a.window, a.k, a.prompt_rows, a.prompt_rows_short, tf(a.prompt_sp), tf(ex.roce), ex.roce_ops, ex.tiles_changed, ex.unpack_hits, ex.unpack_misses, ex.bmm_probe, ex.cublas_version, ex.experts, ex.pe_det, tf(ex.mtp_dense) });
    defer gpa.free(head1);
    try out.appendSlice(gpa, head1);
    const head = try std.fmt.allocPrint(gpa, "\"mtp_dense_gb\": {d:.2}, \"mtp_chunks\": {d}, \"mtp_gemms\": {d}, \"vocab_off\": {d}, \"vocab_part\": {d}, \"nccl_version\": {d}, \"inv_freq\": \"{s}\", \"load_s\": {d:.1}, \"read_gib\": {d:.2}, \"prewarm_graphs\": {d}, \"prewarm_s\": {d:.1}, \"dspark\": \"{s}\", \"dflash\": \"{s}\", \"copy_max\": {d}, \"runs\": [", .{ ex.mtp_dense_gb, ex.mtp_chunks, ex.mtp_gemms, w.vocab_off, w.vocab_part, nccl_version, inv_status, load_s, read_gib, prewarm_graphs, prewarm_s, std.fs.path.basename(a.dspark), std.fs.path.basename(a.dflash), a.copy_max });
    defer gpa.free(head);
    try out.appendSlice(gpa, head);
    for (done, 0..) |dd, i| {
        const p = prompts[dd.prompt];
        const rs = p.runs[dd.run];
        const n = dd.tokens.len;
        const acc: f64 = if (dd.st.drafted > 0) @as(f64, @floatFromInt(dd.st.accepted)) / @as(f64, @floatFromInt(dd.st.drafted)) else 0;
        const ph = try std.fmt.allocPrint(gpa, "{s}\n {{\"name\": \"{s}\", \"run\": \"{s}\", \"prompt_len\": {d}, \"seed\": \"{d}\", \"seed_ok\": {s}, \"k\": {d}, \"greedy\": {s}, \"top_k\": {d}, \"prefill_s\": {d:.3}, \"decode_s\": {d:.3}, \"tok_s\": {d:.3}, \"rounds\": {d}, \"tokens_per_round\": {d:.4}, \"accept\": {d:.4}, \"capture_s\": {d:.2}, \"graphs\": {d}, \"draft_mode\": \"{t}\", \"copies\": {s}, \"copy_rounds\": {d}, \"copy_drafted\": {d}, \"copy_accepted\": {d}, \"drafted\": {d}, \"accepted\": {d}, \"tokens\": [", .{ if (i > 0) "," else "", p.name, rs.label, p.ids.len, p.seed, tf(dd.seed_ok), rs.k, tf(rs.greedy), rs.top_k, dd.st.prefill_s, dd.st.decode_s, dd.st.tokS(n), dd.st.rounds, dd.st.perRound(n), acc, dd.st.capture_s, dd.st.graphs, rs.mode, tf(rs.copies), dd.st.copy_rounds, dd.st.copy_drafted, dd.st.copy_accepted, dd.st.drafted, dd.st.accepted });
        defer gpa.free(ph);
        try out.appendSlice(gpa, ph);
        for (dd.tokens, 0..) |t, j| {
            var buf: [16]u8 = undefined;
            try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "{s}{d}", .{ if (j > 0) ", " else "", t }));
        }
        try out.appendSlice(gpa, "]}");
    }
    try out.appendSlice(gpa, "\n]}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = out.items });
    std.debug.print("rank {d}: wrote {s}\n", .{ a.rank, a.out });
}

/// --sampler-check: sampling.zig against the Python engine's exact_sampling on generated cases (no GPU).
fn samplerCheck(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    var bad: usize = 0;
    var bad_keep: usize = 0;
    var total: usize = 0;
    var cands: std.ArrayList(glm.sampling.Cand) = .empty;
    defer cands.deinit(gpa);
    for ((root.get("cases") orelse return error.BadCases).array.items) |cv| {
        const o = cv.object;
        const vals = o.get("values").?.array.items;
        const ids = o.get("ids").?.array.items;
        if (vals.len != ids.len) return error.BadCases;
        cands.clearRetainingCapacity();
        for (vals, ids) |v, id| try cands.append(gpa, .{ .v = @floatCast(try jsonFloat(v)), .id = id.integer });
        const s: glm.sampling.Sampling = .{
            .seed = try jsonU64(o.get("seed").?),
            .temperature = try jsonFloat(o.get("temperature").?),
            .top_k = @intCast(o.get("top_k").?.integer),
            .top_p = try jsonFloat(o.get("top_p").?),
            .min_p = try jsonFloat(o.get("min_p").?),
        };
        var nk: usize = 0;
        const got = glm.sampling.chooseKeep(cands.items, @intCast(o.get("position").?.integer), s, &nk);
        const want: u32 = @intCast(o.get("pick").?.integer);
        total += 1;
        if (got != want) {
            bad += 1;
            if (bad <= 10) std.debug.print("case {d}: zig {d}, python {d} (top_k {d}, top_p {d}, {d} candidates)\n", .{ total - 1, got, want, s.top_k, s.top_p, vals.len });
        }
        if (o.get("keep")) |kv| {
            if (nk != @as(usize, @intCast(kv.integer))) bad_keep += 1;
        }
    }
    var seed_bad: usize = 0;
    var seed_total: usize = 0;
    if (root.get("seeds")) |sv| {
        for (sv.array.items) |e| {
            const o = e.object;
            const list = o.get("ids").?.array.items;
            const ids = try gpa.alloc(u32, list.len);
            defer gpa.free(ids);
            for (list, ids) |x, *y| y.* = @intCast(x.integer);
            seed_total += 1;
            if (glm.sampling.seedFor(ids, o.get("salt").?.integer) != try jsonU64(o.get("seed").?)) seed_bad += 1;
        }
    }
    const ok = bad == 0 and bad_keep == 0 and seed_bad == 0 and total > 0;
    std.debug.print("SAMPLER {s} picks {d}/{d} nucleus sizes {d}/{d} seeds {d}/{d}\n", .{ if (ok) "PASS" else "FAIL", total - bad, total, total - bad_keep, total, seed_total - seed_bad, seed_total });
    return ok;
}

/// --mode 3b: the served engine (cuda_engine.Boot + Session, as tensorfold-native serve runs it on every rank) over the
/// prompts file - every rank reads the same file and runs the same requests in the same order, so no rank needs
/// rank 0's header. Each run starts cold (--reuse 1: the kept states forgotten, the prompt cut at its keep points as
/// the server cuts a cold prompt), stops at EOS (--stop-eos 1: the prompt's "eos", else the model's end ids), and
/// samples with the prompt's own temperature / top_p / min_p over the file's.
fn run3b(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    var prompts_text: []u8 = &.{};
    var sset: Sampling2b = .{};
    const prompts = try readPrompts2b(gpa, io, a.prompts, a.tokens, &prompts_text, &sset);
    defer gpa.free(prompts_text);
    defer {
        for (prompts) |p| {
            gpa.free(p.name);
            gpa.free(p.ids);
            if (p.eos) |e| gpa.free(e);
            for (p.runs) |rs| gpa.free(rs.label);
            gpa.free(p.runs);
        }
        gpa.free(prompts);
    }
    var o: glm.engine.Options = .{ .rank = a.rank, .world = a.world, .master = a.master, .model = a.model, .aot = a.aot, .nccl_lib = a.nccl_lib, .cublas_lib = a.cublas_lib, .cublas_ws = a.cublas_ws, .cublas_math = a.cublas_math, .context = a.context, .k = a.k, .draft_vocab = a.draft_vocab, .mtp_reuse = a.mtp_reuse, .graphs = a.graphs, .prewarm = a.prewarm, .fast_load = a.fast_load orelse true, .roce = a.roce, .hcas = a.hcas, .gid = a.gid, .roce_health = a.roce_health, .spin_limit = a.spin_limit, .tiles = a.tiles, .inv_ref = a.inv_ref, .bmm_probe = a.bmm_probe, .prompt_rows = a.prompt_rows, .prompt_rows_short = a.prompt_rows_short, .prompt_sp = a.prompt_sp, .experts = a.experts, .pe_det = a.pe_det, .unpack_mb = a.unpack_mb, .side = a.side, .l2pf = a.l2pf, .l2pf_mb = a.l2pf_mb, .multi_select = a.multi_select, .device_cands = a.device_cands, .mtp_dense = a.mtp_dense, .dcp = a.dcp, .timeout_s = a.timeout_s };
    o.reuse = try glm.reuse.Settings.fromEnv();
    o.reuse.on = a.reuse;
    // Phase 4: the drafters, their settings, copy drafts (on when --copy 1 or a run asks for them)
    var any_copies = false;
    for (prompts) |p| for (p.runs) |rs| {
        any_copies = any_copies or rs.copies;
        if (rs.mode == .dspark and a.dspark.len == 0) return error.NoDSpark;
        if (rs.mode == .dflash and a.dflash.len == 0) return error.NoDFlash;
    };
    o.dspark = a.dspark;
    o.dflash = a.dflash;
    o.dspark_env = .{ .policy = a.dspark_policy, .confidence = a.dspark_confidence, .depth = a.dspark_depth, .top_k = a.dspark_topk };
    o.dflash_set = .{ .depth = a.dflash_depth, .confidence = a.dflash_confidence };
    o.copy = if (a.copy orelse any_copies) .{ .on = true, .match = a.copy_min, .most = a.copy_max } else .{ .on = false, .match = 0, .most = 0 };
    o.verify_rows = a.verify_rows;
    std.debug.print("rank {d} of {d}: phase 3b, {d} prompts, context {d}, MTP drafts {d}, stop at EOS {s}, prompt-reuse cuts {s}, DCP {d} (0: auto)\n", .{ a.rank, a.world, prompts.len, a.context, a.k, tf(a.stop_eos), tf(a.reuse), a.dcp });
    const boot = try glm.engine.Boot.init(gpa, io, o);
    defer boot.deinit();
    var sess = try glm.engine.Session.init(gpa, boot);
    defer sess.deinit();
    var done: std.ArrayList(Done) = .empty;
    defer {
        for (done.items) |dd| gpa.free(dd.tokens);
        done.deinit(gpa);
    }
    var begins: std.ArrayList(usize) = .empty;
    defer begins.deinit(gpa);
    var toks: std.ArrayList(u32) = .empty;
    defer toks.deinit(gpa);
    var all_tokens = std.hash.Wyhash.init(0);
    var seeds_ok = true;
    for (prompts, 0..) |p, pi| {
        const seed_ok = !p.seed_from_prompt or glm.sampling.seedFor(p.ids, 0) == p.seed;
        if (!seed_ok) seeds_ok = false;
        for (p.runs, 0..) |rs, ri| {
            const temperature = p.temperature orelse sset.temperature;
            const s: ?glm.sampling.Sampling = if (rs.greedy or temperature <= 0) null else .{ .seed = p.seed, .temperature = temperature, .top_k = rs.top_k, .top_p = p.top_p orelse sset.top_p, .min_p = p.min_p orelse sset.min_p };
            if (rs.k > a.k) return error.TooManyDrafts;
            const max_tokens = try sess.clampTokens(p.ids.len, p.tokens);
            // the server's cold prompt: nothing kept, cut at its keep points (drafted runs keep their states as a
            // served request does, then the next run forgets them)
            if (sess.reuse) |pr| try pr.flush();
            const draft = rs.k > 0 or rs.mode != .mtp;
            const c = try sess.choose(p.ids, draft);
            defer if (sess.reuse != null) c.deinit(gpa);
            const eos: []const u32 = if (!a.stop_eos) &.{} else p.eos orelse sess.eos;
            const res = try sess.exec(.{ .prompt = p.ids, .max_tokens = max_tokens, .sampling = s, .draft = draft, .eos = eos, .begin = c.begin, .stops = c.stops, .keeps = c.keeps, .mode = rs.mode, .copies = rs.copies }, &toks);
            gpa.free(res.kept);
            const st = res.stats;
            all_tokens.update(std.mem.sliceAsBytes(toks.items));
            const n = toks.items.len;
            try done.append(gpa, .{ .prompt = pi, .run = ri, .tokens = try gpa.dupe(u32, toks.items), .st = st, .seed_ok = seed_ok });
            try begins.append(gpa, c.begin);
            std.debug.print("rank {d} {s} {s}: prompt {d} ({d} cuts) in {d:.2} s ({d:.0} tok/s); {d} tokens{s}, decode {d:.2} tok/s, {d:.3} tokens/round ({d} rounds), accept {d:.3}; first {any}\n", .{ a.rank, p.name, rs.label, p.ids.len, c.stops.len, st.prefill_s, @as(f64, @floatFromInt(p.ids.len)) / @max(st.prefill_s, 1e-9), n, if (st.eos) " (EOS)" else "", st.tokS(n), st.perRound(n), st.rounds, if (st.drafted > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0, toks.items[0..@min(8, n)] });
        }
    }
    const ex: Extra3a = .{ .roce = boot.fast != null, .roce_ops = if (boot.fast) |f| f.stats()[0] else 0, .tiles_changed = 0, .unpack_hits = if (boot.cache) |c| c.hits else 0, .unpack_misses = if (boot.cache) |c| c.misses else 0, .bmm_probe = if (a.bmm_probe != null) "equal" else "not run", .cublas_version = try boot.blas.version(), .experts = a.experts, .pe_det = a.pe_det, .mtp_dense = boot.mtp_dense != null, .mtp_dense_gb = if (boot.mtp_dense) |x| @as(f64, @floatFromInt(x.bytes)) / 1e9 else 0, .mtp_chunks = if (boot.mtp_dense) |x| x.chunks else 0, .mtp_gemms = if (boot.mtp_dense) |x| x.gemms else 0 };
    try writeOut3a(gpa, io, a, &boot.w, boot.cfg.layers, a.context + a.k + 1, boot.nccl_version, "device", boot.load_s, @as(f64, @floatFromInt(boot.w.read_bytes)) / (1 << 30), boot.prewarm_graphs, boot.prewarm_s, prompts, done.items, ex);
    std.debug.print("rank {d}: phase 3b DCP {d}, resumed begins {any}\n", .{ a.rank, boot.dcp, begins.items });
    const agree = boot.rdv.sync("done", all_tokens.final(), a.timeout_s);
    if (agree) {
        std.debug.print("RESULT glm53-generate {s} rank {d}: phase 3b, {d} runs, load {d:.0} s, DCP {d}\n", .{ if (seeds_ok) "OK" else "FAIL", a.rank, done.items.len, boot.load_s, boot.dcp });
        return seeds_ok;
    } else |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: the ranks' token streams differ ({t})\n", .{ a.rank, e });
        return false;
    }
}

// ------------------------------------------------------------------------------------------------- Phase 4b ---

/// A group of the 4b prompts file: prompts decoded together (indexes into "prompts"), each with its seed.
const Group4b = struct { name: []const u8, members: []usize, seeds: []u64, greedy: bool };

/// One stream's outcome in a 4b run.
const Out4b = struct {
    tokens: []u32 = &.{},
    ttft_s: f64 = 0,
    prefill_s: f64 = 0,
    decode_s: f64 = 0,
    tok_s: f64 = 0,
    rounds: usize = 0,
    drafted: usize = 0,
};

fn freeOuts(gpa: std.mem.Allocator, outs: []Out4b) void {
    for (outs) |x| gpa.free(x.tokens);
    gpa.free(outs);
}

fn nsToS(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

/// The lead's loop on every rank (no messages: every rank decides alike): the prompts `idx` admitted together, rounds
/// until every stream finished. Outcomes in admission order; `wall`: admission to the last stream's end (seconds).
fn concurrent4b(gpa: std.mem.Allocator, io: std.Io, mu: *glm.multi.Multi, mode: glm.multi.Mode, prompts: []const Prompt, idx: []const usize, samps: []const ?glm.sampling.Sampling, count: usize, eos: []const u32, wall: *f64) ![]Out4b {
    const outs = try gpa.alloc(Out4b, idx.len);
    for (outs) |*x| x.* = .{};
    errdefer freeOuts(gpa, outs);
    var sids: [glm.multi.max_streams]u32 = undefined;
    const t0 = now(io);
    for (idx, 0..) |pi, i| {
        const c = try mu.choose(prompts[pi].ids, true);
        defer c.deinit(gpa);
        const sid = mu.next_sid;
        const limit = mu.r.capacity - mu.k - 1; // Glm53Engine.limit: the count within what is left after the prompt
        if (prompts[pi].ids.len >= limit) return error.PromptTooLong;
        const n_tok = @max(1, @min(count, limit - prompts[pi].ids.len));
        _ = try mu.admit(.{ .sid = sid, .slot = c.slot, .mode = mode, .sampling = samps[i], .count = n_tok, .prompt = prompts[pi].ids, .eos = eos, .begin = c.begin, .src = c.src, .stops = c.stops, .keeps = c.keeps });
        sids[i] = sid;
    }
    var buf: [glm.multi.max_streams]u32 = undefined;
    while (mu.live() > 0) {
        try mu.step(null);
        for (mu.doneList(&buf)) |sid| {
            const st = mu.find(sid).?;
            const i = std.mem.indexOfScalar(u32, sids[0..idx.len], sid) orelse return error.UnknownStream;
            outs[i] = .{ .tokens = try gpa.dupe(u32, st.out.items), .ttft_s = nsToS(st.t_first - st.t_admit), .prefill_s = st.prefill_s, .decode_s = st.decodeS(), .tok_s = st.tokS(), .rounds = st.rounds, .drafted = st.drafted };
            mu.finish(sid);
        }
    }
    wall.* = since(io, t0);
    return outs;
}

fn jsonU64List(gpa: std.mem.Allocator, v: std.json.Value) ![]u64 {
    if (v != .array) return error.BadPrompts;
    const items = v.array.items;
    const out = try gpa.alloc(u64, items.len);
    errdefer gpa.free(out);
    for (items, out) |x, *y| y.* = try jsonU64(x);
    return out;
}

/// --mode 4b: the served engine with --parallel N over the prompts file's groups (see the header).
fn run4b(gpa: std.mem.Allocator, io: std.Io, a: Args) !bool {
    if (a.parallel < 2) return error.ParallelNeeded;
    var prompts_text: []u8 = &.{};
    const prompts = try readPrompts(gpa, io, a.prompts, &prompts_text);
    defer {
        for (prompts) |p| {
            gpa.free(p.name);
            gpa.free(p.ids);
        }
        gpa.free(prompts);
        gpa.free(prompts_text);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, prompts_text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const count: usize = if (root.get("tokens")) |v| @intCast(try jsonU64(v)) else a.tokens;
    const temperature = if (root.get("temperature")) |v| try jsonFloat(v) else 1.0;
    const top_p = if (root.get("top_p")) |v| try jsonFloat(v) else 0.95;
    const min_p = if (root.get("min_p")) |v| try jsonFloat(v) else 0.0;
    const top_k: usize = if (root.get("top_k")) |v| @intCast(try jsonU64(v)) else 20;
    var groups: std.ArrayList(Group4b) = .empty;
    defer {
        for (groups.items) |g| {
            gpa.free(g.members);
            gpa.free(g.seeds);
        }
        groups.deinit(gpa);
    }
    for ((root.get("groups") orelse return error.BadPrompts).array.items) |gv| {
        const go = gv.object;
        const members64 = try jsonU64List(gpa, go.get("members") orelse return error.BadPrompts);
        defer gpa.free(members64);
        const members = try gpa.alloc(usize, members64.len);
        errdefer gpa.free(members);
        for (members64, members) |x, *y| {
            if (x >= prompts.len) return error.BadPrompts;
            y.* = @intCast(x);
        }
        const seeds = try jsonU64List(gpa, go.get("seeds") orelse return error.BadPrompts);
        errdefer gpa.free(seeds);
        if (seeds.len != members.len or members.len == 0 or members.len > a.parallel) return error.BadPrompts;
        const greedy = if (go.get("greedy")) |v| v == .bool and v.bool else true;
        const gname = go.get("name") orelse return error.BadPrompts;
        if (gname != .string) return error.BadPrompts;
        try groups.append(gpa, .{ .name = gname.string, .members = members, .seeds = seeds, .greedy = greedy });
    }
    var o: glm.engine.Options = .{ .rank = a.rank, .world = a.world, .master = a.master, .model = a.model, .aot = a.aot, .nccl_lib = a.nccl_lib, .cublas_lib = a.cublas_lib, .cublas_ws = a.cublas_ws, .cublas_math = a.cublas_math, .context = a.context, .k = a.k, .draft_vocab = a.draft_vocab, .mtp_reuse = a.mtp_reuse, .graphs = a.graphs, .prewarm = a.prewarm, .fast_load = a.fast_load orelse true, .roce = a.roce, .hcas = a.hcas, .gid = a.gid, .roce_health = a.roce_health, .spin_limit = a.spin_limit, .tiles = a.tiles, .inv_ref = a.inv_ref, .bmm_probe = a.bmm_probe, .prompt_rows = a.prompt_rows, .prompt_rows_short = a.prompt_rows_short, .prompt_sp = a.prompt_sp, .experts = a.experts, .pe_det = a.pe_det, .unpack_mb = a.unpack_mb, .side = a.side, .l2pf = a.l2pf, .l2pf_mb = a.l2pf_mb, .multi_select = a.multi_select, .device_cands = a.device_cands, .mtp_dense = a.mtp_dense, .dcp = a.dcp, .timeout_s = a.timeout_s, .parallel = a.parallel };
    o.reuse = try glm.reuse.Settings.fromEnv();
    o.reuse.on = a.reuse;
    o.multi_set = try glm.multi.Settings.fromEnv();
    // Phase 4c: the drafters (as --mode 4 loads them) and the streams' draft mode
    o.dspark = a.dspark;
    o.dflash = a.dflash;
    o.dspark_env = .{ .policy = a.dspark_policy, .confidence = a.dspark_confidence, .depth = a.dspark_depth, .top_k = a.dspark_topk };
    o.dflash_set = .{ .depth = a.dflash_depth, .confidence = a.dflash_confidence };
    o.verify_rows = a.verify_rows;
    o.mode = a.draft_mode;
    std.debug.print("rank {d} of {d}: phase 4b, --parallel {d}, {d} groups, context {d}, MTP drafts {d}, {d} tokens a stream, stop at EOS {s}, one-stream check {s}, draft cut {d} from {d} streams\n", .{ a.rank, a.world, a.parallel, groups.items.len, a.context, a.k, count, tf(a.stop_eos), tf(a.single), o.multi_set.draft_cut, o.multi_set.cut_streams });
    const boot = try glm.engine.Boot.init(gpa, io, o);
    defer boot.deinit();
    var sess = try glm.engine.Session.init(gpa, boot);
    defer sess.deinit();
    const mu = boot.multi orelse return error.NoConcurrentDecoder;
    const mode = glm.multi.modeOf(boot.mode);
    std.debug.print("rank {d}: phase 4b streams draft with {t} (windows up to {d} rows; drafter rows {d})\n", .{ a.rank, mode, mu.rows, mu.frows });
    const eos: []const u32 = if (a.stop_eos) sess.eos else &.{};
    var all_tokens = std.hash.Wyhash.init(0);
    var pass = true;
    var w: std.Io.Writer.Allocating = .init(gpa);
    defer w.deinit();
    const out = &w.writer;
    try out.print("{{\"draft_mode\": \"{t}\", \"frows\": {d}, ", .{ mode, mu.frows });
    try out.print("\"tool\": \"tf-glm53-generate\", \"mode\": \"4b\", \"rank\": {d}, \"world\": {d}, \"parallel\": {d}, \"k\": {d}, \"context\": {d}, \"tokens\": {d}, \"stop_eos\": {s}, \"draft_cut\": {d}, \"cut_streams\": {d}, \"load_s\": {d:.1}, \"prewarm_graphs\": {d}, \"prewarm_s\": {d:.1}, \"groups\": [", .{ a.rank, a.world, a.parallel, a.k, a.context, count, tf(a.stop_eos), o.multi_set.draft_cut, o.multi_set.cut_streams, boot.load_s, boot.prewarm_graphs, boot.prewarm_s });
    var toks: std.ArrayList(u32) = .empty;
    defer toks.deinit(gpa);
    for (groups.items, 0..) |g, gi| {
        const n = g.members.len;
        const samps = try gpa.alloc(?glm.sampling.Sampling, n);
        defer gpa.free(samps);
        for (samps, g.seeds) |*x, seed| {
            x.* = if (g.greedy or temperature <= 0) null else .{ .seed = seed & 0x7FFFFFFFFFFFFFFF, .temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p };
        }
        const cuts_before = mu.cuts;
        const rounds_before = mu.rounds;
        var wall: f64 = 0;
        const conc = try concurrent4b(gpa, io, mu, mode, prompts, g.members, samps, count, eos, &wall);
        defer freeOuts(gpa, conc);
        const conc_rounds = mu.rounds - rounds_before;
        const conc_cuts = mu.cuts - cuts_before;
        // each prompt alone through the concurrent decoder
        const alone = try gpa.alloc(Out4b, n);
        for (alone) |*x| x.* = .{};
        defer freeOuts(gpa, alone);
        for (0..n) |i| {
            var wall1: f64 = 0;
            const one = try concurrent4b(gpa, io, mu, mode, prompts, g.members[i .. i + 1], samps[i .. i + 1], count, eos, &wall1);
            alone[i] = one[0];
            gpa.free(one);
        }
        // each prompt through the one-stream path (Runner.run in slot 0, copy drafts off)
        const single = try gpa.alloc(Out4b, n);
        for (single) |*x| x.* = .{};
        defer freeOuts(gpa, single);
        if (a.single) {
            for (0..n) |i| {
                const ids = prompts[g.members[i]].ids;
                const max_tokens = try sess.clampTokens(ids.len, count);
                const res = try sess.exec(.{ .prompt = ids, .max_tokens = max_tokens, .sampling = samps[i], .draft = true, .eos = eos, .copies = false }, &toks);
                gpa.free(res.kept);
                single[i] = .{ .tokens = try gpa.dupe(u32, toks.items), .prefill_s = res.stats.prefill_s, .decode_s = res.stats.decode_s, .tok_s = res.stats.tokS(toks.items.len), .rounds = res.stats.rounds };
            }
        }
        var total: usize = 0;
        var ttft_max: f64 = 0;
        var tps_sum: f64 = 0;
        var eq_alone = true;
        var eq_single = true;
        for (0..n) |i| {
            total += conc[i].tokens.len;
            ttft_max = @max(ttft_max, conc[i].ttft_s);
            tps_sum += conc[i].tok_s;
            eq_alone = eq_alone and std.mem.eql(u32, conc[i].tokens, alone[i].tokens);
            if (a.single) eq_single = eq_single and std.mem.eql(u32, alone[i].tokens, single[i].tokens);
            all_tokens.update(std.mem.sliceAsBytes(conc[i].tokens));
        }
        pass = pass and eq_alone and eq_single;
        const nf: f64 = @floatFromInt(n);
        const agg = @as(f64, @floatFromInt(total)) / @max(wall, 1e-9);
        std.debug.print("rank {d} RESULT4B[{t}] {s} {s}: {d} streams, aggregate {d:.1} tok/s ({d} tokens in {d:.2} s, {d} rounds, {d} drafts cut), per-stream {d:.1} tok/s, TTFT max {d:.3} s; concurrent == alone {s}; alone == one-stream {s}\n", .{ a.rank, mode, g.name, if (g.greedy) "greedy" else "sampled", n, agg, total, wall, conc_rounds, conc_cuts, tps_sum / nf, ttft_max, tf(eq_alone), if (a.single) tf(eq_single) else "not run" });
        try out.print("{s}\n {{\"name\": \"{s}\", \"greedy\": {s}, \"streams\": {d}, \"wall_s\": {d:.4}, \"aggregate_tps\": {d:.3}, \"per_stream_tps\": {d:.3}, \"ttft_max_s\": {d:.4}, \"rounds\": {d}, \"cuts\": {d}, \"concurrent_eq_alone\": {s}, \"alone_eq_single\": {s}, \"single_run\": {s}, \"members\": [", .{ if (gi > 0) "," else "", g.name, tf(g.greedy), n, wall, agg, tps_sum / nf, ttft_max, conc_rounds, conc_cuts, tf(eq_alone), tf(eq_single), tf(a.single) });
        for (0..n) |i| {
            const c = conc[i];
            const tpr = @as(f64, @floatFromInt(c.tokens.len -| 1)) / @as(f64, @floatFromInt(@max(c.rounds, 1)));
            const atpr = @as(f64, @floatFromInt(alone[i].tokens.len -| 1)) / @as(f64, @floatFromInt(@max(alone[i].rounds, 1)));
            const same_single = !a.single or std.mem.eql(u32, alone[i].tokens, single[i].tokens);
            try out.print("{s}\n  {{\"prompt\": \"{s}\", \"seed\": \"{d}\", \"tokens_n\": {d}, \"ttft_s\": {d:.4}, \"prefill_s\": {d:.4}, \"decode_s\": {d:.4}, \"tok_s\": {d:.3}, \"rounds\": {d}, \"tokens_per_round\": {d:.4}, \"alone_tok_s\": {d:.3}, \"alone_tokens_per_round\": {d:.4}, \"single_tok_s\": {d:.3}, \"eq_alone\": {s}, \"eq_single\": {s}, \"tokens\": [", .{ if (i > 0) "," else "", prompts[g.members[i]].name, g.seeds[i], c.tokens.len, c.ttft_s, c.prefill_s, c.decode_s, c.tok_s, c.rounds, tpr, alone[i].tok_s, atpr, single[i].tok_s, tf(std.mem.eql(u32, c.tokens, alone[i].tokens)), tf(same_single) });
            for (c.tokens, 0..) |t, q| try out.print("{s}{d}", .{ if (q > 0) "," else "", t });
            try out.writeAll("]}");
        }
        try out.writeAll("]}");
    }
    try out.print("\n], \"pass\": {s}}}\n", .{tf(pass)});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = a.out, .data = w.written() });
    const agree = boot.rdv.sync("done", all_tokens.final(), a.timeout_s);
    if (agree) {
        std.debug.print("RESULT glm53-generate {s} rank {d}: phase 4b, --parallel {d}, {d} groups, load {d:.0} s{s}\n", .{ if (pass) "OK" else "FAIL", a.rank, a.parallel, groups.items.len, boot.load_s, if (pass) "" else " (a stream differs from its alone / one-stream run)" });
        return pass;
    } else |e| {
        std.debug.print("RESULT glm53-generate FAIL rank {d}: the ranks' token streams differ ({t})\n", .{ a.rank, e });
        return false;
    }
}
