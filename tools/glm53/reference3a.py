#!/usr/bin/env python3
"""GLM-5.3's Python CUDA engine in its SERVED configuration as the Zig port's Phase 3a reference: ``Glm53Engine``
itself (engine.py) built as ``tensorfold.cli serve`` builds it from the 2026-10-05 image, then its ``Runner`` driven
request by request as reference2b.py drives it.

Served = the image's defaults, so the reference runs what the server runs: the RoCE one-shot for decode windows
(TF_GLM53_ROCE=1, started once every rank has loaded, its bit check against NCCL), the sequence-parallel prompt
chunks (TF_GLM53_PROMPT_SP=1, PROMPT_ROWS 8192 / SHORT 4096, NCCL's bf16 ring reduce), the prompt GEMM with its
unpack cache, torch.bmm absorb / expand, prompt experts (cuda_exl3 present: SharedExperts), radix top-k for prompt
blocks, timed EXL3 tiles shared from rank 0 (TF_GLM53_TUNE=1), the decode side stream and the L2 prefetch. Changed,
and only these (FROZEN_3A, recorded in ref-r<R>.json):

* ``TF_EXL3_PROMPT_DET=slots16`` - the default (fp32 red.add in arrival order) makes a long prompt's bits differ run to
  run, so no two runs - of either engine - could be token-identical; slots16 is the image's documented reproducible
  mode (~4-6 % slower prefill), and the Zig engine runs it too. run_phase3a_cluster.sh SPEED_DEFAULT=1 adds a
  speed-only Python pass at the default for reference.
* ``TF_GLM53_TILES=save:<out>/tiles-r<R>.json`` - the boot's timed tiles written (they are not changed; the Zig ranks
  load the table, as a restarted server would with load:).
* ``TF_GLM53_COPY_DRAFTS=0`` (Phase 4), ``TF_GLM53_PROMPT_REUSE=0`` (3b; generate() is called directly anyway), no
  DFlash2 / capture / DCP (context < DCP_AUTO), profiling off.

Also written for the Zig side: ``inv_freq.bin`` (torch's bytes), ``nccl.txt`` (the libnccl loaded), ``blas.json``
(the libcublas file torch loaded, its workspace and math settings, torch's preferred BLAS backend, the experts
implementation), ``bmm_probe.json`` (sha256 of torch.bmm on attention_core's wide absorb / expand shapes over fixed
inputs: tf-glm53-generate --bmm-probe checks cuBLAS gives the same bits before any prompt runs) and, with --record,
the Triton AOT set of everything the served engine launched.

Profiling (run_phase3a_cluster.sh PROFILE=1): ``--profile-prefill NAME`` points TF_GLM53_PROFILE_FLAG at
<out>/prof/PROFILE and touches its ``_PREFILL`` flag before the first run of the prompt named NAME*, so
Runner.prefill wraps that prompt's prefill in torch.profiler (its own per-kernel table, prof/prof/
prefill_rank<R>.txt); this tool then prints the table rolled up into the phases tf-glm53-generate --prompt-prof
times (``[ref3a] PREFILL-PROF ...``). Only that run's prefill is profiled (and slowed): the gate takes each prompt's
best run.

Modes:
  --make-prompts OUT   (CPU) reference2b's prompts (3 chat prompts, the ~4,500-token one, the 32K prose prompt) plus
                       the 128K prefill prompt (the 32K background text four times over); runs per prompt from
                       --runs / --runs-long / --runs-128k
  (default)            one rank: --rank R --master IP --port P --model DIR --prompts FILE --out DIR [--context N]
                       [--k 2] [--record]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference import nccl_path, nccl_version, pack  # noqa: E402

FROZEN_3A = {
    "TF_EXL3_PROMPT_DET": "slots16",
    "TF_GLM53_COPY_DRAFTS": "0",
    "TF_GLM53_PROMPT_REUSE": "0",
    "TF_GLM53_CAPTURE_LABEL": "0",
    "TF_GLM53_PROFILE_FLAG": "/tmp/tf-glm53-no-profile/PROFILE",
}
UNSET_3A = ("TF_GLM53_DFLASH", "TF_GLM53_CAPTURE_DIR", "TF_GLM53_DCP", "TF_GLM53_DEFAULT_DRAFT", "TF_GLM53_TILES")
# the served knobs the Zig engine mirrors; recorded so a changed image default shows in the comparison
WATCHED = ("TF_GLM53_ROCE", "TF_GLM53_PROMPT_SP", "TF_GLM53_PROMPT_ROWS", "TF_GLM53_PROMPT_ROWS_SHORT",
           "TF_GLM53_PREFILL_REDUCE", "TF_GLM53_PROMPT_OVERLAP", "TF_GLM53_SP_SELECT", "TF_GLM53_TUNE",
           "TF_GLM53_TUNE_SHARED", "TF_GLM53_SIDE", "TF_GLM53_L2PF", "TF_GLM53_EXPERTS", "TF_GLM53_RADIX",
           "TF_GLM53_RADIX_MIN_ROWS", "TF_GLM53_ATTN_PROMPT", "TF_GLM53_FOLD_SHARED", "TF_GLM53_INDEX_SPLIT",
           "TF_GLM53_UNPACK_CACHE_MB", "TF_GLM53_MTP", "TF_GLM53_MTP_REUSE", "TF_GLM53_DRAFT_HEAD",
           "TF_GLM53_DRAFT_VOCAB", "TF_GLM53_VERIFY_HEAD", "TF_GLM53_SHARDED_SAMPLE", "TF_GLM53_FAST_GATHER",
           "TF_GLM53_ROCE_HEALTH", "TF_GLM53_ROCE_HCA", "TF_GLM53_ROCE_GIDS", "TF_GLM53_RAILS", "TF_EXL3_LINEAR_LOADS",
           "TF_EXL3_EXPERTS_LOADS", "TF_EXL3_EXPERTS_FUSE", "TF_EXL3_PROMPT_EXPERTS", "TF_EXL3_PROMPT_F16_ACC",
           "TF_GLM53_DEPTH_POLICY", "CUBLAS_WORKSPACE_CONFIG", "TORCH_BLAS_PREFER_CUBLASLT", "NCCL_PROTO", "NCCL_ALGO",
           "NCCL_MAX_NCHANNELS", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX")

# attention_core's wide shapes at TP4 (16 heads a rank): a 4,500-token prompt's halves, a 32K chunk's half, the MTP
# rows of a full 8192-row chunk
PROBE_ROWS = (1125, 4088, 8177)


def probe_tensor(n: int, salt: int):
    """tf-glm53-generate probeFill: element i = bf16(((h >> 8) % 4001 - 2000) / 2000), h = (i + 1000003 salt) *
    2654435761 mod 2^32."""
    import numpy as np
    import torch

    i = np.arange(n, dtype=np.uint64)
    h = ((i + np.uint64(salt * 1000003)) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    v = (((h >> np.uint64(8)) % np.uint64(4001)).astype(np.int64) - 2000).astype(np.float32) / np.float32(2000.0)
    return torch.from_numpy(v).to(torch.bfloat16)


def bmm_probe(heads: int, nope: int, rope: int, lw: int, vd: int) -> dict:
    """torch.bmm exactly as fused.attention_core's wide path calls it (the same views), hashed."""
    import torch

    qd = nope + rope
    cases = []
    for R in PROBE_ROWS:
        q = probe_tensor(R * heads * qd, 1).cuda().view(R, heads * qd)
        wk = probe_tensor(heads * nope * lw, 2).cuda().view(heads, nope, lw)
        qn = q.view(R, heads, qd)[:, :, :nope].transpose(0, 1)
        out = torch.bmm(qn, wk).contiguous()
        cases.append({"kind": "absorb", "H": heads, "R": R, "nope": nope, "qd": qd, "lw": lw, "vd": vd,
                      "sha256": hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()})
        ol = probe_tensor(R * heads * lw, 3).cuda().view(R, heads, lw)
        wv = probe_tensor(heads * vd * lw, 4).cuda().view(heads, vd, lw)
        out = torch.bmm(ol.transpose(0, 1), wv.transpose(1, 2)).contiguous()
        cases.append({"kind": "expand", "H": heads, "R": R, "nope": nope, "qd": qd, "lw": lw, "vd": vd,
                      "sha256": hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()})
    torch.cuda.synchronize()
    return {"cases": cases}


# torch.profiler kernel names -> the phases tf-glm53-generate --prompt-prof reports (first match wins)
PY_PHASES = (
    ("routed experts (prompt_experts)", ("prompt_gateup", "prompt_down", "slot_sum16", "tf_exl3p")),
    ("MTP experts (cuda-exl3 GEMM)", ("exl3_moe", "moe_align", "cuda_exl3")),
    ("experts, decode kernel", ("grouped_kernel", "gateup_epilogue", "tf_exl3x", "group_kernel")),
    ("dense linears (prompt GEMM + unpack)", ("unpack_kernel", "rot_in_kernel", "tf_exl3_linear")),
    ("attention", ("_attn_chunks", "_merge", "_attn_")),
    ("indexer (scores, top-k, keys)", ("_index_scores", "_select_keys", "_ik_write", "_iq_rope", "topk", "sort",
                                       "radix")),
    ("collectives (NCCL)", ("nccl",)),
    ("router", ("_router", "_topk")),
    ("cuBLAS bmm (absorb / expand)", ("gemm", "cutlass", "xmma", "sm90", "sm100")),
)


def roll_up(path: Path) -> list[str]:
    """Runner.prefill's profile table (wall, GPU busy, the top kernels) rolled up into PY_PHASES."""
    lines = path.read_text().splitlines()
    if not lines:
        return []
    sums: dict[str, float] = {}
    counted = 0.0
    for ln in lines[1:]:
        parts = ln.split(None, 3)
        if len(parts) < 4 or parts[1] != "s":
            continue
        t, name = float(parts[0]), parts[3]
        counted += t
        if name == "_gemm":
            phase = "dense linears (prompt GEMM + unpack)"
        else:
            phase = next((ph for ph, keys in PY_PHASES if any(k in name for k in keys)), "glue (norms, adds, copies)")
        sums[phase] = sums.get(phase, 0.0) + t
    out = [lines[0]]
    out += [f"{v * 1e3:10.1f} ms  {k}" for k, v in sorted(sums.items(), key=lambda kv: -kv[1])]
    busy = None
    for tok in lines[0].split("gpu kernels")[1:]:
        try:
            busy = float(tok.split()[0])
        except (ValueError, IndexError):
            pass
    if busy is not None and busy > counted:
        out.append(f"{(busy - counted) * 1e3:10.1f} ms  (kernels outside the top 40)")
    return out


def loaded_lib(fragment: str) -> str:
    """The file of a loaded shared library whose name contains ``fragment`` (/proc/self/maps)."""
    try:
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 6 and fragment in os.path.basename(parts[5]):
                return os.path.realpath(parts[5])
    except OSError:
        pass
    return ""


def make_prompts(a) -> int:
    import reference2b as r2b

    rc = r2b.make_prompts(a)
    if rc or not (a.context_file and a.tokens_128k):
        return rc
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    model = Path(a.model)
    tmpl = ChatTemplate(model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    text = Path(a.context_file).read_text()
    body = "Background reading:\n" + "\n\n".join([text] * 4) + "\n\nTask:\n" + r2b.BEEKEEPER
    rendered = tmpl.render([{"role": "user", "content": body}], tools=None, enable_thinking=True)
    ids = [int(t) for t in tok.encode(rendered, add_special_tokens=False).ids]
    doc = json.loads(Path(a.make_prompts).read_text())
    doc["prompts"].append({"name": "long128k", "ids": ids,
                           "text_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "tokens": a.tokens_128k,
                           "runs": [r for r in a.runs_128k.split(",") if r], "seed": r2b.BENCH_SEED,
                           "seed_from_prompt": False})
    doc["context_needed"] = max(len(p["ids"]) + p["tokens"] for p in doc["prompts"])
    doc["generator"] = "tools/glm53/reference3a.py --make-prompts"
    Path(a.make_prompts).write_text(json.dumps(doc) + "\n")
    print(f"[ref3a] + long128k: {len(ids)} ids x {a.runs_128k}; context needed {doc['context_needed']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--make-prompts", default="")
    ap.add_argument("--long", type=int, default=4500)
    ap.add_argument("--context-file", default="")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--tokens-long", type=int, default=512)
    ap.add_argument("--tokens-128k", type=int, default=32)
    ap.add_argument("--runs", default="g:2,s20:2,s20:0")
    ap.add_argument("--runs-long", default="s20:2,s20:0")
    ap.add_argument("--runs-128k", default="g:2,g:0")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29571)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--speed-only", action="store_true", help="no Triton record, no probes: timing at the env given")
    ap.add_argument("--tools", default="")
    ap.add_argument("--profile-prefill", default="", help="torch.profiler over the first run's prefill of this prompt")
    a = ap.parse_args()
    if a.make_prompts:
        return make_prompts(a)
    if not (a.prompts and a.out):
        ap.error("--prompts and --out are needed for a run")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = {}
    frozen = dict(FROZEN_3A)
    if a.speed_only:
        frozen.pop("TF_EXL3_PROMPT_DET")          # the image default (fp32 red.add): speed only
    for k, v in frozen.items():
        os.environ[k] = v
        env[k] = v
    for k in UNSET_3A:
        os.environ.pop(k, None)
    if not a.speed_only:
        os.environ["TF_GLM53_TILES"] = env["TF_GLM53_TILES"] = f"save:{out}/tiles-r{a.rank}.json"
    prof_flag = None
    if a.profile_prefill:                       # Runner.prefill's own profile (read once, at import: set it first)
        (out / "prof").mkdir(parents=True, exist_ok=True)
        prof_flag = str(out / "prof" / "PROFILE")
        os.environ["TF_GLM53_PROFILE_FLAG"] = env["TF_GLM53_PROFILE_FLAG"] = prof_flag
    tree = Path(a.tools) if a.tools else HERE.parents[1]
    rec = None
    if a.record:
        if not os.environ.get("TRITON_CACHE_DIR"):
            raise SystemExit("--record needs TRITON_CACHE_DIR (a fresh directory: the cubins are packed from it)")
        sys.path.insert(0, str(tree / "tools" / "zig"))
        import triton_aot_manifest as aotm

        rec = aotm.Recorder().install()

    import torch

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.glm_moe_dsa.cuda import engine as engine_mod
    from tensorfold.families.glm_moe_dsa.cuda import fused, prefixes, weights
    from tensorfold.families.glm_moe_dsa.cuda.engine import Glm53Engine

    from reference2b import parse_run

    doc_in = json.loads(Path(a.prompts).read_text())
    prompts = doc_in["prompts"]
    sset = doc_in.get("sampling", {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0})
    context = a.context or int(doc_in.get("context_needed", 32768))
    torch.cuda.set_device(0)
    t0 = time.time()
    eng = Glm53Engine(Path(a.model), rank=a.rank, master=a.master, port=a.port, context=context, mtp_drafts=a.k)
    init_s = time.time() - t0
    rn = eng.runner
    w = rn.w
    if w.dcp != 1:
        raise SystemExit(f"reference: DCP {w.dcp} at context {context} (Phase 3b); keep the context under the DCP cut")
    roce_on = w.fast is not None
    print(f"[ref3a] rank {a.rank}: served engine up in {init_s:.0f}s (load {eng.load_s:.0f}s): RoCE "
          f"{'one-shot' if roce_on else 'OFF (NCCL)'}, experts {weights.EXPERTS_IMPL}, prompt rows {fused.PROMPT_ROWS}/"
          f"{fused.PROMPT_ROWS_SHORT}, SP {fused.PROMPT_SP}, ring {fused.PREFILL_REDUCE}, prompt det "
          f"{os.environ.get('TF_EXL3_PROMPT_DET', '0')}", flush=True)
    lib_path, lib_ver = nccl_path(eng.comm.lib), nccl_version(eng.comm.lib)
    (out / "nccl.txt").write_text(f"{lib_path}\n{lib_ver}\n")
    (out / "inv_freq.bin").write_bytes(w.inv.contiguous().cpu().numpy().tobytes())
    if not a.speed_only:
        probe = bmm_probe(w.heads, w.cfg.qk_nope_head_dim, w.cfg.qk_rope_head_dim, w.cfg.kv_lora_rank,
                          w.cfg.v_head_dim)
        (out / "bmm_probe.json").write_text(json.dumps(probe, indent=1) + "\n")
    blas = {"libcublas": loaded_lib("libcublas.so"), "libcublasLt": loaded_lib("libcublasLt.so"),
            "workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
            "preferred_blas": str(torch.backends.cuda.preferred_blas_library()),
            "allow_bf16_reduced_precision": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "experts_impl": weights.EXPERTS_IMPL,
            "torch": torch.__version__, "capability": list(torch.cuda.get_device_capability())}
    (out / "blas.json").write_text(json.dumps(blas, indent=1) + "\n")

    def zero_state() -> None:
        st = rn.st
        for t in list(st.kc) + list(st.ic.values()) + [x for x in (st.mkc, st.mic) if x is not None]:
            t.zero_()
        st.pos.zero_()
        st.mpos.zero_()
        rn.carry.zero_()
        torch.cuda.synchronize()

    results = []
    profiled = False
    with torch.no_grad():
        for p in prompts:
            ids = [int(x) for x in p["ids"]]
            for run in p["runs"]:
                prof_this = bool(prof_flag) and not profiled and p["name"].startswith(a.profile_prefill)
                if prof_this:
                    Path(prof_flag + "_PREFILL").touch()
                greedy, top_k, k = parse_run(run)
                if k > a.k:
                    raise SystemExit(f"run {run}: more drafts than --k {a.k}")
                s = None if greedy else Sampling(int(p["seed"]), float(sset["temperature"]), top_k,
                                                 float(sset["top_p"]), float(sset.get("min_p", 0.0)))
                sample = None if s is None else (lambda lg, pos, s=s: eng._sample(lg, pos, s))
                zero_state()
                st = rn.generate(ids, int(p["tokens"]), sample, lambda t: False, lambda toks: None, k, mode=None,
                                 sampling=s, copies=False)
                toks = [int(t) for t in st["out"]]
                pf = len(ids) / st["prefill_s"] if st["prefill_s"] > 0 else 0.0
                print(f"[ref3a] rank {a.rank} {p['name']} {run}: prompt {len(ids)} in {st['prefill_s']:.2f}s "
                      f"({pf:.0f} tok/s); {len(toks)} tokens, decode {st['tok_s']:.2f} tok/s, {st['tokens_per_round']} "
                      f"tokens/round ({st['rounds']} rounds), accept {st['accept']}; first {toks[:8]}", flush=True)
                if prof_this:
                    profiled = True
                    table = Path(prof_flag).parent / "prof" / f"prefill_rank{a.rank}.txt"
                    if table.exists():
                        for ln in roll_up(table):
                            print(f"[ref3a] PREFILL-PROF rank {a.rank} {p['name']} {run}: {ln}", flush=True)
                    else:
                        print(f"[ref3a] PREFILL-PROF rank {a.rank}: no table at {table}", flush=True)
                results.append({"name": p["name"], "run": run, "prompt_len": len(ids), "seed": str(p["seed"]),
                                "k": k, "greedy": greedy, "top_k": top_k, "prefill_s": round(st["prefill_s"], 3),
                                "decode_s": round(st["decode_s"], 3), "tok_s": st["tok_s"], "rounds": st["rounds"],
                                "tokens_per_round": st["tokens_per_round"], "accept": st["accept"] or 0.0,
                                "capture_s": st["capture_s"], "graphs": st["graphs"], "tokens": toks})
        torch.cuda.synchronize()
    eng.comm.barrier()

    watched = {k: os.environ.get(k) for k in WATCHED}
    doc = {"tool": "tools/glm53/reference3a.py", "mode": "3a", "speed_only": a.speed_only, "rank": a.rank,
           "world": engine_mod.WORLD, "layers": len(w.layers), "context": context, "capacity": rn.capacity,
           "window": 128, "k": a.k, "vocab_off": w.vocab_off, "vocab_part": w.vocab_part, "nccl_version": lib_ver,
           "nccl_lib": lib_path, "env": env, "watched_env": watched, "roce": roce_on,
           "roce_hcas": getattr(w.fast, "hcas", None) if roce_on else None, "experts_impl": weights.EXPERTS_IMPL,
           "prompt_rows": fused.PROMPT_ROWS, "prompt_rows_short": fused.PROMPT_ROWS_SHORT,
           "prompt_sp": fused.PROMPT_SP, "prefill_reduce": fused.PREFILL_REDUCE, "tune": fused.TUNE,
           "side": fused.SIDE, "unpack_cache_mb": int(os.environ.get("TF_GLM53_UNPACK_CACHE_MB", "384")),
           "prompt_reuse": bool(prefixes.enabled()) if hasattr(prefixes, "enabled") else None,
           "tiles": f"{out}/tiles-r{a.rank}.json" if not a.speed_only else None, "blas": blas,
           "torch": torch.__version__, "load_s": round(eng.load_s, 1), "init_s": round(init_s, 1),
           "prewarm_graphs": len(rn.G.graphs), "sampling": sset,
           "inv_freq_sha256": hashlib.sha256((out / "inv_freq.bin").read_bytes()).hexdigest(), "runs": results}
    try:
        import triton

        doc["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    (out / f"ref-r{a.rank}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[ref3a] rank {a.rank}: wrote {out / f'ref-r{a.rank}.json'}", flush=True)
    if rec is not None:
        from oracle import specialization

        rec.dump(out / "launches.json")
        (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
        pack(tree, out, Path(os.environ["TRITON_CACHE_DIR"]))
    print("[ref3a] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
