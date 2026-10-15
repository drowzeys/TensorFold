"""Drama 3 MLX vs the PyTorch reference dumps (drama3_ref.py greedy runs; schema in the oracle's DUMPS.md).

    python tools/drama3_parity.py --model ~/models/drama-3-mlx --dump dumps/plain [--dump dumps/clone] [--json out.json]

Checks: prompt rows; reference audio codes (when the dump has them); backbone input embeddings and every layer's
hidden state; the final norm and candidate logits; MoE layer-0 routing; tower prefill layers; the first decode step's
tower layers and head logits; teacher-forced decode rows (hidden cosine, whether our argmax picks the oracle's
token, and the margins); a free-running greedy run against the oracle's frames, with the first divergence explained
by both sides' logit margins; and codec waveform SNR on the oracle's frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np


def f64(x) -> np.ndarray:
    return np.asarray(x.astype(mx.float32) if isinstance(x, mx.array) else x, dtype=np.float64)


def cosine(a, b) -> dict:
    a, b = f64(a).reshape(len(f64(a)) if f64(a).ndim > 1 else 1, -1), f64(b)
    b = b.reshape(a.shape)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    rows = num / den
    flat = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {"cos": round(flat, 6), "min_row_cos": round(float(rows.min()), 6),
            "rel_err": round(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)), 5)}


def margin(logits: np.ndarray) -> float:
    top = np.sort(logits)[-2:]
    return float(top[1] - top[0])


def run(model_dir: str, dump_dir: Path, gen, frames_to_check: int, free_frames: int) -> dict:
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    from tensorfold.families.drama3.codec import load_audio
    from tensorfold.families.drama3.prompt import Reference

    model, c = gen.model, gen.model.config
    d = {k: v for k, v in mx.load(str(dump_dir / "vectors.safetensors")).items()}
    meta = json.loads((dump_dir / "meta.json").read_text())
    report: dict = {"dump": str(dump_dir), "text": meta["text"]}

    # -- prompt rows (and the reference encode) ------------------------------------------------------------------
    refs = []
    ref_texts = json.loads(meta.get("ref_text") or "[]")
    for i, transcript in enumerate(ref_texts):
        codes = np.asarray(d[f"ref{i}.codes"]).astype(np.int64)
        refs.append(Reference(transcript=transcript, codes=codes))
        if gen.codec is not None and f"ref{i}.audio_44k" in d:
            ours = np.array(gen.codec.encode(np.asarray(d[f"ref{i}.audio_44k"])))
            n = min(ours.shape[1], codes.shape[1])
            report[f"ref{i}_encode_match_per_codebook"] = [round(float((ours[j, :n] == codes[j, :n]).mean()), 3)
                                                           for j in range(10)]
            report[f"ref{i}_encode_frames"] = [int(ours.shape[1]), int(codes.shape[1])]
    rows = gen.rows(meta["text"], refs)
    oracle_rows = np.asarray(d["prompt_rows"]).astype(np.int64)
    report["prompt_rows_identical"] = bool(rows.shape == oracle_rows.shape and (rows == oracle_rows).all())
    report["prompt_rows"] = int(oracle_rows.shape[0])
    rows = oracle_rows                                    # everything below runs on the oracle's own rows

    # -- backbone, layer by layer --------------------------------------------------------------------------------
    bcache, tcaches = model.make_cache()
    frames = mx.array(rows.astype(np.int32))[None]
    h = model.embed(frames)
    report["input_embeds"] = cosine(d["backbone_input_embeds"], h[0])
    core = model.backbone.model
    fa_mask = create_attention_mask(h, bcache[core.fa_idx])
    ssm_mask = create_ssm_mask(h, bcache[core.ssm_idx])
    layers = []
    for i, (layer, cache) in enumerate(zip(core.layers, bcache)):
        if i == 0:
            normed = layer.input_layernorm(h)
            r = h + layer.linear_attn(normed, ssm_mask, None)     # a probe: leaves the cache alone
            gates = mx.softmax(layer.mlp.gate(layer.post_attention_layernorm(r)).astype(mx.float32), axis=-1)
            ours = np.argsort(-np.array(gates[0]), axis=-1)[:, :8]
            theirs = np.asarray(d["moe0.topk_ids"])
            report["moe0_topk_set_match"] = round(float(np.mean([len(set(a) & set(b)) / 8
                                                                  for a, b in zip(ours, theirs)])), 4)
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=cache)
        mx.eval(h)
        layers.append(cosine(d[f"backbone_hidden.layer{i:02d}"], h[0]))
    report["backbone_layers"] = layers
    report["backbone_layers_min_cos"] = min(x["cos"] for x in layers)
    report["backbone_layers_min_row_cos"] = min(x["min_row_cos"] for x in layers)
    hn = core.norm(h)
    report["final_norm"] = cosine(d["backbone_final_norm"], hn[0])
    logits = np.array(model.candidate_logits(hn[0, -1:]))[0]
    oracle_logits = np.asarray(d["tower_decode0.candidate_logits"])
    report["candidate_logits_last"] = cosine(oracle_logits, logits)
    report["candidate_argmax_same"] = bool(logits.argmax() == oracle_logits.argmax())

    # -- tower prefill -------------------------------------------------------------------------------------------
    next_frames = frames[:, 1:]
    semantic = model.is_semantic(next_frames[..., 0])
    extra = model.shared_extra(frames[:, :-1])
    t = hn[:, :-1]
    tower = []
    for i, layer in enumerate(model.tower.layers):
        codebook = 0 if i < c.shared_layers else i - c.shared_layers
        t = layer(t, next_frames[..., 1 + codebook], semantic, extra if i == 0 else None, tcaches[i])
        mx.eval(t)
        tower.append(cosine(d[f"tower_prefill.layer{i:02d}"], t[0]))
    report["tower_prefill_layers"] = tower
    report["tower_prefill_min_cos"] = min(x["cos"] for x in tower)

    # -- first decode step, teacher-forced to the oracle's frame ---------------------------------------------------
    oracle_frames = np.asarray(d["frames"]).astype(np.int64)
    first = oracle_frames[0]
    main = mx.array([int(first[0])], dtype=mx.int32)
    code0 = mx.array([int(first[1])], dtype=mx.int32)
    sem = model.is_semantic(main)[:, None]
    x = hn[:, -1:]
    extra = model.shared_extra(frames[:, -1:])
    report["decode0_shared_input"] = cosine(d["tower_decode0.shared_input"], extra[0, 0].astype(mx.bfloat16))
    dec_layers, heads, head_same, head_margins = [], [], [], []
    codes = [code0]
    for i, layer in enumerate(model.tower.layers):
        given = code0 if i < c.shared_layers else codes[-1]
        x = layer(x, given[:, None], sem, extra if i == 0 else None, tcaches[i])
        dec_layers.append(cosine(d[f"tower_decode0.layer{i:02d}"], x[0, 0]))
        if i >= c.shared_layers:
            r = i - c.shared_layers
            lg = np.array(model.tower.heads[r](x[:, -1]).astype(mx.float32))[0]
            ol = np.asarray(d[f"tower_decode0.head{r}_logits"])
            heads.append(cosine(ol, lg)["cos"])
            head_same.append(bool(lg.argmax() == ol.argmax()))
            head_margins.append(round(margin(ol), 4))
            codes.append(mx.array([int(first[2 + r])], dtype=mx.int32))   # teacher-forced
    report["tower_decode0_layers_min_cos"] = min(x["cos"] for x in dec_layers)
    report["tower_decode0_head_logit_cos"] = [round(v, 6) for v in heads]
    report["tower_decode0_head_argmax_same"] = head_same
    report["tower_decode0_head_oracle_margins"] = head_margins

    # -- teacher-forced decode rows -------------------------------------------------------------------------------
    hidden_cos, cand_cos, same, margins = [], [], [], []
    oracle_hidden = np.asarray(d["decode.backbone_final_norm"].astype(mx.float32))
    oracle_cand = np.asarray(d["decode.candidate_logits"])
    for n in range(1, min(frames_to_check, len(oracle_frames))):
        row = mx.array(oracle_frames[n - 1].astype(np.int32))[None, None]
        hrow = model.backbone_hidden(row, bcache)
        lg = np.array(model.candidate_logits(hrow[0, -1:]))[0]
        if n < len(oracle_hidden):
            hidden_cos.append(cosine(oracle_hidden[n], hrow[0, 0])["cos"])
            cand_cos.append(cosine(oracle_cand[n], lg)["cos"])
            same.append(bool(lg.argmax() == oracle_cand[n].argmax()))
            margins.append(round(margin(oracle_cand[n]), 3))
        # keep the tower caches in step with the oracle's frame
        fr = oracle_frames[n]
        mx.eval(model.tower_step(hrow, row, mx.array([int(fr[0])]), mx.array([int(fr[1])]), tcaches,
                                 lambda lg_, r_: mx.array([int(fr[2 + r_])], dtype=mx.int32)))
    report["teacher_forced_hidden_cos_min"] = round(min(hidden_cos), 6) if hidden_cos else None
    report["teacher_forced_candidate_cos_min"] = round(min(cand_cos), 6) if cand_cos else None
    report["teacher_forced_main_argmax_same"] = f"{sum(same)}/{len(same)}"
    decided = [s_ for s_, m_ in zip(same, margins) if m_ > 0]
    report["teacher_forced_main_argmax_same_where_margin_gt0"] = f"{sum(decided)}/{len(decided)}"
    report["teacher_forced_disagreements_oracle_margins"] = [m_ for s_, m_ in zip(same, margins) if not s_]
    report["teacher_forced_oracle_margins"] = margins[:40]

    # -- free-running greedy ------------------------------------------------------------------------------------
    from tensorfold.families.drama3.sampling import Sampling

    trace: dict = {}
    result = gen.generate_rows(rows, Sampling(greedy=True), max_frames=free_frames, trace=trace)
    all_oracle = np.asarray(d["frames_all"]).astype(np.int64) if "frames_all" in d else oracle_frames
    n = min(len(result.frames), len(all_oracle))
    equal_rows = (result.frames[:n] == all_oracle[:n]).all(axis=1)
    first_diff = int(np.argmin(equal_rows)) if not equal_rows.all() else None
    report["free_greedy_frames_compared"] = n
    report["free_greedy_identical_frames"] = int(equal_rows.sum())
    report["free_greedy_identical_prefix"] = n if first_diff is None else first_diff
    if first_diff is not None:
        ours, theirs = result.frames[first_diff], all_oracle[first_diff]
        col = int(np.argmax(ours != theirs))
        info = {"frame": first_diff, "column": col, "ours": int(ours[col]), "oracle": int(theirs[col])}
        if col == 0 and first_diff < len(trace["logits"]):
            lg = trace["logits"][first_diff]
            ids = np.array(model._candidate_ids)
            o_idx, m_idx = int(np.where(ids == theirs[0])[0][0]), int(np.where(ids == ours[0])[0][0])
            info["our_logit_gap_ours_minus_oracle_token"] = round(float(lg[m_idx] - lg[o_idx]), 4)
            if first_diff < len(oracle_cand):
                oc = oracle_cand[first_diff]
                info["oracle_logit_gap_oracle_minus_ours_token"] = round(float(oc[o_idx] - oc[m_idx]), 4)
        if col >= 2 and first_diff == 0:
            ol = np.asarray(d[f"tower_decode0.head{col - 2}_logits"])
            info["oracle_head_logit_gap_oracle_minus_ours_code"] = round(float(ol[theirs[col]] - ol[ours[col]]), 4)
        report["free_greedy_first_divergence"] = info
    report["free_greedy_finish"] = result.finish

    # -- codec ----------------------------------------------------------------------------------------------------
    if gen.codec is not None and "waveform" in d:
        ref = np.asarray(d["waveform"], dtype=np.float64)
        ours = np.array(gen.codec.decode(oracle_frames[:, 1:].T)).astype(np.float64)[: len(ref)]
        err = ref[: len(ours)] - ours
        report["codec_waveform_snr_db"] = round(float(10 * np.log10((ref ** 2).sum() / max((err ** 2).sum(), 1e-30))),
                                                2)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="~/models/drama-3-mlx")
    parser.add_argument("--dump", action="append", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20, help="teacher-forced decode rows to check")
    parser.add_argument("--free-frames", type=int, default=300, help="free-running greedy frames to compare")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    from tensorfold.families.drama3.generate import Drama3Generator

    gen = Drama3Generator(args.model)
    reports = []
    for dump in args.dump:
        report = run(args.model, dump, gen, args.frames, args.free_frames)
        reports.append(report)
        brief = {k: v for k, v in report.items() if k not in ("backbone_layers", "tower_prefill_layers")}
        print(json.dumps(brief, indent=1), flush=True)
    if args.json:
        args.json.write_text(json.dumps(reports, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
