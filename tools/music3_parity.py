"""MiniMax Music 3: MLX vs a ComfyUI reference run (tools/ref_comfy.py dumps), stage by stage.

    python tools/music3_parity.py REF_DIR [--model DIR] [--caption-file C --lyrics-file L] [--stages ar,dit,dav,probe]

ar     teacher forcing with the reference's drawn codes: code-0 logits (both rows) and depth logits, cosine and
       argmax agreement per frame; the frame states the DiT reads (8 x 4096 a frame), cosine; prompt ids equal.
dit    the reference's Euler run redone from its own noise and frame states: final latent cosine.
probe  one DiT forward on fixed inputs at sigma 1.0 / 0.5 / 0.1, both branches: cosine, relative error.
dav    the reference's latent decoded: waveform cosine and SNR against the reference's audio.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def snr_db(ref, test) -> float:
    ref, test = np.asarray(ref, np.float64), np.asarray(test, np.float64)
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - test) ** 2).sum(), 1e-30)))


def rows_cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def summary(values) -> dict:
    v = np.asarray(values, np.float64)
    return {"mean": round(float(v.mean()), 6), "min": round(float(v.min()), 6), "p01": round(float(np.percentile(v, 1)), 6)}


def check_ar(gen, ref: Path, caption: str, lyrics: str, report: dict) -> None:
    from tensorfold.families.music3 import prompt as P
    from tensorfold.families.music3.generate import Settings

    ids = P.encode(gen.tokenizer, caption, lyrics)
    ref_ids = json.loads((ref / "ids.json").read_text())["ids"]
    report["prompt_ids_equal"] = ids == ref_ids
    codes = np.load(ref / "codes.npy")
    hidden = np.load(ref / "hidden.npy")
    frames = hidden.shape[0]
    trace: dict = {}
    started = time.perf_counter()
    states, _, _ = gen.ar_frames(caption, lyrics, Settings(seconds=frames / 25 + 1), forced_codes=codes[:frames + 1],
                                 trace=trace)
    report["ar_forced_s"] = round(time.perf_counter() - started, 2)
    mine = np.array(states.astype(mx.float32))
    n = min(len(mine), frames)
    per = rows_cos(mine[:n], hidden[:n])
    layer = np.stack([rows_cos(mine[:n, i * 4096:(i + 1) * 4096], hidden[:n, i * 4096:(i + 1) * 4096])
                      for i in range(8)])
    report["frame_states"] = {"frames": [int(len(mine)), int(frames)], "cos": summary(per),
                              "cos_by_part": [round(float(x), 6) for x in layer.mean(axis=1)]}
    if (ref / "c0_logits.npy").exists():
        ref_c0 = np.load(ref / "c0_logits.npy").astype(np.float32)               # [F, 2, 16385]
        my_c0 = np.stack([np.array(x) for x in trace["c0_logits"]])
        m = min(len(ref_c0), len(my_c0))
        c = rows_cos(my_c0[:m, 0], ref_c0[:m, 0])
        cu = rows_cos(my_c0[:m, 1], ref_c0[:m, 1])
        agree = float((my_c0[:m, 0].argmax(-1) == ref_c0[:m, 0].argmax(-1)).mean())
        g_ref = ref_c0[:m, 1] + (ref_c0[:m, 0] - ref_c0[:m, 1]) * 1.5
        g_my = my_c0[:m, 1] + (my_c0[:m, 0] - my_c0[:m, 1]) * 1.5
        top50 = np.mean([len(set(np.argsort(-a)[:50]) & set(np.argsort(-b)[:50])) / 50 for a, b in
                         zip(my_c0[:m, 0], ref_c0[:m, 0])])
        forced = codes[:m, 0] + 1
        hit = float((g_my.argmax(-1) == g_ref.argmax(-1)).mean())
        report["c0_logits"] = {"frames": m, "cos_cond": summary(c), "cos_uncond": summary(cu),
                               "greedy_agree_cond": round(agree, 4), "greedy_agree_guided": round(hit, 4),
                               "top50_overlap": round(float(top50), 4),
                               "ref_drawn_code_rank_mine_median": int(np.median(
                                   [(row > row[f]).sum() for row, f in zip(g_my, forced)]))}
    if (ref / "depth_logits.npy").exists():
        ref_d = np.load(ref / "depth_logits.npy").astype(np.float32)             # [F, 7, 2, 1024]
        my_d = np.stack([np.array(x) for x in trace["depth_logits"]]).reshape(-1, 7, 2, 1024)
        m = min(len(ref_d), len(my_d))
        report["depth_logits"] = {
            "cos_cond_by_codebook": [round(float(rows_cos(my_d[:m, i, 0], ref_d[:m, i, 0]).mean()), 6) for i in range(7)],
            "greedy_agree_by_codebook": [round(float((my_d[:m, i, 0].argmax(-1) == ref_d[:m, i, 0].argmax(-1)).mean()), 4)
                                         for i in range(7)]}


def check_dit(gen, ref: Path, report: dict, steps: int, cfg: float) -> None:
    from tensorfold.families.music3.generate import Settings

    hidden = mx.array(np.load(ref / "hidden.npy")).astype(mx.bfloat16)
    noise = mx.array(np.load(ref / "noise.npy")[0].T)
    started = time.perf_counter()
    latent = gen.denoise(hidden, Settings(steps=steps, dit_cfg=cfg), noise=noise)
    report["dit_s"] = round(time.perf_counter() - started, 2)
    ref_latent = np.load(ref / "latent.npy")[0].T
    mine = np.array(latent)
    report["dit_final_latent"] = {"cos": round(cos(mine, ref_latent), 6), "snr_db": round(snr_db(ref_latent, mine), 2)}
    np.save(ref / "mlx_latent_from_ref.npy", mine)


def check_probe(gen, ref: Path, report: dict) -> None:
    dit = gen.dit
    hidden = mx.array(np.load(ref / "probe_hidden.npy")).astype(gen.dit_dtype)
    x = mx.array(np.load(ref / "probe_x.npy").T).astype(gen.dit_dtype)[None]
    outs = np.load(ref / "probe_dit.npz")
    cond = dit.condition(hidden)[None]
    null = dit.null_condition(cond.shape[1], gen.dit_dtype)[None]
    tables = dit.rope_tables(cond.shape[1] + 1)
    res = {}
    for sigma in (1.0, 0.5, 0.1):
        t = mx.array([1.0 - sigma], dtype=mx.float32)
        c = np.array(dit(x, t, cond, tables).astype(mx.float32))[0].T
        u = np.array(dit(x, t, null, tables).astype(mx.float32))[0].T
        rc, ru = outs[f"cond_{sigma}"], outs[f"uncond_{sigma}"]
        res[str(sigma)] = {"cos_cond": round(cos(c, rc), 6), "cos_uncond": round(cos(u, ru), 6),
                           "rel_err_cond": round(float(np.linalg.norm(c - rc) / np.linalg.norm(rc)), 5),
                           "cos_guided_1.7": round(cos(u + (c - u) * 1.7, ru + (rc - ru) * 1.7), 6)}
    report["dit_probe"] = res
    if (ref / "probe_dav.npy").exists():
        lat = mx.array(np.load(ref / "probe_latent.npy").T)
        mine = np.array(gen.dav(lat[None])[0])
        r = np.load(ref / "probe_dav.npy")
        report["dav_probe"] = {"cos": round(cos(mine, r), 7), "snr_db": round(snr_db(r, mine), 2)}


def check_dav(gen, ref: Path, report: dict) -> None:
    latent = mx.array(np.load(ref / "latent.npy")[0].T)
    started = time.perf_counter()
    mine = gen.decode(latent)
    report["dav_s"] = round(time.perf_counter() - started, 2)
    r = np.load(ref / "audio.npy")
    n = min(mine.shape[1], r.shape[1])
    report["dav_waveform"] = {"samples": [int(mine.shape[1]), int(r.shape[1])], "cos": round(cos(mine[:, :n], r[:, :n]), 7),
                              "snr_db": round(snr_db(r[:, :n], mine[:, :n]), 2)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ref", type=Path)
    p.add_argument("--model", default="~/studio/models/MiniMax-Music3-Comfy")
    p.add_argument("--caption-file", type=Path)
    p.add_argument("--lyrics-file", type=Path)
    p.add_argument("--stages", default="ar,probe,dit,dav")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dit-cfg", type=float, default=1.7)
    p.add_argument("--dit-dtype", default="float16")
    p.add_argument("--comfy-rope", action="store_true", help="round DiT rope angles to the DiT dtype (as ComfyUI)")
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    from tensorfold.families.music3.generate import Music3Generator

    stages = a.stages.split(",")
    gen = Music3Generator(a.model, dit_dtype=getattr(mx, a.dit_dtype), load_ar="ar" in stages,
                          load_dit=bool({"dit", "probe"} & set(stages)), load_dav=bool({"dav", "probe"} & set(stages)))
    if gen.dit is not None and a.comfy_rope:
        gen.dit.angle_dtype = gen.dit_dtype
    report: dict = {"ref": str(a.ref), "dit_dtype": a.dit_dtype, "comfy_rope": a.comfy_rope}
    if "ar" in stages:
        check_ar(gen, a.ref, a.caption_file.read_text(), a.lyrics_file.read_text(), report)
    if "probe" in stages:
        check_probe(gen, a.ref, report)
    if "dit" in stages:
        check_dit(gen, a.ref, report, a.steps, a.dit_cfg)
    if "dav" in stages:
        check_dav(gen, a.ref, report)
    text = json.dumps(report, indent=1)
    print(text)
    if a.report:
        a.report.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
