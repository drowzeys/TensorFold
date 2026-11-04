#!/usr/bin/env python3
"""The oracle's prompt pass (backbone + tower prefill) on a dump's prompt rows in bf16 (as dumped) and again with
every dense tensor and activation in fp32 (TF32 off; the routed experts keep SGLang's FP8 weights and activation
rounding). Writes OUT/vectors.safetensors in the dump schema with the fp32 results (for drama3-generate --check)
and prints, per row, how far the oracle's own bf16 rows are from fp32.
usage: oracle_fp32.py DUMP_DIR OUT_DIR"""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, "/drama3/oracle")
import drama3_ref as R  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def prompt_pass(m, rows):
    dev = m.dev
    L = rows.shape[0]
    st = m.new_state(L + 4)
    rec = {}
    pos = torch.arange(L, device=dev)
    rd = rows.to(dev)
    hn, _ = m.backbone(rd, pos, st, rec)
    rec["backbone_final_norm"] = hn
    m.tower_prefill(hn[:-1], pos[:-1], rd[:-1], rd[1:], st, rec)
    return {k: v.float().cpu() for k, v in rec.items()}


def rowcos(a, b):
    return torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=-1)


def main():
    dump, out = sys.argv[1], sys.argv[2]
    v = load_file(os.path.join(dump, "vectors.safetensors"))
    rows = v["prompt_rows"]
    m = R.Drama3("/w")
    b16 = prompt_pass(m, rows)
    for k in list(m.W):
        m.W[k] = m.W[k].float()
    R.BF16 = torch.float32
    f32 = prompt_pass(m, rows)
    names = [f"backbone_hidden.layer{i:02d}" for i in range(34, 40)] + ["backbone_final_norm"] + [f"tower_prefill.layer{j:02d}" for j in range(13)]
    for n in names:
        c = rowcos(b16[n], f32[n])
        worst = torch.argsort(c)[:4].tolist()
        print(f"{n:28s} oracle bf16 vs fp32: mean cos {c.mean():.6f}  worst rows " + ", ".join(f"{r}:{c[r]:.4f}" for r in worst))
    os.makedirs(out, exist_ok=True)
    keep = {k: f32[k].contiguous() for k in f32}
    keep["prompt_rows"] = rows
    keep["frames_all"] = v["frames_all"]
    for k in ("decode.candidate_logits", "decode.backbone_final_norm"):
        if k in v:
            keep[k] = v[k]
    save_file(keep, os.path.join(out, "vectors.safetensors"))
    save_file({k: b16[k].contiguous() for k in b16}, os.path.join(out, "oracle_bf16.safetensors"))
    print("written", out)


if __name__ == "__main__":
    main()
