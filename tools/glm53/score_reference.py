#!/usr/bin/env python3
"""The reference of the as-served scorer: a model's own next-token distributions over a rows file.

  score_reference.py --model BF16_DIR --rows rows.bin --out reference.bin [--topk 4096] [--exllamav3 DIR] [--device cuda:0]

Runs `--model` (the unquantized GLM-5.3, or any checkpoint exllamav3 loads) over every row of score_rows.py's file
the way exllamav3's eval/model_diff.py runs its two models: one module at a time over all rows, so one layer's
weights are on the device at once. At the last module each position keeps its --topk most likely tokens (id and
log-probability at temperature 1 over the whole vocabulary), the mass of the rest, the real next token's
log-probability and the entropy (score_format.py's records). tf-glm53-generate --mode score reads the result.

Needs the exllamav3 tree the checkpoint was quantized with (--exllamav3, default ~/glm53-requant/exllamav3) and a
GPU. Memory: one layer's weights plus every row's hidden state ([tokens, hidden] a row).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_format as sf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=4096)
    ap.add_argument("--exllamav3", default=os.path.expanduser("~/glm53-requant/exllamav3"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--block", type=int, default=256, help="positions a pass at the last module")
    a = ap.parse_args()

    sys.path.insert(0, a.exllamav3)
    import torch
    from exllamav3 import Config, Model
    from exllamav3.util.memory import free_mem

    src = sf.read(a.rows)
    vocab = src["vocab"]
    if not 0 < a.topk <= vocab:
        sys.exit(f"--topk 1..{vocab}")
    device = torch.device(a.device)
    config = Config.from_directory(a.model)
    config.override_dynamic_seq_len(max(len(r) for r in src["rows"]))
    model = Model.from_config(config)
    ids = [torch.from_numpy(r.astype("int64")).unsqueeze(0) for r in src["rows"]]
    params = [{} for _ in ids]
    writer = sf.ReferenceWriter(a.rows, a.out, a.topk)
    t0 = time.time()

    with torch.inference_mode():
        states = [model.prepare_inputs(x, p) for x, p in zip(ids, params)]
        for mi, module in enumerate(model.modules):
            last = module is model.modules[-1]
            config.stc.begin_deferred_load()
            module.load(device if not module.caps.get("prefer_cpu") else "cpu")
            config.stc.end_deferred_load()
            for b in range(len(states)):
                params[b]["dev_cache"] = None
                x = module.prepare_for_device(states[b], params[b])
                x = module.forward(x, params[b]).clone()
                if not last:
                    states[b] = x
                    continue
                states[b] = None
                logits = x.view(-1, x.shape[-1])
                if logits.shape[-1] != vocab:
                    sys.exit(f"the model gives {logits.shape[-1]} logits, the rows file says {vocab}")
                n = logits.shape[0] - 1                      # the last token predicts nothing the row holds
                labels = ids[b][0, 1:].to(logits.device)
                parts = []
                for s in range(0, n, a.block):
                    e = min(s + a.block, n)
                    lp = torch.log_softmax(logits[s:e].double(), dim=-1)
                    top, idx = torch.topk(lp, a.topk, dim=-1)
                    p = lp.exp()
                    tail = (1.0 - top.exp().sum(-1)).clamp_min(0) if a.topk < vocab else torch.zeros(e - s, device=lp.device, dtype=lp.dtype)
                    entropy = -(p * lp).sum(-1)
                    label_lp = lp.gather(1, labels[s:e].unsqueeze(1)).squeeze(1)
                    parts.append(sf.position_records(top.float().cpu().numpy(), idx.cpu().numpy(), tail.cpu().numpy(),
                                                     label_lp.cpu().numpy(), entropy.cpu().numpy()))
                writer.add_row(b"".join(parts))
                del x, logits
            module.unload()
            config.stc.close()
            free_mem()
            print(f"module {mi + 1} of {len(model.modules)} ({module.key}) in {time.time() - t0:.0f} s", flush=True)
    writer.close()
    print(f"wrote {a.out}: {len(ids)} rows, {writer.n_pos} positions, {a.topk} of {vocab} tokens kept a position")


if __name__ == "__main__":
    main()
