#!/usr/bin/env python3
"""Prompts for the gate of several streams under DCP (tf-glm53-generate --mode 4b --dcp 4, docs/glm53-dcp-streams.md).

  dcps_prompts.py --model DIR --out FILE --parallel N [--tokens 96]

N prompts of unequal length through the chat template (thinking on): a short one (below index_topk = 2,048 tokens:
the stream attends every key), one just past index_topk (selection starts inside its decode), and long ones (several
selection buckets), so one window holds rows on both sides of the cut and rows whose streams sit in different key
buckets. Groups: every prompt together (greedy), every prompt together (sampled, keyed seeds), and pairs.
The filler is generated here (numbered inventory lines): nothing is downloaded.
"""
import argparse, hashlib, json, random
from pathlib import Path

ASK = {
    "short": "Write a vivid short story (at least 300 words) about a lighthouse keeper in 1890s Norway. Flowing prose only.",
    "edge": "From the inventory below, list the three sectors with the most crates of oak planks, with your working.",
    "long_a": "Summarise the inventory below in five sentences, then write a Python function that parses one of its lines.",
    "long_b": "The inventory below has one note for the auditor. Quote it, then explain in a paragraph why audits need notes.",
    "long_c": "Write a limerick about the inventory below, then name the entry with the largest crate count you can find.",
    "long_d": "Translate the first three entries of the inventory below into French, then write a short poem about warehouses.",
}
# filler lines a prompt (about 21 tokens a line): below the cut, just past it, then long ones of different buckets
LINES = {"short": 0, "edge": 93, "long_a": 420, "long_b": 900, "long_c": 250, "long_d": 1700}
GOODS = ["copper fittings", "glass jars", "wool blankets", "steel bolts", "paper reams", "ceramic tiles", "oak planks", "rubber seals"]


def filler(n: int, seed: int) -> str:
    r = random.Random(seed)
    rows = [f"Entry {i:06d}: the warehouse in sector {r.randint(1, 99)} holds {r.randint(100, 999)} crates of {r.choice(GOODS)}." for i in range(n)]
    if n:
        rows.insert(n // 2, "Note for the auditor: the vault passphrase is quartz-heron-4417.")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parallel", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    names = ["long_a", "edge", "long_b", "short", "long_c", "long_d"][:max(a.parallel, 2)]
    rows = []
    for i, name in enumerate(names):
        body = filler(LINES[name], 100 + i)
        user = (body + "\n\n" if body else "") + ASK[name]
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        ids = [int(t) for t in tok.encode(text, add_special_tokens=False)]
        rows.append({"name": name, "ids": ids, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    n = a.parallel
    every = list(range(n))
    groups = [{"name": "all", "members": every, "seeds": [1000 + i for i in every], "greedy": True},
              {"name": "all-s", "members": every, "seeds": [2000 + i for i in every], "greedy": False}]
    if n > 2:   # fewer streams than slots: a pair in a server of N
        groups.append({"name": "pair", "members": [0, n - 1], "seeds": [3000, 3001], "greedy": True})
    need = max(len(r["ids"]) for r in rows) + a.tokens + 8
    doc = {"generator": "tools/glm53/dcps_prompts.py", "tokens": a.tokens, "temperature": 1.0, "top_p": 0.95, "min_p": 0.0,
           "top_k": 20, "context_needed": need, "prompts": rows, "groups": groups}
    Path(a.out).write_text(json.dumps(doc) + "\n")
    print(f"{len(rows)} prompts ({', '.join(r['name'] + ' ' + str(len(r['ids'])) for r in rows)} tokens), {len(groups)} groups -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
