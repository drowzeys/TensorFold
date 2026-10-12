#!/usr/bin/env python3
"""The token rows the as-served scorer runs (tf-glm53-generate --mode score), from openly licensed sources.

  score_rows.py --model DIR --out rows.bin [--length 2048] [--wiki 32] [--code 8] [--math 8] [--chat 16] [--tools 16]

Sets (a count of 0 leaves a set out):
  wiki   wikitext-2-raw-v1 test (CC BY-SA 3.0), its lines joined by a blank line and cut into rows of --length tokens
         with no special tokens: the rows exllamav3's eval/model_diff.py scores (our 2.75 bpw figure, KL 0.124, is
         its first 32).
  code   openai_humaneval (MIT): each task's prompt and canonical solution, joined, cut the same way.
  math   gsm8k main test (MIT): each question and its worked answer, joined, cut the same way.
  chat   HuggingFaceH4/ultrachat_200k test_sft (MIT): whole conversations through the model's chat template
         (thinking on, the template's default), one a row, cut at --length; conversations under --min-chat tokens are
         skipped.
  tools  NousResearch/hermes-function-calling-v1 func_calling (Apache-2.0): the conversation's tool schemas, tool
         calls and tool replies through the model's chat template, one conversation a row, cut at --length.
Token ids only go into the file; a .json beside it records the sources, counts and the tokenizer.
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_format as sf


def cut(ids, length, n):
    return [ids[a:a + length] for a in range(0, len(ids) - length, length)][:n]


def hermes_messages(row):
    """(messages, tools) for the chat template, or None when a turn does not parse."""
    tools = json.loads(row["tools"])
    out = []
    for turn in row["conversations"]:
        who, text = turn["from"], turn["value"]
        if who == "system":
            continue   # the dataset's own tool preamble; the model's template writes its own from `tools`
        if who == "human":
            out.append({"role": "user", "content": text})
        elif who == "gpt":
            calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
            if calls:
                parsed = []
                for i, c in enumerate(calls):
                    try:
                        j = json.loads(c)
                    except ValueError:
                        return None
                    if not isinstance(j, dict) or "name" not in j:
                        return None
                    parsed.append({"id": f"call_{len(out)}_{i}", "type": "function",
                                   "function": {"name": j["name"], "arguments": j.get("arguments", {})}})
                rest = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
                out.append({"role": "assistant", "content": rest, "tool_calls": parsed})
            else:
                out.append({"role": "assistant", "content": text})
        elif who == "tool":
            replies = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, re.S) or [text]
            for r in replies:
                out.append({"role": "tool", "content": r})
        else:
            return None
    if not any(m.get("tool_calls") for m in out):
        return None
    return out, tools


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a GLM-5.3 checkpoint directory (tokenizer.json, chat_template.jinja)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--wiki", type=int, default=32)
    ap.add_argument("--code", type=int, default=8)
    ap.add_argument("--math", type=int, default=8)
    ap.add_argument("--chat", type=int, default=16)
    ap.add_argument("--tools", type=int, default=16)
    ap.add_argument("--min-chat", type=int, default=512)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    vocab = json.load(open(os.path.join(a.model, "config.json")))["vocab_size"]

    def plain(text):
        return tok.encode(text, add_special_tokens=False)

    sets, rows, note = [], [], {}

    def add(name, got, source, license_, how):
        if not got:
            return
        sets.append((name, len(got)))
        rows.extend(got)
        note[name] = {"source": source, "license": license_, "rows": len(got), "tokens": sum(len(r) for r in got), "how": how}

    if a.wiki:
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        add("wiki", cut(plain("\n\n".join(d["text"])), a.length, a.wiki), "Salesforce/wikitext wikitext-2-raw-v1 test",
            "CC BY-SA 3.0", "lines joined by a blank line, no special tokens, consecutive rows (exllamav3 eval/model_diff.py's rows)")
    if a.code:
        d = load_dataset("openai/openai_humaneval", split="test")
        text = "\n\n".join(p + s for p, s in zip(d["prompt"], d["canonical_solution"]))
        add("code", cut(plain(text), a.length, a.code), "openai/openai_humaneval test", "MIT",
            "prompt + canonical_solution of each task, joined by a blank line, consecutive rows")
    if a.math:
        d = load_dataset("openai/gsm8k", "main", split="test")
        text = "\n\n".join(q + "\n" + s for q, s in zip(d["question"], d["answer"]))
        add("math", cut(plain(text), a.length, a.math), "openai/gsm8k main test", "MIT",
            "question, newline, worked answer; joined by a blank line, consecutive rows")
    if a.chat:
        d = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        got = []
        for r in d:
            ids = tok.apply_chat_template(r["messages"], tokenize=False)
            ids = plain(ids)
            if len(ids) >= a.min_chat:
                got.append(ids[:a.length])
            if len(got) == a.chat:
                break
        add("chat", got, "HuggingFaceH4/ultrachat_200k test_sft", "MIT",
            f"whole conversations through the chat template (thinking on), first {a.chat} of at least {a.min_chat} tokens, cut at {a.length}")
    if a.tools:
        d = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling", split="train")
        got = []
        for r in d:
            m = hermes_messages(r)
            if m is None:
                continue
            try:
                text = tok.apply_chat_template(m[0], tools=m[1], tokenize=False)
            except Exception:
                continue
            ids = plain(text)
            if len(ids) >= a.min_chat:
                got.append(ids[:a.length])
            if len(got) == a.tools:
                break
        add("tools", got, "NousResearch/hermes-function-calling-v1 func_calling train", "Apache-2.0",
            f"tool schemas, calls and replies through the chat template (thinking on), first {a.tools} conversations of at least {a.min_chat} tokens, cut at {a.length}")

    if not rows:
        sys.exit("no rows: every set is 0")
    n_pos = sf.write_rows(a.out, vocab, sets, rows)
    note["_file"] = {"rows": len(rows), "positions": n_pos, "vocab": vocab, "length": a.length,
                     "tokenizer_sha256": hashlib.sha256(open(os.path.join(a.model, "tokenizer.json"), "rb").read()).hexdigest(),
                     "chat_template_sha256": hashlib.sha256(open(os.path.join(a.model, "chat_template.jinja"), "rb").read()).hexdigest()}
    json.dump(note, open(a.out + ".json", "w"), indent=1)
    for name, n in sets:
        print(f"{name}: {n} rows, {note[name]['tokens']} tokens")
    print(f"wrote {a.out}: {len(rows)} rows, {n_pos} positions")


if __name__ == "__main__":
    main()
