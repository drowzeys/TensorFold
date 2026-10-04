"""Prompt reuse checks against a running full GLM-5.3 TensorFold server (TF_GLM53_PROMPT_REUSE=1): turn-2 TTFT of a
long conversation, an identical resend, a shared system prompt, and reply equality of each resumed request with its
cold reference ("draft": false: cut at the same keep points, never resumed). Standard library only.

    python3 tools/glm53_reuse_check.py URL [--ctx 32768] [--system 8192] [--max-tokens 192] [--mode dflash]
                                           [--seed 7] [--temperature 1.0] [--out reuse.json]

Equality is guaranteed only with a reproducible prefill (serve with TF_EXL3_PROMPT_DET=slots16, and
TF_GLM53_PREFILL_REDUCE=rs TF_GLM53_PROMPT_OVERLAP=0 if slots16 alone does not match); with the default arrival-order
expert sums a resumed prompt is one valid prefill and replies may differ from the cold one after a few tokens.
"""

import argparse
import json
import time
import urllib.request

WORDS = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow nickel orchid pepper "
         "quartz raven saffron timber umber violet willow xenon yarrow zephyr").split()


def filler(tokens: int, salt: str) -> str:
    """About ``tokens`` tokens of numbered lines (~12 tokens a line), unique per ``salt``."""
    lines = []
    for i in range(max(1, tokens // 12)):
        w = [WORDS[(i * 7 + j * 3 + len(salt)) % len(WORDS)] for j in range(5)]
        lines.append(f"{salt}-{i}: {' '.join(w)} {i * 37 % 1009}")
    return "\n".join(lines)


def ask(url: str, messages: list, args, draft: bool = True) -> dict:
    body = {"model": "x", "messages": messages, "max_tokens": args.max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True, "temperature": args.temperature,
            "seed": args.seed}
    if args.temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if args.mode:
        body["tf_mtp"] = args.mode
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, content, reasoning, usage, stats = None, [], [], {}, {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            msg = json.loads(line[5:])
            usage = msg.get("usage") or usage
            stats = msg.get("tensorfold") or stats
            for ch in msg.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                    ttft = time.perf_counter() - t0
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or "")
    return {"ttft": round(ttft or -1, 3), "prompt_tokens": usage.get("prompt_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prefill_s": stats.get("prefill_s"), "replay": bool(stats.get("replay")), "ids": stats.get("token_ids"),
            "content": "".join(content), "reasoning": "".join(reasoning)}


def show(name: str, r: dict) -> None:
    print(f"{name:14s} prompt {r['prompt_tokens']:>6} cached {r['cached']!s:>6} ttft {r['ttft']:8.2f}s "
          f"prefill {r['prefill_s']!s:>8} replay {r['replay']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--system", type=int, default=8192)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--mode", default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default="")
    args = p.parse_args()
    salt = str(int(time.time()))                        # nothing from an earlier run resumes
    res, same = {}, {}

    # 1. a long conversation's turn 2 (and its cold reference), 2. the identical resend
    t1 = [{"role": "user", "content": filler(args.ctx, "doc" + salt) + "\n\nWhich word appears most often above?"}]
    res["turn1_cold"] = r1 = ask(args.url, t1, args)
    t2 = t1 + [{"role": "assistant", "content": r1["content"], "reasoning_content": r1["reasoning"]},
               {"role": "user", "content": "Answer again in one short sentence."}]
    res["turn2_warm"] = ask(args.url, t2, args)
    res["turn2_cold"] = ask(args.url, t2, args, draft=False)
    res["turn2_resend"] = ask(args.url, t2, args)
    same["turn2 warm == cold"] = res["turn2_warm"]["ids"] == res["turn2_cold"]["ids"]
    same["resend == warm"] = res["turn2_resend"]["ids"] == res["turn2_warm"]["ids"]

    # 3. a shared system prompt: a second conversation pays only past the system block
    system = {"role": "system", "content": filler(args.system, "sys" + salt)}
    res["system_a"] = ask(args.url, [system, {"role": "user", "content": "List three words from the system text."}],
                          args)
    b = [system, {"role": "user", "content": "Count the lines of the system text, roughly."}]
    res["system_b_warm"] = ask(args.url, b, args)
    res["system_b_cold"] = ask(args.url, b, args, draft=False)
    same["system b warm == cold"] = res["system_b_warm"]["ids"] == res["system_b_cold"]["ids"]

    for name, r in res.items():
        show(name, r)
    for name, ok in same.items():
        print(f"{name:24s} {'EQUAL' if ok else 'DIFFERENT'}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": {k: {kk: vv for kk, vv in v.items() if kk not in ("content", "reasoning")}
                                   for k, v in res.items()}, "equal": same}, f, indent=1)


if __name__ == "__main__":
    main()
