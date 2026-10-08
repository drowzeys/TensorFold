#!/usr/bin/env python3
"""Phase 4b's prompt maker, concurrent chat client and gates (run_phase4b_cluster.sh): --parallel N, concurrent
streams. Runs inside the GLM-5.3 TP4 image on rank 0's node (python3 stdlib; the tokenizer / chat template only for
make-prompts).

Subcommands:
  make-prompts --model DIR --out prompts-4b.json [--parallel 4] [--tokens 512] [--sampled 1]
      bench/conc_chat.py's prompts (bench.py's PROMPTS: prose = beekeeper, lighthouse, nurse; code = cache, parser;
      stream i takes prompt i % len) through the chat template with thinking on, as tf-glm53-generate --mode 4b's
      groups: "prose" and "code" greedy, and with --sampled 1 "prose-s" / "code-s" (T 1.0, top_p 0.95, top_k 20, seeds
      1000 + i as conc_chat.py sends them)
  client --url U --kind prose|code --streams 1,4 [--tokens 512] [--greedy] --out X.json [--idle S]
      conc_chat.py's runs (the same request bodies and timing: aggregate tok/s = every reply's completion tokens over
      the wall time from the first request sent to the last reply finished; per-stream tok/s = a reply's tokens after
      its first over its own decode time; TTFT = the first content / reasoning delta), recording each stream's text
  gate --work DIR [--parallel 4] [--ratio 0.97] [--drafter mtp|dflash|dspark] [--bar-ratio 1.0]
      every gate from the files the run left; prints "GATE <name> PASS|FAIL ..." and "REPORT ..." lines:
        cli            tf-glm53-generate --mode 4b: every stream's tokens == its alone run == its one-stream run, every
                       group (greedy and sampled), and every rank agreed
        text-prose / text-code   the Zig server's replies == the Python server's, stream by stream, at 1 and N streams
                       (greedy: Python is deterministic there)
        speed-prose / speed-code the Zig server's aggregate tok/s at N streams >= ratio x the Python server's
        served-cli     the Zig server's replies (TF_GLM53_DUMP: its prompt ids and tokens) == the CLI's concurrent
                       tokens of the same prompt ids (when the ids match; else a REPORT)
      Phase 4c (--drafter dflash | dspark; run_phase4c_cluster.sh): every stream drafts with that drafter.
        cli            also: the CLI ran that draft mode (cli-4b.json "draft_mode")
        conc-prose / conc-code   the Zig server's 1-stream reply == its reply of the same prompt and seed among N
                       concurrent streams (greedy): single-stream vs concurrent identity through the server
        text-*         against py-<kind>.json (the Python server with the same drafter: dflash, mtp) or, without it
                       (dspark: Python refuses --parallel with DSpark), py-mtp-<kind>.json (greedy replies do not
                       depend on the drafts)
        speed-*        against the Python server with the same drafter (--ratio); dspark: against the Python MTP
                       server, the speed bar (--bar-ratio; 0: reported only). dflash also reports the MTP bar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL_NAME = "glm-5.3-tf"
# bench/bench.py's PROMPTS (keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark), which bench/conc_chat.py sends
PROMPTS = {
    "prose_beekeeper": "Write a vivid literary short story (at least 700 words) about a beekeeper during a drought. No headings, flowing prose only.",
    "prose_lighthouse": "Write a vivid literary short story (at least 700 words) about a lighthouse keeper in 1890s Norway. No headings, flowing prose only.",
    "prose_nurse": "Write a vivid literary short story (at least 700 words) about a night-shift nurse in Lagos. No headings, flowing prose only.",
    "code_cache": "Write a complete, well-documented Python module implementing an LRU cache with TTL expiry, thread safety, statistics, and a small CLI demo. Name the class Cache7. Code only, no prose.",
    "code_parser": "Write a complete Python recursive-descent parser for arithmetic expressions with parentheses, unary minus, exponentiation, multiplication and addition. Include helpful syntax errors, examples, and unittest tests. Code only.",
}
KIND = {"prose": [k for k in PROMPTS if k.startswith(("prose", "essay", "story", "explain"))],
        "code": [k for k in PROMPTS if "code" in k]}
SEED0 = 1000
# the published reference (README, PARALLEL=4 CONTEXT=32768, greedy chat, thinking on, MTP drafts)
PUBLISHED = {"prose": (34.7, 78.7), "code": (39.1, 100.2)}


def say(*a) -> None:
    print("[gate4b]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- make-prompts ---
def make_prompts(a) -> int:
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    model = Path(a.model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    tmpl = ChatTemplate(model)
    names = list(PROMPTS)
    rows = []
    for name in names:
        text = tmpl.render([{"role": "user", "content": PROMPTS[name]}], tools=None, enable_thinking=True)
        ids = [int(t) for t in tok.encode(text, add_special_tokens=False).ids]
        rows.append({"name": name, "ids": ids, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    groups = []
    for kind in ("prose", "code"):
        ks = KIND[kind]
        members = [names.index(ks[i % len(ks)]) for i in range(a.parallel)]
        seeds = [SEED0 + i for i in range(a.parallel)]
        groups.append({"name": kind, "members": members, "seeds": seeds, "greedy": True})
        if a.sampled:
            groups.append({"name": kind + "-s", "members": members, "seeds": seeds, "greedy": False})
    need = max(len(r["ids"]) for r in rows) + a.tokens + 8
    doc = {"generator": "tools/glm53/gate4b.py make-prompts", "tokens": a.tokens, "temperature": 1.0, "top_p": 0.95,
           "min_p": 0.0, "top_k": 20, "context_needed": need, "prompts": rows, "groups": groups}
    Path(a.out).write_text(json.dumps(doc) + "\n")
    say(f"make-prompts: {len(rows)} prompts ({', '.join(str(len(r['ids'])) for r in rows)} tokens), "
        f"{len(groups)} groups of {a.parallel} -> {a.out}")
    return 0


# ---------------------------------------------------------------------------------------------------- client ---
IDLE_S = 7200.0


def one(url: str, prompt: str, seed: int, tokens: int, greedy: bool, out: list, lock: threading.Lock) -> None:
    """conc_chat.one, keeping the reply's text (reasoning + content) and errors."""
    body = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "stream": True, "stream_options": {"include_usage": True}, "seed": seed,
            "temperature": 0.0 if greedy else 1.0, "top_p": 0.95}
    rec = {"seed": seed, "content": "", "reasoning": "", "usage": {}, "error": None, "finish": None}
    t0 = time.time()
    first = last = None
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=IDLE_S) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("error"):
                    err = ev["error"]
                    rec["error"] = (err.get("message") if isinstance(err, dict) else None) or str(err)
                    break
                rec["usage"] = ev.get("usage") or rec["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    c, rc = d.get("content") or "", d.get("reasoning_content") or d.get("reasoning") or ""
                    if c or rc:
                        now = time.time()
                        first = first or now
                        last = now
                    rec["content"] += c
                    rec["reasoning"] += rc
                    if ch.get("finish_reason"):
                        rec["finish"] = ch["finish_reason"]
    except (urllib.error.URLError, OSError, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    n = (rec["usage"] or {}).get("completion_tokens", 0)
    rec.update({"tokens": n, "ttft": (first or time.time()) - t0, "end": time.time(),
                "tps": (n - 1) / (last - first) if first and last and last > first else 0.0,
                "text_sha256": hashlib.sha256((rec["reasoning"] + "\x00" + rec["content"]).encode()).hexdigest()})
    with lock:
        out.append(rec)


def client(a) -> int:
    global IDLE_S
    IDLE_S = float(a.idle)
    prompts = [PROMPTS[k] for k in KIND[a.kind]]
    runs = []
    bad = 0
    for n in [int(x) for x in a.streams.split(",")]:
        out, threads, lock = [], [], threading.Lock()
        t0 = time.time()
        for i in range(n):
            th = threading.Thread(target=one, args=(a.url, prompts[i % len(prompts)], SEED0 + i, a.tokens, a.greedy,
                                                    out, lock))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        out.sort(key=lambda r: r["seed"])
        errs = [r["error"] for r in out if r["error"]]
        wall = max(r["end"] for r in out) - t0
        row = {"kind": a.kind, "greedy": a.greedy, "streams": n,
               "aggregate_tps": round(sum(r["tokens"] for r in out) / wall, 2),
               "per_stream_tps": round(statistics.mean(r["tps"] for r in out), 2),
               "ttft_max_s": round(max(r["ttft"] for r in out), 3), "wall_s": round(wall, 3),
               "tokens": sum(r["tokens"] for r in out), "errors": errs, "replies": out}
        runs.append(row)
        say(json.dumps({k: v for k, v in row.items() if k != "replies"}))
        bad += len(errs)
        Path(a.out).write_text(json.dumps({"url": a.url, "runs": runs}) + "\n")
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------------------ gate ---
def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_dump(path: Path) -> list[dict]:
    out = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return out


def gate(a) -> int:
    work = Path(a.work)
    fails = []

    def result(name: str, ok: bool, text: str) -> None:
        print(f"GATE {name} {'PASS' if ok else 'FAIL'} {text}", flush=True)
        if not ok:
            fails.append(name)

    # -- the CLI: concurrent == alone == one stream, every group, every rank agreed
    cli = load(work / "cli-4b.json")
    if cli is None:
        result("cli", False, "no cli-4b.json (tf-glm53-generate --mode 4b failed?)")
    else:
        bad = []
        ran = cli.get("draft_mode", "mtp")
        if ran != a.drafter:
            bad.append(f"draft mode {ran}, not {a.drafter}")
        for g in cli.get("groups", []):
            if not g.get("concurrent_eq_alone") or (g.get("single_run") and not g.get("alone_eq_single")):
                bad.append(g["name"])
            print(f"REPORT cli {g['name']}: {g['streams']} streams aggregate {g['aggregate_tps']:.1f} tok/s, "
                  f"per-stream {g['per_stream_tps']:.1f}, TTFT max {g['ttft_max_s']:.3f} s, {g['rounds']} rounds, "
                  f"{g['cuts']} drafts cut; == alone {g['concurrent_eq_alone']}, alone == one-stream "
                  f"{g['alone_eq_single'] if g.get('single_run') else 'not run'}", flush=True)
            for m in g.get("members", []):
                print(f"REPORT cli {g['name']} {m['prompt']} seed {m['seed']}: {m['tokens_n']} tokens, {m['tok_s']:.1f} "
                      f"tok/s concurrent ({m['tokens_per_round']:.2f} tokens/round), alone {m['alone_tok_s']:.1f} "
                      f"({m['alone_tokens_per_round']:.2f}), one-stream {m['single_tok_s']:.1f}; TTFT "
                      f"{m['ttft_s']:.3f} s", flush=True)
        ok = bool(cli.get("pass")) and not bad and bool(cli.get("groups"))
        result("cli", ok, f"{len(cli.get('groups', []))} groups at --parallel {cli.get('parallel')}"
                          f"{'; differ: ' + ','.join(bad) if bad else ''}")

    # -- the servers: Zig == Python reply by reply (greedy), speed at N streams
    for kind in ("prose", "code"):
        same_py = load(work / f"py-{kind}.json")       # the Python server with the same drafter (mtp, dflash)
        bar = load(work / f"py-mtp-{kind}.json")        # Phase 4c: the Python MTP server (the speed bar)
        py = same_py if same_py is not None else bar
        zg = load(work / f"zig-{kind}.json")
        if zg is not None and a.drafter != "mtp":       # single-stream vs concurrent identity through the Zig server
            zr0 = {r["streams"]: r for r in zg["runs"]}
            one_ = {x["seed"]: x for x in zr0.get(1, {}).get("replies", [])}
            checked = equal = 0
            for n in sorted(zr0):
                if n == 1:
                    continue
                for x in zr0[n]["replies"]:
                    y = one_.get(x["seed"])
                    if y is None:
                        continue
                    checked += 1
                    equal += int(x["text_sha256"] == y["text_sha256"] and x["tokens"] == y["tokens"]
                                 and not x["error"] and not y["error"])
            result(f"conc-{kind}", checked > 0 and equal == checked,
                   f"{equal}/{checked} concurrent {a.drafter} replies == the same prompt and seed alone (greedy)")
        if py is None or zg is None:
            result(f"text-{kind}", False, f"missing {'py(-mtp)' if py is None else 'zig'}-{kind}.json")
            result(f"speed-{kind}", False, "no client results")
            continue
        pr = {r["streams"]: r for r in py["runs"]}
        zr = {r["streams"]: r for r in zg["runs"]}
        same, total, diffs = 0, 0, []
        for n in sorted(set(pr) & set(zr)):
            for p, z in zip(pr[n]["replies"], zr[n]["replies"]):
                total += 1
                if p["text_sha256"] == z["text_sha256"] and p["tokens"] == z["tokens"] and not p["error"] and not z["error"]:
                    same += 1
                else:
                    diffs.append(f"{n}x seed {p['seed']} ({p['tokens']} / {z['tokens']} tokens)")
        errs = sum(len(r["errors"]) for r in zg["runs"]) + sum(len(r["errors"]) for r in pr.values())
        result(f"text-{kind}", total > 0 and same == total and errs == 0,
               f"{same}/{total} replies equal (greedy){'; differ: ' + ', '.join(diffs[:4]) if diffs else ''}"
               f"{f'; {errs} request errors' if errs else ''}")
        N = a.parallel
        # the speed gate: against the Python server with the same drafter (--ratio); without one (dspark) against
        # the Python MTP bar (--bar-ratio; 0: reported only)
        need = a.ratio if same_py is not None else a.bar_ratio
        what = f"python {a.drafter}" if same_py is not None else "python mtp (the bar)"
        if N in pr and N in zr:
            pt, zt = pr[N]["aggregate_tps"], zr[N]["aggregate_tps"]
            ratio = zt / pt if pt else 0.0
            line = (f"{N} streams aggregate: zig {a.drafter} {zt:.1f} tok/s, {what} {pt:.1f} ({ratio:.3f}x, need "
                    f"{need}); TTFT max zig {zr[N]['ttft_max_s']:.2f} s, python {pr[N]['ttft_max_s']:.2f} s")
            if need > 0:
                result(f"speed-{kind}", ratio >= need, line)
            else:
                print(f"REPORT speed-{kind}: {line}", flush=True)
        else:
            result(f"speed-{kind}", False, f"no {N}-stream run in both")
        if same_py is not None and bar is not None and a.drafter != "mtp":
            br = {r["streams"]: r for r in bar["runs"]}
            if N in br and N in zr:
                print(f"REPORT bar-{kind}: {N} streams aggregate zig {a.drafter} {zr[N]['aggregate_tps']:.1f} tok/s vs "
                      f"python mtp {br[N]['aggregate_tps']:.1f} ({zr[N]['aggregate_tps'] / max(br[N]['aggregate_tps'], 1e-9):.3f}x); "
                      f"python {a.drafter} {pr[N]['aggregate_tps'] if N in pr else '-'}", flush=True)
        pub1, pubN = PUBLISHED[kind]
        for n in sorted(set(pr) | set(zr)):
            z, p = zr.get(n), pr.get(n)
            print(f"REPORT {kind} {n} stream{'s' if n > 1 else ''}: zig "
                  f"{z['aggregate_tps'] if z else '-'} tok/s aggregate / {z['per_stream_tps'] if z else '-'} per stream / "
                  f"TTFT max {z['ttft_max_s'] if z else '-'} s; python {p['aggregate_tps'] if p else '-'} / "
                  f"{p['per_stream_tps'] if p else '-'} / {p['ttft_max_s'] if p else '-'} s; published "
                  f"{pub1 if n == 1 else pubN} (README, {'1 stream' if n == 1 else f'{N} streams aggregate'})",
                  flush=True)
        for label in ("s",):   # sampled client runs, when the run made them: reported, not gated
            ps = load(work / f"py-{kind}-{label}.json") or load(work / f"py-mtp-{kind}-{label}.json")
            zs = load(work / f"zig-{kind}-{label}.json")
            if ps and zs:
                a_ = {r["streams"]: r for r in ps["runs"]}
                b_ = {r["streams"]: r for r in zs["runs"]}
                eq = [n for n in a_ if n in b_ and [x["text_sha256"] for x in a_[n]["replies"]]
                      == [x["text_sha256"] for x in b_[n]["replies"]]]
                print(f"REPORT {kind} sampled (T 1.0, top_p 0.95): replies equal at {sorted(eq)} streams of "
                      f"{sorted(set(a_) & set(b_))}; aggregate zig "
                      f"{[b_[n]['aggregate_tps'] for n in sorted(b_)]} python {[a_[n]['aggregate_tps'] for n in sorted(a_)]}",
                      flush=True)

    # -- served == CLI: the Zig server's dumped requests against the CLI's concurrent tokens of the same prompt ids
    prompts = load(work / "prompts-4b.json")
    dump = read_dump(work / "dump-zig.jsonl")
    if cli is not None and prompts is not None and dump:
        by_ids = {}
        for g in cli.get("groups", []):
            if not g.get("greedy"):
                continue
            for m in g["members"]:
                ids = next(p["ids"] for p in prompts["prompts"] if p["name"] == m["prompt"])
                by_ids.setdefault(tuple(ids), m["tokens"])
        checked, equal = 0, 0
        for d in dump:
            if not d.get("sampling", {}).get("greedy"):
                continue
            want = by_ids.get(tuple(d.get("prompt_ids", [])))
            if want is None:
                continue
            checked += 1
            got = d.get("tokens", [])
            if got == want[:len(got)] and (len(got) == len(want) or d.get("finish") == "stop"):
                equal += 1
        if checked:
            result("served-cli", equal == checked, f"{equal}/{checked} served greedy replies == the CLI's tokens")
        else:
            print("REPORT served-cli: no served request had the CLI's prompt ids (the server's chat template rendered "
                  "them differently?) - not gated", flush=True)
    else:
        print("REPORT served-cli: no dump-zig.jsonl / cli-4b.json / prompts-4b.json - not checked", flush=True)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-prompts")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--sampled", type=int, default=1)
    p = sub.add_parser("client")
    p.add_argument("--url", required=True)
    p.add_argument("--kind", default="prose", choices=list(KIND))
    p.add_argument("--streams", default="1,4")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--idle", type=float, default=IDLE_S)
    p = sub.add_parser("gate")
    p.add_argument("--work", required=True)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--ratio", type=float, default=0.97)
    p.add_argument("--drafter", default="mtp", choices=("mtp", "dflash", "dspark"))
    p.add_argument("--bar-ratio", type=float, default=1.0)
    a = ap.parse_args()
    return {"make-prompts": make_prompts, "client": client, "gate": gate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
