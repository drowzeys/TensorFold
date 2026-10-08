#!/usr/bin/env python3
"""Phase 3b's client, prompt makers and gates (run_phase3b_cluster.sh), inside the GLM-5.3 TP4 image on rank 0's
node (python3 stdlib; the tokenizer / chat template only for make-needle, make-dcp and the 1M needle check).

Subcommands:
  make-conv   --context-file F --out conv.json             the 3-turn conversation (system prompt + tools, turn 1's
                                                           user message carries the 32K text; sampled, fixed seeds)
  conv        --url U --spec conv.json --out X.json [--warm 1,2,3] [--cold 2,3] [--serial 3]
              [--flush-file /work/REUSE_FLUSH] [--history convA.json] [--idle S]
              streams each turn (client TTFT = first content / reasoning delta); turn n's history is turn 1..n-1's
              replies from --history when given, else this run's own warm replies. --cold: the flush file is touched
              before each cold turn (the server forgets every kept state). --serial: the turn again with "draft": false
  api         --url U --out api.json [--idle S]            non-streaming, streaming, a tool call, a Hermes-style
                                                           3-request multi-turn smoke
  dump2cli    --dump D.jsonl --out prompts.json [--skip N]  the dump's finished requests as a tf-glm53-generate
                                                           --mode 3b prompts file (ids, seed, sampling, token count)
  make-needle --model DIR --context-file F --out prompts-1m.json [--depth 830000] [--tokens 256]
  make-dcp    --model DIR --context-file F --out prompts-dcp.json [--length 20000] [--tokens 16]
  gate        --work /work [--mem1m TEXT] [--py1m 0|1] [--ratio 0.97] [--model DIR]
              every gate from the files the run left: prints "GATE a|b|c|d PASS|FAIL ..." and "REPORT ..." lines
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_NAME = "glm-5.3-tf"
SEED0 = 4242
TOOLS = [
    {"type": "function", "function": {
        "name": "search_notes", "description": "Search the user's research notes for a phrase and return matching "
                                              "paragraphs with their dates.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "words to search for"},
            "limit": {"type": "integer", "description": "most paragraphs to return", "default": 5}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "the city name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
]
QUESTIONS = [
    "Question 1: In a few sentences, what is the central argument of the background reading, and which passage "
    "supports it best?",
    "Question 2: Name two weaknesses in that argument and say how the author could answer each one.",
    "Question 3: Write a short, plain summary of our discussion so far for a reader who has not seen the text.",
]


def say(*a) -> None:
    print("[gate3b]", *a, flush=True)


# ------------------------------------------------------------------------------------------------------- http ---
# socket timeout of every request: no byte from the server for this long fails it (--idle; the run script passes
# PROGRESS_MIN minutes) instead of a request hanging the gate (run 2, 2026-10-08)
IDLE_S = 7200.0


def post(url: str, body: dict, timeout: float | None = None) -> tuple[int, dict, float]:
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout or IDLE_S) as r:
            return r.status, json.load(r), time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}"), time.time() - t0
        except ValueError:
            return e.code, {}, time.time() - t0
    except (urllib.error.URLError, OSError, ValueError) as e:  # refused, reset, timed out, or a broken body
        say(f"request failed: {type(e).__name__}: {e}")
        return 0, {"error": f"{type(e).__name__}: {e}"}, time.time() - t0


def stream(url: str, body: dict, timeout: float | None = None) -> dict:
    """One streamed request: status, content, reasoning, tool call deltas, finish, usage, client TTFT, [DONE] seen."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    out = {"status": 0, "content": "", "reasoning": "", "tool_calls": {}, "finish": None, "usage": None,
           "ttft": None, "done": False, "chunks": 0, "tensorfold": None, "error": None}
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout or IDLE_S)
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["error"] = (e.read() or b"")[:500].decode("utf-8", "replace")
        return out
    except (urllib.error.URLError, OSError) as e:   # refused, reset, or no byte for `timeout` s (socket timeout)
        out["error"] = f"{type(e).__name__}: {e}"
        out["seconds"] = time.time() - t0
        return out
    out["status"] = r.status
    with r:
        try:
            lines = iter(r)
            for raw in lines:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    out["done"] = True
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                if ev.get("error"):   # the server's stream error event: the request failed
                    err = ev["error"]
                    out["error"] = (err.get("message") if isinstance(err, dict) else None) or str(err)
                    break
                out["chunks"] += 1
                _delta(out, ev, t0)
        except (OSError, ValueError) as e:  # a stalled stream (socket timeout) or a broken connection
            out["error"] = f"stream broke after {out['chunks']} chunks: {type(e).__name__}: {e}"
    out["seconds"] = time.time() - t0
    return out


def _delta(out: dict, ev: dict, t0: float) -> None:
    """One SSE event into the stream's record."""
    if ev.get("usage"):
        out["usage"] = ev["usage"]
    if ev.get("tensorfold"):
        out["tensorfold"] = ev["tensorfold"]
    for ch in ev.get("choices") or []:
        d = ch.get("delta") or {}
        c, rc = d.get("content") or "", d.get("reasoning_content") or d.get("reasoning") or ""
        if (c or rc or d.get("tool_calls")) and out["ttft"] is None:
            out["ttft"] = time.time() - t0
        out["content"] += c
        out["reasoning"] += rc
        for tc in d.get("tool_calls") or []:
            slot = out["tool_calls"].setdefault(int(tc.get("index", 0)), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            slot["name"] += fn.get("name") or ""
            slot["arguments"] += fn.get("arguments") or ""
        if ch.get("finish_reason"):
            out["finish"] = ch["finish_reason"]


# ----------------------------------------------------------------------------------------------- conversation ---
def make_conv(a) -> int:
    text = Path(a.context_file).read_text()
    cut = min(len(text) // 3, 12000)
    system = ("You are a careful research assistant. Answer from the material the user gives you, quote it when it "
              "helps, and say plainly when the material does not settle a question. Use the tools when the user asks "
              "about their notes or the weather.\n\nHouse style handbook (excerpt, for tone only):\n" + text[:cut])
    turns = []
    for i, q in enumerate(QUESTIONS):
        user = ("Background reading:\n" + text[cut:] + "\n\n" + q) if i == 0 else q
        turns.append({"user": user, "seed": SEED0 + i})
    spec = {"system": system, "tools": TOOLS, "turns": turns, "max_tokens": a.max_tokens,
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}}
    Path(a.out).write_text(json.dumps(spec) + "\n")
    say(f"conversation: system {len(system)} chars, turn 1 {len(turns[0]['user'])} chars, {len(turns)} turns")
    return 0


def turn_body(spec: dict, n: int, history: list[dict]) -> dict:
    msgs = [{"role": "system", "content": spec["system"]}]
    for i in range(n - 1):
        msgs.append({"role": "user", "content": spec["turns"][i]["user"]})
        msgs.append(history[i])
    msgs.append({"role": "user", "content": spec["turns"][n - 1]["user"]})
    s = spec["sampling"]
    return {"model": MODEL_NAME, "messages": msgs, "tools": spec["tools"], "max_tokens": spec["max_tokens"],
            "temperature": s["temperature"], "top_p": s["top_p"], "top_k": s["top_k"],
            "seed": spec["turns"][n - 1]["seed"]}


def assistant_of(r: dict) -> dict:
    m = {"role": "assistant", "content": r["content"]}
    if r["reasoning"]:
        m["reasoning_content"] = r["reasoning"]
    return m


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def conv(a) -> int:
    spec = json.loads(Path(a.spec).read_text())
    hist_src = json.loads(Path(a.history).read_text())["history"] if a.history else None
    history: list[dict] = list(hist_src) if hist_src else []
    reqs = []

    def run(label: str, n: int, extra: dict | None = None) -> dict:
        if n - 1 > len(history):
            raise SystemExit(f"turn {n}: no history for turn {n - 1} (send it warm first, or --history)")
        body = turn_body(spec, n, history)
        if extra:
            body.update(extra)
        r = stream(a.url, body)
        rec = {"label": label, "turn": n, "status": r["status"], "ttft": r["ttft"], "seconds": r.get("seconds"),
               "finish": r["finish"], "usage": r["usage"], "done": r["done"], "content": r["content"],
               "reasoning": r["reasoning"], "error": r["error"], "tensorfold": r["tensorfold"]}
        reqs.append(rec)
        pt = (r["usage"] or {}).get("prompt_tokens")
        say(f"{label}: status {r['status']} ttft {r['ttft'] if r['ttft'] is None else round(r['ttft'], 3)} s, "
            f"prompt {pt}, cached {((r['usage'] or {}).get('prompt_tokens_details') or {}).get('cached_tokens')}, "
            f"finish {r['finish']}, {len(r['content'])} + {len(r['reasoning'])} chars")
        if r["status"] != 200 or not r["done"] or r["error"]:
            # fail fast: the later turns depend on this one (and a failed server request may leave it wedged)
            Path(a.out).write_text(json.dumps({"url": a.url, "requests": reqs, "history": history}, indent=1) + "\n")
            say(f"FAILED request {label}: status {r['status']}, done {r['done']}, error {r['error']!r}")
            raise SystemExit(1)
        return r

    for n in ints(a.warm):
        r = run(f"warm{n}", n)
        if hist_src is None and r["status"] == 200:
            history[n - 1:n] = [assistant_of(r)]
    for n in ints(a.cold):
        if not a.flush_file:
            raise SystemExit("--cold needs --flush-file")
        Path(a.flush_file).touch()
        run(f"cold{n}", n)
    for n in ints(a.serial):
        run(f"serial{n}", n, {"draft": False})
    Path(a.out).write_text(json.dumps({"url": a.url, "requests": reqs, "history": history}, indent=1) + "\n")
    bad = [r["label"] for r in reqs if r["status"] != 200 or not r["done"]]
    if bad:
        say(f"FAILED requests: {bad}")
        return 1
    return 0


# ------------------------------------------------------------------------------------------------------- api ---
def api(a) -> int:
    res: dict = {}
    s, b, dt = post(a.url, {"model": MODEL_NAME, "max_tokens": 400, "temperature": 0,
                            "messages": [{"role": "user", "content": "Say hello in exactly five words."}]})
    ch = (b.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    res["nonstream"] = {"status": s, "finish": ch.get("finish_reason"), "content": msg.get("content"),
                        "reasoning": msg.get("reasoning_content"), "usage": b.get("usage"), "seconds": dt,
                        "ok": s == 200 and ch.get("finish_reason") in ("stop", "length") and
                        bool((msg.get("content") or "") + (msg.get("reasoning_content") or ""))}
    r = stream(a.url, {"model": MODEL_NAME, "max_tokens": 400, "temperature": 0,
                       "messages": [{"role": "user", "content": "Count from one to ten in words."}]})
    res["stream"] = {k: r[k] for k in ("status", "finish", "content", "reasoning", "done", "chunks", "ttft", "usage")}
    res["stream"]["ok"] = r["status"] == 200 and r["done"] and r["chunks"] > 1 and r["finish"] in ("stop", "length")
    s, b, dt = post(a.url, {"model": MODEL_NAME, "max_tokens": 2048, "temperature": 0, "tools": TOOLS,
                            "tool_choice": "auto", "messages": [
                                {"role": "user", "content": "What is the weather in Paris right now? Use the "
                                                            "get_weather tool."}]})
    ch = (b.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = msg.get("tool_calls") or []
    args_ok = False
    if calls:
        try:
            args = json.loads(calls[0]["function"].get("arguments") or "{}")
            args_ok = isinstance(args, dict) and "city" in args
        except (ValueError, KeyError, TypeError):
            args_ok = False
    res["tool"] = {"status": s, "finish": ch.get("finish_reason"), "tool_calls": calls, "content": msg.get("content"),
                   "ok": s == 200 and ch.get("finish_reason") == "tool_calls" and bool(calls) and
                   calls[0].get("function", {}).get("name") == "get_weather" and args_ok}
    # Hermes-style: system + tools, a tool call, its result, a final answer, a follow-up
    htools = [{"type": "function", "function": {
        "name": "terminal", "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {
            "name": "read_file", "description": "Read a text file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    msgs = [{"role": "system", "content": "You are Hermes, an agent that uses tools to help the user. Call a tool when "
                                          "you need information; answer briefly."},
            {"role": "user", "content": "List the files in /tmp/demo with the terminal tool, then tell me how many "
                                        "there are."}]
    hs = []
    s, b, dt = post(a.url, {"model": MODEL_NAME, "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "seed": 7,
                            "tools": htools, "messages": msgs})
    m1 = ((b.get("choices") or [{}])[0]).get("message") or {}
    hs.append({"status": s, "finish": ((b.get("choices") or [{}])[0]).get("finish_reason"), "message": m1})
    calls = m1.get("tool_calls") or []
    if calls:
        msgs.append({"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls})
        msgs.append({"role": "tool", "tool_call_id": calls[0].get("id", "call_0"),
                     "content": "notes.txt\nplan.md\nresults.csv\n"})
    else:
        msgs.append({"role": "assistant", "content": m1.get("content") or "I will check."})
        msgs.append({"role": "user", "content": "The terminal printed: notes.txt plan.md results.csv"})
    s, b, dt = post(a.url, {"model": MODEL_NAME, "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "seed": 8,
                            "tools": htools, "messages": msgs})
    m2 = ((b.get("choices") or [{}])[0]).get("message") or {}
    hs.append({"status": s, "finish": ((b.get("choices") or [{}])[0]).get("finish_reason"), "message": m2})
    msgs.append({"role": "assistant", "content": m2.get("content") or ""})
    msgs.append({"role": "user", "content": "Thanks. Which of those files is most likely a spreadsheet? One word."})
    s, b, dt = post(a.url, {"model": MODEL_NAME, "max_tokens": 1024, "temperature": 0.6, "top_p": 0.95, "seed": 9,
                            "tools": htools, "messages": msgs})
    m3 = ((b.get("choices") or [{}])[0]).get("message") or {}
    hs.append({"status": s, "finish": ((b.get("choices") or [{}])[0]).get("finish_reason"), "message": m3})
    text = lambda m: (m.get("content") or "") + (m.get("reasoning_content") or "")  # noqa: E731
    res["hermes"] = {"requests": hs, "first_called_tool": bool(calls),
                     "ok": all(h["status"] == 200 for h in hs) and bool(text(m2).strip()) and bool(text(m3).strip())}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        say(f"api {k}: {'ok' if v['ok'] else 'FAILED'} (status {v.get('status', [h['status'] for h in v.get('requests', [])])})")
    return 0 if all(v["ok"] for v in res.values()) else 1


# ------------------------------------------------------------------------------------------------- dump / cli ---
def read_dump(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def dump2cli(a) -> int:
    rows = []
    for i, d in enumerate(read_dump(a.dump)):
        if i < a.skip or d.get("finish") not in ("stop", "length"):
            continue
        s = d.get("sampling") or {}
        k = int(d.get("k", 2)) if d.get("drafts", True) else 0
        run = f"g:{k}" if s.get("greedy", True) else f"s{int(s.get('top_k', 0))}:{k}"
        n = len(d["tokens"])
        rows.append({"name": f"req{i:03d}", "ids": d["prompt_ids"], "dump_index": i, "runs": [run],
                     "tokens": n if d["finish"] == "length" else n + 8, "seed": int(s.get("seed", 0)),
                     "seed_from_prompt": False, "temperature": float(s.get("temperature", 0.0) or 0.0),
                     "top_p": float(s.get("top_p", 1.0)), "min_p": float(s.get("min_p", 0.0))})
    need = max((len(r["ids"]) + r["tokens"] for r in rows), default=0)
    Path(a.out).write_text(json.dumps({"generator": "tools/glm53/gate3b.py dump2cli", "source": a.dump,
                                       "sampling": {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0},
                                       "context_needed": need, "prompts": rows}) + "\n")
    say(f"dump2cli: {len(rows)} requests -> {a.out} (context needed {need})")
    return 0 if rows else 1


# --------------------------------------------------------------------------------------------- prompt makers ---
def tokenizer_and_template(model: str):
    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    return Tokenizer.from_file(str(Path(model) / "tokenizer.json")), ChatTemplate(Path(model))


def make_needle(a) -> int:
    tok, tmpl = tokenizer_and_template(a.model)
    text = Path(a.context_file).read_text()
    paras = [p for p in text.split("\n\n") if p.strip()]
    n_text = len(tok.encode(text, add_special_tokens=False).ids)
    secret = "AMBER-HERON-60417"
    needle = f"(A note found in the archive: the vault passphrase is {secret}. Remember it.)"
    reps = max(1, a.depth // max(n_text, 1))
    before = "\n\n".join([text] * reps)
    # top up with paragraphs until ~depth tokens precede the needle
    have = n_text * reps
    extra = []
    for p in paras:
        if have >= a.depth:
            break
        extra.append(p)
        have += len(tok.encode(p, add_special_tokens=False).ids) + 1
    after = "\n\n".join(paras[: max(4, len(paras) // 8)])
    body = (before + ("\n\n" + "\n\n".join(extra) if extra else "") + "\n\n" + needle + "\n\n" + after +
            "\n\nQuestion: what is the vault passphrase in the archive note? Answer with the passphrase.")
    rendered = tmpl.render([{"role": "user", "content": body}], tools=None, enable_thinking=True)
    ids = [int(t) for t in tok.encode(rendered, add_special_tokens=False).ids]
    nids = tok.encode(needle, add_special_tokens=False).ids
    doc = {"generator": "tools/glm53/gate3b.py make-needle", "needle_secret": secret, "needle_depth_est": have,
           "sampling": {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0},
           "context_needed": len(ids) + a.tokens,
           "prompts": [{"name": "needle1m", "ids": ids, "tokens": a.tokens, "runs": ["g:2"], "seed": 1729,
                        "seed_from_prompt": False}]}
    Path(a.out).write_text(json.dumps(doc) + "\n")
    say(f"needle: {len(ids)} prompt tokens, needle ({len(nids)} tokens) after ~{have}, secret {secret}")
    return 0


def make_dcp(a) -> int:
    tok, tmpl = tokenizer_and_template(a.model)
    text = Path(a.context_file).read_text()
    ids_all = tok.encode(text, add_special_tokens=False).ids
    body = tok.decode(ids_all[: a.length]) + "\n\nIn one sentence, what is this text about?"
    rendered = tmpl.render([{"role": "user", "content": body}], tools=None, enable_thinking=True)
    ids = [int(t) for t in tok.encode(rendered, add_special_tokens=False).ids]
    short = tmpl.render([{"role": "user", "content": "Name three rivers."}], tools=None, enable_thinking=True)
    sids = [int(t) for t in tok.encode(short, add_special_tokens=False).ids]
    doc = {"generator": "tools/glm53/gate3b.py make-dcp", "sampling": {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0},
           "context_needed": len(ids) + a.tokens,
           "prompts": [{"name": "dcp-long", "ids": ids, "tokens": a.tokens, "runs": ["g:2", "g:0", "s20:2"],
                        "seed": 1729, "seed_from_prompt": False},
                       {"name": "dcp-short", "ids": sids, "tokens": a.tokens, "runs": ["g:2", "s20:2"],
                        "seed": 1730, "seed_from_prompt": False}]}
    Path(a.out).write_text(json.dumps(doc) + "\n")
    say(f"dcp capture prompts: {len(ids)} and {len(sids)} tokens")
    return 0


# --------------------------------------------------------------------------------------------------- gates ---
def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def by_label(conv_doc, dump: list[dict], offset: int = 0) -> dict:
    """conv requests (in order) matched with the dump's lines from ``offset``; checks prompt lengths agree."""
    out = {}
    if not conv_doc:
        return out
    for i, r in enumerate(conv_doc["requests"]):
        d = dump[offset + i] if offset + i < len(dump) else None
        pt = (r.get("usage") or {}).get("prompt_tokens")
        if d is not None and pt is not None and pt != len(d["prompt_ids"]):
            d = dict(d, mismatch=f"usage {pt} vs dump {len(d['prompt_ids'])}")
        out[r["label"]] = (r, d)
    return out


def fmt(x, n=3):
    return "-" if x is None else round(x, n)


def gate(a) -> int:
    w = Path(a.work)
    lines = []
    ok_all = True

    def result(name: str, ok: bool, text: str) -> None:
        nonlocal ok_all
        ok_all = ok_all and ok
        lines.append(f"GATE {name} {'PASS' if ok else 'FAIL'} {text}")

    # (a) reuse
    dump_a = read_dump(str(w / "dump-A.jsonl"))
    conv_a = load(w / "convA.json")
    la = by_label(conv_a, dump_a)
    a_ok, why = bool(la), []
    for n in (2, 3):
        wr, cr = la.get(f"warm{n}"), la.get(f"cold{n}")
        if not wr or not cr or wr[1] is None or cr[1] is None:
            a_ok = False
            why.append(f"turn {n}: missing")
            continue
        wd, cd = wr[1], cr[1]
        if wd.get("mismatch") or cd.get("mismatch"):
            a_ok = False
            why.append(f"turn {n}: dump out of step ({wd.get('mismatch') or cd.get('mismatch')})")
        if wd["prompt_ids"] != cd["prompt_ids"]:
            a_ok = False
            why.append(f"turn {n}: warm/cold prompts differ")
        same = wd["tokens"] == cd["tokens"]
        a_ok = a_ok and same and wd.get("begin", 0) > 0 and cd.get("begin", 0) == 0
        why.append(f"t{n} warm==cold {same} ({len(wd['tokens'])} tok) begin {wd.get('begin')}/{cd.get('begin')}")
    sr, wr3 = la.get("serial3"), la.get("warm3")
    if sr and wr3 and sr[1] and wr3[1]:
        same = sr[1]["tokens"] == wr3[1]["tokens"]
        a_ok = a_ok and same
        why.append(f"serial3==warm3 {same}")
    else:
        a_ok = False
        why.append("serial3 missing")
    result("a", a_ok, "; ".join(why))
    conv_p = load(w / "convP.json")
    lp = {r["label"]: r for r in conv_p["requests"]} if conv_p else {}
    for n in (2, 3):
        zw = (la.get(f"warm{n}") or ({}, None))[0].get("ttft")
        zc = (la.get(f"cold{n}") or ({}, None))[0].get("ttft")
        zd = ((la.get(f"warm{n}") or (None, {}))[1] or {}).get("ttft_s")
        pw = (lp.get(f"warm{n}") or {}).get("ttft")
        pc = (lp.get(f"cold{n}") or {}).get("ttft")
        soft = "" if not (zw and pw) else f" zig/py {zw / pw:.2f} ({'ok' if zw <= 1.10 * pw else 'over 1.10'})"
        lines.append(f"REPORT ttft turn {n}: zig warm {fmt(zw)} s (engine {fmt(zd)}), cold {fmt(zc)}; python warm "
                     f"{fmt(pw)}, cold {fmt(pc)}{soft}")
    if conv_p:
        same_text = [lp.get(f"warm{n}", {}).get("content") == la.get(f"warm{n}", ({}, None))[0].get("content")
                     for n in (1, 2, 3)]
        lines.append(f"REPORT python vs zig reply text equal (info, warm 1..3): {same_text}")

    # (b) --learn
    dump_b, dump_c = read_dump(str(w / "dump-B.jsonl")), read_dump(str(w / "dump-C.jsonl"))
    conv_c = load(w / "convC.json")
    lc = by_label(conv_c, dump_c)
    b_ok, why = bool(lc) and bool(dump_b), []
    if dump_b and la.get("warm1") and la["warm1"][1]:
        s1 = dump_b[0]["tokens"] == la["warm1"][1]["tokens"]
        why.append(f"turn1 after learn == A {s1}")
        b_ok = b_ok and s1
    for n in (2, 3):
        cr, ar = lc.get(f"warm{n}"), la.get(f"cold{n}")
        if not cr or not ar or cr[1] is None or ar[1] is None:
            b_ok = False
            why.append(f"turn {n}: missing")
            continue
        same = cr[1]["tokens"] == ar[1]["tokens"] and cr[1]["prompt_ids"] == ar[1]["prompt_ids"]
        b_ok = b_ok and same
        why.append(f"t{n} restarted==cold {same}")
    first = lc.get("warm2")
    learned = (first[1] or {}).get("learned", 0) if first else 0
    b_ok = b_ok and learned > 0
    why.append(f"learned {learned} tokens from disk, begin {((first or (None, {}))[1] or {}).get('begin')}")
    if first:
        tc = (la.get("cold2") or ({}, None))[0].get("ttft")
        why.append(f"ttft from disk {fmt(first[0].get('ttft'))} s vs cold {fmt(tc)} s")
    result("b", b_ok, "; ".join(why))

    # (c) 1M
    g1 = load(w / "gen1m-r0.json")
    p1 = load(w / "prompts-1m.json")
    c_ok, why = bool(g1 and p1), []
    if g1 and p1:
        run = g1["runs"][0]
        text = ""
        try:
            tok, _ = tokenizer_and_template(a.model)
            text = tok.decode(run["tokens"], skip_special_tokens=False)
        except Exception as e:  # noqa: BLE001
            why.append(f"decode failed: {e}")
        found = p1["needle_secret"] in text
        c_ok = found
        why.append(f"needle {'found' if found else 'NOT found'} ({run['prompt_len']} prompt tokens, "
                   f"{len(run['tokens'])} picks), decode {fmt(run.get('tok_s'), 2)} tok/s, prefill "
                   f"{fmt(run.get('prefill_s'), 1)} s")
        ref = load(w / "ref1m" / "ref-r0.json")
        if a.py1m and ref:
            pt = ref["runs"][0]["tok_s"]
            sp = run.get("tok_s", 0) >= a.ratio * pt
            c_ok = c_ok and sp
            why.append(f"python {pt} tok/s at DCP {ref.get('dcp')}: zig/py {run.get('tok_s', 0) / max(pt, 1e-9):.3f} "
                       f"({'>=' if sp else '<'} {a.ratio}); python found needle "
                       f"{p1['needle_secret'] in tok.decode(ref['runs'][0]['tokens']) if text else '?'}")
        elif a.py1m:
            c_ok = False
            why.append("PY1M=1 but no python 1M result")
        else:
            why.append("python 1M not run (PY1M=0): gated on needle + memory only")
    else:
        why.append("no 1M result")
    if a.mem1m:
        why.append(f"min MemAvailable GB during 1M {a.mem1m}")
        try:
            low = min(float(x.split(":")[1]) for x in a.mem1m.split(",") if ":" in x)
            c_ok = c_ok and low >= 4.0
        except ValueError:
            pass
    result("c", c_ok, "; ".join(why))

    # (d) server
    apid = load(w / "apiA.json")
    d_ok, why = bool(apid), []
    if apid:
        for k in ("nonstream", "stream", "tool", "hermes"):
            d_ok = d_ok and apid[k]["ok"]
            why.append(f"{k} {'ok' if apid[k]['ok'] else 'FAIL'}")
    cli = load(w / "cli" / "gen-r0.json")
    cp = load(w / "cli-prompts.json")
    if cli and cp:
        dumps = read_dump(str(w / "dump-A.jsonl"))
        runs = {r["name"]: r for r in cli["runs"]}
        eq = tot = 0
        diffs = []
        for p in cp["prompts"]:
            tot += 1
            r = runs.get(p["name"])
            want = dumps[p["dump_index"]]["tokens"]
            if r is not None and r["tokens"] == want:
                eq += 1
            else:
                got = r["tokens"] if r else None
                at = next((i for i, (x, y) in enumerate(zip(got or [], want)) if x != y),
                          min(len(got or []), len(want)))
                diffs.append(f"{p['name']}@{at}")
        d_ok = d_ok and tot > 0 and eq == tot
        why.append(f"served==CLI {eq}/{tot}" + (f" (first differences {diffs[:4]})" if diffs else ""))
    else:
        d_ok = False
        why.append("no CLI comparison")
    result("d", d_ok, "; ".join(why))
    for ln in lines:
        print(ln, flush=True)
    Path(w / "gate.txt").write_text("\n".join(lines) + "\n")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-conv")
    p.add_argument("--context-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=300)
    p = sub.add_parser("conv")
    p.add_argument("--url", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warm", default="1,2,3")
    p.add_argument("--cold", default="")
    p.add_argument("--serial", default="")
    p.add_argument("--flush-file", default="")
    p.add_argument("--history", default="")
    p.add_argument("--idle", type=float, default=0, help="seconds without a byte that fail a request (0: 7200)")
    p = sub.add_parser("api")
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--idle", type=float, default=0, help="seconds without a byte that fail a request (0: 7200)")
    p = sub.add_parser("dump2cli")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--skip", type=int, default=0)
    p = sub.add_parser("make-needle")
    p.add_argument("--model", required=True)
    p.add_argument("--context-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--depth", type=int, default=830000)
    p.add_argument("--tokens", type=int, default=256)
    p = sub.add_parser("make-dcp")
    p.add_argument("--model", required=True)
    p.add_argument("--context-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--length", type=int, default=20000)
    p.add_argument("--tokens", type=int, default=16)
    p = sub.add_parser("gate")
    p.add_argument("--work", required=True)
    p.add_argument("--model", default="/model")
    p.add_argument("--mem1m", default="")
    p.add_argument("--py1m", type=int, default=0)
    p.add_argument("--ratio", type=float, default=0.97)
    a = ap.parse_args()
    global IDLE_S
    if getattr(a, "idle", None):
        IDLE_S = float(a.idle)
    return {"make-conv": make_conv, "conv": conv, "api": api, "dump2cli": dump2cli, "make-needle": make_needle,
            "make-dcp": make_dcp, "gate": gate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
