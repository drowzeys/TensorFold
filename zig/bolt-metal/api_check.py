"""OpenAI API checks and decode timing for a Bolt server: chat, streaming, forced tool calls, JSON schema, speed.

python3 api_check.py http://127.0.0.1:8093 [MODEL_ID] [--speed-only] [--parallel N]
"""
import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "nvidia-nemotron-3.5-bolt"
SPEED_ONLY = "--speed-only" in sys.argv
PAR = int(sys.argv[sys.argv.index("--parallel") + 1]) if "--parallel" in sys.argv else 0

PROMPTS = {
    "prose": [{"role": "user", "content": "Write a vivid short story (about 400 words) about a cyberpunk heist: a crew of three slips into a neon-drenched orbital bank vault during a solar storm, and the getaway goes wrong in the last minute."}],
    "code": [{"role": "user", "content": "Write a Python module that simulates a street-dance battle bracket: dancers with skill ratings, a seeded single-elimination bracket, judges scoring each round with randomness, and a CLI that prints each round. Include type hints and unit tests."}],
    "shots": [{"role": "system", "content": "You are the First AD of a film crew. Answer concisely."}, {"role": "user", "content": "Break this scene into 6 numbered shots with lens and camera-move notes: a sci-fi chase through a flooded megacity at night, a hover-bike weaving between holographic billboards, a drone swarm closing in, and the rider jumping onto a moving maglev train."}],
}


def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "temperature": 0, **kw}
    with post("/v1/chat/completions", body) as r:
        return json.load(r)


def stream_timed(messages, max_tokens):
    """Streams a reply; returns (tokens, first-token seconds, decode tok/s after the first token, text)."""
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    first = None
    text = []
    usage = None
    with post("/v1/chat/completions", body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content")
                if piece:
                    if first is None:
                        first = time.time()
                    text.append(piece)
    t1 = time.time()
    n = usage["completion_tokens"] if usage else len(text)
    rate = (n - 1) / (t1 - first) if first and t1 > first and n > 1 else 0.0
    return n, (first - t0) if first else None, rate, "".join(text)


def checks():
    ok = True
    r = chat([{"role": "user", "content": "In one sentence: what makes a great street-dance battle?"}], max_tokens=80)
    msg = r["choices"][0]["message"]
    print("chat:", r["model"], "|", (msg.get("content") or "").strip()[:160].replace("\n", " "))
    ok &= bool(msg.get("content")) and "<think>" not in (msg.get("content") or "")
    n, ttft, rate, text = stream_timed([{"role": "user", "content": "Name three iconic sci-fi chase scenes, one line each."}], 120)
    print(f"stream: {n} tokens, first token {ttft:.2f} s, {rate:.1f} tok/s |", text.strip()[:120].replace("\n", " "))
    ok &= n > 5
    tools = [{"type": "function", "function": {"name": "book_shot", "description": "Schedule a film shot", "parameters": {"type": "object", "properties": {"scene": {"type": "string"}, "lens_mm": {"type": "integer"}, "move": {"type": "string", "enum": ["dolly", "crane", "handheld", "static"]}}, "required": ["scene", "lens_mm", "move"]}}}]
    r = chat([{"role": "user", "content": "Plan the opening shot of a cyberpunk heist on a rooftop at night."}], tools=tools, tool_choice="required", max_tokens=200)
    calls = r["choices"][0]["message"].get("tool_calls") or []
    print("tool_choice required:", json.dumps([c["function"] for c in calls])[:240])
    try:
        args = json.loads(calls[0]["function"]["arguments"])
        ok &= calls[0]["function"]["name"] == "book_shot" and isinstance(args.get("lens_mm"), int) and args.get("move") in ("dolly", "crane", "handheld", "static")
    except Exception as e:
        print("  tool call not parsed:", e)
        ok = False
    schema = {"type": "object", "properties": {"crew": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}}, "required": ["name", "role"]}}, "vault_floor": {"type": "integer"}}, "required": ["crew", "vault_floor"]}
    r = chat([{"role": "user", "content": "Invent the crew of an orbital bank heist and the vault's floor number."}], max_tokens=300, response_format={"type": "json_schema", "json_schema": {"name": "heist", "schema": schema}})
    content = r["choices"][0]["message"].get("content") or ""
    print("json_schema:", content.strip()[:200].replace("\n", " "))
    try:
        obj = json.loads(content)
        ok &= isinstance(obj.get("vault_floor"), int) and isinstance(obj.get("crew"), list) and all("name" in c and "role" in c for c in obj["crew"])
    except Exception as e:
        print("  not JSON:", e)
        ok = False
    print("API CHECKS", "PASS" if ok else "FAIL")


def speed(max_tokens=512, reps=2):
    for name, msgs in PROMPTS.items():
        best = None
        for _ in range(reps):
            n, ttft, rate, _ = stream_timed(msgs, max_tokens)
            if best is None or rate > best[2]:
                best = (n, ttft, rate)
        print(f"speed {name}: {best[0]} tokens, first token {best[1]:.2f} s, decode {best[2]:.1f} tok/s")


def parallel(n, max_tokens=384):
    names = list(PROMPTS)
    out = [None] * n
    def run(i):
        out[i] = stream_timed(PROMPTS[names[i % len(names)]], max_tokens)
    t0 = time.time()
    th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    total = sum(o[0] for o in out)
    print(f"parallel {n}: {total} tokens in {wall:.1f} s = {total / wall:.1f} tok/s aggregate; per stream " + ", ".join(f"{o[2]:.1f}" for o in out))


if __name__ == "__main__":
    if PAR:
        parallel(PAR)
    else:
        if not SPEED_ONLY:
            checks()
        speed()
