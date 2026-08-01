#!/usr/bin/env python3
"""Acceptance gates for a River House brain candidate. usage: gates.py PORT MODEL"""
import os, sys, json, urllib.request, urllib.error, time, threading

PORT, MODEL = sys.argv[1], sys.argv[2]
KEY = os.environ["VLLM_API_KEY"]
BASE = f"http://127.0.0.1:{PORT}"
results = []


def post(path, body, key=KEY, stream=False, timeout=180):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=h)
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else json.loads(r.read())


def gate(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {str(e)[:130]}"
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def g_auth_denied():
    try:
        post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}, key=None)
        return False, "no-key request was ACCEPTED"
    except urllib.error.HTTPError as e:
        return e.code == 401, f"no key -> HTTP {e.code}"


def g_identity():
    req = urllib.request.Request(BASE + "/v1/models", headers={"Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(req, timeout=20).read())["data"][0]
    return d["id"] == MODEL, f"id={d['id']} max_model_len={d.get('max_model_len')}"


def g_exact():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0,
             "messages": [{"role": "user", "content": "Reply with exactly the single word AWAKE and nothing else."}],
             "max_tokens": 2000})
    t = (d["choices"][0]["message"].get("content") or "").strip()
    return t == "AWAKE", f"content={t!r}"


def g_reasoning_separated():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0,
             "messages": [{"role": "user", "content": "What is 17*23? Think briefly, then answer."}],
             "max_tokens": 3000})
    m = d["choices"][0]["message"]
    rc = m.get("reasoning_content") or m.get("reasoning")
    c = m.get("content") or ""
    return ("391" in c), f"reasoning_field={'yes' if rc else 'no'} content_has_391={'391' in c}"


def g_json():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0,
             "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": 'Return ONLY JSON: {"city":"Paris","country":"France"}'}],
             "max_tokens": 3000})
    c = d["choices"][0]["message"]["content"]
    j = json.loads(c)
    return j.get("city") == "Paris", f"parsed={j}"


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def g_tool_auto():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "tools": TOOLS,
             "messages": [{"role": "user", "content": "What is the weather in Tokyo? Use the tool."}],
             "max_tokens": 3000})
    tc = d["choices"][0]["message"].get("tool_calls") or []
    if not tc:
        return False, "no tool_calls emitted"
    a = json.loads(tc[0]["function"]["arguments"])
    return tc[0]["function"]["name"] == "get_weather" and "tokyo" in str(a).lower(), f"{tc[0]['function']['name']}({a})"


def g_tool_forced():
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "tools": TOOLS,
             "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
             "messages": [{"role": "user", "content": "Weather in Oslo."}], "max_tokens": 3000})
    tc = d["choices"][0]["message"].get("tool_calls") or []
    return bool(tc) and tc[0]["function"]["name"] == "get_weather", f"tool_calls={len(tc)}"


def g_tool_roundtrip():
    d1 = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "tools": TOOLS,
              "messages": [{"role": "user", "content": "Weather in Cairo? Use the tool."}], "max_tokens": 3000})
    m1 = d1["choices"][0]["message"]
    tc = m1.get("tool_calls") or []
    if not tc:
        return False, "stage 1 emitted no tool_call"
    msgs = [{"role": "user", "content": "Weather in Cairo? Use the tool."},
            {"role": "assistant", "content": m1.get("content") or "", "tool_calls": tc},
            {"role": "tool", "tool_call_id": tc[0]["id"], "content": '{"temp_c": 41, "sky": "clear"}'}]
    d2 = post("/v1/chat/completions", {"model": MODEL, "temperature": 0, "tools": TOOLS,
              "messages": msgs, "max_tokens": 3000})
    c = (d2["choices"][0]["message"].get("content") or "")
    return "41" in c, f"final mentions 41: {'41' in c}"


def g_sse():
    r = post("/v1/chat/completions", {"model": MODEL, "stream": True,
             "messages": [{"role": "user", "content": "Count from 1 to 10."}], "max_tokens": 400}, stream=True)
    chunks = sum(1 for line in r if line.strip().startswith(b"data:"))
    return chunks > 3, f"{chunks} SSE chunks"


def g_cancel():
    state = {"got": 0}

    def go():
        try:
            r = post("/v1/chat/completions", {"model": MODEL, "stream": True,
                     "messages": [{"role": "user", "content": "Write a very long essay about turbines."}],
                     "max_tokens": 4000}, stream=True)
            for line in r:
                if line.strip().startswith(b"data:"):
                    state["got"] += 1
                    if state["got"] > 5:
                        r.close()
                        return
        except Exception:
            pass
    t = threading.Thread(target=go); t.start(); t.join(timeout=60)
    time.sleep(3)
    d = post("/v1/chat/completions", {"model": MODEL, "temperature": 0,
             "messages": [{"role": "user", "content": "Reply with exactly OK."}], "max_tokens": 1500})
    c = (d["choices"][0]["message"].get("content") or "").strip()
    return "OK" in c.upper(), f"aborted after {state['got']} chunks; post-cancel reply={c[:30]!r}"


for n, f in [("auth 401 without key", g_auth_denied), ("model identity", g_identity),
             ("exact output", g_exact), ("reasoning separation", g_reasoning_separated),
             ("strict JSON", g_json), ("tool call (auto)", g_tool_auto),
             ("tool call (forced)", g_tool_forced), ("tool-result roundtrip", g_tool_roundtrip),
             ("SSE streaming", g_sse), ("in-flight cancel + recover", g_cancel)]:
    gate(n, f)

p = sum(1 for _, ok, _ in results if ok)
print(f"\n{p}/{len(results)} gates passed")
sys.exit(0 if p == len(results) else 1)
