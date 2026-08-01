#!/usr/bin/env python3
"""Code-at-temperature-0 decode benchmark with draft-acceptance capture.

Rationale (Light Foundry, 2026-08-01): with DSpark the engine runs at a roughly
fixed verification-step rate, so throughput is ~= steps/s * (1 + accepted draft
tokens per step). Acceptance is workload dependent -- ~73% on code at temp 0,
~33% on prose -- so a prose benchmark measures the pessimistic regime.

Token counts come from streamed usage (include_usage), never from SSE delta
counts: with speculative decoding one SSE chunk can carry several tokens.

usage: codebench.py PORT MODEL [max_tokens] [concurrency] [runs]
"""
import os, sys, json, time, urllib.request, threading, statistics, re

PORT, MODEL = sys.argv[1], sys.argv[2]
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 512
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 1
RUNS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
KEY = os.environ["VLLM_API_KEY"]
BASE = f"http://127.0.0.1:{PORT}"

CODE_PROMPT = (
    "Write a complete, production-quality Python module that implements a "
    "thread-safe LRU cache with TTL expiry. Include: the full class with type "
    "hints, __getitem__/__setitem__/__delitem__, a background sweeper thread, "
    "explicit locking, a stats() method returning hits/misses/evictions, and "
    "pytest unit tests covering eviction order, TTL expiry, and concurrent "
    "access. Output only code."
)


def spec_counters():
    """Scrape spec-decode acceptance counters from /metrics if exposed."""
    try:
        req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
        body = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    except Exception:
        return {}
    out = {}
    for name in ("spec_decode_num_accepted_tokens_total",
                 "spec_decode_num_draft_tokens_total",
                 "spec_decode_num_emitted_tokens_total"):
        m = re.findall(rf"^{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+]+)$", body, re.M)
        if m:
            out[name] = sum(float(x) for x in m)
    return out


def one(i, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": CODE_PROMPT}],
            "max_tokens": MAXTOK, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t_first = t_last = None; ctoks = 0; chunks = 0; t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except Exception:
                continue
            ch = j.get("choices") or []
            if ch:
                d = ch[0].get("delta", {})
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    now = time.time(); chunks += 1
                    if t_first is None:
                        t_first = now
                    t_last = now
            u = j.get("usage")
            if u and u.get("completion_tokens"):
                ctoks = u["completion_tokens"]
    ttft = (t_first - t0) if t_first else float("nan")
    dec = (t_last - t_first) if (t_first and t_last and t_last > t_first) else float("nan")
    out[i] = (ctoks, ttft, dec, (ctoks / dec) if dec == dec and dec > 0 else float("nan"), chunks)


# warmup
w = {}; one(0, w)
before = spec_counters()

per_run, agg = [], []
for _ in range(RUNS):
    out = {}
    ths = [threading.Thread(target=one, args=(i, out)) for i in range(CONC)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
    tot = sum(o[0] for o in out.values())
    per_run.extend(o[3] for o in out.values() if o[3] == o[3])
    agg.append(tot / wall)

after = spec_counters()
acc = None
try:
    da = after.get("spec_decode_num_draft_tokens_total", 0) - before.get("spec_decode_num_draft_tokens_total", 0)
    aa = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get("spec_decode_num_accepted_tokens_total", 0)
    if da > 0:
        acc = aa / da
except Exception:
    pass

label = os.environ.get("SWEEP_LABEL", "run")
print(json.dumps({
    "label": label, "conc": CONC, "runs": RUNS, "max_tokens": MAXTOK,
    "per_stream_tok_s_median": round(statistics.median(per_run), 1) if per_run else None,
    "per_stream_tok_s_min": round(min(per_run), 1) if per_run else None,
    "per_stream_tok_s_max": round(max(per_run), 1) if per_run else None,
    "aggregate_tok_s_median": round(statistics.median(agg), 1) if agg else None,
    "draft_acceptance": round(acc, 4) if acc is not None else None,
    "chunks_last_run": max((o[4] for o in out.values()), default=None),
}))
