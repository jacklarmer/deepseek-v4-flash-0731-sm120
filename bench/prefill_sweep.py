#!/usr/bin/env python3
"""Prompt-length sweep proving 'prefill tok/s' is promptTokens/TTFT, not throughput.

Runs each prompt length twice: with a repeated (cacheable) prompt, and with a unique
leading nonce per request so no prompt can reuse another's prefix-cache blocks. The
gap between the two columns is the prefix cache; the cache-busted column converging
to a flat ceiling is the machine's real sustained prefill rate.

Usage:
    export VLLM_API_KEY=...        # or edit auth below
    python3 prefill_sweep.py <port> <served-model-name>

See RESULTS.md ("prefill on leaderboards") for measured output on 4x RTX PRO 6000.
"""
import json, os, statistics, sys, time, urllib.request

PORT, MODEL = sys.argv[1], sys.argv[2]
KEY = os.environ["VLLM_API_KEY"]
FILLER = ("The subsystem records each transaction in an append-only ledger and "
          "reconciles the balance against the upstream snapshot on every commit. ")
N = [0]

def one(prompt):
    body = json.dumps({"model": MODEL, "stream": True, "temperature": 0, "max_tokens": 32,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; ptoks = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            if not raw.startswith(b"data: "): continue
            c = raw[6:].strip()
            if c == b"[DONE]": break
            d = json.loads(c)
            if d.get("usage"): ptoks = d["usage"]["prompt_tokens"]
            ch = d.get("choices") or []
            if ttft is None and ch:
                dl = ch[0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"):
                    ttft = (time.perf_counter() - t0) * 1000
    return ttft, ptoks

def build(target, nonce):
    # The nonce LEADS the prompt, so every request diverges at block 0.
    return f"[trace {nonce:09d}] Summarize this log in one sentence.\n\n" + \
        FILLER * max(1, round(target / 22))

print(f"{'promptTok':>10} {'warm TTFT':>10} {'warm tok/s':>11} | {'busted TTFT':>12} {'busted tok/s':>13}")
print("-" * 64)
for target in (250, 750, 2000, 8000, 32000):
    p = build(target, 0); one(p)                       # warmup + seed the cache
    warm = [one(p) for _ in range(3)]
    w_ttft = statistics.median(s[0] for s in warm); ptoks = warm[0][1]
    cold = []
    for _ in range(3):
        N[0] += 1
        cold.append(one(build(target, N[0])))
    c_ttft = statistics.median(s[0] for s in cold)
    print(f"{ptoks:>10} {w_ttft:>9.1f}ms {ptoks/(w_ttft/1000):>11.1f} | "
          f"{c_ttft:>11.1f}ms {ptoks/(c_ttft/1000):>13.1f}")

print("\nReport the cache-busted column. Always state prompt length with any prefill figure.")
