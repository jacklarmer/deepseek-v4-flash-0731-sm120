# Measured results

4× RTX PRO 6000 Blackwell (SM120), TP4, PCIe-only host with P2P disabled.
`deepseek-ai/DeepSeek-V4-Flash-0731` (official weights, MXFP4 experts + FP8 block
attention/dense), DSpark speculative decoding at K5, SparkInfer/B12X MoE backend.

Token counts come from streamed `usage`, never SSE chunk counts.

> **Correction (2026-08-01).** An earlier revision of this file said prefix caching
> was disabled for all measurements. That was wrong. vLLM V1 enables prefix caching
> by default and `--no-enable-prefix-caching` was never passed; the server reports a
> 73–78% hit rate. The decode numbers below are unaffected — every timed prompt is
> 24–85 tokens and `--block-size` is 256, so no prompt ever completes a single
> cacheable block. This was verified directly: the same code prompt repeated scores
> 508.8, and with a unique nonce per call scores 513.0, a 9 ms spread. The
> **prefill** numbers are a different story — see below.

---

## The finding that matters most: workload dominates every flag

Same server, same configuration, measured minutes apart:

| Workload | tok/s (single stream) |
|---|---:|
| Code, temperature 0 | **345 – 358** |
| Prose, temperature 0.6 | 250 – 283 |

That is a **~40% swing from the prompt alone**, and it is larger than the total
spread of every tuning flag tested below combined.

The mechanism, per community measurement on this model: with DSpark the engine
runs at a roughly fixed verification-step rate, so

```
tok/s ≈ (verification steps/s) × (1 + accepted draft tokens per step)
```

Draft acceptance is far higher on deterministic code than on open-ended prose.
Benchmark the workload you actually run, and always report which one you used.

---

## "Prefill tok/s" on leaderboards is not a throughput measurement

It is a derived ratio: **`promptTokens ÷ TTFT`**. Confirmed to the decimal against
two published entries from this server (25 tok / 51.01 ms → 490.1; 85 tok /
137.36 ms → 618.8).

Because TTFT has a fixed floor of roughly 40–50 ms on this stack (HTTP round trip,
tokenization, scheduler admission, first forward, speculative draft head), the ratio
scales almost linearly with prompt length until the prompt is large enough to
dominate that floor. Same server, same config, only the prompt changed:

| Prompt tokens | Prefill tok/s, prefix cache warm | Prefill tok/s, **cache-busted** |
|---:|---:|---:|
| 24 | 508.8 | 513.0 |
| 273 | 4,934.8 | 1,755.8 |
| 802 | 16,401.0 | 5,753.5 |
| 2,113 | 15,326.2 | 10,652.6 |
| 8,392 | 58,745.3 | 11,226.4 |
| 33,485 | 180,455.7 | **11,269.4** |

Two things fall out of this.

**The cached column is meaningless as a hardware measure.** 180,455 tok/s on a 304B
MoE would require roughly 11 PFLOPS of FP8, several times what four of these cards
can physically deliver. It is measuring cache lookups. The cache-busted column
converges to a stable **~11,200 tok/s**, which is physically coherent and is the
honest sustained prefill figure for this box. A published TP2 measurement of the
same model independently reports ~10.6k tok/s, which corroborates it.

**Never compare prefill figures across entries without knowing prompt length and
cache state.** A 24-token prompt and a 33k-token prompt on identical hardware differ
by 778× on the warm path. Report prompt length alongside any prefill number, and say
whether the cache was warm.

---

## Tuning sweep — every lever lost

Greedy single-variable sweep, code at temperature 0, 5-run median per config, each
lever kept only if it beat the running best.

| Config | tok/s | Verdict |
|---|---:|---|
| **baseline** (`--block-size 256`) | **357.9** | **kept** |
| `MOE_FORCE_A16=1` | 354.8 | reverted |
| `BREAKABLE_CUDAGRAPH=0` | 354.6 | reverted |
| `MAX_NUM_SEQS=16` | 353.0 | reverted |
| `WO_PROJECTION=1` | 352.7 | reverted |
| `GPU_MEM_UTIL=0.95` | 347.8 | reverted |

Total spread: 347.8 – 357.9, about **2.8%**. On a five-sample median that is close
to run-to-run scatter, so the honest reading is that these levers are **neutral on
this hardware**, not that the baseline is finely tuned.

### Negative results worth recording

**`VLLM_USE_BREAKABLE_CUDAGRAPH=0` does not transfer from SM121 to SM120.** A
widely-cited GB10/DGX Spark measurement puts this at **+28.6%** on single-stream
decode, because the flag being set disables vLLM's `torch.compile` pipeline
entirely (`Equivalent to -cc.mode=none`). On RTX PRO 6000 it measured *slower*
than leaving it auto-enabled. Do not assume Spark tuning carries over.

**`B12X_MOE_FORCE_A16=1` did not win either**, despite a published RTX PRO 6000
backend sweep showing B12X-A16 fastest at concurrency 1. That sweep used a
different checkpoint revision.

**The only flag that helped was `--block-size 256`**, worth roughly +2% over
leaving it unset (351.2 → 357.9), and mandatory anyway (see below).

---

## Concurrency (code, temperature 0)

| Concurrency | Per-stream | Aggregate |
|---:|---:|---:|
| 1 | 345.4 | 315.7 |
| 2 | 318.4 | 576.9 |
| 4 | 243.9 | 823.3 |
| 8 | 183.0 | **1260.5** |

Aggregate scales cleanly to 8 concurrent streams. Per-stream degrades as expected
under scheduler contention.

---

## Correctness gates

**10/10 pass** on the winning configuration:

auth 401 without key · model identity and advertised context · exact output at
temperature 0 · reasoning separated from content · strict JSON via `response_format`
· automatic tool call · forced tool call · tool-result roundtrip · SSE streaming ·
in-flight cancellation with clean recovery.

Re-run these after any config change. A speed win that broke forced tool calls is
not a win.

---

## Published to LocalMaxxing

Measured with the `lmx` harness (2 warmups, 5 timed iterations, median) rather than
the harness above, so the numbers are comparable to other board entries.

Two entries from the same server, same configuration, **differing only in the
prompt**. `lmx benchmark run` accepts `--prompt` for remote endpoints (unlike
`--temperature`, which applies only to eval-shard runs):

| Metric | Default lmx prompt | Code-generation prompt |
|---|---:|---:|
| promptTokens | 25 | 85 |
| tokSOut | 225.0 | **345.8** |
| tokSPrefill | 490.1 | 618.8 |
| tokSTotal | 231.3 | 369.6 |
| ttftMs | 51.01 | 137.36 |

The 345.8 figure lands within 0.4 tok/s of the independent codebench measurement of
345.4 — two unrelated harnesses agreeing that closely is good evidence neither is
measuring an artifact. Both submissions carry an explicit disclosure of which prompt
produced them.

Note that neither `tokSPrefill` figure is a hardware capability number; both are
overhead-dominated at these prompt lengths. See the prefill section above.

Submitted as `quantization: FP8` to match how the rest of the board files this
checkpoint. An earlier submission of the same server used `MXFP4` — technically
accurate for the experts, but it lands the entry in a separate bucket where nobody
will compare it. Match the community's label.

---

## Failure modes catalogued along the way

### `--block-size 16` is invalid for the DS-MLA layout

```
AssertionError
  File "vllm/v1/core/kv_cache_utils.py", line 1684, in _get_kv_cache_groups_uniform_groups
    assert max(sm_page_sizes) <= max(all_page_sizes)
```

Fails several minutes into boot, after weights load, with a traceback that points
at KV cache internals rather than at the flag. Use **256**. This one cost two
entire sweep runs before it was isolated.

### The r16 release image fails two ways on this box

`voipmonitor/vllm:gilded-gnosis-v20-...-20260731-r16` is the purpose-built
DeepSeek-V4 release image and is validated on **two-GPU TP2** setups. On this
four-GPU TP4 host:

- **`BACKEND=b12x-*`** — loads weights (41 GiB/rank), completes `torch.compile`,
  then hangs indefinitely in the memory-profiling forward pass. All GPUs at 100%,
  KV cache never allocated, zero further log output. Not fixed by disabling
  flashinfer autotune, nor by reducing `MAX_NUM_SEQS` 16 → 8.
- **`BACKEND=lucifer-*`** — dies in ~90s:
  `Attempted to call function marked as skipped ... Dynamo does not know how to
  trace the builtin deep_gemm._C.pybind11_detail_function_record_...`

The older SparkInfer 0.30.0 image works because it does **not** route DeepSeek-V4
through `--attention-backend B12X_MLA_SPARSE`.

### `ALLREDUCE_MODE=b12x` does not set `NCCL_P2P_DISABLE`

It sets `VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=0`, which reads like it covers you. It
does not. On a host that deadlocks on PCIe P2P, the symptom is:

```
0, 1044 MiB, 100 %
1, 1044 MiB, 100 %
2, 1044 MiB, 100 %
3, 1052 MiB, 100 %
```

All GPUs pinned at 100% with ~1 GB allocated, logs stopped right after
`world_size=N ... backend=nccl`. Set `NCCL_P2P_DISABLE=1` yourself.

### The LocalMaxxing CLI silently fails to submit

`lmx` v0.1.30 posts speed tests to `/api/benchmarks`, which returns **HTTP 200 with
the site's HTML**. The CLI sees a 2xx and prints
`benchmark_submitted / status: submitted` while nothing is recorded.

The live endpoint is `/api/speed-tests`. Validate at `/api/speed-tests/dry-run`
first — it does not consume rate limit — then POST. A real submission returns
**HTTP 201** with a record id and `"status": "APPROVED"`.

Use `curl`, not `python-urllib`: Cloudflare returns **403 error 1010** (browser
signature block) for the default `Python-urllib/3.x` user agent.
