# Measured results

4× RTX PRO 6000 Blackwell (SM120), TP4, PCIe-only host with P2P disabled.
`deepseek-ai/DeepSeek-V4-Flash-0731` (official weights, MXFP4 experts + FP8 block
attention/dense), DSpark speculative decoding at K5, SparkInfer/B12X MoE backend.

Prefix caching **disabled** for all measurements. Token counts come from streamed
`usage`, never SSE chunk counts.

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
the harness above, so the numbers are comparable to other board entries. lmx uses a
short prose prompt, which is the low-acceptance regime — hence the gap from the
code figures.

| Metric | Value |
|---|---:|
| tokSOut | 225.0 |
| tokSPrefill | 490.1 |
| tokSTotal | 231.3 |
| ttftMs | 51.01 |

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
