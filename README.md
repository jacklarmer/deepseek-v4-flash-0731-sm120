# DeepSeek-V4-Flash-0731 on 4× RTX PRO 6000 Blackwell (SM120)

A working, measured recipe for serving **`deepseek-ai/DeepSeek-V4-Flash-0731`** with
**DSpark speculative decoding** and the **SparkInfer/B12X** kernel library on four
RTX PRO 6000 Blackwell (SM120) GPUs, on a PCIe-only host where peer-to-peer is
disabled.

Everything here was measured on real hardware. Where a number is estimated or
unverified, it says so.

---

## Results

### Baseline — hand-rolled config on an older image

Published to the LocalMaxxing community leaderboard. Measured on an **older
(June) SparkInfer 0.30.0 image** with a hand-written compose, `--moe-backend
b12x` but **no** b12x MLA attention (see blocker 4 below), on a **prose** prompt.

| Metric | Value |
|---|---:|
| Decode (tok/s out) | 231.5 |
| Prefill | 497.9 tok/s |
| Total | 237.9 tok/s |
| TTFT | 50.21 ms |
| KV pool | 2,573,911 tokens @ 262,144 ctx (9.82× concurrency) |

`lmx` harness, 2 warmups + 5 timed iterations, median, prefix caching disabled,
token counts from streamed `usage`.

Concurrency on the same setup, fixed-prompt harness:

| Concurrency | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Aggregate tok/s | ~267 | 433.6 | 615.6 | 749.3 |

### Recommended setup — r16 release image

The configuration this repository ships uses the **r16 release image** (built
2026-07-31), which fixes the SparkInfer DSV4 bug and adds `--linear-backend
b12x`, `--block-size 256`, InstantTensor loading and flashinfer autotune. It is
a different and better starting point than the baseline above.

A single-variable tuning sweep across backends, all-reduce modes, draft depth,
sampling method and topology is in progress; results will be added here. Until
then, treat the numbers above as a floor measured on an inferior configuration,
not as this repository's result.

**Important caveat about the headline number.** Both figures above were measured
on a *prose* prompt. With DSpark, throughput is roughly

```
tok/s ≈ (verification steps/s) × (1 + accepted draft tokens per step)
```

and draft acceptance is strongly workload-dependent — community measurements put
it near **73% on code at temperature 0** and near **33% on prose**. A prose
benchmark therefore measures the pessimistic regime. Benchmark on the workload
you actually run.

---

## Hardware and host

| | |
|---|---|
| GPU | 4 × RTX PRO 6000 Blackwell, 96 GB each, **SM120** |
| Host | x86-64, 32-core workstation CPU, ~500 GB RAM |
| OS / driver | Ubuntu 26.04 LTS, kernel 7.x, driver 595.84, CUDA 13.2 |
| PCIe P2P | **Disabled** — this topology deadlocks at NCCL's first P2P collective |

Note SM120 (RTX PRO 6000 / RTX 5090) is **not** SM121 (DGX Spark / GB10), and
neither is SM100 (B200/GB200). Kernels are frequently built for one and not the
others. Check before assuming a recipe transfers.

---

## The four blockers, and how to get past them

These cost the most time. Each produces a distinct, searchable error.

### 1. DSpark requires `decode-context-parallel-size 1`

```
NotImplementedError: DSpark currently requires decode context parallel size 1.
```

If you are coming from a GLM-5.2 profile, it very likely uses DCP4. DeepSeek's
DSpark path cannot. Drop the DCP flags entirely.

### 2. `DeepseekV4` accepts only `fp8_ds_mla` KV

```
AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache,
got nvfp4_ds_mla
```

GLM-5.2 profiles commonly use `nvfp4_ds_mla`. DeepSeek-V4 asserts on it. Use
`--kv-cache-dtype fp8_ds_mla`.

The cost is about 1.52× lower KV density (`kv_gmem_stride` 656 vs 432 bytes),
which is free when a ~156 GiB checkpoint sits in 384 GiB of VRAM. Note the
common claim that FP4 KV halves memory is wrong — the BF16 RoPE block stays
uncompressed.

Separately: **NVFP4 KV cache does not work on SM120 at all.** vLLM issue #43562
was closed by adding a capability check that fails fast, because the trtllm-gen
FP4 FMHA kernels exist only for SM100/SM103. Use FP8.

### 3. B12X FP8 GEMM must be off with DSpark

```
RuntimeError: Assertion error (.../utils/layout.hpp:39): t.dim() == N
  ... in vllm/utils/deep_gemm.py, line 460, in fp8_einsum
```

`VLLM_USE_B12X_FP8_GEMM=1` drives `deep_gemm`'s `fp8_einsum` into a layout
assertion during DSpark drafter warmup. Set it to `0`. This was independently
documented by another group on a different image and reproduced here.

### 4. B12X sparse MLA cannot serve DeepSeek-V4 on older SparkInfer

```
TypeError: io_issue_gather_dsv4_nope() got an unexpected keyword argument
'scale_format'
```

On SparkInfer/B12X **0.30.0**, the DSV4 topk-128 multi-group prefill dispatcher
passes `scale_format=` to a gather helper that does not accept it. GLM never hits
this because it takes the `GLM_NSA` gate with a different layout.

Do **not** work around it by setting `B12X_MLA_SM120_PREFILL_MG=0` — the MG gates
are the only ones that match DSV4, and disabling them produces a hard failure
with a useful shape dump:

```
ValueError: SM120 sparse MLA prefill: unsupported shape
(model_type=0, heads=16, topk=128, compute_mode=0, ...,
 B12X_MLA_SM120_PREFILL_MG=0). ... No decode-reuse fallback.
```

The fix is either a newer SparkInfer, or omit `--attention-backend
B12X_MLA_SPARSE` and keep `--moe-backend b12x`. MoE is where the bulk of the
compute lives, so most of the benefit survives.

---

## Quick start

```bash
cp .env.example .env
# edit MODEL_ROOT and PORT
./ds4-ctl.sh up
./ds4-ctl.sh health
./ds4-ctl.sh gates      # 10 correctness gates
./ds4-ctl.sh bench      # code-at-temp-0 decode benchmark
```

The compose file is deliberately explicit rather than clever: every flag that
matters is visible, and the non-negotiable host settings carry comments
explaining why they cannot be removed.

---

## Non-negotiable settings on a P2P-disabled host

```yaml
NCCL_P2P_DISABLE: "1"
VLLM_ENABLE_PCIE_ALLREDUCE: "0"
```
```
--disable-custom-all-reduce
```

`--disable-custom-all-reduce` is **separate from and not implied by**
`NCCL_P2P_DISABLE`. vLLM's CustomAllreduce kernel uses CUDA P2P directly and will
deadlock regardless of the environment variable. Both are required.

---

## Correctness gates

Speed is worthless if tool calling is broken. `gates.py` checks ten things and
exits non-zero on any failure:

1. HTTP 401 without an API key
2. Model identity and advertised context from `/v1/models`
3. Exact single-word output at temperature 0
4. Reasoning separated from content
5. Strict JSON via `response_format`
6. Automatic tool call
7. Forced tool call (`tool_choice`)
8. Tool-result roundtrip (the model must use the returned value)
9. SSE streaming
10. In-flight cancellation followed by clean recovery

All ten pass on the configuration in this repository.

---

## Benchmarking notes

**Never count SSE chunks.** With speculative decoding a single SSE chunk can
carry several tokens, so chunk counting measures verification steps, not
generated tokens. Use streamed `usage` (`stream_options: {include_usage: true}`)
and read `completion_tokens`.

**Disable prefix caching when benchmarking.** A fixed-prompt harness with warmup
will hit cache on later iterations and inflate prefill and TTFT into numbers that
mean nothing. Enable it in production if you want; disable it to measure.

**Report the workload.** See the acceptance-rate caveat above. A prose number and
a code number for the same server can differ by nearly 2×.

---

## Known-good and known-bad combinations

| Setting | Verdict |
|---|---|
| `--kv-cache-dtype fp8_ds_mla` | required |
| `--kv-cache-dtype nvfp4_ds_mla` | asserts on DeepseekV4 |
| `--kv-cache-dtype nvfp4` | unsupported on SM120 (vLLM #43562) |
| `decode-context-parallel-size > 1` | rejected by DSpark |
| `num_speculative_tokens` < 5 | rejected — DSpark block size is 5 |
| `VLLM_USE_B12X_FP8_GEMM=1` | breaks DSpark drafter warmup |
| `--moe-backend b12x` | works, recommended |
| `--attention-backend B12X_MLA_SPARSE` | broken for DSV4 on SparkInfer 0.30.0 |
| `--disable-custom-all-reduce` | required on PCIe-only hosts |

---

## Useful references

- SparkInfer / B12X kernel library (SM120/SM121): `local-inference-lab/sparkinfer`
- Community RTX PRO 6000 wiki: `local-inference-lab/rtx6kpro`
- Two-node DGX Spark DeepSeek recipe: `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`
- DSpark concurrency patch: `drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash`
- LocalMaxxing leaderboard and CLI: `localmaxxing.com`, `LottoLottoLotto/localmaxxing-cli`

## A bug worth knowing about in the LocalMaxxing CLI

`lmx` v0.1.30 posts speed tests to `/api/benchmarks`. That path returns **HTTP 200
with the site's HTML**, so the CLI sees a 2xx and prints
`benchmark_submitted / status: submitted` while nothing is recorded.

The live endpoint is `/api/speed-tests`:

```bash
# validate (does not consume rate limit)
curl -sS -X POST https://www.localmaxxing.com/api/speed-tests/dry-run \
  -H "Authorization: Bearer $LMX_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @lmx-run.json

# submit
curl -sS -X POST https://www.localmaxxing.com/api/speed-tests \
  -H "Authorization: Bearer $LMX_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @lmx-run.json
```

A successful submission returns HTTP 201 with a record id and `"status": "APPROVED"`.
Verify independently — the model page is cached, and results are bucketed by the
`quantization` string you submit.

## License

MIT

---

## A P2P gotcha the launcher does not handle for you

`ALLREDUCE_MODE=b12x` sets `VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=0`, but it does
**not** set `NCCL_P2P_DISABLE`. On a host whose topology deadlocks at NCCL's
first peer-to-peer collective, that is not enough on its own.

The failure looks like this: the container never becomes healthy, and

```
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
0, 1044 MiB, 100 %
1, 1044 MiB, 100 %
2, 1044 MiB, 100 %
3, 1052 MiB, 100 %
```

All GPUs pinned at 100% with ~1 GB allocated, logs stopped right after
`world_size=N ... backend=nccl`. Set `NCCL_P2P_DISABLE=1` yourself.
A healthy load instead shows weights arriving quickly — with InstantTensor,
about 41 GiB per rank in ~37 seconds.
