# Tuning surface

The r16 launcher (`/usr/local/bin/serve-ds4-flash.sh`, inside the image) is
entirely environment-driven and exposes far more than the upstream example
compose shows. This page documents the knobs, what they do, and which ones are
worth measuring.

**Start here:**

```bash
docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -e DRY_RUN=1 -e TP_SIZE=4 -e BACKEND=b12x-a8 -e ALLREDUCE_MODE=b12x \
  -v /path/to/checkpoint:/model:ro "$IMAGE"
```

`DRY_RUN=1` prints the exact `vllm serve` command the launcher will build,
without starting anything. Use it before every experiment.

---

## Kernel backend

`BACKEND` selects the MoE and linear kernel path.

| Value | Notes |
|---|---|
| `b12x-a8` | Release default. SparkInfer W4A8. |
| `b12x-a16` | W4A16 — BF16 activations, inline FP4 weight dequant, no activation-scale math. |
| `b12x-a8-dglin` | W4A8 with DeepGEMM linear. |
| `lucifer-default` | Alternative kernel family. |
| `lucifer-cutlass` | CUTLASS variant. |

A published community sweep on 4× RTX PRO 6000 (different checkpoint revision)
found **B12X-A16 fastest at concurrency 1** and **Lucifer-CUTLASS fastest at
concurrency 64**, with B12X winning prefill by ~6% and losing 21–27% at cc64.
There is no universal winner — pick by whether you optimise single-stream
latency or aggregate throughput.

`BACKEND=b12x` is an alias for `b12x-a16`.

---

## Collective path — the important one on PCIe-only hosts

`ALLREDUCE_MODE` sets four environment variables at once. This matters enormously
on a host where GPUs cannot do peer-to-peer.

| Value | Effect |
|---|---|
| `b12x` (default) | `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, oneshot max 64KB, and critically **`VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=0`** |
| `vllm-custom` | Allows the custom CUDA-P2P all-reduce. **Deadlocks on PCIe-only topologies.** |
| `vllm-custom-2stage` | Same, with the 2-stage algorithm. |
| `nccl` | Disables both PCIe and custom paths, adds `--disable-custom-all-reduce`. Conservative. |

The `b12x` mode is purpose-built for this situation: it uses SparkInfer's own
PCIe collective while explicitly forbidding the CUDA-P2P path that hangs. If you
have previously hard-disabled PCIe all-reduce out of caution, this is worth
re-measuring — `nccl` is safe but may be leaving performance on the table.

Related: `B12X_PCIE_DMA` (0/1) enables SparkInfer's compressed PCIe DMA. Every
compressed wire mode moves 132 bytes per 128 values instead of 256 for BF16, a
48.4% wire-byte reduction. Codecs include E4M3, INT8 and MXFP8 in ring,
all-gather-only and all-to-all variants. Per the library's own docs these help
**large prefill collectives** and usually do not affect small decode collectives.

---

## Speculative decoding

`num_speculative_tokens` **must be ≥ 5** — the fused DSpark draft head has block
size 5, and lower values are rejected during config validation before CUDA
initialises.

| Knob | Values | Notes |
|---|---|---|
| `DSPARK_TOKENS` | 5 (default), 7 | Matched TP2 measurements favour K5: 217.8 vs 192.1 tok/s sustained decode, and K5 did not regress the coding probe. K7 also showed more long low-acceptance runs. |
| `DSPARK_DEPTH_MODE` | `fixed` (default), `dynamic` | Dynamic uses confidence-controlled draft depth. Not the release default. |
| `DRAFT_SAMPLE_METHOD` | `probabilistic` (default), `greedy` | Worth testing at temperature 0. |
| `DSPARK_CONFIDENCE_THRESHOLD` / `_TEMPERATURE` | — | Only meaningful with `DSPARK_DEPTH_MODE=dynamic`. |
| `DSPARK_DYNAMIC_DRAFT_DEPTH` / `_WINDOW` | — | Dynamic depth control. |
| `DSPARK_FP8_DRAFT_HEAD` | 0/1 | FP8 draft head. |
| `DSPARK_DRAFT_ATTENTION_BACKEND` | `auto`, `B12X_MLA_SPARSE`, … | Draft-side attention. |
| `DSPARK_SPS_CURVE`, `DSPARK_SPS_OVERHEAD_MS` | — | Speculation scheduling model. |

### Why acceptance rate is the whole game

With DSpark the engine runs at a roughly fixed verification-step rate, so

```
tok/s ≈ (verification steps/s) × (1 + accepted draft tokens per step)
```

Community measurement on this hardware class puts acceptance near **73% on code
at temperature 0** and near **33% on prose** — nearly a 2× throughput difference
on the same server from workload alone. Always report the workload with the
number, and tune against the workload you actually run.

---

## Loading

| Knob | Values |
|---|---|
| `LOAD_FORMAT` | `instanttensor` (default), `safetensors` |
| `INSTANTTENSOR_BACKEND` | `BUFFERED` (default) — reuses checkpoint pages from the Linux page cache |

InstantTensor is the required default loader for this image line.

---

## Native CPU KV offload

`KV_OFFLOADING_SIZE` is total host capacity in GiB across all TP ranks; `0`
disables it. Positive decimal and non-power-of-two values are accepted. The
launcher expands it to `--kv-offloading-size <GiB> --kv-offloading-backend native`.

This is independent of LMCache. **Do not enable `LMCACHE_MODE` at the same time** —
they are separate cache implementations and must be qualified independently.

Reported release-gate figures: a 70k/80k/100k prefix sequence produced a 5.22 GB
GPU→CPU store, and a 70k replay produced a 635.5 MB CPU→GPU load with 69,888
external prefix-cache hits, with no material decode regression (222.9 vs 220.6
tok/s).

On a host with a large amount of system RAM this is the cheapest route to very
long effective context.

**Cleanup caveat:** normal teardown closes and unlinks the shared mmap. A forced
`SIGKILL` cannot, and can orphan `/dev/shm/vllm_offload_*.mmap`. After a crashed
container, confirm no vLLM process still references the file before deleting it.

---

## Context length

`131072` is the conservative release default, not a model limit. Community runs
report roughly **650k with a 4096 batched-token budget** and **up to 1M with a
2048 budget** — neither certified by the release gate. Raise `MAX_MODEL_LEN` only
together with an appropriate `MAX_NUM_BATCHED_TOKENS` and a real long-context
retrieval test, not just a successful boot.

---

## Other knobs the launcher reads

`BLOCK_SIZE`, `CUDAGRAPH_CAPTURE_SIZES`, `INDEXER_BACKEND` (`auto`/`b12x`/`native`),
`ENABLE_FLASHINFER_AUTOTUNE`, `B12X_MHC_MAX_TOKENS`, `B12X_MOE_FORCE_A16` /
`B12X_MOE_FORCE_A8`, `VLLM_USE_B12X_WO_PROJECTION`, `VLLM_USE_B12X_MHC`,
`VLLM_USE_B12X_SPARSE_INDEXER`, `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS`,
`VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`,
`VLLM_USE_AOT_COMPILE`, `VLLM_USE_MEGA_AOT_ARTIFACT`, `VLLM_USE_BREAKABLE_CUDAGRAPH`,
`VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_SYMM_MEM_PCIE_SAFE_BARRIER`,
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, and `EXTRA_VLLM_ARGS` as a general
escape hatch.

### `VLLM_USE_BREAKABLE_CUDAGRAPH`

On some architectures vLLM auto-enables this and logs:

```
Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1 for this architecture,
disabling VLLM_USE_AOT_COMPILE=1.
VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline.
Equivalent to -cc.mode=none.
```

That second line matters: it turns `torch.compile` off entirely. A community
measurement on GB10 found setting it to `0` worth **+28.6% on single-stream
decode**. Check your boot log for this line before assuming your compile pipeline
is active.

---

## Benchmarking discipline

1. **Disable prefix caching while measuring** (`EXTRA_VLLM_ARGS=--no-enable-prefix-caching`).
   A fixed-prompt harness with warmup will otherwise hit cache on later
   iterations and report meaningless prefill and TTFT.
2. **Never count SSE chunks.** One chunk can carry several tokens under
   speculative decoding. Use `stream_options: {include_usage: true}` and read
   `completion_tokens`.
3. **Report the workload and the acceptance rate**, not just tok/s.
4. **Warm up** — the first request pays JIT and autotune costs.
