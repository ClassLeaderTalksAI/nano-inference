# Native Mac flavor verification

These reports come from native vLLM Metal execution on an Apple M1 Pro Mac with
32 GB unified memory and macOS 26.7.1. Two model replicas share this physical GPU.
The profile, pinned model revision and full model-file hashes are recorded in
`verification.json`. Runtime dependency versions and a direct Metal computation
are recorded under `environment`; that environment snapshot was collected after
the workload validation and is not a peak-memory measurement.

| Check | Recorded result |
|---|---|
| Tokenization | Long-prefix and multilingual/multi-turn samples match each engine exactly |
| Native cache counters | 420 queried tokens, 192 hit tokens over the cold/warm request pair |
| Prefix workload | 12 successful requests |
| Mixed workload | 9 successful requests |
| Open-loop overload | 4 successful, 8 explicitly rejected |
| Cancellation | Running state observed, then gateway/engines idle without normal-finish increment |
| Mid-stream worker kill | Error without DONE; survivor serves new requests |
| Restart | Both replicas serve requests again |
| Cleanup | Owned services stopped and ports released |
| Offline tests | 68 passed; lint and formatting checks passed |

`prefix.json`, `mixed.json` and `overload.json` retain per-request results, errors,
routing decisions, offered load, runtime labels and gateway snapshots. The compact
compatibility/failure report is `verification.json`. Reproduce with:

```bash
uv sync --locked --extra dev --extra tokenizer
uv run python scripts/prepare_metal.py
uv run --extra tokenizer python scripts/smoke_vllm.py
```

The cache ratio includes a cold request and a warm request; it is neither a
warm-only hit ratio nor a latency improvement. Native TTFT/TPOT histograms and
cache counters describe the engine; client first-content timing includes gateway
and network effects. Upstream's `request_success_total` exposes finish reasons
including abort/error, so its unfiltered total must not be treated as successful
user-request throughput.

The HTTP-cancellation path can clear an engine request without incrementing its
abort finish-reason counter. The check records those counters but proves cleanup
through a sampled running-to-idle transition, gateway state and unchanged stop/length
counters. Counter deltas are not computed across a worker restart.

These short runs establish functional behavior for this pinned pair. They do not
establish stable tail-latency statistics, a policy performance ranking, independent
GPU scaling, or CUDA/RunPod compatibility. See the [Chinese review](../../docs/mac-flavor-review.zh-CN.md).
