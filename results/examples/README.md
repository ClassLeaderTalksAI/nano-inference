# Local simulator verification

These reference batches predate the native vLLM Metal flavor. Current Apple GPU
verification is recorded separately in [results/macos-metal](../macos-metal/README.md).

These are functional workload checks on macOS arm64 with Python 3.13.6. They use
deterministic CPU simulators, not model inference. Token counts and delays are
synthetic; none of these rates are GPU performance claims or policy comparisons.

Start the services:

```bash
uv run nano-inference local --capacity 2 --max-queue 4
```

Run the workloads sequentially, with default simulator behavior:

```bash
uv run nano-inference benchmark --requests 40 --concurrency 4 --max-tokens 16 \
  --output results/examples/local-closed.json

uv run nano-inference benchmark --requests 100 --rate 200 --max-tokens 64 \
  --output results/examples/local-overload.json
```

| Run | Arrivals | Successful | HTTP 429 rejected | Other failures / client drops |
|---|---:|---:|---:|---:|
| Closed loop | 40 | 40 | 0 | 0 |
| Open-loop overload | 100 | 12 | 88 | 0 |

Both reports retain per-request measurements, client settings, runtime details,
and gateway snapshots before/after the run. Both finished with zero queued and
in-flight requests. Exact outcomes and timings can change between machines and
runs; the acceptance criteria are bounded admission, explicit outcomes, valid stream
completion, and cleanup rather than a particular throughput or rejection count.

The first-batch test suite passed 33 tests on the same local environment,
including live TCP streaming, cancellations, FIFO admission, errors, worker health,
and an abrupt CLI worker-process kill followed by restart. Formatting and lint
checks passed. The local launcher also shut down all three child services on exit.

## Prefix routing and real CPU model: second batch

The second batch adds these independent checks:

| Report | Workload | Result |
|---|---|---|
| `local-prefix-affinity.json` | 24 sequential requests across four synthetic prefix groups | 24 successes, four affinity misses and 20 affinity decisions |
| `local-prefix-load-override.json` | 16 concurrent requests for a previously warmed group | 16 successes, eight affinity decisions and eight load overrides |
| `real-cpu-smoke.json` | One pinned real Qwen3-4B-Instruct-2507 Q4 CPU replica | Exact token-ID agreement, ordinary/streaming responses, real usage, cancellation and subsequent request verified |

Generate the two simulator reports sequentially, starting a fresh service group:

```bash
uv run nano-inference local --policy prefix_affinity --capacity 2 --max-queue 4 \
  --prefix-preparations 16
```

In another terminal, run the first workload and immediately follow it with the
second so the 60-second affinity hints are still warm:

```bash
uv run nano-inference benchmark --requests 24 --concurrency 1 --max-tokens 8 \
  --workload repeated_prefix --prefix-groups 4 --prefix-repetitions 32 \
  --output results/examples/local-prefix-affinity.json

uv run nano-inference benchmark --requests 16 --concurrency 8 --max-tokens 16 \
  --workload repeated_prefix --prefix-groups 1 --prefix-repetitions 32 \
  --output results/examples/local-prefix-load-override.json
```

Both reports ended with zero tokenization preparations, queued requests and in-flight
reservations. Synthetic affinity decisions describe routing, not real KV cache hits.
Counts depend on timing and workload; these runs test behavior, not performance gains.

Concurrent testing exposed a cancellation race in the original header-handoff path:
a polling disconnect check could swallow cancellation of the watcher, leaving the
response waiting for that watcher to end. The gateway now waits directly for the
ASGI disconnect event after parsing the body. A 128-request concurrent regression
test covers timely handoff and cleanup along with the earlier cancellation tests.
The launcher also handles Ctrl-C delivered to the whole process group, reaping
children and exiting successfully even if a child exits first.
The second-batch suite passed 51 tests on macOS arm64 with Python 3.13.6 and the
tokenizer extra installed; formatting and lint checks also passed.

For the CPU check, follow [the local-model guide](../../docs/local-model.md). The
report pins the official runtime, community GGUF and original tokenizer separately,
and records CPU-only flags. It verifies a single replica's interface; two-replica
load balancing remains covered by simulator tests. vLLM GPU compatibility and GPU
performance remain unverified. The GitHub Actions workflow separately targets
Python 3.12 and 3.13 on Linux with the optional tokenizer dependencies installed;
its remote status must be checked after pushing.
