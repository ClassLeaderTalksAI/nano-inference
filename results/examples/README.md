# Local simulator verification

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

The accompanying test suite passed 33 tests on the same local environment,
including live TCP streaming, cancellations, FIFO admission, errors, worker health,
and an abrupt CLI worker-process kill followed by restart. Formatting and lint
checks passed. The local launcher also shut down all three child services on exit.

Real-model CPU integration, prefix affinity, vLLM/GPU compatibility, and GPU
performance remain unverified. The GitHub Actions workflow separately targets
Python 3.12 and 3.13 on Linux; its remote status must be checked after pushing.
