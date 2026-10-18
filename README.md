# nano-inference

**Learn LLM inference systems by building a small serving platform.**

A Python gateway for exploring request routing, streaming, bounded admission,
failure handling, and performance measurement around independent inference replicas.
The platform runs on a Mac or Linux CPU with two deterministic simulated workers.
Prefix affinity and a real-model llama.cpp CPU compatibility check are implemented.
vLLM integration and real GPU measurements are upcoming milestones.

## Run locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --extra dev
uv run nano-inference local
```

This starts a gateway on `127.0.0.1:8000` and simulated workers on ports `8101`
and `8102`. Keep it running and use another terminal for requests. Stop with Ctrl-C;
the launcher shuts down its children. All three ports must be available.
Use `--port 8001 --worker-ports 8111 8112` to run a separate local cluster.

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nano-model","messages":[{"role":"user","content":"Explain a queue."}],"stream":true,"max_tokens":8}'

curl http://127.0.0.1:8000/workers
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/metrics
```

`nano-model` is an API alias for synthetic output, not a downloaded language model.
No model weights or GPU are needed. Simulator delays and usage counts are synthetic;
they do not predict GPU speed or real tokenizer behavior.

## Components

```text
Benchmark / client
        |
  Gateway :8000
  validation -> bounded FIFO admission -> atomic route + reserve
        |                                      |
  simulator A :8101                       simulator B :8102
        \__________ streaming HTTP ____________/
```

| Module | Responsibility |
|---|---|
| `protocol.py` | Explicit chat request subset and SSE framing/parser |
| `mock_worker.py` | Deterministic output, delays, HTTP failures, mid-stream errors |
| `routing.py` | Worker registry, round robin / least in-flight, capacity reservations, FIFO queue |
| `prefix.py` | Token-prefix identity, bounded tokenizer work, expiring LRU affinity hints |
| `gateway.py` | Forwarding, cancellation, deadlines, health probes, metrics, request logs |
| `benchmark.py` | Closed-loop / open-loop workload generation and raw JSON results |
| `cli.py` | Individual services and local process launcher |

The gateway is **one process with one event loop**. Worker choice and in-flight
reservation happen together under a lock. Multiple gateway processes would each
have independent counters and queues; shared coordination is outside this milestone.

`least_inflight` selects the healthy replica with the fewest assigned, unfinished
requests, using rotating ties. `round_robin` rotates through healthy replicas with
available capacity. Neither policy estimates prompt cost or GPU utilization yet.

## Prefix-affinity routing

```bash
uv run nano-inference local --policy prefix_affinity --prefix-tokens 64 \
  --affinity-capacity 1024 --affinity-ttl 60 --affinity-max-skew 1

uv run nano-inference benchmark --requests 24 --concurrency 1 \
  --workload repeated_prefix --prefix-groups 4 --prefix-repetitions 32
```

The router hashes a fixed prefix of the **fully templated token sequence**, together
with model revision, tokenizer fingerprint, template fingerprint and prefix length.
Short prompts fall back to least in-flight. The simulator uses explicitly labeled
synthetic byte IDs; real models require a prepared fast Hugging Face tokenizer and
an explicit model revision. The real CPU check below verifies token IDs against the
serving engine for the tested model/template pair.

After a request completes successfully, the selected worker becomes the hint's
home. Later requests prefer that home if healthy, below capacity, and no more than
`affinity-max-skew` assigned requests busier than the least-loaded eligible worker
**before assignment**. Otherwise they use the less-loaded worker immediately. FIFO
admission still applies; a request never waits solely for its preferred worker.

Hints are bounded by LRU capacity, expire after their last successful home completion,
and are cleared on observed worker health transitions. Reads and fallback completions
do not extend a home's lifetime. Completions from an older worker generation cannot
recreate a hint after observed failure/recovery. A restart between health probes can
go unobserved; TTL remains a fallback, and hints never guarantee cache residency.

`X-Route-Decision`, request logs, `/workers`, and `nano_route_decisions_total` report
`affinity_miss`, `affinity_hit`, `affinity_load_override`, or `prefix_too_short`.
**An affinity hit is a router decision, not a measured GPU cache hit.**

Tokenization runs in a bounded admission stage (default four concurrent preparations),
before the bounded generation queue. Excess preparation work gets HTTP 429
`prefix_preparation_full`; callers exceeding 200,000 aggregate message characters get
413 for this policy. Cancellation retains a preparation permit until the actual
tokenizer thread finishes. The overall request deadline includes this stage. Real
tokenizer loading is local-only and uses no remote Python code.

## Real-model CPU check on Apple Silicon

```bash
uv sync --locked --extra dev --extra tokenizer
uv run python scripts/prepare_llama.py
uv run --extra tokenizer python scripts/smoke_llama.py
```

Preparation downloads about 2.6 GB into ignored `.local/`: a pinned official
llama.cpp Mac ARM runtime, a community Q4_K_M conversion of
`Qwen3-4B-Instruct-2507`, and the original model's tokenizer. Downloads use immutable
revisions and SHA256 checks. No system package installation is needed. Model files
stay local and are not committed to Git.

The smoke script launches one CPU-only llama.cpp replica and a prefix-affinity
gateway on temporary local ports. It checks exact template/tokenization agreement,
ordinary and streaming completions, usage, affinity reuse, generation cancellation,
backend-slot cleanup, and a successful request after cancellation. Both child
services are stopped in `finally`. The JSON evidence is saved to
`results/examples/real-cpu-smoke.json`; service logs stay in `.local/`.

CPU execution uses `--device none --n-gpu-layers 0 --no-op-offload`. This verifies
real-model compatibility on the tested Mac, not vLLM GPU behavior or routing speedups.
The runtime preparer targets macOS arm64; other platforms can supply their own
compatible llama-server. See [the CPU integration guide](docs/local-model.md) for
pinned artifacts and commands to keep a real-model gateway running.

## Overload and failure behavior

```bash
uv run nano-inference local --policy round_robin --capacity 2 \
  --max-queue 8 --queue-timeout 1 --request-timeout 10
```

- `capacity` limits assigned requests per worker, including upstream connection time.
- Requests wait in a bounded FIFO queue when every healthy worker is full.
- Queue full or queue deadline exceeded: HTTP 429 with a reason and `Retry-After`.
- No healthy workers: HTTP 503. `/health` stays live while `/ready` returns 503.
- Upstream connection/HTTP failure before streaming: HTTP 502; timeout: HTTP 504.
  Upstream HTTP 429 is preserved as 429.
- After streaming headers have been sent, an upstream failure emits an SSE `error`
  event and ends **without `[DONE]`**. HTTP status is already 200; clients must inspect
  the stream. Partial generations are never replayed automatically.
- Client disconnect, timeout, and completion close the upstream response and release
  the reservation. Queued clients that disconnect are removed from the queue.
- The overall request deadline includes queueing, connection time, and generation.
  Periodic `/health` probes remove/re-admit workers; probes are hints, not guarantees.

`/workers` exposes current admission and health state. `/metrics` exports gateway
request outcomes, admission rejections, durations, first-content timing, and gauges.
JSON request logs contain request/worker IDs, policy, outcome, queue delay, duration,
first-content timing, and backend usage when available. Prompts are not logged.

## Inject a failure

The simulators expose a local administration endpoint. `PUT` replaces the complete
configuration; omitted fields return to defaults. Changes apply to new requests.

```bash
# Remove worker B from the healthy pool on the next health probe.
curl -X PUT http://127.0.0.1:8102/admin/config \
  -H 'Content-Type: application/json' -d '{"healthy":false}'

# Restore default healthy behavior.
curl -X PUT http://127.0.0.1:8102/admin/config \
  -H 'Content-Type: application/json' -d '{}'

# Worker A emits two content chunks, then an error event without DONE.
curl -X PUT http://127.0.0.1:8101/admin/config \
  -H 'Content-Type: application/json' -d '{"fail_after_chunks":2}'
```

Other settings: `first_content_delay`, `chunk_delay`, and `http_status`.
For an actual process crash, run services separately and stop one worker:

```bash
uv run nano-inference worker --name mock-a --port 8101
uv run nano-inference worker --name mock-b --port 8102
uv run nano-inference gateway --worker-urls http://127.0.0.1:8101 http://127.0.0.1:8102
```

Each command runs in its own terminal. The `local` launcher intentionally stops
the whole group if a child exits; use separate commands for process-crash exercises.

## Measure behavior

```bash
# Closed loop: maintain at most four concurrent requests.
uv run nano-inference benchmark --requests 100 --concurrency 4 \
  --output results/closed.json

# Open loop: schedule arrivals independently of response completion.
uv run nano-inference benchmark --requests 200 --rate 50 \
  --max-outstanding 128 --output results/open.json
```

Reports include every request outcome, worker, HTTP status, dispatch lag,
arrival-to-first-content latency, completion latency, content-event count, and
backend completion-token usage. Summary output reports success throughput, p50/p95
for successful requests, and successful requests meeting the first-content SLO per
second. Compare these alongside rejection, failure, and client-drop counts.

An SSE content event can contain multiple tokens. Output tokens/s uses backend
usage, never chunk count; it is null if any successful request lacks token usage.
First-content timing ignores empty role deltas. Client-observed first content is
not an exact measurement of engine TTFT or per-token decode latency.

Open-loop work is bounded at the client: arrivals exceeding `max-outstanding` are
recorded as `client_dropped`, distinct from server rejection. Dispatch lag and
arrival-based timing expose client scheduling delays. Throughput uses the full run
including drain time. Workloads include `short`, `mixed` (three prompt/output limits),
`repeated_prefix` (multiple system-prefix groups), and `single_prefix` (one-prefix
control). `prefix-repetitions` controls text repetition, not an exact token length;
backend prompt-token usage records the actual count. Every result also retains its
routing decision. The runner measures one workload run; it does not automatically
perform policy sweeps, warmups, cache resets, or significance analysis.

## API scope

`POST /v1/chat/completions` accepts text messages (`system`, `user`, `assistant`),
`model`, `stream`, `max_tokens` (1–4096), `temperature`, `top_p`, `seed`, and
`stream_options.include_usage`. Unsupported fields are rejected with 422.
Streaming and non-streaming completions are supported; tool calls, multimodal
inputs, authentication, and the full OpenAI API surface are outside this version.

The CLI binds to localhost. This is an educational local service; the simulator's
administration API is intended only for local experiments.

## Test

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev --extra tokenizer pytest -q
```

Tests use real TCP servers for incremental delivery, cancellation before response
headers and during streaming, queueing, failures, health recovery, and workload
runs. Unit tests cover atomic routing, FIFO ordering, cancellation, stale health
probes, SSE framing, and reporting without token usage. GitHub Actions runs these
checks on Python 3.12 and 3.13.

See [the roadmap](docs/roadmap.md) for the Mac-to-GPU progression and experiment plan.
