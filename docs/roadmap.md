# Learning roadmap

## Objective

Build a small inference serving system and use repeatable experiments to explain
load distribution, cache locality, overload, and failure behavior. Implement the
Python serving components; use vLLM for model execution, internal token scheduling,
continuous batching, and KV-cache allocation.

## Milestones and acceptance gates

| Milestone | Deliverable | Acceptance gate |
|---|---|---|
| 1 — Local protocol and workers | Two deterministic CPU simulators and explicit chat API subset | Normal output, injected delays, HTTP and streaming errors |
| 2 — Gateway and routing | Streaming/non-streaming forwarding, worker health, round robin and least in-flight | Incremental delivery; atomic reservations; no leaked slots on disconnect or failure |
| 3 — Admission and measurement | Bounded FIFO queue, queue/overall deadlines, workload runner, metrics | Visible rejection/failure rates; bounded client and server work; raw results preserved |
| 4 — Prefix affinity | Stable token-prefix identity, bounded/expiring affinity hints, load override | Health/capacity always respected; identity includes model revision and chat template |
| 5 — Local real-model smoke test | A small GGUF model served through llama.cpp | Verify supported request fields, SSE, usage, cancellation; no GPU-performance claims |
| 6 — One GPU | Pinned vLLM image, model/tokenizer revision and launch configuration | Real completion, cancellation, engine metrics and prefix-caching smoke test |
| 7 — Two GPUs | Two independent replicas, one full model per GPU | Reproducible routing, cache, overload and failure experiments |
| 8 — Analysis | Raw data, configuration, plots and documented limitations | Every performance claim traceable to measured runs |

Milestones 1–3 were delivered in the first implementation batch. Milestone 4 is
implemented and tested with two simulated replicas; milestone 5 is verified with a
pinned Qwen3-4B-Instruct-2507 Q4 model on the local Mac CPU. See
[the real-model integration guide](local-model.md) and the raw example reports.
Milestones 6–8 remain pending.
CLI and request fields may evolve as real backend integration reveals differences.

## Model and hardware starting point

- Baseline: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
  Its non-thinking output and standard attention structure simplify the first
  cache/routing experiments. BF16 weight storage is roughly 8 GB; this excludes
  KV cache, execution buffers, graph capture, and all other runtime allocations.
- Start with a 4,096-token model length, 128–256 output-token limits, and low
  concurrency. Measure actual memory consumption before raising limits.
- Final target: two identical NVIDIA GPUs with approximately 24 GB each. Run one
  full-model vLLM replica per GPU; no tensor parallelism or KV transfer initially.
- Later comparison: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B).
  Its hybrid attention/state architecture adds a separate cache-management topic.
- A local quantized GGUF smoke test validates interfaces only. Its timing and
  cache behavior do not establish vLLM GPU performance.

Pin actual versions and model revisions at integration time. No GPU capacity or
backend compatibility has been validated by the simulator milestone.

## Experiment matrix

| Question | Workload / control | Measurements |
|---|---|---|
| When does least in-flight help? | Round robin vs least in-flight; mixed prompt/output lengths; identical replicas | Client first content, engine TTFT/TPOT, success throughput, per-replica distribution |
| When does affinity help or hurt? | Multiple exact token-prefix groups, skew and cache pressure; one-shared-prefix control; APC on/off | Prefix-cache metrics, prefill time, first content, load imbalance, throughput |
| Does admission protect useful work? | Open-loop arrival-rate sweep, bounded queue vs configured alternatives | Successful latency and SLO goodput alongside rejections, failures and client drops |
| How does a failed replica affect service? | Stop/kill and restart a worker under load | Detection/re-admission time, interrupted streams, new-request success, counter cleanup |

One shared prefix alone is an insufficient affinity benchmark: round robin can
warm that prefix on both replicas. An affinity mapping is only a hint, not evidence
that GPU cache blocks still exist. Prefix reuse primarily helps prefill; long decode
workloads may see little benefit.

Record cold/warm-cache conditions, prompt token lengths after chat templating,
sampling/output settings, offered rate, client dispatch lag, run duration and sample
count. Repeat runs and vary policy order. Report unsuccessful requests alongside
successful latency; avoid stable P99 claims from small samples. Keep the load
generator and gateway near the GPU workers for cloud experiments and separate
network latency from engine timings.

## Boundaries

The first version is a single gateway process, with static worker URLs and local
state. Prefix hints, mixed/prefix workload generation and a pinned local CPU model
check are implemented. Token-aware cost models, Grafana, vLLM GPU deployment scripts
and measured GPU performance improvements remain pending. Distributed KV
storage, prefill/decode disaggregation, autoscaling, Kubernetes, and custom GPU
kernels are follow-up projects rather than first-version requirements.

Planning basis: [vLLM CPU support](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/),
[prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/),
[metrics](https://docs.vllm.ai/en/latest/design/metrics/), and the model cards above.
