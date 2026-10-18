# Native Metal tenant verification

Captured on the local M1 Pro / 32 GB Mac using the pinned `macos-metal` profile.
`verification.json` records the exact runtime, model revision, native cache-counter
deltas, operator snapshot and workload outcomes. `noisy-neighbor.json` preserves
all 18 arrivals and per-tenant results. Generated credentials and cache salts are
not included.

The cache experiment uses **one worker** for both tenants. The same 308-token
prompt is cold for interactive (0 cached tokens), warm for interactive (304), cold
for batch (0), and warm for batch (304). A fresh experiment marker is used on each
run, so prompt token lengths can vary slightly. This verifies the pinned engine's
cache-salt behavior, independently of replica placement.

The subsequent short burst had 12 successes and 6 rejections. It is a functional
admission/monitoring exercise, not a statistically meaningful FIFO/WDRR comparison
or a hard latency guarantee. Native metrics are sampled periodically and can lag
the gateway's immediate state; sample timestamps are retained. The one-worker
experiment does not measure two-GPU throughput. All owned processes were stopped
and their ports checked closed.

Reproduce with `uv run --extra tokenizer python scripts/smoke_tenants.py`.
See `docs/multi-tenant.zh-CN.md` for the two-worker interactive demo and limitations.
