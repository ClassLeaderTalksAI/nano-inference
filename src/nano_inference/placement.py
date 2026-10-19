"""Explainable TTFT placement, using approximate cache hints and bounded telemetry.

This is a configurable analytical model, not a learned latency predictor or a KV
block directory. All mutation runs under WorkerPool.condition (no network waits).
"""

import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ServiceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    base_seconds: float = Field(default=0.02, ge=0, le=60)
    prefill_tokens_per_second: float = Field(default=1000, gt=0, le=1e9)
    decode_tokens_per_second: float = Field(default=40, gt=0, le=1e9)
    parallelism: float = Field(default=1, ge=1, le=1024)


class PlacementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    service: ServiceModel = Field(default_factory=ServiceModel)
    workers: dict[str, ServiceModel] = Field(default_factory=dict)
    cache_capacity: int = Field(default=4096, ge=1, le=100_000)
    cache_ttl_seconds: float = Field(default=30, gt=0, le=3600)
    cache_reuse_discount: float = Field(default=0.8, ge=0, le=1)
    metrics_max_age_seconds: float = Field(default=5, gt=0, le=300)
    external_request_seconds: float = Field(default=0.5, gt=0, le=3600)
    kv_soft_limit: float = Field(default=0.8, ge=0, lt=1)
    kv_penalty_seconds: float = Field(default=0.5, ge=0, le=60)
    default_slo_seconds: float = Field(default=2, gt=0, le=3600)
    slo_admission: bool = False
    slo_safety_factor: float = Field(default=1.2, ge=1, le=10)

    @classmethod
    def load(cls, path):
        return cls.model_validate_json(Path(path).read_text()) if path else cls()


@dataclass(frozen=True)
class RouteRequest:
    input_tokens: int = 0
    output_tokens: int = 32
    # Hashes of complete token prefixes, never raw prompts; tenant-scoped upstream.
    prefixes: tuple[tuple[str, int], ...] = ()
    deadline: float | None = None  # Monotonic first-content deadline from API arrival.


@dataclass(frozen=True)
class Prediction:
    queue_seconds: float
    prefill_seconds: float
    kv_seconds: float
    base_seconds: float
    cached_tokens: int
    native_fresh: bool
    kv_cache_fraction: float | None
    remaining_budget_seconds: float | None
    ttft_seconds: float
    slo_feasible: bool | None


class Placement:
    def __init__(self, config: PlacementConfig, clock=time.monotonic):
        self.config, self.clock = config, clock
        # A prefix may live on several replicas. Entries expire from successful
        # completions, not reads; engine evictions are NOT observed by this index.
        self.cache: OrderedDict[tuple[str, str], tuple[int, float]] = OrderedDict()
        self.native: dict[str, tuple[int, float, dict, int]] = {}
        self.active: dict[int, tuple[str, int, float]] = {}
        self.last_scores: dict[str, Prediction] = {}
        self.last_choice: str | None = None

    def prune(self):
        for key, (_, expiry) in list(self.cache.items()):
            if expiry <= self.clock():
                del self.cache[key]

    def invalidate(self, worker):
        self.native.pop(worker, None)
        for key in [k for k in self.cache if k[1] == worker]:
            del self.cache[key]
        # Keep old reservations until their own release, but exclude generations
        # from predictions after recovery. Old completions cannot clear new work.

    def observe(self, worker, generation, values):
        if values is None:
            self.native.pop(worker, None)
            return
        clean = {}
        for name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
            value = values.get("vllm:" + name)
            if value is not None and math.isfinite(value) and value >= 0:
                if name != "kv_cache_usage_perc" or value <= 1:
                    clean[name] = value
        local_count = sum(name == worker and g == generation for name, g, _ in self.active.values())
        self.native[worker] = generation, self.clock(), clean, local_count

    def predict(self, worker, request: RouteRequest) -> Prediction:
        cfg, now = self.config, self.clock()
        service = cfg.workers.get(worker.name, cfg.service)
        local = [
            seconds
            for name, generation, seconds in self.active.values()
            if name == worker.name and generation == worker.generation
        ]
        sample = self.native.get(worker.name)
        fresh = bool(
            sample
            and sample[2]
            and sample[0] == worker.generation
            and 0 <= now - sample[1] <= cfg.metrics_max_age_seconds
        )
        native = sample[2] if fresh else {}
        # Native counts include our leases. Add only excess work, not both totals.
        native_count = native.get("num_requests_running", 0) + native.get("num_requests_waiting", 0)
        # Anchor the overlap to the scrape, not the current lease count. Otherwise
        # a completed local request becomes phantom external work until next poll.
        extra = max(0, native_count - sample[3]) if fresh else 0
        queue = (sum(local) + extra * cfg.external_request_seconds) / service.parallelism
        cached = 0
        for key, length in request.prefixes:
            hint = self.cache.get((key, worker.name))
            if hint and hint[0] == worker.generation and hint[1] > now:
                cached = max(cached, min(length, max(0, request.input_tokens - 1)))
                self.cache.move_to_end((key, worker.name))
        effective = max(1, request.input_tokens - cached * cfg.cache_reuse_discount)
        prefill = effective / service.prefill_tokens_per_second
        kv = native.get("kv_cache_usage_perc")
        penalty = (
            max(0, kv - cfg.kv_soft_limit) / (1 - cfg.kv_soft_limit) * cfg.kv_penalty_seconds
            if kv is not None
            else 0
        )
        ttft = service.base_seconds + queue + prefill + penalty
        budget = request.deadline - now if request.deadline is not None else None
        return Prediction(
            queue,
            prefill,
            penalty,
            service.base_seconds,
            cached,
            fresh,
            kv,
            budget,
            ttft,
            ttft * cfg.slo_safety_factor <= budget if budget is not None else None,
        )

    def select_worker(self, request: RouteRequest, workers):
        """Caller filters health, capacity and tenant allowlist before scoring."""
        self.prune()
        self.last_scores = {w.name: self.predict(w, request) for w in workers}
        # SLO feasibility is monotonic in TTFT. Choose the minimum, never a slower
        # feasible worker; admission separately decides what to do if none can fit.
        worker = min(workers, key=lambda w: self.last_scores[w.name].ttft_seconds)
        self.last_choice = worker.name
        return worker

    def reserve(self, lease, request):
        model = self.config.workers.get(lease.worker.name, self.config.service)
        p = lease.prediction
        seconds = p.base_seconds + p.prefill_seconds + p.kv_seconds
        seconds += request.output_tokens / model.decode_tokens_per_second
        self.active[id(lease)] = lease.worker.name, lease.generation, seconds

    def release(self, lease, successful):
        self.active.pop(id(lease), None)
        if not successful or not lease.route_request:
            return
        self.prune()
        for key, _ in lease.route_request.prefixes:
            cache_key = key, lease.worker.name
            self.cache[cache_key] = (lease.generation, self.clock() + self.config.cache_ttl_seconds)
            self.cache.move_to_end(cache_key)
        while len(self.cache) > self.config.cache_capacity:
            self.cache.popitem(last=False)

    def snapshot(self):
        self.prune()
        return {
            "model": "analytical-ttft-v1",
            "cache_tracking": "approximate-success-history",
            "config": self.config.model_dump(),
            "cache_entries": len(self.cache),
            "active_reservations": len(self.active),
            "last_choice": self.last_choice,
            "last_scores": {k: asdict(v) for k, v in self.last_scores.items()},
        }
