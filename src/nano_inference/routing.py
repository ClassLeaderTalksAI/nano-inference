"""Atomic routing reservations, bounded FIFO admission, and health state.

One process owns all counters. Selection and reservation happen under the same
condition lock, before any network await, so concurrent requests see reservations.
"""

import asyncio
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal

from .placement import Placement, PlacementConfig, Prediction, RouteRequest
from .prefix import AffinityTable

Policy = Literal[
    "round_robin", "least_inflight", "prefix_affinity", "predicted_ttft", "cache_aware"
]


class AdmissionError(Exception):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code
        super().__init__(code)


@dataclass
class Worker:
    name: str
    url: str
    healthy: bool = False
    inflight: int = 0
    last_health_check: float | None = None
    generation: int = 0


@dataclass
class Lease:
    worker: Worker
    generation: int
    queue_seconds: float
    released: bool = False
    prefix_key: str | None = None
    decision: str = "least_inflight"
    tenant_id: str | None = None
    estimated_cost: float = 0
    reserved_tokens: int = 0
    reserved_cost: float = 0
    route_request: RouteRequest | None = None
    prediction: Prediction | None = None


class WorkerPool:
    def __init__(
        self,
        urls: list[str],
        policy: Policy = "least_inflight",
        capacity: int = 4,
        max_queue: int = 16,
        queue_timeout: float = 2,
        affinity_capacity: int = 1024,
        affinity_ttl: float = 60,
        affinity_max_skew: int = 1,
        *,
        placement: PlacementConfig | None = None,
    ):
        urls = [url.rstrip("/") for url in urls]
        if not urls or len(set(urls)) != len(urls):
            raise ValueError("Provide at least one distinct worker URL")
        if capacity < 1 or max_queue < 0 or not math.isfinite(queue_timeout) or queue_timeout <= 0:
            raise ValueError("Invalid admission limits")
        if policy not in (
            "round_robin",
            "least_inflight",
            "prefix_affinity",
            "predicted_ttft",
            "cache_aware",
        ):
            raise ValueError("Unknown routing policy")
        if affinity_max_skew < 0:
            raise ValueError("Affinity load skew must be nonnegative")
        self.workers = [Worker(f"worker-{i + 1}", url.rstrip("/")) for i, url in enumerate(urls)]
        self.policy = policy
        self.capacity = capacity
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.condition = asyncio.Condition()
        self.waiters: deque[object] = deque()
        self.cursor = 0
        self.affinity = AffinityTable(affinity_capacity, affinity_ttl)
        self.affinity_max_skew = affinity_max_skew
        self.decisions: Counter[str] = Counter()
        self.placement = Placement(placement or PlacementConfig())
        if set(self.placement.config.workers) - {w.name for w in self.workers}:
            raise ValueError("Unknown worker in placement model")

    def _select(
        self,
        prefix_key: str | None,
        allowed: list[str] | None = None,
        route_request: RouteRequest | None = None,
    ) -> tuple[Worker, str] | None:
        ordered = self.workers[self.cursor :] + self.workers[: self.cursor]
        candidates = [
            w
            for w in ordered
            if w.healthy and w.inflight < self.capacity and (not allowed or w.name in allowed)
        ]
        if not candidates:
            return None
        worker = (
            candidates[0]
            if self.policy == "round_robin"
            else min(candidates, key=lambda w: w.inflight)
        )
        decision = self.policy
        if self.policy == "predicted_ttft":
            worker = self.placement.select_worker(route_request or RouteRequest(), candidates)
            prediction = self.placement.last_scores[worker.name]
            decision = "predicted_ttft" if prediction.native_fresh else "predicted_ttft_local"
        elif self.policy == "cache_aware":
            worker = self.placement.select_cache_aware(route_request or RouteRequest(), candidates)
            decision = self.placement.last_reason
        if self.policy == "prefix_affinity":
            decision = "prefix_too_short" if prefix_key is None else "affinity_miss"
            hint = self.affinity.get(prefix_key)
            if hint is not None:
                home = next((w for w in self.workers if w.name == hint.worker), None)
                if home is None or not home.healthy or home.generation != hint.generation:
                    self.affinity.invalidate(hint.worker)
                elif (
                    home in candidates and home.inflight <= worker.inflight + self.affinity_max_skew
                ):
                    worker, decision = home, "affinity_hit"
                else:
                    decision = "affinity_load_override"
        self.cursor = (self.workers.index(worker) + 1) % len(self.workers)
        return worker, decision

    def _reserve(self, selection, prefix_key, waited, route_request=None):
        worker, decision = selection
        prediction = None
        if self.policy in ("predicted_ttft", "cache_aware"):
            route_request = route_request or RouteRequest()
            prediction = self.placement.last_scores[worker.name]
            if (
                self.policy == "predicted_ttft"
                and self.placement.config.slo_admission
                and prediction.slo_feasible is False
            ):
                self.decisions["slo_budget_exceeded"] += 1
                raise AdmissionError(429, "slo_budget_exceeded")
        worker.inflight += 1
        self.decisions[decision] += 1
        lease = Lease(
            worker,
            worker.generation,
            waited,
            prefix_key=prefix_key,
            decision=decision,
            route_request=route_request,
            prediction=prediction,
        )
        if prediction:
            self.placement.reserve(lease, route_request)
        return lease

    async def acquire(
        self, prefix_key: str | None = None, *, route_request: RouteRequest | None = None
    ) -> Lease:
        started = time.perf_counter()
        ticket = object()
        async with self.condition:
            if not any(w.healthy for w in self.workers):
                raise AdmissionError(503, "no_healthy_workers")
            selected = (
                self._select(prefix_key, route_request=route_request) if not self.waiters else None
            )
            if selected is not None:
                return self._reserve(selected, prefix_key, 0, route_request)
            if len(self.waiters) >= self.max_queue:
                raise AdmissionError(429, "queue_full")
            self.waiters.append(ticket)
            try:
                async with asyncio.timeout(self.wait_budget(route_request)):
                    while True:
                        if not any(w.healthy for w in self.workers):
                            raise AdmissionError(503, "no_healthy_workers")
                        if self.waiters[0] is ticket:
                            selected = self._select(prefix_key, route_request=route_request)
                            if selected is not None:
                                return self._reserve(
                                    selected,
                                    prefix_key,
                                    time.perf_counter() - started,
                                    route_request,
                                )
                        await self.condition.wait()
            except TimeoutError as exc:
                raise AdmissionError(429, self.wait_timeout_code(route_request)) from exc
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def wait_budget(self, request):
        if self.policy == "predicted_ttft" and self.placement.config.slo_admission:
            if request and request.deadline is not None:
                return max(0, min(self.queue_timeout, request.deadline - self.placement.clock()))
        return self.queue_timeout

    def wait_timeout_code(self, request):
        if request and request.deadline is not None and self.wait_budget(request) <= 0:
            return "slo_budget_exceeded"
        return "queue_timeout"

    async def observe_native(self, worker, generation, values):
        async with self.condition:
            if worker.healthy and worker.generation == generation:
                self.placement.observe(worker.name, generation, values)

    async def release(self, lease: Lease, *, successful: bool = False) -> None:
        async with self.condition:
            self._release(lease, successful)
            self.condition.notify_all()

    def _release(self, lease, successful):
        if not lease.released:
            if lease.prediction:
                self.placement.release(
                    lease,
                    successful
                    and lease.worker.healthy
                    and lease.worker.generation == lease.generation,
                )
            if (
                successful
                and self.policy == "prefix_affinity"
                and lease.prefix_key
                and lease.worker.healthy
                and lease.worker.generation == lease.generation
            ):
                self.affinity.remember(lease.prefix_key, lease.worker.name, lease.generation)
            lease.worker.inflight -= 1
            lease.released = True

    async def set_health(
        self, worker: Worker, healthy: bool, expected_generation: int | None = None
    ) -> None:
        async with self.condition:
            # A probe started before a request failure must not undo that failure.
            if expected_generation is not None and worker.generation != expected_generation:
                return
            if worker.healthy != healthy:
                worker.generation += 1
                self.affinity.invalidate(worker.name)
                self.placement.invalidate(worker.name)
            worker.healthy = healthy
            worker.last_health_check = time.time()
            self.condition.notify_all()

    def snapshot(self) -> dict:
        self.affinity.prune()
        return {
            "policy": self.policy,
            "capacity_per_worker": self.capacity,
            "queued": len(self.waiters),
            "max_queue": self.max_queue,
            "queue_timeout_seconds": self.queue_timeout,
            "route_decisions": dict(self.decisions),
            "placement": self.placement.snapshot()
            if self.policy in ("predicted_ttft", "cache_aware")
            else None,
            "affinity": {
                "entries": len(self.affinity.entries),
                "capacity": self.affinity.capacity,
                "ttl_seconds": self.affinity.ttl,
                "max_load_skew": self.affinity_max_skew,
            },
            "workers": [
                {
                    "id": w.name,
                    "healthy": w.healthy,
                    "inflight": w.inflight,
                    "last_health_check": w.last_health_check,
                }
                for w in self.workers
            ],
        }
