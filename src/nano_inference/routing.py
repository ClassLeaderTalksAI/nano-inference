"""Atomic routing reservations, bounded FIFO admission, and health state.

One process owns all counters. Selection and reservation happen under the same
condition lock, before any network await, so concurrent requests see reservations.
"""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

Policy = Literal["round_robin", "least_inflight"]


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


class WorkerPool:
    def __init__(
        self,
        urls: list[str],
        policy: Policy = "least_inflight",
        capacity: int = 4,
        max_queue: int = 16,
        queue_timeout: float = 2,
    ):
        urls = [url.rstrip("/") for url in urls]
        if not urls or len(set(urls)) != len(urls):
            raise ValueError("Provide at least one distinct worker URL")
        if capacity < 1 or max_queue < 0 or not math.isfinite(queue_timeout) or queue_timeout <= 0:
            raise ValueError("Invalid admission limits")
        if policy not in ("round_robin", "least_inflight"):
            raise ValueError("Unknown routing policy")
        self.workers = [Worker(f"worker-{i + 1}", url.rstrip("/")) for i, url in enumerate(urls)]
        self.policy = policy
        self.capacity = capacity
        self.max_queue = max_queue
        self.queue_timeout = queue_timeout
        self.condition = asyncio.Condition()
        self.waiters: deque[object] = deque()
        self.cursor = 0

    def _select(self) -> Worker | None:
        ordered = self.workers[self.cursor :] + self.workers[: self.cursor]
        candidates = [w for w in ordered if w.healthy and w.inflight < self.capacity]
        if not candidates:
            return None
        worker = (
            candidates[0]
            if self.policy == "round_robin"
            else min(candidates, key=lambda w: w.inflight)
        )
        self.cursor = (self.workers.index(worker) + 1) % len(self.workers)
        return worker

    async def acquire(self) -> Lease:
        started = time.perf_counter()
        ticket = object()
        async with self.condition:
            if not any(w.healthy for w in self.workers):
                raise AdmissionError(503, "no_healthy_workers")
            worker = self._select() if not self.waiters else None
            if worker is not None:
                worker.inflight += 1
                return Lease(worker, worker.generation, 0)
            if len(self.waiters) >= self.max_queue:
                raise AdmissionError(429, "queue_full")
            self.waiters.append(ticket)
            try:
                async with asyncio.timeout(self.queue_timeout):
                    while True:
                        if not any(w.healthy for w in self.workers):
                            raise AdmissionError(503, "no_healthy_workers")
                        if self.waiters[0] is ticket:
                            worker = self._select()
                            if worker is not None:
                                worker.inflight += 1
                                return Lease(
                                    worker, worker.generation, time.perf_counter() - started
                                )
                        await self.condition.wait()
            except TimeoutError as exc:
                raise AdmissionError(429, "queue_timeout") from exc
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    async def release(self, lease: Lease) -> None:
        async with self.condition:
            if not lease.released:
                lease.worker.inflight -= 1
                lease.released = True
                self.condition.notify_all()

    async def set_health(
        self, worker: Worker, healthy: bool, expected_generation: int | None = None
    ) -> None:
        async with self.condition:
            # A probe started before a request failure must not undo that failure.
            if expected_generation is not None and worker.generation != expected_generation:
                return
            if worker.healthy != healthy:
                worker.generation += 1
            worker.healthy = healthy
            worker.last_health_check = time.time()
            self.condition.notify_all()

    def snapshot(self) -> dict:
        return {
            "policy": self.policy,
            "capacity_per_worker": self.capacity,
            "queued": len(self.waiters),
            "max_queue": self.max_queue,
            "queue_timeout_seconds": self.queue_timeout,
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
