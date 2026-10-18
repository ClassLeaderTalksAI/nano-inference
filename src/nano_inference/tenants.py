"""Single-gateway tenant admission and token-cost weighted deficit round robin.

The pool condition owns queues, quotas and reservations together. No worker slot
is held while waiting for a tenant permit, and cache placement runs only after
fair scheduling. These are request-level controls, not GPU time guarantees.
"""

import asyncio
import hashlib
import hmac
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from .routing import AdmissionError, Lease, WorkerPool


class Tenant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    weight: int = Field(default=1, ge=1, le=100)
    rpm: int = Field(default=120, ge=1, le=1_000_000)
    tpm: int = Field(default=100_000, ge=1, le=100_000_000)
    max_inflight: int = Field(default=2, ge=1, le=1024)
    max_queue: int = Field(default=8, ge=0, le=10_000)
    expected_output_tokens: int = Field(default=32, ge=1, le=4096)
    ttft_slo_seconds: float = Field(default=2, gt=0, le=3600, allow_inf_nan=False)
    # Optional placement boundary, using server-owned worker IDs (worker-1, ...).
    workers: list[str] | None = Field(default=None, min_length=1)


class TenantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenants: list[Tenant] = Field(min_length=1, max_length=64)
    scheduler: str = Field(default="wdrr", pattern=r"^(wdrr|fifo)$")
    quantum: int = Field(default=32, ge=1, le=1_000_000)
    decode_cost: float = Field(default=2, gt=0, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique(self):
        for attr in ("id", "api_key_env"):
            values = [getattr(t, attr) for t in self.tenants]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate tenant {attr}")
        return self

    @classmethod
    def load(cls, path):
        return cls.model_validate_json(Path(path).read_text())


@dataclass
class Bucket:
    capacity: float
    balance: float
    updated: float

    def refill(self, now):
        self.balance = min(self.capacity, self.balance + (now - self.updated) * self.capacity / 60)
        self.updated = now


@dataclass
class Ticket:
    tenant: str
    key: str | None
    cost: float
    reserved: int
    reserved_cost: float
    started: float
    lease: Lease | None = None


@dataclass
class TenantState:
    config: Tenant
    rpm: Bucket
    tpm: Bucket
    queue: deque = field(default_factory=deque)
    inflight: int = 0
    deficit: float = 0
    dispatched_cost: float = 0
    actual_cost: float = 0
    usage_settlements: int = 0
    conservative_settlements: int = 0
    conservative_cost: float = 0


class TenantPool(WorkerPool):
    def __init__(self, *args, tenants: TenantConfig, clock=time.monotonic, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = tenants
        self.clock = clock
        self.states = {}
        self.keys: dict[str, SecretStr] = {}
        self.salt_secret = secrets.token_bytes(32)
        now = clock()
        known_workers = {w.name for w in self.workers}
        for tenant in tenants.tenants:
            key = os.environ.get(tenant.api_key_env, "")
            if len(key) < 16:
                raise ValueError(f"Set {tenant.api_key_env} to a key of at least 16 characters")
            if any(
                hmac.compare_digest(key.encode(), other.get_secret_value().encode())
                for other in self.keys.values()
            ):
                raise ValueError("Tenant API keys must be distinct")
            if tenant.workers and not set(tenant.workers) <= known_workers:
                raise ValueError(f"Unknown worker in tenant {tenant.id}")
            self.keys[tenant.id] = SecretStr(key)
            self.states[tenant.id] = TenantState(
                tenant, Bucket(tenant.rpm, tenant.rpm, now), Bucket(tenant.tpm, tenant.tpm, now)
            )
        self.rotation: deque[str] = deque()
        self.visiting: str | None = None
        # A single maximum-size accepted request must eventually fit its credit.
        self.deficit_cap = 200_000 + 4096 * tenants.decode_cost + 100 * tenants.quantum

    def authenticate(self, authorization: str | None) -> str:
        if authorization and authorization.startswith("Bearer "):
            supplied = authorization[7:]
            for tenant, key in self.keys.items():
                if hmac.compare_digest(supplied.encode(), key.get_secret_value().encode()):
                    return tenant
        raise AdmissionError(401, "invalid_api_key")

    def salt(self, tenant: str) -> str:
        # Process-lifetime secret: stable within a run, regenerated on gateway restart.
        return hmac.new(self.salt_secret, tenant.encode(), hashlib.sha256).hexdigest()

    def _eligible(self, tenant):
        allowed = self.states[tenant].config.workers
        return [w for w in self.workers if w.healthy and (not allowed or w.name in allowed)]

    def _candidate(self, tenant):
        s = self.states[tenant]
        return s.inflight < s.config.max_inflight and any(
            w.inflight < self.capacity for w in self._eligible(tenant)
        )

    def _next(self):
        eligible = [t for t in self.rotation if self._candidate(t)]
        if not eligible:
            return None
        if self.config.scheduler == "fifo":
            return min(eligible, key=lambda t: self.states[t].queue[0].started)
        # DRR retains the current visit while its credit can pay for more requests.
        # Skipping a concurrency-blocked tenant never accrues idle credit.
        while True:
            tenant = self.rotation[0]
            s = self.states[tenant]
            if tenant not in eligible:
                self.visiting = None
                self.rotation.rotate(-1)
                continue
            if self.visiting != tenant:
                s.deficit = min(self.deficit_cap, s.deficit + s.config.weight * self.config.quantum)
                self.visiting = tenant
            if s.queue[0].cost <= s.deficit:
                s.deficit -= s.queue[0].cost
                return tenant
            self.visiting = None
            self.rotation.rotate(-1)

    def _pump(self):
        while (tenant := self._next()) is not None:
            s = self.states[tenant]
            ticket = s.queue.popleft()
            allowed = s.config.workers
            selected = self._select(ticket.key, allowed)
            assert selected is not None  # Eligibility and reservation share the condition lock.
            ticket.lease = self._reserve(selected, ticket.key, self.clock() - ticket.started)
            ticket.lease.tenant_id = tenant
            ticket.lease.estimated_cost = ticket.cost
            ticket.lease.reserved_tokens = ticket.reserved
            ticket.lease.reserved_cost = ticket.reserved_cost
            s.inflight += 1
            s.dispatched_cost += ticket.cost
            self.waiters.remove(ticket)
            if not s.queue:
                self.rotation.remove(tenant)
                self.visiting = None
                s.deficit = min(0, s.deficit)  # No credit banking while idle.
        self.condition.notify_all()

    async def acquire(self, prefix_key=None, *, tenant: str, input_tokens: int, max_tokens: int):
        s = self.states[tenant]
        cost = input_tokens + self.config.decode_cost * min(
            max_tokens, s.config.expected_output_tokens
        )
        reserved = input_tokens + max_tokens  # Quota reserves the full output limit.
        ticket = Ticket(
            tenant,
            prefix_key,
            cost,
            reserved,
            input_tokens + self.config.decode_cost * max_tokens,
            self.clock(),
        )
        self.deficit_cap = max(self.deficit_cap, cost + s.config.weight * self.config.quantum)
        async with self.condition:
            self._pump()  # First serve older eligible work; blocked tenants cannot hold idle slots.
            if not self._eligible(tenant):
                raise AdmissionError(503, "no_healthy_workers")
            now = self.clock()
            s.rpm.refill(now)
            s.tpm.refill(now)
            if s.rpm.balance < 1:
                raise AdmissionError(429, "tenant_rpm")
            if s.tpm.balance < reserved:
                raise AdmissionError(429, "tenant_tpm")
            # Capacity checks precede quota consumption. A tenant's full queue
            # cannot evict already accepted requests from another tenant.
            can_start = not s.queue and self._candidate(tenant)
            if not can_start:
                if len(s.queue) >= s.config.max_queue:
                    raise AdmissionError(429, "tenant_queue_full")
                if len(self.waiters) >= self.max_queue:
                    raise AdmissionError(429, "queue_full")
            s.rpm.balance -= 1
            s.tpm.balance -= reserved
            s.queue.append(ticket)
            self.waiters.append(ticket)
            if tenant not in self.rotation:
                self.rotation.append(tenant)
            returned = False
            try:
                async with asyncio.timeout(self.queue_timeout):
                    while True:
                        self._pump()
                        if ticket.lease is not None:
                            returned = True
                            return ticket.lease
                        if not self._eligible(tenant):
                            raise AdmissionError(503, "no_healthy_workers")
                        await self.condition.wait()
            except TimeoutError as exc:
                raise AdmissionError(429, "queue_timeout") from exc
            finally:
                if not returned:
                    if ticket.lease is not None:
                        self._finish(ticket.lease, None, False, dispatched=False)
                    else:
                        s.queue.remove(ticket)
                        self.waiters.remove(ticket)
                        s.tpm.refill(self.clock())
                        s.tpm.balance = min(s.tpm.capacity, s.tpm.balance + reserved)
                        if not s.queue:
                            self.rotation.remove(tenant)
                            self.visiting = None
                            s.deficit = min(0, s.deficit)
                    self._pump()

    def _finish(self, lease, usage, successful, dispatched=True):
        if lease.released:
            return
        s = self.states[lease.tenant_id]
        s.tpm.refill(self.clock())
        if not dispatched:
            s.tpm.balance = min(s.tpm.capacity, s.tpm.balance + lease.reserved_tokens)
            s.deficit = min(self.deficit_cap, s.deficit + lease.estimated_cost)
        elif usage and all(
            type(usage.get(k)) is int and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            actual_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
            actual_cost = (
                usage["prompt_tokens"] + self.config.decode_cost * usage["completion_tokens"]
            )
            s.tpm.balance = min(
                s.tpm.capacity, s.tpm.balance + lease.reserved_tokens - actual_tokens
            )
            s.deficit = max(
                -self.deficit_cap,
                min(self.deficit_cap, s.deficit + lease.estimated_cost - actual_cost),
            )
            s.actual_cost += actual_cost
            s.usage_settlements += 1
        else:
            # Disconnects/partial failures usually have no final usage. Keep the
            # full reservation charged; never turn cancellation into free tokens.
            s.conservative_settlements += 1
            s.conservative_cost += lease.reserved_cost
            s.deficit = max(
                -self.deficit_cap, s.deficit + lease.estimated_cost - lease.reserved_cost
            )
        s.inflight -= 1
        if not s.queue and not s.inflight:
            s.deficit = min(0, s.deficit)
        self._release(lease, successful)

    async def release(self, lease, *, successful=False, usage=None):
        async with self.condition:
            self._finish(lease, usage, successful)
            self._pump()

    def snapshot(self):
        now = self.clock()
        tenants = []
        for s in self.states.values():
            s.rpm.refill(now)
            s.tpm.refill(now)
            tenants.append(
                {
                    "id": s.config.id,
                    "weight": s.config.weight,
                    "inflight": s.inflight,
                    "queued": len(s.queue),
                    "max_inflight": s.config.max_inflight,
                    "max_queue": s.config.max_queue,
                    "rpm_remaining": s.rpm.balance,
                    "tpm_remaining": s.tpm.balance,
                    "ttft_slo_seconds": s.config.ttft_slo_seconds,
                    "deficit": s.deficit,
                    "estimated_dispatched_cost": s.dispatched_cost,
                    "actual_settled_cost": s.actual_cost,
                    "usage_settlements": s.usage_settlements,
                    "conservative_settlements": s.conservative_settlements,
                    "conservative_cost": s.conservative_cost,
                    "workers": s.config.workers,
                }
            )
        return {**super().snapshot(), "scheduler": self.config.scheduler, "tenants": tenants}
