"""A single-process gateway; the inference engine remains an HTTP dependency."""

import asyncio
import contextlib
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .monitoring import Monitor
from .prefix import PrefixEncoder, PrefixPreparer, digest
from .protocol import ChatRequest, has_content, sse, sse_data
from .routing import AdmissionError, Lease, Policy, WorkerPool
from .tenants import TenantConfig, TenantPool

logger = logging.getLogger("nano_inference.requests")


@dataclass
class Settings:
    worker_urls: list[str] = field(
        default_factory=lambda: ["http://127.0.0.1:8101", "http://127.0.0.1:8102"]
    )
    model: str = "nano-model"
    policy: Policy = "least_inflight"
    capacity: int = 4
    max_queue: int = 16
    queue_timeout: float = 2
    request_timeout: float = 30
    health_interval: float = 1
    health_timeout: float = 1
    model_revision: str = "simulator-v1"
    prefix_tokenizer: str | None = None
    prefix_tokens: int = 64
    prefix_preparations: int = 4
    affinity_capacity: int = 1024
    affinity_ttl: float = 60
    affinity_max_skew: int = 1
    tenant_config: str | None = None

    def __post_init__(self):
        timings = (self.request_timeout, self.health_interval, self.health_timeout)
        if any(not math.isfinite(t) or t <= 0 for t in timings):
            raise ValueError("Timeouts and health interval must be positive")
        if self.prefix_tokens < 1 or self.prefix_preparations < 1 or not self.model_revision:
            raise ValueError("Prefix settings require positive counts and a model revision")
        if (self.policy == "prefix_affinity" or self.tenant_config) and self.model != "nano-model":
            if not self.prefix_tokenizer or self.model_revision == "simulator-v1":
                raise ValueError(
                    "Real-model affinity requires a local tokenizer and model revision"
                )
        for url in self.worker_urls:
            parsed = httpx.URL(url)
            if parsed.scheme not in ("http", "https") or not parsed.host:
                raise ValueError("Workers require an absolute HTTP(S) URL")


class ManagedStream(StreamingResponse):
    """Release even when disconnected before the body iterator starts."""

    def __init__(self, content, cleanup: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(content, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette may cancel its task group on disconnect; shield cleanup from that scope.
            with anyio.CancelScope(shield=True):
                await self.cleanup()


def create_gateway(
    settings: Settings | None = None, *, transport=None, prefix_encoder=None
) -> FastAPI:
    settings = settings or Settings()
    tenant_config = TenantConfig.load(settings.tenant_config) if settings.tenant_config else None
    pool_type = TenantPool if tenant_config else WorkerPool
    pool = pool_type(
        settings.worker_urls,
        settings.policy,
        settings.capacity,
        settings.max_queue,
        settings.queue_timeout,
        settings.affinity_capacity,
        settings.affinity_ttl,
        settings.affinity_max_skew,
        **({"tenants": tenant_config} if tenant_config else {}),
    )
    registry = CollectorRegistry()
    completed = Counter(
        "nano_requests_total", "Completed request lifecycles", ["outcome"], registry=registry
    )
    rejected = Counter(
        "nano_rejections_total", "Admission rejections", ["reason"], registry=registry
    )
    duration = Histogram(
        "nano_request_seconds", "Gateway request lifecycle duration", registry=registry
    )
    first_content = Histogram(
        "nano_first_content_seconds", "Gateway arrival to first content", registry=registry
    )
    queue_delay = Histogram("nano_queue_seconds", "Gateway admission wait", registry=registry)
    active = Gauge("nano_worker_inflight", "Reserved requests", ["worker"], registry=registry)
    queued = Gauge("nano_queued_requests", "Requests waiting for admission", registry=registry)
    queued.set_function(lambda: len(pool.waiters))
    route_decisions = Counter(
        "nano_route_decisions_total",
        "Routing decisions, not cache hits",
        ["decision"],
        registry=registry,
    )
    prefix_inflight = Gauge(
        "nano_prefix_preparations", "Running tokenizer preparations", registry=registry
    )
    prefix_inflight.set_function(lambda: len(app.state.preparer.tasks) if app.state.preparer else 0)
    healthy_metric = Gauge(
        "nano_worker_healthy", "Worker health state", ["worker"], registry=registry
    )
    for worker in pool.workers:
        healthy_metric.labels(worker.name).set_function(lambda w=worker: int(w.healthy))
        active.labels(worker.name).set_function(lambda w=worker: w.inflight)

    tenant_requests = Counter(
        "nano_tenant_requests_total",
        "Tenant request lifecycles",
        ["tenant", "outcome"],
        registry=registry,
    )
    tenant_rejections = Counter(
        "nano_tenant_rejections_total",
        "Tenant admission rejections",
        ["tenant", "reason"],
        registry=registry,
    )
    tenant_tokens = Counter(
        "nano_tenant_tokens_total",
        "Final backend usage tokens",
        ["tenant", "kind"],
        registry=registry,
    )
    tenant_ttft = Histogram(
        "nano_tenant_first_content_seconds",
        "Arrival to first content",
        ["tenant"],
        registry=registry,
    )
    tenant_queue = Histogram(
        "nano_tenant_queue_seconds", "Tenant admission wait", ["tenant"], registry=registry
    )
    tenant_slo = Counter(
        "nano_tenant_slo_met_total",
        "Successful streams meeting tenant TTFT SLO",
        ["tenant"],
        registry=registry,
    )
    tenant_slo_observed = Counter(
        "nano_tenant_slo_observed_total",
        "Successful streams with first content",
        ["tenant"],
        registry=registry,
    )
    for metric, attr in (
        ("inflight", "inflight"),
        ("queued", "queued"),
        ("rpm_remaining", "rpm_remaining"),
        ("tpm_remaining", "tpm_remaining"),
        ("estimated_dispatched_cost", "estimated_dispatched_cost"),
        ("actual_settled_cost", "actual_settled_cost"),
        ("deficit", "deficit"),
        ("usage_settlements", "usage_settlements"),
        ("conservative_settlements", "conservative_settlements"),
        ("conservative_cost", "conservative_cost"),
    ):
        gauge = Gauge("nano_tenant_" + metric, "Tenant " + metric, ["tenant"], registry=registry)
        if tenant_config:
            for config in tenant_config.tenants:

                def value(tenant=config.id, field=attr):
                    return next(t[field] for t in pool.snapshot()["tenants"] if t["id"] == tenant)

                gauge.labels(config.id).set_function(value)

    async def probe():
        async def check(worker):
            generation = worker.generation
            try:
                response = await app.state.client.get(
                    worker.url + "/health", timeout=settings.health_timeout
                )
                healthy = response.status_code == 200
            except httpx.HTTPError:
                healthy = False
            await pool.set_health(worker, healthy, expected_generation=generation)

        await asyncio.gather(*(check(w) for w in pool.workers))

    async def health_loop():
        while True:
            await asyncio.sleep(settings.health_interval)
            await probe()

    @asynccontextmanager
    async def lifespan(app):
        if settings.policy == "prefix_affinity" or tenant_config:
            encoder = prefix_encoder
            if encoder is None:
                if settings.prefix_tokenizer:
                    encoder = await asyncio.to_thread(
                        PrefixEncoder.from_pretrained,
                        settings.model,
                        settings.model_revision,
                        settings.prefix_tokens,
                        settings.prefix_tokenizer,
                    )
                else:
                    encoder = PrefixEncoder(
                        settings.model, settings.model_revision, settings.prefix_tokens
                    )
            app.state.preparer = PrefixPreparer(encoder, settings.prefix_preparations)
        # Keep one pooled client per gateway; health requests have their own short timeout.
        async with httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(settings.request_timeout, connect=3),
            limits=httpx.Limits(max_connections=None),
            trust_env=False,
        ) as client:
            app.state.client = client
            await probe()
            monitor = asyncio.create_task(health_loop())
            try:
                async with dashboard.lifespan():
                    yield
            finally:
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor
                if app.state.preparer:
                    await app.state.preparer.close()

    app = FastAPI(title="nano-inference", lifespan=lifespan)
    app.state.pool = pool
    app.state.probe = probe
    app.state.preparer = None
    app.state.registry = registry
    dashboard = Monitor(app, pool, registry, settings.model)
    app.state.dashboard = dashboard
    dashboard.install()

    @app.get("/health")
    async def health():
        return {"status": "alive"}

    @app.get("/ready")
    async def ready():
        healthy = any(w.healthy for w in pool.workers)
        return JSONResponse({"ready": healthy}, status_code=200 if healthy else 503)

    @app.get("/workers")
    async def workers():
        return {
            **pool.snapshot(),
            "prefix_preparation": {
                "capacity": settings.prefix_preparations,
                "inflight": len(app.state.preparer.tasks) if app.state.preparer else 0,
            },
            "prefix_identity": app.state.preparer.encoder.identity if app.state.preparer else None,
        }

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        tenant_id = "default"
        if tenant_config:
            try:
                tenant_id = pool.authenticate(request.headers.get("authorization"))
            except AdmissionError as exc:
                rejected.labels(exc.code).inc()
                completed.labels(exc.code).inc()
                return JSONResponse(
                    {"error": {"code": exc.code}},
                    status_code=exc.status,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        if body.model != settings.model:
            return JSONResponse({"error": {"code": "unknown_model"}}, status_code=404)
        started = time.perf_counter()
        deadline = asyncio.get_running_loop().time() + settings.request_timeout
        request_id = uuid.uuid4().hex
        lease: Lease | None = None
        upstream: httpx.Response | None = None
        outcome = "cancelled"
        first_seconds: float | None = None
        usage: dict = {}
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                if lease is not None:
                    await pool.release(
                        lease,
                        successful=outcome == "success",
                        **({"usage": usage} if tenant_config else {}),
                    )
                elapsed = time.perf_counter() - started
                completed.labels(outcome).inc()
                duration.observe(elapsed)
                tenant_requests.labels(tenant_id, outcome).inc()
                if outcome == "success":
                    for name, kind in (
                        ("prompt_tokens", "prompt"),
                        ("completion_tokens", "output"),
                    ):
                        if type(usage.get(name)) is int and usage[name] >= 0:
                            tenant_tokens.labels(tenant_id, kind).inc(usage[name])
                    if first_seconds is not None and tenant_config:
                        tenant_slo_observed.labels(tenant_id).inc()
                        if first_seconds <= pool.states[tenant_id].config.ttft_slo_seconds:
                            tenant_slo.labels(tenant_id).inc()
                logger.info(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "tenant": tenant_id,
                            "worker": lease.worker.name if lease else None,
                            "policy": settings.policy,
                            "route_decision": lease.decision if lease else None,
                            "outcome": outcome,
                            "queue_seconds": lease.queue_seconds if lease else None,
                            "first_content_seconds": first_seconds,
                            "duration_seconds": elapsed,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "output_tokens": usage.get("completion_tokens"),
                        }
                    )
                )

        def error(status: int, code: str):
            return JSONResponse(
                {"error": {"code": code, "request_id": request_id}},
                status_code=status,
                headers={
                    "X-Request-ID": request_id,
                    **({"Retry-After": "1"} if status == 429 else {}),
                },
            )

        async def fail_worker():
            if lease is not None:
                # Only invalidate this generation; an older request cannot undo newer recovery.
                await pool.set_health(lease.worker, False, expected_generation=lease.generation)

        # During admission and response-header wait there is no StreamingResponse yet to
        # observe http.disconnect. Cancel this task if the caller leaves during that phase.
        owner = asyncio.current_task()
        disconnected = False

        async def watch_disconnect():
            nonlocal disconnected
            while True:
                # The request body has already been parsed. Wait for the ASGI
                # disconnect event directly; is_disconnected() uses an AnyIO
                # cancellation scope that can swallow watcher.cancel() at handoff.
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    disconnected = True
                    owner.cancel()
                    return

        watcher = asyncio.create_task(watch_disconnect())
        handing_off = False
        try:
            async with asyncio.timeout_at(deadline):
                key = None
                input_tokens = 0
                if app.state.preparer:
                    if sum(len(m.content) for m in body.messages) > 200_000:
                        outcome = "prefix_input_too_large"
                        return error(413, outcome)
                    if tenant_config:
                        key, input_tokens = await app.state.preparer.prepare(body.messages)
                        key = digest([tenant_id, key]) if key is not None else None
                    else:
                        key = await app.state.preparer.key(body.messages)
                lease = await pool.acquire(
                    key,
                    **(
                        {
                            "tenant": tenant_id,
                            "input_tokens": input_tokens,
                            "max_tokens": body.max_tokens,
                        }
                        if tenant_config
                        else {}
                    ),
                )
                route_decisions.labels(lease.decision).inc()
                queue_delay.observe(lease.queue_seconds)
                tenant_queue.labels(tenant_id).observe(lease.queue_seconds)
                payload = body.model_dump(exclude_none=True)
                if tenant_config:
                    payload["cache_salt"] = pool.salt(tenant_id)
                if body.stream:
                    # Ask the backend for exact usage; SSE chunks are not tokenizer tokens.
                    payload["stream_options"] = {"include_usage": True}
                upstream = await app.state.client.send(
                    app.state.client.build_request(
                        "POST", lease.worker.url + "/v1/chat/completions", json=payload
                    ),
                    stream=True,
                )
                if upstream.status_code != 200:
                    outcome = "upstream_http_error"
                    if upstream.status_code >= 500:
                        await fail_worker()
                    return error(429 if upstream.status_code == 429 else 502, outcome)
                headers = {
                    "X-Request-ID": request_id,
                    "X-Worker-ID": lease.worker.name,
                    "X-Route-Decision": lease.decision,
                }
                if not body.stream:
                    data = json.loads(await upstream.aread())
                    if not isinstance(data, dict) or not data.get("choices") or "error" in data:
                        raise ValueError("Invalid completion")
                    usage.update(data.get("usage") or {})
                    outcome = "success"
                    return JSONResponse(data, headers=headers)
                if "text/event-stream" not in upstream.headers.get("content-type", ""):
                    outcome = "invalid_upstream_response"
                    return error(502, outcome)

            async def forward():
                nonlocal outcome, first_seconds
                saw_finish = False
                try:
                    async with asyncio.timeout_at(deadline):
                        async for data in sse_data(upstream.aiter_lines()):
                            if data == "[DONE]":
                                if not saw_finish:
                                    raise ValueError("Missing finish_reason")
                                outcome = "success"
                                yield sse("[DONE]")
                                return
                            event = json.loads(data)
                            if not isinstance(event, dict):
                                raise ValueError("Invalid SSE JSON")
                            if "error" in event:
                                raise ValueError("Upstream SSE error")
                            if isinstance(event.get("usage"), dict):
                                usage.update(event["usage"])
                            if has_content(event) and first_seconds is None:
                                first_seconds = time.perf_counter() - started
                                first_content.observe(first_seconds)
                                tenant_ttft.labels(tenant_id).observe(first_seconds)
                            saw_finish |= any(
                                c.get("finish_reason") is not None for c in event.get("choices", [])
                            )
                            if (
                                not event.get("choices")
                                and event.get("usage")
                                and not (body.stream_options and body.stream_options.include_usage)
                            ):
                                continue
                            yield sse(event)
                    raise ValueError("Stream ended without DONE")
                except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError):
                    outcome = "stream_error"
                    await fail_worker()
                    # Headers are already sent. An error event without DONE makes the
                    # partial response visibly unsuccessful; never silently replay it.
                    yield sse(
                        {"error": {"code": "upstream_stream_error", "request_id": request_id}}
                    )

            response = ManagedStream(
                forward(),
                cleanup,
                media_type="text/event-stream",
                headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
            handing_off = True
            return response
        except AdmissionError as exc:
            outcome = exc.code
            rejected.labels(exc.code).inc()
            tenant_rejections.labels(tenant_id, exc.code).inc()
            return error(exc.status, exc.code)
        except asyncio.CancelledError:
            if disconnected:
                # The connection has gone; finish quietly after cancelling upstream work.
                return Response(status_code=499)
            raise
        except (httpx.HTTPError, TimeoutError) as exc:
            outcome = (
                "upstream_timeout"
                if isinstance(exc, (TimeoutError, httpx.TimeoutException))
                else "upstream_unavailable"
            )
            await fail_worker()
            return error(504 if outcome == "upstream_timeout" else 502, outcome)
        except (ValueError, TypeError):
            outcome = "invalid_upstream_response"
            return error(502, outcome)
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            if not handing_off:
                with anyio.CancelScope(shield=True):
                    await cleanup()

    return app
