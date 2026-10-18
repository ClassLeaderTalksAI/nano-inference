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

from .protocol import ChatRequest, has_content, sse, sse_data
from .routing import AdmissionError, Lease, Policy, WorkerPool

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

    def __post_init__(self):
        timings = (self.request_timeout, self.health_interval, self.health_timeout)
        if any(not math.isfinite(t) or t <= 0 for t in timings):
            raise ValueError("Timeouts and health interval must be positive")
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


def create_gateway(settings: Settings | None = None, *, transport=None) -> FastAPI:
    settings = settings or Settings()
    pool = WorkerPool(
        settings.worker_urls,
        settings.policy,
        settings.capacity,
        settings.max_queue,
        settings.queue_timeout,
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
    for worker in pool.workers:
        active.labels(worker.name).set_function(lambda w=worker: w.inflight)

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
                yield
            finally:
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor

    app = FastAPI(title="nano-inference", lifespan=lifespan)
    app.state.pool = pool
    app.state.probe = probe

    @app.get("/health")
    async def health():
        return {"status": "alive"}

    @app.get("/ready")
    async def ready():
        healthy = any(w.healthy for w in pool.workers)
        return JSONResponse({"ready": healthy}, status_code=200 if healthy else 503)

    @app.get("/workers")
    async def workers():
        return pool.snapshot()

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
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
                    await pool.release(lease)
                elapsed = time.perf_counter() - started
                completed.labels(outcome).inc()
                duration.observe(elapsed)
                logger.info(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "worker": lease.worker.name if lease else None,
                            "policy": settings.policy,
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
                if await request.is_disconnected():
                    disconnected = True
                    owner.cancel()
                    return
                await asyncio.sleep(0.02)

        watcher = asyncio.create_task(watch_disconnect())
        handing_off = False
        try:
            async with asyncio.timeout_at(deadline):
                lease = await pool.acquire()
                queue_delay.observe(lease.queue_seconds)
                payload = body.model_dump(exclude_none=True)
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
                headers = {"X-Request-ID": request_id, "X-Worker-ID": lease.worker.name}
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
