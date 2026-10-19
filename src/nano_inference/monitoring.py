"""Local read-only dashboard and bounded native-engine metrics polling."""

import asyncio
import contextlib
import time
from pathlib import Path

import httpx
from fastapi.responses import HTMLResponse
from prometheus_client import Gauge

from .engine_metrics import engine_snapshot


def histogram_summary(count, total, buckets):
    if not count:
        return {"count": 0, "mean": None, "p95_upper_bound": None}
    bound = next((b for b, n in sorted(buckets) if n >= count * 0.95), None)
    return {
        "count": count,
        "mean": total / count,
        "p95_upper_bound": bound if bound is not None and bound != float("inf") else None,
    }


def metric_summary(registry):
    """Expose only our bounded labels; never prompts, credentials or request IDs."""
    tenants = {}
    overall = {}
    histograms = {}
    for family in registry.collect():
        for sample in family.samples:
            if not sample.name.startswith("nano_") or sample.name.endswith("_created"):
                continue
            labels = sample.labels
            target = tenants.setdefault(labels["tenant"], {}) if "tenant" in labels else overall
            name = sample.name
            if name == "nano_tenant_requests_total":
                outcomes = target.setdefault("outcomes", {})
                outcomes[labels["outcome"]] = sample.value
            elif name == "nano_tenant_rejections_total":
                target.setdefault("rejections", {})[labels["reason"]] = sample.value
            elif name == "nano_tenant_tokens_total":
                target[labels["kind"] + "_tokens"] = sample.value
            elif name == "nano_tenant_slo_observed_total":
                target["slo_observed"] = sample.value
            elif name == "nano_tenant_slo_met_total":
                target["slo_met"] = sample.value
            elif name.startswith(
                ("nano_tenant_first_content_seconds_", "nano_tenant_queue_seconds_")
            ):
                base = name.rsplit("_", 1)[0]
                key = (labels["tenant"], base)
                h = histograms.setdefault(key, {"count": 0, "sum": 0, "buckets": []})
                if name.endswith("_bucket"):
                    h["buckets"].append((float(labels["le"]), sample.value))
                else:
                    h[name.rsplit("_", 1)[1]] = sample.value
            elif name == "nano_requests_total":
                target.setdefault("outcomes", {})[labels["outcome"]] = sample.value
    for (tenant, base), h in histograms.items():
        field = "ttft" if "first_content" in base else "queue_wait"
        tenants[tenant][field] = histogram_summary(h["count"], h["sum"], h["buckets"])
    return {"overall": overall, "tenants": tenants}


class Monitor:
    def __init__(self, app, pool, registry, model):
        self.app, self.pool, self.registry, self.model = app, pool, registry, model
        self.started = time.monotonic()
        self.engines = {
            w.name: {"available": False, "sampled_at": None, "values": None} for w in pool.workers
        }
        self.up = Gauge(
            "nano_engine_metrics_up",
            "Native metrics scrape succeeded",
            ["worker"],
            registry=registry,
        )
        self.native = {}
        for name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
            self.native[name] = Gauge(
                "nano_engine_" + name, "Last native vLLM observation", ["worker"], registry=registry
            )
        for w in pool.workers:
            self.up.labels(w.name).set(0)
            for gauge in self.native.values():
                gauge.labels(w.name).set(float("nan"))
        for field in (
            "ttft_seconds",
            "queue_seconds",
            "prefill_seconds",
            "kv_seconds",
            "cached_tokens",
        ):
            gauge = Gauge(
                "nano_route_last_" + field,
                "Last candidate score (approximate)",
                ["worker"],
                registry=registry,
            )
            for w in pool.workers:

                def value(worker=w.name, attr=field):
                    prediction = pool.placement.last_scores.get(worker)
                    return getattr(prediction, attr) if prediction else float("nan")

                gauge.labels(w.name).set_function(value)

    async def poll(self):
        async def one(worker):
            generation = worker.generation
            try:
                async with asyncio.timeout(1):
                    data = await engine_snapshot(self.app.state.client, worker.url)
                self.engines[worker.name] = {"available": True, "sampled_at": time.time(), **data}
                self.up.labels(worker.name).set(1)
                await self.pool.observe_native(worker, generation, data["values"])
                for name, gauge in self.native.items():
                    gauge.labels(worker.name).set(data["values"].get("vllm:" + name, float("nan")))
            except (httpx.HTTPError, TimeoutError, ValueError):
                await self.pool.observe_native(worker, generation, None)
                self.engines[worker.name] = {
                    "available": False,
                    "sampled_at": time.time(),
                    "values": None,
                }
                self.up.labels(worker.name).set(0)
                for gauge in self.native.values():
                    gauge.labels(worker.name).set(float("nan"))

        await asyncio.gather(*(one(w) for w in self.pool.workers))

    async def run(self):
        while True:
            await self.poll()
            await asyncio.sleep(2)

    def install(self):
        @self.app.get("/", include_in_schema=False)
        @self.app.get("/monitor", include_in_schema=False)
        async def page():
            return HTMLResponse(Path(__file__).with_name("dashboard.html").read_text())

        @self.app.get("/experiments", include_in_schema=False)
        async def experiments():
            path = Path.cwd() / "results/experiments/report.html"
            if not path.exists():
                return HTMLResponse(
                    "Run the control and Metal experiment batches to generate this report.",
                    status_code=404,
                )
            return HTMLResponse(path.read_text())

        @self.app.get("/monitor/api", include_in_schema=False)
        async def snapshot():
            return {
                "sampled_at": time.time(),
                "uptime_seconds": time.monotonic() - self.started,
                "model": self.model,
                "simulated": self.model == "nano-model",
                "pool": self.pool.snapshot(),
                "metrics": metric_summary(self.registry),
                "engines": self.engines,
            }

    @contextlib.asynccontextmanager
    async def lifespan(self):
        task = asyncio.create_task(self.run())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
