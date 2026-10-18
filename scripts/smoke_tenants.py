"""Verify tenant cache salts on the prepared native Metal runtime, then clean up.

Uses one worker so cross-tenant cold-cache behavior cannot be explained by routing
requests to different replicas. Does not alter the pinned inference environment.
"""

import asyncio
import json
import os
import socket
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from nano_inference.engine_metrics import cache_delta, engine_snapshot
from nano_inference.flavors import FlavorCluster, FlavorProfile, free_port, inspect_runtime
from nano_inference.tenant_benchmark import run_tenant_benchmark
from nano_inference.tenants import TenantConfig

ROOT = Path(__file__).resolve().parents[1]


async def main():
    profile = FlavorProfile.load(ROOT / "configs/macos-metal.json")
    config = TenantConfig.load(ROOT / "configs/tenants.example.json")
    for t in config.tenants:
        os.environ[t.api_key_env] = uuid.uuid4().hex
    ports = [free_port(), free_port()]
    output = ROOT / "results/macos-metal/tenants"
    output.mkdir(parents=True, exist_ok=True)
    runtime = await inspect_runtime(profile, ROOT)
    with tempfile.TemporaryDirectory(dir=ROOT / ".local", prefix="tenant-smoke-") as tmp:
        config_path = Path(tmp) / "tenants.json"
        config_path.write_text(config.model_dump_json())
        cluster = FlavorCluster(
            profile,
            ROOT,
            workers=1,
            port=ports[0],
            worker_ports=ports[1:],
            tenant_config=str(config_path),
            max_queue=24,
            request_timeout=90,
        )
        started = time.perf_counter()
        checks = []
        try:
            await cluster.start()
            async with httpx.AsyncClient(trust_env=False, timeout=95) as client:
                data = {
                    "model": profile.model_alias,
                    "stream": False,
                    "max_tokens": 8,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Isolated experiment "
                            + uuid.uuid4().hex
                            + ". A queue holds pending work until a worker can process it. " * 20,
                        },
                        {"role": "user", "content": "Name two colors."},
                    ],
                }
                worker = cluster.worker_urls[0]
                for tenant in ["interactive", "interactive", "batch", "batch"]:
                    before = await engine_snapshot(client, worker)
                    entry = next(t for t in config.tenants if t.id == tenant)
                    response = await client.post(
                        cluster.url + "/v1/chat/completions",
                        json=data,
                        headers={"Authorization": "Bearer " + os.environ[entry.api_key_env]},
                    )
                    response.raise_for_status()
                    async with asyncio.timeout(10):
                        while True:
                            after = await engine_snapshot(client, worker)
                            delta = cache_delta(before, after)
                            if delta["queried_tokens"] > 0:
                                break
                            await asyncio.sleep(0.2)
                    checks.append(
                        {
                            "tenant": tenant,
                            "worker": response.headers["x-worker-id"],
                            "route_decision": response.headers["x-route-decision"],
                            "usage": response.json()["usage"],
                            "cache": delta,
                        }
                    )
                assert checks[0]["cache"]["hit_tokens"] == 0, checks
                assert checks[1]["cache"]["hit_tokens"] > 0, checks
                assert checks[2]["cache"]["hit_tokens"] == 0, checks
                assert checks[3]["cache"]["hit_tokens"] > 0, checks
                assert {c["worker"] for c in checks} == {"worker-1"}
                # Authenticate failures, client cache-salt injection and local operator page.
                assert (
                    await client.post(cluster.url + "/v1/chat/completions", json=data)
                ).status_code == 401
                data["cache_salt"] = "client-cannot-select-this"
                h = {"Authorization": "Bearer " + os.environ[config.tenants[0].api_key_env]}
                assert (
                    await client.post(cluster.url + "/v1/chat/completions", json=data, headers=h)
                ).status_code == 422
                assert (await client.get(cluster.url + "/monitor")).status_code == 200
                await cluster_snapshot_ready(client, cluster.url)
                snapshot = (await client.get(cluster.url + "/monitor/api")).json()
                assert snapshot["engines"]["worker-1"]["available"]
                metrics = (await client.get(cluster.url + "/metrics")).text
                assert "nano_tenant_tokens_total" in metrics
                assert "nano_engine_kv_cache_usage_perc" in metrics
            load = await run_tenant_benchmark(
                cluster.url,
                profile.model_alias,
                tenant_config=str(config_path),
                requests=6,
                rate=5,
                max_tokens=32,
                output=str(output / "noisy-neighbor.json"),
            )
        finally:
            await cluster.close()
        for port in ports:
            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", port)) != 0
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "runtime": runtime,
        "model": profile.model_repo,
        "model_revision": profile.model_revision,
        "worker_count": 1,
        "cache_isolation_checks": checks,
        "operator_snapshot": snapshot,
        "workload_summary": load["summary"],
        "elapsed_seconds": time.perf_counter() - started,
        "ports_closed": True,
        "note": "One physical Apple GPU. Cache isolation is verified by native token counters; "
        "request fairness does not provide hard GPU latency isolation.",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


async def cluster_snapshot_ready(client, url):
    async with asyncio.timeout(10):
        while True:
            snapshot = (await client.get(url + "/monitor/api")).json()
            if snapshot["engines"]["worker-1"]["available"]:
                return
            await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main())
