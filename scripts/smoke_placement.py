"""Native Metal smoke: predicted routing, tenant namespaces and model error.

Starts two owned replicas on free ports and always reaps them. This is a small
correctness/observability run, not a comparative optimization benchmark.
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

from nano_inference.benchmark import send_request
from nano_inference.engine_metrics import cache_delta
from nano_inference.experiments import ExperimentConfig, ExperimentRunner, source_metadata
from nano_inference.flavors import FlavorCluster, FlavorProfile, free_port, inspect_runtime
from nano_inference.tenants import TenantConfig

ROOT = Path(__file__).resolve().parents[1]


async def main():
    profile = FlavorProfile.load(ROOT / "configs/macos-metal.json")
    tenants = TenantConfig.load(ROOT / "configs/tenants.example.json")
    keys = {t.id: uuid.uuid4().hex for t in tenants.tenants}
    for t in tenants.tenants:
        os.environ[t.api_key_env] = keys[t.id]
    runtime = await inspect_runtime(profile, ROOT)
    ports = [free_port() for _ in range(3)]
    output = ROOT / "results/placement/metal-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(dir=ROOT / ".local", prefix="placement-smoke-") as tmp:
        path = Path(tmp) / "tenants.json"
        path.write_text(tenants.model_dump_json())
        cluster = FlavorCluster(
            profile,
            ROOT,
            workers=2,
            port=ports[0],
            worker_ports=ports[1:],
            policy="predicted_ttft",
            tenant_config=str(path),
            request_timeout=90,
            placement_config=str(ROOT / "configs/placement.example.json"),
            log_dir=ROOT / ".local/logs/placement-smoke",
        )
        runner = ExperimentRunner(ROOT, ExperimentConfig(), Path(tmp) / "counters")
        try:
            await cluster.start()
            messages = [
                {
                    "role": "system",
                    "content": "Placement experiment "
                    + uuid.uuid4().hex
                    + ". "
                    + "A distributed queue holds pending requests until a worker is available. "
                    * 32,
                },
                {"role": "user", "content": "Name two colors."},
            ]
            async with httpx.AsyncClient(trust_env=False, timeout=95) as client:
                for i, tenant in enumerate(["interactive", "interactive", "batch", "batch"]):
                    before = await runner.native_samples(client, cluster.worker_urls)
                    result = await send_request(
                        client,
                        cluster.url,
                        profile.model_alias,
                        8,
                        i,
                        time.perf_counter(),
                        messages,
                        {"Authorization": "Bearer " + keys[tenant]},
                    )
                    assert result.outcome == "success", result
                    after = await runner.wait_native_usage(
                        client, cluster.worker_urls, before, [result]
                    )
                    snapshot = (await client.get(cluster.url + "/monitor/api")).json()
                    placement = snapshot["pool"]["placement"]
                    prediction = placement["last_scores"][result.worker]
                    deltas = [cache_delta(b, a) for b, a in zip(before, after, strict=True)]
                    hits = sum(d["hit_tokens"] for d in deltas)
                    assert (hits > 0) == (i % 2 == 1), (i, hits)
                    assert (prediction["cached_tokens"] > 0) == (i % 2 == 1), prediction
                    checks.append(
                        {
                            "tenant": tenant,
                            "result": result.__dict__,
                            "prediction": prediction,
                            "cache": deltas,
                        }
                    )
                metrics = (await client.get(cluster.url + "/metrics")).text
                assert "nano_route_prediction_absolute_error_seconds_count" in metrics
                assert placement["active_reservations"] == 0
                assert all(w["inflight"] == 0 for w in snapshot["pool"]["workers"])
                assert (await client.get(cluster.url + "/monitor")).status_code == 200
        finally:
            await cluster.close()
    for port in ports:
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0
    report = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "metadata": source_metadata(ROOT),
        "runtime": runtime,
        "checks": checks,
        "final_placement": placement,
        "ports_closed": True,
        "scope": "Two replicas share one Apple GPU. Four sequential requests verify native "
        "cache namespaces and routing observability, not performance improvement or SLO goodput. "
        "Service rates are uncalibrated example priors; compare predictions with observations.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "requests": len(checks), "ports_closed": True}))


if __name__ == "__main__":
    asyncio.run(main())
