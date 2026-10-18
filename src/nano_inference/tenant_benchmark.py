"""Bounded open-loop noisy-neighbor workload; credentials never enter reports."""

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import httpx

from .benchmark import Result, send_request, summarize, workload_messages
from .tenants import TenantConfig


async def run_tenant_benchmark(
    url="http://127.0.0.1:8000",
    model="nano-model",
    tenant_config="configs/tenants.example.json",
    requests=30,
    rate=5.0,
    max_outstanding=64,
    max_tokens=32,
    output="results/tenants-latest.json",
):
    config = TenantConfig.load(tenant_config)
    if requests < 1 or max_outstanding < 1 or not 1 <= max_tokens <= 4096:
        raise ValueError("Positive request limits required")
    if not 0 < rate <= 1000:
        raise ValueError("Rate must be finite and between 0 and 1000")
    credentials = {t.id: os.environ[t.api_key_env] for t in config.tenants}
    all_results = []
    started = time.perf_counter()
    pending = set()
    async with httpx.AsyncClient(
        trust_env=False, timeout=120, limits=httpx.Limits(max_connections=None)
    ) as client:
        before = (await client.get(url + "/monitor/api")).json()

        async def one(tenant, index, scheduled):
            # Batch sends longer output and 4x arrivals; interactive requests stay small.
            limit = max_tokens if tenant.id == "batch" else max(1, max_tokens // 4)
            messages = workload_messages(index, "short", 1, 1)
            result = await send_request(
                client,
                url,
                model,
                limit,
                index,
                scheduled,
                messages,
                headers={"Authorization": "Bearer " + credentials[tenant.id]},
            )
            result.tenant = tenant.id
            all_results.append(result)

        arrivals = sorted(
            (i / (rate * (4 if t.id == "batch" else 1)), t.id, i)
            for t in config.tenants
            for i in range(requests)
        )
        by_id = {t.id: t for t in config.tenants}
        try:
            for offset, tenant_id, index in arrivals:
                scheduled = started + offset
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                if len(pending) >= max_outstanding:
                    all_results.append(Result(index, "client_dropped", tenant=tenant_id))
                    continue
                task = asyncio.create_task(one(by_id[tenant_id], index, scheduled))
                pending.add(task)
                task.add_done_callback(pending.discard)
            if pending:
                await asyncio.gather(*pending)
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        # Allow response finalizers to settle before the final operator snapshot.
        await asyncio.sleep(0.1)
        after = (await client.get(url + "/monitor/api")).json()
    elapsed = time.perf_counter() - started
    summaries = {}
    costs = {}
    for t in config.tenants:
        results = [r for r in all_results if r.tenant == t.id]
        summaries[t.id] = summarize(results, elapsed, t.ttft_slo_seconds)
        costs[t.id] = sum(
            r.prompt_tokens + config.decode_cost * r.completion_tokens
            for r in results
            if r.outcome == "success"
            and r.prompt_tokens is not None
            and r.completion_tokens is not None
        )
    total = sum(costs.values())
    report = {
        "schema_version": 1,
        "model": model,
        "scheduler": before["pool"].get("scheduler"),
        "config": {
            "requests_per_tenant": requests,
            "base_rate": rate,
            "batch_rate_multiplier": 4,
            "max_outstanding": max_outstanding,
            "max_tokens": max_tokens,
        },
        "summary": summaries,
        "observed_successful_token_cost": costs,
        "observed_cost_share": {t: c / total if total else None for t, c in costs.items()},
        "note": "Finite, unequal arrivals demonstrate admission and noisy neighbors. "
        "Shares are not a saturated fairness proof or measured GPU time.",
        "before": before,
        "after": after,
        "results": [asdict(r) for r in all_results],
    }
    path = Path(output)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, json.dumps(report, indent=2) + "\n")
    return report
