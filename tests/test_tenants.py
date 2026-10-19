import asyncio
import json
from collections import Counter

import pytest

from nano_inference.routing import AdmissionError
from nano_inference.tenants import TenantConfig, TenantPool

from .conftest import eventually
from .test_gateway import configure, idle, payload


def config(monkeypatch, **overrides):
    tenants = []
    for name, weight in (("interactive", 5), ("standard", 3), ("batch", 1)):
        env = "NANO_" + name.upper() + "_KEY"
        monkeypatch.setenv(env, name + "-test-secret-key-12345")
        tenants.append(
            {
                "id": name,
                "api_key_env": env,
                "weight": weight,
                "rpm": 10000,
                "tpm": 1000000,
                "max_queue": 400,
                "max_inflight": 1,
                **overrides,
            }
        )
    return TenantConfig(tenants=tenants)


def config_file(tmp_path, value):
    path = tmp_path / "tenants.json"
    path.write_text(value.model_dump_json())
    return str(path)


async def pool_for(monkeypatch, **tenant_options):
    pool = TenantPool(
        ["http://a"],
        tenants=config(monkeypatch, **tenant_options),
        capacity=1,
        max_queue=1200,
        queue_timeout=10,
        clock=lambda: 0,
    )
    await pool.set_health(pool.workers[0], True)
    return pool


async def acquire(pool, tenant="interactive", input_tokens=0, max_tokens=16):
    return await pool.acquire(tenant=tenant, input_tokens=input_tokens, max_tokens=max_tokens)


@pytest.mark.parametrize("policy", ["least_inflight", "predicted_ttft", "cache_aware"])
async def test_saturated_wdrr_shares_token_cost_5_3_1_without_starvation(monkeypatch, policy):
    pool = await pool_for(monkeypatch)
    pool.policy = policy
    held = await acquire(pool)
    dispatched = asyncio.Queue()

    async def one(tenant, prompt, output):
        lease = await acquire(pool, tenant, prompt, output)
        await dispatched.put(lease)

    # Different request sizes: fairness must be cost-based, not request counts.
    costs = {"interactive": (0, 16), "standard": (32, 16), "batch": (0, 8)}
    tasks = [asyncio.create_task(one(t, *costs[t])) for t in costs for _ in range(300)]
    await asyncio.sleep(0.02)
    assert len(pool.waiters) == 900
    await pool.release(held)
    totals = Counter()
    try:
        for _ in range(150):
            lease = await asyncio.wait_for(dispatched.get(), 1)
            totals[lease.tenant_id] += lease.estimated_cost
            p, o = costs[lease.tenant_id]
            await pool.release(lease, usage={"prompt_tokens": p, "completion_tokens": o})
        total = sum(totals.values())
        for tenant, expected in (("interactive", 5 / 9), ("standard", 3 / 9), ("batch", 1 / 9)):
            assert abs(totals[tenant] / total - expected) < 0.04
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Tasks that delivered leases have already returned; release the last reservation.
        while not dispatched.empty():
            await pool.release(dispatched.get_nowait())
    assert not pool.waiters
    assert all(s.inflight == 0 for s in pool.states.values())


async def test_tenant_concurrency_block_does_not_hold_other_workers(monkeypatch):
    pool = TenantPool(["http://a", "http://b"], tenants=config(monkeypatch), capacity=1)
    for w in pool.workers:
        await pool.set_health(w, True)
    held = await acquire(pool, "batch")
    queued = asyncio.create_task(acquire(pool, "batch"))
    await asyncio.sleep(0.01)
    other = await asyncio.wait_for(acquire(pool, "interactive"), 0.5)
    assert other.worker is not held.worker
    assert not queued.done()
    await pool.release(held)
    await pool.release(await queued)
    await pool.release(other)


async def test_quotas_settle_actual_usage_and_unknown_usage_remains_charged(monkeypatch):
    pool = await pool_for(monkeypatch, tpm=100, rpm=3)
    first = await acquire(pool, input_tokens=10, max_tokens=40)
    assert pool.states["interactive"].tpm.balance == 50
    await pool.release(first, usage={"prompt_tokens": 10, "completion_tokens": 5})
    assert pool.states["interactive"].tpm.balance == 85
    assert pool.states["interactive"].actual_cost == 20
    second = await acquire(pool, input_tokens=10, max_tokens=40)
    await pool.release(second)
    await pool.release(second)  # No double settlement.
    assert pool.states["interactive"].tpm.balance == 35
    assert pool.states["interactive"].conservative_settlements == 1
    with pytest.raises(AdmissionError, match="tenant_tpm"):
        await acquire(pool, input_tokens=10, max_tokens=40)
    third = await acquire(pool)
    await pool.release(third)
    with pytest.raises(AdmissionError, match="tenant_rpm"):
        await acquire(pool)
    # Another tenant retains an independent token bucket.
    await pool.release(await acquire(pool, "standard"))


async def test_queue_cancel_refunds_unused_tokens_and_never_leaks_tickets(monkeypatch):
    pool = await pool_for(monkeypatch, max_queue=1)
    held = await acquire(pool)
    waiting = asyncio.create_task(acquire(pool))
    await asyncio.sleep(0.01)
    with pytest.raises(AdmissionError, match="tenant_queue_full"):
        await acquire(pool)
    balance = pool.states["interactive"].tpm.balance
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert pool.states["interactive"].tpm.balance == balance + 16
    assert not pool.waiters
    await pool.release(held)


async def test_timeout_health_and_assigned_then_cancelled_ticket_cleanup(monkeypatch):
    pool = await pool_for(monkeypatch)
    pool.queue_timeout = 0.02
    held = await acquire(pool)
    with pytest.raises(AdmissionError, match="queue_timeout"):
        await acquire(pool)
    assert not pool.waiters
    waiting = asyncio.create_task(acquire(pool, "standard"))
    await asyncio.sleep(0.005)
    # release reserves a slot for the waiter synchronously; cancel before it resumes.
    await pool.release(held)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert pool.workers[0].inflight == 0
    assert all(s.inflight == 0 for s in pool.states.values())
    assert not pool.waiters
    held = await acquire(pool)
    waiting = asyncio.create_task(acquire(pool))
    await asyncio.sleep(0.005)
    await pool.set_health(pool.workers[0], False)
    with pytest.raises(AdmissionError, match="no_healthy_workers"):
        await waiting
    await pool.release(held)


async def test_authentication_scoping_and_operator_metrics(cluster, tmp_path, monkeypatch):
    path = config_file(tmp_path, config(monkeypatch))
    async with cluster(tenant_config=path, policy="prefix_affinity", prefix_tokens=2) as (
        app,
        c,
        _,
    ):
        assert (await c.post("/v1/chat/completions", json=payload())).status_code == 401
        key = "interactive-test-secret-key-12345"
        headers = {"Authorization": "Bearer " + key, "X-Tenant-ID": "batch"}
        assert (
            await c.post("/v1/chat/completions", headers=headers, json=payload(cache_salt="spoof"))
        ).status_code == 422
        assert (
            await c.post("/v1/chat/completions", headers=headers, json=payload(tenant_id="batch"))
        ).status_code == 422
        first = await c.post("/v1/chat/completions", headers=headers, json=payload())
        assert first.status_code == 200
        assert first.headers["x-route-decision"] == "affinity_miss"
        again = await c.post("/v1/chat/completions", headers=headers, json=payload())
        assert again.headers["x-route-decision"] == "affinity_hit"
        other = await c.post(
            "/v1/chat/completions",
            json=payload(),
            headers={"Authorization": "Bearer batch-test-secret-key-12345"},
        )
        assert other.headers["x-route-decision"] == "affinity_miss"
        assert app.state.pool.salt("batch") != app.state.pool.salt("interactive")
        state = (await c.get("/monitor/api")).json()
        assert state["metrics"]["tenants"]["interactive"]["outcomes"]["success"] == 2
        assert state["metrics"]["tenants"]["batch"]["outcomes"]["success"] == 1
        assert state["metrics"]["tenants"]["interactive"]["ttft"]["mean"] > 0
        assert state["metrics"]["tenants"]["interactive"]["slo_met"] == 2
        metrics = (await c.get("/metrics")).text
        for secret in (key, app.state.pool.salt("batch")):
            assert secret not in metrics + json.dumps(state)
        assert 'nano_tenant_tokens_total{kind="output",tenant="interactive"} 16.0' in metrics
        assert "nano_tenant_queued" in metrics
        assert "nano_engine_metrics_up" in metrics
        assert await idle(c)


async def test_http_disconnect_while_queued_and_streaming_settles_tenant(
    cluster, tmp_path, monkeypatch
):
    path = config_file(tmp_path, config(monkeypatch, max_inflight=1))
    async with cluster(tenant_config=path, capacity=1) as (app, c, urls):
        await configure(c, urls, first_content_delay=0.05, chunk_delay=0.03)
        h = {"Authorization": "Bearer interactive-test-secret-key-12345"}
        async with c.stream(
            "POST", "/v1/chat/completions", headers=h, json=payload(max_tokens=100)
        ) as response:
            assert response.status_code == 200
            task = asyncio.create_task(c.post("/v1/chat/completions", headers=h, json=payload()))

            async def queued():
                return len(app.state.pool.waiters) == 1

            await eventually(queued)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            async for line in response.aiter_lines():
                if '"content":" token"' in line:
                    break
        await eventually(lambda: idle(c))
        s = app.state.pool.states["interactive"]
        assert s.inflight == 0 and not s.queue
        assert s.conservative_settlements == 1


async def test_worker_allowlist_overrides_affinity_and_other_tenant_failure(monkeypatch):
    cfg = config(monkeypatch, workers=["worker-2"])
    pool = TenantPool(["http://a", "http://b"], tenants=cfg, policy="prefix_affinity")
    for w in pool.workers:
        await pool.set_health(w, True)
    pool.affinity.remember("hot", "worker-1", pool.workers[0].generation)
    lease = await pool.acquire("hot", tenant="interactive", input_tokens=5, max_tokens=5)
    assert lease.worker.name == "worker-2"
    await pool.release(lease)
    await pool.set_health(pool.workers[1], False)
    with pytest.raises(AdmissionError, match="no_healthy_workers"):
        await acquire(pool)


def test_config_fails_closed_for_missing_duplicate_keys_and_unknown_workers(monkeypatch):
    cfg = config(monkeypatch)
    monkeypatch.delenv("NANO_BATCH_KEY")
    with pytest.raises(ValueError, match="NANO_BATCH_KEY"):
        TenantPool(["http://a"], tenants=cfg)
    monkeypatch.setenv("NANO_BATCH_KEY", "interactive-test-secret-key-12345")
    with pytest.raises(ValueError, match="distinct"):
        TenantPool(["http://a"], tenants=cfg)
    with pytest.raises(ValueError, match="Unknown worker"):
        TenantPool(["http://a"], tenants=config(monkeypatch, workers=["worker-9"]))


async def test_actual_decode_overrun_and_missing_usage_charge_future_credit(monkeypatch):
    pool = await pool_for(monkeypatch, expected_output_tokens=1)
    lease = await acquire(pool, input_tokens=2, max_tokens=10)
    assert lease.estimated_cost == 4
    await pool.release(lease, usage={"prompt_tokens": 2, "completion_tokens": 10})
    assert pool.states["interactive"].deficit == -18
    unknown = await acquire(pool, input_tokens=2, max_tokens=10)
    await pool.release(unknown)
    assert pool.states["interactive"].deficit <= -18
    assert pool.states["interactive"].conservative_cost == 22


async def test_refill_does_not_create_idle_fairness_credit(monkeypatch):
    now = [0.0]
    pool = TenantPool(
        ["http://a"], tenants=config(monkeypatch, rpm=1, tpm=50), clock=lambda: now[0]
    )
    await pool.set_health(pool.workers[0], True)
    lease = await acquire(pool, max_tokens=20)
    await pool.release(lease)
    with pytest.raises(AdmissionError, match="tenant_rpm"):
        await acquire(pool)
    now[0] = 60
    state = pool.snapshot()["tenants"][0]
    assert state["rpm_remaining"] == 1
    assert state["tpm_remaining"] == 50
    assert state["deficit"] <= 0
    lease = await acquire(pool)
    await pool.release(lease)


async def test_fifo_baseline_is_arrival_order_among_eligible_tenants(monkeypatch):
    cfg = config(monkeypatch).model_copy(update={"scheduler": "fifo"})
    pool = TenantPool(["http://a"], tenants=cfg, capacity=1)
    await pool.set_health(pool.workers[0], True)
    held = await acquire(pool)
    batch = asyncio.create_task(acquire(pool, "batch"))
    await asyncio.sleep(0.01)
    interactive = asyncio.create_task(acquire(pool, "interactive"))
    await asyncio.sleep(0.01)
    await pool.release(held)
    first = await batch
    assert not interactive.done()
    await pool.release(first)
    await pool.release(await interactive)


async def test_zero_queue_tenant_can_use_idle_slot_despite_another_blocked_queue(monkeypatch):
    cfg = config(monkeypatch, max_queue=0)
    cfg.tenants[2].max_queue = 1
    pool = TenantPool(["http://a", "http://b"], tenants=cfg, capacity=1)
    for worker in pool.workers:
        await pool.set_health(worker, True)
    held = await acquire(pool, "batch")
    queued = asyncio.create_task(acquire(pool, "batch"))
    await asyncio.sleep(0.01)
    interactive = await acquire(pool)
    assert interactive.worker is not held.worker
    await pool.release(interactive)
    await pool.release(held)
    await pool.release(await queued)
