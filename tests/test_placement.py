import asyncio
import json
import time
from dataclasses import replace

import pytest

from nano_inference.placement import PlacementConfig, RouteRequest, ServiceModel
from nano_inference.prefix import PrefixEncoder
from nano_inference.protocol import Message
from nano_inference.routing import AdmissionError, WorkerPool
from nano_inference.tenants import TenantPool

from .conftest import eventually
from .test_gateway import payload
from .test_tenants import config, config_file

HOT = RouteRequest(1000, 4, (("prefix-64", 64), ("prefix-960", 960)))


async def pool_for(**options):
    pool = WorkerPool(["http://a", "http://b"], policy="predicted_ttft", **options)
    for w in pool.workers:
        await pool.set_health(w, True)
    return pool


async def test_cache_locality_can_win_but_long_decode_backlog_overrides_it():
    pool = await pool_for(capacity=4)
    seed = await pool.acquire(route_request=HOT)
    await pool.release(seed, successful=True)
    warm = await pool.acquire(route_request=HOT)
    assert warm.worker is seed.worker
    assert warm.prediction.cached_tokens == 960
    assert warm.prediction.ttft_seconds < 0.3
    await pool.release(warm)
    # Same request count, much larger outstanding work: route away from home.
    held = await pool.acquire(route_request=replace(HOT, output_tokens=1000))
    moved = await pool.acquire(route_request=HOT)
    assert moved.worker is not held.worker
    assert pool.placement.last_scores[held.worker.name].queue_seconds > 25
    await pool.release(held)
    await pool.release(moved)
    assert not pool.placement.active


async def test_atomic_predictions_balance_simultaneous_cold_requests():
    pool = await pool_for(capacity=8)
    leases = await asyncio.gather(*(pool.acquire(route_request=HOT) for _ in range(16)))
    assert [w.inflight for w in pool.workers] == [8, 8]
    assert len(pool.placement.active) == 16
    for lease in leases:
        await pool.release(lease)
    await pool.release(leases[0])
    assert not pool.placement.active


async def test_tracks_multiple_replicas_longest_match_and_failed_work_does_not_warm():
    pool = await pool_for(capacity=1)
    first = await pool.acquire(route_request=HOT)
    second = await pool.acquire(route_request=HOT)
    await pool.release(first, successful=True)
    await pool.release(second, successful=True)
    assert len(pool.placement.cache) == 4
    both = [pool.placement.predict(w, HOT).cached_tokens for w in pool.workers]
    assert both == [960, 960]
    partial = replace(HOT, prefixes=(("prefix-64", 64), ("different-960", 960)))
    assert pool.placement.predict(first.worker, partial).cached_tokens == 64
    failed = await pool.acquire(route_request=replace(HOT, prefixes=(("failed", 960),)))
    await pool.release(failed, successful=False)
    assert not any(k[0] == "failed" for k in pool.placement.cache)


async def test_prefix_hints_are_bounded_expire_and_old_generations_cannot_rewarm():
    pool = await pool_for(placement=PlacementConfig(cache_capacity=2, cache_ttl_seconds=3))
    now = [0.0]
    pool.placement.clock = lambda: now[0]
    seed = await pool.acquire(route_request=HOT)
    await pool.release(seed, successful=True)
    other = await pool.acquire(route_request=replace(HOT, prefixes=(("third", 128),)))
    await pool.release(other, successful=True)
    assert len(pool.placement.cache) == 2
    now[0] = 3
    assert pool.placement.predict(seed.worker, HOT).cached_tokens == 0
    assert pool.placement.snapshot()["cache_entries"] == 0
    old = await pool.acquire(route_request=HOT)
    await pool.set_health(old.worker, False)
    await pool.set_health(old.worker, True)
    await pool.release(old, successful=True)
    assert not pool.placement.cache


async def test_native_pressure_changes_choice_and_stale_samples_do_not_look_live():
    pool = await pool_for(placement=PlacementConfig(kv_penalty_seconds=2))
    now = [0.0]
    pool.placement.clock = lambda: now[0]
    seed = await pool.acquire(route_request=HOT)
    await pool.release(seed, successful=True)
    await pool.observe_native(
        seed.worker,
        seed.worker.generation,
        {
            "vllm:num_requests_running": 0,
            "vllm:num_requests_waiting": 0,
            "vllm:kv_cache_usage_perc": 0.99,
        },
    )
    moved = await pool.acquire(route_request=HOT)
    assert moved.worker is not seed.worker
    assert pool.placement.last_scores[seed.worker.name].kv_seconds > 1.8
    await pool.release(moved)
    now[0] = 6
    warm = await pool.acquire(route_request=HOT)
    assert warm.worker is seed.worker
    assert not warm.prediction.native_fresh
    assert warm.prediction.kv_seconds == 0
    assert warm.decision == "predicted_ttft_local"
    await pool.release(warm)


async def test_native_counts_do_not_double_count_own_leases_and_probe_generation_is_guarded():
    pool = await pool_for()
    held = await pool.acquire(route_request=HOT)
    w = held.worker
    before = pool.placement.predict(w, HOT).queue_seconds
    await pool.observe_native(
        w,
        w.generation,
        {
            "vllm:num_requests_running": 1,
            "vllm:num_requests_waiting": 0,
        },
    )
    assert pool.placement.predict(w, HOT).queue_seconds == before
    await pool.release(held)
    assert pool.placement.predict(w, HOT).queue_seconds == 0
    held = await pool.acquire(route_request=HOT)
    w = held.worker
    before = pool.placement.predict(w, HOT).queue_seconds
    await pool.observe_native(
        w,
        w.generation,
        {
            "vllm:num_requests_running": 1,
            "vllm:num_requests_waiting": 4,
        },
    )
    assert pool.placement.predict(w, HOT).queue_seconds == before + 2
    generation = w.generation
    await pool.set_health(w, False)
    await pool.set_health(w, True)
    await pool.observe_native(w, generation, {"vllm:num_requests_waiting": 100})
    assert w.name not in pool.placement.native
    assert pool.placement.predict(w, HOT).queue_seconds == 0
    await pool.release(held, successful=True)


async def test_failed_or_malformed_native_metrics_are_not_zero_observations():
    pool = await pool_for()
    w = pool.workers[0]
    await pool.observe_native(
        w,
        w.generation,
        {
            "vllm:kv_cache_usage_perc": float("nan"),
            "vllm:num_requests_running": -1,
        },
    )
    assert pool.placement.native[w.name][2] == {}
    await pool.observe_native(w, w.generation, None)
    assert not pool.placement.predict(w, HOT).native_fresh


async def test_slo_rejection_is_opt_in_and_never_reserves_a_worker():
    deadline = time.monotonic() + 0.01
    pool = await pool_for()
    accepted = await pool.acquire(route_request=replace(HOT, deadline=deadline))
    assert accepted.prediction.slo_feasible is False
    await pool.release(accepted)
    pool.placement.config.slo_admission = True
    with pytest.raises(AdmissionError, match="slo_budget_exceeded"):
        await pool.acquire(route_request=replace(HOT, deadline=deadline))
    assert not pool.placement.active
    assert sum(w.inflight for w in pool.workers) == 0


async def test_tenant_slo_rejection_refunds_quota_credit_and_continues_other_work(monkeypatch):
    pool = TenantPool(
        ["http://a"],
        policy="predicted_ttft",
        tenants=config(monkeypatch),
        capacity=1,
        clock=lambda: 0,
        placement=PlacementConfig(slo_admission=True),
    )
    await pool.set_health(pool.workers[0], True)
    held = await pool.acquire(tenant="batch", input_tokens=1, max_tokens=1, route_request=HOT)
    state = pool.states["interactive"]
    request = replace(HOT, deadline=time.monotonic() + 0.5)
    waiting = asyncio.create_task(
        pool.acquire(
            tenant="interactive",
            input_tokens=1000,
            max_tokens=4,
            route_request=request,
        )
    )
    other = asyncio.create_task(pool.acquire(tenant="standard", input_tokens=1, max_tokens=1))
    await asyncio.sleep(0.01)
    await pool.release(held, usage={"prompt_tokens": 1, "completion_tokens": 1})
    with pytest.raises(AdmissionError, match="slo_budget_exceeded"):
        await waiting
    assert state.tpm.balance == state.tpm.capacity
    assert state.rpm.balance == state.rpm.capacity - 1
    assert state.deficit == 0 and state.inflight == 0
    assert state.dispatched_cost == 0
    await pool.release(await other)
    assert not pool.waiters and not pool.placement.active


async def test_queued_slo_expiry_and_cancellation_release_all_ledgers(monkeypatch):
    pool = TenantPool(
        ["http://a"],
        policy="predicted_ttft",
        tenants=config(monkeypatch),
        capacity=1,
        clock=lambda: 0,
        placement=PlacementConfig(slo_admission=True),
    )
    await pool.set_health(pool.workers[0], True)
    held = await pool.acquire(tenant="batch", input_tokens=1, max_tokens=1)
    with pytest.raises(AdmissionError, match="slo_budget_exceeded"):
        await pool.acquire(
            tenant="interactive",
            input_tokens=1000,
            max_tokens=4,
            route_request=replace(HOT, deadline=time.monotonic() + 0.03),
        )
    waiting = asyncio.create_task(pool.acquire(tenant="interactive", input_tokens=1, max_tokens=1))
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not pool.states["interactive"].queue
    assert pool.states["interactive"].tpm.balance == pool.states["interactive"].tpm.capacity
    await pool.release(held)
    assert not pool.waiters and not pool.placement.active


async def test_placement_never_bypasses_tenant_allowlist(monkeypatch):
    pool = TenantPool(
        ["http://a", "http://b"],
        policy="predicted_ttft",
        tenants=config(monkeypatch, workers=["worker-2"]),
        placement=PlacementConfig(workers={"worker-2": ServiceModel(prefill_tokens_per_second=1)}),
    )
    for w in pool.workers:
        await pool.set_health(w, True)
    lease = await pool.acquire(
        tenant="interactive", input_tokens=1000, max_tokens=4, route_request=HOT
    )
    assert lease.worker.name == "worker-2"
    assert set(pool.placement.last_scores) == {"worker-2"}
    await pool.release(lease)


def test_multilength_features_distinguish_common_short_prefixes():
    class Tokenizer:
        is_fast = True
        chat_template = "test"

        @property
        def backend_tokenizer(self):
            return self

        def to_str(self):
            return "tokens"

        def apply_chat_template(self, data, **kwargs):
            return [1] * 64 + [ord(data[0]["content"][0])] * 1000

    encoder = PrefixEncoder("test-model", "r1", tokenizer=Tokenizer())
    a = encoder.routing_features([Message(role="user", content="alpha")])
    b = encoder.routing_features([Message(role="user", content="beta")])
    assert a[0] == b[0]  # Previous single-prefix policy sees the same key.
    assert a[1] == 1064
    assert a[2][0] == b[2][0]
    assert a[2][-1] != b[2][-1]
    assert len(a[2]) <= 4


@pytest.mark.parametrize(
    "options",
    [
        {"service": {"prefill_tokens_per_second": 0}},
        {"kv_soft_limit": 1},
        {"cache_reuse_discount": float("nan")},
        {"cache_capacity": 0},
        {"metrics_max_age_seconds": float("inf")},
        {"unknown": True},
    ],
)
def test_invalid_analytical_model_fails_fast(options):
    with pytest.raises(ValueError):
        PlacementConfig(**options)


async def test_http_slo_admission_is_distinct_from_backend_failure(cluster, tmp_path):
    path = tmp_path / "placement.json"
    path.write_text(
        PlacementConfig(
            slo_admission=True,
            default_slo_seconds=0.001,
        ).model_dump_json()
    )
    async with cluster(policy="predicted_ttft", placement_config=str(path)) as (app, c, _):
        response = await c.post("/v1/chat/completions", json=payload())
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "slo_budget_exceeded"
        assert response.headers["retry-after"] == "1"
        assert all(w.healthy and w.inflight == 0 for w in app.state.pool.workers)
        assert not app.state.pool.placement.active
        assert 'reason="slo_budget_exceeded"' in (await c.get("/metrics")).text


async def test_http_features_are_tenant_scoped_and_predictions_are_observable(
    cluster, tmp_path, monkeypatch
):
    cfg = config_file(tmp_path, config(monkeypatch))
    async with cluster(policy="predicted_ttft", tenant_config=cfg) as (app, c, _):
        body = payload()
        body["messages"][0]["content"] = "private-prompt " * 300
        for index, tenant in enumerate(("interactive", "batch", "interactive")):
            response = await c.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer " + tenant + "-test-secret-key-12345"},
            )
            assert response.status_code == 200
            assert float(response.headers["X-Predicted-TTFT-Seconds"]) > 0
            p = app.state.pool.placement.last_scores[response.headers["X-Worker-ID"]]
            assert (p.cached_tokens > 0) == (index == 2)

        async def idle():
            return not app.state.pool.placement.active

        await eventually(idle)
        snapshot = (await c.get("/monitor/api")).json()
        assert snapshot["pool"]["placement"]["model"] == "analytical-ttft-v1"
        assert "private-prompt" not in json.dumps(snapshot)
        metrics = (await c.get("/metrics")).text
        assert "nano_route_prediction_absolute_error_seconds_count" in metrics
        assert "nano_route_predicted_ttft_seconds_count" in metrics
        assert 'nano_route_slo_predictions_total{feasible="true",tenant="interactive"}' in metrics
