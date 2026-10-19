import asyncio

import pytest

from nano_inference.placement import PlacementConfig, RouteRequest
from nano_inference.routing import WorkerPool
from nano_inference.tenants import TenantPool

from .test_tenants import config

PROMPT = RouteRequest(150, 4, tuple(range(150)), "tenant-a:model")


async def pool_for(**options):
    pool = WorkerPool(["http://a", "http://b"], policy="cache_aware", **options)
    for worker in pool.workers:
        await pool.set_health(worker, True)
    return pool


async def test_arbitrary_lpm_drives_affinity_and_weak_matches_fall_back():
    pool = await pool_for()
    seed = await pool.acquire(route_request=PROMPT)
    assert seed.decision == "cache_miss"
    await pool.release(seed, successful=True)
    # Match ends inside the original compressed edge, at neither 64 nor 256.
    request = RouteRequest(150, 4, tuple(range(93)) + (999,) * 57, PROMPT.namespace)
    warm = await pool.acquire(route_request=request)
    assert warm.worker is seed.worker and warm.decision == "cache_affinity"
    assert warm.prediction.matched_tokens == warm.prediction.cached_tokens == 93
    await pool.release(warm)  # Failed work must not extend the index.
    weak = RouteRequest(150, 4, tuple(range(50)) + (888,) * 100, PROMPT.namespace)
    cold = await pool.acquire(route_request=weak)
    assert cold.worker is not seed.worker and cold.decision == "cache_weak_match"
    await pool.release(cold)


async def test_load_imbalance_overrides_hot_prefix_and_success_warms_second_replica():
    pool = await pool_for(capacity=8)
    seed = await pool.acquire(route_request=PROMPT)
    await pool.release(seed, successful=True)
    held = [await pool.acquire(route_request=PROMPT) for _ in range(3)]
    assert all(lease.worker is seed.worker for lease in held)
    moved = await pool.acquire(route_request=PROMPT)
    assert moved.worker is not seed.worker and moved.decision == "cache_load_fallback"
    await pool.release(moved, successful=True)
    assert (
        pool.placement.index.match(
            PROMPT.namespace, PROMPT.token_ids, moved.worker.name, moved.generation
        )
        == 150
    )
    for lease in held:
        await pool.release(lease)
    assert not pool.placement.active


@pytest.mark.parametrize(
    "loads,absolute,relative,fallback",
    [
        ((2, 0), 2, 1.5, False),  # Equal to absolute threshold is not above it.
        ((3, 2), 0, 2, False),  # Absolute alone cannot trigger fallback.
        ((4, 2), 1, 2, False),  # Equal to relative threshold is not above it.
        ((4, 1), 2, 2, True),
        ((0, 0), 0, 1, False),
    ],
)
async def test_both_balance_thresholds_are_required(loads, absolute, relative, fallback):
    pool = await pool_for(
        placement=PlacementConfig(balance_abs_threshold=absolute, balance_rel_threshold=relative)
    )
    home = pool.workers[0]
    pool.placement.index.remember(PROMPT.namespace, PROMPT.token_ids, home.name, home.generation)
    for worker, load in zip(pool.workers, loads, strict=True):
        worker.inflight = load  # Pure policy decision test, not a lease lifecycle.
    chosen = pool.placement.select_cache_aware(PROMPT, pool.workers)
    assert (pool.placement.last_reason == "cache_load_fallback") is fallback
    assert chosen is (pool.workers[1] if fallback else home)


async def test_fresh_native_queue_triggers_fallback_but_stale_queue_does_not():
    pool = await pool_for()
    now = [0.0]
    pool.placement.clock = lambda: now[0]
    seed = await pool.acquire(route_request=PROMPT)
    await pool.release(seed, successful=True)
    await pool.observe_native(seed.worker, seed.generation, {"vllm:num_requests_waiting": 10})
    moved = await pool.acquire(route_request=PROMPT)
    assert moved.worker is not seed.worker and moved.decision == "cache_load_fallback"
    await pool.release(moved)
    now[0] = 6
    warm = await pool.acquire(route_request=PROMPT)
    assert warm.worker is seed.worker and warm.decision == "cache_affinity"
    await pool.release(warm)


async def test_capacity_is_a_hard_filter_even_when_balance_threshold_is_disabled():
    pool = await pool_for(capacity=1, placement=PlacementConfig(balance_abs_threshold=1024))
    seed = await pool.acquire(route_request=PROMPT)
    await pool.release(seed, successful=True)
    home = await pool.acquire(route_request=PROMPT)
    other = await pool.acquire(route_request=PROMPT)
    assert other.worker is not home.worker
    assert home.worker.name not in pool.placement.last_scores
    waiting = asyncio.create_task(pool.acquire(route_request=PROMPT))
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await pool.release(home)
    await pool.release(other)
    assert not pool.waiters and not pool.placement.active


async def test_cache_policy_does_not_bypass_tenant_worker_allowlist(monkeypatch):
    pool = TenantPool(
        ["http://a", "http://b"],
        policy="cache_aware",
        tenants=config(monkeypatch, workers=["worker-2"]),
    )
    for worker in pool.workers:
        await pool.set_health(worker, True)
    home = pool.workers[0]
    pool.placement.index.remember(PROMPT.namespace, PROMPT.token_ids, home.name, home.generation)
    lease = await pool.acquire(
        tenant="interactive", input_tokens=150, max_tokens=4, route_request=PROMPT
    )
    assert lease.worker.name == "worker-2"
    assert set(pool.placement.last_scores) == {"worker-2"}
    await pool.release(lease)


async def test_old_generation_completion_cannot_restore_invalidated_tree():
    pool = await pool_for()
    old = await pool.acquire(route_request=PROMPT)
    await pool.set_health(old.worker, False)
    await pool.set_health(old.worker, True)
    await pool.release(old, successful=True)
    assert not pool.placement.index.records
    fresh = await pool.acquire(route_request=PROMPT)
    await pool.release(fresh, successful=True)
    await pool.set_health(fresh.worker, False)
    assert not pool.placement.index.records


@pytest.mark.parametrize(
    "options",
    [
        {"cache_max_tokens": 0},
        {"cache_max_nodes": 0},
        {"cache_max_prefix_tokens": 0},
        {"cache_threshold": 1.01},
        {"balance_abs_threshold": -1},
        {"balance_rel_threshold": 0.9},
    ],
)
def test_invalid_prefix_policy_configuration(options):
    with pytest.raises(ValueError):
        PlacementConfig(**options)
