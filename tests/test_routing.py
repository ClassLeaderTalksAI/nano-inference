import asyncio

import pytest

from nano_inference.routing import AdmissionError, WorkerPool


async def healthy_pool(**kwargs):
    pool = WorkerPool(["http://a", "http://b"], **kwargs)
    for w in pool.workers:
        await pool.set_health(w, True)
    return pool


async def test_atomic_reservations_balance_concurrent_requests():
    pool = await healthy_pool(capacity=50)
    leases = await asyncio.gather(*(pool.acquire() for _ in range(100)))
    assert [w.inflight for w in pool.workers] == [50, 50]
    await asyncio.gather(*(pool.release(lease) for lease in leases))
    await pool.release(leases[0])  # Cleanup is idempotent.
    assert [w.inflight for w in pool.workers] == [0, 0]


async def test_round_robin_and_least_inflight_diverge_when_one_worker_is_busy():
    for policy, expected in [("round_robin", "worker-1"), ("least_inflight", "worker-2")]:
        pool = await healthy_pool(policy=policy)
        first = await pool.acquire()
        second = await pool.acquire()
        await pool.release(second)
        third = await pool.acquire()
        assert third.worker.name == expected
        await pool.release(first)
        await pool.release(third)


async def test_fifo_queue_is_bounded_and_cancelled_waiters_are_removed():
    pool = await healthy_pool(capacity=1, max_queue=2)
    leases = [await pool.acquire(), await pool.acquire()]
    first = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)
    second = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)
    with pytest.raises(AdmissionError, match="queue_full"):
        await pool.acquire()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await pool.release(leases[0])
    next_lease = await second
    assert next_lease.worker is leases[0].worker
    assert not pool.waiters
    await pool.release(next_lease)
    await pool.release(leases[1])


async def test_queue_timeout_does_not_leak_ticket_or_capacity():
    pool = await healthy_pool(capacity=1, max_queue=1, queue_timeout=0.02)
    leases = [await pool.acquire(), await pool.acquire()]
    with pytest.raises(AdmissionError, match="queue_timeout"):
        await pool.acquire()
    assert not pool.waiters
    assert sum(w.inflight for w in pool.workers) == 2
    for lease in leases:
        await pool.release(lease)


async def test_fifo_waiter_gets_the_first_freed_slot():
    pool = await healthy_pool(capacity=1)
    leases = [await pool.acquire(), await pool.acquire()]
    first = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)
    second = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)
    await pool.release(leases[0])
    acquired = await first
    assert not second.done()
    await pool.release(acquired)
    await pool.release(await second)
    await pool.release(leases[1])


async def test_health_failure_wakes_waiters_and_stale_probes_cannot_recover_worker():
    pool = await healthy_pool(capacity=1)
    leases = [await pool.acquire(), await pool.acquire()]
    waiting = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.01)
    generation = pool.workers[0].generation
    for w in pool.workers:
        await pool.set_health(w, False)
    await pool.set_health(pool.workers[0], True, expected_generation=generation)
    assert not pool.workers[0].healthy
    with pytest.raises(AdmissionError, match="no_healthy_workers"):
        await waiting
    assert not pool.waiters
    for lease in leases:
        await pool.release(lease)


@pytest.mark.parametrize(
    "options", [{"capacity": 0}, {"max_queue": -1}, {"queue_timeout": 0}, {"policy": "magic"}]
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        WorkerPool(["http://a"], **options)
