from nano_inference.benchmark import run_benchmark

from .test_gateway import configure, idle, payload


def shared(group="a", question="one"):
    return payload(
        messages=[
            {"role": "system", "content": f"Group {group}. " + "Shared document context. " * 20},
            {"role": "user", "content": question},
        ],
        stream=False,
    )


async def test_shared_prefix_affinity_and_different_groups(cluster):
    async with cluster(policy="prefix_affinity") as (_, client, _):
        a = await client.post("/v1/chat/completions", json=shared())
        b = await client.post("/v1/chat/completions", json=shared(question="a new question"))
        other = await client.post("/v1/chat/completions", json=shared(group="b"))
        assert a.headers["x-route-decision"] == "affinity_miss"
        assert b.headers["x-route-decision"] == "affinity_hit"
        assert a.headers["x-worker-id"] == b.headers["x-worker-id"]
        assert other.headers["x-route-decision"] == "affinity_miss"
        state = (await client.get("/workers")).json()
        assert state["affinity"]["entries"] == 2
        assert state["prefix_identity"]["encoding"] == "synthetic_bytes"
        assert "affinity_hit" in (await client.get("/metrics")).text
        assert await idle(client)


async def test_failed_stream_does_not_create_affinity(cluster):
    async with cluster(policy="prefix_affinity") as (_, client, urls):
        await configure(client, urls, fail_after_chunks=1)
        body = shared()
        body["stream"] = True
        response = await client.post("/v1/chat/completions", json=body)
        assert "upstream_stream_error" in response.text
        assert (await client.get("/workers")).json()["affinity"]["entries"] == 0


async def test_recovered_worker_loses_previous_affinity(cluster):
    async with cluster(policy="prefix_affinity") as (app, client, urls):
        first = await client.post("/v1/chat/completions", json=shared())
        worker_index = int(first.headers["x-worker-id"].split("-")[-1]) - 1
        await configure(client, [urls[worker_index]], healthy=False)
        await app.state.probe()
        assert (await client.get("/workers")).json()["affinity"]["entries"] == 0
        second = await client.post("/v1/chat/completions", json=shared())
        assert second.headers["x-worker-id"] != first.headers["x-worker-id"]
        assert second.headers["x-route-decision"] == "affinity_miss"
        await configure(client, [urls[worker_index]], healthy=True)
        await app.state.probe()
        third = await client.post("/v1/chat/completions", json=shared())
        assert third.headers["x-worker-id"] == second.headers["x-worker-id"]
        assert third.headers["x-route-decision"] == "affinity_hit"


async def test_short_prompt_uses_load_routing(cluster):
    async with cluster(policy="prefix_affinity", prefix_tokens=1000) as (_, client, _):
        response = await client.post("/v1/chat/completions", json=payload(stream=False))
        assert response.headers["x-route-decision"] == "prefix_too_short"


async def test_prefix_workload_keeps_groups_local_and_records_decisions(cluster, tmp_path):
    async with cluster(policy="prefix_affinity") as (_, client, _):
        report = await run_benchmark(
            url=str(client.base_url),
            requests=12,
            concurrency=1,
            workload="repeated_prefix",
            prefix_groups=3,
            prefix_repetitions=8,
            max_tokens=2,
            output=str(tmp_path / "prefix.json"),
        )
        assert report["summary"]["outcomes"] == {"success": 12}
        assert [r["route_decision"] for r in report["results"]].count("affinity_miss") == 3
        assert [r["route_decision"] for r in report["results"]].count("affinity_hit") == 9
        for group in range(3):
            assert len({r["worker"] for r in report["results"] if r["index"] % 3 == group}) == 1


async def test_concurrent_header_handoff_never_stalls_the_response(cluster, tmp_path):
    async with cluster(
        policy="prefix_affinity",
        prefix_preparations=16,
        capacity=8,
        max_queue=16,
        request_timeout=2,
    ) as (_, client, urls):
        await configure(client, urls, first_content_delay=0, chunk_delay=0)
        report = await run_benchmark(
            url=str(client.base_url),
            requests=128,
            concurrency=16,
            timeout=3,
            max_tokens=1,
            workload="repeated_prefix",
            prefix_groups=4,
            prefix_repetitions=8,
            output=str(tmp_path / "handoff.json"),
        )
        assert report["summary"]["outcomes"] == {"success": 128}
        assert await idle(client)


async def test_preparation_overload_is_reported_separately_and_recovers(cluster, tmp_path):
    async with cluster(policy="prefix_affinity", prefix_preparations=1) as (app, client, _):
        permits = app.state.preparer.permits
        await permits.acquire()
        try:
            report = await run_benchmark(
                url=str(client.base_url),
                requests=1,
                max_tokens=1,
                output=str(tmp_path / "prep-full.json"),
            )
            assert report["summary"]["outcomes"] == {"rejected": 1}
            assert report["results"][0]["error_code"] == "prefix_preparation_full"
            assert await idle(client)
        finally:
            permits.release()
        assert (await client.post("/v1/chat/completions", json=shared())).status_code == 200
