import asyncio
import json
import time

import httpx
import pytest
from starlette.requests import ClientDisconnect

from nano_inference.benchmark import run_benchmark
from nano_inference.gateway import ManagedStream, Settings, create_gateway
from nano_inference.protocol import has_content, sse_data

from .conftest import eventually, serve


def payload(**kwargs):
    return {
        "model": "nano-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "stream": True,
        **kwargs,
    }


async def idle(client):
    state = (await client.get("/workers")).json()
    return state["queued"] == 0 and all(w["inflight"] == 0 for w in state["workers"])


async def configure(client, urls, **behavior):
    for url in urls:
        response = await client.put(url + "/admin/config", json=behavior)
        response.raise_for_status()


async def test_real_stream_is_incremental_and_role_delta_is_not_content(cluster):
    async with cluster() as (_, client, urls):
        await configure(client, urls, first_content_delay=0.12, chunk_delay=0.04)
        started = time.perf_counter()
        contents = []
        first_at = None
        async with client.stream("POST", "/v1/chat/completions", json=payload()) as response:
            assert response.status_code == 200
            assert response.headers["x-worker-id"]
            async for data in sse_data(response.aiter_lines()):
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if has_content(event):
                    first_at = first_at or time.perf_counter() - started
                    contents.append(event)
        elapsed = time.perf_counter() - started
        assert len(contents) == 8
        assert first_at >= 0.1
        assert elapsed - first_at >= 0.20  # Full-body buffering would fail this check.
        await eventually(lambda: idle(client))


async def test_non_streaming_validation_and_metrics(cluster):
    async with cluster() as (_, client, _):
        response = await client.post("/v1/chat/completions", json=payload(stream=False))
        assert response.status_code == 200
        assert response.json()["usage"]["completion_tokens"] == 8
        assert (
            await client.post("/v1/chat/completions", json=payload(tools=[]))
        ).status_code == 422
        assert (
            await client.post("/v1/chat/completions", json=payload(model="other"))
        ).status_code == 404
        metrics = (await client.get("/metrics")).text
        assert 'nano_requests_total{outcome="success"} 1.0' in metrics
        assert "nano_worker_inflight" in metrics
        assert await idle(client)


@pytest.mark.parametrize("behavior", [{"fail_after_chunks": 2}, {"http_status": 500}])
async def test_upstream_errors_do_not_look_like_success_or_leak_capacity(cluster, behavior):
    async with cluster() as (_, client, urls):
        await configure(client, urls, **behavior)
        response = await client.post("/v1/chat/completions", json=payload())
        if "http_status" in behavior:
            assert response.status_code == 502
        else:
            assert response.status_code == 200
            assert "upstream_stream_error" in response.text
            assert "[DONE]" not in response.text
        await eventually(lambda: idle(client))


async def test_absolute_stream_deadline_releases_slot(cluster):
    async with cluster(request_timeout=0.1) as (_, client, urls):
        await configure(client, urls, first_content_delay=0.5)
        response = await client.post("/v1/chat/completions", json=payload())
        assert "upstream_stream_error" in response.text
        assert "[DONE]" not in response.text
        await eventually(lambda: idle(client))


async def test_disconnect_mid_stream_releases_upstream_and_reservation(cluster):
    async with cluster(capacity=1) as (_, client, urls):
        await configure(client, urls, first_content_delay=0.01, chunk_delay=0.1)
        async with client.stream("POST", "/v1/chat/completions", json=payload(max_tokens=500)) as r:
            async for data in sse_data(r.aiter_lines()):
                if '"content":" token"' in data:
                    break
        await eventually(lambda: idle(client))

        async def upstream_idle():
            states = await asyncio.gather(*(client.get(url + "/admin/config") for url in urls))
            return all(s.json()["active"] == 0 for s in states)

        await eventually(upstream_idle)


async def test_bounded_admission_and_queued_client_disconnect(cluster):
    async with cluster(capacity=1, max_queue=1, queue_timeout=1) as (app, client, urls):
        await configure(client, urls, first_content_delay=0.5, chunk_delay=0.01)
        first = await app.state.pool.acquire()
        second = await app.state.pool.acquire()
        async with httpx.AsyncClient(
            base_url=str(client.base_url), trust_env=False, timeout=0.08
        ) as impatient:
            queued = asyncio.create_task(impatient.post("/v1/chat/completions", json=payload()))

            async def has_waiter():
                return len(app.state.pool.waiters) == 1

            await eventually(has_waiter)
            rejected = await client.post("/v1/chat/completions", json=payload())
            assert rejected.status_code == 429
            assert rejected.json()["error"]["code"] == "queue_full"
            with pytest.raises(httpx.ReadTimeout):
                await queued

        async def no_waiters():
            return not app.state.pool.waiters

        await eventually(no_waiters)
        await app.state.pool.release(first)
        await app.state.pool.release(second)
        assert await idle(client)


async def test_health_removal_and_recovery(cluster):
    async with cluster() as (app, client, urls):
        await configure(client, [urls[0]], healthy=False)
        await app.state.probe()
        for _ in range(3):
            response = await client.post("/v1/chat/completions", json=payload(stream=False))
            assert response.headers["x-worker-id"] == "worker-2"
        await configure(client, urls, healthy=False)
        await app.state.probe()
        assert (await client.get("/ready")).status_code == 503
        assert (await client.get("/health")).status_code == 200
        assert (await client.post("/v1/chat/completions", json=payload())).status_code == 503
        await configure(client, urls, healthy=True)
        await app.state.probe()
        assert (await client.get("/ready")).status_code == 200


async def test_gateway_refuses_truncated_upstream_stream():
    def respond(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        )

    app = create_gateway(
        Settings(worker_urls=["http://fake"]), transport=httpx.MockTransport(respond)
    )
    async with serve(app) as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        response = await client.post("/v1/chat/completions", json=payload())
        assert "upstream_stream_error" in response.text
        assert "[DONE]" not in response.text
        assert await idle(client)


async def test_benchmark_reports_real_usage_and_all_outcomes(cluster, tmp_path):
    async with cluster() as (_, client, _):
        report = await run_benchmark(
            url=str(client.base_url),
            requests=12,
            concurrency=4,
            max_tokens=5,
            output=str(tmp_path / "run.json"),
        )
        assert report["summary"]["outcomes"] == {"success": 12}
        assert sum(r["completion_tokens"] for r in report["results"]) == 60
        assert sum(r["content_events"] for r in report["results"]) == 60
        assert json.loads((tmp_path / "run.json").read_text())["schema_version"] == 1
        assert await idle(client)


async def test_disconnect_before_upstream_headers_releases_reservation():
    async def respond(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        await asyncio.sleep(0.5)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    app = create_gateway(
        Settings(worker_urls=["http://fake"]), transport=httpx.MockTransport(respond)
    )
    async with serve(app) as url, httpx.AsyncClient(base_url=url, trust_env=False) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.post("/v1/chat/completions", json=payload(), timeout=0.06)
        await eventually(lambda: idle(client))


async def test_cleanup_runs_even_when_stream_body_never_starts():
    cleaned = False

    async def body():
        pytest.fail("The response should fail before starting the body")
        yield b"unreachable"

    async def cleanup():
        nonlocal cleaned
        cleaned = True

    async def send(message):
        raise OSError("caller disappeared before response headers")

    async def receive():
        return {"type": "http.disconnect"}

    response = ManagedStream(body(), cleanup)
    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert cleaned


async def test_usage_is_only_exposed_when_requested(cluster):
    async with cluster() as (_, client, _):
        regular = await client.post("/v1/chat/completions", json=payload())
        requested = await client.post(
            "/v1/chat/completions", json=payload(stream_options={"include_usage": True})
        )
        assert '"usage"' not in regular.text
        assert '"completion_tokens":8' in requested.text


async def test_open_loop_reports_client_drops_instead_of_hiding_them(cluster, tmp_path):
    async with cluster() as (_, client, urls):
        await configure(client, urls, first_content_delay=0.1, chunk_delay=0.001)
        report = await run_benchmark(
            url=str(client.base_url),
            requests=20,
            rate=1000,
            max_outstanding=1,
            max_tokens=2,
            output=str(tmp_path / "open.json"),
        )
        assert report["summary"]["requests"] == 20
        assert report["summary"]["outcomes"]["client_dropped"] > 0
        assert report["summary"]["outcomes"]["success"] >= 1
