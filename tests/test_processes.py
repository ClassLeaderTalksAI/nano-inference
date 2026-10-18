"""Exercise actual CLI subprocesses and an abrupt worker process death."""

import asyncio
import sys

import httpx

from nano_inference.protocol import sse_data

from .conftest import eventually
from .test_gateway import payload


async def test_cli_worker_crash_interrupts_stream_and_restart_recovers(unused_tcp_port_factory):
    ports = [unused_tcp_port_factory() for _ in range(3)]
    urls = [f"http://127.0.0.1:{p}" for p in ports]
    children = []

    async def start(*args):
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "nano_inference",
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        children.append(child)
        return child

    try:
        a = await start("worker", "--port", str(ports[0]), "--name", "a", "--chunk-delay", ".05")
        await start("worker", "--port", str(ports[1]), "--name", "b")
        await start(
            "gateway",
            "--port",
            str(ports[2]),
            "--worker-urls",
            *urls[:2],
            "--health-interval",
            ".05",
            "--policy",
            "round_robin",
        )
        async with httpx.AsyncClient(trust_env=False, timeout=3) as client:

            async def both_healthy():
                try:
                    response = await client.get(urls[2] + "/workers")
                    return all(w["healthy"] for w in response.json()["workers"])
                except httpx.HTTPError:
                    return False

            await eventually(both_healthy, seconds=10)
            frames = []
            async with client.stream(
                "POST", urls[2] + "/v1/chat/completions", json=payload(max_tokens=500)
            ) as response:
                assert response.headers["x-worker-id"] == "worker-1"
                async for data in sse_data(response.aiter_lines()):
                    frames.append(data)
                    if '"content":" token"' in data and a.returncode is None:
                        a.kill()
                        await a.wait()
            assert any("upstream_stream_error" in frame for frame in frames)
            assert "[DONE]" not in frames

            async def a_removed():
                state = (await client.get(urls[2] + "/workers")).json()
                return not state["workers"][0]["healthy"] and all(
                    w["inflight"] == 0 for w in state["workers"]
                )

            await eventually(a_removed)
            response = await client.post(
                urls[2] + "/v1/chat/completions", json=payload(stream=False)
            )
            assert response.status_code == 200
            assert response.headers["x-worker-id"] == "worker-2"
            await start("worker", "--port", str(ports[0]), "--name", "a-restarted")
            await eventually(both_healthy, seconds=10)
            chosen = set()
            for _ in range(2):
                response = await client.post(
                    urls[2] + "/v1/chat/completions", json=payload(stream=False)
                )
                assert response.status_code == 200
                chosen.add(response.headers["x-worker-id"])
            assert chosen == {"worker-1", "worker-2"}
    finally:
        for child in children:
            if child.returncode is None:
                child.terminate()
        for child in children:
            try:
                await asyncio.wait_for(child.wait(), 5)
            except TimeoutError:
                child.kill()
                await child.wait()
