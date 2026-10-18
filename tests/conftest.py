import asyncio
import contextlib
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn

from nano_inference.gateway import Settings, create_gateway
from nano_inference.mock_worker import Behavior, create_mock_worker


@asynccontextmanager
async def serve(app):
    """Use actual TCP sockets: in-process ASGI transports buffer streaming bodies."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
    # Do not let embedded test servers install process-wide signal handlers.
    server.capture_signals = contextlib.nullcontext
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Test server failed to start")
                await asyncio.sleep(0.005)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            sock.close()


@pytest.fixture
async def cluster():
    @asynccontextmanager
    async def start(**overrides):
        a = create_mock_worker("a", Behavior(first_content_delay=0.02, chunk_delay=0.002))
        b = create_mock_worker("b", Behavior(first_content_delay=0.02, chunk_delay=0.002))
        async with serve(a) as a_url, serve(b) as b_url:
            options = {
                "worker_urls": [a_url, b_url],
                "health_interval": 0.05,
                "health_timeout": 0.2,
                **overrides,
            }
            app = create_gateway(Settings(**options))
            async with (
                serve(app) as url,
                httpx.AsyncClient(base_url=url, trust_env=False, timeout=5) as client,
            ):
                yield app, client, [a_url, b_url]

    return start


async def eventually(check, seconds=2):
    async with asyncio.timeout(seconds):
        while not await check():  # noqa: ASYNC110 -- Poll an observable HTTP condition in tests.
            await asyncio.sleep(0.01)
