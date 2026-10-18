"""Own one embedded localhost HTTP server for controlled experiments."""

import asyncio
import contextlib
import socket
from contextlib import asynccontextmanager

import uvicorn


@asynccontextmanager
async def serve_app(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
    server.capture_signals = contextlib.nullcontext
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(30):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Experiment server exited during startup")
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 5)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            sock.close()
