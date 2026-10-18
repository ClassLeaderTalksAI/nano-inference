"""Deterministic protocol simulator. It does not simulate GPU or KV-cache performance."""

import asyncio
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .protocol import ChatRequest, sse


class Behavior(BaseModel):
    model_config = ConfigDict(extra="forbid")
    healthy: bool = True
    first_content_delay: float = Field(default=0.03, ge=0, le=60)
    chunk_delay: float = Field(default=0.005, ge=0, le=10)
    http_status: int = Field(default=200, ge=200, le=599)
    fail_after_chunks: int | None = Field(default=None, ge=0)


def create_mock_worker(name: str = "mock-a", behavior: Behavior | None = None) -> FastAPI:
    app = FastAPI(title=f"nano-inference simulator: {name}")
    app.state.behavior = behavior or Behavior()
    app.state.active = 0

    @app.get("/health")
    async def health():
        return JSONResponse(
            {"worker": name}, status_code=200 if app.state.behavior.healthy else 503
        )

    @app.get("/admin/config")
    async def get_config():
        return {**app.state.behavior.model_dump(), "active": app.state.active}

    @app.put("/admin/config")
    async def set_config(config: Behavior):
        # Replace the complete config. In-progress requests retain their own snapshot.
        app.state.behavior = config
        return config

    @app.post("/v1/chat/completions")
    async def complete(body: ChatRequest):
        config = app.state.behavior.model_copy()
        status = config.http_status if config.healthy else 503
        if status != 200:
            return JSONResponse({"error": {"message": "injected worker error"}}, status_code=status)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        common = {"id": request_id, "created": int(time.time()), "model": body.model}
        # Each synthetic output item is counted as one simulated token, not a real tokenizer token.
        usage = {
            "prompt_tokens": sum(len(m.content.split()) for m in body.messages),
            "completion_tokens": body.max_tokens,
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        if not body.stream:
            app.state.active += 1
            try:
                await asyncio.sleep(
                    config.first_content_delay + config.chunk_delay * body.max_tokens
                )
                return {
                    **common,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": " token" * body.max_tokens},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": usage,
                }
            finally:
                app.state.active -= 1

        async def generate():
            app.state.active += 1
            try:

                def chunk(delta: dict, finish: str | None = None) -> dict:
                    return {
                        **common,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }

                yield sse(chunk({"role": "assistant", "content": ""}))
                await asyncio.sleep(config.first_content_delay)
                for i in range(body.max_tokens):
                    if config.fail_after_chunks is not None and i >= config.fail_after_chunks:
                        yield sse({"error": {"code": "injected_stream_failure"}})
                        return
                    if i:
                        await asyncio.sleep(config.chunk_delay)
                    yield sse(chunk({"content": " token"}))
                yield sse(chunk({}, "length"))
                if body.stream_options and body.stream_options.include_usage:
                    yield sse({**common, "choices": [], "usage": usage})
                yield sse("[DONE]")
            finally:
                app.state.active -= 1

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app
