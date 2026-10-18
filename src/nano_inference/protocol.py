"""The deliberately small chat API shared by the gateway and simulated workers."""

import json
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=200_000)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1, max_length=128)
    stream: bool = True
    max_tokens: int = Field(default=32, ge=1, le=4096)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int | None = None
    stream_options: StreamOptions | None = None


def sse(data: dict | str) -> bytes:
    value = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return f"data: {value}\n\n".encode()


async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Parse events across arbitrary HTTP chunks; comments and non-data fields are ignored."""
    parts: list[str] = []
    async for line in lines:
        if not line:
            if parts:
                yield "\n".join(parts)
                parts.clear()
        elif line.startswith("data:"):
            value = line[5:]
            parts.append(value[1:] if value.startswith(" ") else value)
    # A truncated frame is deliberately not emitted at EOF.


def has_content(event: dict) -> bool:
    return any(choice.get("delta", {}).get("content") for choice in event.get("choices", []))
