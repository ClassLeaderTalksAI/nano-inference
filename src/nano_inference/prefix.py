"""Stable token-prefix identity and a small, expiring routing-hint table."""

import asyncio
import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

from .protocol import Message


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class PrefixEncoder:
    def __init__(self, model: str, revision: str, tokens: int = 64, tokenizer=None):
        if not revision or tokens < 1:
            raise ValueError("Prefix identity requires a revision and positive token count")
        if tokenizer is None and model != "nano-model":
            raise ValueError("Synthetic prefix encoding is only valid for nano-model")
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.kind = "huggingface" if tokenizer is not None else "synthetic_bytes"
        if tokenizer is not None:
            if not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str):
                raise ValueError("Prefix routing requires a fast tokenizer with one chat template")
            tokenizer_fingerprint = digest(tokenizer.backend_tokenizer.to_str())
            template = tokenizer.chat_template
        else:
            tokenizer_fingerprint = "utf8-v1"
            template = "nano-json-messages-v1"
        self.identity = {
            "model": model,
            "model_revision": revision,
            "encoding": self.kind,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "chat_template_fingerprint": digest(template),
            "prefix_tokens": tokens,
        }

    @classmethod
    def from_pretrained(cls, model: str, revision: str, tokens: int, path: str):
        # Load a prepared local directory only. Model weights and remote Python code
        # are never loaded by the gateway; preparation/download is a separate step.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        return cls(model, revision, tokens, tokenizer)

    def tokenize(self, messages: list[Message]) -> list[int]:
        data = [m.model_dump() for m in messages]
        if self.tokenizer is None:
            return list(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode())
        return self.tokenizer.apply_chat_template(data, tokenize=True, add_generation_prompt=True)

    def prepare(self, messages: list[Message]) -> tuple[str | None, int]:
        ids = self.tokenize(messages)
        key = digest([self.identity, ids[: self.tokens]]) if len(ids) >= self.tokens else None
        count = len(ids) if self.tokenizer else sum(len(m.content.split()) for m in messages)
        return key, count

    def key(self, messages: list[Message]) -> str | None:
        return self.prepare(messages)[0]

    def routing_features(self, messages: list[Message]):
        """Tokenize once for real arbitrary-length LPM; retain the legacy key."""
        ids = self.tokenize(messages)
        key = digest([self.identity, ids[: self.tokens]]) if len(ids) >= self.tokens else None
        if self.tokenizer is None:
            ids = [
                int(digest([m.role, word])[:32], 16) for m in messages for word in m.content.split()
            ]
        count = len(ids)
        return key, count, tuple(ids)


class PrefixPreparer:
    """Bound tokenization work, including work whose HTTP caller has disconnected.

    A cancelled asyncio caller cannot stop a running tokenizer thread. Shield the
    task and release its permit only after the actual computation completes.
    """

    def __init__(self, encoder: PrefixEncoder, capacity: int = 4):
        if capacity < 1:
            raise ValueError("Tokenization capacity must be positive")
        self.encoder = encoder
        self.permits = asyncio.Semaphore(capacity)
        self.tasks: set[asyncio.Task] = set()

    async def key(self, messages: list[Message]) -> str | None:
        return await self._run(self.encoder.key, messages)

    async def prepare(self, messages: list[Message]) -> tuple[str | None, int]:
        return await self._run(self.encoder.prepare, messages)

    async def routing_features(self, messages: list[Message]):
        return await self._run(self.encoder.routing_features, messages)

    async def _run(self, function, messages):
        from .routing import AdmissionError

        if self.permits.locked():
            raise AdmissionError(429, "prefix_preparation_full")
        await self.permits.acquire()
        task = asyncio.create_task(asyncio.to_thread(function, messages))
        self.tasks.add(task)

        def finished(done):
            self.tasks.discard(done)
            self.permits.release()
            if not done.cancelled():
                done.exception()  # Retrieve errors even if the HTTP caller already left.

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def close(self):
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


@dataclass(frozen=True)
class Hint:
    worker: str
    generation: int
    expires: float


class AffinityTable:
    """LRU bounds memory; TTL is refreshed by successful work, not by lookups."""

    def __init__(self, capacity: int = 1024, ttl: float = 60, clock=time.monotonic):
        if capacity < 1 or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Affinity capacity and TTL must be positive")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self.entries: OrderedDict[str, Hint] = OrderedDict()

    def prune(self):
        now = self.clock()
        for key in [key for key, hint in self.entries.items() if hint.expires <= now]:
            del self.entries[key]

    def get(self, key: str | None) -> Hint | None:
        if key is None:
            return None
        hint = self.entries.get(key)
        if hint is not None:
            if hint.expires <= self.clock():
                del self.entries[key]
                return None
            self.entries.move_to_end(key)
        return hint

    def remember(self, key: str, worker: str, generation: int):
        self.prune()
        existing = self.get(key)
        if existing is not None and (existing.worker, existing.generation) != (worker, generation):
            # A load override may warm a second replica; retain a stable home until
            # invalidation or expiry rather than oscillating on every completion.
            return
        self.entries[key] = Hint(worker, generation, self.clock() + self.ttl)
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def invalidate(self, worker: str):
        for key in [key for key, hint in self.entries.items() if hint.worker == worker]:
            del self.entries[key]
