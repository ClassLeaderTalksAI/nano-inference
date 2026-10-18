"""Small workload runner. Client content events and backend token counts stay distinct."""

import asyncio
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .protocol import has_content, sse_data


@dataclass
class Result:
    index: int
    outcome: str
    http_status: int | None = None
    worker: str | None = None
    latency_seconds: float = 0
    dispatch_lag_seconds: float = 0
    first_content_seconds: float | None = None
    completion_tokens: int | None = None
    content_events: int = 0
    route_decision: str | None = None
    prompt_tokens: int | None = None
    error_code: str | None = None
    tenant: str | None = None


def workload_messages(index: int, workload: str, groups: int, repetitions: int) -> list[dict]:
    if workload in ("repeated_prefix", "single_prefix"):
        group = index % groups if workload == "repeated_prefix" else 0
        return [
            {
                "role": "system",
                "content": f"Document group {group}. "
                + "Requests enter a queue and workers consume them. " * repetitions,
            },
            {"role": "user", "content": f"Summarize this document. Question {index}."},
        ]
    if workload == "mixed":
        return [
            {
                "role": "user",
                "content": f"Request {index}. "
                + "Explain a distributed queue. " * (1, 32, 128)[index % 3],
            }
        ]
    return [{"role": "user", "content": f"Explain a queue in one paragraph. ID {index}"}]


async def send_request(
    client,
    url: str,
    model: str,
    max_tokens: int,
    index: int,
    scheduled: float,
    messages: list[dict] | None = None,
    headers: dict | None = None,
):
    dispatched = time.perf_counter()
    result = Result(index, "client_error", dispatch_lag_seconds=dispatched - scheduled)
    try:
        async with client.stream(
            "POST",
            url.rstrip("/") + "/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages or workload_messages(index, "short", 1, 1),
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            result.http_status = response.status_code
            result.worker = response.headers.get("x-worker-id")
            result.route_decision = response.headers.get("x-route-decision")
            if response.status_code != 200:
                result.outcome = "rejected" if response.status_code == 429 else "http_error"
                try:
                    result.error_code = (
                        json.loads(await response.aread()).get("error", {}).get("code")
                    )
                except (ValueError, AttributeError):
                    pass
                return result
            result.outcome = "incomplete_stream"
            finished = False
            async for data in sse_data(response.aiter_lines()):
                if data == "[DONE]":
                    if finished and result.content_events:
                        result.outcome = "success"
                    break
                event = json.loads(data)
                if "error" in event:
                    result.outcome = "stream_error"
                    result.error_code = event["error"].get("code")
                    break
                if has_content(event):
                    result.content_events += 1
                    if result.first_content_seconds is None:
                        result.first_content_seconds = time.perf_counter() - scheduled
                finished |= any(
                    c.get("finish_reason") is not None for c in event.get("choices", [])
                )
                tokens = (event.get("usage") or {}).get("completion_tokens")
                if type(tokens) is int and tokens >= 0:
                    result.completion_tokens = tokens
                prompt_tokens = (event.get("usage") or {}).get("prompt_tokens")
                if type(prompt_tokens) is int and prompt_tokens >= 0:
                    result.prompt_tokens = prompt_tokens
    except (httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError):
        result.outcome = "client_error"
    finally:
        result.latency_seconds = time.perf_counter() - scheduled
    return result


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(results: list[Result], elapsed: float, slo: float) -> dict:
    successes = [r for r in results if r.outcome == "success"]
    token_counts = [r.completion_tokens for r in successes]
    complete_usage = bool(successes) and all(t is not None for t in token_counts)
    firsts = [r.first_content_seconds for r in successes if r.first_content_seconds is not None]
    return {
        "requests": len(results),
        "outcomes": {
            key: sum(r.outcome == key for r in results)
            for key in sorted({r.outcome for r in results})
        },
        "elapsed_seconds": elapsed,
        "successful_requests_per_second": len(successes) / elapsed,
        "output_tokens_per_second": sum(token_counts) / elapsed if complete_usage else None,
        "successful_requests_meeting_ttft_slo_per_second": sum(t <= slo for t in firsts) / elapsed,
        "ttft_slo_seconds": slo,
        "successful_first_content_seconds": {
            "p50": percentile(firsts, 0.50),
            "p95": percentile(firsts, 0.95),
        },
        "successful_latency_seconds": {
            "p50": percentile([r.latency_seconds for r in successes], 0.50),
            "p95": percentile([r.latency_seconds for r in successes], 0.95),
        },
        "max_dispatch_lag_seconds": max((r.dispatch_lag_seconds for r in results), default=0),
        "note": "First content is a client observation; content events are not token counts.",
    }


async def run_benchmark(
    url: str = "http://127.0.0.1:8000",
    model: str = "nano-model",
    requests: int = 50,
    concurrency: int = 4,
    rate: float | None = None,
    max_outstanding: int = 128,
    max_tokens: int = 32,
    timeout: float = 60,  # noqa: ASYNC109 -- This is a workload setting, enforced with asyncio.timeout.
    slo: float = 2,
    output: str = "results/latest.json",
    label: str = "simulated",
    workload: str = "short",
    prefix_groups: int = 8,
    prefix_repetitions: int = 128,
) -> dict:
    if min(requests, concurrency, max_outstanding, max_tokens, timeout, slo) <= 0 or any(
        not math.isfinite(t) for t in (timeout, slo)
    ):
        raise ValueError("Counts and timeouts must be positive")
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError("Rate must be finite and positive")
    if workload not in ("short", "mixed", "repeated_prefix", "single_prefix"):
        raise ValueError("Unknown workload")
    if prefix_groups < 1 or not 1 <= prefix_repetitions <= 2048:
        raise ValueError("Prefix groups must be positive and repetitions must be 1–2048")
    results: list[Result] = []
    timestamp = datetime.now(UTC).isoformat()
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_connections=None),
        trust_env=False,
    ) as client:

        async def gateway_snapshot():
            try:
                response = await client.get(url.rstrip("/") + "/workers", timeout=2)
                return response.json() if response.status_code == 200 else None
            except (httpx.HTTPError, ValueError):
                return None

        before = await gateway_snapshot()
        started = time.perf_counter()

        async def one(index, scheduled):
            async with asyncio.timeout(timeout):
                output_limit = max_tokens
                if workload == "mixed":
                    output_limit = max(1, max_tokens // (4, 2, 1)[index % 3])
                messages = workload_messages(index, workload, prefix_groups, prefix_repetitions)
                return await send_request(
                    client, url, model, output_limit, index, scheduled, messages
                )

        async def collect(index, scheduled):
            try:
                result = await one(index, scheduled)
            except TimeoutError:
                result = Result(
                    index, "client_timeout", latency_seconds=time.perf_counter() - scheduled
                )
            results.append(result)

        if rate is None:
            indices = iter(range(requests))

            async def loop():
                for index in indices:
                    await collect(index, time.perf_counter())

            await asyncio.gather(*(loop() for _ in range(min(concurrency, requests))))
        else:
            # Open loop: arrivals do not wait for responses. Bound client-side outstanding
            # work explicitly and report every dropped arrival instead of hiding overload.
            pending: set[asyncio.Task] = set()
            try:
                for index in range(requests):
                    scheduled = started + index / rate
                    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                    if len(pending) >= max_outstanding:
                        results.append(
                            Result(
                                index,
                                "client_dropped",
                                dispatch_lag_seconds=time.perf_counter() - scheduled,
                            )
                        )
                        continue
                    task = asyncio.create_task(collect(index, scheduled))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                if pending:
                    await asyncio.gather(*pending)
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
        elapsed = time.perf_counter() - started
        after = await gateway_snapshot()
    report = {
        "schema_version": 1,
        "label": label,
        "started_at_utc": timestamp,
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "gateway_before": before,
        "gateway_after": after,
        "config": {
            "url": url,
            "model": model,
            "requests": requests,
            "concurrency": concurrency,
            "rate": rate,
            "max_outstanding": max_outstanding,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "slo": slo,
            "workload": workload,
            "prefix_groups": prefix_groups,
            "prefix_repetitions": prefix_repetitions,
        },
        "summary": summarize(results, elapsed, slo),
        "results": [asdict(r) for r in sorted(results, key=lambda r: r.index)],
    }
    path = Path(output)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, json.dumps(report, indent=2) + "\n")
    return report
