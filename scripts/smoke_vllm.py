"""Exercise the common platform against a prepared vLLM flavor, then stop it.

Mac: uv run --extra tokenizer python scripts/smoke_vllm.py
The same HTTP checks can later be run inside the approved Linux/CUDA host.
"""

import argparse
import asyncio
import json
import platform
import signal
import socket
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

from nano_inference.benchmark import run_benchmark
from nano_inference.engine_metrics import cache_delta, engine_snapshot
from nano_inference.flavors import FlavorCluster, FlavorProfile, free_port, inspect_runtime
from nano_inference.prefix import PrefixEncoder
from nano_inference.protocol import Message, has_content, sse_data

ROOT = Path(__file__).resolve().parents[1]


async def wait_until(check, seconds=30):
    async with asyncio.timeout(seconds):
        while not await check():  # noqa: ASYNC110 -- Observe independent HTTP services.
            await asyncio.sleep(0.1)


def messages(tag: str, question="Name two colors."):
    return [
        {
            "role": "system",
            "content": f"Experiment {tag}. Be concise. "
            + "A queue stores requests until a worker is ready to process them. " * 12,
        },
        {"role": "user", "content": question},
    ]


async def checks(cluster: FlavorCluster, output: Path):
    profile = cluster.profile
    encoder = PrefixEncoder.from_pretrained(
        profile.model_alias, profile.model_revision, 64, str(ROOT / profile.model_path)
    )
    observations = {}
    tag = uuid.uuid4().hex
    base = {
        "model": profile.model_alias,
        "messages": messages(tag),
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    async with httpx.AsyncClient(trust_env=False, timeout=65) as client:
        parity = []
        for url in cluster.worker_urls:
            version = await client.get(url + "/version")
            version.raise_for_status()
            # The public endpoint reports the engine release without the +cpu
            # packaging suffix of the macOS core wheel. Metal owns execution.
            if (
                version.json()["version"].split("+", 1)[0]
                != profile.runtime_version.split("+", 1)[0]
            ):
                raise ValueError("Serving runtime does not match the pinned profile")
            for data in [
                base["messages"],
                [
                    {"role": "user", "content": "你好，queue 是什么？"},
                    {"role": "assistant", "content": "A queue holds work."},
                    {"role": "user", "content": "Name one use."},
                ],
            ]:
                tokenized = await client.post(
                    url + "/tokenize",
                    json={
                        "model": profile.model_alias,
                        "messages": data,
                        "add_generation_prompt": True,
                    },
                )
                tokenized.raise_for_status()
                expected = encoder.tokenize([Message(**m) for m in data])
                assert tokenized.json()["tokens"] == expected, "Backend/tokenizer mismatch"
                parity.append(
                    {"worker": url.rsplit(":", 1)[-1], "tokens": len(expected), "exact_match": True}
                )
        observations["tokenization"] = parity
        initial = [await engine_snapshot(client, url) for url in cluster.worker_urls]
        first = await client.post(cluster.url + "/v1/chat/completions", json=base)
        first.raise_for_status()
        assert first.json()["choices"][0]["message"]["content"]
        home = int(first.headers["x-worker-id"].split("-")[-1]) - 1
        observations["non_streaming"] = {
            "usage": first.json()["usage"],
            "worker": first.headers["x-worker-id"],
            "content": first.json()["choices"][0]["message"]["content"],
        }
        done, content, usage = False, [], None
        async with client.stream(
            "POST",
            cluster.url + "/v1/chat/completions",
            json={
                **base,
                "messages": messages(tag, "Name two animals."),
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            response.raise_for_status()
            assert response.headers["x-route-decision"] == "affinity_hit"
            assert response.headers["x-worker-id"] == first.headers["x-worker-id"]
            async for data in sse_data(response.aiter_lines()):
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                assert "error" not in event
                if has_content(event):
                    content.extend(c["delta"].get("content", "") for c in event["choices"])
                usage = event.get("usage") or usage
        assert done and content and usage["completion_tokens"] > 0
        observations["streaming"] = {"done": True, "content": "".join(content), "usage": usage}

        async def cache_observed():
            now = await engine_snapshot(client, cluster.worker_urls[home])
            return cache_delta(initial[home], now)["hit_tokens"] > 0

        await wait_until(cache_observed)
        after_cache = [await engine_snapshot(client, url) for url in cluster.worker_urls]
        observations["automatic_prefix_cache"] = {
            "before": initial,
            "after": after_cache,
            "home_delta": cache_delta(initial[home], after_cache[home]),
            "scope": "Token counters within one engine lifetime; distinct from router affinity.",
        }

        async def idle():
            state = (await client.get(cluster.url + "/workers")).json()
            engines = [await engine_snapshot(client, url) for url in cluster.worker_urls]
            return (
                state["queued"] == 0
                and all(w["inflight"] == 0 for w in state["workers"])
                and all(
                    e["values"]["vllm:num_requests_running"] == 0
                    and e["values"]["vllm:num_requests_waiting"] == 0
                    for e in engines
                )
            )

        before_abort = [await engine_snapshot(client, url) for url in cluster.worker_urls]
        async with client.stream(
            "POST",
            cluster.url + "/v1/chat/completions",
            json={
                **base,
                "messages": messages(tag, "List the integers from 1 to 500, one per line."),
                "max_tokens": 512,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            cancelled_worker = int(response.headers["x-worker-id"].split("-")[-1]) - 1
            saw_content = False
            async for data in sse_data(response.aiter_lines()):
                assert data != "[DONE]", "Generation ended before the cancellation check"
                if has_content(json.loads(data)):
                    saw_content = True
                    break
            assert saw_content

            active_snapshot = None

            async def active_observed():
                nonlocal active_snapshot
                active_snapshot = await engine_snapshot(
                    client, cluster.worker_urls[cancelled_worker]
                )
                return active_snapshot["values"]["vllm:num_requests_running"] > 0

            await wait_until(active_observed, seconds=10)
            cancel_started = time.perf_counter()

        await wait_until(idle)
        after_abort = await engine_snapshot(client, cluster.worker_urls[cancelled_worker])
        for reason in ("stop", "length"):
            assert after_abort["finished_by_reason"].get(reason, 0) == before_abort[
                cancelled_worker
            ]["finished_by_reason"].get(reason, 0)
        observations["cancellation"] = {
            "engine_was_observed_running": True,
            "gateway_and_engines_idle": True,
            "successful_finish_counters_unchanged": True,
            "observed_cleanup_seconds": time.perf_counter() - cancel_started,
            "before": before_abort[cancelled_worker],
            "during": active_snapshot,
            "after": after_abort,
            "note": "The abort finish-reason counter is recorded but not required: this runtime "
            "can cancel an HTTP stream without incrementing it.",
        }

        runs = {}
        for name, workload, count, concurrency, rate, tokens in [
            ("prefix", "repeated_prefix", 12, 2, None, 16),
            ("mixed", "mixed", 9, 4, None, 16),
            ("overload", "short", 12, 4, 100, 64),
        ]:
            report = await run_benchmark(
                url=cluster.url,
                model=profile.model_alias,
                requests=count,
                concurrency=concurrency,
                rate=rate,
                max_outstanding=32,
                max_tokens=tokens,
                timeout=60,
                workload=workload,
                prefix_groups=3,
                prefix_repetitions=16,
                label=profile.flavor,
                output=str(output / f"{name}.json"),
            )
            outcomes = report["summary"]["outcomes"]
            assert set(outcomes) <= {"success", "rejected"}, outcomes
            if name != "overload":
                assert outcomes == {"success": count}, outcomes
            else:
                assert outcomes.get("rejected", 0) > 0 and outcomes.get("success", 0) > 0
            runs[name] = report["summary"]
            await wait_until(idle)
            print(f"Verified {name} workload: {outcomes}", flush=True)
        observations["workloads"] = runs

        if len(cluster.worker_urls) == 2:
            interrupted, saw_done, killed = [], False, False
            async with client.stream(
                "POST",
                cluster.url + "/v1/chat/completions",
                json={
                    **base,
                    "messages": messages(tag, "List the integers from 1 to 500, one per line."),
                    "max_tokens": 512,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                victim = int(response.headers["x-worker-id"].split("-")[-1]) - 1
                async for data in sse_data(response.aiter_lines()):
                    if data == "[DONE]":
                        saw_done = True
                        break
                    event = json.loads(data)
                    if has_content(event) and not killed:
                        failed_at = time.perf_counter()
                        await cluster.stop_process(cluster.worker_processes[victim], signal.SIGKILL)
                        killed = True
                    if "error" in event:
                        interrupted.append(event["error"]["code"])
            assert killed and not saw_done and "upstream_stream_error" in interrupted

            async def removed():
                state = (await client.get(cluster.url + "/workers")).json()
                return not state["workers"][victim]["healthy"]

            await wait_until(removed)
            detected = time.perf_counter() - failed_at
            survivor = await client.post(cluster.url + "/v1/chat/completions", json=base)
            survivor.raise_for_status()
            assert survivor.headers["x-worker-id"] != f"worker-{victim + 1}"
            restarting = time.perf_counter()
            await cluster.start_worker(victim)

            async def rejoined():
                state = (await client.get(cluster.url + "/workers")).json()
                return all(w["healthy"] for w in state["workers"])

            await wait_until(rejoined)
            restored = set()
            for i in range(2):
                result = await client.post(
                    cluster.url + "/v1/chat/completions",
                    json={
                        **base,
                        "messages": messages(f"{i}-{uuid.uuid4().hex}"),
                    },
                )
                result.raise_for_status()
                restored.add(result.headers["x-worker-id"])
            assert restored == {"worker-1", "worker-2"}
            observations["failure_recovery"] = {
                "interrupted_stream_has_error_and_no_done": True,
                "survivor_served": True,
                "both_workers_served_after_restart": True,
                "observed_detection_seconds": detected,
                "restart_and_verification_seconds": time.perf_counter() - restarting,
            }
        await wait_until(idle)
        observations["final_gateway"] = (await client.get(cluster.url + "/workers")).json()
    return observations


async def run(args):
    profile = FlavorProfile.load(ROOT / args.profile)
    environment = await inspect_runtime(profile, ROOT)
    ports = [free_port() for _ in range(args.workers + 1)]
    cluster = FlavorCluster(
        profile, ROOT, workers=args.workers, port=ports[0], worker_ports=ports[1:]
    )
    output = ROOT / "results" / profile.flavor
    print(f"Starting {profile.flavor}: {args.workers} replicas", flush=True)
    try:
        await cluster.start()
        print("Workers and shared gateway ready", flush=True)
        observations = await checks(cluster, output)
    finally:
        await cluster.close()
    for port in ports:
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0, (
                "Service still listening after cleanup"
            )
    return {
        "schema_version": 1,
        "label": profile.flavor,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "application_python": platform.python_version(),
        },
        "deployment": cluster.description(),
        "environment": environment,
        "checks": observations,
        "services_stopped": True,
        "note": "Mac replicas share one physical GPU; CUDA flavor remains separately unverified.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="configs/macos-metal.json")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    path = ROOT / "results" / report["label"] / "verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified and stopped services. Report: {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
