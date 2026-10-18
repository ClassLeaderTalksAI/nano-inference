"""Run a real CPU model through the gateway and save checkable compatibility evidence.

Run scripts/prepare_llama.py first, then:
  uv run --extra tokenizer python scripts/smoke_llama.py
The two child services are always stopped, including after a failed check.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from nano_inference.prefix import PrefixEncoder
from nano_inference.protocol import Message, has_content, sse_data

ROOT = Path(__file__).resolve().parents[1]


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_until(check, seconds=60):
    async with asyncio.timeout(seconds):
        while not await check():  # noqa: ASYNC110 -- Poll the independent child services.
            await asyncio.sleep(0.05)


async def run(manifest, logs):
    worker_port, gateway_port = unused_port(), unused_port()
    worker_url, gateway_url = f"http://127.0.0.1:{worker_port}", f"http://127.0.0.1:{gateway_port}"
    revision = "sha256:" + manifest["model_sha256"]
    encoder = PrefixEncoder.from_pretrained(
        "qwen-local", revision, 64, str(ROOT / manifest["tokenizer"])
    )
    server_command = [
        str(ROOT / manifest["server"]),
        "--model",
        str(ROOT / manifest["model"]),
        "--alias",
        "qwen-local",
        "--host",
        "127.0.0.1",
        "--port",
        str(worker_port),
        "--ctx-size",
        "2048",
        "--parallel",
        "1",
        "--threads",
        "4",
        "--device",
        "none",
        "--n-gpu-layers",
        "0",
        "--no-op-offload",
        "--jinja",
        "--slots",
        "--metrics",
    ]
    gateway_command = [
        sys.executable,
        "-m",
        "nano_inference",
        "gateway",
        "--port",
        str(gateway_port),
        "--worker-urls",
        worker_url,
        "--model",
        "qwen-local",
        "--model-revision",
        revision,
        "--prefix-tokenizer",
        str(ROOT / manifest["tokenizer"]),
        "--policy",
        "prefix_affinity",
        "--capacity",
        "1",
        "--request-timeout",
        "60",
        "--prefix-tokens",
        "64",
    ]
    children = []
    observations = {}
    environment = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    try:
        for command, log in zip([server_command, gateway_command], logs, strict=True):
            children.append(
                await asyncio.create_subprocess_exec(
                    *command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=log,
                    env=environment,
                )
            )
        async with httpx.AsyncClient(trust_env=False, timeout=65) as client:

            async def ready():
                if any(child.returncode is not None for child in children):
                    raise RuntimeError("A child service exited; inspect .local/llama-smoke-*.log")
                try:
                    return (await client.get(gateway_url + "/ready")).status_code == 200
                except httpx.HTTPError:
                    return False

            await wait_until(ready)
            base_messages = [
                {
                    "role": "system",
                    "content": "You are a concise assistant. "
                    + "A queue holds requests until a worker can process them. " * 8,
                },
                {"role": "user", "content": "Name two colors."},
            ]
            parity = []
            for messages in [
                base_messages,
                [
                    {"role": "user", "content": "你好，queue 是什么？"},
                    {"role": "assistant", "content": "A queue stores work."},
                    {"role": "user", "content": "Name one use."},
                ],
            ]:
                applied = await client.post(
                    worker_url + "/apply-template",
                    json={
                        "messages": messages,
                        "add_generation_prompt": True,
                    },
                )
                applied.raise_for_status()
                encoded = await client.post(
                    worker_url + "/tokenize",
                    json={
                        "content": applied.json()["prompt"],
                        "add_special": True,
                        "parse_special": True,
                    },
                )
                encoded.raise_for_status()
                expected = encoder.tokenize([Message(**m) for m in messages])
                assert expected == encoded.json()["tokens"], "Backend/tokenizer mismatch"
                parity.append({"token_ids_match": True, "prompt_tokens": len(expected)})
            observations["tokenization_checks"] = parity

            payload = {
                "model": "qwen-local",
                "messages": base_messages,
                "max_tokens": 16,
                "temperature": 0,
                "stream": False,
            }
            first = await client.post(gateway_url + "/v1/chat/completions", json=payload)
            first.raise_for_status()
            assert first.json()["choices"][0]["message"]["content"]
            observations["non_streaming"] = {
                "content": first.json()["choices"][0]["message"]["content"],
                "usage": first.json()["usage"],
                "decision": first.headers["x-route-decision"],
                "backend_fingerprint": first.json().get("system_fingerprint"),
            }
            content, usage, done = [], None, False
            async with client.stream(
                "POST",
                gateway_url + "/v1/chat/completions",
                json={
                    **payload,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                response.raise_for_status()
                assert response.headers["x-route-decision"] == "affinity_hit"
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
            observations["streaming"] = {
                "done": done,
                "content": "".join(content),
                "usage": usage,
                "decision": "affinity_hit",
            }

            cancelled_after_content = False
            async with client.stream(
                "POST",
                gateway_url + "/v1/chat/completions",
                json={
                    **payload,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a numbered list from 1 to 500, one item per line.",
                        }
                    ],
                    "stream": True,
                    "max_tokens": 512,
                },
            ) as response:
                response.raise_for_status()
                async for data in sse_data(response.aiter_lines()):
                    if data == "[DONE]":
                        raise AssertionError("Expected to cancel a live generation")
                    if has_content(json.loads(data)):
                        cancelled_after_content = True
                        break  # Leave the HTTP stream before generation completes.
            assert cancelled_after_content

            async def all_idle():
                state = (await client.get(gateway_url + "/workers")).json()
                slots = (await client.get(worker_url + "/slots")).json()
                return (
                    state["queued"] == 0
                    and all(w["inflight"] == 0 for w in state["workers"])
                    and all(not slot["is_processing"] for slot in slots)
                )

            await wait_until(all_idle, seconds=5)
            metrics = (await client.get(gateway_url + "/metrics")).text
            assert 'nano_requests_total{outcome="cancelled"} 1.0' in metrics
            observations["cancellation"] = {"gateway_released": True, "backend_slot_idle": True}
            followup = await client.post(gateway_url + "/v1/chat/completions", json=payload)
            followup.raise_for_status()
            assert followup.json()["choices"][0]["message"]["content"]
            observations["request_after_cancellation_succeeded"] = True
            observations["gateway_final"] = (await client.get(gateway_url + "/workers")).json()
            return {
                "schema_version": 1,
                "label": "real_cpu_compatibility_smoke",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "runtime": {
                    "python": platform.python_version(),
                    "system": platform.system(),
                    "machine": platform.machine(),
                },
                "artifacts": manifest,
                "server_flags": [arg.replace(str(ROOT) + "/", "") for arg in server_command[1:]],
                "prefix_identity": encoder.identity,
                "checks": observations,
                "note": "CPU interface/tokenizer check; GPU cache gains remain unverified.",
            }
    finally:
        for child in reversed(children):
            if child.returncode is None:
                child.terminate()
            try:
                await asyncio.wait_for(child.wait(), 5)
            except TimeoutError:
                child.kill()
                await child.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=".local/manifest.json")
    parser.add_argument("--output", default="results/examples/real-cpu-smoke.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text())
    with (ROOT / manifest["model"]).open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == manifest["model_sha256"]
    for name, expected in manifest["tokenizer_files"].items():
        assert (
            hashlib.sha256((ROOT / manifest["tokenizer"] / name).read_bytes()).hexdigest()
            == expected
        )
    with (
        (ROOT / ".local/llama-smoke-server.log").open("w") as server_log,
        (ROOT / ".local/llama-smoke-gateway.log").open("w") as gateway_log,
    ):
        report = asyncio.run(run(manifest, [server_log, gateway_log]))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
