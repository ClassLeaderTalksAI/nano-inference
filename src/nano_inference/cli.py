import argparse
import asyncio
import json
import logging
import signal
import subprocess
import sys

import uvicorn

from .benchmark import run_benchmark
from .experiments import run_experiments
from .flavors import serve_flavor
from .gateway import Settings, create_gateway
from .mock_worker import Behavior, create_mock_worker
from .tenant_benchmark import run_tenant_benchmark


def gateway_options(parser):
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--policy",
        choices=["round_robin", "least_inflight", "prefix_affinity"],
        default="least_inflight",
    )
    parser.add_argument("--capacity", type=int, default=4, help="In-flight limit per worker")
    parser.add_argument("--max-queue", type=int, default=16)
    parser.add_argument("--queue-timeout", type=float, default=2)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--affinity-capacity", type=int, default=1024)
    parser.add_argument("--affinity-ttl", type=float, default=60)
    parser.add_argument("--affinity-max-skew", type=int, default=1)
    parser.add_argument("--prefix-preparations", type=int, default=4)
    parser.add_argument("--tenant-config", help="Tenant policy JSON; API keys come from env")


def run_local(args):
    """Start three ordinary processes and reap all children on exit or startup failure."""
    base = [sys.executable, "-m", "nano_inference"]
    commands = [
        [*base, "worker", "--name", "mock-a", "--port", str(args.worker_ports[0])],
        [*base, "worker", "--name", "mock-b", "--port", str(args.worker_ports[1])],
        [
            *base,
            "gateway",
            "--port",
            str(args.port),
            "--policy",
            args.policy,
            "--capacity",
            str(args.capacity),
            "--max-queue",
            str(args.max_queue),
            "--queue-timeout",
            str(args.queue_timeout),
            "--request-timeout",
            str(args.request_timeout),
            "--prefix-tokens",
            str(args.prefix_tokens),
            "--affinity-capacity",
            str(args.affinity_capacity),
            "--affinity-ttl",
            str(args.affinity_ttl),
            "--affinity-max-skew",
            str(args.affinity_max_skew),
            "--prefix-preparations",
            str(args.prefix_preparations),
            "--worker-urls",
            *[f"http://127.0.0.1:{p}" for p in args.worker_ports],
        ],
    ]
    if args.tenant_config:
        commands[-1].extend(["--tenant-config", args.tenant_config])
    children = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for command in commands:
            children.append(subprocess.Popen(command))
        while not stopping:
            for child in children:
                try:
                    code = child.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
                if stopping:
                    break
                raise SystemExit(code or 1)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(prog="nano-inference")
    commands = parser.add_subparsers(dest="command", required=True)
    local = commands.add_parser("local", help="Run gateway and two CPU-only simulators")
    gateway_options(local)
    local.add_argument("--worker-ports", type=int, nargs=2, default=[8101, 8102])
    worker = commands.add_parser("worker", help="Run one simulated inference worker")
    worker.add_argument("--name", default="mock-a")
    worker.add_argument("--port", type=int, default=8101)
    worker.add_argument("--first-content-delay", type=float, default=0.03)
    worker.add_argument("--chunk-delay", type=float, default=0.005)
    gateway = commands.add_parser("gateway")
    gateway_options(gateway)
    gateway.add_argument("--worker-urls", nargs="+", default=Settings().worker_urls)
    gateway.add_argument("--model", default="nano-model")
    gateway.add_argument("--max-model-len", type=int, help="Input + output token budget guard")
    gateway.add_argument("--health-interval", type=float, default=1)
    gateway.add_argument("--model-revision", default="simulator-v1")
    gateway.add_argument(
        "--prefix-tokenizer", help="Prepared local Hugging Face tokenizer directory"
    )
    serve = commands.add_parser("serve", help="Run the shared platform with a vLLM flavor")
    serve.add_argument("--tenant-config", help="Tenant policy JSON; API keys come from env")
    serve.add_argument("--profile", default="configs/macos-metal.json")
    serve.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    serve.add_argument("--context-guard", action=argparse.BooleanOptionalAction, default=True)
    serve.add_argument("--workers", type=int, choices=[1, 2], default=2)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--worker-ports", type=int, nargs="+")
    serve.add_argument(
        "--policy",
        choices=["round_robin", "least_inflight", "prefix_affinity"],
        default="prefix_affinity",
    )
    serve.add_argument("--capacity", type=int, default=2)
    serve.add_argument("--max-queue", type=int, default=4)
    serve.add_argument("--request-timeout", type=float, default=60)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--url", default="http://127.0.0.1:8000")
    benchmark.add_argument("--model", default="nano-model")
    benchmark.add_argument("--requests", type=int, default=50)
    benchmark.add_argument("--concurrency", type=int, default=4)
    benchmark.add_argument("--rate", type=float, help="Open-loop requests per second")
    benchmark.add_argument("--max-outstanding", type=int, default=128)
    benchmark.add_argument("--max-tokens", type=int, default=32)
    benchmark.add_argument("--timeout", type=float, default=60)
    benchmark.add_argument("--slo", type=float, default=2, help="First-content SLO in seconds")
    benchmark.add_argument("--output", default="results/latest.json")
    benchmark.add_argument("--label", default="simulated")
    benchmark.add_argument(
        "--workload",
        choices=["short", "mixed", "repeated_prefix", "single_prefix"],
        default="short",
    )
    benchmark.add_argument("--prefix-groups", type=int, default=8)
    benchmark.add_argument("--prefix-repetitions", type=int, default=128)
    tenant_load = commands.add_parser("tenant-load", help="Open-loop noisy-neighbor demo")
    tenant_load.add_argument("--url", default="http://127.0.0.1:8000")
    tenant_load.add_argument("--model", default="nano-model")
    tenant_load.add_argument("--tenant-config", default="configs/tenants.example.json")
    tenant_load.add_argument("--requests", type=int, default=30, help="Arrivals per tenant")
    tenant_load.add_argument("--rate", type=float, default=5, help="Base arrivals/s; batch uses 4x")
    tenant_load.add_argument("--max-outstanding", type=int, default=64)
    tenant_load.add_argument("--max-tokens", type=int, default=32)
    tenant_load.add_argument("--output", default="results/tenants-latest.json")
    experiment = commands.add_parser("experiment", help="Paired interview evidence batches")
    experiment.add_argument("--batch", choices=["control", "metal"], default="control")
    experiment.add_argument("--config", default="configs/experiments.local.json")
    experiment.add_argument("--output")
    experiment.add_argument("--profile", default="configs/macos-metal.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "local":
        run_local(args)
    elif args.command == "worker":
        uvicorn.run(
            create_mock_worker(
                args.name,
                Behavior(
                    first_content_delay=args.first_content_delay,
                    chunk_delay=args.chunk_delay,
                ),
            ),
            host="127.0.0.1",
            port=args.port,
        )
    elif args.command == "gateway":
        settings = Settings(**{k: v for k, v in vars(args).items() if k not in ("command", "port")})
        uvicorn.run(create_gateway(settings), host="127.0.0.1", port=args.port)
    elif args.command == "serve":
        try:
            asyncio.run(serve_flavor(args))
        except KeyboardInterrupt:
            pass
    elif args.command == "experiment":
        asyncio.run(run_experiments(**{k: v for k, v in vars(args).items() if k != "command"}))
    else:
        runner = run_tenant_benchmark if args.command == "tenant-load" else run_benchmark
        report = asyncio.run(runner(**{k: v for k, v in vars(args).items() if k != "command"}))
        print(json.dumps(report["summary"], indent=2))
