import argparse
import asyncio
import json
import logging
import signal
import subprocess
import sys

import uvicorn

from .benchmark import run_benchmark
from .gateway import Settings, create_gateway
from .mock_worker import Behavior, create_mock_worker


def gateway_options(parser):
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--policy", choices=["round_robin", "least_inflight"], default="least_inflight"
    )
    parser.add_argument("--capacity", type=int, default=4, help="In-flight limit per worker")
    parser.add_argument("--max-queue", type=int, default=16)
    parser.add_argument("--queue-timeout", type=float, default=2)
    parser.add_argument("--request-timeout", type=float, default=30)


def run_local(args):
    """Start three ordinary processes and reap all children on exit or startup failure."""
    base = [sys.executable, "-m", "nano_inference"]
    commands = [
        [*base, "worker", "--name", "mock-a", "--port", "8101"],
        [*base, "worker", "--name", "mock-b", "--port", "8102"],
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
        ],
    ]
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
    worker = commands.add_parser("worker", help="Run one simulated inference worker")
    worker.add_argument("--name", default="mock-a")
    worker.add_argument("--port", type=int, default=8101)
    worker.add_argument("--first-content-delay", type=float, default=0.03)
    worker.add_argument("--chunk-delay", type=float, default=0.005)
    gateway = commands.add_parser("gateway")
    gateway_options(gateway)
    gateway.add_argument("--worker-urls", nargs="+", default=Settings().worker_urls)
    gateway.add_argument("--model", default="nano-model")
    gateway.add_argument("--health-interval", type=float, default=1)
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
    else:
        report = asyncio.run(
            run_benchmark(**{k: v for k, v in vars(args).items() if k != "command"})
        )
        print(json.dumps(report["summary"], indent=2))
