"""Launch profiles keep the platform independent of the vLLM hardware backend.

Both flavors run the same gateway and speak the same vLLM HTTP contract. Their
runtime environments, model artifacts and memory budgets remain separate.
"""

import asyncio
import contextlib
import json
import os
import platform
import signal
import socket
import sys
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FlavorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flavor: Literal["macos-metal", "linux-cuda"]
    runtime_executable: str
    runtime_version: str
    plugin_version: str | None = None
    model_repo: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_path: str
    model_alias: str = "nano-qwen"
    model_files: dict[str, str] = Field(default_factory=dict)
    max_model_len: int = Field(default=2048, ge=256)
    max_num_seqs: int = Field(default=2, ge=1)
    max_num_batched_tokens: int = Field(default=2048, ge=256)
    gpu_memory_utilization: float = Field(default=0.15, gt=0, le=1)
    dtype: Literal["float16", "bfloat16", "auto"] = "float16"

    @field_validator("model_files")
    @classmethod
    def safe_artifact_names(cls, values):
        import re

        for name, sha in values.items():
            if (
                Path(name).name != name
                or name in (".", "..")
                or not re.fullmatch("[0-9a-f]{64}", sha)
            ):
                raise ValueError("Model artifacts require plain filenames and SHA256 hashes")
        return values

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate_json(Path(path).read_text())

    def worker_command(self, root: Path, port: int, *, prefix_caching: bool = True) -> list[str]:
        return [
            str(root / self.runtime_executable),
            "serve",
            str(root / self.model_path),
            "--served-model-name",
            self.model_alias,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--enforce-eager",
            "--generation-config",
            "vllm",
        ]

    def worker_environment(self, root: Path, index: int) -> dict[str, str]:
        cache = root / ".local" / f"{self.flavor}-cache"
        values = {
            **os.environ,
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
            "HF_HOME": str(cache / "huggingface"),
            "XDG_CACHE_HOME": str(cache),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "VLLM_LOGGING_INTERVAL": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        if self.flavor == "macos-metal":
            values["VLLM_MLX_DEVICE"] = "gpu"
        else:
            values["CUDA_VISIBLE_DEVICES"] = str(index)
        return values

    def validate_host(self):
        expected = ("Darwin", "arm64") if self.flavor == "macos-metal" else ("Linux", "x86_64")
        if (platform.system(), platform.machine()) != expected:
            raise ValueError(f"{self.flavor} requires {expected[0]} {expected[1]}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a")


async def inspect_runtime(profile: FlavorProfile, root: Path) -> dict:
    """Collect package identity; Metal inspection also verifies a GPU computation."""
    python = (root / profile.runtime_executable).with_name("python")
    names = (
        ["vllm", "vllm-metal", "mlx", "mlx-lm", "transformers"]
        if profile.flavor == "macos-metal"
        else ["vllm", "torch", "transformers"]
    )
    script = (
        "import importlib.metadata as m,json,platform;data={};"
        "data['python']=platform.python_version();"
        "data['packages']={n:m.version(n) for n in " + repr(names) + "};"
    )
    if profile.flavor == "macos-metal":
        script += (
            "import mlx.core as mx;mx.set_default_device(mx.gpu);"
            "x=mx.array([1.,2.,3.]);y=x*x;mx.eval(y);"
            "data['accelerator']=mx.device_info();data['metal_computation']=y.tolist();"
        )
    script += "print(json.dumps(data))"
    process = await asyncio.create_subprocess_exec(
        str(python),
        "-c",
        script,
        env=profile.worker_environment(root, 0),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"Runtime/device inspection failed: {stderr.decode()[-1000:]}")
    info = json.loads(stdout)
    if info["packages"]["vllm"] != profile.runtime_version:
        raise ValueError("Installed vLLM package does not match the flavor profile")
    if (
        profile.plugin_version is not None
        and info["packages"].get("vllm-metal") != profile.plugin_version
    ):
        raise ValueError("Installed Metal plugin does not match the flavor profile")
    return info


class FlavorCluster:
    """Own the worker process groups and gateway; never install or rent resources."""

    def __init__(
        self,
        profile: FlavorProfile,
        root: Path,
        *,
        workers: int = 2,
        port: int = 8000,
        worker_ports: list[int] | None = None,
        policy: str = "prefix_affinity",
        capacity: int = 2,
        max_queue: int = 4,
        request_timeout: float = 60,
        prefix_caching: bool = True,
        log_dir: Path | None = None,
        tenant_config: str | None = None,
        context_guard: bool = True,
    ):
        if workers not in (1, 2):
            raise ValueError("This learning profile supports one or two replicas")
        self.profile, self.root = profile, root.resolve()
        self.ports = worker_ports or [8101 + i for i in range(workers)]
        if len(self.ports) != workers or len(set([port, *self.ports])) != workers + 1:
            raise ValueError("Gateway and workers need distinct ports, one per replica")
        self.url = f"http://127.0.0.1:{port}"
        self.worker_urls = [f"http://127.0.0.1:{p}" for p in self.ports]
        self.prefix_caching = prefix_caching
        self.log_dir = log_dir or self.root / ".local" / "logs"
        self.gateway_command = [
            sys.executable,
            "-m",
            "nano_inference",
            "gateway",
            "--port",
            str(port),
            "--worker-urls",
            *self.worker_urls,
            "--model",
            profile.model_alias,
            "--model-revision",
            profile.model_revision,
            "--prefix-tokenizer",
            str(self.root / profile.model_path),
            "--policy",
            policy,
            "--capacity",
            str(capacity),
            "--max-queue",
            str(max_queue),
            "--queue-timeout",
            "2",
            "--request-timeout",
            str(request_timeout),
            "--health-interval",
            ".2",
        ]
        if context_guard:
            self.gateway_command.extend(["--max-model-len", str(profile.max_model_len)])
        if tenant_config:
            self.gateway_command.extend(["--tenant-config", str(Path(tenant_config).resolve())])
        self.children: list[asyncio.subprocess.Process] = []
        self.logs = []
        self.worker_processes: list[asyncio.subprocess.Process] = []

    async def _spawn(self, command, name, env=None):
        log = open_log(self.log_dir / f"{self.profile.flavor}-{name}.log")
        self.logs.append(log)
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.root,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        self.children.append(process)
        return process

    async def start_worker(self, index):
        process = await self._spawn(
            self.profile.worker_command(
                self.root, self.ports[index], prefix_caching=self.prefix_caching
            ),
            f"worker-{index + 1}",
            self.profile.worker_environment(self.root, index),
        )
        if index < len(self.worker_processes):
            self.worker_processes[index] = process
        else:
            self.worker_processes.append(process)
        await self._ready(self.worker_urls[index] + "/health", process)

    async def _ready(self, url, process):
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            async with asyncio.timeout(180):
                while True:
                    if process.returncode is not None:
                        raise RuntimeError(
                            f"Service exited; inspect .local/logs/{self.profile.flavor}-*.log"
                        )
                    try:
                        if (await client.get(url)).status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.2)

    async def start(self):
        self.profile.validate_host()
        for path in (self.profile.runtime_executable, self.profile.model_path):
            if not (self.root / path).exists():
                raise FileNotFoundError(f"Missing {path}; prepare the runtime profile first")
        try:
            # Serial startup limits transient memory and gives each failure its own log.
            for i in range(len(self.ports)):
                await self.start_worker(i)
            self.gateway = await self._spawn(self.gateway_command, "gateway")
            await self._ready(self.url + "/ready", self.gateway)
        except BaseException:
            await self.close()
            raise

    @staticmethod
    async def stop_process(process, sig=signal.SIGTERM):
        if process.returncode is not None:
            return
        # The group was created exclusively for this child, including vLLM EngineCore.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            await asyncio.wait_for(process.wait(), 10)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    async def close(self):
        for process in reversed(self.children):
            await self.stop_process(process)
        self.children.clear()
        for log in self.logs:
            log.close()
        self.logs.clear()

    def description(self):
        return {
            "profile": self.profile.model_dump(),
            "gateway_url": self.url,
            "worker_urls": self.worker_urls,
            "prefix_caching": self.prefix_caching,
            "physical_gpu_sharing": self.profile.flavor == "macos-metal",
        }


async def serve_flavor(args):
    profile = FlavorProfile.load(args.profile)
    cluster = FlavorCluster(
        profile,
        Path.cwd(),
        workers=args.workers,
        port=args.port,
        worker_ports=args.worker_ports,
        policy=args.policy,
        capacity=args.capacity,
        max_queue=args.max_queue,
        request_timeout=args.request_timeout,
        tenant_config=args.tenant_config,
        prefix_caching=args.prefix_caching,
        context_guard=args.context_guard,
    )
    try:
        await cluster.start()
        print(json.dumps(cluster.description(), indent=2), flush=True)
        await cluster.gateway.wait()
        raise RuntimeError("Gateway exited; inspect its log")
    finally:
        await cluster.close()
