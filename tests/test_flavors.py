import asyncio
import os
import platform
import signal
import socket
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from nano_inference.flavors import FlavorCluster, FlavorProfile

from .conftest import eventually


def profile(**overrides):
    return FlavorProfile.model_validate(
        {
            "flavor": "macos-metal",
            "runtime_executable": ".local/metal-venv/bin/vllm",
            "runtime_version": "0.31.0+cpu",
            "model_repo": "example/model",
            "model_revision": "a" * 40,
            "model_path": ".local/metal-model",
            **overrides,
        }
    )


def test_runtime_profile_keeps_the_serving_contract_and_changes_device_selection(tmp_path):
    metal = profile()
    cuda = profile(flavor="linux-cuda", runtime_executable=".local/cuda-venv/bin/vllm")
    # The server arguments (API contract and limits) are common; the executable differs.
    assert metal.worker_command(tmp_path, 8101)[1:] == cuda.worker_command(tmp_path, 8101)[1:]
    assert metal.worker_environment(tmp_path, 0)["VLLM_MLX_DEVICE"] == "gpu"
    assert cuda.worker_environment(tmp_path, 1)["CUDA_VISIBLE_DEVICES"] == "1"
    assert "--no-enable-prefix-caching" in metal.worker_command(
        tmp_path, 8101, prefix_caching=False
    )
    assert "--enable-prefix-caching" in metal.worker_command(tmp_path, 8101)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_revision": "main"},
        {"gpu_memory_utilization": 0},
        {"max_num_seqs": 0},
        {"model_files": {"../weights": "a" * 64}},
        {"model_files": {"weights": "not-a-sha"}},
    ],
)
def test_profiles_reject_unpinned_or_invalid_resources(overrides):
    with pytest.raises(ValidationError):
        profile(**overrides)


def test_host_validation_is_explicit(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="Darwin arm64"):
        profile().validate_host()
    profile(flavor="linux-cuda").validate_host()


@pytest.mark.parametrize("workers,ports", [(0, []), (2, [8101]), (2, [8000, 8101])])
def test_invalid_cluster_topology_is_rejected(tmp_path, workers, ports):
    with pytest.raises(ValueError):
        FlavorCluster(profile(), tmp_path, workers=workers, worker_ports=ports)


def fake_runtime(tmp_path: Path, *, fail=False):
    worker = tmp_path / "fake-vllm"
    worker.write_text(
        f"#!{sys.executable}\n"
        + (
            "raise SystemExit(1)\n"
            if fail
            else """
import argparse
import uvicorn
from nano_inference.mock_worker import create_mock_worker
p=argparse.ArgumentParser()
p.add_argument('serve');p.add_argument('model')
p.add_argument('--port',type=int);p.add_argument('--served-model-name')
args,_=p.parse_known_args()
uvicorn.run(create_mock_worker(),host='127.0.0.1',port=args.port,log_level='critical')
"""
        )
    )
    worker.chmod(0o755)
    return profile(
        runtime_executable=str(worker), model_path=str(tmp_path), model_alias="nano-model"
    )


async def test_shared_launcher_failover_restart_and_cleanup(
    tmp_path, unused_tcp_port_factory, monkeypatch
):
    monkeypatch.setattr(FlavorProfile, "validate_host", lambda self: None)
    ports = [unused_tcp_port_factory() for _ in range(3)]
    cluster = FlavorCluster(
        fake_runtime(tmp_path),
        Path.cwd(),
        workers=2,
        port=ports[0],
        worker_ports=ports[1:],
        policy="round_robin",
        log_dir=tmp_path / "logs",
    )
    try:
        await cluster.start()
        async with httpx.AsyncClient(trust_env=False) as client:
            await cluster.stop_process(cluster.worker_processes[0], signal.SIGKILL)

            async def removed():
                state = (await client.get(cluster.url + "/workers")).json()
                return not state["workers"][0]["healthy"]

            await eventually(removed)
            response = await client.post(
                cluster.url + "/v1/chat/completions",
                json={
                    "model": "nano-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "max_tokens": 1,
                },
            )
            assert response.status_code == 200
            assert response.headers["x-worker-id"] == "worker-2"
            await cluster.start_worker(0)

            async def rejoined():
                return all(
                    w["healthy"]
                    for w in (await client.get(cluster.url + "/workers")).json()["workers"]
                )

            await eventually(rejoined)
    finally:
        await cluster.close()
    await cluster.close()  # Cleanup is safe after a prior cleanup.
    for port in ports:
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", port)) != 0


async def test_startup_failure_reaps_partial_cluster(
    tmp_path, unused_tcp_port_factory, monkeypatch
):
    monkeypatch.setattr(FlavorProfile, "validate_host", lambda self: None)
    cluster = FlavorCluster(
        fake_runtime(tmp_path, fail=True),
        Path.cwd(),
        workers=1,
        port=unused_tcp_port_factory(),
        worker_ports=[unused_tcp_port_factory()],
        log_dir=tmp_path / "logs",
    )
    with pytest.raises(RuntimeError, match="Service exited"):
        await cluster.start()
    assert not cluster.children
    assert not cluster.logs
    assert all(p.returncode is not None for p in cluster.worker_processes)


async def test_serve_cli_ctrl_c_stops_private_worker_groups(tmp_path, unused_tcp_port_factory):
    fake = fake_runtime(tmp_path)
    if platform.system() == "Linux":
        fake = fake.model_copy(update={"flavor": "linux-cuda"})
    path = tmp_path / "profile.json"
    path.write_text(fake.model_dump_json())
    ports = [unused_tcp_port_factory() for _ in range(3)]
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "nano_inference",
        "serve",
        "--profile",
        str(path),
        "--port",
        str(ports[0]),
        "--worker-ports",
        str(ports[1]),
        str(ports[2]),
        "--policy",
        "round_robin",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:

            async def ready():
                if child.returncode is not None:
                    raise RuntimeError("Flavor launcher exited before readiness")
                try:
                    return (
                        await client.get(f"http://127.0.0.1:{ports[0]}/ready")
                    ).status_code == 200
                except httpx.HTTPError:
                    return False

            await eventually(ready, seconds=15)
        os.killpg(child.pid, signal.SIGINT)
        await asyncio.wait_for(child.wait(), 15)
        assert child.returncode == 0
        for port in ports:
            with socket.socket() as sock:
                assert sock.connect_ex(("127.0.0.1", port)) != 0
    finally:
        if child.returncode is None:
            os.killpg(child.pid, signal.SIGINT)
            await asyncio.wait_for(child.wait(), 15)
