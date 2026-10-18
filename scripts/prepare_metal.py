"""Prepare the pinned Mac flavor without touching the application's environment."""

import json
import os
import subprocess
from pathlib import Path

from nano_inference.artifacts import download_verified, file_sha256
from nano_inference.flavors import FlavorProfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    profile_path = ROOT / "configs/macos-metal.json"
    profile = FlavorProfile.load(profile_path)
    profile.validate_host()
    env = {
        **os.environ,
        "UV_CACHE_DIR": str(ROOT / ".local/uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(ROOT / ".local/python"),
    }
    python = ROOT / ".local/metal-venv/bin/python"
    if not python.exists():
        subprocess.run(
            ["uv", "venv", "--python", "3.12", "--managed-python", str(python.parent.parent)],
            cwd=ROOT,
            env=env,
            check=True,
        )
    subprocess.run(
        ["uv", "pip", "sync", "--python", str(python), str(ROOT / "requirements/macos-metal.lock")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    for name, expected in profile.model_files.items():
        download_verified(
            f"https://huggingface.co/{profile.model_repo}/resolve/{profile.model_revision}/{name}",
            ROOT / profile.model_path / name,
            expected,
        )
    script = (
        "import importlib.metadata as m,json,platform;print(json.dumps({"
        "'python':platform.python_version(),'packages':"
        "{n:m.version(n) for n in ['vllm','vllm-metal','mlx','mlx-lm','transformers']}}))"
    )
    versions = json.loads(
        subprocess.run(
            [str(python), "-c", script], capture_output=True, text=True, check=True
        ).stdout
    )
    if (
        versions["packages"]["vllm"] != profile.runtime_version
        or versions["packages"]["vllm-metal"] != profile.plugin_version
    ):
        raise ValueError("Installed runtime does not match the flavor profile")
    manifest = {
        "profile": profile.model_dump(),
        "runtime": versions,
        "profile_sha256": file_sha256(profile_path),
        "runtime_lock_sha256": file_sha256(ROOT / "requirements/macos-metal.lock"),
    }
    path = ROOT / ".local/macos-metal-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared Mac flavor. Run: uv run --extra tokenizer nano-inference serve", flush=True)


if __name__ == "__main__":
    main()
