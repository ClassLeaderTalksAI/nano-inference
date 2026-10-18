"""Download a pinned Mac ARM llama.cpp runtime, Q4 model, and tokenizer locally.

Usage: python scripts/prepare_llama.py
Downloads about 2.6 GB under .local/; never installs system packages.
"""

import hashlib
import json
import platform
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".local"
LLAMA_TAG = "b11719"
LLAMA_SHA256 = "076a2b0d15253078b90bd520a82dc2edc6c17657968878876ba3def3e80f54ba"
MODEL_REPO = "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF"
MODEL_REVISION = "4edb920b6f14e3b9284d4502a6485103d72cde05"
MODEL_FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_SHA256 = "8cdb57cbb880d313736a9bc4e3d3d2485f145b5e19cf33783746e753e82641fc"
TOKENIZER_REPO = "Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOKENIZER_HASHES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
    "config.json": "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
}


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(url, path, expected=None):
    if path.exists() and (expected is None or sha256(path) == expected):
        print(f"Already prepared: {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as target:
            total = 0
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
                total += len(block)
                if total % (256 * 1024 * 1024) == 0:
                    print(f"  {total // (1024 * 1024)} MiB", flush=True)
        if expected is not None and sha256(partial) != expected:
            raise ValueError(f"SHA256 mismatch: {path.name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(
            "This pinned runtime is for macOS arm64; provide your own llama-server elsewhere."
        )
    archive = DEST / f"llama-{LLAMA_TAG}-bin-macos-arm64.tar.gz"
    download(
        f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{archive.name}",
        archive,
        LLAMA_SHA256,
    )
    runtime = DEST / "llama"
    runtime.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        source.extractall(runtime, filter="data")
    download(
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}",
        DEST / MODEL_FILE,
        MODEL_SHA256,
    )
    tokenizer = DEST / "tokenizer"
    for filename, expected in TOKENIZER_HASHES.items():
        download(
            f"https://huggingface.co/{TOKENIZER_REPO}/resolve/{TOKENIZER_REVISION}/{filename}",
            tokenizer / filename,
            expected,
        )
    binary = next(runtime.rglob("llama-server"))
    manifest = {
        "llama_tag": LLAMA_TAG,
        "llama_archive_sha256": LLAMA_SHA256,
        "server": str(binary.relative_to(ROOT)),
        "model": str((DEST / MODEL_FILE).relative_to(ROOT)),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer": str(tokenizer.relative_to(ROOT)),
        "tokenizer_files": {p.name: sha256(p) for p in tokenizer.iterdir() if p.is_file()},
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared .local/manifest.json", flush=True)


if __name__ == "__main__":
    main()
