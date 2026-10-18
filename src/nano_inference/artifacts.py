"""Download immutable artifacts into a caller-selected local directory."""

import hashlib
import urllib.request
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download_verified(url: str, path: Path, expected: str) -> None:
    if path.exists() and file_sha256(path) == expected:
        print(f"Verified existing {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        print(f"Downloading {path.name}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as target:
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
        if file_sha256(partial) != expected:
            raise ValueError(f"SHA256 mismatch for {path.name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
