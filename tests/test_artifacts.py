import hashlib
import io

import pytest

from nano_inference.artifacts import download_verified


def test_verified_download_is_atomic_and_reuses_valid_file(tmp_path, monkeypatch):
    content = b"a pinned model artifact"
    expected = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(content))
    target = tmp_path / "weights"
    download_verified("https://example.test/weights", target, expected)
    assert target.read_bytes() == content
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **kw: pytest.fail("Unexpected download")
    )
    download_verified("https://example.test/weights", target, expected)


def test_corrupt_download_does_not_replace_old_file_or_leave_partial(tmp_path, monkeypatch):
    target = tmp_path / "weights"
    target.write_bytes(b"old file")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download_verified("https://example.test/weights", target, "a" * 64)
    assert target.read_bytes() == b"old file"
    assert not target.with_suffix(".part").exists()
