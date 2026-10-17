import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "sf_asset_downloader", Path("scripts/sf_assets.py")
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class Response(BytesIO):
    status = 200


def row(data):
    return dict(
        path="checkpoint/file",
        size=len(data),
        algorithm="sha256",
        digest=hashlib.sha256(data).hexdigest(),
        url="https://example.invalid/file",
    )


def test_zero_byte_checkpoint_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "DEST", tmp_path)
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: Response(b""))
    assert assets.get(row(b"")) == "checkpoint/file"
    assert (tmp_path / "checkpoint/file").is_file()
    assert assets.get(row(b""), verify_only=True) == "checkpoint/file"


def test_range_ignored_restarts_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "DEST", tmp_path)
    partial = tmp_path / "checkpoint/file.part"
    partial.parent.mkdir()
    partial.write_bytes(b"abc")
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *a, **k: Response(b"abcdef")
    )
    assets.get(row(b"abcdef"))
    assert partial.with_suffix("").read_bytes() == b"abcdef"


def test_corrupt_asset_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "DEST", tmp_path)
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *a, **k: Response(b"bad")
    )
    monkeypatch.setattr(assets.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Hash/size mismatch"):
        assets.get(row(b"yes"))
    assert not (tmp_path / "checkpoint/file").exists()
