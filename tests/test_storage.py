"""Storage tests — the B2/S3 egress path, with a fake client.

The real value here is asserting the *contract*: response carries URLs, never
bytes, and every artifact that lands in the workdir gets uploaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from config import StorageConfig
from storage import B2Storage, StorageError, UploadedArtifact, job_prefix

if TYPE_CHECKING:
    pass


def storage_config(**overrides: Any) -> StorageConfig:
    base: dict[str, Any] = {
        "endpoint_url": "https://s3.us-west-004.backblazeb2.com",
        "key_id": "kid",
        "app_key": "app",
        "bucket": "bucket",
        "region": "us-west-004",
    }
    base.update(overrides)
    return StorageConfig(**base)


def test_from_env_requires_every_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("B2_ENDPOINT_URL", raising=False)
    with pytest.raises(Exception, match="B2_ENDPOINT_URL"):
        StorageConfig.from_env()


def test_from_env_reads_all_values(b2_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B2_URL_TTL_SECONDS", "3600")
    cfg = StorageConfig.from_env()
    assert cfg.bucket == "yue2-test"
    assert cfg.url_ttl_seconds == 3600


def test_upload_file_returns_url_not_bytes(tmp_path: Path, fake_s3: object) -> None:
    path = tmp_path / "audio.flac"
    path.write_bytes(b"fLaC" + b"\x00" * 32)

    storage = B2Storage(storage_config(), client=fake_s3)
    artifact = storage.upload_file(path, "jobs/1/audio.flac")

    assert isinstance(artifact, UploadedArtifact)
    assert artifact.key == "jobs/1/audio.flac"
    assert artifact.content_type == "audio/flac"
    assert artifact.size_bytes == path.stat().st_size
    assert artifact.url.startswith("https://bucket.example.invalid/")


def test_upload_file_missing_file_raises(tmp_path: Path, fake_s3: object) -> None:
    storage = B2Storage(storage_config(), client=fake_s3)
    with pytest.raises(StorageError, match="missing file"):
        storage.upload_file(tmp_path / "nope.flac", "k")


def test_upload_directory_covers_every_file(tmp_path: Path, fake_s3: object) -> None:
    workdir = tmp_path / "artifacts"
    workdir.mkdir()
    (workdir / "audio.flac").write_bytes(b"fLaC")
    (workdir / "score.abc").write_text("X:1\n", encoding="utf-8")
    (workdir / "result.json").write_text("{}", encoding="utf-8")

    storage = B2Storage(storage_config(), client=fake_s3)
    uploaded = storage.upload_directory(workdir, "jobs/7/song")

    assert set(uploaded) == {"audio.flac", "score.abc", "result.json"}
    assert uploaded["score.abc"].key == "jobs/7/song/score.abc"
    assert len(fake_s3.uploads) == 3  # type: ignore[attr-defined]


def test_upload_directory_missing_dir_raises(tmp_path: Path, fake_s3: object) -> None:
    storage = B2Storage(storage_config(), client=fake_s3)
    with pytest.raises(StorageError, match="missing directory"):
        storage.upload_directory(tmp_path / "nope", "p")


def test_presign_uses_configured_ttl(fake_s3: object) -> None:
    storage = B2Storage(storage_config(url_ttl_seconds=1234), client=fake_s3)
    assert "ttl=1234" in storage.presign("some/key")


def test_content_types_are_guessed_per_artifact(tmp_path: Path, fake_s3: object) -> None:
    storage = B2Storage(storage_config(), client=fake_s3)
    for name, expected in [
        ("audio.flac", "audio/flac"),
        # `.abc` is a registered MIME type (ABC notation), so this is not
        # octet-stream. Asserted explicitly so the guess is pinned, not assumed.
        ("score.abc", "text/vnd.abc"),
        ("result.json", "application/json"),
    ]:
        path = tmp_path / name
        path.write_bytes(b"x")
        assert storage.upload_file(path, name).content_type == expected


def test_upload_failure_is_normalised(tmp_path: Path) -> None:
    class Boom:
        def upload_file(self, *a: object, **k: object) -> None:
            raise OSError("connection reset")

    path = tmp_path / "audio.flac"
    path.write_bytes(b"fLaC")
    storage = B2Storage(storage_config(), client=Boom())
    with pytest.raises(StorageError, match="Upload of"):
        storage.upload_file(path, "k")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("abc123", "jobs/abc123"),
        ("a/b", "jobs/a-b"),
        ("../../etc", "jobs/etc"),
        ("./hidden", "jobs/hidden"),
        ("///", "jobs/job"),
        ("", "jobs/job"),
    ],
)
def test_job_prefix_is_path_safe(raw: str, expected: str) -> None:
    assert job_prefix(raw) == expected


def test_job_prefix_never_contains_dotdot() -> None:
    """Guards the invariant directly, so a future edit to the sanitiser is caught."""
    for hostile in ["..", "../..", "a/../../b", "...", ".hidden"]:
        assert ".." not in job_prefix(hostile)
        assert ".." not in job_prefix(hostile, hostile)


def test_job_prefix_preserves_hierarchy_across_segments() -> None:
    """Segments are sanitised individually; the separator must survive."""
    assert job_prefix("job-1", "song-a") == "jobs/job-1/song-a"


def test_job_prefix_sanitises_each_segment_not_the_join() -> None:
    """A hostile job id cannot swallow the request-id segment."""
    assert job_prefix("../../x", "y") == "jobs/x/y"
