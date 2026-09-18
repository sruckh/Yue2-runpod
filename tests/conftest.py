"""Shared test fixtures.

Everything here exists so the suite runs on a laptop with no GPU, no network
and no B2 credentials — which is exactly what the Stage 02 contract requires
("GPU + S3 mocked, no GPU needed to run locally").

`conftest.py` puts `worker/` on `sys.path` so tests import the modules exactly
the way the container does (`import config`, not `from worker import config`).
That matters: the flat-import convention is load-bearing in the image, and a
test that imports differently would not catch a break in it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = REPO_ROOT / "worker"

if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test hermetic.

    The worker reads a dozen env vars; a developer with `HF_TOKEN` or `B2_*`
    exported would otherwise get different behaviour from CI. Autouse so no
    test can forget it.
    """
    for name in (
        "B2_ENDPOINT_URL",
        "B2_KEY_ID",
        "B2_APP_KEY",
        "B2_BUCKET",
        "B2_REGION",
        "B2_URL_TTL_SECONDS",
        "VOLUME_ROOT",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_TOKEN",
        "HF_LOCAL_FILES_ONLY",
        "YUE2_VAE",
        "YUE2_PROGRESS",
        "JOB_TIMEOUT_SECONDS",
        "DEFAULT_COT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty fake network volume — nothing cached yet.

    Deliberately empty: most behaviours worth testing are about what happens
    when the cache is *not* populated. Use `populate_runpod_cache` to simulate
    the endpoint's cached-models feature having done its job.
    """
    root = tmp_path / "volume"
    (root / "scratch").mkdir(parents=True)
    monkeypatch.setenv("VOLUME_ROOT", str(root))
    return root


#: The hub-cache path RunPod's cached-models feature mounts, relative to the volume.
HUB_CACHE = Path("huggingface-cache") / "hub"


def hub_cache_root(volume: Path) -> Path:
    return volume / HUB_CACHE


def snapshot_dir(volume: Path, repo_id: str, revision: str = "abc123") -> Path:
    """The hub-cache snapshot directory for a repo, in RunPod's layout."""
    org, name = repo_id.split("/", 1)
    return hub_cache_root(volume) / f"models--{org}--{name}" / "snapshots" / revision


def populate_runpod_cache(
    volume: Path,
    repo_id: str,
    files: tuple[str, ...],
    revision: str = "abc123",
    *,
    write_ref: bool = True,
) -> Path:
    """Simulate RunPod having cached a model at the documented path.

    Creates the standard hub layout: `refs/main` naming a revision, and a
    `snapshots/<rev>/` directory holding the files. That structure is the
    contract — a worker reading anywhere else misses the platform's cache.
    """
    snapshot = snapshot_dir(volume, repo_id, revision)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    if write_ref:
        refs = snapshot.parent.parent / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision, encoding="utf-8")
    return snapshot


@pytest.fixture
def b2_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the mandatory B2 env vars with obviously-fake values."""
    monkeypatch.setenv("B2_ENDPOINT_URL", "https://s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("B2_KEY_ID", "test-key-id")
    monkeypatch.setenv("B2_APP_KEY", "test-app-key")
    monkeypatch.setenv("B2_BUCKET", "yue2-test")


class FakeS3Client:
    """Minimal stand-in for a boto3 S3 client.

    Records what was uploaded so tests can assert on keys and content without
    a network. Deliberately does not implement the full S3 surface — an
    accidental real call should fail loudly, not silently pass.
    """

    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.presigned: list[str] = []

    def upload_file(self, filename: str, bucket: str, key: str, **kwargs: Any) -> None:
        self.uploads.append(
            {
                "filename": filename,
                "bucket": bucket,
                "key": key,
                "extra_args": kwargs.get("ExtraArgs", {}),
                "content": Path(filename).read_bytes(),
            }
        )

    def generate_presigned_url(self, operation: str, Params: dict[str, Any], ExpiresIn: int) -> str:
        assert operation == "get_object", operation
        url = f"https://{Params['Bucket']}.example.invalid/{Params['Key']}?ttl={ExpiresIn}"
        self.presigned.append(url)
        return url


@pytest.fixture
def fake_s3() -> FakeS3Client:
    return FakeS3Client()


class FakeSong:
    """Stands in for the pipeline's `SongResult`.

    Writes the same artifact set the real `save_artifacts()` produces, so the
    handler's directory scan, response assembly and upload path are all
    exercised for real rather than mocked out.
    """

    def __init__(
        self,
        audio_seconds: float = 214.85,
        sample_rate: int = 48000,
        truncated_abc: bool = False,
        truncated_semantic: bool = False,
    ) -> None:
        self.audio_seconds = audio_seconds
        self.sample_rate = sample_rate
        self.truncated_abc = truncated_abc
        self.truncated_semantic = truncated_semantic
        self.saved_to: str | None = None

    def save(self, path: str) -> str:
        Path(path).write_bytes(b"fLaC" + b"\x00" * 64)
        return path

    def save_artifacts(self, directory: str) -> None:
        import json

        self.saved_to = directory
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "audio.flac").write_bytes(b"fLaC" + b"\x00" * 1024)
        (target / "score.abc").write_text("X:1\nT:Test\nK:C\nC D E F|\n", encoding="utf-8")
        (target / "request.json").write_text(json.dumps({"style": "test"}), encoding="utf-8")
        (target / "config.json").write_text(json.dumps({"cot": "full"}), encoding="utf-8")
        (target / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    # A DICT, not a bool. `SongResult.truncated` is a property
                    # returning `{"abc": bool, "semantic": bool}`
                    # (pipeline.py:90-91), written here verbatim
                    # (pipeline.py:113). This double originally wrote a scalar,
                    # which is why the suite could not see the handler turning
                    # every completed song into `truncated: true` via `bool()`.
                    "truncated": {"abc": self.truncated_abc, "semantic": self.truncated_semantic},
                    "sample_rate": self.sample_rate,
                    "audio_seconds": self.audio_seconds,
                    "timing": {"abc": 1.0, "semantic": 2.0, "nar_seconds": 3.0},
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture
def fake_song() -> FakeSong:
    return FakeSong()


class FakePipeline:
    """Records the exact kwargs the handler passes to `pipe(...)`.

    This is the assertion surface that keeps our call honest against the real
    `YuE2Pipeline.__call__` signature verified from the upstream wheel.
    """

    def __init__(self, song: FakeSong) -> None:
        self.song = song
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def __call__(self, **kwargs: Any) -> FakeSong:
        self.calls.append(kwargs)
        return self.song

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_pipeline(fake_song: FakeSong) -> FakePipeline:
    return FakePipeline(fake_song)
