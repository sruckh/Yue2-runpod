"""Boot tests — volume caching and pipeline load, with HuggingFace faked.

The important behaviour here is the *warm* path: a worker restarting on an
already-populated volume must not download anything. That is the entire point of
the network-volume decision, and it is easy to regress by dropping the
missing-files check and always calling `snapshot_download`.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

import boot
from boot import (
    REQUIRED_MODEL_FILES,
    REQUIRED_VAE_FILES,
    BootError,
    ensure_models,
    health,
    install_model_wheel,
)
from config import MODEL_WHEEL, CacheConfig


class SnapshotRecorder:
    """Stands in for `huggingface_hub.snapshot_download`."""

    def __init__(self, *, populate: dict[str, tuple[str, ...]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.populate = populate or {}

    def __call__(self, repo_id: str, local_dir: str, **kwargs: Any) -> str:
        self.calls.append({"repo_id": repo_id, "local_dir": local_dir, **kwargs})
        target = Path(local_dir)
        for name in self.populate.get(repo_id, ()):
            (target / name).write_bytes(b"x")
        return local_dir


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> SnapshotRecorder:
    recorder = SnapshotRecorder()
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return recorder


def populate_volume(root: Path, *, model: bool = True, vae: bool = True) -> None:
    for name, present, required in (
        ("YuE2-3B", model, REQUIRED_MODEL_FILES),
        ("YuE2-Vae", vae, REQUIRED_VAE_FILES),
    ):
        directory = root / "models" / name
        directory.mkdir(parents=True, exist_ok=True)
        if present:
            for filename in required:
                (directory / filename).write_bytes(b"x")


# --- warm cache --------------------------------------------------------------


def test_warm_volume_downloads_nothing(volume: Path, fake_hf: SnapshotRecorder) -> None:
    populate_volume(volume)
    report = ensure_models(CacheConfig())

    assert report.cache_hit is True
    assert report.downloaded == []
    assert set(report.already_present) == {"m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"}
    assert fake_hf.calls == [], "a warm cache must not hit the network"


def test_warm_report_names_no_downloads(volume: Path, fake_hf: SnapshotRecorder) -> None:
    populate_volume(volume)
    assert ensure_models(CacheConfig()).downloaded == []


# --- cold cache --------------------------------------------------------------


def test_cold_volume_downloads_both_repos(volume: Path, fake_hf: SnapshotRecorder) -> None:
    fake_hf.populate = {
        "m-a-p/YuE2-3B": REQUIRED_MODEL_FILES,
        "m-a-p/YuE2-Vae": REQUIRED_VAE_FILES,
    }
    report = ensure_models(CacheConfig())

    assert report.cache_hit is False
    assert set(report.downloaded) == {"m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"}
    assert [c["repo_id"] for c in fake_hf.calls] == ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]


def test_partial_cache_refetches_only_the_incomplete_repo(volume: Path, fake_hf: SnapshotRecorder) -> None:
    populate_volume(volume, model=True, vae=False)
    fake_hf.populate = {"m-a-p/YuE2-Vae": REQUIRED_VAE_FILES}

    report = ensure_models(CacheConfig())
    assert report.downloaded == ["m-a-p/YuE2-Vae"]
    assert report.already_present == ["m-a-p/YuE2-3B"]
    assert [c["repo_id"] for c in fake_hf.calls] == ["m-a-p/YuE2-Vae"]


def test_incomplete_download_raises_with_repo_layout_hint(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """A repo that downloads but stays incomplete is a layout change, not a blip."""
    fake_hf.populate = {}  # pretends to succeed but writes nothing
    with pytest.raises(BootError, match=r"model-facts\.md"):
        ensure_models(CacheConfig())


def test_local_files_only_hint_in_cold_start_error(
    volume: Path, fake_hf: SnapshotRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> str:
        raise OSError("no network")

    # Patch the module attribute, not the instance: Python resolves `__call__`
    # on the type, so an instance-level setattr here would be silently ignored.
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", boom)
    monkeypatch.setenv("HF_LOCAL_FILES_ONLY", "true")
    with pytest.raises(BootError, match="HF_LOCAL_FILES_ONLY"):
        ensure_models(CacheConfig())


def test_snapshot_download_receives_volume_cache_dir(volume: Path, fake_hf: SnapshotRecorder) -> None:
    fake_hf.populate = {
        "m-a-p/YuE2-3B": REQUIRED_MODEL_FILES,
        "m-a-p/YuE2-Vae": REQUIRED_VAE_FILES,
    }
    ensure_models(CacheConfig())
    # Every download must land under the volume, never the container disk.
    for call in fake_hf.calls:
        assert str(volume) in call["local_dir"]
        assert str(volume) in call["cache_dir"]


def test_allow_patterns_exclude_demo_assets(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """A cold start must not pull the repo's demo audio and artwork.

    `m-a-p/YuE2-3B` ships sample MP3s under `assets/audio/`, a PDF and a logo.
    Without `allow_patterns`, `snapshot_download` fetches all of it — megabytes
    of cold-start transfer the worker never reads.
    """
    fake_hf.populate = {
        "m-a-p/YuE2-3B": REQUIRED_MODEL_FILES,
        "m-a-p/YuE2-Vae": REQUIRED_VAE_FILES,
    }
    ensure_models(CacheConfig())

    for call in fake_hf.calls:
        patterns = call["allow_patterns"]
        assert patterns, "every download must be pattern-limited"
        assert not any(p.startswith("assets/") for p in patterns)
        # The weight files themselves must still be included.
        assert "model.safetensors" in patterns


def test_model_patterns_still_include_the_wheel(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """The `yue2_infer` wheel lives in this repo; the image build needs it cached."""
    fake_hf.populate = {
        "m-a-p/YuE2-3B": REQUIRED_MODEL_FILES,
        "m-a-p/YuE2-Vae": REQUIRED_VAE_FILES,
    }
    ensure_models(CacheConfig())
    model_call = next(c for c in fake_hf.calls if c["repo_id"] == "m-a-p/YuE2-3B")
    assert "*.whl" in model_call["allow_patterns"]


# --- wheel -------------------------------------------------------------------


def test_install_model_wheel_finds_the_cached_wheel(volume: Path) -> None:
    wheel = volume / "models" / "YuE2-3B" / MODEL_WHEEL
    wheel.write_bytes(b"PK")
    assert install_model_wheel(CacheConfig()) == wheel


def test_install_model_wheel_explains_where_it_comes_from(volume: Path) -> None:
    with pytest.raises(BootError, match="ships inside"):
        install_model_wheel(CacheConfig())


# --- health ------------------------------------------------------------------


def test_health_reports_a_populated_volume(volume: Path) -> None:
    populate_volume(volume)
    report = health()
    assert report["volume_mounted"] is True
    assert report["model_files_missing"] == []
    assert report["vae_files_missing"] == []


def test_health_lists_missing_files_on_an_empty_volume(volume: Path) -> None:
    report = health()
    assert report["volume_mounted"] is True
    assert "model.safetensors" in report["model_files_missing"]


def test_health_does_not_require_a_gpu(volume: Path) -> None:
    """A health probe that allocates VRAM can fail for the wrong reason."""
    assert "cuda_visible_devices" in health()


# --- env plumbing ------------------------------------------------------------


def test_apply_hf_env_points_every_hf_path_at_the_volume(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    CacheConfig().apply_hf_env()
    import os

    assert os.environ["HF_HOME"].startswith(str(volume))
    assert os.environ["HUGGINGFACE_HUB_CACHE"].startswith(str(volume))
    assert os.environ["HF_XET_CACHE"].startswith(str(volume))


def test_load_pipeline_reports_missing_wheel_module(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the wheel installed, the error must say so rather than NameError."""
    populate_volume(volume)
    monkeypatch.setitem(sys.modules, "yue2", None)
    with pytest.raises(BootError, match="yue2_infer is not installed"):
        boot.load_pipeline(None, cache=CacheConfig())
