"""Handler tests — the full create-mode job path with GPU and B2 faked.

These are the tests that would catch a regression in the real API call. The
assertions on `fake_pipeline.calls` encode the `YuE2Pipeline.__call__` contract
verified directly from the upstream `yue2_infer-0.1.5` wheel, so if the
handler ever starts passing an argument the pipeline does not accept — or stops
passing one it requires — a test fails here rather than on a paid GPU.

`handler` is imported via importlib with `VOLUME_ROOT` already set, because
`handler.py` reads the HF cache env at import time by design.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = REPO_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


def valid_input(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"style": "City Pop, upbeat", "lyrics": "[Verse]\nhello"}
    payload.update(overrides)
    return payload


@pytest.fixture
def handler_module(volume: Path, fake_pipeline: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import `handler` with the pipeline stubbed out.

    `boot.load_pipeline` is patched *before* the module's lazy `get_pipeline()`
    ever runs, so no torch import, no weights, no GPU.
    """
    module = importlib.import_module("handler")
    importlib.reload(module)
    monkeypatch.setattr(module.boot, "load_pipeline", lambda *a, **k: fake_pipeline)
    module._pipeline = None
    yield module
    module._pipeline = None


class RecordingStorage:
    """Storage double that keeps artifacts and mints predictable URLs."""

    def __init__(self) -> None:
        self.prefixes: list[str] = []
        self.artifacts: dict[str, Any] = {}

    def upload_directory(self, directory: Path, prefix: str) -> dict[str, Any]:
        from storage import UploadedArtifact

        self.prefixes.append(prefix)
        for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
            relative = path.relative_to(directory).as_posix()
            self.artifacts[relative] = UploadedArtifact(
                key=f"{prefix}/{relative}",
                url=f"https://b2.example.invalid/{prefix}/{relative}",
                size_bytes=path.stat().st_size,
                content_type="application/octet-stream",
            )
        return self.artifacts


@pytest.fixture
def recording_storage() -> RecordingStorage:
    return RecordingStorage()


# --- the happy path ----------------------------------------------------------


def test_create_job_returns_the_documented_response_shape(handler_module: Any, recording_storage: Any) -> None:
    result = handler_module.run_create_job({"id": "job-1", "input": valid_input()}, storage=recording_storage)

    assert "error" not in result
    # The exact keys the Phase 1 contract names.
    for key in ("audio_url", "score_abc_url", "duration", "seed", "cot", "decoder", "timings"):
        assert key in result, f"response is missing the contracted key {key!r}"
    assert result["audio_url"].startswith("https://")
    assert result["score_abc_url"].startswith("https://")
    assert result["seed"] == 831001
    assert result["cot"] == "full"
    assert result["duration"] == pytest.approx(214.85)


def test_response_carries_urls_and_never_raw_audio(handler_module: Any, recording_storage: Any) -> None:
    result = handler_module.run_create_job({"id": "job-2", "input": valid_input()}, storage=recording_storage)

    blob = json.dumps(result)
    # Locked decision 5: the response is metadata, not payload.
    assert "fLaC" not in blob
    assert not any(isinstance(v, bytes) for v in result.values())


def test_every_artifact_is_uploaded(handler_module: Any, recording_storage: Any) -> None:
    handler_module.run_create_job({"id": "job-3", "input": valid_input()}, storage=recording_storage)

    # Whatever `save_artifacts` writes must reach storage; these are the ones
    # the real pipeline produces.
    assert {"audio.flac", "score.abc", "result.json"} <= set(recording_storage.artifacts)


def test_upload_prefix_is_scoped_to_the_job(handler_module: Any, recording_storage: Any) -> None:
    handler_module.run_create_job({"id": "job-abc", "input": valid_input()}, storage=recording_storage)
    # job id and request id stay separate path segments.
    assert recording_storage.prefixes == ["jobs/job-abc/job-abc"]


# --- the pipeline contract ---------------------------------------------------


def test_pipeline_called_with_exactly_the_documented_kwargs(
    handler_module: Any, fake_pipeline: Any, recording_storage: Any
) -> None:
    """Encodes `YuE2Pipeline.__call__(style, lyrics, *, tags, ...)`'s accepted args."""
    handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)

    assert len(fake_pipeline.calls) == 1
    call = fake_pipeline.calls[0]
    assert call["style"] == "City Pop, upbeat"
    assert call["lyrics"] == "[Verse]\nhello"
    assert call["cot"] == "full"
    assert call["seed"] == 831001
    # Optional args must be *absent*, not None: the pipeline treats `abc=None`
    # differently from an omitted abc.
    assert "abc" not in call
    assert "cfg_scale" not in call


def test_cfg_scale_forwarded_only_when_supplied(
    handler_module: Any, fake_pipeline: Any, recording_storage: Any
) -> None:
    handler_module.run_create_job({"id": "j", "input": valid_input(cfg_scale=1.2)}, storage=recording_storage)
    assert fake_pipeline.calls[0]["cfg_scale"] == 1.2


def test_abc_forwarded_for_a_external_score(handler_module: Any, fake_pipeline: Any, recording_storage: Any) -> None:
    score = "X:1\nK:C\nC D E F|"
    handler_module.run_create_job({"id": "j", "input": valid_input(abc=score, cot="melody")}, storage=recording_storage)
    assert fake_pipeline.calls[0]["abc"] == score
    assert fake_pipeline.calls[0]["cot"] == "melody"


def test_save_artifacts_is_used_not_save(handler_module: Any, fake_song: Any, recording_storage: Any) -> None:
    """`save()` alone would lose score.abc and result.json."""
    handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    assert fake_song.saved_to is not None


# --- error handling ----------------------------------------------------------


def test_missing_input_returns_error_not_exception(handler_module: Any, recording_storage: Any) -> None:
    result = handler_module.run_create_job({"id": "j"}, storage=recording_storage)
    assert "error" in result
    assert "input" in result["error"]


def test_invalid_seed_returns_error_without_generating(
    handler_module: Any, fake_pipeline: Any, recording_storage: Any
) -> None:
    result = handler_module.run_create_job({"id": "j", "input": valid_input(seed="nope")}, storage=recording_storage)
    assert "error" in result
    # The whole point of validating early: no GPU work happened.
    assert fake_pipeline.calls == []


def test_cover_mode_rejected_with_stage_03_message(handler_module: Any, recording_storage: Any) -> None:
    result = handler_module.run_create_job({"id": "j", "input": valid_input(mode="cover")}, storage=recording_storage)
    assert "error" in result
    assert "Stage 03" in result["error"]


def test_generation_failure_is_reported_not_raised(
    handler_module: Any, recording_storage: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Exploding:
        def __call__(self, **kwargs: Any) -> Any:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(handler_module, "get_pipeline", lambda *a, **k: Exploding())
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    assert "error" in result
    assert "Generation failed" in result["error"]


def test_boot_failure_is_reported_not_raised(
    handler_module: Any, recording_storage: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold-start failure (missing wheel, unreadable volume) is a job error.

    Regression guard: `BootError` was originally absent from the handler's catch
    tuple, so this path escaped `run_create_job` as a traceback.
    """
    from boot import BootError

    def boom(*a: Any, **k: Any) -> Any:
        raise BootError("yue2_infer is not installed")

    monkeypatch.setattr(handler_module.boot, "load_pipeline", boom)
    handler_module._pipeline = None
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    assert "error" in result
    assert "yue2_infer is not installed" in result["error"]


def test_storage_failure_is_reported_not_raised(
    handler_module: Any, recording_storage: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storage import StorageError

    def boom(directory: Path, prefix: str) -> dict[str, Any]:
        raise StorageError("bucket unreachable")

    monkeypatch.setattr(recording_storage, "upload_directory", boom)
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    assert "error" in result
    assert "bucket unreachable" in result["error"]


def test_missing_storage_config_names_the_env_vars(handler_module: Any) -> None:
    """No B2 env, no injected storage — the error must be actionable."""
    result = handler_module.run_create_job({"id": "j", "input": valid_input()})
    assert "error" in result
    assert "B2_ENDPOINT_URL" in result["error"]


def test_missing_storage_fails_before_generating(handler_module: Any, fake_pipeline: Any) -> None:
    """A misconfigured bucket must not cost a 70-second GPU generation.

    Regression guard: storage was originally resolved after `generate()`, so an
    unset credential produced a song and only then failed to deliver it.
    """
    result = handler_module.run_create_job({"id": "j", "input": valid_input()})
    assert "error" in result
    assert fake_pipeline.calls == [], "generated before checking whether we could deliver"


# --- workdir hygiene ---------------------------------------------------------


def test_workdir_is_cleaned_up_after_upload(handler_module: Any, recording_storage: Any, volume: Path) -> None:
    handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    scratch = volume / "scratch"
    leftovers = list(scratch.iterdir()) if scratch.is_dir() else []
    assert leftovers == [], f"job left {leftovers} on the container disk"


def test_workdir_cleaned_up_even_when_upload_fails(
    handler_module: Any, recording_storage: Any, volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from storage import StorageError

    def boom(directory: Path, prefix: str) -> dict[str, Any]:
        raise StorageError("nope")

    monkeypatch.setattr(recording_storage, "upload_directory", boom)
    handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    scratch = volume / "scratch"
    assert list(scratch.iterdir()) == []


# --- decoder reporting -------------------------------------------------------


def test_decoder_defaults_to_the_current_vae(
    handler_module: Any, recording_storage: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("YUE2_VAE", raising=False)
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    assert result["decoder"] == "m-a-p/YuE2-Vae"


def test_legacy_decoder_is_reported_when_selected(
    handler_module: Any, recording_storage: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YUE2_VAE", "m-a-p/YuE2-Vae-legacy")
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=recording_storage)
    assert result["decoder"] == "m-a-p/YuE2-Vae-legacy"


# --- direct handler entrypoint ----------------------------------------------


def test_handler_wraps_run_create_job(
    handler_module: Any, recording_storage: Any, monkeypatch: pytest.MonkeyPatch, b2_env: None
) -> None:
    """End-to-end through the real `handler()` entrypoint, storage swapped."""
    monkeypatch.setattr(handler_module, "B2Storage", lambda *a, **k: recording_storage)
    result = handler_module.handler({"id": "job-x", "input": valid_input()})
    assert "audio_url" in result


def test_handler_never_raises_on_a_bad_job(handler_module: Any) -> None:
    """A malformed job must produce a result dict, not kill the worker."""
    assert "error" in handler_module.handler({"id": "j", "input": {"style": 5, "lyrics": None}})
    assert "error" in handler_module.handler({})
