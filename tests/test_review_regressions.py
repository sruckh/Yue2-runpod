"""Regression tests for defects found by adversarial review.

Each test here exists because a specific bug shipped and was **not** caught by
the existing suite. They are grouped by the reason they were missed, because in
every case the miss was more interesting than the bug:

- the test double disagreed with the real artifact format, so the suite could
  not see what the handler was doing to it;
- the assertion checked that a string appeared, not that it was true;
- the behaviour had no test at all because it was assumed to be free.

Keeping them together makes the failure mode legible: a green run here means
these specific traps are still guarded, not merely that the code compiles.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeSong

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
    """A booted `handler` module, without a GPU or the runpod SDK.

    `handler` boots at module scope and ends by calling
    `runpod.serverless.start`, so both the loader and the SDK are stubbed before
    the import.
    """
    import boot

    monkeypatch.setattr(boot, "load_pipeline", lambda *a, **k: fake_pipeline)
    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(start=lambda *a, **k: None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod", fake)
    module = importlib.import_module("handler")
    importlib.reload(module)
    yield module
    module._pipeline = None
    module._boot_error = None


class RecordingStorage:
    def __init__(self) -> None:
        self.artifacts: dict[str, Any] = {}

    def upload_directory(self, directory: Path, prefix: str) -> dict[str, Any]:
        from storage import UploadedArtifact

        for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
            rel = path.relative_to(directory).as_posix()
            self.artifacts[rel] = UploadedArtifact(
                key=f"{prefix}/{rel}",
                url=f"https://b2.example.invalid/{prefix}/{rel}",
                size_bytes=path.stat().st_size,
                content_type="application/octet-stream",
            )
        return self.artifacts


@pytest.fixture
def storage() -> RecordingStorage:
    return RecordingStorage()


# =============================================================================
# Trap 1: the test double disagreed with reality
# =============================================================================


def test_the_double_writes_the_real_truncated_shape() -> None:
    """The double must write a DICT, because the real `save_artifacts` does.

    `SongResult.truncated` is a property returning
    `{"abc": bool, "semantic": bool}` (pipeline.py:90-91). The double originally
    wrote `False`, so every test exercising a *completed* song was running
    against an artifact set the pipeline never produces.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        FakeSong().save_artifacts(d)
        written = json.loads((Path(d) / "result.json").read_text())
    assert isinstance(written["truncated"], dict)
    assert set(written["truncated"]) == {"abc", "semantic"}


def test_completed_song_is_not_reported_as_truncated(handler_module: Any, storage: RecordingStorage) -> None:
    """The bug this whole module exists for.

    `bool({"abc": False, "semantic": False})` is `True`, so a handler coercing
    the dict with `bool()` reported *every successfully generated song* as
    truncated. A caller retrying on `truncated: true` would retry forever.
    """
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=storage)
    assert result["truncated"] is False, "a complete song must not be reported as truncated"
    assert result["truncation_by_stage"] == {"abc": False, "semantic": False}


def test_a_genuinely_truncated_stage_is_reported(
    handler_module: Any, monkeypatch: pytest.MonkeyPatch, storage: RecordingStorage
) -> None:
    """And the signal still works in the other direction."""
    truncated_song = FakeSong(truncated_abc=True)
    monkeypatch.setattr(handler_module, "get_pipeline", lambda *a, **k: lambda **kw: truncated_song)
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=storage)
    assert result["truncated"] is True
    assert result["truncation_by_stage"] == {"abc": True, "semantic": False}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"abc": False, "semantic": False}, False),
        ({"abc": True, "semantic": False}, True),
        ({"abc": False, "semantic": True}, True),
        ({"abc": True, "semantic": True}, True),
        (None, None),
        (False, False),
        (True, True),
    ],
)
def test_truncation_normalisation(handler_module: Any, raw: Any, expected: Any) -> None:
    """Derived from the stage flags, never the container's truthiness."""
    assert handler_module._normalise_truncation(raw)["truncated"] is expected


def test_missing_truncation_is_reported_as_unknown(handler_module: Any) -> None:
    """`None` means "not reported", which is not the same as "not truncated"."""
    assert handler_module._normalise_truncation(None)["truncated"] is None


# =============================================================================
# Trap 2: the assertion checked a string appeared, not that it was true
# =============================================================================


def test_reported_decoder_is_the_one_the_pipeline_was_built_with(
    handler_module: Any, storage: RecordingStorage
) -> None:
    """`decoder` must describe reality, not echo an env var back.

    The old field read `YUE2_VAE` — an env var with **zero hits** anywhere in
    the `yue2_infer` wheel. Setting it changed nothing, yet the response
    reported the name it claimed. The old test asserted only that the string
    appeared in the result, which is exactly the assertion that cannot fail.
    """
    from config import CacheConfig, WorkerConfig

    cfg = WorkerConfig(cache=CacheConfig(), vae_repo="m-a-p/YuE2-Vae-legacy")
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=storage, config=cfg)
    assert result["decoder"] == "m-a-p/YuE2-Vae-legacy"


def test_decoder_is_not_taken_from_the_bare_environment(
    handler_module: Any, storage: RecordingStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray `YUE2_VAE` in the environment must not change the reported value."""
    monkeypatch.setenv("YUE2_VAE", "nonsense/not-a-real-vae")
    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=storage)
    assert result["decoder"] == "m-a-p/YuE2-Vae"


# =============================================================================
# Trap 3: assumed to be free, never tested
# =============================================================================


def test_import_boots_then_hands_the_handler_to_runpod(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The standard RunPod startup shape, in order.

    This is the contract the platform expects and the one the reference workers
    use: configure at module scope, then `runpod.serverless.start({"handler": ...})`.
    Boot must come **first** — a lazy first load would put a ~12 GB download plus
    model construction inside a job's own timeout budget, so the first job on a
    cold volume would be killed for taking longer than a generation is allowed.

    This test also pins the *shape*, because an earlier version wrapped startup in
    a custom `main()`. That worked, but it hid `runpod.serverless.start` inside a
    function where nobody looks for it — and RunPod's SDK is what discovers
    `--test_input`, so the wrapper was redundant as well as obscure.
    """
    import boot

    calls: list[str] = []
    started_with: list[Any] = []

    def fake_load(*a: Any, **k: Any) -> Any:
        calls.append("boot")
        return object()

    def fake_start(config: Any) -> None:
        calls.append("start")
        started_with.append(config)

    monkeypatch.setattr(boot, "load_pipeline", fake_load)
    fake_runpod = types.ModuleType("runpod")
    fake_runpod.serverless = types.SimpleNamespace(start=fake_start)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod", fake_runpod)

    module = importlib.import_module("handler")
    importlib.reload(module)

    assert calls == ["boot", "start"], f"expected boot then start, got {calls}"
    assert started_with and "handler" in started_with[0], "start() must be given the handler"
    assert started_with[0]["handler"] is module.handler
    module._pipeline = None


def test_handler_is_a_module_level_callable(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`handler` must be importable by name — that is what `start()` receives."""
    import boot

    monkeypatch.setattr(boot, "load_pipeline", lambda *a, **k: object())
    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(start=lambda *a, **k: None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod", fake)

    module = importlib.import_module("handler")
    importlib.reload(module)

    assert callable(module.handler)
    assert module.handler.__name__ == "handler"
    module._pipeline = None


def test_a_failed_boot_is_not_retried(handler_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """One cold start, not one per job."""
    from boot import BootError

    attempts: list[int] = []

    def boom(*a: Any, **k: Any) -> Any:
        attempts.append(1)
        raise BootError("volume unavailable")

    monkeypatch.setattr(handler_module.boot, "load_pipeline", boom)
    handler_module._pipeline = None
    handler_module.boot_worker()

    for _ in range(3):
        handler_module.handler({"id": "j", "input": valid_input()})

    assert len(attempts) == 1


def test_workdirs_are_unique_per_invocation(handler_module: Any, volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two jobs sharing a request id must not share a directory.

    `id` is caller-supplied. With a workdir keyed on it alone, a second job with
    the same id `rmtree`s the first job's artifacts mid-flight and uploads to the
    same object prefix — silent corruption of both results, not a clean failure.
    """
    seen: list[Path] = []
    real_reset = handler_module._reset_dir

    def recording_reset(path: Path) -> None:
        seen.append(path)
        real_reset(path)

    monkeypatch.setattr(handler_module, "_reset_dir", recording_reset)

    class Ok:
        def upload_directory(self, directory: Path, prefix: str) -> dict[str, Any]:
            from storage import UploadedArtifact

            return {
                "audio.flac": UploadedArtifact("k", "https://x/audio.flac", 1, "audio/flac"),
                "score.abc": UploadedArtifact("k2", "https://x/score.abc", 1, "text/vnd.abc"),
            }

    job = {"id": "same-job-id", "input": valid_input()}
    handler_module.run_create_job(job, storage=Ok())
    handler_module.run_create_job(job, storage=Ok())

    assert len(seen) == 2
    assert seen[0] != seen[1], "two invocations with the same id shared a workdir"


def test_response_is_built_before_cleanup(handler_module: Any, volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local run's `file://` URLs must point at files that still exist.

    Building the response after the `finally` cleanup returned URLs to deleted
    files — invisible with a real bucket, broken with local storage.
    """
    from storage import UploadedArtifact

    observed: dict[str, bool] = {}

    class Localish:
        def upload_directory(self, directory: Path, prefix: str) -> dict[str, Any]:
            audio = Path(directory) / "audio.flac"
            observed["existed_at_response_build"] = audio.is_file()
            return {
                "audio.flac": UploadedArtifact("k", f"file://{audio}", 1, "audio/flac"),
                "score.abc": UploadedArtifact("k2", f"file://{directory}/score.abc", 1, "text/vnd.abc"),
            }

    handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=Localish())
    assert observed["existed_at_response_build"] is True


def test_build_response_failure_is_a_structured_error(handler_module: Any, storage: RecordingStorage) -> None:
    """A response-assembly failure must not escape as a traceback."""

    class NoAudio:
        def upload_directory(self, directory: Path, prefix: str) -> dict[str, Any]:
            return {}  # nothing uploaded -> build_response raises

    result = handler_module.run_create_job({"id": "j", "input": valid_input()}, storage=NoAudio())
    assert "error" in result
    assert "no audio artifact" in result["error"]


def test_unwritable_workdir_is_a_named_failure(tmp_path: Path) -> None:
    """An unwritable volume must name itself, not surface as "Unhandled error".

    `_reset_dir` originally sat outside the try, so an `OSError` here escaped
    the named-failure contract: every failure is reported, never a bare traceback.

    This exercises the real `_reset_dir` — replacing it with a function that
    raises would bypass the very wrapping under test and prove nothing. A file
    where the directory should go makes `mkdir` raise a genuine `OSError`.
    """
    from handler import WorkerError, _reset_dir

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("in the way")

    with pytest.raises(WorkerError, match="work directory"):
        _reset_dir(blocker / "workdir")


def test_reset_dir_wraps_the_oserror_it_raises(tmp_path: Path) -> None:
    """The wrapper is what puts the failure inside the handler's catch set."""
    from handler import WorkerError, _reset_dir

    blocker = tmp_path / "blocker"
    blocker.write_text("x")

    with pytest.raises(WorkerError) as excinfo:
        _reset_dir(blocker / "child")

    # The original cause is preserved for the log, not swallowed.
    assert isinstance(excinfo.value.__cause__, OSError)


# =============================================================================
# The prompt budget: the pipeline's limit is tokens; ours is a byte pre-check
# =============================================================================


def test_oversized_prompt_is_rejected_before_the_pipeline(
    handler_module: Any, fake_pipeline: Any, storage: RecordingStorage
) -> None:
    """The pipeline refuses a long prompt with `ValueError` only after loading.

    `sampling.py:62` raises when `len(prefix) + max_tokens > 24576`. Our byte
    cap is a conservative pre-check so the common case fails in microseconds
    rather than after ~70 seconds of model traffic.
    """
    from schema import _MAX_PROMPT_BYTES

    big = "word " * (_MAX_PROMPT_BYTES // 4)
    result = handler_module.run_create_job({"id": "j", "input": valid_input(lyrics=big)}, storage=storage)

    assert "error" in result
    assert "context window" in result["error"]
    assert fake_pipeline.calls == [], "the pipeline was invoked for a prompt it cannot accept"


def test_a_normal_song_still_passes_the_prompt_budget(handler_module: Any, storage: RecordingStorage) -> None:
    lyrics = "[Verse]\n" + "word " * 500
    result = handler_module.run_create_job({"id": "j", "input": valid_input(lyrics=lyrics)}, storage=storage)
    assert "error" not in result


def test_style_and_lyrics_share_one_budget(handler_module: Any, storage: RecordingStorage) -> None:
    """Each field can be under the cap while the sum is over it."""
    from schema import _MAX_PROMPT_BYTES

    half = "x" * (_MAX_PROMPT_BYTES // 2 + 1024)
    result = handler_module.run_create_job({"id": "j", "input": valid_input(style=half, lyrics=half)}, storage=storage)
    assert "error" in result


def test_abc_counts_toward_the_prompt_budget(handler_module: Any, storage: RecordingStorage) -> None:
    """A supplied score is part of the prompt, so it consumes the same budget."""
    from schema import _MAX_PROMPT_BYTES

    abc = "X:1\n" + "C D E F |" * (_MAX_PROMPT_BYTES // 8)
    result = handler_module.run_create_job({"id": "j", "input": valid_input(abc=abc, cot="full")}, storage=storage)
    assert "error" in result
    assert "context window" in result["error"]
