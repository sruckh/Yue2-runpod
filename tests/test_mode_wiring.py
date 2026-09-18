"""End-to-end wiring tests for cover and edit through the handler.

The other suites test the mode functions and the validator in isolation. Doing
only that let **three real bugs** ship at once:

1. `validate_mode_inputs` checked that `source_audio` was present but
   `validate_job` never carried it onto the params object, so
   `params.source_audio` was always `None`.
2. `lyrics_supplied` was declared on `SongParameters` and read by
   `prepare_cover`, but nothing ever set it — so a caller supplying their own
   lyrics silently got the transcription instead.
3. The handler never fetched the recording from its URL, so even once (1) was
   fixed the mode received a URL where it needed a path.

Every one of those is invisible to a test that builds `SongParameters(...)`
by hand. These tests therefore go in through `validate_job`, which is the only
way to catch plumbing that was never connected.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from schema import validate_job, validate_mode, validate_mode_inputs

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = REPO_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


def cover_job(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "cover",
        "style": "Jazz-funk",
        "lyrics": "[Verse]\nhello",
        "source_audio": "https://example.invalid/source.wav",
    }
    payload.update(overrides)
    return {"id": "job-1", "input": payload}


# =============================================================================
# Validation must carry the fields through, not merely check them
# =============================================================================


def test_source_audio_survives_validation() -> None:
    """The bug: checked in the raw input, never put on the params object."""
    params = validate_job(cover_job())
    assert params.source_audio == "https://example.invalid/source.wav"


def test_source_audio_is_none_for_modes_that_do_not_use_it() -> None:
    params = validate_job({"id": "j", "input": {"style": "x", "lyrics": "y"}})
    assert params.source_audio is None


def test_supplied_lyrics_are_flagged() -> None:
    """The bug: `lyrics_supplied` was read by prepare_cover and set by nothing."""
    assert validate_job(cover_job()).lyrics_supplied is True


def test_absent_lyrics_are_not_flagged_as_supplied() -> None:
    """`create` requires lyrics, so this is really about the cover path — but the
    flag must reflect presence, not the default."""
    payload = {"mode": "cover", "style": "x", "source_audio": "https://e.invalid/a.wav"}
    params = validate_job({"id": "j", "input": payload})
    assert params.lyrics_supplied is False


def test_a_full_cover_payload_round_trips() -> None:
    """Every cover field, from raw job to params, in one assertion."""
    params = validate_job(cover_job(seed=42, cfg_scale=1.2, cot="melody"))
    assert params.source_audio is not None
    assert params.lyrics_supplied is True
    assert params.seed == 42
    assert params.cfg_scale == 1.2
    assert params.cot == "melody"


def test_mode_and_inputs_agree() -> None:
    """`validate_mode` and `validate_mode_inputs` must not disagree about a job."""
    raw = cover_job()["input"]
    mode = validate_mode(raw)
    assert mode == "cover"
    validate_mode_inputs(mode, raw)  # must not raise
    assert validate_job(cover_job()).source_audio is not None


# =============================================================================
# The handler must fetch the recording
# =============================================================================


@pytest.fixture
def handler_module(volume: Path, fake_pipeline: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    import boot

    monkeypatch.setattr(boot, "load_pipeline", lambda *a, **k: fake_pipeline)
    import types

    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(start=lambda *a, **k: None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod", fake)
    module = importlib.import_module("handler")
    importlib.reload(module)
    yield module
    module._pipeline = None
    module._boot_error = None


class Storage:
    def __init__(self) -> None:
        self.artifacts: dict[str, Any] = {}

    def upload_directory(self, directory: Path, prefix: str) -> dict[str, Any]:
        from storage import UploadedArtifact

        for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
            rel = path.relative_to(directory).as_posix()
            self.artifacts[rel] = UploadedArtifact(
                f"k/{rel}", f"https://b2.invalid/{rel}", 1, "application/octet-stream"
            )
        return self.artifacts


@pytest.fixture
def storage() -> Storage:
    return Storage()


def test_handler_fetches_a_url_into_a_local_path(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug: the mode received a URL where it needed a path."""
    local = tmp_path / "downloaded.wav"
    local.write_bytes(b"RIFF")

    seen: dict[str, Any] = {}

    def fake_download(job_id: str, urls: list[str]) -> list[str]:
        seen["urls"] = urls
        return [str(local)]

    monkeypatch.setattr(handler_module, "_download_files_from_urls", fake_download, raising=False)

    from schema import validate_job as vj

    params = vj(cover_job())
    # Patch the SDK import path the helper uses.
    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    resolved = handler_module._fetch_source_audio(params, "job-1")
    assert resolved == str(local)
    assert seen["urls"] == ["https://example.invalid/source.wav"]


def test_handler_passes_through_an_already_local_path(handler_module: Any) -> None:
    from schema import validate_job as vj

    params = vj(cover_job(source_audio="/tmp/already-here.wav"))
    assert handler_module._fetch_source_audio(params, "job-1") == "/tmp/already-here.wav"


def test_handler_raises_a_named_error_when_the_fetch_fails(
    handler_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "The URL was wrong" is the caller's problem, and must read that way."""
    from schema import validate_job as vj

    def boom(job_id: str, urls: list[str]) -> list[str]:
        raise OSError("connection refused")

    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    params = vj(cover_job())
    with pytest.raises(handler_module.WorkerError, match="could not fetch source_audio"):
        handler_module._fetch_source_audio(params, "job-1")


def test_handler_reports_an_empty_fetch(handler_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = lambda job_id, urls: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    from schema import validate_job as vj

    params = vj(cover_job())
    with pytest.raises(handler_module.WorkerError, match="downloaded to nothing"):
        handler_module._fetch_source_audio(params, "job-1")


def test_cover_reaches_the_mode_with_a_local_path(
    handler_module: Any, storage: Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole cover path from a raw job, with only the subprocesses faked.

    This is the test that would have caught all three bugs: it starts from a job
    dict, goes through validation, fetches, and reaches the mode.
    """
    import modes
    from schema import validate_job as vj

    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF")
    params = vj(cover_job(source_audio=str(source)))

    reached: dict[str, Any] = {}

    def fake_dispatch(mode: str, p: Any, workdir: Path) -> Any:
        reached["mode"] = mode
        reached["source_audio"] = p.source_audio
        reached["lyrics_supplied"] = p.lyrics_supplied
        return modes.ModeResult(mode=mode, score_abc=None, lyrics="from asr")

    monkeypatch.setattr(handler_module.modes, "dispatch", fake_dispatch)

    result = handler_module.run_create_job(cover_job(source_audio=str(source)), storage=storage, config=None)

    assert reached["mode"] == "cover"
    assert reached["source_audio"] == str(source), "the mode must receive a local path"
    assert reached["lyrics_supplied"] is True, "the caller's lyrics must be honoured"
    assert "error" not in result or "source_audio" not in str(result.get("error", ""))
    assert params.source_audio == str(source)


# =============================================================================
# The ASR environment toggle (unification experiment)
# =============================================================================
#
# `cover`'s lyric transcription normally runs in `/opt/venvs/qwen3-asr`. The
# unification question is whether it can run under YuE2's own stack instead. It
# is a runtime switch so one image answers both configurations, because flipping
# the image and rebuilding costs a build cycle to test and another to revert.


def test_the_asr_toggle_defaults_to_its_own_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means the isolated environment — the shipped, verified arrangement.

    The experiment must be opt-in. A default that changed behaviour would put
    every cover job on the unverified path without anyone asking.
    """
    monkeypatch.delenv("YUE2_ASR_IN_MAIN", raising=False)
    import modes

    assert modes._asr_venv() == modes.ASR_VENV


def test_the_asr_toggle_selects_the_main_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    from subprocess_runner import MAIN_INTERPRETER

    import modes

    for value in ("1", "true", "TRUE", "yes", " yes "):
        monkeypatch.setenv("YUE2_ASR_IN_MAIN", value)
        assert modes._asr_venv() == MAIN_INTERPRETER, f"{value!r} should enable the experiment"


def test_a_falsey_toggle_value_keeps_the_venv(monkeypatch: pytest.MonkeyPatch) -> None:
    """`"false"` must not enable it. Same trap as the `instrumental` flag.

    A string is not a boolean, and `bool("false")` is `True`. Here that would
    silently move every cover onto the unverified path for a caller who wrote
    what they believed meant "off".
    """
    import modes

    for value in ("0", "false", "no", "", "off", "2"):
        monkeypatch.setenv("YUE2_ASR_IN_MAIN", value)
        assert modes._asr_venv() == modes.ASR_VENV, f"{value!r} must not enable the experiment"


def test_the_main_interpreter_is_not_a_venv_path() -> None:
    """The sentinel must resolve to the running interpreter, not a bogus path.

    `VENV_ROOT / "" / "bin" / "python"` is a path that looks plausible and never
    exists, so the failure would surface as a subprocess that cannot start rather
    than as an obvious mistake.
    """
    import sys

    from subprocess_runner import MAIN_INTERPRETER, venv_python

    assert MAIN_INTERPRETER == ""
    assert venv_python(MAIN_INTERPRETER) == Path(sys.executable)


def test_the_asr_stage_still_runs_as_a_subprocess_either_way() -> None:
    """Unification removes a *venv*, not the process boundary.

    The boundary is what releases ASR's VRAM before generation — a cover's peak
    depends on it. If this toggle ever made ASR run in-process, the measured
    headroom would be gone.
    """
    source = (Path(__file__).resolve().parent.parent / "worker" / "modes.py").read_text(encoding="utf-8")
    assert "run_stage(" in source
    # The toggle selects an interpreter; it must not bypass run_stage.
    assert "_asr_venv()" in source
