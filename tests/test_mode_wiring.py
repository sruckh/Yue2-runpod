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

import base64
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from schema import validate_job, validate_mode, validate_mode_inputs

#: A real (if silent) WAV header — 44 bytes, no samples.
#:
#: Fixtures here used to write `b"RIFF"` as a stand-in for "some audio". That is a
#: *truncated* file, and `audio_probe` now refuses it before a transcription model
#: loads on it — correctly, since 4 bytes cannot be decoded. A fixture that is not
#: real audio makes a test pass for the wrong reason.
_MINIMAL_WAV = (
    b"RIFF"
    + (36).to_bytes(4, "little")
    + b"WAVEfmt "
    + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little")
    + (1).to_bytes(2, "little")
    + (24000).to_bytes(4, "little")
    + (48000).to_bytes(4, "little")
    + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little")
    + b"data"
    + (0).to_bytes(4, "little")
)

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
    # A real (if silent) WAV header. `b"RIFF"` alone is a truncated file, which
    # `audio_probe` now refuses before a model loads on it — see
    # tests/test_audio_probe.py. The fixture has to be actual audio.
    local = tmp_path / "downloaded.wav"
    local.write_bytes(_MINIMAL_WAV)

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


def test_handler_passes_through_an_already_local_path(handler_module: Any, tmp_path: Path) -> None:
    """A local path is used as-is, with no fetch.

    The file has to exist and be real audio: the probe reads it on both routes,
    so a path to nothing is now a named failure rather than a silent pass-through
    (which is the point — a wrong local path used to reach SheetSage2 and fail
    there, 32 s later, with a decoder message).
    """
    from schema import validate_job as vj

    local = tmp_path / "already-here.wav"
    local.write_bytes(_MINIMAL_WAV)

    params = vj(cover_job(source_audio=str(local)))
    assert handler_module._fetch_source_audio(params, "job-1") == str(local)


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
    source.write_bytes(_MINIMAL_WAV)
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
# Both model families run in the main environment
# =============================================================================
#
# The unification experiment is answered for both, verified on hardware:
#
#   ASR        job 0576cb7f, v25   reported torch 2.10.0+cu128
#   SheetSage2 job f5792fce, v27   reported torch 2.10.0+cu128, transformers 4.57.6
#
# So the two venvs were deleted and these stages now run in the ONE environment.
# The `run_stage` parameter remains — a future model family whose pins genuinely
# conflict can still be given its own venv — but nothing uses it today, and these
# tests pin that, because "one environment" is now a claim the image depends on.


def test_both_stages_run_in_the_main_interpreter() -> None:
    """No venv indirection left for either stage.

    If a stage silently went back to a venv, the image would fail at runtime —
    the venv is no longer built — and the failure would name a missing interpreter
    path rather than a missing environment.

    This checks the *constant* the stages use, not just the absent string
    `/opt/venvs/...`. An earlier version of this guard only scanned for that
    path, so setting `_ASR_ENV = "qwen3-asr"` — a venv name with no path in it —
    passed while pointing the ASR stage at an environment the image does not
    build. Caught by mutation, which is the only reason it is checked here.
    """
    from subprocess_runner import MAIN_INTERPRETER

    import modes

    assert modes._ASR_ENV == MAIN_INTERPRETER, (
        f"the ASR stage points at {modes._ASR_ENV!r}; the image builds only the main environment"
    )
    assert modes._SHEETSAGE_ENV == MAIN_INTERPRETER, (
        f"the SheetSage2 stage points at {modes._SHEETSAGE_ENV!r}; the image builds only the main environment"
    )
    assert modes._asr_venv() == MAIN_INTERPRETER
    assert modes._sheetsage_venv() == MAIN_INTERPRETER


def test_the_main_interpreter_is_the_running_one() -> None:
    """The sentinel must resolve to this interpreter, not a path under /opt/venvs."""
    import sys

    from subprocess_runner import MAIN_INTERPRETER, venv_python

    assert venv_python(MAIN_INTERPRETER) == Path(sys.executable)
    assert "/opt/venvs" not in str(venv_python(MAIN_INTERPRETER))


def test_no_code_names_a_venv_the_image_no_longer_builds() -> None:
    """The regression that matters after the collapse.

    The Dockerfile stopped building both venvs, so any code still naming one fails
    only on a GPU node — with an error about a missing interpreter path rather
    than a missing environment.
    """
    root = Path(__file__).resolve().parent.parent / "worker"
    offenders = []
    for path in root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "opt/venvs/sheetsage2" in line or "opt/venvs/qwen3-asr" in line:
                offenders.append(f"{path.relative_to(root)}: {line.strip()[:80]}")
    assert not offenders, "code still names a venv the image no longer builds:\n" + "\n".join(offenders)


def test_both_stages_still_report_their_environment() -> None:
    """The reporting stayed useful after the experiment.

    It is now how a job says which stack ran — and it is what made the experiment
    readable in the first place.
    """
    root = Path(__file__).resolve().parent.parent / "worker"
    for child in ("transcribe_asr/run.py", "transcribe_sheetsage/run.py"):
        assert "_environment()" in (root / child).read_text(encoding="utf-8")


def _stage_record_source(stage: str) -> str:
    """The `stages["<stage>"] = {…}` literal from modes.py, as written.

    Scoped to one stage on purpose. A file-wide substring check passes even after
    a field is deleted from one stage, because the other stage still has it —
    which is exactly how an earlier version of these tests reported success
    against a broken change.

    Slices to the matching close brace by brace-counting rather than to the first
    `}`, because the field values contain braces: `(asr.payload or {})`.
    """
    source = (Path(__file__).resolve().parent.parent / "worker" / "modes.py").read_text(encoding="utf-8")
    start = source.index(f'stages["{stage}"] = {{')
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced braces while reading the {stage} stage record")


def test_the_detected_language_reaches_the_response() -> None:
    """The child computed it; the parent must not drop it.

    `transcribe_asr/run.py` returns `{"text": ..., "language": ...}` and the
    language was being discarded between the child and `stages["asr"]` — computed
    work thrown away at the boundary. The official demo surfaces a language field;
    for a cover the only honest version is the detected one, since no caller sets
    it.
    """
    record = _stage_record_source("asr")
    assert '"language"' in record, "modes.py drops the ASR-detected language"


def test_the_language_comes_from_the_child_not_the_request() -> None:
    """It must be the payload's value, not a field the caller supplied.

    If this ever read from `params`, a cover would report the caller's assumption
    rather than what the recording actually contains.
    """
    record = _stage_record_source("asr")
    assert "asr.payload" in record, "the language does not come from the child's result"
    assert "params." not in record.split('"language"')[1].split("\n")[0], (
        "the language is read from the request, not the transcription"
    )


# =============================================================================
# The fetch identifies what arrived, before a model loads on it
# =============================================================================
#
# Written against a real job failure. A cover job's `source_audio` URL served
# something that was not audio, and the job died ~32 s in — after SheetSage2 had
# loaded — with ffmpeg's message, which is byte-for-byte identical for an empty
# file, an HTML page, a JSON body and a plain-text denial. The operator learned
# nothing, and paid a model load to learn it.
#
# These tests pin the two things that make the message useful: it names *what*
# arrived, and the remedy matches the kind.


@pytest.mark.parametrize(
    ("label", "body", "expected"),
    [
        ("html", b"<!DOCTYPE html><html><body>404 Not Found</body></html>", "HTML page"),
        ("json", b'{"error":"not found","status":404}', "JSON body"),
        ("plain", b"Access Denied", "plain text"),
        ("empty", b"", "no data"),
    ],
)
def test_a_non_audio_download_is_named_before_the_mode_runs(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, body: bytes, expected: str
) -> None:
    """The failure names what arrived, and says to check the URL."""
    served = tmp_path / "downloaded"  # no extension, as the SDK writes it
    served.write_bytes(body)

    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = lambda job_id, urls: [str(served)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    params = validate_job(cover_job())

    with pytest.raises(handler_module.WorkerError) as caught:
        handler_module._fetch_source_audio(params, "job-1")

    message = str(caught.value)
    assert expected in message, f"{label}: message does not name the kind — {message}"
    assert "Check that the URL serves the recording itself" in message
    # The failing production message blamed nothing in particular; this one must
    # also pre-empt the wrong conclusion, since the file genuinely has no suffix.
    assert "no extension by design" in message


def test_a_truncated_download_says_retry_not_check_the_url(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cut-off download and a wrong URL have different fixes.

    Telling someone to check their URL when the bytes were truncated sends them
    to the one place the problem is not.
    """
    served = tmp_path / "downloaded"
    served.write_bytes(b"RIFF")  # a WAV that never finished arriving

    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = lambda job_id, urls: [str(served)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    params = validate_job(cover_job())

    with pytest.raises(handler_module.WorkerError) as caught:
        handler_module._fetch_source_audio(params, "job-1")

    message = str(caught.value)
    assert "cut off" in message
    assert "Retry the job" in message
    assert "Check that the URL serves" not in message


def test_a_url_that_yields_nothing_is_named_not_silently_returned(
    handler_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`download_files_from_urls` can return an empty list without raising.

    That used to be reported as "downloaded to nothing", which is accurate but
    does not say what to do. The message now points at the SDK's silence.
    """
    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = lambda job_id, urls: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    params = validate_job(cover_job())

    with pytest.raises(handler_module.WorkerError) as caught:
        handler_module._fetch_source_audio(params, "job-1")

    assert "downloaded to nothing" in str(caught.value)
    assert "serves the file directly" in str(caught.value)


def _sdk_that_refuses_data(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake SDK with the real one's scheme check.

    runpod 1.12's `safe_get` allows only http/https and raises
    `blocked URL scheme 'data'` for anything else. The earlier fake accepted any
    scheme, which is how a suite that routed `data:` into the SDK stayed green
    while every such job failed on the endpoint.
    """
    called: dict[str, Any] = {}

    def fake_download(job_id: str, urls: list[str]) -> list[str]:
        called["urls"] = urls
        for url in urls:
            scheme = url.split(":", 1)[0]
            if scheme not in ("http", "https"):
                raise ValueError(f"blocked URL scheme {scheme!r}: {url[:40]}")
        return []

    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)
    return called


def test_a_data_uri_is_decoded_by_the_worker_not_the_sdk(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1: a base64 `data:` URI is documented input, and the SDK refuses it.

    It is also not a path — the route test once was `len(source) > 512`, so a
    short base64 URI fell through to the local-path branch and was stat()'d.
    """
    monkeypatch.chdir(tmp_path)
    called = _sdk_that_refuses_data(monkeypatch)

    encoded = base64.b64encode(_MINIMAL_WAV).decode()
    params = validate_job(cover_job(source_audio=f"data:audio/wav;base64,{encoded}"))
    resolved = Path(handler_module._fetch_source_audio(params, "job-1"))

    assert "urls" not in called, "a data: URI must not reach the SDK, which blocks the scheme"
    assert resolved.read_bytes() == _MINIMAL_WAV
    # The SDK's scratch location, not the workdir: the workdir is uploaded.
    assert resolved.parent == tmp_path / "jobs" / "job-1" / "downloaded_files"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("data:audio/wav;base64", "no ','"),
        ("data:audio/wav,RIFF", "not base64-encoded"),
        ("data:audio/wav;base64,not*base64!", "not valid base64"),
        ("data:audio/wav;base64,", "empty payload"),
    ],
)
def test_a_malformed_data_uri_is_named(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, expected: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _sdk_that_refuses_data(monkeypatch)

    params = validate_job(cover_job(source_audio=source))
    with pytest.raises(handler_module.WorkerError, match=expected):
        handler_module._fetch_source_audio(params, "job-1")


def test_a_data_uri_that_is_not_audio_is_refused_before_the_model(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decoded bytes go through the same `audio_probe` check as a download."""
    monkeypatch.chdir(tmp_path)
    _sdk_that_refuses_data(monkeypatch)

    page = base64.b64encode(b"<!DOCTYPE html><html><body>Not found</body></html>").decode()
    params = validate_job(cover_job(source_audio=f"data:audio/wav;base64,{page}"))
    with pytest.raises(handler_module.WorkerError, match="not a decodable audio file"):
        handler_module._fetch_source_audio(params, "job-1")


@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_a_cover_job_leaves_nothing_on_the_container_disk(
    handler_module: Any, storage: Storage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """`jobs/<job_id>` is off the volume, so it is removed on every exit path.

    The SDK's `clean()` does not touch `jobs/`; before this, each cover's source
    recording stayed on the container disk until the worker was recycled.
    """
    import modes

    monkeypatch.chdir(tmp_path)
    _sdk_that_refuses_data(monkeypatch)

    seen: dict[str, Any] = {}

    def fake_dispatch(mode: str, p: Any, workdir: Path) -> Any:
        seen["source"] = Path(p.source_audio)
        seen["existed"] = seen["source"].exists()
        if fails:
            raise modes.ModeError("transcription failed")
        return modes.ModeResult(mode=mode, score_abc=None, lyrics="from asr")

    monkeypatch.setattr(handler_module.modes, "dispatch", fake_dispatch)

    encoded = base64.b64encode(_MINIMAL_WAV).decode()
    job = {**cover_job(source_audio=f"data:audio/wav;base64,{encoded}"), "id": "job-7"}
    result = handler_module.run_create_job(job, storage=storage, config=None)

    assert seen["existed"], "the mode must receive a file that exists"
    assert ("error" in result) is fails
    assert not (tmp_path / "jobs" / "job-7").exists()
    assert not seen["source"].exists()


@pytest.mark.parametrize("job_id", ["", ".", "..", "../elsewhere", "a/b"])
def test_download_cleanup_never_reaches_outside_its_own_job(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_id: str
) -> None:
    """The id is caller-influenced; it must not turn the delete into `rm -r jobs/` or worse."""
    monkeypatch.chdir(tmp_path)
    sibling = tmp_path / "jobs" / "other-job" / "downloaded_files" / "keep"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"x")
    outside = tmp_path / "elsewhere" / "keep"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    (tmp_path / "jobs" / "a" / "b").mkdir(parents=True)

    handler_module._cleanup_job_downloads(job_id)

    assert sibling.exists()
    assert outside.exists()


def test_real_audio_still_passes_through_the_handler(
    handler_module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not break the case it exists to protect."""
    served = tmp_path / "b3f1c2d4-0000-4000-8000-000000000000"  # bare UUID, no ext
    served.write_bytes(_MINIMAL_WAV)

    fake_utils = type(sys)("runpod.serverless.utils")
    fake_utils.download_files_from_urls = lambda job_id, urls: [str(served)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "runpod.serverless.utils", fake_utils)

    params = validate_job(cover_job())
    assert handler_module._fetch_source_audio(params, "job-1") == str(served)
