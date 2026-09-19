"""Tests for the subprocess isolation and the cover/edit mode pipelines.

Nothing here imports torch, transformers, qwen_asr or any model library, and
nothing creates a virtual environment. That is not a testing convenience — it is
the point. This machine has no GPU and packages must not be installed on it, so
the *only* way to exercise this code locally is to mock the subprocess boundary
and assert on the contract the child is expected to honour.

What that means for confidence, stated plainly: these tests prove the parent
writes the right request, reads the right result, handles every failure shape,
and never lets a broken child look like a success. They cannot prove SheetSage2
or Qwen3-ASR actually behave as documented — that is the container run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import abc_score
import pytest
import subprocess_runner
from subprocess_runner import SubprocessError, run_stage, scratch_dir, venv_python, write_request

import modes

#: A venv name for exercising `run_stage` itself. The image builds no venvs —
#: both model families run in the main environment — but the runner still takes a
#: venv name, and these tests are about the runner, not about which environments
#: exist. A literal here rather than a `modes` constant, because there is no
#: longer a constant: the one that existed named a venv that is gone.
SAMPLE_VENV = "sheetsage2"

from modes import ModeError
from schema import SongParameters

MELODY_ABC = """X:1
T:
M:4/4
L:1/32
Q:1/4=118
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Dm
V: Vocal
z32|z32|
V: Ins
z8f8f8f8|z8e8e8e8|
"""

CHORDY_ABC = MELODY_ABC.replace("z8f8f8f8|", '"Gm7"z8f8f8f8|', 1)


def params(**overrides: Any) -> SongParameters:
    base: dict[str, Any] = {
        "style": "Jazz-funk",
        "lyrics": "[Verse]\nhello",
        "cot": "melody",
        "seed": 831001,
        "cfg_scale": None,
        "abc": None,
        "id": "job1",
    }
    base.update(overrides)
    return SongParameters(**base)


@pytest.fixture
def fake_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A venv root whose interpreters are stubs, so `run_stage` can find them.

    `VENV_ROOT` is read at call time by `venv_python`, so patching the module
    attribute is enough — no reload needed.
    """
    root = tmp_path / "venvs"
    bin_dir = root / SAMPLE_VENV / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("#!/bin/sh\n")
    monkeypatch.setattr(subprocess_runner, "VENV_ROOT", root)
    return root


class FakeCompleted:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = ""
        self.stderr = stderr


@pytest.fixture
def child(monkeypatch: pytest.MonkeyPatch):
    """Stand in for `subprocess.run`, running a Python callable in-process.

    Using the real `subprocess.run` would need real interpreters with the model
    libraries installed, which is exactly what must not happen here. This
    replaces the process boundary while keeping everything on either side of it
    real — the request file, the result file, the artifact scan.
    """

    def install(handler):
        def fake_run(cmd, cwd=None, env=None, capture_output=None, text=None, timeout=None, check=None):
            workdir = Path(cwd)
            request = json.loads((workdir / "request.json").read_text())
            try:
                result = handler(request, workdir)
                (workdir / "result.json").write_text(json.dumps(result), encoding="utf-8")
                return FakeCompleted(0)
            except Exception as exc:
                (workdir / "result.json").write_text(
                    json.dumps({"status": "failed", "error": str(exc)}), encoding="utf-8"
                )
                return FakeCompleted(2, stderr=f"{type(exc).__name__}: {exc}")

        monkeypatch.setattr(subprocess_runner.subprocess, "run", fake_run)
        return install

    return install


# =============================================================================
# The subprocess contract
# =============================================================================


def test_missing_interpreter_is_an_image_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing venv means the *image* is wrong, so it raises rather than
    returning a failed result a caller might retry forever."""
    monkeypatch.setattr(subprocess_runner, "VENV_ROOT", tmp_path / "nowhere")
    with pytest.raises(SubprocessError, match="interpreter not found"):
        run_stage("sheetsage2", tmp_path / "run.py", tmp_path, timeout_seconds=5)


def test_missing_entrypoint_is_an_image_error(fake_venv: Path, tmp_path: Path) -> None:
    write_request(tmp_path, {"audio": "x"})
    with pytest.raises(SubprocessError, match="entrypoint not found"):
        run_stage(SAMPLE_VENV, tmp_path / "absent.py", tmp_path, timeout_seconds=5)


def test_missing_request_file_is_an_error(fake_venv: Path, tmp_path: Path) -> None:
    entry = tmp_path / "run.py"
    entry.write_text("")
    with pytest.raises(SubprocessError, match=r"no request\.json"):
        run_stage(SAMPLE_VENV, entry, tmp_path, timeout_seconds=5)


def test_a_successful_child_is_read_back(fake_venv: Path, tmp_path: Path, child) -> None:
    entry = tmp_path / "run.py"
    entry.write_text("")

    def handler(request, workdir):
        (workdir / "score.abc").write_text("X:1\n", encoding="utf-8")
        return {"status": "complete", "abc": "X:1\n"}

    child(handler)
    write_request(tmp_path, {"audio": "x"})
    result = run_stage(SAMPLE_VENV, entry, tmp_path, timeout_seconds=5)

    assert result.ok
    assert result.payload["abc"] == "X:1\n"
    assert "score.abc" in result.artifacts
    assert result.error is None


def test_request_is_not_treated_as_an_artifact(fake_venv: Path, tmp_path: Path, child) -> None:
    """`request.json` is ours, not the stage's output."""
    entry = tmp_path / "run.py"
    entry.write_text("")
    child(lambda request, workdir: {"status": "complete"})
    write_request(tmp_path, {"audio": "x"})
    result = run_stage(SAMPLE_VENV, entry, tmp_path, timeout_seconds=5)
    assert "request.json" not in result.artifacts


def test_nonzero_exit_without_a_result_is_a_failure(fake_venv: Path, tmp_path: Path, monkeypatch) -> None:
    """A child that dies before writing anything must not read as success."""
    entry = tmp_path / "run.py"
    entry.write_text("")
    monkeypatch.setattr(subprocess_runner.subprocess, "run", lambda *a, **k: FakeCompleted(1, stderr="Killed"))
    write_request(tmp_path, {"audio": "x"})
    result = run_stage(SAMPLE_VENV, entry, tmp_path, timeout_seconds=5)

    assert not result.ok
    assert result.returncode == 1
    assert "Killed" in (result.error or "")


def test_status_failed_is_not_success_even_on_exit_zero(fake_venv: Path, tmp_path: Path, monkeypatch) -> None:
    """The child's own verdict counts, not just its exit code.

    A stage that writes `{"status": "failed"}` and exits 0 is a stage that
    failed; treating the exit code as the only signal would let it through.
    """
    entry = tmp_path / "run.py"
    entry.write_text("")

    def fake_run(cmd, cwd=None, **kwargs):
        (Path(cwd) / "result.json").write_text(json.dumps({"status": "failed", "error": "no ABC"}))
        return FakeCompleted(0)

    monkeypatch.setattr(subprocess_runner.subprocess, "run", fake_run)
    write_request(tmp_path, {"audio": "x"})
    result = run_stage(SAMPLE_VENV, entry, tmp_path, timeout_seconds=5)

    assert not result.ok
    assert "no ABC" in (result.error or "")


def test_timeout_is_reported_as_a_budget_failure(fake_venv: Path, tmp_path: Path, monkeypatch) -> None:
    entry = tmp_path / "run.py"
    entry.write_text("")

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=5, stderr="still going")

    monkeypatch.setattr(subprocess_runner.subprocess, "run", timeout)
    write_request(tmp_path, {"audio": "x"})
    result = run_stage(SAMPLE_VENV, entry, tmp_path, timeout_seconds=5)

    assert not result.ok
    assert "budget" in (result.error or "")


def test_child_does_not_inherit_the_parent_pythonpath(fake_venv: Path, tmp_path: Path, monkeypatch) -> None:
    """The exact confusion the boundary exists to prevent.

    If the child inherited this worker's `PYTHONPATH`, it could import our
    modules against a different torch — the two environments would silently
    blend, which is the one thing the design is meant to make impossible.
    """
    entry = tmp_path / "run.py"
    entry.write_text("")
    seen: dict[str, Any] = {}

    def fake_run(cmd, cwd=None, env=None, **kwargs):
        seen.update(env or {})
        (Path(cwd) / "result.json").write_text(json.dumps({"status": "complete"}))
        return FakeCompleted(0)

    monkeypatch.setattr(subprocess_runner.subprocess, "run", fake_run)
    write_request(tmp_path, {"audio": "x"})
    run_stage(SAMPLE_VENV, entry, tmp_path, timeout_seconds=5)

    assert seen.get("PYTHONPATH") == ""
    # And it does inherit the HF cache, or the child would re-download the very
    # weights the endpoint cached. Asserted properly rather than with a fallback
    # that can never fail.
    assert any(k in seen for k in ("HF_HOME", "HF_HUB_CACHE"))


def test_scratch_dir_is_fresh_each_time(tmp_path: Path) -> None:
    first = scratch_dir(tmp_path, "stage")
    (first / "stale.txt").write_text("old")
    second = scratch_dir(tmp_path, "stage")
    assert second == first
    assert not (second / "stale.txt").exists(), "a reused directory could mix two runs' artifacts"


def test_venv_python_path_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess_runner, "VENV_ROOT", tmp_path)
    assert venv_python("sheetsage2") == tmp_path / "sheetsage2" / "bin" / "python"


# =============================================================================
# cover
# =============================================================================


def test_cover_requires_source_audio(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="source_audio"):
        modes.prepare_cover(params(source_audio=None), tmp_path)


def test_cover_rejects_a_missing_source_file(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="not found"):
        modes.prepare_cover(params(source_audio=str(tmp_path / "nope.wav")), tmp_path)


def test_cover_runs_both_stages_in_order(fake_venv: Path, tmp_path: Path, child) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF")
    calls: list[str] = []

    def handler(request, workdir):
        calls.append(Path(workdir).name)
        if "sheetsage" in str(workdir):
            return {"status": "complete", "abc": MELODY_ABC}
        return {"status": "complete", "text": "hello world"}

    child(handler)
    result = modes.prepare_cover(params(source_audio=str(source)), tmp_path)

    assert calls == ["sheetsage", "asr"], "the melody must be transcribed before the lyrics"
    assert result.score_abc == MELODY_ABC
    assert result.lyrics == "hello world"
    assert result.stages["sheetsage"]["ok"] and result.stages["asr"]["ok"]


def test_cover_sends_melody_prompts_and_no_chord_prompt(fake_venv: Path, tmp_path: Path, child) -> None:
    """The prompt set is the mechanism by which chords stay out of the melody."""
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")
    seen: dict[str, Any] = {}

    def handler(request, workdir):
        if "sheetsage" in str(workdir):
            seen.update(request)
            return {"status": "complete", "abc": MELODY_ABC}
        return {"status": "complete", "text": "x"}

    child(handler)
    modes.prepare_cover(params(source_audio=str(source)), tmp_path)

    assert seen["melody_only"] is True
    assert "melody_full" in seen["prompts"]
    assert "chord_full" not in seen["prompts"], "requesting chord_full would return harmony"


def test_cover_strips_chords_the_transcriber_left_in(fake_venv: Path, tmp_path: Path, child) -> None:
    """Defence in depth: verify the melody rather than trust the flag."""
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")

    def handler(request, workdir):
        if "sheetsage" in str(workdir):
            return {"status": "complete", "abc": CHORDY_ABC}
        return {"status": "complete", "text": "x"}

    child(handler)
    result = modes.prepare_cover(params(source_audio=str(source)), tmp_path)

    assert abc_score.find_chords(result.score_abc or "") == []
    assert result.stages["sheetsage"]["chords_stripped"] == ["Gm7"]


def test_cover_rejects_a_melody_that_cannot_be_cleaned(fake_venv: Path, tmp_path: Path, child) -> None:
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")

    def handler(request, workdir):
        if "sheetsage" in str(workdir):
            return {"status": "complete", "abc": "not abc at all"}
        return {"status": "complete", "text": "x"}

    child(handler)
    with pytest.raises(ModeError, match=r"not usable|score"):
        modes.prepare_cover(params(source_audio=str(source)), tmp_path)


def test_cover_fails_when_melody_transcription_fails(fake_venv: Path, tmp_path: Path, child) -> None:
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")

    def handler(request, workdir):
        raise RuntimeError("this SheetSage2 revision does not expose melody_only")

    child(handler)
    with pytest.raises(ModeError, match="melody transcription failed"):
        modes.prepare_cover(params(source_audio=str(source)), tmp_path)


def test_cover_fails_when_asr_fails(fake_venv: Path, tmp_path: Path, child) -> None:
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")

    def handler(request, workdir):
        if "sheetsage" in str(workdir):
            return {"status": "complete", "abc": MELODY_ABC}
        raise RuntimeError("no speech detected")

    child(handler)
    with pytest.raises(ModeError, match="lyric transcription failed"):
        modes.prepare_cover(params(source_audio=str(source)), tmp_path)


def test_cover_prefers_caller_lyrics_over_transcription(fake_venv: Path, tmp_path: Path, child) -> None:
    """The upstream reference says to obtain or check the lyrics separately —
    a listener usually knows them better than an ASR pass over a full mix."""
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")

    def handler(request, workdir):
        if "sheetsage" in str(workdir):
            return {"status": "complete", "abc": MELODY_ABC}
        return {"status": "complete", "text": "transcribed words"}

    child(handler)
    result = modes.prepare_cover(
        params(source_audio=str(source), lyrics="the caller's words", lyrics_supplied=True), tmp_path
    )

    assert result.lyrics == "the caller's words"
    assert result.stages["asr"]["used"] is False


def test_cover_fails_when_no_lyrics_are_available_at_all(fake_venv: Path, tmp_path: Path, child) -> None:
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")

    def handler(request, workdir):
        if "sheetsage" in str(workdir):
            return {"status": "complete", "abc": MELODY_ABC}
        return {"status": "complete", "text": "   "}  # whitespace is not lyrics

    child(handler)
    with pytest.raises(ModeError, match="returned no lyrics"):
        modes.prepare_cover(params(source_audio=str(source)), tmp_path)


# =============================================================================
# edit
# =============================================================================


def test_edit_with_a_supplied_score_validates_it(tmp_path: Path) -> None:
    result = modes.prepare_edit(params(abc=MELODY_ABC, cot="full"), tmp_path)
    assert result.score_abc == MELODY_ABC
    assert result.stages["score"]["source"] == "supplied"


def test_edit_rejects_a_malformed_score(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="edit mode"):
        modes.prepare_edit(params(abc="X:1\nT:\nnot a score\n", cot="full"), tmp_path)


def test_edit_reports_no_score_when_the_caller_wants_a_plan(tmp_path: Path) -> None:
    """No `abc` means "write me a plan to edit", which the handler does."""
    result = modes.prepare_edit(params(abc=None), tmp_path)
    assert result.score_abc is None
    assert result.stages["score"]["source"] == "plan"


def test_edit_does_not_require_melody_only(tmp_path: Path) -> None:
    """An edit keeps the harmony — that is what `cot="full"` means."""
    result = modes.prepare_edit(params(abc=CHORDY_ABC, cot="full"), tmp_path)
    assert abc_score.find_chords(result.score_abc or ""), "chords must survive an edit"


# =============================================================================
# dispatch
# =============================================================================


def test_dispatch_routes_each_mode(fake_venv: Path, tmp_path: Path, child) -> None:
    source = tmp_path / "s.wav"
    source.write_bytes(b"RIFF")

    def handler(request, workdir):
        if "sheetsage" in str(workdir):
            return {"status": "complete", "abc": MELODY_ABC}
        return {"status": "complete", "text": "words"}

    child(handler)
    assert modes.dispatch("create", params(), tmp_path).mode == "create"
    assert modes.dispatch("cover", params(source_audio=str(source)), tmp_path).mode == "cover"
    assert modes.dispatch("edit", params(abc=MELODY_ABC), tmp_path).mode == "edit"


def test_dispatch_rejects_an_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ModeError, match="unknown mode"):
        modes.dispatch("remix", params(), tmp_path)


def test_every_mode_eventually_needs_the_pipeline() -> None:
    """All three end in generation, which is why the boot-before-first-job
    contract applies to every mode rather than only `create`."""
    for mode in modes.MODES:
        assert modes.mode_needs_pipeline(mode), mode


# =============================================================================
# The invariant that matters most
# =============================================================================


def test_nothing_in_the_new_modules_imports_a_model_library() -> None:
    """The worker process must never import torch, transformers or qwen_asr.

    Those belong to the child environments. An import here would either fail —
    they are not installed in the main image's interpreter at the versions the
    children need — or, worse, succeed against the wrong torch and blend the
    environments the design keeps apart.
    """
    forbidden = ("torch", "transformers", "qwen_asr", "torchaudio", "safetensors")
    root = Path(modes.__file__).parent
    for name in ("modes.py", "abc_score.py", "subprocess_runner.py", "handler.py", "schema.py"):
        source = (root / name).read_text(encoding="utf-8")
        for module in forbidden:
            assert f"import {module}" not in source, f"{name} imports {module}"
            assert f"from {module}" not in source, f"{name} imports from {module}"


def test_child_entrypoints_are_the_only_place_models_are_imported() -> None:
    """The reverse of the rule above: the children *do* import them, lazily.

    They must import inside the function, not at module scope, so a missing
    library surfaces as a result.json error rather than an import traceback the
    parent cannot interpret.
    """
    root = Path(modes.__file__).parent
    for rel in ("transcribe_sheetsage/run.py", "transcribe_asr/run.py"):
        source = (root / rel).read_text(encoding="utf-8")
        top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
        for line in top_level:
            assert "torch" not in line and "transformers" not in line and "qwen_asr" not in line, (
                f"{rel} imports a model library at module scope: {line}"
            )


def test_the_melody_prompt_set_matches_upstream() -> None:
    """The prompt set is the mechanism that keeps harmony out of a cover melody.

    The YuE2 authors' transcriber builds its prompt list by task: a melody task
    gets the four structural prompts plus `melody_full`, and only `full` adds
    `chord_full`. Sending `chord_full` would return harmony we then have to
    strip, so the set itself is part of the contract, not an implementation
    detail. Asserted against the actual list rather than a text search — a
    comment naming `chord_full` is not a request for it.
    """
    import ast
    from pathlib import Path

    source = (Path(modes.__file__).parent / "transcribe_sheetsage" / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    prompts = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MELODY_PROMPTS":
                    prompts = [element.value for element in node.value.elts]

    assert prompts is not None, "MELODY_PROMPTS not found in the sheetsage entrypoint"
    assert "chord_full" not in prompts, "requesting chord_full would return harmony"
    assert "melody_full" in prompts
    for structural in ("timestamp", "downbeat_meter", "structure", "key"):
        assert structural in prompts, structural


def test_no_worker_module_imports_a_model_library_at_any_scope() -> None:
    """Stronger than the module-scope check above: these modules import *nothing*
    from the model stack, at any indentation."""
    import ast
    from pathlib import Path

    forbidden = {"torch", "transformers", "qwen_asr", "torchaudio", "safetensors"}
    root = Path(modes.__file__).parent
    for name in ("modes.py", "abc_score.py", "subprocess_runner.py", "handler.py", "schema.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, f"{name} imports from {node.module}"
