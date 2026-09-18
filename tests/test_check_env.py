"""Tests for `check_env.py` — the version checker that guards the build.

A checker is only worth having if it can fail, and the one it replaced could
not: it printed versions without comparing them. So the tests here are mostly
about the *failure* paths — that a mismatch is detected, that a missing package
is detected, and that the two cases a naive string comparison would get wrong
are handled.

There is also a test for the bug this file's own author wrote: the first version
of `check_env.py` crashed on an attribute that belonged to a different class.
A checker that raises is not a checker that reports, and it is worth pinning.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import check_env
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER = REPO_ROOT / "worker"
CHECKER = WORKER / "check_env.py"


@pytest.fixture
def pins(tmp_path: Path):
    """Write a requirements file and return its path."""

    def write(text: str) -> Path:
        path = tmp_path / "requirements.txt"
        path.write_text(text, encoding="utf-8")
        return path

    return write


# =============================================================================
# Reading the pins
# =============================================================================


def test_parses_exact_pins(pins) -> None:
    path = pins("torch==2.10.0\nnumpy==2.2.6\n")
    assert check_env.parse_requirements(path) == [("torch", "2.10.0"), ("numpy", "2.2.6")]


def test_ignores_comments_options_and_blanks(pins) -> None:
    path = pins(
        "# a comment\n--extra-index-url https://example.invalid\n\ntorch==2.10.0  # trailing comment\n[some-section]\n"
    )
    assert check_env.parse_requirements(path) == [("torch", "2.10.0")]


def test_loose_requirements_are_not_treated_as_pins(pins) -> None:
    """`torch>=2` states no target, so nothing can be asserted about it.

    Inventing a version to compare against would be making up a fact — the
    failure mode this whole project keeps running into.
    """
    path = pins("torch>=2.0\ntransformers~=4.57\nnumpy\n")
    assert check_env.parse_requirements(path) == []


def test_the_real_requirements_files_parse() -> None:
    """Both shipped pins files must be readable and non-trivial."""
    main = check_env.parse_requirements(WORKER / "requirements.txt")
    sheetsage = check_env.parse_requirements(WORKER / "transcribe_sheetsage" / "requirements.txt")
    assert len(main) >= 8, main
    assert len(sheetsage) >= 8, sheetsage
    assert ("torch", "2.10.0") in main
    assert ("torch", "2.8.0") in sheetsage
    assert ("numpy", "1.24.3") in sheetsage


# =============================================================================
# Comparing versions — the two cases a string compare gets wrong
# =============================================================================


def test_local_version_identifier_is_ignored() -> None:
    """`torch==2.10.0` is satisfied by `2.10.0+cu128`.

    PEP 440: a requirement without a local segment matches any. Comparing
    strings directly would report every CUDA build as a mismatch — which is
    every build we produce.
    """
    assert check_env.versions_match("2.10.0", "2.10.0+cu128")
    assert check_env.versions_match("2.8.0", "2.8.0+cu126")
    assert check_env.versions_match("2.10.0", "2.10.0")


def test_a_different_version_does_not_match() -> None:
    assert not check_env.versions_match("2.10.0", "2.9.0")
    assert not check_env.versions_match("1.24.3", "2.2.6")
    assert not check_env.versions_match("4.45.2", "4.57.6")


def test_a_different_base_version_does_not_match_even_with_a_local_segment() -> None:
    """The local-segment rule must not become a wildcard."""
    assert not check_env.versions_match("2.8.0", "2.10.0+cu128")
    assert not check_env.versions_match("4.45.2", "4.57.6")


def test_distribution_names_normalise_per_pep503() -> None:
    assert check_env.normalise("yue2_infer") == check_env.normalise("yue2-infer")
    assert check_env.normalise("pretty.midi") == "pretty-midi"
    assert check_env.normalise("HuggingFace-Hub") == "huggingface-hub"


# =============================================================================
# The CUDA check — a version pin cannot express this
# =============================================================================


def test_cpu_only_torch_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """`torch==2.10.0` matches the CPU wheel from PyPI and the CUDA wheel.

    Only one of them can use the GPU, and the requirement cannot tell them
    apart. The failure mode is a worker that boots, accepts jobs, and then fails
    at `device="cuda"` — so it is checked explicitly.
    """
    monkeypatch.setattr(check_env, "torch_cuda_version", lambda: None)
    problem = check_env.verify_cuda_build("torch")
    assert problem and "CUDA" in problem


def test_cuda_torch_with_a_local_version_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PyTorch-index arrangement: `2.10.0+cu128`."""
    monkeypatch.setattr(check_env, "torch_cuda_version", lambda: "12.8")
    assert check_env.verify_cuda_build("torch") == ""


def test_cuda_torch_without_a_local_version_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PyPI arrangement: a plain `2.14.0` that bundles `nvidia-*` packages.

    This is the case the check used to fail. `torch==2.14.0` from PyPI carries no
    local version identifier and is nonetheless a CUDA build — its 554 MB wheel
    pulls `nvidia-cudnn-cu13`, `nvidia-cublas` and `cuda-toolkit` as ordinary
    dependencies.

    The false failure was not hypothetical: it stopped a build over the ASR
    environment, which had already transcribed lyrics on a GPU in a completed
    cover job. A check that rejects a working environment is worse than no check,
    because it sends you to fix something that is not broken.
    """
    monkeypatch.setattr(check_env, "torch_cuda_version", lambda: "13.0")
    # `installed_version` is pinned too, and that is not decoration. Without it
    # this test passes against the *old* identifier-only logic, because torch is
    # not installed on a dev box and that logic returns early on an empty version.
    # Pinning it means the old logic reaches its own `"+" not in version` branch
    # and fails here — which is the whole point of the test.
    monkeypatch.setattr(check_env, "installed_version", lambda p: "2.14.0")
    assert check_env.verify_cuda_build("torch") == ""


def test_an_unimportable_torch_falls_back_to_the_version_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """`""` means "cannot tell", which must not be read as "no CUDA".

    A build step that has not installed torch yet gets the weaker string check
    rather than a false failure.
    """
    monkeypatch.setattr(check_env, "torch_cuda_version", lambda: "")
    monkeypatch.setattr(check_env, "installed_version", lambda p: "2.10.0+cu128")
    assert check_env.verify_cuda_build("torch") == ""

    monkeypatch.setattr(check_env, "installed_version", lambda p: "2.10.0")
    assert "cannot confirm" in check_env.verify_cuda_build("torch")


def test_non_torch_packages_are_not_cuda_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_env, "installed_version", lambda p: "4.57.6")
    assert check_env.verify_cuda_build("transformers") == ""


# =============================================================================
# Building a report — and being able to fail
# =============================================================================


def test_a_mismatch_is_a_failure(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_env, "installed_version", lambda p: "2.9.0")
    report = check_env.check_environment(pins("torch==2.10.0\n"), do_imports=False)
    assert not report.ok
    assert report.failures and report.failures[0].package == "torch"


def test_a_match_passes(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_env, "installed_version", lambda p: "2.10.0+cu128")  # torch only

    report = check_env.check_environment(pins("torch==2.10.0\n"), do_imports=False)
    assert report.ok, report.render()


def test_a_missing_package_is_a_failure(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_env, "installed_version", lambda p: None)
    report = check_env.check_environment(pins("torch==2.10.0\n"), do_imports=False)
    assert not report.ok
    assert "not installed" in report.render()


def test_a_broken_import_is_a_failure_even_when_the_version_matches(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason the checker imports and does not only compare versions.

    A wheel can install cleanly with the right version and still fail to load —
    which is exactly how a missing `libsndfile1` presents: `soundfile` reports
    0.13.1 and then raises on import.
    """
    monkeypatch.setattr(check_env, "installed_version", lambda p: "0.13.1")
    monkeypatch.setattr(check_env, "check_import", lambda name: "OSError: libsndfile not found")
    report = check_env.check_environment(pins("soundfile==0.13.1\n"), do_imports=True)
    assert not report.ok
    assert "import failed" in report.render()


def test_report_renders_every_package(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"torch": "2.10.0+cu128", "numpy": "2.2.6"}
    monkeypatch.setattr(check_env, "installed_version", versions.get)
    report = check_env.check_environment(pins("torch==2.10.0\nnumpy==2.2.6\n"), do_imports=False)
    rendered = report.render()
    assert "torch" in rendered and "numpy" in rendered
    assert "match" in rendered, rendered


def test_advisory_findings_do_not_fail_the_report(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unpinned package resolving unexpectedly is reported, not asserted.

    `qwen-asr` declares no torch pin, so no pin says what it should have been.
    Failing the build over it would be asserting a fact we do not have.
    """
    monkeypatch.setattr(check_env, "installed_version", lambda p: None)
    report = check_env.check_environment(pins("qwen-asr==0.0.6\n"), do_imports=False)
    assert report.ok, "an advisory finding must not fail the check"
    assert report.warnings, "but it must still be reported"


# =============================================================================
# The command itself
# =============================================================================


def test_exit_code_is_nonzero_on_a_mismatch(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_env, "installed_version", lambda p: "0.0.1")
    assert check_env.main(["--requirements", str(pins("torch==2.10.0\n")), "--no-imports"]) == 1


def test_exit_code_is_zero_when_everything_matches(pins, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_env, "installed_version", lambda p: "2.10.0+cu128")
    assert check_env.main(["--requirements", str(pins("torch==2.10.0\n")), "--no-imports"]) == 0


def test_a_missing_requirements_file_is_a_checker_error(tmp_path: Path) -> None:
    """Exit 2, not 1: a missing pins file is a build error, not a version error."""
    assert check_env.main(["--requirements", str(tmp_path / "absent.txt")]) == 2


def test_json_output_is_parseable(pins, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(check_env, "installed_version", lambda p: "2.10.0+cu128")
    check_env.main(["--requirements", str(pins("torch==2.10.0\n")), "--json", "--no-imports"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["packages"][0]["package"] == "torch"


def test_the_checker_does_not_crash_on_a_real_environment() -> None:
    """Regression for a bug in the checker's own first version.

    `Report.failures` referenced `self.advisory`, which belongs to `Finding`.
    The checker raised `AttributeError` instead of reporting — the same class of
    defect it was written to catch: a check that cannot do its job.

    Running it as a subprocess on the real pins exercises the whole path. It is
    allowed to report mismatches; it is not allowed to crash.
    """
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--requirements", str(WORKER / "requirements.txt"), "--json", "--no-imports"],
        capture_output=True,
        text=True,
    )
    assert result.returncode in (0, 1), f"checker crashed:\n{result.stderr}"
    payload = json.loads(result.stdout)
    assert payload.get("packages")


def test_the_dockerfile_runs_the_checker_in_every_environment() -> None:
    """The build must assert, not print.

    The checks this replaced printed versions and could not fail, so a venv that
    resolved the wrong torch reported the wrong number and the build passed.
    """
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "check_env.py" in dockerfile, "the checker is not run at build time"
    # Against both pinned environments, not just one.
    assert "main-requirements.txt" in dockerfile
    assert "sheetsage-requirements.txt" in dockerfile
    assert "print('sheetsage2'" not in dockerfile, "the print-only check is back"
