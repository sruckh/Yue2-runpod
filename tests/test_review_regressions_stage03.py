"""Regression tests for defects found by adversarial review of Stage 03.

An independent critic compared this module against the YuE2 authors' own
`abc_tools.py` and `transcribe.py`, **executing both** on the same inputs rather
than reading them. It found one defect that defeated a guarantee this code
claimed was load-bearing, and several more that were real on their own terms.

Every test here names the defect it pins. They are grouped by kind, because the
two kinds have different lessons:

- **Fail-open vs fail-closed** — a validator that ignores what it does not
  understand is not a validator, and the thing it lets through is exactly the
  thing it was written to stop.
- **Ordering and plumbing** — checks that run in an order making them
  unreachable, stages whose failure is fatal when a fallback exists, and
  cleanup that swallows the failure that matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import subprocess_runner
from abc_score import AbcError, inspect, strip_chords, validate_native
from subprocess_runner import SubprocessError, scratch_dir

import modes

SCORE = """X:1
T:
M:4/4
L:1/32
Q:1/4=118
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Dm
V: Vocal
{line}
V: Ins
z32|
"""


# =============================================================================
# Fail-open vs fail-closed
# =============================================================================


@pytest.mark.parametrize("token", ["Cmaj9", "Chorus", "N.C.", "Cadd9", "C/Guitar"])
def test_a_quoted_token_that_is_not_a_known_chord_is_rejected(token: str) -> None:
    """The defect that defeated the cover guarantee.

    Upstream's policy is that *any* quoted token on a music line is a chord
    symbol, and one that does not parse is an error. This module originally
    invented a "text annotation" exemption and ignored anything outside its
    vocabulary — so `"Cmaj9"`, a real chord, passed validation *and* survived
    `strip_chords`, arriving at YuE2 with `cot="melody"` and producing something
    that was not a cover.

    Ignoring an unrecognised token is the one behaviour that cannot be right:
    we do not know whether it is harmony.
    """
    score = SCORE.format(line=f'"{token}"C8D8E8G8|')
    with pytest.raises(AbcError, match="unrecognised chord"):
        validate_native(score, source="test")


def test_unrecognised_quoted_tokens_are_reported_separately_from_chords() -> None:
    """A recognised chord and an unrecognised one are different findings."""
    score = SCORE.format(line='"Gm7"C8D8E8G8|')
    report = inspect(score)
    assert report.chords == ["Gm7"]
    assert report.unrecognised_quoted == []

    score = SCORE.format(line='"Cmaj9"C8D8E8G8|')
    report = inspect(score)
    assert report.chords == []
    assert report.unrecognised_quoted == ["Cmaj9"]


def test_strip_refuses_an_unrecognised_token_rather_than_passing_it_through() -> None:
    """Stripping must not silently succeed while leaving possible harmony behind."""
    score = SCORE.format(line='"Cmaj9"C8D8E8G8|')
    with pytest.raises(AbcError, match="cannot strip unrecognised"):
        strip_chords(score, source="test")


def test_a_clean_melody_is_still_accepted() -> None:
    """The stricter rule must not break the ordinary case."""
    score = SCORE.format(line="C8D8E8G8|")
    report = validate_native(score, source="test", require_melody_only=True)
    assert not report.chords and not report.unrecognised_quoted


def test_real_chords_are_still_stripped() -> None:
    score = SCORE.format(line='"Gm7"C8D8E8G8|')
    stripped, removed = strip_chords(score, source="test")
    assert removed == ["Gm7"]
    validate_native(stripped, source="test", require_melody_only=True)


# =============================================================================
# Ordering: a check that cannot fire is not a check
# =============================================================================


def test_cover_validates_before_stripping_so_the_melody_check_is_reachable() -> None:
    """The `require_melody_only` branch was dead at its only call site.

    `prepare_cover` stripped first, and `strip_chords` already guarantees
    chord-freedom or raises — so validating afterwards restated a fact rather
    than asserting a property. The order is now inspect → strip → validate, and
    the inspect step is what actually protects the guarantee.

    Pinned by source position, because the bug *was* the order.
    """
    import inspect as pyinspect

    source = pyinspect.getsource(modes.prepare_cover)
    inspect_at = source.index("abc_score.inspect(")
    strip_at = source.index("abc_score.strip_chords(")
    assert inspect_at < strip_at, "inspection must precede stripping or the check cannot fail"


def test_cover_rejects_unrecognised_tokens_with_its_own_message() -> None:
    """The failure should name the cover path, not surface as a bare AbcError."""
    import inspect as pyinspect

    source = pyinspect.getsource(modes.prepare_cover)
    assert "unrecognised_quoted" in source, "prepare_cover must check for unrecognised tokens"


# =============================================================================
# A stage whose failure should not fail the job
# =============================================================================


def test_run_stage_records_advisory_intent(caplog: pytest.LogCaptureFixture, tmp_path: Path, monkeypatch) -> None:
    """`required=False` marks a failure as tolerated, and says so in the log."""
    import logging

    root = tmp_path / "venvs" / "x" / "bin"
    root.mkdir(parents=True)
    (root / "python").write_text("#!/bin/sh\n")
    monkeypatch.setattr(subprocess_runner, "VENV_ROOT", tmp_path / "venvs")

    entry = tmp_path / "run.py"
    entry.write_text("")
    subprocess_runner.write_request(tmp_path, {"audio": "x"})

    class Dead:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(subprocess_runner.subprocess, "run", lambda *a, **k: Dead())

    with caplog.at_level(logging.WARNING, logger="subprocess_runner"):
        result = subprocess_runner.run_stage("x", entry, tmp_path, timeout_seconds=5, name="Advisory", required=False)

    assert not result.ok
    assert any("advisory" in r.message.lower() for r in caplog.records)


def test_cover_exposes_whether_asr_was_used() -> None:
    """A caller reading the response must be able to tell which source won."""
    import inspect as pyinspect

    source = pyinspect.getsource(modes.prepare_cover)
    assert '"used"' in source and '"reason"' in source


# =============================================================================
# Cleanup that must not swallow its own failure
# =============================================================================


def test_scratch_dir_error_is_not_swallowed(tmp_path: Path, monkeypatch) -> None:
    """The defect: `rmtree(ignore_errors=True)` left the directory intact.

    The old code removed with errors ignored and then created with
    `exist_ok=True`, so a failed removal looked like success — and the next
    stage's artifact sweep reported the *previous* run's `score.abc` and
    `lyrics.txt` as this run's output. Silent wrongness that reads as success.
    """
    target = tmp_path / "stage"
    target.mkdir()
    (target / "old.txt").write_text("previous run")

    real_rmtree = subprocess_runner.shutil.rmtree

    def failing_rmtree(path, *a, **k):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(subprocess_runner.shutil, "rmtree", failing_rmtree)
    with pytest.raises(SubprocessError, match="could not clear"):
        scratch_dir(tmp_path, "stage")

    monkeypatch.setattr(subprocess_runner.shutil, "rmtree", real_rmtree)


def test_scratch_dir_returns_an_empty_directory(tmp_path: Path) -> None:
    target = scratch_dir(tmp_path, "stage")
    (target / "junk.txt").write_text("x")
    again = scratch_dir(tmp_path, "stage")
    assert again == target
    assert not any(again.iterdir()), "a reused directory could report the previous run's artifacts"


def test_scratch_dir_refuses_a_directory_it_could_not_empty(tmp_path: Path, monkeypatch) -> None:
    """Belt and braces: even if removal silently no-ops, a non-empty dir is fatal."""
    target = tmp_path / "stage"
    target.mkdir()
    (target / "stale.txt").write_text("x")

    monkeypatch.setattr(subprocess_runner.shutil, "rmtree", lambda *a, **k: None)
    with pytest.raises(SubprocessError, match="not empty after clearing"):
        scratch_dir(tmp_path, "stage")


# =============================================================================
# Paths and environment at the process boundary
# =============================================================================


def test_entrypoints_are_absolute() -> None:
    """A relative entrypoint re-resolves against the child's own cwd.

    This worked by accident — Python absolutises `__file__` — which is exactly
    the kind of accident that stops working when something changes.
    """
    assert modes.SHEETSAGE_ENTRYPOINT.is_absolute()
    assert modes.ASR_ENTRYPOINT.is_absolute()


def test_child_environment_includes_home() -> None:
    """The HF libraries consult `~/.cache` and `~/.netrc`.

    Dropping HOME makes them log errors and, in some versions, fall back to a
    read-only location.
    """
    assert "HOME" in subprocess_runner._INHERITED_ENV
    assert "TRANSFORMERS_OFFLINE" in subprocess_runner._INHERITED_ENV


def test_cover_asks_the_child_to_work_offline() -> None:
    """`offline` must be sent, or `local_files_only` is always False and the child
    re-downloads what the endpoint already cached."""
    import inspect as pyinspect

    source = pyinspect.getsource(modes.prepare_cover)
    assert '"offline": True' in source


# =============================================================================
# Edits: an empty score is a caller error, not a request for a plan
# =============================================================================


def test_edit_distinguishes_no_score_from_an_empty_one() -> None:
    """`if params.abc:` treated `""` as "no score supplied" and silently planned.

    Validation rejects a blank score before it reaches here, so this is
    defence in depth — but the difference between "I did not send a score" and
    "I sent an empty one" is the difference between a plan and a typo.
    """
    import inspect as pyinspect

    source = pyinspect.getsource(modes.prepare_edit)
    assert "if params.abc is not None:" in source
    assert "if params.abc:" not in source
