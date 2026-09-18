"""Tests for ABC score validation and chord stripping.

`abc_score.py` adapts the YuE2 authors' own `abc_tools.py` (Apache 2.0). The
fixtures here are real native-format scores, including one produced by an actual
job, so the validator is exercised against the format YuE2 really emits rather
than one we imagined.
"""

from __future__ import annotations

import abc_score
import pytest
from abc_score import AbcError, find_chords, inspect, strip_chords, validate_native

#: A minimal but complete native two-voice score.
VALID = """X:1
T:
M:4/4
L:1/32
Q:1/4=118
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Dm
% intro
V: Vocal
"Gm7"z32|"C7"z32|
V: Ins
z8f8f8f8|z8e8e8e8|
V: Vocal
"Dm7"a8f8d8"C7"z4a4-|
V: Ins
z8d8d8d8|
"""

#: The same score with no chord symbols — what a cover melody must look like.
MELODY_ONLY = """X:1
T:
M:4/4
L:1/32
Q:1/4=118
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Dm
% intro
V: Vocal
z32|z32|
V: Ins
z8f8f8f8|z8e8e8e8|
V: Vocal
a8f8d8z4a4-|
V: Ins
z8d8d8d8|
"""


# --- structural validation ---------------------------------------------------


def test_a_native_score_passes() -> None:
    report = validate_native(VALID)
    assert report.voices == ["Vocal", "Ins"]
    assert report.key == "Dm"
    assert report.meter == "4/4"
    assert report.unit == "1/32"
    assert report.tempo == 118


def test_chords_are_reported_not_rejected_by_default() -> None:
    """`cot="full"` legitimately carries harmony, so chords are only an error
    when a caller asks for melody-only."""
    report = validate_native(VALID)
    assert report.chords == ["Gm7", "C7", "Dm7", "C7"]


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("empty", lambda s: "", "empty"),
        ("no X:", lambda s: s.replace("X:1", "Q:1", 1), "X: index"),
        ("no meter", lambda s: s.replace("M:4/4\n", "", 1), "M: meter"),
        ("bad meter", lambda s: s.replace("M:4/4", "M:4/3", 1), "power of two"),
        ("no unit", lambda s: s.replace("L:1/32\n", "", 1), "L: unit"),
        ("bad unit", lambda s: s.replace("L:1/32", "L:1/24", 1), "power of two"),
        ("no key", lambda s: s.replace("K:Dm\n", "", 1), "K: key"),
        ("unsupported key", lambda s: s.replace("K:Dm", "K:Hm", 1), "unsupported key"),
        # Removing just the declaration is not enough — a score repeats `V:`
        # before every section. This drops every Ins line, which is what
        # "a score with one voice" actually means.
        (
            "one voice",
            lambda s: "\n".join(ln for ln in s.splitlines() if not ln.startswith("V: Ins")) + "\n",
            "native voices",
        ),
        ("no tempo", lambda s: s.replace("Q:1/4=118\n", "", 1), "tempo"),
    ],
)
def test_structural_failures_name_the_problem(label: str, mutate, match: str) -> None:
    with pytest.raises(AbcError, match=match):
        validate_native(mutate(VALID), source="test")


def test_chords_rejected_when_melody_only_is_required() -> None:
    with pytest.raises(AbcError, match="chord symbols"):
        validate_native(VALID, source="cover melody", require_melody_only=True)


def test_melody_only_score_accepted_when_required() -> None:
    report = validate_native(MELODY_ONLY, source="cover melody", require_melody_only=True)
    assert not report.has_chords


def test_the_error_names_the_source() -> None:
    """A caller editing a score needs to know *which* score was rejected."""
    with pytest.raises(AbcError, match="edited score"):
        validate_native("", source="edited score")


# --- chord detection ---------------------------------------------------------


def test_find_chords_on_a_clean_melody_is_empty() -> None:
    assert find_chords(MELODY_ONLY) == []


def test_header_quotes_are_not_mistaken_for_chords() -> None:
    """The whole reason stripping is done per music line.

    Voice definitions legitimately contain quoted text — `name="Vocal Melody"`.
    A naive `"..."` removal would corrupt the header, and `Vocal Melody` would
    be reported as a chord if the matcher were not grammar-aware.
    """
    report = inspect(MELODY_ONLY)
    assert report.chords == []
    assert "Vocal" in report.voices and "Ins" in report.voices


def test_quoted_non_chord_text_on_a_music_line_is_not_a_chord() -> None:
    """Only text matching the chord grammar counts as harmony."""
    score = MELODY_ONLY.replace("z8f8f8f8|", '"see note"z8f8f8f8|', 1)
    assert find_chords(score) == []


def test_chord_qualities_are_recognised() -> None:
    for symbol in ("C", "Cm", "C7", "Cmaj7", "Cm7", "Cdim7", "Cm7b5", "Csus4", "Cm(maj7)", "F#m7b5/A"):
        score = MELODY_ONLY.replace("z8f8f8f8|", f'"{symbol}"z8f8f8f8|', 1)
        assert find_chords(score) == [symbol], symbol


# --- chord stripping ---------------------------------------------------------


def test_strip_removes_chords_and_keeps_the_melody() -> None:
    stripped, removed = strip_chords(VALID)
    assert removed == ["Gm7", "C7", "Dm7", "C7"]
    assert find_chords(stripped) == []
    # The notes are untouched — that is the invariant that makes stripping safe.
    before = list(abc_score._note_signature(VALID))
    after = list(abc_score._note_signature(stripped))
    assert before == after


def test_strip_preserves_the_header() -> None:
    stripped, _ = strip_chords(VALID)
    assert 'name="Vocal Melody"' in stripped
    assert 'name="Ins Melody"' in stripped


def test_stripping_a_clean_melody_is_a_no_op() -> None:
    stripped, removed = strip_chords(MELODY_ONLY)
    assert removed == []
    assert stripped == MELODY_ONLY


def test_stripped_score_still_validates_as_melody_only() -> None:
    stripped, _ = strip_chords(VALID)
    validate_native(stripped, source="stripped", require_melody_only=True)


def test_strip_is_idempotent() -> None:
    once, _ = strip_chords(VALID)
    twice, removed = strip_chords(once)
    assert removed == []
    assert once == twice


# --- the real generated score ------------------------------------------------


REAL_SCORE = """X:1
T:
M:4/4
L:1/32
Q:1/4=118
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Dm
% intro
V: Vocal
Z|"Gm7"z32|"C7"z32|"Am7"z32|
V: Ins
Z|f8f4z4f4z2f4z2f2f2|z4e4e4z4e4z2e4z2e2e2|z8e4z4e4z2e4z2e2e2|
V: Vocal
"Dm7"z32|"Gm7"z32|"C7"z32|"Dm7"z32|
V: Ins
z4f4f4z4f4f4e8|f8f4z4f6f6f2f2|z4e4e4z4e6e6e2e2|z8e4z4e6e6d2d2|
V: Vocal
"Dm7"z32|
V: Ins
z8d8d8d8|
% verse
V: Vocal
"Gm7"a8f8d8"C7"z4a4-|"C7"a4f8d8c8e4-|"Am7"e16-e4c8c4|"Dm7"A8z24|
V: Ins
Z4|
"""


def test_a_score_from_a_real_job_validates() -> None:
    """Taken verbatim from an actual generation.

    The validator is only useful if it accepts what YuE2 really emits — this is
    the regression guard against a rule stricter than the format.
    """
    report = validate_native(REAL_SCORE, source="real job")
    assert report.key == "Dm"
    assert report.is_two_voice
    # Every chord symbol in the fixture, in order — counted, not guessed.
    assert report.chords == ["Gm7", "C7", "Am7", "Dm7", "Gm7", "C7", "Dm7", "Dm7", "Gm7", "C7", "C7", "Am7", "Dm7"]


def test_the_real_score_strips_cleanly() -> None:
    stripped, removed = strip_chords(REAL_SCORE)
    assert len(removed) == 13
    validate_native(stripped, source="real job melody", require_melody_only=True)


def test_multi_measure_rests_survive() -> None:
    """`Z` rests are part of the format; stripping must not disturb them."""
    stripped, _ = strip_chords(REAL_SCORE)
    assert "Z4|" in stripped
