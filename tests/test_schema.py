"""Validation tests.

Every boundary asserted here is mirrored from `yue2.protocol.SongRequest`'s own
`__post_init__`, so these tests double as a contract check: if upstream tightens
a bound, one of these should fail and point at the divergence.
"""

from __future__ import annotations

import pytest

from schema import (
    MissingInputError,
    ValidationError,
    validate_job,
    validate_mode,
    validate_mode_inputs,
)

VALID = {"style": "City Pop, upbeat", "lyrics": "[Verse]\nhello"}


def job(**overrides: object) -> dict:
    payload = {**VALID, **overrides}
    return {"id": "job-1", "input": payload}


# --- happy path --------------------------------------------------------------


def test_minimal_job_uses_protocol_defaults() -> None:
    params = validate_job(job())
    assert params.style == VALID["style"]
    assert params.lyrics == VALID["lyrics"]
    # Defaults must match yue2.protocol.SongRequest's own.
    assert params.cot == "full"
    assert params.seed == 831001
    assert params.cfg_scale is None
    assert params.abc is None


def test_id_defaults_to_sanitised_job_id() -> None:
    assert validate_job({"id": "abc123", "input": dict(VALID)}).id == "abc123"


def test_explicit_id_is_honoured() -> None:
    assert validate_job(job(id="my-song_v2")).id == "my-song_v2"


@pytest.mark.parametrize("cot", ["full", "melody", "off"])
def test_each_valid_cot_accepted(cot: str) -> None:
    assert validate_job(job(cot=cot)).cot == cot


# --- missing / malformed input ----------------------------------------------


@pytest.mark.parametrize("bad", [None, {}, {"input": None}, {"input": "nope"}, {"input": []}])
def test_missing_input_object(bad: object) -> None:
    with pytest.raises(MissingInputError):
        validate_job(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["style", "lyrics"])
def test_required_text_fields(field: str) -> None:
    payload = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ValidationError, match=field):
        validate_job({"id": "x", "input": payload})


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_blank_text_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        validate_job(job(style=bad))


@pytest.mark.parametrize("bad", [123, ["a"], {"a": 1}, True])
def test_non_string_text_rejected(bad: object) -> None:
    with pytest.raises(ValidationError):
        validate_job(job(lyrics=bad))


# --- seed --------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 831001, 2**63 - 1])
def test_valid_seeds(seed: int) -> None:
    assert validate_job(job(seed=seed)).seed == seed


@pytest.mark.parametrize("seed", [-1, 2**63, "abc", 1.5, None])
def test_invalid_seeds(seed: object) -> None:
    if seed is None:
        # None is the documented "use the default" signal, not an error.
        assert validate_job(job(seed=None)).seed == 831001
        return
    with pytest.raises(ValidationError):
        validate_job(job(seed=seed))


def test_bool_seed_rejected_despite_being_an_int() -> None:
    """`True` is an int in Python; it is still a caller bug."""
    with pytest.raises(ValidationError):
        validate_job(job(seed=True))


def test_numeric_string_seed_coerced() -> None:
    """JSON callers send "12345" often enough that rejecting it is hostile."""
    assert validate_job(job(seed="12345")).seed == 12345


def test_integral_float_seed_coerced() -> None:
    assert validate_job(job(seed=100.0)).seed == 100


# --- cfg_scale ---------------------------------------------------------------


@pytest.mark.parametrize("scale", [0, 1.0, 1.2, 20, 19.99])
def test_valid_cfg_scales(scale: float) -> None:
    assert validate_job(job(cfg_scale=scale)).cfg_scale == float(scale)


@pytest.mark.parametrize("scale", [-0.1, 20.1, float("inf"), float("nan"), "high", True])
def test_invalid_cfg_scales(scale: object) -> None:
    with pytest.raises(ValidationError):
        validate_job(job(cfg_scale=scale))


# --- cot / abc interaction ---------------------------------------------------


def test_unknown_cot_rejected() -> None:
    with pytest.raises(ValidationError, match="cot"):
        validate_job(job(cot="melody-only"))


def test_abc_with_full_is_allowed() -> None:
    params = validate_job(job(abc="X:1\nK:C\nC D E F|", cot="full"))
    assert params.abc is not None


def test_abc_with_off_rejected() -> None:
    """Mirrors the pipeline: an external score needs a plan to hang off."""
    with pytest.raises(ValidationError, match="cot='melody' or cot='full'"):
        validate_job(job(abc="X:1\nK:C\nC|", cot="off"))


def test_blank_abc_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_job(job(abc="   "))


# --- id ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../etc/passwd", ".", "..", "-leading", "a" * 200, 12])
def test_unsafe_id_rejected(bad: object) -> None:
    with pytest.raises(ValidationError, match="id"):
        validate_job(job(id=bad))


# --- size guard --------------------------------------------------------------


def test_oversized_lyrics_rejected() -> None:
    with pytest.raises(ValidationError, match="limit is"):
        validate_job(job(lyrics="x" * (2 * 1024 * 1024 + 1)))


# --- mode --------------------------------------------------------------------


def test_create_mode_accepted() -> None:
    assert validate_mode({"mode": "create"}) == "create"
    assert validate_mode({}) == "create"


@pytest.mark.parametrize("mode", ["cover", "edit"])
def test_all_three_modes_are_accepted(mode: str) -> None:
    """Stage 03 added `cover` and `edit`. This test previously asserted they were
    *rejected* as unimplemented, so it is inverted rather than deleted — the
    vocabulary is still worth pinning."""
    assert validate_mode({"mode": mode}) == mode


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValidationError, match="must be one of"):
        validate_mode({"mode": "creat"})


def test_cover_mode_requires_source_audio() -> None:
    """Without a recording there is no cover, and YuE2 takes no audio argument."""
    with pytest.raises(ValidationError, match="source_audio"):
        validate_mode_inputs("cover", {"mode": "cover"})


def test_cover_mode_accepts_source_audio() -> None:
    validate_mode_inputs("cover", {"mode": "cover", "source_audio": "https://example.invalid/a.wav"})


def test_other_modes_do_not_require_source_audio() -> None:
    """The check is per-mode; `create` must not be asked for a recording."""
    validate_mode_inputs("create", {"style": "x", "lyrics": "y"})
    validate_mode_inputs("edit", {"style": "x", "lyrics": "y"})


# --- echo shape --------------------------------------------------------------


def test_to_request_json_omits_full_text() -> None:
    """The echoed request must not carry the whole lyric sheet back."""
    params = validate_job(job(abc="X:1\nK:C\nC|"))
    echoed = params.to_request_json()
    assert "lyrics" not in echoed
    assert echoed["lyrics_chars"] == len(VALID["lyrics"])
    assert echoed["has_abc"] is True
