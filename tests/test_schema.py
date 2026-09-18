"""Validation tests.

Every boundary asserted here is mirrored from `yue2.protocol.SongRequest`'s own
`__post_init__`, so these tests double as a contract check: if upstream tightens
a bound, one of these should fail and point at the divergence.
"""

from __future__ import annotations

import pytest

from schema import (
    INSTRUMENTAL_LYRICS,
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


# =============================================================================
# instrumental — a defined input, not a promise
# =============================================================================
#
# YuE2 has no instrumental mode. `yue2_infer.protocol.SongRequest` declares
# `lyrics: str` as required, the prompt is always `[Tags]\n{style}\n[Lyrics]\n
# {lyrics}\n` at every CoT, and the word "instrumental" appears nowhere in the
# wheel — verified by extracting it. The authors' own skill describes every path
# as `style + lyrics`.
#
# So a caller wanting no vocals has to put something in the slot. `instrumental`
# is that something, and these tests pin the input contract. Whether the *audio*
# is instrumental is a separate question, answerable only on a GPU.


def test_instrumental_fills_the_lyrics_slot() -> None:
    params = validate_job({"input": {"style": "lofi", "instrumental": True}})
    assert params.instrumental is True
    assert params.lyrics == INSTRUMENTAL_LYRICS
    assert params.lyrics.strip(), "the slot must not be empty — the prompt always carries it"


def test_the_placeholder_is_a_section_tag_not_words() -> None:
    """It matches the bracketed-tag convention of the authors' own example.

    Their skill says to put "section tags and actual words" in `lyrics`, and
    their example prompt uses `[verse]` / `[chorus]`. A single wordless tag is
    the closest thing to 'no words' the format has.
    """
    assert INSTRUMENTAL_LYRICS.startswith("[")
    assert INSTRUMENTAL_LYRICS.endswith("]")
    # No words outside the brackets.
    assert INSTRUMENTAL_LYRICS[1:-1].replace(" ", "").isalpha()


def test_instrumental_defaults_to_false() -> None:
    params = validate_job({"input": {"style": "lofi", "lyrics": "[verse]\nhi"}})
    assert params.instrumental is False


def test_str_is_not_a_boolean() -> None:
    """`bool("false")` is `True`, so the string must be rejected, not coerced.

    Same reasoning as `seed`, where `int(True) == 1` would have silently given a
    seed of 1. A caller sending `"false"` and receiving an instrumental is the
    kind of failure that only shows up in the audio.
    """
    for value in ("false", "true", "0", "", 1, 0):
        with pytest.raises(ValidationError):
            validate_job({"input": {"style": "lofi", "instrumental": value, "lyrics": "x"}})


def test_instrumental_with_lyrics_is_a_contradiction() -> None:
    """Not interpreted — refused. One of the two is a mistake.

    Guessing which would silently produce the wrong song: the caller either
    wanted words they just sent, or wanted no vocals and sent words by copy-paste.
    """
    with pytest.raises(ValidationError) as excinfo:
        validate_job({"input": {"style": "lofi", "instrumental": True, "lyrics": "real words"}})
    message = str(excinfo.value)
    assert "instrumental" in message and "lyrics" in message
    assert "one or the other" in message, "the error must say what to do, not just what is wrong"


def test_an_empty_lyrics_string_is_not_a_contradiction() -> None:
    """`lyrics: ""` alongside `instrumental: true` is redundant, not conflicting.

    Some clients default optional strings to empty. Refusing that would make the
    flag unusable from exactly the callers most likely to want it, so an empty
    string is treated as absent while a non-empty one is a real conflict.
    """
    params = validate_job({"input": {"style": "lofi", "instrumental": True, "lyrics": ""}})
    assert params.instrumental is True
    assert params.lyrics == INSTRUMENTAL_LYRICS


def test_instrumental_rejects_lyrics_but_accepts_none() -> None:
    """The boundary, stated once: None and "" are absent; anything else is words."""
    assert validate_job({"input": {"style": "s", "instrumental": True, "lyrics": None}}).instrumental
    with pytest.raises(ValidationError):
        validate_job({"input": {"style": "s", "instrumental": True, "lyrics": " "}})


def test_instrumental_is_reported_in_the_echoed_request() -> None:
    """The response records it, so a caller can tell what was generated.

    Without this, `instrumental: true` and a caller-passing-`[instrumental]` are
    indistinguishable after the fact — and the persisted `request.json` is the
    only durable record of what was asked for.
    """
    params = validate_job({"input": {"style": "lofi", "instrumental": True}})
    assert params.to_request_json()["instrumental"] is True
    other = validate_job({"input": {"style": "lofi", "lyrics": "words"}})
    assert other.to_request_json()["instrumental"] is False


def test_cover_still_derives_lyrics_when_none_are_supplied() -> None:
    """The flag must not disturb the one mode that legitimately has no lyrics.

    A cover with no lyrics and no words relies on transcription; adding
    `instrumental` must not pre-empt that with the placeholder.
    """
    params = validate_job(
        {"input": {"mode": "cover", "style": "lofi", "source_audio": "https://example.invalid/a.flac"}}
    )
    assert params.lyrics == "", "cover leaves the slot empty for the transcription to fill"
    assert params.instrumental is False
