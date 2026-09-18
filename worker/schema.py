"""Request validation for the YuE2 worker.

Validation boundaries are mirrored deliberately from the pipeline's own
`yue2.protocol.SongRequest.__post_init__` (yue2_infer 0.1.5). The pipeline does
validate — but it validates *after* the model is resident and the job has
started, which on a 4090 costs ~70 seconds before the caller learns their seed
was a string. This module rejects the same inputs in microseconds.

Where the pipeline is more permissive than makes sense for a network-facing
endpoint, we are stricter and say so in a comment.

The public surface is `validate_job()` -> `SongParameters` and two error types:
`ValidationError` for bad input (the caller's fault, safe to echo back) and
`MissingInputError` for a job with no `input` key at all.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from config import (
    CFG_SCALE_MAX,
    CFG_SCALE_MIN,
    DEFAULT_CFG_SCALE,
    DEFAULT_COT,
    DEFAULT_SEED,
    SEED_MAX,
    VALID_COT,
)

#: Matches `yue2.protocol.SongRequest.id`'s filename-safety rule, which exists
#: because the pipeline writes `plan.json`/`score.abc` into a directory named
#: after it.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")

#: Guard against a caller posting an unbounded body, before any token counting.
_MAX_TEXT_BYTES = 2 * 1024 * 1024

#: Combined style + lyrics + abc budget, in bytes.
#:
#: The pipeline has a hard token ceiling, not a byte one: `sampling.py:62`
#: raises `ValueError` when `len(prefix) + max_tokens > CONTEXT`, where CONTEXT
#: is 24576 and the default `max_tokens` is 9000 — leaving ~15 576 tokens for
#: the prompt. Measured against the real checkpoint tokenizer, the largest
#: lyrics string that fits is on the order of 78 KB, and style and a supplied
#: ABC each eat into that.
#:
#: 64 KiB is deliberately below the measured ceiling rather than at it: token
#: density varies with language (CJK lyrics tokenize far denser per byte than
#: English), and a limit that is right for one song is wrong for the next. The
#: point is to refuse in microseconds what the pipeline would refuse after
#: loading a 12 GB model, not to squeeze out the last kilobyte.
#:
#: This is a *pre-check*, not a substitute: `GenerationFailed` still catches the
#: pipeline's own ValueError for anything that slips through.
_MAX_PROMPT_BYTES = 64 * 1024


#: The modes this worker accepts. Defined here rather than in `modes.py` so the
#: validator owns the vocabulary and `modes.py` can import it without a cycle.
CREATE, COVER, EDIT = "create", "cover", "edit"
MODES = (CREATE, COVER, EDIT)


class ValidationError(ValueError):
    """Bad job input. Safe to return verbatim to the caller."""


class MissingInputError(ValidationError):
    """The job dict had no usable `input` object."""


@dataclass(frozen=True)
class SongParameters:
    """A validated generation request — the only thing `handler` passes onward."""

    style: str
    lyrics: str
    cot: str
    seed: int
    cfg_scale: float | None
    abc: str | None
    #: Per-job artifact directory name. Defaults to a value derived from the job
    #: id so two concurrent jobs on the same worker cannot collide.
    id: str
    #: `cover` only: local path to the recording being covered. The handler
    #: fetches it from the job's URL before dispatch, so this is always a path on
    #: the container's disk by the time a mode sees it.
    source_audio: str | None = None
    #: `cover` only: whether the caller supplied lyrics rather than relying on
    #: transcription. The upstream reference says to "obtain or check the lyrics
    #: separately" — a listener usually knows them better than an ASR pass over a
    #: full mix does, so an explicit lyric wins over the transcription.
    lyrics_supplied: bool = False

    def to_request_json(self) -> dict[str, Any]:
        """The subset we echo back in the response and persist beside the audio."""
        return {
            "style": self.style,
            "cot": self.cot,
            "seed": self.seed,
            "cfg_scale": self.cfg_scale,
            "id": self.id,
            "has_abc": self.abc is not None,
            "lyrics_chars": len(self.lyrics),
        }


def _require_text(value: Any, name: str, *, max_bytes: int = _MAX_TEXT_BYTES) -> str:
    if value is None:
        raise ValidationError(f"{name!r} is required")
    if not isinstance(value, str):
        raise ValidationError(f"{name!r} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValidationError(f"{name!r} must not be empty")
    encoded = len(value.encode("utf-8"))
    if encoded > max_bytes:
        raise ValidationError(f"{name!r} is {encoded} bytes; limit is {max_bytes}")
    return value


def _coerce_seed(value: Any) -> int:
    if value is None:
        return DEFAULT_SEED
    # Bools must be rejected *before* any coercion: `int(True) == 1`, so a
    # `seed=True` caller would otherwise silently get seed 1. Mirrors the
    # pipeline's `type(self.seed) is not int` check.
    if isinstance(value, bool):
        raise ValidationError(f"'seed' must be an integer, got {value!r}")
    if not isinstance(value, int):
        try:
            # Accept "12345" and 12345.0 — JSON callers send both — but never a
            # lossy float like 1.5.
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            value = int(value)
        except (TypeError, ValueError):
            raise ValidationError(f"'seed' must be an integer, got {value!r}") from None
    if not 0 <= value < SEED_MAX:
        raise ValidationError(f"'seed' must be in [0, 2**63), got {value}")
    return value


def _coerce_cfg_scale(value: Any) -> float | None:
    if value is None:
        return DEFAULT_CFG_SCALE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"'cfg_scale' must be a number, got {type(value).__name__}")
    scale = float(value)
    if not math.isfinite(scale) or not CFG_SCALE_MIN <= scale <= CFG_SCALE_MAX:
        raise ValidationError(f"'cfg_scale' must be finite and in [{CFG_SCALE_MIN}, {CFG_SCALE_MAX}], got {value}")
    return scale


def _resolve_cot(value: Any) -> str:
    cot = DEFAULT_COT if value is None else value
    if not isinstance(cot, str) or cot not in VALID_COT:
        raise ValidationError(f"'cot' must be one of {list(VALID_COT)}, got {value!r}")
    return cot


def validate_job(job: Mapping[str, Any] | None) -> SongParameters:
    """Validate a raw RunPod job into `SongParameters`.

    Raises `MissingInputError` when there is no `input` object and
    `ValidationError` for any field-level problem.
    """
    if not isinstance(job, Mapping):
        raise MissingInputError(f"job must be an object, got {type(job).__name__}")
    raw = job.get("input")
    if not isinstance(raw, Mapping):
        raise MissingInputError("job is missing an 'input' object")

    # Resolved here so the per-mode lyric rule below has it. The handler calls
    # `validate_mode` as well, but this function must be usable on its own.
    mode = validate_mode(raw)

    style = _require_text(raw.get("style"), "style")

    # Lyrics are required for `create` and `edit`, but **not** for `cover`: a
    # cover's words come from the recording, transcribed by Qwen3-ASR. Requiring
    # them unconditionally would make the whole cover path unusable and would
    # contradict the upstream reference, which says to "obtain or check the
    # lyrics separately" — supplying them is an *option*, not a precondition.
    #
    # `validate_mode_inputs` has already confirmed the mode is one this worker
    # knows, so an unrecognised mode never reaches here.
    lyrics_raw = raw.get("lyrics")
    if mode == COVER:
        lyrics = _require_text(lyrics_raw, "lyrics") if lyrics_raw is not None else ""
    else:
        lyrics = _require_text(lyrics_raw, "lyrics")
    cot = _resolve_cot(raw.get("cot"))
    seed = _coerce_seed(raw.get("seed"))
    cfg_scale = _coerce_cfg_scale(raw.get("cfg_scale"))

    abc_raw = raw.get("abc")
    abc: str | None
    if abc_raw is None:
        abc = None
    else:
        abc = _require_text(abc_raw, "abc")
        # Same rule the pipeline enforces: an external score needs a plan to
        # hang off, and `cot="off"` has none.
        if cot == "off":
            raise ValidationError("'abc' requires cot='melody' or cot='full'")

    # The pipeline refuses an over-long prompt only after the model is resident,
    # ~70 seconds in. Its ceiling is measured in tokens; ours is a conservative
    # byte pre-check so the common case fails here instead.
    prompt_bytes = len(style.encode("utf-8")) + len(lyrics.encode("utf-8"))
    if abc is not None:
        prompt_bytes += len(abc.encode("utf-8"))
    if prompt_bytes > _MAX_PROMPT_BYTES:
        raise ValidationError(
            f"style + lyrics + abc is {prompt_bytes} bytes; the pipeline's context "
            f"window allows about {_MAX_PROMPT_BYTES}. Shorten the lyrics or the style prompt."
        )

    request_id = raw.get("id")
    if request_id is None:
        # RunPod job ids are already filename-safe, but they are not guaranteed
        # to be, so sanitise rather than trust.
        request_id = _sanitise_id(str(job.get("id", "song")))
    elif not isinstance(request_id, str) or not _ID_PATTERN.fullmatch(request_id) or request_id in {".", ".."}:
        raise ValidationError(f"'id' must match {_ID_PATTERN.pattern}, got {request_id!r}")

    # `cover` fields, validated above by `validate_mode_inputs` — but checking
    # that a key exists in the raw input and then not carrying it forward is
    # precisely how three cover bugs shipped at once. They must reach the params.
    source_audio = raw.get("source_audio")
    # A caller-supplied lyric wins over the transcription. Detected here rather
    # than inferred in the mode, because only this layer knows whether the key
    # was present or merely defaulted.
    lyrics_supplied = "lyrics" in raw

    return SongParameters(
        style=style,
        lyrics=lyrics,
        cot=cot,
        seed=seed,
        cfg_scale=cfg_scale,
        abc=abc,
        id=request_id,
        source_audio=str(source_audio) if source_audio is not None else None,
        lyrics_supplied=lyrics_supplied,
    )


def _sanitise_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value).lstrip(".-")[:180]
    return cleaned or "song"


def validate_mode(raw: Mapping[str, Any] | None) -> str:
    """Resolve the worker's `mode` field: `create`, `cover` or `edit`.

    All three are implemented. An unrecognised value is rejected here rather
    than deep inside a pipeline, so a typo fails in microseconds.
    """
    if not isinstance(raw, Mapping):
        return CREATE
    mode = raw.get("mode", CREATE)
    if mode in MODES:
        return str(mode)
    raise ValidationError(f"'mode' must be one of {list(MODES)}, got {mode!r}")


def validate_mode_inputs(mode: str, raw: Mapping[str, Any] | None) -> None:
    """Check the fields a mode needs but that `validate_job` cannot know about.

    Separate from `validate_job` because these requirements are per-mode;
    folding them in would make `create` reject payloads it has no opinion about.
    Called by the handler once the mode is known.
    """
    if mode != COVER:
        return
    if not isinstance(raw, Mapping):
        raise ValidationError("cover mode requires an 'input' object with 'source_audio'")
    source = raw.get("source_audio")
    if source is None:
        raise ValidationError(
            "cover mode requires 'source_audio' — a URL or base64 payload for the recording to cover. "
            "YuE2 has no direct audio-upload argument; the recording is transcribed first."
        )
    if not isinstance(source, str) or not source.strip():
        raise ValidationError("'source_audio' must be a non-empty string")
