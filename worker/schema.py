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

#: A FLAC of a long song is tens of MB. 2 MiB of UTF-8 is far more lyrics than
#: any song needs and caps the damage a hostile caller can do to our memory.
_MAX_TEXT_BYTES = 2 * 1024 * 1024


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

    style = _require_text(raw.get("style"), "style")
    lyrics = _require_text(raw.get("lyrics"), "lyrics")
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

    request_id = raw.get("id")
    if request_id is None:
        # RunPod job ids are already filename-safe, but they are not guaranteed
        # to be, so sanitise rather than trust.
        request_id = _sanitise_id(str(job.get("id", "song")))
    elif not isinstance(request_id, str) or not _ID_PATTERN.fullmatch(request_id) or request_id in {".", ".."}:
        raise ValidationError(f"'id' must match {_ID_PATTERN.pattern}, got {request_id!r}")

    return SongParameters(
        style=style,
        lyrics=lyrics,
        cot=cot,
        seed=seed,
        cfg_scale=cfg_scale,
        abc=abc,
        id=request_id,
    )


def _sanitise_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value).lstrip(".-")[:180]
    return cleaned or "song"


def validate_mode(raw: Mapping[str, Any] | None) -> str:
    """Resolve the worker's `mode` field.

    Only `create` is implemented in this stage. `cover` and `edit` are named in
    the locked product decisions and arrive in Stage 03 — rejecting them by name
    (rather than as an unknown value) tells a caller the difference between
    "you made a typo" and "not built yet".
    """
    if not isinstance(raw, Mapping):
        return "create"
    mode = raw.get("mode", "create")
    if mode == "create":
        return "create"
    if mode in {"cover", "edit"}:
        raise ValidationError(f"mode {mode!r} is not implemented in this worker build (Stage 03)")
    raise ValidationError(f"'mode' must be 'create', got {mode!r}")
