"""Mode dispatch and the cover/edit pipelines.

Three modes, one endpoint:

- **create** — style + lyrics → song. Implemented in `handler.py`; this module
  only routes to it.
- **cover** — a source recording → its melody and lyrics → a new song in a
  different style. Two transcription subprocesses, then generation.
- **edit** — an existing score → a revised score → the song re-rendered. No
  subprocesses at all.

The cover chain, and why each step exists
----------------------------------------
```
source audio
  → SheetSage2 subprocess  → melody.abc   (melody only, no chord symbols)
  → Qwen3-ASR subprocess   → lyrics.txt   (words, with language auto-detected)
  → YuE2, in this process  → song         (cot="melody", target style)
```

The two transcription steps are separate *operations*, not one — the YuE2
authors' own reference is explicit that "source-separation, transcription, lyric
recognition and score-conditioned generation are distinct operations", and that
for an audio cover you transcribe the melody, review it, remove chord
annotations, and "obtain or check the lyrics separately". Keeping them as two
subprocesses also means each releases its VRAM before the next starts, which is
what holds the job's peak at YuE2's own ceiling rather than the sum of three
models.

The edit chain
--------------
```
existing score (or a fresh plan)
  → caller/agent revises score.abc
  → YuE2, in this process  → song   (cot="full", abc=revised)
```
No subprocess: editing is symbolic, and YuE2 is already resident. Editing
re-renders the whole song — the waveform outside the edit is **not** preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import abc_score
from subprocess_runner import (
    MAIN_INTERPRETER,
    run_stage,
    scratch_dir,
    write_request,
)

from schema import SongParameters

log = logging.getLogger(__name__)

#: Where the two isolated environments' entrypoints live, relative to this file.
SHEETSAGE_ENTRYPOINT = Path(__file__).parent / "transcribe_sheetsage" / "run.py"
ASR_ENTRYPOINT = Path(__file__).parent / "transcribe_asr" / "run.py"

#: Virtual environment directory names, under `subprocess_runner.VENV_ROOT`.
#: Both model families now run in the **main** environment. There is one
#: environment, not three.
#:
#: Verified on hardware, 2026-09-19, rather than assumed:
#:
#:   ASR        job 0576cb7f, v25   env reported torch 2.10.0+cu128
#:   SheetSage2 job f5792fce, v27   env reported torch 2.10.0+cu128,
#:                                  transformers 4.57.6
#:
#: SheetSage2's own pins were torch 2.8.0 / transformers 4.45.2 / numpy 1.24.3 and
#: qwen-asr had resolved torch 2.14.0, so both jumps are confirmed working. The
#: SheetSage2 score was checked rather than trusted: 168 note tokens across two
#: named melody voices and **zero chord symbols**, which is the melody-only
#: contract — an empty or chords-carrying score would also have returned 200.
#:
#: `MAIN_INTERPRETER` is kept as the constant because it is what the subprocess
#: runner understands, and the parameter remains: a future model family whose
#: pins genuinely conflict can still be given its own venv by passing its name
#: here. That was the original design and it is not wrong — it was simply
#: unnecessary for these two.
_SHEETSAGE_ENV = MAIN_INTERPRETER
_ASR_ENV = MAIN_INTERPRETER


def _asr_venv() -> str:
    """ASR's interpreter. The main environment — see the note above."""
    return _ASR_ENV


def _sheetsage_venv() -> str:
    """SheetSage2's interpreter. The main environment — see the note above."""
    return _SHEETSAGE_ENV


#: Per-stage budgets. Transcription is bounded by song length; the checkpoints
#: are lazy graphs, so these are generous but not unbounded.
SHEETSAGE_TIMEOUT_SECONDS = 900
ASR_TIMEOUT_SECONDS = 600

#: Re-exported from `schema`, which owns the vocabulary. One definition, so the
#: validator and the dispatcher cannot disagree about what a mode is called.
from schema import COVER, CREATE, EDIT, MODES  # noqa: E402  (grouped with the other schema import above)


class ModeError(RuntimeError):
    """A mode could not run, for a reason the caller should see."""


@dataclass
class ModeResult:
    """The outcome of one mode, in the shape the handler turns into a response."""

    mode: str
    #: The ABC score used for generation, if the mode supplied one.
    score_abc: str | None = None
    #: Lyrics to generate from. For `cover` this is either the transcription or
    #: the caller's own; for the other modes it is whatever the caller sent.
    lyrics: str | None = None
    #: Per-stage outcome, for the record and for response metadata.
    stages: dict[str, Any] = field(default_factory=dict)


def dispatch(mode: str, params: SongParameters, workdir: Path) -> ModeResult:
    """Route a validated job to its mode.

    `create` is handled entirely by the caller (`handler.generate`) because it
    needs the resident pipeline; this function exists so `cover` and `edit` are
    addressed the same way and so an unknown mode fails in one place.
    """
    if mode == CREATE:
        return ModeResult(mode=CREATE)
    if mode == COVER:
        return prepare_cover(params, workdir)
    if mode == EDIT:
        return prepare_edit(params, workdir)
    raise ModeError(f"unknown mode {mode!r}; expected one of {list(MODES)}")


# --- cover -------------------------------------------------------------------


def prepare_cover(params: SongParameters, workdir: Path) -> ModeResult:
    """Transcribe a source recording, returning the melody and lyrics to generate from.

    Requires `source_audio` — a path on the container's disk that the handler has
    already fetched from the job's URL. The two subprocesses run sequentially and
    exit, so their VRAM is released before generation begins.
    """
    if not params.source_audio:
        raise ModeError("cover mode needs 'source_audio': a URL or base64 payload for the recording to cover")
    source = Path(params.source_audio)
    if not source.is_file():
        raise ModeError(f"cover mode: source audio not found at {source}")

    stages: dict[str, Any] = {}

    # --- stage 1: melody, as ABC, in SheetSage2's own environment -------------
    sheetsage_dir = scratch_dir(workdir, "sheetsage")
    write_request(
        sheetsage_dir,
        {
            "audio": str(source),
            "melody_only": True,
            # These prompt selections are the upstream reference's for a melody
            # task. `chord_full` is deliberately absent: we want no harmony.
            "prompts": ["timestamp", "downbeat_meter", "structure", "key", "melody_full"],
        },
    )
    melody = run_stage(
        _sheetsage_venv(),
        SHEETSAGE_ENTRYPOINT,
        sheetsage_dir,
        timeout_seconds=SHEETSAGE_TIMEOUT_SECONDS,
        name="SheetSage2",
    )
    stages["sheetsage"] = {
        "ok": melody.ok,
        "seconds": round(melody.elapsed_seconds, 2),
        "error": melody.error,
        # Which stack produced this score — see the note on the ASR stage below.
        "environment": (melody.payload or {}).get("environment"),
    }
    if not melody.ok:
        raise ModeError(f"cover mode: melody transcription failed — {melody.error}")

    melody_abc = melody.payload.get("abc")
    if not melody_abc:
        raise ModeError("cover mode: SheetSage2 reported success but returned no ABC")

    # The subprocess is supposed to return melody-only ABC. Verify rather than
    # trust, and in this order:
    #
    #   1. Inspect. A quoted token that is not a valid chord is a hard error —
    #      we cannot tell whether it is harmony, so we can neither strip it nor
    #      pass it on. This is the check that actually protects the guarantee.
    #   2. Strip recognised chords. Their presence means the model ignored
    #      melody_only; upstream raises here, but a stripped melody still
    #      produces the cover that was asked for. The fact is recorded.
    #   3. Validate melody-only *after* stripping, which is now a real assertion
    #      rather than a restatement of step 1.
    #
    # Validating before stripping, as an earlier version did, made step 3
    # unreachable: strip_chords had already guaranteed chord-freedom.
    try:
        report = abc_score.inspect(melody_abc)
    except abc_score.AbcError as exc:
        raise ModeError(f"cover mode: transcribed melody is not usable — {exc}") from exc

    if report.unrecognised_quoted:
        preview = ", ".join(repr(c) for c in report.unrecognised_quoted[:6])
        raise ModeError(
            f"cover mode: SheetSage2 returned unrecognised chord symbol(s) {preview}. "
            "They may be harmony we cannot remove, which would make this something other than a cover."
        )

    try:
        melody_abc, removed = abc_score.strip_chords(melody_abc, source="SheetSage2 melody")
        abc_score.validate_native(melody_abc, source="SheetSage2 melody", require_melody_only=True)
    except abc_score.AbcError as exc:
        raise ModeError(f"cover mode: transcribed melody is not usable — {exc}") from exc

    if removed:
        # Recorded, not just logged: it means the model ignored melody_only, and
        # that is worth being able to see in the job's response.
        log.warning("cover mode: stripped %d chord symbols the transcriber left in", len(removed))
        stages["sheetsage"]["chords_stripped"] = removed

    # --- stage 2: lyrics, in Qwen3-ASR's own environment ----------------------
    #
    # A caller may supply their own lyrics — the upstream reference says to
    # "obtain or check the lyrics separately", and a listener knows them better
    # than an ASR pass over a full mix does. When they have, the transcription is
    # a convenience rather than a requirement: it still runs (it fills
    # lyrics.txt for the record), but its failure must not fail the job.
    #
    # An earlier version ran ASR unconditionally and raised on any failure
    # *before* consulting the flag, so supplying lyrics bought nothing and a
    # flaky transcription killed a job that already had everything it needed.
    asr_dir = scratch_dir(workdir, "asr")
    write_request(asr_dir, {"audio": str(source), "language": None, "offline": True})
    asr = run_stage(
        _asr_venv(),
        ASR_ENTRYPOINT,
        asr_dir,
        timeout_seconds=ASR_TIMEOUT_SECONDS,
        name="Qwen3-ASR",
        required=not params.lyrics_supplied,
    )
    stages["asr"] = {
        "ok": asr.ok,
        "seconds": round(asr.elapsed_seconds, 2),
        "error": asr.error,
        "used": not params.lyrics_supplied,
        "reason": "caller supplied lyrics" if params.lyrics_supplied else "transcribed",
        # The language the model detected in the recording. It was already
        # computed and discarded here — the child has returned it all along — so
        # this is a pass-through rather than new work. The official demo shows a
        # language field; ours reports what the model found rather than what a
        # caller set, which is the only version that can be honest for a cover.
        "language": (asr.payload or {}).get("language"),
        # Which stack produced this transcription. `cover` can run the ASR stage
        # under its own venv or the main environment, and the response is
        # otherwise identical — so without this the unification experiment cannot
        # be read from its own result. The child reports it; we pass it through.
        "environment": (asr.payload or {}).get("environment"),
    }

    if params.lyrics_supplied:
        lyrics = params.lyrics
        log.info("cover mode: using caller-supplied lyrics (ASR ok=%s, advisory only)", asr.ok)
    else:
        if not asr.ok:
            raise ModeError(f"cover mode: lyric transcription failed — {asr.error}")
        lyrics = str(asr.payload.get("text") or "").strip()
        if not lyrics:
            raise ModeError("cover mode: Qwen3-ASR returned no lyrics and none were supplied")

    log.info("cover mode: melody %d chars, lyrics %d chars", len(melody_abc), len(lyrics))
    return ModeResult(mode=COVER, score_abc=melody_abc, lyrics=lyrics, stages=stages)


# --- edit --------------------------------------------------------------------


def prepare_edit(params: SongParameters, workdir: Path) -> ModeResult:
    """Return the score an edit job should re-render from.

    An edit job either supplies a revised score (`abc=`) or asks for the plan
    YuE2 would write and edits it externally. Both end in generation with
    `cot="full"` so the supplied harmony is kept — an edit that dropped the
    chords would not be an edit.

    Note that editing re-renders the whole song. YuE2 does not preserve the
    waveform outside the edited region, which callers routinely assume it does.
    """
    # `is not None`, not truthiness: `""` is a caller sending an empty score,
    # which validation rejects — while `None` is a caller asking for a plan. The
    # difference is a typo versus a request.
    if params.abc is not None:
        try:
            report = abc_score.validate_native(params.abc, source="edited score")
        except abc_score.AbcError as exc:
            raise ModeError(f"edit mode: {exc}") from exc
        log.info(
            "edit mode: rendering a supplied score (key %s, %d music lines)",
            report.key,
            report.music_lines,
        )
        return ModeResult(mode=EDIT, score_abc=params.abc, stages={"score": {"source": "supplied"}})

    # No score supplied: the caller wants the plan YuE2 writes, to edit it.
    # Generating it here would require the pipeline, so the handler does that;
    # signal intent rather than pretend to have a score.
    log.info("edit mode: no abc supplied — the handler will write a plan to edit")
    return ModeResult(mode=EDIT, score_abc=None, stages={"score": {"source": "plan"}})


def mode_needs_pipeline(mode: str) -> bool:
    """Whether a mode requires the resident YuE2 pipeline.

    All three do eventually — every mode ends in generation. Kept explicit
    because that is the property that makes the boot-before-first-job contract
    apply to every mode, not just `create`.
    """
    return mode in MODES
