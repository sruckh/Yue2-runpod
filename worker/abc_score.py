"""ABC score validation and chord handling for the YuE2 worker.

YuE2 consumes a *native two-voice ABC* format — the same shape it emits itself.
Anything fed back through `abc=` (a cover melody, or an edited score) has to be
in that format or generation fails deep inside the pipeline.

This module is the pre-flight check for that, plus the chord handling the cover
path needs.

Provenance
----------
The regexes and the structural rules here are adapted from the YuE2 authors' own
`skills/yue2-music/scripts/abc_tools.py` in
`github.com/multimodal-art-projection/YuE`, which is Apache 2.0 — the same
license as this repository. The upstream rules are the authority on what YuE2
accepts; where this module is deliberately less strict, it says so.

What is *not* ported
--------------------
Upstream also validates every note: octave-mark consistency, tie resolution and
pitch equivalence, MIDI range, and per-measure duration arithmetic. Those are
left to YuE2's own tokenizer, which is the final authority and which we cannot
run here (no GPU, no weights). Duplicating them would risk rejecting input that
YuE2 would have accepted — a worse failure than deferring.

What *is* enforced is everything that fails cheaply and loudly: missing headers,
a malformed key or meter, music lines that do not close, an unbalanced
Vocal/Ins pair, and above all **chord symbols in a melody**, which is the one
error the cover path is specifically exposed to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The two voices YuE2's native format requires, in order.
VOICES = ("Vocal", "Ins")

#: Note lengths the native format allows, in 1/32-note units.
NATIVE_DURATIONS = frozenset({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48})

#: Chord qualities upstream's grammar recognises.
_CHORD_QUALITIES = (
    "",
    "m",
    "dim",
    "aug",
    "7",
    "maj7",
    "m7",
    "dim7",
    "m7b5",
    "sus4",
    "sus2",
    "6",
    "m6",
    "7sus4",
    "m(maj7)",
)

_PITCH_NAME = r"[A-G](?:bb|##|b|#)?"

#: A musical chord symbol as written inside double quotes: `"Gm7"`, `"C7"`, `"F#m7b5/A"`.
CHORD_RE = re.compile(
    _PITCH_NAME + "(?:" + "|".join(re.escape(q) for q in _CHORD_QUALITIES) + r")(?:/" + _PITCH_NAME + r")?"
)

#: A single event token on a music line — a chord annotation or a note/rest.
#: Faithful to upstream's `TOKEN`, which is what makes chord detection reliable:
#: it distinguishes a quoted chord from a quoted *header* value.
TOKEN_RE = re.compile(
    r'"(?P<chord>[^"\n]*)"|'
    r"\[K:(?P<key>[^\]\n]+)\]|"
    r"(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-gz])"
    r"(?P<oct>[,']*)(?P<duration>[0-9]*)(?P<tie>-?)"
)

#: Standard major and minor keys, mapped to their accidental count. A `K:` field
#: outside this set is something YuE2 was not trained to read.
_MAJOR_KEYS = ("Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#")
_MINOR_KEYS = ("Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m")
_KEYS = frozenset(_MAJOR_KEYS) | frozenset(_MINOR_KEYS)

_FIELD_RE = re.compile(r"^([A-Za-z]):")
_METER_RE = re.compile(r"^(\d+)/(\d+)$")
_UNIT_RE = re.compile(r"^1/(\d+)$")
_TEMPO_RE = re.compile(r"^1/4=(\d+)$")


class AbcError(ValueError):
    """An ABC score is malformed, or unsuitable for the path that supplied it.

    The message names the offending construct, because the caller is usually a
    person editing a score by hand and needs to know which line to fix.
    """


@dataclass
class AbcReport:
    """What a structural read of a score found."""

    voices: list[str] = field(default_factory=list)
    #: Quoted tokens that parse as chord symbols.
    chords: list[str] = field(default_factory=list)
    #: Quoted tokens that do **not** parse as chord symbols.
    #:
    #: In YuE2's native format a quoted token on a music line *is* chord
    #: notation — there is no separate "text annotation" case, which is why
    #: upstream treats an unparseable one as an error rather than ignoring it.
    #: An earlier version of this module invented that exemption and silently
    #: ignored them, so `"Cmaj9"` — a real chord outside the recognised quality
    #: list — passed validation *and* survived chord stripping, reaching YuE2
    #: with `cot="melody"` and producing something that was not a cover.
    unrecognised_quoted: list[str] = field(default_factory=list)
    key: str | None = None
    meter: str | None = None
    unit: str | None = None
    tempo: int | None = None
    music_lines: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def has_chords(self) -> bool:
        return bool(self.chords)

    @property
    def is_two_voice(self) -> bool:
        return set(self.voices) == set(VOICES)

    def to_dict(self) -> dict[str, object]:
        return {
            "voices": self.voices,
            "chords": self.chords,
            "unrecognised_quoted": self.unrecognised_quoted,
            "key": self.key,
            "meter": self.meter,
            "unit": self.unit,
            "tempo": self.tempo,
            "music_lines": self.music_lines,
            "warnings": self.warnings,
        }


def _is_music_line(line: str) -> bool:
    """True for a line carrying notes, as opposed to a header or a comment.

    This distinction is the whole reason chord stripping is safe: the header
    lines legitimately contain quoted text (`name="Vocal Melody"`), and a naive
    `"..."` removal would corrupt them.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("%"):
        return False
    return _FIELD_RE.match(stripped) is None


def _iter_music_lines(text: str):
    """Yield `(index, line)` for every music line, skipping the header block.

    Lines are yielded **with their trailing newline**, and the index matches
    `text.splitlines(keepends=True)`. Both details matter: `strip_chords` writes
    back into a keepends list by index, so yielding a newline-stripped line would
    drop the line ending on every rewrite and merge the next line into it. That
    bug shipped once and was caught by the note-invariant check in
    `strip_chords` — which is precisely why that check exists.
    """
    seen_key = False
    for index, line in enumerate(text.splitlines(keepends=True)):
        stripped = line.strip()
        if stripped.startswith("K:") and not seen_key:
            seen_key = True
            continue
        if not seen_key:
            continue
        if _is_music_line(line):
            yield index, line


def inspect(text: str) -> AbcReport:
    """Read a score's structure without judging it. Never raises.

    Used for reporting and for deciding what needs repairing; `validate_native`
    is the gate that raises.
    """
    report = AbcReport()
    if not isinstance(text, str) or not text.strip():
        report.warnings.append("score is empty")
        return report

    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("K:") and report.key is None:
            report.key = stripped[2:].strip()
        elif stripped.startswith("M:") and report.meter is None:
            report.meter = stripped[2:].strip()
        elif stripped.startswith("L:") and report.unit is None:
            report.unit = stripped[2:].strip()
        elif stripped.startswith("Q:"):
            match = _TEMPO_RE.match(stripped[2:].strip())
            if match:
                report.tempo = int(match.group(1))
            elif report.tempo is None:
                report.warnings.append(f"tempo field not in Q:1/4=<BPM> form: {stripped!r}")
        elif stripped.startswith("V:"):
            # `V: Vocal clef=treble name="Vocal Melody"` → take the name token.
            # A score repeats its `V:` header before every section, so the list is
            # deduplicated in first-seen order — the *set* of voices is what
            # matters, not how many times each was declared.
            head = stripped[2:].strip().split()[0] if stripped[2:].strip() else ""
            if head and head not in report.voices:
                report.voices.append(head)

    for _, line in _iter_music_lines(text):
        report.music_lines += 1
        for match in TOKEN_RE.finditer(line):
            chord = match.group("chord")
            if chord is None:
                continue
            # Every quoted token on a music line is chord notation. Those that do
            # not match the grammar are recorded rather than ignored — see
            # `unrecognised_quoted`.
            if CHORD_RE.fullmatch(chord.strip()):
                report.chords.append(chord)
            elif chord.strip():
                report.unrecognised_quoted.append(chord)

    return report


def find_chords(text: str) -> list[str]:
    """Chord symbols written on music lines, in order. Empty for a clean melody."""
    return inspect(text).chords


def validate_native(text: str, *, source: str = "score", require_melody_only: bool = False) -> AbcReport:
    """Check that a score is YuE2-native, raising `AbcError` with a precise reason.

    `source` names where the score came from, so the error reads as
    "edited score: ..." rather than a bare assertion.

    `require_melody_only=True` additionally rejects chord symbols. The cover path
    sets it: YuE2's `cot="melody"` does **not** strip chords itself, so a melody
    that still carries harmony would silently produce something other than a
    cover. That check is upstream's, and it is load-bearing.
    """
    if not isinstance(text, str) or not text.strip():
        raise AbcError(f"{source}: score is empty")

    report = inspect(text)
    lines = text.splitlines()

    if len(lines) < 8:
        raise AbcError(f"{source}: incomplete score — expected at least 8 lines, found {len(lines)}")

    if not lines[0].strip().startswith("X:"):
        raise AbcError(f"{source}: first line must be an X: index, found {lines[0].strip()!r}")

    if report.meter is None:
        raise AbcError(f"{source}: missing M: meter field")
    meter = _METER_RE.match(report.meter)
    if meter is None:
        raise AbcError(f"{source}: unsupported meter {report.meter!r}; write an explicit fraction such as 4/4")
    denominator = int(meter.group(2))
    if denominator > 1024 or denominator & (denominator - 1):
        raise AbcError(f"{source}: meter denominator {denominator} must be a power of two")

    if report.unit is None:
        raise AbcError(f"{source}: missing L: unit-note-length field")
    unit = _UNIT_RE.match(report.unit)
    if unit is None:
        raise AbcError(f"{source}: expected L:1/<power of two>, found {report.unit!r}")
    # The regex alone accepts any `1/N`; the denominator must be a power of two
    # for the duration arithmetic to be exact. Upstream enforces this, and
    # without it a score written in `L:1/24` would pass here and misbehave
    # downstream — a validator that is laxer than the consumer is worse than none.
    unit_denominator = int(unit.group(1))
    if unit_denominator > 1024 or unit_denominator & (unit_denominator - 1):
        raise AbcError(f"{source}: unit denominator {unit_denominator} must be a power of two")

    if report.key is None:
        raise AbcError(f"{source}: missing K: key field")
    if report.key not in _KEYS:
        raise AbcError(f"{source}: unsupported key {report.key!r}; use a standard major or minor key")

    if not report.is_two_voice:
        raise AbcError(f"{source}: expected exactly the native voices {list(VOICES)}, found {report.voices or 'none'}")

    if report.music_lines == 0:
        raise AbcError(f"{source}: no music lines found after the header block")

    if report.tempo is None:
        raise AbcError(f"{source}: missing or malformed tempo; expected Q:1/4=<integer BPM>")

    # Fail closed on any quoted token we do not recognise as a chord.
    #
    # This is upstream's policy and it is the right one: in this format a quoted
    # token on a music line *is* chord notation, so one we cannot parse is either
    # a chord we cannot handle or something the format does not allow. Ignoring
    # it lets harmony through a path that assumes none — which is exactly what
    # `"Cmaj9"` did before this check existed.
    if report.unrecognised_quoted:
        preview = ", ".join(repr(c) for c in report.unrecognised_quoted[:6])
        raise AbcError(
            f"{source}: unrecognised chord symbol(s) {preview}. Every quoted token on a music line is "
            "a chord symbol; one that does not parse cannot be validated, stripped, or safely passed on."
        )

    if require_melody_only and report.chords:
        preview = ", ".join(repr(c) for c in report.chords[:6])
        raise AbcError(
            f"{source}: contains chord symbols ({preview}). A cover melody must be melody-only — "
            'YuE2\'s cot="melody" does not strip chords, so supplying them changes the result.'
        )
    return report


def _note_signature(text: str) -> list[tuple[str, str, str, str]]:
    """The sequence of note events across all music lines, chords excluded.

    Used as the invariant for chord stripping: removing harmony must not disturb
    a single note or its timing.
    """
    signature: list[tuple[str, str, str, str]] = []
    for _, line in _iter_music_lines(text):
        for match in TOKEN_RE.finditer(line):
            if match.group("chord") is not None:
                continue
            signature.append(
                (
                    match.group("acc") or "",
                    match.group("note") or "",
                    match.group("oct") or "",
                    (match.group("duration") or "") + (match.group("tie") or ""),
                )
            )
    return signature


def strip_chords(text: str, *, source: str = "score") -> tuple[str, list[str]]:
    """Remove chord symbols from music lines, preserving header quotes.

    Returns `(stripped_text, removed_chords)`. Idempotent: a melody-only score
    comes back unchanged.

    The removal is verified, not assumed — the note sequence before and after
    must be identical, which is the same invariant upstream enforces. Removing
    harmony is only safe if the melody underneath is untouched, and a regex that
    over-matched would be exactly the kind of silent corruption this worker's
    other guards exist to prevent.
    """
    report = inspect(text)
    if not report.chords:
        if report.unrecognised_quoted:
            # Refuse rather than pass it through: we do not know whether this is
            # harmony, so we can neither remove it nor vouch for its absence.
            preview = ", ".join(repr(c) for c in report.unrecognised_quoted[:6])
            raise AbcError(
                f"{source}: cannot strip unrecognised chord symbol(s) {preview} — "
                "they may be harmony, and leaving them would silently change the result"
            )
        return text, []

    before = _note_signature(text)
    lines = text.splitlines(keepends=True)
    for index, line in _iter_music_lines(text):
        lines[index] = TOKEN_RE.sub(lambda m: "" if m.group("chord") is not None else m.group(0), line)
    output = "".join(lines)

    after = _note_signature(output)
    if before != after:
        raise AbcError(f"{source}: chord removal would have altered the melody — refusing to modify the score")

    remaining = find_chords(output)
    if remaining:
        raise AbcError(f"{source}: chord removal left {remaining[:6]} behind")

    return output, report.chords
