"""Identify what a downloaded file actually is, before a model spends 30 s on it.

Why this exists
---------------
A cover job's `source_audio` is a URL the caller supplies. The worker fetches it
with RunPod's SDK and hands the resulting path to SheetSage2, which decodes it
with ffmpeg. If the bytes are not audio, that failure arrives **after** the
transcription model has loaded:

    cover mode: melody transcription failed — Cannot decode audio:
      [in#0 @ 0x...] Error opening input: Invalid data found when processing input
      Error opening input file /app/jobs/<job>/downloaded_files/f1e9571d-....
      Error opening input files: Invalid data found when processing input

That message is ffmpeg's, and it is **identical for every kind of wrong input** —
an empty file, an HTML page, a JSON error body and a plain-text denial all
produce it byte for byte (verified against ffmpeg 6.1). So an operator reading
the job response cannot tell whether their URL 404'd to an HTML page, returned an
API envelope, or served a truncated file. They get a path with no extension and
a decoder complaint, ~32 s and one model load later.

Nothing about that is ffmpeg's fault, and nothing about it is fixable inside
SheetSage2 — it is remote code we do not control. The fix belongs here: look at
what arrived, and say so, while it is still cheap.

What it does and does not do
----------------------------
It reads a small head sample and reports a **kind**. It deliberately does not try
to be a decoder: a file that is positively **not** audio is rejected with a
useful message, and a file that is unrecognised is **passed through**. That
asymmetry is the point — refusing an exotic-but-valid container would break a job
that works today, while the failure it prevents is already fatal, just slower and
more confusing.

Extension is not consulted. That is deliberate and tested: ffmpeg's demuxer
probes content, and a valid WAV, MP3, M4A, FLAC or OGG decodes identically with
or without a suffix. RunPod's SDK names downloads `downloaded_files/<uuid>` with
no extension whenever the URL path has none, so an extension check here would
condemn exactly the files that work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

#: How much of the file to read for identification. Every signature below lives
#: in the first 16 bytes; the rest is for the text/HTML/JSON heuristics.
SAMPLE_BYTES = 4096

#: A file this small cannot be a recording. Used only to sharpen a message, never
#: as the reject test — an empty file is caught by the `empty` kind, and a tiny
#: but valid audio file should still be handed to the decoder that can judge it.
_TINY_BYTES = 1024


def _at(head: bytes, offset: int, magic: bytes) -> bool:
    return head[offset : offset + len(magic)] == magic


#: Container signatures, checked in order. `matcher` receives the head sample.
#: Only formats a caller could plausibly upload for a cover are listed — this is
#: an identification aid, not a registry, which is why an unknown file passes.
_AUDIO_SIGNATURES: tuple[tuple[str, Callable[[bytes], bool]], ...] = (
    ("flac", lambda h: h.startswith(b"fLaC")),
    ("wav", lambda h: h.startswith(b"RIFF") and _at(h, 8, b"WAVE")),
    ("aiff", lambda h: h.startswith(b"FORM") and h[8:12] in (b"AIFF", b"AIFC")),
    ("ogg", lambda h: h.startswith(b"OggS")),  # Vorbis, Opus, FLAC-in-Ogg
    ("mp3", lambda h: h.startswith(b"ID3") or (len(h) > 1 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0)),
    ("m4a", lambda h: _at(h, 4, b"ftyp")),  # MP4/M4A/AAC-in-MP4
    ("aac", lambda h: len(h) > 1 and h[0] == 0xFF and (h[1] & 0xF6) == 0xF0),  # ADTS
    ("webm", lambda h: h.startswith(b"\x1a\x45\xdf\xa3")),  # Matroska / WebM
    ("wma", lambda h: h.startswith(b"\x30\x26\xb2\x75")),  # ASF
    ("amr", lambda h: h.startswith(b"#!AMR")),
    ("au", lambda h: h.startswith(b".snd")),
    ("caf", lambda h: h.startswith(b"caff")),
    ("wavpack", lambda h: h.startswith(b"wvpk")),
    ("midi", lambda h: h.startswith(b"MThd")),
)

#: Kinds that are positively audio.
AUDIO_KINDS = frozenset(name for name, _ in _AUDIO_SIGNATURES)

#: Kinds that are positively *not* audio, and are worth naming precisely because
#: each points at a different mistake by the caller. `truncated` is here because a
#: cut-off download is not decodable either — but it is reported separately from
#: the others so the message can say "retry" rather than "check your URL".
NON_AUDIO_KINDS = frozenset({"empty", "truncated", "html", "xml", "json", "text"})

_TEXT_BYTES = frozenset(bytes(range(0x20, 0x7F))) | {0x09, 0x0A, 0x0D, 0x0C, 0x0B}


@dataclass(frozen=True)
class Probe:
    """What a downloaded file turned out to be."""

    path: Path
    size_bytes: int
    kind: str
    detail: str

    @property
    def is_audio(self) -> bool:
        return self.kind in AUDIO_KINDS

    @property
    def is_known_non_audio(self) -> bool:
        """True only when the bytes positively identify as something not audio.

        An `unknown` binary returns False on purpose — see the module docstring.
        """
        return self.kind in NON_AUDIO_KINDS

    def describe(self) -> str:
        size = f"{self.size_bytes} bytes" if self.size_bytes else "0 bytes (empty)"
        return f"{size}, {self.detail}"


def _truncated_as(head: bytes) -> str | None:
    """Name the container whose signature this head is a *prefix* of.

    A four-byte `RIFF` is a WAV whose download was cut off, not a text file. The
    distinction matters because the two have different fixes: a truncated file is
    a retry, a text file is a wrong URL. An earlier version fell through to the
    text heuristic here and reported `plain text, starting 'RIFF'`, which would
    send someone looking at their URL for a problem that is in their upload.
    """
    if len(head) >= 12:
        return None  # long enough to have matched a full signature already
    for name, _ in _AUDIO_SIGNATURES:
        for magic in _MAGIC_PREFIXES.get(name, ()):
            if magic.startswith(head) and head:
                return name
    return None


#: The leading bytes of each signature, for the truncation check above. Kept
#: beside the signatures rather than derived from them — the lambdas cannot be
#: introspected, and a derived value would silently stop covering a new entry.
_MAGIC_PREFIXES: dict[str, tuple[bytes, ...]] = {
    "flac": (b"fLaC",),
    "wav": (b"RIFF",),
    "aiff": (b"FORM",),
    "ogg": (b"OggS",),
    "mp3": (b"ID3",),
    "m4a": (b"\x00\x00\x00\x20ftyp",),
    "webm": (b"\x1a\x45\xdf\xa3",),
    "wma": (b"\x30\x26\xb2\x75",),
    "amr": (b"#!AMR",),
    "au": (b".snd",),
    "caf": (b"caff",),
    "wavpack": (b"wvpk",),
    "midi": (b"MThd",),
}


def _classify(head: bytes) -> tuple[str, str]:
    """Return `(kind, detail)` for a head sample."""
    if not head:
        return "empty", "the download contained no data at all"

    for name, matcher in _AUDIO_SIGNATURES:
        if matcher(head):
            return name, f"a {name.upper()} audio container"

    truncated = _truncated_as(head)
    if truncated:
        # `describe()` already prints the byte count, so this does not repeat it.
        return "truncated", (f"the start of a {truncated.upper()} file and nothing after it — the download was cut off")

    stripped = head.lstrip()
    lowered = stripped[:512].lower()

    if lowered.startswith(b"<?xml"):
        return "xml", "an XML document — API and storage errors arrive this way"
    if lowered.startswith(b"<!doctype") or lowered.startswith(b"<html") or b"<html" in lowered:
        return "html", "an HTML page — the URL returned a web page, not a file"
    if stripped[:1] in (b"{", b"["):
        return "json", "a JSON body — the URL returned an API response, not a file"

    # Mostly-printable content that matched none of the above is a text error
    # page, a redirect notice, or a plain denial.
    sample = head[:512]
    if sample and sum(byte in _TEXT_BYTES for byte in sample) / len(sample) > 0.95:
        first_line = sample.splitlines()[0][:80].decode("utf-8", "replace") if sample.splitlines() else ""
        return "text", f"plain text, starting {first_line!r}"

    return "unknown", "not a container this probe recognises"


def probe(path: Path) -> Probe:
    """Identify the file at `path` from a small head sample.

    Never raises for a missing or unreadable file — that is itself the finding,
    and it is reported as a kind so the caller has one code path.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return Probe(path, 0, "empty", f"cannot be read: {exc}")

    try:
        with path.open("rb") as handle:
            head = handle.read(SAMPLE_BYTES)
    except OSError as exc:
        return Probe(path, size, "empty", f"cannot be read: {exc}")

    kind, detail = _classify(head)

    # A container signature in a file too small to hold audio is worth saying out
    # loud: it means a download that started and was cut off, which is otherwise
    # indistinguishable from a bad URL.
    if kind in AUDIO_KINDS and size < _TINY_BYTES:
        detail += f" — but only {size} bytes arrived, so the download was truncated"

    return Probe(path, size, kind, detail)
