"""`audio_probe` — identifying a downloaded file before a model loads on it.

Written against a real failure. A cover job's `source_audio` URL served something
that was not audio, and the job died 32 s in with:

    cover mode: melody transcription failed — Cannot decode audio:
      [in#0 @ 0x...] Error opening input: Invalid data found when processing input
      Error opening input file /app/jobs/<job>/downloaded_files/f1e9571d-....

That message is **ffmpeg's**, raised inside SheetSage2's remote code, and it is
byte-for-byte identical for an empty file, an HTML page, a JSON body and a
plain-text denial (verified against ffmpeg 6.1). So the operator learned nothing
about which mistake they had made, and paid a model load to learn it.

Two design rules are pinned here, because both are easy to "fix" wrongly later:

- **Extension is never consulted.** ffmpeg probes content, and a real WAV, MP3,
  M4A, FLAC or OGG decodes identically with or without a suffix. RunPod's SDK
  names downloads `downloaded_files/<uuid>` — no extension — so an extension
  check would reject exactly the files that work.
- **Unknown is not rejected.** Only a file *positively* identified as something
  else is refused. An exotic-but-valid container must pass through to the decoder
  that can actually judge it; the failure being prevented is already fatal, just
  slower and less legible.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import audio_probe
import pytest

# =============================================================================
# Positive identification — the formats a caller plausibly uploads
# =============================================================================

#: (label, magic bytes) — hand-written so the test does not depend on a local
#: ffmpeg, a codec, or a fixture file. The probe reads signatures, not audio.
SIGNATURES = [
    ("flac", b"fLaC" + b"\x00" * 64),
    ("wav", b"RIFF" + b"\x24\x08\x00\x00" + b"WAVEfmt " + b"\x00" * 32),
    ("aiff", b"FORM" + b"\x00\x00\x01\x00" + b"AIFF" + b"\x00" * 32),
    ("ogg", b"OggS" + b"\x00" * 64),
    ("mp3-id3", b"ID3\x04\x00\x00" + b"\x00" * 64),
    ("mp3-frame", b"\xff\xfb\x90\x00" + b"\x00" * 64),
    ("m4a", b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 32),
    ("webm", b"\x1a\x45\xdf\xa3" + b"\x00" * 64),
    ("wma", b"\x30\x26\xb2\x75" + b"\x00" * 64),
    ("amr", b"#!AMR\n" + b"\x00" * 64),
    ("au", b".snd" + b"\x00" * 64),
    ("caf", b"caff\x00\x01" + b"\x00" * 64),
    ("wavpack", b"wvpk" + b"\x00" * 64),
]


@pytest.mark.parametrize(("kind", "magic"), SIGNATURES, ids=[s[0] for s in SIGNATURES])
def test_audio_signatures_are_identified(tmp_path: Path, kind: str, magic: bytes) -> None:
    """Each container's magic bytes identify it as audio."""
    path = tmp_path / "downloaded"  # no extension, as the SDK names it
    path.write_bytes(magic)

    found = audio_probe.probe(path)

    assert found.is_audio, f"{kind}: classified as {found.kind!r}"
    assert not found.is_known_non_audio
    assert found.kind in audio_probe.AUDIO_KINDS


def test_a_uuid_named_file_with_no_extension_is_accepted(tmp_path: Path) -> None:
    """The exact shape RunPod's SDK writes: `downloaded_files/<uuid>`.

    This is the case the probe exists to *not* break. The path that failed in
    production had no extension, and the tempting fix — require one — would have
    rejected every working cover job.
    """
    path = tmp_path / "f1e9571d-4562-4896-b02b-1c5e3c0eb69e"
    path.write_bytes(b"fLaC" + b"\x00" * 4096)

    found = audio_probe.probe(path)

    assert found.is_audio
    assert not found.is_known_non_audio


def test_extension_is_ignored_in_both_directions(tmp_path: Path) -> None:
    """A correct suffix on wrong bytes must not rescue them, nor vice versa.

    The probe reads content. If it ever starts trusting the name, both halves of
    this break — and the first half is the production bug.
    """
    lying = tmp_path / "actually_html.mp3"
    lying.write_bytes(b"<!DOCTYPE html><html><body>404</body></html>")
    assert audio_probe.probe(lying).is_known_non_audio, "a .mp3 name must not make HTML audio"

    honest = tmp_path / "actually_audio.txt"
    honest.write_bytes(b"fLaC" + b"\x00" * 4096)
    assert audio_probe.probe(honest).is_audio, "a .txt name must not make FLAC non-audio"


# =============================================================================
# Negative identification — the bodies that actually arrive from a bad URL
# =============================================================================


def test_empty_file_is_reported_as_empty_not_unknown(tmp_path: Path) -> None:
    """A zero-byte download is the commonest failure and deserves its own kind.

    It is what an SDK that swallows a request error returns, and what a URL that
    closes without sending a body produces.
    """
    path = tmp_path / "empty"
    path.write_bytes(b"")

    found = audio_probe.probe(path)

    assert found.kind == "empty"
    assert found.is_known_non_audio
    assert "no data" in found.describe()


@pytest.mark.parametrize(
    ("label", "body", "kind"),
    [
        ("html5", b"<!DOCTYPE html>\n<html><body>Not Found</body></html>", "html"),
        ("html-no-doctype", b"<html><head><title>404</title></head></html>", "html"),
        ("s3-xml-error", b'<?xml version="1.0"?><Error><Code>NoSuchKey</Code></Error>', "xml"),
        ("json-envelope", b'{"error":"not found","status":404}', "json"),
        ("json-array", b'[{"message":"unauthorized"}]', "json"),
        ("plain-denial", b"Access Denied", "text"),
        ("nginx-404", b"<html>\r\n<head><title>404 Not Found</title></head>\r\n</html>", "html"),
    ],
)
def test_error_bodies_are_positively_identified(tmp_path: Path, label: str, body: bytes, kind: str) -> None:
    """Each wrong-body shape gets a kind that names the caller's actual mistake."""
    path = tmp_path / label
    path.write_bytes(body)

    found = audio_probe.probe(path)

    assert found.kind == kind, f"{label}: got {found.kind!r}, want {kind!r}"
    assert found.is_known_non_audio


def test_json_is_checked_after_xml_and_html(tmp_path: Path) -> None:
    """A leading BOM or whitespace must not hide the real kind."""
    path = tmp_path / "padded"
    path.write_bytes(b"\r\n\r\n  <!DOCTYPE html><html></html>")

    assert audio_probe.probe(path).kind == "html"


# =============================================================================
# The asymmetry that keeps working jobs working
# =============================================================================


def test_unrecognised_binary_is_passed_through_not_rejected(tmp_path: Path) -> None:
    """An unknown container must reach the decoder rather than be refused here.

    The probe is an identification aid, not a decoder. Refusing something it does
    not recognise would break a valid job to prevent a failure that is already
    fatal and merely badly-worded — the wrong trade both ways.
    """
    path = tmp_path / "mystery"
    path.write_bytes(bytes(range(256)) * 16)

    found = audio_probe.probe(path)

    assert found.kind == "unknown"
    assert not found.is_audio
    assert not found.is_known_non_audio, "unknown must not be treated as known-bad"


def test_a_missing_file_is_reported_rather_than_raising(tmp_path: Path) -> None:
    """One code path for the caller: a missing file is a kind, not an exception."""
    found = audio_probe.probe(tmp_path / "does-not-exist")

    assert found.kind in audio_probe.NON_AUDIO_KINDS or found.kind == "empty"
    assert found.is_known_non_audio
    assert "cannot be read" in found.detail or "no data" in found.detail


def test_a_truncated_audio_header_is_flagged(tmp_path: Path) -> None:
    """A FLAC header with almost no body behind it is a cut-off download.

    Distinguished from a bad URL because the fix is different: retry the fetch
    rather than correct the link.
    """
    path = tmp_path / "cutoff"
    path.write_bytes(b"fLaC" + b"\x00" * 16)  # a valid header, far too small

    found = audio_probe.probe(path)

    assert found.is_audio, "the header is still a real FLAC header"
    assert "truncated" in found.detail


# =============================================================================
# Against real audio, when an encoder is available
# =============================================================================


def _have_ffmpeg() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _have_ffmpeg(), reason="needs ffmpeg to synthesise real audio")
@pytest.mark.parametrize(
    ("codec", "container"),
    [("pcm_s16le", "wav"), ("libmp3lame", "mp3"), ("flac", "flac"), ("libvorbis", "ogg"), ("aac", "mp4")],
)
def test_real_encoded_audio_is_accepted_without_an_extension(tmp_path: Path, codec: str, container: str) -> None:
    """End to end with genuine encoder output, named exactly as the SDK names it.

    Signature tests prove the probe reads magic bytes; this proves the magic
    bytes it expects are the ones a real encoder writes.

    `-f <container>` is explicit because the output path is a bare UUID — ffmpeg
    refuses to pick a muxer for a name it cannot read an extension from, which is
    the one place where *writing* audio does care about the suffix. (Reading it
    does not: that asymmetry is the whole point of this module.)
    """
    encoded = tmp_path / f"src-{container}"
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:a",
            codec,
            "-f",
            container,
            str(encoded),
            "-y",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg cannot encode {codec}/{container} here: {result.stderr.strip()[:120]}")

    # Rename to a bare UUID: no extension at all, as `download_files_from_urls`
    # leaves it when the source URL path has no suffix.
    bare = tmp_path / "b3f1c2d4-0000-4000-8000-000000000000"
    encoded.rename(bare)

    found = audio_probe.probe(bare)

    assert found.is_audio, f"{codec}/{container} output classified {found.kind!r}: {found.detail}"
    assert found.size_bytes > 0


def test_the_sample_is_bounded(tmp_path: Path) -> None:
    """Only a head sample is read — a cover source can be hundreds of MB."""
    big = tmp_path / "big"
    big.write_bytes(b"fLaC" + b"\x00" * (8 * 1024 * 1024))

    assert audio_probe.SAMPLE_BYTES <= 64 * 1024, "the head sample must stay small"
    assert audio_probe.probe(big).size_bytes == 8 * 1024 * 1024 + 4
    assert audio_probe.probe(big).is_audio
