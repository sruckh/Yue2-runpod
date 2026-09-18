#!/usr/bin/env python3
"""Transcribe lyrics from audio, inside Qwen3-ASR's own environment.

Run as a subprocess by `worker/modes.py`. This file executes in the **qwen3-asr**
virtual environment. Nothing here may import this worker's modules.

Contract
--------
Reads `<workdir>/request.json`:

    {"audio": "/path/to/source.wav", "language": null}

Writes `<workdir>/result.json` and `<workdir>/lyrics.txt`:

    {"status": "complete", "text": "...", "language": "English"}
    {"status": "failed", "error": "..."}

Why this is a separate stage from the melody transcription
----------------------------------------------------------
The YuE2 authors' reference is explicit that "source-separation, transcription,
lyric recognition and score-conditioned generation are distinct operations".
Running them as two subprocesses is also what keeps the job's VRAM peak at
YuE2's ceiling: each process exits and the driver reclaims its memory before the
next loads.

No vocal-isolation preprocessing
--------------------------------
Qwen3-ASR is documented to handle singing voice and full songs with background
music directly, so there is deliberately no source-separation step here — one
was considered and is not needed at this scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"


def write_result(workdir: Path, payload: dict) -> None:
    (workdir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(request: dict, workdir: Path) -> dict:
    audio = Path(request["audio"])
    if not audio.is_file():
        raise FileNotFoundError(f"audio not found: {audio}")

    model_id = request.get("model", DEFAULT_MODEL)
    language = request.get("language")  # None means auto-detect

    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=int(request.get("max_inference_batch_size", 32)),
        max_new_tokens=int(request.get("max_new_tokens", 256)),
    )

    results = model.transcribe(audio=str(audio), language=language)
    if not results:
        raise ValueError("transcription returned no results")

    first = results[0]
    # The result object exposes `.text`; tolerate a mapping return so a library
    # version bump does not silently produce an empty lyric.
    text = (first.text if hasattr(first, "text") else first.get("text")) or ""
    detected = getattr(first, "language", None) or (first.get("language") if isinstance(first, dict) else None)

    text = str(text).strip()
    if not text:
        raise ValueError("transcription returned empty text")

    (workdir / "lyrics.txt").write_text(text + "\n", encoding="utf-8")
    return {"status": "complete", "text": text, "language": detected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()

    workdir = args.request.parent
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        write_result(workdir, run(request, workdir))
        return 0
    except Exception as exc:
        write_result(workdir, {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
