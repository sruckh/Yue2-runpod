#!/usr/bin/env python3
"""Transcribe audio to melody-only ABC, inside SheetSage2's own environment.

Run as a subprocess by `worker/modes.py`. This file executes in the
**sheetsage2** virtual environment, which pins torch 2.8.0 / transformers 4.45.2
/ numpy 1.24.3 — none of which can coexist with YuE2's torch 2.10.0 / numpy
2.2.6. Nothing here may import this worker's modules.

Contract
--------
Reads `<workdir>/request.json`:

    {"audio": "/path/to/source.wav",
     "melody_only": true,
     "prompts": ["timestamp", "downbeat_meter", "structure", "key", "melody_full"]}

Writes `<workdir>/result.json` and `<workdir>/score.abc`:

    {"status": "complete", "abc": "X:1\\n...", "warnings": [...]}
    {"status": "failed", "error": "..."}

Exit code is 0 on success and 2 on failure, so the parent can distinguish "the
model refused" from "the process died".

Why the interface is probed rather than assumed
-----------------------------------------------
`melody_only` is not present in every SheetSage2 revision. The YuE2 authors'
own transcriber refuses to guess — it inspects the signature and aborts with
"refresh the model code to a reviewed revision exposing melody_only
explicitly". We do the same, because the alternative is a run that returns
chords while we believe we asked for a melody, which the cover path turns into a
wrong song rather than an error.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

#: Model id and the prompt set for a melody task. `chord_full` is deliberately
#: absent — this stage must return harmony-free melody.
DEFAULT_MODEL = "m-a-p/SheetSage2"
# SheetSage2 loads `m-a-p/MERT-v2-FullSong` internally, at a pinned revision
# recorded in its own config.json. That parent is NOT named here and cannot be —
# it is fetched inside SheetSage2's `from_pretrained` — so it must be cached
# independently or offline mode makes it unreachable. `boot.CACHED_REPOS` covers
# it; this note exists so the next reader does not have to re-derive the chain.
MELODY_PROMPTS = ["timestamp", "downbeat_meter", "structure", "key", "melody_full"]


def write_result(workdir: Path, payload: dict) -> None:
    (workdir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(request: dict, workdir: Path) -> dict:
    audio = Path(request["audio"])
    if not audio.is_file():
        raise FileNotFoundError(f"audio not found: {audio}")

    model_id = request.get("model", DEFAULT_MODEL)
    prompts = request.get("prompts") or MELODY_PROMPTS
    melody_only = bool(request.get("melody_only", True))

    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = request.get("dtype") or ("bf16" if device == "cuda" else "fp32")

    # `local_files_only` when the cache is warm: the model is on the volume, and
    # reaching the network here would defeat the point of caching it.
    loader: dict = {"trust_remote_code": True, "local_files_only": bool(request.get("offline", False))}
    if request.get("base_model_path"):
        # SheetSage2 auto-loads its MERT-v2-FullSong parent from its own config.
        # Only an explicit, verified snapshot is passed through.
        loader["base_model_path"] = request["base_model_path"]

    model = AutoModel.from_pretrained(model_id, **loader).eval().to(device)

    options: dict = {}
    if melody_only:
        # Refuse rather than guess. See the module docstring.
        try:
            parameter = inspect.signature(model.transcribe).parameters.get("melody_only")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "cannot verify SheetSage2's melody_only interface; refresh the model code to a "
                "reviewed revision that exposes melody_only explicitly"
            ) from exc
        if parameter is None or parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise RuntimeError(
                f"this SheetSage2 revision ({model_id}) does not expose melody_only; "
                "refresh the model and remote code to a revision supporting melody_only=True"
            )
        options["melody_only"] = True

    result = model.transcribe(
        str(audio),
        output_dir=str(workdir),
        prompts=prompts,
        dtype=dtype,
        **options,
    )

    abc = result.get("abc")
    if result.get("abc_error") or not abc:
        raise ValueError(f"transcription produced no usable ABC: {result.get('abc_error')}")

    saved = workdir / "score.abc"
    if not saved.is_file():
        # The model is documented to write this; if it did not, the parent's
        # file-based contract is broken and it should know now.
        saved.write_text(abc, encoding="utf-8")

    return {"status": "complete", "abc": abc, "warnings": list(result.get("warnings") or [])}


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
