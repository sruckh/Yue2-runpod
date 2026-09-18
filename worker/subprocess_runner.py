"""Run a model family in its own virtual environment, as a subprocess.

The cover path needs two model families whose dependencies cannot coexist with
YuE2's, or with each other:

| Environment | torch | transformers | numpy |
|---|---|---|---|
| YuE2 (this process) | 2.10.0 | 4.57.6 | 2.2.6 |
| SheetSage2 | 2.8.0 | 4.45.2 | 1.24.3 |
| Qwen3-ASR | unpinned | 4.57.6 | — |

A subprocess boundary buys two things, and the *second* is the one that matters
here. The first is dependency isolation — YuE2's torch 2.10 never has to coexist
with SheetSage2's 2.8 in one interpreter. The second is **guaranteed VRAM
release**: when the subprocess exits, the driver reclaims its memory, with no
reliance on in-process `del model; torch.cuda.empty_cache()` being correct for a
`trust_remote_code` model class we do not control.

That is what keeps a cover job's peak VRAM at YuE2's own ceiling instead of the
sum of three models — see `shared/vram-budget.md`.

Protocol
--------
Files, not pipes or shared memory. The parent writes `request.json` into a fresh
work directory, runs `<venv>/bin/python <entrypoint> --request <path>`, and reads
back `result.json` plus whatever artifacts the stage wrote. Everything the child
produces is inspectable after the fact, which matters when the only other
diagnostic channel is a container log.

The virtual environments are built into the **image**, never on a dev box — see
the root `Dockerfile`. Nothing in this module creates an environment; if the
interpreter is missing, that is an image-build failure and it says so.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Where each model family's virtual environment lives in the image.
VENV_ROOT = Path(os.environ.get("YUE2_VENV_ROOT", "/opt/venvs"))

#: Environment passed to a child. Deliberately minimal: inheriting the parent's
#: `PYTHONPATH` would let the child import this worker's modules against a
#: different torch, which is the exact confusion the boundary exists to prevent.
_INHERITED_ENV = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_TOKEN", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")


class SubprocessError(RuntimeError):
    """A model-family subprocess could not be run, or failed."""


@dataclass
class SubprocessResult:
    """What a model-family subprocess produced."""

    name: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    returncode: int | None = None
    stderr_tail: str = ""

    @property
    def error(self) -> str | None:
        if self.ok:
            return None
        return str(self.payload.get("error") or self.stderr_tail or f"{self.name} failed")


def venv_python(venv_name: str) -> Path:
    """The interpreter inside a model family's virtual environment."""
    return VENV_ROOT / venv_name / "bin" / "python"


def run_stage(
    venv_name: str,
    entrypoint: Path,
    workdir: Path,
    *,
    timeout_seconds: int,
    name: str | None = None,
) -> SubprocessResult:
    """Run one isolated stage to completion.

    `workdir` is the contract between parent and child: the parent has already
    written `request.json` into it, and the child writes `result.json` and its
    artifacts there. The parent owns creating and cleaning the directory.

    A missing interpreter or entrypoint is raised rather than returned as a
    failed result — both mean the *image* is wrong, not that the job's input was
    bad, and a caller retrying a job cannot fix either.
    """
    name = name or venv_name
    python = venv_python(venv_name)
    if not python.is_file():
        raise SubprocessError(
            f"{name}: interpreter not found at {python}. The virtual environment is built into the "
            "image; a missing one means the image build did not create it."
        )
    entrypoint = Path(entrypoint)
    if not entrypoint.is_file():
        raise SubprocessError(f"{name}: entrypoint not found at {entrypoint}")

    request_path = workdir / "request.json"
    if not request_path.is_file():
        raise SubprocessError(f"{name}: no request.json in {workdir}")

    env = {k: os.environ[k] for k in _INHERITED_ENV if k in os.environ}
    env["PYTHONPATH"] = ""  # never let the child see this worker's modules
    env["PYTHONUNBUFFERED"] = "1"

    log.info("%s: running %s in its own environment", name, entrypoint.name)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(python), str(entrypoint), "--request", str(request_path)],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        log.error("%s: timed out after %.0fs", name, elapsed)
        return SubprocessResult(
            name=name,
            ok=False,
            payload={"error": f"{name} exceeded its {timeout_seconds}s budget"},
            elapsed_seconds=elapsed,
            stderr_tail=_tail(exc.stderr),
        )

    elapsed = time.perf_counter() - started
    result = _read_result(workdir)
    ok = completed.returncode == 0 and bool(result.get("status") == "complete")

    if not ok and not result.get("error"):
        # A child that died before writing a result leaves only its exit code and
        # stderr. Reporting the bare code ("exited 1") tells an operator nothing;
        # the stderr tail is usually the whole diagnosis, so it belongs in the
        # message rather than only in the log.
        detail = _tail(completed.stderr)
        result["error"] = f"{name} exited {completed.returncode}" + (f": {detail}" if detail else "")

    artifacts = {
        str(p.relative_to(workdir)): p for p in sorted(workdir.rglob("*")) if p.is_file() and p.name != "request.json"
    }
    log.info("%s: %s in %.1fs (%d artifacts)", name, "ok" if ok else "failed", elapsed, len(artifacts))

    return SubprocessResult(
        name=name,
        ok=ok,
        payload=result,
        artifacts=artifacts,
        elapsed_seconds=elapsed,
        returncode=completed.returncode,
        stderr_tail=_tail(completed.stderr),
    )


def write_request(workdir: Path, request: dict[str, Any]) -> Path:
    """Write the child's input. Kept here so both sides agree on the name."""
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "request.json"
    path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return path


def _read_result(workdir: Path) -> dict[str, Any]:
    """Read `result.json`, tolerating its absence.

    A child that crashed before writing one still produced a return code and
    stderr, and those are more useful than a bare "no result" — so this returns
    an empty dict and lets the caller combine the evidence.
    """
    path = workdir / "result.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {"result": loaded}
    except (OSError, ValueError):
        log.warning("unreadable result.json in %s", workdir, exc_info=True)
        return {}


def _tail(text: str | bytes | None, limit: int = 2000) -> str:
    """The last part of a child's stderr — enough to diagnose, short enough to log."""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def scratch_dir(root: Path, label: str) -> Path:
    """A fresh directory for one stage. Never reused, so artifacts cannot mix."""
    path = root / label
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
