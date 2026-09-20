"""Run a model family as its own subprocess, under a named interpreter.

Cover needs three model families in sequence. **They share one environment** —
the boundary below is a *process* boundary, and that is the point:

**Guaranteed VRAM release.** When the subprocess exits, the driver reclaims its
memory, with no reliance on in-process `del model; torch.cuda.empty_cache()`
being correct for a `trust_remote_code` model class we do not control. That is
what keeps a cover job's peak at YuE2's own ceiling instead of the sum of three
models — measured, not assumed: see `shared/vram-budget.md`.

The environments were originally separate, on the assumption the pins could not
coexist:

| Environment | torch | transformers | numpy |
|---|---|---|---|
| YuE2 | 2.10.0 | 4.57.6 | 2.2.6 |
| SheetSage2 | 2.8.0 | 4.45.2 | 1.24.3 |
| Qwen3-ASR | unpinned (resolved 2.14.0) | 4.57.6 | — |

That assumption was never tested. The unification experiment tested it — both
families load and produce correct output on YuE2's stack — and both venvs were
removed on 2026-09-19, taking ~6.5 GB of duplicated torch with them. Dependency
isolation was the *first* reason for this boundary and it turned out not to be
needed; VRAM release was the second and it is the one carrying the design.

`venv_name` survives so a future family whose pins genuinely conflict can be
given its own interpreter by naming it, without redesigning this module.

Protocol
--------
Files, not pipes or shared memory. The parent writes `request.json` into a fresh
work directory, runs `<interpreter> <entrypoint> --request <path>`, and reads
back `result.json` plus whatever artifacts the stage wrote. Everything the child
produces is inspectable after the fact, which matters when the only other
diagnostic channel is a container log.

Any environment this names is built into the **image**, never on a dev box — see
the root `Dockerfile`. Nothing in this module creates one; if the interpreter is
missing, that is an image-build failure and it says so.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Where each model family's virtual environment lives in the image.
VENV_ROOT = Path(os.environ.get("YUE2_VENV_ROOT", "/opt/venvs"))

#: Environment passed to a child. Deliberately narrow: inheriting the parent's
#: `PYTHONPATH` would let the child import this worker's modules against a
#: different torch, which is the exact confusion the boundary exists to prevent.
#:
#: `HOME` is included because the HuggingFace libraries consult `~/.cache` and
#: `~/.netrc`; without it they log errors and, in some versions, fall back to a
#: read-only location. `TRANSFORMERS_OFFLINE` and `XDG_CACHE_HOME` are included
#: for the same reason — the child should see the same cache the parent does.
_INHERITED_ENV = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_TOKEN",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "XDG_CACHE_HOME",
    "HOME",
    "CUDA_VISIBLE_DEVICES",
)


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


#: Passed as a `venv_name` to mean "the interpreter running this worker".
#:
#: An explicit sentinel rather than `""`, which would resolve to
#: `VENV_ROOT / "" / "bin" / "python"` — a path that happens to be wrong in a way
#: that only surfaces as a subprocess that cannot start.
MAIN_INTERPRETER = ""


def venv_python(venv_name: str) -> Path:
    """The interpreter inside a model family's virtual environment."""
    if venv_name == MAIN_INTERPRETER:
        return Path(sys.executable)
    return VENV_ROOT / venv_name / "bin" / "python"


def run_stage(
    venv_name: str,
    entrypoint: Path,
    workdir: Path,
    *,
    timeout_seconds: int,
    name: str | None = None,
    required: bool = True,
) -> SubprocessResult:
    """Run one isolated stage to completion.

    `required=False` marks a stage whose failure is advisory — the caller has
    another source for its output and will decide. It changes nothing about how
    the stage runs; it only records the intent in the log, so an operator reading
    a job's story can tell a tolerated failure from a fatal one.

    `workdir` is the contract between parent and child: the parent has already
    written `request.json` into it, and the child writes `result.json` and its
    artifacts there. The parent owns creating and cleaning the directory.

    A missing interpreter or entrypoint is raised rather than returned as a
    failed result — both mean the *image* is wrong, not that the job's input was
    bad, and a caller retrying a job cannot fix either.
    """
    name = name or venv_name or "main"
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

    # The log line names the interpreter, not a fixed phrase. It used to read
    # "in its own environment", which was true while each model family had its own
    # venv and became misleading once they moved into the main one — a reader
    # chasing configuration would go looking for a venv that no longer exists. The
    # path is the fact; the phrase was a description of a design that changed.
    log.info("%s: running %s under %s", name, entrypoint.name, python)
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
    if not ok and not required:
        log.warning("%s: failed after %.1fs — advisory, the caller supplied a fallback", name, elapsed)
    else:
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
    """A fresh directory for one stage, guaranteed empty.

    `ignore_errors=True` on the removal was a bug: if `rmtree` failed, the
    directory survived, `mkdir(exist_ok=True)` succeeded, and the next stage's
    artifact sweep reported the *previous* run's `score.abc` and `lyrics.txt` as
    this run's output. Silent, and exactly the kind of wrongness that looks like
    success. A directory that cannot be emptied is now a hard error.
    """
    path = root / label
    if path.exists():
        try:
            shutil.rmtree(path)
        except OSError as exc:
            raise SubprocessError(
                f"could not clear {path} before this stage: {exc}. Refusing to run in a directory that "
                "may hold a previous run's artifacts."
            ) from exc
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise SubprocessError(f"{path} is not empty after clearing; refusing to mix runs")
    return path
