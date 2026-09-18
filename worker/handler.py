"""RunPod serverless handler for YuE2 create-mode song generation.

Boot order matters and is the first thing this module does:

1. `CacheConfig().apply_hf_env()` — point HuggingFace at the network volume
   *before* anything imports `huggingface_hub`, which reads `HF_HOME` at import
   time. Doing this after the import silently puts a 12 GB download on the
   container disk and dies at 20 GB.
2. `boot.ensure_models()` — idempotent cache of both HF repos onto the volume.
3. `boot.load_pipeline()` — construct the resident `YuE2Pipeline` once.

All three run at module import, so RunPod pays for them on worker start rather
than on the first job. The per-job path (`run_create_job`) is then short:
validate → generate → persist artifacts → upload → return URLs.

Per-job flow, matching the Phase 1 contract in `shared/worker-shape.md`:

    pipe(style=..., lyrics=..., cot=..., seed=..., cfg_scale=..., abc=...)
      -> song.save_artifacts(dir)   # writes audio.flac, score.abc, result.json
      -> upload dir to B2
      -> {audio_url, score_abc_url, duration, seed, cot, decoder, timings}

Running standalone (outside RunPod) uses the bar's own convention: pass
`--test_input='{"input": {...}}'` and the job runs once against local disk.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# --- 1. HuggingFace cache env, before any HF import --------------------------
from config import CacheConfig, ConfigError, WorkerConfig

_CONFIG_CACHE = CacheConfig()
_CONFIG_CACHE.apply_hf_env()

import runpod  # noqa: E402 - must follow apply_hf_env()

import boot  # noqa: E402 - must follow apply_hf_env()
from schema import MissingInputError, SongParameters, ValidationError, validate_job, validate_mode  # noqa: E402
from storage import B2Storage, StorageError, job_prefix  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("yue2.handler")

#: The resident pipeline. Set exactly once, by `boot_worker()` at import.
#: Intentionally not module-constant-style uppercase: it is mutable state, and
#: naming it like a constant invites a reader to assume it never changes.
_pipeline: Any | None = None

#: Why boot failed, if it did. Doubles as the circuit breaker: once boot has
#: failed, every job returns this immediately instead of re-attempting a ~12 GB
#: download. Without it, a volume that cannot be hydrated turns every job into a
#: fresh cold start, each one burning its whole timeout before failing.
_boot_error: BaseException | None = None

#: Default job timeout; the pipeline's own generation is bounded by context.
DEFAULT_JOB_TIMEOUT_SECONDS = 1800


class WorkerError(RuntimeError):
    """Anything the caller should see as a job failure rather than a crash."""


class BootFailedError(WorkerError):
    """The worker never reached a usable state; no job can run."""


def boot_worker(config: WorkerConfig | None = None) -> None:
    """Cache weights and construct the pipeline. Runs once, at module import.

    This is deliberately *not* lazy. The contract calls for the pipeline to be
    resident before the first job, and the difference is not cosmetic: a lazy
    first call would put a ~12 GB download and a model construction inside a
    job's own timeout budget, so the first job on a cold volume would be killed
    for taking longer than a generation is allowed to take.

    A failure here is recorded rather than raised. RunPod still needs a module
    that imports, so the handler can answer with a structured error naming the
    cause, instead of the worker crash-looping on an opaque traceback.
    """
    global _pipeline, _boot_error
    try:
        _pipeline = boot.load_pipeline(config, cache=_CONFIG_CACHE)
        _boot_error = None
        log.info("boot complete; pipeline resident")
    except BaseException as exc:
        _boot_error = exc
        log.exception("boot failed; this worker cannot serve jobs until restarted")


def get_pipeline(config: WorkerConfig | None = None) -> Any:
    """Return the resident pipeline.

    Never triggers a load — `boot_worker()` owns that. A second load attempt
    here is exactly the re-download-per-job failure the circuit breaker exists
    to prevent.
    """
    if _pipeline is None:
        if _boot_error is not None:
            raise BootFailedError(f"Worker failed to boot: {_boot_error}") from _boot_error
        raise BootFailedError("Worker is not booted yet")
    return _pipeline


def close_pipeline() -> None:
    """Release the pipeline and its VRAM. Used on shutdown and by tests."""
    global _pipeline
    if _pipeline is not None:
        try:
            # Documented teardown from the YuE2 README's low-level API section.
            _pipeline.close()
        except Exception:  # pragma: no cover - teardown must never raise
            log.warning("pipeline.close() raised during shutdown", exc_info=True)
        _pipeline = None


# --- generation --------------------------------------------------------------


def generate(params: SongParameters, workdir: Path, config: WorkerConfig) -> dict[str, Any]:
    """Run one generation and persist its artifacts into `workdir`.

    Returns the metadata RunPod will carry back: everything needed to build the
    response *before* any upload happens, so a storage failure cannot lose the
    record of what was generated.
    """
    pipe = get_pipeline(config)
    workdir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    # `abc=None` is omitted rather than passed: SongRequest validates that an
    # external ABC is nonempty, and `abc=None` with cot="full" is the create path.
    call_kwargs: dict[str, Any] = {
        "style": params.style,
        "lyrics": params.lyrics,
        "cot": params.cot,
        "seed": params.seed,
    }
    if params.abc is not None:
        call_kwargs["abc"] = params.abc
    if params.cfg_scale is not None:
        call_kwargs["cfg_scale"] = params.cfg_scale

    log.info(
        "generating: cot=%s seed=%s style=%d chars lyrics=%d chars abc=%s",
        params.cot,
        params.seed,
        len(params.style),
        len(params.lyrics),
        params.abc is not None,
    )
    try:
        song = pipe(**call_kwargs)
    except Exception as exc:
        raise WorkerError(f"Generation failed: {exc}") from exc

    # `save_artifacts` is the reproducible path: it writes audio.flac, score.abc
    # (via the plan), request.json, config.json and result.json into one dir.
    song.save_artifacts(str(workdir))
    elapsed = time.perf_counter() - started

    result = _read_result_json(workdir)
    audio_seconds = float(result.get("audio_seconds", 0.0))
    log.info(
        "generated %.1fs of audio in %.1fs (real-time factor %.2fx)",
        audio_seconds,
        elapsed,
        (audio_seconds / elapsed) if elapsed else 0.0,
    )

    return {
        "elapsed_seconds": round(elapsed, 2),
        "audio_seconds": round(audio_seconds, 3),
        "sample_rate": result.get("sample_rate"),
        "truncation": _normalise_truncation(result.get("truncated")),
        "timing": result.get("timing", {}),
    }


def _normalise_truncation(raw: Any) -> dict[str, Any]:
    """Normalise `result.json`'s `truncated` field.

    `SongResult.truncated` is a **property returning a dict** —
    `{"abc": <bool>, "semantic": <bool>}` (pipeline.py:90-91) — which
    `save_artifacts` writes into `result.json` verbatim (pipeline.py:113).

    This is a real trap: `bool({"abc": False, "semantic": False})` is `True`,
    because a non-empty dict is truthy. A worker that coerces this with `bool()`
    reports *every successfully generated song as truncated*. That bug shipped
    here once; the test double wrote a scalar, so the suite could not see it.

    So: preserve the per-stage detail rather than flattening it, and derive the
    summary flag from the values, not from the container's truthiness.
    """
    if isinstance(raw, dict):
        per_stage = {str(k): bool(v) for k, v in raw.items()}
        return {"truncated": any(per_stage.values()), "by_stage": per_stage}
    if raw is None:
        # Older or unexpected result.json — say so rather than assume the happy
        # case. `None` means "not reported", which is not "not truncated".
        return {"truncated": None, "by_stage": {}}
    return {"truncated": bool(raw), "by_stage": {}}


def _read_result_json(workdir: Path) -> dict[str, Any]:
    """Read the `result.json` that `save_artifacts` writes.

    Tolerated if absent: the pipeline is the source of truth for whether a song
    was produced, and we would rather return a song with thinner metadata than
    fail a job that actually succeeded.
    """
    path = workdir / "result.json"
    if not path.is_file():
        log.warning("save_artifacts wrote no result.json in %s", workdir)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("result.json in %s was unreadable", workdir, exc_info=True)
        return {}


# --- response assembly -------------------------------------------------------


def _decoder_name(config: WorkerConfig | None = None) -> str:
    """Which VAE decoder produced this audio — recorded per locked decision 6.

    Reports the repo the pipeline was actually constructed with, not an
    environment variable that merely claims to have influenced it. An earlier
    version of this read `YUE2_VAE`, which **nothing in the package reads** — a
    grep over the whole `yue2_infer` wheel returns zero hits for that name. It
    was a provenance field that could not be wrong in a way anyone would notice
    and could not be right in a way that mattered.

    `boot.load_pipeline` passes `config.vae_repo` as `vae=`, so this is the one
    place the real choice is visible.
    """
    if config is not None:
        return config.vae_repo
    return WorkerConfig.autoload().vae_repo


def build_response(
    params: SongParameters,
    artifacts: dict[str, Any],
    generation: dict[str, Any],
    config: WorkerConfig,
) -> dict[str, Any]:
    """Assemble the documented response shape.

    Shape is fixed by the Phase 1 contract: URLs plus metadata, never bytes.
    """
    audio = artifacts.get("audio.flac") or artifacts.get("audio.wav")
    score = artifacts.get("score.abc")
    if audio is None:
        raise WorkerError("Generation produced no audio artifact to return")

    truncation = generation.get("truncation") or {"truncated": None, "by_stage": {}}

    return {
        "audio_url": audio.url,
        "score_abc_url": score.url if score else None,
        "duration": generation.get("audio_seconds", 0.0),
        "seed": params.seed,
        "cot": params.cot,
        "decoder": config.vae_repo,
        "timings": generation.get("timing", {}),
        # Additive metadata beyond the contract's required keys — callers that
        # only read the documented shape are unaffected.
        "sample_rate": generation.get("sample_rate"),
        # A real boolean summary plus the per-stage detail. `truncated` here is
        # derived from the stage flags, never from the truthiness of the dict
        # the pipeline writes — see `_normalise_truncation`.
        "truncated": truncation["truncated"],
        "truncation_by_stage": truncation["by_stage"],
        "elapsed_seconds": generation.get("elapsed_seconds"),
        "artifact_urls": {name: art.url for name, art in sorted(artifacts.items())},
        "request": params.to_request_json(),
    }


# --- job entrypoint ----------------------------------------------------------


def run_create_job(
    job: dict[str, Any], *, storage: Any | None = None, config: WorkerConfig | None = None
) -> dict[str, Any]:
    """Handle one create-mode job. Returns the job result dict.

    `storage` is injectable so tests can exercise the whole path — validation,
    artifact layout, response shape — with a fake in place of B2.
    """
    job_id = str(job.get("id", "local"))
    raw_input = job.get("input")

    try:
        # Rejects `cover`/`edit` by name. Only `create` reaches the code below,
        # so the result needs no further dispatch — the call is for its
        # validation, and the assignment documents that.
        _mode = validate_mode(raw_input)
        params = validate_job(job)
    except MissingInputError as exc:
        log.warning("job %s rejected: %s", job_id, exc)
        return {"error": str(exc)}
    except ValidationError as exc:
        log.warning("job %s rejected: %s", job_id, exc)
        return {"error": str(exc)}

    # `autoload` (not a bare constructor) so storage credentials actually reach
    # the config from the environment — a bare `WorkerConfig(cache=...)` leaves
    # storage=None and every real job fails at upload.
    config = config or WorkerConfig.autoload()

    # Resolve storage *before* generating. A missing bucket credential is
    # knowable in microseconds, and discovering it after a ~70-second generation
    # would burn GPU time to produce a song we then cannot deliver. This is the
    # same fail-fast rule schema.py applies to bad input.
    try:
        bucket = storage or B2Storage(config.require_storage())
    except ConfigError as exc:
        log.warning("job %s rejected before generation: %s", job_id, exc)
        return {"error": str(exc)}

    # A fresh directory per *invocation*, not per request id. `params.id` is
    # caller-supplied, so two jobs sharing an id would otherwise `rmtree` each
    # other's artifacts mid-flight and upload to the same object prefix —
    # silent corruption of both results rather than a clean failure. The
    # uniqueness comes from this process, which the caller cannot influence.
    workdir = _CONFIG_CACHE.models_dir / f"{params.id}-{uuid.uuid4().hex[:12]}"

    try:
        _reset_dir(workdir)
        generation = generate(params, workdir, config)
        artifacts = bucket.upload_directory(workdir, job_prefix(job_id, params.id))
        # Built inside the `try`, and *before* cleanup: assembling the response
        # can fail, and a local-storage run returns `file://` URLs that would
        # point at files the `finally` had already deleted.
        response = build_response(params, artifacts, generation, config)
    # `boot.BootError` and `BootFailedError` are included deliberately: a failed
    # model cache or a missing wheel is a job failure the caller should see as
    # `{"error": ...}`, not a traceback out of the handler. Omitting `BootError`
    # once meant a cold-start failure escaped as an unhandled exception.
    except (WorkerError, StorageError, ConfigError, boot.BootError) as exc:
        log.exception("job %s failed", job_id)
        return {"error": str(exc)}
    finally:
        _cleanup(workdir)

    log.info("job %s complete: %s", job_id, response["audio_url"])
    return response


def _reset_dir(path: Path) -> None:
    """Start each job from an empty directory so artifacts cannot mix.

    Raises `WorkerError` rather than letting an `OSError` escape: an unwritable
    volume is a named failure with a cause, and this worker's contract is that
    failures are reported, not that they reach the caller as a generic
    "Unhandled worker error".
    """
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkerError(f"Could not prepare the work directory {path}: {exc}") from exc


def _cleanup(path: Path) -> None:
    """Remove the local artifact copy after upload.

    Not optional: the container disk is 20 GB and a handful of 48 kHz FLACs plus
    latent arrays will fill it. The volume keeps the durable copies.
    """
    shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _job_timeout(seconds: int) -> Iterator[None]:
    """Bound one job with SIGALRM.

    RunPod enforces its own timeout by killing the worker, which loses the
    in-flight job but leaves no diagnostics. A local alarm lets us log which
    stage was running and return a structured error first. Unix-only, which is
    all the container ever is.
    """

    def _raise(_signum: int, _frame: Any) -> None:
        raise WorkerError(f"Job exceeded its {seconds}s budget")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """RunPod entrypoint."""
    job_id = str(job.get("id", "local"))

    # Answer from the recorded boot failure rather than re-attempting a ~12 GB
    # download inside this job's timeout. See `boot_worker`.
    if _pipeline is None:
        detail = str(_boot_error) if _boot_error is not None else "worker is still booting"
        log.error("job %s refused: boot incomplete", job_id)
        return {"error": f"Worker unavailable: {detail}"}

    log.info("job %s received", job_id)
    try:
        with _job_timeout(_job_timeout_seconds()):
            return run_create_job(job)
    except WorkerError as exc:
        log.error("job %s aborted: %s", job_id, exc)
        return {"error": str(exc)}
    except Exception as exc:  # never let the worker die on one bad job
        log.exception("job %s crashed", job_id)
        return {"error": f"Unhandled worker error: {exc}"}


def _job_timeout_seconds() -> int:
    """This worker's job budget, in seconds.

    Read from `config` rather than straight from the environment so the value
    has one home and can be validated. A budget of 0 would call `signal.alarm(0)`,
    which *disarms* the alarm instead of setting a zero-second one — silently
    removing the guard this function exists to install.
    """
    configured = WorkerConfig.autoload().job_timeout_seconds
    if configured <= 0:
        log.warning("JOB_TIMEOUT_SECONDS=%s would disarm the alarm; using %s", configured, DEFAULT_JOB_TIMEOUT_SECONDS)
        return DEFAULT_JOB_TIMEOUT_SECONDS
    return configured


# --- worker startup -----------------------------------------------------------
# The canonical RunPod pattern, matching the reference workers: configure the
# runtime at module scope, then hand the handler to the SDK.
#
# `boot_worker()` runs here, before `serverless.start`, so the pipeline is
# resident before the first job and no job's timeout budget ever has to cover a
# ~12 GB weight download.
#
# It is deliberately NOT inside `main()` or behind a flag: RunPod's SDK discovers
# `--test_input` itself and drives the same handler, so the reference shape is
# both the simplest and the one the platform expects. An earlier version of this
# file wrapped startup in a custom `main()`, which worked but hid the
# `runpod.serverless.start` call where nobody looks for it.
#
# Importing this module therefore has a side effect: with no patched loader it
# will attempt a boot. Tests patch `boot.load_pipeline` before importing.
boot_worker()

runpod.serverless.start({"handler": handler})
