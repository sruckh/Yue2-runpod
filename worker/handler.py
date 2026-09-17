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
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# --- 1. HuggingFace cache env, before any HF import --------------------------
from config import CacheConfig, ConfigError, WorkerConfig

_CONFIG_CACHE = CacheConfig()
_CONFIG_CACHE.apply_hf_env()

import boot  # noqa: E402 - must follow apply_hf_env()
from schema import MissingInputError, SongParameters, ValidationError, validate_job, validate_mode  # noqa: E402
from storage import B2Storage, StorageError, job_prefix  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("yue2.handler")

#: The pipeline is loaded once, lazily, and reused for every job. `None` means
#: "not loaded yet" — `get_pipeline()` is the only reader. Intentionally not
#: module-constant-style uppercase: it is mutable state, and naming it like a
#: constant invites a reader to assume it never changes.
_pipeline: Any | None = None

#: Default job timeout; the pipeline's own generation is bounded by context.
JOB_TIMEOUT_SECONDS = 1800


class WorkerError(RuntimeError):
    """Anything the caller should see as a job failure rather than a crash."""


def get_pipeline(config: WorkerConfig | None = None) -> Any:
    """Return the resident pipeline, loading it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = boot.load_pipeline(config, cache=_CONFIG_CACHE)
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
        "truncated": bool(result.get("truncated", False)),
        "timing": result.get("timing", {}),
    }


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

    `YuE2-Vae-legacy` is env-switchable for benchmark reproduction only, so the
    response names whichever is actually in use.
    """
    return os.environ.get("YUE2_VAE", "m-a-p/YuE2-Vae")


def build_response(
    params: SongParameters,
    artifacts: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the documented response shape.

    Shape is fixed by the Phase 1 contract: URLs plus metadata, never bytes.
    """
    audio = artifacts.get("audio.flac") or artifacts.get("audio.wav")
    score = artifacts.get("score.abc")
    if audio is None:
        raise WorkerError("Generation produced no audio artifact to return")

    return {
        "audio_url": audio.url,
        "score_abc_url": score.url if score else None,
        "duration": generation.get("audio_seconds", 0.0),
        "seed": params.seed,
        "cot": params.cot,
        "decoder": _decoder_name(),
        "timings": generation.get("timing", {}),
        # Additive metadata beyond the contract's required keys — callers that
        # only read the documented shape are unaffected.
        "sample_rate": generation.get("sample_rate"),
        "truncated": generation.get("truncated", False),
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
        mode = validate_mode(raw_input)
        params = validate_job(job)
    except MissingInputError as exc:
        log.warning("job %s rejected: %s", job_id, exc)
        return {"error": str(exc)}
    except ValidationError as exc:
        log.warning("job %s rejected: %s", job_id, exc)
        return {"error": str(exc)}

    assert mode == "create"  # validate_mode rejects the others by name

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

    workdir = _CONFIG_CACHE.models_dir / params.id
    _reset_dir(workdir)

    try:
        generation = generate(params, workdir, config)
        artifacts = bucket.upload_directory(workdir, job_prefix(job_id, params.id))
    # `boot.BootError` is included deliberately: a failed model cache or a
    # missing wheel is a job failure the caller should see as `{"error": ...}`,
    # not a traceback out of the handler. Omitting it here meant a cold-start
    # failure produced an unhandled exception instead of a structured result.
    except (WorkerError, StorageError, ConfigError, boot.BootError) as exc:
        log.exception("job %s failed", job_id)
        return {"error": str(exc)}
    finally:
        _cleanup(workdir)

    response = build_response(params, artifacts, generation)
    log.info("job %s complete: %s", job_id, response["audio_url"])
    return response


def _reset_dir(path: Path) -> None:
    """Start each job from an empty directory so artifacts cannot mix."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


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
    log.info("job %s received", job_id)
    try:
        with _job_timeout(int(os.environ.get("JOB_TIMEOUT_SECONDS", JOB_TIMEOUT_SECONDS))):
            return run_create_job(job)
    except WorkerError as exc:
        log.error("job %s aborted: %s", job_id, exc)
        return {"error": str(exc)}
    except Exception as exc:  # never let the worker die on one bad job
        log.exception("job %s crashed", job_id)
        return {"error": f"Unhandled worker error: {exc}"}


def main(argv: list[str] | None = None) -> int:
    """Standalone entry: either serve via RunPod, or run one test input."""
    argv = sys.argv[1:] if argv is None else argv
    try:
        import runpod
    except ImportError:
        runpod = None  # type: ignore[assignment]

    if runpod is None:
        # Local dry run without the SDK installed — still useful in CI.
        log.warning("runpod SDK not installed; running in test-input mode only")
        return _run_test_input(argv, storage=None)

    test_input = next((a.split("=", 1)[1] for a in argv if a.startswith("--test_input=")), None)
    if test_input:
        return _run_test_input(argv, storage=None)

    log.info("starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
    return 0


def _run_test_input(argv: list[str], storage: Any | None) -> int:
    """Execute a single job from `--test_input` and print its result.

    Storage is replaced with a local stand-in so the path is exercisable with no
    B2 credentials and no GPU-side upload — this is the hook the CI workflow in
    `.github/workflows/` drives.
    """
    raw = next((a.split("=", 1)[1] for a in argv if a.startswith("--test_input=")), None)
    if raw is None:
        print("usage: handler.py --test_input='{\"input\": {...}}'", file=sys.stderr)
        return 2
    job = json.loads(raw)
    result = run_create_job(job, storage=storage or _LocalStorage())
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 1


class _LocalStorage:
    """Drop-in stand-in for `B2Storage` that keeps artifacts on local disk.

    Used only by `--test_input` runs. It returns `file://` URLs so the printed
    result has the same shape as production without pretending to be reachable.
    """

    def upload_directory(self, directory: Path, prefix: str) -> dict[str, Any]:
        from storage import UploadedArtifact

        return {
            path.relative_to(directory).as_posix(): UploadedArtifact(
                key=f"{prefix}/{path.relative_to(directory).as_posix()}",
                url=f"file://{path}",
                size_bytes=path.stat().st_size,
                content_type="application/octet-stream",
            )
            for path in sorted(p for p in Path(directory).rglob("*") if p.is_file())
        }


if __name__ == "__main__":
    sys.exit(main())
