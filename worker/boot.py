"""Boot: cache model weights on the network volume, then load the pipeline once.

Two responsibilities, in order:

1. `ensure_models()` — idempotently `snapshot_download` both HuggingFace repos
   (`m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae`) into the volume's HF cache. Weights are
   never baked into the image (locked decision 4), so a cold worker pays for
   this once and every subsequent worker on the same volume skips it.

2. `load_pipeline()` — construct the resident `YuE2Pipeline`. This is the
   expensive step (BF16 AR/NAR + FP32 VAE onto a 24 GB card) and it happens once
   per worker lifetime, at module import, not per job.

The `runpod.serverless.VolumeCache` context manager noted in
`shared/dependency-pins.md` as worth evaluating is deliberately *not* used: it
hydrates a cache around a load, but our contract is that the volume cache
outlives the container entirely (a warm volume is the whole point of the
network-volume decision). `snapshot_download(local_dir=...)` is the documented,
already-validated pattern and gives us an explicit idempotency check we can
assert on in tests.

Nothing here is imported by `schema.py` or `config.py`, so unit tests can
exercise validation and storage without torch installed.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import MODEL_REPO, MODEL_WHEEL, VAE_REPO, CacheConfig

log = logging.getLogger(__name__)

#: Files that must be present for the YuE2 repo to be considered fully cached.
#:
#: Aligned with the constants the pipeline itself uses in
#: `yue2.storage.MODEL_FILES` when it downloads over the Hub — this is the set
#: its loader actually reads (config, generation configs, the weight manifest,
#: the safetensors, the tokenizer, and the `trust_remote_code` model module).
#: Keeping our check identical means "we think it is cached" and "the pipeline
#: can load it" cannot drift apart.
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "yue2_generation_config.json",
    "weights_manifest.json",
    "model.safetensors",
    "qwen.tiktoken",
    "modeling_yue2.py",
)

#: The VAE repo is smaller but equally mandatory — it is the default decoder.
REQUIRED_VAE_FILES = (
    "config.json",
    "model.safetensors",
    "modeling_vae.py",
    "weights_manifest.json",
)

#: Glob patterns limiting what `snapshot_download` fetches.
#:
#: Without these, `snapshot_download` pulls the **entire** repository — which
#: for `m-a-p/YuE2-3B` includes demo MP3s under `assets/audio/`, a PDF and a
#: logo. Those are megabytes of cold-start download the worker never reads. The
#: pipeline's own Hub path passes an equivalent `allow_patterns`, so matching it
#: also guarantees we cache exactly what it will look for.
MODEL_FILE_PATTERNS = (
    *REQUIRED_MODEL_FILES,
    "model.safetensors.index.json",  # present only for sharded checkpoints
    "modeling_vae.py",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/*",
    # The `yue2_infer` wheel ships inside this repo rather than on PyPI, and the
    # image build fetches it from the cache. See `install_model_wheel`.
    "*.whl",
)

VAE_FILE_PATTERNS = (
    *REQUIRED_VAE_FILES,
    "model.safetensors.index.json",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/*",
)


class BootError(RuntimeError):
    """Weights could not be cached, or the pipeline could not be constructed."""


@dataclass
class ModelCacheReport:
    """What `ensure_models()` found and did — surfaced in the health response."""

    model_dir: Path
    vae_dir: Path
    downloaded: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def cache_hit(self) -> bool:
        """True when nothing had to be fetched — the warm-volume path."""
        return not self.downloaded


def _missing_files(directory: Path, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not (directory / name).is_file()]


def _snapshot_download(
    repo_id: str,
    target: Path,
    cache: CacheConfig,
    allow_patterns: tuple[str, ...],
) -> Path:
    """Fetch one repo, reusing anything already on disk.

    `local_files_only=True` deliberately makes a warm volume prove itself: if a
    required blob is missing the call raises instead of silently reaching for
    the network, which is exactly the failure we want caught at boot rather
    than mid-generation.

    `allow_patterns` is not an optimisation we invented — the pipeline's own Hub
    path restricts the download the same way, and without it a cold start pulls
    the repo's demo audio and artwork for nothing.
    """
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            cache_dir=str(cache.hf_home / "hub") if cache.hf_home else None,
            token=cache.hf_token,
            local_files_only=cache.local_files_only,
            allow_patterns=list(allow_patterns),
            # `local_dir` gives us a stable, inspectable tree; without this the
            # hub writes only into its content-addressed cache and `local_dir`
            # ends up holding symlinks that confuse the pipeline's own
            # `resolve_model`.
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        hint = ""
        if cache.local_files_only:
            hint = " (HF_LOCAL_FILES_ONLY=true; unset it for the first cold start)"
        raise BootError(f"Could not cache {repo_id} into {target}{hint}: {exc}") from exc
    return target


def ensure_models(cache: CacheConfig | None = None) -> ModelCacheReport:
    """Ensure both HF repos are fully cached on the volume. Idempotent.

    Returns a report rather than a bool so the caller can log *what* was
    fetched — on a cold start that is minutes of wall time worth explaining.
    """
    cache = cache or CacheConfig()
    cache.apply_hf_env()

    started = time.perf_counter()
    model_dir = cache.volume_root / "models" / MODEL_REPO.split("/")[-1]
    vae_dir = cache.volume_root / "models" / VAE_REPO.split("/")[-1]
    report = ModelCacheReport(model_dir=model_dir, vae_dir=vae_dir)

    for repo_id, target, required, patterns in (
        (MODEL_REPO, model_dir, REQUIRED_MODEL_FILES, MODEL_FILE_PATTERNS),
        (VAE_REPO, vae_dir, REQUIRED_VAE_FILES, VAE_FILE_PATTERNS),
    ):
        missing = _missing_files(target, required)
        if missing:
            log.info("cold cache for %s; fetching (%d files missing)", repo_id, len(missing))
            _ = _snapshot_download(repo_id, target, cache, patterns)
            still_missing = _missing_files(target, required)
            if still_missing:
                raise BootError(
                    f"{repo_id} is still missing {still_missing} after download. "
                    + "The repo layout may have changed — check shared/model-facts.md."
                )
            report.downloaded.append(repo_id)
        else:
            log.info("warm cache for %s at %s", repo_id, target)
            report.already_present.append(repo_id)

    report.elapsed_seconds = time.perf_counter() - started
    return report


def install_model_wheel(cache: CacheConfig | None = None) -> Path:
    """Return the path to the `yue2_infer` wheel cached on the volume.

    The wheel ships inside the `m-a-p/YuE2-3B` repo rather than on PyPI, so it
    is fetched alongside the weights. Image build installs it from here.
    """
    cache = cache or CacheConfig()
    wheel = cache.volume_root / "models" / MODEL_REPO.split("/")[-1] / MODEL_WHEEL
    if not wheel.is_file():
        raise BootError(
            f"Missing {MODEL_WHEEL} at {wheel}. It ships inside {MODEL_REPO}; "
            + "re-run ensure_models() with HF_LOCAL_FILES_ONLY unset."
        )
    return wheel


def load_pipeline(config: Any = None, *, cache: CacheConfig | None = None) -> Any:
    """Construct the resident `YuE2Pipeline`. Once per worker lifetime.

    Imported lazily so this module stays importable on a machine without torch
    (every unit test in `tests/` runs that way).
    """
    cache = cache or CacheConfig()
    cache.apply_hf_env()

    try:
        from yue2 import YuE2Pipeline
    except ImportError as exc:  # pragma: no cover - image always has the wheel
        raise BootError("yue2_infer is not installed; see Dockerfile wheel install") from exc

    report = ensure_models(cache)
    kwargs: dict[str, Any] = {
        "device": "cuda",
        "progress": bool(getattr(config, "show_progress", False)),
        # Point the pipeline at the volume copies. `from_pretrained` accepts a
        # local path as readily as a repo id — `resolve_model` returns any
        # existing directory as-is — and a local path cannot silently
        # re-download a blob we believe is cached.
        "local_files_only": cache.local_files_only,
    }
    # Forwarded explicitly: the pipeline's own default (24) is an assumption
    # about the card, and it converts this into a hard
    # `set_per_process_memory_fraction` cap rather than a hint.
    budget = getattr(config, "memory_budget_gib", None)
    if budget is not None:
        kwargs["memory_budget_gib"] = int(budget)
    if cache.hf_token:
        kwargs["token"] = cache.hf_token

    started = time.perf_counter()
    log.info("loading YuE2Pipeline from %s (cache_hit=%s)", report.model_dir, report.cache_hit)
    try:
        pipe = YuE2Pipeline.from_pretrained(str(report.model_dir), vae=str(report.vae_dir), **kwargs)
    except Exception as exc:
        raise BootError(f"YuE2Pipeline.from_pretrained failed: {exc}") from exc

    log.info("pipeline resident in %.1fs", time.perf_counter() - started)
    return pipe


def health() -> dict[str, Any]:
    """Cheap readiness probe: is the volume mounted and are the weights there?

    Deliberately does not touch the GPU — RunPod calls a health check often, and
    a probe that allocates VRAM is a probe that can fail for the wrong reason.
    """
    cache = CacheConfig()
    model_dir = cache.volume_root / "models" / MODEL_REPO.split("/")[-1]
    vae_dir = cache.volume_root / "models" / VAE_REPO.split("/")[-1]
    return {
        "volume_root": str(cache.volume_root),
        "volume_mounted": cache.volume_root.is_dir(),
        "model_files_missing": _missing_files(model_dir, REQUIRED_MODEL_FILES),
        "vae_files_missing": _missing_files(vae_dir, REQUIRED_VAE_FILES),
        "hf_home": str(cache.hf_home),
        "local_files_only": cache.local_files_only,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def make_on_token(callback: Callable[[str, Any], None] | None) -> Callable[[str, Any], None] | None:
    """Adapt the pipeline's `on_token(phase, token)` hook, if the caller wants one.

    The pipeline has streamed token callbacks built in. We do not stream to the
    client — RunPod serverless has no partial-result channel for a job like this
    — but wiring the hook means a future stage can log progress without touching
    the pipeline call.
    """
    return callback
