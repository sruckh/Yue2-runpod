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

from config import ASR_REPO, MODEL_REPO, MODEL_WHEEL, SHEETSAGE_REPO, VAE_REPO, CacheConfig

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

# --- Cover/edit transcription models -----------------------------------------
#
# Cached for the same reason YuE2 is: `ensure_models` enables HF offline mode
# once it has verified the cache, and that switch is inherited by the cover
# subprocesses via `subprocess_runner._INHERITED_ENV`. A repo the cover path
# needs but this list omits is therefore **unreachable**, not merely slow — the
# child cannot fall back to the network because we told it not to.
#
# That is not hypothetical: it is what made the first real cover job fail in
# 5.06 s with "We couldn't connect to 'https://huggingface.co' ... and it looks
# like m-a-p/SheetSage2 is not the path to a directory containing a file named
# config.json", which reads as a network fault and is really a scope error here.
REQUIRED_SHEETSAGE_FILES = (
    "config.json",
    "model.safetensors",
)
SHEETSAGE_FILE_PATTERNS = (
    *REQUIRED_SHEETSAGE_FILES,
    "model.safetensors.index.json",
    "LICENSE",
)
REQUIRED_ASR_FILES = (
    "config.json",
    "model.safetensors",
)
ASR_FILE_PATTERNS = (
    *REQUIRED_ASR_FILES,
    "model.safetensors.index.json",
    "generation_config.json",
    "LICENSE",
)

#: `(repo_id, required files, download patterns)` for every repo the worker can
#: need. `ensure_models` iterates this, and tests assert the cover path's repos
#: are covered — a repo absent from here is one offline mode makes unreachable.
CACHED_REPOS = (
    (MODEL_REPO, REQUIRED_MODEL_FILES, MODEL_FILE_PATTERNS),
    (VAE_REPO, REQUIRED_VAE_FILES, VAE_FILE_PATTERNS),
    (SHEETSAGE_REPO, REQUIRED_SHEETSAGE_FILES, SHEETSAGE_FILE_PATTERNS),
    (ASR_REPO, REQUIRED_ASR_FILES, ASR_FILE_PATTERNS),
)


class BootError(RuntimeError):
    """Weights could not be cached, or the pipeline could not be constructed."""


@dataclass
class ModelCacheReport:
    """What `ensure_models()` found and did — surfaced in the health response.

    Splits the two sources explicitly, because they mean different things in
    production: `from_cache` is RunPod's caching mechanism working (the intended
    path, near-instant), while `downloaded` is the network-volume fallback being
    exercised — a first cold start, or a model the endpoint's cached-models
    configuration does not cover.
    """

    #: Repo id -> resolved snapshot directory, for models RunPod had cached.
    from_cache: dict[str, Path] = field(default_factory=dict)
    #: Repo id -> resolved snapshot directory, for models we had to fetch.
    downloaded: dict[str, Path] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def cache_hit(self) -> bool:
        """True when nothing had to be fetched — RunPod's cache served everything."""
        return not self.downloaded

    @property
    def model_dir(self) -> Path | None:
        """The YuE2 snapshot directory in use, whichever source provided it."""
        return self.from_cache.get(MODEL_REPO) or self.downloaded.get(MODEL_REPO)

    @property
    def vae_dir(self) -> Path | None:
        """The VAE snapshot directory in use, whichever source provided it."""
        return self.from_cache.get(VAE_REPO) or self.downloaded.get(VAE_REPO)


def _missing_files(directory: Path, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not (directory / name).is_file()]


def _download_into_cache(repo_id: str, cache: CacheConfig, allow_patterns: tuple[str, ...]) -> None:
    """Download a repo into the HF cache root — the network-volume fallback.

    **`cache_dir`, not `local_dir`.** This is the distinction that matters:

    - `local_dir=` produces a flat directory in a layout only this worker knows
      about. RunPod's cached-models feature cannot see it, so the model is
      invisible to the platform and gets downloaded again.
    - `cache_dir=` produces the standard hub layout
      (`models--{org}--{name}/snapshots/{rev}/`) under the path RunPod mounts
      its own cache at, so both sources are one tree and
      `resolve_cached_snapshot` reads either.

    An earlier version used `local_dir` and was therefore incompatible with the
    endpoint's caching configuration.

    `allow_patterns` is not an optimisation we invented — the pipeline's own Hub
    path restricts the download the same way, and without it a cold start pulls
    the repo's demo audio and artwork for nothing.
    """
    from huggingface_hub import snapshot_download

    cache.hub_cache.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache.hub_cache),
            token=cache.hf_token,
            # Not `cache.local_files_only`: this only runs when the model was
            # absent, so forbidding network access here would make the fallback
            # impossible by construction. Offline mode is enabled once the cache
            # is verified complete.
            local_files_only=False,
            allow_patterns=list(allow_patterns),
            # No `local_dir_use_symlinks`: hf-hub 0.36.2 declares the parameter
            # but ignores it ("deprecated and will be ignored"), and the
            # behaviour it requested is what the cache layout does anyway.
        )
    except Exception as exc:
        hint = ""
        if cache.local_files_only:
            hint = " (HF_LOCAL_FILES_ONLY=true forbids downloads; unset it for the first cold start)"
        raise BootError(f"Could not cache {repo_id} into {cache.hub_cache}{hint}: {exc}") from exc


def resolve_cached_snapshot(repo_id: str, cache: CacheConfig | None = None) -> Path | None:
    """Locate a repo in the HuggingFace cache, or `None` if it is absent.

    This is RunPod's documented resolution, and it is the *only* way models are
    located here — whether they arrived via the endpoint's cached-models
    configuration or via our own fallback download, they end up in the same
    tree. One resolver, so the two paths cannot disagree about where a model is.

    The layout is the standard HF hub cache::

        {hub_cache}/models--{org}--{name}/snapshots/{revision}/

    with `refs/main` naming the current revision. Reading `refs/main` is tried
    first because it is authoritative; falling back to any snapshot directory
    keeps a cache usable when `refs/` is absent.
    """
    cache = cache or CacheConfig()
    if "/" not in repo_id:
        raise BootError(f"repo id {repo_id!r} must be in 'org/name' form")

    org, name = repo_id.split("/", 1)
    root = cache.hub_cache / f"models--{org}--{name}"
    snapshots = root / "snapshots"

    refs_main = root / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        candidate = snapshots / revision
        if candidate.is_dir():
            return candidate

    if snapshots.is_dir():
        revisions = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if revisions:
            return revisions[-1]
    return None


def ensure_models(cache: CacheConfig | None = None) -> ModelCacheReport:
    """Make both repos available, preferring RunPod's cache. Idempotent.

    Order per repo:

    1. Resolve it from the cache. On an endpoint with cached models configured
       this succeeds immediately and costs nothing — the intended path.
    2. Only if absent, download it into the same cache root (the network-volume
       fallback), then resolve again.

    Returns a report saying which path each repo took, so a cold start's minutes
    of wall time are attributable rather than mysterious.
    """
    cache = cache or CacheConfig()
    cache.apply_hf_env()

    started = time.perf_counter()
    report = ModelCacheReport()

    for repo_id, required, patterns in CACHED_REPOS:
        found = resolve_cached_snapshot(repo_id, cache)
        if found is not None and not _missing_files(found, required):
            log.info("%s resolved from cache at %s", repo_id, found)
            report.from_cache[repo_id] = found
            continue

        if found is not None:
            # Present but incomplete: a partial download, or an upstream layout
            # change. Re-fetch rather than boot on a broken tree.
            log.warning("%s is cached but missing %s; re-fetching", repo_id, _missing_files(found, required))
        else:
            log.info("%s not in the cache; downloading (network-volume fallback)", repo_id)

        _download_into_cache(repo_id, cache, patterns)
        found = resolve_cached_snapshot(repo_id, cache)
        if found is None:
            raise BootError(
                f"{repo_id} is not in the cache at {cache.hub_cache} after downloading. "
                + "Check that the endpoint's cached-models configuration and the volume agree."
            )
        still_missing = _missing_files(found, required)
        if still_missing:
            raise BootError(
                f"{repo_id} is still missing {still_missing} after download. "
                + "The repo layout may have changed — check shared/model-facts.md."
            )
        report.downloaded[repo_id] = found

    # Every repo is verified present, so resolution can safely go offline from
    # here. This is what makes a later missing blob fail loudly rather than
    # quietly re-downloading 12 GB.
    cache.enable_offline_mode()

    report.elapsed_seconds = time.perf_counter() - started
    return report


def install_model_wheel(cache: CacheConfig | None = None) -> Path:
    """Return the path to the `yue2_infer` wheel, wherever the cache put it.

    The wheel ships inside the `m-a-p/YuE2-3B` repo rather than on PyPI, so it
    arrives with the weights. In the hub cache layout large files are symlinks
    into `blobs/`; `is_file()` follows them, so this reads either a real file or
    one.

    The image installs the wheel directly from HuggingFace at build time rather
    than from here — the cache does not exist during a build. This exists for a
    runtime check that the wheel's source repo is genuinely present.
    """
    cache = cache or CacheConfig()
    snapshot = resolve_cached_snapshot(MODEL_REPO, cache)
    if snapshot is None:
        raise BootError(f"{MODEL_REPO} is not in the cache at {cache.hub_cache}")

    wheel = snapshot / MODEL_WHEEL
    if not wheel.is_file():
        # `allow_patterns` includes `*.whl`, so its absence means either a cache
        # entry predating that pattern or an upstream change.
        raise BootError(
            f"Missing {MODEL_WHEEL} in {snapshot}. It ships inside {MODEL_REPO}; "
            + "re-run ensure_models() with HF_LOCAL_FILES_ONLY unset to refresh the cache."
        )
    return wheel


def _resolve_vae(config: Any, report: ModelCacheReport) -> Any:
    """Pick the VAE directory for the decoder in use.

    The default decoder is the one `ensure_models()` cached, so we pass the
    local path — a local path cannot silently re-download. A non-default decoder
    (the legacy benchmark repo) has not been cached by `ensure_models`, so its
    repo id is passed instead and `resolve_model` fetches it into the HF cache.
    """
    requested = getattr(config, "vae_repo", None) or VAE_REPO
    if requested == VAE_REPO:
        if report.vae_dir is None:
            raise BootError(f"{VAE_REPO} was not resolved by ensure_models(); cannot build the pipeline")
        return report.vae_dir
    log.info("using non-default decoder %s (default is %s)", requested, VAE_REPO)
    return requested


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
        # `ensure_models` enables offline mode once the cache is verified
        # complete, so the pipeline resolves from disk. Passing a local snapshot
        # path (below) also short-circuits `resolve_model` on `is_dir()`, so it
        # never reaches the Hub regardless.
        "local_files_only": True,
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
        # `vae=` is the repo the decoder is actually built from, and the
        # handler reports this same value back as `decoder`. `report.vae_dir`
        # is where the default VAE was cached; a legacy override is passed by
        # repo id so `resolve_model` fetches or resolves it itself.
        pipe = YuE2Pipeline.from_pretrained(
            str(report.model_dir),
            vae=str(_resolve_vae(config, report)),
            **kwargs,
        )
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
    model_dir = resolve_cached_snapshot(MODEL_REPO, cache)
    vae_dir = resolve_cached_snapshot(VAE_REPO, cache)
    return {
        "volume_root": str(cache.volume_root),
        "volume_mounted": cache.volume_root.is_dir(),
        # The path RunPod's cached-models feature writes to. Reporting it makes a
        # misconfigured endpoint diagnosable: an empty directory here means the
        # cache was never populated and every start pays the fallback download.
        "hub_cache": str(cache.hub_cache),
        "hub_cache_populated": cache.hub_cache.is_dir(),
        "model_source": "cache" if model_dir else "absent",
        "model_snapshot": str(model_dir) if model_dir else None,
        "model_files_missing": (
            _missing_files(model_dir, REQUIRED_MODEL_FILES) if model_dir else list(REQUIRED_MODEL_FILES)
        ),
        "vae_source": "cache" if vae_dir else "absent",
        "vae_snapshot": str(vae_dir) if vae_dir else None,
        "vae_files_missing": _missing_files(vae_dir, REQUIRED_VAE_FILES) if vae_dir else list(REQUIRED_VAE_FILES),
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
