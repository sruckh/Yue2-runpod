"""Environment-driven configuration for the YuE2 serverless worker.

Every value here comes from the environment so the same image runs unchanged
against a local volume, a staging bucket, or production. Nothing in this module
imports torch or the pipeline — it is safe to import in tests and in any
subprocess.

Source of truth for the shapes and limits referenced below:
`stages/01-research-and-decisions/output/` and the `shared/` factory files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Absolute path to the directory holding this module. The container's layout
#: (`/app`) and the repo's (`worker/`) share it, which is what makes the flat
#: imports work in both. Used by tests to locate sibling files like
#: `.runpod/tests.json` without hardcoding a repo-relative guess.
WORKER_DIR = Path(__file__).resolve().parent

# --- Model identifiers -------------------------------------------------------
# Weights are never baked into the image (locked decision 4). These are the
# repos `boot.ensure_models()` caches onto the network volume.
MODEL_REPO = "m-a-p/YuE2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"
MODEL_WHEEL = "yue2_infer-0.1.5-py3-none-any.whl"

# Model-family defaults mirrored from the pipeline's own protocol.SongRequest,
# so the worker rejects out-of-range input before a GPU is ever touched.
DEFAULT_COT = "full"
DEFAULT_SEED = 831001
DEFAULT_CFG_SCALE = None
VALID_COT = ("off", "melody", "full")

# `SongRequest.__post_init__` bounds. Duplicated deliberately: the pipeline
# raises deep inside a 70-second generation, we want to fail in milliseconds.
SEED_MAX = 2**63
CFG_SCALE_MIN = 0.0
CFG_SCALE_MAX = 20.0


class ConfigError(RuntimeError):
    """Raised when required environment configuration is missing or malformed."""


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise ConfigError(
            f"Missing required environment variable {name!r}. Set it on the "
            "RunPod endpoint (never bake secrets into the image)."
        )
    return value or ""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StorageConfig:
    """Backblaze B2 (S3-compatible) egress settings — locked decision 5."""

    endpoint_url: str
    key_id: str
    app_key: str
    bucket: str
    region: str = "us-west-004"
    #: Presigned GET lifetime in seconds. Long enough for a caller to download a
    #: multi-minute FLAC, short enough that a leaked URL expires.
    url_ttl_seconds: int = 7 * 24 * 3600
    #: B2 multipart uploads require >= 5 MiB parts; 8 MiB keeps a 48 kHz stereo
    #: FLAC comfortably under the 10 000-part ceiling.
    multipart_chunk_bytes: int = 8 * 1024 * 1024

    @classmethod
    def from_env(cls) -> StorageConfig:
        return cls(
            endpoint_url=_env("B2_ENDPOINT_URL", required=True),
            key_id=_env("B2_KEY_ID", required=True),
            app_key=_env("B2_APP_KEY", required=True),
            bucket=_env("B2_BUCKET", required=True),
            region=_env("B2_REGION", "us-west-004"),
            url_ttl_seconds=_env_int("B2_URL_TTL_SECONDS", 7 * 24 * 3600),
        )


@dataclass(frozen=True)
class CacheConfig:
    """Where model weights live. A RunPod network volume, not the image."""

    #: Mount point of the network volume. RunPod mounts it here by convention.
    volume_root: Path = field(default_factory=lambda: Path(_env("VOLUME_ROOT", "/runpod-volume")))
    #: HuggingFace hub cache inside the volume. Both HF repos share it so a
    #: second cold start re-uses blobs already on disk.
    hf_home: Path | None = None
    #: Set true to forbid network access during model resolution — the correct
    #: setting once a volume is warm, and the thing that proves caching works.
    local_files_only: bool = field(default_factory=lambda: _env_bool("HF_LOCAL_FILES_ONLY", False))
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN") or None)

    def __post_init__(self) -> None:
        if self.hf_home is None:
            object.__setattr__(self, "hf_home", self.volume_root / "hf")

    @property
    def models_dir(self) -> Path:
        """Directory the container writes generated artifacts into, per job."""
        return self.volume_root / "scratch"

    def apply_hf_env(self) -> None:
        """Point the HuggingFace libraries at the volume.

        Must run before anything imports `huggingface_hub`, which reads these
        at import time. `handler.py` calls this as its first statement.
        """
        assert self.hf_home is not None
        os.environ["HF_HOME"] = str(self.hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.hf_home / "hub")
        # Keep the xet transfer cache on the volume too — it is the bulk of
        # the bytes on a cold start. Assigned, not `setdefault`: if a base image
        # happened to export this, `setdefault` would quietly leave the bulk of
        # the download on the container disk, which is the exact failure this
        # function exists to prevent.
        os.environ["HF_XET_CACHE"] = str(self.hf_home / "xet")
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)


@dataclass(frozen=True)
class WorkerConfig:
    """Top-level worker configuration."""

    cache: CacheConfig
    #: Optional so validation/generation can be exercised without B2 credentials
    #: or a network — tests inject their own storage object instead.
    storage: StorageConfig | None = None
    #: Generation timeout handed to RunPod. A full-CoT 3.6-minute song measures
    #: ~71 s on a 4090; 30 min leaves generous headroom for longer songs.
    job_timeout_seconds: int = field(default_factory=lambda: _env_int("JOB_TIMEOUT_SECONDS", 1800))
    #: `progress=False` on the pipeline — its English progress spinner writes to
    #: stderr, which is noise in a serverless log.
    show_progress: bool = field(default_factory=lambda: _env_bool("YUE2_PROGRESS", False))
    default_cot: str = field(default_factory=lambda: _env("DEFAULT_COT", DEFAULT_COT))
    #: Which VAE decoder to build with. Default is the current release; the
    #: `-legacy` repo exists only to reproduce the published benchmark protocol
    #: (locked decision 6) and is not the default.
    #:
    #: This is *our* switch, honoured because `boot.load_pipeline` passes it as
    #: `vae=` to `from_pretrained` and the response reports it back. Do not
    #: confuse it with a package-side env var — the `yue2_infer` wheel reads no
    #: environment variables at all on this code path.
    vae_repo: str = field(default_factory=lambda: _env("YUE2_VAE_REPO", VAE_REPO))
    #: VRAM ceiling handed to the pipeline. It is not advisory: the pipeline turns
    #: this into `torch.cuda.set_per_process_memory_fraction((n-2)/total)`, so it
    #: caps the process hard. The pipeline's own default is 24 — an assumption
    #: about the card, not about our job. Ours is 24 GiB per the locked GPU tier;
    #: set it lower to run on a smaller card rather than hitting an OOM at load.
    memory_budget_gib: int = field(default_factory=lambda: _env_int("MEMORY_BUDGET_GIB", 24))

    @classmethod
    def from_env(cls) -> WorkerConfig:
        """Strict: every B2 variable must be present. Use in a startup check."""
        return cls(cache=CacheConfig(), storage=StorageConfig.from_env())

    @classmethod
    def autoload(cls) -> WorkerConfig:
        """Build from the environment, tolerating absent storage credentials.

        Storage is loaded leniently because a worker must still be able to
        *validate* and *generate* before its bucket credentials are configured —
        and because the actionable error ("set B2_ENDPOINT_URL") belongs at the
        point of upload, not at import, where it would read as a boot failure.
        """
        try:
            storage = StorageConfig.from_env()
        except ConfigError:
            storage = None
        return cls(cache=CacheConfig(), storage=storage)

    def require_storage(self) -> StorageConfig:
        """Return storage config, or explain which env vars are missing."""
        if self.storage is None:
            raise ConfigError(
                "No storage configured. Set B2_ENDPOINT_URL, B2_KEY_ID, "
                "B2_APP_KEY and B2_BUCKET on the endpoint, or inject a storage "
                "object in tests."
            )
        return self.storage

    def validate(self) -> None:
        if self.default_cot not in VALID_COT:
            raise ConfigError(f"DEFAULT_COT must be one of {VALID_COT}, got {self.default_cot!r}")
        # The pipeline reserves 2 GiB, so anything at or below that leaves no
        # budget at all and raises deep inside its constructor.
        if self.memory_budget_gib <= 2:
            raise ConfigError(
                f"MEMORY_BUDGET_GIB must exceed the pipeline's 2 GiB reserve, got {self.memory_budget_gib}"
            )
        # A budget of 0 reaches `signal.alarm(0)`, which *disarms* the alarm
        # rather than setting a zero-second one — the guard would silently
        # vanish instead of firing immediately.
        if self.job_timeout_seconds <= 0:
            raise ConfigError(f"JOB_TIMEOUT_SECONDS must be positive, got {self.job_timeout_seconds}")
