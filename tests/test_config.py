"""Config tests — env parsing, defaults, and the validation that guards boot.

`config.py` is small but load-bearing: a wrong default here surfaces as a boot
failure on paid hardware, or as a 12 GB download on a 20 GB container disk.
"""

from __future__ import annotations

import pytest

from config import (
    CFG_SCALE_MAX,
    DEFAULT_COT,
    DEFAULT_SEED,
    MODEL_REPO,
    VAE_REPO,
    VALID_COT,
    CacheConfig,
    ConfigError,
    StorageConfig,
    WorkerConfig,
)

# --- defaults ----------------------------------------------------------------


def test_model_defaults_match_the_locked_decision() -> None:
    assert MODEL_REPO == "m-a-p/YuE2-3B"
    assert VAE_REPO == "m-a-p/YuE2-Vae"
    assert DEFAULT_COT == "full"
    assert DEFAULT_SEED == 831001
    assert CFG_SCALE_MAX == 20.0
    assert VALID_COT == ("off", "melody", "full")


def test_worker_defaults(volume: None) -> None:
    cfg = WorkerConfig(cache=CacheConfig())
    assert cfg.job_timeout_seconds == 1800
    assert cfg.show_progress is False
    assert cfg.memory_budget_gib == 24
    assert cfg.storage is None


# --- storage is optional until upload ----------------------------------------


def test_autoload_tolerates_missing_storage(volume: None) -> None:
    """A worker must be able to validate and generate before B2 is configured."""
    assert WorkerConfig.autoload().storage is None


def test_autoload_picks_up_storage_when_present(volume: None, b2_env: None) -> None:
    cfg = WorkerConfig.autoload()
    assert cfg.storage is not None
    assert cfg.storage.bucket == "yue2-test"


def test_from_env_is_strict_and_names_the_missing_var(volume: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B2_ENDPOINT_URL", "https://example.invalid")
    with pytest.raises(ConfigError, match="B2_KEY_ID"):
        WorkerConfig.from_env()


def test_require_storage_explains_what_to_set(volume: None) -> None:
    with pytest.raises(ConfigError, match="B2_ENDPOINT_URL"):
        WorkerConfig(cache=CacheConfig()).require_storage()


def test_require_storage_returns_config_when_present(volume: None, b2_env: None) -> None:
    assert WorkerConfig.autoload().require_storage().bucket == "yue2-test"


# --- validation --------------------------------------------------------------


def test_validate_rejects_an_unknown_default_cot(volume: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_COT", "melody-only")
    with pytest.raises(ConfigError, match="DEFAULT_COT"):
        WorkerConfig(cache=CacheConfig()).validate()


def test_validate_rejects_a_budget_inside_the_pipeline_reserve(volume: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline reserves 2 GiB; at or below that there is no budget left."""
    for bad in ("2", "1"):
        monkeypatch.setenv("MEMORY_BUDGET_GIB", bad)
        with pytest.raises(ConfigError, match="2 GiB reserve"):
            WorkerConfig(cache=CacheConfig()).validate()


@pytest.mark.parametrize("good", ["3", "12", "24", "48", "80"])
def test_validate_accepts_sane_budgets(volume: None, monkeypatch: pytest.MonkeyPatch, good: str) -> None:
    monkeypatch.setenv("MEMORY_BUDGET_GIB", good)
    WorkerConfig(cache=CacheConfig()).validate()


# --- env parsing -------------------------------------------------------------


def test_env_int_rejects_garbage(volume: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "thirty")
    with pytest.raises(ConfigError, match="must be an integer"):
        WorkerConfig(cache=CacheConfig())


def test_env_bool_accepts_common_truthy_spellings(volume: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for raw in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("YUE2_PROGRESS", raw)
        assert WorkerConfig(cache=CacheConfig()).show_progress is True
    for raw in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("YUE2_PROGRESS", raw)
        assert WorkerConfig(cache=CacheConfig()).show_progress is False


# --- cache -------------------------------------------------------------------


def test_cache_paths_follow_runpods_documented_layout(volume) -> None:
    """`huggingface-cache/hub` on the volume — where the platform mounts models.

    Not a path of our choosing: RunPod's cached-models feature writes here, so a
    worker reading anywhere else never sees the cache it was configured with.
    """
    cache = CacheConfig()
    assert cache.hf_home == volume / "huggingface-cache"
    assert cache.hub_cache == volume / "huggingface-cache" / "hub"
    assert cache.models_dir == volume / "scratch"


def test_volume_root_is_overridable(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("VOLUME_ROOT", str(tmp_path / "other"))
    assert CacheConfig().volume_root == tmp_path / "other"


def test_apply_hf_env_respects_an_explicit_token(volume: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.setenv("HF_TOKEN", "hf_explicit")
    CacheConfig().apply_hf_env()
    assert os.environ["HF_TOKEN"] == "hf_explicit"


def test_local_files_only_is_off_by_default(volume: None) -> None:
    assert CacheConfig().local_files_only is False


# --- storage config ----------------------------------------------------------


def test_storage_defaults(b2_env: None) -> None:
    cfg = StorageConfig.from_env()
    assert cfg.region == "us-west-004"
    assert cfg.url_ttl_seconds == 7 * 24 * 3600
    assert cfg.multipart_chunk_bytes == 8 * 1024 * 1024


def test_storage_rejects_a_non_integer_ttl(b2_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B2_URL_TTL_SECONDS", "forever")
    with pytest.raises(ConfigError, match="B2_URL_TTL_SECONDS"):
        StorageConfig.from_env()
