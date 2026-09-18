"""Boot tests — RunPod's model cache first, network volume as fallback.

The behaviour under test is the *contract with the platform*: models declared in
the endpoint's cached-models configuration are mounted at
`{volume}/huggingface-cache/hub` in the standard HuggingFace cache layout, and a
worker must read them from exactly there.

An earlier version of `boot.py` downloaded with `local_dir=`, producing a tree in
a layout only this worker understood. RunPod's cache could not see it, so every
start re-downloaded 12 GB of weights the platform had already fetched — the
caching never actually worked. These tests pin the layout so that cannot recur.
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import hub_cache_root, populate_runpod_cache, snapshot_dir

import boot
from boot import (
    CACHED_REPOS,
    REQUIRED_MODEL_FILES,
    REQUIRED_VAE_FILES,
    BootError,
    ensure_models,
    health,
    install_model_wheel,
    resolve_cached_snapshot,
)
from config import ASR_REPO, MODEL_REPO, MODEL_WHEEL, VAE_REPO, CacheConfig


class SnapshotRecorder:
    """Stands in for `huggingface_hub.snapshot_download`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, repo_id: str, **kwargs: Any) -> str:
        self.calls.append({"repo_id": repo_id, **kwargs})
        # Mimic the real thing: write into the hub cache layout under cache_dir.
        cache_dir = Path(kwargs["cache_dir"])
        org, name = repo_id.split("/", 1)
        snap = cache_dir / f"models--{org}--{name}" / "snapshots" / "deadbeef"
        snap.mkdir(parents=True, exist_ok=True)
        # Derive this repo's own spec rather than assuming one of two. The
        # previous form wrote REQUIRED_VAE_FILES for *any* non-YuE2 repo, so a
        # third and fourth repo appeared to download files they never asked for
        # — and the presence check then failed against the fake.
        spec = next((entry for entry in CACHED_REPOS if entry[0] == repo_id), None)
        files = list(spec[1]) if spec else []
        if spec and spec[3]:
            # spec[3] is a glob like "*.safetensors"; write a real weight file
            # that satisfies it. Index-slicing a glob would write a file named
            # "*", which matches nothing.
            files.append("model.safetensors")
        for filename in files:
            (snap / filename).write_bytes(b"x")
        if repo_id == MODEL_REPO:
            (snap / MODEL_WHEEL).write_bytes(b"PK")
        refs = snap.parent.parent / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text("deadbeef", encoding="utf-8")
        return str(snap)


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> SnapshotRecorder:
    recorder = SnapshotRecorder()
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return recorder


def cached_all(volume: Path) -> None:
    """Every repo present, as RunPod would leave them.

    Driven by `CACHED_REPOS` rather than a hand-written pair, because the pair
    is exactly what went stale: the list grew to four and this helper kept
    populating two, so a "cache hit" test would have passed against a cache the
    worker no longer considers complete. Deriving it means a repo added to the
    worker is populated here automatically.
    """
    for repo_id, required, _patterns, weights_glob in CACHED_REPOS:
        files = list(required)
        if weights_glob:
            # A real weight file. `model.safetensors` satisfies the glob whatever
            # the repo's real layout is, and — unlike an index — it is weights.
            files.append("model.safetensors")
        if repo_id == MODEL_REPO:
            files.append(MODEL_WHEEL)
        populate_runpod_cache(volume, repo_id, tuple(files))


# =============================================================================
# Path resolution — the contract with the platform
# =============================================================================


def test_cache_paths_match_runpods_documented_layout(volume: Path) -> None:
    """The paths must be exactly RunPod's, not something equivalent-but-different."""
    assert CacheConfig().hub_cache == volume / "huggingface-cache" / "hub"


def test_resolves_a_model_runpod_cached(volume: Path) -> None:
    expected = populate_runpod_cache(volume, MODEL_REPO, REQUIRED_MODEL_FILES)
    assert resolve_cached_snapshot(MODEL_REPO) == expected


def test_resolves_via_refs_main(volume: Path) -> None:
    """`refs/main` is authoritative — a stale second snapshot must not win."""
    populate_runpod_cache(volume, MODEL_REPO, REQUIRED_MODEL_FILES, revision="old")
    newest = populate_runpod_cache(volume, MODEL_REPO, REQUIRED_MODEL_FILES, revision="current")
    assert resolve_cached_snapshot(MODEL_REPO) == newest


def test_falls_back_to_a_snapshot_without_refs(volume: Path) -> None:
    """A cache with no `refs/` is still usable rather than fatal."""
    snap = populate_runpod_cache(volume, MODEL_REPO, REQUIRED_MODEL_FILES, write_ref=False)
    assert resolve_cached_snapshot(MODEL_REPO) == snap


def test_absent_repo_resolves_to_none(volume: Path) -> None:
    assert resolve_cached_snapshot(MODEL_REPO) is None


def test_malformed_repo_id_is_rejected(volume: Path) -> None:
    with pytest.raises(BootError, match="org/name"):
        resolve_cached_snapshot("no-org-prefix")


# =============================================================================
# RunPod's cache is the primary path
# =============================================================================


def test_runpod_cache_hit_downloads_nothing(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """The whole point: if RunPod cached it, the worker must not fetch it again."""
    cached_all(volume)
    report = ensure_models(CacheConfig())

    assert report.cache_hit is True
    assert fake_hf.calls == [], "a populated RunPod cache must not trigger a download"
    # All four, not the two the create path uses: offline mode is global, so a
    # repo the cover path needs but this list omits is unreachable, not slow.
    assert set(report.from_cache) == {repo for repo, _, _, _ in CACHED_REPOS}
    assert report.downloaded == {}


def test_cache_hit_reports_the_snapshot_paths(volume: Path, fake_hf: SnapshotRecorder) -> None:
    cached_all(volume)
    report = ensure_models(CacheConfig())
    assert report.model_dir == snapshot_dir(volume, MODEL_REPO)
    assert report.vae_dir == snapshot_dir(volume, VAE_REPO)


# =============================================================================
# Network-volume fallback
# =============================================================================


def test_missing_model_falls_back_to_download(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """Only the absent repo is fetched; the cached ones are left alone.

    The assertion is set-based on purpose. Listing the expected repos by hand is
    what let this test keep asserting a two-repo scope after the worker grew to
    four — it was describing the code it was written against, not the contract.
    """
    cached_all(volume)
    # Drop exactly one repo's tree so it is the only miss.
    shutil.rmtree(volume / "huggingface-cache" / "hub" / f"models--{MODEL_REPO.replace('/', '--')}")

    report = ensure_models(CacheConfig())

    assert [c["repo_id"] for c in fake_hf.calls] == [MODEL_REPO], "only the absent repo may be fetched"
    assert set(report.downloaded) == {MODEL_REPO}
    assert set(report.from_cache) == {repo for repo, _, _, _ in CACHED_REPOS} - {MODEL_REPO}


def test_fallback_uses_cache_dir_not_local_dir(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """The bug this whole module guards.

    `local_dir=` produces a layout RunPod's cache feature cannot see, so the
    platform's cache is ignored and every start re-downloads. `cache_dir=` is
    what produces the shared hub layout.
    """
    ensure_models(CacheConfig())
    for call in fake_hf.calls:
        assert "cache_dir" in call, "must pass cache_dir"
        assert "local_dir" not in call, "local_dir produces a layout RunPod cannot read"
        assert call["cache_dir"] == str(volume / "huggingface-cache" / "hub")


def test_fallback_writes_into_the_runpod_cache_root(volume: Path, fake_hf: SnapshotRecorder) -> None:
    ensure_models(CacheConfig())
    for repo_id in (MODEL_REPO, VAE_REPO):
        resolved = resolve_cached_snapshot(repo_id)
        assert resolved is not None
        assert resolved.is_relative_to(hub_cache_root(volume))


def test_second_call_after_fallback_needs_no_network(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """The fallback must warm the cache for the next start, not just this one."""
    ensure_models(CacheConfig())
    first = len(fake_hf.calls)

    report = ensure_models(CacheConfig())
    assert len(fake_hf.calls) == first, "the fallback did not populate the cache it reads"
    assert report.cache_hit is True


def test_fallback_downloads_both_when_volume_is_cold(volume: Path, fake_hf: SnapshotRecorder) -> None:
    report = ensure_models(CacheConfig())
    assert set(report.downloaded) == {repo for repo, _, _, _ in CACHED_REPOS}
    assert report.from_cache == {}


# =============================================================================
# Offline mode
# =============================================================================


def test_offline_mode_enabled_once_the_cache_is_verified(
    volume: Path, fake_hf: SnapshotRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RunPod's documented pattern: cache verified, then resolve offline."""
    import os

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    cached_all(volume)
    ensure_models(CacheConfig())

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_offline_mode_does_not_block_the_fallback(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """Enabling offline mode first would make the fallback impossible."""
    ensure_models(CacheConfig())
    assert fake_hf.calls, "the fallback must be allowed to reach the network"
    for call in fake_hf.calls:
        assert call["local_files_only"] is False, "offline mode leaked into the download call"


# =============================================================================
# Incomplete caches
# =============================================================================


def test_incomplete_cache_is_refetched(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """Present-but-broken must not boot: re-fetch instead."""
    populate_runpod_cache(volume, MODEL_REPO, ("config.json",))  # missing the weights
    populate_runpod_cache(volume, VAE_REPO, REQUIRED_VAE_FILES)

    report = ensure_models(CacheConfig())
    assert MODEL_REPO in report.downloaded


def test_download_that_writes_nothing_names_the_endpoint_config(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that leaves the cache empty must say what to check.

    The operationally important failure: the endpoint's cache was configured for
    a repo the volume does not have, and the fallback also came up empty. The
    message points at the one place to look.
    """
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda *a, **k: "ok"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(BootError, match="cached-models configuration"):
        ensure_models(CacheConfig())


def test_incomplete_after_a_real_download_names_the_repo_layout(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A download that lands *some* files but not the required ones.

    An upstream layout change rather than a misconfiguration — a different
    diagnosis, so a different message.
    """
    module = types.ModuleType("huggingface_hub")

    def partial(repo_id: str, **kwargs: Any) -> str:
        cache_dir = Path(kwargs["cache_dir"])
        org, name = repo_id.split("/", 1)
        snap = cache_dir / f"models--{org}--{name}" / "snapshots" / "rev1"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_bytes(b"x")  # present, but no weights
        refs = snap.parent.parent / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text("rev1", encoding="utf-8")
        return str(snap)

    module.snapshot_download = partial  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(BootError, match=r"model-facts\.md"):
        ensure_models(CacheConfig())


# =============================================================================
# allow_patterns
# =============================================================================


def test_download_excludes_demo_assets(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """The repo ships demo MP3s and artwork the worker never reads."""
    ensure_models(CacheConfig())
    for call in fake_hf.calls:
        patterns = call["allow_patterns"]
        assert patterns
        assert not any(p.startswith("assets/") for p in patterns)
        # Either spelling downloads the weights: `model.safetensors` for a
        # single-file repo, `*.safetensors` to catch shards. Asserting one
        # spelling is what let the ASR repo ship with its weights unreachable.
        assert "model.safetensors" in patterns or "*.safetensors" in patterns, (
            f"{call['repo_id']}: patterns cannot download the weights: {patterns}"
        )


def test_model_download_includes_the_wheel(volume: Path, fake_hf: SnapshotRecorder) -> None:
    """`yue2_infer` ships in this repo rather than on PyPI."""
    ensure_models(CacheConfig())
    model_call = next(c for c in fake_hf.calls if c["repo_id"] == MODEL_REPO)
    assert "*.whl" in model_call["allow_patterns"]


# =============================================================================
# Wheel lookup
# =============================================================================


def test_install_model_wheel_finds_it_in_the_cache(volume: Path) -> None:
    populate_runpod_cache(volume, MODEL_REPO, (*REQUIRED_MODEL_FILES, MODEL_WHEEL))
    assert install_model_wheel(CacheConfig()) == snapshot_dir(volume, MODEL_REPO) / MODEL_WHEEL


def test_install_model_wheel_reports_a_missing_repo(volume: Path) -> None:
    with pytest.raises(BootError, match="not in the cache"):
        install_model_wheel(CacheConfig())


def test_install_model_wheel_reports_a_missing_wheel(volume: Path) -> None:
    populate_runpod_cache(volume, MODEL_REPO, REQUIRED_MODEL_FILES)  # no wheel
    with pytest.raises(BootError, match="ships inside"):
        install_model_wheel(CacheConfig())


# =============================================================================
# health
# =============================================================================


def test_health_reports_a_populated_cache(volume: Path) -> None:
    cached_all(volume)
    h = health()
    assert h["volume_mounted"] is True
    assert h["hub_cache_populated"] is True
    assert h["model_source"] == "cache"
    assert h["model_files_missing"] == []
    assert h["vae_files_missing"] == []


def test_health_reports_an_absent_cache_without_raising(volume: Path) -> None:
    """A probe must describe a cold volume, not fail on one."""
    h = health()
    assert h["model_source"] == "absent"
    assert h["hub_cache_populated"] is False
    assert "model.safetensors" in h["model_files_missing"]


def test_health_names_the_hub_cache_path(volume: Path) -> None:
    """A misconfigured endpoint should be diagnosable from this field."""
    assert health()["hub_cache"] == str(volume / "huggingface-cache" / "hub")


def test_health_does_not_require_a_gpu(volume: Path) -> None:
    assert "cuda_visible_devices" in health()


# =============================================================================
# env plumbing
# =============================================================================


def test_apply_hf_env_points_at_runpods_cache(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for var in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        monkeypatch.delenv(var, raising=False)
    CacheConfig().apply_hf_env()

    assert os.environ["HF_HOME"] == str(volume / "huggingface-cache")
    assert os.environ["HF_HUB_CACHE"] == str(volume / "huggingface-cache" / "hub")
    assert os.environ["HF_XET_CACHE"].startswith(str(volume))


def test_load_pipeline_reports_missing_wheel_module(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cached_all(volume)
    monkeypatch.setitem(sys.modules, "yue2", None)
    with pytest.raises(BootError, match="yue2_infer is not installed"):
        boot.load_pipeline(None, cache=CacheConfig())


# =============================================================================
# The cache list must cover every repo the worker can need
# =============================================================================
#
# The bug these exist for, found by the first real cover job (2026-09-18):
#
#     cover mode: melody transcription failed — We couldn't connect to
#     'https://huggingface.co' ... and it looks like m-a-p/SheetSage2 is not the
#     path to a directory containing a file named config.json
#
# `ensure_models` cached only YuE2's two repos, then called
# `cache.enable_offline_mode()` — which sets `HF_HUB_OFFLINE` and
# `TRANSFORMERS_OFFLINE` **globally**, and `subprocess_runner._INHERITED_ENV`
# passes both to the cover children. So SheetSage2 was not merely uncached, it
# was unreachable: the subprocess had been told not to use the network.
#
# The failure reads as a network fault and is really a scope error in this list,
# which is why it is pinned structurally rather than by checking one repo.


def test_every_repo_the_cover_children_default_to_is_cached() -> None:
    """The ids the subprocesses actually request must all be in `CACHED_REPOS`.

    Read from the child entrypoints rather than hardcoded here, so renaming a
    model in `transcribe_sheetsage/run.py` without caching it fails this test
    instead of failing a GPU job five minutes in.
    """
    import ast

    worker = Path(__file__).resolve().parent.parent / "worker"
    cached = {repo for repo, _, _, _ in CACHED_REPOS}

    child_defaults: dict[str, str] = {}
    for entrypoint in ("transcribe_sheetsage/run.py", "transcribe_asr/run.py"):
        tree = ast.parse((worker / entrypoint).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEFAULT_MODEL" for t in node.targets):
                child_defaults[entrypoint] = ast.literal_eval(node.value)

    assert child_defaults, "no DEFAULT_MODEL found — did the entrypoints get renamed?"
    for entrypoint, repo_id in child_defaults.items():
        assert repo_id in cached, (
            f"{entrypoint} defaults to {repo_id!r}, which is not in CACHED_REPOS. "
            "Offline mode is enabled after the cache is verified and is inherited by "
            "this subprocess, so an uncached repo is unreachable, not merely slow."
        )


def test_offline_mode_is_what_makes_an_uncached_repo_unreachable() -> None:
    """Pins the causal link, so the reasoning above is not just a story.

    If offline mode ever stops being global or stops being inherited, this test
    fails and the constraint on `CACHED_REPOS` can be reconsidered. Until then
    it is load-bearing.
    """
    from subprocess_runner import _INHERITED_ENV

    assert "HF_HUB_OFFLINE" in _INHERITED_ENV
    assert "TRANSFORMERS_OFFLINE" in _INHERITED_ENV


def test_offline_mode_is_enabled_only_after_every_repo_is_verified() -> None:
    """Offline before the check would make the fallback download impossible.

    `enable_offline_mode` must run *after* the loop, or the network-volume
    fallback — the whole point of the second half of `ensure_models` — could
    never fetch anything.
    """
    import inspect

    source = inspect.getsource(ensure_models)
    loop_at = source.index("for repo_id, required, patterns, weights_glob in CACHED_REPOS")
    offline_at = source.index("cache.enable_offline_mode()")
    assert loop_at < offline_at, "offline mode is enabled before the cache is verified"


def test_the_cover_repos_are_required_not_merely_downloaded() -> None:
    """A repo listed but with an empty `required` tuple would never be checked.

    `ensure_models` treats a repo as present when it resolves *and* holds every
    required file. An empty tuple makes "present" mean "a directory exists",
    which is how a truncated download would slip through.
    """
    for repo_id, required, patterns, _glob in CACHED_REPOS:
        assert required, f"{repo_id} has no required files, so its presence is never verified"
        assert "config.json" in required, f"{repo_id} does not require config.json"
        for name in required:
            assert name in patterns, f"{repo_id}: {name!r} is required but not downloaded"


#: Cross-repo references recorded upstream, verified 2026-09-18 by reading each
#: repo's `config.json`. A repo that loads another inside its own
#: `from_pretrained` cannot be covered by scanning this repo's entrypoints —
#: nothing here names the parent. Pinning the known chain lets a *change* be
#: detected without the network.
KNOWN_TRANSITIVE_LOADS = {
    "m-a-p/SheetSage2": "m-a-p/MERT-v2-FullSong",
}


def test_a_parent_model_loaded_transitively_is_cached() -> None:
    """The second instance of the offline-mode trap, one level deeper.

    Caching `m-a-p/SheetSage2` is not enough: it is a 229 MB adapter, and
    `SheetSage2Model.from_pretrained` fetches its 2.5 GB encoder parent
    (`m-a-p/MERT-v2-FullSong`) at a revision pinned in SheetSage2's own
    config.json. Nothing in *this* repo names that parent, so a scan of our
    entrypoints cannot see it — which is exactly why it was missed.

    The mapping is asserted against the cache list rather than fetched, so the
    test needs no network. When upstream changes the chain this fails, and the
    fix is to re-read the configs and update both.
    """
    cached = {repo for repo, _, _, _ in CACHED_REPOS}
    for child, parent in KNOWN_TRANSITIVE_LOADS.items():
        assert child in cached, f"{child} loads {parent} but is not itself cached"
        assert parent in cached, (
            f"{child} loads {parent} inside its own from_pretrained. Offline mode "
            "is inherited by the subprocess, so an uncached parent is unreachable — "
            "the cover job fails with a HuggingFace connection error that reads like "
            "a network fault."
        )


def test_the_transitive_chain_is_documented_where_a_reader_will_look() -> None:
    """The chain must be discoverable without re-reading upstream configs.

    It was invisible once: `DEFAULT_MODEL` looked like the whole story. The note
    in the entrypoint and the constant in `config` are what make it findable.
    """
    worker = Path(__file__).resolve().parent.parent / "worker"
    entrypoint = (worker / "transcribe_sheetsage" / "run.py").read_text(encoding="utf-8")
    assert "MERT-v2-FullSong" in entrypoint, "the transitive parent is not mentioned at the entrypoint"
    assert "MERT_REPO" in (worker / "config.py").read_text(encoding="utf-8")


# =============================================================================
# Weights may be single-file or sharded
# =============================================================================
#
# The bug these exist for, from a production boot (2026-09-18):
#
#     Worker unavailable: Qwen/Qwen3-ASR-1.7B is still missing
#     ['model.safetensors'] after download.
#
# Two faults in one list, both from writing ASR's entry by analogy with the
# other four repos instead of reading ASR's own file listing:
#
#   1. `REQUIRED_ASR_FILES` demanded `model.safetensors`, which does not exist
#      in that repo — it ships two shards plus an index.
#   2. `ASR_FILE_PATTERNS` did not match `model-0000*-of-00002.safetensors`, so
#      the weights had not been downloaded at all. The error named a file that
#      had never existed upstream.
#
# The fix is to stop naming a spelling: `WEIGHTS_ANY` accepts either layout, and
# the patterns use `*.safetensors`.


def test_weights_check_accepts_a_single_file(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"w")
    assert boot._missing_files(tmp_path, ("config.json",), boot.WEIGHTS_GLOB) == []


def test_weights_check_accepts_shards_plus_an_index(tmp_path: Path) -> None:
    """The ASR repo's actual layout, which the old check rejected."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"w")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"w")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert boot._missing_files(tmp_path, ("config.json",), boot.WEIGHTS_GLOB) == []


def test_weights_check_reports_a_repo_with_no_weights(tmp_path: Path) -> None:
    """A config-only tree must fail, and say what was expected."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    missing = boot._missing_files(tmp_path, ("config.json",), boot.WEIGHTS_GLOB)
    assert missing, "a repo with no weights must not pass the presence check"
    assert "safetensors" in missing[0], "the error must name what was expected"


#: Repos whose weights ship as multiple shards, verified against the Hub on
#: 2026-09-18 by reading each listing. The other four each ship exactly
#: `model.safetensors` and are single-file.
KNOWN_SHARDED_REPOS = frozenset({ASR_REPO})


def test_a_sharded_repo_can_actually_download_its_shards() -> None:
    """A glob, not a spelled-out filename — the assertion that was missing.

    Accepting `model.safetensors` as "can download weights" is not enough: for a
    sharded repo that name matches nothing, and the list looks fine while the
    weights are unreachable. The check therefore keys on the *known layout*
    rather than on either spelling being present.
    """
    for repo_id, _required, patterns, _glob in CACHED_REPOS:
        if repo_id in KNOWN_SHARDED_REPOS:
            assert "*.safetensors" in patterns, (
                f"{repo_id} ships shards; without a glob its weights cannot be downloaded at all. patterns={patterns}"
            )
        else:
            assert "model.safetensors" in patterns, f"{repo_id}: single-file weights not downloaded"


def test_the_sharded_layout_fact_is_what_the_asr_entry_relies_on() -> None:
    """Ties the verified layout to the entry, so a change to either is caught.

    If ASR ever ships single-file, this fails and both `KNOWN_SHARDED_REPOS` and
    the patterns can be updated together — rather than one drifting.
    """
    assert ASR_REPO in KNOWN_SHARDED_REPOS
    patterns = next(spec for spec in CACHED_REPOS if spec[0] == ASR_REPO)[2]
    assert "*.safetensors" in patterns
    # The four single-file repos must NOT be in the sharded set.
    for repo_id, _r, _p, _w in CACHED_REPOS:
        if repo_id != ASR_REPO:
            assert repo_id not in KNOWN_SHARDED_REPOS


def test_required_files_are_downloadable_and_weights_are_checked_by_layout() -> None:
    """Two rules, both checkable without the network.

    1. Every `required` name must appear in `patterns`, or the check demands a
       file the download never fetches — a guaranteed boot failure.
    2. `WEIGHTS_ANY` must be set for every repo, so presence is satisfied by
       either layout.

    Note what this does *not* assert: that `required` avoids naming
    `model.safetensors`. The four single-file repos legitimately name it, and
    their listings were verified against the Hub — YuE2-3B, YuE2-Vae, SheetSage2
    and MERT-v2-FullSong all ship exactly `model.safetensors`. Only ASR is
    sharded, and only ASR must therefore avoid the literal. The rule is "required
    names files the repo actually has", not "required avoids weights".
    """
    for repo_id, required, patterns, weights_glob in CACHED_REPOS:
        assert weights_glob, f"{repo_id} has no weights check"
        for name in required:
            assert name in patterns, f"{repo_id}: {name!r} is required but never downloaded"

    # The sharded repo must not name a single-file spelling.
    asr = next(spec for spec in CACHED_REPOS if spec[0] == ASR_REPO)
    assert not any(name.endswith(".safetensors") for name in asr[1]), (
        f"{ASR_REPO} is sharded; requiring a single-file spelling is what failed a "
        f"production boot. Its required list is {asr[1]}."
    )


# =============================================================================
# `trust_remote_code` repos need their Python, not just their weights
# =============================================================================
#
# The bug, from a live cover job (2026-09-18):
#
#     cover mode: melody transcription failed — ... it looks like
#     m-a-p/SheetSage2 is not the path to a directory containing a file named
#     configuration_sheetsage2.py
#
# `config.json` and the weights resolved; the *model code* did not. SheetSage2
# and MERT are loaded with `trust_remote_code=True`, so transformers imports
# `configuration_*.py`, `modeling_*.py` and every sibling those reach by relative
# import. A pattern list of "config + weights" caches a repo that cannot load.
#
# The probe in `transcribe_sheetsage/` had `*.py` in its allow_patterns from the
# start — the cache config was the one that omitted it.

#: Repos whose `config.json` declares `auto_map`, meaning `trust_remote_code`
#: will import their Python. Verified against the Hub on 2026-09-18.
TRUST_REMOTE_CODE_REPOS = frozenset({"m-a-p/SheetSage2", "m-a-p/MERT-v2-FullSong"})


def test_trust_remote_code_repos_download_their_python() -> None:
    """A `*.py` pattern is mandatory for these, and must not be trimmed later.

    Without it the download succeeds, the presence check passes, and the failure
    surfaces at load time inside a GPU job — as a message about a missing
    `configuration_*.py`, which reads like a cache miss rather than a pattern
    omission.
    """
    for repo_id, _required, patterns, _glob in CACHED_REPOS:
        if repo_id in TRUST_REMOTE_CODE_REPOS:
            assert "*.py" in patterns, (
                f"{repo_id} uses trust_remote_code, so its model code must be downloaded. patterns={patterns}"
            )


def test_trust_remote_code_repos_require_their_code_entry_points() -> None:
    """Presence must include the files transformers actually imports.

    A repo holding only `config.json` and weights would pass a weights-only
    check and then fail at load. Requiring the two entry points makes the check
    fail at *cache* time, where the error names the repo and the file.
    """
    for repo_id, required, _patterns, _glob in CACHED_REPOS:
        if repo_id not in TRUST_REMOTE_CODE_REPOS:
            continue
        code = [name for name in required if name.endswith(".py")]
        assert code, f"{repo_id} requires no .py file, so remote code is unchecked"
        assert any("configuration" in name for name in code), f"{repo_id}: no configuration_*.py required"
        assert any("modeling" in name for name in code), f"{repo_id}: no modeling_*.py required"


def test_a_repo_needing_only_weights_is_unaffected() -> None:
    """YuE2 and the VAE are loaded from the wheel, not from remote code.

    Their patterns stay minimal on purpose — the wheel supplies the Python, so
    pulling the repo's source would be dead weight on every cold start.
    """
    for repo_id, _required, patterns, _glob in CACHED_REPOS:
        if repo_id in TRUST_REMOTE_CODE_REPOS or repo_id == ASR_REPO:
            continue
        assert "*.py" not in patterns or repo_id == MODEL_REPO, repo_id


# =============================================================================
# An index is not weights
# =============================================================================
#
# The bug, from a live cover job on image v20 (2026-09-18):
#
#     cover mode: lyric transcription failed — Qwen/Qwen3-ASR-1.7B does not
#     appear to have files named ('model-00001-of-00002.safetensors',
#     'model-00002-of-00002.safetensors')
#
# The cache held `model.safetensors.index.json` and none of the shards it
# names — the volume had been filled by an earlier image whose ASR patterns
# lacked `*.safetensors`. The weights check accepted the index as proof of
# weights, so boot verified a broken cache and the failure surfaced after 51 s
# of real work inside a GPU job.
#
# `model.safetensors.index.json` is a *manifest*: a map from tensor names to
# shard filenames. It is not a weight file, and nothing about its presence says
# the shards exist.


def test_an_index_alone_does_not_satisfy_the_weights_check(tmp_path: Path) -> None:
    """The exact shape of the broken cache: config + index, no shards."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}', encoding="utf-8")

    missing = boot._missing_files(tmp_path, ("config.json",), boot.WEIGHTS_GLOB)

    assert missing, (
        "an index with no shards passed the weights check — this is the bug that "
        "let a shard-less cache boot and fail 51 s into a job"
    )
    assert boot.WEIGHTS_GLOB in missing


def test_the_weights_glob_is_a_glob_not_a_filename() -> None:
    """A named spelling cannot cover both layouts; a glob can."""
    assert boot.WEIGHTS_GLOB.startswith("*"), boot.WEIGHTS_GLOB
    assert boot.WEIGHTS_GLOB.endswith(".safetensors")


def test_no_repo_treats_the_index_as_weights() -> None:
    """And the index is not smuggled in through `required` instead.

    A sharded repo genuinely needs its index — but as a *separate* requirement,
    not as the thing standing in for weights. The distinction is what the check
    depends on.
    """
    for repo_id, required, _patterns, weights_glob in CACHED_REPOS:
        assert weights_glob == boot.WEIGHTS_GLOB, repo_id
        assert "model.safetensors.index.json" not in weights_glob
        # Where the index IS required, it must be for a repo that ships shards.
        if "model.safetensors.index.json" in required:
            assert repo_id == ASR_REPO, f"{repo_id} requires an index but is not the sharded repo"
