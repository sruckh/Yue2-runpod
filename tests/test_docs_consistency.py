"""Doc-consistency tests.

The payloads in `worker/.runpod/tests.json` are what a person pastes into the
RunPod console to check a deployment. They are documentation, and documentation
drifts: a validator change can quietly make every example invalid, and the
failure only shows up when someone is already debugging a live endpoint.

These tests read the real file and push every payload through the real
validator, so the examples cannot rot unnoticed.
"""

from __future__ import annotations

import json
import os

import pytest

from config import MODEL_REPO, VAE_REPO, WORKER_DIR
from schema import (
    MissingInputError,
    ValidationError,
    validate_job,
    validate_mode,
)

RUNPOD_DIR = WORKER_DIR / ".runpod"

pytestmark = pytest.mark.skipif(
    not (RUNPOD_DIR / "tests.json").is_file(),
    reason="RunPod example payloads are not present",
)


def _example_payloads() -> list[tuple[str, dict]]:
    spec = json.loads((RUNPOD_DIR / "tests.json").read_text(encoding="utf-8"))
    return [(t["name"], t["input"]) for t in spec["tests"]]


def test_there_is_at_least_one_example() -> None:
    assert _example_payloads(), "tests.json has no examples to check"


@pytest.mark.parametrize(("name", "payload"), _example_payloads(), ids=lambda v: v if isinstance(v, str) else "")
def test_example_payloads_pass_the_real_validator(name: str, payload: dict) -> None:
    """Every shipped example must survive validation unchanged."""
    validate_mode(payload)
    params = validate_job({"id": "example-job", "input": payload})
    assert params.style
    assert params.lyrics
    assert params.cot in {"off", "melody", "full"}


def test_hub_json_matches_the_documented_gpu_tier() -> None:
    """The endpoint config must agree with the locked 24 GB single-GPU decision."""
    hub = json.loads((RUNPOD_DIR / "hub.json").read_text(encoding="utf-8"))
    config = hub["config"]
    assert hub["type"] == "serverless"
    assert config["gpuCount"] == 1
    assert config["gpuIds"] == "ADA_24", "the locked tier is a 24 GB RTX 4090-class card"
    # The built image measures 12.9 GB (nvidia CUDA wheels + torch), so the
    # container disk must clear that with room for the pull/unpack phase. 20 GB
    # was the original value and left too little headroom.
    assert config["containerDiskInGb"] >= 25, "container disk must clear the 12.9 GB image with headroom"


def test_model_repos_named_in_docs_exist_as_constants() -> None:
    """The README and .env.example name these; config.py is the one home for them."""
    assert MODEL_REPO.count("/") == 1
    assert VAE_REPO.count("/") == 1


def test_all_three_modes_are_accepted_by_the_validator() -> None:
    """Stage 02 was create-only; Stage 03 added cover and edit.

    Behavioural rather than textual: asserts what the validator *does*, not what
    a docstring says. An earlier version of this checked module docstrings for a
    keyword, which tested prose rather than behaviour and failed on modules that
    legitimately never mention the word.
    """
    for mode in ("create", "cover", "edit"):
        assert validate_mode({"mode": mode}) == mode


def test_env_example_documents_every_mandatory_variable() -> None:
    """A required variable missing from .env.example is a failed first deploy."""
    example = (WORKER_DIR / ".env.example").read_text(encoding="utf-8")
    for required in ("B2_ENDPOINT_URL", "B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET", "VOLUME_ROOT"):
        assert required in example, f"{required} is required but undocumented in .env.example"


def test_dockerfile_hf_home_matches_the_configured_cache_path() -> None:
    """The image's default HF path must equal what the code computes.

    These are two independent declarations of the same location — one in the
    Dockerfile, one derived from `VOLUME_ROOT` at runtime — and nothing at build
    time compares them. That gap is not hypothetical: the Dockerfile carried
    `/runpod-volume/hf` while the code used a different path, and it went
    unnoticed because runtime always overwrote it. A divergence only bites when
    something writes to the cache *before* `apply_hf_env()` runs, and then it
    fills the container disk instead of the volume.
    """
    import re
    from pathlib import Path

    import config

    dockerfile = (Path(config.WORKER_DIR).parent / "Dockerfile").read_text(encoding="utf-8")
    # `HF_HOME=` sits on a continuation line inside the ENV instruction, so it
    # is indented rather than at column zero.
    match = re.search(r"^\s*HF_HOME=(\S+)\s*$", dockerfile, re.M)
    assert match, "Dockerfile declares no HF_HOME default"

    os.environ.pop("VOLUME_ROOT", None)
    expected = str(config.CacheConfig().hf_home)
    assert match.group(1) == expected, (
        f"Dockerfile HF_HOME={match.group(1)!r} but CacheConfig.hf_home={expected!r} — they must agree"
    )


def test_invalid_payload_shape_is_rejected_not_crashed() -> None:
    """Guards the error type contract the docs promise."""
    for bad in ({}, {"input": {}}, {"input": {"style": "x"}}):
        with pytest.raises((ValidationError, MissingInputError)):
            validate_job(bad)
