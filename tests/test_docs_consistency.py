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
    """Every shipped example must survive validation unchanged.

    The lyrics assertion is per-mode on purpose. It previously read
    `assert params.lyrics` for every payload, which was true while the only
    examples were `create` — and false for a `cover`, whose whole point is that
    ASR supplies the words when the caller does not. The example was right and
    the assertion was stale, so adding a cover example surfaced it.
    """
    mode = validate_mode(payload)
    params = validate_job({"id": "example-job", "input": payload})
    assert params.style
    assert params.cot in {"off", "melody", "full"}

    if mode == "cover":
        # `""` is legitimate here: the transcription fills it. What must hold is
        # that a cover carries *something* for generation to use — either
        # supplied words, or the emptiness that signals "transcribe them".
        assert params.source_audio, "a cover example without source_audio cannot run"
    elif params.instrumental:
        assert params.lyrics, "instrumental fills the lyrics slot rather than leaving it empty"
    else:
        assert params.lyrics, f"{name}: {mode} requires lyrics"


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


# =============================================================================
# README is a contract, so it is checked like one
# =============================================================================
#
# The front-end is built against README.md. It drifted anyway, because nothing
# checked it: `tests.json` was validated, `hub.json` was validated, and the
# README was not. A doc that nothing compares to the code is a doc that describes
# whatever was true when someone last read it.
#
# These tests read the README the way a front-end developer does — as a list of
# field names — and compare it to what the handler actually returns.

README = WORKER_DIR.parent / "README.md"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _response_keys_from_code() -> set[str]:
    """Every key a job response can carry, read from the source.

    Two sources, because they differ and the difference hid a miss. The dict
    `build_response` returns is the bulk; `mode`, `stages` and `vram` are attached
    afterwards by `response["key"] = ...` in `run_create_job`, and an earlier
    version of this scan only read the dict — so `vram`, the field most likely to
    be added without the README, was invisible to the check meant to catch it.

    Parsed rather than called: building a response needs a pipeline and a storage
    client, and the point here is the *shape*.
    """
    import ast

    tree = ast.parse((WORKER_DIR / "handler.py").read_text(encoding="utf-8"))
    keys: set[str] = {
        "artifact_urls",
        "audio_url",
        "cot",
        "decoder",
        "duration",
        "elapsed_seconds",
        "request",
        "sample_rate",
        "score_abc_url",
        "seed",
        "timings",
        "truncated",
        "truncation_by_stage",
    }
    for node in ast.walk(tree):
        # `response["k"] = ...` anywhere in the module.
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "response"
            and isinstance(node.targets[0].slice, ast.Constant)
            and isinstance(node.targets[0].slice.value, str)
        ):
            keys.add(node.targets[0].slice.value)
    return keys


def test_the_readme_documents_every_response_key() -> None:
    """A field the API returns and the README omits is a field nobody will use.

    `mode`, `stages` and `vram` were all added to the response after the README
    was last updated, and none of them appeared in it.
    """
    documented = _readme()
    missing = sorted(k for k in _response_keys_from_code() if k not in documented)
    assert not missing, (
        f"the response carries {missing}, which README.md never mentions. A "
        "front-end built from the README cannot see these fields."
    )

    # A bare substring match is too weak: every one of these words appears
    # elsewhere in the README, so deleting a field's documentation would still
    # pass. Each must appear as a JSON key, which is how a front-end sees it.
    undocumented = sorted(k for k in _response_keys_from_code() if f'"{k}"' not in documented)
    assert not undocumented, (
        f"{undocumented} appear in prose but not as response fields. README.md must "
        "show every key the response carries, as a key."
    )


def test_the_readme_documents_every_request_field() -> None:
    """And the reverse: every field a caller may send must be discoverable.

    `instrumental` was added to the schema and not to the README, so the only way
    to find it was to read `schema.py`.

    The accepted keys are read from the validator's own `raw.get(...)` calls
    rather than from `SongParameters` fields. `mode` is resolved separately and
    never lands on the dataclass, so a fields-based check would both reject it and
    miss anything else handled outside the dataclass.
    """
    import ast

    tree = ast.parse((WORKER_DIR / "schema.py").read_text(encoding="utf-8"))
    accepted: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "raw"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            accepted.add(node.args[0].value)

    assert {"style", "lyrics", "mode", "instrumental", "source_audio"} <= accepted, (
        f"the validator no longer reads an expected field; it reads {sorted(accepted)}"
    )

    documented = _readme()
    missing = sorted(field for field in accepted if field not in documented)
    assert not missing, (
        f"the validator accepts {missing}, which README.md never mentions. A "
        "caller cannot use a field the docs do not name."
    )


def test_the_readme_states_the_per_mode_lyrics_rule() -> None:
    """The rule a front-end gets wrong: `cover` may omit lyrics, `create` may not.

    Stated once here so it stays stated in the README.
    """
    text = _readme()
    assert "required for `create`" in text or "Required for `create`" in text
    assert "cover" in text and "transcrib" in text.lower()


def test_the_readme_does_not_claim_cover_is_unverified() -> None:
    """It said so for most of a day after it stopped being true.

    "Cover and edit are implemented and statically verified ... but have not yet
    run on a GPU" was accurate when written and misleading within hours.
    """
    text = _readme().lower()
    for stale in (
        "have not yet run on a gpu",
        "not yet been run on a gpu",
        "statically verified",
    ):
        assert stale not in text, (
            f"README still says {stale!r}. Cover and edit are verified end to end "
            "on hardware; a stale scope claim sends the front-end after a "
            "non-problem."
        )


def test_the_readme_does_not_promise_more_than_one_cached_model() -> None:
    """RunPod's cached-models feature holds a single repo.

    The README previously told a deployer to "declare" four repos there, which is
    not possible, and omitted the fifth repo that `cover` needs transitively.
    """
    text = _readme()
    assert "cached models are optional and limited to one" in text.lower(), (
        "the README must state the one-cached-model limit; otherwise a deployer "
        "tries to declare several and one is silently ignored"
    )
    # And the transitive parent has to be named, or `cover` boots and then fails.
    assert "MERT-v2-FullSong" in text, "the README omits SheetSage2's encoder parent"
