"""Static checks on the Dockerfile.

These exist because the image is built on RunPod's platform, not here, and a
failed build costs a ten-minute round trip with no local iteration. Every rule
below is one that has either already broken a build or would break one silently.

Two kinds of defect are worth catching statically:

- **A build that fails.** A `COPY` whose source is excluded by `.dockerignore`,
  a comment inside a `RUN` continuation, an apt package that no longer exists.
- **A build that succeeds wrongly.** A venv path the code does not look in, a
  base image allowed to drift, a guard that was meant to run and does not.

The comment-in-RUN rule is the sharpest example: Docker joins continued lines
into a single shell command *before* executing it, so a `#` mid-chain comments
out every following `&&` — the step "succeeds" having skipped half its work.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"

sys.path.insert(0, str(REPO_ROOT / "worker"))


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def joined_runs(dockerfile: str) -> list[str]:
    """Each RUN instruction with its continuations joined — as sh receives it."""
    runs: list[str] = []
    current: list[str] = []
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if stripped.startswith("RUN "):
            current = [stripped[4:].rstrip("\\").strip()]
        elif (current and current[-1].rstrip().endswith("\\")) or (current and line.rstrip().endswith("\\")):
            current.append(stripped.rstrip("\\").strip())
        elif current:
            runs.append(" ".join(p for p in current if p))
            current = []
    if current:
        runs.append(" ".join(p for p in current if p))
    return runs


# =============================================================================
# A build that fails
# =============================================================================


def test_no_comments_inside_run_chains(dockerfile: str) -> None:
    """A `#` mid-chain comments out the rest of the joined command.

    Docker joins the continued lines into one shell command before running it,
    so this:

        RUN a \\
            # note
            && b

    becomes `a # note && b`, and `b` never runs. The step reports success having
    done half its work — which is worse than failing.
    """
    in_run = False
    offenders: list[int] = []
    for number, line in enumerate(dockerfile.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("RUN ") or stripped == "RUN":
            in_run = stripped.endswith("\\")
            continue
        if in_run:
            if stripped.startswith("#"):
                offenders.append(number)
            if not stripped.endswith("\\"):
                in_run = False
    assert not offenders, f"comment inside a RUN continuation at line(s) {offenders}"


def test_every_run_is_valid_shell(joined_runs: list[str]) -> None:
    """`bash -n` each joined RUN. Catches quoting and continuation mistakes."""
    broken = []
    for command in joined_runs:
        if command.startswith("--mount") or command.startswith("["):
            continue  # exec-form or BuildKit mount: not shell
        result = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
        if result.returncode != 0:
            broken.append((command[:70], result.stderr.strip()[:120]))
    assert not broken, f"RUN lines with shell syntax errors: {broken}"


def test_copy_sources_exist_and_are_not_ignored(dockerfile: str) -> None:
    """A COPY of a file `.dockerignore` excludes fails the build.

    `worker/AGENTS.md` was exactly this shape once: referenced in the Dockerfile
    and unpublished from the repository at the same time.
    """
    patterns = [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    problems = []
    for source in re.findall(r"^COPY\s+(?:--from=\S+\s+)?(\S+)", dockerfile, re.M):
        if source.startswith("/"):
            continue  # copied from another build stage
        path = REPO_ROOT / source
        if not path.exists():
            problems.append(f"{source}: does not exist")
            continue
        excluded = [p for p in patterns if p == source or p.rstrip("/") == source]
        negated = [p for p in patterns if p.startswith("!") and source.endswith(p[1:])]
        if excluded and not negated:
            problems.append(f"{source}: excluded by .dockerignore")
    assert not problems, "; ".join(problems)


def test_base_image_pins_a_debian_release(dockerfile: str) -> None:
    """The bare `python:3.11-slim` tag is a moving target.

    It moved once: `python:3.12-slim` resolved to Debian trixie, which has no
    `python3.11` package, and a build that installed one failed. Naming the
    release makes the base reproducible.
    """
    from_lines = re.findall(r"^FROM\s+(\S+)", dockerfile, re.M)
    assert from_lines, "no FROM"
    base = from_lines[-1]
    assert re.search(r"-(bookworm|trixie|bullseye|noble|jammy)$", base), (
        f"base image {base!r} does not pin a distribution release"
    )


def test_no_apt_package_that_the_base_cannot_have(dockerfile: str) -> None:
    """Guard against re-adding `apt-get install python3.11`.

    The base interpreter is already the version every environment needs, so no
    Python is installed via apt. Pinning one that the distribution dropped is
    what failed the build.
    """
    installed = " ".join(re.findall(r"apt-get install[^&|]*", dockerfile))
    stray = re.findall(r"python3\.\d+", installed)
    assert not stray, f"apt-installed Python {stray} — the base image already provides the interpreter"


# =============================================================================
# A build that succeeds wrongly
# =============================================================================


def test_venv_paths_match_what_the_code_looks_for(dockerfile: str) -> None:
    """The Dockerfile builds venvs where `subprocess_runner` looks for them.

    Nothing at runtime checks this: a mismatch means `cover` fails with
    "interpreter not found" on the first job, after a successful build.
    """
    import subprocess_runner

    built = set(re.findall(r"python -m venv (/opt/venvs/\S+)", dockerfile))
    assert built, "no venvs are built"
    root = str(subprocess_runner.VENV_ROOT)
    assert root == "/opt/venvs", f"VENV_ROOT is {root}, but the image builds into /opt/venvs"

    # And the names the runner uses are among the ones built.
    import modes

    for name in (modes.SHEETSAGE_VENV, modes.ASR_VENV):
        assert f"/opt/venvs/{name}" in built, f"{name} is not created by the Dockerfile"


def test_child_entrypoints_are_copied_into_the_image(dockerfile: str) -> None:
    """Both subprocess entrypoints must reach /app, or every cover job fails."""
    assert re.search(r"^COPY worker/ /app/", dockerfile, re.M), "worker/ is not copied"
    for relative in ("transcribe_sheetsage/run.py", "transcribe_asr/run.py"):
        assert (REPO_ROOT / "worker" / relative).is_file(), f"{relative} is missing"


def test_the_weight_guard_is_present(dockerfile: str) -> None:
    """Locked decision 4, enforced at build time rather than by convention.

    Convention failed once already: moving the boot to module scope made the
    old `import handler` smoke test hydrate the volume inside a layer and bake
    ~12 GB of weights into the image.
    """
    assert "safetensors" in dockerfile and "-size +50M" in dockerfile, "the weight guard is gone"
    assert "py_compile" in dockerfile, "the non-executing syntax check is gone"


def test_the_dockerfile_never_imports_the_handler(dockerfile: str) -> None:
    """`import handler` boots the pipeline and starts the SDK.

    At build time that downloads weights into a layer; it must stay `py_compile`.
    """
    for line in dockerfile.splitlines():
        if line.strip().startswith("RUN") and "import" in line:
            assert "import handler" not in line, f"imports handler at build: {line.strip()[:80]}"


def test_sheetsage_requirements_are_vendored_not_fetched(dockerfile: str) -> None:
    """A build-time URL fetch makes the image depend on what is served that day.

    The file is in the repository now; a `curl` of it would silently reintroduce
    the dependency.
    """
    assert (REPO_ROOT / "worker" / "transcribe_sheetsage" / "requirements.txt").is_file(), (
        "SheetSage2's requirements are no longer vendored"
    )
    assert "huggingface.co/m-a-p/SheetSage2/resolve" not in dockerfile, (
        "the requirements are fetched from a URL at build time again"
    )


# =============================================================================
# The compiler triton needs at runtime
# =============================================================================
#
# A live cover job failed 50 s in with:
#
#     lyric transcription failed — Failed to find C compiler.
#     Please specify via CC environment variable or set triton.knobs.build.impl.
#
# Triton JIT-compiles CUDA kernels and needs a host C compiler. The image
# installed only `libsndfile1 ffmpeg`, because the Dockerfile's own reasoning
# about `triton` — that it appears only in the `yue2/fast.py` vLLM extra — was
# correct for YuE2 and had been generalised to the whole image. `qwen-asr` pulls
# triton into its own dependency tree.


def test_a_c_compiler_is_installed(dockerfile: str) -> None:
    """`gcc` must be in the apt install line, not merely assumed.

    A comment in this Dockerfile once argued the slim base was sufficient
    *because* nothing in YuE2 needs triton. That argument was about one of the
    three environments.
    """
    installs = [
        line for line in dockerfile.splitlines() if "apt-get install" in line and not line.strip().startswith("#")
    ]
    assert installs, "no apt-get install line found"
    joined = " ".join(installs)
    assert "gcc" in joined, (
        "gcc is not installed. Triton JIT-compiles kernels at runtime and shells "
        "out to a C compiler; without one, cover/edit fail inside a GPU job."
    )


def test_the_compiler_is_verified_at_build_time(dockerfile: str) -> None:
    """A mistyped package name must fail the build, not the job.

    The failure this guards against costs a cold start, ~50 s of GPU time, and a
    round trip through the build system to discover.
    """
    assert "gcc --version" in dockerfile, "the compiler is installed but never checked"


def test_the_compiler_env_vars_are_set(dockerfile: str) -> None:
    """Triton's own error message asks for `CC`.

    `gcc` would be found via the default PATH that `exec` falls back to, but the
    subprocess runner hands its children a curated environment, so naming it
    removes the question.
    """
    assert "CC=gcc" in dockerfile
    assert "CXX=g++" in dockerfile


def test_the_env_assertion_runs_after_the_env_it_checks(dockerfile: str) -> None:
    """The assertion must follow the `ENV` instruction, not precede it.

    This was a real build failure: the check was placed before the `ENV`, so it
    compared against variables that had not been set yet and the image failed to
    build. The check was right about the question and wrong about its position —
    and nothing but the build would have said so.
    """
    env_at = dockerfile.index("ENV PYTHONUNBUFFERED")
    check_at = dockerfile.index('test "$PYTHONUNBUFFERED"')
    assert env_at < check_at, "the ENV assertion appears before the ENV instruction it verifies, so it can only fail"


def test_env_variables_are_asserted_not_assumed(dockerfile: str) -> None:
    """The ENV block must be proven to have parsed as written.

    The block previously carried comment lines *inside* its continuation,
    including between `HF_HUB_DISABLE_TELEMETRY=1` and `HF_HOME=...`. Whether
    Docker strips such comments before joining or joins first — making `#`
    swallow the rest of the instruction — could not be settled by reading, and
    `config.CacheConfig` sets the same paths at runtime, so nothing failed loudly
    enough to reveal which had happened. The comments are hoisted out and the
    values are now asserted at build time.
    """
    assert "env verified" in dockerfile, "the ENV block is never verified"
    # And the block itself must be comment-free.
    env_block = dockerfile[dockerfile.index("ENV PYTHONUNBUFFERED") :]
    env_block = env_block[: env_block.index("\n\n")]
    for line in env_block.splitlines():
        assert not line.strip().startswith("#"), (
            f"a comment sits inside the ENV continuation: {line!r}. Docker's handling "
            "of that is exactly what this check exists because we could not confirm."
        )
