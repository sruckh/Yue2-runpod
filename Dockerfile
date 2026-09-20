# YuE2 RunPod Serverless worker — create, cover and edit modes.
#
# Base image: `python:3.11-slim-trixie`.
#
#   3.11 is now a **conservative default, not a requirement.** It was originally
#   forced: cover mode had its own SheetSage2 environment pinning numpy==1.24.3,
#   which predates Python 3.12 and publishes no cp312 wheels. That environment is
#   gone — both model families run in this one, on numpy 2.2.6 — so nothing in the
#   image needs 3.11 any more. Moving to 3.12 would work on paper (every current
#   pin has cp312 wheels) but has not been built or run on hardware, so the base
#   stays where it is until someone does that deliberately.
#
#   `-trixie` is pinned explicitly because the bare `python:3.11-slim` tag is a
#   moving target. It moved once already: `python:3.12-slim` resolved to Debian
#   trixie, which dropped the `python3.11` apt package, and a build that tried
#   to install it there failed. Naming the Debian release makes the base
#   reproducible.
#
# No weights, no vLLM, no flash-attn compile step.
#
# On flash-attn specifically: the YuE2 model card says the HF package "uses
# PyTorch, CUDA graphs, and FlashAttention", which reads like a build
# requirement. It is not one — grepping the `yue2_infer-0.1.5` wheel source
# finds no `flash_attn` import at all, and `triton` appears only inside
# `yue2/fast.py`, which is the separate `fast` (vLLM) extra we deliberately do
# not install. The HF path goes through PyTorch's own SDPA.
#
# **That reasoning was correct for YuE2 and was wrongly generalised to the
# whole image.** It says nothing about the other two environments, and
# `qwen-asr` pulls `triton` into its dependency tree. So `cover` failed on real
# hardware with:
#
#     lyric transcription failed — Failed to find C compiler.
#     Please specify via CC environment variable or set triton.knobs.build.impl.
#
# **What needs the compiler, precisely** — because "triton needs a C compiler"
# is the wrong summary. Triton compiles GPU kernels with the LLVM bundled in its
# wheel; no host compiler is involved in that. What it builds at runtime is the
# host-side *launcher*: `third_party/nvidia/backend/driver.py` calls
# `compile_module_from_file(src_path=.../"driver.c", name="cuda_utils")` on first
# use, producing a `cuda_utils.so` that wraps `cuModuleLoad`/`cuLaunchKernel`.
# `triton/runtime/build.py::_find_compiler` looks for `$CC`, then `clang`, then
# `gcc`, and raises the message above when all three are absent.
#
# That is also why the official installation page lists no C compiler: the wheel
# is binary and `pip install triton` needs nothing. The requirement is at
# *runtime*, on the launcher-build path, which is why the error names `CC`.
#
# `triton.knobs.build.impl` is not an escape hatch — it is `Optional[BuildImpl]
# = None`, a Python callable you would have to supply yourself to replace the
# entire build step. Without writing our own build backend, `_find_compiler` is
# the only path, so a compiler is genuinely required.
#
# Create mode never touched that path, which is why the slim base looked
# sufficient for as long as it did. The compiler is now installed rather than
# reasoned away.

FROM python:3.11-slim-trixie

# libsndfile1 is a hard runtime dependency of `soundfile`, which the pipeline
# uses to write the 48 kHz FLAC. Without it `import soundfile` raises at import
# time — inside the handler, not at build, so it fails on the first job rather
# than at image build.
#
# `gcc` builds triton's runtime launcher, as set out above. `g++` is along for
# the C++ path in the same module: `_find_compiler("c++")` handles `.cc/.cpp`
# sources, and although the NVIDIA backend's `driver.c` is C today, a backend
# that ships C++ would otherwise reintroduce this failure in an environment we
# are not looking at. Both are system binaries, so one install serves all three
# environments.
#
# `--no-install-recommends` still applies. This is a compiler for one build step,
# not a development toolchain, and it costs ~1 s per container: the result is
# cached under `TRITON_HOME`, but containers are ephemeral, so each cold start
# rebuilds `cuda_utils.so` once.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libsndfile1 ffmpeg gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# `CC`/`CXX` name the compiler for triton, whose own error message asks for
# exactly this. `gcc` would be found anyway via the default PATH that `exec`
# falls back to when an environment lacks one, but `subprocess_runner` hands the
# children a curated environment and naming it removes the question entirely.
#
# Verified below rather than assumed: a mistyped package name would otherwise
# surface as the same runtime failure this change exists to fix.
# The compiler triton needs must actually be present and runnable. This is a
# build-time assertion on purpose: the failure it guards against is a JIT
# compile inside a GPU job 50 seconds in, which is both slower to discover and
# more expensive than a red build here.
RUN gcc --version > /dev/null && g++ --version > /dev/null \
    && echo "compiler present: $(gcc -dumpversion)" \
    && python -c "import shutil,sys; sys.exit(0 if shutil.which('gcc') else 'gcc not on PATH')"


ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CC=gcc \
    CXX=g++ \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/runpod-volume/huggingface-cache

# Every variable the ENV block just above claims to set must actually be set —
# which is why this RUN sits *after* the ENV and not before it. The first
# version of this check was placed before the ENV instruction, so it failed on
# variables that had not been set yet: a check that was right about the question
# and wrong about its own position in the file.
#
# `HF_HOME` is the discriminator for the comment question: if Docker had joined
# the comment lines into the instruction rather than stripping them, the `#` would
# have swallowed everything after it and `HF_HOME` would be unset. Plain shell,
# no clever quoting — the first version of this check was a Python one-liner that
# `bash -n` rejected, which is the test suite doing its job.
RUN test "$PYTHONUNBUFFERED" = "1" \
    && test "$CC" = "gcc" \
    && test "$CXX" = "g++" \
    && test "$HF_HUB_DISABLE_TELEMETRY" = "1" \
    && test "$HF_HOME" = "/runpod-volume/huggingface-cache" \
    && echo "env verified: CC=$CC CXX=$CXX HF_HOME=$HF_HOME"

WORKDIR /app

# Dependencies first, so the ~2.5 GB torch layer is cached across code changes.
COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# The `yue2_infer` wheel ships inside the model repo rather than on PyPI, so it
# is fetched here. This is code, not weights — the ~12 GB of safetensors stay on
# the network volume and are never part of the image (locked decision 4).
#
# Installed with --no-deps: requirements.txt above already carries every
# `Requires-Dist` entry at the exact version the wheel asks for, and letting the
# resolver run here would let it re-pick torch from PyPI's CPU index.
ARG YUE2_WHEEL=yue2_infer-0.1.5-py3-none-any.whl
RUN pip install --no-cache-dir --upgrade "huggingface-hub==0.36.2" \
    && python -c "from huggingface_hub import hf_hub_download; hf_hub_download('m-a-p/YuE2-3B', '${YUE2_WHEEL}', local_dir='/tmp/wheel')" \
    && pip install --no-cache-dir --no-deps "/tmp/wheel/${YUE2_WHEEL}" \
    && rm -rf /tmp/wheel

# `qwen-asr` into the **main** environment.
#
# It was originally installed here so the unification experiment could ask whether
# it *could* run on the main stack. It now simply does — there is no other stack.
#
# `--no-deps` is load-bearing, not a shortcut. qwen-asr pins
# `accelerate==1.12.0` and the YuE2 wheel pins `accelerate==1.13.0`; two exact
# pins cannot both hold, and a resolver would either fail or quietly downgrade
# accelerate under YuE2. Neither package imports accelerate (zero references in
# either wheel), so the conflict is between two decorative pins — blocked on
# nothing — and --no-deps sidesteps it. Everything qwen_asr actually imports is
# already present above or pinned in worker/requirements.txt.
#
# Its declared `gradio`, `flask` and `vllm` are deliberately absent: they are the
# CLI and the optional vllm backend, and the one inference-path vllm import sits
# inside a `try/except`.
#
# Asserted here rather than by `check_env` reading a pins file, because it is
# installed outside that file on purpose — so nothing else would notice a wrong
# version. This is the check that would have caught the build failure that put
# `qwen-asr` in `requirements.txt`: it names the version, so a change to it fails
# here instead of silently resolving something else.
RUN pip install --no-cache-dir --no-deps "qwen-asr==0.0.6" \
    && python -c "import qwen_asr, importlib.metadata as m, sys; sys.exit(0 if m.version('qwen-asr') == '0.0.6' else 'expected 0.0.6')" \
    && python -c "import qwen_asr; print('qwen_asr importable in the main environment')"

# --- the subprocess boundary (cover mode) -------------------------------------
#
# Cover runs two more model families beside YuE2's. **They share this one
# environment** — the isolation that matters is the *process*, not the venv:
#
#   a subprocess exit returns its memory to the driver, which is what keeps a
#   cover job's peak at YuE2's own ceiling rather than the sum of three models.
#
# Both families were originally given virtual environments, on the assumption
# their pins could not coexist:
#
#   YuE2 (main)   torch 2.10.0  transformers 4.57.6  numpy 2.2.6
#   SheetSage2    torch 2.8.0   transformers 4.45.2  numpy 1.24.3
#   Qwen3-ASR     torch (unpinned, resolved 2.14.0)  accelerate 1.12.0
#
# The unification experiment showed the assumption was never tested: both
# families load and produce correct output on the main stack. BOTH VENVS WERE
# REMOVED 2026-09-19, taking ~6.5 GB of duplicated torch with them. See "Why
# there is one environment now" below.
#
# The `venv_name` parameter in `subprocess_runner` remains, so a future family
# whose pins genuinely conflict can be given its own environment without a
# redesign. Nothing uses it today.
#
# Anything built here is built in the image, on RunPod's platform — never on a
# dev box. Requirements are vendored in the repository, not fetched from a URL at
# build time: a fetched file makes the image depend on whatever is served that day.

# --- verify the environment ---------------------------------------------------------
#
# These checks assert; they do not print. The previous versions here did
# `print(torch.__version__)`, which reports the version without comparing it to
# anything — so a wheel that resolved the wrong torch printed the wrong number and
# the build passed. The version pin is the entire reason the check exists, so a
# check that cannot fail on it is not a check.
#
# `check_env.py` reads the pins from the requirements files rather than restating
# them, and exits non-zero on any mismatch. It also imports each package, which
# is what catches a wheel that installed cleanly and cannot load — the failure a
# missing system library produces.

# The main environment, against worker/requirements.txt.
COPY worker/requirements.txt /tmp/main-requirements.txt
COPY worker/check_env.py /tmp/check_env.py
RUN python /tmp/check_env.py --requirements /tmp/main-requirements.txt

# The two model-family environments no longer exist, so neither has pins to
# assert. What replaced those checks is below: an import of each family's package
# in the ONE environment, which is the claim that now needs holding.
RUN python -c "import qwen_asr; print('qwen_asr imports in the main environment')" \
    && python -c "import mir_eval.chord, pretty_midi, mido; print('SheetSage2 deps import')"

# A single summary of the environment, so the build log states plainly what it
# produced instead of leaving it scattered across pip transcripts.
RUN echo "=== installed environment ===" \
    && python -c "import sys, importlib.metadata as m; print('  main  py', sys.version.split()[0], '| torch', m.version('torch'), '| transformers', m.version('transformers'), '| yue2-infer', m.version('yue2-infer'), '| qwen-asr', m.version('qwen-asr'))" \
    && echo "=============================="

# --- the unified-stack guard --------------------------------------------------------
#
# **The experiment is over; its answer is what ships.** SheetSage2's code loads
# under YuE2's stack — verified on hardware, not just at import — so the image
# carries ONE environment and the probe below is no longer deciding anything.
#
# It stays because the question it asks is now a standing one: *do SheetSage2's
# remote-code modules still import under the torch/transformers this image
# installed?* The model ships `.py` files loaded with `trust_remote_code`, so a
# future transformers bump can break the cover path without breaking anything
# this file otherwise checks. That is the failure this catches, at build time,
# from the Builds tab, before a job pays for it.
#
# This RUN is deliberately NON-FATAL, and that is a considered choice rather than
# leftover caution. The probe fetches SheetSage2's code from the Hub, so making it
# fatal would let a transient network failure or an HF outage — neither of which
# says anything about this image — produce a red build. A build that fails for
# reasons unrelated to the product trains people to ignore red builds.
#
# NOT `--offline`. The first version passed it, and that made the probe useless:
# nothing is cached at this point in the build, so it could never fetch
# SheetSage2's code and reported "not viable" on every run regardless of the
# stack. The build has network — it downloads the model wheel a few steps above —
# so the probe is allowed to use it. Only the SheetSage2 `.py` files come down
# (a few hundred KB), never the weights.
#
# Read the result in the Builds tab: `VERDICT: ...` near the end of this step.
COPY worker/transcribe_sheetsage/probe_unified_stack.py /tmp/probe_unified_stack.py
RUN python /tmp/probe_unified_stack.py || echo "probe: SheetSage2 does NOT import under this stack (see VERDICT above)"

# Handler code last — it changes most often, so it invalidates the least.
COPY worker/ /app/

# Syntax-check every module WITHOUT executing it.
#
# This must not `import handler`. `handler` boots the pipeline at module scope
# (the standard RunPod shape) and instantiates `runpod.serverless.start`, so
# importing it here would hydrate the network volume inside a build layer and
# bake ~12 GB of model weights into the image. That happened during development;
# `py_compile` catches a syntax error just as well and runs nothing.
RUN python -m py_compile /app/*.py && echo "all modules compile"

# Confirm the wheel actually landed and imports (it has no boot side effects).
RUN python -c "import yue2; print('yue2_infer', getattr(yue2, '__version__', 'unknown'))"

# Build-time guard: no *model weights* may be baked into the image.
#
# Enforces locked decision 4, which was previously held by convention only — the
# Dockerfile simply did not copy weights. Convention did not survive an unrelated
# refactor: moving the pipeline boot into module scope made the old `import
# handler` smoke test hydrate the network volume inside a build layer, and ~12 GB
# of weights landed in the image. This assertion would have caught it.
#
# Size-filtered deliberately. The rule is about *weights* — hundreds of MB to
# several GB — not about the extension. A dependency shipping a 2 KB
# `.safetensors` test fixture is not a violation, and failing the build for one
# would train whoever hits it to distrust or delete this check. Only files over
# 50 MB are considered, which no fixture reaches and every real checkpoint
# exceeds by orders of magnitude.
RUN big=$(find / -xdev -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name '*.ckpt' \) \
        -size +50M -not -path '/proc/*' 2>/dev/null); \
    if [ -n "$big" ]; then \
        echo 'ERROR: model weights found in the image.'; \
        echo 'Weights belong on the network volume, never in the image (locked decision 4).'; \
        echo 'Likely cause: a step that imports handler.py, which boots the pipeline at module scope.'; \
        echo 'Use `python -m py_compile` to check the code instead of importing it.'; \
        echo "$big"; \
        exit 1; \
    fi; \
    echo "weight guard: clean (no model weights over 50M in the image)"

# `-u` keeps stdout/stderr unbuffered so RunPod's log stream shows progress
# during a 70-second generation rather than in one burst at the end.
CMD ["python", "-u", "/app/handler.py"]
