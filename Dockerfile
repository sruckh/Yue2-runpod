# YuE2 RunPod Serverless worker — create, cover and edit modes.
#
# Base image: `python:3.11-slim-trixie`, deliberately.
#
#   3.11, not 3.12, because cover mode's SheetSage2 environment pins
#   numpy==1.24.3, which predates Python 3.12 and publishes no cp312 wheels —
#   see worker/transcribe_sheetsage/requirements.txt. Every pin YuE2 needs has
#   cp311 wheels (torch 2.10.0 on both cu128 and cu126, triton 3.6.0,
#   tiktoken 0.12.0, numpy 2.2.6), so one interpreter serves all three
#   environments.
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
# `qwen-asr` pulls `triton` into its dependency tree. Triton JIT-compiles CUDA
# kernels at runtime and needs a host C compiler to do it, so `cover` failed on
# real hardware with:
#
#     lyric transcription failed — Failed to find C compiler.
#     Please specify via CC environment variable or set triton.knobs.build.impl.
#
# Create mode never touched that path, which is why the slim base looked
# sufficient for as long as it did. The compiler is now installed for the
# interpreter that needs it, rather than reasoned away.

FROM python:3.11-slim-trixie

# libsndfile1 is a hard runtime dependency of `soundfile`, which the pipeline
# uses to write the 48 kHz FLAC. Without it `import soundfile` raises at import
# time — inside the handler, not at build, so it fails on the first job rather
# than at image build.
#
# `gcc`/`g++` are a runtime dependency of triton, which JIT-compiles kernels and
# shells out to a C compiler to build its launcher. They are needed by the
# cover/edit path only (`qwen-asr` -> triton), but the compiler is a system
# binary, so one install serves all three environments and the same gap cannot
# reappear in the sheetsage2 venv later. `--no-install-recommends` still
# applies: this is a compiler for one build step, not a development toolchain.
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

# --- isolated model-family environments (cover mode) --------------------------
#
# The cover path runs two more model families, and their dependencies cannot
# coexist with YuE2's or with each other:
#
#   YuE2 (main, /app)  torch 2.10.0  transformers 4.57.6  numpy 2.2.6
#   SheetSage2         torch 2.8.0   transformers 4.45.2  numpy 1.24.3
#   Qwen3-ASR          torch (unpinned)  transformers 4.57.6  accelerate 1.12.0
#
# So each gets its own venv, invoked as a subprocess. The second reason is VRAM:
# a subprocess exit returns its memory to the driver, which is what keeps a
# cover job's peak at YuE2's own ceiling rather than the sum of three models.
#
# All three are Python 3.11 — the base interpreter — because that is what
# SheetSage2's numpy pin requires. The isolation is about torch/numpy versions,
# not about the interpreter.
#
# Built here, in the image, and nowhere else — never on a dev box.

# Requirements are vendored in the repository, not fetched from a URL at build
# time: a fetched file makes the image depend on whatever is served that day.
COPY worker/transcribe_sheetsage/requirements.txt /tmp/sheetsage-requirements.txt

# torch comes from the CUDA 12.6 index, and goes in *before* the requirements
# file. That file pins `torch==2.8.0` with no index, so letting it resolve from
# PyPI would install the CPU-only build. Installed here first, those pins are
# already satisfied and pip skips them.
#
# NOTE: no comments inside the RUN chain below. Docker joins continued lines into
# one shell command *before* running it, so a `#` mid-chain comments out
# everything after it — including the following `&&`. Keeping the explanation out
# here is correctness, not style.
RUN python -m venv /opt/venvs/sheetsage2 \
    && /opt/venvs/sheetsage2/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/venvs/sheetsage2/bin/python -m pip install --no-cache-dir "huggingface-hub==0.36.0" \
    && /opt/venvs/sheetsage2/bin/python -m pip install --no-cache-dir \
         torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 \
    && /opt/venvs/sheetsage2/bin/python -m pip install --no-cache-dir -r /tmp/sheetsage-requirements.txt \
    && rm -f /tmp/sheetsage-requirements.txt

# Qwen3-ASR: the transformers backend, not vLLM. There is no concurrency need at
# one-job-at-a-time, and the `vllm` extra would pull a second torch.
RUN python -m venv /opt/venvs/qwen3-asr \
    && /opt/venvs/qwen3-asr/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/venvs/qwen3-asr/bin/python -m pip install --no-cache-dir "qwen-asr==0.0.6"

# --- verify all three environments -------------------------------------------------
#
# These checks assert; they do not print. The previous versions here did
# `print(torch.__version__)`, which reports the version without comparing it to
# anything — so a venv that resolved the wrong torch printed the wrong number and
# the build passed. The version pin is the entire reason that venv exists, so a
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

# SheetSage2's environment, against its own vendored pins — this is where the
# torch 2.8.0 pin lives, and where a mis-resolution is most likely.
COPY worker/transcribe_sheetsage/requirements.txt /tmp/sheetsage-requirements.txt
RUN /opt/venvs/sheetsage2/bin/python /tmp/check_env.py --requirements /tmp/sheetsage-requirements.txt

# The Qwen3-ASR environment. It declares no torch pin, so there is nothing to
# assert about which torch it resolved — the checker reports it. What *is*
# asserted is that the package imports, which is the part a pin can speak to.
RUN /opt/venvs/qwen3-asr/bin/python -c "import qwen_asr" && echo "qwen3-asr: qwen_asr imports"

# A single summary of all three environments, so the build log states plainly
# what it produced instead of leaving it scattered across three pip transcripts.
RUN echo "=== installed environments ===" \
    && python -c "import sys, importlib.metadata as m; print('  main       py', sys.version.split()[0], '| torch', m.version('torch'), '| transformers', m.version('transformers'), '| yue2-infer', m.version('yue2-infer'))" \
    && /opt/venvs/sheetsage2/bin/python -c "import sys, importlib.metadata as m; print('  sheetsage2 py', sys.version.split()[0], '| torch', m.version('torch'), '| transformers', m.version('transformers'), '| numpy', m.version('numpy'))" \
    && /opt/venvs/qwen3-asr/bin/python -c "import sys, importlib.metadata as m; print('  qwen3-asr  py', sys.version.split()[0], '| torch', m.version('torch'), '| transformers', m.version('transformers'))" \
    && echo "=============================="

# --- the unification experiment -----------------------------------------------------
#
# Does SheetSage2's code load under YuE2's stack instead of its own pins? If it
# does, the three environments can collapse into one and the image loses two
# torch installs. If it does not, the split is required and we will know exactly
# which import broke.
#
# This RUN is deliberately NON-FATAL. It reports; it does not gate. The split
# stack is what ships today, so a failure here is information rather than a
# build error — and making it fatal before we know the answer would mean a red
# build that says nothing about whether the *product* is broken.
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
RUN python /tmp/probe_unified_stack.py || echo "probe: unified stack NOT viable (see VERDICT above)"

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
