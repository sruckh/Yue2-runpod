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
# not install. The HF path goes through PyTorch's own SDPA. That is why the
# slim base is sufficient and why there is no multi-hour attention build here.

FROM python:3.11-slim-trixie

# libsndfile1 is a hard runtime dependency of `soundfile`, which the pipeline
# uses to write the 48 kHz FLAC. Without it `import soundfile` raises at import
# time — inside the handler, not at build, so it fails on the first job rather
# than at image build.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Point HuggingFace at RunPod's cache location. `config.CacheConfig` derives
    # the same paths from VOLUME_ROOT at runtime and is authoritative; this is a
    # belt-and-braces default so that even a step running before `apply_hf_env()`
    # writes to the volume rather than filling the container disk.
    #
    # Keep in sync with CacheConfig.hf_home. The build cannot check that, and a
    # divergence would be silent — which is exactly how the previous value
    # (/runpod-volume/hf) sat here unnoticed while the code used a different path.
    HF_HOME=/runpod-volume/huggingface-cache

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

# Each child environment must be able to import its own model library. This
# fails the build rather than the first cover job, and it is the only check of
# these environments that can run without a GPU.
RUN /opt/venvs/sheetsage2/bin/python -c "import torch, transformers; print('sheetsage2', torch.__version__, transformers.__version__)" \
    && /opt/venvs/qwen3-asr/bin/python -c "import torch, qwen_asr; print('qwen3-asr', torch.__version__)"

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
