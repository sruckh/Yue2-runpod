# YuE2 create-mode RunPod Serverless worker.
#
# Shape follows the Phase 1 contract in `shared/worker-shape.md`:
#   python:3.12-slim + libsndfile1, wheel pulled from the HF repo at build time.
# No weights, no vLLM, no flash-attn compile step.
#
# On flash-attn specifically: the YuE2 model card says the HF package "uses
# PyTorch, CUDA graphs, and FlashAttention", which reads like a build
# requirement. It is not one — grepping the `yue2_infer-0.1.5` wheel source
# finds no `flash_attn` import at all, and `triton` appears only inside
# `yue2/fast.py`, which is the separate `fast` (vLLM) extra we deliberately do
# not install. The HF path goes through PyTorch's own SDPA. That is why the
# slim base is sufficient and why there is no multi-hour attention build here.

FROM python:3.12-slim

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
