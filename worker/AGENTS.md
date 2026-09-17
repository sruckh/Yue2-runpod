# AGENTS.md — `worker/`

Work contract for the YuE2 RunPod Serverless worker. Read the root `AGENTS.md`
first; this file adds local rules and never weakens the root.

## Purpose

The create-mode serverless worker: a RunPod handler that caches YuE2-3B weights
onto a network volume, generates a song per job, and uploads the artifacts to
Backblaze B2. One job in, one song out, unattended, on a paid 24 GB GPU.

Stage 02 of the ICM pipeline at the repository root. Cover and edit modes are
Stage 03 and are not implemented here.

## Ownership

The `worker/` tree owns the running service. `tests/` owns proof that it works
without a GPU. Neither owns the facts about the model — those live in `shared/`
at the repository root and are cited, never restated.

## Local contracts

### Imports are flat, deliberately

`import config`, never `from worker import config`. The container runs
`python -u /app/handler.py` with `/app` as the working directory, matching the
layout of the reference RunPod workers. `pyrightconfig.json` sets
`extraPaths: ["worker"]` so tooling resolves the same way the container does.

**`handler.py` must call `apply_hf_env()` before importing anything that touches
HuggingFace.** The cache location is read at import time; getting this order
wrong writes a 12 GB download to the 20 GB container disk and fails there rather
than here. The `# noqa: E402` on the imports below it is load-bearing.

### Every version is an exact pin

`requirements.txt` carries `==` on every line. These wheels are known to conflict
across model families, and a resolver left to its own judgement will pick a
combination that imports and then fails mid-generation.

- torch comes from the CUDA 12.8 index. `torch==2.10.0` on plain PyPI is CPU-only
  and fails on `device="cuda"`.
- The `yue2_infer` wheel ships inside the HuggingFace model repo, not PyPI. Install
  it with `--no-deps` so the resolver cannot re-pick a CPU torch.
- The wheel's own `Requires-Dist` list is the source of truth for model-side pins —
  not the model card, which omits three of them.

### Failures return, they do not raise

Every path that can fail — validation, boot, generation, upload — returns
`{"error": "..."}`. A worker that dies on one bad job is a worker that keeps
costing money. If you add a new failure mode, add it to the handler's catch tuple
**and** to `tests/test_handler.py`; a missing `BootError` there was a real bug.

### Validation mirrors the pipeline's bounds

`schema.py` duplicates `yue2.protocol.SongRequest.__post_init__`'s rules on
purpose. The pipeline validates too, but only after the model is resident — about
70 seconds before a caller learns their seed was a string. If upstream tightens a
bound, a test in `tests/test_schema.py` should fail and point at the divergence.

### The response carries URLs, never bytes

A 48 kHz stereo FLAC does not belong in a job response. `storage.py` is the only
module that talks to object storage; `handler.py` never imports boto3 directly.

### Secrets come from the environment

B2 credentials are set on the RunPod endpoint template. A key baked into the
`Dockerfile` persists in the image layers forever — never do it, not even
temporarily, not even for a test.

## Verification

```bash
python -m pytest tests/ -q                              # 112 tests, no GPU, no network
ruff check worker tests && ruff format --check worker tests
python -c "import config, schema, storage, boot, handler"   # from worker/
```

The suite mocks the GPU and B2 by design, so it runs anywhere. That is also its
**ceiling**: it proves the wiring, the contract adherence and the artifact
plumbing, not that a song sounds right.

**What only a person can verify:** one end-to-end `create` job against a warm
volume — submit, poll, and confirm the returned FLAC, `score.abc` and settings
JSON match the documented response shape. This needs a 24 GB GPU, a RunPod API
key and B2 credentials. It is Stage 02's `## Human check` and cannot be automated
from a box without a GPU.

## Child DOX Index

None. `worker/` is flat — five modules, no subpackages. Stage 03 will add
`worker/transcribe_sheetsage/` and `worker/transcribe_asr/` as isolated venv
subprocesses; each becomes its own boundary with its own contract when it lands.
