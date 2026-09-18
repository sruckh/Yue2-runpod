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
wrong writes a 12 GB download to the container disk and fails there rather
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

Every path that can fail — validation, boot, generation, upload, response
assembly — returns `{"error": "..."}`. A worker that dies on one bad job is a
worker that keeps costing money. If you add a new failure mode, add it to the
handler's catch tuple **and** to the tests; a missing `BootError` there was a
real bug.

### The pipeline boots at module scope, then hands off to RunPod

`handler.py` ends with:

```python
boot_worker()
runpod.serverless.start({"handler": handler})
```

This is the **standard RunPod worker shape** — the reference workers do the same,
and the SDK is what discovers `--test_input` and drives the handler. Do not wrap
it in a custom `main()`: an earlier version did, and it worked, but it hid the
`runpod.serverless.start` call inside a function where neither a reader nor a
build script looks for it.

Boot must come **before** `start()`. Lazy-loading on the first job would put a
~12 GB download plus model construction inside a *job's* timeout budget, so the
first job on a cold volume would be killed for taking longer than a generation is
allowed to take. A failed boot is recorded in `_boot_error` and reported per job:
one cold start, never one per job.

**Consequence: importing `handler` has side effects** — it boots, and it starts a
server. Anything that inspects this code (the Dockerfile, CI, tests) must use
`python -m py_compile` rather than `import`. Tests stub `boot.load_pipeline` and
`runpod.serverless` *before* importing.

### What the pipeline reports is not always what it looks like

Two traps found the hard way, both silent — a green suite and a `200` response
through both:

- **`SongResult.truncated` is a dict**, not a bool (`{"abc": ..., "semantic": ...}`).
  `bool()` of it is always `True`. Derive summary flags from values, never from
  a container's truthiness. See `_normalise_truncation`.
- **The `yue2_infer` package reads no environment variables** on this code path.
  A config knob only works if *we* honour it — e.g. `YUE2_VAE_REPO` works because
  `boot.load_pipeline` passes it as `vae=`. Do not invent a variable and report
  it back as provenance; report what the pipeline was actually built with.

### Test doubles must match the real artifact shapes

`tests/conftest.py`'s `FakeSong` writes the exact artifact set the real
`save_artifacts` produces — including `truncated` as a dict. A double that is
convenient rather than faithful is worse than no double: it makes the suite green
about behaviour the pipeline never exhibits.

`tests/test_review_regressions.py` holds the tests named for the defects that got
through. Keep it that way — each one documents a failure mode, not just an
assertion.

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
the root `Dockerfile` persists in the image layers forever — never do it, not even
temporarily, not even for a test.

## Verification

```bash
python -m pytest tests/ -q                              # 173 tests, no GPU, no network
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
