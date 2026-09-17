# Worker repo shape — Phase 1 (create) & Phase 2 (cover/edit)

> Layer 3 · factory reference, stable across every stage.
> Source of truth: Outline "YuE2 Serverless Roadmap"
> (id `b6d45319-5ec4-44c7-a138-ee3a1855d180`).

## Phase 1 — create mode (Stage 02 builds this)

Repo layout (local `/opt/docker/Yue2`, container `WORKDIR /app`):

```
Yue2/
├── worker/
│   ├── handler.py      # runpod handler; pipeline loaded once at module import
│   ├── boot.py         # ensure_models(): idempotent snapshot_download -> volume cache
│   ├── schema.py       # request validation (style, lyrics, cot, abc, seed, cfg_scale)
│   ├── storage.py      # boto3 -> Backblaze B2, upload + URL minting
│   └── config.py       # env vars: cache root, B2 endpoint/keys/bucket, defaults
├── tests/               # GPU + S3 mocked, no GPU needed to run locally
├── Dockerfile
├── requirements.txt     # runpod, boto3 (+ dev: ruff, mypy, pytest)
├── .env.example
└── README.md
```

**Handler flow:**
1. Boot: set HF cache env vars → `ensure_models()` (both HF repos) →
   `YuE2Pipeline.from_pretrained(...)` once, resident in VRAM.
2. Per job: validate input → `pipe(style, lyrics, cot, abc, seed, cfg_scale)` →
   save FLAC to container-disk tmp → upload FLAC + score.abc + settings JSON
   to B2 → return `{audio_url, score_abc_url, duration, seed, cot, decoder, timings}`.

**Dockerfile:** `python:3.12-slim` + `libsndfile1`; installs the wheel pulled
from the `m-a-p/YuE2-3B` HF repo + `runpod` + `boto3`. No weights, no vLLM, no
flash-attn compile step.

**Endpoint config:** GPU RTX 4090 24 GB class · workers min 0, scale after
warm · job timeout ≥ 30 min · container disk ≥ 20 GB · network volume mounted
at `/runpod-volume` · one RunPod datacenter (volume is DC-specific).

## Phase 2 — cover + edit modes (Stage 03 builds this)

Added repo layout:

```
worker/
├── modes.py                  # mode dispatch: create | cover | edit
├── transcribe_sheetsage/     # isolated venv, invoked as a subprocess
│   └── ...                   # SheetSage2 + MERT-v2-FullSong -> melody-only score.abc
├── transcribe_asr/           # isolated venv, invoked as a subprocess
│   └── ...                   # Qwen3-ASR-1.7B -> lyrics text
```

**Sequential-loading design** (see `vram-budget.md` for the numbers): `cover`
spawns the SheetSage2 subprocess → writes `melody.abc`, exits → spawns the
Qwen3-ASR subprocess → writes `lyrics.txt`, exits → the main process (YuE2
already resident) generates with `cot="melody"`. `edit` needs no subprocess
stages — `pipe.plan()` → caller/agent revises the score → regenerate with
`cot="full"` in the main process.

**Weight caching additions:** `m-a-p/SheetSage2`, `m-a-p/MERT-v2-FullSong`,
`Qwen/Qwen3-ASR-1.7B` — same idempotent `snapshot_download` pattern as
Phase 1, added to `boot.py`'s `ensure_models()`.
