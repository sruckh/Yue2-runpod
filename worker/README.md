# YuE2 — RunPod Serverless Worker

Generate full songs — vocals and accompaniment — from a style prompt and lyrics,
on a single 24 GB GPU. Built on [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B):
symbolic planning first, then realization, 48 kHz stereo, no quantization.

This is **Stage 02** of the project's ICM pipeline: create mode, end to end.
Cover and edit modes are Stage 03.

## What it does

One job in, one song out.

```jsonc
// POST /run — input
{
  "input": {
    "style": "City Pop, upbeat, groovy bass, electric guitar, neon city night",
    "lyrics": "[Verse]\nStreetlights blink, watching every passer-by\n[Chorus]\nTonight we stay awake",
    "cot": "full",        // "full" | "melody" | "off"   (default "full")
    "seed": 12300,        // optional, default 831001
    "cfg_scale": 1.2,     // optional, [0, 20]
    "abc": "X:1\n..."     // optional, supply your own score (needs cot != "off")
  }
}
```

```jsonc
// response
{
  "audio_url":     "https://…/audio.flac",   // presigned, 7-day default
  "score_abc_url": "https://…/score.abc",    // the planned score
  "duration":      214.85,                   // seconds of audio
  "seed":          12300,
  "cot":           "full",
  "decoder":       "m-a-p/YuE2-Vae",
  "timings":       { "abc": …, "semantic": …, "nar_seconds": … },
  "artifact_urls": { … }                     // every artifact, incl. result.json
}
```

The response carries **URLs and metadata, never raw bytes** — a 48 kHz stereo
FLAC does not belong in a job response.

## Layout

```
worker/
├── handler.py       RunPod entrypoint; boots at import, generates per job
├── boot.py          Idempotent weight caching onto the network volume
├── schema.py        Job validation, mirroring the pipeline's own bounds
├── storage.py       Backblaze B2 egress over the S3 API
├── config.py        Environment-driven settings
├── requirements.txt Exact pins — never a loose range (see below)
├── Dockerfile       python:3.12-slim + libsndfile1; wheel from the HF repo
└── .runpod/         Endpoint config and example job payloads
tests/               173 tests, GPU and B2 both mocked
```

The modules use **flat imports** (`import config`, not `from worker import config`)
because the container runs `python -u /app/handler.py` with `/app` as the working
directory. `handler.py` must also call `apply_hf_env()` *before* importing
anything that touches HuggingFace — the cache location is read at import time,
and getting that order wrong puts a 12 GB download on a 20 GB container disk.

## Running it

### Locally (no GPU needed)

```bash
python -m pytest tests/ -q          # 173 tests, no GPU, no network, no credentials
ruff check worker tests && ruff format --check worker tests
```

### One job without RunPod

```bash
cd worker
VOLUME_ROOT=/tmp/yue2 \
  python handler.py --test_input='{"input":{"style":"City Pop","lyrics":"[Verse]\nhello"}}'
```

Artifacts stay on local disk and are reported as `file://` URLs — same response
shape, no B2 credentials. Still needs a GPU and the weights, though: the
generation is real.

### Endpoint configuration

| Setting | Value | Why |
|---|---|---|
| GPU | RTX 4090 24 GB (ADA_24), ×1 | Peak is 14.08 GiB; one song at a time |
| Workers | min 0, scale after warm | Cold start pays the weight download |
| Job timeout | ≥ 30 min | ~71 s generation for a 3.6-min song, plus slow boots |
| Container disk | **30 GB** | The image alone is 12.9 GB — see below |
| Network volume | mounted at `/runpod-volume` | One datacenter — the volume is DC-specific |

The image measures **12.9 GB**, dominated by the `nvidia` CUDA pip wheels (4.3 GB)
that `torch` pulls in, plus `torch` itself (1.8 GB). The original 20 GB container
disk left too little room for the pull/unpack phase, so the endpoint config asks
for 30. **No model weights are baked in** — that is verified from inside the built
container, not assumed.

Stage 03 will grow this considerably: SheetSage2 and Qwen3-ASR each get their own
venv with their own torch (2.8.0 and unpinned respectively), so expect to raise
the container disk again.

### Environment

See `.env.example` for the annotated list. The four that must be set:

```
B2_ENDPOINT_URL   B2_KEY_ID   B2_APP_KEY   B2_BUCKET
```

Set them on the **endpoint template**, never in the image — a key baked into a
Dockerfile persists in the layers forever. The B2 key needs `writeFiles` on the
bucket; it never needs `listBuckets`, because nothing here enumerates.

Once the volume is warm, set `HF_LOCAL_FILES_ONLY=true` so a missing blob fails
loudly instead of silently re-downloading.

Two knobs worth knowing: `MEMORY_BUDGET_GIB` (default 24) is a **hard** VRAM cap
— the pipeline turns it into `set_per_process_memory_fraction`, so lower it to run
on a smaller card. `YUE2_VAE_REPO` selects the decoder; `m-a-p/YuE2-Vae-legacy`
reproduces the published benchmark protocol and makes the worker fetch that repo
too. Note this is *our* variable — the `yue2_infer` package reads no environment
variables on this path.

## Design notes

**Weights are never baked into the image.** Both repos — `m-a-p/YuE2-3B` and
`m-a-p/YuE2-Vae` — are cached on the network volume on first boot and reused by
every later worker. `ensure_models()` is idempotent and checks for the required
files before calling `snapshot_download`, so a warm start touches no network.

**Validation happens before the GPU does.** `schema.py` mirrors the pipeline's
own `SongRequest` bounds on purpose. The pipeline does validate, but it validates
after the model is resident — ~70 seconds before a caller learns their seed was a
string.

**Failures return structured errors.** Every path that can fail — validation,
boot, generation, upload, response assembly — returns `{"error": "..."}` rather
than raising. A worker that dies on one bad job is a worker that keeps costing
money.

**The pipeline is resident before the first job.** Boot runs at module import,
not on first use. A lazy boot would put a ~12 GB download and model construction
inside a job's own timeout budget, so the first job on a cold volume would be
killed for taking longer than a generation may take. If boot fails, the failure
is remembered and every job answers from it immediately — one cold start, not
one per job.

**`truncated` is derived, not coerced.** The pipeline's `SongResult.truncated` is
a dict (`{"abc": bool, "semantic": bool}`), and `bool()` of a non-empty dict is
always `True` — so a naive coercion reports every successful song as truncated.
The response reports a real boolean plus `truncation_by_stage`, and reports
`null` when the field is absent, because "not reported" is not "not truncated".

**One decoder, named in the response.** `m-a-p/YuE2-Vae` is the default.
`YuE2-Vae-legacy` is env-switchable for reproducing the published benchmark
protocol, and the response reports whichever actually ran.

**Only the files the pipeline reads get downloaded.** `ensure_models()` passes
`allow_patterns` matching the file set the pipeline's own Hub loader uses. The
`m-a-p/YuE2-3B` repo also ships demo audio and artwork under `assets/`, which a
bare `snapshot_download` would pull on every cold start for nothing.

**The VRAM ceiling is explicit.** The pipeline converts `memory_budget_gib` into
`torch.cuda.set_per_process_memory_fraction`, so it is a hard cap rather than a
hint — and its own default of 24 is an assumption about the card, not about the
job. Ours is set from `MEMORY_BUDGET_GIB` (default 24) so running on a smaller
GPU is a config change rather than an OOM at load.

## Pins

Every version in `requirements.txt` is exact. These wheels are known to conflict
across model families, and a resolver left to its own judgement will happily pick
a combination that imports and then fails mid-generation.

Two things worth knowing:

- **torch comes from the CUDA 12.8 index.** `torch==2.10.0` on plain PyPI is a
  CPU-only build — it installs cleanly and then fails on `device="cuda"`.
- **The `yue2_infer` wheel is not on PyPI.** It ships inside the model repo, so
  the Dockerfile fetches it from there. Its `Requires-Dist` list is the source of
  truth for the eight model-side pins; installing it with `--no-deps` prevents
  the resolver from re-picking a CPU torch.

FlashAttention is **not** a dependency, despite the model card mentioning it: the
wheel imports no `flash_attn`, and `triton` appears only in the vLLM `fast` extra
that this worker deliberately does not install.

## Licensing

YuE2 weights are **CC BY-NC 4.0** with an additional creator permission:
individuals and creators may use and monetize outputs royalty-free; academic
non-commercial use is free; **commercial use by companies requires contacting the
YuE2 authors**. The code and docs here are Apache 2.0. SheetSage2 and
MERT-v2-FullSong weights (Stage 03) are CC BY-NC 4.0 and not covered by YuE2's
grant — confirm terms separately before any commercial use.
