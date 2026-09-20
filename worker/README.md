# YuE2 — RunPod Serverless Worker

Generate full songs — vocals and accompaniment — from a style prompt and lyrics,
on a single 24 GB GPU. Built on [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B):
symbolic planning first, then realization, 48 kHz stereo, no quantization.

**All three modes are implemented and verified end to end on hardware.** The API
contract — request fields, response shape, artifact keys, error kinds — lives in
the repository root `README.md`; this file covers the worker's internals.

## What it does

One job in, one song out.

```jsonc
// POST /run — input. Full field table in the root README.
{
  "input": {
    "mode":   "create",   // "create" | "cover" | "edit"  — default "create"
    "style":  "City Pop, upbeat, groovy bass, electric guitar, neon city night",
    "lyrics": "[Verse]\nStreetlights blink, watching every passer-by\n[Chorus]\nTonight we stay awake",
    "cot":    "full",     // "full" | "melody" | "off"   (default "full")
    "seed":   12300,      // optional, default 831001
    "cfg_scale": 1.2,     // optional, [0, 20]
    "abc":    "X:1\n...",  // optional score; requires cot != "off"
    "instrumental": false // optional; true fills the lyrics slot, see root README
  }
}
```

The response adds `mode`, `stages` (cover/edit), `vram`, `truncated`,
`truncation_by_stage`, `elapsed_seconds` and `request` to the fields below.

The response carries **URLs and metadata, never raw bytes** — a 48 kHz stereo
FLAC does not belong in a job response.

## Layout

```
worker/
├── handler.py         RunPod entrypoint; boots at import, generates per job
├── boot.py            Idempotent weight caching onto the network volume
├── schema.py          Job validation, mirroring the pipeline's own bounds
├── storage.py         Backblaze B2 egress over the S3 API
├── config.py          Environment-driven settings
├── modes.py           create | cover | edit dispatch
├── abc_score.py       ABC validation and chord stripping
├── subprocess_runner.py  Runs one model family as its own subprocess
├── vram.py            Device-wide memory sampling, per job
├── check_env.py       Asserts the environment matches its pins
├── transcribe_sheetsage/  audio -> melody.abc  (subprocess entrypoint)
├── transcribe_asr/        audio -> lyrics.txt  (subprocess entrypoint)
├── requirements.txt   Exact pins — never a loose range
└── .runpod/           Endpoint config and example job payloads

Dockerfile             At the REPO ROOT (RunPod's GitHub build looks there by
                       default). python:3.11-slim-trixie + libsndfile1 + ffmpeg
                       + gcc, which triton needs at runtime.
tests/                 GPU and B2 both mocked — runs with no hardware
```

The modules use **flat imports** (`import config`, not `from worker import config`)
because the container runs `python -u /app/handler.py` with `/app` as the working
directory. `handler.py` must also call `apply_hf_env()` *before* importing
anything that touches HuggingFace — the cache location is read at import time,
and getting that order wrong puts a 12 GB download on a 20 GB container disk.

## Running it

### Locally (no GPU needed)

```bash
python -m pytest tests/ -q          # no GPU, no network, no credentials
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
| GPU | any 24 GB card, ×1 | Measured peaks 8.3–9.1 GiB; one song at a time |
| Workers | min 0, scale after warm | Cold start pays the weight download |
| Job timeout | ≥ 30 min | ~40–70 s generation, plus a slow first boot |
| Container disk | **30 GB** | The image is a few GB once pulled — see below |
| Network volume | mounted at `/runpod-volume` | One datacenter — the volume is DC-specific |
| Cached models | `m-a-p/YuE2-3B` | RunPod's cached-models feature holds **one** repo. The worker caches all five into the volume and downloads the rest on first boot |

**The GPU list may name several 24 GB types; the worker runs on whichever the pool
provides.** The pool is not homogeneous, and `device_total_mib` is how a job says
which card it actually got: **23034 MiB on an NVIDIA L4**, **24564 MiB on an
NVIDIA GeForce RTX 4090**. Host RAM has varied between 126 GB and 507 GB across
workers. Quote the card alongside any peak you record.

**Peak VRAM scales with song length.** A 57 s song measured 8.5 GiB; 197 s
measured 9.1 GiB; 282 s measured 9.4 GiB. The published 14.08 GiB figure was a
~3.6-minute song at "full context" on a 4090 — and our 282 s run, which is
*longer*, on the same card, peaked ~4.4 GiB below it. So length is not the whole
story; the difference is most likely context length rather than song duration, and
**14.08 GiB should be treated as a ceiling for a configuration we have not
reproduced**, not as refuted. Compare like with like.

**No model weights are baked in** — verified from inside the built container by a
build-time guard, not assumed.

The image was **10.8 GB** while SheetSage2 and Qwen3-ASR each had their own venv
with their own torch stack. Both venvs were removed after the unification
experiment showed both families run correctly on the main stack (verified on
hardware), taking ~6.5 GB with them. The 20 GB container disk that was too small
now has ample room.

### Environment

See `.env.example` for the annotated list. The four that must be set:

```
B2_ENDPOINT_URL   B2_KEY_ID   B2_APP_KEY   B2_BUCKET
```

Set them on the **endpoint template**, never in the image — a key baked into a
Dockerfile persists in the layers forever. Note that RunPod's config API reports
only *some* of an endpoint's env vars; a variable missing from the API response is
not necessarily missing from the container. The B2 key needs `writeFiles` on the
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

**Weights are never baked into the image.** RunPod's **cached-models** feature is
the primary source: models declared in the endpoint's configuration are fetched
by the platform and mounted on the network volume at
`/runpod-volume/huggingface-cache/hub` in the standard HuggingFace cache layout
(`models--{org}--{name}/snapshots/{rev}/`, resolved via `refs/main`).

A **network-volume fallback** covers anything that cache does not: if a repo is
absent, `ensure_models()` downloads it into the *same* cache root with
`cache_dir=`, so both sources are one tree and the next start is a cache hit.
Once every repo is verified present, offline mode (`HF_HUB_OFFLINE=1`) is
enabled — which is what makes a later missing blob fail loudly instead of
quietly re-downloading 12 GB.

The pattern RunPod documents is followed exactly, including the paths. A worker
that downloads with `local_dir=` instead invents a layout the platform's cache
cannot see, so the cache is ignored and every start re-downloads — which is what
this worker did before the layout was corrected.

**Validation happens before the GPU does.** `schema.py` mirrors the pipeline's
own `SongRequest` bounds on purpose. The pipeline does validate, but it validates
after the model is resident — ~70 seconds before a caller learns their seed was a
string.

**Failures return structured errors.** Every path that can fail — validation,
boot, generation, upload, response assembly — returns `{"error": "..."}` rather
than raising. A worker that dies on one bad job is a worker that keeps costing
money.

**The pipeline is resident before the first job.** `handler.py` ends with
`boot_worker()` followed by `runpod.serverless.start({"handler": handler})` — the
standard RunPod shape. Boot runs once at worker start, not lazily on first use,
which would put a ~12 GB download and model construction inside a job's own
timeout budget. If boot fails, the failure is remembered and every job answers
from it immediately: one cold start, not one per job.

Importing `handler` therefore has side effects. Anything that inspects the code
without running it — the Dockerfile, CI — uses `python -m py_compile`, and the
Dockerfile fails the build if any `*.safetensors` ends up in the image.

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
