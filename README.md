<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="YuE2 RunPod Worker — style and lyrics become a song. The panel plots a melody from the ABC score YuE2 writes before it synthesizes any audio.">
</p>

<p align="center">
  <code>POST a style + lyrics</code> → <code>get back a song</code>
</p>

---

A RunPod Serverless worker that turns a style prompt and lyrics into a complete
song — vocals and accompaniment, 48 kHz stereo — on a single 24 GB GPU. Built on
[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), which plans the music as an ABC
score first and then realizes it as audio.

One request in, one song out, and nothing for you to run.

## What you send, what you get

```jsonc
// POST https://api.runpod.ai/v2/<endpoint>/run
{
  "input": {
    "style":  "City Pop, upbeat, groovy bass, electric guitar, neon city night",
    "lyrics": "[Verse]\nStreetlights blink, watching every passer-by\n[Chorus]\nTonight we stay awake",
    "cot":    "full",     // "full" | "melody" | "off"   — default "full"
    "seed":   12300       // optional, default 831001
  }
}
```

```jsonc
// response
{
  "audio_url":     "https://…/audio.flac",  // presigned, 7-day expiry
  "score_abc_url": "https://…/score.abc",   // the plan the model wrote first
  "duration":      214.85,                  // seconds of audio
  "sample_rate":   48000,
  "seed":          12300,
  "cot":           "full",
  "decoder":       "m-a-p/YuE2-Vae",
  "truncated":     false,
  "timings":       { "abc": {…}, "semantic": {…}, "nar_seconds": … },
  "artifact_urls": { … }                    // every artifact, incl. result.json
}
```

The response carries **URLs and metadata, never audio bytes**. A full song does
not belong in a job response, and polling stays cheap.

## How it works

```
  POST /run
      │
      ├─ validate ──────── against the pipeline's own bounds, before any GPU work
      ├─ resolve storage ─ a bad bucket fails here, not after a long generation
      │
      ├─ generate
      │    ├─ plan   → score.abc      the model writes the music as notation first
      │    ├─ semantic → tokens
      │    ├─ NAR    → latents
      │    └─ decode → audio.flac     YuE2-Vae, 48 kHz stereo
      │
      ├─ upload → B2                  presigned URLs minted locally, no round trip
      └─ respond → URLs + metadata
```

**Weights live on a network volume, never in the image.** RunPod's own cached-models
feature is the primary source; anything it does not cover is downloaded on first
boot into the same cache. Once the cache verifies complete, offline mode is
enabled so a missing file fails loudly instead of quietly re-downloading the
whole cache.

## Layout

```
Dockerfile                 at the repo root — RunPod's default build path
worker/
├── handler.py             RunPod entrypoint; boots at import, generates per job
├── boot.py                model cache resolution + fallback, pipeline construction
├── schema.py              job validation, mirroring the pipeline's own bounds
├── storage.py             Backblaze B2 egress over the S3 API
├── config.py              environment-driven settings
├── requirements.txt       exact pins — never a loose range
└── .runpod/               endpoint config and example job payloads
tests/                     GPU and B2 both mocked — runs with no hardware
```

## Running it

The test suite needs **no GPU, no network and no credentials** — the GPU and B2
are both mocked, by design:

```bash
python -m pytest tests/ -q
```

The image is built by **RunPod's platform** from this repository, not locally.
Push, then watch the endpoint's **Builds** tab.

### Deploying

1. Import this repo as a Serverless endpoint. The Dockerfile is at the root, so
   no build-path configuration is needed.
2. Attach a network volume at `/runpod-volume`.
3. Declare `m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae` under **cached models**.
4. Set the four storage variables on the endpoint template:

   ```
   B2_ENDPOINT_URL   B2_KEY_ID   B2_APP_KEY   B2_BUCKET
   ```

   Credentials belong on the endpoint, never in the image — a key in a
   Dockerfile persists in the layers forever.

Optional: `VOLUME_ROOT`, `MEMORY_BUDGET_GIB`, `JOB_TIMEOUT_SECONDS`,
`YUE2_VAE_REPO`, `YUE2_PROGRESS`, `DEFAULT_COT`, `LOG_LEVEL`. See
[`worker/.env.example`](worker/.env.example).

| Endpoint setting | Value | Why |
|---|---|---|
| GPU | RTX 4090 24 GB, ×1 | peak 14.08 GiB; one song at a time |
| Container disk | 30 GB | the image alone is 12.9 GB (CUDA wheels + torch) |
| Job timeout | ≥ 30 min | generation plus model load, with headroom |
| Network volume | `/runpod-volume` | datacenter-specific — endpoint and volume must share a DC |

## Design notes

**Validation happens before the GPU does.** `schema.py` mirrors the pipeline's own
limits on purpose. The pipeline validates too, but only once the model is
resident — so a mistyped seed costs a full generation's worth of GPU time instead
of a millisecond. A malformed job is rejected before any GPU work begins.

**Failures return structured errors.** Validation, boot, generation, upload and
response assembly all return `{"error": "..."}` rather than raising. A worker that
dies on one bad job is a worker that keeps costing money.

**The pipeline boots before the first job**, not lazily. A lazy first load would
put the weight download inside a job's own timeout, so the first job on a cold
volume would be killed for taking longer than a generation is allowed to take.

**Importing `handler` has side effects** — it boots the model and starts the SDK.
Anything inspecting the code uses `python -m py_compile`, never `import`.

## Current scope

**Create mode is implemented and verified end to end.** The worker accepts
`mode: "create"` and rejects `cover` and `edit` by name, so a caller gets
`not implemented in this worker build (Stage 03)` rather than a confusing
validation error. Cover and edit modes — requiring SheetSage2 and Qwen3-ASR as
isolated subprocess environments — are not built.

## Licensing

The **YuE2-3B weights are CC BY-NC 4.0** with an additional creator permission:
individuals and creators may use and monetize outputs royalty-free, academic
non-commercial use is free, and **commercial use by companies requires contacting
the YuE2 authors**. The code in this repository is Apache 2.0. SheetSage2 and
MERT-v2-FullSong weights, needed for cover mode, are CC BY-NC 4.0 and not covered
by YuE2's grant — confirm terms separately before any commercial use.
