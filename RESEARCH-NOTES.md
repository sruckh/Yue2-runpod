# YuE2-3B RunPod Serverless — Research Notes

> Recovered and distilled from Claude Code session 4ada5709-9ec7-49f5-bddd-1870d9d59938
> (2026-09-17, "YuE2-3B music model"). Sources: HuggingFace model card m-a-p/YuE2-3B,
> GitHub multimodal-art-projection/YuE (release yue2-v0.1.6), RunPod docs (Context7 +
> web search), and direct inspection of the yue2_infer-0.1.5 wheel metadata.

## 1. What YuE2-3B is

- Successor to YuE 1 (arXiv:2503.08638; YuE2 tech report pending). Open music generation.
- Core idea: symbolic planning — plans a melody+chord composition as an ABC score,
  then realizes it as a complete song (vocals + accompaniment), 48 kHz stereo.
- 4B params BF16 safetensors (name says "3B").
- Quality: tops WildSongBench — YuE2 best-of-8 6.9632 SongBench Avg vs Suno v5 (6.8721),
  Mureka 9 (6.9377), Suno v6 (6.5562). YuE 1: 4.9165.
- Licensing: weights CC BY-NC 4.0 + additional creator permission (individuals may
  monetize outputs; companies need a commercial license). Code/docs/agent skill Apache 2.0.

## 2. Pipeline API

    from yue2 import YuE2Pipeline
    pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda")
    song = pipe(style=..., lyrics=..., cot=..., abc=..., seed=..., cfg_scale=...)
    song.save("x.flac")            # 48 kHz stereo FLAC
    song.save_artifacts(dir)       # + score.abc, tokens, latents, settings

- cot: "full" (melody+chord plan, default) / "melody" (covers; does NOT strip chord
  symbols from a supplied ABC) / "off" (direct generation).
- abc=: supply your own score (use cot="full" to keep supplied harmony).
- Edit loop: pipe.plan(**request) -> edit score.abc -> regenerate with abc=.
  Editing re-renders the whole song; waveform outside the edit is NOT preserved.
- Covers (phase 2): SheetSage2 transcription (separate env) -> cot="melody".

## 3. Performance (model card)

- RTX 4090 24 GB, full CoT: 3.6-min song ~71 s gen (~139 LM tok/s), peak 11.18 GiB
  VRAM (max-context test 14.08 GiB). Needs ~24 GB host RAM. One song at a time.
- vLLM 0.19 on H800 80 GB: 32-concurrency ~373 songs/h (out of scope for us).
- HF runtime: PyTorch 2.10, Transformers 4.57.6, CUDA graphs, BF16 AR/NAR, FP32 VAE.

## 4. Verified dependency pins (wheel METADATA, 0.1.5)

Wheel ships INSIDE the HF repo (not on PyPI): yue2_infer-0.1.5-py3-none-any.whl
(repo also has 0.1.3; 0.1.6 exists ONLY as a GitHub release asset). 66 KB pure python.

Hard pins: torch==2.10.0, transformers==4.57.6, huggingface-hub==0.36.2,
safetensors==0.7.0, tiktoken==0.12.0, numpy==2.2.6, soundfile==0.13.1,
accelerate==1.13.0. Extras: [fast]=vllm+triton (unused), [serve]=fastapi+uvicorn
(unused - RunPod SDK serves), [test], [bench]. No flash-attn package needed
(card's "FlashAttention" = torch SDPA). pip pulls CUDA torch -> ~8-9 GB image,
NO weights baked.

Weights live in TWO HF repos - both must be cached:
- m-a-p/YuE2-3B: model.safetensors, config.json, modeling_yue2.py (custom code ->
  trust_remote_code), qwen.tiktoken, generation configs, weights_manifest.json,
  wheels, examples. Skip demo assets (mp3s, figures) via allow_patterns.
- m-a-p/YuE2-Vae: default decoder (YuE2-Vae-legacy = benchmark decoder only).

## 5. Locked design decisions (user, 2026-09-17)

1. Runs ENTIRELY as a RunPod serverless endpoint. Nothing installed/executed on the
   dev box. App can live anywhere on "/" in the container (we use /app).
2. 24 GB VRAM GPU preferred -> HF YuE2Pipeline mode (not vLLM).
3. Audio egress: Backblaze B2 S3-compatible bucket (boto3,
   endpoint_url https://s3.<region>.backblazeb2.com, keys from env).
4. Models NOT baked into the image. Download-on-boot into the network volume at
   /runpod-volume (RunPod's caching default). First cold start slower — accepted.

## 6. RunPod model-caching research findings

- Network volumes mount at /runpod-volume in serverless workers; persistent across
  worker restarts; DATACENTER-SPECIFIC (pin endpoint to one DC). Concurrent
  multi-worker writes need app-level care.
- RunPod's documented HF cache root: /runpod-volume/huggingface-cache/hub (used
  verbatim in docs.runpod.io HF guide + model-caching tutorial). Pointing HF env
  vars there aligns our cache with RunPod's native model-caching feature — both
  share one location.
- Official offline pattern (HF_HUB_OFFLINE=1, local_files_only=True) is for
  pre-warmed endpoints; we default OFFLINE off so first boot can download;
  env-switchable.
- runpod.serverless.start({"handler": handler}); sync handler def handler(job)->dict
  with job["input"]; load model at MODULE level (once per worker, not per job).
  RunPod docs explicitly list Backblaze B2 as a supported S3-compatible target.
- Reference worker: github.com/runpod-workers/model-store-cache-example.

## 7. Implementation plan (presented in the session, awaiting "proceed")

Repo layout (local: /opt/docker/Yue2, container WORKDIR /app):

    Yue2/
    ├── worker/
    │   ├── handler.py      # runpod handler; pipeline loaded once at module import
    │   ├── boot.py         # ensure_models(): idempotent snapshot_download -> volume cache
    │   ├── schema.py       # request validation (style, lyrics, cot, abc, seed, cfg_scale)
    │   ├── storage.py      # boto3 -> Backblaze B2, upload + URL minting
    │   └── config.py       # env vars: cache root, B2 endpoint/keys/bucket, defaults
    ├── tests/              # GPU + S3 mocked (no GPU needed locally)
    ├── Dockerfile
    ├── requirements.txt    # runpod, boto3 (+ dev: ruff, mypy, pytest)
    ├── .env.example
    └── README.md           # endpoint config, env vars, example request, cost notes

Model caching flow:
- HF_HOME=/runpod-volume/huggingface, hub cache /runpod-volume/huggingface-cache/hub.
- ensure_models() at worker boot: snapshot_download for BOTH m-a-p/YuE2-3B and
  m-a-p/YuE2-Vae, allow_patterns skipping demo MP3s/figures/wheels. Idempotent —
  bootstrap and fallback in one.

Handler flow:
1. Boot: env -> ensure_models() -> YuE2Pipeline.from_pretrained once, resident in
   VRAM (~11 GiB of 24 GB).
2. Per job: validate -> pipe(...) -> FLAC to container-disk tmp -> upload FLAC +
   score.abc + settings JSON to B2 -> return {audio_url, score_abc_url, duration,
   seed, cot, decoder, timings}. Presigned GET URLs by default.

Dockerfile: python:3.12-slim + libsndfile1; pip install the 0.1.5 wheel from the HF
repo + runpod + boto3. No weights, no vLLM, no flash-attn compile.

Endpoint config (README): GPU RTX 4090 24 GB class · workers min 0, scale after warm ·
job timeout >= 30 min (cold first job may include the 8 GB download; warm ~75 s gen
+ upload) · container disk >= 20 GB · network volume at /runpod-volume · one datacenter.

Out of scope (phase 2): vLLM runtime, SheetSage2 covers, YuE2-Vae-legacy (env-switchable).

## 8. Project state / next step

- Greenfield: nothing scaffolded. The session ended immediately after presenting the
  plan above — NEXT STEP: scaffold the worker (that plan is approved-pending).
- Serena memories preserved in .serena/memories/ (core is the entry point).
  The copies in /root/.serena/memories/ were written under the mis-located project
  "root" and are superseded by this project's copies.
