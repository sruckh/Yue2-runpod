# Stage 02 — create-mode worker: build record

> What was built, what deviated from the Roadmap, and what only a person can
> still verify. The code lives in `worker/`; this file describes it and does not
> restate it (per `_config/conventions.md`).

## Status

**Build complete, verified as far as this machine allows.**
Awaiting the stage's `## Human check`, which needs a GPU.

| Check | Result |
|---|---|
| Test suite | 146 passing — GPU and B2 mocked |
| Lint (`ruff check`) | clean |
| Format (`ruff format --check`) | clean |
| Container layout (`import config, schema, storage, boot, handler`) | OK |
| Dockerfile syntax (BuildKit `--check`) | no warnings |
| Docker image build | **built: 12.9 GB**, `yue2-worker:stage02` |
| In-container import + wheel check | **passed** — flat layout resolves and `yue2_infer 0.1.5` imports inside the image |
| No weights baked into the image | **verified from inside the container** — no `*.safetensors` anywhere |
| API contract vs the real wheel | verified by AST — all kwargs valid; see below |
| ICM audit | OK (0 warnings) |
| Pushed to GitHub | `github.com/sruckh/Yue2-runpod`, private, `main` @ `442c209` |
| End-to-end generation | **not possible here** — no GPU, no RunPod key, no B2 credentials |

## What was built

`worker/` — five modules, plus packaging, tests and CI:

| Artifact | Notes |
|---|---|
| `handler.py` | RunPod entrypoint. Boots at import; per job validates → generates → uploads → responds. SIGALRM job budget. |
| `boot.py` | `ensure_models()` idempotent volume cache for both HF repos; `load_pipeline()`; `health()`. |
| `schema.py` | Job validation mirroring the pipeline's own bounds; `validate_mode()` rejects Stage 03 modes by name. |
| `storage.py` | B2/S3 egress, presigned URLs, injectable client. |
| `config.py` | Env-driven settings; HF cache env applied before any HF import. |
| `Dockerfile` | `python:3.12-slim` + `libsndfile1` + `ffmpeg`; `yue2_infer` wheel installed `--no-deps` from the HF repo. |
| `requirements.txt` | Exact pins; torch from the CUDA 12.8 index. |
| `.runpod/` | Endpoint config (`hub.json`) and three example payloads (`tests.json`). |
| `tests/` | 146 tests across six suites. |
| `.github/workflows/ci-test-worker.yml` | Lint, type check, tests, image build — no GPU or secrets needed. |
| `README.md`, `AGENTS.md` | Operator docs and the local DOX work contract. |

Root-level config added: `pyrightconfig.json` (flat-module resolution),
`ruff.toml`, `.dockerignore`.

## Deviations from the Roadmap

Four, all deliberate:

1. **`ffmpeg` added to the image, not just `libsndfile1`.** `worker-shape.md`
   names only `libsndfile1`. `soundfile` needs it, but so does any audio
   inspection and Stage 03's transcription subprocesses will need it regardless.
   It costs image size and saves a rebuild later.
2. **`allow_patterns` on `snapshot_download`.** The Roadmap describes a plain
   "both HF repos" download. The `m-a-p/YuE2-3B` repo also ships demo audio and
   artwork under `assets/`, so a bare download pulls megabytes the worker never
   reads. The file set now matches what the pipeline's own Hub loader requests.
3. **`memory_budget_gib` forwarded explicitly.** Not in the Roadmap. The
   pipeline converts it into `torch.cuda.set_per_process_memory_fraction`, a hard
   cap, and its own default of 24 is an assumption about the card. Exposed as
   `MEMORY_BUDGET_GIB` so a smaller GPU is a config change, not an OOM.
4. **A local job timeout (SIGALRM).** The Roadmap relies on RunPod's own timeout.
   RunPod's timeout kills the worker with no diagnostics; this returns a
   structured error naming the stage first.

## Corrections to the factory reference

Reading the shipped `yue2_infer-0.1.5` wheel contradicted
`shared/dependency-pins.md` and `shared/model-facts.md` in four places. All four
are corrected in `shared/dependency-pins.md`; summarized here because they
changed the build:

1. **Three required pins were missing** from the YuE2 row: `safetensors==0.7.0`,
   `tiktoken==0.12.0`, `soundfile==0.13.1` — all hard `Requires-Dist` entries.
   `soundfile` is the reason the image needs `libsndfile1`.
2. **FlashAttention is not a dependency.** The model card's "uses PyTorch, CUDA
   graphs, and FlashAttention" reads like a build requirement. The wheel imports
   no `flash_attn` anywhere; `triton` appears only inside `yue2/fast.py`, the
   separate vLLM `fast` extra. The slim base is sufficient and there is no
   attention compile step. This is what made the Roadmap's "no flash-attn
   compile step" line true rather than lucky.
3. **The default decoder is a repo, not a flag.** `from_pretrained` takes
   `vae=` and resolves it to a directory; both repos must be cached.
4. **`__call__` exposes more than documented** — `cancelled` and `on_token`
   callbacks, and `from_pretrained(progress=False)` to silence the progress
   spinner that would otherwise spam serverless logs.

## Defects the loop caught

Each was found by running something, not by reading the code once:

| # | Defect | Found by | Fix |
|---|---|---|---|
| 1 | Every real job failed at upload — `WorkerConfig(cache=...)` left storage unset | test | `WorkerConfig.autoload()` |
| 2 | `seed: true` silently became seed 1 (bools are ints) | test | reject before coercion |
| 3 | Boot failure escaped as a traceback instead of a structured error | smoke run | `boot.BootError` added to the catch |
| 4 | Object keys flattened; `..` survived sanitisation | test | per-segment `_safe_segment` |
| 5 | `HF_XET_CACHE` used `setdefault`, so the bulk of a cold-start download could land on the container disk | test | assigned, not defaulted |
| 6 | Storage resolved *after* generation — a bad bucket cost 70 s of GPU before failing | self-review | resolve storage before generating |

## What this stage does *not* prove

Stated plainly, because the contract's Human check exists for exactly this:

- **No song has been generated.** The suite proves wiring, contract adherence
  and artifact plumbing. It cannot prove the output sounds right, nor that the
  ~14 GiB VRAM peak holds in practice.
- **The image has not been run on a GPU.** It builds; it has not loaded weights.
- **No upload has happened.** The B2 path is exercised against a fake client, so
  credentials, bucket permissions and real presigned-URL expiry are unverified.
- **The `yue2_infer` wheel has not been imported.** `boot.load_pipeline()` has
  never executed against a real install.

### The Human check

> Run one end-to-end `create` job (submit → poll → B2 URL returned) against a
> warm volume and confirm the returned FLAC/score.abc/settings JSON match the
> Roadmap's documented response shape before Stage 03 starts.

Needs: an RTX 4090-class GPU, a RunPod API key, B2 credentials, and a network
volume. None exist on the machine this was built on.

## Adversarial review

Two critics were spawned with fresh context and no visibility into the builder's
reasoning:

- **Architecture critic** — blind A/B against `runpod-workers/worker-faster_whisper`
  and `worker-comfyui`, labels stripped, order randomized per round.
- **API critic** — instructed to check every pipeline call against the real
  `yue2_infer-0.1.5` wheel source unpacked from the model repo.

**Neither critic returned a report.** Both were spawned, both ran (the API critic
was observed writing and executing probe scripts against the wheel), and both
went idle without delivering findings, despite three requests. This is recorded
as a failure of the review step, not as a clean bill of health — a critic that
does not report is not a critic that found nothing.

**In its place, the builder ran the API-contract check directly** against the
same ground truth, by AST rather than regex, so it is reproducible:

| Check | Method | Result |
|---|---|---|
| Call sites into the pipeline | AST walk of `handler.py` | exactly one: `pipe(**call_kwargs)` |
| Kwargs vs `SongRequest` fields | AST of `protocol.py` | `style`, `lyrics`, `cot`, `seed`, `abc`, `cfg_scale` — all valid |
| `from_pretrained` keywords | AST of `pipeline.py` | `progress`/`local_files_only` accepted directly; `device`/`memory_budget_gib` reach `__init__` via `**kwargs` |
| `save_artifacts` output | AST + source of `SongResult`/`SymbolicPlan` | writes `audio.flac` and calls `plan.save()`, which writes `score.abc` |

An earlier pass of this same check reported two failures. Both were false
positives in the check itself — a loose regex that matched the *response* dict
instead of the call kwargs, and a pattern that missed a `.write_bytes` call. They
are noted because a self-run verification that cannot be wrong is not a
verification; the first pattern was wrong twice before the AST version was
trustworthy.

**What this does not substitute for.** The architecture critic's blind A/B never
happened, so no independent comparison against the RunPod bar was ever returned.
The worker has been read against that bar by its builder (boot/caching, error
taxonomy, response shape all follow it), but self-assessment is not the
independent judgment the contest was set up to produce. Treat the architecture
comparison as outstanding.

## Open question carried forward

`shared/dependency-pins.md` flags that Qwen3-ASR could plausibly share YuE2's
venv rather than needing full isolation — the design decision (subprocess) still
stands, but for the VRAM-residency reason, not the dependency-conflict one that
was originally assumed. Worth re-litigating in Stage 03. Nothing in this stage
depends on the answer.
