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
| **Image build on RunPod** | **succeeded** — built by the platform from this repo, not locally |
| **Endpoint live** | `/health` reports `workers: ready 1, idle 1` — the module imported and booted |
| **Error paths in production** | **3/3 verified**, each in ~250–300 ms with no GPU time (see below) |
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

## Defects found and fixed

Eleven total. Six were caught during the build by running something; five were
caught afterwards by independent critics — including two that produced a
**successful response containing a false statement**, which no amount of green
tests would have surfaced.

| # | Defect | Found by | Fix |
|---|---|---|---|
| 1 | Every real job failed at upload — `WorkerConfig(cache=...)` left storage unset | test | `WorkerConfig.autoload()` |
| 2 | `seed: true` silently became seed 1 (bools are ints) | test | reject before coercion |
| 3 | Boot failure escaped as a traceback instead of a structured error | smoke run | `BootError` added to the catch |
| 4 | Object keys flattened; `..` survived sanitisation | test | per-segment `_safe_segment` |
| 5 | `HF_XET_CACHE` used `setdefault`, so the bulk of a cold-start download could land on the container disk | test | assigned, not defaulted |
| 6 | Storage resolved *after* generation — a bad bucket cost 70 s of GPU before failing | self-review | resolve storage before generating |
| 7 | Boot was lazy despite its docstring; a ~12 GB download ran inside the first job's timeout, and every later job retried it | arch critic | `boot_worker()` at import + a circuit breaker |
| 8 | `build_response()` outside the `try`, so the documented `--test_input` path crashed | arch critic | moved inside the `try` |
| 9 | Cleanup ran before response assembly — local runs returned `file://` URLs to deleted files | arch critic | response built before cleanup |
| 10 | Workdir keyed on the caller-supplied `id` alone — two same-id jobs destroyed each other's artifacts (silent corruption) | arch critic | per-invocation `uuid4` suffix |
| 11 | **`truncated` is a dict** — `bool({...})` is always `True`, so every successful song was reported truncated | API critic | `_normalise_truncation()` |
| 12 | **`YUE2_VAE` was invented** — zero hits in the package; the `decoder` field reported a name that changed nothing | API critic | `YUE2_VAE_REPO`, honoured via `vae=` |
| 13 | **The fix for #7 broke the image.** Moving boot to module import made the Dockerfile's `import handler` smoke test hydrate the volume inside a build layer, baking ~12 GB of weights into the image — directly against locked decision 4 | builder, on rebuild | boot moved to `main()`; Dockerfile now fails the build if any `*.safetensors` is present |

Defects 11 and 12 are the instructive ones: both are silent, both shipped through
a green suite, and both were missed by a builder-run verification that checked
*shape* rather than *semantics*.

Defect 13 is a different kind of instructive: **the fix for one defect created
another**, and only a rebuild could reveal it. The first two images built during
this stage were 12.9 GB and correct; the third silently became a 24 GB image with
the model weights inside it. Nothing in the test suite could have caught that —
it is a property of the *build*, not the code — which is why the Dockerfile now
asserts it directly rather than trusting a reviewer to notice.

Full analysis, per-defect reasoning, and a correction to the comparison harness:
**`review-findings.md`**.

### The image-build guard

`worker/Dockerfile` now fails the build if any `*.safetensors` or
`pytorch_model*.bin` exists anywhere in the image. The rule "weights live on the
network volume, never in the image" (locked decision 4) was previously enforced
by convention and by the Dockerfile simply not copying them. Convention was not
enough: an unrelated refactor moved the boot into module scope, which turned the
import smoke test into a weight download. The guard makes the rule a build-time
assertion, so the next such refactor fails loudly and cheaply instead of
producing a fat image that deploys fine and costs storage forever.

## Production verification

The image built on RunPod's platform and the endpoint is live. Three jobs were
submitted and all three failed by design, in 235–300 ms each — which is the
evidence that matters: the fail-fast ordering holds in production, so a
misconfigured worker costs milliseconds rather than 70 seconds of GPU time.

Full evidence, including what each failure establishes:
**`review-findings.md`**.

**Still unverified: no song has been generated.** Every job failed before
generation, which the missing B2 configuration guarantees. The remaining gap is
one endpoint setting, not a code change.

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

## Layout corrections and a process failure

Two placement mistakes and one rule violation are documented in full in
`review-findings.md`. In short, because the next stage depends on the outcomes:

- **`Dockerfile` is at the repository root**, not `worker/` — RunPod's GitHub
  build looks there by default.
- **`runpod.serverless.start` is at module scope** in `handler.py`, preceded by
  `boot_worker()`. Importing that module therefore downloads weights and starts a
  server; use `python -m py_compile` to check it, never `import`.
- **Nothing is built locally.** Locked decision 1 forbids it, and it was violated
  five times during this stage. The rule now lives in the root `AGENTS.md` under
  **Build & deploy**.

## Open question carried forward

`shared/dependency-pins.md` flags that Qwen3-ASR could plausibly share YuE2's
venv rather than needing full isolation — the design decision (subprocess) still
stands, but for the VRAM-residency reason, not the dependency-conflict one that
was originally assumed. Worth re-litigating in Stage 03. Nothing in this stage
depends on the answer.
