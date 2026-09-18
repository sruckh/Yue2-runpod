# Stage 02 — adversarial review findings

> Companion record to `create-mode-worker.md`. This is **history**: what two
> independent critics found, what was fixed, and what the comparison got wrong.
> Stage 03 does not need to walk it — the handoff record carries everything the
> next stage depends on.
>
> Kept rather than deleted because the defects here were the most valuable output
> of the stage, and because the last section documents a flaw in the review
> harness that would otherwise be repeated.

Two critics were spawned with fresh context and no visibility into the builder's
reasoning:

- **Architecture critic** — blind A/B against `runpod-workers/worker-faster_whisper`
  and `worker-comfyui`, labels stripped, order randomized per round.
- **API critic** — instructed to check every pipeline call against the real
  `yue2_infer-0.1.5` wheel source unpacked from the model repo.

Both critics eventually reported. Their findings are the most valuable output of
this stage: **five real defects, two of them silent-wrong-output on the happy
path**, none of which the 146-test suite could see.

### Architecture critic — verdict

**worker-2 (ours) wins, "not close."** worker-1 was the reference
`worker-faster_whisper` tree; the critic found it implements only the front half
of the job (a grep of its whole tree for `boto3|s3|bucket|upload|presign|volume`
returns zero hits — no storage client, no artifact written, no URL returned),
ships no test suite, and its Dockerfile `COPY`s `builder/`, `src/` and
`test_input.json`, none of which exist in the delivered tree.

It also confirmed the biggest thing that could not be tested here: our
`signal.signal(SIGALRM, ...)` job timeout is **safe**. It read RunPod's published
1.9.0 and 1.12.0 wheels and found both invoke a sync handler directly
(`handler_return = handler(job)`, no thread pool), with `JobScaler.start()`
running the loop via `asyncio.run` on the main thread.

### Defects it found in ours, and what changed

| # | Defect | Severity | Fix |
|---|---|---|---|
| 1 | Boot was **lazy, contradicting its own docstring** — a ~12 GB download ran inside the *first job's* timeout, and every later job re-attempted the cold start | high | `boot_worker()` runs at module import; a recorded `_boot_error` acts as a circuit breaker |
| 2 | `build_response()` sat outside the `try` and raises — the documented `--test_input` path crashed with a traceback | medium | moved inside the `try` |
| 3 | `_cleanup()` ran *before* `build_response()`, so local runs returned `file://` URLs to deleted files | medium | response built before cleanup |
| 4 | Workdir keyed on the caller-supplied `id` alone — two same-id jobs `rmtree` each other and share an upload prefix (silent corruption) | high | workdir gains a per-invocation `uuid4` suffix |
| 5 | `_reset_dir` outside the `try`; an unwritable volume bypassed the named-failure contract | low | wraps `OSError` as `WorkerError` |

### API critic — verdict

**Call surface correct** — "no wrong keyword, no missing argument, no wrong
filename, no uncaught exception", confirmed by reading the wheel *and* a 16-case
differential fuzz of our validator and kwargs-builder against the real
`yue2.protocol` (all 16 agree).

It then found two silent defects that the builder's own verification had missed,
because that verification checked *shape* (does the keyword exist) and not
*semantics* (what is the value):

| # | Defect | Why the suite missed it |
|---|---|---|
| 1 | **`truncated` is a dict** — `SongResult.truncated` returns `{"abc": bool, "semantic": bool}` (pipeline.py:90-91); `bool({...})` is always `True`, so **every successful song was reported as truncated** | the test double wrote a *scalar* where reality writes a dict |
| 2 | **`YUE2_VAE` was invented** — a grep over the entire wheel returns **zero hits**; setting it changed nothing, yet the response reported the name it claimed | the test asserted only that the string appeared, never that it was true |
| 3 | No worker-side context budget — the pipeline raises above ~78 KB of prompt, while we permitted 2 MiB | the limit had no test because it was assumed free |

Defects 1 and 2 are the ones that matter: both produce a *successful* response
containing a false statement. Neither crashes. The suite was green through both.

### What changed as a result

- `_normalise_truncation()` derives the flag from the stage values and preserves
  the per-stage detail; `truncated` is a real boolean, `truncation_by_stage`
  carries the breakdown, and an absent field reports `null` rather than `False`.
- `YUE2_VAE` → `YUE2_VAE_REPO` in config, honoured by passing `vae=` to
  `from_pretrained`; the handler reports `config.vae_repo`, so the provenance
  field describes what actually ran.
- `_MAX_PROMPT_BYTES` (64 KiB) pre-checks the combined prompt, below the measured
  ~78 KB ceiling to leave margin for languages that tokenize denser per byte.
- `tests/test_review_regressions.py` — 22 tests named for the *reason* each was
  missed, so a green run means these traps are guarded, not merely that the code
  compiles. The `FakeSong` double now writes the real dict shape.
- 146 → 173 tests.

### On the builder's own verification

Before the critics reported, the builder ran an AST-based API check and called it
clean. That check was correct about what it measured and **wrong to be
satisfied**: it verified keyword existence, not value semantics, and missed both
silent defects. Recorded here rather than quietly dropped — a self-run check
tends to test the shape of the thing its author already believed.

Its first version also produced two false failures from loose regexes before the
AST version was trustworthy.

### A correction to the comparison itself

The critic's findings split into two groups with **opposite** reliability, and
both trace to how the comparison tree was staged.

**Stands:** "worker-1 has no delivery path" — verified against the *real* clone,
not just the staged copy. `grep -rniE "boto3|s3|bucket|upload|presign"` over the
entire `runpod-workers/worker-faster_whisper` repository returns nothing. It
genuinely returns its transcript inline and has no object-storage step. That is a
true architectural difference, and the reason `worker-comfyui` was pulled as a
second sample.

**False positive:** "worker-1 is not buildable — it `COPY`s `builder/`, `src/` and
`test_input.json`, none of which exist anywhere in the delivered tree." All four
paths **do** exist in the real repository. The staging step for the blind
comparison copied only `src/*`, the Dockerfile, `requirements.txt` and the test
fixture into worker-1 — so the tree the critic received was one the *builder* had
already broken. It judged what it was given and was explicit that it had not
built either image, which is exactly the right thing to have done with a tree
that looked wrong.

The lesson is about the harness, not the critic: a blind comparison is only as
good as the normalisation step, and this one silently dropped a directory. Had
the verdict been closer, that would have decided it for the wrong reason.

**A real difference the critic's framing obscured:** worker-1's
`builder/fetch_models.py` downloads all ten Whisper checkpoints **into the image
at build time**. That is a legitimate design for models that small, and it is the
opposite of locked decision 4 (weights on a volume, never in the image) — which
exists because YuE2's weights are ~12 GB, not ~150 MB. Neither is wrong; they are
answers to different size constraints.

