# Stage 03 — cover-and-edit-modes

> Layer 2 · "What do I do?" — the control point of the whole system.

**Purpose:** Extend the worker with cover and edit modes via mode routing and the subprocess-per-model-family isolation design per Roadmap Phase 2

One job: this stage does that and nothing else. A stage that fetches does not
also filter; a stage that filters does not also format.

## Inputs
| Kind | File/Location | Scope | Why |
|------|---------------|-------|-----|
| working | ../02-create-mode-worker/output/create-mode-worker.md | full | the worker repo this stage extends |
| reference | ../../shared/model-facts.md | SheetSage2 + Qwen3-ASR-1.7B sections | cover-path transcription API shapes |
| reference | ../../shared/dependency-pins.md | full | subprocess-per-model-family isolation design + open questions |
| reference | ../../shared/vram-budget.md | cover stage rows | sequential-loading budget this design must respect |
| reference | ../../shared/locked-decisions.md | decision 6 + Phase 3 list | Qwen3-ASR over Parakeet; what stays out of scope |
| reference | ../../shared/worker-shape.md | Phase 2 section | added repo layout, mode dispatch |

Exact paths only. **working** = this stage's starting point (the create-mode
worker). **reference** = stable factory material. Anything not listed here is
not loaded.

## Process
1. Read the inputs above — only those.
2. Add `worker/modes.py` (dispatch on `mode: create | cover | edit`),
   `worker/transcribe_sheetsage/` and `worker/transcribe_asr/` as isolated
   venv subprocesses per `dependency-pins.md`.
3. Implement `cover`: SheetSage2 subprocess → `melody.abc`, exit → Qwen3-ASR
   subprocess → `lyrics.txt`, exit → main YuE2 process generates with
   `cot="melody"`.
4. Implement `edit`: `pipe.plan()` → exported score → regenerate with
   `cot="full"` and the edited `abc=` — no subprocess stages needed.
5. Write a short record of what was built to `output/`.

## Outputs
| Artifact | Location | Format |
|----------|----------|--------|
| worker repo additions | ../../worker/modes.py, transcribe_sheetsage/, transcribe_asr/ | code |
| build record | output/cover-and-edit-modes.md | markdown |

## Human check
Run one `cover` job and one `edit` job end-to-end on the same 24 GB GPU tier
as `create`, and confirm via VRAM sampling that no stage exceeded the
`create`-job peak (14.08 GiB) before calling this stage complete.

## Checkpoints _(optional — creative stages)_
- [ ] pause point for human steering

## Audits _(optional — creative stages)_
- [ ] quality gate before writing output

## Status caveat — read before trusting the ICM stage table

The root `CONTEXT.md`'s generated table reports this stage **COMPLETE**. That is
the ICM rule applied mechanically — *a stage is COMPLETE when its `output/` holds
a file other than `.gitkeep`* — and this stage's `output/` does.

**It does not mean this stage's `## Human check` has passed.** It has not. That
check requires one `cover` job and one `edit` job on a 24 GB GPU with VRAM
sampling, and no cover or edit job has ever run: the transcription models have
never been invoked on real audio.

Treat the mechanical status as "the build record is written", not as "the stage is
done". The distinction is tracked in Outline as the task **"Stage 03 Human check —
cover and edit on GPU"** (status `blocked`, needs hardware).

Do not start a Stage 04, or treat cover/edit as production-ready, until that check
passes. If the two ever disagree in the other direction — table says empty but
output exists — the table is stale; run `icm sync`.
