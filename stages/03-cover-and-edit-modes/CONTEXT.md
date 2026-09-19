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

## Human check — PASSED 2026-09-19

Run one `cover` job and one `edit` job end-to-end on the same 24 GB GPU tier as
`create`, and confirm via VRAM sampling that no stage exceeded the `create`-job
peak. **Done.**

| Job | Mode | Result | Peak |
|---|---|---|---|
| `0b424184` | cover | COMPLETED | 9270 MiB |
| `471bda93` | edit | COMPLETED | 9348 MiB |
| `3b23b730` | create (same song) | COMPLETED | 9072 MiB |

All on NVIDIA L4, device total 23034 MiB. No stage exceeded the create peak; a
cover's peak equals its `generate` phase rather than `transcribe + generate`,
which is the subprocess boundary holding as designed. `edit` honoured its
supplied score, retaining chord symbols (`"Bb"`, `"Dm7"`, `"Eb"`) under
`cot="full"`.

**Note the 14.08 GiB above has been corrected to a measured comparison.** That
figure is the published YuE2 benchmark at ~3.6 minutes; our songs were 197 s.
Peak VRAM scales with song length, so the honest comparison is like-for-like —
see `shared/vram-budget.md`.

## Checkpoints _(optional — creative stages)_
- [ ] pause point for human steering

## Audits _(optional — creative stages)_
- [ ] quality gate before writing output

## Status — the caveat that used to be here is resolved

This section previously warned that the ICM table's **COMPLETE** was mechanical
only — *a stage is COMPLETE when its `output/` holds a file* — while the Human
check had not passed and no cover or edit job had ever run.

**The check has since passed.** Both modes are verified on hardware (table
above), the Outline task "Stage 03 Human check — cover and edit on GPU" is
resolved, and cover and edit are production-ready in the sense the check defines.

Two things the caveat was right about, kept here because they recur:

- **A mechanical status is not a verified one.** The rule that flipped this stage
  to COMPLETE fired when the build record was written, months of hardware
  verification before the check it was standing in for. That gap is worth
  re-reading whenever a stage table says done.
- **The reverse direction is also possible** — table says empty but output
  exists — and then the table is stale; run `icm sync`.

What remains genuinely unverified is narrower and listed in the root `README.md`
under *Current scope*: whether `instrumental: true` produces instrumental audio
(nobody has listened), and the numerical quality of a cover (the transcription is
confirmed sane; the *musical* result has not been judged).
