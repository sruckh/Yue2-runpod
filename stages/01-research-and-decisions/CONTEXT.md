# Stage 01 — research-and-decisions

> Layer 2 · "What do I do?" — the control point of the whole system.

**Purpose:** Distill the researched YuE2/SheetSage2/Qwen3-ASR facts, dependency pins, VRAM budget, subprocess-isolation design, and licensing terms into stable factory reference material every later stage cites

One job: this stage does that and nothing else. A stage that fetches does not
also filter; a stage that filters does not also format.

## Inputs
| Kind | File/Location | Scope | Why |
|------|---------------|-------|-----|
| reference | `<Outline project "Yue2" — see shared/*.md file headers for exact doc IDs>` | all | canonical, most recently updated (2026-09-17); not a local path, fetched via the outline skill's CLI |
| reference | ../../.serena/memories/yue2-model.md | all | same facts, shorter form, already in sync with Outline |
| reference | ../../RESEARCH-NOTES.md | all — flag drift only | STALE: still lists SheetSage2 covers as out-of-scope; superseded by the Outline docs above |

Exact paths only. This is the first stage — there is no prior stage `output/`
to read; every input here is external reference material, not another
stage's product.

## Process
1. Read the three reference sources above, Outline first, RESEARCH-NOTES.md
   last and only to spot where it has drifted from Outline.
2. Distill dependency pins, VRAM budget, API shapes, licensing terms, and
   locked/deferred decisions into `../../shared/*.md` (one topic per file) —
   this is the actual product; do not restate it a second time in this
   stage's own `output/`.
3. Write a short index in `output/` that points at the `shared/` files
   instead of duplicating them.

## Outputs
| Artifact | Location | Format |
|----------|----------|--------|
| factory reference | ../../shared/model-facts.md, dependency-pins.md, vram-budget.md, locked-decisions.md | markdown |
| index | output/research-and-decisions.md | markdown |

## Human check
Read `../../shared/dependency-pins.md`'s open questions and confirm they are
still genuinely open (not already resolved elsewhere) before Stage 02 starts.

## Checkpoints _(optional — creative stages)_
- [ ] pause point for human steering

## Audits _(optional — creative stages)_
- [ ] quality gate before writing output
