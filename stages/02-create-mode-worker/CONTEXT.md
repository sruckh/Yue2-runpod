# Stage 02 — create-mode-worker

> Layer 2 · "What do I do?" — the control point of the whole system.

**Purpose:** Scaffold and implement the create-mode RunPod Serverless worker end-to-end per Roadmap Phase 1: handler, boot/model-caching, schema, storage, Dockerfile

One job: this stage does that and nothing else. A stage that fetches does not
also filter; a stage that filters does not also format.

## Inputs
| Kind | File/Location | Scope | Why |
|------|---------------|-------|-----|
| working | ../01-research-and-decisions/output/research-and-decisions.md | full | index into this stage's factory reference material |
| reference | ../../shared/model-facts.md | YuE2-3B section only | pipeline API, weight layout |
| reference | ../../shared/dependency-pins.md | YuE2 row only | exact wheel pins for the Dockerfile |
| reference | ../../shared/vram-budget.md | create/edit rows | performance baseline for the README/endpoint config |
| reference | ../../shared/locked-decisions.md | decisions 1–5 | deployment shape, storage egress, GPU tier |
| reference | ../../shared/worker-shape.md | Phase 1 section | repo layout, handler flow, Dockerfile, endpoint config |

Exact paths only. **working** = this run's product so far. **reference** =
stable factory material. Anything not listed here is not loaded.

## Process
1. Read the inputs above — only those.
2. Scaffold the `worker/` repo layout named in the Roadmap Phase 1 section:
   `handler.py`, `boot.py`, `schema.py`, `storage.py`, `config.py`, plus
   `tests/`, `Dockerfile`, `requirements.txt`, `.env.example`, `README.md`.
3. Implement the `create` mode: boot → `ensure_models()` → resident
   `YuE2Pipeline` → per-job validate → generate → upload to B2 → return.
4. Write a short record of what was built (and any deviations from the
   Roadmap) to `output/`.

Constraints (exact pins, VRAM figures, licensing terms) live in the
`shared/` reference files, not restated here.

## Outputs
| Artifact | Location | Format |
|----------|----------|--------|
| worker repo (code) | ../../worker/, Dockerfile, requirements.txt, README.md | code |
| build record | output/create-mode-worker.md | markdown |

## Human check
Run one end-to-end `create` job (submit → poll → B2 URL returned) against a
warm volume and confirm the returned FLAC/score.abc/settings JSON match the
Roadmap's documented response shape before Stage 03 starts.

## Checkpoints _(optional — creative stages)_
- [ ] pause point for human steering

## Audits _(optional — creative stages)_
- [ ] quality gate before writing output
