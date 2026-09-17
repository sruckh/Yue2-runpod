# Research & decisions — index

> This stage's product is the factory reference material in `../../shared/`,
> not a copy of it here. This file is the pointer the next stages' Inputs
> tables cite.

- `../../shared/model-facts.md` — YuE2-3B, SheetSage2, MERT-v2-FullSong,
  Qwen3-ASR-1.7B: API shapes, weight layout, licensing.
- `../../shared/dependency-pins.md` — verified pins per model family, the
  subprocess-per-model-family isolation design, and the two still-open
  dependency questions.
- `../../shared/vram-budget.md` — sequential-loading VRAM budget for a single
  24 GB RTX 4090, plus the measured create/edit performance baseline.
- `../../shared/locked-decisions.md` — locked design decisions and the
  Phase 3 deferred/out-of-scope list.

Canonical source: Outline project "Yue2" (docs listed in each shared file's
header) and `.serena/memories/yue2-model.md`. `RESEARCH-NOTES.md` at the repo
root is a superseded historical snapshot — see the note in
`locked-decisions.md`.
