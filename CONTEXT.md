# Context — YuE2 RunPod Serverless Worker

> Layer 1 · "Where do I go?"

The flow in one line: _research & lock decisions → build the create-mode
worker → extend it with cover & edit modes — one RunPod Serverless endpoint,
nothing installed locally._

<!-- icm:sync:begin -->
## Stages
| # | Stage | Job | Output | Status |
|---|---|---|---|---|
| 01 | research-and-decisions | Distill the researched YuE2/SheetSage2/Qwen3-ASR facts, dependency pins, VRAM budget, subprocess-isolation design, and licensing terms into stable factory reference material every later stage cites | `stages/01-research-and-decisions/output/` | COMPLETE |
| 02 | create-mode-worker | Scaffold and implement the create-mode RunPod Serverless worker end-to-end per Roadmap Phase 1: handler, boot/model-caching, schema, storage, Dockerfile | `stages/02-create-mode-worker/output/` | COMPLETE |
| 03 | cover-and-edit-modes | Extend the worker with cover and edit modes via mode routing and the subprocess-per-model-family isolation design per Roadmap Phase 2 | `stages/03-cover-and-edit-modes/output/` | empty |
<!-- icm:sync:end -->

## Factory / product
- **Factory** (stable every run): `_config/`, `shared/`
- **Product** (new every run): each `stages/NN-*/output/`

Status is whatever exists: a stage is COMPLETE when its `output/` holds a file
other than `.gitkeep`. Nothing moves forward until a person has read the last
output.
