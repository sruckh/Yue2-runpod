---
icm_form: pipeline
domain: YuE2 RunPod Serverless Worker
created: 2026-09-17
---

# YuE2 RunPod Serverless Worker — ICM workspace

> Layer 0 · "Where am I?"

An **Interpretable Context Methodology** workspace. The folder structure *is*
the architecture: numbered folders carry sequencing, hierarchy carries context
scoping, plain files carry state. One agent reads the right files at the right
moment — no multi-agent framework.

**Form:** pipeline — the same sequence runs repeatedly and a deliverable leaves
each run.

## Layers
- **L0** — `IDENTITY.md` / `CLAUDE.md` (here): "Where am I?"
- **L1** — `CONTEXT.md`: "Where do I go?" (routing)
- **L2** — `stages/NN-*/CONTEXT.md`: "What do I do?" (the control point)
- **L3** — `_config/`, `shared/`, stage `references/`: "What rules apply?" (factory)
- **L4** — `stages/NN-*/output/`: "What am I working with?" (product)

L0–L2 are the catalog: small, stable, no content payload. If a routing file is
growing, it is absorbing payload — move the payload to a shelf and leave a link.

## How to use
1. Answer `setup/questionnaire.md` once; the answers become `_config/` files.
2. Add each step with `icm stage <dir> <name>` — one folder, one job.
3. Refresh routing with `icm sync <dir>`.
4. Walk it with `icm run <dir>`; edit each `output/` before the next stage runs.
5. Validate with `icm audit <dir>` before you finish.
