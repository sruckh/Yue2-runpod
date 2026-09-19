# Glossary — YuE2 RunPod Serverless Worker

Domain terms used across the workspace (Layer 3 · factory). Define a term once
here; every other file links here rather than re-defining it.

**CoT (`cot=`)** — YuE2's chain-of-thought mode: `"full"` (melody+chord plan,
default), `"melody"` (melody-only, used for covers), `"off"` (no symbolic
plan). See `shared/model-facts.md`.

**ABC score (`score.abc`)** — the editable symbolic composition YuE2 plans
before rendering audio; the "white-box interface" the model card refers to.

**Mode** — the job payload's `mode: "create" | "cover" | "edit"` field that
`worker/modes.py` dispatches on. Not to be confused with `cot`.

**Subprocess isolation** — the design where SheetSage2 and Qwen3-ASR each run as
a **subprocess** of the main handler, rather than in-process. The point is
guaranteed **VRAM release on exit**: a child that exits returns its memory to the
driver, so a cover's peak is its largest phase rather than the sum of its model
families. Measured — see `shared/vram-budget.md`.

Each family *originally* had its own venv as well, on the assumption the pins
could not coexist. They can, and the venvs were removed 2026-09-19. There is one
environment; the process boundary is what remains and what matters. See
`shared/dependency-pins.md`.

**Network volume** — the RunPod-persistent, datacenter-specific storage
mounted at `/runpod-volume`, used to cache model weights across worker boots.

**Cold boot / warm boot** — a worker's first job on fresh storage (pays the
full model download) versus a subsequent job (weights already cached).
