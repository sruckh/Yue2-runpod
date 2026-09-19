# Conventions — YuE2 RunPod Serverless Worker

Project-specific conventions (Layer 3 · factory). The methodology's own rules
live in the `icm` skill's `references/icm-conventions.md` — do not copy them
here; one home per fact.

- Python: pin exact versions in `worker/requirements.txt` per
  `shared/dependency-pins.md` — never a loose `>=`, since these wheels are
  known to conflict across model families.
- Each subprocess model family (`transcribe_sheetsage/`, `transcribe_asr/`) runs
  as its own **process**. It does *not* get its own venv: there is one
  environment, verified 2026-09-19 to run all three families correctly. The
  earlier rule here also had the path wrong — the venvs lived at `/opt/venvs/`,
  never under `worker/`, so `COPY worker/ /app/` could not overwrite them.
  `run_stage` still accepts a venv name for a future family that genuinely needs
  one; nothing uses it today.
- Stage output records (`stages/NN-*/output/*.md`) describe what was built
  and any deviation from the stage's contract — they are not a copy of the
  code itself; the code lives in `worker/`.
- One fact, one file: dependency pins, VRAM figures, and API shapes live only
  in `shared/`; stage contracts and code comments cite them, never restate
  the numbers.
