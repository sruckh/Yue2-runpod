# Conventions — YuE2 RunPod Serverless Worker

Project-specific conventions (Layer 3 · factory). The methodology's own rules
live in the `icm` skill's `references/icm-conventions.md` — do not copy them
here; one home per fact.

- Python: pin exact versions in `worker/requirements.txt` per
  `shared/dependency-pins.md` — never a loose `>=`, since these wheels are
  known to conflict across model families.
- Each subprocess model family (`transcribe_sheetsage/`, `transcribe_asr/`)
  gets its own venv directory under `worker/`, never sharing site-packages
  with the main YuE2 process or with each other.
- Stage output records (`stages/NN-*/output/*.md`) describe what was built
  and any deviation from the stage's contract — they are not a copy of the
  code itself; the code lives in `worker/`.
- One fact, one file: dependency pins, VRAM figures, and API shapes live only
  in `shared/`; stage contracts and code comments cite them, never restate
  the numbers.
