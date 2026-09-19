# Stage 03 — cover and edit modes: build record

> What was built, what it deviates from, and what only the container run can
> prove. The code lives in `worker/`; this file describes it.

## Status

**Built and statically verified. Awaiting the stage's `## Human check`.**

> **Addendum, 2026-09-19 — the check passed.** This section is the record as
> written when the build finished, and is left as written. Both modes have since
> run on hardware:
>
> | Job | Mode | Result | Peak |
> |---|---|---|---|
> | `0b424184` | cover | COMPLETED | 9270 MiB |
> | `471bda93` | edit | COMPLETED | 9348 MiB |
>
> Neither exceeded the create peak for the same song, and `edit` retained the
> chord symbols of its supplied score. The test count below (281) is also a
> snapshot — the suite is larger now. Current state: the root `README.md` under
> *Current scope* and `shared/vram-budget.md` for measured peaks.

| Check | Result |
|---|---|
| Test suite | 281 passing (was 188) — no GPU, no network, no model libraries |
| Lint / format | clean |
| `py_compile` | clean |
| Import safety | asserted by test: no module in the worker process imports torch/transformers/qwen_asr |
| Container run | **not possible here** — no GPU, and packages must not be installed on this box |

## What was built

| Artifact | Role |
|---|---|
| `worker/modes.py` | Mode dispatch; the cover and edit pipelines |
| `worker/abc_score.py` | ABC validation and chord stripping, adapted from upstream (Apache 2.0) |
| `worker/subprocess_runner.py` | Runs a model family as its own subprocess; file-based protocol |
| `worker/transcribe_sheetsage/run.py` | Melody transcription, subprocess entrypoint |
| `worker/transcribe_asr/run.py` | Lyric transcription, subprocess entrypoint |

_(Row text updated 2026-09-19: they read "in its own venv" / "executed in the
… venv" as written. The venvs were removed the next day — see the addendum at the
top. The process boundary these rows describe is unchanged.)_
| `Dockerfile` | Builds both venvs into the image |
| `tests/test_modes.py`, `tests/test_abc_score.py`, `tests/test_mode_wiring.py`, `tests/test_review_regressions_stage03.py` | 93 new tests |

## The bar, and what it found

The bar was the YuE2 authors' own implementation, which turned out to exist and
to be directly comparable: `skills/yue2-music/` in
`github.com/multimodal-art-projection/YuE` ships their SheetSage2 transcribe
script, an ABC parsing/validation library (`abc_tools.py`), the documented
cover and edit flows, and tests for the melody-only path. Apache 2.0, same as
this repository.

Reading it contradicted `shared/model-facts.md` in ways that would have produced
a working-looking but wrong cover:

**1. The SheetSage2 `transcribe()` signature is richer than recorded.** Our notes
said `model.transcribe("song.mp3", output_dir=..., melody_only=True)`. The real
call takes `prompts=[...]`, `dtype`, `preset`, `max_seconds`, and
`base_model_path`. The prompt set is the mechanism that keeps chords out:
melody tasks get `["timestamp","downbeat_meter","structure","key","melody_full"]`
and `full` adds `chord_full`. **We now pass the melody prompt set and never
`chord_full`.**

**2. `melody_only` is not guaranteed to exist on a given revision.** Upstream
inspects the signature and refuses rather than guessing — *"refresh the model
code to a reviewed revision exposing melody_only explicitly"*. We do the same,
because the failure mode is a run that returns chords while we believe we asked
for a melody.

**3. Transcribing and lyric-writing are separate operations.** Upstream is
explicit: *"Source-separation, transcription, lyric recognition and
score-conditioned generation are distinct operations"*, and for an audio cover
you transcribe the melody, review it, remove chord annotations, and *"obtain or
check the lyrics separately"*. The contract implied one pass; it is two.

**4. `strip_chords` exists and is load-bearing.** Upstream ships a
chord-removal function whose own invariant is *"Chord removal changed melody"*.
We adapt it, and added a defence the contract did not ask for: `cover` now
verifies the returned melody is chord-free rather than trusting
`melody_only=True`, because **YuE2's `cot="melody"` does not strip chords
itself** — a melody that still carries harmony silently produces something that
is not a cover.

## Deviations from the Roadmap

1. **`worker/abc_score.py`, `subprocess_runner.py` — not in the contract.** The
   contract named `modes.py` plus two subprocess directories. The ABC validator
   and the subprocess runner are separate modules because both are independently
   testable and both carry logic worth isolating. `abc_score` in particular
   enforces the format contract the whole cover path depends on.
2. **The venv layout was `/opt/venvs/<family>/`, not under `worker/`.** _(Updated
   2026-09-19: both venvs were removed; there is one environment. The path point
   below still explains why they were never put under `worker/`, which is worth
   keeping if one is ever needed again.)_ The
   contract said `worker/transcribe_sheetsage/` and `worker/transcribe_asr/` hold
   the subprocess code, which they do — but the *environments* are built to
   `/opt/venvs/` so that `COPY worker/ /app/` does not overwrite them and so the
   image layout is visible in one place. The entrypoints stay with the worker
   code, as specified.
3. **`validate_mode_inputs` added to `schema.py`.** Per-mode requirements (cover
   needs `source_audio`) had no home; folding them into `validate_job` would make
   `create` reject payloads it has no opinion about.
4. **`abc.py` was renamed to `abc_score.py`.** The first name shadows Python's
   stdlib `abc` module for anything importing it — a silent hazard, caught before
   it landed.

## Defects found and fixed during this stage

| # | Defect | Found by |
|---|---|---|
| 1 | `abc.py` shadowed the stdlib `abc` module | builder, on import |
| 2 | `strip_chords` dropped the trailing newline from every rewritten line, merging music lines and **corrupting the score** | the note-invariant check inside `strip_chords`, added for exactly this |
| 3 | `L:1/24` passed validation — the regex accepted any `1/N`, so a validator laxer than its consumer | a test written to assert the rejection |
| 4 | The unit-denominator check was documented but not implemented | the same test |
| 5 | A child exiting non-zero with no `result.json` reported only "exited 1", discarding the stderr that explains it | a test asserting the message content |
| 6 | `voices` accumulated duplicates from repeated `V:` headers | a test on a real score |
| 7 | A test asserted `assert X or True` — vacuous, could never fail | ruff |
| 8 | **`source_audio` was validated but never carried onto `SongParameters`**, so `params.source_audio` was always `None` and every cover job failed | self-review |
| 9 | **`lyrics_supplied` was declared and read but set by nothing** — a caller supplying their own lyrics silently got the transcription instead | self-review |
| 10 | **The handler never fetched the recording from its URL**, so the mode received a URL where it needed a path | self-review |
| 11 | **`lyrics` was unconditionally required**, which made every cover job impossible: a cover's words come from transcription | self-review |
| 12 | A child exiting non-zero without a `result.json` reported only the exit code | test |
| 13 | **Chord detection failed open.** Any quoted token matching a *closed* quality list counted as a chord; `"Cmaj9"`, `"Chorus"`, `"N.C."` fell through an invented "text annotation" exemption. `"Cmaj9"` is real harmony — it passed validation **and** survived `strip_chords`, reaching YuE2 with `cot="melody"` | critic, executed against upstream |
| 14 | **The `require_melody_only` check was unreachable** — `strip_chords` ran first and already guarantees chord-freedom, so the flag could never fire at its only call site | critic |
| 15 | ASR ran unconditionally and its failure was fatal *before* consulting `lyrics_supplied` — supplying lyrics bought nothing | critic |
| 16 | `offline` was never sent to either child, so `local_files_only` was always `False` and a child could re-download what the endpoint had cached | critic |
| 17 | `scratch_dir` used `rmtree(ignore_errors=True)` then `mkdir(exist_ok=True)`: a failed removal left the directory intact, and the next stage reported the *previous* run's artifacts as its own | critic |
| 18 | Entrypoints were never `.resolve()`d — worked only because Python absolutises `__file__` | critic |
| 19 | `HOME` was dropped from the child environment, so the HF libraries lost `~/.cache` and `~/.netrc` | critic |
| 20 | `[K:...]` inline key changes were missing from `TOKEN_RE`; `_note_signature` produced a spurious note from one | critic, executed |
| 21 | `if params.abc:` treated an empty score as "no score", silently planning instead of failing | critic |

Defects 2, 8, 9, 10 and 11 are the instructive ones, and they split into two
kinds.

**Defect 2** — the file-based rewrite lost line endings and silently merged
measures — was caught by a check the *upstream* code also performs. That is the
argument for porting an invariant rather than a function: the invariant knew
something the function's author did not.

**Defects 8–11 are one failure mode wearing four faces: plumbing that was never
connected.** A field validated and not carried forward; a flag read and never
set; a URL handed where a path was needed; a required field that contradicts the
mode's own premise. Every one of them is invisible to a test that constructs
`SongParameters(...)` by hand — which is exactly what the mode and subprocess
suites do.

They were found by tracing one raw job end to end rather than by reading code,
and the fix is structural: `tests/test_mode_wiring.py` now goes in through
`validate_job` and the handler, so a field that exists in validation and is
never used by the mode fails a test rather than a customer's job.

They were also found *late* — after the suite was green at 248 tests. A green
suite that never exercises the seam between two tested components is not
evidence about the seam.

**Defects 13–21 came from an independent critic that executed both
implementations rather than reading them.** Defect 13 is the one that matters:
it defeated the exact guarantee this stage describes as load-bearing, and it did
so by *inventing an exemption* — "anything else in quotes is a text annotation".
In YuE2's native format a quoted token on a music line **is** chord notation;
there is no annotation case. The lesson generalises: **a validator that ignores
what it does not understand is not a validator, and the thing it lets through is
exactly the thing it was written to stop.**

The fix is upstream's policy — fail closed on any quoted token that does not
parse — plus running inspection *before* stripping so the melody-only check can
actually fail.

Defect 14 is the same family as defects 8–11 from a different angle: a check
that exists but cannot fire. It looked like protection and was decoration.

### Where the critic and the builder disagreed

Two of the critic's own flags resolved in the code's favour, and it was right to
mark them unverifiable rather than assert them:

- `schema.SongParameters` was outside its corpus, so it could not confirm
  `lyrics_supplied` existed. It does, and `tests/test_mode_wiring.py` now proves
  the wire end to end.
- It noted our child writes `result.json` where upstream writes `failure.json`.
  That is a difference in convention, not a defect: our parent reads one file for
  both outcomes, which is simpler and is what the tests pin.

## The unification experiment — first run produced a false negative

Two model stacks in this image look incompatible but are not, by their own
declarations: torch declares no numpy constraint, transformers declares no torch
constraint, SheetSage2's `config.json` records a *tested-with* version rather
than a requirement, and its code contains no numpy calls at all. Only one
structural risk remained — an internal `transformers` import path.

So `probe_unified_stack.py` was written to answer it empirically, in the image.
**Its first run answered wrongly, for two reasons that were both bugs in the
probe**, and its printed verdict was read as an answer before either was spotted:

1. It listed `transformers.models.bart.modeling_bart.BartDecoder` as a *module*
   and passed it to `importlib.import_module`. `BartDecoder` is a **class**.
   `import_module` takes a module path, so this raised
   `ModuleNotFoundError: 'modeling_bart' is not a package` — in every
   environment, at every version, permanently. It reported a structural
   incompatibility that did not exist.
2. The Dockerfile ran it with `--offline`, in an image where nothing is cached
   yet. It could never fetch SheetSage2's code at all, so the code path under
   test was never reached.

Both produced the same printed line: *"VERDICT: it does not. The pinned split
stack is required; do not unify."* That was not evidence.

**The lesson is sharper than the previous ones.** A probe that lies is worse
than no probe, because it converts "unknown" into "no" — and "no" is an answer
that stops investigation. The earlier failures this stage found were checks that
*could not fail*; this was a check that failed for the wrong reason and was
believed.

Fixed: the internal is probed as module-plus-attribute (the module importing is
not the same claim as the class surviving), and the probe runs with network
access, since the build has it and downloads the model wheel a few steps above.
`tests/test_probe_unified_stack.py` pins both, including that the attribute check
can fail.

**The unification question is therefore still open.** It has not been tested yet.

### Second run: the probe worked, and found an absent package, not a conflict

With both bugs fixed the probe ran properly — `BartDecoder` now resolves, and the
SheetSage2 fetch completed (25 files). It then reported:

```
FAIL  ModuleNotFoundError: No module named 'torchaudio'
VERDICT: it does not. The pinned split stack is required; do not unify.
```

**That verdict again overstated its evidence.** `torchaudio` was not
*incompatible* — it was **absent**, because nothing in the main environment had
ever needed it (SheetSage2 always lived in its own venv). Checked: torchaudio
2.10.0 ships for cu128/cp311, and SheetSage2 uses only `torchaudio.info`,
`.load` and `.functional.resample` — stable APIs. So the answer was to install
the package, not to abandon the question.

Fixed twice over:

1. **`torchaudio==2.10.0` added to `worker/requirements.txt`**, version-matched
   to torch. The probe can now complete instead of stopping at the first absent
   import.
2. **The probe distinguishes `missing` from `structural`.** A
   `ModuleNotFoundError` means "the test could not complete", and the verdict now
   reads `INCONCLUSIVE` rather than "do not unify". A missing package and a
   version incompatibility both print as failure, but only the second answers the
   question asked.

The pattern is worth naming, because it is now the third instance this stage: the
probe's *logic* was fixed and its *conclusion* was still wrong. Correct execution
of a check is not the same as a correct claim drawn from it — and the gap is
invisible from the check's own output, which looks equally confident either way.

## What this stage does **not** prove

- **Neither venv has ever been built.** They are created in the RunPod image
  build; the local suite mocks the subprocess boundary entirely.
- **No transcription has run.** SheetSage2 and Qwen3-ASR have not been invoked.
- **The cover chain is unverified end to end** — including whether SheetSage2's
  prompt set returns the ABC shape our validator accepts.
- **VRAM has not been sampled.** The sequential-loading design's whole premise is
  that no stage exceeds the `create` peak; that needs a GPU to confirm.

### The Human check

> Run one `cover` job and one `edit` job end-to-end on the same 24 GB GPU tier as
> `create`, and confirm via VRAM sampling that no stage exceeded the create-job
> peak (14.08 GiB) before calling this stage complete.

## Open questions for Stage 03

- **The `.env.example` and endpoint config need the venv path** and any new
  variables (`YUE2_VENV_ROOT`).
- **SheetSage2's own `requirements.txt` is fetched at build time** from the HF
  repo rather than pinned here — it is not published on PyPI and its contents may
  change. Pinning it would mean copying it into the repo after a verified build.
- **The Qwen3-ASR torch pin is unknown** (`dependency-pins.md` records no torch
  pin for it), so its venv resolves torch at build time. Worth pinning after the
  first successful build.
