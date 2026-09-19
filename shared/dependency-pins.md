# Dependency pins & the subprocess-isolation design

> Layer 3 · factory reference, stable across every stage.
> Source of truth: Outline "YuE2-3B Technical Reference" (id
> `29975c75-556c-48a0-be67-f29f3f79e729`).

## Verified pins per model family (fully resolved 2026-09-17)

> **Corrected in Stage 02.** The YuE2 row was missing three pins. The
> authoritative source is the `yue2_infer-0.1.5` wheel's own `Requires-Dist`
> list, read directly from the wheel (unpacked from `m-a-p/YuE2-3B`), not the
> model card — which omits them. `soundfile` is why the image needs
> `libsndfile1` installed; without it `import soundfile` raises at runtime.
>
> Also settled in Stage 02: **FlashAttention is not a dependency.** The model
> card's "uses PyTorch, CUDA graphs, and FlashAttention" reads like a build
> requirement, but the wheel imports no `flash_attn` anywhere, and `triton`
> appears only inside `yue2/fast.py` — the separate `fast` (vLLM) extra, which
> stays out of scope per `locked-decisions.md`. The HF path uses PyTorch SDPA.
> No attention compile step is needed.

| Component | torch | transformers | Other notable pins |
|---|---|---|---|
| YuE2-3B (`yue2_infer` 0.1.5) | `==2.10.0` | `==4.57.6` | `accelerate==1.13.0`, `numpy==2.2.6`, `huggingface-hub==0.36.2`, `safetensors==0.7.0`, `tiktoken==0.12.0`, `soundfile==0.13.1` |
| SheetSage2 (own `requirements.txt`, fetched from HF) | `==2.8.0`, `torchaudio==2.8.0` | `==4.45.2` | `numpy==1.24.3` (1.x!), `huggingface-hub==0.36.0`, `safetensors==0.5.3`, `scipy`, `mir_eval`, `pretty_midi`, `mido` |
| MERT-v2-FullSong (quick-start card — SheetSage2's own pins above take priority when running SheetSage2) | `==2.6.0` | `==4.53.2` | inconsistent with SheetSage2's own `requirements.txt` (4.45.2) — use SheetSage2's pins when running it |
| Qwen3-ASR-1.7B (`qwen-asr` 0.0.6) | **no torch pin** — `accelerate` asks for `>=2.0.0`, so pip took the newest (resolved 2.14.0 while it had its own venv) | `==4.57.6` — **identical to YuE2's pin** | `accelerate==1.12.0` (vs YuE2's `1.13.0` — a one-patch-version clash between two pins that **neither package uses**), Python `>=3.9` |

SheetSage2/MERT-v2-FullSong (torch 2.x/2.6-2.8, transformers 4.45-4.53,
numpy 1.24) were assumed mutually incompatible with YuE2 (torch 2.10,
transformers 4.57.6, numpy 2.2.6) — **and the assumption was wrong.** Both run
correctly on the main stack, verified on hardware; see the design-decision
section below. The version numbers do differ; the consequence does not follow.

**Qwen3-ASR is a
different story**: its `transformers` pin exactly matches YuE2's, and it has
no `torch` pin at all — the only clash is `accelerate` (1.12.0 vs 1.13.0),
a single patch version apart. It could plausibly share YuE2's venv with an
`accelerate` version compromise, rather than needing full isolation.
**Design decision stands as subprocess isolation for Qwen3-ASR too**, but for
the VRAM-residency reason alone (see `vram-budget.md`), not the dependency
conflict originally assumed — worth re-litigating in Stage 03 if the VRAM
argument changes (e.g. if Qwen3-ASR turns out cheap enough to keep resident
alongside YuE2 within budget).

## Design decision: one process per model family — and, as it turned out, one venv total

The **subprocess boundary** is the load-bearing half, and it stands. Each model
family runs as a subprocess invoked by the main handler (which runs YuE2
in-process, since YuE2 is resident for every job). That buys **guaranteed VRAM
release on exit** — no reliance on in-process `del model;
torch.cuda.empty_cache()` correctness, which is fragile with custom
`trust_remote_code` model classes. Measured: a cover's peak equals its largest
phase, never the sum of its families (see `shared/vram-budget.md`).

**The venv half was removed 2026-09-19.** The reasoning above assumed the pins
could not coexist:

    YuE2        torch 2.10.0   transformers 4.57.6   numpy 2.2.6
    SheetSage2  torch 2.8.0    transformers 4.45.2   numpy 1.24.3
    Qwen3-ASR   torch 2.14.0   transformers 4.57.6

They can. Both families were run in the main environment on real hardware and
produced correct output — SheetSage2 a 96–168 note melody with the chord symbols
correctly absent, Qwen3-ASR a near-verbatim lyric transcription. The two venvs
were deleted, taking ~6.5 GB of duplicated torch stack with them.

The apparent blocker was `accelerate`: `yue2_infer` pins `==1.13.0` and
`qwen-asr` pins `==1.12.0`, two exact pins that cannot both hold — but **neither
package imports accelerate**. Zero references in either wheel. A declared-but-
unused pin on both sides, so it blocked on nothing; installing `qwen-asr` with
`--no-deps` sidesteps it.

`run_stage` still takes a venv name. That is deliberate: a future model family
whose pins genuinely conflict can be given its own environment by passing its
name, and the runner will find it. Nothing uses it today.

Stages communicate via small files on the container's local disk (source
audio in; `score.abc` / `lyrics.txt` out) — not shared memory or IPC.

**A caution for whoever adds the next family.** "These pins cannot coexist" is a
claim worth testing before building two images around it. Here it was right
about the versions and wrong about the consequence: the conflicting pin was
unused, and two ~3 GB environments were maintained to satisfy a constraint that
did not exist. The probe that settled it read both wheels with `ast` and looked
for the import rather than reading the dependency metadata.

## Open questions — resolved 2026-09-17

Both prior open questions are now answered (see table above, fetched via
firecrawl from HuggingFace's `resolve/main/requirements.txt` for SheetSage2
and GitHub's `pyproject.toml` for Qwen3-ASR). Findings appended to the
Outline Technical Reference doc (id `29975c75-556c-48a0-be67-f29f3f79e729`).

## New item to verify in Stage 02 (from a context7 spot-check, not an open
question that blocks anything)

`runpod-python`'s current docs (via context7, `/runpod/runpod-python`) show
a built-in `runpod.serverless.VolumeCache` context manager for hydrating a
network-volume cache around model loading — a cleaner alternative to the
hand-rolled `ensure_models()`/`snapshot_download` pattern in
`worker-shape.md`. The core `runpod.serverless.start({"handler": handler})`
API itself has not drifted from what's already documented. Worth evaluating
`VolumeCache` when Stage 02 actually implements `boot.py`, but not required
— the hand-rolled pattern is already a validated, documented approach.
