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
| Qwen3-ASR-1.7B (`qwen-asr` 0.0.6 `pyproject.toml`, fetched from GitHub) | **unpinned** (no torch pin at all) | `==4.57.6` — **identical to YuE2's pin** | `accelerate==1.12.0` (vs YuE2's `1.13.0` — a one-patch-version clash, not a major gap), Python `>=3.9` |

SheetSage2/MERT-v2-FullSong (torch 2.x/2.6-2.8, transformers 4.45-4.53,
numpy 1.24) are mutually incompatible with YuE2 (torch 2.10, transformers
4.57.6, numpy 2.2.6) — a real venv-isolation case. **Qwen3-ASR is a
different story**: its `transformers` pin exactly matches YuE2's, and it has
no `torch` pin at all — the only clash is `accelerate` (1.12.0 vs 1.13.0),
a single patch version apart. It could plausibly share YuE2's venv with an
`accelerate` version compromise, rather than needing full isolation.
**Design decision stands as subprocess isolation for Qwen3-ASR too**, but for
the VRAM-residency reason alone (see `vram-budget.md`), not the dependency
conflict originally assumed — worth re-litigating in Stage 03 if the VRAM
argument changes (e.g. if Qwen3-ASR turns out cheap enough to keep resident
alongside YuE2 within budget).

## Design decision: one venv per model family, run as a subprocess

Each model family gets its own venv inside the worker image, invoked as a
**subprocess** by the main handler process (which runs YuE2 in-process, since
YuE2 is resident for every job). A subprocess boundary buys two things:

1. **Guaranteed VRAM release on exit** — no reliance on in-process
   `del model; torch.cuda.empty_cache()` correctness, which is fragile with
   custom `trust_remote_code` model classes.
2. **Dependency isolation** — YuE2's `torch==2.10.0` never has to coexist with
   SheetSage2's `torch==2.8.0` or MERT-v2-FullSong's `torch==2.6.0` /
   `transformers==4.53.2` in the same interpreter.

Stages communicate via small files on the container's local disk (source
audio in; `score.abc` / `lyrics.txt` out) — not shared memory or IPC.

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
