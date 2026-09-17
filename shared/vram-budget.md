# VRAM budget — single 24 GB RTX 4090, sequential not concurrent

> Layer 3 · factory reference, stable across every stage.
> Source of truth: Outline "YuE2-3B Technical Reference" (id
> `29975c75-556c-48a0-be67-f29f3f79e729`).

| Stage | Resident model(s) | Approx. VRAM |
|---|---|---|
| `create` / `edit` generation | YuE2-3B (BF16) + YuE2-Vae (FP32) | 11.18 GiB typical, **14.08 GiB peak** |
| `cover` stage 1 (transcription) | MERT-v2-FullSong (632M, F32) + SheetSage2 adapter (57.2M, F32) | weights ≈ 2.7 GiB; activations scale with song length |
| `cover` stage 2 (lyrics ASR) | Qwen3-ASR-1.7B (2B params, BF16) | weights ≈ 4 GiB + KV cache/activation overhead |
| `cover` stage 3 (generation) | YuE2-3B + YuE2-Vae (stages 1–2 already exited) | same as `create`, 14.08 GiB peak |

Because stages 1 and 2 run as separate subprocesses that exit — and fully
release their VRAM — before stage 3 starts, **no job's peak VRAM exceeds
YuE2's own ~14 GiB ceiling**. A single RTX 4090 24 GB has comfortable
headroom at every stage. Running all three model families concurrently
resident (no subprocess boundary) would instead sit around 18–22 GB of
weights alone before KV cache and multi-framework overhead — plausible but
with little safety margin, which is why sequential loading is the chosen
design, not a fallback.

## Performance baseline (create/edit path, measured)

| GPU | CoT | Gen time / audio duration | Peak VRAM |
|-----|-----|---------------------------|-----------|
| RTX 4090 24 GB | full | 71.04 s / 214.85 s (~3.6 min song) | 11.18 GiB |
| RTX 4090 24 GB | melody | 68.68 s / 214.67 s | 11.02 GiB |
| RTX 4090 24 GB | off | 57.91 s / 196.88 s | 11.09 GiB |

One song at a time on the HF pipeline. ~24 GB available host RAM needed for
YuE2 alone — the cover pipeline's extra subprocesses will need some headroom
above that; exact figure TBD (open question, no `cover` baseline measured
yet).
