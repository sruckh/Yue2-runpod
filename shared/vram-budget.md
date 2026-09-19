# VRAM budget — one 24 GB card, sequential not concurrent

> Layer 3 · factory reference, stable across every stage.
>
> Published figures below come from Outline "YuE2-3B Technical Reference" (id
> `29975c75-556c-48a0-be67-f29f3f79e729`). **Measured** figures come from this
> worker's own jobs, sampled device-wide per phase by `worker/vram.py`. The two
> are labelled, because they disagree and the reason matters (see *Length
> dependence*).

## The design claim, and that it holds

Each model family runs as its **own subprocess** that exits — releasing its VRAM
to the driver — before the next starts. So no job's peak is the sum of the model
families it uses.

**This is now measured, not just designed.** From cover job `64f6c711` (endpoint
v28, 56 s song):

    per_phase {job: 702, transcribe: 5426, generate: 8788, upload: 906}
    peak 8788 MiB      <- equals the generate phase, not transcribe + generate
    if concurrent: 5426 + 8788 = 14214 MiB

The peak equals its largest phase, never their sum. A cover costs **no more VRAM
than a create of the same song** despite running two extra model families.

| Phase | Resident model(s) | Measured |
|---|---|---|
| `transcribe` (cover/edit input) | MERT-v2 (632M, F32) + SheetSage2 adapter (57M, F32), then Qwen3-ASR-1.7B (BF16) | 4446–5764 MiB, scaling with song length |
| `generate` (all modes) | YuE2-3B (BF16) + YuE2-Vae (FP32) | 8403–9348 MiB, scaling with song length |
| `upload` | none | 675–1214 MiB |

Two transcription families run in that one phase, sequentially, each exiting
before the next — so `transcribe` reports the larger of the two, not both.

## Measured peaks, by mode and song length

All on **NVIDIA L4**, `device_total_mib = 23034`, baseline 272–756 MiB.

| Mode | Song | Peak | Note |
|---|---|---|---|
| `create` | 37 s | 8423 MiB | |
| `create` | 53.6 s | 8486 MiB | |
| `create` | 56.9 s | 8494 MiB | |
| `create` | 197 s | 9072 MiB | same length as the edit below |
| `cover` | 56.2 s | 8788 MiB | transcribe 5426 |
| `cover` | 75.9 s | 8800 MiB | transcribe 5764 |
| `edit` | 198.8 s | 9348 MiB | supplied score, no transcription |

Headroom on the worst measured run: **23034 − 9348 = 13686 MiB**, i.e. a 24 GB
card is not close to its limit at any song length observed.

## Length dependence — why the published 14.08 GiB disagrees

The published figure is **11.18 GiB typical, 14.08 GiB peak**, measured on an
RTX 4090 at full context with a **~3.6-minute** song. Our measured peaks are
lower because our songs have been shorter:

    create,  57 s song -> 8494 MiB
    create, 197 s song -> 9072 MiB
    cover,  199 s song -> 9270 MiB

Peak VRAM scales with song length. **Compare like with like** — a 8.5 GiB reading
from a one-minute song is not evidence that a 3.6-minute song fits in 9 GiB.

## The GPU pool is not homogeneous

The same endpoint has reported `device_total_mib` of **23034** and **24564** from
cards sold as the same 24 GB tier, and host RAM has varied between **126 GB** and
**507 GB** across workers. The compute benchmark has varied ~3×.

So: **quote the card and the device total alongside any peak.** A number without
them cannot be compared against another number.

`mig.mode.current` returns `[N/A]` on this driver, so MIG cannot be confirmed or
excluded from the card's own report. If a MIG instance is ever used, treat its
figures as a separate population — a MIG slice may report the parent device's
memory, which would make a peak describe a machine the job did not run on.

## Host RAM — the open question, now closed

This section previously read *"~24 GB available host RAM needed for YuE2 alone —
the cover pipeline's extra subprocesses will need some headroom above that; exact
figure TBD (open question, no cover baseline measured yet)."*

**Measured: not a constraint on this hardware.** Workers have reported 407–421 GB
available of ~503 GB total — an order of magnitude above the requirement. Every
cover run to date has completed without memory pressure, with both transcription
families and the pipeline in sequence.

## Running all three families concurrently — still untested, still not done

The design keeps the subprocess boundary for VRAM release. Running everything
resident would put roughly 14 GiB (YuE2 + VAE) plus the transcription families'
weights together before KV cache and multi-framework overhead. That was estimated
at 18–22 GB and is *plausible* on a 24 GB card — but it has never been measured,
it is not the design, and nothing needs it.
