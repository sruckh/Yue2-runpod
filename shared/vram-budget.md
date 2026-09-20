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

| Mode | Song | Peak | Card / device total | Note |
|---|---|---|---|---|
| `create` | 37 s | 8423 MiB | L4 · 23034 | |
| `create` | 53.6 s | 8486 MiB | L4 · 23034 | |
| `create` | 56.9 s | 8494 MiB | L4 · 23034 | |
| `create` | 197 s | 9072 MiB | L4 · 23034 | same length as the edit below |
| **`create`** | **281.7 s** | **9628 MiB** | **RTX 4090 · 24564** | **longest song measured** |
| `cover` | 56.2 s | 8788 MiB | L4 · 23034 | transcribe 5426 |
| `cover` | 75.9 s | 8800 MiB | L4 · 23034 | transcribe 5764 |
| `edit` | 198.8 s | 9348 MiB | L4 · 23034 | supplied score, no transcription |

Baselines ranged 272–756 MiB on the L4 and 510 MiB on the 4090.

Headroom on the worst measured run: **24564 − 9628 = 14936 MiB**. On the smaller
card, 23034 − 9348 = 13686 MiB. Neither is close to its limit at any song length
observed.

**The 281.7 s run is the first data point past the earlier ceiling, and it moves
it by only ~550 MiB** (9072 → 9628) for a song 43% longer. Growth is real but
sub-linear in this range, which is what makes a 24 GB card comfortable rather
than marginal.

## Length dependence — and why it does not explain the published figure

The published figure is **11.18 GiB typical, 14.08 GiB peak**, measured on an
RTX 4090 at full context with a **~3.6-minute** song. Our measured peaks are
lower:

    create,  57 s song -> 8494 MiB
    create, 197 s song -> 9072 MiB
    create, 282 s song -> 9628 MiB
    cover,  199 s song -> 9270 MiB

Peak VRAM scales with song length. **Compare like with like** — an 8.5 GiB
reading from a one-minute song is not evidence that a 3.6-minute song fits in
9 GiB.

**But length alone does not close the gap, and an earlier version of this section
implied it did.** The 282 s create ran on an **RTX 4090 — the same card as the
published benchmark — with a song longer than the published 3.6 minutes**, and
peaked at 9628 MiB: **~4.4 GiB below** the 14.08 GiB figure.

So the working hypothesis that "our songs are simply shorter" is not sufficient.
The remaining difference is most likely **context length rather than song
duration**: the published run is described as *full context*, and
`max_seconds`/context is a separate knob from how long the rendered song is.
That is unconfirmed — we have one data point on a 4090 and no run at a
deliberately extended context. **Do not treat 14.08 GiB as refuted**; treat it as
a ceiling for a configuration we have not reproduced.

What is safe to say: at every song length we have measured, on both cards, the
worst peak is **9628 MiB against a 24564 MiB device**.

## The GPU pool is not homogeneous

The same endpoint has reported `device_total_mib` of **23034** and **24564**, and
host RAM has varied between **126 GB** and **507 GB** across workers. The compute
benchmark has varied ~3×.

**A refinement from the 282 s run:** `device_total_mib` now has a known card on
each side — **23034 on an NVIDIA L4**, **24564 on an NVIDIA GeForce RTX 4090**.
Earlier versions of this file called those "two cards sold as the same 24 GB
tier", which framed the difference as noise within a tier. If the total tracks
the card, it is the opposite: a **reliable discriminator** for which card a job
actually ran on. Two observations is not a mapping — but it is enough to stop
calling it inconsistent, and enough to keep recording the field.

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
