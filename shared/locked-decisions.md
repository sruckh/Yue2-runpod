# Locked design decisions & deferred scope

> Layer 3 · factory reference, stable across every stage.
> Source of truth: Outline "YuE2-3B RunPod Serverless — Product Requirements"
> (id `ad5ae631-23bd-4735-bc36-cfdaa38f81db`) and "YuE2 Serverless Roadmap"
> (id `b6d45319-5ec4-44c7-a138-ee3a1855d180`).

## Locked decisions

1. Runs **entirely** as a RunPod Serverless endpoint — nothing installed or
   executed on any dev/ops box we control.
2. One endpoint, one worker image, `mode: "create" | "cover" | "edit"` in the
   job payload — not three separate RunPod endpoints.
3. 24 GB VRAM GPU tier (RTX 4090 class) for every mode, via the sequential
   model-loading design in `vram-budget.md`.
4. Model weights (YuE2-3B, YuE2-Vae, SheetSage2, MERT-v2-FullSong,
   Qwen3-ASR-1.7B) are never baked into the container image — cached on a
   RunPod network volume, downloaded on first boot (`snapshot_download`,
   idempotent).
5. Audio egress: Backblaze B2, S3-compatible API via boto3. Job response
   returns URLs + metadata, not raw bytes.
6. ASR for covers: Qwen3-ASR-1.7B via the documented YuE2 pipeline — not the
   project's existing standalone Parakeet RunPod endpoint.

## Deferred / out of scope (Roadmap Phase 3 — no stage folder for these)

- vLLM concurrent serving runtime (needs an 80 GB-class GPU).
- `YuE2-Vae-legacy` decoder — kept env-switchable for benchmark reproduction,
  not the default.
- Multi-datacenter / multi-volume replication.
- Streaming / real-time ASR.
- The existing standalone Parakeet RunPod endpoint (superseded by decision 6
  above).

## A note on RESEARCH-NOTES.md

`/opt/docker/Yue2/RESEARCH-NOTES.md` predates the cover/edit scope decision
and still lists "SheetSage2 covers" as out-of-scope Phase 2 — that has since
been reversed. Treat this `shared/` directory and the Outline docs above as
current; RESEARCH-NOTES.md is a historical snapshot, not corrected in place.
