# Model facts — YuE2-3B, SheetSage2, MERT-v2-FullSong, Qwen3-ASR-1.7B

> Layer 3 · factory reference, stable across every stage.
> Source of truth: Outline project "Yue2" (canonical) — "YuE2-3B Model Overview
> & Licensing" (id `2bb1f7cd-18c5-4188-b237-b0fcf77deece`), "YuE2-3B Technical
> Reference — API, Performance & Dependencies" (id
> `29975c75-556c-48a0-be67-f29f3f79e729`) — mirrored in
> `.serena/memories/yue2-model.md`. Do not restate this file's content
> elsewhere; link to it instead.

## YuE2-3B

- `m-a-p/YuE2-3B` on HuggingFace; GitHub `multimodal-art-projection/YuE`,
  release `yue2-v0.1.6`. 4B params, BF16 safetensors (name says "3B"). Tech
  report pending — cite arXiv:2503.08638 (YuE 1) meanwhile.
- Core idea: symbolic planning — plans a melody+chord composition as an ABC
  score, then realizes it as a complete song (vocals + accompaniment), 48 kHz
  stereo, no quantization.
- Pipeline API:
  ```python
  from yue2 import YuE2Pipeline
  pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda")
  song = pipe(style=..., lyrics=..., cot=..., abc=..., seed=..., cfg_scale=...)
  song.save("x.flac")            # 48 kHz stereo FLAC
  song.save_artifacts(dir)       # + score.abc, tokens, latents, settings
  ```
  - `cot`: `"full"` (melody+chord plan, default) · `"melody"` (covers — does
    **not** strip chord symbols from a supplied ABC; strip manually or use
    `cot="full"` to keep supplied harmony) · `"off"` (direct generation).
  - `abc=`: supply your own score.
  - Edit loop: `pipe.plan(**request)` → `plan.save(dir)` → edit `score.abc` →
    regenerate with `abc=` + `cot="full"`. Editing re-renders the whole song;
    the waveform outside the edit is NOT preserved.
- Weight layout — **two** HF repos, both must be cached: `m-a-p/YuE2-3B`
  (`model.safetensors`, `config.json`, `modeling_yue2.py` — custom code,
  requires `trust_remote_code`, `qwen.tiktoken`, generation configs,
  `weights_manifest.json`) and `m-a-p/YuE2-Vae` (default decoder;
  `YuE2-Vae-legacy` exists only for the published benchmark protocol — not
  the default).
- Licensing: weights CC BY-NC 4.0 + additional creator permission —
  individuals/creators may use and monetize outputs royalty-free; academic
  non-commercial use free; **commercial use by companies requires contacting
  the YuE2 authors**. Code/docs/agent skill: Apache 2.0.

## SheetSage2 (cover transcription, stage 1 of the cover pipeline)

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True).eval().to("cuda")
result = model.transcribe("song.mp3", output_dir="cover-score", melody_only=True)
abc = result["abc"]  # chord-free melody ABC, also saved as cover-score/score.abc
```

- `melody_only=True` keeps both `Vocal` and `Ins` melodies, omits chord
  symbols — exactly the score shape YuE2's cover path (`cot="melody"`) wants.
- Built on `m-a-p/MERT-v2-FullSong` (632M params, F32) as a base model, with a
  57.2M-param F32 adapter that merges automatically on load — **both** repos
  must be cached.
- Requires Python 3.10 or 3.11 + FFmpeg 6.1 with shared libraries. Its own
  `requirements.txt` pins `torch==2.8.0`, `torchaudio==2.8.0` (cu126 index),
  `transformers==4.45.2`, `numpy==1.24.3`, `scipy==1.13.1`, `mir_eval==0.8.2`,
  `pretty_midi==0.2.10`, `mido==1.3.3`, `setuptools==78.1.1`.
- **Those pins are not binding.** SheetSage2 runs correctly on the main
  environment's torch 2.10.0 / transformers 4.57.6 / numpy 2.2.6 — verified on
  hardware 2026-09-19 (96–168 note melodies, chord symbols correctly absent
  under `cot="melody"`). The pins describe what it was tested with upstream.
- Weights: CC BY-NC 4.0. Not covered by YuE2's license grant — confirm terms
  separately before any commercial use.

## Qwen3-ASR-1.7B (cover lyric transcription, stage 2 of the cover pipeline)

```python
import torch
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0",
    max_inference_batch_size=32, max_new_tokens=256,
)
results = model.transcribe(audio="song.mp3", language=None)  # auto language ID
text = results[0].text
```

- Despite the "1.7B" name, the HF repo reports **2B params, BF16**
  safetensors.
- Use the **transformers backend** (`pip install qwen-asr`), not vLLM — no
  concurrency need at our single-job-at-a-time scale.
- Handles singing voice and full songs with BGM directly — no vocal-isolation
  preprocessing step required.
- Chosen over the project's existing standalone Parakeet RunPod endpoint, to
  follow the documented YuE2 cover pipeline exactly (locked decision,
  2026-09-17).
- No hard `torch`/`transformers` pin documented on the model card — see
  `dependency-pins.md` open question.

## Cover pipeline shape (why two transcription steps exist)

A `cover` job needs TWO transcription steps before YuE2 generates:
1. audio → melody ABC via SheetSage2 (`melody_only=True`)
2. audio → lyrics text via Qwen3-ASR-1.7B

YuE2 itself only does step 3, generation with `cot="melody"` and the
transcribed score + lyrics + the requested target style.
