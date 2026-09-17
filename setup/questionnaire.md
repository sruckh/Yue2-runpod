# Setup questionnaire — YuE2 RunPod Serverless Worker

Answer once. These configure the **factory** (this workspace), not a single run.
Every answer is written into a Layer 3 file that every future run reads, so no
run should ever have to ask them again.

1. **Goal** — what repeatable work does this workspace do? (→ `IDENTITY.md`)
2. **Repeating unit** — a run, a record, a node, a body of work? (→ chooses the form)
3. **Audience & done** — who reads the deliverable, and what does finished look
   like? (→ `_config/definition-of-done.md`)
4. **Voice** — tone and formality; paste two examples that sound right and one
   that sounds wrong. (→ `_config/voice.md`)
5. **Hard constraints** — length, format, brand, compliance; the rules that
   never bend. (→ `_config/conventions.md`)
6. **Steps** — walk one run start to finish. Where do you stop and check
   something before continuing? (→ stage boundaries)
7. **Human checks** — what does a person always verify before anything ships?
   (→ each contract's `## Human check`)
8. **Reuse** — what already exists that runs should reuse: templates, examples,
   data sources? (→ linked from `_config/`, one home per fact)
