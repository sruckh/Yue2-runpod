# AGENTS.md

Instructions for AI coding agents working in this project.

<!-- outline:global-rules (managed by the outline skill) -->
## Global Agent Rules

The shared Global Agent Rules for this brain are imported below. They are
refreshed from Outline into `.outline/global-rules.md` at session start — edit
them in the Outline "Global Agent Rules" page, not here.

@.outline/global-rules.md
<!-- /outline:global-rules -->

## Build & deploy — read before running anything

**Nothing is built or executed locally.** Locked decision 1: this project *"runs
entirely as a RunPod Serverless endpoint — nothing installed or executed on any
dev/ops box we control."* That includes `docker build`. Do not install build
packages on this machine either.

The image is built by **RunPod's platform** from this GitHub repository. After a
push, watch the endpoint's **Builds** tab — that is the real build signal.

Because of that rule, everything below is a *static* check that needs no GPU, no
Docker and no network:

```bash
python -m pytest tests/ -q                    # GPU and B2 mocked
python -m py_compile worker/*.py              # syntax, without importing
ruff check worker tests
```

**Never `import worker/handler.py` to check it.** `handler` boots the pipeline
and calls `runpod.serverless.start` at module scope — the standard RunPod worker
shape — so importing it downloads model weights and tries to start a server. Use
`py_compile`. This has already gone wrong once: a Dockerfile `import handler`
step hydrated the volume inside a build layer and baked ~12 GB of weights into
the image.

### Layout RunPod expects

| File | Location | Why |
|---|---|---|
| `Dockerfile` | **repo root** | RunPod's GitHub build looks there by default (a path is configurable, but root needs no configuration) |
| Entrypoint | `CMD ["python", "-u", "/app/handler.py"]` | Runs the handler, which ends by calling `runpod.serverless.start` |
| Handler | `worker/handler.py` | Copied into the image at `/app` |

### DOX

`worker/AGENTS.md` is the contract for the service code. Read it before editing
anything under `worker/`.

