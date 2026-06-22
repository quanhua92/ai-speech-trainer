# Docker

Ship the whole thing — API + interactive demo + baked models + default
references — as one image you can `docker-compose up`.

## Quick start

```bash
docker compose up --build      # build (if needed) and run
# → open http://127.0.0.1:8000/demo
```

The first build installs dependencies (torch ~0.5 GB wheel included) and bakes
the Kokoro (~330 MB) + Wav2Vec2 (~1.2 GB) models into the image. Subsequent
builds reuse those cached layers. The container starts with four default
Kokoro references already populated, so the demo is usable the instant it's up —
fully offline.

## What's in the image

| Path | Contents |
| --- | --- |
| `/app` | The installed package (`ai_speech_shadowing`) |
| `/app/data/references/` | Bundled default references (shipped in git) |
| `/models` | Pre-warmed HF cache (`HF_HOME=/models`): Kokoro-82M + Wav2Vec2 |
| Entrypoint | `ai-speech-shadowing serve --host 0.0.0.0 --port 8000` |

## Volumes & persistence

`docker-compose.yml` mounts a named volume `app-data` at `/app/data`. On first
create, Docker seeds it from the image (so the bundled defaults appear), then
user recordings, history, and any newly-generated references persist across
restarts. The model cache (`/models`) is **baked in, not a volume** — that's
what makes `up` work offline instantly.

## Faster local builds (pre-warm the model cache)

The models are almost certainly already in your HuggingFace cache if you've run
the app on the host. Copy them into `data/models/` and the build skips the
~1.5 GB download:

```bash
mkdir -p data/models/hub
cp -R ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M data/models/hub/
cp -R ~/.cache/huggingface/hub/models--facebook--wav2vec2-lv-60-espeak-cv-ft data/models/hub/
docker compose build
```

`data/models/` is gitignored (machine-specific, huge) — it only affects local
builds. A clean build (CI, a fresh server) with an empty `data/models/` falls
back to downloading during `docker build` via `scripts/prewarm_models.py`.

> Symlinks (`ln -s ~/.cache/huggingface ...`) don't work here — Docker won't
> follow symlinks that escape the build context, so a real copy is required.

## Platform notes

- **Apple Silicon:** Docker Desktop builds native `linux/arm64` images (fast).
  MPS GPU acceleration isn't available inside the container — inference runs on
  CPU, which is fine for a demo.
- **Linux x86_64 server:** the same `Dockerfile` / `compose` builds `linux/amd64`
  natively (the lockfile pins both arches' torch wheels). This is the primary
  deployment target.

## Useful commands

```bash
docker compose up --build       # build + run
docker compose logs -f          # tail logs
docker compose down             # stop + remove container (volume kept)
docker compose down -v          # also wipe the app-data volume
```

The Justfile wraps these: `just compose-up`, `just compose-logs`,
`just compose-down`.
