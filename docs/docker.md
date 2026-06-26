# Docker

Ship the whole thing — API + interactive demo + baked models + default
references — as one image you can `docker-compose up`.

## Quick start

```bash
docker compose up --build      # build (if needed) and run
# → open http://127.0.0.1:8000/
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
| `/app/data/storage/` | Leaderboard `db.json` + `hashes/` (created at runtime) |
| `/app/docker/` | `nginx.conf` + `supervisord.conf` (the in-container supervisor/proxy) |
| `/models` | Pre-warmed HF cache (`HF_HOME=/models`): Kokoro-82M + Wav2Vec2 |
| Entrypoint | `supervisord -c /app/docker/supervisord.conf` (PID 1) |

## In-container architecture (supervisord → nginx → uvicorn)

The container runs **one process tree**, all as non-root `appuser`:

```
:8000 → nginx  (cookie-sticky: hash $cookie_user_id consistent)
            ├─ 127.0.0.1:8001  uvicorn-1  (WORKERS=1)
            └─ 127.0.0.1:8002  uvicorn-2  (WORKERS=1)
supervisord (PID 1) manages all three; restarts any that crash.
```

- **`supervisord`** is PID 1 (`docker/supervisord.conf`). It launches nginx +
  two single-worker uvicorns and revives any that crash. On `docker stop`
  (SIGTERM) it stops the uvicorns first (`stopsignal=TERM`) so each runs its
  lifespan shutdown → the final leaderboard flush.
- **`nginx`** (`docker/nginx.conf`) listens on `:8000` and routes by the
  `user_id` cookie: `hash $cookie_user_id consistent`. A cookie-less first visit
  is routed on `$request_id` (round-robin) until the app sets the cookie. It
  forwards `Upgrade`/`Connection` (WebSocket/ASGI-aware) and the usual
  `X-Forwarded-*` headers. Plain HTTP inside — TLS is terminated by the external
  prod proxy.
- **uvicorn** workers are launched with the existing CLI
  (`ai-speech-shadowing serve --host 127.0.0.1 --port 800X`, `WORKERS=1`) so they
  reuse the logging config + lifespan (history/reference/leaderboard tasks).

Each worker loads its own ~2 GB of models and keeps its own in-memory
leaderboard cache; they share only the filesystem. The number of workers is the
count of `[program:uvicorn-N]` entries (2) — there is no `WORKERS` env var
anymore. Memory ≈ workers × ~2 GB; the compose `mem_limit: 8g` covers 2 with
headroom. Stickiness gives each user an instant *own*-count; the global
leaderboard view is still cross-worker (see [db.md](db.md#eventual-consistency)).

## Volumes & persistence

`docker-compose.yml` mounts a named volume `app-data` at `/app/data`. On first
create, Docker seeds it from the image (so the bundled defaults appear), then
user recordings, history, and any newly-generated references persist across
restarts. The model cache (`/models`) is **baked in, not a volume** — that's
what makes `up` work offline instantly.

## Offline mode (`HF_HUB_OFFLINE=1`)

Because every model is baked into `/models` at **build time**
(`scripts/prewarm_models.py` runs during `docker build`), the container sets
`HF_HUB_OFFLINE=1` in `docker-compose.yml`. This tells `huggingface_hub` /
`transformers` to use only the local cache and **skip the per-file etag/HEAD
validation** against `huggingface.co` that `from_pretrained` performs even when
files are fully cached. That validation otherwise costs **~5-6 s per cold model
load** (paid once per worker); offline mode drops it to ~0.01 s.

| Path | Offline behaviour |
| --- | --- |
| **Docker** | `HF_HUB_OFFLINE=1` is set in `docker-compose.yml` — always safe, models are baked. |
| **`scripts/serve.sh`** (local dev) | A pre-flight checks `$HF_HOME` (default `~/.cache/huggingface`) for the 3 required models. If all present → enables offline. If any missing → runs `scripts/prewarm_models.py` **once in the foreground** (same as the build), then enables offline. |

> Offline mode fails fast ("file not found in cached path") if a model isn't
> cached — so `serve.sh` prewarms *before* going offline rather than downloading
> in the background (a background download would race the offline-mode server
> and fail the first request). To force a re-download locally, clear the cache
> dir and re-run `serve.sh`, or `unset HF_HUB_OFFLINE` for a one-off online run.

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

Both arches are verified working. The Dockerfile pins no `platform` — it builds
natively for whatever the host is.

- **Apple Silicon (arm64):** praat-parselmouth has no aarch64 wheel, so the
  builder stage compiles it from source (`build-essential` + `cmeta` + `ninja`).
  MPS GPU acceleration isn't available inside the container — inference runs on
  CPU, which is fine for a demo.
- **Linux x86_64 server (amd64):** praat-parselmouth uses the prebuilt
  manylinux wheel (no compilation). This is the primary deployment target.

## Useful commands

```bash
docker compose up --build       # build + run
docker compose logs -f          # tail logs
docker compose down             # stop + remove container (volume kept)
docker compose down -v          # also wipe the app-data volume
```

The Justfile wraps these: `just compose-up`, `just compose-logs`,
`just compose-down`.
