# REST API (Phase 8A)

> **Phase 8A deliverable.** Exposes the evaluation engine over HTTP via FastAPI.
> All routes are versioned under `/api/v1`; OpenAPI docs are auto-generated at
> `/docs` and `/redoc`.
>
> **Phase 8B** (the TanStack Start web UI) is deferred to a later phase — this
> API is the complete, stable contract any frontend (or third-party client) can
> build against.

## Run the server

```bash
ai-speech-shadowing serve                       # http://127.0.0.1:8000
ai-speech-shadowing serve --port 8000 --reload  # dev mode
```

Programmatically:

```python
import uvicorn
from ai_speech_shadowing.api.app import create_app

uvicorn.run(create_app(), host="127.0.0.1", port=8000)
```

CORS is enabled for `localhost:3000` / `localhost:5173` (frontend dev servers).

## Interactive demo

A single-page, dependency-free demo is served at **`GET /demo`** (not under
`/api/v1` — it's a page, not an API resource). It showcases every feature with
vanilla JS: mic **record-to-WAV** (Web Audio API + in-browser PCM encode) or
`.wav` upload, **reference** generation + list/playback, **full and quick
evaluation**, **colour-coded phoneme diff**, score cards, feedback, and
**history** browsing — all against the live `/api/v1/*` endpoints.

```bash
ai-speech-shadowing serve   # then open http://127.0.0.1:8000/demo
```

Source: [`src/ai_speech_shadowing/api/demo.html`](../src/ai_speech_shadowing/api/demo.html).
The full TanStack Start SPA is deferred to Phase 10.

## Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service & model load status |
| `POST` | `/evaluate` | Evaluate user audio against a pre-generated reference |
| `POST` | `/evaluate/quick` | Evaluate against a TTS reference generated on-the-fly |
| `POST` | `/references` | Generate a TTS reference from text |
| `GET` | `/references` | List all references |
| `GET` | `/references/{slug}` | Reference metadata |
| `GET` | `/references/{slug}/audio` | Stream the reference WAV |
| `DELETE` | `/references/{slug}` | Delete a reference |
| `GET` | `/history` | Paginated past evaluations (`?limit=&offset=&sort=`) |
| `GET` | `/history/{id}` | Full evaluation detail |
| `GET` | `/history/stats` | Aggregated progress statistics (`?period_days=30`) |

### Health

```bash
curl localhost:8000/api/v1/health
```
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models": {
    "wav2vec2": {"loaded": false, "load_time_ms": null},
    "tts":      {"loaded": false, "load_time_ms": null}
  }
}
```

Models load lazily on first use; after the first `/evaluate`, both report
`loaded: true` with their measured load times.

### Generate a reference

```bash
curl -X POST localhost:8000/api/v1/references \
  -H 'Content-Type: application/json' \
  -d '{"text": "The quick brown fox", "language": "en", "speaker": "default"}'
```
```json
{
  "id": "the-quick-brown-fox",
  "text": "The quick brown fox",
  "language": "en-us",
  "speaker": "af_heart",
  "duration_seconds": 2.41,
  "audio_url": "/api/v1/references/the-quick-brown-fox/audio",
  "created_at": "2026-06-22T..."
}
```

`speaker: "default"` is the API sentinel — it resolves to the configured Kokoro
voice (`af_heart`). Any other value is passed straight to Kokoro
(`am_adam`, `jf_alpha`, …). `language` is ISO-ish (`en`, `ja`, `zh`, …) and is
mapped to Kokoro's internal single-letter codes.

### Evaluate

```bash
curl -X POST localhost:8000/api/v1/evaluate \
  -F 'audio=@user.wav' \
  -F 'reference_id=the-quick-brown-fox'
```

`/evaluate/quick` skips the pre-generated reference — send `text` + `audio` and
the server synthesises the reference on the fly:

```bash
curl -X POST localhost:8000/api/v1/evaluate/quick \
  -F 'audio=@user.wav' -F 'text=Hello world' -F 'language=en'
```

Response (`EvaluationResponse`):

```json
{
  "id": "eval_a1b2c3d4",
  "created_at": "2026-06-22T...",
  "reference_id": "hello-world",
  "scores": {
    "pronunciation": {"phoneme_error_rate": 0.12, "score": 88, "grade": "good"},
    "intonation":    {"pitch_range_ratio": 0.68, "monotone": false, "score": 62, "grade": "fair"},
    "fluency":       {"dtw_normalized_distance": 0.05, "syllable_rate": 3.2, "pause_count": 1, "score": 81, "grade": "good"},
    "composite":     {"score": 77, "grade": "fair"}
  },
  "phoneme_diff": [
    {"type": "match", "phoneme": "h"},
    {"type": "sub",   "expected": "l", "actual": "ɹ"}
  ],
  "feedback": ["Phoneme /l/ was substituted with /ɹ/ — focus on tongue placement."]
}
```

Every evaluation is persisted to the history store automatically.

### History & stats

```bash
curl 'localhost:8000/api/v1/history?limit=10&offset=0&sort=desc'
curl   localhost:8000/api/v1/history/eval_a1b2c3d4
curl 'localhost:8000/api/v1/history/stats?period_days=30'
```

`/history/stats` returns `total_evaluations`, per-pillar averages, a coarse
`trend` (`improving` / `steady` / `declining` / `insufficient`), the most
frequently mispronounced `weakest_phonemes`, and a `daily_breakdown`.

## Architecture

```
api/
├── app.py            # create_app(): FastAPI factory, CORS, /api/v1 mounting
├── deps.py           # EngineState singleton (lazy extractor + load timing, refs, history)
├── schemas.py        # Pydantic models + report → EvaluationResponse adapter
└── routes/
    ├── health.py
    ├── evaluate.py   # /evaluate, /evaluate/quick (multipart upload)
    ├── reference.py  # CRUD + audio streaming
    └── history.py    # list / detail / stats
```

### Design decisions

- **Lazy models.** The Wav2Vec2 phoneme extractor and the Kokoro pipeline load
  on first request, not at import — so `uvicorn` starts in milliseconds and
  `/health` reports the real `loaded` + `load_time_ms`.
- **One extractor per process.** `EngineState` caches the phoneme extractor, so
  concurrent evaluations reuse it.
- **Sync handlers.** The CPU-bound endpoints (`/evaluate*`, `/references`) are
  plain `def` — FastAPI runs them in its threadpool, so the event loop never
  blocks on inference.
- **`speaker="default"` sentinel.** Keeps the spec's ergonomic default while
  mapping to Kokoro's real voice names.
- **History = JSON files.** No DB; the same store the CLI `report` command and
  the future web UI read from, trivially greppable and inspectable.
- **Versioned prefix.** Everything under `/api/v1` so future breaking changes
  can land under `/api/v2` without disturbing existing clients.

## End-to-end test

[`scripts/test_e2e.py`](../scripts/test_e2e.py) starts a real uvicorn server in
a thread and exercises the full flow over HTTP. To make the comparison
meaningful (not a trivially-identical match), it creates **two near-identical
references** — `"The quick brown fox…"` (reference) vs `"A quick brown fox…"`
(attempt, one word changed) — and feeds B's real Kokoro audio as the user
upload. The result is a high-but-not-perfect score (~90/100) with a non-zero
PER, proving the phoneme diff genuinely fires.

Flow: health → create reference A + B → list → download B's audio as the
attempt → `/evaluate/quick` (A vs B) → `/evaluate` (A vs B) → history list/
detail → stats → delete.

```bash
uv run python scripts/test_e2e.py
```

## Test coverage

- `tests/test_api.py` (fast, `TestClient`): health, references list/get-missing/
  delete-missing, history list/pagination/stats/get-missing, validation (422 on
  missing fields / empty text). The engine state is pointed at `tmp_path`.
- `tests/test_api.py::TestEvaluateFlow` (opt-in slow): `/evaluate/quick` on a
  Kokoro clip → asserts the full response shape, that the reference is now
  listed, history recorded it, stats reflect it, and `/health` flips to loaded.
