#!/usr/bin/env bash
# Serve the REST API + demo over HTTPS on 0.0.0.0:8000.
# Generates a self-signed cert on first run (needed for the mic on non-localhost).
#
# Usage:
#   bash scripts/serve.sh                        # foreground, port 8000
#   bash scripts/serve.sh 9000                   # foreground, custom port
#   nohup bash scripts/serve.sh 9000 &           # background (log auto-named)
#
# Foreground prints to the terminal. Background (non-tty, e.g. under nohup)
# redirects to /tmp/ai-speech-shadowing-<PORT>.log so concurrent ports don't
# clobber each other.
#
# Then open https://localhost:8000/  (accept the self-signed cert warning)
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-8000}"
CERT=/tmp/ai-speech-shadowing-cert.pem
KEY=/tmp/ai-speech-shadowing-key.pem

# If the port is held by a previous instance, stop it so this one can bind.
# Idempotent: a no-op when the port is free; force-kills after a short grace.
existing=$(lsof -nP -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$existing" ]; then
  echo "Port ${PORT} in use by PID(s) ${existing}; stopping previous instance..."
  kill $existing 2>/dev/null || true
  for _ in $(seq 1 20); do
    [ -z "$(lsof -nP -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)" ] && break
    sleep 0.25
  done
  remaining=$(lsof -nP -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$remaining" ]; then
    echo "Port still held; force killing PID(s) ${remaining}..."
    for pid in $remaining; do
      if [ -f "/proc/$pid/cmdline" ] && grep -qa 'ai-speech-shadowing\|uvicorn' "/proc/$pid/cmdline" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      else
        echo "WARNING: PID $pid on port ${PORT} is not ai-speech-shadowing; not killing." >&2
      fi
    done
    sleep 0.5
  fi
fi

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
  echo "Generating self-signed TLS cert..."
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY" -out "$CERT" -days 365 \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null
  chmod 600 "$KEY"
  chmod 644 "$CERT"
fi

# ── Pre-flight model cache check ────────────────────────────────────────────
# Run fully offline when the models are already cached: this skips HuggingFace's
# per-file etag/HEAD validation, which otherwise adds ~5-6s to every cold model
# load (paid once per worker). If anything is missing, prewarm once in the
# foreground — the SAME script the Docker build uses (scripts/prewarm_models.py)
# — then go offline. A background download would race the offline-mode server
# and fail the first request, so it stays synchronous.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
REQUIRED_MODELS=(
  "models--slplab--wav2vec2-large-robust-L2-english-phoneme-recognition"
  "models--facebook--wav2vec2-lv-60-espeak-cv-ft"
  "models--hexgrad--Kokoro-82M"
)
_models_cached() {
  for m in "${REQUIRED_MODELS[@]}"; do
    ls -d "$HF_CACHE/hub/$m/snapshots"/*/ >/dev/null 2>&1 || return 1
  done
}
if [ -n "${HF_HUB_OFFLINE:-}" ]; then
  echo "HF_HUB_OFFLINE already set (${HF_HUB_OFFLINE}); respecting it."
elif _models_cached; then
  export HF_HUB_OFFLINE=1
  echo "Models cached in ${HF_CACHE} -> HF_HUB_OFFLINE=1 (offline, fast cold loads)."
else
  echo "Some models missing in ${HF_CACHE}; prewarming (one-time download)..."
  uv run python scripts/prewarm_models.py
  export HF_HUB_OFFLINE=1
  echo "Prewarm done -> HF_HUB_OFFLINE=1."
fi
# ────────────────────────────────────────────────────────────────────────────

# MPS fallback only applies on macOS; harmless but unnecessary on Linux.
# export PYTORCH_ENABLE_MPS_FALLBACK=1

# Leaderboard counts are in-memory and flushed to data/storage/db.json every
# LEADERBOARD_FLUSH_SECONDS. Shorter here so the e2e test (and the UI) see new
# evaluations quickly during local dev; prod overrides as needed.
export LEADERBOARD_FLUSH_SECONDS="${LEADERBOARD_FLUSH_SECONDS:-10}"

# Dev: serve static/index.html fresh on every request (no browser caching) so
# edits show on a plain reload — no restart, no hard-refresh. Prod leaves this
# unset and caches the HTML once at import.
export STATIC_NOCACHE="${STATIC_NOCACHE:-1}"

# Dev: a single worker. The CLI already defaults to WORKERS=1 (set here
# explicitly so it's overridable). Prod doesn't use WORKERS at all — supervisord
# manages 2 cookie-sticky uvicorns, each WORKERS=1. FastAPI runs sync endpoints
# in a threadpool, so one worker still handles concurrency (torch/numpy release
# the GIL).
export WORKERS="${WORKERS:-1}"
LOG="/tmp/ai-speech-shadowing-${PORT}.log"
if [ -t 1 ]; then
  exec uv run ai-speech-shadowing serve \
    --host 0.0.0.0 --port "${PORT}" \
    --ssl-certfile "$CERT" --ssl-keyfile "$KEY"
else
  echo "Background mode — logging to ${LOG}"
  exec uv run ai-speech-shadowing serve \
    --host 0.0.0.0 --port "${PORT}" \
    --ssl-certfile "$CERT" --ssl-keyfile "$KEY" > "$LOG" 2>&1
fi
