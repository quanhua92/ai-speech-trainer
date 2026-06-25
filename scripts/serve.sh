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

# MPS fallback only applies on macOS; harmless but unnecessary on Linux.
# export PYTORCH_ENABLE_MPS_FALLBACK=1
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
