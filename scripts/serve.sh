#!/usr/bin/env bash
# Serve the REST API + demo over HTTPS on 0.0.0.0:8000.
# Generates a self-signed cert on first run (needed for the mic on non-localhost).
#
# Usage:
#   bash scripts/serve.sh                        # foreground
#   nohup bash scripts/serve.sh > /tmp/serve.log 2>&1 &   # background
#
# Then open https://localhost:8000/  (accept the self-signed cert warning)
set -euo pipefail
cd "$(dirname "$0")/.."

CERT=/tmp/ai-speech-shadowing-cert.pem
KEY=/tmp/ai-speech-shadowing-key.pem

if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
  echo "Generating self-signed TLS cert..."
  mkdir -p tmp
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY" -out "$CERT" -days 365 -subj "/CN=localhost" 2>/dev/null
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
exec uv run ai-speech-shadowing serve \
  --host 0.0.0.0 --port 8000 \
  --ssl-certfile "$CERT" --ssl-keyfile "$KEY"
