# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# uv: copy (not symlink) the venv, compile bytecode, pin the env path.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/models \
    PATH=/app/.venv/bin:$PATH

# Runtime system libraries:
#   espeak-ng   — kokoro's misaki English OOD fallback
#   libsndfile1 — soundfile WAV decode (the wheel bundles it too, belt-and-braces)
#   libgomp1    — torch's OpenMP runtime on linux
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies first (cached layer, rebuilds only when the lock changes).
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Bake the models. If data/models/hub/ was pre-warmed locally (copied from
#    ~/.cache/huggingface) the COPY already populated /models and we skip the
#    download. Otherwise (e.g. a clean CI/server build) prewarm downloads them
#    now. Placed before source so code edits don't re-trigger the model layer.
COPY data/models/ /models/
COPY scripts/prewarm_models.py ./scripts/
RUN ls -A /models/hub >/dev/null 2>&1 \
    && echo ">>> /models populated from build context; skipping download." \
    || ( echo ">>> /models empty — downloading models (~1.5 GB)…" \
         && /app/.venv/bin/python scripts/prewarm_models.py ) \
    && rm -rf /root/.cache

# 3) Application code + bundled default references (shipped in git).
COPY README.md ./
COPY src/ ./src/
COPY data/references/ ./data/references/
RUN uv sync --frozen --no-dev

EXPOSE 8000
ENTRYPOINT ["/app/.venv/bin/ai-speech-shadowing"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
