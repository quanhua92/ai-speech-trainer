# syntax=docker/dockerfile:1
# Multi-stage: the builder carries compilers (some deps, notably
# praat-parselmouth, ship no linux/aarch64 wheel and must build from sdist on
# Apple Silicon). The runtime stage copies only the finished venv + models.

# ── builder ──────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/models

# Build tools for any sdist (praat-parselmouth uses scikit-build + CMake/ninja).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies first (cached on the lock). praat-parselmouth compiles here.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Bake models. Copy pre-warmed cache from data/models/ if present (fast,
#    offline); otherwise download. Placed before source so code edits don't
#    re-trigger this layer.
COPY data/models/ /models/
COPY scripts/prewarm_models.py ./scripts/
RUN ls -A /models/hub >/dev/null 2>&1 \
    && echo ">>> /models populated from build context; skipping download." \
    || ( echo ">>> /models empty — downloading models (~1.5 GB)…" \
         && /app/.venv/bin/python scripts/prewarm_models.py ) \
    && rm -rf /root/.cache

# 3) Application code + bundled default references; build & install the project.
COPY README.md ./
COPY src/ ./src/
COPY data/references/ ./data/references/
COPY data/default.txt ./data/default.txt
RUN uv sync --frozen --no-dev

# ── runtime ──────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV HF_HOME=/models \
    PATH=/app/.venv/bin:$PATH

# Runtime system libraries only (no compilers → smaller image):
#   libsndfile1 — soundfile WAV decode (wheel bundles it too)
#   libgomp1    — torch's OpenMP runtime on linux
#   nginx       — in-container reverse proxy (cookie-sticky across uvicorns)
#   supervisor  — PID 1 that manages nginx + the uvicorn workers
# Note: espeak-ng is NOT needed — the espeakng-loader wheel (a misaki[en] dep)
#   vendors libespeak-ng + espeak-ng-data for linux x86-64/arm64.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 libgomp1 ca-certificates nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv        /app/.venv
COPY --from=builder /app/src          /app/src
COPY --from=builder /models           /models
COPY --from=builder /app/data/references /app/data/references
COPY static/                          /app/static/
COPY docker/                          /app/docker/

# appuser needs a real, owned HOME so libraries that cache to $HOME/.cache
# (numba/librosa's @jit cache, etc.) have a writable location. Without it they
# try to write next to root-owned site-packages and fail ("no locator
# available"). --home both sets the passwd field AND creates the dir; note
# --create-home is rejected by adduser under --system, but --home works.
# We do NOT chmod the venv: this app ingests untrusted audio, and a writable
# package tree would let an audio-parsing RCE trojanize installed libraries,
# undoing the non-root hardening.
RUN adduser --system --group --home /home/appuser appuser \
    && mkdir -p /app/data/history /app/data/recordings /app/data/storage \
    && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000
# supervisord (PID 1) manages nginx + 2 single-worker uvicorns (see
# docker/supervisord.conf). nginx does cookie-sticky routing across the uvicorns
# (docker/nginx.conf). Replaces uvicorn's built-in multi-worker spawner.
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/app/docker/supervisord.conf"]
CMD []
