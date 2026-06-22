# ai-speech-shadowing — common dev tasks. Run `just` with no args to list recipes.

# Run `uv sync` to install/sync runtime + dev deps into .venv.
install:
    uv sync

# Alias for `install`.
sync:
    uv sync

# Lint with ruff.
lint:
    uv run ruff check .

# Auto-format with ruff.
format:
    uv run ruff format .

# Check formatting without writing (used by pre-commit hook).
format-check:
    uv run ruff format --check .

# Run the test suite.
test:
    uv run pytest

# Run slow tests that download/load ML models (wav2vec2, kokoro). ~15s after cache.
test-slow:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run pytest --runslow

# Serve the REST API (FastAPI + uvicorn) at /api/v1.
serve:
    uv run ai-speech-shadowing serve

# End-to-end REST API test (starts its own server, real Kokoro A-vs-B speech).
e2e:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/test_e2e.py

# Run tests with coverage.
test-cov:
    uv run pytest --cov=ai_speech_shadowing --cov-report=term-missing

# Quick type check.
typecheck:
    uv run mypy src

# Explore Kokoro TTS: text -> 24kHz WAV in tmp/audio/.
# Usage: just explore "Hello world"
explore text:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/explore_kokoro.py --text "{{text}}"

# Point git at the project-local hooks (run once after cloning).
hooks:
    git config core.hooksPath githooks

# Verify: lint + format-check + tests (what the CI gate should mirror).
verify: lint format-check test
