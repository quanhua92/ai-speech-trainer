"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_speech_shadowing import __version__
from ai_speech_shadowing.api.routes import demo, evaluate, health, history, reference


def create_app() -> FastAPI:
    """Build the configured FastAPI app with all routers under /api/v1."""
    app = FastAPI(
        title="ai-speech-shadowing",
        version=__version__,
        description="Local-first speech shadowing evaluation engine — REST API.",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(evaluate.router, prefix="/api/v1")
    app.include_router(reference.router, prefix="/api/v1")
    app.include_router(history.router, prefix="/api/v1")
    app.include_router(demo.router)  # "/" — the demo page, not an API resource
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown hook (models stay lazy — loaded on first request)."""
    yield


app = create_app()
