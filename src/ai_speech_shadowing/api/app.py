"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_speech_shadowing import __version__
from ai_speech_shadowing.api.routes import demo, evaluate, health, history, reference
from ai_speech_shadowing.tts.generator import PathEscapeError

# Origins the demo/dev frontends may use. Used both for the CORS allowlist and
# for the CSRF Origin check below.
_CORS_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
]
_CORS_ORIGIN_HOSTS: set[str] = {urlparse(o).netloc for o in _CORS_ORIGINS}

# CSRF: state-changing methods a browser will send with an Origin header.
_STATE_CHANGING: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Hard cap on evaluation upload bodies (bounded before read into memory / disk).
MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
_UPLOAD_PATHS: frozenset[str] = frozenset({"/api/v1/evaluate", "/api/v1/evaluate/quick"})


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
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _csrf_and_upload_guard(
        request: Request, call_next: Callable[[Request], Awaitable]
    ) -> JSONResponse | object:
        # CSRF defense: a browser sends `Origin` on state-changing requests; a
        # non-browser client usually omits it (and is allowed). When present,
        # the origin must be same-origin (== Host) or one of the configured
        # origins — this stops cross-site multipart POSTs / drive-by abuse.
        if request.method in _STATE_CHANGING:
            origin = request.headers.get("origin")
            if origin:
                origin_host = urlparse(origin).netloc
                host = request.headers.get("host", "")
                if origin_host != host and origin_host not in _CORS_ORIGIN_HOSTS:
                    return JSONResponse(
                        status_code=403, content={"detail": "cross-origin request not allowed"}
                    )
        # Upload cap: reject oversized evaluation uploads up front so they are
        # never read into memory or written verbatim to disk.
        if request.url.path in _UPLOAD_PATHS:
            cl = request.headers.get("content-length")
            if cl:
                try:
                    if int(cl) > MAX_UPLOAD_BYTES:
                        return JSONResponse(status_code=413, content={"detail": "upload too large"})
                except ValueError:
                    pass
        return await call_next(request)

    # Path-traversal attempts (from slug / voice / speaker / report_id) surface
    # as PathEscapeError deep in the manager; map them to a generic 400 so the
    # response neither confirms nor leaks the target path.
    @app.exception_handler(PathEscapeError)
    async def _path_escape_handler(_request: Request, _exc: PathEscapeError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "invalid path segment"})

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
