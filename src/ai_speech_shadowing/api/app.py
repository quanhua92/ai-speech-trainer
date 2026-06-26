"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_speech_shadowing import __version__
from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.identity import (
    COOKIE_MAX_AGE,
    USER_ID_COOKIE,
    generate_token,
    hash_token,
)
from ai_speech_shadowing.api.routes import (
    demo,
    evaluate,
    health,
    history,
    leaderboard,
    reference,
)
from ai_speech_shadowing.core.history import cleanup_old_reports
from ai_speech_shadowing.tts.generator import PathEscapeError, cleanup_old_references

logger = logging.getLogger(__name__)

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
MAX_UPLOAD_BYTES: int = 5 * 1024 * 1024
_UPLOAD_PATHS: frozenset[str] = frozenset({"/api/v1/evaluate", "/api/v1/evaluate/quick"})

# History retention — env-configurable daily cleanup.
DEFAULT_RETENTION_DAYS: int = 7


def _retention_days() -> int:
    try:
        return int(os.environ.get("HISTORY_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS)))
    except ValueError:
        return DEFAULT_RETENTION_DAYS


def _cleanup_interval_seconds() -> float:
    try:
        hours = int(os.environ.get("HISTORY_CLEANUP_INTERVAL_HOURS", "24"))
    except ValueError:
        hours = 24
    return max(1.0, hours * 3600.0)


# Reference retention — user-generated references (source="user") are swept
# hourly by default; seed references are never touched (see cleanup_old_references).
DEFAULT_REFERENCE_RETENTION_HOURS: int = 1


def _reference_retention_hours() -> int:
    try:
        return int(
            os.environ.get("REFERENCES_RETENTION_HOURS", str(DEFAULT_REFERENCE_RETENTION_HOURS))
        )
    except ValueError:
        return DEFAULT_REFERENCE_RETENTION_HOURS


def _reference_cleanup_interval_seconds() -> float:
    try:
        hours = int(os.environ.get("REFERENCES_CLEANUP_INTERVAL_HOURS", "1"))
    except ValueError:
        hours = 1
    return max(1.0, hours * 3600.0)


# Leaderboard flush — the in-memory counters are merged to db.json periodically.
DEFAULT_LEADERBOARD_FLUSH_SECONDS: float = 60.0


def _leaderboard_flush_seconds() -> float:
    try:
        secs = float(
            os.environ.get("LEADERBOARD_FLUSH_SECONDS", str(DEFAULT_LEADERBOARD_FLUSH_SECONDS))
        )
        return max(1.0, secs)
    except ValueError:
        return DEFAULT_LEADERBOARD_FLUSH_SECONDS


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
    async def _user_id_cookie(
        request: Request, call_next: Callable[[Request], Awaitable]
    ) -> JSONResponse | object:
        """Assign each browser a stable, hashed identity cookie on first visit.

        The cookie carries a random uuid4 token; the on-disk user id is its
        SHA-256 digest (the raw token is never persisted). Populates
        ``request.state.user_id`` for every downstream handler.
        """
        raw = request.cookies.get(USER_ID_COOKIE)
        new_token: str | None = None
        if raw:
            # Treat any non-empty cookie as a raw token (UUID4 hex) and hash
            # it for storage.  The 64-hex digest passthrough was removed in
            # pre-1.0 because it allowed user-id hash enumeration to produce
            # a valid cookie (impersonation).  Old-format cookies are treated
            # as new visitors — their orphaned history is cleaned up by the
            # retention policy.
            request.state.user_id = hash_token(raw)
        else:
            # first visit: mint a token, hash for storage, set cookie with raw
            new_token = generate_token()
            request.state.user_id = hash_token(new_token)
        response = await call_next(request)
        if new_token is not None:
            secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
            response.set_cookie(
                USER_ID_COOKIE,
                new_token,
                httponly=True,
                samesite="lax",
                secure=secure,
                max_age=COOKIE_MAX_AGE,
                path="/",
            )
        return response

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
            # Reject chunked transfer encoding — no Content-Length to check.
            if request.headers.get("transfer-encoding", "").lower() not in ("", "identity"):
                return JSONResponse(
                    status_code=413, content={"detail": "chunked transfer encoding not allowed"}
                )
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
    app.include_router(leaderboard.router, prefix="/api/v1")
    app.include_router(demo.router)  # "/" — the demo page, not an API resource
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown hook. Models stay lazy; cleanup tasks run periodically."""
    history_task = asyncio.create_task(_periodic_cleanup())
    reference_task = asyncio.create_task(_periodic_reference_cleanup())
    leaderboard_task = asyncio.create_task(_periodic_leaderboard_flush())
    try:
        yield
    finally:
        history_task.cancel()
        reference_task.cancel()
        leaderboard_task.cancel()
        with suppress(asyncio.CancelledError):
            await history_task
        with suppress(asyncio.CancelledError):
            await reference_task
        with suppress(asyncio.CancelledError):
            await leaderboard_task
        # commit pending counts so they survive a normal restart
        try:
            get_state().leaderboard.flush()
        except Exception:
            logger.exception("leaderboard shutdown flush failed")


async def _periodic_cleanup() -> None:
    """Delete aged-out history reports at startup, then every interval."""
    interval = _cleanup_interval_seconds()
    # stagger the first run slightly off startup to avoid a cold-start spike
    await asyncio.sleep(5)
    while True:
        try:
            deleted = cleanup_old_reports(get_state().history_dir, _retention_days())
            if deleted:
                logger.info("periodic cleanup removed %d report(s)", deleted)
        except Exception:
            logger.exception("history cleanup task failed")
        await asyncio.sleep(interval)


async def _periodic_reference_cleanup() -> None:
    """Delete aged-out user references at startup, then every interval.

    Only ``source="user"`` references are eligible — seed references are
    preserved by cleanup_old_references itself.
    """
    interval = _reference_cleanup_interval_seconds()
    # stagger off both startup and the history sweep
    await asyncio.sleep(15)
    while True:
        try:
            deleted = cleanup_old_references(
                get_state().reference_manager.config.base_dir,
                _reference_retention_hours(),
            )
            if deleted:
                logger.info("periodic cleanup removed %d user reference(s)", deleted)
        except Exception:
            logger.exception("reference cleanup task failed")
        await asyncio.sleep(interval)


async def _periodic_leaderboard_flush() -> None:
    """Merge in-memory evaluation counts to db.json every ~minute if changed.

    Per-worker phase offset (PID-based) + interval jitter spread the two
    workers' flushes apart so the flock is rarely contended. The flock itself
    makes simultaneous flushes correct regardless — jitter just smooths it.
    """
    import random

    store = get_state().leaderboard
    interval = _leaderboard_flush_seconds()
    # per-worker phase offset: separate processes have separate pids
    await asyncio.sleep(os.getpid() % 30)
    while True:
        try:
            flushed = store.flush_if_dirty()
            if flushed:
                logger.info("leaderboard flushed %d increment(s)", flushed)
        except Exception:
            logger.exception("leaderboard flush task failed")
        await asyncio.sleep(max(1.0, interval + random.uniform(-5.0, 5.0)))


app = create_app()
