"""GET / — serve the single-page interactive demo HTML.

Not under ``/api/v1`` (it's a page, not an API resource) and hidden from the
OpenAPI schema to keep ``/docs`` focused on the API.

In dev (``STATIC_NOCACHE=1``, set by ``scripts/serve.sh``) the HTML is re-read
on every request and served with ``Cache-Control: no-store`` so edits to
``static/index.html`` show on a plain reload — no server restart, no
hard-refresh. In prod the content is read once at import (cached) for speed.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["demo"])


def _find_index() -> Path:
    """Walk upward from this file to find static/index.html.

    Works regardless of CWD — in dev (src/ai_speech_shadowing/api/routes/)
    and in Docker (editable install at /app/src/ai_speech_shadowing/...).
    """
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        candidate = parent / "static" / "index.html"
        if candidate.is_file():
            return candidate
    return Path("static/index.html")  # fallback to CWD


_INDEX_HTML = _find_index()
# Dev mode (STATIC_NOCACHE) reads the file fresh per request so HTML edits are
# visible without a restart; prod caches the content once at import.
_NOCACHE = bool(os.environ.get("STATIC_NOCACHE"))
_INDEX_HTML_CONTENT = None if _NOCACHE else _INDEX_HTML.read_text(encoding="utf-8")


@router.get("/", include_in_schema=False)
def demo() -> HTMLResponse:
    if _NOCACHE:
        return HTMLResponse(
            _INDEX_HTML.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    return HTMLResponse(_INDEX_HTML_CONTENT)
