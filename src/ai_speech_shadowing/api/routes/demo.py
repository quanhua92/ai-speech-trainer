"""GET / — serve the single-page interactive demo HTML.

Not under ``/api/v1`` (it's a page, not an API resource) and hidden from the
OpenAPI schema to keep ``/docs`` focused on the API.
"""

from __future__ import annotations

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


@router.get("/", include_in_schema=False)
def demo() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))
