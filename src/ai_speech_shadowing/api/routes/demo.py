"""GET / — serve the single-page interactive demo HTML.

Not under ``/api/v1`` (it's a page, not an API resource) and hidden from the
OpenAPI schema to keep ``/docs`` focused on the API.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["demo"])
_DEMO_HTML = Path(__file__).resolve().parent.parent / "demo.html"


@router.get("/", include_in_schema=False)
def demo() -> HTMLResponse:
    return HTMLResponse(_DEMO_HTML.read_text(encoding="utf-8"))
