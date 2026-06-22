"""GET /history, /history/{id}, /history/stats — past evaluations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.schemas import (
    PaginatedHistory,
    StatsResponse,
    build_history_item,
)
from ai_speech_shadowing.core.history import compute_stats, list_reports, load_report

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=PaginatedHistory)
def list_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
) -> PaginatedHistory:
    state = get_state()
    entries = list_reports(state.history_dir)
    if sort == "asc":
        entries = list(reversed(entries))
    total = len(entries)
    page = entries[offset : offset + limit]
    return PaginatedHistory(
        items=[
            build_history_item(load_report(e.id, state.history_dir) or {"id": e.id}) for e in page
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=StatsResponse)
def history_stats(period_days: int = Query(30, ge=1, le=365)) -> StatsResponse:
    state = get_state()
    return StatsResponse(**compute_stats(state.history_dir, period_days=period_days))


@router.get("/{report_id}")
def get_history(report_id: str) -> dict:
    state = get_state()
    data = load_report(report_id, state.history_dir)
    if data is None:
        raise HTTPException(status_code=404, detail=f"report {report_id!r} not found")
    return data
