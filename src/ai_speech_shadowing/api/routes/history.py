"""GET /history, /history/{id}, /history/stats — past evaluations.

All endpoints are scoped to the requesting browser via the ``user_id`` cookie
(see :mod:`ai_speech_shadowing.api.identity`); a user only sees their own
history.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.schemas import (
    HistoryItem,
    PaginatedHistory,
    StatsResponse,
)
from ai_speech_shadowing.core.history import (
    HistoryEntry,
    compute_stats,
    delete_report,
    list_reports,
    load_report,
    report_path,
)

router = APIRouter(prefix="/history", tags=["history"])


def _uid(request: Request) -> str | None:
    """The on-disk user id from the identity middleware (or None)."""
    return getattr(request.state, "user_id", None)


@router.get("", response_model=PaginatedHistory)
def list_history(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
) -> PaginatedHistory:
    state = get_state()
    uid = _uid(request)
    entries = list_reports(state.history_dir, user_id=uid)
    # Sort chronologically by created_at (ISO strings sort lexicographically).
    # desc = newest first; asc = oldest first.
    entries.sort(key=lambda e: e.created_at or "", reverse=(sort == "desc"))
    total = len(entries)
    page = entries[offset : offset + limit]
    return PaginatedHistory(
        items=[build_history_item_from_entry(e) for e in page],
        total=total,
        limit=limit,
        offset=offset,
    )


def build_history_item_from_entry(e: HistoryEntry) -> HistoryItem:
    """Build a HistoryItem from a HistoryEntry, avoiding a file re-read."""
    return HistoryItem(
        id=e.id,
        created_at=e.created_at,
        reference_id=e.reference_id,
        composite_score=e.composite_score,
        composite_grade=e.composite_grade,
    )


@router.get("/stats", response_model=StatsResponse)
def history_stats(request: Request, period_days: int = Query(30, ge=1, le=365)) -> StatsResponse:
    state = get_state()
    return StatsResponse(**compute_stats(state.history_dir, _uid(request), period_days=period_days))


@router.get("/{report_id}")
def get_history(request: Request, report_id: str) -> dict:
    state = get_state()
    data = load_report(report_id, state.history_dir, user_id=_uid(request))
    if data is None:
        raise HTTPException(status_code=404, detail=f"report {report_id!r} not found")
    return data


@router.delete("/{report_id}", status_code=204)
def delete_history(request: Request, report_id: str) -> None:
    """Delete a saved evaluation (report JSON + any stored attempt audio)."""
    state = get_state()
    uid = _uid(request)
    removed = delete_report(report_id, state.history_dir, user_id=uid)
    wav_path = report_path(report_id, state.history_dir, user_id=uid, suffix=".wav")
    had_wav = wav_path is not None and wav_path.is_file()
    if had_wav and wav_path is not None:
        wav_path.unlink()
    if not removed and not had_wav:
        raise HTTPException(status_code=404, detail=f"report {report_id!r} not found")


@router.get("/{report_id}/audio")
def get_history_audio(request: Request, report_id: str):
    """Stream the user's recorded attempt for a past evaluation."""
    from fastapi.responses import FileResponse

    state = get_state()
    audio_path = report_path(report_id, state.history_dir, user_id=_uid(request), suffix=".wav")
    if audio_path is None or not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"no attempt audio for {report_id!r}")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        content_disposition_type="inline",  # play in <audio>, don't download
    )
