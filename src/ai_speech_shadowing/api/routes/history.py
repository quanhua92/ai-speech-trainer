"""GET /history, /history/{id}, /history/stats — past evaluations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.schemas import (
    PaginatedHistory,
    StatsResponse,
    build_history_item,
)
from ai_speech_shadowing.core.history import compute_stats, delete_report, list_reports, load_report

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=PaginatedHistory)
def list_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
) -> PaginatedHistory:
    state = get_state()
    entries = list_reports(state.history_dir)
    # Sort chronologically by created_at (ISO strings sort lexicographically).
    # desc = newest first; asc = oldest first.
    entries.sort(key=lambda e: e.created_at or "", reverse=(sort == "desc"))
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


@router.delete("/{report_id}", status_code=204)
def delete_history(report_id: str) -> None:
    """Delete a saved evaluation (report JSON + any stored attempt audio)."""
    state = get_state()
    removed = delete_report(report_id, state.history_dir)
    wav_path = state.history_dir / f"{report_id}.wav"
    had_wav = wav_path.is_file()
    if had_wav:
        wav_path.unlink()
    if not removed and not had_wav:
        raise HTTPException(status_code=404, detail=f"report {report_id!r} not found")


@router.get("/{report_id}/audio")
def get_history_audio(report_id: str):
    """Stream the user's recorded attempt for a past evaluation."""
    from fastapi.responses import FileResponse

    state = get_state()
    audio_path = state.history_dir / f"{report_id}.wav"
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"no attempt audio for {report_id!r}")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        content_disposition_type="inline",  # play in <audio>, don't download
    )
