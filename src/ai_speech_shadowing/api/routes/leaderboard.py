"""GET /leaderboard — per-user evaluation-count ranking.

Counts are accumulated per evaluation (see ``evaluate`` route) and read from the
in-memory store described in ``docs/db.md``. Reads are served from memory; a
worker's view can lag the global state by up to the flush interval.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.schemas import LeaderboardResponse

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("", response_model=LeaderboardResponse)
def get_leaderboard(
    request: Request,
    limit: int = Query(10, ge=1, le=100, description="Number of top users to return."),
) -> LeaderboardResponse:
    uid = getattr(request.state, "user_id", None)
    data = get_state().leaderboard.leaderboard(limit, me_uid=uid)
    return LeaderboardResponse(**data)
