"""GET /health — service & model status."""

from __future__ import annotations

from fastapi import APIRouter

from ai_speech_shadowing import __version__
from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.schemas import HealthResponse, ModelStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = get_state()
    return HealthResponse(
        status="healthy",
        version=__version__,
        models={
            "wav2vec2": ModelStatus(
                loaded=state._extractor is not None,
                load_time_ms=state.extractor_load_time_ms,
            ),
            "tts": ModelStatus(loaded=state.tts_available, load_time_ms=state.tts_load_time_ms),
        },
    )
