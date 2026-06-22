"""Reference CRUD + audio download.

References are stored by the ``ReferenceManager`` under ``data/references``.
The reference id is the slug. ``GET /references/{id}/audio`` streams the WAV.
"""

from __future__ import annotations

import shutil
import time

import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.schemas import ReferenceCreateRequest, ReferenceResponse
from ai_speech_shadowing.tts.generator import KOKORO_LANGUAGES, slugify

router = APIRouter(prefix="/references", tags=["references"])


def _iso_lang_to_kokoro(language: str) -> str:
    """Map an ISO-ish language code to Kokoro's single-letter code (default 'a')."""
    reverse = {iso: code for code, iso in KOKORO_LANGUAGES.items()}
    reverse.setdefault("en", "a")
    return reverse.get(language, "a")


def _build_response(slug: str) -> ReferenceResponse:
    state = get_state()
    meta = state.reference_manager.read_metadata(slug)
    if not meta:
        raise HTTPException(status_code=404, detail=f"reference {slug!r} not found")
    profile = state.reference_manager._profile_for_slug(slug)
    audio_file = state.reference_manager.audio_file(slug, profile)
    duration_seconds = 0.0
    if audio_file.is_file():
        try:
            info = sf.info(str(audio_file))
            duration_seconds = info.frames / info.samplerate
        except RuntimeError:
            pass
    return ReferenceResponse(
        id=slug,
        text=str(meta.get("text", "")),
        language=str(meta.get("language", "")),
        speaker=str(meta.get("default_speaker", "default")),
        duration_seconds=round(duration_seconds, 3),
        audio_url=f"/api/v1/references/{slug}/audio",
        created_at=str(meta.get("updated_at", "")),
    )


@router.post("", response_model=ReferenceResponse, status_code=201)
def create_reference(req: ReferenceCreateRequest) -> ReferenceResponse:
    """Generate a Kokoro TTS reference from text."""
    state = get_state()
    lang_code = _iso_lang_to_kokoro(req.language)
    # "default" is the API sentinel; resolve it to the configured Kokoro voice.
    voice = (
        state.reference_manager.config.default_voice if req.speaker == "default" else req.speaker
    )
    t0 = time.perf_counter()
    state.reference_manager.generate(req.text, voice=voice, lang=lang_code)
    state.mark_tts_loaded(load_time_ms=int((time.perf_counter() - t0) * 1000))
    return _build_response(slugify(req.text))


@router.get("", response_model=list[ReferenceResponse])
def list_references() -> list[ReferenceResponse]:
    return [
        _build_response(str(entry["slug"]))
        for entry in get_state().reference_manager.list_references()
    ]


@router.get("/{slug}", response_model=ReferenceResponse)
def get_reference(slug: str) -> ReferenceResponse:
    return _build_response(slug)


@router.delete("/{slug}", status_code=204)
def delete_reference(slug: str) -> None:
    state = get_state()
    slug_dir = state.reference_manager.slug_path(slug)
    if not slug_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"reference {slug!r} not found")
    shutil.rmtree(slug_dir)


@router.get("/{slug}/audio")
def get_reference_audio(slug: str) -> FileResponse:
    state = get_state()
    profile = state.reference_manager._profile_for_slug(slug)
    audio_file = state.reference_manager.audio_file(slug, profile)
    if not audio_file.is_file():
        raise HTTPException(status_code=404, detail=f"no audio for reference {slug!r}")
    return FileResponse(
        path=str(audio_file),
        media_type="audio/wav",
        content_disposition_type="inline",  # play in <audio>, don't download
    )
