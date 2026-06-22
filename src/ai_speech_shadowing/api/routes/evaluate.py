"""POST /evaluate and /evaluate/quick — the core evaluation endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.routes.reference import _iso_lang_to_kokoro
from ai_speech_shadowing.api.schemas import EvaluationResponse, build_evaluation_response
from ai_speech_shadowing.core.audio import AudioLoadError, AudioSample
from ai_speech_shadowing.core.feedback import evaluate as evaluate_pipeline
from ai_speech_shadowing.core.history import save_report
from ai_speech_shadowing.core.preprocess import preprocess

router = APIRouter(tags=["evaluate"])
logger = logging.getLogger(__name__)


def _decode_upload(upload: UploadFile) -> AudioSample:
    """Read an UploadFile and decode it into an AudioSample (400 on failure)."""
    data = upload.file.read()
    try:
        return AudioSample.from_bytes(data)
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=f"unreadable audio: {e}") from e


def _run_evaluation(
    *,
    reference_audio: AudioSample,
    user_audio: AudioSample,
    reference_id: str | None,
    reference_text: str | None = None,
) -> EvaluationResponse:
    """Preprocess both signals, evaluate, persist, and build the API response."""
    state = get_state()
    extractor = state.phoneme_extractor()
    ref = preprocess(reference_audio)
    hyp = preprocess(user_audio)
    report = evaluate_pipeline(ref, hyp, phoneme_extractor=extractor, reference_text=reference_text)

    path = save_report(report, history_dir=state.history_dir)
    # stamp reference_id onto the saved report for history/stats
    _stamp_reference_id(path, reference_id)

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return build_evaluation_response(
        report,
        reference_id=reference_id,
        eval_id=str(data.get("id", path.stem)),
        created_at=str(data.get("created_at", "")),
    )


def _stamp_reference_id(path, reference_id: str | None) -> None:
    import json

    if reference_id is None:
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["reference_id"] = reference_id
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    audio: Annotated[UploadFile, File(description="User audio (WAV/WebM/MP3).")],
    reference_id: Annotated[str, Form(description="Reference slug to compare against.")],
) -> EvaluationResponse:
    """Evaluate user audio against a pre-generated reference."""
    state = get_state()
    profile = state.reference_manager.voice_profile()
    ref_file = state.reference_manager.audio_file(reference_id, profile)
    if not ref_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"reference {reference_id!r} not found; POST /references first",
        )
    reference_audio = AudioSample.from_wav(ref_file)
    user_audio = _decode_upload(audio)
    # pull the reference's source text for word-level highlighting
    reference_text = str(state.reference_manager.read_metadata(reference_id).get("text", ""))
    return _run_evaluation(
        reference_audio=reference_audio,
        user_audio=user_audio,
        reference_id=reference_id,
        reference_text=reference_text or None,
    )


@router.post("/evaluate/quick", response_model=EvaluationResponse)
def evaluate_quick(
    audio: Annotated[UploadFile, File(description="User audio (WAV/WebM/MP3).")],
    text: Annotated[str, Form(description="Target sentence in the target language.")],
    language: Annotated[str, Form(description="Language code, e.g. 'en'.")] = "en",
) -> EvaluationResponse:
    """Evaluate user audio against a TTS reference generated on-the-fly."""
    import time

    state = get_state()
    lang_code = _iso_lang_to_kokoro(language)
    t0 = time.perf_counter()
    ref_path = state.reference_manager.generate(text, lang=lang_code)
    state.mark_tts_loaded(load_time_ms=int((time.perf_counter() - t0) * 1000))
    reference_audio = AudioSample.from_wav(ref_path)
    user_audio = _decode_upload(audio)
    from ai_speech_shadowing.tts.generator import slugify

    return _run_evaluation(
        reference_audio=reference_audio,
        user_audio=user_audio,
        reference_id=slugify(text),
        reference_text=text,
    )
