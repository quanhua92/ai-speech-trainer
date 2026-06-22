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


def _decode_upload(upload: UploadFile) -> tuple[AudioSample, bytes]:
    """Read an UploadFile → (AudioSample, raw bytes). 400 on failure."""
    data = upload.file.read()
    try:
        return AudioSample.from_bytes(data), data
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=f"unreadable audio: {e}") from e


def _run_evaluation(
    *,
    reference_audio: AudioSample,
    user_audio: AudioSample,
    reference_id: str | None,
    reference_text: str | None = None,
    attempt_bytes: bytes | None = None,
    weights: tuple[float, float, float] | None = None,
    dtw_score_scale: float | None = None,
    feedback_language: str = "en",
) -> EvaluationResponse:
    """Preprocess both signals, evaluate, persist (report + attempt audio), respond."""
    state = get_state()
    extractor = state.phoneme_extractor()
    ref = preprocess(reference_audio)
    hyp = preprocess(user_audio)
    eval_kwargs: dict[str, object] = {}
    if weights:
        eval_kwargs["weights"] = weights
    if dtw_score_scale is not None:
        eval_kwargs["dtw_score_scale"] = dtw_score_scale
    if feedback_language and feedback_language != "en":
        eval_kwargs["feedback_language"] = feedback_language
    report = evaluate_pipeline(
        ref,
        hyp,
        phoneme_extractor=extractor,
        reference_text=reference_text,
        **eval_kwargs,
    )

    path = save_report(report, history_dir=state.history_dir)
    eval_id = path.stem
    # stamp reference_id onto the saved report for history/stats
    _stamp_reference_id(path, reference_id)

    # persist the user's attempt audio so history rows can replay it
    audio_url: str | None = None
    if attempt_bytes:
        audio_path = state.history_dir / f"{eval_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(attempt_bytes)
        audio_url = f"/api/v1/history/{eval_id}/audio"
        _stamp_field(path, "audio_url", audio_url)

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return build_evaluation_response(
        report,
        reference_id=reference_id,
        eval_id=str(data.get("id", eval_id)),
        created_at=str(data.get("created_at", "")),
        audio_url=audio_url,
    )


def _stamp_field(path, field: str, value: object) -> None:
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data[field] = value
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        pass


def _stamp_reference_id(path, reference_id: str | None) -> None:
    if reference_id is not None:
        _stamp_field(path, "reference_id", reference_id)


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    audio: Annotated[UploadFile, File(description="User audio (WAV/WebM/MP3).")],
    reference_id: Annotated[str, Form(description="Reference slug to compare against.")],
    weight_pronunciation: Annotated[
        float, Form(ge=0, le=1, description="Pronunciation weight (0-1).")
    ] = 0.4,
    weight_intonation: Annotated[
        float, Form(ge=0, le=1, description="Intonation weight (0-1).")
    ] = 0.3,
    weight_fluency: Annotated[float, Form(ge=0, le=1, description="Fluency weight (0-1).")] = 0.3,
    dtw_scale: Annotated[
        float, Form(ge=0.1, le=10, description="DTW score scale - higher = more lenient fluency.")
    ] = 1.0,
    feedback_language: Annotated[str, Form(description="Feedback language: 'en' or 'vi'.")] = "en",
) -> EvaluationResponse:
    """Evaluate user audio against a pre-generated reference."""
    state = get_state()
    profile = state.reference_manager._profile_for_slug(reference_id)
    ref_file = state.reference_manager.audio_file(reference_id, profile)
    if not ref_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"reference {reference_id!r} not found; POST /references first",
        )
    reference_audio = AudioSample.from_wav(ref_file)
    user_audio, attempt_bytes = _decode_upload(audio)
    reference_text = str(state.reference_manager.read_metadata(reference_id).get("text", ""))

    # normalise weights to sum 1
    raw_w = (weight_pronunciation, weight_intonation, weight_fluency)
    wsum = sum(raw_w) or 1.0
    weights = tuple(w / wsum for w in raw_w)

    return _run_evaluation(
        reference_audio=reference_audio,
        user_audio=user_audio,
        reference_id=reference_id,
        reference_text=reference_text or None,
        attempt_bytes=attempt_bytes,
        weights=weights,
        dtw_score_scale=dtw_scale,
        feedback_language=feedback_language,
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
    user_audio, attempt_bytes = _decode_upload(audio)
    from ai_speech_shadowing.tts.generator import slugify

    return _run_evaluation(
        reference_audio=reference_audio,
        user_audio=user_audio,
        reference_id=slugify(text),
        reference_text=text,
        attempt_bytes=attempt_bytes,
    )
