"""POST /evaluate and /evaluate/quick — the core evaluation endpoints."""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ai_speech_shadowing.api.deps import get_state
from ai_speech_shadowing.api.routes.reference import _iso_lang_to_kokoro
from ai_speech_shadowing.api.schemas import EvaluationResponse, build_evaluation_response
from ai_speech_shadowing.core.audio import AudioLoadError, AudioSample
from ai_speech_shadowing.core.feedback import evaluate as evaluate_pipeline
from ai_speech_shadowing.core.history import save_report
from ai_speech_shadowing.core.leaderboard import audio_hash
from ai_speech_shadowing.core.preprocess import preprocess

router = APIRouter(tags=["evaluate"])
logger = logging.getLogger(__name__)


def _leaderboard_min_score() -> int:
    """Composite score (0-100) an evaluation must reach to count on the leaderboard."""
    try:
        return max(0, min(100, int(os.environ.get("LEADERBOARD_MIN_SCORE", "30"))))
    except ValueError:
        return 30


def _decode_upload(upload: UploadFile) -> tuple[AudioSample, bytes]:
    """Read an UploadFile → (AudioSample, raw bytes). 400 on failure."""
    data = upload.file.read()
    try:
        return AudioSample.from_bytes(data), data
    except AudioLoadError as e:
        raise HTTPException(status_code=400, detail=f"unreadable audio: {e}") from e


def _run_evaluation(
    *,
    request: Request,
    reference_audio: AudioSample,
    user_audio: AudioSample,
    reference_id: str | None,
    reference_text: str | None = None,
    reference_phonemes: list[str] | None = None,
    attempt_bytes: bytes | None = None,
    weights: tuple[float, float, float] | None = None,
    dtw_score_scale: float | None = None,
    feedback_language: str = "en",
    reference_language: str | None = None,
) -> EvaluationResponse:
    """Preprocess both signals, evaluate, persist (report + attempt audio), respond."""
    state = get_state()
    extractor = state.phoneme_extractor()
    user_id = getattr(request.state, "user_id", None)
    ref = preprocess(reference_audio)
    hyp = preprocess(user_audio)
    eval_kwargs: dict[str, object] = {}
    if weights:
        eval_kwargs["weights"] = weights
    if dtw_score_scale is not None:
        eval_kwargs["dtw_score_scale"] = dtw_score_scale
    if feedback_language and feedback_language != "en":
        eval_kwargs["feedback_language"] = feedback_language
    if reference_phonemes is not None:
        eval_kwargs["reference_phonemes"] = reference_phonemes
    if reference_language:
        eval_kwargs["reference_language"] = reference_language
    report = evaluate_pipeline(
        ref,
        hyp,
        phoneme_extractor=extractor,
        reference_text=reference_text,
        **eval_kwargs,
    )

    path = save_report(report, history_dir=state.history_dir, user_id=user_id)
    eval_id = path.stem
    # stamp reference_id onto the saved report for history/stats
    _stamp_reference_id(path, reference_id)

    # bump the per-user evaluation count (in-memory; flushed to db.json later).
    # Only real attempts count: a minimum composite score gates out noise/silence,
    # and the leaderboard dedupes replays of the same audio. Best-effort: a
    # leaderboard failure must never break an evaluation.
    if user_id and report.composite_score >= _leaderboard_min_score():
        try:
            ahash = audio_hash(attempt_bytes) if attempt_bytes else None
            state.leaderboard.increment(user_id, ahash)
        except Exception:
            logger.warning("leaderboard increment failed", exc_info=True)

    # persist the user's attempt audio so history rows can replay it
    audio_url: str | None = None
    if attempt_bytes:
        audio_path = state.history_dir / (user_id or "_cli") / f"{eval_id}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(attempt_bytes)
        audio_url = f"/api/v1/history/{eval_id}/audio"
        _stamp_field(path, "audio_url", audio_url)

    data = json.loads(path.read_text(encoding="utf-8"))
    return build_evaluation_response(
        report,
        reference_id=reference_id,
        eval_id=str(data.get("id", eval_id)),
        created_at=str(data.get("created_at", "")),
        audio_url=audio_url,
    )


def _stamp_field(path, field: str, value: object) -> None:
    import os

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data[field] = value
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, json.JSONDecodeError):
        logger.warning("failed to stamp %s on %s", field, path, exc_info=True)


def _stamp_reference_id(path, reference_id: str | None) -> None:
    if reference_id is not None:
        _stamp_field(path, "reference_id", reference_id)


def _read_reference_phonemes(manager, reference_id: str) -> list[str] | None:
    """Read cached G2P phonemes for a reference slug, if present.

    Returns ``None`` when the metadata is absent, the ``phonemes`` block was
    never written (e.g. an uploaded clip without transcript), or the cached
    tokens are empty. A ``None`` return triggers the acoustic fallback in
    :func:`evaluate`.
    """
    meta = manager.read_metadata(reference_id)
    if not meta:
        return None
    block = meta.get("phonemes")
    if not isinstance(block, dict):
        return None
    tokens = block.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        return None
    return [str(t) for t in tokens]


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    request: Request,
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
    meta = state.reference_manager.read_metadata(reference_id)
    reference_text = str(meta.get("text", ""))
    reference_language = str(meta.get("language", "")) or None
    reference_phonemes = _read_reference_phonemes(state.reference_manager, reference_id)

    # normalise weights to sum 1
    raw_w = (weight_pronunciation, weight_intonation, weight_fluency)
    if sum(raw_w) == 0:
        raise HTTPException(
            status_code=400,
            detail="at least one scoring weight must be non-zero",
        )
    wsum = sum(raw_w)
    weights = tuple(w / wsum for w in raw_w)

    return _run_evaluation(
        request=request,
        reference_audio=reference_audio,
        user_audio=user_audio,
        reference_id=reference_id,
        reference_text=reference_text or None,
        reference_phonemes=reference_phonemes,
        reference_language=reference_language,
        attempt_bytes=attempt_bytes,
        weights=weights,
        dtw_score_scale=dtw_scale,
        feedback_language=feedback_language,
    )


@router.post("/evaluate/quick", response_model=EvaluationResponse)
def evaluate_quick(
    request: Request,
    audio: Annotated[UploadFile, File(description="User audio (WAV/WebM/MP3).")],
    text: Annotated[str, Form(min_length=1, max_length=500, description="Target sentence.")],
    language: Annotated[str, Form(description="Language code, e.g. 'en'.")] = "en",
) -> EvaluationResponse:
    """Evaluate user audio against a TTS reference generated on-the-fly."""
    import time

    state = get_state()
    lang_code = _iso_lang_to_kokoro(language)
    t0 = time.perf_counter()
    ref_path = state.reference_manager.generate(text, lang=lang_code, source="user")
    state.mark_tts_loaded(load_time_ms=int((time.perf_counter() - t0) * 1000))
    reference_audio = AudioSample.from_wav(ref_path)
    user_audio, attempt_bytes = _decode_upload(audio)
    from ai_speech_shadowing.tts.generator import slugify

    slug = slugify(text)
    # generate() captured Kokoro's G2P at synthesis time; reuse it as the
    # reference phoneme source so we don't re-run Wav2Vec2 on the synthesized audio.
    reference_phonemes = _read_reference_phonemes(state.reference_manager, slug)

    return _run_evaluation(
        request=request,
        reference_audio=reference_audio,
        user_audio=user_audio,
        reference_id=slug,
        reference_text=text,
        reference_phonemes=reference_phonemes,
        reference_language=language,
        attempt_bytes=attempt_bytes,
    )
