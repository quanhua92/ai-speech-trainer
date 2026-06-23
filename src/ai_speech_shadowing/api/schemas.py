"""Pydantic request/response models matching the REST API spec.

Every endpoint's payload is one of these models, so the auto-generated OpenAPI
document (``/docs``) is the source of truth for the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ai_speech_shadowing.core.feedback import FeedbackReport, _op_to_dict, grade_for

if TYPE_CHECKING:
    from collections.abc import Iterable


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class ModelStatus(BaseModel):
    loaded: bool
    load_time_ms: int | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict[str, ModelStatus]


# --------------------------------------------------------------------------- #
# Scores / evaluation
# --------------------------------------------------------------------------- #
class PronunciationScore(BaseModel):
    phoneme_error_rate: float
    score: int
    grade: str


class IntonationScore(BaseModel):
    pitch_range_ratio: float
    monotone: bool
    score: int
    grade: str


class FluencyScore(BaseModel):
    dtw_normalized_distance: float
    syllable_rate: float
    pause_count: int
    score: int
    grade: str


class CompositeScore(BaseModel):
    score: int
    grade: str


class Scores(BaseModel):
    pronunciation: PronunciationScore
    intonation: IntonationScore
    fluency: FluencyScore
    composite: CompositeScore


class PhonemeDiffItem(BaseModel):
    type: str
    phoneme: str | None = None
    expected: str | None = None
    actual: str | None = None


class WordDiffItem(BaseModel):
    word: str
    status: str  # match | sub | del | ins
    errors: list[PhonemeDiffItem] = []


class EvaluationResponse(BaseModel):
    id: str
    created_at: str
    reference_id: str | None
    scores: Scores
    phoneme_diff: list[PhonemeDiffItem]
    words: list[WordDiffItem] = []
    feedback: list[str]
    audio_url: str | None = None


def build_evaluation_response(
    report: FeedbackReport,
    *,
    reference_id: str | None,
    eval_id: str,
    created_at: str,
    audio_url: str | None = None,
) -> EvaluationResponse:
    """Adapt a FeedbackReport + envelope into the API EvaluationResponse."""
    return EvaluationResponse(
        id=eval_id,
        created_at=created_at,
        reference_id=reference_id,
        scores=Scores(
            pronunciation=PronunciationScore(
                phoneme_error_rate=round(report.phoneme_error_rate, 4),
                score=report.pronunciation_score,
                grade=grade_for(report.pronunciation_score),
            ),
            intonation=IntonationScore(
                pitch_range_ratio=round(report.pitch_range_ratio, 3),
                monotone=report.monotone,
                score=report.intonation_score,
                grade=grade_for(report.intonation_score),
            ),
            fluency=FluencyScore(
                dtw_normalized_distance=round(report.dtw_normalized_distance, 4),
                syllable_rate=round(report.syllable_rate_hypothesis, 3),
                pause_count=report.pause_count_hypothesis,
                score=report.fluency_score,
                grade=grade_for(report.fluency_score),
            ),
            composite=CompositeScore(score=report.composite_score, grade=report.composite_grade),
        ),
        phoneme_diff=[PhonemeDiffItem(**_op_to_dict(op)) for op in report.phoneme_diff.operations],
        words=[
            WordDiffItem(
                word=w.word,
                status=w.status,
                errors=[PhonemeDiffItem(**e) for e in w.errors],
            )
            for w in report.words
        ],
        feedback=list(report.feedback),
        audio_url=audio_url,
    )


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #
class ReferenceCreateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    language: str = "en"
    speaker: str = "default"


class ReferenceResponse(BaseModel):
    id: str
    text: str
    language: str
    speaker: str
    duration_seconds: float
    audio_url: str
    created_at: str


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
class HistoryItem(BaseModel):
    id: str
    created_at: str
    reference_id: str | None = None
    composite_score: int
    composite_grade: str


class PaginatedHistory(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


class AverageScores(BaseModel):
    pronunciation: float
    intonation: float
    fluency: float
    composite: float


class DailyBreakdownItem(BaseModel):
    date: str
    count: int
    avg_composite: int


class StatsResponse(BaseModel):
    period_days: int
    total_evaluations: int
    average_scores: AverageScores
    trend: str
    weakest_phonemes: list[str]
    daily_breakdown: list[DailyBreakdownItem]


def build_history_item(data: dict) -> HistoryItem:
    """Build a HistoryItem from a saved report dict."""
    composite = data.get("composite", {}) or {}
    return HistoryItem(
        id=str(data.get("id", "")),
        created_at=str(data.get("created_at", "")),
        reference_id=data.get("reference_id"),
        composite_score=int(composite.get("score", 0)),
        composite_grade=str(composite.get("grade", "")),
    )


def iter_history_dicts(history_dir) -> Iterable[dict]:
    """Yield every saved report dict in a history dir (used by stats too)."""
    import json
    from pathlib import Path

    base = Path(history_dir)
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.glob("eval_*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out
