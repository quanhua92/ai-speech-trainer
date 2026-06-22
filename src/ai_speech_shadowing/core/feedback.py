"""Feedback engine: unifies the three pillars into a weighted composite report.

Aggregates the phoneme (PER), prosody (pitch-range-ratio), and fluency (DTW)
sub-scores into a single :class:`FeedbackReport` with:

- a configurable weighted composite score (default 40 / 30 / 30),
- colour-coded severity grades (good / fair / needs_work),
- targeted textual feedback derived from the weakest pillars,
- three renderers: JSON (programmatic), terminal (pretty), Markdown.

``build_report`` is pure and fast (unit-tested with synthetic diffs). ``evaluate``
runs the full pipeline — it loads the phoneme model, so it is opt-in slow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.fluency import FluencyDiff
from ai_speech_shadowing.core.phoneme import PhonemeDiff, PhonemeOp
from ai_speech_shadowing.core.prosody import ProsodyDiff
from ai_speech_shadowing.core.wordalign import WordDiff, word_level_diff

DEFAULT_WEIGHTS: tuple[float, float, float] = (0.4, 0.3, 0.3)
"""Composite weights for (pronunciation, intonation, fluency)."""

GOOD_THRESHOLD: float = 80.0
FAIR_THRESHOLD: float = 50.0

_SEVERITY: dict[str, str] = {"good": "🟢", "fair": "🟡", "needs_work": "🔴"}


def grade_for(score100: float) -> str:
    """Map a 0-100 score to a grade."""
    if score100 >= GOOD_THRESHOLD:
        return "good"
    if score100 >= FAIR_THRESHOLD:
        return "fair"
    return "needs_work"


def _score100(x: float) -> int:
    return round(max(0.0, min(1.0, x)) * 100)


# --------------------------------------------------------------------------- #
# Report model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FeedbackReport:
    """The unified, renderer-ready evaluation report."""

    composite_score: int
    composite_grade: str
    pronunciation_score: int
    intonation_score: int
    fluency_score: int
    weights: tuple[float, float, float]
    phoneme_error_rate: float
    pitch_range_ratio: float
    monotone: bool
    dtw_normalized_distance: float
    syllable_rate_reference: float
    syllable_rate_hypothesis: float
    pause_count_reference: int
    pause_count_hypothesis: int
    phoneme_diff: PhonemeDiff
    feedback: tuple[str, ...]
    words: tuple[WordDiff, ...] = ()


def build_report(
    phoneme_diff: PhonemeDiff,
    prosody_diff: ProsodyDiff,
    fluency_diff: FluencyDiff,
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    reference_text: str | None = None,
) -> FeedbackReport:
    """Combine the three pillar diffs into a FeedbackReport.

    Pronunciation score = phoneme accuracy (1 - PER); intonation = prosody
    sub-score; fluency = fluency sub-score. Each is in [0, 100]. The composite
    is the weighted sum. When ``reference_text`` is given, a best-effort
    word-level diff is attached (see :mod:`ai_speech_shadowing.core.wordalign`).
    """
    if not (0.99 <= sum(weights) <= 1.01):
        raise ValueError(f"weights must sum to ~1.0; got {sum(weights)}")

    pron = _score100(phoneme_diff.accuracy)
    into = _score100(prosody_diff.score)
    flu = _score100(fluency_diff.score)
    w0, w1, w2 = weights
    composite = round(pron * w0 + into * w1 + flu * w2)

    feedback = _generate_feedback(phoneme_diff, prosody_diff, fluency_diff, pron, into, flu)

    words: tuple[WordDiff, ...] = ()
    if reference_text:
        words = tuple(
            word_level_diff(reference_text, phoneme_diff.reference, phoneme_diff.operations)
        )

    return FeedbackReport(
        composite_score=composite,
        composite_grade=grade_for(composite),
        pronunciation_score=pron,
        intonation_score=into,
        fluency_score=flu,
        weights=weights,
        phoneme_error_rate=phoneme_diff.phoneme_error_rate,
        pitch_range_ratio=prosody_diff.pitch_range_ratio,
        monotone=prosody_diff.monotone,
        dtw_normalized_distance=fluency_diff.dtw.normalized_distance,
        syllable_rate_reference=fluency_diff.syllable_rate_reference,
        syllable_rate_hypothesis=fluency_diff.syllable_rate_hypothesis,
        pause_count_reference=fluency_diff.reference_pauses.count,
        pause_count_hypothesis=fluency_diff.hypothesis_pauses.count,
        phoneme_diff=phoneme_diff,
        feedback=tuple(feedback),
        words=words,
    )


def _generate_feedback(
    phoneme_diff: PhonemeDiff,
    prosody_diff: ProsodyDiff,
    fluency_diff: FluencyDiff,
    pron_score: int,
    into_score: int,
    flu_score: int,
) -> list[str]:
    """Produce targeted, deterministic suggestions for the weakest pillars."""
    msgs: list[str] = []

    if pron_score < GOOD_THRESHOLD:
        sub = next((op for op in phoneme_diff.operations if op.tag == "sub"), None)
        if sub is not None and sub.ref and sub.hyp:
            msgs.append(
                f"Phoneme /{sub.ref}/ was substituted with /{sub.hyp}/ — focus on tongue placement."
            )
        else:
            msgs.append(
                "Several phonemes were mispronounced; isolate and drill the difficult sounds."
            )

    if prosody_diff.monotone or (
        prosody_diff.pitch_range_ratio < 0.5 and into_score < GOOD_THRESHOLD
    ):
        msgs.append(
            "Your pitch range is narrower than the reference. Try exaggerating "
            "rising tones on question endings."
        )

    if flu_score < GOOD_THRESHOLD and fluency_diff.dtw.normalized_distance > 0.05:
        msgs.append("Your rhythm diverges from the reference; shadow the native pacing.")

    if fluency_diff.hypothesis_pauses.count > fluency_diff.reference_pauses.count:
        msgs.append(
            f"You paused {fluency_diff.hypothesis_pauses.count}x vs the "
            f"reference's {fluency_diff.reference_pauses.count}x — aim for a "
            "steadier flow."
        )

    if fluency_diff.syllable_rate_reference > 0:
        ratio = fluency_diff.syllable_rate_ratio
        if ratio < 0.7:
            msgs.append("You're speaking slower than the reference; try to pick up the pace.")
        elif ratio > 1.3:
            msgs.append("You're speaking faster than the reference; slow down for clarity.")

    if not msgs:
        msgs.append("Great job — your delivery closely matches the reference.")
    return msgs


# --------------------------------------------------------------------------- #
# Full-pipeline orchestrator (loads the phoneme model → opt-in slow)
# --------------------------------------------------------------------------- #
def evaluate(
    reference_sample: AudioSample,
    hypothesis_sample: AudioSample,
    *,
    phoneme_extractor: object | None = None,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    reference_text: str | None = None,
) -> FeedbackReport:
    """Run the full evaluation pipeline and return a FeedbackReport.

    If ``reference_text`` is supplied (the native sentence the user was aiming
    at), the report carries a best-effort word-level diff in addition to the
    exact phoneme-level one.
    """
    from ai_speech_shadowing.core.fluency import compare_fluency
    from ai_speech_shadowing.core.phoneme import diff_phonemes, get_extractor
    from ai_speech_shadowing.core.prosody import compare_pitch, extract_pitch

    extractor = phoneme_extractor or get_extractor()
    ref_phonemes = extractor.extract(reference_sample).phonemes
    hyp_phonemes = extractor.extract(hypothesis_sample).phonemes
    phoneme_diff = diff_phonemes(ref_phonemes, hyp_phonemes)

    prosody_diff = compare_pitch(extract_pitch(reference_sample), extract_pitch(hypothesis_sample))
    fluency_diff = compare_fluency(reference_sample, hypothesis_sample)
    return build_report(
        phoneme_diff,
        prosody_diff,
        fluency_diff,
        weights=weights,
        reference_text=reference_text,
    )


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _op_to_dict(op: PhonemeOp) -> dict[str, str]:
    if op.tag == "match":
        return {"type": "match", "phoneme": op.ref or ""}
    if op.tag == "sub":
        return {"type": "sub", "expected": op.ref or "", "actual": op.hyp or ""}
    if op.tag == "del":
        return {"type": "del", "expected": op.ref or ""}
    return {"type": "ins", "actual": op.hyp or ""}


def report_to_dict(report: FeedbackReport) -> dict[str, object]:
    """Serialize a report to a JSON-friendly dict (matches the API schema)."""
    return {
        "composite": {
            "score": report.composite_score,
            "grade": report.composite_grade,
        },
        "scores": {
            "pronunciation": {
                "score": report.pronunciation_score,
                "grade": grade_for(report.pronunciation_score),
            },
            "intonation": {
                "score": report.intonation_score,
                "grade": grade_for(report.intonation_score),
                "pitch_range_ratio": round(report.pitch_range_ratio, 3),
                "monotone": report.monotone,
            },
            "fluency": {
                "score": report.fluency_score,
                "grade": grade_for(report.fluency_score),
                "dtw_normalized_distance": round(report.dtw_normalized_distance, 4),
                "syllable_rate_reference": round(report.syllable_rate_reference, 3),
                "syllable_rate_hypothesis": round(report.syllable_rate_hypothesis, 3),
                "pause_count_reference": report.pause_count_reference,
                "pause_count_hypothesis": report.pause_count_hypothesis,
            },
        },
        "phoneme_error_rate": round(report.phoneme_error_rate, 4),
        "weights": list(report.weights),
        "phoneme_diff": [_op_to_dict(op) for op in report.phoneme_diff.operations],
        "words": [
            {"word": w.word, "status": w.status, "errors": [dict(e) for e in w.errors]}
            for w in report.words
        ],
        "feedback": list(report.feedback),
    }


def to_json(report: FeedbackReport, *, indent: int = 2) -> str:
    return json.dumps(report_to_dict(report), indent=indent, ensure_ascii=False)


def to_terminal(report: FeedbackReport) -> str:
    """A colour-coded, human-readable terminal report."""
    sep = "─" * 52
    pg, ig, fg = (
        grade_for(report.pronunciation_score),
        grade_for(report.intonation_score),
        grade_for(report.fluency_score),
    )
    lines = [
        "AI Speech Shadowing — Report",
        sep,
        f"Pronunciation (PER):   {report.pronunciation_score:>3}  {_SEVERITY[pg]} {pg}",
        f"Intonation (Pitch):    {report.intonation_score:>3}  {_SEVERITY[ig]} {ig}",
        f"Fluency (DTW):         {report.fluency_score:>3}  {_SEVERITY[fg]} {fg}",
        sep,
        f"Composite Score:       {report.composite_score}/100  "
        f"{_SEVERITY[report.composite_grade]} {report.composite_grade}",
        sep,
    ]
    if report.feedback:
        lines.append("Feedback:")
        for msg in report.feedback:
            lines.append(f"  • {msg}")
    if report.words:
        lines.append("Words (best-effort):")
        rendered = " ".join(f"[{w.word}]" if w.status != "match" else w.word for w in report.words)
        lines.append(f"  {rendered}")
    return "\n".join(lines)


def to_markdown(report: FeedbackReport) -> str:
    """A Markdown rendering of the report."""
    pron_g = grade_for(report.pronunciation_score)
    into_g = grade_for(report.intonation_score)
    flu_g = grade_for(report.fluency_score)
    lines = [
        "# AI Speech Shadowing — Report",
        "",
        f"**Composite Score: {report.composite_score}/100** "
        f"{_SEVERITY[report.composite_grade]} _{report.composite_grade}_",
        "",
        "| Pillar | Score | Grade | Key metric |",
        "| --- | --- | --- | --- |",
        f"| Pronunciation | {report.pronunciation_score} | "
        f"{_SEVERITY[pron_g]} {pron_g} | PER = {report.phoneme_error_rate:.3f} |",
        f"| Intonation | {report.intonation_score} | "
        f"{_SEVERITY[into_g]} {into_g} | range ratio = {report.pitch_range_ratio:.2f} |",
        f"| Fluency | {report.fluency_score} | "
        f"{_SEVERITY[flu_g]} {flu_g} | DTW = {report.dtw_normalized_distance:.3f} |",
        "",
    ]
    if report.feedback:
        lines.append("**Feedback:**")
        for msg in report.feedback:
            lines.append(f"- {msg}")
    if report.words:
        lines.append("")
        lines.append("**Words (best-effort highlight):**")
        rendered = " ".join(
            f"**{w.word}**" if w.status != "match" else w.word for w in report.words
        )
        lines.append(rendered)
    return "\n".join(lines)
