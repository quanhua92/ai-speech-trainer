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
from typing import TYPE_CHECKING

from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.fluency import FluencyDiff
from ai_speech_shadowing.core.phoneme import PhonemeDiff, PhonemeOp
from ai_speech_shadowing.core.prosody import ProsodyDiff
from ai_speech_shadowing.core.wordalign import WordDiff, word_level_diff

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    reference_phoneme_source: str = "wav2vec2-acoustic"
    """Provenance of the reference phoneme sequence.

    One of: ``"kokoro-g2p"`` (captured from Kokoro at synthesis), 
    ``"transcript-g2p"`` (misaki on a user-supplied transcript), or
    ``"wav2vec2-acoustic"`` (the fallback when no text is available).
    """


def build_report(
    phoneme_diff: PhonemeDiff,
    prosody_diff: ProsodyDiff,
    fluency_diff: FluencyDiff,
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    reference_text: str | None = None,
    language: str = "en",
    reference_phoneme_source: str = "wav2vec2-acoustic",
) -> FeedbackReport:
    """Combine the three pillar diffs into a FeedbackReport.

    Pronunciation score = phoneme accuracy (1 - PER); intonation = prosody
    sub-score; fluency = fluency sub-score. Each is in [0, 100]. The composite
    is the weighted sum. When ``reference_text`` is given, a best-effort
    word-level diff is attached (see :mod:`ai_speech_shadowing.core.wordalign`).

    ``reference_phoneme_source`` records where the reference phoneme sequence
    came from so the client/UI can render it honestly (G2P target vs. acoustic
    recognition).
    """
    if not (0.99 <= sum(weights) <= 1.01):
        raise ValueError(f"weights must sum to ~1.0; got {sum(weights)}")

    pron = _score100(phoneme_diff.accuracy)
    into = _score100(prosody_diff.score)
    flu = _score100(fluency_diff.score)
    w0, w1, w2 = weights
    composite = round(pron * w0 + into * w1 + flu * w2)

    feedback = _generate_feedback(
        phoneme_diff, prosody_diff, fluency_diff, pron, into, flu, language=language
    )

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
        reference_phoneme_source=reference_phoneme_source,
    )


# Bilingual feedback message templates (en / vi). Placeholder fields are
# filled by _fb() at format time.
_FB: dict[str, dict[str, str]] = {
    "pron_excellent": {
        "en": "Pronunciation is excellent (PER {per:.3f}) - all sounds matched.",
        "vi": "Phát âm xuất sắc (PER {per:.3f}) - tất cả âm đều khớp.",
    },
    "pron_sub": {
        "en": "Pronunciation needs work (PER {per:.2f}): /{exp}/ -> /{act}/"
        " - focus on tongue placement.",
        "vi": "Phát âm cần cải thiện (PER {per:.2f}): /{exp}/ -> /{act}/ - chú ý vị trí lưỡi.",
    },
    "pron_general": {
        "en": "Pronunciation needs work (PER {per:.2f}, {n} error(s))"
        " - drill the difficult sounds.",
        "vi": "Phát âm cần cải thiện (PER {per:.2f}, {n} lỗi) - tập luyện các âm khó.",
    },
    "into_narrow": {
        "en": "Your pitch range is narrower than the reference"
        " (ratio {ratio:.2f}) - exaggerate rising tones.",
        "vi": "Dải cao độ hẹp hơn bản gốc (tỉ lệ {ratio:.2f}) - thử nhấn mạnh ngữ điệu lên.",
    },
    "into_excellent": {
        "en": "Intonation is excellent (ratio {ratio:.2f}) - pitch contour matches.",
        "vi": "Ngữ điệu xuất sắc (tỉ lệ {ratio:.2f}) - đường cao độ khớp bản gốc.",
    },
    "flu_weak": {
        "en": "Fluency is the weak spot (DTW {dtw:.3f}) - shadow the native pacing.",
        "vi": "Nhịp điệu là điểm yếu (DTW {dtw:.3f}) - nhái theo nhịp chuẩn.",
    },
    "flu_good": {
        "en": "Fluency is good (DTW {dtw:.3f}) - rhythm closely matches.",
        "vi": "Nhịp điệu tốt (DTW {dtw:.3f}) - nhịp khá khớp.",
    },
    "pauses": {
        "en": "You paused {hyp}x vs the reference's {ref}x - steadier flow.",
        "vi": "Bạn dừng {hyp} lần, bản gốc {ref} lần - nói trôi chảy hơn.",
    },
    "rate_slow": {
        "en": "Speaking {pct:.0f}% slower ({hyp:.1f} vs {ref:.1f} syll/s).",
        "vi": "Nói chậm hơn {pct:.0f}% ({hyp:.1f} vs {ref:.1f} âm/tiết/giây).",
    },
    "rate_fast": {
        "en": "Speaking {pct:.0f}% faster ({hyp:.1f} vs {ref:.1f}) syll/s - slow down.",
        "vi": "Nói nhanh hơn {pct:.0f}% ({hyp:.1f} vs {ref:.1f}) âm/tiết/giây - chậm lại.",
    },
    "great": {
        "en": "Great job - your delivery closely matches the reference.",
        "vi": "Rất tốt - phần đọc của bạn gần giống bản gốc.",
    },
}


def _fb(key: str, lang: str = "en", **kw: object) -> str:
    """Look up a bilingual feedback template and format it."""
    entry = _FB.get(key, {"en": key})
    template = entry.get(lang, entry["en"])
    return template.format(**kw)


def _generate_feedback(
    phoneme_diff: PhonemeDiff,
    prosody_diff: ProsodyDiff,
    fluency_diff: FluencyDiff,
    pron_score: int,
    into_score: int,
    flu_score: int,
    language: str = "en",
) -> list[str]:
    """Produce targeted, deterministic feedback in the requested language."""
    msgs: list[str] = []
    lang = language if language in ("en", "vi") else "en"

    # ---- Pronunciation ----
    if pron_score < GOOD_THRESHOLD:
        sub = next((op for op in phoneme_diff.operations if op.tag == "sub"), None)
        if sub is not None and sub.ref and sub.hyp:
            msgs.append(
                _fb("pron_sub", lang, per=phoneme_diff.phoneme_error_rate, exp=sub.ref, act=sub.hyp)
            )
        else:
            errs = phoneme_diff.substitutions + phoneme_diff.deletions + phoneme_diff.insertions
            msgs.append(_fb("pron_general", lang, per=phoneme_diff.phoneme_error_rate, n=errs))
    elif pron_score >= 90:
        msgs.append(_fb("pron_excellent", lang, per=phoneme_diff.phoneme_error_rate))

    # ---- Intonation ----
    if prosody_diff.monotone or (
        prosody_diff.pitch_range_ratio < 0.5 and into_score < GOOD_THRESHOLD
    ):
        msgs.append(_fb("into_narrow", lang, ratio=prosody_diff.pitch_range_ratio))
    elif into_score >= 90:
        msgs.append(_fb("into_excellent", lang, ratio=prosody_diff.pitch_range_ratio))

    # ---- Fluency ----
    if flu_score < GOOD_THRESHOLD and fluency_diff.dtw.normalized_distance > 0.05:
        msgs.append(_fb("flu_weak", lang, dtw=fluency_diff.dtw.normalized_distance))
    elif flu_score >= 80:
        msgs.append(_fb("flu_good", lang, dtw=fluency_diff.dtw.normalized_distance))

    # ---- Pauses ----
    if fluency_diff.hypothesis_pauses.count > fluency_diff.reference_pauses.count:
        msgs.append(
            _fb(
                "pauses",
                lang,
                hyp=fluency_diff.hypothesis_pauses.count,
                ref=fluency_diff.reference_pauses.count,
            )
        )

    # ---- Speaking rate ----
    if fluency_diff.syllable_rate_reference > 0:
        ratio = fluency_diff.syllable_rate_ratio
        if ratio < 0.7:
            msgs.append(
                _fb(
                    "rate_slow",
                    lang,
                    pct=(1 - ratio) * 100,
                    hyp=fluency_diff.syllable_rate_hypothesis,
                    ref=fluency_diff.syllable_rate_reference,
                )
            )
        elif ratio > 1.3:
            msgs.append(
                _fb(
                    "rate_fast",
                    lang,
                    pct=(ratio - 1) * 100,
                    hyp=fluency_diff.syllable_rate_hypothesis,
                    ref=fluency_diff.syllable_rate_reference,
                )
            )

    if not msgs:
        msgs.append(_fb("great", lang))
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
    reference_phonemes: Sequence[str] | None = None,
    dtw_score_scale: float | None = None,
    feedback_language: str = "en",
) -> FeedbackReport:
    """Run the full evaluation pipeline and return a FeedbackReport.

    Phoneme sourcing is **asymmetric**:

    - When ``reference_phonemes`` is provided (the canonical G2P target —
      typically captured at TTS synthesis time and read from
      ``metadata.json["phonemes"]["tokens"]``), it is used directly as the
      reference sequence. No acoustic model is run on the reference audio, and
      ``reference_phoneme_source`` is stamped ``"kokoro-g2p"``. This is the
      correct path for any reference whose text is known a priori.
    - When ``reference_phonemes`` is ``None`` (e.g. a future user-uploaded clip
      without transcript), the reference phonemes are decoded acoustically via
      the Wav2Vec2 model, and ``reference_phoneme_source`` is stamped
      ``"wav2vec2-acoustic"``. This preserves the legacy behavior.

    The hypothesis (user) side always goes through the acoustic recognizer —
    that's the only way to hear what the user physically said.
    """
    from ai_speech_shadowing.core.fluency import compare_fluency
    from ai_speech_shadowing.core.phoneme import diff_phonemes, get_extractor
    from ai_speech_shadowing.core.prosody import compare_pitch, extract_pitch

    extractor = phoneme_extractor or get_extractor()
    if reference_phonemes is not None:
        # G2P target path: skip acoustic recognition on the reference side.
        ref_phonemes = tuple(reference_phonemes)
        ref_source = "kokoro-g2p"
    else:
        # Acoustic fallback (no transcript / uploaded clip path).
        ref_phonemes = extractor.extract(reference_sample).phonemes
        ref_source = "wav2vec2-acoustic"
    hyp_phonemes = extractor.extract(hypothesis_sample).phonemes
    phoneme_diff = diff_phonemes(ref_phonemes, hyp_phonemes)

    prosody_diff = compare_pitch(extract_pitch(reference_sample), extract_pitch(hypothesis_sample))
    fluency_kwargs: dict[str, object] = {}
    if dtw_score_scale is not None:
        fluency_kwargs["dtw_score_scale"] = dtw_score_scale
    fluency_diff = compare_fluency(reference_sample, hypothesis_sample, **fluency_kwargs)
    return build_report(
        phoneme_diff,
        prosody_diff,
        fluency_diff,
        weights=weights,
        reference_text=reference_text,
        language=feedback_language,
        reference_phoneme_source=ref_source,
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
        "reference_phoneme_source": report.reference_phoneme_source,
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
