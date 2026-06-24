"""Tests for the feedback engine: scoring, weighting, feedback, renderers.

``build_report`` and the renderers are pure → fast unit tests with synthetic
diffs. ``evaluate`` (full pipeline) is opt-in slow — under ``--runslow`` it
generates Kokoro references and loads the phoneme model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.feedback import (
    FeedbackReport,
    build_report,
    evaluate,
    grade_for,
    report_to_dict,
    to_json,
    to_markdown,
    to_terminal,
)
from ai_speech_shadowing.core.fluency import DtwResult, FluencyDiff, PauseInfo
from ai_speech_shadowing.core.phoneme import diff_phonemes
from ai_speech_shadowing.core.preprocess import preprocess
from ai_speech_shadowing.core.prosody import PitchStats, ProsodyDiff


# --------------------------------------------------------------------------- #
# Synthetic-diff factories (no model, no audio)
# --------------------------------------------------------------------------- #
def _pitch_stats(range_hz: float = 200.0, voiced: bool = True) -> PitchStats:
    return PitchStats(
        f0_contour=np.zeros(1, dtype=np.float64),
        times=np.zeros(1, dtype=np.float64),
        mean_hz=200.0,
        median_hz=200.0,
        min_hz=100.0,
        max_hz=100.0 + range_hz,
        range_hz=range_hz if voiced else 0.0,
        std_hz=20.0,
        voiced_ratio=1.0 if voiced else 0.0,
        pitch_floor=75.0,
        pitch_ceiling=500.0,
    )


def _prosody(score: float, ratio: float = 1.0, monotone: bool = False) -> ProsodyDiff:
    return ProsodyDiff(
        reference=_pitch_stats(),
        hypothesis=_pitch_stats(range_hz=max(ratio * 200.0, 1.0)),
        pitch_range_ratio=ratio,
        monotone=monotone,
        monotone_threshold=0.5,
        score=score,
    )


def _pauses(count: int) -> PauseInfo:
    durations = tuple(0.3 for _ in range(count))
    return PauseInfo(count=count, total_seconds=float(sum(durations)), durations=durations)


def _fluency(
    score: float = 1.0,
    norm: float = 0.0,
    ref_pauses: int = 0,
    hyp_pauses: int = 0,
    ref_rate: float = 2.0,
    hyp_rate: float = 2.0,
) -> FluencyDiff:
    return FluencyDiff(
        dtw=DtwResult(distance=norm * 10, path_length=10, normalized_distance=norm),
        score=score,
        reference_pauses=_pauses(ref_pauses),
        hypothesis_pauses=_pauses(hyp_pauses),
        syllable_rate_reference=ref_rate,
        syllable_rate_hypothesis=hyp_rate,
        syllable_rate_ratio=(hyp_rate / ref_rate) if ref_rate > 0 else 0.0,
    )


def _perfect_report() -> FeedbackReport:
    return build_report(
        diff_phonemes(["h", "ə", "l", "oʊ"], ["h", "ə", "l", "oʊ"]),
        _prosody(score=1.0, ratio=1.0),
        _fluency(score=1.0, norm=0.0),
    )


# --------------------------------------------------------------------------- #
# Scoring & weighting
# --------------------------------------------------------------------------- #
class TestBuildReport:
    def test_perfect_report(self) -> None:
        report = _perfect_report()
        assert report.composite_score == 100
        assert report.composite_grade == "good"
        assert (report.pronunciation_score, report.intonation_score, report.fluency_score) == (
            100,
            100,
            100,
        )
        # perfect delivery -> the positive "Great job" message
        assert any("closely matches" in m for m in report.feedback)

    def test_composite_is_weighted(self) -> None:
        pron = diff_phonemes(["a"], ["a"])  # perfect -> 100
        report = build_report(pron, _prosody(0.0), _fluency(0.0))
        # 100*0.4 + 0*0.3 + 0*0.3 == 40
        assert report.composite_score == 40
        assert report.composite_grade == "needs_work"

    def test_custom_weights(self) -> None:
        pron = diff_phonemes(["a"], ["a"])
        # weight pronunciation exclusively
        report = build_report(pron, _prosody(0.0), _fluency(0.0), weights=(1.0, 0.0, 0.0))
        assert report.composite_score == 100

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="weights must sum"):
            build_report(
                diff_phonemes(["a"], ["a"]),
                _prosody(1.0),
                _fluency(1.0),
                weights=(0.5, 0.3, 0.3),
            )

    def test_per_drives_score_from_accuracy(self) -> None:
        # one substitution out of four -> PER 0.25 -> accuracy 0.75 -> 75
        diff = diff_phonemes(["a", "b", "c", "d"], ["a", "x", "c", "d"])
        report = build_report(diff, _prosody(1.0), _fluency(1.0))
        assert report.pronunciation_score == 75
        assert report.phoneme_error_rate == pytest.approx(0.25)


class TestGradeThresholds:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (100, "good"),
            (80, "good"),
            (79, "fair"),
            (50, "fair"),
            (49, "needs_work"),
            (0, "needs_work"),
        ],
    )
    def test_grade_for(self, score: int, expected: str) -> None:
        assert grade_for(score) == expected


# --------------------------------------------------------------------------- #
# Feedback messages
# --------------------------------------------------------------------------- #
class TestFeedback:
    def test_substitution_message(self) -> None:
        diff = diff_phonemes(["h", "ə", "l", "oʊ"], ["h", "ə", "ɹ", "oʊ"])
        report = build_report(diff, _prosody(1.0), _fluency(1.0))
        assert any("/l/" in m and "/ɹ/" in m for m in report.feedback)

    def test_monotone_message(self) -> None:
        report = build_report(
            diff_phonemes(["a"], ["a"]),
            _prosody(score=0.3, ratio=0.3, monotone=True),
            _fluency(1.0),
        )
        assert any("pitch range is narrower" in m for m in report.feedback)

    def test_rhythm_message_when_fluency_weak(self) -> None:
        report = build_report(
            diff_phonemes(["a"], ["a"]),
            _prosody(1.0),
            _fluency(score=0.3, norm=0.2),
        )
        assert any("shadow" in m for m in report.feedback)

    def test_pause_message(self) -> None:
        report = build_report(
            diff_phonemes(["a"], ["a"]),
            _prosody(1.0),
            _fluency(score=0.5, norm=0.05, ref_pauses=0, hyp_pauses=2),
        )
        assert any("paused" in m for m in report.feedback)

    def test_rate_drift_slow_message(self) -> None:
        report = build_report(
            diff_phonemes(["a"], ["a"]),
            _prosody(1.0),
            _fluency(score=1.0, norm=0.0, ref_rate=3.0, hyp_rate=1.5),
        )
        assert any("slower" in m for m in report.feedback)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
class TestRenderers:
    def test_to_json_round_trips(self) -> None:
        report = _perfect_report()
        data = json.loads(to_json(report))
        assert data["composite"]["score"] == 100
        assert data["composite"]["grade"] == "good"
        assert "scores" in data and "pronunciation" in data["scores"]
        assert isinstance(data["phoneme_diff"], list)
        assert isinstance(data["feedback"], list)

    def test_report_to_dict_phoneme_ops(self) -> None:
        diff = diff_phonemes(["a", "b"], ["a", "c"])  # one sub
        report = build_report(diff, _prosody(1.0), _fluency(1.0))
        ops = report_to_dict(report)["phoneme_diff"]
        kinds = {op["type"] for op in ops}
        assert "match" in kinds and "sub" in kinds

    def test_to_terminal_has_scores_and_severity(self) -> None:
        report = _perfect_report()
        text = to_terminal(report)
        assert "Composite Score" in text
        assert "🟢" in text  # good severity marker
        assert "Pronunciation" in text

    def test_to_markdown_has_table(self) -> None:
        report = _perfect_report()
        md = to_markdown(report)
        assert md.startswith("# AI Speech Shadowing")
        assert "| Pillar | Score |" in md


# --------------------------------------------------------------------------- #
# Full pipeline (opt-in slow: needs the phoneme model + Kokoro)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestEvaluate:
    def test_identical_clip_scores_high(self, kokoro_ref_wav: Path) -> None:
        ref = preprocess(AudioSample.from_wav(kokoro_ref_wav))
        report = evaluate(ref, ref)
        assert report.composite_score >= 90
        assert report.composite_grade == "good"
        # renderers work on a real report
        assert json.loads(to_json(report))["composite"]["score"] >= 90
        assert "Composite Score" in to_terminal(report)

    def test_different_clips_produce_feedback(
        self, kokoro_ref_wav: Path, kokoro_alt_wav: Path
    ) -> None:
        ref = preprocess(AudioSample.from_wav(kokoro_ref_wav))
        hyp = preprocess(AudioSample.from_wav(kokoro_alt_wav))
        report = evaluate(ref, hyp)
        assert report.composite_score < 100
        assert len(report.feedback) > 0


# --------------------------------------------------------------------------- #
# Asymmetric phoneme sourcing (fast: uses a fake extractor, no model load)
# --------------------------------------------------------------------------- #
class _FakeExtractor:
    """Records ``extract()`` calls; returns canned phonemes regardless of audio.

    Mirrors the ``PhonemeExtractor`` shape used by :func:`evaluate` (which
    accesses it duck-typed via the ``phoneme_extractor=`` parameter).
    """

    def __init__(self, phonemes: tuple[str, ...] = ("h", "ə", "l", "oʊ")) -> None:
        from ai_speech_shadowing.core.phoneme import PhonemeResult

        self._result = PhonemeResult(phonemes=phonemes, raw_text=" ".join(phonemes))
        self.calls: list[AudioSample] = []

    def extract(self, sample: AudioSample):  # type: ignore[no-untyped-def]
        self.calls.append(sample)
        return self._result


class TestReferencePhonemeSource:
    """Verify the G2P-vs-acoustic branch in :func:`evaluate`.

    These tests run fast because they inject a fake extractor instead of loading
    the 1.2 GB Wav2Vec2 model — the only thing under test is the branching
    logic and the ``reference_phoneme_source`` stamp.
    """

    def test_g2p_path_skips_extractor_on_reference(
        self, mono_44100_wav: Path, silent_wav: Path
    ) -> None:
        ref = preprocess(AudioSample.from_wav(mono_44100_wav))
        hyp = preprocess(AudioSample.from_wav(silent_wav))
        fake = _FakeExtractor()

        report = evaluate(
            ref,
            hyp,
            phoneme_extractor=fake,
            reference_phonemes=["h", "ə", "l", "oʊ"],
        )

        # Only the hypothesis went through the acoustic recognizer.
        assert len(fake.calls) == 1
        assert fake.calls[0] is hyp
        # The reference sequence came from the G2P input verbatim.
        assert report.phoneme_diff.reference == ("h", "ə", "l", "oʊ")
        assert report.reference_phoneme_source == "kokoro-g2p"

    def test_acoustic_fallback_runs_extractor_on_both(
        self, mono_44100_wav: Path, silent_wav: Path
    ) -> None:
        ref = preprocess(AudioSample.from_wav(mono_44100_wav))
        hyp = preprocess(AudioSample.from_wav(silent_wav))
        fake = _FakeExtractor()

        report = evaluate(ref, hyp, phoneme_extractor=fake)  # no reference_phonemes

        # Both sides went through the recognizer (legacy behavior).
        assert len(fake.calls) == 2
        assert report.reference_phoneme_source == "wav2vec2-acoustic"

    def test_default_source_is_acoustic(self) -> None:
        # A report built directly (not through evaluate) defaults to acoustic
        # provenance, matching the legacy behavior before the G2P path existed.
        report = build_report(diff_phonemes(["a"], ["a"]), _prosody(1.0), _fluency(1.0))
        assert report.reference_phoneme_source == "wav2vec2-acoustic"

    def test_report_to_dict_carries_source(self) -> None:
        report = build_report(
            diff_phonemes(["a"], ["a"]),
            _prosody(1.0),
            _fluency(1.0),
            reference_phoneme_source="kokoro-g2p",
        )
        data = report_to_dict(report)
        assert data["reference_phoneme_source"] == "kokoro-g2p"
