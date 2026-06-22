"""Tests for pitch/prosody extraction and comparison.

These are deterministic DSP tests on synthetic tones — no model download, so
they run in the fast suite (unlike the phoneme model tests).
"""

from __future__ import annotations

import pytest

from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.prosody import (
    DEFAULT_MONOTONE_THRESHOLD,
    PitchStats,
    ProsodyDiff,
    compare_pitch,
    extract_pitch,
)


class TestExtractPitch:
    def test_tracks_fundamental(self, pitched_200_sample: AudioSample) -> None:
        stats = extract_pitch(pitched_200_sample)
        assert stats.is_voiced
        assert stats.mean_hz == pytest.approx(200.0, abs=5.0)
        assert stats.voiced_ratio > 0.9

    def test_range_concatenation(self, wide_range_sample: AudioSample) -> None:
        stats = extract_pitch(wide_range_sample)
        assert stats.is_voiced
        assert stats.min_hz == pytest.approx(100.0, abs=10.0)
        assert stats.max_hz == pytest.approx(300.0, abs=10.0)
        assert stats.range_hz == pytest.approx(200.0, abs=20.0)
        assert stats.max_hz - stats.min_hz == pytest.approx(stats.range_hz)

    def test_contour_and_times_aligned(self, pitched_200_sample: AudioSample) -> None:
        stats = extract_pitch(pitched_200_sample)
        assert stats.f0_contour.shape == stats.times.shape
        assert stats.f0_contour.ndim == 1
        # times are increasing seconds within the 1s clip
        assert stats.times[0] >= 0.0
        assert stats.times[-1] <= 1.0 + 1e-6

    def test_unvoiced_noise(self, noise_sample: AudioSample) -> None:
        stats = extract_pitch(noise_sample)
        assert not stats.is_voiced
        assert stats.voiced_ratio == 0.0
        assert stats.mean_hz == 0.0
        assert stats.range_hz == 0.0

    def test_invalid_pitch_bounds(self, pitched_200_sample: AudioSample) -> None:
        with pytest.raises(ValueError, match="pitch_floor"):
            extract_pitch(pitched_200_sample, pitch_floor=0.0)
        with pytest.raises(ValueError, match="pitch_floor"):
            extract_pitch(pitched_200_sample, pitch_floor=400.0, pitch_ceiling=300.0)

    def test_returns_typed_stats(self, pitched_200_sample: AudioSample) -> None:
        stats = extract_pitch(pitched_200_sample)
        assert isinstance(stats, PitchStats)
        assert stats.pitch_floor == 75.0
        assert stats.pitch_ceiling == 500.0


class TestComparePitch:
    def test_wide_vs_wide_not_monotone(
        self, wide_range_sample: AudioSample, pitched_200_sample: AudioSample
    ) -> None:
        ref = extract_pitch(wide_range_sample)  # range ~200
        hyp = extract_pitch(wide_range_sample)  # identical -> ratio 1.0
        d = compare_pitch(ref, hyp)
        assert d.pitch_range_ratio == pytest.approx(1.0)
        assert not d.monotone
        assert d.score == pytest.approx(1.0)
        assert d.grade == "good"

    def test_monotone_narrow_against_wide(
        self, wide_range_sample: AudioSample, narrow_sample: AudioSample
    ) -> None:
        ref = extract_pitch(wide_range_sample)  # range ~200
        hyp = extract_pitch(narrow_sample)  # pure 150 Hz -> range ~0
        d = compare_pitch(ref, hyp)
        assert d.pitch_range_ratio < DEFAULT_MONOTONE_THRESHOLD
        assert d.monotone is True
        assert d.score < 0.5

    def test_threshold_is_configurable(
        self, wide_range_sample: AudioSample, narrow_sample: AudioSample
    ) -> None:
        ref = extract_pitch(wide_range_sample)
        hyp = extract_pitch(narrow_sample)
        # with a very permissive threshold, the narrow tone isn't "monotone"
        d = compare_pitch(ref, hyp, monotone_threshold=0.0)
        assert d.monotone is False
        assert d.monotone_threshold == 0.0

    def test_score_caps_exaggerated_range(self, wide_range_sample: AudioSample) -> None:
        ref = extract_pitch(wide_range_sample)
        # user with double the reference range: ratio 2.0 -> score capped at 1.0
        exaggerated = PitchStats(
            f0_contour=ref.f0_contour,
            times=ref.times,
            mean_hz=ref.mean_hz,
            median_hz=ref.median_hz,
            min_hz=ref.min_hz,
            max_hz=ref.max_hz + 2 * ref.range_hz,
            range_hz=3 * ref.range_hz,
            std_hz=ref.std_hz,
            voiced_ratio=ref.voiced_ratio,
            pitch_floor=ref.pitch_floor,
            pitch_ceiling=ref.pitch_ceiling,
        )
        d = compare_pitch(ref, exaggerated)
        assert d.pitch_range_ratio == pytest.approx(3.0)
        assert d.score == 1.0  # capped, not 3.0
        assert d.grade == "good"

    def test_unvoiced_hypothesis(
        self, wide_range_sample: AudioSample, noise_sample: AudioSample
    ) -> None:
        ref = extract_pitch(wide_range_sample)
        hyp = extract_pitch(noise_sample)
        d = compare_pitch(ref, hyp)
        assert d.pitch_range_ratio == 0.0
        assert d.score == 0.0
        assert d.monotone is False  # unvoiced != monotone
        assert d.grade == "needs_work"

    def test_unvoiced_reference(
        self, noise_sample: AudioSample, wide_range_sample: AudioSample
    ) -> None:
        ref = extract_pitch(noise_sample)
        hyp = extract_pitch(wide_range_sample)
        d = compare_pitch(ref, hyp)
        assert d.pitch_range_ratio == 0.0
        assert d.score == 0.0

    def test_returns_typed_diff(
        self, wide_range_sample: AudioSample, narrow_sample: AudioSample
    ) -> None:
        ref = extract_pitch(wide_range_sample)
        hyp = extract_pitch(narrow_sample)
        d = compare_pitch(ref, hyp)
        assert isinstance(d, ProsodyDiff)
        assert isinstance(d.reference, PitchStats)
        assert isinstance(d.hypothesis, PitchStats)
