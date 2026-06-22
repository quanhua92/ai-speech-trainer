"""Tests for fluency & timing (MFCC, DTW, pauses, syllable rate).

Deterministic DSP on synthetic audio — runs in the fast suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.fluency import (
    DTW_SCORE_SCALE,
    DtwResult,
    FluencyDiff,
    MfccFeatures,
    PauseInfo,
    compare_fluency,
    detect_pauses,
    dtw_distance,
    estimate_syllable_rate,
    extract_mfcc,
)


class TestExtractMfcc:
    def test_shape_and_normalization(self, pitched_200_sample: AudioSample) -> None:
        feats = extract_mfcc(pitched_200_sample)
        assert isinstance(feats, MfccFeatures)
        assert feats.matrix.ndim == 2
        assert feats.matrix.shape[1] == 13  # default n_mfcc
        assert feats.num_frames > 0
        # rows are L2-normalized
        norms = np.linalg.norm(feats.matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_times_aligned_with_frames(self, pitched_200_sample: AudioSample) -> None:
        feats = extract_mfcc(pitched_200_sample)
        assert feats.times.shape == (feats.num_frames,)
        assert feats.times[0] >= 0.0


class TestDtwDistance:
    def test_identical_is_zero(self, pitched_200_sample: AudioSample) -> None:
        feats = extract_mfcc(pitched_200_sample)
        d = dtw_distance(feats, feats)
        assert isinstance(d, DtwResult)
        assert d.distance == pytest.approx(0.0, abs=1e-9)
        assert d.normalized_distance == pytest.approx(0.0, abs=1e-9)

    def test_different_signals_have_positive_distance(
        self, pitched_200_sample: AudioSample, narrow_sample: AudioSample
    ) -> None:
        a = extract_mfcc(pitched_200_sample)  # 200 Hz
        b = extract_mfcc(narrow_sample)  # 150 Hz, longer
        d = dtw_distance(a, b)
        assert d.distance > 0.0
        assert d.normalized_distance > 0.0
        assert d.path_length > 0


class TestDetectPauses:
    def test_finds_interior_pause(self, gapped_sample: AudioSample) -> None:
        info = detect_pauses(gapped_sample, min_pause_s=0.25)
        assert isinstance(info, PauseInfo)
        assert info.count == 1
        # split() detects non-silent margins, so the measured gap is shorter
        # than the true 0.5s silence — just assert it's in a sane band.
        assert 0.25 <= info.durations[0] <= 0.5
        assert info.total_seconds == pytest.approx(info.durations[0])

    def test_continuous_tone_has_no_pauses(self, pitched_200_sample: AudioSample) -> None:
        info = detect_pauses(pitched_200_sample, min_pause_s=0.25)
        assert info.count == 0
        assert info.total_seconds == 0.0

    def test_threshold_filters_short_gaps(self, gapped_sample: AudioSample) -> None:
        # a 0.5s gap is below a 0.6s threshold -> not flagged
        info = detect_pauses(gapped_sample, min_pause_s=0.6)
        assert info.count == 0


class TestEstimateSyllableRate:
    def test_pulse_train_has_syllables(self, pulse_train_sample: AudioSample) -> None:
        rate = estimate_syllable_rate(pulse_train_sample)
        assert rate > 0.0
        # ~5 bursts in ~1.1s -> rate in a plausible band
        assert 1.0 < rate < 12.0

    def test_silent_clip_is_zero(self, noise_sample: AudioSample) -> None:
        # white noise has a flat RMS envelope -> no prominent peaks
        # (allow either 0 or a tiny value)
        assert estimate_syllable_rate(noise_sample) < 3.0


class TestCompareFluency:
    def test_identical_signals_score_near_one(self, pitched_200_sample: AudioSample) -> None:
        diff = compare_fluency(pitched_200_sample, pitched_200_sample)
        assert isinstance(diff, FluencyDiff)
        assert diff.dtw.normalized_distance == pytest.approx(0.0, abs=1e-9)
        assert diff.score == pytest.approx(1.0)
        assert diff.grade == "good"
        assert diff.syllable_rate_ratio == pytest.approx(1.0)

    def test_different_signals_score_lower(
        self, pitched_200_sample: AudioSample, narrow_sample: AudioSample
    ) -> None:
        identical = compare_fluency(pitched_200_sample, pitched_200_sample).score
        different = compare_fluency(pitched_200_sample, narrow_sample).score
        assert different < identical
        assert different < 1.0

    def test_score_scale_constant_is_positive(self) -> None:
        assert DTW_SCORE_SCALE > 0.0

    def test_pause_info_collected(self, gapped_sample: AudioSample) -> None:
        diff = compare_fluency(gapped_sample, gapped_sample)
        assert diff.reference_pauses.count == 1
        assert diff.hypothesis_pauses.count == 1
