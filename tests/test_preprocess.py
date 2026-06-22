"""Tests for the preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_speech_shadowing.core.audio import TARGET_SAMPLE_RATE, AudioSample
from ai_speech_shadowing.core.preprocess import (
    normalize_volume,
    preprocess,
    resample,
    to_mono,
    trim_silence,
)


class TestToMono:
    def test_stereo_downmix_is_mono_and_length_preserved(self, stereo_48000_wav: Path) -> None:
        s = AudioSample.from_wav(stereo_48000_wav)
        assert not s.is_mono
        mono = to_mono(s)
        assert mono.is_mono
        assert mono.sample_rate == s.sample_rate
        assert mono.num_samples == s.num_samples

    def test_mono_is_noop(self, mono_44100_wav: Path) -> None:
        s = AudioSample.from_wav(mono_44100_wav)
        assert to_mono(s) is s


class TestResample:
    def test_downsample_changes_length(self, mono_44100_wav: Path) -> None:
        s = AudioSample.from_wav(mono_44100_wav)
        out = resample(s, TARGET_SAMPLE_RATE)
        assert out.sample_rate == TARGET_SAMPLE_RATE
        # 44100 -> 16000 keeps ratio
        assert out.num_samples == pytest.approx(
            int(s.num_samples * TARGET_SAMPLE_RATE / s.sample_rate), abs=2
        )

    def test_noop_at_target_rate(self, mono_44100_wav: Path) -> None:
        s = AudioSample.from_wav(mono_44100_wav)
        # build a sample already at target
        already = AudioSample(waveform=s.waveform, sample_rate=TARGET_SAMPLE_RATE)
        assert resample(already, TARGET_SAMPLE_RATE) is already

    def test_stereo_resample_preserves_channels(self, stereo_48000_wav: Path) -> None:
        s = AudioSample.from_wav(stereo_48000_wav)
        out = resample(s, TARGET_SAMPLE_RATE)
        assert out.channels == 2
        assert out.sample_rate == TARGET_SAMPLE_RATE


class TestTrimSilence:
    def test_strips_silence_pads(self, silence_padded_wav: Path) -> None:
        s = AudioSample.from_wav(silence_padded_wav)
        # original has ~0.3s silence on each side of 0.5s tone (~1.1s total)
        assert s.duration > 1.0
        trimmed = trim_silence(s)
        # tone core (~0.5s) survives; silence gone
        assert trimmed.duration < 0.7
        assert trimmed.duration > 0.3

    def test_requires_mono(self, stereo_48000_wav: Path) -> None:
        s = AudioSample.from_wav(stereo_48000_wav)
        with pytest.raises(ValueError, match="mono"):
            trim_silence(s)

    def test_all_silence_returned_unchanged(self, silent_wav: Path) -> None:
        s = AudioSample.from_wav(silent_wav)
        out = trim_silence(s)
        assert out.num_samples == s.num_samples


class TestNormalizeVolume:
    def test_peak_hits_target(self, quiet_wav: Path) -> None:
        s = AudioSample.from_wav(quiet_wav)
        assert np.max(np.abs(s.waveform)) == pytest.approx(0.01, abs=1e-4)
        out = normalize_volume(s, method="peak")
        assert np.max(np.abs(out.waveform)) == pytest.approx(0.99, abs=1e-3)

    def test_rms_raises_amplitude(self, quiet_wav: Path) -> None:
        s = AudioSample.from_wav(quiet_wav)
        out = normalize_volume(s, method="rms")
        assert np.sqrt(np.mean(out.waveform**2)) > np.sqrt(np.mean(s.waveform**2))

    def test_silent_is_noop(self, silent_wav: Path) -> None:
        s = AudioSample.from_wav(silent_wav)
        out = normalize_volume(s, method="peak")
        assert np.array_equal(out.waveform, s.waveform)

    def test_unknown_method(self, mono_44100_wav: Path) -> None:
        s = AudioSample.from_wav(mono_44100_wav)
        with pytest.raises(ValueError, match="unknown normalize method"):
            normalize_volume(s, method="loud")


class TestPreprocessPipeline:
    def test_full_pipeline_stereo_48k_to_mono_16k(self, stereo_48000_wav: Path) -> None:
        s = AudioSample.from_wav(stereo_48000_wav)
        out = preprocess(s)
        assert out.is_mono
        assert out.sample_rate == TARGET_SAMPLE_RATE
        assert out.waveform.dtype == np.float32

    def test_pipeline_disables_trim_and_normalize(self, silence_padded_wav: Path) -> None:
        s = AudioSample.from_wav(silence_padded_wav)
        out = preprocess(s, trim_top_db=None, normalize=None)
        # with trim disabled, only silence on each side remains (post-resample,
        # still ~0.3s pads kept)
        assert out.duration > 1.0

    def test_pipeline_sets_peak_near_target(self, quiet_wav: Path) -> None:
        s = AudioSample.from_wav(quiet_wav)
        out = preprocess(s)
        assert np.max(np.abs(out.waveform)) == pytest.approx(0.99, abs=2e-2)

    def test_pipeline_silent_input_safe(self, silent_wav: Path) -> None:
        s = AudioSample.from_wav(silent_wav)
        out = preprocess(s)
        assert out.is_mono
        assert out.sample_rate == TARGET_SAMPLE_RATE
