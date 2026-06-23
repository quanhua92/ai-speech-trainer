"""Tests for the AudioSample dataclass and WAV I/O."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ai_speech_shadowing.core.audio import AudioLoadError, AudioSample


def _pcm_wav(sample_rate: int, n_samples: int) -> bytes:
    """Build a minimal valid 16-bit PCM mono WAV with an arbitrary sample rate."""
    data = b"\x00\x00" * n_samples
    return (
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(data),
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            len(data),
        )
        + data
    )


class TestAudioSampleConstruction:
    def test_mono_construction_and_defaults(self) -> None:
        wav = np.zeros(16000, dtype=np.float32)
        s = AudioSample(waveform=wav)
        assert s.sample_rate == 16000
        assert s.is_mono
        assert s.channels == 1
        assert s.num_samples == 16000
        assert s.duration == pytest.approx(1.0)

    def test_stereo_shape(self) -> None:
        wav = np.zeros((16000, 2), dtype=np.float32)
        s = AudioSample(waveform=wav, sample_rate=16000)
        assert not s.is_mono
        assert s.channels == 2

    def test_float32_coercion(self) -> None:
        wav = np.ones(8, dtype=np.float64)
        s = AudioSample(waveform=wav)
        assert s.waveform.dtype == np.float32

    def test_rejects_non_numpy(self) -> None:
        with pytest.raises(TypeError):
            AudioSample(waveform=[0.0, 1.0, 2.0])  # type: ignore[arg-type]

    def test_rejects_3d(self) -> None:
        with pytest.raises(ValueError, match="must be 1D"):
            AudioSample(waveform=np.zeros((4, 2, 2), dtype=np.float32))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            AudioSample(waveform=np.array([], dtype=np.float32))

    def test_rejects_nonpositive_rate(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            AudioSample(waveform=np.zeros(8, dtype=np.float32), sample_rate=0)


class TestAudioPlausibility:
    """Regression tests for the sample_rate=1 memory-amplification DoS.

    A WAV declaring a tiny sample_rate made librosa.resample allocate
    len*16000/sr samples (~4 GB from a 122 KB upload). The rate range and
    duration cap reject these before the DSP pipeline runs.
    """

    @pytest.mark.parametrize("sr", [1, 500, 999, 200_000, 1_000_000])
    def test_rejects_implausible_rate(self, sr: int) -> None:
        with pytest.raises(ValueError, match="plausible range"):
            AudioSample(waveform=np.zeros(8, dtype=np.float32), sample_rate=sr)

    def test_rejects_too_long(self) -> None:
        # one sample over the 120s cap at 16 kHz
        n = 16_000 * 120 + 1
        with pytest.raises(ValueError, match="too long"):
            AudioSample(waveform=np.zeros(n, dtype=np.float32), sample_rate=16_000)

    @pytest.mark.parametrize("sr", [8_000, 16_000, 44_100, 48_000, 96_000])
    def test_accepts_plausible_rates(self, sr: int) -> None:
        s = AudioSample(waveform=np.zeros(sr, dtype=np.float32), sample_rate=sr)
        assert s.sample_rate == sr

    def test_from_bytes_rejects_amplification_wav(self) -> None:
        # the original attack: 122 KB WAV declaring sample_rate=1, 64000 frames
        with pytest.raises(AudioLoadError, match="plausible range"):
            AudioSample.from_bytes(_pcm_wav(1, 64_000))

    def test_frozen(self) -> None:
        s = AudioSample(waveform=np.zeros(8, dtype=np.float32))
        with pytest.raises(AttributeError):
            s.sample_rate = 8000  # type: ignore[misc]


class TestWavIO:
    def test_from_wav_mono(self, mono_44100_wav: Path) -> None:
        s = AudioSample.from_wav(mono_44100_wav)
        assert s.is_mono
        assert s.sample_rate == 44100
        assert s.duration == pytest.approx(1.0, abs=1e-3)
        assert s.waveform.dtype == np.float32

    def test_from_wav_stereo(self, stereo_48000_wav: Path) -> None:
        s = AudioSample.from_wav(stereo_48000_wav)
        assert s.channels == 2
        assert s.sample_rate == 48000

    def test_from_wav_missing(self, tmp_path: Path) -> None:
        with pytest.raises(AudioLoadError, match="not found"):
            AudioSample.from_wav(tmp_path / "nope.wav")

    def test_from_wav_corrupt(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a wav")
        with pytest.raises(AudioLoadError):
            AudioSample.from_wav(bad)

    def test_round_trip(self, tmp_path: Path, silence_padded_wav: Path) -> None:
        s = AudioSample.from_wav(silence_padded_wav)
        out = tmp_path / "out.wav"
        s.to_wav(out)
        reloaded = AudioSample.from_wav(out)
        assert reloaded.sample_rate == s.sample_rate
        assert reloaded.num_samples == s.num_samples
        assert np.allclose(reloaded.waveform, s.waveform, atol=1e-5)

    def test_to_wav_creates_parents(self, tmp_path: Path, mono_44100_wav: Path) -> None:
        s = AudioSample.from_wav(mono_44100_wav)
        out = tmp_path / "nested" / "dir" / "out.wav"
        s.to_wav(out)
        data, sr = sf.read(str(out))
        assert sr == 44100
        assert len(data) == s.num_samples
