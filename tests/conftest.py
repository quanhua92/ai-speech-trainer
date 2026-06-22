"""Shared pytest fixtures: synthetic audio fixtures generated on the fly.

We synthesize WAVs with numpy + soundfile so the test suite stays deterministic
and free of committed binary blobs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _sine(
    duration: float = 1.0,
    sr: int = 16000,
    freq: float = 220.0,
    amplitude: float = 0.8,
    channels: int = 1,
) -> np.ndarray:
    """Return a float32 sine wave of shape (n,) or (n, channels)."""
    n = round(duration * sr)
    t = np.arange(n, dtype=np.float32) / sr
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 1:
        return tone
    # slightly different phase per channel so downmix isn't trivially identical
    return np.stack([tone, np.roll(tone, 7)], axis=1).astype(np.float32)


def _write(path: Path, data: np.ndarray, sr: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="FLOAT")
    return path


@pytest.fixture
def mono_44100_wav(tmp_path: Path) -> Path:
    """A 1.0s mono 220Hz sine at 44.1kHz."""
    return _write(tmp_path / "mono_44100.wav", _sine(1.0, 44100, channels=1), 44100)


@pytest.fixture
def stereo_48000_wav(tmp_path: Path) -> Path:
    """A 1.0s stereo 220Hz sine at 48kHz."""
    return _write(tmp_path / "stereo_48000.wav", _sine(1.0, 48000, channels=2), 48000)


@pytest.fixture
def silence_padded_wav(tmp_path: Path) -> Path:
    """A 220Hz tone at 16kHz padded with 0.3s of silence at both ends."""
    sr = 16000
    pad = int(0.3 * sr)
    tone = _sine(0.5, sr, channels=1)
    silence = np.zeros(pad, dtype=np.float32)
    padded = np.concatenate([silence, tone, silence])
    return _write(tmp_path / "padded.wav", padded, sr)


@pytest.fixture
def quiet_wav(tmp_path: Path) -> Path:
    """A mono tone at very low amplitude (0.01) — for normalization tests."""
    return _write(tmp_path / "quiet.wav", _sine(1.0, 16000, amplitude=0.01), 16000)


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A 0.5s mono pure-silence clip."""
    return _write(tmp_path / "silent.wav", np.zeros(8000, dtype=np.float32), 16000)
