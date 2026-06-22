"""Audio preprocessing pipeline: mono downmix, resample, silence trim, normalize.

The canonical output of :func:`preprocess` is a mono ``float32`` waveform at
``TARGET_SAMPLE_RATE`` (16 kHz) — the exact input contract of the downstream
Wav2Vec2-CTC phoneme model and the Parselmouth / MFCC feature extractors.
"""

from __future__ import annotations

from typing import Final

import librosa
import numpy as np

from ai_speech_shadowing.core.audio import TARGET_SAMPLE_RATE, AudioSample

DEFAULT_TRIM_TOP_DB: Final[int] = 30
DEFAULT_NORMALIZE: Final[str] = "peak"
RMS_TARGET: Final[float] = 0.06
"""RMS target for ``normalize_volume(method="rms")`` (~ -24 dBFS, good for speech)."""

PEAK_TARGET: Final[float] = 0.99
"""Peak target for ``normalize_volume(method="peak")`` (leaves ~0.1 dB headroom)."""

GAIN_SAFETY_CAP: Final[float] = 100.0
"""Hard cap on applied gain (40 dB) — generous enough for quiet recordings
while still preventing pathological blow-up of near-silent input."""

_EPS: Final[float] = 1e-8


def to_mono(sample: AudioSample) -> AudioSample:
    """Downmix to mono by averaging channels. No-op if already mono."""
    if sample.is_mono:
        return sample
    mono = np.mean(sample.waveform, axis=1, dtype=np.float32)
    return AudioSample(waveform=mono, sample_rate=sample.sample_rate)


def resample(sample: AudioSample, target_sr: int = TARGET_SAMPLE_RATE) -> AudioSample:
    """Resample to ``target_sr``. No-op if already at that rate.

    Multi-channel input is resampled per-channel and re-stacked. Phase is
    preserved by the default ``soxr_hq`` resampler.
    """
    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive; got {target_sr}")
    if sample.sample_rate == target_sr:
        return sample
    data = sample.waveform
    if data.ndim == 2:
        channels = [
            librosa.resample(data[:, c], orig_sr=sample.sample_rate, target_sr=target_sr)
            for c in range(data.shape[1])
        ]
        out = np.stack(channels, axis=1)
    else:
        out = librosa.resample(data, orig_sr=sample.sample_rate, target_sr=target_sr)
    return AudioSample(waveform=np.ascontiguousarray(out, dtype=np.float32), sample_rate=target_sr)


def trim_silence(sample: AudioSample, *, top_db: int = DEFAULT_TRIM_TOP_DB) -> AudioSample:
    """Strip leading/trailing (and interior) silence below ``top_db`` threshold.

    Requires mono input — call :func:`to_mono` first for multi-channel audio.
    If the entire clip is below threshold, the input is returned unchanged.
    """
    if not sample.is_mono:
        raise ValueError("trim_silence requires mono input; call to_mono() first")
    intervals = librosa.effects.split(sample.waveform, top_db=top_db)
    if intervals.size == 0:
        return sample
    pieces = [sample.waveform[a:b] for a, b in intervals]
    trimmed = np.concatenate(pieces).astype(np.float32)
    return AudioSample(waveform=trimmed, sample_rate=sample.sample_rate)


def normalize_volume(sample: AudioSample, *, method: str = DEFAULT_NORMALIZE) -> AudioSample:
    """Apply peak or RMS volume normalization.

    Args:
        method: ``"peak"`` scales so max abs(sample) == ``PEAK_TARGET``;
            ``"rms"`` scales so RMS == ``RMS_TARGET``.
    """
    data = sample.waveform
    if method == "peak":
        peak = float(np.max(np.abs(data)))
        if peak < _EPS:
            return sample
        gain = PEAK_TARGET / peak
    elif method == "rms":
        rms = float(np.sqrt(np.mean(data * data)))
        if rms < _EPS:
            return sample
        gain = RMS_TARGET / rms
    else:
        raise ValueError(f"unknown normalize method {method!r}; expected 'peak' or 'rms'")

    gain = min(gain, GAIN_SAFETY_CAP)
    return AudioSample(waveform=(data * gain).astype(np.float32), sample_rate=sample.sample_rate)


def preprocess(
    sample: AudioSample,
    *,
    target_sr: int = TARGET_SAMPLE_RATE,
    trim_top_db: int | None = DEFAULT_TRIM_TOP_DB,
    normalize: str | None = DEFAULT_NORMALIZE,
) -> AudioSample:
    """Run the full preprocessing chain.

    Order is fixed: downmix → resample → trim silence → normalize.
    Any step can be disabled by passing ``None`` (for ``trim_top_db`` /
    ``normalize``) — but mono + 16 kHz is always enforced.
    """
    out = to_mono(sample)
    out = resample(out, target_sr)
    if trim_top_db is not None:
        out = trim_silence(out, top_db=trim_top_db)
    if normalize is not None:
        out = normalize_volume(out, method=normalize)
    return out
