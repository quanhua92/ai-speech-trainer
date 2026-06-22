"""Fluency & timing analysis via MFCC + Dynamic Time Warping.

Evaluates pacing and rhythm by:

1. Extracting a per-frame MFCC feature matrix from a canonical AudioSample.
2. Aligning reference vs. user matrices with Dynamic Time Warping (``fastdtw``,
   Euclidean) to isolate rhythm from simple speed differences.
3. Detecting abnormal interior pauses and estimating a syllable rate.

The key metric is the **normalized DTW distance** — lower means a closer rhythm
match. Like prosody, this is deterministic DSP, so the suite runs fast on
synthetic audio with no model download.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_speech_shadowing.core.audio import AudioSample

DEFAULT_N_MFCC: int = 13
DEFAULT_N_FFT: int = 2048
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_DTW_RADIUS: int = 1
DEFAULT_TRIM_TOP_DB: int = 30
DEFAULT_MIN_PAUSE_S: float = 0.25
"""A silence gap strictly inside the clip of at least this length (seconds) is
flagged as an abnormal pause."""
DTW_SCORE_SCALE: float = 0.5
"""Per-frame DTW distance at which the provisional sub-score hits 0. Phase 5
calibrates this against real speech pairs."""


def _mono(sample: AudioSample) -> np.ndarray:
    wav = sample.waveform
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav


# --------------------------------------------------------------------------- #
# MFCC features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MfccFeatures:
    """Per-frame MFCC feature matrix (L2-normalized rows)."""

    matrix: np.ndarray  # shape (n_frames, n_mfcc)
    sample_rate: int
    hop_length: int
    times: np.ndarray  # timestamp (s) of each frame

    @property
    def num_frames(self) -> int:
        return int(self.matrix.shape[0])


def extract_mfcc(
    sample: AudioSample,
    *,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> MfccFeatures:
    """Extract an L2-normalized MFCC matrix.

    Each frame's MFCC vector is scaled to unit L2 norm so the DTW Euclidean
    distance is bounded and comparable across recordings (independent of
    absolute loudness / MFCC scale).
    """
    import librosa

    wav = _mono(sample).astype(np.float32)
    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=sample.sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    matrix = mfcc.T.astype(np.float64)  # (n_frames, n_mfcc)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms
    times = librosa.frames_to_time(
        np.arange(matrix.shape[0]),
        sr=sample.sample_rate,
        hop_length=hop_length,
    )
    return MfccFeatures(
        matrix=matrix,
        sample_rate=sample.sample_rate,
        hop_length=hop_length,
        times=times,
    )


# --------------------------------------------------------------------------- #
# Dynamic Time Warping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DtwResult:
    distance: float  # accumulated DTW cost
    path_length: int
    normalized_distance: float  # distance / path_length — average per-frame cost


def dtw_distance(
    reference: MfccFeatures,
    hypothesis: MfccFeatures,
    *,
    radius: int = DEFAULT_DTW_RADIUS,
) -> DtwResult:
    """Align two MFCC matrices with fastdtw (Euclidean) and return the cost.

    DTW warps the time axis to find the optimal alignment, isolating rhythm
    quality from speaking-rate differences.
    """
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    distance, path = fastdtw(reference.matrix, hypothesis.matrix, radius=radius, dist=euclidean)
    path_length = max(len(path), 1)
    return DtwResult(
        distance=float(distance),
        path_length=len(path),
        normalized_distance=float(distance) / path_length,
    )


# --------------------------------------------------------------------------- #
# Pauses & syllable rate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PauseInfo:
    """Summary of abnormal interior pauses (gaps between speech segments)."""

    count: int
    total_seconds: float
    durations: tuple[float, ...]


def detect_pauses(
    sample: AudioSample,
    *,
    top_db: int = DEFAULT_TRIM_TOP_DB,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
) -> PauseInfo:
    """Find silence gaps strictly inside the clip (between speech segments)."""
    import librosa

    wav = _mono(sample)
    intervals = librosa.effects.split(wav, top_db=top_db)
    durations: list[float] = []
    for k in range(1, len(intervals)):
        gap = (intervals[k][0] - intervals[k - 1][1]) / sample.sample_rate
        if gap >= min_pause_s:
            durations.append(float(gap))
    return PauseInfo(
        count=len(durations),
        total_seconds=float(sum(durations)),
        durations=tuple(durations),
    )


def estimate_syllable_rate(
    sample: AudioSample,
    *,
    frame_length: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> float:
    """Approximate syllables/second by counting peaks in the RMS contour.

    This is a heuristic — true syllable detection needs linguistic cues. It
    works best on preprocessed (trimmed, normalized) speech. Returns 0.0 for
    empty or silent input.
    """
    import librosa
    from scipy.signal import find_peaks

    wav = _mono(sample).astype(np.float32)
    if wav.size == 0:
        return 0.0
    rms = librosa.feature.rms(y=wav, frame_length=frame_length, hop_length=hop_length)[0]
    rms = np.convolve(rms, np.ones(3) / 3.0, mode="same")  # light smoothing
    span = float(rms.max() - rms.min())
    if span < 1e-8:
        return 0.0
    peaks, _ = find_peaks(rms, prominence=0.15 * span)
    duration = sample.duration
    return float(len(peaks)) / duration if duration > 0 else 0.0


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FluencyDiff:
    dtw: DtwResult
    score: float  # provisional fluency sub-score in [0, 1]
    reference_pauses: PauseInfo
    hypothesis_pauses: PauseInfo
    syllable_rate_reference: float
    syllable_rate_hypothesis: float
    syllable_rate_ratio: float

    @property
    def grade(self) -> str:
        if self.score >= 0.8:
            return "good"
        if self.score >= 0.5:
            return "fair"
        return "needs_work"


def compare_fluency(
    reference_sample: AudioSample,
    hypothesis_sample: AudioSample,
    *,
    n_mfcc: int = DEFAULT_N_MFCC,
    hop_length: int = DEFAULT_HOP_LENGTH,
    radius: int = DEFAULT_DTW_RADIUS,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
) -> FluencyDiff:
    """Full fluency comparison: MFCC → DTW + pauses + syllable rate."""
    ref_mfcc = extract_mfcc(reference_sample, n_mfcc=n_mfcc, hop_length=hop_length)
    hyp_mfcc = extract_mfcc(hypothesis_sample, n_mfcc=n_mfcc, hop_length=hop_length)
    dtw = dtw_distance(ref_mfcc, hyp_mfcc, radius=radius)

    score = max(0.0, 1.0 - dtw.normalized_distance / DTW_SCORE_SCALE)

    ref_rate = estimate_syllable_rate(reference_sample, hop_length=hop_length)
    hyp_rate = estimate_syllable_rate(hypothesis_sample, hop_length=hop_length)
    rate_ratio = hyp_rate / ref_rate if ref_rate > 0 else 0.0

    return FluencyDiff(
        dtw=dtw,
        score=score,
        reference_pauses=detect_pauses(reference_sample, min_pause_s=min_pause_s),
        hypothesis_pauses=detect_pauses(hypothesis_sample, min_pause_s=min_pause_s),
        syllable_rate_reference=ref_rate,
        syllable_rate_hypothesis=hyp_rate,
        syllable_rate_ratio=rate_ratio,
    )
