"""Pitch & prosody analysis via praat-parselmouth.

Extracts the fundamental frequency (F0) contour from a canonical
:class:`AudioSample`, summarises it into descriptive statistics, and compares a
reference vs. user contour to detect monotone delivery and derive a prosody
sub-score (the key metric is the **pitch range ratio**).

Unlike the phoneme stage, this is deterministic DSP — no model download — so
the whole module is fast and fully unit-testable on synthetic tones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_speech_shadowing.core.audio import AudioSample

DEFAULT_PITCH_FLOOR: float = 75.0
"""Praat pitch floor (Hz). 75 Hz is Praat's default for human speech."""
DEFAULT_PITCH_CEILING: float = 500.0
"""Praat pitch ceiling (Hz). 500 Hz covers adult male through female speech."""
DEFAULT_MONOTONE_THRESHOLD: float = 0.5
"""A user whose pitch range is below this fraction of the reference's is
considered monotone."""


# --------------------------------------------------------------------------- #
# Pitch statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PitchStats:
    """Summary of the F0 contour of one clip.

    Unvoiced frames (silence / noise) are excluded from the descriptive
    statistics; they appear as ``0.0`` in ``f0_contour`` and lower the
    ``voiced_ratio``.
    """

    f0_contour: np.ndarray  # full contour, 0.0 for unvoiced frames
    times: np.ndarray  # timestamp (s) of each frame
    mean_hz: float
    median_hz: float
    min_hz: float
    max_hz: float
    range_hz: float
    std_hz: float
    voiced_ratio: float
    pitch_floor: float
    pitch_ceiling: float

    @property
    def is_voiced(self) -> bool:
        return self.voiced_ratio > 0.0


def extract_pitch(
    sample: AudioSample,
    *,
    pitch_floor: float = DEFAULT_PITCH_FLOOR,
    pitch_ceiling: float = DEFAULT_PITCH_CEILING,
) -> PitchStats:
    """Extract the F0 contour and descriptive pitch statistics.

    Multi-channel input is downmixed to mono on the fly. Any sample rate is
    accepted (parselmouth handles it), though 16 kHz mono (the canonical form)
    is the expected input from the pipeline.
    """
    import parselmouth

    if pitch_floor <= 0 or pitch_ceiling <= pitch_floor:
        raise ValueError(
            f"need 0 < pitch_floor < pitch_ceiling; got floor={pitch_floor}, "
            f"ceiling={pitch_ceiling}"
        )
    wav = sample.waveform
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    sound = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sample.sample_rate)
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)

    f0 = pitch.selected_array["frequency"].astype(np.float64)
    times = np.asarray(pitch.xs(), dtype=np.float64)
    voiced = f0[f0 > 0]
    voiced_ratio = float(voiced.size / f0.size) if f0.size else 0.0

    if voiced.size == 0:
        mean = median = vmin = vmax = vrng = vstd = 0.0
    else:
        mean = float(voiced.mean())
        median = float(np.median(voiced))
        vmin = float(voiced.min())
        vmax = float(voiced.max())
        vrng = vmax - vmin
        vstd = float(voiced.std())

    return PitchStats(
        f0_contour=f0,
        times=times,
        mean_hz=mean,
        median_hz=median,
        min_hz=vmin,
        max_hz=vmax,
        range_hz=vrng,
        std_hz=vstd,
        voiced_ratio=voiced_ratio,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )


# --------------------------------------------------------------------------- #
# Reference vs. user comparison
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ProsodyDiff:
    """Result of comparing a reference pitch contour to the user's."""

    reference: PitchStats
    hypothesis: PitchStats
    pitch_range_ratio: float
    monotone: bool
    monotone_threshold: float
    score: float  # prosody sub-score in [0, 1]; Phase 5 unifies all sub-scores

    @property
    def grade(self) -> str:
        if self.score >= 0.8:
            return "good"
        if self.score >= 0.5:
            return "fair"
        return "needs_work"


def compare_pitch(
    reference: PitchStats,
    hypothesis: PitchStats,
    *,
    monotone_threshold: float = DEFAULT_MONOTONE_THRESHOLD,
) -> ProsodyDiff:
    """Compare two pitch contours → range ratio, monotone flag, sub-score.

    - **pitch_range_ratio** = hypothesis range / reference range (undefined → 0).
    - **monotone** = True when the user *is* voiced but their range falls below
      ``monotone_threshold`` of the reference's.
    - **score** = ``min(1, ratio)`` — capped so an exaggerated range isn't
      over-rewarded. 0 when either side is unvoiced.
    """
    ref_ok = reference.is_voiced and reference.range_hz > 0
    hyp_ok = hypothesis.is_voiced

    if ref_ok and hyp_ok:
        ratio = hypothesis.range_hz / reference.range_hz
        monotone = ratio < monotone_threshold
        score = min(1.0, ratio)
    else:
        ratio = 0.0
        monotone = False  # can't call unvoiced audio "monotone delivery"
        score = 0.0

    return ProsodyDiff(
        reference=reference,
        hypothesis=hypothesis,
        pitch_range_ratio=ratio,
        monotone=monotone,
        monotone_threshold=monotone_threshold,
        score=score,
    )
