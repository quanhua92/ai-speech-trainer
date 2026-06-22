# Pitch & Prosody Analysis

> **Phase 3 deliverable.** Detects monotone delivery and compares intonation
> contours via fundamental-frequency (F0) extraction. Takes a canonical
> `AudioSample` (from [Phase 1](audio-preprocessing.md)) and produces pitch
> statistics, a reference-vs-user comparison, and a prosody sub-score.

## Overview

The prosody stage lives in `ai_speech_shadowing.core.prosody` and answers two
questions:

1. **Extraction** — what does this clip's pitch contour look like?
   (`extract_pitch` → `PitchStats`)
2. **Comparison** — how does the user's intonation compare to the native
   reference's? (`compare_pitch` → `ProsodyDiff`)

Unlike the phoneme stage, this is **deterministic DSP via
[`praat-parselmouth`](https://github.com/YannickJadoul/Parselmouth)** — no
model download, no GPU, fully unit-testable on synthetic tones.

## Extraction

`extract_pitch` runs Praat's pitch tracker on a (downmixed) mono signal and
summarises the voiced frames:

```python
from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.prosody import extract_pitch
from ai_speech_shadowing.core.preprocess import preprocess

sample = preprocess(AudioSample.from_wav("user.wav"))
stats = extract_pitch(sample)

stats.mean_hz       # 203.9
stats.median_hz     # 202.3
stats.min_hz        # 83.5
stats.max_hz        # 325.2
stats.range_hz      # 241.8  (= max - min)
stats.std_hz        # 44.4
stats.voiced_ratio  # 0.576  (fraction of frames with a detectable F0)
stats.is_voiced     # True
stats.f0_contour    # ndarray — full contour, 0.0 for unvoiced frames
stats.times         # ndarray — timestamp (s) of each frame
```

### Praat parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `pitch_floor` | 75 Hz | Lowest F0 the tracker will consider |
| `pitch_ceiling` | 500 Hz | Highest F0 the tracker will consider |

The 75–500 Hz window covers adult male through female speech (Praat's own
defaults). Tighten it (e.g. `pitch_floor=100`) for a known speaker to reduce
octave errors.

### Why not Praat's `get_mean()` etc.?

Recent `praat-parselmouth` builds dropped those accessor methods from the
`Pitch` object. We pull the contour straight from `selected_array['frequency']`
and compute the statistics with numpy — version-proof and a single source of
truth for the contour.

## Comparison

`compare_pitch(reference, hypothesis)` is the key producer of the prosody
sub-score:

```python
from ai_speech_shadowing.core.prosody import compare_pitch

ref = extract_pitch(reference_sample)
hyp = extract_pitch(user_sample)
d = compare_pitch(ref, hyp)

d.pitch_range_ratio   # hyp.range_hz / ref.range_hz  (undefined → 0.0)
d.monotone            # True if the user's range is below the threshold
d.score               # prosody sub-score in [0, 1]
d.grade               # "good" | "fair" | "needs_work"
```

### The pitch range ratio

```
pitch_range_ratio = hypothesis.range_hz / reference.range_hz
```

This is **the** key metric for Phase 3. It captures how much intonation the
user reproduces *relative to the native target*, so it's robust to a speaker's
natural register (a low-voiced user isn't penalised against a high-voiced
reference — only their *variation* is compared).

### Monotone detection

A user is flagged **monotone** when they are voiced *and* their pitch range
falls below `monotone_threshold` (default `0.5` → 50 %) of the reference's:

> *"Your pitch range is narrower than the reference. Try exaggerating the
> rising tone on question endings."*

Unvoiced audio is deliberately **not** called monotone — that's a different
problem (silence / no delivery) handled upstream.

### Sub-score

```
score = min(1.0, pitch_range_ratio)
```

Capped so an exaggerated range isn't over-rewarded. `0` when either side is
unvoiced. The grade thresholds are `≥0.8 good`, `≥0.5 fair`, else
`needs_work`. **Note:** this is a standalone sub-score; Phase 5 unifies it
with the phoneme (PER) and fluency (DTW) sub-scores into a weighted composite.

### CLI

```bash
# extract pitch statistics from a file (preprocesses automatically)
ai-speech-shadowing prosody user.wav

# custom Praat pitch bounds
ai-speech-shadowing prosody user.wav --pitch-floor 100 --pitch-ceiling 400

# skip preprocessing
ai-speech-shadowing prosody canonical.wav --no-preprocess
```

Output (e.g. the Kokoro `af_heart` reference):
```
mean 203.9 Hz | median 202.3 Hz
min 83.5 Hz | max 325.2 Hz | range 241.8 Hz
std 44.4 Hz | voiced 57.6%
```

## Design decisions

- **Pitch range over pitch height.** Absolute F0 depends on the speaker's
  anatomy; *range* is the register-independent signal of intonation quality, so
  the ratio (not, say, mean-difference) drives the score.
- **`voiced_ratio` is first-class.** A low voiced ratio signals a tracking
  problem or a noisy recording before any comparison runs.
- **No hard 16 kHz requirement.** Praat handles any sample rate; the contract
  is mono input (downmixed automatically). The canonical pipeline still feeds
  16 kHz mono for consistency with the other stages.
- **Deterministic & testable.** Pure tones have known F0, so the suite asserts
  e.g. a 200 Hz sine → `mean ≈ 200 Hz`, and a 100 + 300 Hz concatenation →
  `range ≈ 200 Hz` — no `--runslow` needed.

## Test coverage

`tests/test_prosody.py` (13 tests, all in the fast suite):

- `extract_pitch` tracks a 200 Hz fundamental within ±5 Hz; recovers the
  100 + 300 Hz concatenation's range within ±20 Hz; flags white noise as
  unvoiced; validates the `pitch_floor`/`pitch_ceiling` precondition; keeps
  `f0_contour` and `times` shape-aligned.
- `compare_pitch`: identical contours → ratio 1.0, not monotone, score 1.0;
  narrow tone vs wide reference → monotone; threshold is configurable; an
  exaggerated range caps the score at 1.0; unvoiced reference/hypothesis both
  score 0 without raising.
