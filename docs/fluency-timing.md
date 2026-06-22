# Fluency & Timing (DTW)

> **Phase 4 deliverable.** Evaluates pacing, rhythm, and temporal alignment.
> Aligns the user's MFCC feature matrix against the reference's with Dynamic
> Time Warping and produces a normalized distance, pause analysis, a syllable
> rate estimate, and a fluency sub-score.

## Overview

The fluency stage lives in `ai_speech_shadowing.core.fluency` and answers:
how well does the user's *rhythm* match the native reference — independent of
how fast they spoke? Three building blocks feed the comparison:

| Component | Function | Output |
| --- | --- | --- |
| MFCC features | `extract_mfcc` | `MfccFeatures` (L2-normalized per-frame matrix) |
| DTW alignment | `dtw_distance` | `DtwResult` (accumulated + normalized distance) |
| Pause analysis | `detect_pauses` | `PauseInfo` (count, total, durations) |
| Syllable rate | `estimate_syllable_rate` | `float` (syllables/sec, heuristic) |

`compare_fluency(reference_sample, hypothesis_sample)` runs all four and
returns a `FluencyDiff`. Like prosody, this is **deterministic DSP** — no model
download — so the whole suite runs fast on synthetic audio.

## Why DTW (not frame-by-frame)?

People speak at different speeds. A naive frame-by-frame comparison would
penalise someone who speaks 10 % slower even with perfect pronunciation.
Dynamic Time Warping stretches/bends the time axis to find the optimal
alignment, so the resulting cost reflects **rhythm and acoustic similarity**
rather than raw duration. (See the project README's "Why DTW instead of simple
time-alignment?" note.)

## MFCC features

```python
from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.fluency import extract_mfcc
from ai_speech_shadowing.core.preprocess import preprocess

feats = extract_mfcc(preprocess(AudioSample.from_wav("user.wav")))

feats.matrix       # ndarray, shape (n_frames, 13) — one row per ~32ms frame
feats.times        # ndarray — timestamp (s) of each frame
feats.num_frames
```

Each frame is a 13-coefficient MFCC vector **L2-normalised to unit length**, so
the DTW Euclidean distance is bounded and comparable across recordings
(independent of absolute loudness / MFCC scale). Defaults: `n_mfcc=13`,
`n_fft=2048`, `hop_length=512`.

## DTW

```python
from ai_speech_shadowing.core.fluency import dtw_distance

dtw = dtw_distance(ref_feats, hyp_feats)
dtw.distance              # accumulated cost
dtw.path_length           # number of aligned steps
dtw.normalized_distance   # distance / path_length — average per-frame cost
```

`fastdtw` with Euclidean distance and `radius=1` (configurable). The
**normalized distance** is the key metric — lower means a closer match. It is
~0 for identical input and grows with acoustic divergence.

## Pauses

```python
from ai_speech_shadowing.core.fluency import detect_pauses

pauses = detect_pauses(sample, min_pause_s=0.25)
pauses.count          # number of interior pauses ≥ min_pause_s
pauses.total_seconds  # combined duration
pauses.durations      # tuple of each pause length (s)
```

Uses `librosa.effects.split` to find non-silent segments; the gaps *between*
consecutive segments are the interior pauses. Leading/trailing silence is
already stripped by `preprocess`, so this targets hesitations *within* the
utterance. (Note: `split`'s dB threshold pulls the detected segment edges
slightly into the silence, so measured gaps are a touch shorter than the raw
silence — expected DSP behaviour.)

## Syllable rate

```python
from ai_speech_shadowing.core.fluency import estimate_syllable_rate

estimate_syllable_rate(sample)  # ~ syllables / second
```

A heuristic: count prominence-thresholded peaks in the smoothed RMS energy
contour. It's a stand-in for true syllable detection (which needs linguistic
cues) and works best on preprocessed speech. The comparison exposes a
`syllable_rate_ratio = hypothesis / reference` to flag speaking-rate drift.

## Comparison

```python
from ai_speech_shadowing.core.fluency import compare_fluency

diff = compare_fluency(ref_sample, hyp_sample)

diff.dtw.normalized_distance   # the key metric
diff.score                     # provisional sub-score in [0, 1]
diff.grade                     # "good" | "fair" | "needs_work"
diff.reference_pauses          # PauseInfo
diff.hypothesis_pauses         # PauseInfo
diff.syllable_rate_reference   # float
diff.syllable_rate_hypothesis  # float
diff.syllable_rate_ratio       # hyp / ref (0 if ref is 0)
```

### The provisional sub-score

```
score = max(0.0, 1.0 - normalized_distance / DTW_SCORE_SCALE)
```

with `DTW_SCORE_SCALE = 0.5`. Identical input → 1.0; the score decays linearly
to 0 as the normalized distance grows. **This mapping is provisional** — Phase
5 calibrates it against real speech pairs and folds it into the weighted
composite with the phoneme (PER) and prosody (pitch-range-ratio) sub-scores.

### CLI

```bash
# compare a user recording against a reference (both preprocessed first)
ai-speech-shadowing fluency reference.wav user.wav

# custom minimum pause length
ai-speech-shadowing fluency reference.wav user.wav --min-pause 0.4
```

Output (e.g. identical file vs. itself):
```
DTW distance 0.00 | normalized 0.000 (path 84)
score 100/100 (good)
pauses: ref 0 (0.00s) | hyp 0 (0.00s)
syllable rate: ref 1.51/s | hyp 1.51/s
```

## Design decisions

- **L2-normalised MFCC frames.** Bounds the DTW Euclidean distance and makes it
  independent of recording level — a quiet and a loud take of the same speech
  align with near-zero distance.
- **`normalized_distance` over raw.** Accumulated cost scales with clip length;
  dividing by path length yields a length-independent per-frame cost suitable
  for scoring.
- **Pauses are interior only.** Leading/trailing silence is the preprocess
  stage's job; `detect_pauses` targets the hesitations that actually hurt
  fluency.
- **Syllable rate is explicitly heuristic.** Documented as an approximation —
  robust enough to flag "speaking too fast/slow" via the ratio without
  pretending to be a true syllable counter.
- **Provisional score.** Phase 4 ships a simple linear mapping; Phase 5 owns the
  calibrated, weighted composite.

## Test coverage

`tests/test_fluency.py` (13 tests, all in the fast suite):

- `extract_mfcc`: shape `(n_frames, 13)`, rows are unit-norm, times aligned.
- `dtw_distance`: identical features → 0; different-frequency tones → positive.
- `detect_pauses`: finds the interior gap in a tone–silence–tone clip; no pause
  in a continuous tone; threshold filters short gaps.
- `estimate_syllable_rate`: a 5-burst pulse train yields a positive, plausible
  rate; flat noise envelope → low.
- `compare_fluency`: identical input → score 1.0 / grade "good" / ratio 1.0;
  different signals score strictly lower; pause info flows through.
