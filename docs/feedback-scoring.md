# Feedback Engine & Scoring

> **Phase 5 deliverable.** Unifies the three pillars — phoneme (PER), prosody
> (pitch-range-ratio), fluency (DTW) — into a single `FeedbackReport` with a
> weighted composite score, colour-coded severity grades, targeted textual
> feedback, and JSON / terminal / Markdown renderers.

## Overview

`ai_speech_shadowing.core.feedback` provides two entry points:

| Function | What it does | Cost |
| --- | --- | --- |
| `build_report` | Aggregate three pillar diffs into a `FeedbackReport` | pure / fast |
| `evaluate` | Run the **full pipeline** (audio → report), loading the phoneme model | opt-in slow |

Plus three renderers: `to_json`, `to_terminal`, `to_markdown`.

## The composite score

Each pillar contributes a normalized score in `[0, 100]`:

| Pillar | Source metric | Sub-score |
| --- | --- | --- |
| **Pronunciation** | Phoneme Error Rate (PER) | `(1 − PER) × 100` |
| **Intonation** | pitch-range-ratio | prosody sub-score × 100 |
| **Fluency** | normalized DTW distance | fluency sub-score × 100 |

The composite is the weighted sum (default **40 / 30 / 30**):

```
composite = pronunciation·w0 + intonation·w1 + fluency·w2
```

```python
from ai_speech_shadowing.core.feedback import DEFAULT_WEIGHTS, build_report

report = build_report(phoneme_diff, prosody_diff, fluency_diff)
report.composite_score   # 0..100
report.composite_grade   # "good" | "fair" | "needs_work"

# custom weights (must sum to ~1.0)
report = build_report(..., weights=(0.5, 0.25, 0.25))
```

### Grade thresholds

| Composite | Grade | Marker |
| --- | --- | --- |
| `≥ 80` | `good` | 🟢 |
| `50 – 79` | `fair` | 🟡 |
| `< 50` | `needs_work` | 🔴 |

Each pillar is also graded individually with the same thresholds.

## Textual feedback

`build_report` generates deterministic, targeted suggestions off the weakest
pillars (and speaking-rate drift):

- a phoneme **substitution** → *"Phoneme /l/ was substituted with /ɹ/ — focus
  on tongue placement."* (pulls the first `sub` op from the phoneme diff)
- a **monotone** / narrow pitch range → *"Your pitch range is narrower than the
  reference. Try exaggerating rising tones on question endings."*
- weak **rhythm** (high DTW) → *"Your rhythm diverges from the reference; shadow
  the native pacing."*
- more **pauses** than the reference → *"You paused N× vs the reference's M× —
  aim for a steadier flow."*
- syllable-rate drift outside `[0.7, 1.3]×` → *"You're speaking slower/faster
  than the reference…"*

If nothing is weak, it emits a positive *"Great job — your delivery closely
matches the reference."*

## Full pipeline: `evaluate`

```python
from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.feedback import evaluate
from ai_speech_shadowing.core.preprocess import preprocess

ref = preprocess(AudioSample.from_wav("reference.wav"))
hyp = preprocess(AudioSample.from_wav("user.wav"))
report = evaluate(ref, hyp)  # loads the Wav2Vec2 phoneme model on first call
```

`evaluate` runs, in order: phoneme extraction + `diff_phonemes`, `extract_pitch`
+ `compare_pitch`, `compare_fluency`, then `build_report`. Pass
`phoneme_extractor=` to inject a pre-loaded model, and `weights=` to override
the composite weighting.

### Asymmetric reference sourcing

`evaluate` accepts an optional `reference_phonemes=` parameter. When provided
(typically read from the reference's cached `metadata.json["phonemes"]["tokens"]`
— captured from Kokoro's G2P at synthesis time, see
[`tts-reference.md`](tts-reference.md)), the reference audio is **not** passed
through the Wav2Vec2 model at all — the G2P tokens become the reference
sequence directly. This is the canonical-target path: text-derived, voice-
invariant, and free of recognizer noise.

When `reference_phonemes` is `None` (legacy callers, or a future uploaded clip
without transcript), `evaluate` falls back to decoding the reference audio
acoustically — the original behavior.

The chosen path is recorded on the report as `reference_phoneme_source`
(`"kokoro-g2p"` or `"wav2vec2-acoustic"`) and surfaced through the API. See
[`phoneme-extraction.md`](phoneme-extraction.md) for the full rationale and
the calibration caveat.

### CLI

```bash
# default: terminal report
ai-speech-shadowing evaluate reference.wav user.wav

# machine-readable
ai-speech-shadowing evaluate reference.wav user.wav --format json
ai-speech-shadowing evaluate reference.wav user.wav --format markdown

# custom composite weights (pron, into, flu)
ai-speech-shadowing evaluate reference.wav user.wav --weights 0.5,0.25,0.25
```

The first call downloads/loads the ~1.2 GB Wav2Vec2 model.

Sample terminal output (identical clip vs. itself):
```
AI Speech Shadowing — Report
────────────────────────────────────────────────────
Pronunciation (PER):   100  🟢 good
Intonation (Pitch):    100  🟢 good
Fluency (DTW):         100  🟢 good
────────────────────────────────────────────────────
Composite Score:       100/100  🟢 good
────────────────────────────────────────────────────
Feedback:
  • Great job — your delivery closely matches the reference.
```

## Renderers

- **`to_json(report)`** — a dict matching the Phase 8 `EvaluationResponse`
  schema (composite, per-pillar scores with key metrics, phoneme-diff op list,
  feedback array). `report_to_dict` returns the raw dict if you need to embed it.
- **`to_terminal(report)`** — the colour-coded human view above.
- **`to_markdown(report)`** — a Markdown table + feedback bullets (useful for
  docs / PR comments).

The JSON `phoneme_diff` entries follow the wire schema:
`{"type": "match"|"sub"|"del"|"ins", …}` (`phoneme` for match, `expected`/`actual`
for sub, `expected` for del, `actual` for ins).

## Design decisions

- **`build_report` is pure.** It takes already-computed diffs, so scoring,
  weighting, feedback generation, and rendering are all unit-testable with
  synthetic diffs — no model, no audio. Only `evaluate` is slow.
- **Pronunciation from accuracy (1 − PER).** PER is the natural phoneme metric;
  using `1 − PER` (clamped, since PER can exceed 1 on heavy insertions) maps it
  into the same `[0, 1]` scale the other pillars use.
- **Weights validated to sum to ~1.0.** A misconfigured weight tuple fails fast
  rather than silently producing scores outside `[0, 100]`.
- **Feedback is deterministic.** No LLM call — rules keyed off thresholds, so
  the same report always yields the same messages (reproducible tests & API).
- **Provisional DTW/phoneme score mappings carry forward.** Phase 5 inherits the
  Phase 2–4 sub-scores as-is; later calibration against real speech pairs can
  tune `DTW_SCORE_SCALE` (fluency) without changing this layer.

## Test coverage

`tests/test_feedback.py` (24 fast + 2 slow):

- **Fast (synthetic diffs):** perfect report → composite 100 / "good"; composite
  weighting math (40/30/30 default and custom); weights-validation error;
  PER→accuracy mapping; all six `grade_for` thresholds; per-pillar feedback
  messages (substitution, monotone, rhythm, pauses, rate drift); JSON
  round-trip + phoneme-op serialization; terminal & markdown structure;
  **reference phoneme source** (G2P path skips acoustic recognition on
  reference, acoustic fallback runs on both, default is acoustic, source
  propagates through `report_to_dict`).
- **Slow (full `evaluate`):** identical Kokoro clip → composite ≥ 90 / "good"
  and all renderers consume the real report; two different clips → composite
  `< 100` with non-empty feedback.
