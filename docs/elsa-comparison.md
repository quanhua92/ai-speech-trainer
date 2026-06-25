# Comparison with ELSA Speak

> How `ai-speech-shadowing`'s scoring differs from [ELSA Speak](https://elsaspeak.com/),
> a commercial cloud-based pronunciation coach. This is an independent, technical
> comparison based on ELSA's public product behaviour and this project's source.

## TL;DR

The two tools answer **different questions**:

| | ELSA Speak | ai-speech-shadowing |
| --- | --- | --- |
| Question it answers | *"Did you say this like a native speaker?"* | *"Did you sound like **this** speaker?"* |
| Scoring paradigm | **Norm-referenced** (against a learned native norm baked into the model) | **Criterion-referenced** (against a specific reference audio clip) |
| Inputs | User audio + target text | User audio + **reference audio** + optional reference text |
| Model | Proprietary, closed-source, cloud-only | Open-source, local, offline |
| Dimensions scored | Pronunciation, intonation, fluency, **word stress**, listening, grammar/vocab | Pronunciation, intonation, fluency |
| Licence & cost | **Commercial, VC-funded startup** — paid subscription | **Free, open-source (MIT)** — zero cost |

This project is **not** a drop-in replacement for ELSA. It is a narrower, transparent,
local-first engine built around the **shadowing technique** — you imitate a specific
reference clip, and the engine measures how closely you matched *that* clip.

---

## The fundamental difference: shadowing vs. correctness

ELSA's scoring is **accuracy-based**. Its model has already internalised a "native
norm," so it evaluates your utterance against that abstract standard. It does not need
a reference *audio* clip — it only needs the target *text* of the exercise.

`ai-speech-shadowing`'s scoring is **imitation-based**. It always compares your audio
to a **specific reference clip** (a Kokoro/Qwen TTS render). Every sub-score is
implicitly relative to that reference:

- Fluency is DTW spectral distance *to the clip* (`core/fluency.py:112-132`).
- Intonation is pitch-range *ratio* against the clip (`core/prosody.py:136-169`).
- Pronunciation is the only pillar that is also absolute (PER against the IPA),
  but even there, the reference's phonemes are the comparison target.

This is why two utterances that are both "correct" can score very differently here
depending on which reference clip they are paired with — a feature, not a bug, of
shadowing practice.

---

## Scoring dimensions

| Dimension | ELSA Speak | ai-speech-shadowing | Weight here |
| --- | :---: | :---: | :---: |
| Pronunciation | ✅ | ✅ | **40%** |
| Intonation / prosody | ✅ | ✅ | **30%** |
| Fluency | ✅ | ✅ | **30%** |
| Word stress | ✅ (dedicated score) | ❌ | — |
| Listening | ✅ | ❌ | — |
| Grammar / vocabulary | ✅ (some exercises) | ❌ | — |

ELSA scores **more dimensions**. This project intentionally keeps three pillars,
focused on *how* you speak rather than *what* you say or understand
(`docs/README.md:61-92`).

---

## How each pillar differs

### Pronunciation

| | ELSA Speak | ai-speech-shadowing |
| --- | --- | --- |
| Model | Proprietary DL, trained on millions of **non-native** samples | `slplab/wav2vec2-large-robust-L2-english-phoneme-recognition` — trained on **Korean L2 English** learner speech (default); multilingual `espeak` model also available (`core/phoneme.py`) |
| L1 awareness | **Yes** — weights errors by speaker's native language | **Partial** — the default model was trained on L2 learner errors (Korean-accented English) and tags mispronunciations with an `_err` suffix; the diff itself is language-agnostic edit distance |
| Metric | Per-sound confidence scores | Phoneme Error Rate = `(sub+del+ins)/len(ref)` (`core/phoneme.py`) |
| Score | 0–100 per sound and per word | `(1 − PER) × 100`, clamped to ≥ 0 |

**Implication:** ELSA's pronunciation score adapts to typical L1-interference
patterns across *many* native languages (e.g. a Vietnamese speaker's `/θ/`
errors are scored with that population's distribution in mind). This project's
default model has seen **one** L2 population (Korean learners) during training,
so it recognizes the accent patterns of that group more forgivingly than a
native-trained model would — but it does not re-weight errors per speaker L1.
The diff that produces the PER applies a uniform cost to every phoneme edit.

> **Measured with:** the slplab L2-English Wav2Vec2-CTC model (ARPAbet output
> mapped 1:1 to espeak IPA, segments rejoined to espeak units) + `difflib`
> sequence alignment. The reference side uses kokoro's G2P output, not the
> recognizer. See [`phoneme-extraction.md`](phoneme-extraction.md) for the full
> pipeline.

> **Calibration note (pre- vs. post-cutover).** Earlier PER figures in this
> document were recorded when *both* reference and hypothesis were decoded
> acoustically by the Wav2Vec2 model. The reference side is now sourced from
> Kokoro's G2P output when text is known (captured at synthesis time), which
> removes the recognizer's own noise from the baseline. PERs measured after
> the cutover are not directly comparable to the pre-cutover numbers — they
> are typically slightly higher for the same audio because the baseline is no
> longer forgiving the recognizer's own mistakes.

### Intonation / prosody

| | ELSA Speak | ai-speech-shadowing |
| --- | --- | --- |
| Signals | Pitch contour + energy + rhythm, modelled against native contours | **Pitch-range ratio only** (`core/prosody.py:136-169`) |
| Score | Learned similarity to native intonation | `min(1, user_range / ref_range) × 100` |
| Over-expression | Penalised by the model | **Capped, not penalised** — `min(1, ratio)` |

**Implication:** This project's intonation score is **coarser**. It rewards matching
the *width* of the reference's F0 excursion, not the *shape* of the contour. A
rising-then-falling curve can score the same as falling-then-rising if the ranges
match.

> **Measured with:** `praat-parselmouth` (Praat) for F0 pitch extraction — no model
> download, pure DSP. See [`pitch-prosody.md`](pitch-prosody.md) for details.

### Fluency

| | ELSA Speak | ai-speech-shadowing |
| --- | --- | --- |
| Approach | Models rate, pausing, hesitation, rhythm against native baselines | **DTW over L2-normalised MFCC frames** (`core/fluency.py:112-132`) |
| What it measures | Pure fluency, separated from pronunciation | **Acoustic closeness over time** to the reference clip |
| Secondary signals | — | Pause count, syllable-rate ratio (`core/fluency.py:147-195`) |
| Score | 0–100 | `1 − DTW_normalised / DTW_SCORE_SCALE` (`core/fluency.py:31-35`) |

**Implication:** This project's fluency pillar **conflates fluency with timbre and
pronunciation** — it measures how spectrally similar your clip is to the reference
over time, not how "fluent" you are in the abstract. ELSA separates these more
cleanly because it has no reference clip to be similar *to*.

> **Measured with:** `librosa` (13-frame MFCC extraction) + `fastdtw` (Euclidean
> dynamic time warping). See [`fluency-timing.md`](fluency-timing.md) for details.

---

## Granularity

| Level | ELSA Speak | ai-speech-shadowing |
| --- | --- | --- |
| **Phoneme** | ✅ Numeric per-sound score + error type | ⚠️ Categorical ops only: `match` / `sub` / `del` / `ins` (`core/phoneme.py:40-49`) — **no numeric per-phoneme score** |
| **Word** | ✅ Numeric 0–100 + flagged problem sounds | ⚠️ Categorical status only, projected via misaki G2P (`core/wordalign.py:170-225`) — **no numeric per-word score** |
| **Sentence** | ✅ Numeric 0–100 | ❌ Sentences used only as alignment boundaries |
| **Skill tracking** | ✅ Adaptive mastery curves over time | ⚠️ Trend (improving/declining), weakest-5 phonemes, daily averages (`core/history.py:202-229`) |

If you need **numeric per-word or per-phoneme scores**, ELSA provides them and this
project currently does not. This project's phoneme diff is a sequence of categorical
operations, useful for *showing* what was substituted/inserted/deleted but not for
ranking individual sounds by quality.

---

## Architectural differences

| | ELSA Speak | ai-speech-shadowing |
| --- | --- | --- |
| Source | Closed-source | Open-source (MIT) |
| Funding | **Commercial startup**, venture-backed, subscription-funded | **Free, community open-source** project (unfunded) |
| Runtime | Cloud only | **Local / offline** (`docs/README.md:725-731`) |
| Privacy | Audio sent to servers | Audio never leaves the machine |
| Cost | Per-user subscription | Zero marginal cost per evaluation |
| Latency | Network round-trip | Sub-second on CPU after model load |
| Feedback generation | Adaptive, personalised | **Deterministic, rule-based** (`core/feedback.py:190-267`) |
| Tunability | Fixed weights & thresholds | Exposed via `--weights` and `DTW_SCORE_SCALE` (`docs/cli.md`) |

---

## When to use which

**Use ELSA Speak if you want:**
- The broadest set of scored dimensions (stress, listening, grammar).
- **Multi-L1** pronunciation scoring calibrated to *your* native language
  (this project's model has seen only Korean-accented English).
- Numeric per-word and per-phoneme scores.
- A polished, adaptive consumer product with a curriculum.

**Use ai-speech-shadowing if you want:**
- A **local, private** evaluation pipeline (no audio leaves your machine).
- **Shadowing practice** — imitating a specific reference clip.
- A **transparent, hackable** scoring formula you can tune and inspect.
- Zero per-evaluation cost, or offline operation.
- A developer framework to build your own pronunciation tooling on top of.

---

## Honest limitations of this project

To set expectations clearly, relative to ELSA this project currently:

- Scores **fewer dimensions** (no word stress, listening, or grammar).
- Has a **coarser intonation** model (range ratio, not contour shape).
- Has a **fluency** pillar that overlaps with pronunciation/timbre.
- Produces **categorical**, not numeric, per-phoneme and per-word output.
- Uses a **single-L2** phoneme model (Korean learner English) with no per-speaker
  L1 adaptation — more forgiving than a native-trained model for that accent, but
  not re-weighted per user.
- Generates **deterministic, rule-based** feedback (no adaptive personalisation).

These are deliberate scoping choices for a local-first shadowing engine, not oversights.
Several are tracked as future work in `docs/README.md:751-766` (word-level drill mode,
formant analysis, accent-transfer detection).
