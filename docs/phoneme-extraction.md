# Phoneme Extraction & Comparison

> **Phase 2 deliverable.** Pronunciation feedback at the phoneme level.
> Takes a canonical `AudioSample` (from [Phase 1](audio-preprocessing.md)) and
> produces an espeak-IPA phoneme sequence, plus a structured diff and Phoneme
> Error Rate (PER) against a reference sequence.

## Overview

Two independent concerns live in `ai_speech_shadowing.core.phoneme`:

1. **Extraction** — decode an `AudioSample` into a phoneme sequence using a
   Wav2Vec2-CTC model trained on an espeak IPA vocabulary. Used for the **user**
   side of every comparison and as a **fallback** for the reference side when no
   transcript is available (e.g. an uploaded clip without text).
2. **Comparison** — align a reference vs. hypothesis phoneme sequence with
   `difflib` to produce a structured diff (`PhonemeDiff`) and a numeric PER.

The comparison half is pure Python (no model) and is unit-tested exhaustively.
The extraction half loads a large transformer and is therefore lazy and
opt-in for tests.

### Asymmetric sourcing: reference vs. hypothesis

Phoneme sourcing is **not symmetric** — that's the field-standard design for
pronunciation assessment (SpeechRater / ELSA / GOP literature):

| Side | Source | Why |
| --- | --- | --- |
| **Reference** (when text is known) | G2P from the known text — captured from Kokoro at synthesis time and cached in `metadata.json["phonemes"]` | The text is ground truth; no acoustic model can be more correct than the phoneme string the synthesizer was told to produce. Already sentence-level (handles context, weak forms, linking) because misaki is utterance-level, not per-word. |
| **Reference** (when text is unknown) | Wav2Vec2 acoustic recognition — the fallback path | A future "upload your own .mp3" feature with no transcript has no G2P target; the recognizer is the only option. |
| **Hypothesis** (always) | Wav2Vec2 acoustic recognition | Captures what the user physically said, including genuine connected-speech reductions and errors. |

The branch lives in `feedback.py::evaluate`:

```python
if reference_phonemes is not None:
    ref_phonemes = tuple(reference_phonemes)          # G2P target path
    ref_source = "kokoro-g2p"
else:
    ref_phonemes = extractor.extract(reference_sample).phonemes  # acoustic fallback
    ref_source = "wav2vec2-acoustic"
```

The provenance is recorded on the `FeedbackReport` as `reference_phoneme_source`
and surfaced through the API (`EvaluationResponse.reference_phoneme_source`)
and the demo UI (the "Phoneme alignment" heading flips between `(target)` and
`(recognized)` accordingly).

The shared G2P machinery (misaki normalization + espeak tokenization) lives in
`ai_speech_shadowing.core.g2p`. See [`tts-reference.md`](tts-reference.md) for
how Kokoro's per-chunk `_ps` output is captured at synthesis time.

## The model

| | |
| --- | --- |
| **Model** | `facebook/wav2vec2-lv-60-espeak-cv-ft` |
| **Base** | Wav2Vec2-Large (LV-60k), CTC-fine-tuned on Common Voice |
| **Vocabulary** | 392 espeak IPA phoneme tokens (`<pad>` is the CTC blank) |
| **Input contract** | mono `float32` @ **16 kHz** (exactly what `preprocess()` outputs) |
| **Size** | ~1.2 GB (downloaded once into the HuggingFace cache) |

> **Why this model and not Whisper?** Whisper is an ASR model that uses
> language-model context to *guess* words, masking pronunciation errors. This
> Wav2Vec2-CTC model reports the phonemes it actually heard at the sub-word
> acoustic level — which is the entire point of a pronunciation coach. This
> rationale applies to the **user** side. The **reference** side (when text is
> known) goes further still: it skips the recognizer entirely and uses the G2P
> target captured from Kokoro, which is by definition the intended
> pronunciation rather than a recognition of it.

### A note on the tokenizer

The model's `Wav2Vec2PhonemeCTCTokenizer` eagerly initialises an espeak
backend at construction time (it needs `phonemizer`/`espeakng-loader` for the
text→phoneme direction we don't use). To keep the **decode** path free of that
runtime coupling, `PhonemeExtractor` loads `vocab.json` directly and performs
CTC collapsing itself — see `_ctc_collapse`. The net effect: extracting
phonemes from audio does not require espeak to be linkable.

## Extraction

```python
from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.phoneme import PhonemeExtractor
from ai_speech_shadowing.core.preprocess import preprocess

sample = preprocess(AudioSample.from_wav("user.wav"))

extractor = PhonemeExtractor()              # loads the model (one-time, ~1.2GB)
result = extractor.extract(sample)

print(result.phonemes)   # ('h', 'ə', 'l', 'oʊ', ...)
print(result.raw_text)   # "h ə l oʊ ..."
```

`get_extractor(...)` returns a process-wide cached singleton so repeated calls
(CLI, future API) reuse the loaded model:

```python
from ai_speech_shadowing.core.phoneme import get_extractor
extractor = get_extractor()   # loads on first call, cached thereafter
```

**Edge cases / validation:**
- Input not at 16 kHz → `ValueError` ("… Run preprocess() first.").
- Multi-channel input is downmixed to mono on the fly (lenient).
- Silence / noise-only audio → collapses to an empty phoneme tuple (handled
  cleanly by the diff layer).

### CLI

```bash
# extract phonemes from any audio file (preprocesses automatically)
ai-speech-shadowing phoneme user.wav

# skip preprocessing (input must already be 16kHz mono)
ai-speech-shadowing phoneme canonical.wav --no-preprocess

# pin a device / alternate model
ai-speech-shadowing phoneme user.wav --device cpu
```

## Comparison

`diff_phonemes(reference, hypothesis)` aligns two phoneme sequences with
`difflib.SequenceMatcher` (`autojunk=False` so short sequences aren't
heuristically distorted) and returns a `PhonemeDiff`:

```python
from ai_speech_shadowing.core.phoneme import diff_phonemes

# reference (native) vs. hypothesis (user) — note the /l/ → /ɹ/ substitution
d = diff_phonemes(
    ["h", "ə", "l", "oʊ"],
    ["h", "ə", "ɹ", "oʊ"],
)

d.matches          # 3
d.substitutions    # 1
d.deletions        # 0
d.insertions       # 0
d.phoneme_error_rate   # 0.25
d.accuracy         # 0.75
d.operations       # (PhonemeOp('match','h','h'), ..., PhonemeOp('sub','l','ɹ'), ...)
```

Each `PhonemeOp` has a `tag` (`match` | `sub` | `del` | `ins`) plus the
`ref`/`hyp` phoneme strings — ready for the colour-coded diff rendering planned
in Phase 5 / Phase 8.

### How unequal replacements are counted

A `replace` block where the reference and hypothesis differ in length is split
into `min(len(ref), len(hyp))` substitutions plus a surplus of insertions or
deletions. This keeps PER well-defined under insertion/deletion-heavy errors.

### PER definition

```
PER = (substitutions + deletions + insertions) / len(reference)
```

Lower is better. Edge cases:
- both sequences empty → `0.0`
- empty reference with any errors → `1.0`
- PER can exceed `1.0` when the hypothesis inserts many extra phonemes

## Design decisions

- **Decode from `vocab.json`, not the tokenizer.** Avoids the hard espeak
  runtime dependency for the decode path and a known `phonemizer` ≥ 3.3 /
  `misaki` API incompatibility. The map is a 392-entry dict; trivial to load.
- **Lazy heavy imports.** `torch` and `transformers` are imported inside
  `PhonemeExtractor.__init__`, so `import ai_speech_shadowing.core.phoneme`
  (and the pure-diff unit tests) stay light and don't require the ML runtime.
- **Fixed 16 kHz contract.** Mirrors Phase 1: the model only accepts 16 kHz.
  Callers run `preprocess()` first; the extractor raises a clear error if not.
- **`difflib` over a custom Levenshtein.** `SequenceMatcher` gives aligned
  opcodes for free (needed for the structured diff), and its `autojunk=False`
  mode is exact for sequences this short.
- **Asymmetric sourcing.** Reference phonemes come from G2P when the text is
  known (the canonical target); only the user side pays the 1.2 GB inference
  cost. The acoustic path stays as a fallback so uploaded clips without a
  transcript still work.

### Calibration caveat (PER pre- vs. post-cutover)

Before the asymmetric sourcing shipped, *both* reference and hypothesis went
through the same Wav2Vec2 recognizer. A perfect user mimic therefore scored
~100% PER, because the recognizer made identical errors on both clips and the
errors cancelled. With the reference side now sourced from G2P, that
forgiving symmetry is gone — the reference is the canonical target, so any
drift between (a) misaki's canonical form and (b) what the recognizer hears in
the user's audio correctly registers as error.

The new scoring is **more accurate** (the reference no longer contributes its
own recognition noise to the baseline) but **not directly comparable** to
scores recorded before the cutover. Saved history across the cutover is
therefore invalid as a trend signal — old `eval_*.json` files carry the
`reference_phoneme_source: "wav2vec2-acoustic"` stamp (or no stamp, for
pre-cutover reports), which can be used to filter them out of any trend
analysis.

## Test coverage

- `tests/test_phoneme.py::TestDiffPhonemes` — substitutions, deletions,
  insertions, mixed errors, empty-sequence edge cases, PER > 1.0.
- `tests/test_phoneme.py::TestPhonemeErrorRateHelper` — helper parity.
- `tests/test_phoneme.py::TestExtractionWithModel` — marked `slow`; runs only
  under `uv run pytest --runslow`. Generates a Kokoro native reference,
  preprocesses it, and asserts the decoded phonemes for "Hello world, this is
  a Kokoro TTS test." start with `/h/`.
- `tests/test_g2p.py` — pure-string unit tests for `norm_misaki` and
  `misaki_to_espeak_tokens` (stress stripping, affricate unfolding, diphthong
  mapping, multi-char tokenization, mocked espeak vocab — no HF download).
- `tests/test_feedback.py::TestReferencePhonemeSource` — fast tests using a
  fake extractor: verifies the G2P path skips acoustic recognition on the
  reference, the acoustic fallback runs on both sides, and the source stamp
  propagates through `FeedbackReport` and `report_to_dict`.

```bash
uv run pytest                           # fast unit tests (50+, 2 slow skipped)
uv run pytest --runslow tests/test_phoneme.py   # opt-in model tests (~15s)
```
