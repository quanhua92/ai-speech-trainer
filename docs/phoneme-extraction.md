# Phoneme Extraction & Comparison

> **Phase 2 deliverable.** Pronunciation feedback at the phoneme level.
> Takes a canonical `AudioSample` (from [Phase 1](audio-preprocessing.md)) and
> produces an espeak-IPA phoneme sequence, plus a structured diff and Phoneme
> Error Rate (PER) against a reference sequence.

## Overview

Two independent concerns live in `ai_speech_shadowing.core.phoneme`:

1. **Extraction** — decode an `AudioSample` into a phoneme sequence using a
   Wav2Vec2-CTC model trained on an espeak IPA vocabulary.
2. **Comparison** — align a reference vs. hypothesis phoneme sequence with
   `difflib` to produce a structured diff (`PhonemeDiff`) and a numeric PER.

The comparison half is pure Python (no model) and is unit-tested exhaustively.
The extraction half loads a large transformer and is therefore lazy and
opt-in for tests.

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
> acoustic level — which is the entire point of a pronunciation coach.

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

## Test coverage

- `tests/test_phoneme.py::TestDiffPhonemes` — substitutions, deletions,
  insertions, mixed errors, empty-sequence edge cases, PER > 1.0.
- `tests/test_phoneme.py::TestPhonemeErrorRateHelper` — helper parity.
- `tests/test_phoneme.py::TestExtractionWithModel` — marked `slow`; runs only
  under `uv run pytest --runslow`. Generates a Kokoro native reference,
  preprocesses it, and asserts the decoded phonemes for "Hello world, this is
  a Kokoro TTS test." start with `/h/`.

```bash
uv run pytest                           # fast unit tests (47, 2 slow skipped)
uv run pytest --runslow tests/test_phoneme.py   # opt-in model tests (~15s)
```
