# Phoneme Extraction & Comparison

> **Phase 2 deliverable.** Pronunciation feedback at the phoneme level.
> Takes a canonical `AudioSample` (from [Phase 1](audio-preprocessing.md)) and
> produces an espeak-IPA phoneme sequence, plus a structured diff and Phoneme
> Error Rate (PER) against a reference sequence.

## Overview

Two independent concerns live in `ai_speech_shadowing.core.phoneme`:

1. **Extraction** — decode an `AudioSample` into a phoneme sequence using a
   Wav2Vec2-CTC backend, then normalize the output to a canonical espeak-IPA
   notation. Used for the **user** side of every comparison and as a
   **fallback** for the reference side when no transcript is available.
2. **Comparison** — align a reference vs. hypothesis phoneme sequence with
   `difflib` to produce a structured diff (`PhonemeDiff`) and a numeric PER.

The comparison half is pure Python (no model) and is unit-tested exhaustively.
The extraction half loads a large transformer and is therefore lazy and
opt-in for tests.

### Pluggable backends

Extraction is backed by a `PhonemeModel` subclass, selected at runtime via the
`PHONEME_MODEL` env var (default `slplab-l2`). Both backends **never drop
tokens** — every recognized phoneme is mapped to a canonical espeak-IPA target,
never silently deleted.

| Key | Model | Output | Normalization |
| --- | --- | --- | --- |
| `slplab-l2` *(default)* | `slplab/wav2vec2-large-robust-L2-english-phoneme-recognition` | ARPAbet-39 (+ `_err`/`*` variants) | Mapped 1:1 to espeak IPA via `ARPABET_TO_IPA`, then segments rejoined to espeak units (`ɛ ɹ` → `ɛɹ`) so the hypothesis matches the kokoro G2P notation exactly |
| `espeak` | `facebook/wav2vec2-lv-60-espeak-cv-ft` | espeak IPA (native) | Tone markers dropped, length/stress stripped, everything else kept |

The default `slplab-l2` is trained on **L2 (non-native) English learner
speech**, so it recognizes the accented pronunciations a pronunciation coach
must catch — and even tags mispronounced phonemes with an `_err` suffix. The
multilingual `espeak` backend remains available for reference-side acoustic
fallback and its `vocab.json` is still the tokenization target of the G2P
reference pipeline (`core.g2p`).

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

## The backends

### `slplab-l2` (default) — L2 English learner model

| | |
| --- | --- |
| **Model** | `slplab/wav2vec2-large-robust-L2-english-phoneme-recognition` |
| **Base** | Wav2Vec2-Large-Robust, CTC-fine-tuned on Korean-English learner speech (ETRI/SNU) |
| **Vocabulary** | 89 ARPAbet tokens (39 base phonemes + `_err` mispronunciation flags + `*` unreleased-stop variants) |
| **Output mapping** | ARPAbet → espeak IPA segments via `ARPABET_TO_IPA` (42-entry table, 100 % coverage, 0 drops), then segments rejoined to espeak combined units (`ɛ ɹ` → `ɛɹ`) via the same greedy longest-match tokenization the G2P reference uses |
| **Input contract** | mono `float32` @ **16 kHz** |
| **Size** | ~0.3 B params |

The model was trained specifically on **non-native English** annotated with
pronunciation errors, which is exactly the shadowing use case. The `_err`
suffix marks phonemes the learner mispronounced (e.g. `g_err`); for diffing
the suffix is stripped and the base phoneme is scored, so the diff measures
accuracy against the *intended* phoneme.

After the per-token ARPAbet→IPA map, the segment sequence is rejoined into
espeak's combined units — `("ɛ", "ɹ")` becomes `("ɛɹ",)` — by re-tokenizing
the concatenated IPA string against the espeak vocabulary (the same
`_tokenize` pass `core.g2p` runs for the reference). This guarantees the
hypothesis and reference share one notation, eliminating the granularity
mismatch that would otherwise inflate PER (e.g. "bear": `ɛ ɹ` vs `ɛɹ`).

### `espeak` — multilingual fallback

| | |
| --- | --- |
| **Model** | `facebook/wav2vec2-lv-60-espeak-cv-ft` |
| **Base** | Wav2Vec2-Large (LV-60k), CTC-fine-tuned on Common Voice (60 languages) |
| **Vocabulary** | 392 espeak IPA phoneme tokens (`<pad>` is the CTC blank) |
| **Output mapping** | Tone-marker tokens dropped, length/stress marks stripped; everything else kept |
| **Size** | ~1.2 GB |

Its `vocab.json` doubles as the tokenization target for the G2P **reference**
side (`core.g2p._get_espeak_tokens`), so the reference pipeline is pinned to
espeak regardless of which hypothesis backend is active.

> **Why phoneme CTC models and not Whisper?** Whisper is an ASR model that uses
> language-model context to *guess* words, masking pronunciation errors. These
> Wav2Vec2-CTC models report the phonemes they actually heard at the sub-word
> acoustic level — which is the entire point of a pronunciation coach.

### A note on the tokenizer

The espeak model's `Wav2Vec2PhonemeCTCTokenizer` eagerly initialises an espeak
backend at construction time (it needs `phonemizer`/`espeakng-loader` for the
text→phoneme direction we don't use). To keep the **decode** path free of that
runtime coupling, the backend loads `vocab.json` directly and performs CTC
collapsing itself — see `PhonemeModel._ctc_collapse`. The net effect:
extracting phonemes from audio does not require espeak to be linkable.

## Extraction

```python
from ai_speech_shadowing.core.audio import AudioSample
from ai_speech_shadowing.core.phoneme import get_phoneme_model
from ai_speech_shadowing.core.preprocess import preprocess

sample = preprocess(AudioSample.from_wav("user.wav"))

extractor = get_phoneme_model()            # loads the default backend (one-time)
result = extractor.extract(sample)

print(result.phonemes)   # ('b', 'ɪ', 'ɡ', 'b', 'ɛ', 'ɹ', ...)
print(result.raw_text)   # "b ɪ ɡ b ɛ ɹ ..."
```

`get_phoneme_model(...)` returns a process-wide cached singleton so repeated
calls (CLI, API) reuse the loaded model. Backend selection, in priority order:

1. the `key=` argument,
2. the `PHONEME_MODEL` env var,
3. `DEFAULT_MODEL_KEY` (`"slplab-l2"`).

```python
get_phoneme_model(key="espeak")   # force the multilingual backend
# or: PHONEME_MODEL=espeak in the environment
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

# pin a device / alternate backend
ai-speech-shadowing phoneme user.wav --device cpu
ai-speech-shadowing phoneme user.wav --model espeak
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

- **Map, never drop.** Every backend maps recognized phonemes to a canonical
  espeak-IPA target. An earlier inventory-membership filter used to silently
  delete legitimate English vowels the multilingual model emits in alternate
  notations (bare `e`, barred `ᵻ`, r-coloured `ɔːɹ`), which destroyed real
  pronunciation info and turned "big bear" into `v k b`. The ARPAbet backend
  maps all 89 of its tokens 1:1; the espeak backend keeps everything except
  tone markers.
- **One canonical notation.** The system speaks a single phoneme notation —
  the espeak IPA units the kokoro G2P emits. The ARPAbet backend's output is
  rejoined into those units (`ɛ ɹ` → `ɛɹ`) via the same tokenization pass the
  reference uses, so hypothesis and reference never differ on granularity.
- **Pluggable backends via ABC + registry.** `PhonemeModel` is the abstract
  contract; `MODELS` registers concrete subclasses by short key; the
  `PHONEME_MODEL` env var selects at runtime. Add a backend by subclassing and
  registering one line.
- **Decode from `vocab.json`, not the tokenizer** (espeak backend). Avoids the
  hard espeak runtime dependency for the decode path and a known `phonemizer`
  ≥ 3.3 / `misaki` API incompatibility.
- **Lazy heavy imports.** `torch` and `transformers` are imported inside
  `_load()`, so `import ai_speech_shadowing.core.phoneme` (and the pure-diff
  unit tests) stay light and don't require the ML runtime.
- **Fixed 16 kHz contract.** The backends only accept 16 kHz. Callers run
  `preprocess()` first; the extractor raises a clear error if not.
- **`difflib` over a custom Levenshtein.** `SequenceMatcher` gives aligned
  opcodes for free (needed for the structured diff), and its `autojunk=False`
  mode is exact for sequences this short.
- **Asymmetric sourcing.** Reference phonemes come from G2P when the text is
  known (the canonical target); only the user side pays the inference cost.
  The acoustic path stays as a fallback so uploaded clips without a transcript
  still work. The reference side is pinned to the espeak vocabulary
  regardless of the active hypothesis backend.

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
- `tests/test_phoneme.py::TestStripTonesMarks` — tone-marker dropping,
  length/stress stripping, and the no-drop regression guard for alternate-
  notation vowels (`e`, `ᵻ`, `ɔːɹ`).
- `tests/test_phoneme.py::TestArpabetMapping` — `_err`/`*` suffix stripping,
  full `ARPABET_TO_IPA` coverage of all 42 base phonemes, end-to-end
  "big bear" mapping with zero drops.
- `tests/test_phoneme.py::TestSegmentCollapse` — r-coloured vowel rejoin
  (`ɛ ɹ` → `ɛɹ`), affricate rejoin (`t ʃ` → `tʃ`), maximal-token pass-through.
- `tests/test_phoneme.py::TestRegistry` — default key, unknown-key error.
- `tests/test_phoneme.py::TestExtractionWithModel` — marked `slow`; runs only
  under `uv run pytest --runslow`. Generates a Kokoro native reference,
  preprocesses it, and asserts the decoded phonemes start with `/h/`.
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
