# TTS Reference Generation

> **Phase 6 deliverable.** Automates creation of gold-standard native reference
> audio with **Kokoro**. Each reference lives under a deterministic slug folder
> with a `metadata.json`, is cached, and can be generated singly or in batch.

## Overview

`ai_speech_shadowing.tts.generator` provides:

| Component | Purpose |
| --- | --- |
| `slugify` | Deterministic filesystem-safe slug from any text |
| `ReferenceManager` | Owns the directory layout, `metadata.json`, and caching |
| `ReferenceManager.generate` | Synthesize one sentence with Kokoro (opt-in slow) |
| `ReferenceManager.generate_batch` | Synthesize a list of sentences |
| `parse_sentence_list` | Read a `#`-comment sentence file |

The slug/metadata/cache logic is pure and unit-tested; only `generate` loads
Kokoro (slow).

## Directory layout

```
<base_dir>/                              # default: data/references/
└── <slug>/                              # e.g. hello-world
    ├── metadata.json
    └── audio/
        └── kokoro-en-us/                # one folder per voice profile
            └── ref.wav
```

The **voice profile** folder is named `{engine}-{language}` (e.g.
`kokoro-en-us`, `kokoro-ja`). Kokoro's single-letter lang codes are mapped to
ISO-ish codes via `KOKORO_LANGUAGES` (`a→en-us`, `b→en-gb`, `e→es`, `f→fr`,
`j→ja`, `z→zh`, …).

### `metadata.json`

```json
{
  "text": "Hello world from Kokoro",
  "language": "en-us",
  "default_speaker": "af_heart",
  "phonemes": {
    "tokens": ["h", "ə", "l", "oʊ", "w", "ɜ˞", "l", "d", "f", "ɹ", "ʌ", "m", "k", "oʊ", "ɹ", "oʊ"],
    "source": "kokoro-g2p",
    "notation": "espeak-wav2vec2"
  },
  "updated_at": "2026-06-22T12:42:20+00:00",
  "audio": {
    "kokoro-en-us": {
      "file": "audio/kokoro-en-us/ref.wav",
      "sample_rate": 24000,
      "engine": "kokoro"
    }
  }
}
```

The `audio` dict is **merged** across profiles: regenerating the same slug with
a different voice/language adds a new entry without clobbering the existing ones
(`text`/`language`/`default_speaker`/`phonemes` are set with `setdefault`, so the
first generation wins).

### Captured phonemes

The `phonemes` block holds the canonical target pronunciation of the reference
text, captured at synthesis time from Kokoro's internal G2P (the same engine
misaki provides). It is:

- **A pure function of the text.** Voice-invariant — `af_heart`, `am_adam`, and
  any other Kokoro voice all produce identical `tokens` for the same `text`,
  because the G2P step is text-only. (This is asserted by
  `tests/test_tts.py::TestGenerate::test_phonemes_invariant_under_voice`.)
- **Normalized onto the Wav2Vec2 espeak inventory.** Stress marks, length marks,
  and misaki-specific notations (`ʤ` → `dʒ`, uppercase stressed diphthongs) are
  mapped onto the 392-token espeak vocabulary that `PhonemeExtractor._ctc_collapse`
  emits. See `ai_speech_shadowing.core.g2p.misaki_to_espeak_tokens`.
- **Cached, never recomputed.** Because the field is `setdefault`-ed, the first
  generation establishes it and subsequent regenerations with other voices reuse
  it. The evaluation pipeline can therefore skip running the 1.2 GB Wav2Vec2
  model on the reference audio entirely.

`source: "kokoro-g2p"` identifies the provenance — other sources may appear in
the future (e.g. `transcript-g2p` for uploaded references with a user-supplied
transcript). `notation: "espeak-wav2vec2"` records the normalization target so
future vocabulary changes can be detected.

## Slug derivation

`slugify` lowercases, strips accents (NFKD + ASCII), and replaces non-alphanumerics
with hyphens. Non-Latin text that yields no ASCII (e.g. CJK) falls back to a
stable 12-char SHA-1 hash, so every input maps to a valid slug:

```python
slugify("Hello world!")        # "hello-world"
slugify("Xin chào")            # "xin-chao"
slugify("你好世界")             # stable 12-char hash
```

## Generating references

### Python

```python
from ai_speech_shadowing.tts.generator import ReferenceConfig, ReferenceManager

mgr = ReferenceManager(ReferenceConfig(base_dir="data/references"))

# single — cached; re-running with the same text is a no-op
path = mgr.generate("Hello world", voice="af_heart", lang="a")
# → data/references/hello-world/audio/kokoro-en-us/ref.wav

mgr.generate("Hello world", force=True)   # bypass cache

# batch
mgr.generate_batch(["Hello world", "Goodbye world"])
```

### CLI

```bash
# single sentence
ai-speech-shadowing generate-reference --text "Hello world"

# batch from a sentence file (one per line; '#' lines are comments)
ai-speech-shadowing generate-reference --list sentences.txt

# custom voice/language/output dir; force regeneration
ai-speech-shadowing generate-reference --text "Xin chào" --voice af_heart \
    --output-dir data/references --force
```

The first call downloads the Kokoro-82M weights (~330 MB) into the HuggingFace
cache; subsequent calls reuse them.

### Listing references

```python
for ref in mgr.list_references():
    print(ref["slug"], ref["text"], list(ref["audio"]))
```

## High-fidelity offline references (Qwen TTS et al.)

Kokoro is optimised for speed; for higher-fidelity offline references the
intended workflow is:

1. Generate audio with your heavyweight TTS of choice (Qwen TTS, etc.) as a
   24 kHz mono WAV.
2. Drop it into the matching voice-profile folder manually, e.g.
   `data/references/<slug>/audio/qwen-en-us/ref.wav`.
3. Run `mgr.write_metadata(slug, text, lang, voice)` (or hand-edit
   `metadata.json`) to register the new profile alongside the Kokoro one.

Static, curated references give **reproducible evaluations** — the same
reference always yields the same baseline.

## Caching

`generate` checks `audio/<profile>/ref.wav` first and returns immediately if it
exists, unless `force=True`. This makes batch regeneration idempotent and keeps
re-evaluations fast (the reference is synthesized once, then reused by the
evaluation pipeline forever).

## Design decisions

- **Slug is the identity.** Text → slug is deterministic and stable, so a
  reference is regenerated into the same folder every time and the cache is
  keyed purely on the filesystem.
- **Hash fallback for non-Latin text.** Vietnamese/Chinese/Japanese still map to
  valid, stable slugs rather than empty strings.
- **Metadata merges, never clobbers.** Adding a second voice profile (a
  different speaker, a higher-fidelity engine) augments the same `audio` dict.
- **`data/` is gitignored.** Generated audio is regenerable; the repo ships no
  binary blobs. Curate and force-add specific references if you want them
  tracked.
- **Kokoro output is concatenated.** Long text that splits into multiple chunks
  is `np.concatenate`-d into one `ref.wav` at 24 kHz. The per-chunk G2P phoneme
  strings are likewise space-joined and normalized as one sequence, so
  `metadata.json["phonemes"]["tokens"]` reflects the whole utterance.

## Test coverage

`tests/test_tts.py` (19 fast + 4 slow):

- **Fast (pure, tmp_path):** `slugify` (ascii, punctuation, accents, CJK hash,
  truncation, whitespace); `voice_profile` naming; path layout; `exists()` cache
  check; metadata write/read, multi-profile merge, and `phonemes` persistence
  (`setdefault` semantics, `None` omission); `list_references` (sorted, empty);
  `parse_sentence_list` (skips blanks + `#` comments).
- **Slow (Kokoro):** single generation writes a 24 kHz WAV + metadata with
  captured G2P `phonemes`; cache skips regeneration without `force`; batch
  generates multiple references; **voice-invariance** — `af_heart` and `am_adam`
  yield identical `phonemes.tokens` for the same text.
