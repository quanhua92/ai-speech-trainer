# Audio Preprocessing

> **Phase 1 deliverable.** Foundational audio I/O and normalization layer.
> All downstream engine stages (phoneme, prosody, fluency) consume the
> standardized `AudioSample` produced by this module.

## Overview

The preprocessing layer is responsible for turning heterogeneous input audio
(arbitrary sample rate, channel count, loudness, lead-in/trailing silence) into
the single canonical form demanded by the downstream ML models:

> **mono `float32` @ 16 kHz**, silence-trimmed and volume-normalized.

The pipeline lives in two modules:

| Module | Responsibility |
| --- | --- |
| `ai_speech_shadowing.core.audio` | The `AudioSample` container + WAV I/O with format validation |
| `ai_speech_shadowing.core.preprocess` | Mono downmix, resampling, silence trim, volume normalization |

## The `AudioSample` contract

`AudioSample` is the typed envelope passed between every pipeline stage:

```python
from ai_speech_shadowing.core.audio import AudioSample, TARGET_SAMPLE_RATE

# fields
sample.waveform      # np.ndarray, float32, shape (n,) mono or (n, channels)
sample.sample_rate   # int, Hz

# derived properties
sample.num_samples   # int
sample.channels      # int (1 for mono)
sample.is_mono       # bool
sample.duration      # float, seconds
```

It is a **frozen** (immutable) dataclass — transforms return *new* instances
rather than mutating, so a sample can be safely shared across stages. The
constructor validates invariants and rejects bad input eagerly:

- non-`ndarray` waveforms → `TypeError`
- wrong dtype → coerced to `float32`
- not 1D/2D, empty, or `sample_rate <= 0` → `ValueError`

### Loading audio

```python
from ai_speech_shadowing.core.audio import AudioSample, AudioLoadError

try:
    sample = AudioSample.from_wav("recording.wav")
except AudioLoadError as e:
    ...  # missing file, unreadable header, empty audio, bad sample rate
```

`from_wav` reads anything `soundfile` can decode (WAV, FLAC, OGG, …), validates
the header via `soundfile.info`, and returns a multi-channel sample unchanged —
downmixing happens in the preprocess step, not at load time.

### Writing audio

```python
sample.to_wav("out.wav")                 # 32-bit float WAV (lossless default)
sample.to_wav("out.wav", subtype="PCM_16")  # 16-bit PCM for distribution
```

The default `subtype="FLOAT"` is deliberate: the engine never wants quantization
loss between pipeline stages. Use `PCM_16` only for human-facing distribution
files.

## The preprocessing pipeline

```
input ──► to_mono ──► resample(16k) ──► trim_silence ──► normalize_volume ──► output
```

The order is fixed because each step's assumptions depend on the previous one
(`trim_silence` requires mono input; normalization is meaningful only after
silence is removed). `preprocess(...)` encodes this order; individual functions
are also exposed for advanced use.

```python
from ai_speech_shadowing.core.preprocess import preprocess

canonical = preprocess(sample)                     # all defaults
canonical = preprocess(sample, trim_top_db=None)   # keep silence
canonical = preprocess(sample, normalize="rms")    # RMS instead of peak
canonical = preprocess(sample, normalize=None)     # no normalization
```

| Stage | Function | Default | Notes |
| --- | --- | --- | --- |
| Mono downmix | `to_mono(sample)` | — | Averages channels; no-op if already mono |
| Resample | `resample(sample, target_sr=16000)` | 16 kHz | `soxr_hq` resampler; per-channel for stereo |
| Silence trim | `trim_silence(sample, *, top_db=30)` | 30 dB | `librosa.effects.split`; strips interior gaps too |
| Normalize | `normalize_volume(sample, *, method="peak")` | peak | `"peak"` → 0.99; `"rms"` → 0.06 (~ -24 dBFS) |

## CLI usage

The `preprocess` command runs the full pipeline on a file (useful for
inspecting intermediate output or preparing fixtures):

```bash
# default: mono → 16kHz → trim @30dB → peak-normalize; writes <input>.preprocessed.wav
ai-speech-shadowing preprocess recording.wav

# explicit output + RMS normalization
ai-speech-shadowing preprocess recording.wav -o tmp/audio/clean.wav --normalize rms

# disable trimming (pass 0) and normalization
ai-speech-shadowing preprocess recording.wav --trim-top-db 0 --normalize none
```

Output line: `wrote <path> (<duration>s, <sr> Hz, <channels>ch)`.

## Design decisions

- **16 kHz mono target.** Wav2Vec2-CTC and the MFCC/Parselmouth extractors all
  expect a single 16 kHz channel. Enforcing it once at the front keeps every
  downstream module branch-free.
- **Frozen `AudioSample`.** Prevents accidental in-place mutation across stages;
  copies are cheap relative to the model inference that follows.
- **Lossless float WAV by default.** 16-bit PCM introduces ~3e-5 quantization
  noise — acceptable for playback, undesirable when the same file may be
  re-read by the phoneme engine. `subtype="FLOAT"` makes round-trips exact.
- **Gain safety cap of 100 (40 dB).** Generous enough to rescue genuinely quiet
  recordings, while still preventing pathological blow-up of near-silent input
  (true silence is guarded by an epsilon check and left untouched).
- **Silence trim before normalize.** Normalizing with large silent margins
  skews RMS; trimming first yields stable, comparable loudness.
- **`trim_silence` requires mono.** `librosa.effects.split` operates on a 1D
  signal; the pipeline guarantees mono by the time trim runs. Calling it
  directly on stereo audio raises a clear `ValueError`.

## Edge cases handled

- **Empty / corrupt files** → `AudioLoadError` at load time.
- **All-silence clip** → `trim_silence` returns it unchanged; `normalize_volume`
  detects near-zero amplitude and returns it unchanged (no divide-by-zero).
- **Already-canonical input** → `to_mono` and `resample` are no-ops (identity).
- **Stereo at non-16 kHz** → resampled per-channel, then downmixed to mono.

## Test coverage

Tests are deterministic and binary-free — WAV fixtures are synthesized on the
fly in `tests/conftest.py` (sine tones at various sample rates, stereo pairs,
silence-padded clips, quiet and pure-silence clips). See:

- `tests/test_audio.py` — `AudioSample` construction/validation + WAV I/O round-trips
- `tests/test_preprocess.py` — each pipeline stage + the full pipeline
- `tests/test_cli.py` — `preprocess` command end-to-end via Typer's `CliRunner`
