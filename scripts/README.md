# scripts/

Exploration harnesses for kicking the tires on the libraries in the stack. Not
part of the shipped package — these are throwaway/dev scripts to validate
behavior before wiring a library into the engine.

## explore_kokoro.py

Synthesize text to a 24kHz WAV using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).

```bash
# American English (default voice af_heart)
uv run python scripts/explore_kokoro.py --text "Hello world"

# Apple Silicon: some torch ops lack MPS kernels, so enable CPU fallback
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/explore_kokoro.py --text "Hello world"

# Custom output path
uv run python scripts/explore_kokoro.py --text "Hello world" --out tmp/audio/hello.wav
```

The first run downloads the ~330MB Kokoro weights from HuggingFace into the HF
cache (`~/.cache/huggingface`). Outputs default to `tmp/audio/` (gitignored).

### Language codes & voices

`--lang` is the Kokoro single-letter code; the voice must match the family:

| Code | Language        | Example voices        |
| ---- | --------------- | --------------------- |
| `a`  | American English | `af_heart`, `am_adam` |
| `b`  | British English | `bf_emma`, `bm_george`|
| `e`  | Spanish         | `ef_silvia`           |
| `f`  | French          | `ff_siwis`            |
| `j`  | Japanese        | `jf_alpha`            |
| `z`  | Mandarin        | `zf_xiaobei`          |

### Requirements

No system packages needed — `uv sync` pulls in `espeakng-loader`, which
vendors `libespeak-ng` (kokoro's English OOD fallback) for macOS, Linux,
and Windows.
