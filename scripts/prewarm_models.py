"""Build-time helper: pre-download the Kokoro + Wav2Vec2 models into HF_HOME.

Uses raw ``transformers`` / ``kokoro`` imports (NOT the project wrappers) so it
can run during ``docker build`` right after dependencies are installed —
*before* the application source is copied. That keeps this layer cached on the
dependency lock, so editing source code doesn't re-download ~1.5 GB of models.

Idempotent: if HF_HOME is already populated (e.g. cache-mount hit), it's a fast
no-op.
"""

from __future__ import annotations

import sys

WAV2VEC2 = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main() -> int:
    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    print(">>> prewarming Wav2Vec2 phoneme model (~1.2 GB)...", flush=True)
    Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2)
    Wav2Vec2ForCTC.from_pretrained(WAV2VEC2)
    hf_hub_download(WAV2VEC2, "vocab.json")

    print(">>> prewarming Kokoro TTS model + af_heart voice (~330 MB)...", flush=True)
    list(KPipeline(lang_code="a")("hello", voice="af_heart"))

    print(">>> prewarm complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
