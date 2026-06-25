"""Build-time helper: pre-download the Kokoro + Wav2Vec2 models into HF_HOME.

Uses raw ``transformers`` / ``kokoro`` imports (NOT the project wrappers) so it
can run during ``docker build`` right after dependencies are installed —
*before* the application source is copied. That keeps this layer cached on
the dependency lock, so editing source code doesn't re-download ~1.5 GB of
models.

Also downloads all US + UK English Kokoro voices so the pregenerate step can
run without network. Idempotent.

Prewarms every registered phoneme backend (default ``slplab-l2`` + ``espeak``)
so either can be selected at runtime via ``PHONEME_MODEL`` without a cold
download. The espeak model's ``vocab.json`` is also fetched because the G2P
reference-side tokenizer reads it directly.
"""

from __future__ import annotations

import sys

from ai_speech_shadowing.core.phoneme import MODELS

# US + UK English voices to pre-download (each ~3 MB, one-time)
US_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael"]
UK_VOICES = ["bf_emma", "bf_isabella", "bf_alice", "bm_george", "bm_lewis"]


def main() -> int:
    # Prewarm every registered phoneme backend. Each loads its own processor /
    # model weights; the espeak backend additionally needs vocab.json (the G2P
    # tokenizer reads it). AutoProcessor/AutoModelForCTC cover both families.
    import contextlib

    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline
    from transformers import AutoModelForCTC, AutoProcessor

    for key, cls in MODELS.items():
        mid = cls.model_id
        print(f">>> prewarming phoneme model [{key}] {mid}…", flush=True)
        AutoProcessor.from_pretrained(mid)
        AutoModelForCTC.from_pretrained(mid)
        # not every backend ships a top-level vocab.json (the ARPAbet model
        # keeps its vocab inside the tokenizer); that's fine.
        with contextlib.suppress(Exception):
            hf_hub_download(mid, "vocab.json")

    print(">>> prewarming Kokoro TTS model + all US/UK English voices…", flush=True)
    p = KPipeline(lang_code="a")
    for v in US_VOICES:
        list(p("hello", voice=v))
        print(f"    downloaded voice: {v}", flush=True)
    p = KPipeline(lang_code="b")
    for v in UK_VOICES:
        list(p("hello", voice=v))
        print(f"    downloaded voice: {v}", flush=True)

    print(">>> prewarm complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
