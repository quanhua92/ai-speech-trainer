"""TTS reference generation: text → native audio via Kokoro."""

from ai_speech_shadowing.tts.generator import (
    KOKORO_LANGUAGES,
    KOKORO_SAMPLE_RATE,
    ReferenceConfig,
    ReferenceManager,
    slugify,
)

__all__ = [
    "KOKORO_LANGUAGES",
    "KOKORO_SAMPLE_RATE",
    "ReferenceConfig",
    "ReferenceManager",
    "slugify",
]
