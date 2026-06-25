"""Grapheme-to-phoneme (G2P) helpers for the reference side.

When the reference text is known (TTS-generated, or a transcript supplied
alongside an uploaded clip), the canonical target pronunciation is the G2P
output — not what an acoustic model "heard" in the synthesized audio. This
module wraps misaki (the same G2P engine Kokoro uses internally) and normalizes
its output onto the Wav2Vec2 espeak inventory so reference and hypothesis
phonemes are directly comparable.

The pure-Python helpers (``norm_misaki``, ``_tokenize``, ``_get_espeak_tokens``,
``g2p_words``) live here so they can be shared by:

- :mod:`ai_speech_shadowing.tts.generator` — captures Kokoro's G2P at synthesis
  time and persists it to ``metadata.json`` (the reference phoneme source).
- :mod:`ai_speech_shadowing.core.wordalign` — best-effort word-level diff.

``misaki_to_espeak_tokens`` is the one-call wrapper for converting a raw misaki
phoneme string (e.g. Kokoro's per-chunk ``_ps``) into a tuple of espeak tokens.
"""

# IPA characters in this module are intentional.
# ruff: noqa: RUF001

from __future__ import annotations

from pathlib import Path

from ai_speech_shadowing.core.phoneme import ESPEAK_MODEL_ID

# misaki uppercase "stress" vowel codes -> espeak equivalents.
_MISAKI_UPPER: dict[str, str] = {
    "A": "eɪ",  # FACE
    "O": "oʊ",  # GOAT
    "I": "aɪ",  # PRICE
    "W": "aʊ",  # MOUTH
    "U": "u",  # GOOSE (best-effort)
    "Y": "aɪ",
    "E": "ɛ",  # DRESS
}

_espeak_tokens: list[str] | None = None


def _get_espeak_tokens(model_id: str = ESPEAK_MODEL_ID) -> list[str]:
    """Lazy-load the espeak vocab tokens (longest-first) from the HF cache."""
    global _espeak_tokens
    if _espeak_tokens is None:
        import json

        from huggingface_hub import hf_hub_download

        vocab = json.loads(Path(hf_hub_download(model_id, "vocab.json")).read_text("utf-8"))
        _espeak_tokens = sorted((t for t in vocab if not t.startswith("<")), key=len, reverse=True)
    return _espeak_tokens


def _tokenize(s: str, tokens: list[str]) -> list[str]:
    """Greedy longest-match tokenization against an espeak vocab."""
    out: list[str] = []
    while s:
        for tok in tokens:
            if s.startswith(tok):
                out.append(tok)
                s = s[len(tok) :]
                break
        else:
            s = s[1:]  # drop unrecognised char
    return out


def norm_misaki(phonemes: str) -> str:
    """Normalize a misaki phoneme string toward espeak (pure string ops)."""
    for ch in "ˈˌː":
        phonemes = phonemes.replace(ch, "")
    phonemes = phonemes.replace("ʤ", "dʒ").replace("ʧ", "tʃ")
    for upper, espeak in _MISAKI_UPPER.items():
        phonemes = phonemes.replace(upper, espeak)
    return phonemes


def misaki_to_espeak_tokens(phonemes: str) -> tuple[str, ...]:
    """Normalize a misaki phoneme string and tokenize it against the espeak vocab.

    Use this to convert Kokoro's per-chunk G2P output (or any other misaki
    output) into a tuple of espeak tokens compatible with the Wav2Vec2 decode
    path. Drops stress marks, length marks, and unrecognized characters.

    Note: the input must be a **phoneme** string (already G2P'd), not the raw
    reference text. For the text → tokens path use :func:`text_to_espeak_tokens`.
    """
    return tuple(_tokenize(norm_misaki(phonemes), _get_espeak_tokens()))


def text_to_espeak_tokens(text: str) -> tuple[str, ...]:
    """Run misaki G2P on ``text`` and return the normalized espeak token sequence.

    End-to-end text → phonemes path: G2P the text, drop non-alphabetic tokens
    (punctuation), normalize each word's phonemes onto the espeak inventory,
    and concatenate into one flat tuple. Used by the ``backfill-phonemes`` CLI
    to populate references whose phonemes weren't captured at synthesis time.
    """
    import misaki.en as en

    g2p = en.G2P()
    _full, mtokens = g2p(text)
    tokens = _get_espeak_tokens()
    out: list[str] = []
    for mt in mtokens:
        if mt.phonemes and any(c.isalpha() for c in mt.text):
            out.extend(_tokenize(norm_misaki(mt.phonemes), tokens))
    return tuple(out)


def g2p_words(text: str) -> list[tuple[str, list[str]]]:
    """Phonemize ``text`` into ``(word, normalized-phoneme-tokens)`` pairs.

    Punctuation / non-alphabetic tokens are dropped.
    """
    import misaki.en as en

    g2p = en.G2P()
    _full, mtokens = g2p(text)
    tokens = _get_espeak_tokens()
    out: list[tuple[str, list[str]]] = []
    for mt in mtokens:
        if mt.phonemes and any(c.isalpha() for c in mt.text):
            out.append((mt.text, _tokenize(norm_misaki(mt.phonemes), tokens)))
    return out
