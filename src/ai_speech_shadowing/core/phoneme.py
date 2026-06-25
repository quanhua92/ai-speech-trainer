"""Phoneme-level pronunciation analysis: diff/PER + acoustic extraction.

Two concerns live here:

1. **Diff / alignment** (pure Python, no ML deps): :class:`PhonemeOp`,
   :class:`PhonemeDiff`, :func:`diff_phonemes` — align two phoneme token
   sequences in a canonical espeak-IPA notation and compute a PER.

2. **Acoustic extraction** (lazy ML deps): the :class:`PhonemeModel` ABC plus
   concrete backends. Each decodes audio → phoneme tokens and normalizes them
   to the canonical espeak-IPA notation so the diff against the G2P reference
   works regardless of which recognizer produced them. **No backend drops
   tokens** — every recognized phoneme is mapped to a canonical target, never
   silently deleted (an earlier inventory-membership filter used to delete
   legitimate vowels the multilingual model emits in alternate notations).

   Backends are selectable at runtime via the ``PHONEME_MODEL`` env var:

   - ``"slplab-l2"`` (default): ``slplab/wav2vec2-large-robust-L2-english-
     phoneme-recognition``. Trained on L2 (non-native) English learner speech;
     emits ARPAbet-39 which is mapped 1:1 to espeak IPA (100 % coverage, 0
     drops) via :data:`ARPABET_TO_IPA`.
   - ``"espeak"``: ``facebook/wav2vec2-lv-60-espeak-cv-ft``. Multilingual;
     emits espeak IPA natively. Tone-marker tokens (digit-bearing) are dropped
     and length/stress marks stripped, but every other token is kept.
"""

from __future__ import annotations

# IPA characters in this module are intentional.
# ruff: noqa: RUF001, RUF002, RUF003
import difflib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ai_speech_shadowing.core.audio import TARGET_SAMPLE_RATE, AudioSample

if TYPE_CHECKING:
    from collections.abc import Sequence

ESPEAK_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
"""The multilingual espeak-IPA Wav2Vec2 model.

Also the model whose ``vocab.json`` the G2P/reference-side tokenizer
(:mod:`ai_speech_shadowing.core.g2p`) tokenizes against — the reference side
is pinned to espeak regardless of which hypothesis backend is active, so the
two sides share a vocabulary.
"""

ARPABET_MODEL_ID = "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition"
"""The L2-English ARPAbet Wav2Vec2 model (default hypothesis backend)."""

MODEL_SAMPLE_RATE: int = TARGET_SAMPLE_RATE  # 16000 — the only contract every backend imposes
BLANK_TOKEN = "<pad>"  # CTC blank token (vocab id 0 on both backends)


# --------------------------------------------------------------------------- #
# ARPAbet -> espeak-IPA mapping (covers every token the slplab model can emit)
# --------------------------------------------------------------------------- #
# The slplab model emits ARPAbet-39 plus ``_err`` (mispronunciation) and ``*``
# (unreleased-stop) suffix variants. After stripping those suffixes every base
# token has exactly one espeak-IPA equivalent below — 100 % coverage, no drops.
# Note: ARPAbet "g" maps to Unicode U+0261 (ɡ), NOT ASCII "g", to match the
# espeak inventory the G2P reference is normalized to.
ARPABET_TO_IPA: dict[str, str] = {
    # monophthongs / diphthongs
    "aa": "ɑ",
    "ae": "æ",
    "ah": "ʌ",
    "ao": "ɔ",
    "aw": "aʊ",
    "ax": "ə",
    "ay": "aɪ",
    "eh": "ɛ",
    "er": "ɝ",
    "eu": "ɜ",
    "ey": "eɪ",
    "ih": "ɪ",
    "iy": "i",
    "o": "oʊ",
    "ow": "oʊ",
    "oy": "ɔɪ",
    "uh": "ʊ",
    "uw": "u",
    # consonants
    "b": "b",
    "ch": "tʃ",
    "d": "d",
    "dh": "ð",
    "f": "f",
    "g": "ɡ",
    "hh": "h",
    "jh": "dʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "ng": "ŋ",
    "p": "p",
    "r": "ɹ",
    "s": "s",
    "sh": "ʃ",
    "t": "t",
    "th": "θ",
    "ts": "ts",
    "v": "v",
    "w": "w",
    "y": "j",
    "z": "z",
    "zh": "ʒ",
}


def _arpabet_base(tok: str) -> str:
    """Strip the ``_err`` (mispronunciation) and ``*`` (unreleased) suffixes.

    The slplab model tags learner errors with ``_err`` (e.g. ``g_err``) and
    unreleased stops with ``*`` (e.g. ``b*``). For diffing purposes the base
    phoneme is the intended target, so the suffix is dropped before lookup.
    """
    if tok.endswith("_err"):
        tok = tok[:-4]
    return tok[:-1] if tok.endswith("*") else tok


def _segments_to_espeak_units(segments: tuple[str, ...]) -> tuple[str, ...]:
    """Rejoin IPA segments into espeak's combined units via greedy longest-match.

    Segmental recognizers (slplab) emit vowels and glides as separate tokens
    (``ɛ ɹ``), while the kokoro/espeak G2P reference bundles the same sounds
    into single tokens (``ɛɹ``). Concatenating the IPA segments and
    re-tokenizing against the espeak vocabulary — the **same** longest-match
    pass the G2P reference uses (see :func:`ai_speech_shadowing.core.g2p._tokenize`)
    — collapses them into exactly the units the reference produces, so the
    hypothesis and reference share one notation with no granularity mismatch.

    Examples:
        ``("ɛ", "ɹ")``            -> ``("ɛɹ",)``       r-coloured vowel rejoined
        ``("b", "ɪ", "ɡ")``       -> ``("b", "ɪ", "ɡ")``  already maximal
        ``("t", "ʃ")``            -> ``("tʃ",)``        affricate rejoined
    """
    from ai_speech_shadowing.core.g2p import _get_espeak_tokens, _tokenize

    return tuple(_tokenize("".join(segments), _get_espeak_tokens()))


def _strip_tones_marks(phonemes: tuple[str, ...], language: str | None) -> tuple[str, ...]:
    """Normalize multilingual espeak output for a non-tonal reference language.

    Drops digit-bearing tokens (Mandarin tone markers like ``"ɑ5"`` are never
    valid in a non-tonal reference) and strips length (``"ː"``) and stress
    (``"ˈ"``, ``"ˌ"``) marks so the hypothesis aligns with the G2P reference
    normalization (see :func:`ai_speech_shadowing.core.g2p.norm_misaki`).

    **Nothing else is dropped.** An earlier inventory-membership filter used to
    delete legitimate English vowels the model emits in alternate notations
    (bare ``"e"``, barred ``"ᵻ"``, r-coloured ``"ɔːɹ"``); those are now kept so
    the learner's pronunciation isn't silently censored. On clean English
    speech this retains ~98.5 % of tokens — the only removed tokens are the
    tone markers this function drops.

    Non-English references pass through unchanged (the model covers them
    natively, tones and all).
    """
    if not language or not language.lower().startswith("en"):
        return phonemes
    out: list[str] = []
    for tok in phonemes:
        if any(c.isdigit() for c in tok):  # tone markers — never English
            continue
        out.append(tok.replace("ː", "").replace("ˈ", "").replace("ˌ", ""))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Diff structures (model-agnostic)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PhonemeOp:
    """A single aligned operation between reference and hypothesis.

    tag is one of: ``match`` | ``sub`` | ``del`` | ``ins``.
    """

    tag: str
    ref: str | None
    hyp: str | None


@dataclass(frozen=True, slots=True)
class PhonemeDiff:
    """Structured result of aligning two phoneme sequences."""

    operations: tuple[PhonemeOp, ...]
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]

    @property
    def phoneme_error_rate(self) -> float:
        """PER = (sub + del + ins) / len(reference). Lower is better.

        Edge cases: empty reference with any errors -> 1.0; both empty -> 0.0.
        PER can exceed 1.0 when the hypothesis inserts many extra phonemes.
        """
        errors = self.substitutions + self.deletions + self.insertions
        n = len(self.reference)
        if n == 0:
            return 1.0 if errors else 0.0
        return errors / n

    @property
    def accuracy(self) -> float:
        return max(0.0, 1.0 - self.phoneme_error_rate)

    @property
    def is_perfect(self) -> bool:
        return self.phoneme_error_rate == 0.0


def diff_phonemes(reference: Sequence[str], hypothesis: Sequence[str]) -> PhonemeDiff:
    """Align two phoneme sequences and return a structured diff + counts.

    Uses :class:`difflib.SequenceMatcher` (``autojunk=False`` so short phoneme
    sequences aren't heuristically distorted). Unequal-length ``replace`` blocks
    are split into substitutions + a surplus of insertions/deletions.
    """
    ref = list(reference)
    hyp = list(hypothesis)

    ops: list[PhonemeOp] = []
    matches = subs = dels = inss = 0

    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_blk = ref[i1:i2]
        hyp_blk = hyp[j1:j2]
        if tag == "equal":
            for r in ref_blk:
                ops.append(PhonemeOp("match", r, r))
                matches += 1
        elif tag == "replace":
            n = min(len(ref_blk), len(hyp_blk))
            for k in range(n):
                ops.append(PhonemeOp("sub", ref_blk[k], hyp_blk[k]))
                subs += 1
            for k in range(n, len(ref_blk)):
                ops.append(PhonemeOp("del", ref_blk[k], None))
                dels += 1
            for k in range(n, len(hyp_blk)):
                ops.append(PhonemeOp("ins", None, hyp_blk[k]))
                inss += 1
        elif tag == "delete":
            for r in ref_blk:
                ops.append(PhonemeOp("del", r, None))
                dels += 1
        elif tag == "insert":
            for h in hyp_blk:
                ops.append(PhonemeOp("ins", None, h))
                inss += 1

    return PhonemeDiff(
        operations=tuple(ops),
        matches=matches,
        substitutions=subs,
        deletions=dels,
        insertions=inss,
        reference=tuple(ref),
        hypothesis=tuple(hyp),
    )


def phoneme_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """Convenience wrapper returning just the PER float."""
    return diff_phonemes(reference, hypothesis).phoneme_error_rate


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PhonemeResult:
    """The decoded phoneme sequence from one audio clip."""

    phonemes: tuple[str, ...]
    raw_text: str

    def __len__(self) -> int:
        return len(self.phonemes)

    def __iter__(self):
        return iter(self.phonemes)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PhonemeModel(ABC):
    """Abstract acoustic phoneme recognizer: :class:`AudioSample` → espeak-IPA tokens.

    Concrete backends decode audio via a Wav2Vec2-CTC model, CTC-collapse the
    output, then normalize tokens to the canonical espeak-IPA notation the diff
    layer and the G2P reference share. **No backend drops tokens** — every
    recognized phoneme is mapped to a canonical target.

    Heavy dependencies (``torch``, ``transformers``) are imported inside
    :meth:`_load`, so merely importing this module (and the pure-diff unit
    tests) stays light and does not require the ML runtime.
    """

    model_id: ClassVar[str]
    """HuggingFace model id, set by each subclass."""

    def __init__(self, device: str = "auto") -> None:
        self.device = _resolve_device(device)
        self._load()

    # Subclasses implement ----------------------------------------------------
    @abstractmethod
    def _load(self) -> None:
        """Load the model/processor/vocab.

        Must set ``_feature_extractor``, ``_model``, ``_id2token`` and
        ``_blank_id`` on ``self``.
        """

    @abstractmethod
    def _normalize(self, raw: tuple[str, ...], language: str | None) -> tuple[str, ...]:
        """Map raw CTC tokens to canonical espeak-IPA notation. Never drop."""

    # Shared decode path ------------------------------------------------------
    def extract(self, sample: AudioSample, *, language: str | None = None) -> PhonemeResult:
        """Decode a canonical AudioSample (16 kHz) into a phoneme sequence.

        Args:
            language: Reference language code (e.g. ``"en-us"``). Used by the
                espeak backend to drop tone markers / strip length & stress for
                non-tonal references; ignored by the ARPAbet backend (English
                only).

        Raises:
            ValueError: If the sample is not at ``MODEL_SAMPLE_RATE`` (16 kHz).
        """
        import torch

        if sample.sample_rate != MODEL_SAMPLE_RATE:
            raise ValueError(
                f"phoneme model requires {MODEL_SAMPLE_RATE} Hz input; "
                f"got {sample.sample_rate}. Run preprocess() first."
            )
        wav = sample.waveform
        if wav.ndim == 2:  # be lenient: downmix multi-channel on the fly
            wav = wav.mean(axis=1, dtype=np.float32)

        inputs = self._feature_extractor(wav, sampling_rate=MODEL_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(inputs.input_values.to(self.device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
        raw = self._ctc_collapse(predicted_ids)
        phonemes = self._normalize(raw, language)
        return PhonemeResult(phonemes=phonemes, raw_text=" ".join(phonemes))

    def _ctc_collapse(self, ids: Sequence[int]) -> tuple[str, ...]:
        """Collapse CTC: drop blank + special tokens, merge consecutive repeats."""
        tokens: list[str] = []
        prev: int | None = None
        for idx in ids:
            if idx == self._blank_id:
                prev = None
                continue
            if idx == prev:
                continue
            tok = self._id2token.get(idx)
            if tok is not None and not tok.startswith("<"):
                tokens.append(tok)
            prev = idx
        return tuple(tokens)


class EspeakPhonemeModel(PhonemeModel):
    """``facebook/wav2vec2-lv-60-espeak-cv-ft`` — multilingual, espeak-IPA native.

    The model already emits espeak IPA tokens, so normalization is light:
    tone-marker tokens (digit-bearing, e.g. Mandarin ``"ɑ5"``) are dropped and
    length/stress marks stripped for non-tonal references — but every other
    recognized token is kept. An earlier inventory-membership filter used to
    silently delete legitimate English vowels emitted in alternate notations
    (bare ``"e"``, barred ``"ᵻ"``, r-coloured ``"ɔːɹ"``); that destroyed real
    pronunciation information and is gone.
    """

    model_id = ESPEAK_MODEL_ID

    def _load(self) -> None:
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id).to(self.device).eval()

        vocab_path = hf_hub_download(self.model_id, "vocab.json")
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self._id2token: dict[int, str] = {int(idx): tok for tok, idx in vocab.items()}
        self._blank_id: int = int(vocab.get(BLANK_TOKEN, 0))

    def _normalize(self, raw: tuple[str, ...], language: str | None) -> tuple[str, ...]:
        return _strip_tones_marks(raw, language)


class ArpabetPhonemeModel(PhonemeModel):
    """``slplab/wav2vec2-large-robust-L2-english-phoneme-recognition`` — the default.

    Trained on L2 (non-native) English learner speech, so it recognizes the
    accented pronunciations a pronunciation coach must catch (it even tags
    mispronounced phonemes with an ``_err`` suffix). Emits ARPAbet-39 (plus the
    ``_err`` / ``*`` variants); every token maps 1:1 to espeak IPA via
    :data:`ARPABET_TO_IPA` — 100 % coverage, zero drops. The ``_err`` /
    ``*`` suffixes are stripped before lookup so the diff scores against the
    intended phoneme. The mapped IPA segments are then rejoined into espeak's
    combined units (``ɛ ɹ`` → ``ɛɹ``) so the hypothesis lands in the exact
    notation the kokoro G2P reference uses.
    """

    model_id = ARPABET_MODEL_ID

    def _load(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        self._feature_extractor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()

        vocab = self._feature_extractor.tokenizer.get_vocab()
        self._id2token: dict[int, str] = {int(idx): tok for tok, idx in vocab.items()}
        self._blank_id: int = int(vocab.get(BLANK_TOKEN, vocab.get("<pad>", 0)))

    def _normalize(self, raw: tuple[str, ...], language: str | None) -> tuple[str, ...]:
        segments = tuple(ARPABET_TO_IPA[_arpabet_base(tok)] for tok in raw)
        return _segments_to_espeak_units(segments)


# --------------------------------------------------------------------------- #
# Registry + process-wide cached singleton
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_KEY = "slplab-l2"
"""Default :data:`MODELS` key. Overridable via the ``PHONEME_MODEL`` env var."""

MODELS: dict[str, type[PhonemeModel]] = {
    "slplab-l2": ArpabetPhonemeModel,
    "espeak": EspeakPhonemeModel,
}
"""Registry of selectable phoneme backends, keyed by short name. Add a new
backend by subclassing :class:`PhonemeModel` and registering it here."""

_model_instance: PhonemeModel | None = None
_model_instance_key: str | None = None


def get_phoneme_model(
    key: str | None = None,
    device: str = "auto",
    *,
    reload: bool = False,
) -> PhonemeModel:
    """Return a process-wide cached :class:`PhonemeModel`.

    Args:
        key: Registry key (e.g. ``"slplab-l2"``, ``"espeak"``). Defaults to the
            ``PHONEME_MODEL`` env var, then :data:`DEFAULT_MODEL_KEY`.
        device: ``"auto"`` (default), ``"cpu"``, ``"mps"``, or ``"cuda"``.
        reload: Force re-instantiation even if a different model is cached
            (mainly for tests).
    """
    import os

    global _model_instance, _model_instance_key
    key = key or os.environ.get("PHONEME_MODEL", DEFAULT_MODEL_KEY)
    if _model_instance is None or reload or _model_instance_key != key:
        cls = MODELS.get(key)
        if cls is None:
            raise ValueError(f"unknown phoneme model {key!r}; available: {sorted(MODELS)}")
        _model_instance = cls(device=device)
        _model_instance_key = key
    return _model_instance


# --------------------------------------------------------------------------- #
# Backward-compat aliases (callers in deps/feedback/cli import these)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_ID = ESPEAK_MODEL_ID
"""Legacy alias for :data:`ESPEAK_MODEL_ID`. Kept so the G2P/reference side
(which is pinned to the espeak vocabulary) can keep importing a stable name."""


def get_extractor(
    model_id: str | None = None,
    device: str = "auto",
    *,
    reload: bool = False,
) -> PhonemeModel:
    """Deprecated alias for :func:`get_phoneme_model`.

    The ``model_id`` arg is accepted for call-site compatibility but ignored —
    model selection is now registry-key based via ``PHONEME_MODEL``.
    """
    return get_phoneme_model(device=device, reload=reload)


PhonemeExtractor = PhonemeModel
"""Type alias kept for the type hints in ``api/deps.py`` etc. Instantiate via
:func:`get_phoneme_model` or a concrete subclass, not this name directly."""
