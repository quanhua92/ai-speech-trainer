"""Phoneme-level pronunciation analysis via Wav2Vec2-CTC + difflib.

Extracts an espeak-IPA phoneme sequence from a canonical :class:`AudioSample`
using Facebook's ``wav2vec2-lv-60-espeak-cv-ft`` model, then aligns a reference
vs. hypothesis sequence with :mod:`difflib` to produce a structured diff and a
Phoneme Error Rate (PER).

The decoder is implemented directly on the model's ``vocab.json`` rather than
via ``Wav2Vec2PhonemeCTCTokenizer`` — that tokenizer eagerly inits an espeak
backend we don't need for the decode direction, so we avoid the hard runtime
dependency on espeak linkage for the extraction path.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ai_speech_shadowing.core.audio import TARGET_SAMPLE_RATE, AudioSample

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
"""HuggingFace model: wav2vec2-large-lv60k fine-tuned on Common Voice with an
espeak IPA phoneme vocabulary (392 tokens)."""

MODEL_SAMPLE_RATE: int = TARGET_SAMPLE_RATE  # 16000 — the model's only contract
BLANK_TOKEN = "<pad>"  # CTC blank token (vocab id 0)


# --------------------------------------------------------------------------- #
# Diff structures
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


class PhonemeExtractor:
    """Loads the Wav2Vec2-CTC phoneme model and decodes audio → phonemes.

    Heavy dependencies (``torch``, ``transformers``) are imported lazily so that
    merely importing this module (e.g. in the pure-diff unit tests) does not pay
    the load cost or require the model runtime.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> None:
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self.model_id = model_id
        self.device = _resolve_device(device)

        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device).eval()

        vocab_path = hf_hub_download(model_id, "vocab.json")
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self._id2token: dict[int, str] = {int(idx): tok for tok, idx in vocab.items()}
        self._blank_id: int = int(vocab.get(BLANK_TOKEN, 0))

    def extract(self, sample: AudioSample) -> PhonemeResult:
        """Decode a canonical AudioSample (16 kHz) into a phoneme sequence.

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
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            logits = self._model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)[0].tolist()
        phonemes = self._ctc_collapse(predicted_ids)
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


# --------------------------------------------------------------------------- #
# Lazy singleton — reuse one loaded model across calls (CLI, API).
# --------------------------------------------------------------------------- #
_extractor_instance: PhonemeExtractor | None = None


def get_extractor(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    *,
    reload: bool = False,
) -> PhonemeExtractor:
    """Return a process-wide cached :class:`PhonemeExtractor` (loads on first use)."""
    global _extractor_instance
    if _extractor_instance is None or reload:
        _extractor_instance = PhonemeExtractor(model_id=model_id, device=device)
    return _extractor_instance
