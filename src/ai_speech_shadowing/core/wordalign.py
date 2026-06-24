"""Best-effort word-level diff: project phoneme-level errors onto reference-text words.

The phoneme diff (Wav2Vec2 espeak) is exact but hard for a learner to read.
This module G2P-ps the *known* reference text with misaki (the same engine Kokoro
uses), normalizes its phoneme notation onto the espeak inventory, aligns it to
the reference phoneme sequence, and rolls the per-phoneme errors up to words.

Caveat: misaki's notation ≠ the Wav2Vec2 espeak inventory (stress marks,
``ʤ`` vs ``dʒ``, uppercase stressed diphthongs). Normalization covers the common
cases; alignment is best-effort, so an occasional boundary phoneme may be
attributed to the neighbouring word. The underlying phoneme diff stays exact.

The G2P / normalization helpers themselves live in :mod:`ai_speech_shadowing.core.g2p`
so they can be shared with the TTS reference generator (which captures Kokoro's
G2P output at synthesis time).
"""

# This module deliberately handles IPA characters that RUF001 flags as
# "ambiguous" — they are intentional, not typos.
# ruff: noqa: RUF001

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_speech_shadowing.core.g2p import g2p_words, norm_misaki  # noqa: F401  (re-export)
from ai_speech_shadowing.core.phoneme import PhonemeOp

if TYPE_CHECKING:
    from collections.abc import Sequence


def _norm_ctc(tok: str) -> str:
    return tok.replace("ː", "")


@dataclass(frozen=True, slots=True)
class WordDiff:
    """One word of the reference text with its rolled-up pronunciation status."""

    word: str
    status: str  # "match" | "sub" | "del" | "ins"
    errors: tuple[dict[str, str], ...]  # {type, expected?, actual?}


def _op_to_dict(op: PhonemeOp) -> dict[str, str]:
    if op.tag == "match":
        return {"type": "match", "phoneme": op.ref or ""}
    if op.tag == "sub":
        return {"type": "sub", "expected": op.ref or "", "actual": op.hyp or ""}
    if op.tag == "del":
        return {"type": "del", "expected": op.ref or ""}
    return {"type": "ins", "actual": op.hyp or ""}


def _split_sentences(text: str) -> list[str]:
    """Split reference text into sentences on terminal punctuation + whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _align_sentence(
    sent_words: list[tuple[str, list[str]]],
    ref_norm: list[str],
    ctc_ptr: int,
) -> tuple[dict[int, str], int]:
    """Align one sentence's word phonemes to ``ref_norm`` starting at ``ctc_ptr``.

    Re-anchoring per sentence keeps alignment drift from compounding across a
    long passage. Returns ``(ctc_idx -> word, next_ctc_ptr)``.
    """
    sent_seq: list[str] = []
    offsets: list[tuple[int, int]] = []
    for _w, toks in sent_words:
        start = len(sent_seq)
        sent_seq.extend(toks)
        offsets.append((start, len(sent_seq)))

    slack = len(sent_seq) // 2 + 8
    window_end = min(len(ref_norm), ctc_ptr + len(sent_seq) + slack)
    window = ref_norm[ctc_ptr:window_end]

    text2win: dict[int, int] = {}
    for tag, i1, i2, j1, _j2 in difflib.SequenceMatcher(
        None, sent_seq, window, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                text2win[i1 + k] = j1 + k

    ctc2word: dict[int, str] = {}
    matched: list[int] = []
    for (word, _w), (s, e) in zip(sent_words, offsets, strict=True):
        for i in range(s, e):
            if i in text2win:
                abs_idx = ctc_ptr + text2win[i]
                ctc2word[abs_idx] = word
                matched.append(abs_idx)

    next_ptr = (max(matched) + 1) if matched else min(ctc_ptr + len(sent_seq), len(ref_norm))
    return ctc2word, next_ptr


def word_level_diff(
    text: str,
    reference_phonemes: Sequence[str],
    phoneme_ops: Sequence[PhonemeOp],
) -> list[WordDiff]:
    """Project phoneme-level ops onto the words of ``text`` (best-effort).

    Aligns per-sentence (re-anchoring each sentence) so accuracy holds up on
    long passages. Returns one ``WordDiff`` per content word in text order;
    empty list if the text or phoneme sequence is empty.
    """
    sentences = _split_sentences(text)
    if not sentences or not reference_phonemes:
        return []

    ref_norm = [_norm_ctc(t) for t in reference_phonemes]
    ctc2word: dict[int, str] = {}
    words_order: list[str] = []
    ctc_ptr = 0
    for sent in sentences:
        sent_words = g2p_words(sent)
        if not sent_words:
            continue
        words_order.extend(w for w, _ in sent_words)
        sent_map, ctc_ptr = _align_sentence(sent_words, ref_norm, ctc_ptr)
        ctc2word.update(sent_map)

    if not words_order:
        return []

    # Walk the phoneme ops (tracked by reference index) and attribute errors.
    word_errors: dict[str, list[dict[str, str]]] = {w: [] for w in words_order}
    ref_idx = -1
    for op in phoneme_ops:
        if op.tag in ("match", "sub", "del"):
            ref_idx += 1
            if op.tag != "match":
                word = ctc2word.get(ref_idx)
                if word is not None:
                    word_errors[word].append(_op_to_dict(op))
        else:  # ins — attribute to the preceding word
            word = ctc2word.get(ref_idx) if ref_idx >= 0 else None
            if word is not None:
                word_errors[word].append(_op_to_dict(op))

    # Dedup word_errors keys while preserving text order.
    result: list[WordDiff] = []
    for word in dict.fromkeys(words_order):
        errs = word_errors[word]
        if not errs:
            status = "match"
        else:
            kinds = {e["type"] for e in errs}
            status = "sub" if "sub" in kinds else ("del" if "del" in kinds else "ins")
        result.append(WordDiff(word=word, status=status, errors=tuple(errs)))
    return result
