"""Tests for word-level diff projection."""

# IPA characters in the test fixtures are intentional.
# ruff: noqa: RUF001, RUF003

from __future__ import annotations

import pytest

from ai_speech_shadowing.core.phoneme import diff_phonemes
from ai_speech_shadowing.core.wordalign import norm_misaki, word_level_diff


class TestNormMisaki:
    """Pure string normalization (no HF / no misaki)."""

    def test_strips_stress_marks(self) -> None:
        assert norm_misaki("ˈʌmps") == "ʌmps"
        assert norm_misaki("ˌWn") == "aʊn"  # stress gone, W -> aʊ

    def test_strips_length_mark(self) -> None:
        assert norm_misaki("ɑː") == "ɑ"

    def test_unfolds_affricates(self) -> None:
        assert norm_misaki("ʤˈʌmpt") == "dʒʌmpt"
        assert norm_misaki("ʧ") == "tʃ"

    def test_uppercase_diphthongs(self) -> None:
        assert norm_misaki("lˈAzi") == "leɪzi"  # A -> eɪ
        assert norm_misaki("ˈOvəɹ") == "oʊvəɹ"  # O -> oʊ
        assert norm_misaki("nˈIt") == "naɪt"  # I -> aɪ

    def test_passthrough_plain(self) -> None:
        assert norm_misaki("ðə") == "ðə"


@pytest.mark.slow
class TestWordLevelDiff:
    """End-to-end projection (needs misaki + the espeak vocab from the HF cache)."""

    def test_attributes_substitution_to_the_right_word(self) -> None:
        # reference phonemes for "the jumps" but the user said /t/ instead of /s/
        # in "jumps" — exactly the jumps->jumped-style mistake.
        reference = ["ð", "ə", "dʒ", "ʌ", "m", "p", "s"]
        hypothesis = ["ð", "ə", "dʒ", "ʌ", "m", "p", "t"]
        ops = diff_phonemes(reference, hypothesis).operations

        words = word_level_diff("the jumps", reference, ops)
        by_word = {w.word.lower(): w for w in words}

        assert by_word["the"].status == "match"
        assert by_word["jumps"].status == "sub"
        jumps_errs = by_word["jumps"].errors
        assert len(jumps_errs) == 1
        assert jumps_errs[0]["expected"] == "s"
        assert jumps_errs[0]["actual"] == "t"

    def test_empty_when_no_text(self) -> None:
        ops = diff_phonemes(["a"], ["a"]).operations
        assert word_level_diff("", ["a"], ops) == []
